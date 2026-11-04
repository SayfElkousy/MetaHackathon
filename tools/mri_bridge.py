"""MRI reader -> ACL modeler bridge. Transfer only: it changes neither side.

    upload page (/upload)
      -> KneeScan app (mri_analysis/app.py), through its own Gradio API "/analyze"
      <- KneeScan results JSON (tear status and level, ACL location, ACL length and width, femur-fibula distance)
      -> to_acl_mri_input(): the modeler's documented input format, acl-mri-input 1.0 (docs/PATIENT_INPUT.md)
      -> acl_engine.v2.patient.import_document(): the modeler's own importer -> design state
      -> apply_readings(): readings the acl-mri-input format has no field for (target length; recorded-only values)
      -> state_store -> designer at /?state=<id>, which builds the model as it always does.

The reader runs as its own process (locally: `python mri_analysis/app.py`, http://127.0.0.1:7860; or the Colab
notebook's gradio.live link). Only the Python standard library is used here, like the rest of the designer server.

How each reading reaches the modeler:
  tear status + level   -> injury.type (intact / partial / complete)
  ACL width (mm)        -> lig.width_profile: the template profile scaled so its mid-substance median equals the reading
  ACL length (mm)       -> lig.length_mode = target, lig.length_target_mm (the modeler solves the anterior bow)
  ACL location          -> recorded only (the modeler places the ligament from footprints in its own knee frame)
  femur-fibula distance -> recorded only (the modeler has no fibula)
"""
import copy
import datetime
import json
import math
import os
import re
import shutil
import tempfile
import threading
import time
import traceback
import uuid
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_READER_URL = os.environ.get("KNEESCAN_URL", "http://127.0.0.1:7860")
API_NAME = "analyze"                       # mri_analysis/app.py: go.click(analyze, ...)
SCAN_EXTS = (".zip", ".dcm", ".nii", ".nii.gz", ".npy", ".pck")
MAX_SCAN_BYTES = 1024 * 1024 * 1024
READER_TIMEOUT_S = float(os.environ.get("KNEESCAN_TIMEOUT", "900"))
SOURCE = "KneeScan"
PARTIAL_INVOLVEMENT = "AM"                 # partial tears: KneeScan gives no bundle; the modeler requires one


class ReaderError(RuntimeError):
    """The MRI reader could not be reached, or it rejected the scan."""


# --------------------------------------------------------------------------- reader client (Gradio 5 HTTP API)
def normalize_url(url):
    url = (url or DEFAULT_READER_URL).strip().rstrip("/")
    if not re.match(r"^https?://[^\s/]+", url):
        raise ReaderError("reader address must start with http:// or https://")
    return url


def _http(req, timeout):
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:400]
        raise ReaderError("MRI reader answered HTTP %d: %s" % (e.code, body or e.reason))
    except (urllib.error.URLError, OSError) as e:
        raise ReaderError("cannot reach the MRI reader: %s" % getattr(e, "reason", e))


def reader_info(url=None, timeout=6):
    """Is a KneeScan app listening at `url`, and does it expose the analyze endpoint?"""
    try:
        url = normalize_url(url)
        with _http(urllib.request.Request(url + "/gradio_api/info"), timeout) as r:
            info = json.loads(r.read().decode("utf-8"))
    except (ReaderError, ValueError) as e:
        return {"ok": False, "url": url, "error": str(e)}
    eps = sorted((info.get("named_endpoints") or {}).keys())
    if "/" + API_NAME not in eps:
        return {"ok": False, "url": url, "endpoints": eps,
                "error": "a Gradio app is running there, but it has no /%s endpoint (is it KneeScan's mri_analysis/app.py?)" % API_NAME}
    return {"ok": True, "url": url, "endpoints": eps}


def _upload(url, path, filename):
    boundary = "----aclbridge" + uuid.uuid4().hex
    with open(path, "rb") as f:
        data = f.read()
    head = ('--%s\r\nContent-Disposition: form-data; name="files"; filename="%s"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n" % (boundary, filename.replace('"', "_"))).encode("utf-8")
    body = head + data + ("\r\n--%s--\r\n" % boundary).encode("ascii")
    req = urllib.request.Request(url + "/gradio_api/upload", data=body, method="POST",
                                 headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
    with _http(req, READER_TIMEOUT_S) as r:
        paths = json.loads(r.read().decode("utf-8"))
    if not isinstance(paths, list) or not paths:
        raise ReaderError("MRI reader did not accept the upload")
    return paths[0]


def _fn_index(url):
    """Index of the analyze event in the app's config (the queue API addresses events by index)."""
    with _http(urllib.request.Request(url + "/config"), 30) as r:
        cfg = json.loads(r.read().decode("utf-8"))
    for k, d in enumerate(cfg.get("dependencies") or []):
        if d.get("api_name") == API_NAME:
            tg = d.get("targets") or [[None]]
            return d.get("id", k), (tg[0][0] if tg and tg[0] else None)
    raise ReaderError("the app at %s has no %s event (is it KneeScan's mri_analysis/app.py?)" % (url, API_NAME))


def _queue_result(resp, event_id, on_progress):
    """Read the Gradio queue stream for one event until it completes. Unlike the plain /call route, the
    queue stream carries the reader's own error text and its progress messages."""
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        try:
            m = json.loads(line[5:].strip())
        except ValueError:
            continue
        if m.get("event_id") not in (None, event_id):
            continue
        kind = m.get("msg")
        if kind == "progress":
            for p in m.get("progress_data") or []:
                if p.get("desc"):
                    on_progress(p["desc"])
        elif kind == "process_completed":
            out = m.get("output") or {}
            if not m.get("success", True) or "error" in out:
                raise ReaderError("the MRI reader rejected the scan: %s" % (out.get("error") or "no message (see the reader's console)"))
            return out.get("data")
        elif kind == "close_stream":
            break
    raise ReaderError("the MRI reader closed the connection before finishing")


def _download(url, fd):
    if not isinstance(fd, dict) or not (fd.get("url") or fd.get("path")):
        raise ReaderError("the MRI reader returned no results file")
    link = fd.get("url") or "%s/gradio_api/file=%s" % (url, urllib.parse.quote(fd["path"]))
    if link.startswith("/"):
        link = url + link
    with _http(urllib.request.Request(link), READER_TIMEOUT_S) as r:
        return json.loads(r.read().decode("utf-8"))


def analyze_scan(path, filename, do_bones=True, url=None, on_step=None):
    """Send one scan to the KneeScan app and return its results JSON (the same file its web page offers)."""
    url = normalize_url(url)
    step = on_step or (lambda *_: None)
    fn, trigger = _fn_index(url)
    step("upload", "sending the scan to the MRI reader")
    remote = _upload(url, path, filename)
    fd = {"path": remote, "orig_name": filename, "meta": {"_type": "gradio.FileData"}}
    session = uuid.uuid4().hex[:11]
    body = {"data": [fd, bool(do_bones)], "fn_index": fn, "trigger_id": trigger, "session_hash": session, "event_data": None}
    req = urllib.request.Request(url + "/gradio_api/queue/join", method="POST", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with _http(req, 60) as r:
        event_id = json.loads(r.read().decode("utf-8")).get("event_id")
    if not event_id:
        raise ReaderError("the MRI reader did not start the analysis")
    base = "the MRI reader is analysing the scan" + (" (femur-fibula adds about a minute)" if do_bones else "")
    step("analyze", base)
    with _http(urllib.request.Request("%s/gradio_api/queue/data?session_hash=%s" % (url, session)), READER_TIMEOUT_S) as r:
        out = _queue_result(r, event_id, lambda desc: step("analyze", "MRI reader: " + desc))
    if not isinstance(out, list) or len(out) < 4:
        raise ReaderError("unexpected answer from the MRI reader (is it KneeScan's mri_analysis/app.py?)")
    step("results", "fetching the readings")
    res = _download(url, out[3])
    if not isinstance(res, dict) or "tear" not in res:
        raise ReaderError("the MRI reader's results file has no tear reading")
    return res


# --------------------------------------------------------------------------- readings
def _num(v):
    return None if v is None else float(v)


def readings(r):
    """The five readings the modeler is fed, picked out of a KneeScan results JSON."""
    t, a, loc, b = r.get("tear") or {}, r.get("acl") or {}, r.get("acl_location") or {}, r.get("femur_fibula")
    vol = r.get("volume") or {}
    out = {
        "tear_status": {"value": t.get("status"), "probability": t.get("probability"), "threshold": t.get("threshold")},
        "tear_level": {"value": t.get("level"), "threshold": t.get("level_threshold"), "reliability": t.get("level_reliability")},
        "acl_length": {"value_mm": _num(a.get("length_mm")), "value_px": a.get("length_px"), "mm_source": a.get("mm_source"),
                       "passed_checks": a.get("passed_checks"), "confidence": a.get("confidence"), "error": a.get("error")},
        "acl_width": {"value_mm": _num(a.get("width_mm")), "value_px": a.get("width_px"), "mm_source": a.get("mm_source"),
                      "passed_checks": a.get("passed_checks"), "confidence": a.get("confidence"), "error": a.get("error"),
                      "definition": "median thickness over the middle half of the ligament, sagittal slice"},
        "acl_location": {"slice": loc.get("slice"), "slices": vol.get("slices"), "box_px": loc.get("box"), "score": loc.get("score"),
                         "outline_slice": a.get("slice"), "femoral_or_tibial_end_px": [a.get("end0"), a.get("end1")] if a.get("end0") else None,
                         "frame": "image pixels (x = column, y = row) of the uploaded scan; slice index from 0"},
        "femur_fibula": None,
    }
    if b is not None:
        out["femur_fibula"] = {"value_mm": _num(b.get("distance_mm")), "value_px": b.get("distance_px"), "mm_source": b.get("mm_source"),
                               "slice": b.get("slice"), "passed_checks": b.get("passed_checks"), "experimental": b.get("experimental", True),
                               "error": b.get("error")}
    return out


def injury_type(r):
    t = r.get("tear") or {}
    if t.get("status") == "intact":
        return "intact"
    if t.get("status") == "torn":
        return "complete" if str(t.get("level", "")).startswith("complete") else "partial"
    return None


def _size_status(r):
    """KneeScan's own warnings become the modeler's 'awaiting_review' status."""
    a = r.get("acl") or {}
    return "measured" if a.get("passed_checks") and (r.get("tear") or {}).get("status") != "torn" else "awaiting_review"


def _mid_median(profile):
    """Median of a 5-knot profile over s = 0.25..0.75 (linear between knots): KneeScan's width definition."""
    from acl_engine.v2.design import S_KNOTS
    import numpy as np
    return float(np.median(np.interp(np.linspace(0.25, 0.75, 101), S_KNOTS, profile)))


def width_profile_for(width_mm):
    """The modeler's template width profile, scaled so its mid-substance median equals the measured width.
    Returns (values or None, reason)."""
    from acl_engine.v2 import design as D
    spec = D.BY_ID["lig.width_profile"]
    base = [float(x) for x in spec["default"]]
    k = float(width_mm) / _mid_median(base)
    vals = [round(v * k, 2) for v in base]
    lo, hi = spec["min"], spec["max"]
    if min(vals) < lo or max(vals) > hi:
        return None, "the scaled profile %s mm leaves the modeler's limits (%g-%g mm)" % (vals, lo, hi)
    return vals, "template profile scaled x%.3f" % k


def _safe_id(name):
    stem = re.sub(r"(\.nii)?\.[A-Za-z0-9]+$", "", os.path.basename(name or "scan"))
    return re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")[:40] or "scan"


def to_acl_mri_input(r, laterality, case_id, run_id=None):
    """KneeScan results -> acl-mri-input 1.0 document (the format acl_engine.v2.patient reads).
    Returns (document, notes) where notes explain anything left out."""
    from acl_engine.v2.design import S_KNOTS
    notes = []
    val = r.get("validation") or {}
    spacing = (r.get("volume") or {}).get("spacing_mm")
    inp = r.get("input") or {}
    doc = {"schema": "acl-mri-input", "schema_version": "1.0.0", "case_id": case_id, "synthetic": False,
           "source": {"model_name": SOURCE, "model_version": "mri_analysis", "run_id": run_id,
                      "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")},
           "imaging": {"laterality": laterality, "units": "mm"},
           "notes": "Readings from KneeScan (mri_analysis) for %s; knee side given at upload." % (inp.get("file") or "an uploaded scan")}
    if spacing:
        doc["imaging"]["voxel_spacing_mm"] = spacing
    t = r.get("tear") or {}
    it = injury_type(r)
    if it:
        lv = val.get("level") or {}
        doc["injury"] = {"type": {"value": it, "status": "inferred", "source": "KneeScan tear ensemble",
                                  "note": "KneeScan: %s, tear probability %s (cut-off %s); level '%s' (reliability %s%s)" % (
                                      t.get("status"), t.get("probability"), t.get("threshold"), t.get("level"),
                                      t.get("level_reliability", "?"),
                                      ", partial vs complete AUC %s" % lv["partial_vs_complete_auc"] if lv.get("partial_vs_complete_auc") else "")}}
        if it == "partial":
            # The modeler needs to know which part is torn (AM, PL, sector or per_bundle); KneeScan does not say.
            doc["injury"]["involvement"] = {"value": PARTIAL_INVOLVEMENT, "status": "awaiting_review",
                                            "note": "KneeScan reports a partial tear without the bundle; %s assumed "
                                                    "(as in the modeler's partial-tear demo). Review in the designer." % PARTIAL_INVOLVEMENT}
            notes.append("partial tear: KneeScan does not say which bundle, so the %s bundle is assumed (awaiting review)" % PARTIAL_INVOLVEMENT)
    else:
        notes.append("no tear status in the reader's results; the modeler keeps its default injury")
    a = r.get("acl") or {}
    if a.get("width_mm") is not None:
        vals, why = width_profile_for(a["width_mm"])
        if vals:
            err = (val.get("acl_size") or {}).get("width_error_median_pct")
            doc["ligament"] = {"width_profile": {
                "value": {"s": list(S_KNOTS), "values": vals}, "units": "mm", "status": _size_status(r),
                "confidence": a.get("confidence"), "uncertainty": {"median_error_pct": err} if err is not None else None,
                "source": "KneeScan MedSAM ACL outline",
                "note": "KneeScan mid-substance width %s mm (slice %s, %s); %s" % (a["width_mm"], (a.get("slice") or 0) + 1,
                                                                                   a.get("mm_source", "?"), why)}}
        else:
            notes.append("ACL width %s mm not applied: %s" % (a["width_mm"], why))
    elif "width_px" in a:
        notes.append("ACL width has no millimetre value (no pixel spacing, and the femur-fibula step that estimates it was off)")
    else:
        notes.append("ACL width not measured: %s" % a.get("error", "no outline"))
    return doc, notes


def apply_readings(state, r, run_id=None):
    """After the modeler's importer: set what acl-mri-input cannot carry, and record the rest.
    Changes `state` in place (params, provenance, imported copy, assets, import report). Returns the mapping rows."""
    from acl_engine.v2 import design as D
    rd = readings(r)
    rep = state["imported"]["meta"]["report"]
    rows = []
    t = rd["tear_status"]
    it = injury_type(r)
    rows.append({"reading": "ACL tear status", "value": t["value"], "param": "injury.type",
                 "result": "applied" if it else "not applied", "sent": it})
    rows.append({"reading": "ACL tear level", "value": rd["tear_level"]["value"],
                 "param": "injury.type" + (" + injury.involvement" if it == "partial" else ""),
                 "result": "applied" if it else "not applied", "sent": it + " (%s)" % PARTIAL_INVOLVEMENT if it == "partial" else it,
                 "note": ("low reliability (partial vs complete)" if t["value"] == "torn" else None) if it != "partial" else
                 "low reliability; KneeScan gives no bundle, %s assumed (awaiting review)" % PARTIAL_INVOLVEMENT})
    wp = state["params"].get("lig.width_profile")
    wp = wp.get("v") if isinstance(wp, dict) else wp          # imported profiles keep their samples: {s, v}
    w_applied = (state["provenance"].get("lig.width_profile") or {}).get("source", "").startswith(SOURCE)
    rows.append({"reading": "ACL width", "value": rd["acl_width"]["value_mm"], "units": "mm", "param": "lig.width_profile",
                 "result": "applied" if w_applied else "not applied", "sent": wp if w_applied else None,
                 "note": "template profile scaled to the measured mid-substance width" if w_applied else None})
    # ---- length: acl-mri-input records ligament.length_mm only for comparison, so set the modeler's target-length mode
    L = rd["acl_length"]["value_mm"]
    spec = D.BY_ID["lig.length_target_mm"]
    if L is not None and spec["min"] <= L <= spec["max"]:
        a = r.get("acl") or {}
        st = _size_status(r)
        prov = {"status": st, "source": "%s mri_analysis" % SOURCE, "run_id": run_id, "path": "kneescan.acl.length_mm",
                "units": "mm", "confidence": a.get("confidence"),
                "uncertainty": {"median_error_pct": ((r.get("validation") or {}).get("acl_size") or {}).get("length_error_median_pct")},
                "region": None, "note": "KneeScan ACL length %s mm (slice %s, %s), applied as the target length" % (
                    L, (a.get("slice") or 0) + 1, a.get("mm_source", "?"))}
        mode_prov = dict(prov, path="kneescan.acl.length_mm (mode)", units="", note="target mode so the measured length is used")
        for pid, v, pv in (("lig.length_mode", "target", mode_prov), ("lig.length_target_mm", round(L, 2), prov)):
            for tgt in (state, state["imported"]):
                tgt["params"][pid] = v
                tgt["provenance"][pid] = copy.deepcopy(pv)
            rep["mapped"].append({"path": pv["path"], "param": pid, "status": st})
        state["assets"].setdefault("reported", {})["ligament.length_mm"] = L
        rows.append({"reading": "ACL length", "value": L, "units": "mm", "param": "lig.length_target_mm (lig.length_mode = target)",
                     "result": "applied", "sent": round(L, 2)})
    else:
        why = ("no millimetre value" if L is None else "%.1f mm is outside the modeler's %g-%g mm range" % (L, spec["min"], spec["max"]))
        rep["warnings"].append("KneeScan ACL length not applied: %s" % why)
        rows.append({"reading": "ACL length", "value": L, "units": "mm", "param": "lig.length_target_mm", "result": "not applied", "note": why})
    # ---- recorded only
    loc = rd["acl_location"]
    if loc["slice"] is not None:
        rep["warnings"].append("KneeScan ACL location: slice %d of %s, box x %s-%s, y %s-%s px. Recorded only: the modeler places the ligament "
                               "from its footprints in its own knee frame and has no image-location input." % (
                                   loc["slice"] + 1, loc["slices"], loc["box_px"][0], loc["box_px"][2], loc["box_px"][1], loc["box_px"][3]))
    rows.append({"reading": "ACL location", "value": loc, "param": None, "result": "recorded",
                 "note": "the modeler has no image-location input; stored with the design"})
    ff = rd["femur_fibula"]
    if ff and ff.get("value_mm") is not None:
        rep["warnings"].append("KneeScan femur to fibular head distance %s mm (experimental). Recorded only: the modeler has no fibula." % ff["value_mm"])
    rows.append({"reading": "Femur to fibular head", "value": (ff or {}).get("value_mm"), "units": "mm", "param": None,
                 "result": "recorded" if ff and ff.get("value_mm") is not None else "not measured",
                 "note": "the modeler has no fibula; stored with the design" if ff and ff.get("value_mm") is not None
                 else ((ff or {}).get("error") or "femur-fibula step was off")})
    for w in r.get("warnings") or []:
        rep["warnings"].append("KneeScan: " + w)
    rec = {"source": SOURCE, "run_id": run_id, "file": (r.get("input") or {}).get("file"), "readings": rd}
    state["assets"]["mri_readings"] = rec
    state["imported"]["assets"]["mri_readings"] = copy.deepcopy(rec)
    return rows


def length_check(state):
    """What the modeler will say about the target length (cheap: builds only the ligament centreline)."""
    if state["params"].get("lig.length_mode") != "target":
        return None
    try:
        from acl_engine.v2.design import coerce_types, migrate_params
        from acl_engine.v2.bones import ProceduralBone
        from acl_engine.v2.ligament import Ligament
        params = dict(state["params"])
        migrate_params(params)
        p = coerce_types(params)
        lig = Ligament(p, ProceduralBone("femur", p, float(p["knee.flexion_deg"])), ProceduralBone("tibia", p, 0.0), "preview")
        li = lig.length_info
        msgs = [i["message"] for i in lig.issues if i["param"] == "lig.length_target_mm"]
        return {"target_mm": float(p["lig.length_target_mm"]), "achievable_mm": [round(float(x), 1) for x in li.get("achievable_mm", [])],
                "reached_mm": round(float(li["length_mm"]), 1), "ok": not msgs, "message": msgs[0] if msgs else None}
    except Exception as e:                       # a preview only; the designer reports the real checks
        return {"ok": None, "message": "length preview unavailable: %s" % e}


def build_state(r, laterality, filename, run_id=None):
    """KneeScan results -> (design state, mapping rows, acl-mri-input document, notes)."""
    from acl_engine.v2.patient import import_document
    from acl_engine.v2 import design as D
    case_id = "mri_" + _safe_id(filename)
    doc, notes = to_acl_mri_input(r, laterality, case_id, run_id)
    state, report = import_document(doc, tempfile.gettempdir())
    if state is None:
        raise ValueError("the modeler rejected the readings: " + "; ".join(report["errors"]))
    rows = apply_readings(state, r, run_id)
    for n in notes:
        state["imported"]["meta"]["report"]["warnings"].append(n)
    errs, _ = D.validate_params(dict(state["params"]))
    if errs:
        raise ValueError("the readings give invalid modeler parameters: " + "; ".join("%s: %s" % (e["param"], e["message"]) for e in errs))
    state["case"]["label"] = "MRI upload: %s (%s, %s knee)" % (os.path.basename(filename), SOURCE, laterality)
    state["case"]["notes"] = doc["notes"]
    state["history"].append({"action": "mri_upload", "case": case_id, "source": SOURCE, "run_id": run_id,
                             "t": datetime.datetime.now().isoformat(timespec="seconds")})
    return state, rows, doc, notes


# --------------------------------------------------------------------------- upload jobs
JOBS = {}
JOBS_LOCK = threading.Lock()
STEPS = [("upload", "Send the scan to the MRI reader"), ("analyze", "MRI reader analyses the scan"),
         ("results", "Fetch the readings"), ("convert", "Convert readings to modeler parameters"),
         ("save", "Save the design for the modeler")]


def _public(job):
    return {k: job[k] for k in ("id", "status", "steps", "message", "error", "filename", "side", "do_bones", "reader_url",
                                "started", "finished", "result") if k in job}


def start_job(scan_path, filename, side, do_bones, reader_url):
    """Run one upload in the background; `scan_path` (a temp file) is deleted when the reader is done with it."""
    jid = uuid.uuid4().hex[:12]
    job = {"id": jid, "status": "running", "steps": [{"id": s, "label": l, "state": "pending"} for s, l in STEPS],
           "message": "starting", "error": None, "filename": filename, "side": side, "do_bones": bool(do_bones),
           "reader_url": reader_url, "started": time.time(), "finished": None, "result": None}
    with JOBS_LOCK:
        JOBS[jid] = job
        for old in sorted(JOBS, key=lambda k: JOBS[k]["started"])[:-20]:
            JOBS.pop(old, None)
    threading.Thread(target=_run_job, args=(job, scan_path), daemon=True).start()
    return jid


def _set_step(job, sid, message):
    seen = False
    for s in job["steps"]:
        if s["id"] == sid:
            s["state"], seen = "running", True
        elif not seen:
            s["state"] = "done"
    job["message"] = message


def _run_job(job, scan_path):
    try:
        try:
            r = analyze_scan(scan_path, job["filename"], job["do_bones"], job["reader_url"], lambda s, m: _set_step(job, s, m))
        finally:
            shutil.rmtree(os.path.dirname(scan_path), ignore_errors=True)
        _set_step(job, "convert", "converting the readings into modeler parameters")
        state, rows, doc, notes = build_state(r, job["side"], job["filename"], job["id"])
        lc = length_check(state)
        _set_step(job, "save", "saving the design")
        import server_v2 as V2
        sid = V2.save_state(state)
        for s in job["steps"]:
            s["state"] = "done"
        job["result"] = {"state_id": sid, "readings": readings(r), "mapping": rows, "length_check": lc, "notes": notes,
                         "warnings": r.get("warnings") or [], "seconds": r.get("seconds"), "input": r.get("input"),
                         "document": doc, "kneescan": r}
        job["status"], job["message"] = "done", "done"
    except (ReaderError, ValueError) as e:
        job["status"], job["error"] = "error", str(e)
    except Exception as e:
        traceback.print_exc()
        job["status"], job["error"] = "error", "bridge failed: %s" % e
    finally:
        job["finished"] = time.time()
        for s in job["steps"]:
            if s["state"] == "running" and job["status"] == "error":
                s["state"] = "failed"


def job_status(jid):
    with JOBS_LOCK:
        job = JOBS.get(jid)
        return copy.deepcopy(_public(job)) if job else None


def receive_scan(rfile, length, filename):
    """Stream an uploaded scan body to a private temp folder; returns its path."""
    name = os.path.basename(filename or "")
    if not name.lower().endswith(SCAN_EXTS):
        raise ValueError("unsupported file type; use %s" % ", ".join(SCAN_EXTS))
    if length <= 0:
        raise ValueError("empty upload")
    if length > MAX_SCAN_BYTES:
        raise ValueError("scan over %d MB" % (MAX_SCAN_BYTES // 2 ** 20))
    d = tempfile.mkdtemp(prefix="mri_upload_")
    path = os.path.join(d, re.sub(r"[^A-Za-z0-9_.-]+", "_", name))
    left = length
    with open(path, "wb") as f:
        while left > 0:
            chunk = rfile.read(min(1 << 20, left))
            if not chunk:
                break
            f.write(chunk)
            left -= len(chunk)
    if left:
        shutil.rmtree(d, ignore_errors=True)
        raise ValueError("upload interrupted")
    return path
