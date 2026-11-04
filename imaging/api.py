"""HTTP handlers for /api/img/* (wired into tools/server.py). Handlers return (status, payload) where payload is a
JSON-able dict, or ("raw", bytes, headers) for binary voxel data."""
import json
import os
import shutil
import tempfile
import zipfile

import numpy as np

from . import evidence as E
from . import model_map as MM
from . import qc as QC
from . import store as ST
from . import volumes as V

ROOT = ST.ROOT
FIXTURES = os.path.join(ROOT, "fixtures", "imaging")


def _case(body):
    return ST.load(body.get("case") or "")


# --------------------------------------------------------------------------- GET
def get(path, q):
    one = lambda k, d="": (q.get(k, [d]) or [d])[0]
    if path == "/api/img/cases":
        return 200, {"cases": ST.list_cases()}
    if path == "/api/img/case":
        return 200, public_case(ST.load(one("case")))
    if path == "/api/img/job":
        return 200, ST.job_status(one("id"))
    if path == "/api/img/volume":
        p, v = ST.volume_bytes_path(one("case"), one("id"))
        with open(p, "rb") as f:
            b = f.read()
        return "raw", b, {"Content-Type": "application/octet-stream", "X-Shape": json.dumps(v["shape"]), "X-Dtype": v["dtype"],
                          "ETag": '"%s"' % v["sha1"], "Cache-Control": "private, max-age=86400"}
    if path == "/api/img/mask":
        m = ST.load(one("case"))
        k = m["masks"][one("id")]
        with open(os.path.join(ST.case_dir(m["id"]), k["file"]), "rb") as f:
            b = f.read()
        return "raw", b, {"Content-Type": "application/octet-stream", "ETag": '"%s"' % k["sha1"],
                          "Cache-Control": "private, max-age=86400"}
    if path == "/api/img/reader":
        import mri_bridge as MB
        return 200, dict(MB.reader_info(one("url") or MB.DEFAULT_READER_URL), default_url=MB.DEFAULT_READER_URL)
    if path == "/api/img/vocab":
        return 200, {"measurements": E.MEASUREMENT_KINDS, "landmarks": E.LANDMARK_KINDS, "label_colors": ST.LABEL_COLORS,
                     "mappings": [{k: m[k] for k in ("id", "needs", "transform", "relation", "assumptions", "review_required")}
                                  | {"targets": list(m["targets"])} for m in E.MAPPINGS], "not_mapped": E.NOT_MAPPED}
    return 404, {"error": "not found"}


def public_case(m):
    return {k: m.get(k) for k in ("id", "label", "synthetic", "created", "volumes", "masks", "annotations", "findings", "report_text")}


# --------------------------------------------------------------------------- uploads (raw body)
def upload(q, rfile, length):
    one = lambda k, d="": (q.get(k, [d]) or [d])[0]
    if length > ST.MAX_UPLOAD:
        raise ValueError("file over 1 GB")
    d = ST.upload_dir(one("case"), one("batch"))
    name = ST.safe_name(one("name"))
    dst = os.path.join(d, name)
    left = length
    with open(dst, "wb") as f:
        while left > 0:
            chunk = rfile.read(min(1 << 20, left))
            if not chunk:
                break
            f.write(chunk)
            left -= len(chunk)
    if left:
        raise ValueError("upload incomplete")
    return 200, {"name": name, "bytes": length}


def save_mask(q, rfile, length, headers):
    one = lambda k, d="": (q.get(k, [d]) or [d])[0]
    m = ST.load(one("case"))
    vid = one("volume")
    v = m["volumes"][vid]
    n = int(np.prod(v["shape"]))
    if length != n:
        raise ValueError("mask has %d bytes; the volume grid needs %d" % (length, n))
    raw = rfile.read(length)
    arr = np.frombuffer(raw, np.uint8).reshape(v["shape"][2], v["shape"][1], v["shape"][0]).transpose(2, 1, 0)
    labels = json.loads(headers.get("X-Labels") or "[]")
    parent = one("parent") or None
    origin = one("origin") or ("edited" if parent else "manual")
    if origin not in ("edited", "manual"):
        raise ValueError("origin must be edited | manual")
    ent = ST.add_mask(m["id"], vid, arr, labels, origin, parent=parent, note=headers.get("X-Note"),
                      checks=[V.chk("grid", "alignment", "pass", "Drawn on the volume grid in the viewer.")])
    return 200, {"mask": ent}


# --------------------------------------------------------------------------- POST (JSON)
def post(path, body):
    if path == "/api/img/cases":
        return 200, public_case(ST.create_case(body.get("label") or "Untitled case"))
    if path == "/api/img/demo":
        return 200, demo_case()
    if path == "/api/img/import":
        return 200, start_import(body)
    if path == "/api/img/job/cancel":
        return 200, ST.cancel_job(body.get("id"))
    if path == "/api/img/qc":
        m = _case(body)
        v = m["volumes"][body["volume"]]
        if body.get("mask"):
            arr, mk = ST.mask_array(m["id"], body["mask"])
            return 200, QC.run(v, mk, arr)
        return 200, QC.run(v)
    if path == "/api/img/annotation":
        return 200, {"annotation": ST.upsert_annotation(body["case"], body.get("item") or {})}
    if path == "/api/img/review":
        return 200, {"mask": ST.set_review(body["case"], body["mask"], body.get("status"), note=body.get("note"))}
    if path == "/api/img/label":
        return 200, {"mask": ST.rename_label(body["case"], body["mask"], body["value"], body["name"])}
    if path == "/api/img/remove":
        return 200, ST.remove(body["case"], body.get("kind"), body.get("id"))
    if path == "/api/img/report":
        m = _case(body)
        text = str(body.get("text") or "")[:20000]
        found = E.extract_report(text)
        with ST.LOCK:
            m = ST.load(m["id"])
            m["report_text"], m["findings"] = text, found
            ST.log(m, "report", findings=len(found))
            ST.save(m["id"], m)
        return 200, {"findings": found, "note": "Report findings are stored as evidence. Nothing in the design changed."}
    if path == "/api/img/propose":
        m = _case(body)
        return 200, E.propose(m, body.get("params") or {}, body.get("provenance") or {}, scope=body.get("scope"),
                              include_assumptions=bool(body.get("include_assumptions")))
    if path == "/api/img/segment":
        return 200, start_segmentation(body)
    if path == "/api/img/segment_result":
        m = _case(body)
        v = m["volumes"][body["volume"]]
        ent = _store_model_result(m["id"], v, body.get("result") or {}, {"source": "KneeScan results file supplied by the user",
                                                                          "file": body.get("filename")})
        return 200, ent
    return 404, {"error": "not found"}


# --------------------------------------------------------------------------- import
def start_import(body):
    cid = body.get("case")
    m = ST.load(cid)
    role = body.get("role", "image")
    if role not in ("image", "mask", "derived"):
        raise ValueError("role must be image | mask | derived")
    d = ST.upload_dir(cid, body.get("batch"))
    files = [(os.path.join(d, n), n) for n in sorted(os.listdir(d))]
    if not files:
        raise ValueError("no uploaded files in this batch")
    if role == "mask" and body.get("for_volume") not in m["volumes"]:
        raise ValueError("choose the image volume this mask belongs to")

    def work(step):
        step(0.05, "reading %d file(s)" % len(files))
        recs = V.load_files(files)
        step(0.6, "validating")
        out = {"volumes": [], "masks": [], "failed": []}
        for r in recs:
            if r.get("failed"):
                out["failed"].append({"name": r.get("name"), "checks": r["checks"]})
                continue
            if role == "mask":
                img = ST.load(cid)["volumes"][body["for_volume"]]
                try:
                    arr, checks = V.align_mask(r, img, allow_resample=bool(body.get("resample")))
                except V.ImportFailed as e:
                    out["failed"].append({"name": r.get("name"), "checks": e.checks})
                    continue
                names = body.get("label_names") or {}
                labels = [{"value": int(v), "name": names.get(str(int(v))) or "label %d" % int(v), "source": "import"}
                          for v in np.unique(arr) if v]
                out["masks"].append(ST.add_mask(cid, body["for_volume"], arr, labels, "import", checks=checks,
                                                note="Imported from %s" % r["name"]))
            else:
                src = _keep_sources(cid, r, files)
                out["volumes"].append(ST.add_volume(cid, r, role=role, quantity=body.get("quantity"),
                                                    value_units=body.get("value_units"), source_files=src))
        shutil.rmtree(d, ignore_errors=True)
        step(1.0, "done")
        if not out["volumes"] and not out["masks"]:
            err = V.ImportFailed("Nothing could be imported.", [c for f in out["failed"] for c in f["checks"]])
            raise err
        return out
    return ST.start_job("import", work, case=cid, role=role)


def _keep_sources(cid, rec, files):
    """Keep the original files of a series (needed to send the exact scan to the segmentation service)."""
    paths = rec.get("source_paths") or ([files[0][0]] if len(files) == 1 else [])
    if not paths:
        return []
    dst = os.path.join(ST.case_dir(cid), "sources", ST.new_id("s"))
    os.makedirs(dst, exist_ok=True)
    out = []
    for p in paths:
        q = os.path.join(dst, os.path.basename(p))
        shutil.copyfile(p, q)
        out.append(os.path.relpath(q, ST.case_dir(cid)))
    return out


def demo_case():
    """Synthetic phantom case (never patient data): sagittal + axial DICOM in one frame, labels, uncalibrated stack."""
    from .synthetic import make_fixtures
    if not os.path.isdir(os.path.join(FIXTURES, "dicom_sagittal")):
        make_fixtures(FIXTURES)
    m = ST.create_case("SYNTHETIC knee phantom (not patient data)", synthetic=True)
    cid = m["id"]
    vids = {}
    for key, files in (("sag", "dicom_sagittal"), ("ax", "dicom_axial")):
        d = os.path.join(FIXTURES, files)
        fl = [(os.path.join(d, n), n) for n in sorted(os.listdir(d))]
        r = V.load_files(fl)[0]
        vids[key] = ST.add_volume(cid, r, source_files=_keep_sources(cid, r, fl))["id"]
    lab = V.load_files([(os.path.join(FIXTURES, "sagittal_labels.nii.gz"), "sagittal_labels.nii.gz")])[0]
    img = ST.load(cid)["volumes"][vids["sag"]]
    arr, checks = V.align_mask(lab, img)
    from .synthetic import LABELS
    ST.add_mask(cid, vids["sag"], arr, [dict(l, source="synthetic ground truth") for l in LABELS], "import", checks=checks,
                note="Synthetic phantom labels (ground truth of the phantom, not a model output)")
    d = os.path.join(FIXTURES, "png_stack")
    fl = [(os.path.join(d, n), n) for n in sorted(os.listdir(d))]
    ST.add_volume(cid, V.load_files(fl)[0])
    return public_case(ST.load(cid))


# --------------------------------------------------------------------------- segmentation (KneeScan service)
def start_segmentation(body):
    import mri_bridge as MB
    m = ST.load(body.get("case"))
    v = m["volumes"][body["volume"]]
    url = MB.normalize_url(body.get("reader_url") or MB.DEFAULT_READER_URL)
    do_bones = bool(body.get("do_bones", False))
    MM.slice_geometry(v)                         # refuse early when outputs could not be mapped back
    src = [os.path.join(ST.case_dir(m["id"]), p) for p in v.get("source_files") or []]
    if not src:
        raise ValueError("The original files of this volume were not kept, so the exact scan cannot be sent to the model.")

    def work(step):
        step(0.02, "checking the segmentation service at %s" % url)
        info = MB.reader_info(url)
        if not info.get("ok"):
            raise RuntimeError("Segmentation service unavailable: %s" % info.get("error"))
        if v["fmt"] == "dicom":
            tmp = tempfile.mkdtemp(prefix="seg_")
            path = os.path.join(tmp, "series.zip")
            with zipfile.ZipFile(path, "w") as z:
                for p in src:
                    z.write(p, os.path.basename(p))
            fname = "series.zip"
        else:
            path, fname = src[0], os.path.basename(src[0])
        step(0.1, "sending the scan to KneeScan")
        res = MB.analyze_scan(path, fname, do_bones=do_bones, url=url, on_step=lambda k, msg: step(0.5, msg))
        step(0.9, "mapping the model output into the volume")
        return _store_model_result(m["id"], v, res, {"service": url, "do_bones": do_bones, "api": "Gradio /analyze"})
    return ST.start_job("segment", work, case=m["id"], volume=v["id"])


def _store_model_result(cid, v, res, settings):
    if "tear" not in res and "acl" not in res:
        raise ValueError("This is not a KneeScan results file (no 'acl' or 'tear').")
    mvol = (res.get("volume") or {})
    n_model = mvol.get("slices")
    if n_model is not None and n_model != MM._n_slices(v):
        raise ValueError("The results are for a volume with %s slices; this volume has %d. Choose the matching series."
                         % (n_model, MM._n_slices(v)))
    mask, labels, rep = MM.outlines_to_mask(v, res)
    rec = MM.model_record(res, settings, rep)
    checks = [V.chk("model_transform", "alignment", "pass" if rep["round_trip_max_vox"] < 0.01 and not rep["outside_points"] else "warn",
                    "Model outlines mapped from %s; pixel-to-voxel offset %.3f voxel; %d point(s) outside the image."
                    % (rep["source_space"], rep["round_trip_max_vox"], rep["outside_points"]), rep.get("note"))]
    ent = ST.add_mask(cid, v["id"], mask, labels, "model", model=rec, checks=checks,
                      note="KneeScan prediction (2D outline). Original prediction: edits are saved as a separate version.")
    anns = []
    for a in MM.model_measurements(v, res):
        try:
            anns.append(ST.upsert_annotation(cid, a))
        except ValueError:
            pass
    return {"mask": ent, "annotations": anns}
