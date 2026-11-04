"""Per-case imaging store (state_store/imaging/<case>/).

manifest.json holds metadata only; voxel data lives in raw little-endian files:
    vol_<vid>.raw   canonical [i, j, k] volume, written in k-major order (index = i + nx*(j + ny*k))
    mask_<mid>.raw  uint8 labels on the same grid as its volume
Masks are versioned: editing a model prediction writes a new mask whose `parent` is the prediction, so the original
stays available for comparison. Closing a view never touches this store; only `remove()` deletes data.
"""
import copy
import datetime
import hashlib
import json
import os
import re
import shutil
import threading
import time
import traceback
import uuid

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.environ.get("IMAGING_STORE", os.path.join(ROOT, "state_store", "imaging"))
ID_RE = re.compile(r"^[a-z0-9_-]{4,40}$")
NAME_RE = re.compile(r"[^A-Za-z0-9_.-]")
LOCK = threading.RLock()
MAX_UPLOAD = 1024 * 1024 * 1024

# One colour per label name everywhere (all panes, all masks). Unknown names get a stable colour from their hash.
LABEL_COLORS = {"ACL": "#f59e0b", "tear region": "#ef4444", "femur": "#38bdf8", "tibia": "#34d399", "fibula": "#f472b6",
                "PCL": "#a78bfa", "meniscus": "#fb923c", "effusion": "#22d3ee", "cartilage": "#84cc16"}
PALETTE = ["#eab308", "#06b6d4", "#d946ef", "#10b981", "#f97316", "#6366f1", "#14b8a6", "#e11d48"]


def label_color(name):
    if name in LABEL_COLORS:
        return LABEL_COLORS[name]
    return PALETTE[int(hashlib.sha1(name.encode()).hexdigest(), 16) % len(PALETTE)]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def new_id(prefix=""):
    return prefix + uuid.uuid4().hex[:10]


def case_dir(cid):
    if not ID_RE.match(cid or ""):
        raise KeyError("bad case id")
    return os.path.join(BASE, cid)


def _mpath(cid):
    return os.path.join(case_dir(cid), "manifest.json")


def list_cases():
    out = []
    if os.path.isdir(BASE):
        for d in sorted(os.listdir(BASE)):
            p = os.path.join(BASE, d, "manifest.json")
            if os.path.isfile(p):
                try:
                    m = json.load(open(p))
                    out.append({"id": d, "label": m.get("label"), "synthetic": m.get("synthetic", False),
                                "volumes": len(m.get("volumes", {})), "created": m.get("created")})
                except Exception:
                    pass
    return out


def create_case(label, synthetic=False):
    cid = new_id("c")
    os.makedirs(case_dir(cid), exist_ok=True)
    m = {"id": cid, "label": label or "Untitled case", "synthetic": bool(synthetic), "created": now(), "volumes": {},
         "masks": {}, "annotations": [], "findings": [], "scans": {}, "report_text": "", "log": []}
    save(cid, m)
    return m


def load(cid):
    p = _mpath(cid)
    if not os.path.isfile(p):
        raise KeyError("case not found")
    with open(p) as f:
        return json.load(f)


def save(cid, m):
    with LOCK:
        p = _mpath(cid)
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(m, f, indent=1, default=float)
        os.replace(tmp, p)


def log(m, action, **kw):
    m.setdefault("log", []).append(dict(action=action, t=now(), **kw))
    if len(m["log"]) > 400:
        m["log"] = m["log"][-400:]


def _write_raw(path, arr):
    a = np.ascontiguousarray(np.transpose(arr, (2, 1, 0)))     # [k, j, i] C-order == i fastest
    b = a.astype(a.dtype.newbyteorder("<"), copy=False).tobytes()
    with open(path, "wb") as f:
        f.write(b)
    return hashlib.sha1(b).hexdigest()


def read_raw(path, shape, dtype):
    a = np.fromfile(path, dtype=np.dtype(dtype).newbyteorder("<"))
    return np.transpose(a.reshape(shape[2], shape[1], shape[0]), (2, 1, 0))


# --------------------------------------------------------------------------- volumes
VOL_PUBLIC = ("name", "fmt", "shape", "dtype", "affine", "zooms", "units", "orientation_known", "spacing_known", "checks",
              "meta", "frame_id", "range", "auto_window", "initial_window", "anisotropy", "obliquity_deg", "native_axis",
              "missing_slices", "source_axcodes", "is_2d")


def add_volume(cid, rec, role="image", quantity=None, value_units=None, source_files=None, scan_id=None):
    """Store a loaded series record. Returns the public volume entry."""
    with LOCK:
        m = load(cid)
        vid = new_id("v")
        sha = _write_raw(os.path.join(case_dir(cid), "vol_%s.raw" % vid), rec["data"])
        ent = {k: copy.deepcopy(rec.get(k)) for k in VOL_PUBLIC if k in rec}
        ent.update(id=vid, role=role, file="vol_%s.raw" % vid, sha1=sha, added=now(), source_files=source_files or [],
                   model_index_map=rec.get("model_index_map"), scan_id=scan_id)
        if role == "derived":
            ent["quantity"] = quantity or "unspecified quantity"
            ent["value_units"] = value_units or ""
        m["volumes"][vid] = ent
        log(m, "add_volume", id=vid, name=ent["name"])
        save(cid, m)
        return ent


# --------------------------------------------------------------------------- scan identity / analysis records
def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def create_scan(cid, original_filename, content_hash, constituent_files=None, status="uploaded"):
    """Create one persistent imaging identity. Filename is display data; content_hash is identity."""
    with LOCK:
        m = load(cid)
        sid = new_id("scan")
        rec = {"id": sid, "original_filename": str(original_filename or "scan")[:240],
               "content_sha256": content_hash, "constituent_files": list(constituent_files or []),
               "uploaded_at": now(), "analysis_status": status, "analysis_revision": 0,
               "analysis": None, "volume_ids": [], "mask_ids": [], "measurements": [],
               "quality_flags": [], "review_status": "unreviewed", "linked_designs": []}
        m.setdefault("scans", {})[sid] = rec
        log(m, "create_scan", scan=sid, filename=rec["original_filename"], sha256=content_hash)
        save(cid, m)
        return copy.deepcopy(rec)


def update_scan(cid, sid, **patch):
    with LOCK:
        m = load(cid)
        rec = m.setdefault("scans", {}).get(sid)
        if rec is None:
            raise KeyError("scan not found")
        rec.update(copy.deepcopy(patch))
        rec["updated_at"] = now()
        log(m, "update_scan", scan=sid, fields=sorted(patch))
        save(cid, m)
        return copy.deepcopy(rec)


def begin_scan_analysis(cid, sid, model, settings=None):
    with LOCK:
        m = load(cid); rec = m.setdefault("scans", {})[sid]
        rec["analysis_revision"] = int(rec.get("analysis_revision", 0)) + 1
        rec["analysis_status"] = "running"; rec["analysis_started_at"] = now()
        rec["analysis_model"] = copy.deepcopy(model); rec["analysis_settings"] = copy.deepcopy(settings or {})
        cache_basis = {"content_sha256": rec.get("content_sha256"), "model": model, "settings": settings or {},
                       "series": [{k: s.get(k) for k in ("name", "shape", "zooms", "fmt", "frame_id")} for s in rec.get("series") or []]}
        rec["analysis_cache_key"] = hashlib.sha256(json.dumps(cache_basis, sort_keys=True).encode()).hexdigest()
        rev = rec["analysis_revision"]
        log(m, "begin_analysis", scan=sid, revision=rev)
        save(cid, m)
        return rev


def finish_scan_analysis(cid, sid, revision, **patch):
    """Apply only the result for the current scan revision; late jobs cannot overwrite newer work."""
    with LOCK:
        m = load(cid); rec = m.setdefault("scans", {})[sid]
        if int(rec.get("analysis_revision", 0)) != int(revision):
            raise ValueError("stale analysis result for scan %s revision %s" % (sid, revision))
        rec.update(copy.deepcopy(patch)); rec["analysis_finished_at"] = now(); rec["updated_at"] = now()
        log(m, "finish_analysis", scan=sid, revision=revision, status=rec.get("analysis_status"))
        save(cid, m)
        return copy.deepcopy(rec)


def scan_analysis_current(cid, sid, revision):
    rec = load(cid).get("scans", {}).get(sid) or {}
    return int(rec.get("analysis_revision", 0)) == int(revision)


def link_scan_design(cid, sid, design_id, mapping, status, revision=None):
    with LOCK:
        m = load(cid); rec = m.setdefault("scans", {})[sid]
        link = {"design_id": design_id, "created_at": now(), "analysis_revision": revision or rec.get("analysis_revision"),
                "personalization_status": status, "mapping": copy.deepcopy(mapping)}
        rec.setdefault("linked_designs", []).append(link)
        log(m, "link_design", scan=sid, design=design_id)
        save(cid, m)
        return copy.deepcopy(link)


def volume_array(cid, vid):
    m = load(cid)
    v = m["volumes"][vid]
    return read_raw(os.path.join(case_dir(cid), v["file"]), v["shape"], v["dtype"]), v


def volume_bytes_path(cid, vid):
    m = load(cid)
    v = m["volumes"][vid]
    return os.path.join(case_dir(cid), v["file"]), v


# --------------------------------------------------------------------------- masks
def add_mask(cid, vid, arr, labels, origin, parent=None, model=None, checks=None, note=None, review=None):
    with LOCK:
        m = load(cid)
        if vid not in m["volumes"]:
            raise KeyError("volume not found")
        if list(arr.shape) != m["volumes"][vid]["shape"]:
            raise ValueError("mask shape %s differs from volume %s" % (list(arr.shape), m["volumes"][vid]["shape"]))
        mid = new_id("m")
        sha = _write_raw(os.path.join(case_dir(cid), "mask_%s.raw" % mid), arr.astype(np.uint8))
        present = sorted(int(x) for x in np.unique(arr) if x)
        labs = []
        for lab in labels:
            lab = dict(lab)
            lab.setdefault("color", label_color(lab["name"]))
            lab["present"] = int(lab["value"]) in present
            labs.append(lab)
        known = {int(l["value"]) for l in labs}
        for v in present:
            if v not in known:
                labs.append({"value": v, "name": "label %d" % v, "color": label_color("label %d" % v), "present": True})
        ent = {"id": mid, "volume": vid, "labels": labs, "origin": origin, "parent": parent, "model": model, "file": "mask_%s.raw" % mid,
               "sha1": sha, "created": now(), "checks": checks or [], "note": note,
               "edited": origin == "edited", "review": review or {"status": "unreviewed"}}
        if parent and parent in m["masks"]:
            p = m["masks"][parent]
            ent["model"] = ent["model"] or p.get("model")
            ent["root"] = p.get("root") or parent
            diff = read_raw(os.path.join(case_dir(cid), p["file"]), p_shape(m, p), "uint8") != arr
            ent["edit_voxels"] = int(diff.sum())
        m["masks"][mid] = ent
        log(m, "add_mask", id=mid, origin=origin, parent=parent)
        save(cid, m)
        return ent


def p_shape(m, mask):
    return m["volumes"][mask["volume"]]["shape"]


def mask_array(cid, mid):
    m = load(cid)
    k = m["masks"][mid]
    return read_raw(os.path.join(case_dir(cid), k["file"]), p_shape(m, k), "uint8"), k


def set_review(cid, mid, status, who="user", note=None):
    if status not in ("unreviewed", "reviewed", "rejected"):
        raise ValueError("status must be unreviewed | reviewed | rejected")
    with LOCK:
        m = load(cid)
        k = m["masks"][mid]
        k["review"] = {"status": status, "by": who, "at": now(), "note": note}
        log(m, "review_mask", id=mid, status=status)
        save(cid, m)
        return k


def rename_label(cid, mid, value, name):
    with LOCK:
        m = load(cid)
        for lab in m["masks"][mid]["labels"]:
            if int(lab["value"]) == int(value):
                lab["name"] = str(name)[:40]
                lab["color"] = label_color(lab["name"])
        save(cid, m)
        return m["masks"][mid]


# --------------------------------------------------------------------------- annotations / findings
ANN_TYPES = ("landmark", "distance", "region")


def upsert_annotation(cid, item):
    with LOCK:
        m = load(cid)
        if item.get("type") not in ANN_TYPES:
            raise ValueError("annotation type must be one of %s" % (ANN_TYPES,))
        vid = item.get("volume")
        if vid not in m["volumes"]:
            raise KeyError("volume not found")
        v = m["volumes"][vid]
        A = np.array(v["affine"])
        pts = item.get("points_voxel") or []
        if not pts or any(len(p) != 3 for p in pts):
            raise ValueError("points_voxel must be a list of [i, j, k]")
        for p in pts:
            if not all(-0.5 <= p[a] <= v["shape"][a] - 0.5 for a in range(3)):
                raise ValueError("point %s is outside the volume" % (p,))
        # world coordinates and distances are computed here, from the stored affine, never taken from the client
        world = [list(map(float, (A @ np.r_[np.array(p, float), 1.0])[:3])) for p in pts]
        out = {k: item.get(k) for k in ("id", "type", "kind", "name", "volume", "note")}
        out["id"] = out["id"] or new_id("a")
        out["points_voxel"] = [[float(x) for x in p] for p in pts]
        out["points_world"] = world
        out["calibrated"] = bool(v["spacing_known"])
        out["frame"] = "RAS+ mm (%s)" % v["name"] if v["orientation_known"] else ("voxel grid of %s (no orientation)" % v["name"])
        if item["type"] == "distance":
            if len(pts) != 2:
                raise ValueError("a distance needs exactly 2 points")
            if v["spacing_known"]:
                out["value"] = round(float(np.linalg.norm(np.subtract(world[1], world[0]))), 2)
                out["units"] = "mm"
            else:
                out["value"] = round(float(np.linalg.norm(np.subtract(pts[1], pts[0]))), 2)
                out["units"] = "voxel (uncalibrated)"
        prev = next((a for a in m["annotations"] if a["id"] == out["id"]), None)
        if prev and prev.get("status") == "reviewed" and prev["points_voxel"] != out["points_voxel"]:
            out["status"] = "draft"                                  # moving a reviewed point voids the review
        else:
            out["status"] = item.get("status") if item.get("status") in ("draft", "reviewed") else (prev or {}).get("status", "draft")
        if out["status"] == "reviewed":
            if not out["calibrated"] and item["type"] == "distance":
                raise ValueError("an uncalibrated distance cannot be marked as a reviewed millimetre measurement")
            out["reviewed_at"] = (prev or {}).get("reviewed_at") if prev and prev.get("status") == "reviewed" else now()
        out["updated"] = now()
        m["annotations"] = [a for a in m["annotations"] if a["id"] != out["id"]] + [out]
        log(m, "annotation", id=out["id"], status=out["status"])
        save(cid, m)
        return out


def remove(cid, kind, oid):
    """Explicit data removal (closing a view never calls this)."""
    with LOCK:
        m = load(cid)
        d = case_dir(cid)
        if kind == "volume":
            v = m["volumes"].pop(oid)
            _rm(os.path.join(d, v["file"]))
            for mid in [k for k, mk in m["masks"].items() if mk["volume"] == oid]:
                _rm(os.path.join(d, m["masks"].pop(mid)["file"]))
            m["annotations"] = [a for a in m["annotations"] if a.get("volume") != oid]
        elif kind == "mask":
            _rm(os.path.join(d, m["masks"].pop(oid)["file"]))
        elif kind == "annotation":
            m["annotations"] = [a for a in m["annotations"] if a["id"] != oid]
        elif kind == "case":
            shutil.rmtree(d, ignore_errors=True)
            return {"removed": oid}
        else:
            raise ValueError("kind must be volume | mask | annotation | case")
        log(m, "remove", kind=kind, id=oid)
        save(cid, m)
        return {"removed": oid}


def _rm(p):
    try:
        os.remove(p)
    except OSError:
        pass


def upload_dir(cid, batch):
    if not re.match(r"^[a-z0-9]{6,32}$", batch or ""):
        raise KeyError("bad batch id")
    d = os.path.join(case_dir(cid), "uploads", batch)
    os.makedirs(d, exist_ok=True)
    return d


def safe_name(name):
    base = os.path.basename(name or "file")
    return (NAME_RE.sub("_", base)[:120] or "file").lstrip(".") or "file"


# --------------------------------------------------------------------------- background jobs
JOBS = {}


class Cancelled(Exception):
    pass


def start_job(kind, fn, **meta):
    jid = new_id("j")
    job = {"id": jid, "kind": kind, "status": "running", "progress": 0.0, "message": "starting", "started": time.time(),
           "cancel": False, "result": None, "error": None, "meta": meta}
    JOBS[jid] = job

    def step(p, msg):
        if job["cancel"]:
            raise Cancelled()
        job["progress"], job["message"] = float(p), msg

    def run():
        try:
            job["result"] = fn(step)
            job["status"] = "cancelled" if job["cancel"] else "done"
            job["progress"] = 1.0
        except Cancelled:
            job["status"], job["message"] = "cancelled", "cancelled by the user"
        except Exception as e:
            traceback.print_exc()
            job["status"], job["error"] = "error", {"message": str(e), "checks": getattr(e, "checks", None)}
        job["seconds"] = round(time.time() - job["started"], 2)
    threading.Thread(target=run, daemon=True).start()
    return public_job(job)


def public_job(job):
    return {k: job.get(k) for k in ("id", "kind", "status", "progress", "message", "result", "error", "seconds", "meta")}


def job_status(jid):
    j = JOBS.get(jid)
    if not j:
        raise KeyError("job not found")
    return public_job(j)


def cancel_job(jid):
    j = JOBS.get(jid)
    if not j:
        raise KeyError("job not found")
    j["cancel"] = True
    if j["status"] == "running":
        j["message"] = "cancelling…"
    return public_job(j)
