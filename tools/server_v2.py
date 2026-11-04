"""v2 API (acl-engine v2): anatomical scene, patient import, state store, exports.

Routes (all JSON unless noted):
  GET  /api/v2/registry                 parameter registry, groups, materials
  GET  /api/v2/cases                    built-in synthetic cases + bundled patient inputs
  POST /api/v2/new      {case}          new design state for a built-in case
  POST /api/v2/import   {case | document, assets?, current?, mode}
                                         -> {state, report, conflicts}
  POST /api/v2/build    {state, quality, have_keys, seq}
  POST /api/v2/detail   {state, selection}
  POST /api/v2/section  {state, kind, pos}  planar section of the built geometry (cached scene)
  POST /api/v2/resolutions {state}      verified fixes for a design that does not fit
  POST /api/v2/state    {state}         -> {id}  (persisted to state_store/)
  GET  /api/v2/state?id=...             -> {state} | 404 with a recovery hint
  POST /api/v2/export   {state}         -> application/zip
"""
import base64
import copy
from collections import OrderedDict
import hashlib
import json
import os
import re
import tempfile
import threading
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE = os.path.join(ROOT, "state_store")
PATIENTS = os.path.join(ROOT, "fixtures", "patients")
UPLOADS = os.path.join(STORE, "uploads")
ID_RE = re.compile(r"^[a-f0-9]{8,40}$")
LOCK = threading.Lock()
SCENES = OrderedDict()          # design_hash/quality -> built scene (sections, fix verification)
SAFE_ASSET = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")


def builtin_cases():
    from acl_engine.v2.design import new_state
    cases = [{"id": "demo_complete", "label": "Synthetic demo: complete midsubstance tear (right)", "kind": "builtin"},
             {"id": "demo_partial_am", "label": "Synthetic demo: partial AM tear (right)", "kind": "builtin"},
             {"id": "demo_proximal", "label": "Synthetic demo: proximal tear near femur (right)", "kind": "builtin"},
             {"id": "demo_intact", "label": "Synthetic demo: intact ACL (right)", "kind": "builtin"}]
    if os.path.isdir(PATIENTS):
        for d in sorted(os.listdir(PATIENTS)):
            if os.path.isfile(os.path.join(PATIENTS, d, "input.json")):
                cases.append({"id": d, "label": "MRI-input fixture: %s" % d.replace("_", " "), "kind": "patient"})
    return cases


def new_case(case_id):
    from acl_engine.v2.design import new_state
    st = new_state()
    p = st["params"]
    lab = {"demo_complete": "Synthetic demo: complete midsubstance tear",
           "demo_partial_am": "Synthetic demo: partial AM tear",
           "demo_proximal": "Synthetic demo: proximal tear near femur",
           "demo_intact": "Synthetic demo: intact ACL"}
    if case_id not in lab:
        raise KeyError(case_id)
    if case_id == "demo_partial_am":
        p.update({"injury.type": "partial", "injury.involvement": "AM", "scaffold.arch": "aligned",
                  "scaffold.engagement": "abut", "scaffold.yarn_diameter_mm": 0.3, "injury.center_s": 0.6})
    elif case_id == "demo_proximal":
        p.update({"injury.center_s": 0.8, "injury.gap_mm": 4.0, "scaffold.arch": "aligned", "scaffold.engagement": "abut",
                  "scaffold.yarn_diameter_mm": 0.3, "scaffold.yarn_count": 16})
    elif case_id == "demo_intact":
        p.update({"injury.type": "intact"})
    st["case"] = {"id": case_id, "label": lab[case_id] + " (SYNTHETIC, literature template)", "synthetic": True,
                  "notes": "Reproducible demonstration case. Not patient anatomy."}
    for k in ("injury.type", "injury.involvement", "injury.center_s", "injury.gap_mm", "scaffold.arch", "scaffold.engagement", "scaffold.yarn_diameter_mm", "scaffold.yarn_count"):
        st["provenance"][k] = {"status": "default", "source": "synthetic demo case", "units": st["provenance"][k].get("units", "")}
    return st


def _state_id(state):
    core = {k: state.get(k) for k in ("case", "params", "provenance", "imported", "assets", "history", "display")}
    return hashlib.sha1(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()[:20]


def save_state(state):
    os.makedirs(STORE, exist_ok=True)
    sid = _state_id(state)
    p = os.path.join(STORE, sid + ".json")
    if not os.path.exists(p):
        with open(p + ".tmp", "w") as f:
            json.dump(state, f)
        os.replace(p + ".tmp", p)
    return sid


def load_state(sid):
    if not ID_RE.match(sid or ""):
        return None
    p = os.path.join(STORE, sid + ".json")
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)


def import_request(body):
    from acl_engine.v2.patient import import_file, import_document, merge_import, AdapterError
    mode = body.get("mode", "replace")
    try:
        if body.get("case"):
            name = body["case"]
            if not re.match(r"^[A-Za-z0-9_-]{1,60}$", name):
                return 400, {"error": "bad case name"}
            path = os.path.join(PATIENTS, name, "input.json")
            if not os.path.isfile(path):
                return 404, {"error": "case not found", "hint": "see /api/v2/cases"}
            st, rep = import_file(path)
        elif isinstance(body.get("document"), dict):
            doc = body["document"]
            assets = body.get("assets") or {}
            h = hashlib.sha1(json.dumps(doc, sort_keys=True).encode() + b"".join(sorted(k.encode() for k in assets))).hexdigest()[:16]
            d = os.path.join(UPLOADS, h)
            os.makedirs(d, exist_ok=True)
            for name, b64 in assets.items():
                if not SAFE_ASSET.match(name):
                    return 400, {"error": "bad asset file name: %s" % name, "hint": "letters, digits, . _ - only"}
                with open(os.path.join(d, name), "wb") as f:
                    f.write(base64.b64decode(b64))
            st, rep = import_document(doc, d)
        else:
            return 400, {"error": "send {case} or {document, assets}"}
    except AdapterError as e:
        return 422, {"error": "import rejected: %s" % e, "report": {"errors": [str(e)]}}
    if st is None:
        return 422, {"error": "import rejected: %d problem(s)" % len(rep["errors"]), "report": rep}
    conflicts = []
    cur = body.get("current")
    if cur and mode == "merge":
        st, conflicts = merge_import(cur, st, "merge")
    elif cur:
        st, conflicts = merge_import(cur, st, "replace")
    return 200, {"state": st, "report": rep, "conflicts": conflicts}


def build_request(body):
    from acl_engine.v2.scene import build_scene
    from acl_engine.v2.export import payload
    st = body.get("state")
    if not isinstance(st, dict) or "params" not in st:
        return 400, {"error": "missing state"}
    q = body.get("quality", "full")
    if q not in ("full", "preview"):
        return 400, {"error": "quality must be full|preview"}
    t0 = time.time()
    with LOCK:
        sc = build_scene(st, q)
        if sc.get("ok"):
            _remember(sc)
    if not sc.get("ok"):
        return 422, {"error": "invalid parameters", "errors": sc["errors"], "warnings": sc.get("warnings", []), "seq": body.get("seq")}
    out = payload(sc, set(body.get("have_keys") or []))
    out["seq"] = body.get("seq")
    out["server_s"] = time.time() - t0
    return 200, out


def _remember(sc):
    key = (sc["design_hash"], sc["quality"])
    SCENES[key] = sc
    SCENES.move_to_end(key)
    while len(SCENES) > 4:
        SCENES.popitem(last=False)


def _scene_for(st, quality="full"):
    from acl_engine.v2.design import design_hash
    from acl_engine.v2.scene import build_scene
    key = (design_hash(st), quality)
    sc = SCENES.get(key)
    if sc is None:
        sc = build_scene(st, quality)
        if not sc.get("ok"):
            return sc
        _remember(sc)
    return sc


def section_request(body):
    """Section of the SAME built geometry the viewer shows (cached by design hash)."""
    from acl_engine.v2 import sections as secmod
    st = body.get("state")
    if not isinstance(st, dict) or "params" not in st:
        return 400, {"error": "missing state"}
    kind = body.get("kind", "ligament")
    if kind not in ("ligament", "am", "pl", "axial", "sagittal", "coronal"):
        return 400, {"error": "unknown section kind"}
    with LOCK:
        sc = _scene_for(st, body.get("quality", "full"))
        if not sc.get("ok"):
            return 422, {"error": "invalid parameters", "errors": sc["errors"]}
        pos = float(body.get("pos", 0.5 if kind in ("ligament", "am", "pl") else 0.0))
        unc = float(sc["lig"].p["bundle.interface_uncertainty_mm"])
        sec = secmod.section(sc["objects"], sc["lig"], kind, pos, sc["side"], unc, (sc["metrics"].get("fit") or {}).get("violations"))
    sec["design_hash"] = sc["design_hash"]
    return 200, {"section": sec}


def resolutions_request(body):
    from acl_engine.v2.scene import verify_resolutions
    st = body.get("state")
    if not isinstance(st, dict) or "params" not in st:
        return 400, {"error": "missing state"}
    t0 = time.time()
    with LOCK:
        sc = _scene_for(st, "full")
        if not sc.get("ok"):
            return 422, {"error": "invalid parameters", "errors": sc["errors"]}
        res = verify_resolutions(sc)
    return 200, {"resolutions": res, "design_hash": sc["design_hash"], "candidates": len(sc.get("resolution_candidates") or []),
                 "server_s": time.time() - t0}


def detail_request(body):
    from acl_engine.v2.scene import detail_fibers
    from acl_engine.v2.export import _b64
    import numpy as np
    st = body.get("state")
    sel = body.get("selection") or {}
    with LOCK:
        r = detail_fibers(st, sel)
    if not r.get("ok"):
        return 400, r
    lines = r["lines"]
    pts = np.vstack([ln["P"] for ln in lines]) if lines else np.zeros((0, 3))
    offs = np.cumsum([0] + [ln["P"].shape[0] for ln in lines])
    rad = np.concatenate([ln["r"] for ln in lines]) if lines else np.zeros(0)
    return 200, {"ok": True, "p": _b64(pts.ravel(), "<f4"), "o": _b64(offs, "<u4"), "r": _b64(rad, "<f4"),
                 "n_lines": len(lines), "note": r["note"], "window_mm": r["window_mm"], "physical": r["physical"]}


def export_request(body):
    from acl_engine.v2.scene import build_scene
    from acl_engine.v2.stl_export import build_export
    st = body.get("state")
    if not isinstance(st, dict) or "params" not in st:
        return 400, {"error": "missing state"}, None
    # JSON round-trip creates an immutable request snapshot even if the browser
    # continues editing while the threaded export is running.
    st = json.loads(json.dumps(st))
    with LOCK:
        sc = build_scene(st, "full")
    if not sc.get("ok"):
        return 422, {"error": "invalid parameters", "errors": sc["errors"]}, None
    try:
        man, data = build_export(st, sc, body.get("options") or {})
    except NotImplementedError as e:
        return 422, {"error": str(e), "category": "unsupported_fused_solid"}, None
    except ValueError as e:
        return 422, {"error": str(e), "category": "invalid_export_selection"}, None
    return 200, man, data


def export_inspect_request(body):
    from acl_engine.v2.scene import build_scene
    from acl_engine.v2.stl_export import build_export
    st = body.get("state")
    if not isinstance(st, dict) or "params" not in st:
        return 400, {"error": "missing state"}
    st = json.loads(json.dumps(st))
    with LOCK:
        sc = build_scene(st, "full")
    if not sc.get("ok"):
        return 422, {"error": "invalid parameters", "errors": sc["errors"]}
    try:
        out, _ = build_export(st, sc, body.get("options") or {}, preview=True)
    except NotImplementedError as e:
        return 422, {"error": str(e), "category": "unsupported_fused_solid"}
    except ValueError as e:
        return 422, {"error": str(e), "category": "invalid_export_selection"}
    return 200, out


def level_detail_request(body):
    """Magnified true-scale detail of one scaffold level (geometry from the same parameters)."""
    from acl_engine.v2.detail import level_detail
    from acl_engine.v2.design import coerce_types, migrate_params
    from acl_engine.v2.export import _b64
    import numpy as np
    st = body.get("state")
    if not isinstance(st, dict) or "params" not in st:
        return 400, {"error": "missing state"}
    try:
        level = int(body.get("level"))
    except (TypeError, ValueError):
        return 400, {"error": "level must be 1-6"}
    if not 1 <= level <= 6:
        return 400, {"error": "level must be 1-6"}
    region = body.get("region")
    relief = float(body.get("relief_x", 1.0))
    with LOCK:
        sc = _scene_for(st, "full")
        if not sc.get("ok"):
            return 422, {"error": "invalid parameters", "errors": sc["errors"]}
        rep = (sc["metrics"] or {}).get("layered")
        params = dict(st["params"])
        migrate_params(params)
        r = level_detail(coerce_types(params), level, region, rep, max(1.0, min(relief, 50.0)))
    parts = []
    for prt in r["parts"]:
        q = {"name": prt["name"], "color": prt["color"], "opacity": prt.get("opacity", 1.0),
             "v": _b64(np.asarray(prt["V"]).ravel(), "<f4"), "f": _b64(np.asarray(prt["F"]).ravel(), "<u4")}
        if prt.get("vcol") is not None:
            q["vcol"] = _b64((np.asarray(prt["vcol"]) * 255).astype(np.uint8).ravel(), "u1")
        parts.append(q)
    return 200, {"ok": True, "level": level, "region": region, "parts": parts, "window_mm": r["window_mm"], "note": r["note"],
                 "relief_x": r["relief_x"], "up": r.get("up", "z"), "design_hash": sc["design_hash"]}


def layer_fixes_request(body):
    """Pre-test the concrete fixes offered for the layered construct's conflicts."""
    from acl_engine.v2.scene import verify_layer_fixes
    st = body.get("state")
    if not isinstance(st, dict) or "params" not in st:
        return 400, {"error": "missing state"}
    t0 = time.time()
    with LOCK:
        sc = _scene_for(st, "full")
        if not sc.get("ok"):
            return 422, {"error": "invalid parameters", "errors": sc["errors"]}
        res = verify_layer_fixes(sc)
    return 200, {"fixes": res, "design_hash": sc["design_hash"], "server_s": time.time() - t0}


def inactive_request(body):
    from acl_engine.v2.design import inactive_params, migrate_params, coerce_types
    params = dict((body.get("state") or {}).get("params") or {})
    migrate_params(params)
    return 200, {"inactive": inactive_params(coerce_types(params))}
