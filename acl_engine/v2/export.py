"""Exports and the browser payload.

Units
-----
- STL: millimetres, model frame (+X lateral for right knees; left knees are
  exported in true patient orientation, i.e. mirrored), binary.
- GLB: glTF convention is metres, so positions are mm / 1000. Every node has
  extras {id, category, status, region, material, composition, units}.
  Scaffold meshes carry a per-vertex custom attribute _REGION (float, region
  index into regions.json) and fascicle/footprint meshes _BUNDLE (0 AM, 1 PL).
- JSON: design.json (complete design state incl. provenance, imported baseline
  and edit history), regions.json (multiphasic assignments), validation.json
  (checks + metrics), manifest.json (hashes, units, file list).
"""
import base64
import hashlib
import io
import json
import struct
import zipfile
import numpy as np
from acl_engine.v2 import mesher
from acl_engine.v2.design import REGION_IDS, MATERIALS

# ---------------------------------------------------------------- browser payload
def _b64(a, dt):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dt).tobytes()).decode("ascii")


def payload(scene, have_keys=()):
    """Compact JSON for the viewer. Bone/cartilage meshes the client already holds
    (by cache key) are sent without vertex data."""
    objs = []
    for o in scene["objects"]:
        meta = {k: o.get(k) for k in ("id", "label", "category", "kind", "status", "note", "region", "region_kind",
                                       "material", "composition", "color", "cache_key", "ckey", "unit", "units",
                                       "level", "hollow", "shape", "aspect", "exponent", "end", "markers")}
        if o["kind"] == "mesh":
            if o.get("cache_key") and o["cache_key"] in have_keys:
                meta["cached"] = True
            else:
                meta["v"] = _b64(o["V"].ravel(), "<f4")
                meta["f"] = _b64(o["F"].ravel(), "<u4")
                for an, av in (o.get("attr") or {}).items():
                    if av is not None:
                        meta.setdefault("attr", {})[an] = _b64(av, "u1")
            meta["nv"] = int(o["V"].shape[0])
        else:
            lines = o["lines"]
            pts = np.vstack([ln["P"] for ln in lines]) if lines else np.zeros((0, 3))
            offs = np.cumsum([0] + [ln["P"].shape[0] for ln in lines])
            rad = np.concatenate([np.asarray(ln["r"], float) for ln in lines]) if lines else np.zeros(0)
            meta["p"] = _b64(pts.ravel(), "<f4")
            meta["o"] = _b64(offs, "<u4")
            meta["r"] = _b64(rad, "<f4")
            per_line = {}
            for key in ("bundle", "fascicle", "fiber", "yarn", "collide", "bridging", "unit", "layer", "group", "channel",
                        "strand", "outline", "where"):
                if lines and key in lines[0]:
                    per_line[key] = [int(ln.get(key, 0)) for ln in lines]
            if lines and "family" in lines[0]:
                per_line["family"] = [ln["family"] for ln in lines]
            meta["line_meta"] = per_line
            if lines and "region" in lines[0]:
                meta["region_pts"] = _b64(np.concatenate([np.asarray(ln["region"]) for ln in lines]), "u1")
            if lines and lines[0].get("viol") is not None:
                meta["viol_pts"] = _b64(np.concatenate([np.asarray(ln["viol"], np.uint8) for ln in lines]), "u1")
            if lines and "ri" in lines[0]:
                meta["ri_pts"] = _b64(np.concatenate([np.asarray(ln["ri"], float) for ln in lines]), "<f4")
            if lines and "u" in lines[0]:
                meta["u_pts"] = _b64(np.vstack([ln["u"] for ln in lines]).ravel(), "<f4")
            if lines and "coat" in lines[0]:
                meta["coat_pts"] = _b64(np.concatenate([np.asarray(ln["coat"], np.uint8) for ln in lines]), "u1")
            if lines and "mcol" in lines[0]:
                meta["mcol_pts"] = _b64(np.vstack([np.asarray(ln["mcol"], np.uint8).reshape(-1, 3) for ln in lines]).ravel(), "u1")
            if lines and "damage" in lines[0]:
                meta["damage_pts"] = _b64(np.clip(np.concatenate([ln["damage"] for ln in lines]) * 255, 0, 255), "u1")
            meta["n_lines"] = len(lines)
        objs.append(meta)
    return {"ok": True, "objects": objs, "checks": scene["checks"], "metrics": _jsonable(scene["metrics"]),
            "validity": scene.get("validity"), "resolution_candidates": len(scene.get("resolution_candidates") or []),
            "section": scene["section"], "warnings": scene["warnings"], "timings_s": scene["timings_s"],
            "design_hash": scene["design_hash"], "geometry_hash": scene["geometry_hash"], "handles": scene["handles"],
            "side": scene["side"], "quality": scene["quality"], "frame": scene["frame"], "inactive": scene.get("inactive", {}),
            "regions": REGION_IDS, "materials": MATERIALS}


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


# ---------------------------------------------------------------- tube solids
def tube_solids(obj, by="region", nsec=10):
    """Polyline set -> {key: (V, F)} closed tube solids, split by region (or 'all')."""
    groups = {}
    for ln in obj["lines"]:
        P = ln["P"]
        r = np.asarray(ln["r"], float)
        if by == "region" and "region" in ln:
            reg = np.asarray(ln["region"])
            m = np.concatenate([[True], reg[1:] != reg[:-1]])
            starts = np.nonzero(m)[0].tolist() + [P.shape[0]]
            for a, b in zip(starts[:-1], starts[1:]):
                a0 = max(0, a - 1) if a > 0 else 0   # share the boundary sample: pieces touch
                if b - a0 < 2:
                    continue
                groups.setdefault(int(reg[a]), []).append(mesher.tube_mesh(P[a0:b], r[a0:b], nsec))
        else:
            if P.shape[0] >= 2:
                groups.setdefault("all", []).append(mesher.tube_mesh(P, r, nsec))
    return {k: mesher.concat(v) for k, v in groups.items()}


# ---------------------------------------------------------------- STL / GLB writers
def stl_bytes(V, F, header="acl-engine v2, units mm"):
    tri = V[F].astype(np.float32)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(F.shape[0], dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"] = n
    rec["v"] = tri
    return header.encode()[:80].ljust(80, b" ") + struct.pack("<I", F.shape[0]) + rec.tobytes()


def glb_bytes(nodes):
    """nodes: list of {name, V (mm), F, color [r,g,b,a], extras, attrs {NAME: per-vertex floats}}."""
    bin_chunks, views, accessors, meshes, gnodes, mats = [], [], [], [], [], []
    off = 0

    def add(data, target, comp, count, typ, mn=None, mx=None):
        nonlocal off
        b = data.tobytes()
        pad = (-len(b)) % 4
        views.append({"buffer": 0, "byteOffset": off, "byteLength": len(b), **({"target": target} if target else {})})
        bin_chunks.append(b + b"\0" * pad)
        off += len(b) + pad
        acc = {"bufferView": len(views) - 1, "componentType": comp, "count": count, "type": typ}
        if mn is not None:
            acc["min"], acc["max"] = mn, mx
        accessors.append(acc)
        return len(accessors) - 1

    for nd in nodes:
        V = (np.asarray(nd["V"], float) / 1000.0).astype(np.float32)
        F = np.asarray(nd["F"], np.uint32)
        if V.shape[0] == 0 or F.shape[0] == 0:
            continue
        pa = add(V, 34962, 5126, V.shape[0], "VEC3", V.min(0).tolist(), V.max(0).tolist())
        ia = add(F.ravel(), 34963, 5125, F.size, "SCALAR")
        attrs = {"POSITION": pa}
        for an, av in (nd.get("attrs") or {}).items():
            attrs[an] = add(np.asarray(av, np.float32), 34962, 5126, V.shape[0], "SCALAR")
        col = nd.get("color") or [0.8, 0.8, 0.8, 1.0]
        mats.append({"name": nd["name"] + "_mat", "pbrMetallicRoughness": {"baseColorFactor": col, "metallicFactor": 0.0, "roughnessFactor": 0.8},
                     **({"alphaMode": "BLEND"} if col[3] < 1 else {}), "doubleSided": True})
        meshes.append({"name": nd["name"], "primitives": [{"attributes": attrs, "indices": ia, "material": len(mats) - 1}]})
        gnodes.append({"name": nd["name"], "mesh": len(meshes) - 1, "extras": nd.get("extras", {})})
    binb = b"".join(bin_chunks)
    doc = {"asset": {"version": "2.0", "generator": "acl-engine v2", "extras": {"units": "metres (mm / 1000)", "frame": "+X lateral, +Y anterior, +Z proximal"}},
           "scene": 0, "scenes": [{"nodes": list(range(len(gnodes)))}], "nodes": gnodes, "meshes": meshes,
           "materials": mats, "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(binb)}]}
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    total = 12 + 8 + len(js) + 8 + len(binb)
    return (struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(js), 0x4E4F534A) + js
            + struct.pack("<II", len(binb), 0x004E4942) + binb)


STATUS_COLOR = {"layer_sheath": [0.80, 0.89, 0.91, 0.5], "layer_end": [0.85, 0.75, 0.85, 0.8], "layer_plug": [0.75, 0.7, 0.62, 1.0],
                "layer_pores": [0.1, 0.12, 0.14, 1.0], "layer_coating": [0.18, 0.77, 0.71, 0.6],
                "layer_interface": [0.55, 0.45, 0.75, 0.35], "layer_anchors": [0.95, 0.65, 0.25, 1.0],
                "bone": [0.90, 0.87, 0.80, 1.0], "cartilage": [0.62, 0.74, 0.92, 0.9], "footprint": [0.98, 0.83, 0.45, 1.0],
                "retained": [0.86, 0.45, 0.42, 0.8], "injury": [0.85, 0.15, 0.15, 0.35], "reference": [0.95, 0.65, 0.6, 0.2],
                "bundle": [0.9, 0.6, 0.4, 0.3], "landmark": [0.75, 0.75, 0.82, 0.6], "observed": [0.3, 0.8, 0.9, 0.6]}


def export_zip(state, scene):
    """Complete, reproducible export package (bytes of a .zip)."""
    files = {}
    nodes = []
    regions_table = []
    for o in scene["objects"]:
        ex = {"id": o["id"], "category": o["category"], "status": o["status"], "note": o.get("note", ""),
              "units_source": "mm"}
        if o["kind"] == "mesh":
            col = list(o.get("color") or STATUS_COLOR.get(o["category"], [0.8, 0.8, 0.8]))[:3]
            col = col + [STATUS_COLOR.get(o["category"], [0, 0, 0, 1.0])[3] if len(col) == 3 else 1.0]
            attrs = {}
            if o.get("attr", {}).get("bundle") is not None:
                attrs["_BUNDLE"] = o["attr"]["bundle"].astype(np.float32)
            if o["category"] == "scaffold_region":
                ri = REGION_IDS.index(o["region"])
                attrs["_REGION"] = np.full(o["V"].shape[0], ri, np.float32)
                ex.update(region=o["region"], region_kind=o["region_kind"], material=o["material"], composition=o["composition"])
            nodes.append({"name": o["id"], "V": o["V"], "F": o["F"], "color": col, "extras": ex, "attrs": attrs})
            # layer_interface is a zero-thickness visualization sheet and layer_anchors are
            # schematic markers: both ship in the GLB/scene but are not printable solids.
            if o["category"] not in ("bone", "cartilage", "landmark", "bundle", "reference", "violation",
                                     "layer_interface", "layer_anchors") and o["F"].shape[0]:
                files["stl/%s.stl" % o["id"]] = stl_bytes(o["V"], o["F"], "acl-engine v2 %s mm" % o["id"])
        elif o["id"] == "scaffold_yarns":
            solids = tube_solids(o, "region")
            allp = []
            for ri, (V, F) in sorted(solids.items()):
                name = REGION_IDS[ri]
                mat = state["params"]["regions.%s.material" % name]
                col = MATERIALS[mat]["color"] + [1.0]
                nodes.append({"name": "scaffold_yarns_%s" % name, "V": V, "F": F, "color": col,
                              "extras": dict(ex, region=name, material=mat), "attrs": {"_REGION": np.full(V.shape[0], ri, np.float32)}})
                files["stl/scaffold_yarns_%s.stl" % name] = stl_bytes(V, F, "acl-engine v2 scaffold %s mm" % name)
                allp.append((V, F))
            if allp:
                V, F = mesher.concat(allp)
                files["stl/scaffold_yarns_all.stl"] = stl_bytes(V, F, "acl-engine v2 scaffold all regions mm")
        elif o.get("hollow"):
            from acl_engine.v2.layered import hollow_tube_mesh
            parts = [hollow_tube_mesh(ln["P"], ln["r"], ln["ri"], ln["u"], nsec=12, aspect=o.get("aspect", 1.0),
                                      exponent=o.get("exponent", 2.0)) for ln in o["lines"] if ln["P"].shape[0] >= 2]
            if parts:
                V, F = mesher.concat(parts)
                nodes.append({"name": o["id"], "V": V, "F": F, "color": [0.95, 0.93, 0.86, 1.0], "extras": dict(ex, level=o.get("level"))})
                files["stl/%s.stl" % o["id"]] = stl_bytes(V, F, "acl-engine v2 %s (hollow, closed annular solids) mm" % o["id"])
        elif o["id"] == "layer_braid":
            for g, lab in ((0, "am"), (1, "pl")):
                parts = [mesher.tube_mesh(ln["P"], ln["r"], 10) for ln in o["lines"] if ln.get("group") == g]
                if parts:
                    V, F = mesher.concat(parts)
                    comp = state["params"].get("rf.%s.composition" % lab)
                    nodes.append({"name": "layer_braid_%s" % lab, "V": V, "F": F, "color": [0.91, 0.64, 0.23, 1.0] if g == 0 else [0.31, 0.6, 0.84, 1.0],
                                  "extras": dict(ex, level=4, region=lab + "-like", composition=comp)})
                    files["stl/layer_braid_%s.stl" % lab] = stl_bytes(V, F, "acl-engine v2 braid %s mm" % lab)
        elif o["id"] == "native_fascicles":
            V, F = tube_solids(o, "all", 8).get("all", (np.zeros((0, 3)), np.zeros((0, 3), int)))
            nodes.append({"name": "native_fascicles", "V": V, "F": F, "color": [0.86, 0.45, 0.42, 1.0], "extras": ex})
    sm = scene["metrics"].get("scaffold") or {}
    for r in sm.get("regions", []):
        regions_table.append(dict(r, index=REGION_IDS.index(r["id"]),
                                  note="material and composition are design metadata; no mechanical, degradation or biological model is evaluated"))
    lay = (scene["metrics"] or {}).get("layered")
    if lay:
        from acl_engine.v2.design import LEVELS, BY_ID
        from acl_engine.v2.detail import detail_micro
        spec = {}
        for L in LEVELS:
            spec[str(L["n"])] = {"name": L["name"], "title": L["title"], "component": L["component"],
                                 "controls": {pid: {"value": state["params"].get(pid), "units": q["units"], "component": q.get("component"),
                                                    "status": (state.get("provenance") or {}).get(pid, {}).get("status"),
                                                    "basis": (state.get("provenance") or {}).get(pid, {}).get("basis", q.get("basis"))}
                                              for pid, q in BY_ID.items() if q.get("level") == L["n"] and not q.get("legacy")}}
        files["layers.json"] = json.dumps(_jsonable({
            "note": "Six-level construct. 'calculated' rows come from the geometry (and, where stated, from entered assumptions); "
                    "assumed properties are inputs, not results. Nothing here is a validated mechanical or biological prediction.",
            "levels": spec, "derived": lay["derived"], "calculated": lay["calculated"], "assumptions": lay["assumptions"],
            "ends": lay["ends"], "footprint_source": lay.get("footprint_source"), "conflicts": lay["conflicts"]}), indent=1).encode()
        try:
            g = detail_micro(state["params"], lay)
            V, F = g["parts"][0]["V"], g["parts"][0]["F"]
            files["stl/detail_grooved_channel_segment.stl"] = stl_bytes(V, F, "acl-engine v2 grooved channel half-segment, true scale mm")
        except Exception:
            pass
    files["scene.glb"] = glb_bytes(nodes)
    files["design.json"] = json.dumps(_jsonable(state), indent=1).encode()
    files["regions.json"] = json.dumps(_jsonable({"regions": regions_table, "region_ids": REGION_IDS}), indent=1).encode()
    files["validation.json"] = json.dumps(_jsonable({"validity": scene.get("validity"), "checks": scene["checks"], "metrics": scene["metrics"],
                                                     "design_hash": scene["design_hash"], "geometry_hash": scene["geometry_hash"]}), indent=1).encode()
    files["history.json"] = json.dumps(_jsonable({"history": state.get("history", []),
                                                  "imported_case": (state.get("imported") or {}).get("meta")}), indent=1).encode()
    manifest = {"design_hash": scene["design_hash"], "geometry_hash": scene["geometry_hash"],
                "design_valid": bool((scene.get("validity") or {}).get("valid")),
                "units": {"stl": "mm", "glb": "m (mm/1000)", "json": "mm, degrees"},
                "frame": scene["frame"], "side": scene["side"],
                "case": state.get("case"), "files": {k: hashlib.sha256(v).hexdigest()[:16] for k, v in files.items()},
                "reproduce": "load design.json in the designer (Import > design file) or acl_engine.v2.scene.build_scene(json.load(open('design.json')))"}
    files["manifest.json"] = json.dumps(_jsonable(manifest), indent=1).encode()
    files["README.txt"] = ("ACL repair-scaffold export (acl-engine v2)\n"
                           "Units: STL in mm; GLB in metres (glTF convention, mm/1000).\n"
                           "Frame: +X lateral, +Y anterior, +Z proximal, origin tibial plateau centre (left knees mirrored to patient orientation).\n"
                           "Statuses: observed = patient data, assumed = template/synthetic, estimated = reconstruction, proposed = scaffold design.\n"
                           "Material and composition are metadata; strength, degradation, healing and re-tear risk are NOT evaluated.\n").encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for k in sorted(files):
            z.writestr(k, files[k])
    return buf.getvalue(), manifest
