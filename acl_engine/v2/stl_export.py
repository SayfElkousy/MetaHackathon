"""Selection-aware scaffold STL export.

The scene builder is the geometry authority.  This module only converts its
physical scaffold objects (plus audited parametric custom parts) to immutable
triangle meshes in the engine's millimetre model frame.  Viewer transforms,
visibility, clipping, opacity and exploded-layout offsets never enter here.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import zipfile

import numpy as np

from acl_engine.context import box_mesh
from acl_engine.v2 import mesher
from acl_engine.v2.export import _jsonable, stl_bytes
from acl_engine.v2.layered import hollow_tube_mesh


DETAIL_TOLERANCE_MM = {"draft": 0.15, "standard": 0.06, "high": 0.025}
DISPLAY_ONLY = {
    "layer_grooves": "Microgrooves are assembled-view guide lines; only the separate true-scale detail specimen has groove geometry.",
    "layer_interface": "The inter-bundle interface is a zero-thickness visualization sheet.",
    "layer_anchors": "Anchor points are schematic markers, not fixation hardware.",
    "layer_footprint_tib": "The tibial footprint outline is a viewer guide.",
    "layer_footprint_fem": "The femoral footprint outline is a viewer guide.",
}
COMPONENT_ALIASES = {
    "sheath": ["layer_sheath"], "pores": ["layer_pores"], "mantle_am": ["layer_braid_am"],
    "mantle_pl": ["layer_braid_pl"], "core_am": ["layer_channels_am"],
    "core_pl": ["layer_channels_pl"], "core_shared": ["layer_channels_shared"],
    "ports": ["layer_ports"], "end_tib": ["layer_end_tib", "layer_plug_tib"],
    "end_fem": ["layer_end_fem", "layer_plug_fem"], "coating": ["layer_coating", "layer_coating_sheath"],
}


def _nsec(radius, tolerance):
    radius = max(float(radius), 1e-6)
    tolerance = max(min(float(tolerance), radius * 0.95), 1e-5)
    # Sagitta <= tolerance for a circular polygon.
    n = math.ceil(math.pi / math.acos(max(-1.0, min(1.0, 1.0 - tolerance / radius))))
    return max(8, min(96, n))


def _safe_id(value):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")[:72] or "part"


def _mesh_part(pid, label, level, V, F, source="engine", **meta):
    V = np.asarray(V, dtype=float)
    F = np.asarray(F, dtype=np.int64)
    if F.size and mesher.audit(V, F).get("volume_mm3", 0.0) < 0:
        F = F[:, [0, 2, 1]]  # mirrored transforms need their winding reversed
        meta["winding_flipped"] = True
    return {"id": _safe_id(pid), "label": label, "level": level, "V": V, "F": F,
            "source": source, **meta}


def _tube_object(o, pid, label, lines, tolerance, hollow=False):
    parts = []
    for ln in lines:
        if np.asarray(ln["P"]).shape[0] < 2:
            continue
        ro = np.asarray(ln["r"], float)
        ns = _nsec(float(np.max(ro)), tolerance)
        if hollow:
            parts.append(hollow_tube_mesh(ln["P"], ro, ln["ri"], ln["u"], nsec=ns,
                                          aspect=o.get("aspect", 1.0), exponent=o.get("exponent", 2.0)))
        else:
            parts.append(mesher.tube_mesh(ln["P"], ro, ns))
    if not parts:
        return None
    V, F = mesher.concat(parts)
    return _mesh_part(pid, label, o.get("level"), V, F, hollow=hollow,
                      note=o.get("note", ""), category=o.get("category"))


def _scene_parts(scene, tolerance):
    out, omitted = [], []
    for o in scene["objects"]:
        oid = o["id"]
        if oid in DISPLAY_ONLY:
            omitted.append({"id": oid, "level": o.get("level"), "reason": DISPLAY_ONLY[oid]})
            continue
        if o["kind"] == "mesh" and o.get("category") in {
            "scaffold_region", "layer_sheath", "layer_pores", "layer_end", "layer_plug"
        }:
            if o["F"].shape[0]:
                out.append(_mesh_part(oid, o.get("label", oid), o.get("level"), o["V"], o["F"],
                                      note=o.get("note", ""), category=o.get("category")))
        elif oid == "scaffold_yarns":
            q = _tube_object(o, "scaffold_yarns", "Scaffold yarns", o["lines"], tolerance)
            if q: out.append(q)
        elif oid == "layer_braid":
            for group, suffix, label in ((0, "am", "AM braided mantle"), (1, "pl", "PL braided mantle")):
                q = _tube_object(o, "layer_braid_" + suffix, label,
                                 [x for x in o["lines"] if x.get("group") == group], tolerance)
                if q: out.append(q)
        elif o.get("hollow"):
            groups = [(None, oid, o.get("label", oid))]
            if oid == "layer_channels":
                groups = [(0, "layer_channels_am", "AM hollow core channels"),
                          (1, "layer_channels_pl", "PL hollow core channels"),
                          (2, "layer_channels_shared", "Shared hollow channels")]
            for group, pid, label in groups:
                lines = o["lines"] if group is None else [x for x in o["lines"] if x.get("group") == group]
                q = _tube_object(o, pid, label, lines, tolerance, True)
                if q: out.append(q)
        elif oid in ("layer_grooves", "layer_interface", "layer_anchors"):
            # Defensive for scenes that do not attach the usual category/id metadata.
            omitted.append({"id": oid, "level": o.get("level"), "reason": DISPLAY_ONLY[oid]})
    return out, omitted


def _rotation_xyz(degrees):
    x, y, z = [math.radians(float(v)) for v in (degrees or [0, 0, 0])]
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _custom_local(part, tolerance):
    d = part.get("dimensions") or {}
    length = float(d.get("length_mm", 12.0)); diameter = float(d.get("outer_diameter_mm", 2.0))
    radius = diameter / 2.0; ns = _nsec(radius, tolerance); op = part.get("operation", "tube")
    axis = np.array([[0.0, -length / 2, 0.0], [0.0, length / 2, 0.0]])
    if op == "band":
        return box_mesh(np.zeros(3), np.array([float(d.get("width_mm", 3.0)), length, float(d.get("thickness_mm", .6))]))
    if op == "sleeve":
        inner = max(0.01, radius - float(d.get("wall_mm", .3)))
        u = np.tile([1.0, 0.0, 0.0], (2, 1))
        return hollow_tube_mesh(axis, radius, inner, u, nsec=ns)
    if op == "curved_tube":
        P = np.asarray(part.get("path_points_mm") or [[0, -length / 2, 0], [float(d.get("width_mm", 3.0)), 0, 0], [0, length / 2, 0]], float)
        return mesher.tube_mesh(P, radius, ns)
    if op == "channel_array":
        count = max(1, min(64, int(round(float(d.get("count", 3)))))); spacing = float(d.get("spacing_mm", 2.0))
        return mesher.concat([mesher.tube_mesh(axis + [((i - (count - 1) / 2) * spacing), 0, 0], radius, ns) for i in range(count)])
    if op == "lattice":
        count = max(1, min(20, int(round(float(d.get("count", 3)))))); spacing = float(d.get("spacing_mm", 2.0))
        rr = float(d.get("thickness_mm", .6)) / 2.0; pieces = []
        for i in range(count):
            a = (1 if i % 2 else -1) * math.pi / 6; R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
            P = axis @ R.T + [((i - (count - 1) / 2) * spacing / 2), 0, 0]
            pieces.append(mesher.tube_mesh(P, rr, _nsec(rr, tolerance)))
        return mesher.concat(pieces)
    return mesher.tube_mesh(axis, radius, ns)


def _centres(scene, parts):
    centres = {}
    for o in scene["objects"]:
        if o["kind"] == "mesh" and o["V"].size:
            centres[o["id"]] = (np.min(o["V"], 0) + np.max(o["V"], 0)) / 2
        elif o.get("lines"):
            P = np.vstack([x["P"] for x in o["lines"] if len(x["P"])])
            if P.size: centres[o["id"]] = (P.min(0) + P.max(0)) / 2
    # Component aliases used by the browser's builder tree.
    for alias, ids in COMPONENT_ALIASES.items():
        vals = [centres[x] for x in ids if x in centres]
        if vals: centres[alias] = np.mean(vals, axis=0)
    for p in parts:
        if p["V"].size: centres[p["id"]] = (p["V"].min(0) + p["V"].max(0)) / 2
    return centres


def _custom_parts(state, scene, tolerance):
    out, warnings = [], []
    centres = _centres(scene, out)
    for definition in state.get("custom_parts") or []:
        try:
            V, F = _custom_local(definition, tolerance)
            R = _rotation_xyz(definition.get("rotation_deg"))
            base = np.asarray(definition.get("position_mm") or [0, 0, 0], float)
            rel = definition.get("relationship") or {}
            target = rel.get("target") if rel.get("mode", "independent") != "independent" else None
            if target:
                if target in centres: base = centres[target] + base
                else: warnings.append(f'Custom part {definition.get("id")} references unresolved target {target}; model-frame offset was used.')
            V = V @ R.T + base
            q = _mesh_part("custom_" + str(definition.get("id", "part")), definition.get("name") or definition.get("id"),
                           definition.get("level"), V, F, "custom", operation=definition.get("operation"),
                           material=definition.get("material"), relationship=rel, purpose=definition.get("purpose"))
            out.append(q); centres[definition.get("id")] = (q["V"].min(0) + q["V"].max(0)) / 2
        except Exception as exc:
            warnings.append(f'Custom part {definition.get("id", "unnamed")} could not be meshed: {type(exc).__name__}.')
    return out, warnings


def _select(parts, options):
    scope = options.get("scope", "entire")
    if scope == "entire": return parts
    if scope == "custom": return [p for p in parts if p["source"] == "custom"]
    if scope == "levels":
        levels = {int(x) for x in options.get("levels") or []}
        return [p for p in parts if p.get("level") in levels]
    if scope == "components":
        wanted = set(options.get("component_ids") or [])
        expanded = set(wanted)
        for x in wanted: expanded.update(COMPONENT_ALIASES.get(x, []))
        return [p for p in parts if p["id"] in expanded or p["id"].removeprefix("custom_") in expanded]
    raise ValueError("scope must be entire, components, levels, or custom")


def _repair(V, F):
    tri = V[F]
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    keep = area2 > 2e-10
    removed_degenerate = int((~keep).sum()); F = F[keep]
    key = np.sort(F, axis=1)
    _, ix = np.unique(key, axis=0, return_index=True)
    keep_ix = np.sort(ix); removed_duplicate = int(F.shape[0] - keep_ix.size); F = F[keep_ix]
    return V, F, {"removed_degenerate_faces": removed_degenerate, "removed_duplicate_faces": removed_duplicate,
                  "dimensions_changed": False, "openings_filled": False, "vertices_merged": 0}


def _validate_part(part, repair=False):
    V, F = part["V"], part["F"]
    checks = []
    finite = bool(np.isfinite(V).all())
    checks.append({"check": "finite coordinates", "status": "passed" if finite else "failed"})
    indices = bool(F.ndim == 2 and F.shape[1:] == (3,) and (not F.size or (F.min() >= 0 and F.max() < len(V))))
    checks.append({"check": "triangle indices", "status": "passed" if indices else "failed"})
    changes = None
    if finite and indices and repair: V, F, changes = _repair(V, F)
    if not finite or not indices:
        return V, F, checks, changes, True
    audit = mesher.audit(V, F)
    for name, ok, fail in (
        ("non-empty geometry", F.shape[0] > 0, "failed"),
        ("degenerate triangles", audit["degenerate_faces"] == 0, "warning"),
        ("manifold edges", audit["nonmanifold_edges"] == 0, "warning"),
        ("watertight material boundary", audit["closed"], "warning"),
        ("consistent winding", audit["orientation_consistent"], "warning"),
    ):
        checks.append({"check": name, "status": "passed" if ok else fail,
                       "detail": audit.get({"degenerate triangles": "degenerate_faces", "manifold edges": "nonmanifold_edges"}.get(name, ""))})
    key = np.sort(F, axis=1)
    duplicate = int(F.shape[0] - np.unique(key, axis=0).shape[0])
    checks.append({"check": "duplicate faces", "status": "passed" if duplicate == 0 else "warning", "detail": duplicate})
    shells, _ = mesher.components(F, V.shape[0])
    checks.append({"check": "connected shells", "status": "passed" if shells == 1 else "warning", "detail": int(shells),
                   "note": "Multiple shells can be intentional for yarns, channels, lattices, or assemblies."})
    checks.extend([{"check": "self-intersections", "status": "not evaluated"},
                   {"check": "unexpected component overlap", "status": "not evaluated"}])
    return V, F, checks, changes, F.shape[0] == 0


def _process_profile(state, selected, profile):
    if not profile:
        return {"status": "not evaluated", "message": "Geometry exported. Manufacturing suitability has not been assessed."}
    limits = {k: float(v) for k, v in profile.items() if v not in (None, "")}
    if not limits:
        return {"status": "not evaluated", "message": "Geometry exported. Manufacturing suitability has not been assessed."}
    walls, openings = [], []
    levels = {p.get("level") for p in selected}
    params = state.get("params") or {}
    if 5 in levels and params.get("sheath.wall_mm") is not None: walls.append(("sheath wall", float(params["sheath.wall_mm"])))
    if 3 in levels and params.get("ch.wall_um") is not None: walls.append(("channel wall", float(params["ch.wall_um"]) / 1000.0))
    if 3 in levels and params.get("ch.lumen_um") is not None: openings.append(("channel lumen", float(params["ch.lumen_um"]) / 1000.0))
    for p in selected:
        if p["source"] == "custom" and p.get("operation") == "sleeve":
            # Exact custom dimensions remain in design.json; arbitrary mesh wall thickness is not inferred.
            pass
    checks = []
    if "minimum_wall_mm" in limits:
        checks += [{"feature": n, "value_mm": v, "limit_mm": limits["minimum_wall_mm"],
                    "status": "passed" if v >= limits["minimum_wall_mm"] else "warning"} for n, v in walls]
        if not walls: checks.append({"feature": "minimum wall thickness", "status": "not evaluated"})
    if "minimum_opening_mm" in limits:
        checks += [{"feature": n, "value_mm": v, "limit_mm": limits["minimum_opening_mm"],
                    "status": "passed" if v >= limits["minimum_opening_mm"] else "warning"} for n, v in openings]
        if not openings: checks.append({"feature": "minimum opening size", "status": "not evaluated"})
    return {"status": "evaluated where exact parametric dimensions exist", "profile": limits, "checks": checks,
            "message": "These comparisons are not proof of printability, implant readiness, or clinical validation."}


def build_export(state, scene, options=None, preview=False):
    """Return (summary, zip bytes or None).  State/scene are treated as immutable snapshots."""
    options = dict(options or {})
    detail = options.get("detail", "standard")
    tolerance = float(options.get("tolerance_mm") or DETAIL_TOLERANCE_MM.get(detail, DETAIL_TOLERANCE_MM["standard"]))
    if not 0.005 <= tolerance <= 0.5: raise ValueError("custom tolerance must be between 0.005 and 0.5 mm")
    mode = options.get("mode", "combined")
    if mode == "fused": raise NotImplementedError("Fused-solid export is unavailable: this engine has no validated Boolean-union kernel. Use combined assembly or separate parts.")
    if mode not in ("combined", "separate"): raise ValueError("mode must be combined, separate, or fused")
    scene_parts, omitted = _scene_parts(scene, tolerance)
    custom_parts, custom_warnings = _custom_parts(state, scene, tolerance)
    selected = _select(scene_parts + custom_parts, options)
    if not selected: raise ValueError("The selection contains no physical exportable geometry.")
    repair = bool(options.get("repair")); processed = []; blocked = False; repair_log = []
    for p in selected:
        V, F, checks, changes, bad = _validate_part(p, repair)
        q = dict(p, V=V, F=F, validation=checks)
        processed.append(q); blocked |= bad
        if changes: repair_log.append({"component": p["id"], **changes})
    allV = np.vstack([p["V"] for p in processed if p["V"].size])
    lo, hi = allV.min(0), allV.max(0); dims = hi - lo
    triangles = sum(int(p["F"].shape[0]) for p in processed)
    selected_levels = {p.get("level") for p in processed}
    omitted_selected = [x for x in omitted if options.get("scope", "entire") == "entire" or x.get("level") in selected_levels]
    validation = {"components": [{"id": p["id"], "checks": p["validation"]} for p in processed],
                  "blocked": blocked, "repair": repair_log,
                  "assembly": [{"check": "separate shells in one STL", "status": "warning" if mode == "combined" and len(processed) > 1 else "passed",
                                "note": "Combined assembly preserves distinct shells; it is not a Boolean union."},
                               {"check": "manufacturing suitability", "status": "not evaluated"}]}
    summary = {"ok": not blocked, "design_hash": scene["design_hash"], "geometry_hash": scene["geometry_hash"],
               "revision": options.get("revision"), "snapshot": True, "units": "mm", "detail": detail,
               "tolerance_mm": tolerance, "mode": mode, "triangle_count": triangles,
               "estimated_binary_stl_bytes": 84 + 50 * triangles,
               "bounds_mm": {"min": lo.tolist(), "max": hi.tolist(), "dimensions": dims.tolist()},
               "components": [{k: p.get(k) for k in ("id", "label", "level", "source", "category", "operation", "material")} | {"triangles": int(p["F"].shape[0])} for p in processed],
               "omitted_display_only_features": omitted_selected,
               "warnings": custom_warnings,
               "validation": validation,
               "manufacturing": _process_profile(state, processed, options.get("process_profile")),
               "cad_note": "STL is a triangulated mesh; it does not preserve sliders, construction history, material assignments, or editable CAD features.",
               "units_note": "Exported in millimeters. Select millimeters when importing into your slicer or CAD software."}
    if preview: return summary, None
    if blocked: raise ValueError("Export blocked: empty or nonfinite/invalid-index geometry is present.")
    files = {}
    if mode == "combined":
        V, F = mesher.concat([(p["V"], p["F"]) for p in processed])
        files["stl/scaffold_assembly.stl"] = stl_bytes(V, F, "ACL scaffold assembly; units mm; distinct shells")
    else:
        for p in processed:
            V = p["V"].copy()
            if options.get("center_parts") and V.size: V -= (V.min(0) + V.max(0)) / 2
            files["stl/%s.stl" % p["id"]] = stl_bytes(V, p["F"], "ACL scaffold %s; units mm" % p["id"])
    manifest = dict(summary)
    manifest["components"] = summary["components"]
    files["design.json"] = json.dumps(_jsonable(state), indent=2).encode()
    files["component_manifest.json"] = json.dumps(_jsonable(manifest), indent=2).encode()
    files["validation.json"] = json.dumps(_jsonable(validation), indent=2).encode()
    files["README.txt"] = (summary["units_note"] + "\n" + summary["cad_note"] + "\n"
        + "Multiple shells in one STL are not fused. Manufacturing suitability is not implied.\n"
        + "Native ACL, bones, MRI, annotations, labels, viewer helpers, clipping, exploded offsets, and magnified display details are excluded.\n").encode()
    manifest["files"] = {k: hashlib.sha256(v).hexdigest()[:16] for k, v in files.items()}
    files["component_manifest.json"] = json.dumps(_jsonable(manifest), indent=2).encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(files): z.writestr(name, files[name])
    return manifest, buf.getvalue()
