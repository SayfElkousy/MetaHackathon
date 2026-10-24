"""Meshing, validation and export of scaffold candidates.

Validation covers: watertightness / manifoldness, winding consistency, self-intersections
(exact segment-triangle tests on AABB-overlapping non-adjacent faces), connected components,
minimum solid feature thickness, channel openness, porosity, bone collision/clearance and
remnant interference. Passing these checks says the file is a valid research mesh,
NOT that the design is implant-ready.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from ..annotation.schema import Label
from ..imaging.coords import apply_affine
from ..provenance import Provenance, utcnow, write_json
from .design import ScaffoldCandidate


def occupancy_to_mesh(occ: np.ndarray, grid_affine: np.ndarray):
    import trimesh
    from skimage import measure

    m = np.pad(occ.astype(np.float32), 1)
    verts, faces, normals, _ = measure.marching_cubes(m, 0.5, method="lewiner")
    verts = apply_affine(grid_affine, verts - 1)
    mesh = trimesh.Trimesh(verts, faces, process=True)
    mesh.fix_normals()
    return mesh


# ----------------------------------------------------------------- self-intersection


def _seg_tri(p0, p1, a, b, c, eps=1e-9):
    """Vectorised segment-triangle intersection (Moller-Trumbore), excluding touching at endpoints."""
    d = p1 - p0
    e1, e2 = b - a, c - a
    h = np.cross(d, e2)
    det = (e1 * h).sum(-1)
    ok = np.abs(det) > eps
    inv = np.where(ok, 1.0 / np.where(ok, det, 1), 0)
    s = p0 - a
    u = inv * (s * h).sum(-1)
    q = np.cross(s, e1)
    v = inv * (d * q).sum(-1)
    t = inv * (e2 * q).sum(-1)
    tol = 1e-6
    return ok & (u > tol) & (v > tol) & (u + v < 1 - tol) & (t > tol) & (t < 1 - tol)


def self_intersections(mesh, max_pairs: int = 2_000_000) -> dict:
    """Count intersecting non-adjacent triangle pairs using an AABB broad phase (rtree)."""
    from rtree import index

    V, F = mesh.vertices, mesh.faces
    tri = V[F]
    lo, hi = tri.min(1), tri.max(1)
    p = index.Property()
    p.dimension = 3
    idx = index.Index(((i, (*lo[i], *hi[i]), None) for i in range(len(F))), properties=p)
    A, Bp = [], []
    for i in range(len(F)):
        for j in idx.intersection((*lo[i], *hi[i])):
            if j > i:
                A.append(i)
                Bp.append(j)
        if len(A) > max_pairs:
            return {"checked": False, "reason": "too many candidate pairs"}
    A, Bp = np.array(A, int), np.array(Bp, int)
    if len(A) == 0:
        return {"checked": True, "intersecting_pairs": 0}
    shared = (F[A][:, :, None] == F[Bp][:, None, :]).any(-1).any(-1)
    A, Bp = A[~shared], Bp[~shared]
    hit = np.zeros(len(A), bool)
    for (s, t) in ((0, 1), (1, 2), (2, 0)):
        hit |= _seg_tri(tri[A, s], tri[A, t], tri[Bp, 0], tri[Bp, 1], tri[Bp, 2])
        hit |= _seg_tri(tri[Bp, s], tri[Bp, t], tri[A, 0], tri[A, 1], tri[A, 2])
    return {"checked": True, "candidate_pairs": int(len(A)), "intersecting_pairs": int(hit.sum())}


# ----------------------------------------------------------------- validation


def min_feature_thickness(occ: np.ndarray, res: float) -> float:
    """2 x minimum inscribed-sphere radius along the solid's skeleton (mm, voxel-limited)."""
    from skimage.morphology import skeletonize

    if not occ.any():
        return float("nan")
    edt = ndimage.distance_transform_edt(np.pad(occ, 1)) [1:-1, 1:-1, 1:-1] * res
    sk = skeletonize(occ)
    vals = edt[sk > 0]
    return float(2 * vals.min()) if vals.size else float("nan")


def validate_candidate(cand: ScaffoldCandidate, mesh, labels: np.ndarray, label_affine: np.ndarray) -> dict:
    res = cand.params.grid_res_mm
    rep: dict = {"timestamp": utcnow()}
    rep["mesh"] = {
        "vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight), "winding_consistent": bool(mesh.is_winding_consistent),
        "is_volume": bool(mesh.is_volume), "euler_number": int(mesh.euler_number),
        "volume_mm3": float(mesh.volume) if mesh.is_volume else None, "area_mm2": float(mesh.area),
    }
    try:
        import manifold3d

        mf = manifold3d.Manifold(manifold3d.Mesh(np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.uint32)))
        rep["mesh"]["manifold3d_status"] = str(mf.status())
    except Exception as e:  # pragma: no cover
        rep["mesh"]["manifold3d_status"] = f"error: {e}"
    rep["mesh"]["self_intersections"] = self_intersections(mesh)
    parts = mesh.split(only_watertight=False)
    rep["mesh"]["connected_components"] = int(len(parts))
    lab, n = ndimage.label(cand.occupancy)
    rep["voxel_components"] = int(n)
    env_vol = float(cand.envelope.sum() * res ** 3)
    solid = float(cand.occupancy.sum() * res ** 3)
    rep["porosity"] = {"envelope_mm3": env_vol, "solid_mm3": solid,
                       "void_fraction_within_envelope": 1 - solid / env_vol if env_vol else None,
                       "note": "includes channels, remnant pockets and clearance cut-outs"}
    rep["min_solid_feature_mm"] = min_feature_thickness(cand.occupancy, res)
    p = cand.params
    rep["channels"] = {"n": len(cand.channels_polylines), "design_diameter_mm": p.channel_diameter_mm, "pattern": p.alignment_pattern,
                       "twist_deg": p.twist_deg if p.alignment_pattern == "twisted" else 0.0}
    if cand.channels_polylines:
        # openness: each channel's void must connect to outside the envelope at both ends
        void = cand.envelope & ~cand.occupancy
        rep["channels"]["void_components"] = int(ndimage.label(void)[1])
        # resolvability on the grid
        rep["channels"]["grid_voxels_across"] = float(p.channel_diameter_mm / res)
        cl = np.concatenate([c[None] for c in cand.channels_polylines])
        dmin = np.inf
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                dmin = min(dmin, float(np.linalg.norm(cl[i] - cl[j], axis=1).min()))
        rep["channels"]["min_center_distance_mm"] = None if not np.isfinite(dmin) else dmin
        rep["channels"]["min_wall_between_channels_mm"] = None if not np.isfinite(dmin) else dmin - p.channel_diameter_mm
    # bone collision / clearance and remnant interference, evaluated at mesh vertices
    sp = np.linalg.norm(label_affine[:3, :3], axis=0)
    vox = apply_affine(np.linalg.inv(label_affine), mesh.vertices)
    bone = np.isin(labels, [Label.FEMUR, Label.TIBIA])
    if bone.any():
        d_out = ndimage.distance_transform_edt(~bone, sampling=sp)
        d_in = ndimage.distance_transform_edt(bone, sampling=sp)
        sdist = d_out - d_in  # >0 outside bone
        s = ndimage.map_coordinates(sdist, vox.T, order=1, mode="nearest")
        rep["bone"] = {"min_clearance_mm": float(s.min()), "vertices_inside_bone": int((s < 0).sum()),
                       "note": "voxel-resolution estimate from segmentation; inherits segmentation error"}
    rem = np.isin(labels, [Label.ACL_REMNANT_FEMORAL, Label.ACL_REMNANT_TIBIAL, Label.ACL_FRAGMENT_OTHER, Label.ACL_INTACT])
    if rem.any():
        G = cand.grid_affine
        gi = np.indices(cand.occupancy.shape).reshape(3, -1).T[cand.occupancy.ravel()]
        Xw = apply_affine(G, gi.astype(float))
        rv = np.round(apply_affine(np.linalg.inv(label_affine), Xw)).astype(int)
        inb = np.all((rv >= 0) & (rv < np.array(labels.shape)), axis=1)
        overlap = np.zeros(len(rv), bool)
        overlap[inb] = rem[tuple(rv[inb].T)]
        rep["remnant"] = {"overlap_mm3": float(overlap.sum() * res ** 3)}
    # pass/fail summary (geometric validity only)
    fails = []
    m = rep["mesh"]
    if not (m["watertight"] and m["winding_consistent"] and m["is_volume"]):
        fails.append("mesh not a closed consistently-oriented volume")
    si = m["self_intersections"]
    if not si.get("checked") or si.get("intersecting_pairs", 1) > 0:
        fails.append("self-intersections present or not checked")
    if m["connected_components"] != 1:
        fails.append(f"{m['connected_components']} disconnected mesh components")
    if p.min_feature_mm and rep["min_solid_feature_mm"] < p.min_feature_mm:
        fails.append(f"min solid feature {rep['min_solid_feature_mm']:.2f} mm < fabrication limit {p.min_feature_mm} mm")
    if rep.get("bone", {}).get("vertices_inside_bone", 0) > 0:
        fails.append("scaffold intersects segmented bone")
    if p.min_wall_mm and rep["channels"].get("min_wall_between_channels_mm") is not None and rep["channels"]["min_wall_between_channels_mm"] < p.min_wall_mm - res:
        fails.append("channel wall below min_wall_mm")
    rep["geometric_checks_passed"] = not fails
    rep["failures"] = fails
    rep["disclaimer"] = "Geometric validity of a research mesh only. Not evidence of implant readiness, mechanical adequacy, or biological performance."
    return rep


# ----------------------------------------------------------------- export


def export_candidate(cand: ScaffoldCandidate, mesh, validation: dict, out_dir, *, source: dict, gate: dict, provenance: Provenance,
                     uncertainty: dict | None = None, assumptions: list | None = None, synthetic: bool = False) -> dict:
    """Write STL + polylines + metadata JSON. Caller must have obtained gate['allowed'] == True."""
    if not gate.get("allowed"):
        raise PermissionError("export blocked: " + "; ".join(gate.get("reasons", [])))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = "SYNTHETIC_" if synthetic else ""
    stl = out / f"{tag}scaffold_candidate.stl"
    mesh.export(stl)
    pl = out / f"{tag}hypothesized_guidance_pathways.json"
    pl.write_text(json.dumps({"label": "hypothesized internal guidance architecture - NOT recovered original fibres",
                              "units": "mm", "frame": "RAS+ world of injured image", "polylines": [p.tolist() for p in cand.channels_polylines]}))
    meta = {
        "WARNING": ("SYNTHETIC DEMONSTRATION - no clinical meaning. " if synthetic else "")
        + "Investigational research geometry. Not implant-ready. Not validated for any clinical use.",
        "units": "mm",
        "coordinate_frame": "RAS+ world coordinates of the injured-knee source image (x->patient Right, y->Anterior, z->Superior)",
        "to_LPS": "multiply x and y by -1 (DICOM patient coordinates)",
        "source": source,
        "files": {"mesh_stl": stl.name, "guidance_pathways": pl.name},
        "evidence_levels": cand.evidence,
        "span_t": cand.span_t,
        "parameters": {k: v for k, v in cand.params.__dict__.items()},
        "stations": cand.stations.to_dict(),
        "design_notes": cand.notes,
        "validation": validation,
        "uncertainty": uncertainty or {},
        "assumptions": assumptions or [],
        "export_gate": gate,
        "review_status": provenance.review_status.value,
        "provenance": provenance.to_dict(),
        "not_established": ["mechanical adequacy", "biological healing", "degradation behaviour", "clinical outcome / reinjury",
                            "fibre correspondence", "manufacturability on a specific process"],
    }
    write_json(out / f"{tag}scaffold_candidate.meta.json", meta)
    return {"stl": str(stl), "meta": str(out / f"{tag}scaffold_candidate.meta.json"), "pathways": str(pl)}
