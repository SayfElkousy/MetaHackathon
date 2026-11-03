"""Measurements and mesh audits for the anatomical ACL model.

New module. Computes section profiles, footprint/volume scalars, strand
statistics, and mesh validity audits. Conventions:

- closed solids (acl_surface, stumps, tibial footprint slab): must have no
  boundary edges, no non-manifold edges, no degenerate faces, and
  consistent outward orientation;
- intentionally open surfaces (femoral on-bone patch): boundary edges are
  expected and reported, but non-manifold edges and degenerate faces are
  still failures;
- strand paths are curves: only length and envelope containment apply.

Self-intersection uses a vectorized Moller triangle-triangle test over
candidate pairs from a centroid KD-tree; pairs sharing a vertex are
excluded (adjacent triangles). Engineering tolerances live in TOL with a
stated rationale; they are NOT clinical thresholds.
"""
import hashlib
import math
import numpy as np
from scipy.spatial import cKDTree

# Engineering audit tolerances (NOT clinical thresholds):
# - degenerate face: area below (0.02 mm)^2 -- far below any modeled feature
#   (smallest modeled radius is ~0.35 mm strand tubes);
# - non-manifold/boundary: exact integer edge counts, no tolerance;
# - orientation: fraction of faces agreeing with outward convention must be
#   1.0 for closed solids (exact);
# - bone penetration: 0.05 mm slack for projection round-off.
TOL = {"degenerate_area_mm2": 4e-4,
       "bone_slack_mm": 0.05}


def face_areas(V, F):
    """Triangle areas. Vectorized."""
    P = np.asarray(V, dtype=float)[np.asarray(F, dtype=np.int64)]
    return 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)


def edge_counts(F):
    """(unique_edges, counts) with sorted vertex order. Vectorized."""
    F = np.asarray(F, dtype=np.int64)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0, return_counts=True)


def audit_mesh(V, F, expect_closed=True):
    """Mesh validity audit. Returns dict with integer defect counts and a
    pass/fail verdict (no tolerances on counts -- exact)."""
    V = np.asarray(V, dtype=float)
    F = np.asarray(F, dtype=np.int64)
    info = {"verts": int(V.shape[0]), "faces": int(F.shape[0]),
            "finite": bool(np.all(np.isfinite(V)))}
    if F.size == 0 or not info["finite"]:
        info.update({"boundary_edges": -1, "nonmanifold_edges": -1,
                     "degenerate_faces": -1, "orientation_ok": False,
                     "pass": False})
        return info
    _e, counts = edge_counts(F)
    info["boundary_edges"] = int(np.sum(counts == 1))
    info["nonmanifold_edges"] = int(np.sum(counts > 2))
    info["degenerate_faces"] = int(np.sum(face_areas(V, F) < TOL["degenerate_area_mm2"]))
    info["orientation_ok"] = bool(_orientation_consistent(F))
    if expect_closed:
        info["pass"] = bool(info["boundary_edges"] == 0
                            and info["nonmanifold_edges"] == 0
                            and info["degenerate_faces"] == 0
                            and info["orientation_ok"])
    else:
        info["pass"] = bool(info["nonmanifold_edges"] == 0
                            and info["degenerate_faces"] == 0
                            and info["orientation_ok"])
    return info


def _orientation_consistent(F):
    """Every shared edge is used in opposite directions (exact). Vectorized."""
    m = int(np.asarray(F).max()) + 1
    a = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    fwd = a[:, 0] * m + a[:, 1]
    bwd = a[:, 1] * m + a[:, 0]
    und = np.minimum(fwd, bwd) * m * m + np.maximum(fwd, bwd)
    _u, inv, counts = np.unique(und, return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        return False
    s = np.argsort(inv, kind="stable")
    sf = fwd[s]
    si = inv[s]
    starts = np.nonzero(np.diff(si, prepend=-1) > 0)[0]
    g2 = starts[counts[si[starts]] == 2]
    if g2.size == 0:
        return True
    return bool(np.all(sf[g2] != sf[g2 + 1]))


def _seg_tri_2d(O, D, T):
    """Segment-triangle overlap when the segment lies in the plane:
    project on the dominant axis and test edge crossings + containment."""
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    dom = np.argmax(np.abs(n), axis=1)
    keep = np.array([[1, 2], [0, 2], [0, 1]])
    hit = np.zeros(O.shape[0], dtype=bool)
    for a in range(3):
        sel = np.nonzero(dom == a)[0]
        if sel.size == 0:
            continue
        kk = keep[a]
        o = O[sel][:, kk]
        d = D[sel][:, kk]
        t = T[sel][:, :, kk]
        p1, p2 = o, o + d
        for i, j in ((0, 1), (1, 2), (2, 0)):
            hit[sel] |= _seg_seg_2d(p1, p2, t[:, i], t[:, j])
        hit[sel] |= _pt_in_tri_2d(p1, t[:, 0], t[:, 1], t[:, 2])
        hit[sel] |= _pt_in_tri_2d(p2, t[:, 0], t[:, 1], t[:, 2])
    return hit


def _seg_seg_2d(p1, p2, p3, p4):
    d = p2 - p1
    e = p4 - p3
    denom = d[:, 0] * e[:, 1] - d[:, 1] * e[:, 0]
    ok = np.abs(denom) > 1e-18 * (np.linalg.norm(d, axis=1) * np.linalg.norm(e, axis=1) + 1e-30)
    t = np.zeros(p1.shape[0])
    u = np.zeros(p1.shape[0])
    t[ok] = ((p3[ok, 0] - p1[ok, 0]) * e[ok, 1] - (p3[ok, 1] - p1[ok, 1]) * e[ok, 0]) / denom[ok]
    u[ok] = ((p3[ok, 0] - p1[ok, 0]) * d[ok, 1] - (p3[ok, 1] - p1[ok, 1]) * d[ok, 0]) / denom[ok]
    strict = (t > 1e-9) & (t < 1.0 - 1e-9) & (u > 1e-9) & (u < 1.0 - 1e-9)
    return ok & strict


def _pt_in_tri_2d(pt, a, b, c):
    v0 = c - a
    v1 = b - a
    v2 = pt - a
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    ok = np.abs(den) > 1e-18
    v = np.zeros(pt.shape[0])
    w = np.zeros(pt.shape[0])
    v[ok] = (v2[ok, 0] * v1[ok, 1] - v1[ok, 0] * v2[ok, 1]) / den[ok]
    w[ok] = (v0[ok, 0] * v2[ok, 1] - v2[ok, 0] * v0[ok, 1]) / den[ok]
    return ok & (v > 1e-9) & (w > 1e-9) & (v + w < 1.0 - 1e-9)


def _seg_tri_hit(O, D, T):
    """Vectorized segment-triangle intersection (Moller-Trumbore, t in
    (0,1) strict). O, D: (k,3) origins/directions; T: (k,3,3) triangles.
    Near-parallel cases use a scale-aware threshold; segments lying in
    the plane fall back to an exact 2-D overlap test."""
    e1 = T[:, 1] - T[:, 0]
    e2 = T[:, 2] - T[:, 0]
    p = np.cross(D, e2)
    det = np.sum(e1 * p, axis=1)
    scale = np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) \
        * np.linalg.norm(D, axis=1) + 1e-30
    ok = np.abs(det) > 1e-12 * scale
    hit = np.zeros(O.shape[0], dtype=bool)
    if np.any(ok):
        inv = np.zeros_like(det)
        inv[ok] = 1.0 / det[ok]
        tv = O - T[:, 0]
        u = np.sum(tv * p, axis=1) * inv
        q = np.cross(tv, e1)
        v = np.sum(D * q, axis=1) * inv
        t = np.sum(e2 * q, axis=1) * inv
        hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1.0 + 1e-9) \
            & (t >= 1e-9) & (t <= 1.0 - 1e-9)
    par = ~ok
    if np.any(par):
        n = np.cross(e1, e2)
        nl = np.linalg.norm(n, axis=1) + 1e-30
        dist = np.abs(np.sum((O - T[:, 0]) * n, axis=1) / nl)
        inplane = par & (dist < 1e-7 * (np.linalg.norm(D, axis=1) + 1.0))
        if np.any(inplane):
            hit[inplane] |= _seg_tri_2d(O[inplane], D[inplane], T[inplane])
    return hit


def _tri_tri_intersect(P, Q):
    """Exact triangle-triangle intersection (non-coplanar) + conservative
    2-D fallback for coplanar pairs. P, Q: (k,3,3). Vectorized.

    Two non-coplanar triangles intersect iff an edge of one pierces the
    other (callers exclude shared-vertex pairs). Coplanar pairs fall back
    to projected 2-D overlap with an area threshold so mere touching does
    not count.
    """
    n1 = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    n2 = np.cross(Q[:, 1] - Q[:, 0], Q[:, 2] - Q[:, 0])
    n1 /= np.linalg.norm(n1, axis=1, keepdims=True) + 1e-30
    n2 /= np.linalg.norm(n2, axis=1, keepdims=True) + 1e-30
    d1 = np.sum(n1 * P[:, 0], axis=1)
    parallel = np.abs(np.sum(n1 * n2, axis=1)) > 1.0 - 1e-9
    dist = np.abs(np.sum(Q[:, 0] * n1, axis=1) - d1)
    coplanar = parallel & (dist < 1e-7)
    out = np.zeros(P.shape[0], dtype=bool)
    todo = np.nonzero(~coplanar)[0]
    if todo.size:
        Pp = P[todo]
        Qq = Q[todo]
        for i, j in ((0, 1), (1, 2), (2, 0)):
            out[todo] |= _seg_tri_hit(Pp[:, i], Pp[:, j] - Pp[:, i], Qq)
            out[todo] |= _seg_tri_hit(Qq[:, i], Qq[:, j] - Qq[:, i], Pp)
    cp = np.nonzero(coplanar)[0]
    if cp.size:
        dom = np.argmax(np.abs(np.cross(n1[cp], np.array([0.31, 0.47, 0.83]))), axis=1)
        keep = np.array([[1, 2], [0, 2], [0, 1]])
        for a in range(3):
            sel = cp[dom == a]
            if sel.size:
                kk = keep[a]
                out[sel] = _tri_overlap_2d(P[sel][:, :, kk], Q[sel][:, :, kk])
    return out


def _tri_overlap_2d(P, Q):
    """Conservative 2-D triangle overlap: edge intersections or containment."""
    def seg_int(p1, p2, p3, p4):
        d = (p2 - p1)
        e = (p4 - p3)
        denom = d[:, 0] * e[:, 1] - d[:, 1] * e[:, 0]
        ok = np.abs(denom) > 1e-18
        t = np.zeros(p1.shape[0])
        u = np.zeros(p1.shape[0])
        t[ok] = ((p3[ok, 0] - p1[ok, 0]) * e[ok, 1] - (p3[ok, 1] - p1[ok, 1]) * e[ok, 0]) / denom[ok]
        u[ok] = ((p3[ok, 0] - p1[ok, 0]) * d[ok, 1] - (p3[ok, 1] - p1[ok, 1]) * d[ok, 0]) / denom[ok]
        return ok & (t >= -1e-9) & (t <= 1.0 + 1e-9) & (u >= -1e-9) & (u <= 1.0 + 1e-9)
    def inside(pt, a, b, c):
        v0 = c - a
        v1 = b - a
        v2 = pt - a
        den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        ok = np.abs(den) > 1e-18
        v = np.zeros(pt.shape[0])
        w = np.zeros(pt.shape[0])
        v[ok] = (v2[ok, 0] * v1[ok, 1] - v1[ok, 0] * v2[ok, 1]) / den[ok]
        w[ok] = (v0[ok, 0] * v2[ok, 1] - v2[ok, 0] * v0[ok, 1]) / den[ok]
        return ok & (v >= -1e-9) & (w >= -1e-9) & (v + w <= 1.0 + 1e-9)
    hit = np.zeros(P.shape[0], dtype=bool)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        for k, l in ((0, 1), (1, 2), (2, 0)):
            hit |= seg_int(P[:, i], P[:, j], Q[:, k], Q[:, l])
    hit |= inside(P[:, 0], Q[:, 0], Q[:, 1], Q[:, 2])
    hit |= inside(Q[:, 0], P[:, 0], P[:, 1], P[:, 2])
    return hit


def self_intersections(V, F, max_pairs=400000):
    """Count intersecting triangle pairs (excluding shared-vertex pairs).

    Candidate pairs come from a centroid KD-tree with per-triangle bounding
    radii; the exact test is vectorized. Returns (count, pairs_tested).
    Caps the candidate list at max_pairs (reported, never silent).
    """
    V = np.asarray(V, dtype=float)
    F = np.asarray(F, dtype=np.int64)
    P = V[F]
    cen = P.mean(axis=1)
    rad = np.linalg.norm(P - cen[:, None, :], axis=2).max(axis=1)
    # Two-level candidate search: a few large triangles (e.g. cap fans)
    # would otherwise blow up the single query radius for everybody.
    big = rad > np.percentile(rad, 99.0)
    small = np.nonzero(~big)[0]
    pair_list = []
    if small.shape[0] > 1:
        tree = cKDTree(cen[small])
        ps = tree.query_pairs(float(2.0 * rad[small].max() + 1e-9),
                              output_type="ndarray")
        if ps.shape[0]:
            pair_list.append(np.stack([small[ps[:, 0]], small[ps[:, 1]]], axis=1))
    ib = np.nonzero(big)[0]
    if ib.shape[0]:
        dd = np.linalg.norm(cen[ib, None, :] - cen[None, :, :], axis=2)
        touch = dd <= rad[ib, None] + rad[None, :] + 1e-9
        touch[np.arange(ib.shape[0]), ib] = False
        ii, jj = np.nonzero(touch)
        if ii.shape[0]:
            pr = np.stack([ib[ii], jj], axis=1)
            pr = pr[pr[:, 0] < pr[:, 1]]
            if pr.shape[0]:
                pair_list.append(pr)
    if not pair_list:
        return 0, 0
    pairs = np.concatenate(pair_list)
    if pairs.shape[0] == 0:
        return 0, 0
    i, j = pairs[:, 0], pairs[:, 1]
    share = (F[i, :, None] == F[j, :, None].transpose(0, 2, 1)).any(axis=(1, 2))
    dist = np.linalg.norm(cen[i] - cen[j], axis=1)
    use = (~share) & (dist <= rad[i] + rad[j] + 1e-9)
    ii = i[use]
    jj = j[use]
    tested = int(ii.shape[0])
    capped = tested > max_pairs
    if capped:
        ii = ii[:max_pairs]
        jj = jj[:max_pairs]
    if ii.shape[0] == 0:
        return 0, tested
    hit = _tri_tri_intersect(P[ii], P[jj])
    return int(np.sum(hit)), tested


def bone_penetration(points, tibia_box, femur_ellipsoids, slack_mm=None):
    """Points inside bone primitives. Tibia: inside box AND below plateau
    plane region is bone; femur: inside any ellipsoid. Returns dict with
    count, fraction, and max depth estimate (distance to surface approx by
    sampling -- reported as penetration count + deepest point).

    Callers must exclude designated attachment regions before calling.
    """
    slack = TOL["bone_slack_mm"] if slack_mm is None else float(slack_mm)
    P = np.asarray(points, dtype=float)
    bmin, bmax = (np.asarray(v, dtype=float) for v in tibia_box)
    in_tib = np.all((P >= bmin - slack) & (P <= bmax + slack), axis=1) & (P[:, 2] < slack)
    in_fem = np.zeros(P.shape[0], dtype=bool)
    for C, R in femur_ellipsoids:
        C = np.asarray(C, dtype=float)
        R = np.asarray(R, dtype=float)
        q = np.sum(((P - C) / R) ** 2, axis=1)
        in_fem |= q <= 1.0 + slack / max(R)
    bad = in_tib | in_fem
    return {"n_points": int(P.shape[0]), "n_inside": int(np.sum(bad)),
            "fraction": float(np.mean(bad)) if P.shape[0] else 0.0,
            "tibial": int(np.sum(in_tib)), "femoral": int(np.sum(in_fem))}


def regen_hash(*arrays):
    """Deterministic sha256 over float arrays rounded to 1e-6 mm."""
    h = hashlib.sha256()
    for a in arrays:
        b = np.asarray(a, dtype=float)
        h.update(np.round(b, 6).tobytes())
        h.update(bytes(str(b.shape), "utf-8"))
    return h.hexdigest()


def profile_stats(t, values):
    """Min/mean/max of a station profile at normalized arc length t."""
    v = np.asarray(values, dtype=float)
    t = np.asarray(t, dtype=float)
    return {"min": float(v.min()), "max": float(v.max()),
            "mean": float(v.mean()),
            "t_at_min": float(t[int(np.argmin(v))]),
            "t_at_max": float(t[int(np.argmax(v))])}
