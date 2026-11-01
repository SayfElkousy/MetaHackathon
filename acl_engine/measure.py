"""Measurements used by the N2 CLI: curvature and all-pairs strand clearance.

Written by Claude to match the interface Muse's N2 code expects.
"""
import math
import numpy as np
from scipy.spatial import cKDTree

from acl_engine.paths import straight_pos_vel_acc


def curvature_max_straight(ss, pf, a, b, q, phase_offset, mirror):
    """Max centerline curvature (1/mm) over all three strands, analytic derivatives."""
    phi = pf.phi_array(ss)
    rate = pf.rate_array(ss)
    rate2 = pf.rate2_array(ss)
    kmax = 0.0
    for i in range(3):
        _p, vel, acc = straight_pos_vel_acc(ss, phi, rate, rate2, a, b, q, phase_offset, mirror, i)
        cr = np.linalg.norm(np.cross(vel, acc), axis=1)
        sp = np.maximum(np.linalg.norm(vel, axis=1) ** 3, 1e-30)
        kmax = max(kmax, float((cr / sp).max()))
    return kmax


def min_surface_gap(centers, s, r, d):
    """Smallest surface-to-surface gap between strands (and nonlocal self-gap),
    checking ALL parameter pairs, not just equal-s samples.
    centers: list of 3 arrays (n, 3) sampled at parameters s."""
    s = np.asarray(s, dtype=float)
    pts = np.vstack(centers)
    sid = np.concatenate([np.full(len(c), k) for k, c in enumerate(centers)])
    sval = np.concatenate([s for _ in centers])
    # arc length along each strand, for the same-strand exclusion
    arcs = []
    for c in centers:
        seg = np.linalg.norm(np.diff(c, axis=0), axis=1)
        arcs.append(np.concatenate([[0.0], np.cumsum(seg)]))
    arc = np.concatenate(arcs)
    tree = cKDTree(pts)
    search = 2.0 * r + 3.0 * d + 2.0
    # FIX (Claude): vectorized; the first version looped over pairs in Python and took ~70 s.
    pairs = tree.query_pairs(search, output_type="ndarray")
    if len(pairs) == 0:
        return {"min_surface_mm": None, "status": "no_pairs_within_search_radius"}
    i, j = pairs[:, 0], pairs[:, 1]
    keep = ~((sid[i] == sid[j]) & (np.abs(arc[i] - arc[j]) < math.pi * d))
    i, j = i[keep], j[keep]
    if len(i) == 0:
        return {"min_surface_mm": None, "status": "no_pairs_within_search_radius"}
    dist = np.linalg.norm(pts[i] - pts[j], axis=1)
    k = int(np.argmin(dist))
    best = (float(dist[k]), (int(sid[i[k]]), int(sid[j[k]]), float(sval[i[k]]), float(sval[j[k]])))
    spacing = float(np.max(np.diff(s))) if len(s) > 1 else 0.0
    return {
        "min_center_mm": best[0],
        "min_surface_mm": best[0] - 2.0 * r,
        "strands": [best[1][0], best[1][1]],
        "s_mm": [best[1][2], best[1][3]],
        "method": "KD-tree all-pairs on centerline samples",
        "sample_spacing_mm": spacing,
    }
