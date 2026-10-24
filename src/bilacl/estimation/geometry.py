"""Ligament geometry descriptors: centreline, cross-sectional profile, length, tube rendering."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from ..imaging.coords import apply_affine


def mask_points(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    return apply_affine(affine, np.argwhere(mask))


def centerline_profile(mask: np.ndarray, affine: np.ndarray, fp_f: np.ndarray, fp_t: np.ndarray, n_bins: int = 12) -> dict:
    """Centroid-per-station centreline and area-equivalent cross-section along the femoral->tibial axis.

    Stations are slabs perpendicular to the straight footprint-to-footprint axis.
    Area per station = voxel volume in slab / slab thickness."""
    P = mask_points(mask, affine)
    vv = float(abs(np.linalg.det(affine[:3, :3])))
    ax = fp_t - fp_f
    L = float(np.linalg.norm(ax))
    u = ax / L
    t = ((P - fp_f) @ u) / L
    edges = np.linspace(0, 1, n_bins + 1)
    cl, area = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        k = (t >= a) & (t < b)
        if k.sum() < 3:
            cl.append(fp_f + u * L * (a + b) / 2)
            area.append(0.0)
            continue
        cl.append(P[k].mean(0))
        area.append(k.sum() * vv / ((b - a) * L))
    cl = np.array([fp_f, *cl, fp_t])
    seg = np.linalg.norm(np.diff(cl, axis=0), axis=1)
    return {"centerline": cl, "area_mm2": np.array(area), "t_stations": (edges[:-1] + edges[1:]) / 2,
            "straight_length_mm": L, "path_length_mm": float(seg.sum()), "volume_mm3": float(len(P) * vv)}


def render_tube(shape, affine, centerline: np.ndarray, radii: np.ndarray, roi_margin: float = 10.0) -> np.ndarray:
    """Voxelise a tube with circular cross-section of per-sample radius along a polyline."""
    cl = np.asarray(centerline, float)
    # densify
    seg = np.linalg.norm(np.diff(cl, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    ss = np.linspace(0, s[-1], max(50, int(s[-1] / 0.25)))
    C = np.stack([np.interp(ss, s, cl[:, i]) for i in range(3)], 1)
    R = np.interp(ss, np.linspace(0, s[-1], len(radii)), radii)
    inv = np.linalg.inv(affine)
    # voxel bbox of the world bbox (affine may rotate: take corners)
    wlo, whi = C.min(0) - R.max() - 1, C.max(0) + R.max() + 1
    corners = np.array([[x, y, z] for x in (wlo[0], whi[0]) for y in (wlo[1], whi[1]) for z in (wlo[2], whi[2])])
    vc = apply_affine(inv, corners)
    lo = np.maximum(np.floor(vc.min(0)).astype(int), 0)
    hi = np.minimum(np.ceil(vc.max(0)).astype(int) + 1, np.array(shape))
    out = np.zeros(shape, bool)
    if np.any(hi <= lo):
        return out
    ijk = np.indices(hi - lo).reshape(3, -1).T + lo
    X = apply_affine(affine, ijk)
    d, j = cKDTree(C).query(X)
    inside = d <= R[j]
    # exclude end-cap overshoot
    inside &= (j > 0) & (j < len(C) - 1) | (d <= R[j] * 0.5)
    out[tuple(ijk[inside].T)] = True
    return out


def displace_mask(mask: np.ndarray, affine: np.ndarray, a_f: np.ndarray, a_t: np.ndarray, shift_f: np.ndarray, shift_t: np.ndarray) -> np.ndarray:
    """Resample a mask under a smooth translation field blended along the ligament axis:
    d(x) = (1-w) shift_f + w shift_t. Constrained (6 parameters total)."""
    from ..registration.bilateral import blend_weights

    if not mask.any():
        return mask.copy()
    ijk = np.argwhere(ndimage.binary_dilation(mask, iterations=int(np.ceil(max(np.linalg.norm(shift_f), np.linalg.norm(shift_t)) / min(np.linalg.norm(affine[:3, :3], axis=0)))) + 2))
    X = apply_affine(affine, ijk)
    w = blend_weights(X, a_f, a_t)[:, None]
    src = X - ((1 - w) * shift_f + w * shift_t)
    sv = apply_affine(np.linalg.inv(affine), src)
    vals = ndimage.map_coordinates(mask.astype(np.uint8), sv.T, order=0, mode="constant")
    out = np.zeros_like(mask, dtype=bool)
    out[tuple(ijk[vals > 0].T)] = True
    return out


def surface_distance(a: np.ndarray, b: np.ndarray, affine: np.ndarray) -> dict:
    """Symmetric surface distances between two masks on the same grid (mm)."""
    if not a.any() or not b.any():
        return {"assd_mm": float("nan"), "hd95_mm": float("nan")}
    sa = a & ~ndimage.binary_erosion(a)
    sb = b & ~ndimage.binary_erosion(b)
    pa, pb = mask_points(sa, affine), mask_points(sb, affine)
    da, _ = cKDTree(pb).query(pa)
    db, _ = cKDTree(pa).query(pb)
    d = np.concatenate([da, db])
    return {"assd_mm": float(d.mean()), "hd95_mm": float(np.percentile(d, 95))}
