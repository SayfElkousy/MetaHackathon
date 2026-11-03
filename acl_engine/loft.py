"""Parametric ACL loft: curved centerline, varying superellipse sections,
bundle masks, internal strands, and separate rotation channels.

New module. Replaces the straight-axis elliptical sweep with a
spline-based loft driven by editable control points and section keys:

- centerline: centripetal Catmull-Rom through attachment endpoints plus
  intermediate controls, resampled uniformly by arc length;
- frames: rotation-minimizing (parallel-transport) frames from
  acl_engine.frames.compute_rmf -- no sudden flips;
- sections: superellipse rings with per-station width, thickness, shape
  exponent, and orientation offset (smooth PCHIP profiles);
- bundles: AM/PL membership is a smooth mask over section angle with
  configurable sharpness -- a partition of unity, never a mandatory gap;
- strands: curves in section parameter space, clamped inside the envelope,
  running from distributed tibial endpoints to distributed femoral
  endpoints (endpoint sets come from acl_engine.attachments);
- rotation: section orientation, bundle-boundary angle, and per-strand
  swirl are INDEPENDENT smooth functions of normalized arc length t,
  reported in degrees from the frame E1 direction at t = 0.

All array work is vectorized. Units: mm, degrees.
"""
import math
import numpy as np
from scipy.interpolate import PchipInterpolator
from acl_engine.frames import compute_rmf


def catmull_rom(points, per_seg=40):
    """Centripetal Catmull-Rom through control points (k,3).

    Returns a dense polyline (samples do not include the duplicated final
    point of each segment except at the very end). Vectorized per segment.
    """
    P = np.asarray(points, dtype=float)
    if P.shape[0] < 2:
        raise ValueError("need >= 2 control points")
    Q = np.concatenate([P[:1], P, P[-1:]])
    out = []
    for i in range(1, Q.shape[0] - 2):
        p0, p1, p2, p3 = Q[i - 1], Q[i], Q[i + 1], Q[i + 2]
        t0 = 0.0
        t1 = t0 + math.sqrt(max(float(np.linalg.norm(p1 - p0)), 1e-12))
        t2 = t1 + math.sqrt(max(float(np.linalg.norm(p2 - p1)), 1e-12))
        t3 = t2 + math.sqrt(max(float(np.linalg.norm(p3 - p2)), 1e-12))
        s = np.linspace(t1, t2, per_seg + 1)
        if i > 1:
            s = s[1:]
        A1 = ((t1 - s) / (t1 - t0))[:, None] * p0 + ((s - t0) / (t1 - t0))[:, None] * p1
        A2 = ((t2 - s) / (t2 - t1))[:, None] * p1 + ((s - t1) / (t2 - t1))[:, None] * p2
        A3 = ((t3 - s) / (t3 - t2))[:, None] * p2 + ((s - t2) / (t3 - t2))[:, None] * p3
        B1 = ((t2 - s) / (t2 - t0))[:, None] * A1 + ((s - t0) / (t2 - t0))[:, None] * A2
        B2 = ((t3 - s) / (t3 - t1))[:, None] * A2 + ((s - t1) / (t3 - t1))[:, None] * A3
        C = ((t2 - s) / (t2 - t1))[:, None] * B1 + ((s - t1) / (t2 - t1))[:, None] * B2
        out.append(C)
    return np.concatenate(out)


def resample_arclength(C, n):
    """Resample polyline C uniformly by arc length to n points. Vectorized."""
    C = np.asarray(C, dtype=float)
    seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])
    if total <= 0:
        raise ValueError("degenerate centerline")
    u = np.linspace(0.0, total, n)
    return np.stack([np.interp(u, arc, C[:, k]) for k in range(3)], axis=1), total


def smooth_profile(keys_t, keys_v, t, default=0.0):
    """Smooth 1-D profile through keys (PCHIP). Vectorized over t."""
    kt = np.asarray(keys_t, dtype=float)
    kv = np.asarray(keys_v, dtype=float)
    t = np.asarray(t, dtype=float)
    if kt.size == 0:
        return np.full_like(t, float(default))
    if kt.size == 1:
        return np.full_like(t, float(kv[0]))
    order = np.argsort(kt, kind="stable")
    ku = kt[order]
    vu = kv[order]
    _u, keep = np.unique(ku, return_index=True)
    f = PchipInterpolator(_u, vu[np.sort(keep)], extrapolate=True)
    return f(np.clip(t, 0.0, 1.0))


def superellipse_map(r, phi, a, b, e):
    """Map polar (r, phi) to superellipse coords (x, y) with semi-axes a, b.

    e = 1 gives the ellipse; e < 1 boxier, e > 1 more diamond-like.
    r may be an array; a, b, e broadcast against it. Vectorized."""
    c = np.cos(phi)
    s = np.sin(phi)
    x = a * np.sign(c) * np.abs(c) ** e * r
    y = b * np.sign(s) * np.abs(s) ** e * r
    return x, y


def loft_rings(center, N1, N2, width, thick, expo, rot_rad, m=48):
    """Superellipse rings (n,m,3) with per-station rotation. Vectorized."""
    n = center.shape[0]
    ph = np.linspace(0.0, 2.0 * math.pi, m, endpoint=False)[None, :]
    a = (np.asarray(width, dtype=float) / 2.0)[:, None]
    b = (np.asarray(thick, dtype=float) / 2.0)[:, None]
    e = np.asarray(expo, dtype=float)[:, None]
    rr = np.ones_like(a)
    x, y = superellipse_map(rr, np.broadcast_to(ph, (n, m)), a, b, e)
    cr = np.cos(rot_rad)[:, None]
    sr = np.sin(rot_rad)[:, None]
    xr = x * cr - y * sr
    yr = x * sr + y * cr
    return (center[:, None, :] + xr[..., None] * N1[:, None, :]
            + yr[..., None] * N2[:, None, :])


def close_solid(rings):
    """Close a ring tube with fan caps. Returns (V, F) with outward normals
    (orientation fixed by signed volume, same convention as sweep_strand)."""
    from acl_engine.sweep import signed_volume
    R = np.asarray(rings, dtype=float)
    n, m, _ = R.shape
    V = R.reshape(-1, 3)
    ii = np.arange(n - 1)[:, None]
    jj = np.arange(m)[None, :]
    j2 = (jj + 1) % m
    a = ii * m + jj
    b = ii * m + j2
    c = (ii + 1) * m + jj
    d = (ii + 1) * m + j2
    F = [np.stack([np.stack([a, b, c], axis=-1),
                   np.stack([b, d, c], axis=-1)], axis=2).reshape(-1, 3)]
    c0 = V.shape[0]
    c1 = V.shape[0] + 1
    V = np.concatenate([V, R[0].mean(axis=0)[None, :], R[-1].mean(axis=0)[None, :]])
    j = np.arange(m)
    jn = (j + 1) % m
    F.append(np.stack([np.full(m, c0), jn, j], axis=1))
    F.append(np.stack([np.full(m, c1), (n - 1) * m + j, (n - 1) * m + jn], axis=1))
    F = np.concatenate(F).astype(np.int64)
    if signed_volume(V, F) < 0:
        F = F[:, [0, 2, 1]]
    return V, F


def bundle_weights(phi, t, boundary_deg, sharpness=2.0):
    """AM/PL membership weights over section angle phi at normalized
    stations t. Partition of unity (wAM + wPL = 1): soft overlap when
    sharpness is low, crisp split when high. Vectorized.

    boundary_deg may be an array over t (bundle-boundary rotation channel).
    """
    phi = np.asarray(phi, dtype=float)
    t = np.asarray(t, dtype=float)
    psi = np.radians(np.asarray(boundary_deg, dtype=float))
    k = 1.0 + 8.0 * float(np.clip(sharpness, 0.0, 1.0))
    d = np.cos(phi[None, :] - psi[:, None])
    wam = 0.5 + 0.5 * np.tanh(k * d)
    return wam, 1.0 - wam


def strand_paths(center, N1, N2, width, thick, expo, rot_rad, t,
                 tib_polar, fem_polar, swirl_rad, margin=0.92):
    """Internal strand paths from tibial to femoral endpoint distributions.

    tib_polar / fem_polar: (ns, 2) arrays of (r, phi) in unit-disk section
    coords. Each strand interpolates (r, phi) along t with an added swirl
    channel; radius is clamped to `margin` so strands stay inside the
    envelope by construction. Returns (ns, n, 3) paths. Vectorized.
    """
    ns = tib_polar.shape[0]
    n = center.shape[0]
    tt = np.asarray(t, dtype=float)[None, :]
    blend = tt * tt * (3.0 - 2.0 * tt)
    r = tib_polar[:, 0:1] + (fem_polar[:, 0:1] - tib_polar[:, 0:1]) * blend
    r = np.clip(r, 0.0, float(margin))
    dphi = (fem_polar[:, 1:2] - tib_polar[:, 1:2] + math.pi) % (2.0 * math.pi) - math.pi
    phi = tib_polar[:, 1:2] + dphi * blend + np.asarray(swirl_rad, dtype=float)[None, :]
    a = (np.asarray(width, dtype=float) / 2.0)[None, :]
    b = (np.asarray(thick, dtype=float) / 2.0)[None, :]
    e = np.asarray(expo, dtype=float)[None, :]
    x, y = superellipse_map(r, phi, np.broadcast_to(a, (ns, n)),
                            np.broadcast_to(b, (ns, n)),
                            np.broadcast_to(e, (ns, n)))
    cr = np.cos(rot_rad)[None, :]
    sr = np.sin(rot_rad)[None, :]
    xr = x * cr - y * sr
    yr = x * sr + y * cr
    return (center[None, :, :] + xr[..., None] * N1[None, :, :]
            + yr[..., None] * N2[None, :, :])


def strand_lengths(paths):
    """Arc length of each strand path. Vectorized."""
    return np.linalg.norm(np.diff(np.asarray(paths, dtype=float), axis=1), axis=2).sum(axis=1)


def section_area(width, thick, expo):
    """Superellipse section area from the closed form
    A = 4ab * Gamma(1+1/p)^2 / Gamma(1+2/p), p = 2/e. Vectorized."""
    from scipy.special import gamma
    a = np.asarray(width, dtype=float) / 2.0
    b = np.asarray(thick, dtype=float) / 2.0
    p = 2.0 / np.asarray(expo, dtype=float)
    return 4.0 * a * b * gamma(1.0 + 1.0 / p) ** 2 / gamma(1.0 + 2.0 / p)


def flexion_matrix(deg):
    """Tibia-fixed flexion rotation about the mediolateral (+X) axis.

    Positive flexion moves points above the axis anteriorly (femoral shaft
    tilts forward, as in a squat with the foot planted). Illustrative rigid
    pose, not validated kinematics: magnitude of length change with flexion
    is not physiological (real knees roll and slide about a moving center).
    """
    a = math.radians(float(-deg))
    c = math.cos(a)
    s = math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def apply_flexion(points, deg, axis_point):
    """Rigid rotation of points about the flexion axis. Vectorized."""
    R = flexion_matrix(deg)
    P = np.asarray(points, dtype=float)
    C = np.asarray(axis_point, dtype=float)
    return C + (P - C) @ R.T


def frame_continuity(N1, N2, Tang):
    """Min over stations of consecutive-frame alignment (1 = perfect).

    Guards against sudden orientation flips along the centerline."""
    d1 = np.sum(N1[:-1] * N1[1:], axis=1).min()
    d2 = np.sum(N2[:-1] * N2[1:], axis=1).min()
    dt = np.sum(Tang[:-1] * Tang[1:], axis=1).min()
    return float(min(d1, d2, dt))


def build_loft(controls, section_keys, n_station=160, m_ring=48,
               boundary_keys=None, sharpness=0.35):
    """Full loft from control points + section keys.

    controls: (k,3) centerline controls (endpoints = attachment centers).
    section_keys: list of dicts {t, width, thickness, exponent, rotation_deg}.
    boundary_keys: list of {t, angle_deg} for the AM/PL boundary channel
      (defaults to constant 90 deg, i.e. split across the thickness axis).
    Returns dict with centerline, frames, sections, rings, profiles.
    """
    dense = catmull_rom(controls, per_seg=40)
    center, length = resample_arclength(dense, n_station)
    tang = np.gradient(center, axis=0)
    N1, N2, Tang = compute_rmf(center, tang)
    t = np.linspace(0.0, 1.0, n_station)
    kt = [k["t"] for k in section_keys]
    width = smooth_profile(kt, [k["width"] for k in section_keys], t)
    thick = smooth_profile(kt, [k["thickness"] for k in section_keys], t)
    expo = smooth_profile(kt, [k.get("exponent", 1.0) for k in section_keys], t)
    rot = np.radians(smooth_profile(
        kt, [k.get("rotation_deg", 0.0) for k in section_keys], t))
    if boundary_keys:
        bt = [k["t"] for k in boundary_keys]
        bv = [k["angle_deg"] for k in boundary_keys]
    else:
        bt, bv = [0.0, 1.0], [90.0, 90.0]
    boundary = smooth_profile(bt, bv, t)
    rings = loft_rings(center, N1, N2, width, thick, expo, rot, m=m_ring)
    return {"t": t, "center": center, "length_mm": float(length),
            "N1": N1, "N2": N2, "Tang": Tang,
            "width": width, "thick": thick, "expo": expo, "rot_rad": rot,
            "boundary_deg": boundary, "sharpness": float(sharpness),
            "rings": rings, "m_ring": int(m_ring),
            "continuity": frame_continuity(N1, N2, Tang)}
