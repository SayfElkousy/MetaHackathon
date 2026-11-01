"""Tube sweep with planar end caps on strand RMF frames."""
import math
import numpy as np


def segments_around(r, tol, vol_rel_err=0.005):
    """Segments around a circle: meet BOTH the chordal tolerance and a
    relative cross-section area error (polygon area ~ 1 - (2*pi^2/3)/N^2)."""
    if tol <= 0:
        raise ValueError("tol must be > 0")
    n_chord = 8 if tol >= r else int(math.ceil(math.pi / math.acos(1.0 - tol / r)))
    n_area = int(math.ceil(math.pi * math.sqrt(2.0 / (3.0 * vol_rel_err))))
    return max(8, n_chord, n_area)


def build_sampling_y(L, bounds, windows, events, p_event_min,
                     kappa_max, tol):
    if kappa_max is None or kappa_max < 1e-9:
        step_chord = 1.0
    else:
        step_chord = math.sqrt(8.0 * tol / kappa_max)
    step = min(step_chord, p_event_min / 12.0)
    step = min(max(step, 0.05), 2.0)
    n = max(2, int(math.ceil(L / step)) + 1)
    ys = np.linspace(0.0, L, n).tolist()
    req = [0.0, L]
    for b in bounds:
        req.append(float(b))
    for (yb, w, _rl, _rr) in windows:
        if w > 0:
            req.append(float(yb - w / 2.0))
            req.append(float(yb + w / 2.0))
    for e in events:
        req.append(float(e["y_mm"]))
    ys = ys + req
    ys = [float(min(max(y, 0.0), L)) for y in ys]
    ys.sort()
    uniq = []
    for y in ys:
        if not uniq or abs(y - uniq[-1]) > 1e-9:
            uniq.append(y)
    return np.array(uniq, dtype=float)


def signed_volume(V, F):
    V = np.asarray(V, dtype=float)
    F = np.asarray(F, dtype=np.int64)
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    return float(np.sum(np.sum(v0 * np.cross(v1, v2), axis=1)) / 6.0)


def sweep_strand(centers, N1, N2, r, nsec):
    centers = np.asarray(centers, dtype=float)
    N1 = np.asarray(N1, dtype=float)
    N2 = np.asarray(N2, dtype=float)
    n = centers.shape[0]
    th = np.linspace(0.0, 2.0 * math.pi, nsec, endpoint=False)
    ct = np.cos(th)
    st = np.sin(th)
    Vside = np.zeros((n * nsec, 3))
    for k in range(n):
        ring = (centers[k][None, :]
                + r * (ct[:, None] * N1[k][None, :]
                       + st[:, None] * N2[k][None, :]))
        Vside[k * nsec:(k + 1) * nsec, :] = ring
    faces = []
    for k in range(n - 1):
        for j in range(nsec):
            j2 = (j + 1) % nsec
            a = k * nsec + j
            b = k * nsec + j2
            c = (k + 1) * nsec + j
            d = (k + 1) * nsec + j2
            faces.append([a, b, c])
            faces.append([b, d, c])
    cs = centers[0]
    ce = centers[-1]
    V = np.vstack([Vside, cs[None, :], ce[None, :]])
    ics = n * nsec
    ice = n * nsec + 1
    for j in range(nsec):
        j2 = (j + 1) % nsec
        faces.append([ics, j2, j])
    base = (n - 1) * nsec
    for j in range(nsec):
        j2 = (j + 1) % nsec
        faces.append([ice, base + j, base + j2])
    F = np.array(faces, dtype=np.int64)
    vol = signed_volume(V, F)
    if vol < 0:
        F = F[:, [0, 2, 1]]
        vol = signed_volume(V, F)
    if vol <= 0:
        raise ValueError("non-positive swept volume")
    return V, F
