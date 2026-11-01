"""Double-reflection rotation-minimizing frames (Wang et al. 2008, Table I)."""
import numpy as np


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-30:
        raise ValueError("zero vector in frame init")
    return v / n


def initial_normal(t0):
    ex = np.array([1.0, 0.0, 0.0])
    u = ex - float(np.dot(ex, t0)) * t0
    if np.linalg.norm(u) < 1e-9:
        ez = np.array([0.0, 0.0, 1.0])
        u = ez - float(np.dot(ez, t0)) * t0
    if np.linalg.norm(u) < 1e-12:
        raise ValueError("degenerate initial tangent")
    return _normalize(u)


def compute_rmf(centers, tangents):
    centers = np.asarray(centers, dtype=float)
    tangents = np.asarray(tangents, dtype=float)
    n = centers.shape[0]
    if n < 2:
        raise ValueError("need >= 2 points")
    T = np.zeros_like(tangents)
    for k in range(n):
        T[k] = _normalize(tangents[k])
    N1 = np.zeros_like(T)
    N1[0] = initial_normal(T[0])
    for k in range(n - 1):
        p0 = centers[k]
        p1 = centers[k + 1]
        t0 = T[k]
        t1 = T[k + 1]
        u0 = N1[k]
        v1 = p1 - p0
        c1 = float(np.dot(v1, v1))
        if c1 < 1e-24:
            uL = u0.copy()
            tL = t0.copy()
        else:
            uL = u0 - (2.0 / c1) * float(np.dot(v1, u0)) * v1
            tL = t0 - (2.0 / c1) * float(np.dot(v1, t0)) * v1
        v2 = t1 - tL
        c2 = float(np.dot(v2, v2))
        if c2 < 1e-24:
            u1 = uL
        else:
            u1 = uL - (2.0 / c2) * float(np.dot(v2, uL)) * v2
        u1 = u1 - float(np.dot(u1, t1)) * t1
        nu = np.linalg.norm(u1)
        if nu < 1e-12:
            u1 = initial_normal(t1)
        else:
            u1 = u1 / nu
        N1[k + 1] = u1
    N2 = np.cross(T, N1)
    for k in range(n):
        N2[k] = _normalize(N2[k] - float(np.dot(N2[k], T[k])) * T[k]
                           - float(np.dot(N2[k], N1[k])) * N1[k])
    # verify
    d12 = np.abs(np.sum(N1 * N2, axis=1)).max()
    d1t = np.abs(np.sum(N1 * T, axis=1)).max()
    d2t = np.abs(np.sum(N2 * T, axis=1)).max()
    cr = np.cross(N1, N2)
    rh = np.linalg.norm(cr - T, axis=1).max()
    if d12 > 1e-9 or d1t > 1e-9 or d2t > 1e-9 or rh > 1e-9:
        raise ValueError("RMF orthonormality failed: %g %g %g %g"
                         % (d12, d1t, d2t, rh))
    return N1, N2, T
