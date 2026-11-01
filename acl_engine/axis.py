"""Scaffold axis C(s) and its frame, parameterized by arc length s (mm).

Written by Claude to match the interface Muse's N2 code expects.
Convention (v2 section 3 / v3 section 2.3): right-handed (N1, N2, T) with
N1 x N2 = T. Straight axis: C = (0, s, 0), N1 = +x, N2 = -z, T = +y.
A braid offset (x, z) maps to world as C + x*N1 - z*N2.
"""
import numpy as np


def eval_axis(s, spec=None):
    s = np.asarray(s, dtype=float)
    spec = spec or {"type": "straight"}
    n = s.shape[0]
    if spec.get("type", "straight") == "straight":
        C = np.stack([np.zeros(n), s, np.zeros(n)], axis=1)
        T = np.tile([0.0, 1.0, 0.0], (n, 1))
        N1 = np.tile([1.0, 0.0, 0.0], (n, 1))
        N2 = np.tile([0.0, 0.0, -1.0], (n, 1))
        return C, N1, N2, T
    if spec["type"] == "arc":
        # planar arc in the x-y plane, bending toward +x
        R = float(spec["radius_mm"])
        th = s / R
        C = np.stack([R * (1.0 - np.cos(th)), R * np.sin(th), np.zeros(n)], axis=1)
        T = np.stack([np.sin(th), np.cos(th), np.zeros(n)], axis=1)
        N1 = np.stack([np.cos(th), -np.sin(th), np.zeros(n)], axis=1)
        N2 = np.tile([0.0, 0.0, -1.0], (n, 1))
        return C, N1, N2, T
    raise ValueError("unknown axis type %r" % spec.get("type"))
