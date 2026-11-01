"""Knee frame, native ACL envelope, placement, checks and schematic bone meshes (Muse N2).

Knee frame: +X lateral, +Y anterior, +Z proximal, origin at the tibial plateau
center, right knee (mirrored for left). Schematic, not patient anatomy.
"""
import math
import numpy as np


def quintic(t):
    t = np.asarray(t, dtype=float)
    return 6.0 * t ** 5 - 15.0 * t ** 4 + 10.0 * t ** 3


def unit_dir_from_inclinations(sag_deg, cor_deg):
    ts = math.tan(math.radians(float(sag_deg)))
    tc = math.tan(math.radians(float(cor_deg)))
    v = np.array([1.0 / tc, -1.0 / ts, 1.0], dtype=float)
    return v / np.linalg.norm(v)


def tibial_center(side="right"):
    x = 4.0 if side == "right" else -4.0
    return np.array([x, 8.0, 0.0], dtype=float)


def build_native_axis(acl_len, sag_deg, cor_deg, side="right"):
    T = tibial_center(side)
    u = unit_dir_from_inclinations(sag_deg, cor_deg)
    if side == "left":
        u = np.array([-u[0], u[1], u[2]])
    F = T + float(acl_len) * u
    return T, F, u


def measured_inclinations(u):
    ux, uy, uz = float(u[0]), float(u[1]), float(u[2])
    sag = math.degrees(math.atan2(abs(uz), abs(uy)))
    cor = math.degrees(math.atan2(abs(uz), abs(ux)))
    return sag, cor


def area_base(s, L, profile):
    """Area at distance s from the TIBIAL end; profile is listed femur -> tibia."""
    prof = np.asarray(profile, dtype=float)
    pos = np.array([L, 0.75 * L, 0.5 * L, 0.25 * L, 0.0])
    return float(np.interp(float(s), pos[::-1], prof[::-1]))


def build_envelope_fields(L, Wmid, Tmid, profile, flare_ratio, flare_len=8.0, flat_femoral=False):
    aspect = float(Wmid) / float(Tmid)
    Amid = area_base(L * 0.5, L, profile)
    n = 241
    ss = np.linspace(0.0, L, n)
    A = np.zeros(n)
    for k, s in enumerate(ss):
        base = area_base(s, L, profile)
        w = 0.0
        if s < flare_len:
            w = max(w, 1.0 - float(quintic(np.array([s / flare_len]))[0]))
        if s > L - flare_len:
            w = max(w, float(quintic(np.array([(s - (L - flare_len)) / flare_len]))[0]))
        A[k] = (1.0 - w) * base + w * float(flare_ratio) * Amid
    Ws = np.sqrt(4.0 * A * aspect / math.pi)
    Ts = np.sqrt(4.0 * A / (aspect * math.pi))
    if flat_femoral:
        for k, s in enumerate(ss):
            if s > L - flare_len:
                t = (s - (L - flare_len)) / flare_len
                f = 1.0 - 0.3 * float(quintic(np.array([t]))[0])
                Ts[k] = Ts[k] * f
                Ws[k] = Ws[k] / math.sqrt(f)
    return ss, A, Ws, Ts, Amid


def basis_for_dir(u, side="right"):
    u = np.asarray(u, dtype=float)
    u = u / np.linalg.norm(u)
    X = np.array([1.0, 0.0, 0.0]) if side == "right" else np.array([-1.0, 0.0, 0.0])
    e1 = X - float(np.dot(X, u)) * u
    e1 = e1 / np.linalg.norm(e1)
    e3 = np.cross(e1, u)
    e3 = e3 / np.linalg.norm(e3)
    return e1, u, e3


def box_mesh(center, size):
    c = np.asarray(center, dtype=float)
    sx, sy, sz = float(size[0]) / 2.0, float(size[1]) / 2.0, float(size[2]) / 2.0
    V = np.array([[c[0] - sx, c[1] - sy, c[2] - sz], [c[0] + sx, c[1] - sy, c[2] - sz], [c[0] + sx, c[1] + sy, c[2] - sz],
                  [c[0] - sx, c[1] + sy, c[2] - sz], [c[0] - sx, c[1] - sy, c[2] + sz], [c[0] + sx, c[1] - sy, c[2] + sz],
                  [c[0] + sx, c[1] + sy, c[2] + sz], [c[0] - sx, c[1] + sy, c[2] + sz]])
    F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [3, 6, 2], [3, 7, 6],
                  [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]], dtype=np.int64)
    return V, F


def ellipsoid_mesh(center, radii, nlat=24, nlon=36):
    c = np.asarray(center, dtype=float)
    rx, ry, rz = float(radii[0]), float(radii[1]), float(radii[2])
    V = []
    for i in range(nlat + 1):
        th = math.pi * i / nlat
        for j in range(nlon):
            ph = 2.0 * math.pi * j / nlon
            V.append([c[0] + rx * math.sin(th) * math.cos(ph), c[1] + ry * math.sin(th) * math.sin(ph), c[2] + rz * math.cos(th)])
    V = np.array(V)
    F = []
    for i in range(nlat):
        for j in range(nlon):
            j2 = (j + 1) % nlon
            a = i * nlon + j
            b = i * nlon + j2
            c2 = (i + 1) * nlon + j
            d = (i + 1) * nlon + j2
            F.append([a, c2, b])
            F.append([b, c2, d])
    return V, np.array(F, dtype=np.int64)


def cylinder_mesh(p0, p1, radius, nsec=24):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    ax = p1 - p0
    L = float(np.linalg.norm(ax))
    ax = ax / L
    tmp = np.array([1.0, 0.0, 0.0]) if abs(float(np.dot(ax, np.array([1.0, 0.0, 0.0])))) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = tmp - float(np.dot(tmp, ax)) * ax
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(ax, e1)
    th = np.linspace(0.0, 2.0 * math.pi, nsec, endpoint=False)
    V = []
    for s in [0.0, 1.0]:
        pc = p0 + (p1 - p0) * s
        for k in range(nsec):
            V.append(pc + radius * (math.cos(th[k]) * e1 + math.sin(th[k]) * e2))
    V = np.array(V)
    F = []
    for k in range(nsec):
        k2 = (k + 1) % nsec
        F.append([k, k2, nsec + k])
        F.append([k2, nsec + k2, nsec + k])
    return V, np.array(F, dtype=np.int64)


def ghost_mesh(T, u, e1, e3, ss, Ws, Ts, nsec=32):
    n = len(ss)
    V = np.zeros((n * nsec, 3))
    th = np.linspace(0.0, 2.0 * math.pi, nsec, endpoint=False)
    ct = np.cos(th)
    st = np.sin(th)
    for k in range(n):
        Ck = T + ss[k] * u
        for j in range(nsec):
            V[k * nsec + j] = Ck + (Ws[k] / 2.0) * ct[j] * e1 + (Ts[k] / 2.0) * st[j] * e3
    F = []
    for k in range(n - 1):
        for j in range(nsec):
            j2 = (j + 1) % nsec
            F.append([k * nsec + j, k * nsec + j2, (k + 1) * nsec + j])
            F.append([k * nsec + j2, (k + 1) * nsec + j2, (k + 1) * nsec + j])
    return V, np.array(F, dtype=np.int64)


def ellipse_boundary_dist(xl, zl, ae, be, n=720):
    th = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    ex = ae * np.cos(th)
    ez = be * np.sin(th)
    return float(np.sqrt(((xl - ex) ** 2 + (zl - ez) ** 2).min()))


def containment_of_points(P, s_lig, T, u, e1, e3, ss_env, Ws, Ts):
    maxpen = 0.0
    loc = None
    for k in range(P.shape[0]):
        s = float(np.clip(s_lig[k], ss_env[0], ss_env[-1]))
        ae = float(np.interp(s, ss_env, Ws)) / 2.0
        be = float(np.interp(s, ss_env, Ts)) / 2.0
        Ck = T + s * u
        d = P[k] - Ck
        xl = float(np.dot(d, e1))
        zl = float(np.dot(d, e3))
        rn = (xl / ae) ** 2 + (zl / be) ** 2
        if rn <= 1.0:
            continue
        pen = ellipse_boundary_dist(xl, zl, ae, be)
        if pen > maxpen:
            maxpen = pen
            loc = {"s_mm": s, "x_mm": xl, "z_mm": zl, "point_mm": [float(v) for v in P[k]]}
    return maxpen, loc


def footprint_coverage(Ws_end, Ts_end, Wf, Lf, grid=101):
    xs = np.linspace(-Wf / 2.0, Wf / 2.0, grid)
    ys = np.linspace(-Lf / 2.0, Lf / 2.0, grid)
    inf = 0
    ins = 0
    for x in xs:
        for y in ys:
            if (x / (Wf / 2.0)) ** 2 + (y / (Lf / 2.0)) ** 2 <= 1.0:
                inf += 1
                if (x / (Ws_end / 2.0)) ** 2 + (y / (Ts_end / 2.0)) ** 2 <= 1.0:
                    ins += 1
    return float(ins) / float(inf) if inf > 0 else 0.0


def dist_point_box(P, bmin, bmax):
    q = np.maximum(bmin - P, 0.0) + np.maximum(P - bmax, 0.0)
    outside = float(np.linalg.norm(q, axis=1).min()) if P.shape[0] > 0 else 0.0
    inside = np.all((P >= bmin) & (P <= bmax), axis=1)
    return outside, bool(np.any(inside))


# --- Milestone A: two-bundle anatomical ACL (implementation in
# acl_engine/anatomy.py; old functions above are unchanged) ---
from acl_engine.anatomy import (REALISM_MODES, BUNDLE_DEFS, twist_angle,
                                rotated_frame, ribbon_aspect_scale,
                                build_bundle_fields, build_bundle_meshes,
                                max_bundle_radius, containment_union,
                                anatomical_context)  # noqa: E402,F401
