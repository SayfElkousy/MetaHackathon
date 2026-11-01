"""Milestone A: anatomical ACL bundles, twist and ribbon cross-sections.

New module (no old code changed). Only used when ``anatomy.realism`` is
``"anatomical"``; schematic mode never calls this file, so old numbers cannot
move. Conventions match acl_engine/context.py: ``s`` is distance (mm) from the
TIBIAL insertion along the ACL axis ``u``; ``e1`` is lateral-ish, ``e3`` is
``e1 x u``. All array work is vectorized (no per-point Python loops).

Bundles (AM/PL) share the twist-rotated frame (E1(s), E3(s)) with the overall
envelope. Bundle centres are fixed fractions of the local envelope
semi-axes, so bundles stay inside the overall ACL by construction for any
twist: max normalized radius = (offset_frac + sqrt(area_frac))^2 < 1.

Sources: bundles L11/L12/L16/L17, twist L13, ribbon L02, fan/flare L01/L04.
"""
import math
import numpy as np

REALISM_MODES = ("schematic", "anatomical")

# lateral: fraction of local semi-width along +e1 (+ = lateral, - = medial).
# anterior: fraction of local semi-thickness along anatomical anterior.
# area_frac: bundle ellipse area / envelope ellipse area (bundles overlap).
BUNDLE_DEFS = {
    "AM": {"lateral": -0.14, "anterior": 0.20, "area_frac": 0.50},
    "PL": {"lateral": 0.18, "anterior": -0.16, "area_frac": 0.36},
}
BUNDLE_SOURCES = {"AM": ["L11", "L12", "L16"], "PL": ["L11", "L12", "L16"]}


def _sstep(t):
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def twist_angle(s, L, total_deg, distribution="uniform"):
    """Twist angle (radians) at station(s) s. Vectorized.

    distribution: uniform | femoral_weighted (turns near femur) |
    tibial_weighted (turns near tibia) | custom (no custom table field
    exists in schema 4.0.0, so custom falls back to uniform).
    """
    t = np.clip(np.asarray(s, dtype=float) / float(L), 0.0, 1.0)
    if distribution == "uniform":
        w = t
    elif distribution == "femoral_weighted":
        w = t * t
    elif distribution == "tibial_weighted":
        w = 1.0 - (1.0 - t) ** 2
    elif distribution == "custom":
        w = t
    else:
        raise ValueError("bad twist_distribution %r" % (distribution,))
    return math.radians(float(total_deg)) * w


def rotated_frame(e1, e3, theta):
    """Rotate (e1, e3) about u by theta (radians). Vectorized over theta."""
    e1 = np.asarray(e1, dtype=float)
    e3 = np.asarray(e3, dtype=float)
    th = np.asarray(theta, dtype=float)
    c = np.cos(th)
    s = np.sin(th)
    E1 = c[..., None] * e1 + s[..., None] * e3
    E3 = -s[..., None] * e1 + c[..., None] * e3
    return E1, E3


def ribbon_aspect_scale(s, L):
    """Width/thickness multiplier: flat-wide near femur (L02), rounder
    mid-substance, widening fan at tibia (L01). Area-preserving; the caller
    scales Ws by sqrt(k) and Ts by 1/sqrt(k). Vectorized."""
    t = np.clip(np.asarray(s, dtype=float) / float(L), 0.0, 1.0)
    w_fem = _sstep((t - 0.60) / 0.40)
    w_tib = _sstep((0.40 - t) / 0.40)
    w_mid = np.clip(1.0 - w_fem - w_tib, 0.0, 1.0)
    return 1.40 * w_fem + 0.80 * w_mid + 1.15 * w_tib


def anterior_sign(u, e1, e3):
    """Sign such that anterior_dir = s * e3 (+1 or -1). e3 flips
    anterior/posterior between right and left knees, so this is computed
    from the vectors, not from side."""
    u = np.asarray(u, dtype=float)
    u = u / np.linalg.norm(u)
    e1 = np.asarray(e1, dtype=float)
    e1 = e1 / np.linalg.norm(e1)
    Y = np.array([0.0, 1.0, 0.0])
    a = Y - float(Y @ u) * u - float(Y @ e1) * e1
    if float(np.linalg.norm(a)) < 1e-9:
        return 1.0
    return 1.0 if float(a @ np.asarray(e3, dtype=float)) >= 0.0 else -1.0


def build_bundle_fields(T, u, e1, e3, ss_env, Ws_env, Ts_env,
                        total_deg=0.0, distribution="uniform"):
    """Per-station anatomical fields. Returns a dict consumed by
    build_bundle_meshes, containment_union and acl_engine/tear.py."""
    T = np.asarray(T, dtype=float)
    u = np.asarray(u, dtype=float)
    u = u / np.linalg.norm(u)
    e1 = np.asarray(e1, dtype=float)
    e3 = np.asarray(e3, dtype=float)
    ss = np.asarray(ss_env, dtype=float)
    L = float(ss[-1] - ss[0])
    th = twist_angle(ss, L, total_deg, distribution)
    E1s, E3s = rotated_frame(e1, e3, th)
    sc = ribbon_aspect_scale(ss, L)
    Ws = np.asarray(Ws_env, dtype=float) * np.sqrt(sc)
    Ts = np.asarray(Ts_env, dtype=float) / np.sqrt(sc)
    ae = Ws / 2.0
    be = Ts / 2.0
    C = T[None, :] + ss[:, None] * u[None, :]
    s_ant = anterior_sign(u, e1, e3)
    bundles = {}
    for name, d in BUNDLE_DEFS.items():
        f = float(d["area_frac"])
        off = (float(d["lateral"]) * ae[:, None] * E1s
               + float(d["anterior"]) * s_ant * be[:, None] * E3s)
        Cb = C + off
        seg = np.linalg.norm(np.diff(Cb, axis=0), axis=1)
        bundles[name] = {
            "center": Cb,
            "ae": ae * math.sqrt(f),
            "be": be * math.sqrt(f),
            "area_frac": f,
            "length_mm": float(seg.sum()),
            "tibial_mm": [float(v) for v in Cb[0]],
            "femoral_mm": [float(v) for v in Cb[-1]],
            "sources": list(BUNDLE_SOURCES[name]),
        }
    return {"T": T, "u": u, "e1": e1, "e3": e3, "s_ant": float(s_ant),
            "ss": ss, "L": L, "theta": th, "E1s": E1s, "E3s": E3s,
            "ae": ae, "be": be, "C": C, "Ws": Ws, "Ts": Ts,
            "total_deg": float(total_deg), "distribution": distribution,
            "bundles": bundles, "nsec": 0}


def _tube_faces(n, m):
    ii = np.arange(n - 1)[:, None]
    jj = np.arange(m)[None, :]
    j2 = (jj + 1) % m
    a = ii * m + jj
    b = ii * m + j2
    c = (ii + 1) * m + jj
    d = (ii + 1) * m + j2
    return np.stack([np.stack([a, b, c], axis=-1),
                     np.stack([b, d, c], axis=-1)], axis=2).reshape(-1, 3)


def build_bundle_meshes(fields, nsec=32):
    """Open see-through bundle tubes (side walls only, like ghost_mesh).
    Returns {name: (V, F)}. Vectorized."""
    n = fields["ss"].shape[0]
    ph = np.linspace(0.0, 2.0 * math.pi, nsec, endpoint=False)
    ct = np.cos(ph)
    st = np.sin(ph)
    E1s = fields["E1s"]
    E3s = fields["E3s"]
    out = {}
    for name, b in fields["bundles"].items():
        R = (b["center"][:, None, :]
             + (b["ae"][:, None] * ct[None, :])[..., None] * E1s[:, None, :]
             + (b["be"][:, None] * st[None, :])[..., None] * E3s[:, None, :])
        out[name] = (R.reshape(-1, 3), _tube_faces(n, nsec).astype(np.int64))
    fields["nsec"] = int(nsec)
    return out


def max_bundle_radius(fields):
    """Largest normalized radius of any bundle surface point inside the
    overall (ribbon + twist) envelope. Must be <= 1. Vectorized."""
    n = fields["ss"].shape[0]
    ph = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
    ct = np.cos(ph)
    st = np.sin(ph)
    worst = 0.0
    for b in fields["bundles"].values():
        d = (b["center"] - fields["C"])[:, None, :] \
            + (b["ae"][:, None] * ct[None, :])[..., None] * fields["E1s"][:, None, :] \
            + (b["be"][:, None] * st[None, :])[..., None] * fields["E3s"][:, None, :]
        xl = np.sum(d * fields["E1s"][:, None, :], axis=-1) / fields["ae"][:, None]
        zl = np.sum(d * fields["E3s"][:, None, :], axis=-1) / fields["be"][:, None]
        worst = max(worst, float((xl ** 2 + zl ** 2).max()))
    return math.sqrt(worst)


def containment_union(P, s_lig, fields, n_gon=720):
    """Penetration of points P outside the AM/PL bundle union.

    Same ellipse-boundary (n_gon) method as context.containment_of_points,
    but vectorized with a mask: only outside points pay for the polygon.
    Returns (max_penetration_mm, location|None) with the same location keys
    plus the bundle used.
    """
    P = np.asarray(P, dtype=float)
    ss = fields["ss"]
    s = np.clip(np.asarray(s_lig, dtype=float), ss[0], ss[-1])
    th = twist_angle(s, fields["L"], fields["total_deg"], fields["distribution"])
    E1, E3 = rotated_frame(fields["e1"], fields["e3"], th)
    ae = np.interp(s, ss, fields["ae"])
    be = np.interp(s, ss, fields["be"])
    C = fields["T"][None, :] + s[:, None] * fields["u"][None, :]
    names = list(BUNDLE_DEFS)
    pens = np.zeros((len(names), P.shape[0]))
    xls = np.zeros((len(names), P.shape[0]))
    zls = np.zeros((len(names), P.shape[0]))
    g = np.linspace(0.0, 2.0 * math.pi, n_gon, endpoint=False)
    cg = np.cos(g)[None, :]
    sg = np.sin(g)[None, :]
    for bi, name in enumerate(names):
        d = BUNDLE_DEFS[name]
        f = math.sqrt(float(d["area_frac"]))
        Cb = (C + float(d["lateral"]) * ae[:, None] * E1
              + float(d["anterior"]) * fields["s_ant"] * be[:, None] * E3)
        ae_b = ae * f
        be_b = be * f
        dv = P - Cb
        xl = np.sum(dv * E1, axis=1)
        zl = np.sum(dv * E3, axis=1)
        xls[bi] = xl
        zls[bi] = zl
        outside = (xl / ae_b) ** 2 + (zl / be_b) ** 2 > 1.0
        if np.any(outside):
            ex = ae_b[outside, None] * cg
            ez = be_b[outside, None] * sg
            dx = xl[outside, None] - ex
            dz = zl[outside, None] - ez
            pens[bi, outside] = np.sqrt(dx * dx + dz * dz).min(axis=1)
    best = pens.min(axis=0)
    k = int(np.argmax(best))
    if best[k] <= 0.0:
        return 0.0, None
    bi = int(np.argmin(pens[:, k]))
    return float(best[k]), {"s_mm": float(s[k]), "x_mm": float(xls[bi, k]),
                            "z_mm": float(zls[bi, k]),
                            "point_mm": [float(v) for v in P[k]],
                            "bundle": names[bi]}


def anatomical_context(fields, cov_femoral_anat, stump_info):
    """context.json payload for realism == 'anatomical'."""
    from acl_engine.footprints import (TIBIAL_OVAL_ML, TIBIAL_OVAL_AP,
                                       FEMORAL_R_OUT, FEMORAL_R_IN,
                                       FEMORAL_SPAN_DEG, FEMORAL_ANNULUS_OFFSET)
    out = {
        "mode": "anatomical",
        "twist": {"total_deg": float(fields["total_deg"]),
                  "distribution": fields["distribution"],
                  "source": "L13"},
        "bundles": {name: {"tibial_mm": b["tibial_mm"],
                            "femoral_mm": b["femoral_mm"],
                            "length_mm": float(b["length_mm"]),
                            "area_fraction": float(b["area_frac"]),
                            "sources": list(b["sources"])}
                    for name, b in fields["bundles"].items()},
        "max_bundle_envelope_rn": float(max_bundle_radius(fields)),
        "footprints": {
            "tibial": {"shape": "oval", "ml_mm": float(TIBIAL_OVAL_ML),
                       "ap_mm": float(TIBIAL_OVAL_AP), "sources": ["L15", "L04"]},
            "femoral": {"shape": "crescent", "r_out_mm": float(FEMORAL_R_OUT),
                        "r_in_mm": float(FEMORAL_R_IN),
                        "span_deg": float(FEMORAL_SPAN_DEG),
                        "annulus_offset_ant_mm": float(FEMORAL_ANNULUS_OFFSET),
                        "sources": ["L14", "L04"]}},
        "footprint_coverage_anatomical": {
            "femoral_crescent": cov_femoral_anat,
            "note": "fraction of crescent area under the scaffold end; reported, no threshold"},
    }
    if stump_info is not None:
        out["tear"] = {k: stump_info[k] for k in
                       ("severity", "involvement", "location_s", "obliquity_deg",
                        "azimuth_deg", "morphology", "gap_mm", "gap_nominal_mm",
                        "tibial_stump_mm", "femoral_stump_mm", "cut_bundles",
                        "intact_bundles", "watertight") if k in stump_info}
    return out
