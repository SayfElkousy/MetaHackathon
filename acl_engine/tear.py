"""Milestone A tear geometry (T3): presets, oblique cut, retracted stumps.

New module. Uses the existing tear fields (severity, involvement,
location_s, obliquity_deg, azimuth_deg, morphology, retraction_*_mm,
tissue_loss_mm, preset). location_s is the fraction of ACL length from the
TIBIAL insertion (0 = tibia, 1 = femur); this unit was not defined in the
attached files and is defined here. All array work is vectorized.

Cut faces get a deterministic ragged profile (function of ring angle, no
RNG) that only indents, and each stump is closed with fan caps so the cut
faces are watertight. Source L18.
"""
import math
import numpy as np
from scipy.spatial import cKDTree

TEAR_PRESETS = {
    "intact": {"severity": "intact"},
    "proximal_complete": {
        "severity": "complete", "involvement": "whole", "location_s": 0.80,
        "obliquity_deg": 15.0, "azimuth_deg": 30.0, "morphology": "oblique",
        "retraction_femoral_mm": 4.0, "retraction_tibial_mm": 2.0,
        "tissue_loss_mm": 0.0, "disrupted_fraction": 1.0},
    "midsubstance_complete": {
        "severity": "complete", "involvement": "whole", "location_s": 0.50,
        "obliquity_deg": 10.0, "azimuth_deg": 0.0, "morphology": "transverse",
        "retraction_femoral_mm": 3.0, "retraction_tibial_mm": 3.0,
        "tissue_loss_mm": 0.0, "disrupted_fraction": 1.0},
    "partial_AM": {
        "severity": "partial", "involvement": "AM", "location_s": 0.55,
        "obliquity_deg": 20.0, "azimuth_deg": 0.0, "morphology": "oblique",
        "retraction_femoral_mm": 1.5, "retraction_tibial_mm": 0.5,
        "tissue_loss_mm": 0.0, "disrupted_fraction": 0.5},
}

INVOLVEMENTS = ("whole", "AM", "PL")

RAGGEDNESS_BY_MORPHOLOGY = {
    "transverse": 0.35, "oblique": 0.50, "frayed": 0.70, "mop_end": 0.90,
    "partial": 0.50,
}
RAGGEDNESS_DEFAULT = 0.40


def apply_tear_preset(tear):
    """Fill missing tear fields from tear['preset']. Explicit fields win.
    Unknown preset names raise ValueError."""
    t = dict(tear)
    preset = t.get("preset", None)
    if preset is None:
        return t
    if preset not in TEAR_PRESETS:
        raise ValueError("unknown tear preset %r (known: %s)"
                         % (preset, sorted(TEAR_PRESETS)))
    for k, v in TEAR_PRESETS[preset].items():
        t.setdefault(k, v)
    return t


def tear_defaults(tear):
    """Fill defaults, validate ranges, set the geometry status. Intact
    keeps the historical status string so old outputs are byte-comparable."""
    t = dict(tear)
    sev = t.get("severity", "intact")
    if sev not in ("intact", "partial", "complete"):
        raise ValueError("bad tear severity %r" % (sev,))
    if sev == "intact":
        t.setdefault("location_s", None)
        t["status"] = "not_implemented_geometry_T3"
        return t
    loc = t.get("location_s", None)
    if loc is None:
        loc = 0.5
    loc = float(loc)
    if not (0.0 < loc < 1.0):
        raise ValueError("tear.location_s must be a fraction in (0,1) from the tibial insertion")
    t["location_s"] = loc
    for k in ("retraction_femoral_mm", "retraction_tibial_mm", "tissue_loss_mm"):
        v = float(t.get(k, 0.0))
        if v < 0:
            raise ValueError("%s must be >= 0" % k)
        t[k] = v
    obl = float(t.get("obliquity_deg", 0.0))
    if not (0.0 <= obl <= 60.0):
        raise ValueError("tear.obliquity_deg must be in [0,60]")
    t["obliquity_deg"] = obl
    t["azimuth_deg"] = float(t.get("azimuth_deg", 0.0))
    inv = t.get("involvement", "whole")
    if inv not in INVOLVEMENTS:
        raise ValueError("tear.involvement must be whole|AM|PL, got %r" % (inv,))
    t["involvement"] = inv
    t["status"] = "anatomical_a"
    return t


def cut_plane(T, u, e1, e3, L, location_s, obliquity_deg, azimuth_deg,
              twist_total_deg=0.0, twist_distribution="uniform"):
    """Cut plane (Q, n): Q on the ACL axis, n tilted from u by obliquity
    toward the azimuth direction in the (twisted) frame at the cut."""
    from acl_engine.anatomy import twist_angle, rotated_frame
    T = np.asarray(T, dtype=float)
    u = np.asarray(u, dtype=float)
    u = u / np.linalg.norm(u)
    s_cut = float(location_s) * float(L)
    Q = T + s_cut * u
    th = float(twist_angle(np.array([s_cut]), float(L), twist_total_deg,
                           twist_distribution)[0])
    E1, E3 = rotated_frame(np.asarray(e1, float), np.asarray(e3, float),
                           np.array([th]))
    az = math.radians(float(azimuth_deg))
    d = math.cos(az) * E1[0] + math.sin(az) * E3[0]
    obl = math.radians(float(obliquity_deg))
    n = math.cos(obl) * u + math.sin(obl) * d
    return Q, n / np.linalg.norm(n), float(s_cut)


def ragged_profile(theta, amp):
    """Deterministic indent depth in [0.1*amp, amp]. Vectorized."""
    th = np.asarray(theta, dtype=float)
    w = 0.55 + 0.45 * np.sin(3.0 * th + 1.3) * np.sin(2.0 * th + 0.5)
    return float(amp) * np.clip(w, 0.1, 1.0)


def is_watertight(V, F):
    """Closed-mesh check: every edge used by exactly 2 triangles.
    Vectorized. Returns (ok, info)."""
    V = np.asarray(V, dtype=float)
    F = np.asarray(F, dtype=np.int64)
    if F.size == 0 or np.any(F < 0) or np.any(F >= V.shape[0]):
        return False, {"open_edges": -1, "verts": int(V.shape[0]),
                       "faces": int(F.shape[0])}
    if not np.all(np.isfinite(V)):
        return False, {"open_edges": -1, "verts": int(V.shape[0]),
                       "faces": int(F.shape[0])}
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.sort(e, axis=1)
    _u, counts = np.unique(e, axis=0, return_counts=True)
    n_open = int(np.sum(counts != 2))
    return n_open == 0, {"open_edges": n_open, "verts": int(V.shape[0]),
                         "faces": int(F.shape[0])}


def _strip_faces(ring0, ring1):
    m = ring0.shape[0]
    j = np.arange(m)
    j2 = (j + 1) % m
    a = ring0[j]
    b = ring0[j2]
    c = ring1[j]
    d = ring1[j2]
    return np.stack([np.stack([a, b, c], axis=1),
                     np.stack([b, d, c], axis=1)], axis=1).reshape(-1, 3)


def _fan(ctr, ring, flip):
    m = ring.shape[0]
    j = np.arange(m)
    j2 = (j + 1) % m
    if flip:
        return np.stack([np.full(m, ctr), ring[j2], ring[j]], axis=1)
    return np.stack([np.full(m, ctr), ring[j], ring[j2]], axis=1)


def _wall_faces(nrows, m, off):
    ii = np.arange(nrows - 1)[:, None] * m + off
    jj = np.arange(m)[None, :]
    j2 = (jj + 1) % m
    a = ii + jj
    b = ii + j2
    c = ii + m + jj
    d = ii + m + j2
    return np.stack([np.stack([a, b, c], axis=-1),
                     np.stack([b, d, c], axis=-1)], axis=2).reshape(-1, 3)


def split_tube(rings, Q, nvec, side, rag_amp=0.4):
    """Clip a parametric tube (n,m,3) at a plane, close both ends.

    side=+1 keeps the d>0 (femoral) part, -1 keeps d<0 (tibial). The cut
    boundary is the per-generator plane crossing plus an indent-only ragged
    profile, closed with a fan cap; the insertion end keeps its end ring
    plus a fan cap. Returns (V, F, info). Raises ValueError if the plane
    misses the tube wall (reduce obliquity or move location_s).
    """
    from acl_engine.sweep import signed_volume
    R = np.asarray(rings, dtype=float)
    n, m, _ = R.shape
    nv = np.asarray(nvec, dtype=float)
    nv = nv / np.linalg.norm(nv)
    d = (R - np.asarray(Q, dtype=float)) @ nv
    if side > 0:
        crossed = np.any(d <= 0.0, axis=0) & np.any(d > 0.0, axis=0)
    else:
        crossed = np.any(d < 0.0, axis=0) & np.any(d >= 0.0, axis=0)
    if not bool(np.all(crossed)):
        raise ValueError("cut plane misses the tube wall for %d/%d generators; "
                         "reduce tear.obliquity_deg or move tear.location_s"
                         % (int((~crossed).sum()), m))
    th = np.linspace(0.0, 2.0 * math.pi, m, endpoint=False)
    rough = ragged_profile(th, rag_amp)
    jj = np.arange(m)
    # Fan orientation is fixed by shared-edge consistency with the strip
    # pattern (fans on the LOW ring use flip=True, on the HIGH ring False),
    # then one global flip by signed volume (same as sweep_strand).
    if side > 0:
        k1 = np.argmax(d > 0.0, axis=0)
        t = d[k1 - 1, jj] / (d[k1 - 1, jj] - d[k1, jj])
        B = R[k1 - 1, jj] + t[:, None] * (R[k1, jj] - R[k1 - 1, jj])
        B = B + rough[:, None] * nv[None, :]
        kmax = int(k1.max())
        wall = R[kmax:]
        base = wall.shape[0]
        V = np.concatenate([B, wall.reshape(-1, 3)])
        ringB = np.arange(m)
        ringW = m + np.arange(m)
        F = [_strip_faces(ringB, ringW)]
        if base > 1:
            F.append(_wall_faces(base, m, m))
        V = np.concatenate([V, B.mean(axis=0)[None, :],
                            wall[-1].mean(axis=0)[None, :]])
        c_cut = V.shape[0] - 2
        c_end = V.shape[0] - 1
        F.append(_fan(c_cut, ringB, True))
        F.append(_fan(c_end, m + (base - 1) * m + np.arange(m), False))
    else:
        k1 = R.shape[0] - 1 - np.argmax(d[::-1] < 0.0, axis=0)
        t = d[k1 + 1, jj] / (d[k1 + 1, jj] - d[k1, jj])
        B = R[k1 + 1, jj] + t[:, None] * (R[k1, jj] - R[k1 + 1, jj])
        B = B - rough[:, None] * nv[None, :]
        kmin = int(k1.min())
        wall = R[:kmin + 1]
        base = wall.shape[0]
        V = np.concatenate([wall.reshape(-1, 3), B])
        ringW = (base - 1) * m + np.arange(m)
        ringB = base * m + np.arange(m)
        F = []
        if base > 1:
            F.append(_wall_faces(base, m, 0))
        F.append(_strip_faces(ringW, ringB))
        V = np.concatenate([V, wall[0].mean(axis=0)[None, :],
                            B.mean(axis=0)[None, :]])
        c_end = V.shape[0] - 2
        c_cut = V.shape[0] - 1
        # end ring is the LOW end here (flip=True), cut ring the HIGH end
        F.append(_fan(c_end, np.arange(m), True))
        F.append(_fan(c_cut, ringB, False))
    F = np.concatenate(F).astype(np.int64)
    if signed_volume(V, F) < 0:
        F = F[:, [0, 2, 1]]
    if signed_volume(V, F) <= 0:
        raise ValueError("stump mesh has non-positive volume")
    ok, info = is_watertight(V, F)
    info["cut_centroid_mm"] = [float(v) for v in B.mean(axis=0)]
    if not ok:
        raise ValueError("stump mesh is not closed: %s" % (info,))
    return V, F, info


def build_stumps(bundle_fields, bundle_meshes, tear):
    """Cut the involved bundles, retract the ends, report the gap.

    Returns {"stump_tibial": (V,F), "stump_femoral": (V,F), metrics...}.
    Gap is the shortest vertex distance (cKDTree) between the stumps;
    gap_nominal_mm is (ret_fem + ret_tib + tissue_loss) * cos(obliquity).
    """
    t = tear_defaults(apply_tear_preset(dict(tear)))
    F_fields = bundle_fields
    u = F_fields["u"]
    nsec = int(F_fields["nsec"])
    n = int(F_fields["ss"].shape[0])
    Q, nv, s_cut = cut_plane(
        F_fields["T"], u, F_fields["e1"], F_fields["e3"], F_fields["L"],
        t["location_s"], t["obliquity_deg"], t["azimuth_deg"],
        F_fields["total_deg"], F_fields["distribution"])
    inv = t["involvement"]
    cut_names = ["AM", "PL"] if inv == "whole" else [inv]
    intact = [b for b in ("AM", "PL") if b not in cut_names]
    amp = RAGGEDNESS_BY_MORPHOLOGY.get(t.get("morphology", "transverse"),
                                       RAGGEDNESS_DEFAULT)
    shift_f = float(t["retraction_femoral_mm"]) + float(t["tissue_loss_mm"]) / 2.0
    shift_t = float(t["retraction_tibial_mm"]) + float(t["tissue_loss_mm"]) / 2.0
    tib_parts = []
    fem_parts = []
    for name in cut_names:
        V, _F = bundle_meshes[name]
        rings = np.asarray(V, dtype=float).reshape(n, nsec, 3)
        Vt, Ft, it = split_tube(rings, Q, nv, -1, rag_amp=amp)
        Vf, Ff, iff_ = split_tube(rings, Q, nv, +1, rag_amp=amp)
        tib_parts.append((Vt - shift_t * u[None, :], Ft))
        fem_parts.append((Vf + shift_f * u[None, :], Ff))
    stump_tib = _concat(tib_parts)
    stump_fem = _concat(fem_parts)
    tree = cKDTree(stump_fem[0])
    gap = float(tree.query(stump_tib[0], k=1)[0].min())
    nominal = (shift_f + shift_t) * math.cos(math.radians(float(t["obliquity_deg"])))
    okt, infot = is_watertight(*stump_tib)
    okf, infof = is_watertight(*stump_fem)
    L = float(F_fields["L"])
    return {
        "stump_tibial": stump_tib, "stump_femoral": stump_fem,
        "severity": t["severity"], "involvement": inv,
        "location_s": float(t["location_s"]),
        "obliquity_deg": float(t["obliquity_deg"]),
        "azimuth_deg": float(t["azimuth_deg"]),
        "morphology": t.get("morphology", "transverse"),
        "gap_mm": gap, "gap_nominal_mm": float(nominal),
        "tibial_stump_mm": float(s_cut), "femoral_stump_mm": float(L - s_cut),
        "cut_bundles": cut_names, "intact_bundles": intact,
        "retraction_femoral_mm": float(t["retraction_femoral_mm"]),
        "retraction_tibial_mm": float(t["retraction_tibial_mm"]),
        "watertight": {"tibial": bool(okt), "femoral": bool(okf),
                       "tibial_open_edges": infot["open_edges"],
                       "femoral_open_edges": infof["open_edges"]},
    }


def _concat(parts):
    Vs = []
    Fs = []
    off = 0
    for V, F in parts:
        Vs.append(np.asarray(V, dtype=float))
        Fs.append(np.asarray(F, dtype=np.int64) + off)
        off += V.shape[0]
    return np.concatenate(Vs), np.concatenate(Fs).astype(np.int64)
