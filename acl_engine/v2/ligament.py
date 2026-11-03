"""Anatomical ACL model: footprints, envelope, bundles, fascicles, fibers, injury.

Coordinates
-----------
- s in [0, 1]: normalized arc length of the centreline, tibia (0) to femur (1).
- material coordinates m = (m1, m2) in the unit disk: the identity of a point
  in the cross-section. A fascicle keeps its material position along its
  whole length (except for its own within-bundle twist), so its two ends map
  to well-defined footprint locations and fascicles cannot cross.
- section frame at s: N1 (starts anterior at the tibia), N2 = T x N1 (starts
  medial for a right knee), both rotation-minimizing (double reflection), so
  there are no sudden flips on curved paths.

Angles (all documented relative to a named axis)
------------------------------------------------
- section orientation: rotation of the section's wide axis from N1, about T.
- bundle-region rotation theta_b(s): rotation of the AM/PL arrangement (all
  content) about T, 0 at the tibia, total at the femur.
- fascicle angle: angle between the outermost fascicle of a bundle and the
  local centreline tangent T (rigid twist about the bundle centroid).
- fiber angle: angle between the outermost fiber of a fascicle and the
  fascicle tangent.
- crimp angle (derived) = atan(2 pi A / lambda).
"""
import math
import numpy as np
from acl_engine.v2.design import S_KNOTS, segment_offset_for_fraction, fascicle_sizing, PACKING_LIMIT
from acl_engine.v2 import mesher

GOLDEN = math.pi * (3 - math.sqrt(5))
SEC_K = 192      # contour samples per station
RULE_K = 65      # samples along the interface / outer arcs (ruled map)
WARP_AL = np.linspace(-np.pi, np.pi, 256, endpoint=False)   # angle grid of the radial warp


def _unit(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-12)


def _sstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def profile_eval(values, s):
    """Monotone (PCHIP) interpolation of a profile.

    A profile is either a list of values at S_KNOTS or a dict {"s": [...], "v": [...]}
    with any number of samples (e.g. measured along the ligament by the MRI model);
    imported spatial detail is kept, never averaged down to the default knots."""
    from scipy.interpolate import PchipInterpolator
    if isinstance(values, dict):
        xs = np.asarray(values["s"], float)
        vs = np.asarray(values["v"], float)
        o = np.argsort(xs)
        xs, vs = xs[o], vs[o]
        if xs.size == 1:
            return np.full(np.shape(s), vs[0])
        return PchipInterpolator(xs, vs, extrapolate=True)(np.clip(s, xs[0], xs[-1]))
    return PchipInterpolator(S_KNOTS, np.asarray(values, float))(np.clip(s, 0, 1))


def rot2(m, ang):
    c, s = np.cos(ang), np.sin(ang)
    return np.stack([c * m[..., 0] - s * m[..., 1], s * m[..., 0] + c * m[..., 1]], -1)


def sunflower(n, rmax=1.0):
    k = np.arange(n) + 0.5
    r = np.sqrt(k / n) * rmax
    a = k * GOLDEN
    return np.stack([r * np.cos(a), r * np.sin(a)], 1)


def catmull_rom(P, per_seg=60):
    P = np.asarray(P, float)
    Q = np.concatenate([P[:1] * 2 - P[1:2], P, P[-1:] * 2 - P[-2:-1]])
    out = []
    for i in range(1, Q.shape[0] - 2):
        p0, p1, p2, p3 = Q[i - 1], Q[i], Q[i + 1], Q[i + 2]
        t = np.linspace(0, 1, per_seg + 1)[:, None]
        if i > 1:
            t = t[1:]
        c = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)
        out.append(c)
    return np.concatenate(out)


def resample(C, n):
    d = np.linalg.norm(np.diff(C, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(d)])
    L = cum[-1]
    t = np.linspace(0, L, n)
    out = np.stack([np.interp(t, cum, C[:, a]) for a in range(3)], 1)
    return out, L


def rmf(C, n0):
    """Double-reflection rotation-minimizing frames (Wang et al. 2008, L10)."""
    T = np.gradient(C, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    N = np.zeros_like(C)
    N[0] = _unit(n0 - np.dot(n0, T[0]) * T[0])
    for k in range(C.shape[0] - 1):
        v1 = C[k + 1] - C[k]
        c1 = v1 @ v1
        rL = N[k] - (2 / c1) * (v1 @ N[k]) * v1
        tL = T[k] - (2 / c1) * (v1 @ T[k]) * v1
        v2 = T[k + 1] - tL
        c2 = v2 @ v2
        N[k + 1] = rL - (2 / c2) * (v2 @ rL) * v2 if c2 > 1e-20 else rL
        N[k + 1] = _unit(N[k + 1] - np.dot(N[k + 1], T[k + 1]) * T[k + 1])
    B = np.cross(T, N)
    return T, N, B


# ---------------------------------------------------------------- footprints
class Footprint:
    """Attachment region: a 2D chart in the bone's tangent plane, projected onto the bone."""

    def __init__(self, side, p, bone, nr=20, na=96):
        self.side = side
        self.bone = bone
        key = "fp.%s." % side
        c_local = np.asarray(p[key + "center"], float)
        c_world = bone.to_world(c_local[None])[0] if hasattr(bone, "to_world") else c_local
        self.center = bone.project(c_world[None])[0]
        self.normal = _unit(bone.normal(self.center[None])[0])
        if side == "tib":
            ref = np.array([0.0, 1.0, 0.0])          # anterior
        else:
            z = np.array([0.0, 0.0, 1.0])
            ref = bone.to_world(z[None])[0] - bone.to_world(np.zeros((1, 3)))[0] if hasattr(bone, "to_world") else z
        el = _unit(ref - np.dot(ref, self.normal) * self.normal)
        side_dir = _unit(np.cross(el, self.normal))  # tibia: lateral; femur (lateral wall): posterior
        sgn = 1.0 if side == "tib" else -1.0          # femur: rotate toward anterior
        o = math.radians(float(p[key + "orientation_deg"]))
        self.e_l = _unit(math.cos(o) * el + sgn * math.sin(o) * side_dir)
        self.e_s = _unit(np.cross(self.normal, self.e_l))
        self.shape = p[key + "shape"]
        self.L = float(p[key + "length_mm"])
        self.W = float(p[key + "width_mm"])
        self.Rc = float(p[key + "curvature_mm"])
        self.poly = np.asarray(p[key + "polygon"], float) if p[key + "polygon"] else None
        # lookup grid in (rho, phi): projected surface points
        self.nr, self.na = nr, na
        rho = np.linspace(0, 1, nr)
        phi = np.linspace(0, 2 * np.pi, na, endpoint=False)
        R, PH = np.meshgrid(rho, phi, indexing="ij")
        uv = np.stack([R * np.cos(PH), R * np.sin(PH)], -1).reshape(-1, 2)
        ab = self.chart(uv)
        P = self.center + ab[:, :1] * self.e_l + ab[:, 1:] * self.e_s
        P = bone.project(P)
        self.grid = P.reshape(nr, na, 3)
        self.grid_ab = ab.reshape(nr, na, 2)

    def chart(self, uv):
        u, v = uv[:, 0], uv[:, 1]
        if self.shape == "oval":
            return np.stack([0.5 * self.L * u, 0.5 * self.W * v], 1)
        if self.shape == "crescent":
            span = self.L / self.Rc
            al = u * span / 2
            r = self.Rc + v * self.W / 2
            return np.stack([r * np.sin(al), self.Rc - r * np.cos(al)], 1)
        # polygon: star-shaped about its centroid
        P = self.poly
        c = P.mean(0)
        rho = np.hypot(u, v)
        ph = np.arctan2(v, u)
        d = np.stack([np.cos(ph), np.sin(ph)], 1)
        A = P - c
        Bq = np.roll(P, -1, 0) - c
        E = Bq - A
        # ray c + t d hits edge A + w E: solve [d, -E][t, w] = A
        det = d[:, None, 0] * (-E[None, :, 1]) - d[:, None, 1] * (-E[None, :, 0])
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        t = (A[None, :, 0] * (-E[None, :, 1]) - A[None, :, 1] * (-E[None, :, 0])) / det
        w = (d[:, None, 0] * A[None, :, 1] - d[:, None, 1] * A[None, :, 0]) / det
        t = np.where((w >= -1e-9) & (w <= 1 + 1e-9) & (t > 0), t, np.inf)
        rb = t.min(1)
        rb = np.where(np.isfinite(rb), rb, 0.0)
        return c + (rho * rb)[:, None] * d

    def points(self, f):
        """Surface points for footprint coordinates f (k,2) in the unit disk (bilinear lookup)."""
        rho = np.clip(np.hypot(f[:, 0], f[:, 1]), 0, 1) * (self.nr - 1)
        ph = (np.arctan2(f[:, 1], f[:, 0]) % (2 * np.pi)) / (2 * np.pi) * self.na
        i0 = np.clip(np.floor(rho).astype(int), 0, self.nr - 2)
        tr = rho - i0
        j0 = np.floor(ph).astype(int) % self.na
        j1 = (j0 + 1) % self.na
        tp = ph - np.floor(ph)
        g = self.grid
        a = g[i0, j0] * (1 - tp)[:, None] + g[i0, j1] * tp[:, None]
        b = g[i0 + 1, j0] * (1 - tp)[:, None] + g[i0 + 1, j1] * tp[:, None]
        return a * (1 - tr)[:, None] + b * tr[:, None]

    def area(self):
        g = self.grid
        a = g[:-1, :]
        b = g[1:, :]
        c = np.roll(g[1:, :], -1, 1)
        d = np.roll(g[:-1, :], -1, 1)
        t1 = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=-1)
        t2 = 0.5 * np.linalg.norm(np.cross(c - a, d - a), axis=-1)
        return float((t1 + t2).sum())

    def mesh(self, lift=0.12, thick=0.3, bundle_fn=None):
        """Closed thin solid following the bone surface. Returns V, F, bundle-id per vertex."""
        g = self.grid
        nr, na = self.nr, self.na
        n = self.bone.normal(g.reshape(-1, 3)).reshape(nr, na, 3)
        top = g + n * (lift + thick)
        bot = g + n * lift
        V = np.vstack([top.reshape(-1, 3), bot.reshape(-1, 3)])
        idx = np.arange(nr * na).reshape(nr, na)
        F = []
        for off, flip in ((0, False), (nr * na, True)):
            a = idx[:-1, :] + off
            b = idx[1:, :] + off
            c = np.roll(idx[1:, :], -1, 1) + off
            d = np.roll(idx[:-1, :], -1, 1) + off
            t1 = np.stack([a, b, c], -1).reshape(-1, 3)
            t2 = np.stack([a, c, d], -1).reshape(-1, 3)
            T = np.vstack([t1, t2])
            F.append(T[:, ::-1] if flip else T)
        # rim
        rt = idx[-1, :]
        rb = rt + nr * na
        rt2 = np.roll(rt, -1)
        rb2 = np.roll(rb, -1)
        F.append(np.stack([rt, rb, rb2], 1))
        F.append(np.stack([rt, rb2, rt2], 1))
        F = np.vstack(F)
        # the rho=0 ring is degenerate (all na points coincide): merge to one vertex per layer
        for off in (0, nr * na):
            F = np.where((F >= off) & (F < off + na), off, F)
        keep = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
        F = F[keep]
        used = np.unique(F)
        remap = -np.ones(V.shape[0], np.int64)
        remap[used] = np.arange(used.size)
        V2 = V[used]
        F2 = remap[F]
        if np.dot(np.cross(V2[F2[0, 1]] - V2[F2[0, 0]], V2[F2[0, 2]] - V2[F2[0, 0]]), self.normal) < 0:
            F2 = F2[:, ::-1]
        bid = None
        if bundle_fn is not None:
            rho = np.linspace(0, 1, nr)
            phi = np.linspace(0, 2 * np.pi, na, endpoint=False)
            R_, P_ = np.meshgrid(rho, phi, indexing="ij")
            f = np.stack([R_ * np.cos(P_), R_ * np.sin(P_)], -1).reshape(-1, 2)
            b = bundle_fn(f)
            bid = np.concatenate([b, b])[used].astype(np.uint8)
        return V2, F2.astype(np.int64), bid


# ---------------------------------------------------------------- ligament
class Ligament:
    """Intact ACL envelope with contour sections and an AM/PL partition.

    Each station along the centreline carries a closed CONTOUR (mm, in the
    rotation-minimizing frame N1/N2) and an AM/PL INTERFACE (a line, optionally
    curved, crossing the contour at E- and E+). The two bundle regions are the
    parts of the contour on either side of the interface.

    Material coordinates m (unit disk) are partitioned by a fixed chord into an
    AM segment and a PL segment. Each segment maps to its bundle region by a
    RULED map: material chord -> metric interface, material arc -> the
    bundle's outer contour arc (both by arc-length fraction t), linear in
    between (u). Both bundles share the interface curve exactly, so the two
    regions tile the section without gaps or overlaps at every station, and a
    material point stays in its bundle along the whole ligament.
    """

    def __init__(self, p, femur, tibia, quality="full"):
        self.p = p
        self.quality = quality
        self.issues = []          # section-model problems: {param, level, message}
        self.fp = {"tib": Footprint("tib", p, tibia), "fem": Footprint("fem", p, femur)}
        n_st = 180 if quality == "full" else 80
        self.s = np.linspace(0, 1, n_st)
        self.length_info = {}
        self._centreline(n_st)
        self.T, self.N1, self.N2 = rmf(self.C, np.array([0.0, 1.0, 0.0]))
        s = self.s
        self.theta_b = self._rotation(s)
        # material partition (fixed along the ligament)
        f = float(p["bundle.am_fraction"])
        self.chord_c = segment_offset_for_fraction(f)
        b0 = math.radians(p["bundle.boundary_angle_deg"])
        self.chord_d = np.array([math.cos(b0), math.sin(b0)])
        self._material_arcs()
        self._profiles()
        self._build_sections()
        self.align = {"tib": 0.0, "fem": 0.0}
        self.align["tib"] = self._auto_align("tib")
        self.align["fem"] = self._auto_align("fem")
        self.inclinations = self._inclinations()

    # ------------------------------------------------------------ centreline
    def _end_thickness(self):
        p = self.p
        mode = p["lig.section_mode"]
        sc = float(p["lig.size_scale"])
        tau0 = 1 + float(p["lig.taper_tib"])
        tau1 = 1 + float(p["lig.taper_fem"])
        if mode == "area_aspect":
            out = []
            for sv in (0.0, 1.0):
                A = float(profile_eval(p["lig.area_profile"], np.array([sv]))[0])
                a = float(profile_eval(p["lig.aspect_profile"], np.array([sv]))[0])
                E = float(profile_eval(p["lig.shape_profile"], np.array([sv]))[0])
                out.append(math.sqrt(A / (a * _sq_area_factor(E))))
            t0, t1 = out
        elif mode == "per_bundle":
            ev = lambda k, sv: float(profile_eval(p[k], np.array([sv]))[0])
            t0 = max(ev("am.thickness_profile", 0.0), ev("pl.thickness_profile", 0.0))
            t1 = max(ev("am.thickness_profile", 1.0), ev("pl.thickness_profile", 1.0))
        else:
            t0 = float(profile_eval(p["lig.thickness_profile"], np.array([0.0]))[0])
            t1 = float(profile_eval(p["lig.thickness_profile"], np.array([1.0]))[0])
        return t0 * sc * tau0, t1 * sc * tau1

    def _ctrl_points(self, bow_a):
        p = self.p
        tk0, tk1 = self._end_thickness()
        # the body starts/ends half a thickness off each bone surface; the
        # insertion flare carries the tissue onto the footprint itself
        A = self.fp["tib"].center + self.fp["tib"].normal * 0.5 * tk0
        B = self.fp["fem"].center + self.fp["fem"].normal * 0.5 * tk1
        ab = B - A
        t = ab / np.linalg.norm(ab)
        ant = np.array([0.0, 1.0, 0.0])
        y_perp = _unit(ant - np.dot(ant, t) * t)
        x_perp = _unit(np.cross(y_perp, t))
        if np.dot(x_perp, [1, 0, 0]) < 0:
            x_perp = -x_perp
        offs = np.asarray(p["lig.ctrl_offsets"], float)
        ctrl = [A]
        imported = p.get("lig.centerline_points") or []
        for k, f in enumerate((1 / 3, 2 / 3)):
            bow = math.sin(math.pi * f) * (bow_a * y_perp + p["lig.bow_lateral_mm"] * x_perp)
            if len(imported) >= 3:
                # imported centreline (model frame): interior points become controls; the
                # ends stay on the (possibly edited) attachments; offsets/bow still apply
                Q = np.asarray(imported, float)
                d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))])
                t_ = d / d[-1]
                q = np.array([np.interp(f, t_, Q[:, a]) for a in range(3)])
                base = q + (A - Q[0]) * (1 - f) + (B - Q[-1]) * f
            else:
                base = A + f * ab
            ctrl.append(base + bow + offs[k])
        ctrl.append(B)
        return np.array(ctrl), float(np.linalg.norm(ab))

    def _centreline(self, n_st):
        """Centreline through the attachments. In 'target' length mode the attachments
        stay where they are (they are anatomy) and the anterior bow is SOLVED so the
        path length meets the target; nothing else is moved or stretched."""
        p = self.p
        bow0 = float(p["lig.bow_anterior_mm"])

        def make(b):
            ctrl, chord = self._ctrl_points(b)
            C, L = resample(catmull_rom(ctrl, 80), n_st)
            return ctrl, C, L, chord
        ctrl, C, L, chord = make(bow0)
        info = {"mode": p["lig.length_mode"], "attachment_distance_mm": chord, "bow_anterior_mm": bow0,
                "free": ["anterior bow (solved in target mode)", "lateral bow", "control-point offsets",
                         "imported centreline shape"],
                "locked": ["tibial and femoral attachment centres (bone anatomy; move them explicitly)",
                           "bone geometry (never scaled to fit a length)"]}
        if p["lig.length_mode"] == "target":
            target = float(p["lig.length_target_mm"])
            lo_b, hi_b = 0.0, 8.0
            L_lo = make(lo_b)[2]
            L_hi = make(hi_b)[2]
            info["achievable_mm"] = [L_lo, L_hi]
            if target < L_lo - 1e-6:
                self.issues.append({"param": "lig.length_target_mm", "level": "fail",
                                    "message": "target %.1f mm is shorter than the shortest path between the locked attachments (%.1f mm with no anterior bow). "
                                               "Move a footprint centre, or reduce the lateral bow / control offsets." % (target, L_lo)})
                bow = lo_b
            elif target > L_hi + 1e-6:
                self.issues.append({"param": "lig.length_target_mm", "level": "fail",
                                    "message": "target %.1f mm needs more than %.0f mm of anterior bow (max reachable %.1f mm). "
                                               "Move a footprint centre to lengthen the ligament." % (target, hi_b, L_hi)})
                bow = hi_b
            else:
                for _ in range(40):
                    mid = 0.5 * (lo_b + hi_b)
                    if make(mid)[2] < target:
                        lo_b = mid
                    else:
                        hi_b = mid
                bow = 0.5 * (lo_b + hi_b)
            ctrl, C, L, chord = make(bow)
            info["bow_anterior_mm"] = bow
            info["solved"] = "lig.bow_anterior_mm"
        info["length_mm"] = L
        self.length_info = info
        self.ctrl, self.C, self.L = ctrl, C, L

    def _auto_orientation(self):
        """Baseline section orientation: the wide axis turns from the tibial
        footprint's long axis to the femoral footprint's long axis (ribbon lies
        flat against the notch wall). User orientation is added on top."""
        def ang(k, e):
            return math.atan2(float(np.dot(e, self.N2[k])), float(np.dot(e, self.N1[k])))
        a0 = ang(0, self.fp["tib"].e_l)
        a1 = ang(-1, _unit(np.cross(self.T[-1], self.fp["fem"].normal)))
        d = (a1 - a0 + math.pi / 2) % math.pi - math.pi / 2
        a0 = (a0 + math.pi / 2) % math.pi - math.pi / 2
        w = _sstep(self.s)
        return a0 + d * w

    # -- profiles and rotation
    def _rotation(self, s):
        p = self.p
        tot = math.radians(p["bundle.rotation_total_deg"])
        dist = p["bundle.rotation_distribution"]
        if dist == "uniform":
            D = s
        elif dist == "femoral_weighted":
            D = s ** 2
        elif dist == "tibial_weighted":
            D = 1 - (1 - s) ** 2
        else:
            D = profile_eval(p["bundle.rotation_profile"], s)
        return tot * D

    def _taper(self, s):
        p = self.p
        return (1 + float(p["lig.taper_tib"]) * np.clip(1 - 2 * s, 0, 1)
                + float(p["lig.taper_fem"]) * np.clip(2 * s - 1, 0, 1))

    def _profiles(self):
        """Overall section size along the ligament. Width/thickness and
        area/aspect are two parameterizations of the same superellipse; the
        unused pair is derived (never both independent)."""
        p, s = self.p, self.s
        mode = p["lig.section_mode"]
        sc = float(p["lig.size_scale"]) * self._taper(s)
        self.E = profile_eval(p["lig.shape_profile"], s)
        if mode == "area_aspect":
            A = profile_eval(p["lig.area_profile"], s)
            asp = profile_eval(p["lig.aspect_profile"], s)
            g = np.array([_sq_area_factor(e) for e in self.E])
            Tk = np.sqrt(A / (asp * g))
            W = asp * Tk
        else:
            W = profile_eval(p["lig.width_profile"], s)
            Tk = profile_eval(p["lig.thickness_profile"], s)
        self.W = W * sc
        self.Tk = Tk * sc
        self.O_auto = self._auto_orientation()
        self.O = self.O_auto + np.radians(profile_eval(p["lig.orientation_profile"], s))

    def _interp(self, arr, s):
        x = np.clip(s, 0, 1) * (self.s.size - 1)
        i0 = np.clip(np.floor(x).astype(int), 0, self.s.size - 2)
        t = x - i0
        if arr.ndim == 1:
            return arr[i0] * (1 - t) + arr[i0 + 1] * t
        sh = (-1,) + (1,) * (arr.ndim - 1)
        return arr[i0] * (1 - t).reshape(sh) + arr[i0 + 1] * t.reshape(sh)

    # ------------------------------------------------------------ sections
    def _interface_normal_angle(self):
        p = self.p
        b0 = math.radians(p["bundle.boundary_angle_deg"])
        return self.O + b0 + self.theta_b + np.radians(profile_eval(p["bundle.boundary_angle_profile"], self.s))

    def _am_fraction_target(self):
        p = self.p
        if p["bundle.am_fraction_mode"] == "profile":
            return profile_eval(p["bundle.am_fraction_profile"], self.s)
        return np.full(self.s.size, float(p["bundle.am_fraction"]))

    def _irregularity_offsets(self, rb):
        """Radial bumps (mm) on the uniform angle grid WARP_AL for each station, given
        the base radial function rb (n, A). Varies smoothly along s."""
        p = self.p
        amp = float(p["lig.contour_irregularity_mm"])
        if amp <= 0:
            return None
        lobes = int(p["lig.contour_lobes"])
        rng = np.random.default_rng(int(p["lig.contour_seed"]))
        fr = np.array([lobes, lobes + 1, 2 * lobes - 1, max(2, lobes - 1)], float)
        a = np.array([1.0, 0.55, 0.35, 0.45])
        a /= a.sum()
        ph0 = rng.uniform(0, 2 * np.pi, 4)
        ph1 = rng.uniform(0, 2 * np.pi, 4)
        s = self.s[:, None]
        ang = WARP_AL[None]
        d = np.zeros((self.s.size, WARP_AL.size))
        for k in range(4):
            d += a[k] * np.sin(fr[k] * ang + ph0[k] * (1 - s) + ph1[k] * s)
        d *= amp * (0.6 + 0.4 * np.sin(np.pi * s))     # gentler near the insertions
        rmin = rb.min(1)
        if np.any(amp > 0.35 * rmin):
            k = int(np.argmin(rmin))
            self.issues.append({"param": "lig.contour_irregularity_mm", "level": "fail",
                                "message": "irregularity %.2f mm is too large for the %.2f mm section half-thickness at s=%.2f (max 35%%)"
                                           % (amp, rmin[k], self.s[k])})
        return d

    def _superellipse_contours(self, ph):
        a, b, E, O = 0.5 * self.W, 0.5 * self.Tk, self.E, self.O
        c, sn = np.cos(ph)[None], np.sin(ph)[None]
        x = a[:, None] * np.sign(c) * np.abs(c) ** (2 / E[:, None])
        y = b[:, None] * np.sign(sn) * np.abs(sn) ** (2 / E[:, None])
        co, so = np.cos(O)[:, None], np.sin(O)[:, None]
        return np.stack([co * x - so * y, so * x + co * y], -1)

    def _imported_radial(self):
        """Measured contours (3D points, model frame, at given s) projected into the
        station frames, as radial functions about each contour's centroid on the
        uniform angle grid, interpolated along s. Returns (n, A) radii."""
        items = sorted(self.p["lig.section_contours"], key=lambda it: float(it["s"]))
        rad, ss, shifts = [], [], []
        for it in items:
            sv = float(it["s"])
            i = int(round(np.clip(sv, 0, 1) * (self.s.size - 1)))
            P = np.asarray(it["points"], float)
            if P.shape[1] == 3:
                Q = P - self.C[i]
                X = np.stack([Q @ self.N1[i], Q @ self.N2[i]], 1)
            else:
                X = P
            c = _poly_centroid(X)
            rad.append(_radial_one(X - c, WARP_AL))
            ss.append(sv)
            shifts.append({"s": sv, "centroid_offset_mm": [float(c[0]), float(c[1])]})
        rad = np.array(rad)
        ss = np.array(ss)
        out = np.stack([np.interp(self.s, ss, rad[:, k]) for k in range(WARP_AL.size)], 1) if len(ss) > 1 else np.repeat(rad, self.s.size, 0)
        self.contour_shift = shifts
        return out

    def _lobe_sections(self, ph):
        """Per-bundle mode: AM and PL lobes (each its own superellipse) placed on
        either side of the interface direction so they overlap by a set contact
        depth. The envelope is their union (re-centred on the centreline) and the
        interface runs through the two points where the lobe outlines cross."""
        p, s = self.p, self.s
        n = s.size
        K = ph.size
        sc = float(p["lig.size_scale"]) * self._taper(s)
        beta = self._interface_normal_angle()
        nv = np.stack([np.cos(beta), np.sin(beta)], 1)
        pv = np.stack([-nv[:, 1], nv[:, 0]], 1)
        ov = float(p["bundle.lobe_overlap_mm"])
        lobes = []
        for b, key in ((0, "am"), (1, "pl")):
            a = 0.5 * profile_eval(p[key + ".width_profile"], s) * sc
            t = 0.5 * profile_eval(p[key + ".thickness_profile"], s) * sc
            E = float(p[key + ".shape"])
            om = self.O + math.radians(float(p[key + ".orientation_deg"]))
            c, sn = np.cos(ph)[None], np.sin(ph)[None]
            x = a[:, None] * np.sign(c) * np.abs(c) ** (2 / E)
            y = t[:, None] * np.sign(sn) * np.abs(sn) ** (2 / E)
            co, so = np.cos(om)[:, None], np.sin(om)[:, None]
            L0 = np.stack([co * x - so * y, so * x + co * y], -1)        # lobe at origin
            sgn = 1.0 if b == 0 else -1.0
            h = np.einsum("nkd,nd->nk", L0, -sgn * nv).max(1)        # extent toward the interface
            sh = profile_eval(p[key + ".shift_profile"], s)
            cen = (sgn * (h - 0.5 * ov))[:, None] * nv + sh[:, None] * pv
            lobes.append({"a": a, "t": t, "E": E, "om": om, "cen": cen, "poly": L0 + cen[:, None, :]})
        # union as a radial function about the origin (both lobes must contain it)
        dirs = np.stack([np.cos(ph), np.sin(ph)], 1)
        R = []
        for lb in lobes:
            co, so = np.cos(lb["om"])[:, None], np.sin(lb["om"])[:, None]

            def F(tt, lb=lb, co=co, so=so):
                X = tt[..., None] * dirs[None] - lb["cen"][:, None, :]
                u = co * X[..., 0] + so * X[..., 1]
                v = -so * X[..., 0] + co * X[..., 1]
                return np.abs(u / lb["a"][:, None]) ** lb["E"] + np.abs(v / lb["t"][:, None]) ** lb["E"]
            f0 = F(np.zeros((n, K)))
            if np.any(f0 > 1):
                i = int(np.argmax(f0.max(1)))
                self.issues.append({"param": "bundle.lobe_overlap_mm", "level": "fail",
                                    "message": "the AM and PL lobes do not overlap around the centreline at s=%.2f; increase the lobe contact depth or reduce the bundle shift" % s[i]})
            lo = np.zeros((n, K))
            hi = np.full((n, K), 4 * float(max(lb["a"].max(), lb["t"].max())) + 20)
            for _ in range(44):
                mid = 0.5 * (lo + hi)
                ins = F(mid) <= 1
                lo = np.where(ins, mid, lo)
                hi = np.where(ins, hi, mid)
            R.append(0.5 * (lo + hi))
        R = np.array(R)                                   # (2, n, K)
        r = R.max(0)
        owner = np.argmax(R, 0)                           # 0 AM, 1 PL
        X = r[..., None] * dirs[None]
        cen = np.array([_poly_centroid(X[i]) for i in range(n)])
        X = X - cen[:, None, :]
        for lb in lobes:
            lb["poly"] = lb["poly"] - cen[:, None, :]
        # interface through the two owner switches (PL->AM is E-, AM->PL is E+, CCW)
        E_minus = np.zeros((n, 2))
        E_plus = np.zeros((n, 2))
        diff = R[0] - R[1]
        for i in range(n):
            o = owner[i]
            sw = np.nonzero(o != np.roll(o, -1))[0]
            if sw.size != 2:
                self.issues.append({"param": "bundle.lobe_overlap_mm", "level": "fail",
                                    "message": "the AM/PL lobes meet in %d places at s=%.2f (need exactly 2); adjust lobe sizes or orientations" % (sw.size, s[i])})
                sw = np.array([0, K // 2]) if sw.size < 2 else sw[:2]
            for k in sw:
                k1 = (k + 1) % K
                d0, d1 = diff[i, k], diff[i, k1]
                tt = d0 / (d0 - d1) if d0 != d1 else 0.5
                pt = X[i, k] + tt * (X[i, k1] - X[i, k])
                if o[k] == 1 and o[k1] == 0:
                    E_minus[i] = pt
                else:
                    E_plus[i] = pt
        dv = E_plus - E_minus
        nrm = np.stack([dv[:, 1], -dv[:, 0]], 1)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        c = (nrm * E_minus).sum(1)
        self.lobes = lobes
        # overall extents along the section axes (reported, used for sizing)
        co, so = np.cos(self.O)[:, None], np.sin(self.O)[:, None]
        u = co * X[..., 0] + so * X[..., 1]
        v = -so * X[..., 0] + co * X[..., 1]
        self.W = u.max(1) - u.min(1)
        self.Tk = v.max(1) - v.min(1)
        self.E = np.full(n, 2.0)
        return X, nrm, c

    def _build_sections(self):
        """Base section (smooth superellipse, lobe union, or an ellipse matched to the
        measured contour) carries the ruled AM/PL map; irregular or measured outlines
        are applied on top as a RADIAL WARP about the centreline (angle-preserving,
        always one-to-one), so irregular sections can never fold the map."""
        p = self.p
        n = self.s.size
        K = SEC_K
        ph = np.linspace(0, 2 * np.pi, K, endpoint=False)
        mode = p["lig.section_mode"]
        self.lobes = None
        self.contour_shift = None
        r_target = None
        if mode == "per_bundle":
            Xb, nrm, c = self._lobe_sections(ph)
        else:
            if mode == "imported":
                if p["lig.section_contours"]:
                    r_target = self._imported_radial()
                    # base: ellipse with the measured area and principal axes
                    Pt = r_target[..., None] * np.stack([np.cos(WARP_AL), np.sin(WARP_AL)], 1)[None]
                    W, Tk, O = [], [], []
                    for i in range(n):
                        Q = Pt[i]
                        cov = np.cov(Q.T)
                        ev, evec = np.linalg.eigh(cov)
                        A = abs(_poly_area(Q))
                        ratio = math.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9))
                        tk = math.sqrt(4 * A / (math.pi * ratio))
                        W.append(ratio * tk)
                        Tk.append(tk)
                        O.append(math.atan2(evec[1, 1], evec[0, 1]))
                    self.W, self.Tk, self.E = np.array(W), np.array(Tk), np.full(n, 2.0)
                    self.O = np.unwrap(np.array(O) * 2) / 2
                else:
                    self.issues.append({"param": "lig.section_mode", "level": "fail",
                                        "message": "section mode 'imported' needs measured contours (lig.section_contours is empty)"})
            Xb = self._superellipse_contours(ph)
            beta = self._interface_normal_angle()
            nrm = np.stack([np.cos(beta), np.sin(beta)], 1)
            c = None
        # radial warp (irregularity / measured outline)
        rb = np.array([_radial_one(Xb[i], WARP_AL) for i in range(n)])
        if r_target is None:
            off = self._irregularity_offsets(rb)
            r_target = None if off is None else rb + off
        self.warp = None if r_target is None else r_target / np.maximum(rb, 1e-9)
        self.contour_base = Xb
        self.contour = self.warp_points(np.repeat(self.s, K), Xb.reshape(-1, 2)).reshape(n, K, 2) if self.warp is not None else Xb
        self.kappa = float(p["bundle.interface_curvature"])
        f_target = self._am_fraction_target() if c is None else None
        corr = np.zeros(n)
        nt = RULE_K
        tt = np.linspace(0, 1, nt)
        for it in range(3 if c is None else 1):
            if f_target is not None:
                c = self._offset_for_fraction(Xb, nrm, np.clip(f_target + corr, 0.02, 0.98))
            self.iface_n = nrm
            self.iface_c = c
            self.I = np.zeros((n, nt, 2))
            self.B = np.zeros((2, n, nt, 2))
            self.E_ends = np.zeros((n, 2, 2))
            multi = []
            for i in range(n):
                res = _arcs(Xb[i], nrm[i], c[i], nt)
                if res is None:
                    if it == 0:
                        self.issues.append({"param": "bundle.boundary_angle_deg", "level": "fail",
                                            "message": "the AM/PL interface misses the section at s=%.2f" % self.s[i]})
                    res = _arcs(Xb[i], nrm[i], float((Xb[i] @ nrm[i]).mean()), nt)
                Em, Ep, Bam, Bpl, many = res
                if many:
                    multi.append(i)
                Lc = np.linalg.norm(Ep - Em)
                self.I[i] = Em[None] + tt[:, None] * (Ep - Em)[None] + (self.kappa * 4 * tt * (1 - tt) * Lc)[:, None] * nrm[i][None]
                self.B[0, i] = Bam
                self.B[1, i] = Bpl
                self.E_ends[i] = (Em, Ep)
            self._areas()
            if f_target is None or abs(self.kappa) < 1e-9 and self.warp is None:
                break
            corr = corr + (f_target - self.am_fraction_actual)
        if multi:
            self.issues.append({"param": "lig.contour_irregularity_mm", "level": "warn",
                                "message": "the AM/PL interface crosses the contour more than twice at %d stations (non-convex section); the outermost crossings are used" % len(multi)})
        if abs(self.kappa) > 0:
            mid = self.I[:, nt // 2]
            ok = [_pip(Xb[i], mid[i]) for i in range(n)]
            if not all(ok):
                self.issues.append({"param": "bundle.interface_curvature", "level": "fail",
                                    "message": "interface curvature %.2f pushes the AM/PL boundary outside the section" % self.kappa})

    def _areas(self):
        n = self.s.size
        I = self.I
        B = self.B
        if self.warp is not None:
            sk = np.repeat(self.s, I.shape[1])
            I = self.warp_points(sk, I.reshape(-1, 2)).reshape(I.shape)
            B = np.stack([self.warp_points(sk, B[b].reshape(-1, 2)).reshape(B[b].shape) for b in (0, 1)])
        self.I_w, self.B_w = I, B
        self.area = np.abs(_poly_area(self.contour))
        a_am = np.abs(np.array([_poly_area(np.vstack([B[0, i], I[i][::-1]])) for i in range(n)]))
        self.area_bundle = np.stack([a_am, self.area - a_am], 1)
        self.am_fraction_actual = a_am / np.maximum(self.area, 1e-9)
        self.bundle_centroid = np.zeros((n, 2, 2))
        for i in range(n):
            for b in (0, 1):
                self.bundle_centroid[i, b] = _poly_centroid(np.vstack([B[b, i], I[i][::-1]]))

    # ------------------------------------------------------------ radial warp
    def _warp_ratio(self, s, X):
        i0, ws = self._st(s)
        A = WARP_AL.size
        al = np.arctan2(X[:, 1], X[:, 0])
        y = (al - WARP_AL[0]) / (2 * np.pi) * A
        k0 = np.floor(y).astype(int) % A
        k1 = (k0 + 1) % A
        wk = y - np.floor(y)
        W = self.warp
        a = W[i0, k0] * (1 - wk) + W[i0, k1] * wk
        b = W[i0 + 1, k0] * (1 - wk) + W[i0 + 1, k1] * wk
        return a * (1 - ws) + b * ws

    def warp_points(self, s, X):
        if self.warp is None:
            return X
        return X * self._warp_ratio(np.asarray(s, float), X)[:, None]

    def unwarp_points(self, s, Y):
        if self.warp is None:
            return Y
        return Y / self._warp_ratio(np.asarray(s, float), Y)[:, None]

    def _offset_for_fraction(self, X, nrm, f):
        """Interface offset c per station so the AM side (X.n > c) has area fraction f."""
        n, K, _ = X.shape
        M = 128
        g = np.einsum("nkd,nd->nk", X, nrm)
        pv = np.stack([-nrm[:, 1], nrm[:, 0]], 1)
        q = np.einsum("nkd,nd->nk", X, pv)
        lo, hi = g.min(1), g.max(1)
        lev = lo[:, None] + (hi - lo)[:, None] * np.linspace(0, 1, M)[None]
        ga = g[:, None, :] - lev[:, :, None]
        gb = np.roll(ga, -1, axis=2)
        qa = q[:, None, :]
        qb = np.roll(q, -1, axis=1)[:, None, :]
        cr = (ga * gb) <= 0
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(np.abs(ga - gb) > 1e-12, ga / (ga - gb), 0.0)
        y = qa + t * (qb - qa)
        ymax = np.where(cr, y, -np.inf).max(2)
        ymin = np.where(cr, y, np.inf).min(2)
        w = np.where(np.isfinite(ymax) & np.isfinite(ymin), ymax - ymin, 0.0)
        dl = ((hi - lo) / (M - 1))[:, None]
        seg = 0.5 * (w[:, 1:] + w[:, :-1]) * dl
        above = np.concatenate([np.cumsum(seg[:, ::-1], 1)[:, ::-1], np.zeros((n, 1))], 1)
        frac = above / np.maximum(above[:, :1], 1e-12)
        c = np.array([np.interp(f[i], frac[i, ::-1], lev[i, ::-1]) for i in range(n)])
        return c

    # ------------------------------------------------------------ material side
    def _material_arcs(self):
        d = self.chord_d
        c0 = self.chord_c
        pm = np.array([-d[1], d[0]])
        h = math.sqrt(max(0.0, 1 - c0 * c0))
        self.m_em = c0 * d - h * pm
        self.m_ep = c0 * d + h * pm
        self._ang_d = math.atan2(d[1], d[0])
        self._gam = math.acos(max(-1.0, min(1.0, c0)))

    def _m_chord(self, t):
        return self.m_em[None] + t[:, None] * (self.m_ep - self.m_em)[None]

    def _m_arc(self, b, t):
        if b == 0:
            a = self._ang_d - self._gam + t * 2 * self._gam
        else:
            a = self._ang_d - self._gam - t * (2 * np.pi - 2 * self._gam)
        return np.stack([np.cos(a), np.sin(a)], 1)

    def bundle_of(self, m):
        """0 = AM, 1 = PL for material coords (k,2)."""
        return np.where(m @ self.chord_d > self.chord_c, 0, 1).astype(np.int8)

    def material_tu(self, m):
        """Ruled-map coordinates of material points: bundle b, t in [0,1] along the
        chord/arc, u in [0,1] from the chord (interface) to the arc (outer surface)."""
        m = np.asarray(m, float).reshape(-1, 2)
        if m.shape[0] > 4000:
            q = np.round(m * 1e6).astype(np.int64)
            key = (q[:, 0] + (1 << 30)) * (1 << 32) + (q[:, 1] + (1 << 30))
            _, first, inv = np.unique(key, return_index=True, return_inverse=True)
            if first.size < 0.6 * m.shape[0]:
                b, t, u = self.material_tu(m[first])
                inv = inv.ravel()
                return b[inv], t[inv], u[inv]
        b = self.bundle_of(m)
        sg = np.where(b == 0, 1.0, -1.0)
        lo = np.zeros(m.shape[0])
        hi = np.ones(m.shape[0])
        arcs = (lambda t, bb: np.where((bb == 0)[:, None], self._m_arc(0, t), self._m_arc(1, t)))
        for _ in range(34):
            mid = 0.5 * (lo + hi)
            Ch = self._m_chord(mid)
            A = arcs(mid, b)
            f = _cross2(A - Ch, m - Ch) * sg
            lo = np.where(f > 0, mid, lo)
            hi = np.where(f > 0, hi, mid)
        t = 0.5 * (lo + hi)
        Ch = self._m_chord(t)
        A = arcs(t, b)
        den = np.linalg.norm(A - Ch, axis=1)
        u = np.where(den > 1e-9, ((m - Ch) * (A - Ch)).sum(1) / np.maximum(den, 1e-12) ** 2, 0.0)
        return b, t, u

    def material_from_tu(self, b, t, u):
        Ch = self._m_chord(t)
        A = np.where((b == 0)[:, None], self._m_arc(0, t), self._m_arc(1, t))
        return Ch + u[:, None] * (A - Ch)

    # ------------------------------------------------------------ metric side
    def _st(self, s):
        x = np.clip(np.asarray(s, float), 0, 1) * (self.s.size - 1)
        i0 = np.clip(np.floor(x).astype(int), 0, self.s.size - 2)
        return i0, x - i0

    def _curve(self, arr, i0, ws, t):
        """Bilinear lookup of a per-station curve arr (n, nt, 2) at stations (i0, ws) and t."""
        nt = arr.shape[1]
        y = np.clip(t, 0, 1) * (nt - 1)
        k0 = np.clip(np.floor(y).astype(int), 0, nt - 2)
        wt = (y - k0)[:, None]
        a = arr[i0, k0] * (1 - wt) + arr[i0, k0 + 1] * wt
        b = arr[i0 + 1, k0] * (1 - wt) + arr[i0 + 1, k0 + 1] * wt
        return a * (1 - ws)[:, None] + b * ws[:, None]

    def material_to_metric(self, s, m):
        """Body-section coordinates (mm, N1/N2 frame) of material points at s."""
        s = np.broadcast_to(np.asarray(s, float), (np.asarray(m).reshape(-1, 2).shape[0],))
        b, t, u = self.material_tu(m)
        return self._metric_from_tu(s, b, t, u)

    def _metric_from_tu(self, s, b, t, u):
        i0, ws = self._st(s)
        I = self._curve(self.I, i0, ws, t)
        Bam = self._curve(self.B[0], i0, ws, t)
        Bpl = self._curve(self.B[1], i0, ws, t)
        Bv = np.where((b == 0)[:, None], Bam, Bpl)
        return self.warp_points(s, I + u[:, None] * (Bv - I))

    def metric_to_material(self, s, X):
        """Inverse of material_to_metric (points outside the contour get u > 1)."""
        X = np.asarray(X, float).reshape(-1, 2)
        s = np.broadcast_to(np.asarray(s, float), (X.shape[0],))
        X = self.unwarp_points(s, X)
        i0, ws = self._st(s)
        b = np.where(self._iface_base(s, X) > 0, 0, 1).astype(np.int8)
        sg = np.where(b == 0, 1.0, -1.0)
        # per-point curves (station blend done once), then bisection on t
        w = ws[:, None, None]
        Ic = self.I[i0] * (1 - w) + self.I[i0 + 1] * w
        Bsel = np.where((b == 0)[:, None, None], self.B[0][i0] * (1 - w) + self.B[0][i0 + 1] * w,
                        self.B[1][i0] * (1 - w) + self.B[1][i0 + 1] * w)
        nt = Ic.shape[1]
        rows = np.arange(X.shape[0])

        def ev(A, t):
            y = np.clip(t, 0, 1) * (nt - 1)
            k0 = np.clip(np.floor(y).astype(int), 0, nt - 2)
            wt = (y - k0)[:, None]
            return A[rows, k0] * (1 - wt) + A[rows, k0 + 1] * wt
        lo = np.zeros(X.shape[0])
        hi = np.ones(X.shape[0])
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            I = ev(Ic, mid)
            Bv = ev(Bsel, mid)
            f = _cross2(Bv - I, X - I) * sg
            lo = np.where(f > 0, mid, lo)
            hi = np.where(f > 0, hi, mid)
        t = 0.5 * (lo + hi)
        I = ev(Ic, t)
        Bv = ev(Bsel, t)
        den = np.linalg.norm(Bv - I, axis=1)
        u = np.where(den > 1e-9, ((X - I) * (Bv - I)).sum(1) / np.maximum(den, 1e-12) ** 2, 0.0)
        return self.material_from_tu(b, t, u)

    def metric_bundle(self, s, X):
        """Bundle of metric section points (side of the interface curve)."""
        return np.where(self.interface_distance(s, X) > 0, 0, 1).astype(np.int8)

    def interface_distance(self, s, X):
        """Signed distance (mm, + on the AM side) from the AM/PL interface curve."""
        X = np.asarray(X, float).reshape(-1, 2)
        s = np.broadcast_to(np.asarray(s, float), (X.shape[0],))
        if self.warp is None:
            return self._iface_base(s, X)
        Xb = self.unwarp_points(s, X)
        return self._iface_base(s, Xb) * self._warp_ratio(s, Xb)

    def _iface_base(self, s, X):
        i0, ws = self._st(s)
        nrm = self.iface_n[i0] * (1 - ws)[:, None] + self.iface_n[i0 + 1] * ws[:, None]
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        Em = self.E_ends[i0, 0] * (1 - ws)[:, None] + self.E_ends[i0 + 1, 0] * ws[:, None]
        Ep = self.E_ends[i0, 1] * (1 - ws)[:, None] + self.E_ends[i0 + 1, 1] * ws[:, None]
        Lc = np.linalg.norm(Ep - Em, axis=1)
        pv = np.stack([-nrm[:, 1], nrm[:, 0]], 1)
        t = np.clip(((X - Em) * pv).sum(1) / np.maximum(Lc, 1e-9), 0, 1)
        bulge = self.kappa * 4 * t * (1 - t) * Lc
        return ((X - Em) * nrm).sum(1) - bulge

    def contour_at(self, s):
        """Contour polygon (K,2) at s (linear between stations)."""
        i0, ws = self._st(np.array([s]))
        return self.contour[i0[0]] * (1 - ws[0]) + self.contour[i0[0] + 1] * ws[0]

    def body_point(self, s, m):
        s = np.asarray(s, float)
        X = self.material_to_metric(s, m)
        return self.frame_point(s, X)

    def frame_point(self, s, X):
        s = np.asarray(s, float)
        return (self._interp(self.C, s) + X[:, :1] * self._interp(self.N1, s) + X[:, 1:] * self._interp(self.N2, s))

    # ------------------------------------------------------------ flare + footprints
    def blend_weights(self, s, m=None):
        p = self.p
        g = float(p["lig.flare_exponent"])
        dt = s * self.L
        df = (1 - s) * self.L
        ft = float(p["lig.flare_tib_mm"])
        ff = float(p["lig.flare_fem_mm"])
        if m is not None and p["bundle.structure_mode"] == "independent":
            # per-bundle insertion transitions, blended smoothly across the interface
            w = _sstep((np.asarray(m) @ self.chord_d - self.chord_c) / 0.3 + 0.5)
            ft = w * float(p["am.flare_tib_mm"]) + (1 - w) * float(p["pl.flare_tib_mm"])
            ff = w * float(p["am.flare_fem_mm"]) + (1 - w) * float(p["pl.flare_fem_mm"])
        wt = _sstep(1 - dt / ft) ** g
        wf = _sstep(1 - df / ff) ** g
        return wt, wf

    def _fp_warp(self, side, q, inverse=False):
        """Footprint subregions: move the AM/PL split on a footprint to its own AM
        share (a monotone warp of the unit disk along the split direction)."""
        p = self.p
        th = float(self.theta_b[0 if side == "tib" else -1])
        d = rot2(self.chord_d[None], th)[0]
        pv = np.array([-d[1], d[0]])
        c0 = self.chord_c
        c1 = segment_offset_for_fraction(float(p["fp.%s.am_share" % side]))
        if abs(c1 - c0) < 1e-9:
            return q
        a, b = (c1, c0) if inverse else (c0, c1)
        x = q @ d
        y = q @ pv
        x2 = np.where(x <= a, -1 + (x + 1) * (b + 1) / (a + 1), b + (x - a) * (1 - b) / (1 - a))
        sc = np.sqrt(np.clip(1 - x2 ** 2, 0, None)) / np.maximum(np.sqrt(np.clip(1 - x ** 2, 0, None)), 1e-9)
        sc = np.where(np.abs(x) < 0.999999, sc, 1.0)
        y2 = y * sc
        return x2[:, None] * d[None] + y2[:, None] * pv[None]

    def footprint_coords(self, side, q):
        rot = self.align[side] + math.radians(float(self.p["fp.%s.partition_rotation_deg" % side]))
        return rot2(self._fp_warp(side, q), rot)

    def footprint_bundle(self, side, f):
        """Bundle id of footprint chart coordinates f (inverse of footprint_coords)."""
        rot = self.align[side] + math.radians(float(self.p["fp.%s.partition_rotation_deg" % side]))
        q = self._fp_warp(side, rot2(f, -rot), inverse=True)
        th = float(self.theta_b[0 if side == "tib" else -1])
        return self.bundle_of(rot2(q, -th))

    def map(self, s, m, extra_rot=None):
        """Material coords m (k,2) at s (k,) -> 3D points (k,3)."""
        s = np.asarray(s, float)
        m = np.asarray(m, float)
        if extra_rot is not None:
            m = rot2(m, extra_rot)
        P = self.body_point(s, m)
        wt, wf = self.blend_weights(s, m)
        q = rot2(m, self._interp(self.theta_b, s))
        if np.any(wt > 0):
            k = wt > 0
            fpt = self.fp["tib"].points(self.footprint_coords("tib", q[k]))
            P[k] = (1 - wt[k])[:, None] * P[k] + wt[k][:, None] * fpt
        if np.any(wf > 0):
            k = wf > 0
            fpt = self.fp["fem"].points(self.footprint_coords("fem", q[k]))
            P[k] = (1 - wf[k])[:, None] * P[k] + wf[k][:, None] * fpt
        return P

    def _auto_align(self, side):
        """Footprint rotation that best matches the body section where the flare starts."""
        if side == "tib":
            s0 = min(0.45, float(self.p["lig.flare_tib_mm"]) / self.L)
        else:
            s0 = 1 - min(0.45, float(self.p["lig.flare_fem_mm"]) / self.L)
        ph = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        q = np.stack([np.cos(ph), np.sin(ph)], 1)
        th = float(self._interp(self.theta_b, np.array([s0]))[0])
        body = self.body_point(np.full(48, s0), rot2(q, -th))
        wq = self._fp_warp(side, q)
        best, ba = None, 0.0
        for a in np.radians(np.arange(0, 360, 5)):
            fp = self.fp[side].points(rot2(wq, a))
            e = ((fp - body) ** 2).sum()
            if best is None or e < best:
                best, ba = e, a
        return ba

    def _inclinations(self):
        d = self.fp["fem"].center - self.fp["tib"].center
        sag = math.degrees(math.atan2(d[2], abs(d[1])))
        cor = math.degrees(math.atan2(d[2], abs(d[0])))
        return {"sagittal_deg": sag, "coronal_deg": cor, "attachment_distance_mm": float(np.linalg.norm(d))}

    # -- bundles (material)
    def bundle_geometry(self, b):
        S = sunflower(6000)
        sel = S[self.bundle_of(S) == b]
        return sel.mean(0), sel

    def bundle_boundary_radius(self, c, ph, b):
        """Distance from centroid c along direction ph to the bundle boundary (unit disk + chord)."""
        d = np.stack([np.cos(ph), np.sin(ph)], -1)
        cd = d @ c
        tc = -cd + np.sqrt(np.maximum(cd ** 2 - (c @ c - 1), 0))
        nd = d @ self.chord_d
        num = self.chord_c - c @ self.chord_d
        with np.errstate(divide="ignore", invalid="ignore"):
            tl = num / nd
        if b == 0:
            tl = np.where(nd < -1e-12, tl, np.inf)
        else:
            tl = np.where(nd > 1e-12, tl, np.inf)
        return np.minimum(tc, tl)

    def bundle_centerline(self, b, s=None):
        """3D path of a bundle's material centroid (attachment subregion to subregion)."""
        s = self.s if s is None else np.asarray(s, float)
        c, _ = self.bundle_geometry(b)
        return self.map(s, np.repeat(c[None], s.size, 0))

    # -- areas
    def section_area(self, s):
        return self._interp(self.area, np.atleast_1d(np.asarray(s, float)))

    def bundle_area(self, s, b):
        return self._interp(self.area_bundle[:, b], np.atleast_1d(np.asarray(s, float)))

    def ring_scale(self):
        """Per-station linear scale of the mapped section relative to midsubstance (flares included)."""
        ph = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        q = np.stack([np.cos(ph), np.sin(ph)], 1)
        n = self.s.size
        R = self.map(np.repeat(self.s, 40), np.tile(q, (n, 1))).reshape(n, 40, 3)
        c = R.mean(1, keepdims=True)
        A = 0.5 * np.linalg.norm(np.cross(R - c, np.roll(R, -1, 1) - c).sum(1), axis=1)
        amid = A[n // 2]
        return np.sqrt(np.maximum(A, 1e-6) / amid), A

    def section_report(self):
        """Derived section quantities (area is computed from the contour, never set
        independently of it)."""
        k = np.linspace(0, self.s.size - 1, 21).astype(int)
        return {"s": self.s[k].tolist(), "area_mm2": self.area[k].tolist(),
                "width_mm": self.W[k].tolist(), "thickness_mm": self.Tk[k].tolist(),
                "aspect": (self.W[k] / np.maximum(self.Tk[k], 1e-9)).tolist(),
                "am_fraction": self.am_fraction_actual[k].tolist(),
                "am_area_mm2": self.area_bundle[k, 0].tolist(), "pl_area_mm2": self.area_bundle[k, 1].tolist()}


def _sq_area_factor(E):
    g = math.gamma
    return g(1 + 1 / E) ** 2 / g(1 + 2 / E)


def _radial_one(X, al):
    """Radius of a closed outline (star-shaped about the origin) at angles al."""
    ang = np.arctan2(X[:, 1], X[:, 0])
    r = np.linalg.norm(X, axis=1)
    o = np.argsort(ang)
    return np.interp(al, ang[o], r[o], period=2 * np.pi)


def _cross2(a, b):
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def _poly_area(P):
    x, y = P[..., 0], P[..., 1]
    return 0.5 * (x * np.roll(y, -1, -1) - np.roll(x, -1, -1) * y).sum(-1)


def _poly_centroid(P):
    x, y = P[:, 0], P[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    A = 0.5 * cr.sum()
    if abs(A) < 1e-12:
        return P.mean(0)
    return np.array([((x + x1) * cr).sum(), ((y + y1) * cr).sum()]) / (6 * A)


def _pip(poly, x):
    xs, ys = poly[:, 0], poly[:, 1]
    xj, yj = np.roll(xs, -1), np.roll(ys, -1)
    c = ((ys > x[1]) != (yj > x[1])) & (x[0] < (xj - xs) * (x[1] - ys) / (yj - ys + 1e-30) + xs)
    return bool(c.sum() % 2 == 1)


def points_in_polygon(poly, X):
    """Vectorized even-odd test of points X (k,2) against polygon (K,2)."""
    xs, ys = poly[:, 0][None], poly[:, 1][None]
    xj, yj = np.roll(poly[:, 0], -1)[None], np.roll(poly[:, 1], -1)[None]
    px, py = X[:, :1], X[:, 1:]
    c = ((ys > py) != (yj > py)) & (px < (xj - xs) * (py - ys) / (yj - ys + 1e-30) + xs)
    return (c.sum(1) % 2) == 1


def polygon_distance(poly, X):
    """Unsigned distance from points X (k,2) to the polygon outline (K,2)."""
    A = poly[None]
    Bq = np.roll(poly, -1, 0)[None]
    E = Bq - A
    Q = X[:, None, :] - A
    t = np.clip((Q * E).sum(2) / np.maximum((E * E).sum(2), 1e-12), 0, 1)
    D = Q - t[..., None] * E
    return np.sqrt((D * D).sum(2)).min(1)


def _resample_poly(P, nt):
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(d)])
    if cum[-1] <= 1e-12:
        return np.repeat(P[:1], nt, 0)
    t = np.linspace(0, cum[-1], nt)
    return np.stack([np.interp(t, cum, P[:, a]) for a in range(2)], 1)


def _arcs(Pc, n, c, nt):
    """Interface line X.n = c against a CCW contour: endpoints E- (entering the
    AM side, CCW) and E+, and the AM (CCW) / PL (CW) outer arcs from E- to E+,
    each resampled by arc length to nt points."""
    K = Pc.shape[0]
    g = Pc @ n - c
    g1 = np.roll(g, -1)
    up = np.nonzero((g < 0) & (g1 >= 0))[0]
    dn = np.nonzero((g >= 0) & (g1 < 0))[0]
    if not up.size or not dn.size:
        return None
    pv = np.array([-n[1], n[0]])

    def cp(k):
        k1 = (k + 1) % K
        t = g[k] / (g[k] - g1[k])
        return Pc[k] + t * (Pc[k1] - Pc[k])
    cu = [cp(k) for k in up]
    cd = [cp(k) for k in dn]
    iu = int(np.argmin([x @ pv for x in cu]))
    idn = int(np.argmax([x @ pv for x in cd]))
    Em, Ep = cu[iu], cd[idn]
    ku, kd = up[iu], dn[idn]
    idx = (ku + 1 + np.arange((kd - ku) % K)) % K
    am = np.vstack([Em, Pc[idx], Ep])
    idx2 = (ku - np.arange((ku - kd) % K)) % K
    pl = np.vstack([Em, Pc[idx2], Ep])
    return Em, Ep, _resample_poly(am, nt), _resample_poly(pl, nt), (up.size > 1 or dn.size > 1)


# ---------------------------------------------------------------- layouts
def layout_points(pattern, n, inside_fn, margin_fn, seed, centroid, min_spacing=0.0):
    """Deterministic 2D layouts of up to n points in a region.

    Points are never closer than min_spacing; when n do not fit, fewer are
    returned (callers report it)."""
    rng = np.random.default_rng(seed)
    if n <= 0:
        return np.zeros((0, 2))
    if pattern == "random":
        pts = []
        cand_all = rng.uniform(-1, 1, (n * 400, 2))
        cand_all = cand_all[inside_fn(cand_all) & margin_fn(cand_all)]
        if cand_all.shape[0] < n:
            return cand_all
        pts.append(cand_all[np.argmin(np.linalg.norm(cand_all - centroid, axis=1))])
        dmin = np.linalg.norm(cand_all - pts[0], axis=1)
        for _ in range(n - 1):
            i = int(np.argmax(dmin))
            if dmin[i] < min_spacing:
                break
            pts.append(cand_all[i])
            dmin = np.minimum(dmin, np.linalg.norm(cand_all - cand_all[i], axis=1))
        return np.array(pts)
    # hexagonal: bisection on lattice spacing
    off = rng.uniform(0, 1, 2)

    def rings(a):
        out = [np.asarray(centroid, float)[None]]
        nr = int(math.ceil(2.2 / a)) + 1
        for j in range(1, nr):
            r = j * a
            c = int(math.floor(2 * math.pi * r / a))
            ang = np.arange(c) * 2 * math.pi / c + off[0] * 2 * math.pi
            out.append(centroid + r * np.stack([np.cos(ang), np.sin(ang)], 1))
        P = np.vstack(out)
        return P[inside_fn(P) & margin_fn(P)]

    def lattice(a):
        if pattern == "radial":
            return rings(a)
        k = int(math.ceil(2.2 / a)) + 2
        i, j = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1), indexing="ij")
        x = (i + 0.5 * (j % 2) + off[0]) * a
        y = (j + off[1]) * a * math.sqrt(3) / 2
        P = np.stack([x.ravel(), y.ravel()], 1)
        return P[inside_fn(P) & margin_fn(P)]

    lo, hi = max(0.01, min_spacing), 2.0
    best = lattice(lo)
    if best.shape[0] <= n:
        return best
    for _ in range(40):
        a = 0.5 * (lo + hi)
        P = lattice(a)
        if P.shape[0] >= n:
            lo = a
            best = P
        else:
            hi = a
    if best.shape[0] > n:
        d = np.linalg.norm(best - centroid, axis=1)
        best = best[np.argsort(d)[:n]]
    return best


def fiber_offsets(m):
    """Representative fiber positions in a unit fascicle disk (sunflower)."""
    if m <= 1:
        return np.zeros((1, 2))
    k = np.arange(m) + 0.5
    r = np.sqrt(k / m) * 0.82
    a = k * GOLDEN
    return np.stack([r * np.cos(a), r * np.sin(a)], 1)
