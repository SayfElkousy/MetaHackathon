"""Layered ligament scaffold: the six-level engineered construct.

Each level owns its controls (see design.LEVELS and the registry):

  1 Surface   nano-texture and protein coating on designated cell-facing surfaces
  2 Micro     grooves in channel lumens and on the inner sheath surface
  3 Fascicle  hollow core channels (lumen, wall, spacing, shape, path, connections)
  4 Bundle    braided reinforcement mantles around the AM-like and PL-like channel groups
  5 Whole     ribbon-shaped outer sheath: the shared envelope
  6 Ends      flared attachment regions with graded material interfaces

How the geometry is built
-------------------------
Every structure is laid out per station in the ligament's section plane (mm, the
N1/N2 frame of the centreline) and mapped to 3D through the same frames. Inside
each end's flare length the section blends onto the scaffold footprint on bone
(Level 6), so footprint edits propagate into channels, braid and sheath.

Radial stack at a mantle, from the inside out:
    channel group (inside the reinforcement envelope)
    -> braid wall (two strand diameters: over/under families)
    -> sheath clearance -> sheath wall.

Mantle centres follow the AM/PL bundle-region centroids of the anatomy, so their
paths are driven by the attachments and the knee pose, not by the sheath. Braid
angle (Level 4), yarn twist (Level 4) and the ligament's own bundle rotation
(Anatomy and Pose) are independent inputs.

Nothing is shrunk to force a fit: requested values are kept, conflicts are
reported with the measured shortfall and concrete fixes, and only the user
applies a fix. Material properties (degradation, permeability, modulus) are
entered assumptions; the outputs computed from them are labelled as such and
are not validated. Units: mm unless a name says otherwise.
"""
import math
import numpy as np
from scipy.spatial import cKDTree
from acl_engine.v2 import mesher
from acl_engine.v2.ligament import profile_eval, points_in_polygon, polygon_distance

LEVEL_NAMES = {1: "Surface", 2: "Micro", 3: "Fascicle", 4: "Bundle", 5: "Whole ligament", 6: "Ends"}
CHANNEL_RANGE_UM = (150.0, 300.0)       # adjustable proposal, not a validated anatomical requirement
MANTLE_EXP = 2.4                          # reinforcement envelope squareness (rounded)
WATER_VISCOSITY_PA_S = 0.00069            # at 37 C, used only for the labelled Darcy estimate
BAND = ("bone", "interface", "ligament")


def _sstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _unit(v):
    v = np.asarray(v, float)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _rot(X, ang):
    """Rotate 2D points X (..., 2) by angles ang (broadcast over leading dims)."""
    c, s = np.cos(ang), np.sin(ang)
    return np.stack([c * X[..., 0] - s * X[..., 1], s * X[..., 0] + c * X[..., 1]], -1)


def _unwrap_pi(a):
    a = np.asarray(a, float).copy()
    for i in range(1, a.size):
        a[i] -= math.pi * round((a[i] - a[i - 1]) / math.pi)
    return a


def poly_moments(P):
    """Area, centroid, principal angle and second-moment eigenvalues of a polygon."""
    x, y = P[:, 0], P[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    A = 0.5 * cr.sum()
    if abs(A) < 1e-12:
        c = P.mean(0)
        return 0.0, c, 0.0, 0.0, 0.0
    cx = ((x + x1) * cr).sum() / (6 * A)
    cy = ((y + y1) * cr).sum() / (6 * A)
    ixx = ((x * x + x * x1 + x1 * x1) * cr).sum() / 12 / A - cx * cx
    iyy = ((y * y + y * y1 + y1 * y1) * cr).sum() / 12 / A - cy * cy
    ixy = ((x * y1 + 2 * x * y + 2 * x1 * y1 + x1 * y) * cr).sum() / 24 / A - cx * cy
    ang = 0.5 * math.atan2(2 * ixy, ixx - iyy)
    tr, det = ixx + iyy, ixx * iyy - ixy * ixy
    disc = math.sqrt(max(tr * tr / 4 - det, 0.0))
    return abs(A), np.array([cx, cy]), ang, tr / 2 + disc, max(tr / 2 - disc, 1e-12)


def se_points(a, b, n, K):
    """Superellipse outlines by angle parameter: a, b, n (m,) -> (m, K, 2), CCW."""
    t = np.linspace(0, 2 * np.pi, K, endpoint=False)
    c, s = np.cos(t)[None], np.sin(t)[None]
    e = 2.0 / np.asarray(n, float)[:, None]
    x = np.asarray(a, float)[:, None] * np.sign(c) * np.abs(c) ** e
    y = np.asarray(b, float)[:, None] * np.sign(s) * np.abs(s) ** e
    return np.stack([x, y], -1)


def se_value(X, a, b, n):
    """Implicit superellipse norm (< 1 inside) of local points X (m, q, 2); a, b, n (m,)."""
    a = np.maximum(np.asarray(a, float), 1e-9)[:, None]
    b = np.maximum(np.asarray(b, float), 1e-9)[:, None]
    n = np.asarray(n, float)[:, None]
    return (np.abs(X[..., 0] / a) ** n + np.abs(X[..., 1] / b) ** n) ** (1.0 / n)


class ArcTable:
    """Arc-length parametrisation of superellipse outlines (one per station)."""

    def __init__(self, a, b, n, K=384):
        self.P = se_points(a, b, n, K)
        seg = np.linalg.norm(np.roll(self.P, -1, 1) - self.P, axis=2)
        cum = np.concatenate([np.zeros((self.P.shape[0], 1)), np.cumsum(seg, 1)], 1)
        self.per = cum[:, -1]
        self.frac = cum / np.maximum(self.per[:, None], 1e-12)
        T = np.roll(self.P, -1, 1) - np.roll(self.P, 1, 1)
        T /= np.maximum(np.linalg.norm(T, axis=2, keepdims=True), 1e-12)
        self.N = np.stack([T[..., 1], -T[..., 0]], -1)          # outward for CCW outlines

    def at(self, rows, f):
        """Points and outward normals at arc fractions f (q,) on outline `rows` (q,)."""
        f = np.mod(f, 1.0)
        K = self.P.shape[1]
        out = np.zeros((f.size, 2))
        nrm = np.zeros((f.size, 2))
        for r in np.unique(rows):
            k = rows == r
            fr = self.frac[r]
            idx = np.clip(np.searchsorted(fr, f[k], side="right") - 1, 0, K - 1)
            w = (f[k] - fr[idx]) / np.maximum(fr[idx + 1] - fr[idx], 1e-12)
            j = (idx + 1) % K
            out[k] = self.P[r, idx] * (1 - w)[:, None] + self.P[r, j] * w[:, None]
            nn = self.N[r, idx] * (1 - w)[:, None] + self.N[r, j] * w[:, None]
            nrm[k] = nn / np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-12)
        return out, nrm


def _hex_layout(a, b, n, pitch, mx, my, limit=4000):
    """Hexagonal lattice of centres inside the superellipse (a - mx, b - my); best of
    four lattice offsets and two row directions (deterministic). Returns (k, 2) mm and
    the candidate lattice just outside (for honest overflow placement)."""
    ai, bi = a - mx, b - my
    if ai <= 1e-6 or bi <= 1e-6 or pitch <= 0:
        return np.zeros((0, 2)), np.zeros((0, 2))
    h = pitch * math.sqrt(3) / 2
    best, best_out = None, None
    for rows_x in (True, False):
        for ox in (0.0, 0.5):
            for oy in (0.0, 0.5):
                R1 = int(math.ceil((a if rows_x else b) / pitch)) + 2
                R2 = int(math.ceil((b if rows_x else a) / h)) + 2
                if (2 * R1 + 1) * (2 * R2 + 1) > 60 * limit:
                    continue
                i = np.arange(-R1, R1 + 1)
                j = np.arange(-R2, R2 + 1)
                I, J = np.meshgrid(i, j, indexing="ij")
                u = (I + ox + 0.5 * (np.abs(J) % 2)) * pitch
                v = (J + oy) * h
                X = np.stack([u, v], -1).reshape(-1, 2) if rows_x else np.stack([v, u], -1).reshape(-1, 2)
                F = se_value(X[None], np.array([ai]), np.array([bi]), np.array([n]))[0]
                inside = X[F <= 1.0]
                outside = X[(F > 1.0) & (F < 1.0 + 3 * pitch / max(min(ai, bi), 1e-6))]
                if best is None or inside.shape[0] > best.shape[0]:
                    best, best_out = inside, outside[np.argsort(se_value(outside[None], np.array([ai]), np.array([bi]), np.array([n]))[0])]
    if best is None:
        return np.zeros((0, 2)), np.zeros((0, 2))
    r = np.hypot(best[:, 0] / max(a, 1e-9), best[:, 1] / max(b, 1e-9))
    o = np.lexsort((np.arctan2(best[:, 1], best[:, 0]), np.round(r, 6)))
    return best[o][:limit], best_out


def _mix_colors(comp, materials):
    c = np.zeros(3)
    tot = 0.0
    for k, v in comp.items():
        if k in materials:
            c += float(v) * np.asarray(materials[k]["color"], float)
            tot += float(v)
    return c / tot if tot > 0 else np.array([0.8, 0.8, 0.8])


def _normalize(comp):
    comp = {k: float(v) for k, v in (comp or {}).items() if float(v) > 0}
    t = sum(comp.values())
    return {k: v / t for k, v in comp.items()} if t > 0 else {}


def _fmt(x, d=2):
    return ("%%.%df" % d) % x


def _sphere(center, r, nlon=14, nlat=10):
    """Small UV sphere mesh (V, F) for schematic markers."""
    u = np.linspace(0, 2 * math.pi, nlon, endpoint=False)
    v = np.linspace(0, math.pi, nlat)
    U, W = np.meshgrid(u, v, indexing="ij")
    V = np.stack([r * np.sin(W) * np.cos(U), r * np.sin(W) * np.sin(U), r * np.cos(W)], -1)
    V = V.reshape(-1, 3) + np.asarray(center, float)
    F = []
    for i in range(nlon):
        j = (i + 1) % nlon
        for k in range(nlat - 1):
            a, b, c, d = i * nlat + k, j * nlat + k, j * nlat + k + 1, i * nlat + k + 1
            F.append([a, b, c])
            F.append([a, c, d])
    return V, np.asarray(F, int)


# ======================================================================== footprints
class EndFootprint:
    """Scaffold attachment footprint for one end (Level 6). Starts from the anatomical
    footprint (Anatomy and Pose) and applies the Level-6 offset, rotation, outline and
    size; points are projected onto the bone surface."""

    def __init__(self, side, p, base, bone):
        k = "end.%s." % side
        self.side, self.bone, self.base = side, bone, base
        c = base.center + float(p[k + "offset_long_mm"]) * base.e_l + float(p[k + "offset_short_mm"]) * base.e_s
        self.center = bone.project(c[None])[0]
        self.normal = _unit(bone.normal(self.center[None])[0])
        rot = math.radians(float(p[k + "rotation_deg"]))
        el = math.cos(rot) * base.e_l + math.sin(rot) * base.e_s
        self.e_l = _unit(el - (el @ self.normal) * self.normal)
        self.e_s = _unit(np.cross(self.normal, self.e_l))
        if p[k + "size_mode"] == "match_anatomy":
            self.L, self.W = float(base.L), float(base.W)
        else:
            self.L, self.W = float(p[k + "length_mm"]), float(p[k + "width_mm"])
        self.kind = p[k + "outline"]
        if self.kind == "anatomical":
            self.kind = base.shape
        self.Rc = float(base.Rc)
        self.poly = None
        if self.kind == "polygon" and base.poly is not None:
            self.poly = base.poly * np.array([self.L / max(base.L, 1e-9), self.W / max(base.W, 1e-9)])
        elif self.kind == "polygon":
            self.kind = "oval"
        # projection lookup: displacement from the tangent plane to the bone surface on a
        # chart grid (one batched projection; bilinear lookups afterwards)
        R = 0.5 * max(self.L, self.W) + 8.0
        g = np.linspace(-R, R, 57)
        G = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
        Pp = self.center + G[:, :1] * self.e_l + G[:, 1:] * self.e_s
        self._grid = (g, (bone.project(Pp) - Pp).reshape(g.size, g.size, 3))

    def _disp(self, Y):
        g, Dg = self._grid
        h = g[1] - g[0]
        x = np.clip((Y[:, 0] - g[0]) / h, 0, g.size - 1.000001)
        y = np.clip((Y[:, 1] - g[0]) / h, 0, g.size - 1.000001)
        i, j = np.floor(x).astype(int), np.floor(y).astype(int)
        tx, ty = (x - i)[:, None], (y - j)[:, None]
        return (Dg[i, j] * (1 - tx) * (1 - ty) + Dg[i + 1, j] * tx * (1 - ty) +
                Dg[i, j + 1] * (1 - tx) * ty + Dg[i + 1, j + 1] * tx * ty)

    def chart(self, uv):
        u, v = uv[:, 0], uv[:, 1]
        if self.kind == "oval":
            return np.stack([0.5 * self.L * u, 0.5 * self.W * v], 1)
        if self.kind == "rounded_rect":
            r = np.hypot(u, v)
            ph = np.arctan2(v, u)
            c, s = np.cos(ph), np.sin(ph)
            return np.stack([0.5 * self.L * r * np.sign(c) * np.abs(c) ** 0.5, 0.5 * self.W * r * np.sign(s) * np.abs(s) ** 0.5], 1)
        if self.kind == "crescent":
            span = self.L / self.Rc
            al = u * span / 2
            rr = self.Rc + v * self.W / 2
            return np.stack([rr * np.sin(al), self.Rc - rr * np.cos(al)], 1)
        P = self.poly
        c = P.mean(0)
        rho = np.hypot(u, v)
        ph = np.arctan2(v, u)
        d = np.stack([np.cos(ph), np.sin(ph)], 1)
        A = P - c
        Bq = np.roll(P, -1, 0) - c
        E = Bq - A
        det = d[:, None, 0] * (-E[None, :, 1]) - d[:, None, 1] * (-E[None, :, 0])
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        t = (A[None, :, 0] * (-E[None, :, 1]) - A[None, :, 1] * (-E[None, :, 0])) / det
        w = (d[:, None, 0] * A[None, :, 1] - d[:, None, 1] * A[None, :, 0]) / det
        t = np.where((w >= -1e-9) & (w <= 1 + 1e-9) & (t > 0), t, np.inf)
        rb = t.min(1)
        rb = np.where(np.isfinite(rb), rb, 0.0)
        return c + (rho * rb)[:, None] * d

    def outline(self, K=128):
        a = np.linspace(0, 2 * np.pi, K, endpoint=False)
        return self.chart(np.stack([np.cos(a), np.sin(a)], 1))

    def planar(self, Y):
        return self.center + Y[:, :1] * self.e_l + Y[:, 1:] * self.e_s

    def to3d(self, Y):
        return self.planar(Y) + self._disp(Y)

    def area(self):
        o = self.outline(256)
        return float(abs(0.5 * (o[:, 0] * np.roll(o[:, 1], -1) - np.roll(o[:, 0], -1) * o[:, 1]).sum()))


# ======================================================================== the construct
class LayeredScaffold:
    def __init__(self, p, lig, bones, quality="full", injury=None, materials=None, prov=None):
        self.p, self.lig, self.quality = p, lig, quality
        self.fem, self.tib = bones
        self.materials = materials or {}
        self.prov = prov or {}
        self.active = p["scaffold.arch"] == "layered" and (injury is None or injury.active)
        self.conflicts = []
        self.derived = {}
        self.derived_profiles = {}
        self.calc = {k: [] for k in range(1, 7)}
        self.assume = {k: [] for k in range(1, 7)}
        self.summary = {k: {} for k in range(1, 7)}
        self.objects = []
        self.end_info = {}
        if not self.active:
            return
        self._read()
        self._stations()
        self._sheath_reference()
        self._mantles()
        if self.p["sheath.size_mode"] == "fit_contents":
            self._sheath_fit_contents()
        self._finish_sheath()
        self._ends_setup()
        self._channels()
        self._build_3d()
        self._checks()
        self._outputs()

    # ------------------------------------------------------------------ inputs
    def _read(self):
        p = self.p
        um = 1e-3
        self.lum = float(p["ch.lumen_um"]) * um
        self.cwall = float(p["ch.wall_um"]) * um
        self.cgap = float(p["ch.spacing_um"]) * um
        self.cshape = p["ch.shape"]
        self.casp = 1.0 if self.cshape == "circular" else float(p["ch.aspect"])
        self.cexp = {"circular": 2.0, "elliptical": 2.0, "rounded_square": 4.0}[self.cshape]
        self.lum_w, self.lum_h = self.lum, self.lum / self.casp
        self.od_w, self.od_h = self.lum_w + 2 * self.cwall, self.lum_h + 2 * self.cwall
        self.od_max = max(self.od_w, self.od_h)
        self.pitch = self.od_max + self.cgap
        loc = p["sf.location"]
        self.coat_ch = loc in ("channels", "channels_and_inner_sheath")
        self.coat_sh = loc in ("inner_sheath", "channels_and_inner_sheath")
        self.coat_t = float(p["sf.coating_nm"]) * 1e-6
        self.coat_explicit = p["sf.coating_explicit"] == "explicit"
        self.coat_geom = self.coat_t if self.coat_explicit else 0.0
        gp = p["gr.placement"]
        self.gr_ch = gp in ("channels", "both")
        self.gr_sh = gp in ("inner_sheath", "both")
        self.tw = float(p["sheath.wall_mm"])
        self.cs = float(p["sheath.clearance_mm"])
        self.wt = float(p["rf.transition_mm"])

    # ------------------------------------------------------------------ stations
    def _stations(self):
        lig = self.lig
        self.s = lig.s
        self.n = lig.s.size
        self.L = lig.L
        n = self.n
        c0 = np.zeros((n, 2))
        ang = np.zeros(n)
        for i in range(n):
            _, c, a, _, _ = poly_moments(lig.contour[i])
            c0[i], ang[i] = c, a
        ang = _unwrap_pi(ang)
        u = np.stack([np.cos(ang), np.sin(ang)], 1)
        v = np.stack([-np.sin(ang), np.cos(ang)], 1)
        Q = lig.contour - c0[:, None]
        pu = (Q * u[:, None]).sum(2)
        pv = (Q * v[:, None]).sum(2)
        self.c0, self.ang_anat = c0, ang
        self.W_env = pu.max(1) - pu.min(1)
        self.T_env = pv.max(1) - pv.min(1)
        self.mid = n // 2

    def _interp(self, arr, s):
        return self.lig._interp(arr, np.asarray(s, float))

    # ------------------------------------------------------------------ Level 5 (reference) + Level 4
    def _sheath_reference(self):
        p, s = self.p, self.s
        mode = p["sheath.size_mode"]
        self.th = self.ang_anat + np.radians(profile_eval(p["sheath.orientation_profile"], s))
        self.nx = np.clip(profile_eval(p["sheath.flattening_profile"], s), 1.6, 6.0)
        if mode == "profiles":
            self.A = 0.5 * profile_eval(p["sheath.width_profile"], s)
            self.B = 0.5 * profile_eval(p["sheath.thickness_profile"], s)
        else:
            f = float(p["sheath.anatomy_fraction"]) if mode == "fit_anatomy" else 1.0
            self.A = 0.5 * f * self.W_env
            self.B = 0.5 * f * self.T_env

    def _iface_lin(self):
        """Per-station linearised AM/PL interface: (point, normal into AM)."""
        lig = self.lig
        Em = lig.E_ends[:, 0]
        nrm = lig.iface_n
        return Em, nrm

    def _mantles(self):
        p, lig, n, s = self.p, self.lig, self.n, self.s
        Em, nrm = self._iface_lin()
        self.mant = []
        auto = p["rf.size_mode"] == "auto"
        lam = self.L / 4.0
        for b, key in ((0, "am"), (1, "pl")):
            d = float(p["rf.%s.strand_diameter_mm" % key])
            Nst = int(p["rf.%s.strands" % key])
            slack = float(p["rf.%s.slack_pct" % key]) / 100.0
            As = lam / math.pi * math.sqrt(max(slack, 0.0))
            cb = np.zeros((n, 2))
            thb = np.zeros(n)
            a0 = np.zeros(n)
            b0 = np.zeros(n)
            for i in range(n):
                RB = np.vstack([lig.B_w[b, i], lig.I_w[i][::-1]])
                _, c, a, l1, l2 = poly_moments(RB)
                cb[i], thb[i] = c, a
                a0[i], b0[i] = 2 * math.sqrt(l1), 2 * math.sqrt(l2)
            thb = _unwrap_pi(thb)
            # keep the mantle's wide axis close to the sheath's (no spurious half turns)
            thb = thb - math.pi * np.round((thb - self.th) / math.pi)
            sgn = 1.0 if b == 0 else -1.0
            own = sgn * nrm
            ub = np.stack([np.cos(thb), np.sin(thb)], 1)
            vb = np.stack([-np.sin(thb), np.cos(thb)], 1)
            m = dict(key=key, bi=b, d=d, N=Nst, m=Nst // 2, alpha=math.radians(float(p["rf.%s.braid_angle_deg" % key])),
                     pattern=p["rf.%s.pattern" % key], slack=slack, As=As, lam=lam, u=ub, v=vb, th=thb, a0=a0, b0=b0, own=own,
                     twist=float(p["rf.%s.yarn_twist_deg" % key]))
            # the fitted (auto) envelope: largest that keeps the sheath clearance and the transition band
            k, delta = self._auto_mantle(cb, thb, a0, b0, own, d, As, vb, Em, nrm, sgn)
            m["c"] = cb + (delta * b0)[:, None] * own
            ao, bo = k * a0, k * self._last_asp * b0
            m["shift"] = float(np.mean(delta * b0))
            if auto:
                m["a"], m["b"] = ao - 2 * d, bo - 2 * d
            else:
                # manual: the user sets the mid-length size; the shape along the length follows the
                # fitted profile, so switching auto -> manual with the calculated values keeps the geometry
                W = float(p["rf.%s.envelope_width_mm" % key])
                T = float(p["rf.%s.envelope_thickness_mm" % key])
                fa, fb = np.maximum(ao - 2 * d, 1e-6), np.maximum(bo - 2 * d, 1e-6)
                m["a"], m["b"] = 0.5 * W * fa / fa[self.mid], 0.5 * T * fb / fb[self.mid]
            m["ao"], m["bo"] = m["a"] + 2 * d, m["b"] + 2 * d
            m["cb_anat"] = cb
            self.mant.append(m)
        self.derived["rf.am.envelope_width_mm"] = float(2 * self.mant[0]["a"][self.mid])
        self.derived["rf.am.envelope_thickness_mm"] = float(2 * self.mant[0]["b"][self.mid])
        self.derived["rf.pl.envelope_width_mm"] = float(2 * self.mant[1]["a"][self.mid])
        self.derived["rf.pl.envelope_thickness_mm"] = float(2 * self.mant[1]["b"][self.mid])

    def _auto_mantle(self, cb, thb, a0, b0, own, d, As, vb, Em, nrm, sgn):
        """Largest scale of the bundle region's equivalent ellipse whose braid OUTER surface
        (incl. slack excursion) keeps the sheath clearance and stays out of the AM/PL
        transition band; vectorised bisection over stations, then a feasible smoothing."""
        n = self.n
        K = 72
        unit = se_points(np.ones(n), np.ones(n), np.full(n, MANTLE_EXP), K)
        fit_contents = self.p["sheath.size_mode"] == "fit_contents"
        if fit_contents:
            lig = self.lig
            Ac = 0.5 * lig.W - self.tw - self.cs
            Bc = 0.5 * lig.Tk - self.tw - self.cs
            thc, nc, cc = lig.O, lig.E, np.zeros((n, 2))
        else:
            Ac = self.A - self.tw - self.cs
            Bc = self.B - self.tw - self.cs
            thc, nc, cc = self.th, self.nx, self.c0
        half_band = 0.5 * self.wt
        best = None
        for asp in (0.8, 1.0, 1.25, 1.5):
            for frac in (0.0, 0.1, 0.2, 0.3):
                ctr = cb + (frac * b0)[:, None] * own
                lo = np.full(n, 0.05)
                hi = np.full(n, 1.6)
                for _ in range(16):
                    k = 0.5 * (lo + hi)
                    ok = self._mantle_ok(ctr, thb, k * a0, k * asp * b0, unit, As, vb, cc, thc, Ac, Bc, nc, Em, nrm, sgn, half_band)
                    lo = np.where(ok, k, lo)
                    hi = np.where(ok, hi, k)
                score = float(np.mean(lo * lo * asp * a0 * b0))
                if best is None or score > best[0] + 1e-9:
                    best = (score, lo.copy(), frac, asp)
        _, k, frac, asp = best
        self._last_asp = asp
        # feasible smoothing: erode, then average inside the erosion radius
        w = 4
        kp = np.pad(k, w, mode="edge")
        er = np.array([kp[i:i + 2 * w + 1].min() for i in range(n)])
        g = np.exp(-0.5 * (np.arange(-w, w + 1) / 2.0) ** 2)
        g /= g.sum()
        ep = np.pad(er, w, mode="edge")
        ks = np.array([(ep[i:i + 2 * w + 1] * g).sum() for i in range(n)])
        return np.minimum(ks, k), np.full(n, frac)

    def _mantle_ok(self, ctr, th, ao, bo, unit, As, vb, cc, thc, Ac, Bc, nc, Em, nrm, sgn, half_band):
        Q = _rot(unit * np.stack([ao, bo], 1)[:, None, :], th[:, None]) + ctr[:, None, :]
        ok = np.ones(ctr.shape[0], bool)
        for sh in ((-As, As) if As > 0 else (0.0,)):
            X = Q + sh * vb[:, None, :]
            loc = _rot(X - cc[:, None, :], -thc[:, None])
            F = se_value(loc, np.maximum(Ac, 1e-6), np.maximum(Bc, 1e-6), nc)
            ok &= (F <= 1.0).all(1)
            di = ((X - Em[:, None, :]) * nrm[:, None, :]).sum(2) * sgn
            ok &= (di >= half_band).all(1)
        return ok & (Ac > 0) & (Bc > 0)

    def _sheath_fit_contents(self):
        """Smallest sheath (anatomical aspect) that holds both mantles with clearance."""
        n = self.n
        K = 72
        unit = se_points(np.ones(n), np.ones(n), np.full(n, MANTLE_EXP), K)
        pts = []
        for m in self.mant:
            Q = _rot(unit * np.stack([m["ao"], m["bo"]], 1)[:, None, :], m["th"][:, None]) + m["c"][:, None, :]
            for sh in ((-m["As"], m["As"]) if m["As"] > 0 else (0.0,)):
                pts.append(Q + sh * m["v"][:, None, :])
        X = np.concatenate(pts, 1)
        loc = _rot(X - self.c0[:, None, :], -self.th[:, None])
        lo = np.full(n, 0.1)
        hi = np.full(n, 3.0)
        for _ in range(22):
            k = 0.5 * (lo + hi)
            A = 0.5 * k * self.W_env - self.tw - self.cs
            B = 0.5 * k * self.T_env - self.tw - self.cs
            ok = (A > 0) & (B > 0) & (se_value(loc, np.maximum(A, 1e-6), np.maximum(B, 1e-6), self.nx) <= 1.0).all(1)
            hi = np.where(ok, k, hi)
            lo = np.where(ok, lo, k)
        k = hi
        w = 4
        kp = np.pad(k, w, mode="edge")
        dl = np.array([kp[i:i + 2 * w + 1].max() for i in range(n)])
        g = np.exp(-0.5 * (np.arange(-w, w + 1) / 2.0) ** 2)
        g /= g.sum()
        dp = np.pad(dl, w, mode="edge")
        ks = np.maximum(np.array([(dp[i:i + 2 * w + 1] * g).sum() for i in range(n)]), k)
        self.A = 0.5 * ks * self.W_env
        self.B = 0.5 * ks * self.T_env

    def _finish_sheath(self):
        self.Ai = self.A - self.tw
        self.Bi = self.B - self.tw
        k = np.linspace(0, self.n - 1, 5).astype(int)
        self.derived_profiles["sheath.width_profile"] = (2 * self.A[k]).round(3).tolist()
        self.derived_profiles["sheath.thickness_profile"] = (2 * self.B[k]).round(3).tolist()
        self.derived["sheath.width_profile"] = float(2 * self.A[self.mid])
        self.derived["sheath.thickness_profile"] = float(2 * self.B[self.mid])
        # slack undulation of each mantle (moves its whole channel group with it)
        sig = self.s * self.L
        for m in self.mant:
            m["off"] = m["As"] * np.sin(2 * np.pi * sig / m["lam"])
            m["cc"] = m["c"] + m["off"][:, None] * m["v"]

    # ------------------------------------------------------------------ Level 3
    def _span(self, extra=10):
        """Stations for paths: the body stations plus a short extension past each end, so
        every point of an obliquely attached section can reach the bone."""
        s_lo, s_hi = self.ends["tib"]["s_ext"], self.ends["fem"]["s_ext"]
        return 0.0, 1.0, np.unique(np.concatenate([self.s, np.linspace(0, self.ends["tib"]["s_fs"], extra),
                                                   np.linspace(self.ends["fem"]["s_fs"], 1, extra)]))

    def _wave(self, ss):
        p = self.p
        sig = ss * self.L
        s0, s1 = sig[0], sig[-1]
        span = max(s1 - s0, 1e-6)
        Aw = float(p["ch.waviness_um"]) * 1e-3
        lam = float(p["ch.wavelength_mm"])
        bow = float(p["ch.curvature_um"]) * 1e-3
        ramp = max(min(lam / 2, span / 4), 1e-6)
        win = _sstep((sig - s0) / ramp) * _sstep((s1 - sig) / ramp)
        return Aw * np.sin(2 * np.pi * (sig - s0) / lam) * win + bow * np.sin(np.pi * (sig - s0) / span), Aw + abs(bow)

    def _channels(self):
        p = self.p
        st, sf, ss = self._span()
        self.ch_s = ss
        wave, wmar = self._wave(ss)
        self.wave = wave
        self.wave_margin = wmar
        span_idx = np.ones(self.n, bool)
        count_mode = p["ch.count_mode"]
        chans = []
        self.group_info = []
        rx = self.od_w / 2 + self.cgap
        ry = self.od_h / 2 + self.cgap
        for m in self.mant:
            a_min = float(m["a"][span_idx].min())
            b_min = float(m["b"][span_idx].min())
            # wide axis of the channel follows the host's wide axis
            inside, outside = _hex_layout(a_min, b_min, MANTLE_EXP, self.pitch, rx + wmar, ry)
            fit = inside.shape[0]
            req = int(p["ch.%s_count" % m["key"]]) if count_mode == "count" else fit
            if req <= fit:
                pts, bad = inside[:req], np.zeros(req, bool)
            else:
                extra = outside[:req - fit]
                pts = np.vstack([inside, extra])
                bad = np.concatenate([np.zeros(fit, bool), np.ones(extra.shape[0], bool)])
            self.group_info.append({"key": m["key"], "fit": fit, "requested": req, "placed": int(pts.shape[0]),
                                    "a_min": a_min, "b_min": b_min})
            self.derived["ch.%s_count" % m["key"]] = fit
            for k in range(pts.shape[0]):
                chans.append({"group": m["bi"], "xi": pts[k, 0] / a_min, "eta": pts[k, 1] / b_min, "over": bool(bad[k])})
        # shared channels in the transition band between the mantles
        self.shared_fit = 0
        if p["ch.transition_channels"] == "fill":
            chans += self._shared_channels(ss, wave, wmar)
        self.chans = chans
        # 2D paths per channel (section coordinates)
        X = np.zeros((len(chans), ss.size, 2))
        U = np.zeros((len(chans), ss.size, 2))
        for k, c in enumerate(chans):
            X[k], U[k] = self._chan_xy(c, ss, wave)
        self.ch_X, self.ch_U = X, U

    def _host(self, group, ss):
        if group in (0, 1):
            m = self.mant[group]
            return (self._interp(m["cc"], ss), self._interp(m["th"], ss), self._interp(m["a"], ss), self._interp(m["b"], ss))
        return (self._interp(self.c0, ss), self._interp(self.th, ss), self._interp(self.Ai, ss), self._interp(self.Bi, ss))

    def _chan_xy(self, c, ss, wave):
        ctr, th, a, b = self._host(c["group"], ss)
        loc = np.stack([c["xi"] * a + wave, c["eta"] * b], 1)
        X = ctr + _rot(loc, th)
        return X, np.stack([np.cos(th), np.sin(th)], 1)

    def _shared_channels(self, ss, wave, wmar):
        Ai = self._interp(self.Ai, ss)
        Bi = self._interp(self.Bi, ss)
        A_min, B_min = float(Ai.min()), float(Bi.min())
        mx = self.od_max / 2 + self.cs + wmar
        cand, _ = _hex_layout(A_min, B_min, 2.0, self.pitch, mx, self.od_max / 2 + self.cs, limit=20000)
        if not cand.shape[0]:
            return []
        xi, eta = cand[:, 0] / A_min, cand[:, 1] / B_min
        ctr = self._interp(self.c0, ss)
        th = self._interp(self.th, ss)
        loc = np.stack([xi[:, None] * Ai[None] + wave[None], eta[:, None] * Bi[None]], -1)      # (c, q, 2)
        X = ctr[None] + _rot(loc, th[None])
        keep = np.ones(cand.shape[0], bool)
        Em, nrm = self._iface_lin()
        Em_s, nrm_s = self._interp(Em, ss), self._interp(nrm, ss)
        di = ((X - Em_s[None]) * nrm_s[None]).sum(2)
        keep &= (np.abs(di) <= 0.5 * self.wt + 0.5 * self.od_max + 1e-9).all(1)
        for m in self.mant:
            c = self._interp(m["cc"], ss)
            thb = self._interp(m["th"], ss)
            ao = self._interp(m["ao"], ss) + self.od_max / 2 + self.cgap
            bo = self._interp(m["bo"], ss) + self.od_max / 2 + self.cgap
            lb = _rot(X - c[None], -thb[None])
            F = (np.abs(lb[..., 0] / ao[None]) ** MANTLE_EXP + np.abs(lb[..., 1] / bo[None]) ** MANTLE_EXP) ** (1 / MANTLE_EXP)
            keep &= (F >= 1.0).all(1)
        out = [{"group": 2, "xi": float(xi[k]), "eta": float(eta[k]), "over": False} for k in np.nonzero(keep)[0]]
        self.shared_fit = len(out)
        return out

    # ------------------------------------------------------------------ Level 6
    def _ends_setup(self):
        p, lig = self.p, self.lig
        self.fp = {}
        self.ends = {}
        for side, bone in (("tib", self.tib), ("fem", self.fem)):
            k = "end.%s." % side
            fp = EndFootprint(side, p, lig.fp[side], bone)
            self.fp[side] = fp
            Lf = min(float(p[k + "flare_length_mm"]), 0.45 * self.L)
            s_fs = Lf / self.L if side == "tib" else 1 - Lf / self.L
            i_fs = int(round(s_fs * (self.n - 1)))
            outline = fp.outline(128)
            oc = outline.mean(0)
            nf = float(self.nx[0 if side == "tib" else -1])
            fill = p[k + "flare_mode"] == "fill_footprint"
            # fill: landing points follow the footprint's own chart (a crescent footprint gets a
            # curved flare base), scaled so the superellipse corners stay 8% inside the outline
            kfill = 0.92 / 2 ** (0.5 - 1.0 / nf)
            if fill:
                Wf, Tf = kfill * fp.L, kfill * fp.W
            else:
                Wf, Tf = float(p[k + "flare_width_mm"]), float(p[k + "flare_thickness_mm"])
            self.derived[k + "flare_width_mm"] = Wf
            self.derived[k + "flare_thickness_mm"] = Tf
            self.derived[k + "length_mm"] = fp.L
            self.derived[k + "width_mm"] = fp.W
            th = float(self.th[i_fs])
            U3 = math.cos(th) * lig.N1[i_fs] + math.sin(th) * lig.N2[i_fs]
            V3 = -math.sin(th) * lig.N1[i_fs] + math.cos(th) * lig.N2[i_fs]
            phi = 0.0 if U3 @ fp.e_l >= 0 else math.pi
            hand = 1.0 if np.cross(U3, V3) @ fp.normal >= 0 else -1.0
            # flare twist: sheath wide axis at the flare start vs the footprint long axis
            u_pl = _unit(U3 - (U3 @ fp.normal) * fp.normal)
            el = math.cos(phi) * fp.e_l + math.sin(phi) * fp.e_s
            twist = math.degrees(math.acos(max(-1.0, min(1.0, float(u_pl @ el)))))
            E1 = el
            E2 = hand * (-math.sin(phi) * fp.e_l + math.cos(phi) * fp.e_s)
            Ff = np.stack([E1, E2, np.cross(E1, E2)], 1)            # columns: footprint frame (right-handed)
            self.ends[side] = dict(Ff=Ff, Cf=fp.planar(oc[None])[0], nrm=fp.normal, fill=fill, kfill=kfill,
                                   Lf=Lf, s_fs=s_fs, i_fs=i_fs, gamma=float(p[k + "curvature"]), Wf=Wf, Tf=Tf,
                                   phi=phi, hand=hand, oc=oc, outline=outline, nf=nf, twist_deg=twist,
                                   central=p[k + "endpoints"] == "central", staggered=p[k + "endpoints"] == "staggered",
                                   A_fs=float(self.A[i_fs]), B_fs=float(self.B[i_fs]),
                                   Lb=float(p[k + "bone_zone_mm"]), Lg=float(p[k + "gradient_mm"]),
                                   stop=float(p[k + "endpoint_stop_mm"]))
            # extension past the end so every point of an oblique section reaches the bone
            s_end = 0.0 if side == "tib" else 1.0
            Xo, _, _ = self._ring("outer", np.array([s_end]), 64)
            Pb = self._body(np.full(64, s_end), Xo[0])
            d = (Pb - self.ends[side]["Cf"]) @ fp.normal
            T = lig.T[0] if side == "tib" else lig.T[-1]
            toward = -T if side == "tib" else T
            rate = max(0.25, float(toward @ -fp.normal))
            self.ends[side]["s_ext"] = 0.0 if side == "tib" else 1.0

    def _body(self, s, X):
        """Body-section point (no flare); beyond the ends the end section is extended along
        the end tangent."""
        lig = self.lig
        s = np.asarray(s, float)
        P = lig.frame_point(np.clip(s, 0, 1), X)
        lo, hi = s < 0, s > 1
        if lo.any():
            P[lo] += (s[lo] * self.L)[:, None] * lig.T[0]
        if hi.any():
            P[hi] += ((s[hi] - 1) * self.L)[:, None] * lig.T[-1]
        return P

    def _map(self, s, X, central=False, U2=None):
        """Section coords X (m,2) at s (m,) -> 3D with the Level-6 flares.

        Every point has its own landing point on the footprint (its normalised section
        position mapped onto the flare base, then onto the bone surface). It flares by its
        OWN remaining distance d to that landing point, measured along the construct axis:
        w = smoothstep(1 - d / flare length) ** curvature, and P = (1 - w) body + w landing.
        Points on the side of a ribbon whose landing point lies further back therefore
        insert earlier along the bone (a staggered, fan-like attachment) and neighbouring
        paths stay parallel instead of shearing across each other; this matters at the
        femur, where the notch wall runs alongside the ligament. Returns P, directions (if
        U2 is given) and each point's remaining distance d to bone, per end."""
        lig = self.lig
        s = np.asarray(s, float)
        X = np.asarray(X, float)
        Pb = self._body(s, X)
        P = Pb.copy()
        sc = np.clip(s, 0, 1)
        D = None
        if U2 is not None:
            D = U2[:, :1] * lig._interp(lig.N1, sc) + U2[:, 1:] * lig._interp(lig.N2, sc)
        dist = {}
        for side, half in (("tib", s < 0.5), ("fem", s >= 0.5)):
            e = self.ends[side]
            ds = sc * self.L if side == "tib" else (1 - sc) * self.L
            dist[side] = np.where(half, ds, np.inf)
            k = half & (ds < e["Lf"])
            if not k.any():
                continue
            sk = sc[k]
            th = self._interp(self.th, sk)
            loc = _rot(X[k] - self._interp(self.c0, sk), -th)
            xi = loc[:, 0] / self._interp(self.A, sk)
            eta = loc[:, 1] / self._interp(self.B, sk)
            Y = self._landing(side, xi, eta, central)
            Pf = self.fp[side].to3d(Y)
            d = ds[k]
            w = _sstep(1 - d / max(e["Lf"], 1e-6)) ** e["gamma"]
            P[k] = (1 - w)[:, None] * Pb[k] + w[:, None] * Pf
            if D is not None:
                loc_u = _rot(U2[k], -th)
                Yd = _rot(np.stack([loc_u[:, 0], e["hand"] * loc_u[:, 1]], 1), e["phi"])
                Df = Yd[:, :1] * self.fp[side].e_l + Yd[:, 1:] * self.fp[side].e_s
                D[k] = (1 - w)[:, None] * D[k] + w[:, None] * Df
        if D is not None:
            D = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
        return P, D, dist

    def _landing(self, side, xi, eta, central=False):
        """Footprint-chart landing point (mm) of normalised section coords (xi, eta)."""
        e = self.ends[side]
        if central and e["central"]:
            return _rot(np.stack([xi * e["A_fs"], e["hand"] * eta * e["B_fs"]], 1), e["phi"]) + e["oc"]
        # land through the footprint's own chart (a crescent footprint gets a curved base); the
        # base spans Wf along the footprint's long axis and Tf across it, in both flare modes
        fp = self.fp[side]
        k = np.array([e["Wf"] / max(fp.L, 1e-9), e["Tf"] / max(fp.W, 1e-9)])
        return fp.chart(_rot(np.stack([xi, e["hand"] * eta], 1), e["phi"]) * k)

    def _flare_base(self, side, K=96):
        e = self.ends[side]
        u = se_points(np.ones(1), np.ones(1), np.array([e["nf"]]), K)[0]
        return self._landing(side, u[:, 0], e["hand"] * u[:, 1])

    def _p3(self, s, X, central=False):
        return self._map(s, X, central)[0]

    # ------------------------------------------------------------------ material field
    def _end_weights(self, side, dd):
        """Zone weights (bone-facing, interface, ligament-facing) at distances dd from the
        footprint plane, and the weight of the component's own composition."""
        e = self.ends[side]
        dd = np.asarray(dd, float)
        Lb, Lg, Lf = e["Lb"], max(e["Lg"], 1e-6), e["Lf"]
        Ldom = max(Lf, Lb + Lg)
        t = np.clip((dd - Lb) / Lg, 0, 1)
        wb = np.where(t < 0.5, 1 - _sstep(2 * t), 0.0)
        wi = np.where(t < 0.5, _sstep(2 * t), 1 - _sstep(2 * t - 1))
        wl = np.where(t < 0.5, 0.0, _sstep(2 * t - 1))
        Lj = max(0.5, min(Lg, Ldom) / 2)
        own = np.where(np.isfinite(dd), _sstep((dd - (Ldom - Lj)) / Lj), 1.0)
        return np.nan_to_num(np.stack([wb, wi, wl], 1)) * (1 - own)[:, None], own

    def material_colors(self, dist, own_comp):
        """Per-point colours of the graded material field: each end's bone-facing ->
        interface -> ligament-facing gradient blending smoothly into the component's own
        composition (dist: per-point distances to the footprint planes, from _map)."""
        p, M = self.p, self.materials
        C_own = _mix_colors(_normalize(own_comp), M)
        m = dist["tib"].size
        col = np.zeros((m, 3))
        e_tot = np.zeros(m)
        for side in ("tib", "fem"):
            Wz, _ = self._end_weights(side, dist[side])
            Cz = np.stack([_mix_colors(_normalize(p["end.%s.comp_%s" % (side, z)]), M) for z in BAND])
            col += Wz @ Cz
            e_tot += Wz.sum(1)
        sc = np.where(e_tot > 1, 1 / np.maximum(e_tot, 1e-9), 1.0)
        col *= sc[:, None]
        e_tot = np.minimum(e_tot, 1)
        return np.clip(col + (1 - e_tot)[:, None] * C_own, 0, 1)

    def _trim(self, P, U, dist, extra=None, stops=None):
        """Keep the part of a path between the endpoint stops (distance from each bone >= the
        end's stop, or this path's own stop when endpoints are staggered), with exact endpoints."""
        st, sf = stops if stops is not None else (self.ends["tib"]["stop"], self.ends["fem"]["stop"])
        dt, df = dist["tib"], dist["fem"]
        ok = (dt >= st) & (df >= sf)
        idx = np.nonzero(ok)[0]
        if idx.size < 2:
            return None
        a, b = idx[0], idx[-1]
        sl = list(range(a, b + 1))
        Pk, Uk = [P[sl]], [U[sl]] if U is not None else None
        ex = {k: [v[sl]] for k, v in (extra or {}).items()}
        if a > 0:
            dd = dt if dt[a - 1] < st else df
            stop = st if dt[a - 1] < st else sf
            t = np.clip((stop - dd[a - 1]) / max(dd[a] - dd[a - 1], 1e-9), 0, 1)
            Pk.insert(0, (P[a - 1] + t * (P[a] - P[a - 1]))[None])
            if U is not None:
                Uk.insert(0, U[a][None])
            for k, v in (extra or {}).items():
                ex[k].insert(0, v[a][None])
        if b < P.shape[0] - 1:
            dd = df if df[b + 1] < sf else dt
            stop = sf if df[b + 1] < sf else st
            t = np.clip((dd[b] - stop) / max(dd[b] - dd[b + 1], 1e-9), 0, 1)
            Pk.append((P[b] + t * (P[b + 1] - P[b]))[None])
            if U is not None:
                Uk.append(U[b][None])
            for k, v in (extra or {}).items():
                ex[k].append(v[b][None])
        out = {k: np.concatenate(v) for k, v in ex.items()}
        return np.vstack(Pk), (np.vstack(Uk) if U is not None else None), out

    # ------------------------------------------------------------------ 3D
    def _build_3d(self):
        p = self.p
        q_full = self.quality == "full"
        self._sheath_3d(96 if q_full else 64)
        self._braid_3d()
        self._channels_3d(2 if q_full else 3)
        self._ports_3d()
        self._coating_3d()
        self._grooves_3d()
        self._ends_3d()
        if p["sheath.pores"] == "explicit":
            self._pores_3d()
        self._interface_3d()
        self._anchors_3d()

    def _ring(self, which, s, K):
        """Sheath outline at stations s: arc-length uniform points (len(s), K, 2) in section coords."""
        A = self._interp(self.A if which == "outer" else self.Ai, s)
        B = self._interp(self.B if which == "outer" else self.Bi, s)
        nx = self._interp(self.nx, s)
        tab = ArcTable(A, B, nx, 256)
        f = np.arange(K) / K
        rows = np.repeat(np.arange(s.size), K)
        pts, nrm = tab.at(rows, np.tile(f, s.size))
        th = self._interp(self.th, s)
        X = self._interp(self.c0, s)[:, None] + _rot(pts.reshape(s.size, K, 2), th[:, None])
        Nw = _rot(nrm.reshape(s.size, K, 2), th[:, None])
        return X, Nw, tab

    def _sheath_s(self):
        """Stations used for the sheath mesh: the extension past each end, denser
        stations inside the flares, and the body stations."""
        extra = []
        for side in ("tib", "fem"):
            e = self.ends[side]
            if side == "tib":
                extra += [np.linspace(0, e["s_fs"], 18)]
            else:
                extra += [np.linspace(e["s_fs"], 1, 18)]
        return np.unique(np.concatenate([self.s, *extra]))

    def _sheath_3d(self, K):
        ss = self._sheath_s()
        self.sh_s = ss
        Xo, _, _ = self._ring("outer", ss, K)
        Xi, _, _ = self._ring("inner", ss, K)
        m = ss.size
        Po, _, do = self._map(np.repeat(ss, K), Xo.reshape(-1, 2))
        Pi, _, _ = self._map(np.repeat(ss, K), Xi.reshape(-1, 2))
        Po, Pi = Po.reshape(m, K, 3), Pi.reshape(m, K, 3)
        self.sh_outer3, self.sh_inner3, self.sh_X_outer, self.sh_X_inner = Po, Pi, Xo, Xi
        self.sh_dist = {k: v.reshape(m, K) for k, v in do.items()}
        V, F = _annular_shell(Po, Pi)
        col = self.material_colors(do, self.p["sheath.composition"])
        col = np.vstack([col, col])
        self.objects.append({"id": "layer_sheath", "label": "Level 5 · Ribbon outer sheath", "category": "layer_sheath",
                             "level": 5, "kind": "mesh", "V": V, "F": F, "status": "proposed",
                             "attr": {"mcol": (col * 255).astype(np.uint8).ravel()},
                             "note": "wall %.2f mm, clearance %.2f mm, mid-length %.1f x %.1f mm (%s)" % (
                                 self.tw, self.cs, 2 * self.A[self.mid], 2 * self.B[self.mid], self.p["sheath.size_mode"].replace("_", " "))})

    def _braid_3d(self):
        lines = []
        self.braid_info = []
        _, _, s = self._span(8)
        n = s.size
        body = (s >= 0) & (s <= 1)
        for m in self.mant:
            d, mm = m["d"], m["m"]
            if mm < 1:
                continue
            sc = np.clip(s, 0, 1)
            am, bm = self._interp(m["a"], sc) + d, self._interp(m["b"], sc) + d
            cc, th = self._interp(m["cc"], sc), self._interp(m["th"], sc)
            tab = ArcTable(am, bm, np.full(n, MANTLE_EXP), 256)
            per = tab.per
            # helix phase: the local braid angle stays alpha as the perimeter changes
            ctr3 = self._body(s, cc)
            dsig = np.linalg.norm(np.diff(ctr3, axis=0), axis=1)
            sig = np.concatenate([[0], np.cumsum(dsig)])
            ta = math.tan(m["alpha"])
            dTheta = 2 * np.pi * ta / np.maximum(per, 1e-9)
            Theta = np.concatenate([[0], np.cumsum(0.5 * (dTheta[1:] + dTheta[:-1]) * dsig)])
            two = m["pattern"] == "2/2"
            for fam in ("+", "-"):
                for i in range(mm):
                    if fam == "+":
                        f = i / mm + Theta / (2 * np.pi)
                        bcoord = 2 * np.pi * f + Theta
                        if two:
                            ph = (mm / 4.0) * (bcoord - np.pi / mm) + np.pi * i / 2 + np.pi / 4
                            off = np.clip(math.sqrt(2) * np.cos(ph), -1, 1)
                        else:
                            ph = (mm / 2.0) * (bcoord - np.pi / mm) + np.pi * i
                            off = np.clip(1.3 * np.cos(ph), -1, 1)
                    else:
                        f = i / mm + 0.5 / mm - Theta / (2 * np.pi)
                        acoord = 2 * np.pi * f - Theta
                        if two:
                            ph = (mm / 4.0) * acoord + np.pi * i / 2 + np.pi / 4
                            off = -np.clip(math.sqrt(2) * np.cos(ph), -1, 1)
                        else:
                            ph = (mm / 2.0) * acoord + np.pi * i
                            off = -np.clip(1.3 * np.cos(ph), -1, 1)
                    pts, nrm = tab.at(np.arange(n), f)
                    loc = pts + (0.5 * d * off)[:, None] * nrm
                    X = cc + _rot(loc, th)
                    P, _, dist = self._map(s, X)
                    tr = self._trim(P, None, dist, {"dt": dist["tib"], "df": dist["fem"]})
                    if tr is None:
                        continue
                    Pt, _, ex = tr
                    col = self.material_colors({"tib": ex["dt"], "fem": ex["df"]}, self.p["rf.%s.composition" % m["key"]])
                    lines.append({"P": Pt, "r": np.full(Pt.shape[0], 0.5 * d), "group": m["bi"], "family": fam,
                                  "strand": len(lines), "mcol": (col * 255).astype(np.uint8)})
            pitch = per / max(ta, 1e-9)
            spacing = per * math.cos(m["alpha"]) / mm - d
            bl = sig[body]
            self.braid_info.append({"key": m["key"], "perimeter_mm": per[body], "pitch_mm": pitch[body], "spacing_mm": spacing[body],
                                    "winding_deg": float(math.degrees(Theta[body][-1] - Theta[body][0])),
                                    "length_mm": float(bl[-1] - bl[0])})
        self.braid_lines = lines
        if lines:
            self.objects.append({"id": "layer_braid", "label": "Level 4 · Braided reinforcement (AM-like and PL-like mantles)",
                                 "category": "layer_braid", "level": 4, "kind": "tubes", "lines": lines, "status": "proposed",
                                 "note": "%s strands; separate mantles are an engineered feature, not a depiction of two isolated native bundles"
                                         % " / ".join("%s %d x %.2f mm at %.0f deg" % (m["key"].upper(), m["N"], m["d"], math.degrees(m["alpha"])) for m in self.mant)})

    def _channels_3d(self, step):
        ss = self.ch_s
        keep = np.unique(np.concatenate([np.arange(0, ss.size, step), [ss.size - 1]]))
        sk = ss[keep]
        self.ch3, self.chU3, self.ch_sv = [], [], []
        lines = []
        cov0, cov1 = float(self.p["sf.coverage_start"]), float(self.p["sf.coverage_end"])
        r_out, r_lum = 0.5 * self.od_w, 0.5 * self.lum_w
        n_ch = len(self.chans)
        if n_ch:
            # one batched mapping for all channels
            Xall = self.ch_X[:, keep].reshape(-1, 2)
            Uall = self.ch_U[:, keep].reshape(-1, 2)
            Pall, Dall, dall = self._map(np.tile(sk, n_ch), Xall, central=True, U2=Uall)
            m = sk.size
            self.stagger = {}
            stops = np.array([[self.ends["tib"]["stop"], self.ends["fem"]["stop"]]] * n_ch, float)
            for j, side in enumerate(("tib", "fem")):
                if self.ends[side]["staggered"]:
                    stops[:, j] = self._stagger(side, Pall.reshape(n_ch, m, 3), dall[side].reshape(n_ch, m))
        for k, c in enumerate(self.chans):
            sl = slice(k * m, (k + 1) * m)
            dist = {"tib": dall["tib"][sl], "fem": dall["fem"][sl]}
            tr = self._trim(Pall[sl], Dall[sl], dist, {"s": sk, "dt": dist["tib"], "df": dist["fem"]}, stops=tuple(stops[k]))
            if tr is None:
                continue
            P, U, ex = tr
            self.ch3.append(P)
            self.chU3.append(U)
            self.ch_sv.append(ex["s"])
            coat = ((ex["s"] >= cov0) & (ex["s"] <= cov1) & self.coat_ch).astype(np.uint8)
            ri = r_lum - coat * self.coat_geom
            col = self.material_colors({"tib": ex["dt"], "fem": ex["df"]}, self.p["ch.wall_composition"])
            lines.append({"P": P, "r": np.full(P.shape[0], r_out), "ri": ri, "u": U, "group": c["group"], "channel": k,
                          "coat": coat, "viol": np.full(P.shape[0], 1 if c["over"] else 0, np.uint8), "s": ex["s"],
                          "mcol": (col * 255).astype(np.uint8)})
        self.ch_lines = lines
        if lines:
            g = [sum(1 for c in self.chans if c["group"] == q) for q in (0, 1, 2)]
            self.objects.append({"id": "layer_channels", "label": "Level 3 · Hollow core channels", "category": "layer_channels",
                                 "level": 3, "kind": "tubes", "hollow": True, "lines": lines, "status": "proposed",
                                 "shape": self.cshape, "aspect": self.casp, "exponent": self.cexp,
                                 "note": "%d channels (AM-like %d, PL-like %d, shared %d); lumen %.0f um, wall %.0f um, outside %.0f um, spacing %.0f um"
                                         % (len(lines), g[0], g[1], g[2], 1000 * self.lum, 1000 * self.cwall, 1000 * self.od_w, 1000 * self.cgap)})

    def _stagger(self, side, P, d):
        """Staggered endpoints: walking from the body toward this bone, a channel ends at the
        last station where it still keeps the requested spacing from every other channel (same
        station and the few stations before it). Returns each channel's stop distance (mm)."""
        e = self.ends[side]
        n, m = d.shape
        need = self.od_max + self.cgap - 0.015
        stop = np.full(n, e["stop"])
        dd = d[0]
        order = [i for i in np.argsort(-dd) if np.isfinite(dd[i]) and e["stop"] <= dd[i] <= e["Lf"] + 1.0]
        alive = np.ones(n, bool)
        last = np.full(n, 10 ** 9)                               # walk position of each ended channel's last point
        rc = np.linalg.norm(P - P.mean(0, keepdims=True), axis=2)    # distance from the bundle centre
        for pos, i in enumerate(order):
            back = [(pp, order[pp]) for pp in range(max(0, pos - 3), pos)]
            while True:
                act = np.nonzero(alive)[0]
                if act.size == 0:
                    break
                new_pts = P[act, i]
                ref_pts, ref_ch = [new_pts], [act]
                for pp, q in back:
                    present = np.nonzero(last >= pp)[0]          # channels that still had a point there
                    ref_pts.append(P[present, q])
                    ref_ch.append(present)
                R = np.vstack(ref_pts)
                RC = np.concatenate(ref_ch)
                dist = np.linalg.norm(new_pts[:, None, :] - R[None], axis=2)
                dist[RC[None, :] == act[:, None]] = 9.0
                bad = dist.min(1) < need
                if not bad.any():
                    break
                # end the crowded channel that is furthest from the bundle centre first
                cand = act[bad]
                kk = cand[np.argmax(rc[cand, i])]
                alive[kk] = False
                if pos > 0:
                    stop[kk] = max(e["stop"], float(d[kk, order[pos - 1]]))
                    last[kk] = pos - 1
                else:
                    stop[kk] = max(e["stop"], float(d[kk, i]) + 0.01)
                    last[kk] = -1
        early = stop > e["stop"] + 1e-6
        self.stagger[side] = {"early": int(early.sum()), "total": n,
                              "range_mm": [float(stop[early].min()), float(stop[early].max())] if early.any() else None}
        return stop

    def _ports_3d(self):
        p = self.p
        self.ports = []
        if p["ch.connections"] != "ports" or len(self.chans) < 2:
            return
        dp = float(p["ch.port_diameter_um"]) * 1e-3
        spc = float(p["ch.port_spacing_mm"])
        CL = self.ch_lines
        lines = []
        pairs = set()
        s0, s1 = self.ends["tib"]["s_fs"], self.ends["fem"]["s_fs"]
        smid = 0.5 * (s0 + s1)

        def at(ln, sv):
            i = int(np.argmin(np.abs(ln["s"] - sv)))
            return ln["P"][i], ln["u"][i]
        for q in (0, 1, 2):
            ids = [k for k, ln in enumerate(CL) if ln["group"] == q]
            if len(ids) < 2:
                continue
            Pm = np.array([at(CL[k], smid)[0] for k in ids])
            _, jj = cKDTree(Pm).query(Pm, k=2)
            for a_, j in enumerate(jj[:, 1]):
                pairs.add(tuple(sorted((ids[a_], ids[j]))))
        for pi, (a, b) in enumerate(sorted(pairs)):
            z0 = s0 * self.L + spc * (0.5 + (0.37 * pi) % 1.0)
            for z in np.arange(z0, s1 * self.L - 0.25, spc):
                A, U = at(CL[a], z / self.L)
                B, _ = at(CL[b], z / self.L)
                t = np.linspace(0, 1, 6)[:, None]
                lines.append({"P": A + t * (B - A), "r": np.full(6, 0.5 * dp + self.cwall), "ri": np.full(6, 0.5 * dp),
                              "u": np.repeat(U[None], 6, 0), "group": CL[a]["group"], "channel": CL[a]["channel"]})
                self.ports.append((CL[a]["channel"], CL[b]["channel"], float(z)))
        if lines:
            self.objects.append({"id": "layer_ports", "label": "Level 3 · Channel connections (ports)", "category": "layer_channels",
                                 "level": 3, "kind": "tubes", "hollow": True, "lines": lines, "status": "proposed",
                                 "shape": "circular", "aspect": 1.0, "exponent": 2.0,
                                 "note": "%d ports, lumen %.0f um, every %.1f mm between nearest neighbours" % (len(lines), 1000 * dp, spc)})

    def _coating_3d(self):
        if self.coat_t <= 0 or not (self.coat_ch or self.coat_sh):
            return
        cov0, cov1 = float(self.p["sf.coverage_start"]), float(self.p["sf.coverage_end"])
        t = max(self.coat_geom, 1e-4)
        if self.coat_ch and self.ch_lines:
            lines = []
            r_lum = 0.5 * self.lum_w
            for ln in self.ch_lines:
                k = ln["coat"] > 0
                if k.sum() < 2:
                    continue
                idx = np.nonzero(k)[0]
                sl = slice(idx[0], idx[-1] + 1)
                n = idx[-1] + 1 - idx[0]
                lines.append({"P": ln["P"][sl], "r": np.full(n, r_lum), "ri": np.full(n, r_lum - t), "u": ln["u"][sl],
                              "group": ln["group"], "channel": ln["channel"]})
            if lines:
                self.objects.append({"id": "layer_coating", "label": "Level 1 · Coated channel lumens", "category": "layer_coating",
                                     "level": 1, "kind": "tubes", "hollow": True, "lines": lines, "status": "proposed",
                                     "shape": self.cshape, "aspect": self.casp, "exponent": self.cexp,
                                     "note": "coating %.0f nm on s %.2f-%.2f (%s); nano-texture shown in the magnified detail"
                                             % (1e6 * self.coat_t, cov0, cov1, "explicit geometry" if self.coat_explicit else "metadata only")})
        if self.coat_sh:
            ss = self.sh_s
            k = (ss >= cov0) & (ss <= cov1)
            if k.sum() >= 2:
                Pi = self.sh_inner3[k]
                C = Pi.mean(1, keepdims=True)
                D = C - Pi
                D /= np.maximum(np.linalg.norm(D, axis=2, keepdims=True), 1e-12)
                V, F = _annular_shell(Pi, Pi + D * max(t, 0.004))
                self.objects.append({"id": "layer_coating_sheath", "label": "Level 1 · Coated inner sheath surface",
                                     "category": "layer_coating", "level": 1, "kind": "mesh", "V": V, "F": F, "status": "proposed",
                                     "note": "coating %.0f nm on the inner sheath surface (drawn at least 4 um thick so it can be seen)" % (1e6 * self.coat_t)})

    def _grooves_3d(self):
        p = self.p
        if not (self.gr_ch or self.gr_sh):
            return
        w = float(p["gr.width_um"]) * 1e-3
        ridge = float(p["gr.ridge_um"]) * 1e-3
        period = w + ridge
        ang = math.radians(float(p["gr.orientation_deg"]))
        cov = float(p["gr.coverage"])
        lo, hi = 0.5 - cov / 2, 0.5 + cov / 2
        lines = []
        r_l = 0.5 * self.lum_w
        n_true = max(1, int(math.floor(math.pi * self.lum_w / period)))
        n_rep = min(n_true, 6)
        if self.gr_ch and self.ch_lines:
            for ln in self.ch_lines:
                sk = ln["s"]
                k = (sk >= lo) & (sk <= hi)
                if k.sum() < 2:
                    continue
                P, U = ln["P"][k], ln["u"][k]
                T = np.gradient(P, axis=0)
                T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
                U = U - (U * T).sum(1, keepdims=True) * T
                U /= np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
                V = np.cross(T, U)
                z = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
                for g in range(n_rep):
                    ph = 2 * np.pi * g / n_rep + z * math.tan(max(min(ang, 1.45), -1.45)) / max(r_l, 1e-6)
                    Q = P + r_l * (np.cos(ph)[:, None] * U + np.sin(ph)[:, None] * V)
                    lines.append({"P": Q, "r": np.full(Q.shape[0], 0.004), "where": 0})
        ss = self.sh_s
        if self.gr_sh:
            k = (ss >= lo) & (ss <= hi)
            if k.sum() >= 2:
                sk = ss[k]
                A, B = self._interp(self.Ai, sk), self._interp(self.Bi, sk)
                tab = ArcTable(A, B, self._interp(self.nx, sk), 256)
                per = tab.per
                n_sh_true = max(1, int(per[len(sk) // 2] / period))
                n_rep_sh = min(n_sh_true, 48)
                z = sk * self.L
                for g in range(n_rep_sh):
                    f = g / n_rep_sh + (z - z[0]) * math.tan(max(min(ang, 1.45), -1.45)) / np.maximum(per, 1e-9)
                    pts, nrm = tab.at(np.arange(sk.size), f)
                    X = self._interp(self.c0, sk) + _rot(pts - 0.002 * nrm, self._interp(self.th, sk))
                    lines.append({"P": self._p3(sk, X), "r": np.full(sk.size, 0.004), "where": 1})
            if p["gr.outer_sheath"] == "grooved":
                k = (ss >= lo) & (ss <= hi)
                sk = ss[k]
                tab = ArcTable(self._interp(self.A, sk), self._interp(self.B, sk), self._interp(self.nx, sk), 256)
                for g in range(48):
                    pts, nrm = tab.at(np.arange(sk.size), np.full(sk.size, g / 48))
                    X = self._interp(self.c0, sk) + _rot(pts + 0.002 * nrm, self._interp(self.th, sk))
                    lines.append({"P": self._p3(sk, X), "r": np.full(sk.size, 0.004), "where": 2})
                self.groove_sheath_count = n_sh_true
        self.groove_rep = (n_rep, n_true)
        if lines:
            self.objects.append({"id": "layer_grooves", "label": "Level 2 · Groove paths on guidance surfaces (representative)",
                                 "category": "layer_grooves", "level": 2, "kind": "lines", "lines": lines, "status": "proposed",
                                 "note": "%d of %d grooves drawn per channel lumen%s; true groove geometry is in the magnified detail"
                                         % (n_rep, n_true, "; representative grooves on the inner sheath" if self.gr_sh else "")})

    def _ends_3d(self):
        p = self.p
        K = 64
        for side in ("tib", "fem"):
            e = self.ends[side]
            fp = self.fp[side]
            Ldom = max(e["Lf"], e["Lb"] + e["Lg"])
            Ldom = min(Ldom, 0.45 * self.L)
            if side == "tib":
                ss = np.linspace(0, Ldom / self.L, 22)
            else:
                ss = np.linspace(1 - Ldom / self.L, 1, 22)
            Xo, _, _ = self._ring("outer", ss, K)
            Po, _, dist = self._map(np.repeat(ss, K), Xo.reshape(-1, 2))
            Po = Po.reshape(ss.size, K, 3)
            V, F = _closed_loft(Po)
            # the domain is coloured by the end gradient only (bone -> interface -> ligament-facing)
            dd = dist[side].reshape(ss.size, K)
            Wz, own = self._end_weights(side, np.concatenate([dd.ravel(), dd[[0, -1]].mean(1)]))
            Cz = np.stack([_mix_colors(_normalize(p["end.%s.comp_%s" % (side, z)]), self.materials) for z in BAND])
            tot = np.maximum(Wz.sum(1, keepdims=True), 1e-9)
            col = np.where(Wz.sum(1, keepdims=True) > 1e-6, (Wz / tot) @ Cz, Cz[2][None])
            lab = "Tibial" if side == "tib" else "Femoral"
            self.objects.append({"id": "layer_end_%s" % side, "label": "Level 6 · %s flared end (graded interface)" % lab,
                                 "category": "layer_end", "level": 6, "kind": "mesh", "V": V, "F": F, "status": "proposed",
                                 "attr": {"mcol": (np.clip(col, 0, 1) * 255).astype(np.uint8).ravel()}, "end": side,
                                 "note": "flare %.1f mm, base %.1f x %.1f mm; bone-facing %.1f mm, gradient %.1f mm" % (
                                     e["Lf"], e["Wf"], e["Tf"], e["Lb"], e["Lg"])})
            # footprint outline and flare base on bone
            ol = fp.outline(96)
            base = self._flare_base(side)
            lines = []
            for Y, kind in ((ol, 0), (base, 1)):
                Y = np.vstack([Y, Y[:1]])
                P = fp.to3d(Y) + 0.12 * fp.normal
                lines.append({"P": P, "r": np.full(P.shape[0], 0.07), "outline": kind})
            self.objects.append({"id": "layer_footprint_%s" % side, "label": "Level 6 · %s scaffold footprint (outline + flare base)" % lab,
                                 "category": "layer_footprint", "level": 6, "kind": "lines", "lines": lines, "status": "proposed",
                                 "end": side, "note": "%s outline %.1f x %.1f mm" % (fp.kind, fp.L, fp.W)})
            if p["end.%s.fixation" % side] == "plug":
                D = float(p["end.%s.plug_diameter_mm" % side])
                H = float(p["end.%s.plug_depth_mm" % side])
                P = np.stack([fp.center + 0.3 * fp.normal, fp.center - H * fp.normal])
                V, F = mesher.tube_mesh(np.linspace(P[0], P[1], 6), 0.5 * D, nsec=32)
                self.objects.append({"id": "layer_plug_%s" % side, "label": "Level 6 · %s fixation plug" % lab,
                                     "category": "layer_plug", "level": 6, "kind": "mesh", "V": V, "F": F, "status": "proposed",
                                     "end": side, "note": "press-fit plug %.1f mm diameter, %.1f mm into bone along the footprint normal" % (D, H)})

    def _pores_3d(self):
        p = self.p
        dp = float(p["sheath.pore_diameter_um"]) * 1e-3
        spc = float(p["sheath.pore_spacing_mm"])
        stag = p["sheath.pore_pattern"] == "staggered"
        cov = float(p["sheath.pore_coverage"])
        e0, e1 = self.ends["tib"], self.ends["fem"]
        s_lo, s_hi = e0["s_fs"], e1["s_fs"]
        mid = 0.5 * (s_lo + s_hi)
        half = 0.5 * cov * (s_hi - s_lo)
        a, b = mid - half, mid + half
        z = np.arange(a * self.L + 0.5 * spc, b * self.L, spc)
        if z.size == 0:
            self.pore_count = 0
            return
        sz = z / self.L
        Xo, _, tab = self._ring("outer", sz, 8)
        per = tab.per
        n_around = max(3, int(per[len(sz) // 2] / spc))
        f = np.arange(n_around) / n_around
        parts = []
        cnt = 0
        A_o, B_o = self._interp(self.A, sz), self._interp(self.B, sz)
        A_i, B_i = self._interp(self.Ai, sz), self._interp(self.Bi, sz)
        nx = self._interp(self.nx, sz)
        to = ArcTable(A_o, B_o, nx, 256)
        ti = ArcTable(A_i, B_i, nx, 256)
        th = self._interp(self.th, sz)
        c0 = self._interp(self.c0, sz)
        for r in range(sz.size):
            fr = f + (0.5 / n_around if (stag and r % 2) else 0.0)
            rows = np.full(fr.size, r)
            po, no = to.at(rows, fr)
            pi, _ = ti.at(rows, fr)
            Xa = c0[r] + _rot(po + 0.02 * no, th[r])
            Xb = c0[r] + _rot(pi - 0.02 * no, th[r])
            Pa = self._p3(np.full(fr.size, sz[r]), Xa)
            Pb = self._p3(np.full(fr.size, sz[r]), Xb)
            for j in range(fr.size):
                parts.append(mesher.tube_mesh(np.stack([Pa[j], Pb[j]]), 0.5 * dp, nsec=10))
                cnt += 1
        self.pore_count = cnt
        V, F = mesher.concat(parts)
        self.objects.append({"id": "layer_pores", "label": "Level 5 · Explicit sheath pores (through-wall)", "category": "layer_pores",
                             "level": 5, "kind": "mesh", "V": V, "F": F, "status": "proposed",
                             "note": "%d pores, %.0f um, %.2f mm apart (%s). Geometric openings only: no fluid-transport model" % (
                                 cnt, 1000 * dp, spc, p["sheath.pore_pattern"])})

    def _interface_3d(self):
        """Level 5 inter-bundle transition band: a translucent sheet following the AM/PL
        interface through the body of the construct, spanning the sheath interior.

        This is a visualization of the band kept free of braid (rf.transition_mm); no
        wall is placed there and the scaffold stays continuous across it. It is not a
        load-bearing partition and is excluded from STL export."""
        half = 0.5 * float(self.p["rf.transition_mm"])
        s0 = self.ends["tib"]["s_fs"]
        s1 = self.ends["fem"]["s_fs"]
        idx = np.where((self.s >= s0) & (self.s <= s1))[0]
        if half <= 0 or idx.size < 2:
            return
        Em, nrm = self._iface_lin()
        tdir = np.stack([-nrm[:, 1], nrm[:, 0]], 1)
        rows = []
        s_sel = []
        for i in idx:
            R = 1.2 * max(float(self.Ai[i]), float(self.Bi[i]))
            if R <= 0:
                continue
            tau = np.linspace(-R, R, 97)
            X = Em[i][None] + tau[:, None] * tdir[i][None]
            loc = _rot(X - self.c0[i], -float(self.th[i]))
            inside = se_value(loc[None], np.array([self.Ai[i]]), np.array([self.Bi[i]]),
                              np.array([self.nx[i]]))[0] <= 1.0
            if not inside.any():
                continue
            tin = tau[inside]
            corners = np.stack([Em[i] + tin.min() * tdir[i] - half * nrm[i],
                                Em[i] + tin.min() * tdir[i] + half * nrm[i],
                                Em[i] + tin.max() * tdir[i] - half * nrm[i],
                                Em[i] + tin.max() * tdir[i] + half * nrm[i]])
            rows.append(corners)
            s_sel.append(self.s[i])
        if len(rows) < 2:
            return
        X = np.stack(rows)
        m = X.shape[0]
        P, _, _ = self._map(np.repeat(np.asarray(s_sel), 4), X.reshape(-1, 2))
        P = P.reshape(m, 4, 3)
        V = P.reshape(-1, 3)
        F = []
        for r in range(m - 1):
            b0, b1 = 4 * r, 4 * (r + 1)
            for k in range(3):
                F.append([b0 + k, b1 + k, b0 + k + 1])
                F.append([b0 + k + 1, b1 + k, b1 + k + 1])
        F = np.asarray(F, int)
        self.objects.append({"id": "layer_interface", "label": "Level 5 · Inter-bundle transition band", "category": "layer_interface",
                             "level": 5, "kind": "mesh", "V": V, "F": F, "status": "proposed",
                             "note": "%.2f mm band around the AM/PL interface kept free of braid (body region). No wall: "
                                     "channels may run through it; not load-bearing" % (2 * half)})

    def _anchors_3d(self):
        """Level 6 bundle anchors: the four AM/PL x tibial/femoral attachment points.

        Each anchor is the bundle's attachment-subregion centroid (the end of the
        bundle centroid path), snapped onto the scaffold footprint surface. Markers
        are schematic visualization aids, excluded from STL export; positions and
        provenance travel in the object's "markers" metadata."""
        lig = self.lig
        parts = []
        markers = []
        bundle_attr = []
        for side, s_end, lab_end in (("tib", 0.0, "tibial"), ("fem", 1.0, "femoral")):
            fp = self.fp[side]
            for b, lab_b in ((0, "AM"), (1, "PL")):
                c, _ = lig.bundle_geometry(b)
                P = lig.map(np.array([s_end]), c[None])[0]
                Y = np.array([[(P - fp.center) @ fp.e_l, (P - fp.center) @ fp.e_s]])
                P3 = fp.to3d(Y)[0] + 0.25 * fp.normal
                V, F = _sphere(P3, 0.55)
                parts.append((V, F + sum(v.shape[0] for v, _ in parts)))
                bundle_attr.append(np.full(V.shape[0], b, np.uint8))
                pids = ["fp.%s.center" % side, "fp.%s.am_share" % side,
                        "end.%s.offset_long_mm" % side, "end.%s.offset_short_mm" % side,
                        "end.%s.rotation_deg" % side]
                sts = ["%s=%s" % (k, self.prov.get(k, {}).get("status", "?")) for k in pids]
                markers.append({"id": "anchor_%s_%s" % (side, lab_b.lower()),
                                "label": "%s %s anchor" % (lab_b, lab_end),
                                "bundle": lab_b, "end": lab_end, "P": [round(float(x), 3) for x in P3],
                                "provenance": "; ".join(sts)})
        V = np.vstack([v for v, _ in parts])
        F = np.vstack([f for _, f in parts])
        self.objects.append({"id": "layer_anchors", "label": "Level 6 · Bundle anchors (AM/PL, tibial/femoral)",
                             "category": "layer_anchors", "level": 6, "kind": "mesh", "V": V, "F": F,
                             "status": "proposed", "markers": markers,
                             "attr": {"bundle": np.concatenate(bundle_attr)},
                             "note": "schematic markers at the four attachment-subregion centroids on the scaffold "
                                     "footprints (not fixation hardware); see markers for positions and provenance"})

    # ------------------------------------------------------------------ conflicts
    def _conf(self, cid, level, status, title, explain, params, fixes=(), levels=None, value=None, threshold=None, units=""):
        self.conflicts.append({"id": cid, "level": level, "levels": sorted(set([level] + list(levels or []))), "status": status,
                               "title": title, "explain": explain, "params": list(params), "value": value,
                               "threshold": threshold, "units": units, "fixes": [f for f in fixes if f]})

    @staticmethod
    def _fix(label, level, **set_):
        return {"label": label, "level": level, "set": {k.replace("__", "."): v for k, v in set_.items()}}

    def _checks(self):
        p = self.p
        self._check_channels()
        self._check_mantles()
        self._check_braid()
        self._check_sheath()
        self._check_micro_surface()
        self._check_ends()
        self._check_compositions()

    def _check_channels(self):
        p = self.p
        lo, hi = CHANNEL_RANGE_UM
        lum_um = float(p["ch.lumen_um"])
        if not lo <= lum_um <= hi:
            self._conf("layer_channel_range", 3, "info", "Channel lumen outside the proposed range",
                       "Lumen %.0f um is outside the proposed %.0f-%.0f um channel range. That range is an adjustable design proposal, "
                       "not a validated anatomical requirement." % (lum_um, lo, hi), ["ch.lumen_um"],
                       [self._fix("Set lumen to 200 um", 3, ch__lumen_um=200.0)], value=lum_um, threshold=[lo, hi], units="um")
        for gi, m in zip(self.group_info, self.mant):
            key = gi["key"]
            lab = key.upper() + "-like"
            if gi["requested"] > gi["fit"]:
                need = gi["requested"] / max(gi["fit"], 1)
                fixes = [self._fix("Set %s channel count to %d (what fits)" % (lab, gi["fit"]), 3, **{"ch__%s_count" % key: gi["fit"]})]
                if p["rf.size_mode"] == "manual":
                    sc = math.sqrt(need) * 1.03
                    fixes.append(self._fix("Enlarge the %s reinforcement envelope to %.2f x %.2f mm" % (
                        lab, sc * float(p["rf.%s.envelope_width_mm" % key]), sc * float(p["rf.%s.envelope_thickness_mm" % key])), 4,
                        **{"rf__%s__envelope_width_mm" % key: round(sc * float(p["rf.%s.envelope_width_mm" % key]), 2),
                           "rf__%s__envelope_thickness_mm" % key: round(sc * float(p["rf.%s.envelope_thickness_mm" % key]), 2)}))
                pitch_new = self.pitch / math.sqrt(need) * 0.97
                lum_new = 1000 * (pitch_new - self.cgap - 2 * self.cwall)
                if lum_new >= 50:
                    fixes.append(self._fix("Reduce channel lumen to %.0f um" % lum_new, 3, ch__lumen_um=round(lum_new)))
                fixes.append(self._fix("Let the count follow the space (max fit)", 3, ch__count_mode="max_fit"))
                self._conf("layer_channels_fit_%s" % key, 3, "fail", "%s channels do not fit" % lab,
                           "%d channels requested, %d fit inside the %s reinforcement envelope (%.2f x %.2f mm at its narrowest) at a %.0f um "
                           "pitch (outside %.0f um + spacing %.0f um%s). The extra %d are drawn in red where they would go; nothing was shrunk."
                           % (gi["requested"], gi["fit"], lab, 2 * gi["a_min"], 2 * gi["b_min"], 1000 * self.pitch, 1000 * self.od_max,
                              1000 * self.cgap, ", waviness/curvature margin %.0f um" % (1000 * self.wave_margin) if self.wave_margin else "",
                              gi["requested"] - gi["fit"]),
                           ["ch.%s_count" % key, "ch.lumen_um", "ch.wall_um", "ch.spacing_um", "rf.%s.envelope_width_mm" % key],
                           fixes, levels=[4], value=gi["fit"], threshold=gi["requested"], units="channels")
            if gi["fit"] == 0:
                self._conf("layer_channels_none_%s" % key, 3, "fail", "No channel fits the %s envelope" % lab,
                           "The %s reinforcement envelope (%.2f x %.2f mm at its narrowest) cannot hold one channel of %.0f um outside "
                           "diameter with %.0f um spacing." % (lab, 2 * gi["a_min"], 2 * gi["b_min"], 1000 * self.od_max, 1000 * self.cgap),
                           ["ch.lumen_um", "ch.wall_um", "rf.%s.envelope_width_mm" % key, "rf.%s.strand_diameter_mm" % key],
                           [self._fix("Reduce channel lumen to 150 um", 3, ch__lumen_um=150.0),
                            self._fix("Size the reinforcement automatically", 4, rf__size_mode="auto") if p["rf.size_mode"] == "manual" else None],
                           levels=[4])
        # measured: channel walls must not intersect (3D, generated paths)
        self.ch_min_gap = None
        placed = [ln for ln in self.ch_lines if not ln["viol"].any()]      # overflow channels are reported by the fit check
        if len(placed) >= 2:
            P = np.vstack([ln["P"] for ln in placed])
            ids = np.repeat(np.arange(len(placed)), [ln["P"].shape[0] for ln in placed])
            tr = cKDTree(P)
            pairs = tr.query_pairs(self.od_max + self.cgap, output_type="ndarray")
            if pairs.size:
                diff = ids[pairs[:, 0]] != ids[pairs[:, 1]]
                pairs = pairs[diff]
            sv = np.concatenate([ln["s"] for ln in placed])
            if pairs.size:
                dist = np.linalg.norm(P[pairs[:, 0]] - P[pairs[:, 1]], axis=1)
                g = float(dist.min() - self.od_max)
                bad = dist - self.od_max < -0.004
                sb = np.concatenate([sv[pairs[bad, 0]], sv[pairs[bad, 1]]]) if bad.any() else np.zeros(0)
            else:
                g, sb = float(self.cgap), np.zeros(0)
            self.ch_min_gap = g
            tol = 0.004
            if g < -tol:
                fixes, where = [], []
                for side in ("tib", "fem"):
                    e = self.ends[side]
                    dd = sb * self.L if side == "tib" else (1 - sb) * self.L
                    near = dd[(dd < e["Lf"] + 0.5) & ((sb < 0.5) if side == "tib" else (sb >= 0.5))]
                    if near.size:
                        lab = "tibial" if side == "tib" else "femoral"
                        where.append("%s flare (%.1f-%.1f mm from bone)" % (lab, near.min(), near.max()))
                        if not e["staggered"]:
                            fixes.append(self._fix("Stagger %s endpoints (crowded channels end early)" % lab, 6,
                                                   **{"end__%s__endpoints" % side: "staggered"}))
                        stop = min(8.0, round(float(near.max()) + 0.4, 1))
                        fixes.append(self._fix("End %s channels %.1f mm from bone" % (lab, stop), 6, **{"end__%s__endpoint_stop_mm" % side: stop}))
                        if e["Lf"] < 13:
                            fixes.append(self._fix("Lengthen the %s flare to %.1f mm" % (lab, min(14.0, e["Lf"] + 3)), 6,
                                                   **{"end__%s__flare_length_mm" % side: min(14.0, e["Lf"] + 3)}))
                fixes.append(self._fix("Increase channel spacing to %.0f um" % (1000 * (self.cgap - g) + 10), 3, ch__spacing_um=round(1000 * (self.cgap - g) + 10)))
                if float(p["ch.waviness_um"]) > 0:
                    fixes.append(self._fix("Remove channel waviness", 3, ch__waviness_um=0.0))
                self._conf("layer_channel_overlap", 3, "fail", "Channel walls intersect",
                           "Measured on the generated channel paths: neighbouring channel walls overlap by %.0f um (outside diameter %.0f um)%s. "
                           "Where the construct turns onto the bone (the femoral notch wall runs alongside the ligament), channels on the "
                           "inside of the bend are pressed together; a flare base smaller than the section or waviness larger than the spacing "
                           "has the same effect." % (-1000 * g, 1000 * self.od_max, (" in the " + " and ".join(where)) if where else ""),
                           ["ch.spacing_um", "end.tib.endpoint_stop_mm", "end.fem.endpoint_stop_mm", "end.tib.flare_length_mm",
                            "end.fem.flare_length_mm", "ch.waviness_um"], fixes, levels=[6], value=g, threshold=0.0, units="mm")
            elif g < self.cgap - 0.02:
                self._conf("layer_channel_spacing", 3, "warn", "Channels closer than the requested spacing",
                           "Smallest measured wall-to-wall gap is %.0f um (requested %.0f um), typically inside a flare that narrows."
                           % (1000 * g, 1000 * self.cgap), ["ch.spacing_um", "end.tib.endpoints", "end.fem.endpoints"],
                           [], levels=[6], value=g, threshold=self.cgap, units="mm")
        if p["ch.connections"] == "ports":
            dp = float(p["ch.port_diameter_um"])
            lim = float(p["ch.lumen_um"]) * 0.8
            if dp > lim:
                self._conf("layer_port_size", 3, "fail", "Port lumen too large for the channels",
                           "A %.0f um connection would cut away most of a %.0f um channel lumen (limit 80%%)." % (dp, float(p["ch.lumen_um"])),
                           ["ch.port_diameter_um", "ch.lumen_um"], [self._fix("Set port lumen to %.0f um" % (0.5 * float(p["ch.lumen_um"])), 3,
                                                                             ch__port_diameter_um=round(0.5 * float(p["ch.lumen_um"])))])

    def _boundary(self, m, which, K=72):
        a = m["ao"] if which == "outer" else m["a"]
        b = m["bo"] if which == "outer" else m["b"]
        Q = se_points(a, b, np.full(self.n, MANTLE_EXP), K)
        return _rot(Q, m["th"][:, None]) + m["cc"][:, None, :]

    def _check_mantles(self):
        p = self.p
        st, sf, _ = self._span()
        n = self.n
        sheath_in = self.c0[:, None, :] + _rot(se_points(self.Ai, self.Bi, self.nx, 160), self.th[:, None])
        self.mantle_margin = {}
        for m in self.mant:
            key, lab = m["key"], m["key"].upper() + "-like"
            Q = self._boundary(m, "outer")
            gaps = np.zeros(n)
            for i in range(n):
                X = Q[i]
                inside = points_in_polygon(sheath_in[i], X)
                d = polygon_distance(sheath_in[i], X)
                gaps[i] = np.where(inside, d, -d).min()
            i0 = int(np.argmin(gaps))
            g = float(gaps[i0])
            self.mantle_margin[key] = g
            if (m["a"] <= 0).any() or (m["b"] <= 0).any():
                self._conf("layer_envelope_size_%s" % key, 4, "fail", "%s reinforcement envelope is smaller than its braid wall" % lab,
                           "The braid wall (2 x %.2f mm strands) leaves no room inside the %s envelope." % (m["d"], lab),
                           ["rf.%s.strand_diameter_mm" % key, "rf.%s.envelope_width_mm" % key],
                           [self._fix("Reduce %s strand diameter to %.2f mm" % (lab, max(0.05, 0.5 * m["d"])), 4, **{"rf__%s__strand_diameter_mm" % key: round(max(0.05, 0.5 * m["d"]), 3)})],
                           levels=[3])
            if g < self.cs - 0.01:
                deficit = self.cs - g
                fixes = []
                mode = p["sheath.size_mode"]
                if mode == "profiles":
                    fixes.append(self._fix("Widen the sheath width and thickness profiles by %.2f mm" % (2 * deficit + 0.05), 5,
                                           sheath__width_profile=[round(x + 2 * deficit + 0.05, 2) for x in self._prof(p["sheath.width_profile"])],
                                           sheath__thickness_profile=[round(x + 2 * deficit + 0.05, 2) for x in self._prof(p["sheath.thickness_profile"])]))
                elif mode == "fit_anatomy":
                    f = float(p["sheath.anatomy_fraction"])
                    fn = f * (1 + (deficit + 0.03) / max(float(self.B[i0]), 1e-6))
                    if fn <= 1.1:
                        fixes.append(self._fix("Raise the sheath's anatomy fraction to %.2f" % fn, 5, sheath__anatomy_fraction=round(fn, 3)))
                if mode != "fit_contents":
                    fixes.append(self._fix("Size the sheath to fit its contents", 5, sheath__size_mode="fit_contents"))
                if deficit <= self.cs:
                    fixes.append(self._fix("Reduce sheath clearance to %.2f mm" % max(0.0, self.cs - deficit - 0.01), 5,
                                           sheath__clearance_mm=round(max(0.0, self.cs - deficit - 0.01), 3)))
                if p["rf.size_mode"] == "manual":
                    for dim in ("width", "thickness"):
                        v = float(p["rf.%s.envelope_%s_mm" % (key, dim)])
                        fixes.append(self._fix("Reduce %s envelope %s to %.2f mm" % (lab, dim, max(0.3, v - 2 * deficit - 0.05)), 4,
                                               **{"rf__%s__envelope_%s_mm" % (key, dim): round(max(0.3, v - 2 * deficit - 0.05), 2)}))
                    fixes.append(self._fix("Size the reinforcement automatically", 4, rf__size_mode="auto"))
                if m["d"] > deficit / 2 + 0.06:
                    fixes.append(self._fix("Reduce %s strand diameter to %.2f mm" % (lab, m["d"] - deficit / 2 - 0.01), 4,
                                           **{"rf__%s__strand_diameter_mm" % key: round(m["d"] - deficit / 2 - 0.01, 3)}))
                self._conf("layer_mantle_sheath_%s" % key, 5, "fail" if g < -0.005 else "fail",
                           "%s braid needs more sheath clearance" % lab,
                           "At s = %.2f the %s braid's outer surface (strand diameter %.2f mm, braid wall %.2f mm%s) is %.2f mm from the inner "
                           "sheath wall; the sheath clearance asks for %.2f mm. Increasing braid thickness or the envelope needs more room inside the sheath."
                           % (self.s[i0], lab, m["d"], 2 * m["d"], ", slack excursion %.2f mm" % m["As"] if m["As"] > 0 else "", g, self.cs),
                           ["sheath.clearance_mm", "rf.%s.strand_diameter_mm" % key, "rf.%s.envelope_width_mm" % key, "sheath.width_profile",
                            "sheath.anatomy_fraction"], fixes, levels=[4], value=g, threshold=self.cs, units="mm")
        # AM and PL mantles must not intersect; their gap is the transition band
        QA = self._boundary(self.mant[0], "outer")
        QP = self._boundary(self.mant[1], "outer")
        gaps = np.zeros(n)
        for i in range(n):
            inA = points_in_polygon(QA[i], QP[i]).any() or points_in_polygon(QP[i], QA[i]).any()
            d = np.linalg.norm(QA[i][:, None, :] - QP[i][None, :, :], axis=2).min()
            gaps[i] = -d if inA else d
        i0 = int(np.argmin(gaps))
        g = float(gaps[i0])
        self.mantle_gap = g
        fixes = []
        if p["rf.size_mode"] == "manual":
            fixes.append(self._fix("Size the reinforcement automatically (keeps the transition band)", 4, rf__size_mode="auto"))
            for key in ("am", "pl"):
                v = float(p["rf.%s.envelope_width_mm" % key])
                need = (self.wt - g) if g < self.wt else 0.0
                fixes.append(self._fix("Reduce %s envelope width to %.2f mm" % (key.upper(), max(0.3, v - need - 0.05)), 4,
                                       **{"rf__%s__envelope_width_mm" % key: round(max(0.3, v - need - 0.05), 2)}))
        if g < -0.005:
            self._conf("layer_mantle_overlap", 4, "fail", "AM-like and PL-like mantles intersect",
                       "At s = %.2f the two braided mantles overlap by %.2f mm. They share the sheath but must not occupy the same space."
                       % (self.s[i0], -g), ["rf.am.envelope_width_mm", "rf.pl.envelope_width_mm", "rf.transition_mm", "rf.size_mode"],
                       fixes, levels=[3, 5], value=g, threshold=0.0, units="mm")
        elif g < self.wt - 0.02:
            if g >= 0:
                fixes.append(self._fix("Narrow the transition width to %.2f mm" % max(0.0, g - 0.01), 4, rf__transition_mm=round(max(0.0, g - 0.01), 2)))
            self._conf("layer_transition_band", 4, "warn", "Mantles intrude into the AM/PL transition band",
                       "The gap between the AM-like and PL-like braids is %.2f mm at s = %.2f; the transition width asks for %.2f mm."
                       % (g, self.s[i0], self.wt), ["rf.transition_mm", "rf.am.envelope_width_mm", "rf.pl.envelope_width_mm"],
                       fixes, levels=[3], value=g, threshold=self.wt, units="mm")

    @staticmethod
    def _prof(v):
        return list(v) if isinstance(v, list) else list(v.get("v", []))

    def _check_braid(self):
        for m, bi in zip(self.mant, self.braid_info or [{}] * 2):
            key, lab = m["key"], m["key"].upper() + "-like"
            N = m["N"]
            mod = 8 if m["pattern"] == "2/2" else 4
            if N % mod:
                lower = max(mod, N - N % mod)
                upper = N - N % mod + mod
                self._conf("layer_braid_count_%s" % key, 4, "fail", "%s strand count does not close the %s braid" % (lab, m["pattern"]),
                           "A %s braid needs strands in multiples of %d (half in each direction, with a whole number of over/under "
                           "repeats around the mantle); %d is not." % (m["pattern"], mod, N),
                           ["rf.%s.strands" % key, "rf.%s.pattern" % key],
                           [self._fix("Use %d strands" % lower, 4, **{"rf__%s__strands" % key: lower}),
                            self._fix("Use %d strands" % upper, 4, **{"rf__%s__strands" % key: upper}) if upper <= 64 else None,
                            self._fix("Use the 1/1 pattern", 4, **{"rf__%s__pattern" % key: "1/1"}) if m["pattern"] == "2/2" and N % 4 == 0 else None])
            if not bi:
                continue
            sp = np.asarray(bi["spacing_mm"])
            i0 = int(np.argmin(sp))
            if sp[i0] < -0.005:
                per = float(bi["perimeter_mm"][i0])
                ca = math.cos(m["alpha"])
                n_ok = int(per * ca / m["d"]) * 2
                n_ok -= n_ok % (8 if m["pattern"] == "2/2" else 4)
                d_ok = per * ca / max(m["m"], 1) * 0.97
                a_ok = math.degrees(math.acos(min(1.0, m["m"] * m["d"] / per))) * 0.97 if m["m"] * m["d"] < per else None
                fixes = [self._fix("Reduce %s strands to %d" % (lab, n_ok), 4, **{"rf__%s__strands" % key: n_ok}) if n_ok >= 4 else None,
                         self._fix("Reduce %s strand diameter to %.3f mm" % (lab, d_ok), 4, **{"rf__%s__strand_diameter_mm" % key: round(d_ok, 3)}) if d_ok >= 0.05 else None,
                         self._fix("Reduce %s braid angle to %.1f deg" % (lab, a_ok), 4, **{"rf__%s__braid_angle_deg" % key: round(a_ok, 1)}) if a_ok and a_ok >= 5 else None]
                self._conf("layer_braid_jam_%s" % key, 4, "fail", "%s braid jams" % lab,
                           "At s = %.2f the mantle perimeter is %.2f mm: %d strands per direction at %.0f deg leave %.3f mm between "
                           "neighbouring strands of one family (needs >= 0); the strands would overlap." % (
                               self.s[i0], per, m["m"], math.degrees(m["alpha"]), sp[i0]),
                           ["rf.%s.strands" % key, "rf.%s.strand_diameter_mm" % key, "rf.%s.braid_angle_deg" % key],
                           fixes, value=float(sp[i0]), threshold=0.0, units="mm")
            elif sp[i0] < 0.15:
                self._conf("layer_braid_pore_%s" % key, 4, "warn", "%s inter-strand pores below 150 um" % lab,
                           "At s = %.2f the gap between neighbouring strands of one family is %.0f um, below the 150 um "
                           "specification threshold for inter-strand pores. The braid still fits; this flags a design "
                           "consideration, not a geometric conflict." % (self.s[i0], 1000 * sp[i0]),
                           ["rf.%s.strands" % key, "rf.%s.strand_diameter_mm" % key, "rf.%s.braid_angle_deg" % key],
                           [], value=float(sp[i0]), threshold=0.15, units="mm")

    def _check_sheath(self):
        p = self.p
        if (self.Ai <= 0).any() or (self.Bi <= 0).any():
            self._conf("layer_sheath_wall", 5, "fail", "Sheath wall is thicker than the sheath",
                       "Wall %.2f mm leaves no inner space at some stations." % self.tw, ["sheath.wall_mm"],
                       [self._fix("Set wall to 0.3 mm", 5, sheath__wall_mm=0.3)])
        # vs the anatomical envelope (2D, per station)
        K = 128
        Q = self.c0[:, None, :] + _rot(se_points(self.A, self.B, self.nx, K), self.th[:, None])
        prot = np.zeros(self.n)
        for i in range(self.n):
            inside = points_in_polygon(self.lig.contour[i], Q[i])
            d = polygon_distance(self.lig.contour[i], Q[i])
            prot[i] = np.where(inside, -d, d).max()
        e0, e1 = self.ends["tib"]["s_fs"], self.ends["fem"]["s_fs"]
        body = (self.s >= e0) & (self.s <= e1)
        i0 = int(np.argmax(np.where(body, prot, -9)))
        pr = float(prot[i0])
        self.envelope_protrusion = pr
        if pr > 0.05:
            fixes = []
            if p["sheath.size_mode"] == "fit_anatomy":
                f = float(p["sheath.anatomy_fraction"])
                fn = f * (1 - (pr + 0.03) / max(float(self.B[i0]), 1e-6))
                fixes.append(self._fix("Lower the anatomy fraction to %.2f" % fn, 5, sheath__anatomy_fraction=round(max(0.6, fn), 3)))
            elif p["sheath.size_mode"] == "profiles":
                fixes.append(self._fix("Narrow the width and thickness profiles by %.2f mm" % (2 * pr + 0.05), 5,
                                       sheath__width_profile=[round(max(3, x - 2 * pr - 0.05), 2) for x in self._prof(p["sheath.width_profile"])],
                                       sheath__thickness_profile=[round(max(1.5, x - 2 * pr - 0.05), 2) for x in self._prof(p["sheath.thickness_profile"])]))
            if abs(float(np.max(np.abs(profile_eval(p["sheath.orientation_profile"], self.s))))) > 1:
                fixes.append(self._fix("Align the sheath with the anatomical section", 5, sheath__orientation_profile=[0.0] * 5))
            self._conf("layer_sheath_envelope", 5, "warn", "Sheath extends beyond the intact ligament envelope",
                       "At s = %.2f the sheath's outer surface lies up to %.2f mm outside the intact reference envelope. Notch clearance "
                       "and impingement are not evaluated." % (self.s[i0], pr),
                       ["sheath.size_mode", "sheath.anatomy_fraction", "sheath.width_profile", "sheath.orientation_profile"], fixes,
                       value=pr, threshold=0.05, units="mm")
        # vs bone (3D) outside the bone-facing zones
        Po = self.sh_outer3
        Lb_t, Lb_f = self.ends["tib"]["Lb"], self.ends["fem"]["Lb"]
        k = (self.sh_dist["tib"] > Lb_t + 0.2) & (self.sh_dist["fem"] > Lb_f + 0.2)
        pen = 0.0
        if k.any():
            X = Po[k]
            pen = float(max(0.0, -min(self.fem.sdf(X).min(), self.tib.sdf(X).min())))
        self.bone_penetration = pen
        if pen > 0.15:
            self._conf("layer_sheath_bone", 5, "fail", "Sheath enters bone outside the bone-facing zone",
                       "The outer sheath reaches %.2f mm into bone beyond the bone-facing zone of the attachment. Only the bone-facing "
                       "zone may contact bone." % pen,
                       ["sheath.anatomy_fraction", "sheath.width_profile", "end.tib.flare_length_mm", "end.fem.flare_length_mm",
                        "end.tib.bone_zone_mm", "end.fem.bone_zone_mm"],
                       [self._fix("Lower the anatomy fraction to %.2f" % (0.9 * float(p["sheath.anatomy_fraction"])), 5,
                                  sheath__anatomy_fraction=round(0.9 * float(p["sheath.anatomy_fraction"]), 3)) if p["sheath.size_mode"] == "fit_anatomy" else None],
                       levels=[6], value=pen, threshold=0.15, units="mm")
        if p["sheath.pores"] == "explicit":
            dp = float(p["sheath.pore_diameter_um"]) * 1e-3
            spc = float(p["sheath.pore_spacing_mm"])
            if spc < dp + 0.05:
                self._conf("layer_pores_overlap", 5, "fail", "Sheath pores merge",
                           "Pores of %.0f um at %.2f mm centre spacing leave less than 50 um of wall between them." % (1000 * dp, spc),
                           ["sheath.pore_diameter_um", "sheath.pore_spacing_mm"],
                           [self._fix("Space pores %.2f mm apart" % (dp + 0.1), 5, sheath__pore_spacing_mm=round(dp + 0.1, 3)),
                            self._fix("Reduce pores to %.0f um" % max(20, 1000 * (spc - 0.1)), 5, sheath__pore_diameter_um=round(max(20, 1000 * (spc - 0.1))))])

    def _check_micro_surface(self):
        p = self.p
        res = float(p["gr.min_residual_um"])
        depth = float(p["gr.depth_um"])
        width = float(p["gr.width_um"])
        rb = float(p["gr.bottom_radius_um"])
        wall_um = float(p["ch.wall_um"])
        sh_um = 1000 * self.tw
        if self.gr_ch and depth > wall_um - res:
            self._conf("layer_groove_channel_wall", 2, "fail", "Grooves would cut through the channel wall",
                       "Groove depth %.1f um leaves %.1f um of a %.0f um channel wall; at least %.0f um must remain. "
                       "Increasing groove depth may require thicker channel walls." % (depth, wall_um - depth, wall_um, res),
                       ["gr.depth_um", "ch.wall_um", "gr.min_residual_um"],
                       [self._fix("Thicken channel walls to %.0f um" % math.ceil(depth + res), 3, ch__wall_um=float(math.ceil(depth + res))),
                        self._fix("Make grooves %.1f um deep" % max(0.5, wall_um - res), 2, gr__depth_um=round(max(0.5, wall_um - res), 1))],
                       levels=[3], value=wall_um - depth, threshold=res, units="um")
        if self.gr_sh and depth > sh_um - res:
            self._conf("layer_groove_sheath_wall", 2, "fail", "Grooves would cut through the sheath wall",
                       "Groove depth %.1f um leaves %.1f um of the %.0f um sheath wall (minimum %.0f um)." % (depth, sh_um - depth, sh_um, res),
                       ["gr.depth_um", "sheath.wall_mm"],
                       [self._fix("Thicken the sheath wall to %.3f mm" % ((depth + res) / 1000 + 0.005), 5, sheath__wall_mm=round((depth + res) / 1000 + 0.005, 3)),
                        self._fix("Make grooves %.1f um deep" % max(0.5, sh_um - res), 2, gr__depth_um=round(max(0.5, sh_um - res), 1))],
                       levels=[5], value=sh_um - depth, threshold=res, units="um")
        if (self.gr_ch or self.gr_sh) and rb > width / 2 + 1e-9:
            self._conf("layer_groove_radius", 2, "fail", "Groove-bottom radius larger than half the groove width",
                       "A rounded bottom of radius %.1f um does not fit a %.1f um wide groove." % (rb, width),
                       ["gr.bottom_radius_um", "gr.width_um"], [self._fix("Set bottom radius to %.1f um" % (width / 2), 2, gr__bottom_radius_um=round(width / 2, 2))])
        if self.gr_ch and math.pi * float(p["ch.lumen_um"]) < width + float(p["gr.ridge_um"]):
            self._conf("layer_groove_period", 2, "warn", "Groove period exceeds the lumen circumference",
                       "One groove plus ridge (%.0f um) is longer than the lumen circumference (%.0f um)." % (
                           width + float(p["gr.ridge_um"]), math.pi * float(p["ch.lumen_um"])), ["gr.width_um", "gr.ridge_um", "ch.lumen_um"], [])
        if p["gr.outer_sheath"] == "grooved":
            self._conf("layer_outer_grooves", 2, "warn", "Outer sheath grooved",
                       "The outer sheath is smooth by default; grooving it is an explicit experimental choice (it faces the notch and synovium).",
                       ["gr.outer_sheath"], [self._fix("Keep the outer sheath smooth", 2, gr__outer_sheath="smooth")], levels=[5])
        # Level 1
        if float(p["sf.coverage_start"]) >= float(p["sf.coverage_end"]):
            self._conf("layer_coating_range", 1, "fail", "Surface-treatment coverage is empty",
                       "Coverage starts at %.2f and ends at %.2f." % (float(p["sf.coverage_start"]), float(p["sf.coverage_end"])),
                       ["sf.coverage_start", "sf.coverage_end"], [self._fix("Cover the full length", 1, sf__coverage_start=0.0, sf__coverage_end=1.0)])
        eff = self.lum_w - 2 * self.coat_geom * (1 if self.coat_ch else 0)
        self.eff_lumen = eff
        if self.coat_ch and self.coat_explicit and eff < 0.5 * self.lum_w:
            self._conf("layer_coating_lumen", 1, "fail", "Coating closes the channel lumen",
                       "A %.1f um coating leaves %.1f um of the %.0f um lumen." % (1e3 * self.coat_t, 1e3 * eff, 1e3 * self.lum_w),
                       ["sf.coating_nm", "ch.lumen_um"],
                       [self._fix("Reduce coating to %.0f nm" % (0.1 * self.lum_w * 1e6), 1, sf__coating_nm=round(0.1 * self.lum_w * 1e6)),
                        self._fix("Widen the lumen to %.0f um" % (1e3 * (self.lum_w + 4 * self.coat_t)), 3, ch__lumen_um=round(1e3 * (self.lum_w + 4 * self.coat_t)))],
                       levels=[3], value=1e3 * eff, threshold=0.5e3 * self.lum_w, units="um")
        amp = float(p["sf.texture_amplitude_nm"])
        if p["sf.texture"] != "none" and (self.coat_ch or self.coat_sh) and float(p["sf.coating_nm"]) >= amp:
            self._conf("layer_coating_texture", 1, "warn", "Coating may smooth the nano-texture",
                       "A %.0f nm coating is at least as thick as the %.0f nm texture relief, so the texture may be filled in."
                       % (float(p["sf.coating_nm"]), amp), ["sf.coating_nm", "sf.texture_amplitude_nm"],
                       [self._fix("Thin the coating to %.0f nm" % (0.5 * amp), 1, sf__coating_nm=round(0.5 * amp, 1)),
                        self._fix("Raise texture amplitude to %.0f nm" % (2 * float(p["sf.coating_nm"])), 1,
                                  sf__texture_amplitude_nm=round(min(2000.0, 2 * float(p["sf.coating_nm"])), 1))])

    def _check_ends(self):
        p = self.p
        for side in ("tib", "fem"):
            e, fp = self.ends[side], self.fp[side]
            lab = "Tibial" if side == "tib" else "Femoral"
            k = "end.%s." % side
            base = self._flare_base(side)
            inside = points_in_polygon(e["outline"], base)
            dist = polygon_distance(e["outline"], base)
            out = float(np.where(inside, 0.0, dist).max())
            e["base_outside_mm"] = out
            if out > 0.05:
                fixes = [self._fix("Fit the flare base to the footprint", 6, **{(k + "flare_mode").replace(".", "__"): "fill_footprint"})]
                if p[k + "flare_mode"] == "manual":
                    sc = 0.95 * min(fp.L / max(e["Wf"], 1e-6), fp.W / max(e["Tf"], 1e-6))
                    fixes.append(self._fix("Shrink the flare base to %.1f x %.1f mm" % (sc * e["Wf"], sc * e["Tf"]), 6,
                                           **{(k + "flare_width_mm").replace(".", "__"): round(sc * e["Wf"], 2),
                                              (k + "flare_thickness_mm").replace(".", "__"): round(sc * e["Tf"], 2)}))
                fixes.append(self._fix("Enlarge the footprint to %.1f x %.1f mm" % (fp.L + 2 * out + 0.4, fp.W + 2 * out + 0.4), 6,
                                       **{(k + "size_mode").replace(".", "__"): "manual",
                                          (k + "length_mm").replace(".", "__"): round(fp.L + 2 * out + 0.4, 2),
                                          (k + "width_mm").replace(".", "__"): round(fp.W + 2 * out + 0.4, 2)}))
                self._conf("layer_flare_footprint_%s" % side, 6, "fail", "%s flare base extends beyond the footprint" % lab,
                           "The flare base (%.1f x %.1f mm) reaches %.2f mm outside the %s footprint outline (%.1f x %.1f mm)."
                           % (e["Wf"], e["Tf"], out, fp.kind, fp.L, fp.W),
                           [k + "flare_width_mm", k + "flare_thickness_mm", k + "length_mm", k + "width_mm", k + "flare_mode"],
                           fixes, levels=[5], value=out, threshold=0.05, units="mm")
            dev = float(np.abs(fp.bone.sdf(fp.to3d(e["outline"]))).max())
            if dev > 0.3:
                self._conf("layer_footprint_bone_%s" % side, 6, "fail", "%s scaffold footprint leaves the bone" % lab,
                           "Footprint points lie up to %.2f mm off the bone surface (offset too far from the anatomical footprint?)." % dev,
                           [k + "offset_long_mm", k + "offset_short_mm"],
                           [self._fix("Reset the %s footprint offset" % lab.lower(), 6, **{(k + "offset_long_mm").replace(".", "__"): 0.0, (k + "offset_short_mm").replace(".", "__"): 0.0})])
            if p[k + "fixation"] == "plug":
                D = float(p[k + "plug_diameter_mm"])
                lim = min(fp.L, fp.W) - 1.0
                if D > lim:
                    self._conf("layer_plug_%s" % side, 6, "fail", "%s fixation plug is wider than the footprint" % lab,
                               "A %.1f mm plug needs at least 0.5 mm of footprint around it; the footprint is %.1f x %.1f mm." % (D, fp.L, fp.W),
                               [k + "plug_diameter_mm", k + "length_mm", k + "width_mm"],
                               [self._fix("Reduce the plug to %.1f mm" % max(2.0, lim), 6, **{(k + "plug_diameter_mm").replace(".", "__"): round(max(2.0, lim), 1)})])
            if e["twist_deg"] > 50:
                self._conf("layer_flare_twist_%s" % side, 6, "warn", "%s flare twists strongly" % lab,
                           "The sheath's wide axis meets the footprint's long axis at %.0f deg, so the flare twists by that much over %.1f mm."
                           % (e["twist_deg"], e["Lf"]), [k + "rotation_deg", "sheath.orientation_profile"],
                           [self._fix("Reset the %s footprint rotation" % lab.lower(), 6, **{(k + "rotation_deg").replace(".", "__"): 0.0})], levels=[5])
            sg = getattr(self, "stagger", {}).get(side)
            if sg and sg["early"]:
                self._conf("layer_staggered_%s" % side, 6, "info", "%s channel endpoints are staggered" % lab,
                           "%d of %d channels end before the %.1f mm stop, between %.1f and %.1f mm from bone, where they would otherwise "
                           "come closer than the %.0f um spacing (the construct turns onto the bone there); the others reach the stop."
                           % (sg["early"], sg["total"], e["stop"], sg["range_mm"][0], sg["range_mm"][1], 1000 * self.cgap),
                           [k + "endpoints", k + "endpoint_stop_mm", k + "flare_length_mm", "ch.spacing_um"], [], levels=[3])
            if e["Lb"] + e["Lg"] > e["Lf"] + 0.01:
                self._conf("layer_gradient_extent_%s" % side, 6, "info", "%s material gradient extends past the flare" % lab,
                           "Bone-facing zone %.1f mm + gradient %.1f mm reach %.1f mm from bone, beyond the %.1f mm flare; the graded "
                           "material continues into the body." % (e["Lb"], e["Lg"], e["Lb"] + e["Lg"], e["Lf"]),
                           [k + "gradient_mm", k + "bone_zone_mm", k + "flare_length_mm"], [], levels=[5])

    def _check_compositions(self):
        from acl_engine.v2.design import COMPOSITION_PARAMS
        for pid, level in COMPOSITION_PARAMS.items():
            v = self.p.get(pid) or {}
            tot = sum(float(x) for x in v.values()) if isinstance(v, dict) else 0.0
            if v and abs(tot - 1) > 1e-3:
                self._conf("layer_comp_%s" % pid.replace(".", "_"), level, "warn", "Composition total is %.3f" % tot,
                           "%s fractions sum to %.3f; geometry colours and exports use the normalised fractions." % (pid, tot),
                           [pid], [{"label": "Normalise to 1", "level": level, "set": {pid: {k: round(float(x) / tot, 4) for k, x in v.items()}}}],
                           value=tot, threshold=1.0)

    # ------------------------------------------------------------------ outputs
    def _outputs(self):
        p, mid = self.p, self.mid
        g = [sum(1 for c in self.chans if c["group"] == q) for q in (0, 1, 2)]
        lum_area = math.pi / 4 * self.lum_w * self.lum_h
        eff = getattr(self, "eff_lumen", self.lum_w)
        sheath_lumen = float(_se_area(self.Ai[mid], self.Bi[mid], self.nx[mid]))
        n_ch = len(self.chans)
        void = n_ch * lum_area / max(sheath_lumen, 1e-9)
        per_lumen = math.pi * self.lum_w
        # Level 3
        self.summary[3] = {"channels": n_ch, "am": g[0], "pl": g[1], "shared": g[2], "outside_um": 1000 * self.od_w,
                           "pitch_um": 1000 * self.pitch, "min_gap_um": None if self.ch_min_gap is None else 1000 * self.ch_min_gap}
        # where the paths end (moves when a footprint, flare or endpoint setting changes)
        for lines, lv, nm in ((self.ch_lines, 3, "channel"), (self.braid_lines, 4, "braid")):
            if lines:
                for side, i in (("tib", 0), ("fem", -1)):
                    c = np.mean([ln["P"][i] for ln in lines], 0)
                    self.summary[lv]["%s_%s_ends_centre_mm" % (side, nm)] = "%.1f, %.1f, %.1f" % tuple(c)
        self.calc[3] += [
            ("Outside diameter (lumen + 2 x wall)", 1000 * self.od_w, "um", "geometry"),
            ("Centre pitch (outside diameter + spacing)", 1000 * self.pitch, "um", "geometry"),
            ("Channels placed: AM-like / PL-like / shared", "%d / %d / %d" % tuple(g), "", "geometry"),
            ("Channel lumen fraction of the sheath lumen (mid-length)", 100 * void, "%", "geometry"),
            ("Lumen wall area per mm of length (all channels)", n_ch * per_lumen, "mm2/mm", "geometry"),
            ("Smallest measured wall-to-wall gap", None if self.ch_min_gap is None else 1000 * self.ch_min_gap, "um", "geometry (3D)"),
        ]
        if p["ch.connections"] == "ports":
            self.calc[3].append(("Connection ports", len(self.ports), "", "geometry"))
        hl = float(p["ch.assumed_degradation_weeks"])
        self.assume[3].append(("Channel-wall degradation half-life", hl, "weeks", "entered assumption"))
        self.calc[3].append(("Wall mass remaining at 12 weeks (first-order decay of the entered half-life)", 100 * 0.5 ** (12 / hl), "%", "calculated from an assumption, not validated"))
        # Level 4
        rows = []
        for m, bi in zip(self.mant, self.braid_info):
            lab = m["key"].upper() + "-like"
            E = float(p["rf.%s.assumed_modulus_mpa" % m["key"]])
            Ay = math.pi * (m["d"] / 2) ** 2
            EA = m["N"] * E * Ay * math.cos(m["alpha"]) ** 4
            rows += [("%s envelope (inside the braid), mid-length" % lab, "%.2f x %.2f" % (2 * m["a"][mid], 2 * m["b"][mid]), "mm",
                      "derived (auto)" if p["rf.size_mode"] == "auto" else "user-set"),
                     ("%s braid wall thickness (2 x strand)" % lab, 2 * m["d"], "mm", "geometry"),
                     ("%s braid pitch at mid-length" % lab, float(bi["pitch_mm"][len(bi["pitch_mm"]) // 2]), "mm", "geometry"),
                     ("%s strand gap within a family (min)" % lab, float(np.min(bi["spacing_mm"])), "mm", "geometry"),
                     ("%s braid winding end to end" % lab, bi["winding_deg"], "deg", "geometry (braid winding, not yarn twist)"),
                     ("%s reference length (path x (1 + slack))" % lab, bi["length_mm"] * (1 + m["slack"]), "mm", "geometry"),
                     ("%s strain before the mantle carries load (= slack)" % lab, 100 * m["slack"], "%", "geometry"),
                     ("%s axial stiffness estimate N*E*A*cos^4(angle)" % lab, EA, "N", "calculated from the entered modulus; first-order, not validated")]
            self.assume[4].append(("%s strand modulus" % lab, E, "MPa", "entered assumption"))
            rows.append(("%s yarn twist (total fiber rotation per strand; rate)" % lab,
                         "%+.1f deg; %+.2f deg/mm" % (m["twist"], m["twist"] / max(bi["length_mm"], 1e-6)), "", "user-set; independent of braid angle"))
            self.summary[4][m["key"] + "_envelope_mm"] = "%.2f x %.2f" % (2 * m["a"][mid], 2 * m["b"][mid])
            self.summary[4][m["key"] + "_pitch_mm"] = float(bi["pitch_mm"][len(bi["pitch_mm"]) // 2])
            self.summary[4][m["key"] + "_perimeter_mm"] = float(bi["perimeter_mm"][len(bi["perimeter_mm"]) // 2])
            self.summary[4][m["key"] + "_strand_length_mm"] = bi["length_mm"]
        rows.append(("Gap between AM-like and PL-like braids (min)", self.mantle_gap, "mm", "geometry"))
        self.summary[4]["mantle_gap_mm"] = self.mantle_gap
        self.calc[4] += rows
        # Level 5
        self.summary[5] = {"outer_mm": "%.2f x %.2f" % (2 * self.A[mid], 2 * self.B[mid]),
                           "clearance_margin_mm": min(self.mantle_margin.values()), "protrusion_mm": self.envelope_protrusion,
                           "transition_band_mm": "%.2f" % float(self.p["rf.transition_mm"])}
        P_o = float(_se_perimeter(self.A[mid], self.B[mid], self.nx[mid]))
        open_frac = 0.0
        if p["sheath.pores"] == "explicit":
            dp = float(p["sheath.pore_diameter_um"]) * 1e-3
            body_len = max(1e-6, (self.ends["fem"]["s_fs"] - self.ends["tib"]["s_fs"]) * self.L)
            open_frac = getattr(self, "pore_count", 0) * math.pi * dp * dp / 4 / max(P_o * body_len, 1e-9)
        kperm = 10 ** float(p["sheath.assumed_permeability_log10"])
        q = kperm * 1000.0 / (WATER_VISCOSITY_PA_S * self.tw * 1e-3)       # m/s per 1 kPa
        hs = float(p["sheath.assumed_degradation_weeks"])
        self.calc[5] += [("Outer size at mid-length", "%.2f x %.2f" % (2 * self.A[mid], 2 * self.B[mid]), "mm",
                          "derived" if p["sheath.size_mode"] != "profiles" else "user-set"),
                         ("Smallest clearance to the braids (requested %.2f)" % self.cs, min(self.mantle_margin.values()), "mm", "geometry"),
                         ("Largest protrusion beyond the intact envelope", self.envelope_protrusion, "mm", "geometry"),
                         ("Explicit pores", getattr(self, "pore_count", 0), "", "geometry"),
                         ("Geometric open-area fraction of the body wall", 100 * open_frac, "%", "geometry (pores only; no transport claim)"),
                         ("Darcy flux per 1 kPa across the wall", q * 1e6, "um/s", "calculated from the entered permeability only; not from pores; not validated"),
                         ("Sheath mass remaining at 12 weeks", 100 * 0.5 ** (12 / hs), "%", "calculated from an assumption, not validated")]
        self.assume[5] += [("Wall permeability", kperm, "m2", "entered assumption (independent of drawn pores)"),
                           ("Sheath degradation half-life", hs, "weeks", "entered assumption")]
        # Level 2
        width, ridge, depth = float(p["gr.width_um"]), float(p["gr.ridge_um"]), float(p["gr.depth_um"])
        period = width + ridge
        n_true = max(1, int(math.floor(math.pi * float(p["ch.lumen_um"]) / period)))
        side_area = 2 * depth + max(width - 2 * float(p["gr.bottom_radius_um"]), 0) + math.pi * float(p["gr.bottom_radius_um"])
        area_factor = (ridge + side_area) / period
        self.summary[2] = {"grooves_per_lumen": n_true if self.gr_ch else 0, "residual_wall_um": float(p["ch.wall_um"]) - depth if self.gr_ch else None}
        self.calc[2] += [("Groove period (width + ridge)", period, "um", "geometry"),
                         ("Grooves around one channel lumen", n_true if self.gr_ch else 0, "", "geometry"),
                         ("Channel wall left under a groove", float(p["ch.wall_um"]) - depth, "um", "geometry"),
                         ("Sheath wall left under a groove", 1000 * self.tw - depth, "um", "geometry"),
                         ("Margin above the minimum residual wall (channel / sheath)", "%.1f / %.1f" % (
                             float(p["ch.wall_um"]) - depth - float(p["gr.min_residual_um"]), 1000 * self.tw - depth - float(p["gr.min_residual_um"])),
                          "um", "geometry vs the design rule"),
                         ("Surface-area factor of the grooved surface", area_factor, "x", "geometry")]
        # Level 1
        amp, lam = float(p["sf.texture_amplitude_nm"]), float(p["sf.texture_period_nm"])
        self.summary[1] = {"effective_lumen_um": 1000 * eff, "coating_nm": float(p["sf.coating_nm"]),
                           "texture": ("none" if p["sf.texture"] == "none" else "%s %.0f nm x %.0f nm at %+.0f deg to the channel" % (
                               p["sf.texture"], lam, amp, float(p["sf.texture_orientation_deg"])))}
        self.calc[1] += [("Effective channel lumen after coating", 1000 * eff, "um", "geometry" if self.coat_explicit else "metadata (coating not in geometry)"),
                         ("Texture features per um2", (1000.0 / lam) ** (2 if p["sf.texture"] == "pillars" else 1) if p["sf.texture"] != "none" else 0,
                          "per um2" if p["sf.texture"] == "pillars" else "per um", "geometry"),
                         ("Texture aspect (amplitude / period)", amp / lam, "", "geometry")]
        # Level 6
        for side in ("tib", "fem"):
            e, fp = self.ends[side], self.fp[side]
            lab = "Tibial" if side == "tib" else "Femoral"
            body = float(_se_area(self.A[e["i_fs"]], self.B[e["i_fs"]], self.nx[e["i_fs"]]))
            base = float(_se_area(e["Wf"] / 2, e["Tf"] / 2, e["nf"]))
            self.calc[6] += [("%s footprint (%s)" % (lab, fp.kind), "%.1f x %.1f" % (fp.L, fp.W), "mm",
                              "derived" if p["end.%s.size_mode" % side] == "match_anatomy" else "user-set"),
                             ("%s footprint area" % lab, fp.area(), "mm2", "geometry"),
                             ("%s flare base" % lab, "%.1f x %.1f" % (e["Wf"], e["Tf"]), "mm",
                              "derived" if p["end.%s.flare_mode" % side] == "fill_footprint" else "user-set"),
                             ("%s flare area ratio (base / body section)" % lab, base / max(body, 1e-9), "x", "geometry"),
                             ("%s flare twist" % lab, e["twist_deg"], "deg", "geometry"),
                             ("%s zones from bone: bone-facing / gradient" % lab, "%.1f / %.1f" % (e["Lb"], e["Lg"]), "mm", "user-set")]
            self.summary[6][side + "_footprint_mm"] = "%.1f x %.1f" % (fp.L, fp.W)
            self.summary[6][side + "_footprint_centre_mm"] = "%.1f, %.1f, %.1f" % tuple(fp.center)
            self.summary[6][side + "_flare_twist_deg"] = round(e["twist_deg"], 1)
            self.summary[6][side + "_flare_base_mm"] = "%.1f x %.1f" % (e["Wf"], e["Tf"])
            self.end_info[side] = {"center": fp.center.tolist(), "normal": fp.normal.tolist(), "outline": fp.kind,
                                   "length_mm": fp.L, "width_mm": fp.W, "flare_length_mm": e["Lf"], "flare_base_mm": [e["Wf"], e["Tf"]],
                                   "twist_deg": e["twist_deg"], "base_outside_mm": e.get("base_outside_mm", 0.0),
                                   "zones_mm": {"bone_facing": [0, e["Lb"]], "gradient": [e["Lb"], e["Lb"] + e["Lg"]],
                                                "ligament_facing": [e["Lb"] + e["Lg"], max(e["Lf"], e["Lb"] + e["Lg"])]},
                                   "composition": {z: _normalize(p["end.%s.comp_%s" % (side, z)]) for z in BAND}}

    def report(self):
        if not self.active:
            return None
        mid = self.mid

        def rows(d):
            return {str(k): [{"label": r[0], "value": r[1], "units": r[2], "basis": r[3]} for r in v] for k, v in d.items()}
        return {"active": True, "derived": self.derived, "derived_profiles": self.derived_profiles,
                "calculated": rows(self.calc), "assumptions": rows(self.assume), "summary": {str(k): v for k, v in self.summary.items()},
                "groups": self.group_info, "shared_fit": self.shared_fit, "ends": self.end_info,
                "sheath_mid_mm": [2 * float(self.A[mid]), 2 * float(self.B[mid])], "wall_mm": self.tw,
                "conflicts": self.conflicts}


# ======================================================================== meshes
def _annular_shell(Po, Pi):
    """Closed shell between outer rings Po and inner rings Pi (m, K, 3), with end rims."""
    m, K, _ = Po.shape
    V = np.vstack([Po.reshape(-1, 3), Pi.reshape(-1, 3)])
    off = m * K
    i = np.arange(m - 1)[:, None]
    j = np.arange(K)[None, :]
    a0 = i * K + j
    a1 = i * K + (j + 1) % K
    b0, b1 = a0 + K, a1 + K
    Fo = np.concatenate([np.stack([a0, a1, b1], -1).reshape(-1, 3), np.stack([a0, b1, b0], -1).reshape(-1, 3)])
    Fi = np.concatenate([np.stack([a0, b1, a1], -1).reshape(-1, 3), np.stack([a0, b0, b1], -1).reshape(-1, 3)]) + off
    jj = np.arange(K)
    r0o, r0i = jj, jj + off
    r1o, r1i = (m - 1) * K + jj, (m - 1) * K + jj + off
    n0o, n0i = (jj + 1) % K, (jj + 1) % K + off
    n1o, n1i = (m - 1) * K + (jj + 1) % K, (m - 1) * K + (jj + 1) % K + off
    R0 = np.concatenate([np.stack([r0o, r0i, n0i], 1), np.stack([r0o, n0i, n0o], 1)])
    R1 = np.concatenate([np.stack([r1o, n1i, r1i], 1), np.stack([r1o, n1o, n1i], 1)])
    F = np.vstack([Fo, Fi, R0, R1]).astype(np.int64)
    if mesher.audit(V, F)["volume_mm3"] < 0:
        F = F[:, ::-1]
    return V, F


def _closed_loft(P):
    m, K, _ = P.shape
    V = np.vstack([P.reshape(-1, 3), P[0].mean(0)[None], P[-1].mean(0)[None]])
    i = np.arange(m - 1)[:, None]
    j = np.arange(K)[None, :]
    a0 = i * K + j
    a1 = i * K + (j + 1) % K
    F = np.concatenate([np.stack([a0, a1, a1 + K], -1).reshape(-1, 3), np.stack([a0, a1 + K, a0 + K], -1).reshape(-1, 3)])
    c0, c1 = m * K, m * K + 1
    jj = np.arange(K)
    F = np.vstack([F, np.stack([np.full(K, c0), (jj + 1) % K, jj], 1),
                   np.stack([np.full(K, c1), (m - 1) * K + jj, (m - 1) * K + (jj + 1) % K], 1)]).astype(np.int64)
    if mesher.audit(V, F)["volume_mm3"] < 0:
        F = F[:, ::-1]
    return V, F


def hollow_tube_mesh(P, ro, ri, U, nsec=12, aspect=1.0, exponent=2.0):
    """Closed annular solid along P (n,3): outer/inner radii (n,), cross-section wide axis
    U (n,3). Non-circular sections are superellipses with the given width/height aspect."""
    P = np.asarray(P, float)
    n = P.shape[0]
    T = np.gradient(P, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    U = np.asarray(U, float) - (np.asarray(U, float) * T).sum(1, keepdims=True) * T
    U /= np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
    W = np.cross(T, U)
    a = np.linspace(0, 2 * np.pi, nsec, endpoint=False)
    c, s = np.cos(a), np.sin(a)
    e = 2.0 / exponent
    cx = np.sign(c) * np.abs(c) ** e * math.sqrt(aspect)
    sy = np.sign(s) * np.abs(s) ** e / math.sqrt(aspect)
    ro = np.broadcast_to(np.asarray(ro, float), (n,))
    ri = np.broadcast_to(np.asarray(ri, float), (n,))
    ring = cx[None, :, None] * U[:, None, :] + sy[None, :, None] * W[:, None, :]
    Vo = P[:, None, :] + ro[:, None, None] * ring
    Vi = P[:, None, :] + ri[:, None, None] * ring
    return _annular_shell(Vo, Vi)


def _se_area(a, b, n):
    g = math.gamma(1 + 1 / n) ** 2 / math.gamma(1 + 2 / n)
    return 4 * a * b * g


def _se_perimeter(a, b, n, K=512):
    Q = se_points(np.array([a]), np.array([b]), np.array([n]), K)[0]
    return float(np.linalg.norm(np.roll(Q, -1, 0) - Q, axis=1).sum())
