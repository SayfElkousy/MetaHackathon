"""Distal femur, proximal tibia and articular cartilage.

Two bone sources share one interface (sdf, project, normal, mesh):

- ProceduralBone: implicit (signed-distance-like) shapes built from smooth
  unions of ellipsoids, superellipse prisms and height fields. Features:
  medial and lateral condyles, intercondylar notch with an inclined roof,
  trochlear groove, epicondyles, metaphyseal flare and shaft (femur); plateau
  with posterior slope, medial concavity, intercondylar eminence (two spines),
  anterior and posterior intercondylar areas, tuberosity and shaft (tibia).
  This is a LABELLED APPROXIMATION of adult anatomy (status "assumed"), not a
  scan. No openly licensed femur/tibia mesh with known scale was available to
  this project (see docs/ANATOMY_SOURCES.md).
- MeshBone: a patient surface mesh (STL/OBJ) supplied by the MRI-analysis
  model, already transformed into the model frame by the adapter. Distances
  use a KD-tree over a densified vertex set (approximate, documented).

Frame: mm, +X lateral, +Y anterior, +Z proximal, origin tibial plateau
centre, right knee, full extension. Knee flexion rotates the femur about the
mediolateral axis through the condyle centres (illustrative pose only).
"""
import hashlib
import json
import math
import numpy as np
from scipy.spatial import cKDTree
from acl_engine.v2 import mesher


# ---------------------------------------------------------------- primitives
def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


def sd_ellipsoid(P, c, r):
    q = (P - np.asarray(c)) / np.asarray(r)
    k0 = np.linalg.norm(q, axis=-1)
    k1 = np.linalg.norm(q / np.asarray(r), axis=-1)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def sd_superellipse2(x, y, a, b, n):
    lev = (np.abs(x / a) ** n + np.abs(y / b) ** n) ** (1.0 / n)
    return (lev - 1.0) * np.minimum(a, b)


def sd_capsule(P, a, b, r):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    pa = P - a
    ba = b - a
    h = np.clip((pa @ ba) / (ba @ ba), 0.0, 1.0)
    return np.linalg.norm(pa - h[..., None] * ba, axis=-1) - r


def _sstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def rot_x(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


# ---------------------------------------------------------------- parameters
BONE_DEFAULTS = {
    "bone.femur_bicondylar_width_mm": 78.0,
    "bone.tibia_plateau_width_mm": 76.0,
    "bone.notch_width_mm": 19.0,
    "bone.tibial_slope_deg": 7.0,
    "bone.cartilage_femoral_mm": 2.2,
    "bone.cartilage_tibial_mm": 2.0,
}
FLEX_AXIS = np.array([0.0, -6.0, 24.0])  # point on the flexion axis (along X)


class ProceduralBone:
    kind = "procedural"

    def __init__(self, which, prm, flexion_deg=0.0):
        self.which = which
        self.p = {k: float(prm.get(k, v)) for k, v in BONE_DEFAULTS.items()}
        self.flex = float(flexion_deg) if which in ("femur", "femoral_cartilage") else 0.0
        self.R = rot_x(-self.flex)  # flexion brings the femoral shaft anterior

    # pose: local (extension) <-> world
    def to_local(self, P):
        if self.flex == 0.0:
            return P
        return (P - FLEX_AXIS) @ self.R + FLEX_AXIS  # R^T applied row-wise

    def to_world(self, P):
        if self.flex == 0.0:
            return P
        return (P - FLEX_AXIS) @ self.R.T + FLEX_AXIS

    def key(self):
        # Only the parameters this bone actually reads: a scaffold or injury
        # edit must not regenerate the bones (they dominate build time).
        if self.which in ("femur", "femoral_cartilage"):
            use = ("bone.femur_bicondylar_width_mm", "bone.notch_width_mm",
                   "bone.cartilage_femoral_mm")
        else:
            use = ("bone.tibia_plateau_width_mm", "bone.tibial_slope_deg",
                   "bone.cartilage_tibial_mm")
        d = {k: self.p.get(k) for k in use}
        d.update(which=self.which, flex=round(self.flex, 3))
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

    # ---------------------------------------------------------------- femur
    def _femur_local(self, P):
        s = self.p["bone.femur_bicondylar_width_mm"] / 78.0
        Q = P / s
        x, y, z = Q[..., 0], Q[..., 1], Q[..., 2]
        lat = sd_ellipsoid(Q, (23.0, -6.0, 23.5), (13.5, 25.0, 19.5))
        med = sd_ellipsoid(Q, (-24.5, -6.0, 24.5), (14.5, 26.5, 20.5))
        cond = smin(lat, med, 3.0)
        # distal anterior block (trochlea), rounded box
        c = np.array([-0.5, 9.0, 36.0])
        hb = np.array([35.0, 12.5, 15.0])
        d = np.abs(Q - c) - (hb - 7.0)
        block = np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(d.max(-1), 0) - 7.0
        f = smin(cond, block, 6.0)
        # metaphysis: section shrinks from condylar block to shaft
        t = _sstep((z - 34.0) / 42.0)
        a = 37.0 * (1 - t) + 15.5 * t
        b = 19.0 * (1 - t) + 14.0 * t
        yc = 1.0 * (1 - t) - 1.0 * t
        meta = smax(sd_superellipse2(x - 0.0, y - yc, a, b, 2.4), 30.0 - z, 4.0)
        f = smin(f, meta, 6.0)
        # epicondyles
        f = smin(f, sd_ellipsoid(Q, (36.5, -3.0, 33.0), (5.0, 7.0, 7.0)), 3.0)
        f = smin(f, sd_ellipsoid(Q, (-39.0, -4.0, 35.0), (5.0, 8.0, 8.0)), 3.0)
        # intercondylar notch (open posteriorly and distally, roof inclined)
        nw = self.p["bone.notch_width_mm"]
        xl, xm = 8.5 * nw / 19.0, -10.5 * nw / 19.0
        notch = self._notch(x, y, z)
        f = smax(f, -notch, 2.0)
        # trochlear groove
        f = smax(f, -sd_capsule(Q, (0.5, 25.0, 20.0), (0.5, 24.0, 52.0), 5.6), 2.5)
        # proximal cut
        f = np.maximum(f, z - 72.0)
        return f * s

    def _notch(self, x, y, z):
        """Arch-shaped intercondylar notch, open posteriorly and distally."""
        nw = self.p["bone.notch_width_mm"]
        xc = -1.0 * nw / 19.0
        hw = 0.5 * nw
        zroof = 16.0 + 0.45 * (8.0 - y)
        side = np.abs(x - xc) - hw
        # rounded arch: the roof corners blend into the walls
        return smax(smax(side, z - zroof, 5.0), y - 9.0, 3.0)

    def _femoral_cartilage_local(self, P):
        s = self.p["bone.femur_bicondylar_width_mm"] / 78.0
        f = self._femur_local(P)
        t = self.p["bone.cartilage_femoral_mm"]
        shell = np.maximum(f - t, -f)
        Q = P / s
        x, y, z = Q[..., 0], Q[..., 1], Q[..., 2]
        notch = self._notch(x, y, z)
        lat = sd_ellipsoid(Q, (23.0, -6.0, 23.5), (13.5, 25.0, 19.5))
        med = sd_ellipsoid(Q, (-24.5, -6.0, 24.5), (14.5, 26.5, 20.5))
        troch = sd_ellipsoid(Q, (-0.5, 17.0, 31.0), (24.0, 12.0, 15.0))
        near = smin(np.minimum(lat, med) - 1.5, troch, 4.0)
        region = smax(smax(1.5 - notch, near, 1.5), z - 47.0, 3.0)
        return np.maximum(shell, region * s)

    # ---------------------------------------------------------------- tibia
    def _tibia_top(self, x, y):
        sl = math.tan(math.radians(self.p["bone.tibial_slope_deg"]))
        zt = (y - 1.0) * sl
        zt = zt - 1.6 * np.exp(-((x + 20.0) ** 2 / 12.0 ** 2 + (y - 0.0) ** 2 / 14.0 ** 2))
        zt = zt + 0.5 * np.exp(-((x - 20.0) ** 2 / 12.0 ** 2 + (y + 1.0) ** 2 / 14.0 ** 2))
        zt = zt + 5.5 * np.exp(-((x + 4.8) ** 2 / 2.4 ** 2 + (y + 3.0) ** 2 / 5.0 ** 2))
        zt = zt + 4.5 * np.exp(-((x - 4.6) ** 2 / 2.4 ** 2 + (y + 5.5) ** 2 / 5.0 ** 2))
        zt = zt - 0.8 * np.exp(-((x - 1.5) ** 2 / 6.0 ** 2 + (y - 8.0) ** 2 / 7.0 ** 2))
        zt = zt - 6.0 / (1.0 + np.exp((y + 15.0) / 2.0)) * np.exp(-(x ** 2) / 9.0 ** 2)
        return zt

    def _tibia_local(self, P):
        s = self.p["bone.tibia_plateau_width_mm"] / 76.0
        Q = P / s
        x, y, z = Q[..., 0], Q[..., 1], Q[..., 2]
        zt = self._tibia_top(x, y)
        top = z - zt
        plate = smax(sd_superellipse2(x, y - 1.0, 38.0, 25.0, 2.6), top, 1.5)
        plate = smax(plate, -14.0 - z, 4.0)
        # posterior concavity between the condyles (PCL fossa outline)
        plate = smax(plate, -sd_ellipsoid(Q, (0.0, -29.0, -3.0), (9.0, 7.0, 12.0)), 2.0)
        t = _sstep((-z - 10.0) / 42.0)
        a = 38.0 * (1 - t) + 15.0 * t
        b = 24.5 * (1 - t) + 14.0 * t
        yc = 1.0 * (1 - t) - 1.0 * t
        meta = smax(sd_superellipse2(x, y - yc, a, b, 2.3), z + 9.0, 3.0)
        f = smin(plate, meta, 5.0)
        f = smin(f, sd_ellipsoid(Q, (6.0, 21.0, -30.0), (8.0, 6.0, 13.0)), 4.0)
        f = np.maximum(f, -58.0 - z)
        return f * s

    def _tibial_cartilage_local(self, P):
        s = self.p["bone.tibia_plateau_width_mm"] / 76.0
        f = self._tibia_local(P)
        t = self.p["bone.cartilage_tibial_mm"]
        shell = np.maximum(f - t, -f)
        Q = P / s
        x, y, z = Q[..., 0], Q[..., 1], Q[..., 2]
        zt = self._tibia_top(x, y)
        # two oval articular facets (medial larger), not the intercondylar area
        med = np.hypot((x + 19.0) / 14.5, (y - 0.0) / 20.0) - 1.0
        lat = np.hypot((x - 19.0) / 13.0, (y + 1.0) / 17.0) - 1.0
        facets = np.minimum(med, lat) * 12.0
        region = np.maximum(zt - 2.5 - z, facets)
        return np.maximum(shell, region * s)

    # ---------------------------------------------------------------- interface
    def sdf(self, P):
        P = np.asarray(P, float)
        L = self.to_local(P)
        return {"femur": self._femur_local, "tibia": self._tibia_local,
                "femoral_cartilage": self._femoral_cartilage_local,
                "tibial_cartilage": self._tibial_cartilage_local}[self.which](L)

    def grad(self, P, eps=0.05):
        P = np.asarray(P, float)
        g = np.zeros_like(P)
        for a in range(3):
            d = np.zeros(3)
            d[a] = eps
            g[:, a] = (self.sdf(P + d) - self.sdf(P - d)) / (2 * eps)
        return g

    def normal(self, P):
        g = self.grad(P)
        return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)

    def project(self, P, iters=8):
        """Newton projection of points onto the zero level set."""
        P = np.array(P, float)
        for _ in range(iters):
            f = self.sdf(P)
            g = self.grad(P)
            P -= (f / np.maximum((g * g).sum(1), 1e-9))[:, None] * g
        return P

    def bounds(self):
        if self.which in ("femur", "femoral_cartilage"):
            s = self.p["bone.femur_bicondylar_width_mm"] / 78.0
            lo, hi = np.array([-47, -38, -1.0]) * s, np.array([45, 31, 74.0]) * s
        else:
            s = self.p["bone.tibia_plateau_width_mm"] / 76.0
            lo, hi = np.array([-43, -31, -60.0]) * s, np.array([43, 33, 12.0]) * s
        return lo, hi

    def mesh(self, h=0.9):
        lo, hi = self.bounds()
        xs = [np.arange(lo[a], hi[a] + h, h) for a in range(3)]
        G = np.stack(np.meshgrid(*xs, indexing="ij"), -1)
        # sample in the LOCAL frame, mesh, then pose (cheaper than posing the grid)
        fn = {"femur": self._femur_local, "tibia": self._tibia_local,
              "femoral_cartilage": self._femoral_cartilage_local,
              "tibial_cartilage": self._tibial_cartilage_local}[self.which]
        F = fn(G.reshape(-1, 3)).reshape(G.shape[:3])
        V, T = mesher.surface_nets(F, lo, h)
        V = mesher.smooth_taubin(V, T, iters=4)
        return self.to_world(V), T


class MeshBone:
    """Patient surface mesh in model coordinates (mm)."""
    kind = "mesh"

    def __init__(self, which, V, F, flexion_deg=0.0, source=None):
        self.which = which
        self.V0 = np.asarray(V, float)
        self.F = np.asarray(F, np.int64)
        self.source = source or {}
        self.flex = float(flexion_deg) if which == "femur" else 0.0
        self.R = rot_x(-self.flex)
        self.V = self.to_world(self.V0)
        # densify with face centroids and edge midpoints for distance queries
        a, b, c = (self.V[self.F[:, i]] for i in range(3))
        n = np.cross(b - a, c - a)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        pts = np.vstack([(a + b + c) / 3, (a + b) / 2, (b + c) / 2, (c + a) / 2])
        nrm = np.vstack([n, n, n, n])
        self._pts = pts
        self._nrm = nrm
        self._tree = cKDTree(pts)

    def to_world(self, P):
        if self.flex == 0.0:
            return P
        return (P - FLEX_AXIS) @ self.R.T + FLEX_AXIS

    def key(self):
        h = hashlib.sha1(self.V0.astype(np.float32).tobytes())
        h.update(self.F.astype(np.int32).tobytes())
        h.update(str(round(self.flex, 3)).encode())
        return h.hexdigest()[:16]

    def sdf(self, P):
        """Signed distance: point-to-plane of the nearest surface sample when close
        (accurate near the surface), point distance otherwise."""
        P = np.asarray(P, float)
        d, i = self._tree.query(P, workers=-1)
        dp = ((P - self._pts[i]) * self._nrm[i]).sum(1)
        sgn = np.where(dp >= 0, 1.0, -1.0)
        return np.where(d < 1.5, dp, d * sgn)

    def normal(self, P):
        _, i = self._tree.query(np.asarray(P, float))
        return self._nrm[i]

    def project(self, P, iters=2):
        P = np.asarray(P, float)
        _, i = self._tree.query(P)
        q = self._pts[i]
        n = self._nrm[i]
        return P - (((P - q) * n).sum(1))[:, None] * n

    def mesh(self, h=None):
        return self.V, self.F


def load_mesh_file(path):
    """Read binary/ASCII STL or OBJ -> (V, F) with merged vertices."""
    with open(path, "rb") as f:
        raw = f.read()
    low = path.lower()
    if low.endswith(".obj"):
        V, F = [], []
        for line in raw.decode("utf-8", "replace").splitlines():
            p = line.split()
            if not p:
                continue
            if p[0] == "v":
                V.append([float(p[1]), float(p[2]), float(p[3])])
            elif p[0] == "f":
                idx = [int(q.split("/")[0]) - 1 for q in p[1:]]
                for k in range(1, len(idx) - 1):
                    F.append([idx[0], idx[k], idx[k + 1]])
        return np.array(V, float), np.array(F, np.int64)
    if low.endswith(".stl"):
        if raw[:5] == b"solid" and b"facet" in raw[:400]:
            pts = []
            for line in raw.decode("utf-8", "replace").splitlines():
                p = line.split()
                if p and p[0] == "vertex":
                    pts.append([float(p[1]), float(p[2]), float(p[3])])
            T = np.array(pts, float).reshape(-1, 3, 3)
        else:
            n = int(np.frombuffer(raw[80:84], np.uint32)[0])
            rec = np.frombuffer(raw[84:84 + 50 * n], dtype=np.dtype(
                [("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
            T = rec["v"].astype(float)
        flat = T.reshape(-1, 3)
        U, inv = np.unique(np.round(flat, 6), axis=0, return_inverse=True)
        return U, inv.reshape(-1, 3).astype(np.int64)
    raise ValueError("unsupported mesh format (use .stl or .obj): %s" % path)


_MESH_CACHE = {}


def cached_mesh(bone, h=0.9):
    k = (bone.key(), h)
    if k not in _MESH_CACHE:
        if len(_MESH_CACHE) > 24:
            _MESH_CACHE.pop(next(iter(_MESH_CACHE)))
        _MESH_CACHE[k] = bone.mesh(h)
    return _MESH_CACHE[k]
