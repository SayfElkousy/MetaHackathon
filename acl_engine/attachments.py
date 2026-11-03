"""Attachment regions: bounded footprint domains on bone surfaces.

New module. Each footprint is an independently controlled parametric domain
(center, axes, rotation, shape exponent, subregion split) whose strand
endpoints are distributed inside the domain -- not collapsed to one point --
and projected onto the corresponding bone surface:

- tibial: plateau plane z = 0 (schematic reference surface);
- femoral: lateral condyle ellipsoid (schematic reference surface), solved
  analytically per ray along X.

Bone surfaces here are the project's schematic primitives, passed in as
explicit arguments so real meshes can replace them later through the same
`project_to_bone` interface. Outputs are labeled reference surfaces, never
patient-specific anatomy. Units: mm, degrees. Vectorized.
"""
import math
import numpy as np


def tibial_plateau_project(points):
    """Project points onto the tibial plateau reference plane (z = 0)."""
    P = np.asarray(points, dtype=float).copy()
    P[:, 2] = 0.0
    return P


def ellipsoid_ray_x(yz, center, radii, side="right"):
    """X of the notch-facing ellipsoid surface along X-rays at (y,z).

    Solves ((x-cx)/rx)^2 = 1 - ((y-cy)/ry)^2 - ((z-cz)/rz)^2 and takes the
    notch-facing root (smaller x for right knees). Returns (x, hit_mask);
    rays that miss the silhouette report hit False (caller must handle).
    Vectorized.
    """
    yz = np.asarray(yz, dtype=float)
    cx, cy, cz = (float(v) for v in center)
    rx, ry, rz = (float(v) for v in radii)
    q = 1.0 - ((yz[:, 0] - cy) / ry) ** 2 - ((yz[:, 1] - cz) / rz) ** 2
    hit = q >= 0.0
    x = np.full(yz.shape[0], np.nan)
    root = np.sqrt(np.clip(q, 0.0, None))
    if side == "right":
        x = cx - rx * root
    else:
        x = cx + rx * root
    return x, hit


def spiral_disk(n, offset=0.0):
    """Deterministic low-discrepancy unit-disk samples (spiral). Returns
    (n,2) polar (r, phi). No RNG: regeneration is deterministic."""
    k = np.arange(n, dtype=float) + 0.5
    r = np.sqrt(k / n)
    phi = 2.0 * math.pi * k * 0.618033988749895 + float(offset)
    return np.stack([r, phi % (2.0 * math.pi)], axis=1)


def split_weights(phi, split_deg, sharp=3.0):
    """AM weight of angles phi across a subregion split line at split_deg.
    Vectorized; partition of unity with PL."""
    d = np.cos(np.asarray(phi, dtype=float) - math.radians(float(split_deg)))
    wam = 0.5 + 0.5 * np.tanh(float(sharp) * d)
    return wam, 1.0 - wam


class Footprint:
    """Bounded 2-D attachment domain mapped onto a bone surface."""

    def __init__(self, name, center2, axes, rotation_deg=0.0, exponent=1.0,
                 split_deg=0.0, transition_mm=6.0):
        self.name = name
        self.center2 = np.asarray(center2, dtype=float)
        self.axes = (float(axes[0]), float(axes[1]))
        self.rotation_deg = float(rotation_deg)
        self.exponent = float(exponent)
        self.split_deg = float(split_deg)
        self.transition_mm = float(transition_mm)

    def domain_points(self, polar):
        """Map unit-disk polar samples to 2-D domain coords. Vectorized."""
        from acl_engine.loft import superellipse_map
        r = polar[:, 0]
        phi = polar[:, 1]
        x, y = superellipse_map(r, phi, self.axes[0], self.axes[1], self.exponent)
        a = math.radians(self.rotation_deg)
        xr = x * math.cos(a) - y * math.sin(a)
        yr = x * math.sin(a) + y * math.cos(a)
        return np.stack([xr, yr], axis=1) + self.center2[None, :]

    def area(self):
        """Closed-form superellipse domain area (mm^2)."""
        from scipy.special import gamma
        p = 2.0 / self.exponent
        return float(4.0 * self.axes[0] * self.axes[1]
                     * gamma(1.0 + 1.0 / p) ** 2 / gamma(1.0 + 2.0 / p))

    def boundary_loop(self, n=128):
        """Closed boundary loop in domain coords (for viewer + labels)."""
        ph = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
        return self.domain_points(np.stack([np.ones(n), ph], axis=1))

    def subregion_areas(self, grid=160):
        """AM/PL subregion areas by masked grid integration. Vectorized."""
        e = max(self.axes) * 1.2
        u = np.linspace(-e, e, grid)
        X, Y = np.meshgrid(u, u)
        a = math.radians(-self.rotation_deg)
        xr = (X - 0.0) * math.cos(a) - (Y - 0.0) * math.sin(a)
        yr = (X - 0.0) * math.sin(a) + (Y - 0.0) * math.cos(a)
        q = (np.abs(xr) / self.axes[0]) ** (2.0 / self.exponent) \
            + (np.abs(yr) / self.axes[1]) ** (2.0 / self.exponent)
        inside = q <= 1.0
        cell = (2.0 * e / grid) ** 2
        phi = np.arctan2(yr, xr)
        wam, _wpl = split_weights(phi, self.split_deg)
        return {"AM": float((inside * wam).sum()) * cell,
                "PL": float((inside * (1.0 - wam)).sum()) * cell}

    def endpoints(self, n_am, n_pl, seed_offset=0.0):
        """Stratified endpoint sets per bundle inside the domain.

        Samples a dense spiral, classifies by the split mask, and takes an
        evenly spaced subset per bundle -- deterministic, distributed, and
        bundle-pure at the attachment. Returns dict AM/PL of (n,2).
        """
        dense = spiral_disk(max(64, 12 * (n_am + n_pl)), offset=seed_offset)
        wam, _wpl = split_weights(dense[:, 1], self.split_deg)
        out = {}
        for key, w, n in (("AM", wam, n_am), ("PL", 1.0 - wam, n_pl)):
            idx = np.nonzero(w > 0.5)[0]
            if idx.size < n:
                idx = np.argsort(-w)[:n]
            else:
                idx = idx[np.linspace(0, idx.size - 1, n).astype(int)]
            out[key] = self.domain_points(dense[idx])
        return out


def tibial_attachment_points(fp, endpoints2):
    """Lift tibial domain endpoints onto the plateau reference surface."""
    out = {}
    for key, uv in endpoints2.items():
        P = np.stack([uv[:, 0], uv[:, 1], np.zeros(uv.shape[0])], axis=1)
        out[key] = tibial_plateau_project(P)
    return out


def femoral_attachment_points(fp, endpoints2, condyle_center, condyle_radii,
                              side="right", proud_mm=0.0):
    """Lift femoral domain (y,z) endpoints onto the condyle surface.

    Returns (points_dict, misses): points on the ellipsoid plus proud_mm
    along +X (right) / -X (left); misses counts rays outside the silhouette
    (reported, never silently dropped -- missed points fall back to the wall
    plane and are flagged in the validation report).
    """
    from acl_engine.footprints import femoral_wall_x
    out = {}
    misses = {}
    sgn = -1.0 if side == "right" else 1.0
    for key, uv in endpoints2.items():
        x, hit = ellipsoid_ray_x(uv, condyle_center, condyle_radii, side)
        xw = femoral_wall_x(np.array([condyle_center[0] + (11.5 if side == "right" else -11.5),
                                      uv[:, 0].mean(), uv[:, 1].mean()]), side)
        xf = np.where(hit, x, xw)
        P = np.stack([xf + sgn * float(proud_mm), uv[:, 0], uv[:, 1]], axis=1)
        out[key] = P
        misses[key] = int((~hit).sum())
    return out, misses


def femoral_patch_mesh(fp, condyle_center, condyle_radii, side="right",
                       nu=25, nv=18, proud_mm=0.05):
    """Open patch mesh of the femoral footprint lying on the condyle.

    Grid over the domain bounding box masked to the superellipse; x solved
    on the ellipsoid. Intentionally open (a surface region, not a solid).
    """
    yy = np.linspace(fp.center2[0] - fp.axes[0] * 1.05,
                     fp.center2[0] + fp.axes[0] * 1.05, nu)
    zz = np.linspace(fp.center2[1] - fp.axes[1] * 1.05,
                     fp.center2[1] + fp.axes[1] * 1.05, nv)
    Y, Z = np.meshgrid(yy, zz)
    a = math.radians(-fp.rotation_deg)
    xr = (Y - fp.center2[0]) * math.cos(a) - (Z - fp.center2[1]) * math.sin(a)
    yr = (Y - fp.center2[0]) * math.sin(a) + (Z - fp.center2[1]) * math.cos(a)
    q = (np.abs(xr) / fp.axes[0]) ** (2.0 / fp.exponent) \
        + (np.abs(yr) / fp.axes[1]) ** (2.0 / fp.exponent)
    inside = q <= 1.0
    x, hit = ellipsoid_ray_x(np.stack([Y.ravel(), Z.ravel()], axis=1),
                             condyle_center, condyle_radii, side)
    sgn = -1.0 if side == "right" else 1.0
    X = (x + sgn * float(proud_mm)).reshape(Y.shape)
    keep = inside & hit.reshape(Y.shape)
    V = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    idx = -np.ones(Y.size, dtype=np.int64)
    idx[np.nonzero(keep.ravel())[0]] = np.arange(int(keep.sum()))
    g = idx.reshape(nv, nu)
    m = (g[:-1, :-1] >= 0) & (g[:-1, 1:] >= 0) & (g[1:, :-1] >= 0) & (g[1:, 1:] >= 0)
    a, b, c, d = g[:-1, :-1][m], g[:-1, 1:][m], g[1:, :-1][m], g[1:, 1:][m]
    F = np.concatenate([np.stack([a, b, c], axis=1),
                        np.stack([b, d, c], axis=1)]).astype(np.int64)
    return V[idx >= 0], F, {"masked_fraction": float(1.0 - keep.mean())}
