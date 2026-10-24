"""Bilateral registration baseline (interpretable, rigid-per-bone).

Pipeline
1. Confirm laterality of both knees (abstain if unknown or identical).
2. Mirror the contralateral knee by the world reflection x -> -x (coords.mirror_matrix).
   Voxel data are untouched; only the affine is reflected, so no interpolation occurs.
3. Extract femur and tibia surfaces (marching cubes, world mm), cropped to a sphere
   around the joint so that differing shaft coverage does not bias alignment.
4. Rigid trimmed ICP, femur first, then tibia initialised from the femur result.
   Separate transforms absorb differences in knee flexion/rotation between scans.
5. Contralateral ACL geometry is carried over with a *blend of the two rigid
   transforms* parameterised by position along the femoral->tibial axis
   (smoothstep weight). This is the only deformation allowed: it cannot create
   shape that is not in the contralateral ACL.

No unconstrained deformable registration is used for ACL transfer.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from ..annotation.schema import Label
from ..imaging.coords import apply_affine, mirror_matrix, rigid_from_params
from ..imaging.volume import Volume

# ----------------------------------------------------------------- surfaces


def surface_points(mask: np.ndarray, affine: np.ndarray, step: int = 1) -> np.ndarray:
    from skimage import measure

    if mask.sum() < 10:
        return np.zeros((0, 3))
    m = np.pad(mask.astype(np.uint8), 1)
    verts, *_ = measure.marching_cubes(m, 0.5, step_size=step)
    verts -= 1
    return apply_affine(affine, verts)


def surface_mesh(mask: np.ndarray, affine: np.ndarray, step: int = 1):
    """World-space triangle mesh of a binary mask (trimesh.Trimesh)."""
    import trimesh
    from skimage import measure

    m = np.pad(mask.astype(np.uint8), 1)
    verts, faces, *_ = measure.marching_cubes(m, 0.5, step_size=step)
    verts = apply_affine(affine, verts - 1)
    if np.linalg.det(affine[:3, :3]) < 0:
        faces = faces[:, ::-1]  # keep outward normals after a reflection
    return trimesh.Trimesh(verts, faces, process=True)


def joint_center(labels: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Midpoint of the closest femur-tibia surface point pair (world mm)."""
    f = surface_points(labels == Label.FEMUR, affine, step=2)
    t = surface_points(labels == Label.TIBIA, affine, step=2)
    if len(f) == 0 or len(t) == 0:
        raise ValueError("femur and tibia labels are required to locate the joint")
    d, j = cKDTree(t).query(f)
    k = np.argsort(d)[: max(20, len(d) // 50)]
    return 0.5 * (f[k].mean(0) + t[j[k]].mean(0))


def crop_sphere(pts: np.ndarray, c: np.ndarray, r: float) -> np.ndarray:
    return pts[np.linalg.norm(pts - c, axis=1) <= r]


# ----------------------------------------------------------------- ICP


def kabsch(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Rigid transform (4x4, det=+1) minimising ||R A + t - B||."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cb - R @ ca
    return T


@dataclasses.dataclass
class ICPResult:
    T: np.ndarray
    rms: float
    inlier_fraction: float
    iterations: int
    converged: bool


def icp(src: np.ndarray, dst: np.ndarray, T0: np.ndarray | None = None, max_iter: int = 80, trim: float = 0.9,
        tol: float = 1e-4, rng: np.random.Generator | None = None, max_pts: int = 6000) -> ICPResult:
    """Trimmed point-to-point ICP (src -> dst)."""
    rng = rng or np.random.default_rng(0)
    if len(src) > max_pts:
        src = src[rng.choice(len(src), max_pts, replace=False)]
    tree = cKDTree(dst)
    T = np.eye(4) if T0 is None else T0.copy()
    prev = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        S = apply_affine(T, src)
        d, j = tree.query(S)
        k = np.argsort(d)[: int(trim * len(d))]
        dT = kabsch(S[k], dst[j[k]])
        T = dT @ T
        rms = float(np.sqrt(np.mean(d[k] ** 2)))
        if abs(prev - rms) < tol:
            break
        prev = rms
    S = apply_affine(T, src)
    d, _ = tree.query(S)
    k = np.argsort(d)[: int(trim * len(d))]
    return ICPResult(T, float(np.sqrt(np.mean(d[k] ** 2))), float(np.mean(d < 2.0)), it, it < max_iter)


def multistart_icp(src, dst, T_init, angles_deg=(0, 12), rng=None, **kw) -> ICPResult:
    """Run ICP from several rotational starts around the source centroid; keep the lowest RMS."""
    c = apply_affine(T_init, src).mean(0)
    best = None
    starts = [(0, 0, 0)]
    for a in angles_deg[1:]:
        for ax in range(3):
            for sgn in (-1, 1):
                v = [0, 0, 0]
                v[ax] = sgn * np.radians(a)
                starts.append(tuple(v))
    for s in starts:
        R = rigid_from_params(*s, center=c)
        r = icp(src, dst, R @ T_init, rng=rng, **kw)
        if best is None or r.rms < best.rms:
            best = r
    return best


# ----------------------------------------------------------------- bilateral registration


@dataclasses.dataclass
class BilateralRegistration:
    mirror: np.ndarray  # contralateral world -> mirrored world
    T_femur: np.ndarray  # mirrored contralateral world -> injured world (femur)
    T_tibia: np.ndarray
    femur_icp: ICPResult
    tibia_icp: ICPResult
    joint_center_injured: np.ndarray
    crop_radius_mm: float
    axis_femoral: np.ndarray  # ACL axis endpoints in *injured* world (from transferred contralateral footprints)
    axis_tibial: np.ndarray

    @property
    def contra_to_injured_femur(self):
        return self.T_femur @ self.mirror

    @property
    def contra_to_injured_tibia(self):
        return self.T_tibia @ self.mirror

    def to_dict(self):
        return {
            "mirror": self.mirror.tolist(), "T_femur": self.T_femur.tolist(), "T_tibia": self.T_tibia.tolist(),
            "femur_icp": {"rms_mm": self.femur_icp.rms, "inlier_fraction_2mm": self.femur_icp.inlier_fraction, "iterations": self.femur_icp.iterations},
            "tibia_icp": {"rms_mm": self.tibia_icp.rms, "inlier_fraction_2mm": self.tibia_icp.inlier_fraction, "iterations": self.tibia_icp.iterations},
            "joint_center_injured": self.joint_center_injured.tolist(), "crop_radius_mm": self.crop_radius_mm,
            "acl_axis_injured_world": [self.axis_femoral.tolist(), self.axis_tibial.tolist()],
            "relative_tibiofemoral_rotation_deg": relative_rotation_deg(self.T_femur, self.T_tibia),
            "method": "mirror(x->-x) + per-bone trimmed rigid ICP + axis-blended rigid transfer",
        }


def relative_rotation_deg(A, B) -> float:
    R = A[:3, :3].T @ B[:3, :3]
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def acl_footprints_from_mask(labels: np.ndarray, affine: np.ndarray, acl_codes=(Label.ACL_INTACT,), radius_vox: int = 2):
    """Attachment-centre *estimates* from a mask: centroid of ACL voxels touching each bone.
    Expert landmarks override these whenever available."""
    acl = np.isin(labels, [int(c) for c in acl_codes])
    out = {}
    for name, bone in (("femoral", Label.FEMUR), ("tibial", Label.TIBIA)):
        near = ndimage.binary_dilation(labels == bone, iterations=radius_vox) & acl
        out[name] = apply_affine(affine, np.argwhere(near).mean(0)[None])[0] if near.any() else None
    return out


def register_bilateral(inj_labels: Volume, con_labels: Volume, crop_radius_mm: float = 38.0, seed: int = 0,
                       con_footprints: dict | None = None, icp_kw: dict | None = None) -> BilateralRegistration:
    rng = np.random.default_rng(seed)
    icp_kw = icp_kw or {}
    M = mirror_matrix()
    A_con_m = M @ con_labels.affine  # mirrored contralateral voxel -> world
    jc_inj = joint_center(inj_labels.data, inj_labels.affine)
    jc_con = joint_center(con_labels.data, A_con_m)
    surf = {}
    for bone in (Label.FEMUR, Label.TIBIA):
        surf[("inj", bone)] = crop_sphere(surface_points(inj_labels.data == bone, inj_labels.affine), jc_inj, crop_radius_mm)
        surf[("con", bone)] = crop_sphere(surface_points(con_labels.data == bone, A_con_m), jc_con, crop_radius_mm)
    T0 = np.eye(4)
    T0[:3, 3] = jc_inj - jc_con
    fem = multistart_icp(surf[("con", Label.FEMUR)], surf[("inj", Label.FEMUR)], T0, rng=rng, **icp_kw)
    tib = multistart_icp(surf[("con", Label.TIBIA)], surf[("inj", Label.TIBIA)], fem.T, angles_deg=(0, 10, 20), rng=rng, **icp_kw)
    fp = con_footprints or acl_footprints_from_mask(con_labels.data, con_labels.affine)
    if fp.get("femoral") is None or fp.get("tibial") is None:
        raise ValueError("contralateral ACL footprints unavailable (no ACL mask touching bone and no landmarks)")
    a_f = apply_affine(fem.T @ M, np.asarray(fp["femoral"])[None])[0]
    a_t = apply_affine(tib.T @ M, np.asarray(fp["tibial"])[None])[0]
    return BilateralRegistration(M, fem.T, tib.T, fem, tib, jc_inj, crop_radius_mm, a_f, a_t)


# ----------------------------------------------------------------- axis-blended transfer


def _smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def blend_weights(pts_injured: np.ndarray, a_f: np.ndarray, a_t: np.ndarray) -> np.ndarray:
    ax = a_t - a_f
    t = ((pts_injured - a_f) @ ax) / (ax @ ax)
    return _smoothstep(t)


def inverse_blended_map(pts_injured: np.ndarray, reg: BilateralRegistration, n_iter: int = 6) -> np.ndarray:
    """For points in the injured frame, find the corresponding point in *contralateral world*.

    Forward model: x = (1-w) Tf M p + w Tt M p, w = w(x).  Solved by fixed-point iteration
    (contracting because Tf and Tt differ by a small rotation)."""
    Ff, Ft = reg.contra_to_injured_femur, reg.contra_to_injured_tibia
    If, It = np.linalg.inv(Ff), np.linalg.inv(Ft)
    w = blend_weights(pts_injured, reg.axis_femoral, reg.axis_tibial)[:, None]
    p = (1 - w) * apply_affine(If, pts_injured) + w * apply_affine(It, pts_injured)
    for _ in range(n_iter):
        x = (1 - w) * apply_affine(Ff, p) + w * apply_affine(Ft, p)
        w = blend_weights(x, reg.axis_femoral, reg.axis_tibial)[:, None]
        # Newton-like correction using the average linear part
        Rinv = (1 - w)[..., None] * If[None, :3, :3] + w[..., None] * It[None, :3, :3]
        p = p + np.einsum("nij,nj->ni", Rinv, pts_injured - x)
    return p


def transfer_to_injured(src: Volume, reg: BilateralRegistration, target: Volume, kind: str = "label", order: int = 1, cval=0) -> np.ndarray:
    """Resample a contralateral volume (image or labels) into the injured grid via the blended map."""
    shape = target.data.shape[:3]
    ijk = np.indices(shape).reshape(3, -1).T.astype(float)
    X = apply_affine(target.affine, ijk)
    P = inverse_blended_map(X, reg)
    src_vox = apply_affine(np.linalg.inv(src.affine), P)
    o = 0 if kind == "label" else order
    out = ndimage.map_coordinates(src.data, src_vox.T, order=o, mode="constant", cval=cval, prefilter=o > 1)
    return out.reshape(shape).astype(src.data.dtype if kind == "label" else np.float32)


def transfer_points(pts_contra_world: np.ndarray, reg: BilateralRegistration) -> np.ndarray:
    """Forward map of contralateral world points into the injured frame (fixed-point on w)."""
    Ff, Ft = reg.contra_to_injured_femur, reg.contra_to_injured_tibia
    xf, xt = apply_affine(Ff, pts_contra_world), apply_affine(Ft, pts_contra_world)
    w = blend_weights(0.5 * (xf + xt), reg.axis_femoral, reg.axis_tibial)[:, None]
    for _ in range(4):
        x = (1 - w) * xf + w * xt
        w = blend_weights(x, reg.axis_femoral, reg.axis_tibial)[:, None]
    return (1 - w) * xf + w * xt


# ----------------------------------------------------------------- registration QC


def surface_distance_stats(a: np.ndarray, b: np.ndarray) -> dict:
    if len(a) == 0 or len(b) == 0:
        return {"rms_mm": None, "mean_mm": None, "hd95_mm": None}
    da, _ = cKDTree(b).query(a)
    db, _ = cKDTree(a).query(b)
    d = np.concatenate([da, db])
    return {"rms_mm": float(np.sqrt(np.mean(d ** 2))), "mean_mm": float(d.mean()), "hd95_mm": float(np.percentile(d, 95))}


FEMORAL_LANDMARKS = ("femur_lateral_epicondyle", "femur_medial_epicondyle", "femur_notch_roof_apex", "acl_femoral_footprint_center")
TIBIAL_LANDMARKS = ("tibia_medial_eminence_tip", "tibia_lateral_eminence_tip", "tibia_tuberosity", "tibia_lateral_plateau_rim",
                    "pcl_tibial_attachment_center", "acl_tibial_footprint_center")


def registration_qc(reg: BilateralRegistration, inj_labels: Volume, con_labels: Volume, inj_landmarks: dict | None = None,
                    con_landmarks: dict | None = None, exclude=("acl_femoral_footprint_center", "acl_tibial_footprint_center")) -> dict:
    """Surface distances after alignment and *independent* landmark errors.

    Landmarks are never used to drive the registration. ACL footprints are excluded by
    default because they are the estimation target, not a registration check; bilateral
    landmark differences include true anatomical asymmetry, so they bound but do not
    isolate registration error."""
    out = {"surface": {}, "landmarks": {}}
    for bone, T in ((Label.FEMUR, reg.contra_to_injured_femur), (Label.TIBIA, reg.contra_to_injured_tibia)):
        a = crop_sphere(surface_points(inj_labels.data == bone, inj_labels.affine), reg.joint_center_injured, reg.crop_radius_mm)
        b = crop_sphere(apply_affine(T, surface_points(con_labels.data == bone, con_labels.affine)), reg.joint_center_injured, reg.crop_radius_mm)
        out["surface"][Label(bone).name.lower()] = surface_distance_stats(a, b)
    if inj_landmarks and con_landmarks:
        errs = {}
        for name in set(inj_landmarks) & set(con_landmarks):
            if name in exclude:
                continue
            T = reg.contra_to_injured_femur if name in FEMORAL_LANDMARKS else reg.contra_to_injured_tibia if name in TIBIAL_LANDMARKS else None
            if T is None:
                continue
            p = apply_affine(T, np.asarray(con_landmarks[name])[None])[0]
            errs[name] = float(np.linalg.norm(p - np.asarray(inj_landmarks[name])))
        out["landmarks"] = {"per_landmark_mm": errs, "mean_mm": float(np.mean(list(errs.values()))) if errs else None,
                            "n": len(errs), "note": "independent landmarks; includes true bilateral asymmetry"}
    return out
