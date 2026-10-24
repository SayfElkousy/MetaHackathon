"""Population-reference and bone-conditioned statistical baselines for ACL geometry.

Both baselines work in a *reference knee frame*: each knee (mirrored to right-knee
orientation if needed) is rigidly aligned, femur and tibia separately, to a
reference knee. An ACL is described by:
    femoral footprint (reference femoral frame), tibial footprint (reference tibial
    frame), area-equivalent radius profile, and centreline offsets from the straight
    footprint-to-footprint line.

population_reference : mean descriptor of training knees (no patient conditioning)
ssm_bone_conditioned : PCA of bone-surface correspondences -> ridge regression to the
                       descriptor (optionally + demographic covariates)

Correspondence here is closest-point after rigid alignment, which is a crude but
transparent baseline (cf. Van Oevelen et al. 2023 and Pedoia et al. 2015 for
proper SSM pipelines).
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy.spatial import cKDTree

from ..annotation.schema import Label
from ..imaging.coords import apply_affine, mirror_matrix
from ..imaging.volume import Volume
from ..registration import bilateral as B
from .geometry import centerline_profile, render_tube

N_STATIONS = 12


@dataclasses.dataclass
class KneeEntry:
    subject_id: str
    side: str
    labels: Volume
    demographics: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class RefFrame:
    femur_pts: np.ndarray
    tibia_pts: np.ndarray
    joint_center: np.ndarray
    crop: float


def _oriented_affine(entry: KneeEntry) -> np.ndarray:
    return mirror_matrix() @ entry.labels.affine if entry.side == "left" else entry.labels.affine


_REF_CACHE: dict = {}


def build_reference(entry: KneeEntry, crop=38.0, n_pts=400, seed=0) -> RefFrame:
    key = (id(entry.labels), crop, n_pts, seed)
    if key not in _REF_CACHE:
        _REF_CACHE[key] = _build_reference(entry, crop, n_pts, seed)
    return _REF_CACHE[key]


def _build_reference(entry: KneeEntry, crop=38.0, n_pts=400, seed=0) -> RefFrame:
    rng = np.random.default_rng(seed)
    A = _oriented_affine(entry)
    jc = B.joint_center(entry.labels.data, A)
    fem = B.crop_sphere(B.surface_points(entry.labels.data == Label.FEMUR, A), jc, crop)
    tib = B.crop_sphere(B.surface_points(entry.labels.data == Label.TIBIA, A), jc, crop)
    fem = fem[rng.choice(len(fem), min(n_pts, len(fem)), replace=False)]
    tib = tib[rng.choice(len(tib), min(n_pts, len(tib)), replace=False)]
    return RefFrame(fem, tib, jc, crop)


_ALIGN_CACHE: dict = {}


def align_to_reference(entry: KneeEntry, ref: RefFrame, seed=0):
    """Return (T_fem, T_tib) mapping the knee's *oriented world* to the reference frame, plus oriented affine.
    Results are memoised per (knee, reference) pair within a process."""
    key = (id(entry.labels), id(ref), seed)
    if key not in _ALIGN_CACHE:
        _ALIGN_CACHE[key] = _align_to_reference(entry, ref, seed)
    return _ALIGN_CACHE[key]


def _align_to_reference(entry: KneeEntry, ref: RefFrame, seed=0):
    A = _oriented_affine(entry)
    jc = B.joint_center(entry.labels.data, A)
    fem = B.crop_sphere(B.surface_points(entry.labels.data == Label.FEMUR, A), jc, ref.crop)
    tib = B.crop_sphere(B.surface_points(entry.labels.data == Label.TIBIA, A), jc, ref.crop)
    T0 = np.eye(4)
    T0[:3, 3] = ref.joint_center - jc
    rng = np.random.default_rng(seed)
    fr = B.multistart_icp(fem, ref.femur_pts, T0, rng=rng, max_pts=2500)
    tr = B.multistart_icp(tib, ref.tibia_pts, fr.T, angles_deg=(0, 12), rng=rng, max_pts=2500)
    return fr.T, tr.T, A, fem, tib


def bone_features(fem_aligned, tib_aligned, ref: RefFrame) -> np.ndarray:
    """Closest-point correspondences of reference surface samples (flattened, mm)."""
    f = fem_aligned[cKDTree(fem_aligned).query(ref.femur_pts)[1]]
    t = tib_aligned[cKDTree(tib_aligned).query(ref.tibia_pts)[1]]
    return np.concatenate([f.ravel(), t.ravel()])


def acl_descriptor(entry: KneeEntry, T_fem, T_tib, A) -> np.ndarray:
    lab = entry.labels.data
    fp = B.acl_footprints_from_mask(lab, entry.labels.affine)
    M = mirror_matrix() if entry.side == "left" else np.eye(4)
    ff = apply_affine(M, fp["femoral"][None])[0]
    ft = apply_affine(M, fp["tibial"][None])[0]
    prof = centerline_profile(lab == Label.ACL_INTACT, A, ff, ft, N_STATIONS)
    r = np.sqrt(np.maximum(prof["area_mm2"], 0) / np.pi)
    cl = prof["centerline"][1:-1]
    t = prof["t_stations"][:, None]
    off = cl - ((1 - t) * ff + t * ft)  # offsets in oriented world
    # express in reference frame orientation (rotation part of the femoral transform)
    off_ref = off @ T_fem[:3, :3].T
    return np.concatenate([apply_affine(T_fem, ff[None])[0], apply_affine(T_tib, ft[None])[0], r, off_ref.ravel()])


def descriptor_to_mask(desc: np.ndarray, target: KneeEntry, T_fem, T_tib) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map a reference-frame descriptor into the target image grid and voxelise it."""
    A = _oriented_affine(target)
    M = mirror_matrix() if target.side == "left" else np.eye(4)
    ff_or = apply_affine(np.linalg.inv(T_fem), desc[0:3][None])[0]
    ft_or = apply_affine(np.linalg.inv(T_tib), desc[3:6][None])[0]
    r = desc[6:6 + N_STATIONS]
    off_ref = desc[6 + N_STATIONS:].reshape(N_STATIONS, 3)
    off_or = off_ref @ np.linalg.inv(T_fem[:3, :3]).T
    t = ((np.arange(N_STATIONS) + 0.5) / N_STATIONS)[:, None]
    cl_or = (1 - t) * ff_or + t * ft_or + off_or
    cl_or = np.vstack([ff_or, cl_or, ft_or])
    # oriented world -> image world (undo mirror if left)
    Minv = np.linalg.inv(M)
    cl = apply_affine(Minv, cl_or)
    radii = np.concatenate([[r[0]], r, [r[-1]]])
    mask = render_tube(target.labels.data.shape, target.labels.affine, cl, np.maximum(radii, 0.5))
    mask &= ~np.isin(target.labels.data, [Label.FEMUR, Label.TIBIA])
    return mask, cl[0], cl[-1]


class StatisticalACLModel:
    def __init__(self, n_modes: int = 5, use_demographics: tuple = ()):
        self.n_modes = n_modes
        self.use_demographics = tuple(use_demographics)

    def fit(self, entries: list[KneeEntry], ref_entry: KneeEntry | None = None, seed=0):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import StandardScaler

        self.ref = build_reference(ref_entry or entries[0], seed=seed)
        X, Y, D = [], [], []
        for e in entries:
            Tf, Tt, A, fem, tib = align_to_reference(e, self.ref, seed)
            X.append(bone_features(apply_affine(Tf, fem), apply_affine(Tt, tib), self.ref))
            Y.append(acl_descriptor(e, Tf, Tt, A))
            D.append([e.demographics.get(k, np.nan) for k in self.use_demographics])
        X, Y = np.array(X), np.array(Y)
        self.mean_desc = Y.mean(0)
        k = min(self.n_modes, len(entries) - 1)
        self.pca = PCA(n_components=k).fit(X)
        Z = self.pca.transform(X)
        if self.use_demographics:
            D = np.array(D, float)
            if np.isnan(D).any():
                raise ValueError("demographic covariate missing for a training knee; impute explicitly or drop the covariate")
            Z = np.hstack([Z, D])
        self.scaler = StandardScaler().fit(Z)
        self.reg = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(self.scaler.transform(Z), Y)
        self.n_train = len(entries)
        self.explained_variance = self.pca.explained_variance_ratio_.tolist()
        return self

    def _align_target(self, target: KneeEntry, seed=0):
        Tf, Tt, A, fem, tib = align_to_reference(target, self.ref, seed)
        return Tf, Tt, apply_affine(Tf, fem), apply_affine(Tt, tib)

    def predict_population(self, target: KneeEntry, seed=0):
        Tf, Tt, *_ = self._align_target(target, seed)
        return descriptor_to_mask(self.mean_desc, target, Tf, Tt)

    def predict_conditioned(self, target: KneeEntry, seed=0):
        Tf, Tt, fa, ta = self._align_target(target, seed)
        z = self.pca.transform(bone_features(fa, ta, self.ref)[None])
        if self.use_demographics:
            d = [target.demographics.get(k, np.nan) for k in self.use_demographics]
            if np.isnan(d).any():
                raise ValueError("target demographic covariate missing")
            z = np.hstack([z, np.array(d, float)[None]])
        desc = self.reg.predict(self.scaler.transform(z))[0]
        return descriptor_to_mask(desc, target, Tf, Tt)
