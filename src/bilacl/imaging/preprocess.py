"""Preprocessing: intensity normalisation, resampling, sequence co-registration, QC views.

Rules enforced here:
* Image interpolation (linear / cubic) is separate from label interpolation
  (nearest neighbour). The UNKNOWN label value survives label resampling.
* Resampling records the source spacing; it never claims to add information.
* Independently acquired sagittal / coronal / axial series are never stacked
  into a pseudo-isotropic volume. `coregister_series` aligns one series to a
  reference grid for joint viewing only; each series remains a separate input.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from .coords import voxel_spacing
from .volume import Volume


def intensity_normalize(vol: Volume, mask: np.ndarray | None = None, p_low=0.5, p_high=99.5) -> Volume:
    """Percentile clip then z-score inside `mask` (or nonzero voxels).
    MRI intensities are not quantitative across scanners; this is a learning
    convenience, not a physical calibration."""
    x = vol.data.astype(np.float32)
    m = mask if mask is not None else x > np.percentile(x, 1)
    if not m.any():
        m = np.ones_like(x, bool)
    lo, hi = np.percentile(x[m], [p_low, p_high])
    x = np.clip(x, lo, hi)
    mu, sd = float(x[m].mean()), float(x[m].std() + 1e-6)
    out = (x - mu) / sd
    return vol.with_data(out, intensity_norm={"method": "percentile-clip+zscore", "p_low": p_low, "p_high": p_high, "lo": float(lo), "hi": float(hi), "mean": mu, "std": sd})


def n4_bias_correct(vol: Volume, shrink: int = 2) -> Volume:
    import SimpleITK as sitk

    from .volume import to_sitk

    img = sitk.Cast(to_sitk(vol), sitk.sitkFloat32)
    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    small = sitk.Shrink(img, [shrink] * 3)
    smask = sitk.Shrink(mask, [shrink] * 3)
    corr = sitk.N4BiasFieldCorrectionImageFilter()
    corr.Execute(small, smask)
    logbias = corr.GetLogBiasFieldAsImage(img)
    out = img / sitk.Exp(logbias)
    arr = np.transpose(sitk.GetArrayFromImage(out), (2, 1, 0))
    return vol.with_data(arr.astype(np.float32), bias_correction="N4 (SimpleITK)")


def _target_grid(vol: Volume, spacing) -> tuple[np.ndarray, tuple[int, int, int]]:
    sp_old = voxel_spacing(vol.affine)
    sp_new = np.asarray(spacing, float)
    direc = vol.affine[:3, :3] / sp_old
    extent = np.array(vol.data.shape[:3]) * sp_old
    shape = tuple(int(max(1, round(e / s))) for e, s in zip(extent, sp_new))
    aff = np.eye(4)
    aff[:3, :3] = direc * sp_new
    # keep the physical position of the volume corner (voxel edge) fixed
    corner = vol.affine[:3, 3] - direc @ (sp_old / 2)
    aff[:3, 3] = corner + direc @ (sp_new / 2)
    return aff, shape


def resample_to_grid(vol: Volume, affine: np.ndarray, shape, kind: str = "image", order: int = 1, cval: float = 0.0) -> Volume:
    """Resample `vol` onto a target grid defined in world coordinates.

    kind="image": spline interpolation of given order (1 = trilinear).
    kind="label": nearest neighbour only, so label codes (incl. UNKNOWN) are preserved.
    """
    if kind not in ("image", "label"):
        raise ValueError(kind)
    T = np.linalg.inv(vol.affine) @ affine  # target voxel -> source voxel
    o = 0 if kind == "label" else order
    out = ndimage.affine_transform(
        vol.data, T[:3, :3], offset=T[:3, 3], output_shape=tuple(shape), order=o, mode="constant", cval=cval, prefilter=o > 1
    )
    if kind == "label":
        out = out.astype(vol.data.dtype)
    v = Volume(out, np.asarray(affine, float), vol.physical, dict(vol.meta))
    v.meta.setdefault("source_spacing_mm", voxel_spacing(vol.affine).tolist())
    v.meta["resampled"] = {"kind": kind, "order": o, "target_spacing_mm": voxel_spacing(affine).tolist(), "note": "resampling interpolates; it does not recover sub-voxel anatomy"}
    return v


def resample_spacing(vol: Volume, spacing, kind: str = "image", order: int = 1, cval: float = 0.0) -> Volume:
    aff, shape = _target_grid(vol, spacing)
    return resample_to_grid(vol, aff, shape, kind=kind, order=order, cval=cval)


def resample_like(vol: Volume, ref: Volume, kind="image", order=1, cval=0.0) -> Volume:
    return resample_to_grid(vol, ref.affine, ref.data.shape[:3], kind=kind, order=order, cval=cval)


def slice_normal(vol: Volume) -> np.ndarray:
    A = vol.affine[:3, :3]
    sp = voxel_spacing(vol.affine)
    k = int(np.argmax(sp))  # the through-plane axis is the coarsest one
    n = A[:, k] / sp[k]
    return n


def assert_not_stacking(vols: list[Volume], tol_deg: float = 10.0) -> None:
    """Raise if someone attempts to fuse differently oriented 2D series into one volume."""
    ns = [slice_normal(v) for v in vols]
    for a in ns[1:]:
        ang = np.degrees(np.arccos(np.clip(abs(float(a @ ns[0])), 0, 1)))
        if ang > tol_deg:
            raise ValueError(
                f"series have slice normals {ang:.0f} deg apart; stacking independently acquired planes into one isotropic volume is not supported"
            )


def coregister_series(moving: Volume, fixed: Volume, max_iter: int = 200) -> tuple[Volume, np.ndarray, dict]:
    """Rigid Mattes-MI registration of a second sequence onto a reference sequence (same exam).

    Returns (moving resampled on fixed grid, 4x4 RAS transform moving->fixed, metrics).
    Patient motion between series is the only thing corrected; this is not fusion.
    """
    import SimpleITK as sitk

    from .coords import LPS_TO_RAS
    from .volume import to_sitk

    f = sitk.Cast(to_sitk(fixed), sitk.sitkFloat32)
    m = sitk.Cast(to_sitk(moving), sitk.sitkFloat32)
    init = sitk.CenteredTransformInitializer(f, m, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.1, seed=0)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(2.0, 1e-4, max_iter, relaxationFactor=0.5)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SetInitialTransform(init, inPlace=False)
    tx = reg.Execute(f, m)
    # sitk transform maps fixed-space LPS points -> moving-space LPS points
    A = np.eye(4)
    tx3 = tx.GetNthTransform(0) if hasattr(tx, "GetNthTransform") and tx.GetNumberOfTransforms() > 0 else tx
    tx3 = sitk.Euler3DTransform(tx3) if not isinstance(tx3, sitk.Euler3DTransform) else tx3
    R = np.array(tx3.GetMatrix()).reshape(3, 3)
    c = np.array(tx3.GetCenter())
    t = np.array(tx3.GetTranslation())
    A[:3, :3] = R
    A[:3, 3] = c + t - R @ c
    fixed_to_moving_ras = LPS_TO_RAS @ A @ LPS_TO_RAS
    moving_to_fixed = np.linalg.inv(fixed_to_moving_ras)
    # resample: target voxel -> world(fixed) -> world(moving) -> moving voxel
    T = np.linalg.inv(moving.affine) @ fixed_to_moving_ras @ fixed.affine
    out = ndimage.affine_transform(moving.data.astype(np.float32), T[:3, :3], offset=T[:3, 3], output_shape=fixed.data.shape[:3], order=1)
    res = Volume(out, fixed.affine.copy(), moving.physical, dict(moving.meta))
    res.meta["coregistered_to"] = fixed.meta.get("series_uid", "reference")
    metrics = {"final_metric_value": float(reg.GetMetricValue()), "iterations": int(reg.GetOptimizerIteration()), "stop": reg.GetOptimizerStopConditionDescription()}
    return res, moving_to_fixed, metrics


# --------------------------------------------------------------------- QC


def qc_png(vol: Volume, path: str | Path, label: np.ndarray | None = None, title: str = "", index=None) -> Path:
    """Three orthogonal mid-slices (voxel axes) with optional label overlay and geometry text."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = vol.data.astype(np.float32)
    idx = index or [s // 2 for s in d.shape[:3]]
    sp = vol.spacing
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.4))
    for ax_i, (ax, name) in enumerate(zip(axs, ("i", "j", "k"))):
        # transpose so the second remaining voxel axis is vertical (with origin="lower",
        # superior is up for axial-acquired RAS-ordered grids)
        sl = np.take(d, idx[ax_i], axis=ax_i).T
        other = [a for a in range(3) if a != ax_i]
        aspect = sp[other[1]] / sp[other[0]]
        ax.imshow(sl, cmap="gray", aspect=aspect, origin="lower")
        if label is not None:
            ls = np.take(label, idx[ax_i], axis=ax_i).T.astype(float)
            ls[ls == 0] = np.nan
            ax.imshow(ls, cmap="tab20", alpha=0.45, aspect=aspect, origin="lower", interpolation="nearest", vmin=0, vmax=20)
        ax.set_title(f"axis {name} = {idx[ax_i]}")
        ax.axis("off")
    s = vol.summary()
    fig.suptitle(f"{title}\nshape {s['shape']}  spacing {s['spacing_mm']} mm  {s['axcodes']}  physical_verified={s['physical_verified']}", fontsize=9)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    return Path(path)
