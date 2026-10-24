"""Evaluation metrics. All spatial metrics are in millimetres and honour ignore masks."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def _valid(ignore, shape):
    return np.ones(shape, bool) if ignore is None else ~ignore


def dice(pred: np.ndarray, gt: np.ndarray, ignore: np.ndarray | None = None) -> float:
    v = _valid(ignore, gt.shape)
    p, g = pred.astype(bool) & v, gt.astype(bool) & v
    s = p.sum() + g.sum()
    return float(2 * (p & g).sum() / s) if s else float("nan")


def surface_metrics(pred: np.ndarray, gt: np.ndarray, spacing, ignore: np.ndarray | None = None) -> dict:
    """ASSD and HD95 (mm). Surface voxels in ignored regions are excluded."""
    v = _valid(ignore, gt.shape)
    p, g = pred.astype(bool), gt.astype(bool)
    if not p.any() or not g.any():
        return {"assd_mm": float("nan"), "hd95_mm": float("nan"), "hd_mm": float("nan")}
    sp = np.asarray(spacing, float)
    sp_ = p & ~ndimage.binary_erosion(p) & v
    sg_ = g & ~ndimage.binary_erosion(g) & v
    if not sp_.any() or not sg_.any():
        return {"assd_mm": float("nan"), "hd95_mm": float("nan"), "hd_mm": float("nan")}
    a, b = np.argwhere(sp_) * sp, np.argwhere(sg_) * sp
    da, _ = cKDTree(b).query(a)
    db, _ = cKDTree(a).query(b)
    d = np.concatenate([da, db])
    return {"assd_mm": float(d.mean()), "hd95_mm": float(np.percentile(d, 95)), "hd_mm": float(d.max())}


def volume_error(pred, gt, voxel_volume_mm3: float, ignore=None) -> dict:
    v = _valid(ignore, gt.shape)
    vp = float((pred.astype(bool) & v).sum() * voxel_volume_mm3)
    vg = float((gt.astype(bool) & v).sum() * voxel_volume_mm3)
    return {"pred_mm3": vp, "gt_mm3": vg, "abs_error_mm3": abs(vp - vg), "rel_error": (vp - vg) / vg if vg else float("nan")}


def segmentation_report(pred, gt, spacing, ignore=None) -> dict:
    vv = float(np.prod(spacing))
    return {"dice": dice(pred, gt, ignore), **surface_metrics(pred, gt, spacing, ignore), "volume": volume_error(pred, gt, vv, ignore)}


def point_error(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


# ----------------------------------------------------------------- classification


def classification_report(y_true, y_score, threshold: float = 0.5, n_bins: int = 10) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, float)
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "threshold": threshold, "threshold_note": "documented operating point; choose on validation data only"}
    if 0 < y.sum() < len(y):
        out["auroc"] = float(roc_auc_score(y, s))
        out["auprc"] = float(average_precision_score(y, s))
    pred = s >= threshold
    tp, fn = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum())
    tn, fp = int((~pred & (y == 0)).sum()), int((pred & (y == 0)).sum())
    out["sensitivity"] = tp / (tp + fn) if tp + fn else float("nan")
    out["specificity"] = tn / (tn + fp) if tn + fp else float("nan")
    out["calibration"] = calibration(y, s, n_bins)
    return out


def calibration(y, s, n_bins=10) -> dict:
    y, s = np.asarray(y, float), np.asarray(s, float)
    edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    ece = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        k = (s >= a) & ((s < b) if b < 1 else (s <= b))
        if k.any():
            conf, acc = float(s[k].mean()), float(y[k].mean())
            ece += k.mean() * abs(conf - acc)
            bins.append({"lo": a, "hi": b, "n": int(k.sum()), "mean_score": conf, "observed_rate": acc})
    return {"ece": float(ece), "brier": float(np.mean((s - y) ** 2)), "bins": bins}


# ----------------------------------------------------------------- subject-level CIs


def bootstrap_ci(values_by_subject: dict, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0, stat=np.mean) -> dict:
    """Resample *subjects* (each subject may contribute several values, which are averaged first)."""
    subj = list(values_by_subject)
    per = np.array([np.nanmean(np.atleast_1d(values_by_subject[s])) for s in subj], float)
    per = per[np.isfinite(per)]
    if len(per) == 0:
        return {"n_subjects": 0}
    rng = np.random.default_rng(seed)
    boots = [stat(per[rng.integers(0, len(per), len(per))]) for _ in range(n_boot)] if len(per) > 1 else [stat(per)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"n_subjects": int(len(per)), "estimate": float(stat(per)), "ci_low": float(lo), "ci_high": float(hi),
            "note": "subject-level bootstrap" + (" (n too small for a meaningful CI)" if len(per) < 10 else "")}
