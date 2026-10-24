"""Anatomy-estimation experiments.

1. healthy_bilateral_holdout: for bilateral healthy pairs, hide the target knee's ACL,
   estimate it, and compare with the hidden annotation. Methods:
     population_reference | contralateral_mirror | ssm_bone_conditioned |
     ssm_bone_conditioned+demographics
   Population/SSM models are refit per fold with BOTH knees of the target subject
   excluded (subject-level leave-one-out; no leakage).
2. synthetic_injury: simulated tears; compares contralateral_mirror with
   remnant_reconciled. Always labelled SYNTHETIC INJURY and never a substitute
   for testing on real ruptures.

This is an anatomy-estimation experiment. It says nothing about healing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..annotation.schema import Label, read_slicer_markups
from ..evaluation.metrics import bootstrap_ci, dice, point_error, surface_metrics
from ..imaging.volume import load_nifti
from ..provenance import EvidenceLevel, Provenance, write_json
from ..registration import bilateral as B
from .anatomy import estimate_contralateral, reconcile_with_remnants
from .geometry import centerline_profile
from .population import KneeEntry, StatisticalACLModel


def _metrics(pred_mask, pred_ff, pred_ft, truth_mask, truth_ff, truth_ft, affine):
    sp = np.linalg.norm(affine[:3, :3], axis=0)
    m = {"femoral_fp_error_mm": point_error(pred_ff, truth_ff), "tibial_fp_error_mm": point_error(pred_ft, truth_ft),
         "length_error_mm": float(np.linalg.norm(pred_ft - pred_ff) - np.linalg.norm(truth_ft - truth_ff)),
         "dice": dice(pred_mask, truth_mask), **surface_metrics(pred_mask, truth_mask, sp)}
    vv = float(np.prod(sp))
    m["volume_rel_error"] = (pred_mask.sum() - truth_mask.sum()) / truth_mask.sum() if truth_mask.sum() else float("nan")
    try:
        pp = centerline_profile(pred_mask, affine, pred_ff, pred_ft)
        tp = centerline_profile(truth_mask, affine, truth_ff, truth_ft)
        m["profile_area_rmse_mm2"] = float(np.sqrt(np.mean((pp["area_mm2"][1:-1] - tp["area_mm2"][1:-1]) ** 2)))
    except Exception:
        m["profile_area_rmse_mm2"] = float("nan")
    m["voxel_volume_mm3"] = vv
    return m


def _summarise(results: dict) -> dict:
    out = {}
    for method, per_subj in results.items():
        out[method] = {}
        keys = sorted({k for v in per_subj.values() for k in v if isinstance(v[k], float)})
        for k in keys:
            vals = {s: v[k] for s, v in per_subj.items() if k in v}
            if k in ("length_error_mm",):
                vals = {s: abs(x) for s, x in vals.items()}
            out[method][k] = bootstrap_ci(vals)
    return out


def load_case(case_dir: Path):
    c = json.loads((case_dir / "case.json").read_text())
    d = {"case": c}
    for role in ("injured", "contralateral"):
        d[f"{role}_labels"] = load_nifti(case_dir / c[role]["labels"])
        lm = case_dir / c[role]["landmarks"]
        d[f"{role}_landmarks"] = read_slicer_markups(lm) if lm.exists() else {}
    return d


def healthy_bilateral_holdout(case_dirs: list[Path], out_dir: Path, demographics: dict | None = None, seed: int = 0) -> dict:
    """case_dirs: bilateral cases whose 'injured' knee is actually intact (healthy pairs)."""
    cases = [load_case(Path(p)) for p in case_dirs]
    demographics = demographics or {}
    entries = []
    for c in cases:
        sid = c["case"]["subject_id"]
        for role in ("injured", "contralateral"):
            entries.append(KneeEntry(sid, c["case"][role]["side"], c[f"{role}_labels"], demographics.get(sid, {})))
    results = {k: {} for k in ("population_reference", "contralateral_mirror", "ssm_bone_conditioned", "ssm_bone_conditioned+demographics")}
    for c in cases:
        sid = c["case"]["subject_id"]
        tgt_lab = c["injured_labels"]
        truth = tgt_lab.data == Label.ACL_INTACT
        if not truth.any():
            continue
        tfp = B.acl_footprints_from_mask(tgt_lab.data, tgt_lab.affine)
        # target input with the ACL annotation REMOVED (held out)
        held = tgt_lab.with_data(np.where(np.isin(tgt_lab.data, [int(l) for l in (Label.ACL_INTACT,)]), 0, tgt_lab.data).astype(np.uint8))
        target = KneeEntry(sid, c["case"]["injured"]["side"], held, demographics.get(sid, {}))
        # contralateral
        reg = B.register_bilateral(held, c["contralateral_labels"], seed=seed)
        est = estimate_contralateral(held, c["contralateral_labels"], reg)
        results["contralateral_mirror"][sid] = _metrics(est.mask, est.femoral_fp, est.tibial_fp, truth, tfp["femoral"], tfp["tibial"], tgt_lab.affine)
        # population / SSM trained without this subject
        train = [e for e in entries if e.subject_id != sid]
        if len(train) >= 4:
            ref_entry = train[0]  # fixed choice so cached alignments are reused across folds
            mdl = StatisticalACLModel(n_modes=5).fit(train, ref_entry=ref_entry, seed=seed)
            m, ff, ft = mdl.predict_population(target, seed)
            results["population_reference"][sid] = _metrics(m, ff, ft, truth, tfp["femoral"], tfp["tibial"], tgt_lab.affine)
            m, ff, ft = mdl.predict_conditioned(target, seed)
            results["ssm_bone_conditioned"][sid] = _metrics(m, ff, ft, truth, tfp["femoral"], tfp["tibial"], tgt_lab.affine)
            if demographics:
                keys = tuple(sorted(next(iter(demographics.values())).keys()))
                mdl2 = StatisticalACLModel(n_modes=5, use_demographics=keys).fit(train, ref_entry=ref_entry, seed=seed)
                m, ff, ft = mdl2.predict_conditioned(target, seed)
                results["ssm_bone_conditioned+demographics"][sid] = _metrics(m, ff, ft, truth, tfp["femoral"], tfp["tibial"], tgt_lab.affine)
    results = {k: v for k, v in results.items() if v}
    synthetic = all(c["case"].get("synthetic") for c in cases)
    report = {
        "experiment": "healthy_bilateral_holdout",
        "evidence_level": EvidenceLevel.SYNTHETIC.value if synthetic else EvidenceLevel.ESTIMATED.value,
        "banner": "SYNTHETIC PHANTOM DATA - tests code paths and relative behaviour only; no clinical or anatomical accuracy" if synthetic else "",
        "n_subjects": len(cases),
        "summary": _summarise(results),
        "per_subject": results,
        "interpretation_limits": [
            "anatomy-estimation experiment; not evidence about ligament healing or clinical outcome",
            "truth is the held-out annotation of the same knee; annotation error is not modelled",
            "footprints are mask-contact centroids for every method (consistent definition)",
        ],
        "provenance": Provenance(EvidenceLevel.SYNTHETIC if synthetic else EvidenceLevel.ESTIMATED, "bilacl.estimation.experiment.healthy_bilateral_holdout",
                                 inputs=[str(p) for p in case_dirs], parameters={"seed": seed}).to_dict(),
    }
    write_json(Path(out_dir) / "healthy_bilateral_holdout.json", report)
    return report


def synthetic_injury_experiment(case_dirs: list[Path], out_dir: Path, seed: int = 0) -> dict:
    results = {"contralateral_mirror": {}, "remnant_reconciled": {}}
    flags = {}
    for p in case_dirs:
        c = load_case(Path(p))
        if not c["case"].get("synthetic"):
            raise ValueError("synthetic_injury_experiment only accepts synthetic cases")
        sid = c["case"]["subject_id"] + "_" + c["case"]["injured"]["tear_location"]
        truth_vol = load_nifti(Path(p) / "truth" / "injured_preinjury_acl.nii.gz")
        truth = truth_vol.data > 0
        il = c["injured_labels"]
        # truth footprints from the hidden pre-injury mask
        tl = il.data.copy()
        tl[truth] = Label.ACL_INTACT
        tfp = B.acl_footprints_from_mask(tl, il.affine)
        reg = B.register_bilateral(il, c["contralateral_labels"], seed=seed)
        a = estimate_contralateral(il, c["contralateral_labels"], reg)
        lm = {k: v for k, v in c["injured_landmarks"].items() if k.startswith("acl_")}
        b = reconcile_with_remnants(a, il, lm)
        for est in (a, b):
            results[est.name][sid] = _metrics(est.mask, est.femoral_fp, est.tibial_fp, truth, tfp["femoral"], tfp["tibial"], il.affine)
        flags[sid] = {"conflicts": b.conflicts, "notes": b.notes, "metrics": {k: v for k, v in b.metrics.items()}}
    report = {
        "experiment": "synthetic_injury",
        "evidence_level": EvidenceLevel.SYNTHETIC.value,
        "banner": "SYNTHETIC INJURY EXPERIMENT - simulated tears on phantoms; does NOT substitute for testing on real ruptures",
        "summary": _summarise(results), "per_case": results, "reconciliation_flags": flags,
    }
    write_json(Path(out_dir) / "synthetic_injury.json", report)
    return report


def markdown_table(report: dict, keys=("femoral_fp_error_mm", "tibial_fp_error_mm", "length_error_mm", "dice", "assd_mm", "hd95_mm")) -> str:
    lines = []
    if report.get("banner"):
        lines.append(f"> **{report['banner']}**\n")
    lines.append("| method | " + " | ".join(keys) + " |")
    lines.append("|---|" + "---|" * len(keys))
    for m, s in report["summary"].items():
        cells = []
        for k in keys:
            v = s.get(k)
            cells.append(f"{v['estimate']:.2f} [{v['ci_low']:.2f}, {v['ci_high']:.2f}]" if v and v.get("n_subjects") else "-")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines.append("\nValues: mean over subjects [subject-level bootstrap 95% CI]; |length error|.")
    return "\n".join(lines)
