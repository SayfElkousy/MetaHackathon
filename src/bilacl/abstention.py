"""Abstention conditions and the physical-export gate.

ALL THRESHOLDS BELOW ARE ENGINEERING SETTINGS, NOT VALIDATED CLINICAL CRITERIA.
They exist so the pipeline stops instead of producing confident-looking output
from unsupported inputs. They must be revisited with real data and inter-reader
variability before any scientific use.
"""
from __future__ import annotations

import dataclasses

from .provenance import ReviewStatus

THRESHOLDS = {
    "_status": "engineering settings - not validated",
    "max_voxel_dim_mm_for_design": 1.0,  # coarser acquisitions (e.g. 3-4 mm 2D slices) cannot support mm-scale design export
    "max_surface_rms_mm": 1.0,  # per-bone ICP residual
    "min_inlier_fraction": 0.8,  # fraction of surface points within 2 mm after ICP
    "max_landmark_mean_mm": 3.0,  # independent landmark error (includes true asymmetry)
    "max_relative_rotation_deg": 45.0,  # tibiofemoral pose difference beyond which rigid-per-bone blending is untrusted
    "max_footprint_sd_mm": 2.5,  # per uncertainty source
    "min_remnant_fraction_consistent": 0.75,
}


@dataclasses.dataclass
class Check:
    name: str
    status: str  # pass | fail | unknown
    detail: str
    blocks: tuple = ("design_export",)  # what a failure blocks: estimation, design_export


def _c(name, ok, detail, blocks=("design_export",)):
    return Check(name, "unknown" if ok is None else ("pass" if ok else "fail"), detail, blocks)


def evaluate(case: dict, inj_vol, con_vol, reg_dict: dict | None, qc: dict | None, est_conflicts: list | None,
             est_metrics: dict | None, unc: dict | None, synthetic: bool, thresholds: dict | None = None) -> dict:
    th = {**THRESHOLDS, **(thresholds or {})}
    checks: list[Check] = []
    inj_side, con_side = case["injured"].get("side"), case["contralateral"].get("side")
    known = inj_side in ("left", "right") and con_side in ("left", "right")
    checks.append(_c("laterality_known_and_opposite", known and inj_side != con_side if known else None,
                     f"injured={inj_side}, contralateral={con_side}", ("estimation", "design_export")))
    for role, v in (("injured", inj_vol), ("contralateral", con_vol)):
        checks.append(_c(f"{role}_physical_metadata_verified", v.physical.verified, f"{v.physical.source}; {'; '.join(v.physical.reasons) or 'ok'}",
                         ("design_export",)))
        vmax = float(max(v.spacing))
        checks.append(_c(f"{role}_resolution_supports_design", vmax <= th["max_voxel_dim_mm_for_design"],
                         f"largest voxel dimension {vmax:.2f} mm (limit {th['max_voxel_dim_mm_for_design']} mm)"))
    conf = case["contralateral"].get("contralateral_confirmed_normal")
    if synthetic:
        checks.append(Check("contralateral_expert_confirmed_normal", "unknown", "synthetic case: no expert confirmation exists", ()))
    else:
        checks.append(_c("contralateral_expert_confirmed_normal", conf if conf is not None else None,
                         f"pathology: {case['contralateral'].get('contralateral_pathology', 'unknown')}", ("estimation", "design_export")))
    if reg_dict:
        for bone in ("femur_icp", "tibia_icp"):
            r = reg_dict[bone]
            checks.append(_c(f"{bone}_residual", r["rms_mm"] <= th["max_surface_rms_mm"], f"rms {r['rms_mm']:.2f} mm (limit {th['max_surface_rms_mm']})", ("estimation", "design_export")))
            checks.append(_c(f"{bone}_inliers", r["inlier_fraction_2mm"] >= th["min_inlier_fraction"], f"{r['inlier_fraction_2mm']:.2f} within 2 mm", ("estimation", "design_export")))
        rot = reg_dict.get("relative_tibiofemoral_rotation_deg")
        checks.append(_c("pose_difference_supported", rot is not None and rot <= th["max_relative_rotation_deg"], f"{rot:.1f} deg"))
    if qc and qc.get("landmarks", {}).get("mean_mm") is not None:
        lm = qc["landmarks"]["mean_mm"]
        checks.append(_c("independent_landmark_error", lm <= th["max_landmark_mean_mm"], f"mean {lm:.2f} mm over {qc['landmarks']['n']} landmarks"))
    else:
        checks.append(_c("independent_landmark_error", None, "no independent landmarks available: registration accuracy unverified"))
    if est_conflicts is not None:
        checks.append(_c("remnant_conflicts_resolved", len(est_conflicts) == 0, "; ".join(est_conflicts) or "none"))
    if est_metrics and "remnant_fraction_consistent" in est_metrics:
        f = est_metrics["remnant_fraction_consistent"]
        checks.append(_c("remnants_consistent_with_reference", f >= th["min_remnant_fraction_consistent"], f"{f:.2f} of remnant voxels within tolerance"))
    if unc:
        for src, u in unc.items():
            m = max(u["femoral_fp_sd_mm"], u["tibial_fp_sd_mm"])
            checks.append(_c(f"uncertainty_{src}", m <= th["max_footprint_sd_mm"], f"footprint SD {m:.2f} mm"))
    blocked_est = [c for c in checks if c.status == "fail" and "estimation" in c.blocks]
    blocked_exp = [c for c in checks if c.status != "pass" and "design_export" in c.blocks]
    return {
        "thresholds": th,
        "checks": [dataclasses.asdict(c) for c in checks],
        "abstain_estimation": bool(blocked_est),
        "abstain_design_export": bool(blocked_exp),
        "reasons_estimation": [f"{c.name}: {c.detail}" for c in blocked_est],
        "reasons_design_export": [f"{c.name} [{c.status}]: {c.detail}" for c in blocked_exp],
    }


def export_gate(abst: dict, anatomy_review: ReviewStatus, mesh_valid: bool, synthetic: bool, inj_physical_verified: bool) -> dict:
    """Decide whether STL / millimetre geometry may be written.

    Hard blocks (no override): unverified physical metadata, invalid mesh.
    Real data additionally require: no abstention, expert-APPROVED anatomy estimate.
    Synthetic data may export for software testing; files are prefixed SYNTHETIC_."""
    reasons = []
    if not inj_physical_verified:
        reasons.append("source image physical spacing/orientation not verified: millimetre export disabled")
    if not mesh_valid:
        reasons.append("mesh failed geometric validation")
    if not synthetic:
        if abst["abstain_design_export"]:
            reasons += abst["reasons_design_export"]
        if anatomy_review != ReviewStatus.APPROVED:
            reasons.append(f"anatomy estimate review status is {anatomy_review.value}; expert approval required")
    allowed = not reasons
    return {"allowed": allowed, "reasons": reasons, "synthetic_demo": synthetic and allowed,
            "note": "synthetic export permitted for software testing only" if synthetic else ""}
