"""Native-ACL anatomy estimation for the injured knee from the contralateral knee.

Outputs are ESTIMATED anatomy (never observed). Alternatives are kept side by side:

* contralateral_mirror   : mirrored + per-bone registered contralateral ACL
* remnant_reconciled     : the above, end-translated toward *observed* injured-side
                           attachments (landmarks or remnant-bone contact), capped,
                           and checked against visible remnants
Uncertainty is computed per source (registration, bilateral asymmetry, contralateral
segmentation) and never merged into one number, because they have different causes.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from ..annotation.schema import Label
from ..imaging.coords import apply_affine, rigid_from_params
from ..imaging.volume import Volume
from ..provenance import EvidenceLevel
from ..registration import bilateral as B
from .geometry import centerline_profile, displace_mask, mask_points

DEFAULTS = {
    # ENGINEERING SETTINGS (not validated)
    "max_end_shift_mm": 4.0,  # largest correction toward observed attachments before flagging conflict
    "remnant_tolerance_mm": 1.5,  # remnant voxels within this distance of the envelope count as consistent
    "asymmetry_sd_mm": 1.5,  # per-axis SD of footprint L/R difference prior (order of Jang et al. 2017, n=10 cadaver pairs)
    "ensemble_size": 12,
    "registration_perturb_deg": 2.0,
    "registration_perturb_mm": 1.0,
}


@dataclasses.dataclass
class AnatomyEstimate:
    name: str
    mask: np.ndarray  # bool, injured grid
    femoral_fp: np.ndarray
    tibial_fp: np.ndarray
    evidence_level: EvidenceLevel = EvidenceLevel.ESTIMATED
    probability: np.ndarray | None = None
    notes: list[str] = dataclasses.field(default_factory=list)
    conflicts: list[str] = dataclasses.field(default_factory=list)
    metrics: dict = dataclasses.field(default_factory=dict)

    def summary(self, affine) -> dict:
        prof = centerline_profile(self.mask, affine, self.femoral_fp, self.tibial_fp) if self.mask.any() else None
        return {
            "name": self.name, "evidence_level": self.evidence_level.value,
            "femoral_fp_world_mm": self.femoral_fp.tolist(), "tibial_fp_world_mm": self.tibial_fp.tolist(),
            "voxels": int(self.mask.sum()),
            "volume_mm3": prof["volume_mm3"] if prof else 0.0,
            "straight_length_mm": prof["straight_length_mm"] if prof else None,
            "mean_area_mm2": float(np.mean(prof["area_mm2"][2:-2])) if prof else None,
            "notes": self.notes, "conflicts": self.conflicts, "metrics": self.metrics,
        }


def roi_transfer(src_mask: np.ndarray, src_affine: np.ndarray, reg: B.BilateralRegistration, target: Volume, margin_mm: float = 14.0) -> np.ndarray:
    """Transfer a contralateral mask to the injured grid, evaluated only in an ROI around the ACL axis."""
    a, b = reg.axis_femoral, reg.axis_tibial
    lo, hi = np.minimum(a, b) - margin_mm, np.maximum(a, b) + margin_mm
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    inv = np.linalg.inv(target.affine)
    vc = apply_affine(inv, corners)
    shape = np.array(target.data.shape[:3])
    vlo = np.clip(np.floor(vc.min(0)).astype(int), 0, shape)
    vhi = np.clip(np.ceil(vc.max(0)).astype(int) + 1, 0, shape)
    out = np.zeros(tuple(shape), bool)
    ijk = np.indices(vhi - vlo).reshape(3, -1).T + vlo
    X = apply_affine(target.affine, ijk)
    P = B.inverse_blended_map(X, reg)
    sv = apply_affine(np.linalg.inv(src_affine), P)
    vals = ndimage.map_coordinates(src_mask.astype(np.uint8), sv.T, order=0, mode="constant")
    out[tuple(ijk[vals > 0].T)] = True
    return out


def observed_attachments(inj_labels: Volume, inj_landmarks: dict | None) -> dict:
    """Injured-side attachment evidence. Expert landmarks take priority; otherwise remnant voxels
    touching the corresponding bone (femoral remnant <-> femur, tibial remnant <-> tibia)."""
    out = {"femoral": None, "tibial": None, "source": {}}
    lm = inj_landmarks or {}
    for side, key in (("femoral", "acl_femoral_footprint_center"), ("tibial", "acl_tibial_footprint_center")):
        if key in lm:
            out[side] = np.asarray(lm[key], float)
            out["source"][side] = "expert_landmark (observed)"
    lab = inj_labels.data
    for side, code, bone in (("femoral", Label.ACL_REMNANT_FEMORAL, Label.FEMUR), ("tibial", Label.ACL_REMNANT_TIBIAL, Label.TIBIA)):
        if out[side] is None:
            near = ndimage.binary_dilation(lab == bone, iterations=2) & (lab == code)
            if near.sum() >= 5:
                out[side] = apply_affine(inj_labels.affine, np.argwhere(near).mean(0)[None])[0]
                out["source"][side] = "remnant-bone contact centroid (estimated from observed remnant)"
    # intact (partial tear) ACL touching bone also counts as attachment evidence
    for side, bone in (("femoral", Label.FEMUR), ("tibial", Label.TIBIA)):
        if out[side] is None:
            near = ndimage.binary_dilation(lab == bone, iterations=2) & (lab == Label.ACL_INTACT)
            if near.sum() >= 5:
                out[side] = apply_affine(inj_labels.affine, np.argwhere(near).mean(0)[None])[0]
                out["source"][side] = "intact-fibre bone contact (partial tear)"
    return out


def estimate_contralateral(inj_labels: Volume, con_labels: Volume, reg: B.BilateralRegistration) -> AnatomyEstimate:
    acl = con_labels.data == Label.ACL_INTACT
    if not acl.any():
        raise ValueError("contralateral knee has no ACL_INTACT annotation; it cannot serve as a reference")
    m = roi_transfer(acl, con_labels.affine, reg, inj_labels)
    # never place estimated ligament inside observed bone
    m &= ~np.isin(inj_labels.data, [Label.FEMUR, Label.TIBIA])
    e = AnatomyEstimate("contralateral_mirror", m, reg.axis_femoral.copy(), reg.axis_tibial.copy())
    e.notes.append("mirrored contralateral ACL transferred by axis-blended per-bone rigid transforms")
    return e


def reconcile_with_remnants(base: AnatomyEstimate, inj_labels: Volume, inj_landmarks: dict | None, cfg: dict | None = None) -> AnatomyEstimate:
    cfg = {**DEFAULTS, **(cfg or {})}
    obs = observed_attachments(inj_labels, inj_landmarks)
    shifts = {}
    conflicts, notes = [], []
    for side, est in (("femoral", base.femoral_fp), ("tibial", base.tibial_fp)):
        if obs[side] is None:
            shifts[side] = np.zeros(3)
            notes.append(f"no observed {side} attachment evidence; contralateral estimate kept")
            continue
        d = obs[side] - est
        n = float(np.linalg.norm(d))
        if n > cfg["max_end_shift_mm"]:
            conflicts.append(f"{side}: observed attachment is {n:.1f} mm from the contralateral estimate (> {cfg['max_end_shift_mm']} mm cap); "
                             "correction not applied; requires expert review")
            shifts[side] = np.zeros(3)
        else:
            shifts[side] = d
            notes.append(f"{side} end translated {n:.2f} mm toward observed attachment ({obs['source'][side]})")
    m = displace_mask(base.mask, inj_labels.affine, base.femoral_fp, base.tibial_fp, shifts["femoral"], shifts["tibial"])
    m &= ~np.isin(inj_labels.data, [Label.FEMUR, Label.TIBIA])
    # remnant consistency
    rem_codes = [Label.ACL_REMNANT_FEMORAL, Label.ACL_REMNANT_TIBIAL, Label.ACL_FRAGMENT_OTHER, Label.ACL_INTACT]
    rem = np.isin(inj_labels.data, rem_codes)
    metrics = {"remnant_voxels": int(rem.sum())}
    if rem.any() and m.any():
        sp = np.linalg.norm(inj_labels.affine[:3, :3], axis=0)
        dist = ndimage.distance_transform_edt(~m, sampling=sp)
        drem = dist[rem]
        inside = drem <= cfg["remnant_tolerance_mm"]
        metrics["remnant_fraction_consistent"] = float(inside.mean())
        metrics["remnant_max_distance_mm"] = float(drem.max())
        consistent = np.zeros_like(rem)
        consistent[rem] = inside
        m |= consistent  # observed tissue consistent with the envelope is part of it
        if (~inside).sum() > 0.1 * rem.sum():
            conflicts.append(f"{int((~inside).sum())} remnant voxels lie > {cfg['remnant_tolerance_mm']} mm outside the envelope "
                             "(displaced/retracted remnant, asymmetry, or registration error); they were NOT absorbed into the estimate")
    elif not rem.any():
        notes.append("no visible remnant tissue annotated; estimate relies on contralateral reference only")
    e = AnatomyEstimate("remnant_reconciled", m, base.femoral_fp + shifts["femoral"], base.tibial_fp + shifts["tibial"],
                        notes=base.notes + notes, conflicts=conflicts, metrics=metrics)
    e.metrics["attachment_sources"] = obs["source"]
    return e


# ----------------------------------------------------------------- uncertainty


def _perturb(T, rng, deg, mm, center):
    R = rigid_from_params(*np.radians(rng.normal(0, deg, 3)), t=rng.normal(0, mm, 3), center=center)
    return R @ T


def uncertainty_ensembles(inj_labels: Volume, con_labels: Volume, reg: B.BilateralRegistration, cfg: dict | None = None, seed: int = 0) -> dict:
    """Separate ensembles per uncertainty source. Returns occupancy probability maps (injured grid)
    and footprint dispersion for each source.

    registration : ICP re-run from perturbed starts with random subsampling
    asymmetry    : random footprint offsets from the bilateral asymmetry prior
    segmentation : contralateral ACL mask randomly eroded/dilated by one voxel

    What is NOT represented: true unknown pre-injury shape beyond the asymmetry prior,
    and anything about microstructure (fibre organisation is not observable here)."""
    cfg = {**DEFAULTS, **(cfg or {})}
    rng = np.random.default_rng(seed)
    acl = con_labels.data == Label.ACL_INTACT
    bone_inj = np.isin(inj_labels.data, [Label.FEMUR, Label.TIBIA])
    A_m = reg.mirror @ con_labels.affine
    surf = {}
    for bone in (Label.FEMUR, Label.TIBIA):
        surf[("inj", bone)] = B.crop_sphere(B.surface_points(inj_labels.data == bone, inj_labels.affine), reg.joint_center_injured, reg.crop_radius_mm)
        surf[("con", bone)] = B.crop_sphere(B.surface_points(con_labels.data == bone, A_m), apply_affine(np.linalg.inv(reg.T_femur), reg.joint_center_injured[None])[0], reg.crop_radius_mm)
    fp_con = B.acl_footprints_from_mask(con_labels.data, con_labels.affine)
    out = {}
    n = cfg["ensemble_size"]
    for source in ("registration", "asymmetry", "segmentation"):
        acc = np.zeros(inj_labels.data.shape[:3], np.float32)
        fps_f, fps_t = [], []
        for _ in range(n):
            r = dataclasses.replace(reg)
            src = acl
            if source == "registration":
                jc = reg.joint_center_injured
                f = B.icp(surf[("con", Label.FEMUR)], surf[("inj", Label.FEMUR)], _perturb(reg.T_femur, rng, cfg["registration_perturb_deg"], cfg["registration_perturb_mm"], jc), rng=rng, max_pts=2500, max_iter=40)
                t = B.icp(surf[("con", Label.TIBIA)], surf[("inj", Label.TIBIA)], _perturb(reg.T_tibia, rng, cfg["registration_perturb_deg"], cfg["registration_perturb_mm"], jc), rng=rng, max_pts=2500, max_iter=40)
                r.T_femur, r.T_tibia = f.T, t.T
                r.axis_femoral = apply_affine(f.T @ reg.mirror, fp_con["femoral"][None])[0]
                r.axis_tibial = apply_affine(t.T @ reg.mirror, fp_con["tibial"][None])[0]
            elif source == "segmentation":
                # random one-voxel boundary perturbation (dilate / erode / boundary-voxel dropout)
                k = rng.integers(-1, 2)
                src = ndimage.binary_dilation(acl) if k > 0 else ndimage.binary_erosion(acl) if k < 0 else acl
                edge = src & ~ndimage.binary_erosion(src)
                src = src & ~(edge & (rng.random(src.shape) < 0.3))
                pl = np.where(src, Label.ACL_INTACT, np.where(con_labels.data == Label.ACL_INTACT, 0, con_labels.data))
                fpp = B.acl_footprints_from_mask(pl, con_labels.affine)
                if fpp["femoral"] is not None and fpp["tibial"] is not None:
                    r.axis_femoral = apply_affine(reg.contra_to_injured_femur, fpp["femoral"][None])[0]
                    r.axis_tibial = apply_affine(reg.contra_to_injured_tibia, fpp["tibial"][None])[0]
            m = roi_transfer(src, con_labels.affine, r, inj_labels) & ~bone_inj
            ff, tt = r.axis_femoral.copy(), r.axis_tibial.copy()
            if source == "asymmetry":
                sf, st = rng.normal(0, cfg["asymmetry_sd_mm"], 3), rng.normal(0, cfg["asymmetry_sd_mm"], 3)
                m = displace_mask(m, inj_labels.affine, ff, tt, sf, st) & ~bone_inj
                ff, tt = ff + sf, tt + st
            acc += m
            fps_f.append(ff)
            fps_t.append(tt)
        prob = acc / n
        fps_f, fps_t = np.array(fps_f), np.array(fps_t)
        out[source] = {
            "probability": prob,
            "femoral_fp_sd_mm": float(np.sqrt(np.trace(np.cov(fps_f.T)))) if n > 1 else 0.0,
            "tibial_fp_sd_mm": float(np.sqrt(np.trace(np.cov(fps_t.T)))) if n > 1 else 0.0,
            "uncertain_voxels": int(((prob > 0.05) & (prob < 0.95)).sum()),
            "n": n,
        }
    return out


def uncertainty_summary(ens: dict) -> dict:
    return {k: {kk: vv for kk, vv in v.items() if kk != "probability"} for k, v in ens.items()}
