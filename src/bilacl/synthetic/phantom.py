"""Synthetic bilateral knee phantoms.

PURPOSE: exercise coordinate handling, mirroring, per-bone registration, remnant
handling and geometry generation with known ground truth. The shapes are crude
geometric primitives (ellipsoids, cylinders, swept tubes). Nothing produced here
has anatomical or clinical accuracy, and every output is stamped
EvidenceLevel.SYNTHETIC.

Model (knee-local frame of a RIGHT knee, mm, RAS-like axes):
    +x lateral (patient right for a right knee), +y anterior, +z superior,
    joint line near z = 0, femur above, tibia below.
A LEFT knee is the right-knee template reflected in x, plus independent
"asymmetry" perturbations. The tibia (and everything attached to it) is rotated
about a flexion axis through the femoral condyles, so the injured and the
contralateral knee can be imaged in different poses. Each image also gets its
own scanner positioning transform and its own voxel grid.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from ..annotation.schema import AnnotationRecord, Label, write_slicer_markups
from ..imaging.coords import apply_affine, mirror_matrix, rigid_from_params
from ..imaging.volume import PhysicalStatus, Volume, save_nifti
from ..provenance import EvidenceLevel, Provenance, ReviewStatus, write_json

FLEX_CENTER = np.array([0.0, -5.0, 12.0])  # approx. condylar centre; flexion axis is +x through here


@dataclasses.dataclass
class KneeParams:
    scale: float = 1.0
    fem_fp: tuple = (8.0, -14.0, 12.0)  # femoral footprint target (knee-local, right-knee template)
    tib_fp: tuple = (-2.0, 10.0, -2.5)
    bend: tuple = (0.0, 1.5, 0.0)  # bezier control offset for ACL curvature
    acl_r: tuple = (4.6, 3.2)  # mid-substance semi-axes (mm)
    fp_flare: float = 1.45  # footprint flare factor
    twist_deg: float = 25.0  # rotation of cross-section ellipse along the ligament
    condyle_r: float = 12.0
    plateau: tuple = (28.0, 20.0)


def sample_subject(rng: np.random.Generator) -> KneeParams:
    p = KneeParams()
    p.scale = float(rng.normal(1.0, 0.06))
    p.fem_fp = tuple(np.array(p.fem_fp) + rng.normal(0, 1.6, 3))
    p.tib_fp = tuple(np.array(p.tib_fp) + rng.normal(0, 1.6, 3))
    p.bend = tuple(np.array(p.bend) + rng.normal(0, 0.8, 3))
    p.acl_r = tuple(np.array(p.acl_r) * rng.normal(1.0, 0.1))
    p.twist_deg = float(rng.normal(25, 8))
    p.condyle_r = float(rng.normal(12.0, 0.6))
    p.plateau = tuple(np.array(p.plateau) * rng.normal(1.0, 0.04))
    return p


def contralateral_of(p: KneeParams, rng: np.random.Generator, asym_mm: float = 1.5) -> KneeParams:
    """Same subject's other knee: small independent left-right differences.
    asym_mm ~1.5 per axis gives footprint centre differences of ~2.6 mm on average,
    the order reported for cadaveric bilateral pairs by Jang et al. 2017 (n=10 pairs)."""
    q = dataclasses.replace(p)
    q.scale = p.scale * float(rng.normal(1.0, 0.01))
    q.fem_fp = tuple(np.array(p.fem_fp) + rng.normal(0, asym_mm, 3))
    q.tib_fp = tuple(np.array(p.tib_fp) + rng.normal(0, asym_mm, 3))
    q.acl_r = tuple(np.array(p.acl_r) * rng.normal(1.0, 0.05))
    q.twist_deg = p.twist_deg + float(rng.normal(0, 3))
    q.condyle_r = p.condyle_r * float(rng.normal(1.0, 0.01))
    return q


# ----------------------------------------------------------------- shape primitives (local frame)


def _ellipsoid(P, c, r):
    return (((P - np.asarray(c)) / np.asarray(r)) ** 2).sum(-1) <= 1.0


def _zcyl(P, c_xy, r_xy, z0, z1):
    q = (P[..., :2] - np.asarray(c_xy)) / np.asarray(r_xy)
    return ((q ** 2).sum(-1) <= 1.0) & (P[..., 2] >= z0) & (P[..., 2] <= z1)


def femur_mask(P, kp: KneeParams):
    s = kp.scale
    Q = P / s
    r = kp.condyle_r
    m = _ellipsoid(Q, (12, -5, 12), (r, r * 1.25, r))
    m |= _ellipsoid(Q, (-12, -5, 12), (r * 1.04, r * 1.3, r * 1.04))
    m |= _ellipsoid(Q, (0, -1, 26), (27, 18, 14))
    m |= _zcyl(Q, (0, 0), (14, 13), 26, 80)
    notch = (np.abs(Q[..., 0]) < 6.5) & (Q[..., 1] < 6) & (Q[..., 2] < 15) & (Q[..., 2] > -5)
    return m & ~notch


def tibia_mask(P, kp: KneeParams):
    s = kp.scale
    Q = P / s
    a, b = kp.plateau
    m = _zcyl(Q, (0, 0), (a, b), -22, -2.0)
    m |= _ellipsoid(Q, (0, 0, -24), (a * 0.85, b * 0.9, 10))
    m |= _zcyl(Q, (0, 2), (15, 14), -80, -22)
    m |= _ellipsoid(Q, (-2.5, -1, -2.0), (5, 6, 3.0))  # intercondylar eminence
    m |= _ellipsoid(Q, (0, 19, -26), (7, 5, 7))  # tuberosity
    return m


def bezier(p0, c, p1, t):
    t = np.asarray(t)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t ** 2 * p1


def ligament_tube(P, p0, p1, bend, r_mid, flare, twist_deg, n=240):
    """Signed occupancy of a swept elliptical tube. Returns (mask, t_param) where t is the
    normalised arc position (0 femoral end, 1 tibial end) of the closest centreline sample."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    c = 0.5 * (p0 + p1) + np.asarray(bend, float)
    ts = np.linspace(-0.08, 1.08, n)
    C = bezier(p0, c, p1, ts)
    T = np.gradient(C, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    ref = np.array([1.0, 0, 0])
    U = np.cross(T, ref)
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    V = np.cross(T, U)
    flat = P.reshape(-1, 3)
    # restrict to bounding box for speed
    lo, hi = C.min(0) - 12, C.max(0) + 12
    inbox = np.all((flat >= lo) & (flat <= hi), axis=1)
    idx = np.where(inbox)[0]
    Q = flat[idx]
    from scipy.spatial import cKDTree

    _, j = cKDTree(C).query(Q)
    rel = Q - C[j]
    tw = np.radians(twist_deg) * np.clip(ts[j], 0, 1)
    u = (rel * U[j]).sum(1)
    v = (rel * V[j]).sum(1)
    uu = u * np.cos(tw) + v * np.sin(tw)
    vv = -u * np.sin(tw) + v * np.cos(tw)
    tt = np.clip(ts[j], 0, 1)
    fl = 1 + (flare - 1) * (np.clip(1 - tt / 0.18, 0, 1) ** 2 + np.clip((tt - 0.82) / 0.18, 0, 1) ** 2)
    inside = (uu / (r_mid[0] * fl)) ** 2 + (vv / (r_mid[1] * fl)) ** 2 <= 1.0
    along = (rel * T[j]).sum(1)
    inside &= np.abs(along) < 1.0  # avoid end-cap spill
    mask = np.zeros(flat.shape[0], bool)
    tpar = np.full(flat.shape[0], np.nan)
    mask[idx] = inside
    tpar[idx] = ts[j]
    return mask.reshape(P.shape[:-1]), tpar.reshape(P.shape[:-1]), C


# ----------------------------------------------------------------- image synthesis


@dataclasses.dataclass
class Injury:
    kind: str = "midsubstance"  # intact|proximal|midsubstance|distal|partial|chronic_absent
    t_center: float = 0.5
    gap: float = 0.18  # fraction of ligament length
    unknown_margin: float = 0.06  # region around the gap annotated UNKNOWN


INJURY_PRESETS = {
    "intact": Injury("intact", 0.5, 0.0, 0.0),
    "proximal": Injury("proximal", 0.15, 0.12, 0.05),
    "midsubstance": Injury("midsubstance", 0.5, 0.18, 0.06),
    "distal": Injury("distal", 0.85, 0.12, 0.05),
    "partial": Injury("partial", 0.5, 0.25, 0.05),
    "chronic_absent": Injury("chronic_absent", 0.5, 1.0, 0.0),
}


def render_knee(side: str, kp: KneeParams, flex_deg: float, scan_T: np.ndarray, shape, spacing, rng, injury: Injury | None = None, noise=0.04):
    """Render one knee into a voxel grid. Returns dict of Volume objects + truth geometry."""
    spacing = np.asarray(spacing, float)
    shape = tuple(shape)
    # voxel -> scanner world (RAS mm): centred grid
    aff = np.diag([*spacing, 1.0])
    aff[:3, 3] = -0.5 * (np.array(shape) - 1) * spacing
    ii = np.indices(shape).reshape(3, -1).T.astype(float)
    W = apply_affine(aff, ii)
    # scanner world -> knee-local; scan_T maps local -> world
    L = apply_affine(np.linalg.inv(scan_T), W)
    M = mirror_matrix() if side == "left" else np.eye(4)
    Ltempl = apply_affine(M, L)  # into right-knee template coordinates
    F = rigid_from_params(np.radians(flex_deg), 0, 0, center=FLEX_CENTER)  # tibia pose in template
    Ltib = apply_affine(np.linalg.inv(F), Ltempl)
    fem = femur_mask(Ltempl, kp).reshape(shape)
    tib = tibia_mask(Ltib, kp).reshape(shape)
    s = kp.scale
    p0 = np.array(kp.fem_fp) * s
    p1_rest = np.array(kp.tib_fp) * s
    p1 = apply_affine(F, p1_rest[None])[0]
    bend = np.array(kp.bend) * s
    acl, tpar, C = ligament_tube(Ltempl.reshape(*shape, 3), p0, p1, bend, np.array(kp.acl_r) * s, kp.fp_flare, kp.twist_deg)
    acl &= ~fem & ~tib
    # PCL for context (posterior)
    pcl_p0 = np.array([-6.0, -4.0, 10.0]) * s
    pcl_p1 = apply_affine(F, (np.array([0.0, -17.0, -6.0]) * s)[None])[0]
    pcl, _, _ = ligament_tube(Ltempl.reshape(*shape, 3), pcl_p0, pcl_p1, (0, -3, 0), np.array([5.0, 3.5]) * s, 1.3, 0)
    pcl &= ~fem & ~tib & ~acl

    labels = np.zeros(shape, np.uint8)
    labels[fem] = Label.FEMUR
    labels[tib] = Label.TIBIA
    labels[pcl] = Label.PCL
    native_acl = acl.copy()
    gap_region = np.zeros(shape, bool)
    unknown = np.zeros(shape, bool)
    inj = injury or INJURY_PRESETS["intact"]
    if inj.kind == "intact":
        labels[acl] = Label.ACL_INTACT
    elif inj.kind == "chronic_absent":
        gap_region = acl.copy()
    else:
        lo_t, hi_t = inj.t_center - inj.gap / 2, inj.t_center + inj.gap / 2
        in_gap = acl & (tpar >= lo_t) & (tpar <= hi_t)
        if inj.kind == "partial":
            # remove only part of the cross-section in the gap band
            Wc = W.reshape(*shape, 3)
            half = Wc[..., 0] > np.nanmedian(Wc[..., 0][in_gap]) if in_gap.any() else np.zeros(shape, bool)
            in_gap &= half
        gap_region = in_gap
        fem_side = acl & ~in_gap & (tpar < inj.t_center)
        tib_side = acl & ~in_gap & (tpar >= inj.t_center)
        if inj.kind == "partial":
            labels[acl & ~in_gap] = Label.ACL_INTACT  # continuous fibres remain
        else:
            labels[fem_side] = Label.ACL_REMNANT_FEMORAL
            labels[tib_side] = Label.ACL_REMNANT_TIBIAL
        unknown = acl & (np.abs(tpar - lo_t) < inj.unknown_margin) | acl & (np.abs(tpar - hi_t) < inj.unknown_margin)
        unknown &= ~fem & ~tib
    # image intensities (arbitrary units)
    img = np.full(shape, 0.38, np.float32)  # soft tissue
    body = np.zeros(shape, bool)
    Lb = Ltempl.reshape(*shape, 3)
    body |= ((Lb[..., 0] / (45 * s)) ** 2 + (Lb[..., 1] / (38 * s)) ** 2) <= 1.0
    img[~body] = 0.02
    for msk in (fem, tib):
        from scipy import ndimage

        er = ndimage.binary_erosion(msk, iterations=2)
        img[msk] = 0.12  # cortex
        img[er] = 0.78  # marrow
    img[labels == Label.PCL] = 0.12
    img[(labels == Label.ACL_INTACT) | (labels == Label.ACL_REMNANT_FEMORAL) | (labels == Label.ACL_REMNANT_TIBIAL)] = 0.18
    img[gap_region] = 0.85  # fluid / oedema in the tear gap
    zz = (W[:, 2].reshape(shape) - W[:, 2].min()) / max(np.ptp(W[:, 2]), 1)
    img *= (0.85 + 0.3 * zz).astype(np.float32)  # bias field
    img += rng.normal(0, noise, shape).astype(np.float32)
    # annotation: UNKNOWN band around tear ends
    ann = labels.copy()
    ann[unknown & (ann == 0)] = Label.UNKNOWN
    ann[unknown & np.isin(ann, [Label.ACL_REMNANT_FEMORAL, Label.ACL_REMNANT_TIBIAL])] = Label.UNKNOWN
    to_world = scan_T @ M  # template -> world
    # truth geometry in world
    F_world = to_world @ F
    cline_world = apply_affine(to_world, C)

    def _surface_exit(mask_bone, from_end):
        pts = C if from_end == 0 else C[::-1]
        vox = np.round(apply_affine(np.linalg.inv(aff), apply_affine(to_world, pts))).astype(int)
        vox = np.clip(vox, 0, np.array(shape) - 1)
        inside = mask_bone[vox[:, 0], vox[:, 1], vox[:, 2]]
        k = np.argmax(~inside) if (~inside).any() else 0
        return apply_affine(to_world, pts[k:k + 1])[0]

    truth = {
        "femoral_footprint_center": _surface_exit(fem, 0).tolist(),
        "tibial_footprint_center": _surface_exit(tib, 1).tolist(),
        "acl_centerline_world": cline_world.tolist(),
        "flexion_deg": flex_deg,
        "template_to_world_femur": to_world.tolist(),
        "template_to_world_tibia": F_world.tolist(),
        # independent landmarks (not used by registration)
        "landmarks": {
            "femur_lateral_epicondyle": apply_affine(to_world, (np.array([[26.0, -3, 18]]) * s))[0].tolist(),
            "femur_medial_epicondyle": apply_affine(to_world, (np.array([[-27.0, -3, 18]]) * s))[0].tolist(),
            "femur_notch_roof_apex": apply_affine(to_world, (np.array([[0.0, -2, 15]]) * s))[0].tolist(),
            "tibia_medial_eminence_tip": apply_affine(F_world, (np.array([[-2.5, -1, 1.0]]) * s))[0].tolist(),
            "tibia_tuberosity": apply_affine(F_world, (np.array([[0.0, 24, -26]]) * s))[0].tolist(),
            "tibia_lateral_plateau_rim": apply_affine(F_world, (np.array([[kp.plateau[0], 0, -2]]) * s))[0].tolist(),
        },
        "injury": dataclasses.asdict(inj),
    }
    phys = PhysicalStatus(True, "synthetic-phantom", ["SYNTHETIC: geometry is exact by construction"])
    vols = {
        "image": Volume(img, aff, phys, {"synthetic": True, "side": side}),
        "labels": Volume(ann, aff, phys, {"synthetic": True}),
        "native_acl": Volume(native_acl.astype(np.uint8), aff, phys, {"synthetic": True}),
    }
    return vols, truth


def make_bilateral_case(out_dir, seed: int = 0, injured_side: str = "right", injury: str = "midsubstance",
                        shape=(120, 110, 100), spacing=(0.7, 0.7, 1.0), flex_injured=12.0, flex_contra=4.0) -> Path:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    kp_inj = sample_subject(rng)
    contra_side = "left" if injured_side == "right" else "right"
    kp_con = contralateral_of(kp_inj, rng)
    T_inj = rigid_from_params(*np.radians(rng.normal(0, 4, 3)), t=rng.normal(0, 4, 3))
    T_con = rigid_from_params(*np.radians(rng.normal(0, 4, 3)), t=rng.normal(0, 4, 3))
    inj_vols, inj_truth = render_knee(injured_side, kp_inj, flex_injured, T_inj, shape, spacing, rng, INJURY_PRESETS[injury])
    con_vols, con_truth = render_knee(contra_side, kp_con, flex_contra, T_con, shape, spacing, rng, INJURY_PRESETS["intact"])
    subj = f"SYN{seed:04d}"
    case = {
        "case_id": f"{subj}_{injury}",
        "subject_id": subj,
        "synthetic": True,
        "evidence_level": EvidenceLevel.SYNTHETIC.value,
        "injured": {"side": injured_side, "exam_id": f"{subj}_{injured_side[0].upper()}_V0", "image": "injured/image.nii.gz",
                     "labels": "injured/labels.nii.gz", "landmarks": "injured/landmarks.mrk.json", "tear_location": injury},
        "contralateral": {"side": contra_side, "exam_id": f"{subj}_{contra_side[0].upper()}_V0", "image": "contralateral/image.nii.gz",
                          "labels": "contralateral/labels.nii.gz", "landmarks": "contralateral/landmarks.mrk.json",
                          "contralateral_confirmed_normal": None, "contralateral_pathology": "unknown (synthetic)"},
        "truth": "truth/truth.json",
        "notes": "SYNTHETIC PHANTOM. Exercises coordinates and geometry only; no clinical or anatomical accuracy.",
    }
    for role, vols, truth, side in (("injured", inj_vols, inj_truth, injured_side), ("contralateral", con_vols, con_truth, contra_side)):
        d = out / role
        d.mkdir(parents=True, exist_ok=True)
        save_nifti(vols["image"], d / "image.nii.gz", np.float32)
        save_nifti(vols["labels"], d / "labels.nii.gz", np.uint8)
        (d / "image.json").write_text(json.dumps({"synthetic": True, "side": side, "sequence": "synthetic-3D", "scanner": "none"}))
        present = sorted({int(v) for v in np.unique(vols["labels"].data) if v not in (0, 255)})
        rec = AnnotationRecord(
            annotation_id=f"{case[role]['exam_id']}_gt", exam_id=case[role]["exam_id"], image_ref=f"{role}/image.nii.gz",
            annotator="synthetic-generator", annotator_role="phantom", source="synthetic",
            annotated_labels=sorted(set(present) | {int(Label.FEMUR), int(Label.TIBIA), int(Label.PCL)} | ({int(Label.ACL_INTACT), int(Label.ACL_REMNANT_FEMORAL), int(Label.ACL_REMNANT_TIBIAL)})),
            review_status=ReviewStatus.UNREVIEWED,
            tear_location=injury if role == "injured" else "intact",
            notes="synthetic phantom ground truth",
        )
        rec.save(d / "labels.annotation.json")
        lms = dict(truth["landmarks"])
        if role == "contralateral" or injury in ("intact",):
            lms["acl_femoral_footprint_center"] = truth["femoral_footprint_center"]
            lms["acl_tibial_footprint_center"] = truth["tibial_footprint_center"]
        write_slicer_markups(lms, d / "landmarks.mrk.json", "synthetic")
    td = out / "truth"
    td.mkdir(parents=True, exist_ok=True)
    save_nifti(inj_vols["native_acl"], td / "injured_preinjury_acl.nii.gz", np.uint8)
    save_nifti(con_vols["native_acl"], td / "contralateral_acl.nii.gz", np.uint8)
    write_json(td / "truth.json", {"injured": inj_truth, "contralateral": con_truth, "params_injured": dataclasses.asdict(kp_inj),
                                   "params_contralateral": dataclasses.asdict(kp_con),
                                   "WARNING": "synthetic truth: hidden from estimation; used only for evaluation"})
    prov = Provenance(EvidenceLevel.SYNTHETIC, "bilacl.synthetic.phantom.make_bilateral_case", parameters={"seed": seed, "injury": injury, "shape": shape, "spacing": spacing})
    case["provenance"] = prov.to_dict()
    write_json(out / "case.json", case)
    return out


def write_dicom_series(vol: Volume, out_dir, series_desc="SYNTHETIC SAG", laterality: str | None = None, shuffle_seed: int | None = 0, uid_root="1.2.826.0.1.3680043.8.498."):
    """Write a Volume as a single-frame MR DICOM series (for exercising the DICOM path).
    File names are shuffled so that ordering must come from ImagePositionPatient."""
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    from ..imaging.coords import ras_affine_to_lps, voxel_spacing

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    A = ras_affine_to_lps(vol.affine)
    sp = voxel_spacing(A)
    row = A[:3, 0] / sp[0]
    col = A[:3, 1] / sp[1]
    data = vol.data.astype(np.float32)
    lo, hi = float(data.min()), float(data.max())
    slope = (hi - lo) / 4095 if hi > lo else 1.0
    study, series, frame = generate_uid(uid_root), generate_uid(uid_root), generate_uid(uid_root)
    n = data.shape[2]
    order = np.arange(n)
    if shuffle_seed is not None:
        np.random.default_rng(shuffle_seed).shuffle(order)
    for fi, k in enumerate(order):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
        meta.MediaStorageSOPInstanceUID = generate_uid(uid_root)
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.Modality = "MR"
        ds.PatientID = "SYNTHETIC"
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study, series, frame
        ds.SeriesDescription = series_desc
        ds.ImageType = ["DERIVED", "SECONDARY", "SYNTHETIC"]
        if laterality:
            ds.Laterality = laterality
        ds.InstanceNumber = int(fi + 1)  # deliberately NOT the spatial order
        ds.ImageOrientationPatient = [float(x) for x in (*row, *col)]
        ds.ImagePositionPatient = [float(x) for x in A[:3, 3] + A[:3, 2] * k]
        ds.PixelSpacing = [float(sp[1]), float(sp[0])]  # [row spacing, column spacing]
        ds.SliceThickness = float(sp[2])
        ds.Rows, ds.Columns = int(data.shape[1]), int(data.shape[0])
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
        ds.RescaleSlope, ds.RescaleIntercept = slope, lo
        pix = np.round((data[:, :, k].T - lo) / slope).astype(np.uint16)  # (i,j) -> (rows=j, cols=i)
        ds.PixelData = pix.tobytes()
        ds.save_as(str(out / f"IM{fi:04d}.dcm"), enforce_file_format=True)
    return out
