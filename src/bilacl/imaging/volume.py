"""Volume container and NIfTI / DICOM I/O that preserves physical coordinates.

A Volume holds voxel data plus a voxel->RAS+ mm affine and a `physical` status.
`physical.verified` is True only when spacing and orientation come from a
trustworthy header (DICOM geometry tags, or NIfTI with a non-zero qform/sform
code). Anything else (bare arrays, NIfTI with qform_code == sform_code == 0,
inconsistent DICOM geometry) is flagged unverified, which permits viewing and
learning tasks but blocks millimetre-based design export (see export_gate.py).
"""
from __future__ import annotations

import dataclasses
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..provenance import UNKNOWN
from .coords import axcodes, lps_affine_to_ras, voxel_spacing


@dataclasses.dataclass
class PhysicalStatus:
    verified: bool
    source: str  # e.g. "dicom-geometry", "nifti-sform", "nifti-qform", "array-no-header"
    reasons: list[str] = dataclasses.field(default_factory=list)  # why not verified / warnings


@dataclasses.dataclass
class Volume:
    data: np.ndarray
    affine: np.ndarray  # voxel (i,j,k) -> RAS+ mm
    physical: PhysicalStatus
    meta: dict = dataclasses.field(default_factory=dict)

    @property
    def spacing(self) -> np.ndarray:
        return voxel_spacing(self.affine)

    @property
    def axcodes(self) -> tuple:
        return axcodes(self.affine)

    def world_coords(self, ijk: np.ndarray) -> np.ndarray:
        ijk = np.asarray(ijk, float)
        return ijk @ self.affine[:3, :3].T + self.affine[:3, 3]

    def voxel_coords(self, xyz: np.ndarray) -> np.ndarray:
        inv = np.linalg.inv(self.affine)
        return np.asarray(xyz, float) @ inv[:3, :3].T + inv[:3, 3]

    def with_data(self, data: np.ndarray, **meta) -> "Volume":
        m = dict(self.meta)
        m.update(meta)
        return Volume(data, self.affine.copy(), dataclasses.replace(self.physical), m)

    def summary(self) -> dict:
        return {
            "shape": list(self.data.shape),
            "dtype": str(self.data.dtype),
            "spacing_mm": [round(float(s), 5) for s in self.spacing],
            "axcodes": "".join(self.axcodes),
            "physical_verified": self.physical.verified,
            "physical_source": self.physical.source,
            "physical_reasons": self.physical.reasons,
        }


# --------------------------------------------------------------------- NIfTI


def load_nifti(path: str | Path) -> Volume:
    import nibabel as nib

    img = nib.load(str(path))
    hdr = img.header
    qcode = int(hdr["qform_code"])
    scode = int(hdr["sform_code"])
    reasons: list[str] = []
    if scode > 0:
        aff, src, verified = img.get_sform(), "nifti-sform", True
    elif qcode > 0:
        aff, src, verified = img.get_qform(), "nifti-qform", True
    else:
        aff, src, verified = img.affine, "nifti-no-orientation", False
        reasons.append("qform_code and sform_code are both 0: orientation/origin not defined by header")
    sp = voxel_spacing(aff)
    if np.any(sp <= 0) or np.any(~np.isfinite(sp)):
        verified = False
        reasons.append(f"invalid voxel spacing {sp}")
    if qcode > 0 and scode > 0 and not np.allclose(img.get_qform(), img.get_sform(), atol=1e-3):
        reasons.append("qform and sform disagree; sform used")
    unit = hdr.get_xyzt_units()[0]
    if unit not in ("mm", "unknown"):
        verified = False
        reasons.append(f"spatial unit is {unit!r}, not mm")
    elif unit == "unknown":
        reasons.append("NIfTI xyz unit field unset; assumed mm per NIfTI convention")
    data = np.asanyarray(img.dataobj)
    meta = {"source_path": str(path), "source_format": "nifti", "source_spacing_mm": sp.tolist()}
    sidecar = Path(str(path).replace(".nii.gz", "").replace(".nii", "") + ".json")
    if sidecar.exists():
        try:
            meta["sidecar"] = json.loads(sidecar.read_text())
        except Exception:
            reasons.append("sidecar JSON unreadable")
    return Volume(data, np.asarray(aff, float), PhysicalStatus(verified, src, reasons), meta)


def save_nifti(vol: Volume, path: str | Path, dtype=None) -> None:
    import nibabel as nib

    data = vol.data if dtype is None else vol.data.astype(dtype)
    img = nib.Nifti1Image(data, vol.affine)
    code = 1 if vol.physical.verified else 0
    img.set_sform(vol.affine, code=code)
    img.set_qform(vol.affine, code=code)
    img.header.set_xyzt_units("mm")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(path))


def volume_from_array(data: np.ndarray, spacing=None, reason: str = "array without geometry header") -> Volume:
    """Wrap a bare array (e.g. MRNet-style .npy). Physical geometry is NOT verified."""
    sp = np.ones(3) if spacing is None else np.asarray(spacing, float)
    aff = np.diag([*sp, 1.0])
    return Volume(np.asarray(data), aff, PhysicalStatus(False, "array-no-header", [reason]), {"source_format": "array"})


# --------------------------------------------------------------------- DICOM


@dataclasses.dataclass
class SeriesReport:
    series_uid: str
    description: str
    modality: str
    n_files: int
    n_slices_used: int = 0
    is_localizer: bool = False
    issues: list[str] = dataclasses.field(default_factory=list)  # blocking problems
    warnings: list[str] = dataclasses.field(default_factory=list)
    laterality: str = UNKNOWN
    laterality_source: str = UNKNOWN
    meta: dict = dataclasses.field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return not self.issues and not self.is_localizer


def _read_headers(paths: list[Path]):
    import pydicom

    out = []
    for p in paths:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=False)
        except Exception:
            continue
        if "SOPClassUID" not in ds and "ImagePositionPatient" not in ds:
            continue
        out.append((p, ds))
    return out


def discover_dicom_series(root: str | Path) -> dict[str, list]:
    """Group readable DICOM files under root by SeriesInstanceUID."""
    root = Path(root)
    paths = [p for p in root.rglob("*") if p.is_file() and not p.name.startswith(".")]
    groups: dict[str, list] = defaultdict(list)
    for p, ds in _read_headers(paths):
        groups[str(getattr(ds, "SeriesInstanceUID", "no-series-uid"))].append((p, ds))
    return dict(groups)


def _laterality(ds_list) -> tuple[str, str]:
    for tag in ("Laterality", "ImageLaterality"):
        vals = {str(getattr(ds, tag, "")).strip().upper() for _, ds in ds_list} - {""}
        if len(vals) == 1:
            v = vals.pop()
            if v in ("L", "R"):
                return ("left" if v == "L" else "right"), f"dicom:{tag}"
        if len(vals) > 1:
            return UNKNOWN, f"conflicting {tag} values {sorted(vals)}"
    desc = " ".join(str(getattr(ds_list[0][1], t, "")) for t in ("SeriesDescription", "StudyDescription", "BodyPartExamined")).upper()
    hits = {s for s, keys in (("left", (" LT", "LEFT", " L KNEE")), ("right", (" RT", "RIGHT", " R KNEE"))) if any(k in " " + desc for k in keys)}
    if len(hits) == 1:
        # Text heuristics are never trusted silently: must be confirmed by a user.
        return hits.pop(), "description-text-heuristic (requires confirmation)"
    return UNKNOWN, "not recorded"


def validate_series(items) -> SeriesReport:
    ds0 = items[0][1]
    rep = SeriesReport(
        series_uid=str(getattr(ds0, "SeriesInstanceUID", "?")),
        description=str(getattr(ds0, "SeriesDescription", "")),
        modality=str(getattr(ds0, "Modality", UNKNOWN)),
        n_files=len(items),
    )
    itype = " ".join(str(x) for x in getattr(ds0, "ImageType", [])).upper()
    if "LOCALIZER" in itype or "SCOUT" in itype or "LOCALIZER" in rep.description.upper() or "SCOUT" in rep.description.upper():
        rep.is_localizer = True
    rep.laterality, rep.laterality_source = _laterality(items)
    rep.meta = {
        "manufacturer": str(getattr(ds0, "Manufacturer", UNKNOWN)) or UNKNOWN,
        "model": str(getattr(ds0, "ManufacturerModelName", UNKNOWN)) or UNKNOWN,
        "field_strength_T": float(ds0.MagneticFieldStrength) if "MagneticFieldStrength" in ds0 else UNKNOWN,
        "sequence_name": str(getattr(ds0, "SequenceName", UNKNOWN)) or UNKNOWN,
        "study_uid": str(getattr(ds0, "StudyInstanceUID", UNKNOWN)),
        "mr_acquisition_type": str(getattr(ds0, "MRAcquisitionType", UNKNOWN)) or UNKNOWN,
    }
    missing = [t for t in ("ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing") if t not in ds0]
    if missing:
        rep.issues.append(f"missing geometry tags {missing}")
        return rep
    iops = np.array([np.array(ds.ImageOrientationPatient, float) for _, ds in items])
    if not np.allclose(iops, iops[0], atol=1e-4):
        n_orient = len({tuple(np.round(i, 3)) for i in iops})
        rep.issues.append(f"{n_orient} different slice orientations in one series (multi-plane/localizer); not stackable")
        if n_orient >= 3 and len(items) <= 12:
            rep.is_localizer = True
        return rep
    shapes = {(int(ds.Rows), int(ds.Columns)) for _, ds in items}
    if len(shapes) > 1:
        rep.issues.append(f"inconsistent matrix sizes {sorted(shapes)}")
    ps = {tuple(np.round(np.array(ds.PixelSpacing, float), 5)) for _, ds in items}
    if len(ps) > 1:
        rep.issues.append(f"inconsistent PixelSpacing {sorted(ps)}")
    if len(items) < 3:
        rep.issues.append("fewer than 3 slices; not a 3D volume")
    row, col = iops[0][:3], iops[0][3:]
    normal = np.cross(row, col)
    pos = np.array([np.array(ds.ImagePositionPatient, float) for _, ds in items])
    proj = pos @ normal
    order = np.argsort(proj)
    sproj = proj[order]
    d = np.diff(sproj)
    dup = int(np.sum(np.abs(d) < 1e-3))
    if dup:
        rep.warnings.append(f"{dup} duplicate slice position(s) (repeat acquisitions / multi-echo?); duplicates dropped, first kept")
    keep = np.concatenate([[True], np.abs(d) >= 1e-3])
    sproj_u = sproj[keep]
    du = np.diff(sproj_u)
    if du.size:
        med = float(np.median(du))
        gaps = np.where(du > 1.5 * med)[0]
        if gaps.size:
            rep.issues.append(f"{gaps.size} gap(s) in slice positions (missing slices); median spacing {med:.3f} mm, max gap {du.max():.3f} mm")
        if np.max(np.abs(du - med)) > 0.01 * max(med, 1e-6) + 1e-3 and not gaps.size:
            rep.warnings.append(f"non-uniform slice spacing (range {du.min():.4f}-{du.max():.4f} mm)")
        st = getattr(ds0, "SliceThickness", None)
        if st is not None and abs(float(st) - med) > 0.05 and float(st) > 0:
            rep.warnings.append(f"SliceThickness {float(st):.3f} mm differs from position spacing {med:.3f} mm (gap or overlap)")
        rep.meta["slice_spacing_mm"] = med
    rep.meta["pixel_spacing_mm"] = [float(x) for x in ds0.PixelSpacing]
    rep.meta["orientation_lps"] = iops[0].tolist()
    rep.n_slices_used = int(keep.sum())
    rep.meta["_order"] = order[keep].tolist()
    return rep


def load_dicom_series(items, report: SeriesReport | None = None) -> Volume:
    """Load one validated series. Slices are ordered by projecting
    ImagePositionPatient onto the slice normal -- never by filename or
    InstanceNumber."""
    import pydicom

    rep = report or validate_series(items)
    if not rep.usable:
        raise ValueError(f"series {rep.series_uid} not loadable: issues={rep.issues} localizer={rep.is_localizer}")
    order = rep.meta["_order"]
    dss = [pydicom.dcmread(str(items[i][0])) for i in order]
    ds0 = dss[0]
    row = np.array(ds0.ImageOrientationPatient[:3], float)
    col = np.array(ds0.ImageOrientationPatient[3:], float)
    pr, pc = (float(x) for x in ds0.PixelSpacing)  # (between rows, between columns)
    slices = []
    for ds in dss:
        a = ds.pixel_array.astype(np.float32)
        a = a * float(getattr(ds, "RescaleSlope", 1) or 1) + float(getattr(ds, "RescaleIntercept", 0) or 0)
        slices.append(a.T)  # (rows, cols) -> (i=col index, j=row index)
    data = np.stack(slices, axis=-1)
    p0 = np.array(dss[0].ImagePositionPatient, float)
    if len(dss) > 1:
        pN = np.array(dss[-1].ImagePositionPatient, float)
        step = (pN - p0) / (len(dss) - 1)
    else:
        step = np.cross(row, col) * float(getattr(ds0, "SliceThickness", 1.0))
    aff_lps = np.eye(4)
    aff_lps[:3, 0] = row * pc  # i runs along the row direction (increasing column index)
    aff_lps[:3, 1] = col * pr  # j runs along the column direction (increasing row index)
    aff_lps[:3, 2] = step
    aff_lps[:3, 3] = p0
    aff = lps_affine_to_ras(aff_lps)
    meta = {k: v for k, v in rep.meta.items() if not k.startswith("_")}
    meta.update(
        source_format="dicom",
        series_uid=rep.series_uid,
        series_description=rep.description,
        laterality=rep.laterality,
        laterality_source=rep.laterality_source,
        source_spacing_mm=voxel_spacing(aff).tolist(),
        warnings=rep.warnings,
    )
    return Volume(data, aff, PhysicalStatus(True, "dicom-geometry", list(rep.warnings)), meta)


def select_series(groups: dict[str, list], prefer: str | None = None) -> list[tuple[SeriesReport, list]]:
    """Validate every series, return (report, items) sorted usable-first then by slice count.
    `prefer` is an optional case-insensitive substring of SeriesDescription."""
    reps = []
    for uid, items in groups.items():
        r = validate_series(items)
        reps.append((r, items))

    def key(x):
        r = x[0]
        return (not r.usable, not (prefer and prefer.lower() in r.description.lower()), -r.n_slices_used)

    return sorted(reps, key=key)


def load_any(path: str | Path, series_uid: str | None = None) -> Volume:
    p = Path(path)
    if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz")):
        return load_nifti(p)
    if p.is_file() and p.suffix.lower() in (".nrrd", ".nhdr", ".mha", ".mhd"):
        return load_sitk(p)
    groups = discover_dicom_series(p if p.is_dir() else p.parent)
    if not groups:
        raise FileNotFoundError(f"no NIfTI or DICOM found at {p}")
    ranked = select_series(groups)
    if series_uid:
        ranked = [x for x in ranked if x[0].series_uid == series_uid]
        if not ranked:
            raise KeyError(series_uid)
    rep, items = ranked[0]
    return load_dicom_series(items, rep)


def load_sitk(path: str | Path) -> Volume:
    """Load NRRD/MHA via SimpleITK (LPS) and convert to RAS."""
    import SimpleITK as sitk

    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    data = np.transpose(arr, (2, 1, 0)) if arr.ndim == 3 else np.transpose(arr, (2, 1, 0, 3))
    D = np.array(img.GetDirection(), float).reshape(3, 3)
    sp = np.array(img.GetSpacing(), float)
    aff_lps = np.eye(4)
    aff_lps[:3, :3] = D * sp
    aff_lps[:3, 3] = img.GetOrigin()
    meta: dict[str, Any] = {"source_path": str(path), "source_format": "sitk"}
    meta["sitk_metadata"] = {k: img.GetMetaData(k) for k in img.GetMetaDataKeys()}
    return Volume(data, lps_affine_to_ras(aff_lps), PhysicalStatus(True, "sitk-header", []), meta)


def to_sitk(vol: Volume):
    import SimpleITK as sitk

    from .coords import ras_affine_to_lps

    arr = np.transpose(vol.data, (2, 1, 0))
    img = sitk.GetImageFromArray(arr)
    A = ras_affine_to_lps(vol.affine)
    sp = voxel_spacing(A)
    img.SetSpacing(sp.tolist())
    img.SetOrigin(A[:3, 3].tolist())
    img.SetDirection((A[:3, :3] / sp).flatten().tolist())
    return img
