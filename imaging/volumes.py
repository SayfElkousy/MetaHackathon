"""Load and validate volumetric inputs.

Every loader returns one or more *series records* (plain dicts) holding:
    data            numpy array indexed [i, j, k]
    affine          4x4 voxel -> world (RAS+, millimetres) when orientation is known, else voxel-size diagonal
    orientation_known, spacing_known
    zooms           voxel size along i, j, k (mm, or 1.0 voxel units when uncalibrated)
    checks          validation results [{id, category, status, message, hint}]
    meta            de-identified acquisition metadata (no patient name / id / birth date is kept)
    frame_id        spatial frame identity (hashed DICOM FrameOfReferenceUID, or None when unknown)
    model_index_map how KneeScan's loader indexes the same file (used to map model outputs back)

Orientation handling is delegated to established libraries:
    DICOM  SimpleITK ImageSeriesReader (GDCM) for pixel data, spacing, origin and direction cosines
    NIfTI  nibabel (sform/qform), and nibabel.as_closest_canonical for every calibrated volume
Calibrated volumes are stored on the nearest canonical RAS+ grid (nibabel convention): +i toward the patient's
Right, +j toward Anterior, +k toward Superior. World coordinates are RAS+ millimetres.
"""
import hashlib
import io
import math
import os
import re
import tempfile
import zipfile

import numpy as np

# nibabel RAS+: +i toward patient Right, +j toward Anterior, +k toward Superior (for an axis-aligned canonical grid)
AXES = ("R", "A", "S")
STORED_DTYPES = ("uint8", "int16", "uint16", "float32")
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
THICK_RATIO = 2.5            # slice spacing / in-plane spacing above which reformatted planes are flagged
MAX_FILES = 4000


class ImportFailed(ValueError):
    """The input cannot be loaded as a usable volume. `checks` explains why."""

    def __init__(self, message, checks=None):
        super().__init__(message)
        self.checks = checks or [chk("load", "file", "fail", message)]


def chk(cid, category, status, message, hint=None, **extra):
    d = {"id": cid, "category": category, "status": status, "message": message}
    if hint:
        d["hint"] = hint
    d.update(extra)
    return d


def _h(uid):
    return hashlib.sha1(str(uid).encode()).hexdigest()[:16] if uid else None


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _store_dtype(a):
    """Choose a stored dtype that keeps the voxel values exactly where possible."""
    if a.dtype.kind in "iu":
        lo, hi = (int(a.min()), int(a.max())) if a.size else (0, 0)
        if 0 <= lo and hi <= 255 and a.dtype == np.uint8:
            return a.astype(np.uint8)
        if -32768 <= lo and hi <= 32767:
            return a.astype(np.int16)
        if 0 <= lo and hi <= 65535:
            return a.astype(np.uint16)
        return a.astype(np.float32)
    f = a.astype(np.float32)
    # float data that is integral and fits int16 (common after DICOM rescale with slope 1) stays exact as int16
    if f.size and np.all(np.isfinite(f)) and np.all(f == np.round(f)) and f.min() >= -32768 and f.max() <= 32767:
        return f.astype(np.int16)
    return f


def canonicalize(data, affine):
    """Reorient to the closest canonical RAS+ voxel grid with nibabel. Returns data, affine, original axcodes."""
    import nibabel as nib
    img = nib.Nifti1Image(np.asarray(data), np.asarray(affine, float))
    codes = "".join(nib.aff2axcodes(img.affine))
    can = nib.as_closest_canonical(img)
    return np.asarray(can.dataobj), np.array(can.affine, float), codes


def obliquity_deg(affine):
    """Largest angle between a voxel axis and its nearest world axis (0 for an axis-aligned grid)."""
    import nibabel.affines as na
    try:
        return float(np.degrees(np.max(np.abs(na.obliquity(affine)))))
    except Exception:
        R = affine[:3, :3] / np.linalg.norm(affine[:3, :3], axis=0)
        return float(np.degrees(np.arccos(np.clip(np.max(np.abs(R), axis=0).min(), -1, 1))))


def finish(rec):
    """Common post-processing: dtype, geometry summary, thick-slice and obliquity notices."""
    a = np.asarray(rec["data"])
    if a.dtype.kind == "f" and not np.all(np.isfinite(a)):
        rec["checks"].append(chk("finite", "file", "warn", "The volume contains NaN or infinite values; they are shown as black.",
                                 "Check the export settings of the source software."))
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    a = _store_dtype(a)
    rec["data"] = np.ascontiguousarray(a)
    rec["shape"] = [int(x) for x in a.shape]
    rec["dtype"] = str(a.dtype)
    aff = np.asarray(rec["affine"], float)
    rec["affine"] = aff.tolist()
    if rec["spacing_known"]:
        z = np.linalg.norm(aff[:3, :3], axis=0)
        rec["zooms"] = [round(float(x), 5) for x in z]
    else:
        rec["zooms"] = [1.0, 1.0, 1.0]
    rec["units"] = "mm" if rec["spacing_known"] else "voxel"
    rec.setdefault("native_axis", int(np.argmax(rec["zooms"])) if rec["spacing_known"] else 2)
    if rec["spacing_known"]:
        z = rec["zooms"]
        ratio = max(z) / max(1e-6, min(z))
        rec["anisotropy"] = round(ratio, 2)
        if ratio >= THICK_RATIO:
            rec["checks"].append(chk("thick_slices", "file", "warn",
                                     "Voxels are %.2f x %.2f x %.2f mm (ratio %.1f): planes reformatted across the %.1f mm axis "
                                     "show no detail finer than that spacing." % (z[0], z[1], z[2], ratio, max(z)),
                                     "Use the acquisition plane for fine detail; reformatted planes are shown with nearest-neighbour "
                                     "blocks, not interpolated."))
    else:
        rec["anisotropy"] = None
    if rec["orientation_known"]:
        ob = obliquity_deg(aff)
        rec["obliquity_deg"] = round(ob, 2)
        if ob > 1.0:
            rec["checks"].append(chk("oblique", "file", "warn",
                                     "The acquisition is oblique by %.1f deg: displayed planes follow the voxel grid nearest to "
                                     "axial / sagittal / coronal, not true anatomical planes." % ob,
                                     "Coordinates and measurements use the full affine and stay exact."))
    else:
        rec["obliquity_deg"] = None
    rec["checks"].append(chk("orientation", "file", "pass" if rec["orientation_known"] else "warn",
                             ("Orientation from %s: stored as RAS+ (source axes %s)." % (rec["fmt"], rec.get("source_axcodes", "?")))
                             if rec["orientation_known"] else
                             "No patient orientation in this file: anatomical directions (left/right, anterior/posterior, "
                             "superior/inferior) are not labelled.",
                             None if rec["orientation_known"] else "Import DICOM or NIfTI with an sform/qform to label orientation."))
    rec["checks"].append(chk("spacing", "file", "pass" if rec["spacing_known"] else "warn",
                             ("Voxel size %s mm." % " x ".join("%.3g" % v for v in rec["zooms"])) if rec["spacing_known"] else
                             "No physical voxel size: distances cannot be expressed in millimetres and measurement tools are "
                             "disabled.", None if rec["spacing_known"] else "DICOM or NIfTI files carry pixel spacing."))
    rng = (float(a.min()), float(a.max())) if a.size else (0.0, 0.0)
    rec["range"] = [rng[0], rng[1]]
    if a.size:
        sample = a.ravel()[:: max(1, a.size // 400000)].astype(np.float64)
        lo, hi = np.percentile(sample, [1.0, 99.5])
        rec["auto_window"] = {"center": float((lo + hi) / 2), "width": float(max(hi - lo, 1e-6)), "method": "1st-99.5th percentile"}
    rec.setdefault("initial_window", None)
    return rec


def _base(fmt, name):
    return {"fmt": fmt, "name": name, "checks": [], "meta": {}, "frame_id": None, "model_index_map": None,
            "orientation_known": False, "spacing_known": False, "missing_slices": []}


# --------------------------------------------------------------------------- NIfTI
def load_nifti(path, name=None):
    import nibabel as nib
    rec = _base("nifti", name or os.path.basename(path))
    try:
        img = nib.load(path)
    except Exception as e:
        raise ImportFailed("Not a readable NIfTI file (%s)." % e)
    hdr = img.header
    data = np.asanyarray(img.dataobj)
    if data.ndim == 4:
        rec["checks"].append(chk("4d", "file", "warn", "4D NIfTI with %d volumes: the first volume is shown." % data.shape[3],
                                 "Split the series to view the other volumes."))
        data = data[..., 0]
    if data.ndim != 3:
        raise ImportFailed("Expected a 3D volume, found %d dimensions %s." % (data.ndim, tuple(data.shape)))
    sc, qc = int(hdr["sform_code"]), int(hdr["qform_code"])
    unit = hdr.get_xyzt_units()[0]
    scale = {"meter": 1000.0, "mm": 1.0, "micron": 0.001}.get(unit, 1.0)
    if unit == "unknown":
        rec["checks"].append(chk("units", "file", "warn", "The NIfTI header does not state spatial units; millimetres are assumed.",
                                 "Most MRI converters write millimetres; confirm with the source."))
    if sc == 0 and qc == 0:
        z = [float(v) * scale for v in hdr.get_zooms()[:3]]
        rec.update(data=data, affine=np.diag(z + [1.0]), orientation_known=False, spacing_known=all(v > 0 for v in z))
        rec["checks"].append(chk("nifti_codes", "file", "warn", "sform_code and qform_code are both 0: voxel size is known but "
                                 "patient orientation is not.", "Re-export with a valid sform or qform."))
        rec["model_index_map"] = {"kind": "nifti_noorient"}
    else:
        aff = np.array(img.affine, float)
        aff[:3, :] *= scale
        d2, a2, codes = canonicalize(data, aff)
        rec.update(data=d2, affine=a2, orientation_known=True, spacing_known=True, source_axcodes=codes)
        rec["source_paths"] = [path]
        rec["meta"]["affine_source"] = "sform (code %d)" % sc if sc else "qform (code %d)" % qc
        if sc in (3, 4):
            rec["frame_id"] = "template:%s" % ("talairach" if sc == 3 else "mni")
        # KneeScan's loader (mri_analysis/kneescan/io.py) uses nib.as_closest_canonical too: identical grid
        rec["model_index_map"] = {"kind": "kneescan_nifti", "shape": [int(x) for x in d2.shape]}
    desc = hdr.get("descrip")
    if desc is not None:
        try:
            s = desc.tobytes().decode("latin-1").strip("\x00 ").strip() if hasattr(desc, "tobytes") else str(desc)
            if s:
                rec["meta"]["description"] = s[:80]
        except Exception:
            pass
    return [finish(rec)]


# --------------------------------------------------------------------------- DICOM
META_TAGS = ("Modality", "SeriesDescription", "ProtocolName", "SequenceName", "ScanningSequence", "MRAcquisitionType",
             "SliceThickness", "SpacingBetweenSlices", "RepetitionTime", "EchoTime", "FlipAngle", "MagneticFieldStrength",
             "Manufacturer", "StudyDescription", "BodyPartExamined", "Laterality", "ImageLaterality", "SeriesNumber")


def _f(v, n=None):
    try:
        if n is None:
            return float(v)
        return [float(x) for x in v][:n]
    except Exception:
        return None


def load_dicom_paths(paths):
    """Group DICOM files into series and load each series. Returns (records, file_checks)."""
    import pydicom
    hdrs, skipped = [], 0
    for p in paths[:MAX_FILES]:
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
        except Exception:
            skipped += 1
            continue
        if "Rows" not in ds or "Columns" not in ds:
            skipped += 1
            continue
        hdrs.append((p, ds))
    notes = []
    if skipped:
        notes.append(chk("non_image", "file", "warn", "%d file(s) are not DICOM images (no pixel matrix) and were skipped." % skipped))
    if len(paths) > MAX_FILES:
        notes.append(chk("too_many", "file", "warn", "Only the first %d files were read." % MAX_FILES))
    if not hdrs:
        raise ImportFailed("No DICOM images found.", notes + [chk("dicom", "file", "fail", "No DICOM images with a pixel matrix were found.",
                                                                     "Select the folder or .zip that contains the .dcm files.")])
    series = {}
    for p, ds in hdrs:
        series.setdefault(str(getattr(ds, "SeriesInstanceUID", "unknown")), []).append((p, ds))
    out = []
    for uid, items in series.items():
        try:
            out.append(_dicom_series(uid, items, notes if not out else []))
        except ImportFailed as e:
            out.append({"failed": True, "name": _series_name(items[0][1]), "checks": e.checks, "fmt": "dicom"})
    grp = chk("grouping", "file", "pass", "%d image(s) grouped into %d series by SeriesInstanceUID." % (len(hdrs), len(series)))
    for r in out:
        r["checks"].insert(0, dict(grp))
    return out


def _series_name(ds):
    d = str(getattr(ds, "SeriesDescription", "") or getattr(ds, "ProtocolName", "") or "DICOM series")
    n = getattr(ds, "SeriesNumber", None)
    return ("%s (series %s)" % (d, n)) if n is not None else d


def _dicom_series(uid, items, notes):
    import pydicom  # noqa: F401
    ds0 = items[0][1]
    rec = _base("dicom", _series_name(ds0))
    rec["checks"].extend(notes)
    nf = int(getattr(ds0, "NumberOfFrames", 1) or 1)
    if len(items) == 1 and nf > 1:
        return _dicom_multiframe(rec, items[0][0], ds0)
    # one sequence type per series is required; drop exact duplicate instances
    seen, uniq, dups = set(), [], 0
    for p, ds in items:
        sop = str(getattr(ds, "SOPInstanceUID", p))
        if sop in seen:
            dups += 1
            continue
        seen.add(sop)
        uniq.append((p, ds))
    if dups:
        rec["checks"].append(chk("duplicates", "file", "warn", "%d duplicate image(s) (same SOPInstanceUID) were ignored." % dups))
    items = uniq
    shapes = {(int(ds.Rows), int(ds.Columns)) for _, ds in items}
    if len(shapes) > 1:
        raise ImportFailed("Images in one series have different sizes.", [chk("dims", "file", "fail",
                           "Images in this series have different matrix sizes %s." % sorted(shapes),
                           "The series mixes acquisitions; export each acquisition as its own series.")])
    iops = [_f(getattr(ds, "ImageOrientationPatient", None), 6) for _, ds in items]
    ipps = [_f(getattr(ds, "ImagePositionPatient", None), 3) for _, ds in items]
    pss = [_f(getattr(ds, "PixelSpacing", None), 2) for _, ds in items]
    if any(v is None for v in iops) or any(v is None for v in ipps):
        return _dicom_unpositioned(rec, items)
    # orientation consistency: keep the majority orientation (e.g. drop localizers), fail if no majority
    keyed = {}
    for (p, ds), o in zip(items, iops):
        keyed.setdefault(tuple(round(x, 3) for x in o), []).append((p, ds))
    if len(keyed) > 1:
        major = max(keyed.values(), key=len)
        if len(major) * 2 <= len(items):
            raise ImportFailed("Mixed slice orientations.", [chk("orientation_mixed", "file", "fail",
                               "The series contains %d different slice orientations with no majority." % len(keyed),
                               "Export one orientation per series.")])
        rec["checks"].append(chk("orientation_mixed", "file", "warn",
                                 "%d image(s) with a different slice orientation (e.g. localizers) were excluded."
                                 % (len(items) - len(major))))
        items = major
        iops = [_f(ds.ImageOrientationPatient, 6) for _, ds in items]
        ipps = [_f(ds.ImagePositionPatient, 3) for _, ds in items]
        pss = [_f(getattr(ds, "PixelSpacing", None), 2) for _, ds in items]
    if any(v is None for v in pss) or len({tuple(round(x, 4) for x in v) for v in pss}) > 1:
        raise ImportFailed("Pixel spacing missing or inconsistent.", [chk("spacing", "file", "fail",
                           "PixelSpacing is missing or differs between images of this series.",
                           "Re-export the series from the scanner or PACS.")])
    o = np.array(iops[0])
    row, col = o[:3], o[3:]
    normal = np.cross(row, col)
    pos = np.array([float(np.dot(np.array(p), normal)) for p in ipps])
    order = np.argsort(pos, kind="stable")
    items = [items[i] for i in order]
    pos = pos[order]
    ipps = [ipps[i] for i in order]
    # same position, different instance -> multiple echoes / phases mixed: refuse
    if len(pos) > 1 and np.min(np.diff(pos)) < 1e-3:
        k = int(np.sum(np.diff(pos) < 1e-3))
        raise ImportFailed("Several images share one slice position.", [chk("same_position", "file", "fail",
                           "%d image(s) share a slice position with another image (multiple echoes, phases or b-values in one "
                           "series)." % k, "Split the series by echo/phase before importing.")])
    if len(pos) < 2:
        raise ImportFailed("Only one slice.", [chk("slices", "file", "fail", "This series has a single image; a volume needs "
                                                   "at least 2 slices.", "Import it as a 2D image instead.")])
    d = np.diff(pos)
    step = float(np.median(d))
    gaps = []
    for i, di in enumerate(d):
        n_missing = int(round(di / step)) - 1
        if n_missing >= 1 and abs(di - (n_missing + 1) * step) < 0.25 * step:
            gaps.append((i, n_missing))
        elif abs(di - step) > 0.1 * step:
            raise ImportFailed("Irregular slice spacing.", [chk("slice_spacing", "file", "fail",
                               "Slice spacing is irregular (%.3g mm vs median %.3g mm between slices %d and %d)." % (di, step, i + 1, i + 2),
                               "The series may combine stacks; export each stack separately.")])
    inst = [int(getattr(ds, "InstanceNumber", 0) or 0) for _, ds in items]
    if len(set(inst)) == len(inst) and inst != sorted(inst) and inst != sorted(inst, reverse=True):
        rec["checks"].append(chk("instance_order", "file", "warn", "InstanceNumber order disagrees with slice position; "
                                 "slices are ordered by ImagePositionPatient."))
    else:
        rec["checks"].append(chk("slice_order", "file", "pass", "%d slices ordered by ImagePositionPatient along the slice "
                                 "normal (spacing %.3g mm)." % (len(items), step)))
    th = _f(getattr(ds0, "SliceThickness", None))
    if th and th > step * 1.05:
        rec["checks"].append(chk("overlap", "file", "pass", "Slices overlap (thickness %.3g mm > spacing %.3g mm)." % (th, step)))
    elif th and th < step * 0.95:
        rec["checks"].append(chk("slice_gap", "file", "warn", "Slice thickness %.3g mm is smaller than the spacing %.3g mm: "
                                 "tissue between slices is not imaged." % (th, step)))
    files = [p for p, _ in items]
    rec["source_paths"] = files
    if not gaps:
        import SimpleITK as sitk
        r = sitk.ImageSeriesReader()
        r.SetFileNames(files)
        img = r.Execute()
        arr = sitk.GetArrayFromImage(img)                     # (z, y, x)
        data = np.transpose(arr, (2, 1, 0))
        sp, org, dr = np.array(img.GetSpacing()), np.array(img.GetOrigin()), np.array(img.GetDirection()).reshape(3, 3)
        aff_lps = np.eye(4)
        aff_lps[:3, :3] = dr * sp
        aff_lps[:3, 3] = org
        rec["checks"].append(chk("slices_complete", "file", "pass", "No missing or duplicate slice positions detected."))
        rec["meta"]["reader"] = "SimpleITK %s ImageSeriesReader (GDCM)" % sitk.Version_VersionString()
    else:
        # missing slices: insert blank planes at the missing positions so the geometry after the gap stays correct
        import pydicom
        planes, missing = [], []
        for i, (p, _) in enumerate(items):
            ds = pydicom.dcmread(p, force=True)
            a = ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1) or 1) + float(getattr(ds, "RescaleIntercept", 0) or 0)
            planes.append(a)
            for gi, n in gaps:
                if gi == i:
                    for _ in range(n):
                        missing.append(len(planes))
                        planes.append(np.zeros_like(a))
        arr = np.stack(planes)                                 # (z, rows, cols)
        data = np.transpose(arr, (2, 1, 0))                    # (cols, rows, z)
        ps = pss[0]
        aff_lps = np.eye(4)
        aff_lps[:3, 0] = row * ps[1]
        aff_lps[:3, 1] = col * ps[0]
        aff_lps[:3, 2] = normal * step
        aff_lps[:3, 3] = ipps[0]
        nmiss = sum(n for _, n in gaps)
        rec["checks"].append(chk("missing_slices", "file", "warn",
                                 "%d slice(s) missing from the series (gap after slice %s). Blank planes were inserted at the "
                                 "missing positions so positions after the gap stay correct." % (nmiss, ", ".join(str(g + 1) for g, _ in gaps)),
                                 "Re-send the complete series; blank planes are marked in the viewer."))
        rec["missing_slices_source"] = missing
        rec["meta"]["reader"] = "pydicom (gap filling); affine from ImageOrientation/PositionPatient"
    aff_ras = np.diag([-1.0, -1.0, 1.0, 1.0]) @ aff_lps          # DICOM LPS -> RAS
    d2, a2, codes = canonicalize(data, aff_ras)
    rec.update(data=d2, affine=a2, orientation_known=True, spacing_known=True, source_axcodes=codes)
    if gaps:
        rec["missing_slices"] = _missing_in_canonical(aff_ras, a2, data.shape, rec.pop("missing_slices_source"))
    for t in META_TAGS:
        v = getattr(ds0, t, None)
        if v not in (None, ""):
            rec["meta"][t] = v if isinstance(v, (int, float, str)) else str(v)
    rec["meta"]["study_key"] = _h(getattr(ds0, "StudyInstanceUID", None))
    rec["meta"]["series_key"] = _h(uid)
    rec["frame_id"] = _h(getattr(ds0, "FrameOfReferenceUID", None))
    if not rec["frame_id"]:
        rec["checks"].append(chk("frame", "file", "warn", "No FrameOfReferenceUID: this series cannot be linked to other series."))
    wc, ww = getattr(ds0, "WindowCenter", None), getattr(ds0, "WindowWidth", None)
    try:
        wc = float(wc[0] if hasattr(wc, "__len__") and not isinstance(wc, str) else wc)
        ww = float(ww[0] if hasattr(ww, "__len__") and not isinstance(ww, str) else ww)
        rec["initial_window"] = {"center": wc, "width": ww, "method": "DICOM WindowCenter/WindowWidth"}
    except Exception:
        pass
    # KneeScan (mri_analysis/kneescan/io.py) sorts by IPP . normal (no dedup) and stacks pixel arrays as (slice, row, col)
    rec["model_index_map"] = {"kind": "kneescan_dicom", "ipps": [list(map(float, p)) for p in ipps], "row": row.tolist(),
                              "col": col.tolist(), "pixel_spacing": list(pss[0]), "dedup_dropped": dups,
                              "rows": int(ds0.Rows), "cols": int(ds0.Columns),
                              "gaps": bool(gaps)}
    return finish(rec)


def _missing_in_canonical(aff_src, aff_can, shape_src, missing_k):
    """Map missing source slice indices (k in the source grid) to (axis, index) in the canonical grid."""
    if not missing_k:
        return []
    ijk = np.array([[shape_src[0] // 2, shape_src[1] // 2, k, 1.0] for k in missing_k]).T
    world = aff_src @ ijk
    can = np.linalg.inv(aff_can) @ world
    # the source k axis maps to the canonical axis with the largest response
    t = np.linalg.inv(aff_can[:3, :3]) @ aff_src[:3, 2]
    ax = int(np.argmax(np.abs(t)))
    return [{"axis": ax, "index": int(round(c))} for c in can[ax]]


def _dicom_multiframe(rec, path, ds0):
    import SimpleITK as sitk
    try:
        img = sitk.ReadImage(path)
    except Exception as e:
        raise ImportFailed("Unreadable multi-frame DICOM (%s)." % e)
    arr = sitk.GetArrayFromImage(img)
    if arr.ndim != 3:
        raise ImportFailed("Multi-frame DICOM is not a 3D volume.")
    data = np.transpose(arr, (2, 1, 0))
    sp, org, dr = np.array(img.GetSpacing()), np.array(img.GetOrigin()), np.array(img.GetDirection()).reshape(3, 3)
    aff = np.eye(4)
    aff[:3, :3] = dr * sp
    aff[:3, 3] = org
    aff = np.diag([-1.0, -1.0, 1.0, 1.0]) @ aff
    d2, a2, codes = canonicalize(data, aff)
    rec.update(data=d2, affine=a2, orientation_known=True, spacing_known=True, source_axcodes=codes)
    rec["checks"].append(chk("multiframe", "file", "warn", "Multi-frame DICOM: per-frame positions are taken from SimpleITK; "
                             "duplicate and missing frames cannot be checked."))
    rec["frame_id"] = _h(getattr(ds0, "FrameOfReferenceUID", None))
    rec["model_index_map"] = {"kind": "unsupported", "why": "KneeScan reads multi-frame DICOM without positions"}
    return finish(rec)


def _dicom_unpositioned(rec, items):
    import pydicom
    items = sorted(items, key=lambda it: int(getattr(it[1], "InstanceNumber", 0) or 0))
    planes = [pydicom.dcmread(p, force=True).pixel_array.astype(np.float32) for p, _ in items]
    arr = np.stack(planes)                                     # (k, rows, cols)
    data = np.transpose(arr, (2, 1, 0))[:, ::-1, :]            # i = col, j = flipped row (upright), k = instance order
    rec.update(data=data, affine=np.eye(4), orientation_known=False, spacing_known=False)
    rec["checks"].append(chk("positions", "file", "warn", "Images have no ImagePositionPatient/ImageOrientationPatient: "
                             "ordered by InstanceNumber; no orientation or physical spacing.",
                             "Missing slices cannot be detected without positions."))
    rec["model_index_map"] = {"kind": "stack", "rows": int(arr.shape[1])}
    return finish(rec)


# --------------------------------------------------------------------------- 2D stacks and arrays
def load_image_stack(paths, names):
    from PIL import Image
    pairs = sorted(zip(names, paths), key=lambda t: natural_key(t[0]))
    planes = []
    for n, p in pairs:
        im = Image.open(p)
        if im.mode not in ("L", "I;16", "I", "F"):
            im = im.convert("L")
        planes.append(np.array(im))
    if len({a.shape for a in planes}) > 1:
        raise ImportFailed("Images differ in size.", [chk("dims", "file", "fail", "The images do not all have the same size "
                                                          "(%s)." % sorted({a.shape for a in planes}), "Select images from one stack.")])
    arr = np.stack(planes)                                       # (k, rows, cols)
    rec = _base("image-stack", "%s … (%d images)" % (pairs[0][0], len(pairs)) if len(pairs) > 1 else pairs[0][0])
    rec.update(data=np.transpose(arr, (2, 1, 0))[:, ::-1, :], affine=np.eye(4), orientation_known=False, spacing_known=False)
    rec["checks"].append(chk("stack_order", "file", "pass" if len(pairs) > 1 else "warn",
                             "Ordered by file name (natural sort): %s … %s." % (pairs[0][0], pairs[-1][0]) if len(pairs) > 1 else
                             "A single 2D image: viewed as one slice; no reformatting.",
                             "File names define slice order; check it matches acquisition order."))
    rec["model_index_map"] = {"kind": "stack", "rows": int(arr.shape[1])}
    rec["is_2d"] = len(pairs) == 1
    return finish(rec)


def load_npy(path, name):
    a = np.load(path, allow_pickle=False)
    if a.ndim != 3:
        raise ImportFailed("Expected a 3D array (slices, rows, cols); got shape %s." % (a.shape,))
    rec = _base("npy", name)
    rec.update(data=np.transpose(a, (2, 1, 0))[:, ::-1, :], affine=np.eye(4), orientation_known=False, spacing_known=False)
    rec["model_index_map"] = {"kind": "stack", "rows": int(a.shape[1])}
    return finish(rec)


# --------------------------------------------------------------------------- entry point
def load_files(files):
    """files: list of (path, original_name). Returns list of series records (possibly with failed entries)."""
    dicoms, images, out = [], [], []
    tmp = None
    for path, name in files:
        low = name.lower()
        if low.endswith(".zip"):
            tmp = tmp or tempfile.mkdtemp(prefix="img_zip_")
            with zipfile.ZipFile(path) as z:
                for k, info in enumerate(z.infolist()[:MAX_FILES]):
                    if info.is_dir() or info.file_size > 200_000_000:
                        continue
                    base = os.path.basename(info.filename)
                    if base.startswith(".") or "__MACOSX" in info.filename:
                        continue
                    dst = os.path.join(tmp, "%05d_%s" % (k, re.sub(r"[^A-Za-z0-9_.-]", "_", base)[:60]))
                    with open(dst, "wb") as f:
                        f.write(z.read(info))
                    if base.lower().endswith(IMAGE_EXT):
                        images.append((dst, base))
                    elif base.lower().endswith((".nii", ".nii.gz")):
                        out.extend(load_nifti(dst, base))
                    else:
                        dicoms.append(dst)
        elif low.endswith((".nii", ".nii.gz")):
            out.extend(load_nifti(path, name))
        elif low.endswith(".npy"):
            out.append(load_npy(path, name))
        elif low.endswith(IMAGE_EXT):
            images.append((path, name))
        else:
            dicoms.append(path)                                   # .dcm or extension-less DICOM
    if dicoms:
        out.extend(load_dicom_paths(dicoms))
    if images:
        out.append(load_image_stack([p for p, _ in images], [n for _, n in images]))
    if not out:
        raise ImportFailed("Nothing to import.", [chk("empty", "file", "fail", "No supported files were selected.",
                                                      "Supported: DICOM (.dcm or .zip), NIfTI (.nii/.nii.gz), .npy, PNG/JPEG/TIFF stacks.")])
    return out


# --------------------------------------------------------------------------- masks
def align_mask(mask_rec, image, allow_resample=False):
    """Validate a mask against its image and return (uint8 array on the image grid, checks)."""
    checks = []
    m = np.asarray(mask_rec["data"])
    if m.dtype.kind == "f" and not np.all(m == np.round(m)):
        raise ImportFailed("Mask values are not integers.", [chk("mask_values", "alignment", "fail",
                           "The mask has non-integer values (range %.3g to %.3g): it looks like a probability or quantitative map."
                           % (float(m.min()), float(m.max())), "Import it as a derived map, or threshold it first.")])
    if m.min() < 0 or m.max() > 255:
        raise ImportFailed("Mask label values out of range.", [chk("mask_values", "alignment", "fail",
                           "Label values must be 0-255 (found %d to %d)." % (int(m.min()), int(m.max())))])
    ishape, ia = tuple(image["shape"]), np.array(image["affine"])
    ma = np.array(mask_rec["affine"])
    if mask_rec["orientation_known"] and image["orientation_known"]:
        same_grid = m.shape == ishape and np.allclose(ma, ia, atol=1e-3)
        if same_grid:
            checks.append(chk("grid", "alignment", "pass", "Mask grid matches the image (dimensions %s, affine within 0.001 mm)."
                              % (list(ishape),)))
        elif allow_resample:
            import nibabel as nib
            from nibabel.processing import resample_from_to
            src = nib.Nifti1Image(m.astype(np.int16), ma)
            out = resample_from_to(src, (ishape, ia), order=0, mode="constant", cval=0)
            m = np.asarray(out.dataobj)
            checks.append(chk("grid", "alignment", "warn", "Mask grid differed from the image (%s vs %s) and was resampled to the "
                              "image grid with nearest-neighbour interpolation." % (list(mask_rec["data"].shape), list(ishape)),
                              "Boundaries can shift by up to half a voxel."))
        else:
            off = float(np.max(np.abs(ma - ia))) if ma.shape == ia.shape else float("nan")
            raise ImportFailed("Mask grid does not match the image.", [chk("grid", "alignment", "fail",
                               "Mask dimensions %s / affine differ from the image %s (largest affine difference %.3g)."
                               % (list(m.shape), list(ishape), off),
                               "Re-export the mask on the image grid, or import again with 'resample to image grid'.")])
    else:
        if m.shape != ishape:
            raise ImportFailed("Mask dimensions differ.", [chk("grid", "alignment", "fail",
                               "Mask dimensions %s differ from the image %s, and without orientation metadata they cannot be "
                               "aligned." % (list(m.shape), list(ishape)))])
        checks.append(chk("grid", "alignment", "warn", "Mask and image have matching dimensions but no orientation metadata: "
                          "alignment is assumed, not verified."))
    return np.ascontiguousarray(m.astype(np.uint8)), checks
