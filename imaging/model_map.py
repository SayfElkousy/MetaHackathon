"""Map KneeScan outputs back into the stored volume.

KneeScan (mri_analysis/) loads a scan with its own loader and reports outlines as [x, y] pixel coordinates on one
slice s of its (slice, row, col) array. Its internal preprocessing (normalisation, crop around the ACL box, resize to
1024 for MedSAM, 256 for the bone registration, possible left/right flip) is already inverted by KneeScan before it
returns outlines (see engine.acl / engine.femur_fibula). What is left is the loader's index convention, which differs
by format:

  NIfTI  KneeScan: vol = transpose(canonical, (0,2,1))[:, ::-1, ::-1]  ->  i = s, j = (ny-1) - c, k = (nz-1) - r
  DICOM  KneeScan: slices sorted by ImagePositionPatient . normal; pixel (r, c) of slice s is at
         IPP_s + c * PixelSpacing[1] * row_dir + r * PixelSpacing[0] * col_dir (LPS)  ->  RAS  ->  inverse affine
  .npy / image stacks: KneeScan uses the array as stored  ->  i = c, j = (rows-1) - r, k = s

The mapping is validated per run: every mapped pixel centre must land on a voxel centre of the stored grid (offset
reported) and outline points outside the image are counted. KneeScan's ACL outline is a 2D result on one slice; it is
stored and labelled as such, never as a 3D segmentation.
"""
import numpy as np
from matplotlib.path import Path

MODEL_NAME = "KneeScan"
MODEL_PARTS = {"acl": "MedSAM (SAM ViT-B) with an ACL mask decoder fine-tuned on 110 CEMRR expert outlines, box from the "
                      "SegFormer ACL localizer (hajirabaloch488/knee-mri-acl-localization)",
               "bones": "MedSAM base decoder, boxes from registration to 3 reference knees (experimental)"}
LABEL_VALUES = {"ACL": 1, "femur": 2, "tibia": 3, "fibula": 4}


class MappingError(ValueError):
    pass


def slice_geometry(v):
    """(rows, cols) of KneeScan's slices and a function (s, r, c) -> continuous canonical voxel coordinates."""
    m = v.get("model_index_map") or {}
    nx, ny, nz = v["shape"]
    kind = m.get("kind")
    if kind == "kneescan_nifti":
        return (nz, ny), lambda s, r, c: np.stack([np.full_like(r, s, float), (ny - 1) - c, (nz - 1) - r], -1), "KneeScan NIfTI index space"
    if kind == "stack":
        rows = m.get("rows", ny)
        return (rows, nx), lambda s, r, c: np.stack([c.astype(float), (rows - 1) - r, np.full_like(r, s, float)], -1), "array index space"
    if kind == "kneescan_dicom":
        if m.get("dedup_dropped"):
            raise MappingError("KneeScan reads duplicate DICOM images that this import dropped, so its slice numbers cannot "
                               "be matched to positions. Remove the duplicates from the series and re-run.")
        ipps = np.array(m["ipps"])
        row, col = np.array(m["row"]), np.array(m["col"])
        ps = m["pixel_spacing"]
        Ainv = np.linalg.inv(np.array(v["affine"]))
        flip = np.array([-1.0, -1.0, 1.0])
        # rows/cols of the source images: recover from the affine's in-plane extent
        rows, cols = (m["rows"], m["cols"]) if "rows" in m else _dicom_rc(v, row, col, ps)

        def f(s, r, c):
            lps = ipps[s] + c[..., None] * ps[1] * row + r[..., None] * ps[0] * col
            ras = lps * flip
            h = np.concatenate([ras, np.ones(ras.shape[:-1] + (1,))], -1)
            return (h @ Ainv.T)[..., :3]
        return (rows, cols), f, "KneeScan DICOM index space (via patient coordinates)"
    raise MappingError("Model outputs cannot be mapped for this input (%s)." % (m.get("why") or kind or "no index map"))


def _dicom_rc(v, row, col, ps):
    A = np.array(v["affine"])[:3, :3]
    t_row = np.abs(np.linalg.inv(A) @ (row * np.array([-1, -1, 1]) * ps[1]))
    t_col = np.abs(np.linalg.inv(A) @ (col * np.array([-1, -1, 1]) * ps[0]))
    cols = int(round(v["shape"][int(np.argmax(t_row))]))
    rows = int(round(v["shape"][int(np.argmax(t_col))]))
    return rows, cols


def rasterize(outline, rows, cols):
    """Pixels (r, c) whose centres lie inside the [x=c, y=r] polygon."""
    P = np.asarray(outline, float)
    if P.ndim != 2 or len(P) < 3:
        return np.zeros((0,), int), np.zeros((0,), int)
    x0, y0 = np.floor(P.min(0)).astype(int)
    x1, y1 = np.ceil(P.max(0)).astype(int)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, cols - 1), min(y1, rows - 1)
    if x1 < x0 or y1 < y0:
        return np.zeros((0,), int), np.zeros((0,), int)
    rr, cc = np.meshgrid(np.arange(y0, y1 + 1), np.arange(x0, x1 + 1), indexing="ij")
    inside = Path(P).contains_points(np.stack([cc.ravel(), rr.ravel()], 1), radius=1e-9).reshape(rr.shape)
    return rr[inside], cc[inside]


def map_point(v, s, xy):
    (rows, cols), f, _ = slice_geometry(v)
    p = f(int(s), np.array([float(xy[1])]), np.array([float(xy[0])]))[0]
    return [float(x) for x in p]


def outlines_to_mask(v, result):
    """Returns (uint8 mask on the volume grid, labels, mapping report, per-label slice info)."""
    (rows, cols), f, space = slice_geometry(v)
    shape = tuple(v["shape"])
    mask = np.zeros(shape, np.uint8)
    rep = {"source_space": space, "outside_points": 0, "round_trip_max_vox": 0.0, "labels": {}}
    items = []
    acl = result.get("acl") or {}
    if acl.get("outline"):
        items.append(("ACL", acl["slice"], acl["outline"]))
    ff = result.get("femur_fibula") or {}
    for nm in ("femur", "tibia", "fibula"):
        if ff.get(nm + "_outline"):
            items.append((nm, ff["slice"], ff[nm + "_outline"]))
    if not items:
        raise MappingError("The model returned no outlines to map (%s)." % (acl.get("error") or "no ACL outline"))
    S = _n_slices(v)
    for name, s, poly in items:
        if not 0 <= int(s) < S:
            raise MappingError("Model slice %s is outside the %d slices of this volume." % (s, S))
        P = np.asarray(poly, float)
        out = int(np.sum((P[:, 0] < -0.5) | (P[:, 0] > cols - 0.5) | (P[:, 1] < -0.5) | (P[:, 1] > rows - 0.5)))
        rep["outside_points"] += out
        rr, cc = rasterize(P, rows, cols)
        vox = f(int(s), rr.astype(float), cc.astype(float)) if len(rr) else np.zeros((0, 3))
        if len(vox):
            off = float(np.max(np.abs(vox - np.round(vox))))
            rep["round_trip_max_vox"] = max(rep["round_trip_max_vox"], off)
            ijk = np.round(vox).astype(int)
            ok = np.all((ijk >= 0) & (ijk < np.array(shape)), axis=1)
            rep["outside_points"] += int((~ok).sum())
            ijk = ijk[ok]
            mask[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = LABEL_VALUES[name]
        rep["labels"][name] = {"model_slice": int(s), "pixels": int(len(rr)), "outline_points": int(len(P))}
    rep["round_trip_max_vox"] = round(rep["round_trip_max_vox"], 4)
    if rep["round_trip_max_vox"] > 0.01:
        rep["note"] = ("Model pixels do not fall on voxel centres (oblique or resampled grid); nearest voxels were used, so "
                       "boundaries can shift by up to half a voxel.")
    labels = [{"value": LABEL_VALUES[n], "name": n, "source": "model", "dimensionality": "2D outline (one slice)",
               "experimental": n != "ACL"} for n in LABEL_VALUES if n in rep["labels"]]
    return mask, labels, rep


def _n_slices(v):
    m = v.get("model_index_map") or {}
    if m.get("kind") == "kneescan_dicom":
        return len(m["ipps"])
    if m.get("kind") == "kneescan_nifti":
        return v["shape"][0]
    return v["shape"][2]


def model_record(result, settings, mapping):
    acl = result.get("acl") or {}
    t = result.get("tear") or {}
    return {"name": MODEL_NAME, "parts": MODEL_PARTS, "settings": settings,
            "dimensionality": "2D: ACL outline on one sagittal slice (plus experimental bone outlines on one slice); "
                              "not a 3D segmentation",
            "confidence": acl.get("confidence"),
            "confidence_meaning": "MedSAM predicted IoU for the ACL outline (model self-estimate, not measured accuracy)",
            "shape_checks": acl.get("checks"), "tear": {k: t.get(k) for k in ("status", "probability", "threshold", "level", "level_reliability")},
            "validation_reported": result.get("validation"), "warnings": result.get("warnings"),
            "seconds": result.get("seconds"), "mapping": mapping}


def model_measurements(v, result):
    """KneeScan's own ACL length endpoints as an unreviewed distance annotation (model estimate)."""
    acl = result.get("acl") or {}
    if not (acl.get("end0") and acl.get("end1")):
        return []
    s = acl["slice"]
    p0, p1 = map_point(v, s, acl["end0"]), map_point(v, s, acl["end1"])
    return [{"type": "distance", "kind": "acl_length", "name": "ACL length (model outline extent)", "volume": v["id"],
             "points_voxel": [p0, p1], "status": "draft",
             "note": "From KneeScan's outline (long-axis extent on one sagittal slice). Model estimate: review before use. "
                     "KneeScan reported %s mm." % acl.get("length_mm")}]
