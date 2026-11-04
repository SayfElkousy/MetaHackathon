"""Load a sagittal knee MRI from the formats people actually have, and return it as (slices, rows, cols) with spacing.

Supported:
  .zip of DICOM files (or a folder) - pixel spacing and slice spacing read from the headers (real millimetres)
  .dcm                              - single multi-frame DICOM
  .nii / .nii.gz                    - NIfTI; reoriented so each slice is sagittal, rows run superior->inferior
  .npy                              - MRNet-style array (slices, rows, cols); no spacing
  .pck                              - KneeMRI (Rijeka) pickle, loaded with a restricted unpickler; no spacing
Images are returned in the orientation they were acquired / stored; the models handle either knee side."""
import io, os, zipfile, tempfile, numpy as np
from . import safe_pck


class LoadError(ValueError):
    pass


def _from_dicoms(datasets):
    import pydicom  # noqa: F401
    ds = [d for d in datasets if hasattr(d, 'pixel_array') or 'PixelData' in d]
    if not ds: raise LoadError('No DICOM images with pixel data found.')
    series = {}
    for d in ds: series.setdefault(getattr(d, 'SeriesInstanceUID', 'x'), []).append(d)
    # prefer the sagittal series with the most slices
    def normal(d):
        o = np.array(getattr(d, 'ImageOrientationPatient', [0, 1, 0, 0, 0, -1]), float)
        return np.cross(o[:3], o[3:])
    cands = []
    for uid, s in series.items():
        n = normal(s[0]); sag = abs(n[0]) >= max(abs(n[1]), abs(n[2]))
        cands.append((sag, len(s), uid))
    cands.sort(reverse=True); sag, _, uid = cands[0]; s = series[uid]
    notes = [] if sag else ['The largest DICOM series is not sagittal; results assume sagittal slices.']
    if len(s) == 1 and int(getattr(s[0], 'NumberOfFrames', 1)) > 1:
        vol = s[0].pixel_array.astype(np.float32); spacing = None
        try:
            fg = s[0].SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0]
            ps = [float(x) for x in fg.PixelSpacing]; st = float(getattr(fg, 'SpacingBetweenSlices', getattr(fg, 'SliceThickness', 0)) or 0)
            spacing = (st or None, ps[0], ps[1])
        except Exception:
            notes.append('Multi-frame DICOM without pixel spacing; millimetres will be estimated.')
        return vol, spacing, notes
    n = normal(s[0])
    pos = [float(np.dot(np.array(getattr(d, 'ImagePositionPatient', [0, 0, i]), float), n)) for i, d in enumerate(s)]
    order = np.argsort(pos); s = [s[i] for i in order]; pos = np.array(pos)[order]
    shapes = {d.pixel_array.shape for d in s}
    if len(shapes) != 1: raise LoadError(f'DICOM slices have different sizes: {shapes}')
    vol = np.stack([d.pixel_array.astype(np.float32) * float(getattr(d, 'RescaleSlope', 1) or 1) + float(getattr(d, 'RescaleIntercept', 0) or 0) for d in s])
    ps = getattr(s[0], 'PixelSpacing', None)
    dz = float(np.median(np.diff(pos))) if len(pos) > 1 else float(getattr(s[0], 'SliceThickness', 0) or 0)
    spacing = (abs(dz) or None, float(ps[0]), float(ps[1])) if ps is not None else None
    if spacing is None: notes.append('DICOM has no PixelSpacing; millimetres will be estimated.')
    return vol, spacing, notes


def _read_dicom_bytes(blobs):
    import pydicom
    out = []
    for b in blobs:
        try: out.append(pydicom.dcmread(io.BytesIO(b), force=True))
        except Exception: pass
    return out


def _from_nifti(path):
    import nibabel as nib
    img = nib.as_closest_canonical(nib.load(path))           # RAS+: axis0 left->right, axis1 posterior->anterior, axis2 inferior->superior
    d = np.asarray(img.dataobj, dtype=np.float32)
    if d.ndim == 4: d = d[..., 0]
    zx, zy, zz = [float(z) for z in img.header.get_zooms()[:3]]
    # sagittal slices along x; rows = superior->inferior, cols = anterior->posterior
    vol = np.transpose(d, (0, 2, 1))[:, ::-1, ::-1].copy()
    return vol, (zx, zz, zy), []


def load_volume(path, filename=None):
    """Returns dict(vol=float32 (S,H,W), spacing=(slice, row, col) mm or None, fmt=str, notes=[str])."""
    name = (filename or os.path.basename(path)).lower()
    notes = []
    if os.path.isdir(path):
        blobs = []
        for root, _, files in os.walk(path):
            for f in files: blobs.append(open(os.path.join(root, f), 'rb').read())
        vol, spacing, notes = _from_dicoms(_read_dicom_bytes(blobs)); fmt = 'dicom'
    elif name.endswith('.zip'):
        with zipfile.ZipFile(path) as z:
            blobs = [z.read(i) for i in z.infolist() if not i.is_dir() and i.file_size < 50_000_000][:2000]
        vol, spacing, notes = _from_dicoms(_read_dicom_bytes(blobs)); fmt = 'dicom'
    elif name.endswith('.dcm'):
        vol, spacing, notes = _from_dicoms(_read_dicom_bytes([open(path, 'rb').read()])); fmt = 'dicom'
    elif name.endswith('.nii') or name.endswith('.nii.gz'):
        vol, spacing, notes = _from_nifti(path); fmt = 'nifti'
    elif name.endswith('.npy'):
        vol = np.load(path, allow_pickle=False).astype(np.float32); spacing = None; fmt = 'npy'
        notes.append('No pixel spacing in .npy files; millimetres are estimated from anatomy.')
    elif name.endswith('.pck'):
        vol = safe_pck.load(path).astype(np.float32); spacing = None; fmt = 'kneemri-pck'
        notes.append('No pixel spacing in KneeMRI .pck files; millimetres are estimated from anatomy.')
    else:
        raise LoadError('Unsupported file type. Use a .zip of DICOM files, .dcm, .nii/.nii.gz, .npy or .pck.')
    if vol.ndim != 3: raise LoadError(f'Expected a 3D volume, got shape {vol.shape}.')
    if vol.shape[0] > vol.shape[1] and vol.shape[0] > vol.shape[2]:
        notes.append('Volume has more slices than rows/columns; check it is a sagittal stack.')
    if vol.shape[0] < 9: raise LoadError(f'Only {vol.shape[0]} slices; need a full sagittal stack.')
    return dict(vol=vol, spacing=spacing, fmt=fmt, notes=notes)
