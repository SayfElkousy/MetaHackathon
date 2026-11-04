"""Synthetic phantoms (never patient data).

The phantom is defined analytically in world RAS+ millimetres, so it can be sampled on any grid: a sagittal DICOM
series, an axial DICOM series in the same frame of reference, and a NIfTI stored in a non-canonical orientation all
show the same anatomy at the same world positions. That makes orientation, crosshair and alignment tests exact.

Right knee, RAS+: +x patient right (= lateral for a right knee), +y anterior, +z superior.
  femur  : distal femur above z = 5 mm with an intercondylar notch
  tibia  : proximal tibia below z = -4 mm
  ACL    : 4 mm-radius cylinder from the femoral attachment F to the tibial attachment T (labels: 1)
  marker : bright 4 mm sphere at (+25, +15, +30) = Right, Anterior, Superior (orientation fiducial)
Labels: 1 ACL, 2 femur, 3 tibia.
"""
import os

import numpy as np

F_ATT = np.array([5.0, -10.0, 12.0])     # femoral ACL attachment centre (lateral wall of the notch, posterior)
T_ATT = np.array([-1.0, 8.0, -4.0])      # tibial ACL attachment centre (anterior, between the spines)
ACL_R = 4.0
MARKER = np.array([25.0, 15.0, 30.0])
MARKER_R = 4.0
LABELS = [{"value": 1, "name": "ACL"}, {"value": 2, "name": "femur"}, {"value": 3, "name": "tibia"}]
ACL_LENGTH_MM = float(np.linalg.norm(F_ATT - T_ATT))


def phantom(world):
    """world: (..., 3) RAS mm -> (intensity float32, label uint8)."""
    x, y, z = world[..., 0], world[..., 1], world[..., 2]
    I = np.full(x.shape, 40.0, np.float32)
    L = np.zeros(x.shape, np.uint8)
    inside = (np.abs(x) < 32) & (np.abs(y) < 28)
    fem = inside & (z > 5 + y ** 2 / 70.0) & (z < 60)
    notch = (np.abs(x) < 7) & (z < 22) & (y < 12)
    fem &= ~notch
    tib = inside & (z < -4) & (z > -60) & ~((np.abs(x) < 3) & (z > -6))
    I[fem] = 700.0
    L[fem] = 2
    I[tib] = 640.0
    L[tib] = 3
    d = T_ATT - F_ATT
    t = np.clip(((world - F_ATT) @ d) / (d @ d), 0, 1)
    dist = np.linalg.norm(world - (F_ATT + t[..., None] * d), axis=-1)
    acl = (dist <= ACL_R) & (((world - F_ATT) @ d) / (d @ d) >= 0) & (((world - F_ATT) @ d) / (d @ d) <= 1)
    I[acl] = 150.0
    L[acl] = 1
    mk = np.linalg.norm(world - MARKER, axis=-1) <= MARKER_R
    I[mk] = 1000.0
    # deterministic texture so windowing is visible
    I += (8.0 * np.sin(x * 0.9) * np.cos(y * 0.7) + 6.0 * np.sin(z * 1.3)).astype(np.float32)
    return I, L


def grid_world(affine, shape):
    i, j, k = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    ijk = np.stack([i, j, k, np.ones_like(i)], -1).astype(np.float64)
    return (ijk @ np.asarray(affine).T)[..., :3]


def sample(affine, shape):
    return phantom(grid_world(affine, shape))


# --------------------------------------------------------------------------- grids
def sagittal_grid():
    """Typical knee sagittal acquisition: 3 mm slices along x, 0.5 mm in-plane. DICOM LPS geometry."""
    rows, cols, nslices = 180, 160, 21
    row_dir = np.array([0.0, 1.0, 0.0])       # LPS +y = posterior: columns run anterior -> posterior
    col_dir = np.array([0.0, 0.0, -1.0])      # rows run superior -> inferior
    normal = np.cross(row_dir, col_dir)        # LPS -x = patient right
    ps = (0.5, 0.5)
    first = np.array([30.0, -40.0, 45.0])     # LPS position of slice 0 pixel (0,0): x=+30 LPS = 30 mm left
    return dict(rows=rows, cols=cols, n=nslices, row=row_dir, col=col_dir, normal=normal, ps=ps, step=3.0, first=first,
                description="SYNTHETIC sagittal PD (phantom)", series_number=3)


def axial_grid():
    rows, cols, nslices = 120, 120, 30
    return dict(rows=rows, cols=cols, n=nslices, row=np.array([1.0, 0.0, 0.0]), col=np.array([0.0, 1.0, 0.0]),
                normal=np.array([0.0, 0.0, 1.0]), ps=(0.6, 0.6), step=3.0, first=np.array([-36.0, -36.0, -45.0]),
                description="SYNTHETIC axial PD (phantom)", series_number=4)


def lps_slice_world(g, k):
    """World RAS coordinates of every pixel of slice k of a DICOM grid -> (rows, cols, 3)."""
    r, c = np.meshgrid(np.arange(g["rows"]), np.arange(g["cols"]), indexing="ij")
    ipp = g["first"] + k * g["step"] * g["normal"]
    lps = ipp + c[..., None] * g["ps"][1] * g["row"] + r[..., None] * g["ps"][0] * g["col"]
    return lps * np.array([-1.0, -1.0, 1.0])


def grid_affine_ras(g):
    """Voxel (i=col, j=row, k=slice) -> RAS world, the same geometry a DICOM reader produces."""
    A = np.eye(4)
    A[:3, 0] = g["row"] * g["ps"][1]
    A[:3, 1] = g["col"] * g["ps"][0]
    A[:3, 2] = g["normal"] * g["step"]
    A[:3, 3] = g["first"]
    return np.diag([-1.0, -1.0, 1.0, 1.0]) @ A


# --------------------------------------------------------------------------- writers
def write_dicom_series(dirpath, g, study_uid, frame_uid, series_uid=None, skip=(), duplicate=(), uid_seed="acl"):
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
    os.makedirs(dirpath, exist_ok=True)
    series_uid = series_uid or generate_uid(entropy_srcs=[uid_seed, "series", str(g["series_number"])])
    paths = []
    for k in range(g["n"]):
        if k in skip:
            continue
        I, _ = phantom(lps_slice_world(g, k))
        sop = generate_uid(entropy_srcs=[uid_seed, series_uid, str(k)])
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = sop
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID, ds.SOPInstanceUID = MRImageStorage, sop
        ds.Modality = "MR"
        ds.PatientName, ds.PatientID = "SYNTHETIC^PHANTOM", "SYNTH-0000"
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study_uid, series_uid, frame_uid
        ds.SeriesNumber, ds.InstanceNumber = g["series_number"], k + 1
        ds.SeriesDescription = g["description"]
        ds.StudyDescription = "SYNTHETIC knee phantom (not patient data)"
        ds.MRAcquisitionType, ds.SequenceName = "2D", "phantom_pd"
        ds.RepetitionTime, ds.EchoTime, ds.MagneticFieldStrength = 3000.0, 30.0, 3.0
        ds.ImagePositionPatient = [float(v) for v in (g["first"] + k * g["step"] * g["normal"])]
        ds.ImageOrientationPatient = [float(v) for v in list(g["row"]) + list(g["col"])]
        ds.PixelSpacing = [float(g["ps"][0]), float(g["ps"][1])]
        ds.SliceThickness = float(g["step"])
        ds.Rows, ds.Columns = g["rows"], g["cols"]
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
        ds.WindowCenter, ds.WindowWidth = 400, 900
        ds.PixelData = np.clip(np.round(I), 0, 65535).astype(np.uint16).tobytes()
        p = os.path.join(dirpath, "IM%04d.dcm" % (k + 1))
        ds.save_as(p, enforce_file_format=True)
        paths.append(p)
        if k in duplicate:
            p2 = os.path.join(dirpath, "IM%04d_copy.dcm" % (k + 1))
            ds.save_as(p2, enforce_file_format=True)
            paths.append(p2)
    return paths


def write_nifti(path, g, label=False, rotate_deg=0.0):
    """NIfTI on the sagittal grid, stored in the acquisition (non-canonical) orientation."""
    import nibabel as nib
    A = grid_affine_ras(g)
    if rotate_deg:
        t = np.radians(rotate_deg)
        R = np.eye(4)
        R[1:3, 1:3] = [[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]
        A = R @ A
    shape = (g["cols"], g["rows"], g["n"])
    I, L = sample(A, shape)
    data = L.astype(np.uint8) if label else np.round(I).astype(np.int16)
    img = nib.Nifti1Image(data, A)
    img.set_sform(A, code=1)
    img.set_qform(A, code=1)
    img.header.set_xyzt_units("mm")
    nib.save(img, path)
    return path


def write_png_stack(dirpath, g, every=1):
    from PIL import Image
    os.makedirs(dirpath, exist_ok=True)
    out = []
    for k in range(0, g["n"], every):
        I, _ = phantom(lps_slice_world(g, k))
        im = Image.fromarray(np.clip(I / 4, 0, 255).astype(np.uint8))
        p = os.path.join(dirpath, "slice_%d.png" % (k + 1))
        im.save(p)
        out.append(p)
    return out


def make_fixtures(root):
    """Write the synthetic fixture set used by tests and the demo case."""
    from pydicom.uid import generate_uid
    os.makedirs(root, exist_ok=True)
    study = generate_uid(entropy_srcs=["acl-phantom-study"])
    frame = generate_uid(entropy_srcs=["acl-phantom-frame"])
    sg, ag = sagittal_grid(), axial_grid()
    write_dicom_series(os.path.join(root, "dicom_sagittal"), sg, study, frame)
    write_dicom_series(os.path.join(root, "dicom_axial"), ag, study, frame)
    write_nifti(os.path.join(root, "sagittal.nii.gz"), sg)
    write_nifti(os.path.join(root, "sagittal_labels.nii.gz"), sg, label=True)
    write_png_stack(os.path.join(root, "png_stack"), sg, every=2)
    with open(os.path.join(root, "README.md"), "w") as f:
        f.write("# Synthetic imaging fixtures\n\nGenerated by `python tools/make_imaging_fixtures.py` from the analytic phantom in "
                "`imaging/synthetic.py`. Not patient data. Labels: 1 ACL, 2 femur, 3 tibia. ACL attachment distance "
                "%.2f mm. Orientation fiducial: bright sphere at RAS (+25, +15, +30).\n" % ACL_LENGTH_MM)
    return root
