"""Left/right transformations, LPS/RAS conversion, physical-coordinate and unit preservation."""
import nibabel as nib
import numpy as np
import SimpleITK as sitk

from bilacl.imaging.coords import LPS_TO_RAS, apply_affine, axcodes, is_rigid, mirror_matrix, rigid_from_params
from bilacl.imaging.preprocess import resample_spacing
from bilacl.imaging.volume import (PhysicalStatus, Volume, discover_dicom_series, load_dicom_series, load_nifti, save_nifti,
                                   select_series, to_sitk, load_sitk)
from bilacl.synthetic.phantom import write_dicom_series


def test_ras_definition_matches_nibabel():
    assert axcodes(np.eye(4)) == ("R", "A", "S")
    assert axcodes(LPS_TO_RAS) == ("L", "P", "S")  # identity-in-LPS expressed in RAS


def test_lps_ras_roundtrip_and_points():
    p_lps = np.array([[10.0, -20.0, 5.0]])
    p_ras = apply_affine(LPS_TO_RAS, p_lps)
    assert np.allclose(p_ras, [[-10, 20, 5]])
    assert np.allclose(apply_affine(LPS_TO_RAS, p_ras), p_lps)


def test_mirror_is_reflection_and_involution():
    M = mirror_matrix(3.0)
    assert np.isclose(np.linalg.det(M[:3, :3]), -1)
    assert np.allclose(M @ M, np.eye(4))
    assert np.allclose(apply_affine(M, [[5.0, 1, 2]]), [[1.0, 1, 2]])
    # lateral side of a right knee (+x in RAS) maps to the lateral side of a left knee (-x)
    assert apply_affine(mirror_matrix(), [[30.0, 0, 0]])[0, 0] < 0


def test_mirrored_affine_does_not_resample():
    v = Volume(np.arange(24.0).reshape(2, 3, 4), np.diag([0.5, 0.6, 2.0, 1.0]), PhysicalStatus(True, "test"))
    A = mirror_matrix() @ v.affine
    assert np.allclose(np.linalg.norm(A[:3, :3], axis=0), v.spacing)  # spacing (mm) preserved
    assert axcodes(A)[0] == "L"


def test_sitk_conversion_matches(tmp_path):
    aff = rigid_from_params(0.2, -0.1, 0.3, t=(4, -7, 11)) @ np.diag([0.4, 0.5, 1.5, 1])
    v = Volume(np.random.default_rng(0).random((10, 12, 8)).astype(np.float32), aff, PhysicalStatus(True, "t"))
    img = to_sitk(v)
    ijk = (3, 5, 2)
    p_lps = np.array(img.TransformIndexToPhysicalPoint(ijk))
    assert np.allclose(apply_affine(LPS_TO_RAS, p_lps[None])[0], v.world_coords(np.array(ijk)[None])[0], atol=1e-5)
    sitk.WriteImage(img, str(tmp_path / "x.nrrd"))
    w = load_sitk(tmp_path / "x.nrrd")
    assert np.allclose(w.affine, v.affine, atol=1e-5) and np.allclose(w.data, v.data)


def test_nifti_roundtrip_preserves_affine_and_units(tmp_path):
    aff = rigid_from_params(0.1, 0.2, 0.0, t=(1, 2, 3)) @ np.diag([0.3, 0.3, 0.8, 1])
    v = Volume(np.ones((5, 6, 7), np.float32), aff, PhysicalStatus(True, "t"))
    save_nifti(v, tmp_path / "a.nii.gz")
    w = load_nifti(tmp_path / "a.nii.gz")
    assert w.physical.verified and np.allclose(w.affine, aff, atol=1e-4)
    assert nib.load(str(tmp_path / "a.nii.gz")).header.get_xyzt_units()[0] == "mm"


def test_resampling_preserves_world_position():
    aff = np.diag([0.5, 0.5, 2.0, 1.0])
    aff[:3, 3] = [10, -5, 3]
    d = np.zeros((40, 40, 10), np.uint8)
    d[20, 20, 5] = 1
    v = Volume(d, aff, PhysicalStatus(True, "t"))
    c0 = v.world_coords(np.argwhere(d > 0))[0]
    r = resample_spacing(v, (0.25, 0.25, 0.5), kind="label")
    c1 = r.world_coords(np.argwhere(r.data > 0)).mean(0)
    assert np.allclose(c0, c1, atol=0.5)
    assert r.meta["source_spacing_mm"] == [0.5, 0.5, 2.0]


def test_dicom_ordering_from_positions_not_filenames(tmp_path):
    aff = rigid_from_params(0.3, 0.1, -0.2, t=(5, 6, 7)) @ np.diag([0.6, 0.5, 1.2, 1])
    data = np.random.default_rng(1).random((16, 14, 9)).astype(np.float32) * 1000
    v = Volume(data, aff, PhysicalStatus(True, "t"))
    write_dicom_series(v, tmp_path / "dcm", laterality="L", shuffle_seed=7)
    (rep, items), = select_series(discover_dicom_series(tmp_path / "dcm"))
    assert rep.usable and rep.laterality == "left" and rep.laterality_source.startswith("dicom:")
    w = load_dicom_series(items, rep)
    assert np.allclose(w.affine, aff, atol=1e-3)
    assert np.allclose(w.data, data, atol=0.5)  # 12-bit quantisation
    assert is_rigid(np.vstack([np.column_stack([w.affine[:3, :3] / w.spacing, [0, 0, 0]]), [0, 0, 0, 1]]))
