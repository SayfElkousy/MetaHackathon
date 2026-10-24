"""Missing metadata, unknown-vs-background labels, annotation traceability."""
import nibabel as nib
import numpy as np
import pydicom
import torch

from bilacl.annotation.schema import AnnotationRecord, Label, effective_target, read_slicer_markups, write_slicer_markups
from bilacl.imaging.preprocess import resample_spacing
from bilacl.imaging.volume import PhysicalStatus, Volume, discover_dicom_series, load_nifti, select_series, volume_from_array
from bilacl.provenance import ReviewStatus
from bilacl.synthetic.phantom import write_dicom_series
from bilacl.training.seg import masked_loss


def test_nifti_without_orientation_is_unverified(tmp_path):
    img = nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), np.eye(4))
    img.set_qform(None, code=0)
    img.set_sform(None, code=0)
    nib.save(img, str(tmp_path / "x.nii.gz"))
    v = load_nifti(tmp_path / "x.nii.gz")
    assert not v.physical.verified and v.physical.reasons


def test_bare_array_is_unverified():
    v = volume_from_array(np.zeros((3, 256, 256)))
    assert not v.physical.verified and v.physical.source == "array-no-header"


def _series(tmp_path, n=10):
    v = Volume(np.random.default_rng(0).random((12, 12, n)).astype(np.float32), np.diag([0.5, 0.5, 1.0, 1]), PhysicalStatus(True, "t"))
    return write_dicom_series(v, tmp_path, shuffle_seed=None)


def test_missing_slice_detected(tmp_path):
    d = _series(tmp_path / "s")
    (d / "IM0004.dcm").unlink()
    (rep, _), = select_series(discover_dicom_series(d))
    assert not rep.usable and any("gap" in i for i in rep.issues)


def test_missing_geometry_tags_blocks(tmp_path):
    d = _series(tmp_path / "s")
    for p in d.glob("*.dcm"):
        ds = pydicom.dcmread(p)
        del ds.ImageOrientationPatient
        ds.save_as(p)
    (rep, _), = select_series(discover_dicom_series(d))
    assert not rep.usable and "missing geometry tags" in rep.issues[0]


def test_localizer_detected(tmp_path):
    d = _series(tmp_path / "s", n=3)
    for p in d.glob("*.dcm"):
        ds = pydicom.dcmread(p)
        ds.ImageType = ["ORIGINAL", "PRIMARY", "LOCALIZER"]
        ds.save_as(p)
    (rep, _), = select_series(discover_dicom_series(d))
    assert rep.is_localizer and not rep.usable


def test_laterality_unknown_without_tags(tmp_path):
    d = _series(tmp_path / "s")
    (rep, _), = select_series(discover_dicom_series(d))
    assert rep.laterality == "unknown"


def _rec(labels, **kw):
    return AnnotationRecord("a", "e", "img", "reader1", annotated_labels=labels, **kw)


def test_unknown_is_ignored_not_background():
    m = np.zeros((4, 4, 4), np.uint8)
    m[0] = Label.UNKNOWN
    m[1, 0, 0] = Label.FEMUR
    t = effective_target(m, _rec([1, 2]), {1: 1, 2: 2})
    assert (t[0] == -1).all()
    assert t[1, 0, 0] == 1 and t[2, 2, 2] == 0


def test_unannotated_structure_zero_is_not_background():
    m = np.zeros((4, 4, 4), np.uint8)
    m[0, 0, 0] = Label.FEMUR
    t = effective_target(m, _rec([int(Label.FEMUR)]), {int(Label.FEMUR): 1, int(Label.TIBIA): 2})  # tibia never annotated
    assert t[0, 0, 0] == 1
    assert (t[1:] == -1).all()  # zeros cannot be trusted as "no tibia"


def test_label_resampling_preserves_unknown_code():
    d = np.zeros((10, 10, 10), np.uint8)
    d[3:7, 3:7, 3:7] = Label.UNKNOWN
    d[5, 5, 5] = Label.ACL_REMNANT_FEMORAL
    r = resample_spacing(Volume(d, np.eye(4), PhysicalStatus(True, "t")), (0.5, 0.5, 0.5), kind="label")
    assert set(np.unique(r.data)) <= {0, int(Label.UNKNOWN), int(Label.ACL_REMNANT_FEMORAL)}


def test_model_predictions_are_not_ground_truth():
    r = _rec([1], source="model_prediction", model_id="m")
    assert not r.usable_as_ground_truth
    r2 = _rec([1], source="model_corrected", model_id="m", review_status=ReviewStatus.CORRECTED)
    assert not r2.usable_as_ground_truth
    r2.review_status = ReviewStatus.APPROVED
    assert r2.usable_as_ground_truth
    assert _rec([1], source="model_prediction").validate()  # model_id missing -> error


def test_masked_loss_ignores_unknown():
    logits = torch.randn(1, 3, 4, 4, 4)
    t = torch.full((1, 4, 4, 4), -1, dtype=torch.long)
    t[0, 0, 0, 0] = 1
    l1 = masked_loss(logits, t, 3)
    logits2 = logits.clone()
    logits2[0, :, 1:] = torch.randn(3, 3, 4, 4) * 100  # change only ignored voxels
    assert torch.isclose(l1, masked_loss(logits2, t, 3))


def test_markups_roundtrip_lps_ras(tmp_path):
    pts = {"a": [1.0, -2.0, 3.0]}
    write_slicer_markups(pts, tmp_path / "m.mrk.json")
    assert np.allclose(read_slicer_markups(tmp_path / "m.mrk.json")["a"], pts["a"])
