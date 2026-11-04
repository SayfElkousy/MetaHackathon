"""Volumetric imaging: orientation, physical proportions, validation, alignment, model-output mapping, QC, evidence.

All data are synthetic (imaging/synthetic.py): an analytic phantom sampled on DICOM and NIfTI grids."""
import glob
import json
import os
import sys
import time
import zipfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "mri_analysis"))

nib = pytest.importorskip("nibabel")
pytest.importorskip("pydicom")
pytest.importorskip("SimpleITK")

from imaging import evidence as E  # noqa: E402
from imaging import model_map as MM  # noqa: E402
from imaging import qc as QC  # noqa: E402
from imaging import store as ST  # noqa: E402
from imaging import synthetic as Y  # noqa: E402
from imaging import volumes as V  # noqa: E402


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("imgfx"))
    Y.make_fixtures(root)
    return root


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(ST, "BASE", str(tmp_path / "store"))
    return ST


def files(d):
    return [(p, os.path.basename(p)) for p in sorted(glob.glob(os.path.join(d, "*")))]


def vox(rec, world):
    return np.linalg.inv(np.array(rec["affine"])) @ np.r_[world, 1.0]


def at(rec, world):
    return rec["data"][tuple(int(round(x)) for x in vox(rec, world)[:3])]


# ------------------------------------------------------------------ orientation and proportions
def test_dicom_and_nifti_reach_the_same_canonical_ras_grid(fx):
    d = V.load_files(files(os.path.join(fx, "dicom_sagittal")))[0]
    n = V.load_files([(os.path.join(fx, "sagittal.nii.gz"), "sagittal.nii.gz")])[0]
    assert d["shape"] == n["shape"] == [21, 160, 180]
    assert np.allclose(d["affine"], n["affine"], atol=1e-4)
    assert "".join(nib.aff2axcodes(np.array(d["affine"]))) == "RAS"
    assert d["source_axcodes"] != "RAS"                      # stored differently on disk; reoriented by nibabel
    assert np.array_equal(d["data"], n["data"])


def test_orientation_fiducial_lands_right_anterior_superior(fx):
    d = V.load_files(files(os.path.join(fx, "dicom_sagittal")))[0]
    assert at(d, Y.MARKER) >= 990                            # bright marker at RAS (+25, +15, +30)
    assert at(d, Y.MARKER * np.array([-1, 1, 1])) < 900      # its mirror on the patient's left is not bright
    i, j, k = vox(d, Y.MARKER)[:3]
    c = (np.array(d["shape"]) - 1) / 2
    # canonical RAS+: higher i = right, higher j = anterior, higher k = superior
    assert i > c[0] and j > c[1] and k > c[2]


def test_anisotropic_spacing_preserves_physical_size(fx):
    d = V.load_files(files(os.path.join(fx, "dicom_sagittal")))[0]
    assert d["zooms"] == [3.0, 0.5, 0.5]
    assert any(c["id"] == "thick_slices" and c["status"] == "warn" for c in d["checks"])
    bright = np.argwhere(d["data"] >= 990)
    ext_mm = (bright.max(0) - bright.min(0) + 1) * np.array(d["zooms"])
    # the 8 mm-diameter sphere spans ~8 mm on every axis, within one voxel of each axis
    for a in range(3):
        assert abs(ext_mm[a] - 2 * Y.MARKER_R) <= d["zooms"][a] + 1e-6


def test_second_series_in_same_frame_shows_same_anatomy(fx):
    sag = V.load_files(files(os.path.join(fx, "dicom_sagittal")))[0]
    ax = V.load_files(files(os.path.join(fx, "dicom_axial")))[0]
    assert sag["frame_id"] and sag["frame_id"] == ax["frame_id"]
    assert at(sag, Y.MARKER) >= 990 and at(ax, Y.MARKER) >= 990
    # a point inside the femur is femur intensity in both series (crosshair agreement across series)
    p = np.array([20.0, 0.0, 30.0])
    assert abs(float(at(sag, p)) - 700) < 30 and abs(float(at(ax, p)) - 700) < 30
    nif = V.load_files([(os.path.join(fx, "sagittal.nii.gz"), "s.nii.gz")])[0]
    assert nif["frame_id"] is None                            # NIfTI scanner space: no verifiable frame -> not linked


def test_oblique_volume_is_flagged(tmp_path):
    p = Y.write_nifti(str(tmp_path / "obl.nii.gz"), Y.sagittal_grid(), rotate_deg=12)
    r = V.load_files([(p, "obl.nii.gz")])[0]
    assert r["obliquity_deg"] == pytest.approx(12, abs=0.5)
    assert any(c["id"] == "oblique" for c in r["checks"])
    assert at(r, Y.MARKER) >= 990 or at(r, Y.MARKER + [0, 0, 0.5]) >= 990


# ------------------------------------------------------------------ DICOM validation
def test_missing_slice_is_reported_and_geometry_kept(tmp_path):
    g = Y.sagittal_grid()
    Y.write_dicom_series(str(tmp_path / "gap"), g, "1.2.3", "1.2.3.4", skip=(10,))
    r = V.load_files(files(str(tmp_path / "gap")))[0]
    assert r["shape"][0] == 21
    assert any(c["id"] == "missing_slices" and c["status"] == "warn" for c in r["checks"])
    assert r["missing_slices"] and r["missing_slices"][0]["axis"] == 0
    assert at(r, Y.MARKER) >= 990                             # geometry after the gap is still right


def test_duplicates_dropped_and_same_position_refused(tmp_path):
    g = Y.sagittal_grid()
    Y.write_dicom_series(str(tmp_path / "dup"), g, "1.2.3", "1.2.3.4", duplicate=(3,))
    r = V.load_files(files(str(tmp_path / "dup")))[0]
    assert any(c["id"] == "duplicates" for c in r["checks"]) and r["shape"][0] == 21
    # two different instances at one position (e.g. two echoes) -> refused with an actionable message
    Y.write_dicom_series(str(tmp_path / "echo"), g, "1.2.3", "1.2.3.4", series_uid="1.2.3.9", uid_seed="e1")
    import pydicom
    p = sorted(glob.glob(str(tmp_path / "echo" / "*.dcm")))[4]
    ds = pydicom.dcmread(p)
    ds.SOPInstanceUID = "1.2.3.9.999"
    ds.save_as(str(tmp_path / "echo" / "extra.dcm"), enforce_file_format=True)
    out = V.load_files(files(str(tmp_path / "echo")))
    assert out[0].get("failed")
    assert any(c["id"] == "same_position" and "hint" in c for c in out[0]["checks"])


def test_simpleitk_and_gap_path_agree_on_affine(tmp_path, fx):
    """The gap-filling path builds its affine from the DICOM tags; it must match SimpleITK's geometry."""
    g = Y.sagittal_grid()
    Y.write_dicom_series(str(tmp_path / "gap"), g, "1.2.3", "1.2.3.4", skip=(3,))
    gap = V.load_files(files(str(tmp_path / "gap")))[0]
    full = V.load_files(files(os.path.join(fx, "dicom_sagittal")))[0]
    assert "pydicom" in gap["meta"]["reader"] and "SimpleITK" in full["meta"]["reader"]
    assert np.allclose(gap["affine"], full["affine"], atol=1e-4)
    keep = np.ones(21, bool)
    keep[gap["missing_slices"][0]["index"]] = False
    assert np.array_equal(gap["data"][keep], full["data"][keep])


def test_uncalibrated_stack(fx):
    r = V.load_files(files(os.path.join(fx, "png_stack")))[0]
    assert not r["spacing_known"] and not r["orientation_known"] and r["units"] == "voxel"
    assert any(c["id"] == "spacing" and c["status"] == "warn" for c in r["checks"])


# ------------------------------------------------------------------ masks
def test_mask_alignment(fx):
    img = V.load_files(files(os.path.join(fx, "dicom_sagittal")))[0]
    lab = V.load_files([(os.path.join(fx, "sagittal_labels.nii.gz"), "l.nii.gz")])[0]
    arr, checks = V.align_mask(lab, img)
    assert checks[0]["status"] == "pass"
    assert arr[tuple(int(round(x)) for x in vox(img, (Y.F_ATT + Y.T_ATT) / 2)[:3])] == 1      # ACL midpoint
    shifted = dict(lab, affine=(np.array(lab["affine"]) + np.r_[[[0, 0, 0, 2.0]] * 3, [[0, 0, 0, 0]]]).tolist())
    with pytest.raises(V.ImportFailed) as e:
        V.align_mask(shifted, img)
    assert e.value.checks[0]["id"] == "grid" and "hint" in e.value.checks[0]
    arr2, ch2 = V.align_mask(shifted, img, allow_resample=True)
    assert ch2[0]["status"] == "warn" and arr2.shape == tuple(img["shape"])
    prob = dict(lab, data=lab["data"].astype(np.float32) * 0.37)
    with pytest.raises(V.ImportFailed) as e2:
        V.align_mask(prob, img)
    assert "derived map" in e2.value.checks[0]["hint"]


# ------------------------------------------------------------------ store, annotations, measurements
def demo(store, fx):
    m = store.create_case("t", synthetic=True)
    r = V.load_files(files(os.path.join(fx, "dicom_sagittal")))[0]
    v = store.add_volume(m["id"], r)
    return m["id"], v, r


def test_volume_bytes_round_trip_unchanged(store, fx):
    cid, v, r = demo(store, fx)
    arr, _ = store.volume_array(cid, v["id"])
    assert arr.dtype == r["data"].dtype and np.array_equal(arr, r["data"])
    p, _ = store.volume_bytes_path(cid, v["id"])
    import hashlib
    assert hashlib.sha1(open(p, "rb").read()).hexdigest() == v["sha1"]


def test_calibrated_distance_matches_physical_distance(store, fx):
    cid, v, r = demo(store, fx)
    pF, pT = vox(r, Y.F_ATT)[:3], vox(r, Y.T_ATT)[:3]
    a = store.upsert_annotation(cid, {"type": "distance", "kind": "acl_length", "volume": v["id"], "points_voxel": [list(pF), list(pT)]})
    assert a["units"] == "mm" and a["value"] == pytest.approx(Y.ACL_LENGTH_MM, abs=0.01)
    assert np.allclose(a["points_world"][0], Y.F_ATT, atol=1e-6)
    a = store.upsert_annotation(cid, dict(a, status="reviewed"))
    assert a["status"] == "reviewed"
    moved = store.upsert_annotation(cid, dict(a, points_voxel=[list(pF + [0, 2, 0]), list(pT)]))
    assert moved["status"] == "draft"                        # moving a reviewed point voids the review


def test_uncalibrated_distance_cannot_be_reviewed_as_mm(store, fx):
    m = store.create_case("u")
    r = V.load_files(files(os.path.join(fx, "png_stack")))[0]
    v = store.add_volume(m["id"], r)
    a = store.upsert_annotation(m["id"], {"type": "distance", "kind": "acl_length", "volume": v["id"], "points_voxel": [[1, 1, 1], [4, 5, 1]]})
    assert a["units"].startswith("voxel") and a["value"] == 5.0
    with pytest.raises(ValueError):
        store.upsert_annotation(m["id"], dict(a, status="reviewed"))


def test_edits_keep_the_original_prediction(store, fx):
    cid, v, r = demo(store, fx)
    pred = np.zeros(r["data"].shape, np.uint8)
    pred[10, 50:60, 50:60] = 1
    m0 = store.add_mask(cid, v["id"], pred, [{"value": 1, "name": "ACL"}], "model", model={"name": "KneeScan"})
    ed = pred.copy()
    ed[10, 60:62, 50:60] = 1
    m1 = store.add_mask(cid, v["id"], ed, [{"value": 1, "name": "ACL"}], "edited", parent=m0["id"])
    assert m1["edited"] and m1["edit_voxels"] == 20 and m1["root"] == m0["id"] and m1["model"]["name"] == "KneeScan"
    a0, _ = store.mask_array(cid, m0["id"])
    assert a0.sum() == 100                                   # prediction unchanged
    store.remove(cid, "volume", v["id"])
    assert not store.load(cid)["masks"]


# ------------------------------------------------------------------ model output mapping (KneeScan index space)
def _kneescan_load(path, name):
    from kneescan.io import load_volume
    return load_volume(path, name)


def _hull_outline(mask2d):
    from scipy.spatial import ConvexHull
    ys, xs = np.nonzero(mask2d)
    pts = np.stack([xs, ys], 1).astype(float)
    h = ConvexHull(pts)
    return [[float(x), float(y)] for x, y in pts[h.vertices]]


def _dice(a, b):
    return 2 * np.logical_and(a, b).sum() / max(1, a.sum() + b.sum())


@pytest.mark.parametrize("fmt", ["nifti", "dicom"])
def test_model_outline_maps_back_onto_the_right_voxels(fx, tmp_path, fmt):
    truth = V.load_files([(os.path.join(fx, "sagittal_labels.nii.gz"), "l.nii.gz")])[0]["data"]
    if fmt == "nifti":
        path, name = os.path.join(fx, "sagittal.nii.gz"), "sagittal.nii.gz"
        rec = V.load_files([(path, name)])[0]
        ks_lab = _kneescan_load(os.path.join(fx, "sagittal_labels.nii.gz"), "l.nii.gz")["vol"]
    else:
        path, name = str(tmp_path / "s.zip"), "s.zip"
        with zipfile.ZipFile(path, "w") as z:
            for p, n in files(os.path.join(fx, "dicom_sagittal")):
                z.write(p, n)
        rec = V.load_files([(path, name)])[0]
        g = Y.sagittal_grid()
        ks_lab = np.stack([Y.phantom(Y.lps_slice_world(g, k))[1] for k in range(g["n"])]).astype(np.float32)
    ks = _kneescan_load(path, name)
    assert ks["vol"].shape == ks_lab.shape
    s = int(np.argmax([(ks_lab[k] == 1).sum() for k in range(ks_lab.shape[0])]))
    outline = _hull_outline(ks_lab[s] == 1)
    ys, xs = np.nonzero(ks_lab[s] == 1)
    e0, e1 = [float(xs.min()), float(ys[np.argmin(xs)])], [float(xs.max()), float(ys[np.argmax(xs)])]
    result = {"volume": {"slices": ks["vol"].shape[0]}, "acl": {"slice": s, "outline": outline, "end0": e0, "end1": e1, "confidence": 0.8,
                                                               "length_mm": None},
              "tear": {"status": "torn", "probability": 0.7, "threshold": 0.5, "level": "partial tear", "level_reliability": "low"}}
    rec["id"] = "v"
    mask, labels, rep = MM.outlines_to_mask(rec, result)
    assert rep["round_trip_max_vox"] < 1e-6 and rep["outside_points"] == 0
    assert labels[0]["name"] == "ACL" and "2D" in labels[0]["dimensionality"]
    sl = np.nonzero(mask.any(axis=(1, 2)))[0]
    assert len(sl) == 1                                      # one sagittal slice: a 2D result, not 3D
    t = truth[sl[0]] == 1
    assert _dice(mask[sl[0]] == 1, t) > 0.9
    # endpoints map into the volume at the ACL ends
    p0 = MM.map_point(rec, s, e0)
    assert truth[tuple(int(round(x)) for x in p0)] == 1


def test_mapping_refuses_when_slices_do_not_match(store, fx):
    from imaging import api as IA
    cid, v, r = demo(store, fx)
    with pytest.raises(ValueError):
        IA._store_model_result(cid, store.load(cid)["volumes"][v["id"]], {"volume": {"slices": 99}, "acl": {}, "tear": {}}, {})


# ------------------------------------------------------------------ QC
def _qc(vrec, arr, labels=None, origin="model", review=None, model=None):
    mk = {"labels": labels or [{"value": 1, "name": "ACL"}], "origin": origin, "review": review or {"status": "unreviewed"},
          "checks": [], "model": model}
    return {c["id"]: c for cat in QC.run(vrec, mk, arr)["categories"] for c in cat["checks"]}


def test_qc_flags(fx):
    r = V.load_files(files(os.path.join(fx, "dicom_sagittal")))[0]
    z = np.zeros(r["data"].shape, np.uint8)
    assert _qc(r, z)["empty"]["status"] == "fail"
    two = z.copy()
    two[5:9, 40:60, 40:60] = 1
    two[5:9, 100:103, 100:103] = 1
    c = _qc(r, two)
    assert c["components_1"]["status"] == "warn" and "location" in c["components_1"]
    assert c["components_1"]["location"]["voxel"][1] == pytest.approx(101, abs=1)
    one = z.copy()
    one[7, 40:60, 40:60] = 1
    assert _qc(r, one)["single_slice_1"]["status"] == "not_evaluated"
    jump = z.copy()
    jump[5, 40:42, 40:42] = 1
    jump[6, 30:70, 30:70] = 1
    assert _qc(r, jump)["slice_jump_1"]["status"] == "warn"
    gap = z.copy()
    gap[4, 40:60, 40:60] = 1
    gap[6, 40:60, 40:60] = 1
    assert _qc(r, gap)["slice_gap_1"]["status"] == "warn"
    c = _qc(r, one, review={"status": "reviewed", "at": "now"}, model={"confidence": 0.8})
    assert c["review"]["status"] == "pass"
    assert "not a measured accuracy" in c["model_confidence"]["message"]


# ------------------------------------------------------------------ evidence -> parameters
from acl_engine.v2.design import new_state  # noqa: E402


def _manifest(meas=(), findings=(), masks=None):
    anns = []
    for i, (kind, val, status) in enumerate(meas):
        anns.append({"id": "a%d" % i, "type": "distance", "kind": kind, "value": val, "units": "mm", "calibrated": True,
                     "status": status, "volume": "v1"})
    return {"volumes": {"v1": {"name": "SAG PD"}}, "annotations": anns, "findings": list(findings), "masks": masks or {}}


def test_report_extraction():
    t = ("MRI of the right knee. Complete tear of the anterior cruciate ligament at the proximal third near the femoral "
         "attachment with a 6 mm gap. ACL length measures 3.4 cm. No meniscal tear.")
    f = {(x["kind"], x["value"]) for x in E.extract_report(t)}
    assert ("tear_type", "complete") in f and ("tear_location", "proximal") in f and ("tear_gap", 6.0) in f
    assert ("acl_length", 34.0) in f
    assert ("tear_type", "intact") not in f                 # "No meniscal tear" says nothing about the ACL
    assert {x["kind"] for x in E.extract_report("The ACL is intact.")} == {"tear_type"}


def test_reviewed_measurement_applies_with_units_and_assumption():
    st = new_state()
    p = st["params"]
    pr = E.propose(_manifest([("acl_length", 36.5, "reviewed")]), p, st["provenance"])
    row = next(r for r in pr["rows"] if r["mapping"] == "acl_length")
    assert row["status"] == "apply" and row["changes"] == {"lig.length_target_mm": 36.5, "lig.length_mode": "target"}
    assert row["relation"] == "assumption" and row["assumptions"] and row["source"]["volume_name"] == "SAG PD"
    assert pr["changes"]["lig.length_target_mm"] == 36.5


def test_unreviewed_measurement_is_not_used():
    st = new_state()
    pr = E.propose(_manifest([("acl_length", 36.5, "draft")]), st["params"], st["provenance"])
    row = next(r for r in pr["rows"] if r["mapping"] == "acl_length")
    assert row["status"] == "missing" and "unreviewed" in row["reason"]


def test_conflicts_surface_instead_of_last_wins():
    st = new_state()
    f = E.extract_report("ACL length measures 40 mm.")
    pr = E.propose(_manifest([("acl_length", 33.0, "reviewed")], f), st["params"], st["provenance"])
    row = next(r for r in pr["rows"] if r["mapping"] == "acl_length")
    assert row["status"] == "conflict" and "lig.length_target_mm" not in pr["changes"]
    # a value the user typed is not silently overwritten
    st["provenance"]["injury.gap_mm"] = {"status": "override", "source": "manual edit"}
    st["params"]["injury.gap_mm"] = 3.0
    pr = E.propose(_manifest([("tear_gap", 7.0, "reviewed")]), st["params"], st["provenance"])
    assert next(r for r in pr["rows"] if r["mapping"] == "tear_gap")["status"] == "conflict"


def test_limits_and_modes_block_mappings():
    st = new_state()
    pr = E.propose(_manifest([("acl_length", 80.0, "reviewed")]), st["params"], st["provenance"])
    row = next(r for r in pr["rows"] if r["mapping"] == "acl_length")
    assert row["status"] == "blocked" and "maximum" in row["reason"]
    st["params"]["lig.section_mode"] = "per_bundle"
    pr = E.propose(_manifest([("acl_width", 9.0, "reviewed")]), st["params"], st["provenance"])
    assert next(r for r in pr["rows"] if r["mapping"] == "acl_width")["status"] == "blocked"


def test_assumption_and_model_inference_need_review():
    st = new_state()
    f = E.extract_report("Partial tear of the anteromedial bundle in the midsubstance.")
    masks = {"m1": {"origin": "model", "model": {"tear": {"status": "torn", "level": "partial tear", "probability": 0.7}}}}
    pr = E.propose(_manifest([], f, masks), st["params"], st["provenance"])
    rows = {r["mapping"]: r for r in pr["rows"]}
    assert rows["tear_type"]["status"] == "apply" and rows["tear_type"]["source"]["certainty"] == "reported"
    assert rows["tear_bundle"]["status"] == "apply" and rows["tear_bundle"]["changes"] == {"injury.involvement": "AM"}
    assert rows["tear_location"]["status"] == "needs_review" and "injury.center_s" not in pr["changes"]
    ids = {n["params"] for n in pr["not_mapped"]}
    assert "gr.*" in ids and "ch.*" in ids


# ------------------------------------------------------------------ API flow
def test_import_job_and_segmentation_service_unavailable(store, fx, monkeypatch):
    from imaging import api as IA
    m = store.create_case("api")
    batch = "abc123"
    d = store.upload_dir(m["id"], batch)
    for p, n in files(os.path.join(fx, "dicom_sagittal")):
        with open(p, "rb") as f:
            IA.upload({"case": [m["id"]], "batch": [batch], "name": [n]}, f, os.path.getsize(p))
    job = IA.start_import({"case": m["id"], "batch": batch, "role": "image"})
    for _ in range(200):
        j = store.job_status(job["id"])
        if j["status"] != "running":
            break
        time.sleep(0.05)
    assert j["status"] == "done", j
    vid = j["result"]["volumes"][0]["id"]
    assert store.load(m["id"])["volumes"][vid]["source_files"]
    monkeypatch.setenv("KNEESCAN_URL", "http://127.0.0.1:9")
    job = IA.start_segmentation({"case": m["id"], "volume": vid, "reader_url": "http://127.0.0.1:9"})
    for _ in range(200):
        j = store.job_status(job["id"])
        if j["status"] != "running":
            break
        time.sleep(0.05)
    assert j["status"] == "error" and "unavailable" in j["error"]["message"]
    assert not store.load(m["id"])["masks"]                  # no fake result
