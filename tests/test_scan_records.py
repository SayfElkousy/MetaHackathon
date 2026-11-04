import os

import pytest

from imaging import store as ST
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import mri_bridge as MB  # noqa: E402


def result(length=34.0, width=8.0):
    return {"volume": {"slices": 20, "rows": 64, "cols": 64, "spacing_mm": [0.5, 0.5, 2.0]},
            "tear": {"status": "torn", "level": "complete rupture", "probability": 0.9, "threshold": 0.5},
            "acl": {"slice": 9, "outline": [[5, 5], [30, 30], [32, 29], [7, 4]], "length_mm": length,
                    "width_mm": width, "passed_checks": True, "confidence": 0.8, "mm_source": "DICOM PixelSpacing"},
            "acl_location": {"slice": 9, "box": [5, 4, 32, 30], "score": 0.8}, "warnings": [],
            "input": {"file": "scan.nii.gz"}}


def test_supported_measurements_change_geometry_not_filename():
    a, _, _, _ = MB.build_state(result(width=7.0), "right", "same_name.nii.gz")
    b, _, _, _ = MB.build_state(result(width=11.0), "right", "same_name.nii.gz")
    c, _, _, _ = MB.build_state(result(width=7.0), "right", "renamed.nii.gz")
    assert a["params"]["lig.width_profile"] != b["params"]["lig.width_profile"]
    assert a["params"]["lig.width_profile"] == c["params"]["lig.width_profile"]


def test_scan_identity_cache_and_stale_revision(tmp_path, monkeypatch):
    monkeypatch.setattr(ST, "BASE", str(tmp_path / "imaging"))
    case = ST.create_case("case")
    digest = "a" * 64
    s1 = ST.create_scan(case["id"], "knee_A.nii.gz", digest, ["knee_A.nii.gz"])
    s2 = ST.create_scan(case["id"], "renamed.nii.gz", digest, ["renamed.nii.gz"])
    series = [{"name": "sag PD", "shape": [20, 64, 64], "zooms": [2, .5, .5], "fmt": "nifti", "frame_id": "x"}]
    ST.update_scan(case["id"], s1["id"], series=series)
    ST.update_scan(case["id"], s2["id"], series=series)
    model = {"name": "KneeScan", "version": "1"}; settings = {"do_bones": False}
    r1 = ST.begin_scan_analysis(case["id"], s1["id"], model, settings)
    r2 = ST.begin_scan_analysis(case["id"], s2["id"], model, settings)
    m = ST.load(case["id"])
    assert m["scans"][s1["id"]]["analysis_cache_key"] == m["scans"][s2["id"]]["analysis_cache_key"]
    newer = ST.begin_scan_analysis(case["id"], s1["id"], model, {"do_bones": True})
    with pytest.raises(ValueError, match="stale"):
        ST.finish_scan_analysis(case["id"], s1["id"], r1, analysis_status="complete")
    ST.finish_scan_analysis(case["id"], s1["id"], newer, analysis_status="complete")
    ST.finish_scan_analysis(case["id"], s2["id"], r2, analysis_status="complete")


def test_linked_design_mapping_is_per_scan(tmp_path, monkeypatch):
    monkeypatch.setattr(ST, "BASE", str(tmp_path / "imaging"))
    case = ST.create_case("case")
    scan = ST.create_scan(case["id"], "patient.zip", "b" * 64, ["patient.zip"])
    ST.link_scan_design(case["id"], scan["id"], "design123", {"rows": [{"source": "ACL width"}]}, "mri_derived")
    rec = ST.load(case["id"])["scans"][scan["id"]]
    assert rec["linked_designs"][0]["design_id"] == "design123"
    assert rec["linked_designs"][0]["personalization_status"] == "mri_derived"
