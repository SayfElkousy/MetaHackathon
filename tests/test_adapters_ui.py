"""Adapters report what is on disk (never guess); interface endpoints serve badges and missing inputs."""
import json

import numpy as np

from bilacl.adapters.base import get_adapter


def test_missing_dataset_reports_blocking(tmp_path):
    rep = get_adapter("oai", tmp_path / "nope").inspect()
    assert not rep.exists and rep.blocking


def test_skm_tea_schema_check(tmp_path):
    import h5py

    root = tmp_path / "skm"
    (root / "annotations" / "v1.0.0").mkdir(parents=True)
    with h5py.File(root / "MTR_001.h5", "w") as f:
        f["echo1"] = np.zeros((4, 4, 4))
        f["seg"] = np.zeros((4, 4, 4, 6))  # echo2 deliberately missing
    (root / "annotations" / "v1.0.0" / "train.json").write_text(json.dumps({"images": [
        {"file_name": "MTR_001.h5", "scan_id": "MTR_001", "subject_id": 7, "voxel_spacing": [0.3125, 0.3125, 0.8], "orientation": ["SI", "AP", "LR"]}]}))
    rep = get_adapter("skm_tea", root).inspect()
    assert any("echo2" in i for i in rep.issues) and not rep.schema_ok
    recs = get_adapter("skm_tea", root).records()
    assert recs[0].official_split == "train" and recs[0].physical_verified == "false"
    assert "voxel:femoral_cartilage" in recs[0].available_labels and "voxel:femur" not in recs[0].available_labels


def test_mrnet_records_have_no_physical_geometry(tmp_path):
    root = tmp_path / "mrnet"
    (root / "train" / "sagittal").mkdir(parents=True)
    np.save(root / "train" / "sagittal" / "0000.npy", np.zeros((20, 256, 256), np.uint8))
    (root / "train-acl.csv").write_text("0000,1\n")
    r = get_adapter("mrnet", root).records()[0]
    assert r.physical_verified == "false" and r.label_granularity == "exam" and "exam:acl=1" in r.available_labels


def test_oai_zib_requires_confirmed_label_map(tmp_path):
    import pytest

    (tmp_path / "z").mkdir()
    rep = get_adapter("oai_zib", tmp_path / "z").inspect()
    assert any("oai_zib.yaml" in b and "confirmed: false" in b for b in rep.blocking)
    with pytest.raises(PermissionError):
        get_adapter("oai_zib", tmp_path / "z").records()


def test_ui_endpoints(syn_case):
    from fastapi.testclient import TestClient

    from bilacl.ui.server import create_app

    ws = syn_case.parent
    c = TestClient(create_app(ws))
    cases = c.get("/api/cases").json()["cases"]
    assert cases and cases[0]["synthetic"]
    d = c.get(f"/api/case/{cases[0]['case_id']}").json()
    assert any("pipeline not yet run" in m for m in d["missing"])
    r = c.get("/api/slice", params={"cid": cases[0]["case_id"], "vol": "injured_image", "axis": 1, "overlay": "injured_labels"})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert c.get("/api/slice", params={"cid": cases[0]["case_id"], "vol": "mirrored_image"}).status_code == 404
    assert c.post(f"/api/case/{cases[0]['case_id']}/review", json={"status": "approved"}).status_code == 400  # reviewer required
    m = c.get("/api/models").json()
    assert m["validated_weights_available"] is False
