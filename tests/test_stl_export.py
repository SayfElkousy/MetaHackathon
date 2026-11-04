"""Selection-aware STL export and scale/geometry regression tests."""
import io
import json
import zipfile

import numpy as np
import pytest

from acl_engine.v2.design import new_state
from acl_engine.v2.scene import build_scene
from acl_engine.v2.stl_export import build_export


def scene(state):
    out = build_scene(state, "full")
    assert out["ok"]
    return out


def stl_triangles(data):
    n = int.from_bytes(data[80:84], "little")
    rec = np.frombuffer(data[84:], dtype=[("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    assert len(rec) == n and len(data) == 84 + 50 * n
    return rec["v"].astype(float)


def test_complete_export_is_mm_and_excludes_anatomy_and_display_helpers():
    st = new_state(); summary, data = build_export(st, scene(st), {"revision": 7})
    ids = {x["id"] for x in summary["components"]}
    assert "layer_sheath" in ids and "layer_channels_am" in ids and "layer_braid_pl" in ids
    assert not ids.intersection({"bone_femur", "bone_tibia", "intact_reference", "retained_tibial_stump", "layer_anchors", "layer_interface"})
    assert summary["units"] == "mm" and summary["revision"] == 7
    assert summary["manufacturing"]["message"] == "Geometry exported. Manufacturing suitability has not been assessed."
    assert all(0 < x < 80 for x in summary["bounds_mm"]["dimensions"])
    omitted = {x["id"] for x in summary["omitted_display_only_features"]}
    assert {"layer_grooves", "layer_interface", "layer_anchors"} <= omitted
    z = zipfile.ZipFile(io.BytesIO(data)); tri = stl_triangles(z.read("stl/scaffold_assembly.stl"))
    assert np.isfinite(tri).all() and np.allclose(np.ptp(tri.reshape(-1, 3), axis=0), summary["bounds_mm"]["dimensions"], atol=1e-4)


def test_known_custom_band_scale_and_transform():
    st = new_state(); st["custom_parts"] = [{"id": "scale_check", "name": "Scale check", "operation": "band", "level": 6,
        "dimensions": {"width_mm": 2, "length_mm": 10, "thickness_mm": 1}, "position_mm": [4, 5, 6], "rotation_deg": [0, 0, 0],
        "relationship": {"mode": "independent", "target": None}, "material": "PCL"}]
    summary, data = build_export(st, scene(st), {"scope": "custom", "mode": "separate"})
    assert summary["bounds_mm"]["dimensions"] == pytest.approx([2, 10, 1])
    assert summary["bounds_mm"]["min"] == pytest.approx([3, 0, 5.5])
    z = zipfile.ZipFile(io.BytesIO(data)); tri = stl_triangles(z.read("stl/custom_scale_check.stl"))
    assert np.ptp(tri.reshape(-1, 3), axis=0) == pytest.approx([2, 10, 1])


def test_hollow_channels_and_braid_are_real_separate_geometry():
    st = new_state(); sc = scene(st)
    summary, data = build_export(st, sc, {"scope": "components", "component_ids": ["core_am", "mantle_am"], "mode": "separate"})
    ids = {x["id"] for x in summary["components"]}
    assert ids == {"layer_channels_am", "layer_braid_am"}
    channel = next(x for x in summary["validation"]["components"] if x["id"] == "layer_channels_am")
    assert next(x for x in channel["checks"] if x["check"] == "watertight material boundary")["status"] == "passed"
    z = zipfile.ZipFile(io.BytesIO(data)); assert {"stl/layer_channels_am.stl", "stl/layer_braid_am.stl"} <= set(z.namelist())
    assert stl_triangles(z.read("stl/layer_channels_am.stl")).shape[0] > 100


def test_exploded_display_does_not_change_export_geometry():
    a = new_state(); b = json.loads(json.dumps(a)); b.setdefault("display", {})["exploded"] = 25
    sa, _ = build_export(a, scene(a), {}, preview=True); sb, _ = build_export(b, scene(b), {}, preview=True)
    assert sa["geometry_hash"] == sb["geometry_hash"]
    assert sa["bounds_mm"] == sb["bounds_mm"] and sa["triangle_count"] == sb["triangle_count"]


def test_separate_parts_manifest_and_current_parameter_snapshot():
    st = new_state(); st["params"]["sheath.wall_mm"] = 0.31
    summary, data = build_export(st, scene(st), {"scope": "levels", "levels": [5, 6], "mode": "separate", "revision": 12})
    z = zipfile.ZipFile(io.BytesIO(data)); stls = [x for x in z.namelist() if x.endswith(".stl")]
    assert len(stls) == len(summary["components"]) and len(stls) >= 3
    assert json.loads(z.read("design.json"))["params"]["sheath.wall_mm"] == 0.31
    manifest = json.loads(z.read("component_manifest.json"))
    assert manifest["revision"] == 12 and manifest["mode"] == "separate"


def test_nonfinite_custom_geometry_is_blocked_and_fused_is_honestly_unsupported():
    st = new_state(); st["custom_parts"] = [{"id": "bad", "name": "Bad", "operation": "tube", "dimensions": {"length_mm": 2, "outer_diameter_mm": 1},
        "position_mm": [float("nan"), 0, 0], "rotation_deg": [0, 0, 0], "relationship": {"mode": "independent"}}]
    with pytest.raises(ValueError, match="blocked"):
        build_export(st, scene(st), {"scope": "custom"})
    good = new_state()
    with pytest.raises(NotImplementedError, match="Boolean"):
        build_export(good, scene(good), {"mode": "fused"})


def test_process_profile_is_only_compared_to_exact_supported_dimensions():
    st = new_state(); summary, _ = build_export(st, scene(st), {"scope": "levels", "levels": [3, 5],
        "process_profile": {"minimum_wall_mm": 0.5, "minimum_opening_mm": 0.1}}, preview=True)
    checks = summary["manufacturing"]["checks"]
    assert any(x["feature"] == "sheath wall" for x in checks)
    assert any(x["feature"] == "channel lumen" for x in checks)
    assert "not proof" in summary["manufacturing"]["message"]
