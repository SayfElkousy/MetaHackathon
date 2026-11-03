"""Scaffold Builder: units, display/geometry separation, AM/PL independence,
new structural objects (inter-bundle zone, anchors) and the pore threshold flag."""
import copy

import numpy as np
import pytest

from acl_engine.v2.design import (LEVELS, LEVEL0, MM_PER_UM, MM_PER_NM, affects_of,
                                  design_hash, new_state, registry_for_ui, to_mm)
from acl_engine.v2.scene import build_scene


def _state(**kw):
    st = new_state()
    st["params"].update(kw)
    return st


def _obj(sc, oid):
    return next((o for o in sc["objects"] if o["id"] == oid), None)


# ---------------------------------------------------------------- units
def test_unit_conversions_documented():
    assert MM_PER_UM == 0.001
    assert MM_PER_NM == 0.000001
    assert to_mm(200.0, "um") == pytest.approx(0.2)
    assert to_mm(400.0, "nm") == pytest.approx(0.0004)
    assert to_mm(2.5, "mm") == pytest.approx(2.5)
    with pytest.raises(ValueError):
        to_mm(1.0, "deg")


def test_channel_lumen_uses_mm_conversion():
    # 200 um lumen + 50 um walls -> 0.3 mm outside width in the built channels
    sc = build_scene(_state(), "preview")
    assert sc["ok"]
    ln = _obj(sc, "layer_channels")["lines"][0]
    assert float(np.asarray(ln["r"]).mean()) == pytest.approx(0.15, abs=1e-9)


# ---------------------------------------------------------------- display vs geometry
def test_display_changes_do_not_change_design_hash():
    a = _state()
    b = copy.deepcopy(a)
    b["display"]["exploded"] = 12.0
    b["display"]["crimp_exaggeration"] = 60.0
    b["display"]["fiber_mode"] = "lines"
    assert design_hash(a) == design_hash(b)
    c = copy.deepcopy(a)
    c["params"]["ch.lumen_um"] = 250.0
    assert design_hash(a) != design_hash(c)


def test_affects_classification():
    assert affects_of("ch.lumen_um") == "geometry"
    assert affects_of("rf.am.braid_angle_deg") == "geometry"
    assert affects_of("crimp_exaggeration") == "visualization"
    assert affects_of("fiber_mode") == "visualization"
    assert affects_of("ch.wall_composition") == "metadata"
    assert affects_of("regions.bridge.material") == "metadata"
    assert affects_of("rf.am.assumed_modulus_mpa") == "metadata"


# ---------------------------------------------------------------- AM / PL independence
def test_registry_am_pl_params_are_distinct():
    reg = registry_for_ui()
    by_id = {p["id"]: p for p in reg["params"]}
    assert by_id["rf.am.strand_diameter_mm"]["default"] == by_id["rf.pl.strand_diameter_mm"]["default"]
    assert "rf.am.strand_diameter_mm" != "rf.pl.strand_diameter_mm"


def test_editing_am_preserves_pl_geometry():
    base = build_scene(_state(), "preview")
    assert base["ok"]
    st = _state(**{"rf.am.strand_diameter_mm": 0.30})
    assert st["params"]["rf.pl.strand_diameter_mm"] == 0.15
    sc = build_scene(st, "preview")
    assert sc["ok"]
    r0 = lambda s, g: [float(ln["r"][0]) for ln in _obj(s, "layer_braid")["lines"] if ln["group"] == g]
    assert r0(sc, 0) and all(v == pytest.approx(0.15) for v in r0(sc, 0))
    assert r0(sc, 1) == r0(base, 1)


# ---------------------------------------------------------------- levels and objects
def test_level_names_and_objects():
    by_n = {L["n"]: L for L in LEVELS}
    assert by_n[5]["name"] == "Sheath and inter-bundle zone"
    assert by_n[6]["name"] == "Insertions and anchors"
    assert "layer_interface" in by_n[5]["objects"]
    assert "layer_anchors" in by_n[6]["objects"]
    assert LEVEL0["name"] == "Patient geometry"
    reg = registry_for_ui()
    assert reg["level0"]["name"] == "Patient geometry"
    assert all("affects" in p for p in reg["params"])


def test_interface_band_object():
    sc = build_scene(_state(), "preview")
    assert sc["ok"]
    o = _obj(sc, "layer_interface")
    assert o is not None and o["level"] == 5 and o["category"] == "layer_interface"
    assert o["V"].shape[1] == 3 and o["F"].shape[1] == 3 and len(o["F"]) > 0
    # the band is thin across the interface and spans the sheath interior along it
    span = o["V"].max(0) - o["V"].min(0)
    assert span.max() > 2.0
    assert "No wall" in o["note"]


def test_interface_band_absent_when_transition_zero():
    sc = build_scene(_state(**{"rf.transition_mm": 0.0}), "preview")
    assert sc["ok"]
    assert _obj(sc, "layer_interface") is None


def test_anchors_object_with_provenance():
    sc = build_scene(_state(), "preview")
    assert sc["ok"]
    o = _obj(sc, "layer_anchors")
    assert o is not None and o["level"] == 6 and o["category"] == "layer_anchors"
    assert len(o["markers"]) == 4
    ids = sorted(m["id"] for m in o["markers"])
    assert ids == ["anchor_fem_am", "anchor_fem_pl", "anchor_tib_am", "anchor_tib_pl"]
    for m in o["markers"]:
        assert len(m["P"]) == 3 and all(np.isfinite(m["P"]))
        assert "fp." in m["provenance"] and "end." in m["provenance"]
    tib = [m for m in o["markers"] if m["end"] == "tibial"]
    fem = [m for m in o["markers"] if m["end"] == "femoral"]
    assert np.mean([m["P"][2] for m in fem]) > np.mean([m["P"][2] for m in tib]) + 5.0


def test_anchors_mirrored_on_left_knee():
    sc = build_scene(_state(**{"case.side": "left"}), "preview")
    assert sc["ok"]
    o = _obj(sc, "layer_anchors")
    sr = build_scene(_state(**{"case.side": "right"}), "preview")
    ro = _obj(sr, "layer_anchors")
    for m, r in zip(sorted(o["markers"], key=lambda x: x["id"]), sorted(ro["markers"], key=lambda x: x["id"])):
        assert m["P"][0] == pytest.approx(-r["P"][0], abs=1e-6)
        assert m["P"][1:] == pytest.approx(r["P"][1:], abs=1e-6)


# ---------------------------------------------------------------- checks
def test_inter_strand_pore_threshold_flag():
    st = _state(**{"rf.am.strands": 64, "rf.am.strand_diameter_mm": 0.25})
    sc = build_scene(st, "preview")
    assert sc["ok"]
    c = {x["id"]: x for x in sc["checks"]}
    assert c["layer_braid_pore_am"]["status"] == "warn"
    assert c["layer_braid_pore_am"]["threshold"] == pytest.approx(0.15)
    assert "pl" not in "".join(k for k in c if "pore" in k)


def test_visualization_objects_excluded_from_stl_but_in_payload():
    from acl_engine.v2.export import export_zip, payload
    sc = build_scene(_state(), "full")
    assert sc["ok"]
    data, man = export_zip(_state(), sc)
    import io
    import zipfile
    names = zipfile.ZipFile(io.BytesIO(data)).namelist()
    assert "stl/layer_interface.stl" not in names
    assert "stl/layer_anchors.stl" not in names
    assert "stl/layer_channels.stl" in names
    pl = payload(sc)
    am = next(o for o in pl["objects"] if o["id"] == "layer_anchors")
    assert len(am["markers"]) == 4
