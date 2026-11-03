"""Six-level layered construct: editing structure, geometry connections, conflicts."""
import copy
import io
import json
import os
import sys
import zipfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from acl_engine.v2 import design as D, mesher  # noqa: E402
from acl_engine.v2.scene import build_scene, verify_layer_fixes  # noqa: E402
from acl_engine.v2.layered import hollow_tube_mesh  # noqa: E402
from acl_engine.v2.export import export_zip  # noqa: E402
from acl_engine.v2 import sections as secmod  # noqa: E402


def state(**upd):
    st = D.new_state()
    st["params"].update(upd)
    return st


def obj(sc, oid):
    return next((o for o in sc["objects"] if o["id"] == oid), None)


def conflicts(sc):
    return {c["id"]: c for c in sc["checks"] if c.get("category") == "layer"}


@pytest.fixture(scope="module")
def base():
    st = state()
    return st, build_scene(st, "full")


# ---------------------------------------------------------------- editing structure
def test_every_scaffold_control_has_exactly_one_level_owner():
    anatomy_prefixes = ("case.", "knee.", "bone.", "fp.", "lig.", "injury.", "native.", "bundle.", "am.", "pl.")
    for p in D.REGISTRY:
        pid = p["id"]
        if pid.startswith(anatomy_prefixes):
            assert p["level"] == 0 and p["group"] in D.ANATOMY_GROUPS, pid        # Anatomy and Pose panel
        else:
            assert p["level"] in range(1, 7) and p["group"] == "lvl%d" % p["level"], pid
            assert p["component"], pid                                            # what it directly modifies
            assert p["basis"] in ("measured", "assumed", "experimental"), pid
            for n, why in p["neighbors"]:
                assert n in range(1, 7) and why, pid
    ui = D.registry_for_ui()
    assert [L["n"] for L in ui["levels"]] == [1, 2, 3, 4, 5, 6]
    assert all(q["panel"] == ("scaffold" if q["owner_level"] else "anatomy") for q in ui["params"])
    # global pose, tear definition and reference anatomy live in Anatomy and Pose
    for pid in ("knee.flexion_deg", "injury.center_s", "fp.tib.center", "lig.width_profile"):
        assert ui and D.BY_ID[pid]["level"] == 0


def test_default_layered_build_is_valid_and_draws_every_level(base):
    st, sc = base
    assert sc["ok"] and sc["validity"]["valid"], sc["validity"]["failed"]
    levels = {o.get("level") for o in sc["objects"] if o["id"].startswith("layer_")}
    assert {1, 2, 3, 4, 5, 6} <= levels
    assert obj(sc, "scaffold_yarns") is None                    # the gap-filling yarns are off
    rep = sc["metrics"]["layered"]
    assert set(rep["summary"]) == {"1", "2", "3", "4", "5", "6"}
    assert rep["footprint_source"]["tib"]["source"] == "default template"


def test_channels_are_genuinely_hollow_with_open_entrances(base):
    st, sc = base
    ch = obj(sc, "layer_channels")
    assert ch["hollow"] and len(ch["lines"]) > 20
    for ln in ch["lines"][:10]:
        assert (ln["ri"] < ln["r"]).all() and (ln["ri"] > 0).all()
        V, F = hollow_tube_mesh(ln["P"], ln["r"], ln["ri"], ln["u"])
        a = mesher.audit(V, F)
        assert a["closed"] and not a["nonmanifold_edges"]
        # annulus volume, not a solid rod
        L = np.linalg.norm(np.diff(ln["P"], axis=0), axis=1).sum()
        rod = np.pi * ln["r"][0] ** 2 * L
        assert a["volume_mm3"] < 0.8 * rod
    g = sc["metrics"]["layered"]["summary"]["3"]
    assert abs(g["outside_um"] - (st["params"]["ch.lumen_um"] + 2 * st["params"]["ch.wall_um"])) < 1e-6
    assert abs(g["pitch_um"] - (g["outside_um"] + st["params"]["ch.spacing_um"])) < 1e-6
    assert g["min_gap_um"] >= st["params"]["ch.spacing_um"] - 20          # measured on the generated paths


def test_coating_reduces_the_open_lumen_only_in_explicit_geometry():
    a = build_scene(state(**{"sf.coating_nm": 5000.0}), "preview")
    b = build_scene(state(**{"sf.coating_nm": 5000.0, "sf.coating_explicit": "metadata"}), "preview")
    ra = obj(a, "layer_channels")["lines"][0]
    rb = obj(b, "layer_channels")["lines"][0]
    lum = 0.5 * D.BY_ID["ch.lumen_um"]["default"] / 1000
    assert abs(ra["ri"].min() - (lum - 0.005)) < 1e-9 and abs(rb["ri"].min() - lum) < 1e-9
    assert abs(a["metrics"]["layered"]["summary"]["1"]["effective_lumen_um"] - 190.0) < 1e-6


# ---------------------------------------------------------------- conflicts and concrete fixes
def _apply_verified(sc, cid):
    fixes = [f for f in verify_layer_fixes(sc) if f["conflict"] == cid]
    assert fixes, cid
    good = [f for f in fixes if f["verified"]]
    assert good, fixes
    return good


def test_larger_channels_need_fewer_channels_or_a_larger_envelope():
    st = state(**{"ch.count_mode": "count", "ch.am_count": 29, "ch.pl_count": 24, "rf.size_mode": "manual"})
    ok = build_scene(st, "preview")
    st2 = copy.deepcopy(st)
    st2["params"]["ch.lumen_um"] = 260.0
    sc = build_scene(st2, "full")
    C = conflicts(sc)
    assert "layer_channels_fit_am" in C and C["layer_channels_fit_am"]["status"] == "fail"
    assert C["layer_channels_fit_am"]["levels"] == [3, 4]
    labels = " | ".join(f["label"] for f in C["layer_channels_fit_am"]["fixes"])
    assert "channel count" in labels and "envelope" in labels                # fewer channels OR a larger envelope
    # requested values are kept: extra channels are drawn (flagged), nothing is shrunk
    req = lambda s_: sum(1 for ln in obj(s_, "layer_channels")["lines"] if ln["group"] in (0, 1))
    assert req(sc) == req(ok) == 29 + 24
    assert any(ln["viol"].any() for ln in obj(sc, "layer_channels")["lines"])
    good = _apply_verified(sc, "layer_channels_fit_am")
    st3 = copy.deepcopy(st2)
    st3["params"].update(good[0]["set"])
    assert "layer_channels_fit_am" not in conflicts(build_scene(st3, "preview"))
    # a fix only sets the parameters it names
    assert set(good[0]["set"]) <= {"ch.am_count", "rf.am.envelope_width_mm", "rf.am.envelope_thickness_mm", "ch.lumen_um", "ch.count_mode"}


def test_thicker_braid_needs_more_sheath_clearance():
    st = state(**{"rf.size_mode": "manual", "rf.am.strand_diameter_mm": 0.45})
    sc = build_scene(st, "full")
    C = conflicts(sc)
    c = C["layer_mantle_sheath_am"]
    assert c["status"] == "fail" and 5 in c["levels"] and 4 in c["levels"] and "clearance" in c["message"]
    good = _apply_verified(sc, "layer_mantle_sheath_am")
    assert any("sheath" in f["label"].lower() for f in good)


def test_deeper_grooves_need_thicker_channel_walls():
    sc = build_scene(state(**{"gr.depth_um": 45.0}), "full")
    c = conflicts(sc)["layer_groove_channel_wall"]
    assert c["status"] == "fail" and c["levels"] == [2, 3] and "thicker channel walls" in c["message"]
    good = _apply_verified(sc, "layer_groove_channel_wall")
    assert any("ch.wall_um" in f["set"] for f in good)


def test_grooves_never_on_braid_and_outer_sheath_smooth_by_default(base):
    st, sc = base
    assert "braid" not in " ".join(D.BY_ID["gr.placement"]["options"])
    assert st["params"]["gr.outer_sheath"] == "smooth"
    where = {ln["where"] for ln in obj(sc, "layer_grooves")["lines"]}
    assert where == {0}                                          # channel lumens only (default placement)
    sc2 = build_scene(state(**{"gr.outer_sheath": "grooved"}), "preview")
    assert "layer_outer_grooves" in conflicts(sc2)


def test_braid_strand_count_must_close_the_pattern():
    sc = build_scene(state(**{"rf.pl.strands": 20, "rf.pl.pattern": "2/2"}), "preview")
    c = conflicts(sc)["layer_braid_count_pl"]
    assert c["status"] == "fail" and any(f["set"].get("rf.pl.strands") in (16, 24) for f in c["fixes"])


# ---------------------------------------------------------------- geometry connections
def _end_pts(sc, oid, side):
    o = obj(sc, oid)
    i = 0 if side == "tib" else -1
    return np.array([ln["P"][i] for ln in o["lines"]])


def test_footprint_edit_propagates_into_channels_braid_and_sheath_and_ends_are_independent(base):
    st, sc = base
    sc2 = build_scene(state(**{"end.tib.offset_long_mm": 2.0}), "full")
    for oid in ("layer_channels", "layer_braid"):
        assert np.abs(_end_pts(sc2, oid, "tib") - _end_pts(sc, oid, "tib")).max() > 0.5          # tibial endpoints moved
        assert np.abs(_end_pts(sc2, oid, "fem") - _end_pts(sc, oid, "fem")).max() < 1e-6         # femoral end untouched
    V0, V1 = obj(sc, "layer_sheath")["V"], obj(sc2, "layer_sheath")["V"]
    assert np.abs(V1 - V0).max() > 0.5
    assert np.abs(obj(sc2, "layer_end_fem")["V"] - obj(sc, "layer_end_fem")["V"]).max() < 1e-6
    assert sc2["metrics"]["layered"]["footprint_source"]["tib"]["source"] == "manual placement"
    assert sc2["metrics"]["layered"]["footprint_source"]["fem"]["source"] == "default template"


def test_am_and_pl_regions_are_edited_independently(base):
    st, sc = base
    sc2 = build_scene(state(**{"rf.pl.strand_diameter_mm": 0.12}), "full")
    am0 = [ln["P"] for ln in obj(sc, "layer_braid")["lines"] if ln["group"] == 0]
    am1 = [ln["P"] for ln in obj(sc2, "layer_braid")["lines"] if ln["group"] == 0]
    pl0 = [ln["r"][0] for ln in obj(sc, "layer_braid")["lines"] if ln["group"] == 1]
    pl1 = [ln["r"][0] for ln in obj(sc2, "layer_braid")["lines"] if ln["group"] == 1]
    assert all(np.allclose(a, b) for a, b in zip(am0, am1))
    assert abs(pl1[0] - 0.06) < 1e-9 and abs(pl0[0] - 0.075) < 1e-9


def test_braid_angle_yarn_twist_and_ligament_rotation_are_independent(base):
    st, sc = base
    braid = lambda s: np.vstack([ln["P"] for ln in obj(s, "layer_braid")["lines"]])
    s_twist = build_scene(state(**{"rf.am.yarn_twist_deg": 25.0}), "preview")
    s0 = build_scene(state(), "preview")
    assert np.allclose(braid(s_twist), braid(s0))                 # yarn twist does not change the braid paths
    s_angle = build_scene(state(**{"rf.am.braid_angle_deg": 35.0}), "preview")
    assert not np.allclose(braid(s_angle), braid(s0))
    assert s_angle["metrics"]["bundle_rotation_total_deg"] == s0["metrics"]["bundle_rotation_total_deg"]
    # the AM/PL arrangement (anatomy) drives the mantle paths; it is not a Level-5 control
    s_rot = build_scene(state(**{"bundle.rotation_total_deg": 40.0}), "preview")
    assert not np.allclose(braid(s_rot), braid(s0))
    assert D.BY_ID["bundle.rotation_total_deg"]["level"] == 0


def test_mantles_stay_continuous_inside_one_sheath_with_shared_transition_channels(base):
    st, sc = base
    groups = {ln["group"] for ln in obj(sc, "layer_channels")["lines"]}
    assert groups == {0, 1, 2}                                     # AM-like, PL-like and shared channels
    assert sc["metrics"]["layered"]["summary"]["4"]["mantle_gap_mm"] >= st["params"]["rf.transition_mm"] - 0.02


# ---------------------------------------------------------------- materials, assumptions, outputs
def test_compositions_are_validated_and_normalisation_is_offered():
    st = state()
    st["params"]["sheath.composition"] = {"fibronectin": 1.0}       # a coating protein is not a sheath material
    sc = build_scene(st, "preview")
    assert not sc["ok"] and sc["errors"][0]["param"] == "sheath.composition"
    sc2 = build_scene(state(**{"end.tib.comp_bone": {"HA": 0.5, "beta_TCP": 0.3}}), "preview")
    c = conflicts(sc2)["layer_comp_end_tib_comp_bone"]
    assert c["status"] == "warn" and abs(sum(c["fixes"][0]["set"]["end.tib.comp_bone"].values()) - 1) < 1e-3


def test_entered_assumptions_are_separate_from_geometry_and_pores():
    a = build_scene(state(), "preview")
    b = build_scene(state(**{"sheath.assumed_permeability_log10": -12.0}), "preview")
    c = build_scene(state(**{"sheath.pores": "explicit"}), "preview")
    assert a["geometry_hash"] == b["geometry_hash"]                  # an assumption is not geometry
    row = lambda s: next(r for r in s["metrics"]["layered"]["calculated"]["5"] if r["label"].startswith("Darcy"))
    assert row(a)["value"] != row(b)["value"] and "assumption" in row(a)["basis"] or "entered" in row(a)["basis"]
    assert row(a)["value"] == row(c)["value"]                        # drawing pores claims no transport
    assert obj(c, "layer_pores") is not None and a["geometry_hash"] != c["geometry_hash"]


def test_sections_cut_channel_annuli_and_braid_strands(base):
    st, sc = base
    sec = secmod.section(sc["objects"], sc["lig"], "ligament", 0.5, sc["side"])
    assert sec["channels"] and all(e["ai"] < e["a"] for e in sec["channels"])
    assert {e["group"] for e in sec["braid"]} == {0, 1}
    assert any(L["category"] == "layer_sheath" for L in sec["layers"])


def test_magnified_detail_is_true_scale_geometry():
    from acl_engine.v2.detail import level_detail
    p = D.default_params()
    r = level_detail(p, 2)
    V = r["parts"][0]["V"]
    rr = np.hypot(V[:, 0], V[:, 1])
    lum, depth = p["ch.lumen_um"] / 2000, p["gr.depth_um"] / 1000
    inner = rr[rr < lum + depth + 1e-6]
    assert abs(inner.min() - lum) < 1e-6 and abs(inner.max() - (lum + depth)) < 2e-4   # groove depth at true scale
    for lv in (1, 3, 4, 5, 6):
        assert level_detail(p, lv)["parts"]


def test_export_contains_level_spec_and_closed_hollow_channels(base):
    st, sc = base
    data, man = export_zip(st, sc)
    z = zipfile.ZipFile(io.BytesIO(data))
    lay = json.loads(z.read("layers.json"))
    assert set(lay["levels"]) == {"1", "2", "3", "4", "5", "6"} and not [c for c in lay["conflicts"] if c["status"] == "fail"]
    assert "stl/layer_channels.stl" in z.namelist() and "stl/detail_grooved_channel_segment.stl" in z.namelist()


def test_left_knee_and_gap_filling_constructs_still_build():
    l = build_scene(state(**{"case.side": "left"}), "preview")
    assert l["ok"] and obj(l, "layer_channels")["lines"][0]["P"][:, 0].mean() < 0
    y = build_scene(state(**{"scaffold.arch": "aligned", "scaffold.yarn_diameter_mm": 0.3}), "preview")
    assert y["ok"] and obj(y, "layer_channels") is None and obj(y, "scaffold_yarns") is not None
    assert "layer_channels" not in [o["id"] for o in build_scene(state(**{"injury.type": "intact"}), "preview")["objects"]]


def test_switching_a_calculated_value_to_user_set_keeps_the_geometry(base):
    st, sc = base
    D_ = sc["metrics"]["layered"]["derived"]
    upd = {"rf.size_mode": "manual", "ch.count_mode": "count"}
    for side in ("tib", "fem"):
        upd.update({"end.%s.flare_mode" % side: "manual", "end.%s.size_mode" % side: "manual"})
        for f in ("flare_width_mm", "flare_thickness_mm", "length_mm", "width_mm"):
            upd["end.%s.%s" % (side, f)] = round(D_["end.%s.%s" % (side, f)], 3)
    for b in ("am", "pl"):
        upd["ch.%s_count" % b] = D_["ch.%s_count" % b]
        for f in ("envelope_width_mm", "envelope_thickness_mm"):
            upd["rf.%s.%s" % (b, f)] = round(D_["rf.%s.%s" % (b, f)], 3)
    sc2 = build_scene(state(**upd), "full")
    assert sc2["validity"]["valid"], sc2["validity"]["failed"]
    for oid in ("layer_end_tib", "layer_end_fem"):
        assert np.abs(obj(sc2, oid)["V"] - obj(sc, oid)["V"]).max() < 0.01        # flare base unchanged (crescent included)
    assert len(obj(sc2, "layer_channels")["lines"]) == len(obj(sc, "layer_channels")["lines"])


def test_femoral_turn_crowding_is_measured_and_staggered_endpoints_resolve_it(base):
    st, sc = base
    assert st["params"]["end.fem.endpoints"] == "staggered"
    info = conflicts(sc)["layer_staggered_fem"]
    assert info["status"] == "info" and "end before" in info["message"]
    g = sc["metrics"]["layered"]["summary"]["3"]["min_gap_um"]
    assert g >= st["params"]["ch.spacing_um"] - 20                     # spacing kept on the generated paths
    fan = build_scene(state(**{"end.fem.endpoints": "fan"}), "full")
    c = conflicts(fan)["layer_channel_overlap"]
    assert c["status"] == "fail" and "femoral flare" in c["message"]
    assert any(f["set"].get("end.fem.endpoints") == "staggered" for f in c["fixes"])
    assert any(f["verified"] for f in verify_layer_fixes(fan) if f["conflict"] == "layer_channel_overlap")
