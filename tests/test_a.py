"""Milestone A tests: anatomical ACL bundles, twist, ribbon, footprints, tear."""
# Marked PENDING_RUN in the milestone report (rule 7); the grader runs check.py.
import json
import math
import numpy as np
from acl_engine.schema import validate_request
from acl_engine.context import (build_native_axis, build_envelope_fields,
                                basis_for_dir)
from acl_engine.anatomy import (build_bundle_fields, build_bundle_meshes,
                                max_bundle_radius, twist_angle,
                                ribbon_aspect_scale)
from acl_engine.footprints import (tibial_footprint_mesh,
                                   femoral_footprint_mesh, femoral_wall_x,
                                   femoral_crescent_coverage, TIBIAL_OVAL_ML,
                                   TIBIAL_OVAL_AP, FEMORAL_R_OUT, FEMORAL_R_IN)
from acl_engine.tear import (TEAR_PRESETS, apply_tear_preset, tear_defaults,
                             build_stumps, is_watertight)
from acl_engine.sweep import signed_volume

FIXTURES = ["fixtures/starter_request.json",
            "fixtures/pass_request.json",
            "fixtures/starter_fiberlength.json",
            "fixtures/acl_context_request.json"]


def _anat_fields(twist=0.0, dist="uniform", side="right"):
    T, F, u = build_native_axis(32.0, 47.4, 75.3, side)
    prof = [34.0, 33.0, 35.0, 38.0, 42.0]
    ss, _A, Ws, Ts, _Am = build_envelope_fields(32.0, 10.0, 5.0, prof, 3.5)
    e1, _eu, e3 = basis_for_dir(u, side)
    fld = build_bundle_fields(T, u, e1, e3, ss, Ws, Ts, twist, dist)
    return T, F, u, fld


def test_a_realism_defaults_schematic():
    for fx in FIXTURES:
        with open(fx) as f:
            norm, _ = validate_request(json.load(f))
        assert norm["_derived"]["realism"] == "schematic"
        assert norm["_derived"]["anatomical_twist_total_deg"] == 0.0
        assert norm["tear"]["severity"] == "intact"
    with open("fixtures/starter_request.json") as f:
        norm, _ = validate_request(json.load(f))
    assert abs(norm["_derived"]["acl_length_mm"] - 30.0) < 1e-9


def test_a_bundles_inside_acl():
    for twist in (0.0, 30.0, 90.0, -60.0):
        _T, _F, _u, fld = _anat_fields(twist=twist)
        assert max_bundle_radius(fld) <= 1.0
    _T, _F, _u, fld = _anat_fields()
    meshes = build_bundle_meshes(fld, nsec=32)
    assert set(meshes) == {"AM", "PL"}
    for V, Fc in meshes.values():
        assert np.all(np.isfinite(V))
        assert V.shape[0] == 241 * 32


def test_a_twist_distributions():
    L = 32.0
    s = np.array([0.0, 0.5 * L, L])
    uni = twist_angle(s, L, 90.0, "uniform")
    assert abs(uni[0]) < 1e-12
    assert abs(uni[1] - math.radians(45.0)) < 1e-12
    assert abs(uni[2] - math.radians(90.0)) < 1e-12
    fem = twist_angle(s, L, 90.0, "femoral_weighted")
    tib = twist_angle(s, L, 90.0, "tibial_weighted")
    assert fem[1] < uni[1] < tib[1]  # femoral turns late, tibial turns early
    assert abs(fem[2] - tib[2]) < 1e-12
    assert np.all(twist_angle(s, L, 0.0, "uniform") == 0.0)


def test_a_ribbon_shape_area_preserving():
    L = 32.0
    sc = ribbon_aspect_scale(np.array([0.0, 0.5 * L, L]), L)
    assert abs(sc[2] - 1.40) < 1e-12  # flat/wide near femur
    assert abs(sc[1] - 0.80) < 1e-12  # rounder mid-substance
    assert abs(sc[0] - 1.15) < 1e-12  # tibial fan
    _T, _F, _u, fld = _anat_fields()
    prof = [34.0, 33.0, 35.0, 38.0, 42.0]
    _ss, _A, Ws, Ts, _Am = build_envelope_fields(32.0, 10.0, 5.0, prof, 3.5)
    assert np.allclose(fld["Ws"] * fld["Ts"], Ws * Ts)


def test_a_footprints_placed_and_closed():
    T, F, _u = build_native_axis(32.0, 47.4, 75.3, "right")[:3]
    Vt, Ft = tibial_footprint_mesh(T)
    assert abs(Vt[:, 2].max() - 0.4) < 1e-9
    assert abs(Vt[:, 0].mean() - T[0]) < 0.5
    assert abs((Vt[:, 0].max() - Vt[:, 0].min()) - TIBIAL_OVAL_ML) < 0.5
    assert abs((Vt[:, 1].max() - Vt[:, 1].min()) - TIBIAL_OVAL_AP) < 0.5
    assert is_watertight(Vt, Ft)[0]
    assert abs(signed_volume(Vt, Ft) - math.pi * 5.5 * 8.5 * 0.4) / 58.0 < 0.05
    Vf, Ff = femoral_footprint_mesh(F, "right")
    assert abs(Vf[:, 0].max() - femoral_wall_x(F, "right")) < 1e-9
    r = np.sqrt((Vf[:, 1] - (F[1] + (FEMORAL_R_OUT + FEMORAL_R_IN) / 2.0)) ** 2
                + (Vf[:, 2] - F[2]) ** 2)
    assert r.min() >= FEMORAL_R_IN - 1e-9
    assert r.max() <= FEMORAL_R_OUT + 1e-9
    assert is_watertight(Vf, Ff)[0]
    assert signed_volume(Vf, Ff) > 0
    Vl, _Fl = femoral_footprint_mesh(
        build_native_axis(32.0, 47.4, 75.3, "left")[1], "left")
    assert Vl[:, 0].min() >= femoral_wall_x(
        build_native_axis(32.0, 47.4, 75.3, "left")[1], "left") - 1e-9
    cov = femoral_crescent_coverage(8.0, 6.4, F, "right")
    assert 0.05 < cov < 0.8


def test_a_tear_presets():
    assert set(TEAR_PRESETS) == {"intact", "proximal_complete",
                                 "midsubstance_complete", "partial_AM"}
    for name, want in (("proximal_complete", ("complete", 0.80)),
                       ("midsubstance_complete", ("complete", 0.50)),
                       ("partial_AM", ("partial", 0.55))):
        t = tear_defaults(apply_tear_preset({"preset": name}))
        assert (t["severity"], t["location_s"]) == want
        assert t["status"] == "anatomical_a"
    t = tear_defaults(apply_tear_preset({"preset": "partial_AM"}))
    assert t["involvement"] == "AM"
    # explicit fields win over the preset
    t = tear_defaults(apply_tear_preset({"preset": "midsubstance_complete",
                                         "location_s": 0.3}))
    assert t["location_s"] == 0.3
    try:
        apply_tear_preset({"preset": "nope"})
    except ValueError:
        pass
    else:
        raise AssertionError("unknown preset must raise")


def test_a_stump_gap_matches_retraction():
    _T, _F, _u, fld = _anat_fields()
    meshes = build_bundle_meshes(fld, nsec=32)
    mk = lambda rf, rt: {"severity": "complete", "location_s": 0.5,
                         "obliquity_deg": 0.0, "azimuth_deg": 0.0,
                         "retraction_femoral_mm": rf, "retraction_tibial_mm": rt}
    g1 = build_stumps(fld, meshes, mk(3.0, 3.0))["gap_mm"]
    assert abs(g1 - 6.0) <= 1.0  # within raggedness of the nominal gap
    g2 = build_stumps(fld, meshes, mk(4.0, 4.0))["gap_mm"]
    assert abs((g2 - g1) - 2.0) < 0.05  # gap follows retraction exactly


def test_a_stump_cut_faces_watertight():
    _T, _F, _u, fld = _anat_fields()
    meshes = build_bundle_meshes(fld, nsec=32)
    for tear in ({"preset": "proximal_complete"},
                 {"preset": "midsubstance_complete"},
                 {"preset": "partial_AM"},
                 {"severity": "complete", "location_s": 0.3,
                  "obliquity_deg": 45.0, "azimuth_deg": 90.0,
                  "retraction_femoral_mm": 2.0, "retraction_tibial_mm": 2.0}):
        info = build_stumps(fld, meshes, tear)
        assert info["watertight"]["tibial"] and info["watertight"]["femoral"]
        assert info["gap_mm"] > 0
    try:
        build_stumps(fld, meshes, {"severity": "complete", "location_s": 0.02,
                                   "obliquity_deg": 60.0, "azimuth_deg": 0.0})
    except ValueError:
        pass
    else:
        raise AssertionError("plane missing the wall must raise")


def test_a_partial_am_keeps_pl_intact():
    _T, _F, _u, fld = _anat_fields()
    meshes = build_bundle_meshes(fld, nsec=32)
    info = build_stumps(fld, meshes, {"preset": "partial_AM"})
    assert info["cut_bundles"] == ["AM"]
    assert info["intact_bundles"] == ["PL"]


def test_a_full_anatomical_build(tmp_path):
    from cli import run_request
    from tools.render import load_glb
    with open("fixtures/acl_context_request.json") as f:
        req = json.load(f)
    req["anatomy"]["realism"] = "anatomical"
    req["tear"] = {"preset": "midsubstance_complete"}
    norm, mig = validate_request(req)
    out = str(tmp_path / "anat")
    res = run_request(norm, mig, "preview", out, skip_stl=True)
    assert res["pen"] is not None
    with open(out + "/context.json") as f:
        ctx = json.load(f)
    assert ctx["realism"] == "anatomical"
    assert ctx["anatomical"]["tear"]["gap_mm"] > 0
    assert ctx["anatomical"]["bundles"]["AM"]["length_mm"] > 0
    with open(out + "/validation.json") as f:
        val = json.load(f)
    ids = {c["check_id"]: c["status"] for c in val["checks"]}
    assert ids.get("bundle_inside_acl") == "pass"
    assert ids.get("stump_closure") == "pass"
    M = load_glb(out + "/knee.glb")
    for name in ("acl_ghost_am", "acl_ghost_pl", "footprint_femoral",
                 "footprint_tibial", "stump_femoral", "stump_tibial"):
        assert name in M, name


def test_a_schematic_build_has_no_new_checks(tmp_path):
    from cli import run_request
    with open("fixtures/acl_context_request.json") as f:
        req = json.load(f)
    norm, mig = validate_request(req)
    out = str(tmp_path / "schem")
    run_request(norm, mig, "preview", out, skip_stl=True)
    with open(out + "/context.json") as f:
        ctx = json.load(f)
    assert ctx["realism"] == "schematic"
    assert "anatomical" not in ctx
    with open(out + "/validation.json") as f:
        val = json.load(f)
    ids = {c["check_id"] for c in val["checks"]}
    assert "bundle_inside_acl" not in ids
    assert "stump_closure" not in ids
