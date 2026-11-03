"""Parametric loft anatomy: centerline, sections, footprints, strands,
rotation channels, flexion pose, mesh audits, and payloads."""
import json
import math
import numpy as np
from acl_engine.schema import validate_request
from acl_engine.context import (build_native_axis, build_envelope_fields,
                                basis_for_dir)
from acl_engine.loft import (catmull_rom, resample_arclength, smooth_profile,
                             superellipse_map, bundle_weights, section_area,
                             flexion_matrix, apply_flexion)
from acl_engine.attachments import (Footprint, spiral_disk, split_weights,
                                    ellipsoid_ray_x)
from acl_engine.metrics import (audit_mesh, self_intersections,
                                bone_penetration, regen_hash)
from acl_engine.anatomy_build import (build_anatomy, build_loft_tear,
                                      anatomy_payload, validation_report)
from acl_engine.sweep import signed_volume


def _base(side="right", **kw):
    req = json.load(open("fixtures/acl_context_request.json"))
    req["anatomy"]["realism"] = "anatomical"
    req["anatomy"]["side"] = side
    for k, v in kw.items():
        req[k] = v
    norm, _mig = validate_request(req)
    d = norm["_derived"]
    T, F, u = build_native_axis(d["acl_length_mm"], d["sagittal_deg"],
                                d["coronal_deg"], side)
    ss, _A, Ws, Ts, _Am = build_envelope_fields(
        d["acl_length_mm"], d["envelope_width_mm"], d["envelope_thickness_mm"],
        d["area_profile_femur_to_tibia"], d["flare_ratio"], 8.0, False)
    e1, _eu, e3 = basis_for_dir(u, side)
    return norm, T, F, u, e1, e3, ss, Ws, Ts


def _anat(quality="preview", **kw):
    norm, T, F, u, e1, e3, ss, Ws, Ts = _base(**kw)
    side = kw.get("side", "right")
    A = build_anatomy(norm, T, F, u, e1, e3, ss, Ws, Ts, side, quality=quality)
    return A, norm


def test_acl_centerline_curved_and_longer_than_chord():
    A, _n = _anat()
    assert A["loft"]["length_mm"] > A["attach_distance_mm"] > 30.0
    # curved: mid centerline deviates from the endpoint chord
    C = A["loft"]["center"]
    chord = C[-1] - C[0]
    rel = C - C[0]
    dev = np.linalg.norm(rel - np.outer(rel @ chord, chord) / (chord @ chord), axis=1)
    assert dev.max() > 0.5


def test_acl_frames_continuous_no_flips():
    A, _n = _anat()
    assert A["loft"]["continuity"] >= 0.999


def test_acl_sections_vary_and_area_sane():
    A, _n = _anat()
    w = A["loft"]["width"]
    assert w.max() - w.min() > 2.0  # flared roots vs body
    a = A["section_area"]
    assert 25.0 < a.min() < 45.0  # midsubstance near the 35 mm2 profile
    assert a.max() > 60.0  # insertion flare
    assert abs(section_area(np.array([10.0]), np.array([5.0]), np.array([1.0]))[0]
               - math.pi * 5.0 * 2.5) < 1e-9  # ellipse closed form


def test_acl_surface_closed_and_volumed():
    A, _n = _anat()
    assert A["audits"]["acl_surface"]["pass"]
    assert A["volume_mm3"] > 500.0
    assert A["volume_mm3"] < 6000.0


def test_acl_strands_inside_and_distributed():
    A, _n = _anat()
    assert A["strand_max_rn"] <= 0.86
    assert A["endpoints_in_domain"]
    # endpoints spread across the footprint, not one point
    for key in ("AM", "PL"):
        P = A["tib_3d"][key]
        assert np.linalg.norm(P - P.mean(axis=0), axis=1).max() > 2.0
    L = np.concatenate([A["strand_lengths"]["AM"], A["strand_lengths"]["PL"]])
    assert L.min() > 20.0 and L.max() < 60.0


def test_acl_footprints_on_bone():
    A, _n = _anat()
    assert A["endpoint_misses"] == {"AM": 0, "PL": 0}
    assert 100.0 < A["footprint_areas"]["tibial"] < 200.0
    assert 100.0 < A["footprint_areas"]["femoral"] < 250.0
    sub = A["footprint_areas"]["tibial_sub"]
    assert abs(sub["AM"] + sub["PL"] - A["footprint_areas"]["tibial"]) < 5.0
    # tibial endpoints exactly on the plateau plane
    for key in ("AM", "PL"):
        assert np.all(np.abs(A["tib_3d"][key][:, 2]) < 1e-9)
    # femoral endpoints on the (extension) condyle ellipsoid
    C = np.array([A["condyle_lat"][0], -5.0, 25.0])
    R = np.array([12.0, 18.0, 16.0])
    for key in ("AM", "PL"):
        q = (((A["fem_3d"][key] - C) / R) ** 2).sum(axis=1)
        assert np.all(np.abs(q - 1.0) < 1e-6)


def test_acl_bundle_masks_partition_unity():
    phi = np.linspace(0, 2 * math.pi, 64, endpoint=False)
    t = np.linspace(0, 1, 10)
    for sharp in (0.0, 0.35, 1.0):
        wam, wpl = bundle_weights(phi, t, 90.0 + 30.0 * t, sharpness=sharp)
        assert np.allclose(wam + wpl, 1.0)
        assert np.all((wam >= 0.0) & (wam <= 1.0))


def test_acl_flexion_poses_femoral_side():
    A0, _n = _anat()
    A1, _n = _anat(knee_scene={"type": "schematic", "flexion_deg": 45.0,
                               "tibial_slope_deg": 7})
    assert abs(A1["flexion"]["deg"] - 45.0) < 1e-12
    assert A1["flexion"]["illustrative"] is True
    # femoral centerline end + patch posed rigidly from extension
    # (endpoint SETS are resampled under the rotated bundle split, so only
    # rigidly-mapped quantities are compared here)
    ax = np.array(A1["flexion"]["axis_point_mm"])
    e0 = A0["loft"]["center"][-1]
    e1 = A1["loft"]["center"][-1]
    assert np.allclose(e1, apply_flexion(e0[None, :], 45.0, ax)[0], atol=1e-9)
    p0 = np.asarray(A0["femoral_patch"][0]).mean(axis=0)
    p1 = np.asarray(A1["femoral_patch"][0]).mean(axis=0)
    assert np.allclose(p1, apply_flexion(p0[None, :], 45.0, ax)[0], atol=1e-9)
    # tibial side unmoved; length responds to the pose
    assert np.allclose(A0["tib_3d"]["AM"], A1["tib_3d"]["AM"])
    assert abs(A1["loft"]["length_mm"] - A0["loft"]["length_mm"]) > 0.1
    # boundary channel gains the illustrative flexion coupling
    assert A1["boundary_total_deg"] != A0["boundary_total_deg"]


def test_acl_deterministic_regen():
    A1, _n = _anat()
    A2, _n = _anat()
    assert A1["regen_hash"] == A2["regen_hash"]
    assert spiral_disk(10).shape == (10, 2)
    assert np.allclose(spiral_disk(10), spiral_disk(10))


def test_acl_loft_tear_separate_and_labeled():
    A, _n = _anat()
    before = A["surface"][0].copy()
    tr = build_loft_tear(A, {"preset": "midsubstance_complete"})
    assert tr["injury_source"] == "synthetic_example"
    assert tr["watertight"]["tibial"] and tr["watertight"]["femoral"]
    assert tr["gap_mm"] > 0
    # reference surface untouched by the injury build
    assert np.array_equal(A["surface"][0], before)
    tr2 = build_loft_tear(A, {"preset": "midsubstance_complete"}, observed=True)
    assert tr2["injury_source"] == "observed_residual"
    trp = build_loft_tear(A, {"preset": "partial_AM"})
    assert trp["cut_bundles"] == ["AM"]
    assert trp["strand_segments"]["PL"]["spanning"] is True
    assert trp["strand_segments"]["AM"]["spanning"] is False


def test_acl_mesh_audit_and_intersections():
    from acl_engine.context import box_mesh
    V1, F1 = box_mesh(np.zeros(3), np.ones(3) * 4)
    V2, F2 = box_mesh(np.ones(3), np.ones(3) * 4)
    V = np.concatenate([V1, V2])
    F = np.concatenate([F1, F2 + 8])
    assert self_intersections(V, F)[0] > 0  # positive control
    Vb, _Fb = box_mesh(np.array([10.0, 0.0, 0.0]), np.ones(3) * 4)
    assert self_intersections(np.concatenate([V1, Vb]), F)[0] == 0
    A, _n = _anat(quality="export")
    assert A["audits"]["acl_surface"]["pass"]
    assert A["self_intersections"]["count"] == 0
    assert A["bone_penetration"]["fraction"] <= 0.02


def test_acl_full_build_payloads_and_checks(tmp_path):
    from cli import run_request
    from tools.render import load_glb
    req = json.load(open("fixtures/acl_context_request.json"))
    req["anatomy"]["realism"] = "anatomical"
    req["tear"] = {"preset": "midsubstance_complete"}
    norm, mig = validate_request(req)
    out = str(tmp_path / "full")
    res = run_request(norm, mig, "preview", out, skip_stl=True)
    assert res["pen"] is not None
    an = json.load(open(out + "/anatomy.json"))
    assert an["label"].startswith("illustrative")
    assert an["regen_check"] == "pass"
    assert an["pose"]["illustrative"] is True
    assert len(an["assumptions"]) > 3
    rep = json.load(open(out + "/validation_report.json"))
    assert rep["failed_checks"] == []
    assert any(c["check"] == "overlap_vs_ground_truth" for c in rep["not_performed"])
    val = json.load(open(out + "/validation.json"))
    ids = {c["check_id"]: c["status"] for c in val["checks"]}
    for cid in ("anatomy_regen", "attachment_on_bone", "frame_continuity",
                "mesh_valid_anatomy", "bone_penetration_midbody"):
        assert ids.get(cid) == "pass", cid
    M = load_glb(out + "/knee.glb")
    for name in ("acl_surface", "acl_strands_am", "acl_strands_pl",
                 "acl_edge_tibial", "acl_edge_femoral"):
        assert name in M, name


def test_acl_user_controls_override():
    A, _n = _anat(acl_model={"strand_count": 8, "bundle_sharpness": 0.9,
                             "swirl_total_deg": 10.0})
    assert A["strand_paths"]["AM"].shape[0] + A["strand_paths"]["PL"].shape[0] == 8
    try:
        _anat(acl_model={"strand_count": 3})
    except ValueError:
        pass
    else:
        raise AssertionError("strand_count=3 must raise")
    try:
        _anat(acl_model={"nope": 1})
    except ValueError:
        pass
    else:
        raise AssertionError("unknown acl_model key must raise")


def test_acl_catmull_rom_and_profiles():
    P = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [2.0, 0.0, 0.0]])
    C = catmull_rom(P, per_seg=10)
    assert np.allclose(C[0], P[0]) and np.allclose(C[-1], P[-1])
    R, L = resample_arclength(C, 25)
    assert R.shape == (25, 3) and L > 2.0
    seg = np.linalg.norm(np.diff(R, axis=0), axis=1)
    assert seg.max() - seg.min() < 0.05  # uniform by arc length
    v = smooth_profile([0.0, 1.0], [2.0, 4.0], np.array([0.0, 0.5, 1.0]))
    assert np.allclose(v, [2.0, 3.0, 4.0])
    x, y = superellipse_map(np.ones(4), np.array([0.0, 1.0, 2.0, 3.0]), 5.0, 2.5, 1.0)
    assert x.shape == (4,) and abs(x[0] - 5.0) < 1e-12
    R = flexion_matrix(90.0)
    assert np.allclose(R @ R.T, np.eye(3))
    wam, wpl = split_weights(np.array([0.0, math.pi]), 0.0)
    assert wam[0] > 0.9 and wpl[1] > 0.9
    x, hit = ellipsoid_ray_x(np.array([[0.0, 0.0]]), np.zeros(3), np.ones(3) * 2)
    assert bool(hit[0]) and x.shape == (1,)
    h1 = regen_hash(np.ones((3, 3)))
    assert h1 == regen_hash(np.ones((3, 3)))
    bp = bone_penetration(np.array([[0.0, 0.0, -5.0]]),
                          (np.array([-1.0, -1.0, -10.0]), np.array([1.0, 1.0, 0.0])), [])
    assert bp["n_inside"] == 1
