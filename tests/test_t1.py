"""T1 tests (Muse). Run from the project root (pytest.ini sets this up)."""
import json
from acl_engine.schema import validate_request
from acl_engine.paths import PhaseField, find_crossings, arc_length_interval
from acl_engine.controls import resolve_rates_closed_loop

FIXTURES = ["fixtures/starter_request.json",
            "fixtures/pass_request.json",
            "fixtures/starter_fiberlength.json",
            "fixtures/acl_context_request.json"]


def test_A11_all_fixtures_load_and_migrate():
    for fx in FIXTURES:
        with open(fx) as f:
            req = json.load(f)
        norm, mig = validate_request(req)
        assert norm["schema_version"] == "4.0.0"
        assert mig["to"] == "4.0.0"


def test_A11_starter_regression():
    with open("fixtures/starter_request.json") as f:
        req = json.load(f)
    norm, _ = validate_request(req)
    a = norm["_derived"]["swing_mm"]
    b = norm["_derived"]["fold_depth_mm"]
    q = norm["_derived"]["fold_sharpness"]
    rates, _ = resolve_rates_closed_loop(norm, a, b, q, "crossing_tilt")
    pf = PhaseField([20.0, 30.0, 20.0], rates, 4.0)
    assert abs(pf.phi_L() - 9.76134) < 1e-3
    ev = find_crossings(pf, a, b, q, 0.0, False, 70.0)
    assert len(ev) == 9
    arcs = [arc_length_interval(0.0, 70.0, pf, a, b, q, 0.0, False, i) for i in range(3)]
    for got, want in zip(arcs, [82.959, 82.074, 82.401]):
        assert abs(got - want) < 0.02


def test_fiber_length_closed_loop_hits_target():
    with open("fixtures/starter_fiberlength.json") as f:
        req = json.load(f)
    norm, _ = validate_request(req)
    d = norm["_derived"]
    _rates, achieved = resolve_rates_closed_loop(norm, d["swing_mm"], d["fold_depth_mm"],
                                                 d["fold_sharpness"], d["angle_mode"])
    assert achieved
    for rec in achieved.values():
        assert abs(rec["achieved"] - rec["target"]) <= 0.002


def test_records_unknown_uses_preset():
    with open("fixtures/starter_request.json") as f:
        req = json.load(f)
    req["anatomy"] = {"acl_length_mm": None, "acl_width_mm": 10.0, "acl_thickness_mm": 5.0}
    norm, _ = validate_request(req)
    assert any(a["field"] == "acl_length_mm" for a in norm["_assumptions"])
    assert abs(norm["_derived"]["acl_length_mm"] - 32.0) < 1e-9


def test_measurement_conditions_derived():
    with open("fixtures/starter_request.json") as f:
        req = json.load(f)
    req["measurement_conditions"] = {
        "side": "right",
        "knee_position": {"flexion_deg": 0},
        "imaging": {"injury_date": "2026-01-01", "scan_date": "2026-01-11"}}
    norm, _ = validate_request(req)
    assert norm["measurement_conditions"]["derived"]["days_injury_to_imaging"] == 10


def test_stl_has_three_corners(tmp_path):
    # Guards against the STL bug that came back in M1, N2 and T1.
    import numpy as np
    from acl_engine.export import write_stl, read_stl_vertices
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    F = np.array([[0, 1, 2]])
    p = tmp_path / "t.stl"
    write_stl(str(p), [(V, F)])
    assert p.stat().st_size == 84 + 50
    assert np.allclose(read_stl_vertices(str(p)), V)
