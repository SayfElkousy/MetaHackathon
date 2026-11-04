"""N1 tests, written here (Muse's test_n1.py was never shared).
Checks the new functions against the v2/v3 reference values."""
import json
import math
import numpy as np
from acl_engine.schema import validate_request
from acl_engine.paths import (PhaseField, build_rates, find_crossings, arc_length_interval,
                              mean_lambda_interval, lambda_for_rate)
from acl_engine.fold import slope_factor_q
from acl_engine.axis import eval_axis
from acl_engine.frames import compute_rmf
from acl_engine.measure import curvature_max_straight, min_surface_gap
from acl_engine.paths import straight_pos_vel_acc


def _setup(path):
    norm, _ = validate_request(json.load(open(path)))
    d = norm["_derived"]
    a, b, q = d["swing_mm"], d["fold_depth_mm"], d["fold_sharpness"]
    rates = build_rates(norm, a, b, q, d["angle_mode"])
    pf = PhaseField([z["length_mm"] for z in norm["zones"]], rates, norm["transitions"]["length_mm"])
    return norm, a, b, q, pf


def test_T1_starter_regression():
    norm, a, b, q, pf = _setup("fixtures/starter_request.json")
    assert abs(slope_factor_q(a, b, q) - 3.98792) < 1e-4
    assert abs(pf.phi_L() - 9.76134) < 1e-3
    ev = find_crossings(pf, a, b, q, 0.0, False, 70.0)
    assert len(ev) == 9
    for got, want in zip([e["y_mm"] for e in ev[:6]], [2.982, 8.946, 14.910, 21.682, 33.156, 44.630]):
        assert abs(got - want) < 0.01
    arcs = [arc_length_interval(0, 70, pf, a, b, q, 0.0, False, i) for i in range(3)]
    for got, want in zip(arcs, [82.959, 82.074, 82.401]):
        assert abs(got - want) < 0.02
    ss = np.linspace(0, 70, 2001)
    assert abs(1.0 / curvature_max_straight(ss, pf, a, b, q, 0.0, False) - 3.673) < 0.01
    s = np.linspace(0, 70, 3501)
    C, N1, N2, _ = eval_axis(s, {"type": "straight"})
    cen = []
    for i in range(3):
        p, _, _ = straight_pos_vel_acc(s, pf.phi_array(s), pf.rate_array(s), pf.rate2_array(s), a, b, q, 0.0, False, i)
        cen.append(C + p[:, 0][:, None] * N1 - p[:, 2][:, None] * N2)
    gap = min_surface_gap(cen, s, 2.85 / 2, 2.85)
    assert abs(gap["min_surface_mm"] - 0.930) < 0.01


def test_T2_helix_lambda():
    for th, want in [(12, 1.0223), (20, 1.0642), (29, 1.1434), (35, 1.2208)]:
        assert abs(1.0 / math.cos(math.radians(th)) - want) < 1e-4


def test_T3_fiber_length_control():
    norm, a, b, q, pf = _setup("fixtures/acl_context_request.json")
    lo, hi = pf.bounds[1], pf.bounds[2]
    lam = np.mean([mean_lambda_interval(lo, hi, pf, a, b, q, 0.0, False, i) for i in range(3)])
    assert abs(lam - 1.06) < 0.002


def test_T5_arc_axis_frames():
    s = np.linspace(0, 70, 700)
    C, N1, N2, T = eval_axis(s, {"type": "arc", "radius_mm": 60.0})
    assert np.abs(np.sum(N1 * N2, axis=1)).max() < 1e-9
    assert np.linalg.norm(np.cross(N1, N2) - T, axis=1).max() < 1e-9
    g = np.gradient(C, s, axis=0, edge_order=2)
    a1, a2, at = compute_rmf(C, g)
    assert np.linalg.norm(np.diff(a1, axis=0), axis=1).max() < 0.2


def test_fold_sharpness_keeps_crossing_angle():
    """Sharper folds change the slope factor; the crossing tilt must still hit the target."""
    norm, a, b, _, _ = _setup("fixtures/starter_request.json")
    q = 0.7
    rates = build_rates(norm, a, b, q, "crossing_tilt")
    pf = PhaseField([20, 30, 20], rates, 4.0)
    ev = find_crossings(pf, a, b, q, 0.0, False, 70.0)
    for e in ev:
        if min(abs(e["y_mm"] - 20), abs(e["y_mm"] - 50)) < 2.0:
            continue
        want = 35.0 if (e["y_mm"] < 20 or e["y_mm"] > 50) else 20.0
        assert abs(e["tilt_i_deg"] - want) < 0.05
