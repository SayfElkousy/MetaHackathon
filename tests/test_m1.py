import json
import math
import numpy as np
_trapz = getattr(np, "trapezoid", None) or np.trapz
from acl_engine.schema import validate_request
from acl_engine.paths import (build_phase_field, strand_angles,
                              tilt_deg_from_vel, arc_length, find_crossings,
                              slope_factor, rate_for_theta)
from acl_engine.frames import compute_rmf
from acl_engine.sweep import sweep_strand, signed_volume, segments_around


def load_fixture(path):
    with open(path) as f:
        req = json.load(f)
    norm, _ = validate_request(req)
    return norm


def test_rate_and_pitch():
    a, b = 3.55, 2.54
    S = slope_factor(a, b)
    assert abs(S - 3.98792) < 1e-4
    r35 = rate_for_theta(35.0, a, b, "crossing_tilt")
    r20 = rate_for_theta(20.0, a, b, "crossing_tilt")
    assert abs(r35 - 0.175582) < 1e-5
    assert abs(r20 - 0.091268) < 1e-5
    assert abs(math.pi / (3 * r35) - 5.9641) < 1e-3
    assert abs(math.pi / (3 * r20) - 11.4738) < 1e-3


def test_starter_phase_events_tilts_arcs():
    norm = load_fixture("fixtures/starter_request.json")
    pf, rates, S, p_events = build_phase_field(norm)
    assert abs(pf.phi_L() - 9.76134) < 1e-3
    assert abs(pf.phi_L() / (2 * math.pi) - 1.5536) < 1e-3
    equiv = pf.phi_L() / (math.pi / 3.0)
    assert abs(equiv - 9.3214) < 1e-3
    ev = find_crossings(pf, 3.55, 2.54, 0.0, False, 70.0)
    assert len(ev) == 9
    ref_y = [2.982, 8.946, 14.910, 21.682, 33.156, 44.630]
    for got, want in zip([e["y_mm"] for e in ev[:6]], ref_y):
        assert abs(got - want) < 0.01
    # tilt at crossings per zone
    for e in ev:
        y = e["y_mm"]
        want = 35.0 if (y < 20.0 or y >= 50.0) else 20.0
        if min(abs(y-20.0),abs(y-50.0)) < 2.0: continue
        # events near transitions excluded by construction of refs
        assert abs(e["tilt_i_deg"] - want) < 0.02
        assert abs(e["tilt_j_deg"] - want) < 0.02
    arcs = [arc_length(0.0, 70.0, pf, 3.55, 2.54, 0.0, False, i)
            for i in range(3)]
    for got, want in zip(arcs, [82.959, 82.074, 82.401]):
        assert abs(got - want) < 0.02
    # bounds from centerlines + diameter
    ys = np.linspace(0, 70, 7001)
    phi = pf.phi_array(ys)
    rate = pf.rate_array(ys)
    r2 = np.zeros_like(ys)
    xs, zs = [], []
    for i in range(3):
        pos, _v, _a = strand_angles(ys, phi, rate, r2, 3.55, 2.54, 0.0, False, i)
        xs.append(pos[:, 0])
        zs.append(pos[:, 2])
    assert abs((max(x.max() for x in xs) - min(x.min() for x in xs) + 2.85) - 9.950) < 0.005
    assert abs((max(z.max() for z in zs) - min(z.min() for z in zs) + 2.85) - 7.930) < 0.005
    # legacy cross-check
    l20 = math.degrees(math.atan(rate_for_theta(20.0, 3.55, 2.54, "legacy_lateral_pitch_rule") * S))
    l35 = math.degrees(math.atan(rate_for_theta(35.0, 3.55, 2.54, "legacy_lateral_pitch_rule") * S))
    assert abs(l20 - 22.24) < 0.02
    assert abs(l35 - 38.19) < 0.02


def test_cylinder_volume():
    r, L = 1.0, 10.0
    nsec = segments_around(r, 0.03)
    y = np.linspace(0, L, 60)
    c = np.stack([np.zeros_like(y), y, np.zeros_like(y)], axis=1)
    t = np.tile([0.0, 1.0, 0.0], (len(y), 1))
    N1, N2, _T = compute_rmf(c, t)
    V, F = sweep_strand(c, N1, N2, r, nsec)
    assert abs(signed_volume(V, F) - 10 * math.pi) / (10 * math.pi) < 0.005


def test_helix():
    R, w, L = 2.0, 0.3, 10.0
    y = np.linspace(0, L, 2001)
    c = np.stack([R * np.cos(w * y), y, R * np.sin(w * y)], axis=1)
    vel = np.stack([-R * w * np.sin(w * y), np.ones_like(y), R * w * np.cos(w * y)], axis=1)
    acc = np.stack([-R * w * w * np.cos(w * y), np.zeros_like(y), -R * w * w * np.sin(w * y)], axis=1)
    sp = np.linalg.norm(vel, axis=1)
    assert abs(_trapz(sp, y) - L * math.sqrt(1 + R**2 * w**2)) < 1e-6
    tilt = math.degrees(math.atan(abs(R * w)))
    got = math.degrees(np.arccos(1 / math.sqrt(1 + R**2 * w**2)))
    assert abs(got - tilt) < 1e-12
    k = R * w * w / (1 + R**2 * w**2)
    cr = np.linalg.norm(np.cross(vel, acc), axis=1) / np.linalg.norm(vel, axis=1)**3
    assert abs(cr.mean() - k) < 1e-6


def test_s_inflection_rmf():
    y = np.linspace(0, 10, 400)
    x = 2.0 * np.sin(y * 2 * math.pi / 10.0)
    c = np.stack([x, y, np.zeros_like(y)], axis=1)
    g = np.gradient(c, y, axis=0, edge_order=2)
    N1, N2, T = compute_rmf(c, g)
    assert np.abs(np.sum(N1 * N2, axis=1)).max() < 1e-9
    # continuity: frame should not jump
    assert np.linalg.norm(np.diff(N1, axis=0), axis=1).max() < 0.2
