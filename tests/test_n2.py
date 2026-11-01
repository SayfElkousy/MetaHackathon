"""Muse N2 tests (T6 inclination, T7 flare, schema backward compatibility)."""
import json
import numpy as np
from acl_engine.schema import validate_request
from acl_engine.context import build_native_axis, measured_inclinations, build_envelope_fields


def test_T6_inclination():
    T, F, u = build_native_axis(32.0, 47.4, 75.3, "right")
    sag, cor = measured_inclinations(u)
    assert abs(sag - 47.4) < 0.1
    assert abs(cor - 75.3) < 0.1
    assert abs(float(np.linalg.norm(F - T)) - 32.0) < 1e-9


def test_T7_flare():
    prof = [34.0, 33.0, 35.0, 38.0, 42.0]
    ss, A, Ws, Ts, Amid = build_envelope_fields(32.0, 10.0, 5.0, prof, 3.5)
    assert abs(A[0] / Amid - 3.5) / 3.5 < 0.02
    assert abs(A[-1] / Amid - 3.5) / 3.5 < 0.02


def test_T1_still_passes_schema():
    with open("fixtures/starter_request.json") as f:
        req = json.load(f)
    norm, _ = validate_request(req)
    assert abs(norm["_derived"]["acl_length_mm"] - 30.0) < 1e-9
