import numpy as np
import pytest

from bilacl.evaluation.metrics import bootstrap_ci, classification_report, dice, surface_metrics
from bilacl.mechanics.axial import AxialModelInputs, axial_response


def test_axial_model_matches_closed_form():
    prof = {"area_mm2": np.full(10, 20.0), "ds_mm": 3.0, "s_mm": np.arange(10) * 3.0}
    r = axial_response(prof, AxialModelInputs(E_solid_MPa=100.0, E_source="test", force_N=60.0))
    assert np.isclose(r["axial_stiffness_N_per_mm"], 100 * 20 / 30)  # EA/L
    assert np.isclose(r["max_nominal_stress_MPa"], 3.0)


def test_axial_model_requires_material_input():
    prof = {"area_mm2": np.ones(3), "ds_mm": 1.0}
    with pytest.raises(ValueError):
        axial_response(prof, AxialModelInputs(E_solid_MPa=None, E_source="", force_N=1.0))


def test_dice_and_surface_ignore_region():
    a = np.zeros((20, 20, 20), bool)
    a[5:10, 5:10, 5:10] = True
    assert dice(a, a) == 1.0
    ig = np.zeros_like(a)
    ig[5:10, 5:10, 5:7] = True
    b = a.copy()
    b[5:10, 5:10, 5:7] = False  # disagreement only inside ignored region
    assert dice(b, a, ig) == 1.0
    assert surface_metrics(a, a, (1, 1, 1))["hd95_mm"] == 0


def test_bootstrap_is_subject_level():
    r = bootstrap_ci({"s1": [1.0, 1.0, 1.0, 1.0], "s2": [3.0]})
    assert r["n_subjects"] == 2 and np.isclose(r["estimate"], 2.0)


def test_classification_report():
    r = classification_report([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert 0 <= r["auroc"] <= 1 and "ece" in r["calibration"]
