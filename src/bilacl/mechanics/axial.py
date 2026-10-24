"""Simplified mechanical exploration: 1D axial bar-in-series model.

THIS IS NOT A FINITE-ELEMENT MODEL and produces no stress fields.

Model
  The scaffold is a bar along its axis with cross-sectional solid area A(s)
  measured from the candidate geometry (envelope area minus channels/pockets).
  Material law: linear elastic, small strain, homogeneous solid modulus E_s
  (USER INPUT with citation; no default). For channels aligned with the axis,
  the axial effective area is the solid area (exact under 1D assumptions;
  transverse effects ignored).
  Fixation: optional series springs k_fix at each end (USER INPUT).
  Loading: prescribed axial force F or prescribed elongation dL.
Outputs: axial stiffness, nominal stress per station (F/A_solid), strain.
Sensitivity: one-at-a-time and grid sweep over uncertain inputs.

Separate concerns (kept separate in reports): geometric fit, predicted mechanical
behaviour under this model, biological healing, and clinical reinjury. This module
addresses only the second, under the stated simplifications.
"""
from __future__ import annotations

import dataclasses
import itertools

import numpy as np

from ..imaging.coords import apply_affine


@dataclasses.dataclass
class AxialModelInputs:
    E_solid_MPa: float | None  # required
    E_source: str  # citation or explicit assumption
    k_fixation_N_per_mm: float | None = None  # per end; None = rigid fixation (assumption)
    fixation_source: str = "assumption: rigid fixation (upper bound on stiffness)"
    force_N: float | None = None
    elongation_mm: float | None = None

    def validate(self):
        e = []
        if self.E_solid_MPa is None:
            e.append("E_solid_MPa is required (no default material property is assumed)")
        if not self.E_source.strip():
            e.append("E_source must cite a measurement/publication or state the assumption")
        if (self.force_N is None) == (self.elongation_mm is None):
            e.append("specify exactly one of force_N or elongation_mm")
        return e


def solid_area_profile(occ: np.ndarray, grid_affine: np.ndarray, fp_f, fp_t, n: int = 40) -> dict:
    """Solid cross-sectional area per station perpendicular to the femoral->tibial axis (mm^2)."""
    res = float(np.linalg.norm(grid_affine[:3, 0]))
    P = apply_affine(grid_affine, np.argwhere(occ).astype(float))
    ax = np.asarray(fp_t, float) - np.asarray(fp_f, float)
    L = np.linalg.norm(ax)
    u = ax / L
    s = (P - fp_f) @ u
    edges = np.linspace(s.min(), s.max(), n + 1)
    cnt, _ = np.histogram(s, edges)
    ds = edges[1] - edges[0]
    area = cnt * res ** 3 / ds
    return {"s_mm": (edges[:-1] + edges[1:]) / 2, "ds_mm": ds, "area_mm2": area, "length_mm": float(s.max() - s.min())}


def axial_response(profile: dict, inp: AxialModelInputs) -> dict:
    errs = inp.validate()
    if errs:
        raise ValueError("; ".join(errs))
    A = np.maximum(profile["area_mm2"], 1e-9)
    ds = profile["ds_mm"]
    compliance_bar = float(np.sum(ds / (inp.E_solid_MPa * A)))  # mm/N (MPa = N/mm^2)
    c_fix = 0.0 if inp.k_fixation_N_per_mm is None else 2.0 / inp.k_fixation_N_per_mm
    k = 1.0 / (compliance_bar + c_fix)
    if inp.force_N is not None:
        F = inp.force_N
    else:
        F = k * inp.elongation_mm
    stress = F / A
    strain = stress / inp.E_solid_MPa
    return {
        "model": "1D linear-elastic bar in series (not FE)",
        "axial_stiffness_N_per_mm": k, "bar_only_stiffness_N_per_mm": 1.0 / compliance_bar,
        "force_N": F, "elongation_mm": F / k,
        "min_solid_area_mm2": float(A.min()), "max_nominal_stress_MPa": float(stress.max()), "max_strain": float(strain.max()),
        "small_strain_assumption_violated": bool(strain.max() > 0.05),
        "inputs": dataclasses.asdict(inp),
        "assumptions": [
            "linear elastic, homogeneous, isotropic solid; no viscoelasticity, creep, fatigue or degradation",
            "uniaxial load along the femoral-tibial axis; bending, torsion and contact ignored",
            "channels parallel to axis (axial area = solid area); twisted channels make this approximate",
            "geometry from voxelised candidate at grid resolution",
            "remnant tissue, healing tissue and biological load sharing are not modelled",
        ],
    }


def sensitivity(profile: dict, base: AxialModelInputs, ranges: dict) -> dict:
    """ranges: {"E_solid_MPa": [..], "area_scale": [..], "k_fixation_N_per_mm": [..]} -> grid of responses."""
    keys = list(ranges)
    rows = []
    for combo in itertools.product(*[ranges[k] for k in keys]):
        d = dict(zip(keys, combo))
        prof = dict(profile)
        if "area_scale" in d:
            prof["area_mm2"] = profile["area_mm2"] * d["area_scale"]
        inp = dataclasses.replace(base, **{k: v for k, v in d.items() if k in AxialModelInputs.__dataclass_fields__})
        r = axial_response(prof, inp)
        rows.append({**d, "stiffness_N_per_mm": r["axial_stiffness_N_per_mm"], "max_stress_MPa": r["max_nominal_stress_MPa"], "max_strain": r["max_strain"]})
    st = [r["stiffness_N_per_mm"] for r in rows]
    return {"grid": rows, "stiffness_range_N_per_mm": [float(min(st)), float(max(st))], "varied": keys}
