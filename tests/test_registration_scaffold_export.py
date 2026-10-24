"""Registration recovers known geometry; scaffold meshes are valid; unsupported exports are blocked."""
import json

import numpy as np
import pytest
import trimesh

from bilacl import abstention
from bilacl.annotation.schema import Label
from bilacl.estimation.anatomy import estimate_contralateral, reconcile_with_remnants
from bilacl.imaging.coords import apply_affine
from bilacl.imaging.volume import load_nifti
from bilacl.provenance import EvidenceLevel, Provenance, ReviewStatus
from bilacl.registration.bilateral import register_bilateral
from bilacl.scaffold.design import ScaffoldParams, endpoint_matching_hypotheses, generate_candidate
from bilacl.scaffold.mesh import export_candidate, occupancy_to_mesh, self_intersections, validate_candidate

PARAMS = dict(alignment_pattern="parallel", channel_diameter_mm=0.6, min_wall_mm=0.4, grid_res_mm=0.2,
              assumption_sources={"channel_diameter_mm": "test", "min_wall_mm": "test"})


@pytest.fixture(scope="module")
def reg_case(syn_case):
    c = json.loads((syn_case / "case.json").read_text())
    il = load_nifti(syn_case / c["injured"]["labels"])
    cl = load_nifti(syn_case / c["contralateral"]["labels"])
    reg = register_bilateral(il, cl)
    truth = json.loads((syn_case / "truth" / "truth.json").read_text())
    return c, il, cl, reg, truth


def test_per_bone_registration_recovers_truth(reg_case):
    c, il, cl, reg, truth = reg_case
    # true mapping contralateral world -> injured world for the femur: Tinj_fem @ inv(Tcon_fem)
    Tf = np.array(truth["injured"]["template_to_world_femur"]) @ np.linalg.inv(np.array(truth["contralateral"]["template_to_world_femur"]))
    pts = np.array([truth["contralateral"]["landmarks"]["femur_lateral_epicondyle"], truth["contralateral"]["landmarks"]["femur_notch_roof_apex"]])
    err = np.linalg.norm(apply_affine(reg.contra_to_injured_femur, pts) - apply_affine(Tf, pts), axis=1)
    assert err.max() < 2.5  # mm; includes simulated L/R size asymmetry
    assert reg.femur_icp.rms < 1.0 and reg.tibia_icp.rms < 1.0
    assert np.linalg.det(reg.contra_to_injured_femur[:3, :3]) < 0  # includes the mirror


def test_estimate_respects_bone_and_reports(reg_case):
    c, il, cl, reg, _ = reg_case
    a = estimate_contralateral(il, cl, reg)
    assert a.mask.any() and not (a.mask & np.isin(il.data, [Label.FEMUR, Label.TIBIA])).any()
    b = reconcile_with_remnants(a, il, None)
    assert b.evidence_level == EvidenceLevel.ESTIMATED
    assert "remnant_fraction_consistent" in b.metrics


def test_scaffold_mesh_is_valid(reg_case):
    c, il, cl, reg, _ = reg_case
    a = estimate_contralateral(il, cl, reg)
    cand = generate_candidate(a.mask, a.femoral_fp, a.tibial_fp, il.data, il.affine, ScaffoldParams(**PARAMS))
    assert cand.applicable
    mesh = occupancy_to_mesh(cand.occupancy, cand.grid_affine)
    v = validate_candidate(cand, mesh, il.data, il.affine)
    assert v["mesh"]["watertight"] and v["mesh"]["winding_consistent"]
    assert v["mesh"]["self_intersections"]["intersecting_pairs"] == 0
    assert v["mesh"]["connected_components"] == 1
    assert v["bone"]["vertices_inside_bone"] == 0
    assert cand.evidence["internal_channels"] == "hypothesized"


def test_self_intersection_detector_finds_overlap():
    m = trimesh.util.concatenate([trimesh.creation.box((2, 2, 2)), trimesh.creation.box((2, 2, 2), transform=trimesh.transformations.translation_matrix((1, 0.3, 0.2)))])
    assert self_intersections(m)["intersecting_pairs"] > 0
    assert self_intersections(trimesh.creation.icosphere())["intersecting_pairs"] == 0


def test_bridging_not_applicable_without_gap(reg_case):
    c, il, cl, reg, _ = reg_case
    a = estimate_contralateral(il, cl, reg)
    lab = il.data.copy()
    lab[np.isin(lab, [Label.ACL_REMNANT_FEMORAL, Label.ACL_REMNANT_TIBIAL])] = 0  # chronic: no remnants
    cand = generate_candidate(a.mask, a.femoral_fp, a.tibial_fp, lab, il.affine, ScaffoldParams(**PARAMS))
    assert not cand.applicable and any("NOT APPLICABLE" in n for n in cand.notes)


def test_scaffold_params_require_cited_dimensions():
    assert ScaffoldParams(alignment_pattern="parallel").validate()
    assert ScaffoldParams(alignment_pattern="parallel", channel_diameter_mm=0.5, min_wall_mm=0.3).validate()  # no sources
    assert ScaffoldParams(enthesis_transition=True, alignment_pattern="none").validate()  # needs justification


def test_endpoint_matching_blocked_by_default_and_allows_unmatched():
    assert endpoint_matching_hypotheses(np.zeros((2, 3)), np.zeros((2, 3)), 2.0)["status"] == "blocked"
    r = endpoint_matching_hypotheses(np.array([[0, 0, 0], [10, 0, 0]]), np.array([[0, 0, 1]]), 3.0, acquisition_supports_segments=True)
    assert r["hypotheses"][0]["unmatched_femoral"] == [1]


def _abst(ok=True):
    return {"abstain_design_export": not ok, "reasons_design_export": [] if ok else ["x"]}


def test_export_gate_blocks_unverified_physical_metadata_even_for_synthetic():
    g = abstention.export_gate(_abst(), ReviewStatus.APPROVED, True, synthetic=True, inj_physical_verified=False)
    assert not g["allowed"]


def test_export_gate_requires_review_and_no_abstention_for_real_data():
    assert not abstention.export_gate(_abst(), ReviewStatus.UNREVIEWED, True, False, True)["allowed"]
    assert not abstention.export_gate(_abst(False), ReviewStatus.APPROVED, True, False, True)["allowed"]
    assert not abstention.export_gate(_abst(), ReviewStatus.APPROVED, False, False, True)["allowed"]
    assert abstention.export_gate(_abst(), ReviewStatus.APPROVED, True, False, True)["allowed"]


def test_export_function_refuses_blocked_gate(tmp_path, reg_case):
    c, il, cl, reg, _ = reg_case
    a = estimate_contralateral(il, cl, reg)
    cand = generate_candidate(a.mask, a.femoral_fp, a.tibial_fp, il.data, il.affine, ScaffoldParams(**PARAMS))
    mesh = occupancy_to_mesh(cand.occupancy, cand.grid_affine)
    with pytest.raises(PermissionError):
        export_candidate(cand, mesh, {}, tmp_path, source={}, gate={"allowed": False, "reasons": ["test"]},
                         provenance=Provenance(EvidenceLevel.HYPOTHESIZED, "test"))
    assert not list(tmp_path.glob("*.stl"))


def test_abstention_on_coarse_and_unverified_inputs(reg_case):
    from bilacl.imaging.volume import volume_from_array

    c, il, cl, reg, _ = reg_case
    coarse = volume_from_array(np.zeros((4, 4, 4)), spacing=(0.5, 0.5, 4.0))
    r = abstention.evaluate(c, coarse, coarse, None, None, None, None, None, synthetic=False)
    names = {x["name"]: x["status"] for x in r["checks"]}
    assert names["injured_physical_metadata_verified"] == "fail"
    assert names["injured_resolution_supports_design"] == "fail"
    assert names["contralateral_expert_confirmed_normal"] == "unknown"
    assert r["abstain_design_export"]


def test_twisted_channels_keep_min_wall(reg_case):
    c, il, cl, reg, _ = reg_case
    a = estimate_contralateral(il, cl, reg)
    p = ScaffoldParams(**{**PARAMS, "alignment_pattern": "twisted", "twist_deg": 60.0})
    cand = generate_candidate(a.mask, a.femoral_fp, a.tibial_fp, il.data, il.affine, p)
    v = validate_candidate(cand, occupancy_to_mesh(cand.occupancy, cand.grid_affine), il.data, il.affine)
    w = v["channels"]["min_wall_between_channels_mm"]
    assert w is None or w >= p.min_wall_mm - p.grid_res_mm
