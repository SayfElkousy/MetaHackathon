"""Tests for the v2 anatomical engine, patient-input adapter, exports and server API."""
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
from acl_engine.v2.bones import ProceduralBone, load_mesh_file  # noqa: E402
from acl_engine.v2.scene import build_scene  # noqa: E402
from acl_engine.v2.patient import (import_file, import_document, merge_import, apply_edit, reset_param,  # noqa: E402
                                   diff_vs_imported, AdapterError)
from acl_engine.v2.export import export_zip  # noqa: E402

PATIENTS = os.path.join(ROOT, "fixtures", "patients")


def chk(scene, cid):
    return next(c for c in scene["checks"] if c["id"] == cid)


def obj(scene, oid):
    return next(o for o in scene["objects"] if o["id"] == oid)


def braided(st):
    """These tests exercise the gap-filling braid; the default construct is now the six-level
    layered construct (tests/test_layered.py), so select the braid explicitly."""
    if st["params"]["scaffold.arch"] == "layered":
        st["params"]["scaffold.arch"] = "braided"
    return st


@pytest.fixture(scope="module")
def demo():
    st = braided(D.new_state())
    return st, build_scene(st, "full")


# ---------------------------------------------------------------- anatomy
def test_bones_are_closed_and_recognizable():
    for which in ("femur", "tibia"):
        V, F = ProceduralBone(which, {}).mesh(1.0)
        a = mesher.audit(V, F)
        assert a["closed"] and a["nonmanifold_edges"] == 0 and a["volume_mm3"] > 50000
    fem = ProceduralBone("femur", {})
    # intercondylar notch: open between the condyles, bone on both sides
    assert fem.sdf(np.array([[0.0, -15.0, 15.0]]))[0] > 0
    assert fem.sdf(np.array([[23.0, -6.0, 15.0]]))[0] < 0 and fem.sdf(np.array([[-24.5, -6.0, 15.0]]))[0] < 0


def test_attachments_on_bone_and_no_midbody_penetration(demo):
    st, sc = demo
    assert chk(sc, "footprint_on_bone_tib")["status"] == "pass"
    assert chk(sc, "footprint_on_bone_fem")["status"] == "pass"
    assert chk(sc, "ligament_bone_penetration")["status"] == "pass"
    inc = sc["metrics"]["inclinations"]
    assert 40 < inc["sagittal_deg"] < 55 and 65 < inc["coronal_deg"] < 85     # L03: 47.4 / 75.3
    assert 25 < sc["metrics"]["ligament_length_mm"] < 41                      # L01: 22-41


def test_solid_meshes_closed(demo):
    st, sc = demo
    assert chk(sc, "solid_meshes")["status"] == "pass", chk(sc, "solid_meshes")["message"]


def test_deterministic_rebuild(demo):
    st, sc = demo
    sc2 = build_scene(copy.deepcopy(st), "full")
    assert sc2["geometry_hash"] == sc["geometry_hash"] and sc2["design_hash"] == sc["design_hash"]


def test_display_density_does_not_change_design(demo):
    st, sc = demo
    s2 = copy.deepcopy(st)
    s2["display"]["native_fibers_per_fascicle"] = 3
    s2["display"]["crimp_exaggeration"] = 5
    sc2 = build_scene(s2, "preview")
    assert sc2["design_hash"] == sc["design_hash"]


# ---------------------------------------------------------------- injury & scaffold
def test_moving_injury_moves_scaffold():
    a = braided(D.new_state())
    b = braided(D.new_state())
    b["params"]["injury.center_s"] = 0.42
    sa, sb = build_scene(a, "preview"), build_scene(b, "preview")
    assert sb["metrics"]["scaffold"]["start_mm"] < sa["metrics"]["scaffold"]["start_mm"] - 1.0
    assert sb["metrics"]["injury"]["tibial_stump_min_mm"] < sa["metrics"]["injury"]["tibial_stump_min_mm"] - 2.0


def test_gap_and_stumps_for_complete_tear(demo):
    st, sc = demo
    ids = [o["id"] for o in sc["objects"]]
    assert "retained_tibial_stump" in ids and "retained_femoral_stump" in ids and "injury_region" in ids
    inj = sc["metrics"]["injury"]
    assert inj["measured_gap_mm"] > 1.0
    assert chk(sc, "scaffold_spans_gap")["status"] == "pass"


def test_partial_tear_preserves_unaffected_tissue():
    st = D.new_state()
    st["params"].update({"injury.type": "partial", "injury.involvement": "AM", "scaffold.arch": "aligned",
                         "scaffold.engagement": "abut", "scaffold.yarn_diameter_mm": 0.3})
    sc = build_scene(st, "full")
    assert chk(sc, "partial_preserves")["status"] == "pass"
    ids = [o["id"] for o in sc["objects"]]
    assert "retained_tissue" in ids and "retained_tibial_stump" not in ids      # one continuous remnant
    assert sc["metrics"]["injury"]["affected_area_fraction"] < 0.75
    assert chk(sc, "yarn_clearance")["status"] == "pass"


def test_bone_interface_only_when_reaching_bone(demo):
    st, sc = demo
    kinds = [r["kind"] for r in sc["metrics"]["scaffold"]["regions"]]
    assert "bone_interface" not in kinds                                       # midsubstance repair
    s2 = braided(D.new_state())
    s2["params"]["injury.center_s"] = 0.15
    sc2 = build_scene(s2, "preview")
    assert sc2["metrics"]["scaffold"]["regions"][0]["kind"] == "bone_interface"
    assert sc2["metrics"]["scaffold"]["regions"][2]["kind"] == "tissue_interface"


def test_braid_clearance_by_construction(demo):
    st, sc = demo
    assert chk(sc, "yarn_clearance")["status"] == "pass"
    b = sc["metrics"]["scaffold"]["braid"][0]["layers"][0]
    # dependent pair: tan(angle) = 2 pi R / pitch
    assert abs(np.tan(np.radians(b["angle_deg"])) - 2 * np.pi * b["radius_mm"] / b["pitch_mm"]) < 1e-6
    # measured, not assumed: the braid's strands keep their clearance
    assert b["measured_min_clearance_mm"] >= -0.005


def test_incompatible_settings_are_detected():
    st = braided(D.new_state())
    st["params"]["scaffold.yarn_count"] = 40
    st["params"]["scaffold.yarn_diameter_mm"] = 1.2
    sc = build_scene(st, "preview")
    assert any(c["status"] == "fail" and c["id"] in ("yarn_clearance", "scaffold_layout", "scaffold_note_0", "scaffold_note_1") for c in sc["checks"])
    assert not sc["validity"]["valid"]


def test_invalid_values_are_rejected_not_clamped():
    st = D.new_state()
    st["params"]["injury.gap_mm"] = 99.0
    sc = build_scene(st, "preview")
    assert not sc["ok"] and sc["errors"][0]["param"] == "injury.gap_mm"


def test_composition_normalized_in_export(demo):
    st, sc = demo
    for r in sc["metrics"]["scaffold"]["regions"]:
        assert abs(sum(r["composition"].values()) - 1.0) < 1e-9


def test_sampled_profile_keeps_spatial_detail():
    st = D.new_state()
    st["params"]["lig.width_profile"] = {"s": [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1], "v": [12, 11, 10, 9, 8, 12, 8, 9, 10, 11, 12]}
    sc = build_scene(st, "preview")
    assert sc["ok"]
    from acl_engine.v2.ligament import profile_eval
    assert abs(profile_eval(st["params"]["lig.width_profile"], np.array([0.5]))[0] - 12) < 1e-9   # a local bump survives


# ---------------------------------------------------------------- patient input
def test_patient_A_and_B_import():
    a, ra = import_file(os.path.join(PATIENTS, "patient_A_synthetic", "input.json"))
    b, rb = import_file(os.path.join(PATIENTS, "patient_B_synthetic", "input.json"))
    assert not ra["errors"] and not rb["errors"]
    assert "attachments.femoral.shape.length_mm" in rb["missing"]
    assert "unknown_block_from_newer_model" in rb["unsupported"]
    assert b["provenance"]["fp.fem.length_mm"]["status"] == "missing"
    assert a["provenance"]["bone.femur_bicondylar_width_mm"]["status"] == "measured"
    assert a["provenance"]["bone.femur_bicondylar_width_mm"]["uncertainty"] == {"sd": 1.0}
    assert a["provenance"]["scaffold.arch"]["status"] == "proposed"
    assert isinstance(a["params"]["lig.width_profile"], dict) and len(a["params"]["lig.width_profile"]["s"]) == 9
    sa, sb = build_scene(a, "full"), build_scene(b, "full")
    assert sa["side"] == "right" and sb["side"] == "left"
    assert sa["metrics"]["injury"]["type"] == "complete" and sb["metrics"]["injury"]["type"] == "partial"
    assert sb["metrics"]["injury"]["boundary_source"] == "observed mask"
    assert sa["metrics"]["scaffold"]["architecture"] != sb["metrics"]["scaffold"]["architecture"]
    assert abs(sa["metrics"]["ligament_length_mm"] - sb["metrics"]["ligament_length_mm"]) > 0.5
    assert obj(sb, "bone_tibia")["status"] == "observed" and obj(sa, "bone_tibia")["status"] == "assumed"
    # left knee is mirrored for display: lateral is -X
    assert obj(sb, "footprint_femoral")["V"][:, 0].mean() < 0


def test_coordinate_alignment_of_imported_mesh():
    """Patient B's tibia STL is stored in a rotated + translated image frame; after the
    adapter's image_to_model transform it must coincide with the procedural tibia."""
    b, _ = import_file(os.path.join(PATIENTS, "patient_B_synthetic", "input.json"))
    a = b["assets"]["tibia_mesh"]
    V, F = load_mesh_file(a["path"])
    T = np.asarray(a["transform"])
    Vm = V @ T[:3, :3].T + T[:3, 3]
    ref = ProceduralBone("tibia", {"bone.tibia_plateau_width_mm": 71.0, "bone.tibial_slope_deg": 9.5})
    d = np.abs(ref.sdf(Vm[::50]))
    assert np.median(d) < 0.25 and d.max() < 1.5


def test_adapter_rejects_bad_input():
    with pytest.raises(AdapterError):
        import_document({"schema": "acl-mri-input", "schema_version": "2.0.0", "imaging": {"laterality": "right"}})
    st, rep = import_document({"schema": "acl-mri-input", "schema_version": "1.0.0",
                               "imaging": {"laterality": "right"},
                               "injury": {"gap_length_mm": {"value": 300, "units": "mm"}}})
    assert st is None and any("injury.gap_mm" in e for e in rep["errors"])
    st, rep = import_document({"schema": "acl-mri-input", "schema_version": "1.0.0", "imaging": {"laterality": "right"},
                               "injury": {"gap_length_mm": {"value": 0.6, "units": "cm"}}})
    assert st["params"]["injury.gap_mm"] == pytest.approx(6.0)


def test_override_reset_and_merge():
    st, _ = import_file(os.path.join(PATIENTS, "patient_A_synthetic", "input.json"))
    apply_edit(st, "scaffold.yarn_count", 20)
    assert st["provenance"]["scaffold.yarn_count"]["status"] == "override"
    assert diff_vs_imported(st) == [{"param": "scaffold.yarn_count", "imported": 24, "current": 20}]
    new, _ = import_file(os.path.join(PATIENTS, "patient_A_synthetic", "input.json"))
    new["params"]["scaffold.yarn_count"] = 28                  # upstream changed its proposal
    new["imported"]["params"]["scaffold.yarn_count"] = 28
    merged, conflicts = merge_import(st, new, "merge")
    assert merged["params"]["scaffold.yarn_count"] == 20 and conflicts and conflicts[0]["new_upstream"] == 28
    replaced, _ = merge_import(st, new, "replace")
    assert replaced["params"]["scaffold.yarn_count"] == 28
    reset_param(st, "scaffold.yarn_count")
    assert st["params"]["scaffold.yarn_count"] == 24 and st["provenance"]["scaffold.yarn_count"]["status"] == "proposed"
    assert [h["action"] for h in st["history"]] == ["edit", "reset"]


# ---------------------------------------------------------------- exports
def test_export_package_reproduces_design(demo):
    st, sc = demo
    data, man = export_zip(st, sc)
    z = zipfile.ZipFile(io.BytesIO(data))
    names = set(z.namelist())
    for n in ("scene.glb", "design.json", "regions.json", "validation.json", "manifest.json", "history.json",
              "stl/scaffold_yarns_bridge.stl", "stl/scaffold_yarns_all.stl", "stl/retained_tibial_stump.stl"):
        assert n in names, n
    regs = json.loads(z.read("regions.json"))["regions"]
    assert [r["id"] for r in regs] == D.REGION_IDS and all("material" in r for r in regs)
    glb = z.read("scene.glb")
    js = json.loads(glb[20:20 + int.from_bytes(glb[12:16], "little")])
    node = next(n for n in js["nodes"] if n["name"] == "scaffold_yarns_bridge")
    assert node["extras"]["region"] == "bridge" and node["extras"]["material"] == st["params"]["regions.bridge.material"]
    prim = js["meshes"][node["mesh"]]["primitives"][0]
    assert "_REGION" in prim["attributes"]
    # STL in mm: the bridge yarns lie within the ligament's extent
    tri = np.frombuffer(z.read("stl/scaffold_yarns_bridge.stl")[84:], dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])["v"]
    assert 0 < np.ptp(tri.reshape(-1, 3), axis=0).max() < 60
    st2 = json.loads(z.read("design.json"))
    sc2 = build_scene(st2, "full")
    assert sc2["geometry_hash"] == man["geometry_hash"]


# ---------------------------------------------------------------- server
def test_server_state_roundtrip(tmp_path, monkeypatch):
    import server_v2 as V
    monkeypatch.setattr(V, "STORE", str(tmp_path))
    st = V.new_case("demo_partial_am")
    sid = V.save_state(st)
    assert V.load_state(sid)["params"] == st["params"]
    assert V.load_state("0" * 20) is None and V.load_state("../etc") is None
    code, out = V.import_request({"case": "patient_B_synthetic"})
    assert code == 200 and out["state"]["case"]["synthetic"]
    code, out = V.build_request({"state": out["state"], "quality": "preview", "seq": 7})
    assert code == 200 and out["seq"] == 7 and any(o["id"] == "scaffold_yarns" for o in out["objects"])


def _all_geometry(sc):
    import hashlib
    h = hashlib.sha256()
    for o in sc["objects"]:
        if o["kind"] == "mesh":
            h.update(np.round(o["V"], 3).astype(np.float32).tobytes())
        else:
            for ln in o["lines"]:
                h.update(np.round(ln["P"], 4).astype(np.float32).tobytes())
    return h.hexdigest()


@pytest.mark.parametrize("arch", ["layered", "braided"])
def test_registry_controls_change_geometry(arch):
    """Every scalar design parameter that applies to the demo changes the geometry or
    the reported design (audit; parameters that are inactive in this mode are skipped).
    Run for the six-level construct and for the gap-filling braid."""
    base = D.new_state()
    base["params"]["scaffold.arch"] = arch
    b0 = build_scene(base, "preview")
    g0 = _all_geometry(b0)
    skip_mode = set(D.inactive_params(base["params"])) | {
        # only matter when a remnant is shorter than the bone-reach limit (not in the demo)
        "scaffold.bone_reach_mm", "fit.end_transition_mm",
        # the demo's whole-section repair has no AM/PL interface constraint
        "fit.interface_clearance_mm",
    }
    changed, unchanged = [], []
    for p in D.REGISTRY:
        if p["kind"] not in ("scalar", "int") or p["id"] in skip_mode:
            continue
        st = copy.deepcopy(base)
        v = st["params"][p["id"]]
        lo, hi = p["min"], p["max"]
        nv = v + 0.15 * (hi - lo) if v + 0.15 * (hi - lo) <= hi else v - 0.15 * (hi - lo)
        if p["kind"] == "int":
            nv = int(round(nv))
            if nv == v:
                nv = v + 1 if v + 1 <= hi else v - 1
        st["params"][p["id"]] = nv
        sc = build_scene(st, "preview")
        if not sc["ok"]:
            continue
        diff = (_all_geometry(sc) != g0 or json.dumps(sc["metrics"], sort_keys=True, default=str) != json.dumps(b0["metrics"], sort_keys=True, default=str)
                or json.dumps(sc["checks"], sort_keys=True, default=str) != json.dumps(b0["checks"], sort_keys=True, default=str))
        (changed if diff else unchanged).append(p["id"])
    assert not unchanged, unchanged


# ---------------------------------------------------------------- 3A: containment and fit
from acl_engine.v2.ligament import Ligament, points_in_polygon, polygon_distance, sunflower
from acl_engine.v2.bones import ProceduralBone
from scipy.spatial import cKDTree as _KD


def _strand_protrusion(sc):
    """Independent check: strand surfaces vs the section contour of the nearest station
    (body region; insertion flares are covered by the engine's 3D envelope check)."""
    lig = sc["lig"]
    o = [o for o in sc["objects"] if o["id"] == "scaffold_yarns"]
    if not o:
        return -1.0
    worst = -9.0
    M = np.array([-1.0, 1, 1]) if sc["side"] == "left" else 1.0
    fl_t, fl_f = lig.p["lig.flare_tib_mm"], lig.p["lig.flare_fem_mm"]
    for y in o[0]["lines"]:
        P = y["P"] * M
        idx = _KD(lig.C).query(P)[1]
        for k in range(0, P.shape[0], 2):
            i = idx[k]
            if not fl_t < lig.s[i] * lig.L < lig.L - fl_f:
                continue
            Q = P[k] - lig.C[i]
            x = np.array([[Q @ lig.N1[i], Q @ lig.N2[i]]])
            d = polygon_distance(lig.contour[i], x)[0]
            pen = (-d if points_in_polygon(lig.contour[i], x)[0] else d) + y["r"][k]
            worst = max(worst, pen)
    return worst


@pytest.mark.parametrize("case,upd", [
    ("demo_complete", {}),
    ("demo_partial_am", {}),
    ("demo_proximal", {}),
    ("demo_complete", {"lig.size_scale": 1.4}),
    ("demo_complete", {"lig.section_mode": "area_aspect", "lig.taper_fem": 0.25}),
    ("demo_complete", {"fit.partition": "per_bundle", "scaffold.yarn_count": 12, "scaffold.yarn_diameter_mm": 0.35}),
    ("demo_complete", {"injury.type": "partial", "injury.involvement": "per_bundle", "injury.pl.depth_fraction": 0.5,
                       "scaffold.arch": "aligned", "scaffold.yarn_diameter_mm": 0.3, "scaffold.yarn_count": 12}),
    ("demo_complete", {"injury.center_s": 0.85, "injury.gap_mm": 3}),
])
def test_scaffold_fits_repair_space(case, upd):
    import server_v2 as V
    st = braided(V.new_case(case))
    st["params"].update(upd)
    sc = build_scene(st, "full")
    assert sc["ok"] and sc["validity"]["valid"], sc["validity"]["failed"]
    F = sc["metrics"]["fit"]
    assert F["volume_outside_envelope_mm3"] <= 1e-3 and F["volume_outside_repair_space_mm3"] <= 1e-3
    assert F["interface_crossings"] == 0 and F["samples_beyond_overlap"] == 0 and not F["unit_collisions"]
    assert F["min_envelope_clearance_mm"] >= float(st["params"]["fit.boundary_clearance_mm"]) - 0.03
    assert _strand_protrusion(sc) <= 0.02


def test_partial_tear_scaffold_stays_in_its_bundle():
    import server_v2 as V
    sc = build_scene(V.new_case("demo_partial_am"), "full")
    lig = sc["lig"]
    ys = [o for o in sc["objects"] if o["id"] == "scaffold_yarns"][0]["lines"]
    for y in ys:
        i = _KD(lig.C).query(y["P"])[1]
        Q = y["P"] - lig.C[i]
        X = np.stack([(Q * lig.N1[i]).sum(1), (Q * lig.N2[i]).sum(1)], 1)
        d = lig.interface_distance(lig.s[i], X)          # + on the AM side
        assert (d - y["r"] >= -0.02).all()
    assert chk(sc, "partial_preserves")["status"] == "pass"


def test_unfittable_design_is_reported_not_shrunk():
    import server_v2 as V
    from acl_engine.v2.scene import verify_resolutions
    st = braided(V.new_case("demo_complete"))
    st["params"].update({"scaffold.yarn_count": 24, "scaffold.yarn_diameter_mm": 0.55})
    sc = build_scene(st, "full")
    assert not sc["validity"]["valid"]
    # requested values are kept: nothing thinned or removed to force a fit
    assert sc["metrics"]["scaffold"]["braid"][0]["layers"][0]["per_direction"] == 12
    assert abs(sc["metrics"]["scaffold"]["braid"][0]["layers"][0]["max_yarn_d_mm"] - 0.55 * 1.15) < 1e-9
    fixes = verify_resolutions(sc)
    assert fixes and all(r["verified"] for r in fixes)
    st2 = copy.deepcopy(st)
    st2["params"][fixes[0]["param"]] = fixes[0]["value"]
    assert build_scene(st2, "full")["validity"]["valid"]


def test_small_template_braid_does_not_fit_and_says_why():
    import server_v2 as V
    st = braided(V.new_case("demo_complete"))
    st["params"]["lig.size_scale"] = 0.7
    sc = build_scene(st, "preview")
    assert not sc["validity"]["valid"]
    assert any("does not fit" in c["message"] for c in sc["checks"] if c["status"] == "fail")


def _lig(upd=None):
    st = D.new_state()
    st["params"].update(upd or {})
    p = st["params"]
    return Ligament(p, ProceduralBone("femur", p, 0), ProceduralBone("tibia", p, 0))


@pytest.mark.parametrize("upd", [{}, {"lig.contour_irregularity_mm": 0.6}, {"lig.section_mode": "per_bundle"},
                                 {"bundle.interface_curvature": 0.15}, {"lig.section_mode": "area_aspect"}])
def test_section_map_is_one_to_one_and_partitions(upd):
    lig = _lig(upd)
    assert not [i for i in lig.issues if i["level"] == "fail"]
    M = sunflower(3000, 0.999)
    s = np.full(M.shape[0], 0.5)
    X = lig.material_to_metric(s, M)
    assert np.abs(lig.metric_to_material(s, X) - M).max() < 1e-6          # invertible (no folds)
    assert points_in_polygon(lig.contour_at(0.5), X).mean() > 0.995        # material disk -> inside the contour
    assert (lig.metric_bundle(s, X) == lig.bundle_of(M)).all()              # AM/PL regions tile the section
    assert np.allclose(lig.area_bundle.sum(1), lig.area)


def test_area_is_derived_from_the_contour():
    lig = _lig({"lig.section_mode": "area_aspect", "lig.area_profile": [50, 45, 40, 38, 36], "lig.aspect_profile": [2, 2, 2.5, 3, 3]})
    i = lig.s.size // 2
    assert abs(lig.area[i] - 40) < 0.6                                      # contour area = requested area
    assert abs(lig.W[i] / lig.Tk[i] - 2.5) < 0.05
    lig2 = _lig({"bundle.am_fraction_mode": "profile", "bundle.am_fraction_profile": [0.4, 0.5, 0.6, 0.65, 0.7]})
    assert abs(lig2.am_fraction_actual[0] - 0.4) < 0.01 and abs(lig2.am_fraction_actual[-1] - 0.7) < 0.01


def test_length_target_keeps_attachments_locked():
    a = _lig()
    b = _lig({"lig.length_mode": "target", "lig.length_target_mm": a.L + 3})
    assert abs(b.L - (a.L + 3)) < 0.05
    assert np.allclose(a.fp["tib"].center, b.fp["tib"].center) and np.allclose(a.fp["fem"].center, b.fp["fem"].center)
    c = _lig({"lig.length_mode": "target", "lig.length_target_mm": 20})
    assert any(i["param"] == "lig.length_target_mm" and i["level"] == "fail" for i in c.issues)


def test_footprint_am_share_is_applied():
    lig = _lig({"fp.tib.am_share": 0.35})
    F = sunflower(4000, 0.99)
    assert abs((lig.footprint_bundle("tib", F) == 0).mean() - 0.35) < 0.03


def test_sections_are_cut_from_the_built_geometry():
    import server_v2 as V
    from acl_engine.v2 import sections as secmod
    sc = build_scene(braided(V.new_case("demo_complete")), "full")
    lig = sc["lig"]
    s = 0.55
    sec = secmod.section(sc["objects"], lig, "ligament", s, sc["side"])
    env = next(L for L in sec["layers"] if L["id"] == "intact_reference")
    P = np.array(env["segs"]).reshape(-1, 2)
    i = int(round(s * (lig.s.size - 1)))
    assert polygon_distance(lig.contour[i], P).max() < 0.05               # envelope slice = section contour
    ys = [o for o in sc["objects"] if o["id"] == "scaffold_yarns"][0]["lines"]
    crossing = sum(1 for y in ys if ((y["P"] - lig.C[i]) @ lig.T[i]).min() < 0 < ((y["P"] - lig.C[i]) @ lig.T[i]).max())
    assert len(sec["yarns"]) >= crossing > 0
    for kind in ("am", "pl", "axial", "sagittal", "coronal"):
        sx = secmod.section(sc["objects"], lig, kind, 0.5 if kind in ("am", "pl") else 0.0, sc["side"])
        assert sx["layers"] and sx["axes"]["x"] and sx["title"]


def test_retired_sleeve_is_migrated():
    st = D.new_state()
    st["params"]["scaffold.engagement"] = "sleeve"
    del st["params"]["fit.boundary_clearance_mm"]
    sc = build_scene(st, "preview")
    assert sc["ok"] and any(c["id"] == "migration" for c in sc["checks"])


def test_import_measured_contours_and_per_bundle_tears():
    from acl_engine.v2.patient import import_document
    lig = _lig()
    items = []
    for sv, (a, b) in ((0.2, (5.6, 2.9)), (0.5, (4.6, 2.6)), (0.8, (5.4, 2.0))):
        i = int(round(sv * (lig.s.size - 1)))
        ph = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        r = 1 + 0.06 * np.sin(5 * ph)                      # measured outlines are irregular
        X = np.stack([a * r * np.cos(ph), b * r * np.sin(ph)], 1)
        P = lig.C[i] + X[:, :1] * lig.N1[i] + X[:, 1:] * lig.N2[i]
        items.append({"s": sv, "points": P.tolist(), "frame": "model"})
    doc = {"schema": "acl-mri-input", "schema_version": "1.0.0", "case_id": "contours", "synthetic": True,
           "imaging": {"laterality": "right"},
           "ligament": {"section_contours": items},
           "injury": {"type": "partial", "involvement": "per_bundle",
                      "bundles": {"AM": {"location_s": 0.55, "gap_length_mm": 4, "depth_fraction": 1.0},
                                  "PL": {"location_s": 0.45, "gap_length_mm": 3, "depth_fraction": 0.0}}},
           "scaffold_design": {"architecture": "aligned", "yarn_diameter_mm": 0.3, "yarn_count": 12}}
    st, rep = import_document(doc, ".")
    assert st is not None, rep["errors"]
    p = st["params"]
    assert p["lig.section_mode"] == "imported" and len(p["lig.section_contours"]) == 3
    assert p["injury.involvement"] == "per_bundle" and p["injury.am.gap_mm"] == 4
    sc = build_scene(st, "full")
    assert sc["ok"]
    L2 = sc["lig"]
    i = int(round(0.5 * (L2.s.size - 1)))
    assert abs(L2.area[i] - np.pi * 4.6 * 2.6) < 0.08 * np.pi * 4.6 * 2.6     # measured area is used
    assert sc["validity"]["valid"], sc["validity"]["failed"]
    assert [u["id"] for u in sc["metrics"]["scaffold"]["units"]] == ["AM"]    # PL depth 0: untouched


# ---------------------------------------------------------------- surgeon-informed fiber rotation (+-30 deg per unit)
ROT_IDS = ["native.fascicle_rotation_deg", "native.fiber_rotation_deg", "am.fascicle_rotation_deg", "pl.fascicle_rotation_deg",
           "am.fiber_rotation_deg", "pl.fiber_rotation_deg", "scaffold.yarn_rotation_deg", "scaffold.fiber_rotation_deg"]


def test_rotation_controls_label_range_default():
    for pid in ROT_IDS:
        spec = D.BY_ID[pid]
        assert spec["label"] == "Total fiber rotation along unit (°)"
        assert spec["description"].startswith("End-to-end rotation around the local axis; surgeon-informed design range: −30° to +30°.")
        assert (spec["min"], spec["max"], spec["default"]) == (-30, 30, 0.0)
    st = D.new_state()
    st["params"]["scaffold.fiber_rotation_deg"] = 30.5
    sc = build_scene(st, "preview")
    assert not sc["ok"] and sc["errors"][0]["param"] == "scaffold.fiber_rotation_deg"     # rejected, never clamped
    assert not any(k for k in D.BY_ID if "helix" in k or "handedness" in k)               # no helix-angle controls left


def _rot_rows(sc):
    return {(r["level"].split(":")[0], r["unit"]): r for r in sc["metrics"]["rotation"]["rows"] if r.get("n")}


def test_generated_paths_rotate_by_the_requested_total():
    import server_v2 as V
    st = V.new_case("demo_partial_am")
    st["params"].update({"native.fascicle_rotation_deg": 20, "native.fiber_rotation_deg": -30,
                         "scaffold.yarn_rotation_deg": 25, "scaffold.fiber_rotation_deg": 30})
    sc = build_scene(st, "full")
    R = _rot_rows(sc)
    want = {("native bundle", "AM"): 20, ("native bundle", "PL"): 20, ("native fascicle", "AM fascicles"): -30,
            ("native fascicle", "PL fascicles"): -30, ("scaffold repair unit", "AM repair"): 25, ("scaffold yarn", "every yarn"): 30}
    for k, v in want.items():
        r = R[k]
        assert abs(r["net_mean_deg"] - v) < 1.0, (k, r)
        assert r["net_max_abs_deg"] <= 30 + 1.0 and r["excursion_max_deg"] < 1.0, (k, r)       # within range, monotone
    assert not [c for c in sc["checks"] if c["category"] == "rotation" and c["status"] == "fail"]


def test_parent_rotation_is_not_applied_twice():
    import server_v2 as V
    st = V.new_case("demo_complete")
    st["params"].update({"bundle.rotation_total_deg": 60, "scaffold.arch": "aligned", "scaffold.yarn_diameter_mm": 0.3})
    sc = build_scene(st, "full")
    R = _rot_rows(sc)
    for b in ("AM", "PL"):
        assert abs(R[("native bundle", b)]["net_at_max_deg"]) < 1.0           # arrangement turns ~60 deg; contents add nothing
        assert abs(R[("native bundle", b)]["region_turn_deg"]) > 40
        assert R[("native fascicle", b + " fascicles")]["net_max_abs_deg"] < 1.5
    assert abs(R[("scaffold repair unit", "Whole-section repair")]["net_at_max_deg"]) < 1.0
    assert R[("scaffold yarn", "every yarn")]["net_max_abs_deg"] < 1.0


def test_independent_bundles_and_braid_winding_reported():
    import server_v2 as V
    st = braided(V.new_case("demo_complete"))
    st["params"].update({"bundle.structure_mode": "independent", "am.fascicle_rotation_deg": 30, "pl.fascicle_rotation_deg": -15,
                         "am.fiber_rotation_deg": 12, "pl.fiber_rotation_deg": -30})
    sc = build_scene(st, "full")
    R = _rot_rows(sc)
    assert abs(R[("native bundle", "AM")]["net_mean_deg"] - 30) < 1 and abs(R[("native bundle", "PL")]["net_mean_deg"] + 15) < 1
    assert abs(R[("native fascicle", "AM fascicles")]["net_mean_deg"] - 12) < 1
    braid = [r for r in sc["metrics"]["rotation"]["rows"] if r.get("braid_winding")]
    assert braid and braid[0]["net_max_abs_deg"] > 30
    assert any(c["id"].startswith("rotation_braid") and c["status"] == "warn" for c in sc["checks"])


def test_unwrapped_measurement_counts_full_turns():
    from acl_engine.v2 import rotation as RM
    sig = np.linspace(0, 20, 400)
    axis = np.stack([np.zeros_like(sig), np.zeros_like(sig), sig], 1)
    for total in (25.0, -30.0, 400.0):
        a = np.radians(total) * sig / sig[-1] + 0.3
        P = np.stack([np.cos(a), np.sin(a), sig], 1)
        net, exc, _ = RM.unit_rotation([(P, sig)], lambda sg: axis, 0.1)
        assert abs(abs(net) - abs(total)) < 0.5 and exc < 0.5            # 400 deg is NOT reported as 40 deg
    # back-and-forth that cancels numerically is caught as excursion
    a = np.radians(80) * np.sin(np.pi * sig / sig[-1])
    P = np.stack([np.cos(a), np.sin(a), sig], 1)
    net, exc, _ = RM.unit_rotation([(P, sig)], lambda sg: axis, 0.1)
    assert abs(net) < 1 and exc > 60


def test_retired_helix_parameters_are_migrated_with_a_note():
    st = D.new_state()
    st["params"]["native.fiber_helix_deg"] = 2.0
    st["params"]["scaffold.fiber_helix_deg"] = 12.0
    st["params"]["am.rotation_total_deg"] = 10.0
    sc = build_scene(st, "preview")
    assert sc["ok"]
    notes = " ".join(c["message"] for c in sc["checks"] if c["id"] == "migration")
    assert "helix angle" in notes


def test_importer_maps_total_rotation_and_flags_helix_angles():
    from acl_engine.v2.patient import import_document
    doc = {"schema": "acl-mri-input", "schema_version": "1.0.0", "case_id": "r", "synthetic": True, "imaging": {"laterality": "right"},
           "structure": {"fascicle_rotation_deg": {"value": 12, "units": "deg", "status": "inferred"}, "fiber_angle_deg": 3}}
    st, rep = import_document(doc, ".")
    assert st["params"]["native.fascicle_rotation_deg"] == 12
    assert any("helix angle" in w for w in rep["warnings"])
