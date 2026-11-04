"""T1 CLI: build the scaffold, place it in a schematic knee, run checks, write STL + GLB + JSON.

Written by Muse (N2, T1). Fixes are marked "FIX".
Run:  python cli.py --request fixtures/acl_context_request.json --out outputs/acl
Other code can call run_request(norm, mig, quality, out) directly (check.py does).
"""
import argparse
import copy
import json
import math
import os
import time
import numpy as np
from acl_engine import GEOM_KERNEL, SOLVER
from acl_engine.schema import validate_request, canonical_hash
from acl_engine.paths import (PhaseField, find_crossings, arc_length_interval,
                              mean_lambda_interval, straight_pos_vel_acc)
from acl_engine.fold import slope_factor_q, gc_cross
from acl_engine.axis import eval_axis
from acl_engine.frames import compute_rmf
from acl_engine.sweep import (segments_around, build_sampling_y,
                              sweep_strand, signed_volume)
from acl_engine.measure import curvature_max_straight, min_surface_gap
from acl_engine.context import (build_native_axis, measured_inclinations,
                                build_envelope_fields, basis_for_dir, box_mesh,
                                ellipsoid_mesh, cylinder_mesh, ghost_mesh,
                                containment_of_points, footprint_coverage,
                                dist_point_box)
from acl_engine.export import write_stl, write_glb, write_json, read_stl_vertices
from acl_engine.controls import resolve_rates_closed_loop
# Milestone A: anatomical ACL + tear geometry (new modules, schematic path untouched)
from acl_engine.anatomy import (build_bundle_fields, build_bundle_meshes,
                                containment_union, max_bundle_radius,
                                anatomical_context)
from acl_engine.footprints import (tibial_footprint_mesh, femoral_footprint_mesh,
                                   femoral_crescent_coverage)
from acl_engine.tear import build_stumps
from acl_engine.anatomy_build import (build_anatomy, build_loft_tear,
                                      anatomy_payload, validation_report)
from acl_engine.loft import apply_flexion as _apply_flexion_pts


def run_request(norm, mig, quality, out, skip_stl=False):
    """skip_stl=True (N3 preview) skips scaffold.stl. timings_s reports each stage."""
    t0 = time.time()
    tm = {}
    s1 = time.time()
    a = float(norm["_derived"]["swing_mm"])
    b = float(norm["_derived"]["fold_depth_mm"])
    q = float(norm["_derived"]["fold_sharpness"])
    angle_mode = norm["_derived"]["angle_mode"]
    d = float(norm["strand"]["diameter_mm"])
    r = d / 2.0
    phase_offset = math.radians(float(norm["architecture"].get("phase_offset_deg", 0.0)))
    mirror = bool(norm["architecture"].get("mirror", False))
    zones = norm["zones"]
    L = float(norm["_derived"]["axis_length_mm"])
    axis_spec = norm.get("axis", {"type": "straight"})
    tm["setup"] = time.time() - s1
    s1 = time.time()
    rates, achieved = resolve_rates_closed_loop(norm, a, b, q, angle_mode, phase_offset, mirror)
    lens = [float(z["length_mm"]) for z in zones]
    trans = float(norm["transitions"]["length_mm"])
    pf = PhaseField(lens, rates, trans)
    S = slope_factor_q(a, b, q)
    p_events = [math.pi / (3.0 * rr) for rr in rates]
    phaseL = pf.phi_L()
    equiv = phaseL / (math.pi / 3.0)
    events = find_crossings(pf, a, b, q, phase_offset, mirror, L)
    arcs = [arc_length_interval(0.0, L, pf, a, b, q, phase_offset, mirror, i) for i in range(3)]
    # FIX: Muse's T1 measured fiber length on pf.constant_interval(), which does not
    # exist. Measure the whole zone, mean of the 3 strands (same as the closed-loop solve).
    zlams = []
    for k, z in enumerate(zones):
        y0, y1 = pf.bounds[k], pf.bounds[k + 1]
        per = [mean_lambda_interval(y0, y1, pf, a, b, q, phase_offset, mirror, i) for i in range(3)]
        lam = sum(per) / 3.0
        zlams.append({"id": z["id"], "control": z.get("control"),
                      "path_length_ratio_mean": float(lam),
                      "path_length_ratio_per_strand": [float(v) for v in per],
                      "eps_derived_geometry": float(lam - 1.0)})
    tm["rates_crossings_lengths"] = time.time() - s1
    s1 = time.time()
    tol = float(norm["quality"]["export_tol_mm"]) if quality == "export" \
        else float(norm["quality"]["preview_tol_mm"])
    nsec = segments_around(r, tol)
    ss_fine = np.linspace(0.0, L, 2001)
    kmax = curvature_max_straight(ss_fine, pf, a, b, q, phase_offset, mirror)
    pmin = min(p_events)
    s_s = build_sampling_y(L, pf.bounds, pf.windows, events, pmin, kmax, tol)
    phi_s = pf.phi_array(s_s)
    rate_s = pf.rate_array(s_s)
    rate2_s = pf.rate2_array(s_s)
    Cax, N1ax, N2ax, Tax = eval_axis(s_s, axis_spec)
    local_centers = []
    for i in range(3):
        pos_l, vel_l, _a = straight_pos_vel_acc(s_s, phi_s, rate_s, rate2_s, a, b, q, phase_offset, mirror, i)
        local_centers.append(Cax + pos_l[:, 0][:, None] * N1ax - pos_l[:, 2][:, None] * N2ax)
    s_coarse = np.linspace(0.0, L, max(100, int(L / 0.2)))
    phi_c = pf.phi_array(s_coarse)
    rate_c = pf.rate_array(s_coarse)
    rate2_c = pf.rate2_array(s_coarse)
    Cc, N1c, N2c, _T = eval_axis(s_coarse, axis_spec)
    wc0 = []
    for i in range(3):
        pos_l, _v, _a = straight_pos_vel_acc(s_coarse, phi_c, rate_c, rate2_c, a, b, q, phase_offset, mirror, i)
        wc0.append(Cc + pos_l[:, 0][:, None] * N1c - pos_l[:, 2][:, None] * N2c)
    tm["sampling"] = time.time() - s1
    s1 = time.time()
    gap = min_surface_gap(wc0, s_coarse, r, d)
    tm["gap"] = time.time() - s1
    s1 = time.time()
    gap_mm = gap.get("min_surface_mm")
    acl_len = float(norm["_derived"]["acl_length_mm"])
    Wmid = float(norm["_derived"]["envelope_width_mm"])
    Tmid = float(norm["_derived"]["envelope_thickness_mm"])
    sag = float(norm["_derived"]["sagittal_deg"])
    cor = float(norm["_derived"]["coronal_deg"])
    flare = float(norm["_derived"]["flare_ratio"])
    prof = norm["_derived"]["area_profile_femur_to_tibia"]
    side = norm["_derived"]["side"]
    realism = norm["_derived"].get("realism", "schematic")
    twist_total = float(norm["_derived"].get("anatomical_twist_total_deg", 0.0))
    twist_dist = norm["_derived"].get("twist_distribution", "uniform")
    flexion_deg = float(norm.get("knee_scene", {}).get("flexion_deg", 0.0))
    Ttib, Ffem, u = build_native_axis(acl_len, sag, cor, side)
    sag_m, cor_m = measured_inclinations(u)
    flat = any(z.get("cross_section", "round") == "flat" for z in zones)
    ss_env, A_env, Ws_env, Ts_env, Amid = build_envelope_fields(acl_len, Wmid, Tmid, prof, flare, 8.0, flat)
    flare_m = float(A_env[0] / Amid)
    e1, eu, e3 = basis_for_dir(u, side)
    # Milestone A: anatomical bundle fields (skipped entirely when schematic)
    bundle_data = None
    if realism == "anatomical":
        bundle_data = build_bundle_fields(Ttib, u, e1, e3, ss_env, Ws_env, Ts_env,
                                           twist_total, twist_dist)
        bundle_data["max_bundle_rn"] = max_bundle_radius(bundle_data)
    # Parametric loft anatomy (intact reference object; schematic skips it)
    anat = None
    regen_ok = None
    if realism == "anatomical":
        anat = build_anatomy(norm, Ttib, Ffem, u, e1, e3, ss_env, Ws_env, Ts_env,
                             side, quality=quality, with_tubes=True)
        _re = build_anatomy(norm, Ttib, Ffem, u, e1, e3, ss_env, Ws_env, Ts_env,
                            side, quality=quality, with_tubes=False)
        regen_ok = bool(_re["regen_hash"] == anat["regen_hash"])
    lig_idx = len(zones) // 2
    Lpre = sum(lens[:lig_idx])

    def local_to_world(P):
        P = np.asarray(P, dtype=float)
        return Ttib + P[:, 0][:, None] * e1 + (P[:, 1] - Lpre)[:, None] * eu + P[:, 2][:, None] * e3

    tm["context"] = time.time() - s1
    s1 = time.time()
    bodies_world = []
    vols = []
    for i in range(3):
        grad = np.gradient(local_centers[i], s_s, axis=0, edge_order=2)
        N1, N2, _T = compute_rmf(local_centers[i], grad)
        V, F = sweep_strand(local_centers[i], N1, N2, r, nsec)
        Vw = local_to_world(V)
        bodies_world.append((Vw, F))
        vols.append(signed_volume(Vw, F))
    tm["sweep"] = time.time() - s1
    s1 = time.time()
    lig_mask = (s_coarse >= pf.bounds[lig_idx]) & (s_coarse <= pf.bounds[lig_idx + 1])
    s_lig = s_coarse[lig_mask] - pf.bounds[lig_idx]
    lig_pts = []
    for i in range(3):
        Ww = local_to_world(wc0[i][lig_mask])
        lig_pts.append(Ww)
        for ang in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False):
            lig_pts.append(Ww + r * (math.cos(ang) * e1 + math.sin(ang) * e3))
    Pall = np.vstack(lig_pts)
    sall = np.concatenate([s_lig] * (3 * 9))
    pen, penloc = containment_of_points(Pall, sall, Ttib, u, e1, e3, ss_env, Ws_env, Ts_env)
    # Milestone A: containment uses the bundle union when anatomical
    if realism == "anatomical" and bundle_data is not None:
        pen, penloc = containment_union(Pall, sall, bundle_data)
    Ws_tib = 2.0 * a + d
    Ts_tib = 2.0 * b + d
    cov_tib = footprint_coverage(Ws_tib, Ts_tib, 11.0, 17.0)
    cov_fem = footprint_coverage(Ws_tib, Ts_tib, 10.0, 17.0)
    # Milestone A: femoral coverage against the crescent footprint when anatomical
    cov_fem_anat = None
    if realism == "anatomical":
        cov_fem_anat = femoral_crescent_coverage(Ws_tib, Ts_tib, Ffem, side)
    tib_box_min = np.array([-37.5, -25.0, -10.0])
    tib_box_max = np.array([37.5, 25.0, 0.0])
    # FIX: the ligament starts exactly on the plateau, so its first ring of points
    # always touched the tibia box and the check always warned. Ignore the first 0.5 mm.
    above = Pall[:, 2] > 0.5
    d_tib, inside_tib = dist_point_box(Pall[above], tib_box_min, tib_box_max)
    notch_min = np.array([-9.0, -15.0, float(Ffem[2]) + 8.0 - 2.0])
    notch_max = np.array([9.0, 5.0, float(Ffem[2]) + 8.0 + 2.0])
    d_notch, _in = dist_point_box(Pall, notch_min, notch_max)

    tm["containment"] = time.time() - s1
    s1 = time.time()
    os.makedirs(out, exist_ok=True)
    if not skip_stl:
        write_stl(os.path.join(out, "scaffold.stl"), bodies_world)
    # Separately exportable anatomy (closed solids only; same mm units)
    if anat is not None and not skip_stl:
        write_stl(os.path.join(out, "anatomy.stl"),
                  [anat["surface"], anat["tibial_patch"]])
    tm["stl"] = time.time() - s1
    s1 = time.time()
    Vt, Ft = box_mesh(np.array([0.0, 0.0, -5.0]), np.array([75.0, 50.0, 10.0]))
    # FIX: Muse placed the condyles at x = +/-15 mm with 12 mm radius (again in T1),
    # leaving a 6 mm notch, so the ACL ran straight through the lateral condyle. Condyles moved
    # apart so the ACL sits in the notch and the femoral footprint lands on the lateral wall.
    lat_x = float(Ffem[0]) + 11.5 if side == "right" else float(Ffem[0]) - 11.5
    med_x = -18.0 if side == "right" else 18.0
    Vl, Fl = ellipsoid_mesh(np.array([lat_x, -5.0, 25.0]), np.array([12.0, 18.0, 16.0]))
    Vm, Fm = ellipsoid_mesh(np.array([med_x, -5.0, 25.0]), np.array([12.0, 18.0, 16.0]))
    Vg, Fg = ghost_mesh(Ttib, u, e1, e3, ss_env, Ws_env, Ts_env)
    # Milestone A: bundle ghosts, footprints and stumps (anatomical only)
    extra_named = []
    stump_info = None
    if realism == "anatomical" and bundle_data is not None:
        bundle_meshes = build_bundle_meshes(bundle_data, nsec=32)
        bundle_data["meshes_built"] = True
        for _bn in ("AM", "PL"):
            extra_named.append({"name": "acl_ghost_%s" % _bn.lower(),
                                "vertices": bundle_meshes[_bn][0],
                                "faces": bundle_meshes[_bn][1]})
        if anat is None:
            # Legacy fallback (kept so old bundle/footprint outputs still exist)
            _Vt, _Ft = tibial_footprint_mesh(Ttib)
            extra_named.append({"name": "footprint_tibial", "vertices": _Vt, "faces": _Ft})
            _Vf, _Ff = femoral_footprint_mesh(Ffem, side)
            extra_named.append({"name": "footprint_femoral", "vertices": _Vf, "faces": _Ff})
            if norm["tear"].get("severity", "intact") != "intact":
                stump_info = build_stumps(bundle_data, bundle_meshes, norm["tear"])
                extra_named.append({"name": "stump_tibial",
                                    "vertices": stump_info["stump_tibial"][0],
                                    "faces": stump_info["stump_tibial"][1]})
                extra_named.append({"name": "stump_femoral",
                                    "vertices": stump_info["stump_femoral"][0],
                                    "faces": stump_info["stump_femoral"][1]})
    # Loft anatomy meshes: intact reference surface + strands + on-bone
    # footprints + boundary loops; injury object cut from a loft copy.
    loft_tear = None
    if anat is not None:
        extra_named.append({"name": "acl_surface",
                            "vertices": anat["surface"][0], "faces": anat["surface"][1]})
        for _bn in ("AM", "PL"):
            if _bn in anat["strand_tubes"]:
                extra_named.append({"name": "acl_strands_%s" % _bn.lower(),
                                    "vertices": anat["strand_tubes"][_bn][0],
                                    "faces": anat["strand_tubes"][_bn][1]})
        extra_named.append({"name": "footprint_tibial",
                            "vertices": anat["tibial_patch"][0],
                            "faces": anat["tibial_patch"][1]})
        extra_named.append({"name": "footprint_femoral",
                            "vertices": anat["femoral_patch"][0],
                            "faces": anat["femoral_patch"][1]})
        if anat["edge_tibial"][0].shape[0]:
            extra_named.append({"name": "acl_edge_tibial",
                                "vertices": anat["edge_tibial"][0],
                                "faces": anat["edge_tibial"][1]})
        if anat["edge_femoral"][0].shape[0]:
            extra_named.append({"name": "acl_edge_femoral",
                                "vertices": anat["edge_femoral"][0],
                                "faces": anat["edge_femoral"][1]})
        if norm["tear"].get("severity", "intact") != "intact":
            loft_tear = build_loft_tear(anat, norm["tear"],
                                        observed=bool(norm["tear"].get("observed", False)))
            stump_info = loft_tear
            extra_named.append({"name": "stump_tibial",
                                "vertices": loft_tear["stump_tibial"][0],
                                "faces": loft_tear["stump_tibial"][1]})
            extra_named.append({"name": "stump_femoral",
                                "vertices": loft_tear["stump_femoral"][0],
                                "faces": loft_tear["stump_femoral"][1]})
    Rtun = (2.0 * a + d) / 2.0 + 0.5
    Vtib_tun, Ftib_tun = cylinder_mesh(Ttib - 20.0 * u, Ttib, Rtun)
    Vfem_tun, Ffem_tun = cylinder_mesh(Ffem, Ffem + 20.0 * u, Rtun)
    # Flexion pose (illustrative rigid transform of the femoral side; no-op at 0 deg)
    if abs(flexion_deg) > 1e-12 and anat is not None:
        _ax = np.array(anat["flexion"]["axis_point_mm"], dtype=float)
        Vl = _apply_flexion_pts(Vl, flexion_deg, _ax)
        Vm = _apply_flexion_pts(Vm, flexion_deg, _ax)
        Vfem_tun = _apply_flexion_pts(Vfem_tun, flexion_deg, _ax)
    named = [{"name": "scaffold_strand_%d" % i, "vertices": bodies_world[i][0], "faces": bodies_world[i][1]}
             for i in range(3)]
    named += [{"name": "acl_ghost", "vertices": Vg, "faces": Fg},
              {"name": "bone_tibia", "vertices": Vt, "faces": Ft},
              {"name": "bone_femur_lat", "vertices": Vl, "faces": Fl},
              {"name": "bone_femur_med", "vertices": Vm, "faces": Fm},
              {"name": "tunnel_tibial", "vertices": Vtib_tun, "faces": Ftib_tun},
              {"name": "tunnel_femoral", "vertices": Vfem_tun, "faces": Ffem_tun}]
    named += extra_named  # Milestone A (empty list when schematic)
    write_glb(os.path.join(out, "knee.glb"), named)

    assumptions = norm.get("_assumptions", [])
    resolved = copy.deepcopy(norm)
    resolved["_resolved"] = {"kernel": GEOM_KERNEL, "solver": SOLVER, "quality": quality,
                             "tol_mm": tol, "nsec": nsec, "S_mm": S, "gc": float(gc_cross(q)),
                             "rates_per_mm": rates, "p_event_mm": p_events,
                             "phase_L_rad": phaseL, "phase_cycles": phaseL / (2.0 * math.pi),
                             "crossing_equivalents": equiv, "events": events,
                             "arc_lengths_mm": arcs, "zone_lambda": zlams,
                             "fiber_length_closed_loop": achieved,
                             "mesh_volumes_mm3": vols, "kmax_per_mm": kmax,
                             "gap": gap, "migration": mig,
                             "labels": {"lambda_eps": "derived geometry, not mechanics"}}
    resolved["design_hash"] = canonical_hash(norm)
    resolved["assumptions_used"] = assumptions
    write_json(os.path.join(out, "parameters.json"), resolved)
    write_json(os.path.join(out, "migration_report.json"),
               {"from": mig["from"], "to": mig["to"], "notes": mig["notes"],
                "assumptions_used": assumptions})
    context = {"knee_frame": "+X lateral, +Y anterior, +Z proximal, origin tibial plateau center, right knee mirrored for left. Schematic, not patient anatomy.",
               "tibial_center_mm": [float(v) for v in Ttib], "femoral_center_mm": [float(v) for v in Ffem],
               "axis_dir": [float(v) for v in u], "sagittal_measured_deg": sag_m, "coronal_measured_deg": cor_m,
               "sagittal_target_deg": sag, "coronal_target_deg": cor, "acl_length_mm": acl_len,
               "midsubstance_area_mm2": Amid, "footprint_area_mm2": float(A_env[0]),
               "flare_measured": flare_m, "flare_target": flare,
               "native_envelope": "elliptical sweep, area profile L01, flare L04, ribbon flatten L02 if requested; engineering assumption",
               "containment": {"max_penetration_mm": pen, "location": penloc},
               "footprint_coverage": {"tibial": cov_tib, "femoral": cov_fem, "note": "reported, no threshold"},
               "bone_clearance": {"min_dist_ligament_to_tibia_box_mm": float(d_tib), "inside_tibia_box": bool(inside_tib),
                                  "note": "schematic; tunnels excluded by construction"},
               "notch_distance_schematic_mm": float(d_notch)}
    # Milestone A: realism flag + anatomical payload (gap, stumps, bundles)
    context["realism"] = realism
    context["pose_flexion_deg"] = flexion_deg
    if realism == "anatomical" and bundle_data is not None:
        context["anatomical"] = anatomical_context(bundle_data, cov_fem_anat, stump_info)
    if anat is not None:
        context["anatomy_ref"] = {
            "centerline_length_mm": anat["loft"]["length_mm"],
            "volume_mm3": anat["volume_mm3"],
            "gap_mm": loft_tear["gap_mm"] if loft_tear else None,
            "injury_source": loft_tear["injury_source"] if loft_tear else "intact_reference",
            "detail_file": "anatomy.json"}
    write_json(os.path.join(out, "context.json"), context)

    gap_ok = gap_mm is not None and gap_mm > 0
    fix = ["reduce swing_mm, fold_depth_mm or strand diameter_mm"]
    checks = [
        {"check_id": "inclination", "status": "pass" if abs(sag_m - sag) < 0.1 and abs(cor_m - cor) < 0.1 else "fail",
         "threshold": 0.1, "actual": [sag_m, cor_m], "target": [sag, cor], "units": "deg", "method": "projection",
         "location": {}, "suggested_parameter_changes": []},
        # FIX: Muse's T1 dropped the flare check; kept.
        {"check_id": "flare", "status": "pass" if abs(flare_m - flare) / flare < 0.02 else "fail",
         "threshold": 0.02, "actual": flare_m, "target": flare, "units": "ratio",
         "method": "footprint area / midsubstance area", "location": {}, "suggested_parameter_changes": []},
        {"check_id": "native_containment", "status": "pass" if pen <= 0.0 else "fail",
         "threshold": 0.0, "actual": pen, "units": "mm", "method": "ellipse boundary distance",
         "location": penloc, "suggested_parameter_changes": ["fit_envelope"] + fix},
        {"check_id": "strand_gap", "status": "pass" if gap_ok else "fail",
         "threshold": 0.0, "actual": gap_mm, "units": "mm",
         "method": "all-pairs spline refine, excludes |s-t|<pi*d",
         "location": {"pair": gap.get("pair"), "s_mm": gap.get("s_mm")}, "suggested_parameter_changes": fix},
        {"check_id": "block_wall", "status": "pass" if gap_ok and gap_mm >= 0.8 else "fail",
         "threshold": 0.8, "actual": gap_mm, "units": "mm", "method": "same as strand_gap",
         "location": {"pair": gap.get("pair")}, "suggested_parameter_changes": fix},
        {"check_id": "tube_mold_feasible", "status": "pass" if gap_ok and gap_mm >= 1.7 else "warn",
         "threshold": 1.7, "actual": gap_mm, "units": "mm", "method": "two-wall rule",
         "location": {}, "suggested_parameter_changes": []},
        # FIX: Muse's T1 dropped the bone-clearance check; kept.
        {"check_id": "bone_clearance_schematic", "status": "warn" if bool(inside_tib) else "pass",
         "threshold": 0.0, "actual": float(d_tib), "units": "mm", "method": "distance to tibia box, first 0.5 mm skipped",
         "location": {}, "suggested_parameter_changes": []},
        {"check_id": "process_compat", "status": "unknown"},
        {"check_id": "mechanical_validation", "status": "not_evaluated"},
        {"check_id": "biological_validation", "status": "not_evaluated"}]
    # Milestone A: new checks (only present when their geometry exists)
    if realism == "anatomical" and bundle_data is not None:
        _mrn = float(bundle_data["max_bundle_rn"])
        checks.append({"check_id": "bundle_inside_acl",
                       "status": "pass" if _mrn <= 1.0 + 1e-9 else "fail",
                       "threshold": 1.0, "actual": _mrn, "units": "ratio",
                       "method": "max bundle surface radius in overall envelope",
                       "location": {}, "suggested_parameter_changes": []})
        if stump_info is not None:
            _open = int(stump_info["watertight"]["tibial_open_edges"]
                        + stump_info["watertight"]["femoral_open_edges"])
            checks.append({"check_id": "stump_closure",
                           "status": "pass" if _open == 0 else "fail",
                           "threshold": 0, "actual": _open, "units": "open edges",
                           "method": "every mesh edge shared by exactly 2 triangles",
                           "location": {}, "suggested_parameter_changes": []})
    # Loft anatomy checks (only when the reference object exists)
    if anat is not None:
        checks.append({"check_id": "anatomy_regen",
                       "status": "pass" if regen_ok else "fail",
                       "threshold": None, "actual": anat["regen_hash"][:16], "units": "hash16",
                       "method": "rebuild without tubes reproduces the surface hash",
                       "location": {}, "suggested_parameter_changes": []})
        _miss = sum(anat["endpoint_misses"].values())
        checks.append({"check_id": "attachment_on_bone",
                       "status": "pass" if _miss == 0 else "fail",
                       "threshold": 0, "actual": _miss, "units": "missed rays",
                       "method": "footprint endpoint rays hit the bone surface",
                       "location": {}, "suggested_parameter_changes": []})
        _co = float(anat["loft"]["continuity"])
        checks.append({"check_id": "frame_continuity",
                       "status": "pass" if _co >= 0.999 else "fail",
                       "threshold": 0.999, "actual": _co, "units": "min dot",
                       "method": "consecutive RMF frame alignment; engineering threshold",
                       "location": {}, "suggested_parameter_changes": []})
        _aud = all(v["pass"] for v in anat["audits"].values())
        checks.append({"check_id": "mesh_valid_anatomy",
                       "status": "pass" if _aud else "fail",
                       "threshold": None, "actual": {k: v["pass"] for k, v in anat["audits"].items()},
                       "units": "audits", "method": "edge counts + degenerate + orientation per solid/open class",
                       "location": {}, "suggested_parameter_changes": []})
        _bp = float(anat["bone_penetration"]["fraction"])
        checks.append({"check_id": "bone_penetration_midbody",
                       "status": "pass" if _bp <= 0.02 else "warn",
                       "threshold": 0.02, "actual": _bp, "units": "fraction",
                       "method": "mid-body surface samples inside bone; 2% engineering allowance (bulbous schematic condyle)",
                       "location": {}, "suggested_parameter_changes": []})
        if quality == "export":
            _si = anat["self_intersections"]["count"]
            checks.append({"check_id": "anatomy_self_intersection",
                           "status": "pass" if _si == 0 else "fail",
                           "threshold": 0, "actual": _si, "units": "pairs",
                           "method": "exact segment-triangle test over KD-tree candidates",
                           "location": {}, "suggested_parameter_changes": []})
        if anat is not None and not skip_stl:
            _soup = read_stl_vertices(os.path.join(out, "anatomy.stl"))
            _mm = np.concatenate([anat["surface"][0], anat["tibial_patch"][0]])
            _dd = float(max(np.abs(_soup.min(axis=0) - _mm.min(axis=0)).max(),
                            np.abs(_soup.max(axis=0) - _mm.max(axis=0)).max()))
            checks.append({"check_id": "export_scale",
                           "status": "pass" if _dd < 0.01 else "fail",
                           "threshold": 0.01, "actual": _dd, "units": "mm",
                           "method": "anatomy.stl bounds vs source bounds (mm, float32 round-trip)",
                           "location": {}, "suggested_parameter_changes": []})
    for zl, z in zip(zlams, zones):
        if z.get("control") == "fiber_length":
            tgt = float(z["path_length_ratio"])
            checks.append({"check_id": "fiber_length_%s" % z["id"],
                           "status": "pass" if abs(zl["path_length_ratio_mean"] - tgt) <= 0.002 else "fail",
                           "threshold": 0.002, "actual": zl["path_length_ratio_mean"], "target": tgt,
                           "units": "ratio", "method": "closed-loop rate solve, whole zone, mean of 3 strands",
                           "location": {}, "suggested_parameter_changes": []})
    validation = {"checks": checks, "uses_assumptions": len(assumptions) > 0,
                  "overall": {"geometry_valid": bool(gap_ok), "target_fit_satisfied": bool(pen <= 0.0)},
                  "wording": "Schematic knee, not patient anatomy. Outside native ACL envelope (schematic) if containment fails."}
    write_json(os.path.join(out, "validation.json"), validation)
    # Parameter record + validation report for the anatomy object
    if anat is not None:
        write_json(os.path.join(out, "anatomy.json"),
                   anatomy_payload(anat, norm, side, quality, loft_tear, regen_ok))
        write_json(os.path.join(out, "validation_report.json"),
                   validation_report(anat, loft_tear, quality))

    summary = ["phaseL=%.5f equiv=%.4f nevents=%d" % (phaseL, equiv, len(events)),
               "zone fiber length (path-length ratio, mean of 3 strands): " +
               ", ".join("%s %.4f" % (zl["id"], zl["path_length_ratio_mean"]) for zl in zlams),
               "sag %.3f/%.3f cor %.3f/%.3f" % (sag_m, sag, cor_m, cor),
               "flare %.4f/%.3f Amid %.3f" % (flare_m, flare, Amid),
               "penetration %.4f coverage tib %.3f fem %.3f" % (pen, cov_tib, cov_fem),
               "gap surf=%.4f Rmin=%.4f" % (gap_mm, (1.0 / kmax if kmax > 0 else float("inf"))),
               "assumptions used: %d" % len(assumptions),
               "wrote %s: %sknee.glb, parameters.json, context.json, validation.json, migration_report.json%s"
               % (out, "" if skip_stl else "scaffold.stl, ",
                  "" if anat is None else ", anatomy.stl, anatomy.json, validation_report.json")]
    # Milestone A: one-line anatomical summary
    if realism == "anatomical" and bundle_data is not None:
        _am = bundle_data["bundles"]["AM"]["length_mm"]
        _pl = bundle_data["bundles"]["PL"]["length_mm"]
        summary.append("anatomical: AM %.2fmm PL %.2fmm twist %.1fdeg (%s)" % (
            _am, _pl, twist_total, twist_dist))
        if stump_info is not None:
            summary.append("tear %s(%s) gap %.3fmm nominal %.3fmm stumps tib %.2fmm fem %.2fmm" % (
                stump_info["severity"], stump_info["involvement"], stump_info["gap_mm"],
                stump_info["gap_nominal_mm"], stump_info["tibial_stump_mm"],
                stump_info["femoral_stump_mm"]))
    if anat is not None:
        summary.append("loft ACL len %.2fmm vol %.0fmm3 strands %d/%d flexion %.1fdeg (%s)" % (
            anat["loft"]["length_mm"], anat["volume_mm3"],
            anat["strand_paths"]["AM"].shape[0], anat["strand_paths"]["PL"].shape[0],
            flexion_deg, loft_tear["injury_source"] if loft_tear else "intact_reference"))
    tm["glb_json"] = time.time() - s1
    return {"phaseL": phaseL, "design_hash": resolved["design_hash"], "timings_s": tm, "nevents": len(events), "arcs": arcs, "gap": gap_mm, "pen": pen,
            "kmax": kmax, "elapsed_s": time.time() - t0, "uses_assumptions": len(assumptions) > 0,
            "checks": checks, "summary": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quality", default="export", choices=["preview", "export"])
    args = ap.parse_args()
    with open(args.request) as f:
        req = json.load(f)
    norm, mig = validate_request(req)
    res = run_request(norm, mig, args.quality, args.out)
    print("\n".join(res["summary"]))
    print("time %.1fs" % res["elapsed_s"])


if __name__ == "__main__":
    main()
