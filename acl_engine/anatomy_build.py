"""Assemble the reference anatomical ACL object (intact loft + footprints +
strands), the injury object (loft-based residual stumps), metrics, and the
anatomy.json / validation_report.json payloads.

New module. Called from cli.py only when realism == 'anatomical'. The
legacy milestone-A bundle ghosts keep their GLB names; this builder owns
acl_surface, acl_strands_*, footprint patches, boundary loops, and loft
stumps. The intact reference surface is never modified by tear or scaffold
parameters: injury meshes are separate objects derived from a copy.
"""
import math
import numpy as np
from scipy.spatial import cKDTree
from acl_engine.loft import (build_loft, close_solid, strand_paths,
                             strand_lengths, section_area, apply_flexion,
                             flexion_matrix, smooth_profile)
from acl_engine.attachments import (Footprint, tibial_attachment_points,
                                    femoral_attachment_points,
                                    femoral_patch_mesh, spiral_disk,
                                    split_weights, ellipsoid_ray_x)
from acl_engine.metrics import (audit_mesh, self_intersections,
                                bone_penetration, regen_hash, profile_stats,
                                TOL)
from acl_engine.tear import split_tube, is_watertight, tear_defaults, apply_tear_preset
from acl_engine.sweep import sweep_strand, signed_volume
from acl_engine.frames import compute_rmf

FLEXION_AXIS_YZ = (-5.0, 25.0)  # through the schematic condyle centers
TUBE_NSEC = {"preview": 5, "export": 6}


def _assumed(lst, field, value, origin, source=None):
    lst.append({"field": field, "value": value, "origin": origin,
                "source": source})


def default_section_keys(Ws_env, Ts_env, fp_tib, fp_fem, acl_len):
    """Body section keys from the validated envelope + flared root ends
    matched approximately to the footprint domains (engineering choice:
    root sections span the footprint minor extent, not its full planar
    area, since the loft axis meets the footprint obliquely).

    Flare ramps are placed from the footprint transition lengths so the
    widening happens INSIDE the insertion zones. Femoral root rotation
    ramps so the fan's long axis lies ALONG the notch wall (fan-like
    extension, L19 Fig.3/4) instead of into the bone.
    """
    Ws = np.asarray(Ws_env, dtype=float)
    Ts = np.asarray(Ts_env, dtype=float)
    n = Ws.shape[0]
    at = lambda f: (float(Ws[int(np.clip(f, 0.0, 1.0) * (n - 1))]),
                    float(Ts[int(np.clip(f, 0.0, 1.0) * (n - 1))]))
    tt = float(fp_tib.transition_mm) / float(acl_len)
    tf = float(fp_fem.transition_mm) / float(acl_len)
    keys = []
    keys.append({"t": 0.0, "width": 2.0 * fp_tib.axes[0],
                 "thickness": 0.55 * 2.0 * fp_tib.axes[1],
                 "exponent": 0.9, "rotation_deg": 0.0})
    for f in (tt, 0.30, 0.50, 0.70, 1.0 - tf):
        w, h = at(f)
        keys.append({"t": f, "width": w, "thickness": h, "exponent": 1.0,
                     "rotation_deg": 0.0 if f < 0.5 else 6.0 + 24.0 * (f - 0.5)})
    keys.append({"t": 1.0 - tf / 2.0, "width": at(1.0 - tf / 2.0)[0],
                 "thickness": at(1.0 - tf / 2.0)[1], "exponent": 0.9,
                 "rotation_deg": 55.0})
    keys.append({"t": 1.0, "width": 2.0 * fp_fem.axes[0],
                 "thickness": 2.0 * fp_fem.axes[1], "exponent": 0.85,
                 "rotation_deg": 70.0})
    keys.sort(key=lambda k: k["t"])
    return keys


def boundary_loop_ribbon(loop3, width_mm=0.35):
    """Flat ribbon mesh along a closed 3-D loop (for footprint boundaries).
    Ribbon lies in the loop's best-fit plane. Vectorized."""
    P = np.asarray(loop3, dtype=float)
    m = P.shape[0]
    C = P.mean(axis=0)
    _u, _s, Vt = np.linalg.svd(P - C, full_matrices=False)
    n = Vt[2] / np.linalg.norm(Vt[2])
    radial = P - C
    radial /= np.linalg.norm(radial, axis=1, keepdims=True) + 1e-30
    side = np.cross(n[None, :], radial)
    w = float(width_mm) / 2.0
    V = np.concatenate([P - w * side, P + w * side])
    j = np.arange(m)
    j2 = (j + 1) % m
    F = np.concatenate([np.stack([j, j2, m + j], axis=1),
                        np.stack([j2, m + j2, m + j], axis=1)]).astype(np.int64)
    return V, F


def tibial_patch_solid(fp, thick_mm=0.3, nrad=12, nang=64):
    """Closed thin solid of the footprint domain on the plateau (z=0..h).

    Polar-grid slab with center fans and outer wall (same proven topology
    as footprints.tibial_footprint_mesh, generalized to any domain).
    """
    r = np.linspace(0.0, 1.0, nrad + 1)[1:]
    ph = np.linspace(0.0, 2.0 * math.pi, nang, endpoint=False)
    rr, pp = np.meshgrid(r, ph, indexing="ij")
    pol = np.stack([rr.ravel(), pp.ravel()], axis=1)
    xy = fp.domain_points(pol).reshape(nrad, nang, 2)
    top = np.concatenate([xy, np.full((nrad, nang, 1), thick_mm)], axis=-1)
    bot = np.concatenate([xy, np.zeros((nrad, nang, 1))], axis=-1)
    cx, cy = float(fp.center2[0]), float(fp.center2[1])
    V = np.concatenate([np.array([[cx, cy, thick_mm], [cx, cy, 0.0]]),
                        top.reshape(-1, 3), bot.reshape(-1, 3)])
    n = nrad * nang
    ii = np.arange(nrad - 1)[:, None]
    jj = np.arange(nang)[None, :]
    j2 = (jj + 1) % nang
    a = ii * nang + jj
    b = ii * nang + j2
    c = (ii + 1) * nang + jj
    d = (ii + 1) * nang + j2
    grid = np.stack([np.stack([a, b, c], axis=-1),
                     np.stack([b, d, c], axis=-1)], axis=2).reshape(-1, 3)
    Ft = grid + 2
    Fb = grid[:, [0, 2, 1]] + 2 + n
    j = np.arange(nang)
    jn = (j + 1) % nang
    cap_t = np.stack([np.full(nang, 0), 2 + jn, 2 + j], axis=1)
    cap_b = np.stack([np.full(nang, 1), 2 + n + j, 2 + n + jn], axis=1)
    ot = 2 + (nrad - 1) * nang + j
    ot2 = 2 + (nrad - 1) * nang + jn
    ob = 2 + n + (nrad - 1) * nang + j
    ob2 = 2 + n + (nrad - 1) * nang + jn
    wall = np.concatenate([np.stack([ot, ot2, ob], axis=1),
                           np.stack([ot2, ob2, ob], axis=1)])
    F = np.concatenate([Ft, Fb, cap_t, cap_b, wall]).astype(np.int64)
    if signed_volume(V, F) < 0:
        F = F[:, [0, 2, 1]]
    return V, F


def _strand_tubes(paths, radius_mm, nsec):
    """Thin swept tubes per path (RMF frames). Returns list of (V,F)."""
    out = []
    for p in paths:
        grad = np.gradient(p, axis=0, edge_order=2)
        N1, N2, _T = compute_rmf(p, grad)
        out.append(sweep_strand(p, N1, N2, float(radius_mm), int(nsec)))
    return out


def _concat(parts):
    Vs, Fs, off = [], [], 0
    for V, F in parts:
        Vs.append(np.asarray(V, dtype=float))
        Fs.append(np.asarray(F, dtype=np.int64) + off)
        off += V.shape[0]
    return np.concatenate(Vs), np.concatenate(Fs).astype(np.int64)


def anatomy_payload(A, norm, side, quality, tear_result, regen_ok):
    """JSON-serializable parameter record: units, conventions, laterality,
    pose, controls, profiles, assumptions, provenance. Downsampled section
    polygons included for the slice viewer (not for metrology)."""
    loft = A["loft"]
    soon = [0, 15, 30, 45, 60, 75, 90, 105, 119]
    soon = [min(i, loft["center"].shape[0] - 1) for i in soon]
    sections = [{"t": float(loft["t"][i]),
                 "center_mm": [float(v) for v in loft["center"][i]],
                 "width_mm": float(loft["width"][i]),
                 "thickness_mm": float(loft["thick"][i]),
                 "area_mm2": float(A["section_area"][i]),
                 "rotation_deg": float(np.degrees(loft["rot_rad"][i])),
                 "ring_mm": [[float(v) for v in p] for p in loft["rings"][i]]}
                for i in soon]
    strands = {k: {"count": int(v.shape[0]),
                   "length_min_mm": float(A["strand_lengths"][k].min()),
                   "length_max_mm": float(A["strand_lengths"][k].max()),
                   "length_mean_mm": float(A["strand_lengths"][k].mean())}
               for k, v in A["strand_paths"].items()}
    out = {
        "model": "parametric_acl_loft",
        "label": "illustrative anatomical model -- literature template, not patient-specific",
        "units": {"length": "mm", "angle": "deg"},
        "coordinate_frame": "+X lateral, +Y anterior, +Z proximal, origin tibial plateau center",
        "side": side,
        "pose": A["flexion"],
        "quality": quality,
        "centerline": {"length_mm": loft["length_mm"],
                       "attachment_distance_mm": A["attach_distance_mm"],
                       "frame": "rotation-minimizing (parallel transport), E1/E2 reference at t=0",
                       "continuity_min_dot": loft["continuity"]},
        "rotation_channels": {
            "reference": "degrees from section frame E1 at t=0, positive toward E2",
            "section_orientation_total_deg": float(np.degrees(loft["rot_rad"][-1] - loft["rot_rad"][0])),
            "bundle_boundary_total_deg": float(A["boundary_total_deg"]),
            "strand_swirl": "per-strand smooth profile, see strand paths"},
        "footprints": {
            "tibial": {"area_mm2": A["footprint_areas"]["tibial"],
                       "subregions_mm2": A["footprint_areas"]["tibial_sub"],
                       "surface": "plateau reference plane z=0 (schematic)"},
            "femoral": {"area_mm2": A["footprint_areas"]["femoral"],
                        "subregions_mm2": A["footprint_areas"]["femoral_sub"],
                        "surface": "lateral condyle ellipsoid (schematic)"},
            "endpoint_misses": A["endpoint_misses"]},
        "strands": strands,
        "strand_max_section_radius": A["strand_max_rn"],
        "endpoints_in_domain": A["endpoints_in_domain"],
        "volume_mm3": A["volume_mm3"],
        "mesh_audits": A["audits"],
        "self_intersections": A["self_intersections"],
        "bone_penetration_midbody": A["bone_penetration"],
        "regen_hash": A["regen_hash"],
        "regen_check": "pass" if regen_ok else "fail",
        "assumptions": A["assumptions"],
        "tolerances": {"audit": TOL, "tolerance_note": "engineering settings with stated rationale, NOT clinical thresholds"},
        "provenance": {"sources": ["L01", "L02", "L04", "L11", "L12", "L13", "L14", "L15",
                                   "L19", "L20", "L21"],
                       "surface_origin": "parametric loft from validated envelope + footprint domains",
                       "mri_masks": None,
                       "tear_label": tear_result["injury_source"] if tear_result else "intact_reference"},
        "sections": sections,
    }
    if tear_result:
        out["tear"] = {k: tear_result[k] for k in
                       ("severity", "involvement", "location_s", "location_mm",
                        "obliquity_deg", "azimuth_deg", "morphology", "gap_mm",
                        "gap_nominal_mm", "tibial_stump_mm", "femoral_stump_mm",
                        "cut_bundles", "intact_bundles", "injury_source",
                        "watertight") if k in tear_result}
    return out


def validation_report(A, tear_result, quality):
    """Separate measured results, failed checks, and checks not performed."""
    measured = {
        "centerline_length_mm": A["loft"]["length_mm"],
        "attachment_distance_mm": A["attach_distance_mm"],
        "volume_mm3": A["volume_mm3"],
        "footprint_area_tibial_mm2": A["footprint_areas"]["tibial"],
        "footprint_area_femoral_mm2": A["footprint_areas"]["femoral"],
        "section_area_min_mm2": float(A["section_area"].min()),
        "section_area_max_mm2": float(A["section_area"].max()),
        "frame_continuity_min_dot": A["loft"]["continuity"],
        "strand_max_section_radius": A["strand_max_rn"],
        "midbody_bone_penetration_fraction": A["bone_penetration"]["fraction"],
        "regen_hash": A["regen_hash"][:16],
    }
    failed = []
    for name, au in A["audits"].items():
        if not au["pass"]:
            failed.append({"check": "mesh_valid:" + name, "detail": au})
    if A["strand_max_rn"] > 1.0:
        failed.append({"check": "strand_containment", "actual": A["strand_max_rn"]})
    if not A["endpoints_in_domain"]:
        failed.append({"check": "endpoints_in_domain"})
    if A["bone_penetration"]["fraction"] > 0.02:
        failed.append({"check": "bone_penetration_midbody",
                       "actual": A["bone_penetration"]["fraction"],
                       "note": "engineering allowance 2%; schematic condyle is bulbous near the notch"})
    if tear_result and not (tear_result["watertight"]["tibial"]
                            and tear_result["watertight"]["femoral"]):
        failed.append({"check": "loft_stump_closure", "detail": tear_result["watertight"]})
    not_performed = []
    if quality != "export":
        not_performed.append({"check": "self_intersection_exact",
                              "reason": "export quality only (cost); preview runs edge-count audits"})
    not_performed.append({"check": "overlap_vs_ground_truth",
                          "reason": "no ground-truth segmentation supplied with this project"})
    not_performed.append({"check": "validated_kinematics",
                          "reason": "flexion pose is an illustrative rigid transform, not validated data"})
    return {"measured": measured, "failed_checks": failed,
            "not_performed": not_performed}


def build_loft_tear(A, tear, observed=False):
    """Injury object: residual stumps cut from a COPY of the intact loft.

    The reference surface in A is never modified. Cut plane from the
    existing tear fields (location_s fraction along the centerline,
    obliquity/azimuth in the local loft frame); stumps retracted along the
    local tangent and closed (watertight). Partial tears additionally split
    the involved bundle's strand paths at the cut while the intact bundle's
    strands keep spanning (labeled). Returns dict with meshes + metrics.
    """
    t = tear_defaults(apply_tear_preset(dict(tear)))
    loft = A["loft"]
    n = loft["center"].shape[0]
    tcut = float(t["location_s"])
    icut = int(np.clip(tcut * (n - 1), 1, n - 2))
    Q = loft["center"][icut]
    Tang = loft["Tang"][icut]
    Tang = Tang / np.linalg.norm(Tang)
    E1 = loft["N1"][icut]
    E2 = loft["N2"][icut]
    az = math.radians(float(t["azimuth_deg"]))
    obl = math.radians(float(t["obliquity_deg"]))
    d = math.cos(az) * E1 + math.sin(az) * E2
    nv = math.cos(obl) * Tang + math.sin(obl) * d
    nv = nv / np.linalg.norm(nv)
    from acl_engine.tear import RAGGEDNESS_BY_MORPHOLOGY, RAGGEDNESS_DEFAULT
    amp = RAGGEDNESS_BY_MORPHOLOGY.get(t.get("morphology", "transverse"),
                                       RAGGEDNESS_DEFAULT)
    Vt, Ft, _it = split_tube(loft["rings"], Q, nv, -1, rag_amp=amp)
    Vf, Ff, _if = split_tube(loft["rings"], Q, nv, +1, rag_amp=amp)
    shift_f = float(t["retraction_femoral_mm"]) + float(t["tissue_loss_mm"]) / 2.0
    shift_t = float(t["retraction_tibial_mm"]) + float(t["tissue_loss_mm"]) / 2.0
    Vt = Vt - shift_t * Tang[None, :]
    Vf = Vf + shift_f * Tang[None, :]
    tree = cKDTree(Vf)
    gap = float(tree.query(Vt, k=1)[0].min())
    nominal = (shift_f + shift_t) * math.cos(math.radians(float(t["obliquity_deg"])))
    okt, infot = is_watertight(Vt, Ft)
    okf, infof = is_watertight(Vf, Ff)
    inv = t["involvement"]
    cut_names = ["AM", "PL"] if inv == "whole" else [inv]
    # Strand handling: involved bundle paths split at the cut and shifted
    # with their stump; intact bundle paths keep spanning the gap.
    t_grid = loft["t"]
    below = t_grid <= tcut
    above = t_grid >= tcut
    strand_segs = {}
    for key, P in A["strand_paths"].items():
        if key in cut_names:
            strand_segs[key] = {
                "tibial": P[:, below, :] - shift_t * Tang[None, None, :],
                "femoral": P[:, above, :] + shift_f * Tang[None, None, :],
                "spanning": False}
        else:
            strand_segs[key] = {"spanning": True, "paths": P}
    L = loft["length_mm"]
    return {
        "stump_tibial": (Vt, Ft), "stump_femoral": (Vf, Ff),
        "severity": t["severity"], "involvement": inv,
        "location_s": tcut, "location_mm": float(tcut * L),
        "obliquity_deg": float(t["obliquity_deg"]),
        "azimuth_deg": float(t["azimuth_deg"]),
        "morphology": t.get("morphology", "transverse"),
        "gap_mm": gap, "gap_nominal_mm": float(nominal),
        "tibial_stump_mm": float(tcut * L), "femoral_stump_mm": float((1.0 - tcut) * L),
        "cut_bundles": cut_names,
        "intact_bundles": [b for b in ("AM", "PL") if b not in cut_names],
        "retraction_femoral_mm": float(t["retraction_femoral_mm"]),
        "retraction_tibial_mm": float(t["retraction_tibial_mm"]),
        "watertight": {"tibial": bool(okt), "femoral": bool(okf),
                       "tibial_open_edges": infot["open_edges"],
                       "femoral_open_edges": infof["open_edges"]},
        "strand_segments": strand_segs,
        "injury_source": "observed_residual" if observed else "synthetic_example",
    }


def build_anatomy(norm, Ttib, Ffem, u, e1, e3, ss_env, Ws_env, Ts_env, side,
                  quality="export", with_tubes=True):
    """Build the anatomy object. Returns dict A with meshes, loft,
    footprints, strands, tear result, metrics, payloads, and assumptions."""
    A = {"assumptions": []}
    asm = A["assumptions"]
    am = norm.get("acl_model", {"status": "derived"})
    flexion = float(norm.get("knee_scene", {}).get("flexion_deg", 0.0))
    axis_pt = np.array([0.0, FLEXION_AXIS_YZ[0], FLEXION_AXIS_YZ[1]])
    R = flexion_matrix(flexion)
    F_pos = apply_flexion(np.asarray(Ffem, float)[None, :], flexion, axis_pt)[0]
    A["flexion"] = {"deg": flexion, "axis_point_mm": [float(v) for v in axis_pt],
                    "matrix": R.tolist(), "illustrative": True,
                    "note": "tibia-fixed rigid pose of femoral side; not validated kinematics"}
    _lx0 = float(np.asarray(Ffem, float)[0]) + 11.5 if side == "right" \
        else float(np.asarray(Ffem, float)[0]) - 11.5
    # Extension-frame condyle (unrotated: ray-casting is exact here); the
    # cli draws this same ellipsoid posed, and all femoral-side products
    # below are posed with the identical rigid transform.
    condyle_ext = np.array([_lx0, -5.0, 25.0])
    condyle_radii = np.array([12.0, 18.0, 16.0])
    condyle_lat = apply_flexion(condyle_ext[None, :], flexion, axis_pt)[0]
    A["condyle_lat"] = [float(v) for v in condyle_lat]
    # --- footprints (user override or derived defaults) ---
    tfp = (am.get("tibial_footprint") or {})
    ffp = (am.get("femoral_footprint") or {})
    fp_tib = Footprint("tibial",
                       tfp.get("center", [float(Ttib[0]), float(Ttib[1])]),
                       tfp.get("axes", [5.5, 8.5]),
                       tfp.get("rotation_deg", 0.0), tfp.get("exponent", 1.0),
                       tfp.get("split_deg", 0.0), tfp.get("transition_mm", 6.0))
    F_unposed = np.asarray(Ffem, float)
    fp_fem = Footprint("femoral",
                       ffp.get("center", [float(F_unposed[1]), float(F_unposed[2])]),
                       ffp.get("axes", [6.5, 8.0]),
                       ffp.get("rotation_deg", 0.0), ffp.get("exponent", 1.0),
                       ffp.get("split_deg", 90.0), ffp.get("transition_mm", 7.0))
    if not tfp:
        _assumed(asm, "tibial_footprint", {"center": "tibial ACL center", "axes": [5.5, 8.5]},
                 "literature_template", "L15")
    if not ffp:
        _assumed(asm, "femoral_footprint", {"center": "femoral ACL center on wall", "axes": [6.5, 8.0]},
                 "literature_template", "L14")
    A["fp_tib"], A["fp_fem"] = fp_tib, fp_fem
    # --- centerline controls (endpoints = attachment centers on bone) ---
    tib_c3 = np.array([fp_tib.center2[0], fp_tib.center2[1], 0.0])
    fem_yz = np.array([fp_fem.center2])
    fem_x, fem_hit = ellipsoid_ray_x(fem_yz, condyle_ext, condyle_radii, side)
    if bool(fem_hit[0]):
        fem_c3e = np.array([float(fem_x[0]), fp_fem.center2[0], fp_fem.center2[1]])
    else:
        from acl_engine.footprints import femoral_wall_x
        fem_c3e = np.array([femoral_wall_x(F_unposed, side), fp_fem.center2[0],
                            fp_fem.center2[1]])
        _assumed(asm, "femoral_center_fallback", "wall plane (domain center off condyle silhouette)",
                 "engineering_fallback", None)
    fem_c3 = apply_flexion(fem_c3e[None, :], flexion, axis_pt)[0]
    if am.get("centerline_controls") is not None:
        controls = np.array(am["centerline_controls"], dtype=float)
    else:
        T = np.asarray(Ttib, float)
        F = np.asarray(F_pos, float)
        e1n = np.asarray(e1, float) / np.linalg.norm(e1)
        bow = np.array([0.0, 1.5, 0.0])  # gentle anterior bow, L19-inspired assumption
        medial = -1.5 * e1n  # femoral-third clearance from the notch wall
        controls = np.array([tib_c3, 0.75 * T + 0.25 * F + 0.5 * bow,
                             0.5 * T + 0.5 * F + bow,
                             0.25 * T + 0.75 * F + 0.5 * bow + medial,
                             fem_c3])
        _assumed(asm, "centerline_controls",
                 "attachment centers + 1.5 mm mid anterior bow + 1.5 mm femoral-third medial offset",
                 "literature_template", "L19")
    # --- sections + boundary channels ---
    acl_len0 = float(norm["_derived"]["acl_length_mm"])
    if am.get("section_keys") is None:
        # Body sections follow the UNFLARED profile: insertion flare now
        # lives in the root sections + transition zones, not the body.
        from acl_engine.context import build_envelope_fields as _bef
        _ss, _A, _W, _T, _Am = _bef(
            acl_len0, float(norm["_derived"]["envelope_width_mm"]),
            float(norm["_derived"]["envelope_thickness_mm"]),
            norm["_derived"]["area_profile_femur_to_tibia"], 1.0)
        skeys = default_section_keys(_W, _T, fp_tib, fp_fem, acl_len0)
    else:
        skeys = am["section_keys"]
    if am.get("section_keys") is None:
        _assumed(asm, "section_keys", "envelope-derived body + flared roots + 0-70 deg femoral fan ramp",
                 "literature_template", "L01/L02/L19")
    twist_total = float(norm["_derived"].get("anatomical_twist_total_deg", 0.0))
    twist_dist = norm["_derived"].get("twist_distribution", "uniform")
    coupling = float(am.get("flexion_coupling_deg", 15.0))
    if "flexion_coupling_deg" not in am:
        _assumed(asm, "flexion_coupling_deg", 15.0, "illustrative_pose", None)
    from acl_engine.anatomy import twist_angle
    acl_len = float(norm["_derived"]["acl_length_mm"])
    n_station = int(am.get("n_station", 160 if quality == "export" else 96))
    m_ring = int(am.get("m_ring", 48 if quality == "export" else 32))
    t_grid = np.linspace(0.0, 1.0, n_station)
    w_prof = twist_angle(t_grid * acl_len, acl_len, 1.0, twist_dist)
    w_prof = w_prof / (w_prof[-1] if abs(w_prof[-1]) > 1e-12 else 1.0)
    boundary_total = twist_total + coupling * flexion / 90.0
    if am.get("boundary_keys") is not None:
        bkeys = am["boundary_keys"]
    else:
        bkeys = [{"t": float(t), "angle_deg": float(90.0 + boundary_total * w)}
                 for t, w in zip(t_grid[::max(1, n_station // 8)], w_prof[::max(1, n_station // 8)])]
        if abs(boundary_total) > 1e-9 or flexion != 0.0:
            _assumed(asm, "boundary_keys", "90 deg + %.1f deg profile (%s)" % (boundary_total, twist_dist),
                     "user_or_pose" if "boundary_keys" in am else "literature_template", "L13")
    sharpness = float(am.get("bundle_sharpness", 0.35))
    loft = build_loft(controls, skeys, n_station=n_station, m_ring=m_ring,
                      boundary_keys=bkeys, sharpness=sharpness)
    if am.get("section_keys") is None:
        # Fan-rotation sign is frame-relative, so it mirrors with side: try
        # both signs and keep the one whose rings touch the condyle less
        # (deterministic engineering choice, recorded below).
        flip = [dict(k, rotation_deg=-k["rotation_deg"]) for k in skeys]
        loft_f = build_loft(controls, flip, n_station=n_station, m_ring=m_ring,
                            boundary_keys=bkeys, sharpness=sharpness)
        def _hits(lf):
            q = ((lf["rings"] - condyle_lat) / condyle_radii) ** 2
            return int(np.sum(q.sum(-1) <= 1.0))
        if _hits(loft_f) < _hits(loft):
            loft, skeys = loft_f, flip
            _assumed(asm, "fan_rotation_sign", "negative (minimizes condyle contact)",
                     "engineering_choice", None)
        else:
            _assumed(asm, "fan_rotation_sign", "positive (minimizes condyle contact)",
                     "engineering_choice", None)
    A["loft"] = loft
    A["section_keys"] = skeys
    A["boundary_total_deg"] = float(boundary_total)
    # --- strands ---
    n_str = int(am.get("strand_count", 24 if quality == "export" else 16))
    margin = float(am.get("strand_margin", 0.85))
    radius = float(am.get("strand_radius_mm", 0.35))
    swirl_total = float(am.get("swirl_total_deg", 0.0))
    n_am = int(round(n_str * 0.58))
    n_pl = n_str - n_am
    if "strand_count" not in am:
        _assumed(asm, "strand_bundle_split", {"AM": n_am, "PL": n_pl},
                 "literature_template", "L12")
    tib_pol = spiral_disk(n_am + n_pl, offset=0.0)
    fem_pol = spiral_disk(n_am + n_pl, offset=math.radians(boundary_total))
    wam_t, _w = split_weights(tib_pol[:, 1], fp_tib.split_deg)
    order = np.argsort(-wam_t, kind="stable")
    am_idx = order[:n_am]
    pl_idx = order[n_am:n_am + n_pl]
    tib_ep = {"AM": tib_pol[am_idx], "PL": tib_pol[pl_idx]}
    wam_f, _w = split_weights(fem_pol[:, 1], fp_fem.split_deg)
    forder = np.argsort(-wam_f, kind="stable")
    fem_ep = {"AM": fem_pol[forder[:n_am]], "PL": fem_pol[forder[n_am:n_am + n_pl]]}
    # endpoint 3-D positions on bone (distributed in footprint domains)
    tib_dom = {"AM": fp_tib.domain_points(tib_ep["AM"]),
               "PL": fp_tib.domain_points(tib_ep["PL"])}
    fem_dom = {"AM": fp_fem.domain_points(fem_ep["AM"]), "PL": fp_fem.domain_points(fem_ep["PL"])}
    tib_3d = tibial_attachment_points(fp_tib, tib_dom)
    fem_3d, misses = femoral_attachment_points(fp_fem, fem_dom, condyle_ext, condyle_radii, side)
    if abs(flexion) > 1e-12:
        fem_3d = {k: apply_flexion(v, flexion, axis_pt) for k, v in fem_3d.items()}
    A["endpoint_misses"] = misses
    A["tib_3d"] = tib_3d
    A["fem_3d"] = fem_3d
    swirl = np.radians(swirl_total) * t_grid
    paths = {}
    for key in ("AM", "PL"):
        r_t = np.clip(tib_ep[key][:, 0], 0.0, margin)
        r_f = np.clip(fem_ep[key][:, 0], 0.0, margin)
        paths[key] = strand_paths(loft["center"], loft["N1"], loft["N2"],
                                  loft["width"], loft["thick"], loft["expo"],
                                  loft["rot_rad"], t_grid,
                                  np.stack([r_t, tib_ep[key][:, 1]], axis=1),
                                  np.stack([r_f, fem_ep[key][:, 1]], axis=1),
                                  swirl, margin=margin)
    # Blend strand ends onto their exact 3-D footprint endpoints over the
    # transition lengths, so strands terminate distributed ON the bone.
    L_len = loft["length_mm"]
    tt = float(fp_tib.transition_mm) / L_len
    tf = float(fp_fem.transition_mm) / L_len
    for key in ("AM", "PL"):
        P = paths[key]
        n = P.shape[1]
        wt = np.clip(1.0 - t_grid / max(tt, 1e-9), 0.0, 1.0)
        wt = wt * wt * (3.0 - 2.0 * wt)
        wf = np.clip((t_grid - (1.0 - tf)) / max(tf, 1e-9), 0.0, 1.0)
        wf = wf * wf * (3.0 - 2.0 * wf)
        P = P * (1.0 - wt)[None, :, None] + tib_3d[key][:, None, :] * wt[None, :, None]
        P = P * (1.0 - wf)[None, :, None] + fem_3d[key][:, None, :] * wf[None, :, None]
        paths[key] = P
    A["strand_paths"] = paths
    A["strand_lengths"] = {k: strand_lengths(v) for k, v in paths.items()}
    # Section-space containment over the BODY stations only: inside the
    # transition zones strands flare onto the footprint domains (root-like
    # spread, L19 Fig.4) and are checked against the domains instead.
    body = (t_grid >= tt) & (t_grid <= 1.0 - tf)
    worst = 0.0
    for key, P in paths.items():
        d = P - loft["center"][None, :, :]
        xl = np.sum(d * loft["N1"][None, :, :], axis=2)
        zl = np.sum(d * loft["N2"][None, :, :], axis=2)
        cr = np.cos(-loft["rot_rad"])[None, :]
        sr = np.sin(-loft["rot_rad"])[None, :]
        xu = xl * cr - zl * sr
        zu = xl * sr + zl * cr
        a = (loft["width"] / 2.0)[None, :]
        b = (loft["thick"] / 2.0)[None, :]
        e = loft["expo"][None, :]
        q = (np.abs(xu) / a) ** (2.0 / e) + (np.abs(zu) / b) ** (2.0 / e)
        rn = q ** (e / 2.0)
        worst = max(worst, float(rn[:, body].max()))
    A["strand_max_rn"] = worst
    A["strand_body_fraction"] = float(body.mean())
    # Endpoint-in-domain proof (exact by construction; verified numerically)
    ep_ok = True
    for key in ("AM", "PL"):
        for dom, fp in ((tib_dom[key], fp_tib), (fem_dom[key], fp_fem)):
            rel = dom - fp.center2[None, :]
            ang = math.radians(-fp.rotation_deg)
            xr = rel[:, 0] * math.cos(ang) - rel[:, 1] * math.sin(ang)
            yr = rel[:, 0] * math.sin(ang) + rel[:, 1] * math.cos(ang)
            q = (np.abs(xr) / fp.axes[0]) ** (2.0 / fp.exponent) \
                + (np.abs(yr) / fp.axes[1]) ** (2.0 / fp.exponent)
            ep_ok = ep_ok and bool(np.all(q <= 1.0 + 1e-9))
    A["endpoints_in_domain"] = bool(ep_ok)
    # --- meshes ---
    Vsurf, Fsurf = close_solid(loft["rings"])
    A["surface"] = (Vsurf, Fsurf)
    A["tibial_patch"] = tibial_patch_solid(fp_tib)
    _pf = femoral_patch_mesh(fp_fem, condyle_ext, condyle_radii, side)
    if abs(flexion) > 1e-12:
        _pf = (apply_flexion(_pf[0], flexion, axis_pt), _pf[1], _pf[2])
    A["femoral_patch"] = _pf
    loop_t = fp_tib.boundary_loop()
    loop_f = fp_fem.boundary_loop()
    loop_t3 = np.stack([loop_t[:, 0], loop_t[:, 1], np.full(loop_t.shape[0], 0.35)], axis=1)
    fx, fhit = ellipsoid_ray_x(loop_f, condyle_ext, condyle_radii, side)
    sgn = -1.0 if side == "right" else 1.0
    loop_f3 = np.stack([np.where(fhit, fx, np.nan), loop_f[:, 0], loop_f[:, 1]], axis=1)
    if abs(flexion) > 1e-12:
        _ok0 = np.all(np.isfinite(loop_f3), axis=1)
        loop_f3[_ok0] = apply_flexion(loop_f3[_ok0], flexion, axis_pt)
    ok = np.all(np.isfinite(loop_f3), axis=1)
    A["edge_tibial"] = boundary_loop_ribbon(loop_t3)
    A["edge_femoral"] = boundary_loop_ribbon(loop_f3[ok] + np.array([sgn * 0.1, 0.0, 0.0])) \
        if int(ok.sum()) >= 8 else (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    A["strand_tubes"] = {}
    if with_tubes:
        nsec = TUBE_NSEC.get(quality, 6)
        for key, P in paths.items():
            A["strand_tubes"][key] = _concat(_strand_tubes([p for p in P], radius, nsec))
    # --- metrics ---
    from acl_engine.loft import section_area as _area
    areas = _area(loft["width"], loft["thick"], loft["expo"])
    A["section_area"] = areas
    A["volume_mm3"] = float(signed_volume(Vsurf, Fsurf))
    A["attach_distance_mm"] = float(np.linalg.norm(controls[-1] - controls[0]))
    A["audits"] = {"acl_surface": audit_mesh(Vsurf, Fsurf, True),
                   "tibial_patch": audit_mesh(*A["tibial_patch"], True),
                   "femoral_patch": audit_mesh(A["femoral_patch"][0], A["femoral_patch"][1], False)}
    if quality == "export":
        n_self, n_tested = self_intersections(Vsurf, Fsurf)
        A["self_intersections"] = {"count": int(n_self), "pairs_tested": int(n_tested)}
    else:
        A["self_intersections"] = {"count": None, "pairs_tested": 0,
                                   "reason": "not_performed_in_preview"}
    tib_box = (np.array([-37.5, -25.0, -10.0]), np.array([37.5, 25.0, 0.0]))
    # Penetration counts only OUTSIDE the attachment transition zones; the
    # transitions intentionally enter bone (insertion zones, L19 Fig.8).
    # The condyle test uses the posed (rotated) ellipsoid frame, exact at
    # any flexion: q = |M (p - C1) / radii|^2 with M = R(flexion).
    midbody = (t_grid >= tt + 0.03) & (t_grid <= 1.0 - tf - 0.03)
    body_pts = loft["rings"][midbody].reshape(-1, 3)
    Mm = flexion_matrix(flexion)
    ploc = (body_pts - condyle_lat) @ Mm / condyle_radii
    in_fem = (ploc ** 2).sum(axis=1) <= 1.0 + TOL["bone_slack_mm"] / max(condyle_radii)
    tib_only = bone_penetration(body_pts, tib_box, [])
    A["bone_penetration"] = {"n_points": int(body_pts.shape[0]),
                             "n_inside": int(tib_only["n_inside"] + np.sum(in_fem)),
                             "fraction": float((tib_only["n_inside"] + np.sum(in_fem))
                                               / max(body_pts.shape[0], 1)),
                             "tibial": int(tib_only["n_inside"]),
                             "femoral": int(np.sum(in_fem))}
    A["bone_penetration"]["excluded_transition_fraction"] = float(1.0 - midbody.mean())
    A["regen_hash"] = regen_hash(Vsurf, loft["center"],
                                 np.concatenate([v.ravel() for v in paths.values()]))
    A["footprint_areas"] = {"tibial": fp_tib.area(), "femoral": fp_fem.area(),
                            "tibial_sub": fp_tib.subregion_areas(),
                            "femoral_sub": fp_fem.subregion_areas()}
    A["profiles"] = {
        "t": t_grid,
        "width": loft["width"], "thick": loft["thick"], "area": areas,
        "section_rotation_deg": np.degrees(loft["rot_rad"]),
        "boundary_deg": loft["boundary_deg"]}
    return A
