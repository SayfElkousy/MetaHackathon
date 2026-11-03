"""Schema 4.0.0. Accepts 2.0.0/3.0.0/4.0.0, migrates, wraps parameter records.

Sec 1.3: sagittal / coronal / axial / twist / attachment / braid / tear angles
are separate stored variables, never merged. Tear / scaffold-role / composition
inputs are accepted and stored; their GEOMETRY is not_implemented until T3/T4/T5.
"""
import copy
import hashlib
import json
import math
from datetime import date
from acl_engine.records import as_record, get_value, is_unknown

FORMAT = "acl-scaffold-geometry"
VERSIONS = ("2.0.0", "3.0.0", "4.0.0")
CANON_VERSION = "4.0.0"
TOP_REQ = {"format", "schema_version", "units", "concept", "anatomy",
           "architecture", "strand", "zones", "core",
           "porosity", "fit", "mold", "quality", "materials"}
TOP_OPT = {"axis", "knee_scene", "hierarchy", "transitions",
           "measurement_conditions", "tear", "scaffold", "composition",
           "surrounding", "acl_model"}


def _err(m):
    raise ValueError(m)


def _finite(x, w):
    if not isinstance(x, (int, float)):
        _err("%s must be number, got %r" % (w, x))
    if not math.isfinite(float(x)):
        _err("%s must be finite" % w)
    return float(x)


def _pos(x, w):
    v = _finite(x, w)
    if v <= 0:
        _err("%s must be >0" % w)
    return v


def _nonneg(x, w):
    v = _finite(x, w)
    if v < 0:
        _err("%s must be >=0" % w)
    return v


def _frac(x, w):
    v = _finite(x, w)
    if v < 0 or v > 1:
        _err("%s fraction must be in [0,1]" % w)
    return v


def _parse_date(s):
    if s is None:
        return None
    try:
        y, m, d = [int(v) for v in str(s).split("-")]
        return date(y, m, d)
    except Exception:
        _err("bad date %r, use YYYY-MM-DD" % (s,))


def parse_measurement_conditions(d):
    if d is None:
        d = {}
    if not isinstance(d, dict):
        _err("measurement_conditions must be object")
    if set(d.keys()) - {"side", "coordinate_system", "knee_position",
                         "imaging", "observers", "segmentation_uncertainty_mm"}:
        _err("bad measurement_conditions keys")
    side = d.get("side", "right")
    if side not in ("right", "left"):
        _err("bad side")
    cs = d.get("coordinate_system", {"axes": "+X lateral, +Y anterior, +Z proximal",
                                     "origin": "tibial plateau center",
                                     "landmarks": []})
    kp = d.get("knee_position", {})
    flex = float(kp.get("flexion_deg", 0.0))
    if not (0 <= flex <= 135):
        _err("flexion out of range")
    rot = float(kp.get("tibial_rotation_deg", 0.0))
    loaded = kp.get("loaded", None)
    img = d.get("imaging", {})
    scan = _parse_date(img.get("scan_date", None))
    inj = _parse_date(img.get("injury_date", None))
    plan = _parse_date(img.get("planned_intervention_date", None))
    derived = {}
    if inj is not None and scan is not None:
        derived["days_injury_to_imaging"] = (scan - inj).days
    else:
        derived["days_injury_to_imaging"] = None
    if inj is not None and plan is not None:
        derived["days_injury_to_intervention"] = (plan - inj).days
    else:
        derived["days_injury_to_intervention"] = None
    obs = d.get("observers", {"count": 1, "agreement_kappa": None})
    seg = d.get("segmentation_uncertainty_mm", None)
    if seg is not None:
        _nonneg(float(seg), "segmentation_uncertainty_mm")
    return {"side": side, "coordinate_system": cs,
            "knee_position": {"flexion_deg": flex,
                              "tibial_rotation_deg": rot, "loaded": loaded},
            "imaging": {"modality": img.get("modality", None),
                        "sequences": img.get("sequences", []),
                        "voxel_size_mm": img.get("voxel_size_mm", None),
                        "slice_thickness_mm": img.get("slice_thickness_mm", None),
                        "scan_date": img.get("scan_date", None),
                        "injury_date": img.get("injury_date", None),
                        "planned_intervention_date": img.get("planned_intervention_date", None)},
            "observers": obs, "segmentation_uncertainty_mm": seg,
            "derived": derived}


TEAR_KEYS = {"severity", "disrupted_fraction", "involvement",
             "preserved_bridges", "location_s", "distal_remnant_percent",
             "obliquity_deg", "azimuth_deg", "morphology",
             "retraction_femoral_mm", "retraction_tibial_mm",
             "tissue_loss_mm", "remnant_end_bulb", "lateral_offset_mm",
             "offset_direction_deg", "relative_rotation_deg", "sag_deg",
             "femoral_remnant_flip_deg", "avulsion", "tissue_quality",
             "remnant_mobility", "synovial_covering", "scarring_adhesion",
             "gap_reference_flexion_deg", "preset", "observed"}


def parse_tear(d):
    if d is None:
        return {"severity": "intact", "disrupted_fraction": 0.0,
                "involvement": "whole", "location_s": None,
                "observed": False, "status": "not_implemented_geometry_T3"}
    if not isinstance(d, dict):
        _err("tear must be object")
    extra = set(d.keys()) - TEAR_KEYS
    if extra:
        _err("unknown tear field(s): %s" % sorted(extra))
    # Milestone A: expand tear presets on the raw input so explicit fields win
    if d.get("preset", None) is not None:
        from acl_engine.tear import apply_tear_preset
        d = apply_tear_preset(d)
    sev = d.get("severity", "intact")
    if sev not in ("intact", "partial", "complete"):
        _err("bad tear severity")
    out = {"severity": sev,
           "disrupted_fraction": d.get("disrupted_fraction",
                                       1.0 if sev == "complete" else 0.0),
           "involvement": d.get("involvement", "whole"),
           "location_s": d.get("location_s", None),
           "obliquity_deg": d.get("obliquity_deg", 0.0),
           "azimuth_deg": d.get("azimuth_deg", 0.0),
           "morphology": d.get("morphology", "transverse"),
           "preset": d.get("preset", None),
           "observed": bool(d.get("observed", False)),
           "status": "not_implemented_geometry_T3"}
    for k in ("retraction_femoral_mm", "retraction_tibial_mm",
              "tissue_loss_mm", "lateral_offset_mm", "sag_deg",
              "femoral_remnant_flip_deg", "gap_reference_flexion_deg"):
        if k in d:
            out[k] = d[k]
    for k in ("preserved_bridges", "avulsion", "tissue_quality",
              "remnant_mobility", "synovial_covering", "scarring_adhesion",
              "remnant_end_bulb", "offset_direction_deg",
              "relative_rotation_deg", "distal_remnant_percent"):
        if k in d:
            out[k] = d[k]
    # Milestone A: T3 defaults + validation (new module acl_engine/tear.py)
    from acl_engine.tear import tear_defaults
    out = tear_defaults(out)
    return out


def parse_scaffold_block(d):
    if d is None:
        return {"role": "repair_bridge",
                "status": "not_implemented_geometry_T4"}
    if not isinstance(d, dict):
        _err("scaffold must be object")
    role = d.get("role", "repair_bridge")
    if role not in ("repair_bridge", "augmentation", "replacement"):
        _err("bad scaffold role")
    out = {"role": role, "status": "not_implemented_geometry_T4"}
    for k in ("end_overlap_femoral_mm", "end_overlap_tibial_mm",
              "fit_allowance_mm", "taper_length_mm", "conform_ends",
              "transition_length_mm", "sector_center_deg", "offset_mm",
              "pretension_N", "fixation_knee_flexion_deg"):
        if k in d:
            out[k] = d[k]
    return out


def parse_composition_block(d):
    if d is None:
        return {"status": "not_implemented_T5"}
    if not isinstance(d, dict):
        _err("composition must be object")
    out = {"status": "not_implemented_T5"}
    for k in ("polymers", "regions", "components", "gradients",
              "fraction_basis", "distribution_pattern",
              "composition_tolerance"):
        if k in d:
            out[k] = d[k]
    return out


ACL_MODEL_KEYS = {"centerline_controls", "section_keys", "boundary_keys",
                  "bundle_sharpness", "strand_count", "strand_margin",
                  "strand_radius_mm", "swirl_total_deg",
                  "tibial_footprint", "femoral_footprint",
                  "flexion_coupling_deg", "n_station", "m_ring", "label"}
FOOTPRINT_KEYS = {"center", "axes", "rotation_deg", "exponent", "split_deg",
                  "transition_mm"}


def _parse_footprint(d, w):
    if d is None:
        return None
    if not isinstance(d, dict):
        _err("%s must be object" % w)
    extra = set(d.keys()) - FOOTPRINT_KEYS
    if extra:
        _err("unknown %s field(s): %s" % (w, sorted(extra)))
    out = {}
    if "center" in d:
        c = d["center"]
        if not (isinstance(c, list) and len(c) == 2):
            _err("%s.center must be [x, y]" % w)
        out["center"] = [float(_finite(v, w + ".center")) for v in c]
    if "axes" in d:
        a = d["axes"]
        if not (isinstance(a, list) and len(a) == 2):
            _err("%s.axes must be [a, b]" % w)
        out["axes"] = [float(_pos(v, w + ".axes")) for v in a]
    for k in ("rotation_deg", "split_deg"):
        if k in d:
            out[k] = float(_finite(d[k], "%s.%s" % (w, k)))
    if "exponent" in d:
        out["exponent"] = float(_pos(d["exponent"], w + ".exponent"))
    if "transition_mm" in d:
        out["transition_mm"] = float(_nonneg(d["transition_mm"], w + ".transition_mm"))
    return out


def parse_acl_model(d):
    """Optional parametric-loft controls. All keys optional; None/absent
    means derive from the validated anatomy (recorded as assumptions)."""
    if d is None:
        return {"status": "derived"}
    if not isinstance(d, dict):
        _err("acl_model must be object")
    extra = set(d.keys()) - ACL_MODEL_KEYS
    if extra:
        _err("unknown acl_model field(s): %s" % sorted(extra))
    out = {"status": "user_controls"}
    if d.get("centerline_controls", None) is not None:
        pts = d["centerline_controls"]
        if not (isinstance(pts, list) and len(pts) >= 2):
            _err("acl_model.centerline_controls must be a list of >= 2 [x,y,z]")
        clean = []
        for p in pts:
            if not (isinstance(p, list) and len(p) == 3):
                _err("acl_model.centerline_controls entries must be [x,y,z]")
            clean.append([float(_finite(v, "centerline_controls")) for v in p])
        out["centerline_controls"] = clean
    if d.get("section_keys", None) is not None:
        keys = []
        for k in d["section_keys"]:
            if set(k.keys()) - {"t", "width", "thickness", "exponent", "rotation_deg"}:
                _err("bad acl_model.section_keys entry %r" % (k,))
            keys.append({"t": float(_finite(k["t"], "section t")),
                         "width": float(_pos(k["width"], "section width")),
                         "thickness": float(_pos(k["thickness"], "section thickness")),
                         "exponent": float(_pos(k.get("exponent", 1.0), "section exponent")),
                         "rotation_deg": float(_finite(k.get("rotation_deg", 0.0),
                                                       "section rotation"))})
        out["section_keys"] = keys
    if d.get("boundary_keys", None) is not None:
        out["boundary_keys"] = [{"t": float(_finite(k["t"], "boundary t")),
                                 "angle_deg": float(_finite(k["angle_deg"], "boundary angle"))}
                                for k in d["boundary_keys"]]
    for k in ("bundle_sharpness",):
        if k in d:
            v = float(_finite(d[k], "acl_model." + k))
            if not (0.0 <= v <= 1.0):
                _err("acl_model.bundle_sharpness must be in [0,1]")
            out[k] = v
    if "strand_count" in d:
        n = d["strand_count"]
        if not (isinstance(n, int) and 4 <= n <= 200):
            _err("acl_model.strand_count must be int in [4,200]")
        out["strand_count"] = n
    if "strand_margin" in d:
        v = float(_finite(d["strand_margin"], "acl_model.strand_margin"))
        if not (0.5 <= v < 1.0):
            _err("acl_model.strand_margin must be in [0.5,1)")
        out["strand_margin"] = v
    if "strand_radius_mm" in d:
        out["strand_radius_mm"] = float(_pos(d["strand_radius_mm"], "acl_model.strand_radius_mm"))
    if "swirl_total_deg" in d:
        out["swirl_total_deg"] = float(_finite(d["swirl_total_deg"], "acl_model.swirl_total_deg"))
    if "flexion_coupling_deg" in d:
        out["flexion_coupling_deg"] = float(_finite(d["flexion_coupling_deg"],
                                                    "acl_model.flexion_coupling_deg"))
    for k in ("n_station", "m_ring"):
        if k in d:
            v = d[k]
            if not (isinstance(v, int) and v >= 8):
                _err("acl_model.%s must be int >= 8" % k)
            out[k] = v
    for k in ("tibial_footprint", "femoral_footprint"):
        if d.get(k, None) is not None:
            out[k] = _parse_footprint(d[k], "acl_model." + k)
    if "label" in d:
        if not isinstance(d["label"], str):
            _err("acl_model.label must be string")
        out["label"] = d["label"]
    return out


def validate_request(obj):
    if not isinstance(obj, dict):
        _err("request must be object")
    extra = set(obj.keys()) - (TOP_REQ | TOP_OPT)
    if extra:
        _err("unknown top field(s): %s" % sorted(extra))
    for k in TOP_REQ:
        if k not in obj:
            _err("missing top key %s" % k)
    if obj["format"] != FORMAT:
        _err("bad format")
    if obj["schema_version"] not in VERSIONS:
        _err("bad schema_version")
    from_ver = obj["schema_version"]
    u = obj["units"]
    if u.get("length") != "mm" or u.get("angle") != "deg" or u.get("fraction") != "0_to_1":
        _err("bad units")
    if obj["concept"] not in ("tunnel_spanning_research", "repair_bridge"):
        _err("bad concept")
    assumptions = []
    an = obj["anatomy"]
    if not isinstance(an, dict):
        _err("anatomy must be object")
    if "envelope" in an:
        env = an["envelope"]
        W = _pos(env["width_mm"], "envelope.width_mm")
        Tk = _pos(env["thickness_mm"], "envelope.thickness_mm")
        acl_len = None
        sag = 47.4
        cor = 75.3
        axial = 0.0
        twist = 0.0
        twist_dist = "uniform"
        attach_rot = 0.0
        flare = 3.5
        profile = [34.0, 33.0, 35.0, 38.0, 42.0]
        realism = an.get("realism", "schematic")
        if realism not in ("schematic", "anatomical"):
            _err("anatomy.realism must be schematic|anatomical")
        assumptions.append({"field": "anatomy.acl_length_mm",
                            "preset": "middle zone length",
                            "origin": "prototype_placeholder"})
    else:
        r_len = as_record(an.get("acl_length_mm", 32.0), "mm", "acl_length_mm")
        r_w = as_record(an.get("acl_width_mm", 10.0), "mm", "acl_width_mm")
        r_t = as_record(an.get("acl_thickness_mm", 5.0), "mm", "acl_thickness_mm")
        r_sag = as_record(an.get("sagittal_inclination_deg", 47.4), "deg", "sagittal")
        r_cor = as_record(an.get("coronal_inclination_deg", 75.3), "deg", "coronal")
        r_ax = as_record(an.get("axial_orientation_deg", 0.0), "deg", "axial")
        r_tw = as_record(an.get("anatomical_twist_total_deg", 0.0), "deg", "twist")
        r_ar = as_record(an.get("attachment_rotation_deg", 0.0), "deg", "attach_rot")
        r_fl = as_record(an.get("insertion_flare_ratio", 3.5), "fraction", "flare")
        if is_unknown(r_len):
            assumptions.append({"field": "acl_length_mm", "preset": 32.0,
                                "origin": "prototype_placeholder", "source": "L01"})
            acl_len = 32.0
        else:
            acl_len = _pos(get_value(r_len), "acl_length_mm")
        if is_unknown(r_w):
            assumptions.append({"field": "acl_width_mm", "preset": 10.0,
                                "origin": "prototype_placeholder"})
            W = 10.0
        else:
            W = _pos(get_value(r_w), "acl_width_mm")
        if is_unknown(r_t):
            assumptions.append({"field": "acl_thickness_mm", "preset": 5.0,
                                "origin": "prototype_placeholder", "source": "L02"})
            Tk = 5.0
        else:
            Tk = _pos(get_value(r_t), "acl_thickness_mm")
        if is_unknown(r_sag):
            sag = 47.4
            assumptions.append({"field": "sagittal_inclination_deg", "preset": 47.4,
                                "origin": "literature_example", "source": "L03"})
        else:
            sag = float(_finite(get_value(r_sag), "sagittal"))
        if is_unknown(r_cor):
            cor = 75.3
            assumptions.append({"field": "coronal_inclination_deg", "preset": 75.3,
                                "origin": "literature_example", "source": "L03"})
        else:
            cor = float(_finite(get_value(r_cor), "coronal"))
        if not (20.0 <= sag <= 80.0):
            _err("sagittal out of plausible range")
        if not (40.0 <= cor <= 90.0):
            _err("coronal out of plausible range")
        axial = float(_finite(get_value(r_ax), "axial_orientation_deg"))
        twist = float(_finite(get_value(r_tw), "anatomical_twist_total_deg"))
        if abs(twist) > 180:
            _err("twist out of range")
        twist_dist = an.get("twist_distribution", "uniform")
        if twist_dist not in ("uniform", "femoral_weighted",
                              "tibial_weighted", "custom"):
            _err("bad twist_distribution")
        realism = an.get("realism", "schematic")
        if realism not in ("schematic", "anatomical"):
            _err("anatomy.realism must be schematic|anatomical")
        attach_rot = float(_finite(get_value(r_ar), "attachment_rotation_deg"))
        if is_unknown(r_fl):
            flare = 3.5
            assumptions.append({"field": "insertion_flare_ratio", "preset": 3.5,
                                "origin": "literature_example", "source": "L04"})
        else:
            flare = float(_finite(get_value(r_fl), "flare"))
        prof_raw = an.get("cross_section_area_profile",
                          [34.0, 33.0, 35.0, 38.0, 42.0])
        if isinstance(prof_raw, dict) and "value" in prof_raw:
            prof_raw = prof_raw["value"]
        if not (isinstance(prof_raw, list) and len(prof_raw) == 5):
            _err("area profile must be list of 5")
        profile = [_pos(float(v), "area_profile") for v in prof_raw]
    ar = obj["architecture"]
    if ar.get("style", "three_strand_plait") != "three_strand_plait":
        if ar.get("style") in ("braid_twist", "twisted_rope",
                               "biaxial_tubular", "triaxial_tubular",
                               "flat_ribbon_braid"):
            _err("style %s is not_implemented in this release" % ar.get("style"))
        _err("bad style")
    _finite(ar.get("phase_offset_deg", 0.0), "phase_offset_deg")
    if not isinstance(ar.get("mirror", False), bool):
        _err("mirror must be bool")
    a = _pos(ar["swing_mm"], "swing_mm")
    fold = ar.get("fold", None)
    q = 0.0
    if fold is not None:
        if not isinstance(fold, dict):
            _err("fold must be object")
        if set(fold.keys()) - {"crossing_pattern", "fold_depth_mm", "fold_sharpness"}:
            _err("bad fold keys")
        b = _pos(get_value(as_record(fold["fold_depth_mm"], "mm", "fold_depth")),
                 "fold.fold_depth_mm")
        qs = get_value(as_record(fold.get("fold_sharpness", 0.0), "fraction", "q"))
        q = _finite(qs, "fold.fold_sharpness")
        if q < 0 or q > 1:
            _err("fold_sharpness must be in [0,1]")
    else:
        if "lift_mm" not in ar:
            _err("need fold.fold_depth_mm or lift_mm")
        b = _pos(ar["lift_mm"], "lift_mm")
    angle_mode = ar.get("angle_control_mode", "crossing_tilt")
    if angle_mode not in ("crossing_tilt", "legacy_lateral_pitch_rule"):
        _err("bad angle_control_mode")
    d = _pos(obj["strand"]["diameter_mm"], "strand.diameter_mm")
    mats = obj["materials"]
    ids = [m["id"] for m in mats]
    if len(set(ids)) != len(ids):
        _err("duplicate material ids")
    zones = obj["zones"]
    if not isinstance(zones, list) or len(zones) < 1:
        _err("zones must be non-empty list")
    ZONE_ALLOW = {"id", "role", "length_mm", "material_id",
                  "planned_porogen_fraction", "control", "braid_angle_deg",
                  "path_length_ratio", "pitch_mm", "cross_section",
                  "braid_tilt_target_deg"}
    zids = []
    for z in zones:
        if set(z.keys()) - ZONE_ALLOW:
            _err("bad zone keys %s" % sorted(set(z.keys()) - ZONE_ALLOW))
        zids.append(z.get("id"))
        _pos(z["length_mm"], "zones.length_mm")
        if z["material_id"] not in ids:
            _err("unknown material_id")
        _frac(z["planned_porogen_fraction"], "zones.planned_porogen_fraction")
        ctl = z.get("control", None)
        if ctl is None:
            if "braid_tilt_target_deg" in z or "braid_angle_deg" in z:
                ctl = "braid_angle"
            elif "path_length_ratio" in z:
                ctl = "fiber_length"
            elif "pitch_mm" in z:
                ctl = "pitch"
            else:
                _err("zone %s needs a control/target" % z.get("id"))
        if ctl not in ("braid_angle", "fiber_length", "pitch"):
            _err("bad zone control")
        if ctl == "braid_angle":
            th = z.get("braid_angle_deg", z.get("braid_tilt_target_deg", None))
            if th is None:
                _err("braid_angle zone needs target")
            th = _finite(get_value(as_record(th, "deg", "braid_angle")), "braid_angle")
            if th <= 0 or th >= 89:
                _err("braid_angle must be in (0,89)")
        elif ctl == "fiber_length":
            lam = z.get("path_length_ratio", None)
            if lam is None:
                _err("fiber_length zone needs path_length_ratio")
            lam = _finite(get_value(as_record(lam, "fraction", "lambda")), "path_length_ratio")
            if lam < 1.0 or lam > 2.0:
                _err("path_length_ratio must be in [1,2]")
        else:
            _pos(get_value(as_record(z.get("pitch_mm", None), "mm", "pitch")), "pitch_mm")
        if "cross_section" in z and z["cross_section"] not in ("round", "ellipse", "flat"):
            _err("bad cross_section")
    if len(set(zids)) != len(zids):
        _err("duplicate zone ids")
    if "transitions" in obj and obj["transitions"] is not None:
        _nonneg(obj["transitions"]["length_mm"], "transitions.length_mm")
        trans_len = float(obj["transitions"]["length_mm"])
    else:
        trans_len = 4.0
        assumptions.append({"field": "transitions.length_mm", "preset": 4.0,
                            "origin": "prototype_placeholder"})
    co = obj["core"]
    if not isinstance(co["enabled"], bool):
        _err("core.enabled bool")
    _pos(co["diameter_mm"], "core.diameter_mm")
    _nonneg(co["minimum_cover_mm"], "core.minimum_cover_mm")
    po = obj["porosity"]
    if po.get("mode") not in ("dense", "process_plan"):
        _err("porosity.mode must be dense|process_plan in this release")
    _pos(po["target_pore_size_um"], "porosity.target_pore_size_um")
    if obj["fit"].get("mode") not in ("manual", "fit_width", "fit_envelope"):
        _err("bad fit.mode")
    mo = obj["mold"]
    if not isinstance(mo["enabled"], bool):
        _err("mold.enabled bool")
    _pos(mo.get("lead_length_mm", 10.0), "mold.lead_length_mm")
    _pos(mo.get("funnel_depth_mm", 3.0), "mold.funnel_depth_mm")
    _pos(obj["quality"]["preview_tol_mm"], "quality.preview_tol_mm")
    _pos(obj["quality"]["export_tol_mm"], "quality.export_tol_mm")
    axis = obj.get("axis", {"type": "straight"})
    if not isinstance(axis, dict) or axis.get("type") not in ("straight", "arc"):
        _err("bad axis")
    if axis.get("type") == "arc":
        _pos(axis["radius_mm"], "axis.radius_mm")
    ks = obj.get("knee_scene", {"type": "schematic", "flexion_deg": 0,
                                "tibial_slope_deg": 7})
    if not isinstance(ks, dict) or ks.get("type") not in ("schematic",):
        _err("knee_scene.type must be schematic in this release")
    if not (0 <= float(ks.get("flexion_deg", 0)) <= 90):
        _err("flexion must be in [0,90]")
    hier = obj.get("hierarchy", {"enabled": False})
    if hier.get("enabled", False):
        _err("hierarchy.enabled is not_implemented in this release (T5)")
    mcond = parse_measurement_conditions(obj.get("measurement_conditions", None))
    tear = parse_tear(obj.get("tear", None))
    scaf = parse_scaffold_block(obj.get("scaffold", None))
    comp = parse_composition_block(obj.get("composition", None))
    amod = parse_acl_model(obj.get("acl_model", None))
    side = "right"
    if isinstance(an.get("side", None), str):
        if an["side"] not in ("right", "left"):
            _err("bad side")
        side = an["side"]
    norm = copy.deepcopy(obj)
    norm["schema_version"] = CANON_VERSION
    if "axis" not in norm:
        norm["axis"] = {"type": "straight"}
    if "knee_scene" not in norm:
        norm["knee_scene"] = {"type": "schematic", "flexion_deg": 0,
                              "tibial_slope_deg": 7}
    if "hierarchy" not in norm:
        norm["hierarchy"] = {"enabled": False}
    if "transitions" not in norm or norm["transitions"] is None:
        norm["transitions"] = {"length_mm": 4.0}
    norm["measurement_conditions"] = mcond
    norm["tear"] = tear
    norm["scaffold"] = scaf
    norm["composition"] = comp
    norm["acl_model"] = amod
    L = sum(float(z["length_mm"]) for z in zones)
    bb = [0.0]
    for z in zones:
        bb.append(bb[-1] + float(z["length_mm"]))
    if acl_len is None:
        acl_len = float(zones[len(zones) // 2]["length_mm"])
    norm["_derived"] = {"axis_length_mm": L, "zone_bounds_mm": bb,
                        "swing_mm": a, "fold_depth_mm": b,
                        "fold_sharpness": q, "angle_mode": angle_mode,
                        "envelope_width_mm": W, "envelope_thickness_mm": Tk,
                        "acl_length_mm": float(acl_len),
                        "sagittal_deg": float(sag), "coronal_deg": float(cor),
                        "axial_orientation_deg": float(axial),
                        "anatomical_twist_total_deg": float(twist),
                        "twist_distribution": twist_dist,
                        "realism": realism,
                        "attachment_rotation_deg": float(attach_rot),
                        "flare_ratio": float(flare),
                        "area_profile_femur_to_tibia": [float(v) for v in profile]}
    nzz = []
    for z in zones:
        zn = dict(z)
        if "control" not in zn:
            if "braid_tilt_target_deg" in zn and "braid_angle_deg" not in zn:
                zn["braid_angle_deg"] = zn.pop("braid_tilt_target_deg")
                zn["control"] = "braid_angle"
            elif "braid_angle_deg" in zn:
                zn["control"] = "braid_angle"
            elif "path_length_ratio" in zn:
                zn["control"] = "fiber_length"
            elif "pitch_mm" in zn:
                zn["control"] = "pitch"
        else:
            if zn["control"] == "braid_angle" and "braid_angle_deg" not in zn \
               and "braid_tilt_target_deg" in zn:
                zn["braid_angle_deg"] = zn.pop("braid_tilt_target_deg")
        nzz.append(zn)
    norm["zones"] = nzz
    if "fold" not in norm["architecture"]:
        norm["architecture"]["fold"] = {"crossing_pattern": "1/1",
                                        "fold_depth_mm": b, "fold_sharpness": q}
    norm["_derived"]["side"] = side
    norm["_assumptions"] = assumptions
    mig = {"from": from_ver, "to": CANON_VERSION,
           "notes": ["2.0.0 lift_mm -> fold.fold_depth_mm",
                     "braid_tilt_target_deg -> control braid_angle (old name still accepted)",
                     "bare numbers wrapped to parameter records",
                     "transitions default 4.0 if missing",
                     "1.3 angles stored separately; tear/scaffold/composition stored, geometry deferred"]}
    return norm, mig


def canonical_hash(normalized):
    c = copy.deepcopy(normalized)
    c.pop("_timestamps", None)
    s = json.dumps(c, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
