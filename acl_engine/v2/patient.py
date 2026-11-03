"""Adapter: external MRI-analysis output -> geometry-engine design state.

Input schema "acl-mri-input" (documented in docs/PATIENT_INPUT.md):

    {"schema": "acl-mri-input", "schema_version": "1.0.0",
     "case_id": "...", "synthetic": true|false,
     "source": {"model_name", "model_version", "run_id", "created"},
     "imaging": {"units", "voxel_spacing_mm", "orientation", "coordinate_convention",
                 "image_to_model" (4x4), "laterality", "acquisition_pose": {"flexion_deg"}},
     "bone": {...}, "attachments": {...}, "ligament": {...}, "injury": {...},
     "structure": {...}, "scaffold_design": {...}, "reference": {...}}

Every leaf may be a bare value or a QUANTITY:
    {"value": ..., "units": "mm", "status": "measured|inferred|proposed|missing|
     unsupported|awaiting_review", "confidence": 0-1, "uncertainty": {...},
     "region": "...", "source": "...", "note": "..."}

Points are in the IMAGE frame unless "frame": "model" is given; image points are
mapped with imaging.image_to_model (mm). The model frame is knee-local:
+X lateral (for THIS knee), +Y anterior, +Z proximal, origin at the tibial
plateau centre. Left knees therefore need no special handling here; the engine
mirrors X only for display.

Large data (meshes, masks) are referenced, not embedded:
    {"$ref": "relative/path.stl", "format": "stl", "units": "mm", "frame": "image"}
"""
import copy
import json
import math
import os
import numpy as np
from acl_engine.v2 import design as D

SUPPORTED_MAJOR = 1
UNIT = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "um": 1e-3, "µm": 1e-3, "deg": 1.0, "rad": 180.0 / math.pi,
        "": 1.0, "fraction": 1.0, "ratio": 1.0, "count": 1.0, "x": 1.0}
STATUS_OK = {"measured", "inferred", "proposed", "missing", "unsupported", "awaiting_review"}


class AdapterError(ValueError):
    pass


def _is_q(v):
    return isinstance(v, dict) and "value" in v


def _mat(T):
    T = np.asarray(T, float)
    if T.shape != (4, 4):
        raise AdapterError("image_to_model must be a 4x4 matrix")
    if abs(T[3, 3] - 1) > 1e-6 or np.abs(T[3, :3]).max() > 1e-6:
        raise AdapterError("image_to_model must be affine (last row 0 0 0 1)")
    det = np.linalg.det(T[:3, :3])
    if abs(det) < 1e-9:
        raise AdapterError("image_to_model is singular")
    return T


class Importer:
    def __init__(self, data, base_dir="."):
        self.d = data
        self.base = base_dir
        self.report = {"mapped": [], "missing": [], "unsupported": [], "warnings": [], "errors": []}
        self.params = {}
        self.prov = {}
        self.assets = {}
        src = data.get("source") or {}
        self.src = "%s %s" % (src.get("model_name", "unknown model"), src.get("model_version", "?"))
        self.run = src.get("run_id")

    # ------------------------------------------------------------ helpers
    def units_factor(self, q, expect):
        u = (q.get("units") if isinstance(q, dict) else None)
        if u is None:
            u = expect
        if u not in UNIT:
            raise AdapterError("unknown units '%s'" % u)
        if expect in ("mm",) and u not in ("mm", "cm", "m", "um", "µm"):
            raise AdapterError("expected a length, got units '%s'" % u)
        if expect in ("deg",) and u not in ("deg", "rad"):
            raise AdapterError("expected an angle, got units '%s'" % u)
        if expect == "um":
            return UNIT[u] * 1000.0
        return UNIT[u] if expect in ("mm", "deg") else 1.0

    def put(self, pid, value, q, path, default_status="measured", region=None):
        spec = D.BY_ID[pid]
        st = (q.get("status") if isinstance(q, dict) else None) or default_status
        if st not in STATUS_OK:
            self.report["warnings"].append("%s: unknown status '%s' treated as awaiting_review" % (path, st))
            st = "awaiting_review"
        if st in ("missing", "unsupported") or value is None:
            self.prov[pid] = {"status": "missing", "source": self.src, "path": path, "units": spec["units"],
                              "note": "upstream marked as %s; generic default shown" % st}
            self.report["missing"].append(path)
            return
        self.params[pid] = value
        self.prov[pid] = {"status": st, "source": self.src, "run_id": self.run, "path": path, "units": spec["units"],
                          "confidence": q.get("confidence") if isinstance(q, dict) else None,
                          "uncertainty": q.get("uncertainty") if isinstance(q, dict) else None,
                          "region": (q.get("region") if isinstance(q, dict) else None) or region,
                          "note": q.get("note") if isinstance(q, dict) else None}
        self.report["mapped"].append({"path": path, "param": pid, "status": st})

    def scalar(self, node, key, pid, units, path, status="measured"):
        if node is None or key not in node:
            return
        q = node[key]
        v = q["value"] if _is_q(q) else q
        try:
            if v is not None:
                v = float(v) * self.units_factor(q if isinstance(q, dict) else {}, units)
        except AdapterError as e:
            self.report["errors"].append("%s: %s" % (path, e))
            return
        self.put(pid, v, q if isinstance(q, dict) else {}, path, status)

    def point(self, q, path):
        """A point (image frame unless frame=model) -> model frame [x,y,z] mm."""
        v = q["value"] if _is_q(q) else q
        fr = (q.get("frame") if isinstance(q, dict) else None) or self.default_frame
        f = self.units_factor(q if isinstance(q, dict) else {}, "mm")
        P = np.asarray(v, float).reshape(-1, 3) * f
        if fr == "image":
            P = P @ self.T[:3, :3].T + self.T[:3, 3]
        elif fr != "model":
            raise AdapterError("%s: frame must be 'image' or 'model'" % path)
        return P

    def profile(self, node, key, pid, units, path, status="measured"):
        if node is None or key not in node:
            return
        q = node[key]
        v = q["value"] if _is_q(q) else q
        if isinstance(v, dict):
            s, vals = v.get("s"), v.get("values", v.get("v"))
        else:
            self.report["errors"].append("%s: profile needs {s: [...], values: [...]}" % path)
            return
        f = self.units_factor(q if isinstance(q, dict) else {}, units) if units in ("mm", "deg") else 1.0
        try:
            s = [float(x) for x in s]
            vals = [float(x) * f for x in vals]
        except (TypeError, ValueError):
            self.report["errors"].append("%s: profile samples must be numbers" % path)
            return
        if len(s) != len(vals) or len(s) < 2:
            self.report["errors"].append("%s: s and values must have the same length (>= 2)" % path)
            return
        if v.get("direction", "tibia_to_femur") == "femur_to_tibia":
            s = [1 - x for x in s]
        self.put(pid, {"s": s, "v": vals}, q if isinstance(q, dict) else {}, path, status)

    def ref(self, q, path):
        if not (isinstance(q, dict) and "$ref" in q):
            self.report["errors"].append("%s: expected a {$ref: ...} asset reference" % path)
            return None
        p = q["$ref"]
        full = p if os.path.isabs(p) else os.path.normpath(os.path.join(self.base, p))
        if not os.path.isfile(full):
            self.report["errors"].append("%s: referenced file not found: %s" % (path, p))
            return None
        return full

    # ------------------------------------------------------------ main
    def run_import(self):
        d = self.d
        if d.get("schema") != "acl-mri-input":
            raise AdapterError("not an acl-mri-input document (schema = %r)" % d.get("schema"))
        ver = str(d.get("schema_version", ""))
        try:
            major = int(ver.split(".")[0])
        except ValueError:
            raise AdapterError("missing or invalid schema_version")
        if major != SUPPORTED_MAJOR:
            raise AdapterError("schema_version %s is not supported (this engine reads 1.x)" % ver)
        known = {"schema", "schema_version", "case_id", "synthetic", "source", "imaging", "bone", "attachments",
                 "ligament", "injury", "structure", "scaffold_design", "reference", "notes"}
        for k in d:
            if k not in known:
                self.report["unsupported"].append(k)
        im = d.get("imaging") or {}
        units = im.get("units", "mm")
        if units != "mm":
            self.report["warnings"].append("imaging.units = %s; coordinates are converted to mm" % units)
        self.T = _mat(im.get("image_to_model", np.eye(4).tolist()))
        self.default_frame = im.get("coordinate_convention_default_frame", "image")
        lat = (im.get("laterality") or "").lower()
        if lat not in ("left", "right"):
            raise AdapterError("imaging.laterality must be 'left' or 'right'")
        self.put("case.side", lat, {}, "imaging.laterality", "measured")
        pose = im.get("acquisition_pose") or {}
        self.scalar(pose, "flexion_deg", "knee.flexion_deg", "deg", "imaging.acquisition_pose.flexion_deg")
        # ---- bone
        b = d.get("bone") or {}
        for key, which in (("femur_surface", "femur_mesh"), ("tibia_surface", "tibia_mesh")):
            if key in b:
                full = self.ref(b[key], "bone." + key)
                if full:
                    fr = b[key].get("frame", "image")
                    Tm = self.T if fr == "image" else np.eye(4)
                    f = self.units_factor(b[key], "mm")
                    S = np.diag([f, f, f, 1.0])
                    self.assets[which] = {"path": full, "transform": (Tm @ S).tolist(), "status": b[key].get("status", "measured"),
                                          "source": self.src}
                    self.report["mapped"].append({"path": "bone." + key, "param": "assets." + which, "status": "measured"})
        sc = b.get("scale") or {}
        self.scalar(sc, "bicondylar_width_mm", "bone.femur_bicondylar_width_mm", "mm", "bone.scale.bicondylar_width_mm")
        self.scalar(sc, "tibial_plateau_width_mm", "bone.tibia_plateau_width_mm", "mm", "bone.scale.tibial_plateau_width_mm")
        self.scalar(sc, "notch_width_mm", "bone.notch_width_mm", "mm", "bone.scale.notch_width_mm")
        self.scalar(sc, "posterior_tibial_slope_deg", "bone.tibial_slope_deg", "deg", "bone.scale.posterior_tibial_slope_deg")
        lm = b.get("landmarks") or {}
        if lm:
            self.assets["landmarks"] = {}
            for k, q in lm.items():
                try:
                    self.assets["landmarks"][k] = self.point(q, "bone.landmarks." + k)[0].tolist()
                except Exception as e:
                    self.report["errors"].append("bone.landmarks.%s: %s" % (k, e))
        # ---- attachments
        at = d.get("attachments") or {}
        for side, nm in (("tib", "tibial"), ("fem", "femoral")):
            a = at.get(nm)
            if not a:
                continue
            path = "attachments." + nm
            if "center" in a:
                try:
                    c = self.point(a["center"], path + ".center")[0]
                    self.put("fp.%s.center" % side, [round(float(x), 4) for x in c], a["center"] if isinstance(a["center"], dict) else {}, path + ".center")
                except AdapterError as e:
                    self.report["errors"].append(str(e))
            shp = a.get("shape") or {}
            if "kind" in shp:
                self.put("fp.%s.shape" % side, shp["kind"], shp, path + ".shape.kind")
            self.scalar(shp, "length_mm", "fp.%s.length_mm" % side, "mm", path + ".shape.length_mm")
            self.scalar(shp, "width_mm", "fp.%s.width_mm" % side, "mm", path + ".shape.width_mm")
            self.scalar(shp, "orientation_deg", "fp.%s.orientation_deg" % side, "deg", path + ".shape.orientation_deg")
            self.scalar(shp, "curvature_radius_mm", "fp.%s.curvature_mm" % side, "mm", path + ".shape.curvature_radius_mm")
            if "boundary" in a:
                try:
                    B3 = self.point(a["boundary"], path + ".boundary")
                    poly, ctr = boundary_to_chart(B3, side)
                    self.put("fp.%s.polygon" % side, poly, a["boundary"] if isinstance(a["boundary"], dict) else {}, path + ".boundary")
                    self.put("fp.%s.shape" % side, "polygon", {"status": "measured"}, path + ".boundary (shape)")
                    if "center" not in a:
                        self.put("fp.%s.center" % side, ctr.tolist(), {"status": "inferred"}, path + ".boundary (centroid)")
                    self.prov["fp.%s.polygon" % side]["note"] = "boundary projected to the footprint tangent chart (plane fit)"
                except AdapterError as e:
                    self.report["errors"].append(str(e))
            if "area_mm2" in a:
                q = a["area_mm2"]
                self.assets.setdefault("reported", {})[path + ".area_mm2"] = q["value"] if _is_q(q) else q
            bun = a.get("bundles") or {}
            if side == "tib" and "AM" in bun and "PL" in bun:
                try:
                    am = self.point(bun["AM"]["center"], path + ".bundles.AM.center")[0]
                    pl = self.point(bun["PL"]["center"], path + ".bundles.PL.center")[0]
                    v = am - pl
                    # boundary direction in the tibial chart frame at s=0 (N1 ~ anterior, N2 ~ medial)
                    ang = math.degrees(math.atan2(-v[0], v[1]))
                    self.put("bundle.boundary_angle_deg", round(ang, 2), {"status": "inferred"}, path + ".bundles (AM-PL direction)")
                except Exception as e:
                    self.report["errors"].append("%s.bundles: %s" % (path, e))
        # ---- ligament
        lg = d.get("ligament") or {}
        if "centerline" in lg:
            try:
                C = self.point(lg["centerline"], "ligament.centerline")
                if C.shape[0] >= 3:
                    self.put("lig.centerline_points", [[round(float(x), 4) for x in r] for r in C], lg["centerline"] if isinstance(lg["centerline"], dict) else {}, "ligament.centerline")
            except AdapterError as e:
                self.report["errors"].append(str(e))
        if "length_mm" in lg:
            q = lg["length_mm"]
            self.assets.setdefault("reported", {})["ligament.length_mm"] = q["value"] if _is_q(q) else q
            self.report["warnings"].append("ligament.length_mm is recorded for comparison; geometric length follows the attachments and centreline")
        self.profile(lg, "width_profile", "lig.width_profile", "mm", "ligament.width_profile")
        self.profile(lg, "thickness_profile", "lig.thickness_profile", "mm", "ligament.thickness_profile")
        self.profile(lg, "shape_profile", "lig.shape_profile", "", "ligament.shape_profile", "inferred")
        self.profile(lg, "orientation_profile", "lig.orientation_profile", "deg", "ligament.orientation_profile", "inferred")
        self.scalar(lg, "flare_tibial_mm", "lig.flare_tib_mm", "mm", "ligament.flare_tibial_mm")
        self.scalar(lg, "flare_femoral_mm", "lig.flare_fem_mm", "mm", "ligament.flare_femoral_mm")
        bo = lg.get("bundles") or {}
        self.scalar(bo, "am_area_fraction", "bundle.am_fraction", "", "ligament.bundles.am_area_fraction", "inferred")
        self.profile(bo, "am_area_fraction_profile", "bundle.am_fraction_profile", "", "ligament.bundles.am_area_fraction_profile", "inferred")
        if "am_area_fraction_profile" in bo and "bundle.am_fraction_profile" in self.params:
            self.put("bundle.am_fraction_mode", "profile", {"status": "inferred"}, "ligament.bundles.am_area_fraction_profile (mode)")
        self.scalar(bo, "boundary_uncertainty_mm", "bundle.interface_uncertainty_mm", "mm", "ligament.bundles.boundary_uncertainty_mm", "inferred")
        if "section_contours" in lg:
            # measured cross-section outlines: [{s, points: [[x,y,z],...] (image frame unless frame=model)}]
            try:
                items = []
                for k, it in enumerate(lg["section_contours"]):
                    sv = it["s"]["value"] if _is_q(it.get("s")) else it.get("s")
                    pts = self.point({"value": it["points"], "units": it.get("units", "mm"), "frame": it.get("frame", "image")},
                                     "ligament.section_contours[%d]" % k)
                    items.append({"s": float(sv), "points": [[round(float(x), 4) for x in r] for r in pts]})
                if items:
                    self.put("lig.section_contours", items, {"status": "measured"}, "ligament.section_contours")
                    self.put("lig.section_mode", "imported", {"status": "measured"}, "ligament.section_contours (mode)")
            except (AdapterError, KeyError, TypeError, ValueError) as e:
                self.report["errors"].append("ligament.section_contours: %s" % e)
        # ---- injury
        inj = d.get("injury") or {}
        if "type" in inj:
            q = inj["type"]
            self.put("injury.type", q["value"] if _is_q(q) else q, q if isinstance(q, dict) else {}, "injury.type")
        self.scalar(inj, "location_s", "injury.center_s", "", "injury.location_s")
        self.scalar(inj, "gap_length_mm", "injury.gap_mm", "mm", "injury.gap_length_mm")
        tp = inj.get("tear_plane") or {}
        self.scalar(tp, "obliquity_deg", "injury.obliquity_deg", "deg", "injury.tear_plane.obliquity_deg")
        self.scalar(tp, "azimuth_deg", "injury.azimuth_deg", "deg", "injury.tear_plane.azimuth_deg")
        self.scalar(inj, "irregularity_mm", "injury.irregularity_mm", "mm", "injury.irregularity_mm", "inferred")
        if "involvement" in inj:
            q = inj["involvement"]
            self.put("injury.involvement", q["value"] if _is_q(q) else q, q if isinstance(q, dict) else {}, "injury.involvement")
        for bname, key in (("AM", "am"), ("PL", "pl")):
            bi = (inj.get("bundles") or {}).get(bname)
            if bi:
                # per-bundle tear (location, missing length, depth from the outer surface)
                self.scalar(bi, "location_s", "injury.%s.center_s" % key, "", "injury.bundles.%s.location_s" % bname)
                self.scalar(bi, "gap_length_mm", "injury.%s.gap_mm" % key, "mm", "injury.bundles.%s.gap_length_mm" % bname)
                self.scalar(bi, "depth_fraction", "injury.%s.depth_fraction" % key, "", "injury.bundles.%s.depth_fraction" % bname)
        sec = inj.get("sector") or {}
        self.scalar(sec, "center_deg", "injury.sector_center_deg", "deg", "injury.sector.center_deg")
        self.scalar(sec, "width_deg", "injury.sector_width_deg", "deg", "injury.sector.width_deg")
        self.scalar(sec, "depth_fraction", "injury.depth_fraction", "", "injury.sector.depth_fraction")
        self.scalar(inj, "continuity_fraction", "injury.continuity", "", "injury.continuity_fraction", "inferred")
        rt = inj.get("retraction") or {}
        self.scalar(rt, "tibial_mm", "injury.retraction_tib_mm", "mm", "injury.retraction.tibial_mm")
        self.scalar(rt, "femoral_mm", "injury.retraction_fem_mm", "mm", "injury.retraction.femoral_mm")
        self.profile(inj, "regional_damage", "injury.damage_profile", "", "injury.regional_damage", "inferred")
        if "mask" in inj:
            m = inj["mask"]
            full = self.ref(m, "injury.mask")
            if full:
                aff = np.asarray(m.get("voxel_to_image", np.eye(4).tolist()), float)
                self.assets["injury_mask"] = {"path": full, "voxel_to_model": (self.T @ aff).tolist(),
                                              "status": m.get("status", "measured"), "source": self.src}
                self.report["mapped"].append({"path": "injury.mask", "param": "assets.injury_mask", "status": m.get("status", "measured")})
        if "stump_surfaces" in inj:
            self.assets["stump_meshes"] = []
            for k, q in enumerate(inj["stump_surfaces"]):
                full = self.ref(q, "injury.stump_surfaces[%d]" % k)
                if full:
                    self.assets["stump_meshes"].append({"path": full, "transform": (self.T if q.get("frame", "image") == "image" else np.eye(4)).tolist(),
                                                        "label": q.get("label", "stump %d" % k), "status": q.get("status", "measured")})
        # ---- structure
        st = d.get("structure") or {}
        self.scalar(st, "bundle_rotation_deg", "bundle.rotation_total_deg", "deg", "structure.bundle_rotation_deg", "inferred")
        self.profile(st, "twist_profile", "bundle.rotation_profile", "", "structure.twist_profile", "inferred")
        if "twist_profile" in st and "bundle.rotation_profile" in self.params:
            self.put("bundle.rotation_distribution", "custom", {"status": "inferred"}, "structure.twist_profile (distribution)")
        # total end-to-end rotation within each unit (surgeon-informed range -30..30 deg)
        self.scalar(st, "fascicle_rotation_deg", "native.fascicle_rotation_deg", "deg", "structure.fascicle_rotation_deg", "inferred")
        self.scalar(st, "fiber_rotation_deg", "native.fiber_rotation_deg", "deg", "structure.fiber_rotation_deg", "inferred")
        for k in ("fascicle_angle_deg", "fiber_angle_deg"):
            if k in st:
                q = st[k]
                self.assets.setdefault("reported", {})["structure." + k] = q["value"] if _is_q(q) else q
                self.report["warnings"].append("structure.%s is a local helix angle; it is recorded but not applied (a constant helix "
                                               "angle winds far beyond 30 deg over a unit). Send structure.%s (total end-to-end "
                                               "rotation, -30..30 deg) instead" % (k, k.replace("angle", "rotation")))
        self.scalar(st, "orientation_dispersion_deg", "native.dispersion_deg", "deg", "structure.orientation_dispersion_deg", "inferred")
        self.scalar(st, "fascicle_count", "native.fascicle_count", "", "structure.fascicle_count", "inferred")
        # ---- scaffold proposals
        sd = d.get("scaffold_design") or {}
        mp = {"overlap_tibial_mm": ("scaffold.overlap_tib_mm", "mm"), "overlap_femoral_mm": ("scaffold.overlap_fem_mm", "mm"),
              "fill_fraction": ("scaffold.fill_fraction", ""), "yarn_count": ("scaffold.yarn_count", ""),
              "yarn_diameter_mm": ("scaffold.yarn_diameter_mm", "mm"), "braid_angle_deg": ("scaffold.braid_angle_deg", "deg"),
              "fiber_diameter_um": ("scaffold.fiber_diameter_um", "um")}
        for k, (pid, u) in mp.items():
            self.scalar(sd, k, pid, u, "scaffold_design." + k, "proposed")
        for k, pid in (("architecture", "scaffold.arch"), ("engagement", "scaffold.engagement"), ("braid_pattern", "scaffold.braid_pattern")):
            if k in sd:
                q = sd[k]
                v = q["value"] if _is_q(q) else q
                if k == "engagement" and v == "sleeve":
                    # retired: a sleeve wraps the remnants OUTSIDE the anatomical envelope
                    self.report["warnings"].append("scaffold_design.engagement 'sleeve' is no longer supported (it places scaffold outside the "
                                                "ligament envelope); imported as 'abut'")
                    v = "abut"
                self.put(pid, v, q if isinstance(q, dict) else {}, "scaffold_design." + k, "proposed")
        for reg in sd.get("regions", []) or []:
            rid = reg.get("id")
            if rid not in D.REGION_IDS:
                self.report["unsupported"].append("scaffold_design.regions[%s]" % rid)
                continue
            for k, (field, u) in {"material": ("material", None), "composition": ("composition", None),
                                  "fiber_volume_fraction": ("fiber_volume_fraction", ""), "alignment_dispersion_deg": ("alignment_dispersion_deg", "deg"),
                                  "strand_scale": ("strand_scale", ""), "transition_mm": ("transition_mm", "mm"),
                                  "transition_mode": ("transition_mode", None)}.items():
                if k in reg:
                    pid = "regions.%s.%s" % (rid, field)
                    q = reg[k]
                    v = q["value"] if _is_q(q) else q
                    if u in ("mm", "deg"):
                        v = float(v) * self.units_factor(q if isinstance(q, dict) else {}, u)
                    if field == "composition":
                        v = D.normalize_composition(v)
                    self.put(pid, v, q if isinstance(q, dict) else {}, "scaffold_design.regions[%s].%s" % (rid, k), "proposed")
        # ---- reference anatomy
        rf = d.get("reference") or {}
        con = rf.get("contralateral") or {}
        if con:
            self.assets["reference_source"] = "contralateral"
            cl = con.get("ligament") or {}
            for key, pid in (("width_profile", "lig.width_profile"), ("thickness_profile", "lig.thickness_profile")):
                if key in cl and pid not in self.params:
                    self.profile(cl, key, pid, "mm", "reference.contralateral.ligament." + key, "inferred")
                    self.prov[pid]["note"] = "from the contralateral knee (mirrored); used as the intact estimate"
            if "registration" in con:
                self.assets["contralateral_registration"] = con["registration"]
        if "intact_estimate" in rf:
            self.assets["intact_estimate_note"] = "upstream intact estimate supplied"
        # ---- validate against the engine registry
        base = D.default_params()
        trial = dict(base, **self.params)
        errs, warns = D.validate_params(trial)
        for e in errs:
            if e["param"] in self.params:
                self.report["errors"].append("%s (%s): %s" % (self.prov.get(e["param"], {}).get("path", e["param"]), e["param"], e["message"]))
        for w in warns:
            self.report["warnings"].append("%s: %s" % (w["param"], w["message"]))
        return self


def boundary_to_chart(B3, side):
    """3D boundary (model frame) -> 2D polygon in the footprint tangent chart + centroid.

    Chart axes follow acl_engine.v2.ligament.Footprint: e_l = reference axis
    (tibia +Y, femur +Z) projected on the plane, e_s = n x e_l. The plane is a
    least-squares fit to the boundary (normal oriented away from bone: +Z for the
    tibia, -X (medial, into the notch) for the femur)."""
    B3 = np.asarray(B3, float)
    if B3.shape[0] < 3:
        raise AdapterError("footprint boundary needs >= 3 points")
    c = B3.mean(0)
    U, S, Vt = np.linalg.svd(B3 - c)
    n = Vt[2]
    want = np.array([0, 0, 1.0]) if side == "tib" else np.array([-1.0, 0, 0])
    if np.dot(n, want) < 0:
        n = -n
    ref = np.array([0, 1.0, 0]) if side == "tib" else np.array([0, 0, 1.0])
    el = ref - np.dot(ref, n) * n
    el /= np.linalg.norm(el)
    es = np.cross(n, el)
    Q = B3 - c
    poly = np.stack([Q @ el, Q @ es], 1)
    return [[round(float(a), 4), round(float(b), 4)] for a, b in poly], c


def import_document(data, base_dir="."):
    imp = Importer(data, base_dir).run_import()
    if imp.report["errors"]:
        return None, imp.report
    params = D.default_params()
    prov = D.default_provenance()
    params.update(imp.params)
    prov.update(imp.prov)
    for pid in params:
        if pid not in imp.params and prov[pid]["status"] in ("default", "literature"):
            prov[pid] = dict(prov[pid], note="not supplied by the MRI model: generic value")
    case = {"id": data.get("case_id", "imported"), "label": "%s%s" % (data.get("case_id", "imported case"),
            " (SYNTHETIC)" if data.get("synthetic") else ""), "synthetic": bool(data.get("synthetic")),
            "source": data.get("source"), "notes": data.get("notes", "")}
    imported = {"params": copy.deepcopy(params), "provenance": copy.deepcopy(prov), "assets": copy.deepcopy(imp.assets),
                "meta": {"case_id": case["id"], "source": data.get("source"), "schema_version": data.get("schema_version"),
                         "report": imp.report}}
    state = {"format": D.SCHEMA_STATE if hasattr(D, "SCHEMA_STATE") else "acl-design-state/2.0", "case": case,
             "params": params, "provenance": prov, "imported": imported, "assets": imp.assets, "history": [],
             "display": dict(D.DISPLAY_DEFAULTS)}
    state["format"] = "acl-design-state/2.0"
    return state, imp.report


def import_file(path):
    with open(path) as f:
        data = json.load(f)
    return import_document(data, os.path.dirname(os.path.abspath(path)))


def merge_import(current, new_state, mode="merge"):
    """Bring new upstream results into an edited design without silently losing edits.

    mode "replace": take the new import entirely (edits are discarded; history keeps them).
    mode "merge": for every parameter the user overrode, keep the user's value and report a
    conflict if the upstream value changed; all other parameters take the new upstream value.
    Returns (state, conflicts)."""
    if mode == "replace" or not current:
        st = copy.deepcopy(new_state)
        st["history"] = list((current or {}).get("history", [])) + [{"action": "import_replace", "case": st["case"]["id"]}]
        return st, []
    st = copy.deepcopy(new_state)
    conflicts = []
    old_imp = (current.get("imported") or {}).get("params", {})
    for pid, pv in current["provenance"].items():
        if pv.get("status") == "override":
            mine = current["params"][pid]
            up_old = old_imp.get(pid)
            up_new = new_state["params"].get(pid)
            st["params"][pid] = mine
            st["provenance"][pid] = dict(pv)
            if up_new != up_old:
                conflicts.append({"param": pid, "yours": mine, "previous_upstream": up_old, "new_upstream": up_new})
    st["history"] = list(current.get("history", [])) + [{"action": "import_merge", "case": st["case"]["id"], "conflicts": len(conflicts)}]
    st["display"] = dict(current.get("display") or D.DISPLAY_DEFAULTS)
    return st, conflicts


def apply_edit(state, pid, value, actor="user"):
    """Manual override of one parameter with history (no clamping; validation reports)."""
    if pid not in D.BY_ID:
        raise KeyError(pid)
    old = state["params"].get(pid)
    state["params"][pid] = value
    pv = dict(state["provenance"].get(pid, {}))
    if pv.get("status") != "override":
        pv["previous_status"] = pv.get("status")
    pv["status"] = "override"
    pv["source"] = actor
    state["provenance"][pid] = pv
    state.setdefault("history", []).append({"action": "edit", "param": pid, "from": old, "to": value, "actor": actor})
    return state


def reset_param(state, pid):
    """Back to the imported value (or the engine default when nothing was imported)."""
    imp = state.get("imported")
    if imp and pid in imp["params"]:
        val, pv = copy.deepcopy(imp["params"][pid]), copy.deepcopy(imp["provenance"][pid])
    else:
        val, pv = copy.deepcopy(D.BY_ID[pid]["default"]), D.default_provenance()[pid]
    old = state["params"].get(pid)
    state["params"][pid] = val
    state["provenance"][pid] = pv
    state.setdefault("history", []).append({"action": "reset", "param": pid, "from": old, "to": val})
    return state


def diff_vs_imported(state):
    imp = state.get("imported")
    if not imp:
        return []
    out = []
    for pid, v in state["params"].items():
        if imp["params"].get(pid) != v:
            out.append({"param": pid, "imported": imp["params"].get(pid), "current": v})
    return out


# ---------------------------------------------------------------- masks -> boundary table
def mask_boundary_table(asset, lig, theta_tear, nbins=36):
    """Damaged-voxel mask (npy, 0/1) -> per-angle damaged interval along the centreline.

    Voxel centres are mapped to the model frame, projected onto the nearest
    centreline station (sigma) and the section angle about N1 (rotated by the
    bundle-region rotation at the tear, like Injury.in_region)."""
    from scipy.spatial import cKDTree
    M = np.load(asset["path"])
    idx = np.argwhere(M > 0.5)
    if idx.shape[0] == 0:
        return None
    T = np.asarray(asset["voxel_to_model"], float)
    P = idx @ T[:3, :3].T + T[:3, 3]
    tree = cKDTree(lig.C)
    _, k = tree.query(P)
    sig = lig.s[k] * lig.L
    Q = P - lig.C[k]
    a1 = (Q * lig.N1[k]).sum(1)
    a2 = (Q * lig.N2[k]).sum(1)
    ang = np.degrees(np.arctan2(a2, a1) - theta_tear)
    ang = (ang + 180) % 360 - 180
    edges = np.linspace(-180, 180, nbins + 1)
    lo, hi, cen = [], [], []
    for i in range(nbins):
        sel = (ang >= edges[i]) & (ang < edges[i + 1])
        if sel.sum() < 2:
            continue
        cen.append(0.5 * (edges[i] + edges[i + 1]))
        lo.append(float(sig[sel].min()))
        hi.append(float(sig[sel].max()))
    if len(cen) < 3:
        return None
    return {"angles_deg": cen, "lo_mm": lo, "hi_mm": hi, "voxels": int(idx.shape[0])}
