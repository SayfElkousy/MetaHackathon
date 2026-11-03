"""Design-state model: parameter registry, defaults, provenance, validation.

A design state is plain JSON:

    {"format": "acl-design-state/2.0",
     "case": {...},                      # id, label, synthetic flag, notes
     "params": {pid: value},             # geometry-driving DESIGN parameters
     "provenance": {pid: {...}},         # status/source/confidence per param
     "imported": {"params": {...}, "provenance": {...}, "meta": {...}} | None,
     "assets": {...},                    # referenced meshes / masks
     "history": [...],                   # manual edits, newest last
     "display": {...}}                   # display-only settings (not hashed)

Parameter kinds: scalar, int, enum, bool, profile (values at S_KNOTS),
point3, offsets (list of point3), polygon2 (list of 2D points, mm),
composition ({material: fraction}).

Provenance status values:
  measured | inferred | proposed | override | literature | default |
  missing | awaiting_review | unsupported
"default" and "literature" values are generic (NOT patient-specific).
"""
import copy
import hashlib
import json
import math

S_KNOTS = [0.0, 0.25, 0.5, 0.75, 1.0]

# Canonical internal geometry unit: millimetres. Micro/nano inputs convert explicitly:
#   1 um = 0.001 mm, 1 nm = 0.000001 mm.
# Parameter ids carry their unit as a suffix (_mm, _um, _nm, _deg, ...) wherever the
# value is unit-bearing; ranges in the registry use the same unit as the value.
MM_PER_UM = 0.001
MM_PER_NM = 0.000001
UM_PER_MM = 1000.0
NM_PER_MM = 1000000.0


def to_mm(value, units):
    """Convert a unit-bearing value to the canonical internal unit (mm)."""
    if units in ("mm", "mm2", "mm3"):
        return float(value)
    if units in ("um", "µm"):
        return float(value) * MM_PER_UM
    if units == "nm":
        return float(value) * MM_PER_NM
    raise ValueError("no mm conversion for units %r" % (units,))

# Design tab: the "Anatomy and Pose" panel (reference anatomy, knee pose, tear
# definition, native tissue) holds the first six groups; the "Scaffold Levels"
# panel holds lvl1..lvl6. Every scaffold control is owned by exactly one level.
GROUPS = [
    ("anatomy", "Knee pose and reference anatomy"),
    ("am", "AM region (native) — anteromedial"),
    ("pl", "PL region (native) — posterolateral"),
    ("injury", "Tear definition and retained tissue"),
    ("fibers", "Native fascicles and fibers"),
    ("rotation", "Native rotation and bundle arrangement"),
    ("lvl1", "Level 1 — Surface: nano-texture and protein coating"),
    ("lvl2", "Level 2 — Micro: grooves in channels and inner sheath"),
    ("lvl3", "Level 3 — Fascicle: hollow core channels"),
    ("lvl4", "Level 4 — Bundle: braided reinforcement around AM-like and PL-like regions"),
    ("lvl5", "Level 5 — Whole ligament: ribbon-shaped outer sheath"),
    ("lvl6", "Level 6 — Ends: flared attachment regions and graded interfaces"),
]
ANATOMY_GROUPS = ("anatomy", "am", "pl", "injury", "fibers", "rotation")

# The six structural levels. "objects" are the scene objects each level draws
# (used to highlight / isolate the level in the viewer). Level 0 (patient
# geometry: knee pose, reference anatomy, tear definition, native tissue) is
# the "Anatomy and Pose" panel; it positions the construct but is not scaffold.
LEVELS = [
    {"n": 1, "id": "surface", "name": "Surface coating and nanotexture", "title": "Coating and nanotexture on cell-facing surfaces",
     "component": "Cell-facing surfaces: channel lumens and the inner sheath",
     "objects": ["layer_coating", "layer_coating_sheath"]},
    {"n": 2, "id": "micro", "name": "Microgrooves", "title": "Grooves in channel lumens and on the inner sheath",
     "component": "Guidance surfaces: channel lumen walls and the inner sheath wall",
     "objects": ["layer_grooves"]},
    {"n": 3, "id": "fascicle", "name": "Hollow fascicle channels", "title": "Hollow core channels with walls and lumens",
     "component": "Hollow channels (lumen, wall) and their connections",
     "objects": ["layer_channels", "layer_ports"]},
    {"n": 4, "id": "bundle", "name": "Braided mantles", "title": "Braided reinforcement around AM-like and PL-like regions",
     "component": "Braided mantles around the AM-like and PL-like channel groups",
     "objects": ["layer_braid"]},
    {"n": 5, "id": "whole", "name": "Sheath and inter-bundle zone", "title": "Ribbon-shaped outer sheath and the AM/PL transition band",
     "component": "The shared outer envelope (sheath wall, pores) and the transition band between mantles",
     "objects": ["layer_sheath", "layer_pores", "layer_interface"]},
    {"n": 6, "id": "ends", "name": "Insertions and anchors", "title": "Flared attachment regions, graded interfaces and anchors",
     "component": "Femoral and tibial flared ends, footprints, fixation features and bundle anchors",
     "objects": ["layer_end_tib", "layer_end_fem", "layer_footprint_tib", "layer_footprint_fem", "layer_plug_tib", "layer_plug_fem",
                 "layer_anchors"]},
]
# Level 0 is patient geometry (reference anatomy, pose, tear, native tissue).
LEVEL0 = {"n": 0, "id": "patient", "name": "Patient geometry",
          "title": "Patient frame, pose, footprints, tear definition and native tissue",
          "component": "Reference anatomy that positions the construct (not scaffold)"}
# objects of the gap-filling yarn scaffold, by owning level
LEGACY_LEVEL_OBJECTS = {2: ["scaffold_fibers"], 3: ["scaffold_yarns"], 4: ["scaffold_yarns"],
                        5: ["repair_space", "repair_space_am", "repair_space_pl"], 6: ["scaffold_region_tibial_interface",
                                                                                      "scaffold_region_femoral_interface"]}

MATERIALS = {
    "PLLA": {"label": "poly(L-lactide)", "color": [0.93, 0.80, 0.35]},
    "PCL": {"label": "polycaprolactone", "color": [0.55, 0.80, 0.55]},
    "PLGA": {"label": "poly(lactide-co-glycolide)", "color": [0.85, 0.60, 0.85]},
    "silk": {"label": "silk fibroin", "color": [0.96, 0.94, 0.86]},
    "collagen_I": {"label": "type I collagen", "color": [0.95, 0.70, 0.60]},
    "PET": {"label": "poly(ethylene terephthalate)", "color": [0.60, 0.75, 0.95]},
    "beta_TCP": {"label": "beta-tricalcium phosphate", "color": [0.85, 0.85, 0.80]},
    "HA": {"label": "hydroxyapatite", "color": [0.92, 0.92, 0.95]},
    "fibronectin": {"label": "fibronectin (protein coating)", "color": [0.98, 0.55, 0.45]},
    "laminin": {"label": "laminin (protein coating)", "color": [0.80, 0.50, 0.85]},
    "gelatin": {"label": "gelatin (coating)", "color": [0.98, 0.85, 0.55]},
    "polydopamine": {"label": "polydopamine (adhesion layer)", "color": [0.35, 0.28, 0.25]},
    "hyaluronic_acid": {"label": "hyaluronic acid (coating)", "color": [0.55, 0.85, 0.95]},
}
STRUCTURAL_MATERIALS = ["PLLA", "PCL", "PLGA", "silk", "collagen_I", "PET", "beta_TCP", "HA"]
COATING_MATERIALS = ["collagen_I", "fibronectin", "laminin", "gelatin", "polydopamine", "hyaluronic_acid"]

REGION_IDS = ["tibial_interface", "bridge", "femoral_interface"]


def _p(pid, group, label, kind, default, units="", lo=None, hi=None, step=None,
       sources=None, desc="", role="design", options=None, sub=None, depends=None):
    return {"id": pid, "group": group, "label": label, "kind": kind, "default": default,
            "units": units, "min": lo, "max": hi, "step": step, "sources": sources or [],
            "description": desc, "role": role, "options": options, "subgroup": sub,
            "depends": depends}


def _L(level, pid, sub, label, kind, default, units="", lo=None, hi=None, step=None, desc="", component="",
       neighbors=None, basis="assumed", options=None, region=None, pair=None, materials=None, objects=None, sources=None):
    """A scaffold-level control: owner level, the component it directly modifies, the
    neighbouring levels that may need updating, and its default basis."""
    q = _p(pid, "lvl%d" % level, label, kind, default, units, lo, hi, step, sources, desc, options=options, sub=sub)
    q.update(level=level, component=component, neighbors=[list(x) for x in (neighbors or [])], basis=basis,
             region=region, pair=pair, materials=materials, objects=objects)
    return q


ROT_LABEL = "Total fiber rotation along unit (°)"
ROT_DESC = "End-to-end rotation around the local axis; surgeon-informed design range: −30° to +30°."
ROT_NOTE = (" Signed (the sign gives the direction), spread evenly along the unit and measured on the generated paths in a "
            "non-twisting frame that follows the unit's centreline. A design rule reported by the user from surgeon guidance, "
            "not an anatomical limit on whole-ACL torsion, knee rotation or the AM/PL arrangement.")


def _rot(pid, group, sub, unit_desc, sources=None):
    return _p(pid, group, ROT_LABEL, "scalar", 0.0, "deg", -30, 30, 0.5, sources, ROT_DESC + " Unit: " + unit_desc + "." + ROT_NOTE, sub=sub)


def _registry():
    R = []
    A = "anatomy"
    R += [
        _p("case.side", A, "Knee side", "enum", "right", options=["right", "left"],
           desc="Laterality. Geometry is generated in a right-knee frame and mirrored in X for left knees.", sub="Knee"),
        _p("knee.flexion_deg", A, "Knee flexion (illustrative pose)", "scalar", 0.0, "deg", 0, 90, 1,
           ["L19"], "Rigid rotation of the femur about the mediolateral axis through the condyle centres. "
           "Not a kinematic model: real knees roll and glide.", sub="Knee"),
        _p("bone.femur_bicondylar_width_mm", A, "Femoral bicondylar width", "scalar", 78.0, "mm", 60, 95, 0.5,
           desc="Scales the procedural femur. Ignored when a patient femur mesh is imported.", sub="Bones"),
        _p("bone.tibia_plateau_width_mm", A, "Tibial plateau width", "scalar", 76.0, "mm", 60, 92, 0.5, sub="Bones"),
        _p("bone.notch_width_mm", A, "Intercondylar notch width", "scalar", 19.0, "mm", 12, 26, 0.5, sub="Bones"),
        _p("bone.tibial_slope_deg", A, "Posterior tibial slope", "scalar", 7.0, "deg", 0, 15, 0.5, sub="Bones"),
        _p("bone.cartilage_femoral_mm", A, "Femoral cartilage thickness", "scalar", 2.2, "mm", 0.5, 4, 0.1, sub="Bones"),
        _p("bone.cartilage_tibial_mm", A, "Tibial cartilage thickness", "scalar", 2.0, "mm", 0.5, 4, 0.1, sub="Bones"),
        _p("fp.tib.center", A, "Tibial footprint centre", "point3", [3.0, 9.5, 0.0], "mm",
           sources=["L15"], desc="Projected onto the tibial surface. Drag the handle in the viewer.", sub="Tibial footprint"),
        _p("fp.tib.shape", A, "Tibial footprint shape", "enum", "oval", options=["oval", "crescent", "polygon"], sub="Tibial footprint"),
        _p("fp.tib.length_mm", A, "Tibial footprint length", "scalar", 17.0, "mm", 6, 28, 0.5, ["L15"], sub="Tibial footprint"),
        _p("fp.tib.width_mm", A, "Tibial footprint width", "scalar", 11.0, "mm", 4, 20, 0.5, ["L15"], sub="Tibial footprint"),
        _p("fp.tib.orientation_deg", A, "Tibial footprint orientation", "scalar", 0.0, "deg", -90, 90, 1,
           desc="Long axis angle in the surface tangent plane, measured from anterior (+Y) toward lateral.", sub="Tibial footprint"),
        _p("fp.tib.curvature_mm", A, "Tibial crescent radius", "scalar", 14.0, "mm", 5, 40, 0.5, sub="Tibial footprint"),
        _p("fp.tib.polygon", A, "Tibial footprint boundary", "polygon2", [], "mm",
           desc="Custom boundary in the tangent chart (long axis first). Used when shape = polygon.", sub="Tibial footprint"),
        _p("fp.tib.am_share", A, "AM share of the tibial footprint", "scalar", 0.55, "", 0.2, 0.8, 0.01, ["L12"],
           "Area fraction of the tibial footprint attached to the AM bundle (AM attachment subregion).", sub="Tibial footprint"),
        _p("fp.tib.partition_rotation_deg", A, "AM/PL split rotation on the tibial footprint", "scalar", 0.0, "deg", -90, 90, 1,
           desc="Rotates where the AM and PL subregions sit on the tibial footprint.", sub="Tibial footprint"),
        _p("fp.fem.center", A, "Femoral footprint centre", "point3", [8.0, -15.0, 24.0], "mm",
           sources=["L14", "L16"], desc="Projected onto the lateral wall of the notch (femur frame, extension).", sub="Femoral footprint"),
        _p("fp.fem.shape", A, "Femoral footprint shape", "enum", "crescent", options=["oval", "crescent", "polygon"],
           sources=["L14"], sub="Femoral footprint"),
        _p("fp.fem.length_mm", A, "Femoral footprint length", "scalar", 16.0, "mm", 6, 26, 0.5, ["L14"], sub="Femoral footprint"),
        _p("fp.fem.width_mm", A, "Femoral footprint width", "scalar", 8.0, "mm", 3, 16, 0.5, ["L14"], sub="Femoral footprint"),
        _p("fp.fem.orientation_deg", A, "Femoral footprint orientation", "scalar", 0.0, "deg", -90, 90, 1,
           desc="Long axis angle from proximal (+Z in the femur frame) toward anterior, in the wall tangent plane.", sub="Femoral footprint"),
        _p("fp.fem.curvature_mm", A, "Femoral crescent radius", "scalar", 11.0, "mm", 5, 40, 0.5, ["L14"], sub="Femoral footprint"),
        _p("fp.fem.polygon", A, "Femoral footprint boundary", "polygon2", [], "mm", sub="Femoral footprint"),
        _p("fp.fem.am_share", A, "AM share of the femoral footprint", "scalar", 0.55, "", 0.2, 0.8, 0.01, ["L12"],
           "Area fraction of the femoral footprint attached to the AM bundle.", sub="Femoral footprint"),
        _p("fp.fem.partition_rotation_deg", A, "AM/PL split rotation on the femoral footprint", "scalar", 0.0, "deg", -90, 90, 1,
           desc="Rotates where the AM and PL subregions sit on the femoral footprint.", sub="Femoral footprint"),
        _p("lig.bow_anterior_mm", A, "Centreline bow (anterior +)", "scalar", 0.8, "mm", -6, 6, 0.1,
           desc="Mid-length offset of the centreline in the sagittal direction.", sub="Ligament"),
        _p("lig.bow_lateral_mm", A, "Centreline bow (lateral +)", "scalar", 0.0, "mm", -6, 6, 0.1, sub="Ligament"),
        _p("lig.ctrl_offsets", A, "Centreline control handles", "offsets", [[0, 0, 0], [0, 0, 0]], "mm",
           desc="Extra offsets of the control points at 1/3 and 2/3 length. Drag in the viewer.", sub="Ligament"),
        _p("lig.centerline_points", A, "Imported centreline", "points3", [], "mm",
           desc="Centreline samples from imaging (model frame). Empty = spline through the attachments.", sub="Ligament"),
        _p("lig.length_mode", A, "Ligament length", "enum", "from_attachments", options=["from_attachments", "target"],
           desc="from_attachments: length follows the attachments and centreline. target: attachments stay locked to bone and the anterior bow is solved to reach the target length.",
           sub="Ligament"),
        _p("lig.length_target_mm", A, "Target ligament length", "scalar", 33.0, "mm", 15, 60, 0.1, ["L01"],
           "Centreline length (tibial to femoral attachment). Only the anterior bow changes to reach it.", sub="Ligament"),
        _p("lig.section_mode", A, "Cross-section definition", "enum", "width_thickness",
           options=["width_thickness", "area_aspect", "per_bundle", "imported"],
           desc="width_thickness: width and thickness profiles (area derived). area_aspect: area and aspect profiles (width and thickness derived). "
                "per_bundle: AM and PL lobes with their own dimensions (envelope = their union). imported: measured contours.",
           sub="Ligament size and shape"),
        _p("lig.size_scale", A, "Section size scale", "scalar", 1.0, "x", 0.6, 1.5, 0.01,
           desc="Uniform scale of the cross-section (small/large templates). Does not change the length or the attachments.", sub="Ligament size and shape"),
        _p("lig.width_profile", A, "Width along ligament", "profile", [12.0, 10.5, 9.5, 10.5, 12.5], "mm", 3, 22, 0.1,
           ["L01", "L17"], "Section width from tibia (s=0) to femur (s=1), before the insertion flare.", sub="Ligament size and shape"),
        _p("lig.thickness_profile", A, "Thickness along ligament", "profile", [6.0, 5.6, 5.0, 4.3, 3.6], "mm", 1.5, 12, 0.1,
           ["L02", "L17"], "Ribbon-like flattening toward the femur (Smigielski).", sub="Ligament size and shape"),
        _p("lig.area_profile", A, "Cross-sectional area along ligament", "profile", [58.5, 46.2, 37.3, 37.7, 39.7], "mm2", 8, 160, 0.5,
           ["L01", "L17"], "Used in area_aspect mode; width and thickness are then derived from area, aspect and squareness.", sub="Ligament size and shape"),
        _p("lig.aspect_profile", A, "Aspect ratio (width / thickness)", "profile", [2.0, 1.88, 1.9, 2.44, 3.47], "", 1.0, 6.0, 0.01,
           desc="Used in area_aspect mode. 1 = round, larger = flatter.", sub="Ligament size and shape"),
        _p("lig.shape_profile", A, "Section squareness", "profile", [2.2, 2.0, 2.0, 2.4, 3.0], "", 1.5, 6, 0.1,
           desc="Superellipse exponent: 2 = ellipse, larger = flatter ribbon.", sub="Ligament size and shape"),
        _p("lig.orientation_profile", A, "Section orientation", "profile", [0.0, 0.0, 0.0, 0.0, 0.0], "deg", -90, 90, 1,
           desc="Rotation of the section's wide axis about the centreline, relative to the rotation-minimizing frame.", sub="Ligament size and shape"),
        _p("lig.taper_tib", A, "Distal (tibial) taper", "scalar", 0.0, "", -0.4, 0.6, 0.01,
           desc="Relative change of section size from mid-length to the tibial end (+0.2 = 20% larger at the tibia).", sub="Ligament size and shape"),
        _p("lig.taper_fem", A, "Proximal (femoral) taper", "scalar", 0.0, "", -0.4, 0.6, 0.01,
           desc="Relative change of section size from mid-length to the femoral end.", sub="Ligament size and shape"),
        _p("lig.contour_irregularity_mm", A, "Contour irregularity", "scalar", 0.0, "mm", 0, 1.5, 0.05,
           desc="Smooth radial bumps on the section outline (irregular cross-sections). Must stay below 35% of the half-thickness.", sub="Ligament size and shape"),
        _p("lig.contour_lobes", A, "Contour irregularity frequency", "int", 4, "lobes", 2, 9, 1, sub="Ligament size and shape"),
        _p("lig.contour_seed", A, "Contour irregularity seed", "int", 7, "", 0, 9999, 1, sub="Ligament size and shape"),
        _p("lig.section_contours", A, "Measured section contours", "contours", [], "mm",
           desc="Contours from imaging: [{s, points: [[x,y,z],...]}] in the model frame. Used in 'imported' section mode.", sub="Ligament size and shape"),
        _p("lig.flare_tib_mm", A, "Tibial transition length", "scalar", 6.0, "mm", 1, 14, 0.5, ["L04", "L19"],
           "Length over which the section blends into the tibial footprint (root-like flare).", sub="Ligament"),
        _p("lig.flare_fem_mm", A, "Femoral transition length", "scalar", 5.0, "mm", 1, 14, 0.5, ["L04", "L19"], sub="Ligament"),
        _p("lig.flare_exponent", A, "Flare profile exponent", "scalar", 1.0, "", 0.4, 3.0, 0.05,
           desc="Shape of the blend: <1 flares early, >1 flares close to bone.", sub="Ligament"),
    ]
    I = "injury"
    R += [
        _p("injury.type", I, "Injury type", "enum", "complete", options=["intact", "partial", "complete"], sub="Tear"),
        _p("injury.center_s", I, "Tear position (0 tibia, 1 femur)", "scalar", 0.55, "", 0.05, 0.95, 0.01, ["L18"], sub="Tear"),
        _p("injury.gap_mm", I, "Missing-tissue length", "scalar", 5.0, "mm", 0, 25, 0.25,
           desc="Longitudinal extent of missing tissue at the tear (before retraction).", sub="Tear"),
        _p("injury.retraction_tib_mm", I, "Tibial stump retraction", "scalar", 1.0, "mm", 0, 10, 0.25, sub="Tear"),
        _p("injury.retraction_fem_mm", I, "Femoral stump retraction", "scalar", 1.5, "mm", 0, 10, 0.25, sub="Tear"),
        _p("injury.obliquity_deg", I, "Tear-plane obliquity", "scalar", 12.0, "deg", 0, 70, 1,
           desc="Angle between the tear-plane normal and the local centreline tangent.", sub="Tear plane"),
        _p("injury.azimuth_deg", I, "Tear-plane azimuth", "scalar", 0.0, "deg", -180, 180, 1,
           desc="Direction of the tilt in the section, from the frame's anterior axis toward medial.", sub="Tear plane"),
        _p("injury.irregularity_mm", I, "Tear irregularity amplitude", "scalar", 0.8, "mm", 0, 4, 0.05, ["L18"], sub="Tear plane"),
        _p("injury.irregularity_lobes", I, "Tear irregularity frequency", "int", 5, "lobes", 1, 12, 1, sub="Tear plane"),
        _p("injury.seed", I, "Irregularity seed", "int", 3, "", 0, 9999, 1, sub="Tear plane"),
        _p("injury.involvement", I, "Affected region", "enum", "whole", options=["whole", "AM", "PL", "sector", "per_bundle"],
           desc="Which part of the cross-section is torn. Partial tears preserve the rest. per_bundle: each bundle has its own tear (location, length, depth).", sub="Affected region"),
        _p("injury.sector_center_deg", I, "Sector centre", "scalar", 45.0, "deg", -180, 180, 1, sub="Affected region"),
        _p("injury.sector_width_deg", I, "Sector width", "scalar", 140.0, "deg", 10, 360, 1, sub="Affected region"),
        _p("injury.depth_fraction", I, "Sector depth", "scalar", 1.0, "", 0.1, 1.0, 0.05,
           desc="Fraction of the radius, from the surface inward, that is torn inside the sector.", sub="Affected region"),
        _p("injury.continuity", I, "Residual continuity", "scalar", 0.0, "", 0, 1, 0.05,
           desc="Fraction of affected fascicles that still bridge the tear.", sub="Affected region"),
        _p("injury.damage_profile", I, "Regional damage estimate", "profile", [0.1, 0.2, 0.5, 0.3, 0.2], "", 0, 1, 0.05,
           desc="Damage score of RETAINED tissue along the ligament (colours remnants; no mechanics).", sub="Affected region"),
    ]
    for key, lab, cs, gap, dep in (("am", "AM", 0.55, 5.0, 1.0), ("pl", "PL", 0.45, 3.0, 0.0)):
        R += [
            _p("injury.%s.center_s" % key, I, "%s tear position" % lab, "scalar", cs, "", 0.05, 0.95, 0.01, sub="Per-bundle tears",
               desc="Used when the affected region is per_bundle (0 tibia, 1 femur)."),
            _p("injury.%s.gap_mm" % key, I, "%s missing-tissue length" % lab, "scalar", gap, "mm", 0, 25, 0.25, sub="Per-bundle tears"),
            _p("injury.%s.depth_fraction" % key, I, "%s tear depth" % lab, "scalar", dep, "", 0.0, 1.0, 0.05, sub="Per-bundle tears",
               desc="Fraction of the bundle's thickness torn, from its outer surface inward (0 = this bundle is intact, 1 = torn through)."),
        ]
    Fi = "fit"
    R += [
        _p("scaffold.engagement", Fi, "Engagement with remnants", "enum", "abut", options=["interdigitate", "abut"],
           desc="Interdigitate: aligned (and braid-core) yarns continue into the remnants between retained fascicles for the overlap length. "
                "Abut: every yarn ends at the local tear face. Braids always abut (a tube cannot pass between fascicles).", sub="Engagement"),
        _p("scaffold.overlap_tib_mm", Fi, "Overlap into tibial remnant", "scalar", 3.0, "mm", 0, 15, 0.25, sub="Engagement",
           desc="Length the interdigitating yarns run into the remnant past its deepest tear face (permitted interface overlap with retained tissue)."),
        _p("scaffold.overlap_fem_mm", Fi, "Overlap into femoral remnant", "scalar", 3.0, "mm", 0, 15, 0.25, sub="Engagement"),
        _p("scaffold.bone_reach_mm", Fi, "Anchor to bone if remnant shorter than", "scalar", 3.0, "mm", 0, 10, 0.25,
           desc="When a remnant is shorter than this, the repair extends to the footprint (attachment interface) and gets a bone-interface region.", sub="Engagement"),
        _p("fit.end_gap_mm", Fi, "Gap between yarn ends and tear face", "scalar", 0.25, "mm", 0, 2, 0.05,
           desc="Intended distance between abutting yarn ends and the local tear face.", sub="Engagement"),
        _p("fit.end_transition_mm", Fi, "End-transition length (bone anchoring)", "scalar", 6.0, "mm", 1, 14, 0.25,
           desc="Length over which a bone-anchored braid unbraids into a fan onto the footprint.", sub="Engagement"),
        _p("fit.partition", Fi, "Repair units", "enum", "single", options=["single", "per_bundle"],
           desc="single: one scaffold over the affected region. per_bundle: separate AM and PL scaffolds, each kept inside its own bundle region (no shared volume).",
           sub="Repair space"),
        _p("scaffold.fill_fraction", Fi, "Cross-sectional extent", "scalar", 0.85, "", 0.3, 1.0, 0.01,
           desc="Fraction of the affected section area that the scaffold may occupy (scaled about the region centroid, then inset by the clearances).", sub="Repair space"),
        _p("fit.boundary_clearance_mm", Fi, "Clearance from the ligament boundary", "scalar", 0.15, "mm", 0, 1.0, 0.01,
           desc="Minimum distance between any strand SURFACE and the target ligament boundary.", sub="Repair space"),
        _p("fit.interface_clearance_mm", Fi, "Clearance from the AM/PL interface", "scalar", 0.2, "mm", 0, 1.0, 0.01,
           desc="Minimum distance between strand surfaces and the AM/PL interface when only one bundle is repaired, and between AM and PL repair units.", sub="Repair space"),
        _p("fit.layer_spacing_mm", Fi, "Braid layer spacing", "scalar", 0.1, "mm", 0, 1.0, 0.01,
           desc="Gap between concentric braid layers (in addition to their over-under depth).", sub="Repair space"),
    ]
    for key, lab in (("am", "AM"), ("pl", "PL")):
        R += [
            _p("fit.%s.strength" % key, Fi, "%s fitting strength" % lab, "scalar", 1.0, "", 0, 1, 0.05, sub="%s repair unit" % lab,
               desc="1 = the scaffold section follows the local contour everywhere; 0 = a constant section that fits the narrowest station."),
            _p("fit.%s.overlap_tib_mm" % key, Fi, "%s overlap into tibial remnant" % lab, "scalar", 3.0, "mm", 0, 15, 0.25, sub="%s repair unit" % lab,
               desc="Used with per_bundle repair units."),
            _p("fit.%s.overlap_fem_mm" % key, Fi, "%s overlap into femoral remnant" % lab, "scalar", 3.0, "mm", 0, 15, 0.25, sub="%s repair unit" % lab),
        ]
    B = "bundles"
    R += [
        _p("bundle.am_fraction", B, "AM area fraction", "scalar", 0.55, "", 0.2, 0.8, 0.01, ["L12"]),
        _p("bundle.boundary_angle_deg", B, "AM/PL boundary direction at tibia", "scalar", 45.0, "deg", -180, 180, 1, ["L11", "L16"],
           "Direction (from the frame's anterior axis toward medial) pointing into the AM region."),
        _p("bundle.rotation_total_deg", B, "Bundle-region rotation (tibia to femur)", "scalar", 0.0, "deg", -120, 120, 1, ["L13", "L19"],
           "Rotation of the AM/PL arrangement about the ACL centreline (the parent level). Near 0 in extension (parallel bundles). "
           "Not limited by the +-30 deg fiber-rotation rule (that rule is about fibers within each unit); the within-bundle rotation "
           "is measured relative to the rotating bundle region, so the two are never added twice."),
        _p("bundle.rotation_distribution", B, "Rotation distribution", "enum", "uniform",
           options=["uniform", "femoral_weighted", "tibial_weighted", "custom"]),
        _p("bundle.rotation_profile", B, "Custom rotation profile", "profile", [0.0, 0.25, 0.5, 0.75, 1.0], "fraction", 0, 1, 0.01,
           desc="Cumulative fraction of the total rotation reached at each position (custom distribution)."),
        _p("bundle.am_fraction_mode", B, "AM contribution along the ligament", "enum", "constant", options=["constant", "profile"],
           desc="constant: the AM area fraction is the same at every section. profile: it varies along the ligament.", sub="AM/PL interface"),
        _p("bundle.am_fraction_profile", B, "AM area fraction along ligament", "profile", [0.55, 0.55, 0.55, 0.55, 0.55], "", 0.2, 0.8, 0.01,
           sub="AM/PL interface"),
        _p("bundle.boundary_angle_profile", B, "Interface angle offset along ligament", "profile", [0.0, 0.0, 0.0, 0.0, 0.0], "deg", -90, 90, 1,
           desc="Extra rotation of the AM/PL interface at each position (bundle position/orientation within the ACL).", sub="AM/PL interface"),
        _p("bundle.interface_curvature", B, "Interface curvature", "scalar", 0.0, "", -0.25, 0.25, 0.01,
           desc="Bulge of the AM/PL boundary as a fraction of its length (+ bulges into AM).", sub="AM/PL interface"),
        _p("bundle.interface_uncertainty_mm", B, "Interface uncertainty", "scalar", 1.0, "mm", 0, 3, 0.05,
           desc="Half-width of the band in which the true AM/PL boundary is expected (shown in views; boundaries are not sharply visible on MRI).", sub="AM/PL interface"),
        _p("bundle.lobe_overlap_mm", B, "Lobe contact depth", "scalar", 1.5, "mm", 0.2, 6, 0.05,
           desc="per_bundle section mode: how far the AM and PL lobes overlap at their interface (they always touch; no gap).", sub="AM/PL interface"),
        _p("bundle.structure_mode", B, "Per-bundle structure", "enum", "shared", options=["shared", "independent"],
           desc="shared: both bundles use the common fascicle/fiber settings. independent: each bundle has its own.", sub="AM/PL interface"),
    ]
    for key, lab, w, t, sh, o, fd in (("am", "AM", [6.5, 5.6, 5.2, 5.6, 6.5], [6.0, 5.6, 5.0, 4.3, 3.6], 0.0, 0.0, 1.0),
                                     ("pl", "PL", [6.0, 5.0, 4.6, 5.0, 5.8], [5.6, 5.2, 4.6, 3.9, 3.2], 0.0, 0.0, 1.0)):
        sub = "%s bundle" % lab
        R += [
            _p("%s.width_profile" % key, B, "%s lobe width" % lab, "profile", w, "mm", 1.5, 16, 0.1, sub=sub,
               desc="per_bundle section mode: lobe extent along the section's wide axis. The lobes are stacked across the AM/PL interface "
                    "(boundary angle 0 = side by side along the wide axis)."),
            _p("%s.thickness_profile" % key, B, "%s lobe thickness" % lab, "profile", t, "mm", 1.0, 12, 0.1, sub=sub),
            _p("%s.shape" % key, B, "%s lobe squareness" % lab, "scalar", 2.0, "", 1.5, 5, 0.1, sub=sub),
            _p("%s.orientation_deg" % key, B, "%s lobe orientation" % lab, "scalar", o, "deg", -90, 90, 1, sub=sub,
               desc="Rotation of the lobe's wide axis relative to the section's wide axis."),
            _p("%s.shift_profile" % key, B, "%s lobe shift along the interface" % lab, "profile", [sh] * 5, "mm", -4, 4, 0.1, sub=sub,
               desc="Moves the lobe along the interface (trajectory of the bundle within the ACL)."),
            _p("%s.fascicle_diameter_scale" % key, B, "%s fascicle diameter scale" % lab, "scalar", fd, "x", 0.5, 2.0, 0.01, sub=sub,
               desc="independent structure: multiplies the fascicle diameter in this bundle; the count follows from the packing fraction."),
            _rot("%s.fascicle_rotation_deg" % key, B, sub, "the %s bundle (its fascicles turn about the bundle path, relative to the bundle region; "
                 "independent structure). 0 = parallel, as in extension (Parrilli 2024)" % lab, ["L19"]),
            _rot("%s.fiber_rotation_deg" % key, B, sub, "each fascicle group of the %s bundle (its fibers turn about the fascicle axis; independent structure)" % lab),
            _p("%s.dispersion_deg" % key, B, "%s orientation dispersion" % lab, "scalar", 4.0, "deg", 0, 25, 0.5, sub=sub),
            _p("%s.flare_tib_mm" % key, B, "%s tibial transition length" % lab, "scalar", 6.0, "mm", 1, 14, 0.5, sub=sub),
            _p("%s.flare_fem_mm" % key, B, "%s femoral transition length" % lab, "scalar", 5.0, "mm", 1, 14, 0.5, sub=sub),
        ]
    Fb = "fibers"
    R += [
        _p("native.fascicle_count", Fb, "Fascicle groups", "int", 36, "count", 4, 160, 1,
           desc="Representative fascicle groups across the section (each stands for many fascicles).", sub="Native fascicles"),
        _p("native.sizing_mode", Fb, "Fascicle sizing: independent pair", "enum", "count+packing",
           options=["count+packing", "count+diameter", "diameter+packing"],
           desc="Count, diameter and packing fraction are linked (packing = N*pi*d^2/4 / A). Choose two; the third is computed.",
           sub="Native fascicles"),
        _p("native.fascicle_packing", Fb, "Fascicle packing fraction", "scalar", 0.50, "", 0.1, 0.9, 0.01, sub="Native fascicles"),
        _p("native.fascicle_diameter_mm", Fb, "Fascicle group diameter", "scalar", 1.4, "mm", 0.2, 5.0, 0.05, sub="Native fascicles"),
        _p("native.packing_pattern", Fb, "Packing pattern", "enum", "hexagonal", options=["hexagonal", "random", "radial"], sub="Native fascicles"),
        _p("native.layout_seed", Fb, "Layout seed", "int", 1, "", 0, 9999, 1, sub="Native fascicles"),
        _p("native.fiber_diameter_um", Fb, "Collagen fiber diameter", "scalar", 8.0, "um", 1, 50, 0.5, ["L07"], sub="Native fibers"),
        _p("native.fiber_packing", Fb, "Fiber packing fraction", "scalar", 0.60, "", 0.1, 0.85, 0.01, sub="Native fibers"),
        _p("native.dispersion_deg", Fb, "Orientation dispersion", "scalar", 4.0, "deg", 0, 25, 0.5, sub="Native fibers"),
        _p("native.dispersion_profile", Fb, "Dispersion along length (x)", "profile", [1.5, 1.0, 1.0, 1.0, 1.5], "", 0, 3, 0.05, sub="Native fibers"),
        _p("native.crimp_amplitude_um", Fb, "Crimp amplitude", "scalar", 4.0, "um", 0, 100, 1, ["L07"], sub="Native fibers"),
        _p("native.crimp_wavelength_um", Fb, "Crimp wavelength", "scalar", 80.0, "um", 10, 500, 1, ["L07"], sub="Native fibers"),
        _p("native.crimp_profile", Fb, "Crimp along length (x)", "profile", [1.3, 1.0, 1.0, 1.0, 1.3], "", 0, 3, 0.05, sub="Native fibers"),
        _p("scaffold.fiber_diameter_um", Fb, "Scaffold fiber diameter", "scalar", 20.0, "um", 5, 300, 1, ["L05"], sub="Scaffold fibers"),
        _p("scaffold.crimp_amplitude_um", Fb, "Scaffold fiber waviness amplitude", "scalar", 0.0, "um", 0, 200, 1, sub="Scaffold fibers"),
        _p("scaffold.crimp_wavelength_um", Fb, "Scaffold fiber waviness wavelength", "scalar", 400.0, "um", 20, 2000, 5, sub="Scaffold fibers"),
    ]
    T = "twist"
    R += [
        _rot("native.fascicle_rotation_deg", T, "Native bundles (unit: AM or PL bundle)",
             "each native bundle (its fascicles and their fibers turn about the bundle path, relative to the bundle region). "
             "0 = parallel bundles, as in knee extension (Parrilli 2024, Fig. 4)", ["L19"]),
        _rot("native.fiber_rotation_deg", T, "Native fascicles (unit: fascicle group)", "each native fascicle group (its fibers turn about the fascicle axis)"),
        _p("scaffold.arch", T, "Scaffold construct", "enum", "layered", options=["layered", "braided", "aligned"],
           desc="layered: the six-level engineered construct (hollow channels, braided AM-like/PL-like mantles, ribbon sheath, "
                "flared graded ends) spanning footprint to footprint. braided / aligned: the earlier gap-filling yarn scaffold "
                "fitted inside the repair space (tubular braid with over-under crossings, or fascicle-like yarns).", sub="Scaffold"),
        _p("scaffold.yarn_count", T, "Yarns / braid carriers", "int", 16, "count", 4, 96, 1, ["L09"], sub="Scaffold",
           desc="Braid: carriers per layer (half in each direction). Aligned: yarn count (count sizing)."),
        _p("scaffold.yarn_diameter_mm", T, "Base yarn diameter", "scalar", 0.5, "mm", 0.1, 2.0, 0.01, ["L06"],
           "Regions scale this with their strand-scale setting.", sub="Scaffold"),
        _rot("scaffold.yarn_rotation_deg", T, "Scaffold repair unit (aligned yarns)",
             "each aligned-yarn repair unit (its yarns, and their fibers, turn about the unit's axis; any twist the unit would "
             "inherit from the ligament's shape is removed, so the total equals this value)"),
        _p("scaffold.braid_mode", T, "Braid: independent variable", "enum", "angle", options=["angle", "pitch"],
           desc="Braid angle and pitch are linked through the braid radius: tan(angle) = 2*pi*R / pitch.", sub="Scaffold"),
        _p("scaffold.braid_angle_deg", T, "Braid angle", "scalar", 25.0, "deg", 5, 60, 0.5, ["L06", "L09"],
           "Angle between yarn and scaffold axis.", sub="Scaffold"),
        _p("scaffold.braid_pitch_mm", T, "Braid pitch", "scalar", 12.0, "mm", 2, 80, 0.5, sub="Scaffold"),
        _p("scaffold.braid_pattern", T, "Over-under pattern", "enum", "1/1", options=["1/1", "2/2"], sub="Scaffold"),
        _p("scaffold.braid_radius_fraction", T, "Braid size (fraction of the fitted space)", "scalar", 1.0, "", 0.3, 1.0, 0.01, sub="Scaffold",
           desc="1 = the outer braid layer sits at the clearance from the repair-space boundary; smaller values shrink it toward the region centroid."),
        _p("scaffold.braid_layers", T, "Braid layers", "int", 1, "count", 1, 3, 1, sub="Scaffold",
           desc="Concentric braid layers, each fitted inside the previous one (spacing set under Repair space)."),
        _p("scaffold.core_mode", T, "Core filling", "enum", "max", options=["count", "max"],
           desc="count: the given number of axial core yarns. max: as many core yarns as fit inside the innermost braid layer.", sub="Scaffold"),
        _p("scaffold.core_yarns", T, "Axial core yarns", "int", 12, "count", 0, 60, 1, sub="Scaffold",
           desc="Aligned yarns inside the braid. With 'interdigitate' they continue into the remnants between retained fascicles."),
        _p("scaffold.aligned_sizing", T, "Aligned yarns: set by", "enum", "count", options=["count", "packing"],
           desc="count: the yarn count is given. packing: the count follows from a target strand packing fraction of the repair space.", sub="Scaffold"),
        _p("scaffold.packing_fraction", T, "Aligned strand packing fraction", "scalar", 0.35, "", 0.05, 0.8, 0.01, sub="Scaffold",
           desc="Strand area / repair-space area at the narrowest station (aligned architecture, packing sizing)."),
        _p("scaffold.clearance_mm", T, "Minimum yarn clearance", "scalar", 0.03, "mm", 0, 0.5, 0.01, sub="Scaffold"),
        _rot("scaffold.fiber_rotation_deg", T, "Scaffold yarns (unit: yarn)", "each scaffold yarn (its fibers turn about the yarn axis)"),
    ]
    M = "materials"
    region_defaults = {
        "tibial_interface": ("PLGA", {"PLGA": 0.7, "collagen_I": 0.3}, 12.0, 0.45, 250.0, 1.0, 2.0, "graded"),
        "bridge": ("PLLA", {"PLLA": 0.8, "PCL": 0.2}, 4.0, 0.60, 180.0, 1.15, 0.0, "graded"),
        "femoral_interface": ("PLGA", {"PLGA": 0.7, "collagen_I": 0.3}, 12.0, 0.45, 250.0, 1.0, 2.0, "graded"),
    }
    for rid, (mat, comp, disp, vf, pore, scale, tw, mode) in region_defaults.items():
        lab = rid.replace("_", " ")
        R += [
            _p("regions.%s.material" % rid, M, "Primary material", "enum", mat, options=list(MATERIALS), sub=lab,
               desc="Metadata: sets colour and export labels. No material model is evaluated."),
            dict(_p("regions.%s.composition" % rid, M, "Composition fractions", "composition", comp, "fraction", sub=lab,
                    desc="Normalized to 1. Metadata only."), materials=STRUCTURAL_MATERIALS),
            _p("regions.%s.fiber_volume_fraction" % rid, M, "Fiber volume fraction (density)", "scalar", vf, "", 0.1, 0.85, 0.01, ["L05"], sub=lab,
               desc="Fraction of yarn volume occupied by fibers. Porosity = 1 - this value."),
            _p("regions.%s.alignment_dispersion_deg" % rid, M, "Fiber alignment dispersion", "scalar", disp, "deg", 0, 30, 0.5, sub=lab),
            _p("regions.%s.strand_scale" % rid, M, "Strand diameter scale", "scalar", scale, "x", 0.4, 2.0, 0.01, sub=lab,
               desc="Multiplies the base yarn diameter inside this region."),
            _p("regions.%s.transition_mm" % rid, M, "Transition width toward bridge", "scalar", tw if rid != "bridge" else 0.0, "mm", 0, 8, 0.25, sub=lab,
               desc="Width of the graded zone between this region and the bridge (0 = abrupt)."),
            _p("regions.%s.transition_mode" % rid, M, "Transition", "enum", mode, options=["graded", "abrupt"], sub=lab),
        ]
    R += _layer_registry()
    _apply_group_placement(R)
    return R


# ---------------------------------------------------------------- six-level construct controls
EXP = "experimental"
_N_CH = [3, "the channels' lumens carry it"]
_N_SH = [5, "the inner sheath surface carries it when selected"]


def _layer_registry():
    R = []
    # ---------------- Level 1 — Surface
    C1 = "Surface treatment on designated cell-facing surfaces (channel lumens, inner sheath)"
    R += [
        _L(1, "sf.location", "Location and coverage", "Treated surfaces", "enum", "channels",
           options=["channels", "inner_sheath", "channels_and_inner_sheath", "none"], component=C1, neighbors=[_N_CH, _N_SH], basis=EXP,
           desc="Which cell-facing surfaces receive the nano-texture and coating. The outer sheath is never treated here."),
        _L(1, "sf.coverage_start", "Location and coverage", "Coverage starts at (0 tibia, 1 femur)", "scalar", 0.0, "", 0, 1, 0.01,
           component=C1, neighbors=[_N_CH], basis=EXP, desc="Fraction of the construct length where the treatment begins."),
        _L(1, "sf.coverage_end", "Location and coverage", "Coverage ends at (0 tibia, 1 femur)", "scalar", 1.0, "", 0, 1, 0.01,
           component=C1, neighbors=[_N_CH], basis=EXP),
        _L(1, "sf.texture", "Nano-texture", "Texture pattern", "enum", "ridges", options=["ridges", "pillars", "none"],
           component="Nano-texture on the treated surfaces", neighbors=[[2, "the texture rides on groove and ridge surfaces"]], basis=EXP),
        _L(1, "sf.texture_orientation_deg", "Nano-texture", "Texture orientation to the local channel direction", "scalar", 0.0, "deg", -90, 90, 1,
           component="Nano-texture on the treated surfaces", neighbors=[[2, "compare with the groove orientation"]], basis=EXP,
           desc="0 = ridges run along the channel (local fiber path); 90 = across it."),
        _L(1, "sf.texture_period_nm", "Nano-texture", "Texture scale (period)", "scalar", 400.0, "nm", 50, 5000, 10,
           component="Nano-texture on the treated surfaces", basis=EXP),
        _L(1, "sf.texture_amplitude_nm", "Nano-texture", "Texture amplitude (relief height)", "scalar", 100.0, "nm", 5, 2000, 1,
           component="Nano-texture on the treated surfaces", neighbors=[[1, "a coating thicker than the relief may fill it"]], basis=EXP),
        _L(1, "sf.coating_nm", "Protein coating", "Coating thickness", "scalar", 40.0, "nm", 0, 20000, 1,
           component="Coating layer on the treated surfaces", neighbors=[[3, "reduces the available lumen when in explicit geometry"]], basis=EXP),
        _L(1, "sf.coating_explicit", "Protein coating", "Coating in the geometry", "enum", "explicit", options=["explicit", "metadata"],
           component="Coating layer on the treated surfaces", neighbors=[[3, "explicit: the lumen is narrowed by 2 x thickness"]], basis=EXP,
           desc="explicit: the coating is modelled as a layer inside the lumen (the open lumen shrinks). metadata: recorded only."),
        _L(1, "sf.coating_composition", "Protein coating", "Coating composition", "composition", {"collagen_I": 0.7, "fibronectin": 0.3},
           "fraction", component="Coating layer on the treated surfaces", basis=EXP, materials=COATING_MATERIALS,
           desc="Fractions must total 1. Recorded as design metadata; no biological effect is modelled."),
    ]
    # ---------------- Level 2 — Micro
    C2 = "Grooves cut into the selected guidance surfaces"
    nd = [[3, "groove depth must leave a minimum channel wall"], [5, "groove depth must leave a minimum sheath wall"]]
    R += [
        _L(2, "gr.placement", "Placement", "Grooved surfaces", "enum", "channels", options=["channels", "inner_sheath", "both", "none"],
           component=C2, neighbors=[_N_CH, _N_SH], basis=EXP,
           desc="Channel interiors, the inner sheath, or both. Braid strands are load-bearing and are never grooved."),
        _L(2, "gr.coverage", "Placement", "Surface coverage (centred along the length)", "scalar", 0.8, "", 0.1, 1.0, 0.01,
           component=C2, basis=EXP),
        _L(2, "gr.outer_sheath", "Placement", "Outer sheath surface", "enum", "smooth", options=["smooth", "grooved"],
           component="Outer sheath surface", neighbors=[[5, "the outer sheath faces the notch and synovium"]], basis=EXP,
           desc="Smooth by default. Grooving the outer sheath is an explicit experimental choice."),
        _L(2, "gr.width_um", "Groove profile", "Groove width", "scalar", 10.0, "um", 1, 100, 0.5, component=C2, basis=EXP),
        _L(2, "gr.depth_um", "Groove profile", "Groove depth", "scalar", 5.0, "um", 0.5, 100, 0.5, component=C2, neighbors=nd, basis=EXP),
        _L(2, "gr.ridge_um", "Groove profile", "Ridge width (groove spacing)", "scalar", 10.0, "um", 1, 200, 0.5, component=C2, basis=EXP,
           desc="Flat land between neighbouring grooves; groove period = width + ridge."),
        _L(2, "gr.bottom_radius_um", "Groove profile", "Groove-bottom radius", "scalar", 2.0, "um", 0, 50, 0.25, component=C2, basis=EXP,
           desc="Rounded groove bottom; at most half the groove width."),
        _L(2, "gr.orientation_deg", "Groove profile", "Groove orientation to the local fiber path", "scalar", 0.0, "deg", -80, 80, 1,
           component=C2, neighbors=[[1, "compare with the texture orientation"]], basis=EXP,
           desc="0 = grooves follow the channel axis (the local fiber path)."),
        _L(2, "gr.min_residual_um", "Groove profile", "Minimum wall left under a groove", "scalar", 15.0, "um", 1, 100, 1,
           component=C2, neighbors=nd, desc="Design rule that keeps grooves from cutting through walls."),
    ]
    # ---------------- Level 3 — Fascicle
    C3 = "Hollow core channels"
    nl4 = [4, "the reinforcement envelope must hold the channel group"]
    R += [
        _L(3, "ch.lumen_um", "Channel size", "Lumen diameter (width for non-circular)", "scalar", 200.0, "um", 50, 600, 5,
           component=C3, neighbors=[[1, "coating narrows the open lumen"], [2, "the groove period must fit the lumen"], nl4], basis=EXP,
           desc="Open inner diameter. Proposed range about 150-300 um: an adjustable proposal, not a validated anatomical requirement."),
        _L(3, "ch.wall_um", "Channel size", "Channel wall thickness", "scalar", 50.0, "um", 10, 300, 1,
           component="Channel walls", neighbors=[[2, "groove depth must leave a minimum wall"], nl4], basis=EXP,
           desc="Outside diameter = lumen + 2 x wall (shown as a calculated value)."),
        _L(3, "ch.spacing_um", "Count and spacing", "Spacing between channel walls", "scalar", 80.0, "um", 10, 1000, 5,
           component=C3, neighbors=[nl4], basis=EXP,
           desc="Edge-to-edge gap between neighbouring outside walls (and to the reinforcement). Pitch = outside diameter + spacing."),
        _L(3, "ch.count_mode", "Count and spacing", "Channel count", "enum", "max_fit", options=["max_fit", "count"],
           component=C3, neighbors=[nl4], basis=EXP, desc="max_fit: as many as fit at the pitch (calculated). count: the numbers below."),
        _L(3, "ch.am_count", "Count and spacing", "Channels in the AM-like region", "int", 40, "count", 0, 600, 1,
           component="AM-like channel group", neighbors=[nl4], basis=EXP, region="am"),
        _L(3, "ch.pl_count", "Count and spacing", "Channels in the PL-like region", "int", 32, "count", 0, 600, 1,
           component="PL-like channel group", neighbors=[nl4], basis=EXP, region="pl"),
        _L(3, "ch.transition_channels", "Count and spacing", "Shared channels in the AM/PL transition", "enum", "fill", options=["fill", "none"],
           component="Shared (transition) channels", neighbors=[[4, "need free space in the transition band"]], basis=EXP,
           desc="fill: channels also run in the band between the mantles, so the channel field is continuous across AM-like and PL-like regions."),
        _L(3, "ch.shape", "Shape and path", "Cross-sectional shape", "enum", "circular", options=["circular", "elliptical", "rounded_square"],
           component=C3, neighbors=[[2, "groove layout around the lumen"], nl4], basis=EXP),
        _L(3, "ch.aspect", "Shape and path", "Width / height (non-circular)", "scalar", 1.5, "", 1.0, 3.0, 0.05,
           component=C3, neighbors=[nl4], basis=EXP),
        _L(3, "ch.waviness_um", "Shape and path", "Waviness amplitude", "scalar", 0.0, "um", 0, 300, 5,
           component="Channel paths", neighbors=[[4, "uses clearance inside the envelope; fewer channels fit"]], basis=EXP),
        _L(3, "ch.wavelength_mm", "Shape and path", "Waviness wavelength", "scalar", 4.0, "mm", 0.5, 20, 0.1,
           component="Channel paths", basis=EXP),
        _L(3, "ch.curvature_um", "Shape and path", "Channel curvature (mid-length bow)", "scalar", 0.0, "um", -400, 400, 5,
           component="Channel paths", neighbors=[[4, "uses clearance inside the envelope"]], basis=EXP,
           desc="Single arc of the channel group within its host, zero at both ends."),
        _L(3, "ch.connections", "Connections", "Connections between channels", "enum", "none", options=["none", "ports"],
           component="Channel connections", basis=EXP, desc="ports: short hollow cross-connections between nearest neighbours."),
        _L(3, "ch.port_spacing_mm", "Connections", "Port spacing along the channels", "scalar", 3.0, "mm", 0.5, 15, 0.1,
           component="Channel connections", basis=EXP),
        _L(3, "ch.port_diameter_um", "Connections", "Port lumen diameter", "scalar", 60.0, "um", 10, 300, 5,
           component="Channel connections", basis=EXP),
        _L(3, "ch.wall_composition", "Channel-wall material", "Channel-wall composition", "composition", {"PLGA": 0.5, "PCL": 0.5},
           "fraction", component="Channel walls", neighbors=[[6, "blends into the graded end composition near bone"]],
           materials=STRUCTURAL_MATERIALS, desc="Fractions must total 1. Design metadata."),
        _L(3, "ch.assumed_degradation_weeks", "Assumed properties (entered, not calculated)", "Channel-wall degradation half-life (assumed)",
           "scalar", 26.0, "weeks", 1, 260, 1, component="Channel walls",
           desc="Entered assumption. Used only for the labelled 'remaining at 12 weeks' estimate; no degradation model is evaluated."),
    ]
    # ---------------- Level 4 — Bundle
    for key, lab, W, T in (("am", "AM-like", 4.5, 1.6), ("pl", "PL-like", 3.9, 1.25)):
        other = "pl" if key == "am" else "am"
        C4 = "%s braided mantle" % lab
        n35 = [[3, "%s channels pack inside the envelope" % lab], [5, "the braid must keep the sheath clearance"]]
        R += [
            _L(4, "rf.%s.strands" % key, "Braid", "Braid strand count", "int", 16, "count", 8, 64, 4, component=C4,
               neighbors=[[5, "more strands at a fixed perimeter can jam"]], region=key, pair="rf.%s.strands" % other,
               desc="Total strands, half in each direction. 1/1 needs a multiple of 4, 2/2 a multiple of 8."),
            _L(4, "rf.%s.strand_diameter_mm" % key, "Braid", "Strand diameter", "scalar", 0.15, "mm", 0.05, 0.6, 0.01, component=C4,
               neighbors=n35, region=key, pair="rf.%s.strand_diameter_mm" % other,
               desc="Braid wall = 2 x strand diameter (over/under families). Thicker braids need more sheath clearance."),
            _L(4, "rf.%s.braid_angle_deg" % key, "Braid", "Braid angle", "scalar", 25.0, "deg", 5, 60, 0.5, component=C4,
               neighbors=[[5, "steeper angles reduce the strand gap and can jam"]], region=key, pair="rf.%s.braid_angle_deg" % other,
               desc="Angle between a strand and the mantle axis. Independent of yarn twist and of the ligament's bundle rotation."),
            _L(4, "rf.%s.pattern" % key, "Braid", "Braid pattern", "enum", "1/1", options=["1/1", "2/2"], component=C4,
               region=key, pair="rf.%s.pattern" % other),
            _L(4, "rf.%s.yarn_twist_deg" % key, "Braid", "Yarn twist: total fiber rotation along each strand", "scalar", 0.0, "deg", -30, 30, 0.5,
               component="%s strands (fibers inside each strand)" % lab, region=key, pair="rf.%s.yarn_twist_deg" % other,
               desc=ROT_DESC + " Unit: each braid strand. Independent of the braid angle and of the overall ligament rotation."),
            _L(4, "rf.%s.envelope_width_mm" % key, "Reinforcement envelope", "Envelope width (inside the braid, mid-length)", "scalar", W, "mm",
               0.5, 12, 0.05, component=C4, neighbors=n35, region=key, pair="rf.%s.envelope_width_mm" % other,
               desc="Used with manual sizing: sets the mid-length size; along the length the envelope keeps the fitted shape."),
            _L(4, "rf.%s.envelope_thickness_mm" % key, "Reinforcement envelope", "Envelope thickness (inside the braid, mid-length)", "scalar", T,
               "mm", 0.5, 10, 0.05, component=C4, neighbors=n35, region=key, pair="rf.%s.envelope_thickness_mm" % other),
            _L(4, "rf.%s.slack_pct" % key, "Reinforcement envelope", "Regional reference length (slack)", "scalar", 0.0, "%", 0, 8, 0.1,
               component=C4, neighbors=[[3, "the channel group undulates with its mantle"], [5, "the excursion needs sheath clearance"]],
               region=key, pair="rf.%s.slack_pct" % other,
               desc="Extra path length of this region relative to the construct (crimp of the whole mantle). The region carries load "
                    "only after this strain is taken up."),
            _L(4, "rf.%s.composition" % key, "Reinforcement material", "Reinforcement composition", "composition", {"PLLA": 0.85, "PCL": 0.15},
               "fraction", component="%s strands" % lab, neighbors=[[6, "blends into the graded end composition near bone"]],
               region=key, pair="rf.%s.composition" % other, materials=STRUCTURAL_MATERIALS, desc="Fractions must total 1."),
            _L(4, "rf.%s.assumed_modulus_mpa" % key, "Assumed properties (entered, not calculated)", "Strand material modulus (assumed)", "scalar",
               3000.0, "MPa", 10, 10000, 10, component="%s strands" % lab, region=key, pair="rf.%s.assumed_modulus_mpa" % other,
               desc="Entered assumption, used only for the labelled first-order stiffness estimate."),
        ]
    R += [
        _L(4, "rf.size_mode", "Reinforcement envelope", "Envelope sizing", "enum", "auto", options=["auto", "manual"],
           component="Both braided mantles", neighbors=[[3, "envelope size sets how many channels fit"], [5, "auto keeps the sheath clearance"]],
           desc="auto: each envelope is the largest that keeps the sheath clearance and the transition band (calculated). manual: the widths and thicknesses below."),
        _L(4, "rf.transition_mm", "Reinforcement envelope", "AM/PL transition width", "scalar", 0.6, "mm", 0, 3, 0.05,
           component="Band between the AM-dominant and PL-dominant mantles",
           neighbors=[[3, "shared channels fill the band"], [5, "the regions stay inside one continuous sheath"]],
           desc="Width of the band around the AM/PL interface kept free of braid. No wall is placed there: the scaffold stays continuous."),
    ]
    # ---------------- Level 5 — Whole ligament
    C5 = "Outer sheath (shared envelope)"
    n46 = [[4, "the mantles must fit inside with clearance"], [6, "each flare starts from the sheath section"]]
    R += [
        _L(5, "sheath.size_mode", "Overall shape", "Sheath sizing", "enum", "fit_anatomy", options=["fit_anatomy", "profiles", "fit_contents"],
           component=C5, neighbors=n46,
           desc="fit_anatomy: a fraction of the intact envelope (calculated). profiles: the width/thickness curves. fit_contents: the smallest "
                "sheath that holds both mantles with clearance (calculated)."),
        _L(5, "sheath.anatomy_fraction", "Overall shape", "Fraction of the anatomical envelope", "scalar", 0.96, "x", 0.6, 1.1, 0.01,
           component=C5, neighbors=n46),
        _L(5, "sheath.width_profile", "Overall shape", "Width along the length", "profile", [11.5, 10.1, 9.1, 10.1, 12.0], "mm", 3, 22, 0.1,
           component=C5, neighbors=n46, desc="Outer width from tibia (s=0) to femur (s=1); used with profile sizing."),
        _L(5, "sheath.thickness_profile", "Overall shape", "Thickness along the length", "profile", [5.75, 5.4, 4.8, 4.15, 3.45], "mm", 1.5, 12, 0.1,
           component=C5, neighbors=n46),
        _L(5, "sheath.flattening_profile", "Overall shape", "Cross-section flattening along the length", "profile", [2.2, 2.0, 2.0, 2.4, 3.0], "",
           1.6, 6, 0.05, component=C5, neighbors=n46, desc="Superellipse exponent: 2 = elliptical, larger = flatter ribbon."),
        _L(5, "sheath.orientation_profile", "Overall shape", "Cross-section orientation along the length", "profile", [0.0] * 5, "deg", -45, 45, 1,
           component=C5, neighbors=[[4, "turning the ribbon against the bundle arrangement can squeeze a mantle"], [6, "changes the flare twist"]],
           desc="Rotation of the ribbon's wide axis relative to the anatomical section. It does not impose crossing of the AM-like and PL-like regions: "
                "their paths follow the attachments and the knee pose."),
        _L(5, "sheath.wall_mm", "Wall and clearance", "Sheath wall thickness", "scalar", 0.25, "mm", 0.05, 1.5, 0.01,
           component="Sheath wall", neighbors=[[2, "inner-sheath grooves must leave a minimum wall"], [4, "less inner space for the mantles"]]),
        _L(5, "sheath.clearance_mm", "Wall and clearance", "Clearance around internal structures", "scalar", 0.08, "mm", 0, 1.0, 0.01,
           component="Gap between sheath and mantles", neighbors=[[4, "the braids must keep this gap"]]),
        _L(5, "sheath.composition", "Sheath material", "Sheath composition", "composition", {"PCL": 0.6, "PLGA": 0.4}, "fraction",
           component="Sheath wall", neighbors=[[6, "blends into the graded end composition near bone"]], materials=STRUCTURAL_MATERIALS,
           desc="Fractions must total 1."),
        _L(5, "sheath.pores", "Explicit pores (geometry)", "Pores in the sheath wall", "enum", "none", options=["none", "explicit"],
           component="Sheath wall pores", neighbors=[[2, "inner-sheath grooves and pores share the wall"]], basis=EXP,
           desc="Geometric through-wall openings. Drawing pores does not imply selective fluid transport."),
        _L(5, "sheath.pore_diameter_um", "Explicit pores (geometry)", "Pore diameter", "scalar", 150.0, "um", 20, 800, 5,
           component="Sheath wall pores", basis=EXP),
        _L(5, "sheath.pore_spacing_mm", "Explicit pores (geometry)", "Pore centre spacing", "scalar", 0.8, "mm", 0.1, 3, 0.05,
           component="Sheath wall pores", basis=EXP),
        _L(5, "sheath.pore_pattern", "Explicit pores (geometry)", "Pore distribution", "enum", "staggered", options=["staggered", "square"],
           component="Sheath wall pores", basis=EXP),
        _L(5, "sheath.pore_coverage", "Explicit pores (geometry)", "Pore coverage (centred on the body)", "scalar", 0.8, "", 0.1, 1.0, 0.01,
           component="Sheath wall pores", basis=EXP),
        _L(5, "sheath.assumed_permeability_log10", "Assumed properties (entered, not calculated)", "Wall permeability, log10 (assumed)", "scalar",
           -15.0, "log10 m2", -20, -9, 0.1, component="Sheath wall",
           desc="Entered assumption, independent of the drawn pores. Used only for a labelled Darcy estimate; no transport model."),
        _L(5, "sheath.assumed_degradation_weeks", "Assumed properties (entered, not calculated)", "Sheath degradation half-life (assumed)", "scalar",
           52.0, "weeks", 1, 260, 1, component="Sheath wall"),
    ]
    # ---------------- Level 6 — Ends
    for side, lab, Lfp, Wfp, Lf, Wf, Tf in (("tib", "Tibial", 17.0, 11.0, 6.0, 14.0, 9.0), ("fem", "Femoral", 16.0, 8.0, 5.0, 12.0, 6.0)):
        other = "fem" if side == "tib" else "tib"
        k = "end.%s." % side
        C6 = "%s flared end" % lab
        CF = "%s scaffold footprint" % lab
        nprop = [[3, "channel endpoints move with it"], [4, "braid endpoints move with it"], [5, "the flared sheath follows it"]]
        ncomp = [[3, "graded composition of the channel walls near bone"], [4, "and of the braid"], [5, "and of the sheath"]]

        def E(name, sub, label, kind, default, units="", lo=None, hi=None, step=None, desc="", component=C6, neighbors=nprop, **kw):
            return _L(6, k + name, sub, label, kind, default, units, lo, hi, step, desc, component, neighbors, region=side, pair="end.%s.%s" % (other, name), **kw)
        R += [
            E("offset_long_mm", "Footprint", "Footprint position along its long axis", "scalar", 0.0, "mm", -6, 6, 0.1, component=CF,
              desc="Offset of the scaffold footprint from the anatomical footprint (Anatomy and Pose), along its long axis; projected onto bone."),
            E("offset_short_mm", "Footprint", "Footprint position across its long axis", "scalar", 0.0, "mm", -6, 6, 0.1, component=CF),
            E("rotation_deg", "Footprint", "Footprint orientation", "scalar", 0.0, "deg", -90, 90, 1, component=CF,
              desc="Rotation of the scaffold footprint about the bone normal, relative to the anatomical footprint."),
            E("outline", "Footprint", "Footprint outline", "enum", "anatomical", options=["anatomical", "oval", "rounded_rect"], component=CF,
              desc="anatomical: the shape from Anatomy and Pose (imaging, manual placement or template)."),
            E("size_mode", "Footprint", "Footprint size", "enum", "match_anatomy", options=["match_anatomy", "manual"], component=CF,
              desc="match_anatomy: length and width of the anatomical footprint (calculated). manual: the values below."),
            E("length_mm", "Footprint", "Footprint length", "scalar", Lfp, "mm", 4, 30, 0.5, component=CF),
            E("width_mm", "Footprint", "Footprint width", "scalar", Wfp, "mm", 3, 20, 0.5, component=CF),
            E("flare_length_mm", "Flare", "Flare length", "scalar", Lf, "mm", 1, 14, 0.25,
              desc="Distance from bone over which the sheath section blends onto the footprint."),
            E("flare_mode", "Flare", "Flare base size", "enum", "fill_footprint", options=["fill_footprint", "manual"],
              desc="fill_footprint: the largest base that fits inside the footprint outline (calculated). manual: width and thickness below."),
            E("flare_width_mm", "Flare", "Flare base width", "scalar", Wf, "mm", 2, 30, 0.25),
            E("flare_thickness_mm", "Flare", "Flare base thickness", "scalar", Tf, "mm", 1, 20, 0.25),
            E("curvature", "Flare", "Transition curvature", "scalar", 1.0, "", 0.4, 3.0, 0.05,
              desc="Shape of the blend: below 1 flares early (trumpet), above 1 flares close to bone."),
            E("endpoints", "Endpoints", "Channel and braid endpoint distribution", "enum", "fan" if side == "tib" else "staggered",
              options=["fan", "central", "staggered"],
              desc="fan: endpoints spread with the flare across the footprint. central: channels keep their body-section pattern. "
                   "staggered: as fan, but each channel ends where it would come closer to a neighbour than the spacing (measured on "
                   "the paths), so crowded channels stop early and the rest reach the stop distance."),
            E("endpoint_stop_mm", "Endpoints", "Endpoints stop this far from bone", "scalar", 0.5, "mm", 0, 8, 0.1,
              desc="Channels and braid strands end this far from bone (measured along the construct). With staggered endpoints a "
                   "channel may end earlier, where it would crowd its neighbours: the femoral end meets the notch wall side-on, so "
                   "channels on the inside of that turn are pressed together (the default femoral distribution is staggered)."),
            E("bone_zone_mm", "Graded interface", "Bone-facing zone length", "scalar", 1.0, "mm", 0, 6, 0.1, neighbors=ncomp,
              desc="Only this zone may contact bone."),
            E("gradient_mm", "Graded interface", "Material-gradient length", "scalar", 3.0, "mm", 0.25, 12, 0.25, neighbors=ncomp,
              desc="Smooth transition bone-facing -> interface -> ligament-facing composition."),
            E("comp_ligament", "Graded interface", "Ligament-facing composition", "composition", {"PLLA": 0.8, "PCL": 0.2}, "fraction",
              neighbors=ncomp, materials=STRUCTURAL_MATERIALS),
            E("comp_interface", "Graded interface", "Interface composition", "composition", {"PLGA": 0.6, "collagen_I": 0.2, "beta_TCP": 0.2},
              "fraction", neighbors=ncomp, materials=STRUCTURAL_MATERIALS),
            E("comp_bone", "Graded interface", "Bone-facing composition", "composition", {"PLGA": 0.4, "beta_TCP": 0.4, "HA": 0.2}, "fraction",
              neighbors=ncomp, materials=STRUCTURAL_MATERIALS),
            E("fixation", "Fixation feature", "Fixation feature", "enum", "none", options=["none", "plug"], neighbors=[],
              desc="This design has no fixation feature unless a press-fit plug is added here."),
            E("plug_diameter_mm", "Fixation feature", "Plug diameter", "scalar", 5.0, "mm", 2, 12, 0.25, neighbors=[]),
            E("plug_depth_mm", "Fixation feature", "Plug depth into bone", "scalar", 8.0, "mm", 2, 25, 0.5, neighbors=[]),
        ]
    return R


# Gap-filling yarn scaffold (braided / aligned): every control owned by a level.
_LEGACY_LEVEL = {
    2: (["scaffold.fiber_diameter_um", "scaffold.crimp_amplitude_um", "scaffold.crimp_wavelength_um", "scaffold.fiber_rotation_deg"],
        "Gap-filling yarns: fibers inside each yarn", "Fibers inside the gap-filling yarns"),
    3: (["scaffold.core_mode", "scaffold.core_yarns", "scaffold.aligned_sizing", "scaffold.packing_fraction", "scaffold.yarn_rotation_deg"],
        "Gap-filling yarns: core and aligned yarns", "Core / aligned gap-filling yarns"),
    4: (["scaffold.yarn_count", "scaffold.yarn_diameter_mm", "scaffold.braid_mode", "scaffold.braid_angle_deg", "scaffold.braid_pitch_mm",
         "scaffold.braid_pattern", "scaffold.braid_radius_fraction", "scaffold.braid_layers", "fit.layer_spacing_mm", "scaffold.clearance_mm",
         "fit.am.strength", "fit.pl.strength"],
        "Gap-filling yarns: braid", "Gap-filling braid"),
    5: (["fit.partition", "scaffold.fill_fraction", "fit.boundary_clearance_mm", "fit.interface_clearance_mm"] +
        ["regions.bridge.%s" % f for f in ("material", "composition", "fiber_volume_fraction", "alignment_dispersion_deg",
                                           "strand_scale", "transition_mm", "transition_mode")],
        "Gap-filling yarns: repair space and bridge region", "Repair space of the gap-filling yarns"),
    6: (["scaffold.engagement", "scaffold.overlap_tib_mm", "scaffold.overlap_fem_mm", "scaffold.bone_reach_mm", "fit.end_gap_mm",
         "fit.end_transition_mm", "fit.am.overlap_tib_mm", "fit.am.overlap_fem_mm", "fit.pl.overlap_tib_mm", "fit.pl.overlap_fem_mm"] +
        ["regions.%s.%s" % (r, f) for r in ("tibial_interface", "femoral_interface")
         for f in ("material", "composition", "fiber_volume_fraction", "alignment_dispersion_deg", "strand_scale", "transition_mm", "transition_mode")],
        "Gap-filling yarns: ends, engagement and interface regions", "Ends of the gap-filling yarns"),
}
LEGACY_IDS = frozenset(pid for ids, _, _ in _LEGACY_LEVEL.values() for pid in ids)
LAYER_PREFIXES = ("sf.", "gr.", "ch.", "rf.", "sheath.", "end.")


# ---------------------------------------------------------------- control placement, scope, level
# Design tab sections (single source for the UI). Display-only settings live in
# DISPLAY_DEFAULTS and are rendered under Inspect ("Display and inspection").
_GROUP_REMAP = {
    "bundle.am_fraction": "anatomy", "bundle.am_fraction_mode": "anatomy",
    "bundle.am_fraction_profile": "anatomy", "bundle.boundary_angle_deg": "anatomy",
    "bundle.boundary_angle_profile": "anatomy", "bundle.interface_curvature": "anatomy",
    "bundle.interface_uncertainty_mm": "anatomy", "bundle.lobe_overlap_mm": "anatomy",
    "bundle.structure_mode": "anatomy",
    "bundle.rotation_total_deg": "rotation", "bundle.rotation_distribution": "rotation",
    "bundle.rotation_profile": "rotation",
    "native.fascicle_rotation_deg": "rotation", "native.fiber_rotation_deg": "rotation",
}

_ANAT_COMPONENT = {"anatomy": "Reference anatomy, knee pose and attachments", "am": "Native AM bundle region",
                   "pl": "Native PL bundle region", "injury": "Tear definition and retained tissue",
                   "fibers": "Native fascicles and fibers", "rotation": "Native bundle arrangement and rotation"}


def _apply_group_placement(R):
    legacy = {pid: (lvl, sub, comp) for lvl, (ids, sub, comp) in _LEGACY_LEVEL.items() for pid in ids}
    for p in R:
        pid = p["id"]
        if pid in _GROUP_REMAP:
            p["group"] = _GROUP_REMAP[pid]
        elif pid.startswith("am."):
            p["group"] = "am"
        elif pid.startswith("pl."):
            p["group"] = "pl"
        if pid in legacy:
            lvl, sub, comp = legacy[pid]
            p.update(group="lvl%d" % lvl, level=lvl, subgroup=sub, component=comp, basis="assumed",
                     neighbors=[], legacy=True, objects=LEGACY_LEVEL_OBJECTS.get(lvl))
        elif pid == "scaffold.arch":
            p.update(group="lvl5", level=5, subgroup="Construct type", basis="assumed",
                     component="The whole scaffold: layered construct or gap-filling yarns",
                     neighbors=[[k, "its controls apply only to the selected construct"] for k in (1, 2, 3, 4, 6)])
        elif "level" not in p:
            p.update(level=0, component=_ANAT_COMPONENT.get(p["group"], "Reference anatomy"), neighbors=[], basis="assumed")
        if p["group"] == "fit" or p["group"] == "materials":
            raise AssertionError("scaffold control without a level owner: %s" % pid)


# Commonly used controls (focused default view); everything else is advanced
# but stays accessible through expansion.
BASIC_PARAMS = {
    "lig.size_scale", "lig.width_profile", "lig.thickness_profile",
    "lig.length_target_mm", "fp.tib.length_mm", "fp.tib.width_mm",
    "fp.fem.length_mm", "fp.fem.width_mm", "bundle.am_fraction",
    "injury.type", "injury.involvement", "injury.center_s", "injury.gap_mm",
    "injury.continuity",
    "scaffold.arch", "scaffold.yarn_diameter_mm", "scaffold.yarn_count",
    "scaffold.engagement", "scaffold.fill_fraction", "fit.partition",
    "native.fascicle_count", "scaffold.braid_layers",
    "native.fascicle_rotation_deg", "native.fiber_rotation_deg",
    "scaffold.yarn_rotation_deg", "scaffold.fiber_rotation_deg",
    "scaffold.braid_angle_deg", "bundle.rotation_total_deg",
    "am.fascicle_rotation_deg", "am.fiber_rotation_deg",
    "pl.fascicle_rotation_deg", "pl.fiber_rotation_deg",
    "regions.tibial_interface.material", "regions.bridge.material",
    "regions.femoral_interface.material",
    # six-level construct
    "sf.location", "sf.coating_nm", "sf.texture_period_nm", "sf.texture_amplitude_nm", "sf.coating_composition",
    "gr.placement", "gr.width_um", "gr.depth_um", "gr.ridge_um",
    "ch.lumen_um", "ch.wall_um", "ch.spacing_um", "ch.count_mode", "ch.am_count", "ch.pl_count", "ch.shape", "ch.wall_composition",
    "rf.size_mode", "rf.transition_mm",
    "sheath.size_mode", "sheath.anatomy_fraction", "sheath.width_profile", "sheath.thickness_profile", "sheath.wall_mm",
    "sheath.clearance_mm", "sheath.composition", "sheath.pores",
} | {"rf.%s.%s" % (b, f) for b in ("am", "pl") for f in ("strands", "strand_diameter_mm", "braid_angle_deg", "pattern",
                                                           "envelope_width_mm", "envelope_thickness_mm", "slack_pct", "composition")} \
  | {"end.%s.%s" % (s, f) for s in ("tib", "fem") for f in ("offset_long_mm", "offset_short_mm", "rotation_deg", "outline", "size_mode",
                                                           "length_mm", "width_mm", "flare_length_mm", "gradient_mm", "comp_bone",
                                                           "comp_interface", "comp_ligament", "fixation")}


def affects_of(pid):
    """What editing a parameter changes: geometry, visualization, or metadata.

    geometry: the generated scaffold/anatomy shape (and the design hash).
    visualization: only what is drawn (representative density, magnification).
    metadata: recorded with the design (materials, assumed properties); no physical
    simulation reads these, so they never produce mechanical/biological results.
    """
    if pid in DISPLAY_DEFAULTS:
        return "visualization"
    spec = BY_ID.get(pid, {})
    if (spec.get("kind") == "composition" or ".material" in pid or "composition" in pid
            or "assumed" in pid or pid.endswith("_seed") or ".seed" in pid):
        return "metadata"
    return "geometry"


def scope_of(pid):
    """Where a parameter acts: whole ACL, AM region, PL region, injury, scaffold."""
    if pid.startswith("am.") or pid == "injury.am.center_s" or pid.startswith("injury.am."):
        return "AM region"
    if pid.startswith("pl.") or pid.startswith("injury.pl."):
        return "PL region"
    if pid.startswith("injury."):
        return "injury"
    if pid.startswith("rf.am.") or pid == "ch.am_count":
        return "scaffold: AM-like region"
    if pid.startswith("rf.pl.") or pid == "ch.pl_count":
        return "scaffold: PL-like region"
    if pid.startswith("end.tib."):
        return "scaffold: tibial end"
    if pid.startswith("end.fem."):
        return "scaffold: femoral end"
    if pid.startswith(("scaffold.", "regions.", "fit.") + LAYER_PREFIXES):
        return "scaffold"
    return "whole ACL"


REGISTRY = _registry()
BY_ID = {p["id"]: p for p in REGISTRY}
LAYER_IDS = frozenset(p["id"] for p in REGISTRY if p["id"].startswith(LAYER_PREFIXES))
# composition controls of the layered construct -> owning level (validated totals)
COMPOSITION_PARAMS = {p["id"]: p["level"] for p in REGISTRY if p["kind"] == "composition" and p["id"] in LAYER_IDS}

DISPLAY_DEFAULTS = {
    "quality": "full",
    "native_fibers_per_fascicle": 7,       # representative density (display)
    "scaffold_fibers_per_yarn": 7,         # representative density (display)
    "fiber_mode": "auto",                  # auto | lines | tubes | hidden
    "crimp_exaggeration": 25.0,            # display only: crimp is microscopic
    "exploded": 0.0,                       # mm, display only
    "section_s": 0.5,
    "section_kind": "ligament",            # ligament | am | pl | axial | sagittal | coronal
    "section_offset_mm": 0.0,              # anatomical planes: offset from the ACL midpoint
}


def default_params():
    return {p["id"]: copy.deepcopy(p["default"]) for p in REGISTRY}


def default_provenance():
    prov = {}
    for p in REGISTRY:
        st = "literature" if p["sources"] else "default"
        prov[p["id"]] = {"status": st, "source": ",".join(p["sources"]) or "engine default",
                         "units": p["units"], "confidence": None}
    return prov


def new_state(case=None):
    return {"format": "acl-design-state/2.0",
            "case": case or {"id": "generic", "label": "Generic adult right knee (synthetic demonstration)",
                             "synthetic": True, "notes": "Literature-template geometry. Not patient anatomy."},
            "params": default_params(), "provenance": default_provenance(),
            "imported": None, "assets": {}, "history": [], "display": dict(DISPLAY_DEFAULTS)}


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def design_hash(state):
    """Hash of everything that changes geometry (not display, not history)."""
    core = {"params": state["params"], "assets": state.get("assets", {})}
    return hashlib.sha256(canonical(core).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- derived values
def segment_offset_for_fraction(f):
    """Signed chord offset c in the unit disk so that the area with x > c is f."""
    lo, hi = -1.0, 1.0
    for _ in range(60):
        c = 0.5 * (lo + hi)
        frac = (math.acos(c) - c * math.sqrt(max(0.0, 1 - c * c))) / math.pi
        if frac > f:
            lo = c
        else:
            hi = c
    return 0.5 * (lo + hi)


PACKING_LIMIT = {"hexagonal": 0.9069, "random": 0.80, "radial": 0.85}


def fascicle_sizing(p, area_mid_mm2):
    """Resolve count/diameter/packing with the chosen independent pair."""
    mode = p["native.sizing_mode"]
    n = int(p["native.fascicle_count"])
    phi = float(p["native.fascicle_packing"])
    d = float(p["native.fascicle_diameter_mm"])
    if mode == "count+packing":
        d = math.sqrt(4 * phi * area_mid_mm2 / (n * math.pi))
    elif mode == "count+diameter":
        phi = n * math.pi * d * d / 4 / area_mid_mm2
    else:
        n = max(1, int(round(4 * phi * area_mid_mm2 / (math.pi * d * d))))
        phi = n * math.pi * d * d / 4 / area_mid_mm2
    return n, d, phi


def braid_dependents(p, radius_mm):
    """Braid angle <-> pitch through the braid radius."""
    if radius_mm <= 0:
        return float(p["scaffold.braid_angle_deg"]), float(p["scaffold.braid_pitch_mm"])
    if p["scaffold.braid_mode"] == "angle":
        a = float(p["scaffold.braid_angle_deg"])
        pitch = 2 * math.pi * radius_mm / math.tan(math.radians(a))
    else:
        pitch = float(p["scaffold.braid_pitch_mm"])
        a = math.degrees(math.atan(2 * math.pi * radius_mm / pitch))
    return a, pitch


def normalize_composition(comp):
    comp = {k: float(v) for k, v in (comp or {}).items() if float(v) > 0}
    t = sum(comp.values())
    if t <= 0:
        return {}
    return {k: v / t for k, v in comp.items()}


# ---------------------------------------------------------------- validation
def validate_params(params):
    """Return (errors, warnings) as lists of {param, message}. No clamping."""
    errs, warns = [], []
    for pid, spec in BY_ID.items():
        if pid not in params:
            errs.append({"param": pid, "message": "missing value"})
            continue
        v = params[pid]
        k = spec["kind"]
        try:
            if k in ("scalar", "int"):
                if v is None or not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(float(v)):
                    errs.append({"param": pid, "message": "must be a finite number"})
                    continue
                if k == "int" and int(v) != v:
                    errs.append({"param": pid, "message": "must be a whole number"})
                if spec["min"] is not None and v < spec["min"]:
                    errs.append({"param": pid, "message": "%g is below the minimum %g %s" % (v, spec["min"], spec["units"])})
                if spec["max"] is not None and v > spec["max"]:
                    errs.append({"param": pid, "message": "%g is above the maximum %g %s" % (v, spec["max"], spec["units"])})
            elif k == "enum":
                if v not in spec["options"]:
                    errs.append({"param": pid, "message": "must be one of %s" % spec["options"]})
            elif k == "profile":
                if isinstance(v, dict):
                    xs, vs = v.get("s"), v.get("v")
                    if not (isinstance(xs, list) and isinstance(vs, list) and len(xs) == len(vs) and len(xs) >= 2):
                        errs.append({"param": pid, "message": "sampled profile needs matching s and v lists (>= 2 samples)"})
                    elif any((not isinstance(x, (int, float))) or x < 0 or x > 1 for x in xs):
                        errs.append({"param": pid, "message": "profile positions s must be within 0-1"})
                    elif any((spec["min"] is not None and x < spec["min"]) or (spec["max"] is not None and x > spec["max"]) for x in vs):
                        errs.append({"param": pid, "message": "values must be within %g-%g" % (spec["min"], spec["max"])})
                elif not isinstance(v, list) or len(v) != len(S_KNOTS):
                    errs.append({"param": pid, "message": "needs %d values or {s, v} samples" % len(S_KNOTS)})
                else:
                    for x in v:
                        if not isinstance(x, (int, float)) or (spec["min"] is not None and x < spec["min"]) or (spec["max"] is not None and x > spec["max"]):
                            errs.append({"param": pid, "message": "values must be within %g-%g" % (spec["min"], spec["max"])})
                            break
            elif k == "point3":
                if not (isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v)):
                    errs.append({"param": pid, "message": "needs [x, y, z] in mm"})
            elif k == "offsets":
                if not (isinstance(v, list) and len(v) == 2 and all(isinstance(q, list) and len(q) == 3 for q in v)):
                    errs.append({"param": pid, "message": "needs two [x, y, z] offsets"})
                elif max(abs(x) for q in v for x in q) > 15:
                    errs.append({"param": pid, "message": "control offsets above 15 mm are not supported"})
            elif k == "points3":
                if not isinstance(v, list) or (v and (len(v) < 3 or not all(isinstance(q, list) and len(q) == 3 for q in v))):
                    errs.append({"param": pid, "message": "needs >= 3 [x, y, z] points or an empty list"})
            elif k == "polygon2":
                if not isinstance(v, list) or (v and (len(v) < 3 or not all(isinstance(q, list) and len(q) == 2 for q in v))):
                    errs.append({"param": pid, "message": "needs >= 3 [a, b] points or an empty list"})
            elif k == "contours":
                if not isinstance(v, list):
                    errs.append({"param": pid, "message": "needs a list of {s, points} contours"})
                else:
                    for it in v:
                        pts = it.get("points") if isinstance(it, dict) else None
                        sv = it.get("s") if isinstance(it, dict) else None
                        if not isinstance(sv, (int, float)) or not 0 <= sv <= 1:
                            errs.append({"param": pid, "message": "each contour needs s within 0-1"})
                            break
                        if not (isinstance(pts, list) and len(pts) >= 8 and all(isinstance(q, list) and len(q) in (2, 3) for q in pts)):
                            errs.append({"param": pid, "message": "each contour needs >= 8 points [x, y, z] (model frame, mm)"})
                            break
            elif k == "composition":
                if not isinstance(v, dict) or not v:
                    errs.append({"param": pid, "message": "needs at least one material fraction"})
                else:
                    allowed = spec.get("materials") or list(MATERIALS)
                    bad = [m for m in v if m not in allowed]
                    if bad:
                        errs.append({"param": pid, "message": "material(s) %s not allowed here (choose from %s)" % (bad, ", ".join(allowed))})
                    if any(float(x) < 0 for x in v.values()):
                        errs.append({"param": pid, "message": "fractions must be >= 0"})
                    s = sum(float(x) for x in v.values())
                    if s <= 0:
                        errs.append({"param": pid, "message": "fractions sum to 0"})
                    elif abs(s - 1) > 1e-6:
                        warns.append({"param": pid, "message": "fractions sum to %.3f; normalized to 1 for geometry and export" % s})
        except (TypeError, ValueError):
            errs.append({"param": pid, "message": "invalid value"})
    if errs:
        return errs, warns
    # dependency rules
    for side in ("tib", "fem"):
        if params["fp.%s.shape" % side] == "polygon" and len(params["fp.%s.polygon" % side]) < 3:
            errs.append({"param": "fp.%s.polygon" % side, "message": "shape is 'polygon' but no boundary is defined"})
        if params["fp.%s.shape" % side] == "crescent":
            L = params["fp.%s.length_mm" % side]
            R = params["fp.%s.curvature_mm" % side]
            if L / R > 1.8 * math.pi:
                errs.append({"param": "fp.%s.curvature_mm" % side,
                             "message": "crescent radius %.1f mm is too small for a %.1f mm long footprint (arc > 324 deg)" % (R, L)})
            if params["fp.%s.width_mm" % side] >= 2 * R:
                errs.append({"param": "fp.%s.width_mm" % side, "message": "crescent width must be < 2 x radius"})
    if params["injury.type"] == "partial" and params["injury.involvement"] == "whole":
        errs.append({"param": "injury.involvement", "message": "a partial tear needs AM, PL, a sector or per_bundle (not 'whole')"})
    if params["injury.type"] == "partial" and params["injury.involvement"] == "per_bundle" and \
            params["injury.am.depth_fraction"] <= 0 and params["injury.pl.depth_fraction"] <= 0:
        errs.append({"param": "injury.am.depth_fraction", "message": "per_bundle tear: at least one bundle needs a tear depth above 0"})
    if params["lig.section_mode"] == "imported" and not params["lig.section_contours"]:
        errs.append({"param": "lig.section_mode", "message": "'imported' sections need measured contours (import them from the MRI model)"})
    if params["injury.type"] == "complete" and params["injury.involvement"] != "whole":
        warns.append({"param": "injury.involvement", "message": "complete tear: the whole section is treated as torn"})
    if params["scaffold.arch"] == "braided" and int(params["scaffold.yarn_count"]) % 2:
        errs.append({"param": "scaffold.yarn_count", "message": "a braid needs an even number of carriers (half per direction)"})
    if params["native.sizing_mode"] != "count+packing":
        pass  # checked against geometry in the build (needs the section area)
    return errs, warns


def migrate_params(params, prov=None):
    """Bring a design from an older version up to the current registry: missing
    parameters get their defaults (status 'default') and retired options are
    mapped. Returns human-readable notes; nothing else is changed."""
    notes = []
    added = []
    for pid, spec in BY_ID.items():
        if pid not in params:
            params[pid] = copy.deepcopy(spec["default"])
            added.append(pid)
            if prov is not None and pid not in prov:
                st = "literature" if spec["sources"] else "default"
                prov[pid] = {"status": st, "source": ",".join(spec["sources"]) or "engine default",
                             "units": spec["units"], "confidence": None}
    if added:
        notes.append("%d new parameters set to their defaults (%s%s)" % (len(added), ", ".join(added[:6]), "..." if len(added) > 6 else ""))
    retired = {"native.fascicle_helix_deg": "native.fascicle_rotation_deg", "native.fiber_helix_deg": "native.fiber_rotation_deg",
               "am.fiber_helix_deg": "am.fiber_rotation_deg", "pl.fiber_helix_deg": "pl.fiber_rotation_deg",
               "scaffold.yarn_helix_deg": "scaffold.yarn_rotation_deg", "scaffold.fiber_helix_deg": "scaffold.fiber_rotation_deg"}
    for old, new in retired.items():
        if old in params:
            v = params.pop(old)
            if v not in (0, 0.0, None):
                notes.append("%s = %s was a constant helix angle (it winds far beyond 30 deg over a unit); it is replaced by %s "
                             "(total end-to-end rotation, surgeon-informed range -30..30 deg), which stays at %s" % (old, v, new, params.get(new)))
    for old, new in (("am.rotation_total_deg", "am.fascicle_rotation_deg"), ("pl.rotation_total_deg", "pl.fascicle_rotation_deg")):
        if old in params:
            v = float(params.pop(old))
            if abs(v) <= 30 and new in params and old not in ("",):
                params[new] = v
            elif v:
                notes.append("%s = %g deg is outside the surgeon-informed range -30..30 deg; %s stays at %s (set it explicitly)" % (old, v, new, params.get(new)))
    for old in ("native.fascicle_handedness", "native.fiber_handedness", "scaffold.yarn_handedness", "scaffold.fiber_handedness"):
        params.pop(old, None)                 # the sign of the total rotation gives the direction
    if params.get("scaffold.engagement") == "sleeve":
        params["scaffold.engagement"] = "abut"
        notes.append("engagement 'sleeve' (a braid wrapped outside the ligament) was retired because it puts scaffold outside the "
                     "anatomical envelope; 'abut' is used instead")
    return notes


def coerce_types(params):
    out = dict(params)
    for pid, spec in BY_ID.items():
        if pid in out and spec["kind"] == "int" and isinstance(out[pid], float) and out[pid].is_integer():
            out[pid] = int(out[pid])
    return out


def registry_for_ui():
    params = []
    for p in REGISTRY:
        q = dict(p)
        q["scope"] = scope_of(p["id"])
        q["affects"] = affects_of(p["id"])              # geometry | visualization | metadata
        q["owner_level"] = p.get("level", 0)            # structural level (0 = Anatomy and Pose)
        q["level"] = "basic" if p["id"] in BASIC_PARAMS else "advanced"
        q["help"] = help_for(p)
        q["panel"] = "scaffold" if p.get("level") else "anatomy"
        params.append(q)
    return {"groups": [{"id": g, "label": l, "panel": "anatomy" if g in ANATOMY_GROUPS else "scaffold"} for g, l in GROUPS],
            "params": params, "levels": LEVELS, "level0": LEVEL0, "s_knots": S_KNOTS, "materials": MATERIALS,
            "display_defaults": DISPLAY_DEFAULTS,
            "regions": REGION_IDS, "channel_range_um": [150, 300],
            "basis_values": ["measured", "assumed", "experimental"]}


# ---------------------------------------------------------------- explanations
# Curated per-parameter help. Fields: what / increase / decrease / look /
# depends. Units come from the parameter definition; value source comes from
# live provenance. All text describes what the engine does with the value;
# nothing here promises healing, strength or stability effects.
_HELP_OVERRIDE = {
    "scaffold.fiber_rotation_deg": {
        "what": "Total fiber rotation: how far fibers turn around their parent scaffold yarn from one end to the other.",
        "increase": "More twist toward +30°; the sign selects the direction under the displayed orientation convention.",
        "decrease": "Less twist toward −30° (through zero = straight fibers); opposite direction from positive values.",
        "look": "Scaffold yarns in the 3D view and the per-unit rotation table under Checks. This is total end-to-end rotation, not the local angle between a fiber and the yarn axis. Design range −30° to +30°.",
        "depends": "Applies to every scaffold yarn; braid winding is reported separately under Checks.",
    },
    "scaffold.yarn_rotation_deg": {
        "what": "Total fiber rotation: how far aligned scaffold yarns (and their fibers) turn around their parent repair unit from one end to the other. Aligned architecture only.",
        "increase": "More twist toward +30°; the sign selects the direction.",
        "decrease": "Less twist toward −30° (through zero = straight yarns).",
        "look": "Scaffold yarns in the 3D view and the rotation table under Checks. Total end-to-end rotation; design range −30° to +30°.",
        "depends": "Used only when scaffold architecture is aligned (a braid winds by construction).",
    },
    "native.fascicle_rotation_deg": {
        "what": "Total fiber rotation: how far native fascicles turn around their parent bundle (AM or PL) from one end to the other, measured relative to the rotating bundle region. Shared structure only.",
        "increase": "More twist toward +30°; 0 leaves the bundle's fascicles parallel, as in knee extension.",
        "decrease": "Less twist toward −30° (through zero = parallel).",
        "look": "Retained fascicle groups in the 3D view. Total end-to-end rotation; design range −30° to +30°.",
        "depends": "Replaced by the per-bundle rotations when per-bundle structure is independent.",
    },
    "native.fiber_rotation_deg": {
        "what": "Total fiber rotation: how far fibers turn around their parent fascicle group from one end to the other. Shared structure only.",
        "increase": "More twist toward +30°.",
        "decrease": "Less twist toward −30° (through zero = straight fibers).",
        "look": "Representative native fibers in the 3D view and the rotation table under Checks. Total end-to-end rotation; design range −30° to +30°.",
        "depends": "Replaced by the per-bundle fiber rotations when per-bundle structure is independent.",
    },
    "am.fascicle_rotation_deg": {
        "what": "Total fiber rotation of AM fascicles about the AM bundle path. Independent structure only.",
        "increase": "More twist toward +30°.",
        "decrease": "Less twist toward −30° (through zero = parallel).",
        "look": "AM fascicle groups. Total end-to-end rotation; design range −30° to +30°.",
        "depends": "Used when per-bundle structure is independent; otherwise the shared native rotation applies.",
    },
    "am.fiber_rotation_deg": {
        "what": "Total fiber rotation of AM fibers about their parent fascicle. Independent structure only.",
        "increase": "More twist toward +30°.",
        "decrease": "Less twist toward −30° (through zero = straight).",
        "look": "AM representative fibers. Total end-to-end rotation; design range −30° to +30°.",
        "depends": "Used when per-bundle structure is independent.",
    },
    "pl.fascicle_rotation_deg": {
        "what": "Total fiber rotation of PL fascicles about the PL bundle path. Independent structure only.",
        "increase": "More twist toward +30°.",
        "decrease": "Less twist toward −30° (through zero = parallel).",
        "look": "PL fascicle groups. Total end-to-end rotation; design range −30° to +30°.",
        "depends": "Used when per-bundle structure is independent; otherwise the shared native rotation applies.",
    },
    "pl.fiber_rotation_deg": {
        "what": "Total fiber rotation of PL fibers about their parent fascicle. Independent structure only.",
        "increase": "More twist toward +30°.",
        "decrease": "Less twist toward −30° (through zero = straight).",
        "look": "PL representative fibers. Total end-to-end rotation; design range −30° to +30°.",
        "depends": "Used when per-bundle structure is independent.",
    },
    "bundle.rotation_total_deg": {
        "what": "Rotation of the whole AM/PL bundle arrangement about the ACL centreline from tibia to femur (the parent level). Near 0 in extension.",
        "increase": "The AM/PL regions wind further in the positive direction along the ligament.",
        "decrease": "Less winding, or winding in the opposite direction for negative values.",
        "look": "AM/PL bundle regions and their centroid paths. Not limited by the ±30° fiber-rotation rule, which applies to fibers within each unit.",
        "depends": "Distributed along the length by the rotation distribution; within-bundle rotation is measured relative to it.",
    },
    "injury.gap_mm": {
        "what": "Longitudinal length of missing tissue at the tear, before stump retraction.",
        "increase": "A longer missing-tissue interval and shorter remnants.",
        "decrease": "A shorter interval and longer remnants.",
        "look": "The missing/damaged tissue volume between the tibial and femoral remnants.",
        "depends": "Total repair length also includes stump retraction and, for interdigitating designs, the overlap lengths.",
    },
    "injury.center_s": {
        "what": "Tear position along the ligament, 0 at the tibia and 1 at the femur.",
        "increase": "Moves the tear toward the femur.",
        "decrease": "Moves the tear toward the tibia.",
        "look": "The missing-tissue volume and the remnant lengths on each side.",
        "depends": "Per-bundle tears use the per-bundle positions instead.",
    },
    "scaffold.fill_fraction": {
        "what": "Fraction of the affected cross-section area the scaffold may occupy, scaled about the region centroid and then inset by the clearances.",
        "increase": "The scaffold cross-section fills more of the repair space.",
        "decrease": "The scaffold cross-section fills less of the repair space.",
        "look": "Scaffold yarns against the dashed repair-space boundary in the section view.",
        "depends": "Clearances always apply on top; the scaffold never enlarges the ligament to fit.",
    },
    "scaffold.yarn_diameter_mm": {
        "what": "Base diameter of scaffold yarns; material regions scale it with their strand-scale setting.",
        "increase": "Thicker yarns; fewer fit at the same packing.",
        "decrease": "Thinner yarns; more fit at the same packing.",
        "look": "Scaffold yarn thickness in the 3D view and in cross-section.",
        "depends": "Scaled per region by the strand diameter scale; checked against yarn clearance.",
    },
    "scaffold.engagement": {
        "what": "How yarns meet the remnants. Interdigitate: aligned and core yarns continue into the remnants between retained fascicles for the overlap length. Abut: every yarn ends at the local tear face.",
        "increase": "Choose interdigitate for yarns that continue into the remnants.",
        "decrease": "Choose abut for yarns that end at the tear face.",
        "look": "Yarn ends at the remnant faces. Braids always abut.",
        "depends": "Interdigitate enables the overlap lengths; abut disables them.",
    },
    "fit.partition": {
        "what": "Whether the scaffold is one unit over the affected region or separate AM and PL units, each kept inside its own bundle region.",
        "increase": "Choose per_bundle for separate AM and PL scaffolds.",
        "decrease": "Choose single for one scaffold over the affected region.",
        "look": "Repair-space volumes and yarn grouping per unit.",
        "depends": "Per-bundle units use the per-unit overlap lengths and fitting strengths.",
    },
    "fit.boundary_clearance_mm": {
        "what": "Minimum distance between any strand surface and the target ligament boundary.",
        "increase": "Strands must stay further inside; less room for the scaffold.",
        "decrease": "Strands may approach the boundary more closely.",
        "look": "Outer yarns against the ligament envelope in the section view.",
        "depends": "Violations fail the fit checks; shortfalls below this are warnings.",
    },
    "fit.layer_spacing_mm": {
        "what": "Gap between concentric braid layers, in addition to their over-under depth.",
        "increase": "Layers sit further apart; the braid occupies more of the space.",
        "decrease": "Layers sit closer together.",
        "look": "Adjacent braid layers in cross-section.",
        "depends": "Used with two or more braid layers in the braided architecture.",
    },
    "scaffold.braid_angle_deg": {
        "what": "Angle between braid yarns and the scaffold axis. Linked to the pitch through the braid radius: tan(angle) = 2πR / pitch.",
        "increase": "Steeper yarns and a shorter pitch.",
        "decrease": "Shallower yarns and a longer pitch.",
        "look": "Yarn directions on the braid surface.",
        "depends": "One of angle/pitch is computed from the other via the braid mode.",
    },
    "scaffold.braid_pitch_mm": {
        "what": "Axial length of one braid repeat. Linked to the angle through the braid radius: tan(angle) = 2πR / pitch.",
        "increase": "Longer repeat; shallower yarn angle.",
        "decrease": "Shorter repeat; steeper yarn angle.",
        "look": "Yarn directions on the braid surface.",
        "depends": "One of angle/pitch is computed from the other via the braid mode.",
    },
}


def _generic_help(spec):
    pid, label, kind = spec["id"], spec["label"], spec["kind"]
    units, lo, hi = spec["units"], spec["min"], spec["max"]
    bounds = ""
    if lo is not None and hi is not None and kind in ("scalar", "int"):
        bounds = " Allowed range %s–%s%s." % (lo, hi, (" " + units) if units else "")
    what = (spec.get("description") or ("Sets the " + label.lower() + " used when generating the geometry.")).strip()
    L = label.lower()
    sub = (spec.get("subgroup") or "").lower()
    if kind == "enum":
        opts = ", ".join(spec.get("options") or [])
        return {"what": what + " Options: %s." % opts,
                "increase": "Select a different option; each option changes which geometry is generated, as described above.",
                "decrease": "Select a different option; each option changes which geometry is generated, as described above.",
                "look": _look_for(pid, L, sub)}
    if kind == "profile":
        return {"what": what + " Edited as a curve from the tibia (s=0) to the femur (s=1)." + bounds,
                "increase": "Raising curve points increases the value along the ligament.",
                "decrease": "Lowering curve points decreases the value along the ligament.",
                "look": _look_for(pid, L, sub)}
    if kind in ("point3", "offsets"):
        return {"what": what + " Editable in the 3D view by dragging the handle, in millimetres.",
                "increase": "Moves the point toward larger coordinates.",
                "decrease": "Moves the point toward smaller coordinates.",
                "look": _look_for(pid, L, sub)}
    if kind == "polygon2":
        return {"what": what,
                "increase": "Adding or outward-dragged vertices enlarges the boundary.",
                "decrease": "Removing or inward-dragged vertices shrinks the boundary.",
                "look": _look_for(pid, L, sub)}
    if kind == "composition":
        return {"what": what + " Geometry and exports use normalized fractions.",
                "increase": "Raising a fraction gives that material a larger share.",
                "decrease": "Lowering a fraction gives that material a smaller share.",
                "look": _look_for(pid, L, sub)}
    if kind in ("points3", "contours"):
        return {"what": what or label + ".",
                "increase": "Imported data: re-import to change it.",
                "decrease": "Imported data: re-import to change it.",
                "look": _look_for(pid, L, sub)}
    # scalar / int
    if any(k in L for k in ("seed",)) or "seed" in pid:
        up, down = ("Selects a different pseudo-random arrangement.", "Selects a different pseudo-random arrangement.")
    elif "s (" in L or pid.endswith("center_s") or "position" in L or L.startswith("tear position"):
        up, down = ("Moves it toward the femur (s = 1).", "Moves it toward the tibia (s = 0).")
    elif any(k in L for k in ("angle", "rotation", "orientation", "obliquity", "azimuth", "dispersion", "tilt", "slope", "flexion")) or units == "deg":
        up, down = ("Larger angle, toward +%s°." % hi if hi is not None else "Larger angle.",
                    "Smaller angle, toward %s°; the sign selects the direction." % lo if lo is not None else "Smaller angle; the sign selects the direction.")
    elif any(k in L for k in ("count", "number", "layers", "lobes", "yarns", "groups", "carriers")) or units == "count":
        up, down = ("More items are generated.", "Fewer items are generated.")
    elif any(k in L for k in ("fraction", "share", "ratio", "packing", "continuity", "strength", "extent", "density", "contribution")) or units in ("", "x"):
        if units == "x" or "scale" in L:
            up, down = ("Multiplies the size up (larger geometry).", "Multiplies the size down (smaller geometry).")
        else:
            up, down = ("A larger share.", "A smaller share.")
    else:
        up, down = ("Larger value: the affected geometry grows.", "Smaller value: the affected geometry shrinks.")
    return {"what": what + bounds, "increase": up, "decrease": down, "look": _look_for(pid, L, sub)}


_LAYER_LOOK = {
    "sf.": "Level 1: coated channel lumens (teal) in the 3D view; texture and coating at true scale in the magnified detail.",
    "gr.": "Level 2: representative groove paths inside the lumens; the true groove profile is in the magnified detail.",
    "ch.": "Level 3: the hollow channels (isolate the level, or use the cross-section to see lumens, walls and spacing).",
    "rf.": "Level 4: the braided mantles (orange AM-like, blue PL-like) in 3D and their strands in the cross-section.",
    "sheath.": "Level 5: the translucent outer sheath, and its wall in the cross-section.",
    "end.": "Level 6: the flared ends, the footprint outlines on bone and the graded colours near each bone.",
}


def _look_for(pid, L, sub):
    for pre, txt in _LAYER_LOOK.items():
        if pid.startswith(pre):
            return txt
    if pid.startswith("fp.tib"):
        return "The tibial footprint patch on the tibial plateau."
    if pid.startswith("fp.fem"):
        return "The femoral footprint patch on the lateral wall of the notch."
    if pid.startswith("bone."):
        return "The procedural bone and cartilage surfaces."
    if pid.startswith("am."):
        return "The AM bundle region, its lobe and its fascicles."
    if pid.startswith("pl."):
        return "The PL bundle region, its lobe and its fascicles."
    if pid.startswith("injury."):
        return "The missing/damaged tissue volume and the remnants."
    if pid.startswith("regions.") or pid.startswith("scaffold.") or pid.startswith("fit."):
        if "braid" in pid or "yarn" in pid or "core" in pid:
            return "The scaffold yarns in the 3D view and in cross-section."
        if "layer" in pid:
            return "Adjacent braid layers in cross-section."
        if "clearance" in pid or "fill" in pid or "overlap" in pid or "gap" in pid:
            return "Yarn placement against the repair-space boundary in the section view."
        return "The scaffold and its repair space."
    if "bundle" in pid or "interface" in pid or "am_fraction" in pid or "lobe" in pid:
        return "The AM/PL boundary surface and the bundle regions."
    if "rotation" in pid or "twist" in L:
        return "Fascicle and fiber directions in the 3D view and the rotation table under Checks."
    if "fascicle" in pid or "fiber" in pid or "crimp" in pid:
        return "Retained fascicle groups and representative fibers."
    if "centreline" in L or "bow" in pid or "length" in pid or "ctrl" in pid:
        return "The ligament centreline and overall length."
    return "The ligament in the 3D view and in cross-section."


_DEPENDS = {
    "native.fascicle_count": "Linked to diameter and packing by the sizing mode: count, diameter and packing satisfy packing = N·π·d²/4 / A; one of the three is computed.",
    "native.fascicle_diameter_mm": "Linked to count and packing by the sizing mode; one of the three is computed.",
    "native.fascicle_packing": "Linked to count and diameter by the sizing mode; one of the three is computed.",
    "native.sizing_mode": "Chooses which two of count, diameter and packing are independent; the third is computed from the mid-section area.",
    "scaffold.braid_angle_deg": "Linked to the pitch through the braid radius (tan(angle) = 2πR / pitch); one of the two is computed.",
    "scaffold.braid_pitch_mm": "Linked to the angle through the braid radius; one of the two is computed.",
    "scaffold.braid_mode": "Chooses whether the angle or the pitch is the independent braid variable.",
    "lig.section_mode": "Chooses how the cross-section is defined; the unused profiles become derived (calculated) values.",
    "lig.width_profile": "Independent in width/thickness mode; derived from area/aspect or from the lobes in other modes.",
    "lig.thickness_profile": "Independent in width/thickness mode; derived in other modes.",
    "lig.area_profile": "Independent in area/aspect mode; derived from the contour otherwise.",
    "lig.aspect_profile": "Independent in area/aspect mode; derived from the contour otherwise.",
    "bundle.am_fraction_mode": "Constant keeps one AM share everywhere; profile varies it along the ligament.",
    "bundle.structure_mode": "Shared uses the common fascicle/fiber settings; independent enables the per-bundle settings.",
    "scaffold.arch": "Braided enables braid, layer and core settings; aligned enables yarn rotation and aligned sizing.",
    "scaffold.aligned_sizing": "Count uses the yarn count; packing derives the count from the packing fraction.",
    "scaffold.core_mode": "Count uses the given core yarn number; max fills with as many as fit.",
    "scaffold.engagement": "Interdigitate enables the overlap lengths; abut disables them.",
    "fit.partition": "Per-bundle units use the per-unit overlaps and fitting strengths; single uses the shared overlaps.",
    "injury.type": "Intact disables injury and scaffold settings; partial enables the affected-region settings.",
    "injury.involvement": "Selects which region is torn; sector and per-bundle options enable their own settings.",
    "lig.length_mode": "From-attachments follows the anatomy; target solves the anterior bow to reach the target length.",
    "lig.contour_irregularity_mm": "Zero disables the lobe count and seed.",
    "native.crimp_amplitude_um": "Zero disables the crimp wavelength and profile.",
    "scaffold.crimp_amplitude_um": "Zero disables the waviness wavelength.",
    "scaffold.arch": "layered enables Levels 1-6 of the construct; braided/aligned enable the gap-filling yarn controls in each level.",
    "sheath.size_mode": "fit_anatomy and fit_contents calculate the width/thickness profiles; profiles makes them user-set.",
    "rf.size_mode": "auto calculates each reinforcement envelope; manual makes the widths and thicknesses user-set.",
    "ch.count_mode": "max_fit calculates the channel counts from the space and pitch; count makes them user-set.",
    "ch.lumen_um": "Outside diameter = lumen + 2 x wall; pitch = outside diameter + spacing. Both are calculated and shown.",
    "ch.wall_um": "Outside diameter = lumen + 2 x wall; pitch = outside diameter + spacing.",
    "ch.spacing_um": "Pitch = outside diameter + spacing.",
    "end.tib.size_mode": "match_anatomy calculates the footprint size from Anatomy and Pose; manual makes it user-set.",
    "end.fem.size_mode": "match_anatomy calculates the footprint size from Anatomy and Pose; manual makes it user-set.",
    "end.tib.flare_mode": "fill_footprint calculates the flare base; manual makes it user-set.",
    "end.fem.flare_mode": "fill_footprint calculates the flare base; manual makes it user-set.",
    "sf.coating_nm": "With explicit coating, the open lumen = lumen - 2 x thickness.",
    "gr.depth_um": "Must leave the minimum residual wall in the channel wall (Level 3) and in the sheath wall (Level 5).",
}


def _guide_for(pid):
    L = pid.lower()
    if "width" in L and ("lig." in L or "am." in L or "pl." in L):
        return "measure_width"
    if "thickness" in L and ("lig." in L or "am." in L or "pl." in L):
        return "measure_thickness"
    if ".rotation_" in L or L.endswith("_deg") and any(k in L for k in ("rotation", "braid_angle", "obliquity", "azimuth", "orientation")):
        return "arc"
    if L in ("injury.center_s", "injury.gap_mm") or L.startswith("injury.am.") or L.startswith("injury.pl."):
        return "span"
    if L.startswith("fp."):
        return "footprint"
    if "layer_spacing" in L or "braid_layers" in L:
        return "layers"
    if "yarn_diameter" in L or "strand_scale" in L or "fiber_diameter" in L:
        return "strand"
    if "clearance" in L or "fill_fraction" in L or "overlap" in L:
        return "region"
    return "highlight"


def help_for(spec):
    pid = spec["id"]
    h = _generic_help(spec)
    ov = _HELP_OVERRIDE.get(pid)
    if ov:
        h.update({k: v for k, v in ov.items() if k in ("what", "increase", "decrease", "look")})
    h["depends"] = _DEPENDS.get(pid, ov.get("depends", "") if ov else "")
    if not h["depends"]:
        h["depends"] = "No other setting constrains it directly; greyed-out controls explain why when inactive."
    h["guide"] = _guide_for(pid)
    return h


# ---------------------------------------------------------------- stage dependencies
# Explicit parameter sets per build stage. A stage rebuilds only when its own
# subset (or an upstream stage it consumes) changes, so editing a scaffold
# slider never regenerates the bones. Verified by tests/test_perf_cache.py.
_FP_FIELDS = ("center", "shape", "length_mm", "width_mm", "orientation_deg",
              "curvature_mm", "polygon", "am_share", "partition_rotation_deg")
_LOBE_FIELDS = ("width_profile", "thickness_profile", "shape", "orientation_deg",
                "shift_profile", "flare_tib_mm", "flare_fem_mm")
BONE_PARAMS = frozenset({
    "bone.femur_bicondylar_width_mm", "bone.tibia_plateau_width_mm",
    "bone.notch_width_mm", "bone.tibial_slope_deg",
    "bone.cartilage_femoral_mm", "bone.cartilage_tibial_mm",
    "knee.flexion_deg",
})
FEMUR_PARAMS = frozenset({
    "bone.femur_bicondylar_width_mm", "bone.notch_width_mm",
    "bone.cartilage_femoral_mm", "knee.flexion_deg",
})
TIBIA_PARAMS = frozenset({
    "bone.tibia_plateau_width_mm", "bone.tibial_slope_deg",
    "bone.cartilage_tibial_mm",
})
LIG_PARAMS = frozenset(
    ["fp.%s.%s" % (s, f) for s in ("tib", "fem") for f in _FP_FIELDS] +
    [p["id"] for p in REGISTRY if p["id"].startswith("lig.")] +
    ["bundle.am_fraction", "bundle.am_fraction_mode", "bundle.am_fraction_profile",
     "bundle.boundary_angle_deg", "bundle.boundary_angle_profile",
     "bundle.interface_curvature", "bundle.lobe_overlap_mm",
     "bundle.rotation_distribution", "bundle.rotation_profile",
     "bundle.rotation_total_deg", "bundle.structure_mode"] +
    ["%s.%s" % (b, f) for b in ("am", "pl") for f in _LOBE_FIELDS])
TISSUE_PARAMS = frozenset(
    [p["id"] for p in REGISTRY if p["id"].startswith("injury.")] +
    [p["id"] for p in REGISTRY if p["id"].startswith("native.")] +
    ["bundle.structure_mode",
     "am.fascicle_diameter_scale", "am.fascicle_rotation_deg",
     "am.fiber_rotation_deg", "am.dispersion_deg",
     "pl.fascicle_diameter_scale", "pl.fascicle_rotation_deg",
     "pl.fiber_rotation_deg", "pl.dispersion_deg"])
SCAFFOLD_PARAMS = frozenset(
    [p["id"] for p in REGISTRY if p["id"].startswith(("scaffold.", "fit.", "regions."))] +
    ["lig.flare_tib_mm", "lig.flare_fem_mm", "native.layout_seed"])
# six-level construct stage: its own controls + the construct switch + whether a scaffold is needed
LAYER_PARAMS = LAYER_IDS | {"scaffold.arch", "injury.type"}


def layer_inactive(p):
    """Six-level controls with no effect in the current configuration -> reason (shown greyed
    out with the reason; 'calculated' entries also show their derived value)."""
    out = {}

    def na(cond, ids, why):
        if cond:
            for i in ids:
                out.setdefault(i, why)
    layered = p.get("scaffold.arch") == "layered"
    na(p.get("injury.type") == "intact", list(LAYER_IDS), "not used: injury type is intact (no scaffold)")
    na(not layered, list(LAYER_IDS), "layered construct only (Level 5: Scaffold construct)")
    na(layered, list(LEGACY_IDS), "gap-filling yarn scaffold only (Level 5: Scaffold construct = braided/aligned)")
    mode = p.get("sheath.size_mode")
    na(mode != "fit_anatomy", ["sheath.anatomy_fraction"], "used with fit-to-anatomy sizing")
    if mode != "profiles":
        out.setdefault("sheath.width_profile", "calculated (%s sizing)" % mode.replace("_", " "))
        out.setdefault("sheath.thickness_profile", "calculated (%s sizing)" % mode.replace("_", " "))
    na(p.get("sheath.pores") != "explicit", ["sheath.pore_diameter_um", "sheath.pore_spacing_mm", "sheath.pore_pattern",
                                              "sheath.pore_coverage"], "pores are not modelled (Pores in the sheath wall = none)")
    if p.get("rf.size_mode") != "manual":
        for b in ("am", "pl"):
            for f in ("envelope_width_mm", "envelope_thickness_mm"):
                out.setdefault("rf.%s.%s" % (b, f), "calculated (automatic envelope sizing)")
    if p.get("ch.count_mode") != "count":
        out.setdefault("ch.am_count", "calculated (as many as fit)")
        out.setdefault("ch.pl_count", "calculated (as many as fit)")
    na(p.get("ch.shape") == "circular", ["ch.aspect"], "circular channels have no aspect")
    na(p.get("ch.connections") != "ports", ["ch.port_spacing_mm", "ch.port_diameter_um"], "no connections between channels")
    na(float(p.get("ch.waviness_um", 0)) == 0, ["ch.wavelength_mm"], "no effect while the waviness amplitude is 0")
    na(p.get("gr.placement") == "none", ["gr.coverage", "gr.width_um", "gr.depth_um", "gr.ridge_um", "gr.bottom_radius_um",
                                         "gr.orientation_deg", "gr.min_residual_um"], "no grooved surfaces selected")
    na(p.get("sf.location") == "none", ["sf.coverage_start", "sf.coverage_end", "sf.texture", "sf.texture_orientation_deg",
                                        "sf.texture_period_nm", "sf.texture_amplitude_nm", "sf.coating_nm", "sf.coating_explicit",
                                        "sf.coating_composition"], "no surface treatment (Treated surfaces = none)")
    na(p.get("sf.texture") == "none", ["sf.texture_orientation_deg", "sf.texture_period_nm", "sf.texture_amplitude_nm"], "no texture")
    for s in ("tib", "fem"):
        k = "end.%s." % s
        if p.get(k + "size_mode") != "manual":
            out.setdefault(k + "length_mm", "calculated (matches the anatomical footprint)")
            out.setdefault(k + "width_mm", "calculated (matches the anatomical footprint)")
        if p.get(k + "flare_mode") != "manual":
            out.setdefault(k + "flare_width_mm", "calculated (largest base inside the footprint)")
            out.setdefault(k + "flare_thickness_mm", "calculated (largest base inside the footprint)")
        na(p.get(k + "fixation") != "plug", [k + "plug_diameter_mm", k + "plug_depth_mm"], "no fixation feature at this end")
    return out


def inactive_params(p):
    """Parameters with no effect in the current configuration -> reason.
    (Mirrors the designer's greyed-out controls; used by the registry audit.)"""
    out = {}

    def na(cond, ids, why):
        if cond:
            for i in ids:
                out.setdefault(i, why)
    sm = p["native.sizing_mode"]
    out[{"count+packing": "native.fascicle_diameter_mm", "count+diameter": "native.fascicle_packing"}.get(sm, "native.fascicle_count")] = "computed (sizing mode)"
    out["scaffold.braid_pitch_mm" if p["scaffold.braid_mode"] == "angle" else "scaffold.braid_angle_deg"] = "computed (braid mode)"
    partial = p["injury.type"] == "partial"
    na(p["injury.type"] == "intact", [k for k in BY_ID if k.startswith(("injury.", "scaffold.", "regions.", "fit.")) and k != "injury.type"], "intact")
    na(not (partial and p["injury.involvement"] == "sector"), ["injury.sector_center_deg", "injury.sector_width_deg", "injury.depth_fraction"], "sector only")
    na(not (partial and p["injury.involvement"] == "per_bundle"), [k for k in BY_ID if k.startswith(("injury.am.", "injury.pl."))], "per-bundle tears only")
    braided = p["scaffold.arch"] == "braided"
    na(not braided, ["scaffold.braid_mode", "scaffold.braid_angle_deg", "scaffold.braid_pitch_mm", "scaffold.braid_pattern",
                     "scaffold.braid_radius_fraction", "scaffold.core_yarns", "scaffold.braid_layers", "fit.layer_spacing_mm",
                     "scaffold.core_mode", "fit.end_transition_mm"], "braid only")
    na(braided, ["scaffold.yarn_rotation_deg", "scaffold.aligned_sizing", "scaffold.packing_fraction"], "aligned only")
    na(not braided and p["scaffold.aligned_sizing"] != "packing", ["scaffold.packing_fraction"], "packing sizing only")
    na(braided and p["scaffold.core_mode"] == "max", ["scaffold.core_yarns"], "core = as many as fit")
    na(int(p["scaffold.braid_layers"]) < 2, ["fit.layer_spacing_mm"], "one braid layer")
    na(p["bundle.rotation_distribution"] != "custom", ["bundle.rotation_profile"], "custom only")
    na(float(p["scaffold.crimp_amplitude_um"]) == 0, ["scaffold.crimp_wavelength_um"], "amplitude 0")
    na(float(p["native.crimp_amplitude_um"]) == 0, ["native.crimp_wavelength_um", "native.crimp_profile"], "amplitude 0")
    abut = p["scaffold.engagement"] == "abut"
    na(abut, ["scaffold.overlap_tib_mm", "scaffold.overlap_fem_mm", "fit.am.overlap_tib_mm", "fit.am.overlap_fem_mm",
              "fit.pl.overlap_tib_mm", "fit.pl.overlap_fem_mm"], "abut has no overlap")
    na(p["fit.partition"] != "per_bundle", ["fit.am.overlap_tib_mm", "fit.am.overlap_fem_mm", "fit.pl.overlap_tib_mm", "fit.pl.overlap_fem_mm"], "per-bundle units only")
    na(p["fit.partition"] == "per_bundle", ["scaffold.overlap_tib_mm", "scaffold.overlap_fem_mm"], "set per unit")
    for s in ("tib", "fem"):
        na(p["fp.%s.shape" % s] != "crescent", ["fp.%s.curvature_mm" % s], "crescent only")
    mode = p["lig.section_mode"]
    na(mode != "area_aspect", ["lig.area_profile", "lig.aspect_profile"], "area/aspect mode only")
    na(mode == "area_aspect", ["lig.width_profile", "lig.thickness_profile"], "derived")
    na(mode in ("per_bundle", "imported"), ["lig.width_profile", "lig.thickness_profile", "lig.shape_profile"], "derived")
    lobe = [k for k in BY_ID if k.startswith(("am.", "pl.")) and k.split(".")[1] in ("width_profile", "thickness_profile", "shape", "orientation_deg", "shift_profile")]
    na(mode != "per_bundle", lobe + ["bundle.lobe_overlap_mm"], "per-bundle sections only")
    na(mode == "per_bundle", ["bundle.am_fraction_mode", "bundle.am_fraction_profile", "bundle.am_fraction", "bundle.interface_curvature"], "set by the lobes")
    na(p["bundle.am_fraction_mode"] != "profile", ["bundle.am_fraction_profile"], "profile mode only")
    na(p["bundle.am_fraction_mode"] == "profile" and mode != "per_bundle", [], "")
    na(mode != "imported", ["lig.section_contours"], "imported only")
    na(float(p["lig.contour_irregularity_mm"]) == 0, ["lig.contour_lobes", "lig.contour_seed"], "irregularity 0")
    na(p["lig.length_mode"] != "target", ["lig.length_target_mm"], "target mode only")
    na(p["lig.length_mode"] == "target", ["lig.bow_anterior_mm"], "solved")
    struct = [k for k in BY_ID if k.startswith(("am.", "pl.")) and k.split(".")[1] in
              ("fascicle_diameter_scale", "fascicle_rotation_deg", "fiber_rotation_deg", "dispersion_deg", "flare_tib_mm", "flare_fem_mm")]
    na(p["bundle.structure_mode"] != "independent", struct, "independent structure only")
    na(p["bundle.structure_mode"] == "independent", ["native.fascicle_rotation_deg", "native.fiber_rotation_deg",
                                                      "native.dispersion_deg"], "set per bundle")
    for k, why in layer_inactive(p).items():
        out.setdefault(k, why)
    return out
