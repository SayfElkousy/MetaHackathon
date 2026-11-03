"""Generate the two SYNTHETIC patient datasets used to demonstrate the MRI-input
integration (no real patient data). Deterministic: re-running reproduces the files.

    python tools/make_synthetic_patients.py

Writes fixtures/patients/<case>/input.json plus referenced assets:
- patient_A_synthetic: right knee, complete midsubstance tear, sampled width /
  thickness profiles, imported centreline, image frame = model frame shifted by
  a translation (exercises point transforms).
- patient_B_synthetic: LEFT knee, partial AM tear given as a voxel MASK, tibia
  surface mesh (STL) in a rotated + translated image frame, measured femoral
  footprint boundary, contralateral profiles, uncertainty values, and a value
  marked "missing" (exercises meshes, masks, laterality, missing inputs).
"""
import json
import math
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from acl_engine.v2.bones import ProceduralBone, cached_mesh  # noqa: E402
from acl_engine.v2 import design as D  # noqa: E402
from acl_engine.v2.ligament import Ligament  # noqa: E402
from acl_engine.v2.tissue import Injury, envelope_samples  # noqa: E402

OUT = os.path.join(ROOT, "fixtures", "patients")


def q(v, units=None, status="measured", conf=None, sd=None, note=None, frame=None):
    d = {"value": v, "status": status}
    if units:
        d["units"] = units
    if conf is not None:
        d["confidence"] = conf
    if sd is not None:
        d["uncertainty"] = {"sd": sd}
    if note:
        d["note"] = note
    if frame:
        d["frame"] = frame
    return d


def rigid(rz_deg, t):
    a = math.radians(rz_deg)
    T = np.eye(4)
    T[:3, :3] = [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
    T[:3, 3] = t
    return T


def to_image(P, T_img2model):
    Ti = np.linalg.inv(T_img2model)
    return P @ Ti[:3, :3].T + Ti[:3, 3]


def write_stl(path, V, F):
    tri = V[F].astype(np.float32)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(F.shape[0], dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"] = n
    rec["v"] = tri
    with open(path, "wb") as f:
        f.write(b"synthetic tibia (procedural, NOT patient data)".ljust(80, b" "))
        f.write(np.uint32(F.shape[0]).tobytes())
        f.write(rec.tobytes())


def patient_a():
    case = "patient_A_synthetic"
    d = os.path.join(OUT, case)
    os.makedirs(d, exist_ok=True)
    T = rigid(0.0, [-118.0, 42.0, -310.0])     # image -> model: translation only
    # geometry in the model frame, written in image coordinates
    tib_c = np.array([[2.5, 10.0, 0.0]])
    fem_c = np.array([[8.5, -14.0, 23.0]])
    s = np.linspace(0, 1, 9)
    cl = np.array([[2.5 + 6.0 * t, 10.0 - 24.0 * t + 1.2 * math.sin(math.pi * t), 2.8 + 18.5 * t] for t in s])
    doc = {
        "schema": "acl-mri-input", "schema_version": "1.0.0", "case_id": case, "synthetic": True,
        "notes": "SYNTHETIC demonstration case. Not patient data.",
        "source": {"model_name": "synthetic-generator", "model_version": "0.1", "run_id": "A-0001", "created": "2026-11-01"},
        "imaging": {"units": "mm", "voxel_spacing_mm": [0.4, 0.4, 0.8], "orientation": "RAS",
                    "coordinate_convention": "image physical mm (RAS)", "image_to_model": T.tolist(),
                    "laterality": "right", "acquisition_pose": {"flexion_deg": q(10.0, "deg", conf=0.9)}},
        "bone": {"scale": {"bicondylar_width_mm": q(82.0, "mm", conf=0.95, sd=1.0),
                           "tibial_plateau_width_mm": q(79.0, "mm", conf=0.95, sd=1.0),
                           "notch_width_mm": q(20.5, "mm", conf=0.8, sd=1.2),
                           "posterior_tibial_slope_deg": q(8.5, "deg", conf=0.8, sd=1.5)}},
        "attachments": {
            "tibial": {"center": q(to_image(tib_c, T)[0].round(3).tolist(), "mm", conf=0.85, sd=1.0),
                       "shape": {"kind": "oval", "length_mm": q(18.5, "mm", sd=1.2), "width_mm": q(11.5, "mm", sd=1.0),
                                 "orientation_deg": q(6.0, "deg", "inferred", conf=0.6)}},
            "femoral": {"center": q(to_image(fem_c, T)[0].round(3).tolist(), "mm", conf=0.8, sd=1.5),
                        "shape": {"kind": "crescent", "length_mm": q(17.0, "mm", sd=1.5), "width_mm": q(8.5, "mm", sd=1.0),
                                  "curvature_radius_mm": q(12.0, "mm", "inferred", conf=0.5)}}},
        "ligament": {
            "centerline": q(to_image(cl, T).round(3).tolist(), "mm", conf=0.7, note="from the pre-injury estimate"),
            "length_mm": q(33.5, "mm", "measured", sd=1.5),
            "width_profile": q({"s": s.round(3).tolist(), "values": [12.8, 12.0, 11.2, 10.6, 10.1, 10.4, 11.0, 11.9, 12.8]}, "mm", conf=0.75),
            "thickness_profile": q({"s": s.round(3).tolist(), "values": [6.2, 6.0, 5.8, 5.5, 5.2, 4.8, 4.4, 4.0, 3.7]}, "mm", conf=0.6),
            "bundles": {"am_area_fraction": q(0.57, "", "inferred", conf=0.5)}},
        "injury": {"type": q("complete", status="measured", conf=0.9), "location_s": q(0.52, "", conf=0.8, sd=0.05),
                   "gap_length_mm": q(6.5, "mm", sd=1.5), "tear_plane": {"obliquity_deg": q(14.0, "deg", "inferred", conf=0.5),
                                                                          "azimuth_deg": q(30.0, "deg", "inferred", conf=0.4)},
                   "irregularity_mm": q(0.9, "mm", "inferred", conf=0.4), "involvement": q("whole"),
                   "retraction": {"tibial_mm": q(1.0, "mm", "inferred"), "femoral_mm": q(2.0, "mm", "inferred")},
                   "regional_damage": q({"s": [0.0, 0.25, 0.4, 0.6, 0.75, 1.0], "values": [0.1, 0.3, 0.6, 0.6, 0.35, 0.15]}, "", "inferred", conf=0.4)},
        "structure": {"bundle_rotation_deg": q(8.0, "deg", "inferred", conf=0.3), "fascicle_rotation_deg": q(0.0, "deg", "inferred", conf=0.3)},
        "scaffold_design": {"architecture": q("braided", status="proposed"), "engagement": q("abut", status="proposed"),
                            "overlap_tibial_mm": q(5.0, "mm", "proposed"), "overlap_femoral_mm": q(4.0, "mm", "proposed"),
                            "yarn_count": q(24, "", "proposed"), "braid_angle_deg": q(24.0, "deg", "proposed"),
                            "regions": [{"id": "bridge", "material": q("PLLA", status="proposed"),
                                         "composition": q({"PLLA": 0.75, "PCL": 0.25}, status="proposed")}]},
    }
    with open(os.path.join(d, "input.json"), "w") as f:
        json.dump(doc, f, indent=1)
    return d


def patient_b():
    case = "patient_B_synthetic"
    d = os.path.join(OUT, case)
    os.makedirs(d, exist_ok=True)
    T = rigid(12.0, [55.0, -20.0, -140.0])     # image -> model: rotation about Z + translation
    # tibia surface: procedural tibia with this patient's size, written in IMAGE coords
    p = D.default_params()
    p.update({"bone.tibia_plateau_width_mm": 71.0, "bone.tibial_slope_deg": 9.5})
    tib = ProceduralBone("tibia", p)
    V, F = tib.mesh(1.0)
    write_stl(os.path.join(d, "tibia_surface.stl"), to_image(V, T), F)
    # femoral footprint boundary (model frame -> image)
    fem_c = np.array([8.0, -15.0, 23.5])
    ang = np.linspace(0, 2 * np.pi, 14, endpoint=False)
    bnd = np.stack([np.full(ang.size, 8.4), fem_c[1] + 3.6 * np.cos(ang) * 1.0, fem_c[2] + 7.2 * np.sin(ang)], 1)
    doc = {
        "schema": "acl-mri-input", "schema_version": "1.0.0", "case_id": case, "synthetic": True,
        "notes": "SYNTHETIC demonstration case (left knee). Not patient data.",
        "source": {"model_name": "synthetic-generator", "model_version": "0.1", "run_id": "B-0001", "created": "2026-11-01"},
        "imaging": {"units": "mm", "voxel_spacing_mm": [0.4, 0.4, 0.4], "orientation": "LPS",
                    "coordinate_convention": "image physical mm", "image_to_model": T.tolist(),
                    "laterality": "left", "acquisition_pose": {"flexion_deg": q(0.0, "deg")}},
        "bone": {"tibia_surface": {"$ref": "tibia_surface.stl", "format": "stl", "units": "mm", "frame": "image", "status": "measured"},
                 "scale": {"bicondylar_width_mm": q(73.0, "mm", sd=1.0), "notch_width_mm": q(17.5, "mm", conf=0.7),
                           "tibial_plateau_width_mm": q(71.0, "mm")}},
        "attachments": {
            "tibial": {"center": q(to_image(np.array([[3.0, 8.5, 0.0]]), T)[0].round(3).tolist(), "mm", conf=0.8),
                       "shape": {"kind": "oval", "length_mm": q(15.5, "mm"), "width_mm": q(10.0, "mm")},
                       "bundles": {"AM": {"center": q(to_image(np.array([[1.5, 12.5, 0.0]]), T)[0].round(3).tolist(), "mm", "inferred")},
                                   "PL": {"center": q(to_image(np.array([[4.5, 5.0, 0.0]]), T)[0].round(3).tolist(), "mm", "inferred")}}},
            "femoral": {"boundary": q(to_image(bnd, T).round(3).tolist(), "mm", conf=0.6, note="segmented footprint outline"),
                        "shape": {"length_mm": q(None, "mm", "missing")}}},
        "ligament": {"width_profile": q({"s": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], "values": [11.0, 10.0, 9.2, 9.0, 9.8, 11.2]}, "mm", conf=0.6),
                     "bundles": {"am_area_fraction": q(0.52, "", "inferred", conf=0.5)}},
        "injury": {"type": q("partial"), "location_s": q(0.68, "", conf=0.6), "involvement": q("AM", conf=0.6),
                   "gap_length_mm": q(3.0, "mm", "inferred"), "continuity_fraction": q(0.15, "", "inferred", conf=0.3),
                   "mask": {"$ref": "injury_mask.npy", "format": "npy", "voxel_to_image": None, "status": "measured"}},
        "reference": {"contralateral": {"ligament": {"thickness_profile": q({"s": [0.0, 0.25, 0.5, 0.75, 1.0], "values": [5.6, 5.3, 4.8, 4.2, 3.6]}, "mm", conf=0.7)},
                                        "registration": np.eye(4).tolist()}},
        "scaffold_design": {"architecture": q("aligned", status="proposed"), "engagement": q("abut", status="proposed"),
                            "yarn_count": q(14, "", "proposed"), "yarn_diameter_mm": q(0.28, "mm", "proposed"),
                            "overlap_tibial_mm": q(4.0, "mm", "proposed"), "overlap_femoral_mm": q(3.0, "mm", "proposed")},
        "unknown_block_from_newer_model": {"note": "kept for the import report (unsupported)"},
    }
    # ---- synthetic damage mask: build the case without the mask, take the AM
    # injury region, voxelize it in the IMAGE frame (0.4 mm voxels)
    from acl_engine.v2.patient import import_document
    tmp = json.loads(json.dumps(doc))
    tmp["injury"].pop("mask")
    with open(os.path.join(d, "tibia_surface.stl"), "rb"):
        pass
    st, rep = import_document(tmp, d)
    if st is None:
        raise SystemExit("synthetic B import failed: %s" % rep["errors"])
    pp = st["params"]
    from acl_engine.v2.scene import _load_bone
    tibb = _load_bone(st["assets"]["tibia_mesh"], "tibia", 0.0)
    femb = ProceduralBone("femur", pp, 0.0)
    lig = Ligament(pp, femb, tibb, "full")
    inj = Injury(pp, lig, {})
    S, M, P, Dm = envelope_samples(lig, inj, "full")
    Pd = to_image(P[Dm], T)
    h = 0.4
    lo = Pd.min(0) - 2 * h
    idx = np.floor((Pd - lo) / h).astype(int)
    shape = idx.max(0) + 2
    mask = np.zeros(shape, np.uint8)
    mask[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
    np.save(os.path.join(d, "injury_mask.npy"), mask)
    aff = np.eye(4)
    aff[:3, :3] *= h
    aff[:3, 3] = lo + 0.5 * h
    doc["injury"]["mask"]["voxel_to_image"] = aff.tolist()
    with open(os.path.join(d, "input.json"), "w") as f:
        json.dump(doc, f, indent=1)
    return d


if __name__ == "__main__":
    print(patient_a())
    print(patient_b())
