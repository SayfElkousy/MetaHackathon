# Patient input: `acl-mri-input` v1.0.0

How the external MRI-analysis model hands patient-specific data to the geometry engine.

```
MRI → analysis model → acl-mri-input JSON (+ mesh / mask files)
    → adapter (acl_engine/v2/patient.py) → design state → geometry engine (acl_engine/v2/scene.py)
    → designer / viewer (manual refinement) → export (zip: GLB, STL, design.json, history)
```

The geometry engine never calls the analysis model. It reads only this document, so either side can change without touching the other.

## Entry points

| Where | How |
|---|---|
| Python | `acl_engine.v2.patient.import_file(path)` → `(state, report)`, `import_document(dict, base_dir)` |
| HTTP | `POST /api/v2/import` with `{"case": "<fixture dir>"}` or `{"document": {...}, "assets": {"name.stl": "<base64>"}, "mode": "replace"|"merge", "current": <state>}` |
| Browser | **Open / import** → choose the JSON and every file it references |
| MRI upload | `/upload`: a scan goes to the KneeScan reader (`mri_analysis/app.py`), and `tools/mri_bridge.py` turns its results into this document, imports it and opens the design (see README, "MRI upload") |
| Merge | `merge_import(current_state, new_state, "merge")` keeps manual overrides and returns conflicts |

The import is rejected (HTTP 422 with a list of problems) when the schema version is unsupported, a transform is not affine, units are wrong for a quantity, a referenced file is missing, or a value breaks a parameter's limits. Nothing is silently clamped.

## Conventions

- **Units**: lengths in `mm` (also `cm`, `m`, `um`), angles in `deg` (also `rad`). Each quantity can state its own `units`.
- **Frames**: points and meshes are in the **image frame** unless they say `"frame": "model"`. `imaging.image_to_model` (4×4 affine, mm) maps image points into the **model frame**: knee-local, +X lateral (for that knee), +Y anterior, +Z proximal, origin at the tibial plateau centre. Encode the image orientation (RAS/LPS) in this matrix. Left knees need no special handling: the engine models every knee lateral = +X and mirrors X only for display and export.
- **Positions along the ligament**: `s` runs from 0 (tibial end) to 1 (femoral end). Profiles can set `"direction": "femur_to_tibia"`.
- **Large data** is referenced, not embedded: `{"$ref": "tibia_surface.stl", "format": "stl", "units": "mm", "frame": "image"}`. Supported: meshes `.stl` (binary/ASCII) and `.obj`; masks `.npy` (0/1 voxels) with `voxel_to_image` (4×4).

## Quantities, provenance and uncertainty

Any leaf can be a bare value or a quantity:

```json
{"value": 17.0, "units": "mm", "status": "measured", "confidence": 0.85,
 "uncertainty": {"sd": 1.2}, "region": "AM", "source": "seg-v3", "note": "..."}
```

`status` is one of `measured`, `inferred` (model estimate, e.g. fibre angles), `proposed` (design suggestion), `missing`, `unsupported`, `awaiting_review`. The engine adds `override` (manual edit), `literature` (cited generic default) and `default` (engine default). Missing values stay visibly **missing** in the designer and use a labelled generic value. They are never presented as patient data.

For each parameter the design state keeps: value, units, source and model version, run id, input path, status, confidence, uncertainty and region. The imported dataset is stored separately (`state.imported`) from the user's edits (`state.params`), so every parameter can be reset to its imported value, imported and edited designs can be compared, and the full edit history is exported (`history.json`).

## Document structure

| Block | Fields (all optional except `imaging.laterality`) | Engine parameter / asset |
|---|---|---|
| top | `schema`, `schema_version` (1.x), `case_id`, `synthetic`, `source{model_name, model_version, run_id, created}`, `notes` | case info |
| `imaging` | `units`, `voxel_spacing_mm`, `orientation`, `coordinate_convention`, `image_to_model` (4×4), `laterality` (`left`/`right`), `acquisition_pose.flexion_deg` | `case.side`, `knee.flexion_deg` |
| `bone` | `femur_surface`, `tibia_surface` ($ref meshes) | replace the procedural bones |
| | `scale.bicondylar_width_mm`, `tibial_plateau_width_mm`, `notch_width_mm`, `posterior_tibial_slope_deg` | `bone.*` (procedural bones) |
| | `landmarks{name: point}` | stored in `assets.landmarks` |
| `attachments.tibial` / `.femoral` | `center` (point), `shape{kind, length_mm, width_mm, orientation_deg, curvature_radius_mm}`, `boundary` (point list → footprint polygon), `area_mm2` (recorded), `bundles{AM{center}, PL{center}}` | `fp.tib.*`, `fp.fem.*`, `bundle.boundary_angle_deg` |
| `ligament` | `centerline` (points), `length_mm` (recorded for comparison), `width_profile`, `thickness_profile`, `shape_profile`, `orientation_profile` (`{s:[...], values:[...]}`, any resolution), `flare_tibial_mm`, `flare_femoral_mm`, `bundles.am_area_fraction`, `bundles.am_area_fraction_profile`, `bundles.boundary_uncertainty_mm` | `lig.*`, `bundle.am_fraction`, `bundle.am_fraction_profile`, `bundle.interface_uncertainty_mm` |
| | `section_contours`: `[{s, points: [[x,y,z], ...], frame?, units?}]`, measured cross-section outlines (≥ 8 points each, image frame unless `frame: "model"`) | `lig.section_contours`; sets `lig.section_mode = imported` (interpolated along the ligament; area, width, thickness are then derived from them) |
| `injury` | `bundles.{AM,PL}.{location_s, gap_length_mm, depth_fraction}` with `involvement: "per_bundle"` (each bundle's own tear; depth 0 = intact bundle) | `injury.am.*`, `injury.pl.*` |
| | `type` (intact/partial/complete), `location_s`, `gap_length_mm`, `tear_plane{obliquity_deg, azimuth_deg}`, `irregularity_mm`, `involvement` (whole/AM/PL/sector), `sector{center_deg, width_deg, depth_fraction}`, `continuity_fraction`, `retraction{tibial_mm, femoral_mm}`, `regional_damage` (profile), `mask` ($ref npy + `voxel_to_image`), `stump_surfaces` ($ref meshes) | `injury.*`; a mask replaces the plane model (the injury region is then the voxels themselves); stump surfaces are shown as observed overlays |
| `structure` | `bundle_rotation_deg` (AM/PL arrangement), `twist_profile`, `fascicle_rotation_deg` and `fiber_rotation_deg` (TOTAL end-to-end rotation within each bundle / each fascicle, signed, surgeon-informed range −30..30°), `orientation_dispersion_deg`, `fascicle_count` | `bundle.*`, `native.*` |
| | `fascicle_angle_deg`, `fiber_angle_deg` (local helix angles, legacy) | recorded only, with a warning: a constant helix angle winds far beyond 30° over a unit, so it is not applied |
| `scaffold_design` | `architecture`, `engagement` (`interdigitate` or `abut`; the retired `sleeve` is imported as `abut` with a warning because it lies outside the ligament envelope), `overlap_tibial_mm`, `overlap_femoral_mm`, `fill_fraction`, `yarn_count`, `yarn_diameter_mm`, `braid_angle_deg`, `braid_pattern`, `fiber_diameter_um`, `regions[{id, material, composition, fiber_volume_fraction, alignment_dispersion_deg, strand_scale, transition_mm, transition_mode}]` | `scaffold.*`, `regions.*` (status `proposed`) |
| `reference` | `contralateral.ligament.{width_profile, thickness_profile}`, `contralateral.registration`, `intact_estimate` | the intact estimate when the injured side lacks a value |

Unknown top-level blocks are kept out of the design and listed as **unsupported** in the import report.

Spatial profiles keep their sampling: a 9-sample width profile stays 9 samples (monotone PCHIP between them) and is edited point by point in the curve editor. It is not averaged onto the default 5 knots.

## Examples

- `fixtures/patients/patient_A_synthetic/input.json`: right knee, complete midsubstance tear, profiles, imported centreline, translated image frame, uncertainty values, scaffold proposal.
- `fixtures/patients/patient_B_synthetic/input.json`: **left** knee, partial AM tear given as a voxel **mask**, tibia **surface mesh** in a rotated + translated image frame, measured femoral footprint **boundary**, contralateral profiles, a value marked `missing`, and an unsupported block.

Both are labelled `"synthetic": true`. They are produced deterministically by `python tools/make_synthetic_patients.py` and contain no patient data.

Minimal document:

```json
{"schema": "acl-mri-input", "schema_version": "1.0.0", "case_id": "example", "synthetic": true,
 "imaging": {"laterality": "right", "image_to_model": [[1,0,0,-118],[0,1,0,42],[0,0,1,-310],[0,0,0,1]]},
 "attachments": {"tibial": {"center": {"value": [-115.5, 52, -310], "units": "mm", "status": "measured", "confidence": 0.8}}},
 "injury": {"type": "complete", "location_s": {"value": 0.5, "status": "measured"},
            "gap_length_mm": {"value": 6, "units": "mm", "status": "measured", "uncertainty": {"sd": 1.5}}}}
```

## Updating with new upstream results

`mode: "merge"` keeps every parameter the user overrode and returns a conflict for each one whose upstream value changed (the designer asks: keep mine / use new). `mode: "replace"` takes the new import as is. Both are recorded in the history.
