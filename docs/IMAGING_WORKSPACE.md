# Imaging workspace and Muse controller

Open `http://127.0.0.1:8000/workspace` after `python tools/server.py` (or `tools/run_workspace.sh`, which uses `.venv`
and opens no browser tab). The previous designer stays at `/`; the workspace loads the same designer script in the
same page, so the scaffold controls, the 3D view and Muse share **one** design state.

Extra Python dependencies for imaging: `nibabel`, `pydicom`, `SimpleITK`, `pillow` (already in
`mri_analysis/requirements.txt`). Tests: `pytest tests/test_imaging.py` and `node --test tests/js/test_ws.mjs`.
All test and demo data are synthetic (`imaging/synthetic.py`, written to `fixtures/imaging/` by
`python tools/make_imaging_fixtures.py`). **Demo** in the imaging browser loads that phantom; it is labelled as such.

## What existed before

| Input | Where | Calibration |
|---|---|---|
| DICOM series (.zip, folder, .dcm) | KneeScan's own loader (`mri_analysis/kneescan/io.py`) only | pixel + slice spacing; orientation used only to pick sagittal |
| NIfTI | KneeScan loader | spacing; reoriented with nibabel |
| `.npy`, KneeMRI `.pck` | KneeScan loader | none |
| Masks / model outputs | KneeScan returns 2D outlines (ACL on one sagittal slice, bones on one slice) as JSON; the designer imported `injury_mask.npy` for its own knee frame | in KneeScan pixel space |

The designer itself displayed no image volumes and had no MPR, windowing or segmentation editing.

## Library choice

Orientation and reconstruction are delegated to maintained libraries rather than re-implemented:
**SimpleITK** (GDCM `ImageSeriesReader`) reads DICOM series, positions and direction cosines; **nibabel** reads
NIfTI sform/qform and reorients every calibrated volume to the nearest canonical RAS+ grid
(`as_closest_canonical`); **pydicom** reads headers for validation. The browser receives the canonical grid plus its
affine and only slices it (nearest-neighbour, at physical scale). Browser-side MPR libraries (Cornerstone3D, NiiVue)
were considered; they would have needed a bundler or replaced the tabbed layout, and the canonical-grid approach keeps
every coordinate transform in tested Python code.

## Capability status

### Functional (tested)

- **Loading and validation**: DICOM series grouping, ordering by position, duplicate removal, refusal of mixed
  echoes at one position, missing slices detected and filled with marked blank planes (geometry after the gap stays
  correct; cross-checked against SimpleITK), irregular spacing refused, mixed orientations (localizers) excluded,
  inconsistent matrix/spacing refused; NIfTI sform/qform, 4D (first volume), missing units; ordered PNG/JPEG/TIFF
  stacks and `.npy` as **uncalibrated** volumes. Every problem is listed with a hint; malformed input is not loaded silently.
- **Imaging browser**: cases, studies, series with dimensions, spacing, sequence (TR/TE/field), frame, validation
  flags, masks with origin/review status, annotation counts; import with progress and cancel; explicit **Remove…**
  (closing a view never deletes data).
- **MPR**: axial/sagittal/coronal panes, orientation letters computed from the affine, physical proportions with
  anisotropic voxels, nearest-neighbour display, thick-slice notice on reformatted planes, oblique-grid notice,
  missing-slice notice. Preset **MPR + Scaffold**; tabs can be moved between panes, split, resized, maximized and restored.
- **Linked crosshairs** in world RAS mm: click/drag, arrows, PageUp/PageDown, wheel, numeric RAS and voxel entry,
  show/hide, **Center**, **Go to selected annotation**, go to label centroid. Series link only when they share a
  DICOM FrameOfReferenceUID; series can be unlinked; uncalibrated/unoriented series are never linked.
- **Windowing**: drag tool, numeric width/level, reset to the DICOM window (or automatic range), automatic
  1–99.5 percentile range, invert, link/unlink across views. Voxel data are checksummed: display never changes them.
  Intensities are labelled as arbitrary MRI units (never Hounsfield units).
- **Colour maps** (gray default, hot, viridis, cool-warm) with legend, range, reset, and the displayed quantity;
  on MRI intensity the legend says it is not a tissue or damage map. Imported **derived maps** carry their quantity and units.
- **Overlays**: per-label colour (same in all panes), visibility, opacity, global overlay opacity — separate from
  windowing and from 3D materials.
- **Annotation editing**: brush, erase, adjustable size (mm), polygon fill, label selection/addition, undo/redo,
  save as a new version (the prediction is kept; edited voxel count recorded), mark reviewed/rejected.
- **Landmarks and distances** computed server-side from the affine; reviewed status; moving a reviewed point voids
  the review; uncalibrated distances are in voxels and cannot be reviewed as millimetres.
- **Quality control** (file/metadata, image–mask alignment, geometry, human review; pass/warn/fail/not evaluated):
  empty masks, undefined labels, border contact, disconnected parts, slice gaps, abrupt area changes, single-slice
  (2D) outlines, model-transform check, missing spatial metadata. Flags with a location navigate the crosshair.
  No accuracy score; model confidence is labelled as the model's own estimate.
- **Model output mapping**: KneeScan outlines are mapped from KneeScan's own index space (NIfTI and DICOM
  conventions verified against its real loader) into the volume, with the offset and out-of-image points reported,
  stored as a 2D outline with model identity and settings, editable, never shown as 3D. KneeScan's length endpoints
  become a draft (unreviewed) measurement.
- **Evidence → scaffold**: report extraction (explicit statements only), explicit mapping table (ACL length, width,
  tear gap, footprint dimensions, tear type/bundle; tear location as a labelled assumption), certainty kept apart
  (measured / reported / inferred / assumption), conflicts surfaced (between sources and with values you typed),
  engine limits enforced, modes respected. Each transfer shows source series, annotation, value, units, target and
  whether it is direct or assumption-based. Not mapped (by design): coatings, grooves, channels, braid angle,
  composition, pretension, attachment *positions*.
- **Muse**: control registry built from the live app (≈400 controls: all six scaffold levels, anatomy/injury/pose,
  display, visibility, cutaways, sections, components, imaging, workspace; 22 actions); tool definitions generated
  from it; coverage check (0 unregistered controls at the time of testing; **Muse coverage** view). One transaction per request:
  atomic validation, one undo step, one preview + one final rebuild, slider/provenance/history updates, change
  highlight, change card with before/after, sources, assumptions, validation and separately measured interpretation /
  validation / preview / final-rebuild times. Stale responses, duplicates, kept settings and read-only derived values
  are refused; a failed rebuild restores the previous design. "Undo that change", "Keep the geometry but restore
  the previous view", "Show which settings came from the MRI" work. Ambiguous targets and goals without an objective
  ("optimize healing") get one focused question.

### Needs metadata or model access

- **Running segmentation** calls the configured KneeScan service (`KNEESCAN_URL`, default `http://127.0.0.1:7860`).
  Its weights are in a private Hugging Face repo (`tariksyed/kneescan`, needs `HF_TOKEN`), so it could not be run here;
  the workspace reports the service as unreachable and creates nothing. A KneeScan results JSON (e.g. from the Colab
  notebook) can be imported instead and goes through the same validated mapping.
- **Muse language model**: set `MUSE_API_URL` (OpenAI-compatible base URL), `MUSE_API_KEY`, `MUSE_MODEL`. Without
  them the built-in deterministic parser is used (and labelled). The model path was implemented but not exercised
  against a live model here.
- **Orientation/calibration** depend on the input: PNG/JPEG stacks and `.npy` have none (viewing only; no mm,
  no labels, no linking). NIfTI with sform/qform codes 0 has spacing but no orientation. NIfTI series are not linked
  to each other (no verifiable frame identity).
- **Model mapping** is refused for multi-frame DICOM, NIfTI without orientation codes, and DICOM series from which
  duplicates were dropped (KneeScan's slice numbering would differ).

### Not implemented

- 3D volume/surface rendering of MRI or masks; scaffold overlaid on MRI (no MRI↔engine registration exists, so the
  two are shown side by side and labelled **unregistered**).
- Registration between series in different frames; DICOM SEG / RTSTRUCT import; probability/confidence volumes
  (KneeScan produces none).
- Contour interpolation between slices; 3D brush; per-label opacity in the 3D scaffold view.
- Mapping MRI to microscopic structure, materials or pretension (deliberately excluded: no justified method).

## Measured performance (this machine, synthetic phantom, browser pane in the background)

Interpretation by the built-in parser 0–7 ms; validation ≤1 ms; a 3-level scaffold edit: preview 1.9 s, final
rebuild 4.4 s (engine 1.2 s; the rest is network/3D update while the page was in a background pane); evidence
transfer (length + width profile): preview 1.5 s, final 4.8 s (engine 2.1 s). Display, camera, overlay and layout
changes do not rebuild geometry. These are single measurements, not benchmarks.
