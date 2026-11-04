# Imaging workspace and Muse controller

Open `http://127.0.0.1:8000/` after `python tools/server.py` (or `tools/run_workspace.sh`, which uses `.venv`
and opens no browser tab). This is the single MRI-first product page; `/workspace` redirects there. The workspace loads the designer script in the
same page, so the scaffold controls, the 3D view and Muse share **one** design state.

Bundlr does not preload a scaffold and imply that it belongs to a patient. Without a saved state link it opens the
MRI upload dialog and waits for KneeScan readings. Run `python tools/setup_kneescan.py` once; the application server
then starts and monitors that separate local reader automatically. Synthetic cases remain explicit demo fixtures.

Publication-derived measurement ranges are non-blocking for MRI imports. Outlier measurements are preserved in the
case provenance; when the current parametric anatomy cannot represent one directly, the exploratory design uses its
attachment-derived geometry instead. File integrity, finite numeric data, and coordinate-transform validation remain
hard requirements.

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

The designer itself displayed no image volumes and had no MPR, windowing or segmentation editing. The integrated
workspace now uses a gray/white, V7-inspired clinical-instrument shell; MRI panes remain near-black for diagnostic
contrast, while navigation, validation state and scaffold provenance use a restrained teal/violet status language.

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
- **Muse**: control registry built from the live app (all six scaffold levels, anatomy/injury/pose,
  display, visibility, cutaways, sections, components, imaging and workspace); tool definitions generated
  from it; coverage check (0 unregistered controls at the time of testing; **Muse coverage** view). One transaction per request:
  atomic validation, one undo step, one preview + one final rebuild, slider/provenance/history updates, change
  highlight, change card with before/after, sources, assumptions, validation and separately measured interpretation /
  validation / preview / final-rebuild times. Stale responses, duplicates, kept settings and read-only derived values
  are refused; a failed rebuild restores the previous design. "Undo that change", "Keep the geometry but restore
  the previous view", "Show which settings came from the MRI" work. Ambiguous targets and goals without an objective
  ("optimize healing") get one focused question.
- **Fresh design inspection and grounded response loop**: current revision, case, parameters and provenance,
  selected and visible components, layout/camera, measurements, imaging evidence, validation, custom definitions and
  recent actions are read at execution time. Muse receives actual action results in a second response pass; state
  questions are revision-labelled. The built-in fallback executes common edits and navigation without pretending to
  be model inference.
- **Parametric custom parts**: tube, curved tube, band, sleeve, channel array, strut and lattice primitives are stored
  as editable definitions with name, purpose, dimensions, material, model-frame placement, level, assumptions and
  relationship metadata. They appear in **Custom components**, can be renamed, repositioned, edited, duplicated,
  focused, isolated, deleted and undone, and persist in the saved application state. Referenced parts track the
  current target origin on rebuild. Construction is restricted to audited data definitions; arbitrary code is never run.
- **External evidence**: a server-side PubMed E-utilities connector accepts generalized topics only, rejects patient
  identifiers, and returns title, authors, year, journal, PMID, DOI, clickable PubMed/PMC URL, abstract passage and
  access type. Search results are action results, never instructions, and do not modify the design automatically.

### Needs metadata or model access

- **Running segmentation** calls the configured KneeScan service (`KNEESCAN_URL`, default `http://127.0.0.1:7860`).
  Its weights are in a private Hugging Face repo (`tariksyed/kneescan`, needs `HF_TOKEN`), so it could not be run here;
  the workspace reports the service as unreachable and creates nothing. A KneeScan results JSON (e.g. from the Colab
  notebook) can be imported instead and goes through the same validated mapping.
- **Muse Spark model**: the workspace defaults to Meta Model API (`https://api.meta.ai/v1`) and
  `muse-spark-1.3`. Copy `.env.muse.example` to `.env.muse` and set `MODEL_API_KEY`; the local config is gitignored.
  `MUSE_API_URL`, `MUSE_MODEL`, and `MUSE_TIMEOUT` remain available as overrides. Without the key, the UI accurately
  says **API key required** and uses the labelled deterministic parser. The patient-details field is optional,
  session-only context sent with the instruction; use de-identified information for demos. Explicit user-provided
  values may map to supported controls, but Muse is instructed not to diagnose or invent scan findings.
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
- Arbitrary custom meshes, perforation/Boolean operations, flared custom transitions and automatic design variants.
  Path conformance, collision/intersection prevention, clearance solving, contact, fusion and bond strength are not
  modeled. Non-independent custom parts track a target origin only, and the inspector labels that limitation.
- Custom tube, curved-tube, band, sleeve, channel-array, strut and lattice definitions are included as physical
  geometry in the STL workflow with their model-frame positions and rotations. Export supports full scaffold,
  structural-level, component and custom-only selection; combined distinct-shell or separate-part packages; explicit
  millimetre bounds; validation; optional limited face cleanup; and the recreatable design JSON. Visual grooves,
  zero-thickness interfaces, schematic anchors and magnified display effects are listed as omitted rather than claimed
  as printable geometry. STEP and Boolean fused-solid export remain unavailable because the engine has no validated
  CAD-solid/Boolean kernel. Published evidence is PubMed metadata/abstract plus a PMC link when available; the app
  does not scrape journal full text or classify study design automatically.

## Measured performance (this machine, synthetic phantom, browser pane in the background)

Interpretation by the built-in parser 0–7 ms; validation ≤1 ms; a 3-level scaffold edit: preview 1.9 s, final
rebuild 4.4 s (engine 1.2 s; the rest is network/3D update while the page was in a background pane); evidence
transfer (length + width profile): preview 1.5 s, final 4.8 s (engine 2.1 s). Display, camera, overlay and layout
changes do not rebuild geometry. These are single measurements, not benchmarks.

A live Muse Spark connection test completed successfully, and a real model-selected sheath-wall edit completed with
registered tool arguments, a deterministic preview/final build, passing fit checks, and a second grounded response.
Provider latency varies and is displayed separately from validation and geometry timing. See `DEMO_RUNBOOK.md` for
the repeatable presentation path and fallback language.
