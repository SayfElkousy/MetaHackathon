# Bundlr v2: historical implementation and verification report

> This report records the earlier v2 engine milestone. For the current integrated imaging workspace, Muse Spark
> controller, capability limits, and verification status, see `IMAGING_WORKSPACE.md` and `DEMO_RUNBOOK.md`.

Run: `python tools/server.py` → designer at http://127.0.0.1:8000/ , separate viewer at `/view?state=<id>`, previous page at `/classic`.
Engine: `acl_engine/v2/`. Tests: `pytest` (68 tests). Patient input: `docs/PATIENT_INPUT.md`.

## 1. The empty-canvas bug

**Cause.** `setCam()` in the old page measured the model with `Box3.expandByObject()` immediately after adding a root group that was scaled ×1000 (metres → mm) and rotated, but **before three.js had updated that group's world matrix**. `expandByObject` only updates the child's matrix, so the box was computed with the parent's stale identity transform: about 1000× too small. The camera was then placed about 0.1 mm from the centre, inside a bone, so the canvas showed flat grey surfaces and no model. It "worked" after pressing Fit because by then a frame had been rendered and the matrices were current. Reproduced in headless Chrome (`docs/screenshots/before_first_load.png`).

**Fix.**
- Geometry stays in mm (no scaling).
- Every bounds query calls `updateMatrixWorld(true)` first.
- The camera frames the ligament plus both bone ends (bone bounds clipped to the ligament's height ± 10–12 mm), and near/far planes follow the scene size.
- Rendering is on demand after every change.
- The renderer is sized from the real container (ResizeObserver).
- Loading shows progress steps and errors with actions. If the 3D library cannot load, the page says so.
- After the first framing the user's camera is kept; **Fit model** re-frames.

**Separate tab.** The design state is persisted to a server store (`state_store/<id>.json`, content-addressed) and the URL carries `?state=<id>`. The viewer tab (`/view?state=<id>`) needs nothing from the opening tab. It follows designer edits through a BroadcastChannel. An unknown id shows "This saved model is not available" with three recovery actions.

## 2. Geometry (engine `acl_engine/v2`)

| Part | Implementation |
|---|---|
| Bones (`bones.py`) | Distal femur (condyles, arched intercondylar notch with inclined roof, trochlear groove, epicondyles, metaphysis) and proximal tibia (sloped plateau, medial concavity, two tibial spines, PCL fossa, tuberosity), as implicit shapes meshed with a vectorized surface-nets mesher. Femoral and tibial cartilage (oval facets). **Labelled "assumed": procedural approximation**, scaled by bicondylar / plateau width, notch width and slope. Imported patient meshes (STL/OBJ) replace them. Exact distance queries keep attachments on bone and measure penetration. |
| Footprints (`ligament.Footprint`) | Oval, crescent (L14) or measured polygon in the bone's tangent chart, projected onto the bone (max deviation ≈ 0 mm), AM/PL split coloured. |
| Envelope (`ligament.Ligament`) | Spline centreline between points offset half a thickness from each bone (the body runs alongside the notch wall), rotation-minimizing frames (no flips, check `frame_continuity`), width / thickness / squareness / orientation profiles (any sample count), ribbon flattening toward the femur that turns parallel to the notch wall (L02), insertion flares that blend each section into its footprint (L04, L19). Endpoint correspondence: each material position maps to a fixed footprint position (automatic rotation alignment), so strands cannot cross arbitrarily. |
| Hierarchy | envelope → AM/PL bundle regions (area fraction, boundary direction) → fascicle groups (hexagonal / random / radial packing, laid out in true mm) → representative fibers (twist, crimp, dispersion) → localized detail view at true crimp scale. Count, diameter and packing fraction are linked: two are chosen, the third is computed and shown. |
| Twist (separate channels) | (1) bundle-region rotation about the centreline; (2) fascicle angle to the local centreline tangent within each bundle; (3) fiber angle to the fascicle tangent; (4) scaffold braid angle to the scaffold axis, with pitch computed from tan α = 2πR/p (or the reverse); handedness for each. |
| Injury (`tissue.Injury`) | Spatial region: tilted (obliquity/azimuth), irregular tear faces, missing length, retractions, affected region (whole / AM / PL / sector with depth), residual continuity, regional damage profile; or an **observed voxel mask**. Retained tissue = everything outside the region; remnants and the missing volume are closed meshes. A partial tear keeps one continuous remnant. |
| Scaffold (`scaffold.py`) | Domain = affected region × fill fraction, from the tibial tear face − overlap to the femoral face + overlap. **Bone-interface phases are added only when a remnant is shorter than the bone-reach limit** (the scaffold then fans onto the footprint). Architectures: **braided** (± families on an arc-length-parametrized tube following the section shape; flat-topped over-under offset along the true surface normal, so crossings are separated by construction; 1/1 or 2/2; continuous sleeve over the remnants with axial core yarns filling the gap, each ending at its local tear face) and **aligned** (fascicle-like yarns, mm-true hexagonal layout checked at the thinnest station). Engagement: sleeve / interdigitate (between retained fascicles) / abut (conformal end at the local tear face). Where the remnant lies against the notch wall the sleeve is impossible; the braid ends conformally there and says why. |
| Multiphasic regions | tibial interface / bridge / femoral interface. Each has: material, normalized composition, fiber volume fraction (porosity = 1 − Vf, inter-fiber pore size computed), alignment dispersion, strand-diameter scale, transition width, graded/abrupt. Region weights drive yarn diameter and fiber dispersion along the scaffold. Region ids go into the GLB (`_REGION` attribute, node extras), per-region STL files and `regions.json`. **Material and composition are metadata**; no strength, degradation, healing or re-tear model exists, and the checks list them as "not evaluated". |
| Validation (`scene.run_checks`) | footprints on bone, ligament and scaffold clear of bone, frame continuity, packing feasibility, fascicle overlap, closed edge-manifold solids, partial tear preserves unaffected tissue, scaffold spans every yarn's local tear faces, achieved overlaps, yarn–yarn clearance, yarn–tissue clearance, braid jamming, core-yarn fit, the bone-interface rule, input warnings, provenance. |

## 3. Interface

- **Design tab:** the 6 required groups, all generated from the parameter registry. Each control has units, value, numeric entry, a reset to the imported value or default, a provenance chip with a tooltip (source, confidence, uncertainty, input path), its description and literature source, and messages from failing checks.
- **Editing:**
  - Out-of-range numbers are not applied, and the message appears under the control.
  - Dependent values are disabled and show their computed value (sizing mode, braid mode, settings inactive in the current mode).
  - Curve editors handle the profiles, a boundary editor handles footprint polygons, and 3D drag handles move the footprint centres and centreline controls (snapped onto bone).
  - Composition editors normalize to 1.
- **Inspect tab:** per-object visibility and opacity with status chips; colour by anatomy, braid family or **status** (observed / assumed / estimated / proposed); fiber render mode; display-only density and crimp magnification (they do not change the design hash); exploded view (badge shown); bone cutaway; clipping plane; cross-section at any s (2D); thin-slab 3D section; selection (click a fascicle, fiber or yarn); detail view; legend; performance numbers.
- **Data & history tab:** case and source, imported vs current table with per-parameter reset, import report (mapped / missing / unsupported / warnings, status counts), edit history, downloads.
- **Builds:** previews while dragging, then a full build; stale results are dropped (sequence numbers plus aborts). Generation is deterministic: the design and geometry hashes are reproduced exactly from a saved `design.json`.

## 4. Verification (measured in this session)

Environment: macOS, headless Chrome (puppeteer-core 23.11.1, SwiftShader software GL), local server, Python 3.15 + NumPy/SciPy.

| Test | Result |
|---|---|
| Initial viewer renders the model | yes. Old page: blank (`before_first_load.png`); new: framed knee (`after_first_load.png`) |
| Separate tab opened from the app | renders in 1.95 s, geometry hash equals the designer's |
| … refreshed | 2.3 s, same hash |
| … loaded directly in a fresh browser profile | 4.6 s, same hash |
| … unknown id | recovery overlay with 3 actions; "Open the synthetic demo" loads the model |
| Viewer follows designer edits | yes (BroadcastChannel) |
| Injury edit moves the repair | tear 0.55 → 0.42: scaffold 7.6–24.0 mm → 6.0–19.8 mm, tibial stump 12.6 → 8.3 mm |
| Partial tear keeps unaffected tissue | `partial_preserves` pass (17–20 unaffected fascicle groups continuous) |
| Controls change their parameters | automated audit over every scalar/int design parameter: all change geometry or reported design values, except 3 that are inactive in the demo configuration (and marked inactive in the UI) |
| Material assignments survive export | GLB nodes carry region/material extras and `_REGION`; per-region STL; regions.json |
| Saved parameters reproduce the model | design.json from the export rebuilds to the same geometry hash |
| Invalid values / meshes / intersections | out-of-range values rejected (422, message at the control); closed-solid audit; yarn and tissue clearance; incompatible yarn counts detected |
| Two synthetic patients differ | A: right, complete tear (plane model), braided, L 32.1 mm; B: left, partial AM tear from the voxel mask (58 % of the section), aligned yarns, imported tibia mesh, L 31.2 mm |
| Coordinate alignment | patient B's tibia STL (rotated + translated image frame) lands on the reference after the transform: median distance 0.006 mm |
| Missing input handling | B's missing femoral footprint length shows as "missing"; the unsupported block is listed |
| Manual override, reset, merge | override recorded and shown in the imported-vs-current table; merge keeps it and reports conflicts; reset restores the imported value |
| Initial load | 2.3–4.5 s from navigation to the framed model (warm server). The first build after a server start adds bone meshing, which a warm-up thread now does at startup |
| Regeneration | edit → full model on screen 1.4–2.1 s (engine 1.2–1.8 s, 3D objects 38–46 ms); preview engine 0.6 s |
| Interactive rate | about 58–67 fps with about 650 k triangles (software GL) |
| Memory | GPU geometries constant (13) over 30 rebuilds; JS heap after forced GC 55 → 29 → 38 MB (no growth trend) |
| Test suite | 68 passed (47 existing + 21 new in `tests/test_v2.py`); `check.py` PASS |

Screenshots: `docs/screenshots/` (before_*, after_*, patient_*). Matching viewpoints: anterior and lateral.

## 5. References

- **Parrilli et al. 2024** (Surg Radiol Anat 46:249–258, CC BY 4.0, PMC10861685; source L19). Read: abstract through discussion, captions of Figs 3–6 and 8; Figs 3–5 viewed. Used qualitatively: double-bundle organization, fan-like femoral extension along the lateral wall of the notch, root-like branched tibial insertion, **parallel bundles in extension** (default within-bundle twist 0°), cartilage covering the condyles. **No dimensions are reported** (n = 2 specimens from one donor), so none were taken from it. The Springer page redirected to a login; the PMC copy was used.
- **arXiv 2608.18012** (Cheng et al. 2026, L21). Only the abstract could be read (the PDF did not extract). It describes footprint-centre identification from 3D MR images (error 2.1 mm image-based). Not applied, because no weights or code are available. Its output (a footprint centre) maps directly onto `attachments.femoral.center` in the input schema.
- **Shively clinic page** (L20). Readable, but the rendering is not described in text and no tear classification is given. No findings were taken from it.
- Dimensional defaults come from the project's existing sources: L01 (length, width), L02 (ribbon thickness), L03 (inclinations: model gives 46.5° / 74.2° vs 47.4° / 75.3°), L04 (insertion area), L11/L12/L16 (bundle split), L14 (femoral crescent), L15 (tibial oval 11 × 17 mm).
- No openly licensed distal femur / proximal tibia mesh with known scale was found in a search (msk-STAPLE is CC BY-NC and needs segmentation; others unclear). The bones are therefore procedural and labelled as such.

## 6. Limitations

- **Bones and cartilage are procedural approximations**, not scans. Menisci are not modelled; the PCL is a schematic landmark tube.
- **Flexion** is a rigid illustrative rotation of the femur. The ligament is regenerated between the moved attachments, with no kinematics or tension.
- **Fascicle groups and representative fibers are visualization density.** Physical counts are reported (for example about 220 000 collagen fibers), not drawn. Fascicles are round tubes, so where the section flattens they can come closer than at mid-length (overlap is checked at mid-length only). Within-bundle twist uses a bundle-shaped mapping that is not rigid.
- **Scaffold feasibility:** some combinations fail their checks and are reported, not hidden. Examples: patient A's proposed braid (−0.15 mm yarn contact), the default braid with the tear moved to s ≈ 0.30 or 0.65, and anchoring into a crescent femoral footprint. Interdigitation has little room at 0.5 fascicle packing. Small yarn–tissue contacts (≤ 0.3 mm, < 2 % of samples) appear as warnings in most cases.
- **Masks** are `.npy` voxels plus an affine (no NIfTI/DICOM reader). Imported-mesh distances are approximate (nearest dense sample, point-to-plane).
- **UI:** curve editors move values, not sample positions; drag handles exist for footprint centres and centreline controls, not for bundle subregions; selection picks fascicles, fibers and yarns, not meshes. three.js loads from cdn.jsdelivr.net (needs internet).
- **No mechanical, degradation, healing or re-tear model.** Material settings are design metadata.

## 7. Scaffold containment, cross-sectional fit and bundle-specific sizing (3A)

### What was wrong
Measured on the previous version (same demo cases, strand radius included):

| Demo | Strand samples outside the ACL envelope | Worst protrusion |
|---|---|---|
| Complete midsubstance tear (braid) | 81% | 2.3 mm |
| Proximal tear (bone-anchored) | 21% | 2.3 mm |
| Partial AM tear | 0% (but not checked against the AM/PL boundary) | — |

Causes: the default "sleeve" engagement deliberately wrapped the braid *outside* the remnants and across the gap; distances were laid out in stretched material coordinates, not millimetres; the strand radius and over-under depth were not part of the fit; the old section view drew yarns on an independent schematic. Before/after at matching viewpoints: `screenshots/3A_before_*.png` vs `screenshots/3A_after_*.png` (anterior, lateral, along the ACL, and the section at s = 0.5).

### Separate representations (all in the viewer, the sections and the export)
- **Intact envelope** (`intact_reference`): the target ACL, measured or a labelled template.
- **AM / PL regions** (`bundle_am_reference`, `bundle_pl_reference`): exact lofts of each bundle's outline, plus the **boundary uncertainty band** (`bundle_interface`, ± `bundle.interface_uncertainty_mm`) and the bundle paths.
- **Retained tissue** (remnant meshes, retained fascicles) and **damaged volume** (`injury_region`).
- **Repair space** (`repair_space[_am|_pl]`): the permitted scaffold volume = affected region × (damaged interval + permitted overlap), reaching bone only when a remnant is shorter than the bone-reach limit.
- **Scaffold**: strands (actual material) and region domains (material assignment volumes inside the repair space).

### Section model
Every station along the ligament carries a closed contour (mm) and an AM/PL interface. Material coordinates are mapped to each bundle region by a ruled map (interface ↔ chord, outer arc ↔ arc), so AM and PL tile the section exactly and a fascicle never changes bundle. Irregular or measured outlines are applied as a radial warp (one-to-one). Cross-section definitions: width+thickness (area derived from the contour), area+aspect (width/thickness derived), per-bundle lobes (envelope = union of the AM and PL lobes, which always overlap by the contact depth, so there is no gap), or imported contours. Added controls: length (target mode solves only the anterior bow; attachments and bones stay locked, and the free/locked degrees of freedom are reported), size scale, proximal/distal taper, irregularity, AM fraction along the ligament, interface angle along the ligament, interface curvature, footprint AM share and split rotation per footprint, per-bundle lobe dimensions/position/orientation, and per-bundle structure (fascicle diameter, within-bundle rotation, fiber angle, dispersion, transition lengths). Per-bundle tears (location, length, depth) are an injury option.

### How the scaffold is fitted
For each repair unit (whole section, one bundle, sector or mask region) a millimetre distance field is rasterized at stations every 0.25 mm. It is the distance to the nearest constraint minus that constraint's clearance: ligament boundary, AM/PL interface (single-bundle units), affected-region edge, cross-sectional extent, retained fascicles, and bone. A strand of radius r fits where the field is at least r, and every layout is required to fit at **every** station of its span.
- Aligned yarns: positions are chosen so the margin and spacing hold at all stations.
- Braids: each layer is the region at depth r + over-under, opened by the tightest bend the crossings can follow. The spacing is checked at every position along the braid, the real 3D clearance is measured, and the tube is re-rounded if strands touch.
- Yarn ends that abut a tear stop at the remnant surface (tear face or nearest retained fascicle).

Nothing is thinned, removed or re-sized to force a fit. A design that does not fit is reported, with fixes that were each rebuilt and checked before being offered.

The "sleeve" engagement was retired (it is outside the envelope by definition); older designs are migrated to "abut" with a note.

### Fit checks (measured on the full 3D scaffold)
Tolerance 0.02 mm; raster 0.05 mm; strands sampled every 0.2 mm.
- **Strand volume outside the ligament envelope:** signed distance to the envelope surface for every strand sample. Bone-anchored ends inside the insertion are reported separately as a permitted attachment interface.
- **Strand volume outside the repair region:** per station; includes the AM/PL boundary for single-bundle units.
- **Strands crossing into the other bundle; contacts between AM and PL units; samples beyond the permitted overlap.**
- **Clearance:** minimum to the ligament boundary and to bone, and clearance shortfalls (warning).
- **Contacts:** yarn–yarn, and yarn–retained-fascicle (unintended tissue overlap).
- **Yarn ends:** gap to the remnant, and misalignment.
- **Occupancy along each unit:** strand area ÷ repair-space area. A porous scaffold is not expected to fill the space.
- **Connectivity:** yarn pieces and interlaced braid groups.

The design badge shows "design valid" only when no check fails. Violating strand samples are drawn red in 3D and outlined red in sections. An invalid design shows a banner with the failed checks, the tested fixes, and "Restore last valid design".

### Sections
Every section is sliced from the same meshes and strands the viewer draws and the export writes. Strands appear as their true elliptical cuts. Available planes:
- perpendicular to the ACL centreline;
- perpendicular to the AM or PL bundle path;
- axial, sagittal or coronal planes at a chosen offset from the ACL midpoint.

Each section is labelled with its position, its orientation (compound anatomical directions for oblique axes) and a scale bar. Tissue fills are clipped to the exact envelope cut.

### Verified (this session)
Each case was built at full quality and also checked independently against the section contour of the nearest station. In every passing case, 0 strand samples were outside the envelope, the strand volume outside the envelope and outside the repair region was 0 mm³, and the strands kept 0.29–0.50 mm from the boundary (0.15 mm requested). Passing cases:
- complete tear with a braid
- partial AM tear
- proximal tear
- large template (× 1.4)
- area/aspect sections with taper
- per-bundle repair units
- per-bundle tears
- sector tear
- tear near the femur
- tear near the tibia
- patient B (mask injury, mesh tibia, left knee)

Cases correctly reported as not fitting, each with fixes that were tested:
- **Small template (× 0.7):** the braid doesn't fit.
- **Irregular contour (0.6 mm):** the 16-carrier braid jams.
- **Per-bundle lobes:** the braid jams.
- **14 mm gap with 2 braid layers:** layer 2 doesn't fit.
- **30 interdigitating yarns:** only 21 fit.
- **The old default (24 carriers × 0.55 mm):** it jams.
- **Patient A's proposed 24-carrier braid:** it jams. 14 carriers or 0.31 mm yarns fit.

In the browser, an invalid design showed the banner and three tested fixes, and applying a fix returned "design valid".

Tests: 90 automated tests pass (`tests/`), including:
- containment for 8 configurations;
- a partial tear kept inside its bundle;
- a design that doesn't fit is reported with its requested values kept, and its fixes verified;
- the section map is one-to-one, with AM and PL tiling every section;
- area derived from the contour;
- a target length keeps the attachments locked;
- the footprint AM share is applied;
- sections match the built geometry;
- retired-sleeve migration;
- importing measured contours and per-bundle tears;
- the control audit, in which every active control changes the geometry, metrics or checks.

Speed (warm, this machine):
| Build | Preview | Full |
|---|---|---|
| Braided default | 1.8 s | 3.9 s |
| Aligned yarns | 1.1 s | 2.1 s |

Testing fixes runs as a separate background request, about 5–20 s for a design that doesn't fit.

### Limitations of 3A
- The in-section fields use the body cross-section. Inside an insertion flare (bone-anchored units only) the final 3D envelope and bone checks are the authority. Braid yarns that converge there are trimmed back from their bone end, and the distance short of the footprint is reported.
- The braid is a geometric model: crossing separation, bend radius and spacing are enforced and measured. Yarn deformation and flattening are not modelled.
- The per-bundle lobe model and the interface are templates unless the MRI model supplies contours and bundle data.
- The surgeon-informed rotation constraint was supplied afterwards and is implemented in §8.
- Full braided builds are slower than before (about 2 s → about 4 s) because every braid layer is measured and regenerated when strands touch.

## 8. Surgeon-informed fiber rotation limit (±30° per unit)

**Rule (as reported by the user from the surgeon's guidance):** within each cylindrical structural unit that contains fibers, the fibers rotate no more than 30° in total around the unit's local longitudinal axis over the unit's whole length. The rotation is signed, so the range is −30° … +30°, and the default is 0°. A design rule, not an anatomical limit on whole-ACL torsion, knee rotation or the AM/PL arrangement.

**What was wrong before:** fiber twist was a constant local helix angle, which winds far past 30° over a unit. Measured on the generated paths of the default design:

| Unit | Rotation before |
|---|---|
| Native fibers in each fascicle (2° helix) | up to 303° |
| Fibers inside each scaffold yarn (12° helix) | up to 464–570° |

The scaffold-yarn fiber frame was also built against a fixed world axis, so fibers turned with the braid's winding: the parent's rotation applied a second time.

**Units and controls:** every control is labelled "Total fiber rotation along unit (°)", with the explanation "End-to-end rotation around the local axis; surgeon-informed design range: −30° to +30°." It also names its unit:

| Unit | Rotating content | Control |
|---|---|---|
| Native bundle (AM, PL) | fascicles about the bundle path, relative to the bundle region | `native.fascicle_rotation_deg`, or `am.` / `pl.fascicle_rotation_deg` (independent structure) |
| Native fascicle group (each one) | fibers about the fascicle axis | `native.fiber_rotation_deg`, or `am.` / `pl.fiber_rotation_deg` |
| Aligned scaffold repair unit (whole, AM, PL, sector) | yarns about the unit axis | `scaffold.yarn_rotation_deg` |
| Scaffold yarn (each one) | fibers about the yarn axis | `scaffold.fiber_rotation_deg` |

The old helix-angle and handedness controls were removed; the sign of the total rotation now gives the direction. Values outside −30 … +30 are rejected, never clamped. Older designs are migrated with a note. The importer accepts `structure.fascicle_rotation_deg` and `structure.fiber_rotation_deg`, and records legacy helix angles with a warning instead of applying them.

**How it is generated:**
- **Spread:** rotation runs linearly from 0 to the total over each unit's own length, so it is monotone. It is never applied repeatedly to successive segments.
- **Fibers inside fascicles and yarns:** placed in a rotation-minimizing (non-twisting) frame along that fascicle or yarn, so nothing from the parent's motion is inherited.
- **Native bundles:** rotation is applied relative to the bundle region. The schedule is corrected against the measured result so the true rotation in millimetres matches the request even where the section map stretches angles. The AM/PL arrangement rotation (`bundle.rotation_total_deg`) is the parent level and is not added a second time.
- **Aligned and core scaffold yarns:** these follow the repair region's centre and size by a similarity transform (uniform scale plus rotation), so the turn is exactly the requested angle. They do not take on the ligament's own section turn. They run straight into bone at anchored ends, ending at the bone surface instead of being redistributed over the footprint or pushed sideways along it.
- **Orientation dispersion:** a radial wander made of whole half-waves, zero at both ends of the unit. It scatters fiber orientation but creates no net or back-and-forth rotation.

**How it is validated:** every full build measures the generated paths, at true scale, in a rotation-minimizing frame along each unit's axis. Angles are unwrapped along each path, so 400° is reported as 400°, not 40°. Each unit's rotation is the rigid turn shared by its paths (their mean). Back-and-forth is measured as the excursion outside the start–end interval. Single-path spread is reported separately. For native bundles the reference is each fascicle's parent-only path, so only the bundle's own rotation counts. Check statuses:
- **Fail:** any unit total beyond ±31° (1° measurement tolerance).
- **Warn:** the measured value differs from the setting by more than 3°, or back-and-forth exceeds 3°.

The Checks tab lists every unit.

**Verified (this session, measured / requested):**
- **Native bundles:** +20.0° / +20°, +30.1° / +30° (AM, independent), −14.8° / −15° (PL, independent).
- **Native fascicles:** −30.0° / −30° (mean over 20 AM and 16 PL fascicle units; largest unit −30.4°).
- **Aligned scaffold unit:** +25.0° / +25°, −30.0° / −30°, −20.0° / −20° (bone-anchored proximal tear).
- **Scaffold yarn fibers:** +30.0° / +30°.
- **Back-and-forth:** below 1° everywhere.
- **No double application:** with the AM/PL arrangement turned 60° and every unit set to 0°, contents measured 0° (the bundle regions themselves turn 63–75°).
- **Browser:** typing 35° is refused ("35 is outside −30–30 deg; not applied").
- **Tests:** 7 new automated tests (97 in total).

**Consequences and open question:**
- **Braids (decided by the user: warning only):** a braid's yarns wind around the repair unit's axis by construction (about 70–106° in the demos). When this exceeds ±30° the build shows a warning, listed in Checks with ⚠ in the rotation table, but the design stays valid. The rule is enforced as a hard check on the fibers inside each braid yarn.
- **Fit:** straight (non-turning) aligned yarns no longer follow the retained fascicles or a narrow region as those turn with the ligament ribbon. Fewer yarns may therefore fit when interdigitating or in thin crescent-shaped regions (for example 15 of 30 instead of 21). This is reported, with tested fixes.
- **Native anatomy:** the rule constrains rotation within each native unit relative to its region. The ligament's own section orientation between the attachments is whole-ACL geometry and is reported, not limited.

## 9. Six scaffold levels as the editing structure

The Design tab now has two panels:

- **Anatomy and Pose:** the knee pose, the tear definition, the reference anatomy (bones, footprints, ligament shape, AM/PL arrangement) and the native tissue.
- **Scaffold Levels:** six expandable sections. Every scaffold control belongs to exactly one of them (registry fields `level`, `component`, `neighbors`, `basis`; checked by `tests/test_layered.py`).

The default construct (`scaffold.arch = layered`, engine `acl_engine/v2/layered.py`) is built as explicit geometry. It spans footprint to footprint:

| Level | Geometry it changes | Controls (abridged) |
|---|---|---|
| 1 Surface | coating layer inside the treated lumens (narrows the open lumen when explicit); nano-texture at true scale in the magnified detail | treated surfaces, coverage, texture pattern/orientation/period/amplitude, coating thickness, explicit vs metadata, coating composition |
| 2 Micro | groove paths in channel lumens and/or on the inner sheath; true groove profile in the detail and in `stl/detail_grooved_channel_segment.stl` | placement, coverage, width, depth, ridge, bottom radius, orientation to the fiber path, minimum residual wall; outer sheath smooth by default; braid never grooved |
| 3 Fascicle | hollow channels (closed annular solids with open entrances), shared transition channels, optional ports | lumen, wall (outside diameter and pitch are calculated), spacing, count (max fit or per region), shape, waviness, curvature, connections, wall composition, assumed degradation |
| 4 Bundle | braided mantles around the AM-like and PL-like channel groups (1/1 or 2/2 over/under) | per region or both (linked): strands, strand diameter, braid angle, pattern, yarn twist, envelope (auto or manual), slack, composition, assumed modulus; transition width |
| 5 Whole ligament | ribbon sheath (wall, clearance), explicit pores | construct type, sizing (fit to anatomy, profiles, fit to contents), width/thickness/flattening/orientation along the length, wall, clearance, composition, pores; permeability and degradation as entered assumptions |
| 6 Ends | flared ends blending onto the scaffold footprints, graded material domains, footprint outlines, optional plug | per end or both: footprint offset/rotation/outline/size, flare length/base/curvature, endpoint distribution and stop distance, bone-facing zone, gradient length, three compositions, fixation plug |

**How the levels connect:**
- Channels pack inside each reinforcement envelope at the narrowest station. The braid wall (2 × strand diameter) sits outside the envelope, and the sheath clearance and wall sit outside the braid.
- Mantle centres follow the anatomical AM/PL bundle centroids, so the attachments and knee pose drive their paths. The sheath orientation does not.
- In each flare, every point blends onto its landing point on the scaffold footprint; the graded composition follows the distance from bone.
- The flare base is laid out through the footprint's own outline (a crescent footprint gets a curved base), in both the calculated and the manual flare mode.
- Switching a calculated value to user-set (sheath profiles, envelope size, channel counts, footprint size, flare base) copies the calculated values in, so the geometry does not jump. The copied values are marked user-set. A manual envelope sets the mid-length size and keeps the fitted shape along the length.
- Conflicts are measured on the generated geometry and reported in every level they involve, each with an explanation and concrete fixes. The fixes are pre-tested in the background (`/api/v2/layer_fixes`) and applied only on click, setting only the parameters they name. Requested values are kept: a channel that does not fit is drawn in red.

**Viewer:**
- Selecting a level highlights it.
- *Isolate* shows it alone; *Neighbours transparent* shows the adjacent levels faintly.
- A cross-section inset dims the other levels.
- A magnified detail (`/api/v2/level_detail`) shows the level at true scale with its enlargement labelled.

**Each control shows:**
- what it controls and its units;
- the component it modifies and the neighbour levels it may affect;
- whether the value is user-set or calculated;
- a basis selector (measured / assumed / experimental, stored in provenance).

Each level has a reset button and lists its calculated outputs separately from the entered assumptions. After an edit, each level shows what changed, so consequences in neighbouring levels are visible.

**Limitations:**
- Material properties are assumptions, not models. Outputs computed from them (stiffness estimate, Darcy flux, remaining mass) are labelled "not validated". Drawn pores do not change the permeability value.
- The femoral notch wall runs alongside the ligament, so the construct turns sharply onto it and channels on the inside of that turn are pressed together. The default femoral endpoint distribution is therefore **staggered**: each channel ends where it would come closer to a neighbour than the requested spacing, measured on the generated paths at the same station and the stations just before it. With the default design, 10 of 59 channels reach the 0.5 mm stop and the rest end 2.7–4.4 mm from bone; Level 6 reports the count and range. With "fan" endpoints the crowding is reported as a conflict with fixes. Changing the femoral flare length, its curvature or an oval scaffold footprint changes the count, but the turn itself remains. A rotation-interpolated flare was tried and rejected: paths folded across stations and the sheath entered bone.
- The inner sheath is the outer superellipse inset by the wall thickness (an approximation of a true offset).
- Grooves are drawn as representative paths at whole-ligament scale; their true profile is in the detail view and export.
- The earlier gap-filling yarn scaffold remains available as the construct "braided" or "aligned"; its controls are listed under the level that owns them.
