# Prompt for Muse: next milestones

Copy everything below the line into a new Muse chat. Attach the project zip (without `.venv`
and `outputs`) and paste the latest `check.py` summary underneath it.

---

Attached is the current working project (zip) and the latest `python check.py` summary.
It passes 22 tests. A slider rebuild takes about 0.2 s, and an export about 0.15 s.
Fixes are marked `FIX` and are described in REPORT.md. Read REPORT.md first.

## Rules (each was broken in an earlier milestone)

1. **Do not reprint or rewrite cli.py, export.py, glb.py, schema.py or paths.py.** In T1 and N3,
   "unchanged pipeline" rewrites brought back fixed bugs: condyles inside the ACL, a missing
   STL triangle corner (three times), `PhaseField.constant_interval()` which does not exist,
   and dropped checks. Instead, send **new functions in new modules**, plus an exact
   **insertion note** for existing files ("in cli.py run_request, after the line
   `pen, penloc = containment_of_points(...)`, add: ...").
2. **Only call functions that exist in the attached zip.** Check each name and signature
   against the attached files. If something you need is missing, say so instead of guessing.
3. **Vectorize.** No per-point or per-triangle Python loops on arrays (paths.py and
   export.py were just rewritten for this, which made rebuilds 4-7x faster). Budget: preview
   rebuild under 0.5 s, export under 5 s, preview GLB under 15 MB.
4. **Keep every old design file working.** `starter_request`, `pass_request`,
   `starter_fiberlength` and `acl_context_request` must build with unchanged numbers unless
   the new feature is switched on.
5. **For each new parameter**, add a schema field with a default that leaves today's geometry
   unchanged, a slider in `tools/web/sliders.json`, a hint in `tools/web/hints.json` for each
   new check, and a test.
6. **Cite every anatomical default** with a source id in `sources.json`. Also send entries
   L11-L18, which you have listed as "registered" twice but never sent.
7. Report results as PENDING_RUN. Never state numbers you have not computed.

## The problem to solve

Today the scaffold is three smooth round tubes running in a straight braid from tunnel to
tunnel. It doesn't look like, or work like, something that joins torn ligament tissue.
The real situation is a torn ACL: a **femoral stump** and a **tibial stump**, often with
ragged, oblique ends and a gap between them. A repair-bridge scaffold has to:

- **span the gap** with a load-bearing braided core,
- **grip and blend into each stump**: ends that open into a sleeve, cuff or fan of fibers
  that overlaps and interlaces with the remaining tissue (the fields `scaffold.role`,
  `end_overlap_femoral_mm`, `end_overlap_tibial_mm`, `taper_length_mm` and `conform_ends`
  already exist in schema.py, but have no geometry),
- **have real fiber structure**: strands made of yarns made of fibers (the `hierarchy`
  block exists but raises "not_implemented"), so the model shows the texture of a
  textile, not plastic tubes,
- **sit in a realistic ACL and knee**, so the fit can be judged.

Do **milestone A only** in this reply. B-E are listed so your design for A leaves room for
them.

---

## Milestone A: realistic ACL and torn stumps (R1 + T3)

1. **Two bundles** in `context.py` (new functions, old ones kept): anteromedial (AM) and
   posterolateral (PL), each with its own attachment points, length and cross-section,
   drawn as two overlapping see-through meshes `acl_ghost_am` and `acl_ghost_pl`. Cite
   defaults (Duthon L01, Harner L04, or new).
2. **Twist:** apply `anatomical_twist_total_deg` and `twist_distribution` (stored in the
   schema today but unused) along the ligament.
3. **Ribbon shape:** flat and wide near the femur, rounder mid-substance, widening fan at
   the tibia (Smigielski L02, Duthon L01).
4. **Footprints:** crescent-shaped femoral footprint on the lateral wall of the notch, broad
   oval tibial footprint between the tibial spines, as thin meshes `footprint_femoral` and
   `footprint_tibial`.
5. **Tear geometry (T3)**, using the existing `tear` fields: for `severity` partial or
   complete, cut the ACL mesh at `location_s` with `obliquity_deg` and `azimuth_deg`. Shift
   the ends by `retraction_femoral_mm` and `retraction_tibial_mm`, giving meshes
   `stump_femoral` and `stump_tibial` with a slightly irregular (ragged) cut surface. Report
   `gap_mm` (shortest distance between stumps) and stump lengths in `context.json`. Add
   presets: intact, proximal complete, midsubstance complete, partial AM.
6. **Setting `anatomy.realism`:** `"schematic"` (default, today's output unchanged) or
   `"anatomical"`.
7. Containment, coverage and the GLB use the new shapes when `realism` is `"anatomical"`.
8. **Sliders:** realism toggle, twist, tear severity, tear location, obliquity and retraction.
9. **Tests:** old fixtures unchanged; bundles inside the overall ACL; the stump gap matches
   the retraction; cut faces are closed meshes (watertight).
10. Describe what `tools/render.py` should show from 3 angles. We will check the pictures.

## Milestone B: repair-bridge scaffold (T4)

1. **Roles** from `scaffold.role`:
   - `replacement`: today's tunnel-to-tunnel braid.
   - `repair_bridge`: the braid spans the stump gap plus `end_overlap_*_mm` into each stump,
     with no bone tunnels.
   - `augmentation`: a slimmer braid running alongside a preserved bundle, placed by
     `sector_center_deg` and `offset_mm`.
2. **End anchors**, chosen per end by a new field `end_style`:
   - `fan`: the 3 strands unbraid and split into N splayed fiber bundles that spread over
     the stump surface, following the ACL's flare.
   - `sleeve`: a tapered braided tube (more, finer strands) that slides over the stump,
     with `taper_length_mm`.
   - `suture_loops`: whipstitch-style loops through the stump.
   Transitions between braid and anchor are smooth (no kinks; use the existing
   rotation-minimizing frames).
3. **Optional central core** (`core.enabled`, `core.diameter_mm` exist): a straight or
   lightly twisted core fiber through the braid centre, with `minimum_cover_mm` checked.
4. **New checks:** anchor overlap length on each stump; anchor stays inside the stump surface
   (no fibers floating in the joint); minimum bend radius at the transitions; gap fully
   bridged.
5. **Tests**, and sliders for role, end style, overlap, taper length and fan count.

## Milestone C: fiber hierarchy (T5)

1. **Levels:** strand → yarns → fibers, e.g. the 3×3×6 braid-twist preset (Freeman L06).
   Yarns twist around each strand centreline at a set helix angle; fibers twist inside each
   yarn.
2. **Level of detail:** preview draws yarns only; export draws fibers. Stay within the
   budgets in rule 3. Generate tubes in batch with numpy, and use one mesh per level so the
   GLB stays small.
3. **Report** fiber count, total fiber length, an approximate porosity (open volume ÷
   envelope volume) and pore-size estimate, next to the targets in `porosity`.
4. **Checks:** fibers don't overlap within a yarn; yarns don't overlap within a strand.
5. **Sliders:** yarns per strand, fibers per yarn, yarn twist angle, fiber diameter.

## Milestone D: realistic bones and knee bending (R2)

1. Replace the box and ovals with real femur and tibia surface meshes from an openly
   licensed knee model. State the license and source, and keep the mesh under 5 MB.
2. Map the mesh into our frame (+X lateral, +Y anterior, +Z proximal, origin at the tibial
   plateau centre) using named landmarks. Document the mapping.
3. Bone clearance and tunnel placement use the real surface. Keep the schematic knee as a
   fallback.
4. **Knee flexion 0-90°:** femur, ACL and scaffold move together. Report how ACL length and
   angle change with flexion.

## Milestone E: clean-up

1. `sources.json`: send L11-L18 in full.
2. `requirements.txt`: list only packages the code imports. The last two milestones added
   trimesh, manifold3d, pygltflib and scikit-image, which nothing uses.

## What to send back (for A now)

- New files in full; insertion notes for existing files (rule 1).
- A file manifest: new / changed-by-insertion / unchanged.
- Tests added, marked PENDING_RUN.
- Any name you could not find in the zip.
