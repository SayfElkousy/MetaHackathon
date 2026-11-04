# What Muse built, and what was wrong

Checked 30 October 2026 by running every file. Python 3.13, NumPy 2.

## Summary

Muse has written about 1,900 lines over three milestones, and the core math is
right. When run, its braid paths match every reference number (phase, all 9
crossing positions, strand lengths to a millionth of a mm, gap 0.930 mm, bend
radius 3.673 mm), and the knee placement hits the requested ACL angles exactly
(47.4° and 75.3°).

But Muse has never run any of it. It can't in that chat. So each milestone
arrived with bugs that would have stopped it from working. **After 14
fixes, all 13 tests pass and both designs build in about 3 seconds each.**

## What Muse built

| Milestone | What it does | Status |
|---|---|---|
| M1 | Design file checking, braid paths, 3D tubes, STL file | Worked after 4 fixes |
| N1 | Fold depth/sharpness, fiber-length control, curved axis, gap and curvature checks | **Code never shared.** Rebuilt by Claude to match what N2 expects |
| N2 | Schematic knee, native ACL ghost (angled per published MRI data), scaffold placement, tunnels, fit checks, 3D scene file | Worked after 10 fixes |

## The problems, in plain language

### 1. Muse can't test its own work (the root cause)
Every report says "Executed code: no." Muse writes code the way you'd write an
essay without proofreading. It's mostly right, but small mistakes slip through,
and each milestone builds on the last, so mistakes stack up.
**What to do:** run `python check.py` after every milestone and paste the summary
back. Or use Muse Code, which can run things itself.

### 2. The same bug came back
M1's file writer forgot the third corner of every triangle, so it crashed. That
was fixed and Muse was told about it. N2 wrote a *new* copy of the file writer
with the **same bug**. Muse doesn't reliably remember earlier fixes.
**What to do:** tell Muse to reuse existing working functions instead of
rewriting them. Attach the current project zip each time.

### 3. A whole milestone's files went missing
N2 said "N1 files are unchanged, keep your copies," but those copies never
existed here. Muse only sends files that changed, so the project only works if
you saved every earlier file correctly.
**What to do:** after each milestone, copy every file Muse lists into this
folder, then zip the folder. Start each new Muse chat by attaching that zip.

### 4. Muse's example design failed Muse's own rules
The ACL-in-knee example file was missing a required setting (`transitions`), so
the first run would have crashed with "missing top key transitions."
(My v3 prompt's example also left it out, so this one was partly my fault.)

### 5. Renaming a setting broke older code
N2 renamed `braid_tilt_target_deg` to `braid_angle_deg` but didn't update M1's
code that still used the old name.

### 6. Fiber length came out slightly wrong
Asking for fiber length 1.060 gave 1.063. The ligament zone holds only half a
braid twist, and the steeper zones on either side bleed in at the transitions.
Fixed: the engine now measures the actual result and corrects itself. It gives
exactly 1.060.

### 7. The knee looked wrong
Muse put the two femur bumps (condyles) so close together that the ACL ran
straight through the bone. They're now spaced so the scaffold sits in the notch
between them, with the femoral end on the lateral condyle's inner wall, like a
real ACL.

### 8. The bones would have hidden the scaffold
Bones and the ACL ghost were meant to be see-through, but the 3D file was
missing the setting that turns transparency on. Fixed, and each part now has its
own color.

### 9. Checks that were computed but not used
- The strand-to-strand gap was measured but never put in the pass/fail list.
- The bone-clearance check always said "warn," because the scaffold starts
  exactly on the bone surface.
- Fiber length per zone was computed but never reported.

All three are fixed.

### 10. It was slow
The gap check took 70 seconds because it compared points one at a time. It now
takes under a second, which matters once sliders need instant updates (N3).

### 11. Smaller things
- The box shape used for the tibia had its surfaces facing inward (not visible,
  but technically an invalid solid). Fixed.
- Allowing fiber length exactly 1.0 (a perfectly straight strand) would have
  made the solver hang. Now rejected with a clear message.
- `np.trapz` no longer exists in newer NumPy. It now works on old and new.

## What the results say about the design

These are real outputs, not targets:

| | Starter design | ACL-in-knee design |
|---|---|---|
| Strand gap | 0.93 mm (pass) | 1.22 mm (pass) |
| Tightest bend | R 3.67 mm | R 3.09 mm |
| Fiber length, tunnel zones / ligament | 1.25 / 1.08 | 1.25 / **1.060** (as requested) |
| ACL angles | 47.4° / 75.3° (exact) | 47.4° / 75.3° (exact) |
| **Fits inside the native ACL?** | **No: sticks out 1.87 mm** | **No: sticks out 1.13 mm** |

**Both designs fail the fit check, and that's correct.** The ghost ACL uses a
5 mm thickness (a placeholder), but the braid is 6.4–7.9 mm deep. To fit, the
braid needs less fold depth or thinner strands, or the thickness needs to come
from real MRI. This is the kind of tradeoff the tool is meant to show.

## Known limits (honest, and labeled in the output)

- The knee is boxes and ellipsoids, clearly marked "schematic." For the demo, a
  real knee model would look much better.
- "Flat" cross-section only flattens the ghost, not the braid (planned for N5).
- Footprint coverage is a rough estimate; there's no pass/fail on it.
- Only the three-strand plait exists. The other braid styles are N4–N5.

## What to send Muse next

1. Attach the zip of this folder (the fixed code).
2. Paste the summary from `python check.py`.
3. Say: "This is the current working code with fixes marked `FIX (Claude)`.
   Build on these files. Do not rewrite cli.py's STL writer or any function that
   already passes tests. N1 files here were rebuilt to match your N2 interfaces;
   use them as-is. Next: N3 (local server + slider page). Reuse cli.py's
   pipeline; don't duplicate it."

---

# T1 (Consolidate), applied 30 October 2026

Applied by Claude on top of the fixed code. **19 tests pass; all 4 example designs
build** (results unchanged from before T1).

**Added from Muse's T1:** `records.py` (parameter records), `export.py` (one shared
STL/GLB/JSON writer), `controls.py` (fiber-length closed loop), `migration_report.json`,
new checks (`block_wall`, `tube_mold_feasible`, `fiber_length_<zone>`),
`uses_assumptions`/`overall` in `validation.json`, `cli.run_request()`, Muse's T1 tests,
version 4.0.0.

**Not applied as written, because it would have broken or undone earlier fixes:**
1. The STL writer dropped the third corner of every triangle **for the third time**
   (M1, N2, T1). Fixed in `export.py`, and a test now catches it.
2. `controls.py` and `cli.py` called `PhaseField.constant_interval()`, which doesn't exist.
3. `cli.py` put the condyles back where the ACL runs through bone, and dropped the
   flare and bone-clearance checks.
4. `export.write_glb` lost the colors and transparency. It now reuses `glb.py`.
5. `render.py` drew a grey dot cloud with no knee. The colored 3-view picture is kept.
6. `schema.py`: the newer version already in the folder (tear/scaffold/composition)
   was kept. Muse's T1 copy was older.
7. `requirements.txt` pins were not applied: they add 4 unused packages (trimesh,
   manifold3d, pygltflib, scikit-image) and force older NumPy/matplotlib.

**Missing (message was cut off):** the rest of `sources.json` (L11-L18) and
`fixtures/starter_fiberlength.json`. A stand-in fiberlength fixture was made (starter
design, ligament zone fiber length 1.08). Replace it with Muse's if it sends one.

---

# N3 (Interactive 3D designer), applied 30 October 2026

**22 tests pass, all 4 designs build, a slider rebuild takes about 0.3-1.0 s** (target 1.5 s).
Run it: Run and Debug → "4. Open interactive designer (N3)", or `python tools/server.py`.

**Added from Muse's N3:** `tools/server.py` (local web server), `tools/web/` (page, sliders,
hints), `tests/test_n3.py`, `skip_stl` + per-stage `timings_s` in `cli.run_request`,
speed profile in `check.py`.

**Problems fixed before it would work:**
1. `cli.py` was again a full rewrite, not "only optional kwargs". It brought back the condyles
   inside the ACL, `pf.constant_interval()` (crashes), the always-warning bone check, and dropped
   the flare check. Only `skip_stl` and `timings_s` were merged into the working file.
2. The server could not find the engine when started as `python tools/server.py` (import path).
3. The 3D view had no lights (everything black) and a camera set for mm while the model is in
   metres (model a speck), and it never aimed the camera at the first model.
4. Sliders did nothing or crashed on the older designs (starter, pass): their files use
   `lift_mm`, `braid_tilt_target_deg` and an anatomy "envelope", so the slider fields did not
   exist or were ignored. The server now upgrades field names when a design is opened
   (design unchanged; a test checks this).
5. `hints.json` was written but never used; checks showed raw JSON. Now each failed check
   shows its plain-language hint.
6. Every slider move left a temporary folder behind; STL export errors dropped the connection.
7. Muse's own test would have failed (it required a `fold` field the starter design lacks), and
   its hint test always passed because of a "fallback" escape.
8. `requirements.txt` again lists 4 unused packages. Not applied.

**Still open:** references L11-L18 for `sources.json` (asked for at T1, still not sent).
The page needs internet once to load three.js.

---

# Speed-up, 30 October 2026 (Claude)

Rebuilds are faster with **identical results** (every number and the STL bytes compared
against the previous version):

| | Before | After |
|---|---|---|
| Slider rebuild (preview) | 0.6 s | 0.14-0.22 s |
| Full-quality export | 1.0 s | 0.15 s |

- `paths.py`: `PhaseField.rate_array` / `rate2_array` worked on one point at a time (85% of
  build time). They now work on whole arrays.
- `export.py`: `write_stl` wrote one triangle at a time. It now writes all at once.
- The designer waits 0.12 s after a slider stops moving, instead of 0.25 s.

Also added: `USER_GUIDE.md` (how to use the designer) and `MUSE_NEXT_PROMPT.md` (milestones A-E).
