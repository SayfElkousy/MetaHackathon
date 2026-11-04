# ACL Scaffold Designer: User Guide

This tool designs a braided scaffold meant to replace or bridge a torn ACL. It builds the
scaffold as a 3D model, places it in a simple knee, checks whether it fits and can be made,
and exports a file for 3D printing.

The knee is a simplified model made of boxes and ovals, not patient anatomy. The checks
cover geometry only. Strength and biology are not evaluated.

---

## 1. Start the designer

**In VS Code:** click the **Run and Debug** icon on the left (triangle with a bug), choose
**"4. Open interactive designer (N3)"** from the dropdown at the top, and press the green ▶.

**Or in the terminal** (Terminal → New Terminal):

```
.venv/bin/python tools/server.py
```

Your browser opens at **http://127.0.0.1:8000/**. If it doesn't, open that address yourself.
The first time, it needs internet to load the 3D library.

**To stop:** click in the terminal and press **Ctrl+C** (or the red ■ in VS Code).

> If you see "ports 8000-8010 busy", the designer is probably already running in another
> terminal. Use that one, or stop it first.

---

## 2. The screen

| Area | What it does |
|---|---|
| **3D view** (left) | Drag to rotate. Scroll or pinch to zoom. Right-drag to move the view. |
| **Front / Side / 3/4 / Fit** | Jump to a standard camera angle, aimed at the scaffold. |
| **Checkboxes** under the view | Show or hide the scaffold, the natural ACL outline, the bones and the bone tunnels. |
| **Start from** (top right) | Load a saved design. |
| **Sliders** | Change the design. The model rebuilds in a fraction of a second. |
| **Checks** | Pass / fail / warn for each rule, with a hint when something fails. |
| **Save design / Export STL** | Keep your design, or download the scaffold for 3D printing. |

**Colors in the view:** orange, blue and pink tubes are the three scaffold strands. The faint
red shape is the natural ACL. Pale shapes are bone. Grey cylinders are the drilled tunnels.

---

## 3. What each slider does

The scaffold has three zones: **tibial** (inside the shin-bone tunnel), **ligament** (the
part inside the knee joint) and **femoral** (inside the thigh-bone tunnel).

| Slider | What it changes | Effect you'll see |
|---|---|---|
| **Strand diameter** | Thickness of each of the 3 strands (mm) | Thicker = stronger-looking but bulkier, harder to fit |
| **Swing a** | How far strands move side to side (mm) | Wider braid, more room between strands |
| **Fold depth b** | How far a strand dips to pass under another (mm) | Deeper over/under, thicker braid front to back |
| **Fold sharpness** | 0 = smooth waves, 1 = sharp over/under steps | More "woven" look; strands get closer at crossings |
| **Braid angle tibial / ligament / femoral** | Angle of strands at each crossing, per zone (°) | Higher = tighter braid, more crossings, more fiber |
| **Ligament fiber length** | How much longer each strand is than the ligament zone (1.06 = 6% longer) | Sets crimp/slack in the ligament part |
| **ACL length** | Length of the natural ACL (mm). Studied range 22–41 mm | Moves the femoral end |
| **Sagittal / Coronal inclination** | The ACL's lean, seen from the side and from the front (°) | Tilts the whole scaffold |
| **Knee side** | Right or left knee | Mirrors everything |

**Braid angle vs fiber length:** the ligament zone is controlled by one or the other. Moving
one switches the zone to it, and the other slider fades out. Hover over a faded slider to see why.

**Studied range:** under some sliders, a grey line shows the range reported in published
measurements (source code like L01 in `sources.json`). You can go outside it, but the result
is less realistic.

---

## 4. Reading the checks

| Check | Passes when | If it fails, try |
|---|---|---|
| **Fits inside natural ACL** | No part of the scaffold sticks out of the natural ACL outline | Lower fold depth or strand diameter |
| **Strands don't touch** | The strands never overlap | More swing, or thinner strands |
| **Mold wall ≥ 0.8 mm** | There's at least 0.8 mm between strands, enough for a casting mold wall | More swing, less fold sharpness |
| **Tube mold ≥ 1.7 mm** | Gap is wide enough for the simpler tube mold (a warning, not a failure) | Same as above; otherwise use a block mold |
| **ACL angles match** | Scaffold leans at the angles you set | Should always pass; report it if not |
| **Insertion flare** | The ACL outline widens at the bone by the expected amount | Should always pass |
| **Clears tibia** | The scaffold doesn't cut into the shin bone outside the tunnel | Change the inclination |
| **Fiber length on target** | The ligament fiber length matches the slider within 0.002 | Lower the target or lengthen the zone |
| **Manufacturing / Mechanics / Biology** | Not evaluated yet | — |

A yellow note listing "placeholders" means some inputs are default values, not measurements.

**Common tradeoff:** making the scaffold fit inside the ACL (thinner, flatter) usually shrinks
the gap between strands below the 0.8 mm mold wall. With the current 5 mm ACL thickness,
both pass only with thin strands. For example, start from `acl_context_request`, then set
diameter 1.2, swing 2.4, fold depth 1.5: it fits, with a 1.1 mm gap. At 1.4 mm or thicker,
none of about 300 tested combinations passed both. A thicker ACL from a real MRI may change that.

---

## 5. Saving and exporting

- **Save design** asks for a name (letters, digits, `-` and `_`) and writes
  `fixtures/<name>.json`. It then appears in the **Start from** list. It won't overwrite an
  existing design.
- **Export STL** builds the scaffold at full quality and downloads `scaffold.stl`, in
  millimetres. Open it in your slicer, or preview it at <https://www.viewstl.com/>.

---

## 6. Full check (after every code change)

Run and Debug → **"1. Check everything"** → ▶, or:

```
.venv/bin/python check.py
```

This runs every test, builds every example design, draws `outputs/<name>/knee.png`, measures
rebuild speed, and prints a summary between `=====` lines. **Paste that summary to Muse** with
each new milestone.

Each build writes these files to `outputs/<name>/`:

| File | Contents |
|---|---|
| `knee.png` | Picture from 3 angles |
| `knee.glb` | 3D scene (open at <https://gltf-viewer.donmccurdy.com/>) |
| `scaffold.stl` | Scaffold only, for 3D printing |
| `validation.json` | All checks with values |
| `context.json` | Knee placement numbers |
| `parameters.json` | Every input and computed value |
| `migration_report.json` | How an older design file was upgraded, and which defaults were used |

---

## 7. Working with Muse (the code-writing assistant)

1. Zip this folder (without `.venv` and `outputs`) and attach it to every new Muse chat.
2. Paste the next prompt (see `MUSE_NEXT_PROMPT.md`) and the latest `check.py` summary.
3. When Muse replies, apply the changes and run the checks before you build on
   them. Muse can't run code, and past milestones each had bugs that stopped them working
   (see `REPORT.md`).
4. Before each milestone is applied, the project is backed up to
   `~/Downloads/acl_engine_project_backup_before_<milestone>`.

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| Blank 3D view, "Loading the 3D library" stays | You're offline, or a network filter blocks cdn.jsdelivr.net. Connect and reload. |
| "No module named pytest / numpy" | You're using the system Python. Use `.venv/bin/python`, or select the `.venv` interpreter in VS Code (Cmd+Shift+P → Python: Select Interpreter). |
| Red error under the view after moving a slider | The design is out of range (the message says which field). Move the slider back. |
| Changes to the code don't show in the designer | Stop the designer (Ctrl+C) and start it again. |
| Tests not showing in VS Code's Testing panel | Select the `.venv` interpreter, then click Refresh in the Testing panel. |
