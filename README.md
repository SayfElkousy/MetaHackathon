# ACL Scaffold Geometry Engine

Builds a three-strand braided ACL scaffold, places it inside a simple 3D knee,
checks it, and saves files you can look at. This is Muse's code (milestones M1,
N1, N2), assembled into one project and fixed so it runs. See `REPORT.md` for
what was changed.

**New here?** Read [USER_GUIDE.md](USER_GUIDE.md) for how to use the designer. The next prompt for Muse
is in [MUSE_NEXT_PROMPT.md](MUSE_NEXT_PROMPT.md).

## One-time setup (about 10 minutes)

1. **Install Python** 3.11 or newer from <https://www.python.org/downloads/>.
   On Windows, tick **"Add python.exe to PATH"** in the installer.
2. **Install VS Code** from <https://code.visualstudio.com/>, then install the
   **Python** extension (from Microsoft) inside VS Code: the square Extensions
   icon on the left, search "Python", click Install.
3. **Open this folder** in VS Code: File → Open Folder… → pick `acl_engine_project`.
4. **Make a virtual environment** (a private copy of Python for this project):
   press `Ctrl+Shift+P` (Mac: `Cmd+Shift+P`), type **"Python: Create Environment"**,
   choose **Venv**, pick your Python, and tick `requirements.txt` when asked.
   VS Code installs everything for you. If it doesn't ask about requirements,
   open a terminal (Terminal → New Terminal) and run:
   ```
   pip install -r requirements.txt
   ```

## Running it

The easiest way: click the **Run and Debug** icon on the left (triangle with a
bug), pick **"1. Check everything"** from the dropdown at the top, and press the
green ▶ button. Or, in the terminal:

```
python check.py
```

It runs all the tests, builds both example designs, draws a picture of each,
and prints a summary block between `=====` lines. **Paste that block back to
Muse** after each milestone.

## Interactive designer (sliders)

Run and Debug → **"4. Open interactive designer (N3)"** → ▶, or in the terminal:

```
python tools/server.py
```

Your browser opens the designer. Move a slider and the 3D model rebuilds in about a second,
with pass/fail checks and hints on the right. **Save design** writes a file into `fixtures/`;
**Export STL** downloads the scaffold for 3D printing. Press Ctrl+C in the terminal to stop.
It runs only on your computer, and needs internet the first time to load the 3D library.

## Seeing the results

After a run, look in the `outputs/` folder:

| File | What it is | How to open it |
|---|---|---|
| `knee.png` | Picture of the scaffold in the knee, from 3 angles | Click it in VS Code |
| `knee.glb` | 3D scene you can rotate (scaffold + ACL ghost + bones + tunnels) | Drag into <https://gltf-viewer.donmccurdy.com/>, or install the VS Code extension "glTF Tools" and right-click → glTF: Preview |
| `scaffold.stl` | Just the scaffold, ready for 3D-printing software | Drag into <https://www.viewstl.com/> |
| `validation.json` | Pass/fail checks | Click it in VS Code |
| `context.json` | Knee placement numbers (angles, fit, coverage) | Click it in VS Code |
| `parameters.json` | Every input and computed value | Click it in VS Code |
| `migration_report.json` | How an older design file was upgraded, and which defaults were assumed | Click it in VS Code |

## Changing the design

Open `fixtures/acl_context_request.json` in VS Code and edit the numbers, then
run again. Useful ones:

| Setting | Where | What it changes |
|---|---|---|
| `acl_length_mm`, `acl_width_mm`, `acl_thickness_mm` | `anatomy` | Size of the native ACL (the red ghost) |
| `sagittal_inclination_deg`, `coronal_inclination_deg` | `anatomy` | How steeply the ACL leans |
| `swing_mm` | `architecture` | How far strands move side to side |
| `fold_depth_mm` | `architecture.fold` | How far a strand dips to pass under another |
| `fold_sharpness` | `architecture.fold` | 0 = smooth waves, 1 = sharper over/under steps |
| `diameter_mm` | `strand` | Thickness of each strand |
| `braid_angle_deg` | each zone | Angle of the strands at each crossing |
| `path_length_ratio` | each zone (with `"control": "fiber_length"`) | Fiber length: 1.06 = each strand is 6% longer than the zone |


## Project layout

```
cli.py                 builds one design: python cli.py --request <file> --out <folder>
check.py               runs everything and prints the summary for Muse
fixtures/              example designs (JSON)
acl_engine/            the engine
  schema.py            checks the design file is valid (schema 4.0.0, migrates 2.0/3.0)
  records.py           parameter records (value + origin + source + confidence)
  controls.py          fiber-length closed loop (hits the target to 0.002)
  export.py            the one shared STL / GLB / JSON writer
  paths.py             strand paths, crossings, angle and fiber-length control
  fold.py              over/under shape (fold depth and sharpness)
  axis.py              straight or curved scaffold axis
  frames.py            keeps each strand's cross-section from twisting
  sweep.py             turns strand paths into 3D tubes
  measure.py           curvature and strand-to-strand gap
  context.py           knee, native ACL ghost, placement, fit checks
  glb.py               writes the 3D scene file
tests/                 automatic tests (22)
tools/render.py        draws knee.png from knee.glb
tools/server.py        the interactive designer (N3); page files in tools/web/
sources.json           the publications each literature value comes from
```
