# ACL repair designer (v2)

**3A (scaffold fit):** the scaffold is generated inside a millimetre repair space (target envelope ∩ affected AM/PL region ∩ damaged interval + permitted overlap, with strand radius and clearances), verified in 3D, highlighted where it fails, and sectioned from the same geometry along the ACL, either bundle, or axial/sagittal/coronal planes. See `docs/V2_REPORT.md` §7.

`python tools/server.py` then open http://127.0.0.1:8000/. That is the one page: it asks for an MRI, holds the scaffold builder and is the imaging workspace. `/view?state=<id>` is its pop-out scaffold view; the old addresses `/workspace`, `/upload` and `/classic` lead to `/`.

- Anatomical scene: procedural distal femur / proximal tibia with cartilage (or imported patient meshes), footprints on bone, retained ligament tissue, injury region, intact reference, and a scaffold placed by the injury.
- Adjustable hierarchy (bundles → fascicle groups → representative fibers), separate twist channels, multiphasic regions.
- **Scaffold Levels** (default construct): six editable levels — 1 surface coating/nano-texture, 2 grooves, 3 hollow channels, 4 braided AM-like/PL-like mantles, 5 ribbon sheath, 6 flared graded ends — with highlight/isolate/neighbour views, a level cross-section, true-scale magnified details, and conflicts with tested fixes. Knee pose, tear and reference anatomy are in the separate **Anatomy and Pose** panel. See `docs/V2_REPORT.md` §9; engine `acl_engine/v2/layered.py`. The earlier gap-filling braided/aligned scaffold remains available as a construct type.
- **Imaging workspace** (the page itself): gray/white V7-inspired tabbed workspace with MPR (axial/sagittal/coronal, linked crosshairs, windowing, colour maps), segmentation overlays and editing, quality control, landmark/distance measurements, traceable evidence → scaffold mappings, and the Muse controller (every control registered, one undoable transaction per request). What works, what needs model access, what is not implemented: `docs/IMAGING_WORKSPACE.md`.
- **Muse Spark setup:** copy `.env.muse.example` to `.env.muse`, add a Meta Model API `MODEL_API_KEY`, and restart the server. The app defaults to `muse-spark-1.3` at `https://api.meta.ai/v1`; `.env.muse` is gitignored.
- Patient-specific input from an MRI-analysis model: `docs/PATIENT_INPUT.md`; synthetic fixtures in `fixtures/patients/`.
- **MRI upload** (opens on its own when no design is loaded, or **Upload MRI** in the header): send a scan to the KneeScan MRI reader (`mri_analysis/`) and build the scaffold model from its readings. See "MRI upload" below.

## MRI upload: scan → KneeScan → scaffold builder

1. Once per computer: `python tools/setup_kneescan.py`. It creates KneeScan's own Python environment, installs `mri_analysis/requirements.txt` (PyTorch CPU, MONAI, transformers, Gradio; about 1 GB) and downloads the model weights (about 470 MB) from the private Hugging Face repo `tariksyed/kneescan`, asking for a read token. Everything goes to `%LOCALAPPDATA%\KneeScan` (`~/.kneescan` on macOS/Linux; set `KNEESCAN_HOME` to change it), outside the project folder, which may be synced. Re-running skips finished steps.
2. `python tools/server.py`, open http://127.0.0.1:8000/. The server starts KneeScan in the background (`tools/reader_launcher.py`, the way the Colab notebook runs it) and stops it on exit; the upload window shows **starting…** while the models load (about a minute on a CPU), then **connected**. Nothing to paste. To use another reader instead (e.g. the Colab notebook's `gradio.live` link), paste its address in the upload window or set `KNEESCAN_URL`; `KNEESCAN_AUTOSTART=0` turns the automatic start off. KneeScan's output goes to `kneescan.log` in the KneeScan folder.
3. With no design loaded the page asks for an MRI: choose the scan and the knee side, press **Analyze and send to the scaffold builder**.
4. Review the readings and what was sent, then **Build the scaffold model**: the design loads in the same page.

There is no preloaded sample model: the scaffold view shows **Waiting for an MRI** until a design made from KneeScan readings exists. It comes from the upload dialog, or from **AI segmentation → Run on this series** (or importing a KneeScan results JSON) when no design is loaded yet; the panel then asks for the knee side (preset from the DICOM Laterality tag when present). A design opened by link (`?state=<id>`) loads as before, and the synthetic demo cases stay available under **Open / import**.

`tools/mri_bridge.py` does the transfer only; neither the reader nor the modeler changed. It calls the reader's own Gradio API (`/analyze`, the same call its web page makes), converts the results to `acl-mri-input` and runs the modeler's importer (`acl_engine/v2/patient.py`):

| KneeScan reading | Modeler parameter |
|---|---|
| tear status + level | `injury.type` (intact / partial / complete). Partial tears also get `injury.involvement = AM`, marked *awaiting review*, because the modeler needs a torn region and KneeScan does not give one |
| ACL width (mid-substance) | `lig.width_profile`: the template profile scaled so its mid-substance median equals the reading |
| ACL length | `lig.length_mode = target`, `lig.length_target_mm`. With the template attachments the modeler reaches about 32.5–36.8 mm and flags anything outside |
| ACL location (slice, box) | recorded with the design (`assets.mri_readings`); the modeler has no image-location input |
| femur → fibular head | recorded with the design; the modeler has no fibula |

Sizes from a torn ACL or a failed outline check are marked *awaiting review*, following KneeScan's own warnings. Tests: `tests/test_mri_bridge.py`.
- What changed, how it was verified, limitations: `docs/V2_REPORT.md`; screenshots in `docs/screenshots/`.
- Engine: `acl_engine/v2/`; tests: `pytest` (`tests/test_v2.py`).

---

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
