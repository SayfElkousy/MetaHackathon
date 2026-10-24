# bilacl: bilateral knee MRI ACL anatomy estimation and scaffold hypothesis research platform

Research software that combines MRI of an injured knee with MRI of the contralateral knee to **estimate** native ACL geometry. It then generates **investigational** scaffold geometry under explicit assumptions.

> **Not a medical device. Not for clinical decisions.** Restoring external shape is not restoring biological function. Nothing here shows that any scaffold heals an ACL or reduces reinjury. Routine MRI does not resolve individual collagen fibres, and the internal pathways produced here are **hypothesized guidance architecture**, not recovered fibres. Current status is in [docs/STATUS.md](docs/STATUS.md).

## Evidence levels (carried in data, files, CLI output and UI badges)

| level | meaning |
|---|---|
| `observed` | Directly annotated in the input imaging |
| `estimated` | Model or registration-derived anatomy |
| `hypothesized` | Generated internal architecture or correspondences from priors |
| `validated` | Supported by actual evaluation data (none yet) |
| `synthetic` | Produced from software phantoms; no clinical meaning |

## Install

Python 3.12 (PyTorch has no wheels for the system Python 3.15). Using [uv](https://github.com/astral-sh/uv):

```bash
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[test]'
.venv/bin/python -m pytest -q          # 48 tests
```

## Commands (all implemented and exercised)

```bash
B=.venv/bin/bilacl

# audit & data
$B hardware
$B registry --markdown                                   # dataset registry with verification state
$B acquire report --out reports/acquisition_report.json  # dry run: local status, sizes, disk, requirements
$B acquire download oai_zib [--confirm]                  # only public, permitted resources; resumable + sha256
$B inspect oai_zib data/oai_zib                          # adapters: bilateral_case, local, oai, oai_zib, skm_tea, mrnet
$B manifest build --sources bilateral_case=workspace/cases --out m.csv
$B manifest split --manifest m.csv                       # subject-level; preserves official splits; leakage audit
$B manifest audit --manifest m.csv
$B manifest subset --manifest m.csv --n-subjects 5 --out dev.csv
$B tasks --manifest m.csv                                # which tasks are runnable / blocked and why

# imaging
$B dicom <dicom_dir> --out out/ --convert                # series discovery, ordering by IPP, localizer/gap/duplicate checks
$B preprocess image.nii.gz --labels labels.nii.gz --spacing 0.5 0.5 0.5 --n4 --out pre/   # + QC before/after

# annotation (3D Slicer)
$B annotate color-table --out bilacl_labels.ctbl
$B annotate export-slicer --labels labels.nii.gz --out labels.seg.nrrd
$B annotate import-slicer --seg edited.seg.nrrd --reference image.nii.gz --out v2.nii.gz --annotator R07 --version 2
$B annotate agreement --masks readerA.nii.gz readerB.nii.gz
$B annotate review --record v2.annotation.json --status approved --reviewer R02

# models
$B train --config configs/train_bone_smoke.yaml --manifest m.csv      # CPU smoke test (synthetic, labelled smoke_test)
$B train --config configs/train_bone.yaml --manifest real.csv         # real task-A baseline (needs eligible data)
$B infer --checkpoint run/best.pt --image image.nii.gz --out pred.nii.gz   # saved as UNREVIEWED model prediction
$B evaluate --pred pred.nii.gz --gt labels.nii.gz --labels 1 2        # Dice/ASSD/HD95/volume, UNKNOWN ignored

# bilateral pipeline
$B synth --out workspace/cases --seed 0 --injury midsubstance          # SYNTHETIC phantoms (intact|proximal|midsubstance|distal|partial|chronic_absent)
$B register workspace/cases/SYN0000_midsubstance --out out/
$B run workspace/cases/SYN0000_midsubstance --out workspace/runs/SYN0000_midsubstance \
      --scaffold-config configs/scaffold_demo.yaml --mechanics-config configs/mechanics_demo.yaml
$B review workspace/runs/<case> --status approved --reviewer R02 --select remnant_reconciled
$B experiment healthy --out workspace/experiments/healthy --n 12       # hold-out anatomy estimation (synthetic unless --cases)
$B experiment injury  --out workspace/experiments/injury               # SYNTHETIC injury experiment

# interface (binds to 127.0.0.1; imaging stays local)
$B serve --workspace workspace --port 8765    # http://127.0.0.1:8765
```

## Pipeline

1. **Import and validate.** DICOM slices are ordered by ImagePositionPatient projected on the slice normal. The importer detects localizers, gaps, duplicates and inconsistent geometry. NIfTI without qform/sform, and bare arrays, are flagged `physical_verified=false`. Orthogonal 2D series are never stacked (`assert_not_stacking`).
2. **Coordinates.** Everything internal is RAS+ mm with a voxel-to-world affine (`imaging/coords.py`). DICOM LPS is converted with diag(−1, −1, 1). Image interpolation (linear) is separate from label interpolation (nearest, so UNKNOWN=255 survives). Source spacing is always recorded.
3. **Bilateral registration** (`registration/bilateral.py`). Explicit mirror x → −x, then trimmed rigid ICP **per bone** (femur, then tibia) on joint-cropped surfaces. The contralateral ACL is transferred by a smoothstep blend of the two rigid transforms along the ACL axis. No free-form deformation is used, so registration cannot invent ACL shape. QC uses surface distances plus **independent** landmarks that never drive the registration.
4. **Anatomy estimates** (`estimation/anatomy.py`). `contralateral_mirror` and `remnant_reconciled`. The latter moves the ends toward *observed* injured-side attachments, capped at 4 mm; beyond the cap it flags a conflict instead of correcting. Displaced remnants are reported and not absorbed. Uncertainty is computed **separately** for registration, bilateral asymmetry, and contralateral segmentation.
5. **Baselines** (`estimation/population.py`). Population reference and bone-conditioned statistical model (PCA + ridge, optional demographic covariates), refit leave-one-subject-out.
6. **Abstention** (`abstention.py`). Laterality, physical metadata, resolution, contralateral expert confirmation, ICP residuals, independent landmarks, remnant conflicts, uncertainty. **All thresholds are unvalidated engineering settings.**
7. **Scaffold candidate** (`scaffold/`). Per-station ellipse envelope from the selected estimate. Bridging mode spans the remnant gap plus a user-set overlap; attachments are preserved by default; remnants plus clearance are excluded; channels follow a user-chosen alignment hypothesis. Bridging is refused for partial tears and for knees without remnants. Channel size, wall thickness and material values are required inputs with cited sources; there are no "optimal" defaults.
8. **Validation and export.** Watertightness, winding, exact self-intersection test, components, minimum feature, channel walls, porosity, bone clearance, remnant overlap. The export gate hard-blocks unverified physical metadata and invalid meshes. Real data also need expert-approved anatomy and no abstention. Output is STL plus metadata JSON (units, frame, affine, source spacing, transforms, assumptions, uncertainty, review status).
9. **Simplified mechanics** (`mechanics/axial.py`). 1D linear-elastic bar in series, with sensitivity sweeps. Not finite-element; no stress fields.

## Layout

```
src/bilacl/        package (imaging, annotation, adapters, training, registration, estimation, scaffold, mechanics, evaluation, ui)
registry/          datasets.yaml (verified registry)
configs/           training, scaffold, mechanics, label-map configs
docs/              STATUS, DATASETS, EVIDENCE, ANNOTATION_SPEC, COHORT_SPEC, ROADMAP
reports/           acquisition report, OAI-ZIB inspection, experiment results, hardware
tests/             48 tests
data/, workspace/, runs/   local only (gitignored): imaging, salts, runs
```
