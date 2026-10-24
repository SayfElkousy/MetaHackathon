# Annotation specification

Applies to every label volume used for training or evaluation. Implemented in `src/bilacl/annotation/schema.py`.

## Label codes (one volume per exam, uint8)

| code | name | meaning |
|---|---|---|
| 0 | BACKGROUND | Annotator asserts none of the annotated structures is present |
| 1 | FEMUR | |
| 2 | TIBIA | |
| 3 | PATELLA | optional |
| 4 | FIBULA | optional |
| 5–9 | cartilage / menisci | optional, when available |
| 10 | PCL | optional (useful registration context) |
| 20 | ACL_INTACT | Continuous ACL tissue (intact knee, or the continuous part of a partial tear) |
| 21 | ACL_REMNANT_FEMORAL | Visible remnant attached on the femoral side |
| 22 | ACL_REMNANT_TIBIAL | Visible remnant attached on the tibial side |
| 23 | ACL_FRAGMENT_OTHER | Visible ACL tissue that cannot be attributed to either side |
| 24 | ACL_UNCERTAIN | ACL-region tissue whose identity or continuity is uncertain |
| 255 | UNKNOWN | Not assessed, unassessable, or outside the annotated region (ignored in training and metrics) |

Rules:

1. **Do not label a tear gap as tissue.** Mark the gap BACKGROUND only if it can be assessed as empty (fluid, haemorrhage); otherwise mark it UNKNOWN.
2. **Absence of annotation is not background.** Each annotation file lists `annotated_labels`. During training, a 0 voxel counts as background for a task only if every structure in that task was annotated (`effective_target`). Otherwise it is ignored.
3. Never stack differently oriented series into one label volume. Annotate on one acquired series and record which one (`image_ref`).
4. Store landmarks in Slicer markups files (`*.mrk.json`), never burned into masks.

## Landmarks (`LANDMARKS` in schema.py)

ACL: femoral and tibial footprint centres (optional outlines), femoral remnant tip, tibial remnant tip, and tear-region centre.

Independent registration-QC landmarks: lateral and medial femoral epicondyles, notch roof apex, medial and lateral tibial eminence tips, tibial tuberosity, PCL tibial attachment. These are **never used to drive registration**. They exist only to check it.

## Per-exam metadata (`<labels>.annotation.json`)

`annotation_id`, `exam_id`, `image_ref`, `annotator` (pseudonymous), `annotator_role`, `version`, `parent_annotation_id` (for corrections), `annotated_labels`, `source` (manual | model_prediction | model_corrected | synthetic), `model_id`, `review_status` (unreviewed | in_review | corrected | approved | rejected), `reviewer`, `tear_location` (intact | proximal_avulsion | proximal | midsubstance | distal | distal_avulsion | partial | unknown, following van der List et al. 2017 plus partial/unknown), `chronicity`, `remnant_quality` per side, `contralateral_pathology`, `contralateral_confirmed_normal`, `notes`.

Ground-truth rule (`usable_as_ground_truth`):
- Unreviewed model predictions are **never** ground truth.
- Model-corrected masks become ground truth only after expert approval.
- Evaluation sets should contain only `approved` manual annotations.

## 3D Slicer workflow

```bash
bilacl annotate color-table --out bilacl_labels.ctbl                 # load in Slicer's Colors module
bilacl annotate export-slicer --labels labels.nii.gz --out labels.seg.nrrd   # open as Segmentation
# ... edit in Slicer; segment names must match label names (e.g. ACL_REMNANT_FEMORAL) ...
bilacl annotate import-slicer --seg edited.seg.nrrd --reference image.nii.gz --out labels_v2.nii.gz \
    --exam-id E1 --annotator R07 --role "MSK radiologist" --version 2 --parent E1_v1 --tear-location midsubstance
bilacl annotate review --record labels_v2.annotation.json --status approved --reviewer R02
bilacl annotate agreement --masks readerA.nii.gz readerB.nii.gz      # pairwise Dice per label + STAPLE
```

Segment names that do not map to a project label cause the import to fail, so nothing is silently dropped. Markups are read and written in LPS (Slicer's convention) and converted to RAS internally.

## Reader disagreement

Pairwise Dice ignores voxels that any reader marked UNKNOWN. STAPLE gives a consensus estimate plus per-reader sensitivity and specificity. Inter-reader variability is the reference against which anatomy-estimation errors must be judged.
