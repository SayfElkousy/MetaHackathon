# Status report (2026-10-24)

## What works now

- **End-to-end vertical slice on SYNTHETIC bilateral phantoms**: import → QC → mirror + per-bone registration → two alternative anatomy estimates → per-source uncertainty → abstention → scaffold candidate → mesh validation → export gate → STL + metadata → 1D mechanics → research report. About 15–20 s per case on an Apple M3 CPU.
- Demonstration cases in `workspace/runs/`:
  - midsubstance (right knee): exported.
  - proximal (left knee): exported. Femoral-end conflict of 12.7 mm flagged and not corrected.
  - partial: bridging refused (fibres continuous).
  - chronic/absent: bridging refused (no remnants).
- DICOM/NIfTI/NRRD import with geometry validation; LPS↔RAS handling tested against nibabel and SimpleITK; DICOM round-trip with shuffled filenames.
- Annotation schema with UNKNOWN vs background semantics, 3D Slicer `.seg.nrrd` / markups import and export, versioning, review status, reader agreement (pairwise Dice + STAPLE).
- Dataset adapters (bilateral_case, local DICOM/NIfTI, OAI, OAI-ZIB, SKM-TEA, MRNet), manifest with pseudonymised subject keys, subject-level splitting, cross-dataset leakage audit, task-eligibility and blocked-task reports.
- 3D U-Net training (MONAI) with ignore-aware loss, foreground patch sampling, seeds, checkpoint/resume, validation-based selection, provenance record; inference writes **unreviewed** predictions; exam-level classifier code for task D.
- Local web interface with inventory, linked multiplanar views (injured ↔ mirrored-registered contralateral), overlays, 3D view, QC/uncertainty, provenance and review, correction import, export, reports, and experiments.
- 48 passing tests covering left/right transforms, physical-coordinate and unit preservation, missing metadata, split leakage, unknown vs background, mesh validity, and blocked exports.

## Data actually acquired and used

| dataset | acquired? | used for |
|---|---|---|
| OAI-ZIB archive (CC BY 4.0, 90 MB) | **Yes**, downloaded and audited | Audit only: 507 masks, label values 0–4 (meanings unconfirmed), 116-mask second subset overlapping 108 of the same scans. **Not used for training**: MR images require OAI/NDA access |
| OAI | No (NDA approval required; NDA pages unreadable to the audit tool) | — |
| SKM-TEA | No (AIMI account required; ~900 GB compressed exceeds 270 GB free) | — |
| MRNet | No (AIMI/Redivis account required) | — |
| Expert bilateral ACL cohort | Does not exist | — |
| Synthetic phantoms | Generated | All pipeline runs, experiments, and the training smoke test |

**No patient data were used.** No public dataset with paired injured and contralateral MRI plus ACL annotations was found.

## Labels missing

- ACL voxel masks (intact) for any accessible dataset → task B blocked.
- Torn-ACL remnant masks, remnant quality, tear location → task C blocked.
- Bilateral pairs with ACL and footprint annotations → task E on real data blocked.
- OAI-ZIB label meanings are undocumented. `configs/label_maps/oai_zib.yaml` stays `confirmed: false` until someone verifies them visually.
- Bone masks in SKM-TEA (it provides cartilage and menisci only).
- Physical geometry for MRNet (not documented).

## Models actually trained

- **None for anatomical use.** `workspace/train/smoke_bone` is a 2-epoch CPU **smoke test** on synthetic phantoms, labelled `smoke_test` everywhere. Validation Dice of 0.31 on phantoms means only that the code paths run.

## Evaluations that actually ran (all SYNTHETIC)

Healthy bilateral hold-out, 12 synthetic subjects, leave-one-subject-out (`reports/healthy_bilateral_holdout.md`). Mean [subject-level bootstrap 95% CI]:

| method | femoral footprint error (mm) | tibial footprint error (mm) | Dice | ASSD (mm) |
|---|---|---|---|---|
| population reference | 3.48 [2.57, 4.44] | 2.33 [1.72, 3.08] | 0.56 [0.48, 0.63] | 1.89 [1.55, 2.24] |
| contralateral mirror | 3.31 [2.75, 3.95] | 2.36 [2.00, 2.76] | 0.66 [0.61, 0.71] | 0.95 [0.81, 1.10] |
| bone-conditioned SSM | 3.46 [2.52, 4.42] | 2.28 [1.77, 3.00] | 0.56 [0.48, 0.63] | 1.91 [1.54, 2.28] |
| SSM + demographic covariate | 3.39 [2.48, 4.36] | 2.16 [1.57, 2.90] | 0.57 [0.49, 0.64] | 1.88 [1.51, 2.27] |

Synthetic injury experiment, 7 simulated tears (`reports/synthetic_injury.md`): remnant reconciliation raised Dice from 0.73 to 0.81 and lowered ASSD from 0.73 to 0.50 mm compared with mirroring alone.

These numbers describe how the software behaves on phantoms whose asymmetry and shape variation **were chosen by the developer**. They are not evidence that bilateral personalisation works on real knees. On these phantoms, contralateral mirroring improved overlap and surface distance but not footprint-centre error over the population baseline, and the demographic covariate did not help.

## Hardware

- Used: Apple M3, 8 cores, 16 GB RAM, MPS available, no CUDA, about 270 GB free.
- Synthetic pipeline: CPU, 11–20 s per case, ~0.8 GB peak memory (measured).
- Real task-A/C training (96³ patches, 3D U-Net): realistically one CUDA GPU with ≥16 GB (24 GB preferred), on the order of a day per model. MPS/CPU work but are slow. **Not started; requires your resource authorisation.**
- Storage: OAI imaging is large (exact size unverified). SKM-TEA needs about 0.9–1.6 TB, more than this machine has free.

## Remaining clinical and experimental dependencies

1. Ethics approval and collection of the bilateral cohort (docs/COHORT_SPEC.md).
2. NDA access to OAI images, plus confirmation of the OAI-ZIB label map.
3. Expert readers for annotation, review, and inter-reader variability.
4. Validation of all abstention thresholds on real data.
5. Fabrication process parameters and measured material properties (replacing placeholders).
6. Bench, fatigue, degradation, cadaveric, animal and clinical studies (docs/ROADMAP.md).

## Next experiment

The healthy bilateral hold-out on **real** MRI (docs/ROADMAP.md). Hide one knee's expert ACL annotation and estimate it from the contralateral knee, from bone shape alone, and from a population reference. Compare paired footprint-centre errors against inter-reader variability. `bilacl experiment healthy --cases <real case dirs>` runs this as soon as such cases exist in the `case.json` format.

## Known limitations of the implementation

- Phantoms are geometric primitives. Registration and estimation will behave differently on real anatomy, partial bone coverage, and segmentation noise.
- The statistical baseline uses closest-point correspondence, not a full SSM.
- The asymmetry prior (1.5 mm/axis) is an engineering choice of the order reported for 10 cadaver pairs (Jang 2017), not a population estimate.
- Scaffold cross-sections are measured perpendicular to the straight footprint axis, which is approximate for strongly curved ligaments. Channel cross-sections are in-plane circles.
- The project directory is not a git repository, so provenance records `no-vcs`. Run `git init` to get commit hashes in run records.
