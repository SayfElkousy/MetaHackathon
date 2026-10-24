# Expert-annotated bilateral ACL cohort: collection requirement

**Status: does not exist.** The 2026-10-24 audit found no public dataset with paired injured and contralateral knee MRI plus expert ACL/remnant annotations. OAI, OAI-ZIB, SKM-TEA and MRNet do not fill this gap (see DATASETS.md). The core tasks (C: remnant segmentation; E: bilateral native-anatomy estimation evaluated on real ruptures) are blocked until this cohort is collected under ethics approval.

## Inclusion

- Unilateral ACL injury (all tear locations, including partial and chronic tears); patients scanned bilaterally.
- Healthy bilateral volunteers (both knees), for the hold-out anatomy-estimation experiment.
- Exclusions recorded, not silently dropped: prior surgery on either knee, contralateral ligament injury, contralateral pathology. A contralateral knee requires **expert confirmation** before it can serve as a reference (`contralateral_confirmed_normal`).

## Imaging (per knee, same session where possible)

- A 3D isotropic or near-isotropic sequence suitable for millimetre geometry (largest voxel ≤ 1.0 mm is the current engineering export threshold), e.g. 3D PD/DESS-type. Exact protocol to be set by the imaging team.
- Routine 2D clinical series (sagittal PD/T2 FS, etc.) kept as **separate** series.
- Full DICOM with geometry tags (ImagePositionPatient, ImageOrientationPatient, PixelSpacing), DICOM Laterality, scanner make, model and field strength.
- Recorded knee position (flexion angle, coil). Positions differ between knees, and the registration handles this per bone.
- Optional research sequences for remnant tissue characterisation (e.g. T2*/UTE): analysed separately and never conflated with routine MRI.

## Annotations (ANNOTATION_SPEC.md)

- Femur and tibia (≥1 reader); ACL intact / remnant labels; UNKNOWN where unassessable.
- Footprint centres where identifiable; remnant tips; independent QC landmarks.
- At least 20% of cases annotated by two readers for inter-reader variability.
- Tear location, chronicity, and remnant quality per side.

## Clinical metadata (pseudonymised)

Injury date, imaging date, surgery date (repair timing), arthroscopic findings (tear location, remnant quality, viability judgement), associated injuries (meniscus, cartilage, bone bruise, collaterals), demographics (optional weak priors), activity level. Follow-up: imaging time points, reinjury and reoperation events, and patient-reported outcomes. These are stored for future research only. No individual predictions are produced without outcome data and validation.

## Sample size

Not set yet. Plan it from the healthy-hold-out experiment (ROADMAP.md, next experiment). The quantity of interest is the paired difference in footprint error between contralateral and bone-only estimates, and the minimum important difference should be anchored to inter-reader footprint variability.

## Governance

Ethics approval and consent covering bilateral research imaging and data sharing. Identifiable imaging stays on local or approved infrastructure; this software sends no imaging to external services. Pseudonymisation uses `manifest.pseudonym` with a locally held salt.
