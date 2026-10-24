# Roadmap and validation requirements

The four questions are kept separate. Progress on one is not evidence for another.

| question | what would answer it | status |
|---|---|---|
| 1. Geometric fit: does the candidate fit the estimated anatomy? | Comparison against expert anatomy, perturbation sensitivity | Implemented on synthetic data; real data needed |
| 2. Mechanical behaviour under a stated model | Measured material properties, then validated FE, then bench tests | Only a 1D axial model with placeholder inputs |
| 3. Biological healing | Cell culture, then animal models, then histology and imaging biomarkers | Not addressed |
| 4. Clinical reinjury | Prospective trials with adequate power | Not addressed |

## Next experiment (most direct test of bilateral personalisation)

**Healthy bilateral hold-out on real MRI.** About 30–40 healthy volunteers (both knees, 3D sequence, expert ACL, footprint, and QC landmarks; 20% double-read):

1. Hide the target knee's ACL annotation. Estimate it with (a) a population reference, (b) a bone-conditioned statistical model, (c) contralateral mirroring with per-bone registration, and (d) (c) plus demographics.
2. Primary endpoint: femoral and tibial footprint-centre error (mm), compared **paired within subject** between (c) and (b). Secondary: length, cross-sectional profile, ASSD/HD95, Dice.
3. Judge errors against inter-reader footprint variability.
4. Leave-one-subject-out for the statistical baselines; subject-level bootstrap CIs. Use `bilacl experiment healthy --cases ...` (implemented).

The synthetic version of this experiment already runs. Its numbers are not evidence about real anatomy.

## Engineering milestones

1. Confirm the OAI-ZIB label map; obtain OAI images (NDA); train task A; evaluate externally on a non-OAI set.
2. Collect the bilateral cohort (COHORT_SPEC.md); train and evaluate task C on real remnants (task-B models must be tested on C before use).
3. Replace closest-point correspondence with a proper SSM (e.g. GPA + dense correspondence).
4. Recalibrate the abstention thresholds against inter-reader variability and real registration errors.
5. Longitudinal module: register follow-up imaging to baseline; track healing-tissue volume, CSA, and signal (cf. Kiapour 2019; Pruneski 2026), as research measures only.

## Pre-clinical validation requirements (none started)

- **Material characterisation**: tensile and stiffness tests of the actual fabricated construct (and after sterilisation), replacing `E_solid_MPa` placeholders.
- **FE analysis** with a validated constitutive law, mesh-convergence study, documented boundary conditions and fixation, and sensitivity analysis.
- **Bench tests**: ultimate load, stiffness, creep, cyclic **fatigue**, and suture/fixation pull-out.
- **Degradation**: mass loss and mechanical retention over time in physiologically relevant media.
- **Cadaveric**: fit, insertion feasibility, and knee kinematics (AP laxity, rotation) with scaffold vs intact vs ACL-deficient.
- **Biological**: cytocompatibility, cell infiltration along channels, in-vivo large-animal healing with histology (e.g. ligament maturity index) and MRI biomarkers. Animal-to-human differences must be documented.
- **Clinical**: regulatory pathway, then feasibility and randomised studies. BEAR II is the comparator context. Its reinjury difference (14% vs 6%) was not significant (P=.32), and this project makes no claim of superiority.
