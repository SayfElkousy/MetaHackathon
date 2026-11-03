# Anatomical ACL loft — review notes

Illustrative anatomical model (literature template). Not patient-specific,
not mechanically validated, no healing claims. Full parameter record per
build: `anatomy.json`. Measured/failed/unperformed checks per build:
`validation_report.json`.

## Coordinates, units, laterality, pose

- mm, degrees. +X lateral, +Y anterior, +Z proximal, origin at tibial
  plateau centre; left knees mirrored. Frame recorded in every
  `anatomy.json`.
- `knee_scene.flexion_deg` (0–90, default 0) now drives geometry: the
  femoral side (condyles, femoral tunnel, footprint, centerline end,
  strands) poses rigidly about the mediolateral axis through the condyle
  centres. Tibia-fixed, **illustrative** — real knees roll/slide about a
  moving centre, so length change with flexion here is not physiological
  and no tension is computed.

## What improves geometric fidelity

- Curved centerline (Catmull-Rom through attachment endpoints + editable
  controls, arc-length resampled) instead of a straight axis.
- Varying superellipse cross-sections (width/thickness/shape/orientation
  profiles) instead of constant ellipses; body follows the unflared
  midsubstance profile, flare lives in the insertion transitions.
- Parallel-transport (RMF) frames; section orientation, bundle-boundary
  angle, and strand swirl are independent functions of normalized arc
  length in documented units.
- AM/PL as a partition-of-unity angular mask (configurable sharpness, no
  mandatory gap); internal strands clamped inside the envelope, ending
  distributed ON footprint domains projected onto the bone surfaces.
- Femoral fan rotation auto-sign so the fan lies along the notch wall.
- Loft-based residual stumps (watertight) cut from a copy; the intact
  reference surface is never modified by tear or scaffold parameters.

## What improves presentation only

- Offline 6-view renders with triad/scale/labels, section-profile figure,
  before/after comparison, web-viewer layer toggles and anatomical cameras.
  Resemblance to reference figures is qualitative evidence only.

## Known limitations

- Schematic bones (box + ellipsoids); footprints sit on reference surfaces.
- Tear/injury defaults are synthetic examples unless `tear.observed` is set
  with user-supplied residual data; partial tears cut the full surface with
  reduced gap while intact-bundle strands span (documented simplification).
- No MRI fitting inputs exist in this project; Source C (arXiv:2608.18012,
  abstract-verified) needs MR volumes + trained weights and was not applied.
- Source B is an educational rendering (view-only, not redistributed).
- Parrilli et al. (L19): two specimens, one donor, preparation deformation —
  morphology template, not universal dimensions.
