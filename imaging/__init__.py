"""Volumetric imaging for the ACL workspace.

volumes   load and validate DICOM series, NIfTI volumes, ordered 2D stacks and masks; every calibrated
          volume is reoriented to the nearest canonical RAS+ grid with nibabel (orientation logic is
          SimpleITK's for DICOM and nibabel's for NIfTI, not re-implemented here)
store     per-case store: volumes, masks (versioned), annotations, report findings, jobs
qc        review flags for image/mask pairs (not proof of correctness)
model_map maps KneeScan model outputs (2D outlines in KneeScan's own volume index space) into the volume
evidence  report extraction and the explicit evidence -> scaffold-parameter mapping layer
synthetic synthetic phantoms used for tests and the demo case (never patient data)
"""
