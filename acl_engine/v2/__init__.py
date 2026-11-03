"""ACL repair-scaffold geometry engine, version 2.

Builds an anatomical scene (distal femur, proximal tibia, cartilage, footprints,
retained ligament tissue, injury region, intact reference, scaffold) from a
design state. The design state can come from a preset or from patient-specific
MRI-analysis output (see acl_engine/v2/patient.py and docs/PATIENT_INPUT.md).

Units: mm and degrees. Frame: +X lateral, +Y anterior, +Z proximal, origin at
the tibial plateau centre, right-knee canonical (left knees are mirrored in X).
"""
SCHEMA_STATE = "acl-design-state/2.0"
