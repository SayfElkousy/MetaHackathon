# Dataset audit (2026-10-24)

The machine-readable source of truth is `registry/datasets.yaml`. Every field there carries a `state` (verified / partially_verified / unverified). Run `bilacl registry --markdown` for a field-by-field view, or `bilacl acquire report` for current local status. This page summarises what was actually verified, how, and what it means for the project.

## How it was verified

- Official pages were fetched automatically. **Readable**: SKM-TEA GitHub (README + DATASET.md), AIMI SKM-TEA and MRNet pages, MRNet competition page, OAI-ZIB landing page (ZIB OPUS4). **Not readable**: nda.nih.gov/oai (JavaScript shell), so OAI facts remain *unverified*. The KneeMRI (Rijeka) site was blocked by a firewall.
- **OAI-ZIB archive downloaded and audited** (public, CC BY 4.0, 89,972,103 bytes, sha256 `361d0179…35caf7`). This is the only dataset acquired. It was not used for training, because its MR images must come from OAI through NDA.

## Summary

| dataset | access | what is actually there | bilateral? | role here | blocked tasks |
|---|---|---|---|---|---|
| **OAI** (NDA) | NDA account / data-use certification (**unverified**; page unreadable) | Unverified. Expected: DICOM knee MRI, multiple visits; no ACL labels assumed | Must be audited per subject/side/visit, never assumed | Unlabelled pretraining, bone geometry, bilateral registration research, L/R asymmetry | All ACL tasks (no ACL masks) |
| **OAI-ZIB** | Public zip (downloaded) + OAI images via NDA | `segmentation/`: 507 masks (.mhd/.raw uint8, 160×384×384, 0.7×0.365×0.365 mm), one per OAI participant, label values 0–4 (**meanings undocumented**; inferred femur / femoral cartilage / tibia / tibial cartilage, unconfirmed), OAI image path per ID, published 2-fold CV lists. `classification/`: 116 masks (values 0–2); **108 are the same participants and the same scans** as `segmentation/`. MetaImage headers are internally inconsistent about orientation (identity matrix + "RPI") | No: one knee per participant | Task A (femur/tibia) once OAI images are obtained and the label map is confirmed | A until OAI images are available; B, C, E (no ACL labels) |
| **SKM-TEA** | Stanford AIMI account | 155 qDESS scans; HDF5 (echo1, echo2, 6-class one-hot seg) + NIfTI masks + DICOM; tissue masks for patellar, femoral and tibial (med/lat) cartilage and menisci (med/lat); **16 pathology categories as 3D bounding boxes from reports**; official split 86/33/36; ~900 GB compressed / 1.6 TB expanded, **more than the 270 GB free here** | Not documented | Cartilage/meniscus segmentation, representation learning, sequence generalisation | A (no bone masks), B, C (no ACL masks), E |
| **MRNet** | Stanford AIMI / Redivis | 1,370 exams (1,312 patients); patient-level split 1,130/120/120 (test hidden); **exam-level** labels: abnormal, ACL tear, meniscal tear; 5 planar series per exam; GE, 1.5 T / 3 T; **file format and geometry not stated** | Not documented | Task D auxiliary classifier; representation learning | A, B, C, E, physical export |
| **KneeMRI (Rijeka)** | Unverified (site blocked) | Secondary source only (Awan et al. 2022): 917 exams | Unknown | Candidate, verify first | — |
| **Bilateral ACL cohort** | **Does not exist** | — | — | Core tasks C and E | Collection required (COHORT_SPEC.md) |

## Distribution shift (documented, not corrected)

OAI and OAI-ZIB are osteoarthritis cohorts (older adults). The target population (BEAR II median age 17, acute midsubstance tears) differs in age, bone morphology, cartilage, and injury pattern. Models or shape statistics from OAI need external testing on the target population.

## Leakage controls

- OAI-ZIB subjects are pseudonymised in the **OAI** namespace, so the same person has the same `subject_key` in OAI and OAI-ZIB.
- The OAI-ZIB `classification/` masks are on the same scans as `segmentation/` masks. They must stay in the same partition (the adapter reports this; records use `segmentation/` only).
- MRNet exam filenames carry no patient ID. MRNet records are kept only under the official, patient-level split; otherwise they are excluded (`excluded_unknown_subject`).

## Acquisition commands

```bash
bilacl acquire report --out reports/acquisition_report.json      # dry run, local status, disk space
bilacl acquire download oai_zib                                   # dry run: size + disk check
bilacl acquire download oai_zib --confirm                         # resumable; sha256 recorded
bilacl inspect oai_zib data/oai_zib                               # archive audit
bilacl inspect skm_tea data/skm_tea                               # after manual AIMI download (image track)
bilacl inspect mrnet data/mrnet
bilacl inspect oai data/oai                                       # after NDA download
```

Downloads over 20 GB require `--allow-large`. Registration- or DUA-gated datasets are never downloaded automatically. Credentials, if ever needed, come from `BILACL_<NAME>_TOKEN` or `~/.config/bilacl/credentials.json`, both outside the repository.
