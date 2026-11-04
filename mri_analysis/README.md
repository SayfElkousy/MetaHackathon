# KneeScan: MRI analysis

The MRI-analysis side of the project: it reads a knee MRI and measures the ACL and the femur → fibular head gap. It does not yet write the scaffold designer's `acl-mri-input` JSON (`docs/PATIENT_INPUT.md`); results come out as KneeScan's own JSON.

Upload a sagittal knee MRI and get:

| Output | Model | Validation |
|---|---|---|
| ACL tear (torn / intact) | MONAI 3D ResNet10, ensemble of 5 models trained on KneeMRI | Full automatic pipeline on 917 KneeMRI scans (5-fold CV grouped by patient): AUC 0.85 (95% CI 0.82–0.88), sensitivity 0.79, specificity 0.78, complete vs healthy AUC 0.94. With the hand-drawn ACL box and known knee side: AUC 0.90 |
| Tear level (partial / complete) | MONAI tear probability | AUC 0.64 for partial vs complete (low reliability): 47% of complete ruptures flagged complete, 76% of partial tears labelled partial |
| ACL location | SegFormer ACL localizer + centre-slice prior | Chosen slice inside KneeMRI's hand-marked ACL slices in 90% of scans, within one slice in 99.5%; median box overlap (IoU) 0.84 (optimistic: the localizer was likely trained on these boxes) |
| ACL length and width | MedSAM (built on Meta's Segment Anything) with a mask decoder fine-tuned on 110 expert ACL outlines (CEMRR kneeMRI-ACL) | Held-out expert slices: Dice 0.82, median length error 5%, median width error 15% |
| Femur → fibular head distance | Registration to 3 hand-labelled reference knees, then MedSAM outlines of femur, tibia and fibula | Experimental: no public ground truth exists to validate it |

**Inputs:** a `.zip` of a sagittal DICOM series, a `.dcm`, `.nii` / `.nii.gz`, a KneeMRI `.pck`, or an MRNet `.npy`.
DICOM and NIfTI carry pixel spacing, so millimetres are real; for `.pck` / `.npy` they are estimated from the reference knees.

**Not for clinical use.** Research prototype built for a hackathon.

## Run it on Google Colab (free GPU)

Open `mri_analysis/KneeScan_Colab.ipynb` in Colab, choose **Runtime → Change runtime type → T4 GPU**, add a Colab secret `HF_TOKEN` (a Hugging Face read token for this private repo), then **Runtime → Run all**. The last cell prints a public `gradio.live` link.

## Run it locally

```bash
cd mri_analysis
pip install -r requirements.txt
HF_TOKEN=hf_xxx python fetch_weights.py   # once: downloads ~470 MB of weights + 2 example scans
python app.py                             # open http://localhost:7860  (add --share for a public link)
```

The weights live in the private Hugging Face repo `tariksyed/kneescan` because they are too large for git and include
data (MRNet reference knees) that can't be redistributed; ask Tarik for read access. Only `weights/calibration.json` is versioned here.

Or from Python:

```python
from kneescan.io import load_volume
from kneescan.engine import KneeScan
d = load_volume('scan.zip')
result = KneeScan().load().analyze(d['vol'], d['spacing'])
```

## How it works

1. **ACL location.** The localizer marks the ACL region on every slice; the slice nearest the centre of the stack with the strongest detection is used (on KneeMRI the ACL lies within 2 slices of the centre in 99% of scans).
2. **Tear.** A 9-slice crop around that region goes through 5 MONAI models, each viewed four ways (both slice orders, mirrored), because the knee side is unknown.
3. **Level.** Torn scans with a tear probability above the calibrated cut-off are reported as complete ruptures.
4. **ACL size.** MedSAM outlines the ligament on the ACL slice and its neighbours; length is the outline's extent along its long axis, width the median thickness over the middle half.
5. **Femur–fibula.** The scan is aligned to three reference knees to find the lateral femoral condyle and fibular head, MedSAM outlines the bones, and the shortest gap between femur and fibula outlines is reported. Plausibility checks flag doubtful outlines.

## Data and licences

- **KneeMRI** (tear models, examples): Štajduhar et al., *Semi-automated detection of anterior cruciate ligament injury from MRI*, Comput Methods Programs Biomed 2017; Clinical Hospital Centre Rijeka. CC BY-NC-ND 4.0 (non-commercial).
- **CEMRR kneeMRI-ACL** expert ACL outlines (ACL decoder fine-tuning).
- **MRNet** (Stanford): three reference knees used for femur–fibula alignment, under the MRNet Research Use Agreement. Keep this repository private, or remove `weights/templates`, before sharing it publicly.
- **MedSAM** (Ma et al., Nature Communications 2024), built on Meta's Segment Anything; **SegFormer** ACL localizer from Hugging Face (`hajirabaloch488/knee-mri-acl-localization`); **MONAI**.

## Training and validation scripts

`training/` holds the scripts that produced the models and every number above, with the results in `training/results/`:

| Script | What it does |
|---|---|
| `prep_kneemri.py`, `km_folds.py` | Crop KneeMRI around the annotated ACL region; shared 5-fold split grouped by patient |
| `km_monai_cv.py` | MONAI 3D ResNet10 tear model (binary; `GRADE=1` for the 3-class grade model), resumable per fold |
| `km_feat2d.py`, `km_eval.py` | DINOv2 and BiomedCLIP baselines; AUCs, CIs, sensitivity/specificity, ensemble |
| `validate_e2e.py`, `e2e_side.py` | Full automatic pipeline on all 917 KneeMRI scans; effect of unknown knee side |
| `embed_cemrr.py`, `embed_cemrr_aug.py`, `aug.py`, `train_dec.py`, `msam.py` | MedSAM embeddings of the CEMRR expert ACL outlines and decoder fine-tuning with cross-validation |
| `segformer_roi.py`, `embed_mrnet.py`, `infer_mrnet.py`, `viz_acl.py`, `km_on_mrnet.py` | Runs on MRNet (localizer, ACL outlines, external tear test) |

They were run on a remote workspace and use its absolute paths (`/home/user/...`); change those to your data locations.
Data sources: KneeMRI (Zenodo record 14789903), CEMRR kneeMRI-ACL (Google Drive link in their GitHub README), MRNet (Stanford).
