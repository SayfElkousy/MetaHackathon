"""End-to-end check on KneeMRI with the automatic ACL localizer instead of the dataset's hand-drawn ROI.
For every scan: localizer -> ACL slice + box -> tear crop -> the MONAI fold model that never trained on that scan
(knee side treated as unknown, as in the deployed system). Reports slice accuracy, box overlap with the hand ROI,
and tear AUC. usage: python -I validate_e2e.py OUT_JSON"""
import sys, os, json, time, numpy as np, pandas as pd, torch, warnings
sys.path.insert(0, '/home/claude/kneescan_system'); sys.path.insert(0, '/home/claude/acl_ext')
warnings.filterwarnings('ignore')
from kneescan.engine import KneeScan, _norm
from kneescan.safe_pck import load
from km_folds import folds
from sklearn.metrics import roc_auc_score
from monai.networks.nets import resnet10
torch.set_num_threads(2)
outp = sys.argv[1]
ks = KneeScan(); from transformers import SegformerForSemanticSegmentation
ks.loc = SegformerForSemanticSegmentation.from_pretrained('/home/claude/kneescan_system/weights/acl_localizer', num_labels=2).eval()
F, y, meta = folds(); fold_of = {int(i): f for f, (_, vi) in enumerate(F) for i in vi}
models = []
for f in range(5):
    m = resnet10(spatial_dims=3, n_input_channels=1, num_classes=1, widen_factor=0.5)
    m.load_state_dict(torch.load(f'/home/claude/acl_ext/km/monai_s0_f{f}.pt', weights_only=True)); models.append(m.eval())
md = pd.read_csv('/home/claude/ext_data/kneemri/metadata.csv')
res = json.load(open(outp)) if os.path.exists(outp) else {}
t0 = time.time()
for i, r in md.iterrows():
    if str(i) in res: continue
    v = _norm(load(os.path.join('/home/claude/ext_data/kneemri/x/volumetric_data', r.volumeFilename)).astype(np.float32))
    loc = ks.localize(v)
    if loc is None: res[str(i)] = dict(fail=True); continue
    xb = ks._tear_crop(v, loc['slice'], loc['box']); m = models[fold_of[i]]
    with torch.no_grad():
        p = float(np.mean([torch.sigmoid(m(t)).item() for t in (xb, torch.flip(xb, dims=[2]), torch.flip(xb, dims=[4]), torch.flip(xb, dims=[2, 4]))]))
    x0, y0, x1, y1 = loc['box']; gx0, gy0, gx1, gy1 = r.roiX, r.roiY, r.roiX + r.roiWidth - 1, r.roiY + r.roiHeight - 1
    ix = max(0, min(x1, gx1) - max(x0, gx0) + 1) * max(0, min(y1, gy1) - max(y0, gy0) + 1)
    iou = ix / ((x1 - x0 + 1) * (y1 - y0 + 1) + r.roiWidth * r.roiHeight - ix)
    res[str(i)] = dict(slice=loc['slice'], roi_z=[int(r.roiZ), int(r.roiZ + r.roiDepth - 1)], box_iou=round(float(iou), 3), p=round(p, 4), y=int(y[i]), grade=int(r.aclDiagnosis))
    if i % 50 == 0:
        json.dump(res, open(outp, 'w')); print(i, round(time.time() - t0), flush=True)
json.dump(res, open(outp, 'w'))
R = [v for v in res.values() if not v.get('fail')]
inz = np.array([v['roi_z'][0] <= v['slice'] <= v['roi_z'][1] for v in R]); dz = np.array([min(abs(v['slice'] - v['roi_z'][0]), abs(v['slice'] - v['roi_z'][1])) * (not ok) for v, ok in zip(R, inz)])
iou = np.array([v['box_iou'] for v in R]); yy = np.array([v['y'] for v in R]); pp = np.array([v['p'] for v in R])
summ = dict(n=len(R), failed=len(res) - len(R), slice_in_hand_roi=round(float(inz.mean()), 3), slice_within_1=round(float((dz <= 1).mean()), 3),
            box_iou_median=round(float(np.median(iou)), 3), box_iou_ge_0_5=round(float((iou >= 0.5).mean()), 3), tear_auc_end_to_end=round(roc_auc_score(yy, pp), 3))
print(json.dumps(summ)); json.dump(dict(summary=summ, per_scan=res), open(outp, 'w'))
