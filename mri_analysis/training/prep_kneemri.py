"""Prepare KneeMRI (Rijeka, PD fat-suppressed sagittal, 917 volumes) for ACL-tear models.
For each volume: intensity-normalise, centre on the annotated ACL region of interest, and save
  - a 3D crop: 9 slices x 80 x 80 (in-plane side = 1.3 x the ROI's longer side), slice order standardised by knee side
  - three 2D slices (ROI centre slice and its neighbours) at 224 x 224 for pretrained 2D feature extractors
usage: python -I prep_kneemri.py VOLUME_DIR METADATA_CSV OUT_DIR"""
import sys, os, json, numpy as np, pandas as pd, torch, torch.nn.functional as Fn, warnings
sys.path.insert(0, '/home/claude/acl_ext')
from safe_pck import load
warnings.filterwarnings('ignore')
vdir, mcsv, out = sys.argv[1], sys.argv[2], sys.argv[3]
md = pd.read_csv(mcsv)
D, S, S2 = 9, 80, 224


def resize(a, n):
    return Fn.interpolate(torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[:, None], size=(n, n), mode='area')[:, 0].numpy()


X3 = np.zeros((len(md), D, S, S), np.float16); X2 = np.zeros((len(md), 3, S2, S2), np.float16); meta = []
for i, r in md.iterrows():
    v = load(os.path.join(vdir, r.volumeFilename)).astype(np.float32)
    lo, hi = np.percentile(v, [1, 99.5]); v = np.clip((v - lo) / (hi - lo + 1e-6), 0, 1)
    n = v.shape[0]
    cz = int(round(r.roiZ + (r.roiDepth - 1) / 2)); cx = r.roiX + r.roiWidth / 2; cy = r.roiY + r.roiHeight / 2
    side = int(round(1.3 * max(r.roiWidth, r.roiHeight)))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    ks = np.clip(np.arange(cz - D // 2, cz + D // 2 + 1), 0, n - 1)
    if r.kneeLR == 1: ks = ks[::-1]                      # one consistent medial/lateral slice direction
    p = side + 2; pad = np.pad(v, ((0, 0), (p, p), (p, p)))
    crop = pad[ks][:, y0 + p:y0 + p + side, x0 + p:x0 + p + side]
    X3[i] = resize(crop, S).astype(np.float16)
    side2 = int(round(1.6 * max(r.roiWidth, r.roiHeight))); x2, y2 = int(round(cx - side2 / 2)), int(round(cy - side2 / 2)); p2 = side2 + 2
    pad2 = np.pad(v, ((0, 0), (p2, p2), (p2, p2)))
    ks2 = np.clip([cz - 1, cz, cz + 1], 0, n - 1)
    X2[i] = resize(pad2[ks2][:, y2 + p2:y2 + p2 + side2, x2 + p2:x2 + p2 + side2], S2).astype(np.float16)
    meta.append(dict(i=int(i), exam=int(r.examId), series=int(r.seriesNo), grade=int(r.aclDiagnosis), y=int(r.aclDiagnosis > 0),
                     knee=int(r.kneeLR), n=int(n), shape=list(v.shape), cz=cz, side=side))
    if i % 100 == 0: print('prepared', i, v.shape, flush=True)
np.save(f'{out}/km_x3.npy', X3); np.save(f'{out}/km_x2.npy', X2); json.dump(meta, open(f'{out}/km_meta.json', 'w'))
print('done', X3.shape, X2.shape, 'torn', sum(m['y'] for m in meta))
