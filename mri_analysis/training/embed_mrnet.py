"""MedSAM embeddings of square crops around the ACL on MRNet sagittal slices (original file order, k from scans.json).
Crop centre = external ACL localizer box centre on slice k (fallback: earlier atlas ACL midpoint); side = 1.6 x the
localizer box's longer side (clamped 80-170 px). The same window is used for slices k-1 and k+1.
usage: python -I embed_mrnet.py part nparts which    which = center | sides. Resumable."""
import sys, os, json, numpy as np
sys.path.insert(0, '/home/user/acl_ext')
from msam import embed
part, nparts, which = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
O = '/home/user/acl_ext/emb/mrnet'
R = json.load(open('/home/user/acl_ext/segformer_rois.json'))
scans = sorted(json.load(open('/home/user/viewer/scans.json')), key=lambda s: (s['split'], s['id']))


def window(s):
    roi = R[f"{s['split']}_{s['id']}"]['rois'].get(str(s['k']))
    if roi:
        b = roi['box']; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; side = 1.6 * max(b[2] - b[0] + 1, b[3] - b[1] + 1); src = 'localizer'
    else:
        cx, cy = (s['F'][0] + s['T'][0]) / 2, (s['F'][1] + s['T'][1]) / 2; side = 2.2 * np.hypot(s['F'][0] - s['T'][0], s['F'][1] - s['T'][1]); src = 'atlas'
    side = int(round(np.clip(side, 80, 170)))
    return int(round(cx - side / 2)), int(round(cy - side / 2)), side, src


def crop(a, x0, y0, s):
    p = s + 2; ap = np.pad(a, p); return ap[y0 + p:y0 + p + s, x0 + p:x0 + p + s]


meta_f = f'{O}/windows.json'
W = json.load(open(meta_f)) if os.path.exists(meta_f) else {}
for i, s in enumerate(scans):
    key = f"{s['split']}_{s['id']}"
    if key not in W:
        x0, y0, side, src = window(s); W[key] = dict(x0=x0, y0=y0, side=side, src=src, k=int(s['k']))
if part == 0: json.dump(W, open(meta_f, 'w'))
for i, s in enumerate(scans):
    if i % nparts != part: continue
    key = f"{s['split']}_{s['id']}"; f = f"{O}/{key}_{which}.npy"
    if os.path.exists(f): continue
    v = np.load(f"/mnt/user-data/uploads/MRNet-v1.0/{s['split']}/sagittal/{s['id']}.npy").astype(np.float32)
    w = W[key]; k = w['k']; ks = (k,) if which == 'center' else (k - 1, k + 1)
    ks = [min(max(kk, 0), v.shape[0] - 1) for kk in ks]
    E = np.stack([embed(crop(v[kk], w['x0'], w['y0'], w['side']))[0].numpy().astype(np.float16) for kk in ks])
    np.save(f + '.tmp.npy', E); os.replace(f + '.tmp.npy', f)
    print('done', i, key, flush=True)
