"""External ACL localizer (hajirabaloch488/knee-mri-acl-localization, SegFormer-b0): ACL region box on the ACL slice
(original file order, as in scans.json) and its neighbours, for every sagittal scan.
usage: python -I segformer_roi.py MODEL_DIR OUT_JSON"""
import sys, json, numpy as np, torch
from scipy import ndimage as ndi
from transformers import SegformerForSemanticSegmentation
torch.set_num_threads(1)
mdir, outp = sys.argv[1], sys.argv[2]
m = SegformerForSemanticSegmentation.from_pretrained(mdir, num_labels=2).eval()
scans = json.load(open('/home/user/viewer/scans.json')); out = {}
for s in scans:
    v = np.load(f"/mnt/user-data/uploads/MRNet-v1.0/{s['split']}/sagittal/{s['id']}.npy").astype(np.float32)
    k = int(s['k']); res = {}
    for kk in (k - 1, k, k + 1):
        if kk < 0 or kk >= v.shape[0]: continue
        img = v[kk]; x = (img - img.min()) / (img.max() - img.min() + 1e-6)
        t = torch.nn.functional.interpolate(torch.from_numpy(x)[None, None], size=(224, 224), mode='bilinear', align_corners=False).repeat(1, 3, 1, 1)
        with torch.no_grad():
            p = torch.softmax(torch.nn.functional.interpolate(m(pixel_values=t).logits, size=(256, 256), mode='bilinear', align_corners=False), 1)[0, 1].numpy()
        lab, n = ndi.label(p > 0.5)
        if n == 0: res[kk] = None; continue
        sz = ndi.sum(np.ones_like(p), lab, range(1, n + 1)); big = lab == 1 + int(np.argmax(sz))
        ys, xs = np.nonzero(big)
        res[kk] = dict(box=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], pmax=round(float(p.max()), 3), area=int(big.sum()))
    out[f"{s['split']}_{s['id']}"] = dict(k=k, rois=res)
json.dump(out, open(outp, 'w')); print('done', len(out))
