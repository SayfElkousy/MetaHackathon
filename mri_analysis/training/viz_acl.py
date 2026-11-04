"""QA sheet: predicted ACL outlines on MRNet (red), earlier atlas ACL axis (cyan dashed), length line (yellow).
usage: python viz_acl.py RESULTS_JSON "key,key,..." OUT_PNG [ncol]"""
import sys, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = json.load(open(sys.argv[1])); keys = sys.argv[2].split(','); out = sys.argv[3]; ncol = int(sys.argv[4]) if len(sys.argv) > 4 else 4
S = {f"{s['split']}_{s['id']}": s for s in json.load(open('/home/claude/viewer/scans.json'))}
nr = int(np.ceil(len(keys) / ncol))
fig, ax = plt.subplots(nr, ncol * 2, figsize=(4.2 * ncol * 2, 4.4 * nr)); ax = np.atleast_2d(ax)
for i, key in enumerate(keys):
    r = R[key]; s = S[key]; sp, sid = key.split('_')
    v = np.load(f'/mnt/user-data/uploads/MRNet-v1.0/{sp}/sagittal/{sid}.npy').astype(float)
    a0, a1 = ax[i // ncol, (i % ncol) * 2], ax[i // ncol, (i % ncol) * 2 + 1]
    for a in (a0, a1): a.axis('off')
    if 'poly' not in r: a0.set_title(f'{key}: no outline', fontsize=8); continue
    img = v[r['slice']]; P = np.array(r['poly']); cx, cy = P[:, 0].mean(), P[:, 1].mean(); H = 42
    lo, hi = np.percentile(img[int(cy - H):int(cy + H), int(cx - H):int(cx + H)], [1, 99.5])
    for a in (a0, a1):
        a.imshow(img, cmap='gray', vmin=lo, vmax=hi, interpolation='lanczos'); a.set_xlim(cx - H, cx + H); a.set_ylim(cy + H, cy - H)
    a1.plot(np.r_[P[:, 0], P[0, 0]], np.r_[P[:, 1], P[0, 1]], '-', color='red', lw=1.2)
    a1.plot([r['end0'][0], r['end1'][0]], [r['end0'][1], r['end1'][1]], '-', color='yellow', lw=0.8)
    a1.plot([s['F'][0], s['T'][0]], [s['F'][1], s['T'][1]], '--', color='cyan', lw=0.8)
    lab = {0: 'intact', 1: 'TORN'}.get(s['label'], 'unlabelled')
    a0.set_title(f"{key} sl{r['slice']} {lab}", fontsize=8)
    a1.set_title(f"L {r['length_px']} W {r['width_px']} px iou {r['iou_pred']} agr {r.get('prompt_agreement')}", fontsize=7,
                 color='k' if r.get('ok', True) else 'crimson')
for j in range(len(keys), nr * ncol):
    for a in ax[j // ncol, (j % ncol) * 2:(j % ncol) * 2 + 2]: a.axis('off')
plt.tight_layout(); plt.savefig(out, dpi=60)
