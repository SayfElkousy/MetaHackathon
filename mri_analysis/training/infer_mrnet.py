"""Outline and measure the ACL on every MRNet sagittal scan with MedSAM + the decoder fine-tuned on CEMRR expert outlines.
Two box prompts per slice: the external ACL localizer box, and a box around the earlier atlas ACL axis; their agreement
is one of the quality checks. Uses slice k and, if embedded, k-1 / k+1 (picks the slice with the most confident,
longest plausible outline).
usage: python -I infer_mrnet.py OUT_JSON"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, '/home/user/acl_ext')
from msam import model as get_model
from acl_measure import measure, largest
from skimage import measure as skm
outp = sys.argv[1]
O = '/home/user/acl_ext/emb/mrnet'
m = get_model(); m.mask_decoder.load_state_dict(torch.load(os.environ.get('DECODER', '/home/user/acl_ext/models/acl_decoder.pt'), weights_only=True)); m.eval()
W = json.load(open(f'{O}/windows.json'))
R = json.load(open('/home/user/acl_ext/segformer_rois.json'))
S = {f"{s['split']}_{s['id']}": s for s in json.load(open('/home/user/viewer/scans.json'))}


def to_crop(b, w):
    f = 256.0 / w['side']; return np.clip([(b[0] - w['x0']) * f, (b[1] - w['y0']) * f, (b[2] - w['x0']) * f, (b[3] - w['y0']) * f], 0, 255)


def predict(emb, box):
    with torch.no_grad():
        out = m(image_embeddings=torch.from_numpy(emb[None].astype(np.float32)), input_boxes=torch.tensor([[box * 4.0]], dtype=torch.float32), multimask_output=False)
    p = torch.sigmoid(out.pred_masks[0, 0, 0]).numpy(); return largest(p > 0.5), float(out.iou_scores[0, 0, 0])


res = {}
for key, s in S.items():
    w = W[key]; k = w['k']; n = s['n']
    if not os.path.exists(f'{O}/{key}_center.npy'): continue
    embs = {k: np.load(f'{O}/{key}_center.npy')[0]}
    if os.path.exists(f'{O}/{key}_sides.npy'):
        e2 = np.load(f'{O}/{key}_sides.npy'); embs[max(k - 1, 0)] = e2[0]; embs[min(k + 1, n - 1)] = e2[1]
    roi = R[key]['rois'].get(str(k))
    F, T = np.array(s['F']), np.array(s['T'])
    abox = [min(F[0], T[0]) - 8, min(F[1], T[1]) - 8, max(F[0], T[0]) + 8, max(F[1], T[1]) + 8]
    prompts = {'localizer': roi['box'] if roi else None, 'atlas': abox}
    best = None
    for kk, emb in embs.items():
        out = {}
        for nm, b in prompts.items():
            if b is None: continue
            mk, iou = predict(emb, to_crop(b, w)); out[nm] = (mk, iou)
        nm = 'localizer' if 'localizer' in out else 'atlas'
        mk, iou = out[nm]
        agree = None
        if len(out) == 2:
            a, b2 = out['localizer'][0], out['atlas'][0]; agree = float(2 * (a & b2).sum() / max(1, a.sum() + b2.sum()))
        met = measure(mk)
        if met is None: continue
        f = w['side'] / 256.0
        cand = dict(slice=int(kk), prompt=nm, iou_pred=round(iou, 3), prompt_agreement=None if agree is None else round(agree, 3),
                    length_px=round(met['length'] * f, 1), width_px=round(met['width_mid'] * f, 1) if met['width_mid'] else None,
                    width_mean_px=round(met['width_mean'] * f, 1), angle=round(met['angle'], 1), elong=round(met['elong'], 2),
                    touches_edge=bool(mk[:2].any() or mk[-2:].any() or mk[:, :2].any() or mk[:, -2:].any()),
                    end0=[round(w['x0'] + met['end0'][0] * f, 1), round(w['y0'] + met['end0'][1] * f, 1)],
                    end1=[round(w['x0'] + met['end1'][0] * f, 1), round(w['y0'] + met['end1'][1] * f, 1)])
        cs = skm.find_contours(mk.astype(float), 0.5)
        if cs:
            c = skm.approximate_polygon(max(cs, key=len), tolerance=0.8)
            cand['poly'] = [[round(w['x0'] + x * f, 1), round(w['y0'] + y * f, 1)] for y, x in c]
        score = iou + 0.002 * met['length'] + (0.2 * agree if agree is not None else 0)
        if best is None or score > best[0]: best = (score, cand)
    res[key] = best[1] if best else dict(error='no outline')
json.dump(res, open(outp, 'w')); print('scans', len(res), 'with outline', sum('length_px' in v for v in res.values()))
