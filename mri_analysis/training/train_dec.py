"""Fine-tune MedSAM's prompt-conditioned mask decoder on the CEMRR kneeMRI-ACL expert outlines (frozen image encoder,
precomputed embeddings). Box prompts are jittered/loosened so the model learns to find the ACL inside a loose box.
usage:
  python -I train_dec.py cv      -> 5-fold cross-validation by image: Dice and length/width error vs the expert outline
  python -I train_dec.py final   -> train on all 110 images, save decoder weights to models/acl_decoder.pt"""
import sys, json, time, copy, numpy as np, torch, torch.nn.functional as Fn
sys.path.insert(0, '/home/claude/acl_ext')
from msam import model as get_model
from acl_measure import measure
torch.manual_seed(0)
D = '/home/claude/acl_ext/emb'
E = np.load(f'{D}/cemrr_emb.npy', mmap_mode='r'); M = np.load(f'{D}/cemrr_masks.npy'); meta = json.load(open(f'{D}/cemrr_meta.json'))
N = len(meta); E = np.asarray(E[:N], dtype=np.float16)   # kept fp16 in memory, cast per batch
import os
if os.environ.get('AUG') and os.path.exists(f'{D}/cemrr_aug_meta.json'):          # low-res / contrast-randomised variants
    m2 = json.load(open(f'{D}/cemrr_aug_meta.json'))
    E = np.concatenate([E, np.asarray(np.load(f'{D}/cemrr_aug_emb.npy', mmap_mode='r')[:len(m2)], dtype=np.float16)])
    M = np.concatenate([M, np.load(f'{D}/cemrr_aug_masks.npy')[:len(m2)]]); meta = meta + m2; N = len(meta)
    print('with augmented variants:', N, flush=True)
ids = sorted(set(m['id'] for m in meta), key=int); idx_of = {i: [j for j, m in enumerate(meta) if m['id'] == i] for i in ids}


def bbox(m):
    ys, xs = np.nonzero(m); return np.array([xs.min(), ys.min(), xs.max(), ys.max()], float)


B = np.stack([bbox(m) for m in M])


def loosen(b, rng=None, expand=None, shift=0.12, lo=0.0, hi=0.7):
    w, h = b[2] - b[0] + 1, b[3] - b[1] + 1
    e = (rng.uniform(lo, hi, 4) if expand is None else np.full(4, expand)) * np.array([w, h, w, h])
    s = (rng.uniform(-shift, shift, 2) if rng is not None else np.zeros(2)) * np.array([w, h])
    return np.clip(np.array([b[0] - e[0] + s[0], b[1] - e[1] + s[1], b[2] + e[2] + s[0], b[3] + e[3] + s[1]]), 0, 255)


def forward(m, emb, boxes):
    out = m(image_embeddings=emb, input_boxes=torch.tensor(np.asarray(boxes) * 4.0, dtype=torch.float32)[:, None, :], multimask_output=False)
    return out.pred_masks[:, 0, 0], out.iou_scores[:, 0, 0]


def dice_loss(logit, y):
    p = torch.sigmoid(logit); inter = (p * y).sum((1, 2)); return (1 - (2 * inter + 1) / (p.sum((1, 2)) + y.sum((1, 2)) + 1)).mean()


def train(tr_idx, epochs=15, bs=8, lr=2e-4, seed=0):
    m = copy.deepcopy(base)
    for p in m.parameters(): p.requires_grad = False
    for p in m.mask_decoder.parameters(): p.requires_grad = True
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = np.random.default_rng(seed); m.train()
    for ep in range(epochs):
        order = rng.permutation(tr_idx); tot = 0
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            emb = torch.from_numpy(E[b].astype(np.float32)); y = torch.from_numpy(M[b].astype(np.float32))
            boxes = [loosen(B[j], rng) for j in b]
            logit, _ = forward(m, emb, boxes)
            loss = Fn.binary_cross_entropy_with_logits(logit, y) + dice_loss(logit, y)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(b)
        sched.step()
        if ep % 5 == 0 or ep == epochs - 1: print(f'  epoch {ep} loss {tot / len(order):.3f}', flush=True)
    return m.eval()


def evaluate(m, te_idx, expand):
    res = []
    with torch.no_grad():
        for j in te_idx:
            logit, iou = forward(m, torch.from_numpy(E[j:j + 1].astype(np.float32)), [loosen(B[j], expand=expand)])
            pr = (torch.sigmoid(logit[0]) > 0.5).numpy(); gt = M[j] > 0
            d = 2 * (pr & gt).sum() / max(1, pr.sum() + gt.sum())
            a, g = measure(pr), measure(gt)
            res.append(dict(idx=int(j), id=meta[j]['id'], dice=float(d), iou_pred=float(iou[0]),
                            len_pred=a and a['length'], len_gt=g and g['length'], wid_pred=a and a['width_mid'], wid_gt=g and g['width_mid']))
    return res


base = get_model()
if __name__ == '__main__':
    mode = sys.argv[1]
    if mode in ('cv', 'cv1'):
        rng = np.random.default_rng(0); perm = rng.permutation(ids); folds = [perm[f::5] for f in range(5)]
        allres = {'zero_shot_tight': [], 'zero_shot_loose': [], 'tight': [], 'loose': [], 'very_loose': [], 'loose_lowres_contrast': []}
        for f, te_ids in enumerate(folds[:1] if mode == 'cv1' else folds):
            te = [idx_of[i][0] for i in te_ids]                    # evaluate on the original (unflipped) image
            tr = [j for i in ids if i not in set(te_ids) for j in idx_of[i]]
            t0 = time.time(); print(f'fold {f}: train {len(tr)} test {len(te)}', flush=True)
            allres['zero_shot_tight'] += evaluate(base, te, 0.05); allres['zero_shot_loose'] += evaluate(base, te, 0.4)
            m = train(tr)
            allres['tight'] += evaluate(m, te, 0.05); allres['loose'] += evaluate(m, te, 0.4); allres['very_loose'] += evaluate(m, te, 0.7)
            te_aug = [j for i in te_ids for j in idx_of[i] if meta[j]['var'] in ('c', 'd')]
            if te_aug: allres['loose_lowres_contrast'] += evaluate(m, te_aug, 0.4)
            print(f'  fold {f} done in {time.time() - t0:.0f}s, loose dice {np.mean([r["dice"] for r in allres["loose"][-len(te):]]):.3f}', flush=True)
        summ = {}
        for k, r in allres.items():
            if not r: continue
            d = np.array([x['dice'] for x in r])
            ok = [x for x in r if x['len_pred'] and x['len_gt'] and x['wid_pred'] and x['wid_gt']]
            le = np.array([abs(x['len_pred'] - x['len_gt']) / x['len_gt'] for x in ok]); we = np.array([abs(x['wid_pred'] - x['wid_gt']) / x['wid_gt'] for x in ok])
            summ[k] = dict(n=len(r), dice_mean=round(float(d.mean()), 3), dice_median=round(float(np.median(d)), 3), dice_q10=round(float(np.percentile(d, 10)), 3),
                           length_err_pct_median=round(float(np.median(le) * 100), 1), width_err_pct_median=round(float(np.median(we) * 100), 1),
                           length_corr=round(float(np.corrcoef([x['len_pred'] for x in ok], [x['len_gt'] for x in ok])[0, 1]), 3),
                           width_corr=round(float(np.corrcoef([x['wid_pred'] for x in ok], [x['wid_gt'] for x in ok])[0, 1]), 3))
        print(json.dumps(summ, indent=1)); json.dump(dict(summary=summ, per_image=allres), open(os.environ.get('CVOUT', '/home/claude/acl_ext/cv_results.json'), 'w'))
    else:
        m = train([j for i in ids for j in idx_of[i]])
        import os; os.makedirs('/home/claude/acl_ext/models', exist_ok=True)
        torch.save(m.mask_decoder.state_dict(), os.environ.get('MODELOUT', '/home/claude/acl_ext/models/acl_decoder.pt')); print('saved')
