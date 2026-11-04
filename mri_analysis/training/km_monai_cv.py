"""MONAI 3D ResNet10 ACL classifier on KneeMRI ACL-region crops (9 x 80 x 80), 5-fold grouped CV.
binary (default): tear = partially or completely ruptured vs healthy.  GRADE=1: 3 classes healthy / partial / complete.
Writes out-of-fold probabilities and fold weights.
usage: [GRADE=1] python -I km_monai_cv.py [seed]"""
import sys, os, json, time, numpy as np, torch
sys.path.insert(0, '/home/claude/acl_ext')
from km_folds import folds, O
from monai.networks.nets import resnet10
from monai.transforms import Compose, RandFlip, RandAffine, RandGaussianNoise, RandAdjustContrast, RandScaleIntensity
from monai.utils import set_determinism
from sklearn.metrics import roc_auc_score
torch.set_num_threads(2)
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
set_determinism(seed=seed)
X = np.load(f'{O}/km_x3.npy')[:, None]                                # (N,1,9,80,80) fp16
F, y, meta = folds()
GRADE = bool(os.environ.get('GRADE')); g = np.array([m['grade'] for m in meta]); tag = 'grade' if GRADE else 'monai'
NC = 3 if GRADE else 1
EPOCHS, BS = 30, 16
aug = Compose([RandFlip(prob=0.5, spatial_axis=2),                    # anterior-posterior mirror (left/right knee look)
               RandAffine(prob=0.8, rotate_range=(0.17, 0, 0), scale_range=(0, 0.1, 0.1), translate_range=(1, 6, 6), padding_mode='zeros'),
               RandGaussianNoise(prob=0.3, std=0.03), RandAdjustContrast(prob=0.3, gamma=(0.8, 1.25)), RandScaleIntensity(prob=0.3, factors=0.1)])


def make(): return resnet10(spatial_dims=3, n_input_channels=1, num_classes=NC, widen_factor=0.5)


def predict(m, idx):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(idx), 32):
            xb = torch.from_numpy(X[idx[i:i + 32]].astype(np.float32))
            act = (lambda z: torch.softmax(z, 1)) if GRADE else torch.sigmoid
            p = act(m(xb)) + act(m(torch.flip(xb, dims=[4])))
            out.append((p / 2).numpy() if GRADE else (p / 2).squeeze(1).numpy())
    return np.concatenate(out)


oof = np.zeros((len(y), 3)) if GRADE else np.zeros(len(y)); log = []
score = lambda p: p[:, 1:].sum(1) if GRADE else p
ONLY = [int(x) for x in os.environ['FOLDS'].split(',')] if os.environ.get('FOLDS') else None   # train only these folds
for f, (ti, vi) in enumerate(F):
    wf = f'{O}/{tag}_s{seed}_f{f}.pt'
    if os.path.exists(wf) or (ONLY is not None and f not in ONLY):             # resumable: reuse saved fold weights
        if not os.path.exists(wf): continue
        m = make(); m.load_state_dict(torch.load(wf, weights_only=True)); oof[vi] = predict(m, vi)
        log.append(dict(fold=f, auc=float(roc_auc_score(y[vi], score(oof[vi]))), reused=True)); continue
    m = make(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=EPOCHS * int(np.ceil(len(ti) / BS)))
    pw = torch.tensor([(y[ti] == 0).sum() / max(1, (y[ti] == 1).sum())]); t0 = time.time()
    cw = torch.tensor([len(ti) / (3 * max(1, (g[ti] == c).sum())) for c in range(3)], dtype=torch.float32)
    for ep in range(EPOCHS):
        m.train(); perm = np.random.RandomState(ep + 100 * f + 1000 * seed).permutation(ti); tot = 0
        for i in range(0, len(perm), BS):
            bi = perm[i:i + BS]
            xb = torch.stack([torch.as_tensor(aug(X[j].astype(np.float32))) for j in bi]).float()
            if GRADE: opt.zero_grad(); loss = torch.nn.functional.cross_entropy(m(xb), torch.from_numpy(g[bi]).long(), weight=cw)
            else: opt.zero_grad(); loss = torch.nn.functional.binary_cross_entropy_with_logits(m(xb), torch.from_numpy(y[bi]).float()[:, None], pos_weight=pw)
            loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(bi)
        if ep % 10 == 9 or ep == EPOCHS - 1:
            print(f'fold {f} ep {ep + 1} loss {tot / len(ti):.3f} heldout tear AUC {roc_auc_score(y[vi], score(predict(m, vi))):.3f} ({time.time() - t0:.0f}s)', flush=True)
    oof[vi] = predict(m, vi); torch.save(m.state_dict(), f'{O}/{tag}_s{seed}_f{f}.pt')
    log.append(dict(fold=f, auc=float(roc_auc_score(y[vi], score(oof[vi])))))
if len(log) < len(F): print('incomplete:', len(log), 'of', len(F), 'folds'); sys.exit(0)
np.save(f'{O}/oof_{tag}_s{seed}.npy', oof)
print(json.dumps(dict(auc_oof=float(roc_auc_score(y, score(oof))), folds=log), indent=1))
