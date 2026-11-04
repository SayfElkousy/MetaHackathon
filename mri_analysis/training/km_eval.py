"""Evaluate KneeMRI ACL-tear models from their out-of-fold predictions (5-fold, grouped by exam).
Reports AUC with 95% bootstrap CI, sensitivity/specificity at the Youden threshold, detection by tear grade, and the
pre-specified ensemble (mean percentile rank of all models). Writes km/km_results.json and km/km_predictions.csv.
usage: python -I km_eval.py name=path.npy [name=path.npy ...]"""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, '/home/claude/acl_ext')
from km_folds import O
from sklearn.metrics import roc_auc_score, roc_curve
meta = json.load(open(f'{O}/km_meta.json')); y = np.array([m['y'] for m in meta]); grade = np.array([m['grade'] for m in meta])
P = {a.split('=')[0]: np.load(a.split('=')[1]) for a in sys.argv[1:]}
rank = lambda p: (np.argsort(np.argsort(p)) + 0.5) / len(p)
if len(P) > 1: P['ensemble'] = np.mean([rank(p) for p in P.values()], 0)
rng = np.random.default_rng(0); B = [rng.integers(0, len(y), len(y)) for _ in range(2000)]


def summarise(p):
    auc = roc_auc_score(y, p); bs = [roc_auc_score(y[b], p[b]) for b in B]
    f, t, th = roc_curve(y, p); j = int(np.argmax(t - f)); thr = th[j]; pred = p >= thr
    hc = grade != 1                                                     # healthy + complete only
    return dict(auc=round(auc, 3), ci=[round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
                threshold=float(thr), sensitivity=round(float(pred[y == 1].mean()), 3), specificity=round(float((~pred[y == 0]).mean()), 3),
                detected_partial=round(float(pred[grade == 1].mean()), 3), detected_complete=round(float(pred[grade == 2].mean()), 3),
                auc_complete_vs_healthy=round(roc_auc_score(y[hc], p[hc]), 3))


res = {k: summarise(v) for k, v in P.items()}
res['_n'] = dict(volumes=len(y), torn=int(y.sum()), partial=int((grade == 1).sum()), complete=int((grade == 2).sum()), exams=len({m['exam'] for m in meta}))
print(json.dumps(res, indent=1)); json.dump(res, open(f'{O}/km_results.json', 'w'), indent=1)
df = pd.DataFrame(dict(volume=[f"{m['exam']}-{m['series']}" for m in meta], acl_grade=grade, torn=y, **{f'prob_{k}': np.round(v, 4) for k, v in P.items()}))
df.to_csv(f'{O}/km_predictions.csv', index=False)
