"""Shared 5-fold split for KneeMRI: stratified by tear (partial or complete vs healthy), grouped by exam so a patient's
series never sit on both sides of a split."""
import json, numpy as np
from sklearn.model_selection import StratifiedGroupKFold
O = '/home/claude/acl_ext/km'


def folds():
    meta = json.load(open(f'{O}/km_meta.json'))
    y = np.array([m['y'] for m in meta]); g = np.array([m['exam'] for m in meta])
    return list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(np.zeros(len(y)), y, g)), y, meta
