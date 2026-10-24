"""Subject-level splitting and leakage audit.

Rules
* The unit of splitting is the person (subject_key), across ALL datasets that share a
  source cohort. Both knees, repeat exams and longitudinal visits stay together.
* Official benchmark splits are preserved when present (official_split field); a
  subject with conflicting official assignments is reported, not silently resolved.
* Splitting happens before any training or learned preprocessing.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .manifest import ManifestRecord
from .provenance import UNKNOWN


def assign_splits(records: list[ManifestRecord], fractions=(0.7, 0.15, 0.15), names=("train", "val", "test"), seed: int = 0,
                  preserve_official: bool = True) -> dict:
    by_subj: dict[str, list[ManifestRecord]] = defaultdict(list)
    unknown_subject = 0
    for r in records:
        if r.subject_key in (UNKNOWN, "", None):
            # Without a subject identity, grouping is impossible. Only an official split that
            # the dataset documents as subject-level may be used; otherwise the record is excluded.
            if preserve_official and r.official_split not in (UNKNOWN, "", None):
                r.split = r.official_split
            else:
                r.split = "excluded_unknown_subject"
                unknown_subject += 1
            continue
        by_subj[r.subject_key].append(r)
    conflicts = []
    free = []
    for s, rs in by_subj.items():
        off = {r.official_split for r in rs if r.official_split not in (UNKNOWN, "", None)}
        if preserve_official and len(off) == 1:
            official = next(iter(off))
            for r in rs:
                r.split = official
        elif preserve_official and len(off) > 1:
            conflicts.append({"subject_key": s, "official_splits": sorted(off)})
            for r in rs:
                r.split = "excluded_conflict"
        else:
            free.append(s)
    rng = np.random.default_rng(seed)
    free = sorted(free)
    rng.shuffle(free)
    n = len(free)
    cuts = np.cumsum([int(round(f * n)) for f in fractions[:-1]])
    groups = np.split(np.array(free, dtype=object), cuts)
    for name, g in zip(names, groups):
        for s in g:
            for r in by_subj[s]:
                r.split = name
    return {"subjects_free": n, "official_conflicts": conflicts, "excluded_unknown_subject": unknown_subject, "seed": seed, "fractions": fractions}


def audit_leakage(records: list[ManifestRecord]) -> dict:
    """Report any subject_key appearing in more than one split (within or across datasets)."""
    splits: dict[str, set] = defaultdict(set)
    datasets: dict[str, set] = defaultdict(set)
    for r in records:
        if r.subject_key in (UNKNOWN, "", None):
            continue
        if r.split not in (UNKNOWN, "excluded_conflict", "excluded_unknown_subject"):
            splits[r.subject_key].add(r.split)
        datasets[r.subject_key].add(r.dataset)
    leaks = {s: sorted(v) for s, v in splits.items() if len(v) > 1}
    cross = {s: sorted(v) for s, v in datasets.items() if len(v) > 1}
    return {"leaking_subjects": leaks, "n_leaking": len(leaks), "subjects_in_multiple_datasets": len(cross),
            "cross_dataset_examples": dict(list(cross.items())[:10]), "ok": not leaks}
