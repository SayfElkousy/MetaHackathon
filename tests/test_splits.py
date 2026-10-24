"""Subject-level split leakage."""
from bilacl.manifest import ManifestRecord, pseudonym
from bilacl.splits import assign_splits, audit_leakage
from bilacl.training.tasks import eligibility


def _r(ds, subj, exam, side="left", visit="V0", off="unknown", cohort="oai"):
    return ManifestRecord(ds, pseudonym(cohort, subj, key=b"k"), cohort, exam, side=side, visit=visit, official_split=off)


def test_both_knees_and_visits_stay_together():
    recs = []
    for s in range(30):
        for side in ("left", "right"):
            for v in ("V0", "V1"):
                recs.append(_r("oai", str(s), f"{s}{side}{v}", side, v))
    assign_splits(recs, seed=1)
    by = {}
    for r in recs:
        by.setdefault(r.subject_key, set()).add(r.split)
    assert all(len(v) == 1 for v in by.values())
    assert audit_leakage(recs)["ok"]


def test_derived_dataset_shares_subject_identity():
    a = _r("oai", "9001", "x")
    b = _r("oai_zib", "9001", "y")  # OAI-ZIB uses the OAI namespace
    assert a.subject_key == b.subject_key
    assign_splits([a, b], seed=0)
    assert a.split == b.split


def test_audit_detects_leak():
    a, b = _r("oai", "1", "x"), _r("oai_zib", "1", "y")
    a.split, b.split = "train", "test"
    rep = audit_leakage([a, b])
    assert not rep["ok"] and rep["n_leaking"] == 1


def test_official_split_preserved_and_conflict_reported():
    a, b = _r("skm", "1", "x", off="train", cohort="skm"), _r("skm", "1", "y", off="train", cohort="skm")
    c, d = _r("skm", "2", "z", off="train", cohort="skm"), _r("skm", "2", "w", off="test", cohort="skm")
    info = assign_splits([a, b, c, d])
    assert a.split == b.split == "train"
    assert c.split == d.split == "excluded_conflict" and info["official_conflicts"]


def test_unknown_subject_not_grouped():
    rs = [ManifestRecord("mrnet", "unknown", "mrnet", f"e{i}") for i in range(5)]
    info = assign_splits(rs)
    assert all(r.split == "excluded_unknown_subject" for r in rs) and info["excluded_unknown_subject"] == 5


def test_exam_level_labels_never_eligible_for_segmentation():
    r = ManifestRecord("mrnet", "s", "mrnet", "e", available_labels="exam:acl=1", label_granularity="exam")
    for task in ("A_bone", "B_acl_intact", "C_acl_remnant"):
        assert eligibility(task, [r])["n_eligible"] == 0
    assert eligibility("D_tear_cls", [r])["n_eligible"] == 1
