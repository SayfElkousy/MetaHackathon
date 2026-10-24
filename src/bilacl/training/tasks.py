"""Task definitions and dataset eligibility rules.

Each task declares the label codes it learns and the evidence a record must have.
Examination-level labels never qualify for segmentation tasks. Intact-ACL and
torn-remnant segmentation are separate tasks (different distributions): a model
trained for B must be evaluated on C data before being used for remnants.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from ..annotation.schema import AnnotationRecord, Label
from ..manifest import ManifestRecord


@dataclasses.dataclass
class Task:
    key: str
    description: str
    kind: str  # segmentation | classification | registration
    classes: dict  # project label code -> task class index (segmentation)
    required_annotated: tuple  # project labels that must be in annotated_labels
    allowed_tear_locations: tuple | None = None  # None = any


TASKS = {
    "A_bone": Task("A_bone", "Femur and tibia segmentation", "segmentation", {int(Label.FEMUR): 1, int(Label.TIBIA): 2},
                   (int(Label.FEMUR), int(Label.TIBIA))),
    "B_acl_intact": Task("B_acl_intact", "Intact ACL segmentation", "segmentation", {int(Label.ACL_INTACT): 1}, (int(Label.ACL_INTACT),), ("intact",)),
    "C_acl_remnant": Task("C_acl_remnant", "Torn-ACL remnant segmentation", "segmentation",
                          {int(Label.ACL_REMNANT_FEMORAL): 1, int(Label.ACL_REMNANT_TIBIAL): 2, int(Label.ACL_FRAGMENT_OTHER): 3},
                          (int(Label.ACL_REMNANT_FEMORAL), int(Label.ACL_REMNANT_TIBIAL)),
                          ("proximal_avulsion", "proximal", "midsubstance", "distal", "distal_avulsion", "partial")),
    "D_tear_cls": Task("D_tear_cls", "Exam-level ACL tear classification (auxiliary)", "classification", {}, ()),
    "E_bilateral": Task("E_bilateral", "Bilateral native-anatomy estimation", "registration", {}, (int(Label.FEMUR), int(Label.TIBIA))),
}


def eligibility(task_key: str, records: list[ManifestRecord], allow_synthetic: bool = False) -> dict:
    t = TASKS[task_key]
    ok, excluded = [], []
    for r in records:
        why = []
        if "SYNTHETIC" in r.notes and not allow_synthetic:
            why.append("synthetic record (allowed only for smoke tests)")
        if t.kind == "segmentation":
            if r.label_granularity != "voxel":
                why.append(f"label granularity {r.label_granularity!r}: segmentation requires voxel labels (exam-level/bbox labels never qualify)")
            else:
                ann_p = Path(str(r.label_path).replace(".nii.gz", ".annotation.json"))
                if not ann_p.exists():
                    why.append("no annotation record (annotator/version/annotated_labels unknown)")
                else:
                    a = AnnotationRecord.load(ann_p)
                    missing = set(t.required_annotated) - set(a.annotated_labels)
                    if missing:
                        why.append(f"structures not annotated: {[Label(m).name for m in missing]}")
                    if not a.usable_as_ground_truth:
                        why.append(f"annotation not usable as ground truth (source={a.source}, review={a.review_status.value})")
                    if t.allowed_tear_locations and a.tear_location not in t.allowed_tear_locations:
                        why.append(f"tear_location {a.tear_location!r} not in task distribution {t.allowed_tear_locations}")
        elif t.kind == "classification":
            if r.label_granularity != "exam" or "exam:acl" not in r.available_labels:
                why.append("no exam-level ACL tear label")
        (excluded if why else ok).append((r, why))
    return {"task": task_key, "eligible": [r for r, _ in ok], "excluded": [(r.exam_id, w) for r, w in excluded],
            "n_eligible": len(ok), "n_excluded": len(excluded)}


def blocked_task_report(records: list[ManifestRecord]) -> dict:
    out = {}
    for k in TASKS:
        e = eligibility(k, records)
        out[k] = {"description": TASKS[k].description, "n_eligible_real": e["n_eligible"],
                  "status": "runnable" if e["n_eligible"] >= 2 else "BLOCKED: insufficient eligible labelled data",
                  "top_exclusion_reasons": _top_reasons(e["excluded"])}
    return out


def _top_reasons(excl):
    from collections import Counter

    c = Counter(w for _, ws in excl for w in ws)
    return dict(c.most_common(5))


def dump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=str))
