"""Imaging manifest: one row per (exam, series). Unknown fields stay "unknown".

Subject IDs are pseudonymised with a keyed hash (HMAC-SHA256) using a local salt
stored outside source control (data/.bilacl_salt by default). The hash namespace is
the *source cohort*, so derived datasets (e.g. OAI-ZIB from OAI) map the same person
to the same pseudonymous subject_key and leakage can be detected across datasets.
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import hmac
import json
import os
import secrets
from pathlib import Path

from .provenance import UNKNOWN

FIELDS = [
    "dataset", "subject_key", "cohort", "exam_id", "series_id", "side", "side_source", "visit", "sequence", "plane",
    "scanner_manufacturer", "scanner_model", "field_strength_T", "spacing_mm", "orientation", "physical_verified",
    "image_path", "label_path", "available_labels", "label_granularity", "annotation_review", "official_split",
    "split", "provenance", "notes",
]


@dataclasses.dataclass
class ManifestRecord:
    dataset: str
    subject_key: str
    cohort: str
    exam_id: str
    series_id: str = UNKNOWN
    side: str = UNKNOWN  # left | right | unknown
    side_source: str = UNKNOWN
    visit: str = UNKNOWN
    sequence: str = UNKNOWN
    plane: str = UNKNOWN
    scanner_manufacturer: str = UNKNOWN
    scanner_model: str = UNKNOWN
    field_strength_T: str = UNKNOWN
    spacing_mm: str = UNKNOWN
    orientation: str = UNKNOWN
    physical_verified: str = UNKNOWN  # "true" | "false" | unknown
    image_path: str = UNKNOWN
    label_path: str = UNKNOWN
    available_labels: str = ""  # ';'-joined label kinds, e.g. "voxel:femur;voxel:tibia" or "exam:acl_tear"
    label_granularity: str = UNKNOWN  # voxel | bbox | exam | none
    annotation_review: str = UNKNOWN
    official_split: str = UNKNOWN
    split: str = UNKNOWN
    provenance: str = ""
    notes: str = ""

    def labels(self) -> set[str]:
        return {x for x in self.available_labels.split(";") if x}


def salt(path: str | Path | None = None) -> bytes:
    p = Path(path or os.environ.get("BILACL_SALT_FILE", "data/.bilacl_salt"))
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(secrets.token_hex(32))
        os.chmod(p, 0o600)
    return bytes.fromhex(p.read_text().strip())


def pseudonym(cohort: str, source_subject_id: str, key: bytes | None = None) -> str:
    key = key if key is not None else salt()
    return "S" + hmac.new(key, f"{cohort}:{source_subject_id}".encode(), hashlib.sha256).hexdigest()[:16]


def write_manifest(records: list[ManifestRecord], path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in records:
            w.writerow({k: getattr(r, k) for k in FIELDS})


def read_manifest(path) -> list[ManifestRecord]:
    with open(path, newline="") as f:
        return [ManifestRecord(**{k: row.get(k, UNKNOWN) or ("" if k in ("available_labels", "provenance", "notes") else UNKNOWN) for k in FIELDS})
                for row in csv.DictReader(f)]


def summarize(records: list[ManifestRecord]) -> dict:
    from collections import Counter

    subj = {r.subject_key for r in records}
    by_subj_side: dict = {}
    for r in records:
        by_subj_side.setdefault((r.dataset, r.subject_key, r.visit), set()).add(r.side)
    bilateral = sum(1 for s in by_subj_side.values() if {"left", "right"} <= s)
    return {
        "records": len(records), "subjects": len(subj),
        "datasets": dict(Counter(r.dataset for r in records)),
        "sides": dict(Counter(r.side for r in records)),
        "physical_verified": dict(Counter(r.physical_verified for r in records)),
        "label_granularity": dict(Counter(r.label_granularity for r in records)),
        "splits": dict(Counter(r.split for r in records)),
        "subject_visits_with_both_sides": bilateral,
        "note": "bilateral count requires both sides at the same visit with known laterality",
    }


def to_json(records):
    return json.dumps([dataclasses.asdict(r) for r in records], indent=1)
