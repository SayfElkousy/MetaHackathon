"""Annotation schema.

Label codes are fixed project-wide. Two values are reserved:

* 0   BACKGROUND -- the annotator *asserted* this voxel is none of the labelled structures
* 255 UNKNOWN    -- not assessed / unassessable / outside the annotated region

An annotation file also declares which structures were actually annotated
(`annotated_labels`). For a structure that was NOT annotated, a 0 voxel does not
mean "absent" -- training code converts such voxels to UNKNOWN for that task
(see `effective_target`). This is how absence of annotation is kept from
silently becoming background.

Tear gaps are never labelled as tissue. The gap between remnants is either
background (asserted empty / fluid) or UNKNOWN, at the annotator's discretion.
"""
from __future__ import annotations

import dataclasses
import json
from enum import IntEnum
from pathlib import Path

import numpy as np

from ..provenance import ReviewStatus, utcnow


class Label(IntEnum):
    BACKGROUND = 0
    FEMUR = 1
    TIBIA = 2
    PATELLA = 3
    FIBULA = 4
    FEMORAL_CARTILAGE = 5
    TIBIAL_CARTILAGE = 6
    PATELLAR_CARTILAGE = 7
    MENISCUS_MEDIAL = 8
    MENISCUS_LATERAL = 9
    PCL = 10
    ACL_INTACT = 20
    ACL_REMNANT_FEMORAL = 21  # visible femoral-side remnant
    ACL_REMNANT_TIBIAL = 22  # visible tibial-side remnant
    ACL_FRAGMENT_OTHER = 23  # additional visible fragment not attributable to either side
    ACL_UNCERTAIN = 24  # ACL-region tissue whose identity/continuity is uncertain
    UNKNOWN = 255


LABEL_COLORS = {
    Label.FEMUR: (230, 220, 170), Label.TIBIA: (200, 190, 230), Label.PATELLA: (240, 200, 160),
    Label.FIBULA: (180, 200, 160), Label.FEMORAL_CARTILAGE: (110, 180, 255), Label.TIBIAL_CARTILAGE: (90, 140, 230),
    Label.PATELLAR_CARTILAGE: (60, 110, 200), Label.MENISCUS_MEDIAL: (120, 220, 120), Label.MENISCUS_LATERAL: (80, 180, 80),
    Label.PCL: (200, 120, 200), Label.ACL_INTACT: (240, 80, 60), Label.ACL_REMNANT_FEMORAL: (255, 160, 0),
    Label.ACL_REMNANT_TIBIAL: (255, 220, 0), Label.ACL_FRAGMENT_OTHER: (255, 120, 180), Label.ACL_UNCERTAIN: (150, 150, 150),
    Label.UNKNOWN: (60, 60, 60),
}

ACL_LABELS = (Label.ACL_INTACT, Label.ACL_REMNANT_FEMORAL, Label.ACL_REMNANT_TIBIAL, Label.ACL_FRAGMENT_OTHER, Label.ACL_UNCERTAIN)
BONE_LABELS = (Label.FEMUR, Label.TIBIA, Label.PATELLA, Label.FIBULA)

# Landmarks are stored separately (Slicer markups JSON), never burned into masks.
LANDMARKS = {
    "acl_femoral_footprint_center": "Centre of femoral ACL attachment (if identifiable)",
    "acl_tibial_footprint_center": "Centre of tibial ACL attachment (if identifiable)",
    "acl_femoral_footprint_outline": "Closed curve on femoral attachment (optional)",
    "acl_tibial_footprint_outline": "Closed curve on tibial attachment (optional)",
    "femoral_remnant_tip": "Distal-most visible point of femoral-side remnant",
    "tibial_remnant_tip": "Proximal-most visible point of tibial-side remnant",
    "tear_region_center": "Estimated centre of tear region",
    # independent registration-QC landmarks (never used to drive registration)
    "femur_lateral_epicondyle": "", "femur_medial_epicondyle": "", "femur_notch_roof_apex": "",
    "tibia_medial_eminence_tip": "", "tibia_lateral_eminence_tip": "", "tibia_tuberosity": "",
    "pcl_tibial_attachment_center": "",
}

TEAR_LOCATIONS = ("intact", "proximal_avulsion", "proximal", "midsubstance", "distal", "distal_avulsion", "partial", "unknown")
# Tear location categories follow van der List et al. 2017 (Orthop J Sports Med 5:2325967117709966)
# remnant-length classification; "partial" and "unknown" are added.
CHRONICITY = ("acute", "subacute", "chronic", "unknown")
REMNANT_QUALITY = ("good", "fair", "poor", "absent", "unassessable", "unknown")


@dataclasses.dataclass
class AnnotationRecord:
    """Metadata for one label volume. Stored as <mask>.annotation.json next to the mask."""

    annotation_id: str
    exam_id: str
    image_ref: str  # path/ID of the image the mask is defined on
    annotator: str  # pseudonymous reader ID
    annotator_role: str = "unknown"  # e.g. "MSK radiologist", "trained annotator", "model"
    version: int = 1
    parent_annotation_id: str | None = None  # for corrections
    annotated_labels: list[int] = dataclasses.field(default_factory=list)
    source: str = "manual"  # manual | model_prediction | model_corrected | synthetic
    model_id: str | None = None  # required if source involves a model
    review_status: ReviewStatus = ReviewStatus.UNREVIEWED
    reviewer: str | None = None
    tear_location: str = "unknown"
    chronicity: str = "unknown"
    remnant_quality: dict = dataclasses.field(default_factory=lambda: {"femoral": "unknown", "tibial": "unknown"})
    contralateral_pathology: str = "unknown"  # for contralateral knees: expert-recorded findings
    contralateral_confirmed_normal: bool | None = None  # explicit expert confirmation; None = not assessed
    notes: str = ""
    created: str = dataclasses.field(default_factory=utcnow)

    def validate(self) -> list[str]:
        errs = []
        if self.tear_location not in TEAR_LOCATIONS:
            errs.append(f"tear_location {self.tear_location!r} not in {TEAR_LOCATIONS}")
        if self.chronicity not in CHRONICITY:
            errs.append(f"chronicity {self.chronicity!r} invalid")
        for side, q in self.remnant_quality.items():
            if q not in REMNANT_QUALITY:
                errs.append(f"remnant_quality[{side}]={q!r} invalid")
        valid = {int(l) for l in Label}
        bad = [l for l in self.annotated_labels if l not in valid or l in (0, 255)]
        if bad:
            errs.append(f"annotated_labels contains invalid codes {bad}")
        if self.source in ("model_prediction", "model_corrected") and not self.model_id:
            errs.append("model-derived annotation must record model_id")
        return errs

    @property
    def usable_as_ground_truth(self) -> bool:
        """Unreviewed model predictions are never ground truth."""
        if self.source == "model_prediction":
            return False
        if self.source == "model_corrected":
            return self.review_status == ReviewStatus.APPROVED
        return self.review_status in (ReviewStatus.APPROVED, ReviewStatus.CORRECTED, ReviewStatus.UNREVIEWED) and self.source in ("manual", "synthetic")

    def to_dict(self):
        d = dataclasses.asdict(self)
        d["review_status"] = self.review_status.value
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["review_status"] = ReviewStatus(d.get("review_status", "unreviewed"))
        return cls(**d)

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))


def effective_target(mask: np.ndarray, record: AnnotationRecord, task_labels: dict[int, int]) -> np.ndarray:
    """Map a project label volume to a task target with explicit ignore voxels.

    task_labels: {project_label_code: task_class_index}; task class 0 = background.
    Returns int16 array with -1 for ignored voxels.

    A voxel is ignored when:
      * it is UNKNOWN (255) or ACL_UNCERTAIN in the source, or
      * it is 0 but at least one of the task's labels was not annotated in this record
        (so 0 cannot be trusted as "absent" for that task).
    """
    annotated = set(record.annotated_labels)
    needed = set(task_labels)
    out = np.zeros(mask.shape, np.int16)
    for src, dst in task_labels.items():
        out[mask == src] = dst
    ignore = (mask == Label.UNKNOWN) | (mask == Label.ACL_UNCERTAIN)
    missing = needed - annotated
    if missing:
        # Background is unverifiable for this task: only voxels positively labelled with an
        # annotated task class remain supervised.
        supervised = np.isin(mask, list(needed & annotated))
        ignore |= ~supervised
    out[ignore] = -1
    return out


# --------------------------------------------------------------------- 3D Slicer I/O


def write_slicer_color_table(path) -> None:
    """3D Slicer color table (.ctbl / .txt): `index name R G B A` per line."""
    lines = ["# bilacl label table for 3D Slicer"]
    for lab in Label:
        r, g, b = LABEL_COLORS.get(lab, (0, 0, 0))
        lines.append(f"{int(lab)} {lab.name} {r} {g} {b} {0 if lab == Label.BACKGROUND else 255}")
    Path(path).write_text("\n".join(lines) + "\n")


def read_slicer_seg_nrrd(path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Import a Slicer .seg.nrrd (labelmap representation) into project label codes.

    Segment names must match Label names (case-insensitive) or a name listed in
    LABEL_ALIASES; unmatched segments raise, so nothing is silently dropped.
    Returns (label array in RAS voxel order, RAS affine, info).
    """
    from ..imaging.volume import load_sitk

    vol = load_sitk(path)
    md = vol.meta["sitk_metadata"]
    data = vol.data
    n = 0
    segs = []
    while f"Segment{n}_Name" in md:
        segs.append({
            "name": md[f"Segment{n}_Name"],
            "label_value": int(md.get(f"Segment{n}_LabelValue", n + 1)),
            "layer": int(md.get(f"Segment{n}_Layer", 0)),
        })
        n += 1
    out = np.zeros(data.shape[:3], np.uint8)
    for s in segs:
        code = _label_from_name(s["name"])
        layer = data[..., s["layer"]] if data.ndim == 4 else data
        out[layer == s["label_value"]] = code
    return out, vol.affine, {"segments": segs}


LABEL_ALIASES = {
    "acl": Label.ACL_INTACT, "anterior cruciate ligament": Label.ACL_INTACT,
    "femoral remnant": Label.ACL_REMNANT_FEMORAL, "tibial remnant": Label.ACL_REMNANT_TIBIAL,
    "unknown": Label.UNKNOWN, "unassessable": Label.UNKNOWN, "ignore": Label.UNKNOWN,
}


def _label_from_name(name: str) -> int:
    k = name.strip().lower()
    for lab in Label:
        if lab.name.lower() == k.replace(" ", "_"):
            return int(lab)
    if k in LABEL_ALIASES:
        return int(LABEL_ALIASES[k])
    raise KeyError(f"segment name {name!r} does not map to a project label; rename it or extend LABEL_ALIASES")


def write_slicer_seg_nrrd(label: np.ndarray, affine: np.ndarray, path) -> None:
    """Export a project label volume as a Slicer-readable .seg.nrrd (single layer)."""
    import SimpleITK as sitk

    from ..imaging.volume import PhysicalStatus, Volume, to_sitk

    img = to_sitk(Volume(label.astype(np.uint8), affine, PhysicalStatus(True, "export")))
    present = [int(v) for v in np.unique(label) if v != 0]
    for i, code in enumerate(present):
        lab = Label(code)
        r, g, b = LABEL_COLORS.get(lab, (128, 128, 128))
        img.SetMetaData(f"Segment{i}_ID", f"Segment_{code}")
        img.SetMetaData(f"Segment{i}_Name", lab.name)
        img.SetMetaData(f"Segment{i}_LabelValue", str(code))
        img.SetMetaData(f"Segment{i}_Layer", "0")
        img.SetMetaData(f"Segment{i}_Color", f"{r/255:.3f} {g/255:.3f} {b/255:.3f}")
    sitk.WriteImage(img, str(path), useCompression=True)


def read_slicer_markups(path) -> dict[str, list[float]]:
    """Read Slicer .mrk.json control points -> {label: [x,y,z] RAS mm}.

    Slicer >= 4.11 stores `coordinateSystem` ("LPS" or "RAS") per markup; LPS is converted."""
    d = json.loads(Path(path).read_text())
    out = {}
    for mk in d.get("markups", []):
        cs = mk.get("coordinateSystem", "LPS").upper()
        for cp in mk.get("controlPoints", []):
            p = np.array(cp["position"], float)
            if cs == "LPS":
                p = p * np.array([-1, -1, 1])
            out[cp.get("label") or cp.get("id")] = p.tolist()
    return out


def write_slicer_markups(points: dict[str, list[float]], path, description: str = "") -> None:
    """Write RAS mm points as a Slicer fiducial markups file (stored in LPS, as Slicer does)."""
    cps = []
    for i, (name, p) in enumerate(points.items()):
        q = (np.array(p, float) * np.array([-1, -1, 1])).tolist()
        cps.append({"id": str(i + 1), "label": name, "description": description, "position": q, "positionStatus": "defined"})
    doc = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [{"type": "Fiducial", "coordinateSystem": "LPS", "coordinateUnits": "mm", "controlPoints": cps}],
    }
    Path(path).write_text(json.dumps(doc, indent=2))


# --------------------------------------------------------------------- reader agreement


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / s) if s else float("nan")


def inter_reader_agreement(masks: dict[str, np.ndarray], labels=None) -> dict:
    """Pairwise Dice between readers per label, ignoring voxels any reader marked UNKNOWN,
    plus a STAPLE consensus estimate per label (SimpleITK)."""
    import itertools

    import SimpleITK as sitk

    readers = list(masks)
    stack = np.stack([masks[r] for r in readers])
    known = ~(stack == Label.UNKNOWN).any(0)
    labels = labels or sorted({int(v) for v in np.unique(stack) if v not in (0, 255)})
    res = {"readers": readers, "per_label": {}}
    for lab in labels:
        pair = {}
        for r1, r2 in itertools.combinations(readers, 2):
            pair[f"{r1}|{r2}"] = dice((masks[r1] == lab) & known, (masks[r2] == lab) & known)
        entry = {"pairwise_dice": pair, "mean_pairwise_dice": float(np.nanmean(list(pair.values()))) if pair else None}
        if len(readers) >= 2:
            imgs = [sitk.GetImageFromArray(((masks[r] == lab) & known).astype(np.uint8)) for r in readers]
            st = sitk.STAPLEImageFilter()
            st.SetForegroundValue(1)
            prob = sitk.GetArrayFromImage(st.Execute(imgs))
            entry["staple_sensitivity"] = list(st.GetSensitivity())
            entry["staple_specificity"] = list(st.GetSpecificity())
            entry["staple_consensus_voxels"] = int((prob > 0.5).sum())
        res["per_label"][Label(lab).name] = entry
    return res
