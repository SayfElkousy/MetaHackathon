"""Concrete dataset adapters. See registry/datasets.yaml for what is verified about each dataset."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..manifest import ManifestRecord, pseudonym
from ..provenance import UNKNOWN
from .base import Adapter, InspectionReport, register


def _fmt_spacing(sp) -> str:
    return "x".join(f"{float(s):.4g}" for s in sp)


# ---------------------------------------------------------------- bilateral case (project format)


@register
class BilateralCaseAdapter(Adapter):
    """Project case format: <root>/<case>/case.json (+ injured/, contralateral/). Used for synthetic
    phantoms and for the future expert-annotated bilateral cohort."""

    name = "bilateral_case"
    cohort = "bilateral_case"

    def inspect(self) -> InspectionReport:
        rep = self._scan()
        if not rep.exists:
            return rep
        cases = sorted(self.root.glob("**/case.json"))
        rep.found["cases"] = len(cases)
        for c in cases:
            d = json.loads(c.read_text())
            for role in ("injured", "contralateral"):
                for k in ("image", "labels"):
                    if not (c.parent / d[role][k]).exists():
                        rep.issues.append(f"{c.parent.name}: missing {role}/{k}")
                if d[role].get("side") not in ("left", "right"):
                    rep.issues.append(f"{c.parent.name}: {role} laterality unknown")
        rep.found["synthetic_cases"] = sum(1 for c in cases if json.loads(c.read_text()).get("synthetic"))
        rep.schema_ok = bool(cases) and not rep.issues
        return rep

    def records(self) -> list[ManifestRecord]:
        from ..imaging.volume import load_nifti

        out = []
        for c in sorted(self.root.glob("**/case.json")):
            d = json.loads(c.read_text())
            syn = bool(d.get("synthetic"))
            for role in ("injured", "contralateral"):
                img = c.parent / d[role]["image"]
                v = load_nifti(img)
                ann = c.parent / str(d[role]["labels"]).replace(".nii.gz", ".annotation.json")
                labels = []
                review = UNKNOWN
                if ann.exists():
                    a = json.loads(ann.read_text())
                    from ..annotation.schema import Label

                    labels = [f"voxel:{Label(x).name.lower()}" for x in a.get("annotated_labels", [])]
                    review = a.get("review_status", UNKNOWN)
                out.append(ManifestRecord(
                    dataset="synthetic" if syn else self.name, subject_key=d["subject_id"] if syn else pseudonym(self.cohort, d["subject_id"]),
                    cohort="synthetic" if syn else self.cohort, exam_id=d[role]["exam_id"], series_id=role, side=d[role]["side"],
                    side_source="case.json", visit=d[role].get("visit", "V0"), sequence=d[role].get("sequence", "synthetic-3D" if syn else UNKNOWN),
                    spacing_mm=_fmt_spacing(v.spacing), orientation="".join(v.axcodes), physical_verified=str(v.physical.verified).lower(),
                    image_path=str(img), label_path=str(c.parent / d[role]["labels"]), available_labels=";".join(labels),
                    label_granularity="voxel" if labels else "none", annotation_review=review,
                    provenance=f"case:{d['case_id']}", notes="SYNTHETIC" if syn else ""))
        return out


# ---------------------------------------------------------------- generic local DICOM / NIfTI


@register
class LocalImagingAdapter(Adapter):
    """Any local folder of DICOM series and/or NIfTI files. Laterality comes from DICOM tags; text
    heuristics are marked as requiring confirmation."""

    name = "local"
    cohort = "local"

    def inspect(self) -> InspectionReport:
        from ..imaging.volume import discover_dicom_series, select_series

        rep = self._scan()
        if not rep.exists:
            return rep
        groups = discover_dicom_series(self.root)
        ranked = select_series(groups) if groups else []
        rep.found["dicom_series"] = len(ranked)
        rep.found["usable_series"] = sum(1 for r, _ in ranked if r.usable)
        rep.found["localizers"] = sum(1 for r, _ in ranked if r.is_localizer)
        rep.found["nifti"] = len(list(self.root.rglob("*.nii*")))
        for r, _ in ranked:
            if r.issues:
                rep.issues.append(f"series {r.series_uid[-12:]} '{r.description}': {r.issues}")
        rep.schema_ok = rep.found["usable_series"] + rep.found["nifti"] > 0
        return rep

    def records(self) -> list[ManifestRecord]:
        from ..imaging.volume import discover_dicom_series, load_nifti, select_series

        out = []
        for r, items in select_series(discover_dicom_series(self.root)):
            if not r.usable:
                continue
            ds0 = items[0][1]
            pid = str(getattr(ds0, "PatientID", UNKNOWN))
            sp = r.meta.get("pixel_spacing_mm", [UNKNOWN, UNKNOWN]) + [r.meta.get("slice_spacing_mm", UNKNOWN)]
            out.append(ManifestRecord(
                dataset=self.name, subject_key=pseudonym(self.cohort, pid), cohort=self.cohort, exam_id=pseudonym("exam", r.meta.get("study_uid", "")),
                series_id=pseudonym("series", r.series_uid), side=r.laterality, side_source=r.laterality_source, sequence=r.description or UNKNOWN,
                scanner_manufacturer=r.meta.get("manufacturer", UNKNOWN), scanner_model=r.meta.get("model", UNKNOWN),
                field_strength_T=str(r.meta.get("field_strength_T", UNKNOWN)), spacing_mm=_fmt_spacing(sp) if UNKNOWN not in sp else UNKNOWN,
                physical_verified="true", image_path=str(items[0][0].parent), label_granularity="none", provenance="local-dicom",
                notes="; ".join(r.warnings)))
        for p in sorted(self.root.rglob("*.nii*")):
            v = load_nifti(p)
            out.append(ManifestRecord(dataset=self.name, subject_key=UNKNOWN, cohort=self.cohort, exam_id=p.name, spacing_mm=_fmt_spacing(v.spacing),
                                      orientation="".join(v.axcodes), physical_verified=str(v.physical.verified).lower(), image_path=str(p),
                                      label_granularity="none", provenance="local-nifti", notes="subject/side unknown: supply a sidecar or case.json"))
        return out


# ---------------------------------------------------------------- OAI


@register
class OAIAdapter(LocalImagingAdapter):
    """OAI imaging obtained through NDA (user-provided). Contents are NOT assumed: the adapter
    audits subject/side/visit/sequence from DICOM headers and folder names."""

    name = "oai"
    cohort = "oai"

    def inspect(self) -> InspectionReport:
        rep = super().inspect()
        rep.notes.append("OAI access requires NDA approval; this tool never downloads OAI data.")
        rep.notes.append("Bilateral pairs must be derived from audited side + visit; not every participant has an eligible pair.")
        return rep


# ---------------------------------------------------------------- OAI-ZIB


@register
class OAIZIBAdapter(Adapter):
    """OAI-ZIB segmentations (CC BY 4.0).

    Verified from the archive (2026-10-24): OAI-ZIB/segmentation/segmentation_masks/<OAI ID>.segmentation_masks.mhd/.raw
    (507 masks, one per participant), doc/oai_mri_paths.txt (OAI image path per ID), doc/2foldCrossValidation-List{1,2}.txt.
    MR images are NOT included (obtain from OAI via NDA). Label meanings are NOT documented in the archive:
    the adapter requires configs/label_maps/oai_zib.yaml with confirmed: true. MetaImage headers carry an identity
    TransformMatrix and AnatomicalOrientation=RPI, so mask geometry is taken from the paired OAI DICOM, not the header."""

    name = "oai_zib"
    cohort = "oai"  # derived from OAI: same pseudonym namespace for leakage control

    def __init__(self, root, label_map: dict | None = None):
        super().__init__(root, label_map)
        if self.label_map is None:
            import yaml

            p = Path(__file__).resolve().parents[3] / "configs" / "label_maps" / "oai_zib.yaml"
            if p.exists():
                lm = yaml.safe_load(p.read_text())
                if lm.get("confirmed"):
                    self.label_map = {int(k): v for k, v in lm["values"].items()}
                else:
                    self.unconfirmed_map = lm

    def _masks(self, subset="segmentation"):
        return sorted(p for p in self.root.rglob("*.segmentation_masks.mhd") if p.parent.parent.name == subset)

    def _paths(self, subset="segmentation"):
        out = {}
        for p in self.root.rglob("oai_mri_paths.txt"):
            if p.parent.parent.name != subset:
                continue
            for line in p.read_text().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    out[k.strip()] = v.strip()
        return out

    def _folds(self):
        f = {}
        for p in self.root.rglob("2foldCrossValidation-List*.txt"):
            for line in p.read_text().split():
                f[line.strip()] = p.stem.split("-")[-1]
        return f

    def inspect(self) -> InspectionReport:
        rep = self._scan()
        if not rep.exists:
            rep.notes.append("public archive: bilacl acquire download oai_zib --confirm (shows size and disk check first)")
            return rep
        import SimpleITK as sitk

        masks = self._masks()
        ids = {p.name.split(".")[0] for p in masks}
        paths, folds = self._paths(), self._folds()
        cls = self._masks("classification")
        cls_ids = {p.name.split(".")[0] for p in cls}
        cls_paths = self._paths("classification")
        same_scan = sum(1 for k in cls_ids & ids if cls_paths.get(k) == self._paths().get(k))
        rep.found["classification_subset"] = {"masks": len(cls), "ids_also_in_segmentation_subset": len(cls_ids & ids),
                                              "of_which_same_image_path": same_scan,
                                              "note": "different label set (values 0-2) on the SAME scans: never split these across train/test"}
        rep.found.update(masks=len(masks), unique_subject_ids=len(ids), image_paths_listed=len(paths), cv_fold_assignments=len(folds),
                         raw_files_present=sum(1 for p in masks if p.with_suffix(".raw").exists()))
        vals, spacing = set(), set()
        for p in masks[:: max(1, len(masks) // 10)]:
            im = sitk.ReadImage(str(p))
            vals |= {int(x) for x in np.unique(sitk.GetArrayViewFromImage(im))}
            spacing.add(tuple(round(s, 3) for s in im.GetSpacing()))
        rep.found["label_values_seen"] = sorted(vals)
        rep.found["spacings_seen"] = sorted(spacing)
        if ids - set(paths):
            rep.issues.append(f"{len(ids - set(paths))} mask IDs without an OAI image path")
        rep.notes.append("MR images are not included: pair with OAI DICOM via doc/oai_mri_paths.txt (requires NDA access)")
        rep.notes.append("one knee per participant: no bilateral pairs within OAI-ZIB")
        if self.label_map is None:
            rep.blocking.append("label meanings not documented in the archive: confirm configs/label_maps/oai_zib.yaml (currently confirmed: false)")
        rep.blocking.append("paired OAI images not available locally (NDA access required) - segmentation training blocked")
        rep.schema_ok = bool(masks) and not rep.issues
        return rep

    def records(self) -> list[ManifestRecord]:
        if self.label_map is None:
            raise PermissionError("OAI-ZIB label map not confirmed (configs/label_maps/oai_zib.yaml confirmed: false); refusing to create labelled records")
        paths, folds = self._paths(), self._folds()
        out = []
        for p in self._masks():
            sid = p.name.split(".")[0]
            out.append(ManifestRecord(dataset=self.name, subject_key=pseudonym(self.cohort, sid), cohort=self.cohort, exam_id=f"oai_zib_{sid}",
                                      image_path=paths.get(sid, UNKNOWN), label_path=str(p),
                                      available_labels=";".join(f"voxel:{v}" for v in self.label_map.values()),
                                      label_granularity="voxel", annotation_review="published_manual", physical_verified="false",
                                      provenance="OAI-ZIB (Ambellan et al. 2019), CC BY 4.0",
                                      notes=f"image_path is an OAI archive path (not local); published 2-fold CV list {folds.get(sid, 'none')}"))
        return out


# ---------------------------------------------------------------- SKM-TEA


@register
class SKMTEAAdapter(Adapter):
    """SKM-TEA image (DICOM) track. Verified layout: HDF5 per scan with echo1/echo2/seg (6 tissue
    classes), annotations/<version>/{train,val,test}.json with voxel_spacing/orientation.

    Orientation in DATASET.md is documented as SI/AP/LR, with the third axis described as
    "RL/LR" - ambiguous handedness. Physical geometry is therefore taken from the DICOMs when
    present; otherwise physical_verified = false."""

    name = "skm_tea"
    cohort = "skm_tea"
    TISSUES = ["patellar_cartilage", "femoral_cartilage", "tibial_cartilage_medial", "tibial_cartilage_lateral", "meniscus_medial", "meniscus_lateral"]

    def inspect(self) -> InspectionReport:
        rep = self._scan()
        if not rep.exists:
            rep.notes.append("Stanford AIMI account required; download the image track only (raw k-space not needed)")
            return rep
        h5 = list(self.root.rglob("*.h5"))
        anns = list(self.root.rglob("annotations/*/*.json"))
        rep.found.update(h5_files=len(h5), annotation_files=[str(a.relative_to(self.root)) for a in anns],
                         dicom_dirs=len(list(self.root.rglob("dicoms/*"))))
        if h5:
            import h5py

            with h5py.File(h5[0], "r") as f:
                keys = list(f.keys())
                rep.found["first_h5_keys"] = keys
                if "seg" in f:
                    rep.found["seg_shape"] = list(f["seg"].shape)
                    if f["seg"].shape[-1] != 6:
                        rep.issues.append(f"seg last dim {f['seg'].shape[-1]} != 6 documented classes")
                for k in ("echo1", "echo2", "seg"):
                    if k not in keys:
                        rep.issues.append(f"documented key {k!r} missing in {h5[0].name}")
                if "kspace" in keys:
                    rep.notes.append("raw-data track file detected; not used by this project")
        if not anns:
            rep.issues.append("annotations/<version>/{train,val,test}.json not found")
        rep.notes.append("pathology annotations are 3D bounding boxes from reports: NOT voxel labels; no bone or ACL masks exist")
        rep.schema_ok = bool(h5) and bool(anns) and not rep.issues
        return rep

    def records(self) -> list[ManifestRecord]:
        out = []
        for split_file in sorted(self.root.rglob("annotations/*/*.json")):
            split = split_file.stem
            d = json.loads(split_file.read_text())
            for im in d.get("images", []):
                sid = str(im.get("subject_id", UNKNOWN))
                h5 = next(iter(self.root.rglob(f"{im.get('file_name', '__none__')}")), None)
                dcm = self.root / "dicoms" / str(im.get("scan_id", ""))
                out.append(ManifestRecord(
                    dataset=self.name, subject_key=pseudonym(self.cohort, sid), cohort=self.cohort, exam_id=str(im.get("scan_id", UNKNOWN)),
                    visit=str(im.get("timepoint", UNKNOWN)), sequence="qDESS", spacing_mm=_fmt_spacing(im["voxel_spacing"]) if "voxel_spacing" in im else UNKNOWN,
                    orientation="/".join(im.get("orientation", [])) or UNKNOWN,
                    physical_verified="true" if dcm.exists() else "false",
                    image_path=str(h5) if h5 else UNKNOWN, label_path=str(h5) if h5 else UNKNOWN,
                    available_labels=";".join(f"voxel:{t}" for t in self.TISSUES) + ";bbox:pathology",
                    label_granularity="voxel", annotation_review="published_manual", official_split=split,
                    provenance=f"SKM-TEA {split_file.parent.name}",
                    notes="" if dcm.exists() else "geometry from HDF5 metadata only: third-axis handedness ambiguous -> physical export disabled"))
        return out


# ---------------------------------------------------------------- MRNet


@register
class MRNetAdapter(Adapter):
    """MRNet: exam-level labels only. Distributed file format is not stated on the official pages,
    so discovery is format-agnostic and arrays are wrapped WITHOUT physical geometry."""

    name = "mrnet"
    cohort = "mrnet"

    def inspect(self) -> InspectionReport:
        rep = self._scan()
        if not rep.exists:
            rep.notes.append("Stanford AIMI/Redivis download (v1.0); user-provided")
            return rep
        npy = list(self.root.rglob("*.npy"))
        csvs = list(self.root.rglob("*.csv"))
        rep.found.update(npy=len(npy), csv=[c.name for c in csvs])
        if npy:
            a = np.load(npy[0], mmap_mode="r")
            rep.found["first_array_shape"] = list(a.shape)
        rep.notes.append("no spacing/orientation headers documented: physical geometry unavailable; design export always disabled")
        rep.notes.append("labels are exam-level (abnormal / ACL tear / meniscal tear): never usable as voxel labels")
        rep.schema_ok = bool(npy) and bool(csvs)
        return rep

    def records(self) -> list[ManifestRecord]:
        labels: dict = {}
        for c in self.root.rglob("*.csv"):
            name = c.stem.lower()
            task = next((t for t in ("acl", "meniscus", "abnormal") if t in name), None)
            split = "train" if "train" in name else "valid" if "valid" in name else UNKNOWN
            if task is None:
                continue
            with open(c) as f:
                for row in csv.reader(f):
                    if len(row) >= 2 and row[0].strip().isdigit():
                        labels.setdefault((split, row[0].strip().zfill(4)), {})[task] = int(row[1])
        out = []
        for p in sorted(self.root.rglob("*.npy")):
            parts = p.parts
            split = "train" if "train" in parts else "valid" if "valid" in parts else UNKNOWN
            plane = next((x for x in ("sagittal", "coronal", "axial") if x in parts), UNKNOWN)
            exam = p.stem
            lab = labels.get((split, exam), {})
            out.append(ManifestRecord(
                dataset=self.name, subject_key=UNKNOWN, cohort=self.cohort, exam_id=exam, plane=plane, physical_verified="false",
                image_path=str(p), available_labels=";".join(f"exam:{k}={v}" for k, v in sorted(lab.items())),
                label_granularity="exam" if lab else "none", official_split=split, provenance="MRNet v1.0",
                notes="patient IDs not in exam filenames; official split is patient-level per MRNet documentation"))
        return out
