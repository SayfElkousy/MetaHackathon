"""3D U-Net segmentation baseline (MONAI) with partial-label / ignore handling.

* targets: int64 with -1 = ignore (UNKNOWN, uncertain, or unannotated-for-this-task voxels)
* loss: masked cross-entropy + masked soft Dice (ignored voxels contribute nothing)
* patch sampling: foreground-biased random crops
* reproducibility: seeds, config + manifest hashes, environment and code version in run.json
* checkpoint/resume: last.pt every epoch, best.pt by validation Dice
* AMP: CUDA float16 autocast when available; MPS/CPU run in float32 (documented)
"""
from __future__ import annotations

import dataclasses
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..annotation.schema import AnnotationRecord, effective_target
from ..imaging.preprocess import intensity_normalize, resample_spacing
from ..imaging.volume import load_nifti, save_nifti
from ..manifest import ManifestRecord
from ..provenance import EvidenceLevel, ModelStatus, Provenance, ReviewStatus, environment, sha256_json, write_json
from .tasks import TASKS


@dataclasses.dataclass
class TrainConfig:
    task: str = "A_bone"
    out_dir: str = "runs/train"
    seed: int = 0
    spacing_mm: tuple = (0.7, 0.7, 1.0)  # training grid; source spacing recorded per case
    patch: tuple = (64, 64, 48)
    batch_size: int = 2
    epochs: int = 50
    iters_per_epoch: int = 50
    lr: float = 3e-4
    fg_prob: float = 0.67
    channels: tuple = (16, 32, 64, 128)
    strides: tuple = (2, 2, 2)
    num_res_units: int = 2
    dice_weight: float = 1.0
    amp: bool = True
    device: str = "auto"
    init_checkpoint: str | None = None  # transfer-learning initialisation (recorded)
    init_note: str = ""
    val_every: int = 1
    sw_overlap: float = 0.25
    allow_synthetic: bool = False
    smoke_test: bool = False
    max_cases: int | None = None

    @classmethod
    def from_yaml(cls, path):
        import yaml

        d = yaml.safe_load(Path(path).read_text())
        for k in ("spacing_mm", "patch", "channels", "strides"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(pref="auto"):
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(cfg: TrainConfig, n_classes: int):
    from monai.networks.nets import UNet

    return UNet(spatial_dims=3, in_channels=1, out_channels=n_classes, channels=cfg.channels, strides=cfg.strides, num_res_units=cfg.num_res_units)


def masked_loss(logits, target, n_classes, dice_weight=1.0, eps=1e-5):
    valid = target >= 0
    ce = F.cross_entropy(logits, target.clamp(min=0), reduction="none")
    ce = (ce * valid).sum() / valid.sum().clamp(min=1)
    prob = logits.softmax(1)
    oh = F.one_hot(target.clamp(min=0), n_classes).permute(0, 4, 1, 2, 3).float()
    m = valid.unsqueeze(1).float()
    inter = (prob * oh * m).sum((0, 2, 3, 4))
    den = ((prob + oh) * m).sum((0, 2, 3, 4))
    present = oh.mul(m).sum((0, 2, 3, 4)) > 0
    dice = (2 * inter + eps) / (den + eps)
    dloss = 1 - dice[1:][present[1:]].mean() if present[1:].any() else torch.zeros((), device=logits.device)
    return ce + dice_weight * dloss


class CaseCache:
    """Loads, resamples (image: linear, labels: nearest) and normalises cases once."""

    def __init__(self, records: list[ManifestRecord], cfg: TrainConfig):
        self.items = []
        task = TASKS[cfg.task]
        for r in records[: cfg.max_cases or None]:
            img = load_nifti(r.image_path)
            lab = load_nifti(r.label_path)
            ann = AnnotationRecord.load(str(r.label_path).replace(".nii.gz", ".annotation.json"))
            img_r = intensity_normalize(resample_spacing(img, cfg.spacing_mm, "image", order=1))
            lab_r = resample_spacing(lab, cfg.spacing_mm, "label")
            tgt = effective_target(lab_r.data.astype(np.int32), ann, task.classes)
            self.items.append({"exam_id": r.exam_id, "image": img_r.data.astype(np.float32), "target": tgt.astype(np.int64),
                               "source_spacing": img.spacing.tolist(), "label_source": ann.annotation_id})

    def sample(self, rng, patch, fg_prob):
        it = self.items[rng.integers(len(self.items))]
        img, tgt = it["image"], it["target"]
        pad = [(0, max(0, p - s)) for p, s in zip(patch, img.shape)]
        if any(p[1] for p in pad):
            img = np.pad(img, pad)
            tgt = np.pad(tgt, pad, constant_values=-1)
        if rng.random() < fg_prob and (tgt > 0).any():
            fg = np.argwhere(tgt > 0)
            c = fg[rng.integers(len(fg))]
        else:
            c = np.array([rng.integers(s) for s in img.shape])
        lo = np.clip(c - np.array(patch) // 2, 0, np.array(img.shape) - patch)
        sl = tuple(slice(a, a + p) for a, p in zip(lo, patch))
        x = img[sl].copy()
        x = x * rng.uniform(0.9, 1.1) + rng.uniform(-0.1, 0.1)  # intensity augmentation only (no L/R flips: laterality matters)
        return x[None], tgt[sl]


def _dice_per_class(pred, tgt, n_classes):
    out = {}
    v = tgt >= 0
    for c in range(1, n_classes):
        p, g = (pred == c) & v, (tgt == c) & v
        s = p.sum() + g.sum()
        out[c] = float(2 * (p & g).sum() / s) if s else float("nan")
    return out


def train(cfg: TrainConfig, train_recs: list[ManifestRecord], val_recs: list[ManifestRecord], log=print) -> dict:
    from monai.inferers import sliding_window_inference

    seed_everything(cfg.seed)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = pick_device(cfg.device)
    task = TASKS[cfg.task]
    n_classes = 1 + len(set(task.classes.values()))
    tr, va = CaseCache(train_recs, cfg), CaseCache(val_recs, cfg)
    if not tr.items:
        raise ValueError("no eligible training cases")
    train_subj = {r.subject_key for r in train_recs}
    val_subj = {r.subject_key for r in val_recs}
    if train_subj & val_subj:
        raise ValueError(f"subject leakage between train and val: {sorted(train_subj & val_subj)[:5]}")
    model = build_model(cfg, n_classes).to(dev)
    init_info = None
    if cfg.init_checkpoint:
        sd = torch.load(cfg.init_checkpoint, map_location="cpu")["model"]
        own = model.state_dict()
        loadable = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(loadable, strict=False)
        init_info = {"checkpoint": cfg.init_checkpoint, "loaded_tensors": len(loadable), "total_tensors": len(own), "note": cfg.init_note}
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    use_amp = cfg.amp and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start, best = 0, -1.0
    prev_run: dict = {}
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start, best = ck["epoch"] + 1, ck["best"]
        log(f"resumed from epoch {start}")
        prev_run = json.loads((out / "run.json").read_text()) if (out / "run.json").exists() else {}
    status = ModelStatus.SMOKE_TEST if cfg.smoke_test or cfg.allow_synthetic else ModelStatus.TRAINED
    run = {"config": dataclasses.asdict(cfg), "config_hash": sha256_json(dataclasses.asdict(cfg)), "device": str(dev), "amp": use_amp,
           "n_classes": n_classes, "task": dataclasses.asdict(task),
           "train_exams": [i["exam_id"] for i in tr.items], "val_exams": [i["exam_id"] for i in va.items],
           "label_sources": {i["exam_id"]: i["label_source"] for i in tr.items + va.items},
           "manifest_hash": sha256_json([dataclasses.asdict(r) for r in train_recs + val_recs]),
           "init": init_info, "environment": environment(), "model_status": status.value,
           "history": prev_run.get("history", []), "resumed_at_epoch": start if start else None}
    if prev_run.get("config_hash") not in (None, run["config_hash"]):
        raise ValueError("cannot resume: configuration changed since the checkpoint was written")
    if best >= 0:
        run["best_val_mean_dice"] = best
    rng = np.random.default_rng(cfg.seed)
    for ep in range(start, cfg.epochs):
        model.train()
        t0, losses = time.time(), []
        for _ in range(cfg.iters_per_epoch):
            xb, yb = zip(*[tr.sample(rng, cfg.patch, cfg.fg_prob) for _ in range(cfg.batch_size)])
            x = torch.from_numpy(np.stack(xb)).to(dev)
            y = torch.from_numpy(np.stack(yb)).to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = masked_loss(model(x), y, n_classes, cfg.dice_weight)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        rec = {"epoch": ep, "train_loss": float(np.mean(losses)), "sec": round(time.time() - t0, 1)}
        if va.items and (ep + 1) % cfg.val_every == 0:
            model.eval()
            dices = []
            with torch.no_grad():
                for it in va.items:
                    x = torch.from_numpy(it["image"][None, None]).to(dev)
                    lg = sliding_window_inference(x, cfg.patch, 2, model, overlap=cfg.sw_overlap)
                    pred = lg.argmax(1)[0].cpu().numpy()
                    dices.append(_dice_per_class(pred, it["target"], n_classes))
            mean = {c: float(np.nanmean([d[c] for d in dices])) for c in dices[0]}
            rec["val_dice"] = mean
            score = float(np.nanmean(list(mean.values())))
            rec["val_mean_dice"] = score
            if score > best:
                best = score
                torch.save({"model": model.state_dict(), "config": dataclasses.asdict(cfg), "n_classes": n_classes, "epoch": ep,
                            "val_mean_dice": score, "model_status": status.value, "task": cfg.task}, out / "best.pt")
        run["history"].append(rec)
        log(json.dumps(rec))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep, "best": best, "config": dataclasses.asdict(cfg),
                    "n_classes": n_classes, "model_status": status.value, "task": cfg.task}, out / "last.pt")
        run["best_val_mean_dice"] = best
        write_json(out / "run.json", run)
    run["note"] = ("SMOKE TEST on synthetic phantoms: verifies code paths only; NOT a trained anatomical model"
                   if status == ModelStatus.SMOKE_TEST else "model selected on validation Dice; test set untouched")
    write_json(out / "run.json", run)
    return run


def infer(checkpoint, image_path, out_path, device="auto", sw_overlap=0.5) -> dict:
    """Predict a label volume on the source grid. Output is an UNREVIEWED model prediction:
    it can be imported for expert correction but is never ground truth by itself."""
    from monai.inferers import sliding_window_inference

    from ..imaging.preprocess import resample_like

    ck = torch.load(checkpoint, map_location="cpu")
    cfg = TrainConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ck["config"].items()})
    dev = pick_device(device)
    model = build_model(cfg, ck["n_classes"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    img = load_nifti(image_path)
    x_r = intensity_normalize(resample_spacing(img, cfg.spacing_mm, "image", order=1))
    with torch.no_grad():
        lg = sliding_window_inference(torch.from_numpy(x_r.data[None, None].astype(np.float32)).to(dev), cfg.patch, 2, model, overlap=sw_overlap)
    pred = lg.argmax(1)[0].cpu().numpy().astype(np.uint8)
    inv = {v: k for k, v in TASKS[cfg.task].classes.items()}
    proj = np.zeros_like(pred)
    for c, code in inv.items():
        proj[pred == c] = code
    back = resample_like(x_r.with_data(proj), img, kind="label")
    save_nifti(back, out_path, np.uint8)
    status = ck.get("model_status", "unknown")
    rec = AnnotationRecord(annotation_id=Path(out_path).name, exam_id=Path(image_path).parent.name, image_ref=str(image_path),
                           annotator="model", annotator_role="model", source="model_prediction", model_id=str(checkpoint),
                           annotated_labels=sorted(inv.values()), review_status=ReviewStatus.UNREVIEWED,
                           notes=f"model_status={status}; task={cfg.task}; unreviewed prediction - not ground truth")
    rec.save(str(out_path).replace(".nii.gz", ".annotation.json"))
    prov = Provenance(EvidenceLevel.ESTIMATED, "bilacl.training.seg.infer", inputs=[str(image_path)], model_status=ModelStatus(status) if status in ModelStatus._value2member_map_ else ModelStatus.UNTRAINED,
                      model_id=str(checkpoint))
    write_json(str(out_path).replace(".nii.gz", ".provenance.json"), prov.to_dict())
    return {"prediction": str(out_path), "model_status": status, "task": cfg.task}
