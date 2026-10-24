"""Task D: auxiliary exam-level ACL-tear classification (MRNet-style baseline).

Input: one 2D-series array per exam (slices, H, W), e.g. MRNet sagittal series.
Model: shared 2D CNN per slice -> max-pool over slices -> logit.
Labels: exam-level only. The output is an exam-level probability; it is never used
as, or converted into, a voxel label or a remnant annotation.
Physical geometry is not needed and not assumed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..evaluation.metrics import classification_report
from ..provenance import ModelStatus, environment, write_json
from .seg import pick_device, seed_everything


class SliceCNN(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        c = width
        self.f = nn.Sequential(
            nn.Conv2d(1, c, 5, 2, 2), nn.BatchNorm2d(c), nn.ReLU(),
            nn.Conv2d(c, 2 * c, 3, 2, 1), nn.BatchNorm2d(2 * c), nn.ReLU(),
            nn.Conv2d(2 * c, 4 * c, 3, 2, 1), nn.BatchNorm2d(4 * c), nn.ReLU(),
            nn.Conv2d(4 * c, 4 * c, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(4 * c, 1)

    def forward(self, x):  # x: (S, 1, H, W) for one exam
        z = self.f(x).flatten(1)
        return self.head(z.max(0).values)


def _prep(a: np.ndarray, size=128) -> torch.Tensor:
    a = a.astype(np.float32)
    a = (a - a.mean()) / (a.std() + 1e-6)
    t = torch.from_numpy(a)[:, None]
    return torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)


def train_cls(train_items, val_items, out_dir, epochs=10, lr=1e-4, seed=0, smoke_test=False, device="auto", log=print) -> dict:
    """items: list of (exam_id, array_path_or_array, label 0/1). Threshold for reports fixed at 0.5
    and documented; choose an operating point on validation data only."""
    seed_everything(seed)
    dev = pick_device(device)
    model = SliceCNN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    load = lambda a: np.load(a) if isinstance(a, (str, Path)) else a  # noqa: E731
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hist, best = [], -1
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train()
        ls = []
        for i in rng.permutation(len(train_items)):
            _, a, y = train_items[i]
            opt.zero_grad()
            lg = model(_prep(load(a)).to(dev))
            loss = lossf(lg, torch.tensor([float(y)], device=dev))
            loss.backward()
            opt.step()
            ls.append(float(loss))
        model.eval()
        with torch.no_grad():
            s = [float(torch.sigmoid(model(_prep(load(a)).to(dev)))) for _, a, _ in val_items]
        rep = classification_report([y for *_, y in val_items], s)
        rec = {"epoch": ep, "train_loss": float(np.mean(ls)), "val": {k: rep.get(k) for k in ("auroc", "auprc", "sensitivity", "specificity")}}
        hist.append(rec)
        log(json.dumps(rec))
        score = rep.get("auroc", -1)
        if score is not None and score > best:
            best = score
            torch.save({"model": model.state_dict(), "epoch": ep}, out / "best.pt")
    status = ModelStatus.SMOKE_TEST if smoke_test else ModelStatus.TRAINED
    run = {"task": "D_tear_cls", "model_status": status.value, "history": hist, "best_val_auroc": best, "environment": environment(),
           "note": "exam-level classifier; outputs are not segmentations" + ("; SMOKE TEST ONLY" if smoke_test else "")}
    write_json(out / "run.json", run)
    return run
