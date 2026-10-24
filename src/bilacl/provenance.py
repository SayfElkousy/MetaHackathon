"""Evidence levels, review status and provenance records.

Every artefact the platform produces carries an EvidenceLevel so that observed
anatomy, model estimates, hypothesised architecture and validated findings are
never mixed up downstream (data model, CLI output, and UI badges all use this).
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import platform
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any


class EvidenceLevel(str, Enum):
    OBSERVED = "observed"  # directly annotated / resolved in the input imaging
    ESTIMATED = "estimated"  # model-derived segmentation or reconstructed geometry
    HYPOTHESIZED = "hypothesized"  # generated internal pathways from priors
    VALIDATED = "validated"  # supported by actual evaluation data (cite the evaluation)
    SYNTHETIC = "synthetic"  # produced from a synthetic phantom; no clinical meaning


class ModelStatus(str, Enum):
    NONE = "none"
    UNTRAINED = "untrained"  # random init / smoke test only
    TRAINED = "trained"  # trained, see run record for data and metrics
    SMOKE_TEST = "smoke_test"  # trained for a few steps on synthetic data to test code paths


class ReviewStatus(str, Enum):
    UNREVIEWED = "unreviewed"
    IN_REVIEW = "in_review"
    CORRECTED = "corrected"  # expert edited; needs sign-off
    APPROVED = "approved"  # expert approved for research use
    REJECTED = "rejected"


UNKNOWN = "unknown"  # canonical value for unknown metadata. Never replace with a guess.


def is_unknown(v: Any) -> bool:
    return v is None or v == UNKNOWN or (isinstance(v, float) and v != v)


def utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def code_version() -> dict:
    from . import __version__

    info = {"package_version": __version__}
    try:
        root = Path(__file__).resolve().parents[2]
        rev = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        info["git_commit"] = rev.stdout.strip() if rev.returncode == 0 else "no-vcs"
        if rev.returncode == 0:
            dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True, timeout=5)
            info["git_dirty"] = bool(dirty.stdout.strip())
    except Exception:
        info["git_commit"] = "no-vcs"
    return info


def environment() -> dict:
    env = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "torch", "monai", "SimpleITK", "nibabel", "trimesh"):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", None) or getattr(m, "Version_VersionString", lambda: "?")()
        except Exception:
            env[mod] = "not-installed"
    return env


@dataclasses.dataclass
class Provenance:
    """Minimal provenance stamp attached to every produced artefact."""

    evidence_level: EvidenceLevel
    producer: str  # module / command that produced the artefact
    inputs: list[str] = dataclasses.field(default_factory=list)  # paths or IDs
    input_hashes: dict[str, str] = dataclasses.field(default_factory=dict)
    model_status: ModelStatus = ModelStatus.NONE
    model_id: str | None = None
    review_status: ReviewStatus = ReviewStatus.UNREVIEWED
    reviewer: str | None = None
    parameters: dict = dataclasses.field(default_factory=dict)
    notes: list[str] = dataclasses.field(default_factory=list)
    created: str = dataclasses.field(default_factory=utcnow)
    code: dict = dataclasses.field(default_factory=code_version)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        for k in ("evidence_level", "model_status", "review_status"):
            d[k] = getattr(self, k).value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Provenance":
        d = dict(d)
        d["evidence_level"] = EvidenceLevel(d["evidence_level"])
        d["model_status"] = ModelStatus(d.get("model_status", "none"))
        d["review_status"] = ReviewStatus(d.get("review_status", "unreviewed"))
        return cls(**d)


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=_json_default))


def _json_default(o):
    import numpy as np

    if isinstance(o, Enum):
        return o.value
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")
