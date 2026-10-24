"""Dataset adapter interface.

An adapter never guesses dataset contents. `inspect` reports exactly what is on disk,
what matches the documented schema, and what is missing. Label-value mappings that
the official documentation does not state must be supplied in a user-confirmed
label-map file; otherwise label loading is refused.
"""
from __future__ import annotations

import dataclasses
from collections import Counter
from pathlib import Path

from ..manifest import ManifestRecord


@dataclasses.dataclass
class InspectionReport:
    dataset: str
    root: str
    exists: bool
    n_files: int = 0
    extensions: dict = dataclasses.field(default_factory=dict)
    found: dict = dataclasses.field(default_factory=dict)
    schema_ok: bool = False
    issues: list = dataclasses.field(default_factory=list)
    blocking: list = dataclasses.field(default_factory=list)
    notes: list = dataclasses.field(default_factory=list)


class Adapter:
    name = "base"
    cohort = "base"

    def __init__(self, root, label_map: dict | None = None):
        self.root = Path(root)
        self.label_map = label_map

    def _scan(self) -> InspectionReport:
        rep = InspectionReport(self.name, str(self.root), self.root.exists())
        if not rep.exists:
            rep.blocking.append(f"{self.root} does not exist: dataset not installed locally")
            return rep
        files = [p for p in self.root.rglob("*") if p.is_file()]
        rep.n_files = len(files)
        rep.extensions = dict(Counter("".join(p.suffixes[-2:]) or "<none>" for p in files).most_common(20))
        return rep

    def inspect(self) -> InspectionReport:
        return self._scan()

    def records(self) -> list[ManifestRecord]:
        raise NotImplementedError


ADAPTERS: dict[str, type[Adapter]] = {}


def register(cls):
    ADAPTERS[cls.name] = cls
    return cls


def get_adapter(name: str, root, **kw) -> Adapter:
    from . import datasets  # noqa: F401  (registers adapters)

    if name not in ADAPTERS:
        raise KeyError(f"no adapter {name!r}; available: {sorted(ADAPTERS)}")
    return ADAPTERS[name](root, **kw)
