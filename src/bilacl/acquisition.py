"""Dataset acquisition: dry-run reports, disk checks, resumable checksummed downloads.

Policy
* Nothing behind registration / data-use agreements is downloaded automatically.
  Those datasets use local-import adapters and the report states the missing requirement.
* Public downloads require an explicit confirm flag and pass a disk-space check first.
* Downloads above LARGE_DOWNLOAD_BYTES additionally require allow_large=True
  (resource authorisation), and multi-terabyte resources are never started by default.
* Credentials are read from environment variables or ~/.config/bilacl/credentials.json
  (outside the repository) and never logged.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import yaml

from .provenance import sha256_file, utcnow, write_json

LARGE_DOWNLOAD_BYTES = 20 * 1024 ** 3
REGISTRY = Path(__file__).resolve().parents[2] / "registry" / "datasets.yaml"
KNOWN_SIZES = {"skm_tea": 900 * 1024 ** 3}  # verified: 900 GB compressed (DATASET.md)


def load_registry(path=None) -> dict:
    return yaml.safe_load(Path(path or REGISTRY).read_text())


def credentials(name: str) -> dict | None:
    env = os.environ.get(f"BILACL_{name.upper()}_TOKEN")
    if env:
        return {"token": env}
    p = Path.home() / ".config" / "bilacl" / "credentials.json"
    if p.exists():
        return json.loads(p.read_text()).get(name)
    return None


def disk_free(path) -> int:
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


def remote_size(url: str, timeout=20) -> int | None:
    import httpx

    try:
        r = httpx.head(url, follow_redirects=True, timeout=timeout)
        n = r.headers.get("content-length")
        return int(n) if n else None
    except Exception:
        return None


def acquisition_report(data_root="data", probe_remote: bool = True, registry=None) -> dict:
    from .adapters.base import get_adapter

    reg = registry or load_registry()
    out = {"generated": utcnow(), "data_root": str(Path(data_root).resolve()), "disk_free_gb": round(disk_free(data_root) / 1e9, 1), "datasets": {}}
    for key, d in reg.items():
        entry = {"name": d.get("name"), "download_mode": d.get("download_mode"), "official_source": d.get("official_source"),
                 "page_verified": d.get("page_readable"), "verified_on": d.get("verified_on")}
        root = Path(data_root) / key
        if d.get("adapter") and d["adapter"] != "none":
            try:
                rep = get_adapter(d["adapter"], root).inspect()
                entry["local"] = rep.__dict__
            except Exception as e:
                entry["local"] = {"error": str(e)}
        mode = d.get("download_mode")
        if mode == "public_http":
            size = remote_size(d["download_url"]) if probe_remote else None
            entry["remote_size_bytes"] = size
            entry["requirement"] = "explicit --confirm; disk check" + ("" if size else ("; remote size not probed" if not probe_remote else "; remote size unknown (server did not report it)"))
        elif mode == "user_provided_archive":
            entry["requirement"] = f"obtain from official source under its terms, then place/extract under {root}"
            if key in KNOWN_SIZES:
                entry["expected_size_bytes"] = KNOWN_SIZES[key]
                entry["fits_on_disk"] = disk_free(data_root) > KNOWN_SIZES[key]
        else:
            entry["requirement"] = d.get("status", "no acquisition route")
        out["datasets"][key] = entry
    return out


def download(url: str, dest, *, confirm: bool = False, sha256: str | None = None, allow_large: bool = False, chunk=1 << 20, log=print) -> dict:
    """Resumable HTTP download with optional checksum verification."""
    import httpx

    dest = Path(dest)
    size = remote_size(url)
    free = disk_free(dest.parent)
    plan = {"url": url, "dest": str(dest), "remote_size_bytes": size, "disk_free_bytes": free}
    if not confirm:
        plan["status"] = "dry-run (pass confirm=True / --confirm to download)"
        return plan
    if size is not None and size > LARGE_DOWNLOAD_BYTES and not allow_large:
        raise PermissionError(f"download is {size/1e9:.1f} GB > {LARGE_DOWNLOAD_BYTES/1e9:.0f} GB: needs explicit resource authorisation (--allow-large)")
    if size is not None and size * 1.1 > free:
        raise OSError(f"insufficient disk space: need ~{size/1e9:.1f} GB, free {free/1e9:.1f} GB")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=60) as r:
        if have and r.status_code != 206:
            have = 0  # server ignored range: restart
        r.raise_for_status()
        with open(part, "ab" if have else "wb") as f:
            done = have
            for b in r.iter_bytes(chunk):
                f.write(b)
                done += len(b)
        log(f"downloaded {done} bytes")
    if size is not None and part.stat().st_size != size:
        raise IOError(f"size mismatch: got {part.stat().st_size}, expected {size} (re-run to resume)")
    digest = sha256_file(part)
    if sha256 and digest != sha256:
        raise IOError(f"checksum mismatch: {digest} != {sha256}")
    part.rename(dest)
    rec = {**plan, "status": "complete", "sha256": digest, "checksum_verified": bool(sha256), "completed": utcnow()}
    write_json(dest.with_suffix(dest.suffix + ".provenance.json"), rec)
    return rec


def select_dev_subset(records, n_subjects: int, seed: int = 0, require: str | None = None):
    """Pick whole subjects (all their exams) for a small development subset."""
    import numpy as np

    pool = sorted({r.subject_key for r in records if not require or require in r.available_labels})
    rng = np.random.default_rng(seed)
    pick = set(rng.choice(pool, size=min(n_subjects, len(pool)), replace=False)) if pool else set()
    return [r for r in records if r.subject_key in pick]
