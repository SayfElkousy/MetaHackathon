"""Shared writers. The STL writer lives here (P1). Import it, never rewrite it."""
import json
import struct
import numpy as np
from acl_engine.glb import write_glb  # noqa: F401  (re-exported; keeps colors + transparency)


def write_stl(path, bodies, header=b"ACL scaffold STL mm"):
    """Binary STL in mm. bodies = list of (V (N,3), F (M,3)). 50-byte records."""
    # FIX: Muse's T1 copy dropped the third corner (p2) of every triangle again,
    # the same bug fixed in M1 and N2. Each record needs normal + 3 corners + attribute.
    # FIX (speed): whole arrays at once instead of one struct.pack per triangle.
    # Output is byte-identical to the per-triangle version.
    rec = np.dtype([("n", "<f4", 3), ("p", "<f4", (3, 3)), ("attr", "<u2")])
    parts = []
    for (V, F) in bodies:
        V = np.asarray(V, dtype=float)
        F = np.asarray(F, dtype=np.int64)
        if F.size == 0:
            continue
        P = V[F]  # (M, 3 corners, 3)
        n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        l = np.linalg.norm(n, axis=1)
        ok = l > 1e-30
        n[ok] = n[ok] / l[ok][:, None]
        n[~ok] = (0.0, 0.0, 1.0)
        r = np.zeros(F.shape[0], dtype=rec)
        r["n"] = n
        r["p"] = P
        parts.append(r)
    data = np.concatenate(parts) if parts else np.zeros(0, dtype=rec)
    with open(path, "wb") as f:
        f.write(header + b"\x00" * (80 - len(header)))
        f.write(struct.pack("<I", len(data)))
        f.write(data.tobytes())


def read_stl_vertices(path):
    """Minimal binary-STL reader for render/check. Returns (N*3, 3) triangle soup."""
    with open(path, "rb") as f:
        f.read(80)
        n = struct.unpack("<I", f.read(4))[0]
        raw = np.frombuffer(f.read(50 * n), dtype=np.uint8).reshape(n, 50)
    floats = raw[:, 12:48].copy().view("<f4").reshape(n, 3, 3)
    return floats.reshape(-1, 3).astype(float)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
