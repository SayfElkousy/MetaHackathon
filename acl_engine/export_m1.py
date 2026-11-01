"""Minimal M1 export: binary STL (mm) and parameters.json."""
import json
import struct
import numpy as np


def write_stl_binary(path, bodies):
    """bodies: list of (V, F). Writes one binary STL with all triangles."""
    tris = []
    for (V, F) in bodies:
        V = np.asarray(V, dtype=float)
        F = np.asarray(F, dtype=np.int64)
        for t in F:
            tris.append((V[t[0]], V[t[1]], V[t[2]]))
    with open(path, "wb") as f:
        f.write(b"ACL scaffold M1 STL mm" + b"\x00" * (80 - 22))
        f.write(struct.pack("<I", len(tris)))
        for (p0, p1, p2) in tris:
            n = np.cross(p1 - p0, p2 - p0)
            l = np.linalg.norm(n)
            if l < 1e-30:
                n = np.array([0.0, 0.0, 1.0])
            else:
                n = n / l
            f.write(struct.pack("<12fH",
                                float(n[0]), float(n[1]), float(n[2]),
                                float(p0[0]), float(p0[1]), float(p0[2]),
                                float(p1[0]), float(p1[1]), float(p1[2]),
                                float(p2[0]), float(p2[1]), float(p2[2]),
                                0))


def write_parameters_json(path, resolved):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(resolved, f, indent=2, sort_keys=True)
