"""GLB writer with named nodes; coordinates converted mm -> m (Muse N2)."""
import json
import struct
import numpy as np
from acl_engine import GEOM_KERNEL

# FIX: Muse set alpha 0.6 on bones/ghost but no alphaMode, so viewers drew them
# solid and the scaffold was hidden inside. Colors per part also added so parts are distinguishable.
COLORS = {
    "scaffold_strand_0": [0.89, 0.61, 0.18, 1.0],
    "scaffold_strand_1": [0.23, 0.40, 0.86, 1.0],
    "scaffold_strand_2": [0.81, 0.29, 0.46, 1.0],
    "acl_ghost": [0.95, 0.45, 0.45, 0.25],
    "bone": [0.92, 0.89, 0.82, 0.35],
    "tunnel": [0.55, 0.60, 0.68, 0.30],
    # Milestone A: anatomical parts (alpha<1 renders see-through)
    "acl_ghost_am": [0.95, 0.55, 0.45, 0.28],
    "acl_ghost_pl": [0.55, 0.55, 0.95, 0.28],
    "footprint_femoral": [0.98, 0.95, 0.88, 0.85],
    "footprint_tibial": [0.98, 0.95, 0.88, 0.85],
    "stump_femoral": [0.78, 0.20, 0.22, 0.90],
    "stump_tibial": [0.85, 0.35, 0.30, 0.90],
    "acl_surface": [0.90, 0.62, 0.50, 0.35],
    "acl_strands_am": [0.75, 0.25, 0.20, 1.0],
    "acl_strands_pl": [0.25, 0.35, 0.75, 1.0],
    "acl_edge_tibial": [0.20, 0.20, 0.20, 1.0],
    "acl_edge_femoral": [0.20, 0.20, 0.20, 1.0],
}


def _color(name):
    if name in COLORS:
        return COLORS[name]
    for k in ("bone", "tunnel"):
        if name.startswith(k):
            return COLORS[k]
    return [0.75, 0.75, 0.78, 1.0]


def write_glb(path, named_meshes):
    items = []
    for m in named_meshes:
        V = (np.asarray(m["vertices"], dtype=np.float64) * 0.001).astype(np.float32)
        F = np.asarray(m["faces"], dtype=np.int64).astype(np.uint32)
        items.append((m["name"], V, F))
    binbuf = b""
    views, accs, meshes, nodes, mats = [], [], [], [], []
    for i, (name, V, F) in enumerate(items):
        vb = V.tobytes()
        ib = F.tobytes()
        off_v = len(binbuf)
        binbuf += vb
        off_i = len(binbuf)
        binbuf += ib
        vi = len(views)
        views.append({"buffer": 0, "byteOffset": off_v, "byteLength": len(vb), "target": 34962})
        views.append({"buffer": 0, "byteOffset": off_i, "byteLength": len(ib), "target": 34963})
        ai = len(accs)
        accs.append({"bufferView": vi, "componentType": 5126, "count": int(V.shape[0]), "type": "VEC3",
                     "min": [float(v) for v in V.min(axis=0)], "max": [float(v) for v in V.max(axis=0)]})
        accs.append({"bufferView": vi + 1, "componentType": 5125, "count": int(F.size), "type": "SCALAR"})
        meshes.append({"primitives": [{"attributes": {"POSITION": ai}, "indices": ai + 1, "material": i}], "name": name})
        col = _color(name)
        mat = {"name": name, "pbrMetallicRoughness": {"baseColorFactor": col, "metallicFactor": 0.0, "roughnessFactor": 0.6},
               "doubleSided": True}
        if col[3] < 1.0:
            mat["alphaMode"] = "BLEND"
        mats.append(mat)
        nodes.append({"mesh": i, "name": name})
    js = {"asset": {"version": "2.0", "generator": GEOM_KERNEL}, "scene": 0,
          "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes, "meshes": meshes, "materials": mats,
          "accessors": accs, "bufferViews": views, "buffers": [{"byteLength": len(binbuf)}]}
    j = json.dumps(js, separators=(",", ":")).encode("utf-8")
    while len(j) % 4 != 0:
        j += b" "
    while len(binbuf) % 4 != 0:
        binbuf += b"\x00"
    total = 12 + 8 + len(j) + 8 + len(binbuf)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(j), 0x4E4F534A))
        f.write(j)
        f.write(struct.pack("<II", len(binbuf), 0x004E4942))
        f.write(binbuf)
