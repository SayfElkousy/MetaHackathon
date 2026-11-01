"""Draw outputs/<name>/knee.glb from three angles into knee.png (no internet needed).

Usage:  python tools/render.py outputs/acl_context/knee.glb outputs/acl_context/knee.png "title"
   or:  from tools.render import render_knee; render_knee("outputs/acl_context")

FIX (Claude): Muse's T1 render plotted a grey point cloud from the STL with no bones or
ghost. This keeps the shaded, colored 3-view picture and adds Muse's render_knee() entry point.
"""
import json
import os
import struct
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ORDER = ["bone_tibia", "bone_femur_med", "bone_femur_lat", "tunnel_tibial", "tunnel_femoral",
         "footprint_tibial", "footprint_femoral",
         "acl_ghost", "acl_ghost_am", "acl_ghost_pl",
         "stump_tibial", "stump_femoral",
         "scaffold_strand_0", "scaffold_strand_1", "scaffold_strand_2"]
VIEWS = [(8, -90, "Front (coronal)"), (8, 0, "Side (sagittal)"), (25, -55, "3/4 view")]


def load_glb(p):
    b = open(p, "rb").read()
    jl = struct.unpack("<I", b[12:16])[0]
    js = json.loads(b[20:20 + jl])
    binb = b[20 + jl + 8:]
    out = {}
    for n in js["nodes"]:
        pr = js["meshes"][n["mesh"]]["primitives"][0]
        a = js["accessors"][pr["attributes"]["POSITION"]]
        v = js["bufferViews"][a["bufferView"]]
        V = np.frombuffer(binb[v["byteOffset"]:v["byteOffset"] + v["byteLength"]], dtype="<f4").reshape(-1, 3) * 1000
        a = js["accessors"][pr["indices"]]
        v = js["bufferViews"][a["bufferView"]]
        F = np.frombuffer(binb[v["byteOffset"]:v["byteOffset"] + v["byteLength"]], dtype="<u4").reshape(-1, 3)
        col = js["materials"][pr["material"]]["pbrMetallicRoughness"]["baseColorFactor"]
        out[n["name"]] = (V, F, col)
    return out


def shade(tri, c, al):
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    light = np.array([0.4, 0.6, 0.7])
    light /= np.linalg.norm(light)
    s = 0.45 + 0.55 * np.abs(n @ light)
    rgb = np.clip(np.outer(s, c[:3]), 0, 1)
    return np.hstack([rgb, np.full((len(rgb), 1), al)])


def _alpha(name):
    if name.startswith("bone"):
        return 0.10
    if name.startswith("tunnel"):
        return 0.18
    if name == "acl_ghost":
        return 0.22
    # Milestone A: anatomical parts
    if name in ("acl_ghost_am", "acl_ghost_pl"):
        return 0.22
    if name.startswith("footprint"):
        return 0.85
    if name.startswith("stump"):
        return 0.55
    return 1.0


def render_glb(glb_path, png_path, title=""):
    M = load_glb(glb_path)
    fig = plt.figure(figsize=(14, 7))
    for k, (el, az, t) in enumerate(VIEWS):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        for name in ORDER:
            if name not in M:
                continue
            V, F, col = M[name]
            tri = V[F]
            ax.add_collection3d(Poly3DCollection(tri, facecolors=shade(tri, col, _alpha(name)), edgecolor="none"))
        ax.set_xlim(-35, 35)
        ax.set_ylim(-35, 35)
        ax.set_zlim(-25, 45)
        ax.set_box_aspect((70, 70, 70))
        ax.view_init(el, az)
        ax.set_axis_off()
        ax.set_title(t, fontsize=10)
    # Milestone A: name the ACL mode shown (anatomical adds bundle ghosts)
    _amode = "anatomical ACL" if "acl_ghost_am" in M else "schematic knee"
    plt.suptitle(((title + " - ") if title else "") + _amode + ", not patient anatomy", fontsize=11)
    plt.tight_layout()
    plt.savefig(png_path, dpi=80)
    plt.close(fig)
    return png_path


def render_knee(out_dir, title=None):
    """Muse T1 entry point: reads <out_dir>/knee.glb, writes <out_dir>/knee.png."""
    if title is None:
        title = os.path.basename(os.path.normpath(out_dir))
    return render_glb(os.path.join(out_dir, "knee.glb"), os.path.join(out_dir, "knee.png"), title)


if __name__ == "__main__":
    render_glb(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
