"""Offline GLB review renders (matplotlib). Prints camera, scale bar, and
orientation triad on every figure; visual resemblance is qualitative
evidence only, never an accuracy metric.

Standard anatomical views: anterior, posterior, lateral, medial, superior,
three-quarter. Orthographic mode (--ortho) for true-scale inspection.
Cross-section + footprint slice figure from anatomy.json when present.
Reference/source images are never bundled here (see sources.json L20).
"""
import json
import math
import os
import struct
import sys
import numpy as np

ORDER = ["bone_tibia", "bone_femur_med", "bone_femur_lat", "tunnel_tibial", "tunnel_femoral",
         "footprint_tibial", "footprint_femoral",
         "acl_ghost", "acl_ghost_am", "acl_ghost_pl",
         "acl_surface", "acl_strands_am", "acl_strands_pl",
         "acl_edge_tibial", "acl_edge_femoral",
         "stump_tibial", "stump_femoral",
         "scaffold_strand_0", "scaffold_strand_1", "scaffold_strand_2"]

# (elev, azim) in matplotlib convention; x right-ish, y depth-ish.
VIEWS = {"anterior": (8, -90), "posterior": (8, 90),
         "lateral": (8, 0), "medial": (8, 180),
         "superior": (89, -90), "three-quarter": (18, -30)}
VIEW_TITLES = {"anterior": "Anterior", "posterior": "Posterior",
               "lateral": "Lateral", "medial": "Medial",
               "superior": "Superior", "three-quarter": "3/4 view"}


def load_glb(path):
    raw = open(path, "rb").read()
    ln = struct.unpack("<I", raw[12:16])[0]
    js = json.loads(raw[20:20 + ln])
    binoff = 20 + ln
    clen = struct.unpack("<I", raw[binoff:binoff + 4])[0]
    binc = raw[binoff + 8:binoff + 8 + clen]
    bufs = js["bufferViews"]
    accs = js["accessors"]
    ncomp = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}

    def acc(i):
        a = accs[i]
        bv = bufs[a["bufferView"]]
        start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        dt = {"5126": np.float32, "5123": np.uint16, "5125": np.uint32}[str(a["componentType"])]
        return np.frombuffer(binc[start:start + bv["byteLength"]],
                             dtype=dt).reshape(a["count"], ncomp[a["type"]])
    out = {}
    for m in js["meshes"]:
        p = m["primitives"][0]
        V = acc(p["attributes"]["POSITION"]).astype(float) * 1000.0
        F = acc(p["indices"]).astype(np.int64).reshape(-1, 3)
        col = js["materials"][p.get("material", 0)]["pbrMetallicRoughness"]["baseColorFactor"]
        out[m["name"]] = (V, F, col)
    return out


def _color(name, col):
    c = np.array(col[:3], dtype=float)
    if name.startswith("bone") or name.startswith("tunnel"):
        return np.array([0.92, 0.90, 0.86])
    if name.startswith("scaffold"):
        return np.array([0.55, 0.70, 0.95])
    return np.clip(c, 0, 1)


def _alpha(name):
    if name == "acl_ghost":
        return 0.22
    if name in ("acl_ghost_am", "acl_ghost_pl"):
        return 0.22
    if name == "acl_surface":
        return 0.35
    if name.startswith("footprint"):
        return 0.85
    if name.startswith("stump"):
        return 0.55
    if name.startswith("acl_edge"):
        return 1.0
    return 1.0


def shade(V, F, base):
    P = V[F]
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    li = np.array([0.4, 0.5, 0.75])
    li = li / np.linalg.norm(li)
    b = 0.35 + 0.65 * np.abs(n @ li)
    return b[:, None] * base[None, :]


def _triad(ax, corner, size=8.0):
    cols = {"X": (0.8, 0.2, 0.2), "Y": (0.2, 0.6, 0.2), "Z": (0.2, 0.3, 0.8)}
    for i, k in enumerate("XYZ"):
        d = np.zeros(3)
        d[i] = size
        ax.quiver(*corner, *d, color=cols[k], linewidth=1.5, arrow_length_ratio=0.15)
        ax.text(*(corner + d * 1.15), k, color=cols[k], fontsize=8)


def _scale_bar(ax, at, length_mm=10.0):
    x0, y0, z0 = (float(v) for v in at)
    ax.plot([x0, x0 + length_mm], [y0, y0], [z0, z0], color="k", linewidth=2)
    ax.text(x0 + length_mm / 2.0, y0, z0, "%g mm" % length_mm, fontsize=7,
            ha="center", va="bottom")


def draw(M, names, ortho=False, labels=None, title_note=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    allv = np.concatenate([M[n][0] for n in names if n in M])
    lo = allv.min(axis=0)
    hi = allv.max(axis=0)
    fig, axs = plt.subplots(1, len(VIEWS), figsize=(4 * len(VIEWS), 4.4),
                            subplot_kw={"projection": "3d"})
    for ax, key in zip(axs, VIEWS):
        for n in names:
            if n not in M:
                continue
            V, F, col = M[n]
            if V.shape[0] == 0 or F.shape[0] == 0:
                continue
            pc = Poly3DCollection(V[F], facecolors=shade(V, F, _color(n, col)),
                                 edgecolors="none", alpha=_alpha(n))
            ax.add_collection3d(pc)
        span = (hi - lo).max()
        mid = (hi + lo) / 2
        ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
        ax.set_ylim(mid[1] - span / 2, mid[1] + span / 2)
        ax.set_zlim(mid[2] - span / 2, mid[2] + span / 2)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        if ortho:
            try:
                ax.set_proj_type("ortho")
            except Exception:
                pass
        ax.view_init(elev=VIEWS[key][0], azim=VIEWS[key][1])
        ax.set_title(VIEW_TITLES[key], fontsize=9)
        ax.set_axis_off()
        _triad(ax, lo + np.array([0, 0, 0]), size=span * 0.12)
        _scale_bar(ax, (hi[0] - span * 0.25, lo[1], lo[2]), 10.0)
        if labels:
            for txt, pos in labels:
                ax.text(*pos, txt, fontsize=7)
    mode = "orthographic" if ortho else "perspective"
    fig.suptitle(title_note + " (%s, qualitative only)" % mode, fontsize=10)
    fig.tight_layout()
    return fig


def _labels_from_anatomy(out_dir):
    path = os.path.join(out_dir, "anatomy.json")
    if not os.path.exists(path):
        return []
    try:
        an = json.load(open(path))
    except Exception:
        return []
    out = []
    secs = an.get("sections", [])
    if secs:
        mid = secs[len(secs) // 2]
        out.append(("ACL mid", mid["center_mm"]))
        out.append(("tibial FP", secs[0]["center_mm"]))
        out.append(("femoral FP", secs[-1]["center_mm"]))
    return out


def render_knee(glb, png=None, title="", views=None, ortho=False):
    if png is None:
        # check.py convention: render_knee(out_dir) renders knee.glb inside it
        out = glb
        glb = os.path.join(out, "knee.glb")
        png = os.path.join(out, "knee.png")
        if not title:
            title = os.path.basename(os.path.normpath(out))
    M = load_glb(glb)
    names = [n for n in ORDER if n in M]
    anat = "acl_surface" in M
    labels = _labels_from_anatomy(os.path.dirname(os.path.abspath(glb)))
    fig = draw(M, names, ortho=ortho, labels=labels,
               title_note=((title + " - ") if title else "")
               + ("anatomical ACL loft" if anat else "schematic knee")
               + ", not patient anatomy")
    fig.savefig(png, dpi=80)
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass
    return png


def render_sections(out_dir, png_name="sections.png"):
    """Cross-section rings + profiles + footprint loops from anatomy.json
    and the GLB edge meshes. True-shape 2-D plots (equal aspect)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    apath = os.path.join(out_dir, "anatomy.json")
    if not os.path.exists(apath):
        return None
    an = json.load(open(apath))
    secs = an.get("sections", [])
    if not secs:
        return None
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    for i, s in enumerate(secs[::max(1, len(secs) // 3)][:3]):
        ax = axs[0][i]
        ring = np.array(s["ring_mm"])
        ax.plot(ring[:, 0], ring[:, 1], "b-")
        ax.set_aspect("equal")
        ax.set_title("t=%.2f A=%.1fmm2" % (s["t"], s["area_mm2"]), fontsize=9)
    t = [s["t"] for s in secs]
    axs[1][0].plot(t, [s["width_mm"] for s in secs], label="width")
    axs[1][0].plot(t, [s["thickness_mm"] for s in secs], label="thickness")
    axs[1][0].legend(fontsize=8)
    axs[1][0].set_title("Width / thickness (mm)", fontsize=9)
    axs[1][1].plot(t, [s["area_mm2"] for s in secs])
    axs[1][1].set_title("Section area (mm2)", fontsize=9)
    axs[1][2].plot(t, [s["rotation_deg"] for s in secs])
    axs[1][2].set_title("Section rotation (deg)", fontsize=9)
    for ax in axs[1]:
        ax.set_xlabel("normalized arc length", fontsize=8)
    fig.suptitle("ACL sections + profiles (from anatomy.json)", fontsize=10)
    fig.tight_layout()
    p = os.path.join(out_dir, png_name)
    fig.savefig(p, dpi=80)
    plt.close(fig)
    return p


def render_compare(before_glb, after_glb, png, title="before vs after"):
    """Same cameras, two rows: legacy ghost vs loft surface. Qualitative."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    Mb = load_glb(before_glb)
    Ma = load_glb(after_glb)
    rows = [(Mb, ["acl_ghost"], "before: schematic ghost"),
            (Ma, ["acl_surface", "acl_strands_am", "acl_strands_pl"], "after: loft + strands")]
    keys = ["anterior", "lateral", "three-quarter"]
    fig, axs = plt.subplots(2, len(keys), figsize=(12, 8),
                            subplot_kw={"projection": "3d"})
    for r, (M, names, note) in enumerate(rows):
        names = [n for n in names if n in M]
        if not names:
            continue
        allv = np.concatenate([M[n][0] for n in names])
        lo = allv.min(axis=0)
        hi = allv.max(axis=0)
        for c, key in enumerate(keys):
            ax = axs[r][c]
            for n in names:
                V, F, col = M[n]
                pc = Poly3DCollection(V[F], facecolors=shade(V, F, _color(n, col)),
                                     edgecolors="none", alpha=_alpha(n))
                ax.add_collection3d(pc)
            span = (hi - lo).max()
            mid = (hi + lo) / 2
            ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
            ax.set_ylim(mid[1] - span / 2, mid[1] + span / 2)
            ax.set_zlim(mid[2] - span / 2, mid[2] + span / 2)
            try:
                ax.set_box_aspect((1, 1, 1))
            except Exception:
                pass
            try:
                ax.set_proj_type("ortho")
            except Exception:
                pass
            ax.view_init(elev=VIEWS[key][0], azim=VIEWS[key][1])
            ax.set_title(("%s | " % note if c == 0 else "") + VIEW_TITLES[key], fontsize=8)
            ax.set_axis_off()
    fig.suptitle(title + " (matching ortho cameras, qualitative)", fontsize=10)
    fig.tight_layout()
    fig.savefig(png, dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "compare" and len(a) >= 5:
        render_compare(a[2], a[3], a[4], a[5] if len(a) > 5 else "before vs after")
    elif a[1] == "sections" and len(a) >= 3:
        print(render_sections(a[2], a[3] if len(a) > 3 else "sections.png"))
    else:
        ortho = "--ortho" in a
        a = [x for x in a if x != "--ortho"]
        render_knee(a[1], a[2], a[3] if len(a) > 3 else "", ortho=ortho)
