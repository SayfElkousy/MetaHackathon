"""Planar sections of the scene geometry.

Every section is cut from the SAME objects the 3D viewer draws and the exports
contain (meshes are sliced triangle by triangle; strands and fascicles are
cut as tubes, giving their true elliptical cross-sections). Nothing is drawn
from a separate schematic. Interface lines and the boundary-uncertainty band
come from the same section model the meshes were built from.

Plane kinds
-----------
ligament      perpendicular to the ACL centreline at s (0 tibia .. 1 femur)
am / pl       perpendicular to that bundle's centroid path at s
axial         horizontal (normal +Z, proximal), offset in mm from the ACL midpoint
sagittal      normal +X (lateral for a right knee)
coronal       normal +Y (anterior)
"""
import math
import numpy as np
from acl_engine.v2 import mesher

AXIS_WORDS = {0: ("medial", "lateral"), 1: ("posterior", "anterior"), 2: ("distal", "proximal")}
SECTION_CATEGORIES = {"reference", "bundle", "retained", "injury", "repair_space", "bone", "cartilage",
                      "footprint", "landmark", "observed", "scaffold_region", "layer_sheath", "layer_end", "layer_plug", "layer_pores", "layer_coating",
                      "layer_interface"}


# ---------------------------------------------------------------- lofts
def ring_loft(lig, ring_m, s0=0.0, s1=1.0, n=None):
    """Closed tube through material ring ring_m (K,2) from s0 to s1 (capped)."""
    ring_m = np.asarray(ring_m, float)
    K = ring_m.shape[0]
    if n is None:
        n = max(8, int(round((s1 - s0) * lig.s.size)) + 1)
    ss = np.linspace(s0, s1, n)
    R = lig.map(np.repeat(ss, K), np.tile(ring_m, (n, 1))).reshape(n, K, 3)
    V = R.reshape(-1, 3)
    i = np.arange(n - 1)[:, None]
    j = np.arange(K)[None, :]
    a0 = i * K + j
    a1 = i * K + (j + 1) % K
    F = np.concatenate([np.stack([a0, a1 + K, a1], -1).reshape(-1, 3), np.stack([a0, a0 + K, a1 + K], -1).reshape(-1, 3)])
    cm = ring_m.mean(0)
    c0 = V.shape[0]
    caps = lig.map(np.array([s0, s1]), np.repeat(cm[None], 2, 0))
    V = np.vstack([V, caps])
    jj = np.arange(K)
    F = np.vstack([F, np.stack([np.full(K, c0), jj, (jj + 1) % K], 1),
                   np.stack([np.full(K, c0 + 1), (n - 1) * K + (jj + 1) % K, (n - 1) * K + jj], 1)])
    if mesher.audit(V, F)["volume_mm3"] < 0:
        F = F[:, ::-1]
    return V, F.astype(np.int64)


def bundle_ring(lig, b, k_chord=24, k_arc=72):
    """Material outline of one bundle: its arc plus the AM/PL chord (closed, CCW)."""
    t = np.linspace(0, 1, k_arc, endpoint=False)
    arc = lig._m_arc(b, t)
    tc = np.linspace(1, 0, k_chord, endpoint=False)
    chord = lig._m_chord(tc)
    ring = np.vstack([arc, chord])
    a = 0.5 * (ring[:, 0] * np.roll(ring[:, 1], -1) - np.roll(ring[:, 0], -1) * ring[:, 1]).sum()
    return ring if a > 0 else ring[::-1]


def interface_band(lig, unc, n_t=24):
    """Thin closed slab around the AM/PL interface (half-width = uncertainty), whole length."""
    t = np.linspace(0, 1, n_t)
    chord = lig._m_chord(t)
    n = lig.s.size
    S = np.repeat(lig.s, n_t)
    Pc = lig.map(S, np.tile(chord, (n, 1))).reshape(n, n_t, 3)
    nrm = lig.iface_n
    off = (nrm[:, :1] * lig.N1 + nrm[:, 1:] * lig.N2)[:, None, :]
    half = max(unc, 0.05)
    top = Pc + half * off
    bot = Pc - half * off
    V = np.vstack([top.reshape(-1, 3), bot.reshape(-1, 3)])
    idx = np.arange(n * n_t).reshape(n, n_t)
    F = []
    for offv, flip in ((0, False), (n * n_t, True)):
        a = idx[:-1, :-1] + offv
        b = idx[1:, :-1] + offv
        c = idx[1:, 1:] + offv
        d = idx[:-1, 1:] + offv
        T = np.vstack([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
        F.append(T[:, ::-1] if flip else T)
    B = n * n_t
    for edge in (idx[0, :], idx[-1, ::-1], idx[::-1, 0], idx[:, -1]):
        e0, e1 = edge[:-1], edge[1:]
        F.append(np.stack([e0, e1, e1 + B], 1))
        F.append(np.stack([e0, e1 + B, e0 + B], 1))
    F = np.vstack(F)
    if mesher.audit(V, F)["volume_mm3"] < 0:
        F = F[:, ::-1]
    return V, F.astype(np.int64)


# ---------------------------------------------------------------- planes
def _words(v):
    """Anatomical words for both directions of an in-plane axis; an oblique axis gets
    a compound name (e.g. anterior-distal) so the label is not misleading."""
    a = np.abs(v)
    ks = [k for k in np.argsort(-a) if a[k] >= 0.45 * a.max()][:2]
    pos = "-".join(AXIS_WORDS[k][1] if v[k] > 0 else AXIS_WORDS[k][0] for k in ks)
    neg = "-".join(AXIS_WORDS[k][0] if v[k] > 0 else AXIS_WORDS[k][1] for k in ks)
    return neg, pos


def plane_for(kind, pos, lig):
    kind = kind or "ligament"
    if kind in ("ligament", "am", "pl"):
        s = float(np.clip(pos, 0.0, 1.0))
        if kind == "ligament":
            i = int(round(s * (lig.s.size - 1)))
            O, n, e1 = lig.C[i], lig.T[i], lig.N1[i]
            title = "Perpendicular to the ACL centreline at s = %.2f (%.1f mm from the tibial attachment)" % (s, s * lig.L)
        else:
            b = 0 if kind == "am" else 1
            ss = np.clip(np.array([s - 0.01, s, s + 0.01]), 0, 1)
            P = lig.bundle_centerline(b, ss)
            O = P[1]
            n = P[2] - P[0]
            n /= max(np.linalg.norm(n), 1e-12)
            i = int(round(s * (lig.s.size - 1)))
            e1 = lig.N1[i] - (lig.N1[i] @ n) * n
            title = "Perpendicular to the %s bundle path at s = %.2f" % (kind.upper(), s)
        e1 = e1 / max(np.linalg.norm(e1), 1e-12)
        e2 = np.cross(n, e1)
        return {"kind": kind, "pos": s, "origin": O, "normal": n, "e1": e1, "e2": e2, "title": title,
                "s": s, "station": int(round(s * (lig.s.size - 1)))}
    mid = lig.C[lig.s.size // 2]
    d = float(pos)
    if kind == "axial":
        n, e1 = np.array([0, 0, 1.0]), np.array([1.0, 0, 0])
        title = "Axial section, %+.1f mm from the ACL midpoint (z = %.1f mm)" % (d, mid[2] + d)
    elif kind == "sagittal":
        n, e1 = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
        title = "Sagittal section, %+.1f mm from the ACL midpoint (x = %.1f mm)" % (d, mid[0] + d)
    elif kind == "coronal":
        n, e1 = np.array([0, 1.0, 0]), np.array([1.0, 0, 0])
        title = "Coronal section, %+.1f mm from the ACL midpoint (y = %.1f mm)" % (d, mid[1] + d)
    else:
        raise ValueError("unknown section kind %r" % kind)
    O = mid + d * n
    e2 = np.cross(n, e1)
    return {"kind": kind, "pos": d, "origin": O, "normal": n, "e1": e1, "e2": e2, "title": title, "s": None, "station": None}


# ---------------------------------------------------------------- slicing
def slice_mesh(V, F, O, n, e1, e2):
    """Triangle mesh x plane -> 2D segments (m, 4) [x0, y0, x1, y1] in the plane axes."""
    d = (V - O) @ n
    D = d[F]
    pos = D > 0
    cnt = pos.sum(1)
    cross = (cnt == 1) | (cnt == 2)
    if not cross.any():
        return np.zeros((0, 4))
    Fc = F[cross]
    Dc = D[cross]
    Pc = V[Fc]
    segs = []
    pts = []
    for (a, b) in ((0, 1), (1, 2), (2, 0)):
        da, db = Dc[:, a], Dc[:, b]
        hit = (da > 0) != (db > 0)
        t = np.where(hit, da / np.where(hit, da - db, 1.0), 0.0)
        P = Pc[:, a] + t[:, None] * (Pc[:, b] - Pc[:, a])
        pts.append((P, hit))
    out = np.zeros((Fc.shape[0], 2, 3))
    k = np.zeros(Fc.shape[0], int)
    for P, hit in pts:
        sel = hit & (k < 2)
        out[sel, k[sel]] = P[sel]
        k[sel] += 1
    ok = k == 2
    Q = out[ok] - O
    return np.concatenate([np.stack([Q[:, 0, :] @ e1, Q[:, 0, :] @ e2], 1),
                           np.stack([Q[:, 1, :] @ e1, Q[:, 1, :] @ e2], 1)], 1)


def slice_tubes(lines, O, n, e1, e2, meta_keys=()):
    """Polyline tubes x plane -> ellipses {x, y, a, b, ang} (true tube cross-sections)."""
    out = []
    for ln in lines:
        P = ln["P"]
        if P.shape[0] < 2:
            continue
        d = (P - O) @ n
        sgn = d > 0
        idx = np.nonzero(sgn[:-1] != sgn[1:])[0]
        for i in idx:
            t = d[i] / (d[i] - d[i + 1])
            X = P[i] + t * (P[i + 1] - P[i])
            tan = P[i + 1] - P[i]
            tan /= max(np.linalg.norm(tan), 1e-12)
            r = float(ln["r"][i] * (1 - t) + ln["r"][i + 1] * t)
            ri = float(ln["ri"][i] * (1 - t) + ln["ri"][i + 1] * t) if "ri" in ln else None
            c = abs(float(tan @ n))
            a = r / max(c, 0.15)                       # long semi-axis along the projected tangent
            tp = np.array([tan @ e1, tan @ e2])
            ang = math.atan2(tp[1], tp[0]) if np.linalg.norm(tp) > 1e-9 else 0.0
            Q = X - O
            e = {"x": float(Q @ e1), "y": float(Q @ e2), "a": a, "b": r, "ang": ang}
            if ri is not None:
                e["ai"], e["bi"] = ri / max(c, 0.15), ri
            if "coat" in ln:
                e["coat"] = int(max(ln["coat"][i], ln["coat"][i + 1]))
            for k in meta_keys:
                if k in ln:
                    v = ln[k]
                    if isinstance(v, np.ndarray):
                        v = v[i]
                    e[k] = v.item() if hasattr(v, "item") else v
            if "viol" in ln:
                e["viol"] = int(max(ln["viol"][i], ln["viol"][i + 1]))
            out.append(e)
    return out


def section(objects, lig, kind, pos, side, unc_mm=0.0, violations=None):
    pl = plane_for(kind, pos, lig)
    O, n, e1, e2 = pl["origin"], pl["normal"], pl["e1"], pl["e2"]
    if side == "left":
        M = np.array([-1.0, 1, 1])
        Om, nm, e1m, e2m = O * M, n * M, e1 * M, e2 * M
    else:
        Om, nm, e1m, e2m = O, n, e1, e2
    layers = []
    for o in objects:
        if o["kind"] != "mesh" or o["category"] not in SECTION_CATEGORIES:
            continue
        sg = slice_mesh(o["V"], o["F"], Om, nm, e1m, e2m)
        if sg.shape[0]:
            layers.append({"id": o["id"], "label": o["label"], "category": o["category"], "status": o.get("status"), "level": o.get("level"),
                           "region": o.get("region"), "segs": np.round(sg, 4).ravel().tolist()})
    tubes = {}
    for o in objects:
        if o["id"] == "scaffold_yarns":
            tubes["yarns"] = slice_tubes(o["lines"], Om, nm, e1m, e2m, ("yarn", "family", "unit", "layer"))
        elif o["id"] == "native_fascicles":
            tubes["fascicles"] = slice_tubes(o["lines"], Om, nm, e1m, e2m, ("fascicle", "bundle", "bridging"))
        elif o["id"] in ("layer_channels", "layer_ports"):
            tubes.setdefault("channels", []).extend(slice_tubes(o["lines"], Om, nm, e1m, e2m, ("group", "channel")))
        elif o["id"] == "layer_braid":
            tubes["braid"] = slice_tubes(o["lines"], Om, nm, e1m, e2m, ("group", "family"))
    extra = {}
    if kind in ("ligament", "am", "pl") and pl["station"] is not None and kind == "ligament":
        i = pl["station"]
        I = lig.I_w[i]
        P3 = lig.frame_point(np.full(I.shape[0], lig.s[i]), I)
        if side == "left":
            P3 = P3 * np.array([-1.0, 1, 1])
        Q = P3 - Om
        iface = np.stack([Q @ e1m, Q @ e2m], 1)
        extra["interface"] = np.round(iface, 4).ravel().tolist()
        if unc_mm > 0:
            nv = lig.iface_n[i]
            band = []
            for sgn in (1, -1):
                Pb = lig.frame_point(np.full(I.shape[0], lig.s[i]), I + sgn * unc_mm * nv[None])
                if side == "left":
                    Pb = Pb * np.array([-1.0, 1, 1])
                Qb = Pb - Om
                band.append(np.stack([Qb @ e1m, Qb @ e2m], 1))
            poly = np.vstack([band[0], band[1][::-1]])
            extra["uncertainty_band"] = np.round(poly, 4).ravel().tolist()
        extra["am_fraction"] = float(lig.am_fraction_actual[i])
        extra["area_mm2"] = float(lig.area[i])
        extra["bundle_area_mm2"] = lig.area_bundle[i].tolist()
    marks = []
    for v in (violations or []):
        P = np.asarray(v["P"], float)
        if side == "left":
            P = P * np.array([-1.0, 1, 1])
        dz = float((P - Om) @ nm)
        if abs(dz) < 0.6:
            Q = P - Om
            marks.append({"x": float(Q @ e1m), "y": float(Q @ e2m), "bits": v["bits"]})
    return {"kind": pl["kind"], "pos": pl["pos"], "title": pl["title"], "s": pl["s"],
            "sigma_mm": None if pl["s"] is None else pl["s"] * lig.L,
            "axes": {"x": list(_words(e1)), "y": list(_words(e2))},
            "origin_mm": O.tolist(), "normal": n.tolist(), "mirrored": side == "left",
            "layers": layers, "yarns": tubes.get("yarns", []), "fascicles": tubes.get("fascicles", []),
            "channels": tubes.get("channels", []), "braid": tubes.get("braid", []),
            "violations": marks, **extra}
