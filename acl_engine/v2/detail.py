"""Magnified level details: true-scale local geometry of one scaffold level.

The whole-ligament view is drawn in millimetres; nano-texture, coatings and grooves
are far too small to see there. Each function below builds a small patch of the
SAME design parameters at true scale (coordinates in mm), so the viewer can show
it in a magnified inset whose enlargement it labels. Nothing here is exaggerated
unless `relief_x` > 1, which is reported in the note.
"""
import math
import numpy as np
from acl_engine.v2 import mesher
from acl_engine.v2.design import MATERIALS, normalize_composition

UM = 1e-3
NM = 1e-6


def _col(comp, fallback=(0.8, 0.8, 0.8)):
    comp = normalize_composition(comp or {})
    if not comp:
        return list(fallback)
    c = np.zeros(3)
    for k, v in comp.items():
        c += v * np.asarray(MATERIALS.get(k, {"color": fallback})["color"])
    return np.clip(c, 0, 1).tolist()


def _grid_solid(Z_top, X, Y, z_bottom):
    """Closed slab under a height field Z_top (nx, ny) over grid X, Y; bottom plane z_bottom."""
    nx, ny = Z_top.shape
    top = np.stack([X, Y, Z_top], -1).reshape(-1, 3)
    bot = np.stack([X, Y, np.full_like(Z_top, z_bottom)], -1).reshape(-1, 3)
    V = np.vstack([top, bot])
    off = nx * ny
    i, j = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
    a = (i * ny + j).ravel()
    b, c, d = a + ny, a + ny + 1, a + 1
    F = [np.stack([a, b, c], 1), np.stack([a, c, d], 1), np.stack([a, c, b], 1) + off, np.stack([a, d, c], 1) + off]
    # side walls
    def wall(idx):
        idx = np.asarray(idx)
        t0, t1 = idx[:-1], idx[1:]
        return [np.stack([t0, t1, t1 + off], 1), np.stack([t0, t1 + off, t0 + off], 1)]
    for e in (np.arange(ny)[::-1], (nx - 1) * ny + np.arange(ny), np.arange(nx) * ny, (np.arange(nx) * ny + ny - 1)[::-1]):
        F += wall(e)
    F = np.vstack(F).astype(np.int64)
    if mesher.audit(V, F)["volume_mm3"] < 0:
        F = F[:, ::-1]
    return V, F


def _sheet(Z, X, Y):
    """Open (two-sided) surface over a grid: for thin translucent layers."""
    nx, ny = Z.shape
    V = np.stack([X, Y, Z], -1).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
    a = (i * ny + j).ravel()
    F = np.vstack([np.stack([a, a + ny, a + ny + 1], 1), np.stack([a, a + ny + 1, a + 1], 1)]).astype(np.int64)
    return V, F


def _groove_depth(q, p):
    """Depth (mm) below the surface at across-groove coordinate q (mm): flat ridge, straight
    walls, rounded bottom of the given radius."""
    w = float(p["gr.width_um"]) * UM
    h = float(p["gr.depth_um"]) * UM
    ridge = float(p["gr.ridge_um"]) * UM
    rb = min(float(p["gr.bottom_radius_um"]) * UM, w / 2)
    per = w + ridge
    u = np.mod(q, per) - ridge / 2                 # groove occupies 0 .. w after the half ridge
    inside = (u > 0) & (u < w)
    x = np.clip(u, 0, w)
    d = np.where(inside, h, 0.0)
    # rounded corners at the bottom
    for c in (rb, w - rb):
        dx = np.abs(x - c)
        corner = inside & (((c == rb) & (x < rb)) | ((c != rb) & (x > w - rb)))
        if rb > 0:
            d = np.where(corner, h - rb + np.sqrt(np.clip(rb * rb - dx * dx, 0, None)), d)
    return d


def _half_tube_wall(r_in_fn, r_out, length, nphi=160, nz=60, phi_span=math.pi):
    """Closed solid between an inner surface r_in_fn(phi, z) and an outer cylinder r_out,
    over phi in [-span/2, span/2] (a tube cut open lengthwise so the lumen is visible)."""
    ph = np.linspace(-phi_span / 2, phi_span / 2, nphi)
    z = np.linspace(0, length, nz)
    PH, Z = np.meshgrid(ph, z, indexing="ij")
    Ri = r_in_fn(PH, Z)
    inner = np.stack([Ri * np.cos(PH), Ri * np.sin(PH), Z], -1)
    outer = np.stack([r_out * np.cos(PH), r_out * np.sin(PH), Z], -1)
    V = np.vstack([inner.reshape(-1, 3), outer.reshape(-1, 3)])
    off = nphi * nz
    i, j = np.meshgrid(np.arange(nphi - 1), np.arange(nz - 1), indexing="ij")
    a = (i * nz + j).ravel()
    b, c, d = a + nz, a + nz + 1, a + 1
    F = [np.stack([a, c, b], 1), np.stack([a, d, c], 1), np.stack([a, b, c], 1) + off, np.stack([a, c, d], 1) + off]

    def wall(idx):
        t0, t1 = idx[:-1], idx[1:]
        return [np.stack([t0, t1, t1 + off], 1), np.stack([t0, t1 + off, t0 + off], 1)]
    for e in (np.arange(nz), (nphi - 1) * nz + np.arange(nz)[::-1], (np.arange(nphi) * nz)[::-1], np.arange(nphi) * nz + nz - 1):
        F += wall(e)
    F = np.vstack(F).astype(np.int64)
    if mesher.audit(V, F)["volume_mm3"] < 0:
        F = F[:, ::-1]
    return V, F


def _part(name, V, F, color, opacity=1.0):
    return {"name": name, "V": V, "F": F, "color": list(color), "opacity": opacity}


def _arrow(P0, P1, r):
    V, F = mesher.tube_mesh(np.linspace(P0, P1, 4), r, nsec=10)
    return V, F


# ---------------------------------------------------------------- levels
def detail_surface(p, rep, relief_x=1.0):
    lam = float(p["sf.texture_period_nm"]) * NM
    amp = float(p["sf.texture_amplitude_nm"]) * NM * relief_x
    t = float(p["sf.coating_nm"]) * NM
    W = max(5 * lam, 1.2e-3)
    n = 160
    g = np.linspace(-W / 2, W / 2, n)
    X, Y = np.meshgrid(g, g, indexing="ij")             # Y = local channel axis
    th = math.radians(float(p["sf.texture_orientation_deg"]))
    q = X * math.cos(th) - Y * math.sin(th)              # across the ridges (ridges run at th to the axis)
    kind = p["sf.texture"]
    if kind == "ridges":
        Z = amp * (0.5 + 0.5 * np.cos(2 * np.pi * q / lam))
    elif kind == "pillars":
        q2 = X * math.sin(th) + Y * math.cos(th)
        Z = amp * np.clip(np.cos(np.pi * q / lam) * np.cos(np.pi * q2 / lam), 0, None) ** 0.6
    else:
        Z = np.zeros_like(X)
    base = -max(0.25 * W, 3 * amp)
    parts = [_part("lumen wall (channel material)", *_grid_solid(Z, X, Y, base), _col(p["ch.wall_composition"]))]
    if p["sf.location"] != "none" and t > 0:
        parts.append(_part("coating (%.0f nm)" % (t / NM), *_sheet(Z + t, X, Y), _col(p["sf.coating_composition"]), 0.5))
    parts.append(_part("channel axis (local fiber path)", *_arrow(np.array([-0.45 * W, -0.45 * W, 2 * amp + t]),
                                                                  np.array([-0.45 * W, 0.45 * W, 2 * amp + t]), W / 120), [0.1, 0.45, 0.42]))
    note = ("Patch %.2f x %.2f um of a treated lumen surface (arrow = channel axis). Texture: %s, period %.0f nm, amplitude %.0f nm, "
            "%+.0f deg to the channel axis. Coating %.0f nm (translucent, %s). True scale inside the inset%s."
            % (W / UM, W / UM, kind, lam / NM, float(p["sf.texture_amplitude_nm"]), float(p["sf.texture_orientation_deg"]),
               t / NM, "in the geometry" if p["sf.coating_explicit"] == "explicit" else "metadata only",
               "; relief drawn x%g" % relief_x if relief_x != 1 else ""))
    return {"parts": parts, "window_mm": W, "note": note, "relief_x": relief_x}


def detail_micro(p, rep, relief_x=1.0):
    r_l = 0.5 * float(p["ch.lumen_um"]) * UM
    wall = float(p["ch.wall_um"]) * UM
    th = math.radians(float(p["gr.orientation_deg"]))
    per = (float(p["gr.width_um"]) + float(p["gr.ridge_um"])) * UM
    length = max(6 * per, 0.8 * r_l)
    placement = p["gr.placement"]
    parts = []

    def r_in(PH, Z):
        if placement not in ("channels", "both"):
            return np.full_like(PH, r_l)
        q = (r_l * PH) * math.cos(th) - Z * math.sin(th)
        return r_l + relief_x * _groove_depth(q, p)
    nphi = int(min(900, max(160, 10 * math.pi * r_l / max(per, 1e-6))))
    V, F = _half_tube_wall(r_in, r_l + wall, length, nphi=nphi, nz=80)
    parts.append(_part("channel wall (half, cut lengthwise)", V, F, _col(p["ch.wall_composition"])))
    lab = []
    if placement in ("inner_sheath", "both"):
        tw = float(p["sheath.wall_mm"])
        n = 220
        g = np.linspace(0, max(8 * per, 2.2 * r_l), n)
        X, Y = np.meshgrid(g, np.linspace(0, length, 80), indexing="ij")
        q = X * math.cos(th) - Y * math.sin(th)
        Zt = -relief_x * _groove_depth(q, p)
        Vs, Fs = _grid_solid(Zt, X + 1.6 * (r_l + wall), Y, -min(tw, 6 * float(p["gr.depth_um"]) * UM + 0.02))
        parts.append(_part("inner sheath wall (grooved face up)", Vs, Fs, _col(p["sheath.composition"])))
        lab.append("inner-sheath patch on the right (wall cut short for the view)")
    residual = float(p["ch.wall_um"]) - float(p["gr.depth_um"])
    note = ("Half of one channel, %.0f um lumen and %.0f um wall, %.0f um long; grooves %.1f um wide, %.1f um deep, ridges %.1f um, "
            "bottom radius %.1f um, %+.0f deg to the channel axis; %.1f um of wall remains under a groove. Braid strands are never grooved. %s"
            "True scale inside the inset%s." % (2 * r_l / UM, wall / UM, length / UM, float(p["gr.width_um"]), float(p["gr.depth_um"]),
                                             float(p["gr.ridge_um"]), float(p["gr.bottom_radius_um"]), float(p["gr.orientation_deg"]),
                                             residual, ("; ".join(lab) + ". ") if lab else "",
                                             "; relief drawn x%g" % relief_x if relief_x != 1 else ""))
    if placement == "none":
        note = "No grooved surfaces are selected (Level 2: Grooved surfaces = none). " + note
    return {"parts": parts, "window_mm": max(length, 2 * (r_l + wall)), "note": note, "relief_x": relief_x}


def detail_fascicle(p, rep, relief_x=1.0):
    lum = float(p["ch.lumen_um"]) * UM
    wall = float(p["ch.wall_um"]) * UM
    gap = float(p["ch.spacing_um"]) * UM
    asp = 1.0 if p["ch.shape"] == "circular" else float(p["ch.aspect"])
    od = lum + 2 * wall
    pitch = od * math.sqrt(asp) + gap
    length = 4 * od
    pos = [(0, 0)] + [(pitch * math.cos(a), pitch * math.sin(a)) for a in np.radians(np.arange(0, 360, 60))]
    exp = 4.0 if p["ch.shape"] == "rounded_square" else 2.0
    coat = float(p["sf.coating_nm"]) * NM if (p["sf.coating_explicit"] == "explicit" and p["sf.location"] in ("channels", "channels_and_inner_sheath")) else 0.0
    parts = []
    for k, (x, y) in enumerate(pos):
        P = np.stack([np.full(30, x), np.full(30, y), np.linspace(0, length, 30)], 1)
        U = np.tile([1.0, 0, 0], (30, 1))
        from acl_engine.v2.layered import hollow_tube_mesh
        V, F = hollow_tube_mesh(P, od / 2, lum / 2 - coat, U, nsec=40, aspect=asp, exponent=exp)
        if k == 0:                      # cut the centre channel open lengthwise
            keep = (V[F][:, :, 1].mean(1) <= 0)
            F = F[keep]
        parts.append(_part("channel %d" % k, V, F, _col(p["ch.wall_composition"]), 1.0 if k else 0.95))
    if p["ch.connections"] == "ports":
        dp = float(p["ch.port_diameter_um"]) * UM
        from acl_engine.v2.layered import hollow_tube_mesh
        P = np.stack([np.linspace(0, pos[1][0], 8), np.zeros(8), np.full(8, length / 2)], 1)
        V, F = hollow_tube_mesh(P, dp / 2 + wall, dp / 2, np.tile([0, 0, 1.0], (8, 1)), nsec=24)
        parts.append(_part("connection port", V, F, [0.95, 0.6, 0.3]))
    note = ("Seven neighbouring channels at true scale (the centre one cut open): lumen %.0f um%s, wall %.0f um, outside diameter %.0f um, "
            "spacing %.0f um, pitch %.0f um%s. Proposed lumen range 150-300 um is an adjustable proposal, not a validated requirement."
            % (lum / UM, " (open %.1f um after coating)" % ((lum - 2 * coat) / UM) if coat else "", wall / UM, od / UM, gap / UM, pitch / UM,
               "; a connection port joins the centre channel to its neighbour" if p["ch.connections"] == "ports" else ""))
    return {"parts": parts, "window_mm": 2 * pitch + od, "note": note, "relief_x": 1.0}


def detail_bundle(p, rep, region="am", relief_x=1.0):
    key = region if region in ("am", "pl") else "am"
    d = float(p["rf.%s.strand_diameter_mm" % key])
    alpha = math.radians(float(p["rf.%s.braid_angle_deg" % key]))
    N = int(p["rf.%s.strands" % key])
    m = max(1, N // 2)
    per = None
    try:
        per = float(next(r["value"] for r in rep["calculated"]["4"] if "pitch" in r["label"] and key.upper() in r["label"]))
    except Exception:
        per = None
    # unrolled patch: strands of one family are spaced s_perp along the circumference
    summ = ((rep or {}).get("summary") or {}).get("4", {})
    circ = float(summ.get(key + "_perimeter_mm", 2 * math.pi * 1.5))
    strand_len_rep = float(summ.get(key + "_strand_length_mm", 30.0))
    s_c = circ / m                                   # spacing along the circumference
    Wp = 3 * s_c
    Lp = min(3 * s_c / max(math.tan(alpha), 0.1), 2.5 * Wp)
    two = p["rf.%s.pattern" % key] == "2/2"
    parts = []
    n = 120
    z = np.linspace(0, Lp, n)
    for fam, sgn in (("+", 1), ("-", -1)):
        for i in range(-3, 7):
            u0 = i * s_c + (0 if fam == "+" else s_c / 2)
            u = u0 + sgn * z * math.tan(alpha)
            # over/under: alternates at each crossing with the other family
            ph = (np.pi / s_c) * (u0 + (z * math.tan(alpha)) * 2 * sgn) / (2 if two else 1)
            off = (0.5 * d) * np.clip((1 if fam == "+" else -1) * 1.3 * np.cos(ph + np.pi * i), -1, 1)
            keep = (u > -0.2 * Wp) & (u < 1.2 * Wp)
            if keep.sum() < 3:
                continue
            P = np.stack([u[keep], z[keep], off[keep]], 1)
            V, F = mesher.tube_mesh(P, 0.5 * d, nsec=16)
            parts.append(_part("strand %s%d" % (fam, i), V, F, _col(p["rf.%s.composition" % key], (0.9, 0.7, 0.3)), 0.55 if (fam == "+" and i == 1) else 1.0))
    # representative fibers inside one strand with the yarn twist (total over the full strand)
    twist = float(p["rf.%s.yarn_twist_deg" % key])
    strand_len = strand_len_rep
    u0 = 1 * s_c
    rate = math.radians(twist) / strand_len
    sel = (u0 + z * math.tan(alpha) > -0.2 * Wp) & (u0 + z * math.tan(alpha) < 1.2 * Wp)
    zz = z[sel]
    c = np.stack([u0 + zz * math.tan(alpha), zz, np.zeros_like(zz)], 1)
    T = np.array([math.sin(alpha), math.cos(alpha), 0.0])
    N1 = np.array([math.cos(alpha), -math.sin(alpha), 0.0])
    N2 = np.cross(T, N1)
    sl = zz / max(math.cos(alpha), 1e-6)
    for k in range(7):
        r = 0 if k == 0 else 0.32 * d
        a = 2 * np.pi * k / 6 + rate * sl
        P = c + r * (np.cos(a)[:, None] * N1 + np.sin(a)[:, None] * N2)
        V, F = mesher.tube_mesh(P, 0.06 * d, nsec=8)
        parts.append(_part("yarn fiber %d" % k, V, F, [0.25, 0.3, 0.35]))
    note = ("Unrolled patch of the %s-like braid (true scale): %d strands (%s), %.2f mm diameter, %.1f deg to the mantle axis; the "
            "translucent strand shows representative fibers with the yarn twist (%+.1f deg over a whole ~%.0f mm strand: %+.2f deg in "
            "this window). Braid angle, yarn twist and the ligament's own rotation are separate inputs."
            % (key.upper(), N, p["rf.%s.pattern" % key], d, math.degrees(alpha), twist, strand_len, math.degrees(rate * sl[-1]) if sl.size else 0))
    return {"parts": parts, "window_mm": max(Wp, Lp), "note": note, "relief_x": 1.0}


def detail_whole(p, rep, relief_x=1.0):
    tw = float(p["sheath.wall_mm"])
    spc = float(p["sheath.pore_spacing_mm"])
    dp = float(p["sheath.pore_diameter_um"]) * UM
    W = max(3.5 * spc, 2.0) if p["sheath.pores"] == "explicit" else 2.0
    n = 120
    g = np.linspace(0, W, n)
    X, Y = np.meshgrid(g, g, indexing="ij")
    Z = np.zeros_like(X)
    parts = [_part("sheath wall", *_grid_solid(Z, X, Y, -tw), _col(p["sheath.composition"]), 0.85)]
    cnt = 0
    if p["sheath.pores"] == "explicit":
        rows = np.arange(spc / 2, W, spc)
        for r, y in enumerate(rows):
            xs = np.arange(spc / 2 + (spc / 2 if (p["sheath.pore_pattern"] == "staggered" and r % 2) else 0), W, spc)
            for x in xs:
                V, F = mesher.tube_mesh(np.array([[x, y, 0.01], [x, y, -tw - 0.01]]), dp / 2, nsec=24)
                parts.append(_part("pore", V, F, [0.12, 0.15, 0.17]))
                cnt += 1
    open_frac = cnt * math.pi * dp * dp / 4 / (W * W)
    note = ("Patch %.1f x %.1f mm of the sheath wall (%.2f mm thick, outer face up)%s. Geometric open area %.1f%% of this patch. "
            "Pores are geometry only: the entered permeability (an assumption) is independent of them and no fluid transport is modelled."
            % (W, W, tw, (": %d through-wall pores of %.0f um, %.2f mm apart (%s)" % (cnt, dp / UM, spc, p["sheath.pore_pattern"]))
               if cnt else ": no explicit pores modelled", 100 * open_frac))
    return {"parts": parts, "window_mm": W, "note": note, "relief_x": 1.0}


def detail_ends(p, rep, region="tib", relief_x=1.0):
    side = region if region in ("tib", "fem") else "tib"
    k = "end.%s." % side
    ends = (rep or {}).get("ends", {}).get(side, {})
    Lf = float(ends.get("flare_length_mm", p[k + "flare_length_mm"]))
    base = ends.get("flare_base_mm", [p[k + "flare_width_mm"], p[k + "flare_thickness_mm"]])
    body = (rep or {}).get("sheath_mid_mm", [9.0, 4.8])
    Lb, Lg = float(p[k + "bone_zone_mm"]), float(p[k + "gradient_mm"])
    Ldom = max(Lf, Lb + Lg) + 1.5
    gam = float(p[k + "curvature"])
    nz = 90
    z = np.linspace(0, Ldom, nz)
    t = np.clip(1 - z / max(Lf, 1e-6), 0, 1)
    w = (t * t * (3 - 2 * t)) ** gam
    half = 0.5 * ((1 - w) * body[0] + w * float(base[0]))
    thick = 0.5 * ((1 - w) * body[1] + w * float(base[1]))
    from acl_engine.v2.layered import _closed_loft
    rings = []
    for i in range(nz):
        a = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        rings.append(np.stack([half[i] * np.cos(a), np.full(48, z[i]), thick[i] * np.sin(a)], 1))
    V, F = _closed_loft(np.stack(rings))
    # colour by the graded composition along the distance from bone
    tt = np.clip((V[:, 1] - Lb) / max(Lg, 1e-6), 0, 1)
    cb, ci, cl = (np.asarray(_col(p[k + "comp_" + zname])) for zname in ("bone", "interface", "ligament"))
    s2 = lambda x: x * x * (3 - 2 * x)
    col = np.where((tt < 0.5)[:, None], (1 - s2(2 * tt))[:, None] * cb + s2(2 * tt)[:, None] * ci,
                   (1 - s2(np.clip(2 * tt - 1, 0, 1)))[:, None] * ci + s2(np.clip(2 * tt - 1, 0, 1))[:, None] * cl)
    parts = [{"name": "flared end (graded)", "V": V, "F": F, "color": [1, 1, 1], "opacity": 1.0, "vcol": np.clip(col, 0, 1)}]
    Vb, Fb = _grid_solid(np.full((2, 2), -0.02), *np.meshgrid([-0.7 * max(half), 0.7 * max(half)], [-0.6 * max(thick), 0.6 * max(thick)], indexing="ij"), -0.8)
    Vb = Vb[:, [0, 2, 1]]
    parts.append(_part("bone surface", Vb, Fb, [0.9, 0.87, 0.8], 0.6))
    if p[k + "fixation"] == "plug":
        D, H = float(p[k + "plug_diameter_mm"]), float(p[k + "plug_depth_mm"])
        Vp, Fp = mesher.tube_mesh(np.array([[0, 0.2, 0], [0, -min(H, 6.0), 0]]), D / 2, nsec=32)
        parts.append(_part("fixation plug (cut at 6 mm)", Vp, Fp, [0.75, 0.7, 0.62], 0.8))
    stop = float(p[k + "endpoint_stop_mm"])
    note = ("%s end, bone at the bottom: flare %.1f mm from the %.1f x %.1f mm body section to the %.1f x %.1f mm base (transition curvature %.2f). "
            "Colour = graded composition: bone-facing %.1f mm, gradient %.1f mm (bone -> interface -> ligament-facing). Channels and braid "
            "end %.1f mm from bone. True scale." % ("Tibial" if side == "tib" else "Femoral", Lf, body[0], body[1], float(base[0]), float(base[1]),
                                                     gam, Lb, Lg, stop))
    return {"parts": parts, "window_mm": max(Ldom, max(half) * 2), "note": note, "relief_x": 1.0, "up": "y"}


def level_detail(p, level, region=None, rep=None, relief_x=1.0):
    fn = {1: detail_surface, 2: detail_micro, 3: detail_fascicle, 5: detail_whole}
    if level in fn:
        return fn[level](p, rep, relief_x)
    if level == 4:
        return detail_bundle(p, rep, region or "am", relief_x)
    if level == 6:
        return detail_ends(p, rep, region or "tib", relief_x)
    raise ValueError("level must be 1-6")
