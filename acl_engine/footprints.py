"""Milestone A footprints: tibial oval + femoral crescent (new module).

Thin closed solids, built vectorized. The tibial oval (11 x 17 mm) matches
the schematic coverage constants in cli.py, so tibial coverage numbers carry
over; the femoral crescent is new (numerical coverage). Sources L14/L15/L04.
"""
import math
import numpy as np

TIBIAL_OVAL_ML = 11.0
TIBIAL_OVAL_AP = 17.0
TIBIAL_THICK = 0.4
FEMORAL_R_OUT = 9.0
FEMORAL_R_IN = 4.5
FEMORAL_SPAN_DEG = 210.0
FEMORAL_THICK = 0.4
# The annulus centre sits this far ANTERIOR of the femoral ACL centre so the
# crescent band (not the hole) is centred on the attachment (L14).
FEMORAL_ANNULUS_OFFSET = (FEMORAL_R_OUT + FEMORAL_R_IN) / 2.0


def _grid_faces(nr, na, closed_loop=True):
    ii = np.arange(nr - 1)[:, None]
    jj = np.arange(na)[None, :]
    j2 = (jj + 1) % na if closed_loop else jj + 1
    if not closed_loop:
        jj = jj[:, :-1]
        j2 = j2[:, :-1]
    a = ii * na + jj
    b = ii * na + j2
    c = (ii + 1) * na + jj
    d = (ii + 1) * na + j2
    return np.stack([np.stack([a, b, c], axis=-1),
                     np.stack([b, d, c], axis=-1)], axis=2).reshape(-1, 3)


def _orient_outward(V, F):
    from acl_engine.sweep import signed_volume
    if signed_volume(V, F) < 0:
        return V, F[:, [0, 2, 1]]
    return V, F


def tibial_footprint_mesh(T, nrad=12, nang=64):
    """Broad oval on the plateau (z in [0, thick]), centred at the tibial
    ACL centre, between the tibial spines. Closed thin solid."""
    cx, cy = float(T[0]), float(T[1])
    r = np.linspace(0.0, 1.0, nrad + 1)[1:]
    ph = np.linspace(0.0, 2.0 * math.pi, nang, endpoint=False)
    rr, pp = np.meshgrid(r, ph, indexing="ij")
    x = cx + (TIBIAL_OVAL_ML / 2.0) * rr * np.cos(pp)
    y = cy + (TIBIAL_OVAL_AP / 2.0) * rr * np.sin(pp)
    top = np.stack([x, y, np.full_like(x, TIBIAL_THICK)], axis=-1)
    bot = np.stack([x, y, np.zeros_like(x)], axis=-1)
    Vc = np.array([[cx, cy, TIBIAL_THICK], [cx, cy, 0.0]])
    V = np.concatenate([Vc, top.reshape(-1, 3), bot.reshape(-1, 3)])
    n = nrad * nang
    Ft = _grid_faces(nrad, nang) + 2
    Fb = _grid_faces(nrad, nang)[:, [0, 2, 1]] + 2 + n
    ring0 = 2 + np.arange(nang)
    ring0n = 2 + (np.arange(nang) + 1) % nang
    cap_t = np.stack([np.full(nang, 0), ring0n, ring0], axis=1)
    bring0 = 2 + n + np.arange(nang)
    bring0n = 2 + n + (np.arange(nang) + 1) % nang
    cap_b = np.stack([np.full(nang, 1), bring0, bring0n], axis=1)
    outer_t = 2 + (nrad - 1) * nang + np.arange(nang)
    outer_t2 = 2 + (nrad - 1) * nang + (np.arange(nang) + 1) % nang
    outer_b = 2 + n + (nrad - 1) * nang + np.arange(nang)
    outer_b2 = 2 + n + (nrad - 1) * nang + (np.arange(nang) + 1) % nang
    wall = np.concatenate([np.stack([outer_t, outer_t2, outer_b], axis=1),
                           np.stack([outer_t2, outer_b2, outer_b], axis=1)])
    F = np.concatenate([Ft, Fb, cap_t, cap_b, wall]).astype(np.int64)
    return _orient_outward(V, F)


def femoral_wall_x(Ffem, side="right"):
    """Inner face of the lateral condyle (matches the condyle placement in
    cli.py: centre Ffem[0] +/- 11.5, x-radius 12)."""
    return float(Ffem[0]) - 0.5 if side == "right" else float(Ffem[0]) + 0.5


def femoral_footprint_mesh(Ffem, side="right", nrad=6, nang=72):
    """Crescent (annular sector, 210 deg) on the lateral wall of the notch,
    its band centred on the femoral ACL centre, wrapping the posterior
    margin. Closed thin solid proud of the wall toward the notch."""
    cy, cz = float(Ffem[1]) + FEMORAL_ANNULUS_OFFSET, float(Ffem[2])
    xw = femoral_wall_x(Ffem, side)
    # slab proud of the wall toward the notch; lo/hi ordered so the winding
    # derivation is identical for both sides (uniformly flipped at the end)
    if side == "right":
        x0, x1 = xw - FEMORAL_THICK, xw
    else:
        x0, x1 = xw, xw + FEMORAL_THICK
    span = math.radians(FEMORAL_SPAN_DEG)
    r = np.linspace(FEMORAL_R_IN, FEMORAL_R_OUT, nrad)
    ph = np.linspace(-span / 2.0, span / 2.0, nang)
    rr, pp = np.meshgrid(r, ph, indexing="ij")
    # angle measured from posterior (-y) toward proximal (+z)
    y = cy - rr * np.cos(pp)
    z = cz + rr * np.sin(pp)
    lay0 = np.stack([np.full_like(y, x0), y, z], axis=-1)
    lay1 = np.stack([np.full_like(y, x1), y, z], axis=-1)
    V = np.concatenate([lay0.reshape(-1, 3), lay1.reshape(-1, 3)])
    n = nrad * nang
    F = [_grid_faces(nrad, nang, closed_loop=False),
         _grid_faces(nrad, nang, closed_loop=False)[:, [0, 2, 1]] + n]
    # four rim walls: inner, outer, and the two span ends
    i0 = np.arange(nang)
    i1 = n + np.arange(nang)
    F.append(np.concatenate(
        [np.stack([i0[:-1], i1[:-1], i0[1:]], axis=1),
         np.stack([i0[1:], i1[:-1], i1[1:]], axis=1)]))
    base = (nrad - 1) * nang
    o0 = base + np.arange(nang)
    o1 = n + base + np.arange(nang)
    F.append(np.concatenate(
        [np.stack([o0[:-1], o0[1:], o1[:-1]], axis=1),
         np.stack([o0[1:], o1[1:], o1[:-1]], axis=1)]))
    for e in (0, nang - 1):
        e0 = e + np.arange(nrad) * nang
        e1 = n + e + np.arange(nrad) * nang
        if e == 0:
            F.append(np.concatenate(
                [np.stack([e0[:-1], e0[1:], e1[:-1]], axis=1),
                 np.stack([e0[1:], e1[1:], e1[:-1]], axis=1)]))
        else:
            F.append(np.concatenate(
                [np.stack([e0[:-1], e1[:-1], e0[1:]], axis=1),
                 np.stack([e0[1:], e1[:-1], e1[1:]], axis=1)]))
    return _orient_outward(V, np.concatenate(F).astype(np.int64))


def femoral_crescent_coverage(Ws_end, Ts_end, Ffem, side="right", grid=201):
    """Fraction of the femoral crescent area under the scaffold end ellipse.

    Documented simplification (same spirit as context.footprint_coverage):
    axis-aligned comparison in the wall plane, scaffold y-radius Ts_end/2,
    z-radius Ws_end/2, centred at the femoral ACL centre; the crescent
    annulus is centred FEMORAL_ANNULUS_OFFSET anterior of it. Vectorized."""
    ay, az = float(Ffem[1]), float(Ffem[2])
    cy, cz = ay + FEMORAL_ANNULUS_OFFSET, az
    span = math.radians(FEMORAL_SPAN_DEG)
    e = FEMORAL_R_OUT + 1.0
    yy, zz = np.meshgrid(np.linspace(cy - e, cy + e, grid),
                         np.linspace(cz - e, cz + e, grid))
    post = -(yy - cy)
    prox = zz - cz
    rr = np.sqrt(post * post + prox * prox)
    ang = np.arctan2(prox, post)
    in_crescent = (rr >= FEMORAL_R_IN) & (rr <= FEMORAL_R_OUT) \
        & (np.abs(ang) <= span / 2.0)
    in_scaffold = ((yy - ay) / (float(Ts_end) / 2.0)) ** 2 \
        + ((zz - az) / (float(Ws_end) / 2.0)) ** 2 <= 1.0
    n = int(in_crescent.sum())
    if n == 0:
        return 0.0
    return float((in_crescent & in_scaffold).sum()) / float(n)
