"""Total rotation of fibers around their unit's local axis, measured on the
generated paths.

Surgeon-informed design constraint (as reported by the user): within each
cylindrical structural unit that contains fibers, the fibers rotate no more
than 30 degrees IN TOTAL around the unit's local longitudinal axis over the
unit's whole length (signed: -30 .. +30). It is a design rule for the repair
and the model, not an anatomical limit on whole-ACL torsion, knee rotation or
the AM/PL arrangement.

Units and what rotates inside them
---------------------------------
- native bundle (AM, PL)      fascicles (with their fibers) about the bundle path
- native fascicle             fibers about the fascicle centreline
- aligned scaffold repair unit  yarns about the unit's centre path
- scaffold yarn               fibers about the yarn centreline
- braided repair unit         yarns wind about the unit axis by construction
                              (reported separately; see the report)

Measurement: the unit axis is a sampled 3D curve; a rotation-minimizing frame
(double reflection) is carried along it, so the frame itself does not twist and
the unit's overall oblique orientation between its endpoints is not counted.
Each path point is expressed in that frame; the angle is unwrapped along the
path (never wraps by +-360). Net rotation = smoothed end angle - smoothed start
angle. Back-and-forth = how far the smoothed angle strays outside the interval
between its start and end values (a monotone path has 0).
"""
import math
import numpy as np
from scipy import ndimage

LIMIT_DEG = 30.0
TOL_DEG = 1.0          # measurement tolerance on the limit


def rmf_frames(C, ref=None):
    """Rotation-minimizing frames (T, U1, U2) along the curve C (n,3)."""
    from acl_engine.v2.ligament import rmf
    T = np.gradient(C, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    if ref is None:
        ref = np.array([0.0, 1.0, 0.0]) if abs(T[0, 1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    T, U1, U2 = rmf(C, np.asarray(ref, float))
    return T, U1, U2


def path_angles(P, C, U1, U2):
    """Unwrapped angle (rad) of path points P (n,3) around the axis samples C (n,3)."""
    Q = P - C
    a = np.arctan2((Q * U2).sum(1), (Q * U1).sum(1))
    return np.unwrap(a)


def summarize(ang, sig, smooth_mm=2.0):
    """Net (deg) and back-and-forth excursion (deg) of an unwrapped angle series."""
    if ang.size < 4:
        return 0.0, 0.0
    ds = max(float(np.median(np.abs(np.diff(sig)))), 1e-6)
    w = max(1, int(round(smooth_mm / ds)))
    a = ndimage.uniform_filter1d(ang, size=min(2 * (w // 2) + 1, ang.size if ang.size % 2 else ang.size - 1), mode="nearest")
    net = a[-1] - a[0]
    lo, hi = min(a[0], a[-1]), max(a[0], a[-1])
    exc = max(0.0, float(lo - a.min()), float(a.max() - hi))
    return math.degrees(net), math.degrees(exc)


def unit_rotation(paths, axis_fn, min_radius=0.15, smooth_mm=2.0, nres=60, ref_paths=None, unit_span=None, return_series=False):
    """Rigid rotation of ONE unit's contents about its axis, from the generated paths.

    paths: list of (P (n,3), sig (n,)); axis_fn(sig) -> axis points. Each path's
    angle is unwrapped along the path (never wraps), taken relative to its start,
    and resampled on its normalized length; the unit rotation is the mean over
    paths (zero-mean scatter such as orientation dispersion cancels). End values
    are read directly (dispersion is zero at both ends by construction); the
    back-and-forth excursion is read on a lightly smoothed series.

    ref_paths: optional parent-only paths (same order): their angles are
    subtracted, so only the unit's OWN rotation is measured.
    unit_span: if given, a path covering only part of the unit is scaled by
    span / its length (its share of a rotation spread over the unit).
    Returns (unit_net_deg, unit_excursion_deg, per_path_net_deg array)."""
    series, nets = [], []
    frame_cache = {}
    grid = np.linspace(0, 1, nres)
    for k, (P, sig) in enumerate(paths):
        if P.shape[0] < 6:
            continue
        C = axis_fn(sig)
        rr = np.linalg.norm(P - C, axis=1)
        if rr.mean() < min_radius or rr.min() < 0.35 * rr.mean():
            continue                       # too close to the axis for a meaningful angle
        key = (id(C), C.shape[0], float(C[0, 0]), float(C[-1, 2]))
        if key not in frame_cache:
            frame_cache[key] = rmf_frames(C)
        T, U1, U2 = frame_cache[key]
        a = path_angles(P, C, U1, U2)
        a = a - a[0]
        if ref_paths is not None:
            ar = path_angles(ref_paths[k][0], C, U1, U2)
            a = a - (ar - ar[0])
        scale = 1.0
        if unit_span:
            frac = (sig[-1] - sig[0]) / unit_span
            if frac < 0.3:
                continue
            scale = 1.0 / frac
        t = (sig - sig[0]) / max(sig[-1] - sig[0], 1e-9)
        series.append(np.interp(grid, t, a) * scale)
        nets.append(math.degrees(a[-1]) * scale)
    if not series:
        return (None, None, np.zeros(0), None) if return_series else (None, None, np.zeros(0))
    mean = np.mean(series, axis=0)
    span = unit_span or np.mean([sig[-1] - sig[0] for _, sig in paths if sig.size > 1])
    _, exc = summarize(mean, grid * span, smooth_mm)
    if return_series:
        return math.degrees(mean[-1] - mean[0]), exc, np.array(nets), mean
    return math.degrees(mean[-1] - mean[0]), exc, np.array(nets)


def _row(level, unit, requested, unit_nets, unit_excs, path_nets, extra=None):
    unit_nets = np.asarray([x for x in unit_nets if x is not None], float)
    unit_excs = np.asarray([x for x in unit_excs if x is not None], float)
    if unit_nets.size == 0:
        return {"level": level, "unit": unit, "requested_deg": requested, "n": 0}
    k = int(np.argmax(np.abs(unit_nets)))
    pn = np.asarray(path_nets, float)
    row = {"level": level, "unit": unit, "requested_deg": requested, "n": int(unit_nets.size),
           "net_mean_deg": float(unit_nets.mean()), "net_max_abs_deg": float(np.abs(unit_nets).max()),
           "net_at_max_deg": float(unit_nets[k]), "excursion_max_deg": float(unit_excs.max()) if unit_excs.size else 0.0,
           "paths": int(pn.size), "path_spread_deg": float(np.abs(pn - unit_nets.mean()).max()) if pn.size else 0.0,
           "within_limit": bool(np.abs(unit_nets).max() <= LIMIT_DEG + TOL_DEG)}
    if extra:
        row.update(extra)
    return row


def region_drift_deg(lig, b, s, smooth_mm=2.0):
    """Rotation (deg) the bundle REGION itself makes about its own path in a
    non-twisting frame (the ligament's section orientation and the AM/PL
    arrangement): the parent motion, which the within-bundle rule does not count."""
    cen, _ = lig.bundle_geometry(b)
    ph = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    rb = lig.bundle_boundary_radius(cen, ph, b)
    ring = cen + (0.6 * rb)[:, None] * np.stack([np.cos(ph), np.sin(ph)], 1)
    P = lig.map(np.repeat(s, ring.shape[0]), np.tile(ring, (s.size, 1))).reshape(s.size, ring.shape[0], 3)
    Cb = lig.bundle_centerline(b, s)
    net, _, _, series = unit_rotation([(P[:, k], s * lig.L) for k in range(ring.shape[0])], lambda sg: Cb, 0.0, smooth_mm,
                                      return_series=True)
    return (net or 0.0), series


def evaluate(p, lig, nat, fasc, scf=None, yarns=None, scaffold_fibers=None, native_fibers=None):
    """Measured total rotation for every unit level present in the build."""
    rows = []
    L = lig.L
    req = nat.rotation_requested_deg
    # native bundles: the bundle's fascicles about the bundle path, RELATIVE to the parent:
    # each fascicle is compared with its parent-only path (same material position carried
    # by the section map, without the within-bundle rotation)
    s_axis = fasc[0]["s"] if fasc else lig.s
    for b, bn in ((0, "AM"), (1, "PL")):
        Cb = lig.bundle_centerline(b, s_axis)
        fb = [fa for fa in fasc if fa["bundle"] == b]
        paths = [(fa["P"], fa["s"] * L) for fa in fb]
        refs = [(lig.map(fa["s"], np.repeat(nat.fasc_m0[fa["id"]][None], fa["s"].size, 0)), fa["s"] * L) for fa in fb]
        net, exc, pn = unit_rotation(paths, lambda sg, Cb=Cb: Cb, 0.4, 2.0, ref_paths=refs)
        parent, _, _ = unit_rotation(refs, lambda sg, Cb=Cb: Cb, 0.4, 2.0)
        rows.append(_row("native bundle: fascicles about the bundle path (relative to the bundle region)", bn, req["bundle"][b],
                         [net], [exc], pn, {"region_turn_deg": parent}))
    # native fascicles: each fascicle's fibers about its own centreline (generated paths, true scale)
    fib = native_fibers if native_fibers is not None else nat.fibers(fasc, 7, 1.0)
    for b, bn in ((0, "AM"), (1, "PL")):
        un, ue, pn = [], [], []
        for fr in fib:
            if fr["bundle"] != b:
                continue
            C = fr["C"]
            sig = fr["s"] * L
            n, e, pp = unit_rotation([(fr["P"][k], sig) for k in range(fr["P"].shape[0])], lambda sg, C=C: C,
                                     0.3 * fr["r_fasc"], 4.5)
            un.append(n)
            ue.append(e)
            pn.extend(pp.tolist())
        rows.append(_row("native fascicle: fibers about the fascicle axis", "%s fascicles" % bn, req["fiber"][b], un, ue, pn))
    if scf is not None and getattr(scf, "active", False) and yarns:
        req_y = float(p["scaffold.yarn_rotation_deg"])
        for u in scf.units:
            uy = [y for y in yarns if y["unit"] == u.id]
            md = getattr(u, "axis_model", None)
            if md is not None:
                axis = (lambda sig, md=md: scf._to3d(sig, np.stack([np.interp(sig, md["sig"], md["c"][:, a]) for a in range(2)], 1), blend=False))
            else:
                axis = (lambda sig, u=u: lig.map(sig / L, np.repeat(u.dom_c[None], sig.size, 0)))
            for fam, lvl, rq in ((("axial",), "scaffold repair unit: yarns about the unit axis", req_y),
                                 (("core",), "scaffold repair unit: core yarns about the unit axis", 0.0)):
                yy = [y for y in uy if y["family"] in fam]
                if yy:
                    span = (md["span"] if md is not None else (u.sig_b - u.sig_a))
                    n, e, pn = unit_rotation([(y["P"], y["sig"]) for y in yy], axis, 0.25, 2.0, unit_span=span)
                    # the hard rule on the ACTUAL paths (a trimmed yarn turns by its share only)
                    _, _, raw = unit_rotation([(y["P"], y["sig"]) for y in yy], axis, 0.25, 2.0)
                    rows.append(_row(lvl, u.label, rq, [n], [e], pn,
                                     {"actual_path_max_abs_deg": float(np.abs(raw).max()) if raw.size else 0.0}))
            braid = [y for y in uy if y["family"] in ("+", "-")]
            if braid:
                nets = []
                for y in braid:
                    _, _, pn = unit_rotation([(y["P"], y["sig"])], axis, 0.25, 0.6)
                    nets.extend(pn.tolist())
                nets = np.array(nets)
                if nets.size:
                    rows.append({"level": "braided repair unit: braid yarns wind about the unit axis", "unit": u.label, "requested_deg": None,
                                 "n": int(nets.size), "net_max_abs_deg": float(np.abs(nets).max()), "net_at_max_deg": float(nets[np.argmax(np.abs(nets))]),
                                 "net_mean_deg": float(nets.mean()), "excursion_max_deg": 0.0, "braid_winding": True,
                                 "within_limit": bool(np.abs(nets).max() <= LIMIT_DEG + TOL_DEG)})
        if scaffold_fibers:
            ymap = {y["id"]: y for y in yarns}
            un, ue, pn = [], [], []
            for fr in scaffold_fibers:
                y = ymap.get(fr["yarn"])
                if y is None:
                    continue
                C = fr["C"]
                n, e, pp = unit_rotation([(fr["P"][k], fr["sig"]) for k in range(fr["P"].shape[0])], lambda sg, C=C: C,
                                         0.3 * float(np.mean(y["r"])), 1.5)
                un.append(n)
                ue.append(e)
                pn.extend(pp.tolist())
            rows.append(_row("scaffold yarn: fibers about the yarn axis", "every yarn", float(p["scaffold.fiber_rotation_deg"]), un, ue, pn))
    return {"limit_deg": LIMIT_DEG, "tolerance_deg": TOL_DEG, "rows": rows,
            "note": "surgeon-informed design range for fibers within each cylindrical unit (as reported by the user); "
                    "not a limit on whole-ACL torsion, knee rotation or the AM/PL arrangement. Each unit's rotation is the "
                    "rigid turn shared by its fibers (mean over its generated paths); path_spread is how far single paths "
                    "deviate from it (orientation dispersion, crimp, section shape changes)."}
