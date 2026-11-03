"""Repair scaffold fitted inside the repair space.

Repair space
------------
The scaffold is generated per REPAIR UNIT. A unit is the affected part of the
section that one scaffold fills: the whole section (complete tear), one bundle
(AM-only / PL-only tears, or per-bundle repair), a sector, or a voxel-mask
region. Along the ligament a unit spans the damaged interval plus any
explicitly permitted interface overlap with the remnants; it reaches bone only
when a remnant is shorter than the bone-reach limit.

At stations every ~0.25 mm along the unit, an AVAILABLE-DISTANCE field D (mm)
is rasterized in the true section (the same contour the viewer and exports
use). D is the distance to the nearest constraint, minus that constraint's
clearance:

- the target ligament boundary (contour)                 - boundary clearance
- the AM/PL interface (single-bundle units)               - interface clearance
- the affected-region edge (sector / mask / depth)        - boundary clearance
- the cross-sectional extent (fill fraction)              - boundary clearance
- retained fascicles (and continuous fascicles in the gap) - yarn clearance
- bone                                                    - yarn clearance

A strand of radius r fits at X when D(X) >= r. Every layout uses this test at
EVERY station of its span, so a design that fits at mid-length but not near
an interface is not accepted. Nothing is shrunk, thinned or removed to force a
fit: when the requested architecture does not fit, the build reports it with
explicit resolutions (parameter values that would fit) for the user to apply.

Architectures
-------------
- aligned: fascicle-like yarns at fixed section (material) coordinates.
  With 'interdigitate' they continue into the remnants between retained
  fascicles for the overlap length; with 'abut' they end at the local tear face.
- braided: one or more concentric braid layers. Each layer is a tube at the
  level set D = r + over-under depth (+ layer pitch), so the outermost strand
  surface sits at the clearance from the boundary. Crossing yarns are
  separated by a flat-topped over-under offset along the tube's true normal.
  Axial core yarns fill the inside of the innermost layer. Braids always end
  at the tear faces (a tube cannot pass between retained fascicles).

Regions (multiphasic) along each unit: tibial_interface | bridge |
femoral_interface, split at the unit's mean tear faces; bone interfaces only
where the unit reaches bone. Material identity is metadata (no material model).
"""
import math
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from acl_engine.v2.design import braid_dependents, normalize_composition, REGION_IDS, MATERIALS
from acl_engine.v2.ligament import rot2, sunflower, fiber_offsets, _sstep, layout_points

BUNDLE_NAMES = ["AM", "PL"]


# ---------------------------------------------------------------- rasters
def scanline_mask(poly, xs, ys):
    """Even-odd fill of a closed polygon on pixel centres -> bool (nx, ny)."""
    x0, y0 = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    Y = ys[:, None]
    cond = (y0[None] > Y) != (y1[None] > Y)
    xc = x0[None] + (Y - y0[None]) * (x1 - x0)[None] / np.where(np.abs(y1 - y0) < 1e-30, 1e-30, (y1 - y0))[None]
    xc = np.where(cond, xc, np.inf)
    xc.sort(1)
    m = int(min(xc.shape[1], max(2, np.isfinite(xc).sum(1).max())))
    xc = xc[:, :m]
    cnt = (xc[:, None, :] < xs[None, :, None]).sum(2)
    return (cnt % 2 == 1).T


def signed_edt(mask, h):
    """Signed distance (mm) of pixel centres to the mask outline, + inside."""
    din = ndimage.distance_transform_edt(mask) * h
    dout = ndimage.distance_transform_edt(~mask) * h
    return np.where(mask, din - 0.5 * h, -(dout - 0.5 * h))


class Field:
    """Available distance D (mm) on a station raster: D >= r means a strand of radius r fits."""

    def __init__(self, sig, s, xs, ys, h, D, Danat):
        self.sig, self.s, self.xs, self.ys, self.h = sig, s, xs, ys, h
        self.D = D              # all constraints incl. every retained fascicle (interdigitation)
        self.Danat = Danat      # anatomical repair region only (no clearances): repair-space area
        self.Dbound = D         # boundary/interface/region/extent/bone constraints with clearances, no tissue
        self.Dcont = D          # Dbound + fascicles that stay continuous through the gap

    def sample(self, X, which="D"):
        A = getattr(self, which)
        i = (X[:, 0] - self.xs[0]) / self.h
        j = (X[:, 1] - self.ys[0]) / self.h
        return ndimage.map_coordinates(A, [i, j], order=1, mode="constant", cval=-5.0)

    def grid(self):
        GX, GY = np.meshgrid(self.xs, self.ys, indexing="ij")
        return np.stack([GX.ravel(), GY.ravel()], 1)

    def area(self, which="D", r=0.0):
        return float((getattr(self, which) >= r).sum()) * self.h * self.h


# ---------------------------------------------------------------- units
class RepairUnit:
    def __init__(self, uid, label, region_fn, iface_side, restricted):
        self.id = uid
        self.label = label
        self.region_fn = region_fn          # material predicate (affected part of the section)
        self.iface_side = iface_side        # 0 AM / 1 PL when kept to one bundle, else None
        self.restricted = restricted        # region is smaller than the bundle / section
        self.fields = []
        self.messages = []


class Scaffold:
    def __init__(self, p, lig, injury, tissue, fascicles, quality="full", bones=None):
        self.p, self.lig, self.inj, self.tis = p, lig, injury, tissue
        self.fascicles = fascicles
        self.bones = bones or []
        self.quality = quality
        # the gap-filling yarn scaffold; the six-level construct is acl_engine.v2.layered
        self.active = injury.active and p["scaffold.arch"] != "layered"
        self.messages = []
        self.resolutions = []
        self.units = []
        self.braid_info = None
        self.fan_info = {}
        if not self.active:
            return
        self.d0 = float(p["scaffold.yarn_diameter_mm"])
        self.clr = float(p["scaffold.clearance_mm"])
        self.c_env = float(p["fit.boundary_clearance_mm"])
        self.c_if = float(p["fit.interface_clearance_mm"])
        self.fill = float(p["scaffold.fill_fraction"])
        self.k = math.sqrt(self.fill)
        self.engagement = p["scaffold.engagement"]
        self.arch = p["scaffold.arch"]
        self.reg = {}
        for rid in REGION_IDS:
            g = lambda f, rid=rid: p["regions.%s.%s" % (rid, f)]
            self.reg[rid] = {"material": g("material"), "composition": normalize_composition(g("composition")),
                             "vf": float(g("fiber_volume_fraction")), "disp": float(g("alignment_dispersion_deg")),
                             "scale": float(g("strand_scale")), "tw": float(g("transition_mm")),
                             "mode": g("transition_mode")}
        self.dmax = self.d0 * max(self.reg[r]["scale"] for r in REGION_IDS)
        self.h = 0.05 if quality == "full" else 0.09
        self.step = 0.25 if quality == "full" else 0.6
        for u in self._make_units():
            if self._setup_unit(u):
                self.units.append(u)
        if not self.units:
            self.active = False
            self.messages.append({"param": "injury.involvement", "level": "fail",
                                  "message": "the injury leaves no repairable region"})

    # ------------------------------------------------------------ units
    def _make_units(self):
        p, lig, inj = self.p, self.lig, self.inj
        per_bundle = p["fit.partition"] == "per_bundle" or inj.inv == "per_bundle"
        M = sunflower(3000)
        inr = inj.in_region(M)
        b = lig.bundle_of(M)
        units = []
        only = {"AM": 0, "PL": 1}.get(inj.inv)
        if only is not None:
            # AM-only / PL-only tear: one unit, kept inside that bundle (a mask straying
            # across the interface does not open a repair in the unaffected bundle)
            sel = inr & (b == only)
            if sel.sum() < 15:
                return []
            restricted = bool(sel.sum() < 0.97 * (b == only).sum())
            fn = (lambda m, k=only: self.inj.in_region(m) & (self.lig.bundle_of(m) == k))
            return [RepairUnit(BUNDLE_NAMES[only], "%s repair" % BUNDLE_NAMES[only], fn, only, restricted)]
        if per_bundle:
            for k in (0, 1):
                sel = inr & (b == k)
                if sel.sum() < 15:
                    continue
                restricted = bool(sel.sum() < 0.97 * (b == k).sum())
                fn = (lambda m, k=k: self.inj.in_region(m) & (self.lig.bundle_of(m) == k))
                units.append(RepairUnit(BUNDLE_NAMES[k], "%s repair" % BUNDLE_NAMES[k], fn, k, restricted))
        else:
            if inr.sum() < 15:
                return []
            restricted = bool(inr.sum() < 0.97 * M.shape[0])
            side = None
            if inr.any() and (b[inr] == 0).all():
                side = 0
            elif inr.any() and (b[inr] == 1).all():
                side = 1
            uid = {"whole": "whole", "sector": "sector"}.get(inj.inv, "region")
            if inj.mask is not None:
                uid = "mask"
            units.append(RepairUnit(uid, {"whole": "Whole-section repair", "sector": "Sector repair",
                                          "mask": "Mask-region repair"}.get(uid, "Region repair"),
                                    lambda m: self.inj.in_region(m), side, restricted))
        return units

    def _setup_unit(self, u):
        p, lig, inj = self.p, self.lig, self.inj
        L = lig.L
        ext = inj.region_extent(sel_fn=lambda M, u=u: u.region_fn(M))
        if ext is None:
            return False
        u.ext = ext
        key = {"AM": "am", "PL": "pl"}.get(u.id)
        if p["fit.partition"] == "per_bundle" and key:
            ov_t = float(p["fit.%s.overlap_tib_mm" % key])
            ov_f = float(p["fit.%s.overlap_fem_mm" % key])
        else:
            ov_t = float(p["scaffold.overlap_tib_mm"])
            ov_f = float(p["scaffold.overlap_fem_mm"])
        u.strength = float(p["fit.%s.strength" % key]) if key else 0.5 * (float(p["fit.am.strength"]) + float(p["fit.pl.strength"]))
        reach = float(p["scaffold.bone_reach_mm"])
        u.bone_tib = bool(ext["lo_min"] < reach)
        u.bone_fem = bool((L - ext["hi_max"]) < reach)
        if self.engagement == "abut":
            ov_t = ov_f = 0.0
        fl_t = float(p["lig.flare_tib_mm"])
        fl_f = float(p["lig.flare_fem_mm"])
        # permitted span: damaged interval + permitted overlap (never into an insertion
        # flare unless the unit reaches bone)
        a = 0.0 if u.bone_tib else max(fl_t, ext["lo_min"] - ov_t)
        b = L if u.bone_fem else min(L - fl_f, ext["hi_max"] + ov_f)
        if not u.bone_tib and ext["lo_min"] - ov_t < fl_t and ov_t > 0:
            u.messages.append({"param": "scaffold.overlap_tib_mm", "level": "warn",
                               "message": "%s: tibial overlap limited to %.1f mm (the remnant enters its insertion zone)" % (u.label, max(0.0, ext["lo_min"] - a))})
        if not u.bone_fem and ext["hi_max"] + ov_f > L - fl_f and ov_f > 0:
            u.messages.append({"param": "scaffold.overlap_fem_mm", "level": "warn",
                               "message": "%s: femoral overlap limited to %.1f mm (the remnant enters its insertion zone)" % (u.label, max(0.0, b - ext["hi_max"]))})
        u.ov_t, u.ov_f = ov_t, ov_f
        u.sig_a, u.sig_b = a, b
        # braid span (gap only, or to bone)
        u.braid_a = 0.0 if u.bone_tib else max(a, ext["lo_min"])
        u.braid_b = L if u.bone_fem else min(b, ext["hi_max"])
        u.b1, u.b2 = ext["lo_mean"], ext["hi_mean"]
        u.region_kind = {"tibial_interface": "bone_interface" if u.bone_tib else "tissue_interface", "bridge": "bridge",
                         "femoral_interface": "bone_interface" if u.bone_fem else "tissue_interface"}
        # material centroid of the unit region
        M = sunflower(3000)
        Md = M[u.region_fn(M)]
        u.dom_c = Md.mean(0) if Md.shape[0] else np.zeros(2)
        u.sig = np.unique(np.concatenate([np.arange(a, b, self.step), [b]]))
        self._fields(u)
        return True

    # ------------------------------------------------------------ fields
    def _obstacles(self, u, s, i_st):
        """Retained fascicle circles at a station: (centres (k,2), radii (k,), continuous flag (k,))."""
        lig = self.lig
        if not self.fascicles:
            return np.zeros((0, 2)), np.zeros(0), np.zeros(0, bool)
        L = lig.L
        sig = s * L
        ms, rs, cont = [], [], []
        for fa in self.fascicles:
            if fa["damaged"][i_st]:
                continue
            # remnants shorter than the bone-reach limit are occupied by the repair (debrided)
            if u.bone_tib and sig < u.ext["lo_max"] and fa["affected"] and not fa["bridging"]:
                continue
            if u.bone_fem and sig > u.ext["hi_min"] and fa["affected"] and not fa["bridging"]:
                continue
            ms.append(fa["m"][i_st])
            rs.append(fa["r"][i_st])
            cont.append((not fa["affected"]) or fa["bridging"])
        if not ms:
            return np.zeros((0, 2)), np.zeros(0), np.zeros(0, bool)
        X = lig.material_to_metric(np.full(len(ms), s), np.array(ms))
        return X, np.array(rs), np.array(cont)

    def _fields(self, u):
        lig, p = self.lig, self.p
        L = lig.L
        h = self.h
        u.fields = []
        near_bone = None
        M_ref = sunflower(5000 if self.quality == "full" else 1200)
        tu = lig.material_tu(M_ref)
        reg_in = u.region_fn(M_ref) if u.restricted else None
        for sig in u.sig:
            s = sig / L
            i_st = int(round(np.clip(s, 0, 1) * (lig.s.size - 1)))
            poly = lig.contour_at(s)
            lo = poly.min(0) - 0.35
            hi = poly.max(0) + 0.35
            xs = np.arange(lo[0], hi[0] + h, h)
            ys = np.arange(lo[1], hi[1] + h, h)
            inside = scanline_mask(poly, xs, ys)
            Danat = signed_edt(inside, h)
            GX, GY = np.meshgrid(xs, ys, indexing="ij")
            G = np.stack([GX.ravel(), GY.ravel()], 1)
            if u.iface_side is not None:
                sg = 1.0 if u.iface_side == 0 else -1.0
                di = sg * lig.interface_distance(np.full(G.shape[0], s), G).reshape(GX.shape)
                Danat = np.minimum(Danat, di)
                D = np.minimum(Danat - self.c_env, di - self.c_if)
                D = np.minimum(D, Danat - self.c_env)
            else:
                D = Danat - self.c_env
            if u.restricted:
                Xr = lig._metric_from_tu(np.full(M_ref.shape[0], s), *tu)
                tree = cKDTree(Xr)
                sel = inside.ravel()
                rm = np.zeros(G.shape[0], bool)
                if sel.any():
                    _, j = tree.query(G[sel])
                    rm[sel] = reg_in[j]
                dr = signed_edt(rm.reshape(GX.shape), h)
                Danat = np.minimum(Danat, dr)
                D = np.minimum(D, dr - self.c_env)
            if self.fill < 0.999:
                ok = Danat >= 0
                if ok.any():
                    c = G[ok.ravel()].mean(0)
                    Q = c + (G - c) / self.k
                    i = (Q[:, 0] - xs[0]) / h
                    j = (Q[:, 1] - ys[0]) / h
                    de = self.k * ndimage.map_coordinates(Danat, [i, j], order=1, mode="constant", cval=-5.0)
                    D = np.minimum(D, de.reshape(GX.shape) - self.c_env)
            # bone (only where the section runs near bone)
            if self.bones:
                Pp = lig.frame_point(np.full(poly.shape[0], s), poly)
                dmin = min(float(b.sdf(Pp).min()) for b in self.bones)
                if dmin < 3.0:
                    # bone distance is smooth: evaluate on a 3x coarser grid, interpolate
                    xs3, ys3 = xs[::3], ys[::3]
                    G3 = np.stack([a.ravel() for a in np.meshgrid(xs3, ys3, indexing="ij")], 1)
                    P3 = lig.frame_point(np.full(G3.shape[0], s), G3)
                    db = (np.min([b.sdf(P3) for b in self.bones], axis=0) - self.clr).reshape(xs3.size, ys3.size)
                    ii = (GX - xs3[0]) / (3 * h)
                    jj = (GY - ys3[0]) / (3 * h)
                    Db = ndimage.map_coordinates(db, [ii.ravel(), jj.ravel()], order=1, mode="nearest").reshape(GX.shape)
                    D = np.minimum(D, Db)
            f = Field(sig, s, xs, ys, h, D, Danat)
            # retained-fascicle obstacles: all retained (interdigitation) / continuous only (gap, braid)
            Xo, ro, cont = self._obstacles(u, s, i_st)
            if Xo.shape[0]:
                dd = np.linalg.norm(G[:, None, :] - Xo[None], axis=2) - ro[None] - self.clr
                f.D = np.minimum(D, dd.min(1).reshape(GX.shape))
                if cont.any():
                    f.Dcont = np.minimum(D, dd[:, cont].min(1).reshape(GX.shape))
                else:
                    f.Dcont = D
            else:
                f.D = D
                f.Dcont = D
            u.fields.append(f)

    def _fields_in(self, u, a, b):
        return [f for f in u.fields if a - 1e-6 <= f.sig <= b + 1e-6]

    # ------------------------------------------------------------ regions
    def region_weights(self, u, sig):
        def step(x, b, w, mode):
            if mode == "abrupt" or w <= 1e-9:
                return (x >= b).astype(float)
            return _sstep((x - (b - w / 2)) / w)
        rt = self.reg["tibial_interface"]
        rf = self.reg["femoral_interface"]
        wt = 1 - step(sig, u.b1, rt["tw"], rt["mode"])
        wf = step(sig, u.b2, rf["tw"], rf["mode"])
        wb = np.clip(1 - wt - wf, 0, 1)
        W = np.stack([wt, wb, wf], 1)
        return W / np.maximum(W.sum(1, keepdims=True), 1e-9)

    def region_values(self, u, sig, key):
        W = self.region_weights(u, sig)
        v = np.array([self.reg[r][key] for r in REGION_IDS], float)
        return W @ v

    def region_id(self, u, sig):
        return np.argmax(self.region_weights(u, sig), axis=1).astype(np.uint8)

    def samples_sig(self, a, b):
        n = int(math.ceil((b - a) / (0.2 if self.quality == "full" else 0.45))) + 1
        return np.linspace(a, b, max(n, 12))

    # ------------------------------------------------------------ yarns
    def yarns(self, bones=None):
        out = []
        self.braid_info = []
        for u in self.units:
            ys = self._braided(u) if self.arch == "braided" else self._aligned(u)
            ys = self._conform_ends(u, ys)
            for y in ys:
                y["id"] = len(out)
                y["unit"] = u.id
                out.append(y)
        if bones:
            out = self._conform_bone(out, bones)
        out = self._trim_anchor_convergence(out)
        for u in self.units:
            self.messages.extend(u.messages)
        return out

    def _yarn_record(self, u, family, sig, m=None, P=None, layer=0):
        lig = self.lig
        s = sig / lig.L
        if P is None:
            P = lig.map(s, m)
        dia = self.d0 * self.region_values(u, sig, "scale")
        return {"id": -1, "family": family, "sig": sig, "s": s, "m": m, "P": P, "layer": layer,
                "r": 0.5 * dia, "region": self.region_id(u, sig)}

    def _trim_anchor_convergence(self, ys):
        """Inside an insertion flare the section narrows onto the footprint and braid
        yarns can converge. Yarns are trimmed back from their BONE end (never in the
        middle) until no strand touches another there; how far each braid then stops
        short of the footprint is reported."""
        L = self.lig.L
        fl_t = float(self.p["lig.flare_tib_mm"])
        fl_f = float(self.p["lig.flare_fem_mm"])
        units = {u.id: u for u in self.units}
        for _ in range(12):
            if len(ys) < 2:
                break
            P = np.vstack([y["P"] for y in ys])
            R = np.concatenate([y["r"] for y in ys])
            O = np.concatenate([np.full(y["P"].shape[0], i) for i, y in enumerate(ys)])
            SG = np.concatenate([y["sig"] for y in ys])
            d, idx = cKDTree(P).query(P, k=min(16, P.shape[0]))
            gap = np.where(O[idx] != O[:, None], d - R[:, None] - R[idx] - self.clr, np.inf).min(1)
            changed = False
            for i, y in enumerate(ys):
                u = units[y["unit"]]
                sel = O == i
                bad = gap[sel] < -0.005
                if not bad.any():
                    continue
                sg = SG[sel]
                if u.bone_fem:
                    zone = bad & (sg > L - fl_f)
                    if zone.any():
                        k = int(np.nonzero(zone)[0].min())
                        if k >= 3:
                            for key in ("sig", "s", "m", "P", "r", "region"):
                                y[key] = y[key][:k]
                            y["trimmed_fem"] = True
                            changed = True
                            continue
                if u.bone_tib:
                    zone = bad & (sg < fl_t)
                    if zone.any():
                        k = int(np.nonzero(zone)[0].max()) + 1
                        if y["sig"].size - k >= 3:
                            for key in ("sig", "s", "m", "P", "r", "region"):
                                y[key] = y[key][k:]
                            y["trimmed_tib"] = True
                            changed = True
            if not changed:
                break
        for u in self.units:
            for side, flag, nm in (("fem", "trimmed_fem", "femoral"), ("tib", "trimmed_tib", "tibial")):
                tr = [y for y in ys if y["unit"] == u.id and y.get(flag)]
                if tr:
                    short = max((L - y["sig"][-1]) if side == "fem" else y["sig"][0] for y in tr)
                    u.messages.append({"param": "fit.end_transition_mm", "level": "info",
                                       "message": "%s: %d yarns converge inside the %s insertion and stop up to %.1f mm short of the footprint "
                                                  "(trimmed from their bone end so no strands touch)" % (u.label, len(tr), nm, short)})
        return ys

    def _conform_bone(self, ys, bones):
        """Keep yarns off bone at bone-anchored ends (inside the insertion flare the
        fields use the body section, so a final push is applied). Braid yarns are
        shifted together per position so their crossings stay separated."""
        clr = self.clr
        moved = 0

        def need_vec(P, r):
            push = np.zeros(P.shape[0])
            nrm = np.zeros_like(P)
            for b in bones:
                d = b.sdf(P)
                nd = r + clr - d
                sel = nd > push
                if sel.any():
                    push = np.where(sel, nd, push)
                    nrm[sel] = b.normal(P[sel])
            return push, nrm
        for uid in set(y["unit"] for y in ys):
            braid = [y for y in ys if y["unit"] == uid and y["family"] in ("+", "-")]
            if braid:
                allsig = np.concatenate([y["sig"] for y in braid])
                vecs = []
                for y in braid:
                    pu, nr = need_vec(y["P"], y["r"])
                    vecs.append(np.maximum(pu, 0)[:, None] * nr)
                V = np.vstack(vecs)
                mag = np.linalg.norm(V, axis=1)
                if (mag > 0).any():
                    edges = np.arange(allsig.min(), allsig.max() + 0.21, 0.2)
                    bi = np.clip(np.digitize(allsig, edges) - 1, 0, edges.size - 2)
                    best = np.zeros((edges.size - 1, 3))
                    bm = np.zeros(edges.size - 1)
                    for k in np.nonzero(mag > 0)[0]:
                        if mag[k] > bm[bi[k]]:
                            bm[bi[k]] = mag[k]
                            best[bi[k]] = V[k]
                    ker = np.ones(5) / 5.0
                    sm = np.stack([np.convolve(np.pad(best[:, a], 2, mode="edge"), ker, "valid") for a in range(3)], 1)
                    sm = np.where(np.linalg.norm(sm, axis=1, keepdims=True) < np.linalg.norm(best, axis=1, keepdims=True), best, sm)
                    cen = 0.5 * (edges[:-1] + edges[1:])
                    for y in braid:
                        y["P"] = y["P"] + np.stack([np.interp(y["sig"], cen, sm[:, a]) for a in range(3)], 1)
                    moved += int((mag > 0).sum())
        L = self.lig.L
        fl_t = float(self.p["lig.flare_tib_mm"])
        fl_f = float(self.p["lig.flare_fem_mm"])
        units = {u.id: u for u in self.units}
        trimmed = 0
        for y in ys:
            if y["family"] in ("+", "-"):
                continue
            pu, nr = need_vec(y["P"], y["r"])
            u = units.get(y["unit"])
            # an aligned yarn that anchors to bone ENDS where it meets the bone surface
            # (pushing it sideways along the bone would twist it about the unit axis)
            if u is not None and (pu > 0).any():
                near = pu > 0
                keep = np.ones(near.size, bool)
                if u.bone_fem:
                    zone = near & (y["sig"] > L - fl_f)
                    if zone.any():
                        keep[int(np.nonzero(zone)[0].min()):] = False
                if u.bone_tib:
                    zone = near & (y["sig"] < fl_t)
                    if zone.any():
                        keep[:int(np.nonzero(zone)[0].max()) + 1] = False
                if not keep.all() and keep.sum() >= 3:
                    idx = np.nonzero(keep)[0]
                    a, b = idx[0], idx[-1] + 1
                    for key in ("sig", "s", "m", "P", "r", "region"):
                        y[key] = y[key][a:b]
                    trimmed += 1
                    pu, nr = need_vec(y["P"], y["r"])
            if (pu > 0).any():
                k = np.ones(7) / 7.0
                pw = np.convolve(np.pad(np.maximum(pu, 0), 3, mode="edge"), k, "valid")
                pw = np.maximum(pw, np.maximum(pu, 0))
                y["P"] = y["P"] + pw[:, None] * nr
                moved += int((pu > 0).sum())
        if trimmed:
            self.messages.append({"param": "scaffold.bone_reach_mm", "level": "info",
                                  "message": "%d bone-anchored aligned yarns end where they meet the bone surface (strand radius + clearance)" % trimmed})
        if moved:
            self.messages.append({"param": "scaffold.bone_reach_mm", "level": "info",
                                  "message": "%d yarn samples inside the insertion flare were moved off the bone surface; the fit checks below measure the result" % moved})
        return ys

    def _conform_ends(self, u, ys):
        """Yarns that abut a tear face end at their own local face (minus the end gap),
        and before any retained fascicle surface (strand radius + yarn clearance), so
        the scaffold end follows the actual irregular remnant surface."""
        out = []
        gap = float(self.p["fit.end_gap_mm"])
        tree, frad = self._retained_tree(u)
        for y in ys:
            abut = y["family"] in ("+", "-") or self.engagement == "abut"
            if not abut:
                out.append(y)
                continue
            lo, hi = self.inj.bounds(y["s"], y["m"], y["P"])
            keep = np.ones(y["sig"].shape, bool)
            if not u.bone_tib:
                keep &= y["sig"] >= lo + gap + y["r"]
            if not u.bone_fem:
                keep &= y["sig"] <= hi - gap - y["r"]
            if tree is not None:
                dd, jj = tree.query(y["P"])
                clash = dd < frad[jj] + y["r"] + self.clr + 0.01
                if clash.any():
                    # trim only from the ends inward (a clash in the middle is a real violation)
                    ok = keep & ~clash
                    idx = np.nonzero(ok)[0]
                    if idx.size:
                        mid = int(np.argmin(np.abs(y["sig"] - 0.5 * (y["sig"][idx[0]] + y["sig"][idx[-1]]))))
                        k0 = mid
                        while k0 > 0 and not clash[k0 - 1] and keep[k0 - 1]:
                            k0 -= 1
                        k1 = mid
                        while k1 < keep.size - 1 and not clash[k1 + 1] and keep[k1 + 1]:
                            k1 += 1
                        if not clash[mid]:
                            keep = np.zeros_like(keep)
                            keep[k0:k1 + 1] = True
            if keep.all():
                out.append(self._end_gaps(u, y, tree, frad, gap))
                continue
            m = np.concatenate([[False], keep, [False]]).astype(np.int8)
            st = np.nonzero(np.diff(m) == 1)[0]
            en = np.nonzero(np.diff(m) == -1)[0]
            if not st.size:
                continue
            k = int(np.argmax(en - st))
            a, b = st[k], en[k]
            if b - a < 3:
                continue
            y2 = dict(y)
            for key in ("sig", "s", "m", "P", "r", "region"):
                y2[key] = y[key][a:b]
            y2["conformed"] = True
            out.append(self._end_gaps(u, y2, tree, frad, gap))
        return out

    def _end_gaps(self, u, y, tree, frad, gap):
        """Distance from each abutting yarn end to the remnant: the nearer of the local
        tear face and the nearest retained fascicle surface (mm, strand radius included)."""
        lo, hi = self.inj.bounds(y["s"][[0, -1]], y["m"][[0, -1]], y["P"][[0, -1]])
        g = []
        for e, face, sg, anch in ((0, lo[0], 1.0, u.bone_tib), (-1, hi[-1], -1.0, u.bone_fem)):
            if anch:
                g.append(None)
                continue
            gf = sg * (y["sig"][e] - face) - y["r"][e]
            if tree is not None:
                dd, jj = tree.query(y["P"][e])
                gf = min(gf, float(dd - frad[jj] - y["r"][e]))
            g.append(float(gf))
        y["end_gaps"] = g
        return y

    def _retained_tree(self, u):
        """KD-tree of retained fascicle samples that a yarn END must stop short of
        (remnant tissue of affected fascicles near the faces)."""
        if not self.fascicles:
            return None, None
        L = self.lig.L
        pts, rr = [], []
        for fa in self.fascicles:
            if not fa["affected"] or fa["bridging"]:
                continue
            keep = ~fa["damaged"]
            sg = fa["s"] * L
            if u.bone_tib:
                keep &= sg > u.ext["lo_max"]
            if u.bone_fem:
                keep &= sg < u.ext["hi_min"]
            keep &= (sg > u.sig_a - 2) & (sg < u.sig_b + 2)
            if keep.any():
                # densify along the fascicle (samples are ~0.18 mm apart)
                pts.append(fa["P"][keep])
                rr.append(fa["r"][keep])
        if not pts:
            return None, None
        return cKDTree(np.vstack(pts)), np.concatenate(rr)

    # ------------------------------------------------------------ aligned yarn paths
    def _aligned_model(self, u, fields, rot_deg):
        """Path model of aligned (and core) yarns in true section millimetres.

        Each yarn keeps a fixed offset d from the repair region's centre in the
        section frame (the ligament's rotation-minimizing frame, which does not
        twist); the offset is scaled along the frame's two axes with the region's
        size (fitting strength) and turned rigidly by phi(sigma), which goes
        linearly from 0 to rot_deg over the unit (signed total rotation; the
        surgeon-informed range is -30..30 deg). The ligament's own section
        orientation is NOT passed on, so the measured total equals rot_deg."""
        sig = np.array([f.sig for f in fields])
        C, SX, SY = [], [], []
        for f in fields:
            Q = f.grid()[f.Danat.ravel() >= 0]
            if not Q.shape[0]:
                Q = f.grid()[[np.argmax(f.Danat.ravel())]]
            C.append(Q.mean(0))
            SX.append(max(float(Q[:, 0].std()), 1e-3))
            SY.append(max(float(Q[:, 1].std()), 1e-3))
        C = np.array(C)
        k = min(7, 2 * (len(fields) // 2) - 1) if len(fields) > 2 else 1
        Cs = np.stack([np.convolve(np.pad(C[:, a], k // 2, mode="edge"), np.ones(k) / k, "valid") for a in range(2)], 1)
        span = max(u.sig_b - u.sig_a, 1e-9)
        md = {"sig": sig, "c": Cs, "sx": np.array(SX), "sy": np.array(SY), "w": u.strength,
              "tot": math.radians(float(rot_deg)), "a": u.sig_a, "span": span, "corr": np.zeros(sig.size)}
        self._cancel_inherited_twist(md)
        return md

    def _cancel_inherited_twist(self, md, iters=3):
        """Inside an insertion flare the section blends onto the footprint chart, which
        can turn the yarns about the unit axis. That inherited turn is measured on a
        ring of reference paths (in a non-twisting frame along the unit axis) and
        cancelled, so the yarns' total rotation stays the requested value."""
        from acl_engine.v2.rotation import rmf_frames
        lig = self.lig
        wt, wf = lig.blend_weights(md["sig"] / lig.L)
        if not ((wt > 0) | (wf > 0)).any():
            return
        sig = md["sig"]
        r0 = 0.6 * math.sqrt(float(md["sx"].mean() * md["sy"].mean()))
        ph = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        ring = r0 * np.stack([np.cos(ph), np.sin(ph)], 1)
        axis = self._to3d(sig, md["c"], blend=False)
        _, U1, U2 = rmf_frames(axis)
        target = md["tot"] * (sig - md["a"]) / md["span"]
        for _ in range(iters):
            ang = []
            for d in ring:
                X = self._model_X(md, 0, d, sig)
                Q = self._to3d(sig, X, blend=False) - axis
                a = np.unwrap(np.arctan2((Q * U2).sum(1), (Q * U1).sum(1)))
                ang.append(a - a[0])
            meas = np.mean(ang, axis=0)
            err = target - meas
            if np.abs(err).max() < math.radians(0.2):
                break
            md["corr"] = md["corr"] + err

    def _model_X(self, md, jref, d, sig):
        """Metric positions of yarns with reference offsets d (k,2) at arc positions sig."""
        sig = np.atleast_1d(np.asarray(sig, float))
        c = np.stack([np.interp(sig, md["sig"], md["c"][:, a]) for a in range(2)], 1)
        w = md["w"]
        # similarity transform (uniform scale + rotation): it preserves angles, so the
        # yarns' measured rotation about the unit axis is exactly phi
        k = (1 - w) + w * np.sqrt(np.interp(sig, md["sig"], md["sx"]) * np.interp(sig, md["sig"], md["sy"])
                                  / (md["sx"][jref] * md["sy"][jref]))
        kx = ky = k
        phi = md["tot"] * (sig - md["a"]) / md["span"] + np.interp(sig, md["sig"], md["corr"])
        if d.ndim == 2 and sig.size == 1:
            Y = d * np.array([kx[0], ky[0]])
            cs, sn = math.cos(phi[0]), math.sin(phi[0])
            return c[0] + np.stack([cs * Y[:, 0] - sn * Y[:, 1], sn * Y[:, 0] + cs * Y[:, 1]], 1)
        Y = d[None] * np.stack([kx, ky], 1) if d.ndim == 1 else d * np.stack([kx, ky], 1)
        cs, sn = np.cos(phi), np.sin(phi)
        return c + np.stack([cs * Y[:, 0] - sn * Y[:, 1], sn * Y[:, 0] + cs * Y[:, 1]], 1)

    def _ref_offsets(self, md, jref, Xref):
        """Offsets d whose position at the reference station is Xref."""
        sref = md["sig"][jref]
        c = md["c"][jref]
        phi = md["tot"] * (sref - md["a"]) / md["span"] + md["corr"][jref]
        cs, sn = math.cos(-phi), math.sin(-phi)
        Y = Xref - c
        return np.stack([cs * Y[:, 0] - sn * Y[:, 1], sn * Y[:, 0] + cs * Y[:, 1]], 1)

    def _to3d(self, sig, X, blend=True):
        """Section (metric) points -> 3D. blend=False keeps them in the non-twisting
        section frames inside the insertion flares too (used for aligned yarns, which
        run straight into bone instead of being redistributed over the footprint)."""
        lig = self.lig
        s = np.asarray(sig, float) / lig.L
        P = lig.frame_point(s, X)
        if not blend:
            return P
        wt, wf = lig.blend_weights(s)
        fl = (wt > 0) | (wf > 0)
        if fl.any():
            P[fl] = lig.map(s[fl], lig.metric_to_material(s[fl], X[fl]))
        return P

    # ------------------------------------------------------------ aligned layout
    def _layout(self, u, n_req, r_mm, fields, which, s_ref=None, extra=None, seed_off=5, rot_deg=0.0):
        """Reference positions (metric, at the reference station) of up to n_req
        strands of radius r_mm whose PATHS fit at EVERY station in `fields`
        (D >= r) with centre spacing >= 2r + clearance at every station.
        extra(j, X) optionally adds a per-station constraint."""
        if not fields:
            return np.zeros((0, 2)), {}
        areas = np.array([f.area(which, r_mm) for f in fields])
        jr = int(np.argmin(areas)) if s_ref is None else int(np.argmin([abs(f.s - s_ref) for f in fields]))
        fr = fields[jr]
        md = self._aligned_model(u, fields, rot_deg)
        G = fr.grid()
        Dr = getattr(fr, which).ravel()
        cand = Dr >= r_mm
        if extra is not None:
            cand &= extra(jr, G) >= r_mm
        info = {"s_ref": fr.s, "jref": jr, "model": md, "narrowest_area_mm2": float(areas.min()) if areas.size else 0.0}
        if not cand.any():
            return np.zeros((0, 2)), info
        Gc = G[cand]
        dc = self._ref_offsets(md, jr, Gc)
        minD = np.full(Gc.shape[0], np.inf)
        for j, f in enumerate(fields):
            Xj = self._model_X(md, jr, dc, f.sig)
            dj = f.sample(Xj, which)
            if extra is not None:
                dj = np.minimum(dj, extra(j, Xj))
            minD = np.minimum(minD, dj)
        ok_px = np.zeros(G.shape[0], bool)
        ok_px[np.nonzero(cand)[0][minD >= r_mm]] = True
        info["min_available_mm"] = float(minD.max()) if minD.size else 0.0
        if not ok_px.any():
            return np.zeros((0, 2)), info
        mask = ok_px.reshape(fr.xs.size, fr.ys.size)
        lo = np.array([fr.xs[0], fr.ys[0]])
        span = max(fr.xs[-1] - fr.xs[0], fr.ys[-1] - fr.ys[0]) / 2.0

        def lookup(Pn):
            X = Pn * span
            i = np.rint((X[:, 0] - lo[0]) / fr.h).astype(int)
            j = np.rint((X[:, 1] - lo[1]) / fr.h).astype(int)
            okk = (i >= 0) & (i < fr.xs.size) & (j >= 0) & (j < fr.ys.size)
            out = np.zeros(Pn.shape[0], bool)
            out[okk] = mask[i[okk], j[okk]]
            return out
        cen_x = md["c"][jr]
        if not lookup(cen_x[None] / span)[0]:
            cen_x = G[ok_px].mean(0)
        need = 2 * r_mm + self.clr
        sp = need
        seed = int(self.p["native.layout_seed"]) + seed_off
        best = np.zeros((0, 2))
        for _ in range(8):
            X = layout_points("hexagonal", n_req, lookup, lambda Pn: np.ones(Pn.shape[0], bool), seed, cen_x / span,
                              min_spacing=sp / span) * span
            if X.shape[0] == 0:
                break
            worst = np.inf
            if X.shape[0] > 1:
                d = self._ref_offsets(md, jr, X)
                for f in fields:
                    Xj = self._model_X(md, jr, d, f.sig)
                    dd = cKDTree(Xj).query(Xj, k=2)[0][:, 1]
                    worst = min(worst, float(dd.min()))
            best = X
            if worst >= need - 1e-6:
                break
            sp *= (need / max(worst, 1e-6)) * 1.01
        info["spacing_mm"] = sp
        return best, info

    def _aligned_paths(self, u, Xref, family, sig, info, layer=0):
        """Yarn records from reference positions, following the unit's path model."""
        lig = self.lig
        md, jr = info["model"], info["jref"]
        u.axis_model = md
        out = []
        d_all = self._ref_offsets(md, jr, Xref) if len(Xref) else np.zeros((0, 2))
        for d in d_all:
            X = self._model_X(md, jr, d, sig)
            P = self._to3d(sig, X, blend=False)
            m = lig.metric_to_material(sig / lig.L, X)
            out.append(self._yarn_record(u, family, sig, m=m, P=P, layer=layer))
        return out

    def _aligned(self, u):
        p = self.p
        r = 0.5 * self.dmax
        which = "D" if self.engagement == "interdigitate" else "Dcont"
        fields = u.fields
        if p["scaffold.aligned_sizing"] == "packing":
            areas = [f.area(which, 0.0) for f in fields]
            a_min = min(areas) if areas else 0.0
            n = max(1, int(round(float(p["scaffold.packing_fraction"]) * a_min / (math.pi * r * r))))
            u.sizing = {"mode": "packing", "target": float(p["scaffold.packing_fraction"]), "repair_area_min_mm2": a_min, "count": n}
        else:
            n = int(p["scaffold.yarn_count"])
            u.sizing = {"mode": "count", "count": n}
        rot_deg = float(p["scaffold.yarn_rotation_deg"])      # total rotation about the unit axis
        X, info = self._layout(u, n, r, fields, which, rot_deg=rot_deg)
        u.layout_info = {k: v for k, v in info.items() if k not in ("model", "jref")}
        if X.shape[0] < n:
            self._fit_failure(u, "aligned", n, X.shape[0], r, fields, which, rot_deg)
        sig = self.samples_sig(u.sig_a, u.sig_b)
        return self._aligned_paths(u, X, "axial", sig, info) if "model" in info else []

    def _fit_failure(self, u, kind, n_req, n_got, r, fields, which, rot_deg=0.0, extra=None, param="scaffold.yarn_count"):
        """Explicit resolutions: what WOULD fit (nothing is applied automatically)."""
        p = self.p
        what = {"aligned": "yarns", "core": "core yarns"}[kind]
        msg = "%s: only %d of %d %s of %.2f mm fit inside the repair space at every station (clearance %.2f mm from the boundary)" % (
            u.label, n_got, n_req, what, 2 * r, self.c_env)
        u.messages.append({"param": param, "level": "fail", "message": msg})
        res = []
        if n_got > 0:
            res.append({"param": param, "value": int(n_got) if kind == "aligned" or param != "scaffold.core_yarns" else int(n_got),
                        "label": "Use %d %s (the most that fit at %.2f mm)" % (n_got, what, 2 * r), "unit": u.id})
        # largest diameter that fits n_req (bisection on the layout)
        lo, hi = 0.05, r
        if self._layout(u, n_req, lo, fields, which, rot_deg=rot_deg, extra=extra)[0].shape[0] >= n_req:
            for _ in range(12):
                mid = 0.5 * (lo + hi)
                if self._layout(u, n_req, mid, fields, which, rot_deg=rot_deg, extra=extra)[0].shape[0] >= n_req:
                    lo = mid
                else:
                    hi = mid
            scale = max(self.reg[k]["scale"] for k in REGION_IDS)
            dnew = math.floor(2 * lo / scale * 100) / 100
            if dnew >= 0.1:
                res.append({"param": "scaffold.yarn_diameter_mm", "value": dnew,
                            "label": "Use %.2f mm yarns (base diameter) so %d %s fit" % (dnew, n_req, what), "unit": u.id})
        if self.c_env > 0.05:
            res.append({"param": "fit.boundary_clearance_mm", "value": 0.05,
                        "label": "Reduce the boundary clearance to 0.05 mm (check the result)", "unit": u.id})
        self.resolutions.extend(res)

    # ------------------------------------------------------------ braid
    def _tube(self, u, fields, depth, rho, nph=180):
        """Braid tube for one layer.

        The tube is the outline of the region {D >= depth} OPENED by a disk of
        radius rho (erode by rho, dilate by rho): it stays inside {D >= depth} and
        no convex part bends tighter than rho, so strands offset inward by the
        over-under depth cannot be squeezed together at the narrow edges of a
        flattened (ribbon) section. Then it is smoothed only inward."""
        phs = np.linspace(0, 2 * np.pi, nph, endpoint=False)
        dirs = np.stack([np.cos(phs), np.sin(phs)], 1)
        C = np.zeros((len(fields), 2))
        R = np.zeros((len(fields), nph))
        bad = []
        opened = []
        nwin = 4
        for j, f in enumerate(fields):
            # region shared by the neighbouring stations (smoothing along the ligament
            # done on the REGION, so the opening below keeps its bend radius)
            G = f.grid()
            A = (f.Dcont >= depth + rho).ravel()
            for jj in range(max(0, j - nwin), min(len(fields), j + nwin + 1)):
                if jj != j and A.any():
                    A &= fields[jj].sample(G, "Dcont") >= depth + rho
            A = A.reshape(f.Dcont.shape)
            if not A.any():
                bad.append(j)
                opened.append(None)
                continue
            Gf = rho - ndimage.distance_transform_edt(~A) * f.h      # >= 0 inside the opened region
            opened.append(Gf)
            okv = (Gf >= 0).ravel()
            wts = np.maximum(f.Dcont.ravel()[okv] - depth, 0) + 1e-6
            C[j] = (G[okv] * wts[:, None]).sum(0) / wts.sum()
            if _samp(Gf, f, C[j][None])[0] < 0:
                C[j] = G[A.ravel()][np.argmax(f.Dcont.ravel()[A.ravel()])]
        if bad:
            return None, None, phs, bad
        # one polar centre for the whole tube when possible: yarns are spaced by arc
        # length measured from the centre's phi = 0 ray, so a centre that moves
        # between stations would slide the whole braid sideways
        cbar = C.mean(0)
        inside_all = all(_samp(opened[j], f, cbar[None])[0] >= 0.05 for j, f in enumerate(fields))
        if inside_all:
            Cs = np.repeat(cbar[None], len(fields), 0)
        else:
            Cs = C.copy()
            k = min(31, 2 * (len(fields) // 2) - 1) if len(fields) > 2 else 1
            for a in range(2):
                Cs[:, a] = np.convolve(np.pad(C[:, a], k // 2, mode="edge"), np.ones(k) / k, "valid")
            for j, f in enumerate(fields):
                if _samp(opened[j], f, Cs[j][None])[0] < 0:
                    Cs[j] = C[j]
        for j, f in enumerate(fields):
            # rays only as long as the station raster (from the centre to its far corner)
            corners = np.array([[f.xs[0], f.ys[0]], [f.xs[-1], f.ys[0]], [f.xs[0], f.ys[-1]], [f.xs[-1], f.ys[-1]]])
            tmax = float(np.linalg.norm(corners - Cs[j], axis=1).max())
            nt = max(20, int(tmax / (self.h * 0.75)))
            tt = np.linspace(0, tmax, nt)
            X = Cs[j][None, None, :] + tt[None, :, None] * dirs[:, None, :]
            Dv = _samp(opened[j], f, X.reshape(-1, 2)).reshape(nph, nt)
            okk = Dv >= 0
            first_bad = np.argmax(~okk, axis=1)
            first_bad = np.where(okk.all(1), nt - 1, first_bad)
            k0 = np.maximum(first_bad - 1, 0)
            d0 = Dv[np.arange(nph), k0]
            d1 = Dv[np.arange(nph), first_bad]
            frac = np.where(d0 - d1 > 1e-12, d0 / np.maximum(d0 - d1, 1e-12), 0.0)
            R[j] = tt[k0] + np.clip(frac, 0, 1) * (tt[1] - tt[0])
        # along the ligament: each station's R is valid for the 2 neighbours on each
        # side, so a +/-2 moving average stays inside; around the tube only a tiny
        # non-increasing de-noise (no min-filter: it would sharpen the tube's tips)
        Ra = ndimage.uniform_filter1d(R, size=2 * nwin + 1, axis=0, mode="nearest")
        R = Ra
        R = np.minimum(R, ndimage.uniform_filter1d(R, size=3, axis=1, mode="wrap"))
        # concave bends compress the OUTWARD-offset yarns the same way convex bends
        # compress the inward ones: widen the (inward-only) smoothing where needed
        for j in range(R.shape[0]):
            w = 3
            while _min_concave_radius(R[j], phs) < rho and w < nph // 4:
                w = int(w * 1.6) + 1
                R[j] = _inward_smooth(R[j][None], w, axis=1, wrap=True)[0]
        return Cs, R, phs, []

    def _braided(self, u):
        p = self.p
        lig = self.lig
        L = lig.L
        n = int(p["scaffold.yarn_count"])
        npd = n // 2
        nl = int(p["scaffold.braid_layers"])
        bfields = self._fields_in(u, u.braid_a, u.braid_b)
        if len(bfields) < 2:
            u.messages.append({"param": "injury.gap_mm", "level": "fail", "message": "%s: the gap is too short for a braid" % u.label})
            return []
        d = self.dmax
        r = 0.5 * d
        delta = 0.5 * (d + self.clr) * 1.2
        pitch_l = 2 * delta + 2 * r + float(p["fit.layer_spacing_mm"])
        sig = self.samples_sig(u.braid_a, u.braid_b)
        s_ = sig / L
        fsig = np.array([f.sig for f in bfields])
        clr = self.clr
        out = []
        layers = []
        frac = float(p["scaffold.braid_radius_fraction"])
        dmax_avail = min(float(f.Dcont.max()) for f in bfields)
        need = d + clr
        rho0 = delta + r + 0.5 * clr
        info_layers = []
        last = None
        self.fan_info = {"tibial": bool(u.bone_tib), "femoral": bool(u.bone_fem)}
        for li in range(nl):
            depth = r + delta + li * pitch_l
            # tightest bend the braid can follow: the strand itself, and yarns offset
            # inward on a bend, whose chord shrinks to 2 (rho - delta) sin(arc / 2 rho)
            rho = rho0
            n_msg, n_res = len(u.messages), len(self.resolutions)
            accepted = None
            best_g = -np.inf
            rho_ok = None            # largest bend radius known to fit
            rho_bad = None           # smallest bend radius known NOT to fit
            msgs_keep = res_keep = None
            for attempt in range(7):
                del u.messages[n_msg:]
                del self.resolutions[n_res:]
                Cs, R, phs, bad = self._tube(u, bfields, depth, rho)
                if bad:
                    rho_bad = rho
                    if rho_ok is None:
                        break                    # does not fit even at the strand's own bend radius
                    rho = 0.5 * (rho_ok + rho_bad)
                    if rho_bad - rho_ok < 0.03:
                        break
                    continue
                rho_ok = rho if rho_ok is None else max(rho_ok, rho)
                if attempt == 0:
                    per = float(np.min([_perimeter(Cs[j], R[j], phs) for j in range(R.shape[0])]))
                    alpha, _ = braid_dependents(p, per / (2 * math.pi))
                    need_plane = need / max(math.cos(math.radians(alpha)), 0.2)
                    rr = _bend_radius_for(per / max(npd, 1), delta, need_plane, rho0, 3.0 * rho0)
                    if rr > rho + 0.02:
                        rho = rr
                        continue
                Rl = R.copy()
                w = u.strength
                if w < 1:
                    Rl = w * Rl + (1 - w) * Rl.min(0)[None]
                Rl = Rl * frac
                ys, info = self._braid_layer(u, sig, fsig, Cs, Rl, phs, npd, delta, li)
                info["min_bend_radius_mm"] = rho
                g = _min_gap(ys, clr)
                info["measured_min_clearance_mm"] = g
                if g > best_g:
                    best_g = g
                    accepted = (Cs, Rl, depth, rho, ys, info)
                    msgs_keep = list(u.messages[n_msg:])
                    res_keep = list(self.resolutions[n_res:])
                if g >= -0.005:
                    break
                nxt = min(rho * 1.3, 3.0 * rho0)       # measured contact: round the tube more and regenerate
                if rho_bad is not None:
                    nxt = min(nxt, 0.5 * (rho + rho_bad))
                if nxt <= rho + 0.01:
                    break
                rho = nxt
            if accepted is not None:
                del u.messages[n_msg:]
                del self.resolutions[n_res:]
                u.messages.extend(msgs_keep)
                self.resolutions.extend(res_keep)
                if best_g < -0.005:
                    scale = max(self.reg[k]["scale"] for k in REGION_IDS)
                    if npd > 2:
                        self.resolutions.append({"param": "scaffold.yarn_count", "value": 2 * (npd - 1), "unit": u.id,
                                                 "label": "Use %d carriers (layer %d: measured strand contact %.2f mm)" % (2 * (npd - 1), li + 1, -best_g)})
                    dnew = math.floor((d + best_g - 0.01) / scale * 100) / 100
                    if 0.1 <= dnew < self.d0:
                        self.resolutions.append({"param": "scaffold.yarn_diameter_mm", "value": dnew, "unit": u.id,
                                                 "label": "Use %.2f mm yarns (base) to remove the measured %.2f mm contact on layer %d" % (dnew, -best_g, li + 1)})
            if accepted is None:
                ss = bfields[bad[0]].s if bad else 0.5
                u.messages.append({"param": "scaffold.braid_layers" if li else "scaffold.yarn_diameter_mm", "level": "fail",
                                   "message": "%s: braid layer %d does not fit at s=%.2f: it needs %.2f mm of free depth (strand radius + over-under + the bend radius the "
                                              "crossings need) but the deepest point there is %.2f mm from the constraints"
                                              % (u.label, li + 1, ss, depth + rho, float(bfields[bad[0]].Dcont.max()) if bad else 0.0)})
                if li > 0:
                    self.resolutions.append({"param": "scaffold.braid_layers", "value": li, "unit": u.id,
                                             "label": "Use %d braid layer%s (the most that fit)" % (li, "s" if li > 1 else "")})
                dfit = (dmax_avail - 1.7 * clr - li * (1.2 * clr + float(p["fit.layer_spacing_mm"]))) / (2.2 + 2.2 * li)
                scale = max(self.reg[k]["scale"] for k in REGION_IDS)
                dnew = math.floor(dfit / scale * 100) / 100
                if 0.1 <= dnew < self.d0:
                    self.resolutions.append({"param": "scaffold.yarn_diameter_mm", "value": dnew, "unit": u.id,
                                             "label": "Use %.2f mm yarns (base) so %d braid layer%s fit" % (dnew, li + 1, "s" if li else "")})
                break
            Cs, R, depth, rho, ys, info = accepted
            layers.append((Cs, R, depth, rho))
            out.extend(ys)
            info_layers.append(info)
        if not layers:
            return []
        # core yarns inside the innermost layer (+ into the remnants when interdigitating)
        Cs, R, depth, rho = layers[-1]
        gap = delta + r + clr

        from acl_engine.v2.ligament import polygon_distance, points_in_polygon

        tube_cache = {}

        def inside_tube(j, X, Cs=Cs, R=R, fs=fsig):
            # distance (mm) from the innermost tube curve, + inside, minus the braid's inner
            # reach; rasterized once per station on that station's grid (signed EDT)
            sgs = u.fields_sig[j]
            if sgs < fs[0] - 1e-6 or sgs > fs[-1] + 1e-6:
                return np.full(X.shape[0], 50.0)
            if j not in tube_cache:
                c = np.array([np.interp(sgs, fs, Cs[:, a]) for a in range(2)])
                Rj = np.array([np.interp(sgs, fs, R[:, k]) for k in range(R.shape[1])])
                poly = c[None] + Rj[:, None] * np.stack([np.cos(phs), np.sin(phs)], 1)
                f = cf[j]
                tube_cache[j] = signed_edt(scanline_mask(poly, f.xs, f.ys), f.h)
            return _samp(tube_cache[j], cf[j], X) - gap
        nc_req = int(p["scaffold.core_yarns"])
        mode = p["scaffold.core_mode"]
        if mode == "max":
            nc_req = 400
        if nc_req > 0:
            if self.engagement == "interdigitate":
                cf = u.fields
                which = "D"
            else:
                cf = bfields
                which = "Dcont"
            u.fields_sig = np.array([f.sig for f in cf])
            # core yarns run straight along the unit axis (0 rotation; nothing inherited)
            M, info = self._layout(u, nc_req, r, cf, which, extra=inside_tube, seed_off=11, rot_deg=0.0)
            if mode == "count" and M.shape[0] < nc_req:
                self._fit_failure(u, "core", nc_req, M.shape[0], r, cf, which, 0.0, extra=inside_tube, param="scaffold.core_yarns")
            if self.engagement == "interdigitate":
                csig = self.samples_sig(u.sig_a, u.sig_b)
            else:
                csig = sig
            cores = self._aligned_paths(u, M, "core", csig, info, layer=len(layers)) if "model" in info else []
            out.extend(cores)
            u.core_info = {"requested": None if mode == "max" else nc_req, "placed": len(cores), "mode": mode}
        self.braid_info.append({"unit": u.id, "layers": info_layers})
        return out

    def _braid_layer(self, u, sig, fsig, Cs, R, phs, npd, delta, li):
        p = self.p
        lig = self.lig
        L = lig.L
        nph = phs.size
        clr = self.clr
        s_ = sig / L

        def tube_metric(sg, ph):
            c = np.stack([np.interp(sg, fsig, Cs[:, a]) for a in range(2)], 1)
            # R interpolated in sigma, periodic in phi
            x = (np.mod(ph, 2 * np.pi)) / (2 * np.pi) * nph
            k0 = np.floor(x).astype(int) % nph
            k1 = (k0 + 1) % nph
            wk = x - np.floor(x)
            jj = np.interp(sg, fsig, np.arange(fsig.size))
            j0 = np.clip(np.floor(jj).astype(int), 0, fsig.size - 2)
            wj = jj - j0
            Rv = ((R[j0, k0] * (1 - wk) + R[j0, k1] * wk) * (1 - wj) + (R[j0 + 1, k0] * (1 - wk) + R[j0 + 1, k1] * wk) * wj)
            return c + Rv[:, None] * np.stack([np.cos(ph), np.sin(ph)], 1)

        def to3d(sg, X):
            s = sg / L
            P = lig.frame_point(s, X)
            wt, wf = lig.blend_weights(s)
            fl = (wt > 0) | (wf > 0)
            if fl.any():
                m = lig.metric_to_material(s[fl], X[fl])
                P[fl] = lig.map(s[fl], m)
            return P

        # arc-length parametrization of the tube (yarns evenly spaced in mm)
        phg = np.linspace(0, 2 * np.pi, 241)
        SS = np.repeat(sig, phg.size)
        PH = np.tile(phg, sig.size)
        G = tube_metric(SS, PH).reshape(sig.size, phg.size, 2)
        seg = np.linalg.norm(np.diff(G, axis=1), axis=2)
        cum = np.concatenate([np.zeros((sig.size, 1)), np.cumsum(seg, 1)], 1)
        perim = cum[:, -1]
        U = cum / np.maximum(perim[:, None], 1e-9) * 2 * np.pi

        def phi_of_u(uu):
            uu = np.mod(uu, 2 * np.pi)
            return np.array([np.interp(uu[i], U[i], phg) for i in range(sig.size)])
        R_mm = float(np.median(perim)) / (2 * math.pi)
        alpha, pitch = braid_dependents(p, R_mm)
        omega = 2 * math.pi / pitch
        dia = self.d0 * self.region_values(u, sig, "scale")
        Dl = 2 * math.pi / npd
        pat = p["scaffold.braid_pattern"]

        def q(x):
            k = 3.0
            c = np.cos(np.pi * x) if pat == "1/1" else math.sqrt(2) * np.cos(np.pi * x / 2 + np.pi / 4)
            return np.tanh(k * c) / math.tanh(k)

        def radial(ph, off_mm):
            X = tube_metric(sig, ph)
            Pb = to3d(sig, X)
            e = 0.01
            Pa = to3d(sig, tube_metric(sig, ph + e))
            Pm = to3d(sig, tube_metric(sig, ph - e))
            Ps = to3d(sig + 0.05, tube_metric(sig + 0.05, ph))
            Pt = to3d(sig - 0.05, tube_metric(sig - 0.05, ph))
            nrm = np.cross(Pa - Pm, Ps - Pt)
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
            # outward side from the tube outline's own 2D normal (the direction from the
            # tube centre can run along the outline on a flattened tube)
            t2 = tube_metric(sig, ph + e) - tube_metric(sig, ph - e)
            n2 = np.stack([t2[:, 1], -t2[:, 0]], 1)
            n2 /= np.maximum(np.linalg.norm(n2, axis=1, keepdims=True), 1e-12)
            Pr = to3d(sig, X + 0.05 * n2)
            sgn = np.sign(((Pr - Pb) * nrm).sum(1))
            nrm *= np.where(sgn == 0, 1.0, sgn)[:, None]
            P = Pb + off_mm[:, None] * nrm
            m = lig.metric_to_material(s_, X)
            return m, P

        # bone-anchored ends: unbraid into a fan over the end-transition length
        crossing_step = Dl / (2 * omega)
        live = np.ones_like(sig)
        ell = float(p["fit.end_transition_mm"])
        if u.bone_tib:
            live = np.minimum(live, _sstep((sig - 0.25 * ell) / ell))
        adv = np.concatenate([[0.0], np.cumsum(0.5 * (live[1:] + live[:-1]) * np.diff(sig))])
        if u.bone_fem:
            sf0 = L - 1.25 * ell
            best = None
            for sf in np.linspace(sf0 - crossing_step, sf0, 41):
                fade = 1 - _sstep((sig - sf) / ell)
                lv = np.minimum(live, fade)
                a_ = np.concatenate([[0.0], np.cumsum(0.5 * (lv[1:] + lv[:-1]) * np.diff(sig))])
                kk = a_[-1] / crossing_step
                err = abs(kk - round(kk))
                if best is None or err < best[0]:
                    best = (err, lv, a_)
            live, adv = best[1], best[2]
        sp = adv
        out = []
        for i in range(npd):
            uu = i * Dl + omega * sp
            x = (uu + omega * sp) / Dl - 0.5 + i
            ou = delta * q(x)
            m, P = radial(phi_of_u(uu), ou)
            out.append(self._yarn_record(u, "+", sig, m=m, P=P, layer=li))
            out[-1]["ou"] = ou
            out[-1]["uu"] = np.mod(uu, 2 * np.pi)
        for j in range(npd):
            uu = j * Dl + Dl / 2 - omega * sp
            x = j + (uu - omega * sp) / Dl
            ou = -delta * q(x)
            m, P = radial(phi_of_u(uu), ou)
            out.append(self._yarn_record(u, "-", sig, m=m, P=P, layer=li))
            out[-1]["ou"] = ou
            out[-1]["uu"] = np.mod(uu, 2 * np.pi)
        # checked at EVERY position along the braid: the local helix angle follows
        # from the constant advance (tan a = omega * perimeter / 2 pi)
        perl = np.maximum(perim, 1e-6)
        live_d = np.gradient(adv) / np.maximum(np.gradient(sig), 1e-9)
        a_loc = np.arctan(omega * live_d * perl / (2 * math.pi))
        lev_loc = perl * np.cos(a_loc) / (2 * npd)
        # where the braid unbraids into its bone fan the yarns no longer cross: not a jam
        lev_loc = np.where(live_d > 0.5, lev_loc, np.inf)
        if not np.isfinite(lev_loc).any():
            lev_loc = perl * np.cos(a_loc) / (2 * npd)
        kmin = int(np.argmin(lev_loc))
        per = float(perl[kmin])
        ca = math.cos(float(a_loc[kmin]))
        spacing = per / npd * ca                 # same-direction neighbours (at the narrowest position)
        level = 0.5 * spacing                    # opposite directions passing level between crossings
        need = float(dia.max()) + clr
        info = {"layer": li + 1, "angle_deg": alpha, "pitch_mm": pitch, "radius_mm": R_mm, "perimeter_mm": per,
                "per_direction": npd, "same_family_spacing_mm": spacing, "level_pass_spacing_mm": level,
                "narrowest_at_mm": float(sig[kmin]), "perimeter_range_mm": [float(perl.min()), float(perl.max())],
                "max_yarn_d_mm": float(dia.max()), "cover_factor": min(1.0, float(dia.max()) / max(level, 1e-9)),
                "over_under_mm": delta}
        if level < need:
            u.messages.append({"param": "scaffold.yarn_count", "level": "fail",
                               "message": "%s: braid layer %d jams at %.1f mm: between crossings, opposite-direction yarns pass level only %.2f mm apart "
                                          "(braid perimeter %.1f mm x cos %.0f deg / %d carriers), but %.2f mm yarns need %.2f mm"
                                          % (u.label, li + 1, float(sig[kmin]), level, per, math.degrees(float(a_loc[kmin])), 2 * npd, dia.max(), need)})
            nmax = int((per * ca) // (2 * need))
            if nmax >= 2:
                self.resolutions.append({"param": "scaffold.yarn_count", "value": 2 * nmax, "unit": u.id,
                                         "label": "Use %d carriers (the most that pass without touching on layer %d)" % (2 * nmax, li + 1)})
            scale = max(self.reg[k]["scale"] for k in REGION_IDS)
            dnew = math.floor((level - clr) / scale * 100) / 100
            if 0.1 <= dnew < self.d0:
                self.resolutions.append({"param": "scaffold.yarn_diameter_mm", "value": dnew, "unit": u.id,
                                         "label": "Use %.2f mm yarns (base) so layer %d does not jam" % (dnew, li + 1)})
            cos_need = 2 * need * npd / max(per, 1e-9)
            if cos_need < 1 and p["scaffold.braid_mode"] == "angle":
                amax = math.floor(math.degrees(math.acos(cos_need)) * 2) / 2
                if amax >= 5:
                    self.resolutions.append({"param": "scaffold.braid_angle_deg", "value": amax, "unit": u.id,
                                             "label": "Use a braid angle of %.1f deg (yarns then pass without touching)" % amax})
        return out, info

    # ------------------------------------------------------------ fibers in yarns
    def fibers(self, yarns, n_rep, exaggeration, sig_window=None):
        p = self.p
        res = []
        offs = fiber_offsets(n_rep)
        nf = offs.shape[0]
        tot = math.radians(float(p["scaffold.fiber_rotation_deg"]))
        from acl_engine.v2.rotation import rmf_frames
        lam = float(p["scaffold.crimp_wavelength_um"]) / 1000 * exaggeration
        amp = float(p["scaffold.crimp_amplitude_um"]) / 1000 * exaggeration
        units = {u.id: u for u in self.units}
        for y in yarns:
            u = units[y["unit"]]
            if sig_window is None:
                sig, P0, r = y["sig"], y["P"], y["r"]
            else:
                sig = np.linspace(sig_window[0], sig_window[1], 600)
                sig = sig[(sig >= y["sig"][0]) & (sig <= y["sig"][-1])]
                if sig.size < 4:
                    continue
                P0 = np.stack([np.interp(sig, y["sig"], y["P"][:, a]) for a in range(3)], 1)
                r = np.interp(sig, y["sig"], y["r"])
            # non-twisting frame along the yarn (a fixed reference axis would make the fibers
            # turn with the braid's winding: the parent rotation applied twice)
            T, E1, E2 = rmf_frames(P0, self.lig._interp(self.lig.N1, np.array([sig[0] / self.lig.L]))[0])
            # TOTAL rotation over the yarn's full length, linear (monotone)
            y0, y1 = float(y["sig"][0]), float(y["sig"][-1])
            alpha = tot * (sig - y0) / max(y1 - y0, 1e-9)
            disp = np.radians(self.region_values(u, sig, "disp"))
            rng = np.random.default_rng(1000 + y["id"])
            ph = rng.uniform(0, 2 * np.pi, (nf, 3))
            lam_dis = 3.0
            a_dis = np.tan(disp) * lam_dis / (2 * np.pi)
            ang = np.arctan2(offs[:, 1], offs[:, 0])[:, None] + alpha[None]
            rad = np.hypot(offs[:, 0], offs[:, 1])[:, None] * r[None]
            # dispersion as whole half-waves over the yarn (zero at both ends: no net rotation)
            tt = (sig - y0) / max(y1 - y0, 1e-9)
            n1 = max(1, int(round(2 * (y1 - y0) / lam_dis)))
            n2 = max(1, int(round(2 * (y1 - y0) / (lam_dis * 1.3))))
            w1 = np.sin(np.pi * n1 * tt)[None]
            w2 = 0.7 * np.sin(np.pi * n2 * tt)[None]
            # radial only: fibers wander in and out, never around the yarn axis
            amp_r = a_dis[None] * (np.cos(ph[:, :1]) * w1 + np.cos(ph[:, 1:2]) * w2)
            rdir = np.where(rad > 1e-9, ang, ph[:, :1])
            x = rad * np.cos(ang) + amp_r * np.cos(rdir)
            y2 = rad * np.sin(ang) + amp_r * np.sin(rdir)
            if amp > 0:
                x = x + amp * np.sin(2 * np.pi * sig[None] / max(lam, 1e-6) + ph[:, 2:3])
            mag = np.hypot(x, y2)
            lim = 0.95 * r[None]
            f = np.where(mag > lim, lim / np.maximum(mag, 1e-12), 1.0)
            P = P0[None] + (x * f)[..., None] * E1[None] + (y2 * f)[..., None] * E2[None]
            res.append({"yarn": y["id"], "P": P, "region": self.region_id(u, sig), "sig": sig, "C": P0,
                        "r": np.full(nf, max(0.02, 0.8 * float(np.mean(r)) / math.sqrt(max(nf, 1))))})
        return res

    # ------------------------------------------------------------ checks
    def clearance(self, yarns, fascicles):
        """Measured minimum surface clearance between yarns and against retained fascicles."""
        out = {}
        if not yarns:
            return out
        pts, rad, owner, unit = [], [], [], []
        for y in yarns:
            pts.append(y["P"])
            rad.append(y["r"])
            owner.append(np.full(y["P"].shape[0], y["id"]))
            unit.append(np.full(y["P"].shape[0], y["unit"]))
        P = np.vstack(pts)
        R = np.concatenate(rad)
        O = np.concatenate(owner)
        Un = np.concatenate(unit)
        tree = cKDTree(P)
        k = min(24, P.shape[0])
        d, idx = tree.query(P, k=k)
        other = O[idx] != O[:, None]
        gap = np.where(other, d - R[:, None] - R[idx], np.inf)
        g = gap.min(1)
        i = int(np.argmin(g))
        out["yarn_min_clearance_mm"] = float(g[i])
        out["yarn_min_clearance_at"] = P[i].tolist()
        out["yarn_collisions"] = int((g < -0.01).sum())
        out["yarn_bad_ids"] = sorted(set(int(x) for x in O[g < -0.01]))[:200]
        self._sample_yarn_gap = g
        cross = other & (Un[idx] != Un[:, None])
        if cross.any():
            gc = np.where(cross, d - R[:, None] - R[idx], np.inf).min(1)
            out["unit_min_clearance_mm"] = float(gc.min())
            out["unit_collisions"] = int((gc < -0.01).sum())
        fp, fr = [], []
        L = self.lig.L
        units = {u.id: u for u in self.units}
        bt = any(u.bone_tib for u in self.units)
        bf = any(u.bone_fem for u in self.units)
        lo_max = max(u.ext["lo_max"] for u in self.units)
        hi_min = min(u.ext["hi_min"] for u in self.units)
        for fa in fascicles:
            keep = ~fa["damaged"]
            sg = fa["s"] * L
            if fa["affected"] and not fa["bridging"]:
                if bt:
                    keep &= sg > lo_max
                if bf:
                    keep &= sg < hi_min
            if keep.any():
                fp.append(fa["P"][keep])
                fr.append(fa["r"][keep])
        if fp:
            FP = np.vstack(fp)
            FR = np.concatenate(fr)
            ft = cKDTree(FP)
            d2, j = ft.query(P)
            g2 = d2 - R - FR[j]
            self._sample_tissue_gap = g2
            out["tissue_min_clearance_mm"] = float(g2.min())
            out["tissue_collisions"] = int((g2 < -0.01).sum())
            out["tissue_collision_fraction"] = float((g2 < -0.01).mean())
        return out

    def summary(self, yarns):
        if not self.active:
            return {}
        L = self.lig.L
        units = []
        df = float(self.p["scaffold.fiber_diameter_um"]) / 1000
        for u in self.units:
            regions = []
            bounds = {"tibial_interface": (u.sig_a, u.b1), "bridge": (u.b1, u.b2), "femoral_interface": (u.b2, u.sig_b)}
            ny = sum(1 for y in yarns if y["unit"] == u.id)
            areas = [f.area("Danat", 0.0) for f in u.fields]
            for rid in REGION_IDS:
                rg = self.reg[rid]
                a, b = bounds[rid]
                vf = rg["vf"]
                pore_um = df * (math.sqrt(math.pi / (2 * math.sqrt(3) * vf)) - 1) * 1000
                d_y = self.d0 * rg["scale"]
                regions.append({"id": rid, "kind": u.region_kind[rid], "start_mm": a, "end_mm": b,
                                "length_mm": max(0.0, b - a), "material": rg["material"],
                                "material_label": MATERIALS[rg["material"]]["label"],
                                "composition": rg["composition"], "fiber_volume_fraction": vf,
                                "porosity_within_yarn": 1 - vf, "inter_fiber_pore_um": pore_um,
                                "yarn_diameter_mm": d_y, "alignment_dispersion_deg": rg["disp"],
                                "transition_mm": rg["tw"], "transition_mode": rg["mode"],
                                "fibers_per_yarn_physical": vf * (d_y / df) ** 2, "unit": u.id})
            units.append({"id": u.id, "label": u.label, "start_mm": u.sig_a, "end_mm": u.sig_b, "length_mm": u.sig_b - u.sig_a,
                          "braid_span_mm": [u.braid_a, u.braid_b], "start_s": u.sig_a / L, "end_s": u.sig_b / L,
                          "anchored_to_bone": {"tibial": bool(u.bone_tib), "femoral": bool(u.bone_fem)},
                          "interface_side": None if u.iface_side is None else BUNDLE_NAMES[u.iface_side],
                          "repair_area_mm2": {"min": min(areas) if areas else 0, "max": max(areas) if areas else 0},
                          "yarns": ny, "regions": regions, "sizing": getattr(u, "sizing", None),
                          "core": getattr(u, "core_info", None), "layout": getattr(u, "layout_info", None)})
        regions_all = units[0]["regions"] if units else []
        return {"units": units, "regions": regions_all, "architecture": self.arch, "engagement": self.engagement,
                "fill_fraction": self.fill, "braid": self.braid_info, "yarn_count": len(yarns),
                "start_mm": min(u.sig_a for u in self.units), "end_mm": max(u.sig_b for u in self.units),
                "clearances_mm": {"boundary": self.c_env, "interface": self.c_if, "yarn": self.clr}}


def _min_concave_radius(r, phs):
    """Smallest radius of curvature over the concave parts of a polar curve r(phi)."""
    dp = phs[1] - phs[0]
    r1 = (np.roll(r, -1) - np.roll(r, 1)) / (2 * dp)
    r2 = (np.roll(r, -1) - 2 * r + np.roll(r, 1)) / (dp * dp)
    kappa = (r * r + 2 * r1 * r1 - r * r2) / np.maximum((r * r + r1 * r1) ** 1.5, 1e-12)
    neg = kappa < 0
    if not neg.any():
        return np.inf
    return float(1.0 / (-kappa[neg]).max())


def _bend_radius_for(s_arc, delta, need, lo, hi):
    """Smallest bend radius rho so that two yarns s_arc apart along the tube, both
    offset inward by delta, stay >= need apart (chord on the inner offset)."""
    def ok(rho):
        return 2 * (rho - delta) * math.sin(min(s_arc / (2 * rho), math.pi / 2)) >= need
    if ok(lo):
        return lo
    if not ok(hi):
        return hi
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return hi


def _min_gap(ys, clr):
    """Smallest measured surface gap (minus clearance) between different yarns."""
    if len(ys) < 2:
        return float("inf")
    P = np.vstack([y["P"] for y in ys])
    R = np.concatenate([y["r"] for y in ys])
    O = np.concatenate([np.full(y["P"].shape[0], i) for i, y in enumerate(ys)])
    d, idx = cKDTree(P).query(P, k=min(16, P.shape[0]))
    g = np.where(O[idx] != O[:, None], d - R[:, None] - R[idx], np.inf)
    return float(g.min())


def _perimeter(c, Rrow, phs):
    X = c[None] + Rrow[:, None] * np.stack([np.cos(phs), np.sin(phs)], 1)
    return float(np.linalg.norm(np.roll(X, -1, 0) - X, axis=1).sum())


def _samp(A, f, X):
    i = (X[:, 0] - f.xs[0]) / f.h
    j = (X[:, 1] - f.ys[0]) / f.h
    return ndimage.map_coordinates(A, [i, j], order=1, mode="constant", cval=-5.0)


def _inward_smooth(R, w, axis, wrap):
    """Min-filter then moving average with the same half-width: the result never
    exceeds the input (a moving average of a w-min-filtered signal is <= it)."""
    mode = "wrap" if wrap else "nearest"
    m = ndimage.minimum_filter1d(R, size=2 * w + 1, axis=axis, mode=mode)
    return ndimage.uniform_filter1d(m, size=2 * w + 1, axis=axis, mode=mode)
