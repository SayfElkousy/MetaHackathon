"""Injury mask and native tissue (retained fascicles/fibers, remnant surfaces).

The injury is a SPATIAL REGION evaluated per point (arc position sigma in mm,
material coordinates m, 3D position P):

    damaged = (lo(m, P) < sigma < hi(m, P))  and  in_affected_region(m)

lo/hi come from a tilted, irregular tear plane (obliquity relative to the local
tangent, azimuth from the frame's anterior axis), the missing-tissue length and
the stump retractions. When the MRI adapter supplies an observed boundary
table (per section angle), it replaces the plane model. The affected region is
the whole section, one bundle, or a sector (centre, width, depth).

Retained tissue = everything outside the damaged region. A fraction of the
affected fascicles can be kept continuous (residual continuity).
"""
import math
import numpy as np
from scipy.spatial import cKDTree
from acl_engine.v2 import mesher
from acl_engine.v2.design import fascicle_sizing, PACKING_LIMIT
from acl_engine.v2.ligament import (rot2, sunflower, layout_points, fiber_offsets, profile_eval, _unit)


_MASK_CACHE = {}


class Injury:
    def __init__(self, p, lig, assets=None):
        self.p = p
        self.lig = lig
        self.type = p["injury.type"]
        self.active = self.type != "intact"
        L = lig.L
        self.sig0 = float(p["injury.center_s"]) * L
        self.half = 0.5 * float(p["injury.gap_mm"])
        self.rt = float(p["injury.retraction_tib_mm"])
        self.rf = float(p["injury.retraction_fem_mm"])
        self.tan_obl = math.tan(math.radians(float(p["injury.obliquity_deg"])))
        az = math.radians(float(p["injury.azimuth_deg"]))
        self.az = az
        self.irr = float(p["injury.irregularity_mm"])
        lobes = int(p["injury.irregularity_lobes"])
        rng = np.random.default_rng(int(p["injury.seed"]))
        self.terms = []
        for face in range(2):
            fr = [lobes, 2 * lobes + 1, max(1, lobes // 2) + 1]
            amp = np.array([0.6, 0.25, 0.35])
            ph = rng.uniform(0, 2 * np.pi, (3, 2))
            self.terms.append((fr, amp / amp.sum(), ph))
        self.inv = p["injury.involvement"] if self.type == "partial" else "whole"
        # per-bundle tears: each bundle has its own position, length and depth
        self.pb = None
        if self.inv == "per_bundle":
            self.pb = []
            for key, b in (("am", 0), ("pl", 1)):
                cen, _ = lig.bundle_geometry(b)
                self.pb.append({"sig0": float(p["injury.%s.center_s" % key]) * L, "half": 0.5 * float(p["injury.%s.gap_mm" % key]),
                                "depth": float(p["injury.%s.depth_fraction" % key]), "cen": cen})
        self.theta_tear = float(lig._interp(lig.theta_b, np.array([p["injury.center_s"]]))[0])
        self.table = (assets or {}).get("injury_table")
        self.mask = None
        if (assets or {}).get("injury_mask") and self.active:
            a = assets["injury_mask"]
            key = (a["path"], str(a["voxel_to_model"]))
            if key not in _MASK_CACHE:
                M = np.load(a["path"]) > 0.5
                from scipy import ndimage
                M = ndimage.binary_closing(M, iterations=1)
                T = np.asarray(a["voxel_to_model"], float)
                _MASK_CACHE[key] = (M, np.linalg.inv(T))
            self.mask = _MASK_CACHE[key]
            from acl_engine.v2.patient import mask_boundary_table
            self.table_summary = mask_boundary_table(a, lig, self.theta_tear)
            # sigma range that contains damaged voxels (for line sampling)
            self.mask_sig = (min(self.table_summary["lo_mm"]) - 1.0, max(self.table_summary["hi_mm"]) + 1.0) if self.table_summary else (0, L)
        self.status = "observed" if (self.table or self.mask is not None) else "assumed"

    def _rag(self, face, m):
        fr, amp, ph = self.terms[face]
        ang = np.arctan2(m[:, 1], m[:, 0])
        rho = np.clip(np.hypot(m[:, 0], m[:, 1]), 0, 1)
        r = np.zeros(m.shape[0])
        for k in range(3):
            r += amp[k] * (np.sin(fr[k] * ang + ph[k, 0]) * (0.55 + 0.45 * rho) + 0.3 * np.sin(3.1 * rho + ph[k, 1]))
        return self.irr * r

    def _mask_hit(self, P):
        M, Ti = self.mask
        idx = np.rint(P @ Ti[:3, :3].T + Ti[:3, 3]).astype(int)
        ok = np.all((idx >= 0) & (idx < np.array(M.shape)), axis=1)
        out = np.zeros(P.shape[0], bool)
        out[ok] = M[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
        return out

    def _mask_line(self, m, n=48):
        """Damaged interval along each material line (min/max sigma of mask hits)."""
        lig = self.lig
        a, b = self.mask_sig
        sig = np.linspace(max(a, 0), min(b, lig.L), n)
        k = m.shape[0]
        S = np.tile(sig / lig.L, k)
        Mm = np.repeat(m, n, axis=0)
        hit = self._mask_hit(lig.map(S, Mm)).reshape(k, n)
        lo = np.where(hit.any(1), sig[np.argmax(hit, 1)], np.inf)
        hi = np.where(hit.any(1), sig[n - 1 - np.argmax(hit[:, ::-1], 1)], -np.inf)
        return lo, hi, hit.any(1)

    def bounds(self, s, m, P):
        """Lower/upper arc positions (mm) of the damaged interval for each point."""
        lig = self.lig
        if self.mask is not None:
            # unique material lines -> mask intervals
            key = np.round(m, 4)
            u, inv = np.unique(key, axis=0, return_inverse=True)
            lo, hi, _ = self._mask_line(u)
            return lo[inv.ravel()], hi[inv.ravel()]
        if self.table:
            q = rot2(m, self.theta_tear)
            ang = np.degrees(np.arctan2(q[:, 1], q[:, 0]))
            t = self.table
            a = np.asarray(t["angles_deg"], float)
            order = np.argsort(a)
            a = a[order]
            lo = np.interp(ang, a, np.asarray(t["lo_mm"], float)[order], period=360)
            hi = np.interp(ang, a, np.asarray(t["hi_mm"], float)[order], period=360)
            return lo, hi
        C = lig._interp(lig.C, s)
        N1 = lig._interp(lig.N1, s)
        N2 = lig._interp(lig.N2, s)
        d = math.cos(self.az) * N1 + math.sin(self.az) * N2
        tilt = self.tan_obl * ((P - C) * d).sum(1)
        sig0, half = self.sig0, self.half
        if self.pb is not None:
            b = lig.bundle_of(m)
            sig0 = np.where(b == 0, self.pb[0]["sig0"], self.pb[1]["sig0"])
            half = np.where(b == 0, self.pb[0]["half"], self.pb[1]["half"])
        lo = sig0 - half - self.rt + tilt + self._rag(0, m)
        hi = sig0 + half + self.rf + tilt + self._rag(1, m)
        hi = np.maximum(hi, lo)
        return lo, hi

    def in_region(self, m):
        if self.mask is not None:
            key = np.round(m, 4)
            u, inv = np.unique(key, axis=0, return_inverse=True)
            return self._mask_line(u)[2][inv.ravel()]
        if self.table:
            # observed mask: only section angles that contain damaged voxels are affected
            q = rot2(m, self.theta_tear)
            ang = np.degrees(np.arctan2(q[:, 1], q[:, 0]))
            a = np.asarray(self.table["angles_deg"], float)
            bw = 360.0 / 36
            d = np.abs((ang[:, None] - a[None, :] + 180) % 360 - 180).min(1)
            return d <= 0.75 * bw
        if self.inv == "whole":
            return np.ones(m.shape[0], bool)
        if self.inv in ("AM", "PL"):
            return self.lig.bundle_of(m) == (0 if self.inv == "AM" else 1)
        if self.inv == "per_bundle":
            b = self.lig.bundle_of(m)
            out = np.zeros(m.shape[0], bool)
            for k in (0, 1):
                dep = self.pb[k]["depth"]
                sel = b == k
                if dep <= 0 or not sel.any():
                    continue
                cen = self.pb[k]["cen"]
                ph = np.arctan2(m[sel, 1] - cen[1], m[sel, 0] - cen[0])
                rb = self.lig.bundle_boundary_radius(cen, ph, k)
                rho = np.linalg.norm(m[sel] - cen, axis=1) / np.maximum(rb, 1e-9)
                out[sel] = rho >= 1 - dep - 1e-9
            return out
        q = rot2(m, self.theta_tear)
        ang = np.degrees(np.arctan2(q[:, 1], q[:, 0]))
        c = float(self.p["injury.sector_center_deg"])
        w = float(self.p["injury.sector_width_deg"])
        dang = (ang - c + 180) % 360 - 180
        rho = np.hypot(m[:, 0], m[:, 1])
        return (np.abs(dang) <= w / 2) & (rho >= 1 - float(self.p["injury.depth_fraction"]) - 1e-9)

    def damaged(self, s, m, P):
        if not self.active:
            return np.zeros(m.shape[0], bool)
        if self.mask is not None:
            return self._mask_hit(P)
        sig = np.asarray(s) * self.lig.L
        lo, hi = self.bounds(s, m, P)
        return (sig > lo) & (sig < hi) & self.in_region(m)

    def region_extent(self, sel_fn=None):
        """lo/hi statistics over the affected region (sampled at the tear).
        sel_fn optionally restricts the material points (e.g. one bundle)."""
        M0 = sunflower(1500)
        M = M0[self.in_region(M0)]
        if sel_fn is not None:
            M = M[sel_fn(M)]
        if M.shape[0] == 0:
            return None
        sig_c = self.sig0
        if self.pb is not None:
            b = self.lig.bundle_of(M)
            sig_c = np.where(b == 0, self.pb[0]["sig0"], self.pb[1]["sig0"])
        s = np.broadcast_to(np.asarray(sig_c, float) / self.lig.L, (M.shape[0],))
        P = self.lig.map(s, M)
        lo, hi = self.bounds(s, M, P)
        return {"lo_min": float(lo.min()), "lo_mean": float(lo.mean()), "lo_max": float(lo.max()),
                "hi_min": float(hi.min()), "hi_mean": float(hi.mean()), "hi_max": float(hi.max()),
                "area_fraction": float(M.shape[0] / 1500.0)}


# ---------------------------------------------------------------- layouts in mm
def metric_layout(lig, s_ref, pattern, n, region_fn, r_mm, seed, centroid, margin_fn_mm=None, spacing_mm=None, metric_region_fn=None):
    """Lay out up to n circles of radius r_mm inside region_fn (material coords),
    using true millimetre distances in the section contour at station s_ref.

    margin_fn_mm(X) optionally returns an extra available-distance (mm) for metric
    points X (used for constraints checked at other stations). Returns material
    coordinates of the centres."""
    from scipy import ndimage
    poly = lig.contour_at(s_ref)
    lo = poly.min(0) - 0.3
    hi = poly.max(0) + 0.3
    hpx = max(0.015, float((hi - lo).min()) / 150.0)
    xs = np.arange(lo[0], hi[0] + hpx, hpx)
    ys = np.arange(lo[1], hi[1] + hpx, hpx)
    GX, GY = np.meshgrid(xs, ys, indexing="ij")
    G = np.stack([GX.ravel(), GY.ravel()], 1)
    from acl_engine.v2.ligament import points_in_polygon
    inside = points_in_polygon(poly, G)
    mask = np.zeros(G.shape[0], bool)
    if inside.any():
        if metric_region_fn is not None:
            mask[inside] = metric_region_fn(G[inside])
        else:
            Mg = lig.metric_to_material(np.full(int(inside.sum()), s_ref), G[inside])
            mask[inside] = region_fn(Mg)
    mask = mask.reshape(GX.shape)
    edt = ndimage.distance_transform_edt(mask) * hpx
    if margin_fn_mm is not None:
        extra = np.full(G.shape[0], -1.0)
        sel = mask.ravel()
        if sel.any():
            extra[sel] = margin_fn_mm(G[sel])
        edt = np.minimum(edt, extra.reshape(GX.shape))
    scale = float(max(hi - lo)) / 2.0

    def lookup(Pn, arr, fill):
        X = Pn * scale
        i = np.rint((X[:, 0] - xs[0]) / hpx).astype(int)
        j = np.rint((X[:, 1] - ys[0]) / hpx).astype(int)
        ok = (i >= 0) & (i < xs.size) & (j >= 0) & (j < ys.size)
        out = np.full(Pn.shape[0], fill, dtype=arr.dtype)
        out[ok] = arr[i[ok], j[ok]]
        return out
    inside_fn = lambda Pn: lookup(Pn, mask, False)
    margin = lambda Pn: lookup(Pn, edt, 0.0) >= r_mm
    c = lig.material_to_metric(np.array([s_ref]), np.asarray(centroid, float)[None])[0] / scale
    sp = (2 * r_mm if spacing_mm is None else spacing_mm) / scale
    X = layout_points(pattern, n, inside_fn, margin, seed, c, min_spacing=sp)
    if not X.shape[0]:
        return np.zeros((0, 2))
    return lig.metric_to_material(np.full(X.shape[0], s_ref), X * scale)


# ---------------------------------------------------------------- native tissue
def split_runs(mask):
    """Index runs where mask is True: list of (start, stop)."""
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    st = np.nonzero(d == 1)[0]
    en = np.nonzero(d == -1)[0]
    return list(zip(st, en))


class NativeTissue:
    def __init__(self, p, lig, injury, display):
        self.p, self.lig, self.inj = p, lig, injury
        L = lig.L
        self.area_mid = float(lig.section_area(0.5)[0])
        n, d, phi = fascicle_sizing(p, self.area_mid)
        self.n_fasc, self.d_fasc, self.packing = n, d, phi
        semi = math.sqrt(self.area_mid / math.pi)        # equivalent radius at mid-length
        self.semi_mid = semi
        indep = p["bundle.structure_mode"] == "independent"
        self.independent = indep
        a_b = [float(lig.bundle_area(0.5, b)[0]) for b in (0, 1)]
        self.d_b, counts = [], []
        for b, key in ((0, "am"), (1, "pl")):
            if indep:
                db = d * float(p[key + ".fascicle_diameter_scale"])
                nb = max(1, int(round(phi * a_b[b] / (math.pi * db * db / 4))))
            else:
                db = d
                nb = max(1, int(round(n * a_b[b] / max(sum(a_b), 1e-9))))
            self.d_b.append(db)
            counts.append(nb)
        pattern = p["native.packing_pattern"]
        seed = int(p["native.layout_seed"])
        self.fasc_m0, self.fasc_bundle, self.fasc_polar, self.bundle_info = [], [], [], []
        for b in (0, 1):
            cen, _ = lig.bundle_geometry(b)
            region = lambda M, b=b: (lig.bundle_of(M) == b) & (np.hypot(M[:, 0], M[:, 1]) < 1)
            mreg = lambda X, b=b: lig.metric_bundle(np.full(X.shape[0], 0.5), X) == b
            pts = metric_layout(lig, 0.5, pattern, counts[b], region, self.d_b[b] / 2, seed + 17 * b, cen, metric_region_fn=mreg)
            ph = np.arctan2(pts[:, 1] - cen[1], pts[:, 0] - cen[0])
            rb = lig.bundle_boundary_radius(cen, ph, b)
            rho = np.linalg.norm(pts - cen, axis=1) / np.maximum(rb, 1e-9)
            rbm = math.sqrt(a_b[b] / math.pi)
            self.bundle_info.append({"centroid": cen, "radius_mm": rbm, "count": int(pts.shape[0]), "requested": counts[b],
                                     "diameter_mm": self.d_b[b], "area_mid_mm2": a_b[b],
                                     "packing": int(pts.shape[0]) * math.pi * self.d_b[b] ** 2 / 4 / max(a_b[b], 1e-9)})
            for k in range(pts.shape[0]):
                self.fasc_m0.append(pts[k])
                self.fasc_bundle.append(b)
                self.fasc_polar.append((cen, ph[k], rho[k]))
        self.fasc_m0 = np.array(self.fasc_m0).reshape(-1, 2)
        self.fasc_bundle = np.array(self.fasc_bundle, np.int8)
        self.fasc_d = np.array([self.d_b[b] for b in self.fasc_bundle])
        nf = self.fasc_m0.shape[0]
        # smallest surface gap between neighbouring fascicles at mid-length
        if nf > 1:
            X = lig.map(np.full(nf, 0.5), self.fasc_m0)
            dd, jj = cKDTree(X).query(X, k=min(6, nf))
            gaps = dd[:, 1:] - 0.5 * (self.fasc_d[:, None] + self.fasc_d[jj[:, 1:]])
            self.min_gap_mm = float(gaps.min())
        else:
            self.min_gap_mm = float("inf")
        # within-bundle rotation: a TOTAL end-to-end angle (surgeon-informed range -30..30 deg),
        # spread linearly along the bundle (monotone), relative to the bundle region
        if indep:
            rb = [float(p["am.fascicle_rotation_deg"]), float(p["pl.fascicle_rotation_deg"])]
            rf = [float(p["am.fiber_rotation_deg"]), float(p["pl.fiber_rotation_deg"])]
        else:
            rb = [float(p["native.fascicle_rotation_deg"])] * 2
            rf = [float(p["native.fiber_rotation_deg"])] * 2
        self.rotation_requested_deg = {"bundle": rb, "fiber": rf}
        self.gamma_tot = [math.radians(v) for v in rb]
        self.fiber_tot = [math.radians(v) for v in rf]
        # rotation schedule per bundle (material bundle-polar angle vs s); corrected below so the
        # rotation MEASURED relative to the parent region is the requested linear one
        self.gamma_sched = [g * lig.s for g in self.gamma_tot]
        # derived outermost-fascicle angle for each bundle
        self.fascicle_angle_deg = [math.degrees(math.atan(abs(g) * bi["radius_mm"] / L)) for g, bi in zip(self.gamma_tot, self.bundle_info)]
        for b in (0, 1):
            if abs(self.gamma_tot[b]) > 1e-9:
                self._correct_schedule(b)
        # continuity: deterministic subset of fascicles kept intact
        rng = np.random.default_rng(int(p["injury.seed"]) + 991)
        self.exempt = rng.random(nf) < float(p["injury.continuity"])
        self.scale, self.ring_area = lig.ring_scale()
        self.display = display

    def _correct_schedule(self, b, iters=3):
        """The within-bundle rotation is applied in bundle-polar material coordinates;
        the section map stretches angles, so the schedule is corrected (a few fixed-point
        steps) until the rotation measured in millimetres, relative to the parent-only
        paths, follows the requested linear schedule."""
        from acl_engine.v2.rotation import unit_rotation
        lig = self.lig
        s = lig.s
        ks = [k for k in range(self.fasc_m0.shape[0]) if self.fasc_bundle[k] == b]
        if not ks:
            return
        Cb = lig.bundle_centerline(b, s)
        refs = [(lig.map(s, np.repeat(self.fasc_m0[k][None], s.size, 0)), s * lig.L) for k in ks]
        target = self.gamma_tot[b] * s
        grid = np.linspace(0, 1, 60)
        for _ in range(iters):
            paths = [(lig.map(s, self.fascicle_m(k, s)), s * lig.L) for k in ks]
            res = unit_rotation(paths, lambda sg: Cb, 0.4, 2.0, ref_paths=refs, return_series=True)
            if res[0] is None:
                return
            meas = np.interp(s, grid, res[3])
            err = target - meas
            if np.abs(err).max() < math.radians(0.2):
                break
            self.gamma_sched[b] = self.gamma_sched[b] + err

    def fascicle_m(self, k, s):
        cen, ph0, rho = self.fasc_polar[k]
        b = self.fasc_bundle[k]
        ph = ph0 + np.interp(s, self.lig.s, self.gamma_sched[b])
        rb = self.lig.bundle_boundary_radius(cen, ph, b)
        return cen + (rho * rb)[:, None] * np.stack([np.cos(ph), np.sin(ph)], 1)

    def fascicles(self):
        """Fascicle centrelines with retained/damaged split.

        Returns dict with per-fascicle full paths and retained segments."""
        lig = self.lig
        s = lig.s
        out = []
        for k in range(self.fasc_m0.shape[0]):
            m = self.fascicle_m(k, s)
            P = lig.map(s, m)
            dmg = self.inj.damaged(s, m, P) & (not self.exempt[k])
            r = 0.5 * self.fasc_d[k] * self.scale
            out.append({"id": k, "bundle": int(self.fasc_bundle[k]), "s": s, "m": m, "P": P, "r": r,
                        "damaged": dmg, "affected": bool(dmg.any()), "bridging": bool(self.exempt[k] and self.inj.active and self.inj.in_region(m[len(s) // 2:len(s) // 2 + 1]).any())})
        return out

    def fibers(self, fasc, n_rep, exaggeration, s_window=None, n_samples=None):
        """Representative fibers inside each fascicle (vectorized per fascicle)."""
        p, lig = self.p, self.lig
        L = lig.L
        lam = float(p["native.crimp_wavelength_um"]) / 1000.0
        amp = float(p["native.crimp_amplitude_um"]) / 1000.0
        lam_d, amp_d = lam * exaggeration, amp * exaggeration
        if s_window is None:
            s0, s1 = 0.0, 1.0
            if n_samples is None:
                n_samples = int(min(700, max(lig.s.size, math.ceil(L / max(lam_d / 8, 0.05)))))
        else:
            s0, s1 = s_window
            if n_samples is None:
                n_samples = int(min(1500, max(60, math.ceil((s1 - s0) * L / max(lam_d / 10, 0.004)))))
        s = np.linspace(s0, s1, n_samples)
        offs = fiber_offsets(n_rep)
        crimp_prof = profile_eval(p["native.crimp_profile"], s)
        disp_prof = profile_eval(p["native.dispersion_profile"], s)
        if self.independent:
            disp_b = [math.radians(float(p["am.dispersion_deg"])), math.radians(float(p["pl.dispersion_deg"]))]
        else:
            disp_b = [math.radians(float(p["native.dispersion_deg"]))] * 2
        from acl_engine.v2.rotation import rmf_frames
        scale = np.interp(s, lig.s, self.scale)
        sig = s * L
        res = []
        for fa in fasc:
            k = fa["id"]
            disp = disp_b[fa["bundle"]]
            m = self.fascicle_m(k, s)
            C = lig.map(s, m)
            # non-twisting frame following the fascicle centreline (starts on the ligament's N1)
            T, E1, E2 = rmf_frames(C, lig._interp(lig.N1, s[:1])[0])
            r = 0.5 * self.fasc_d[k] * scale
            # fibers rotate rigidly about the fascicle axis by a TOTAL angle over the fascicle's
            # full length (s 0..1), linearly (monotone); a window uses the same absolute angle
            alpha = self.fiber_tot[fa["bundle"]] * s
            rng = np.random.default_rng(int(p["native.layout_seed"]) * 7919 + k)
            nf = offs.shape[0]
            ph_c = rng.uniform(0, 2 * np.pi, nf)
            dir_c = rng.uniform(0, 2 * np.pi, nf)
            ph_d = rng.uniform(0, 2 * np.pi, (nf, 2))
            lam_dis = 4.0  # mm: wavelength of the dispersion drift
            a_dis = math.tan(disp) * lam_dis / (2 * np.pi)
            ang = np.arctan2(offs[:, 1], offs[:, 0])[:, None] + alpha[None, :]
            rad = np.hypot(offs[:, 0], offs[:, 1])[:, None] * r[None, :]
            x = rad * np.cos(ang)
            y = rad * np.sin(ang)
            cr = amp_d * crimp_prof[None, :] * np.sin(2 * np.pi * sig[None, :] / max(lam_d, 1e-6) + ph_c[:, None])
            x = x + cr * np.cos(dir_c)[:, None]
            y = y + cr * np.sin(dir_c)[:, None]
            dr = a_dis * disp_prof[None, :]
            # dispersion: bounded RADIAL wander made of whole half-waves over the fascicle
            # (zero at both ends), so it scatters orientation without adding net rotation
            n1 = max(1, int(round(2 * L / lam_dis)))
            n2 = max(1, int(round(2 * L / (lam_dis * 1.37))))
            w1 = np.sin(np.pi * n1 * s)[None, :]
            w2 = 0.7 * np.sin(np.pi * n2 * s)[None, :]
            # radial only: fibers wander in and out, never around the axis (no back-and-forth rotation)
            rdir = np.arctan2(offs[:, 1], offs[:, 0])[:, None] + alpha[None, :]
            rdir = np.where(rad > 1e-9, rdir, ph_d[:, :1])
            amp_r = dr * (np.cos(ph_d[:, :1]) * w1 + np.cos(ph_d[:, 1:]) * w2)
            x = x + amp_r * np.cos(rdir)
            y = y + amp_r * np.sin(rdir)
            mag = np.hypot(x, y)
            lim = 0.95 * r[None, :]
            f = np.where(mag > lim, lim / np.maximum(mag, 1e-12), 1.0)
            x, y = x * f, y * f
            P = C[None] + x[..., None] * E1[None] + y[..., None] * E2[None]
            dmg = self.inj.damaged(s, m, C) & (not self.exempt[k])
            res.append({"fascicle": k, "bundle": fa["bundle"], "P": P, "damaged": dmg, "s": s, "C": C, "r_fasc": float(np.mean(r)),
                        "r": np.full(nf, max(0.04, 0.8 * 0.5 * self.fasc_d[k] / math.sqrt(max(nf, 1))))})
        return res

    def physical_counts(self):
        d_f = float(self.p["native.fiber_diameter_um"]) / 1000.0
        vf = float(self.p["native.fiber_packing"])
        per = vf * (self.d_fasc / d_f) ** 2
        spacing = d_f * math.sqrt(math.pi / (2 * math.sqrt(3) * vf))
        return {"fibers_per_fascicle_physical": per, "fiber_spacing_um": spacing * 1000,
                "fibers_total_physical": per * self.n_fasc}


def envelope_samples(lig, injury, quality):
    """Interior sample points of the intact envelope + damaged flags."""
    ds = 0.25 if quality == "full" else 0.6
    ns = int(math.ceil(lig.L / ds)) + 1
    nd = 900 if quality == "full" else 200
    disk = np.vstack([sunflower(nd, 0.97), 0.985 * np.stack([np.cos(np.linspace(0, 2 * np.pi, 90, endpoint=False)),
                                                            np.sin(np.linspace(0, 2 * np.pi, 90, endpoint=False))], 1)])
    s = np.linspace(0, 1, ns)
    S = np.repeat(s, disk.shape[0])
    M = np.tile(disk, (ns, 1))
    P = lig.map(S, M)
    D = injury.damaged(S, M, P)
    return S, M, P, D


def intact_loft(lig, m_ring=64, rho=1.0):
    ph = np.linspace(0, 2 * np.pi, m_ring, endpoint=False)
    q = rho * np.stack([np.cos(ph), np.sin(ph)], 1)
    n = lig.s.size
    S = np.repeat(lig.s, m_ring)
    M = np.tile(q, (n, 1))
    R = lig.map(S, M).reshape(n, m_ring, 3)
    V = R.reshape(-1, 3)
    i = np.arange(n - 1)[:, None]
    j = np.arange(m_ring)[None, :]
    a0 = i * m_ring + j
    a1 = i * m_ring + (j + 1) % m_ring
    F = np.concatenate([np.stack([a0, a1 + m_ring, a1], -1).reshape(-1, 3),
                        np.stack([a0, a0 + m_ring, a1 + m_ring], -1).reshape(-1, 3)])
    c0 = V.shape[0]
    V = np.vstack([V, R[0].mean(0), R[-1].mean(0)])
    jj = np.arange(m_ring)
    F = np.vstack([F, np.stack([np.full(m_ring, c0), jj, (jj + 1) % m_ring], 1),
                   np.stack([np.full(m_ring, c0 + 1), (n - 1) * m_ring + (jj + 1) % m_ring, (n - 1) * m_ring + jj], 1)])
    a = mesher.audit(V, F)
    if a["volume_mm3"] < 0:
        F = F[:, ::-1]
    return V, F.astype(np.int64)
