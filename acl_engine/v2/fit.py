"""Fit verification: does the scaffold actually fit its repair space?

Everything here is measured on the FULL 3D scaffold (every yarn sample, with
its real strand radius), not on a few displayed sections:

- envelope containment: signed distance of each strand surface to the intact
  ACL envelope (dense surface samples with outward normals; point-to-plane
  near the surface). Volume outside = sum over samples of the strand-disk
  area beyond the surface x sample length.
- repair-space containment: each sample is located in its station's section
  plane and tested against that station's repair-space field (contour, AM/PL
  interface for single-bundle units, sector/mask/depth edge) and against the
  permitted longitudinal span (damaged interval + permitted overlap).
- bone, other yarns (incl. the other repair unit), retained fascicles.
- clearance shortfalls (fits, but closer than requested) are reported apart
  from geometric violations.

Per-sample violation bits are returned for the viewer:
  1 outside envelope   2 outside repair region (e.g. into the unaffected bundle)
  4 beyond the permitted overlap   8 bone   16 yarn-yarn   32 retained tissue
  64 clearance shortfall (warning)
"""
import math
import numpy as np
from scipy.spatial import cKDTree

TOL = 0.02          # mm, geometric tolerance for "outside"
VBITS = {"outside_envelope": 1, "outside_region": 2, "beyond_overlap": 4, "bone": 8,
         "yarn": 16, "tissue": 32, "clearance": 64}


class EnvelopeSDF:
    """Signed distance (mm, negative inside) to the intact ligament envelope."""

    def __init__(self, lig, n_ring=128):
        ph = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
        q = np.stack([np.cos(ph), np.sin(ph)], 1)
        n = lig.s.size
        S = np.repeat(lig.s, n_ring)
        R = lig.map(S, np.tile(q, (n, 1))).reshape(n, n_ring, 3)
        ds = np.gradient(R, axis=0)
        dp = np.roll(R, -1, 1) - np.roll(R, 1, 1)
        nrm = np.cross(dp, ds)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=2, keepdims=True), 1e-12)
        cen = R.mean(1, keepdims=True)
        sgn = np.sign(((R - cen) * nrm).sum(2, keepdims=True))
        nrm *= np.where(sgn == 0, 1, sgn)
        pts = [R.reshape(-1, 3)]
        nrs = [nrm.reshape(-1, 3)]
        # end caps (attachment faces): disk samples, normals pointing out of the ligament
        from acl_engine.v2.ligament import sunflower
        D = sunflower(500, 0.98)
        for sv, sg in ((0.0, -1.0), (1.0, 1.0)):
            P = lig.map(np.full(D.shape[0], sv), D)
            T = lig.T[0 if sv == 0 else -1]
            pts.append(P)
            nrs.append(np.repeat((sg * T)[None], P.shape[0], 0))
        self.P = np.vstack(pts)
        self.N = np.vstack(nrs)
        self.tree = cKDTree(self.P)

    def sdf(self, X):
        d, i = self.tree.query(X, k=4)
        # weighted point-to-plane over the 4 nearest samples (smooth across samples)
        w = 1.0 / np.maximum(d, 1e-6)
        dp = ((X[:, None, :] - self.P[i]) * self.N[i]).sum(2)
        plane = (dp * w).sum(1) / w.sum(1)
        far = d[:, 0] > 1.0
        return np.where(far, np.sign(plane) * d[:, 0], plane)


def _seg_area_outside(h, r):
    """Area of a disk of radius r beyond a line at signed distance h from its centre
    (h > 0: centre inside by h)."""
    h = np.clip(h, -r, r)
    return r * r * np.arccos(np.clip(h / np.maximum(r, 1e-12), -1, 1)) - h * np.sqrt(np.maximum(r * r - h * h, 0))


def evaluate(lig, inj, scf, yarns, clear, bones, fascicles=None):
    """Full fit report + per-yarn violation arrays (added to each yarn as y['viol'])."""
    out = {"tolerance_mm": TOL, "units": []}
    if not scf.active or not yarns:
        return out
    L = lig.L
    env = EnvelopeSDF(lig)
    units = {u.id: u for u in scf.units}
    fl_t = float(scf.p["lig.flare_tib_mm"])
    fl_f = float(scf.p["lig.flare_fem_mm"])
    allP = np.vstack([y["P"] for y in yarns])
    allR = np.concatenate([y["r"] for y in yarns])
    sd_all = env.sdf(allP)
    bone_all = np.min([b.sdf(allP) for b in bones], axis=0) if bones else np.full(allP.shape[0], 99.0)
    yg = getattr(scf, "_sample_yarn_gap", None)
    tg = getattr(scf, "_sample_tissue_gap", None)
    off = 0
    tot_vol = 0.0
    vol_env = 0.0
    vol_reg = 0.0
    vol_att = 0.0
    min_env_clear = np.inf
    min_bone_clear = np.inf
    cross_depth = 0.0
    n_cross = 0
    n_beyond = 0
    short_clear = 0
    worst = []
    end_gaps, end_angles = [], []
    for y in yarns:
        u = units[y["unit"]]
        P, r, sig = y["P"], y["r"], y["sig"]
        k = P.shape[0]
        sd = sd_all[off:off + k]
        bd = bone_all[off:off + k]
        seg = np.zeros(k)
        dl = np.linalg.norm(np.diff(P, axis=0), axis=1)
        seg[:-1] += 0.5 * dl
        seg[1:] += 0.5 * dl
        vb = np.zeros(k, np.uint8)
        # permitted attachment interface: bone-anchored ends inside the insertion flare
        att = np.zeros(k, bool)
        if u.bone_tib:
            att |= sig < fl_t
        if u.bone_fem:
            att |= sig > L - fl_f
        # 1) envelope (3D)
        prot = sd + r
        a_out = _seg_area_outside(-sd, r)
        v = a_out * seg
        tot_vol += float((np.pi * r * r * seg).sum())
        vol_env += float(v[~att].sum())
        vol_att += float(v[att].sum())
        bad = (prot > TOL) & ~att
        vb[bad] |= VBITS["outside_envelope"]
        if (~att).any():
            min_env_clear = min(min_env_clear, float((-sd - r)[~att].min()))
        # 2) repair region in the section plane (stations of this unit)
        fsig = np.array([f.sig for f in u.fields])
        j = np.clip(np.searchsorted(fsig, sig), 0, fsig.size - 1)
        j0 = np.clip(j - 1, 0, fsig.size - 1)
        j = np.where(np.abs(fsig[j0] - sig) < np.abs(fsig[j] - sig), j0, j)
        inside_span = (sig >= u.sig_a - r - TOL) & (sig <= u.sig_b + r + TOL)
        beyond = ~inside_span & ~att
        vb[beyond] |= VBITS["beyond_overlap"]
        n_beyond += int(beyond.sum())
        for jj in np.unique(j):
            sel = (j == jj) & inside_span & ~att
            if not sel.any():
                continue
            f = u.fields[jj]
            i_st = int(round(np.clip(f.s, 0, 1) * (lig.s.size - 1)))
            Q = P[sel] - lig.C[i_st]
            X = np.stack([Q @ lig.N1[i_st], Q @ lig.N2[i_st]], 1)
            da = f.sample(X, "Danat")
            db = f.sample(X, "Dbound")
            rr = r[sel]
            out_reg = (da - rr) < -TOL
            idx = np.nonzero(sel)[0]
            vb[idx[out_reg]] |= VBITS["outside_region"]
            vreg = _seg_area_outside(da, rr) * seg[sel]
            vol_reg += float(vreg.sum())
            if u.iface_side is not None:
                sg = 1.0 if u.iface_side == 0 else -1.0
                di = sg * lig.interface_distance(np.full(X.shape[0], f.s), X) - rr
                c = di < -TOL
                n_cross += int(c.sum())
                if c.any():
                    cross_depth = max(cross_depth, float(-di[c].min()))
            shortf = (db - rr) < -TOL
            vb[idx[shortf & ~out_reg]] |= VBITS["clearance"]
            short_clear += int((shortf & ~out_reg).sum())
        # 3) bone
        bc = bd - r
        bb = (bc < -TOL) & ~att
        vb[bb] |= VBITS["bone"]
        if (~att).any():
            min_bone_clear = min(min_bone_clear, float(bc[~att].min()))
        # 4) yarns / tissue
        if yg is not None:
            vb[yg[off:off + k] < -0.01] |= VBITS["yarn"]
        if tg is not None:
            vb[tg[off:off + k] < -0.01] |= VBITS["tissue"]
        y["viol"] = vb
        hard = vb & ~np.uint8(VBITS["clearance"])
        for kk in np.nonzero(hard)[0][:: max(1, int(np.count_nonzero(hard) // 6) or 1)]:
            worst.append({"P": P[kk].tolist(), "bits": int(vb[kk]), "yarn": int(y["id"]), "unit": u.id,
                          "depth_mm": float(max(prot[kk], -bc[kk], 0.0))})
        # interface distances / alignment at abutting ends
        if "end_gaps" in y:
            T = np.gradient(P, axis=0)
            T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
            for e, gap in ((0, y["end_gaps"][0]), (-1, y["end_gaps"][1])):
                if gap is None:
                    continue
                end_gaps.append(float(gap))
                ti = int(round(np.clip(y["s"][e], 0, 1) * (lig.s.size - 1)))
                end_angles.append(float(np.degrees(np.arccos(np.clip(abs(T[e] @ lig.T[ti]), -1, 1)))))
        off += k
    out.update({
        "strand_volume_mm3": tot_vol,
        "volume_outside_envelope_mm3": vol_env,
        "volume_outside_repair_space_mm3": vol_reg,
        "volume_in_attachment_interface_mm3": vol_att,
        "min_envelope_clearance_mm": None if not np.isfinite(min_env_clear) else min_env_clear,
        "min_bone_clearance_mm": None if not np.isfinite(min_bone_clear) else min_bone_clear,
        "interface_crossings": n_cross, "interface_crossing_depth_mm": cross_depth,
        "samples_beyond_overlap": n_beyond, "clearance_shortfall_samples": short_clear,
        "end_gap_mm": ({"min": min(end_gaps), "mean": float(np.mean(end_gaps)), "max": max(end_gaps), "n": len(end_gaps)} if end_gaps else None),
        "end_misalignment_deg": ({"mean": float(np.mean(end_angles)), "max": max(end_angles)} if end_angles else None),
        "violations": worst[:60],
        "unit_min_clearance_mm": clear.get("unit_min_clearance_mm"),
        "unit_collisions": clear.get("unit_collisions", 0),
    })
    # per-unit occupancy and connectivity
    for u in scf.units:
        uy = [y for y in yarns if y["unit"] == u.id]
        occ_s, occ, rep_area, avail_area = [], [], [], []
        for f in u.fields[:: max(1, len(u.fields) // 40)]:
            a_rep = f.area("Danat", 0.0)
            a_av = f.area("Dbound", 0.0)
            strand = 0.0
            i_st = int(round(np.clip(f.s, 0, 1) * (lig.s.size - 1)))
            T0 = lig.T[i_st]
            for y in uy:
                if y["sig"][0] <= f.sig <= y["sig"][-1]:
                    kk = int(np.argmin(np.abs(y["sig"] - f.sig)))
                    k1 = min(kk + 1, y["P"].shape[0] - 1)
                    k0 = max(kk - 1, 0)
                    t = y["P"][k1] - y["P"][k0]
                    c = abs(t @ T0) / max(np.linalg.norm(t), 1e-12)
                    strand += math.pi * float(y["r"][kk]) ** 2 / max(c, 0.2)
            occ_s.append(f.s)
            occ.append(strand / max(a_rep, 1e-9))
            rep_area.append(a_rep)
            avail_area.append(a_av)
        # connectivity: braid yarns of a unit form one interlaced structure?
        comps = None
        pieces = 0
        for y in uy:
            gaps = np.diff(y["sig"])
            pieces += int((gaps > 3 * max(np.median(gaps), 1e-6)).sum()) + 1
        br = [y for y in uy if y["family"] in ("+", "-")]
        if br:
            parent = list(range(len(br)))

            def find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a
            trees = [cKDTree(y["P"]) for y in br]
            for a in range(len(br)):
                for b in range(a + 1, len(br)):
                    lim = float(br[a]["r"].max() + br[b]["r"].max()) + 3 * float(getattr(scf, "clr", 0.03)) + 1.5 * float(br[a]["r"].max())
                    d, _ = trees[b].query(br[a]["P"], distance_upper_bound=lim)
                    if np.isfinite(d).any():
                        parent[find(a)] = find(b)
            comps = len(set(find(a) for a in range(len(br))))
        out["units"].append({"id": u.id, "label": u.label, "occupancy_s": occ_s, "occupancy": occ,
                             "repair_area_mm2": rep_area, "available_area_mm2": avail_area,
                             "yarns": len(uy), "pieces": pieces, "braid_components": comps,
                             "occupancy_mean": float(np.mean(occ)) if occ else 0.0})
    return out
