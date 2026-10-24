"""Investigational scaffold candidate generation (research geometry only).

Geometry is defined implicitly on a fine world-aligned grid in the injured knee's
RAS mm frame, then meshed with marching cubes. Components:

  envelope   (EvidenceLevel.ESTIMATED)   per-station ellipse fit to the approved ACL estimate
  span       bridging region between remnant tips (+ user overlap) or full envelope,
             always limited by attachment-preservation margins unless an enthesis
             transition is explicitly requested and justified
  channels   (EvidenceLevel.HYPOTHESIZED) internal guidance pathways from a user-chosen
             alignment pattern ("hypothesized internal guidance architecture").
             These are NOT recovered fibres.

Nothing here selects "optimal" pore/channel sizes, materials, degradation, or
biologics. Every such quantity is a required user input recorded as an assumption.
"""
from __future__ import annotations

import dataclasses
from typing import Literal

import numpy as np
from scipy import ndimage

from ..annotation.schema import Label
from ..imaging.coords import apply_affine
from ..provenance import EvidenceLevel


@dataclasses.dataclass
class ScaffoldParams:
    mode: Literal["bridge_gap", "full_envelope"] = "bridge_gap"
    envelope_scale: float = 1.0  # multiplies estimated cross-section semi-axes
    overlap_mm: float = 3.0  # extension of the bridge onto each remnant (user design choice)
    remnant_clearance_mm: float = 0.3  # remnant tissue + this clearance is excluded from the scaffold
    bone_clearance_mm: float = 0.5
    preserve_attachments: bool = True
    attachment_margin_mm: float = 2.0  # scaffold kept this far from bone at the ends when preserving attachments
    enthesis_transition: bool = False
    enthesis_justification: str = ""  # required if enthesis_transition
    alignment_pattern: Literal["none", "parallel", "twisted"] = "parallel"
    twist_deg: float = 0.0  # total twist over the span for "twisted" (hypothesis parameter)
    channel_diameter_mm: float | None = None  # REQUIRED for channels; no default "optimal" value exists
    n_channels_max: int = 19
    min_wall_mm: float | None = None  # REQUIRED manufacturing constraint for channel layouts
    min_feature_mm: float | None = None  # fabrication resolution; used in checks only
    grid_res_mm: float = 0.2
    n_stations: int = 24
    assumption_sources: dict = dataclasses.field(default_factory=dict)  # {param: "citation or 'user assumption: ...'"}

    def validate(self) -> list[str]:
        errs = []
        if self.alignment_pattern != "none":
            if self.channel_diameter_mm is None:
                errs.append("channel_diameter_mm must be supplied (no default 'optimal' size exists)")
            if self.min_wall_mm is None:
                errs.append("min_wall_mm (manufacturing constraint) must be supplied for channel layouts")
            for k in ("channel_diameter_mm", "min_wall_mm"):
                if getattr(self, k) is not None and k not in self.assumption_sources:
                    errs.append(f"assumption_sources[{k!r}] must cite a source or state the assumption explicitly")
        if self.enthesis_transition and not self.enthesis_justification.strip():
            errs.append("enthesis_transition requires an explicit justification (attachment interface actually relevant?)")
        if self.grid_res_mm <= 0 or self.grid_res_mm > 0.5:
            errs.append("grid_res_mm must be in (0, 0.5]")
        return errs


@dataclasses.dataclass
class StationModel:
    fp_f: np.ndarray
    fp_t: np.ndarray
    u: np.ndarray  # unit axis femoral -> tibial
    e1: np.ndarray
    e2: np.ndarray
    L: float
    t: np.ndarray
    center: np.ndarray  # (n,3)
    a: np.ndarray
    b: np.ndarray
    theta: np.ndarray  # ellipse major-axis angle in (e1, e2)

    def at(self, tq):
        tq = np.clip(tq, self.t[0], self.t[-1])
        c = np.stack([np.interp(tq, self.t, self.center[:, i]) for i in range(3)], -1)
        # unwrap angle for interpolation (ellipse axes are pi-periodic)
        th = np.interp(tq, self.t, self.theta)
        return c, np.interp(tq, self.t, self.a), np.interp(tq, self.t, self.b), th

    def to_dict(self):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in dataclasses.asdict(self).items()}


def fit_stations(mask: np.ndarray, affine: np.ndarray, fp_f, fp_t, n: int = 24, smooth: int = 2) -> StationModel:
    P = apply_affine(affine, np.argwhere(mask))
    fp_f, fp_t = np.asarray(fp_f, float), np.asarray(fp_t, float)
    ax = fp_t - fp_f
    L = float(np.linalg.norm(ax))
    u = ax / L
    tmp = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(u, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    tP = ((P - fp_f) @ u) / L
    edges = np.linspace(tP.min(), tP.max(), n + 1)
    ts, cs, As, Bs, Ths = [], [], [], [], []
    prev = None
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (tP >= lo) & (tP < hi)
        if k.sum() < 8:
            continue
        Q = P[k]
        tm = (lo + hi) / 2
        c = Q.mean(0)
        c = c - ((c - fp_f) @ u - tm * L) * u  # project centre onto the station plane
        d = Q - c
        uv = np.stack([d @ e1, d @ e2], 1)
        cov = np.cov(uv.T)
        w, V = np.linalg.eigh(cov)
        major = V[:, 1]
        if prev is not None and major @ prev < 0:
            major = -major
        prev = major
        ts.append(tm)
        cs.append(c)
        As.append(2 * np.sqrt(max(w[1], 1e-6)))  # uniform ellipse: var = a^2/4
        Bs.append(2 * np.sqrt(max(w[0], 1e-6)))
        Ths.append(np.arctan2(major[1], major[0]))
    if len(ts) < 4:
        raise ValueError("anatomy estimate too small / fragmented to fit a cross-sectional profile")
    ts, cs, As, Bs, Ths = map(np.array, (ts, cs, As, Bs, Ths))
    Ths = np.unwrap(Ths * 2) / 2
    if smooth:
        k = np.ones(2 * smooth + 1) / (2 * smooth + 1)
        pad = lambda x: np.pad(x, smooth, mode="edge")  # noqa: E731
        As, Bs, Ths = (np.convolve(pad(x), k, "valid") for x in (As, Bs, Ths))
        cs = np.stack([np.convolve(pad(cs[:, i]), k, "valid") for i in range(3)], 1)
    return StationModel(fp_f, fp_t, u, e1, e2, L, ts, cs, As, Bs, Ths)


def channel_layout(params: ScaffoldParams, sm: StationModel, t0: float, t1: float) -> tuple[np.ndarray, list[str]]:
    """Channel positions in normalised ellipse coordinates (rho, phi), hexagonal rings.
    Channels are dropped until the narrowest station satisfies min_wall_mm."""
    notes = []
    if params.alignment_pattern == "none":
        return np.zeros((0, 2)), notes
    r = params.channel_diameter_mm / 2
    w = params.min_wall_mm
    span = (sm.t >= t0) & (sm.t <= t1)
    a_min = float((sm.a[span] if span.any() else sm.a).min() * params.envelope_scale)
    b_min = float((sm.b[span] if span.any() else sm.b).min() * params.envelope_scale)
    total = np.radians(params.twist_deg) if params.alignment_pattern == "twisted" else 0.0
    twists = np.linspace(0, total, 13) if total else np.array([0.0])
    cands = [(0.0, 0.0)]
    for ring in range(1, 4):
        k = 6 * ring
        rho = ring / 3.5
        cands += [(rho, 2 * np.pi * i / k) for i in range(k)]
    chosen = []
    for rho, phi in cands:
        x, y = a_min * rho * np.cos(phi), b_min * rho * np.sin(phi)
        # wall to the outer surface (approximate, along the ellipse's smaller radius in that direction)
        rad_dir = 1.0 / np.sqrt((np.cos(phi) / a_min) ** 2 + (np.sin(phi) / b_min) ** 2) if rho > 0 else min(a_min, b_min)
        outer_wall = rad_dir * (1 - rho) - r if rho > 0 else min(a_min, b_min) - r
        if outer_wall < w:
            continue
        # spacing must hold at every twist angle: rotating in normalised ellipse coordinates
        # changes physical distances when a != b
        ok = all(
            np.hypot(a_min * (rho * np.cos(phi + t) - p * np.cos(f + t)), b_min * (rho * np.sin(phi + t) - p * np.sin(f + t))) - 2 * r >= w
            for p, f in chosen for t in twists)
        if ok:
            chosen.append((rho, phi))
        if len(chosen) >= params.n_channels_max:
            break
    if not chosen:
        notes.append(f"no channel fits: narrowest station semi-axes {a_min:.2f}x{b_min:.2f} mm with channel d={2*r} mm and min wall {w} mm")
    notes.append(f"{len(chosen)} channels placed; layout constrained at narrowest station ({a_min:.2f}x{b_min:.2f} mm)")
    return np.array(chosen).reshape(-1, 2), notes


@dataclasses.dataclass
class ScaffoldCandidate:
    occupancy: np.ndarray  # bool on fine grid
    grid_affine: np.ndarray
    envelope: np.ndarray  # bool, before channels
    channels_polylines: list  # list of (n,3) world mm (hypothesized guidance pathways)
    span_t: tuple
    stations: StationModel
    params: ScaffoldParams
    notes: list
    applicable: bool = True
    evidence: dict = dataclasses.field(default_factory=lambda: {
        "envelope": EvidenceLevel.ESTIMATED.value, "internal_channels": EvidenceLevel.HYPOTHESIZED.value,
        "label": "hypothesized internal guidance architecture (not recovered original fibres)"})


def _t_of(P, sm):
    return ((P - sm.fp_f) @ sm.u) / sm.L


def determine_span(sm: StationModel, labels: np.ndarray, affine: np.ndarray, params: ScaffoldParams) -> tuple[float, float, list[str], bool]:
    notes = []
    sp = np.linalg.norm(affine[:3, :3], axis=0)
    tt = np.linspace(sm.t[0], sm.t[-1], 200)
    c, *_ = sm.at(tt)
    vox = np.clip(np.round(apply_affine(np.linalg.inv(affine), c)).astype(int), 0, np.array(labels.shape) - 1)
    margin = params.attachment_margin_mm if params.preserve_attachments and not params.enthesis_transition else 0.0
    dist = {}
    for name, bone in (("femur", Label.FEMUR), ("tibia", Label.TIBIA)):
        dist[name] = ndimage.distance_transform_edt(labels != bone, sampling=sp)[vox[:, 0], vox[:, 1], vox[:, 2]]
    okf = np.where(dist["femur"] >= max(margin, 1e-6))[0]
    okt = np.where(dist["tibia"] >= max(margin, 1e-6))[0]
    t_min = float(tt[okf[0]]) if okf.size else float(sm.t[0])
    t_max = float(tt[okt[-1]]) if okt.size else float(sm.t[-1])
    if params.enthesis_transition:
        notes.append("enthesis transition requested: ends extend to bone contact. Justification: " + params.enthesis_justification)
    elif params.preserve_attachments:
        notes.append(f"attachment regions preserved: scaffold ends >= {margin} mm from femur/tibia along the centreline")
    if params.mode == "full_envelope":
        return t_min, t_max, notes + ["full-envelope mode (spans attachment limits)"], True
    # bridging: need remnant tips
    rf = apply_affine(affine, np.argwhere(labels == Label.ACL_REMNANT_FEMORAL))
    rt = apply_affine(affine, np.argwhere(labels == Label.ACL_REMNANT_TIBIAL))
    intact = (labels == Label.ACL_INTACT).sum()
    if len(rf) == 0 and len(rt) == 0:
        if intact:
            return 0, 0, notes + ["NOT APPLICABLE: ACL fibres annotated as continuous (partial tear / intact); no gap to bridge"], False
        return 0, 0, notes + ["NOT APPLICABLE for bridging: no visible remnant on either side (e.g. chronic/absent tissue). "
                              "A bridging design presumes remnants; use full_envelope only as a separate, explicitly different hypothesis"], False
    tf = float(np.percentile(_t_of(rf, sm), 98)) if len(rf) else None
    ttb = float(np.percentile(_t_of(rt, sm), 2)) if len(rt) else None
    ov = params.overlap_mm / sm.L
    t0 = max(t_min, tf - ov) if tf is not None else t_min
    t1 = min(t_max, ttb + ov) if ttb is not None else t_max
    if tf is None:
        notes.append("no femoral-side remnant visible: bridge terminates at the femoral preservation boundary (fixation concept out of scope)")
    if ttb is None:
        notes.append("no tibial-side remnant visible: bridge terminates at the tibial preservation boundary (fixation concept out of scope)")
    if tf is not None and ttb is not None:
        gap_mm = (ttb - tf) * sm.L
        notes.append(f"visible gap between remnant tips along axis: {gap_mm:.1f} mm; overlap onto each remnant: {params.overlap_mm} mm (user choice)")
        if gap_mm <= 0:
            return 0, 0, notes + ["NOT APPLICABLE: remnants overlap along the axis; no gap to bridge"], False
    if t1 <= t0:
        return 0, 0, notes + ["NOT APPLICABLE: empty span after preservation margins"], False
    return t0, t1, notes, True


def generate_candidate(estimate_mask, fp_f, fp_t, labels: np.ndarray, affine: np.ndarray, params: ScaffoldParams) -> ScaffoldCandidate:
    errs = params.validate()
    if errs:
        raise ValueError("invalid scaffold parameters: " + "; ".join(errs))
    sm = fit_stations(estimate_mask, affine, fp_f, fp_t, params.n_stations)
    t0, t1, notes, ok = determine_span(sm, labels, affine, params)
    if not ok:
        return ScaffoldCandidate(np.zeros((1, 1, 1), bool), np.eye(4), np.zeros((1, 1, 1), bool), [], (t0, t1), sm, params, notes, applicable=False)
    # fine grid around the span
    tt = np.linspace(t0, t1, 64)
    c, a, b, _ = sm.at(tt)
    R = float(max(a.max(), b.max()) * params.envelope_scale + 1.0)
    lo = c.min(0) - R
    hi = c.max(0) + R
    res = params.grid_res_mm
    shape = np.ceil((hi - lo) / res).astype(int) + 1
    G = np.eye(4) * res
    G[3, 3] = 1
    G[:3, 3] = lo
    ijk = np.indices(shape).reshape(3, -1).T
    X = apply_affine(G, ijk.astype(float))
    t = _t_of(X, sm)
    cc, aa, bb, th = sm.at(t)
    d = X - cc
    x1, x2 = d @ sm.e1, d @ sm.e2
    ca, sa = np.cos(th), np.sin(th)
    p = x1 * ca + x2 * sa
    q = -x1 * sa + x2 * ca
    aa, bb = aa * params.envelope_scale, bb * params.envelope_scale
    env = ((p / aa) ** 2 + (q / bb) ** 2 <= 1.0) & (t >= t0) & (t <= t1)
    occ = env.copy()
    polylines = []
    layout, lnotes = channel_layout(params, sm, t0, t1)
    notes += lnotes
    if len(layout):
        rch = params.channel_diameter_mm / 2
        frac = np.clip((t - t0) / (t1 - t0), 0, 1)
        tw = np.radians(params.twist_deg if params.alignment_pattern == "twisted" else 0.0) * frac
        for rho, phi in layout:
            px = aa * rho * np.cos(phi + tw)
            qy = bb * rho * np.sin(phi + tw)
            inside = (p - px) ** 2 + (q - qy) ** 2 <= rch ** 2
            occ &= ~(inside & env)
            # polyline for export
            ts = np.linspace(t0, t1, 40)
            c2, a2, b2, th2 = sm.at(ts)
            f2 = (ts - t0) / (t1 - t0)
            tw2 = np.radians(params.twist_deg if params.alignment_pattern == "twisted" else 0.0) * f2
            a2, b2 = a2 * params.envelope_scale, b2 * params.envelope_scale
            pp = a2 * rho * np.cos(phi + tw2)
            qq = b2 * rho * np.sin(phi + tw2)
            x1 = pp * np.cos(th2) - qq * np.sin(th2)
            x2 = pp * np.sin(th2) + qq * np.cos(th2)
            polylines.append(c2 + x1[:, None] * sm.e1 + x2[:, None] * sm.e2)
    occ = occ.reshape(shape)
    env = env.reshape(shape)
    # remnant accommodation + bone clearance (sampled from the image grid via distance maps)
    sp = np.linalg.norm(affine[:3, :3], axis=0)
    vox = apply_affine(np.linalg.inv(affine), X)
    rem = np.isin(labels, [Label.ACL_REMNANT_FEMORAL, Label.ACL_REMNANT_TIBIAL, Label.ACL_FRAGMENT_OTHER, Label.ACL_INTACT])
    if rem.any():
        dr = ndimage.distance_transform_edt(~rem, sampling=sp) - 0.5 * sp.min()
        drs = ndimage.map_coordinates(dr, vox.T, order=1, mode="nearest").reshape(shape)
        occ &= drs > params.remnant_clearance_mm
        notes.append(f"visible remnant tissue + {params.remnant_clearance_mm} mm clearance excluded from scaffold volume (remnant accommodation)")
    bone = np.isin(labels, [Label.FEMUR, Label.TIBIA])
    db = ndimage.distance_transform_edt(~bone, sampling=sp) - 0.5 * sp.min()
    dbs = ndimage.map_coordinates(db, vox.T, order=1, mode="nearest").reshape(shape)
    if not params.enthesis_transition:
        occ &= dbs > params.bone_clearance_mm
    # keep the largest connected component; report others
    lab, n = ndimage.label(occ)
    if n > 1:
        sizes = ndimage.sum(occ, lab, range(1, n + 1))
        keep = int(np.argmax(sizes)) + 1
        notes.append(f"{n - 1} small disconnected fragment(s) removed (total {int(sizes.sum() - sizes.max())} voxels)")
        occ = lab == keep
    return ScaffoldCandidate(occ, G, env, polylines, (float(t0), float(t1)), sm, params, notes)


def endpoint_matching_hypotheses(fem_ends: np.ndarray, tib_ends: np.ndarray, max_cost_mm: float, k: int = 3,
                                 acquisition_supports_segments: bool = False) -> dict:
    """OPTIONAL experiment: competing correspondences between observed femoral-side and
    tibial-side segment end points (e.g. from high-resolution research imaging).

    Never forces one-to-one matching: every segment may remain unmatched (cost = max_cost_mm).
    Returns the best k distinct hypotheses. Blocked unless the caller asserts that the
    acquisition and annotation actually resolve individual segments."""
    if not acquisition_supports_segments:
        return {"status": "blocked", "reason": "routine MRI does not resolve individual fibre bundles/ends; supply segment-level imaging and annotation"}
    from scipy.optimize import linear_sum_assignment

    F, T = np.asarray(fem_ends, float), np.asarray(tib_ends, float)
    nf, nt = len(F), len(T)
    D = np.linalg.norm(F[:, None] - T[None], axis=2) if nf and nt else np.zeros((nf, nt))
    big = 1e6
    # augmented matrix: [D, unmatched_f; unmatched_t, 0]
    C = np.full((nf + nt, nt + nf), big)
    C[:nf, :nt] = np.where(D <= max_cost_mm, D, big)
    C[:nf, nt:] = np.where(np.eye(nf) > 0, max_cost_mm, big)
    C[nf:, :nt] = np.where(np.eye(nt) > 0, max_cost_mm, big)
    C[nf:, nt:] = 0

    def solve(Cm):
        r, c = linear_sum_assignment(Cm)
        cost = Cm[r, c].sum()
        pairs = [(int(i), int(j)) for i, j in zip(r, c) if i < nf and j < nt and Cm[i, j] < big]
        return cost, pairs

    sols = []
    base_cost, base = solve(C)
    sols.append((base_cost, base))
    for (i, j) in base:  # alternatives: forbid each matched pair once
        C2 = C.copy()
        C2[i, j] = big
        cost, pairs = solve(C2)
        if cost < big and pairs not in [s[1] for s in sols]:
            sols.append((cost, pairs))
    sols = sorted(sols, key=lambda s: s[0])[:k]
    return {"status": "ok", "evidence_level": EvidenceLevel.HYPOTHESIZED.value,
            "hypotheses": [{"cost": float(cst), "pairs": prs, "unmatched_femoral": sorted(set(range(nf)) - {p[0] for p in prs}),
                            "unmatched_tibial": sorted(set(range(nt)) - {p[1] for p in prs})} for cst, prs in sols],
            "note": "competing hypotheses, not a recovered correspondence"}
