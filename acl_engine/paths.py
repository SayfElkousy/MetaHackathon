"""Phase field, strand centerlines, analytic derivatives, crossings."""
import math
import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz


def quintic(t):
    t = np.asarray(t, dtype=float)
    return 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3


def quintic_d1(t):
    t = np.asarray(t, dtype=float)
    return 30.0 * t**4 - 60.0 * t**3 + 30.0 * t**2


def slope_factor(a, b):
    return math.sqrt(0.75 * a * a + b * b)


def rate_for_theta(theta_deg, a, b, mode="crossing_tilt"):
    th = math.radians(float(theta_deg))
    tan = math.tan(th)
    if mode == "crossing_tilt":
        return tan / slope_factor(a, b)
    elif mode == "legacy_lateral_pitch_rule":
        return tan / a
    else:
        raise ValueError("bad angle mode")


class PhaseField:
    """phi(0)=0, piecewise-constant rate with quintic rate blends."""

    def __init__(self, zone_lengths, rates, trans_len, fine_dy=0.002):
        self.zone_lengths = [float(v) for v in zone_lengths]
        self.rates = [float(v) for v in rates]
        self.trans_len = float(trans_len)
        self.L = sum(self.zone_lengths)
        b = [0.0]
        for v in self.zone_lengths:
            b.append(b[-1] + v)
        self.bounds = b
        self.windows = []  # (yb, w, rl, rr)
        for k in range(1, len(self.zone_lengths)):
            yb = b[k]
            shorter = min(self.zone_lengths[k - 1], self.zone_lengths[k])
            w = min(self.trans_len, 0.3 * shorter)
            if w < 1e-12:
                w = 0.0
            self.windows.append((yb, w, self.rates[k - 1], self.rates[k]))
        self.fine_dy = float(fine_dy)
        n = max(2, int(math.ceil(self.L / self.fine_dy)) + 1)
        self.ys_fine = np.linspace(0.0, self.L, n)
        pp = self.rate_array(self.ys_fine)
        self.phi_fine = np.zeros(n)
        # trapezoidal cumulative integral
        self.phi_fine[1:] = np.cumsum(0.5 * (pp[:-1] + pp[1:]) * np.diff(self.ys_fine))

    def _window_at(self, y):
        for (yb, w, rl, rr) in self.windows:
            if w <= 0:
                continue
            if (yb - w / 2.0) <= y <= (yb + w / 2.0):
                return (yb, w, rl, rr)
        return None

    def rate_scalar(self, y):
        win = self._window_at(float(y))
        if win is not None:
            yb, w, rl, rr = win
            t = (float(y) - (yb - w / 2.0)) / w
            return rl + (rr - rl) * float(quintic(t))
        # constant zone
        b = self.bounds
        for k in range(len(self.rates)):
            if y <= b[k + 1] + 1e-12 or k == len(self.rates) - 1:
                if y >= b[k] - 1e-12:
                    return self.rates[k]
        return self.rates[-1]

    def rate2_scalar(self, y):
        win = self._window_at(float(y))
        if win is not None:
            yb, w, rl, rr = win
            t = (float(y) - (yb - w / 2.0)) / w
            return (rr - rl) * float(quintic_d1(t)) / w
        return 0.0

    # FIX (Claude, speed): rate_array / rate2_array looped over points in Python and were 85% of
    # every rebuild. Same rules as rate_scalar / rate2_scalar, done on whole arrays at once.
    def _windows_vec(self, ys):
        """Yields (mask, t, w, rl, rr) per blend window; earlier windows win, as in _window_at."""
        taken = np.zeros(ys.shape, dtype=bool)
        for (yb, w, rl, rr) in self.windows:
            if w <= 0:
                continue
            m = (ys >= yb - w / 2.0) & (ys <= yb + w / 2.0) & ~taken
            taken |= m
            if m.any():
                yield m, (ys[m] - (yb - w / 2.0)) / w, w, rl, rr

    def rate_array(self, ys):
        ys = np.asarray(ys, dtype=float)
        b = np.asarray(self.bounds, dtype=float)
        k = np.searchsorted(b[1:] + 1e-12, ys, side="left")
        k = np.minimum(k, len(self.rates) - 1)
        out = np.asarray(self.rates, dtype=float)[k]
        for m, t, _w, rl, rr in self._windows_vec(ys):
            out[m] = rl + (rr - rl) * quintic(t)
        return out

    def rate2_array(self, ys):
        ys = np.asarray(ys, dtype=float)
        out = np.zeros_like(ys)
        for m, t, w, rl, rr in self._windows_vec(ys):
            out[m] = (rr - rl) * quintic_d1(t) / w
        return out

    def phi_array(self, ys):
        ys = np.asarray(ys, dtype=float)
        return np.interp(ys, self.ys_fine, self.phi_fine)

    def phi_L(self):
        return float(self.phi_fine[-1])


def build_phase_field(request_norm):
    a = float(request_norm["architecture"]["swing_mm"])
    b = float(request_norm["architecture"]["lift_mm"])
    mode = request_norm["architecture"].get("angle_control_mode", "crossing_tilt")
    zones = request_norm["zones"]
    # FIX (Claude): the N2 schema renames braid_tilt_target_deg -> braid_angle_deg; accept both
    rates = [rate_for_theta(z.get("braid_angle_deg", z.get("braid_tilt_target_deg")), a, b, mode) for z in zones]
    lens = [float(z["length_mm"]) for z in zones]
    trans = float(request_norm["transitions"]["length_mm"])
    pf = PhaseField(lens, rates, trans)
    S = slope_factor(a, b)
    p_events = [math.pi / (3.0 * r) for r in rates]
    return pf, rates, S, p_events


def strand_angles(y, phi, rate, rate2, a, b, phase_offset, mirror, idx):
    """Position, first and second y-derivatives for one strand."""
    off = float(phase_offset)
    base = phi + off + 2.0 * math.pi * float(idx) / 3.0
    if mirror:
        p = -base
        dp = -rate
        d2p = -rate2
    else:
        p = base
        dp = rate
        d2p = rate2
    x = a * np.sin(p)
    z = b * np.sin(2.0 * p)
    dx = a * np.cos(p) * dp
    dz = 2.0 * b * np.cos(2.0 * p) * dp
    d2x = -a * np.sin(p) * dp * dp + a * np.cos(p) * d2p
    d2z = -4.0 * b * np.sin(2.0 * p) * dp * dp + 2.0 * b * np.cos(2.0 * p) * d2p
    pos = np.stack([x, np.asarray(y, dtype=float), z], axis=-1)
    vel = np.stack([dx, np.ones_like(dx), dz], axis=-1)
    acc = np.stack([d2x, np.zeros_like(d2x), d2z], axis=-1)
    return pos, vel, acc


def tilt_deg_from_vel(vel):
    v = np.asarray(vel, dtype=float)
    n = np.linalg.norm(v, axis=-1)
    n = np.where(n < 1e-30, 1e-30, n)
    c = np.abs(v[..., 1] / n)
    c = np.clip(c, 0.0, 1.0)
    return np.degrees(np.arccos(c))


def arc_length(y0, y1, pf, a, b, phase_offset, mirror, idx, dy=0.002):
    y0 = float(y0)
    y1 = float(y1)
    if y1 <= y0:
        return 0.0
    n = max(2, int(math.ceil((y1 - y0) / dy)) + 1)
    ys = np.linspace(y0, y1, n)
    phi = pf.phi_array(ys)
    rate = pf.rate_array(ys)
    rate2 = pf.rate2_array(ys)
    _, vel, _ = strand_angles(ys, phi, rate, rate2, a, b, phase_offset, mirror, idx)
    sp = np.linalg.norm(vel, axis=1)
    return float(_trapz(sp, ys))


def _xdiff(y, pf, a, b, phase_offset, mirror, i, j):
    phi = pf.phi_array(np.array([y]))[0]
    rate = pf.rate_array(np.array([y]))[0]
    rate2 = np.array([0.0])
    pi_, _, _ = strand_angles(np.array([y]), np.array([phi]), np.array([rate]),
                               rate2, a, b, phase_offset, mirror, i)
    pj_, _, _ = strand_angles(np.array([y]), np.array([phi]), np.array([rate]),
                               rate2, a, b, phase_offset, mirror, j)
    return float(pi_[0, 0] - pj_[0, 0])


def _find_crossings_m1(pf, a, b, phase_offset, mirror, L):
    """Bisection roots of x_i - x_j to 1e-9 mm in y."""
    pairs = [(0, 1), (1, 2), (0, 2)]
    dy = 0.01
    n = max(2, int(math.ceil(L / dy)) + 1)
    ys = np.linspace(0.0, L, n)
    phi = pf.phi_array(ys)
    rate = pf.rate_array(ys)
    rate2 = np.zeros_like(ys)
    xs = []
    zs = []
    tilts = []
    for idx in range(3):
        pos, vel, _ = strand_angles(ys, phi, rate, rate2, a, b,
                                    phase_offset, mirror, idx)
        xs.append(pos[:, 0])
        zs.append(pos[:, 2])
        tilts.append(tilt_deg_from_vel(vel))
    xs = np.stack(xs, axis=1)
    zs = np.stack(zs, axis=1)
    events = []
    for (i, j) in pairs:
        f = xs[:, i] - xs[:, j]
        for k in range(n - 1):
            hit = False
            if f[k] == 0.0:
                yr = float(ys[k])
                hit = True
            elif f[k] * f[k + 1] < 0:
                yl = float(ys[k])
                yr = float(ys[k + 1])
                fl = float(f[k])
                # bisection on true function
                for _ in range(80):
                    ym = 0.5 * (yl + yr)
                    fm = _xdiff(ym, pf, a, b, phase_offset, mirror, i, j)
                    if fl == 0.0:
                        yl = ym
                        break
                    if fl * fm <= 0:
                        yr = ym
                    else:
                        yl = ym
                        fl = fm
                    if (yr - yl) < 1e-9:
                        break
                yr = 0.5 * (yl + yr)
                hit = True
            if hit:
                phi_r = float(pf.phi_array(np.array([yr]))[0])
                r_r = float(pf.rate_array(np.array([yr]))[0])
                r2 = np.array([0.0])
                yarr = np.array([yr])
                pharr = np.array([phi_r])
                rarr = np.array([r_r])
                pi_, vi_, _ = strand_angles(yarr, pharr, rarr, r2, a, b,
                                            phase_offset, mirror, i)
                pj_, vj_, _ = strand_angles(yarr, pharr, rarr, r2, a, b,
                                            phase_offset, mirror, j)
                zi = float(pi_[0, 2])
                zj = float(pj_[0, 2])
                if zi > zj:
                    over = i
                elif zj > zi:
                    over = j
                else:
                    over = -1
                ti = float(tilt_deg_from_vel(vi_)[0])
                tj = float(tilt_deg_from_vel(vj_)[0])
                events.append({"y_mm": float(yr), "pair": [i, j],
                               "over": int(over),
                               "tilt_i_deg": ti, "tilt_j_deg": tj})
    events.sort(key=lambda e: e["y_mm"])
    return events


# ---------------------------------------------------------------------------
# N1 additions (written by Claude to match the interface Muse's N2 code uses;
# Muse's N1 version of these functions was never shared).
# ---------------------------------------------------------------------------
from acl_engine import fold as _fold


def straight_pos_vel_acc(s, phi, rate, rate2, a, b, q, phase_offset, mirror, idx):
    """Strand centerline in the straight local frame with fold sharpness q.
    pos = (a sin p, s, b g(2p)); vel, acc are d/ds and d2/ds2."""
    off = float(phase_offset)
    base = np.asarray(phi, dtype=float) + off + 2.0 * math.pi * float(idx) / 3.0
    sign = -1.0 if mirror else 1.0
    p = sign * base
    dp = sign * np.asarray(rate, dtype=float)
    d2p = sign * np.asarray(rate2, dtype=float)
    x = a * np.sin(p)
    z = b * _fold.g(2.0 * p, q)
    dx = a * np.cos(p) * dp
    dz = 2.0 * b * _fold.g1(2.0 * p, q) * dp
    d2x = -a * np.sin(p) * dp * dp + a * np.cos(p) * d2p
    d2z = 4.0 * b * _fold.g2(2.0 * p, q) * dp * dp + 2.0 * b * _fold.g1(2.0 * p, q) * d2p
    s = np.asarray(s, dtype=float)
    pos = np.stack([x, s, z], axis=-1)
    vel = np.stack([dx, np.ones_like(dx), dz], axis=-1)
    acc = np.stack([d2x, np.zeros_like(d2x), d2z], axis=-1)
    return pos, vel, acc


def lambda_for_rate(rate, a, b, q, n=2048):
    """Mean path-length ratio over one full phase cycle at constant rate."""
    p = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    dxp = a * np.cos(p)
    dzp = 2.0 * b * _fold.g1(2.0 * p, q)
    return float(np.mean(np.sqrt(1.0 + rate * rate * (dxp * dxp + dzp * dzp))))


def rate_for_lambda(lam, a, b, q):
    """Phase rate giving a mean path-length ratio lam (bisection)."""
    if lam <= 1.0:
        raise ValueError("path_length_ratio must be > 1")
    lo, hi = 0.0, 1.0
    while lambda_for_rate(hi, a, b, q) < lam:
        hi *= 2.0
        if hi > 1e3:
            raise ValueError("path_length_ratio unreachable")
    for _ in range(80):
        m = 0.5 * (lo + hi)
        if lambda_for_rate(m, a, b, q) < lam:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi)


def build_rates(norm, a, b, q, angle_mode):
    """Per-zone phase rate from each zone's control (braid_angle | fiber_length | pitch)."""
    S = _fold.slope_factor_q(a, b, q)
    rates = []
    for z in norm["zones"]:
        ctl = z.get("control", "braid_angle")
        if ctl == "braid_angle":
            th = math.radians(float(z.get("braid_angle_deg", z.get("braid_tilt_target_deg"))))
            rates.append(math.tan(th) / (a if angle_mode == "legacy_lateral_pitch_rule" else S))
        elif ctl == "fiber_length":
            rates.append(rate_for_lambda(float(z["path_length_ratio"]), a, b, q))
        elif ctl == "pitch":
            rates.append(math.pi / (3.0 * float(z["pitch_mm"])))
        else:
            raise ValueError("bad zone control %r" % ctl)
    # FIX (Claude): rate_for_lambda assumes a full braid cycle at constant rate. A short zone
    # holds only part of a cycle, and the transition blends pull in the neighbors' rates, so
    # the realized fiber length missed the target (1.063 vs 1.060). Correct it by measuring the
    # actual zone (mean of the 3 strands) and adjusting each fiber_length zone's rate.
    fl = [k for k, z in enumerate(norm["zones"]) if z.get("control") == "fiber_length"]
    if fl:
        lens = [float(z["length_mm"]) for z in norm["zones"]]
        trans = float(norm["transitions"]["length_mm"])
        for _ in range(8):
            pf = PhaseField(lens, rates, trans, fine_dy=0.01)
            worst = 0.0
            for k in fl:
                target = float(norm["zones"][k]["path_length_ratio"])
                y0, y1 = pf.bounds[k], pf.bounds[k + 1]
                lam = sum(mean_lambda_interval(y0, y1, pf, a, b, q, 0.0, False, i) for i in range(3)) / 3.0
                worst = max(worst, abs(lam - target))
                # lambda - 1 scales ~ rate^2 for small tilt; use that for the update
                rates[k] *= math.sqrt(max(target - 1.0, 1e-9) / max(lam - 1.0, 1e-9))
            if worst < 2e-4:
                break
    return rates


def _xdiff_q(y, pf, a, b, q, phase_offset, mirror, i, j):
    ya = np.array([float(y)])
    ph = pf.phi_array(ya)
    rt = pf.rate_array(ya)
    z0 = np.zeros(1)
    pi_, _, _ = straight_pos_vel_acc(ya, ph, rt, z0, a, b, q, phase_offset, mirror, i)
    pj_, _, _ = straight_pos_vel_acc(ya, ph, rt, z0, a, b, q, phase_offset, mirror, j)
    return float(pi_[0, 0] - pj_[0, 0])


def find_crossings(pf, a, b, *args):
    """find_crossings(pf, a, b, phase_offset, mirror, L)       (M1 signature)
    find_crossings(pf, a, b, q, phase_offset, mirror, L)    (N1/N2 signature)"""
    if len(args) == 3:
        return _find_crossings_m1(pf, a, b, *args)
    q, phase_offset, mirror, L = args
    ys = np.linspace(0.0, L, max(2, int(math.ceil(L / 0.01)) + 1))
    phi = pf.phi_array(ys)
    rate = pf.rate_array(ys)
    r2 = np.zeros_like(ys)
    X = [straight_pos_vel_acc(ys, phi, rate, r2, a, b, q, phase_offset, mirror, i)[0][:, 0] for i in range(3)]
    events = []
    for (i, j) in [(0, 1), (1, 2), (0, 2)]:
        f = X[i] - X[j]
        for k in np.where(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]:
            yl, yr = float(ys[k]), float(ys[k + 1])
            fl = _xdiff_q(yl, pf, a, b, q, phase_offset, mirror, i, j)
            for _ in range(60):
                ym = 0.5 * (yl + yr)
                fm = _xdiff_q(ym, pf, a, b, q, phase_offset, mirror, i, j)
                if fl * fm <= 0:
                    yr = ym
                else:
                    yl, fl = ym, fm
                if yr - yl < 1e-9:
                    break
            y0 = 0.5 * (yl + yr)
            ya = np.array([y0])
            ph = pf.phi_array(ya)
            rt = pf.rate_array(ya)
            z0 = np.zeros(1)
            pi_, vi_, _ = straight_pos_vel_acc(ya, ph, rt, z0, a, b, q, phase_offset, mirror, i)
            pj_, vj_, _ = straight_pos_vel_acc(ya, ph, rt, z0, a, b, q, phase_offset, mirror, j)
            over = i if pi_[0, 2] > pj_[0, 2] else j
            events.append({"y_mm": y0, "pair": [i, j], "over": int(over),
                           "tilt_i_deg": float(tilt_deg_from_vel(vi_)[0]),
                           "tilt_j_deg": float(tilt_deg_from_vel(vj_)[0])})
    events.sort(key=lambda e: e["y_mm"])
    return events


def arc_length_interval(y0, y1, pf, a, b, q, phase_offset, mirror, idx, dy=0.002):
    if y1 <= y0:
        return 0.0
    ys = np.linspace(y0, y1, max(2, int(math.ceil((y1 - y0) / dy)) + 1))
    _, vel, _ = straight_pos_vel_acc(ys, pf.phi_array(ys), pf.rate_array(ys), pf.rate2_array(ys),
                                     a, b, q, phase_offset, mirror, idx)
    return float(_trapz(np.linalg.norm(vel, axis=1), ys))


def mean_lambda_interval(y0, y1, pf, a, b, q, phase_offset, mirror, idx):
    """Path-length ratio over [y0, y1] for one strand."""
    return arc_length_interval(y0, y1, pf, a, b, q, phase_offset, mirror, idx) / (y1 - y0)
