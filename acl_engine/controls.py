"""Closed-loop zone controls (P6). Fiber-length targets iterate to tolerance.

FIX: Muse's T1 version called PhaseField.constant_interval() and
mean_lambda_interval(..., dy=) without a strand index; neither exists, so it
crashed. This version uses the real paths.py API and measures each zone the same
way the report does (whole zone, mean of the 3 strands).
"""
from scipy.optimize import brentq
from acl_engine.paths import PhaseField, mean_lambda_interval, build_rates

TOL = 2e-3


def zone_lambda(pf, k, a, b, q, phase_offset, mirror):
    y0, y1 = pf.bounds[k], pf.bounds[k + 1]
    return sum(mean_lambda_interval(y0, y1, pf, a, b, q, phase_offset, mirror, i)
               for i in range(3)) / 3.0


def resolve_rates_closed_loop(norm, a, b, q, angle_mode, phase_offset=0.0, mirror=False):
    """Returns (rates, achieved). achieved[zone_id] = {target, achieved, iters}."""
    zones = norm["zones"]
    lens = [float(z["length_mm"]) for z in zones]
    trans = float(norm["transitions"]["length_mm"])
    rates = build_rates(norm, a, b, q, angle_mode)  # already corrected in open loop
    achieved = {}
    for k, z in enumerate(zones):
        if z.get("control", "braid_angle") != "fiber_length":
            continue
        target = float(z["path_length_ratio"])
        calls = [0]

        def miss(r):
            calls[0] += 1
            trial = list(rates)
            trial[k] = float(r)
            pf = PhaseField(lens, trial, trans, fine_dy=0.01)
            return zone_lambda(pf, k, a, b, q, phase_offset, mirror) - target

        r0 = rates[k]
        f0 = miss(r0)
        if abs(f0) > 2e-4:
            lo, hi = r0 * 0.7, r0 * 1.4
            for _ in range(20):
                if miss(lo) < 0 < miss(hi):
                    break
                lo, hi = lo * 0.7, hi * 1.4
            else:
                raise ValueError("fiber_length target %.4f unreachable in zone %s" % (target, z["id"]))
            rates[k] = float(brentq(miss, lo, hi, xtol=1e-7, maxiter=60))
        pf = PhaseField(lens, rates, trans)
        lam = zone_lambda(pf, k, a, b, q, phase_offset, mirror)
        if abs(lam - target) > TOL:
            raise ValueError("fiber_length did not converge in zone %s: %.5f vs %.5f"
                             % (z["id"], lam, target))
        achieved[z["id"]] = {"target": target, "achieved": float(lam), "iters": calls[0]}
    return rates, achieved
