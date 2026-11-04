"""Contrast augmentations so the ACL model learns shape rather than one MRI contrast.
fs_like: darkens the brightest tissue (fat / marrow on non-fat-suppressed scans) with a fold-over curve, so bone looks
         dark and mid-grey fluid stays bright-ish, roughly like a fat-suppressed fluid-sensitive image.
curve:   a random smooth monotonic or inverted intensity curve (Bezier, as in Models Genesis)."""
import numpy as np


def norm(x):
    lo, hi = np.percentile(x, [0.5, 99.5]); return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)


def fs_like(x, rng):
    x = norm(x); t = np.percentile(x, rng.uniform(55, 80))
    y = np.where(x <= t, x, np.clip(t - (x - t) * rng.uniform(1.5, 3.0), 0, None))
    y = norm(y) ** rng.uniform(0.6, 1.2)
    return y.astype(np.float32)


def curve(x, rng):
    x = norm(x); p1, p2 = rng.uniform(0, 1, 2), rng.uniform(0, 1, 2)
    t = np.linspace(0, 1, 256); P0, P3 = np.array([0, 0]), np.array([1, 1])
    B = ((1 - t) ** 3)[:, None] * P0 + (3 * (1 - t) ** 2 * t)[:, None] * p1 + (3 * (1 - t) * t ** 2)[:, None] * p2 + (t ** 3)[:, None] * P3
    xs, ys = B[:, 0], B[:, 1]; o = np.argsort(xs); y = np.interp(x, xs[o], ys[o])
    if rng.uniform() < 0.4: y = 1 - y
    return y.astype(np.float32)
