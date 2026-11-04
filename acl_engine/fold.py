"""Fold (over/under) profile for the three-strand plait.

Written to match the interface Muse's N2 code expects (Muse's N1
version of this file was never shared). Math follows Prompt v3 section 1.3.

z_i = b * g(2*phi_i), where g blends a smooth sine (sharpness q = 0) with a
sharper, plateau-like profile (q = 1):
    g(u) = (1 - q) * sin(u) + q * tanh(K * sin(u)) / tanh(K)
g is smooth (C-infinity), odd, and periodic, so crossings stay where they were.
"""
import math
import numpy as np

K_SHARP = 3.0  # how square the q = 1 profile is (engineering choice)


def g(u, q):
    u = np.asarray(u, dtype=float)
    return (1.0 - q) * np.sin(u) + q * np.tanh(K_SHARP * np.sin(u)) / math.tanh(K_SHARP)


def g1(u, q):
    """dg/du"""
    u = np.asarray(u, dtype=float)
    s, c = np.sin(u), np.cos(u)
    sech2 = 1.0 / np.cosh(K_SHARP * s) ** 2
    return (1.0 - q) * c + q * K_SHARP * c * sech2 / math.tanh(K_SHARP)


def g2(u, q):
    """d2g/du2"""
    u = np.asarray(u, dtype=float)
    s, c = np.sin(u), np.cos(u)
    k = K_SHARP
    th = np.tanh(k * s)
    sech2 = 1.0 - th * th
    d_inner = -s * sech2 + c * (-2.0 * th * sech2 * k * c)
    return (1.0 - q) * (-s) + q * k * d_inner / math.tanh(k)


def gc_cross(q):
    """g'(u) at a projected crossing, where sin(u) = +/-sqrt(3)/2, cos(u) = 1/2."""
    return float(g1(np.array([math.pi / 3.0]), q)[0])


def slope_factor_q(a, b, q):
    """3D slope factor at projected crossings: tan(tilt) = phi' * S.
    q = 0 gives S = sqrt(3a^2/4 + b^2), the v2 formula."""
    return math.sqrt(0.75 * a * a + 4.0 * b * b * gc_cross(q) ** 2)
