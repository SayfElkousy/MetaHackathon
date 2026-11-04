"""ACL length and width from a binary ACL mask (one sagittal slice).
length = extent of the mask along its principal (long) axis, i.e. femoral-to-tibial end of the outlined ligament.
width  = median thickness perpendicular to that axis over the middle 50% of the length (mid-substance,
         so the fanned-out attachments do not inflate it). Also returns the mask's mean width (area / length)."""
import numpy as np
from scipy import ndimage as ndi


def largest(m):
    lab, n = ndi.label(m)
    if n == 0: return m.astype(bool)
    s = ndi.sum(m, lab, range(1, n + 1)); return lab == (1 + int(np.argmax(s)))


def measure(mask):
    m = largest(mask > 0)
    ys, xs = np.nonzero(m)
    if len(xs) < 15: return None
    P = np.stack([xs, ys], 1).astype(float); c = P.mean(0)
    w, V = np.linalg.eigh(np.cov((P - c).T)); ax = V[:, 1]
    if ax[0] < 0: ax = -ax                                   # consistent direction (left to right)
    nrm = np.array([-ax[1], ax[0]])
    s = (P - c) @ ax; q = (P - c) @ nrm
    s0, s1 = s.min(), s.max(); length = float(s1 - s0 + 1)
    th = []
    for b in np.arange(np.floor(s0), np.ceil(s1) + 1):
        sel = np.abs(s - b) < 0.5
        if sel.sum() >= 1:
            frac = (b - s0) / max(1e-6, (s1 - s0))
            if 0.25 <= frac <= 0.75: th.append(q[sel].max() - q[sel].min() + 1)
    width_mid = float(np.median(th)) if th else None
    angle = float(np.degrees(np.arctan2(-ax[1], ax[0])))      # angle of long axis above image horizontal
    return dict(length=length, width_mid=width_mid, width_mean=float(len(xs) / length), area=int(len(xs)),
                elong=float(np.sqrt(w[1] / max(w[0], 1e-6))), angle=angle,
                end0=(c + s0 * ax).tolist(), end1=(c + s1 * ax).tolist(), center=c.tolist(), axis=ax.tolist())
