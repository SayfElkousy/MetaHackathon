"""Overlay images for KneeScan results (used by the web app; works from the JSON so remote results render the same)."""
import io, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ACL, BOX, FEM, FIB, TIB, GAP = '#ff5a4f', '#ffd34d', '#f2c14e', '#ff6fae', '#5fa8ff', '#f4f7f9'


def _show(ax, img, zoom_box=None, pad=0.9):
    lo, hi = np.percentile(img, [1, 99.6]); ax.imshow(img, cmap='gray', vmin=lo, vmax=hi, interpolation='lanczos')
    if zoom_box:
        x0, y0, x1, y1 = zoom_box; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; h = max(x1 - x0, y1 - y0) * pad + 20
        ax.set_xlim(cx - h, cx + h); ax.set_ylim(cy + h, cy - h)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)


def _fig_to_pil(fig):
    b = io.BytesIO(); fig.savefig(b, format='png', dpi=110, bbox_inches='tight', facecolor='#0d1013'); plt.close(fig); b.seek(0)
    return Image.open(b).convert('RGB')


def _closed(p): p = np.asarray(p, float); return np.vstack([p, p[:1]])


def acl_figure(vol, res):
    a = res.get('acl', {}); loc = res['acl_location']; k = a.get('slice', loc['slice'])
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.6), facecolor='#0d1013')
    for ax in axs: _show(ax, vol[k])
    x0, y0, x1, y1 = loc['box']
    axs[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=BOX, lw=1.5))
    axs[0].set_title(f'ACL located: slice {k + 1} of {vol.shape[0]}', color='w', fontsize=10)
    _show(axs[1], vol[k], zoom_box=loc['box'])
    if a.get('outline'):
        P = _closed(a['outline']); axs[1].plot(P[:, 0], P[:, 1], '-', color=ACL, lw=1.6)
        axs[1].plot([a['end0'][0], a['end1'][0]], [a['end0'][1], a['end1'][1]], '-', color=BOX, lw=1.2)
        unit = 'mm' if a.get('length_mm') is not None else 'px'
        L = a.get('length_mm') if unit == 'mm' else a.get('length_px'); W = a.get('width_mm') if unit == 'mm' else a.get('width_px')
        axs[1].set_title(f'ACL outline: length {L} {unit}, width {W} {unit}', color='w', fontsize=10)
    else:
        axs[1].set_title('No ACL outline', color='w', fontsize=10)
    return _fig_to_pil(fig)


def bones_figure(vol, res):
    b = res.get('femur_fibula')
    if not b or 'slice' not in b: return None
    k = b['slice']; fig, ax = plt.subplots(1, 1, figsize=(4.8, 4.8), facecolor='#0d1013'); _show(ax, vol[k])
    for key, col in (('femur_outline', FEM), ('tibia_outline', TIB), ('fibula_outline', FIB)):
        if b.get(key): P = _closed(b[key]); ax.plot(P[:, 0], P[:, 1], '-', color=col, lw=1.4)
    p, q = b['femur_point'], b['fibula_point']; ax.plot([p[0], q[0]], [p[1], q[1]], '-', color=GAP, lw=2)
    xs = [p[0], q[0]]; ys = [p[1], q[1]]; cx, cy = np.mean(xs), np.mean(ys)
    ax.set_xlim(cx - 90 * vol.shape[2] / 256, cx + 90 * vol.shape[2] / 256); ax.set_ylim(cy + 90 * vol.shape[1] / 256, cy - 90 * vol.shape[1] / 256)
    ax.set_title(f"Femur to fibular head: {b['distance_mm']} mm (slice {k + 1})", color='w', fontsize=10)
    return _fig_to_pil(fig)
