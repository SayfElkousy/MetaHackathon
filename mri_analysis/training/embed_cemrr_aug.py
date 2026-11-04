"""Extra CEMRR training variants for MRNet-like images: random flip, ACL-centred crop (1.8-2.6 x ACL size), the crop
downsampled to 96-140 px (MRNet's ACL region is only ~110 px across) and a random contrast curve (sometimes inverted).
usage: python -I embed_cemrr_aug.py DATA_DIR OUT_DIR"""
import sys, os, glob, json, numpy as np, torch, torch.nn.functional as Fn
sys.path.insert(0, '/home/claude/acl_ext')
from msam import embed
from aug import curve, norm
from PIL import Image
src, out = sys.argv[1], sys.argv[2]
ds = sorted(glob.glob(f'{src}/*'), key=lambda p: int(os.path.basename(p)))
N = len(ds) * 2
E = np.lib.format.open_memmap(f'{out}/cemrr_aug_emb.npy', mode='w+', dtype=np.float16, shape=(N, 256, 64, 64))
M = np.zeros((N, 256, 256), np.uint8); meta = []; i = 0


def crop(a, cx, cy, side):
    s = int(round(side)); x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2))
    p = s + 2; ap = np.pad(a, p); return ap[y0 + p:y0 + p + s, x0 + p:x0 + p + s]


def resize(a, n, mode):
    return Fn.interpolate(torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None, None], size=(n, n), mode=mode, **({'align_corners': False} if mode in ('bilinear', 'bicubic') else {}))[0, 0].numpy()


for d in ds:
    img = np.array(Image.open(d + '/img.png').convert('L')).astype(np.float32)
    msk = (np.array(Image.open(d + '/mask.png').convert('L')) > 127).astype(np.float32)
    for var in ('c', 'd'):
        rng = np.random.default_rng(1000 + int(os.path.basename(d)) * 2 + (var == 'd'))
        flip = rng.uniform() < 0.5
        im, mk = (img[:, ::-1].copy(), msk[:, ::-1].copy()) if flip else (img, msk)
        ys, xs = np.nonzero(mk); L = max(np.ptp(xs), np.ptp(ys)) + 1; cx, cy = xs.mean(), ys.mean()
        side = L * rng.uniform(1.8, 2.6); sh = rng.uniform(-0.15, 0.15, 2) * L
        ci = crop(im, cx + sh[0], cy + sh[1], side); cm = crop(mk, cx + sh[0], cy + sh[1], side)
        lr = int(rng.uniform(96, 140)); ci = resize(resize(ci, lr, 'area'), ci.shape[0], 'bilinear')
        ci = curve(ci, rng) if var == 'c' else norm(ci) ** rng.uniform(0.5, 1.6)
        ci = ci + rng.normal(0, rng.uniform(0, 0.04), ci.shape).astype(np.float32)
        E[i] = embed(ci)[0].numpy().astype(np.float16)
        M[i] = resize(cm, 256, 'bilinear') > 0.5
        meta.append(dict(id=os.path.basename(d), var=var, flip=bool(flip), lowres=lr)); i += 1
    print('done', d, flush=True)
E.flush(); np.save(f'{out}/cemrr_aug_masks.npy', M); json.dump(meta, open(f'{out}/cemrr_aug_meta.json', 'w'))
