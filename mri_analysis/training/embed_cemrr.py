"""Embed CEMRR kneeMRI-ACL expert-outlined images with MedSAM (frozen encoder), as square crops around the ACL so the
ligament fills a similar share of the input as it will on MRNet crops. Two variants per image: (a) crop side
1.8-2.6 x ACL size with random shift, (b) the same on the horizontally flipped image. Masks saved on the crop's 256 grid.
usage: python -I embed_cemrr.py DATA_DIR OUT_DIR"""
import sys, os, glob, json, numpy as np, torch, torch.nn.functional as Fn
sys.path.insert(0, '/home/user/acl_ext')
from msam import embed
from PIL import Image
src, out = sys.argv[1], sys.argv[2]
ds = sorted(glob.glob(f'{src}/*'), key=lambda p: int(os.path.basename(p)))
N = len(ds) * 2
E = np.lib.format.open_memmap(f'{out}/cemrr_emb.npy', mode='w+', dtype=np.float16, shape=(N, 256, 64, 64))
M = np.zeros((N, 256, 256), np.uint8); meta = []; i = 0


def crop(a, cx, cy, side):
    s = int(round(side)); x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2))
    p = s + 2; ap = np.pad(a, p); return ap[y0 + p:y0 + p + s, x0 + p:x0 + p + s], x0, y0, s


for d in ds:
    img = np.array(Image.open(d + '/img.png').convert('L')).astype(np.float32)
    msk = (np.array(Image.open(d + '/mask.png').convert('L')) > 127).astype(np.float32)
    for var in ('a', 'b'):
        rng = np.random.default_rng(int(os.path.basename(d)) * 2 + (var == 'b'))
        im, mk = (img, msk) if var == 'a' else (img[:, ::-1].copy(), msk[:, ::-1].copy())
        ys, xs = np.nonzero(mk); L = max(np.ptp(xs), np.ptp(ys)) + 1; cx, cy = xs.mean(), ys.mean()
        side = L * rng.uniform(1.8, 2.6); sh = rng.uniform(-0.15, 0.15, 2) * L
        ci, x0, y0, s = crop(im, cx + sh[0], cy + sh[1], side); cm, _, _, _ = crop(mk, cx + sh[0], cy + sh[1], side)
        E[i] = embed(ci)[0].numpy().astype(np.float16)
        M[i] = Fn.interpolate(torch.from_numpy(cm)[None, None], size=(256, 256), mode='bilinear', align_corners=False)[0, 0].numpy() > 0.5
        meta.append(dict(id=os.path.basename(d), var=var, x0=x0, y0=y0, side=s, acl_size=int(L))); i += 1
    print('done', d, flush=True)
E.flush(); np.save(f'{out}/cemrr_masks.npy', M); json.dump(meta, open(f'{out}/cemrr_meta.json', 'w'))
