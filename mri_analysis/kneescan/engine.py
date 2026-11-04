"""KneeScan engine: one sagittal knee MRI in, ACL and femur-fibula findings out.

Steps (each uses the model that did best in validation for that job):
  1. ACL location      - SegFormer ACL localizer (Hugging Face: hajirabaloch488/knee-mri-acl-localization) on every
                         slice; the ACL slice is the central slice with the strongest, most compact detection.
  2. Tear status       - MONAI 3D ResNet10 ensemble (5 models) trained on KneeMRI (917 PD fat-suppressed scans).
  3. Tear level        - read from the tear probability (partial vs complete AUC 0.68; a dedicated 3-class model did worse).
  4. ACL length/width  - MedSAM (built on Meta's SAM) with a mask decoder fine-tuned on 110 expert ACL outlines (CEMRR).
  5. Femur-fibula      - registration to 3 hand-labelled reference knees for the fibular head and lateral femoral
                         condyle, then MedSAM outlines; distance = shortest gap between the two outlines. Experimental.
Millimetres come from DICOM/NIfTI pixel spacing when present; otherwise they are estimated from anatomy."""
import os, json, time, numpy as np, torch, torch.nn.functional as Fn
from scipy import ndimage as ndi
from skimage import measure as skm
from .acl_measure import measure as acl_shape, largest

D = os.environ.get('KNEESCAN_WEIGHTS', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'weights'))


def _norm(v):
    lo, hi = np.percentile(v, [1, 99.5]); return np.clip((v - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)


def _resize2d(a, h, w, mode='bilinear'):
    t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None, None]
    kw = {'align_corners': False} if mode in ('bilinear', 'bicubic') else {}
    return Fn.interpolate(t, size=(h, w), mode=mode, **kw)[0, 0].numpy()


def _crop(a, x0, y0, s):
    p = s + 2; ap = np.pad(a, p); return ap[y0 + p:y0 + p + s, x0 + p:x0 + p + s]


def _poly(mask, f=1.0, ox=0.0, oy=0.0):
    cs = skm.find_contours(mask.astype(float), 0.5)
    if not cs: return None
    c = skm.approximate_polygon(max(cs, key=len), tolerance=0.8)
    return [[round(ox + x * f, 1), round(oy + y * f, 1)] for y, x in c]


class KneeScan:
    def __init__(self, weights_dir=None, threads=None):
        self.D = weights_dir or D
        if threads: torch.set_num_threads(threads)
        self.dev = torch.device('cuda' if torch.cuda.is_available() and not os.environ.get('KNEESCAN_CPU') else 'cpu')
        self.loaded = False

    # ------------------------------------------------------------------ loading
    def load(self):
        from transformers import SegformerForSemanticSegmentation, SamModel
        from monai.networks.nets import resnet10
        self.loc = SegformerForSemanticSegmentation.from_pretrained(f'{self.D}/acl_localizer', num_labels=2).eval().to(self.dev)
        self.sam = SamModel.from_pretrained(f'{self.D}/medsam').eval().to(self.dev)
        self.sam_dec_base = {k: v.detach().cpu().clone() for k, v in self.sam.mask_decoder.state_dict().items()}
        self.acl_dec = torch.load(f'{self.D}/acl_decoder.pt', weights_only=True, map_location='cpu')
        self.cal = json.load(open(f'{self.D}/calibration.json'))

        def ens(prefix, nc):
            ms = []
            for f in range(5):
                p = f'{self.D}/{prefix}_f{f}.pt'
                if not os.path.exists(p): continue
                m = resnet10(spatial_dims=3, n_input_channels=1, num_classes=nc, widen_factor=0.5)
                m.load_state_dict(torch.load(p, weights_only=True, map_location='cpu')); ms.append(m.eval().to(self.dev))
            return ms
        self.tear_models = ens('tear/monai', 1)
        self.templates = json.load(open(f'{self.D}/templates/templates.json'))
        self.loaded = True
        return self

    # ------------------------------------------------------------------ 1. ACL location
    def localize(self, v):
        """v: normalised (S,H,W). Returns chosen slice, box (x0,y0,x1,y1) in pixels, and per-slice scores."""
        S, H, W = v.shape; rows = []
        with torch.no_grad():
            for k in range(S):
                x = v[k]; x = (x - x.min()) / (x.max() - x.min() + 1e-6)
                t = Fn.interpolate(torch.from_numpy(x)[None, None], size=(224, 224), mode='bilinear', align_corners=False).repeat(1, 3, 1, 1).to(self.dev)
                p = torch.softmax(Fn.interpolate(self.loc(pixel_values=t).logits, size=(H, W), mode='bilinear', align_corners=False), 1)[0, 1].cpu().numpy()
                m = largest(p > 0.5)
                if m.sum() == 0: rows.append(None); continue
                ys, xs = np.nonzero(m)
                rows.append(dict(k=k, box=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], mean_p=float(p[m].mean()), area=int(m.sum())))
        # The ACL sits in the intercondylar notch, near the middle of a sagittal stack (KneeMRI: within 2 slices of the
        # centre in 99% of 917 scans). The localizer also fires on the femoral condyles, so weight its score by a
        # Gaussian prior around the centre slice.
        c, sd = (S - 1) / 2.0, max(1.5, 0.07 * S)
        cand = [r for r in rows if r]
        if not cand: return None
        best = max(cand, key=lambda r: r['mean_p'] * np.exp(-0.5 * ((r['k'] - c) / sd) ** 2))
        # smooth the box over neighbouring slices that agree
        nb = [r for r in rows[max(0, best['k'] - 1):best['k'] + 2] if r]
        box = np.median([r['box'] for r in nb], 0).round().astype(int).tolist()
        return dict(slice=best['k'], box=box, score=round(best['mean_p'], 3), per_slice=[None if r is None else round(r['mean_p'], 3) for r in rows])

    # ------------------------------------------------------------------ 2-3. tear status and level
    def _tear_crop(self, v, k, box, D=9, S=80):
        x0, y0, x1, y1 = box; w, h = x1 - x0 + 1, y1 - y0 + 1
        side = int(round(1.3 * max(w, h))); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        X0, Y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
        ks = np.clip(np.arange(k - D // 2, k + D // 2 + 1), 0, v.shape[0] - 1)
        crop = np.stack([_crop(v[kk], X0, Y0, side) for kk in ks])
        t = Fn.interpolate(torch.from_numpy(crop)[:, None], size=(S, S), mode='area')[:, 0]
        return t[None, None].float()                                     # (1,1,9,80,80)

    def tear(self, v, k, box):
        xb = self._tear_crop(v, k, box).to(self.dev)
        views = [xb, torch.flip(xb, dims=[2]), torch.flip(xb, dims=[4]), torch.flip(xb, dims=[2, 4])]   # knee side unknown
        with torch.no_grad():
            p = float(np.mean([torch.sigmoid(m(t)).item() for m in self.tear_models for t in views]))
        c = self.cal['tear']; torn = p >= c['threshold']
        out = dict(probability=round(p, 3), threshold=round(c['threshold'], 3), status='torn' if torn else 'intact')
        # Tear level: the tear probability separated partial from complete better (AUC 0.68) than a dedicated 3-class
        # MONAI model (0.61) on KneeMRI, so the level is read from it. Low reliability.
        lt = self.cal['level']['threshold']
        out['level'] = ('complete rupture' if p >= lt else 'partial tear') if torn else 'none (intact)'
        out['level_threshold'] = round(lt, 3); out['level_reliability'] = 'low'
        return out

    # ------------------------------------------------------------------ 4. ACL outline, length, width
    def _sam_embed(self, img):
        x = img.astype(np.float32); lo, hi = np.percentile(x, [0.5, 99.5]); x = np.clip((x - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)
        t = Fn.interpolate(torch.from_numpy(x)[None, None], size=(1024, 1024), mode='bicubic', align_corners=False).clamp(0, 1).repeat(1, 3, 1, 1)
        with torch.no_grad(): return self.sam.get_image_embeddings(t.to(self.dev))

    def _sam_masks(self, emb, boxes256, decoder):
        self.sam.mask_decoder.load_state_dict(decoder)
        b = torch.tensor([[[c * 4.0 for c in bx] for bx in boxes256]], dtype=torch.float32, device=self.dev)
        with torch.no_grad():
            out = self.sam(image_embeddings=emb, input_boxes=b, multimask_output=False)
        p = torch.sigmoid(out.pred_masks[0, :, 0]).cpu().numpy()
        return [largest(pi > 0.5) for pi in p], out.iou_scores[0, :, 0].cpu().numpy()

    def acl(self, v, k, box, mm_per_px):
        S, H, W = v.shape; x0, y0, x1, y1 = box
        side = int(round(np.clip(1.6 * max(x1 - x0 + 1, y1 - y0 + 1), 0.3 * min(H, W), 0.7 * min(H, W))))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; X0, Y0 = int(round(cx - side / 2)), int(round(cy - side / 2)); f = side / 256.0
        b256 = np.clip([(x0 - X0) / f, (y0 - Y0) / f, (x1 - X0) / f, (y1 - Y0) / f], 0, 255).tolist()
        best = None
        for kk in sorted({max(0, k - 1), k, min(S - 1, k + 1)}):
            emb = self._sam_embed(_crop(v[kk], X0, Y0, side))
            (mk,), (iou,) = self._sam_masks(emb, [b256], self.acl_dec)
            met = acl_shape(mk)
            if met is None or met['width_mid'] is None: continue
            sc = float(iou) + 0.002 * met['length']
            if best is None or sc > best[0]: best = (sc, kk, mk, float(iou), met)
        if best is None: return dict(error='no ACL outline found')
        _, kk, mk, iou, met = best
        L, Wd = met['length'] * f, met['width_mid'] * f
        ratio = met['width_mid'] / met['length']; checks = dict(elongated=bool(met['elong'] >= 2.2), width_to_length_plausible=bool(ratio <= 0.41),
                                                               model_confident=bool(iou >= 0.55))
        e0 = [X0 + met['end0'][0] * f, Y0 + met['end0'][1] * f]; e1 = [X0 + met['end1'][0] * f, Y0 + met['end1'][1] * f]
        return dict(slice=int(kk), outline=_poly(mk, f, X0, Y0), length_px=round(L, 1), width_px=round(Wd, 1),
                    length_mm=round(L * mm_per_px, 1) if mm_per_px else None, width_mm=round(Wd * mm_per_px, 1) if mm_per_px else None,
                    end0=[round(c, 1) for c in e0], end1=[round(c, 1) for c in e1], confidence=round(iou, 3),
                    checks=checks, passed_checks=all(checks.values()))

    # ------------------------------------------------------------------ 5. femur-fibula
    def _register_landmarks(self, v256):
        import SimpleITK as sitk
        from .reglib import to_img, register, to_fixed, inplane_scale
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(max(1, torch.get_num_threads()))
        T = {sid: to_img(_norm(np.load(f'{self.D}/templates/{sid}.npy').astype(np.float32))) for sid in self.templates}
        trials = []
        for flip in (False, True):
            vv = np.ascontiguousarray(v256[::-1]) if flip else v256
            _, m = register(to_img(vv), T['0016'], fast=True); trials.append((m, flip))
        flip = min(trials)[1]; vv = np.ascontiguousarray(v256[::-1]) if flip else v256; fixed = to_img(vv)
        est, met, sc = {}, {}, {}
        for sid, t in self.templates.items():
            try: tr, mm = register(fixed, T[sid])
            except Exception: continue
            met[sid] = float(mm); est[sid] = {k: to_fixed(tr, t['lm'][k]) for k in ('FIB', 'FEM_L', 'ACL_F', 'ACL_T')}
            sc[sid] = t['mmpx'] * inplane_scale(tr, (est[sid]['ACL_F'] + est[sid]['ACL_T']) / 2)
        if not met: return None
        b = min(met.values()); keep = [s for s in met if met[s] <= b + 0.06]
        lm = {k: np.median([est[s][k] for s in keep], 0) for k in ('FIB', 'FEM_L')}
        return dict(flip=flip, lm=lm, mmpx256=float(np.median([sc[s] for s in keep])), metric=b, vv=vv)

    def femur_fibula(self, v, mm_per_px):
        S, H, W = v.shape
        v256 = np.stack([_resize2d(v[k], 256, 256) for k in range(S)])
        r = self._register_landmarks(v256)
        if r is None: return dict(error='could not align to the reference knees'), None
        vv, FIB, FEM = r['vv'], r['lm']['FIB'], r['lm']['FEM_L']
        clip = lambda b: [float(np.clip(c, 0, 255)) for c in b]
        fb = clip([FIB[0] - 28, FIB[1] - 6, FIB[0] + 28, FIB[1] + 46]); eb = clip([FEM[0] - 75, FEM[1] - 65, FEM[0] + 25, FEM[1] + 6])
        tb = clip([FIB[0] - 115, FEM[1] + 2, FIB[0] + 2, FEM[1] + 85])
        kf = int(round(FIB[2])); best = None
        for kk in (kf - 1, kf, kf + 1):
            if kk < 0 or kk >= S: continue
            (fem, fib, tib), iou = self._sam_masks(self._sam_embed(vv[kk]), [eb, fb, tb], self.sam_dec_base)
            if fem.sum() == 0 or fib.sum() == 0: continue
            dt, ind = ndi.distance_transform_edt(~fem, return_indices=True)
            ys, xs = np.nonzero(fib); d = dt[ys, xs]; j = int(np.argmin(d))
            pb = (float(xs[j]), float(ys[j])); pa = (float(ind[1][ys[j], xs[j]]), float(ind[0][ys[j], xs[j]]))
            chk = self._bone_checks(fem, fib, tib, vv[kk], iou)
            sc = float(iou[1]) + 0.5 * float(iou[0]) + 0.3 * sum(chk.values())
            if best is None or sc > best[0]: best = (sc, kk, float(d[j]), pa, pb, fem, fib, tib, chk, iou)
        if best is None: return dict(error='no femur/fibula pair found'), None
        _, kk, d256, pa, pb, fem, fib, tib, chk, iou = best
        sx, sy = W / 256.0, H / 256.0; s_iso = (sx + sy) / 2
        k_orig = (S - 1 - kk) if r['flip'] else kk
        mm = mm_per_px if mm_per_px else r['mmpx256'] / s_iso                 # estimated: reference-knee scale
        to_orig = lambda p: [round(p[0] * sx, 1), round(p[1] * sy, 1)]
        out = dict(slice=int(k_orig), distance_px=round(d256 * s_iso, 1), distance_mm=round(d256 * s_iso * mm, 1),
                   mm_source='pixel spacing' if mm_per_px else 'estimated from reference knees',
                   femur_point=to_orig(pa), fibula_point=to_orig(pb),
                   femur_outline=_poly(fem, sx, 0, 0) if sx == sy else [[round(x * sx, 1), round(y * sy, 1)] for x, y in (_poly(fem) or [])],
                   fibula_outline=[[round(x * sx, 1), round(y * sy, 1)] for x, y in (_poly(fib) or [])],
                   tibia_outline=[[round(x * sx, 1), round(y * sy, 1)] for x, y in (_poly(tib) or [])],
                   confidence=dict(femur=round(float(iou[0]), 3), fibula=round(float(iou[1]), 3)),
                   checks=chk, passed_checks=all(chk.values()), experimental=True)
        return out, r['mmpx256'] / s_iso

    @staticmethod
    def _bone_checks(fem, fib, tib, img, iou):
        fy, fx = np.nonzero(fib); ey = np.nonzero(fem)[0]; c = {}
        c['fibula below femur'] = bool(fy.mean() > ey.max())
        c['fibula separate from tibia'] = bool((fib & tib).sum() / max(1, fib.sum()) < 0.25)
        c['outlines do not overlap'] = bool((fem & fib).sum() == 0)
        c['fibula size plausible'] = bool(120 <= fib.sum() <= 2200)
        c['fibular apex inside image'] = bool(fy.min() < 238)
        p = skm.regionprops(fib.astype(np.uint8))[0]
        c['fibula is round, not a sliver'] = bool(p.solidity >= 0.7 and p.axis_minor_length >= 15)
        ratio = float(np.median(img[fib]) / max(1e-3, np.median(img[tib]))) if tib.any() else 1.0
        c['fibula looks like bone marrow'] = bool(0.6 <= ratio <= 1.7)
        c['model confident'] = bool(iou[0] > 0.5 and iou[1] > 0.5)
        return c

    # ------------------------------------------------------------------ everything
    def analyze(self, vol, spacing=None, do_bones=True):
        if not self.loaded: self.load()
        t0 = time.time(); v = _norm(vol); S, H, W = v.shape; warn = []
        mmpx = float(np.mean(spacing[1:])) if spacing and spacing[1] and spacing[2] else None
        loc = self.localize(v)
        if loc is None: return dict(error='Could not find the ACL region on any slice.')
        res = dict(volume=dict(slices=S, rows=H, cols=W, spacing_mm=list(spacing) if spacing else None),
                   acl_location=dict(slice=loc['slice'], box=loc['box'], score=loc['score']))
        res['tear'] = self.tear(v, loc['slice'], loc['box'])
        bones, est_mmpx = (self.femur_fibula(v, mmpx) if do_bones else (None, None))
        mm_for_acl = mmpx or est_mmpx
        res['acl'] = self.acl(v, loc['slice'], loc['box'], mm_for_acl)
        if not mmpx and res['acl'].get('length_mm') is not None: res['acl']['mm_source'] = 'estimated from reference knees'
        elif mmpx: res['acl']['mm_source'] = 'pixel spacing'
        if bones is not None: res['femur_fibula'] = bones
        if res['tear']['status'] == 'torn': warn.append('ACL predicted torn: length and width of a torn ligament describe the outlined tissue, not an intact ACL.')
        if not res['acl'].get('passed_checks', False): warn.append('ACL outline failed a shape check; review it before using the length/width.')
        if bones is not None and not bones.get('passed_checks', False): warn.append('Femur-fibula outlines failed a check; review the image.')
        if not mmpx: warn.append('No pixel spacing in this file: millimetres are estimated from anatomy (DICOM or NIfTI gives real millimetres).')
        res['warnings'] = warn; res['seconds'] = round(time.time() - t0, 1)
        res['validation'] = self.cal.get('validation', {})
        return res
