"""Shared MedSAM helpers (flaviagiammarino/medsam-vit-base via transformers)."""
import os, numpy as np, torch, torch.nn.functional as Fn
from transformers import SamModel
torch.set_num_threads(int(os.environ.get('NT', 1)))
_model = None
def model():
    global _model
    if _model is None: _model = SamModel.from_pretrained('flaviagiammarino/medsam-vit-base').eval()
    return _model
def to_square(img):
    """pad a 2D image to square (bottom/right) so aspect is kept; returns padded image and original (h, w)."""
    h, w = img.shape; s = max(h, w); out = np.zeros((s, s), np.float32); out[:h, :w] = img; return out, (h, w)
def prep(img):
    x = img.astype(np.float32); lo, hi = np.percentile(x, [0.5, 99.5]); x = np.clip((x - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)
    t = torch.from_numpy(x)[None, None]
    return Fn.interpolate(t, size=(1024, 1024), mode='bicubic', align_corners=False).clamp(0, 1).repeat(1, 3, 1, 1)
def embed(img):
    with torch.no_grad(): return model().get_image_embeddings(prep(img))
