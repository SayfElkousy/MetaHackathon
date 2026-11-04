"""Frozen pretrained 2D image features for KneeMRI ACL-region slices, then a regularised logistic regression per fold.
Extractors: DINOv2-base (Meta, general images; CLS + mean patch token) and BiomedCLIP (biomedical image-text model).
Features are averaged over the three central ROI slices. Writes out-of-fold tear probabilities.
usage: python -I km_feat2d.py"""
import sys, json, numpy as np, torch
sys.path.insert(0, '/home/user/acl_ext')
from km_folds import folds, O
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
torch.set_num_threads(1)
X2 = np.load(f'{O}/km_x2.npy')                                         # (N,3,224,224) in [0,1]
F, y, meta = folds()


def dinov2():
    from transformers import AutoModel
    m = AutoModel.from_pretrained('facebook/dinov2-base').eval()
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]; std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    def enc(batch):                                                    # batch (B,224,224)
        x = (torch.from_numpy(batch.astype(np.float32))[:, None].repeat(1, 3, 1, 1) - mean) / std
        with torch.no_grad(): h = m(pixel_values=x).last_hidden_state
        return torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).numpy()
    return enc


def biomedclip():
    import open_clip
    from PIL import Image
    m, pre = open_clip.create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'); m.eval()
    def enc(batch):
        x = torch.stack([pre(Image.fromarray((b * 255).astype(np.uint8)).convert('RGB')) for b in batch])
        with torch.no_grad(): return torch.nn.functional.normalize(m.encode_image(x), dim=-1).numpy()
    return enc


res = {}
for name, make in (('dinov2', dinov2), ('biomedclip', biomedclip)):
    enc = make(); feats = []
    for i in range(len(X2)):
        feats.append(enc(X2[i].astype(np.float32)).mean(0))
        if i % 200 == 0: print(name, i, flush=True)
    Fz = np.stack(feats); np.save(f'{O}/feat_{name}.npy', Fz)
    oof = np.zeros(len(y))
    for a, b in F:
        clf = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight='balanced')),
                           {'logisticregression__C': [0.003, 0.01, 0.03, 0.1, 0.3]}, scoring='roc_auc', cv=4)
        clf.fit(Fz[a], y[a]); oof[b] = clf.predict_proba(Fz[b])[:, 1]
    np.save(f'{O}/oof_{name}.npy', oof); res[name] = float(roc_auc_score(y, oof)); print(name, 'OOF AUC', round(res[name], 4), flush=True)
json.dump(res, open(f'{O}/feat2d_auc.json', 'w'), indent=1)
