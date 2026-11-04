"""Is the end-to-end drop (0.90 -> 0.85) caused by not knowing the knee side? Score the hand-ROI crops with both slice
orders averaged (as the deployed system must), using each scan's held-out fold model."""
import sys, os, json, numpy as np, pandas as pd, torch, warnings
sys.path.insert(0, '/home/user/kneescan_system'); sys.path.insert(0, '/home/user/acl_ext'); warnings.filterwarnings('ignore')
from km_folds import folds
from sklearn.metrics import roc_auc_score
from monai.networks.nets import resnet10
torch.set_num_threads(2)
F, y, meta = folds(); X = np.load('/home/user/acl_ext/km/km_x3.npy')[:, None]
pb, pa = np.zeros(len(y)), np.zeros(len(y))
for f, (_, vi) in enumerate(F):
    m = resnet10(spatial_dims=3, n_input_channels=1, num_classes=1, widen_factor=0.5)
    m.load_state_dict(torch.load(f'/home/user/acl_ext/km/monai_s0_f{f}.pt', weights_only=True)); m.eval()
    with torch.no_grad():
        for i in vi:
            xb = torch.from_numpy(X[i:i + 1].astype(np.float32))            # side-standardised crop from the hand ROI
            pa[i] = float(np.mean([torch.sigmoid(m(t)).item() for t in (xb, torch.flip(xb, dims=[4]))]))
            pb[i] = float(np.mean([torch.sigmoid(m(t)).item() for t in (xb, torch.flip(xb, dims=[2]), torch.flip(xb, dims=[4]), torch.flip(xb, dims=[2, 4]))]))
print(json.dumps(dict(hand_roi_known_side=round(roc_auc_score(y, pa), 3), hand_roi_side_unknown=round(roc_auc_score(y, pb), 3))))
