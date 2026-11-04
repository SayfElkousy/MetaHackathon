"""External test: apply the KneeMRI-trained MONAI tear models (mean of the 5 fold models) to MRNet's ACL crops
(different hospital, scanner and sequence: T2 fat-suppressed vs PD fat-suppressed). MRNet scans carry no knee side,
so predictions are averaged over both slice orders and an anterior-posterior mirror.
usage: python -I km_on_mrnet.py seed [seed ...]"""
import sys, json, numpy as np, torch
from monai.networks.nets import resnet10
from sklearn.metrics import roc_auc_score
torch.set_num_threads(2)
X = np.load('/home/user/monai/crops.npy')[:, None].astype(np.float32)
meta = json.load(open('/home/user/monai/meta.json')); y = np.array([m['y'] for m in meta]); tr = np.where(y >= 0)[0]
P = np.zeros(len(y)); n = 0
for seed in sys.argv[1:]:
    for f in range(5):
        m = resnet10(spatial_dims=3, n_input_channels=1, num_classes=1, widen_factor=0.5)
        m.load_state_dict(torch.load(f'/home/user/acl_ext/km/monai_s{seed}_f{f}.pt', weights_only=True)); m.eval()
        with torch.no_grad():
            for i in range(0, len(X), 32):
                xb = torch.from_numpy(X[i:i + 32]); acc = 0
                for t in (xb, torch.flip(xb, dims=[2]), torch.flip(xb, dims=[4]), torch.flip(xb, dims=[2, 4])): acc = acc + torch.sigmoid(m(t))
                P[i:i + 32] += (acc / 4).squeeze(1).numpy()
        n += 1
P /= n; np.save('/home/user/acl_ext/km/mrnet_pred_from_kneemri.npy', P)
mon = np.load('/home/user/monai/prob.npy')
out = dict(auc_kneemri_model_on_mrnet=round(roc_auc_score(y[tr], P[tr]), 3), auc_mrnet_trained_monai_oof=round(roc_auc_score(y[tr], mon[tr]), 3), n=int(len(tr)), torn=int(y[tr].sum()))
print(json.dumps(out)); json.dump(out, open('/home/user/acl_ext/km/external_mrnet.json', 'w'))
