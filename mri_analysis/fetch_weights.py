"""Download KneeScan's model weights (about 470 MB) and two example scans from the private Hugging Face repo
tariksyed/kneescan into this folder. Needs a Hugging Face token with read access to that repo:
    HF_TOKEN=hf_xxx python fetch_weights.py        (or run it and paste the token when asked)"""
import os
from getpass import getpass
from huggingface_hub import snapshot_download

here = os.path.dirname(os.path.abspath(__file__))
token = os.environ.get('HF_TOKEN') or getpass('Hugging Face read token: ')
snapshot_download('tariksyed/kneescan', token=token, local_dir=here, allow_patterns=['weights/*', 'weights/**', 'examples/*'])
print('weights and examples are in', here)
