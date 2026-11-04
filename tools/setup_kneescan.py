"""One-time setup of the KneeScan MRI reader on this computer, so the workspace server starts it by itself.

    python tools/setup_kneescan.py               environment + libraries + model weights (asks for a Hugging Face token)
    python tools/setup_kneescan.py --no-weights  environment + libraries only

Installs into KNEESCAN_HOME (default %LOCALAPPDATA%\\KneeScan on Windows, ~/.kneescan elsewhere), outside the project
folder: a Python environment (venv/, the libraries in mri_analysis/requirements.txt, about 1 GB) and the model weights
(weights/, about 470 MB) from the private Hugging Face repo tariksyed/kneescan - the same download as
mri_analysis/fetch_weights.py, which needs a read token for that repo. Safe to re-run: finished steps are skipped.
Afterwards:  python tools/server.py  (the reader starts in the background; the upload window shows "connected").
"""
import os
import shutil
import subprocess
import sys
import venv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reader_launcher as RL  # noqa: E402

REPO = "tariksyed/kneescan"
DOWNLOAD = ("import os, sys\n"
            "from getpass import getpass\n"
            "from huggingface_hub import snapshot_download\n"
            "token = os.environ.get('HF_TOKEN') or getpass('Hugging Face read token for %s: ')\n"
            "snapshot_download('%s', token=token or None, local_dir=sys.argv[1], allow_patterns=['weights/*', 'weights/**', 'examples/*'])\n"
            % (REPO, REPO))


def step(n, text):
    print("\n[%d/3] %s" % (n, text), flush=True)


def run(cmd):
    print("  $ " + " ".join('"%s"' % c if " " in c else c for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    no_weights = "--no-weights" in sys.argv
    print("KneeScan home: %s" % RL.HOME)
    os.makedirs(RL.HOME, exist_ok=True)

    step(1, "Python environment")
    if os.path.isfile(RL.VENV_PY):
        print("  already there: " + RL.VENV)
    else:
        venv.EnvBuilder(with_pip=True).create(RL.VENV)
        print("  created " + RL.VENV)

    step(2, "Libraries (mri_analysis/requirements.txt: PyTorch CPU, MONAI, transformers, Gradio, ...)")
    run([RL.VENV_PY, "-m", "pip", "install", "--disable-pip-version-check", "-r", os.path.join(RL.MRI, "requirements.txt")])

    step(3, "Model weights from the private Hugging Face repo " + REPO)
    if not [m for m in RL.missing() if "weights" in m]:
        print("  already there: " + RL.WEIGHTS)
    elif no_weights:
        print("  skipped (--no-weights). Run this script again without it to download them.")
    else:
        print("  Needs a Hugging Face read token for %s (set HF_TOKEN, or paste it when asked; it is not stored)." % REPO)
        run([RL.VENV_PY, "-c", DOWNLOAD, RL.HOME])
        cal = os.path.join(RL.WEIGHTS, "calibration.json")
        if not os.path.isfile(cal):                     # versioned in the repo; the reader needs it next to the weights
            shutil.copyfile(os.path.join(RL.MRI, "weights", "calibration.json"), cal)

    gone = RL.missing()
    if gone:
        print("\nNot finished yet - missing: " + "; ".join(gone))
        return 1
    print("\nDone. Start the workspace with:  python tools/server.py\n"
          "It starts KneeScan in the background (loading the models takes about a minute); the upload window then shows 'connected'.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as e:
        print("\nA step failed (exit code %s). Fix the message above and run this script again." % e.returncode)
        sys.exit(1)
