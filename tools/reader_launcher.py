"""Start the KneeScan MRI reader (mri_analysis/app.py) next to the workspace server, so the page finds it by itself.

Set up once with  python tools/setup_kneescan.py  (a Python environment and the model weights in KNEESCAN_HOME).
After that  python tools/server.py  starts the reader in the background on 127.0.0.1:7860 and stops it on exit.
The reader is run the way its Colab notebook runs it (import app, load the models, launch), so mri_analysis/ is
not changed. Skipped when KNEESCAN_AUTOSTART=0, when KNEESCAN_URL points elsewhere, or when a reader already answers.
"""
import atexit
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MRI = os.path.join(ROOT, "mri_analysis")
# Outside the project folder on purpose: it may be synced (OneDrive), and the environment + weights are ~1.5 GB.
HOME = os.environ.get("KNEESCAN_HOME") or (os.path.join(os.environ["LOCALAPPDATA"], "KneeScan") if os.name == "nt" and os.environ.get("LOCALAPPDATA")
                                           else os.path.join(os.path.expanduser("~"), ".kneescan"))
VENV = os.path.join(HOME, "venv")
VENV_PY = os.path.join(VENV, "Scripts", "python.exe") if os.name == "nt" else os.path.join(VENV, "bin", "python")
WEIGHTS = os.path.join(HOME, "weights")
LOG = os.path.join(HOME, "kneescan.log")
PORT = 7860
LOCAL_URL = "http://127.0.0.1:%d" % PORT
# What KneeScan's engine loads (mri_analysis/kneescan/engine.py KneeScan.load)
REQUIRED = ("calibration.json", "acl_decoder.pt", os.path.join("acl_localizer", "config.json"),
            os.path.join("medsam", "config.json"), os.path.join("templates", "templates.json"))
LAUNCH = ("import os, sys; sys.path.insert(0, os.getcwd()); import app; "
          "print('KneeScan: loading models', flush=True); app.ENGINE.load(); "
          "print('KneeScan: models loaded, starting', flush=True); "
          "app.demo.queue(max_size=8).launch(server_name='127.0.0.1', server_port=int(os.environ['KNEESCAN_PORT']), show_error=True)")

_lock = threading.Lock()
_state = {"state": "off", "message": "", "pid": None, "started": None}
_proc = None


def missing():
    """What is not set up yet: [] when the reader can be started."""
    out = []
    if not os.path.isfile(VENV_PY):
        out.append("Python environment (%s)" % VENV)
    gone = [f for f in REQUIRED if not os.path.isfile(os.path.join(WEIGHTS, f))]
    if not os.path.isdir(os.path.join(WEIGHTS, "tear")) or not any(n.startswith("monai_f") for n in os.listdir(os.path.join(WEIGHTS, "tear"))):
        gone.append(os.path.join("tear", "monai_f*.pt"))
    if gone:
        out.append("model weights in %s (%s)" % (WEIGHTS, ", ".join(gone)))
    return out


def _command():
    return [VENV_PY, "-c", LAUNCH]


def _env():
    env = dict(os.environ, KNEESCAN_WEIGHTS=WEIGHTS, KNEESCAN_PORT=str(PORT), GRADIO_ANALYTICS_ENABLED="False",
               PYTHONUNBUFFERED="1")
    env.pop("KNEESCAN_URL", None)
    return env


def _set(state, message, **kw):
    _state.update(state=state, message=message, **kw)


def wanted():
    """Start the local reader? Not when switched off or when the reader address points somewhere else."""
    if os.environ.get("KNEESCAN_AUTOSTART", "1") == "0":
        return False, "automatic start is off (KNEESCAN_AUTOSTART=0)"
    url = os.environ.get("KNEESCAN_URL")
    if url and url.rstrip("/") not in (LOCAL_URL, "http://localhost:%d" % PORT):
        return False, "KNEESCAN_URL points to another reader (%s)" % url
    return True, ""


def autostart(answering=None):
    """Start the reader unless it is not wanted, already answering, or not set up. Returns the status dict."""
    global _proc
    with _lock:
        ok, why = wanted()
        if not ok:
            _set("off", why)
            return status()
        if answering is None:
            import mri_bridge as MB
            answering = MB.reader_info(LOCAL_URL, timeout=2).get("ok")
        if answering:
            _set("external", "a KneeScan reader is already running at %s" % LOCAL_URL)
            return status()
        gone = missing()
        if gone:
            _set("not_installed", "KneeScan is not set up on this computer yet (missing: %s). Run once:  python tools/setup_kneescan.py"
                 % "; ".join(gone), missing=gone)
            return status()
        if _proc is not None and _proc.poll() is None:
            return status()
        os.makedirs(HOME, exist_ok=True)
        log = open(LOG, "w", encoding="utf-8", errors="replace")
        _proc = subprocess.Popen(_command(), cwd=MRI, env=_env(), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        _set("starting", "KneeScan is starting on this computer (loading its models; about a minute on a CPU)",
             pid=_proc.pid, started=time.time())
        return status()


def stop():
    global _proc
    p, _proc = _proc, None
    if p is not None and p.poll() is None:
        p.terminate()
        try:
            p.wait(10)
        except subprocess.TimeoutExpired:
            p.kill()
    if _state["state"] in ("starting", "running"):
        _set("off", "stopped with the workspace server")


atexit.register(stop)


def _log_tail(n=6):
    try:
        with open(LOG, encoding="utf-8", errors="replace") as f:
            lines = [ln.rstrip() for ln in f if ln.strip()]
        return lines[-n:]
    except OSError:
        return []


def status(answering=None):
    """Launcher state for the page: off | external | not_installed | starting | running | exited."""
    st = dict(_state)
    if _proc is not None and st["state"] in ("starting", "running"):
        code = _proc.poll()
        if code is not None:
            _set("exited", "KneeScan stopped (exit code %s); see %s" % (code, LOG))
            st = dict(_state)
        elif answering:
            _set("running", "KneeScan is running on this computer (%s)" % LOCAL_URL)
            st = dict(_state)
    if st["state"] == "exited":
        st["log_tail"] = _log_tail()
    st["home"] = HOME
    return st
