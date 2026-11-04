"""The workspace server starts the KneeScan MRI reader itself (tools/reader_launcher.py).

The real reader needs PyTorch and the model weights; here a tiny stand-in process answers like a Gradio app."""
import os
import socket
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import mri_bridge as MB  # noqa: E402
import reader_launcher as RL  # noqa: E402

STUB = r'''
import json, os
from http.server import BaseHTTPRequestHandler, HTTPServer
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        b = json.dumps({"named_endpoints": {"/analyze": {}}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
print("stand-in reader up", flush=True)
HTTPServer(("127.0.0.1", int(os.environ["KNEESCAN_PORT"])), H).serve_forever()
'''


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture
def home(tmp_path, monkeypatch):
    """An empty KneeScan home on a free port, with the launcher's state reset."""
    port = free_port()
    monkeypatch.setattr(RL, "HOME", str(tmp_path))
    monkeypatch.setattr(RL, "VENV_PY", str(tmp_path / "venv" / "python"))
    monkeypatch.setattr(RL, "WEIGHTS", str(tmp_path / "weights"))
    monkeypatch.setattr(RL, "LOG", str(tmp_path / "kneescan.log"))
    monkeypatch.setattr(RL, "PORT", port)
    monkeypatch.setattr(RL, "LOCAL_URL", "http://127.0.0.1:%d" % port)
    monkeypatch.delenv("KNEESCAN_URL", raising=False)
    monkeypatch.delenv("KNEESCAN_AUTOSTART", raising=False)
    RL._state.update(state="off", message="", pid=None, started=None)
    yield tmp_path
    RL.stop()


def test_not_set_up_says_how_to_set_up(home):
    gone = RL.missing()
    assert any("Python environment" in m for m in gone) and any("model weights" in m for m in gone)
    st = RL.autostart(answering=False)
    assert st["state"] == "not_installed" and "python tools/setup_kneescan.py" in st["message"]
    # the page's reader check carries the same message instead of a bare "not reachable"
    info = MB.reader_info(RL.LOCAL_URL, timeout=1)
    assert not info["ok"] and info["launcher"]["state"] == "not_installed" and "setup_kneescan" in info["error"]


def test_switches_and_an_existing_reader(home, monkeypatch):
    monkeypatch.setenv("KNEESCAN_AUTOSTART", "0")
    assert RL.autostart(answering=False)["state"] == "off"
    monkeypatch.delenv("KNEESCAN_AUTOSTART")
    monkeypatch.setenv("KNEESCAN_URL", "https://abc123.gradio.live")
    assert RL.autostart(answering=False)["state"] == "off"
    monkeypatch.delenv("KNEESCAN_URL")
    assert RL.autostart(answering=True)["state"] == "external"          # started by hand: left alone


def test_detects_a_complete_setup(home):
    (home / "venv").mkdir()
    (home / "venv" / "python").write_text("")
    for f in RL.REQUIRED + (os.path.join("tear", "monai_f0.pt"),):
        p = home / "weights" / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    assert RL.missing() == []


def test_starts_reports_and_stops_the_reader(home, monkeypatch):
    monkeypatch.setattr(RL, "missing", lambda: [])
    monkeypatch.setattr(RL, "_command", lambda: [sys.executable, "-c", STUB])
    st = RL.autostart(answering=False)
    assert st["state"] == "starting" and st["pid"]
    for _ in range(100):
        info = MB.reader_info(RL.LOCAL_URL, timeout=1)
        if info["ok"]:
            break
        assert info["launcher"]["state"] == "starting" and "starting" in info["error"]
        time.sleep(0.1)
    assert info["ok"] and info["launcher"]["state"] == "running"
    proc = RL._proc
    RL.stop()
    assert proc.poll() is not None and RL.status()["state"] == "off"
    assert not MB.reader_info(RL.LOCAL_URL, timeout=1)["ok"]


def test_a_reader_that_crashes_is_reported(home, monkeypatch):
    monkeypatch.setattr(RL, "missing", lambda: [])
    monkeypatch.setattr(RL, "_command", lambda: [sys.executable, "-c", "print('no weights here'); raise SystemExit(3)"])
    RL.autostart(answering=False)
    for _ in range(100):
        if RL._proc.poll() is not None:
            break
        time.sleep(0.05)
    st = RL.status()
    assert st["state"] == "exited" and "exit code 3" in st["message"] and "no weights here" in " ".join(st["log_tail"])
