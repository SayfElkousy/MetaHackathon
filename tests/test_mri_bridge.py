"""MRI upload bridge (tools/mri_bridge.py): KneeScan results -> modeler design state, and the reader client.

The reader itself (torch, model weights) is not needed: results are built in KneeScan's output shape
(mri_analysis/kneescan/engine.py analyze() plus the 'input' block app.py adds), and the Gradio API is
served by a small stand-in speaking the same protocol."""
import copy
import json
import os
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import mri_bridge as MB  # noqa: E402
from acl_engine.v2 import design as D  # noqa: E402


def kneescan_result(status="torn", level="complete rupture", length_mm=33.8, width_mm=7.7, bones=True, passed=True):
    r = {"volume": {"slices": 30, "rows": 256, "cols": 256, "spacing_mm": None},
         "acl_location": {"slice": 14, "box": [112, 98, 150, 140], "score": 0.81},
         "tear": {"probability": 0.931 if status == "torn" else 0.12, "threshold": 0.508, "status": status,
                  "level": level if status == "torn" else "none (intact)", "level_threshold": 0.906, "level_reliability": "low"},
         "acl": {"slice": 14, "outline": [[110.0, 139.0], [152.0, 99.0], [158.0, 104.0], [116.0, 144.0]],
                 "length_px": 66.2, "width_px": 15.1, "length_mm": length_mm, "width_mm": width_mm,
                 "end0": [110.2, 138.9], "end1": [152.0, 99.4], "confidence": 0.71,
                 "checks": {"elongated": True, "width_to_length_plausible": True, "model_confident": passed},
                 "passed_checks": passed, "mm_source": "estimated from reference knees"},
         "warnings": ["No pixel spacing in .npy files; millimetres are estimated from anatomy."],
         "seconds": 61.2,
         "validation": {"level": {"partial_vs_complete_auc": 0.644},
                        "acl_size": {"length_error_median_pct": 5.3, "width_error_median_pct": 14.5}},
         "input": {"file": "0007.npy", "format": "npy", "notes": []}}
    if bones:
        r["femur_fibula"] = {"slice": 22, "distance_px": 41.2, "distance_mm": 20.6, "mm_source": "estimated from reference knees",
                             "passed_checks": True, "experimental": True}
    return r


def row(rows, name):
    return next(x for x in rows if x["reading"] == name)


def test_complete_tear_drives_modeler_parameters():
    st, rows, doc, notes = MB.build_state(kneescan_result(), "right", "0007.npy", "job1")
    p, prov = st["params"], st["provenance"]
    assert doc["schema"] == "acl-mri-input" and doc["imaging"]["laterality"] == "right"
    assert p["case.side"] == "right"
    assert p["injury.type"] == "complete" and prov["injury.type"]["status"] == "inferred"
    # length -> target-length mode
    assert p["lig.length_mode"] == "target" and p["lig.length_target_mm"] == pytest.approx(33.8)
    assert prov["lig.length_target_mm"]["source"].startswith("KneeScan")
    # width -> template profile scaled so the mid-substance median is the reading
    assert p["lig.width_profile"]["s"] == D.S_KNOTS
    wp = p["lig.width_profile"]["v"]
    assert MB._mid_median(wp) == pytest.approx(7.7, abs=0.02)
    base = D.BY_ID["lig.width_profile"]["default"]
    assert wp[0] / wp[2] == pytest.approx(base[0] / base[2], rel=1e-3)
    # torn ligament: KneeScan warns that sizes describe outlined tissue -> awaiting_review
    assert prov["lig.width_profile"]["status"] == "awaiting_review"
    # the imported copy matches, so "reset" in the designer returns to the KneeScan values
    for pid in ("lig.length_mode", "lig.length_target_mm", "lig.width_profile", "injury.type"):
        assert st["imported"]["params"][pid] == p[pid]
    # recorded-only readings travel with the design
    rec = st["assets"]["mri_readings"]["readings"]
    assert rec["femur_fibula"]["value_mm"] == 20.6 and rec["acl_location"]["slice"] == 14
    assert row(rows, "ACL location")["result"] == "recorded"
    assert row(rows, "Femur to fibular head")["result"] == "recorded"
    warns = " ".join(st["imported"]["meta"]["report"]["warnings"])
    assert "fibular head distance 20.6 mm" in warns and "ACL location: slice 15" in warns
    assert not st["case"]["synthetic"]
    errs, _ = D.validate_params(dict(p))
    assert not errs


@pytest.mark.parametrize("status,level,expect", [("torn", "partial tear", "partial"), ("intact", None, "intact"),
                                                  ("torn", "complete rupture", "complete")])
def test_tear_status_and_level(status, level, expect):
    st, rows, _, _ = MB.build_state(kneescan_result(status, level or "none"), "left", "x.nii.gz")
    assert st["params"]["injury.type"] == expect
    assert st["params"]["case.side"] == "left"
    if expect == "partial":    # the modeler requires a torn region; KneeScan has none, so it is flagged for review
        assert st["params"]["injury.involvement"] == MB.PARTIAL_INVOLVEMENT
        assert st["provenance"]["injury.involvement"]["status"] == "awaiting_review"
    assert row(rows, "ACL tear level")["result"] == "applied"


def test_intact_and_clean_outline_is_measured():
    st, _, _, _ = MB.build_state(kneescan_result("intact"), "right", "a.npy")
    assert st["provenance"]["lig.width_profile"]["status"] == "measured"
    assert st["provenance"]["lig.length_target_mm"]["status"] == "measured"


def test_missing_millimetres_leave_modeler_defaults():
    r = kneescan_result(length_mm=None, width_mm=None, bones=False)
    st, rows, doc, notes = MB.build_state(r, "right", "b.npy")
    assert st["params"]["lig.length_mode"] == D.BY_ID["lig.length_mode"]["default"]
    assert st["params"]["lig.width_profile"] == D.BY_ID["lig.width_profile"]["default"]
    assert row(rows, "ACL length")["result"] == "not applied"
    assert row(rows, "ACL width")["result"] == "not applied"
    assert row(rows, "Femur to fibular head")["result"] == "not measured"
    assert any("no millimetre value" in n for n in notes)


def test_out_of_range_width_is_reported_not_clamped():
    st, rows, doc, notes = MB.build_state(kneescan_result(width_mm=1.2), "right", "c.npy")
    assert "ligament" not in doc
    assert st["params"]["lig.width_profile"] == D.BY_ID["lig.width_profile"]["default"]
    assert any("leaves the modeler's limits" in n for n in notes)


def test_length_check_reports_modeler_range():
    st, _, _, _ = MB.build_state(kneescan_result(length_mm=34.0), "right", "d.npy")
    lc = MB.length_check(st)
    assert lc["ok"] is True and lc["reached_mm"] == pytest.approx(34.0, abs=0.2)
    st, _, _, _ = MB.build_state(kneescan_result(length_mm=26.0), "right", "d.npy")
    lc = MB.length_check(st)
    assert lc["ok"] is False and "shorter than the shortest path" in lc["message"]


# --------------------------------------------------------------------------- reader client
class FakeKneeScan(BaseHTTPRequestHandler):
    """Speaks the Gradio 5 HTTP API the way mri_analysis/app.py exposes it."""
    result = None
    seen = {}

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _sse(self, *msgs):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for m in msgs:
            self.wfile.write(("data: %s\n\n" % json.dumps(m)).encode())

    def do_GET(self):
        if self.path == "/gradio_api/info":
            return self._json({"named_endpoints": {"/analyze": {}}, "unnamed_endpoints": {}})
        if self.path == "/config":
            return self._json({"version": "5.50.0", "dependencies": [{"id": 0, "api_name": "analyze", "targets": [[4, "click"]]}]})
        if self.path.startswith("/gradio_api/queue/data?session_hash="):
            ev = "ev1"
            head = [{"msg": "estimation", "event_id": ev, "rank": 0}, {"msg": "process_starts", "event_id": ev},
                    {"msg": "progress", "event_id": ev, "progress_data": [{"progress": 0.2, "desc": "Finding the ACL"}]}]
            if self.seen.get("reject"):
                return self._sse(*head, {"msg": "process_completed", "event_id": ev, "success": False,
                                         "output": {"error": "Only 5 slices; need a full sagittal stack."}}, {"msg": "close_stream", "event_id": None})
            fd = {"path": "/tmp/gradio/abc/kneescan_1.json", "url": "http://%s:%d/gradio_api/file=/tmp/gradio/abc/kneescan_1.json" % self.server.server_address,
                  "meta": {"_type": "gradio.FileData"}}
            return self._sse(*head, {"msg": "process_completed", "event_id": ev, "success": True,
                                     "output": {"data": ["<b>x</b>", None, None, fd]}}, {"msg": "close_stream", "event_id": None})
        if self.path.startswith("/gradio_api/file="):
            return self._json(self.result)
        self._json({"detail": "Not Found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        if self.path == "/gradio_api/upload":
            self.seen["upload_bytes"] = len(body)
            self.seen["upload_has_payload"] = b"SCANBYTES" in body
            return self._json(["/tmp/gradio/abc/scan.npy"])
        if self.path == "/gradio_api/queue/join":
            self.seen["call"] = json.loads(body)
            return self._json({"event_id": "ev1"})
        self._json({"detail": "Not Found"}, 404)


@pytest.fixture
def fake_reader():
    FakeKneeScan.result = kneescan_result()
    FakeKneeScan.seen = {}
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeKneeScan)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield "http://127.0.0.1:%d" % srv.server_address[1]
    srv.shutdown()


def test_reader_info(fake_reader):
    assert MB.reader_info(fake_reader)["ok"]
    down = MB.reader_info("http://127.0.0.1:9")
    assert not down["ok"] and "cannot reach" in down["error"]


def test_analyze_scan_round_trip(fake_reader, tmp_path):
    scan = tmp_path / "scan.npy"
    scan.write_bytes(b"SCANBYTES" * 100)
    steps = []
    r = MB.analyze_scan(str(scan), "scan.npy", True, fake_reader, lambda s, m: steps.append(s))
    assert r == FakeKneeScan.result
    assert FakeKneeScan.seen["upload_has_payload"]
    call = FakeKneeScan.seen["call"]
    assert call["fn_index"] == 0 and call["data"][0]["path"] == "/tmp/gradio/abc/scan.npy" and call["data"][1] is True
    assert steps == ["upload", "analyze", "analyze", "results"]      # second "analyze" = the reader's progress text


def test_reader_rejection_is_reported(fake_reader, tmp_path):
    FakeKneeScan.seen["reject"] = True
    scan = tmp_path / "scan.npy"
    scan.write_bytes(b"SCANBYTES")
    with pytest.raises(MB.ReaderError, match="Only 5 slices"):
        MB.analyze_scan(str(scan), "scan.npy", True, fake_reader)


def test_upload_route_end_to_end(fake_reader, tmp_path, monkeypatch):
    """Designer server: POST the scan, poll the job, get a saved design state the designer can open."""
    import server as SRV
    import server_v2 as V2
    monkeypatch.setattr(V2, "STORE", str(tmp_path / "store"))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), SRV.H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = "http://127.0.0.1:%d" % srv.server_address[1]
    try:
        html = urllib.request.urlopen(base + "/upload").read().decode()
        assert "Analyze and send to the modeler" in html
        st = json.load(urllib.request.urlopen(base + "/api/mri/reader?url=" + fake_reader))
        assert st["ok"]
        req = urllib.request.Request(base + "/api/mri/upload?name=0007.npy&side=right&bones=1&reader=" + fake_reader,
                                     data=b"SCANBYTES" * 1000, method="POST", headers={"Content-Type": "application/octet-stream"})
        jid = json.load(urllib.request.urlopen(req))["job"]
        for _ in range(100):
            job = json.load(urllib.request.urlopen(base + "/api/mri/job?id=" + jid))
            if job["status"] != "running":
                break
            time.sleep(0.1)
        assert job["status"] == "done", job.get("error")
        assert all(s["state"] == "done" for s in job["steps"])
        sid = job["result"]["state_id"]
        saved = json.load(urllib.request.urlopen(base + "/api/v2/state?id=" + sid))["state"]
        assert saved["params"]["injury.type"] == "complete"
        assert saved["params"]["lig.length_target_mm"] == pytest.approx(33.8)
        # bad requests are refused before anything is sent to the reader
        bad = urllib.request.Request(base + "/api/mri/upload?name=notes.txt&side=right", data=b"x", method="POST")
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(bad)
        assert e.value.code == 400
        noside = urllib.request.Request(base + "/api/mri/upload?name=a.npy", data=b"x", method="POST")
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(noside)
        assert e.value.code == 400
    finally:
        srv.shutdown()
