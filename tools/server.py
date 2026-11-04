"""N3 local server. Stdlib only. 127.0.0.1 only.

Run:  python tools/server.py      (opens http://127.0.0.1:8000/)
Written by Muse (N3). Fixes by Claude are marked "FIX (Claude)".
"""
import base64
import copy
import json
import os
import re
import sys
import tempfile
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# FIX (Claude): "python tools/server.py" puts tools/ on the path, not the project root,
# so every build failed with "No module named acl_engine".
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
TOOLS = os.path.dirname(os.path.abspath(__file__))
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)
WEB = os.path.join(ROOT, "tools", "web")
FIX = os.path.join(ROOT, "fixtures")
BUILD_LOCK = threading.Lock()
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,60}$")
MAX_BODY = 2 * 1024 * 1024
MAX_BODY_V2 = 80 * 1024 * 1024   # v2 imports may carry meshes/masks (base64)


def _send_json(h, code, obj):
    b = json.dumps(obj).encode("utf-8")
    try:
        h.send_response(code)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", str(len(b)))
        h.end_headers()
        h.wfile.write(b)
    except (BrokenPipeError, ConnectionResetError):
        pass  # the browser cancelled a superseded request


def _load_fixture_summary(path):
    try:
        with open(path) as f:
            r = json.load(f)
        return {"concept": r.get("concept"), "style": r.get("architecture", {}).get("style"),
                "zones": len(r.get("zones", [])), "schema": r.get("schema_version")}
    except Exception as e:
        return {"error": str(e)}


def editable_request(req):
    """FIX (Claude): upgrade an older design so every slider has a field to edit.

    Without this, sliders did nothing or crashed on the 2.0.0 fixtures (starter, pass):
    - fold depth: they use architecture.lift_mm, not architecture.fold.fold_depth_mm
    - braid angle: they use braid_tilt_target_deg, not braid_angle_deg
    - ACL length/angles: an anatomy "envelope" makes the schema ignore those fields
    The design is unchanged; only field names move.
    """
    r = copy.deepcopy(req)
    ar = r.setdefault("architecture", {})
    if "fold" not in ar and "lift_mm" in ar:
        ar["fold"] = {"crossing_pattern": "1/1", "fold_depth_mm": ar.pop("lift_mm"), "fold_sharpness": 0.0}
    for z in r.get("zones", []):
        if "braid_tilt_target_deg" in z:
            z.setdefault("braid_angle_deg", z.pop("braid_tilt_target_deg"))
            z.setdefault("control", "braid_angle")
    an = r.get("anatomy", {})
    if isinstance(an, dict) and "envelope" in an:
        env = an["envelope"]
        zones = r.get("zones", [])
        mid = float(zones[len(zones) // 2]["length_mm"]) if zones else 32.0
        r["anatomy"] = {"side": an.get("side", "right"),
                        "acl_length_mm": {"value": mid, "origin": "prototype_placeholder"},
                        "acl_width_mm": {"value": env["width_mm"], "origin": an.get("origin", "prototype_placeholder")},
                        "acl_thickness_mm": {"value": env["thickness_mm"], "origin": an.get("origin", "prototype_placeholder")}}
    an = r.setdefault("anatomy", {})
    an.setdefault("side", "right")
    # Same values the schema assumes when these are missing (Reid et al., L03).
    an.setdefault("sagittal_inclination_deg", {"value": 47.4, "origin": "literature_example", "source": "L03"})
    an.setdefault("coronal_inclination_deg", {"value": 75.3, "origin": "literature_example", "source": "L03"})
    # Milestone A: every slider needs a field to edit (schematic + intact
    # defaults leave the validated design unchanged).
    an.setdefault("realism", "schematic")
    an.setdefault("anatomical_twist_total_deg",
                  {"value": 0.0, "origin": "prototype_placeholder", "source": "L13"})
    tr = r.setdefault("tear", {})
    tr.setdefault("severity", "intact")
    tr.setdefault("location_s", 0.5)
    tr.setdefault("obliquity_deg", 0.0)
    tr.setdefault("azimuth_deg", 0.0)
    tr.setdefault("retraction_femoral_mm", 0.0)
    tr.setdefault("retraction_tibial_mm", 0.0)
    # Loft anatomy sliders: pose + strand density (defaults = extension pose,
    # derived strand count; the validated design is unchanged).
    ks = r.setdefault("knee_scene", {})
    ks.setdefault("type", "schematic")
    ks.setdefault("flexion_deg", 0.0)
    am = r.setdefault("acl_model", {})
    am.setdefault("strand_count", 24)
    return r


class H(BaseHTTPRequestHandler):
    server_version = "N3/1.0"

    def log_message(self, *a):
        pass

    def _read_json(self):
        n = int(self.headers.get("Content-Length", "0") or "0")
        if n > (MAX_BODY_V2 if self.path.startswith("/api/v2/") else MAX_BODY):
            return None, "too_large"
        raw = self.rfile.read(n) if n > 0 else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception as e:
            return None, "bad json: %s" % e
        if not isinstance(body, dict):
            return None, "body must be a JSON object"
        return body, None

    def do_GET(self):
        u = urlparse(self.path)
        # v2 designer at "/", separate viewer at "/view", previous N3 page at "/classic"
        if u.path in ("/", "/view", "/index.html"):
            return self._serve_file(os.path.join(WEB, "v2", "index.html"), "text/html")
        if u.path == "/upload":
            return self._serve_file(os.path.join(WEB, "v2", "upload.html"), "text/html")
        if u.path.startswith("/api/mri/"):
            return self._mri_get(u)
        if u.path == "/classic":
            return self._serve_file(os.path.join(WEB, "index.html"), "text/html")
        if u.path.startswith("/v2/"):
            rel = u.path[len("/v2/"):]
            if "/" in rel or "\\" in rel or rel.startswith(".") or not re.match(r"^[A-Za-z0-9_.-]+$", rel):
                return _send_json(self, 400, {"error": "bad path", "field": "path", "hint": "use /v2/<file>"})
            _, ext = os.path.splitext(rel)
            if ext not in (".html", ".js", ".css", ".json"):
                return _send_json(self, 403, {"error": "blocked", "field": "path", "hint": "whitelist html js css json"})
            ctype = {"html": "text/html", "js": "text/javascript", "css": "text/css", "json": "application/json"}[ext[1:]]
            return self._serve_file(os.path.join(WEB, "v2", rel), ctype)
        if u.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        if u.path == "/api/v2/registry":
            from acl_engine.v2.design import registry_for_ui
            return _send_json(self, 200, registry_for_ui())
        if u.path == "/api/v2/cases":
            import server_v2 as V2
            return _send_json(self, 200, {"cases": V2.builtin_cases()})
        if u.path == "/api/v2/state":
            import server_v2 as V2
            sid = (parse_qs(u.query).get("id", [""])[0])
            st = V2.load_state(sid)
            if st is None:
                return _send_json(self, 404, {"error": "saved model not found", "field": "id",
                                              "hint": "the link may be from another computer or the store was cleared; open a case instead"})
            return _send_json(self, 200, {"id": sid, "state": st})
        if u.path.startswith("/web/"):
            rel = u.path[len("/web/"):]
            if "/" in rel or "\\" in rel or rel.startswith("."):
                return _send_json(self, 400, {"error": "bad path", "field": "path", "hint": "use /web/<file>"})
            _, ext = os.path.splitext(rel)
            if ext not in (".html", ".js", ".css", ".json"):
                return _send_json(self, 403, {"error": "blocked", "field": "path", "hint": "whitelist html js css json"})
            ctype = {"html": "text/html", "js": "text/javascript", "css": "text/css", "json": "application/json"}[ext[1:]]
            return self._serve_file(os.path.join(WEB, rel), ctype)
        if u.path == "/api/presets":
            out = []
            for fn in sorted(os.listdir(FIX)):
                if fn.endswith(".json"):
                    out.append({"name": fn[:-5], "file": fn, "summary": _load_fixture_summary(os.path.join(FIX, fn))})
            return _send_json(self, 200, {"presets": out})
        if u.path == "/api/design":
            q = parse_qs(u.query)
            name = (q.get("name", [""])[0] or "")
            if not NAME_RE.match(name):
                return _send_json(self, 400, {"error": "bad name", "field": "name", "hint": "use [A-Za-z0-9_-]{1,60}"})
            p = os.path.join(FIX, name + ".json")
            if not os.path.isfile(p):
                return _send_json(self, 404, {"error": "not found", "field": "name", "hint": "see /api/presets"})
            with open(p) as f:
                return _send_json(self, 200, {"name": name, "request": editable_request(json.load(f))})
        if u.path == "/api/sliders":
            with open(os.path.join(WEB, "sliders.json")) as f:
                return _send_json(self, 200, json.load(f))
        return _send_json(self, 404, {"error": "not found", "field": "path", "hint": "GET / /web/* /api/*"})

    def _serve_file(self, path, ctype):
        if not os.path.isfile(path):
            return _send_json(self, 404, {"error": "not found", "field": "path", "hint": os.path.basename(path)})
        with open(path, "rb") as f:
            b = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    # ---- MRI upload bridge (tools/mri_bridge.py): scan -> KneeScan reader -> readings -> design state
    def _mri_get(self, u):
        import mri_bridge as MB
        q = parse_qs(u.query)
        if u.path == "/api/mri/reader":
            url = q.get("url", [""])[0] or MB.DEFAULT_READER_URL
            return _send_json(self, 200, dict(MB.reader_info(url), default_url=MB.DEFAULT_READER_URL))
        if u.path == "/api/mri/job":
            job = MB.job_status(q.get("id", [""])[0])
            if job is None:
                return _send_json(self, 404, {"error": "upload job not found", "hint": "the server may have restarted; upload again"})
            return _send_json(self, 200, job)
        return _send_json(self, 404, {"error": "not found", "hint": "GET /api/mri/reader /api/mri/job"})

    def _mri_upload(self, u):
        """POST /api/mri/upload?name=<file>&side=right|left&bones=1|0&reader=<url>, body = the scan bytes."""
        import mri_bridge as MB
        q = parse_qs(u.query)
        side = q.get("side", [""])[0]
        if side not in ("right", "left"):
            self.close_connection = True          # the scan body was not read
            return _send_json(self, 400, {"error": "choose the knee side (right or left)", "field": "side"})
        try:
            reader = MB.normalize_url(q.get("reader", [""])[0] or MB.DEFAULT_READER_URL)
            n = int(self.headers.get("Content-Length", "0") or "0")
            path = MB.receive_scan(self.rfile, n, q.get("name", [""])[0])
        except (MB.ReaderError, ValueError) as e:
            self.close_connection = True
            return _send_json(self, 400, {"error": str(e)})
        jid = MB.start_job(path, os.path.basename(path), side, q.get("bones", ["1"])[0] != "0", reader)
        return _send_json(self, 200, {"job": jid})

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/api/mri/upload":
            return self._mri_upload(u)
        body, err = self._read_json()
        if err == "too_large":
            return _send_json(self, 413, {"error": "body over 2MB", "field": "body", "hint": "send a smaller request"})
        if err:
            return _send_json(self, 400, {"error": err, "field": "body", "hint": "send JSON"})
        if u.path.startswith("/api/v2/"):
            return self._v2(u.path, body)
        if u.path == "/api/build":
            return self._build(body)
        if u.path == "/api/save":
            return self._save(body)
        if u.path == "/api/export_stl":
            return self._export_stl(body)
        if u.path == "/api/export_anatomy":
            return self._export_anatomy(body)
        return _send_json(self, 404, {"error": "not found", "field": "path", "hint": "POST /api/build /api/save /api/export_stl /api/export_anatomy"})

    def _v2(self, path, body):
        import server_v2 as V2
        try:
            if path == "/api/v2/new":
                try:
                    st = V2.new_case(body.get("case", "demo_complete"))
                except KeyError:
                    return _send_json(self, 404, {"error": "unknown case", "hint": "see /api/v2/cases"})
                return _send_json(self, 200, {"state": st})
            if path == "/api/v2/import":
                code, out = V2.import_request(body)
                return _send_json(self, code, out)
            if path == "/api/v2/build":
                code, out = V2.build_request(body)
                return _send_json(self, code, out)
            if path == "/api/v2/detail":
                code, out = V2.detail_request(body)
                return _send_json(self, code, out)
            if path == "/api/v2/section":
                code, out = V2.section_request(body)
                return _send_json(self, code, out)
            if path == "/api/v2/level_detail":
                code, out = V2.level_detail_request(body)
                return _send_json(self, code, out)
            if path == "/api/v2/layer_fixes":
                code, out = V2.layer_fixes_request(body)
                return _send_json(self, code, out)
            if path == "/api/v2/inactive":
                code, out = V2.inactive_request(body)
                return _send_json(self, code, out)
            if path == "/api/v2/resolutions":
                code, out = V2.resolutions_request(body)
                return _send_json(self, code, out)
            if path == "/api/v2/state":
                st = body.get("state")
                if not isinstance(st, dict) or "params" not in st:
                    return _send_json(self, 400, {"error": "missing state"})
                return _send_json(self, 200, {"id": V2.save_state(st)})
            if path == "/api/v2/export":
                code, man, data = V2.export_request(body)
                if code != 200:
                    return _send_json(self, code, man)
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Content-Disposition", "attachment; filename=acl_scaffold_%s.zip" % man["design_hash"])
                self.send_header("X-Design-Hash", man["design_hash"])
                self.send_header("X-Geometry-Hash", man["geometry_hash"])
                self.end_headers()
                self.wfile.write(data)
                return
        except Exception as e:
            traceback.print_exc()
            return _send_json(self, 500, {"error": "server error: %s" % e, "hint": "see server log"})
        return _send_json(self, 404, {"error": "not found"})

    def _validate(self, body):
        from acl_engine.schema import validate_request
        req = body.get("request", None)
        if not isinstance(req, dict):
            _send_json(self, 400, {"error": "missing request", "field": "request", "hint": "POST {request, quality, build_id}"})
            return None
        try:
            return validate_request(req)
        except (ValueError, KeyError, TypeError) as e:
            # FIX (Claude): a missing key raised KeyError, which escaped as a dropped connection.
            _send_json(self, 400, {"error": "%s: %s" % (type(e).__name__, e), "field": "request", "hint": "see validation message"})
            return None

    def _build(self, body):
        from cli import run_request
        quality = body.get("quality", "preview")
        build_id = body.get("build_id", 0)
        if quality not in ("preview", "export"):
            return _send_json(self, 400, {"error": "bad quality", "field": "quality", "hint": "preview|export"})
        v = self._validate(body)
        if v is None:
            return
        norm, mig = v
        t0 = time.time()
        # FIX (Claude): Muse used mkdtemp and never deleted it: one folder of files per slider move.
        with BUILD_LOCK, tempfile.TemporaryDirectory(prefix="n3build_") as tmp:
            try:
                res = run_request(norm, mig, quality, tmp, skip_stl=True)
            except ValueError as e:
                return _send_json(self, 400, {"error": str(e), "field": "request", "hint": "pipeline rejected it"})
            except Exception as e:
                traceback.print_exc()
                return _send_json(self, 500, {"error": "build failed: %s" % e, "field": "request", "hint": "see server log"})
            with open(os.path.join(tmp, "knee.glb"), "rb") as f:
                glb = f.read()
            with open(os.path.join(tmp, "validation.json")) as f:
                val = json.load(f)
            with open(os.path.join(tmp, "context.json")) as f:
                ctx = json.load(f)
        total = time.time() - t0
        summary = {"phaseL": res.get("phaseL"), "nevents": res.get("nevents"),
                   "gap_mm": res.get("gap"), "penetration_mm": res.get("pen"),
                   "kmax": res.get("kmax"), "uses_assumptions": res.get("uses_assumptions"),
                   "assumptions": [a.get("field") for a in norm.get("_assumptions", [])]}
        return _send_json(self, 200, {"build_id": build_id, "design_hash": res.get("design_hash", ""),
                                      "glb_base64": base64.b64encode(glb).decode("ascii"),
                                      "checks": val.get("checks", []), "summary": summary,
                                      "context": {"penetration": ctx.get("containment"),
                                                  "sag": ctx.get("sagittal_measured_deg"),
                                                  "cor": ctx.get("coronal_measured_deg")},
                                      "timings_s": dict(res.get("timings_s", {}), total=total)})

    def _save(self, body):
        name = body.get("name", "")
        req = body.get("request", None)
        ow = bool(body.get("overwrite", False))
        if not isinstance(name, str) or not NAME_RE.match(name):
            return _send_json(self, 400, {"error": "bad name", "field": "name", "hint": "use letters, digits, - and _ only"})
        if not isinstance(req, dict):
            return _send_json(self, 400, {"error": "missing request", "field": "request", "hint": "send {name, request}"})
        p = os.path.join(FIX, name + ".json")
        if os.path.exists(p) and not ow:
            return _send_json(self, 409, {"error": "a design with that name exists", "field": "name", "hint": "pick another name"})
        with open(p, "w") as f:
            json.dump(req, f, indent=2, sort_keys=True)
        return _send_json(self, 200, {"saved": name + ".json"})

    def _export_stl(self, body):
        from cli import run_request
        v = self._validate(body)
        if v is None:
            return
        norm, mig = v
        with BUILD_LOCK, tempfile.TemporaryDirectory(prefix="n3stl_") as tmp:
            try:
                run_request(norm, mig, "export", tmp)
            except Exception as e:
                # FIX (Claude): errors here were not caught, so the browser got a dropped connection.
                traceback.print_exc()
                return _send_json(self, 500, {"error": "export failed: %s" % e, "field": "request", "hint": "see server log"})
            with open(os.path.join(tmp, "scaffold.stl"), "rb") as f:
                b = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "model/stl")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Content-Disposition", "attachment; filename=scaffold.stl")
        self.end_headers()
        self.wfile.write(b)

    def _export_anatomy(self, body):
        from cli import run_request
        v = self._validate(body)
        if v is None:
            return
        norm, mig = v
        with BUILD_LOCK, tempfile.TemporaryDirectory(prefix="n3anat_") as tmp:
            try:
                run_request(norm, mig, "export", tmp)
            except Exception as e:
                traceback.print_exc()
                return _send_json(self, 500, {"error": "export failed: %s" % e, "field": "request", "hint": "see server log"})
            p = os.path.join(tmp, "anatomy.stl")
            if not os.path.exists(p):
                return _send_json(self, 400, {"error": "no anatomy in this design", "field": "anatomy.realism", "hint": "set realism to anatomical"})
            with open(p, "rb") as f:
                b = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "model/stl")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Content-Disposition", "attachment; filename=anatomy.stl")
        self.end_headers()
        self.wfile.write(b)


def run(port=8000, open_browser=True):
    srv = None
    used = port
    for p in range(port, port + 11):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", p), H)
            used = p
            break
        except OSError:
            continue
    if srv is None:
        raise OSError("ports %d-%d busy" % (port, port + 10))
    print("ACL designer: http://127.0.0.1:%d/  (viewer: /view, previous page: /classic). Press Ctrl+C to stop." % used)

    def _warm():
        try:
            from acl_engine.v2.design import new_state
            from acl_engine.v2.scene import build_scene
            with BUILD_LOCK:
                build_scene(new_state(), "full")
        except Exception:
            traceback.print_exc()
    threading.Thread(target=_warm, daemon=True).start()
    if open_browser:
        try:
            webbrowser.open("http://127.0.0.1:%d/" % used)
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 8000)
