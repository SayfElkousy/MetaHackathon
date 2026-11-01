"""N3 server tests (Muse), with additions by Claude marked FIX (Claude)."""
import base64
import json
import os
import struct
import threading
import time
import urllib.request

FIXTURES = ["starter_request", "pass_request", "starter_fiberlength", "acl_context_request"]


def _start():
    from tools.server import H
    from http.server import ThreadingHTTPServer
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, "http://127.0.0.1:%d" % srv.server_address[1]


def _post(base, path, obj):
    q = urllib.request.Request(base + path, data=json.dumps(obj).encode(),
                               headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(q) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def _post_raw(base, path, obj):
    q = urllib.request.Request(base + path, data=json.dumps(obj).encode(),
                               headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(q) as r:
        return r.status, r.read()


def _get(base, path):
    with urllib.request.urlopen(base + path) as r:
        return json.loads(r.read().decode())


def _slot(obj, path):
    if path.startswith("zones."):
        _, zid, field = path.split(".")
        return [z for z in obj["zones"] if z["id"] == zid][0], field
    ks = path.split(".")
    o = obj
    for k in ks[:-1]:
        o = o[k]
    return o, ks[-1]


def test_n3_server():
    srv, base = _start()
    try:
        with open("fixtures/acl_context_request.json") as f:
            req = json.load(f)
        t0 = time.time()
        st, j = _post(base, "/api/build", {"request": req, "quality": "preview", "build_id": 1})
        dt = time.time() - t0
        print("preview build %.2fs" % dt)
        assert st == 200
        assert base64.b64decode(j["glb_base64"])[:4] == b"glTF"
        assert len(j["checks"]) > 0
        assert dt < 1.5
        st2, j2 = _post(base, "/api/build", {"request": {"bad": 1}, "quality": "preview", "build_id": 2})
        assert st2 == 400 and "field" in j2
        assert _post(base, "/api/save", {"name": "../x", "request": req})[0] == 400
        assert _post(base, "/api/save", {"name": "a/b", "request": req})[0] == 400
        st5, raw = _post_raw(base, "/api/export_stl", {"request": req})
        assert st5 == 200
        n = struct.unpack("<I", raw[80:84])[0]
        assert len(raw) == 84 + 50 * n
        with open("tools/web/hints.json") as f:
            hints = json.load(f)
        for c in j["checks"]:
            cid = c["check_id"]
            assert cid in hints or cid.startswith("fiber_length"), cid  # FIX (Claude): no "fallback" escape hatch
    finally:
        srv.shutdown()


def test_n3_every_preset_has_every_slider():
    # FIX (Claude): sliders silently did nothing on the 2.0.0 fixtures. /api/design now upgrades
    # field names; every slider path must exist and moving it must change the validated design.
    from acl_engine.schema import validate_request
    srv, base = _start()
    try:
        sl = _get(base, "/api/sliders")["sliders"]
        for name in FIXTURES:
            req = _get(base, "/api/design?name=" + name)["request"]
            n0, _ = validate_request(req)
            for s in sl:
                if s["id"] == "side":
                    continue
                o, k = _slot(req, s["path"])
                if s["path"].startswith("zones."):
                    continue  # zone fields are created on first move
                assert k in o, (name, s["path"])
            # moving ACL length must reach the engine
            o, k = _slot(req, "anatomy.acl_length_mm")
            o[k] = {"value": 36.0, "origin": "user_design"}
            n1, _ = validate_request(req)
            assert n1["_derived"]["acl_length_mm"] == 36.0, name
    finally:
        srv.shutdown()


def test_n3_upgrade_keeps_design_identical():
    # The upgrade must only rename fields: same geometry inputs before and after.
    from acl_engine.schema import validate_request
    from tools.server import editable_request
    for name in ("starter_request", "pass_request"):
        with open(os.path.join("fixtures", name + ".json")) as f:
            req = json.load(f)
        a, _ = validate_request(req)
        b, _ = validate_request(editable_request(req))
        for k in ("swing_mm", "fold_depth_mm", "fold_sharpness", "envelope_width_mm", "envelope_thickness_mm", "acl_length_mm"):
            assert abs(a["_derived"][k] - b["_derived"][k]) < 1e-12, (name, k)
        assert [z.get("braid_angle_deg") for z in a["zones"]] == [z.get("braid_angle_deg") for z in b["zones"]]
