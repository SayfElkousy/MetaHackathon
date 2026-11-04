#!/usr/bin/env python3
"""Concise, secret-safe readiness check for the hackathon demo."""
from __future__ import annotations

import argparse
import importlib
import os
import socket
import stat
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
for path in (ROOT, TOOLS):
    if path not in sys.path:
        sys.path.insert(0, path)


class Report:
    def __init__(self): self.rows = []; self.failed = False
    def add(self, level, name, detail):
        self.rows.append((level, name, detail)); self.failed |= level == "FAIL"
    def print(self):
        print("Bundlr demo preflight")
        for level, name, detail in self.rows: print(f"[{level:4}] {name}: {detail}")
        print("Result:", "NOT READY" if self.failed else "READY (warnings are optional services)")


def port_free(port):
    s = socket.socket();
    try:
        s.bind(("127.0.0.1", port)); return True
    except PermissionError:
        return None
    except OSError:
        return False
    finally:
        s.close()


def service_reachable(host, port, timeout=.35):
    try:
        with socket.create_connection((host, port), timeout=timeout): return True
    except OSError:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--offline", action="store_true", help="Do not make the explicit Meta connectivity probe")
    ap.add_argument("--require-muse", action="store_true", help="Treat missing or failed live Muse access as fatal")
    args = ap.parse_args(argv); r = Report()

    r.add("PASS", "Project root", ROOT)
    for module in ("numpy", "scipy", "PIL", "nibabel", "pydicom"):
        try: importlib.import_module(module); r.add("PASS", "Python package", module)
        except Exception: r.add("FAIL", "Python package", f"missing {module}")

    assets = ["tools/web/v2/index.html", "tools/web/ws/index.html", "tools/web/ws/muse.js",
              "fixtures/imaging/sagittal_labels.nii.gz", "fixtures/imaging/dicom_sagittal/IM0001.dcm"]
    missing = [x for x in assets if not os.path.isfile(os.path.join(ROOT, x))]
    r.add("FAIL" if missing else "PASS", "Workspace and synthetic fixtures", ", ".join(missing) if missing else "present")

    try:
        from tools import server_v2
        from acl_engine.v2.scene import build_scene
        scene = build_scene(server_v2.new_case("demo_complete"), "preview")
        if scene.get("ok"): r.add("PASS", "Synthetic geometry", f"built; valid={scene.get('validity', {}).get('valid')}")
        else: r.add("FAIL", "Synthetic geometry", "build rejected")
    except Exception as e:
        r.add("FAIL", "Synthetic geometry", f"cannot build ({type(e).__name__})")

    env_path = os.path.join(ROOT, ".env.muse")
    if os.path.isfile(env_path):
        mode = stat.S_IMODE(os.stat(env_path).st_mode)
        r.add("PASS" if mode & 0o077 == 0 else "WARN", "Muse credential file", "present with private permissions" if mode & 0o077 == 0 else "present; run chmod 600 .env.muse")
    else: r.add("FAIL" if args.require_muse else "WARN", "Muse credential file", "missing; local fallback remains available")

    import muse_api
    if args.offline:
        state = muse_api.status()["state"]
        r.add("WARN", "Muse connectivity", f"not probed (--offline); {state}")
    else:
        code, out = muse_api.probe()
        ok = code == 200
        r.add("PASS" if ok else ("FAIL" if args.require_muse else "WARN"), "Muse connectivity",
              f"{out.get('state', out.get('category', 'unknown'))}; model={out.get('model', muse_api.status()['model'])}" if ok else out.get("error", "probe failed"))

    free = port_free(args.port)
    r.add("PASS" if free is True else "FAIL" if free is False else "WARN", "Main server port",
          f"127.0.0.1:{args.port} is available" if free is True else f"127.0.0.1:{args.port} is occupied" if free is False else "socket binding is restricted in this environment; check at startup")
    knee = service_reachable("127.0.0.1", 7860)
    r.add("PASS" if knee else "WARN", "Optional KneeScan", "reachable" if knee else "unavailable; synthetic demo is unaffected")
    r.add("PASS", "Demo URLs", f"designer http://127.0.0.1:{args.port}/ · workspace /workspace · upload /upload")
    r.print(); return 1 if r.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
