"""T1 gate. Run:  python check.py

1. Runs every test.
2. Builds every example design in fixtures/.
3. Draws a picture of each knee model (outputs/<name>/knee.png).
4. Prints a short summary you can paste straight back to Muse.
"""
import json
import os
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

FIXTURES = ["fixtures/starter_request.json",
            "fixtures/pass_request.json",
            "fixtures/starter_fiberlength.json",
            "fixtures/acl_context_request.json"]


def main():
    t0 = time.time()
    lines = []
    ok = True
    print("Running tests...")
    env = dict(os.environ, PYTHONPATH=HERE)
    t = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "--no-header", "-p", "no:cacheprovider"],
                       capture_output=True, text=True, env=env)
    summary = [l for l in t.stdout.splitlines()
               if "passed" in l or "failed" in l or l.startswith("FAILED") or l.startswith("ERROR")]
    lines.append("TESTS: " + (" | ".join(summary) if summary else (t.stdout[-500:] + t.stderr[-500:])))
    ok = ok and t.returncode == 0

    from acl_engine import GEOM_KERNEL, SCHEMA_VERSION
    from acl_engine.schema import validate_request
    from cli import run_request
    from tools.render import render_knee
    lines.append("kernel %s schema %s" % (GEOM_KERNEL, SCHEMA_VERSION))
    for fx in FIXTURES:
        name = os.path.splitext(os.path.basename(fx))[0].replace("_request", "")
        out = os.path.join("outputs", name)
        print("Building %s..." % name)
        if not os.path.exists(fx):
            lines.append("%s: MISSING FIXTURE %s" % (name, fx))
            ok = False
            continue
        t1 = time.time()
        try:
            with open(fx) as f:
                req = json.load(f)
            norm, mig = validate_request(req)
            res = run_request(norm, mig, "export", out)
            png = render_knee(out)
            lines.append("%s (from schema %s):\n  %s" % (name, mig["from"], "\n  ".join(res["summary"])))
            lines.append("  checks: " + ", ".join("%s=%s" % (c["check_id"], c["status"]) for c in res["checks"]))
            lines.append("  picture: %s  time %.1fs" % (png, time.time() - t1))
        except Exception as e:  # noqa
            lines.append("%s: CRASHED %s: %s\n%s" % (name, type(e).__name__, e, traceback.format_exc()[-1200:]))
            ok = False
    # N3: preview speed (sliders need < 1.5 s) and the slowest steps, plus server import.
    try:
        import cProfile, io, pstats
        with open("fixtures/acl_context_request.json") as f:
            norm, mig = validate_request(json.load(f))
        pr = cProfile.Profile()
        t1 = time.time()
        pr.enable()
        res = run_request(norm, mig, "preview", os.path.join("outputs", "profile_preview"), skip_stl=True)
        pr.disable()
        dt = time.time() - t1
        lines.append("N3 preview rebuild: %.2fs (target < 1.5s) %s" % (dt, "OK" if dt < 1.5 else "TOO SLOW"))
        lines.append("  stages: " + ", ".join("%s %.2f" % (k, v) for k, v in res["timings_s"].items() if v >= 0.01))
        buf = io.StringIO()
        pstats.Stats(pr, stream=buf).sort_stats("tottime").print_stats(3)
        top = [l.split() for l in buf.getvalue().splitlines()
               if l.strip()[:1].isdigit() and "function calls" not in l][:3]
        lines.append("  slowest functions: " + " | ".join(
            "%s %ss (%s calls)" % (os.path.basename(t[-1]), t[1], t[0]) for t in top))
        import tools.server  # noqa: F401
        lines.append("  server import ok")
    except Exception as e:  # noqa
        lines.append("N3 check FAILED %s: %s" % (type(e).__name__, e))
        ok = False
    lines.append("TOTAL %.1fs  overall=%s" % (time.time() - t0, "PASS" if ok else "FAIL"))
    print("\n===== PASTE THIS BACK TO MUSE =====")
    print("\n".join(lines))
    print("===================================")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
