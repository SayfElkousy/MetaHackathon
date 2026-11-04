#!/bin/sh
# One-command, secret-safe hackathon demo startup. Usage: tools/run_demo.sh [port]
set -eu
cd "$(dirname "$0")/.."
DEMO_PORT="${1:-8000}"
DEMO_PY=.venv/bin/python
[ -x "$DEMO_PY" ] || DEMO_PY=python3
"$DEMO_PY" tools/demo_preflight.py --port "$DEMO_PORT" --require-muse
printf '%s\n' "Designer: http://127.0.0.1:$DEMO_PORT/" "Workspace: http://127.0.0.1:$DEMO_PORT/workspace" "MRI upload: http://127.0.0.1:$DEMO_PORT/upload"
NO_BROWSER=1 exec "$DEMO_PY" tools/server.py "$DEMO_PORT"
