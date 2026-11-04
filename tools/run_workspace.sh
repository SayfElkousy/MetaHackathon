#!/bin/sh
# Start the designer + imaging workspace without opening a browser tab. Usage: tools/run_workspace.sh [port]
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
NO_BROWSER=1 exec "$PY" tools/server.py "${1:-8765}"
