#!/usr/bin/env bash
set -e

# Resolve repository root
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Point HOME to repo root so Streamlit uses project-local .streamlit configs
export HOME="$DIR"
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Ensure virtual environment streamlit is used
if [ -x "$DIR/.venv/bin/streamlit" ]; then
    exec "$DIR/.venv/bin/streamlit" run "$DIR/src/ui/app.py" "$@"
else
    echo "Error: Virtual environment not found at $DIR/.venv" >&2
    exit 1
fi
