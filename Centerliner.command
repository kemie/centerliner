#!/bin/bash
# Double-click to start Centerliner. A browser tab opens; close this window to stop it.
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$DIR/.venv/bin/python" "$DIR/app/server.py" "$@"
