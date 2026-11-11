#!/bin/sh
# Copy the engine files from prototypes/ into docs/engine/ (the browser page loads them from there).
# Run this after any change to the engine, then test the page. Only the files listed here are used.
cd "$(dirname "$0")/.." || exit 1
for f in centerline_exact centerline exact_edges exact_arcs exact_snap exact_ovals exact_shapes common; do
  cp "prototypes/$f.py" "docs/engine/$f.py"
done
echo "engine copied to docs/engine/"
