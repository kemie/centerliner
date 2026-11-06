# Browser edition (docs/)

The same converter as the desktop app, running entirely inside the visitor's browser (Python compiled to WebAssembly via Pyodide; the renderer is resvg compiled to WebAssembly). No server work, so it can be hosted as plain files (GitHub Pages serves this `docs/` folder). Files are never uploaded.

## Files
- `index.html`: the page (same look as `app/index.html`, plus a loading spinner with progress steps, a failure message with "Try again", and an optional tip-jar link: set `TIP_URL` in the script; empty means no link is shown).
- `worker.js`: loads the renderer, Python and libraries, then converts on request; runs off the main thread so the page stays responsive.
- `engine/`: **a copy** of the engine files plus `webconvert.py` (the browser twin of `convert()` in `app/server.py`). Refresh the copy with `./docs/sync-engine.sh` after any change to `prototypes/`. `webconvert.py` and `app/server.py` share the same warning checks, kept in step by hand.

## Try it locally
Must be served over http (opening the file directly will not work):
```
cd /path/to/centerliner && python3 -m http.server 8912
```
then open http://127.0.0.1:8912/docs/index.html

## Needs the internet
Pyodide, its libraries and the renderer are fetched from the jsDelivr CDN (about 50 MB the first time, then cached by the browser); svgpathtools comes from PyPI. To host without depending on a CDN, copy those files into the site (possible, not done).

## Status
Tested in the desktop app's browser pane only: load, convert, corner-radius slider, bad file, file dropped while still loading, load-failure message, narrow (phone) width. Not tested: Safari, Firefox, real phones, slow networks.

## Version
`VERSION` near the top of the script in `index.html` is shown in the header and added to the worker and engine file addresses, so browsers fetch new files instead of old cached ones. Change it whenever the page, worker or engine files change.
