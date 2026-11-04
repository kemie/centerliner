"""Centerliner: a tiny local web server. Serves app/index.html and converts SVGs posted to /convert.
Run with ./Centerliner.command (or: .venv/bin/python app/server.py). Everything stays on this computer."""
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "prototypes"))
import importlib  # noqa: E402

# the engine: the exact-line version by default; CENTERLINE_ENGINE=centerline switches back to the original
C = importlib.import_module(os.environ.get("CENTERLINE_ENGINE", "centerline_exact"))

def code_version():
    """Short git id of the code on disk, with '+edits' when there are uncommitted changes ('unknown' without git)."""
    try:
        g = lambda *a: subprocess.run(["git", "-C", str(HERE.parent), *a], capture_output=True, text=True, timeout=5).stdout.strip()  # noqa: E731
        h = g("rev-parse", "--short", "HEAD")
        return (h + ("+edits" if g("status", "--porcelain", "--", "app", "prototypes") else "")) if h else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


# which code this copy of the app loaded: the engine is read once at start, so a copy started before a change keeps the old code
LAUNCHED = code_version()
STARTED = time.strftime("%H:%M")

PORT = 8765
MAX_BYTES = 20 * 1024 * 1024


def warnings_for(svg_text, ink):
    out = []
    colours = set(re.findall(r'(?:fill|stroke)\s*[:=]\s*["\']?\s*(#[0-9a-fA-F]{3,8}|rgb\([^)]*\)|[a-zA-Z]+)', svg_text))
    colours = {c.lower() for c in colours} - {"none", "currentcolor", "transparent", "inherit", "evenodd", "nonzero", "round", "butt", "square", "miter", "bevel"}
    if len(colours) > 1:
        out.append("This file seems to use several colours. Only one colour is supported; the result uses the dominant one.")
    if "<text" in svg_text:
        out.append("This file contains live text. Convert text to outlines first, or it will be ignored.")
    if re.fullmatch(r"#[0-9a-fA-F]{6}", ink or ""):
        r, g, b = (int(ink[i:i + 2], 16) for i in (1, 3, 5))
        if 0.299 * r + 0.587 * g + 0.114 * b > 200:
            out.append("The artwork's colour is very light, which can confuse the tool. Try it in a dark colour.")
    return out


def convert(svg_bytes, radius):
    svg_text = svg_bytes.decode("utf-8", errors="replace")
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "in.svg"
        f.write_bytes(svg_bytes)
        try:
            res = C.run(str(f), corner_radius=radius)
        except ValueError:
            return dict(error="This SVG has no viewBox or width/height, so I can't tell its size.")
        except Exception as e:  # noqa: BLE001 - show any engine failure as a message instead of a crashed page
            return dict(error=f"Could not convert this file ({type(e).__name__}). It may be unsupported.")
        sc = C.reconstruction_scores(res)
        return dict(
            result=C.result_svg(res),
            compare=C.overlay_svg(str(f), res, size=900),
            stats=dict(paths=len(res["paths"]), dots=len(res["dots"]), anchors=C.anchors(res),
                       stroke=round(res["w_px"] / res["S"], 2), match=round(sc["iou"] * 100)),
            warnings=warnings_for(svg_text, res.get("ink")) + ([] if res["paths"] or res["dots"] else ["Nothing was found to convert."]),
        )


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if urlparse(self.path).path == "/version":
            cur = code_version()
            body = json.dumps(dict(launched=LAUNCHED, current=cur, started=STARTED, engine=C.__name__, stale=(cur != LAUNCHED))).encode()
            self._send(200, body, "application/json")
        elif urlparse(self.path).path in ("/", "/index.html"):
            self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/convert":
            return self._send(404, b"not found", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0 or n > MAX_BYTES:
            return self._send(413, json.dumps(dict(error="File is empty or too large (limit 20 MB).")).encode(), "application/json")
        try:
            radius = float(parse_qs(u.query).get("radius", [C.CORNER_R])[0])
        except ValueError:
            radius = C.CORNER_R
        out = convert(self.rfile.read(n), min(max(radius, 0.0), 3.0))
        self._send(200, json.dumps(out).encode(), "application/json")


def main():
    port = PORT
    while True:
        try:
            srv = HTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            port += 1
    url = f"http://127.0.0.1:{port}/"
    print(f"Centerliner is running at {url}\nClose this window (or press Ctrl+C) to stop it.")
    if "--no-browser" not in sys.argv:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
