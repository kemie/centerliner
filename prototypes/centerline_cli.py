"""Convert a filled-outline SVG to centerline strokes.

    python centerline_cli.py input.svg output.svg [--corner-radius 0.8]

--corner-radius  Corners whose centreline radius is up to this many stroke widths become sharp corners (the stroke's own
                 round/miter join then gives the rounding); larger radii stay curves. Default 0.8. Measured on the sample
                 icons: sharp corners ~0.3, round-join corners ~0.5-0.7, true rounded corners ~1.0 and up.
"""
import argparse
from pathlib import Path

import importlib
import os

C = importlib.import_module(os.environ.get("CENTERLINE_ENGINE", "centerline_exact"))  # the exact-line engine; CENTERLINE_ENGINE=centerline gives the original one

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("input")
ap.add_argument("output")
ap.add_argument("--corner-radius", type=float, default=C.CORNER_R, help="in stroke widths (default %(default)s)")
args = ap.parse_args()

res = C.run(args.input, corner_radius=args.corner_radius)
Path(args.output).write_text(C.result_svg(res))
sc = C.reconstruction_scores(res)
print(f"{len(res['paths'])} paths, {len(res['dots'])} dots, {C.anchors(res)} anchors, stroke width {res['w_px']/res['S']:.2f} units, "
      f"{len(res['fills'])} filled area(s) kept; redrawn match {sc['iou']*100:.0f}%")
