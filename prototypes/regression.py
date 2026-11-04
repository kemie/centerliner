"""Safety net for engine changes: converts all 20 example icons and 12 synthetic shapes, prints one line each
(paths / dots / anchors / redrawn match), and compares with the saved baseline in results/baseline.txt.

    cd prototypes
    ../.venv/bin/python regression.py            # compare with the baseline, list what changed
    ../.venv/bin/python regression.py --save     # accept the current output as the new baseline
Expected: no differences, except for icons you meant to change."""
import sys
from pathlib import Path

import importlib
import os

C = importlib.import_module(os.environ.get("CENTERLINE_ENGINE", "centerline"))  # set CENTERLINE_ENGINE=centerline_exact to try the experiment
from common import ROOT

_ENGINE = os.environ.get("CENTERLINE_ENGINE", "centerline")
BASE = ROOT / "results" / ("baseline.txt" if _ENGINE == "centerline" else f"baseline-{_ENGINE}.txt")  # each engine has its own baseline


def lines():
    out = []
    for folder in (ROOT / "testset" / "examples", ROOT / "testset"):
        for f in sorted(folder.glob("*.svg")):
            res = C.run(f)
            sc = C.reconstruction_scores(res)
            out.append(f"{folder.name}/{f.name}: paths {len(res['paths'])}, dots {len(res['dots'])}, anchors {C.anchors(res)}, match {sc['iou']*100:.0f}%")
    return out


if __name__ == "__main__":
    now = lines()
    if "--save" in sys.argv:
        BASE.write_text("\n".join(now) + "\n")
        print(f"saved {len(now)} lines to {BASE}")
    else:
        old = BASE.read_text().splitlines() if BASE.exists() else []
        was = {l.split(": ")[0]: l for l in old}
        diff = [(was.get(l.split(": ")[0], "(new file, not in baseline)"), l) for l in now if was.get(l.split(": ")[0]) != l]
        print(f"{len(now)} files, {len(diff)} changed or new" + ("" if old else " (no baseline yet: run with --save)"))
        for a, b in diff:
            print("  was:", a, "\n  now:", b)
