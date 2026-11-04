"""Experiment (read-only, changes no engine): how much of the centerline can be read straight from the SVG's exact
straight edges?

Idea: the two sides of a straight stroke are two straight outline segments that run in opposite directions, about one
stroke width apart, with ink between them. Their midline is an exact centerline segment (exact position, exact ends).
This script pairs such segments, then compares them with what the current engine produced:

  covered   share of the engine's path length lying within 0.15 stroke widths of an exact centerline
  exact     length of exact centerlines found; "off" = share of it more than 0.15 stroke widths from every engine path
  corners   places where two exact centerlines meet (extended) close to their ends: candidate sharp corners

Writes results/exact/<name>.svg (grey = original, red = engine, green = exact centerlines, black squares = exact
corners) and results/exact/summary.txt.   Run from prototypes/:  ../.venv/bin/python exact_edges.py [file.svg ...]"""
import sys
from pathlib import Path

import numpy as np
from svgpathtools import svg2paths

import centerline as C
from common import ROOT


def outline_lines(svg_file):
    """All straight outline segments (x0, y0, x1, y1) in the SVG's user units. Curves are ignored."""
    paths, _ = svg2paths(str(svg_file))
    segs = []
    for p in paths:
        for s in p:
            if type(s).__name__ == "Line" and abs(s.end - s.start) > 1e-6:
                segs.append((s.start.real, s.start.imag, s.end.real, s.end.imag))
    return np.array(segs, float).reshape(-1, 4)


def merge_collinear(segs, tol):
    """Join consecutive pieces of one straight edge (e.g. 'h2 1 3') into one segment."""
    segs = [list(s) for s in segs]
    changed = True
    while changed:
        changed = False
        for i in range(len(segs)):
            for j in range(len(segs)):
                if i == j:
                    continue
                a, b = np.array(segs[i]), np.array(segs[j])
                da, db = a[2:] - a[:2], b[2:] - b[:2]
                la, lb = np.linalg.norm(da), np.linalg.norm(db)
                ua, ub = da / la, db / lb
                if np.dot(ua, ub) < 0.9999:
                    continue
                n = np.array([-ua[1], ua[0]])
                if abs(np.dot(b[:2] - a[:2], n)) > tol:
                    continue
                if np.linalg.norm(a[2:] - b[:2]) <= tol:  # b continues a
                    segs[i] = [a[0], a[1], b[2], b[3]]
                    del segs[j]
                    changed = True
                    break
            if changed:
                break
    return np.array(segs, float).reshape(-1, 4)


def pair_edges(segs, w, ink_at, lo=0.6, hi=1.45, ang=3.0):
    """Exact centerline segments (x0, y0, x1, y1, width) from antiparallel edge pairs about one stroke width apart."""
    out = []
    n = len(segs)
    d = segs[:, 2:] - segs[:, :2]
    L = np.linalg.norm(d, axis=1)
    u = d / L[:, None]
    for i in range(n):
        for j in range(i + 1, n):
            if np.dot(u[i], u[j]) > -np.cos(np.radians(ang)):
                continue
            nrm = np.array([-u[i][1], u[i][0]])
            off = np.dot(segs[j, :2] - segs[i, :2], nrm)
            off2 = np.dot(segs[j, 2:] - segs[i, :2], nrm)
            if abs(off - off2) > 0.08 * w:  # not parallel enough
                continue
            gap = abs((off + off2) / 2)
            if not (lo * w <= gap <= hi * w):
                continue
            # overlap along the direction of i
            ti = np.array([0.0, L[i]])
            tj = np.sort([np.dot(segs[j, :2] - segs[i, :2], u[i]), np.dot(segs[j, 2:] - segs[i, :2], u[i])])
            a, b = max(ti[0], tj[0]), min(ti[1], tj[1])
            if b - a < 0.3 * w:
                continue
            side = np.sign((off + off2) / 2)
            mids = [segs[i, :2] + u[i] * t + nrm * side * gap / 2 for t in np.linspace(a, b, 7)[1:-1]]
            if not all(ink_at(m) for m in mids):  # the band between the two edges must be solid ink
                continue
            p0 = segs[i, :2] + u[i] * a + nrm * side * gap / 2
            p1 = segs[i, :2] + u[i] * b + nrm * side * gap / 2
            out.append((p0[0], p0[1], p1[0], p1[1], gap))
    return out


def dist_to_segs(P, segs):
    """Distance from points P (n,2) to the nearest of segs (m,4)."""
    best = np.full(len(P), np.inf)
    for x0, y0, x1, y1 in segs[:, :4]:
        a, b = np.array([x0, y0]), np.array([x1, y1])
        ab = b - a
        t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        best = np.minimum(best, np.linalg.norm(P - (a + t[:, None] * ab), axis=1))
    return best


def corners_of(cl, w):
    out = []
    for i in range(len(cl)):
        for j in range(i + 1, len(cl)):
            a, b = np.array(cl[i][:4]), np.array(cl[j][:4])
            da, db = a[2:] - a[:2], b[2:] - b[:2]
            cr = da[0] * db[1] - da[1] * db[0]
            if abs(cr) < 1e-6 * np.linalg.norm(da) * np.linalg.norm(db):
                continue
            t = ((b[:2] - a[:2])[0] * db[1] - (b[:2] - a[:2])[1] * db[0]) / cr
            s = ((b[:2] - a[:2])[0] * da[1] - (b[:2] - a[:2])[1] * da[0]) / cr
            X = a[:2] + t * da
            ea = min(np.linalg.norm(X - a[:2]), np.linalg.norm(X - a[2:])) if not (0 <= t <= 1) else 0
            eb = min(np.linalg.norm(X - b[:2]), np.linalg.norm(X - b[2:])) if not (0 <= s <= 1) else 0
            if ea < 1.5 * w and eb < 1.5 * w:
                out.append(X)
    return out


def analyse(f, outdir):
    res = C.run(f)
    w = res["w_px"] / res["S"]
    mask, S, vb = res["mask"], res["S"], res["vb"]
    H, W = mask.shape

    def ink_at(p):
        x, y = (p[0] - vb[0]) * S, (p[1] - vb[1]) * S
        return 0 <= int(y) < H and 0 <= int(x) < W and mask[int(y), int(x)]

    segs = outline_lines(f)
    n_raw = len(segs)
    segs = merge_collinear(segs, 0.02 * w) if len(segs) else segs
    cl = pair_edges(segs, w, ink_at) if len(segs) else []
    cl_arr = np.array(cl, float).reshape(-1, 5)
    polys = C.sample_polys(res, 200)
    P = np.vstack(polys) if polys else np.zeros((0, 2))
    covered = float((dist_to_segs(P, cl_arr) < 0.15 * w).mean()) if len(cl_arr) and len(P) else 0.0
    exact_len = float(np.linalg.norm(cl_arr[:, 2:4] - cl_arr[:, :2], axis=1).sum()) if len(cl_arr) else 0.0
    off_len = 0.0
    for x0, y0, x1, y1, _ in cl_arr:
        pts = np.linspace([x0, y0], [x1, y1], 40)
        off_len += np.linalg.norm([x1 - x0, y1 - y0]) * float((_dist_to_polys(pts, polys) > 0.15 * w).mean())
    corners = corners_of(cl, w)

    # picture
    import base64
    x0, y0, vw, vh = vb
    data = base64.b64encode(Path(f).read_bytes()).decode()
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="{x0} {y0} {vw} {vh}" width="700" height="{700*vh/vw:.0f}">',
         f'<rect x="{x0}" y="{y0}" width="{vw}" height="{vh}" fill="#fff"/>',
         f'<image x="{x0}" y="{y0}" width="{vw}" height="{vh}" opacity="0.28" href="data:image/svg+xml;base64,{data}" xlink:href="data:image/svg+xml;base64,{data}"/>']
    for p in res["paths"]:
        o.append(f'<path d="{C.path_d(res, p)}" fill="none" stroke="#e5231b" stroke-width="{w*0.10:.3f}"/>')
    for a, b, c, d, _ in cl:
        o.append(f'<line x1="{a:.3f}" y1="{b:.3f}" x2="{c:.3f}" y2="{d:.3f}" stroke="#1a9850" stroke-width="{w*0.16:.3f}" stroke-linecap="butt" opacity="0.8"/>')
    for X in corners:
        r = w * 0.12
        o.append(f'<rect x="{X[0]-r:.3f}" y="{X[1]-r:.3f}" width="{2*r:.3f}" height="{2*r:.3f}" fill="#000"/>')
    o.append("</svg>")
    (outdir / (Path(f).stem.replace(".svg", "") + ".svg")).write_text("".join(o))
    return dict(name=Path(f).name, lines=n_raw, merged=len(segs), pairs=len(cl), covered=covered, exact_len=exact_len / w,
                off=(off_len / exact_len if exact_len else 0.0), corners=len(corners))


def _dist_to_polys(P, polys):
    best = np.full(len(P), np.inf)
    for poly in polys:
        s = np.c_[poly[:-1], poly[1:]]
        best = np.minimum(best, dist_to_segs(P, s))
    return best


if __name__ == "__main__":
    out = ROOT / "results" / "exact"
    out.mkdir(parents=True, exist_ok=True)
    files = [Path(a) for a in sys.argv[1:] if not a.startswith("-")] or sorted((ROOT / "testset" / "examples").glob("*.svg"))
    rows = []
    for f in files:
        try:
            rows.append(analyse(f, out))
        except Exception as e:  # noqa: BLE001
            print(f.name, "ERROR", type(e).__name__, e)
    hdr = f"{'file':44s} {'lines':>5s} {'edges':>5s} {'pairs':>5s} {'exact len':>9s} {'engine covered':>14s} {'exact off':>9s} {'corners':>7s}"
    txt = [hdr] + [f"{r['name'][:44]:44s} {r['lines']:5d} {r['merged']:5d} {r['pairs']:5d} {r['exact_len']:8.1f}w {r['covered']*100:13.0f}% {r['off']*100:8.0f}% {r['corners']:7d}" for r in rows]
    (out / "summary.txt").write_text("\n".join(txt) + "\n")
    print("\n".join(txt))
