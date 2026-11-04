"""Shared helpers: synthetic test cases (with known true centerlines), SVG output, metrics, overlays."""
import json
import math
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Point, Polygon, MultiPolygon
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent.parent
W, H = 300, 200          # canvas, SVG units
STROKE = 12              # nominal stroke width


def cubic(p0, p1, p2, p3, n=60):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = map(np.array, (p0, p1, p2, p3))
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)


def ellipse(cx, cy, rx, ry, a0=0, a1=360, n=120):
    a = np.radians(np.linspace(a0, a1, n))
    return np.c_[cx + rx * np.cos(a), cy + ry * np.sin(a)]


def build_cases():
    """name -> dict(lines=[Nx2 arrays] true centerlines, cap, join, width or None, note)."""
    c = {}
    c["01_line"] = dict(lines=[[(30, 100), (270, 100)]], note="straight line, round caps")
    c["02_curve"] = dict(lines=[cubic((30, 150), (90, 20), (210, 20), (270, 150))], note="smooth arc")
    c["03_corner_round"] = dict(lines=[[(40, 160), (40, 40), (260, 40)]], note="90 degree corner, round join")
    c["04_corner_miter"] = dict(lines=[[(40, 160), (40, 40), (260, 40)]], join="mitre", note="90 degree corner, sharp outer corner")
    c["05_t_junction"] = dict(lines=[[(30, 60), (270, 60)], [(150, 60), (150, 170)]], note="T junction")
    c["06_x_cross"] = dict(lines=[[(60, 30), (240, 170)], [(60, 170), (240, 30)]], note="X crossing")
    c["07_loop_o"] = dict(lines=[ellipse(150, 100, 70, 55, 0, 360)], closed=True, note="closed loop (like an o)")
    # e-like: bar plus open arc that meets the bar at both ends
    arc = ellipse(150, 100, 70, 60, 180, 180 + 320)
    c["08_letter_e"] = dict(lines=[[(80, 100), (220, 100)], arc], note="e-like: bar + open arc, two T junctions")
    c["09_flat_caps"] = dict(lines=[cubic((30, 150), (90, 20), (210, 20), (270, 150))], cap="flat", note="curve with flat caps")
    s1 = cubic((220, 40), (120, 10), (80, 70), (150, 100))
    s2 = cubic((150, 100), (220, 130), (180, 190), (70, 160))
    c["10_s_curve"] = dict(lines=[np.vstack([s1, s2[1:]])], note="S curve (lettering-like)")
    c["11_nonuniform"] = dict(lines=[cubic((30, 150), (90, 20), (210, 20), (270, 150), 200)], varwidth=(9, 15),
                              note="width varies 9 to 15 along the curve (stress test)")
    c["12_lo_word"] = dict(lines=[[(50, 30), (50, 160)], ellipse(130, 120, 45, 40, 0, 360), cubic((205, 120), (205, 80), (260, 80), (260, 120))],
                           closed_idx=[1], note="'l' + 'o' + arch: several shapes in one drawing")
    return c


def case_geometry(spec):
    cap = {"round": 1, "flat": 2}[spec.get("cap", "round")]
    join = {"round": 1, "mitre": 2}[spec.get("join", "round")]
    if "varwidth" in spec:
        pts = np.array(spec["lines"][0])
        r = np.linspace(spec["varwidth"][0], spec["varwidth"][1], len(pts)) / 2
        return unary_union([Point(p).buffer(ri, 24) for p, ri in zip(pts, r)])
    parts = []
    for ln in spec["lines"]:
        ls = LineString(np.array(ln))
        parts.append(ls.buffer(STROKE / 2, 24, cap_style=cap, join_style=join, mitre_limit=10))
    return unary_union(parts)


def poly_to_path(g):
    polys = list(g.geoms) if isinstance(g, MultiPolygon) else [g]
    d = []
    for p in polys:
        for ring in [p.exterior, *p.interiors]:
            pts = list(ring.coords)[:-1]
            d.append("M" + " L".join(f"{x:.2f} {y:.2f}" for x, y in pts) + " Z")
    return " ".join(d)


def outline_svg(g):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
            f'<rect width="{W}" height="{H}" fill="#fff"/><path d="{poly_to_path(g)}" fill="#000" fill-rule="evenodd"/></svg>')


def truth_lines(spec):
    return [np.array(l, float) for l in spec["lines"]]


def densify(pl, step=0.5):
    pl = np.asarray(pl, float)
    out = [pl[0]]
    for a, b in zip(pl[:-1], pl[1:]):
        n = max(1, int(np.hypot(*(b - a)) / step))
        for i in range(1, n + 1):
            out.append(a + (b - a) * i / n)
    return np.array(out)


def metrics(result_polys, nodes, spec):
    """result_polys: list of Nx2 arrays in SVG units. Returns accuracy numbers (in stroke widths)."""
    truth = unary_union([LineString(t) for t in truth_lines(spec)])
    res_pts = np.vstack([densify(p) for p in result_polys if len(p) > 1]) if result_polys else np.zeros((0, 2))
    if len(res_pts) == 0:
        return dict(mean_err=None, max_err=None, missing=None, paths=0, nodes=0)
    res_union = unary_union([LineString(p) for p in result_polys if len(p) > 1])
    d_res = np.array([truth.distance(Point(p)) for p in res_pts])
    tp = np.vstack([densify(t) for t in truth_lines(spec)])
    d_truth = np.array([res_union.distance(Point(p)) for p in tp])
    return dict(mean_err=float(d_res.mean() / STROKE), max_err=float(d_res.max() / STROKE),
                missing=float(d_truth.max() / STROKE), paths=len(result_polys), nodes=int(nodes))


def overlay_svg(g, result_polys, truth=None, label=""):
    paths = ""
    for p in result_polys:
        if len(p) > 1:
            paths += '<polyline fill="none" stroke="#e5231b" stroke-width="1.4" points="%s"/>' % " ".join(f"{x:.2f},{y:.2f}" for x, y in p)
    dots = ""
    for p in result_polys:
        if len(p) > 1:
            for x, y in (p[0], p[-1]):
                dots += f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.2" fill="#0057ff"/>'
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
            f'<rect width="{W}" height="{H}" fill="#fff"/><path d="{poly_to_path(g)}" fill="#c9c9c9" fill-rule="evenodd"/>'
            f'{paths}{dots}</svg>')
