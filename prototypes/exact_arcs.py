"""Experiment (read-only, changes no engine): how much of the centerline's curved parts can be read from the SVG's own
curves as exact circles and ellipses?

Idea (same as exact_edges.py, for curves): a stroke bent around a circle has two outline curves that are concentric arcs one
stroke width apart. Their middle arc is an exact centerline: exact centre, radius and sweep.
The outlines of most icons are cubic Beziers, so each cubic is tested for being a circular (or elliptical) arc by fitting
and measuring the worst deviation; neighbouring pieces on the same circle are joined; concentric partners are paired.

Output per file: how many cubics are circle / ellipse pieces, how many exact centre arcs come out (centre, radii, sweep).
Run from prototypes/:  ../.venv/bin/python exact_arcs.py [file.svg ...]"""
import os
import sys
from pathlib import Path

import numpy as np
from svgpathtools import svg2paths

from common import ROOT

MIN_SWEEP = 45.0  # degrees: shorter outline arcs are not trusted as exact circles (centre and radius come out slightly wrong)
TOL = 0.012  # worst deviation from the fitted circle / ellipse, in stroke widths


def conic_ellipse(P):
    """Least-squares ellipse through points P: (centre, a, b, theta) or None. a >= b."""
    c0 = P.mean(axis=0)
    x, y = (P - c0).T
    D = np.c_[x * x, x * y, y * y, x, y, np.ones_like(x)]
    _, _, Vt = np.linalg.svd(D, full_matrices=False)
    A, B, C, Dd, E, F = Vt[-1]
    if B * B - 4 * A * C >= 0:
        return None
    den = B * B - 4 * A * C
    cx = (2 * C * Dd - B * E) / den
    cy = (2 * A * E - B * Dd) / den
    M = np.array([[A, B / 2], [B / 2, C]])
    Fc = A * cx * cx + B * cx * cy + C * cy * cy + Dd * cx + E * cy + F
    ev, evec = np.linalg.eigh(M)
    if np.any(ev * -Fc <= 0):
        return None
    ax = np.sqrt(-Fc / ev)  # semi-axes along the eigenvectors
    order = np.argsort(-ax)
    a, b = ax[order]
    v = evec[:, order[0]]
    return np.array([cx, cy]) + c0, float(a), float(b), float(np.arctan2(v[1], v[0]))


def ellipse_dev(P, ell):
    c, a, b, th = ell
    R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
    q = (P - c) @ R.T
    # distance to the ellipse by a few Newton-free refinements: use the radial scale as a first-order measure
    r = np.hypot(q[:, 0] / a, q[:, 1] / b)
    return float(np.abs(r - 1).max() * b)


def piece_fit(P, w):
    """('circle'|'ellipse', centre, a, b, theta, dev) for a sampled curve piece, or None."""
    c0 = P.mean(axis=0)
    x, y = (P - c0).T
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    sol = np.linalg.lstsq(A, x * x + y * y, rcond=None)[0]
    cx, cy = sol[0], sol[1]
    r2 = sol[2] + cx * cx + cy * cy
    if r2 > 0:
        r = np.sqrt(r2)
        dev = float(np.abs(np.hypot(x - cx, y - cy) - r).max())
        if dev <= TOL * w and r < 200 * w:
            return ("circle", np.array([cx, cy]) + c0, r, r, 0.0, dev)
    ell = conic_ellipse(P)
    if ell is None:
        return None
    dev = ellipse_dev(P, ell)
    if dev <= TOL * w and ell[1] < 200 * w:
        return ("ellipse", ell[0], ell[1], ell[2], ell[3], dev)
    return None


def arcs_of(svg_file, w):
    """Exact circular/elliptical arcs of the outline: dicts(kind, c, a, b, th, angles), consecutive pieces joined."""
    paths, _ = svg2paths(str(svg_file))
    out = []
    for p in paths:
        run = []  # consecutive curve pieces on one circle / ellipse
        cuts = []
        for s in p:
            fit = None
            if type(s).__name__ == "CubicBezier":
                t = np.linspace(0, 1, 40)
                P = np.array([[s.point(u).real, s.point(u).imag] for u in t])
                if np.linalg.norm(P[-1] - P[0]) > 0.3 * w:
                    fit = piece_fit(P, w)
                    if fit is not None:
                        fit = fit + (P,)
            cuts.append(fit)
        # join neighbouring pieces with the same circle / ellipse
        cur = None
        for fit in cuts + [None]:
            if fit is not None and cur is not None and fit[0] == cur["kind"] and np.linalg.norm(fit[1] - cur["c"]) < 0.03 * w \
                    and abs(fit[2] - cur["a"]) < 0.03 * w and abs(fit[3] - cur["b"]) < 0.03 * w \
                    and (cur["kind"] == "circle" or abs(np.sin(fit[4] - cur["th"])) < 0.02):
                cur["pts"].append(fit[6])
                continue
            if cur is not None:
                out.append(cur)
                cur = None
            if fit is not None:
                cur = dict(kind=fit[0], c=fit[1], a=fit[2], b=fit[3], th=fit[4], pts=[fit[6]])
        # (a closed outline's last piece may continue its first: not joined here, rare and harmless)
    for arc in out:
        P = np.vstack(arc["pts"])
        d = P - arc["c"]
        ang = np.degrees(np.unwrap(np.arctan2(d[:, 1], d[:, 0])))
        arc["sweep"] = float(abs(ang[-1] - ang[0]))
        arc["start"] = float(ang[0])
        arc["end"] = float(ang[-1])
        arc["len"] = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    return out


def pair_arcs(arcs, w, lo=0.6, hi=1.45):
    """Concentric pairs about one stroke width apart -> exact middle arcs (kind, centre, a, b, theta, overlap degrees)."""
    out = []
    for i in range(len(arcs)):
        for j in range(i + 1, len(arcs)):
            A, B = arcs[i], arcs[j]
            if A["kind"] != B["kind"] or np.linalg.norm(A["c"] - B["c"]) > 0.06 * w:
                continue
            if A["kind"] == "ellipse" and abs(np.sin(A["th"] - B["th"])) > 0.02:
                continue
            da, db = abs(A["a"] - B["a"]), abs(A["b"] - B["b"])
            if not (lo * w <= da <= hi * w and lo * w <= db <= hi * w) and A["kind"] == "ellipse":
                continue
            if A["kind"] == "circle" and not (lo * w <= da <= hi * w):
                continue
            # angular overlap (use the shorter sweep as a bound; exact intervals would need the winding: left for the engine step)
            out.append(dict(kind=A["kind"], c=(A["c"] + B["c"]) / 2, a=(A["a"] + B["a"]) / 2, b=(A["b"] + B["b"]) / 2,
                            th=A["th"], sweep=min(A["sweep"], B["sweep"])))
    return out


def analyse(f):
    import centerline_exact as C
    res = C.run(f)
    w = res["w_px"] / res["S"]
    paths, _ = svg2paths(str(f))
    n_cub = sum(1 for p in paths for s in p if type(s).__name__ == "CubicBezier")
    arcs = arcs_of(f, w)
    pairs = pair_arcs(arcs, w)
    return dict(name=Path(f).name, w=w, cubics=n_cub, arcs=arcs, pairs=pairs)


if __name__ == "__main__":
    files = [Path(a) for a in sys.argv[1:] if not a.startswith("-")] or sorted((ROOT / "testset" / "examples").glob("*.svg"))
    for f in files:
        try:
            r = analyse(f)
        except Exception as e:  # noqa: BLE001
            print(f.name, "ERROR", type(e).__name__, e)
            continue
        in_arcs = sum(len(a["pts"]) for a in r["arcs"])
        print(f"{r['name'][:36]:36s} w={r['w']:.2f}  cubics {r['cubics']:3d}  on a circle/ellipse {in_arcs:3d}  arcs {len(r['arcs']):2d}  middle arcs {len(r['pairs']):2d}")
        if "-v" in sys.argv:
            for a in r["arcs"]:
                print(f"     {a['kind']:7s} c=({a['c'][0]:.2f},{a['c'][1]:.2f}) a={a['a']:.2f} b={a['b']:.2f} sweep={a['sweep']:.0f} len={a['len']:.1f}")
            for p in r["pairs"]:
                print(f"   >> middle {p['kind']:7s} c=({p['c'][0]:.2f},{p['c'][1]:.2f}) a={p['a']:.2f} b={p['b']:.2f} sweep<={p['sweep']:.0f}")


# ---------------------------------------------------------------- use in the engine (centerline_exact.run)
def corner_points(arcs, w, ink_at):
    """Exact stroke corners and ends: the outline of a round-joined / round-capped stroke has an arc of radius w/2 around each
    corner or end, and that arc's centre IS the centerline's corner or end. Centres lying on ink only (a ring's hole is not)."""
    pts = []
    for a in arcs:
        if a["kind"] != "circle" or abs(a["a"] - 0.5 * w) > 0.04 * w or not (15 <= a["sweep"] <= 200):
            continue
        c = a["c"]
        if ink_at(c) and not any(np.linalg.norm(c - q) < 0.15 * w for q in pts):
            pts.append(c)
    return np.array(pts).reshape(-1, 2)


def corner_lines(cpts, w, ink_at, lines):
    """Straight sides between two exact corners, for walls whose own outline edges are broken up (a bar crossing them) so that
    edge pairing finds nothing: the segment must run over ink along its whole length and across its width, hold no other
    corner, and not repeat a line that is already known (lines: existing exact lines, same units)."""
    out = []
    n = len(cpts)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = cpts[i], cpts[j]
            L = float(np.linalg.norm(b - a))
            if L < 1.5 * w:
                continue
            u = (b - a) / L
            nrm = np.array([-u[1], u[0]])
            t = np.arange(0, L + 1e-9, 0.25 * w)
            S0 = a + t[:, None] * u
            if not all(ink_at(p) for p in S0):
                continue
            if sum(ink_at(p + s * 0.4 * w * nrm) for p in S0 for s in (-1, 1)) < 0.95 * 2 * len(S0):
                continue
            if any(k not in (i, j) and 0 < np.dot(cpts[k] - a, u) < L and abs(np.dot(cpts[k] - a, nrm)) < 0.3 * w for k in range(n)):
                continue
            if any(max(_pt_line(a, ln, np.inf), _pt_line(b, ln, np.inf)) < 0.15 * w for ln in lines):
                continue
            out.append(np.r_[a, b])
    return np.array(out).reshape(-1, 4)


def _pt_line(p, ln, ext=0.0):
    """Distance of p from the segment ln (its ends extended by ext, so that p beyond the segment's ends counts as far away)."""
    a, b = ln[:2], ln[2:]
    L = np.linalg.norm(b - a)
    u = (b - a) / L
    d = p - a
    t = d @ u
    if t < -ext or t > L + ext:
        return float("inf")
    return float(abs(d[0] * u[1] - d[1] * u[0]))


def exact_circles(arcs, w, ink_at=None):
    """Exact middle circles (centre, radius, span) from concentric outline circles about one stroke width apart.
    span is None for a full circle. For a ring with a gap it is (start angle, end angle, tight) in degrees (image
    coordinates, the arc runs from start to end): tight when both outline arcs end at the same angles (a stroke with flat
    ends squarely cut across, 3 degrees), else loose (the ends are cut by another stroke, 20 degrees). A loose span
    only counts if the middle of the gap is not ink (a ring crossed by a line is not open there)."""
    out = []
    circles = [a for a in arcs if a["kind"] == "circle" and a["sweep"] >= MIN_SWEEP]  # short pieces give unreliable circles
    ang_d = lambda x, y: abs((x - y + 180) % 360 - 180)  # noqa: E731
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            A, B = circles[i], circles[j]
            if np.linalg.norm(A["c"] - B["c"]) > 0.06 * w or not (0.6 * w <= abs(A["a"] - B["a"]) <= 1.45 * w):
                continue
            c, r = (A["c"] + B["c"]) / 2, (A["a"] + B["a"]) / 2
            span = None
            if A["sweep"] < 340 and B["sweep"] < 340:
                la, ha = sorted((A["start"], A["end"]))
                lb, hb = sorted((B["start"], B["end"]))
                if ang_d(la, lb) < 3 and ang_d(ha, hb) < 3:
                    span = (la, ha, True)
                elif ang_d(la, lb) < 20 and ang_d(ha, hb) < 20 and ink_at is not None:
                    mean = lambda x, y: np.degrees(np.arctan2(np.sin(np.radians(x)) + np.sin(np.radians(y)), np.cos(np.radians(x)) + np.cos(np.radians(y))))  # noqa: E731
                    lo, hi = mean(la, lb), mean(ha, hb)
                    sweep = (hi - lo) % 360
                    mid = np.radians(lo + (sweep + 360) / 2)  # middle of the gap
                    if not ink_at(c + r * np.array([np.cos(mid), np.sin(mid)])):
                        span = (lo, lo + sweep, False)
            out.append((c, r, span))
    return out


def circle_lines(circles, w, ink_at, lines):
    """Straight sides between two equal exact circles (the rounded corners of a rounded rectangle, the ends of a stadium): the
    common tangent segment, on whichever side it runs over ink along its whole length and across its width. Same tests as
    corner_lines; skips a line that is already known."""
    out = []
    cs = [(c, r) for c, r, _ in circles if r >= 0.9 * w]
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            (a, ra), (b, rb) = cs[i], cs[j]
            L = float(np.linalg.norm(b - a))
            if abs(ra - rb) > 0.05 * w or L < 1.5 * w:
                continue
            u = (b - a) / L
            nrm = np.array([-u[1], u[0]])
            for side in (-1, 1):
                p0, p1 = a + side * ra * nrm, b + side * ra * nrm
                t = np.arange(0, L + 1e-9, 0.25 * w)
                S0 = p0 + t[:, None] * u
                if not all(ink_at(p) for p in S0):
                    continue
                if sum(ink_at(p + s * 0.4 * w * nrm) for p in S0 for s in (-1, 1)) < 0.95 * 2 * len(S0):
                    continue
                if any(max(_pt_line(p0, ln, 0.15 * w), _pt_line(p1, ln, 0.15 * w)) < 0.15 * w for ln in lines):
                    continue
                out.append(np.r_[p0, p1])
    return np.array(out).reshape(-1, 4)
