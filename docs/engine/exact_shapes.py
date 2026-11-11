"""Circle and ellipse arcs found along the traced chains (slice 1 of "shapes first").

A balloon, a paw pad or a ring that other strokes touch is not a closed hole of its own, and its outline is bent by the
junctions where the other strokes meet it. Away from the junctions the traced chain is still exactly on the shape. This module
finds those clean stretches, merges the stretches that belong to the same circle or ellipse (also across different chains),
and returns the shapes in the form exact_ovals.py uses (centre, radii, tilt) so that snapping can put the whole outline,
junction zones included, on them. Occluded shapes stay arcs: nothing here completes a shape.

find_chain_ovals(pieces, w, known) -> list of oval dicts (like exact_ovals.find_ovals, plus arc=True)"""
import numpy as np
from scipy import optimize
from scipy.spatial import cKDTree

from exact_ovals import ellipse_points

CORE_TOL = 0.04    # a stretch is on the shape when every point is this close (stroke widths)
MIN_RUN = 5.0      # a clean stretch is at least this long (stroke widths)
MIN_SPAN = 100.0   # ... and an ellipse needs this many degrees of arc to be determined at all
SIZE = (1.5, 25.0)  # radii limits (stroke widths): smaller is a corner, larger a straight line
MAX_ASPECT = 3.0
MIN_COVER_CIRCLE = 120.0   # degrees of a circle that the clean stretches must cover
MIN_COVER_ELLIPSE = 150.0  # ... of an ellipse (a short arc of a gentle curve fits some ellipse)


def _circle(P):
    A = np.c_[2 * P[:, 0], 2 * P[:, 1], np.ones(len(P))]
    sol = np.linalg.lstsq(A, (P ** 2).sum(axis=1), rcond=None)[0]
    r2 = sol[2] + sol[0] ** 2 + sol[1] ** 2
    if r2 <= 0:
        return None
    return np.array([sol[0], sol[1], np.sqrt(r2), np.sqrt(r2), 0.0])


def _conic_to_params(a):
    A = np.array([[a[0], a[1] / 2], [a[1] / 2, a[2]]])
    if np.linalg.det(A) <= 0:
        return None
    c = np.linalg.solve(2 * A, -np.array([a[3], a[4]]))
    F0 = a[5] + 0.5 * (a[3] * c[0] + a[4] * c[1])
    ev, vec = np.linalg.eigh(A)
    if np.any(-F0 / ev <= 0):
        return None
    r = np.sqrt(-F0 / ev)
    k = int(np.argmax(r))
    th = float(np.arctan2(vec[1, k], vec[0, k]))
    return np.array([c[0], c[1], r[k], r[1 - k], th])


def _ellipse(P):
    """Direct least-squares ellipse (Fitzgibbon) on normalised points; (cx, cy, rx, ry, tilt) or None."""
    m, s = P.mean(axis=0), P.std() + 1e-9
    q = (P - m) / s
    x, y = q[:, 0], q[:, 1]
    D = np.c_[x * x, x * y, y * y, x, y, np.ones(len(x))]
    S = D.T @ D
    C = np.zeros((6, 6))
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1
    try:
        ev, vec = np.linalg.eig(np.linalg.solve(S, C))
    except np.linalg.LinAlgError:
        return None
    ev, vec = np.real(ev), np.real(vec)
    k = [i for i in range(6) if np.isfinite(ev[i]) and ev[i] > 1e-12]
    if not k:
        return None
    p = _conic_to_params(vec[:, k[0]])
    if p is None:
        return None
    return np.array([p[0] * s + m[0], p[1] * s + m[1], p[2] * s, p[3] * s, p[4]])


def _resid(p, P):
    """Distance of every point from the ellipse (accurate: against a dense polyline of it)."""
    return cKDTree(ellipse_points(p, 720)).query(P)[0]


def _span_deg(p, P):
    ang = np.unwrap(np.arctan2(P[:, 1] - p[1], P[:, 0] - p[0]))
    return float(np.degrees(abs(ang[-1] - ang[0])))


def _ok_size(p, w):
    return p is not None and SIZE[0] * w <= p[3] and p[2] <= SIZE[1] * w and p[2] <= MAX_ASPECT * p[3]


def _model(P, w):
    """Best circle or ellipse for the points, or None: (params, worst residual)."""
    if len(P) < 12:
        return None
    best = None
    c = _circle(P)
    if _ok_size(c, w):
        r = float(_resid(c, P).max())
        best = (c, r)
    if best is None or best[1] > CORE_TOL * w:
        # an ellipse needs a long arc to be determined; fit it on the whole stretch, then refine geometrically
        e = _ellipse(P)
        if _ok_size(e, w) and _span_deg(e, P) >= MIN_SPAN:
            try:
                sol = optimize.least_squares(lambda q: _resid(q, P), e, x_scale=[1, 1, 1, 1, 0.1], max_nfev=25)
                e = sol.x
            except Exception:  # noqa: BLE001
                pass
            if _ok_size(e, w):
                r = float(_resid(e, P).max())
                if best is None or r < best[1]:
                    best = (e, r)
    return best


def _grow(P, i, j, w, p, step):
    """Extend the stretch P[i:j] at both ends for as long as the points stay on the shape (refitting as it grows)."""
    n = len(P)
    for _ in range(200):
        moved = False
        for side in (0, 1):
            while True:
                if side == 0 and i - step >= 0:
                    cand = P[i - step:j], i - step, j
                elif side == 1 and j + step <= n:
                    cand = P[i:j + step], i, j + step
                else:
                    break
                q = _model(cand[0], w)
                if q is None or q[1] > CORE_TOL * w:
                    break
                i, j, p, moved = cand[1], cand[2], q[0], True
        if not moved:
            break
    return i, j, p


def _same(a, b, w):
    return (np.hypot(a[0] - b[0], a[1] - b[1]) < 0.5 * w and abs(a[2] - b[2]) < 0.5 * w and abs(a[3] - b[3]) < 0.5 * w
            and (abs(a[2] - a[3]) < 0.1 * w or abs(((a[4] - b[4]) + np.pi / 2) % np.pi - np.pi / 2) < np.radians(6)))


def find_chain_ovals(pieces, w, known=(), circles=()):
    """pieces: [(P (x, y), closed, bend_nodes, wp)]. Returns oval dicts for the circles/ellipses that clean stretches of the
    chains lie on (stretches on an already known oval are skipped). Stretches of different chains on the same shape are
    merged and the shape is refitted on all of them."""
    step = max(2, int(0.5 * w))
    found = []  # (params, [point arrays])
    for P, closed, _bn, _wp in pieces:
        if closed:
            continue  # a closed outline has no junction distortion and already comes out as one smooth curve
        P = np.asarray(P, float)
        n = len(P)
        covered = np.zeros(n, bool)
        for L in (int(12 * w), int(MIN_RUN * w + 3 * w)):  # a long window determines an ellipse, a shorter one a circle
            if n < L + 2:
                continue
            for i in range(0, n - L, max(2, int(w))):
                if covered[i:i + L].any():
                    continue
                q = _model(P[i:i + L], w)
                if q is None or q[1] > CORE_TOL * w:
                    continue
                i2, j2, p = _grow(P, i, i + L, w, q[0], step)
                seg = P[i2:j2]
                if len(seg) < int(MIN_RUN * w) or (abs(p[2] - p[3]) > 0.03 * p[2] and _span_deg(p, seg) < MIN_SPAN):
                    continue
                covered[i2:j2] = True
                found.append((p, [seg]))
    # merge the stretches that are the same shape (across chains too), refit on all their points
    shapes = []
    for p, segs in found:
        for s in shapes:
            if _same(p, s[0], w):
                s[1].extend(segs)
                break
        else:
            shapes.append([p, list(segs)])
    shapes.sort(key=lambda s: -sum(len(x) for x in s[1]))  # best supported first
    out = []
    for p, segs in shapes:
        allp = np.vstack(segs)
        q = _model(allp, w) if len(allp) < 3000 else None
        if q is not None and q[1] <= 1.5 * CORE_TOL * w:
            p = q[0]
        # how much of the shape's circumference the stretches cover: a shape that is only a short arc is a curve, not a shape
        ang = np.degrees(np.arctan2(allp[:, 1] - p[1], allp[:, 0] - p[0]))
        cover = (np.histogram(ang, bins=72, range=(-180, 180))[0] > 0).mean() * 360
        if abs(p[2] - p[3]) < 0.03 * p[2]:
            continue  # circles: the circle code handles them; as chain stretches they gain too little and risk rounding a U-bend that is only part of a rounded rectangle
        if cover < MIN_COVER_ELLIPSE:
            continue
        E = ellipse_points(p)
        tree = cKDTree(E)
        if any((tree.query(k["E"][::40])[0] < 0.12 * w).mean() > 0.9 for k in known):
            continue  # an oval the hole finder already has
        if abs(p[2] - p[3]) < 0.03 * p[2] and any(np.hypot(p[0] - c[0], p[1] - c[1]) < 0.5 * w and abs(p[2] - c[2]) < 0.5 * w for c in circles):
            continue  # a circle the circle code already has (exact from the outline, or a completed ring)
        if any((o["tree"].query(allp)[0] < 0.15 * w).mean() > 0.9 for o in out):
            continue  # the same stretch is already explained by a better supported shape
        out.append(dict(p=p, E=E, tree=tree, resid=float(_resid(p, allp).max()), arc=True))
    return out
