"""Exact ovals (ellipse rings) read from the ink itself.

A ring drawn as an oval stroke has an elliptical hole whose boundary lies exactly w/2 inside the stroke's centreline
ellipse. Fitting "every hole-boundary point is w/2 from one ellipse" gives the centreline oval (centre, radii, tilt) to a few
hundredths of a stroke width, also for flat ovals whose ends are tighter than w/2 (where the medial axis fails: the hole
becomes a lens with pointed tips). Works for stroked and outlined drawings alike.

find_ovals(mask, w)               -> ovals (dicts) found among the enclosed holes of the ink mask (pixel space)
apply_ovals(pieces, ovals, w)     -> (pieces without the chains that are an oval, exact Bezier paths for them)
snap_ellipses(P, closed, ovals, w)-> stretches of a chain on an oval are put exactly on it"""
import numpy as np
from scipy import ndimage, optimize
from scipy.spatial import cKDTree
from skimage import measure

DEBUG = False
MAX_RESID = 0.1   # worst distance error (stroke widths) of the hole boundary from "w/2 outside the ellipse"
DENSE = 2400      # points on the dense ellipse polyline


def ellipse_points(p, n=DENSE):
    cx, cy, rx, ry, th = p
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x, y = rx * np.cos(t), ry * np.sin(t)
    c, s = np.cos(th), np.sin(th)
    return np.c_[cx + c * x - s * y, cy + s * x + c * y]


def _fit_hole(B, w):
    c0 = B.mean(axis=0)
    vt = np.linalg.svd(B - c0, full_matrices=False)[2]
    th0 = np.arctan2(vt[0][1], vt[0][0])
    R = np.array([[np.cos(th0), np.sin(th0)], [-np.sin(th0), np.cos(th0)]])
    q = (B - c0) @ R.T
    rx0, ry0 = np.abs(q[:, 0]).max() + w / 2, np.abs(q[:, 1]).max() + w / 2

    def res(p):
        return cKDTree(ellipse_points(p, 900)).query(B)[0] - w / 2

    best = None
    for fx, fy in ((1, 1), (0.97, 1.0), (1.0, 1.15)):
        sol = optimize.least_squares(res, [c0[0], c0[1], rx0 * fx, ry0 * fy, th0], x_scale=[1, 1, 1, 1, 0.1])
        if best is None or sol.cost < best.cost:
            best = sol
    return best.x, float(np.abs(res(best.x)).max())


def find_ovals(mask, w):
    """Ovals (not circles) among the enclosed holes of the boolean ink mask; w = stroke width in pixels."""
    lab, n = ndimage.label(~mask)
    border = set(lab[0]) | set(lab[-1]) | set(lab[:, 0]) | set(lab[:, -1])
    dt = ndimage.distance_transform_edt(mask)
    H, W = mask.shape
    out = []
    for k in range(1, n + 1):
        if k in border:
            continue
        reg = lab == k
        if reg.sum() < 30:
            continue
        cs = measure.find_contours(np.pad(reg, 1).astype(float), 0.5)
        c = max(cs, key=len)
        B = np.c_[c[:, 1] - 1 + 0.5, c[:, 0] - 1 + 0.5]
        if len(B) < 24:
            continue
        p, mx = _fit_hole(B, w)
        if mx > MAX_RESID * w or p[2] <= 0 or p[3] <= 0:
            continue
        if abs(p[2] - p[3]) < 0.03 * max(p[2], p[3]):
            continue  # a circle: handled by the circle code
        E = ellipse_points(p, 600)
        ix, iy = E[:, 0].astype(int), E[:, 1].astype(int)
        ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        if ok.mean() < 1.0 or mask[iy[ok], ix[ok]].mean() < 0.97:
            continue  # the oval's centreline is not ink all round
        if np.median(dt[iy[ok], ix[ok]]) > 1.3 * (w / 2):
            continue  # ink much thicker than a stroke: a plate with an oval hole, not a ring
        E = ellipse_points(p)
        out.append(dict(p=p, E=E, tree=cKDTree(E), resid=mx))
    return out


def _frame(p):
    c, s = np.cos(p[4]), np.sin(p[4])
    return np.array([p[0], p[1]]), np.array([[c, -s], [s, c]])


def _bez_on_oval(o, a0, sweep, full):
    from exact_snap import arc_beziers
    c, R = _frame(o["p"])
    S = np.diag([o["p"][2], o["p"][3]])
    pieces = arc_beziers(np.zeros(2), 1.0, a0, sweep)
    return [[c + R @ S @ q for q in b] for b in pieces]


def apply_ovals(pieces, ovals, w):
    """pieces: [(P, closed, bend_nodes, wp)]. Returns (kept pieces, extra) with extra = [(bez, closed, wp)]: a chain that
    lies on an oval is replaced by the exact oval (the whole oval when no other chain runs along it, else the arc between
    the chain's own ends)."""
    drop, extra = set(), []
    for o in ovals:
        on = []
        for i, (P, closed, bn, wp) in enumerate(pieces):
            if i in drop or len(P) < 8:
                continue
            d = o["tree"].query(P)[0]
            if (d < 1.2 * w).mean() >= 0.8 and (d < 2.0 * w).all():
                on.append(i)
        if not on:
            continue
        rest = [pieces[i][0] for i in range(len(pieces)) if i not in drop and i not in on]
        cover = 0.0
        if rest:
            cover = float((cKDTree(np.vstack(rest)).query(o["E"][::8])[0] < 0.35 * w).mean())
        if cover < 0.35:
            t0 = float(np.round(0.0 / (np.pi / 2)) * (np.pi / 2))
            extra.append((_bez_on_oval(o, t0, 2 * np.pi, True), True, pieces[on[0]][3]))
            drop |= set(on)
            continue
        for i in on:
            P, closed, bn, wp = pieces[i]
            k = o["tree"].query(P)[1]
            t = np.unwrap(k * (2 * np.pi / DENSE) + np.zeros(len(k)))  # nearest-point index as a parameter, unwrapped along the chain
            step = np.diff(t)
            step = (step + np.pi) % (2 * np.pi) - np.pi
            sign = 1.0 if step.sum() >= 0 else -1.0
            total = float(np.abs(step.sum()))
            if closed or total > 2 * np.pi - 0.05:
                extra.append((_bez_on_oval(o, 0.0, 2 * np.pi, True), True, wp))
            else:
                extra.append((_bez_on_oval(o, float(k[0]) * (2 * np.pi / DENSE), sign * total, False), False, wp))
            drop.add(i)
    return [pc for i, pc in enumerate(pieces) if i not in drop], extra


def snap_ellipses(P, closed, ovals, w, tol=0.12, min_len=1.5):
    """Stretches of a chain that lie on an exact oval (at least min_len stroke widths long) are put exactly on it; next to
    points that are not moved the move fades in over about one stroke width."""
    if not ovals or len(P) < 8:
        return P
    P = P.copy()
    for o in ovals:
        n = len(P)
        d, idx = o["tree"].query(P)
        on = d < tol * w
        if not on.any():
            continue
        new = o["E"][idx]
        start = 0
        if closed and not on.all():
            start = int(np.argmin(on))
            P, new, on = (np.roll(a, -start, axis=0) for a in (P, new, on))
        i, runs = 0, []
        while i < n:
            if not on[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and on[j + 1]:
                j += 1
            if (j - i + 1) >= min_len * w:
                runs.append((i, j))
            i = j + 1
        for i, j in runs:
            k = np.arange(i, j + 1)
            f = np.ones(len(k))
            ramp = max(int(1.0 * w), 1)
            if i > 0 or closed:
                f = np.minimum(f, np.clip((k - i) / ramp, 0, 1))
            if j < n - 1 or closed:
                f = np.minimum(f, np.clip((j - k) / ramp, 0, 1))
            P[k] = P[k] + (new[k] - P[k]) * f[:, None]
        if closed and not on.all():
            P = np.roll(P, start, axis=0)
    return P


def _line_oval_points(ln, o):
    """Intersections of the infinite line through segment ln with the dense oval polyline, as (point, oval index)."""
    a = ln[:2]
    u = (ln[2:] - a) / np.linalg.norm(ln[2:] - a)
    E = o["E"]
    sd = (E[:, 0] - a[0]) * u[1] - (E[:, 1] - a[1]) * u[0]
    cross = np.nonzero(np.sign(sd) != np.sign(np.roll(sd, -1)))[0]
    out = []
    for k in cross:
        k2 = (k + 1) % len(E)
        f = sd[k] / (sd[k] - sd[k2])
        out.append((E[k] + (E[k2] - E[k]) * f, k))
    return out


def join_lines_ovals(P, closed, lines, ovals, w, tol=0.03, gap_max=2.5):
    """Where a chain runs along a straight exact line and then along an exact oval, the corner is the exact crossing of the
    line and the oval's centreline: the line run is cut or extended to it and the oval is followed on from it."""
    if not len(lines) or not ovals or len(P) < 20:
        return P
    from exact_snap import _seg_dist
    n = len(P)
    D = np.stack([_seg_dist(P, ln, 0.5 * w) for ln in lines])
    lid = np.where(D.min(axis=0) < tol * w, D.argmin(axis=0), -1)
    for o in ovals:
        d, idx = o["tree"].query(P)
        on_e = d < tol * w

        def runs_of(mask_, key=None):
            out, i = [], 0
            while i < n:
                if not mask_[i] or (key is not None and key[i] < 0):
                    i += 1
                    continue
                j = i
                while j + 1 < n and mask_[j + 1] and (key is None or key[j + 1] == key[i]):
                    j += 1
                if j - i + 1 >= 1.0 * w:
                    out.append((i, j, None if key is None else int(key[i])))
                i = j + 1
            return out

        lruns = runs_of(lid >= 0, lid)
        eruns = runs_of(on_e)
        for (i0, i1, L) in lruns:
            nxt = [r for r in eruns if r[0] > i1 and r[0] - i1 - 1 <= gap_max * w]
            if not nxt:
                continue
            j0, j1, _ = nxt[0]
            gapmid = P[(i1 + j0) // 2]
            cands = _line_oval_points(lines[L], o)
            if not cands:
                continue
            X, kx = min(cands, key=lambda c: np.linalg.norm(c[0] - gapmid))
            if np.linalg.norm(X - gapmid) > 1.6 * w:
                continue
            # line part: cut at X if the run overshoots it, else extend straight to it
            u = (P[i1] - P[i0]) / max(np.linalg.norm(P[i1] - P[i0]), 1e-9)
            s = (P[i0:i1 + 1] - X) @ u
            keep = i0 + int(np.argmax(s > 0)) if (s > 0).any() else i1 + 1
            head = P[:keep]
            seg = np.linalg.norm(X - head[-1])
            ext = [head[-1] + (X - head[-1]) * (t / max(int(np.ceil(seg)), 1)) for t in range(1, max(int(np.ceil(seg)), 1) + 1)] if seg > 0.5 else [X]
            # oval part: the chain travels along the oval from the run start; X is either before it (fill the arc from X) or
            # after it (skip the run's points that lie before X)
            m = len(o["E"])
            k0 = int(idx[j0])
            fwd = (idx[min(j0 + 5, j1)] - k0) % m < m / 2
            step = 1 if fwd else -1
            ahead = ((kx - k0) * step) % m  # how far X lies ahead of the run start along the direction of travel
            if ahead < m / 2:  # X is after the run start: drop the run points that lie before X
                jj = j0
                while jj < j1 and ((int(idx[jj]) - kx) * step) % m > m / 2:
                    jj += 1
                tail = P[jj:]
                arc = np.zeros((0, 2))
            else:  # X is before the run start: follow the oval from X to it
                back = ((k0 - kx) * step) % m
                if back * (2 * np.pi / m) * max(o["p"][2], o["p"][3]) > 4 * w * 4:
                    continue
                arc = np.array([o["E"][(kx + step * q) % m] for q in range(1, back + 1)])
                tail = P[j0:]
            if DEBUG:
                print('  join: line run', (i0, i1), 'oval run', (j0, j1), 'X', np.round(X, 1), 'fwd', fwd, 'ahead', ahead, 'arc points', len(arc))
            return np.vstack([head, np.array(ext), arc, tail])
    return P
