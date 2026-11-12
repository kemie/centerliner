"""Experiment: snap a chain of centreline points onto the exact centerlines read from the SVG (see exact_edges.py).

snap_chain(P, closed, lines, w, corner_r) works in pixel space (same as the engine's chains):
  1. runs of points lying within 0.18 w of one exact line (and at least 0.8 w long) are projected onto that line;
  2. between two runs on different lines, the stretch in between (the cut corner the medial axis makes) is replaced by the
     exact intersection point, if the bend counts as a sharp corner under the usual corner rule (r_eq <= corner_r * w);
     a gap inside one line is bridged straight;
  3. at an open chain's ends, a short leading/trailing stretch is cut back or extended to the point where its line meets a
     second exact line that the end lies on (a stroke ending on a bar, a corner at a free end).
Anything not close to an exact line is left untouched."""
import numpy as np
from scipy import ndimage

DEBUG = False
TOL = 0.18       # a point is "on" an exact line when closer than this many stroke widths
MIN_RUN = 0.8    # shortest run (stroke widths) trusted as lying on an exact line
MAX_GAP = 2.2    # longest stretch (stroke widths) between two runs that is treated as a cut corner
END_REACH = 0.8  # an open end counts as lying on a second line when this close (stroke widths)


def _seg_dist(P, ln, ext):
    a, b = ln[:2], ln[2:]
    ab = b - a
    L = np.linalg.norm(ab)
    u = ab / L
    t = (P - a) @ u
    tc = np.clip(t, -ext, L + ext)
    return np.linalg.norm(P - (a + tc[:, None] * u), axis=1)


def _intersect(l1, l2):
    """Intersection of the infinite lines through two segments; None if (nearly) parallel."""
    a, da = l1[:2], l1[2:] - l1[:2]
    b, db = l2[:2], l2[2:] - l2[:2]
    cr = da[0] * db[1] - da[1] * db[0]
    if abs(cr) < 1e-3 * np.linalg.norm(da) * np.linalg.norm(db):
        return None
    t = ((b - a)[0] * db[1] - (b - a)[1] * db[0]) / cr
    return a + t * da


def _angle(u, v):
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12), -1, 1))))


def _proj(P, ln):
    a = ln[:2]
    u = (ln[2:] - a) / np.linalg.norm(ln[2:] - a)
    return a + ((P - a) @ u)[:, None] * u


def min_bend_radius(seg, w_px):
    """Smallest radius of curvature (pixels) along a chain stretch, from circles through points about 0.35 stroke widths apart."""
    S = ndimage.gaussian_filter1d(np.asarray(seg, float), max(1.0, 0.1 * w_px), axis=0, mode="nearest") if len(seg) >= 5 else np.asarray(seg, float)
    n = len(S)
    q = max(int(round(0.35 * w_px)), 3)
    if n < 2 * q + 1:
        q = max((n - 1) // 2, 1)
    if q < 1 or n < 3:
        return float("inf")
    a, b, c = S[:n - 2 * q], S[q:n - q], S[2 * q:]
    ab, bc, ca = np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1), np.linalg.norm(a - c, axis=1)
    area2 = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))  # twice the triangle area
    with np.errstate(divide="ignore", invalid="ignore"):
        R = ab * bc * ca / (2 * area2)
    R = R[np.isfinite(R)]
    return float(R.min()) if len(R) else float("inf")


def snap_chain(P, closed, lines, w, corner_r):
    if len(lines) == 0 or len(P) < 8:
        return P
    n = len(P)
    D = np.stack([_seg_dist(P, ln, 0.5 * w) for ln in lines])  # (lines, points)
    best = D.argmin(axis=0)
    on = D.min(axis=0) < TOL * w
    ids = np.where(on, best, -1)
    # runs of one line id
    runs, i = [], 0
    while i < n:
        if ids[i] < 0:
            i += 1
            continue
        j = i
        while j + 1 < n and ids[j + 1] == ids[i]:
            j += 1
        if (j - i + 1) >= MIN_RUN * w:
            runs.append((i, j, int(ids[i])))
        i = j + 1
    if not runs:
        return P
    if closed and len(runs) >= 2 and runs[0][0] == 0 and runs[-1][1] == n - 1 and runs[0][2] == runs[-1][2]:
        a = runs.pop()
        runs[0] = (a[0] - n, runs[0][1], runs[0][2])  # wraps around the start: merge the two halves
    out, cursor = [], None
    R = len(runs)
    pts = lambda i0, j0: P[np.arange(i0, j0 + 1) % n]  # noqa: E731 (index range, possibly wrapping)

    # leading stretch of an open chain
    def end_fix(idx_end, first):
        """Return the exact point to start (first=True) or finish the chain at, or None."""
        i0, j0, lid = runs[0] if first else runs[-1]
        stretch = i0 if first else (n - 1 - j0)
        if stretch > MAX_GAP * w:
            return None
        E = P[0] if first else P[-1]
        for k, ln in enumerate(lines):
            if k == lid:
                continue
            if _seg_dist(E[None, :], ln, END_REACH * w)[0] > END_REACH * w:
                continue
            stub = P[:i0 + 1] if first else P[j0:]
            if len(stub) >= 0.5 * w and _seg_dist(stub, ln, END_REACH * w).max() < TOL * w:
                continue  # the stretch lies along that line itself (a leg of the chain on a wall), not an artefact at a bar
            X = _intersect(lines[lid], ln)
            if X is None or np.linalg.norm(X - E) > 1.6 * w:
                continue
            u = (lines[lid][2:] - lines[lid][:2])
            if _angle(u, ln[2:] - ln[:2]) < 20 or abs(_angle(u, ln[2:] - ln[:2]) - 180) < 20:
                continue
            return X
        return None

    pieces = []  # list of arrays in chain order
    first_fix = None if closed else end_fix(0, True)
    last_fix = None if closed else end_fix(n - 1, False)

    # decide what happens in every stretch between two runs: a sharp corner at the exact intersection ("X"), a plain
    # bridge inside one line ("same"), or leave the original points (a real curve, "keep")
    dec = []
    for r in range(R):
        i0, j0, lid = runs[r]
        if r + 1 >= R and not closed:
            break
        nxt = runs[(r + 1) % R]
        gi0, gi1 = j0 + 1, (nxt[0] if nxt[0] > j0 else nxt[0] + n) - 1
        gap = pts(gi0, gi1) if gi1 >= gi0 else P[:0]
        if nxt[2] == lid:
            dec.append(("same" if len(gap) <= MAX_GAP * w else "keep", None, gap))
            continue
        if len(gap) > MAX_GAP * w:
            dec.append(("keep", None, gap))
            continue
        X = _intersect(lines[lid], lines[nxt[2]])
        ua = lines[lid][2:] - lines[lid][:2]
        ub = lines[nxt[2]][2:] - lines[nxt[2]][:2]
        da = P[j0 % n] - P[i0 % n]
        db = P[nxt[1] % n] - P[nxt[0] % n]
        ua = ua if np.dot(ua, da) >= 0 else -ua
        ub = ub if np.dot(ub, db) >= 0 else -ub
        theta = _angle(ua, ub)
        centre = gap.mean(axis=0) if len(gap) else P[j0 % n]  # runs that abut meet at the end of the first one
        if X is None or theta < 20 or np.linalg.norm(X - centre) > 1.2 * w:
            dec.append(("keep", None, gap))
            continue
        d = float(np.min(np.linalg.norm(gap - X, axis=1))) if len(gap) else float(np.linalg.norm(X - centre))
        sec = 1.0 / np.cos(np.radians(theta) / 2)
        r_eq = d / (sec - 1) if sec > 1.0001 else np.inf
        dec.append(("X", X, gap) if r_eq <= corner_r * w else ("keep", None, gap))

    if DEBUG:
        print('  runs', [(a, b, c) for a, b, c in runs], 'n', n)
        print('  dec', [(k, None if X is None else np.round(X), len(g)) for k, X, g in dec])

    def run_points(r, keep_start, keep_end):
        """Projected points of run r; next to a curve that is kept, blend back to the original points over ~1 stroke width."""
        i0, j0, lid = runs[r]
        orig = pts(i0, j0)
        new = _proj(orig, lines[lid])
        k = len(orig)
        f = np.ones(k)
        ramp = max(int(1.0 * w), 1)
        if keep_start:
            f = np.minimum(f, np.clip(np.arange(k) / ramp, 0, 1))
        if keep_end:
            f = np.minimum(f, np.clip((k - 1 - np.arange(k)) / ramp, 0, 1))
        return orig + (new - orig) * f[:, None]

    start = 0
    if not closed:
        if first_fix is not None:
            pieces.append(first_fix[None, :])
        else:
            pieces.append(P[:runs[0][0]])
    for r in range(R):
        prev_dec = dec[r - 1] if (r > 0 or closed) and dec else None
        next_dec = dec[r] if r < len(dec) else None
        ks = prev_dec is not None and prev_dec[0] == "keep"
        ke = next_dec is not None and next_dec[0] == "keep"
        if not closed and r == 0 and first_fix is None:
            ks = True
        if not closed and r == R - 1 and last_fix is None:
            ke = True
        pieces.append(run_points(r, ks, ke))
        if next_dec is None:
            continue
        kind, X, gap = next_dec
        if kind == "X":
            pieces.append(X[None, :])
        elif kind == "keep":
            pieces.append(gap)
    if not closed:
        if last_fix is not None:
            pieces.append(last_fix[None, :])
        else:
            pieces.append(P[runs[-1][1] + 1:])
    Q = np.vstack([p for p in pieces if len(p)])
    keep = np.r_[True, np.linalg.norm(np.diff(Q, axis=0), axis=1) > 1e-6]
    Q = Q[keep]
    dense = [Q[0]]  # the engine expects roughly 1 px spacing: subdivide the long straight jumps, keep every vertex
    for a, b in zip(Q[:-1], Q[1:]):
        k = int(np.ceil(np.linalg.norm(b - a)))
        for t in range(1, k + 1):
            dense.append(a + (b - a) * (t / k))
    return np.array(dense)


def merge_lines(lines, w, max_gap=2.5, ang=2.0, off=0.06):
    """Join exact centerlines that are one straight line interrupted by something crossing it (a wall split where a bar
    meets it): same direction, same position, ends closer than max_gap stroke widths."""
    L = [np.array(l, float) for l in lines]
    changed = True
    while changed:
        changed = False
        for i in range(len(L)):
            for j in range(i + 1, len(L)):
                a, b = L[i], L[j]
                da, db = a[2:] - a[:2], b[2:] - b[:2]
                ua, ub = da / np.linalg.norm(da), db / np.linalg.norm(db)
                if abs(np.dot(ua, ub)) < np.cos(np.radians(ang)):
                    continue
                n = np.array([-ua[1], ua[0]])
                if max(abs(np.dot(b[:2] - a[:2], n)), abs(np.dot(b[2:] - a[:2], n))) > off * w:
                    continue
                ta = np.array([0.0, np.linalg.norm(da)])
                tb = np.sort([np.dot(b[:2] - a[:2], ua), np.dot(b[2:] - a[:2], ua)])
                if max(ta[0], tb[0]) - min(ta[1], tb[1]) > max_gap * w:  # gap between the two pieces
                    continue
                lo, hi = min(ta[0], tb[0]), max(ta[1], tb[1])
                L[i] = np.r_[a[:2] + ua * lo, a[:2] + ua * hi]
                del L[j]
                changed = True
                break
            if changed:
                break
    return np.array(L).reshape(-1, 4)


def _arm(P, i0, i1):
    """Straight-line fit through P[i0..i1] (indices may wrap): (point, unit direction along the chain, worst deviation)."""
    n = len(P)
    Q = P[np.arange(i0, i1 + 1) % n]
    c = Q.mean(axis=0)
    d = np.linalg.svd(Q - c, full_matrices=False)[2][0]
    if np.dot(Q[-1] - Q[0], d) < 0:
        d = -d
    dev = np.abs((Q[:, 0] - c[0]) * d[1] - (Q[:, 1] - c[1]) * d[0]).max()
    return c, d, dev


ON_CHAIN = 0.15   # a corner point this close to the chain (stroke widths) lies on it
OFF_CHAIN = 0.4   # ... one farther than this is off it


def sharpen_bends(P, closed, w, corner_r, min_turn=50.0, max_turn=150.0, corners=None, snap=0.25, circles=None, ovals=None):
    """A tight bend between two straight-ish arms is a corner the medial axis has cut: put the exact intersection of the arms
    there. Same corner rule as everywhere else (equivalent radius <= corner_r * w). It covers what assemble() cannot see,
    a line meeting a long gentle curve (the tip of an eye), where only one side is a detected line.
    corners: exact corner points (pixel space, from the outline's round corners); a corner within snap * w of one sits exactly on it.
    Left alone: a bend on an exact circle wider than the corner rule (a rounded corner, not a cut one), and a bend whose arm runs
    along such a circle; circles are (centre, radius, span) as for snap_circles.
    P is one chain in pixel space; returns a chain with the bends replaced by one point (densified to ~1 px spacing)."""
    n = len(P)
    m, j = int(round(1.0 * w)), int(round(0.8 * w))
    if n < 2 * (m + j) + 3:
        return P
    cand = []
    rng = range(n) if closed else range(m + j, n - m - j)
    for i in rng:
        ca, da, dva = _arm(P, i - m - j, i - m)
        cb, db, dvb = _arm(P, i + m, i + m + j)
        if dva > 0.06 * w or dvb > 0.06 * w:
            continue
        theta = _angle(da, db)
        if not (min_turn <= theta <= max_turn):
            continue
        X = _intersect(np.r_[ca, ca + da], np.r_[cb, cb + db])
        if X is None:
            continue
        zone = P[np.arange(i - m, i + m + 1) % n]
        if circles or ovals:
            arms = P[np.r_[np.arange(i - m - j, i - m + 1), np.arange(i + m, i + m + j + 1)] % n]
            if ovals and any((o["tree"].query(arms)[0] < 0.12 * w).mean() > 0.3 for o in ovals):
                continue  # an arm that runs along an exact oval: the exact junction is already in the chain
            if circles and any(r > corner_r * w and (np.abs(np.linalg.norm(zone - c, axis=1) - r) < 0.12 * w).all() for c, r, _ in circles):
                continue  # a rounded corner lying on an exact circle
            if circles and any(r > corner_r * w and (np.abs(np.linalg.norm(arms - c, axis=1) - r) < 0.12 * w).mean() > 0.3 for c, r, _ in circles):
                continue  # an arm that runs along a rounded corner: its line would tilt and put the corner in the wrong place
        if min_bend_radius(zone, w) > 1.2 * corner_r * w:
            continue  # the chain itself rounds this bend with a radius above the corner rule: a real curve (the arms are only the straight middle of an S)
        d = float(np.linalg.norm(zone - X, axis=1).min())
        if corners is not None and len(corners):
            dc = np.linalg.norm(corners - X, axis=1)
            if dc.min() <= snap * w:
                X = corners[dc.argmin()]
        if d > 1.5 * w:
            continue
        sec = 1.0 / np.cos(np.radians(theta) / 2)
        if d / (sec - 1) > corner_r * w:
            continue
        cand.append((i, X, theta, ca, da, cb, db, d))
    if not cand:
        return P
    # one corner per cluster: the candidate whose bend is sharpest
    clusters, last = [], None
    for c in cand:
        if last is not None and c[0] - last <= m:
            clusters[-1].append(c)
        else:
            clusters.append([c])
        last = c[0]
    if closed and len(clusters) > 1 and clusters[0][0][0] + n - clusters[-1][-1][0] <= m:  # one cluster across the start
        clusters[0] += clusters.pop()

    def best(cl):
        sharp = max(cl, key=lambda c: c[2])
        if sharp[7] > OFF_CHAIN * w:
            # the sharpest bend puts the corner well off the chain: further along a wall that runs into a curve the arm tilts, the
            # turn looks bigger and the corner lands outside the ink (the shoulders of man-avatar); a bend whose corner lies on the chain is the real one
            on = [c for c in cl if c[7] <= ON_CHAIN * w]
            if on:
                return max(on, key=lambda c: c[2])
        return sharp

    picked = [best(cl) for cl in clusters]
    if DEBUG:
        print('  sharpen', [(c[0], np.round(c[1], 1).tolist(), round(c[2])) for c in picked])

    def leave(ca, da, i, step):  # first index, walking from the bend outwards, that is back on the arm's straight line
        k = i
        for _ in range(m):
            k += step
            q = P[k % n] - ca
            if abs(q[0] * da[1] - q[1] * da[0]) <= 0.05 * w:
                break
        return k

    zones = []
    for i, X, _, ca, da, cb, db, _d in picked:
        zones.append((leave(ca, da, i, -1), leave(cb, db, i, 1), X))
    if closed:  # start the loop half-way between two corners so that no replaced stretch wraps around the start
        i0, i1 = zones[0][0], zones[1][0] if len(zones) > 1 else zones[0][0] + n
        shift = ((i0 + i1) // 2) % n
        P = np.roll(P, -shift, axis=0)
        zones = sorted((((a - shift) % n), ((a - shift) % n) + (b - a), X) for a, b, X in zones)
    out, pos = [], 0
    for a, b, X in zones:
        if a < pos or b <= a:
            continue
        out += [P[pos:a + 1], X[None, :]]
        pos = b
    out.append(P[pos:])
    Q = np.vstack([p for p in out if len(p)])
    dense = [Q[0]]
    for a, b in zip(Q[:-1], Q[1:]):
        k = max(int(np.ceil(np.linalg.norm(b - a))), 1)
        for t in range(1, k + 1):
            dense.append(a + (b - a) * (t / k))
    return np.array(dense)


def snap_circles(P, closed, circles, w, tol=0.12, min_len=1.5, reach=1.6):
    """Stretches of a chain that lie on an exact circle (centre, radius, span: read from two concentric outline arcs one stroke
    width apart) are put exactly on it. A stretch must be at least min_len stroke widths long; next to points that are not
    moved, the move fades in over about one stroke width so that no step appears. At an open chain's end the medial axis
    swings off the arc (a flat cut end); if the arc ends squarely at a known angle (span), the chain is run on along the
    circle to that end instead."""
    if not len(circles) or len(P) < 8:
        return P
    P = P.copy()
    for c, r, span in circles:
        n = len(P)
        v = P - c
        dist = np.linalg.norm(v, axis=1)
        on = np.abs(dist - r) < tol * w
        if not on.any():
            continue
        if on.all():
            P = c + v * (r / np.maximum(dist, 1e-9))[:, None]
            if not closed and span is not None and span[2]:
                P = _extend_arc(P, (0, n - 1), c, r, span, w)
            continue
        start = 0
        if closed:  # begin at a point that is off the circle, so that no stretch wraps around the start
            start = int(np.argmin(on))
            P, v, dist, on = (np.roll(a, -start, axis=0) for a in (P, v, dist, on))
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
        if not closed:  # a short stretch at a chain end that leaves the circle gradually is the medial axis bending toward a junction
            runs = [list(q) for q in runs]
            if runs and 0 < runs[0][0] <= reach * w and np.abs(dist[:runs[0][0]] - r).max() <= 0.4 * w:
                runs[0][0] = 0
            if runs and 0 < n - 1 - runs[-1][1] <= reach * w and np.abs(dist[runs[-1][1] + 1:] - r).max() <= 0.4 * w:
                runs[-1][1] = n - 1
        for i, j in runs:
            k = np.arange(i, j + 1)
            f = np.ones(len(k))
            ramp = max(int(1.0 * w), 1)
            if i > 0 or closed:
                f = np.minimum(f, np.clip((k - i) / ramp, 0, 1))
            if j < n - 1 or closed:
                f = np.minimum(f, np.clip((j - k) / ramp, 0, 1))
            new = c + v[k] * (r / np.maximum(dist[k], 1e-9))[:, None]
            P[k] = P[k] + (new - P[k]) * f[:, None]
        if closed:
            P = np.roll(P, start, axis=0)
        elif span is not None and span[2] and len(runs) == 1:
            P = _extend_arc(P, runs[0], c, r, span, w)
    return P


def _extend_arc(P, run, c, r, span, w, reach=1.6):
    """Replace a stub that swings off the arc at either end of an open chain by the arc itself, up to the span's end angle."""
    n = len(P)
    i, j = run
    ang = lambda q: np.degrees(np.arctan2(q[1] - c[1], q[0] - c[0]))  # noqa: E731
    lo, hi = span[:2]
    a_i, a_j = ang(P[i]), ang(P[j])
    inside = lambda a: ((a - lo) % 360) <= ((hi - lo) % 360) + 1.0  # noqa: E731
    if not (inside(a_i) and inside(a_j)):
        return P
    k = min(8, j - i)  # direction of travel from the first few points (the ends alone cannot tell on an arc over 180 degrees)
    s = 1.0 if ((ang(P[i + k]) - a_i + 180) % 360 - 180) > 0 else -1.0  # angle grows along the chain (+) or shrinks (-)
    head, tail = P[:i], P[j + 1:]
    out = [P[i:j + 1]]
    for stub, at_start in ((head, True), (tail, False)):
        if len(stub) > reach * w:
            if at_start:
                out.insert(0, head)
            else:
                out.append(tail)
            continue
        # the end angle lying beyond this end: lo or hi, depending on the direction of travel
        end = (lo if s > 0 else hi) if at_start else (hi if s > 0 else lo)
        a0 = a_i if at_start else a_j
        d = ((end - a0 + 180) % 360) - 180
        if abs(np.radians(d)) * r > reach * w or abs(d) < 0.5 or (np.sign(d) == s) == at_start:
            # not a sensible extension: keep the stub as it is
            if at_start:
                out.insert(0, head)
            else:
                out.append(tail)
            continue
        m = max(int(np.ceil(abs(np.radians(d)) * r)), 1)
        th = np.radians(a0 + d * np.arange(0 if at_start else 1, m + (0 if at_start else 1)) / m) if not at_start else np.radians(a0 + d * np.arange(m, 0, -1) / m)
        arc = c + r * np.c_[np.cos(th), np.sin(th)]
        if at_start:
            out.insert(0, arc)
        else:
            out.append(arc)
    return np.vstack(out)


def arc_of_chain(P, closed, circles, w, tol=0.03):
    """If a chain lies entirely on one exact circle, (centre, radius, start angle, signed sweep in radians) of it, else None.
    A closed chain is a full circle; its start is put at the nearest quarter turn so that the anchors sit at the top, right,
    bottom and left."""
    for c, r, span in circles:
        v = P - c
        if np.abs(np.linalg.norm(v, axis=1) - r).max() > tol * w:
            continue
        ang = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
        sweep = float(ang[-1] - ang[0])
        if closed:
            if abs(sweep) < np.radians(300):
                continue
            if span is not None:  # the artwork's ring has a gap: an open arc between its ends
                return c, r, float(np.radians(span[0])), float(np.radians(span[1] - span[0]))
            q = np.pi / 2
            return c, r, float(np.round(ang[0] / q) * q), float(np.sign(sweep) * 2 * np.pi)
        if np.radians(20) < abs(sweep) < np.radians(340):
            return c, r, float(ang[0]), sweep
    return None


def arc_beziers(c, r, a0, sweep):
    """Cubic Beziers (the engine's [p0, p1, p2, p3] lists) for a circular arc. An arc over 100 degrees is cut at the top, right,
    bottom and left when that needs no more pieces than an even split (so anchors sit where bars and walls meet a ring); otherwise
    (and for shorter arcs) one piece per 90 degrees or less."""
    a1 = a0 + sweep
    if abs(sweep) > np.radians(100):
        lo, hi = min(a0, a1), max(a0, a1)
        q = np.pi / 2
        eps = np.radians(2.0)  # a cut closer than this to an end would only make a sliver of a piece
        inner = [k * q for k in range(int(np.ceil((lo + eps) / q)), int(np.floor((hi - eps) / q)) + 1)]
        cuts = [lo] + inner + [hi]
        if sweep < 0:
            cuts = cuts[::-1]
        if len(cuts) - 1 > int(np.ceil(abs(sweep) / (np.pi / 2) - 0.03)):
            cuts = None  # more pieces than an even split needs: not worth it
    else:
        cuts = None
    if cuts is None:
        k = int(np.ceil(abs(sweep) / (np.pi / 2) - 0.03))
        cuts = list(a0 + sweep * np.arange(k + 1) / k)
    e = lambda x: np.array([np.cos(x), np.sin(x)])  # noqa: E731
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        d = b - a
        t = 4.0 / 3.0 * np.tan(d / 4)
        p0, p3 = c + r * e(a), c + r * e(b)
        t0 = r * np.array([-np.sin(a), np.cos(a)])
        t3 = r * np.array([-np.sin(b), np.cos(b)])
        out.append([p0, p0 + t * t0, p3 - t * t3, p3])
    return out
