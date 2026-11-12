"""Complete rings first (slice 1b of "shapes first").

A ring (a balloon, a bead, the circle of a bow) whose hole is a clean circle or ellipse with ink all the way round is one shape,
however many strokes touch it. The pipeline otherwise draws it in pieces (arcs shared with the strokes that run along it, with a
corner anchor at every junction). Here the ring is drawn as one closed exact shape (4 anchors) and every chain is cut back to
it where it runs along the ring, so strokes end on the ring.

find_rings(mask, w)                         -> rings (dicts like exact_ovals.find_ovals, circles included)
complete_rings(pieces, rings, w, skip)      -> (pieces with the ring stretches cut out, [(bez, True, width)] ring paths)"""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import measure

from exact_ovals import _fit_hole, ellipse_points, _bez_on_oval, MAX_RESID

CORE = 0.35       # a chain point is on the ring when this close (stroke widths)
NEAR = 0.6        # junction zone: points this close next to an on-stretch are cut with it ...
NEAR_LEN = 0.5    # ... for at most this many stroke widths
MIN_ON = 1.0      # an on-stretch shorter than this is a stroke crossing the ring, not running along it
STRAIGHT = 0.65  # a stretch of chain whose sagitta is at most this share of the ring's over the same stretch is a straight stroke ...
CURVED = 0.15    # ... and it only touches the ring when the ring bends away from it by at least this much over the same stretch
SHORT_TOUCH = 4.5  # ... and only when the contact is shorter than this (stroke widths): measured touches 3.0, shared outlines 5.8 and more
HUG = 0.6        # a leftover whose points all lie within this many stroke widths of the ring is dropped
MIN_KEEP = 0.8    # a leftover shorter than this is a stub of the ring itself and is dropped


def find_rings(mask, w):
    """Circles and ellipses among the enclosed holes of the ink mask whose ring of ink is complete and about one stroke thick."""
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
        if mx > MAX_RESID * w or p[2] <= 0 or p[3] <= 0 or p[3] < 0.6 * w:
            continue
        E = ellipse_points(p, 600)
        ix, iy = E[:, 0].astype(int), E[:, 1].astype(int)
        ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        if ok.mean() < 1.0 or mask[iy[ok], ix[ok]].mean() < 0.97:
            continue  # the centreline is not ink all round
        if np.median(dt[iy[ok], ix[ok]]) > 1.3 * (w / 2):
            continue  # ink much thicker than a stroke: a plate with a round hole
        circle = abs(p[2] - p[3]) < 0.03 * max(p[2], p[3])
        if circle:
            p = np.array([p[0], p[1], (p[2] + p[3]) / 2, (p[2] + p[3]) / 2, 0.0])
        E = ellipse_points(p)
        out.append(dict(p=p, E=E, tree=cKDTree(E), resid=mx, circle=circle))
    return out


def _runs(flag):
    out, i, n = [], 0, len(flag)
    while i < n:
        if not flag[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flag[j + 1]:
            j += 1
        out.append((i, j))
        i = j + 1
    return out


def _dev(Q):
    """Largest distance of the points from the chord through the first and last one."""
    d = Q[-1] - Q[0]
    L = np.linalg.norm(d)
    if L < 1e-9:
        return 0.0
    return float(np.abs(d[0] * (Q[:, 1] - Q[0][1]) - d[1] * (Q[:, 0] - Q[0][0])).max() / L)


def _cut(P, closed, ring, w):
    """Cut the stretches of chain P that run along the ring out of it. Returns the leftover chains (each an (x, y) array), and
    whether anything was cut."""
    d, idx = ring["tree"].query(P)
    on = d < CORE * w
    if not on.any():
        return [(P, closed)], False
    near = d < NEAR * w
    n = len(P)
    reach = int(NEAR_LEN * w)
    grown = np.zeros(n, bool)
    for i, j in _runs(on):  # carry the cut a little way into the junction zone, where the skeleton is bent away from the ring
        if (j - i + 1) < MIN_ON * w:
            continue  # a stroke only crossing the ring: not along it, so nothing is cut (and nothing is grown into its neighbours)
        lo, hi = i, j
        while lo > 0 and near[lo - 1] and i - lo < reach:
            lo -= 1
        while hi < n - 1 and near[hi + 1] and hi - j < reach:
            hi += 1
        grown[lo:hi + 1] = True
    for i, j in _runs(grown):
        if (j - i + 1) < MIN_ON * w:
            grown[i:j + 1] = False  # only a stroke crossing the ring
            continue
        # a straight stroke that only touches the curved ring (a wall tangent to an oval, a bar across the top of a paw pad)
        # is its own stroke: it stays whole. A stroke that bends with the ring shares its outline and is cut.
        devr = _dev(ring["E"][idx[i:j + 1]])
        if on[i:j + 1].sum() < SHORT_TOUCH * w and devr >= CURVED * w and _dev(P[i:j + 1]) <= STRAIGHT * devr:
            grown[i:j + 1] = False
    if not grown.any():
        return [(P, closed)], False
    if closed:
        if grown.all():
            return [], True
        s = int(np.argmin(grown))  # start at a point off the ring so that no leftover wraps round the seam
        P, idx, grown = np.roll(P, -s, axis=0), np.roll(idx, -s), np.roll(grown, -s)
        P = np.vstack([P, P[:1]])
        idx, grown = np.r_[idx, idx[0]], np.r_[grown, grown[0]]
    out = []
    n = len(P)
    off = ~grown
    d_full = ring["tree"].query(P)[0]
    for i, j in _runs(off):
        if (j - i + 1) < MIN_KEEP * w:
            continue
        if (d_full[i:j + 1] < HUG * w).all():
            continue  # a leftover that only lies along the ring (the corner bit of a side that was cut): the ring draws it
        seg = P[i:j + 1].copy()
        if i > 0 and grown[i - 1]:
            seg[0] = ring["E"][idx[i]]  # the stroke now starts exactly on the ring
        if j < n - 1 and grown[j + 1]:
            seg[-1] = ring["E"][idx[j]]
        out.append((seg, False))
    return out, True


def complete_rings(pieces, rings, w, skip=()):
    """pieces: [(P (x, y), closed, bend_nodes, wp)]. skip: circles (cx, cy, r) the circle code already draws exactly."""
    extra = []
    pieces = list(pieces)
    for ring in rings:
        p = ring["p"]
        if ring["circle"] and any(np.hypot(p[0] - c[0], p[1] - c[1]) < 0.5 * w and abs(p[2] - c[2]) < 0.5 * w for c in skip):
            continue
        near_len = []
        for P, closed, bn, wp in pieces:
            d = ring["tree"].query(P)[0]
            near_len.append(int((d < CORE * w).sum()))
        if max(near_len, default=0) < 3 * w:
            continue  # no chain runs along this ring: leave it to the rest of the pipeline
        # already one clean closed piece on the ring and nothing else along it: nothing to improve
        ons = [i for i, nl in enumerate(near_len) if nl >= 1.5 * w]
        if len(ons) == 1 and pieces[ons[0]][1] and ring.get("kind") != "free":  # (a thin-hole ring is redrawn anyway: the old piece is whole, its fitting is what fails)
            d = ring["tree"].query(pieces[ons[0]][0])[0]
            if (d < CORE * w).mean() >= 0.95:
                continue
        new_pieces, widths = [], []
        for i, (P, closed, bn, wp) in enumerate(pieces):
            if near_len[i] == 0:
                new_pieces.append((P, closed, bn, wp))
                continue
            left, cut = _cut(np.asarray(P, float), closed, ring, w)
            if not cut:
                new_pieces.append((P, closed, bn, wp))
                continue
            widths.append(wp)
            for seg, c in left:
                new_pieces.append((seg, c, [], wp))
        pieces = new_pieces
        wp = float(np.median(widths)) if widths else w
        ring["done"] = True
        if "parts" in ring:
            extra += [(b, c, wp) for b, c in ring["parts"]]  # a ring that gave up a shared side comes in open pieces
        else:
            extra.append((ring["bez"] if "bez" in ring else _bez_on_oval(ring, 0.0, 2 * np.pi, True), True, wp))
    return pieces, extra


# ---------------------------------------------------------------------------- rounded rectangles (slice 2)
THIN_RR_RESID = 0.08  # a thin plate must fit a rectangle this closely to be drawn as one (a pole with a round top and square foot does not)
RR_RESID = 0.12   # worst distance (stroke widths) of the hole boundary from the fitted rounded rectangle


def _rr_sdf(q, a, b, r):
    """Signed distance from points q (in the rectangle's own frame, centre at 0) to a rounded rectangle of half sizes a, b and corner radius r."""
    d = np.abs(q) - np.array([a - r, b - r])
    return np.hypot(np.maximum(d[:, 0], 0), np.maximum(d[:, 1], 0)) + np.minimum(np.maximum(d[:, 0], d[:, 1]), 0) - r


def _fit_rrect(B, w):
    """Least-squares rounded rectangle through the boundary points B: (cx, cy, a, b, r, theta), worst residual."""
    from scipy import optimize
    c0 = B.mean(axis=0)
    vt = np.linalg.svd(B - c0, full_matrices=False)[2]
    best = None
    for th0 in (0.0, float(np.arctan2(vt[0][1], vt[0][0]))):
        R = np.array([[np.cos(th0), np.sin(th0)], [-np.sin(th0), np.cos(th0)]])
        q = (B - c0) @ R.T
        a0, b0 = np.abs(q[:, 0]).max(), np.abs(q[:, 1]).max()

        def res(p):
            c, s = np.cos(p[5]), np.sin(p[5])
            qq = (B - p[:2]) @ np.array([[c, s], [-s, c]]).T
            return _rr_sdf(qq, p[2], p[3], min(max(p[4], 0.01), min(p[2], p[3]) - 0.01))

        sol = optimize.least_squares(res, [c0[0], c0[1], a0, b0, 0.3 * min(a0, b0), th0], x_scale=[1, 1, 1, 1, 1, 0.1], max_nfev=60)
        if best is None or sol.cost < best.cost:
            best = sol
    p = best.x
    p[4] = min(max(p[4], 0.0), min(p[2], p[3]))
    return p, float(np.abs(best.fun).max())


def _rr_bez(p, grow):
    """Closed path (8 cubics: 4 sides, 4 corner arcs) of the rounded rectangle p grown outward by `grow`."""
    cx, cy, a, b, r, th = p
    a, b, r = a + grow, b + grow, r + grow
    r = min(r, a, b)
    k = 0.5522847498 * r
    c, s = np.cos(th), np.sin(th)
    Rm = np.array([[c, -s], [s, c]])
    T = lambda pt: np.array([cx, cy]) + Rm @ np.asarray(pt, float)  # noqa: E731
    segs = []

    def line(p0, p1):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        segs.append([T(p0), T(p0 + (p1 - p0) / 3), T(p0 + 2 * (p1 - p0) / 3), T(p1)])

    def arc(p0, h0, h1, p1):
        segs.append([T(p0), T(np.asarray(p0) + h0), T(np.asarray(p1) + h1), T(p1)])

    line((-a + r, -b), (a - r, -b))
    arc((a - r, -b), (k, 0), (0, -k), (a, -b + r))
    line((a, -b + r), (a, b - r))
    arc((a, b - r), (0, k), (k, 0), (a - r, b))
    line((a - r, b), (-a + r, b))
    arc((-a + r, b), (-k, 0), (0, k), (-a, b - r))
    line((-a, b - r), (-a, -b + r))
    arc((-a, -b + r), (0, -k), (-k, 0), (-a + r, -b))
    return segs


def _rect_bez(p, grow):
    """Sharp-cornered rectangle (4 straight sides) of the fitted rectangle p grown by `grow`."""
    cx, cy, a, b, r, th = p
    a, b = a + grow, b + grow
    c, s = np.cos(th), np.sin(th)
    Rm = np.array([[c, -s], [s, c]])
    T = lambda q: np.array([cx, cy]) + Rm @ np.asarray(q, float)  # noqa: E731
    pts = [(-a, -b), (a, -b), (a, b), (-a, b)]
    segs = []
    for i in range(4):
        p0, p1 = np.array(pts[i]), np.array(pts[(i + 1) % 4])
        segs.append([T(p0), T(p0 + (p1 - p0) / 3), T(p0 + 2 * (p1 - p0) / 3), T(p1)])
    return segs


def _dense(bez, n=24):
    u = np.linspace(0, 1, n, endpoint=False)[:, None]
    return np.vstack([(1 - u) ** 3 * c[0] + 3 * (1 - u) ** 2 * u * c[1] + 3 * (1 - u) * u ** 2 * c[2] + u ** 3 * c[3] for c in bez])


def find_rrects(mask, w, taken=(), corner_r=0.8):
    """Rounded rectangles among the enclosed holes of the ink mask whose ring of ink is complete and about one stroke thick.
    taken: rings (find_rings) already found; holes at the same place are skipped. corner_r: the engine's corner rule (stroke
    widths): corners rounder than 1.15 times that stay rounded, tighter ones are sharp corners and left to the old pipeline."""
    lab, n = ndimage.label(~mask)
    border = set(lab[0]) | set(lab[-1]) | set(lab[:, 0]) | set(lab[:, -1])
    dt = ndimage.distance_transform_edt(mask)
    H, W = mask.shape
    out = []
    for k in range(1, n + 1):
        if k in border:
            continue
        reg = lab == k
        A = int(reg.sum())
        if A < 60:
            continue
        ys, xs = np.nonzero(reg)
        h, ww = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        thin = max(h, ww) >= FREE_MIN_SIDE * w and 2 * ndimage.distance_transform_edt(reg).max() < FREE_THIN * w and measure.euler_number(reg, connectivity=1) == 1
        if A / (h * ww) < 0.8 or (min(h, ww) < 0.8 * w and not thin):
            continue  # not rectangle-like (an ellipse fills 0.785 of its box), or too small to be a shape
        cs = measure.find_contours(np.pad(reg, 1).astype(float), 0.5)
        c = max(cs, key=len)
        B = np.c_[c[:, 1] - 1 + 0.5, c[:, 0] - 1 + 0.5]
        if len(B) < 40:
            continue
        cx, cy = xs.mean() + 0.5, ys.mean() + 0.5
        if any(np.hypot(cx - r["p"][0], cy - r["p"][1]) < 0.5 * w for r in taken):
            continue
        p, mx = _fit_rrect(B, w)
        rounded = p[4] + w / 2 >= 1.15 * corner_r * w
        capsule = abs(p[4] - min(p[2], p[3])) < 0.15 * min(p[2], p[3])
        if mx > RR_RESID * w or (not rounded and not (thin and mx <= THIN_RR_RESID * w)):
            continue  # (tight corners: only a thin plate that fits a rectangle exactly; the rest is left to the old pipeline)
        bez = _rr_bez(p, w / 2) if (rounded or capsule) else _rect_bez(p, w / 2)
        E = _dense(bez)
        ix, iy = E[:, 0].astype(int), E[:, 1].astype(int)
        ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        if ok.mean() < 1.0 or mask[iy[ok], ix[ok]].mean() < 0.97:
            continue  # the centreline is not ink all round
        if np.median(dt[iy[ok], ix[ok]]) > 1.3 * (w / 2):
            continue  # ink much thicker than a stroke
        out.append(dict(p=np.array([cx, cy, p[2], p[3], p[5]]), E=E, tree=cKDTree(E), resid=mx, circle=False, bez=bez, kind="rrect", rr=p))
    return out


# ---------------------------------------------------------------------------- any complete ring of a thin hole (slice 4)
FREE_THIN = 0.9      # an enclosed hole thinner than this (stroke widths, twice its largest inscribed radius): the line fitting fails at its ends
FREE_MIN_SIDE = 2.0  # ... and at least this long (a tiny slot is not a plate)


def _resample_closed(poly, step=1.0):
    P = np.asarray(poly, float)
    if np.allclose(P[0], P[-1]):
        P = P[:-1]
    Q = np.vstack([P, P[:1]])
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(Q, axis=0).T))]
    t = np.arange(0, seg[-1], step)
    return np.c_[np.interp(t, seg, Q[:, 0]), np.interp(t, seg, Q[:, 1])]


def find_free_rings(mask, w, taken=()):
    """Complete rings of any shape around a thin enclosed hole (a plate, a blade, a capsule): the centreline is the hole's outline
    pushed out by half a stroke. Returns dicts with P, the closed centreline polyline (x, y), 1 pixel apart; the caller fits it
    (fit_chain), so the engine's corner rule and the corner radius setting apply as everywhere else."""
    from shapely.geometry import Polygon
    lab, n = ndimage.label(~mask)
    border = set(lab[0]) | set(lab[-1]) | set(lab[:, 0]) | set(lab[:, -1])
    dt_ink = ndimage.distance_transform_edt(mask)
    H, W = mask.shape
    out = []
    for k in range(1, n + 1):
        if k in border:
            continue
        reg = lab == k
        if reg.sum() < 40:
            continue
        ys, xs = np.nonzero(reg)
        if max(np.ptp(xs), np.ptp(ys)) + 1 < FREE_MIN_SIDE * w:
            continue
        if measure.euler_number(reg, connectivity=1) != 1:
            continue  # an annular gap with an island inside (a ring between two rings): not one hole of a plate
        if 2 * ndimage.distance_transform_edt(reg).max() >= FREE_THIN * w:
            continue
        cx, cy = xs.mean() + 0.5, ys.mean() + 0.5
        if any(np.hypot(cx - r["p"][0], cy - r["p"][1]) < 0.5 * w for r in taken):
            continue
        cs = measure.find_contours(np.pad(reg, 1).astype(float), 0.5)
        c = max(cs, key=len)
        B = np.c_[c[:, 1] - 1 + 0.5, c[:, 0] - 1 + 0.5]
        if len(B) < 30:
            continue
        poly = Polygon(B).buffer(0).buffer(w / 2, join_style=1, resolution=16)
        if poly.is_empty or poly.geom_type != "Polygon":
            continue
        P = _resample_closed(np.array(poly.exterior.coords), 1.0)
        ix, iy = P[:, 0].astype(int), P[:, 1].astype(int)
        if (ix < 0).any() or (iy < 0).any() or (ix >= W).any() or (iy >= H).any():
            continue
        if mask[iy, ix].mean() < 0.97:
            continue  # the centreline is not ink all round: not a complete ring
        d = dt_ink[iy, ix]
        if not (0.8 * (w / 2) <= np.median(d) <= 1.25 * (w / 2)) or np.percentile(d, 10) < 0.6 * (w / 2):
            continue  # the ring is not one stroke thick all round
        out.append(dict(P=P, p=np.array([cx, cy, 0.0, 0.0, 0.0]), E=P, tree=cKDTree(P), circle=False, kind="free", resid=0.0))
    return out
