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
    for i, j in _runs(off):
        if (j - i + 1) < MIN_KEEP * w:
            continue
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
        if len(ons) == 1 and pieces[ons[0]][1]:
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
        extra.append((_bez_on_oval(ring, 0.0, 2 * np.pi, True), True, wp))
    return pieces, extra
