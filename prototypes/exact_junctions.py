"""Ends that meet exactly (slice 3 of "shapes first").

A stroke that runs into another stroke or into a ring should end exactly on that stroke's centreline: then the anchors coincide,
the line can be joined in the design program, and widening the stroke leaves no gap or overshoot. The traced ends miss by a
fraction of a stroke width (median 0.26 over the test set: 219 of 650 path ends). Here every open end that nearly meets another
path, with ink all the way between, is extended along its own direction (or trimmed back, when it overshoots) to the point where
it crosses that path's centreline.

snap_ends(paths, w, mask) -> paths (changed in place)"""
import numpy as np
from scipy import ndimage

REACH_OUT = 1.0     # an end may be extended this far (stroke widths) ...
REACH_BACK = 0.8    # ... or trimmed this far
ATTACHED = 0.12     # closer than this to another path counts as attached already
MIN_SIN = 0.42      # the end must cross the other path at 25 degrees or more (sine of the angle)
ENDS_SKIP = 0.5     # a hit this close to the end of the other path is a corner or continuation, not a T junction
MIN_LEFT = 2.0      # a path is never trimmed to less than this length (stroke widths)
TANGENT_GAP = 0.3   # a tangential contact (no real crossing) is closed up when the gap is at most this


def _dense(bez, n=32):
    u = np.linspace(0, 1, n, endpoint=False)[:, None]
    pts = np.vstack([(1 - u) ** 3 * c[0] + 3 * (1 - u) ** 2 * u * c[1] + 3 * (1 - u) * u ** 2 * c[2] + u ** 3 * c[3] for c in bez])
    return np.vstack([pts, bez[-1][3]])


def _split(c, t):
    """De Casteljau: the part of cubic c from 0 to t."""
    a, b, d, e = c
    ab, bc, cd = a + (b - a) * t, b + (d - b) * t, d + (e - d) * t
    abc, bcd = ab + (bc - ab) * t, bc + (cd - bc) * t
    return [a, ab, abc, abc + (bcd - abc) * t]


def _rev(bz):
    return [[c[3], c[2], c[1], c[0]] for c in bz[::-1]]


def _end_info(p, which):
    bez = p["bez"]
    c = bez[0] if which == 0 else bez[-1]
    if which == 0:
        q, h = c[0], c[0] - c[1]
        if np.linalg.norm(h) < 1e-9:
            h = c[0] - c[2] if np.linalg.norm(c[0] - c[2]) > 1e-9 else c[0] - c[3]
    else:
        q, h = c[3], c[3] - c[2]
        if np.linalg.norm(h) < 1e-9:
            h = c[3] - c[1] if np.linalg.norm(c[3] - c[1]) > 1e-9 else c[3] - c[0]
    n = np.linalg.norm(h)
    return q, (h / n if n > 1e-9 else None)


def snap_ends(paths, w, mask):
    ink = ndimage.binary_dilation(mask, iterations=2)
    Hh, Ww = mask.shape

    def through_ink(a, b):
        n = max(int(np.linalg.norm(b - a) / 1.5), 2)
        pts = a + (b - a) * np.linspace(0, 1, n)[:, None]
        return ink[np.clip(pts[:, 1].astype(int), 0, Hh - 1), np.clip(pts[:, 0].astype(int), 0, Ww - 1)].mean() >= 0.97

    dense = [_dense(p["bez"]) if p["bez"] else np.zeros((0, 2)) for p in paths]
    # all polyline segments of all paths: A, B, path id, arclength position of A along its path
    segA, segB, sid, spos = [], [], [], []
    for i, D in enumerate(dense):
        if len(D) < 2:
            continue
        s = np.r_[0, np.cumsum(np.hypot(*np.diff(D, axis=0).T))]
        segA.append(D[:-1])
        segB.append(D[1:])
        sid.append(np.full(len(D) - 1, i))
        spos.append(s[:-1])
    if not segA:
        return paths
    A, B, SID, SPOS = np.vstack(segA), np.vstack(segB), np.concatenate(sid), np.concatenate(spos)
    V = B - A
    total = {i: float(np.hypot(*np.diff(D, axis=0).T).sum()) for i, D in enumerate(dense) if len(D) > 1}
    todo = []
    for i, p in enumerate(paths):
        if p["closed"] or not p["bez"]:
            continue
        for which in (0, 1):
            todo.append((i, which))
    for i, which in todo:
        p = paths[i]
        q, t = _end_info(p, which)
        if t is None:
            continue
        # distance from q to every other path (own path only away from this end)
        f_ = np.clip(np.einsum("ij,ij->i", q - A, V) / np.maximum(np.einsum("ij,ij->i", V, V), 1e-12), 0, 1)
        d_to = np.linalg.norm(A + V * f_[:, None] - q, axis=1)
        own = SID == i
        if own.any():
            end_pos = 0.0 if which == 0 else total[i]
            far = np.abs(SPOS - end_pos) > 3 * w
            d_to = np.where(own & ~far, 1e9, d_to)
        if d_to.min() < ATTACHED * w or d_to.min() > 1.2 * w:
            continue  # attached already, or free
        # where the end's own line crosses the other paths
        cross_v = V[:, 0] * t[1] - V[:, 1] * t[0]  # V x t
        with np.errstate(divide="ignore", invalid="ignore"):
            u = ((q - A)[:, 0] * t[1] - (q - A)[:, 1] * t[0]) / cross_v
            s = ((q - A)[:, 0] * V[:, 1] - (q - A)[:, 1] * V[:, 0]) / cross_v
        sin = np.abs(cross_v) / np.maximum(np.linalg.norm(V, axis=1), 1e-12)
        ok = np.isfinite(u) & np.isfinite(s) & (u >= 0) & (u <= 1) & (sin >= MIN_SIN) & (s >= -REACH_BACK * w) & (s <= REACH_OUT * w) & (d_to < 1.2 * w)
        if own.any():
            ok &= ~(own & ~far)
        X = None
        if ok.any():
            k = np.nonzero(ok)[0][int(np.argmin(np.abs(s[ok])))]
            X = q + t * s[k]
            j = SID[k]
            near_end = (not paths[j]["closed"]) and (SPOS[k] < ENDS_SKIP * w or total[j] - SPOS[k] < ENDS_SKIP * w)
            if near_end or not through_ink(q, X):
                X = None  # an end meeting another path's end is a corner or a continuation: left to the corner logic
        if X is None:
            # tangential contact: close a small gap straight onto the nearest point of the other path
            k = int(np.argmin(d_to))
            if d_to[k] <= TANGENT_GAP * w:
                f = np.clip(np.dot(q - A[k], V[k]) / max(np.dot(V[k], V[k]), 1e-12), 0, 1)
                Y = A[k] + V[k] * f
                if through_ink(q, Y):
                    X = Y
        if X is None:
            continue
        move = X - q
        if np.linalg.norm(move) < 1e-6:
            continue
        bz = [[np.array(x, float) for x in c] for c in p["bez"]]
        if which == 0:
            bz = _rev(bz)  # work on the end of the list either way
        along = float(move @ t)
        if along < -0.02 * w and total[i] + along < MIN_LEFT * w:
            continue  # trimming would leave a stub (a crown tab on a watch case): the overshoot is part of the drawing
        if along < -0.02 * w:
            # an overshoot: cut the path back (drop whole pieces, then split the one that holds the point)
            left, kept = abs(along), None
            for idx in range(len(bz) - 1, -1, -1):
                D = _dense([bz[idx]], 24)
                step = np.hypot(*np.diff(D, axis=0).T)
                L = float(step.sum())
                if left >= L and idx > 0:
                    left -= L
                    continue
                cum = np.r_[0, np.cumsum(step)]
                target = max(L - left, 0.0)
                j = int(np.clip(np.searchsorted(cum, target), 1, len(cum) - 1))
                frac = (target - cum[j - 1]) / max(cum[j] - cum[j - 1], 1e-12)
                tt = ((j - 1) + frac) / (len(cum) - 1)
                kept = bz[:idx] + [_split(bz[idx], float(np.clip(tt, 0.02, 1.0)))]
                break
            if kept is None:
                continue
            bz = kept
        c = bz[-1]
        shift = X - c[3]
        c[3] = X
        c[2] = c[2] + shift
        if which == 0:
            bz = _rev(bz)
        p["bez"] = bz
    return paths
