"""Method C: raster skeleton -> junction graph -> smooth-continuation pairing at crossings -> corner split -> Bezier fit.

Input: any SVG with a viewBox (filled shapes). Output (run()): dict with Bezier paths (pixel space), dots, stroke width, scale.
"""
import base64
import io
import re

import numpy as np
import resvg_py
from PIL import Image
from scipy import ndimage
from scipy.signal import argrelextrema
from scipy.spatial import Voronoi, cKDTree
from skimage import measure
from skimage.morphology import skeletonize

from exact_ovals import find_ovals, apply_ovals, snap_ellipses, join_lines_ovals
from exact_snap import min_bend_radius, snap_chain, merge_lines, sharpen_bends, snap_circles, arc_of_chain, arc_beziers
from exact_edges import outline_lines, merge_collinear, pair_edges
from exact_shapes import find_chain_ovals
from exact_rings import find_rings, find_rrects, find_free_rings, complete_rings
from exact_rings import _cut as ring_cut
from exact_junctions import snap_ends
from exact_arcs import arcs_of, corner_points, corner_lines, exact_circles, circle_lines

# Corner rule: a bend whose centreline radius is <= CORNER_R x stroke width is a sharp corner (a round join of
# a sharp path); a larger radius is kept as a curve. Measured on the icons, joins sit at 0.3-0.7, true curves at 0.95+.
CORNER_R = 0.8
STRAIGHT_SIDE_DEG = 8.0  # a short bowed piece between two lines counts as a straight side only if both lines continue its direction within this many degrees
SPUR_MAX = 0.92  # a short branch pointing out of a corner is a corner artefact only below this many stroke widths (measured: artefacts reach 0.85, real arms such as sparkle points are 1.0 or more)
FILLET_MAX_DEG = 35.0  # two straight pieces meeting at up to this angle, with a measured bend radius, are joined by a small curve instead of a corner
FILLET_MIN_R = 1.2  # ... when the bend's radius is at least this many stroke widths (the corner rule is 0.8)
FILLET_MIN_LEN = 0.6  # the rounded join must start at least this many stroke widths from the corner point
RING_COVERED = 0.85  # a round ring counts as already drawn when this share of its circumference lies on a path
MAX_DISC_ARMS = 1  # a thick round blob is a dot only with at most this many strokes leaving it (a disc on a stem is one)
DOT_MAX = 1.75 # solid marks smaller than this many stroke widths (longest side) become dots
NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# ---------------------------------------------------------------- rendering
def viewbox(path):
    """The drawing's coordinate box. Uses viewBox; falls back to width/height (Illustrator and Affinity both write viewBox,
    but a hand-made file may not)."""
    text = open(path, encoding="utf-8", errors="ignore").read()
    m = re.search(r'viewBox\s*=\s*"([^"]+)"', text)
    if m:
        x0, y0, w, h = map(float, m.group(1).replace(",", " ").split())
        return x0, y0, w, h
    wm = re.search(r'<svg[^>]*?\swidth\s*=\s*"([\d.]+)', text)
    hm = re.search(r'<svg[^>]*?\sheight\s*=\s*"([\d.]+)', text)
    if wm and hm:
        return 0.0, 0.0, float(wm.group(1)), float(hm.group(1))
    raise ValueError("this SVG has neither a viewBox nor a width and height, so its size is unknown")


def render_mask(path, vb, S):
    png = bytes(resvg_py.svg_to_bytes(svg_path=str(path), width=int(round(vb[2] * S))))
    img = Image.open(io.BytesIO(png)).convert("RGBA")
    comp = Image.alpha_composite(Image.new("RGBA", img.size, (255, 255, 255, 255)), img).convert("L")
    return np.array(comp) < 160, img.size[0] / vb[2]


# ---------------------------------------------------------------- skeleton graph
def trace_graph(sk):
    pts = set(zip(*np.nonzero(sk)))
    deg = {p: sum((p[0] + dy, p[1] + dx) in pts for dy, dx in NB) for p in pts}
    junc = np.zeros(sk.shape, bool)
    for p, d in deg.items():
        if d >= 3:
            junc[p] = True
    lab, nlab = ndimage.label(junc, structure=np.ones((3, 3)))
    node_of, pos, is_junc = {}, {}, {}
    for i in range(1, nlab + 1):
        ys, xs = np.nonzero(lab == i)
        pos[i], is_junc[i] = np.array([ys.mean(), xs.mean()]), True
        for y, x in zip(ys, xs):
            node_of[(y, x)] = i
    nid = nlab
    for p, d in deg.items():
        if d == 1:
            nid += 1
            node_of[p], pos[nid], is_junc[nid] = nid, np.array(p, float), False
    visited, edges, seen_keys = set(), [], set()
    for p, n0 in list(node_of.items()):
        for dy, dx in NB:
            q = (p[0] + dy, p[1] + dx)
            if q not in pts or q in visited or (q in node_of and node_of[q] == n0):
                continue
            path, cur, prev, n1 = [], q, p, None
            while True:
                if cur in node_of:
                    n1 = node_of[cur]
                    break
                visited.add(cur)
                path.append(cur)
                nb = [(cur[0] + a, cur[1] + b) for a, b in NB]
                nb = [n for n in nb if n in pts and n != prev and n not in visited and n not in path[-3:-1]]
                nb = [n for n in nb if n in node_of] or nb
                if not nb:
                    break
                prev, cur = cur, nb[0]
            if n1 is None:
                continue
            if not path:
                key = frozenset((n0, n1))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
            edges.append([n0, n1, path])
    loops = []
    for p in pts:
        if p in visited or p in node_of:
            continue
        loop, cur = [], p
        while cur is not None and cur not in visited:
            visited.add(cur)
            loop.append(cur)
            nb = [(cur[0] + a, cur[1] + b) for a, b in NB if (cur[0] + a, cur[1] + b) in pts and (cur[0] + a, cur[1] + b) not in visited]
            cur = nb[0] if nb else None
        if len(loop) > 8:
            loops.append(np.array(loop + [loop[0]], float))
    return pos, is_junc, edges, loops


def _out_dir(E, e, node, w_px, pos):
    """Unit direction in which edge e leaves `node` (looking ~0.4-1.2 stroke widths along it)."""
    n0, n1, path = E[e]
    pts = [pos[n0]] + [np.asarray(p, float) for p in path] + [pos[n1]]
    if n0 != node:
        pts = pts[::-1]
    pts = np.array(pts)
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    if seg[-1] < 1e-6:
        return None
    i0 = int(np.searchsorted(seg, min(0.4 * w_px, seg[-1] * 0.3)))
    i1 = int(np.clip(np.searchsorted(seg, min(1.2 * w_px, seg[-1])), i0 + 1, len(pts) - 1))
    v = pts[i1] - pts[i0]
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


def is_bisector_spur(E, inc, pos, junction, branch, tip, w_px):
    """A short free branch pointing along the bisector of two other branches at its junction is the axis growing toward the
    centre of a round (or sharp) corner, not a stroke. Real short arms point straight on or sideways, never between two."""
    others = [e for e in inc[junction] if e not in branch]
    if len(others) < 2:
        return False
    d_tip = _out_dir(E, branch[-1], junction, w_px, pos)
    if d_tip is None:
        return False
    dirs = [d for d in (_out_dir(E, e, junction, w_px, pos) for e in others) if d is not None]
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            ang = np.degrees(np.arccos(np.clip(np.dot(dirs[i], dirs[j]), -1, 1)))
            if 40 <= ang <= 150:
                bis = dirs[i] + dirs[j]
                bis /= np.linalg.norm(bis) + 1e-12
                if np.degrees(np.arccos(np.clip(np.dot(-bis, d_tip), -1, 1))) < 25:  # it points away from both sides, into the corner
                    return True
    return False


def simplify_graph(pos, is_junc, edges, w_px, dt, merge_f=1.3, prune_f=1.0):
    """Clean the medial-axis graph. Order matters:
    1) prune short branches that are artefacts (hairs, and corner branches that taper to nothing) - doing this FIRST
       turns the fork at the flat end of a short arm into a plain end, so the arm survives;
    2) then merge junctions that are still junctions and sit close together (a crossing shows up as two nearby junctions)."""
    E = [[n0, n1, path] for n0, n1, path in edges]
    for _ in range(6):
        inc = {}
        for i, (n0, n1, _p) in enumerate(E):
            inc.setdefault(n0, []).append(i)
            inc.setdefault(n1, []).append(i)
        degree = {n: len(v) for n, v in inc.items()}
        remove = set()
        for tip, es in inc.items():
            if len(es) != 1 or es[0] in remove:
                continue
            # walk the whole branch from this free end to the first real junction (through degree-2 nodes)
            branch, cur, length, prev_e = [], tip, 0.0, None
            while True:
                nxt = [i for i in inc[cur] if i != prev_e]
                if len(inc[cur]) != 1 and cur != tip and len(inc[cur]) != 2:
                    break
                if not nxt:
                    break
                e = nxt[0]
                branch.append(e)
                length += len(E[e][2]) + 1
                cur = E[e][1] if E[e][0] == cur else E[e][0]
                prev_e = e
                if len(inc[cur]) != 2:
                    break
            if len(inc.get(cur, [])) < 3:
                continue  # a free stroke, not a branch hanging off a junction
            if length < 0.25 * w_px:
                remove.update(branch)  # a hair a few pixels long is noise (real short arms are longer)
            elif length < SPUR_MAX * w_px and is_bisector_spur(E, inc, pos, cur, branch, tip, w_px):
                remove.update(branch)  # grows toward the centre of a round corner: not a stroke
            elif length < prune_f * w_px:
                ty, tx = pos[tip]
                tip_dt = dt[int(np.clip(ty, 0, dt.shape[0] - 1)), int(np.clip(tx, 0, dt.shape[1] - 1))]
                if tip_dt < 0.65 * w_px / 2:
                    remove.update(branch)  # a corner artefact tapers to nothing at its tip; a real short arm stays full width
        if not remove:
            break
        E = [e for i, e in enumerate(E) if i not in remove]
    degree = {}
    for n0, n1, _p in E:
        degree[n0] = degree.get(n0, 0) + 1
        degree[n1] = degree.get(n1, 0) + 1
    junc = {n: degree.get(n, 0) >= 3 for n in pos}
    parent = {n: n for n in pos}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    contracted = set()
    for i, (n0, n1, path) in enumerate(E):
        if n0 != n1 and junc[n0] and junc[n1] and len(path) + 1 < merge_f * w_px:
            parent[find(n0)] = find(n1)
            contracted.add(i)
    members = {}
    for n in pos:
        members.setdefault(find(n), []).append(n)
    newpos = {r: np.mean([pos[m] for m in ms], axis=0) for r, ms in members.items()}
    E = [[find(n0), find(n1), path] for i, (n0, n1, path) in enumerate(E) if i not in contracted]
    out = [(n0, n1, np.vstack([newpos[n0], np.array(path, float).reshape(-1, 2), newpos[n1]])) for n0, n1, path in E]
    return newpos, out


def end_dir(pts, side, w_px):
    """Direction a branch leaves its junction. Uses the stretch 0.8-2.2 stroke widths away, because the skeleton is
    bent within ~0.8 stroke widths of the junction. Short branches fall back to the whole stretch."""
    p = pts if side == 0 else pts[::-1]
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    if seg[-1] >= 1.6 * w_px:
        i0 = int(np.searchsorted(seg, 0.8 * w_px))
        i1 = int(np.clip(np.searchsorted(seg, min(2.2 * w_px, seg[-1] * 0.95)), i0 + 1, len(p) - 1))
    else:
        i0, i1 = 0, int(np.clip(np.searchsorted(seg, seg[-1] * 0.7), 1, len(p) - 1))
    v = p[i1] - p[i0]
    return v / (np.linalg.norm(v) + 1e-9)


def end_far(pts, side, w_px):
    p = pts if side == 0 else pts[::-1]
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    return p[int(np.clip(np.searchsorted(seg, min(2.0 * w_px, seg[-1] * 0.9)), 1, len(p) - 1))]


def pair_and_chain(newpos, edges, w_px, join_deg=45):
    """At each node, link edge-ends that continue smoothly (angle near 180). Returns list of (polyline (y,x), closed)."""
    at = {}
    for i, (n0, n1, pts) in enumerate(edges):
        at.setdefault(n0, []).append((i, 0))
        at.setdefault(n1, []).append((i, 1))
    link, link_dev = {}, {}

    def make_link(ea, eb, dev):
        link[ea], link[eb] = eb, ea
        link_dev[ea] = link_dev[eb] = dev

    for n, ends in at.items():
        if len(ends) == 2:
            da, db = (end_dir(edges[e[0]][2], e[1], w_px) for e in ends)
            make_link(ends[0], ends[1], float(np.degrees(np.arccos(np.clip(-np.dot(da, db), -1, 1)))))
        elif len(ends) >= 3:
            d = {e: end_dir(edges[e[0]][2], e[1], w_px) for e in ends}
            far = {e: end_far(edges[e[0]][2], e[1], w_px) for e in ends}
            cand = []
            for a in range(len(ends)):
                for b in range(a + 1, len(ends)):
                    cosv = float(np.clip(np.dot(d[ends[a]], d[ends[b]]), -1, 1))
                    dev = float(np.degrees(np.arccos(-cosv)))  # 0 = perfectly straight through
                    da = d[ends[a]]
                    off = abs(da[0] * (far[ends[b]] - far[ends[a]])[1] - da[1] * (far[ends[b]] - far[ends[a]])[0])  # sideways gap
                    if dev <= join_deg and off > 0.4 * w_px:
                        continue  # same direction but offset sideways: two different strokes, not one going straight through
                    cand.append((dev, ends[a], ends[b]))
            used = set()
            for dev, ea, eb in sorted(cand):  # 1) branches that continue smoothly (T, X crossings)
                if dev <= join_deg and ea not in used and eb not in used:
                    make_link(ea, eb, dev)
                    used |= {ea, eb}
            for dev, ea, eb in sorted(cand):  # 2) nothing straight left: the two branches that form the gentlest bend (V, L)
                if dev <= 110 and ea not in used and eb not in used:
                    make_link(ea, eb, dev)
                    used |= {ea, eb}
    visited, chains = set(), []
    for e0 in range(len(edges)):
        if e0 in visited:
            continue
        e, s, seen = e0, 0, {e0}
        while (e, s) in link:
            e2, s2 = link[(e, s)]
            if e2 in seen:
                break
            seen.add(e2)
            e, s = e2, 1 - s2
        seq, closed, cur_e, s_in = [], False, e, s
        while True:
            visited.add(cur_e)
            seq.append((cur_e, s_in))
            nxt = link.get((cur_e, 1 - s_in))
            if nxt is None:
                break
            if nxt[0] in visited:
                closed = nxt[0] == seq[0][0] and nxt[1] == seq[0][1]
                break
            cur_e, s_in = nxt
        parts, bend_nodes = [], []
        for k, (ei, si) in enumerate(seq):
            p = edges[ei][2] if si == 0 else edges[ei][2][::-1]
            n_in, n_out = edges[ei][si], edges[ei][1 - si]
            if k > 0 and (ei, si) in link and len(at[n_in]) >= 3 and link_dev[(ei, si)] > 20:
                bend_nodes.append(newpos[n_in])
            # through a crossing (3+ branches): drop the messy pixels near the junction and let the fit bridge the gap
            if (k > 0 or closed) and (ei, si) in link and len(at[n_in]) >= 3 and link_dev[(ei, si)] <= 20:
                i = 0
                while i < len(p) - 2 and np.linalg.norm(p[i] - newpos[n_in]) < 0.6 * w_px:
                    i += 1
                p = p[i:]
            if (k < len(seq) - 1 or closed) and (ei, 1 - si) in link and len(at[n_out]) >= 3 and link_dev[(ei, 1 - si)] <= 20:
                j = len(p)
                while j > 2 and np.linalg.norm(p[j - 1] - newpos[n_out]) < 0.6 * w_px:
                    j -= 1
                p = p[:j]
            parts.append(p)
        P = np.vstack(parts)
        keep = np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-6]
        P = P[keep]
        if closed and np.allclose(P[0], P[-1]):
            P = P[:-1]
        chains.append((P, closed, bend_nodes))
    return chains


# ---------------------------------------------------------------- Bezier fitting (Schneider)
def _bez(c, u):
    u = u[:, None]
    return (1 - u) ** 3 * c[0] + 3 * u * (1 - u) ** 2 * c[1] + 3 * u ** 2 * (1 - u) * c[2] + u ** 3 * c[3]


def _gen(P, u, t1, t2):
    b0, b1, b2, b3 = (1 - u) ** 3, 3 * u * (1 - u) ** 2, 3 * u ** 2 * (1 - u), u ** 3
    A1, A2 = t1 * b1[:, None], t2 * b2[:, None]
    C = np.array([[np.sum(A1 * A1), np.sum(A1 * A2)], [np.sum(A1 * A2), np.sum(A2 * A2)]])
    tmp = P - (np.outer(b0 + b1, P[0]) + np.outer(b2 + b3, P[-1]))
    X = np.array([np.sum(A1 * tmp), np.sum(A2 * tmp)])
    det = C[0, 0] * C[1, 1] - C[0, 1] ** 2
    seg = np.linalg.norm(P[-1] - P[0])
    al = ar = seg / 3
    if abs(det) > 1e-12:
        al_, ar_ = (X[0] * C[1, 1] - X[1] * C[0, 1]) / det, (C[0, 0] * X[1] - C[0, 1] * X[0]) / det
        if al_ > 1e-6 * seg and ar_ > 1e-6 * seg:
            al, ar = al_, ar_
    return np.array([P[0], P[0] + t1 * al, P[-1] + t2 * ar, P[-1]])


def _reparam(P, u, c):
    d1 = 3 * ((c[1] - c[0]) * ((1 - u) ** 2)[:, None] + 2 * (c[2] - c[1]) * (u * (1 - u))[:, None] + (c[3] - c[2]) * (u ** 2)[:, None])
    d2 = 6 * ((c[2] - 2 * c[1] + c[0]) * (1 - u)[:, None] + (c[3] - 2 * c[2] + c[1]) * u[:, None])
    d = _bez(c, u) - P
    num = np.sum(d * d1, axis=1)
    den = np.sum(d1 * d1, axis=1) + np.sum(d * d2, axis=1)
    with np.errstate(all="ignore"):
        step = np.where(np.abs(den) > 1e-12, num / den, 0)
    return np.clip(u - np.nan_to_num(step), 0, 1)


def is_straight(P, lim):
    a, d = P[0], P[-1] - P[0]
    L = np.linalg.norm(d)
    if L < 1e-9:
        return False
    return np.abs(d[0] * (P[:, 1] - a[1]) - d[1] * (P[:, 0] - a[0])).max() / L < lim


def merge_run(segs, err2):
    """Join neighbouring cubics when one cubic fits both within tolerance (fewer anchors)."""
    i = 0
    while i < len(segs) - 1:
        a, b = segs[i], segs[i + 1]
        t = np.linspace(0, 1, 24)
        P = np.vstack([_bez(a, t), _bez(b, t)[1:]])
        t1, t2 = a[1] - a[0], b[2] - b[3]
        if np.linalg.norm(t1) < 1e-9 or np.linalg.norm(t2) < 1e-9:
            t1, t2 = P[-1] - P[0], P[0] - P[-1]
        r = fit_cubic(P, t1 / np.linalg.norm(t1), t2 / np.linalg.norm(t2), err2 * 0.8)
        if len(r) == 1:
            segs[i:i + 2] = r
            i = max(0, i - 1)
        else:
            i += 1
    return segs


def fit_cubic(P, t1, t2, err2, depth=0):
    n = len(P)
    if n >= 2 and is_straight(P, 0.8 * np.sqrt(err2)):
        d = P[-1] - P[0]
        return [np.array([P[0], P[0] + d / 3, P[0] + 2 * d / 3, P[-1]])]
    if n == 2:
        d = np.linalg.norm(P[1] - P[0]) / 3
        return [np.array([P[0], P[0] + t1 * d, P[1] + t2 * d, P[1]])]
    u = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    u = u / u[-1] if u[-1] > 0 else np.linspace(0, 1, n)
    c = _gen(P, u, t1, t2)
    for _ in range(5):
        dist = np.sum((_bez(c, u) - P) ** 2, axis=1)
        k = int(np.argmax(dist[1:-1])) + 1
        if dist[k] < err2:
            return [c]
        u = _reparam(P, u, c)
        c = _gen(P, u, t1, t2)
    dist = np.sum((_bez(c, u) - P) ** 2, axis=1)
    k = int(np.argmax(dist[1:-1])) + 1
    if dist[k] < err2 or depth > 12 or n < 4:
        return [c]
    tc = P[k - 1] - P[k + 1]
    tc = tc / (np.linalg.norm(tc) + 1e-12)
    return fit_cubic(P[:k + 1], t1, tc, err2, depth + 1) + fit_cubic(P[k:], -tc, t2, err2, depth + 1)


def find_corners(P, k, thr, closed, gap=None):
    N = len(P)
    if N < 2 * k + 2:
        return []
    idx = np.arange(N)
    if closed:
        a, b = P[(idx - k) % N], P[(idx + k) % N]
    else:
        a, b = P[np.clip(idx - k, 0, N - 1)], P[np.clip(idx + k, 0, N - 1)]
    v1, v2 = P - a, b - P
    cosv = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-12)
    ang = np.degrees(np.arccos(np.clip(cosv, -1, 1)))
    if not closed:
        ang[:k], ang[N - k:] = 0, 0
    out = []
    for i in np.nonzero(ang > thr)[0]:
        win = [(i + j) % N for j in range(-k, k + 1)] if closed else [j for j in range(max(0, i - k), min(N, i + k + 1))]
        if ang[i] >= ang[win].max() and not any(abs(i - o) <= k for o in out):
            out.append(int(i))
    if gap and len(out) > 1:  # corners closer than `gap` along the path are one corner, not two
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        groups = [[out[0]]]
        for c in out[1:]:
            if s[c] - s[groups[-1][-1]] <= gap:
                groups[-1].append(c)
            else:
                groups.append([c])
        if closed and len(groups) > 1 and s[-1] - s[groups[-1][-1]] + s[groups[0][0]] <= gap:
            groups[0] = groups.pop() + groups[0]
        out = sorted(g[len(g) // 2] for g in groups)
    return out


def one_sided(Pseg, L, end):
    p = Pseg if end == 0 else Pseg[::-1]
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    j = int(np.clip(np.searchsorted(seg, min(L, seg[-1] * 0.5)), 1, len(p) - 1))
    v = p[j] - p[0]
    return v / (np.linalg.norm(v) + 1e-12)


def smooth(Pseg, sigma, closed=False):
    if len(Pseg) < 5:
        return Pseg
    S = ndimage.gaussian_filter1d(Pseg, sigma, axis=0, mode="wrap" if closed else "nearest")
    if not closed:
        S[0], S[-1] = Pseg[0], Pseg[-1]
    return S


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12), -1, 1))))


# ---------------------------------------------------------------- extrema (circles get 4 anchors)
def find_extrema(S, w_px, closed):
    """Points where the curve is furthest left/right/up/down: standard good anchor positions. Returns sorted [(index, axis)]
    axis 0 = x extreme (vertical tangent), axis 1 = y extreme (horizontal tangent)."""
    N = len(S)
    m = max(4, int(round(2.0 * w_px)))
    thr = 0.15 * w_px
    if N < 2 * m + 3:
        return []
    mode = "wrap" if closed else "clip"
    out = []
    for axis in (0, 1):
        c = S[:, axis]
        for is_max in (True, False):
            idx = argrelextrema(c, np.greater_equal if is_max else np.less_equal, order=m, mode=mode)[0]
            groups = []
            for i in idx:
                if groups and i - groups[-1][-1] <= 2:
                    groups[-1].append(i)
                else:
                    groups.append([i])
            for g in groups:
                i = g[len(g) // 2]
                if not closed and (i < m or i > N - 1 - m):
                    continue
                a, b = c[(i - m) % N], c[(i + m) % N]
                prom = c[i] - max(a, b) if is_max else min(a, b) - c[i]
                if prom > thr:
                    out.append((int(i), axis))
    out.sort()
    keep = []
    for i, ax in out:
        if not keep or i - keep[-1][0] > m // 2:
            keep.append((i, ax))
    return keep


def axis_tangent(S, i, axis, L, closed=False):
    N = len(S)
    if closed:
        d = S[(i + L) % N] - S[(i - L) % N]
    else:
        d = S[min(N - 1, i + L)] - S[max(0, i - L)]
    v = np.zeros(2)
    comp = d[1 - axis]
    if abs(comp) < 1e-9:
        return None
    v[1 - axis] = np.sign(comp)
    return v


def fit_pieces(S, cuts, kinds, t1, t2, w_px, err2):
    """Fit one cubic run between each pair of consecutive cuts. kinds[i] is a forward tangent (extremum) or None (corner/free)."""
    L = max(2, int(0.75 * w_px))
    segs = []
    pts = [0] + list(cuts) + [len(S) - 1]
    for a, b in zip(pts[:-1], pts[1:]):
        if b - a < 1:
            continue
        seg = S[a:b + 1]
        if a == 0:
            ta = t1 if t1 is not None else one_sided(seg, L, 0)
        else:
            ta = kinds[a] if kinds.get(a) is not None else one_sided(seg, L, 0)
        if b == len(S) - 1:
            tb = t2 if t2 is not None else one_sided(seg, L, 1)
        else:
            tb = -kinds[b] if kinds.get(b) is not None else one_sided(seg, L, 1)
        segs += merge_run(fit_cubic(seg, ta, tb, err2), err2)
    return segs


def fit_curve_piece(Pc, w_px, tol, corner_deg, t1=None, t2=None):
    """Open curved run: split at real corners and at x/y extremes, fit cubics."""
    if len(Pc) < 2:
        return []
    k = max(3, int(round(0.35 * w_px)))
    err2 = (tol * w_px) ** 2
    sigma = max(1.0, 0.1 * w_px)
    S = smooth(Pc, sigma)
    L = max(2, int(0.75 * w_px))
    # a short run that stays within 10% of a stroke width of its own chord is a straight side that the skeleton bowed
    chord = S[-1] - S[0]
    clen = float(np.linalg.norm(chord))
    if 0 < clen <= 3.5 * w_px:
        dev = np.abs(chord[0] * (S[:, 1] - S[0][1]) - chord[1] * (S[:, 0] - S[0][0])).max() / clen
        # not when the neighbouring lines point clearly away from the chord: then this is the bending middle of an S (a straight
        # chord would leave a kink at both ends); the 0.10 limit alone flipped with the pixel position
        continues = (t1 is None or angle_deg(t1, chord) <= STRAIGHT_SIDE_DEG) and (t2 is None or angle_deg(t2, -chord) <= STRAIGHT_SIDE_DEG)
        if dev <= 0.10 * w_px and continues:
            return [np.array([S[0], S[0] + chord / 3, S[0] + 2 * chord / 3, S[-1]])]
    if t1 is not None and angle_deg(t1, one_sided(S, L, 0)) > 55:
        t1 = None
    if t2 is not None and angle_deg(t2, one_sided(S, L, 1)) > 55:
        t2 = None
    corners = find_corners(S, k, corner_deg, False, 0.8 * w_px)
    kinds = {}
    for i, ax in find_extrema(S, w_px, False):
        if all(abs(i - c) > k for c in corners):
            t = axis_tangent(S, i, ax, L)
            if t is not None:
                kinds[i] = t
    cuts = sorted(set(corners) | set(kinds))
    for c in corners:
        kinds.setdefault(c, None)
    return fit_pieces(S, cuts, kinds, t1, t2, w_px, err2)


def fit_curve_closed(P, w_px, tol, corner_deg):
    """Closed curve with no straight parts (circles, ellipses, blobs)."""
    k = max(3, int(round(0.35 * w_px)))
    err2 = (tol * w_px) ** 2
    L = max(2, int(0.75 * w_px))
    S = smooth(P, max(1.0, 0.1 * w_px), closed=True)
    corners = find_corners(S, k, corner_deg, True, 0.8 * w_px)
    if corners:
        s = corners[0]
        Q = np.roll(S, -s, axis=0)
        return fit_curve_piece(np.vstack([Q, Q[:1]]), w_px, tol, corner_deg)
    ext = find_extrema(S, w_px, True)
    if len(ext) >= 2:
        s, ax0 = ext[0]
        t0 = axis_tangent(S, s, ax0, L, True)
        Q = np.roll(S, -s, axis=0)
        Q = np.vstack([Q, Q[:1]])
        kinds = {}
        for i, ax in ext[1:]:
            j = (i - s) % len(S)
            t = axis_tangent(S, i, ax, L, True)
            if t is not None:
                kinds[j] = t
        if t0 is not None:
            return fit_pieces(Q, sorted(kinds), kinds, t0, -t0, w_px, err2)
    h = len(S) // 2
    tc0 = S[min(len(S) - 1, L)] - S[-L]
    tc0 /= np.linalg.norm(tc0) + 1e-12
    tc1 = S[(h + L) % len(S)] - S[(h - L) % len(S)]
    tc1 /= np.linalg.norm(tc1) + 1e-12
    a, b = S[:h + 1], np.vstack([S[h:], S[:1]])
    return fit_cubic(a, tc0, -tc1, err2) + fit_cubic(b, tc1, -tc0, err2)


# ---------------------------------------------------------------- straight lines and corners
def rdp_idx(P, eps):
    N = len(P)
    keep = np.zeros(N, bool)
    keep[0] = keep[-1] = True
    stack = [(0, N - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = P[a + 1:b]
        d = P[b] - P[a]
        L = np.linalg.norm(d)
        dist = np.linalg.norm(seg - P[a], axis=1) if L < 1e-9 else np.abs(d[0] * (seg[:, 1] - P[a][1]) - d[1] * (seg[:, 0] - P[a][0])) / L
        k = int(np.argmax(dist))
        if dist[k] > eps:
            m = a + 1 + k
            keep[m] = True
            stack += [(a, m), (m, b)]
    return np.nonzero(keep)[0]


def drop_redundant(P, idx, eps):
    """rdp_idx splits top-down, so it can leave a vertex in the middle of a straight run (it was the farthest point when
    the chain still had a corner elsewhere). Drop such a vertex when the straight run it sits in is bounded by a real corner
    (35 degrees or more) at both ends and the chord passes within eps of every point of it: a straight side between two corners."""
    idx = list(idx)
    k = 1
    while k < len(idx) - 1:
        if k < 2 or k + 2 > len(idx) - 1 or min(
                angle_deg(P[idx[k - 1]] - P[idx[k - 2]], P[idx[k]] - P[idx[k - 1]]),
                angle_deg(P[idx[k + 1]] - P[idx[k]], P[idx[k + 2]] - P[idx[k + 1]])) < 35:
            k += 1
            continue
        a, b = P[idx[k - 1]], P[idx[k + 1]]
        d = b - a
        L = np.linalg.norm(d)
        seg = P[idx[k - 1]:idx[k + 1] + 1]
        dist = np.linalg.norm(seg - a, axis=1) if L < 1e-9 else np.abs(d[0] * (seg[:, 1] - a[1]) - d[1] * (seg[:, 0] - a[0])) / L
        if dist.max() <= eps:
            del idx[k]
            k = max(k - 1, 1)
        else:
            k += 1
    return np.array(idx)


def tls_line(P, i0, i1, trim):
    seg = P[i0:i1 + 1]
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(seg, axis=0).T))]
    sel = (s >= trim) & (s <= s[-1] - trim)
    q = seg[sel] if sel.sum() >= 4 else seg
    c = q.mean(axis=0)
    d = np.linalg.svd(q - c, full_matrices=False)[2][0]
    if d @ (seg[-1] - seg[0]) < 0:
        d = -d
    return c, d


def proj(it, p):
    return it["c"] + it["d"] * ((p - it["c"]) @ it["d"])


def intersect(a, b):
    cross = a["d"][0] * b["d"][1] - a["d"][1] * b["d"][0]
    if abs(cross) < 0.34:  # lines within ~20 degrees of parallel: no sensible corner
        return None
    diff = b["c"] - a["c"]
    t = (diff[0] * b["d"][1] - diff[1] * b["d"][0]) / cross
    return a["c"] + a["d"] * t


def build_items(P, Lmin, tolr, closed=False):
    """Split a chain into straight 'line' items and 'curve' items.
    A long chord from the polygon approximation is only a real line if it is very long, or it ends at a real corner
    (otherwise it is just part of a big arc, e.g. a large circle)."""
    idx = drop_redundant(P, rdp_idx(P, tolr), tolr)
    m = len(idx) - 1
    chords = [P[idx[t + 1]] - P[idx[t]] for t in range(m)]

    def turn(t, free=0.0):  # turning angle at the start vertex of edge t (`free`: value to use at a free chain end)
        if t == 0 and not closed:
            return free
        return angle_deg(chords[t - 1], chords[t])

    def turn_end(t, free=0.0):  # turning angle at the end vertex of edge t
        if t == m - 1 and not closed:
            return free
        return angle_deg(chords[t], chords[(t + 1) % m])

    items = []
    for t in range(m):
        a, b = int(idx[t]), int(idx[t + 1])
        L = np.linalg.norm(chords[t])
        short_between_corners = L >= Lmin / 2 and turn(t, 90) >= 35 and turn_end(t, 90) >= 35  # short sides of a tab, short bars
        # a short flat run beside a corner: the neighbouring piece must be much shorter (a corner arc), otherwise
        # this is just one step of a big arc
        cl = lambda k: float(np.linalg.norm(chords[k % m]))
        # (a free chain end is not a corner here: otherwise the first step of any open arc would count)
        short_start = (t > 0 or closed) and cl(t - 1) < 0.4 * L
        short_end = (t < m - 1 or closed) and cl(t + 1) < 0.4 * L
        seg_pts = P[a:b + 1]
        dvec = P[b] - P[a]
        very_straight = np.abs(dvec[0] * (seg_pts[:, 1] - P[a][1]) - dvec[1] * (seg_pts[:, 0] - P[a][0])).max() / max(L, 1e-9) <= 0.04 * Lmin / 2
        flat_by_corner = very_straight and L >= 0.6 * Lmin and ((turn(t) >= 12 and short_start) or (turn_end(t) >= 12 and short_end))
        genuine = L >= 2 * Lmin or (L >= Lmin and (turn(t) >= 20 or turn_end(t) >= 20)) or short_between_corners or flat_by_corner
        if genuine:
            items.append(dict(kind="line", i0=a, i1=b))
        elif items and items[-1]["kind"] == "curve":
            items[-1]["i1"] = b
        else:
            items.append(dict(kind="curve", i0=a, i1=b))
    return items


def assemble(P, closed, w_px, tol, corner_deg, corner_r=None, junctions=()):
    """Chain with long straight runs: true lines, corners at line intersections, G1 curves in between."""
    corner_r = CORNER_R if corner_r is None else corner_r
    Lmin, tolr, trim = 2.0 * w_px, 0.06 * w_px, 0.4 * w_px
    items = build_items(P, Lmin, tolr, closed)

    def fit_line(it):
        it["c"], it["d"] = tls_line(P, it["i0"], it["i1"], trim)
        it["s"], it["e"] = proj(it, P[it["i0"]]), proj(it, P[it["i1"]])

    for it in items:
        if it["kind"] == "line":
            fit_line(it)
    j = 0
    while j < len(items) - 1:  # merge neighbouring lines that are really one line
        a, b = items[j], items[j + 1]
        if a["kind"] == b["kind"] == "line" and angle_deg(a["d"], b["d"]) < 8:
            seg = P[a["i0"]:b["i1"] + 1]
            c, d = tls_line(P, a["i0"], b["i1"], trim)
            dev = np.abs(d[0] * (seg[:, 1] - c[1]) - d[1] * (seg[:, 0] - c[0])).max()
            if dev <= 2 * tolr:
                items[j:j + 2] = [dict(kind="line", i0=a["i0"], i1=b["i1"])]
                fit_line(items[j])
                continue
        j += 1
    m = len(items)

    def nb(j, off):
        k = j + off
        if closed:
            return items[k % m] if m > 1 else None
        return items[k] if 0 <= k < m else None

    for j, it in enumerate(items):  # curves squeezed between lines / hooks at free ends
        if it["kind"] != "curve":
            continue
        prev, nxt = nb(j, -1), nb(j, 1)
        lp = prev if prev and prev["kind"] == "line" else None
        ln = nxt if nxt and nxt["kind"] == "line" else None
        seg = P[it["i0"]:it["i1"] + 1]
        arc = float(np.hypot(*np.diff(seg, axis=0).T).sum())
        if lp and ln and lp is not ln:
            X = intersect(lp, ln)
            th = angle_deg(lp["d"], ln["d"])
            if X is not None and 25 <= th <= 150:
                d = np.linalg.norm(seg - X, axis=1).min()  # how far the centerline cuts the corner
                r_eq = d / (1 / np.cos(np.radians(th) / 2) - 1)  # radius of the arc that would cut it that much
                # a branch attached at the bend (V with a stem): the skeleton is pulled round by the junction and can't be
                # measured, and the branch touching the inside means the inner corner is sharp
                at_junction = any(np.linalg.norm(seg - jn, axis=1).min() < w_px and np.linalg.norm(X - jn) <= 2 * w_px for jn in junctions)
                # a bend that the chain itself rounds off with a radius clearly above the corner rule is a real curve (the
                # 'lines' next to it are only the nearly straight middle parts of an S): measured on the chain, not guessed
                rounded = min_bend_radius(seg, w_px) > 1.2 * corner_r * w_px
                if (r_eq <= corner_r * w_px and d <= 1.5 * w_px and not rounded) or (at_junction and arc <= 3 * w_px):
                    lp["e"] = ln["s"] = X
                    it["drop"] = True
        elif not closed and lp and j == m - 1 and arc <= 1.2 * w_px:
            end = P[-1]
            # a tiny end piece (< 0.8 stroke widths) is too short to be a deliberate feature, whatever its tilt
            if arc <= 0.8 * w_px or angle_deg(lp["d"], end - lp["e"]) < 35 or np.linalg.norm(end - lp["e"]) < 0.3 * w_px:
                lp["e"] = proj(lp, end)
                it["drop"] = True
        elif not closed and ln and j == 0 and arc <= 1.2 * w_px:
            st = P[0]
            if arc <= 0.8 * w_px or angle_deg(ln["d"], ln["s"] - st) < 35 or np.linalg.norm(st - ln["s"]) < 0.3 * w_px:
                ln["s"] = proj(ln, st)
                it["drop"] = True
    for j in range(m if closed else m - 1):  # line meets line directly
        a, b = items[j], items[(j + 1) % m]
        if a is b or a["kind"] != "line" or b["kind"] != "line":
            continue
        th = angle_deg(a["d"], b["d"])
        X = intersect(a, b) if th >= 20 else None
        if X is not None and np.linalg.norm(X - P[b["i0"]]) <= 1.0 * w_px:
            a["e"] = b["s"] = X
            if th <= FILLET_MAX_DEG:  # a small-angle bend with a real radius (a tight S-bend), not a corner: round it
                v = b["i0"]
                s_ = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
                j0, j1 = int(np.searchsorted(s_, s_[v] - 1.2 * w_px)), int(np.searchsorted(s_, s_[v] + 1.2 * w_px))
                R = min_bend_radius(P[j0:j1 + 1], w_px) if j1 - j0 >= 6 else 0.0
                if R > FILLET_MIN_R * w_px:
                    R = min(R, 8 * w_px)
                    tl = R * np.tan(np.radians(th) / 2)
                    tl = min(tl, 0.4 * np.linalg.norm(a["e"] - a["s"]), 0.4 * np.linalg.norm(b["e"] - b["s"]))
                    h = 4 / 3 * np.tan(np.radians(th) / 4) * tl / np.tan(np.radians(th) / 2)
                    p1, p2 = X - tl * a["d"], X + tl * b["d"]
                    a["fe"], b["fs"] = p1, p2
                    a["fillet"] = np.array([p1, p1 + h * a["d"], p2 - h * b["d"], p2])
        else:
            a["e"] = b["s"] = (a["e"] + b["s"]) / 2
    segs = []
    for j, it in enumerate(items):
        if it.get("drop"):
            continue
        if it["kind"] == "line":
            s, e = it.get("fs", it["s"]), it.get("fe", it["e"])
            if np.linalg.norm(e - s) > 1e-6:
                segs.append(np.array([s, s + (e - s) / 3, s + 2 * (e - s) / 3, e]))
            if "fillet" in it:
                segs.append(it["fillet"])
        else:
            Pc = P[it["i0"]:it["i1"] + 1].copy()
            prev, nxt = nb(j, -1), nb(j, 1)
            t1 = t2 = None
            if prev and prev["kind"] == "line" and not prev.get("drop"):
                Pc[0], t1 = prev["e"], prev["d"]
            if nxt and nxt["kind"] == "line" and not nxt.get("drop"):
                Pc[-1], t2 = nxt["s"], -nxt["d"]
            segs += fit_curve_piece(Pc, w_px, tol, corner_deg, t1, t2)
    return segs


def collapse_double_corners(segs, closed, w_px):
    """Two corners less than ~half a stroke width apart are really one corner: drop the tiny segment between them and
    put a single corner where the neighbouring segments' tangent lines meet."""
    segs = [np.array(c, float) for c in segs]

    def dir_in(c):
        d = c[3] - c[2]
        return d if np.linalg.norm(d) > 1e-9 else c[3] - c[0]

    def dir_out(c):
        d = c[1] - c[0]
        return d if np.linalg.norm(d) > 1e-9 else c[3] - c[0]

    if not closed:  # a tiny segment at a free end is hidden by the round cap: drop it
        while len(segs) >= 3 and np.linalg.norm(segs[0][3] - segs[0][0]) < 0.5 * w_px:
            segs = segs[1:]
        while len(segs) >= 3 and np.linalg.norm(segs[-1][3] - segs[-1][0]) < 0.5 * w_px:
            segs = segs[:-1]
    changed = True
    while changed and len(segs) >= 3:
        changed = False
        for k in range(len(segs)):
            if not closed and (k == 0 or k == len(segs) - 1):
                continue
            prev, cur, nxt = segs[k - 1], segs[k], segs[(k + 1) % len(segs)]
            chord_len = np.linalg.norm(cur[3] - cur[0])
            if prev is nxt or chord_len >= 0.8 * w_px:
                continue
            A, B = cur[0], cur[3]
            d1, d2 = dir_in(prev), dir_out(nxt)
            corner_a = angle_deg(dir_in(prev), dir_out(cur)) >= 12
            corner_b = angle_deg(dir_in(cur), dir_out(nxt)) >= 12
            # two corners < 0.8 stroke widths apart, or a corner with another anchor < 0.5 apart, or any two anchors
            # < 0.2 apart, are one anchor: a tiny segment carries no design intent
            if not ((corner_a and corner_b) or (chord_len < 0.7 * w_px and (corner_a or corner_b)) or chord_len < 0.2 * w_px):
                continue
            if angle_deg(d1, d2) < 25:  # neighbours almost line up: just one smooth join at the middle
                M = (A + B) / 2
                prev[2] = prev[2] + (M - prev[3])
                prev[3] = M
                nxt[1] = nxt[1] + (M - nxt[0])
                nxt[0] = M
                del segs[k]
                changed = True
                break
            cross = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(cross) < 1e-9:
                continue
            u, v = d1 / np.linalg.norm(d1), d2 / np.linalg.norm(d2)
            cr = u[0] * v[1] - u[1] * v[0]
            t = ((B - A)[0] * v[1] - (B - A)[1] * v[0]) / cr
            X = A + u * t
            if np.linalg.norm(X - (A + B) / 2) > 1.0 * w_px:
                continue
            prev[2] = prev[2] + (X - prev[3])
            prev[3] = X
            nxt[1] = nxt[1] + (X - nxt[0])
            nxt[0] = X
            del segs[k]
            changed = True
            break
    return segs


def align_small_kinks(segs, closed, w_px, lo=12.0, hi=16.5, max_move=0.06):
    """Where the two handles at an anchor differ by a few degrees only (just over the 12 degree corner threshold, never more
    than the ~16 degrees the curve merging can leave where a curve meets a line), the join is a smooth one that came out a
    little kinked: turn the handle of the curved side onto the straight side's direction (both onto their bisector when
    both are curves). Handle lengths are kept. Real corners (more than `hi` degrees) are not touched, and so is any anchor where
    turning the handles would move a curve by more than max_move stroke widths (a long arc fitted with that tangent keeps it)."""
    segs = [np.array(c, float) for c in segs]
    n = len(segs)

    def straight(c):
        ch = c[3] - c[0]
        L = np.linalg.norm(ch)
        if L < 1e-9:
            return True
        dev = max(abs(ch[0] * (c[1][1] - c[0][1]) - ch[1] * (c[1][0] - c[0][0])), abs(ch[0] * (c[2][1] - c[0][1]) - ch[1] * (c[2][0] - c[0][0]))) / L
        return dev < 0.02 * L

    for k in range(n if closed else n - 1):
        a, b = segs[k], segs[(k + 1) % n]
        if a is b:
            continue
        ha, hb = a[3] - a[2], b[1] - b[0]
        if np.linalg.norm(ha) < 1e-9 or np.linalg.norm(hb) < 1e-9:
            continue
        ang = angle_deg(ha, hb)
        if not (lo < ang <= hi):
            continue
        ua, ub = ha / np.linalg.norm(ha), hb / np.linalg.norm(hb)
        sa, sb = straight(a), straight(b)
        if sa and not sb:
            tgt_a, tgt_b = ua, ua
        elif sb and not sa:
            tgt_a, tgt_b = ub, ub
        elif sa and sb:
            continue  # two straight pieces meeting at a small angle: a real (small) corner
        else:
            m = ua + ub
            tgt_a = tgt_b = m / (np.linalg.norm(m) + 1e-12)
        na2 = a[3] - tgt_a * np.linalg.norm(ha)
        nb1 = b[0] + tgt_b * np.linalg.norm(hb)
        t = np.linspace(0, 1, 25)[:, None]
        bz = lambda c: (1 - t) ** 3 * c[0] + 3 * (1 - t) ** 2 * t * c[1] + 3 * (1 - t) * t ** 2 * c[2] + t ** 3 * c[3]  # noqa: E731
        moved = max(np.linalg.norm(bz(a) - bz(np.array([a[0], a[1], na2, a[3]])), axis=1).max(),
                    np.linalg.norm(bz(b) - bz(np.array([b[0], nb1, b[2], b[3]])), axis=1).max())
        if moved > max_move * w_px:
            continue
        a[2], b[1] = na2, nb1
    return segs


def round_small_kinks(segs, closed, P, w_px):
    """Two straight pieces that meet at 12 to 35 degrees where the chain itself bends with a real radius (a tight S-bend, not a
    corner) get a small curve between them instead of a corner anchor. The measured radius decides, so real small corners stay."""
    segs = [np.array(c, float) for c in segs]
    n = len(segs)

    def straight(c):
        ch = c[3] - c[0]
        L = np.linalg.norm(ch)
        if L < 1e-9:
            return False
        dev = max(abs(ch[0] * (c[1][1] - c[0][1]) - ch[1] * (c[1][0] - c[0][0])), abs(ch[0] * (c[2][1] - c[0][1]) - ch[1] * (c[2][0] - c[0][0]))) / L
        return dev < 0.02 * L

    out = []
    for k in range(n):
        out.append(segs[k])
    ins = {}  # index of segment -> fillet cubic to put after it
    new = [c.copy() for c in segs]
    for k in range(n if closed else n - 1):
        a, b = new[k], new[(k + 1) % n]
        if a is b or not (straight(a) and straight(b)):
            continue
        ua, ub = a[3] - a[0], b[3] - b[0]
        La, Lb = np.linalg.norm(ua), np.linalg.norm(ub)
        if La < 1e-9 or Lb < 1e-9:
            continue
        ua, ub = ua / La, ub / Lb
        th = angle_deg(ua, ub)
        if not (12 <= th <= FILLET_MAX_DEG) or np.linalg.norm(a[3] - b[0]) > 1e-6 * w_px + 1e-3:
            continue
        X = a[3]
        d = np.linalg.norm(P - X, axis=1)
        i = int(np.argmin(d))
        if d[i] > w_px:
            continue
        lo, hi = i, i
        while lo > 0 and d[lo - 1] < 1.5 * w_px:
            lo -= 1
        while hi < len(P) - 1 and d[hi + 1] < 1.5 * w_px:
            hi += 1
        R = min_bend_radius(P[lo:hi + 1], w_px) if hi - lo >= 6 else 0.0
        if R <= FILLET_MIN_R * w_px:
            continue
        R = min(R, 8 * w_px)
        tl = min(R * np.tan(np.radians(th) / 2), 0.4 * La, 0.4 * Lb)
        if tl < FILLET_MIN_LEN * w_px:
            continue  # a rounded join that small would only crowd three anchors together
        h = 4 / 3 * np.tan(np.radians(th) / 4) * tl / np.tan(np.radians(th) / 2)
        p1, p2 = X - tl * ua, X + tl * ub
        a[3], a[2], a[1] = p1, p1 - (p1 - a[0]) / 3, a[0] + (p1 - a[0]) / 3
        b[0], b[1], b[2] = p2, p2 + (b[3] - p2) / 3, b[3] - (b[3] - p2) / 3
        ins[k] = np.array([p1, p1 + h * ua, p2 - h * ub, p2])
    res = []
    for k in range(n):
        res.append(new[k])
        if k in ins:
            res.append(ins[k])
    return res


def fit_chain(P, closed, w_px, tol=0.07, corner_deg=50, corner_r=None, junctions=()):
    segs = align_small_kinks(collapse_double_corners(_fit_chain(P, closed, w_px, tol, corner_deg, corner_r, junctions), closed, w_px), closed, w_px)
    return round_small_kinks(segs, closed, np.asarray(P, float), w_px)


def _fit_chain(P, closed, w_px, tol=0.07, corner_deg=50, corner_r=None, junctions=()):
    """P: (N,2) points (x,y) in pixels. Returns cubic segments (4,2)."""
    P = np.asarray(P, float)
    if closed and len(P) < 12:
        return []
    Lmin, tolr = 2.0 * w_px, 0.06 * w_px
    if closed:
        Q = np.vstack([P, P[:1]])
        idx = rdp_idx(Q, tolr)
        if not any(it["kind"] == "line" for it in build_items(Q, Lmin, tolr, True)):
            return fit_curve_closed(P, w_px, tol, corner_deg)
        # open the loop at the start of a straight line, never inside a corner, so every corner is seen whole
        items0 = build_items(Q, Lmin, tolr, True)
        cand = [it for it in items0 if it["kind"] == "line" and it["i0"] > 0]
        s = int((cand[0] if cand else next(it for it in items0 if it["kind"] == "line"))["i0"])
        R = np.roll(P, -s, axis=0)
        P = np.vstack([R, R[:1]])
    elif not any(it["kind"] == "line" for it in build_items(P, Lmin, tolr, False)):
        return fit_curve_piece(P, w_px, tol, corner_deg)
    return assemble(P, closed, w_px, tol, corner_deg, corner_r, junctions)



# ---------------------------------------------------------------- sub-pixel medial axis (replaces the pixel skeleton)
def render_cov(path, vb, S):
    """Anti-aliased coverage 0..1 of the artwork (1 = ink)."""
    png = bytes(resvg_py.svg_to_bytes(svg_path=str(path), width=int(round(vb[2] * S))))
    img = Image.open(io.BytesIO(png)).convert("RGBA")
    comp = Image.alpha_composite(Image.new("RGBA", img.size, (255, 255, 255, 255)), img)
    L = np.asarray(comp.convert("L"), float)
    lo = L.min()
    cov = np.clip((255 - L) / max(255 - lo, 1.0), 0, 1) if lo < 250 else np.zeros_like(L)
    solid = cov > 0.97
    rgb = np.asarray(comp.convert("RGB"))[solid] if solid.any() else np.zeros((1, 3), np.uint8)
    render_cov.ink = "#%02x%02x%02x" % tuple(int(v) for v in np.median(rgb, axis=0))  # colour of the artwork
    return cov, img.size[0] / vb[2]


def vector_graph(cov, w_hint, dot_mask=None, step=0.5):
    """Medial axis of the shape from its sub-pixel boundary (Voronoi of boundary samples), as a graph in the same format
    trace_graph() returns. Positions are (y, x) in the same 'pixel index' convention as the skeleton pixels.
    Because the boundary is located to a small fraction of a pixel, the result does not depend on where the artwork
    happens to sit on the pixel grid."""
    H, W = cov.shape
    pad = np.pad(cov, 1)
    pts, cid = [], []
    for k, c in enumerate(measure.find_contours(pad, 0.5)):
        closed = np.allclose(c[0], c[-1])
        yx = c - 1.0
        body = yx if closed else yx
        seg = np.r_[0, np.cumsum(np.hypot(*np.diff(body, axis=0).T))]
        total = seg[-1]
        if total < 4:
            continue
        iy, ix = int(np.clip(yx[:, 0].mean(), 0, H - 1)), int(np.clip(yx[:, 1].mean(), 0, W - 1))
        if max(np.ptp(yx[:, 0]), np.ptp(yx[:, 1])) < 1.6 * w_hint and cov[iy, ix] >= 0.5:
            continue  # a small solid blob: handled as a dot
        n = max(int(total / step), 8)
        t = np.linspace(0, total, n, endpoint=not closed)
        smp = np.c_[np.interp(t, seg, body[:, 0]), np.interp(t, seg, body[:, 1])]
        if len(smp) > 12:  # ~1 px of smoothing removes quantisation noise that would sprout hairs on the axis
            smp = ndimage.gaussian_filter1d(smp, 1.0 / step, axis=0, mode="wrap" if closed else "nearest")
        pts.append(smp)
        cid += [k] * len(pts[-1])
    if not pts:
        return {}, {}, [], [], w_hint
    P = np.vstack(pts)
    cid = np.array(cid)
    order = np.concatenate([np.arange(len(p)) for p in pts])
    sizes = np.concatenate([[len(p)] * len(p) for p in pts])
    vor = Voronoi(P, qhull_options="Qbb Qc Qz")
    V = vor.vertices
    cov_in = cov if dot_mask is None else np.where(ndimage.binary_dilation(dot_mask, iterations=3), 0.0, cov)  # dots are not part of the strokes
    inside = ndimage.map_coordinates(cov_in, [V[:, 0], V[:, 1]], order=1, mode="constant", cval=0.0) >= 0.5
    rad = cKDTree(P).query(V)[0]
    rp = np.asarray(vor.ridge_points)
    rv = np.array([r if len(r) == 2 else (-1, -1) for r in vor.ridge_vertices])
    ok = (rv[:, 0] >= 0) & (rv[:, 1] >= 0)
    rp, rv = rp[ok], rv[ok]
    ok = inside[rv[:, 0]] & inside[rv[:, 1]]
    rp, rv = rp[ok], rv[ok]
    sep = np.linalg.norm(P[rp[:, 0]] - P[rp[:, 1]], axis=1)
    gap = np.abs(order[rp[:, 0]] - order[rp[:, 1]])
    gap = np.minimum(gap, sizes[rp[:, 0]] - gap)
    same = cid[rp[:, 0]] == cid[rp[:, 1]]
    rmin = np.minimum(rad[rv[:, 0]], rad[rv[:, 1]])
    ok = ~(same & (gap <= 3)) & (sep >= np.maximum(3 * step, 0.2 * rmin))
    rv = rv[ok]
    adj = {}
    for a, b in rv:
        if a != b:
            adj.setdefault(int(a), set()).add(int(b))
            adj.setdefault(int(b), set()).add(int(a))
    if not adj:
        return {}, {}, [], [], w_hint
    deg = {v: len(n) for v, n in adj.items()}
    node_ids = {v: i for i, v in enumerate(v for v in adj if deg[v] != 2)}
    pos = {i: V[v].copy() for v, i in node_ids.items()}
    is_junc = {i: deg[v] >= 3 for v, i in node_ids.items()}
    done = set()

    def resample(poly, step1=1.0):
        poly = np.asarray(poly)
        sg = np.r_[0, np.cumsum(np.hypot(*np.diff(poly, axis=0).T))]
        if sg[-1] <= step1:
            return []
        t = np.arange(step1, sg[-1] - 0.5 * step1, step1)
        return list(map(tuple, np.c_[np.interp(t, sg, poly[:, 0]), np.interp(t, sg, poly[:, 1])]))

    edges = []
    for v in list(node_ids):
        for w in adj[v]:
            if frozenset((v, w)) in done:
                continue
            chain, prev, cur = [v], v, w
            done.add(frozenset((v, w)))
            while cur not in node_ids:
                chain.append(cur)
                nxt = [q for q in adj[cur] if q != prev]
                if not nxt:
                    break
                done.add(frozenset((cur, nxt[0])))
                prev, cur = cur, nxt[0]
            chain.append(cur)
            if cur in node_ids:
                edges.append([node_ids[v], node_ids[cur], resample(V[chain])])
    loops = []
    for v in list(adj):
        if v in node_ids:
            continue
        if any(frozenset((v, q)) in done for q in adj[v]):
            continue
        chain, prev, cur = [v], None, v
        while True:
            nxt = [q for q in adj[cur] if q != prev and frozenset((cur, q)) not in done]
            if not nxt:
                break
            done.add(frozenset((cur, nxt[0])))
            prev, cur = cur, nxt[0]
            if cur == v:
                break
            chain.append(cur)
        if len(chain) > 8:
            loops.append(np.array(resample(V[chain + [chain[0]]]) + []))
    loops = [np.vstack([l, l[:1]]) for l in loops if len(l) > 8]
    used = [v for v in adj if deg[v] == 2]
    w_est = float(np.median(2 * rad[used])) if used else w_hint
    return pos, is_junc, edges, loops, w_est


# ---------------------------------------------------------------- driver
def run(path, tol=0.07, join_deg=45, corner_deg=50, target_w_px=20, thick_f=1.7, corner_radius=None, vector=True, keep_fills=False):
    """keep_fills: keep thick solid areas as filled shapes (off by default: output is strokes and dots only).
    corner_radius: bends with a centreline radius up to this many stroke widths become sharp corners (default CORNER_R)."""
    vb = viewbox(path)
    S0 = 1000 / max(vb[2], vb[3])
    mask0, S0 = render_mask(path, vb, S0)
    dt0 = ndimage.distance_transform_edt(mask0)
    sk0 = skeletonize(mask0)
    w_units = float(np.median(2 * dt0[sk0])) / S0
    S = min(target_w_px / max(w_units, 1e-6), 4000 / max(vb[2], vb[3]))
    if vector:
        cov, S = render_cov(path, vb, S)
        mask = cov >= 0.5
        ink = getattr(render_cov, 'ink', '#000000')
    else:
        mask, S = render_mask(path, vb, S)
        ink = '#000000'
    H, Wd = mask.shape
    dt = ndimage.distance_transform_edt(mask)
    sk = skeletonize(mask)
    w_px = float(np.median(2 * dt[sk]))
    lab, n = ndimage.label(mask)  # 4-connected: small solid marks that only touch at a corner (two squares) are separate dots
    dots = []
    dot_mask = np.zeros_like(mask)
    big_dots = []
    short_segs = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if max(h, w) < DOT_MAX * w_px:  # a small solid mark: a dot (a cutoff that is too tight splits identical marks, because their sizes differ by a pixel)
            comp = lab == i
            ys, xs = np.nonzero(comp)
            area = len(xs)
            fill_of_circle = area / (np.pi * (max(h, w) / 2) ** 2)  # about 1.0 for a disc, about 1.27 for a square
            round_disc = max(h, w) < 1.15 * min(h, w) and 0.85 <= fill_of_circle <= 1.1
            cx, cy = xs.mean() + 0.5, ys.mean() + 0.5
            ev, evec = np.linalg.eigh(np.cov(np.vstack([xs, ys])))  # shape of the blob: spread along its long axis against across it
            ax = evec[:, 1]
            t = (xs - xs.mean()) * ax[0] + (ys - ys.mean()) * ax[1]
            u = -(xs - xs.mean()) * ax[1] + (ys - ys.mean()) * ax[0]
            length, width = t.max() - t.min() + 1, u.max() - u.min() + 1
            # a short stroke, not a dot: clearly longer than wide on both measures. Measured: dots stay below 1.25 and 0.2 stroke widths even when pixel-shifted, real short strokes are above 1.35 and 0.35
            elongated = ev[0] > 0 and np.sqrt(ev[1] / ev[0]) >= 1.3 and (length - width) >= 0.28 * w_px
            if round_disc and 2 * np.sqrt(area / np.pi) >= 1.3 * w_px:  # clearly wider than a stroke; otherwise a plain dot is right
                big_dots.append((cx, cy, 2 * np.sqrt(area / np.pi) + 1))  # a round disc wider than a stroke: its own width
            elif elongated:  # a sparkle arm, an eye: a short segment along its long side
                half = (length - width) / 2
                short_segs.append((np.array([cx, cy]) - half * ax, np.array([cx, cy]) + half * ax, width))
            else:
                dots.append((xs.mean() + 0.5, ys.mean() + 0.5))
            sk[comp] = False
            dot_mask |= comp
    # solid round discs (much wider than a stroke) become one fat dot each: a point drawn with a stroke as wide as the disc
    if not keep_fills:
        rr = max(2, int(round(0.85 * w_px)))
        yy, xx = np.ogrid[-rr:rr + 1, -rr:rr + 1]
        opened = ndimage.binary_opening(np.pad(mask, rr + 1), structure=(xx * xx + yy * yy) <= rr * rr)[rr + 1:-(rr + 1), rr + 1:-(rr + 1)]
        olab, on = ndimage.label(opened)
        for i, sl in enumerate(ndimage.find_objects(olab), 1):
            comp = olab == i
            area = comp.sum()
            h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
            if (dot_mask & comp).any():
                continue  # already taken as a small mark above
            # strokes leaving the disc: ink components in a band just outside it. A dot has none, or one (a stem it sits on, the nib of a pen);
            # a thick blob where several strokes meet (the fish end on the board in Gravad Lax, a lime wheel's hub) is not a dot, the strokes run into it
            band = ndimage.binary_dilation(comp, iterations=max(int(0.9 * w_px), 2)) & ~ndimage.binary_dilation(comp, iterations=max(int(0.4 * w_px), 1))
            arms_lab, n_arms = ndimage.label(mask & band, structure=np.ones((3, 3), bool))
            if n_arms and sum(1 for k in range(1, n_arms + 1) if (arms_lab == k).sum() >= 0.3 * w_px) > MAX_DISC_ARMS:
                continue
            if 2 * np.sqrt(area / np.pi) >= 1.7 * w_px and area >= 0.75 * np.pi * (max(h, w) / 2) ** 2:
                ys, xs = np.nonzero(comp)
                big_dots.append((xs.mean() + 0.5, ys.mean() + 0.5, 2 * np.sqrt(area / np.pi) + 1))
                dot_mask |= comp
    if vector:
        pos, is_junc, edges, loops, w_px = vector_graph(cov, w_px, dot_mask)
        if not pos and not loops:
            only_marks = [dict(bez=[[np.array([x, y])] * 2 + [np.array([x + 0.01, y])] * 2], closed=False, w_px=d) for x, y, d in big_dots]
            only_marks += [dict(bez=[[a, a + (b - a) / 3, a + 2 * (b - a) / 3, b]], closed=False, w_px=wd) for a, b, wd in short_segs]
            return dict(paths=only_marks, dots=dots, fills=[], w_px=w_px, S=S, vb=vb, mask=mask)
    else:
        pos, is_junc, edges, loops = trace_graph(sk)
    newpos, E = simplify_graph(pos, is_junc, edges, w_px, dt)

    def widths(pts):
        yy = np.clip(pts[:, 0].astype(int), 0, H - 1)
        xx = np.clip(pts[:, 1].astype(int), 0, Wd - 1)
        return 2 * dt[yy, xx] + 1

    # thick (non-stroke) areas: edges much wider than the typical stroke become filled shapes instead
    if not keep_fills:
        thick_f = 1e9
    thin, thick_sk = [], np.zeros_like(mask)
    for n0, n1, pts in E:
        if len(pts) > 3 and np.percentile(widths(pts), 75) > thick_f * w_px:
            thick_sk[np.clip(pts[:, 0].astype(int), 0, H - 1), np.clip(pts[:, 1].astype(int), 0, Wd - 1)] = True
        else:
            thin.append((n0, n1, pts))
    fills = []
    region = np.zeros_like(mask)
    if thick_sk.any():
        dist, (iy, ix) = ndimage.distance_transform_edt(~thick_sk, return_indices=True)
        region = mask & (dist <= dt[iy, ix] + 1.5)
        region = ndimage.binary_opening(region, iterations=1) | (region & ndimage.binary_dilation(thick_sk, iterations=2))
        kept = []
        for n0, n1, pts in thin:  # thin leftovers lying inside the filled area are artefacts of it
            yy = np.clip(pts[:, 0].astype(int), 0, H - 1)
            xx = np.clip(pts[:, 1].astype(int), 0, Wd - 1)
            if region[yy, xx].mean() < 0.6:
                kept.append((n0, n1, pts))
        thin = kept
        for c in measure.find_contours(np.pad(region, 1).astype(float), 0.5):
            c = c[:-1] if np.allclose(c[0], c[-1]) else c
            Pc = np.c_[c[:, 1] - 1 + 0.5, c[:, 0] - 1 + 0.5]
            if len(Pc) > 6 and np.hypot(*np.diff(Pc, axis=0).T).sum() > 2 * w_px:
                fills.append(dict(bez=fit_chain(Pc, True, w_px, tol, corner_deg, corner_radius), closed=True))
    chains = pair_and_chain(newpos, thin, w_px, join_deg) if thin else []
    chains += [(L_[:-1], True, []) for L_ in loops]
    # pass 1: normalise chains, and find rings (arcs whose missing part is covered by other ink) to complete
    prepared, circles = [], []
    for P, closed, bend_nodes in chains:
        if not closed and len(P) > 20 and np.linalg.norm(P[0] - P[-1]) < 0.6 * w_px:
            # an outline whose two ends meet is a closed outline (it only came out open because the junction where it
            # starts/ends was split); closing it removes the double anchor at the join
            closed, P = True, P[:-1]
        wp = float(np.median(widths(P)))
        wp = w_px if abs(wp - w_px) < 0.25 * w_px else wp
        Pxy = P[:, ::-1] + 0.5  # (y,x) -> (x,y), pixel centres
        is_circle = False
        if not closed and len(Pxy) > 20:
            found = complete_circle(Pxy, mask, w_px)
            if found is not None:
                circ, extra = found
                cc = (circ[:, 0].mean(), circ[:, 1].mean(), float(np.hypot(*(circ[0] - circ.mean(axis=0)))))
                dup = any(np.hypot(cc[0] - o[0], cc[1] - o[1]) < 0.4 * w_px and abs(cc[2] - o[2]) < 0.4 * w_px for o in circles)
                if extra <= 1.5 * w_px:
                    if dup:
                        continue  # a ring that was split in two arcs: both halves complete to the same circle, keep one
                    circles.append(cc)
                    Pxy, closed, bend_nodes, is_circle = circ, True, [], True
                else:
                    # a ring with a long stroke running on from it (a line through a ring): the ring becomes its own
                    # circle, the stroke is cut back to the ring in pass 2
                    if not dup:
                        circles.append(cc)
                        prepared.append([circ, True, [], w_px, True])
        prepared.append([Pxy, closed, bend_nodes, wp, is_circle])
    # rings that other strokes cut into several pieces (a watch case crossed by its straps): one circle fitted across the pieces
    for cc, circ in find_split_rings([q[0] for q in prepared if not q[1] and not q[4]], circles, mask, w_px):
        circles.append(cc)
        prepared.append([circ, True, [], w_px, True])
    # pass 2: a stroke that runs along a completed ring (a slider track passing "through" its knob) stops at the ring
    pieces = []
    for Pxy, closed, bend_nodes, wp, is_circle in prepared:
        if is_circle or not circles:
            pieces.append((Pxy, closed, bend_nodes, wp))
            continue
        on = np.zeros(len(Pxy), bool)
        for cx, cy, R in circles:
            on |= np.abs(np.hypot(Pxy[:, 0] - cx, Pxy[:, 1] - cy) - R) < 0.35 * w_px
        if closed:
            # a closed outline that runs along a ring (a car body over its wheels): open it at a point off the ring
            if on.all():
                continue  # a closed outline lying wholly on a completed ring is the ring itself (already drawn as a circle)
            if not (on.any() and any(len(r) >= w_px for r in np.split(np.arange(len(Pxy)), np.nonzero(np.diff(on.astype(int)))[0] + 1) if on[r[0]])):
                pieces.append((Pxy, closed, bend_nodes, wp))
                continue
            k0 = int(np.nonzero(~on)[0][0])
            Pxy, on = np.roll(Pxy, -k0, axis=0), np.roll(on, -k0)
            Pxy, on, closed = np.vstack([Pxy, Pxy[:1]]), np.r_[on, on[0]], False
        runs = np.split(np.arange(len(Pxy)), np.nonzero(np.diff(on.astype(int)))[0] + 1)
        long_on = [r for r in runs if on[r[0]] and len(r) >= w_px]
        if not long_on:
            pieces.append((Pxy, closed, bend_nodes, wp))
            continue
        near = np.min([np.abs(np.hypot(Pxy[:, 0] - cx, Pxy[:, 1] - cy) - R) for cx, cy, R in circles], axis=0)
        drop = np.zeros(len(Pxy), bool)
        for r in long_on:
            drop[r] = near[r] < 0.2 * w_px  # only the part really on the ring goes; a stub leaving it radially stays
        for r in np.split(np.arange(len(Pxy)), np.nonzero(np.diff(drop.astype(int)))[0] + 1):
            if drop[r[0]]:
                continue
            # a short stub that sticks out of the ring (a watch crown) is kept; short leftovers hugging the ring are not
            off = np.min([np.abs(np.hypot(Pxy[r, 0] - cx, Pxy[r, 1] - cy) - R) for cx, cy, R in circles], axis=0).max()
            if len(r) >= 0.6 * w_px or (len(r) >= 0.2 * w_px and off >= 0.25 * w_px):
                pieces.append((Pxy[r], False, [], wp))
    # exact centerlines from the SVG's own straight edges (experiment): chains get snapped onto them before fitting
    ex_lines = np.zeros((0, 4))
    cpts = np.zeros((0, 2))
    circ_px = []
    try:
        wu = w_px / S

        def ink_at(p):
            x, y = (p[0] - vb[0]) * S, (p[1] - vb[1]) * S
            return 0 <= int(y) < H and 0 <= int(x) < Wd and mask[int(y), int(x)]

        segs = outline_lines(path)
        cl_u = np.zeros((0, 4))  # exact centerlines in SVG units
        if len(segs):
            cl = pair_edges(merge_collinear(segs, 0.02 * wu), wu, ink_at)
            if cl:
                cl_u = np.array(cl)[:, :4]
        # exact corners from the outline's round corners/caps; sides between them where edge pairing found none
        arcs = arcs_of(path, wu)
        cpts = corner_points(arcs, wu, ink_at)
        circ_px = [((c - np.array(vb[:2])) * S, r * S, sp) for c, r, sp in exact_circles(arcs, wu, ink_at)]
        if len(cpts):
            cl_u = np.vstack([cl_u, corner_lines(cpts, wu, ink_at, cl_u)])
        ex_circ = exact_circles(arcs, wu, ink_at)
        if len(ex_circ):
            cl_u = np.vstack([cl_u, circle_lines(ex_circ, wu, ink_at, cl_u)])
        if len(cl_u):
            ex_lines = (cl_u.reshape(-1, 2) - np.array(vb[:2])).reshape(-1, 4) * S
            ex_lines = merge_lines(ex_lines, w_px)
    except Exception:  # noqa: BLE001 - no usable exact edges: behave like the plain engine
        ex_lines = np.zeros((0, 4))
    rad = CORNER_R if corner_radius is None else corner_radius
    ring_ends = []
    # ovals (flat rings whose hole is an ellipse): the exact centreline oval replaces the chains that lie on it
    ovals, oval_paths, ring_paths = [], [], []
    # complete rings (clean circular or elliptical hole, ink all round) are one closed shape each; chains are cut back to them
    rings = []
    try:
        rings = find_rings(mask, w_px)
        rings = rings + find_rrects(mask, w_px, rings, rad)
        free = find_free_rings(mask, w_px, rings)  # plates and blades around a thin hole, any shape: the hole's outline pushed out by half a stroke
        free.sort(key=lambda r: len(r["P"]))  # smallest first: where two rings share a side (a pole standing on its base) the later one gives it up
        drawn_free = list(rings)  # every ring drawn so far, rectangles and ellipses too: a side shared with one of them is theirs
        for fr in free:
            chains = [(fr["P"], True)]
            for prev in drawn_free:
                nxt = []
                for Q, cl in chains:
                    left, cut = ring_cut(Q, cl, prev, w_px)
                    nxt += left if cut else [(Q, cl)]
                chains = nxt
            parts = []
            for Q, cl in chains:
                bez = fit_chain(Q, cl, w_px, tol, corner_deg, corner_radius, [])  # (no corner sharpening: a round end such as a pole's top or a blade's tip stays round)
                if bez:
                    parts.append((bez, cl))
            if parts:
                fr["parts"] = parts
                rings.append(fr)
                drawn_free.append(fr)
        pieces, ring_paths = complete_rings(pieces, rings, w_px, list(circles) + [(c[0][0], c[0][1], c[1]) for c in circ_px])
    except Exception:  # noqa: BLE001 - no usable rings: behave as before
        ring_paths = []
    try:
        ovals = find_ovals(mask, w_px)
        if ovals:
            drawn = [r["p"] for r in rings if r.get("done")]  # rings the step above already drew whole: not again
            left = [o for o in ovals if not any(np.hypot(o["p"][0] - q[0], o["p"][1] - q[1]) < 0.3 * w_px and abs(o["p"][2] - q[2]) < 0.3 * w_px and abs(o["p"][3] - q[3]) < 0.3 * w_px for q in drawn)]
            pieces, oval_paths = apply_ovals(pieces, left, w_px)
    except Exception:  # noqa: BLE001 - no usable ovals: behave as before
        ovals, oval_paths = [], []
    ring_open = [e for e in ring_paths if not e[1]]  # a ring that gave up a shared side: open pieces, not arcs of a shape
    oval_paths = [e for e in ring_paths if e[1]] + oval_paths
    # circles and ellipses that clean stretches of the chains lie on (a balloon that other outlines touch, a round lobe): the
    # chains are snapped onto them below, junction zones included. Shapes are never completed here, only arcs are kept.
    try:
        ovals = ovals + find_chain_ovals(pieces, w_px, ovals, list(circles) + [(c[0][0], c[0][1], c[1]) for c in circ_px])
    except Exception:  # noqa: BLE001 - no usable shapes: behave as before
        pass
    cpts_px = (cpts - np.array(vb[:2])) * S if len(cpts) else cpts
    paths = []
    for Pxy, closed, bend_nodes, wp in pieces:
        Pxy = snap_chain(Pxy, closed, ex_lines, w_px, rad)
        Pxy = snap_circles(Pxy, closed, circ_px, w_px)
        Pxy = snap_ellipses(Pxy, closed, ovals, w_px)
        if ovals and len(ex_lines):
            for _ in range(4):  # one junction per call, a chain can have several (both ends of the rim); also the other way round
                Q = join_lines_ovals(Pxy, closed, ex_lines, ovals, w_px)
                Q = join_lines_ovals(Q[::-1], closed, ex_lines, ovals, w_px)[::-1]
                if len(Q) == len(Pxy) and np.allclose(Q, Pxy):
                    break
                Pxy = Q
        arc = arc_of_chain(Pxy, closed, circ_px, w_px)
        Pxy = sharpen_bends(Pxy, closed, w_px, rad, corners=cpts_px, circles=circ_px, ovals=ovals)
        if len(Pxy) < 2 or np.hypot(*np.diff(Pxy, axis=0).T).sum() < 1:
            continue
        bez = arc_beziers(*arc) if arc else fit_chain(Pxy, closed, w_px, tol, corner_deg, corner_radius, [q[::-1] + 0.5 for q in bend_nodes])
        if bez:
            is_open_arc = bool(arc) and closed and abs(arc[3]) < 2 * np.pi - 0.01
            if is_open_arc:
                ring_ends += [(len(paths), 0, bez[0][0]), (len(paths), -1, bez[-1][3])]
            paths.append(dict(bez=bez, closed=closed and not is_open_arc, w_px=wp))
    for bez, closed_, wp_ in oval_paths:
        if not closed_:
            ring_ends += [(len(paths), 0, bez[0][0]), (len(paths), -1, bez[-1][3])]
        paths.append(dict(bez=bez, closed=closed_, w_px=wp_))
    for bez, closed_, wp_ in ring_open:
        paths.append(dict(bez=bez, closed=False, w_px=wp_))
    # a stroke that ran into a ring now stops at the open end of the exact arc: join it (its line passes through that end)
    for k, pa in enumerate(paths):
        if pa["closed"] or any(k == q[0] for q in ring_ends):
            continue
        for which in (0, -1):
            piece = pa["bez"][which]
            end = piece[0] if which == 0 else piece[3]
            near = [(np.linalg.norm(R - end), R) for kk, _, R in ring_ends if kk != k]
            if not near or min(near)[0] > 0.8 * w_px:
                continue
            R = min(near, key=lambda t: t[0])[1]
            d = R - end
            if which == 0:
                piece[0], piece[1] = piece[0] + d, piece[1] + d
            else:
                piece[3], piece[2] = piece[3] + d, piece[2] + d
    converge_ends(paths, w_px, mask)
    paths = add_missing_rings(paths, mask, w_px)
    try:
        paths = snap_ends(paths, w_px, mask)  # ends that nearly meet another stroke land exactly on its centreline
    except Exception:  # noqa: BLE001 - leave the ends as they were
        pass
    for x, y, d in big_dots:
        paths.append(dict(bez=[[np.array([x, y]), np.array([x, y]), np.array([x + 0.01, y]), np.array([x + 0.01, y])]], closed=False, w_px=d))
    for a, b, wd in short_segs:
        paths.append(dict(bez=[[a, a + (b - a) / 3, a + 2 * (b - a) / 3, b]], closed=False, w_px=wd))
    res = dict(paths=paths, dots=dots, fills=fills, w_px=w_px, S=S, vb=vb, mask=mask, ink=ink)
    if keep_fills:
        add_residual_fills(res, tol, corner_deg, corner_radius)
    return res


def add_residual_fills(res, tol, corner_deg, corner_radius=None):
    """Anything solid that the strokes do not explain (discs, blobs, wedges) is kept as a filled shape."""
    reconstruction_scores(res)
    rec, m, w_px = res["rec"], res["mask"], res["w_px"]
    r = max(2, int(round(0.2 * w_px)))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    miss = ndimage.binary_opening(m & ~rec, structure=(xx * xx + yy * yy) <= r * r)
    if not miss.any():
        return
    r2 = max(2, int(round(0.7 * w_px)))
    yy, xx = np.ogrid[-r2:r2 + 1, -r2:r2 + 1]
    miss = ndimage.binary_dilation(miss, iterations=2) & m  # grow back to the ink edge
    miss = ndimage.binary_fill_holes(ndimage.binary_closing(np.pad(miss, r2 + 1), structure=(xx * xx + yy * yy) <= r2 * r2))[r2 + 1:-(r2 + 1), r2 + 1:-(r2 + 1)] & m  # solid discs, no notches
    for c in measure.find_contours(np.pad(miss, 1).astype(float), 0.5):
        c = c[:-1] if np.allclose(c[0], c[-1]) else c
        Pc = np.c_[c[:, 1] - 1 + 0.5, c[:, 0] - 1 + 0.5]
        if len(Pc) > 6 and np.hypot(*np.diff(Pc, axis=0).T).sum() > 1.5 * w_px:
            res["fills"].append(dict(bez=fit_chain(Pc, True, w_px, tol, corner_deg, corner_radius), closed=True, residual=True))



# ---------------------------------------------------------------- completing circles and converging line ends
def fit_circle(P):
    A = np.c_[2 * P[:, 0], 2 * P[:, 1], np.ones(len(P))]
    sol = np.linalg.lstsq(A, (P ** 2).sum(axis=1), rcond=None)[0]
    return sol[0], sol[1], float(np.sqrt(max(sol[2] + sol[0] ** 2 + sol[1] ** 2, 1e-9)))



def find_split_rings(chains, known, mask, w_px, min_cover=240.0):
    """Circles that several open chains share. A ring crossed by other strokes is cut into arcs that are each too short
    for complete_circle; here candidate circles come from stretches of every chain, and a candidate counts when the chains'
    points on it cover at least `min_cover` degrees and the rest of the circle lies on ink inside the drawing."""
    if len(chains) < 1:
        return []
    cands = []
    win = int(2.5 * w_px)
    for P in chains:
        for s0 in range(0, max(len(P) - win, 1), max(4, win // 8)):
            seed = P[s0:s0 + win]
            if len(seed) < win:
                continue
            cx, cy, R = fit_circle(seed)
            if not (1.5 * w_px <= R <= 25 * w_px):
                continue
            if np.abs(np.hypot(seed[:, 0] - cx, seed[:, 1] - cy) - R).max() > 0.12 * w_px:
                continue
            cands.append((cx, cy, R))
    found = []
    taken = list(known)
    for cx, cy, R in cands:
        if any(np.hypot(cx - o[0], cy - o[1]) < 0.4 * w_px and abs(R - o[2]) < 0.4 * w_px for o in taken):
            continue
        pts = np.vstack([P[np.abs(np.hypot(P[:, 0] - cx, P[:, 1] - cy) - R) < 0.15 * w_px] for P in chains])
        if len(pts) < 12:
            continue
        cx, cy, R = fit_circle(pts)  # refine on everything that lies on the circle
        if any(np.hypot(cx - o[0], cy - o[1]) < 0.4 * w_px and abs(R - o[2]) < 0.4 * w_px for o in taken):
            continue  # the same ring reached from another stretch
        # coverage counts only real arcs along the circle (stretches of at least one stroke width), not the few points where
        # a straight stroke merely crosses or touches it
        bins = set()
        for P in chains:
            on = np.abs(np.hypot(P[:, 0] - cx, P[:, 1] - cy) - R) < 0.15 * w_px
            for run in np.split(np.arange(len(P)), np.nonzero(np.diff(on.astype(int)))[0] + 1):
                if on[run[0]] and len(run) >= w_px:
                    a = np.degrees(np.arctan2(P[run, 1] - cy, P[run, 0] - cx)) % 360
                    bins |= set((a // 3).astype(int).tolist())
        if len(bins) * 3.0 < min_cover:
            continue
        t = np.linspace(0, 2 * np.pi, max(int(2 * np.pi * R), 24), endpoint=False)
        mx, my = cx + R * np.cos(t), cy + R * np.sin(t)
        H, W = mask.shape
        inside = (mx >= 0) & (mx < W) & (my >= 0) & (my < H)
        ink = ndimage.binary_dilation(mask, iterations=2)
        ok = np.zeros(len(t), bool)
        ok[inside] = ink[my[inside].astype(int), mx[inside].astype(int)]
        if ok.mean() < 0.9:
            continue
        taken.append((cx, cy, R))
        found.append(((cx, cy, R), np.c_[mx, my]))
    return found


def complete_circle(Pxy, mask, w_px):
    """If an open path is mostly a circular arc (>= 150 degrees) and the missing part of that circle lies inside other ink
    (e.g. a ring touching a line, where the two fuse), return the full circle's points; otherwise None.
    The arc's own legs/stubs (where it runs off toward the other stroke) are trimmed off before fitting."""
    def trim(arc):  # repeatedly keep the longest run of points lying on the circle fitted to the current points
        for _ in range(8):
            if len(arc) < 12:
                return None
            cx, cy, R = fit_circle(arc)
            good = np.abs(np.hypot(Pxy[:, 0] - cx, Pxy[:, 1] - cy) - R) < 0.12 * w_px
            idx = np.nonzero(good)[0]
            if len(idx) < 12:
                return None
            runs = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
            new = Pxy[max(runs, key=len)]
            if len(new) == len(arc) and np.allclose(new[0], arc[0]) and np.allclose(new[-1], arc[-1]):
                return arc
            arc = new
        return None

    arc = trim(Pxy)
    if arc is None:  # a ring with a long stroke running on from it: the whole path is no circle, so look for a circular stretch
        win = int(2.5 * w_px)
        best = None
        for s0 in range(0, max(len(Pxy) - win, 1), max(4, win // 8)):
            seed = Pxy[s0:s0 + win]
            cx, cy, R = fit_circle(seed)
            if not (0.8 * w_px <= R <= 25 * w_px) or np.abs(np.hypot(seed[:, 0] - cx, seed[:, 1] - cy) - R).max() > 0.12 * w_px:
                continue
            got = trim(seed)
            if got is not None and (best is None or len(got) > len(best)):
                best = got
        arc = best
        if arc is None:
            return None
    cx, cy, R = fit_circle(arc)
    if not (0.8 * w_px <= R <= 25 * w_px) or len(arc) < 2.5 * w_px:
        return None
    ang = np.unwrap(np.arctan2(arc[:, 1] - cy, arc[:, 0] - cx))
    extent = abs(ang[-1] - ang[0])
    if not (np.radians(150) <= extent <= np.radians(345)):
        return None
    # the missing part of the circle must be ink (otherwise this is just an arc, like a smile)
    a0, a1 = ang[0], ang[-1]
    step = 1.0 / R
    if a1 >= a0:
        miss = np.arange(a1 + step, a0 + 2 * np.pi - step, step)
    else:
        miss = np.arange(a0 + step, a1 + 2 * np.pi - step, step)
    if len(miss) == 0:
        return None
    mx, my = cx + R * np.cos(miss), cy + R * np.sin(miss)
    H, W = mask.shape
    ink = ndimage.binary_dilation(mask, iterations=2)
    inside = (mx >= 0) & (mx < W) & (my >= 0) & (my < H)  # parts of the circle outside the drawing are not ink
    ok = np.zeros(len(miss), bool)
    ok[inside] = ink[my[inside].astype(int), mx[inside].astype(int)]
    if ok.mean() < 0.85:
        return None
    t = np.linspace(0, 2 * np.pi, max(int(2 * np.pi * R), 24), endpoint=False)
    return np.c_[cx + R * np.cos(t), cy + R * np.sin(t)], len(Pxy) - len(arc)


def add_missing_rings(paths, mask, w_px):
    """A round ring (a berry, a bead) whose hole is enclosed ink-wise but whose centreline the rest of the pipeline did not
    close: only an arc of it came out, because the strokes of a garland run into it and its thickness is uneven. Where fewer than
    85% of the ring's circumference is covered by a path, the ring is drawn as one closed circle between the hole and the
    ring's outer edge, and the partial arcs lying on it are dropped. Rings that are already complete are left alone."""
    H, W = mask.shape
    bg, n = ndimage.label(~mask)
    cands = []
    for i, sl in enumerate(ndimage.find_objects(bg), 1):
        if sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == H or sl[1].stop == W:
            continue
        h, ww = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        comp = bg == i
        ys, xs = np.nonzero(comp)
        A = len(xs)
        if not (0.6 * w_px <= np.sqrt(A / np.pi) <= 4 * w_px) or max(h, ww) > 1.06 * min(h, ww) or not 0.92 <= A / (np.pi * (max(h, ww) / 2) ** 2) <= 1.08:
            continue
        cands.append((xs.mean() + 0.5, ys.mean() + 0.5))
    if not cands:
        return paths
    samples = []
    for p in paths:
        u = np.linspace(0, 1, 24)[:, None]
        for c in p["bez"]:
            samples.append((1 - u) ** 3 * c[0] + 3 * (1 - u) ** 2 * u * c[1] + 3 * (1 - u) * u ** 2 * c[2] + u ** 3 * c[3])
    tree = cKDTree(np.vstack(samples)) if samples else None
    paths = list(paths)
    for cx, cy in cands:
        inner, runs = [], []
        for ang in np.radians(np.arange(0, 360, 10)):
            d, dx, dy = 0.0, np.cos(ang), np.sin(ang)
            while d < 6 * w_px and not mask[min(max(int(cy + d * dy), 0), H - 1), min(max(int(cx + d * dx), 0), W - 1)]:
                d += 0.5
            e0 = d
            while d < 6 * w_px and mask[min(max(int(cy + d * dy), 0), H - 1), min(max(int(cx + d * dx), 0), W - 1)]:
                d += 0.5
            inner.append(e0)
            if d - e0 < 3.5 * w_px:
                runs.append(d - e0)
        if len(runs) < 24:
            continue
        T = float(np.median(runs))
        if not (0.7 * w_px <= T <= 2.5 * w_px) or min(runs) < 0.6 * w_px:  # no gap in the ring
            continue
        R = float(np.median(inner)) + T / 2
        t = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        ring = np.c_[cx + R * np.cos(t), cy + R * np.sin(t)]
        if tree is not None and (tree.query(ring)[0] < 0.35 * w_px).mean() >= RING_COVERED:
            continue  # already drawn
        keep = []
        for p in paths:
            pts = _samp(p)
            on_ring = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - R) < 0.4 * w_px
            if not p["closed"] and on_ring.mean() >= 0.9:
                continue  # an arc of this very ring
            keep.append(p)
        paths = keep + [dict(bez=arc_beziers(np.array([cx, cy]), R, 0.0, 2 * np.pi), closed=True, w_px=T if T > 1.4 * w_px else w_px)]
        samples = None
        tree = cKDTree(np.vstack([_samp(p) for p in paths]))
    return paths


def _samp(p):
    u = np.linspace(0, 1, 24)[:, None]
    return np.vstack([(1 - u) ** 3 * c[0] + 3 * (1 - u) ** 2 * u * c[1] + 3 * (1 - u) * u ** 2 * c[2] + u ** 3 * c[3] for c in p["bez"]])


def converge_ends(paths, w_px, mask):
    """Free ends of straight strokes that point at each other (the diagonals and stem of a pen nib) are extended to the
    single point where their lines meet, if that point is just ahead of each end."""
    ink = ndimage.binary_dilation(mask, iterations=2)
    Hh, Ww = mask.shape

    def runs_through_ink(a, b):
        n = max(int(np.linalg.norm(b - a) / 2), 2)
        pts = a + (b - a) * np.linspace(0, 1, n)[:, None]
        return ink[np.clip(pts[:, 1].astype(int), 0, Hh - 1), np.clip(pts[:, 0].astype(int), 0, Ww - 1)].mean() >= 0.95

    ends = []
    for pi, p in enumerate(paths):
        if p["closed"]:
            continue
        for which in (0, 1):
            c = p["bez"][0] if which == 0 else p["bez"][-1]
            pt, other = (c[0], c[3]) if which == 0 else (c[3], c[0])
            hdl = (c[0] - c[1]) if which == 0 else (c[3] - c[2])
            chord = pt - other
            if np.linalg.norm(chord) < 0.8 * w_px:
                continue
            d = chord / np.linalg.norm(chord)
            if np.linalg.norm(hdl) > 1e-9 and angle_deg(hdl, chord) > 8:
                continue  # not a straight end
            ends.append((pi, which, pt, d))
    used = set()
    for i in range(len(ends)):
        if i in used:
            continue
        group = [j for j in range(len(ends)) if j not in used and np.linalg.norm(ends[j][2] - ends[i][2]) < 3 * w_px and ends[j][0] != -1]
        for _attempt in range(3):
            if len(group) < 2:
                break
            M, b = np.zeros((2, 2)), np.zeros(2)
            for j in group:
                _, _, pt, d = ends[j]
                Pm = np.eye(2) - np.outer(d, d)
                M += Pm
                b += Pm @ pt
            if abs(np.linalg.det(M)) < 1e-6:
                break
            X = np.linalg.solve(M, b)
            bad = []
            for j in group:
                _, _, pt, d = ends[j]
                t = (X - pt) @ d
                off = np.linalg.norm((X - pt) - t * d)
                if t < -0.3 * w_px or t > 2.5 * w_px or off > 0.25 * w_px or not runs_through_ink(pt, X):
                    bad.append((off + max(0, -t) + max(0, t - 2.5 * w_px) + 1.0, j))
            if not bad:
                if max(angle_deg(ends[a][3], ends[b][3]) for a in group for b in group) < 15:
                    break  # near parallel: the meeting point is meaningless
                for j in group:
                    pi, which, pt, d = ends[j]
                    c = paths[pi]["bez"][0] if which == 0 else paths[pi]["bez"][-1]
                    c[0 if which == 0 else 3] = X  # move the free end to the meeting point, keep the segment straight
                    c[1] = c[0] + (c[3] - c[0]) / 3
                    c[2] = c[0] + 2 * (c[3] - c[0]) / 3
                    used.add(j)
                break
            group.remove(max(bad)[1])


# ---------------------------------------------------------------- output helpers
def to_units(res, pt):
    return np.asarray(pt) / res["S"] + np.array(res["vb"][:2])


def path_d(res, p, pixels=False):
    f = (lambda q: "%.2f %.2f" % tuple(q)) if pixels else (lambda q: "%.3f %.3f" % tuple(to_units(res, q)))
    d = "M" + f(p["bez"][0][0])
    for c in p["bez"]:
        d += " C" + " ".join(f(q) for q in c[1:])
    return d + (" Z" if p["closed"] else "")


def anchors(res):
    n = sum(len(p["bez"]) + (0 if p["closed"] else 1) for p in res["paths"]) + len(res["dots"])
    return n


def fill_anchors(res):
    return sum(len(p["bez"]) for p in res["fills"])


def sample_polys(res, n=40):
    out = []
    t = np.linspace(0, 1, n)
    for p in res["paths"]:
        pts = np.vstack([_bez(c, t) for c in p["bez"]])
        out.append(pts / res["S"] + np.array(res["vb"][:2]))
    return out


def result_svg(res):
    x0, y0, w, h = res["vb"]
    body = ""
    for p in res["paths"]:
        body += f'<path stroke-width="{p["w_px"]/res["S"]:.3f}" d="{path_d(res, p)}"/>'
    wu = res["w_px"] / res["S"]
    for x, y in res["dots"]:
        ux, uy = to_units(res, (x, y))
        body += f'<path stroke-width="{wu:.3f}" d="M{ux:.3f} {uy:.3f}h0.001"/>'
    fills = ""
    if res["fills"]:
        fills = f'<path fill="{res.get("ink", "#000000")}" stroke="none" fill-rule="evenodd" id="thick-areas-kept-as-fills" d="' + " ".join(path_d(res, p) for p in res["fills"]) + '"/>'
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0} {y0} {w} {h}" width="{w}" height="{h}">'
            f'<g fill="none" stroke="{res.get("ink", "#000000")}" stroke-linecap="round" stroke-linejoin="round">{body}</g>{fills}</svg>')


def reconstruction_scores(res):
    """Truth-free check: redraw the result (strokes at their widths + fills) and compare to the original ink."""
    m = res["mask"]
    hgt, wid = m.shape
    body = ""
    for p in res["paths"]:
        body += f'<path stroke-width="{p["w_px"]:.2f}" d="{path_d(res, p, True)}"/>'
    for x, y in res["dots"]:
        body += f'<path stroke-width="{res["w_px"]:.2f}" d="M{x:.2f} {y:.2f}h0.01"/>'
    fills = ""
    if res["fills"]:
        fills = '<path fill="#000" fill-rule="evenodd" d="' + " ".join(path_d(res, p, True) for p in res["fills"]) + '"/>'
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{wid}" height="{hgt}" viewBox="0 0 {wid} {hgt}">'
           f'<rect width="100%" height="100%" fill="#fff"/><g fill="none" stroke="#000" stroke-linecap="round" stroke-linejoin="round">{body}</g>{fills}</svg>')
    img = Image.open(io.BytesIO(bytes(resvg_py.svg_to_bytes(svg_string=svg, width=wid)))).convert("L")
    rec = np.array(img) < 128
    if rec.shape != m.shape:
        return dict(iou=None, missed=None, extra=None)
    res["rec"] = rec
    inter = (rec & m).sum()
    return dict(iou=float(inter / max((rec | m).sum(), 1)), missed=float((m & ~rec).sum() / max(m.sum(), 1)), extra=float((rec & ~m).sum() / max(rec.sum(), 1)))


# colour sets for the Compare view: pale tint of the original, centerline, smooth anchors, corner anchors, point strokes, uncovered ink
PALETTES = {
    "classic": dict(tint=None, line="#e5231b", smooth="#0057ff", corner="#0057ff", dot="#e5231b", miss=(255, 120, 0, 235), ring=None, node=1.0),
    "indigo": dict(tint="#d3d9ee", line="#3d4796", smooth="#5d7fd6", corner="#8a56c2", dot="#3d4796", miss=(240, 138, 58, 235), ring="#ffffff", node=1.2),
    "blue": dict(tint="#d4e1f1", line="#2e4d80", smooth="#4f89cc", corner="#1c2c55", dot="#2e4d80", miss=(240, 136, 62, 235), ring="#ffffff", node=1.2),
}
DEFAULT_PALETTE = "blue"  # "classic" is the original red/blue; "indigo" is the lavender/violet variant


def _diff_png(res, rgba=(255, 120, 0, 235)):
    m, rec = res["mask"], res["rec"]
    a = np.zeros(m.shape + (4,), np.uint8)
    # drop slivers along edges: only show ink that is missing by more than ~a third of a stroke width
    r = max(1, int(0.18 * res["w_px"]))
    miss = ndimage.binary_opening(m & ~rec, structure=np.ones((2 * r + 1,) * 2)) if r > 1 else (m & ~rec)
    a[miss] = rgba
    buf = io.BytesIO()
    Image.fromarray(a, "RGBA").save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def overlay_svg(svg_file, res, size=300, palette=None):
    pal = PALETTES[palette or DEFAULT_PALETTE]
    x0, y0, w, h = res["vb"]
    wu = res["w_px"] / res["S"]
    base = max(w, h)  # marker and line sizes follow the picture's size, not the stroke width, so they look the same on a small icon and a large drawing
    data = base64.b64encode(open(svg_file, "rb").read()).decode()
    pw, ph = (size, size * h / w) if w >= h else (size * w / h, size)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="{x0} {y0} {w} {h}" width="{pw:.0f}" height="{ph:.0f}">',
           f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#fff"/>']
    if pal["tint"]:  # the original, recoloured to one pale tone (whatever its own colour)
        out.append(f'<defs><filter id="tint" x="{x0}" y="{y0}" width="{w}" height="{h}" filterUnits="userSpaceOnUse" color-interpolation-filters="sRGB">'
                   f'<feFlood flood-color="{pal["tint"]}"/><feComposite in2="SourceAlpha" operator="in"/></filter></defs>')
        out.append(f'<image x="{x0}" y="{y0}" width="{w}" height="{h}" filter="url(#tint)" href="data:image/svg+xml;base64,{data}" xlink:href="data:image/svg+xml;base64,{data}"/>')
    else:
        out.append(f'<image x="{x0}" y="{y0}" width="{w}" height="{h}" opacity="0.28" href="data:image/svg+xml;base64,{data}" xlink:href="data:image/svg+xml;base64,{data}"/>')
    if "rec" in res:
        png = _diff_png(res, pal["miss"])
        out.append(f'<image x="{x0}" y="{y0}" width="{w}" height="{h}" href="data:image/png;base64,{png}" xlink:href="data:image/png;base64,{png}"/>')
    if res["fills"]:
        out.append('<path fill="#1a9850" fill-opacity="0.45" stroke="#0b5d2e" stroke-width="%.3f" fill-rule="evenodd" d="%s"/>' % (base * 0.0025, " ".join(path_d(res, p) for p in res["fills"])))
    for p in res["paths"]:
        out.append(f'<path d="{path_d(res, p)}" fill="none" stroke="{pal["line"]}" stroke-width="{base*0.0035:.3f}"/>')
    ring = f' stroke="{pal["ring"]}" stroke-width="{base*0.0016:.3f}"' if pal["ring"] else ""
    for p in res["paths"]:
        bez = p["bez"]
        for k in range(len(bez) + (0 if p["closed"] else 1)):
            q = bez[k][0] if k < len(bez) else bez[-1][3]
            ux, uy = to_units(res, q)
            if 0 < k < len(bez) or p["closed"]:  # interior anchor: corner if the two handles do not line up
                a, b = bez[k - 1], bez[k % len(bez)]
                corner = angle_deg(a[3] - a[2] if np.linalg.norm(a[3] - a[2]) > 1e-9 else a[3] - a[0], b[1] - b[0] if np.linalg.norm(b[1] - b[0]) > 1e-9 else b[3] - b[0]) > 12
            else:
                corner = False
            r = base * 0.006 * pal["node"]
            if corner:
                out.append(f'<rect x="{ux-r*1.25:.3f}" y="{uy-r*1.25:.3f}" width="{r*2.5:.3f}" height="{r*2.5:.3f}" fill="{pal["corner"]}"{ring}/>')
            else:
                out.append(f'<circle cx="{ux:.3f}" cy="{uy:.3f}" r="{r:.3f}" fill="{pal["smooth"]}"{ring}/>')
    for q in res["dots"]:
        ux, uy = to_units(res, q)
        out.append(f'<circle cx="{ux:.3f}" cy="{uy:.3f}" r="{base*0.0083:.3f}" fill="{pal["dot"]}"/>')
    out.append("</svg>")
    return "".join(out)
