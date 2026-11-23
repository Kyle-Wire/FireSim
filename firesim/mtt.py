"""Fire arrival time over a raster by time-dependent minimum travel time.

Provenance: written fresh for FireSim. The method is the minimum-travel-time idea (fire reaches each cell by the
fastest path; a path's time is the sum of its segment times) solved by anisotropic fast marching (Dijkstra ordering,
with each cell updated from pairs of neighbouring stencil directions; a plain shortest path along stencil directions
made a length-to-breadth 6 ellipse 25 % too small, sandbox finding F1), with segment times taken from the weather in force when the fire leaves a cell, so wind may change in time.
The rate in a direction θ off the heading follows an ellipse with the ignition at its rear focus:
R(θ) = R_head (1 − e) / (1 − e cos θ), e = sqrt(LB² − 1) / LB (geometry of an ellipse; no empirical constant here).
Spot fires enter as extra arrival events (see `embers.py`).

Units: metres, minutes, radians. Direction 0 = +x (east), counter-clockwise; row index grows with +y.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit


def stencil(reach: int = 3):
    """Offsets (dy, dx) with coprime components up to `reach`: 8, 16 or 32 directions for reach 1, 2, 3."""
    out = []
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            if (dy or dx) and math.gcd(abs(dy), abs(dx)) == 1:
                out.append((dy, dx))
    out.sort(key=lambda o: math.atan2(o[0], o[1]))      # in order of angle, so neighbours in the list are neighbours in direction
    return np.array(out, dtype=np.int64)


@njit(cache=True)
def _push(ht, hi, n, t, i):
    k = n
    ht[k] = t; hi[k] = i
    while k > 0:
        p = (k - 1) // 2
        if ht[p] <= ht[k]:
            break
        ht[p], ht[k] = ht[k], ht[p]
        hi[p], hi[k] = hi[k], hi[p]
        k = p
    return n + 1


@njit(cache=True)
def _pop(ht, hi, n):
    t = ht[0]; i = hi[0]
    n -= 1
    ht[0] = ht[n]; hi[0] = hi[n]
    k = 0
    while True:
        a = 2 * k + 1
        if a >= n:
            break
        b = a + 1
        c = a
        if b < n and ht[b] < ht[a]:
            c = b
        if ht[k] <= ht[c]:
            break
        ht[k], ht[c] = ht[c], ht[k]
        hi[k], hi[c] = hi[c], hi[k]
        k = c
    return t, i, n


@njit(cache=True)
def _slow(rh, hd, ec, s, j, ang):
    """Minutes per metre at cell j, weather slice s, toward direction ang. inf where nothing burns."""
    r = rh[s, j]
    if r <= 0.0:
        return np.inf
    e = ec[s, j]
    return (1.0 - e * math.cos(ang - hd[s, j])) / (r * (1.0 - e))


@njit(cache=True)
def _edge(rh, hd, ec, s, y, x, dy, dx, nx, ang):
    """Mean minutes per metre along the straight segment from (y, x) to (y+dy, x+dx); inf if any crossed cell cannot burn."""
    q = max(abs(dy), abs(dx))
    sl = 0.0
    for u in range(q + 1):
        yq = int(round(y + dy * u / q)); xq = int(round(x + dx * u / q))
        c = _slow(rh, hd, ec, s, yq * nx + xq, ang)
        if c == np.inf:
            return np.inf
        sl += (0.5 if (u == 0 or u == q) else 1.0) * c
    return sl / q


@njit(cache=True)
def _march(rh, hd, ec, cell, step_min, offs, angs, t0, t_end, nlam):
    """Anisotropic fast marching. A cell's time is the least, over pairs of neighbouring stencil directions whose
    cells are already burning, of (time interpolated along the line joining the pair) + (travel time from that point),
    so fire may arrive from any direction, not only along the stencil's own directions. Cells are
    re-opened when a later neighbour gives them an earlier time (label correcting), which strong wind needs: the pair of
    cells a fast flank run comes through can themselves burn later than the cell they feed in a one-pass march."""
    T, n = rh.shape
    nx = offs[0, 2]; ny = offs[0, 3]
    arr = t0.copy()
    cap = 64 * n + 64
    ht = np.empty(cap); hi = np.empty(cap, np.int64)
    m = 0
    for i in range(n):
        if arr[i] < np.inf:
            m = _push(ht, hi, m, arr[i], i)
    K = offs.shape[0]
    while m > 0:
        t, i, m = _pop(ht, hi, m)
        if t > arr[i]:
            continue          # a stale entry: this cell has since been given an earlier time
        if t > t_end:
            break
        s = min(max(int(t / step_min), 0), T - 1)
        if rh[s, i] <= 0.0:
            continue          # lit (by an ember or as an ignition) but not burnable now: it holds no fire
        y = i // nx; x = i - y * nx
        for k in range(K):
            # target j sees source i in direction k_src = the stencil entry (-dy, -dx) from j's side
            dy = offs[k, 0]; dx = offs[k, 1]
            yy = y + dy; xx = x + dx
            if yy < 0 or yy >= ny or xx < 0 or xx >= nx:
                continue
            j = yy * nx + xx
            if rh[s, j] <= 0.0 or arr[j] <= t:
                continue
            ang = angs[k]                      # travel direction i -> j
            d = cell * math.sqrt(dy * dy + dx * dx)
            ci = _edge(rh, hd, ec, s, y, x, dy, dx, nx, ang)
            if ci == np.inf:
                continue
            best = t + d * ci
            fi = ci / _slow(rh, hd, ec, s, j, ang)
            # the two stencil neighbours of j adjacent (in angle) to i
            for side in range(2):
                k2 = k + 1 if side == 0 else k - 1
                if k2 == K: k2 = 0
                if k2 < 0: k2 = K - 1
                dy2 = offs[k2, 0]; dx2 = offs[k2, 1]
                y2 = yy - dy2; x2 = xx - dx2
                if y2 < 0 or y2 >= ny or x2 < 0 or x2 >= nx:
                    continue
                i2 = y2 * nx + x2
                if arr[i2] == np.inf or rh[s, i2] <= 0.0:
                    continue
                t2 = arr[i2]
                c2 = _edge(rh, hd, ec, s, y2, x2, dy2, dx2, nx, angs[k2])
                if c2 == np.inf:
                    continue
                f2 = c2 / _slow(rh, hd, ec, s, j, angs[k2])
                for u in range(1, nlam):
                    lam = u / nlam
                    vy = lam * dy + (1.0 - lam) * dy2; vx = lam * dx + (1.0 - lam) * dx2
                    a = math.atan2(vy, vx)
                    tv = lam * t + (1.0 - lam) * t2 + cell * math.sqrt(vy * vy + vx * vx) * \
                        _slow(rh, hd, ec, s, j, a) * (lam * fi + (1.0 - lam) * f2)
                    if tv < best:
                        best = tv
            if best < t:
                best = t
            if best < arr[j] - 1e-9:
                arr[j] = best
                if m < cap:
                    m = _push(ht, hi, m, best, j)
    return arr


def arrival(rate_head, heading, ecc, cell, ignitions, step_min=60.0, t_end=np.inf, reach=2, nlam=8):
    """Arrival time (minutes) at each cell.

    rate_head, heading, ecc: (ny, nx) or (T, ny, nx); one slice per `step_min` minutes.
    ignitions: (ny, nx) array of start times (inf where not lit), or a list of (row, col, minute).
    """
    rh = np.asarray(rate_head, np.float64)
    if rh.ndim == 2:
        rh = rh[None]
    T, ny, nx = rh.shape
    hd = np.broadcast_to(np.asarray(heading, np.float64), rh.shape) if np.ndim(heading) < 3 else np.asarray(heading, np.float64)
    ec = np.broadcast_to(np.asarray(ecc, np.float64), rh.shape) if np.ndim(ecc) < 3 else np.asarray(ecc, np.float64)
    if isinstance(ignitions, np.ndarray):
        t0 = ignitions.astype(np.float64).ravel()
    else:
        t0 = np.full(ny * nx, np.inf)
        for r, c, t in ignitions:
            t0[int(r) * nx + int(c)] = min(t0[int(r) * nx + int(c)], float(t))
    st = stencil(reach)
    offs = np.zeros((len(st), 4), np.int64)
    offs[:, :2] = st; offs[:, 2] = nx; offs[:, 3] = ny
    a = _march(np.ascontiguousarray(rh.reshape(T, -1)), np.ascontiguousarray(hd.reshape(T, -1)),
               np.ascontiguousarray(ec.reshape(T, -1)), float(cell), float(step_min), offs,
               np.arctan2(st[:, 0], st[:, 1]).astype(np.float64), t0, float(t_end), int(nlam))
    return a.reshape(ny, nx)


def eccentricity(lb):
    lb = np.maximum(np.asarray(lb, np.float64), 1.0)
    return np.sqrt(lb * lb - 1.0) / lb
