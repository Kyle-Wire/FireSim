"""Fire front by a level set: the reference spread solver.

Provenance: written fresh for FireSim. phi < 0 is burned. The front moves along its normal n at
    F(n) = c cos(th) + sqrt(a^2 cos^2(th) + b^2 sin^2(th)),   th = angle between n and the heading,
which is the support function of an ellipse with semi-axes a (along the heading) and b, drifting at c: every point of
the front then spreads as that ellipse (Huygens' principle; pure geometry, no empirical constant). With head rate R
and length-to-breadth LB, and the ignition at the rear focus: e = sqrt(LB^2-1)/LB, a = R/(1+e), c = a e, b = a/LB.
Numerics: Roe upwinding with a local Lax-Friedrichs fix, second-order one-sided (ENO2) differences, Heun (RK2) steps.
A plain upwind gradient with the normal computed apart made wind-driven ellipses 15 to 37 % too wide (finding F3).
Arrival time is the moment phi crosses zero in a cell, interpolated within the step.
Behind the front a cell keeps falling at the rate it had when the front crossed it. Left to the equation, phi goes
flat behind a front that grew from a point, the flat patch bleeds into the front and the fire runs late by a lag that
grows with the log of distance (sandbox finding F2: 4 % late at 400 m on 10 m cells).

Units: metres, minutes, radians; direction 0 = +x, counter-clockwise; row index grows with +y.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange


@njit(cache=True, inline="always")
def _mm(p, q):
    """minmod-limited second difference"""
    if p * q <= 0.0:
        return 0.0
    return p if abs(p) < abs(q) else q


@njit(cache=True, parallel=True)
def _rhs(phi, a, b, c, hx, hy, cell, rate, out, ay, ax_):
    ny, nx = phi.shape
    inv = 1.0 / cell
    for i in prange(len(ay)):
        y = ay[i]
        ym = y - 1 if y > 0 else 0; yp = y + 1 if y < ny - 1 else ny - 1
        ymm = y - 2 if y > 1 else 0; ypp = y + 2 if y < ny - 2 else ny - 1
        for x in range(ax_[i], ax_[i] + 1):
            if a[y, x] <= 0.0:
                out[y, x] = 0.0
                continue
            if phi[y, x] < 0.0 and rate[y, x] > 0.0:
                out[y, x] = -rate[y, x]      # behind the front: keep falling at the rate it crossed with (see module note)
                continue
            xm = x - 1 if x > 0 else 0; xp = x + 1 if x < nx - 1 else nx - 1
            xmm = x - 2 if x > 1 else 0; xpp = x + 2 if x < nx - 2 else nx - 1
            p0 = phi[y, x]
            # second differences
            sx0 = phi[y, xp] - 2 * p0 + phi[y, xm]
            sxm = phi[y, x] - 2 * phi[y, xm] + phi[y, xmm]
            sxp = phi[y, xpp] - 2 * phi[y, xp] + phi[y, x]
            sy0 = phi[yp, x] - 2 * p0 + phi[ym, x]
            sym = phi[y, x] - 2 * phi[ym, x] + phi[ymm, x]
            syp = phi[ypp, x] - 2 * phi[yp, x] + phi[y, x]
            dxm = (p0 - phi[y, xm] + 0.5 * _mm(sx0, sxm)) * inv
            dxp = (phi[y, xp] - p0 - 0.5 * _mm(sx0, sxp)) * inv
            dym = (p0 - phi[ym, x] + 0.5 * _mm(sy0, sym)) * inv
            dyp = (phi[yp, x] - p0 - 0.5 * _mm(sy0, syp)) * inv
            ax = a[y, x]; bx = b[y, x]; cx = c[y, x]; wx = hx[y, x]; wy = hy[y, x]
            # Roe upwinding with a local Lax-Friedrichs fix (Osher and Shu's scheme for Hamilton-Jacobi equations):
            # the Hamiltonian H(p, q) = c u + sqrt(a^2 u^2 + b^2 v^2), u = p.w along the heading, v across it.
            # Each derivative is taken from the side its characteristic comes from; where that side is ambiguous
            # the two are averaged with just enough smoothing to stay stable.
            pm = 0.5 * (dxm + dxp); qm = 0.5 * (dym + dyp)
            sgx = 0; sgy = 0; first = True; mixx = False; mixy = False
            for ii in range(2):
                p = dxm if ii == 0 else dxp
                for jj in range(2):
                    q = dym if jj == 0 else dyp
                    u = p * wx + q * wy; v = -p * wy + q * wx
                    S = math.sqrt(ax * ax * u * u + bx * bx * v * v)
                    if S > 0.0:
                        h1 = cx * wx + (ax * ax * u * wx - bx * bx * v * wy) / S
                        h2 = cx * wy + (ax * ax * u * wy + bx * bx * v * wx) / S
                    else:
                        h1 = cx * wx; h2 = cx * wy
                    s1 = 1 if h1 > 0 else -1; s2 = 1 if h2 > 0 else -1
                    if first:
                        sgx = s1; sgy = s2; first = False
                    else:
                        if s1 != sgx: mixx = True
                        if s2 != sgy: mixy = True
            if mixx:
                p = pm; dissx = (ax + cx) * 0.5 * (dxp - dxm)
            else:
                p = dxm if sgx > 0 else dxp; dissx = 0.0
            if mixy:
                q = qm; dissy = (ax + cx) * 0.5 * (dyp - dym)
            else:
                q = dym if sgy > 0 else dyp; dissy = 0.0
            u = p * wx + q * wy; v = -p * wy + q * wx
            H = cx * u + math.sqrt(ax * ax * u * u + bx * bx * v * v)
            out[y, x] = -(H - dissx - dissy)


@njit(cache=True, parallel=True)
def _heun(phi, tmp, k1, k2, a, b, c, hx, hy, cell, dt, t, arr, rate, ay, ax_):
    """One Heun step on the active cells (ay, ax_). tmp equals phi on entry and on exit."""
    _rhs(phi, a, b, c, hx, hy, cell, rate, k1, ay, ax_)
    ny, nx = phi.shape
    n = len(ay)
    for i in prange(n):
        tmp[ay[i], ax_[i]] = phi[ay[i], ax_[i]] + dt * k1[ay[i], ax_[i]]
    _rhs(tmp, a, b, c, hx, hy, cell, rate, k2, ay, ax_)
    for i in prange(n):
        y = ay[i]; x = ax_[i]
        old = phi[y, x]
        if a[y, x] <= 0.0:
            tmp[y, x] = old
            continue
        new = old + 0.5 * dt * (k1[y, x] + k2[y, x])
        if new > old:
            new = old                      # fire only advances: phi never rises
        if old > 0.0:
            # Minimum principle: within one step (CFL < 1) fire can only come from a neighbouring cell, so phi cannot
            # fall below the lowest burnable neighbour. Without this a cell cut off behind a firebreak can dig its own
            # minimum and "ignite" with no fire near it (sandbox finding F4).
            lo = old
            for dy in range(-1, 2):
                yy = y + dy
                if yy < 0 or yy >= ny:
                    continue
                for dx in range(-1, 2):
                    xx = x + dx
                    if xx < 0 or xx >= nx or (dy == 0 and dx == 0) or a[yy, xx] <= 0.0:
                        continue
                    if dy != 0 and dx != 0 and a[y, xx] <= 0.0 and a[yy, x] <= 0.0:
                        continue           # a diagonal neighbour reached only through the corner of two firebreak cells
                    if phi[yy, xx] < lo:
                        lo = phi[yy, xx]
            if new < lo:
                new = lo
        k1[y, x] = new
    for i in prange(n):
        y = ay[i]; x = ax_[i]
        if a[y, x] <= 0.0:
            continue
        old = phi[y, x]; new = k1[y, x]
        if old > 0.0 and new <= 0.0 and arr[y, x] == np.inf:
            arr[y, x] = t + dt * old / (old - new)
            rate[y, x] = (old - new) / dt
        phi[y, x] = new
        tmp[y, x] = new


@njit(cache=True)
def _crossed(arr, t, ay, ax_):
    """Active cells whose arrival time is later than t (those that crossed in the step just taken)."""
    k = 0
    oy = np.empty(len(ay), np.int64); ox = np.empty(len(ay), np.int64)
    for i in range(len(ay)):
        v = arr[ay[i], ax_[i]]
        if v > t and v < np.inf:
            oy[k] = ay[i]; ox[k] = ax_[i]; k += 1
    return oy[:k], ox[:k]


@njit(cache=True)
def _grow(phi, tmp, burn, act, K, cy, cx, cell):
    """Add to the band every unburned cell within K cells of the cells (cy, cx) that have just caught fire, and those
    cells themselves. Returns the cells added. The band is kept, not rebuilt: a cell stays in it until `_prune`.
    """
    ny, nx = phi.shape
    cap = max(1024, len(cy) * (2 * K + 1))
    ny_ = np.empty(cap, np.int64); nx_ = np.empty(cap, np.int64); nn = 0
    for i in range(len(cy)):
        y0 = cy[i]; x0 = cx[i]
        for y in range(max(y0 - K, 0), min(y0 + K + 1, ny)):
            for x in range(max(x0 - K, 0), min(x0 + K + 1, nx)):
                if act[y, x] != 0:
                    continue
                if phi[y, x] <= 0.0 and (abs(y - y0) > 2 or abs(x - x0) > 2):
                    continue                   # burned cells away from this one are in the band already if they matter
                if nn == cap:
                    cap *= 2
                    gy = np.empty(cap, np.int64); gx = np.empty(cap, np.int64)
                    gy[:nn] = ny_[:nn]; gx[:nn] = nx_[:nn]; ny_ = gy; nx_ = gx
                act[y, x] = 2; ny_[nn] = y; nx_[nn] = x; nn += 1
    # A cell entering the band carries a phi from long ago, far too high. Give it the value the band implies: continue
    # the slope of phi from its lowest unburned neighbour already in the band (phi_1 + (phi_1 - phi_2), or phi_1 plus
    # the distance where there is no second cell). Only through fuel, so a cell behind a firebreak keeps its value.
    # Cells already in the band are never touched. (Lowering every band cell to "neighbour plus distance" instead made
    # fires 5 to 9 % larger on the synthetic landscapes, because phi ahead of a wind-driven front is steeper than that.)
    d2 = cell * math.sqrt(2.0)
    for sweep in range(4):
        for j in range(nn):
            i = j if sweep % 2 == 0 else nn - 1 - j
            y = ny_[i]; x = nx_[i]
            if act[y, x] != 2:
                continue
            if phi[y, x] <= 0.0 or not burn[y, x]:
                act[y, x] = 1
                continue
            best = 1e300; ext = 0.0
            for dy in range(-1, 2):
                yy = y + dy
                if yy < 0 or yy >= ny:
                    continue
                for dx in range(-1, 2):
                    xx = x + dx
                    if xx < 0 or xx >= nx or (dx == 0 and dy == 0):
                        continue
                    if act[yy, xx] != 1 or not burn[yy, xx] or phi[yy, xx] <= 0.0:
                        continue
                    if dy != 0 and dx != 0 and (not burn[y, xx]) and (not burn[yy, x]):
                        continue
                    v = phi[yy, xx]
                    if v < best:
                        best = v
                        y2 = yy + dy; x2 = xx + dx
                        dd = d2 if (dx != 0 and dy != 0) else cell
                        if 0 <= y2 < ny and 0 <= x2 < nx and act[y2, x2] == 1 and burn[y2, x2] and v > phi[y2, x2]:
                            ext = v + (v - phi[y2, x2])
                        else:
                            ext = v + dd
            if best < 1e299:
                if ext < phi[y, x]:
                    phi[y, x] = ext; tmp[y, x] = ext
                act[y, x] = 1
    for i in range(nn):
        act[ny_[i], nx_[i]] = 1
    return ny_[:nn], nx_[:nn]


@njit(cache=True)
def _prune(phi, act, ay, ax_, n):
    """Drop from the band the burned cells with no unburned cell within two cells. Compacts (ay, ax_) in place and
    returns the new count."""
    ny, nx = phi.shape
    k = 0
    for i in range(n):
        y = ay[i]; x = ax_[i]
        keep = phi[y, x] > 0.0
        if not keep:
            for yy in range(max(y - 2, 0), min(y + 3, ny)):
                for xx in range(max(x - 2, 0), min(x + 3, nx)):
                    if phi[yy, xx] > 0.0:
                        keep = True
        if keep:
            ay[k] = y; ax_[k] = x; k += 1
        else:
            act[y, x] = 0
    return k


def ellipse_abc(rate_head, lb):
    lb = np.maximum(np.asarray(lb, np.float64), 1.0)
    e = np.sqrt(lb * lb - 1.0) / lb
    a = np.asarray(rate_head, np.float64) / (1.0 + e)
    return a, a / lb, a * e


def _light(phi, arr, rate, R, L, r, cc, t0, rad, cell):
    """Light a spot fire at (r, cc): phi = min(phi, distance to the spot) inside a window wide enough to hold the
    region where the spot is nearer than the existing front (half the old phi at the spot, plus a margin)."""
    ny, nx = phi.shape
    if R[r, cc] <= 0 or arr[r, cc] < np.inf:
        return None
    w = int(min(max(0.75 * phi[r, cc] / cell + 6, 6), 400))
    y0, y1, x0, x1 = max(r - w, 0), min(r + w + 1, ny), max(cc - w, 0), min(cc + w + 1, nx)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(yy - r, xx - cc) * cell - max(rad, 0.5 * cell)
    sub = phi[y0:y1, x0:x1]
    lit = (d <= 0) & (arr[y0:y1, x0:x1] == np.inf) & (R[y0:y1, x0:x1] > 0)
    arr[y0:y1, x0:x1][lit] = t0
    a0 = R[y0:y1, x0:x1] / (1 + np.sqrt(np.maximum(L[y0:y1, x0:x1] ** 2 - 1, 0)) / L[y0:y1, x0:x1])
    rate[y0:y1, x0:x1][lit] = a0[lit]
    np.minimum(sub, d, out=sub)
    ly, lx = np.nonzero(lit)
    return (y0, y1, x0, x1), ly + y0, lx + x0


def arrival(rate_head, heading, lb, cell, ignitions, t_end, step_min=60.0, cfl=0.4, phi0=None, spot=None, spot_window=True,
            band=8, refresh=16):
    """Arrival time (minutes) per cell up to t_end.

    rate_head (m/min), heading (rad), lb: (ny, nx) or (T, ny, nx), one slice per `step_min`. For large fires pass
    rate_head as a function s -> (rate, heading, lb) and the number of slices as `heading`.
    ignitions: list of (row, col, minute, radius_m). spot: optional callback(t0, t1, arr, phi, rows, cols) run after
    each step (embers, structures); rows, cols are the cells that caught fire in the step. It may return new ignitions
    (row, col, minute, radius_m); these are applied inside a local window (see `_light`).
    band: the solver works only on cells within `band` cells of where fire has just arrived (`_grow`), drops burned
    cells with nothing left to burn beside them every `refresh` steps (`_prune`), and takes its time step from the
    fastest cell in the band. band=None solves the whole grid every step (the reference
    the band is checked against in the sandbox).
    """
    if callable(rate_head):
        # rate_head(s) -> (rate, heading, lb) for slice s; heading = number of slices. Keeps one slice in memory.
        T = int(heading); _c = {}
        def get(s):
            if _c.get("s") != s:
                r_, h_, l_ = rate_head(s)
                r_ = np.ascontiguousarray(r_, np.float64)
                _c.update(s=s, v=(r_, np.broadcast_to(np.asarray(h_, np.float64), r_.shape), np.broadcast_to(np.asarray(l_, np.float64), r_.shape)))
            return _c["v"]
        ny, nx = get(0)[0].shape
    else:
        R = np.asarray(rate_head, np.float64)
        R3 = R[None] if R.ndim == 2 else R
        T, ny, nx = R3.shape
        H3 = np.broadcast_to(np.asarray(heading, np.float64), R3.shape)
        L3 = np.broadcast_to(np.asarray(lb, np.float64), R3.shape)
        get = lambda s: (R3[s], H3[s], L3[s])
    yy, xx = np.mgrid[:ny, :nx]
    phi = np.full((ny, nx), 1e9) if phi0 is None else phi0.astype(np.float64).copy()
    pending = sorted(ignitions, key=lambda g: g[2])
    arr = np.full((ny, nx), np.inf)
    rate = np.zeros((ny, nx))
    tmp = phi.copy(); k1 = np.empty_like(phi); k2 = np.empty_like(phi)
    act = np.zeros((ny, nx), np.uint8)
    if band is None:
        ay, ax_ = (v.ravel().astype(np.int64) for v in np.mgrid[:ny, :nx]); n_act = len(ay)
    else:
        ay = np.empty(4096, np.int64); ax_ = np.empty(4096, np.int64); n_act = 0
    lit_y = []; lit_x = []; since = 0; vmax = 0.0
    pcy = pcx = np.empty(0, np.int64)
    e64 = np.empty(0, np.int64)
    if phi0 is not None:
        ly, lx = np.nonzero(phi <= 0); lit_y.append(ly); lit_x.append(lx)
    t = pending[0][2] if pending else 0.0
    s_now = -1
    while t < t_end:
        while pending and pending[0][2] <= t + 1e-12:
            g = pending.pop(0)
            r, cc, t0, rad = g[:4]
            s0 = min(int(t0 // step_min), T - 1)
            if len(g) > 4:                                    # spot ignition: local window
                got = _light(phi, arr, rate, get(s0)[0], get(s0)[2], int(r), int(cc), t0, rad, cell)
                if got is not None:
                    w = got[0]
                    tmp[w[0]:w[1], w[2]:w[3]] = phi[w[0]:w[1], w[2]:w[3]]
                    lit_y.append(got[1]); lit_x.append(got[2])
                continue
            d = np.hypot(yy - r, xx - cc) * cell - max(rad, 0.5 * cell)
            lit = (d <= 0) & (arr == np.inf)
            arr[lit] = t0
            a0 = ellipse_abc(get(s0)[0], get(s0)[2])[0]
            rate[lit] = a0[lit]
            phi = np.minimum(phi, d)
            tmp[:] = phi
            ly, lx = np.nonzero(lit); lit_y.append(ly); lit_x.append(lx)
        s = min(int(t // step_min), T - 1)
        new_s = s != s_now
        if new_s:
            Rs, Hs, Ls = get(s)
            a, b, c = ellipse_abc(Rs, Ls)
            burn = Rs > 0
            a = np.where(burn, a, 0.0); hx = np.cos(Hs); hy = np.sin(Hs)
            a = np.ascontiguousarray(a); b = np.ascontiguousarray(b); c = np.ascontiguousarray(c)
            hx = np.ascontiguousarray(hx); hy = np.ascontiguousarray(hy); burn = np.ascontiguousarray(burn)
            s_now = s
        ry = np.concatenate(lit_y) if lit_y else e64; rx = np.concatenate(lit_x) if lit_x else e64
        lit_y = []; lit_x = []
        if band is not None:
            wy = np.concatenate([ry, pcy]) if len(pcy) else ry; wx_ = np.concatenate([rx, pcx]) if len(pcx) else rx
            if len(wy):                                        # cells lit by an ignition, or crossed in the last step
                gy_, gx_ = _grow(phi, tmp, burn, act, band, wy, wx_, float(cell))
                if n_act + len(gy_) > len(ay):
                    cap = max(2 * len(ay), n_act + len(gy_))
                    ay = np.concatenate([ay[:n_act], np.empty(cap - n_act, np.int64)]); ax_ = np.concatenate([ax_[:n_act], np.empty(cap - n_act, np.int64)])
                ay[n_act:n_act + len(gy_)] = gy_; ax_[n_act:n_act + len(gy_)] = gx_; n_act += len(gy_)
                if len(gy_):
                    vmax = max(vmax, float(Rs[gy_, gx_].max()))
            if since >= refresh:
                n_act = _prune(phi, act, ay, ax_, n_act); since = 0; new_s = True
            if new_s:
                vmax = float(Rs[ay[:n_act], ax_[:n_act]].max()) if n_act else 0.0
        elif new_s:
            vmax = float(Rs.max())
        nxt = min(t_end, (s + 1) * step_min if s < T - 1 else t_end)
        if pending:
            nxt = min(nxt, pending[0][2])
        cy_ = e64; cx_ = e64
        if vmax <= 0:
            # nothing can move in this slice of the fields: wait for the next slice, the next ignition or the end
            if spot is None and not pending and s >= T - 1:
                break
            dt = max(nxt - t, 1e-9) if spot is None else min(max(nxt - t, 1e-9), 1.0)
        else:
            dt = min(cfl * cell / vmax, max(nxt - t, 1e-9))
            _heun(phi, tmp, k1, k2, a, b, c, hx, hy, float(cell), dt, t, arr, rate, ay[:n_act], ax_[:n_act])
            since += 1
            if band is not None or spot is not None:
                cy_, cx_ = _crossed(arr, t, ay[:n_act], ax_[:n_act])
        if spot is not None:
            new = spot(t, t + dt, arr, phi, np.concatenate([ry, cy_]) if len(ry) else cy_, np.concatenate([rx, cx_]) if len(rx) else cx_)
            if new:
                pending.extend((r, cc, max(t0, t + dt), rad) + ((True,) if spot_window else ()) for r, cc, t0, rad in new)
                pending.sort(key=lambda g: g[2])
        pcy, pcx = cy_, cx_                                    # they grow the band before the next step
        t += dt
    return arr, phi
