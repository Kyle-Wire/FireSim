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
def _rhs(phi, a, b, c, hx, hy, cell, rate, out):
    ny, nx = phi.shape
    inv = 1.0 / cell
    for y in prange(ny):
        ym = y - 1 if y > 0 else 0; yp = y + 1 if y < ny - 1 else ny - 1
        ymm = y - 2 if y > 1 else 0; ypp = y + 2 if y < ny - 2 else ny - 1
        for x in range(nx):
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
def _heun(phi, tmp, k1, k2, a, b, c, hx, hy, cell, dt, t, arr, rate):
    _rhs(phi, a, b, c, hx, hy, cell, rate, k1)
    ny, nx = phi.shape
    for y in prange(ny):
        for x in range(nx):
            tmp[y, x] = phi[y, x] + dt * k1[y, x]
    _rhs(tmp, a, b, c, hx, hy, cell, rate, k2)
    for y in prange(ny):
        for x in range(nx):
            old = phi[y, x]
            if a[y, x] <= 0.0:
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
            tmp[y, x] = new
    for y in prange(ny):
        for x in range(nx):
            if a[y, x] <= 0.0:
                continue
            old = phi[y, x]; new = tmp[y, x]
            if old > 0.0 and new <= 0.0 and arr[y, x] == np.inf:
                arr[y, x] = t + dt * old / (old - new)
                rate[y, x] = (old - new) / dt
            phi[y, x] = new


def ellipse_abc(rate_head, lb):
    lb = np.maximum(np.asarray(lb, np.float64), 1.0)
    e = np.sqrt(lb * lb - 1.0) / lb
    a = np.asarray(rate_head, np.float64) / (1.0 + e)
    return a, a / lb, a * e


def arrival(rate_head, heading, lb, cell, ignitions, t_end, step_min=60.0, cfl=0.4, phi0=None, spot=None):
    """Arrival time (minutes) per cell up to t_end.

    rate_head (m/min), heading (rad), lb: (ny, nx) or (T, ny, nx), one slice per `step_min`.
    ignitions: list of (row, col, minute, radius_m). spot: optional callback(t0, t1, arr, phi) run after each step,
    which may lower phi to light new cells (embers).
    """
    R = np.asarray(rate_head, np.float64)
    R3 = R[None] if R.ndim == 2 else R
    T, ny, nx = R3.shape
    H3 = np.broadcast_to(np.asarray(heading, np.float64), R3.shape)
    L3 = np.broadcast_to(np.asarray(lb, np.float64), R3.shape)
    yy, xx = np.mgrid[:ny, :nx]
    phi = np.full((ny, nx), 1e9) if phi0 is None else phi0.astype(np.float64).copy()
    pending = sorted(ignitions, key=lambda g: g[2])
    arr = np.full((ny, nx), np.inf)
    rate = np.zeros((ny, nx))
    tmp = np.empty_like(phi); k1 = np.empty_like(phi); k2 = np.empty_like(phi)
    t = pending[0][2] if pending else 0.0
    s_now = -1
    while t < t_end:
        while pending and pending[0][2] <= t + 1e-12:
            r, cc, t0, rad = pending.pop(0)
            d = np.hypot(yy - r, xx - cc) * cell - max(rad, 0.5 * cell)
            lit = (d <= 0) & (arr == np.inf)
            arr[lit] = t0
            a0 = ellipse_abc(R3[min(int(t0 // step_min), T - 1)], L3[min(int(t0 // step_min), T - 1)])[0]
            rate[lit] = a0[lit]
            phi = np.minimum(phi, d)
        s = min(int(t // step_min), T - 1)
        if s != s_now:
            a, b, c = ellipse_abc(R3[s], L3[s])
            burn = R3[s] > 0
            a = np.where(burn, a, 0.0); hx = np.cos(H3[s]); hy = np.sin(H3[s])
            a = np.ascontiguousarray(a); b = np.ascontiguousarray(b); c = np.ascontiguousarray(c)
            hx = np.ascontiguousarray(hx); hy = np.ascontiguousarray(hy)
            vmax = float(R3[s].max())
            s_now = s
        if vmax <= 0:
            break
        dt = cfl * cell / vmax
        nxt = min(t_end, (s + 1) * step_min if s < T - 1 else t_end)
        if pending:
            nxt = min(nxt, pending[0][2])
        dt = min(dt, max(nxt - t, 1e-9))
        _heun(phi, tmp, k1, k2, a, b, c, hx, hy, float(cell), dt, t, arr, rate)
        if spot is not None:
            spot(t, t + dt, arr, phi)
        t += dt
    return arr, phi
