"""Embers (spotting) and the per-step hook that joins the fire front, embers and houses.

Provenance: written fresh for FireSim. The form is a deliberate simplification, not a published model:
each burned cell throws a Poisson number of *effective* embers (embers able to start a fire) in proportion to its
fireline intensity; each lands downwind at a lognormal distance with a normal scatter in direction; an ember that
lands on unburned fuel starts a spot fire with a fixed chance, and one that lands on a house is passed to
`structures.Structures`.

Every number in `PRIORS` is a starting guess to be CALIBRATED on the calibration fires of gridzip-lab
`physics/PROTOCOL-FIRE.md` (Camp, Tubbs, Marshall, East Troublesome). None is a sourced constant, and none may be
quoted as one. The one published landing law we opened (Trucchia et al. 2019, GMD 12:69, eqs 4-12) was not adopted:
its fitted spread (sigma about 7 in log space for a 50 MW/m fire) throws most embers either nowhere or off the map.
"""
import numpy as np

PRIORS = dict(
    gen=2e-5,        # effective embers per m2 burned per MW/m of fireline intensity        (calibrate)
    fli_min=500.0,   # kW/m below which a cell throws none                                  (calibrate)
    d_med=150.0,     # m, median throw at 10 MW/m and 10 m/s open wind                      (calibrate)
    a_i=0.4,         # exponent of intensity in the median throw                            (calibrate)
    b_u=1.0,         # exponent of wind speed in the median throw                           (calibrate)
    sigma=0.9,       # log-space spread of throw distance                                   (calibrate)
    sig_theta=0.2,   # rad, scatter of throw direction about the wind                       (calibrate)
    p_ign=0.3,       # chance an effective ember on unburned fuel starts a spot fire        (calibrate)
    loft_min=1.0,    # min, time from burning to launch; flight time is distance / wind     (calibrate)
)


class Spot:
    """Callback for `levelset.arrival(spot=...)`.

    fli (kW/m), wind (m/s, open) and wind_to (rad, 0 = +x, pi/2 = +row): (ny, nx) or (T, ny, nx) per `step_min`,
    or functions of the slice number returning (ny, nx).
    burnable: bool (ny, nx). fixed=(throw_m, delay_min): every burned cell throws exactly one ember straight
    downwind that always ignites (sandbox case S7). structures: optional `Structures`.
    """

    def __init__(self, cell, fli, wind, wind_to, burnable, seed=0, step_min=60.0, fixed=None, structures=None, **kw):
        bad = set(kw) - set(PRIORS)
        if bad:
            raise TypeError(f"unknown ember parameters {bad}")
        self.p = {**PRIORS, **kw}
        self.cell = float(cell); self.step_min = step_min; self.fixed = fixed; self.st = structures
        self.burnable = np.asarray(burnable, bool)
        sh = self.burnable.shape
        f3 = lambda v: v if callable(v) else (np.broadcast_to(np.asarray(v, np.float64), (1,) + sh) if np.ndim(v) < 3 else np.asarray(v, np.float64))
        self.fli, self.wind, self.wdir = f3(fli), f3(wind), f3(wind_to)
        self.rng = np.random.default_rng(seed)
        self.Lt = np.empty(0); self.Lx = np.empty(0); self.Ly = np.empty(0)
        self.n_thrown = 0; self.n_spots = 0
        if structures is not None:
            structures.embers = self

    def _s(self, t, arr3):
        if callable(arr3):                                     # a function of the slice number (large fires)
            return arr3(int(t // self.step_min))
        return arr3[min(int(t // self.step_min), arr3.shape[0] - 1)]

    def emit(self, rows, cols, times, i_mw, n):
        """Throw n[k] embers from cell (rows[k], cols[k]) at times[k]; i_mw is the intensity scale in MW/m."""
        n = np.asarray(n)
        if n.sum() == 0:
            return
        rep = np.repeat(np.arange(len(n)), n)
        r, c, t, i = rows[rep], cols[rep], times[rep], np.maximum(i_mw[rep], 1e-6)
        u = np.maximum(self._s(float(t.min()), self.wind)[r, c], 0.5)
        th = self._s(float(t.min()), self.wdir)[r, c]
        p = self.p
        if self.fixed is not None:
            d = np.full(len(rep), float(self.fixed[0])); tl = t + self.fixed[1]
        else:
            d = p["d_med"] * (i / 10.0) ** p["a_i"] * (u / 10.0) ** p["b_u"] * np.exp(p["sigma"] * self.rng.standard_normal(len(rep)))
            th = th + p["sig_theta"] * self.rng.standard_normal(len(rep))
            tl = t + p["loft_min"] + d / u / 60.0
        self.Lt = np.concatenate([self.Lt, tl])
        self.Lx = np.concatenate([self.Lx, c * self.cell + d * np.cos(th)])
        self.Ly = np.concatenate([self.Ly, r * self.cell + d * np.sin(th)])
        self.n_thrown += len(rep)

    def __call__(self, t0, t1, arr, phi, rows, cols):
        p = self.p
        i_mw = self._s(t0, self.fli)[rows, cols] / 1000.0
        if self.fixed is not None:
            self.emit(rows, cols, arr[rows, cols], i_mw, np.ones(len(rows), int))
        elif p["gen"] > 0 and len(rows):
            lam = np.where(i_mw * 1000.0 >= p["fli_min"], p["gen"] * i_mw * self.cell ** 2, 0.0)
            self.emit(rows, cols, arr[rows, cols], i_mw, self.rng.poisson(lam))
        if self.st is not None:
            self.st.step(t0, t1, rows, cols, i_mw)
        due = self.Lt <= t1
        if not due.any():
            return None
        lt, lx, ly = self.Lt[due], self.Lx[due], self.Ly[due]
        self.Lt, self.Lx, self.Ly = self.Lt[~due], self.Lx[~due], self.Ly[~due]
        if self.st is not None and self.st.n:
            hit = self.st.ember_hits(lx, ly, lt)
            lt, lx, ly = lt[~hit], lx[~hit], ly[~hit]
        r = np.round(ly / self.cell).astype(int); c = np.round(lx / self.cell).astype(int)
        ny, nx = self.burnable.shape
        ok = (r >= 0) & (r < ny) & (c >= 0) & (c < nx)
        r, c, lt = r[ok], c[ok], lt[ok]
        ok = self.burnable[r, c] & ~np.isfinite(arr[r, c])
        if self.fixed is None:
            ok &= self.rng.random(len(r)) < p["p_ign"]
        r, c, lt = r[ok], c[ok], lt[ok]
        if not len(r):
            return None
        o = np.argsort(lt, kind="stable")
        _, first = np.unique(r[o] * nx + c[o], return_index=True)
        k = o[first]
        self.n_spots += len(k)
        return [(int(a), int(b), float(tt), 0.0) for a, b, tt in zip(r[k], c[k], lt[k])]
