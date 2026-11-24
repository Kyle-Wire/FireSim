"""Houses: ignition by flame, by embers and by a burning neighbour.

Provenance: written fresh for FireSim. Houses are points with a footprint area (a disc of the same area). They do
not burn by the level set. A house lights in one of three ways:

  flame      each step, burning cells within `r_f` of the wall add exposure E = sum I (MW/m) exp(-d/d_f) cell^2/d_f^2;
             the chance of lighting over the fire's passage is 1 - exp(-k_f s_flame E)
  ember      each effective ember landing on the footprint lights it with chance p_e s_ember
  neighbour  when house i lights, each unlit house j within `r_s` (edge to edge distance d) lights with chance
             1 - exp(-k_s s_flame_j exp(-d/d_s)), `tau_s` minutes later; a burning house also throws `brands` embers

`s_ember` and `s_flame` are per-house factors (1 = ordinary house). They are how hardening enters: roof, vents and
the 0-5 ft zone lower s_ember; siding, fences and attachments lower s_flame; spacing enters through d.

Every number in `PRIORS` is a starting guess to be CALIBRATED on the calibration fires (gridzip-lab
`physics/PROTOCOL-FIRE.md`). None is a sourced constant.
"""
import numpy as np
from scipy.spatial import cKDTree

PRIORS = dict(
    k_s=1.5, d_s=10.0, r_s=50.0, tau_s=15.0,       # neighbour: strength, e-fold distance m, cutoff m, delay min
    k_f=0.5, d_f=10.0, r_f=40.0,                    # flame: strength, e-fold distance m, cutoff m
    p_e=0.05, tau_e=10.0,                           # ember: chance per effective ember on the footprint, delay min
    brands=20.0, i_b=5.0, tau_b=60.0,               # embers thrown by a burning house, their intensity scale MW/m, burn time min
    u_ref=0.0, p_u=2.0,                             # rule H1 (protocol A12): k_s and brands x min(1, U / u_ref)^p_u; u_ref 0 = off
)
FLAME, EMBER, NEIGHBOUR, GIVEN = 1, 2, 3, 4


class Structures:
    def __init__(self, x, y, area, cell, seed=0, s_ember=1.0, s_flame=1.0, lit=(), **kw):
        bad = set(kw) - set(PRIORS)
        if bad:
            raise TypeError(f"unknown house parameters {bad}")
        self.p = {**PRIORS, **kw}
        self.x = np.asarray(x, np.float64); self.y = np.asarray(y, np.float64); self.n = len(self.x)
        self.rad = np.sqrt(np.broadcast_to(np.asarray(area, np.float64), (self.n,)) / np.pi)
        self.cell = float(cell)
        self.s_e = np.broadcast_to(np.asarray(s_ember, np.float64), (self.n,)).copy()
        self.s_f = np.broadcast_to(np.asarray(s_flame, np.float64), (self.n,)).copy()
        self.rng = np.random.default_rng(seed + 7919)
        self.t_ign = np.full(self.n, np.inf); self.cause = np.zeros(self.n, np.int8)
        self.t_sched = np.full(self.n, np.inf); self.c_sched = np.zeros(self.n, np.int8)
        self.E = np.zeros(self.n)
        self.embers = None
        for i, t in lit:
            self.t_sched[i] = t; self.c_sched[i] = GIVEN
        if self.n:
            self.tree = cKDTree(np.c_[self.x, self.y])
            self.rmax = float(self.rad.max())

    def _sched(self, j, t, cause):
        better = t < self.t_sched[j]
        self.t_sched[j] = np.where(better, t, self.t_sched[j]); self.c_sched[j] = np.where(better, cause, self.c_sched[j])

    def ember_hits(self, lx, ly, lt):
        """Embers landing at (lx, ly) at times lt. Returns the mask of embers that fell on a footprint."""
        d, j = self.tree.query(np.c_[lx, ly], k=1)
        hit = d <= self.rad[j]
        jj, tt = j[hit], lt[hit]
        ok = (self.rng.random(len(jj)) < self.p["p_e"] * self.s_e[jj]) & ~np.isfinite(self.t_ign[jj])
        for a, b in zip(jj[ok], tt[ok]):
            self._sched(a, b + self.p["tau_e"], EMBER)
        return hit

    def step(self, t0, t1, rows, cols, i_mw):
        p = self.p
        if not self.n:
            return
        if len(rows):
            pts = np.c_[cols * self.cell, rows * self.cell]
            near = self.tree.query_ball_point(pts, p["r_f"] + self.rmax)
            k = np.repeat(np.arange(len(pts)), [len(v) for v in near])
            if len(k):
                j = np.concatenate([np.asarray(v, int) for v in near if len(v)])
                d = np.maximum(np.hypot(pts[k, 0] - self.x[j], pts[k, 1] - self.y[j]) - self.rad[j], 0.0)
                e = np.where(d <= p["r_f"], i_mw[k] * np.exp(-d / p["d_f"]) * self.cell ** 2 / p["d_f"] ** 2, 0.0)
                de = np.bincount(j, e, self.n)
                self.E += de
                cand = np.nonzero((de > 0) & ~np.isfinite(self.t_ign) & (p["k_f"] > 0))[0]
                lit = cand[self.rng.random(len(cand)) < 1 - np.exp(-p["k_f"] * self.s_f[cand] * de[cand])]
                self._sched(lit, t1, FLAME)
        while True:
            due = np.nonzero((self.t_sched <= t1) & ~np.isfinite(self.t_ign))[0]
            if not len(due):
                break
            i = due[np.argmin(self.t_sched[due])]
            ti = self.t_sched[i]
            self.t_ign[i] = ti; self.cause[i] = self.c_sched[i]
            wf = 1.0
            if p["u_ref"] > 0 and self.embers is not None:     # H1: the 20-ft wind at this house when it lights
                e = self.embers; ny, nx = e.burnable.shape
                ri = min(max(int(round(self.y[i] / self.cell)), 0), ny - 1); ci = min(max(int(round(self.x[i] / self.cell)), 0), nx - 1)
                u = float(np.asarray(e._s(ti, e.wind)[np.array([ri]), np.array([ci])])[0])
                wf = min(1.0, max(u, 0.0) / p["u_ref"]) ** p["p_u"]
            if p["k_s"] > 0:
                nb = np.asarray(self.tree.query_ball_point([self.x[i], self.y[i]], p["r_s"] + self.rad[i] + self.rmax), int)
                nb = nb[~np.isfinite(self.t_ign[nb])]
                d = np.maximum(np.hypot(self.x[nb] - self.x[i], self.y[nb] - self.y[i]) - self.rad[nb] - self.rad[i], 0.0)
                nb, d = nb[d <= p["r_s"]], d[d <= p["r_s"]]
                got = nb[self.rng.random(len(nb)) < 1 - np.exp(-p["k_s"] * wf * self.s_f[nb] * np.exp(-d / p["d_s"]))]
                self._sched(got, ti + p["tau_s"], NEIGHBOUR)
            if self.embers is not None and p["brands"] > 0 and self.embers.fixed is None:
                n = self.rng.poisson(p["brands"] * wf)
                if n:
                    r = np.full(n, int(round(self.y[i] / self.cell))); c = np.full(n, int(round(self.x[i] / self.cell)))
                    ny, nx = self.embers.burnable.shape
                    r = np.clip(r, 0, ny - 1); c = np.clip(c, 0, nx - 1)
                    self.embers.emit(r, c, ti + self.rng.random(n) * p["tau_b"], np.full(n, p["i_b"]), np.ones(n, int))
