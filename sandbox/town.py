"""A made-up town in the path of a wind-driven fire: embers, house ignition and house-to-house spread together.

Demonstration only. The ember and house numbers are the uncalibrated starting guesses in `embers.PRIORS` and
`structures.PRIORS`, so the loss counts here mean nothing about a real town. python sandbox/town.py <out_dir> [runs]
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from firesim.levelset import arrival
from firesim.landscape import fields
from firesim.embers import Spot
from firesim.structures import Structures
from firesim import synth

out = sys.argv[1] if len(sys.argv) > 1 else "."; runs = int(sys.argv[2]) if len(sys.argv) > 2 else 40
cell = 10.0; T = 150.0; wind = 10.0
g = synth.on_grid(synth.landscape(4), cell); n = g["n"]
fuel = g["fuel"].copy()
yy, xx = np.mgrid[:n, :n]
fuel[:, :60][fuel[:, :60] == 91] = 102                               # make sure the fire can start: grass on the west edge
town = (xx * cell > 2300) & (xx * cell < 3200) & (yy * cell > 1500) & (yy * cell < 2500)
fuel[town] = 91
rng = np.random.default_rng(0)
hx, hy = np.meshgrid(np.arange(2320, 3190, 28.0), np.arange(1520, 2490, 36.0)); hx = hx.ravel() + rng.uniform(-4, 4, hx.size); hy = hy.ravel() + rng.uniform(-4, 4, hy.size)
area = rng.uniform(110, 220, hx.size)
F = fields(fuel, (.04, .05, .07, .40, .70), wind, 0.0, g["tan"], g["upslope"], g["cc"], g["ch"], g["cbh"], g["cbd"])
ign = [(r, 3, 0, 0) for r in range(150, 250)]


def run(seed, s_ember=1.0, embers=True):
    st = Structures(hx, hy, area, cell, seed=seed, s_ember=s_ember, **({} if embers else dict(brands=0.0)))
    sp = Spot(cell, F["fli"], wind, 0.0, F["rate"] > 0, seed=seed, structures=st, **({} if embers else dict(gen=0.0)))
    a, _ = arrival(F["rate"], F["heading"], F["lb"], cell, ign, T, spot=sp)
    return a, st, sp


a0, st0, sp0 = run(0)
a_no, st_no, _ = run(0, embers=False)
lost = np.zeros(hx.size); lost_h = np.zeros(hx.size); cnt = []; cnt_h = []; causes = np.zeros(5)
for s in range(runs):
    _, st, _ = run(s); l = np.isfinite(st.t_ign); lost += l; cnt.append(int(l.sum())); causes += np.bincount(st.cause[l], minlength=5)
    _, st, _ = run(s, s_ember=0.3); l = np.isfinite(st.t_ign); lost_h += l; cnt_h.append(int(l.sum()))
res = dict(houses=int(hx.size), runs=runs, lost_mean=float(np.mean(cnt)), lost_p10=float(np.percentile(cnt, 10)), lost_p90=float(np.percentile(cnt, 90)),
           lost_mean_hardened=float(np.mean(cnt_h)), lost_no_embers_seed0=int(np.isfinite(st_no.t_ign).sum()),
           cause_share=dict(flame=causes[1] / causes.sum(), ember=causes[2] / causes.sum(), neighbour=causes[3] / causes.sum()),
           embers_thrown_seed0=int(sp0.n_thrown), spot_fires_seed0=int(sp0.n_spots))
json.dump(res, open(os.path.join(os.path.dirname(__file__), "results_town.json"), "w"), indent=1); print(res)

ext = [0, n * cell / 1000, 0, n * cell / 1000]
fig, ax = plt.subplots(1, 3, figsize=(17, 5.8), dpi=150)
bands = dict(levels=[0, 30, 60, 90, 120, 150], colors=["#7a1f12", "#b53a1a", "#dd6b24", "#f0a544", "#f9d98a"])
for k, (a, st, ttl) in enumerate(((a_no, st_no, "Embers off: houses at the edge light, then their neighbours"), (a0, st0, "Embers on, one run: spot fires ahead of the front"))):
    ax[k].imshow(np.where(F["rate"] > 0, 0.93, 0.8), cmap="gray", vmin=0, vmax=1, extent=ext, origin="lower")
    cf = ax[k].contourf(np.where(np.isfinite(a), a, np.nan), extent=ext, origin="lower", **bands)
    u = ~np.isfinite(st.t_ign)
    ax[k].scatter(hx[u] / 1000, hy[u] / 1000, s=5, c="#3b6fb6", label=f"standing ({u.sum()})")
    for c, col, lab in ((1, "#000000", "lit by flame"), (2, "#c2187a", "lit by embers"), (3, "#6a3d9a", "lit by a neighbour")):
        m = st.cause == c
        if m.any(): ax[k].scatter(hx[m] / 1000, hy[m] / 1000, s=7, c=col, label=f"{lab} ({m.sum()})")
    ax[k].set_title(ttl, fontsize=11); ax[k].legend(loc="lower left", fontsize=8, framealpha=.95); ax[k].set_xlabel("km (wind blows left to right, 10 m/s)")
fig.colorbar(cf, ax=ax[:2], shrink=.75, pad=.01, label="minutes after ignition")
sc = ax[2].scatter(hx / 1000, hy / 1000, s=16, c=lost / runs, cmap="magma_r", vmin=0, vmax=1)
ax[2].set_xlim(2.25, 3.25); ax[2].set_ylim(1.45, 2.55); ax[2].set_aspect("equal")
ax[2].set_title(f"Chance each house is lost, {runs} runs (town close-up)\naverage lost {np.mean(cnt):.0f} of {hx.size}; with ember-resistant houses {np.mean(cnt_h):.0f}", fontsize=11)
fig.colorbar(sc, ax=ax[2], shrink=.75, label="chance of loss"); ax[2].set_xlabel("km")
fig.suptitle("FireSim on a made-up town. Ember and house numbers are uncalibrated starting guesses: this shows the mechanics, not a forecast", fontsize=12)
fig.savefig(os.path.join(out, "05_synthetic_town.png"), bbox_inches="tight")
