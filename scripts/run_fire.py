"""Run one real fire and score it against its perimeters and building record.

python scripts/run_fire.py fires/camp_2018.json <data_root> <out_dir> [--candidate M1|M3] [--seeds N] [--hours H]
                           [--wind-scale 1.0] [--cell 10]

Candidates (gridzip-lab physics/PROTOCOL-FIRE.md): M1 = spread only (no embers, houses lit by flame only);
M3 = embers, house ignition and house-to-house spread. Ember and house numbers are the uncalibrated starting guesses
unless --params file.json gives others. Building labels are read only after the model has finished.
"""
import argparse, json, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from firesim import __version__
from firesim.levelset import arrival
from firesim.embers import Spot
from firesim.structures import Structures
from firesim.real import Fire, overlap

ap = argparse.ArgumentParser()
ap.add_argument("cfg"); ap.add_argument("data"); ap.add_argument("out")
ap.add_argument("--candidate", default="M1"); ap.add_argument("--seeds", type=int, default=1); ap.add_argument("--hours", type=float)
ap.add_argument("--wind", default="stations"); ap.add_argument("--gust-share", type=float, default=0.0)
ap.add_argument("--wind-scale", type=float, default=1.0); ap.add_argument("--cell", type=float, default=10.0); ap.add_argument("--params")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip())
t_start = time.time()
fire = Fire(a.cfg, a.data, a.cell)
T = (a.hours or fire.hours) * 60.0
wind_check = None
if a.wind == "hrrr":
    fire.use_hrrr(gust_share=a.gust_share)
    wind_check = fire.hrrr_vs_stations()
    for r in wind_check:
        print("HRRR against station:", r, flush=True)
params = json.load(open(a.params)) if a.params else {}
print(f"grid {fire.ny} x {fire.nx} at {fire.cell} m, grid north is {fire.convergence:.1f} deg clockwise of true north", flush=True)
for h in range(0, fire.hours, 3):
    sp, to = fire.wind(h); t, rh, stick = fire.met[h]; m = fire.moist[h]
    print(f"hour {h:2d}: wind 20 ft mean {sp.mean():.1f} max {sp.max():.1f} m/s, temp {t:.0f} F, RH {rh:.0f} %, dead moisture {m[0]*100:.1f}/{m[1]*100:.1f}/{m[2]*100:.1f} %", flush=True)
bx, by, barea, label, extra = fire.buildings()
print(f"buildings {len(bx)}", flush=True)
kw = dict(wind_scale=a.wind_scale)
prov = fire.provider(**kw)
burnable = fire.slice(0, **kw)["rate"] > 0
lost = np.zeros(len(bx)); t_sum = np.zeros(len(bx)); causes = np.zeros(5); arr0 = None; stats = []
for seed in range(a.seeds):
    t1 = time.time()
    if a.candidate == "M1":
        st = Structures(bx, by, barea, fire.cell, seed=seed, k_s=0.0, brands=0.0, **params.get("houses", {}))
        sp = Spot(fire.cell, lambda s: fire.slice(s, **kw)["fli"], lambda s: fire.slice(s, **kw)["wind"], lambda s: fire.slice(s, **kw)["wind_to"],
                  burnable, seed=seed, structures=st, gen=0.0)
    else:
        st = Structures(bx, by, barea, fire.cell, seed=seed, **params.get("houses", {}))
        sp = Spot(fire.cell, lambda s: fire.slice(s, **kw)["fli"], lambda s: fire.slice(s, **kw)["wind"], lambda s: fire.slice(s, **kw)["wind_to"],
                  burnable, seed=seed, structures=st, **params.get("embers", {}))
    arr, _ = arrival(prov, fire.hours, None, fire.cell, fire.ignitions(), T, spot=sp)
    l = np.isfinite(st.t_ign); lost += l; t_sum[l] += st.t_ign[l]; causes += np.bincount(st.cause[l], minlength=5)
    stats.append(dict(seed=seed, seconds=round(time.time() - t1, 1), burned_km2=float(np.isfinite(arr).sum() * fire.cell ** 2 / 1e6),
                      houses_lit=int(l.sum()), embers=int(sp.n_thrown), spot_fires=int(sp.n_spots)))
    print(stats[-1], flush=True)
    if arr0 is None:
        arr0 = arr
        np.savez_compressed(os.path.join(a.out, "arrival_seed0.npz"), arrival_min=arr[fire.k // 2::fire.k, fire.k // 2::fire.k].astype(np.float32),
                            left=fire.left, bottom=fire.bottom, res=fire.res, crs=str(fire.crs))
p_model = lost / a.seeds
# ---- scoring: nothing below feeds back into the model
k = fire.k; a30 = arr0[k // 2::k, k // 2::k]
per = []
for mins, mask, name in fire.perimeters():
    j, s = overlap(a30 <= mins, mask)
    per.append(dict(source=name, hours=round(mins / 60, 2), observed_km2=float(mask.sum() * fire.res ** 2 / 1e6),
                    model_km2=float((a30 <= mins).sum() * fire.res ** 2 / 1e6), jaccard=j, sorensen=s))
    print(per[-1], flush=True)
res = dict(fire=fire.cfg["name"], role=fire.cfg["role"], candidate=a.candidate, firesim=__version__, commit=commit, dirty=dirty, seeds=a.seeds,
           hours=T / 60, cell=fire.cell, wind_scale=a.wind_scale, wind_source=a.wind, gust_share=a.gust_share, hrrr_vs_stations=wind_check, params=params, convergence_deg=fire.convergence,
           weather=[dict(hour=h, temp_f=fire.met[h][0], rh=fire.met[h][1], stick=fire.met[h][2], moist=fire.moist[h]) for h in range(fire.hours)],
           runs=stats, perimeters=per, seconds=round(time.time() - t_start, 1))
known = label >= 0
if known.any():
    y = label[known]; p = np.clip(p_model[known], 0.005, 0.995)
    bits = float(-(y * np.log2(p) + (1 - y) * np.log2(1 - p)).mean())
    base = float(y.mean()); pb = np.clip(base, 0.005, 0.995)
    bits0 = float(-(y * np.log2(pb) + (1 - y) * np.log2(1 - pb)).mean())
    o = np.argsort(p_model[known], kind="stable"); r = np.empty(len(o)); r[o] = np.arange(1, len(o) + 1)
    # rank AUC with ties averaged
    pm = p_model[known]; u, inv = np.unique(pm, return_inverse=True); mean_rank = np.bincount(inv, r) / np.bincount(inv); r = mean_rank[inv]
    n1 = y.sum(); n0 = len(y) - n1
    auc = float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else None
    res["buildings"] = dict(n=int(known.sum()), destroyed_observed=int(n1), destroyed_model_mean=float(p_model[known].sum()),
                            bits_per_building=bits, bits_same_fire_rate=bits0, auc=auc,
                            cause_share=dict(flame=causes[1] / max(causes.sum(), 1), ember=causes[2] / max(causes.sum(), 1), neighbour=causes[3] / max(causes.sum(), 1)),
                            note="bits_same_fire_rate uses this fire's own destroyed fraction (not baseline B0, which uses the calibration fires' pooled fraction)")
    print(res["buildings"], flush=True)
np.savez_compressed(os.path.join(a.out, "buildings.npz"), x=bx, y=by, p_model=p_model, label=label,
                    t_mean=np.where(lost > 0, t_sum / np.maximum(lost, 1), np.nan), **{k_: v for k_, v in extra.items()})
json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1, default=float)
# ---- chart
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ext = [0, fire.nx30 * fire.res / 1000, 0, fire.ny30 * fire.res / 1000]
    fig, ax = plt.subplots(1, 2, figsize=(17, 8), dpi=140)
    hrs = np.where(np.isfinite(a30), a30 / 60.0, np.nan)
    ax[0].imshow(np.where(burnable[k // 2::k, k // 2::k], 0.95, 0.8), cmap="gray", vmin=0, vmax=1, extent=ext, origin="lower")
    cf = ax[0].contourf(hrs, levels=np.linspace(0, T / 60, 9), cmap="inferno", extent=ext, origin="lower", alpha=.9)
    pers = fire.perimeters()
    if pers:
        mins, mask, name = max([p for p in pers if p[0] <= T + 90] or pers[:1], key=lambda p: p[0])
        ax[0].contour(mask.astype(float), levels=[0.5], colors=["#1e88e5"], linewidths=2, extent=ext, origin="lower")
        ax[0].plot([], [], color="#1e88e5", lw=2, label=f"mapped perimeter at {mins / 60:.0f} h ({name})")
    iy, ix = fire.ignitions()[0][:2]
    ax[0].plot(ix * fire.cell / 1000, iy * fire.cell / 1000, "w*", ms=14, mec="k", label="ignition")
    ax[0].legend(loc="upper left", fontsize=9); ax[0].set_title(f"Modelled fire arrival, first {T / 60:.0f} hours ({a.candidate})"); ax[0].set_xlabel("km"); ax[0].set_ylabel("km")
    fig.colorbar(cf, ax=ax[0], shrink=.7, label="hours after ignition")
    ax[1].imshow(np.where(burnable[k // 2::k, k // 2::k], 0.95, 0.8), cmap="gray", vmin=0, vmax=1, extent=ext, origin="lower")
    cat = np.where(label == 1, np.where(p_model >= 0.5, 0, 1), np.where(p_model >= 0.5, 2, 3))
    for c, col, lab in ((3, "#9e9e9e", "stood, model says stands"), (2, "#fbc02d", "stood, model says lost"), (1, "#1e88e5", "lost, model says stands"), (0, "#c62828", "lost, model says lost")):
        m = (cat == c) & (label >= 0)
        ax[1].scatter(bx[m] / 1000, by[m] / 1000, s=1.5, c=col, label=f"{lab} ({m.sum():,})", linewidths=0)
    ax[1].legend(loc="upper left", fontsize=9, markerscale=6); ax[1].set_title("Buildings: record against model"); ax[1].set_xlabel("km")
    ax[1].set_xlim(ax[0].get_xlim()); ax[1].set_ylim(ax[0].get_ylim()); ax[1].set_aspect("equal")
    fig.suptitle(f"{fire.cfg['name']} ({fire.cfg['role']} fire), FireSim {__version__} {commit[:7]}, {a.candidate}, {a.seeds} run(s). Wind: {a.wind}, gust share {a.gust_share}. No suppression modelled.", fontsize=12)
    fig.savefig(os.path.join(a.out, "map.png"), bbox_inches="tight")
except Exception as e:                                           # the chart never blocks the numbers
    print("chart failed:", repr(e))
print("done", res["seconds"], "s")
