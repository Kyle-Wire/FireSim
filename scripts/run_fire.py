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
ap.add_argument("--max-minutes", type=float, default=110.0, help="wall-clock cap for the whole script; it stops itself")
ap.add_argument("--wind", default="stations"); ap.add_argument("--gust-share", type=float, default=0.0)
ap.add_argument("--wind-scale", type=float, default=1.0); ap.add_argument("--cell", type=float, default=10.0); ap.add_argument("--params")
ap.add_argument("--ember-wind", default="surface", choices=["surface", "80", "850", "blend"], help="protocol A10: wind that carries embers (HRRR only for 80 m and 850 hPa)")
ap.add_argument("--loft-mw", type=float, default=20.0, help="protocol A11, with --ember-wind blend: fireline intensity (MW/m) at which embers ride the 850 hPa wind fully")
ap.add_argument("--windsim", nargs="+", help="protocol A15, wind option W3: WindSim terrain_wind.npz files (one per inflow bearing); needs --wind hrrr")
ap.add_argument("--road-cross", type=float, help="protocol A8 road rule R1: flame length (m) at which thin road lines burn; omitted = rule off")
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
if a.windsim:
    fire.use_windsim(a.windsim)
if a.ember_wind != "surface":
    fire.use_ember_wind("850" if a.ember_wind == "blend" else a.ember_wind)
if a.road_cross is not None:
    fire.use_roads(a.road_cross)
    print(f"road rule R1 on: {int(fire.thin.sum())} thin cells, L_cross {a.road_cross} m", flush=True)
params = json.load(open(a.params)) if a.params else {}
print(f"grid {fire.ny} x {fire.nx} at {fire.cell} m, grid north is {fire.convergence:.1f} deg clockwise of true north", flush=True)
for h in range(0, fire.hours, 3):
    sp, to = fire.wind(h); t, rh, stick = fire.met[h]; m = fire.moist[h]
    print(f"hour {h:2d}: wind 20 ft mean {sp.mean():.1f} max {sp.max():.1f} m/s, temp {t:.0f} F, RH {rh:.0f} %, dead moisture {m[0]*100:.1f}/{m[1]*100:.1f}/{m[2]*100:.1f} %", flush=True)
bx, by, barea, label, extra = fire.buildings()
print(f"buildings {len(bx)}", flush=True)
kw = dict(wind_scale=a.wind_scale)
prov = fire.provider(**kw)
burnable = (fire.slice(0, **kw)["rate"] > 0) & ~np.repeat(np.repeat(fire.thin, fire.k, 0), fire.k, 1)   # embers cannot light a road
lost = np.zeros(len(bx)); t_sum = np.zeros(len(bx)); causes = np.zeros(5); arr0 = None; stats = []
pers = fire.perimeters()
pers = [p for p in pers if p[0] <= T + 1e-6]                  # only outlines inside the run window
sat = [p for p in pers if p[2] != "GeoMAC"] or pers          # A7: the latest satellite (else incident) perimeter in the window
fit_per = max(sat, key=lambda p: p[0]) if sat else None
share = np.zeros((fire.ny30, fire.nx30))
blend = a.ember_wind == "blend"
ewind = (fire.lookup("wind", **kw), fire.lookup("wind_to", **kw)) if blend else (fire.lookup("ewind", **kw), fire.lookup("ewind_to", **kw))
ekw = dict(aloft=(fire.lookup("ewind", **kw), fire.lookup("ewind_to", **kw)), i_loft=a.loft_mw) if blend else {}
stopped = None
phi0 = fire.start_phi()
for seed in range(a.seeds):
    t1 = time.time()
    if a.candidate == "M1":
        st = Structures(bx, by, barea, fire.cell, seed=seed, k_s=0.0, brands=0.0, **params.get("houses", {}))
        sp = Spot(fire.cell, fire.lookup("fli", **kw), *ewind,
                  burnable, seed=seed, structures=st, gen=0.0, **ekw)
    else:
        st = Structures(bx, by, barea, fire.cell, seed=seed, **params.get("houses", {}))
        sp = Spot(fire.cell, fire.lookup("fli", **kw), *ewind,
                  burnable, seed=seed, structures=st, **ekw, **params.get("embers", {}))
    def watched(t0_, t1_, arr_, phi_, r_, c_, sp=sp, state={"h": -1}):
        if time.time() - t_start > a.max_minutes * 60:
            raise TimeoutError(f"stopped by --max-minutes {a.max_minutes} at simulated hour {t1_ / 60:.1f} of seed {seed}")
        if int(t1_ // 60) > state["h"]:
            state["h"] = int(t1_ // 60)
            print(f"  seed {seed} hour {state['h']:2d}: {time.time() - t1:6.0f} s, embers {sp.n_thrown}, spot fires {sp.n_spots}", flush=True)
        return sp(t0_, t1_, arr_, phi_, r_, c_)
    try:
        if phi0 is None:
            arr, _ = arrival(prov, fire.hours, None, fire.cell, fire.ignitions(), T, spot=watched)
        else:                                              # A18: lit over the whole start outline at minute 0
            arr, _ = arrival(prov, fire.hours, None, fire.cell, [], T, spot=watched, phi0=phi0)
            arr[phi0 <= 0] = 0.0
    except TimeoutError as e:                              # keep the seeds that finished; say so in the result
        print(e, flush=True)
        if seed == 0:
            raise SystemExit("no seed finished; nothing written")
        stopped = f"{e}; scored on the {seed} seed(s) that finished, {a.seeds} were asked for"; a.seeds = seed
        break
    l = np.isfinite(st.t_ign); lost += l; t_sum[l] += st.t_ign[l]; causes += np.bincount(st.cause[l], minlength=5)
    stats.append(dict(seed=seed, seconds=round(time.time() - t1, 1), burned_km2=float(np.isfinite(arr).sum() * fire.cell ** 2 / 1e6),
                      houses_lit=int(l.sum()), embers=int(sp.n_thrown), spot_fires=int(sp.n_spots)))
    print(stats[-1], flush=True)
    if fit_per is not None:
        b30 = arr[fire.k // 2::fire.k, fire.k // 2::fire.k] <= fit_per[0]
        share += b30; stats[-1]["jaccard_fit_perimeter"] = overlap(b30, fit_per[1])[0]
        print("  overlap at fit perimeter", stats[-1]["jaccard_fit_perimeter"], flush=True)
    if arr0 is None:
        arr0 = arr
    if True:                                               # every seed's arrival (A19: rescoring needs all seeds)
        np.savez_compressed(os.path.join(a.out, f"arrival_seed{seed}.npz"), arrival_min=arr[fire.k // 2::fire.k, fire.k // 2::fire.k].astype(np.float32),
                            left=fire.left, bottom=fire.bottom, res=fire.res, crs=str(fire.crs))
p_model = lost / a.seeds
# ---- scoring: nothing below feeds back into the model
k = fire.k; a30 = arr0[k // 2::k, k // 2::k]
per = []
for mins, mask, name in pers:
    j, s = overlap(a30 <= mins, mask)
    per.append(dict(source=name, hours=round(mins / 60, 2), observed_km2=float(mask.sum() * fire.res ** 2 / 1e6),
                    model_km2=float((a30 <= mins).sum() * fire.res ** 2 / 1e6), jaccard=j, sorensen=s))
    print(per[-1], flush=True)
det = []
for mins, dr, dc in fire.detections():                     # A11 first-night check, seed 0: detections on or within 190 m of modelled fire
    b = a30 <= mins; w = 6
    pad = np.pad(b, w)
    near = np.array([pad[r:r + 2 * w + 1, c:c + 2 * w + 1].any() for r, c in zip(dr, dc)])
    det.append(dict(hours=round(mins / 60, 2), detections=int(len(dr)), share_on_modelled_fire=float(near.mean()), model_km2=float(b.sum() * fire.res ** 2 / 1e6)))
    print("detections", det[-1], flush=True)
res = dict(fire=fire.cfg["name"], role=fire.cfg["role"], candidate=a.candidate, firesim=__version__, commit=commit, dirty=dirty, seeds=a.seeds, stopped_early=stopped,
           hours=T / 60, cell=fire.cell, wind_scale=a.wind_scale, wind_source=a.wind, gust_share=a.gust_share, road_cross=a.road_cross, windsim=[os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(p)))) for p in a.windsim] if a.windsim else None,
           windsim_hours=dict(used=fire._ws.hours_used, kept_hrrr=fire._ws.hours_kept) if a.windsim else None, ember_wind=a.ember_wind, loft_mw=a.loft_mw if blend else None, hrrr_vs_stations=wind_check, params=params, convergence_deg=fire.convergence,
           weather=[dict(hour=h, temp_f=fire.met[h][0], rh=fire.met[h][1], stick=fire.met[h][2], moist=fire.moist[h]) for h in range(fire.hours)],
           runs=stats, perimeters=per, detections=det, seconds=round(time.time() - t_start, 1))
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
    if fit_per is not None:                                    # protocol A7 fitting score
        q = np.clip(share / a.seeds, 0.005, 0.995); m = fit_per[1]
        area_bits = float(-(np.log2(q[m]).sum() + np.log2(1 - q[~m]).sum()))
        res["fit"] = dict(perimeter=fit_per[2], hours=fit_per[0] / 60, area_bits_per_building=area_bits / int(known.sum()),
                          score=bits + area_bits / int(known.sum()), note="protocol A7: bits per building + area bits / buildings; lower is better")
        fb = float(np.clip(m.mean(), 0.005, 0.995)); cell0 = float(-(m.mean() * np.log2(fb) + (1 - m.mean()) * np.log2(1 - fb)))
        res["fit"].update(bits_per_cell=area_bits / m.size, bits_per_cell_no_skill=cell0, building_ratio=bits / bits0, area_ratio=area_bits / m.size / cell0,
                          score_a8=bits / bits0 + area_bits / m.size / cell0,
                          note_a8="protocol A8: building bits / own-rate bits + cell bits / mapped-share bits; 2.0 = no skill, lower is better")
        print("fit", res["fit"], flush=True)
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
