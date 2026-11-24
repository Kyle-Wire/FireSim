"""Error tables of one finished run, for the compressor (gridzip-lab physics/PROTOCOL-FIRE.md, amendment A14).

python scripts/error_table.py <fire.json> <data root> <run folder> --out <table.npz>

Reads the run's result.json, arrival_seed0.npz and buildings.npz, the fire's outlines and satellite passes, and the
same inputs the model read. Writes one npz with three tables on a 300 m lattice:
  t12_*  points the record or the model shows burned by the last outline (T1 arrival and T2 reach)
  t3_*   300 m blocks with at least 5 scored buildings (no coordinates, addresses or parcel facts; a 3 km block id only)
Provenance: written fresh. Scoring only: nothing here feeds back into the model.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from firesim.landscape import thin_barriers  # noqa: E402
from firesim.real import Fire  # noqa: E402

EMBER_L = (100.0, 300.0, 1000.0, 3000.0)
GAP_D = (3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0)
GROUPS = {"grass": (101, 109), "shrub": (121, 149), "timber": (161, 204), "nonburn": (91, 99)}   # Scott and Burgan 2005 fuel model numbers


def observed_arrival(fire, T):
    """Earliest time (min) each 30 m cell is shown burned by an outline or a satellite pass (detection within 190 m)."""
    from scipy.ndimage import maximum_filter
    obs = np.full((fire.ny30, fire.nx30), np.inf); t_end = 0.0
    for mins, m, _ in fire.perimeters():
        if mins <= T + 1e-6:
            obs = np.minimum(obs, np.where(m, mins, np.inf)); t_end = max(t_end, mins)
    for mins, dr, dc in fire.detections():
        m = np.zeros(obs.shape, bool); m[dr, dc] = True
        obs = np.minimum(obs, np.where(maximum_filter(m, size=13), mins, np.inf))
    return obs, (t_end or T)


def hourly(fire, rows, cols, n_hours, has850):
    """Per hour and point: 10 m speed (m/s), wind angle (solver frame, blown toward), 850 hPa speed and angle."""
    u = np.zeros((n_hours, len(rows))); to = np.zeros_like(u); ua = np.full_like(u, np.nan); toa = np.full_like(u, np.nan)
    for h in range(n_hours):
        sp, a = fire.wind(h); u[h] = sp[rows, cols] * 1.15; to[h] = a[rows, cols]          # 20 ft back to 10 m
        if has850:
            sp, a = fire.wind(h, level="850"); ua[h] = sp[rows, cols]; toa[h] = a[rows, cols]
    return u, to, ua, toa


def columns(fire, rows, cols, t_obs, t_end, hrrr_path):
    """Term inputs at 30 m cells (rows, cols). Suffix _to: over the hours before the observed arrival (the whole
    window where there is none); _all: over the whole window to the last outline."""
    from scipy.ndimage import distance_transform_edt
    res = fire.res; n = len(rows); out = {}
    n_all = int(np.clip(np.ceil(t_end / 60.0), 1, fire.hours))
    n_to = np.where(np.isfinite(t_obs), np.clip(np.ceil(t_obs / 60.0), 1, n_all), n_all).astype(int)
    has850 = "ue850" in getattr(fire, "_hrrr", {})
    u, to, ua, toa = hourly(fire, rows, cols, n_all, has850)
    hh = np.arange(n_all)[:, None]
    lon0 = fire.cfg["ignition"]["lon"]
    solar = (fire.t0.hour + fire.t0.minute / 60.0 + np.arange(n_all) + 0.5 + lon0 / 15.0) % 24.0
    night = ((solar < 6.0) | (solar >= 18.0)).astype(float)[:, None] * np.ones((1, n))
    crew = ((np.arange(n_all) >= 9)[:, None] & (night < 0.5)).astype(float)
    ig_r, ig_c = [v // fire.k for v in fire.ignitions()[0][:2]]
    dy = (rows - ig_r) * res; dx = (cols - ig_c) * res
    dist = np.maximum(np.hypot(dx, dy), res); bearing = np.arctan2(dy, dx)
    moist = np.array([fire.moist[h][0] for h in range(n_all)])[:, None] * np.ones((1, n))
    rh = np.array([fire.met[h][1] for h in range(n_all)])[:, None] * np.ones((1, n))

    def descent(D):                                             # fall of the ground along the 850 hPa wind over D metres
        z0 = fire.elev[rows, cols]; d = np.zeros((n_all, n))
        for h in range(n_all):
            r2 = np.clip(np.rint(rows - D * np.sin(toa[h]) / res), 0, fire.ny30 - 1).astype(int)
            c2 = np.clip(np.rint(cols - D * np.cos(toa[h]) / res), 0, fire.nx30 - 1).astype(int)
            d[h] = fire.elev[r2, c2] - z0
        return d

    series = dict(u10=u, calm=(u < 2.0).astype(float), night=night, crew=crew, moist=moist, rh=rh,
                  run=u * np.cos(to - bearing[None, :]))
    if has850:
        series.update(u850=ua, lee1=descent(1000.0), lee5=descent(5000.0))
    for suf, nn in (("_to", n_to), ("_all", np.full(n, n_all))):
        w = (hh < nn[None, :]).astype(float); ws = w.sum(0)
        for k, v in series.items():
            out[k + suf] = (v * w).sum(0) * (1.0 if k == "crew" else 1.0 / ws)
        out["run" + suf] = out["run" + suf] / np.maximum(out["u10" + suf], 0.1)
    out["dist"] = dist
    # ---- along the straight line from the ignition
    cls = {k: (fire.fuel >= lo) & (fire.fuel <= hi) for k, (lo, hi) in GROUPS.items()}; cls["developed"] = fire.fuel == 91
    t2 = fire.tan ** 2
    line = {k: np.zeros(n) for k in list(cls) + ["canopy", "slope_run", "slope_any"]}
    for i in range(n):
        m = int(max(abs(rows[i] - ig_r), abs(cols[i] - ig_c))) + 1
        yy = np.rint(np.linspace(ig_r, rows[i], m)).astype(int); xx = np.rint(np.linspace(ig_c, cols[i], m)).astype(int)
        for k in cls:
            line[k][i] = cls[k][yy, xx].mean()
        line["canopy"][i] = fire.cc[yy, xx].mean()
        line["slope_run"][i] = (t2[yy, xx] * np.cos(fire.upslope[yy, xx] - bearing[i])).mean()
        line["slope_any"][i] = t2[yy, xx].mean()
    out.update({"line_" + k: v for k, v in line.items()})
    # ---- embers: canopy fuel upwind (whole-window mean 10 m wind) weighted exp(-x / L)
    ux = (u * np.cos(to)).mean(0); uy = (u * np.sin(to)).mean(0); ang = np.arctan2(uy, ux)
    canopy = (fire.cbd > 0).astype(float)
    xs = np.arange(res, 3 * max(EMBER_L) + res, res)
    rr = np.clip(np.rint(rows[:, None] - xs[None, :] * np.sin(ang)[:, None] / res), 0, fire.ny30 - 1).astype(int)
    cc = np.clip(np.rint(cols[:, None] - xs[None, :] * np.cos(ang)[:, None] / res), 0, fire.nx30 - 1).astype(int)
    up = canopy[rr, cc]
    out["ember"] = np.stack([(up * np.exp(-xs / L)).sum(1) / np.exp(-xs / L).sum() for L in EMBER_L], 1)
    # ---- roads, terrain against the weather model's terrain
    thin = thin_barriers(fire.fuel)[0]
    out["road_dist"] = distance_transform_edt(~thin)[rows, cols] * res if thin.any() else np.full(n, np.nan)
    out["elev"] = fire.elev[rows, cols]
    d = np.load(hrrr_path) if hrrr_path and os.path.exists(hrrr_path) else None
    if d is not None and "hgt_m" in d.files and getattr(fire, "_hrrr", None):
        hg = d["hgt_m"]; hg = hg[0] if hg.ndim == 3 else hg
        fire._hrrr["tri"].values = hg.ravel()[:, None].astype(np.float64)
        out["dz_model"] = out["elev"] - fire._hrrr["tri"]((cols + 0.5) * res, (rows + 0.5) * res)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg"); ap.add_argument("data"); ap.add_argument("run"); ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=10, help="lattice step in 30 m cells (10 = 300 m)")
    a = ap.parse_args()
    res = json.load(open(os.path.join(a.run, "result.json")))
    fire = Fire(a.cfg, a.data, res["cell"])
    hrrr_path = os.path.join(fire.dir, "weather", "hrrr_f00.npz")
    if res["wind_source"] == "hrrr":
        fire.use_hrrr(gust_share=res["gust_share"])
    T = res["hours"] * 60.0
    a30 = np.load(os.path.join(a.run, "arrival_seed0.npz"))["arrival_min"].astype(np.float64)
    obs, t_end = observed_arrival(fire, T)
    s = a.step
    R, C = np.meshgrid(np.arange(s // 2, fire.ny30, s), np.arange(s // 2, fire.nx30, s), indexing="ij")
    R, C = R.ravel(), C.ravel()
    keep = ((obs[R, C] <= t_end) | (a30[R, C] <= t_end)) & (fire.fuel[R, C] != 98)
    R, C = R[keep], C[keep]
    out = {"t12_" + k: v for k, v in columns(fire, R, C, obs[R, C], t_end, hrrr_path).items()}
    bx, by, p_model, label = (np.load(os.path.join(a.run, "buildings.npz"))[k] for k in ("x", "y", "p_model", "label"))
    b30r = np.clip(np.rint(by / fire.res - 0.5 + 0.5 * fire.cell / fire.res), 0, fire.ny30 - 1).astype(int)
    b30c = np.clip(np.rint(bx / fire.res - 0.5 + 0.5 * fire.cell / fire.res), 0, fire.nx30 - 1).astype(int)
    from scipy.ndimage import uniform_filter
    cnt = np.zeros((fire.ny30, fire.nx30)); np.add.at(cnt, (b30r, b30c), 1.0)
    near = uniform_filter(cnt, size=7, mode="constant") * 49.0   # buildings within about 100 m (a 210 m box)
    out.update(t12_row=R, t12_col=C, t12_t_obs=obs[R, C], t12_t_mod=a30[R, C], t12_bld100=near[R, C])
    # ---- T3: 300 m blocks with at least 5 scored buildings
    from scipy.spatial import cKDTree
    nn = cKDTree(np.c_[bx, by]).query(np.c_[bx, by], k=2)[0][:, 1] if len(bx) > 1 else np.full(len(bx), np.inf)
    known = label >= 0
    blk = (b30r // s) * 100000 + (b30c // s)
    ids, inv = np.unique(blk[known], return_inverse=True)
    nb = np.bincount(inv); ok = nb >= 5
    mean = lambda v: (np.bincount(inv, v[known]) / nb)[ok]
    br = (ids[ok] // 100000) * s + s // 2; bc = (ids[ok] % 100000) * s + s // 2
    br = np.minimum(br, fire.ny30 - 1); bc = np.minimum(bc, fire.nx30 - 1)
    cols3 = columns(fire, br, bc, obs[br, bc], t_end, hrrr_path)
    out.update({"t3_" + k: v for k, v in cols3.items()})
    out.update(t3_n=nb[ok], t3_obs=mean(label.astype(float)), t3_mod=mean(p_model), t3_bld100=mean(near[b30r, b30c]),
               t3_gap=np.stack([mean((nn < d).astype(float)) for d in GAP_D], 1), t3_block3km=(br // 100) * 1000 + bc // 100)
    meta = dict(fire=fire.cfg["name"], run=os.path.basename(os.path.normpath(a.run)), firesim_commit=res.get("commit"), hours=res["hours"],
                t_end_min=t_end, step_m=s * fire.res, points=int(len(R)), blocks=int(ok.sum()), ember_L=EMBER_L, gap_d=GAP_D,
                outlines=[p[0] for p in fire.perimeters() if p[0] <= T + 1e-6], passes=[d[0] for d in fire.detections()],
                has_850="t12_u850_all" in out, has_model_terrain="t12_dz_model" in out)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, meta=json.dumps(meta), **out)
    both = np.isfinite(out["t12_t_obs"]) & (out["t12_t_obs"] <= t_end) & (out["t12_t_mod"] <= t_end)
    over = (out["t12_t_mod"] <= t_end) & ~(out["t12_t_obs"] <= t_end)
    print(json.dumps(meta)); print(f"points {len(R)}: both {int(both.sum())}, model only {int(over.sum())}, record only {int((~both & ~over).sum())}; blocks {int(ok.sum())}", flush=True)
    print("ALL_DONE rc=0", flush=True)


if __name__ == "__main__":
    main()
