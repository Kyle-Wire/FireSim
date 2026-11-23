"""Synthetic landscape cases Y1-Y5 (CASES.md). python sandbox/run_synth.py [out.json]"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from firesim.levelset import arrival
from firesim.landscape import fields
from firesim.rothermel import FUELS, spread
from firesim.crown import waf_unsheltered, length_to_width
from firesim import synth

D2L2 = (.06, .07, .08, .60, .90)
res = {}


def y1():
    cell = 5.0; ny, nx = 161, 241; cy, cx = 80, 60
    F = fields(np.full((ny, nx), 102), D2L2, 0.0, 0.0, 0.30, 0.0, crown=False)
    s = spread(FUELS["GR2"], D2L2, 0.0, 0.30)
    t_exact = 300.0 / s["ros"]
    a, _ = arrival(F["rate"], F["heading"], F["lb"], cell, [(cy, cx, 0, 0)], t_exact * 1.3)
    t = a[cy, cx + 60]
    return dict(t=t, exact=t_exact, rate=s["ros"], lb=float(F["lb"][0, 0]), ok=bool(abs(t / t_exact - 1) <= 0.04))


def y2():
    cell = 5.0; n = 401; c = 120
    F = fields(np.full((n, n), 102), D2L2, 6.0, 0.0, 0.30, np.pi / 2, crown=False)
    R = float(F["rate"][0, 0]); H = float(F["heading"][0, 0])
    a, _ = arrival(F["rate"], F["heading"], F["lb"], cell, [(c, c, 0, 0)], 30.0)
    yy, xx = np.nonzero(a <= 30.0)
    d = np.hypot(yy - c, xx - c) * cell; k = d.argmax()
    ang = np.arctan2(yy[k] - c, xx[k] - c)
    return dict(dist=d[k], exact_dist=R * 30, angle_deg=np.degrees(ang), exact_angle_deg=np.degrees(H),
                ok=bool(abs(np.degrees(ang - H)) <= 4 and abs(d[k] / (R * 30) - 1) <= 0.05))


def run(seed, cell, wind=6.0, moist=D2L2, T=60.0):
    ls = synth.landscape(seed)
    g = synth.on_grid(ls, cell)
    rng = np.random.default_rng(1000 + seed)
    wto = rng.uniform(0, 2 * np.pi)
    F = fields(g["fuel"], moist, wind, wto, g["tan"], g["upslope"], g["cc"], g["ch"], g["cbh"], g["cbd"])
    # ignition: the burnable block centre nearest a point 1.2 km upwind of the middle
    b = ls["block"]; nb = ls["fuel"].shape[0]
    by, bx = np.mgrid[:nb, :nb]
    tx = ls["size"] / 2 - 1200 * np.cos(wto); ty = ls["size"] / 2 - 1200 * np.sin(wto)
    ok = ls["fuel"] != 91
    d = np.hypot((bx + .5) * b - tx, (by + .5) * b - ty); d[~ok] = 1e9
    iy, ix = np.unravel_index(d.argmin(), d.shape)
    r, c = int((iy + .5) * b / cell), int((ix + .5) * b / cell)
    a, _ = arrival(F["rate"], F["heading"], F["lb"], cell, [(r, c, 0, 15.0)], T)
    return a, g, F


def y3():
    out = []
    for seed in range(12):
        ref, g5, _ = run(seed, 5.0)
        area5 = (ref <= 60).sum() * 25.0
        rng = np.random.default_rng(seed)
        by, bx = np.nonzero((ref <= 60) & (ref > 5))
        pick = rng.choice(len(by), min(200, len(by)), replace=False)
        py = (by[pick] + .5) * 5.0; px = (bx[pick] + .5) * 5.0; tref = ref[by[pick], bx[pick]]
        row = dict(seed=seed, area5_ha=area5 / 1e4)
        for cell in (10.0, 15.0, 30.0):
            a, _, _ = run(seed, cell)
            area = (a <= 60).sum() * cell ** 2
            t = a[(py / cell).astype(int), (px / cell).astype(int)]
            fin = np.isfinite(t)
            row[f"area_{int(cell)}"] = area / area5 - 1
            row[f"t_{int(cell)}"] = float(np.median(np.abs(t[fin] / tref[fin] - 1))) if fin.any() else None
            row[f"missed_{int(cell)}"] = float(1 - fin.mean())
        out.append(row); print("Y3", row, flush=True)
    a10 = np.abs([r["area_10"] for r in out]); t10 = [r["t_10"] for r in out]
    ok = np.median(a10) <= 0.05 and a10.max() <= 0.15 and np.median(t10) <= 0.05
    return dict(rows=out, median_area_10=float(np.median(a10)), worst_area_10=float(a10.max()), median_t_10=float(np.median(t10)),
                median_area_15=float(np.median(np.abs([r["area_15"] for r in out]))), median_area_30=float(np.median(np.abs([r["area_30"] for r in out]))),
                worst_area_30=float(np.abs([r["area_30"] for r in out]).max()), ok=bool(ok))


def y45():
    rows = []; ok4 = True; ok5 = True
    for seed in range(12):
        a4, g, _ = run(seed, 10.0, wind=4.0); a8, _, _ = run(seed, 10.0, wind=8.0); a8b, _, _ = run(seed, 10.0, wind=8.0)
        aw, _, _ = run(seed, 10.0, wind=6.0, moist=(.10, .11, .12, .60, .90)); ad, _, _ = run(seed, 10.0, wind=6.0)
        nb = g["fuel"] == 91
        r = dict(seed=seed, ha4=(a4 <= 60).sum() / 100, ha8=(a8 <= 60).sum() / 100, nb_burned=int(np.isfinite(a8[nb]).sum() + np.isfinite(a4[nb]).sum()),
                 same=bool(np.array_equal(a8, a8b)), ha_dry=(ad <= 60).sum() / 100, ha_wet=(aw <= 60).sum() / 100)
        ok4 &= r["ha8"] >= r["ha4"] and r["nb_burned"] == 0 and r["same"]; ok5 &= r["ha_wet"] <= r["ha_dry"]
        rows.append(r); print("Y4/5", r, flush=True)
    return dict(rows=rows, ok4=bool(ok4), ok5=bool(ok5))


t0 = time.time()
res["Y1"] = y1(); print("Y1", res["Y1"])
res["Y2"] = y2(); print("Y2", res["Y2"])
res["Y3"] = y3(); print("Y3", {k: v for k, v in res["Y3"].items() if k != "rows"})
r = y45(); res["Y4"] = dict(ok=r["ok4"]); res["Y5"] = dict(ok=r["ok5"]); res["Y45_rows"] = r["rows"]
print("Y4", r["ok4"], "Y5", r["ok5"], "seconds", round(time.time() - t0))
if len(sys.argv) > 1:
    json.dump(res, open(sys.argv[1], "w"), indent=1, default=float)
