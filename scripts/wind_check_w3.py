"""Score wind option W3 (WindSim terrain wind, protocol A15) against stations, beside the plain HRRR 10 m wind.

python scripts/wind_check_w3.py fires/<fire>.json <data_root> <stations.json> <hrrr npz> --windsim <terrain_wind.npz> [...] [--out table.csv]

Weather only: no fire outcome is read. Station-hours with an observed speed of 3 m/s or more, at stations inside
the runs' inner domain, in hours when W3 applies. Per class (all, windward, lee): RMSE of speed, median ratio and
mean direction error for W3, for W3 from the 80 m wind over a clearing (z0 0.1 m; reported, decides nothing) and
for HRRR. A station-hour is lee if the ground 5 km upwind along the hour's reference bearing is more than 100 m
higher than the ground at the station (both from the terrain run's own ground heights).
"""
import argparse, csv, json, os, sys
from datetime import datetime, timedelta, timezone
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from firesim.windsim import TerrainWind, load_run  # noqa: E402

KT, MPH = 0.514444, 0.44704
ap = argparse.ArgumentParser()
ap.add_argument("cfg"); ap.add_argument("data"); ap.add_argument("stations"); ap.add_argument("hrrr")
ap.add_argument("--windsim", nargs="+", required=True); ap.add_argument("--hours", type=int); ap.add_argument("--out")
a = ap.parse_args()
from pyproj import Transformer
from scipy.spatial import cKDTree
cfg = json.load(open(a.cfg)); folder = os.path.join(a.data, cfg["folder"])
t0 = datetime.strptime(cfg["ignition_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
hours = a.hours or cfg["hours"]
H = np.load(a.hrrr)
lead = (t0 - datetime.strptime(str(H["first_hour_utc"]), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)) / timedelta(hours=1)
run0 = load_run(a.windsim[0]); tr = Transformer.from_crs("EPSG:4326", f"EPSG:{run0['epsg']}", always_xy=True)
lx, ly = tr.transform(H["lon"].ravel(), H["lat"].ravel())
b = cfg["box"]; lon0, lat0 = 0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])
x0, y0 = tr.transform(lon0, lat0); x1, y1 = tr.transform(lon0, lat0 + 0.01)
gamma = float(np.degrees(np.arctan2(-(x1 - x0), y1 - y0)))


def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def read(path, kind):                                           # as scripts/wind_check.py
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if kind == "asos":
                t = datetime.strptime(r["valid"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
                rows.append((t, f(r.get("sknt")) * KT, f(r.get("drct"))))
            else:
                if r.get("USIRGZ", "") == "":
                    continue
                t = datetime.strptime(r["utc_valid"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                rows.append((t, f(r["USIRGZ"]) * MPH * 1.15, f(r.get("UDIRGZ"))))
    return rows


st = []
for s in json.load(open(a.stations)):
    path = os.path.join(folder, s["file"])
    if os.path.exists(path):
        x, y = tr.transform(s["lon"], s["lat"]); st.append(dict(s, x=x, y=y, rows=read(path, s["kind"])))
sx = np.array([s["x"] for s in st]); sy = np.array([s["y"] for s in st])
near = cKDTree(np.c_[lx, ly]).query(np.c_[sx, sy])[1]            # nearest HRRR lattice point per station
PX, PY = np.r_[lx, sx], np.r_[ly, sy]
tw = TerrainWind(a.windsim, lambda X, Y: (X, Y), gamma)
tw.prepare(PX, PY)
inside = np.isfinite(tw.S[:, len(lx):]).all(0)
gtree = cKDTree(np.c_[run0["x"], run0["y"]]); ground = lambda X, Y: run0["ground"][gtree.query(np.c_[X, Y])[1]]
print(f"stations with files {len(st)}, inside the terrain runs' inner domain {int(inside.sum())}: {[s['id'] for s, k in zip(st, inside) if k]}")
rec = []
for h in range(hours):
    t = lead + h + 0.5; k = int(np.clip(np.floor(t), 0, len(H["ue"]) - 2)); w = float(np.clip(t - k, 0, 1))
    F = {key: ((1 - w) * H[key][k] + w * H[key][k + 1]).ravel() for key in ("ue", "vn", "ue80", "vn80")}
    P = {key: np.r_[v, v[near]] for key, v in F.items()}
    e, n, used = tw.wind(PX, PY, P["ue"], P["vn"], P["ue80"], P["vn80"])
    if not used:
        continue
    ec, nc, _ = tw.wind(PX, PY, P["ue"], P["vn"], P["ue80"], P["vn80"], clearing_z0=0.1)
    tw.hours_used -= 1
    br = np.radians(tw.last_bearing - gamma)                     # reference bearing on the UTM grid, wind from
    for i, s in enumerate(st):
        if not inside[i]:
            continue
        sel = [r for r in s["rows"] if 0 <= (r[0] - t0 - timedelta(hours=h)) / timedelta(hours=1) < 1 and np.isfinite(r[1])]
        if not sel:
            continue
        ob = float(np.mean([r[1] for r in sel]))
        dd = [r for r in sel if np.isfinite(r[2]) and r[1] > 0.5]
        to = np.radians(np.array([r[2] for r in dd]) + 180.0)
        oe = float(np.mean([r[1] * np.sin(q) for r, q in zip(dd, to)])) if dd else np.nan
        on = float(np.mean([r[1] * np.cos(q) for r, q in zip(dd, to)])) if dd else np.nan
        j = len(lx) + i
        dz = float(ground([s["x"] + 5000.0 * np.sin(br)], [s["y"] + 5000.0 * np.cos(br)])[0] - ground([s["x"]], [s["y"]])[0])
        ang = lambda E, N: float(np.degrees(np.arctan2(E * on - N * oe, E * oe + N * on))) if dd else np.nan
        rec.append(dict(id=s["id"], kind=s["kind"], hour=h, ref_bearing=tw.last_bearing, lee=int(dz > 100.0), upwind_rise_m=dz, obs=ob,
                        w3=float(np.hypot(e[j], n[j])), w3_clear=float(np.hypot(ec[j], nc[j])), hrrr=float(np.hypot(P["ue"][j], P["vn"][j])),
                        w3_dir=ang(e[j], n[j]), hrrr_dir=ang(P["ue"][j], P["vn"][j])))
print(f"hours W3 applied {tw.hours_used} of {hours}; station-hours {len(rec)}")
A = lambda rows, k_: np.array([r[k_] for r in rows], float)
res = {}
for name, rows in (("all", [r for r in rec if r["obs"] >= 3.0]), ("windward", [r for r in rec if r["obs"] >= 3.0 and not r["lee"]]),
                   ("lee", [r for r in rec if r["obs"] >= 3.0 and r["lee"]])):
    if not rows:
        print(f"{name:9s} no station-hours"); continue
    o = A(rows, "obs"); res[name] = {}
    for m in ("w3", "w3_clear", "hrrr"):
        v = A(rows, m); res[name][m] = float(np.sqrt(((v - o) ** 2).mean()))
        dkey = "hrrr_dir" if m == "hrrr" else "w3_dir"
        print(f"{name:9s} {m:9s} n {len(rows):4d} stations {len({r['id'] for r in rows}):2d}  rmse {res[name][m]:5.2f}  median ratio {np.median(v / o):5.2f}  bias {(v - o).mean():5.2f}  dir mae {np.nanmean(np.abs(A(rows, dkey))):4.0f}")
n_st = len({r["id"] for r in rec if r["obs"] >= 3.0})
if n_st < 4:
    print(f"A15 rule: only {n_st} stations: too thin to decide; W3 is not used")
elif "all" in res:
    ok = res["all"]["w3"] <= 0.9 * res["all"]["hrrr"] and all(res[c]["w3"] <= 1.1 * res[c]["hrrr"] for c in ("windward", "lee") if c in res)
    print(f"A15 rule (pooled RMSE at least 10 % lower, neither class more than 10 % worse): {'W3 earns a Camp fire test' if ok else 'W3 does not earn a fire test'}")
if a.out and rec:
    with open(a.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, list(rec[0])); wr.writeheader(); wr.writerows(rec)
