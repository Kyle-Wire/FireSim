"""Check the HRRR wind against weather stations for a fire's window (weather inputs only; no fire outcome is read).

python scripts/wind_check.py fires/<fire>.json <data_root> <stations.json> <hrrr npz> [--hours H] [--out table.csv]

stations.json: list of {id, kind ("asos" or "raws"), lat, lon, elev_m (optional), file (path under the fire folder)}.
Per station and hour of the window: observed mean speed and vector-mean direction, observed peak gust where the
file has one (ASOS "gust" in knots, RAWS "UPIRGZ" or "UPHRGZ" in mph), and the HRRR analysis at the nearest lattice
point at the middle of the hour: 10 m wind, surface gust, 80 m wind, 850 hPa wind. Speeds are compared at 10 m: RAWS
anemometers stand at 20 ft, and the comparison multiplies them by 1.15, the same factor the fire runs use.
"""
import argparse, csv, json, os, sys
from datetime import datetime, timedelta, timezone
import numpy as np

KT, MPH = 0.514444, 0.44704
ap = argparse.ArgumentParser()
ap.add_argument("cfg"); ap.add_argument("data"); ap.add_argument("stations"); ap.add_argument("hrrr")
ap.add_argument("--hours", type=int); ap.add_argument("--out")
a = ap.parse_args()
cfg = json.load(open(a.cfg)); folder = os.path.join(a.data, cfg["folder"])
t0 = datetime.strptime(cfg["ignition_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
hours = a.hours or cfg["hours"]
H = np.load(a.hrrr)
first = datetime.strptime(str(H["first_hour_utc"]), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
lead = (t0 - first) / timedelta(hours=1)
lon, lat = H["lon"], H["lat"]


def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def read(path, kind):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if kind == "asos":
                t = datetime.strptime(r["valid"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
                rows.append((t, f(r.get("sknt")) * KT, f(r.get("drct")), f(r.get("gust")) * KT))
            else:
                if r.get("USIRGZ", "") == "":
                    continue
                t = datetime.strptime(r["utc_valid"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                g = f(r.get("UPIRGZ")) if r.get("UPIRGZ", "") != "" else f(r.get("UPHRGZ"))
                rows.append((t, f(r["USIRGZ"]) * MPH * 1.15, f(r.get("UDIRGZ")), g * MPH))
    return rows


def at(key, k, w, j, i):
    return (1 - w) * H[key][k][j, i] + w * H[key][k + 1][j, i]


out = []; summ = []
print("station kind  elev  hrrr_elev  n   obs  h10  ratio  bias  rmse  dir_mae | obs_gust hrrr_gust ratio | h80/obs h850/obs")
for s in json.load(open(a.stations)):
    path = os.path.join(folder, s["file"])
    if not os.path.exists(path):
        print(s["id"], "file missing"); continue
    rows = read(path, s["kind"])
    d2 = ((lon - s["lon"]) * np.cos(np.radians(s["lat"]))) ** 2 + (lat - s["lat"]) ** 2
    j, i = np.unravel_index(np.argmin(d2), d2.shape)
    if np.sqrt(d2[j, i]) * 111.0 > 3.0:
        print(s["id"], "outside the HRRR crop"); continue
    hz = float(H["hgt_m"][j, i]) if "hgt_m" in H.files else np.nan
    rec = []
    for h in range(hours):
        sel = [r for r in rows if 0 <= (r[0] - t0 - timedelta(hours=h)) / timedelta(hours=1) < 1 and np.isfinite(r[1])]
        if not sel:
            continue
        sp = np.mean([r[1] for r in sel])
        dd = [r for r in sel if np.isfinite(r[2]) and r[1] > 0.5]
        to = np.radians(np.array([r[2] for r in dd]) + 180.0) if dd else np.array([])
        oe = np.mean([r[1] * np.sin(t) for r, t in zip(dd, to)]) if dd else np.nan; on = np.mean([r[1] * np.cos(t) for r, t in zip(dd, to)]) if dd else np.nan
        gs = [r[3] for r in sel if np.isfinite(r[3])]
        t = lead + h + 0.5; k = int(np.clip(np.floor(t), 0, len(H["ue"]) - 2)); w = float(np.clip(t - k, 0, 1))
        ue, vn = at("ue", k, w, j, i), at("vn", k, w, j, i)
        ddiff = np.degrees(np.arctan2(ue * on - vn * oe, ue * oe + vn * on)) if dd else np.nan
        r = dict(id=s["id"], kind=s["kind"], hour=h, obs=sp, h10=float(np.hypot(ue, vn)), dir_diff=float(ddiff),
                 obs_gust=max(gs) if gs else np.nan, hrrr_gust=float(at("gust", k, w, j, i)),
                 h80=float(np.hypot(at("ue80", k, w, j, i), at("vn80", k, w, j, i))) if "ue80" in H.files else np.nan,
                 h850=float(np.hypot(at("ue850", k, w, j, i), at("vn850", k, w, j, i))) if "ue850" in H.files else np.nan)
        rec.append(r); out.append(r)
    if not rec:
        print(s["id"], "no readings in the window"); continue
    A = lambda k_: np.array([r[k_] for r in rec])
    o, m = A("obs"), A("h10"); g = np.isfinite(A("obs_gust"))
    row = dict(id=s["id"], kind=s["kind"], elev=s.get("elev_m", np.nan), hrrr_elev=hz, n=len(rec), obs=o.mean(), h10=m.mean(), ratio=m.mean() / max(o.mean(), 1e-9),
               bias=(m - o).mean(), rmse=np.sqrt(((m - o) ** 2).mean()), dir_mae=np.nanmean(np.abs(A("dir_diff"))),
               obs_gust=A("obs_gust")[g].mean() if g.any() else np.nan, hrrr_gust=A("hrrr_gust")[g].mean() if g.any() else np.nan,
               r80=A("h80").mean() / max(o.mean(), 1e-9), r850=A("h850").mean() / max(o.mean(), 1e-9))
    summ.append(row)
    print(f"{row['id']:7s} {row['kind']:4s} {row['elev']:5.0f} {hz:6.0f} {row['n']:3d} {row['obs']:5.1f} {row['h10']:5.1f} {row['ratio']:5.2f} {row['bias']:5.1f} {row['rmse']:5.1f} {row['dir_mae']:6.0f} | "
          f"{row['obs_gust']:6.1f} {row['hrrr_gust']:6.1f} {row['hrrr_gust'] / row['obs_gust'] if row['obs_gust'] > 0 else np.nan:5.2f} | {row['r80']:5.2f} {row['r850']:5.2f}")
if summ:
    S = lambda k_: np.array([r[k_] for r in summ], float)
    windy = S("obs") >= 5.0
    print(f"stations {len(summ)}; all: median ratio {np.median(S('ratio')):.2f}, mean bias {S('bias').mean():.1f}, mean rmse {S('rmse').mean():.1f} m/s")
    if windy.any():
        print(f"stations with observed mean >= 5 m/s ({int(windy.sum())}): median ratio {np.median(S('ratio')[windy]):.2f}, mean bias {S('bias')[windy].mean():.1f}, "
              f"gust ratio median {np.nanmedian((S('hrrr_gust') / S('obs_gust'))[windy]):.2f}")
    dz = S("elev") - S("hrrr_elev"); ok = np.isfinite(dz)
    if ok.sum() >= 4:
        print(f"station height above the HRRR terrain against speed ratio: correlation {np.corrcoef(dz[ok], S('ratio')[ok])[0, 1]:.2f} over {int(ok.sum())} stations")
if a.out and out:
    with open(a.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, list(out[0])); wr.writeheader(); wr.writerows(out)
