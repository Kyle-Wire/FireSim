"""Fetch HRRR analysis wind for a fire (protocol amendment A6, wind source W2).

python scripts/fetch_hrrr.py fires/camp_2018.json <data_root> [pad_degrees]

For each hour of the fire's window, reads the 10 m wind components, the surface gust, and 2 m temperature and
humidity from NOAA's open HRRR archive (hrrr.tHHz.wrfsfcf00.grib2 on noaa-hrrr-bdp-pds; US government work), by byte
range, crops to the fire box plus 0.15 degrees, turns the wind from the HRRR grid to true east/north, and writes
<fire folder>/weather/hrrr_f00.npz: lon, lat (ny, nx); ue, vn, gust (T, ny, nx) in m/s at 10 m; tmp_k, rh; hours;
ue80, vn80 (80 m above ground) and ue850, vn850 (850 hPa), the wind aloft that carries embers (amendment A10).
Needs rasterio (GDAL's GRIB driver) and pyproj.
"""
import json, os, sys, tempfile, urllib.request
from datetime import datetime, timedelta, timezone
import numpy as np
import rasterio
from pyproj import Transformer

WANT = {"UGRD:10 m above ground": "u", "VGRD:10 m above ground": "v", "GUST:surface": "gust",
        "TMP:2 m above ground": "tmp", "RH:2 m above ground": "rh",
        "UGRD:80 m above ground": "u80", "VGRD:80 m above ground": "v80", "UGRD:850 mb": "u850", "VGRD:850 mb": "v850",
        "HGT:surface": "hgt"}   # A10: wind aloft for embers
BASE = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{d:%Y%m%d}/conus/hrrr.t{d:%H}z.wrfsfcf00.grib2"


def get(url, rng=None):
    req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def hour(d, box):
    url = BASE.format(d=d)
    idx = [l.split(":") for l in get(url + ".idx").decode().splitlines()]
    out = {}
    for i, f in enumerate(idx):
        key = f"{f[3]}:{f[4]}"
        if key in WANT and f[5].startswith("anl"):
            end = int(idx[i + 1][1]) - 1 if i + 1 < len(idx) else ""
            with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tf:
                tf.write(get(url, (int(f[1]), end)))
            with rasterio.open(tf.name) as ds:
                if "win" not in out:
                    tr = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
                    xs, ys = tr.transform([box[0], box[2], box[0], box[2]], [box[1], box[1], box[3], box[3]])
                    win = rasterio.windows.from_bounds(min(xs), min(ys), max(xs), max(ys), ds.transform).round_offsets().round_lengths()
                    wt = ds.window_transform(win)
                    ny, nx = int(win.height), int(win.width)
                    X = wt.c + (np.arange(nx) + 0.5) * wt.a; Y = wt.f + (np.arange(ny) + 0.5) * wt.e
                    XX, YY = np.meshgrid(X, Y)
                    inv = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True)
                    lon, lat = inv.transform(XX, YY)
                    x1, y1 = tr.transform(lon, lat + 0.01)                 # where true north points on the HRRR grid
                    out.update(win=win, lon=lon, lat=lat, conv=np.arctan2(-(x1 - XX), y1 - YY))
                out[WANT[key]] = ds.read(1, window=out["win"]).astype(np.float64)
            os.unlink(tf.name)
    c = out["conv"]                                                        # grid +y is c clockwise of true north
    ue = out["u"] * np.cos(c) + out["v"] * np.sin(c); vn = -out["u"] * np.sin(c) + out["v"] * np.cos(c)
    tmp = out["tmp"]; tmp = tmp + 273.15 if np.nanmean(tmp) < 150 else tmp   # GDAL may hand back deg C
    al = []
    for lev in ("80", "850"):                                              # same turn to true east/north
        u, v = out.get("u" + lev), out.get("v" + lev)
        if u is None or v is None:
            print(f"  no {lev} wind in {url}", flush=True); u = v = np.full(ue.shape, np.nan)
        al += [u * np.cos(c) + v * np.sin(c), -u * np.sin(c) + v * np.cos(c)]
    return (out["lon"], out["lat"], ue, vn, out["gust"], tmp, out["rh"], *al, out.get("hgt", np.full(ue.shape, np.nan)))


if __name__ == "__main__":
    cfg = json.load(open(sys.argv[1])); root = sys.argv[2]
    t0 = datetime.strptime(cfg["ignition_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    pad = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15          # degrees around the fire box; a wide pad is for the station check
    b = cfg["box"]; box = [b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad]
    first = t0.replace(minute=0, second=0)
    n = cfg["hours"] + 2
    acc = []
    for k in range(n):
        d = first + timedelta(hours=k)
        r = hour(d, box); acc.append(r[2:])
        sp = np.hypot(r[2], r[3])
        print(f"{d:%Y-%m-%d %H}Z wind mean {sp.mean():.1f} max {sp.max():.1f} m/s, gust max {r[4].max():.1f}, RH min {r[6].min():.0f} %", flush=True)
    out = os.path.join(root, cfg["folder"], "weather", "hrrr_f00.npz" if len(sys.argv) <= 3 else f"hrrr_f00_pad{pad:g}.npz")
    np.savez_compressed(out, lon=r[0], lat=r[1], ue=np.array([a[0] for a in acc]), vn=np.array([a[1] for a in acc]),
                        gust=np.array([a[2] for a in acc]), tmp_k=np.array([a[3] for a in acc]), rh=np.array([a[4] for a in acc]),
                        ue80=np.array([a[5] for a in acc]), vn80=np.array([a[6] for a in acc]), ue850=np.array([a[7] for a in acc]), vn850=np.array([a[8] for a in acc]), hgt_m=acc[0][9],
                        first_hour_utc=first.strftime("%Y-%m-%dT%H:%M:%SZ"))
    print("wrote", out)
