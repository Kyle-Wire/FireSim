"""Print what the model sees along a straight line between two points (diagnosis; reads no outcome data).

python scripts/transect.py fires/<fire>.json <data_root> lon0 lat0 lon1 lat1 [--hours 0,2,4] [--n 25] [--wind hrrr] [--gust-share g] [--road-cross L]
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from firesim.real import Fire

ap = argparse.ArgumentParser()
ap.add_argument("cfg"); ap.add_argument("data")
for k in ("lon0", "lat0", "lon1", "lat1"):
    ap.add_argument(k, type=float)
ap.add_argument("--hours", default="0,2,4"); ap.add_argument("--n", type=int, default=25)
ap.add_argument("--wind", default="hrrr"); ap.add_argument("--gust-share", type=float, default=0.0); ap.add_argument("--road-cross", type=float); ap.add_argument("--aloft", action="store_true", help="also print the 80 m and 850 hPa wind (A10)")
a = ap.parse_args()
f = Fire(a.cfg, a.data)
if a.wind == "hrrr":
    f.use_hrrr(gust_share=a.gust_share)
if a.road_cross is not None:
    f.use_roads(a.road_cross)
r0, c0 = f.rowcol(a.lon0, a.lat0); r1, c1 = f.rowcol(a.lon1, a.lat1)
rr = (np.linspace(r0, r1, a.n) // f.k).astype(int).clip(0, f.ny30 - 1); cc = (np.linspace(c0, c1, a.n) // f.k).astype(int).clip(0, f.nx30 - 1)
line = np.degrees(np.arctan2(r1 - r0, c1 - c0))
km = np.hypot(r1 - r0, c1 - c0) * f.cell / 1000
print(f"line {km:.1f} km, grid direction {line:.0f} deg (0 = +x east, 90 = grid north); convergence {f.convergence:.1f}")
for h in [int(v) for v in a.hours.split(",")]:
    C = f.coarse(h)
    print(f"hour {h}: moisture {f.moist[h]}")
    print(" km   fuel  cc%  ch_m cbh_m  cbd  slope% wind20 wind_to  rate_m/min head_to  lb  flame_m  fli_kW/m kind")
    for i, (r, c) in enumerate(zip(rr, cc)):
        print(f"{km * i / (a.n - 1):5.1f} {f.fuel[r, c]:5d} {f.cc[r, c] * 100:4.0f} {f.ch[r, c]:5.1f} {f.cbh[r, c]:5.1f} {f.cbd[r, c]:5.2f} {f.tan[r, c] * 100:6.0f} "
              f"{C['wind'][r, c]:6.1f} {np.degrees(C['wind_to'][r, c]):7.0f} {C['rate'][r, c]:10.1f} {np.degrees(C['heading'][r, c]):7.0f} {C['lb'][r, c]:4.1f} "
              f"{C['flame'][r, c]:7.1f} {C['fli'][r, c]:9.0f} {int(C['kind'][r, c]):4d}")
    if a.aloft:
        print("  wind along the line (m/s, direction blown toward in grid degrees): km, 20 ft, 80 m, 850 hPa")
        w80 = f.wind(h, level="80"); w85 = f.wind(h, level="850")
        for i, (r, c) in enumerate(zip(rr, cc)):
            if i % 3 == 0:
                print(f"  {km * i / (a.n - 1):5.1f}  {C['wind'][r, c]:5.1f} {np.degrees(C['wind_to'][r, c]):5.0f}   {w80[0][r, c]:5.1f} {np.degrees(w80[1][r, c]):5.0f}   {w85[0][r, c]:5.1f} {np.degrees(w85[1][r, c]):5.0f}")
    burn = C["rate"] > 0
    q = np.percentile(C["rate"][burn], [10, 50, 90, 99])
    print(f"  whole box: burnable share {burn.mean():.2f}; rate percentiles 10/50/90/99: {q.round(1)}; kind shares surface/passive/active: "
          f"{[(round(float((C['kind'][burn] == k).mean()), 3)) for k in (1, 2, 3)]}; mean wind {C['wind'].mean():.1f}")
