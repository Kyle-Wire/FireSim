"""Protocol A25 check: how much of a C1 run's torching lies outside the range Cruz, Alexander and Wakimoto (2005) fitted.

python scripts/cruz_range.py <fire.json> <data> <run folder>

Counts torching cell-hours: each 30 m cell the model burned, in the hour it arrived (per seed), where the model's
fields in that hour call it a passive or active crown fire (kind 2 or 3). For each, the 10 m open wind (km/h), the
canopy bulk density (kg/m3) and the 1-h dead fuel moisture (%, standing for EFFM) are checked against the fitted
range: U10 5 to 32 km/h, CBD 0.12 to 0.48, EFFM 7 to 12 %. Prints and writes <run>/cruz_range.json.
"""
import glob, json, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from firesim.real import Fire  # noqa: E402

RANGE = dict(u10_kmh=(5.0, 32.0), cbd=(0.12, 0.48), effm=(7.0, 12.0))


def main():
    cfg, data, run = sys.argv[1:4]
    res = json.load(open(os.path.join(run, "result.json")))
    fire = Fire(cfg, data, res["cell"])
    if res["wind_source"] == "hrrr":
        fire.use_hrrr(gust_share=res["gust_share"])
    if res.get("crown_rate", "r91") != "r91":
        fire.use_crown_rate(res["crown_rate"])
    if res.get("hold_flame") is not None:
        fire.use_hold(res["hold_flame"], res.get("hold_from") or 0)
    ws = res.get("wind_scale", 1.0)
    arrs = [np.load(p)["arrival_min"] for p in sorted(glob.glob(os.path.join(run, "arrival_seed*.npz")))]
    cbd = np.asarray(fire.cbd)
    n = 0; out_any = 0; out_k = {k: 0 for k in RANGE}; low = {k: 0 for k in RANGE}
    for h in range(fire.hours):
        C = fire.coarse(h, wind_scale=ws); kind = C["kind"]; u = np.asarray(C["wind"]) * 1.15 * 3.6
        effm = fire.moist[h][0] * 100.0
        for A in arrs:
            m = (A >= 60.0 * h) & (A < 60.0 * (h + 1)) & (kind >= 2)
            k = int(m.sum())
            if not k:
                continue
            n += k
            vals = dict(u10_kmh=u[m], cbd=cbd[m], effm=np.full(k, effm))
            bad = np.zeros(k, bool)
            for key, (lo, hi) in RANGE.items():
                b = (vals[key] < lo) | (vals[key] > hi); out_k[key] += int(b.sum()); low[key] += int((vals[key] < lo).sum()); bad |= b
            out_any += int(bad.sum())
    r = dict(run=os.path.basename(os.path.normpath(run)), seeds=len(arrs), torching_cell_hours=n, range=RANGE,
             share_outside_any=round(out_any / max(n, 1), 3), share_outside={k: round(v / max(n, 1), 3) for k, v in out_k.items()},
             share_below={k: round(v / max(n, 1), 3) for k, v in low.items()})
    print(json.dumps(r))
    json.dump(r, open(os.path.join(run, "cruz_range.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
