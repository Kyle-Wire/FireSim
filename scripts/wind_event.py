"""Protocol A24: the end of the wind event from the wind the run is given (HRRR as analysed), not from the fire record.

python scripts/wind_event.py <fire.json> <data> [--json out.json]

A17's cut applied to the HRRR 20-ft wind over the run's box: the hourly box mean, its trailing 3-hour running mean
(hours h-2..h; the first two hours use what there is), and the cut is the first hour after the running mean's peak
from which it stays under half the peak to the end of the window. If it never does, the cut is the window's end.
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from firesim.real import Fire  # noqa: E402


def cut_hour(series):
    s = np.asarray(series, float); n = len(s)
    run = np.array([s[max(0, h - 2):h + 1].mean() for h in range(n)])
    pk = int(np.argmax(run)); half = 0.5 * run[pk]
    for h in range(pk + 1, n):
        if np.all(run[h:] < half):
            return h, run, pk
    return n, run, pk


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cfg"); ap.add_argument("data"); ap.add_argument("--json")
    a = ap.parse_args()
    fire = Fire(a.cfg, a.data, 30.0); fire.use_hrrr(gust_share=0.0)
    mean = [float(fire.wind(h)[0].mean()) for h in range(fire.hours)]
    h, run, pk = cut_hour(mean)
    out = dict(cfg=a.cfg, hours=fire.hours, box_mean_20ft=[round(v, 2) for v in mean], running3=[round(v, 2) for v in run],
               peak_hour=pk, peak=round(float(run[pk]), 2), cut_hour=h)
    print(json.dumps(out))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
