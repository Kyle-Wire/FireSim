"""Rescore finished runs from their saved files, without rerunning the model (protocol A19).

python scripts/rescore.py fires/<fire>.json <data_root> <run_dir> [--out result_a19.json]

Reads <run_dir>/arrival_seed*.npz (arrival minutes on the 30 m grid, one per saved seed) and buildings.npz
(p_model over all the run's seeds, labels), rasterises the fire's outlines with their times as the config now
gives them (A19 pass times for FEDS), and writes, beside the untouched result.json:
- every outline: hours, observed km2, model km2 and overlap (Jaccard) per saved seed and their mean. An outline
  after the window (allowed by the config's slack_min) is compared with the model's end state and marked so;
- the model's burned area at every whole hour, per saved seed;
- the A8 score with the last outline inside the window as the fit outline: the building part from p_model (all
  seeds), the area part from the saved seeds only (stated in the output).
Nothing here feeds back into a model.
"""
import argparse, glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from firesim.real import Fire, overlap

ap = argparse.ArgumentParser()
ap.add_argument("cfg"); ap.add_argument("data"); ap.add_argument("run"); ap.add_argument("--out", default="result_a19.json")
a = ap.parse_args()
old = json.load(open(os.path.join(a.run, "result.json")))
fire = Fire(a.cfg, a.data, old.get("cell", 10.0))
T = float(old["hours"]) * 60.0
files = sorted(glob.glob(os.path.join(a.run, "arrival_seed*.npz")), key=lambda p: int(re.findall(r"seed(\d+)", p)[0]))
arrs = [np.load(p)["arrival_min"] for p in files]
seeds = [int(re.findall(r"seed(\d+)", p)[0]) for p in files]
km2 = fire.res ** 2 / 1e6
per = []
pers = fire.perimeters()
for mins, mask, name in pers:
    t = min(mins, T)
    rows = [overlap(A <= t, mask)[0] for A in arrs]
    per.append(dict(source=name, hours=round(mins / 60, 2), after_window=bool(mins > T + 1e-6), observed_km2=float(mask.sum() * km2),
                    model_km2=[float((A <= t).sum() * km2) for A in arrs], jaccard=rows,
                    model_km2_mean=float(np.mean([(A <= t).sum() * km2 for A in arrs])), jaccard_mean=float(np.mean(rows))))
hourly = [dict(hour=h, model_km2=[float((A <= h * 60).sum() * km2) for A in arrs]) for h in range(int(T // 60) + 1)]
res = dict(fire=old["fire"], run=os.path.basename(os.path.abspath(a.run)), protocol="A19 rescore: outline times from the config as it now stands; no model rerun",
           config=a.cfg, run_seeds=old["seeds"], arrival_seeds_saved=seeds, hours=T / 60, perimeters=per, hourly=hourly)
inside = [p for p in pers if p[0] <= T + 1e-6]
sat = [p for p in inside if p[2] != "GeoMAC"] or inside
fit = max(sat, key=lambda p: p[0]) if sat else None
b = np.load(os.path.join(a.run, "buildings.npz"))
label, p_model = b["label"], b["p_model"]; known = label >= 0
if fit is not None and known.any():
    y = label[known]; p = np.clip(p_model[known], 0.005, 0.995)
    bits = float(-(y * np.log2(p) + (1 - y) * np.log2(1 - p)).mean())
    pb = float(np.clip(y.mean(), 0.005, 0.995)); bits0 = float(-(y * np.log2(pb) + (1 - y) * np.log2(1 - pb)).mean())
    share = np.mean([A <= fit[0] for A in arrs], axis=0)
    q = np.clip(share, 0.005, 0.995); m = fit[1]
    area_bits = float(-(np.log2(q[m]).sum() + np.log2(1 - q[~m]).sum()))
    fb = float(np.clip(m.mean(), 0.005, 0.995)); cell0 = float(-(m.mean() * np.log2(fb) + (1 - m.mean()) * np.log2(1 - fb)))
    res["fit"] = dict(perimeter=fit[2], hours=fit[0] / 60, building_ratio=bits / bits0, area_ratio=area_bits / m.size / cell0,
                      score_a8=bits / bits0 + area_bits / m.size / cell0, bits_per_building=bits, bits_same_fire_rate=bits0,
                      area_seeds=seeds, building_seeds=old["seeds"],
                      note=f"protocol A8 with the last in-window outline as fit outline (A19); area part from {len(seeds)} saved seed(s), building part from all {old['seeds']}")
json.dump(res, open(os.path.join(a.run, a.out), "w"), indent=1, default=float)
for p_ in per:
    print(p_["source"], p_["hours"], "h", "(after window: model end state)" if p_["after_window"] else "", "mapped", round(p_["observed_km2"], 1),
          "model", [round(v, 1) for v in p_["model_km2"]], "overlap", [round(v, 3) for v in p_["jaccard"]])
print("hourly", [(h["hour"], round(float(np.mean(h["model_km2"])), 1)) for h in hourly])
print("fit", res.get("fit"))
