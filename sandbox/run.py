"""Run the closed cases in CASES.md and print pass or fail against the rules fixed there. python sandbox/run.py [cell_m]"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from firesim.levelset import arrival

cell = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
R0 = 10.0
out = {}


def grid(w, h):
    return int(round(h / cell)) + 1, int(round(w / cell)) + 1


def s1():
    ny, nx = grid(3200, 3200); c = ny // 2
    a, _ = arrival(np.full((ny, nx), R0), 0.0, 1.0, cell, [(c, c, 0, 0)], 160.0)
    rng = np.random.default_rng(1)
    r = rng.uniform(300, 1500, 200); th = rng.uniform(0, 2 * np.pi, 200)
    yy = np.round(c + r * np.sin(th) / cell).astype(int); xx = np.round(c + r * np.cos(th) / cell).astype(int)
    d = np.hypot(yy - c, xx - c) * cell
    rel = np.abs(a[yy, xx] / (d / R0) - 1)
    return dict(max_rel=rel.max(), mean_rel=rel.mean(), ok=bool(rel.max() <= 0.03 and rel.mean() <= 0.015))


def s2():
    lb = 3.0; e = np.sqrt(lb * lb - 1) / lb
    ny, nx = grid(1000, 700); cy = ny // 2; cx = int(150 / cell)
    a, _ = arrival(np.full((ny, nx), R0), 0.0, lb, cell, [(cy, cx, 0, 0)], 60.0)
    k = np.arange(1, nx - cx - 1)
    head = np.interp(60, a[cy, cx + k], k * cell)
    kf = np.arange(1, cy); fl = np.interp(60, np.where(np.isfinite(a[cy + kf, cx]), a[cy + kf, cx], 1e9), kf * cell)
    kb = np.arange(1, cx); bk = np.interp(60, np.where(np.isfinite(a[cy, cx - kb]), a[cy, cx - kb], 1e9), kb * cell)
    aa = 600 / (1 + e); area = (a <= 60).sum() * cell ** 2 / (np.pi * aa * aa / lb)
    ex = (600, 600 * (1 - e), 600 * (1 - e) / (1 + e))
    ok = abs(head / ex[0] - 1) <= .03 and abs(fl / ex[1] - 1) <= .03 and abs(bk / ex[2] - 1) <= .03 and abs(area - 1) <= .05
    return dict(head=head, flank=fl, back=bk, exact=ex, area_ratio=area, ok=bool(ok))


def s3():
    ny, nx = grid(1600, 1000); cy = ny // 2; cx = int(300 / cell)
    R = np.full((ny, nx), R0)
    x0 = int(600 / cell); x1 = x0 + int(round(30 / cell))              # strip 300 m east of the ignition
    gy0 = cy + int(200 / cell); gy1 = gy0 + int(round(50 / cell))      # gap 200-250 m north of the axis
    R[:, x0:x1] = 0; R[gy0:gy1, x0:x1] = R0
    a, _ = arrival(R, 0.0, 1.0, cell, [(cy, cx, 0, 0)], 200.0)
    strip = np.ones_like(R, bool); strip[:, :x0] = False; strip[:, x1:] = False; strip[gy0:gy1] = False
    burned = bool(np.isfinite(a[strip]).any())
    errs = []
    gx_in = (x0 - 0.5) * cell; gx_out = (x1 - 0.5) * cell
    for py, px in ((0, 900), (-200, 800), (300, 900), (100, 700), (-100, 950)):
        best = np.inf
        for g in np.linspace((gy0 - cy - 0.5) * cell, (gy1 - cy - 0.5) * cell, 201):
            dd = np.hypot(gx_in - 300, g) + (gx_out - gx_in) + np.hypot(px - gx_out, py - g)
            best = min(best, dd)
        t = a[cy + int(py / cell), int(px / cell)]
        errs.append(t / (best / R0) - 1)
    return dict(strip_burned=burned, rel=errs, ok=bool(not burned and np.abs(errs).max() <= 0.03))


def s4():
    lb = 3.0; e = np.sqrt(lb * lb - 1) / lb
    ny, nx = grid(900, 1000); cy = int(200 / cell); cx = int(200 / cell)
    R = np.full((2, ny, nx), R0); H = np.zeros((2, ny, nx)); H[1] = np.pi / 2
    a, _ = arrival(R, H, lb, cell, [(cy, cx, 0, 0)], 90.0, step_min=30.0)
    k = np.arange(1, nx - cx - 1); t300 = np.interp(300, k * cell, a[cy, cx + k])
    yfl = 300 * (1 - e)
    y = 400.0; tx = a[cy + int(y / cell), cx]; exact = 30 + (y - yfl) / R0
    return dict(t_head_300=t300, t_flank_400=tx, exact_flank=exact, ok=bool(abs(t300 / 30 - 1) <= .04 and abs(tx / exact - 1) <= .04))


def s5():
    ny, nx = grid(900, 1000); cy = ny // 2; cx = int(200 / cell)
    R = np.full((ny, nx), R0); xl = int(500 / cell); R[:, xl:] = 2.0
    a, _ = arrival(R, 0.0, 1.0, cell, [(cy, cx, 0, 0)], 150.0)
    t = a[cy, int(700 / cell)]
    return dict(t=t, exact=130.0, ok=bool(abs(t / 130 - 1) <= .03))


for name, fn in (("S1", s1), ("S2", s2), ("S3", s3), ("S4", s4), ("S5", s5)):
    r = fn(); out[name] = r
    print(name, "PASS" if r["ok"] else "FAIL", {k: (np.round(v, 4).tolist() if not isinstance(v, bool) else v) for k, v in r.items() if k != "ok"})
