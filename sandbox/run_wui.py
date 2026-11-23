"""Ember and house cases S6-S13 of CASES.md. python sandbox/run_wui.py"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from firesim.levelset import arrival
from firesim.embers import Spot
from firesim.structures import Structures

cell = 10.0; R0 = 10.0
out = {}


def s1_scene():
    ny = nx = 321; c = ny // 2
    return np.full((ny, nx), R0), c


def s6():
    R, c = s1_scene()
    a0, _ = arrival(R, 0.0, 1.0, cell, [(c, c, 0, 0)], 160.0)
    sp = Spot(cell, 5000.0, 8.0, 0.0, R > 0, gen=0.0)
    a1, _ = arrival(R, 0.0, 1.0, cell, [(c, c, 0, 0)], 160.0, spot=sp)
    return dict(identical=bool(np.array_equal(a0, a1)), ok=bool(np.array_equal(a0, a1)))


def s8():
    R, c = s1_scene()
    a0, _ = arrival(R, 0.0, 1.0, cell, [(c, c, 0, 0)], 160.0)
    st = Structures([], [], [], cell)
    sp = Spot(cell, 5000.0, 8.0, 0.0, R > 0, gen=0.0, structures=st)
    a1, _ = arrival(R, 0.0, 1.0, cell, [(c, c, 0, 0)], 160.0, spot=sp)
    return dict(identical=bool(np.array_equal(a0, a1)), ok=bool(np.array_equal(a0, a1)))


def s7_scene():
    ny, nx = 161, 101; cy = ny // 2; cx = 30
    R = np.full((ny, nx), R0); x0 = 60; x1 = 63
    R[:, x0:x1] = 0
    return R, cy, cx, x0, x1


def s7():
    R, cy, cx, x0, x1 = s7_scene()
    delay = 2.0
    sp = Spot(cell, 5000.0, 8.0, 0.0, R > 0, fixed=(100.0, delay))
    steps = []
    def cb(t0, t1, arr, phi):
        steps.append(t1 - t0); return sp(t0, t1, arr, phi)
    a, _ = arrival(R, 0.0, 1.0, cell, [(cy, cx, 0, 0)], 120.0, spot=cb)
    strip = bool(np.isfinite(a[:, x0:x1]).any())
    j = np.unravel_index(np.argmin(a[:, x1:]), a[:, x1:].shape)
    first = a[:, x1:][j]; launch = a[j[0], x1 + j[1] - 10]
    err = first - (launch + delay)
    return dict(strip_burned=strip, first_behind=first, launch_plus_delay=launch + delay, err_min=err, step_min=max(steps),
                ok=bool(not strip and abs(err) <= max(steps) + 1e-9))


def s9():
    n = 20; area = 100.0; r = np.sqrt(area / np.pi); pitch = 2 * r + 10.0
    x = np.r_[np.arange(n) * pitch + 100, np.arange(n) * pitch + 100]; y = np.r_[np.full(n, 100.0), np.full(n, 100.0 + 2 * r + 60)]
    st = Structures(x, y, area, cell, lit=[(0, 0.0)], k_s=1e9, d_s=10.0, r_s=15.0, tau_s=5.0, k_f=0.0, brands=0.0)
    R = np.zeros((40, 80)); R[0, 0] = 1.0                       # a token burnable cell far away keeps the clock running
    sp = Spot(cell, 0.0, 5.0, 0.0, R > 0, gen=0.0, structures=st)
    arrival(R, 0.0, 1.0, cell, [(0, 0, 0, 0)], 120.0, spot=sp)
    exact = bool(np.array_equal(st.t_ign[:n], 5.0 * np.arange(n)))
    return dict(times=st.t_ign[:n].tolist(), second_row_lit=int(np.isfinite(st.t_ign[n:]).sum()), ok=bool(exact and not np.isfinite(st.t_ign[n:]).any()))


def s10():
    n = 4000; area = 100.0; r = np.sqrt(area / np.pi)
    x = np.r_[np.arange(n) * 500.0, np.arange(n) * 500.0]; y = np.r_[np.zeros(n), np.full(n, 2 * r + 12.0)]
    st = Structures(x, y, area, cell, seed=3, lit=[(i, 0.0) for i in range(n)], k_s=1.5, d_s=10.0, r_s=50.0, tau_s=5.0, k_f=0.0, brands=0.0)
    st.step(0.0, 100.0, np.empty(0, int), np.empty(0, int), np.empty(0))
    f = float(np.isfinite(st.t_ign[n:]).mean()); p = 1 - np.exp(-1.5 * np.exp(-1.2)); se = np.sqrt(p * (1 - p) / n)
    return dict(fraction=f, expected=p, tol=3 * se, ok=bool(abs(f - p) <= 3 * se))


def s11():
    ny, nx = 61, 331; cy = ny // 2
    R = np.full((ny, nx), R0)
    hx = (np.arange(30) * 10 + 20) * cell; hy = np.full(30, cy * cell)
    yy, xx = np.mgrid[:ny, :nx]
    for a in hx:
        R[np.hypot(xx * cell - a, yy * cell - cy * cell) <= 20.0] = 0
    def run(seed, kf):
        st = Structures(hx, hy, 100.0, cell, seed=seed, k_f=kf, d_f=10.0, r_f=40.0, k_s=0.0, brands=0.0)
        arrival(R, 0.0, 1.0, cell, [(r, 0, 0, 0) for r in range(ny)], 400.0, spot=Spot(cell, 3000.0, 5.0, 0.0, R > 0, gen=0.0, structures=st))
        return st
    st = run(0, 0.0); zero = int(np.isfinite(st.t_ign).sum())
    p_code = 1 - np.exp(-0.02 * st.E)                            # the chance the code states for each house
    full = [np.isfinite(run(seed, 0.02).t_ign).mean() for seed in range(400)]
    p = float(p_code[0]); n = 30 * 400; se = np.sqrt(p * (1 - p) / n); f = float(np.mean(full))
    return dict(p_code_house0=p, p_code_min=float(p_code.min()), p_code_max=float(p_code.max()), fraction=f, tol=3 * se, lit_with_kf0=zero,
                ok=bool(abs(f - p) <= 3 * se and zero == 0))


def s12():
    ny, nx = 81, 161; cy = ny // 2; cx = 20
    R = np.full((ny, nx), R0)
    gx, gy = np.meshgrid(np.arange(12), np.arange(12)); r = np.sqrt(100 / np.pi)
    hx = 900.0 + gx.ravel() * (2 * r + 8); hy = cy * cell - 6 * (2 * r + 8) + gy.ravel() * (2 * r + 8)
    yy, xx = np.mgrid[:ny, :nx]
    R[(xx * cell > 860) & (xx * cell < 900 + 12 * (2 * r + 8) + 40) & (np.abs(yy * cell - cy * cell) < 6 * (2 * r + 8) + 40)] = 0
    def run(fac, seed, **kw):
        st = Structures(hx, hy, 100.0, cell, seed=seed, s_ember=fac, **kw)
        sp = Spot(cell, 8000.0, 10.0, 0.0, R > 0, seed=seed, structures=st, gen=2e-4)
        arrival(R, 0.0, 3.0, cell, [(cy, cx, 0, 0)], 150.0, spot=sp)
        return int(np.isfinite(st.t_ign).sum())
    a = [run(1.0, s) for s in range(200)]; b = [run(0.5, s) for s in range(200)]
    z = [run(0.0, s, k_f=0.0, k_s=0.0) for s in range(20)]
    return dict(mean_lost_1=float(np.mean(a)), mean_lost_half=float(np.mean(b)), lost_factor0=int(np.sum(z)), houses=144,
                ok=bool(np.mean(b) < np.mean(a) and np.sum(z) == 0))


def s13():
    R, cy, cx, x0, x1 = s7_scene()
    res = []
    rng = np.random.default_rng(5); py = rng.integers(10, 150, 200); px = rng.integers(5, 95, 200)
    for seed in range(10):
        o = []
        for win in (True, False):
            sp = Spot(cell, 5000.0, 8.0, 0.0, R > 0, seed=seed, gen=2e-4, d_med=150.0)
            a, _ = arrival(R, 0.0, 2.0, cell, [(cy, cx, 0, 0)], 70.0, spot=sp, spot_window=win)
            o.append(a)
        A = [(np.isfinite(x) & (x <= 70)).sum() for x in o]
        both = np.isfinite(o[0][py, px]) & np.isfinite(o[1][py, px])
        res.append((abs(A[0] / A[1] - 1), float(np.median(np.abs(o[0][py, px][both] / np.maximum(o[1][py, px][both], 1e-9) - 1))), int(sp.n_spots)))
    res = np.array(res)
    return dict(area_diff_max=float(res[:, 0].max()), arrival_diff_median_max=float(res[:, 1].max()), spots_min=int(res[:, 2].min()),
                ok=bool(res[:, 0].max() <= 0.01 and res[:, 1].max() <= 0.02))


def s13b():
    R, cy, cx, x0, x1 = s7_scene()
    res = []
    rng = np.random.default_rng(5); py = rng.integers(10, 150, 200); px = rng.integers(5, 95, 200)
    for seed in range(10):
        sp = Spot(cell, 5000.0, 8.0, 0.0, R > 0, seed=seed, gen=2e-4, d_med=150.0)
        rec = []
        def record(t0, t1, arr, phi):
            new = sp(t0, t1, arr, phi) or []
            rec.extend((r, c, max(t, t1), rad) for r, c, t, rad in new); return new
        arrival(R, 0.0, 2.0, cell, [(cy, cx, 0, 0)], 70.0, spot=record)
        o = []
        for win in (True, False):
            todo = sorted(rec, key=lambda g: g[2])
            def replay(t0, t1, arr, phi):
                new = []
                while todo and todo[0][2] <= t1 + 1e-9:
                    new.append(todo.pop(0))
                return new
            a, _ = arrival(R, 0.0, 2.0, cell, [(cy, cx, 0, 0)], 70.0, spot=replay, spot_window=win)
            o.append(a)
        A = [np.isfinite(x).sum() for x in o]
        both = np.isfinite(o[0][py, px]) & np.isfinite(o[1][py, px])
        res.append((abs(A[0] / A[1] - 1), float(np.median(np.abs(o[0][py, px][both] / np.maximum(o[1][py, px][both], 1e-9) - 1))), len(rec)))
    res = np.array(res)
    return dict(area_diff_max=float(res[:, 0].max()), arrival_diff_median_max=float(res[:, 1].max()), spots_min=int(res[:, 2].min()),
                ok=bool(res[:, 0].max() <= 0.01 and res[:, 1].max() <= 0.02))


only = sys.argv[1:]
for name, fn in (("S6", s6), ("S7", s7), ("S8", s8), ("S9", s9), ("S10", s10), ("S11", s11), ("S12", s12), ("S13", s13), ("S13b", s13b)):
    if only and name not in only:
        continue
    r = fn(); out[name] = r
    show = {k: (np.round(v, 4).tolist() if not isinstance(v, (bool, list)) else v) for k, v in r.items() if k not in ("ok", "times")}
    print(name, "PASS" if r["ok"] else "FAIL", show, flush=True)
if not only:
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "results_wui.json"), "w"), indent=1, default=float)
