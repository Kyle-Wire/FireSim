"""Sandbox case S14 (protocol A8 road rule R1): expected answers are in sandbox/CASES.md, written before this ran."""
import numpy as np

from firesim.landscape import fields, road_gate, thin_barriers
from firesim.levelset import arrival

MOIST = (0.06, 0.07, 0.08, 0.60, 0.90)


def scene(first_line=True):
    fuel = np.full((20, 34), 102)
    if first_line:
        fuel[:, 10] = 91
    fuel[:, 20:22] = 91
    return fuel


def run(fuel, l_cross=None, legacy=False):
    k = 3
    if legacy:
        F = fields(fuel, MOIST, 6.0, 0.0, 0.0, 0.0)
    else:
        thin, filled = thin_barriers(fuel) if l_cross is not None else (np.zeros(fuel.shape, bool), fuel)
        F = fields(filled, MOIST, 6.0, 0.0, 0.0, 0.0)
        if l_cross is not None:
            F = road_gate(F, thin, l_cross)
    up = lambda a: np.repeat(np.repeat(a, k, 0), k, 1)
    arr, _ = arrival(up(F["rate"]), up(F["heading"]), up(F["lb"]), 10.0, [(30, 6, 0.0, 15.0)], 90.0)
    return arr


def test_s14_road_line():
    off = run(scene()); legacy = run(scene(), legacy=True)
    assert np.array_equal(off, legacy, equal_nan=True)
    assert np.isfinite(off[:, :30]).any() and not np.isfinite(off[:, 30:]).any()
    high = run(scene(), 50.0)
    assert not np.isfinite(high[:, 30:]).any()
    cross = run(scene(), 0.0); free = run(scene(first_line=False))
    assert np.array_equal(cross, free, equal_nan=True)
    assert np.isfinite(cross[:, 33:60]).any() and not np.isfinite(cross[:, 60:]).any()


def test_thin_cells_only():
    fuel = scene(); fuel[5:9, 26:30] = 99; fuel[12, 28] = 98
    thin, filled = thin_barriers(fuel)
    assert thin[:, 10].all() and not thin[:, 20:22].any() and not thin[5:9, 26:30][1:-1, 1:-1].any() and not thin[12, 28]
    assert (filled[:, 10] == 102).all()
