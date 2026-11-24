"""Protocol A11 option EP: embers ride a blend of surface wind and wind aloft, weighted by intensity."""
import numpy as np

from firesim.embers import Spot


def throw(i_mw, **kw):
    b = np.ones((50, 50), bool)
    sp = Spot(10.0, 1000.0, 5.0, 0.0, b, seed=1, sigma=0.0, sig_theta=0.0, **kw)
    n = 200
    sp.emit(np.full(n, 25), np.full(n, 25), np.zeros(n), np.full(n, i_mw), np.ones(n, int))
    return sp.Lx - 250.0, sp.Ly - 250.0


def test_blend_limits():
    al = dict(aloft=(10.0, np.pi / 2), i_loft=20.0)
    x0, y0 = throw(1e-9, **al); xs, ys = throw(1e-9)
    assert np.allclose(x0, xs, rtol=1e-3) and np.allclose(y0, ys, atol=1e-3)                # no intensity: the surface wind, toward +x
    x1, y1 = throw(20.0, **al)
    assert np.allclose(x1, 0.0, atol=1e-6) and (y1 > 0).all()         # full loft: the wind aloft, toward +y
    x2, y2 = throw(40.0, **al); xa, ya = throw(40.0, aloft=(10.0, np.pi / 2), i_loft=40.0)
    assert np.allclose(x2, xa) and np.allclose(y2, ya)                # the weight is capped at 1
    xm, ym = throw(20.0 * 0.5 ** 1.5, **al)                           # weight 0.5: vector mean of (5, 0) and (0, 10)
    assert np.allclose(np.arctan2(ym, xm), np.arctan2(5.0, 2.5))
