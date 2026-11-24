"""Rule H1 (protocol A12): house-to-house spread scales with wind at the burning house."""
import numpy as np

from firesim.embers import Spot
from firesim.structures import Structures


def chain(u, **kw):
    n = 200                                                    # pairs 12 m edge to edge, pairs far apart
    r = np.sqrt(150.0 / np.pi)
    x = np.repeat(np.arange(n) * 500.0 + 100.0, 2); x[1::2] += 12.0 + 2 * r
    y = np.full(2 * n, 100.0)
    st = Structures(x, y, 150.0, 10.0, seed=3, lit=[(i, 0.0) for i in range(0, 2 * n, 2)], brands=0.0, k_f=0.0, **kw)
    Spot(10.0, 0.0, u, 0.0, np.zeros((30, 10100), bool), seed=0, structures=st, gen=0.0)
    e = np.empty(0, int)
    st.step(0.0, 1.0, e, e, np.empty(0)); st.step(1.0, 60.0, e, e, np.empty(0))
    return np.isfinite(st.t_ign[1::2]).mean()


def test_h1():
    base = chain(10.0)
    assert chain(10.0, u_ref=10.0) == base and chain(25.0, u_ref=10.0) == base   # at or above u_ref: unchanged
    assert chain(0.0, u_ref=10.0) == 0.0                                         # calm: no spread to neighbours
    half = chain(5.0, u_ref=10.0)                                                # factor 0.25
    want = 1 - np.exp(-1.5 * 0.25 * np.exp(-1.2))
    assert abs(half - want) < 3 * np.sqrt(want * (1 - want) / 200) and half < base
