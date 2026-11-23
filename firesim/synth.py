"""Made-up landscapes for the sandbox: patchy fuel on 30 m blocks, rolling hills. Seeded, so every run can be repeated."""
from __future__ import annotations

import numpy as np

FUELS_USED = (102, 104, 122, 145, 165, 183, 188, 91)     # GR2, GR4, GS2, SH5, TU5, TL3, TL8, non-burnable (NB1)
CANOPY = {165: (0.6, 18.0, 2.0, 0.12), 183: (0.7, 20.0, 6.0, 0.10), 188: (0.7, 20.0, 3.0, 0.15)}   # cover, height m, CBH m, CBD kg/m3


def _smooth(rng, n, scale):
    """Random field with features about `scale` cells across (sum of low-wavenumber cosines)."""
    yy, xx = np.mgrid[:n, :n] / n
    f = np.zeros((n, n))
    for _ in range(24):
        k = rng.uniform(0.5, n / scale, 2) * rng.choice([-1, 1], 2)
        f += rng.normal() * np.cos(2 * np.pi * (k[0] * xx + k[1] * yy) + rng.uniform(0, 2 * np.pi))
    return f


def landscape(seed, size_m=4000.0, block_m=30.0, relief_m=150.0):
    """Returns dict on the 30 m block grid: fuel number, elevation, canopy layers."""
    rng = np.random.default_rng(seed)
    n = int(round(size_m / block_m))
    # patches: each 30 m block takes the fuel of its nearest seed point (irregular cells about 250 m across), so any
    # fuel can border any other. (A first version cut one smooth field into bands, which wrapped every patch in a
    # closed ring of the next class, the non-burnable one included; sandbox note under Y3.)
    k = max(8, int((size_m / 250.0) ** 2))
    sy = rng.uniform(0, n, k); sx = rng.uniform(0, n, k); cls = rng.integers(0, len(FUELS_USED), k)
    yy, xx = np.mgrid[:n, :n]
    near = np.argmin((yy[..., None] - sy) ** 2 + (xx[..., None] - sx) ** 2, axis=-1)
    fuel = np.array(FUELS_USED)[cls][near]
    z = _smooth(rng, n, 40.0); z = (z - z.min()) / (z.max() - z.min()) * relief_m
    cc = np.zeros((n, n)); ch = np.zeros((n, n)); cbh = np.zeros((n, n)); cbd = np.zeros((n, n))
    for num, (a, b, c, d) in CANOPY.items():
        m = fuel == num
        cc[m] = a; ch[m] = b; cbh[m] = c; cbd[m] = d
    return dict(fuel=fuel, z=z, cc=cc, ch=ch, cbh=cbh, cbd=cbd, block=block_m, size=size_m)


def on_grid(ls, cell):
    """Sample the 30 m block maps on a solver grid of `cell` metres (nearest block), with slope from the elevation."""
    n = int(round(ls["size"] / cell))
    c = (np.arange(n) + 0.5) * cell
    idx = np.minimum((c / ls["block"]).astype(int), ls["fuel"].shape[0] - 1)
    pick = lambda a: a[np.ix_(idx, idx)]
    # elevation: bilinear from block centres, so slope is the same field at every solver cell size
    nb = ls["z"].shape[0]
    u = np.clip(c / ls["block"] - 0.5, 0, nb - 1.001); i0 = u.astype(int); w = u - i0
    zr = ls["z"][i0] * (1 - w)[:, None] + ls["z"][i0 + 1] * w[:, None]
    z = zr[:, i0] * (1 - w)[None, :] + zr[:, i0 + 1] * w[None, :]
    gy, gx = np.gradient(z, cell)
    return dict(fuel=pick(ls["fuel"]), z=z, tan=np.hypot(gx, gy), upslope=np.arctan2(gy, gx), cc=pick(ls["cc"]),
                ch=pick(ls["ch"]), cbh=pick(ls["cbh"]), cbd=pick(ls["cbd"]), n=n, cell=cell)
