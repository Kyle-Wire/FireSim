"""Rothermel's surface fire spread model with Albini's changes, for the 40 Scott and Burgan fuel models.

Provenance: written fresh from the equations as set out in Andrews 2018, "The Rothermel surface fire spread model and
associated developments: a comprehensive explanation", USDA Forest Service RMRS-GTR-371 (read 23 Nov 2026 at
https://d10.nrfirescience.org/sites/default/files/2023-07/rmrs_gtr371.pdf: sections 3.1-3.2, Tables 3, 6a-6c, 7).
Every constant below is from that report unless marked. Fuel parameters: Scott and Burgan 2005, RMRS-GTR-153, Table 7
(`data/fbfm40.csv`). English units inside (ft, lb, Btu, min), as the equations are published; SI at the interface.

Dynamic models (GTR-153 p. 7-8, Table 2): live herbaceous load moves to a dead herbaceous class as it cures, none at
120 % herbaceous moisture or more, all at 30 % or less, in proportion between (the report's worked points 98, 90, 75,
60, 53 % sit on that line to the nearest percent); the moved load keeps the herbaceous SAV and takes the 1-h moisture.

Not yet in this module (each needs its source opened first): the wind adjustment from 20 ft / 10 m to midflame, the length-to-breadth relation, crown fire.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import numpy as np

RHO_P = 32.0        # particle density, lb/ft3 (GTR-371 Table 8 footnote; GTR-153 p. 19)
S_T = 0.0555        # total mineral content, fraction (same)
S_E = 0.010         # effective mineral content, fraction (same)
SAV_10H = 109.0     # 1/ft (same)
SAV_100H = 30.0     # 1/ft (same)
TONS_AC = 2000.0 / 43560.0     # t/ac -> lb/ft2 (definition of the units)
FT = 0.3048                    # m per ft (definition)
BTU_FT_S = 3.4613              # Btu/ft/s -> kW/m: 1.05506 kJ/Btu / 0.3048 m/ft (definitions)


@dataclass(frozen=True)
class Fuel:
    code: str
    number: int
    w0: tuple       # lb/ft2: dead 1h, 10h, 100h, live herb, live woody (as tabulated, before curing)
    sav: tuple      # 1/ft, same order
    depth: float    # ft
    mx_dead: float  # fraction
    heat: float     # Btu/lb
    dynamic: bool


def load_fuels(path=None):
    path = path or os.path.join(os.path.dirname(__file__), "data", "fbfm40.csv")
    out = {}
    with open(path) as f:
        rows = csv.DictReader(r for r in f if not r.startswith("#"))
        for r in rows:
            w = tuple(float(r[k]) * TONS_AC for k in ("load_1h", "load_10h", "load_100h", "load_herb", "load_woody"))
            s = (float(r["sav_1h"]), SAV_10H, SAV_100H, float(r["sav_herb"]), float(r["sav_woody"]))
            out[r["code"]] = Fuel(r["code"], int(r["number"]), w, s, float(r["depth_ft"]), float(r["mx_dead"]) / 100.0,
                                  float(r["heat"]), r["dynamic"] == "dynamic")
    return out


FUELS = load_fuels()
BY_NUMBER = {f.number: f for f in FUELS.values()}
_BREAKS = (1200.0, 192.0, 96.0, 48.0, 16.0)      # SAV subclass breaks for the g weights (GTR-371 p. 22-23)


def _subclass(s):
    for k, b in enumerate(_BREAKS):
        if s >= b:
            return k
    return -1          # below 16: weight 0


def cured_fraction(m_herb):
    return float(min(max((1.20 - m_herb) / 0.90, 0.0), 1.0))


def bed(fuel: Fuel, cured=0.0):
    """Moisture-independent fuel bed quantities. Classes: dead 1h, 10h, 100h, dead herb, live herb, live woody.
    cured: fraction of herbaceous load moved to the dead herb class (dynamic models only)."""
    c = cured if fuel.dynamic else 0.0
    w5 = fuel.w0
    w = np.array([w5[0], w5[1], w5[2], w5[3] * c, w5[3] * (1.0 - c), w5[4]], float)
    s = np.array([fuel.sav[0], fuel.sav[1], fuel.sav[2], fuel.sav[3], fuel.sav[3], fuel.sav[4]], float)
    live = np.array([0, 0, 0, 0, 1, 1], bool)
    A = s * w / RHO_P                                   # A_ij
    Ai = np.array([A[~live].sum(), A[live].sum()])
    AT = Ai.sum()
    f_ij = np.where(live, A / Ai[1] if Ai[1] > 0 else 0.0, A / Ai[0] if Ai[0] > 0 else 0.0)
    f_i = Ai / AT
    g = np.zeros(6)
    for cat in (False, True):
        idx = np.where(live == cat)[0]
        for k in range(len(_BREAKS)):
            members = [j for j in idx if _subclass(s[j]) == k and w[j] > 0]
            tot = sum(f_ij[j] for j in members)
            for j in members:
                g[j] = tot
    sig_i = np.array([(f_ij * s)[~live].sum(), (f_ij * s)[live].sum()])
    sigma = (f_i * sig_i).sum()
    rho_b = w.sum() / fuel.depth
    beta = (w / RHO_P).sum() / fuel.depth
    beta_op = 3.348 * sigma ** -0.8189
    wn = w * (1.0 - S_T)
    wn_i = np.array([(g * wn)[~live].sum(), (g * wn)[live].sum()])
    return dict(w=w, s=s, live=live, f_ij=f_ij, f_i=f_i, g=g, sig_i=sig_i, sigma=sigma, rho_b=rho_b, beta=beta,
                beta_op=beta_op, wn_i=wn_i)


def spread(fuel: Fuel, moist, wind_midflame_m_s=0.0, slope_tan=0.0, wind_limit=False, cured=None):
    """Head fire, wind blowing upslope. moist: fractions (1h, 10h, 100h, herb, woody).

    Returns dict: ros (m/min), ir (kW/m2), hpa (kJ/m2), fli (kW/m), flame (m), plus the English intermediates.
    """
    m5 = np.array(moist, float)
    b = bed(fuel, cured_fraction(m5[3]) if cured is None else cured)
    w, s, live, f_ij, f_i = b["w"], b["s"], b["live"], b["f_ij"], b["f_i"]
    sigma, beta, beta_op = b["sigma"], b["beta"], b["beta_op"]
    m = np.array([m5[0], m5[1], m5[2], m5[0], m5[3], m5[4]])
    if w.sum() <= 0 or sigma <= 0:
        return dict(ros=0.0, ir=0.0, hpa=0.0, fli=0.0, flame=0.0)
    # live fuel moisture of extinction (GTR-371 sec. 3.2.2)
    d = ~live
    num = (w[d] * np.exp(-138.0 / s[d])).sum()
    den = (w[live] * np.exp(-500.0 / s[live])).sum()
    if den > 0:
        mf_dead = (m[d] * w[d] * np.exp(-138.0 / s[d])).sum() / num if num > 0 else 0.0
        mx_live = max(2.9 * (num / den) * (1.0 - mf_dead / fuel.mx_dead) - 0.226, fuel.mx_dead)
    else:
        mx_live = fuel.mx_dead
    mx_i = np.array([fuel.mx_dead, mx_live])
    mf_i = np.array([(f_ij * m)[d].sum(), (f_ij * m)[live].sum()])
    rm = np.minimum(mf_i / mx_i, 1.0)
    eta_m = 1.0 - 2.59 * rm + 5.11 * rm ** 2 - 3.52 * rm ** 3
    eta_s = min(0.174 * S_E ** -0.19, 1.0)
    A = 133.0 * sigma ** -0.7913
    gmax = sigma ** 1.5 / (495.0 + 0.0594 * sigma ** 1.5)
    ratio = beta / beta_op
    gamma = gmax * ratio ** A * np.exp(A * (1.0 - ratio))
    ir = gamma * (b["wn_i"] * fuel.heat * eta_m * eta_s).sum()            # Btu/ft2/min
    xi = np.exp((0.792 + 0.681 * sigma ** 0.5) * (beta + 0.1)) / (192.0 + 0.2595 * sigma)
    C = 7.47 * np.exp(-0.133 * sigma ** 0.55)
    B = 0.02526 * sigma ** 0.54
    E = 0.715 * np.exp(-3.59e-4 * sigma)
    U = max(wind_midflame_m_s, 0.0) / FT * 60.0                             # ft/min
    if wind_limit:
        U = min(U, 96.8 * ir ** (1.0 / 3.0))                                # Andrews et al. 2013, as given in GTR-371 sec. 3.2.7
    phi_w = C * U ** B * ratio ** -E if U > 0 else 0.0
    phi_s = 5.275 * beta ** -0.3 * slope_tan ** 2
    eps = np.exp(-138.0 / s)
    qig = 250.0 + 1116.0 * m
    fi_of = np.where(live, f_i[1], f_i[0])
    sink = b["rho_b"] * (fi_of * f_ij * eps * qig)[w > 0].sum()             # Btu/ft3
    r = ir * xi * (1.0 + phi_w + phi_s) / sink if sink > 0 else 0.0         # ft/min
    tr = 384.0 / sigma                                                      # min
    hpa = ir * tr                                                           # Btu/ft2
    ib = hpa * r / 60.0                                                     # Btu/ft/s
    fl = 0.45 * ib ** 0.46 if ib > 0 else 0.0                               # ft
    return dict(ros=r * FT, ir=ir * 1.05506 / FT ** 2 / 60.0, hpa=hpa * 1.05506 / FT ** 2, fli=ib * BTU_FT_S,
                flame=fl * FT, ros_ft_min=r, ir_btu=ir, sigma=sigma, beta=beta, beta_ratio=ratio, mx_live=mx_live,
                phi_w=phi_w, phi_s=phi_s, xi=xi, fl_ft=fl, ros_ch_h=r * 60.0 / 66.0,
                r0_ft_min=ir * xi / sink if sink > 0 else 0.0, C=C, B=B, E=E, hpa_btu=hpa)
