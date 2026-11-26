"""From maps of fuel, terrain, canopy and weather to the fields the spread solver needs.

Provenance: written fresh. Per fuel model the no-wind, no-slope rate R0 and the wind and slope coefficients come from
`rothermel.spread`; per cell the wind and slope factors are applied in closed form. Wind and slope are combined as
vectors (Andrews 2018, RMRS-GTR-371, sec. 6.1, p. 87): the slope vector points upslope with length R0 phi_s, the wind
vector downwind with length R0 phi_w; the head rate is R0 plus the length of their sum, in the sum's direction. The
effective wind is the wind that alone would give the same factor, U_E = [phi_E (beta/beta_op)^E / C]^(1/B) (the report
prints the exponent as -B, which does not invert its own wind factor; 1/B does). Shape: Z = 1 + 0.25 U_E (mi/h), sec. 6.2.
Crown fire per cell follows `crown.fire` (Scott and Reinhardt 2001), vectorised.

Angles: radians, 0 = +x (east), counter-clockwise. `wind_to` is the direction the wind blows toward; `upslope` the
direction of steepest ascent.
"""
from __future__ import annotations

import numpy as np

from .rothermel import BY_NUMBER, FT, spread
from .crown import FM10, H_CANOPY, waf_unsheltered, waf_sheltered

MPH = 0.44704
BTU_FT2_TO_KJ_M2 = 1.05506 / FT ** 2


def _coeffs(fuel, moist):
    s = spread(fuel, moist, 0.0, 0.0)
    if s.get("ros_ft_min") is None:
        return None
    return s


def fields(fuel_number, moist, wind_20ft, wind_to, slope_tan, upslope, canopy_cover=None, canopy_height=None,
           cbh=None, cbd=None, crown_ratio=0.4, fmc=100.0, z_max=8.0, crown=True, crown_rate="r91"):
    """All inputs (ny, nx) arrays or scalars; wind_20ft in m/s. moist: (1h, 10h, 100h, herb, woody) fractions.
    Returns dict of (ny, nx) arrays: rate (head, m/min), heading (rad), lb, flame (m), fli (kW/m), kind (0 none,
    1 surface, 2 passive crown, 3 active crown), plus r0 and midflame wind.
    crown_rate "r91": Rothermel 1991 with Scott and Reinhardt's crown fraction (RP-29); "cruz05": protocol A25 setting
    C1, the Cruz, Alexander and Wakimoto 2005 rate (Can. J. For. Res. 35: 1626-1639, eqs. 1-4) once a cell torches."""
    fn = np.asarray(fuel_number)
    shp = fn.shape
    bc = lambda v: np.broadcast_to(np.asarray(v, float), shp)
    U20, wto, tan, ups = bc(wind_20ft), bc(wind_to), bc(slope_tan), bc(upslope)
    cc = bc(0.0 if canopy_cover is None else canopy_cover); ch = bc(0.0 if canopy_height is None else canopy_height)
    r0 = np.zeros(shp); C = np.ones(shp); B = np.ones(shp); E = np.ones(shp); ratio = np.ones(shp); beta = np.ones(shp)
    hpa = np.zeros(shp); depth = np.ones(shp); sig = np.ones(shp)
    for num in np.unique(fn):
        f = BY_NUMBER.get(int(num))
        if f is None:
            continue                       # non-burnable or unknown code: stays at rate 0
        s = spread(f, moist, 0.0, 0.0)
        if s["ros"] <= 0 and "r0_ft_min" not in s:
            continue
        m = fn == num
        r0[m] = s["r0_ft_min"]; C[m] = s["C"]; B[m] = s["B"]; E[m] = s["E"]; ratio[m] = s["beta_ratio"]
        beta[m] = s["beta"]; hpa[m] = s["hpa_btu"]; depth[m] = f.depth; sig[m] = s["sigma"]
    # midflame wind
    f_fill = crown_ratio * cc / 3.0
    hft = np.maximum(ch / FT, 1e-6)
    waf_open = 1.83 / np.log((20.0 + 0.36 * depth) / (0.13 * depth))
    with np.errstate(divide="ignore", invalid="ignore"):
        waf_shel = 0.555 / (np.sqrt(np.maximum(f_fill * hft, 1e-12)) * np.log((20.0 + 0.36 * hft) / (0.13 * hft)))
    waf = np.where((f_fill > 0.05) & (ch > 0), waf_shel, waf_open)
    Umid = U20 * waf                                        # m/s
    Uft = Umid / FT * 60.0
    phi_w = np.where(Uft > 0, C * np.power(np.maximum(Uft, 1e-12), B) * ratio ** -E, 0.0)
    phi_s = 5.275 * beta ** -0.3 * tan ** 2
    vx = phi_w * np.cos(wto) + phi_s * np.cos(ups); vy = phi_w * np.sin(wto) + phi_s * np.sin(ups)
    phi_e = np.hypot(vx, vy)
    heading = np.arctan2(vy, vx)
    rate_ft = r0 * (1.0 + phi_e)
    ue_ft = np.power(np.maximum(phi_e, 0.0) * ratio ** E / C, 1.0 / B)       # ft/min
    lb = np.minimum(1.0 + 0.25 * (ue_ft * FT / 60.0) / MPH, z_max)
    rate = rate_ft * FT
    hpa_si = hpa * BTU_FT2_TO_KJ_M2
    fli = hpa_si * rate / 60.0                              # kW/m
    kind = np.where(rate > 0, 1, 0)
    r_final = rate.copy(); i_final = fli.copy()
    if crown and cbd is not None and cbh is not None:
        cbd_a, cbh_a = bc(cbd), bc(cbh)
        has = (cbd_a > 0) & (cbh_a > 0) & (rate > 0)
        s10 = spread(FM10, moist, 0.0, 0.0)
        def r10(U20_ms, tan_):
            u = 0.4 * U20_ms / FT * 60.0
            pw = np.where(u > 0, s10["C"] * np.power(np.maximum(u, 1e-12), s10["B"]) * s10["beta_ratio"] ** -s10["E"], 0.0)
            ps = 5.275 * s10["beta"] ** -0.3 * tan_ ** 2
            return 3.34 * s10["r0_ft_min"] * (1.0 + pw + ps) * FT
        # crown rate uses the wind along the surface heading plus slope, as the surface head does
        r_act = r10(U20, tan)
        i_init = (cbh_a * (460.0 + 25.9 * fmc) / 100.0) ** 1.5
        with np.errstate(divide="ignore", invalid="ignore"):
            r_init = np.where(hpa_si > 0, 60.0 * i_init / hpa_si, np.inf)
            r_crit = np.where(cbd_a > 0, 3.0 / cbd_a, np.inf)
            # wind at which FM10 crown rate reaches r_crit (closed form of the bisection in crown.crowning_index)
            need = r_crit / (3.34 * s10["r0_ft_min"] * FT) - 1.0 - 5.275 * s10["beta"] ** -0.3 * tan ** 2
            u_ci = np.where(need > 0, np.power(np.maximum(need, 1e-12) * s10["beta_ratio"] ** s10["E"] / s10["C"], 1.0 / s10["B"]), 0.0)
            ci20 = u_ci * FT / 60.0 / 0.4                    # open wind m/s
            pw_sa = C * np.power(np.maximum(ci20 * waf / FT * 60.0, 1e-12), B) * ratio ** -E
            r_sa = r0 * (1.0 + pw_sa + phi_s) * FT
            cfb = np.clip((rate - r_init) / (r_sa - r_init), 0.0, 1.0)
        cfb = np.where(r_sa <= r_init, 1.0, cfb)
        torch = has & (rate >= r_init)
        active = torch & (r_act >= r_crit)
        cfb = np.where(active, 1.0, np.where(torch, cfb, 0.0))
        r_final = np.where(torch, rate + cfb * (r_act - rate), rate)
        if crown_rate == "cruz05":                        # A25 C1: initiation as before, rate from Cruz et al. 2005
            u10_kmh = np.maximum(U20 * 1.15 * 3.6, 1.0)  # 20 ft to 10 m open wind (GTR-266 p. 16), km/h
            with np.errstate(divide="ignore", invalid="ignore"):
                cros_a = 11.02 * u10_kmh ** 0.90 * np.maximum(cbd_a, 1e-3) ** 0.19 * np.exp(-0.17 * moist[0] * 100.0)   # eq. 3
                cac = cros_a * cbd_a / 3.0                                                                          # eqs. 1-2
            act_c = torch & (cac >= 1.0)
            r_final = np.where(act_c, cros_a, np.where(torch, np.maximum(rate, cros_a * np.exp(-cac)), rate))     # eq. 4
            cfb = np.where(act_c, 1.0, cfb); active = act_c
        elif crown_rate != "r91":
            raise ValueError(crown_rate)
        r_final = np.maximum(r_final, rate)
        load = cbd_a * np.maximum(ch - cbh_a, 0.0)
        i_final = np.where(torch, (hpa_si + load * H_CANOPY * cfb) * r_final / 60.0, fli)
        kind = np.where(active, 3, np.where(torch & (cfb > 0), 2, kind))
    flame = 0.45 * np.power(np.maximum(i_final / 3.4613, 0.0), 0.46) * FT     # Byram, surface form (GTR-371 sec. 4.5)
    return dict(rate=r_final, heading=heading, lb=lb, flame=flame, fli=i_final, kind=kind, r0=r0 * FT, midflame=Umid,
                rate_surface=rate)


def thin_barriers(fuel_number, codes=(91, 99)):
    """Road rule R1 of protocol FIRE-1 amendment A8 (own rule, no source: LANDFIRE draws roads as lines of
    non-burnable cells one cell wide). Returns (thin, filled): thin marks cells of the given codes that have
    burnable cells on both sides along at least one of four axes (N-S, E-W, two diagonals); filled is the fuel
    map with each thin cell given the lower-numbered fuel of the first such pair."""
    fn = np.asarray(fuel_number).astype(int)
    burn = np.isin(fn, list(BY_NUMBER))
    pad = np.pad(fn, 1, constant_values=-1); pb = np.pad(burn, 1, constant_values=False)
    ny, nx = fn.shape
    sl = lambda a, dy, dx: a[1 + dy: 1 + dy + ny, 1 + dx: 1 + dx + nx]
    thin = np.zeros(fn.shape, bool); filled = fn.copy()
    cand = np.isin(fn, codes)
    for dy, dx in ((1, 0), (0, 1), (1, 1), (1, -1)):
        ok = cand & ~thin & sl(pb, dy, dx) & sl(pb, -dy, -dx)
        filled[ok] = np.minimum(sl(pad, dy, dx), sl(pad, -dy, -dx))[ok]
        thin |= ok
    return thin, filled


def road_gate(F, thin, l_cross):
    """A8: a thin cell burns in this hour only if the flame its borrowed fuel would have is at least l_cross (m).
    F is the dict from fields() run on the filled fuel map; rate, fli and kind are edited in place."""
    shut = thin & ~(F["flame"] >= l_cross)
    F["rate"] = np.where(shut, 0.0, F["rate"]); F["fli"] = np.where(shut, 0.0, F["fli"]); F["kind"] = np.where(shut, 0, F["kind"])
    return F
