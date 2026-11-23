"""Crown fire, wind reduction and fire shape.

Provenance: written fresh from
- Scott and Reinhardt 2001, "Assessing crown fire potential by linking models of surface and crown fire behavior",
  USDA Forest Service RMRS-RP-29 (equations 5, 7, 11, 12, 14, 21, 22 and Tables 5-6, read from the PDF 23 Nov 2026);
- Andrews 2012, RMRS-GTR-266 (wind adjustment factor, equations 8 and 12; 10 m to 20 ft, p. 16);
- Andrews 2018, RMRS-GTR-371 section 6.2 (fire shape from effective wind).
Fuel model 10 and 5 parameters: GTR-371 Table 8a (p. 31), as printed.

The crown fraction burned rule below reproduces RP-29 Table 6 (0.25 and 1.00) but the text of its equation 28
(Appendix A) has not been read yet; flagged until it is.
"""
from __future__ import annotations

import math

from .rothermel import Fuel, TONS_AC, spread

FM10 = Fuel("FM10", 10, (3.0 * TONS_AC, 2.0 * TONS_AC, 5.0 * TONS_AC, 0.0, 2.0 * TONS_AC),
            (2000.0, 109.0, 30.0, 9999.0, 1500.0), 1.0, 0.25, 8000.0, False)
FM5 = Fuel("FM5", 5, (1.0 * TONS_AC, 0.5 * TONS_AC, 0.0, 0.0, 2.0 * TONS_AC),
           (2000.0, 109.0, 30.0, 9999.0, 1500.0), 2.0, 0.20, 8000.0, False)
H_CANOPY = 18000.0      # kJ/kg, canopy fuel heat yield (RP-29 p. 21, after FARSITE)
FT = 0.3048


def waf_unsheltered(depth_ft):
    """20-ft wind to midflame, fuel bed depth H in ft (GTR-266 eq. 8)."""
    return 1.83 / math.log((20.0 + 0.36 * depth_ft) / (0.13 * depth_ft))


def waf_sheltered(canopy_cover, crown_ratio, canopy_height_ft):
    """Under a canopy (GTR-266 eq. 12); f = crown ratio x canopy cover / 3. Use when f > 0.05 (p. 8)."""
    f = crown_ratio * canopy_cover / 3.0
    h = canopy_height_ft
    return 0.555 / (math.sqrt(f * h) * math.log((20.0 + 0.36 * h) / (0.13 * h)))


def waf(depth_ft, canopy_cover=0.0, crown_ratio=1.0, canopy_height_ft=0.0):
    f = crown_ratio * canopy_cover / 3.0
    if f > 0.05 and canopy_height_ft > 0:
        return waf_sheltered(canopy_cover, crown_ratio, canopy_height_ft)
    return waf_unsheltered(depth_ft)


def wind_20ft_from_10m(u10):
    return u10 / 1.15       # GTR-266 p. 16 (Turner and Lawson 1978)


def length_to_width(effective_midflame_m_s):
    """Z = 1 + 0.25 U_E, U_E in mi/h (GTR-371 sec. 6.2)."""
    return 1.0 + 0.25 * effective_midflame_m_s / 0.44704


def initiation_intensity(cbh_m, fmc_pct=100.0):
    """Van Wagner's surface intensity needed to ignite the canopy, kW/m (RP-29 eq. 11)."""
    return (cbh_m * (460.0 + 25.9 * fmc_pct) / 100.0) ** 1.5


def active_rate(moist, wind_open_m_s, slope_tan):
    """Rothermel's 1991 crown fire rate, m/min: 3.34 x fuel model 10 at 40 % of the open wind (RP-29 eq. 7)."""
    return 3.34 * spread(FM10, moist, 0.4 * wind_open_m_s, slope_tan)["ros"]


def crowning_index(moist, slope_tan, cbd):
    """Open wind (m/s) at which the crown fire rate reaches 3.0 / CBD m/min (RP-29 eq. 14), by bisection."""
    target = 3.0 / cbd
    lo, hi = 0.0, 100.0
    if active_rate(moist, hi, slope_tan) < target:
        return float("inf")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if active_rate(moist, mid, slope_tan) < target:
            lo = mid
        else:
            hi = mid
    return hi


def fire(fuel: Fuel, moist, wind_open_m_s, slope_tan, wrf, cbh_m, cbd, canopy_load, fmc_pct=100.0):
    """Surface and crown fire together. wind_open: 20-ft (6.1 m) wind, m/s; wrf: wind reduction to midflame;
    cbd kg/m3; canopy_load kg/m2. Returns rates m/min, intensities kW/m, type and crown fraction burned."""
    s = spread(fuel, moist, wrf * wind_open_m_s, slope_tan)
    out = dict(r_surface=s["ros"], i_surface=s["fli"], hpa=s["hpa"], flame_surface=s["flame"])
    if cbd <= 0 or cbh_m <= 0 or s["ros"] <= 0:
        out.update(type="surface", cfb=0.0, r_final=s["ros"], i_final=s["fli"], r_active=0.0)
        return out
    i_init = initiation_intensity(cbh_m, fmc_pct)
    r_init = 60.0 * i_init / s["hpa"]                               # eq. 12, m/min
    r_act = active_rate(moist, wind_open_m_s, slope_tan)
    r_act_crit = 3.0 / cbd
    ci = crowning_index(moist, slope_tan, cbd)
    r_sa = spread(fuel, moist, wrf * ci, slope_tan)["ros"] if math.isfinite(ci) else float("inf")
    if s["ros"] < r_init:
        kind, cfb = "surface", 0.0
    elif r_act >= r_act_crit:
        kind, cfb = "active", 1.0
    else:
        kind = "passive"
        cfb = min(max((s["ros"] - r_init) / (r_sa - r_init), 0.0), 1.0) if r_sa > r_init else 1.0
    r_final = s["ros"] + cfb * (r_act - s["ros"])                   # eq. 21
    i_final = (s["hpa"] + canopy_load * H_CANOPY * cfb) * r_final / 60.0     # eq. 22
    out.update(type=kind, cfb=cfb, r_final=r_final, i_final=i_final, r_active=r_act, r_active_crit=r_act_crit,
               i_init=i_init, r_init=r_init, r_sa=r_sa, crowning_index=ci)
    return out
