"""Station weather to hourly wind and dead fuel moisture.

Provenance: written fresh. Sources:
- Equilibrium moisture content from air temperature and relative humidity: Bradshaw, Deeming, Burgan and Cohen 1983,
  "The 1978 National Fire-Danger Rating System: technical documentation", USDA FS GTR INT-169, Appendix C
  (Simard's equations). 1-hour moisture = 1.03 EMC (eq. 26), 10-hour = 1.28 EMC when no fuel stick is read (eq. 31).
- 100-hour moisture is not measured at the stations used. It is set to the 1-hour value plus 2 points, the spacing of
  the dead fuel moisture scenarios D1-D4 in Scott and Burgan 2005 (RMRS-GTR-153): 3/4/5, 6/7/8, 9/10/11, 12/13/14 %.
- ASOS wind is a 2-minute mean at 10 m; it is brought to 20 ft by dividing by 1.15 (`crown.wind_20ft_from_10m`).
  RAWS wind is a 10-minute mean at 20 ft and is used as read.
Not applied: the fuel-level temperature and humidity adjustment of INT-169 Table 6 (it needs sky cover, and it is a
midafternoon rule). Gusts are not used. Both are stated limits of candidate M1.

Directions: stations report the compass bearing the wind blows FROM, relative to true north. `to_grid_angle` turns that
into the solver's angle (radians, 0 = +x, counter-clockwise, the direction blown TOWARD) on a map grid whose +y axis
is `convergence_deg` clockwise from true north.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone

import numpy as np

KT = 0.514444; MPH = 0.44704


def emc(rh, temp_f):
    """Equilibrium moisture content (%), INT-169 Appendix C. rh in %, temperature in deg F."""
    rh = np.asarray(rh, float); t = np.asarray(temp_f, float)
    # Coefficients as printed in Appendix C (eqs C-1 to C-3, printed p. 39; read from the page image, the scan's text
    # layer is garbled). The printed ranges overlap (h < 11, 10 < h < 51, h > 50); the breaks used here are 10 and 50.
    lo = 0.03299 + 0.281073 * rh - 0.000578 * rh * t
    mid = 2.22749 + 0.160107 * rh - 0.01478 * t
    hi = 21.0606 + 0.005565 * rh ** 2 - 0.00035 * rh * t - 0.483199 * rh
    return np.where(rh < 10.0, lo, np.where(rh <= 50.0, mid, hi))


def dead_moisture(rh, temp_f, stick_10h=None):
    """(1h, 10h, 100h) as fractions."""
    e = float(emc(rh, temp_f))
    m1 = 1.03 * e
    m10 = float(stick_10h) if stick_10h is not None and np.isfinite(stick_10h) else 1.28 * e
    return m1 / 100.0, m10 / 100.0, (m1 + 2.0) / 100.0


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def shef(r):
    """A RAWS row with SHEF codes in their 6-letter form: some DCP feeds (Colorado's) send 7 letters with a trailing
    probability code (USIRGZZ for USIRGZ); the shorter name is added where absent."""
    for k in list(r):
        if k and len(k) == 7 and k.endswith("Z") and k.isupper() and k[:6] not in r:
            r[k[:6]] = r[k]
    return r


def read_station(path, kind):
    """Rows of (utc datetime, speed m/s at 20 ft, from-bearing deg, temp F, RH %, 10-h stick %). kind: 'asos' or 'raws'."""
    out = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if kind == "asos":
                t = datetime.strptime(r["valid"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
                row = (t, _f(r["sknt"]) * KT / 1.15, _f(r["drct"]), _f(r["tmpf"]), _f(r["relh"]), np.nan)
            else:
                r = shef(r)
                if r.get("TAIRGZ", "") == "":
                    continue                                   # daily summary row
                t = datetime.strptime(r["utc_valid"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                row = (t, _f(r["USIRGZ"]) * MPH, _f(r["UDIRGZ"]), _f(r["TAIRGZ"]), _f(r["XRIRGZ"]), _f(r.get("MMIRGZ")))   # no fuel stick in some feeds
            out.append(row)
    return out


def hourly(rows, t0, hours):
    """Hourly means from t0: arrays (hours,) of u_east, v_north (m/s, direction blown toward), temp F, RH, stick."""
    acc = [[] for _ in range(hours)]
    for row in rows:
        k = int((row[0] - t0) / timedelta(hours=1)) if row[0] >= t0 else -1
        if 0 <= k < hours:
            acc[k].append(row[1:])
    out = np.full((hours, 5), np.nan)
    for k, a in enumerate(acc):
        if not a:
            continue
        a = np.array(a, float)
        ok = np.isfinite(a[:, 0]) & np.isfinite(a[:, 1])
        if ok.any():
            to = np.radians(a[ok, 1] + 180.0)
            out[k, 0] = np.mean(a[ok, 0] * np.sin(to)); out[k, 1] = np.mean(a[ok, 0] * np.cos(to))
        for j, c in ((2, 2), (3, 3), (4, 4)):
            if np.isfinite(a[:, c]).any():
                out[k, j] = np.nanmean(a[:, c])
    for j in range(5):                                        # fill gaps from the nearest hour with a reading
        v = out[:, j]; ok = np.isfinite(v)
        if ok.any() and not ok.all() and j < 4:
            idx = np.arange(hours); out[:, j] = np.interp(idx, idx[ok], v[ok])
    return out


def idw(values, sx, sy, gx, gy, power=2.0):
    """Inverse-distance mean of station values (n,) at points gx, gy (any shape). NaN stations are left out."""
    values = np.asarray(values, float); ok = np.isfinite(values)
    d2 = (np.asarray(gx)[..., None] - np.asarray(sx)[ok]) ** 2 + (np.asarray(gy)[..., None] - np.asarray(sy)[ok]) ** 2
    w = 1.0 / np.maximum(d2, 1.0) ** (power / 2.0)
    return (w * values[ok]).sum(-1) / w.sum(-1)


def to_grid(u_east, v_north, convergence_deg):
    """True east/north components to grid x/y, where grid +y points `convergence_deg` clockwise from true north."""
    c = np.radians(convergence_deg)
    return u_east * np.cos(c) - v_north * np.sin(c), u_east * np.sin(c) + v_north * np.cos(c)
