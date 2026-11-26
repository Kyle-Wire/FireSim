"""A real fire from files on disk: LANDFIRE fuels and terrain, station weather, building points.

Provenance: written fresh. Needs rasterio, pyproj and geopandas (not needed by the rest of FireSim).
Grid: the LANDFIRE grid (30 m, a projected CRS in metres), cropped to a box and flipped so that row 0 is the south
edge (+y = grid north). The solver runs on `cell` metres (default 10), each LANDFIRE cell split 3 x 3.
Slope and upslope direction are computed here from the LANDFIRE elevation layer on that grid, so that they share its
orientation (the grid's north is not true north: 15 degrees off in northern California).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np

from . import weather as wx
from .landscape import fields


def _utc(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


class Fire:
    def __init__(self, cfg_path, data_root, cell=10.0):
        import rasterio
        from rasterio.windows import from_bounds
        from pyproj import Transformer
        self.cfg = c = json.load(open(cfg_path))
        self.dir = os.path.join(data_root, c["folder"])
        self.cell = float(cell); self.t0 = _utc(c["ignition_utc"]); self.hours = int(c["hours"])
        lf = lambda k: os.path.join(self.dir, c["landfire"][k])
        with rasterio.open(lf("fbfm40")) as ds:
            self.crs = ds.crs; res = ds.res[0]
            tr = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
            b = c["box"]                                        # west, south, east, north in degrees
            xs, ys = tr.transform([b[0], b[2], b[0], b[2]], [b[1], b[1], b[3], b[3]])
            L, B, R, T = (max(min(xs), ds.bounds.left), max(min(ys), ds.bounds.bottom),
                          min(max(xs), ds.bounds.right), min(max(ys), ds.bounds.top))
            win = from_bounds(L, B, R, T, ds.transform).round_offsets().round_lengths()
            wt = ds.window_transform(win)
            self.left = wt.c; self.top = wt.f
        self.res = res; self.k = int(round(res / self.cell))
        assert abs(self.k * self.cell - res) < 1e-6, "cell must divide the LANDFIRE cell"

        def rd(key, scale=1.0):
            with rasterio.open(lf(key)) as ds:
                a = ds.read(1, window=win).astype(np.float64)
                a[a == ds.nodata] = np.nan
            return a[::-1] * scale                              # row 0 = south

        fuel = rd("fbfm40"); self.fuel = np.where(np.isfinite(fuel), fuel, 99).astype(int)
        z0 = lambda a: np.where(np.isfinite(a) & (a > 0), a, 0.0)
        u = c["landfire"]["units"]
        self.cc = z0(rd("cc", u["cc"])); self.ch = z0(rd("ch", u["ch"])); self.cbh = z0(rd("cbh", u["cbh"])); self.cbd = z0(rd("cbd", u["cbd"]))
        z = rd("elev"); z = np.where(np.isfinite(z), z, np.nanmean(z))
        gy, gx = np.gradient(z, res)
        self.elev = z; self.tan = np.hypot(gx, gy); self.upslope = np.arctan2(gy, gx)
        self.ny30, self.nx30 = self.fuel.shape
        self.bottom = self.top - self.ny30 * res
        self.ny, self.nx = self.ny30 * self.k, self.nx30 * self.k
        self.tr = tr
        # angle of grid north from true north at the box centre, clockwise positive
        lon, lat = 0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])
        x0, y0 = tr.transform(lon, lat); x1, y1 = tr.transform(lon, lat + 0.01)
        self.convergence = float(np.degrees(np.arctan2(-(x1 - x0), y1 - y0)))
        self._weather()
        self._cache = {}; self._coarse = {}
        self.ember_level = None; self.l_cross = None; self.l_hold = None; self.thin = np.zeros(self.fuel.shape, bool); self.fuel_road = self.fuel

    def use_hold(self, l_hold, from_hour=0):
        """Protocol A22 held-edge rule H2: in any hour a cell whose head-fire flame length is under l_hold (m) does not
        carry fire (PMS 437: under 4 ft direct attack by hand, under 8 ft with equipment). A24 (H3): only in hours
        from `from_hour` on (the end of the wind event, scripts/wind_event.py); never in the ignition hour."""
        self.l_hold = float(l_hold); self.hold_from = max(int(from_hour), 1 if from_hour else 0); self._cache = {}; self._coarse = {}

    def use_roads(self, l_cross):
        """Protocol A8 road rule R1: thin non-burnable lines burn as the fuel beside them when its flame >= l_cross (m)."""
        from .landscape import thin_barriers
        self.l_cross = float(l_cross); self.thin, self.fuel_road = thin_barriers(self.fuel); self._cache = {}; self._coarse = {}

    # --- coordinates: solver x, y in metres; cell (row, col) centre is at (col * cell, row * cell)
    def xy(self, lon, lat):
        X, Y = self.tr.transform(np.asarray(lon), np.asarray(lat))
        return np.asarray(X) - self.left - 0.5 * self.cell, np.asarray(Y) - self.bottom - 0.5 * self.cell

    def rowcol(self, lon, lat):
        x, y = self.xy(lon, lat)
        return np.round(y / self.cell).astype(int), np.round(x / self.cell).astype(int)

    def _weather(self):
        c = self.cfg; n = len(c["stations"])
        sx = np.zeros(n); sy = np.zeros(n); H = np.zeros((n, self.hours, 5))
        for i, s in enumerate(c["stations"]):
            sx[i], sy[i] = self.xy(s["lon"], s["lat"])
            H[i] = wx.hourly(wx.read_station(os.path.join(self.dir, s["file"]), s["kind"]), self.t0, self.hours)
        self.sx, self.sy, self.H = sx, sy, H
        ix, iy = self.xy(c["ignition"]["lon"], c["ignition"]["lat"])
        live = c["live_moisture"]
        self.moist = []; self.met = []
        for h in range(self.hours):
            t = float(wx.idw(H[:, h, 2], sx, sy, ix, iy)); rh = float(wx.idw(H[:, h, 3], sx, sy, ix, iy))
            stick = float(wx.idw(H[:, h, 4], sx, sy, ix, iy)) if np.isfinite(H[:, h, 4]).any() else None
            m = wx.dead_moisture(rh, t, stick)
            self.moist.append(m + (live[0], live[1])); self.met.append((t, rh, stick))

    def use_hrrr(self, path=None, gust_share=0.0):
        """Switch the wind source to HRRR analysis (protocol A6, W2). File written by scripts/fetch_hrrr.py."""
        from scipy.interpolate import LinearNDInterpolator
        d = np.load(path or os.path.join(self.dir, "weather", "hrrr_f00.npz"))
        x, y = self.xy(d["lon"].ravel(), d["lat"].ravel())
        self._hrrr = dict(tri=LinearNDInterpolator(np.c_[x, y], np.zeros(len(x))), ue=d["ue"], vn=d["vn"], gust=d["gust"],
                          **{k: d[k] for k in ("ue80", "vn80", "ue850", "vn850", "ue700", "vn700") if k in d.files},
                          lead=(self.t0 - _utc(str(d["first_hour_utc"]))) / timedelta(hours=1), g=float(gust_share))
        self._cache = {}; self._coarse = {}

    def use_windsim(self, paths):
        """Protocol A15, wind option W3: WindSim terrain wind inside its sub-domain, HRRR elsewhere and in hours
        whose 80 m reference bearing is outside the runs' range. Needs use_hrrr first (with the 80 m wind)."""
        from pyproj import Transformer
        from .windsim import TerrainWind, load_run
        if "ue80" not in self._hrrr:
            raise ValueError("W3 needs the HRRR 80 m wind: fetch it again with scripts/fetch_hrrr.py")
        epsg = load_run(paths[0])["epsg"]
        t = Transformer.from_crs(self.crs, f"EPSG:{epsg}", always_xy=True)
        ll = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        b = self.cfg["box"]; lon, lat = 0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])
        x0, y0 = ll.transform(lon, lat); x1, y1 = ll.transform(lon, lat + 0.01)
        gamma = float(np.degrees(np.arctan2(-(x1 - x0), y1 - y0)))
        self._ws = TerrainWind(paths, lambda X, Y: t.transform(X + self.left + 0.5 * self.cell, Y + self.bottom + 0.5 * self.cell), gamma)
        self._cache = {}; self._coarse = {}

    def _hrrr_at(self, h, GX, GY, level=None):
        """True east/north wind at points. level None: 10 m wind with the gust share, brought to 20 ft.
        level "80", "850" or "700": the wind at 80 m above ground or at 850 / 700 hPa, as it is (protocols A10, A24)."""
        H = self._hrrr
        t = H["lead"] + h + 0.5                                 # middle of the fire hour, in HRRR hours
        k = int(np.clip(np.floor(t), 0, len(H["ue"]) - 2)); w = float(np.clip(t - k, 0.0, 1.0))
        out = []
        for key in ("ue", "vn", "gust") if level is None else ("ue" + level, "vn" + level):
            v = (1 - w) * H[key][k] + w * H[key][k + 1]
            H["tri"].values = v.ravel()[:, None].astype(np.float64)
            out.append(H["tri"](GX, GY))
        if level is not None:
            return out[0], out[1]
        ue, vn, gust = out
        sp = np.hypot(ue, vn)
        f = np.where(sp > 0, (sp + H["g"] * np.maximum(gust - sp, 0.0)) / np.maximum(sp, 1e-9), 1.0) / 1.15   # gust share, 10 m to 20 ft
        return ue * f, vn * f

    def use_ember_wind(self, level):
        """Protocol A10: embers ride the wind aloft. level "80" (80 m above ground) or "850" (850 hPa); None = the 20-ft wind."""
        if level is not None and "ue" + level not in self._hrrr:
            raise ValueError(f"the HRRR file has no {level} wind: fetch it again with scripts/fetch_hrrr.py")
        self.ember_level = level; self._coarse = {}

    def hrrr_vs_stations(self):
        """Per station: mean HRRR 10 m speed / station speed (station at its own height) and mean direction difference (deg)."""
        g = self._hrrr["g"]; self._hrrr["g"] = 0.0; rows = []
        for i, s in enumerate(self.cfg["stations"]):
            a = np.array([[*(np.asarray(v).item() * 1.15 for v in self._hrrr_at(h, self.sx[i], self.sy[i])), self.H[i, h, 0], self.H[i, h, 1]] for h in range(self.hours)])
            fac = 1.15 if s["kind"] == "asos" else 1.0
            ok = np.isfinite(a).all(1)
            hs = np.hypot(a[ok, 0], a[ok, 1]); ss = np.hypot(a[ok, 2], a[ok, 3]) * fac
            dd = np.degrees(np.arctan2(a[ok, 0] * a[ok, 3] - a[ok, 1] * a[ok, 2], a[ok, 0] * a[ok, 2] + a[ok, 1] * a[ok, 3]))
            rows.append(dict(station=s["id"], hrrr_mean=float(hs.mean()), station_mean=float(ss.mean()), ratio=float(hs.mean() / max(ss.mean(), 1e-9)), dir_diff_deg=float(np.mean(dd))))
        self._hrrr["g"] = g
        return rows

    def wind(self, h, scale=1.0, turn_deg=0.0, level=None):
        """20-ft wind on the 30 m grid for hour h: speed (m/s) and solver angle blown toward. level: wind aloft (A10)."""
        step = 20                                               # interpolate on a 600 m lattice, then nearest
        gy = (np.arange(0, self.ny30, step) + 0.5 * step) * self.res; gx = (np.arange(0, self.nx30, step) + 0.5 * step) * self.res
        GX, GY = np.meshgrid(gx, gy)
        if getattr(self, "_hrrr", None):
            ue, vn = self._hrrr_at(min(h, self.hours - 1), GX, GY, level)
            if level is None and getattr(self, "_ws", None):
                e80, n80 = self._hrrr_at(min(h, self.hours - 1), GX, GY, "80")
                ue, vn, _ = self._ws.wind(GX, GY, ue * 1.15, vn * 1.15, e80, n80)   # W3 works at 10 m
                ue, vn = ue / 1.15, vn / 1.15
        else:
            ue = wx.idw(self.H[:, h, 0], self.sx, self.sy, GX, GY); vn = wx.idw(self.H[:, h, 1], self.sx, self.sy, GX, GY)
        ux, uy = wx.to_grid(ue, vn, self.convergence)
        if turn_deg:
            a = np.radians(turn_deg); ux, uy = ux * np.cos(a) - uy * np.sin(a), ux * np.sin(a) + uy * np.cos(a)
        up = lambda a: np.repeat(np.repeat(a, step, 0), step, 1)[: self.ny30, : self.nx30]
        ux, uy = up(ux), up(uy)
        return np.hypot(ux, uy) * scale, np.arctan2(uy, ux)

    def slice(self, h, wind_scale=1.0, wind_turn=0.0, moist_add=0.0, crown=True):
        """Fields for hour h on the solver grid: dict rate, heading, lb, fli, wind, wind_to (float32 except rate)."""
        key = (h, wind_scale, wind_turn, moist_add, crown)
        if self._cache.get("key") != key:
            C = self.coarse(h, wind_scale, wind_turn, moist_add, crown)
            up = lambda a, dt=np.float32: np.repeat(np.repeat(a.astype(dt), self.k, 0), self.k, 1)
            self._cache = dict(key=key, rate=up(C["rate"], np.float64), heading=up(C["heading"]), lb=up(C["lb"]),
                               kind=C["kind"], flame=C["flame"])
        return self._cache

    def coarse(self, h, wind_scale=1.0, wind_turn=0.0, moist_add=0.0, crown=True):
        """The same fields on the 30 m fuel grid, a few hours kept (embers and houses look back and ahead an hour,
        and rebuilding the fields each time was half of a run's wall time)."""
        h = min(h, self.hours - 1)
        key = (h, wind_scale, wind_turn, moist_add, crown, self.l_cross, self.ember_level, self.l_hold)
        if key not in self._coarse:
            sp, to = self.wind(h, wind_scale, wind_turn)
            m = self.moist[h]; m = (m[0] + moist_add, m[1] + moist_add, m[2] + moist_add, m[3], m[4])
            F = fields(self.fuel_road, m, sp, to, self.tan, self.upslope, self.cc, self.ch, self.cbh, self.cbd, crown=crown)
            if self.l_cross is not None:
                from .landscape import road_gate
                F = road_gate(F, self.thin, self.l_cross)
            if self.l_hold is not None and h >= getattr(self, "hold_from", 0):   # A22: a weak edge is held for this hour (A24: after the wind event)
                weak = (F["rate"] > 0) & ~(F["flame"] >= self.l_hold)
                F["rate"] = np.where(weak, 0.0, F["rate"]); F["fli"] = np.where(weak, 0.0, F["fli"]); F["kind"] = np.where(weak, 0, F["kind"])
            if len(self._coarse) >= 4:
                self._coarse.pop(next(iter(self._coarse)))
            f32 = lambda a: np.asarray(a).astype(np.float32)
            self._coarse[key] = dict(rate=F["rate"], heading=F["heading"], lb=F["lb"], kind=F["kind"], flame=F["flame"],
                                     fli=f32(F["fli"]), wind=f32(sp), wind_to=f32(to))
            if self.ember_level is not None:
                es, et = self.wind(h, 1.0, wind_turn, self.ember_level)
                self._coarse[key].update(ewind=f32(es), ewind_to=f32(et))
            else:
                self._coarse[key].update(ewind=self._coarse[key]["wind"], ewind_to=self._coarse[key]["wind_to"])
        return self._coarse[key]

    def lookup(self, name, **kw):
        """s -> an object indexed [rows, cols] in solver cells that reads field `name` from the 30 m grid."""
        k = self.k
        class _Up:
            def __init__(self, a): self.a = a
            def __getitem__(self, rc): return self.a[np.asarray(rc[0]) // k, np.asarray(rc[1]) // k]
        return lambda s: _Up(self.coarse(s, **kw)[name])

    def provider(self, **kw):
        return lambda s: (lambda d: (d["rate"], d["heading"], d["lb"]))(self.slice(s, **kw))

    def start_mask(self):
        """Protocol A18: a run that starts from a mapped outline. Config key "start": {file, and time_field + time_utc
        (the row with that time) or nothing (every shape in the file)}; ignition_utc is that outline's time. Returns a
        bool mask on the 30 m grid, or None when the fire starts from a point."""
        st = self.cfg.get("start")
        if not st:
            return None
        import geopandas as gpd
        from rasterio import features
        from rasterio.transform import from_origin
        g = gpd.read_file(os.path.join(self.dir, st["file"])).to_crs(self.crs)
        if st.get("time_field"):
            t = [None if v is None or str(v) in ("", "None", "nan", "NaT") else datetime.fromisoformat(str(v).replace("Z", "+00:00")) for v in g[st["time_field"]]]
            t = [v if v is None or v.tzinfo else v.replace(tzinfo=timezone.utc) for v in t]
            g = g[[v is not None and abs((v - _utc(st["time_utc"])) / timedelta(minutes=1)) < 1.0 for v in t]]
        geoms = [(q, 1) for q in g.geometry if q is not None and not q.is_empty]
        if not geoms:
            raise ValueError("start outline: no shape matches")
        tf = from_origin(self.left, self.top, self.res, self.res)
        return features.rasterize(geoms, out_shape=(self.ny30, self.nx30), transform=tf, fill=0, dtype="uint8")[::-1].astype(bool)

    def start_phi(self):
        """Signed distance (m, negative inside) to the start outline on the solver grid, or None."""
        m = self.start_mask()
        if m is None:
            return None
        from scipy.ndimage import distance_transform_edt
        m = np.repeat(np.repeat(m, self.k, 0), self.k, 1)
        return np.where(m, -distance_transform_edt(m), distance_transform_edt(~m)) * self.cell

    def ignitions(self):
        if self.cfg.get("start"):                                # the middle of the start outline (for charts and tables)
            rr, cc = np.nonzero(self.start_mask())
            return [(int(rr.mean()) * self.k, int(cc.mean()) * self.k, 0.0, 0.0)]
        r, c = self.rowcol(self.cfg["ignition"]["lon"], self.cfg["ignition"]["lat"])
        return [(int(r), int(c), 0.0, float(self.cfg["ignition"].get("radius_m", 0.0)))]

    def buildings(self):
        """Building points (x, y in solver metres, area m2, label) from the fire's building file.
        label: 1 destroyed, 0 not, -1 unknown. The label is returned apart and must not reach the model."""
        import geopandas as gpd
        b = self.cfg["buildings"]
        g = gpd.read_file(os.path.join(os.path.dirname(self.dir), b["file"]) if b.get("national") else os.path.join(self.dir, b["file"]),
                          **({"layer": b["layer"]} if b.get("layer") else {}), **({"where": b["where"]} if b.get("where") else {}))
        g = g.to_crs(self.crs)
        x = g.geometry.x.values - self.left - 0.5 * self.cell; y = g.geometry.y.values - self.bottom - 0.5 * self.cell
        lab = np.full(len(g), -1)
        if b.get("label_field"):
            v = g[b["label_field"]].astype(str).values
            lab = np.where(np.isin(v, b["destroyed"]), 1, 0)
        extra = {k: g[k].astype(str).values for k in b.get("keep", [])}
        ok = (x > 0) & (y > 0) & (x < (self.nx - 1) * self.cell) & (y < (self.ny - 1) * self.cell)
        return x[ok], y[ok], np.full(ok.sum(), float(b.get("area_m2", 150.0))), lab[ok], {k: v[ok] for k, v in extra.items()}

    def perimeters(self):
        """List of (minutes after ignition, bool mask on the 30 m grid) for the progression perimeters inside the run."""
        import geopandas as gpd
        from rasterio import features
        from rasterio.transform import from_origin
        out = []
        tf = from_origin(self.left, self.top, self.res, self.res)
        for p in self.cfg["perimeters"]:
            g = gpd.read_file(os.path.join(self.dir, p["file"])).to_crs(self.crs)
            if p.get("time_field"):                         # A19: an outline with no time (null) is not scored
                t = [None if v is None or str(v) in ("", "None", "nan", "NaT") else datetime.fromisoformat(str(v).replace("Z", "+00:00")) for v in g[p["time_field"]]]
                t = [v if v is None or v.tzinfo else v.replace(tzinfo=timezone.utc) for v in t]
            else:
                t = [_utc(p["time_utc"])] * len(g)
            for geom, tt in zip(g.geometry, t):
                if tt is None:
                    continue
                mins = (tt - self.t0) / timedelta(minutes=1)
                if geom is None or geom.is_empty or mins <= 0 or mins > self.hours * 60 + p.get("slack_min", 0):
                    continue
                m = features.rasterize([(geom, 1)], out_shape=(self.ny30, self.nx30), transform=tf, fill=0, dtype="uint8")[::-1].astype(bool)
                out.append((float(mins), m, p["name"]))
        return sorted(out, key=lambda v: v[0])

    def detections(self):
        """Satellite fire detections for the first-night check (protocol A11): list of (minutes after ignition,
        rows30, cols30), one entry per overpass inside the run. Config key "detections": file (FIRMS csv with
        latitude, longitude, acq_date, acq_time), clip (an outline file) and buffer_m, so that other fires burning
        in the box that night are left out. Scoring only; nothing here reaches the model."""
        d = self.cfg.get("detections")
        if not d:
            return []
        import csv
        import geopandas as gpd
        from shapely.geometry import Point
        from shapely.ops import unary_union
        clip = unary_union(list(gpd.read_file(os.path.join(self.dir, d["clip"])).to_crs(self.crs).geometry)).buffer(float(d.get("buffer_m", 0.0)))
        passes = {}
        with open(os.path.join(self.dir, d["file"]), newline="") as fh:
            for row in csv.DictReader(fh):
                hhmm = str(row["acq_time"]).replace(":", "").zfill(4)
                tt = _utc(f"{row['acq_date']}T{hhmm[:2]}:{hhmm[2:4]}:00Z")
                mins = (tt - self.t0) / timedelta(minutes=1)
                if mins <= 0 or mins > self.hours * 60:
                    continue
                x, y = self.tr.transform(float(row["longitude"]), float(row["latitude"]))
                if not clip.contains(Point(x, y)):
                    continue
                r = int((y - self.bottom) // self.res); c = int((x - self.left) // self.res)
                if 0 <= r < self.ny30 and 0 <= c < self.nx30:
                    passes.setdefault(round(mins / 10.0) * 10.0, []).append((r, c))
        return [(m, np.array([v[0] for v in rc]), np.array([v[1] for v in rc])) for m, rc in sorted(passes.items())]


def overlap(model, obs):
    """Jaccard and Sorensen of two bool masks."""
    i = float((model & obs).sum()); u = float((model | obs).sum()); s = float(model.sum() + obs.sum())
    return (i / u if u else np.nan), (2 * i / s if s else np.nan)
