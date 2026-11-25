"""Wind option W3 (gridzip-lab physics/PROTOCOL-FIRE.md, amendment A15): terrain wind from WindSim.

WindSim's steady RANS solver is run on terrain alone for a few inflow bearings (its `results/terrain_wind.npz`).
Each run gives, at every mesh column, a speed-up S = 10 m speed / the run's own 80 m speed averaged over the upwind
third of its domain, and a turning (the 10 m wind's bearing minus the inflow's). For a fire hour the reference is
the HRRR 80 m wind averaged over the middle run's upwind third; the 10 m wind is S x the reference speed, turned by
the turning, with S and turning interpolated between bearings. Hours whose reference bearing is more than `slack`
degrees outside the runs' range, and points outside the runs' inner domain, keep the wind they were given.
Nothing is fitted. Neutral flow: no stable layer, so no downslope windstorm or gap jet.

Provenance: written fresh. Bearings are degrees clockwise from true north; a run's own bearing is measured from its
UTM grid north and is brought to true north with the grid convergence passed in.
"""
import numpy as np


def _wrap(d):
    return (np.asarray(d) + 180.0) % 360.0 - 180.0


def box_mean(xl, yl, v, size_m=3000.0):
    """Mean of v over a size_m box around each mesh column (columns sit on a regular lattice in the run's frame)."""
    from scipy.ndimage import uniform_filter
    ux = np.unique(np.round(xl, 1)); step = float(np.median(np.diff(ux))) if len(ux) > 1 else 1.0
    ix = np.rint((xl - xl.min()) / step).astype(int); iy = np.rint((yl - yl.min()) / step).astype(int)
    g = np.zeros((iy.max() + 1, ix.max() + 1)); w = np.zeros_like(g)
    np.add.at(g, (iy, ix), v); np.add.at(w, (iy, ix), 1.0)
    k = 2 * int(size_m / step / 2.0) + 1                     # an odd count of cells, so the box is centred
    num = uniform_filter(g, size=k, mode="constant"); den = uniform_filter(w, size=k, mode="constant")
    return (num / np.maximum(den, 1e-9))[iy, ix]


def load_run(path):
    """One WindSim terrain run as a dict: bearing_from (grid), x, y (UTM), S, turn (deg), inner and upwind masks."""
    d = np.load(path)
    half, blend = float(d["half_width_m"]), float(d["edge_blend_m"])
    xl, yl = d["x_local"], d["y_local"]
    inner = np.maximum(np.abs(xl), np.abs(yl)) < half - blend
    up = inner & (xl < -half / 3.0)
    ref = float(np.hypot(d["east_80m"], d["north_80m"])[up].mean())
    e, n = d["east_10m"], d["north_10m"]
    toward = np.degrees(np.arctan2(e, n))
    S = np.hypot(e, n) / ref
    return dict(bearing_from=float(d["direction_deg"]), epsg=int(d["epsg"]), x=d["x_utm"], y=d["y_utm"], S=S,
                Sc=S / np.maximum(box_mean(xl, yl, S), 1e-6),
                S80=np.hypot(d["east_80m"], d["north_80m"]) / ref, ground=d["ground_m"] if "ground_m" in d.files else None,
                turn=_wrap(toward - (float(d["direction_deg"]) + 180.0)), inner=inner, up=up, ref80=ref)


def blend(bearing_from, ref_speed, bearings, S, turn, slack=15.0):
    """10 m wind (true east, north) from the reference: S and turn are (n_bearings, ...) on sorted `bearings`.
    Returns None when the reference bearing is more than `slack` degrees outside the bearings' range."""
    b = np.asarray(bearings, float)
    x = b[0] + _wrap(bearing_from - b[0])                     # same branch as the runs
    if x < b[0] - slack or x > b[-1] + slack:
        return None
    x = float(np.clip(x, b[0], b[-1]))
    j = int(np.clip(np.searchsorted(b, x) - 1, 0, len(b) - 2)) if len(b) > 1 else 0
    w = 0.0 if len(b) == 1 else (x - b[j]) / (b[j + 1] - b[j])
    j1 = min(j + 1, len(b) - 1)
    s = (1 - w) * S[j] + w * S[j1]; t = (1 - w) * turn[j] + w * turn[j1]
    toward = np.radians(bearing_from + 180.0 + t)
    return s * ref_speed * np.sin(toward), s * ref_speed * np.cos(toward)


class TerrainWind:
    def __init__(self, paths, to_utm, convergence_utm_deg=0.0, slack=15.0):
        """to_utm(X, Y): the caller's points to the runs' UTM metres."""
        self.runs = sorted((load_run(p) for p in paths), key=lambda r: r["bearing_from"])
        self.to_utm = to_utm; self.gamma = float(convergence_utm_deg); self.slack = slack
        self.bearings = [r["bearing_from"] + self.gamma for r in self.runs]
        self._key = None; self.hours_used = 0; self.hours_kept = 0

    def prepare(self, GX, GY):
        from scipy.interpolate import LinearNDInterpolator
        from scipy.spatial import cKDTree
        key = (GX.shape, float(GX.ravel()[0]), float(GY.ravel()[-1]))
        if key == self._key:
            return
        x, y = self.to_utm(GX, GY)
        S, T, S80, Sc = [], [], [], []
        for r in self.runs:
            pts = np.c_[r["x"], r["y"]]
            f = LinearNDInterpolator(pts, np.c_[r["S"], np.cos(np.radians(r["turn"])), np.sin(np.radians(r["turn"])), r["S80"], r["Sc"]])
            v = f(x, y); ok = r["inner"][cKDTree(pts).query(np.c_[np.ravel(x), np.ravel(y)])[1]].reshape(np.shape(x))
            v[~ok] = np.nan
            S.append(v[..., 0]); T.append(np.degrees(np.arctan2(v[..., 2], v[..., 1]))); S80.append(v[..., 3]); Sc.append(v[..., 4])
        self.S, self.turn, self.S80, self.Sc = np.array(S), np.array(T), np.array(S80), np.array(Sc)
        mid = self.runs[len(self.runs) // 2]
        d, _ = cKDTree(np.c_[mid["x"][mid["up"]], mid["y"][mid["up"]]]).query(np.c_[np.ravel(x), np.ravel(y)])
        step = np.sqrt((mid["x"].max() - mid["x"].min()) * (mid["y"].max() - mid["y"].min()) / len(mid["x"]))
        self.ref_pts = (d < 1.5 * step).reshape(np.shape(x))   # the caller's points inside the middle run's upwind third
        self._key = key

    def wind(self, GX, GY, e10, n10, e80, n80, clearing_z0=None, contrast=False):
        """10 m wind at the points: W3 where it applies, else the wind passed in. Returns (east, north, used).
        clearing_z0 (A15 note 2, station tables only): the run's 80 m wind brought down to 10 m by the log law over
        that roughness, instead of the run's own 10 m wind over its land-cover roughness."""
        self.prepare(GX, GY)
        if not self.ref_pts.any():
            self.hours_kept += 1; return e10, n10, False
        re, rn = float(np.mean(e80[self.ref_pts])), float(np.mean(n80[self.ref_pts]))
        ref_speed = float(np.mean(np.hypot(e80, n80)[self.ref_pts]))
        S = self.S if clearing_z0 is None else self.S80 * np.log(10.0 / clearing_z0) / np.log(80.0 / clearing_z0)
        self.last_bearing = float(np.degrees(np.arctan2(-re, -rn)) % 360.0)
        out = blend(self.last_bearing, 1.0 if contrast else ref_speed, self.bearings, self.Sc if contrast else S, self.turn, self.slack)
        if out is None:
            self.hours_kept += 1; return e10, n10, False
        if contrast:                                             # out is (ratio, turned unit vector of the reference): keep HRRR's 10 m level and bearing
            ratio = np.hypot(out[0], out[1])
            t = np.radians(np.degrees(np.arctan2(out[0], out[1])) - (self.last_bearing + 180.0))
            out = (ratio * (e10 * np.cos(t) + n10 * np.sin(t)), ratio * (n10 * np.cos(t) - e10 * np.sin(t)))
        ok = np.isfinite(out[0])
        self.hours_used += 1
        return np.where(ok, out[0], e10), np.where(ok, out[1], n10), True
