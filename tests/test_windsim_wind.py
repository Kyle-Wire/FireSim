"""Protocol A15, wind option W3: the blend of WindSim terrain runs with a reference wind."""
import numpy as np

from firesim.windsim import TerrainWind, blend, load_run


def run_file(tmp_path, direction, speedup, turn_deg, half=10000.0, blend_m=2000.0, step=500.0):
    """A made-up terrain run: 80 m wind 12 m/s along the inflow, 10 m wind = speedup x 12 turned by turn_deg."""
    g = np.arange(-half + step / 2, half, step); xl, yl = [a.ravel() for a in np.meshgrid(g, g)]
    b = np.radians(direction + 180.0); ex, ey = np.sin(b), np.cos(b)
    x = 600000.0 + xl * ex - yl * ey; y = 4400000.0 + xl * ey + yl * ex
    t = np.radians(direction + 180.0 + turn_deg)
    p = tmp_path / f"r{direction:g}.npz"
    np.savez(p, direction_deg=direction, epsg=32610, half_width_m=half, edge_blend_m=blend_m, x_local=xl, y_local=yl, x_utm=x, y_utm=y,
             east_80m=np.full(len(x), 12 * ex), north_80m=np.full(len(x), 12 * ey),
             east_10m=np.full(len(x), speedup * 12 * np.sin(t)), north_10m=np.full(len(x), speedup * 12 * np.cos(t)))
    return str(p)


def test_run_gives_speedup_and_turning(tmp_path):
    r = load_run(run_file(tmp_path, 45.0, 0.6, 10.0))
    assert np.allclose(r["S"], 0.6) and np.allclose(r["turn"], 10.0) and np.isclose(r["ref80"], 12.0)
    assert r["up"].any() and not (r["up"] & ~r["inner"]).any()


def test_blend_between_bearings_and_out_of_range():
    S = np.array([[0.5], [0.7]]); T = np.array([[0.0], [20.0]])
    e, n = blend(45.0, 10.0, [30.0, 60.0], S, T)
    sp = np.hypot(e, n); toward = np.degrees(np.arctan2(e, n)) % 360
    assert np.allclose(sp, 6.0) and np.allclose(toward, 45.0 + 180.0 + 10.0)
    e, n = blend(70.0, 10.0, [30.0, 60.0], S, T)                       # within the slack: the nearest run, the hour's own bearing
    assert np.allclose(np.hypot(e, n), 7.0) and np.allclose(np.degrees(np.arctan2(e, n)) % 360, 70.0 + 180.0 + 20.0)
    assert blend(100.0, 10.0, [30.0, 60.0], S, T) is None and blend(200.0, 10.0, [30.0, 60.0], S, T) is None


def test_terrain_wind_inside_and_outside(tmp_path):
    tw = TerrainWind([run_file(tmp_path, 30.0, 0.5, 0.0), run_file(tmp_path, 60.0, 0.7, 0.0)], lambda X, Y: (X + 600000.0, Y + 4400000.0))
    g = np.arange(-15000.0, 15001.0, 1000.0); GX, GY = np.meshgrid(g, g)
    b = np.radians(45.0 + 180.0); e80 = np.full(GX.shape, 20 * np.sin(b)); n80 = np.full(GX.shape, 20 * np.cos(b))
    e10 = np.full(GX.shape, 1.0); n10 = np.zeros(GX.shape)
    e, n, used = tw.wind(GX, GY, e10, n10, e80, n80)
    mid = (np.abs(GX) < 4000) & (np.abs(GY) < 4000); far = np.maximum(np.abs(GX), np.abs(GY)) > 14000
    assert used and np.allclose(np.hypot(e, n)[mid], 0.6 * 20.0) and np.allclose(e[mid] / n[mid], 1.0)   # from the north-east
    assert np.allclose(e[far], 1.0) and np.allclose(n[far], 0.0)        # outside the runs: the wind passed in
    e, n, used = tw.wind(GX, GY, e10, n10, -e80, -n80)                  # reference from the south-west: W3 stands down
    assert not used and np.allclose(e, 1.0) and tw.hours_kept == 1 and tw.hours_used == 1
