"""Protocol A18: a run lit over a whole mapped outline (the solver's phi0 start)."""
import numpy as np
from scipy.ndimage import distance_transform_edt as edt

from firesim.levelset import arrival


def test_fire_grows_from_an_outline_at_the_given_rate():
    ny = nx = 120; cell = 10.0
    m = np.zeros((ny, nx), bool); m[50:70, 40:60] = True
    phi0 = np.where(m, -edt(m), edt(~m)) * cell
    arr, _ = arrival(np.full((ny, nx), 5.0), 0.0, 1.0, cell, [], 60.0, phi0=phi0)
    arr[phi0 <= 0] = 0.0
    burned = np.where(np.isfinite(arr[60]))[0]
    assert abs(burned[0] - 10) <= 1 and abs(burned[-1] - 89) <= 1      # 300 m each way in 60 min at 5 m/min
    assert np.all(arr[m] == 0.0) and arr[60, 75] > arr[60, 65] > 0.0


def test_only_the_edge_of_the_outline_is_handed_to_the_spot_callback():
    ny = nx = 80; cell = 10.0
    m = np.zeros((ny, nx), bool); m[30:50, 30:50] = True
    phi0 = np.where(m, -edt(m), edt(~m)) * cell
    seen = []

    def spot(t0, t1, arr, phi, rows, cols):
        assert np.all(np.isfinite(arr[rows, cols]))                    # embers.py divides these times by the step
        seen.append((rows.copy(), cols.copy())); return []

    arr, _ = arrival(np.full((ny, nx), 5.0), 0.0, 1.0, cell, [], 10.0, phi0=phi0, spot=spot)
    r0, c0 = seen[0]
    inner = np.zeros_like(m); inner[32:48, 32:48] = True
    assert len(r0) and not inner[r0, c0].any() and np.all(arr[m] == 0.0)
