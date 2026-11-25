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
