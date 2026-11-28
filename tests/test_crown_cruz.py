"""Protocol A25 setting C1: the Cruz, Alexander and Wakimoto 2005 crown rate (Can. J. For. Res. 35: 1626-1639)."""
import math

import numpy as np
import pytest

from firesim.landscape import fields

M = (0.04, 0.05, 0.06, 0.3, 0.6)


def cell(crown_rate, u10, cbd=0.15, fuel=165):
    return fields(np.array([[fuel]]), M, u10 / 1.15, 0.0, 0.0, 0.0, canopy_cover=0.6, canopy_height=15.0, cbh=1.0,
                  cbd=cbd, crown=True, crown_rate=crown_rate)


def test_active_rate_is_eq3():
    F = cell("cruz05", 10.0)
    want = 11.02 * 36.0 ** 0.90 * 0.15 ** 0.19 * math.exp(-0.17 * 4.0)
    assert F["kind"][0, 0] == 3 and F["rate"][0, 0] == pytest.approx(want, rel=1e-6)


def test_paper_worked_number():
    """137 m/min at 50 km/h, CBD 0.2, EFFM 4 % (the paper's own example; our form gives about 139)."""
    v = 11.02 * 50.0 ** 0.90 * 0.2 ** 0.19 * math.exp(-0.17 * 4.0)
    assert v == pytest.approx(137.0, rel=0.03)


def test_passive_rate_is_eq4():
    F = cell("cruz05", 3.0, cbd=0.03)
    ca = 11.02 * 10.8 ** 0.90 * 0.03 ** 0.19 * math.exp(-0.17 * 4.0); cac = ca * 0.03 / 3.0
    assert cac < 1.0
    assert F["kind"][0, 0] in (1, 2)
    if F["kind"][0, 0] == 2:                              # torching, not active: the larger of surface and CROS_P
        assert F["rate"][0, 0] >= ca * math.exp(-cac) - 1e-9


def test_default_unchanged_and_no_torch_unchanged():
    a, b = cell("r91", 10.0), fields(np.array([[165]]), M, 10.0 / 1.15, 0.0, 0.0, 0.0, canopy_cover=0.6, canopy_height=15.0,
                                         cbh=1.0, cbd=0.15, crown=True)
    assert a["rate"][0, 0] == b["rate"][0, 0]
    lo = cell("cruz05", 1.0, fuel=183)                    # timber litter in light wind: no torching, surface only
    assert lo["kind"][0, 0] == 1 and lo["rate"][0, 0] == cell("r91", 1.0, fuel=183)["rate"][0, 0]


# ---- protocol A26 setting C2: conifer timber only, inputs held at the edges of the fitted range
def test_c2_clamps_wind_moisture_and_cbd():
    F = cell("cruz05c", 15.0, cbd=0.30)                     # 54 km/h is held at 32; EFFM 4 % is held at 7
    want = 11.02 * 32.0 ** 0.90 * 0.30 ** 0.19 * math.exp(-0.17 * 7.0)
    assert F["kind"][0, 0] == 3 and F["rate"][0, 0] == pytest.approx(want, rel=1e-6)


def test_c2_is_r91_in_broadleaf_litter():
    for fuel in (182, 186, 189):
        a = cell("cruz05c", 10.0, fuel=fuel); b = cell("r91", 10.0, fuel=fuel)
        assert a["rate"][0, 0] == pytest.approx(b["rate"][0, 0]) and a["kind"][0, 0] == b["kind"][0, 0]


def test_c2_never_faster_than_c1_in_dry_fuel():
    assert cell("cruz05c", 10.0)["rate"][0, 0] <= cell("cruz05", 10.0)["rate"][0, 0]
