"""Digit checks against worked numbers in the source reports.
- Andrews 2018 (RMRS-GTR-371) Table 17, p. 59: 2 tons/ac split between 1-h (SAV 2,500) and 10-h, depth 1 ft, moisture of
  extinction 20 %, dead moisture 5 %, midflame wind 5 mi/h, flat.
- Scott and Reinhardt 2001 (RMRS-RP-29) Tables 5-6, p. 28-29: two Bitterroot stands at 40 km/h open wind, slope 20 %.
- Andrews 2012 (RMRS-GTR-266) Table 5, p. 24, Baughman and Albini column: unsheltered wind adjustment factors."""
import pytest
from firesim.rothermel import Fuel, TONS_AC, bed, spread
from firesim.crown import FM5, FM10, fire, waf_unsheltered


@pytest.mark.parametrize("l1,l10,sav,bop,ratio,phiw,ros", [(2, 0, 2500, .0055, .5195, 17.8, 81.6),
                                                            (1, 1, 2400, .0057, .5025, 17.9, 40.2),
                                                            (0.1, 1.9, 1417, .0088, .3263, 20.8, 32.0)])
def test_gtr371_table17(l1, l10, sav, bop, ratio, phiw, ros):
    f = Fuel("T", 0, (l1 * TONS_AC, l10 * TONS_AC, 0, 0, 0), (2500, 109, 30, 9999, 9999), 1.0, 0.20, 8000, False)
    r = spread(f, (.05, .05, .05, 1, 1), 5 * 0.44704); b = bed(f)
    assert b["rho_b"] == pytest.approx(0.0918, abs=5e-5)
    assert b["sigma"] == pytest.approx(sav, abs=0.6)
    assert b["beta_op"] == pytest.approx(bop, abs=6e-5)
    assert r["beta_ratio"] == pytest.approx(ratio, abs=6e-5)
    assert r["phi_w"] == pytest.approx(phiw, abs=0.06)
    assert r["ros_ft_min"] == pytest.approx(ros, abs=0.06)


STANDS = [("A", FM5, (.05, .06, .08, 1.17, 1.17), .15, 1.5, .06, 1.22,
           dict(i_surface=470, i_init=309, r_surface=4.6, r_init=3.0, r_sa=9.2, r_active=23.1, r_active_crit=50,
                cfb=0.25, r_final=9.2, i_final=1788, ci=70), "passive"),
          ("B", FM10, (.08, .09, .10, 1.17, 1.17), .12, 0.9, .21, 2.25,
           dict(i_surface=313, i_init=144, r_surface=1.4, r_init=0.6, r_sa=1.1, r_active=20.4, r_active_crit=14,
                cfb=1.0, r_final=20.4, i_final=18337, ci=30), "active")]


@pytest.mark.parametrize("name,fuel,m,wrf,cbh,cbd,load,exp,kind", STANDS)
def test_rp29_table6(name, fuel, m, wrf, cbh, cbd, load, exp, kind):
    r = fire(fuel, m, 40 / 3.6, 0.20, wrf, cbh, cbd, load)
    assert r["type"] == kind
    r["ci"] = r["crowning_index"] * 3.6
    for k, v in exp.items():
        # the table prints 2 to 3 figures; allow 2 % or half the last printed digit, whichever is larger (stand B crowning index comes out 30.5 against a printed 30)
        tol = max(0.02 * abs(v), 0.05 if v < 10 else 0.5)
        assert r[k] == pytest.approx(v, abs=tol), (k, r[k], v)


@pytest.mark.parametrize("depth,exp", [(1.0, 0.36), (2.5, 0.44), (6.0, 0.55), (0.2, 0.28)])
def test_waf(depth, exp):
    assert waf_unsheltered(depth) == pytest.approx(exp, abs=0.005)
