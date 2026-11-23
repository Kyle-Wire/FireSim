# Sandbox report (running record; failures are results)

23 Nov 2026, first solver work. Cases S1 and S2 only; S3 to S7 not yet run.

| finding | what happened | what was done |
|---|---|---|
| F1 | Shortest path along a fixed set of directions (the minimum-travel-time method, `mtt.py`) passes S1 (circle within 1.3 %) but fails S2: a length-to-breadth 3 ellipse has 83 % of its true area with 32 directions and a length-to-breadth 6 ellipse 56 %. Finer cells do not help; the error is in the directions. Fast marching over pairs of directions lifts these to 94 % and 80 %, still failing the 5 % rule. | `mtt.py` kept for reference only. The level set (`levelset.py`) is the solver. |
| F2 | First level set: fire from a point ran late, 4 % at 400 m on 10 m cells, the lag growing with the log of distance. The field goes flat behind the front and bleeds into it. | Behind the front each cell keeps falling at the rate it crossed with. S1 then within 1.5 % at 10 m, 0.8 % at 5 m, 0.4 % at 2.5 m. |
| F3 | Upwind gradient with the front normal computed apart: ellipses 15 to 37 % too wide. The flank's information travels downwind, which that scheme ignores. | Roe upwinding with a local Lax-Friedrichs fix. Area error now falls with cell size. |

## Closed cases S1 to S5 (`python sandbox/run.py <cell>`), 23 Nov 2026

| case | 10 m cells | 5 m cells |
|---|---|---|
| S1 circle | PASS (max 1.0 %, mean 0.4 %) | PASS (0.5 %, 0.2 %) |
| S2 ellipse, LB 3 | **FAIL**: head 590 of 600 m, area 6.9 % over, flank 40 of 34.3 m, back 20 of 17.7 m | **FAIL**: head 595 m and area 4.2 % over pass; flank 40 m and back 20 m still 17 % and 13 % over |
| S3 strip with a gap | **FAIL**: strip never burns; one of five points 3.5 % late (rule 3 %) | PASS (worst 1.7 %) |
| S4 wind shift | **FAIL**: head reaches 300 m at 31.8 min (6 % late; rule 4 %); new head 2.0 % early | PASS (3.0 % late, 1.4 % early) |
| S5 fast to slow fuel | PASS (2.9 % late) | PASS (1.4 %) |

Reading: every error shrinks as cells shrink, so the scheme is consistent but only first-order near the ignition point
and at sharp corners. The S2 flank and back distances are 34 m and 18 m, a few cells, and the fire starts as a disc
half a cell in radius, so those two stay over at both sizes. On 30 m fuel cells the same errors will be larger: a
wind-driven run is drawn too wide and a few percent late. Open; candidates are a sub-cell ignition and a higher-order
scheme. S6 and S7 wait for the ember code. Area over exact at other shapes: LB 6 1.18 (10 m), 1.09 (5 m), 1.05 (2.5 m);
LB 8 1.27, 1.15, 1.07.

## Fire behaviour (Rothermel), 23 Nov 2026

Checked against numbers printed in the fuel model report (Scott and Burgan 2005), `tests/test_rothermel.py`:
- Characteristic SAV and packing ratio: 39 of 40 models match the value printed on the model's page. GS4 gives 1631 1/ft
  where the page prints 1674; the table's own loads and SAVs give 1631. Left as is.
- Adjective class for spread rate and flame length at D2L2, 5 mi/h, flat (31 models with a class sentence, 62 checks,
  15 % allowance at class edges): 61 agree. SH1 spreads at 2.5 ch/h where the guide says "very low" (under 2).
No worked numeric example was found in either report, so there is not yet a digit-for-digit check of spread rate. Open:
get one (BehavePlus output or the ELMFIRE verification set) before any real-fire run.
