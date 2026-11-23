# Sandbox report (running record; failures are results)

23 Nov 2026, first solver work. Cases S1 and S2 only; S3 to S7 not yet run.

| finding | what happened | what was done |
|---|---|---|
| F1 | Shortest path along a fixed set of directions (the minimum-travel-time method, `mtt.py`) passes S1 (circle within 1.3 %) but fails S2: a length-to-breadth 3 ellipse has 83 % of its true area with 32 directions and a length-to-breadth 6 ellipse 56 %. Finer cells do not help; the error is in the directions. Fast marching over pairs of directions lifts these to 94 % and 80 %, still failing the 5 % rule. | `mtt.py` kept for reference only. The level set (`levelset.py`) is the solver. |
| F2 | First level set: fire from a point ran late, 4 % at 400 m on 10 m cells, the lag growing with the log of distance. The field goes flat behind the front and bleeds into it. | Behind the front each cell keeps falling at the rate it crossed with. S1 then within 1.5 % at 10 m, 0.8 % at 5 m, 0.4 % at 2.5 m. |
| F3 | Upwind gradient with the front normal computed apart: ellipses 15 to 37 % too wide. The flank's information travels downwind, which that scheme ignores. | Roe upwinding with a local Lax-Friedrichs fix. Area error now falls with cell size. |

S2 as it stands (area over exact; head distance at 60 min, exact 600 m):

| cell | LB 3 | LB 6 | LB 8 | head, LB 3 |
|---|---|---|---|---|
| 10 m | 1.07 | 1.18 | 1.27 | 590 m |
| 5 m | 1.04 | 1.09 | 1.15 | 595 m |
| 2.5 m | 1.02 | 1.05 | 1.07 | 597.5 m |

**S2 verdict: fails the 5 % area rule at 10 m cells (7 % over at LB 3), passes at 5 m and 2.5 m.** The half-width of an LB 6
ellipse at one hour is 50 m, five 10 m cells, so a narrow wind-driven run is drawn too wide on coarse cells. Open.
