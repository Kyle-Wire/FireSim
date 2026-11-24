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

## Digit checks found (23 Nov 2026, later)

The open item above is closed. `tests/test_published.py`:
- Andrews 2018 Table 17 (three fuel beds, 5 mi/h): bulk density, characteristic SAV, optimum and relative packing ratio,
  wind factor and spread rate all match as printed (81.6, 40.2, 32.0 ft/min).
- Scott and Reinhardt 2001 Tables 5-6 (two forest stands, 40 km/h): surface intensity, initiation intensity, surface,
  initiation and active spread rates, crown fraction burned, final spread rate and intensity, and crowning index match
  within 2 % or the last printed digit (stand B crowning index 30.5 against 30). Fire type: passive and active, as printed.
- Andrews 2012 Table 5: unsheltered wind adjustment 0.36, 0.44, 0.55, 0.28 for bed depths 1, 2.5, 6, 0.2 ft, as printed.
Flag: the crown fraction burned rule reproduces the table but the text of RP-29 equation 28 has not been read.

## Synthetic landscapes, first attempt (23 Nov 2026): one solver bug found

| finding | what happened | what was done |
|---|---|---|
| F4 | On the first patchy landscapes fire appeared on the far side of a 30 m firebreak with nothing burning between, at 10 m and 15 m cells but not at 5 m or 30 m (burned area differed by up to 3.9x between cell sizes). Cause: the upwinding checked only the corner states for a change of characteristic direction, so a cell cut off behind a firebreak could keep lowering its own value and cross zero by itself. S3 (strip with a gap) had not caught it. | A minimum principle in the time step: a cell cannot fall below its lowest burnable neighbour (corner contact between two firebreak cells does not count), and never rises. S1-S5 unchanged. On the landscape that showed it, areas are now 21.9, 21.5, 20.9, 19.7 ha at 5, 10, 15, 30 m. |

Also changed after that first look, before the scored run: the landscape generator. It cut one smooth field into bands,
which wrapped every patch in a closed ring of the next class, so fires stayed in one patch and the non-burnable class
formed rings. Patches are now irregular cells about 250 m across with independent fuels. The Y3-Y5 rules in CASES.md are
unchanged. The first attempt's partial numbers (4 of 12 landscapes) are not used.

## Synthetic landscapes, scored run (23 Nov 2026, FireSim after F4), `sandbox/results_synth.json`

| case | result |
|---|---|
| Y1 slope | PASS: 300 m upslope in 148.9 min, exact 149.1 (0.1 %) |
| Y2 wind across slope | PASS: farthest point at 8.5 degrees and 338.7 m; exact 8.2 degrees and 339.4 m |
| Y3 cell size (12 landscapes, 5 m reference) | PASS at 10 m: area median 2.1 % off, worst 10.7 %; arrival time median 3.0 % off. Reported: 15 m area median 2.1 %, worst 51 %; **30 m area median 6.2 %, worst 73 % under** (seeds 1 and 8: the fire fails to get through a gap one fuel block wide), arrival time off 4 to 37 % |
| Y4 more wind | PASS on 12 of 12: 8 m/s burns more than 4 m/s (by 1.2x to 74x); no non-burnable cell burned; reruns identical |
| Y5 wetter fuel | PASS on 12 of 12: dead fuel 4 points wetter burns 0.08x to 0.87x the area |

Reading: run the solver on cells finer than the fuel map. At the fuel map's own 30 m a fire can be stopped by a
passage one block wide that a finer solver burns through, and that is the difference between 3 and 11 hectares. The
real-fire runner therefore uses 10 m solver cells on 30 m fuels. Y4 also shows how sharply outcomes turn on wind when
a fire sits near a fuel boundary (0.5 ha at 4 m/s against 39 ha at 8 m/s on seed 1): single runs are not enough, and
the product must report an ensemble.

## Embers and houses, cases S6 to S13b (23 Nov 2026), `sandbox/results_wui.json`

Rules in `CASES.md`, fixed before `embers.py` and `structures.py` ran. `python sandbox/run_wui.py`.

| case | result | numbers |
|---|---|---|
| S6 embers on with count 0 = S1 | pass | bit-identical |
| S7 ember jump over a full strip | pass | strip never burns; first fire behind it 0.06 min after launch + flight delay |
| S8 house layer with no houses = S1 | pass | bit-identical |
| S9 house chain | pass | house n lights at exactly 5 n min; far row never lights |
| S10 neighbour odds | pass | 0.344 lit against 0.3635 expected, tolerance 0.023 |
| S11 flame reach | pass | 0.208 lit against the code's stated 0.207, tolerance 0.011; none with k_f = 0 |
| S12 hardening | pass, weakly | 134.8 of 144 houses lost at ember factor 1, 130.6 at 0.5; none at factor 0 with the other paths off. The block is so exposed that nearly every house is lost either way, so this case says little about how much hardening helps |
| S13 spot window against whole grid, random embers | **fail** | area differed by up to 8 to 12 % between the two rules |
| S13b the same with one recorded list of spot fires replayed under both rules | pass | area differs by 0.0 %, arrival by 0.0 % |

F5 (a test that tested the wrong thing). S13 failed because once two runs differ by a single cell they draw different
embers, so it compared two random fires. S13b, added with that reason written down before it ran, replays one list
of spot fires under both rules and they agree. The spot window is kept.

## Narrow band (23 Nov 2026)

A town-sized fire at 10 m cells is tens of millions of cells, and the first solver updated all of them every step
with the time step set by the fastest cell anywhere. The solver now works only on cells within 8 cells of the front,
rebuilds that band every 6 steps and takes its time step from the fastest cell in the band (`levelset._band`).
A cell entering the band continues the slope of its neighbours already in it. Two wrong turns on the way, both
caught by cases already in this sandbox: lowering new cells through non-burnable cells let fire through the S3 strip
16 % early; lowering every band cell to "neighbour plus distance" made the synthetic fires 5 to 9 % larger.

Against the whole-grid solver (`band=None`): S1 to S5 agree to the third digit; on the 12 synthetic landscapes burned
area is 1 % larger at the median (range -1 to +5 %). Y1 to Y5 pass with the band (10 m against 5 m: area median 1.9 %,
worst 10.9 %; arrival 2.7 %). `results_synth.json` is now the band run.

Two things learned about running it: two FireSim runs at once on the same cores slow each other about twentyfold
(thread contention), so runs go one at a time or with `NUMBA_NUM_THREADS` set; and rebuilding the band costs about
as much as ten steps, which is the next thing to speed up when embers light a spot fire nearly every step.

## Incremental band (24 Nov 2026)

The first Camp runs with embers took 53 to 131 minutes, and profiling a spot-heavy case showed three quarters of
the time in rebuilding the band, which every new spot fire forced. The band is now kept and grown: each cell that
catches fire adds the unburned cells within 8 cells of it (`_grow`), and every 16 steps burned cells with nothing
unburned within two cells are dropped (`_prune`). The spot-heavy benchmark went from 75 s to 19 s.
Rechecked: S1 to S5 the same to the third digit; S6 to S12 and S13b pass, S13 fails as before; Y1 to Y5 pass
(10 m against 5 m: area median 1.8 %, worst 11.2 %); against the whole-grid solver the burned area ratio is 1.01 at
the median, 0.99 to 1.05. `scripts/run_fire.py` now prints progress each simulated hour and stops itself at
`--max-minutes` (default 110), after a run on THE-COW overran a two-hour limit that an outside watcher failed to enforce.

## Road rule R1 (24 Nov 2026, protocol A8)

S14 passes (tests/test_roads.py): with the rule off the result is bit-identical to the old code; with L_cross = 0
the fire passes a one-cell road line exactly as if it were grass; a two-cell line and a high L_cross stop it.
The rule is off by default (`run_fire.py --road-cross <m>` turns it on). L_cross has no source: it is a number to
fit on the calibration fires.

