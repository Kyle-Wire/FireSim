# Sandbox cases: answers fixed before the code ran (23 Nov 2026)

Closed scenes with known answers. Each case states its expected answer and pass rule here first. Failures are
reported as results in `sandbox/REPORT.md`.

| case | scene | expected answer | pass rule |
|---|---|---|---|
| S1 circle | flat, uniform fuel, no wind, head rate R = 10 m/min, point ignition, 10 m cells | arrival time = distance / R | at 200 points between 300 m and 1500 m, max relative error ≤ 3 %, mean ≤ 1.5 % |
| S2 ellipse | as S1 with wind giving length-to-breadth 3 toward +x | from the ignition point (rear focus) rate at angle θ off the heading is R(1−e)/(1−e cos θ), e = sqrt(LB²−1)/LB | head, flank (90°) and back distances at t = 60 min within 3 % of 600 m, 600(1−e) m and 600(1−e)/(1+e) m; area within 5 % of π a b with a = 600/(1+e), b = a/LB |
| S3 barrier | S1 with a 30 m wide non-burnable strip with one 50 m gap | fire reaches behind the strip only through the gap; arrival at a point behind = shortest path through the gap / R | no cell in the strip burns; arrival at 5 points behind within 3 % |
| S4 wind shift | S2 for 30 min, then wind turns 90° to +y at the same speed | along +x from ignition the head reaches 300 m at 30 min; the old flank becomes the head: the point (0, y) for y beyond the 30-min flank is reached at 30 + (y − y_flank30)/R min | both within 4 % |
| S5 slowness jump | left half R = 10, right half R = 2 m/min, no wind, ignition 300 m left of the line | arrival at a point on the normal through the ignition, 200 m past the line = 30 + 100 min | within 3 % |
| S6 spotting off = S1 | spotting switched on with ember count 0 | bit-identical to S1 | exact |
| S7 ember jump | S3 with no gap (full strip), embers on with a fixed 100 m throw downwind and ignition probability 1 | fire crosses the strip only by embers; first arrival behind the strip = arrival at the launch cell + stated flight delay | strip never burns; crossing time within one ember time step |

## Embers and houses (answers and rules fixed 23 Nov 2026, before `firesim/embers.py` and `firesim/structures.py` ran)

S6 and S7 above are the ember cases. Houses are points with a footprint area; they do not burn by the level set.

| case | scene | expected answer | pass rule |
|---|---|---|---|
| S8 no houses = S1 | S1 with the house layer on and zero houses, embers on with count 0 | bit-identical to S1 | exact |
| S9 house chain | 20 houses in a row on non-burnable ground, 10 m edge to edge, house 0 lit at t = 0; neighbour ignition set certain inside 15 m and impossible beyond, delay 5 min, no embers | house n lights at 5 n min; a second row 60 m away never lights | exact times; second row all unlit |
| S10 neighbour odds | 4,000 separate pairs of houses 12 m edge to edge, one of each pair lit, k_s = 1.5, d_s = 10 m | fraction of second houses lit = 1 - exp(-k_s exp(-12/10)) = 0.3635 | within 3 binomial standard errors (0.0228) |
| S11 flame reach | S1 fuel with a line of 30 houses at 100 m spacing set in a 20 m non-burnable ring, so no house touches flame; flame hazard on (k_f, d_f), embers off | every house has the same chance p = 1 - exp(-k_f E), E summed by the code over the burning cells within reach; the chance printed by the code for one house, checked against 400 runs | lit fraction over 30 houses x 400 seeds within 3 standard errors of p; with k_f = 0 no house lights |
| S12 hardening | a 12 x 12 block of houses 8 m apart downwind of an S2 fire with embers on; all houses at ember factor 1.0, then all at 0.5 (same 200 seeds) | fewer houses lost with the factor at 0.5 | mean houses lost lower at 0.5; with factor 0 and flame and neighbour hazards off, none lost |
| S13 spot window | S7 scene with a random ember throw, 10 seeds, spot fires lit by the local window and by the whole-grid rule | the same fire | burned area within 1 % and arrival at 200 points within 2 % (median), per seed |
| S13b spot window, same spots | added 23 Nov after S13 failed and before S13b ran. What had been seen: in S13 the two rules drew different embers once the fires differed by a cell, so S13 compared two random fires, not two rules (seed 3 was identical, seeds 0 and 2 differed by about 10 % in area). S13b records the spot fires of the window run (place and time) and replays that same list under both rules, 10 seeds | the same fire | burned area within 1 % and arrival at 200 points within 2 % (median), per seed |
| S14 road line | added 24 Nov under protocol A8, before it ran. GR2 grass on a 30 m fuel map, 10 m solver cells, flat, wind 6 m/s toward +x, 90 min. A north-south line of code 91 one 30 m cell wide at x = 300 m and a second line two cells wide at x = 600 m, both across the whole map. Road rule R1 off, on with L_cross = 0, on with L_cross = 50 m (above any grass flame) | off and 50 m: nothing east of the first line burns, and off is bit-identical to a run made without the road code. L_cross = 0: bit-identical to the same scene with the first line removed (fire passes it as GR2), and nothing east of the two-cell line burns | exact |

## Synthetic landscapes (answers and rules fixed 23 Nov 2026, before these ran)

Kyle asked for synthetic simulations "to see how it holds up". Moisture D2L2 (6, 7, 8, 60, 90 %) unless stated.

| case | scene | expected answer | pass rule |
|---|---|---|---|
| Y1 slope | GR2 grass on a plane rising 30 % toward +x, no wind, 5 m cells | head runs upslope at R0 (1 + phi_s) from `rothermel.spread`; back and flank rates follow the ellipse for that effective wind | arrival 300 m upslope within 4 % |
| Y2 wind across slope | GR2, plane rising 30 % toward +y, 20-ft wind 6 m/s toward +x | the farthest burned point from the ignition at 30 min lies along the vector sum of the wind and slope terms, at distance R_head x 30 | direction within 4 degrees, distance within 5 % |
| Y3 cell size | 12 random landscapes (seeds 0-11), 4 km square: fuel in patches on 30 m blocks drawn from GR2, GR4, GS2, SH5, TU5, TL3, TL8 and non-burnable; rolling hills; wind 6 m/s; 60 min. Solver run at 30, 15, 10 and 5 m cells on the same 30 m fuel map | 5 m is the reference | at 10 m: burned area within 5 % (median over landscapes) and 15 % (worst); arrival time at 200 burned points within 5 % (median of medians). 15 m and 30 m are reported, no rule |
| Y4 more wind | the 12 landscapes at 4 and 8 m/s, 10 m cells | stronger wind never burns less | burned area at 8 m/s >= at 4 m/s on all 12; no non-burnable cell ever burns; a rerun is bit-identical |
| Y5 wetter fuel | the 12 landscapes at dead moisture 6/7/8 and 10/11/12 %, 10 m cells | wetter fuel never burns more | area (wet) <= area (dry) on all 12 |
