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
