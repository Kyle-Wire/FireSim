# FireSim

SPARC Labs' wildfire engine: fire spread, embers and house ignition at property level, for the Climate Home
Inspection report. Written fresh; no third-party model code.

**Status (23 Nov 2026): early build. Nothing here has been scored against a real fire.** The test rules were fixed
first in `gridzip-lab/physics/PROTOCOL-FIRE.md` (four fires to tune on, six held back blind).

- `firesim/levelset.py`: the spread solver (level set). `firesim/mtt.py`: a minimum-travel-time solver, rejected
  (sandbox finding F1), kept for reference.
- `firesim/rothermel.py` + `firesim/data/fbfm40.csv`: surface fire behaviour for the 40 Scott and Burgan fuel models.
- `sandbox/CASES.md`: closed scenes whose answers are fixed before the code runs. `sandbox/REPORT.md`: what happened,
  failures included.

```bash
pip install numpy numba matplotlib pytest
pytest -q
python sandbox/figures.py out/
```

Every constant states its source, and only sources that were opened are cited.

## Directive for agents: issues go through the compressor (Kyle, 24 Nov 2026)

Kyle's words: "when we run into issues, we should always try to update compressor to include formulas for the
specific domain, and then run it against the model."

So when a FireSim run shows a problem (a fire too slow, too big, wrong houses lost, a wind or fuel mismatch), the
first step is to add the formulas of that domain (for example spread under calm wind, downslope wind, ember
transport, suppression or burnout) to the compressor in `gridzip-lab`, then run the compressor against the model's
output, before tuning a FireSim parameter by hand. Record what the compressor found next to the run's results.
