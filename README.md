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
