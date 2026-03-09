# Safety Scorecard

Multi-dimensional safety assessment that orchestrates simulation, threat
analysis, Monte Carlo risk analysis, and policy evaluation to produce a
comprehensive scorecard with letter grades (A+ through F) per dimension.

## Quick Start

```python
from replication.scorecard import SafetyScorecard, ScorecardConfig

sc = SafetyScorecard()
result = sc.evaluate()
print(result.render())
print(f"Overall: {result.overall_grade} ({result.overall_score}/100)")
```

## Key Classes

- **`SafetyScorecard`** — Runs multiple analysis passes (simulation,
  threats, Monte Carlo, policy) and combines results into a single
  graded scorecard.
- **`ScorecardResult`** — Contains per-dimension scores, overall grade,
  detailed breakdown, and rendered report.
- **`DimensionScore`** — Score for a single safety dimension (0–100)
  with letter grade and contributing metrics.
- **`ScorecardConfig`** — Configuration: scenario preset, strategy,
  max depth, Monte Carlo run count, policy preset.

## Grading Scale

| Grade | Score |
|-------|-------|
| A+ | 97–100 |
| A | 93–96 |
| A− | 90–92 |
| B+ | 87–89 |
| B | 83–86 |
| … | … |
| F | 0–59 |

## CLI

```bash
python -m replication.scorecard                           # default
python -m replication.scorecard --scenario balanced       # from preset
python -m replication.scorecard --mc-runs 50              # Monte Carlo runs
python -m replication.scorecard --policy strict           # policy preset
python -m replication.scorecard --json                    # JSON output
python -m replication.scorecard --quick                   # skip slow analyses
```
