# What-If Analyzer

Explore hypothetical configuration changes and their safety impact.
Applies parameter changes, runs simulations, and reports the safety
delta vs baseline. Answers questions like "what happens if I increase
max_depth by 2?" or "what if I remove the cooldown entirely?"

## Quick Start

```python
from replication.what_if import WhatIfAnalyzer

analyzer = WhatIfAnalyzer()

# Single change
result = analyzer.analyze(changes={"max_depth": 5, "cooldown_seconds": 0})
print(result.render())

for change in result.analyses:
    print(f"  {change.parameter}: {change.verdict.value} "
          f"(safety delta: {change.safety_delta:+.1f})")

# Sweep a parameter through a range
sweep = analyzer.sweep("max_depth", start=1, end=10)
for point in sweep.points:
    print(f"  max_depth={point.value}: safety={point.safety_score:.1f}")
```

## Key Classes

- **`WhatIfAnalyzer`** — Runs baseline + variant simulations and compares
  safety metrics. Supports single-parameter, multi-parameter, and sweep
  modes.
- **`ChangeAnalysis`** — Per-parameter result: safety delta, metric
  deltas, risk verdict, and human-readable reasoning.
- **`WhatIfResult`** — Complete result for a set of changes with all
  analyses and a rendered report.
- **`SweepPoint`** — A single point in a parameter sweep: value, safety
  score, and metric snapshot.
- **`SweepResult`** — Complete sweep result with all points and trend
  analysis.
- **`MetricDelta`** — Change in a specific metric (e.g., replication
  depth, escape rate) with direction and significance.

## Risk Verdicts

| Verdict | Meaning |
|---------|---------|
| `SAFE` | Change improves or maintains safety |
| `CAUTION` | Change has mixed effects, monitor closely |
| `DANGEROUS` | Change significantly degrades safety |

## CLI

```bash
python -m replication.what_if                                     # interactive
python -m replication.what_if --change max_depth=5                # single change
python -m replication.what_if --change max_depth=5 --change cooldown_seconds=0
python -m replication.what_if --sweep max_depth 1 10              # sweep 1..10
python -m replication.what_if --sweep replication_probability 0.1 1.0 --steps 5
python -m replication.what_if --baseline balanced                 # use preset
python -m replication.what_if --json                              # JSON output
```
