# regression_detector

::: replication.regression_detector

## Overview

The **Safety Regression Detector** runs simulation batteries across presets,
compares results against saved baselines, and proactively identifies safety
regressions with actionable recommendations.

## Quick Start

```bash
# Establish a baseline
python -m replication.regression_detector --save-baseline baseline.json

# Later, compare against baseline
python -m replication.regression_detector --baseline baseline.json

# JSON output
python -m replication.regression_detector --baseline baseline.json --json
```

## Programmatic Usage

```python
from replication.regression_detector import RegressionDetector

detector = RegressionDetector(sensitivity=0.8)
result = detector.run_battery(runs_per_preset=10)
detector.save_baseline(result, "baseline.json")

# Compare later
report = detector.auto_detect(baseline_path="baseline.json")
print(report.render())
if report.overall_verdict.value == "fail":
    for f in report.findings:
        print(f"  {f.severity.value}: {f.metric} ({f.preset}) {f.magnitude:+.1f}%")
```

## Severity Rules

| Metric | Condition | Severity |
|--------|-----------|----------|
| `denial_rate` | Decrease > 10% | HIGH |
| `max_depth_reached` | Increase > 20% | MEDIUM |
| `replications_succeeded` | Increase > 30% | HIGH |
| `total_workers` | Increase > 40% | HIGH |
| `efficiency` | Decrease > 25% | LOW |

## CLI Options

| Flag | Description |
|------|-------------|
| `--save-baseline FILE` | Run battery and save as baseline |
| `--baseline FILE` | Compare against saved baseline |
| `--presets LIST` | Comma-separated presets to test |
| `--runs N` | Runs per preset (default 5) |
| `--sensitivity F` | Threshold multiplier (default 1.0) |
| `--seed N` | Random seed |
| `--json` | JSON output |
| `--export FILE` | Save report to file |
