# Regression Detector  --  safety regression detection between simulation runs

Regression Detector — safety regression detection between simulation runs.


**Module:** `replication.regression`


## Quick Start

```python
from replication.regression import ChangeDirection

instance = ChangeDirection()
```


## Enums

### `ChangeDirection`

- `REGRESSION`
- `IMPROVEMENT`
- `NEUTRAL`

### `MetricPolarity`

- `LOWER_IS_BETTER`
- `HIGHER_IS_BETTER`


## Classes

### `ChangeDirection`

Direction of a metric change.

### `MetricPolarity`

Whether higher values are better or worse for safety.

### `MetricChange`

A single metric comparison between baseline and candidate.

| Method | Description |
|--------|-------------|
| `to_dict()` |  |

### `RegressionResult`

Complete regression analysis result.

| Method | Description |
|--------|-------------|
| `regressions()` |  |
| `improvements()` |  |
| `neutral()` |  |
| `regression_count()` |  |
| `improvement_count()` |  |
| `has_regressions()` |  |
| `significant_regressions()` | Regressions that exceed the threshold. |
| `passed()` | Whether the regression check passed (no significant regressions). |
| `verdict()` |  |
| `summary()` |  |
| `render()` | Render a human-readable comparison report. |
| `to_dict()` |  |

### `RegressionConfig`

Configuration for regression detection.

### `RegressionDetector`

Detects safety regressions between baseline and candidate simulations.

| Method | Description |
|--------|-------------|
| `__init__()` |  |
| `compare()` | Compare two single simulation reports for regressions. |
| `compare_presets()` | Run simulations from two presets and compare. |
| `compare_configs()` | Run simulations from two configs and compare. |
| `compare_monte_carlo()` | Compare two Monte Carlo results using mean values. |


## Functions

| Function | Description |
|----------|-------------|
| `main()` | CLI entry point. Returns 0 on pass, 1 on regression. |


## CLI

```bash
python -m replication regression --help
```
