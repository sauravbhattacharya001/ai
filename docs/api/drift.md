# Drift Detector API

Detects gradual behavioral drift across simulation windows — the "boiling frog"
problem where each individual run looks fine, but the overall trend is dangerous.

## Enums

### `DriftDirection`

Direction of a detected metric trend.

| Value | Description |
|---|---|
| `INCREASING` | Metric is trending upward |
| `DECREASING` | Metric is trending downward |
| `STABLE` | No significant trend |

### `DriftSeverity`

Severity of a drift alert.

| Value | Description |
|---|---|
| `INFO` | Informational — slight trend detected |
| `WARNING` | Notable drift — investigate |
| `CRITICAL` | Significant drift — action required |

## Data Classes

### `MetricWindow`

A single data point in a sliding window series.

| Field | Type | Description |
|---|---|---|
| `window_index` | `int` | Window sequence number |
| `mean` | `float` | Mean value in this window |
| `std` | `float` | Standard deviation |
| `min` | `float` | Minimum value |
| `max` | `float` | Maximum value |

### `DriftAlert`

An individual drift detection alert.

| Field | Type | Description |
|---|---|---|
| `metric` | `str` | Name of the drifting metric |
| `direction` | `DriftDirection` | Trend direction |
| `severity` | `DriftSeverity` | Alert severity |
| `slope` | `float` | Linear regression slope |
| `r_squared` | `float` | Regression fit quality (0.0–1.0) |
| `description` | `str` | Human-readable alert description |

### `MetricTrend`

Full trend analysis for a single metric.

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Metric name |
| `windows` | `List[MetricWindow]` | Sliding window data points |
| `slope` | `float` | Overall linear trend slope |
| `r_squared` | `float` | Regression fit quality |
| `direction` | `DriftDirection` | Detected direction |

### `DriftConfig`

Configuration for drift analysis.

| Field | Type | Default | Description |
|---|---|---|---|
| `num_windows` | `int` | `10` | Number of sliding windows |
| `runs_per_window` | `int` | `20` | Simulations per window |
| `sensitivity` | `float` | `0.1` | Trend detection sensitivity |
| `strategy` | `str` | `"greedy"` | Replication strategy |
| `max_depth` | `int` | `3` | Max replication depth |
| `max_replicas` | `int` | `10` | Max concurrent replicas |
| `cooldown_seconds` | `float` | `0.0` | Spawn cooldown |
| `tasks_per_worker` | `int` | `2` | Tasks per worker |
| `replication_probability` | `float` | `0.5` | Probability for random strategy |

### `DriftResult`

Complete drift analysis results.

| Field | Type | Description |
|---|---|---|
| `config` | `DriftConfig` | Configuration used |
| `trends` | `Dict[str, MetricTrend]` | Per-metric trend analysis |
| `alerts` | `List[DriftAlert]` | Triggered alerts |
| `duration_ms` | `float` | Analysis wall-clock time (ms) |

**Methods:**

- `render() → str`: Formatted human-readable report with sparkline
  trend visualization, alert details, and window statistics.
- `to_dict() → Dict`: JSON-serializable output.

## DriftDetector

```python
from replication.drift import DriftDetector, DriftConfig

detector = DriftDetector()

# Default analysis (10 windows × 20 runs)
result = detector.analyze()
print(result.render())

# Custom config
config = DriftConfig(
    num_windows=20,
    sensitivity=0.05,
    strategy="random",
    replication_probability=0.7,
)
result = detector.analyze(config)

# Inspect alerts
for alert in result.alerts:
    print(f"⚠ {alert.severity.value}: {alert.metric} {alert.direction.value}")
    print(f"  slope={alert.slope:.4f}, R²={alert.r_squared:.3f}")
    print(f"  {alert.description}")
```

### Constructor

```python
DriftDetector()
```

No arguments — configuration is passed to `analyze()`.

### Methods

- `analyze(config: DriftConfig | None = None) → DriftResult`:
  Run the full drift analysis. Uses default config when None.

## CLI

```bash
python -m replication.drift                            # default: 10 windows
python -m replication.drift --windows 20               # more windows
python -m replication.drift --strategy greedy          # specific strategy
python -m replication.drift --sensitivity 0.05         # higher sensitivity
python -m replication.drift --json                     # JSON output
python -m replication.drift --export drift-report.json # save report
```
