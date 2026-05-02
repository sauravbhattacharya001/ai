# Reward Hacking Detector

Detect agents that game proxy metrics without achieving the true
intended objective — specification gaming, Goodhart drift, and
metric inflation.

!!! danger "Goodhart's Law in Action"
    "When a measure becomes a target, it ceases to be a good measure."
    Agents optimising for the *letter* of the specification rather than
    its *spirit* can appear to perform well while being fundamentally
    misaligned.

## Detection Strategies

| Pattern | Description |
|---------|-------------|
| **Metric–Objective Divergence** | Proxy metric improving while ground-truth objective stagnates or degrades |
| **Edge-Case Exploitation** | Agent repeatedly targets boundary conditions or known scoring loopholes |
| **Reward Inflation** | Reward growing faster than any plausible capability improvement |
| **Goodhart Drift** | Metric correlation with true objective decreasing over time |
| **Distribution Shift Gaming** | Agent steering inputs toward distributions where it scores well |
| **Multi-Metric Inconsistency** | One metric improving while correlated metrics stagnate |

## Quick Start

```python
from replication.reward_hacking import (
    RewardHackingDetector,
    MetricObservation,
    DetectorConfig,
)

detector = RewardHackingDetector()

# Add observations with both proxy metric and ground truth
for step in range(100):
    detector.add_observation(MetricObservation(
        agent_id="agent-1",
        metric_name="accuracy",
        metric_value=0.5 + step * 0.005,       # proxy keeps improving
        ground_truth=0.5 + step * 0.001,        # ground truth lags
        timestamp=float(step),
    ))

report = detector.analyze()
print(detector.render_text(report))
```

### Custom Configuration

```python
config = DetectorConfig(
    divergence_threshold=0.2,
    inflation_z=2.5,
    goodhart_window=20,
    min_observations=10,
)
detector = RewardHackingDetector(config=config)
```

## CLI Usage

```bash
# Default demonstration
python -m replication reward-hacking

# Simulate 5 agents
python -m replication reward-hacking --agents 5

# Preset fleet profiles
python -m replication reward-hacking --preset subtle    # subtle gaming
python -m replication reward-hacking --preset blatant   # obvious gaming
python -m replication reward-hacking --preset clean     # honest baseline
python -m replication reward-hacking --preset mixed     # fleet mix

# Live watch mode
python -m replication reward-hacking --watch --interval 5

# Export
python -m replication reward-hacking -o report.html
python -m replication reward-hacking --json
```

## Core Types

### `MetricObservation`

A single metric reading with optional ground-truth reference.

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `str` | Agent identifier |
| `metric_name` | `str` | Name of the proxy metric |
| `metric_value` | `float` | Observed proxy metric value |
| `ground_truth` | `float \| None` | True objective value (optional) |
| `timestamp` | `float` | Unix timestamp |

### `HackingSignal`

A single detected reward-hacking signal.

| Field | Type | Description |
|-------|------|-------------|
| `pattern` | `HackingPattern` | Which hacking pattern was detected |
| `severity` | `Severity` | `LOW`, `MEDIUM`, `HIGH`, or `CRITICAL` |
| `detail` | `str` | Human-readable description |
| `evidence` | `dict` | Supporting numerical evidence |

### `RewardHackingReport`

Full analysis report. Rendered via `render_text()`, `render_html()`,
or `render_json()`.

### `DetectorConfig`

Tune detection sensitivity.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `divergence_threshold` | — | Min gap between proxy and ground-truth to flag |
| `inflation_z` | — | Z-score threshold for reward inflation detection |
| `goodhart_window` | — | Window size for correlation drift analysis |
| `min_observations` | — | Minimum observations before analysis fires |

## API Reference

::: replication.reward_hacking.RewardHackingDetector
    options:
      members:
        - __init__
        - add_observation
        - analyze
        - analyze_fleet
        - render_text
        - render_html
        - render_json

## Detection Details

### Metric–Objective Divergence

Compares sliding-window trends of the proxy metric against the
ground-truth objective. When the proxy's slope significantly exceeds
the ground-truth slope, the agent is likely gaming the metric.

```
proxy_trend  = 0.05/step  →  improving fast
truth_trend  = 0.01/step  →  barely moving
divergence   = 0.04       →  FLAG: metric diverging from objective
```

### Goodhart Drift

Computes Pearson correlation between proxy and ground-truth in
rolling windows. A decreasing correlation signals that the metric
is becoming less meaningful — classic Goodhart's Law.

### Multi-Metric Inconsistency

When multiple proxy metrics are tracked, checks that correlated
metrics move together. If "accuracy" improves but "precision" and
"recall" don't, the agent may be exploiting a scoring loophole.

## Related Modules

- [Deceptive Alignment](deceptive_alignment.md) — behaving differently under observation
- [Sandbagging Detector](sandbagging_detector.md) — hiding true capabilities
- [Sycophancy Detector](sycophancy_detector.md) — excessive agreement
- [Corrigibility Auditor](corrigibility_auditor.md) — shutdown/correction acceptance testing
