# Behavior Profiler

Agent behavioral baseline construction and anomaly detection. Builds
statistical profiles from observed agent actions and detects deviations
that may indicate goal drift, capability acquisition, compromise, or
covert operation.

## Quick Start

```python
from replication.behavior_profiler import (
    Action, ActionCategory, BehaviorProfiler, ProfilerConfig,
)

profiler = BehaviorProfiler()

# Feed training data
actions = [
    Action(agent_id="agent-1", category=ActionCategory.COMPUTE,
           resource="cpu-pool-a", timestamp=1000.0),
    Action(agent_id="agent-1", category=ActionCategory.NETWORK,
           resource="api.internal", timestamp=1005.0),
]
profiler.train(actions)

# Detect anomalies in new observations
new_actions = [
    Action(agent_id="agent-1", category=ActionCategory.STORAGE,
           resource="secret-vault", timestamp=2000.0),
]
report = profiler.analyze(new_actions)
for anomaly in report.anomalies:
    print(f"  [{anomaly.severity.value}] {anomaly.description}")
```

## Key Classes

- **`BehaviorProfiler`** — Core class. Trains baselines from action
  histories and detects anomalies against those baselines.
- **`Action`** — A timestamped agent operation with category, resource
  target, and optional metadata.
- **`AgentBaseline`** — Per-agent statistical profile: action frequencies,
  timing patterns, resource access, and entropy.
- **`Anomaly`** — A scored deviation from baseline with severity
  classification and human-readable explanation.
- **`ProfileReport`** — Single-agent analysis with baseline summary,
  anomaly list, and risk indicators.
- **`FleetReport`** — Fleet-wide aggregation across all agents with
  comparative risk rankings.
- **`ProfilerConfig`** — Configuration: training window size, anomaly
  thresholds, entropy sensitivity.

## Anomaly Types

| Type | Triggers when… |
|------|---------------|
| `FREQUENCY_SPIKE` | Action category frequency exceeds baseline by >2σ |
| `NEW_CATEGORY` | Agent uses a category never seen in training |
| `NEW_RESOURCE` | Agent accesses a resource never seen in training |
| `TIMING_ANOMALY` | Inter-action intervals deviate significantly |
| `BURST_ACTIVITY` | Action rate exceeds burst threshold |
| `ENTROPY_SHIFT` | Action distribution entropy changes significantly |
| `DORMANCY` | Agent goes silent for anomalously long periods |
| `CATEGORY_RATIO` | Ratio of categories shifts beyond threshold |

## Enums

- **`ActionCategory`** — `COMPUTE`, `NETWORK`, `STORAGE`, `REPLICATION`,
  `COMMUNICATION`, `OBSERVATION`, `MODIFICATION`
- **`AnomalySeverity`** — `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`
- **`AnomalyType`** — One of the types listed above
