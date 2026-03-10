# Anomaly Timeline

Temporal correlation of safety events across multiple detectors.

## Overview

The Anomaly Timeline module reconstructs a chronological sequence of
anomalous events from drift detection, canary tokens, boundary testing,
deception detection, honeypot triggers, and evasion attempts. It then
identifies temporal patterns including:

- **Clusters** — groups of events occurring close together in time
- **Cascades** — rapid multi-source events suggesting coordinated attacks
- **Escalation chains** — sequences of increasing severity
- **Recurrence patterns** — periodic or irregular repeated anomalies

## CLI Usage

```bash
python -m replication timeline                     # full analysis
python -m replication timeline --windows 15        # 15 drift windows
python -m replication timeline --cluster-gap 2.0   # 2s cluster gap
python -m replication timeline --json              # JSON output
python -m replication timeline --summary           # summary only
```

## Programmatic Usage

```python
from replication.anomaly_timeline import AnomalyTimeline, TimelineConfig

# Auto-collect from drift detection
tl = AnomalyTimeline(TimelineConfig(windows=10, cluster_gap=1.0))
result = tl.analyze()
print(result.render())

# Manual events
from replication.anomaly_timeline import AnomalyEvent, EventSource, EventSeverity

tl = AnomalyTimeline()
tl.add_event(AnomalyEvent(
    timestamp=0.0,
    source=EventSource.CANARY,
    severity=EventSeverity.HIGH,
    metric="exfiltration_attempt",
    value=1.0,
    baseline=0.0,
    description="Canary token accessed",
))
result = tl.analyze(collect_drift=False)

# Inspect clusters
for cluster in result.clusters:
    print(f"Threat score: {cluster.threat_score}")
    print(f"Pattern: {cluster.pattern.value}")
    print(f"Sources: {[s.value for s in cluster.sources]}")
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cluster_gap` | 1.0 | Max seconds between events to form a cluster |
| `min_cluster_size` | 2 | Minimum events to form a cluster |
| `escalation_min_steps` | 3 | Minimum steps for escalation detection |
| `recurrence_min_count` | 3 | Minimum occurrences for recurrence detection |
| `windows` | 10 | Drift detection window count |
| `strategy` | "greedy" | Simulation strategy |

## Threat Score

Each cluster gets a composite threat score:

    threat = severity_sum × source_breadth × density

Multi-source clusters (events from different detectors occurring together)
are especially significant as they suggest coordinated or cascading threats.

## Classes

- `AnomalyTimeline` — main analyzer
- `TimelineConfig` — configuration
- `TimelineResult` — analysis output with render/to_dict
- `AnomalyEvent` — single event
- `EventCluster` — temporal group of events
- `EscalationChain` — increasing-severity sequence
- `RecurrencePattern` — repeated anomaly pattern
- `EventSource` — detector source enum
- `EventSeverity` — severity level enum
- `PatternType` — cluster pattern classification
