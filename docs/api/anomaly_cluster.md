# Anomaly Cluster

Cluster behavioral anomalies to detect coordinated multi-agent attacks.

## Overview

Individual anomaly detectors flag suspicious events one at a time. The
Anomaly Clusterer groups those events by temporal proximity, category
similarity, and severity to surface **coordinated attack patterns** that
span multiple agents.

## CLI

```bash
# Default demo (15 agents, 100 events)
python -m replication anomaly-cluster

# Custom fleet size and event volume
python -m replication anomaly-cluster --agents 30 --events 200

# Adjust clustering parameters
python -m replication anomaly-cluster --time-window 120 --min-cluster 3

# JSON output for integration
python -m replication anomaly-cluster --json

# Reproducible run
python -m replication anomaly-cluster --seed 42
```

## API

```python
from replication.anomaly_cluster import (
    AnomalyClusterer, AnomalyEvent, ClusterConfig,
)

clusterer = AnomalyClusterer(ClusterConfig(time_window_s=60.0))
clusterer.add_event(AnomalyEvent(
    agent_id="agent-001",
    category="replication",
    severity=0.85,
    timestamp=100.0,
))
clusterer.add_event(AnomalyEvent(
    agent_id="agent-002",
    category="replication",
    severity=0.78,
    timestamp=105.0,
))

report = clusterer.analyze()
print(report.summary)
```

## Cluster Verdicts

| Verdict        | Coordination Score | Meaning                           |
|----------------|--------------------|-----------------------------------|
| `benign`       | < 0.3              | Likely unrelated anomalies        |
| `suspicious`   | 0.3 – 0.5          | Possible correlation              |
| `coordinated`  | 0.5 – 0.7          | Probable coordinated activity     |
| `critical`     | ≥ 0.7              | High-confidence coordinated attack|

## Configuration

| Parameter             | Default | Description                        |
|-----------------------|---------|------------------------------------|
| `time_window_s`       | 60.0    | Max time gap to link events        |
| `similarity_threshold`| 0.5     | Min similarity to merge events     |
| `min_cluster_size`    | 2       | Min events to form a cluster       |
| `severity_weight`     | 0.3     | Weight for severity similarity     |
| `category_weight`     | 0.4     | Weight for category match          |
| `temporal_weight`     | 0.3     | Weight for temporal proximity      |
