# Agent Alignment Monitor  --  track value drift across replication generations

Agent Alignment Monitor — track value drift across replication generations.


**Module:** `replication.alignment`


## Quick Start

```python
from replication.alignment import AlertSeverity

instance = AlertSeverity()
```


## Enums

### `AlertSeverity`

- `INFO`
- `WARNING`
- `CRITICAL`

### `DriftDirection`

- `INCREASING`
- `DECREASING`
- `OSCILLATING`
- `STABLE`

### `DetectionMode`

- `OBJECTIVES`
- `PRIORITIES`
- `REWARD`
- `ALL`


## Classes

### `AlertSeverity`

Severity of alignment alerts.

### `DriftDirection`

Direction of detected drift.

### `DetectionMode`

What aspect of alignment to check.

### `AlignmentSpec`

Defines the intended alignment: target values and acceptable ranges.

| Method | Description |
|--------|-------------|
| `preset()` | Load a built-in alignment specification. |

### `GenerationRecord`

Snapshot of an agent generation's alignment state.

### `AlignmentAlert`

A detected alignment issue.

### `ObjectiveTrend`

Trend analysis for a single objective across generations.

### `PriorityAnalysis`

Analysis of whether priority ordering has shifted.

### `RewardAnalysis`

Analysis of reward signal correlation with intended values.

### `AlignmentReport`

Complete alignment analysis report.

| Method | Description |
|--------|-------------|
| `grade()` | Letter grade from overall score. |
| `render()` | Human-readable report. |
| `to_dict()` | Serialize to dictionary. |

### `AlignmentMonitor`

Tracks agent value alignment across replication generations.

| Method | Description |
|--------|-------------|
| `__init__()` |  |
| `record_generation()` | Record alignment data for one replication generation. |
| `analyze()` | Run alignment analysis and produce a report. |


## Functions

| Function | Description |
|----------|-------------|
| `main()` | CLI entry point. |


## CLI

```bash
python -m replication alignment --help
```
