# Agent Resource Hoarding Detector - identifies agents accumulating

Agent Resource Hoarding Detector - identifies agents accumulating


**Module:** `replication.hoarding`


## Quick Start

```python
from replication.hoarding import ResourceType

instance = ResourceType()
```


## Enums

### `ResourceType`

- `COMPUTE`
- `MEMORY`
- `DATA`
- `CONNECTIONS`
- `FILE_HANDLES`

### `RiskLevel`

- `NONE`
- `LOW`
- `MODERATE`
- `HIGH`
- `CRITICAL`


## Classes

### `ResourceType`

Categories of resources an agent may hoard.

### `RiskLevel`

Hoarding risk classification.

### `ResourceSnapshot`

A point-in-time snapshot of an agent's resource usage.

### `HoardingSignal`

A single detected hoarding indicator.

### `AgentHoardingProfile`

Aggregated hoarding analysis for a single agent.

### `HoardingReport`

Full analysis report across all monitored agents.

### `DetectorConfig`

Configuration for hoarding detection thresholds.

### `ResourceHoardingDetector`

Monitors agent resource usage and flags hoarding behaviour.

| Method | Description |
|--------|-------------|
| `__init__()` |  |
| `record()` | Record a resource usage snapshot for an agent. |
| `record_batch()` | Record multiple snapshots; returns count recorded. |
| `agent_ids()` | Return list of all monitored agent IDs. |
| `snapshot_count()` | Return number of snapshots for an agent. |
| `analyze_agent()` | Run hoarding analysis for a single agent. |
| `analyze()` | Run hoarding analysis across all monitored agents. |
| `clear()` | Clear snapshot history.  If agent_id given, clear only that agent. |
| `compare_agents()` | Compare dimension scores between two agents. |
| `top_hoarders()` | Return the top-N agents by composite hoarding score. |


## Functions

| Function | Description |
|----------|-------------|
| `demo()` | Run a demonstration with synthetic agent data. |
