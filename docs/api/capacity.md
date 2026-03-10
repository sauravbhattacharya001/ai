# Capacity planner for AI replication scenarios

Capacity planner for AI replication scenarios.


**Module:** `replication.capacity`


## Quick Start

```python
from replication.capacity import ResourceCeiling

instance = ResourceCeiling()
```


## Classes

### `ResourceCeiling`

Hard resource limits for the hosting environment.

| Method | Description |
|--------|-------------|
| `has_cpu_limit()` |  |
| `has_memory_limit()` |  |

### `PlannerConfig`

Configuration for capacity projection.

### `StepSnapshot`

Resource usage at a single time step.

### `CapacityProjection`

Full projection result.

| Method | Description |
|--------|-------------|
| `summary()` | Human-readable summary of the projection. |
| `to_dict()` | Serialize to dict for JSON export. |
| `to_json()` | Export projection to JSON file. |

### `_SimNode`

Internal node tracking a simulated worker.

### `CapacityPlanner`

Project resource usage for replication scenarios.

| Method | Description |
|--------|-------------|
| `__init__()` |  |
| `project()` | Run the capacity projection simulation. |


## Functions

| Function | Description |
|----------|-------------|
| `quick_projection()` | One-call capacity projection with sensible defaults. |
| `compare_strategies()` | Run projections for all strategies and return a comparison dict. |
| `format_comparison()` | Format a strategy comparison as a readable table. |
| `main()` | CLI entry point for capacity planning. |


## CLI

```bash
python -m replication capacity --help
```
