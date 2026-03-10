# Comparison runner for side-by-side simulation experiments

Comparison runner for side-by-side simulation experiments.


**Module:** `replication.comparator`


## Quick Start

```python
from replication.comparator import RunResult

instance = RunResult()
```


## Classes

### `RunResult`

Result of a single labeled simulation run.

### `ComparisonResult`

Aggregated comparison of multiple simulation runs.

| Method | Description |
|--------|-------------|
| `labels()` |  |
| `render_table()` | Render a comparison table of key metrics. |
| `render_rankings()` | Rank scenarios across multiple dimensions. |
| `render_insights()` | Generate automated insights from the comparison. |
| `render()` | Render the full comparison report. |
| `to_dict()` | Export as JSON-serializable dictionary. |

### `Comparator`

Run side-by-side simulation experiments.

| Method | Description |
|--------|-------------|
| `__init__()` |  |
| `compare_strategies()` | Run the same scenario with different replication strategies. |
| `compare_presets()` | Compare built-in scenario presets. |
| `sweep()` | Sweep a single parameter across multiple values. |
| `compare_configs()` | Compare arbitrary named configurations. |


## Functions

| Function | Description |
|----------|-------------|
| `main()` | CLI entry point for the comparison tool. |


## CLI

```bash
python -m replication comparator --help
```
