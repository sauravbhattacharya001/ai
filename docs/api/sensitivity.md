# Parameter Sensitivity Analyzer for replication safety experiments

Parameter Sensitivity Analyzer for replication safety experiments.


**Module:** `replication.sensitivity`


## Quick Start

```python
from replication.sensitivity import ParameterDef

instance = ParameterDef()
```


## Classes

### `ParameterDef`

Defines a sweepable parameter with its range and type.

### `SweepPoint`

Aggregated metrics at a single parameter value.

### `TippingPoint`

A detected tipping point where a metric changes sharply.

### `SensitivityCurve`

Results of sweeping a single parameter.

| Method | Description |
|--------|-------------|
| `render()` | Render the sensitivity curve as a text table. |
| `to_dict()` |  |

### `SensitivityResult`

Complete sensitivity analysis across all parameters.

| Method | Description |
|--------|-------------|
| `render()` |  |
| `to_dict()` |  |

### `SensitivityConfig`

Configuration for a sensitivity analysis run.

### `SensitivityAnalyzer`

Performs one-at-a-time parameter sensitivity analysis.

| Method | Description |
|--------|-------------|
| `__init__()` |  |
| `sweep_parameter()` | Sweep a single parameter across its value range. |
| `analyze()` | Run full sensitivity analysis across all (or selected) parameters. |


## Functions

| Function | Description |
|----------|-------------|
| `main()` | CLI entry point for parameter sensitivity analysis. |


## CLI

```bash
python -m replication sensitivity --help
```
