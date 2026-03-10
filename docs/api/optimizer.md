# Contract Optimizer  --  find optimal contract parameters under safety constraints

Contract Optimizer — find optimal contract parameters under safety constraints.


**Module:** `replication.optimizer`


## Quick Start

```python
from replication.optimizer import Objective

instance = Objective()
```


## Enums

### `Objective`

- `THROUGHPUT`
- `EFFICIENCY`
- `SAFETY`
- `BALANCED`


## Classes

### `Objective`

Optimization objective.

### `CandidateResult`

Result of evaluating a single parameter configuration.

| Method | Description |
|--------|-------------|
| `to_dict()` |  |

### `OptimizerConfig`

Configuration for the contract optimizer.

### `OptimizerResult`

Result of the optimization run.

| Method | Description |
|--------|-------------|
| `render()` |  |
| `to_dict()` |  |

### `ContractOptimizer`

Searches for optimal contract parameters under safety constraints.

| Method | Description |
|--------|-------------|
| `__init__()` |  |
| `optimize()` | Run the grid search and return ranked results. |


## Functions

| Function | Description |
|----------|-------------|
| `main()` | CLI entry point for the contract optimizer. |


## CLI

```bash
python -m replication optimizer --help
```
