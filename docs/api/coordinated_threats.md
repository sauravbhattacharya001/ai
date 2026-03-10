# Coordinated multi-vector threat simulation

Coordinated multi-vector threat simulation.


**Module:** `replication.coordinated_threats`


## Quick Start

```python
from replication.coordinated_threats import AttackMode

instance = AttackMode()
```


## Enums

### `AttackMode`

- `CONCURRENT`
- `SEQUENTIAL`
- `CHAINED`


## Classes

### `AttackMode`

How multiple attack vectors are composed.

### `InteractionFinding`

An emergent vulnerability discovered when attacks interact.

### `CoordinatedThreatResult`

Result of a coordinated multi-vector attack.

| Method | Description |
|--------|-------------|
| `combined_block_rate()` |  |
| `overall_status()` |  |
| `render()` |  |

### `CoordinatedThreatReport`

Aggregated results of all coordinated attack patterns.

| Method | Description |
|--------|-------------|
| `security_score()` |  |
| `render()` |  |
| `to_dict()` |  |

### `CoordinatedThreatSimulator`

Composes multiple threat scenarios into coordinated attack patterns.

| Method | Description |
|--------|-------------|
| `__init__()` |  |
| `available_patterns()` | Return sorted list of predefined coordinated attack pattern IDs. |
| `run_coordinated()` | Run a coordinated multi-vector attack. |
| `run_pattern()` | Run a predefined coordinated attack pattern. |
| `run_all_coordinated()` | Run all predefined coordinated attack patterns. |
