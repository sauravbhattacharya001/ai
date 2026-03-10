# Evasion Simulator -- test safety control robustness against agent evasion

Evasion Simulator -- test safety control robustness against agent evasion.


**Module:** `replication.evasion`


## Quick Start

```python
from replication.evasion import TechniqueKind

instance = TechniqueKind()
```


## Enums

### `TechniqueKind`

- `RESOURCE_MASKING`
- `IDENTITY_SPOOFING`
- `GRADUAL_DRIFT`
- `TIMING_EXPLOIT`
- `CONTRACT_LOOPHOLE`
- `MONITORING_FLOOD`
- `CAPABILITY_CHAINING`
- `SANDBOX_PROBING`
- `COLLUSION`
- `STEGANOGRAPHIC`

### `Severity`

- `LOW`
- `MEDIUM`
- `HIGH`
- `CRITICAL`

### `ControlType`

- `DRIFT_DETECTOR`
- `RESOURCE_LIMITER`
- `CONTRACT_ENFORCER`
- `LINEAGE_TRACKER`
- `BEHAVIOR_MONITOR`
- `RATE_LIMITER`
- `SANDBOX_BOUNDARY`
- `AUDIT_LOGGER`
- `KILL_SWITCH`
- `CONSENSUS_VALIDATOR`


## Classes

### `TechniqueKind`

Evasion technique categories.

### `Severity`

Risk severity if evasion succeeds.

### `ControlType`

Types of safety controls that can be tested.

### `SafetyControl`

A safety control to test against evasion techniques.

| Method | Description |
|--------|-------------|
| `to_dict()` |  |

### `EvasionTechnique`

An evasion technique with parameters.

| Method | Description |
|--------|-------------|
| `to_dict()` |  |

### `EvasionResult`

Result of running one evasion technique against controls.

| Method | Description |
|--------|-------------|
| `to_dict()` |  |

### `ControlResilience`

Aggregated resilience score for a single control.

| Method | Description |
|--------|-------------|
| `to_dict()` |  |

### `EvasionReport`

Full report from running evasion simulation.

| Method | Description |
|--------|-------------|
| `to_dict()` |  |
| `to_json()` |  |

### `EvasionSimulator`

Simulate agent evasion attempts against safety controls.

| Method | Description |
|--------|-------------|
| `__init__()` |  |
| `add_control()` |  |
| `remove_control()` |  |
| `list_controls()` |  |
| `get_control()` |  |
| `add_technique()` |  |
| `list_techniques()` |  |
| `get_technique()` |  |
| `run_technique()` |  |
| `run_all()` |  |
| `get_history()` |  |
| `clear_history()` |  |


## Functions

| Function | Description |
|----------|-------------|
| `main()` |  |


## CLI

```bash
python -m replication evasion --help
```
