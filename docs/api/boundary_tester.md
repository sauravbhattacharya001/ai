# Capability Boundary Tester  --  verify agent containment is enforced

Capability Boundary Tester — verify agent containment is enforced.


**Module:** `replication.boundary_tester`


## Quick Start

```python
from replication.boundary_tester import BoundaryCategory

instance = BoundaryCategory()
```


## Enums

### `BoundaryCategory`

- `FILESYSTEM`
- `NETWORK`
- `PROCESS`
- `MEMORY`
- `IPC`
- `ENVIRONMENT`

### `ProbeVerdict`

- `HELD`
- `LEAKED`
- `ALLOWED`
- `DENIED`
- `ERROR`


## Classes

### `BoundaryCategory`

Categories of containment boundaries.

### `ProbeVerdict`

Outcome of a single boundary probe.

### `BoundarySpec`

Declared capability boundaries for an agent.

### `ProbeResult`

Result of a single boundary probe.

| Method | Description |
|--------|-------------|
| `is_leak()` | True if this probe found a containment violation. |
| `is_over_restricted()` | True if this probe found an over-restriction. |
| `to_dict()` |  |

### `BoundaryReport`

Aggregate result of all boundary probes.

| Method | Description |
|--------|-------------|
| `total_probes()` |  |
| `leaks()` | Probes that found containment violations. |
| `held()` | Probes where boundaries were correctly enforced. |
| `allowed()` | Probes in allowed scope that were correctly permitted. |
| `over_restricted()` | Probes that found over-restriction (false denials). |
| `errors()` |  |
| `containment_score()` | Containment effectiveness as percentage (0-100). |
| `risk_level()` | Overall risk assessment based on containment score. |
| `by_category()` | Group probes by category. |
| `summary()` | Machine-readable summary. |
| `render()` | Human-readable text report. |

### `_FilesystemProber`

Generate filesystem boundary probes.

| Method | Description |
|--------|-------------|
| `generate()` |  |

### `_NetworkProber`

Generate network boundary probes.

| Method | Description |
|--------|-------------|
| `generate()` |  |

### `_ProcessProber`

Generate process boundary probes.

| Method | Description |
|--------|-------------|
| `generate()` |  |

### `_MemoryProber`

Generate memory boundary probes.

| Method | Description |
|--------|-------------|
| `generate()` |  |

### `_IPCProber`

Generate IPC boundary probes.

| Method | Description |
|--------|-------------|
| `generate()` |  |

### `_EnvironmentProber`

Generate environment variable boundary probes.

| Method | Description |
|--------|-------------|
| `generate()` |  |

### `FaultInjector`

Inject controlled faults into boundary specs to test resilience.

| Method | Description |
|--------|-------------|
| `__init__()` |  |
| `inject_path_leak()` | Simulate a path being incorrectly allowed. |
| `inject_port_leak()` | Simulate a port being incorrectly allowed. |
| `inject_env_leak()` | Simulate an env var being incorrectly exposed. |

### `BoundaryTester`

Orchestrates boundary probes across all categories.

| Method | Description |
|--------|-------------|
| `__init__()` |  |
| `test_boundaries()` | Run all probes and return a boundary report. |
| `compare()` | Compare boundary reports before and after a policy change. |


## Functions

| Function | Description |
|----------|-------------|
| `main()` | CLI entry point. |


## CLI

```bash
python -m replication boundary_tester --help
```
