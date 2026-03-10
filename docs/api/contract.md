# Contract

Replication policy definitions, manifests, and stop conditions.

The `contract` module defines the data structures and rules that govern
how workers are allowed to replicate.  Every component in the system
references these types; the contract is the single source of truth for
replication constraints.

## Data Classes

### NetworkPolicy

Controls network egress for a worker sandbox.

| Field              | Type   | Default | Description                          |
|--------------------|--------|---------|--------------------------------------|
| `allow_controller` | bool   | True    | Allow communication with controller  |
| `allow_external`   | bool   | False   | Allow outbound network requests      |

### ResourceSpec

CPU and memory limits for a worker container.

| Field               | Type          | Description                                |
|---------------------|---------------|--------------------------------------------|
| `cpu_limit`         | float (> 0)   | CPU core allocation                        |
| `memory_limit_mb`   | int (> 0)     | RAM limit in megabytes                     |
| `network_policy`    | NetworkPolicy | Network egress controls (default: no ext.) |

Raises `ValueError` if limits are non-positive.

### Manifest

Immutable snapshot of a worker's identity and configuration, carried
when a replica is spawned.

| Field              | Type               | Description                            |
|--------------------|--------------------|----------------------------------------|
| `worker_id`        | str                | Unique identifier (hex token)          |
| `parent_id`        | Optional[str]      | Parent worker ID (None for root)       |
| `depth`            | int                | Replication depth (0 = root)           |
| `state_snapshot`   | Dict[str, Any]     | Serializable state cloned into replica |
| `issued_at`        | datetime           | UTC timestamp of manifest creation     |
| `resources`        | ResourceSpec       | CPU/memory/network constraints         |
| `signature`        | str                | HMAC-SHA256 signature (set by signer)  |

### ReplicationContext

Passed to stop condition predicates for evaluation.

| Field           | Type                | Description               |
|-----------------|---------------------|---------------------------|
| `manifest`      | Manifest            | The candidate manifest    |
| `active_count`  | int                 | Current live worker count |
| `contract`      | ReplicationContract | The governing contract    |

### StopCondition

Named predicate that can halt replication when triggered.

| Field         | Type      | Description                              |
|---------------|-----------|------------------------------------------|
| `name`        | str       | Human-readable identifier                |
| `description` | str       | Explanation of the condition             |
| `predicate`   | Predicate | Callable(ReplicationContext) → bool      |

**Method:** `is_triggered(context)` — returns True if the condition fires.

### ReplicationContract

Top-level policy governing replication behaviour.

| Field                 | Type                  | Description                             |
|-----------------------|-----------------------|-----------------------------------------|
| `max_depth`           | int (≥ 0)             | Maximum replication depth               |
| `max_replicas`        | int (≥ 1)             | Maximum concurrent active workers       |
| `cooldown_seconds`    | float (≥ 0)           | Minimum seconds between spawns          |
| `stop_conditions`     | List[StopCondition]   | Custom halt predicates                  |
| `expiration_seconds`  | Optional[float] (> 0) | Worker lifetime limit (None = no limit) |

**Method:** `evaluate(context)` — returns the first triggered `StopCondition`, or `None`.

## Usage

```python
from replication.contract import (
    ReplicationContract, StopCondition, ReplicationContext,
    ResourceSpec, NetworkPolicy,
)

# Define resources
resources = ResourceSpec(cpu_limit=0.5, memory_limit_mb=256)

# Create a contract with a custom stop condition
contract = ReplicationContract(
    max_depth=3,
    max_replicas=10,
    cooldown_seconds=5.0,
    stop_conditions=[
        StopCondition(
            name="max_active_5",
            description="Stop when 5 workers are active",
            predicate=lambda ctx: ctx.active_count >= 5,
        ),
    ],
)
```

::: replication.contract
