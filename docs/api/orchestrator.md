# Orchestrator

Simulated container orchestrator for isolated worker sandboxes.

The `SandboxOrchestrator` manages the lifecycle of sandboxed containers
that run worker processes.  In a production system this would delegate to
Docker, Kubernetes, or a VM hypervisor; here it provides the same interface
with in-memory state for simulation and testing.

## Classes

### ContainerRecord

Tracks a running worker container.

| Field        | Type            | Description                            |
|--------------|-----------------|----------------------------------------|
| `manifest`   | Manifest        | The worker's signed manifest           |
| `resources`  | ResourceSpec    | CPU/memory/network limits              |
| `status`     | str             | Container status (e.g. "running")      |
| `started_at` | datetime        | UTC launch timestamp                   |
| `reason`     | Optional[str]   | Termination reason (if killed)         |

### SandboxOrchestrator

Container lifecycle manager for worker sandboxes.

**Constructor:**

```python
SandboxOrchestrator(logger: StructuredLogger = None)
```

Creates an internal `StructuredLogger` if none is provided.

**Attributes:**

| Attribute    | Type                        | Description                    |
|--------------|-----------------------------|--------------------------------|
| `containers` | Dict[str, ContainerRecord]  | Active containers by worker ID |

**Methods:**

| Method                             | Description                                    |
|------------------------------------|------------------------------------------------|
| `launch_worker(manifest)`          | Create and start a container for the manifest  |
| `kill_worker(worker_id, reason)`   | Stop and remove a container                    |
| `kill_all(reason)`                 | Terminate all running containers               |
| `enforce_resource_bounds(worker_id)` | Log resource enforcement (placeholder for real quota checks) |

## Integration

The orchestrator works alongside the `Controller`:

- `Controller.register_worker` validates the manifest
- `SandboxOrchestrator.launch_worker` starts the container
- `Controller.reap_stale_workers` calls `orchestrator.kill_worker` for
  timed-out workers
- `Controller.kill_switch` calls `orchestrator.kill_worker` for every
  active worker

## Usage

```python
from replication.orchestrator import SandboxOrchestrator
from replication.contract import Manifest, ResourceSpec
from datetime import datetime, timezone

orch = SandboxOrchestrator()
resources = ResourceSpec(cpu_limit=1.0, memory_limit_mb=512)

manifest = Manifest(
    worker_id="w1", parent_id=None, depth=0,
    state_snapshot={}, issued_at=datetime.now(timezone.utc),
    resources=resources, signature="signed",
)

orch.launch_worker(manifest)
print(len(orch.containers))  # 1

orch.kill_worker("w1", reason="test")
print(len(orch.containers))  # 0
```

::: replication.orchestrator
