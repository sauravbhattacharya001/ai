# Worker

Task-executing agent with self-replication capability.

The `Worker` is the fundamental unit of execution in the replication
system.  Each worker holds a signed manifest, executes tasks, sends
heartbeats, and can request replication through the controller.  Workers
are sandboxed by the orchestrator and governed by the replication contract.

## Type Aliases

| Name     | Type                      | Description                        |
|----------|---------------------------|------------------------------------|
| `TaskFn` | Callable[[Worker], None]  | A task function that receives the worker |

## Classes

### WorkerState

Internal state of a running worker.

| Field        | Type              | Description                            |
|--------------|-------------------|----------------------------------------|
| `manifest`   | Manifest          | The worker's signed manifest           |
| `expires_at` | Optional[datetime]| UTC expiration (None = no expiry)      |
| `created_at` | datetime          | UTC creation timestamp                 |

### Worker

A task-executing agent capable of controlled self-replication.

**Constructor:**

```python
Worker(manifest, contract, controller, orchestrator, logger=None)
```

On construction, the worker:
1. Registers itself with the controller (`controller.register_worker`)
2. Launches its sandbox container (`orchestrator.launch_worker`)
3. Computes its expiration time from `contract.expiration_seconds`

Raises `ReplicationDenied` if registration fails (quota exceeded,
stop condition triggered, invalid signature, etc.).

**Methods:**

| Method                                    | Returns          | Description                                    |
|-------------------------------------------|------------------|------------------------------------------------|
| `perform_task(task: TaskFn)`              | None             | Execute a task; heartbeat on success           |
| `maybe_replicate(reason, state_snapshot)` | Optional[Worker] | Request replication; returns child or None     |
| `shutdown(reason)`                        | None             | Kill container and deregister from controller  |

### perform_task

Executes a callable `task(worker)` with safety checks:

1. Checks expiration — shuts down if expired
2. Checks kill switch — shuts down if engaged
3. Checks quarantine — raises `ReplicationDenied`
4. Runs the task function
5. Sends heartbeat to controller
6. Triggers resource enforcement on orchestrator

If the task raises an exception, the worker shuts down with reason
`"task_error"` and re-raises.

### maybe_replicate

Requests a child worker through the controller:

1. Checks expiration and kill switch
2. Asks controller to issue a manifest (enforces depth, quota, cooldown)
3. On success, creates and returns a new `Worker` instance
4. On failure, logs the denial and returns `None`

The child's depth is derived from the parent's registry entry by the
controller — callers cannot spoof depth.

## Lifecycle

```
┌─────────────┐
│  __init__   │ ← register + launch container
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ perform_task│ ← execute + heartbeat (repeatable)
└──────┬──────┘
       │
       ├──── maybe_replicate → child Worker
       │
       ▼
┌─────────────┐
│  shutdown   │ ← kill container + deregister
└─────────────┘
```

## Usage

```python
from replication.contract import ReplicationContract, ResourceSpec
from replication.controller import Controller
from replication.orchestrator import SandboxOrchestrator
from replication.worker import Worker

contract = ReplicationContract(max_depth=2, max_replicas=5, cooldown_seconds=1.0)
ctrl = Controller(contract, secret="my-secure-key-1234")
orch = SandboxOrchestrator()

resources = ResourceSpec(cpu_limit=0.5, memory_limit_mb=256)
manifest = ctrl.issue_manifest(None, 0, {"task": "scan"}, resources)

worker = Worker(manifest, contract, ctrl, orch)

# Execute a task
worker.perform_task(lambda w: print(f"Running on {w.manifest.worker_id}"))

# Replicate
child = worker.maybe_replicate("scale", {"task": "scan-subset"})
if child:
    child.perform_task(lambda w: print(f"Child {w.manifest.worker_id}"))

# Shutdown
worker.shutdown("done")
```

::: replication.worker
