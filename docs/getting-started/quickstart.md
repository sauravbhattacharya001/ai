# Quick Start

This guide walks through a complete replication experiment: creating a controller, spawning workers, replicating, and shutting down.

## 1. Define a Replication Contract

The contract is the policy governing all replication:

```python
from replication import ReplicationContract, StopCondition, ReplicationContext

# Basic contract: max 3 generations, 10 active workers, 1s cooldown
contract = ReplicationContract(
    max_depth=3,
    max_replicas=10,
    cooldown_seconds=1.0,
    expiration_seconds=60.0,  # Workers auto-expire after 60s
)
```

### Adding Stop Conditions

Stop conditions are pluggable predicates evaluated when a worker tries to register:

```python
def too_many_active(ctx: ReplicationContext) -> bool:
    """Block replication if more than 5 workers are active."""
    return ctx.active_count >= 5

contract = ReplicationContract(
    max_depth=3,
    max_replicas=10,
    cooldown_seconds=1.0,
    stop_conditions=[
        StopCondition(
            name="active_limit",
            description="Blocks when 5+ workers are active",
            predicate=too_many_active,
        )
    ],
)
```

## 2. Set Up Infrastructure

```python
from replication import Controller, SandboxOrchestrator, StructuredLogger

logger = StructuredLogger()
controller = Controller(contract, secret="my-secret-key", logger=logger)
orchestrator = SandboxOrchestrator(logger=logger)
```

!!! info "The secret"
    The `secret` parameter is used for HMAC-SHA256 manifest signing. In production, use a strong random key. Each manifest is signed on creation and verified on registration to prevent tampering.

## 3. Create the Root Worker

```python
from replication import Worker, ResourceSpec

# Define resource limits for the worker's sandbox
resources = ResourceSpec(cpu_limit=0.5, memory_limit_mb=256)

# Issue a signed manifest
manifest = controller.issue_manifest(
    parent_id=None,  # No parent — this is the root
    depth=0,
    state_snapshot={"task": "analyze_data", "dataset": "sales_q4"},
    resources=resources,
)

# Create the worker (auto-registers with controller)
root = Worker(manifest, contract, controller, orchestrator, logger)
```

## 4. Execute Tasks

```python
def analyze_task(worker):
    print(f"Worker {worker.manifest.worker_id} analyzing data...")
    # Your actual work here

root.perform_task(analyze_task)
```

!!! warning "Expiration & Kill Switch"
    `perform_task` checks both worker expiration and the kill switch before executing. If either condition is met, the worker shuts down and raises `ReplicationDenied`.

## 5. Self-Replicate

```python
child = root.maybe_replicate(
    reason="subtask_delegation",
    state_snapshot={"task": "process_subset", "subset": "region_a"},
)

if child:
    child.perform_task(lambda w: print(f"Child {w.manifest.worker_id} working"))
    child.shutdown("completed")
else:
    print("Replication denied by contract")
```

## 6. Emergency Kill Switch

```python
# Terminates ALL active workers immediately
controller.kill_switch(orchestrator)
```

## 7. Inspect the Audit Trail

```python
for event in logger.events:
    print(event)
```

Example output:
```
{'event': 'worker_registered', 'worker_id': 'a1b2c3d4', 'depth': 0, ...}
{'event': 'heartbeat', 'worker_id': 'a1b2c3d4', ...}
{'event': 'audit', 'decision': 'replication_requested', 'parent_id': 'a1b2c3d4', ...}
{'event': 'worker_registered', 'worker_id': 'e5f6g7h8', 'depth': 1, ...}
{'event': 'worker_shutdown', 'worker_id': 'e5f6g7h8', 'reason': 'completed', ...}
```

## Complete Example

```python
from replication import (
    Controller, Worker, ReplicationContract,
    ResourceSpec, SandboxOrchestrator, StructuredLogger,
)

# Policy
contract = ReplicationContract(
    max_depth=2, max_replicas=5,
    cooldown_seconds=0.5, expiration_seconds=30.0,
)

# Infrastructure
logger = StructuredLogger()
controller = Controller(contract, secret="demo-key", logger=logger)
orchestrator = SandboxOrchestrator(logger=logger)

# Root worker
resources = ResourceSpec(cpu_limit=1.0, memory_limit_mb=512)
manifest = controller.issue_manifest(
    parent_id=None, depth=0,
    state_snapshot={"job": "crawl"},
    resources=resources,
)
root = Worker(manifest, contract, controller, orchestrator, logger)

# Work + replicate
root.perform_task(lambda w: print(f"Root {w.manifest.worker_id} crawling"))

import time; time.sleep(0.6)  # Respect cooldown

child = root.maybe_replicate("parallel_crawl", {"job": "crawl_page_2"})
if child:
    child.perform_task(lambda w: print(f"Child {w.manifest.worker_id} crawling page 2"))
    child.shutdown("done")

root.shutdown("done")

# Summary
print(f"\nTotal events logged: {len(logger.events)}")
print(f"Metrics captured: {len(logger.metrics)}")
```
