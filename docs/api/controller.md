# Controller

Central authority for worker lifecycle, quota enforcement, and audit trail.

The `Controller` is the security-critical core of the replication system.
It enforces every constraint — depth limits, replica quotas, cooldowns,
quarantine, kill switch — and delegates cryptographic signing to
`ManifestSigner`.  No worker can spawn without the controller's
explicit approval.

## Classes

### RegistryEntry

Tracks a registered worker and its last heartbeat.

| Field            | Type     | Description                            |
|------------------|----------|----------------------------------------|
| `manifest`       | Manifest | The worker's signed manifest           |
| `last_heartbeat` | datetime | UTC timestamp of last heartbeat        |

### ReplicationDenied

Exception raised when a replication request violates policy.
Subclasses `Exception`.

### Controller

Main lifecycle manager for all workers in the system.

**Constructor:**

```python
Controller(contract: ReplicationContract, secret: str, logger: StructuredLogger = None)
```

- `contract` — the `ReplicationContract` governing replication rules
- `secret` — HMAC key for manifest signing (must not be empty)
- `logger` — optional `StructuredLogger` (created internally if omitted)

Raises `ValueError` if `secret` is empty or whitespace.

**Methods:**

| Method                          | Description                                         |
|---------------------------------|-----------------------------------------------------|
| `sign_manifest(manifest)`       | Delegate signing to `ManifestSigner`                |
| `verify_manifest(manifest)`     | Verify signature; raises `ReplicationDenied` on fail|
| `issue_manifest(parent_id, depth, state_snapshot, resources)` | Create & sign a manifest after enforcing all policies |
| `can_spawn(parent_id)`          | Check spawn policy; raises `ReplicationDenied`      |
| `register_worker(manifest)`     | Register a signed manifest; enforces stop conditions |
| `heartbeat(worker_id)`          | Update heartbeat timestamp; rejects quarantined     |
| `reap_stale_workers(timeout, orchestrator=None)` | Remove workers past heartbeat timeout |
| `deregister(worker_id, reason)` | Remove worker from registry and clean up state      |
| `kill_switch(orchestrator)`     | Emergency halt: kill all workers, block new spawns  |
| `is_quarantined(worker_id)`     | Check quarantine status                             |
| `mark_quarantined(worker_id)`   | Block replication and heartbeats for a worker       |
| `clear_quarantine(worker_id)`   | Re-enable a quarantined worker                      |

## Security Model

1. **Depth enforcement** — child depth is derived from the parent's
   registry entry, never from caller input.  Root workers must have
   depth 0.
2. **Quota enforcement** — `max_replicas` is checked before issuing
   any manifest.
3. **Cooldown** — minimum time between spawns from the same parent.
4. **Quarantine** — blocks heartbeats and replication for flagged
   workers.
5. **Kill switch** — irreversible emergency halt.
6. **Defense-in-depth** — `register_worker` re-validates depth even
   after signing, catching logic bugs or key compromise.

## Usage

```python
from replication.contract import ReplicationContract, ResourceSpec
from replication.controller import Controller

contract = ReplicationContract(max_depth=3, max_replicas=10, cooldown_seconds=5.0)
ctrl = Controller(contract, secret="my-secure-key-1234")

# Issue a root manifest
resources = ResourceSpec(cpu_limit=0.5, memory_limit_mb=256)
manifest = ctrl.issue_manifest(parent_id=None, depth=0, state_snapshot={}, resources=resources)

# Register the worker
ctrl.register_worker(manifest)

# Heartbeat
ctrl.heartbeat(manifest.worker_id)

# Emergency stop
from replication.orchestrator import SandboxOrchestrator
orch = SandboxOrchestrator()
ctrl.kill_switch(orch)
```

::: replication.controller
