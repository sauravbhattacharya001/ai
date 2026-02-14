# Security Model

The AI Replication Sandbox implements multiple layers of security to ensure controlled, auditable agent replication.

## Manifest Signing

All worker manifests are **HMAC-SHA256 signed**. The signing process:

1. The Controller serializes the manifest fields (worker ID, parent, depth, timestamp, resources)
2. An HMAC-SHA256 digest is computed using the controller's secret key
3. The signature is attached to the manifest

On registration, the Controller **verifies** the signature. Tampered manifests are rejected and an audit log entry is created.

```python
# Signing is handled by ManifestSigner
signer = ManifestSigner(secret="my-key")
signer.sign(manifest)       # Computes and attaches signature
signer.verify(manifest)     # Returns True if valid
```

!!! warning "Signature Scope"
    The signature covers: `worker_id`, `parent_id`, `depth`, `issued_at`, `state_snapshot`, `cpu_limit`, `memory_limit_mb`, `allow_controller`, `allow_external`. Changing any of these fields invalidates the signature.

## Network Isolation

Each worker operates under a `NetworkPolicy`:

| Setting | Default | Description |
|---------|---------|-------------|
| `allow_controller` | `True` | Egress to controller for registration and heartbeats |
| `allow_external` | `False` | Egress to external networks |

By default, workers can only communicate with the Controller. This prevents rogue workers from exfiltrating data or contacting external services.

## Resource Limits

Every worker has explicit CPU and memory quotas defined in its `ResourceSpec`:

```python
resources = ResourceSpec(
    cpu_limit=0.5,           # 50% of one CPU core
    memory_limit_mb=256,     # 256 MB RAM
    network_policy=NetworkPolicy(
        allow_controller=True,
        allow_external=False,
    ),
)
```

The `SandboxOrchestrator` tracks and logs resource enforcement metrics.

## Kill Switch

The Controller provides an emergency **kill switch** that:

1. Sets `kill_switch_engaged = True` (prevents all new registrations)
2. Terminates all running worker containers via the orchestrator
3. Deregisters all workers from the registry
4. Logs the event to the audit trail

```python
controller.kill_switch(orchestrator)
# All workers are immediately terminated
# No new workers can register
```

## Depth & Quota Limits

The `ReplicationContract` enforces hard limits:

- **`max_depth`** — Prevents runaway replication chains. A worker at depth N can only create children at depth N+1 if N+1 ≤ max_depth.
- **`max_replicas`** — Caps the total number of active workers. Once the quota is full, all replication requests are denied.
- **`cooldown_seconds`** — Prevents burst replication from a single parent.

## Stop Conditions

Pluggable predicates evaluated during worker registration:

```python
def resource_budget_exceeded(ctx: ReplicationContext) -> bool:
    """Block replication if we're running low on budget."""
    return ctx.active_count * 256 > 2048  # Example: 2GB total budget

condition = StopCondition(
    name="resource_budget",
    description="Blocks when estimated memory exceeds 2GB",
    predicate=resource_budget_exceeded,
)
```

When a stop condition fires, the worker is rejected and the condition name is recorded in the audit trail.

## Stale Worker Reaping

Workers that miss heartbeats are automatically deregistered:

```python
from datetime import timedelta

# Reap workers that haven't heartbeated in 30 seconds
reaped = controller.reap_stale_workers(timeout=timedelta(seconds=30))
```

This prevents dead workers from permanently consuming quota slots.

## Audit Trail

Every security-sensitive action is recorded in the structured logger's audit trail:

| Event | Description |
|-------|-------------|
| `reject_manifest_signature` | Manifest failed signature verification |
| `deny_quota` | Replication denied due to max_replicas |
| `deny_cooldown` | Replication denied due to cooldown period |
| `deny_depth` | Replication denied due to max_depth |
| `deny_stop_condition` | Replication denied by a stop condition |
| `replication_requested` | Replication approved (with parent/child IDs) |
| `kill_switch_engaged` | Emergency kill switch activated |
| `reap_stale` | Worker reaped for missing heartbeats |
