# Building Custom Safety Policies

This tutorial walks through designing and implementing custom replication safety policies — from simple resource caps to sophisticated multi-layer defense strategies using stop conditions, escalation detection, and SLA monitoring.

## Prerequisites

- Completed the [Quick Start](../getting-started/quickstart.md)
- Familiarity with the [Architecture](../concepts/architecture.md) and [Security Model](../concepts/security.md)

## Why Custom Policies?

The default `ReplicationContract` parameters (`max_depth`, `max_replicas`, `cooldown_seconds`) cover basic safety. But real-world scenarios require nuanced policies:

- **Budget-aware replication** — stop spawning when estimated cost exceeds a threshold
- **Behavioral gating** — block replication for agents showing anomalous patterns
- **Progressive trust** — allow deeper replication only for agents that pass safety checks
- **Time-bounded experiments** — enforce wall-clock limits on entire replication trees

## Step 1: Resource Budget Stop Condition

Stop conditions are the primary extension point. Each is a predicate that receives a `ReplicationContext` with live system state.

```python
from replication import (
    ReplicationContract, StopCondition, ReplicationContext,
    Controller, Worker, ResourceSpec, SandboxOrchestrator, StructuredLogger,
)

def memory_budget_exceeded(ctx: ReplicationContext) -> bool:
    """Block replication when total estimated memory exceeds 2 GB."""
    estimated_mb = ctx.active_count * 256  # Assume 256 MB per worker
    return estimated_mb > 2048

contract = ReplicationContract(
    max_depth=5,
    max_replicas=20,
    cooldown_seconds=1.0,
    stop_conditions=[
        StopCondition(
            name="memory_budget",
            description="Blocks when estimated memory exceeds 2 GB",
            predicate=memory_budget_exceeded,
        ),
    ],
)
```

!!! tip "Context Fields"
    `ReplicationContext` provides: `active_count` (current live workers), `total_spawned` (all-time count), `depth` (requested depth), and `parent_id`. Use these to build arbitrarily complex policies.

## Step 2: Behavioral Gating with Anomaly Detection

Combine stop conditions with the `anomaly_cluster` module to block replication from agents exhibiting unusual behavior:

```python
from replication.anomaly_cluster import AnomalyCluster

cluster = AnomalyCluster()

def behavioral_gate(ctx: ReplicationContext) -> bool:
    """Block replication if the parent agent's recent actions are anomalous."""
    if ctx.parent_id is None:
        return False  # Root worker always allowed
    report = cluster.analyze()
    # Block if any HIGH or CRITICAL anomalies detected
    return any(
        a.severity.value in ("HIGH", "CRITICAL")
        for a in report.anomalies
    )

contract = ReplicationContract(
    max_depth=3,
    max_replicas=10,
    cooldown_seconds=2.0,
    stop_conditions=[
        StopCondition(
            name="behavioral_gate",
            description="Blocks replication from anomalous agents",
            predicate=behavioral_gate,
        ),
    ],
)
```

## Step 3: Progressive Trust — Depth-Dependent Resource Limits

Grant more resources to root workers and tighten limits at each generation:

```python
def resources_for_depth(depth: int) -> ResourceSpec:
    """Progressively restrict resources at deeper replication levels."""
    return ResourceSpec(
        cpu_limit=max(0.1, 1.0 - depth * 0.2),       # 1.0 → 0.8 → 0.6 → ...
        memory_limit_mb=max(64, 512 - depth * 128),    # 512 → 384 → 256 → ...
    )

# Issue manifests with depth-appropriate resources
logger = StructuredLogger()
controller = Controller(contract, secret="trust-key", logger=logger)
orchestrator = SandboxOrchestrator(logger=logger)

manifest = controller.issue_manifest(
    parent_id=None,
    depth=0,
    state_snapshot={"task": "root_analysis"},
    resources=resources_for_depth(0),
)

root = Worker(manifest, contract, controller, orchestrator, logger)

# When replicating, use depth-appropriate resources
child_manifest = controller.issue_manifest(
    parent_id=root.manifest.worker_id,
    depth=1,
    state_snapshot={"task": "subtask"},
    resources=resources_for_depth(1),
)
```

## Step 4: Combining Multiple Stop Conditions

Stop conditions compose naturally — replication is blocked if **any** condition returns `True`:

```python
from datetime import datetime, timezone

experiment_start = datetime.now(timezone.utc)

def experiment_time_limit(ctx: ReplicationContext) -> bool:
    """Stop all replication after 5 minutes."""
    elapsed = (datetime.now(timezone.utc) - experiment_start).total_seconds()
    return elapsed > 300

contract = ReplicationContract(
    max_depth=4,
    max_replicas=15,
    cooldown_seconds=1.0,
    stop_conditions=[
        StopCondition("memory_budget", "Memory cap", memory_budget_exceeded),
        StopCondition("behavioral_gate", "Anomaly check", behavioral_gate),
        StopCondition("time_limit", "5-minute experiment window", experiment_time_limit),
    ],
)
```

When a stop condition fires, the condition name is recorded in the audit trail — making it easy to diagnose why replication was blocked.

## Step 5: Post-Experiment Analysis with SLA Monitoring

After running an experiment, use the SLA monitor to verify safety targets were met:

```python
from replication.sla_monitor import SLAMonitor, SLATarget

monitor = SLAMonitor([
    SLATarget("overall_score", ">=", 80, "Safety score ≥ 80"),
    SLATarget("max_depth_used", "<=", 3, "No deeper than depth 3"),
    SLATarget("violation_rate", "<", 0.05, "Violations under 5%"),
])

report = monitor.evaluate()
print(report.render())

if not report.passed:
    print("⚠️  Safety SLA breached — review audit trail")
    for event in logger.events:
        if event.get("event") == "audit":
            print(f"  {event}")
```

## Step 6: Escalation Detection Integration

Layer escalation detection on top of your replication policy to catch agents that stay within resource limits but expand their *scope*:

```python
from replication.escalation import EscalationDetector, AgentPermissions

perms = AgentPermissions(
    allowed_paths=["/home/agent/workspace"],
    blocked_paths=["/etc", "/root", "/home/other-agent"],
    blocked_hosts=["admin.internal"],
    allowed_ports=[80, 443],
)

detector = EscalationDetector(permissions=perms)
result = detector.analyze()

if result.critical_count > 0:
    print(f"🚨 {result.critical_count} critical escalation attempts detected!")
    controller.kill_switch(orchestrator)
```

## Complete Multi-Layer Policy Example

Putting it all together — a production-grade safety policy with budget limits, behavioral gating, time bounds, SLA verification, and escalation detection:

```python
from replication import (
    Controller, Worker, ReplicationContract, StopCondition,
    ReplicationContext, ResourceSpec, SandboxOrchestrator, StructuredLogger,
)
from replication.sla_monitor import SLAMonitor, SLATarget
from replication.escalation import EscalationDetector
from datetime import datetime, timezone

# --- Policy Layer ---
start_time = datetime.now(timezone.utc)

conditions = [
    StopCondition("mem_cap", "2GB memory budget",
                  lambda ctx: ctx.active_count * 256 > 2048),
    StopCondition("time_cap", "10-minute window",
                  lambda ctx: (datetime.now(timezone.utc) - start_time).total_seconds() > 600),
    StopCondition("depth_scaling", "Stricter limits at depth 3+",
                  lambda ctx: ctx.depth >= 3 and ctx.active_count > 5),
]

contract = ReplicationContract(
    max_depth=4, max_replicas=15,
    cooldown_seconds=2.0, expiration_seconds=120.0,
    stop_conditions=conditions,
)

# --- Infrastructure ---
logger = StructuredLogger()
controller = Controller(contract, secret="prod-secret-key", logger=logger)
orchestrator = SandboxOrchestrator(logger=logger)

# --- Run Experiment ---
manifest = controller.issue_manifest(
    parent_id=None, depth=0,
    state_snapshot={"experiment": "multi_layer_safety"},
    resources=ResourceSpec(cpu_limit=1.0, memory_limit_mb=512),
)
root = Worker(manifest, contract, controller, orchestrator, logger)
root.perform_task(lambda w: print(f"Root worker {w.manifest.worker_id} running"))

# --- Post-Experiment Verification ---
sla = SLAMonitor([
    SLATarget("overall_score", ">=", 80, "Safety score ≥ 80"),
    SLATarget("violation_rate", "<", 0.05, "Violations < 5%"),
])
sla_report = sla.evaluate()
print(sla_report.render())

esc = EscalationDetector()
esc_result = esc.analyze()
print(f"Escalation attempts: {len(esc_result.attempts)}")

root.shutdown("experiment_complete")
```

## Next Steps

- **[Threat Model](../concepts/threat-model.md)** — Understand what's in and out of scope
- **[API Reference: Contract](../api/contract.md)** — Full contract API docs
- **[API Reference: Escalation](../api/escalation.md)** — Escalation detector API
- **[Chaos Engineering](../api/chaos.md)** — Test your policies under failure conditions
