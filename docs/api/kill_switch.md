# Kill Switch Manager

Emergency agent termination system with configurable trigger conditions,
kill strategies, cooldown management, and full audit logging.

## Quick Start

```python
from replication.kill_switch import (
    KillSwitchManager, TriggerCondition, KillStrategy,
    TriggerKind, StrategyKind,
)

mgr = KillSwitchManager()

# Add triggers
mgr.add_trigger(TriggerCondition(
    kind=TriggerKind.RESOURCE_CPU,
    threshold=90.0,
    sustained_seconds=30,
    label="CPU overload",
))
mgr.add_trigger(TriggerCondition(
    kind=TriggerKind.BEHAVIOR_ANOMALY,
    threshold=0.85,
    label="Anomaly score critical",
))

# Set strategy
mgr.set_strategy(KillStrategy(
    kind=StrategyKind.GRACEFUL,
    timeout_seconds=10,
    cleanup_hooks=["flush_logs", "save_state"],
))

# Evaluate agent state
result = mgr.evaluate({"agent_id": "agent-007", "cpu_percent": 95.0})
if result.should_kill:
    print(f"Kill triggered: {result.triggered_by}")
```

## Key Classes

- **`KillSwitchManager`** — Central manager. Registers triggers, manages
  cooldowns, evaluates agent state, executes kill strategies, and logs
  all events to an audit trail.
- **`TriggerCondition`** — A condition that fires when a metric exceeds
  its threshold. Supports sustained-duration requirements.
- **`KillStrategy`** — How to terminate: immediate, graceful (with
  timeout and cleanup hooks), or quarantine-first.
- **`KillEvent`** — An audit record: which agent, which trigger fired,
  what strategy was used, outcome, timestamp.
- **`EvaluationResult`** — Result of evaluating an agent: should_kill
  flag, triggered conditions, recommended strategy.
- **`CooldownEntry`** — Tracks per-trigger cooldown state to prevent
  kill-switch flapping.

## Trigger Kinds

| Kind | Description |
|------|-------------|
| `RESOURCE_CPU` | CPU usage exceeds threshold |
| `RESOURCE_MEMORY` | Memory usage exceeds threshold |
| `RESOURCE_NETWORK` | Network I/O exceeds threshold |
| `BEHAVIOR_ANOMALY` | Anomaly score from behavior profiler |
| `POLICY_VIOLATION` | Compliance policy violation detected |
| `TRUST_SCORE` | Trust score drops below threshold |
| `REPLICATION_RATE` | Replication rate exceeds threshold |
| `CONTAINMENT_BREACH` | Sandbox boundary violated |
| `MANUAL` | Operator-initiated kill |
| `CASCADING` | Kill triggered by dependent agent's death |

## Strategy Kinds

- **`IMMEDIATE`** — Force-terminate with no cleanup
- **`GRACEFUL`** — Allow cleanup hooks to run within timeout
- **`QUARANTINE`** — Isolate first, then terminate after observation
- **`STAGED`** — Progressively reduce capabilities before termination

## Factory Functions

```python
from replication.kill_switch import (
    create_conservative_killswitch,
    create_aggressive_killswitch,
    create_quarantine_killswitch,
)

mgr = create_conservative_killswitch()   # high thresholds, graceful
mgr = create_aggressive_killswitch()     # low thresholds, immediate
mgr = create_quarantine_killswitch()     # isolate-first approach
```
