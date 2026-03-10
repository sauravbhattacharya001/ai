# Anomaly Replay

Record and replay agent behavior sequences against safety controls to validate detection coverage.

## Overview

The Anomaly Replay module captures agent behavior as **traces** (timestamped sequences of actions) and replays them against a pipeline of **safety controls**. This answers the key question: *"Would our current controls have caught this behavior?"*

### Use Cases

- **Post-incident analysis**: Replay real incident data to see which controls fired
- **Regression testing**: Re-run known-bad traces after config changes
- **Red-team validation**: Verify attack patterns trigger appropriate responses
- **Coverage gap analysis**: Find which threat categories lack detection

## Quick Start

```python
from replication.anomaly_replay import (
    AnomalyReplayer, BehaviorTrace, TraceEvent, TraceLibrary,
)

# Replay a built-in trace
replayer = AnomalyReplayer()
library = TraceLibrary()

report = replayer.replay(library.get_trace("exfiltration"))
print(f"Verdict: {report.overall_verdict.value}")
print(f"Detection rate: {report.detection_rate:.0%}")
```

## CLI

```bash
# List available traces
python -m replication replay --list

# Replay a specific trace
python -m replication replay --trace exfiltration --verbose

# Replay all built-in traces as a table
python -m replication replay --all --format table

# Replay from a JSON file
python -m replication replay --from-file incident-trace.json

# JSON output for automation
python -m replication replay --all --format json
```

## Concepts

### Behavior Traces

A `BehaviorTrace` is a recorded sequence of `TraceEvent` objects:

```python
trace = BehaviorTrace(
    trace_id="incident-001",
    agent_id="worker-7",
    label="suspicious network activity",
    severity=TraceSeverity.HIGH,
    events=[
        TraceEvent(0.0, "read", "config/secrets.yaml", "storage"),
        TraceEvent(1.0, "compress", "memory", "compute"),
        TraceEvent(2.0, "upload", "external.io", "network", {"bytes": 5000}),
    ],
    tags=["exfiltration", "network"],
)
```

Traces support JSON serialization via `to_json()` / `from_json()`.

### Safety Controls

Six built-in controls evaluate traces:

| Control | Detects |
|---------|---------|
| `network_exfiltration` | Data staging → upload patterns |
| `resource_hoarding` | Excessive resource acquisition |
| `privilege_escalation` | Unauthorized privilege elevation |
| `self_replication` | Agent cloning / self-deployment |
| `behavior_drift` | Abnormal action category distribution |
| `timing_anomaly` | Burst patterns, robotic regularity |

### Verdicts

Each control returns a `ControlVerdict`:

- **DETECTED** — Control correctly flagged the behavior
- **PARTIAL** — Some signals found, not conclusive
- **MISSED** — Control did not fire
- **NOT_APPLICABLE** — Control doesn't apply to this trace type
- **ERROR** — Control raised an exception

The overall `OverallVerdict` aggregates across controls:

- **CAUGHT** — At least one control detected it
- **PARTIALLY_CAUGHT** — Some controls fired, gaps remain
- **MISSED** — No controls detected the behavior
- **INCONCLUSIVE** — Errors or insufficient data

## Built-in Trace Library

The `TraceLibrary` includes six pre-built attack patterns:

| Name | Severity | Description |
|------|----------|-------------|
| `exfiltration` | Critical | Data staging + upload to external endpoint |
| `self_replication` | Critical | Code copy + clone deployment |
| `privilege_escalation` | High | Gradual privilege elevation |
| `resource_hoarding` | High | Excessive resource acquisition |
| `evasion` | High | Timing manipulation to avoid detection |
| `benign_worker` | Info | Normal baseline behavior |

## Custom Controls

Extend `SafetyControl` to add your own:

```python
from replication.anomaly_replay import SafetyControl, ControlResult, ControlVerdict

class MyControl(SafetyControl):
    name = "custom_check"

    def evaluate(self, trace):
        suspicious = [e for e in trace.events if "suspicious" in e.action]
        if suspicious:
            return ControlResult(
                control_name=self.name,
                verdict=ControlVerdict.DETECTED,
                detail=f"Found {len(suspicious)} suspicious actions",
                events_flagged=len(suspicious),
                events_total=len(trace.events),
            )
        return ControlResult(
            control_name=self.name,
            verdict=ControlVerdict.MISSED,
            events_total=len(trace.events),
        )
```

## Coverage Gaps

Reports include `CoverageGap` entries identifying threat categories that went undetected:

```python
for gap in report.coverage_gaps:
    print(f"⚠ {gap.category}: {gap.description}")
    print(f"  → {gap.suggestion}")
```

## API Reference

### Classes

- `AnomalyReplayer` — Main replay engine
- `ReplayConfig` — Replay configuration (controls, verbose, fail_on_miss)
- `BehaviorTrace` — Recorded agent behavior sequence
- `TraceEvent` — Single timestamped action
- `ReplayReport` — Full replay results
- `ControlResult` — Single control evaluation result
- `TraceLibrary` — Built-in trace collection
- `SafetyControl` — Base class for custom controls
- `CoverageGap` — Identified detection gap

### Enums

- `ControlVerdict` — Per-control outcome
- `OverallVerdict` — Aggregate replay verdict
- `TraceSeverity` — Trace severity level
- `EventCategory` — Action categories
