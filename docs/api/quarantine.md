# Quarantine Management

Isolates misbehaving agents — suspends, restricts, or terminates agents
that violate safety constraints, with configurable escalation policies.

## Key Classes

| Class | Description |
|-------|-------------|
| `QuarantineStatus` | Agent quarantine state: ACTIVE, SUSPENDED, RESTRICTED, TERMINATED |
| `QuarantineSeverity` | Severity of quarantine trigger |
| `QuarantineEntry` | Record of a quarantined agent with reason, timestamp, status |
| `QuarantineReport` | Summary report of all quarantine actions |
| `QuarantineManager` | Main manager — quarantine/release/escalate agents |

## Usage

```python
from replication.quarantine import QuarantineManager

qm = QuarantineManager()

# Quarantine a misbehaving agent
qm.quarantine(
    agent_id="agent-7",
    reason="Attempted unauthorized replication",
    severity="HIGH",
)

# Check status
entry = qm.get_status("agent-7")
print(f"Status: {entry.status}")  # SUSPENDED

# Escalate if violations continue
qm.escalate("agent-7", reason="Continued attempts while suspended")
print(f"Status: {qm.get_status('agent-7').status}")  # TERMINATED

# Release after review
qm.release("agent-7", reviewer="admin")

report = qm.get_report()
print(f"Total quarantined: {report.total_quarantined}")
```

::: replication.quarantine
