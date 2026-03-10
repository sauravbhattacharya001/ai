# Incident Response Playbook  --  automated response plans for safety events

Incident Response Playbook — automated response plans for safety events.


**Module:** `replication.incident`


## Quick Start

```python
from replication.incident import IncidentSeverity

instance = IncidentSeverity()
```


## Enums

### `IncidentSeverity`

- `LOW`
- `MEDIUM`
- `HIGH`
- `CRITICAL`

### `IncidentCategory`

- `DRIFT`
- `COMPLIANCE`
- `QUARANTINE`
- `ANOMALY`
- `ESCAPE`
- `RESOURCE`

### `StepPriority`

- `IMMEDIATE`
- `HIGH`
- `MEDIUM`
- `LOW`
- `FOLLOWUP`

### `StepStatus`

- `PENDING`
- `IN_PROGRESS`
- `COMPLETED`
- `SKIPPED`
- `FAILED`


## Classes

### `IncidentSeverity`

Severity of the incident.

### `IncidentCategory`

What kind of safety event triggered this.

### `StepPriority`

Urgency of each response step.

### `StepStatus`

Tracking status for playbook execution.

### `ResponseStep`

A single action in the incident response playbook.

| Method | Description |
|--------|-------------|
| `complete()` |  |
| `skip()` |  |
| `to_dict()` |  |

### `Playbook`

An ordered set of response steps for an incident.

| Method | Description |
|--------|-------------|
| `progress()` | Fraction of steps completed (0.0-1.0). |
| `pending_steps()` |  |
| `next_step()` |  |
| `resolve()` |  |
| `to_dict()` |  |
| `to_json()` |  |

### `IncidentConfig`

Configuration for the incident responder.

### `IncidentResponder`

Generates and manages incident response playbooks.

| Method | Description |
|--------|-------------|
| `__init__()` |  |
| `from_drift_alert()` | Generate a response playbook from a drift alert. |
| `from_compliance_finding()` | Generate a response playbook from a compliance audit finding. |
| `from_escape_event()` | Generate a CRITICAL playbook for a potential escape event. |
| `from_resource_anomaly()` | Generate a playbook for resource usage anomalies (CPU, memory, network). |
| `get_playbook()` |  |
| `list_active()` |  |
| `list_all()` |  |
| `summary()` | Dashboard-friendly summary of all incidents. |
| `export_all()` | Export all playbooks as JSON. |
