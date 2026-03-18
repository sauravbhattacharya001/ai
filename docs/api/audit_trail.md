# Audit Trail

Tamper-evident, hash-chained event log for recording safety-relevant events.

## Overview

The audit trail provides an append-only, cryptographically chained event log using SHA-256 hashes. Each event is linked to its predecessor, making any tampering or gap detectable via integrity verification.

## Event Categories

| Category | Description |
|----------|-------------|
| `policy` | Policy changes and updates |
| `violation` | Contract or safety violations |
| `killswitch` | Kill switch activations |
| `quarantine` | Worker quarantine actions |
| `escalation` | Threat level escalations |
| `access` | Access control events |
| `config` | Configuration changes |
| `alert` | General safety alerts |

## Severity Levels

- **info** — Informational, no action needed
- **warning** — Potential issue, review recommended
- **critical** — Immediate attention required

## CLI Usage

### Log an Event

```bash
python -m replication audit-trail log \
    --category violation \
    --severity critical \
    --message "Agent X exceeded token budget" \
    --source controller
```

### Search Events

```bash
# By category and severity
python -m replication audit-trail search --category policy --severity warning

# By time range
python -m replication audit-trail search --after 2026-01-01 --before 2026-02-01

# By keyword
python -m replication audit-trail search --keyword "kill switch"
```

### Verify Integrity

Validates the hash chain to detect tampering or gaps:

```bash
python -m replication audit-trail verify
```

### Export

```bash
# Export to HTML timeline
python -m replication audit-trail export --format html -o audit.html

# Export to JSON
python -m replication audit-trail export --format json -o audit.json

# Export to CSV
python -m replication audit-trail export --format csv -o audit.csv
```

### Statistics

```bash
python -m replication audit-trail stats
```

## API Usage

```python
from replication.audit_trail import AuditTrail, AuditEvent

trail = AuditTrail()

# Log an event
trail.log(
    category="violation",
    severity="critical",
    message="Worker exceeded memory limit",
    source="orchestrator",
)

# Search events
results = trail.search(category="violation", severity="critical")

# Verify chain integrity
is_valid = trail.verify()
assert is_valid, "Audit trail has been tampered with!"

# Export
trail.export("audit.html", format="html")
```

## Hash Chain

Each event contains a `prev_hash` field linking to the SHA-256 hash of the previous event. The chain starts with a genesis event (prev_hash = `"0" * 64`). Verification walks the chain and recomputes each hash to confirm continuity.
