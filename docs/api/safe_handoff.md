# Safe Handoff Protocol

Multi-phase agent responsibility transfer with verification, rollback, and monitoring.

## Quick Start

```bash
python -m replication handoff --inject --agents 8 --handoffs 20
python -m replication handoff --inject --scenario cascade
python -m replication handoff --inject --format html > report.html
```

## Phases

NEGOTIATE -> VERIFY_RECIPIENT -> TRANSFER_STATE -> VALIDATE -> ACTIVATE -> MONITOR

## Risk Detection

- Pair incompatibility, cascade chains, handoff storms, capability hoarding, orphaned capabilities
