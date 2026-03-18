---
title: Alert Router
---

# Alert Router

Rule-based safety alert routing with rate limiting, severity escalation, and quiet-hours suppression.

## Overview

The Alert Router evaluates safety events (from the audit trail or any source) against user-defined routing rules and dispatches notifications to configured channels. It supports:

- **Rule matching** by category, severity, source, and keywords
- **4 channel types**: console (coloured), file (append), JSON-lines, webhook (stub)
- **Rate limiting**: cap alerts per rule within a time window
- **Severity escalation**: auto-upgrade severity after repeated triggers
- **Quiet hours**: suppress non-critical alerts during off-hours
- **Dry-run mode**: preview routing without dispatching

## CLI Usage

```bash
# Route a single event
python -m replication alert-router route --category violation --severity critical \
    --message "Token budget exceeded" --source controller

# Dry-run (no dispatch)
python -m replication alert-router route --category escalation --severity warning --dry-run

# Test rules against an event
python -m replication alert-router test \
    --event '{"category":"violation","severity":"critical","message":"test"}'

# Show stats with sample events
python -m replication alert-router stats
```

## Programmatic Usage

```python
from replication.alert_router import AlertRouter, RoutingRule, Channel, QuietHours

# Create router with custom rules
router = AlertRouter(
    quiet_hours=QuietHours(start_hour=22, end_hour=7),
)

router.add_rule(RoutingRule(
    name="critical-violations",
    match_category={"violation", "escalation"},
    match_severity={"critical"},
    channels=[
        Channel(kind="console"),
        Channel(kind="file", path="alerts.log"),
    ],
    rate_limit=5,
    rate_window=60,
    escalate_after=3,
    escalate_to="critical",
))

# Route an event
results = router.route({
    "category": "violation",
    "severity": "critical",
    "message": "Agent exceeded token budget",
    "source": "controller",
})

# Check stats
print(router.render_stats())
print(router.render_rules())
```

## Default Router

`default_router()` creates a router with three sensible rules:

| Rule | Matches | Channels |
|------|---------|----------|
| `critical-all` | All critical events | console + JSONL |
| `violations` | violation/escalation/killswitch | console + file |
| `audit-log` | policy/config/access warnings+ | JSONL |

## Features

### Rate Limiting

Prevent alert storms by capping dispatches per rule:

```python
RoutingRule(
    name="limited",
    match_severity={"warning"},
    channels=[Channel(kind="console")],
    rate_limit=10,    # max 10 alerts
    rate_window=60,   # per 60 seconds
)
```

### Severity Escalation

Auto-upgrade severity when a rule fires repeatedly:

```python
RoutingRule(
    name="escalating",
    match_category={"violation"},
    channels=[Channel(kind="console")],
    escalate_after=3,       # after 3 triggers in window
    escalate_to="critical", # bump to critical
)
```

### Quiet Hours

Suppress non-critical alerts during off-hours:

```python
QuietHours(
    start_hour=22,        # 10 PM
    end_hour=7,           # 7 AM
    suppress_below="critical",  # only critical gets through
)
```
