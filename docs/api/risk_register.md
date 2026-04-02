# Risk Register

Formal risk tracking and lifecycle management for AI agent deployments.

## Overview

The Risk Register provides a structured, auditable system for documenting,
tracking, and managing AI agent risks through their full lifecycle. It bridges
automated risk detection (risk profiler, risk heatmap) with operational risk
management practices.

## Features

- **Structured risk entries** — ID, title, category, likelihood × impact scoring, owner, status
- **Lifecycle states** — Identified → Assessed → Mitigating → Accepted → Closed/Escalated with validated transitions
- **Inherent vs. residual scoring** — track risk reduction from implemented mitigations
- **Owner assignment** — accountability tracking per risk
- **Review scheduling** — configurable review periods with overdue flagging
- **Audit trail** — every state change and mitigation logged with timestamps
- **Score history** — track residual risk over time
- **Multi-format export** — JSON, CSV, and interactive HTML dashboard
- **Import support** — load existing risk data from JSON

## CLI Usage

```bash
# Generate demo register with statistics
python -m replication risk-register

# Simulate with more agents
python -m replication risk-register --agents 15 --seed 42

# Show overdue reviews only
python -m replication risk-register --overdue

# Summary statistics
python -m replication risk-register --stats

# Top 5 riskiest entries
python -m replication risk-register --top 5

# Export formats
python -m replication risk-register --json -o register.json
python -m replication risk-register --csv -o register.csv
python -m replication risk-register --html -o register.html

# Import existing risks
python -m replication risk-register --import risks.json
```

## Programmatic Usage

```python
from replication.risk_register import (
    RiskRegister, RiskEntry, RegisterConfig,
    RiskStatus, RiskCategory, Mitigation
)

# Create and populate
reg = RiskRegister(RegisterConfig(agent_count=10, seed=42))
reg.populate_from_simulation()

# Query
print(reg.summary())
top = reg.top_risks(5)
overdue = reg.overdue_risks()
stats = reg.statistics()

# Lifecycle management
risk = reg.get_risk("RISK-001")
risk.transition(RiskStatus.ASSESSED, user="analyst", note="Initial review complete")
risk.add_mitigation(Mitigation(
    description="Deploy input sanitization",
    effectiveness=0.4,
    status="In Progress",
    owner="security-team",
))
risk.transition(RiskStatus.MITIGATING)

# Export
reg.export_json("register.json")
reg.export_csv("register.csv")
html = reg.to_html()
```

## Risk Lifecycle

```
Identified → Assessed → Mitigating → Accepted → Closed
                    ↘               ↗           ↗
                     → Escalated ──────────────→
```

## Risk Levels

| Score (L×I) | Level    |
|-------------|----------|
| 20–25       | Critical |
| 12–19       | High     |
| 6–11        | Medium   |
| 1–5         | Low      |

## HTML Dashboard

The `--html` flag generates a self-contained interactive dashboard with:
- Summary statistics cards
- Filterable/sortable risk table (search, status, level, category, overdue)
- Click-to-inspect detail panel with mitigations, score history, audit trail
- CSV export button
