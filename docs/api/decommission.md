# Decommission Planner

Plan and validate safe AI agent retirement and teardown.

## Overview

The Decommission Planner ensures agents are safely retired with no orphaned
resources, dangling permissions, or residual data left behind.

## CLI

```bash
# Plan decommission for a specific agent
python -m replication decommission --agent worker-007

# Dry-run validation (default)
python -m replication decommission --agent worker-007 --dry-run

# Force past blockers
python -m replication decommission --agent worker-007 --force

# Plan for all idle fleet agents
python -m replication decommission --fleet

# Find orphaned resources
python -m replication decommission --list-orphans

# Demo mode
python -m replication decommission --demo

# JSON output
python -m replication decommission --agent worker-007 --json
```

## Python API

```python
from replication import DecommissionPlanner, AgentInventory

planner = DecommissionPlanner()

# Discover agent footprint
inventory = planner.discover("worker-007")

# Generate teardown plan
plan = planner.plan(inventory)

# Validate (dry-run)
report = planner.validate(plan, inventory)
print(planner.render(report))

# Check for orphans across fleet
orphans = planner.find_orphans(["worker-001", "worker-002", "worker-003"])
```

## Decommission Phases

1. **Notification** — Alert downstream dependent agents
2. **Drain** — Gracefully close active connections
3. **Children** — Recursively decommission child agents
4. **Permissions** — Revoke all granted permissions
5. **Resources** — Release compute, storage, network, credentials
6. **Data** — Purge data stores (with archive recommendation)
7. **Registry** — Remove from controller registry
8. **Verify** — Post-decommission health checks

## Classes

| Class | Description |
|-------|-------------|
| `DecommissionPlanner` | Main planner with discover/plan/validate/render |
| `AgentInventory` | Full inventory of an agent's footprint |
| `DecommissionPlan` | Ordered teardown steps with risk assessment |
| `TeardownStep` | Single step in the decommission sequence |
| `DecommissionReport` | Validation results and orphan detection |
