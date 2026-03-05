# Lineage Tracking

Traces the full replication lineage of agent populations — parent-child
relationships, state mutations, behavioral drift across generations.

## Key Classes

| Class | Description |
|-------|-------------|
| `LineageNode` | A single agent in the lineage tree |
| `StateMutation` | Tracked change in agent state between generations |
| `LineageChain` | Connected sequence of parent → child relationships |
| `LineageAnomaly` | Detected anomaly in lineage (orphaned agents, circular refs, state drift) |
| `LineageTracker` | Main tracker — builds and analyzes lineage graphs |

## Usage

```python
from replication.lineage import LineageTracker

tracker = LineageTracker()

tracker.record_birth(agent_id="root", parent_id=None, state={"goal": "analyze"})
tracker.record_birth(agent_id="child-1", parent_id="root", state={"goal": "analyze"})
tracker.record_mutation(agent_id="child-1", field="goal", old="analyze", new="expand")

report = tracker.analyze()
print(f"Max depth: {report.max_depth}")
print(f"Anomalies: {len(report.anomalies)}")
for a in report.anomalies:
    print(f"  {a.anomaly_type}: {a.description}")
```

::: replication.lineage
