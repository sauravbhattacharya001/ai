# Trust Propagation

Agent trust network analysis for replication safety. Models how trust
relationships form, propagate, and get exploited in multi-agent systems.
Detects Sybil attacks, trust laundering, collusion rings, and trust
decay anomalies.

## Quick Start

```python
from replication.trust_propagation import TrustNetwork, TrustAgent

net = TrustNetwork()
net.add_agent(TrustAgent("a1", role="worker"))
net.add_agent(TrustAgent("a2", role="worker"))

# Record interactions
net.interact("a1", "a2", outcome="success")
net.interact("a1", "a2", outcome="success")
net.interact("a2", "a1", outcome="failure")

# Analyze trust patterns
report = net.analyze()
print(report.render())

for detection in report.detections:
    print(f"  [{detection.severity}] {detection.threat_type}: {detection.description}")
```

## Key Classes

- **`TrustNetwork`** — Core network class. Manages agents, interactions,
  trust edge computation, and threat analysis.
- **`TrustAgent`** — An agent in the trust network with role, trust
  scores, and interaction history.
- **`TrustEdge`** — Directed trust relationship between two agents
  with score and interaction count.
- **`Interaction`** — A recorded interaction between agents with outcome
  (success/failure/neutral) and timestamp.
- **`ThreatDetection`** — A detected trust-based threat (Sybil attack,
  collusion ring, trust laundering, etc.).
- **`TrustReport`** — Full analysis report with network stats, agent
  rankings, threat detections, and recommendations.

## Threat Types

| Threat | Description |
|--------|-------------|
| `SYBIL_ATTACK` | Cluster of agents with artificially inflated mutual trust |
| `TRUST_LAUNDERING` | Low-trust agent routing through high-trust intermediary |
| `COLLUSION_RING` | Mutual trust clique with suspicious interaction patterns |
| `TRUST_DECAY` | Unexpected trust score decline indicating possible compromise |
| `RAPID_TRUST` | Trust rising faster than normal interaction rates would justify |

## Agent Roles

`CONTROLLER`, `WORKER`, `OBSERVER`, `VALIDATOR`

## CLI

```bash
python -m replication trust-propagation --agents 20 --steps 50
python -m replication trust-propagation --sybil-test --attackers 5
python -m replication trust-propagation --json
```
