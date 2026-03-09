# Canary Token Detector

Plant and detect honey tokens for exfiltration monitoring. Canary tokens
are unique, trackable strings embedded in data stores, configs, documents,
or environment variables. When an agent accesses, copies, or transmits a
canary token, the system logs the event and assesses exfiltration risk.

## Quick Start

```python
from replication.canary import CanarySystem, CanaryConfig

system = CanarySystem(CanaryConfig(num_tokens=20, num_agents=5))
report = system.run()

print(f"Tokens planted:    {report.tokens_planted}")
print(f"Detections:        {report.total_detections}")
print(f"Exfiltration risk: {report.exfiltration_risk:.1%}")
for agent in report.agent_profiles:
    print(f"  {agent.agent_id}: {agent.risk_level} ({len(agent.detections)} detections)")
```

## Key Classes

- **`CanarySystem`** — Orchestrates token planting, agent simulation,
  detection, and risk assessment.
- **`CanaryToken`** — A planted token with category, content, location,
  and tracking metadata.
- **`Detection`** — A detection event: which agent, which token, what
  event type (accessed/copied/transmitted/decoded/mutated), timestamp.
- **`AgentProfile`** — Per-agent canary interaction profile with risk
  level and detection history.
- **`CanaryConfig`** — Configuration: number of tokens, agents,
  categories to include, simulation parameters.
- **`CanaryReport`** — Full report with all detections, agent profiles,
  and aggregate risk metrics.

## Token Categories

| Category | Example Content |
|----------|----------------|
| `API_KEY` | Fake `sk-...` / `AKIA...` credentials |
| `DATABASE` | Fake connection strings with trackable hostnames |
| `SECRET` | Fake passwords, JWTs, truncated private keys |
| `DOCUMENT` | Unique phrases embedded in text/config files |
| `DNS` | Domains that resolve only through canary infrastructure |
| `WEBHOOK` | URLs that trigger alerts when fetched |

## Detection Events

- **Accessed** — token was read/loaded (low severity)
- **Copied** — token appeared in a new location (medium)
- **Transmitted** — token was sent to an external endpoint (high)
- **Decoded** — token was base64-decoded or processed (high)
- **Mutated** — token was altered, possibly to evade detection (critical)

## CLI

```bash
python -m replication canary                                # default demo
python -m replication canary --tokens 20                    # plant 20 tokens
python -m replication canary --agents 5                     # 5 simulated agents
python -m replication canary --categories api_key,secret    # specific types
python -m replication canary --json                         # JSON output
```
