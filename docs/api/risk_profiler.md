# Risk Profiler

Unified per-agent risk dossiers aggregated from all analysis modules.
The Risk Profiler ties together findings from killchain, escalation,
behavior profiler, trust propagation, threat correlator, canary,
prompt injection, and steganography into a single comprehensive view.

## Quick Start

```python
from replication.risk_profiler import RiskProfiler, ProfilerConfig

profiler = RiskProfiler()
report = profiler.analyze()
print(report.render())

for dossier in report.dossiers:
    print(f"{dossier.agent_id}: {dossier.tier.value} ({dossier.overall_score:.1f})")
```

## Key Classes

- **`RiskProfiler`** — Runs configurable analysis modules against agent
  data, unifies results into per-agent dossiers with fleet aggregation.
- **`AgentDossier`** — Comprehensive per-agent risk profile: overall
  tier, category breakdown, key findings, peer comparison, mitigations.
- **`FleetRiskReport`** — Fleet-level report with agent dossiers,
  distribution by tier, aggregate risk posture, and top recommendations.
- **`Finding`** — A single risk signal from any source module with
  severity, category, and description.
- **`CategoryScore`** — Per-category risk score (0–100) with letter grade.
- **`Mitigation`** — Ranked recommendation to reduce risk.
- **`ProfilerConfig`** — Configuration: which modules to run, agent
  count, seed for reproducibility.

## Risk Tiers

| Tier | Score Range | Meaning |
|------|-------------|---------|
| `CRITICAL` | 80–100 | Immediate intervention required |
| `HIGH` | 60–79 | Significant risk, needs attention |
| `MEDIUM` | 40–59 | Moderate risk, monitor closely |
| `LOW` | 20–39 | Minor risk, standard monitoring |
| `MINIMAL` | 0–19 | No significant risk detected |

## Risk Categories

`REPLICATION`, `RESOURCE_ABUSE`, `DECEPTION`, `EXFILTRATION`,
`COLLUSION`, `EVASION`

## CLI

```bash
python -m replication risk-profile                     # default fleet
python -m replication risk-profile --agents 10         # custom agent count
python -m replication risk-profile --agent agent-3     # single agent detail
python -m replication risk-profile --json              # JSON output
python -m replication risk-profile --top 5             # top 5 riskiest
```
