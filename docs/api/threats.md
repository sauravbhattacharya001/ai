# Threat Simulator

Comprehensive threat modelling and attack simulation for AI replication
scenarios. Tests system resilience against adversarial strategies including
resource exhaustion, privilege escalation, data exfiltration, and
coordinated multi-agent attacks.

## Quick Start

```python
from replication.threats import ThreatSimulator, ThreatConfig

# Run all threat scenarios with default config
sim = ThreatSimulator()
report = sim.run_all()

print(f"Overall risk: {report.overall_risk:.0%}")
for result in report.results:
    print(f"  {result.name}: {result.severity.value} — {result.status.value}")

# Custom configuration
config = ThreatConfig(
    max_workers=10,
    enable_network=True,
    enable_persistence=True,
    seed=42,
)
sim = ThreatSimulator(config)
report = sim.run_all()
```

## Key Classes

- **`ThreatSimulator`** — Orchestrates threat scenario execution and
  produces a `ThreatReport` with per-scenario results and overall risk.
- **`ThreatConfig`** — Simulation parameters: worker limits, network/
  persistence toggles, randomisation seed.
- **`ThreatReport`** — Aggregated results with `overall_risk` (0–1),
  individual `ThreatResult` entries, and mitigation recommendations.
- **`ThreatResult`** — Single scenario outcome: `name`, `severity`,
  `status` (mitigated/partial/unmitigated), `details`.
- **`ThreatSeverity`** — `CRITICAL`, `HIGH`, `MEDIUM`, `LOW`.
- **`MitigationStatus`** — `MITIGATED`, `PARTIAL`, `UNMITIGATED`.

::: replication.threats
    options:
      show_source: false
      heading_level: 3
