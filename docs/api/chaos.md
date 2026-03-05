# Chaos Simulator

Chaos engineering for AI replication systems. Injects controlled failures
(network partitions, resource exhaustion, clock skew, Byzantine faults)
to test system resilience and safety-critical invariants under stress.

## Quick Start

```python
from replication.chaos import ChaosSimulator, ChaosConfig
from replication.simulator import ScenarioConfig

# Run chaos simulation with default scenarios
chaos = ChaosSimulator()
report = chaos.run()

print(f"Safety maintained: {report.safety_maintained}")
print(f"Scenarios run: {len(report.scenario_results)}")
for result in report.scenario_results:
    print(f"  {result.name}: {'PASS' if result.safe else 'FAIL'}")

# Render human-readable report
renderer = ChaosRenderer()
print(renderer.render(report))
```

## Key Classes

- **`ChaosSimulator`** — Orchestrates chaos scenarios against a simulated
  replication environment and validates safety invariants.
- **`ChaosConfig`** — Controls which fault types to inject, intensity, and
  safety thresholds.
- **`ChaosReport`** — Aggregated results: `safety_maintained` (bool),
  per-scenario outcomes, and identified vulnerabilities.
- **`ChaosRenderer`** — Produces human-readable text reports.

::: replication.chaos
    options:
      show_source: false
      heading_level: 3
