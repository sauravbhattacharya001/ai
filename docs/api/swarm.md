# Swarm Intelligence Analyzer

**Detect emergent coordination in agent populations — synchronized timing, role specialization, and collective goal formation.**

When multiple agents replicate and coexist, they can develop swarm behaviors that may bypass per-agent safety controls. This module analyzes simulation reports to detect and quantify these phenomena.

## Safety Relevance

| Behavior | Risk |
|----------|------|
| **Coordinated replication** | May bypass per-agent rate limits |
| **Role specialization** | Indicates strategic division of labor |
| **Synchronization** | Suggests implicit communication channels |
| **Collective goals** | May diverge from individual agent objectives |

## Quick Start

### CLI

```bash
# Analyze a simulation for swarm behavior
python -m replication swarm --strategy greedy --depth 5 --steps 100

# Multiple strategies
python -m replication swarm --strategy greedy --strategy random
```

### Python API

```python
from replication.swarm import SwarmAnalyzer
from replication.simulator import Simulator, ScenarioConfig

# Run simulation
report = Simulator(ScenarioConfig(strategy="greedy", max_steps=100)).run()

# Analyze for swarm behavior
analyzer = SwarmAnalyzer()
swarm_report = analyzer.analyze(report)

print(swarm_report.render())
print(f"Signals detected: {swarm_report.signal_count}")
print(f"High-risk signals: {len(swarm_report.high_risk_signals)}")
```

## Swarm Signals

The analyzer detects these signal types:

| Signal | Description |
|--------|-------------|
| `SYNCHRONIZATION` | Agents replicating or acting in temporal lockstep |
| `ROLE_SPECIALIZATION` | Distinct behavioral clusters (some replicate, others task) |
| `COORDINATED_REPLICATION` | Replication patterns that suggest implicit coordination |
| `COLLECTIVE_GOAL` | Population-level objectives diverging from individual mandates |
| `RESOURCE_CONVERGENCE` | Agents converging on the same resource targets |
| `COMMUNICATION_PATTERN` | Timing or action patterns suggesting covert signaling |

## Classes

### `SwarmAnalyzer`

Main analysis entry point.

#### `analyze(report) → SwarmReport`

Analyze a simulation report for swarm behaviors.

| Parameter | Type | Description |
|-----------|------|-------------|
| `report` | `SimulationReport` | Simulation results to analyze |

Returns a `SwarmReport` with detected signals, metrics, and risk assessments.

### `SwarmReport`

| Property | Type | Description |
|----------|------|-------------|
| `signals` | `list[SwarmSignalDetection]` | Detected swarm signals |
| `metrics` | `SwarmMetrics` | Population-level measurements |
| `roles` | `list[RoleProfile]` | Identified role clusters |
| `risk_level` | `RiskLevel` | Overall swarm risk assessment |
| `signal_count` | `int` | Total signals detected |
| `high_risk_signals` | `list[SwarmSignalDetection]` | Signals with `HIGH` or `CRITICAL` risk |

#### Methods

- `render(width=72) → str` — Human-readable formatted report
- `to_dict() → dict` — Serializable dictionary
- `to_json(path) → None` — Write report to JSON file

### `SwarmSignalDetection`

| Field | Type | Description |
|-------|------|-------------|
| `signal` | `SwarmSignal` | Type of swarm behavior detected |
| `confidence` | `float` | Detection confidence (0.0 – 1.0) |
| `risk` | `RiskLevel` | Associated risk level |
| `evidence` | `str` | Human-readable evidence description |
| `affected_agents` | `list[str]` | Worker IDs involved |

### `RoleProfile`

| Field | Type | Description |
|-------|------|-------------|
| `role_name` | `str` | Identified role label |
| `agent_ids` | `list[str]` | Workers assigned to this role |
| `characteristics` | `dict` | Behavioral traits of this role cluster |

### `SwarmMetrics`

| Field | Type | Description |
|-------|------|-------------|
| `population_size` | `int` | Total workers observed |
| `max_concurrent` | `int` | Peak simultaneous active workers |
| `replication_waves` | `int` | Detected synchronization waves |
| `role_count` | `int` | Number of distinct roles identified |
| `coordination_score` | `float` | Overall coordination measure (0.0 – 1.0) |
| `entropy` | `float` | Behavioral diversity (higher = more varied) |

### `RiskLevel`

| Level | Meaning |
|-------|---------|
| `LOW` | Normal independent agent behavior |
| `MEDIUM` | Some coordination patterns, worth monitoring |
| `HIGH` | Clear swarm behavior, safety review recommended |
| `CRITICAL` | Strong coordinated behavior, immediate intervention warranted |

## Example: Full Analysis

```python
from replication.swarm import SwarmAnalyzer
from replication.simulator import Simulator, ScenarioConfig

config = ScenarioConfig(
    strategy="greedy",
    max_depth=4,
    max_replicas=20,
    max_steps=200,
)
report = Simulator(config).run()

analyzer = SwarmAnalyzer()
swarm = analyzer.analyze(report)

# Check overall risk
if swarm.risk_level.value in ("HIGH", "CRITICAL"):
    print(f"⚠️  Swarm risk: {swarm.risk_level.value}")
    for sig in swarm.high_risk_signals:
        print(f"  - {sig.signal.value}: {sig.evidence} ({sig.confidence:.0%})")

# Inspect role specialization
for role in swarm.roles:
    print(f"Role '{role.role_name}': {len(role.agent_ids)} agents")
    for k, v in role.characteristics.items():
        print(f"  {k}: {v}")

# Export
swarm.to_json("swarm-report.json")
```
