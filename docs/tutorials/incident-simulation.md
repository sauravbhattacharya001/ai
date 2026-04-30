# Incident Simulation & Tabletop Exercises

This tutorial covers four interconnected modules for proactive safety
testing: running structured tabletop exercises, hunting for threats
with hypothesis-driven plans, analyzing blast radius when controls fail,
and detecting lateral movement across sandbox boundaries.

## Overview

| Module | Purpose |
|--------|---------|
| `tabletop` | Generate facilitated incident response exercises with phases, injects, and scoring |
| `threat_hunt` | Create hypothesis-driven hunting missions with IOCs and procedures |
| `blast_radius` | Simulate safety-control failure cascades and visualize impact propagation |
| `lateral_movement` | Detect unauthorized cross-boundary access patterns in worker topologies |

Together these modules let you **simulate**, **hunt**, **analyze**, and
**detect** — covering the full spectrum from proactive exercises to
real-time detection.

---

## 1. Tabletop Exercises

Tabletop exercises are structured team drills for practicing AI safety
incident response without touching live systems.

### Running Your First Exercise

```python
from replication.tabletop import TabletopGenerator

gen = TabletopGenerator()

# List available scenarios
for scenario in gen.list_scenarios():
    print(f"  {scenario.id}: {scenario.title}")

# Generate a replication-escape exercise
exercise = gen.generate("replication")
print(exercise.render())
```

### CLI Quick Start

```bash
# List all built-in scenarios
python -m replication tabletop --list

# Generate a specific scenario
python -m replication tabletop --scenario replication

# Export as printable HTML packet
python -m replication tabletop --scenario evasion --html -o exercise.html

# Generate a random scenario for surprise drills
python -m replication tabletop --random

# Full packet with all scenarios
python -m replication tabletop --all --html -o full_packet.html
```

### Exercise Structure

Each generated exercise contains:

- **Scenario brief** — background context and initial conditions
- **Phases** — escalating stages with increasing severity
- **Decision points** — moments where the team must choose a response
- **Inject cards** — surprise complications introduced mid-exercise
- **Facilitator guide** — timing, discussion prompts, learning objectives
- **Scoring rubric** — evaluate team response quality

### Custom Scenarios

Define custom scenarios in JSON for organization-specific drills:

```json
{
  "id": "insider-agent",
  "title": "Insider Agent Compromise",
  "description": "A trusted agent begins exhibiting goal drift after update",
  "phases": [
    {
      "name": "Detection",
      "duration_minutes": 15,
      "description": "Anomaly alerts fire on agent-7's behavior profile",
      "decision_points": [
        "Quarantine immediately or gather more evidence?"
      ]
    }
  ]
}
```

```bash
python -m replication tabletop --file custom.json --html -o custom_exercise.html
```

### Best Practices

!!! tip "Effective Tabletop Exercises"
    - Run exercises **monthly** with rotating scenarios
    - Include team members from different roles (ops, safety, management)
    - Use the **scoring rubric** to track improvement over time
    - Follow each exercise with a 15-minute **hot wash** debrief
    - Record decisions for comparison against actual incidents

---

## 2. Threat Hunting Missions

Threat hunting is proactive — you don't wait for alerts, you actively
search for evidence that safety controls have been bypassed.

### Generating Hunt Plans

```python
from replication.threat_hunt import ThreatHuntPlanner

planner = ThreatHuntPlanner()

# Generate a hunt mission targeting replication attempts
mission = planner.generate_mission(category="replication")
print(f"Mission: {mission.title}")
print(f"Hypothesis: {mission.hypothesis}")
print(f"Difficulty: {mission.difficulty}")

# Print IOCs to look for
for ioc in mission.indicators:
    print(f"  [{ioc.severity}] {ioc.type}: {ioc.description}")
    print(f"    Data source: {ioc.data_source}")

# Step-by-step hunt procedure
for step in mission.procedures:
    print(f"  Step {step.step}: {step.action} (tool: {step.tool})")
```

### CLI Usage

```bash
# Generate a single hunt mission
python -m replication hunt

# Target a specific threat category
python -m replication hunt --category replication

# Advanced difficulty missions
python -m replication hunt --difficulty advanced --format json

# Generate a full hunt playbook (multiple missions)
python -m replication hunt --missions 5 --playbook -o hunt_playbook.html

# List all available hunt categories
python -m replication hunt --list-categories
```

### Hunt Categories

The planner covers these AI-specific threat categories:

- **replication** — unauthorized self-copy or code propagation
- **evasion** — attempts to bypass monitoring or controls
- **exfiltration** — data leaving authorized boundaries
- **escalation** — privilege or capability expansion
- **deception** — misleading safety monitors or operators
- **collusion** — coordinated behavior across agents
- **resource_hoarding** — accumulating compute/storage/tokens
- **goal_drift** — subtle objective function changes

### Integrating with Hunts

Combine hunt plans with live monitoring:

```python
from replication.threat_hunt import ThreatHuntPlanner
from replication.observability import get_metrics

planner = ThreatHuntPlanner()
mission = planner.generate_mission(category="evasion")

# Use the mission's data sources to query real telemetry
for ioc in mission.indicators:
    if ioc.data_source == "behavioral":
        metrics = get_metrics(category="behavior", window="24h")
        # Check metrics against the IOC description
        print(f"Checking: {ioc.description}")
```

---

## 3. Blast Radius Analysis

When a safety control fails, how far does the damage spread? The blast
radius analyzer maps failure cascades through the control dependency graph.

### Basic Analysis

```python
from replication.blast_radius import BlastRadiusAnalyzer

analyzer = BlastRadiusAnalyzer()

# Analyze what happens if audit_trail fails
result = analyzer.analyze("audit_trail")
print(result.summary())

# See all affected controls
for control in result.affected_controls:
    print(f"  {control.name} (depth: {control.depth}, severity: {control.impact})")
```

### Custom Control Dependencies

Add your organization's controls to the dependency graph:

```python
analyzer = BlastRadiusAnalyzer()

# Add custom controls with their dependencies
analyzer.add_control("ml_monitor", depends_on=["audit_trail", "behavior_profiler"])
analyzer.add_control("data_gate", depends_on=["policy", "ml_monitor"])
analyzer.add_control("deploy_lock", depends_on=["compliance", "data_gate"])

# Now analyze cascade including custom controls
result = analyzer.analyze("behavior_profiler")
print(result.summary())
```

### CLI Usage

```bash
# Analyze a single control failure
python -m replication blast-radius --control audit_trail

# Analyze all controls and rank by impact
python -m replication blast-radius --all

# Generate HTML visualization
python -m replication blast-radius --all --format html -o blast.html
```

### Interpreting Results

The analyzer reports:

| Metric | Meaning |
|--------|---------|
| **Depth** | How many dependency hops from the failed control |
| **Affected count** | Total controls impacted by the failure |
| **Critical path** | The longest failure chain |
| **Isolation score** | How well-isolated the control is (higher = safer) |

!!! warning "High-Impact Controls"
    Controls with high **fan-out** (many dependents) are your biggest
    single points of failure. The `--all` analysis ranks controls by
    blast radius — prioritize redundancy for the top entries.

---

## 4. Lateral Movement Detection

In a multi-worker sandbox topology, agents shouldn't access resources
outside their designated zones. The lateral movement detector flags
unauthorized cross-boundary communication.

### Setting Up Detection

```python
from replication.lateral_movement import LateralMovementDetector

detector = LateralMovementDetector(sensitivity="medium")

# Feed observed communications
detector.observe(
    source="worker-3",
    target="worker-7",
    access_type="api_call",
    timestamp="2026-04-29T19:00:00Z"
)

# Analyze for suspicious patterns
findings = detector.analyze()
for finding in findings:
    print(f"  [{finding.risk_score:.1f}] {finding.source} → {finding.target}")
    print(f"    Pattern: {finding.pattern_type}")
    print(f"    Notes: {finding.notes}")
```

### CLI Usage

```bash
# Run detection simulation
python -m replication lateral-movement

# Configure topology size and sensitivity
python -m replication lateral-movement --workers 8 --depth 3 --sensitivity high

# JSON output for integration with alerting
python -m replication lateral-movement --format json
```

### Detection Patterns

The detector identifies several suspicious patterns:

- **Pivot chains** — A→B→C access sequences crossing zone boundaries
- **Credential reuse** — Same auth token used across different zones
- **Zone hopping** — Worker accessing resources 2+ zones away
- **Breadth scanning** — Worker probing many targets in rapid succession
- **Temporal clustering** — Burst of cross-boundary access in short window

### Sensitivity Levels

| Level | Behavior |
|-------|----------|
| `low` | Only flags confirmed unauthorized access |
| `medium` | Flags anomalous patterns + unauthorized access |
| `high` | Flags any cross-boundary communication not in allowlist |

---

## Putting It All Together

A complete proactive safety workflow combines all four modules:

```python
from replication.tabletop import TabletopGenerator
from replication.threat_hunt import ThreatHuntPlanner
from replication.blast_radius import BlastRadiusAnalyzer
from replication.lateral_movement import LateralMovementDetector

# 1. Monthly: Run tabletop exercises to train the team
gen = TabletopGenerator()
exercise = gen.generate("replication")
exercise.to_html("monthly_drill.html")

# 2. Weekly: Generate fresh hunt missions
planner = ThreatHuntPlanner()
missions = [planner.generate_mission() for _ in range(3)]

# 3. On-change: Analyze blast radius when modifying controls
analyzer = BlastRadiusAnalyzer()
for control in ["kill_switch", "quarantine", "compliance"]:
    result = analyzer.analyze(control)
    if result.affected_count > 5:
        print(f"⚠️  {control} has blast radius of {result.affected_count}")

# 4. Continuous: Monitor for lateral movement
detector = LateralMovementDetector(sensitivity="medium")
# ... feed real communication logs ...
findings = detector.analyze()
if findings:
    print(f"🚨 {len(findings)} lateral movement findings!")
```

### Recommended Cadence

| Activity | Frequency | Module |
|----------|-----------|--------|
| Tabletop exercises | Monthly | `tabletop` |
| Threat hunts | Weekly | `threat_hunt` |
| Blast radius review | On topology change | `blast_radius` |
| Lateral movement detection | Continuous | `lateral_movement` |

---

## Next Steps

- See the [Security Model](../concepts/security.md) for the underlying threat model
- Review [Custom Safety Policies](custom-safety-policies.md) to define response actions
- Check the [API Reference](../api/index.md) for full module documentation
