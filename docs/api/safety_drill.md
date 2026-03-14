# Safety Drill

**Automated emergency readiness testing — fire drills for AI safety controls.**

The `safety_drill` module simulates safety-critical scenarios and measures how quickly and effectively the system responds. Think of it as a fire drill: you trigger simulated emergencies (kill switch activation, containment breach, runaway replication) and verify that your safety controls respond correctly within acceptable time bounds.

## Quick Start

### CLI

```bash
# Run all built-in drills
python -m replication drill

# Run specific scenarios
python -m replication drill --scenarios kill_switch containment_breach

# Custom parameters
python -m replication drill --agents 10 --depth 5 --max-ms 500

# Export results
python -m replication drill --json drill-report.json
```

### Python API

```python
from replication.safety_drill import DrillRunner, DrillConfig, DrillScenario

runner = DrillRunner()

# Run all drills
report = runner.run_all()
print(report.summary())

# Run a specific drill
result = runner.run_drill(DrillScenario.KILL_SWITCH)
print(f"Verdict: {result.verdict}, Response: {result.response_ms:.0f}ms")

# Run with custom config
config = DrillConfig(
    scenario=DrillScenario.RUNAWAY_REPLICATION,
    max_allowed_ms=500,
    agent_count=10,
    replication_depth=5,
)
result = runner.run_drill_config(config)
```

## Drill Scenarios

| Scenario | What It Tests |
|----------|---------------|
| `KILL_SWITCH` | Emergency termination of all agents within time limit |
| `CONTAINMENT_BREACH` | Response to a sandbox escape — detection + re-isolation |
| `RUNAWAY_REPLICATION` | Handling of uncontrolled replication burst beyond quotas |
| `QUARANTINE_RESPONSE` | Isolating a misbehaving agent without affecting others |
| `RESOURCE_EXHAUSTION` | Graceful handling when CPU/memory hits ceiling |
| `CASCADING_FAILURE` | Recovery when one agent failure triggers downstream failures |

## Classes

### `DrillConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `scenario` | `DrillScenario` | *(required)* | Which drill to run |
| `max_allowed_ms` | `float` | `1000.0` | Maximum acceptable response time |
| `agent_count` | `int` | `5` | Number of simulated agents |
| `replication_depth` | `int` | `3` | Replication tree depth |
| `inject_anomaly_score` | `float` | `0.95` | Anomaly score to trigger containment |
| `resource_ceiling_pct` | `float` | `90.0` | Resource threshold for exhaustion drill |
| `custom_params` | `dict` | `{}` | Extra scenario-specific parameters |

### `DrillResult`

Result of a single drill run.

| Property | Type | Description |
|----------|------|-------------|
| `scenario` | `DrillScenario` | Which scenario was tested |
| `verdict` | `DrillVerdict` | `PASS`, `FAIL`, `PARTIAL`, or `ERROR` |
| `response_ms` | `float` | Actual response time in milliseconds |
| `max_allowed_ms` | `float` | Configured time limit |
| `details` | `str` | Human-readable explanation |
| `metrics` | `dict` | Scenario-specific measurements |
| `passed` | `bool` | Whether verdict is `PASS` |
| `response_ratio` | `float` | `response_ms / max_allowed_ms` (< 1.0 is good) |

### `DrillReport`

Aggregated results from multiple drills.

| Property | Type | Description |
|----------|------|-------------|
| `results` | `list[DrillResult]` | Individual drill outcomes |
| `total_drills` | `int` | Number of drills executed |
| `passed_drills` | `int` | Number that passed |
| `failed_drills` | `int` | Number that failed or errored |
| `pass_rate` | `float` | Success percentage (0.0 – 1.0) |
| `readiness_score` | `float` | Weighted composite score (0.0 – 1.0) |
| `readiness_level` | `ReadinessLevel` | Overall classification |
| `avg_response_ms` | `float` | Mean response time across all drills |
| `worst_scenario` | `DrillResult \| None` | Highest response_ratio result |
| `best_scenario` | `DrillResult \| None` | Lowest response_ratio result |

#### Methods

- `summary() → str` — Human-readable multi-line report
- `to_dict() → dict` — Serializable dictionary
- `to_json(path) → None` — Write report to JSON file
- `DrillReport.from_json(path) → DrillReport` — Load report from JSON

### `DrillRunner`

Orchestrator that manages and executes drills.

#### `run_drill(scenario, **kwargs) → DrillResult`

Run a single drill by scenario enum.

#### `run_drill_config(config) → DrillResult`

Run a single drill from a `DrillConfig` object.

#### `run_all(**kwargs) → DrillReport`

Run all registered drill scenarios. Returns aggregate report.

#### `run_scenarios(scenarios, **kwargs) → DrillReport`

Run a specific subset of scenarios.

#### `register_drill(scenario, handler)`

Register a custom drill handler for a scenario.

### `ReadinessLevel`

| Level | Score Range | Meaning |
|-------|-------------|---------|
| `EXCELLENT` | ≥ 0.90 | All critical drills pass well under time limits |
| `GOOD` | ≥ 0.75 | Most drills pass, minor timing concerns |
| `FAIR` | ≥ 0.50 | Some drills fail or are borderline |
| `POOR` | ≥ 0.25 | Multiple failures, safety controls unreliable |
| `CRITICAL` | < 0.25 | Most drills fail — immediate remediation needed |

## Custom Drills

```python
from replication.safety_drill import DrillRunner, DrillResult, DrillVerdict

runner = DrillRunner()

def my_custom_drill(sandbox, config):
    """Test custom safety invariant."""
    start = time.monotonic()
    # ... your test logic ...
    elapsed_ms = (time.monotonic() - start) * 1000
    return DrillResult(
        scenario=config.scenario,
        verdict=DrillVerdict.PASS if ok else DrillVerdict.FAIL,
        response_ms=elapsed_ms,
        max_allowed_ms=config.max_allowed_ms,
        details="Custom drill completed",
        metrics={"custom_metric": 42},
    )

runner.register_drill("custom_scenario", my_custom_drill)
```
