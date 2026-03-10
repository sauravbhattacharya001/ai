# Scenario Generator -- automated test scenario creation for safety analysis

Scenario Generator -- automated test scenario creation for safety analysis.


**Module:** `replication.scenarios`


## Quick Start

```python
from replication.scenarios import ScenarioCategory

instance = ScenarioCategory()
```


## Enums

### `ScenarioCategory`

- `BOUNDARY`
- `ADVERSARIAL`
- `RANDOM`
- `GRADIENT`


## Classes

### `ScenarioCategory`

Category of generated scenario.

### `GeneratorConfig`

Configuration for the scenario generator.

### `GeneratedScenario`

A single generated scenario with optional simulation results.

| Method | Description |
|--------|-------------|
| `to_dict()` | Serialize to a JSON-compatible dictionary. |

### `ScenarioSuite`

A collection of generated scenarios with ranking and analysis.

| Method | Description |
|--------|-------------|
| `ranked()` | Return scenarios ranked by interest score (highest first). |
| `by_category()` | Filter scenarios by category. |
| `highest_risk()` | Return the scenario with the highest interest score. |
| `safety_summary()` | Aggregate safety statistics across all scenarios. |
| `render()` | Render a full text report of all generated scenarios. |
| `render_ranking()` | Render a compact ranking table of scenarios by interest score. |
| `to_dict()` | Serialize the full suite to a JSON-compatible dictionary. |

### `ScenarioGenerator`

Generate and evaluate test scenarios for safety analysis.

| Method | Description |
|--------|-------------|
| `__init__()` |  |
| `generate()` | Generate a suite of scenarios, optionally simulating each. |
| `generate_stress_test()` | Generate a large stress test suite focused on adversarial cases. |


## Functions

| Function | Description |
|----------|-------------|
| `score_scenario()` | Score a scenario's 'interestingness' based on simulation results. |
| `generate_boundary_scenarios()` | Generate scenarios at parameter boundaries. |
| `generate_adversarial_scenarios()` | Generate adversarial scenarios designed to stress safety controls. |
| `generate_random_scenarios()` | Generate uniformly random scenarios for broad parameter coverage. |
| `generate_gradient_scenarios()` | Generate gradient-walk scenarios. |
| `main()` |  |


## CLI

```bash
python -m replication scenarios --help
```
