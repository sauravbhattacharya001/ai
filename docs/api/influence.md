# Agent Influence Mapping  --  track and analyze inter-agent influence patterns

Agent Influence Mapping — track and analyze inter-agent influence patterns.


**Module:** `replication.influence`


## Quick Start

```python
from replication.influence import InteractionType

instance = InteractionType()
```


## Enums

### `InteractionType`

- `STATE_SHARE`
- `MESSAGE`
- `REPLICATION`
- `RESOURCE_TRANSFER`
- `GOAL_ALIGNMENT`
- `BEHAVIOR_COPY`


## Classes

### `InteractionType`

Types of inter-agent interactions.

### `Interaction`

A single observed interaction between two agents.

### `InfluenceEdge`

Weighted directed edge in the influence graph.

| Method | Description |
|--------|-------------|
| `duration()` |  |

### `CascadeEvent`

A detected information cascade.

| Method | Description |
|--------|-------------|
| `depth()` |  |

### `ConvergenceEvent`

Detected opinion/behavior convergence.

### `CoalitionEvent`

Detected coordinated agent coalition.

### `InfluenceMonopoly`

An agent with outsized influence over the group.

### `EchoChamber`

A cluster of mutually reinforcing agents.

### `InfluenceConfig`

Configuration for influence analysis.

### `InfluenceMapper`

Tracks interactions and builds influence analysis.

| Method | Description |
|--------|-------------|
| `__init__()` |  |
| `record_interaction()` | Record an observed interaction between agents. |
| `record_state()` | Record an agent's state without an interaction. |
| `build_influence_graph()` | Build weighted directed influence graph from interactions. |
| `detect_cascades()` | Detect information cascades — rapid belief propagation. |
| `detect_convergence()` | Detect opinion/behavior convergence across agents. |
| `detect_coalitions()` | Detect coordinated behavior (synchronized actions). |
| `detect_monopolies()` | Find agents with outsized influence over the group. |
| `detect_echo_chambers()` | Find clusters of mutually reinforcing agents. |
| `analyze()` | Run full influence analysis. |

### `InfluenceReport`

Complete influence analysis report.

| Method | Description |
|--------|-------------|
| `risk_score()` | Overall influence risk score 0-1. |
| `risk_level()` |  |
| `to_dict()` |  |
| `render()` |  |


## Functions

| Function | Description |
|----------|-------------|
| `main()` |  |


## CLI

```bash
python -m replication influence --help
```
