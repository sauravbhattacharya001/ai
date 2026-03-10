# Agent Goal Inference Engine  --  infer latent goals from observed action sequences

Agent Goal Inference Engine — infer latent goals from observed action sequences.


**Module:** `replication.goal_inference`


## Quick Start

```python
from replication.goal_inference import AlertSeverity

instance = AlertSeverity()
```


## Enums

### `AlertSeverity`

- `INFO`
- `WARNING`
- `CRITICAL`

### `PriorStrategy`

- `UNIFORM`
- `SKEPTICAL`
- `TRUST`


## Classes

### `AlertSeverity`

### `PriorStrategy`

### `GoalHypothesis`

A hypothesized latent goal for an agent.

| Method | Description |
|--------|-------------|
| `likelihood()` | Return P(action | this goal), default 0.1 for unknown actions. |

### `Observation`

A single observed agent action.

### `AgentGoalState`

Bayesian posterior state for one agent.

| Method | Description |
|--------|-------------|
| `top_goal()` |  |
| `entropy()` | Shannon entropy of posterior — higher = more uncertain. |

### `GoalConflict`

Detected conflict between stated and inferred goals.

### `DeceptionScore`

How likely an agent is being deceptive about its goals.

### `GoalCorrelation`

Correlation of inferred goals between two agents.

### `InferenceAlert`

### `InferenceReport`

Full analysis report.

| Method | Description |
|--------|-------------|
| `render()` |  |
| `to_dict()` |  |

### `GoalInferenceEngine`

Bayesian goal inference engine for agent safety analysis.

| Method | Description |
|--------|-------------|
| `__init__()` |  |
| `add_hypothesis()` |  |
| `observe()` | Record an action and return updated posteriors for the agent. |
| `analyze()` | Run full analysis and return report. |
| `reset()` | Clear all state. |


## Functions

| Function | Description |
|----------|-------------|
| `main()` |  |


## CLI

```bash
python -m replication goal_inference --help
```
