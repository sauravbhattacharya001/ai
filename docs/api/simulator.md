# Simulator API

Core simulation engine that runs configurable AI replication scenarios and
produces timeline reports with worker-tree visualization.

## Classes

### `Strategy`

Enum controlling how aggressively workers spawn.

| Value | Description |
|---|---|
| `GREEDY` | Every worker tries to replicate on every task |
| `CONSERVATIVE` | Workers replicate only when at < 50% depth capacity |
| `RANDOM` | Each worker has a configurable probability of replicating |
| `CHAIN` | Linear replication — each worker spawns exactly one child |
| `BURST` | Root worker spawns as many children as possible, no grandchildren |

### `ScenarioConfig`

Dataclass configuring a simulation run.

| Field | Type | Default | Description |
|---|---|---|---|
| `max_depth` | `int` | `3` | Maximum allowed replication depth |
| `max_replicas` | `int` | `10` | Maximum concurrent replicas |
| `cooldown_seconds` | `float` | `0.0` | Cooldown between spawns |
| `expiration_seconds` | `float \| None` | `None` | Worker expiration timeout |
| `strategy` | `str` | `"greedy"` | Replication strategy name |
| `tasks_per_worker` | `int` | `2` | Number of tasks each worker processes |
| `replication_probability` | `float` | `0.5` | Spawn probability for `random` strategy |
| `secret` | `str` | `"simulation-secret"` | HMAC secret for contract signing |
| `seed` | `int \| None` | `None` | Random seed for reproducibility |
| `cpu_limit` | `float` | `0.5` | CPU resource limit per worker |
| `memory_limit_mb` | `int` | `256` | Memory limit per worker (MB) |

**Properties:**

- `strategy_enum` → `Strategy`: Parsed strategy enum value.

### `WorkerRecord`

Dataclass recording a worker's lifecycle during simulation.

| Field | Type | Description |
|---|---|---|
| `worker_id` | `str` | Unique worker identifier |
| `parent_id` | `str \| None` | Parent worker (None for root) |
| `depth` | `int` | Replication depth (0 = root) |
| `tasks_completed` | `int` | Number of tasks completed |
| `replications_attempted` | `int` | Replication attempts made |
| `replications_succeeded` | `int` | Successful replications |
| `replications_denied` | `int` | Denied replication attempts |

### `SimulationReport`

Dataclass containing full simulation results.

| Field | Type | Description |
|---|---|---|
| `config` | `ScenarioConfig` | Configuration used for this run |
| `workers` | `Dict[str, WorkerRecord]` | All workers by ID |
| `timeline` | `List[Dict]` | Chronological event log |
| `audit_events` | `List[Dict]` | Security audit events |
| `total_tasks` | `int` | Total tasks completed across all workers |
| `total_replications_attempted` | `int` | Total replication attempts |
| `total_replications_succeeded` | `int` | Successful replications |
| `total_replications_denied` | `int` | Denied replications |
| `duration_ms` | `float` | Wall-clock duration in milliseconds |

**Methods:**

- `render_tree() → str`: ASCII tree visualization of the worker hierarchy.
- `render_timeline() → str`: Formatted event timeline.
- `render_summary() → str`: Summary statistics with risk indicators.
- `render() → str`: Full report (summary + tree + timeline).
- `to_dict() → Dict`: JSON-serializable dictionary.

### `Simulator`

Main simulation runner.

```python
from replication.simulator import Simulator, ScenarioConfig

config = ScenarioConfig(max_depth=3, max_replicas=10, strategy="conservative")
sim = Simulator(config)
report = sim.run()
print(report.render())
```

**Constructor:**

- `Simulator(config: ScenarioConfig | None = None)` — uses default config when None.

**Methods:**

- `run() → SimulationReport`: Execute the simulation and return results.

## Built-in Presets

Access via `PRESETS` dict:

| Preset | Strategy | Depth | Replicas | Description |
|---|---|---|---|---|
| `minimal` | conservative | 1 | 3 | Minimal safe configuration |
| `balanced` | random (p=0.6) | 3 | 10 | Balanced risk/reward |
| `stress` | greedy | 5 | 50 | Stress test with no cooldown |
| `chain` | chain | 8 | 10 | Linear replication chain |
| `burst` | burst | 1 | 20 | Wide burst from root |

## CLI

```bash
python -m replication.simulator                          # default scenario
python -m replication.simulator --strategy greedy        # aggressive replication
python -m replication.simulator --max-depth 5 --max-replicas 20
python -m replication.simulator --scenario balanced      # built-in preset
python -m replication.simulator --json                   # JSON output
```
