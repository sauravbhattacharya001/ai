# Topology Analyzer API

Structural analysis of worker replication trees — depth distribution, branching
factor, balance metrics, hotspot detection, subtree risk, and pathological
pattern recognition.

## Enums

### `RiskLevel`

Overall topology risk classification.

| Value | Description |
|---|---|
| `LOW` | Normal replication structure |
| `MODERATE` | Some structural concerns |
| `HIGH` | Potentially dangerous topology |
| `CRITICAL` | Pathological structure detected |

### `PathologicalPattern`

Known dangerous tree shapes.

| Value | Description |
|---|---|
| `DEEP_CHAIN` | Excessively deep linear chain |
| `WIDE_FAN` | Single node with extreme fan-out |
| `HOTSPOT` | Concentration of replications in one subtree |
| `UNBALANCED` | Highly asymmetric tree |
| `RUNAWAY` | Exponential growth pattern |
| `CLUSTER_BOMB` | Burst replication at multiple levels |

## Data Classes

### `NodeMetrics`

Per-node structural metrics.

| Field | Type | Description |
|---|---|---|
| `worker_id` | `str` | Worker identifier |
| `depth` | `int` | Depth in the tree |
| `subtree_size` | `int` | Total descendants (including self) |
| `children_count` | `int` | Direct child count |
| `branching_factor` | `float` | Average branching in this subtree |

### `SubtreeRisk`

Risk assessment for a specific subtree.

| Field | Type | Description |
|---|---|---|
| `root_id` | `str` | Subtree root worker ID |
| `size` | `int` | Number of workers in subtree |
| `max_depth` | `int` | Deepest level in subtree |
| `risk_score` | `float` | Computed risk (0.0–1.0) |
| `patterns` | `List[PathologicalPattern]` | Detected patterns |

### `TopologyReport`

Complete analysis output.

| Field | Type | Description |
|---|---|---|
| `total_workers` | `int` | Total worker count |
| `max_depth` | `int` | Maximum tree depth |
| `mean_branching_factor` | `float` | Average branching factor |
| `balance_score` | `float` | Tree balance (0.0=unbalanced, 1.0=balanced) |
| `risk_level` | `RiskLevel` | Overall risk classification |
| `risk_score` | `float` | Numeric risk score (0.0–1.0) |
| `patterns` | `List[PathologicalPattern]` | Detected pathological patterns |
| `warnings` | `List[str]` | Human-readable warning messages |
| `node_metrics` | `List[NodeMetrics]` | Per-node metrics |
| `subtree_risks` | `List[SubtreeRisk]` | Per-subtree risk assessments |

**Methods:**

- `render_tree() → str`: ASCII tree with annotated metrics.
- `summary() → str`: Formatted summary string.
- `to_dict() → Dict`: JSON-serializable output.
- `to_json(indent=2) → str`: Pretty-printed JSON.

## TopologyAnalyzer

```python
from replication.topology import TopologyAnalyzer

# Build from controller
analyzer = TopologyAnalyzer.from_controller(controller)
report = analyzer.analyze()

print(report.summary())
print(report.render_tree())
print(f"Risk: {report.risk_level.value}")
for warning in report.warnings:
    print(f"  ⚠ {warning}")
```

### Constructor

```python
TopologyAnalyzer(
    parent_map: Dict[str, str | None],
    depth_map: Dict[str, int],
    logger: StructuredLogger | None = None,
)
```

| Parameter | Description |
|---|---|
| `parent_map` | Worker ID → parent ID mapping (None for roots) |
| `depth_map` | Worker ID → depth mapping |
| `logger` | Optional structured logger for audit events |

### Class Methods

- `from_controller(controller, logger=None) → TopologyAnalyzer`:
  Build from a live `Controller` instance using its worker registry.

### Methods

- `analyze() → TopologyReport`: Run the full topology analysis.

### Internal Methods

These are called by `analyze()` and compute individual metrics:

| Method | Returns | Description |
|---|---|---|
| `_compute_subtree_size(wid)` | `int` | Recursive subtree size |
| `_compute_subtree_depth(wid)` | `int` | Max depth in subtree |
| `_compute_branching_factor(wid)` | `float` | Average branching |
| `_compute_balance_score(metrics)` | `float` | Tree balance 0.0–1.0 |
| `_detect_patterns(metrics, subtrees)` | `List[PathologicalPattern]` | Pattern detection |
| `_compute_risk_score(patterns, metrics, balance)` | `float` | Overall risk 0.0–1.0 |
| `_classify_risk(score)` | `RiskLevel` | Risk classification |
| `_generate_warnings(patterns, metrics, subtrees)` | `List[str]` | Warning messages |
