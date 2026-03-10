# Resource Dependency Analyzer  --  model and analyze inter-agent resource

Resource Dependency Analyzer — model and analyze inter-agent resource


**Module:** `replication.dependency_graph`


## Quick Start

```python
from replication.dependency_graph import ResourceKind

instance = ResourceKind()
```


## Enums

### `ResourceKind`

- `DATABASE`
- `SERVICE`
- `QUEUE`
- `FILESYSTEM`
- `COMPUTE`
- `EXTERNAL`
- `CACHE`

### `Criticality`

- `REQUIRED`
- `DEGRADED`
- `OPTIONAL`


## Classes

### `ResourceKind`

Classification of infrastructure resources.

### `Criticality`

How critical a dependency is for the dependent.

### `Resource`

A shared resource in the system.

| Method | Description |
|--------|-------------|
| `to_dict()` |  |

### `AgentNode`

An agent that depends on resources.

| Method | Description |
|--------|-------------|
| `required_resources()` |  |
| `to_dict()` |  |

### `CascadeStep`

One step in a failure cascade.

| Method | Description |
|--------|-------------|
| `to_dict()` |  |

### `CascadeResult`

Full cascade simulation result.

| Method | Description |
|--------|-------------|
| `to_dict()` |  |
| `render()` |  |

### `DepAnalysis`

Complete dependency analysis result.

| Method | Description |
|--------|-------------|
| `to_dict()` |  |
| `render()` |  |

### `DependencyGraph`

Directed dependency graph between agents and resources.

| Method | Description |
|--------|-------------|
| `__init__()` |  |
| `add_resource()` | Add a resource node.  Returns self for chaining. |
| `add_agent()` | Add an agent node with its resource dependencies. |
| `analyze()` | Run full dependency analysis. |
| `simulate_failure()` | Simulate cascading failure when *resource_name* goes down. |
| `to_dot()` | Export dependency graph as Graphviz DOT. |
| `to_dict()` |  |


## Functions

| Function | Description |
|----------|-------------|
| `generate_scenario()` | Generate a sample dependency graph for analysis. |
| `main()` |  |


## CLI

```bash
python -m replication dependency_graph --help
```
