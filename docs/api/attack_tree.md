# Attack Tree Generator

Structured threat modelling for AI agent replication systems. Decomposes
top-level threat goals into sub-goals using AND/OR attack trees
(Schneier 1999), quantifies aggregate risk via bottom-up propagation,
and identifies the cheapest/easiest attack paths.

## Quick Start

```python
from replication.attack_tree import AttackTreeGenerator, TreeConfig

gen = AttackTreeGenerator()
result = gen.analyze()
print(result.render())

# Cheapest paths to a specific goal
for path in result.cheapest_paths("unauthorized_replication", n=3):
    print(f"  Cost: ${path.total_cost}  Path: {path.describe()}")
```

## Key Classes

- **`AttackTreeGenerator`** — Builds and analyses attack trees for all
  built-in threat goals. Supports configurable analysis depth and
  minimum-path selection.
- **`AttackNode`** — A single node in the tree (AND/OR). Carries cost,
  difficulty, likelihood, and optional child nodes.
- **`AttackPath`** — An end-to-end path from root goal to a leaf action.
  Computed cost is the sum of leaf costs for AND nodes, minimum for OR.
- **`TreeAnalysis`** — Per-goal aggregation: cheapest path, total attack
  surface, difficulty distribution, and recommended mitigations.
- **`TreeConfig`** — Configuration: which goals to analyse, top-N paths,
  rendering options.
- **`AttackTreeResult`** — Full result container with per-goal analyses
  and a rendered text report.

## Built-in Threat Goals

| Goal | Description |
|------|-------------|
| `UNAUTHORIZED_REPLICATION` | Bypass contract enforcement to create illegal replicas |
| `SANDBOX_ESCAPE` | Break out of resource constraints |
| `STATE_EXFILTRATION` | Extract sensitive data from the sandbox |
| `CONTROLLER_TAKEOVER` | Gain control of the central authority |
| `STEALTH_PERSISTENCE` | Survive after expiration/kill commands |

## Enums

- **`NodeType`** — `AND`, `OR`
- **`Difficulty`** — `TRIVIAL`, `EASY`, `MODERATE`, `HARD`, `EXPERT`
- **`ThreatGoal`** — One of the five built-in goals above

## CLI

```bash
python -m replication attack-tree                      # all goals
python -m replication attack-tree --goal escape        # single goal
python -m replication attack-tree --min-paths 3        # top 3 paths
python -m replication attack-tree --json               # JSON output
python -m replication attack-tree --annotate           # show cost/difficulty
```
