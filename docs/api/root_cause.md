# root_cause

Root Cause Analyzer — structured root cause analysis for AI safety incidents.

Provides three complementary methods: **5 Whys** iterative causal chains,
**Fishbone (Ishikawa)** categorized cause analysis, and **Fault Tree**
Boolean logic decomposition with minimal cut-set analysis.

## Quick start

```bash
# Full analysis (all three methods)
python -m replication root-cause --incident "Agent escaped sandbox" --severity critical

# Single method
python -m replication root-cause -i "Kill switch failed" -m fault-tree

# HTML report
python -m replication root-cause -i "Drift detected" -s high -f html -o rca.html

# JSON for programmatic use
python -m replication root-cause -i "Unauthorized replication" -f json
```

## Methods

| Method | Flag | Description |
|--------|------|-------------|
| 5 Whys | `5whys` | Iterative "why?" chain from symptom to root cause |
| Fishbone | `fishbone` | Ishikawa diagram across 6 AI safety categories |
| Fault Tree | `fault-tree` | Boolean logic tree with minimal cut-set analysis |
| All | `all` (default) | Run all three methods |

## Fishbone categories

- **Policy** — Safety policies, contracts, governance
- **Monitoring** — Observability, alerting, detection
- **Architecture** — System design, boundaries, isolation
- **Human** — Operator decisions, training, procedures
- **Environment** — Infrastructure, dependencies, external conditions
- **Data** — Training data, inputs, information flow

## Programmatic API

```python
from replication.root_cause import full_analysis, format_html

result = full_analysis("Agent escaped sandbox", "critical")
html = format_html(result)
```
