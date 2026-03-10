# Interactive HTML report generator for simulation and threat analysis

Interactive HTML report generator for simulation and threat analysis.


**Module:** `replication.reporter`


## Quick Start

```python
from replication.reporter import HTMLReporter

instance = HTMLReporter()
```


## Classes

### `HTMLReporter`

Generate self-contained interactive HTML reports.

| Method | Description |
|--------|-------------|
| `__init__()` |  |
| `simulation_report()` | Generate an interactive HTML report from a SimulationReport. |
| `threat_report()` | Generate an interactive HTML report from a ThreatReport. |
| `comparison_report()` | Generate an interactive HTML report from a ComparisonResult. |
| `combined_report()` | Generate a combined HTML report with tabs for each section. |
| `save()` | Save HTML report to a file. Returns the absolute path. |


## Functions

| Function | Description |
|----------|-------------|
| `main()` | CLI entry point for the HTML report generator. |


## CLI

```bash
python -m replication reporter --help
```
