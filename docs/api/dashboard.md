# Dashboard

**Self-contained HTML report generator for simulation runs.**

The `dashboard` module produces single-file HTML dashboards with embedded CSS and JavaScript — no external dependencies required. Open the output file in any browser to explore simulation results interactively.

## Quick Start

### CLI

```bash
# Single run dashboard
python -m replication dashboard --strategy greedy --depth 5

# Compare two strategies
python -m replication dashboard --compare greedy random --steps 100

# Save to file
python -m replication dashboard -o report.html
```

### Python API

```python
from pathlib import Path
from replication.dashboard import DashboardGenerator, DashboardConfig
from replication.simulator import Simulator, ScenarioConfig

# Run a simulation
report = Simulator(ScenarioConfig(strategy="greedy")).run()

# Generate dashboard
gen = DashboardGenerator()
html = gen.single_report(report, title="Greedy Run Analysis")
Path("report.html").write_text(html)
```

## Configuration

```python
from replication.dashboard import DashboardConfig

config = DashboardConfig(
    title="My Safety Dashboard",  # Page title
    theme="dark",                  # "light" or "dark"
    show_timeline=True,            # Include event timeline
    show_tree=True,                # Include replication tree visualization
    show_audit=True,               # Include audit event log
    max_timeline_events=200,       # Cap timeline entries
    max_audit_events=100,          # Cap audit log entries
)
```

## Classes

### `DashboardConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `title` | `str` | `"Replication Safety Dashboard"` | Dashboard page title |
| `theme` | `str` | `"light"` | Color theme (`"light"` or `"dark"`) |
| `show_timeline` | `bool` | `True` | Show event timeline section |
| `show_tree` | `bool` | `True` | Show replication tree section |
| `show_audit` | `bool` | `True` | Show audit event log section |
| `max_timeline_events` | `int` | `200` | Maximum timeline events to render |
| `max_audit_events` | `int` | `100` | Maximum audit events to render |

### `DashboardGenerator`

The main entry point for generating dashboards.

#### `single_report(report, title=None, config=None) → str`

Generate an HTML dashboard for a single simulation run.

| Parameter | Type | Description |
|-----------|------|-------------|
| `report` | `SimulationReport` | Simulation results |
| `title` | `str \| None` | Override dashboard title |
| `config` | `DashboardConfig \| None` | Custom configuration |

Returns the complete HTML string.

#### `compare_reports(reports, titles=None, config=None) → str`

Generate a side-by-side comparison dashboard for multiple simulation runs.

| Parameter | Type | Description |
|-----------|------|-------------|
| `reports` | `list[SimulationReport]` | Two or more reports to compare |
| `titles` | `list[str] \| None` | Labels for each report |
| `config` | `DashboardConfig \| None` | Custom configuration |

Returns the complete HTML string with tabbed comparison view.

#### `get_report_data(report) → dict`

Extract structured data from a simulation report for rendering.

## Comparison Example

```python
from replication.dashboard import DashboardGenerator
from replication.simulator import Simulator, ScenarioConfig

greedy = Simulator(ScenarioConfig(strategy="greedy", max_steps=100)).run()
random = Simulator(ScenarioConfig(strategy="random", max_steps=100)).run()

gen = DashboardGenerator()
html = gen.compare_reports(
    [greedy, random],
    titles=["Greedy Strategy", "Random Strategy"],
)
Path("comparison.html").write_text(html)
```

## Dashboard Sections

The generated HTML includes:

- **Summary Cards** — Worker count, replication depth, total events, safety score
- **Timeline** — Chronological event log with color-coded severity
- **Replication Tree** — Visual hierarchy of parent → child worker relationships
- **Audit Trail** — Structured log of replication decisions, denials, and kill events
- **Comparison View** — Tabbed side-by-side metrics when comparing multiple runs
