# Timeline Reconstructor

Builds a unified chronological event timeline from multiple analysis modules
(simulation, drift detection, compliance auditing, threat simulation, and
forensic analysis) for post-incident forensic review.

## Quick Start

```bash
# Full timeline
python -m replication timeline

# Filter by severity
python -m replication timeline --severity high

# Only drift events
python -m replication timeline --source drift

# Text search
python -m replication timeline --search "escape"

# Last N events
python -m replication timeline --last 20

# JSON output
python -m replication timeline --json

# Export to file
python -m replication timeline --export timeline.json
```

## Programmatic Usage

```python
from replication.timeline import TimelineReconstructor, TimelineConfig

reconstructor = TimelineReconstructor()
timeline = reconstructor.build()
print(timeline.render())

# With filters
config = TimelineConfig(
    strategy="greedy",
    severity_filter=EventSeverity.HIGH,
    include_sources=[EventSource.THREAT],
)
filtered = reconstructor.build(config)
for event in filtered.events:
    print(f"[{event.timestamp:.2f}s] {event.source.value}: {event.summary}")
```

## Event Sources

| Source       | Description                              |
|-------------|------------------------------------------|
| `simulation`| Replication simulation run events        |
| `drift`     | Behavioral drift detection alerts        |
| `compliance`| Compliance audit findings                |
| `threat`    | Threat scenario simulation results       |
| `forensics` | Near misses, escalation phases, decisions|

## Data Model

- **TimelineEvent** — single event with timestamp, source, severity, category, summary, details
- **TimelineSpan** — contiguous group of 3+ events from same source
- **TimelineStats** — aggregate counts by source/severity/category
- **Timeline** — complete result with events, stats, and render/export support

## CLI Options

| Flag             | Description                          |
|-----------------|--------------------------------------|
| `--strategy`    | Simulation preset (default: balanced)|
| `--max-depth`   | Max simulation depth (default: 4)    |
| `--drift-windows`| Drift detection windows (default: 5)|
| `--severity`    | Min severity filter                  |
| `--source`      | Filter to single event source        |
| `--category`    | Filter to single event category      |
| `--search`      | Text search in summaries/details     |
| `--last N`      | Show only last N events              |
| `--json`        | JSON output                          |
| `--export FILE` | Save timeline to JSON file           |
