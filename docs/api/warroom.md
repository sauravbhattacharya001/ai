# War Room Dashboard

Interactive incident command center for monitoring AI replication events in real-time.

## Features

- **Live Event Feed** — streaming replication events with severity color-coding
- **Fleet Status Grid** — visual worker health map (healthy/warning/critical/dead)
- **Resource Gauges** — CPU, Memory, Network, Disk utilization rings
- **Kill Switch** — big red button to halt all replication fleet-wide
- **Contract Violation Tracker** — ranked violation types with counts
- **Alert Distribution Chart** — severity breakdown (INFO/WARN/HIGH/CRIT)
- **Incident Timeline** — scrolling worker count graph over time
- **Summary Stats** — active workers, replications, violations, kills, max depth, fleet health

## Usage

### CLI

```bash
python -m replication warroom
python -m replication warroom -o warroom.html
python -m replication warroom --open
```

### Python API

```python
from replication.warroom import generate_warroom
from pathlib import Path

html = generate_warroom()
Path("warroom.html").write_text(html)
```

## How It Works

The dashboard generates a self-contained HTML page with a built-in simulation engine. Once opened in a browser:

1. A fleet of 16 workers initializes with random depth levels
2. Every 1.5 seconds, a simulation tick fires random events (spawns, violations, kills, recoveries)
3. All panels update in real-time — no server needed
4. Click the Kill Switch to halt all replication and put the fleet in safe mode

## Dark Ops Aesthetic

The dashboard uses a dark theme inspired by real-world NOC/SOC command centers, with:

- Color-coded severity indicators (green → yellow → orange → red)
- Pulsing live indicator
- Real-time clock and uptime counter
- Responsive layout for different screen sizes
