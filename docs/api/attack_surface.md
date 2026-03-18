# Attack Surface

Interactive HTML attack surface visualizer for replication sandbox threat analysis.

## Overview

Generates a self-contained HTML page rendering the sandbox's attack trees as an interactive sunburst diagram. Each ring represents a depth level; nodes are color-coded by risk (difficulty × likelihood).

## Features

- **Sunburst layout** — hierarchical radial chart; click to zoom into sub-trees, click center to zoom out
- **Risk heatmap coloring** — green (low) → red (critical); mitigated nodes shown in blue
- **Hover details** — cost, likelihood, difficulty, mitigation status
- **Goal selector** — switch between the 5 threat goals
- **Coverage ring** — animated progress showing mitigation coverage
- **Summary stats** — total paths, average risk, cheapest path, leaf count
- **Search** — filter nodes by label
- **Export** — save current view as PNG

## CLI Usage

```bash
# Generate with default output (attack_surface.html)
python -m replication attack-surface

# Custom output path
python -m replication attack-surface -o surface.html

# Generate and open in browser
python -m replication attack-surface --open
```

## API Usage

```python
from replication.attack_surface import generate_surface
from pathlib import Path

# Generate HTML string
html = generate_surface()

# Write to file
Path("attack_surface.html").write_text(html)
```

## Interpreting the Visualization

### Color Coding

| Color | Meaning |
|-------|---------|
| Green | Low risk — high difficulty, low likelihood |
| Yellow | Medium risk |
| Red/Orange | High/critical risk — easy to exploit, likely |
| Blue | Mitigated — countermeasure in place |

### Sunburst Rings

- **Inner ring** — top-level threat goals (e.g., data exfiltration, privilege escalation)
- **Outer rings** — progressively more specific attack steps
- **Ring width** — proportional to relative risk contribution

Click any segment to zoom into that sub-tree. Click the center to zoom back out.
