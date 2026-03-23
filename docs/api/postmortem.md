# Postmortem Generator

Generate structured, blameless postmortem documents from safety incidents.

## Quick Start

```bash
python -m replication postmortem --incident "Agent escaped sandbox" --severity critical
```

## Options

| Flag | Description |
|------|-------------|
| `--incident, -i` | Short incident title (required) |
| `--severity, -s` | `low`, `medium`, `high`, `critical` (default: medium) |
| `--description, -d` | Detailed description |
| `--timeline, -t` | Timeline entries (space-separated quoted strings) |
| `--impact` | Impact statement (auto-generated if omitted) |
| `--detection` | How the incident was detected |
| `--responders, -r` | Responder names/teams |
| `--format, -f` | `text`, `markdown`, `html`, `json` (default: markdown) |
| `--output, -o` | Write to file instead of stdout |

## Examples

### Markdown report (default)

```bash
python -m replication postmortem -i "Drift detected in agent-7" -s medium
```

### HTML report with timeline

```bash
python -m replication postmortem \
  -i "Unauthorized replication attempt" \
  -s critical \
  -t "10:00 Alert fired" "10:05 On-call paged" "10:12 Contained" "10:30 Root cause identified" \
  -f html -o postmortem.html
```

### JSON for programmatic use

```bash
python -m replication postmortem -i "Policy violation" -s high -f json
```

## Programmatic API

```python
from replication.postmortem import Postmortem

pm = Postmortem(
    incident="Agent exceeded resource quota",
    severity="high",
    timeline=["09:00 Quota exceeded", "09:05 Alert triggered", "09:10 Agent throttled"],
)

print(pm.to_markdown())  # or to_text(), to_html(), to_json()
print(pm.to_dict())      # structured dict
```

## Features

- **Blameless by design** — contributing factors focus on systems, not people
- **Auto-populated sections** — severity-appropriate contributing factors and action items
- **Multiple formats** — text, markdown, HTML (with interactive checkboxes), JSON
- **Document IDs** — each postmortem gets a unique hash-based ID for tracking
- **Actionable output** — action items are checkboxes you can track to completion
