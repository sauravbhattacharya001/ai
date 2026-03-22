# Runbook Generator

Generate structured incident-response runbooks from threat scenarios.

## Quick start

```python
from replication.runbook import RunbookGenerator, ThreatScenario

gen = RunbookGenerator()
scenario = ThreatScenario(
    name="Unauthorized Self-Replication",
    severity="critical",
    affected_systems=["orchestrator", "worker-pool"],
)
rb = gen.generate(scenario)
print(rb.to_markdown())
```

## CLI

```bash
# Generate from built-in template
python -m replication.runbook --template self-replication

# Custom threat
python -m replication.runbook --threat "model exfiltrating training data" --severity high

# List templates
python -m replication.runbook --list-templates

# Export as JSON
python -m replication.runbook --template kill-switch-evasion --json

# Save to file
python -m replication.runbook --template data-exfiltration --output runbook.md
```

## Built-in templates

| Template | Severity | Description |
|---|---|---|
| `self-replication` | critical | Unauthorized agent self-replication |
| `data-exfiltration` | high | Data exfiltration attempt |
| `goal-drift` | medium | Goal drift / misalignment |
| `prompt-injection` | high | Prompt injection attack |
| `resource-hoarding` | medium | Resource hoarding |
| `kill-switch-evasion` | critical | Kill switch evasion |

## Runbook sections

Each generated runbook includes:

- **Summary** — what happened and context
- **Indicators of Compromise** — what to look for
- **Triage Checklist** — initial assessment steps
- **Escalation Path** — when and whom to contact
- **Containment Actions** — how to stop the threat
- **Evidence Collection** — what to preserve
- **Recovery Procedure** — step-by-step restoration
- **Post-Incident Review** — lessons learned items

## Output formats

- **Markdown** (`--format markdown`) — wiki/docs friendly with checkboxes
- **JSON** (`--format json` or `--json`) — machine-readable
- **Text** (`--format text`) — plain text for terminals/printing
