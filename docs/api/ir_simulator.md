# IR Simulator

Interactive incident response simulator — practice AI safety incident response through choose-your-own-adventure scenarios.

## CLI

```bash
# Generate the interactive HTML simulator
python -m replication ir-sim -o ir_simulator.html

# Generate a single scenario
python -m replication ir-sim --scenario rogue_agent

# List available scenarios
python -m replication ir-sim --list-scenarios

# Generate and open in browser
python -m replication ir-sim --open
```

## Built-in Scenarios

| ID | Name | Category | Difficulty |
|----|------|----------|------------|
| `rogue_agent` | Rogue Agent Breakout | Containment | Medium |
| `data_poisoning` | Silent Data Poisoning | Integrity | Hard |
| `prompt_injection` | Prompt Injection Chain | Access Control | Easy |

## Features

- **Branching narratives** — every decision leads to different outcomes
- **Impact scoring** — choices carry positive or negative safety impact points
- **Decision trail** — review your choices and their consequences
- **Grade system** — endings are graded A+ through F based on response quality
- **Self-contained HTML** — no external dependencies, works offline

## Programmatic Use

```python
from replication.ir_simulator import generate_html, BUILTIN_SCENARIOS

# Generate HTML for all scenarios
html = generate_html()

# Generate for specific scenarios
html = generate_html(scenarios=["rogue_agent", "data_poisoning"])

# Access scenario data
from replication.ir_simulator import _scenario_rogue_agent
scenario = _scenario_rogue_agent()
print(f"{scenario.name}: {len(scenario.nodes)} decision nodes")
```
