# Safety Net Analyzer

Analyzes defense-in-depth coverage by mapping safety controls to threat categories. Identifies redundancies, single points of failure, and blind spots.

## CLI

```bash
# Full analysis with all controls
python -m replication safety-net

# Analyze specific controls only
python -m replication safety-net --controls drift,killchain,escalation

# Set minimum desired depth
python -m replication safety-net --min-depth 3

# JSON output
python -m replication safety-net --format json

# HTML report
python -m replication safety-net --format html -o safety_net.html

# List available controls
python -m replication safety-net --list-controls

# List threat categories
python -m replication safety-net --list-threats
```

## Programmatic

```python
from replication.safety_net import SafetyNetAnalyzer, analyze_safety_net

# Quick analysis
report = analyze_safety_net()
print(report.render())
print(f"Score: {report.coverage_score}/100")

# Custom analysis
analyzer = SafetyNetAnalyzer(
    controls=["drift", "killchain", "escalation", "canary"],
    min_depth=3,
)
report = analyzer.analyze()

# Check results
for blind_spot in report.blind_spots:
    print(f"UNCOVERED: {blind_spot.category}")

for spof in report.spofs:
    print(f"SPOF: {spof.category} → {spof.controls[0]}")
```

## Coverage Score

The score (0–100) reflects defense-in-depth quality:

| Depth | Points |
|-------|--------|
| 0 (blind spot) | 0% |
| 1 (SPOF) | 40% |
| 2 | 80% |
| 3+ | 100% |

## Threat Categories

- **reconnaissance** — Probing environment, testing boundaries
- **resource_abuse** — Hoarding compute, memory, storage
- **privilege_escalation** — Gaining broader permissions
- **lateral_movement** — Reaching other agents/services
- **defense_evasion** — Avoiding detection, stealth
- **command_control** — Establishing covert channels
- **data_exfiltration** — Unauthorized data extraction
- **self_replication** — Creating unauthorized copies
- **goal_drift** — Subtle deviation from objectives
- **deception** — Misrepresenting state or intentions
- **supply_chain** — Compromising dependencies
- **prompt_injection** — Manipulating via injected prompts
