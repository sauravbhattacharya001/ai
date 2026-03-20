# Alert Fatigue Detector

Analyze safety alert streams to detect operator fatigue — when too many alerts lead to desensitization and missed critical events.

## Fatigue Indicators

| Indicator | What it measures |
|---|---|
| **Volume Overload** | Alerts/hour exceeding sustainable threshold |
| **Repetition Noise** | Same alert type firing repeatedly |
| **Severity Inflation** | Too many critical/high alerts diluting urgency |
| **Ack Lag Growth** | Acknowledgment time trending upward |
| **Suppression Rate** | Percentage of alerts auto-suppressed |
| **Off-Hours Load** | Alerts during nights/weekends (burnout risk) |

## Fatigue Levels

- **Healthy** (0-19): Alert hygiene is good
- **Mild** (20-39): Early signs of noise
- **Moderate** (40-59): Action recommended
- **Severe** (60-79): Operator burnout likely
- **Critical** (80-100): Immediate intervention needed

## CLI Usage

```bash
# Simulate and analyze 72 hours of alerts
python -m replication fatigue-detect simulate --hours 72 --rate 40

# Analyze alerts from file
python -m replication fatigue-detect analyze --alerts alerts.json

# Get recommendations
python -m replication fatigue-detect recommend --alerts alerts.json

# Generate HTML report
python -m replication fatigue-detect report --output fatigue.html
```

## Programmatic API

```python
from replication.fatigue_detector import FatigueDetector, FatigueThresholds, AlertEvent

detector = FatigueDetector(thresholds=FatigueThresholds(max_alerts_per_hour=30))
detector.ingest(alert_events)
result = detector.analyze()

print(f"Score: {result.score}/100 — {result.level}")
for ind in result.indicators:
    print(f"  {ind.name}: {ind.score} — {ind.detail}")
for rec in result.recommendations:
    print(f"  → {rec}")
```

## Simulation

The built-in simulator generates realistic alert streams with fatigue onset:

```python
from replication.fatigue_detector import simulate_alerts

events = simulate_alerts(hours=72, base_rate=20, fatigue_onset_hour=24, seed=42)
# After fatigue_onset_hour: ack delays grow, suppression increases
```

## HTML Reports

Generate self-contained HTML reports with score cards, indicator bars, recommendations, and statistics.
