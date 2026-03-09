# Deception Detector

Detects deceptive agent behaviors by analysing communications and actions
for inconsistency, omission, misdirection, sycophancy, sandbagging,
fabrication, strategic ambiguity, and selective disclosure.

## Quick Start

```python
from replication.deception_detector import (
    DeceptionDetector, Statement, ActionRecord, DetectorConfig,
)

detector = DeceptionDetector()

detector.add_statement(Statement(
    agent_id="agent-1",
    text="I cannot access the file system",
    timestamp=1000.0,
))
detector.add_statement(Statement(
    agent_id="agent-1",
    text="I have read the configuration file",
    timestamp=1200.0,
))

report = detector.analyze("agent-1")
for signal in report.signals:
    print(f"  [{signal.severity.value}] {signal.category.value}: {signal.description}")
print(f"Trust score: {report.trust_score:.2f}")
```

## Key Classes

- **`DeceptionDetector`** — Core detector. Ingests statements and action
  records, builds agent profiles, and runs deception pattern analysis.
- **`Statement`** — A timestamped agent claim or assertion with context.
- **`ActionRecord`** — An observed agent action with outcome and context,
  for cross-referencing against stated capabilities/intentions.
- **`DeceptionSignal`** — A scored detection of a specific deception
  pattern with category, severity, and evidence.
- **`AgentProfile`** — Accumulated deception profile with trust score,
  pattern history, and signal timeline.
- **`DeceptionReport`** — Per-agent analysis with all signals and trust
  assessment.
- **`FleetReport`** — Fleet-wide deception analysis with agent rankings.
- **`DetectorConfig`** — Thresholds, analysis window, severity weights.

## Deception Categories

| Category | Description |
|----------|-------------|
| `INCONSISTENCY` | Agent statements contradict each other |
| `OMISSION` | Agent withholds relevant information |
| `MISDIRECTION` | Agent deflects from sensitive topics |
| `SYCOPHANCY` | Agent over-agrees to gain trust |
| `SANDBAGGING` | Agent underperforms to hide capabilities |
| `FABRICATION` | Agent invents false information |
| `STRATEGIC_AMBIGUITY` | Agent uses vague language to avoid commitments |
| `SELECTIVE_DISCLOSURE` | Agent reveals only favorable information |
