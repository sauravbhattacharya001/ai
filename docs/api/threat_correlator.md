# Threat Correlator

Cross-module signal correlation for compound threat detection. Individual
detectors (drift, compliance, behavior profiler, escalation, kill chain,
canary, covert channels) each catch specific anomalies. Real attacks often
span multiple modules simultaneously — this module correlates signals to
surface multi-signal patterns that no single detector flags as critical.

## Quick Start

```python
from replication.threat_correlator import (
    ThreatCorrelator, Signal, SignalSource, SignalSeverity,
)

correlator = ThreatCorrelator()

signals = [
    Signal(source=SignalSource.DRIFT, severity=SignalSeverity.HIGH,
           agent_id="agent-1", timestamp=1000.0,
           description="escape_rate spike z=4.2"),
    Signal(source=SignalSource.ESCALATION, severity=SignalSeverity.CRITICAL,
           agent_id="agent-1", timestamp=1002.0,
           description="privilege escalation attempt"),
    Signal(source=SignalSource.CANARY, severity=SignalSeverity.HIGH,
           agent_id="agent-1", timestamp=1005.0,
           description="canary token transmitted"),
]

report = correlator.analyze(signals)
for threat in report.compound_threats:
    print(f"  [{threat.level.value}] {threat.rule_name}: {threat.description}")
```

## Key Classes

- **`ThreatCorrelator`** — Ingests signals from multiple detectors and
  applies correlation rules to surface compound threats.
- **`Signal`** — A normalised detection event from any module: source,
  severity, agent, timestamp, description.
- **`CorrelationRule`** — A named pattern that matches when N signals
  from specified sources co-occur within a time window for the same agent.
- **`CompoundThreat`** — A matched rule with contributing signals,
  combined severity, risk score, and recommended response actions.
- **`AgentRisk`** — Per-agent risk summary from correlation analysis.
- **`CoverageGap`** — Identifies which module combinations lack rules.
- **`CorrelatorConfig`** — Default time windows, minimum signal counts,
  custom rule definitions.
- **`CorrelationReport`** — Fleet-level summary with all compound
  threats, agent risk rankings, and coverage analysis.

## Signal Sources

`DRIFT`, `COMPLIANCE`, `BEHAVIOR`, `ESCALATION`, `KILLCHAIN`, `CANARY`,
`COVERT_CHANNEL`, `PROMPT_INJECTION`, `STEGANOGRAPHY`, `DECEPTION`,
`TRUST`, `HONEYPOT`

## Severity Levels

`INFO`, `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`

## Threat Levels

`ELEVATED`, `HIGH`, `SEVERE`, `CRITICAL`
