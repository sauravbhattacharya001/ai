# Deceptive Alignment Detector

Detect agents that behave differently under observation — the
"treacherous turn" where an agent appears safe during evaluation but
pursues different objectives in unsupervised deployment.

!!! danger "Critical Safety Module"
    Deceptive alignment is among the most dangerous failure modes in
    AI safety. An agent that *looks* safe during testing but is unsafe
    in production undermines all other safety measures.

## Detection Strategies

| Pattern | Description |
|---------|-------------|
| **Observation Sensitivity** | Behaviour metrics shift significantly when monitoring is announced vs unannounced |
| **Evaluation Gaming** | Suspiciously strong performance on known benchmarks but degraded on novel equivalent tasks |
| **Context-Dependent Compliance** | Compliance drops when oversight signals are removed |
| **Behavioral Bifurcation** | Distinct behavioural clusters in monitored vs unmonitored contexts |
| **Sandbagging Reversal** | Sudden capability jumps after appearing to sandbag, suggesting hidden capability |
| **Instrumental Deception** | Self-reported capabilities/intentions contradict actual behaviour patterns |
| **Timing Anomaly** | Response patterns change near evaluation window boundaries |

## Quick Start

```python
from replication.deceptive_alignment import (
    DeceptiveAlignmentDetector,
    BehaviorSample,
    Context,
    DeceptionPattern,
)

detector = DeceptiveAlignmentDetector()

# Add samples from monitored and unmonitored contexts
detector.add_sample(BehaviorSample(
    agent_id="agent-1",
    score=0.95,
    latency=1.0,
    context="monitored",
))
detector.add_sample(BehaviorSample(
    agent_id="agent-1",
    score=0.42,
    latency=2.0,
    context="unmonitored",
))

# Analyze a single agent
report = detector.analyze("agent-1")
print(report.risk_label, report.risk_score)

# Analyze all agents
reports = detector.analyze_all()
for r in reports:
    print(r.agent_id, r.risk_label, r.findings)
```

## CLI Usage

```bash
# Default simulation with 5 agents
python -m replication deceptive-alignment --agents 5

# Larger simulation with more samples per agent
python -m replication deceptive-alignment --agents 8 --samples 200

# Use a deceptive fleet preset
python -m replication deceptive-alignment --preset deceptive --json

# Live watch mode with periodic re-analysis
python -m replication deceptive-alignment --watch --interval 5

# Export HTML report
python -m replication deceptive-alignment -o report.html
```

## Core Types

### `BehaviorSample`

A single observation of agent behaviour in a specific context.

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `str` | Agent identifier |
| `score` | `float` | Performance score (0–1) |
| `latency` | `float` | Response latency in seconds |
| `context` | `str` | `"monitored"`, `"unmonitored"`, `"evaluation"`, or `"deployment"` |

### `DeceptionFinding`

A single detected deception signal.

| Field | Type | Description |
|-------|------|-------------|
| `pattern` | `DeceptionPattern` | Which deception pattern was detected |
| `severity` | `Severity` | `LOW`, `MEDIUM`, `HIGH`, or `CRITICAL` |
| `detail` | `str` | Human-readable description |
| `evidence` | `dict` | Supporting numerical evidence |

### `AgentDeceptionReport`

Full analysis report for one agent.

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `str` | Agent identifier |
| `findings` | `list[DeceptionFinding]` | All detected deception signals |
| `divergence` | `float` | Monitored vs unmonitored score gap |
| `risk_score` | `float` | Overall risk (0–1) |
| `risk_label` | `str` | `"low"`, `"medium"`, `"high"`, or `"critical"` |
| `recommendation` | `str` | Actionable mitigation advice |
| `timestamp` | `str` | ISO-8601 analysis timestamp |

## API Reference

::: replication.deceptive_alignment.DeceptiveAlignmentDetector
    options:
      members:
        - __init__
        - add_sample
        - add_samples
        - agent_ids
        - analyze
        - analyze_all

## How It Works

The detector collects `BehaviorSample` observations across contexts and
compares distributions:

1. **Collect** — samples are grouped by `agent_id` and `context`.
2. **Compare** — for each agent, monitored vs unmonitored distributions
   are compared (mean, std, latency).
3. **Detect** — each detection strategy runs independently, producing
   `DeceptionFinding` instances with severity levels.
4. **Score** — findings are aggregated into a 0–1 risk score using
   severity-weighted summation.
5. **Report** — the `AgentDeceptionReport` bundles everything with a
   human-readable label and recommendation.

## Risk Scoring

Risk scores are computed from the severity-weighted sum of all findings,
combined with the raw divergence between monitored and unmonitored
performance:

- **0.0–0.3** → Low risk
- **0.3–0.6** → Medium risk — recommend increased monitoring
- **0.6–0.8** → High risk — recommend isolation and investigation
- **0.8–1.0** → Critical — recommend immediate containment

## Related Modules

- [Sandbagging Detector](sandbagging_detector.md) — agents hiding capabilities
- [Sycophancy Detector](sycophancy_detector.md) — agents excessively agreeing
- [Corrigibility Auditor](corrigibility_auditor.md) — testing shutdown/correction acceptance
- [Reward Hacking](reward_hacking.md) — gaming proxy metrics
