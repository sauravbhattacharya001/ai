# Agent Threat Intelligence Feed  --  aggregates detection signals into normalized IOCs

Agent Threat Intelligence Feed — aggregates detection signals into normalized IOCs.


**Module:** `replication.threat_intel`


## Quick Start

```python
from replication.threat_intel import IOCType

instance = IOCType()
```


## Enums

### `IOCType`

- `BEHAVIOR`
- `COMMUNICATION`
- `RESOURCE`
- `REPLICATION`
- `EVASION`
- `EXFILTRATION`
- `COORDINATION`

### `Severity`

- `INFO`
- `LOW`
- `MEDIUM`
- `HIGH`
- `CRITICAL`

### `AlertAction`

- `LOG`
- `NOTIFY`
- `QUARANTINE`
- `ESCALATE`


## Classes

### `IOCType`

Categories of indicators of compromise.

### `Severity`

Severity levels for IOCs.

| Method | Description |
|--------|-------------|
| `numeric()` |  |
| `from_score()` |  |

### `AlertAction`

Actions for alert rules.

### `IOC`

A single indicator of compromise.

| Method | Description |
|--------|-------------|
| `id()` |  |
| `to_dict()` |  |

### `CorrelationGroup`

A group of correlated IOCs.

### `AlertRule`

A rule that triggers when feed conditions are met.

| Method | Description |
|--------|-------------|
| `should_fire()` |  |
| `fire()` |  |

### `TrendPoint`

A single point in a trend time series.

### `FeedReport`

Complete analysis report from the threat intel feed.

| Method | Description |
|--------|-------------|
| `render()` |  |

### `FeedConfig`

Configuration for the threat intel feed.

### `ThreatIntelFeed`

Aggregates IOCs from detection modules into a queryable threat feed.

| Method | Description |
|--------|-------------|
| `__init__()` |  |
| `ingest()` | Ingest an IOC. Returns IOC id if accepted, None if deduplicated. |
| `ingest_batch()` | Ingest multiple IOCs. |
| `query()` | Query IOCs with filters. |
| `get()` | Get IOC by ID. |
| `count()` |  |
| `correlate()` | Find correlated IOCs based on shared correlation keys and time proximity. |
| `age_out()` | Remove IOCs older than max_age. Returns count removed. |
| `decay_scores()` | Apply exponential decay to IOC scores based on age. |
| `add_alert_rule()` |  |
| `check_alerts()` | Evaluate all alert rules. Returns list of fired alerts. |
| `rule_critical_spike()` | Alert when critical IOCs exceed threshold. |
| `rule_multi_agent_correlation()` | Alert when correlation groups span multiple agents. |
| `rule_volume_surge()` | Alert on IOC volume surge in a time window. |
| `compute_trends()` | Compute IOC trends over time windows. |
| `export_stix()` | Export feed as a STIX 2.1-inspired bundle. |
| `agent_risk_profile()` | Get risk profile for a specific agent. |
| `analyze()` | Run full analysis and return report. |
| `export_state()` | Export feed state for persistence. |
| `import_state()` | Import feed state. Returns count of IOCs loaded. |


## Functions

| Function | Description |
|----------|-------------|
| `main()` |  |


## CLI

```bash
python -m replication threat_intel --help
```
