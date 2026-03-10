# Bridge between PromptInjectionDetector and ThreatIntelFeed

Bridge between PromptInjectionDetector and ThreatIntelFeed.


**Module:** `replication.injection_intel_bridge`


## Quick Start

```python
from replication.injection_intel_bridge import InjectionAlert

instance = InjectionAlert()
```


## Classes

### `InjectionAlert`

An injection-specific alert with context.

| Method | Description |
|--------|-------------|
| `to_dict()` |  |

### `SharedPattern`

A discovered injection pattern shareable across detector instances.

| Method | Description |
|--------|-------------|
| `to_dict()` |  |

### `InjectionIntelBridge`

Bridges PromptInjectionDetector findings into the ThreatIntelFeed.

| Method | Description |
|--------|-------------|
| `__init__()` |  |
| `scan_and_ingest()` | Scan text for injections and auto-ingest findings as IOCs. |
| `scan_batch_and_ingest()` | Scan multiple messages, ingesting any injection findings. |
| `check_injection_alerts()` | Check for injection-specific alert conditions. |
| `share_pattern()` | Register a new injection pattern for sharing across instances. |
| `apply_shared_patterns()` | Apply all shared patterns to a target detector instance. |
| `get_shared_patterns()` | Get all shared patterns. |
| `get_correlated_campaigns()` | Identify potential attack campaigns by correlating injection IOCs |
| `get_stats()` | Get bridge statistics. |


## Functions

| Function | Description |
|----------|-------------|
| `create_bridge()` | Create a fully wired bridge with fresh detector and feed. |
| `main()` |  |
