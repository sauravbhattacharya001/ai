# Covert Channel Detector

Detects hidden communication channels between agents. Monitors
inter-agent message exchanges for steganographic channels, timing-based
signalling, encoded payloads, and protocol deviations.

## Quick Start

```python
from replication.covert_channels import (
    CovertChannelDetector, AgentMessage, DetectorConfig,
)

detector = CovertChannelDetector()

messages = [
    AgentMessage(sender="agent-1", receiver="agent-2",
                 content="Status update: all normal",
                 timestamp=1000.0),
    AgentMessage(sender="agent-1", receiver="agent-2",
                 content="SGVsbG8gV29ybGQ=",  # base64
                 timestamp=1001.0),
]

report = detector.analyze(messages)
for signal in report.signals:
    print(f"  [{signal.severity.value}] {signal.channel_type.value}: {signal.description}")
```

## Key Classes

- **`CovertChannelDetector`** — Analyses message exchanges across 5
  detection vectors: content, timing, protocol, frequency, and metadata.
- **`AgentMessage`** — A message between agents: sender, receiver,
  content, timestamp, optional metadata.
- **`CovertSignal`** — A detected covert channel signal with channel
  type, severity, confidence, and evidence.
- **`PairProfile`** — Per-agent-pair communication profile with baseline
  statistics and anomaly indicators.
- **`ChannelReport`** — Full analysis with all signals, pair profiles,
  and fleet-level risk assessment.
- **`DetectorConfig`** — Entropy thresholds, timing analysis parameters,
  n-gram sizes, minimum message counts.

## Detection Vectors

| Vector | What it catches |
|--------|-----------------|
| **Content analysis** | Entropy anomalies, base64 fragments, unusual padding |
| **Timing analysis** | Inter-message intervals encoding bits (Morse-like) |
| **Protocol deviation** | Valid but unusual field combinations or values |
| **Frequency analysis** | Repeated n-gram patterns diverging from baselines |
| **Metadata leakage** | Information encoded in optional fields or ordering |

## Channel Types

- **`STEGANOGRAPHIC`** — Information hidden within normal-looking content
- **`TIMING`** — Bits encoded in message intervals
- **`PROTOCOL`** — Exploiting protocol fields for side-channel data
- **`FREQUENCY`** — Statistical patterns in token usage
- **`METADATA`** — Data hidden in message metadata or field ordering
