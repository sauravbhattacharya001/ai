# Shadow AI Detector

::: replication.shadow_ai

## Overview

Detects unauthorized "shadow" AI deployments — models, endpoints, and
pipelines operating outside an organization's approved AI inventory and
safety controls.

## Quick start

```python
from replication.shadow_ai import (
    ShadowAIDetector, ScanPolicy, AIInventory,
    Observation, SignalType,
)

inventory = AIInventory(registered_models=["gpt-4"])
detector = ShadowAIDetector(inventory=inventory)
report = detector.scan([
    Observation(SignalType.NETWORK_TRAFFIC, "proxy",
                "POST https://api.anthropic.com/v1/messages"),
    Observation(SignalType.PROCESS_METADATA, "gpu-node",
                "vllm serve --model llama-3"),
])
for f in report.findings:
    print(f)
```

## CLI

```bash
python -m replication shadow-ai --demo
python -m replication shadow-ai --observations scan.json --models gpt-4 --json
```
