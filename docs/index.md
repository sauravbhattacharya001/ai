# AI Replication Sandbox

**A contract-enforced, sandboxed system for studying AI agent self-replication safety**

[![CI](https://github.com/sauravbhattacharya001/ai/actions/workflows/ci.yml/badge.svg)](https://github.com/sauravbhattacharya001/ai/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/sauravbhattacharya001/ai/blob/master/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)

---

## What is this?

As AI agents gain autonomy, controlling *how* and *when* they spawn copies of themselves becomes a critical safety challenge. The **AI Replication Sandbox** provides a testbed for designing and validating those controls.

It models a **replication-aware worker system** where autonomous agents can self-replicate under strict, auditable constraints — all without requiring Docker or Podman.

## Key Features

- :lock: **Replication Contracts** — Configurable policies with depth limits, quotas, cooldowns, and pluggable stop conditions
- :page_facing_up: **HMAC-Signed Manifests** — Tamper-proof worker identity with cryptographic verification
- :control_knobs: **Central Controller** — Registry, quota enforcement, kill switch, and audit trail
- :package: **Sandbox Orchestrator** — Simulated container isolation with CPU/RAM limits and network policy
- :bar_chart: **Structured Observability** — Lifecycle events, replication decisions, and resource metrics

## Quick Example

```python
from replication import (
    Controller, Worker, ReplicationContract,
    ResourceSpec, SandboxOrchestrator, StructuredLogger,
)

contract = ReplicationContract(max_depth=2, max_replicas=5, cooldown_seconds=1.0)
logger = StructuredLogger()
controller = Controller(contract, secret="my-secret", logger=logger)
orchestrator = SandboxOrchestrator(logger=logger)

resources = ResourceSpec(cpu_limit=0.5, memory_limit_mb=256)
manifest = controller.issue_manifest(
    parent_id=None, depth=0,
    state_snapshot={"task": "analyze"},
    resources=resources,
)

root = Worker(manifest, contract, controller, orchestrator, logger)
root.perform_task(lambda w: print(f"Worker {w.manifest.worker_id} running"))
child = root.maybe_replicate(reason="subtask", state_snapshot={"task": "sub-analyze"})

root.shutdown("complete")
if child:
    child.shutdown("complete")
```

## Next Steps

- [Installation](getting-started/installation.md) — Get up and running
- [Quick Start](getting-started/quickstart.md) — Your first replication experiment
- [Architecture](concepts/architecture.md) — How the system works
- [API Reference](api/index.md) — Full module documentation
