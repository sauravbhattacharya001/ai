# API Reference

Auto-generated documentation from source code docstrings.

## Modules

| Module | Description |
|--------|-------------|
| [Contract](contract.md) | `ReplicationContract`, `Manifest`, `StopCondition`, `ResourceSpec`, `NetworkPolicy` |
| [Controller](controller.md) | Central authority — registry, quotas, kill switch, audit trail |
| [Worker](worker.md) | Task-executing agent with self-replication capability |
| [Orchestrator](orchestrator.md) | Simulated container orchestrator for worker sandboxes |
| [Observability](observability.md) | Structured logging and metrics |
| [Signer](signer.md) | HMAC-SHA256 manifest signing and verification |

## Quick Import

```python
from replication import (
    # Contract
    ReplicationContract, Manifest, StopCondition,
    ResourceSpec, NetworkPolicy, ReplicationContext, Predicate,
    # Controller
    Controller, ReplicationDenied, RegistryEntry,
    # Worker
    Worker, WorkerState,
    # Orchestrator
    SandboxOrchestrator, ContainerRecord,
    # Observability
    StructuredLogger, Metric,
    # Signer
    ManifestSigner,
)
```
