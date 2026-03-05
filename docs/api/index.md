# API Reference

Auto-generated documentation from source code docstrings.

## Core Modules

| Module | Description |
|--------|-------------|
| [Contract](contract.md) | `ReplicationContract`, `Manifest`, `StopCondition`, `ResourceSpec`, `NetworkPolicy` |
| [Controller](controller.md) | Central authority - registry, quotas, kill switch, audit trail |
| [Worker](worker.md) | Task-executing agent with self-replication capability |
| [Orchestrator](orchestrator.md) | Simulated container orchestrator for worker sandboxes |
| [Observability](observability.md) | Structured logging and metrics |
| [Signer](signer.md) | HMAC-SHA256 manifest signing and verification |

## Analysis & Detection Modules

| Module | Description |
|--------|-------------|
| [Threats](threats.md) | Threat modelling and attack simulation for replication scenarios |
| [Self-Modification](selfmod.md) | Detect agent self-modification attempts from action traces |
| [Forensics](forensics.md) | Post-incident forensic analysis, timeline reconstruction, root cause identification |
| [Chaos](chaos.md) | Chaos engineering — controlled fault injection to test safety invariants |
| [Game Theory](game_theory.md) | Strategic interaction analysis, Nash equilibria, collusion/defection detection |

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
