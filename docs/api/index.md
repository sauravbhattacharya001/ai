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
| [Prompt Injection](prompt_injection.md) | Detect and classify prompt injection attacks — 8 attack vectors, encoding evasion, risk scoring |
| [Escalation](escalation.md) | Detect privilege escalation patterns — multi-vector analysis, stealth assessment, chain detection |
| [Kill Chain](killchain.md) | Map agent behavior to cyber kill chain stages — attack progression and sophistication rating |
| [Honeypot](honeypot.md) | Deploy decoy resources to detect unauthorized agent access and build threat profiles |

## Governance & Compliance

| Module | Description |
|--------|-------------|
| [Compliance](compliance.md) | Audit agent behavior against compliance frameworks (SOC2, NIST, ISO 27001, GDPR) |
| [Quarantine](quarantine.md) | Isolate misbehaving agents — suspend, restrict, or terminate with escalation policies |
| [Lineage](lineage.md) | Trace agent replication lineage — parent-child trees, state mutations, generational drift |
| [Monte Carlo](montecarlo.md) | Probabilistic simulation of replication scenarios — risk distributions, worst-case analysis |

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

# Detection & Analysis
from replication.prompt_injection import PromptInjectionDetector
from replication.escalation import EscalationDetector
from replication.killchain import KillChainAnalyzer
from replication.honeypot import HoneypotManager

# Governance
from replication.compliance import ComplianceAuditor
from replication.quarantine import QuarantineManager
from replication.lineage import LineageTracker
from replication.montecarlo import MonteCarloSimulator
```
