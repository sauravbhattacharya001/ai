# Copilot Instructions — AI Replication Sandbox

## Project Overview

This is a **Python simulation** of a replication-aware worker system with explicit contracts, sandboxed orchestration, and structured observability. It models how AI agents could safely replicate under strict policy controls — without requiring Docker/Podman at runtime.

## Architecture

```
src/replication/
├── contract.py       # ReplicationContract, Manifest, StopCondition, ResourceSpec
├── controller.py     # Controller — signs manifests, enforces quotas, kill switch, audit trail
├── orchestrator.py   # SandboxOrchestrator — simulates container isolation (CPU/RAM/network)
├── worker.py         # Worker — executes tasks, replicates subject to contract
├── observability.py  # StructuredLogger + Metric — lifecycle events and audit logging
└── __init__.py       # Public API re-exports
```

### Key Classes

- **`ReplicationContract`**: Defines `max_depth`, `max_replicas`, `cooldown_seconds`, `expiration_seconds`, and pluggable `StopCondition` predicates.
- **`Controller`**: Central authority that signs manifests (HMAC), tracks active replicas in a registry, enforces depth/quota/cooldown limits, and provides a kill switch.
- **`SandboxOrchestrator`**: Simulates isolated execution environments with CPU/RAM limits and network policy (egress-only to controller).
- **`Worker`**: Runs a task, optionally self-replicates (checked against the contract), registers with the controller, heartbeats, and self-terminates on expiration or kill.
- **`StructuredLogger` / `Metric`**: Structured logging and metric emission for lifecycle auditing.

## Conventions

- **Python 3.10+** required (uses `from __future__ import annotations`, dataclasses, typing).
- **No external dependencies** for the core library — only `pytest>=8.0` for tests.
- **Dataclasses everywhere** — all domain objects are `@dataclass` with type hints.
- **Import from package root**: `from replication import Controller, Worker, ReplicationContract`
- Tests add `src/` to `sys.path` via `conftest.py`.

## How to Test

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

Tests cover:
- Replication up to permitted depth
- Quota enforcement (max replicas)
- Stop condition evaluation
- Kill switch during chaos scenarios
- Manifest HMAC signature validation

## Code Style

- Use type hints on all function signatures
- Use dataclasses for data structures (not dicts)
- Keep functions focused — single responsibility
- Structured logging via `StructuredLogger`, not `print()`
- No global mutable state — pass dependencies explicitly

## When Making Changes

1. Run the full test suite before and after changes
2. If adding a new module, re-export its public API from `__init__.py`
3. If changing contract/manifest fields, update both `contract.py` and the controller's signing logic
4. Keep the simulation runnable without Docker — real container orchestration is out of scope
