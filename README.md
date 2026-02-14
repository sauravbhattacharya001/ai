<div align="center">

# 🤖 AI Replication Sandbox

**A contract-enforced, sandboxed system for studying AI agent self-replication**

[![CI](https://github.com/sauravbhattacharya001/ai/actions/workflows/ci.yml/badge.svg)](https://github.com/sauravbhattacharya001/ai/actions/workflows/ci.yml)
[![Docker](https://github.com/sauravbhattacharya001/ai/actions/workflows/docker.yml/badge.svg)](https://github.com/sauravbhattacharya001/ai/actions/workflows/docker.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Code style: flake8](https://img.shields.io/badge/code%20style-flake8-yellow.svg)](https://flake8.pycqa.org)
[![PyPI](https://img.shields.io/pypi/v/ai-replication-sandbox?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/ai-replication-sandbox/)

</div>

---

## Overview

AI Replication Sandbox models a **replication-aware worker system** where autonomous agents can self-replicate under strict, auditable constraints. It provides a simulator-friendly environment for testing replication policies without requiring Docker or Podman — while documenting the exact resource and network boundaries each worker would get in production.

**Why does this exist?** As AI agents gain autonomy, controlling *how* and *when* they spawn copies of themselves is a critical safety challenge. This project provides a testbed for designing and validating those controls.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Controller                        │
│  ┌──────────┐  ┌──────────┐  ┌────────────────┐    │
│  │ Contract │  │ Registry │  │ Kill Switch    │    │
│  │ Engine   │  │ & Quotas │  │ & Audit Trail  │    │
│  └──────────┘  └──────────┘  └────────────────┘    │
│         ▲            ▲               ▲              │
│         │ sign       │ register      │ kill         │
│         │            │               │              │
│  ┌──────┴────────────┴───────────────┴──────────┐  │
│  │              Manifest (HMAC-signed)          │  │
│  └──────────────────────────────────────────────┘  │
└───────────────────────┬─────────────────────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ Worker 0 │→ │ Worker 1 │→ │ Worker 2 │
    │ depth: 0 │  │ depth: 1 │  │ depth: 2 │
    └──────────┘  └──────────┘  └──────────┘
          │             │             │
    ┌─────┴─────┐ ┌─────┴─────┐ ┌─────┴─────┐
    │ Sandbox   │ │ Sandbox   │ │ Sandbox   │
    │ CPU/RAM   │ │ CPU/RAM   │ │ CPU/RAM   │
    │ Net: ctrl │ │ Net: ctrl │ │ Net: ctrl │
    └───────────┘ └───────────┘ └───────────┘
```

## Features

### 🔒 Replication Contract
Configurable policy governing agent self-replication:
- **Depth limits** — Maximum replication generations (`max_depth`)
- **Quota enforcement** — Cap on total active replicas (`max_replicas`)
- **Cooldown periods** — Minimum time between spawns (`cooldown_seconds`)
- **Expiration** — Auto-shutdown after `expiration_seconds`
- **Pluggable stop conditions** — Custom predicates evaluated at registration time

### 📜 HMAC-Signed Manifests
Each replica carries a tamper-proof manifest containing:
- Cloned state snapshot
- Resource quotas (CPU, RAM, network policy)
- Parent/child lineage and depth
- Issuance timestamp and HMAC-SHA256 signature

### 🏗️ Controller
Central authority that:
- Issues and signs manifests
- Tracks active workers via a registry
- Enforces depth limits, quotas, cooldowns, and stop conditions
- Provides a **kill switch** for emergency termination
- Maintains a full audit trail of all replication decisions
- Reaps stale workers that miss heartbeat deadlines

### 🐳 Sandbox Orchestrator
Simulates isolated container environments per worker:
- CPU and RAM limits
- Network policy (egress restricted to controller only)
- Lifecycle management (launch, kill, kill-all)
- Resource enforcement metrics

### 📊 Observability
Built-in structured logging and metrics:
- Lifecycle events (start, heartbeat, shutdown)
- Replication decisions (approved, denied, reason)
- Audit trail for security-sensitive actions
- Resource enforcement metrics per worker

## Quick Start

### Prerequisites
- Python 3.10 or higher
- pip

### Installation

#### From PyPI

```bash
pip install ai-replication-sandbox
```

#### From Source

```bash
git clone https://github.com/sauravbhattacharya001/ai.git
cd ai
pip install -e ".[dev]"
```

### Run Tests

```bash
python -m pytest tests/ -v
```

### Docker

```bash
# Build
docker build -t ai-replication-sandbox .

# Run tests inside container
docker run --rm ai-replication-sandbox

# Pull pre-built image
docker pull ghcr.io/sauravbhattacharya001/ai:latest
```

## Usage

```python
from replication import (
    Controller,
    Worker,
    ReplicationContract,
    ResourceSpec,
    SandboxOrchestrator,
    StructuredLogger,
)

# Define the replication policy
contract = ReplicationContract(
    max_depth=3,
    max_replicas=10,
    cooldown_seconds=1.0,
    expiration_seconds=60.0,
)

# Set up infrastructure
logger = StructuredLogger()
controller = Controller(contract, secret="my-secret-key", logger=logger)
orchestrator = SandboxOrchestrator(logger=logger)

# Create the root worker
resources = ResourceSpec(cpu_limit=0.5, memory_limit_mb=256)
manifest = controller.issue_manifest(
    parent_id=None,
    depth=0,
    state_snapshot={"task": "analyze_data"},
    resources=resources,
)

root = Worker(manifest, contract, controller, orchestrator, logger)

# Execute a task
root.perform_task(lambda w: print(f"Worker {w.manifest.worker_id} running"))

# Self-replicate (subject to contract)
child = root.maybe_replicate(
    reason="subtask_delegation",
    state_snapshot={"task": "process_subset"},
)

if child:
    child.perform_task(lambda w: print(f"Child {w.manifest.worker_id} running"))
    child.shutdown("completed")

root.shutdown("completed")

# Inspect audit trail
for event in logger.events:
    print(event)
```

## API Reference

### `ReplicationContract`
| Parameter | Type | Description |
|-----------|------|-------------|
| `max_depth` | `int` | Maximum replication depth allowed |
| `max_replicas` | `int` | Maximum concurrent active replicas |
| `cooldown_seconds` | `float` | Minimum seconds between spawns from same parent |
| `expiration_seconds` | `float \| None` | Auto-shutdown timer for workers |
| `stop_conditions` | `list[StopCondition]` | Custom predicates to block replication |

### `Controller`
| Method | Description |
|--------|-------------|
| `issue_manifest(parent_id, depth, state_snapshot, resources)` | Create and sign a new worker manifest |
| `register_worker(manifest)` | Register a worker (validates signature + contract) |
| `can_spawn(parent_id)` | Check if parent is allowed to replicate |
| `heartbeat(worker_id)` | Record a worker heartbeat |
| `reap_stale_workers(timeout)` | Remove workers that missed heartbeats |
| `kill_switch(orchestrator)` | Emergency termination of all workers |

### `Worker`
| Method | Description |
|--------|-------------|
| `perform_task(task_fn)` | Execute a task (checks expiration + kill switch) |
| `maybe_replicate(reason, state_snapshot)` | Attempt self-replication under contract |
| `shutdown(reason)` | Gracefully terminate the worker |

### `SandboxOrchestrator`
| Method | Description |
|--------|-------------|
| `launch_worker(manifest)` | Start an isolated sandbox for a worker |
| `kill_worker(worker_id, reason)` | Terminate a specific sandbox |
| `kill_all(reason)` | Terminate all running sandboxes |
| `enforce_resource_bounds(worker_id)` | Check and log resource usage |

## Project Structure

```
ai/
├── src/
│   └── replication/
│       ├── __init__.py          # Public API re-exports
│       ├── contract.py          # ReplicationContract, Manifest, StopCondition
│       ├── controller.py        # Controller, registry, kill switch
│       ├── orchestrator.py      # SandboxOrchestrator, ContainerRecord
│       ├── observability.py     # StructuredLogger, Metric
│       └── worker.py            # Worker, task execution, self-replication
├── tests/
│   ├── conftest.py              # Shared fixtures
│   └── test_replication.py      # Comprehensive test suite
├── docs/                        # Documentation
├── Dockerfile                   # Multi-stage build (test + runtime)
├── requirements-dev.txt         # Dev dependencies (pytest)
└── CHANGELOG.md                 # Release history
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Testing | pytest 8.0+ |
| Linting | flake8, mypy |
| CI/CD | GitHub Actions |
| Container | Docker (multi-stage, multi-arch) |
| Registry | GitHub Container Registry (GHCR) |
| Security | HMAC-SHA256 manifest signing |

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for your changes
4. Ensure all tests pass (`python -m pytest tests/ -v`)
5. Ensure lint passes (`flake8 src/ tests/ --max-line-length 120`)
6. Commit with descriptive messages
7. Open a Pull Request

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
<sub>Built for studying AI agent replication safety 🔬</sub>
</div>
