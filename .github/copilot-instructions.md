# Copilot Instructions — AI Replication Sandbox

## Project Overview

A **Python simulation** of a replication-aware worker system with explicit contracts, sandboxed orchestration, and structured observability. Models how AI agents could safely replicate under strict policy controls — pure Python, no Docker/Podman at runtime.

## Architecture

The core package lives in `src/replication/` with 100+ modules spanning several domains:

### Core (Contract & Orchestration)
- `contract.py` — `ReplicationContract`, `Manifest`, `StopCondition`, `ResourceSpec`
- `controller.py` — Signs manifests, enforces quotas, kill switch, audit trail
- `orchestrator.py` — Simulates container isolation (CPU/RAM/network)
- `worker.py` — Executes tasks, replicates subject to contract
- `observability.py` — `StructuredLogger` + `Metric` for lifecycle events
- `signer.py` — HMAC manifest signing utilities
- `_helpers.py` — Internal shared helpers

### Safety & Alignment
- `alignment.py` — Alignment auditing
- `safety_gate.py`, `safety_budget.py`, `safety_drill.py`, `safety_benchmark.py`, `safety_warranty.py`, `safety_quiz.py`, `safety_timeline.py`

### Threat Modeling & Security
- `threats.py`, `threat_intel.py`, `threat_hunt.py`, `threat_correlator.py`
- `attack_surface.py`, `attack_tree.py`, `red_team.py`, `killchain.py`, `stride.py`
- `prompt_injection.py`, `covert_channels.py`, `steganography.py`, `evasion.py`
- `priv_escalation.py`, `escape_route.py`, `vulnerability_scanner.py`

### Incident Response
- `incident.py`, `incident_comms.py`, `incident_cost.py`
- `ir_playbook.py`, `forensics.py`, `postmortem.py`, `evidence_collector.py`
- `kill_switch.py`, `containment_planner.py`, `quarantine.py`, `decommission.py`

### Governance & Compliance
- `policy.py`, `policy_linter.py`, `compliance.py`, `regulatory_mapper.py`
- `access_control.py`, `audit_trail.py`, `model_card.py`, `maturity_model.py`

### Simulation & Analysis
- `chaos.py`, `montecarlo.py`, `scenarios.py`, `tabletop.py`, `what_if.py`
- `game_theory.py`, `swarm.py`, `simulator.py`, `boundary_tester.py`
- `behavior_profiler.py`, `anomaly_cluster.py`, `anomaly_replay.py`

### Observability & Reporting
- `dashboard.py`, `reporter.py`, `exporter.py`, `scorecard.py`, `radar.py`
- `metrics_aggregator.py`, `risk_heatmap.py`, `risk_profiler.py`
- `alert_router.py`, `sla_monitor.py`, `trend_tracker.py`

## Conventions

- **Python 3.10+** — uses `from __future__ import annotations`, dataclasses, typing
- **No external dependencies** for core — only `pytest>=8.0`, `flake8`, `mypy` for dev
- **Install via**: `pip install -e ".[dev]"` (uses hatchling build backend)
- **Dataclasses everywhere** — all domain objects are `@dataclass` with type hints
- **Import from package root**: `from replication import Controller, Worker, ReplicationContract`
- **Import submodules directly**: `from replication.alignment import AlignmentAuditor`
- Tests use pytest with `pythonpath = ["src"]` configured in `pyproject.toml`

## How to Test

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
python -m mypy src/replication/ --ignore-missing-imports
python -m flake8 src/replication/ --max-line-length=120
```

## Code Style

- Type hints on all function signatures
- Dataclasses for data structures (not dicts)
- Single responsibility per function
- Structured logging via `StructuredLogger`, not `print()`
- No global mutable state — pass dependencies explicitly
- Max line length: 120 characters

### Supply Chain & Dependencies
- `supply_chain.py` — Supply chain risk analysis
- `dependency_graph.py` — Dependency graph tracking
- `dlp_scanner.py` — Data loss prevention scanning

### Fleet & Multi-Agent
- `fleet.py` — Fleet-wide agent management
- `consensus.py` — Multi-agent consensus protocols
- `coordinated_threats.py` — Coordinated threat detection
- `trust_propagation.py` — Trust propagation across agent networks
- `influence.py` — Influence analysis between agents
- `topology.py` — Network topology modeling

### Self-Modification & Drift
- `selfmod.py` — Self-modification detection
- `drift.py` — Behavioral drift tracking
- `mutation_tester.py` — Mutation testing for safety invariants
- `emergent.py` — Emergent behavior detection
- `goal_inference.py` — Goal inference from agent behavior

### Defensive & Detection
- `honeypot.py` — Honeypot deployment for agent traps
- `canary.py` — Canary token monitoring
- `deception_detector.py` — Deception detection
- `watermark.py` — Output watermarking
- `capability_fingerprint.py` — Capability fingerprinting
- `capability_catalog.py` — Capability catalog management

### Operations & Readiness
- `runbook.py` — Operational runbooks
- `preflight.py` — Pre-deployment checks
- `hardening_advisor.py` — Security hardening recommendations
- `blast_radius.py` — Blast radius estimation
- `exposure_window.py` — Exposure window calculation
- `fatigue_detector.py` — Alert fatigue detection
- `escalation.py` — Escalation workflow management
- `culture_survey.py` — Security culture assessment

### Analysis & Utilities
- `comparator.py` — Configuration comparison
- `sensitivity.py` — Sensitivity analysis
- `severity_classifier.py` — Severity classification
- `regression.py` — Regression detection
- `optimizer.py` — Resource optimization
- `capacity.py` — Capacity planning
- `lineage.py` — Data/model lineage tracking
- `knowledge_base.py` — Knowledge base for incidents/patterns
- `shadow_ai.py` — Shadow AI detection
- `profiles.py`, `persona.py` — Agent profiling and persona management
- `playground.py` — Safe experimentation sandbox
- `templates.py` — Template management
- `correlation_graph.py` — Event correlation graphing
- `injection_intel_bridge.py` — Prompt injection intelligence bridge

## When Making Changes

1. Run the full test suite before and after changes
2. If adding a new module, re-export its public API from `__init__.py`
3. If changing contract/manifest fields, update both `contract.py` and controller signing logic
4. Keep the simulation runnable without Docker — real container orchestration is out of scope
5. Add type hints and docstrings to new public functions
6. Test files go in `tests/` — use `test_<module_name>.py` naming
