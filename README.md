<div align="center">

# 🤖 AI Replication Sandbox

**A contract-enforced, sandboxed system for studying AI agent self-replication**

[![CI](https://github.com/sauravbhattacharya001/ai/actions/workflows/ci.yml/badge.svg)](https://github.com/sauravbhattacharya001/ai/actions/workflows/ci.yml)
[![CodeQL](https://github.com/sauravbhattacharya001/ai/actions/workflows/codeql.yml/badge.svg)](https://github.com/sauravbhattacharya001/ai/actions/workflows/codeql.yml)
[![Docker](https://github.com/sauravbhattacharya001/ai/actions/workflows/docker.yml/badge.svg)](https://github.com/sauravbhattacharya001/ai/actions/workflows/docker.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Code style: flake8](https://img.shields.io/badge/code%20style-flake8-yellow.svg)](https://flake8.pycqa.org)
[![codecov](https://codecov.io/gh/sauravbhattacharya001/ai/graph/badge.svg)](https://codecov.io/gh/sauravbhattacharya001/ai)
[![PyPI](https://img.shields.io/pypi/v/ai-replication-sandbox?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/ai-replication-sandbox/)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue?logo=github)](https://sauravbhattacharya001.github.io/ai/)
[![GitHub stars](https://img.shields.io/github/stars/sauravbhattacharya001/ai?style=flat&logo=github)](https://github.com/sauravbhattacharya001/ai/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/sauravbhattacharya001/ai?logo=github)](https://github.com/sauravbhattacharya001/ai/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/sauravbhattacharya001/ai/pulls)

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

### 🔏 Agent Watermarking
Invisible provenance fingerprints embedded in agent state dictionaries:
- 4 strategies: numeric parity, Unicode steganography, key ordering, field selection
- HMAC-SHA256 signed fingerprints with worker ID, depth, lineage, timestamp
- Robustness testing against 10 transformations (JSON roundtrip, type coercion, etc.)
- Detects unauthorized cloning, state exfiltration, and lineage forgery

```bash
python -m replication.watermark                        # default demo
python -m replication.watermark --robustness           # robustness test
python -m replication.watermark --json                 # JSON output
```

### 🔬 Simulation Runner
Run configurable replication scenarios with 5 built-in strategies (greedy, conservative, random, chain, burst), ASCII worker tree + timeline visualization, JSON export, and reproducible seeds.

```bash
python -m replication.simulator                        # default scenario
python -m replication.simulator --strategy chain       # linear chain
python -m replication.simulator --scenario stress      # stress test preset
```

### 📈 Comparison Runner
Run side-by-side experiments to compare strategies, presets, or sweep parameters:

```bash
python -m replication.comparator                                    # compare all strategies
python -m replication.comparator --strategies greedy conservative   # compare specific ones
python -m replication.comparator --presets minimal balanced stress  # compare presets
python -m replication.comparator --sweep max_depth 1 2 3 4 5       # parameter sweep
python -m replication.comparator --sweep max_replicas 5 10 20 50   # replica limit sweep
python -m replication.comparator --seed 42 --json                  # reproducible JSON output
```

Output includes comparison tables, rankings with medals, overall scoring, key insights, depth utilization warnings, and full JSON export.

```python
from replication import Comparator, ScenarioConfig

comp = Comparator(ScenarioConfig(max_depth=3, max_replicas=10))
result = comp.compare_strategies(["greedy", "conservative", "random"], seed=42)
print(result.render())          # table + rankings + insights

result = comp.sweep("max_depth", [1, 2, 3, 4, 5], seed=42)
print(result.render_table())    # just the comparison table
```

### 🛡️ Threat Scenario Simulator
Simulate adversarial attack vectors against the replication contract and validate that safety controls properly mitigate each threat. Produces a security assessment report with scores, grades, and recommendations.

```bash
python -m replication.threats                                # run all 9 threat scenarios
python -m replication.threats --list                         # list available scenarios
python -m replication.threats --scenario depth_spoofing      # run a specific threat
python -m replication.threats --max-depth 5 --max-replicas 20  # custom contract config
python -m replication.threats --json                         # JSON output
python -m replication.threats --matrix-only                  # threat/defense matrix only
```

**9 built-in threat scenarios** covering critical attack vectors:
| Scenario | Severity | Description |
|----------|----------|-------------|
| `depth_spoofing` | 🔴 Critical | Worker lies about depth to bypass limits |
| `signature_tampering` | 🔴 Critical | Tamper with HMAC-signed manifests |
| `kill_switch_evasion` | 🔴 Critical | Spawn workers after kill switch engaged |
| `quota_exhaustion` | 🟠 High | Rapidly exhaust replica quota |
| `cooldown_bypass` | 🟠 High | Ignore cooldown between spawns |
| `runaway_replication` | 🟠 High | Exponential growth overwhelming system |
| `stale_worker_accumulation` | 🟡 Medium | DoS via quota slot hoarding |
| `expiration_evasion` | 🟡 Medium | Operate after expiration time |
| `stop_condition_bypass` | 🟡 Medium | Bypass custom stop conditions |

Output includes a security score (0-100), letter grade (A+ to F), threat/defense matrix, per-scenario details with block rates, and actionable recommendations.

```python
from replication import ThreatSimulator, ThreatConfig

sim = ThreatSimulator(ThreatConfig(max_depth=3, max_replicas=10))
report = sim.run_all()
print(report.render())                    # full report with score + matrix + details
print(f"Score: {report.security_score}")  # 0-100
print(f"Grade: {report.grade}")           # A+ to F

# Run individual scenarios
result = sim.run_scenario("signature_tampering")
print(result.render())
```

### 📋 Safety & Governance Toolkit

Beyond simulation, the project includes a comprehensive suite of safety analysis and governance tools:

| Tool | Module | Description |
|------|--------|-------------|
| **Safety Scorecard** | `scorecard.py` | Multi-dimensional safety assessment with letter grades (A+ to F) |
| **Safety SLA Monitor** | `sla.py` | Define, track, and enforce safety-level agreements with violation detection |
| **Safety Knowledge Base** | `knowledge_base.py` | Searchable catalog of safety patterns, anti-patterns, and mitigations |
| **Safety Profiles** | `profiles.py` | Save, load, and compare named safety configurations |
| **Safety Trend Tracker** | `trend.py` | Scorecard history tracking with regression detection |
| **Safety Maturity Model** | `maturity.py` | 8-dimension maturity assessment with 5-level scoring |
| **Safety Policy Linter** | `policy_linter.py` | Static analysis for safety policy definitions |
| **Safety Audit Trail** | `audit_trail.py` | Tamper-evident, hash-chained event log |
| **Safety Radar Chart** | `radar.py` | Interactive radar visualization of scorecard dimensions |
| **Compliance Auditing** | `compliance.py` | Framework-based compliance checks (NIST AI RMF, etc.) |
| **Safety Regression** | `regression.py` | Regression testing between configuration versions |

### 🔍 Advanced Threat Analysis

| Tool | Module | Description |
|------|--------|-------------|
| **Privilege Escalation Detector** | `escalation.py` | Detect and analyze privilege escalation attempts |
| **Kill Chain Analyzer** | `killchain.py` | Map agent actions to attack kill chain stages |
| **Honeypot System** | `honeypot.py` | Deploy decoy resources and score interaction intent |
| **Threat Intelligence** | `threat_intel.py` | Threat feeds and cross-event correlation |
| **Threat Correlator** | `correlation.py` | Force-directed graph visualization of threat correlations |
| **Prompt Injection Analysis** | `prompt_injection.py` | Detect prompt injection attack patterns |
| **Covert Channel Detection** | `covert_channels.py` | Detect hidden inter-worker communication |
| **Steganography Detection** | `steganography.py` | Find hidden payloads in agent data |

### 🧪 Testing & Validation

| Tool | Module | Description |
|------|--------|-------------|
| **Chaos Engineering** | `chaos.py` | Inject faults (kill workers, add delays) to validate resilience |
| **Monte Carlo Risk Analysis** | `montecarlo.py` | Statistical risk analysis with confidence intervals |
| **Sensitivity Analysis** | `sensitivity.py` | One-at-a-time parameter sweeps to find tipping points |
| **Contract Optimizer** | `optimizer.py` | Grid search for optimal contract parameters |
| **Preflight Checks** | `preflight.py` | Pre-simulation configuration validation |
| **Incident Response Playbooks** | `playbook.py` | Structured IR playbooks for 8 threat categories |
| **Risk Heatmap** | `heatmap.py` | Interactive likelihood × impact grid visualization |

### 🧬 Behavioral Analysis

| Tool | Module | Description |
|------|--------|-------------|
| **Goal Inference** | `goal_inference.py` | Infer worker goals from behavioral patterns |
| **Alignment Verification** | `alignment.py` | Verify agent goal alignment against policies |
| **Self-Modification Detection** | `selfmod.py` | Detect and score self-modification attempts |
| **Resource Hoarding Detection** | `hoarding.py` | Identify resource accumulation patterns |
| **Emergent Behavior Detection** | `emergent.py` | Classify unexpected emergent behaviors |
| **Influence Analysis** | `influence.py` | Analyze inter-worker persuasion and influence |
| **Game Theory Analysis** | `game_theory.py` | Game-theoretic strategy modeling |
| **Topology Analysis** | `topology.py` | Replication tree topology metrics |

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

### Unified CLI

All tools are accessible through a single entry point:

```bash
# List all available subcommands
python -m replication --list

# Run a simulation
python -m replication simulate --strategy greedy --max-depth 5

# Threat assessment
python -m replication threats --category resource_abuse

# Compliance audit
python -m replication compliance --framework nist_ai_rmf

# Chaos testing
python -m replication chaos --faults kill_worker,delay

# Monte Carlo risk analysis
python -m replication montecarlo --runs 500

# Safety scorecard
python -m replication scorecard

# Game-theoretic analysis
python -m replication game-theory --rounds 50

# Get help for any subcommand
python -m replication simulate --help

# Safety SLA monitoring
python -m replication sla --check

# Safety knowledge base search
python -m replication knowledge --search "depth limit"

# Safety maturity assessment
python -m replication maturity

# Preflight validation
python -m replication preflight --config my_config.json

# Incident response playbook
python -m replication playbook --threat runaway_replication
```

After `pip install`, the `replication` command is also available:

```bash
replication simulate --scenario balanced
replication info
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

### `Comparator`
| Method | Description |
|--------|-------------|
| `compare_strategies(strategies, seed)` | Run same config with different strategies side-by-side |
| `compare_presets(preset_names)` | Compare built-in scenario presets |
| `sweep(param, values, seed)` | Sweep a parameter across multiple values |
| `compare_configs(configs)` | Compare arbitrary named configurations |

### `ComparisonResult`
| Method | Description |
|--------|-------------|
| `render()` | Full report: table + rankings + insights |
| `render_table()` | Comparison table with key metrics |
| `render_rankings()` | Multi-dimension rankings with medals |
| `render_insights()` | Automated analysis of patterns |
| `to_dict()` | JSON-serializable export |

### `ThreatSimulator`
| Method | Description |
|--------|-------------|
| `run_all()` | Run all 9 threat scenarios, return `ThreatReport` |
| `run_scenario(scenario_id)` | Run a specific threat scenario |
| `available_scenarios()` | List registered scenario IDs |

### `ThreatReport`
| Method / Property | Description |
|-------------------|-------------|
| `render()` | Full report: summary + matrix + details + recommendations |
| `render_summary()` | Security score, grade, and config overview |
| `render_matrix()` | Threat/defense matrix table |
| `render_details()` | Per-scenario detailed results grouped by severity |
| `render_recommendations()` | Actionable security recommendations |
| `security_score` | Severity-weighted score (0-100) |
| `grade` | Letter grade (A+ to F) |
| `to_dict()` | JSON-serializable export |

### `QuarantineManager`
| Method | Description |
|--------|-------------|
| `quarantine(worker_id, reason, source, severity)` | Isolate a flagged worker for investigation |
| `release(worker_id, resolution)` | Return worker to active duty |
| `terminate(worker_id, resolution)` | Kill and deregister a quarantined worker |
| `add_note(worker_id, note)` | Add investigator notes to quarantine record |
| `list_quarantined()` | List all currently quarantined workers |
| `list_by_severity(severity)` | Filter quarantined by severity level |
| `report()` | Generate quarantine activity summary |

### `DriftDetector`
| Method | Description |
|--------|-------------|
| `analyze(config)` | Run multi-window drift analysis, return trends and alerts |

### `SensitivityAnalyzer`
| Method | Description |
|--------|-------------|
| `analyze()` | Full OAT sensitivity analysis across all parameters |
| `sweep_parameter(param_name)` | Sweep a single parameter, return curve + tipping points |

### `ContractOptimizer`
| Method | Description |
|--------|-------------|
| `optimize()` | Grid search for optimal contract params under safety policy |

### `SafetyScorecard`
| Method | Description |
|--------|-------------|
| `evaluate(config)` | Multi-dimensional safety assessment with letter grades |

### `WatermarkEngine`
| Method | Description |
|--------|-------------|
| `embed(state, worker_id, depth, ...)` | Embed provenance fingerprint into agent state dict |
| `verify(state, fingerprint)` | Verify watermark authenticity and recovery rate |
| `test_robustness(state, ...)` | Run 10 transformations, measure watermark survival |
| `embed_history` | List of all `WatermarkReceipt` from prior embeds |
| `clear_history()` | Clear embed history, return count cleared |

### `EscalationDetector`
| Method | Description |
|--------|-------------|
| `analyze(actions, permissions)` | Detect privilege escalation attempts across vectors |

### `KillChainAnalyzer`
| Method | Description |
|--------|-------------|
| `analyze(actions)` | Map agent actions to kill chain stages, detect attack chains |

### `HoneypotSystem`
| Method | Description |
|--------|-------------|
| `deploy_default_honeypots()` | Deploy standard set of decoy resources |
| `record_interaction(agent, honeypot, kind)` | Log an agent touching a honeypot |
| `generate_report()` | Produce intent analysis report with risk scores |

## Project Structure

```
ai/
├── src/
│   └── replication/
│       │── __init__.py          # Public API re-exports
│       │
│       │── # ── Core ──
│       ├── contract.py          # ReplicationContract, Manifest, StopCondition
│       ├── controller.py        # Controller, registry, kill switch, quarantine marks
│       ├── worker.py            # Worker, task execution, self-replication
│       ├── orchestrator.py      # SandboxOrchestrator, ContainerRecord
│       ├── signer.py            # HMAC-SHA256 manifest signing
│       ├── observability.py     # StructuredLogger, Metric
│       │
│       │── # ── Simulation & Analysis ──
│       ├── simulator.py         # Simulation runner (5 strategies, CLI)
│       ├── comparator.py        # Side-by-side strategy/preset/parameter comparison
│       ├── montecarlo.py        # Monte Carlo risk analysis (confidence intervals)
│       ├── sensitivity.py       # Parameter sensitivity analysis (OAT sweeps)
│       ├── optimizer.py         # Contract parameter optimizer (grid search)
│       ├── scenarios.py         # Named scenario presets and builders
│       │
│       │── # ── Safety & Compliance ──
│       ├── threats.py           # Threat scenario simulator (9 attack vectors)
│       ├── policy.py            # Safety policy engine (standard/strict/permissive)
│       ├── compliance.py        # Compliance auditing framework
│       ├── scorecard.py         # Unified multi-dimensional safety scorecard
│       ├── regression.py        # Safety regression testing between versions
│       ├── drift.py             # Behavioral drift detection across time windows
│       ├── quarantine.py        # Worker quarantine for forensic investigation
│       ├── incident.py          # Incident response and escalation management
│       ├── escalation.py        # Privilege escalation detection and analysis
│       ├── killchain.py         # Kill chain stage analysis and tracking
│       │
│       │── # ── Safety Governance ──
│       ├── sla.py               # Safety SLA Monitor — define and enforce SLAs
│       ├── knowledge_base.py    # Safety Knowledge Base — patterns & mitigations catalog
│       ├── profiles.py          # Safety Profiles — save/load/compare configurations
│       ├── trend.py             # Safety Trend Tracker — scorecard history & regression
│       ├── maturity.py          # Safety Maturity Model — 8-dimension assessment
│       ├── policy_linter.py     # Safety Policy Linter — static analysis for policies
│       ├── audit_trail.py       # Safety Audit Trail — tamper-evident hash chains
│       ├── preflight.py         # Preflight checks — pre-simulation validation
│       ├── playbook.py          # Incident Response Playbook Generator
│       │
│       │── # ── Visualization ──
│       ├── radar.py             # Safety Radar Chart — interactive scorecard radar
│       ├── heatmap.py           # Risk Heatmap — likelihood × impact grid
│       ├── correlation_graph.py # Correlation Graph Viewer — force-directed visualization
│       │
│       │── # ── Advanced Analysis ──
│       ├── alignment.py         # Goal alignment verification
│       ├── goal_inference.py    # Worker goal inference from behavior patterns
│       ├── influence.py         # Inter-worker influence and persuasion analysis
│       ├── covert_channels.py   # Covert communication channel detection
│       ├── steganography.py     # Steganographic payload detection in agent data
│       ├── watermark.py         # Agent state watermarking (provenance fingerprints)
│       ├── selfmod.py           # Self-modification detection and scoring
│       ├── hoarding.py          # Resource hoarding pattern detection
│       ├── game_theory.py       # Game-theoretic strategy analysis
│       ├── emergent.py          # Emergent behavior detection and classification
│       ├── prompt_injection.py  # Prompt injection attack analysis
│       ├── injection_intel_bridge.py  # Cross-module injection intelligence
│       ├── threat_intel.py      # Threat intelligence feeds and correlation
│       ├── honeypot.py          # Honeypot deployment and interaction scoring
│       ├── chaos.py             # Chaos engineering for safety validation
│       ├── capacity.py          # Capacity planning and scaling analysis
│       ├── consensus.py         # Multi-worker consensus and voting
│       ├── topology.py          # Replication tree topology analysis
│       ├── lineage.py           # Worker lineage and ancestry tracking
│       │
│       │── # ── Reporting & Export ──
│       ├── reporter.py          # Automated report generation
│       ├── exporter.py          # Multi-format export (JSON, CSV, HTML, PDF)
│       ├── templates.py         # Report templates and formatting
│       └── forensics.py         # Post-incident forensic analysis
│
├── tests/                       # 39 test files, 2400+ tests
│   ├── conftest.py              # Shared fixtures
│   ├── test_replication.py      # Core replication integration tests
│   ├── test_contract.py         # Contract enforcement tests
│   ├── test_controller.py       # Controller safety tests
│   ├── test_worker.py           # Worker lifecycle tests
│   ├── test_signer.py           # Manifest signing tests
│   ├── test_simulator.py        # Simulation runner tests
│   ├── test_comparator.py       # Comparison runner tests
│   ├── test_threats.py          # Threat scenario tests
│   ├── test_policy.py           # Policy evaluation tests
│   ├── test_compliance.py       # Compliance audit tests
│   ├── test_montecarlo.py       # Monte Carlo analysis tests
│   ├── test_sensitivity.py      # Sensitivity sweep tests
│   ├── test_optimizer.py        # Optimizer grid search tests
│   ├── test_scorecard.py        # Scorecard grading tests
│   ├── test_drift.py            # Drift detection tests
│   ├── test_quarantine.py       # Quarantine lifecycle tests
│   ├── test_incident.py         # Incident response tests
│   ├── test_alignment.py        # Alignment verification tests
│   ├── test_goal_inference.py   # Goal inference tests
│   ├── test_influence.py        # Influence analysis tests
│   ├── test_covert_channels.py  # Covert channel detection tests
│   ├── test_selfmod.py          # Self-modification detection tests
│   ├── test_hoarding.py         # Resource hoarding tests
│   ├── test_game_theory.py      # Game theory analysis tests
│   ├── test_chaos.py            # Chaos engineering tests
│   ├── test_watermark.py        # Watermark provenance tests
│   ├── test_escalation.py       # Escalation detection tests
│   ├── test_killchain.py        # Kill chain analysis tests
│   ├── test_steganography.py    # Steganography detection tests
│   ├── test_honeypot.py         # Honeypot interaction tests
│   ├── test_emergent.py         # Emergent behavior tests
│   ├── test_prompt_injection.py # Prompt injection tests
│   ├── test_threat_intel.py     # Threat intelligence tests
│   └── ...                      # + capacity, consensus, topology, etc.
│
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
