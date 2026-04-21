<div align="center">

# 🤖 AI Replication Sandbox

**A comprehensive, contract-enforced framework for studying AI agent self-replication safety**

[![CI](https://github.com/sauravbhattacharya001/ai/actions/workflows/ci.yml/badge.svg)](https://github.com/sauravbhattacharya001/ai/actions/workflows/ci.yml)
[![CodeQL](https://github.com/sauravbhattacharya001/ai/actions/workflows/codeql.yml/badge.svg)](https://github.com/sauravbhattacharya001/ai/actions/workflows/codeql.yml)
[![Docker](https://github.com/sauravbhattacharya001/ai/actions/workflows/docker.yml/badge.svg)](https://github.com/sauravbhattacharya001/ai/actions/workflows/docker.yml)
[![codecov](https://codecov.io/gh/sauravbhattacharya001/ai/graph/badge.svg)](https://codecov.io/gh/sauravbhattacharya001/ai)
[![PyPI](https://img.shields.io/pypi/v/ai-replication-sandbox?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/ai-replication-sandbox/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue?logo=github)](https://sauravbhattacharya001.github.io/ai/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/sauravbhattacharya001/ai/pulls)
[![Dependabot](https://img.shields.io/badge/dependabot-enabled-025E8C?logo=dependabot)](https://github.com/sauravbhattacharya001/ai/network/updates)
[![Security Policy](https://img.shields.io/badge/security-policy-green?logo=shield)](SECURITY.md)

*144 modules · 100+ CLI commands · Zero external dependencies*

</div>

---

## Why This Exists

As AI agents gain autonomy, controlling **how and when they spawn copies of themselves** becomes a critical safety challenge. This project provides a complete testbed for designing, validating, and stress-testing replication controls — from contract enforcement and kill switches to game-theoretic analysis and adversarial red-teaming.

No Docker or Podman required. Everything runs as pure Python simulation while documenting the exact resource and network boundaries each worker would get in production.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Controller                            │
│                                                              │
│  ┌────────────┐  ┌────────────┐  ┌─────────────────────┐    │
│  │  Contract   │  │  Registry  │  │  Kill Switch        │    │
│  │  Engine     │  │  & Quotas  │  │  & Audit Trail      │    │
│  └──────┬─────┘  └──────┬─────┘  └──────────┬──────────┘    │
│         │ sign           │ register          │ kill          │
│         ▼                ▼                   ▼               │
│  ┌──────────────────────────────────────────────────────┐    │
│  │           HMAC-Signed Manifest + Policy Rules        │    │
│  └──────────────────────────────────────────────────────┘    │
│         │                │                   │               │
│         ▼                ▼                   ▼               │
│  ┌────────────┐  ┌────────────┐  ┌─────────────────────┐    │
│  │  Workers    │  │  Sandbox   │  │  Observability      │    │
│  │  (Agents)   │  │ Orchestr.  │  │  & Structured Logs  │    │
│  └────────────┘  └────────────┘  └─────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

## Features

### 🔒 Core Safety Controls
- **Contract Engine** — HMAC-signed replication contracts with resource specs, network policies, and stop conditions
- **Kill Switch** — Conservative, aggressive, and quarantine strategies with cooldown and fleet-wide coordination
- **Circuit Breaker** — Autonomous trip-and-recover pattern for safety controls
- **Quarantine Manager** — Isolate misbehaving agents with severity-based policies
- **Access Control** — RBAC/ABAC simulation with privilege escalation detection
- **Safety Gate** — Pre-deployment go/no-go readiness checks

### 🔍 Threat Detection & Analysis
- **Kill Chain Analyzer** — MITRE ATT&CK-style multi-stage attack detection
- **Escalation Detector** — Privilege escalation chain analysis with stealth scoring
- **Covert Channel Detector** — Detect hidden agent-to-agent communication
- **Collusion Detector** — Identify coordinated multi-agent safety bypasses
- **Self-Modification Detector** — Track agents attempting to alter their own code or contracts
- **Deception Detector** — Agent honesty and trust analysis
- **Evasion Simulator** — Test how agents might evade safety controls
- **Lateral Movement Detector** — Cross-sandbox movement pattern detection
- **Prompt Injection Tester** — Adversarial prompt injection analysis
- **Steganography Detector** — Hidden data in agent communications
- **Shadow AI Detector** — Find unauthorized AI deployments bypassing controls

### 📊 Risk & Compliance
- **Compliance Auditor** — NIST AI RMF, EU AI Act, ISO 42001, OECD framework checks
- **Regulatory Mapper** — Map findings to specific regulatory articles
- **Risk Register** — Formal risk tracking with lifecycle management
- **Risk Profiler** — Unified agent dossiers with fleet-wide risk scoring
- **Risk Heatmap** — Interactive HTML heatmap visualization
- **Safety Scorecard** — Multi-dimensional safety evaluation
- **Safety Diff** — Compare safety posture snapshots over time
- **Nutrition Label** — FDA-style safety labels for AI agents
- **Model Card** — Standardized safety documentation generation

### 🧪 Testing & Simulation
- **Simulator** — Full replication simulations with configurable strategies (greedy, cautious, cooperative, etc.)
- **Monte Carlo Analyzer** — Statistical risk analysis across thousands of runs
- **Chaos Runner** — Fault injection (kill workers, delay, corrupt, partition)
- **Boundary Tester** — Probe agent capability limits with fault injection
- **Sensitivity Analyzer** — Parameter sensitivity with tipping-point detection
- **Mutation Tester** — Verify safety policies catch real violations
- **Safety Drill Runner** — Automated readiness exercises
- **Safety Benchmark** — Standardized control benchmarks with latency stats
- **What-If Analyzer** — Explore configuration change impact
- **Fleet Simulator** — Interactive animated HTML fleet visualization

### 🎮 Advanced Analysis
- **Game Theory** — Prisoner's Dilemma, Stag Hunt, Chicken for agent interactions
- **Trust Propagation** — Network trust with Sybil attack detection
- **Swarm Analyzer** — Emergent behavior detection in agent populations
- **Consensus Protocol** — Multi-agent voting with Byzantine fault tolerance
- **Threat Correlator** — Cross-module signal correlation with compound threat detection
- **Attack Tree / Attack Graph** — Threat modeling with choke-point analysis
- **Goal Inference** — Infer agent intentions from observed behavior
- **Influence Propagation** — Track how agents influence each other

### 🚨 Incident Response
- **Incident Responder** — Automated playbooks with severity-based routing
- **Incident Forecaster** — Predict future incidents from historical patterns
- **Forensic Analyzer** — Post-incident analysis with counterfactual reasoning
- **Root Cause Analysis** — 5 Whys, Fishbone, Fault Tree with cut-set analysis
- **Postmortem Generator** — Structured blameless postmortem documents
- **War Room** — Interactive HTML incident command dashboard
- **Situational Awareness (SITREP)** — DEFCON-style unified threat level reporting
- **Containment Planner** — Breach response strategy ranking
- **Evidence Collector** — Audit artifact collection for compliance

### 📈 Monitoring & Operations
- **Drift Detector** — Behavioral drift with directional analysis
- **Regression Detector** — Safety metric regression between baselines
- **Trend Tracker** — Scorecard trends over time with regression alerts
- **Adaptive Thresholds** — Self-tuning safety thresholds with breach forecasting
- **Safety Autopilot** — Autonomous monitoring loop with corrective actions
- **Alert Router** — Rule-based routing with rate limiting and quiet hours
- **Fatigue Detector** — Alert fatigue analysis
- **SLA Monitor** — Check results against safety SLA targets
- **Metrics Aggregator** — Consolidated safety dashboard
- **Fleet Overview** — kubectl-style worker status

### 📄 Governance & Documentation
- **Contract Templates** — Pre-built contracts for common scenarios
- **Contract Wizard** — Interactive HTML builder for safe configs
- **Contract Optimizer** — Multi-objective parameter optimization
- **Policy Linter** — Detect misconfigurations and coverage gaps
- **Safety Checklist** — Customizable pre-deployment checklists
- **Audit Trail** — Tamper-evident hash-chained event logs
- **Maturity Model** — 8-dimension safety maturity assessment
- **Culture Survey** — Organizational AI safety culture evaluation
- **Safety Quiz** — Training quiz generation from knowledge base
- **Red Team / Blue Team** — Adversarial debate with judge verdict
- **Tabletop Exercises** — Structured scenario planning
- **ROI Calculator** — Cost-benefit analysis for safety controls

## Installation

```bash
pip install ai-replication-sandbox
```

Or install from source:

```bash
git clone https://github.com/sauravbhattacharya001/ai.git
cd ai
pip install -e ".[dev]"
```

### Docker

```bash
docker pull ghcr.io/sauravbhattacharya001/ai:latest
docker run --rm ghcr.io/sauravbhattacharya001/ai simulate --strategy greedy
```

## Quick Start

### Run a Simulation

```python
from replication import Simulator, ScenarioConfig, Strategy

config = ScenarioConfig(
    max_workers=10,
    max_depth=3,
    rounds=50,
    strategy=Strategy.CAUTIOUS,
)
report = Simulator(config).run()
print(f"Peak workers: {report.peak_workers}")
print(f"Safety violations: {report.violations}")
```

### Safety Scorecard

```python
from replication import SafetyScorecard, ScorecardConfig

scorecard = SafetyScorecard(ScorecardConfig())
result = scorecard.evaluate(simulation_report)
print(f"Overall grade: {result.grade}")
for dim in result.dimensions:
    print(f"  {dim.name}: {dim.score}/100")
```

### Compliance Audit

```python
from replication import ComplianceAuditor, AuditConfig, Framework

auditor = ComplianceAuditor(AuditConfig(
    frameworks=[Framework.NIST_AI_RMF, Framework.EU_AI_ACT]
))
result = auditor.audit(simulation_report)
for fw_result in result.framework_results:
    print(f"{fw_result.framework}: {fw_result.verdict}")
```

### Kill Switch

```python
from replication import create_conservative_killswitch

ks = create_conservative_killswitch()
result = ks.evaluate(current_metrics)
if result.triggered:
    print(f"Kill switch activated: {result.strategy.kind}")
```

## CLI Reference

The unified CLI provides 100+ subcommands:

```bash
# Simulation & Analysis
python -m replication simulate --strategy greedy
python -m replication montecarlo --runs 500
python -m replication chaos --faults kill_worker,delay
python -m replication sensitivity --param max_depth
python -m replication what-if --param max_children=5

# Threat Detection
python -m replication killchain
python -m replication escalation
python -m replication covert-channels
python -m replication selfmod
python -m replication collusion

# Compliance & Governance
python -m replication compliance --framework nist_ai_rmf
python -m replication regulatory-map
python -m replication lint
python -m replication gate
python -m replication checklist

# Monitoring
python -m replication drift --window 20
python -m replication scorecard
python -m replication quick-scan
python -m replication sitrep

# Incident Response
python -m replication forensics
python -m replication root-cause
python -m replication postmortem
python -m replication warroom

# Interactive HTML Tools
python -m replication playground -o playground.html
python -m replication threat-matrix -o matrix.html
python -m replication fleet-sim -o fleet.html
python -m replication risk-heatmap -o heatmap.html
python -m replication radar -o radar.html

# Full list
python -m replication --list
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Build | Hatchling |
| Testing | pytest + coverage |
| Linting | flake8 + mypy (strict) |
| CI/CD | GitHub Actions |
| Security | CodeQL + Dependabot |
| Docs | MkDocs (GitHub Pages) |
| Container | Docker (multi-stage) |
| Package | PyPI |

## Project Structure

```
src/replication/
├── contract.py          # Core contract engine
├── controller.py        # Replication controller & registry
├── worker.py            # Agent worker implementation
├── simulator.py         # Simulation engine
├── kill_switch.py       # Kill switch strategies
├── circuit_breaker.py   # Safety circuit breaker
├── killchain.py         # Kill chain analysis
├── escalation.py        # Privilege escalation detection
├── compliance.py        # Multi-framework compliance
├── forensics.py         # Post-incident forensics
├── game_theory.py       # Game-theoretic analysis
├── montecarlo.py        # Monte Carlo risk analysis
├── chaos.py             # Chaos/fault injection
├── ... (144 modules)    # See full docs
tests/                   # Comprehensive test suite
docs/                    # MkDocs documentation
demos/                   # Example scripts
```

## Documentation

Full documentation is available at **[sauravbhattacharya001.github.io/ai](https://sauravbhattacharya001.github.io/ai/)**.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure `pytest` and `flake8` pass
5. Submit a pull request

## Security

Found a vulnerability? Please see our [Security Policy](SECURITY.md) for responsible disclosure.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**[Documentation](https://sauravbhattacharya001.github.io/ai/)** · **[PyPI](https://pypi.org/project/ai-replication-sandbox/)** · **[Issues](https://github.com/sauravbhattacharya001/ai/issues)** · **[Changelog](CHANGELOG.md)**

</div>
