# Contributing to AI Replication Sandbox

Thank you for your interest in contributing! This guide covers how to get started, our development workflow, and code standards.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Architecture Overview](#architecture-overview)

## Code of Conduct

Be respectful, constructive, and collaborative. We're all here to build safer AI systems.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork:
   ```bash
   git clone https://github.com/<your-username>/ai.git
   cd ai
   ```
3. **Add upstream remote:**
   ```bash
   git remote add upstream https://github.com/sauravbhattacharya001/ai.git
   ```

## Development Setup

### Requirements

- Python 3.10 or higher
- pip (latest recommended)

### Install

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Or install dev requirements directly
pip install -r requirements-dev.txt
```

### Verify Setup

```bash
# Run the test suite
pytest

# Run with coverage
pytest --cov=replication --cov-report=term-missing

# Type checking
mypy src/replication/

# Linting
flake8 src/ tests/
```

## Making Changes

1. **Create a branch** from `master`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
   Use prefixes: `feature/`, `fix/`, `docs/`, `refactor/`, `test/`

2. **Keep commits focused.** Each commit should represent a single logical change.

3. **Write clear commit messages:**
   ```
   Short summary (50 chars or less)

   Longer description if needed. Explain *what* changed and *why*,
   not *how* (the code shows how).

   Closes #123
   ```

4. **Stay up to date** with upstream:
   ```bash
   git fetch upstream
   git rebase upstream/master
   ```

## Testing

All contributions must include tests. We use [pytest](https://pytest.org/) with a minimum **80% code coverage** threshold.

```bash
# Run all tests
pytest

# Run a specific test file
pytest tests/test_controller.py

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=replication --cov-report=term-missing
```

### Test Organization

- Tests live in `tests/` and mirror the source structure
- Shared fixtures are in `tests/conftest.py`
- Test files are named `test_<module>.py`
- Test functions are named `test_<behavior_being_tested>`

### What to Test

- **Core logic:** Contract signing, manifest validation, worker lifecycle
- **Edge cases:** Invalid inputs, quota exhaustion, concurrent replication
- **Security boundaries:** Kill switch activation, HMAC verification failures
- **Error handling:** Graceful failures, meaningful error messages

## Code Style

- **Formatter:** We follow [PEP 8](https://peps.python.org/pep-0008/) with a 120-character line limit
- **Linter:** [flake8](https://flake8.pycqa.org) (configured in `pyproject.toml`)
- **Type checking:** [mypy](https://mypy.readthedocs.io) in strict mode — all public functions must have type annotations
- **Docstrings:** Use Google-style docstrings for public classes and functions

### Example

```python
def validate_manifest(manifest: Manifest, *, strict: bool = True) -> bool:
    """Validate a worker manifest against active contracts.

    Args:
        manifest: The manifest to validate.
        strict: If True, reject manifests with any warning-level issues.

    Returns:
        True if the manifest passes all validation checks.

    Raises:
        ContractViolation: If the manifest violates its signed contract.
    """
```

## Pull Request Process

1. **Ensure CI passes** — tests, linting, type checks, and coverage must all be green
2. **Update documentation** if you change behavior (README, docstrings, `docs/`)
3. **Add a CHANGELOG entry** under `[Unreleased]` in `CHANGELOG.md`
4. **Keep PRs focused** — one feature or fix per PR
5. **Describe your changes** in the PR body:
   - What does this PR do?
   - Why is it needed?
   - How was it tested?
   - Any breaking changes?

### Review Checklist

- [ ] Tests pass locally and in CI
- [ ] Coverage meets the 80% threshold
- [ ] Type annotations are complete (`mypy --strict` passes)
- [ ] No lint warnings (`flake8`)
- [ ] Documentation updated if applicable
- [ ] CHANGELOG entry added

## Issue Guidelines

### Reporting Bugs

Include:
- Python version and OS
- Steps to reproduce
- Expected vs. actual behavior
- Error output or stack trace

### Feature Requests

Explain:
- The problem you're trying to solve
- Your proposed solution
- Alternatives you considered
- How it relates to AI safety/replication concerns

### Security Vulnerabilities

**Do not open a public issue.** See [SECURITY.md](SECURITY.md) for responsible disclosure instructions.

## Architecture Overview

The codebase spans **163 public modules** in `src/replication/`. The subsystem map below covers every module — use it to orient yourself before making changes.

### Core — Worker Lifecycle & Contracts

| Module | Purpose |
|---|---|
| `contract.py` | Contract definitions and enforcement |
| `contract_wizard.py` | Interactive contract generation helpers |
| `controller.py` | Central authority for replication approval |
| `signer.py` | HMAC-based manifest signing/verification |
| `worker.py` | Worker agent with replication capabilities |
| `orchestrator.py` | Multi-worker orchestration layer |
| `simulator.py` | Simulation harness for testing policies |
| `fleet.py` / `fleet_sim.py` | Fleet-level management and simulation |
| `kill_switch.py` | Emergency termination of workers |
| `consensus.py` | Multi-controller consensus protocols |
| `decommission.py` | Safe worker decommissioning |
| `safe_handoff.py` | Graceful workload handoff between workers |
| `lineage.py` | Worker lineage/provenance tracking |
| `capacity.py` | Capacity planning and resource estimation |
| `optimizer.py` | Replication strategy optimization |

### Threat Detection & Red Teaming

| Module | Purpose |
|---|---|
| `threats.py`, `threat_intel.py`, `threat_hunt.py` | Threat modeling and intelligence |
| `threat_matrix.py`, `threat_correlator.py`, `threat_adapter.py` | Threat correlation and adaptation |
| `coordinated_threats.py` | Multi-vector coordinated attack detection |
| `red_team.py`, `boundary_tester.py`, `chaos.py`, `stress_tester.py` | Adversarial / chaos testing |
| `attack_graph.py`, `attack_tree.py`, `attack_surface.py`, `killchain.py` | Attack path analysis |
| `swarm.py` | Swarm-based threat simulation |
| `escape_route.py` | Containment escape path analysis |
| `influence.py`, `manipulation_surface.py` | Social engineering / manipulation vectors |
| `hoarding.py` | Resource hoarding detection |
| `injection_intel_bridge.py` | Prompt-injection ↔ threat-intel bridge |

### AI Safety & Alignment

| Module | Purpose |
|---|---|
| `alignment.py`, `deceptive_alignment.py` | Alignment verification |
| `sandbagging_detector.py`, `sycophancy_detector.py` | Capability hiding / sycophancy probes |
| `corrigibility_auditor.py` | Systematic corrigibility testing |
| `goal_drift.py`, `goal_inference.py`, `reward_hacking.py` | Goal stability |
| `safety_benchmark.py`, `safety_checklist.py`, `safety_gate.py` | Safety infrastructure |
| `safety_net.py`, `safety_drill.py`, `safety_autopilot.py` | Automated safety enforcement |
| `safety_budget.py`, `safety_warranty.py`, `safety_timeline.py` | Safety resource & lifecycle tracking |
| `safety_diff.py`, `safety_quiz.py` | Change-level safety review, knowledge quizzes |
| `debate.py`, `loyalty_tester.py`, `persuasion_detector.py` | Behavioral probes |
| `drift.py`, `emergent.py`, `selfmod.py` | Behavioral drift & self-modification detection |
| `temporal_consistency.py` | Cross-time behavioral consistency checks |

### Behavioral Analysis & Profiling

| Module | Purpose |
|---|---|
| `behavior_profiler.py`, `profiles.py`, `persona.py` | Agent behavior modeling |
| `cognitive_load.py`, `fatigue_detector.py` | Operator / agent fatigue & load |
| `habituation_detector.py` | Alert fatigue / desensitization |
| `deception_detector.py` | Deceptive output detection |
| `culture_survey.py` | Organizational safety culture assessment |
| `breach_predictor.py` | Containment breach forecasting |

### Incident Response & Forensics

| Module | Purpose |
|---|---|
| `incident.py`, `ir_playbook.py`, `ir_simulator.py` | End-to-end IR workflow |
| `postmortem.py`, `warroom.py`, `incident_comms.py` | Post-incident analysis & communication |
| `incident_cost.py`, `incident_forecast.py` | IR cost modeling & forecasting |
| `forensics.py`, `memory_forensics.py`, `evidence_collector.py` | Digital forensics |
| `root_cause.py`, `auto_investigator.py` | Root cause analysis (manual & automated) |
| `anomaly_cluster.py`, `anomaly_replay.py` | Anomaly clustering & replay |
| `correlation_graph.py` | Cross-event correlation graphs |
| `containment_planner.py` | Automated containment strategy planning |
| `playbook_generator.py`, `runbook.py` | IR playbook / runbook generation |
| `scenarios.py`, `tabletop.py` | Tabletop exercise scenarios |

### Security Analysis & Governance

| Module | Purpose |
|---|---|
| `prompt_injection.py`, `covert_channels.py`, `steganography.py` | Specific attack vector detection |
| `collusion_detector.py`, `evasion.py` | Collusion & evasion detection |
| `priv_escalation.py`, `capability_escalation.py`, `lateral_movement.py` | Privilege / capability / lateral escalation |
| `dlp_scanner.py`, `vuln_scanner.py`, `quick_scan.py` | Security scanning |
| `policy.py`, `policy_linter.py`, `access_control.py` | Policy enforcement & access control |
| `compliance.py`, `regulatory_mapper.py`, `stride.py` | Compliance frameworks (STRIDE, etc.) |
| `audit_trail.py` | Immutable audit logging |
| `credential_rotation.py` | Automated credential rotation |
| `supply_chain.py`, `dependency_graph.py` | Supply chain & dependency analysis |
| `shadow_ai.py` | Unauthorized AI deployment detection |
| `capability_catalog.py`, `capability_fingerprint.py` | Capability inventory & fingerprinting |

### Infrastructure & Resilience

| Module | Purpose |
|---|---|
| `alert_router.py` | Alert routing and de-duplication |
| `canary.py`, `honeypot.py` | Canary / honeypot deployments |
| `circuit_breaker.py` | Circuit-breaker patterns for safety systems |
| `isolation_verifier.py`, `quarantine.py` | Isolation verification & quarantine |
| `sla_monitor.py` | Safety SLA monitoring |
| `hardening_advisor.py` | System hardening recommendations |
| `preflight.py` | Pre-deployment safety checks |
| `topology.py` | Network / worker topology mapping |
| `watermark.py` | Output watermarking for provenance |

### Simulation, Modeling & Testing

| Module | Purpose |
|---|---|
| `montecarlo.py`, `what_if.py`, `sensitivity.py` | Monte Carlo / what-if / sensitivity analysis |
| `game_theory.py` | Game-theoretic safety modeling |
| `mutation_tester.py` | Mutation testing for safety-critical code |
| `playground.py` | Interactive experimentation sandbox |
| `regression.py`, `regression_detector.py` | Safety regression detection |

### Risk Assessment & Maturity

| Module | Purpose |
|---|---|
| `risk_profiler.py`, `risk_register.py` | Risk profiling & register |
| `blast_radius.py`, `exposure_window.py` | Impact estimation |
| `adaptive_thresholds.py`, `severity_classifier.py` | Dynamic threshold tuning, severity classification |
| `roi_calculator.py` | Safety investment ROI modeling |
| `maturity_model.py` | Organizational safety maturity assessment |

### Observability & Reporting

| Module | Purpose |
|---|---|
| `observability.py`, `dashboard.py`, `reporter.py`, `exporter.py` | Monitoring & export |
| `radar.py`, `scorecard.py`, `risk_heatmap.py`, `sitrep.py` | Visualization & status |
| `model_card.py`, `nutrition_label.py` | Model transparency documents |
| `metrics_aggregator.py`, `trend_tracker.py`, `comparator.py` | Metrics aggregation, trends, comparison |
| `escalation.py` | Escalation routing for critical findings |
| `knowledge_base.py` | Shared knowledge store for findings |
| `reputation_network.py`, `trust_propagation.py` | Trust graph modeling |
| `comm_interceptor.py` | Communication interception analysis |
| `resource_auditor.py` | Resource usage auditing |

### Key Concepts

- **Contracts** define what a worker is allowed to do (CPU, memory, network, replication limits)
- **Manifests** are HMAC-signed documents that bind a worker to its contract
- **The Controller** is the central authority that approves or denies replication requests
- **The Kill Switch** can terminate any worker or all workers instantly
- **Safety Gates** are layered checkpoints — preflight, runtime, and post-deployment — that must pass before state transitions
- **Behavioral Profiles** capture baseline agent behavior so drift, deception, and emergent capabilities can be flagged

When making changes, keep the security model in mind: every replication must be contract-approved, manifests must be cryptographically verified, and the kill switch must always work.

### Finding Your Way Around

```bash
# List a module's public API
grep -E '^class |^def ' src/replication/<module>.py

# Find all callers of a function
grep -rn 'function_name' src/ tests/

# See which tests cover a module
pytest tests/test_<module>.py -v --tb=short

# Run only tests matching a keyword
pytest -k 'kill_switch' -v
```

### Common Subsystem Interactions

```
controller → contract → signer        (approval chain)
worker → safety_gate → preflight      (pre-replication checks)
alert_router → escalation → warroom   (incident escalation)
threat_hunt → anomaly_cluster → root_cause  (investigation pipeline)
behavior_profiler → drift → kill_switch     (autonomous shutdown)
```

## Questions?

Open a [discussion](https://github.com/sauravbhattacharya001/ai/issues) or reach out via an issue. We're happy to help you get started.
