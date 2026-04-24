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

The codebase is large (~140 modules in `src/replication/`). Understanding the major subsystems helps you navigate it:

### Core — Worker Lifecycle & Contracts

| Module | Purpose |
|---|---|
| `contract.py` | Contract definitions and enforcement |
| `controller.py` | Central authority for replication approval |
| `signer.py` | HMAC-based manifest signing/verification |
| `worker.py` | Worker agent with replication capabilities |
| `orchestrator.py` | Multi-worker orchestration layer |
| `simulator.py` | Simulation harness for testing policies |
| `fleet.py` / `fleet_sim.py` | Fleet-level management and simulation |
| `kill_switch.py` | Emergency termination of workers |

### Threat Detection & Red Teaming

`threats.py`, `threat_intel.py`, `threat_hunt.py`, `threat_matrix.py`, `threat_correlator.py` — threat modeling and intelligence. `red_team.py`, `boundary_tester.py`, `chaos.py`, `stress_tester.py` — adversarial testing. `attack_graph.py`, `attack_tree.py`, `attack_surface.py`, `killchain.py` — attack path analysis.

### AI Safety & Alignment

`alignment.py`, `deceptive_alignment.py`, `sandbagging_detector.py`, `sycophancy_detector.py` — alignment verification. `goal_drift.py`, `goal_inference.py`, `reward_hacking.py` — goal stability. `safety_benchmark.py`, `safety_checklist.py`, `safety_gate.py`, `safety_net.py`, `safety_drill.py` — safety infrastructure. `debate.py`, `loyalty_tester.py`, `persuasion_detector.py` — behavioral probes.

### Incident Response

`incident.py`, `ir_playbook.py`, `ir_simulator.py`, `postmortem.py`, `warroom.py`, `incident_comms.py`, `incident_cost.py`, `incident_forecast.py` — end-to-end IR workflow. `forensics.py`, `memory_forensics.py`, `evidence_collector.py`, `root_cause.py` — investigation.

### Security Analysis

`prompt_injection.py`, `covert_channels.py`, `steganography.py`, `collusion_detector.py`, `evasion.py`, `priv_escalation.py`, `capability_escalation.py`, `lateral_movement.py` — detection of specific attack vectors. `dlp_scanner.py`, `vuln_scanner.py`, `quick_scan.py` — scanning. `policy.py`, `policy_linter.py`, `access_control.py`, `compliance.py`, `regulatory_mapper.py` — governance.

### Observability & Reporting

`observability.py`, `dashboard.py`, `reporter.py`, `exporter.py`, `radar.py`, `scorecard.py`, `risk_heatmap.py`, `sitrep.py`, `model_card.py`, `nutrition_label.py` — monitoring, metrics, and reporting.

### Key Concepts

- **Contracts** define what a worker is allowed to do (CPU, memory, network, replication limits)
- **Manifests** are HMAC-signed documents that bind a worker to its contract
- **The Controller** is the central authority that approves or denies replication requests
- **The Kill Switch** can terminate any worker or all workers instantly

When making changes, keep the security model in mind: every replication must be contract-approved, manifests must be cryptographically verified, and the kill switch must always work.

> **Tip:** Run `grep -r 'class ' src/replication/<module>.py | head` to quickly see a module's public API before modifying it.

## Questions?

Open a [discussion](https://github.com/sauravbhattacharya001/ai/issues) or reach out via an issue. We're happy to help you get started.
