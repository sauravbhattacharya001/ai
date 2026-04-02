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
[![Last Commit](https://img.shields.io/github/last-commit/sauravbhattacharya001/ai?logo=github)](https://github.com/sauravbhattacharya001/ai/commits)
[![Repo Size](https://img.shields.io/github/repo-size/sauravbhattacharya001/ai?logo=github)](https://github.com/sauravbhattacharya001/ai)
[![Security Policy](https://img.shields.io/badge/security-policy-green?logo=shield)](SECURITY.md)
[![Dependabot](https://img.shields.io/badge/dependabot-enabled-025E8C?logo=dependabot)](https://github.com/sauravbhattacharya001/ai/network/updates)

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
│  │              Manifest (HMAC-signed)          │
```

## Getting Started

### Installation

```bash
pip install ai-replication-sandbox
```

### Quick Start

```python
from ai_sandbox import Sandbox

# Initialize a controlled environment
sandbox = Sandbox(quota={"max_children": 2, "max_depth": 1})

# Run an agent under a replication contract
sandbox.run("agent_script.py")
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.