# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] — 2026-02-14

### Initial Release

A replication-aware worker system with explicit contracts, sandboxed orchestration, and structured observability. Designed for studying and testing AI agent replication policies in a controlled environment.

### Features

- **Replication Contract** — Configurable policy with `max_depth`, `max_replicas`, `cooldown_seconds`, optional `expiration_seconds`, and pluggable stop conditions evaluated during worker registration.
- **Manifest Signing** — HMAC-SHA256 signed manifests that capture cloned state, resource quotas, parent/child lineage, and issuance time. Tamper-proof verification on registration.
- **Controller** — Central authority that issues/signs manifests, enforces depth limits, replica quotas, cooldown periods, and stop conditions. Includes kill switch, audit trail, and stale worker reaping.
- **Sandbox Orchestrator** — Simulates isolated container environments with CPU/RAM limits and network policy (egress-only to controller). Records lifecycle events and emits resource metrics.
- **Worker** — Task-executing agent that registers with the controller, heartbeats, handles expiration, and optionally self-replicates subject to the contract.
- **Observability** — Structured in-memory logger with audit trail support and metrics emission for lifecycle events, replication decisions, and resource enforcement.

### Infrastructure

- GitHub Actions CI with flake8 linting, mypy type checking, and pytest across Python 3.10/3.11/3.12.
- Multi-stage Dockerfile for containerized deployment.
- Comprehensive test suite covering replication depth, quota enforcement, stop conditions, kill switch, heartbeat reaping, and chaos scenarios.
