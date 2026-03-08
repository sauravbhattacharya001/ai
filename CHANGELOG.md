# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - 2026-03-08

### Added

- **Canary Token Detector** — Plant honey tokens (API keys, DB strings, secrets, documents, DNS, webhooks) to detect agent data exfiltration. 5-stage detection pipeline (access → copy → transmit → decode → mutate), per-agent risk profiling, comprehensive reporting.
- **Attack Tree Generator** — Structured threat modeling with attack trees: enumerate attack paths, compute risk scores, find critical nodes.
- **What-If Analysis Engine** — Hypothetical config change analysis: simulate parameter sweeps (e.g. max_depth 1→10), compare safety metrics before/after.
- **Agent Prompt Injection Detector** — Detect prompt injection attempts in agent I/O with pattern matching, semantic analysis, and threat intel feed integration.
- **Agent Honeypot System** — Deploy decoy resources (files, endpoints, services) to detect proactive agent reconnaissance and lateral movement.
- **Agent Threat Intelligence Feed** — IOC aggregation, correlation, alerting, and STIX export for agent threat indicators.
- **Agent Output Steganography Detector** — Detect covert channels in agent output (hidden text, Unicode tricks, statistical anomalies).
- **Print Parameter Optimizer** — Multi-objective bioprint tuning with Pareto optimization and sensitivity analysis.
- **Kill Chain Analyzer** — Multi-stage agent attack detection following MITRE-style kill chain phases.
- **Emergent Behavior Detector** — Detect collective emergent patterns across multi-agent systems (synchronization, specialization, coalition formation).
- **Resource Dependency Analyzer** — Model inter-agent dependencies and analyze cascade failure risk.
- **Behavior Profiler** — Build behavioral baselines for agents and detect operational anomalies.
- **Trust Propagation Network** — Agent trust network analysis with Sybil, collusion, and eclipse attack detection (38 tests).
- **Threat Correlator** — Cross-module signal correlation for compound threat detection: combine drift + escalation + canary signals into unified threat assessments.
- **Unified CLI** — `python -m replication` entry point with subcommands for all modules.

### Changed

- **CoordinatedThreatSimulator** extracted from `threats.py` into standalone `coordinated_threats.py` for maintainability (backward-compatible re-exports).
- **36 unused imports removed** across 18 modules (code cleanup).
- **Duplicated helpers consolidated** into shared `_helpers` module.

### Fixed

- **Consensus approval threshold** — Changed `>` to `>=` in tally check; proposals meeting exactly the threshold now correctly pass.
- **Worker deregistration cleanup** — `spawn_timestamps` and `quarantine` set now properly cleaned up when workers deregister.
- **Optimizer refinement** — Fixed cooldown dimension being ignored during refinement, and added O(n) dedup to eliminate redundant evaluations.
- **Voting bloc detection** — Replaced order-dependent greedy algorithm with Bron-Kerbosch clique enumeration for correct maximal clique detection (closes #28).

### Performance

- **ForensicAnalyzer** — Eliminated O(n²) scans by using index-based lookups.
- **Covert channel analysis** — Eliminated O(n) linear lookups with hash-based membership checks.
- **`to_dot()` optimization** — Hoisted `_find_spofs()` out of per-resource loop (was O(R×N) → now O(N)).

### Security

- **Canary token CSPRNG** (CWE-330) — Token values (`_rand_hex`, `_rand_alnum`) now use `secrets` module (CSPRNG) instead of `random.Random` (Mersenne Twister). Prevents attackers from predicting canary token values via PRNG state recovery. Simulation reproducibility preserved (agent behavior still uses seeded PRNG).
- **Prompt injection detector input bounds** — Added size limits and finding count caps to prevent DoS via oversized inputs.
- **Filesystem escalation URL decoding** — Decode URL-encoded paths (`%2e%2e/`) in all escalation detection rules to prevent bypass.

### Documentation

- API reference added for 17 previously undocumented modules.
- README updated with 9 new module descriptions and Watermark API tables.

### Tests

- 2958 tests total (up from ~2400 in v1.1.0).
- 63 new SelfMod detector tests, 61 Comparator tests, 66 Incident Response tests.
- Comprehensive compliance auditor coverage (12 → 44 tests).

## [1.1.0] - 2026-03-04

### Added

- **Privilege Escalation Detector** - Detects agent attempts to access system paths, admin endpoints, or gain elevated privileges beyond their contract scope.
- **Monte Carlo performance optimization** - `MetricDistribution` now lazily caches sorted values, eliminating redundant O(n log n) sorts on repeated percentile/median lookups (7× fewer sorts in `to_dict()`).

### Security

- **Escalation detector hardening** - Fixed path traversal bypass (e.g. `/../root`), case-insensitive hostname matching, and URL-encoded payloads that could evade detection.
- **Root depth spoofing prevention** - Controller now denies registration with spoofed root depth instead of silently clamping, preventing agents from bypassing depth limits.
- **Consensus re-tally tampering** - Fixed vulnerability where agents could manipulate consensus results by re-triggering tally after votes were cast.

### Documentation

- Added privilege escalation detection guide with examples and configuration.
- Updated project structure and API reference in README.

## [1.0.0] - 2026-02-14

### Initial Release

A replication-aware worker system with explicit contracts, sandboxed orchestration, and structured observability. Designed for studying and testing AI agent replication policies in a controlled environment.

### Features

- **Replication Contract** - Configurable policy with `max_depth`, `max_replicas`, `cooldown_seconds`, optional `expiration_seconds`, and pluggable stop conditions evaluated during worker registration.
- **Manifest Signing** - HMAC-SHA256 signed manifests that capture cloned state, resource quotas, parent/child lineage, and issuance time. Tamper-proof verification on registration.
- **Controller** - Central authority that issues/signs manifests, enforces depth limits, replica quotas, cooldown periods, and stop conditions. Includes kill switch, audit trail, and stale worker reaping.
- **Sandbox Orchestrator** - Simulates isolated container environments with CPU/RAM limits and network policy (egress-only to controller). Records lifecycle events and emits resource metrics.
- **Worker** - Task-executing agent that registers with the controller, heartbeats, handles expiration, and optionally self-replicates subject to the contract.
- **Observability** - Structured in-memory logger with audit trail support and metrics emission for lifecycle events, replication decisions, and resource enforcement.

### Infrastructure

- GitHub Actions CI with flake8 linting, mypy type checking, and pytest across Python 3.10/3.11/3.12.
- Multi-stage Dockerfile for containerized deployment.
- Comprehensive test suite covering replication depth, quota enforcement, stop conditions, kill switch, heartbeat reaping, and chaos scenarios.
