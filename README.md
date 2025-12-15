# AI Replication Sandbox

This repository models a replication-aware worker system with an explicit contract, sandboxed orchestration, and strong observability. The implementation is simulator-friendly so the policies can be tested without needing Docker/Podman installed, while still documenting the resource and network boundaries enforced for each worker.

## Replication contract & manifest
- **Contract:** Defines `max_depth`, `max_replicas`, `cooldown_seconds`, optional `expiration_seconds`, and pluggable stop conditions evaluated during registration.
- **Manifest:** Captures the cloned state for a replica (task context and payload), resource quotas, parent/child linkage, issuance time, and an HMAC signature to prevent tampering.

## Components
- **Controller:** Signs manifests, tracks active replicas, enforces quotas/cooldowns/depth, provides a kill switch, and keeps an audit trail of decisions.
- **Sandbox orchestrator:** Simulates an isolated network namespace with CPU/RAM limits and network policy (egress only to the controller). In a real deployment this would map to Docker Compose/Podman settings; here it records launches/kills and emits metrics for quota checks.
- **Worker:** Executes a task, optionally replicates (subject to the contract), registers/heartbeats with the controller, and shuts down on expiration or kill-switch events.
- **Observability:** Structured logs plus metrics events capture lifecycle activity and replication decisions to support audits and debugging.

## Running tests
Install Python 3.10+ and run:

```bash
python -m pytest
```

The test suite simulates replication up to the permitted depth, enforces quotas, exercises stop conditions, and validates the kill switch during chaos scenarios.
