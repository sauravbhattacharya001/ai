# Threat Model & Limitations

This page documents the threat model the AI Replication Sandbox is designed around, the attacks it mitigates, and the known limitations of the current implementation.

## Threat Model

The sandbox assumes a **semi-trusted agent** environment: agents may attempt to exceed their replication budget, tamper with manifests, or evade shutdown — but they operate within a Python process (not a fully adversarial attacker with kernel access).

### In Scope

| Threat | Mitigation |
|--------|-----------|
| **Unauthorized replication** | Contract enforcement via Controller (depth, quota, cooldown, stop conditions) |
| **Manifest tampering** | HMAC-SHA256 signature verification on every registration |
| **Runaway replication chains** | `max_depth` hard limit on replication generations |
| **Resource exhaustion** | `max_replicas` quota + per-worker `ResourceSpec` (CPU, RAM) |
| **Burst replication** | `cooldown_seconds` prevents rapid-fire spawning from a single parent |
| **Zombie workers** | Stale worker reaping via heartbeat timeout (`reap_stale_workers`) |
| **Data exfiltration** | `NetworkPolicy` restricts egress to controller-only by default |
| **Loss of control** | Kill switch for emergency termination of all workers |
| **Audit gaps** | Structured logging with dedicated audit trail for all security-sensitive decisions |

### Out of Scope

These threats are **not addressed** by the current implementation:

| Threat | Why |
|--------|-----|
| **OS-level sandbox escape** | The orchestrator is a simulator — no real container isolation is enforced |
| **Side-channel attacks** | No timing or cache side-channel protections |
| **Denial-of-service on the Controller** | The Controller runs in-process with no rate limiting beyond cooldowns |
| **Secret key compromise** | If the HMAC key is leaked, all manifests can be forged |
| **Multi-tenant isolation** | Single Controller instance — no namespace separation between experiments |
| **Persistent storage attacks** | Workers operate in-memory only; no disk I/O is modeled |

## Security Assumptions

1. **The Controller is trusted.** It holds the signing key and is the sole authority for manifest issuance. A compromised Controller invalidates all guarantees.

2. **HMAC key is secret.** The `ManifestSigner` uses a symmetric key. Any entity with the key can forge valid manifests. In production, consider asymmetric signing or HSM-backed keys.

3. **Workers run in a cooperative runtime.** The current implementation is a Python simulation. Workers share the same process and can technically bypass the orchestrator. Real deployment would require OS-level containerization (Docker, gVisor, Firecracker).

4. **Time is monotonic.** Cooldown and expiration checks use `datetime.now(timezone.utc)`. Clock manipulation could bypass time-based controls. Consider monotonic clocks for production.

5. **Network policies are declarative.** The `NetworkPolicy` dataclass records *intended* restrictions but does not enforce them at the network layer. Real enforcement requires iptables/nftables/CNI plugins.

## Known Limitations

### Simulation vs. Production

The `SandboxOrchestrator` **simulates** container lifecycle but does not:

- Create real containers or cgroups
- Enforce CPU/RAM limits at the OS level
- Apply network egress rules
- Isolate filesystem access between workers

This is by design — the project is a **testbed for replication policy logic**, not a production container orchestrator. The architecture separates policy (Controller + Contract) from enforcement (Orchestrator) so that the Orchestrator can be swapped for a real implementation (Docker SDK, Kubernetes CRDs) without changing the policy layer.

### Single-Process Model

All components (Controller, Workers, Orchestrator, Logger) run in a single Python process. This means:

- A misbehaving worker can crash the entire system
- Memory is shared — workers can technically read each other's state
- No true concurrency model (workers execute sequentially unless explicitly threaded)

### In-Memory State

The `StructuredLogger` and Controller registry are in-memory only. If the process crashes:

- All audit trail data is lost
- Active worker state cannot be recovered
- No persistence layer for forensic analysis

For production use, the logger should be backed by a durable store (e.g., SQLite, PostgreSQL, or an append-only log).

### HMAC Signing Limitations

- **Symmetric key**: Anyone with the key can sign. Consider Ed25519 or RSA for environments where workers should not be able to forge manifests.
- **No key rotation**: The signer has no mechanism for rotating keys without restarting the system.
- **No revocation**: Once a manifest is signed, it cannot be revoked — only the worker can be deregistered.

### Stop Condition Evaluation

Stop conditions are evaluated **only during worker registration**, not continuously. A condition that becomes true after a worker registers (e.g., resource budget exceeded) will not retroactively block that worker. Consider periodic re-evaluation for stricter enforcement.

## Extending the Threat Model

The architecture is designed for extensibility:

- **Swap the signer**: Replace `ManifestSigner` with an asymmetric or HSM-backed implementation
- **Swap the orchestrator**: Implement `launch_worker` / `kill_worker` using Docker SDK or Kubernetes API
- **Add persistence**: Wrap `StructuredLogger` with a database-backed adapter
- **Add continuous enforcement**: Run a background loop that re-evaluates stop conditions against active workers
- **Add mutual TLS**: Secure Controller-Worker communication in distributed deployments

## Responsible Use

This project is intended for **research and education**. Self-replication in AI systems is a topic with significant safety implications. Users should:

- Never deploy self-replicating agents outside sandboxed environments without thorough review
- Treat the contract as a starting point — production systems need defense in depth
- Report security issues via GitHub Security Advisories, not public issues
