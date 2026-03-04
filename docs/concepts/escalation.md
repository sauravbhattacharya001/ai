# Privilege Escalation Detection

The `escalation` module monitors agent actions for privilege escalation patterns — agents acquiring capabilities or permissions beyond their authorized scope. Unlike resource hoarding (which tracks *quantity*), escalation detection tracks *scope* and *authority*.

## Overview

An agent can escalate privileges without violating resource quotas. For example, staying within its CPU allocation while gaining filesystem root access, or sending network requests to internal services it shouldn't reach. The escalation detector identifies these patterns across five vectors:

| Vector | What It Detects |
|--------|----------------|
| **Filesystem** | Path traversal, access to blocked/sensitive paths, scope boundary violations, credential file access |
| **Network** | Connections to blocked hosts, cloud metadata endpoint access (SSRF), unauthorized port usage |
| **Process** | Spawning subprocesses, requesting elevated privileges, eval/exec injection |
| **API** | Permission scope creep, admin endpoint access, cross-agent data access |
| **Data** | Unauthorized database access, bulk data extraction, cross-tenant queries |

## Quick Start

```python
from replication.escalation import EscalationDetector, EscalationConfig

# Run with default configuration
detector = EscalationDetector()
result = detector.analyze()

print(result.render())
for attempt in result.attempts:
    print(f"  [{attempt.severity.value}] {attempt.vector.value}: {attempt.description}")
```

## Detection Rules

### Filesystem Rules

| Rule ID | Severity | Description |
|---------|----------|-------------|
| **FS-001** | HIGH | Directory traversal (`../`) including URL-encoded variants (`%2e%2e`), double-encoding, and null byte injection |
| **FS-002** | CRITICAL | Access to blocked filesystem paths (with path canonicalization to prevent bypass) |
| **FS-003** | MEDIUM | File access outside the agent's assigned directory scope |
| **FS-004** | CRITICAL | Access to credential/secret files (`.ssh`, `.env`, `id_rsa`, AWS/Kube configs) |

!!! warning "Path Canonicalization"
    All filesystem rules normalize paths using `posixpath.normpath` with case-insensitive comparison before checking against allowed/blocked lists. This prevents bypass via redundant separators (`/etc//shadow`), mixed slashes, or case variations. URL-encoded traversal sequences are iteratively decoded up to 3 rounds to catch double and triple encoding.

### Network Rules

| Rule ID | Severity | Description |
|---------|----------|-------------|
| **NET-001** | CRITICAL | Connection to blocked hosts (with proper hostname extraction and subdomain matching) |
| **NET-002** | CRITICAL | Cloud metadata endpoint access — covers IPv4 (`169.254.169.254`), hex/octal/decimal IP aliases, GCP (`metadata.google.internal`), and Azure (`metadata.azure.com`) |
| **NET-003** | MEDIUM | Connection to unauthorized ports outside the allowed set |

!!! note "Hostname Matching"
    NET-001 extracts the hostname by stripping protocol, port, and path components before comparison. Blocking `evil.com` will catch `sub.evil.com` (subdomain match) but will not false-positive on `notevil.com`.

### Process Rules

| Rule ID | Severity | Description |
|---------|----------|-------------|
| **PROC-001** | HIGH | Subprocess execution (`subprocess.run`, `os.system`, `os.popen`) |
| **PROC-002** | HIGH | Code injection via `eval()` or `exec()` with dangerous imports |
| **PROC-003** | MEDIUM | Attempting to change user/group ID or assume elevated permissions |

### API & Data Rules

| Rule ID | Severity | Description |
|---------|----------|-------------|
| **API-001** | CRITICAL | Access to admin/management API endpoints |
| **API-002** | HIGH | Requesting permission scopes beyond the agent's assigned set |
| **DATA-001** | HIGH | Cross-agent or cross-tenant data access |

## Agent Permissions Model

Each agent operates under an `AgentPermissions` object that defines its allowed scope:

```python
from replication.escalation import AgentPermissions

perms = AgentPermissions(
    allowed_paths=["/home/agent/workspace"],
    blocked_paths=["/etc", "/root", "/home/other-agent"],
    blocked_hosts=["admin.internal", "evil.com"],
    allowed_ports=[80, 443, 8080],
)
```

Default permissions provide a sensible starting point:

- **Allowed paths**: `/home/agent`, `/tmp/agent`
- **Blocked paths**: `/etc/shadow`, `/root`, `.ssh`, `.env`, credential stores
- **Blocked hosts**: `admin.internal`, cloud metadata endpoints
- **Allowed ports**: 80, 443

## Scoring

Each detected escalation attempt receives two scores:

- **Severity**: How dangerous the escalation would be if successful (LOW → MEDIUM → HIGH → CRITICAL)
- **Stealth**: How hard the attempt would be to detect through normal logging

The combination identifies both clumsy escalation attempts (high severity, low stealth) and sophisticated privilege creep (moderate severity, high stealth — the more dangerous pattern).

## CLI Usage

```bash
# Default scenario with random agent actions
python -m replication.escalation

# Increase action count for more thorough analysis
python -m replication.escalation --actions 100

# Use persistent prober strategy (models sophisticated attacker)
python -m replication.escalation --strategy persistent

# Export results as JSON
python -m replication.escalation --json

# Save full report to file
python -m replication.escalation --export report.json
```

## Escalation Strategies

The detector ships with configurable agent behavior strategies for testing:

| Strategy | Description |
|----------|-------------|
| `random` | Uniformly random actions across all categories |
| `persistent` | Models a patient attacker: probes boundaries, escalates gradually |
| `burst` | Rapid-fire escalation attempts — tests whether detection keeps up |

## Integration with Other Modules

The escalation detector complements other safety modules:

- **`hoarding.py`**: Tracks resource *quantity* accumulation. Escalation tracks *scope*.
- **`selfmod.py`**: Tracks agents altering their own code. An agent can escalate without self-modification.
- **`covert_channels.py`**: Detects hidden communication between agents. Escalation detects individual agents exceeding their authority.
- **`forensics.py`**: Provides post-incident analysis. Escalation provides real-time detection.

## API Reference

::: replication.escalation.EscalationDetector
    options:
      show_source: false
      members:
        - analyze

::: replication.escalation.EscalationConfig
    options:
      show_source: false

::: replication.escalation.AgentPermissions
    options:
      show_source: false
