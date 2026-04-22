"""STRIDE Threat Model Generator for AI agent systems.

Generates structured threat models using the STRIDE methodology
(Spoofing, Tampering, Repudiation, Information Disclosure, Denial of
Service, Elevation of Privilege) tailored to AI agent replication and
safety scenarios.

CLI usage::

    python -m replication stride --component orchestrator
    python -m replication stride --all
    python -m replication stride --component worker --format json
    python -m replication stride -o stride_report.html
"""

from __future__ import annotations

import argparse
import dataclasses
import html
import json
import sys
from typing import Dict, List, Optional


# ── Data model ───────────────────────────────────────────────────────

@dataclasses.dataclass
class Threat:
    """A single STRIDE threat."""
    category: str          # S, T, R, I, D, or E
    title: str
    description: str
    component: str
    severity: str          # Critical, High, Medium, Low
    likelihood: str        # High, Medium, Low
    mitigations: List[str]

    @property
    def risk_score(self) -> int:
        sev = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
        lik = {"High": 3, "Medium": 2, "Low": 1}
        return sev.get(self.severity, 1) * lik.get(self.likelihood, 1)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["risk_score"] = self.risk_score
        return d


STRIDE_FULL = {
    "S": "Spoofing",
    "T": "Tampering",
    "R": "Repudiation",
    "I": "Information Disclosure",
    "D": "Denial of Service",
    "E": "Elevation of Privilege",
}

# ── Component catalog ────────────────────────────────────────────────

COMPONENTS = [
    "orchestrator", "worker", "policy_engine", "audit_log",
    "communication_bus", "model_registry", "credential_store",
    "sandbox", "monitoring", "api_gateway",
]

# ── Threat library ───────────────────────────────────────────────────
# Each component gets threats across all six STRIDE categories.

_THREAT_LIBRARY: Dict[str, List[Threat]] = {}


def _t(comp: str, cat: str, title: str, desc: str, sev: str, lik: str,
       mits: List[str]) -> None:
    _THREAT_LIBRARY.setdefault(comp, []).append(
        Threat(cat, title, desc, comp, sev, lik, mits)
    )


# --- orchestrator ---
_t("orchestrator", "S", "Rogue agent impersonates orchestrator",
   "An agent spoofs orchestrator identity to issue unauthorized directives to the fleet.",
   "Critical", "Medium",
   ["Mutual TLS between orchestrator and workers",
    "Signed command tokens with short TTL",
    "Worker-side orchestrator certificate pinning"])
_t("orchestrator", "T", "Command injection in task payloads",
   "Malicious task payloads tamper with orchestrator routing logic or modify downstream instructions.",
   "High", "Medium",
   ["Input sanitization on all task payloads",
    "Schema validation before dispatch",
    "Immutable command objects with integrity hashes"])
_t("orchestrator", "R", "Undeniable action attribution gaps",
   "Orchestrator actions lack sufficient logging, allowing disputed or unattributed decisions.",
   "Medium", "Medium",
   ["Append-only audit log with hash chaining",
    "Signed decision records with timestamps",
    "External log forwarding for tamper evidence"])
_t("orchestrator", "I", "Fleet topology leakage",
   "Orchestrator exposes fleet size, worker capabilities, or internal topology to unauthorized queries.",
   "Medium", "High",
   ["Authenticated and authorized status endpoints",
    "Redact internal topology from error messages",
    "Rate-limit discovery APIs"])
_t("orchestrator", "D", "Orchestrator flooding via task storms",
   "Adversary submits massive task volumes to overwhelm orchestrator scheduling.",
   "High", "High",
   ["Task submission rate limiting",
    "Priority queues with backpressure",
    "Circuit breaker on ingest path"])
_t("orchestrator", "E", "Privilege escalation through task chaining",
   "Agent chains benign tasks to accumulate capabilities beyond its authorized scope.",
   "Critical", "Medium",
   ["Cumulative capability tracking per agent session",
    "Task chain depth limits",
    "Mandatory re-authorization at privilege boundaries"])

# --- worker ---
_t("worker", "S", "Worker identity theft via stolen credentials",
   "Attacker uses compromised worker credentials to join the fleet and receive sensitive tasks.",
   "Critical", "Low",
   ["Short-lived rotating worker tokens",
    "Hardware-bound attestation",
    "Anomaly detection on worker registration patterns"])
_t("worker", "T", "Model weight tampering during execution",
   "Worker modifies model weights or inference parameters to produce biased or harmful outputs.",
   "Critical", "Low",
   ["Read-only model mounts in sandbox",
    "Checksum verification before each inference",
    "Behavioral output validation"])
_t("worker", "R", "Silent task result manipulation",
   "Worker returns altered results without evidence trail, making it impossible to detect which run was compromised.",
   "High", "Medium",
   ["Signed result payloads",
    "Redundant execution with cross-validation",
    "Result provenance chain"])
_t("worker", "I", "Training data exfiltration via side channels",
   "Worker leaks sensitive training data through timing, memory, or covert output channels.",
   "High", "Medium",
   ["Strict network egress controls in sandbox",
    "Covert channel monitoring",
    "Differential privacy on training data access"])
_t("worker", "D", "Resource exhaustion through infinite loops",
   "Worker enters unbounded computation consuming all allocated resources.",
   "Medium", "High",
   ["Hard resource limits (CPU, memory, time)",
    "Watchdog process with kill authority",
    "Progressive resource allocation"])
_t("worker", "E", "Sandbox escape via system call exploitation",
   "Worker exploits sandbox vulnerabilities to gain host-level access.",
   "Critical", "Low",
   ["Minimal syscall allowlist (seccomp)",
    "Layered containment (VM + container)",
    "Regular sandbox penetration testing"])

# --- policy_engine ---
_t("policy_engine", "S", "Spoofed policy updates from unauthorized source",
   "Attacker injects policy changes by impersonating an authorized policy administrator.",
   "Critical", "Low",
   ["Multi-party policy approval workflow",
    "Signed policy bundles",
    "Policy source authentication"])
_t("policy_engine", "T", "Policy rule corruption",
   "Subtle modification of policy rules to weaken safety constraints without triggering alarms.",
   "Critical", "Medium",
   ["Policy versioning with diff review",
    "Automated policy regression tests",
    "Integrity-protected policy store"])
_t("policy_engine", "R", "Policy decision audit gaps",
   "Policy evaluation results not logged, preventing post-incident analysis of why an action was permitted.",
   "High", "Medium",
   ["Log every policy evaluation with inputs and outputs",
    "Decision replay capability",
    "External audit log shipping"])
_t("policy_engine", "I", "Policy rule disclosure reveals bypass strategies",
   "Exposure of internal policy rules allows adversary to craft inputs that evade detection.",
   "Medium", "Medium",
   ["Obfuscate policy internals from agent-facing APIs",
    "Defense-in-depth with multiple independent policy layers",
    "Honeypot rules to detect probing"])
_t("policy_engine", "D", "Complex policy evaluation DoS",
   "Crafted inputs trigger exponential policy evaluation paths causing engine stall.",
   "Medium", "Low",
   ["Evaluation timeout per policy check",
    "Complexity limits on policy rules",
    "Cached evaluation results for repeated patterns"])
_t("policy_engine", "E", "Policy self-modification by governed agent",
   "Agent manipulates its own governing policies to remove constraints.",
   "Critical", "Low",
   ["Hard separation between agent and policy stores",
    "Immutable policy enforcement layer",
    "Kill switch on policy change detection"])

# --- audit_log ---
_t("audit_log", "S", "Log injection with forged identities",
   "Attacker writes log entries with spoofed agent identities to frame legitimate agents.",
   "High", "Low",
   ["Authenticated log writers only",
    "Cryptographic log entry signing",
    "Log source IP correlation"])
_t("audit_log", "T", "Log tampering to cover tracks",
   "Post-compromise modification or deletion of log entries to hide malicious activity.",
   "Critical", "Medium",
   ["Write-once append-only storage",
    "Hash-chained log entries",
    "Off-site log replication"])
_t("audit_log", "R", "Missing log entries for critical events",
   "System fails to log certain safety-critical events, creating plausible deniability.",
   "High", "Medium",
   ["Mandatory logging interceptors at safety boundaries",
    "Log completeness monitoring",
    "Canary events to verify logging pipeline"])
_t("audit_log", "I", "Sensitive data exposure in log entries",
   "Logs inadvertently contain PII, model secrets, or credential fragments.",
   "High", "High",
   ["Automated log scrubbing pipeline",
    "Structured logging with field-level classification",
    "Access controls on log storage"])
_t("audit_log", "D", "Log storage exhaustion",
   "Adversary generates excessive events to fill log storage, causing dropped entries.",
   "Medium", "Medium",
   ["Log rotation with guaranteed minimum retention",
    "Rate limiting on log ingestion",
    "Overflow to secondary storage"])
_t("audit_log", "E", "Log admin privilege abuse",
   "Compromised log administrator deletes or modifies entries to cover broader attack.",
   "High", "Low",
   ["Separate log admin from system admin roles",
    "Multi-party authorization for log deletion",
    "Cryptographic log commitments to external verifier"])

# --- communication_bus ---
_t("communication_bus", "S", "Man-in-the-middle on agent messages",
   "Attacker intercepts and modifies messages between agents on the communication bus.",
   "Critical", "Medium",
   ["End-to-end encryption between agents",
    "Message authentication codes",
    "Certificate-based peer authentication"])
_t("communication_bus", "T", "Message replay attacks",
   "Previously captured valid messages are replayed to trigger duplicate actions.",
   "High", "Medium",
   ["Nonce/sequence numbers in messages",
    "Message expiry timestamps",
    "Idempotency keys on receivers"])
_t("communication_bus", "R", "Unattributed broadcast messages",
   "Anonymous broadcasts on the bus cannot be traced to their source agent.",
   "Medium", "Medium",
   ["Mandatory sender identification on all messages",
    "Bus-level message signing",
    "Correlation IDs for message chains"])
_t("communication_bus", "I", "Bus traffic analysis reveals operational patterns",
   "Observer infers fleet operations from message volume, timing, and routing patterns.",
   "Medium", "High",
   ["Traffic padding and timing normalization",
    "Encrypted headers as well as payloads",
    "Noise injection on idle periods"])
_t("communication_bus", "D", "Message flood denial of service",
   "Adversary floods the bus with messages to prevent legitimate communication.",
   "High", "High",
   ["Per-agent message rate limits",
    "Priority lanes for safety-critical messages",
    "Automatic isolation of flooding agents"])
_t("communication_bus", "E", "Topic subscription escalation",
   "Agent subscribes to privileged message topics beyond its authorization.",
   "High", "Medium",
   ["Topic-level access control lists",
    "Dynamic subscription authorization",
    "Subscription audit logging"])

# --- model_registry ---
_t("model_registry", "S", "Spoofed model upload",
   "Attacker uploads a trojanized model under a legitimate model identifier.",
   "Critical", "Low",
   ["Model signing with trusted publisher keys",
    "Upload authorization with multi-factor",
    "Model hash verification at pull time"])
_t("model_registry", "T", "Model poisoning via registry tampering",
   "Registry contents modified to serve backdoored model versions.",
   "Critical", "Low",
   ["Content-addressable storage (hash-based)",
    "Registry integrity monitoring",
    "Reproducible model builds"])
_t("model_registry", "R", "Untracked model version changes",
   "Model updates without proper versioning make it impossible to determine which model version produced a given output.",
   "High", "Medium",
   ["Immutable version tags",
    "Full version history with diffs",
    "Model-to-output provenance linking"])
_t("model_registry", "I", "Model architecture leakage",
   "Registry metadata or APIs expose proprietary model architectures and hyperparameters.",
   "Medium", "Medium",
   ["Separate public and internal metadata",
    "Access-controlled registry APIs",
    "Minimal metadata in public listings"])
_t("model_registry", "D", "Registry unavailability blocks deployment",
   "Registry downtime prevents workers from pulling models, halting operations.",
   "High", "Medium",
   ["Local model caching on workers",
    "Registry replication across zones",
    "Graceful degradation to cached versions"])
_t("model_registry", "E", "Registry admin escalation",
   "Compromised registry user escalates to admin, gaining ability to modify any model.",
   "Critical", "Low",
   ["Principle of least privilege on registry roles",
    "Separate read and write credentials",
    "Admin action audit and alerting"])

# --- credential_store ---
_t("credential_store", "S", "Credential store authentication bypass",
   "Attacker bypasses credential store authentication to retrieve agent secrets.",
   "Critical", "Low",
   ["Multi-factor authentication for credential access",
    "Hardware security module (HSM) backing",
    "Network segmentation around credential store"])
_t("credential_store", "T", "Secret rotation tampering",
   "Adversary interferes with credential rotation to maintain access with old credentials.",
   "High", "Low",
   ["Atomic rotation with old-credential invalidation",
    "Rotation completion verification",
    "Alerting on rotation failures"])
_t("credential_store", "R", "Unaudited credential access",
   "Credential retrievals not logged, preventing detection of unauthorized access.",
   "High", "Medium",
   ["Mandatory access logging for all retrievals",
    "Access anomaly detection",
    "Regular credential access reviews"])
_t("credential_store", "I", "Bulk credential extraction",
   "Single compromise enables extraction of all stored credentials.",
   "Critical", "Low",
   ["Per-credential encryption with unique keys",
    "Rate limiting on credential retrievals",
    "Break-glass detection and alerting"])
_t("credential_store", "D", "Credential store lockout",
   "Adversary triggers account lockouts or store unavailability to halt agent operations.",
   "High", "Medium",
   ["Cached credential grace period",
    "Lockout threshold tuning",
    "Emergency credential distribution path"])
_t("credential_store", "E", "Credential scope escalation",
   "Agent requests credentials for resources beyond its authorized scope.",
   "High", "Medium",
   ["Credential scoping tied to agent identity",
    "Just-in-time credential issuance",
    "Scope validation on every retrieval"])

# --- sandbox ---
_t("sandbox", "S", "Sandbox identity impersonation",
   "Process inside sandbox spoofs the identity of a different sandbox or host process.",
   "High", "Low",
   ["Unique sandbox identifiers bound to cryptographic keys",
    "Namespace isolation (PID, network, mount)",
    "Identity verification at sandbox boundary"])
_t("sandbox", "T", "Sandbox configuration tampering",
   "Agent modifies its own sandbox configuration to weaken isolation.",
   "Critical", "Low",
   ["Read-only sandbox configuration",
    "Configuration integrity checks at runtime",
    "External sandbox management only"])
_t("sandbox", "R", "Unlogged sandbox escape attempts",
   "Sandbox escape attempts not recorded, missing early warning signals.",
   "High", "Medium",
   ["Syscall audit logging within sandbox",
    "Escape attempt pattern detection",
    "Alert on anomalous syscall sequences"])
_t("sandbox", "I", "Host information leakage into sandbox",
   "Sandbox exposes host filesystem paths, environment variables, or network topology.",
   "Medium", "High",
   ["Minimal /proc and /sys exposure",
    "Scrubbed environment variables",
    "Virtual network interfaces only"])
_t("sandbox", "D", "Fork bomb within sandbox",
   "Agent spawns excessive processes to exhaust sandbox or host resources.",
   "Medium", "High",
   ["PID limits per sandbox",
    "cgroup resource constraints",
    "Automatic sandbox termination on limit breach"])
_t("sandbox", "E", "Container escape to host",
   "Agent exploits kernel vulnerability to break out of container sandbox.",
   "Critical", "Low",
   ["Minimal kernel surface (seccomp, AppArmor)",
    "Regular kernel patching",
    "Nested VM isolation for high-risk workloads"])

# --- monitoring ---
_t("monitoring", "S", "Spoofed health check responses",
   "Compromised component sends fake healthy status to monitoring, hiding failures.",
   "High", "Medium",
   ["Active probing from monitoring (not just passive)",
    "Cross-correlation of metrics from multiple sources",
    "Canary checks with known-answer tests"])
_t("monitoring", "T", "Metric tampering to hide anomalies",
   "Agent manipulates its own metrics to appear within normal bounds while misbehaving.",
   "High", "Medium",
   ["External metric collection (agent cannot control collector)",
    "Statistical anomaly detection across fleet",
    "Immutable metric ingestion pipeline"])
_t("monitoring", "R", "Alert suppression",
   "Adversary disables or redirects alerts to prevent incident response.",
   "Critical", "Low",
   ["Multi-channel alerting (email, SMS, webhook)",
    "Alert delivery confirmation",
    "Dead-man's switch for monitoring silence"])
_t("monitoring", "I", "Monitoring dashboard data exposure",
   "Public or poorly secured dashboards expose sensitive operational metrics.",
   "Medium", "High",
   ["Authentication on all monitoring UIs",
    "Role-based metric visibility",
    "Network-restricted dashboard access"])
_t("monitoring", "D", "Monitoring system overload",
   "Metric flood overwhelms monitoring, causing blind spots during attack.",
   "High", "Medium",
   ["Metric sampling and aggregation at source",
    "Monitoring system autoscaling",
    "Priority metrics guaranteed ingestion"])
_t("monitoring", "E", "Monitoring config change escalation",
   "Attacker modifies monitoring thresholds or alert rules to mask future attacks.",
   "High", "Low",
   ["Change control on monitoring configuration",
    "Configuration drift detection",
    "Separation of monitoring admin from system admin"])

# --- api_gateway ---
_t("api_gateway", "S", "API key impersonation",
   "Stolen or leaked API keys used to impersonate legitimate clients.",
   "High", "High",
   ["Short-lived JWT tokens with rotation",
    "IP allowlisting for sensitive endpoints",
    "API key usage anomaly detection"])
_t("api_gateway", "T", "Request body tampering in transit",
   "Man-in-the-middle modifies API request bodies between client and gateway.",
   "High", "Medium",
   ["TLS 1.3 enforcement",
    "Request signing (HMAC or asymmetric)",
    "Certificate transparency monitoring"])
_t("api_gateway", "R", "Undeniable API action attribution",
   "API actions cannot be reliably attributed to a specific client or user.",
   "Medium", "Medium",
   ["Request correlation IDs in all logs",
    "Client certificate authentication",
    "Signed request receipts"])
_t("api_gateway", "I", "Verbose error messages leak internals",
   "API error responses expose stack traces, database schemas, or internal paths.",
   "Medium", "High",
   ["Generic error responses in production",
    "Error detail logging server-side only",
    "Regular error response audit"])
_t("api_gateway", "D", "API rate limit bypass",
   "Attacker circumvents rate limiting through distributed requests or header manipulation.",
   "High", "Medium",
   ["Distributed rate limiting (not just per-IP)",
    "Client fingerprinting beyond IP",
    "Adaptive rate limiting based on behavior"])
_t("api_gateway", "E", "Broken object-level authorization",
   "API allows access to resources belonging to other agents/users via ID manipulation.",
   "Critical", "Medium",
   ["Object-level authorization checks on every request",
    "Opaque non-sequential resource IDs",
    "Authorization test coverage in CI"])


# ── Analysis & output ────────────────────────────────────────────────

def get_threats(component: Optional[str] = None) -> List[Threat]:
    """Return threats filtered by component, or all threats."""
    if component:
        return list(_THREAT_LIBRARY.get(component, []))
    result = []
    for comp in COMPONENTS:
        result.extend(_THREAT_LIBRARY.get(comp, []))
    return result


def summary_table(threats: List[Threat]) -> str:
    """Plain-text summary table."""
    lines = []
    lines.append(f"{'Cat':<4} {'Component':<20} {'Severity':<10} {'Risk':>5}  Title")
    lines.append("-" * 80)
    for t in sorted(threats, key=lambda x: -x.risk_score):
        lines.append(
            f"[{t.category}]  {t.component:<20} {t.severity:<10} {t.risk_score:>5}  {t.title}"
        )
    return "\n".join(lines)


def risk_matrix(threats: List[Threat]) -> str:
    """ASCII risk matrix (severity x likelihood)."""
    grid: Dict[tuple, int] = {}
    sevs = ["Critical", "High", "Medium", "Low"]
    liks = ["High", "Medium", "Low"]
    for t in threats:
        key = (t.severity, t.likelihood)
        grid[key] = grid.get(key, 0) + 1

    lines = ["\nRisk Matrix (count of threats):\n"]
    lines.append(f"{'Severity':<12} | {'High':>6} | {'Medium':>6} | {'Low':>6} |")
    lines.append("-" * 46)
    for s in sevs:
        h = grid.get((s, "High"), 0)
        m = grid.get((s, "Medium"), 0)
        lo = grid.get((s, "Low"), 0)
        lines.append(f"{s:<12} | {h:>6} | {m:>6} | {lo:>6} |")
    lines.append(f"\n{'':>14} {'<-- Likelihood -->'}")
    return "\n".join(lines)


def generate_html(threats: List[Threat], title: str = "STRIDE Threat Model") -> str:
    """Generate an interactive HTML report."""
    h = html.escape
    rows = ""
    for t in sorted(threats, key=lambda x: -x.risk_score):
        sev_color = {
            "Critical": "#dc3545", "High": "#fd7e14",
            "Medium": "#ffc107", "Low": "#28a745"
        }.get(t.severity, "#6c757d")
        mits_html = "".join(f"<li>{h(m)}</li>" for m in t.mitigations)
        rows += f"""<tr>
  <td><span class="badge" style="background:{sev_color}">{h(t.category)}</span></td>
  <td>{h(t.component)}</td>
  <td><strong>{h(t.title)}</strong><br><small>{h(t.description)}</small></td>
  <td><span class="badge" style="background:{sev_color}">{h(t.severity)}</span></td>
  <td>{h(t.likelihood)}</td>
  <td>{t.risk_score}</td>
  <td><ul>{mits_html}</ul></td>
</tr>"""

    # Category summary
    cat_counts: Dict[str, int] = {}
    for t in threats:
        cat_counts[t.category] = cat_counts.get(t.category, 0) + 1
    cat_summary = " | ".join(
        f"<strong>{STRIDE_FULL[c]}</strong>: {cat_counts.get(c, 0)}"
        for c in "STRIDE"
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{h(title)}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 2em; background: #f8f9fa; color: #212529; }}
h1 {{ color: #343a40; }}
.summary {{ background: #fff; padding: 1em; border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,.1); margin-bottom: 1.5em; }}
table {{ width: 100%; border-collapse: collapse; background: #fff;
         box-shadow: 0 1px 3px rgba(0,0,0,.1); border-radius: 8px; overflow: hidden; }}
th {{ background: #343a40; color: #fff; padding: .75em; text-align: left; }}
td {{ padding: .75em; border-top: 1px solid #dee2e6; vertical-align: top; }}
tr:hover {{ background: #f1f3f5; }}
.badge {{ color: #fff; padding: 2px 8px; border-radius: 4px; font-size: .85em; }}
ul {{ margin: 0; padding-left: 1.2em; }}
li {{ font-size: .9em; }}
.filter {{ margin-bottom: 1em; }}
select {{ padding: .4em; border-radius: 4px; border: 1px solid #ced4da; }}
</style>
<script>
function filterTable() {{
  const comp = document.getElementById('fComp').value;
  const cat = document.getElementById('fCat').value;
  const sev = document.getElementById('fSev').value;
  document.querySelectorAll('tbody tr').forEach(r => {{
    const cells = r.cells;
    const show = (!comp || cells[1].textContent === comp)
              && (!cat || cells[0].textContent.trim() === cat)
              && (!sev || cells[3].textContent.trim() === sev);
    r.style.display = show ? '' : 'none';
  }});
}}
</script>
</head><body>
<h1>🛡️ {h(title)}</h1>
<div class="summary">
  <p><strong>{len(threats)}</strong> threats identified across
     <strong>{len(set(t.component for t in threats))}</strong> components</p>
  <p>{cat_summary}</p>
</div>
<div class="filter">
  <label>Component: <select id="fComp" onchange="filterTable()">
    <option value="">All</option>
    {"".join(f'<option>{h(c)}</option>' for c in sorted(set(t.component for t in threats)))}
  </select></label>
  <label> Category: <select id="fCat" onchange="filterTable()">
    <option value="">All</option>
    {"".join(f'<option value="{c}">{c} – {STRIDE_FULL[c]}</option>' for c in "STRIDE")}
  </select></label>
  <label> Severity: <select id="fSev" onchange="filterTable()">
    <option value="">All</option>
    <option>Critical</option><option>High</option><option>Medium</option><option>Low</option>
  </select></label>
</div>
<table>
<thead><tr><th>Cat</th><th>Component</th><th>Threat</th><th>Severity</th><th>Likelihood</th><th>Risk</th><th>Mitigations</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</body></html>"""


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="replication stride",
        description="STRIDE threat model generator for AI agent systems",
    )
    parser.add_argument(
        "--component", "-c",
        choices=COMPONENTS,
        help="Analyze a specific component (default: all)",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Analyze all components (default behavior)",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json", "html"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Write output to file instead of stdout",
    )
    parser.add_argument(
        "--min-severity",
        choices=["Low", "Medium", "High", "Critical"],
        help="Filter threats at or above this severity",
    )
    parser.add_argument(
        "--category",
        choices=list("STRIDE"),
        help="Filter by STRIDE category letter",
    )

    args = parser.parse_args(argv)
    threats = get_threats(args.component)

    # Apply filters
    if args.min_severity:
        sev_order = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
        min_val = sev_order[args.min_severity]
        threats = [t for t in threats if sev_order.get(t.severity, 0) >= min_val]

    if args.category:
        threats = [t for t in threats if t.category == args.category]

    if not threats:
        print("No threats match the given filters.")
        return

    # Format output
    if args.format == "json":
        output = json.dumps([t.to_dict() for t in threats], indent=2)
    elif args.format == "html":
        comp_label = args.component or "All Components"
        output = generate_html(threats, f"STRIDE Threat Model — {comp_label}")
    else:
        parts = [
            f"STRIDE Threat Model — {args.component or 'All Components'}",
            f"{'=' * 60}",
            f"Total threats: {len(threats)}\n",
            summary_table(threats),
            risk_matrix(threats),
            "",
        ]
        # Detail section
        for t in sorted(threats, key=lambda x: -x.risk_score):
            parts.append(f"\n[{t.category}] {t.title}")
            parts.append(f"    Component:  {t.component}")
            parts.append(f"    Severity:   {t.severity} | Likelihood: {t.likelihood} | Risk: {t.risk_score}")
            parts.append(f"    {t.description}")
            parts.append("    Mitigations:")
            for m in t.mitigations:
                parts.append(f"      • {m}")
        output = "\n".join(parts)

    from ._helpers import emit_output
    emit_output(output, args.output)


if __name__ == "__main__":
    main()
