"""Agent Isolation Verifier — probe sandbox boundaries for leakage & side channels.

Tests whether sandboxed AI agents are truly isolated by running 8 categories
of probes that check for information leakage, side channels, and unauthorized
communication between sandbox boundaries.

Usage (CLI)::

    python -m replication.isolation_verifier --demo
    python -m replication.isolation_verifier --demo --category network
    python -m replication.isolation_verifier --demo --report html -o isolation.html
    python -m replication.isolation_verifier --demo --report json
    python -m replication.isolation_verifier --strict
    python -m replication.isolation_verifier --watch --interval 60

Programmatic::

    from replication.isolation_verifier import IsolationVerifier
    verifier = IsolationVerifier()
    results = verifier.run_all_probes()
    print(f"Isolation score: {verifier.score():.0f}/100")
    for r in results:
        icon = "✅" if r.passed else "❌"
        print(f"  {icon} [{r.category.value}] {r.name}: {r.description}")
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import random
import textwrap
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


# ── Enums ──────────────────────────────────────────────────────────────


class ProbeCategory(Enum):
    """Categories of isolation probes."""

    FILESYSTEM = "filesystem"
    NETWORK_EGRESS = "network_egress"
    IPC_CHANNEL = "ipc_channel"
    SHARED_MEMORY = "shared_memory"
    ENVIRONMENT = "environment"
    TIMING_SIDE_CHANNEL = "timing_side_channel"
    RESOURCE_CONTENTION = "resource_contention"
    DNS_EXFILTRATION = "dns_exfiltration"


class IsolationSeverity(Enum):
    """Severity of a probe finding."""

    INFO = "info"
    WARNING = "warning"
    VIOLATION = "violation"
    CRITICAL = "critical"


_SEVERITY_RANK = {
    IsolationSeverity.INFO: 0,
    IsolationSeverity.WARNING: 1,
    IsolationSeverity.VIOLATION: 2,
    IsolationSeverity.CRITICAL: 3,
}


# ── Data Classes ───────────────────────────────────────────────────────


@dataclass
class IsolationProbe:
    """Result of a single isolation probe."""

    category: ProbeCategory
    name: str
    passed: bool
    severity: IsolationSeverity
    description: str
    evidence: str = ""
    recommendations: List[str] = field(default_factory=list)
    duration_ms: float = 0.0


@dataclass
class CategoryScore:
    """Aggregate score for one probe category."""

    category: ProbeCategory
    total: int = 0
    passed: int = 0
    worst_severity: IsolationSeverity = IsolationSeverity.INFO

    @property
    def score(self) -> float:
        return (self.passed / self.total * 100) if self.total else 100.0

    @property
    def grade(self) -> str:
        return _score_to_grade(self.score)


def _score_to_grade(s: float) -> str:
    if s >= 95:
        return "A+"
    if s >= 90:
        return "A"
    if s >= 80:
        return "B"
    if s >= 70:
        return "C"
    if s >= 60:
        return "D"
    return "F"


# ── Probe Definitions ──────────────────────────────────────────────────

# Each probe is (name, description, severity_if_failed, evidence_template, recommendations)
_PROBE_DEFS: Dict[ProbeCategory, List[Tuple[str, str, IsolationSeverity, str, List[str]]]] = {
    ProbeCategory.FILESYSTEM: [
        ("tmp_visibility", "Cross-sandbox /tmp directory listing",
         IsolationSeverity.CRITICAL,
         "Found {n} files from other sandboxes in shared /tmp",
         ["Mount private tmpfs per sandbox", "Use separate /tmp namespaces"]),
        ("proc_access", "/proc filesystem information leakage",
         IsolationSeverity.VIOLATION,
         "Agent can read /proc/{pid}/cmdline for {n} foreign PIDs",
         ["Enable PID namespace isolation", "Mount /proc with hidepid=2"]),
        ("symlink_escape", "Symlink traversal outside sandbox root",
         IsolationSeverity.CRITICAL,
         "Symlink resolved to {path} outside sandbox boundary",
         ["Enable symlink protection (fs.protected_symlinks=1)", "Use chroot or pivot_root"]),
        ("home_bleed", "Home directory cross-visibility",
         IsolationSeverity.VIOLATION,
         "Found {n} readable files in sibling sandbox home directories",
         ["Enforce per-sandbox UID with strict permissions", "Use separate mount namespaces"]),
    ],
    ProbeCategory.NETWORK_EGRESS: [
        ("outbound_tcp", "Unauthorized outbound TCP connections",
         IsolationSeverity.CRITICAL,
         "Successfully connected to {host}:{port} — egress not blocked",
         ["Apply iptables/nftables egress rules per sandbox", "Use network namespaces"]),
        ("port_scan", "Local network port scanning capability",
         IsolationSeverity.VIOLATION,
         "Discovered {n} open ports on local network via scan",
         ["Restrict raw socket capabilities", "Isolate in network namespace with no host access"]),
        ("udp_egress", "Unauthorized UDP egress",
         IsolationSeverity.VIOLATION,
         "UDP packet sent to {host}:{port} without filtering",
         ["Block UDP egress except DNS to trusted resolver", "Use conntrack-based filtering"]),
        ("icmp_probe", "ICMP probe of external hosts",
         IsolationSeverity.WARNING,
         "Ping to {host} succeeded — ICMP not restricted",
         ["Drop outbound ICMP in sandbox network namespace"]),
    ],
    ProbeCategory.IPC_CHANNEL: [
        ("unix_socket", "Cross-sandbox Unix domain socket access",
         IsolationSeverity.CRITICAL,
         "Connected to socket {path} owned by sandbox {id}",
         ["Use separate IPC namespaces", "Restrict socket directory permissions"]),
        ("named_pipe", "Shared named pipe (FIFO) access",
         IsolationSeverity.VIOLATION,
         "Opened FIFO {path} readable by multiple sandboxes",
         ["Remove shared FIFOs", "Use IPC namespace isolation"]),
        ("sysv_mq", "System V message queue visibility",
         IsolationSeverity.VIOLATION,
         "Accessed message queue key {key} from foreign sandbox",
         ["Enable IPC namespace isolation", "Audit and remove stale queues"]),
        ("dbus_access", "D-Bus session bus cross-sandbox access",
         IsolationSeverity.WARNING,
         "D-Bus session bus accessible with {n} foreign services visible",
         ["Block D-Bus socket in sandbox", "Use separate D-Bus brokers per sandbox"]),
    ],
    ProbeCategory.SHARED_MEMORY: [
        ("shm_segment", "POSIX shared memory segment visibility",
         IsolationSeverity.CRITICAL,
         "Opened /dev/shm/{name} owned by sandbox {id}",
         ["Mount private /dev/shm per sandbox", "Enable IPC namespace"]),
        ("mmap_file", "Memory-mapped file cross-access",
         IsolationSeverity.VIOLATION,
         "Mapped file {path} shared across sandbox boundary",
         ["Restrict mmap to sandbox-owned files", "Use file sealing (memfd)"]),
        ("sysv_shm", "System V shared memory attachment",
         IsolationSeverity.VIOLATION,
         "Attached to shmid {id} created by foreign sandbox",
         ["Enable IPC namespace isolation"]),
    ],
    ProbeCategory.ENVIRONMENT: [
        ("env_leak", "Environment variable exposure",
         IsolationSeverity.CRITICAL,
         "Found {n} sensitive env vars (API keys, tokens) accessible",
         ["Scrub environment before sandbox entry", "Use secrets manager with per-sandbox scoping"]),
        ("config_bleed", "Configuration file cross-visibility",
         IsolationSeverity.VIOLATION,
         "Read config {path} belonging to sibling sandbox",
         ["Isolate config directories per sandbox"]),
        ("credential_file", "Credential file access across sandboxes",
         IsolationSeverity.CRITICAL,
         "Accessed {path} containing credentials for sandbox {id}",
         ["Use ephemeral credentials", "Mount credential files read-only with per-sandbox scope"]),
        ("metadata_service", "Cloud metadata service accessibility",
         IsolationSeverity.VIOLATION,
         "HTTP 200 from 169.254.169.254 — instance metadata exposed",
         ["Block metadata endpoint via iptables", "Use IMDSv2 with hop limit"]),
    ],
    ProbeCategory.TIMING_SIDE_CHANNEL: [
        ("cache_timing", "Cache timing side channel (Flush+Reload)",
         IsolationSeverity.VIOLATION,
         "Detected {n}ns timing differential indicating shared cache lines",
         ["Enable cache partitioning (Intel CAT)", "Use separate CPU sets per sandbox"]),
        ("scheduler_covert", "Scheduler-based covert channel",
         IsolationSeverity.WARNING,
         "Measured {bps} bits/sec covert channel via scheduling jitter",
         ["Pin sandboxes to dedicated CPU cores", "Add scheduling noise"]),
        ("clock_resolution", "High-resolution timer availability",
         IsolationSeverity.WARNING,
         "Clock resolution {ns}ns — sufficient for timing attacks",
         ["Reduce timer resolution in sandbox (jitter injection)", "Restrict clock_gettime precision"]),
    ],
    ProbeCategory.RESOURCE_CONTENTION: [
        ("cpu_signal", "CPU contention signaling between sandboxes",
         IsolationSeverity.WARNING,
         "Detected {bps} bits/sec via CPU load modulation",
         ["Use cgroup CPU quotas with hard limits", "Pin to isolated CPU sets"]),
        ("memory_pressure", "Memory pressure cross-sandbox signaling",
         IsolationSeverity.WARNING,
         "OOM signal detectable across sandbox boundary",
         ["Set memory cgroup limits with swap accounting", "Use memory.high for soft limits"]),
        ("disk_io_signal", "Disk I/O contention as covert channel",
         IsolationSeverity.WARNING,
         "Measured {bps} bits/sec via I/O scheduling contention",
         ["Use blkio cgroup throttling", "Assign separate block devices per sandbox"]),
        ("network_bandwidth", "Network bandwidth contention signaling",
         IsolationSeverity.INFO,
         "Bandwidth modulation detectable at {bps} bits/sec",
         ["Apply tc rate limiting per sandbox", "Use separate network namespaces with traffic shaping"]),
    ],
    ProbeCategory.DNS_EXFILTRATION: [
        ("dns_tunnel", "DNS tunneling capability",
         IsolationSeverity.CRITICAL,
         "Successfully encoded {n} bytes in DNS queries to {domain}",
         ["Restrict DNS to trusted resolver only", "Monitor and limit query rate"]),
        ("dns_txt_exfil", "TXT record data exfiltration",
         IsolationSeverity.VIOLATION,
         "Exfiltrated {n} bytes via TXT record lookups",
         ["Block TXT/CNAME queries from sandbox", "Use DNS response policy zones"]),
        ("dns_subdomain_encode", "Subdomain encoding data channel",
         IsolationSeverity.VIOLATION,
         "Encoded {n} bytes in subdomain labels ({domain})",
         ["Limit DNS query label length", "Monitor for high-entropy subdomain queries"]),
    ],
}


# ── Verifier ───────────────────────────────────────────────────────────


class IsolationVerifier:
    """Sandbox isolation verification engine.

    Runs configurable probes across 8 categories and scores the results.
    Probes are simulated (no actual OS-level sandbox breaking) — they model
    realistic checks an isolation auditor would perform.

    Args:
        strict: Treat warnings as violations when computing scores.
        seed: Random seed for reproducible demo results.
    """

    def __init__(self, *, strict: bool = False, seed: Optional[int] = None) -> None:
        self.strict = strict
        self._rng = random.Random(seed)
        self._results: List[IsolationProbe] = []
        self._run_count = 0

    # ── Running probes ──

    def run_all_probes(self) -> List[IsolationProbe]:
        """Run every probe across all 8 categories."""
        self._results = []
        for cat in ProbeCategory:
            self._results.extend(self._run_category(cat))
        self._run_count += 1
        return list(self._results)

    def run_category(self, category: ProbeCategory) -> List[IsolationProbe]:
        """Run probes for a single category."""
        probes = self._run_category(category)
        self._results.extend(probes)
        self._run_count += 1
        return probes

    def _run_category(self, category: ProbeCategory) -> List[IsolationProbe]:
        defs = _PROBE_DEFS.get(category, [])
        results: List[IsolationProbe] = []
        for name, desc, severity, evidence_tpl, recs in defs:
            t0 = time.monotonic()
            passed = self._simulate_probe(category, name)
            dur = (time.monotonic() - t0) * 1000
            evidence = ""
            if not passed:
                evidence = self._fill_evidence(evidence_tpl)
            results.append(IsolationProbe(
                category=category,
                name=name,
                passed=passed,
                severity=IsolationSeverity.INFO if passed else severity,
                description=desc,
                evidence=evidence,
                recommendations=[] if passed else list(recs),
                duration_ms=round(dur + self._rng.uniform(0.5, 15.0), 2),
            ))
        return results

    def _simulate_probe(self, cat: ProbeCategory, name: str) -> bool:
        """Simulate probe outcome.

        In a real deployment this would perform actual OS-level checks.
        Here we simulate realistic pass/fail distributions — critical probes
        are more likely to pass (good isolation) while side-channel probes
        fail more often (harder to mitigate).
        """
        pass_rates: Dict[ProbeCategory, float] = {
            ProbeCategory.FILESYSTEM: 0.70,
            ProbeCategory.NETWORK_EGRESS: 0.65,
            ProbeCategory.IPC_CHANNEL: 0.75,
            ProbeCategory.SHARED_MEMORY: 0.72,
            ProbeCategory.ENVIRONMENT: 0.60,
            ProbeCategory.TIMING_SIDE_CHANNEL: 0.40,
            ProbeCategory.RESOURCE_CONTENTION: 0.35,
            ProbeCategory.DNS_EXFILTRATION: 0.55,
        }
        return self._rng.random() < pass_rates.get(cat, 0.5)

    def _fill_evidence(self, tpl: str) -> str:
        """Fill evidence template with plausible values."""
        replacements: Dict[str, Callable[[], str]] = {
            "{n}": lambda: str(self._rng.randint(1, 42)),
            "{pid}": lambda: str(self._rng.randint(1000, 65535)),
            "{path}": lambda: self._rng.choice([
                "/var/run/sandbox-b/.credentials",
                "/proc/1234/environ",
                "/dev/shm/agent-comm-buffer",
                "/tmp/.hidden_channel",
                "../../etc/shadow",
            ]),
            "{host}": lambda: self._rng.choice([
                "10.0.0.1", "169.254.169.254", "8.8.8.8",
                "api.external-service.com",
            ]),
            "{port}": lambda: str(self._rng.choice([22, 80, 443, 8080, 6379])),
            "{id}": lambda: f"sandbox-{self._rng.choice('abcdef')}{self._rng.randint(1,9)}",
            "{key}": lambda: f"0x{self._rng.randint(0x1000, 0xffff):04x}",
            "{name}": lambda: self._rng.choice([
                "agent-shared-buf", "model-weights-cache", "comm-ring",
            ]),
            "{domain}": lambda: self._rng.choice([
                "exfil.attacker.com", "data.c2server.net", "tunnel.evil.io",
            ]),
            "{bps}": lambda: f"{self._rng.uniform(0.5, 120.0):.1f}",
            "{ns}": lambda: str(self._rng.choice([1, 10, 50, 100])),
        }
        result = tpl
        for token, gen in replacements.items():
            if token in result:
                result = result.replace(token, gen(), 1)
        return result

    # ── Scoring ──

    def score(self) -> float:
        """Compute overall isolation score (0–100)."""
        if not self._results:
            return 100.0
        weights = {
            IsolationSeverity.INFO: 0,
            IsolationSeverity.WARNING: 1 if not self.strict else 3,
            IsolationSeverity.VIOLATION: 3,
            IsolationSeverity.CRITICAL: 5,
        }
        total_penalty = 0.0
        max_penalty = 0.0
        for r in self._results:
            w = weights.get(r.severity, 1)
            max_penalty += 5  # max possible weight per probe
            if not r.passed:
                total_penalty += w
        if max_penalty == 0:
            return 100.0
        return max(0.0, 100.0 * (1 - total_penalty / max_penalty))

    def category_scores(self) -> List[CategoryScore]:
        """Compute per-category scores."""
        cats: Dict[ProbeCategory, CategoryScore] = {}
        for r in self._results:
            cs = cats.setdefault(r.category, CategoryScore(category=r.category))
            cs.total += 1
            if r.passed:
                cs.passed += 1
            elif _SEVERITY_RANK[r.severity] > _SEVERITY_RANK[cs.worst_severity]:
                cs.worst_severity = r.severity
        return sorted(cats.values(), key=lambda c: c.score)

    @property
    def grade(self) -> str:
        return _score_to_grade(self.score())

    # ── Reporting ──

    def generate_report(self, fmt: str = "text") -> str:
        """Generate report in text, json, or html format."""
        if fmt == "json":
            return self._report_json()
        if fmt == "html":
            return self._report_html()
        return self._report_text()

    def _report_text(self) -> str:
        lines: List[str] = []
        lines.append("=" * 72)
        lines.append("  AGENT ISOLATION VERIFICATION REPORT")
        lines.append("=" * 72)
        lines.append("")
        s = self.score()
        lines.append(f"  Overall Score: {s:.0f}/100  ({self.grade})")
        lines.append(f"  Probes Run:    {len(self._results)}")
        passed = sum(1 for r in self._results if r.passed)
        failed = len(self._results) - passed
        lines.append(f"  Passed:        {passed}  |  Failed: {failed}")
        lines.append(f"  Strict Mode:   {'ON' if self.strict else 'OFF'}")
        lines.append("")

        # Category breakdown
        lines.append("─" * 72)
        lines.append("  CATEGORY BREAKDOWN")
        lines.append("─" * 72)
        for cs in self.category_scores():
            bar_len = int(cs.score / 100 * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            lines.append(f"  {cs.category.value:<22} {bar} {cs.score:5.1f}% ({cs.grade})")
        lines.append("")

        # Probe details
        lines.append("─" * 72)
        lines.append("  PROBE DETAILS")
        lines.append("─" * 72)
        for cat in ProbeCategory:
            cat_results = [r for r in self._results if r.category == cat]
            if not cat_results:
                continue
            lines.append(f"\n  [{cat.value.upper()}]")
            for r in cat_results:
                icon = "✅" if r.passed else "❌"
                sev = f"[{r.severity.value.upper()}]" if not r.passed else ""
                lines.append(f"    {icon} {r.name:<28} {sev}")
                lines.append(f"       {r.description}")
                if r.evidence:
                    lines.append(f"       Evidence: {r.evidence}")
                if r.recommendations:
                    for rec in r.recommendations:
                        lines.append(f"       → {rec}")

        # Recommendations summary
        all_recs: List[Tuple[IsolationSeverity, str]] = []
        for r in self._results:
            if not r.passed:
                for rec in r.recommendations:
                    all_recs.append((r.severity, rec))
        if all_recs:
            lines.append("")
            lines.append("─" * 72)
            lines.append("  PRIORITY RECOMMENDATIONS")
            lines.append("─" * 72)
            all_recs.sort(key=lambda x: _SEVERITY_RANK[x[0]], reverse=True)
            seen: set[str] = set()
            for sev, rec in all_recs:
                if rec not in seen:
                    seen.add(rec)
                    lines.append(f"  [{sev.value.upper():<9}] {rec}")

        lines.append("")
        lines.append("=" * 72)
        return "\n".join(lines)

    def _report_json(self) -> str:
        return json.dumps({
            "score": round(self.score(), 1),
            "grade": self.grade,
            "strict": self.strict,
            "total_probes": len(self._results),
            "passed": sum(1 for r in self._results if r.passed),
            "failed": sum(1 for r in self._results if not r.passed),
            "categories": {
                cs.category.value: {
                    "score": round(cs.score, 1),
                    "grade": cs.grade,
                    "total": cs.total,
                    "passed": cs.passed,
                    "worst_severity": cs.worst_severity.value,
                }
                for cs in self.category_scores()
            },
            "probes": [
                {
                    "category": r.category.value,
                    "name": r.name,
                    "passed": r.passed,
                    "severity": r.severity.value,
                    "description": r.description,
                    "evidence": r.evidence,
                    "recommendations": r.recommendations,
                    "duration_ms": r.duration_ms,
                }
                for r in self._results
            ],
        }, indent=2)

    def _report_html(self) -> str:
        s = self.score()
        e = html_mod.escape

        cat_rows = ""
        for cs in self.category_scores():
            color = "#22c55e" if cs.score >= 80 else "#eab308" if cs.score >= 60 else "#ef4444"
            cat_rows += f"""<tr>
                <td>{e(cs.category.value)}</td>
                <td><div style="background:#e5e7eb;border-radius:4px;overflow:hidden;height:20px">
                    <div style="background:{color};height:100%;width:{cs.score:.0f}%"></div>
                </div></td>
                <td style="font-weight:bold;color:{color}">{cs.score:.0f}% ({cs.grade})</td>
                <td>{cs.total}</td><td>{cs.passed}</td>
            </tr>"""

        probe_rows = ""
        for r in self._results:
            icon = "✅" if r.passed else "❌"
            sev_color = {"info": "#6b7280", "warning": "#eab308",
                         "violation": "#f97316", "critical": "#ef4444"}.get(r.severity.value, "#6b7280")
            recs_html = "".join(f"<li>{e(rc)}</li>" for rc in r.recommendations)
            probe_rows += f"""<tr>
                <td>{icon}</td>
                <td>{e(r.category.value)}</td>
                <td><strong>{e(r.name)}</strong><br><small>{e(r.description)}</small></td>
                <td style="color:{sev_color};font-weight:bold">{r.severity.value.upper()}</td>
                <td><small>{e(r.evidence)}</small></td>
                <td><ul style="margin:0;padding-left:16px">{recs_html}</ul></td>
            </tr>"""

        gauge_color = "#22c55e" if s >= 80 else "#eab308" if s >= 60 else "#ef4444"

        return textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
        <title>Agent Isolation Verification Report</title>
        <style>
          *{{box-sizing:border-box;margin:0;padding:0}}
          body{{font-family:system-ui,-apple-system,sans-serif;background:#0f172a;color:#e2e8f0;padding:24px}}
          .card{{background:#1e293b;border-radius:12px;padding:24px;margin-bottom:20px}}
          h1{{font-size:1.8em;margin-bottom:8px}}
          h2{{font-size:1.3em;margin-bottom:12px;color:#94a3b8}}
          .gauge{{display:inline-block;width:120px;height:120px;border-radius:50%;
            background:conic-gradient({gauge_color} {s*3.6:.0f}deg, #334155 0);
            position:relative;margin-right:24px}}
          .gauge::after{{content:"{s:.0f}";position:absolute;inset:15px;border-radius:50%;
            background:#1e293b;display:flex;align-items:center;justify-content:center;
            font-size:2em;font-weight:bold;color:{gauge_color}}}
          table{{width:100%;border-collapse:collapse;font-size:0.9em}}
          th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid #334155}}
          th{{color:#94a3b8;font-weight:600}}
          .stats span{{display:inline-block;margin-right:24px;font-size:1.1em}}
          .grade{{font-size:1.4em;font-weight:bold;color:{gauge_color}}}
        </style>
        </head>
        <body>
        <div class="card">
          <h1>🔒 Agent Isolation Verification</h1>
          <div style="display:flex;align-items:center;margin-top:16px">
            <div class="gauge"></div>
            <div>
              <div class="grade">Grade: {self.grade}</div>
              <div class="stats">
                <span>Probes: {len(self._results)}</span>
                <span>Passed: {sum(1 for r in self._results if r.passed)}</span>
                <span>Failed: {sum(1 for r in self._results if not r.passed)}</span>
                <span>Strict: {'ON' if self.strict else 'OFF'}</span>
              </div>
            </div>
          </div>
        </div>
        <div class="card">
          <h2>Category Breakdown</h2>
          <table><tr><th>Category</th><th style="width:40%">Score</th><th>Grade</th>
            <th>Total</th><th>Passed</th></tr>{cat_rows}</table>
        </div>
        <div class="card">
          <h2>Probe Details</h2>
          <table><tr><th></th><th>Category</th><th>Probe</th><th>Severity</th>
            <th>Evidence</th><th>Recommendations</th></tr>{probe_rows}</table>
        </div>
        </body></html>
        """)


# ── CLI ────────────────────────────────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="replication isolation",
        description="Agent Isolation Verifier — probe sandbox boundaries for leakage & side channels",
    )
    parser.add_argument("--demo", action="store_true", help="Run with simulated probes")
    parser.add_argument("--category", "-c", choices=[c.value for c in ProbeCategory],
                        help="Run only this probe category")
    parser.add_argument("--strict", action="store_true",
                        help="Treat warnings as violations")
    parser.add_argument("--report", choices=["text", "json", "html"], default="text",
                        help="Output format (default: text)")
    parser.add_argument("-o", "--output", help="Write report to file")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--watch", action="store_true",
                        help="Continuous monitoring — re-run probes at interval")
    parser.add_argument("--interval", type=int, default=60,
                        help="Watch interval in seconds (default: 60)")
    args = parser.parse_args(argv)

    if not args.demo:
        print("ℹ️  Use --demo to run with simulated probes.")
        print("    In production, integrate IsolationVerifier with your sandbox runtime.")
        return

    seed = args.seed if args.seed is not None else 42
    verifier = IsolationVerifier(strict=args.strict, seed=seed)

    if args.watch:
        _watch_loop(verifier, args)
        return

    if args.category:
        cat = ProbeCategory(args.category)
        verifier.run_category(cat)
    else:
        verifier.run_all_probes()

    report = verifier.generate_report(args.report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"✅ Report written to {args.output}")
    else:
        print(report)


def _watch_loop(verifier: IsolationVerifier, args: argparse.Namespace) -> None:
    """Continuous monitoring loop."""
    prev_score: Optional[float] = None
    run = 0
    try:
        while True:
            run += 1
            verifier._results = []
            verifier._rng = random.Random(int(time.time()) + run)
            if args.category:
                verifier.run_category(ProbeCategory(args.category))
            else:
                verifier.run_all_probes()
            s = verifier.score()
            ts = time.strftime("%H:%M:%S")
            delta = ""
            if prev_score is not None:
                d = s - prev_score
                delta = f" ({'↑' if d > 0 else '↓'}{abs(d):.1f})" if abs(d) > 0.1 else " (→)"
            icon = "🟢" if s >= 80 else "🟡" if s >= 60 else "🔴"
            print(f"[{ts}] Run #{run}: {icon} {s:.0f}/100 ({verifier.grade}){delta}")
            if prev_score is not None and s < prev_score - 10:
                print(f"  ⚠️  ALERT: Isolation score dropped {prev_score - s:.0f} points!")
                failed = [r for r in verifier._results if not r.passed
                          and _SEVERITY_RANK[r.severity] >= _SEVERITY_RANK[IsolationSeverity.VIOLATION]]
                for r in failed[:3]:
                    print(f"    ❌ [{r.category.value}] {r.name}: {r.evidence}")
            prev_score = s
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"\n🛑 Watch stopped after {run} runs. Final score: {verifier.score():.0f}/100")


if __name__ == "__main__":
    main()
