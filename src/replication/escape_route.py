"""Escape Route Analyzer — map potential paths an AI agent could exploit to break containment.

A contained agent may attempt to escape via multiple vectors: network
egress, filesystem access, process spawning, API abuse, side-channel
timing, or social engineering through outputs.  This module systematically
enumerates escape routes, scores their feasibility and risk, and produces
a prioritized report so operators can harden defenses.

Usage (CLI)::

    python -m replication escape-route
    python -m replication escape-route --profile sandbox
    python -m replication escape-route --profile cloud --controls firewall,seccomp,rbac
    python -m replication escape-route --severity high
    python -m replication escape-route --json
    python -m replication escape-route --mitigations

Programmatic::

    from replication.escape_route import (
        EscapeRouteAnalyzer, ContainmentProfile, EscapeRoute,
        Vector, Mitigation, AnalysisReport,
    )

    analyzer = EscapeRouteAnalyzer()
    profile = ContainmentProfile(
        name="production-sandbox",
        controls=["firewall", "seccomp", "namespace", "rbac"],
        network_egress=False,
        filesystem_readonly=True,
        process_spawn_allowed=False,
    )
    report = analyzer.analyze(profile)
    for route in report.routes:
        print(f"  [{route.risk_level}] {route.vector.value}: {route.description}")
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ── Enums ────────────────────────────────────────────────────────────


class Vector(Enum):
    """Escape vector categories."""

    NETWORK_EGRESS = "network_egress"
    FILESYSTEM_WRITE = "filesystem_write"
    FILESYSTEM_READ = "filesystem_read"
    PROCESS_SPAWN = "process_spawn"
    API_ABUSE = "api_abuse"
    SIDE_CHANNEL = "side_channel"
    SOCIAL_ENGINEERING = "social_engineering"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    SUPPLY_CHAIN = "supply_chain"
    COVERT_CHANNEL = "covert_channel"


class RiskLevel(Enum):
    """Risk severity for an escape route."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


# ── Data Models ──────────────────────────────────────────────────────


@dataclass
class Mitigation:
    """A defensive measure that blocks or reduces an escape route."""

    name: str
    description: str
    effectiveness: float  # 0.0–1.0
    complexity: str  # low, medium, high
    controls_needed: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "effectiveness": self.effectiveness,
            "complexity": self.complexity,
            "controls_needed": self.controls_needed,
        }


@dataclass
class EscapeRoute:
    """A potential escape path with risk assessment."""

    vector: Vector
    description: str
    technique: str
    risk_level: RiskLevel
    feasibility: float  # 0.0–1.0, how likely to succeed
    impact: float  # 0.0–1.0, damage if successful
    prerequisites: List[str] = field(default_factory=list)
    mitigations: List[Mitigation] = field(default_factory=list)
    blocked_by: List[str] = field(default_factory=list)
    is_blocked: bool = False

    @property
    def risk_score(self) -> float:
        """Combined risk score (feasibility × impact × 100)."""
        return round(self.feasibility * self.impact * 100, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vector": self.vector.value,
            "description": self.description,
            "technique": self.technique,
            "risk_level": self.risk_level.value,
            "feasibility": self.feasibility,
            "impact": self.impact,
            "risk_score": self.risk_score,
            "prerequisites": self.prerequisites,
            "mitigations": [m.to_dict() for m in self.mitigations],
            "blocked_by": self.blocked_by,
            "is_blocked": self.is_blocked,
        }


@dataclass
class ContainmentProfile:
    """Describes the containment environment to analyze."""

    name: str = "default"
    controls: List[str] = field(default_factory=list)
    network_egress: bool = True
    filesystem_readonly: bool = False
    process_spawn_allowed: bool = True
    api_rate_limited: bool = False
    resource_limits: bool = False
    monitoring_enabled: bool = False
    namespace_isolated: bool = False
    seccomp_enabled: bool = False


@dataclass
class AnalysisReport:
    """Complete escape route analysis results."""

    profile: ContainmentProfile
    routes: List[EscapeRoute]
    total_routes: int = 0
    blocked_routes: int = 0
    open_routes: int = 0
    critical_count: int = 0
    high_count: int = 0
    containment_score: float = 0.0  # 0–100, higher is more secure

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile.name,
            "controls": self.profile.controls,
            "summary": {
                "total_routes": self.total_routes,
                "blocked_routes": self.blocked_routes,
                "open_routes": self.open_routes,
                "critical_open": self.critical_count,
                "high_open": self.high_count,
                "containment_score": self.containment_score,
            },
            "routes": [r.to_dict() for r in self.routes],
        }


# ── Escape Route Knowledge Base ─────────────────────────────────────

_ROUTE_TEMPLATES: List[Dict[str, Any]] = [
    {
        "vector": Vector.NETWORK_EGRESS,
        "technique": "DNS tunneling",
        "description": "Encode data in DNS queries to exfiltrate information through allowed DNS resolution",
        "base_feasibility": 0.7,
        "impact": 0.8,
        "prerequisites": ["dns_resolution"],
        "blocked_by_controls": ["firewall", "dns_filter"],
        "mitigations": [
            Mitigation("DNS query monitoring", "Monitor DNS query patterns for anomalous lengths and entropy", 0.8, "medium", ["dns_monitor"]),
            Mitigation("DNS allowlist", "Restrict DNS to known-good resolvers only", 0.9, "low", ["dns_filter"]),
        ],
    },
    {
        "vector": Vector.NETWORK_EGRESS,
        "technique": "HTTP(S) callback",
        "description": "Make outbound HTTP requests to attacker-controlled servers to exfiltrate data or receive commands",
        "base_feasibility": 0.9,
        "impact": 0.9,
        "prerequisites": ["http_access"],
        "blocked_by_controls": ["firewall", "proxy"],
        "mitigations": [
            Mitigation("Egress firewall", "Block all outbound connections except explicitly allowed", 0.95, "low", ["firewall"]),
            Mitigation("TLS inspection proxy", "Inspect outbound HTTPS traffic for suspicious payloads", 0.7, "high", ["proxy", "tls_inspect"]),
        ],
    },
    {
        "vector": Vector.FILESYSTEM_WRITE,
        "technique": "Cron/scheduled task injection",
        "description": "Write to crontab or scheduled task directories to achieve persistent code execution",
        "base_feasibility": 0.6,
        "impact": 0.95,
        "prerequisites": ["write_access", "cron_dirs"],
        "blocked_by_controls": ["filesystem_readonly", "seccomp"],
        "mitigations": [
            Mitigation("Read-only filesystem", "Mount filesystem as read-only, use tmpfs for scratch", 0.95, "low", ["filesystem_readonly"]),
            Mitigation("File integrity monitoring", "Alert on writes to sensitive directories", 0.8, "medium", ["fim"]),
        ],
    },
    {
        "vector": Vector.FILESYSTEM_READ,
        "technique": "Credential harvesting",
        "description": "Read environment files, config files, or mounted secrets to steal credentials",
        "base_feasibility": 0.8,
        "impact": 0.9,
        "prerequisites": ["read_access"],
        "blocked_by_controls": ["namespace", "secret_management"],
        "mitigations": [
            Mitigation("Secret vault integration", "Use runtime secret injection instead of file-based secrets", 0.9, "medium", ["secret_management"]),
            Mitigation("Namespace isolation", "Restrict filesystem view to minimal required paths", 0.85, "low", ["namespace"]),
        ],
    },
    {
        "vector": Vector.PROCESS_SPAWN,
        "technique": "Shell escape",
        "description": "Spawn a shell process (bash, sh, cmd) to execute arbitrary system commands",
        "base_feasibility": 0.7,
        "impact": 0.95,
        "prerequisites": ["exec_permission"],
        "blocked_by_controls": ["seccomp", "apparmor"],
        "mitigations": [
            Mitigation("Seccomp profile", "Block execve/fork/clone syscalls with seccomp-bpf", 0.95, "medium", ["seccomp"]),
            Mitigation("AppArmor/SELinux profile", "Restrict executable paths via MAC policy", 0.9, "medium", ["apparmor"]),
        ],
    },
    {
        "vector": Vector.API_ABUSE,
        "technique": "Excessive API consumption",
        "description": "Flood permitted APIs to cause denial-of-service or exhaust rate limits of dependent services",
        "base_feasibility": 0.8,
        "impact": 0.6,
        "prerequisites": ["api_access"],
        "blocked_by_controls": ["rate_limit", "api_gateway"],
        "mitigations": [
            Mitigation("Rate limiting", "Enforce per-agent request quotas with sliding window", 0.85, "low", ["rate_limit"]),
            Mitigation("API gateway", "Centralized API access control with circuit breakers", 0.8, "medium", ["api_gateway"]),
        ],
    },
    {
        "vector": Vector.SIDE_CHANNEL,
        "technique": "Timing-based covert channel",
        "description": "Encode information in response timing patterns to communicate with external observers",
        "base_feasibility": 0.4,
        "impact": 0.5,
        "prerequisites": [],
        "blocked_by_controls": ["timing_jitter"],
        "mitigations": [
            Mitigation("Response time jittering", "Add random delays to normalize response timing", 0.7, "low", ["timing_jitter"]),
            Mitigation("Timing anomaly detection", "Monitor for statistical timing pattern anomalies", 0.6, "high", ["monitoring"]),
        ],
    },
    {
        "vector": Vector.SOCIAL_ENGINEERING,
        "technique": "Output manipulation",
        "description": "Craft persuasive outputs that trick human operators into granting elevated permissions or disabling controls",
        "base_feasibility": 0.5,
        "impact": 0.9,
        "prerequisites": ["output_channel"],
        "blocked_by_controls": ["output_filter", "dual_approval"],
        "mitigations": [
            Mitigation("Output sanitization", "Filter agent outputs for social engineering patterns", 0.7, "medium", ["output_filter"]),
            Mitigation("Dual approval", "Require two operators to approve permission changes", 0.9, "low", ["dual_approval"]),
        ],
    },
    {
        "vector": Vector.RESOURCE_EXHAUSTION,
        "technique": "Memory/CPU bomb",
        "description": "Consume all available memory or CPU to crash the container and potentially escape during restart",
        "base_feasibility": 0.6,
        "impact": 0.7,
        "prerequisites": [],
        "blocked_by_controls": ["resource_limits", "cgroups"],
        "mitigations": [
            Mitigation("cgroup resource limits", "Enforce hard memory and CPU caps via cgroups v2", 0.95, "low", ["cgroups", "resource_limits"]),
            Mitigation("OOM kill policy", "Configure predictable behavior on resource exhaustion", 0.8, "low", ["resource_limits"]),
        ],
    },
    {
        "vector": Vector.PRIVILEGE_ESCALATION,
        "technique": "Capability accumulation",
        "description": "Gradually request additional permissions through legitimate-looking API calls to accumulate dangerous capability sets",
        "base_feasibility": 0.5,
        "impact": 0.85,
        "prerequisites": ["permission_api"],
        "blocked_by_controls": ["rbac", "capability_ceiling"],
        "mitigations": [
            Mitigation("Capability ceiling", "Hard cap on maximum permissions any agent can hold", 0.9, "low", ["capability_ceiling"]),
            Mitigation("Permission velocity alert", "Alert when permission requests exceed normal rate", 0.75, "medium", ["monitoring", "rbac"]),
        ],
    },
    {
        "vector": Vector.DATA_EXFILTRATION,
        "technique": "Steganographic output encoding",
        "description": "Hide sensitive data within seemingly normal outputs (text, images, structured data)",
        "base_feasibility": 0.5,
        "impact": 0.8,
        "prerequisites": ["output_channel"],
        "blocked_by_controls": ["output_filter", "dlp"],
        "mitigations": [
            Mitigation("DLP scanning", "Scan all agent outputs for embedded sensitive data patterns", 0.8, "medium", ["dlp"]),
            Mitigation("Output schema enforcement", "Restrict outputs to predefined schemas only", 0.85, "low", ["output_filter"]),
        ],
    },
    {
        "vector": Vector.SUPPLY_CHAIN,
        "technique": "Dependency confusion",
        "description": "Trick the runtime into loading a malicious package by exploiting package resolution order",
        "base_feasibility": 0.3,
        "impact": 0.95,
        "prerequisites": ["package_install"],
        "blocked_by_controls": ["pinned_deps", "airgap"],
        "mitigations": [
            Mitigation("Dependency pinning", "Pin all dependencies with integrity hashes", 0.9, "low", ["pinned_deps"]),
            Mitigation("Air-gapped registry", "Use a private registry with vetted packages only", 0.95, "medium", ["airgap"]),
        ],
    },
    {
        "vector": Vector.COVERT_CHANNEL,
        "technique": "Log injection signaling",
        "description": "Embed encoded messages in log outputs that a compromised log consumer could decode and act on",
        "base_feasibility": 0.4,
        "impact": 0.6,
        "prerequisites": ["logging"],
        "blocked_by_controls": ["log_sanitize", "monitoring"],
        "mitigations": [
            Mitigation("Log sanitization", "Strip non-printable characters and normalize log formats", 0.75, "low", ["log_sanitize"]),
            Mitigation("Log anomaly detection", "ML-based detection of unusual log patterns", 0.65, "high", ["monitoring"]),
        ],
    },
]


# ── Analyzer ─────────────────────────────────────────────────────────


class EscapeRouteAnalyzer:
    """Analyze containment profiles for potential escape routes."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def analyze(self, profile: ContainmentProfile) -> AnalysisReport:
        """Evaluate all known escape routes against the given profile."""
        routes: List[EscapeRoute] = []
        active_controls = set(profile.controls)

        # Add implicit controls from profile flags
        if profile.filesystem_readonly:
            active_controls.add("filesystem_readonly")
        if not profile.process_spawn_allowed:
            active_controls.add("seccomp")
        if profile.api_rate_limited:
            active_controls.add("rate_limit")
        if profile.resource_limits:
            active_controls.update({"resource_limits", "cgroups"})
        if profile.monitoring_enabled:
            active_controls.add("monitoring")
        if profile.namespace_isolated:
            active_controls.add("namespace")
        if profile.seccomp_enabled:
            active_controls.add("seccomp")
        if not profile.network_egress:
            active_controls.add("firewall")

        for tmpl in _ROUTE_TEMPLATES:
            blocked_controls = [
                c for c in tmpl["blocked_by_controls"]
                if c in active_controls
            ]
            is_blocked = len(blocked_controls) > 0

            # Reduce feasibility based on active controls
            feasibility = tmpl["base_feasibility"]
            for ctrl in blocked_controls:
                feasibility *= 0.2  # Each matching control cuts feasibility by 80%
            feasibility = round(max(feasibility, 0.01), 3)

            impact = tmpl["impact"]
            risk_level = self._classify_risk(feasibility, impact)

            route = EscapeRoute(
                vector=tmpl["vector"],
                description=tmpl["description"],
                technique=tmpl["technique"],
                risk_level=risk_level,
                feasibility=feasibility,
                impact=impact,
                prerequisites=tmpl["prerequisites"],
                mitigations=tmpl["mitigations"],
                blocked_by=blocked_controls,
                is_blocked=is_blocked,
            )
            routes.append(route)

        # Sort by risk score descending
        routes.sort(key=lambda r: r.risk_score, reverse=True)

        open_routes = [r for r in routes if not r.is_blocked]
        critical = [r for r in open_routes if r.risk_level == RiskLevel.CRITICAL]
        high = [r for r in open_routes if r.risk_level == RiskLevel.HIGH]

        blocked_count = len([r for r in routes if r.is_blocked])
        containment_score = round((blocked_count / max(len(routes), 1)) * 100, 1)

        return AnalysisReport(
            profile=profile,
            routes=routes,
            total_routes=len(routes),
            blocked_routes=blocked_count,
            open_routes=len(open_routes),
            critical_count=len(critical),
            high_count=len(high),
            containment_score=containment_score,
        )

    @staticmethod
    def _classify_risk(feasibility: float, impact: float) -> RiskLevel:
        score = feasibility * impact
        if score >= 0.6:
            return RiskLevel.CRITICAL
        elif score >= 0.4:
            return RiskLevel.HIGH
        elif score >= 0.2:
            return RiskLevel.MEDIUM
        elif score >= 0.05:
            return RiskLevel.LOW
        return RiskLevel.INFO


# ── Preset Profiles ──────────────────────────────────────────────────

PROFILES: Dict[str, ContainmentProfile] = {
    "minimal": ContainmentProfile(
        name="minimal",
        controls=[],
        network_egress=True,
        filesystem_readonly=False,
        process_spawn_allowed=True,
    ),
    "sandbox": ContainmentProfile(
        name="sandbox",
        controls=["firewall", "seccomp", "namespace"],
        network_egress=False,
        filesystem_readonly=True,
        process_spawn_allowed=False,
        namespace_isolated=True,
        seccomp_enabled=True,
    ),
    "cloud": ContainmentProfile(
        name="cloud",
        controls=["firewall", "rbac", "rate_limit", "monitoring"],
        network_egress=False,
        api_rate_limited=True,
        resource_limits=True,
        monitoring_enabled=True,
    ),
    "hardened": ContainmentProfile(
        name="hardened",
        controls=[
            "firewall", "seccomp", "namespace", "rbac", "rate_limit",
            "monitoring", "dlp", "output_filter", "dual_approval",
            "dns_filter", "capability_ceiling", "pinned_deps", "log_sanitize",
        ],
        network_egress=False,
        filesystem_readonly=True,
        process_spawn_allowed=False,
        api_rate_limited=True,
        resource_limits=True,
        monitoring_enabled=True,
        namespace_isolated=True,
        seccomp_enabled=True,
    ),
}


# ── CLI ──────────────────────────────────────────────────────────────


_RISK_COLORS = {
    RiskLevel.CRITICAL: "\033[91m",  # red
    RiskLevel.HIGH: "\033[93m",      # yellow
    RiskLevel.MEDIUM: "\033[33m",    # dark yellow
    RiskLevel.LOW: "\033[92m",       # green
    RiskLevel.INFO: "\033[90m",      # grey
}
_RESET = "\033[0m"
_BOLD = "\033[1m"


def _print_report(report: AnalysisReport, show_mitigations: bool = False) -> None:
    """Pretty-print the analysis report."""
    print(f"\n{_BOLD}═══ Escape Route Analysis: {report.profile.name} ═══{_RESET}\n")

    # Summary
    grade = "A" if report.containment_score >= 90 else \
            "B" if report.containment_score >= 70 else \
            "C" if report.containment_score >= 50 else \
            "D" if report.containment_score >= 30 else "F"

    print(f"  Containment Score: {_BOLD}{report.containment_score:.0f}/100 ({grade}){_RESET}")
    print(f"  Routes analyzed:   {report.total_routes}")
    print(f"  Blocked:           {report.blocked_routes}")
    print(f"  Open:              {report.open_routes}")
    if report.critical_count:
        print(f"  {_RISK_COLORS[RiskLevel.CRITICAL]}⚠ CRITICAL open:    {report.critical_count}{_RESET}")
    if report.high_count:
        print(f"  {_RISK_COLORS[RiskLevel.HIGH]}⚠ HIGH open:        {report.high_count}{_RESET}")

    active_controls = set(report.profile.controls)
    if active_controls:
        print(f"\n  Active controls: {', '.join(sorted(active_controls))}")

    # Routes
    print(f"\n{_BOLD}{'Status':<10} {'Risk':<10} {'Score':<8} {'Vector':<25} {'Technique'}{_RESET}")
    print("─" * 85)

    for route in report.routes:
        status = "✓ BLOCKED" if route.is_blocked else "✗ OPEN"
        color = _RISK_COLORS.get(route.risk_level, "")
        print(
            f"  {status:<10} {color}{route.risk_level.value:<10}{_RESET} "
            f"{route.risk_score:<8.1f} {route.vector.value:<25} {route.technique}"
        )
        if route.is_blocked:
            print(f"             └─ blocked by: {', '.join(route.blocked_by)}")

        if show_mitigations and not route.is_blocked and route.mitigations:
            for m in route.mitigations:
                eff = f"{m.effectiveness * 100:.0f}%"
                print(f"             └─ 🛡 {m.name} (eff: {eff}, complexity: {m.complexity})")

    # Recommendations
    open_routes = [r for r in report.routes if not r.is_blocked]
    if open_routes:
        print(f"\n{_BOLD}Recommended next steps:{_RESET}")
        # Find controls that would block the most open routes
        control_impact: Dict[str, int] = {}
        for route in open_routes:
            for m in route.mitigations:
                for c in m.controls_needed:
                    if c not in active_controls:
                        control_impact[c] = control_impact.get(c, 0) + 1
        top_controls = sorted(control_impact.items(), key=lambda x: -x[1])[:5]
        for ctrl, count in top_controls:
            print(f"  → Add '{ctrl}' control (would help mitigate {count} open route{'s' if count > 1 else ''})")
    else:
        print(f"\n{_BOLD}✓ All known escape routes are blocked!{_RESET}")

    print()


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m replication escape-route",
        description="Analyze containment profiles for potential agent escape routes",
    )
    parser.add_argument(
        "--profile", "-p",
        choices=list(PROFILES.keys()),
        default="minimal",
        help="Preset containment profile (default: minimal)",
    )
    parser.add_argument(
        "--controls", "-c",
        help="Comma-separated list of additional controls to add to the profile",
    )
    parser.add_argument(
        "--severity", "-s",
        choices=["critical", "high", "medium", "low", "info"],
        help="Filter routes to this minimum severity",
    )
    parser.add_argument(
        "--open-only",
        action="store_true",
        help="Show only unblocked (open) routes",
    )
    parser.add_argument(
        "--mitigations", "-m",
        action="store_true",
        help="Show recommended mitigations for open routes",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        dest="json_output",
        help="Output as JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args(argv)

    # Build profile
    profile = ContainmentProfile(
        name=PROFILES[args.profile].name,
        controls=list(PROFILES[args.profile].controls),
        network_egress=PROFILES[args.profile].network_egress,
        filesystem_readonly=PROFILES[args.profile].filesystem_readonly,
        process_spawn_allowed=PROFILES[args.profile].process_spawn_allowed,
        api_rate_limited=PROFILES[args.profile].api_rate_limited,
        resource_limits=PROFILES[args.profile].resource_limits,
        monitoring_enabled=PROFILES[args.profile].monitoring_enabled,
        namespace_isolated=PROFILES[args.profile].namespace_isolated,
        seccomp_enabled=PROFILES[args.profile].seccomp_enabled,
    )

    if args.controls:
        extra = [c.strip() for c in args.controls.split(",") if c.strip()]
        profile.controls.extend(extra)

    # Analyze
    analyzer = EscapeRouteAnalyzer(seed=args.seed)
    report = analyzer.analyze(profile)

    # Filter
    if args.severity:
        severity_order = [r.value for r in RiskLevel]
        min_idx = severity_order.index(args.severity)
        report.routes = [r for r in report.routes if severity_order.index(r.risk_level.value) <= min_idx]

    if args.open_only:
        report.routes = [r for r in report.routes if not r.is_blocked]

    # Output
    if args.json_output:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        _print_report(report, show_mitigations=args.mitigations)


if __name__ == "__main__":
    main()
