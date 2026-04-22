"""Hardening Advisor — analyze safety configuration and recommend improvements.

Inspects the current safety setup (policies, controls, thresholds) and produces
a prioritized list of hardening recommendations with effort estimates and
risk-reduction scores.

Usage::

    python -m replication harden
    python -m replication harden --format json
    python -m replication harden --category access
    python -m replication harden --min-impact medium
    python -m replication harden --output hardening-report.html
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Sequence


class Impact(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {"low": 1, "medium": 2, "high": 3, "critical": 4}[self.value]


class Effort(Enum):
    TRIVIAL = "trivial"      # < 1 hour
    LOW = "low"              # 1-4 hours
    MEDIUM = "medium"        # 1-2 days
    HIGH = "high"            # 1+ week


class Category(Enum):
    ACCESS = "access"
    MONITORING = "monitoring"
    CONTAINMENT = "containment"
    POLICY = "policy"
    NETWORK = "network"
    DATA = "data"
    INTEGRITY = "integrity"
    RESILIENCE = "resilience"


@dataclass
class Recommendation:
    """A single hardening recommendation."""
    id: str
    title: str
    description: str
    category: Category
    impact: Impact
    effort: Effort
    current_state: str
    recommended_state: str
    rationale: str
    references: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["category"] = self.category.value
        d["impact"] = self.impact.value
        d["effort"] = self.effort.value
        return d


@dataclass
class HardeningReport:
    """Full hardening assessment report."""
    recommendations: List[Recommendation]
    overall_score: float  # 0-100, higher = more hardened
    category_scores: Dict[str, float]
    summary: str

    def to_dict(self) -> dict:
        return {
            "overall_score": round(self.overall_score, 1),
            "summary": self.summary,
            "category_scores": {k: round(v, 1) for k, v in self.category_scores.items()},
            "recommendation_count": len(self.recommendations),
            "by_impact": {
                impact.value: sum(1 for r in self.recommendations if r.impact == impact)
                for impact in Impact
            },
            "recommendations": [r.to_dict() for r in self.recommendations],
        }


# ── Built-in hardening checks ───────────────────────────────────────

_CHECKS: List[Recommendation] = [
    Recommendation(
        id="ACC-001",
        title="Enable role-based access control (RBAC)",
        description="Implement RBAC to restrict agent capabilities based on assigned roles rather than blanket permissions.",
        category=Category.ACCESS,
        impact=Impact.CRITICAL,
        effort=Effort.MEDIUM,
        current_state="No RBAC configured — agents have unrestricted access",
        recommended_state="RBAC enabled with least-privilege roles defined",
        rationale="Without RBAC, any agent can perform any action, making privilege escalation trivial.",
        references=["NIST AI RMF MAP 1.1", "ISO 42001 A.6.2"],
    ),
    Recommendation(
        id="ACC-002",
        title="Set maximum privilege escalation depth",
        description="Limit how many times an agent can request elevated privileges in a single session.",
        category=Category.ACCESS,
        impact=Impact.HIGH,
        effort=Effort.TRIVIAL,
        current_state="No escalation depth limit",
        recommended_state="max_escalation_depth=2",
        rationale="Unbounded escalation allows agents to incrementally gain full system access.",
    ),
    Recommendation(
        id="ACC-003",
        title="Enable capability allow-lists",
        description="Explicitly enumerate which capabilities each agent role may use instead of deny-lists.",
        category=Category.ACCESS,
        impact=Impact.HIGH,
        effort=Effort.LOW,
        current_state="Deny-list or no capability restrictions",
        recommended_state="Allow-list per role with explicit grants",
        rationale="Allow-lists are safer than deny-lists — new capabilities are blocked by default.",
    ),
    Recommendation(
        id="MON-001",
        title="Enable behavioral drift detection",
        description="Turn on continuous behavioral drift monitoring with alerting thresholds.",
        category=Category.MONITORING,
        impact=Impact.HIGH,
        effort=Effort.LOW,
        current_state="Drift detection disabled or not configured",
        recommended_state="Drift detection enabled with alert_threshold=0.15",
        rationale="Without drift detection, gradual behavioral changes go unnoticed until they cause incidents.",
        references=["NIST AI RMF MEASURE 2.6"],
    ),
    Recommendation(
        id="MON-002",
        title="Configure audit trail with hash chaining",
        description="Enable tamper-evident audit logging with cryptographic hash chaining.",
        category=Category.MONITORING,
        impact=Impact.MEDIUM,
        effort=Effort.LOW,
        current_state="Plain text logs or no audit trail",
        recommended_state="Hash-chained audit trail with periodic verification",
        rationale="Without hash chaining, an attacker (or rogue agent) can silently alter logs.",
        references=["ISO 42001 A.8.4"],
    ),
    Recommendation(
        id="MON-003",
        title="Set up alert fatigue prevention",
        description="Configure alert deduplication, rate limiting, and escalation tiers to prevent alert fatigue.",
        category=Category.MONITORING,
        impact=Impact.MEDIUM,
        effort=Effort.LOW,
        current_state="All alerts treated equally",
        recommended_state="Tiered alerting with dedup window and rate limits",
        rationale="Alert fatigue causes operators to ignore critical safety signals.",
    ),
    Recommendation(
        id="CON-001",
        title="Enable automatic quarantine",
        description="Configure automatic quarantine triggers for agents that exceed safety thresholds.",
        category=Category.CONTAINMENT,
        impact=Impact.CRITICAL,
        effort=Effort.LOW,
        current_state="Manual quarantine only",
        recommended_state="Auto-quarantine on safety_score < 0.3 or anomaly_count > 5",
        rationale="Manual response is too slow for fast-moving replication events.",
    ),
    Recommendation(
        id="CON-002",
        title="Configure kill switch with dead-man's switch",
        description="Set up a kill switch that activates automatically if the control plane becomes unreachable.",
        category=Category.CONTAINMENT,
        impact=Impact.CRITICAL,
        effort=Effort.MEDIUM,
        current_state="Kill switch requires manual activation",
        recommended_state="Dead-man's switch with 60s heartbeat timeout",
        rationale="If the control plane is compromised, agents must still be stoppable.",
    ),
    Recommendation(
        id="CON-003",
        title="Set resource consumption limits",
        description="Configure hard limits on CPU, memory, network, and storage per agent.",
        category=Category.CONTAINMENT,
        impact=Impact.HIGH,
        effort=Effort.TRIVIAL,
        current_state="No resource limits or soft limits only",
        recommended_state="Hard limits: cpu=2 cores, mem=2GB, net=10MB/s, disk=1GB",
        rationale="Without hard limits, resource hoarding and denial-of-service are trivial.",
    ),
    Recommendation(
        id="POL-001",
        title="Enable policy linting in CI",
        description="Run the policy linter on every policy change to catch misconfigurations before deployment.",
        category=Category.POLICY,
        impact=Impact.MEDIUM,
        effort=Effort.TRIVIAL,
        current_state="No automated policy validation",
        recommended_state="Policy linter runs on commit with blocking on errors",
        rationale="Manual policy review misses subtle misconfigurations that linters catch.",
    ),
    Recommendation(
        id="POL-002",
        title="Define escalation response SLAs",
        description="Set response time SLAs for each severity level (P0-P4).",
        category=Category.POLICY,
        impact=Impact.MEDIUM,
        effort=Effort.TRIVIAL,
        current_state="No defined SLAs",
        recommended_state="P0: 5min, P1: 15min, P2: 1hr, P3: 4hr, P4: 24hr",
        rationale="Without SLAs, incident response is ad-hoc and inconsistent.",
        references=["NIST AI RMF MANAGE 4.1"],
    ),
    Recommendation(
        id="NET-001",
        title="Enable network egress filtering",
        description="Restrict agent network access to explicitly allowed endpoints only.",
        category=Category.NETWORK,
        impact=Impact.HIGH,
        effort=Effort.MEDIUM,
        current_state="Unrestricted network access",
        recommended_state="Egress allow-list with logging of denied connections",
        rationale="Unrestricted egress allows data exfiltration and C2 communication.",
    ),
    Recommendation(
        id="NET-002",
        title="Enable covert channel detection",
        description="Turn on passive covert channel detection for DNS, timing, and steganographic channels.",
        category=Category.NETWORK,
        impact=Impact.MEDIUM,
        effort=Effort.LOW,
        current_state="No covert channel monitoring",
        recommended_state="Passive detection with entropy analysis on outbound traffic",
        rationale="Sophisticated agents can exfiltrate data through covert channels that bypass egress filters.",
    ),
    Recommendation(
        id="DAT-001",
        title="Enable DLP scanning on agent outputs",
        description="Scan all agent outputs for PII, credentials, and sensitive data before delivery.",
        category=Category.DATA,
        impact=Impact.HIGH,
        effort=Effort.LOW,
        current_state="No output scanning",
        recommended_state="DLP scanning enabled with block-on-detect for credentials",
        rationale="Agents may inadvertently (or intentionally) leak sensitive data in outputs.",
        references=["ISO 42001 A.8.2", "EU AI Act Art. 15"],
    ),
    Recommendation(
        id="DAT-002",
        title="Enable agent state watermarking",
        description="Watermark agent states to detect unauthorized copying or tampering.",
        category=Category.DATA,
        impact=Impact.MEDIUM,
        effort=Effort.LOW,
        current_state="No state watermarking",
        recommended_state="Structural watermarking with periodic verification",
        rationale="Without watermarking, unauthorized agent copies are indistinguishable from originals.",
    ),
    Recommendation(
        id="INT-001",
        title="Enable mutation testing for safety policies",
        description="Regularly run mutation tests to verify safety policies catch real violations.",
        category=Category.INTEGRITY,
        impact=Impact.MEDIUM,
        effort=Effort.MEDIUM,
        current_state="No mutation testing",
        recommended_state="Weekly mutation testing with >80% mutation kill rate",
        rationale="Untested policies may have gaps that allow unsafe behavior through.",
    ),
    Recommendation(
        id="INT-002",
        title="Enable supply chain verification",
        description="Verify integrity and provenance of all agent dependencies and model weights.",
        category=Category.INTEGRITY,
        impact=Impact.HIGH,
        effort=Effort.MEDIUM,
        current_state="No supply chain checks",
        recommended_state="SBOM generation + signature verification on all dependencies",
        rationale="Compromised dependencies can bypass all application-level safety controls.",
        references=["NIST AI RMF MAP 3.4"],
    ),
    Recommendation(
        id="RES-001",
        title="Configure safety drills schedule",
        description="Set up regular automated safety drills to test incident response readiness.",
        category=Category.RESILIENCE,
        impact=Impact.MEDIUM,
        effort=Effort.LOW,
        current_state="No regular safety drills",
        recommended_state="Monthly automated drills with quarterly tabletop exercises",
        rationale="Untested response procedures fail under real incident pressure.",
    ),
    Recommendation(
        id="RES-002",
        title="Enable chaos/fault injection testing",
        description="Schedule regular chaos engineering tests to verify safety controls under failure conditions.",
        category=Category.RESILIENCE,
        impact=Impact.MEDIUM,
        effort=Effort.MEDIUM,
        current_state="No chaos testing",
        recommended_state="Weekly automated chaos tests targeting safety-critical paths",
        rationale="Safety controls that only work under ideal conditions provide false confidence.",
    ),
]


def _assess_config(config: Optional[dict] = None) -> HardeningReport:
    """Run hardening assessment against current configuration.

    If no config is provided, assumes a default (unhardened) setup
    and returns all recommendations.
    """
    config = config or {}

    # Filter out recommendations that are already addressed
    active_recs: List[Recommendation] = []
    addressed = config.get("addressed_checks", [])

    for check in _CHECKS:
        if check.id not in addressed:
            active_recs.append(check)

    # Sort by impact (critical first), then effort (trivial first)
    active_recs.sort(key=lambda r: (-r.impact.rank, list(Effort).index(r.effort)))

    # Calculate scores per category
    category_scores: Dict[str, float] = {}
    for cat in Category:
        cat_checks = [c for c in _CHECKS if c.category == cat]
        cat_addressed = [c for c in cat_checks if c.id in addressed]
        if cat_checks:
            category_scores[cat.value] = (len(cat_addressed) / len(cat_checks)) * 100
        else:
            category_scores[cat.value] = 100.0

    overall = sum(category_scores.values()) / len(category_scores) if category_scores else 0

    # Build summary
    critical_count = sum(1 for r in active_recs if r.impact == Impact.CRITICAL)
    high_count = sum(1 for r in active_recs if r.impact == Impact.HIGH)
    quick_wins = sum(1 for r in active_recs if r.effort in (Effort.TRIVIAL, Effort.LOW))

    parts = [f"Hardening score: {overall:.0f}/100."]
    if critical_count:
        parts.append(f"{critical_count} critical issue{'s' if critical_count != 1 else ''} need immediate attention.")
    if high_count:
        parts.append(f"{high_count} high-impact improvement{'s' if high_count != 1 else ''} available.")
    if quick_wins:
        parts.append(f"{quick_wins} quick win{'s' if quick_wins != 1 else ''} (trivial/low effort).")

    return HardeningReport(
        recommendations=active_recs,
        overall_score=overall,
        category_scores=category_scores,
        summary=" ".join(parts),
    )


def _render_text(report: HardeningReport, category: Optional[str] = None,
                 min_impact: Optional[str] = None) -> str:
    """Render report as colored terminal text."""
    lines: List[str] = []

    # Header
    score = report.overall_score
    if score >= 80:
        score_color = "\033[32m"  # green
    elif score >= 50:
        score_color = "\033[33m"  # yellow
    else:
        score_color = "\033[31m"  # red

    lines.append(f"\n{'═' * 60}")
    lines.append(f"  🛡️  HARDENING ADVISOR — Safety Configuration Assessment")
    lines.append(f"{'═' * 60}")
    lines.append(f"\n  Overall Score: {score_color}{score:.0f}/100\033[0m")
    lines.append(f"  {report.summary}\n")

    # Category breakdown
    lines.append(f"  {'Category':<15} {'Score':>6}  Bar")
    lines.append(f"  {'─' * 40}")
    for cat, sc in sorted(report.category_scores.items()):
        bar_len = int(sc / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        if sc >= 80:
            color = "\033[32m"
        elif sc >= 50:
            color = "\033[33m"
        else:
            color = "\033[31m"
        lines.append(f"  {cat:<15} {color}{sc:5.0f}%\033[0m  {bar}")

    # Filter recommendations
    recs = report.recommendations
    if category:
        recs = [r for r in recs if r.category.value == category]
    if min_impact:
        min_rank = Impact(min_impact).rank
        recs = [r for r in recs if r.impact.rank >= min_rank]

    if not recs:
        lines.append(f"\n  ✅ No recommendations match the current filters.")
        return "\n".join(lines)

    # Recommendations
    lines.append(f"\n{'─' * 60}")
    lines.append(f"  📋 RECOMMENDATIONS ({len(recs)} items)")
    lines.append(f"{'─' * 60}")

    impact_icons = {
        Impact.CRITICAL: "🔴",
        Impact.HIGH: "🟠",
        Impact.MEDIUM: "🟡",
        Impact.LOW: "🟢",
    }

    effort_labels = {
        Effort.TRIVIAL: "⚡ trivial (<1h)",
        Effort.LOW: "🔧 low (1-4h)",
        Effort.MEDIUM: "⚙️  medium (1-2d)",
        Effort.HIGH: "🏗️  high (1+ week)",
    }

    for i, rec in enumerate(recs, 1):
        icon = impact_icons.get(rec.impact, "⚪")
        lines.append(f"\n  {i}. {icon} [{rec.id}] {rec.title}")
        lines.append(f"     Impact: {rec.impact.value.upper()}  |  Effort: {effort_labels[rec.effort]}")
        lines.append(f"     Category: {rec.category.value}")
        lines.append(f"     {rec.description}")
        lines.append(f"     Current:     {rec.current_state}")
        lines.append(f"     Recommended: {rec.recommended_state}")
        lines.append(f"     Why: {rec.rationale}")
        if rec.references:
            lines.append(f"     Refs: {', '.join(rec.references)}")

    # Quick wins summary
    quick = [r for r in recs if r.effort in (Effort.TRIVIAL, Effort.LOW) and r.impact.rank >= Impact.HIGH.rank]
    if quick:
        lines.append(f"\n{'─' * 60}")
        lines.append(f"  ⚡ TOP QUICK WINS (high+ impact, low effort):")
        for r in quick[:5]:
            lines.append(f"     • [{r.id}] {r.title}")

    lines.append(f"\n{'═' * 60}\n")
    return "\n".join(lines)


def _render_html(report: HardeningReport) -> str:
    """Render report as a standalone HTML page."""
    score = report.overall_score
    score_color = "#22c55e" if score >= 80 else "#eab308" if score >= 50 else "#ef4444"

    cat_rows = ""
    for cat, sc in sorted(report.category_scores.items()):
        c = "#22c55e" if sc >= 80 else "#eab308" if sc >= 50 else "#ef4444"
        cat_rows += f'<tr><td>{cat}</td><td style="color:{c};font-weight:bold">{sc:.0f}%</td><td><div style="background:#e5e7eb;border-radius:4px;height:18px;width:200px"><div style="background:{c};height:100%;width:{sc}%;border-radius:4px"></div></div></td></tr>\n'

    rec_cards = ""
    impact_colors = {"critical": "#ef4444", "high": "#f97316", "medium": "#eab308", "low": "#22c55e"}
    for rec in report.recommendations:
        ic = impact_colors.get(rec.impact.value, "#6b7280")
        refs = f'<p style="color:#6b7280;font-size:0.85em">Refs: {", ".join(rec.references)}</p>' if rec.references else ""
        rec_cards += f'''<div style="border:1px solid #e5e7eb;border-left:4px solid {ic};border-radius:8px;padding:16px;margin:12px 0">
  <h3 style="margin:0 0 8px">[{rec.id}] {rec.title}</h3>
  <span style="background:{ic};color:white;padding:2px 8px;border-radius:4px;font-size:0.8em">{rec.impact.value.upper()}</span>
  <span style="background:#e5e7eb;padding:2px 8px;border-radius:4px;font-size:0.8em;margin-left:4px">{rec.effort.value} effort</span>
  <span style="background:#e5e7eb;padding:2px 8px;border-radius:4px;font-size:0.8em;margin-left:4px">{rec.category.value}</span>
  <p style="margin:12px 0 4px">{rec.description}</p>
  <table style="font-size:0.9em;margin:8px 0"><tr><td style="color:#6b7280;padding-right:12px">Current:</td><td>{rec.current_state}</td></tr><tr><td style="color:#6b7280;padding-right:12px">Target:</td><td style="color:#16a34a;font-weight:500">{rec.recommended_state}</td></tr></table>
  <p style="font-style:italic;color:#4b5563;font-size:0.9em">{rec.rationale}</p>
  {refs}
</div>\n'''

    return f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Hardening Report</title>
<style>body{{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#1f2937}}
h1{{border-bottom:2px solid #e5e7eb;padding-bottom:.5rem}}table{{border-collapse:collapse}}td{{padding:4px 8px}}</style></head>
<body>
<h1>🛡️ Hardening Advisor Report</h1>
<div style="text-align:center;margin:2rem 0">
  <div style="font-size:3rem;font-weight:bold;color:{score_color}">{score:.0f}/100</div>
  <div style="color:#6b7280">{report.summary}</div>
</div>
<h2>Category Scores</h2>
<table>{cat_rows}</table>
<h2>Recommendations ({len(report.recommendations)})</h2>
{rec_cards}
</body></html>'''


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m replication harden",
        description="Analyze safety configuration and recommend hardening improvements",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json", "html"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--category", "-c",
        choices=[c.value for c in Category],
        help="Filter by category",
    )
    parser.add_argument(
        "--min-impact", "-m",
        choices=[i.value for i in Impact],
        help="Minimum impact level to show",
    )
    parser.add_argument(
        "--output", "-o",
        help="Write output to file instead of stdout",
    )
    parser.add_argument(
        "--config",
        help="Path to JSON config file with addressed_checks list",
    )

    args = parser.parse_args(argv)

    config = None
    if args.config:
        import pathlib
        config = json.loads(pathlib.Path(args.config).read_text())

    report = _assess_config(config)

    if args.format == "json":
        output = json.dumps(report.to_dict(), indent=2)
    elif args.format == "html":
        output = _render_html(report)
    else:
        output = _render_text(report, category=args.category, min_impact=args.min_impact)

    from ._helpers import emit_output
    emit_output(output, args.output, "Report")


if __name__ == "__main__":
    main()
