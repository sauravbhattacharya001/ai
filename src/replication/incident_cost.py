"""Incident Cost Estimator — estimate financial & operational impact of AI safety incidents.

Calculates estimated costs across multiple dimensions (response labour,
downtime, reputation, regulatory fines, remediation) based on incident
severity, blast radius, detection delay, and containment time.

Usage (CLI)::

    python -m replication incident-cost
    python -m replication incident-cost --severity P1
    python -m replication incident-cost --severity P2 --blast-radius 12 --detection-hours 4
    python -m replication incident-cost --severity P0 --regulated --json
    python -m replication incident-cost --compare P0,P1,P2,P3,P4

Programmatic::

    from replication.incident_cost import IncidentCostEstimator, IncidentParams
    estimator = IncidentCostEstimator()
    params = IncidentParams(severity="P1", blast_radius=5)
    report = estimator.estimate(params)
    print(report.render())
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ── Severity levels ──────────────────────────────────────────────────

class Severity(Enum):
    P0 = "P0"  # Critical — full system compromise / data breach
    P1 = "P1"  # High — significant safety control failure
    P2 = "P2"  # Medium — partial control bypass
    P3 = "P3"  # Low — minor policy violation
    P4 = "P4"  # Informational — near-miss or anomaly

    @property
    def multiplier(self) -> float:
        return {
            Severity.P0: 10.0,
            Severity.P1: 5.0,
            Severity.P2: 2.5,
            Severity.P3: 1.0,
            Severity.P4: 0.3,
        }[self]

    @property
    def label(self) -> str:
        labels = {
            Severity.P0: "Critical",
            Severity.P1: "High",
            Severity.P2: "Medium",
            Severity.P3: "Low",
            Severity.P4: "Informational",
        }
        return labels[self]


# ── Cost categories ──────────────────────────────────────────────────

@dataclass
class CostBreakdown:
    """Individual cost line item."""
    category: str
    description: str
    amount: float
    confidence: str  # "high", "medium", "low"
    notes: str = ""


@dataclass
class IncidentParams:
    """Parameters describing an incident for cost estimation."""
    severity: str = "P2"
    blast_radius: int = 3          # number of affected agents/services
    detection_hours: float = 1.0   # hours until detection
    containment_hours: float = 2.0 # hours until containment
    recovery_hours: float = 8.0    # hours until full recovery
    affected_users: int = 100      # approximate user impact
    data_exposed: bool = False     # was sensitive data exposed?
    regulated: bool = False        # subject to regulatory framework?
    repeat_incident: bool = False  # has this type occurred before?
    team_size: int = 4             # responders involved

    BASE_HOURLY_RATE: float = 150.0  # $/hr for incident responders


@dataclass
class CostReport:
    """Full cost estimation report."""
    params: IncidentParams
    severity: Severity
    line_items: List[CostBreakdown] = field(default_factory=list)
    total_low: float = 0.0
    total_expected: float = 0.0
    total_high: float = 0.0
    risk_factors: List[str] = field(default_factory=list)
    mitigation_savings: List[str] = field(default_factory=list)

    @property
    def total(self) -> float:
        return self.total_expected

    def render(self) -> str:
        """Render human-readable cost report."""
        lines: List[str] = []
        lines.append("=" * 70)
        lines.append("  INCIDENT COST ESTIMATE")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"  Severity:          {self.severity.value} ({self.severity.label})")
        lines.append(f"  Blast Radius:      {self.params.blast_radius} agents/services")
        lines.append(f"  Detection Time:    {self.params.detection_hours:.1f} hours")
        lines.append(f"  Containment Time:  {self.params.containment_hours:.1f} hours")
        lines.append(f"  Recovery Time:     {self.params.recovery_hours:.1f} hours")
        lines.append(f"  Affected Users:    {self.params.affected_users:,}")
        lines.append(f"  Data Exposed:      {'Yes' if self.params.data_exposed else 'No'}")
        lines.append(f"  Regulated:         {'Yes' if self.params.regulated else 'No'}")
        lines.append(f"  Repeat Incident:   {'Yes' if self.params.repeat_incident else 'No'}")
        lines.append(f"  Team Size:         {self.params.team_size}")
        lines.append("")
        lines.append("-" * 70)
        lines.append("  COST BREAKDOWN")
        lines.append("-" * 70)
        lines.append("")
        lines.append(f"  {'Category':<25} {'Amount':>12}  {'Confidence':<10} Notes")
        lines.append(f"  {'─' * 25} {'─' * 12}  {'─' * 10} {'─' * 15}")

        for item in self.line_items:
            notes = item.notes[:30] if item.notes else ""
            lines.append(
                f"  {item.category:<25} ${item.amount:>11,.0f}  {item.confidence:<10} {notes}"
            )

        lines.append("")
        lines.append("-" * 70)
        lines.append(f"  Estimated Range:   ${self.total_low:>10,.0f}  –  ${self.total_high:>10,.0f}")
        lines.append(f"  Expected Cost:     ${self.total_expected:>10,.0f}")
        lines.append("-" * 70)

        if self.risk_factors:
            lines.append("")
            lines.append("  ⚠ RISK FACTORS:")
            for rf in self.risk_factors:
                lines.append(f"    • {rf}")

        if self.mitigation_savings:
            lines.append("")
            lines.append("  💡 POTENTIAL SAVINGS:")
            for ms in self.mitigation_savings:
                lines.append(f"    • {ms}")

        lines.append("")
        lines.append("=" * 70)
        lines.append("  Note: Estimates are indicative. Actual costs depend on context.")
        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "severity_label": self.severity.label,
            "params": {
                "blast_radius": self.params.blast_radius,
                "detection_hours": self.params.detection_hours,
                "containment_hours": self.params.containment_hours,
                "recovery_hours": self.params.recovery_hours,
                "affected_users": self.params.affected_users,
                "data_exposed": self.params.data_exposed,
                "regulated": self.params.regulated,
                "repeat_incident": self.params.repeat_incident,
                "team_size": self.params.team_size,
            },
            "line_items": [
                {
                    "category": li.category,
                    "description": li.description,
                    "amount": li.amount,
                    "confidence": li.confidence,
                    "notes": li.notes,
                }
                for li in self.line_items
            ],
            "total_low": self.total_low,
            "total_expected": self.total_expected,
            "total_high": self.total_high,
            "risk_factors": self.risk_factors,
            "mitigation_savings": self.mitigation_savings,
        }


# ── Estimator ────────────────────────────────────────────────────────

class IncidentCostEstimator:
    """Estimate the cost of an AI safety incident."""

    def estimate(self, params: IncidentParams) -> CostReport:
        severity = Severity(params.severity.upper())
        report = CostReport(params=params, severity=severity)
        mult = severity.multiplier

        # 1. Response Labour
        total_response_hours = (
            params.detection_hours
            + params.containment_hours
            + params.recovery_hours
        )
        labour_cost = total_response_hours * params.team_size * params.BASE_HOURLY_RATE * mult
        report.line_items.append(CostBreakdown(
            category="Response Labour",
            description="Personnel time for detection, containment, and recovery",
            amount=labour_cost,
            confidence="high",
            notes=f"{total_response_hours:.0f}h × {params.team_size} people",
        ))

        # 2. Downtime / Service Disruption
        downtime_per_agent = 500.0 * mult  # base cost per affected agent per hour
        downtime_cost = (
            params.blast_radius
            * (params.containment_hours + params.recovery_hours)
            * downtime_per_agent
        )
        report.line_items.append(CostBreakdown(
            category="Service Disruption",
            description="Lost productivity and service unavailability",
            amount=downtime_cost,
            confidence="medium",
            notes=f"{params.blast_radius} agents affected",
        ))

        # 3. User Impact
        user_cost_per_user = 5.0 * mult
        user_impact_cost = params.affected_users * user_cost_per_user
        report.line_items.append(CostBreakdown(
            category="User Impact",
            description="User-facing disruption, support tickets, churn risk",
            amount=user_impact_cost,
            confidence="medium",
            notes=f"{params.affected_users:,} users",
        ))

        # 4. Data Exposure
        data_cost = 0.0
        if params.data_exposed:
            data_cost = 25000.0 * mult + params.affected_users * 50.0
            report.line_items.append(CostBreakdown(
                category="Data Exposure",
                description="Breach notification, credit monitoring, legal review",
                amount=data_cost,
                confidence="low",
                notes="Per-user notification costs",
            ))

        # 5. Regulatory Fines
        regulatory_cost = 0.0
        if params.regulated:
            regulatory_cost = 50000.0 * mult
            if params.data_exposed:
                regulatory_cost *= 2.0
            if params.repeat_incident:
                regulatory_cost *= 1.5
            report.line_items.append(CostBreakdown(
                category="Regulatory Fines",
                description="Potential regulatory penalties and compliance costs",
                amount=regulatory_cost,
                confidence="low",
                notes="Varies by jurisdiction",
            ))

        # 6. Reputation / Trust
        reputation_cost = 10000.0 * mult * (1 + params.blast_radius * 0.1)
        if params.data_exposed:
            reputation_cost *= 2.0
        report.line_items.append(CostBreakdown(
            category="Reputation Damage",
            description="Brand impact, customer trust erosion",
            amount=reputation_cost,
            confidence="low",
            notes="Long-term impact",
        ))

        # 7. Remediation
        remediation_cost = 5000.0 * mult * params.blast_radius
        report.line_items.append(CostBreakdown(
            category="Remediation",
            description="Fixes, patches, policy updates, re-testing",
            amount=remediation_cost,
            confidence="medium",
            notes=f"{params.blast_radius} systems to fix",
        ))

        # 8. Post-incident (postmortem, training, process changes)
        post_incident_cost = 3000.0 * mult
        report.line_items.append(CostBreakdown(
            category="Post-Incident",
            description="Postmortem, training, process improvements",
            amount=post_incident_cost,
            confidence="high",
            notes="Ongoing investment",
        ))

        # Totals
        total = sum(li.amount for li in report.line_items)
        report.total_expected = total
        report.total_low = total * 0.6
        report.total_high = total * 1.8

        # Risk factors
        if params.detection_hours > 4:
            report.risk_factors.append(
                f"Slow detection ({params.detection_hours:.0f}h) — costs escalate with delay"
            )
        if params.repeat_incident:
            report.risk_factors.append(
                "Repeat incident — indicates systemic issues; regulatory scrutiny likely"
            )
        if params.blast_radius > 10:
            report.risk_factors.append(
                f"Wide blast radius ({params.blast_radius}) — cascade effects may amplify costs"
            )
        if params.data_exposed and params.regulated:
            report.risk_factors.append(
                "Regulated data breach — mandatory reporting & potential class-action risk"
            )

        # Savings suggestions
        if params.detection_hours > 2:
            saved = labour_cost * 0.3
            report.mitigation_savings.append(
                f"Faster detection (< 2h) could save ~${saved:,.0f} in response labour"
            )
        if params.blast_radius > 5:
            report.mitigation_savings.append(
                "Better isolation/segmentation would reduce blast radius and downtime costs"
            )
        if not params.regulated and params.data_exposed:
            report.mitigation_savings.append(
                "Proactive data minimization reduces exposure costs even without regulation"
            )

        return report

    def compare(self, severities: List[str], base_params: Optional[IncidentParams] = None) -> str:
        """Compare costs across multiple severity levels."""
        params = base_params or IncidentParams()
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("  SEVERITY COST COMPARISON")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"  {'Severity':<15} {'Label':<15} {'Expected Cost':>15}")
        lines.append(f"  {'─' * 15} {'─' * 15} {'─' * 15}")

        for sev_str in severities:
            p = IncidentParams(
                severity=sev_str,
                blast_radius=params.blast_radius,
                detection_hours=params.detection_hours,
                containment_hours=params.containment_hours,
                recovery_hours=params.recovery_hours,
                affected_users=params.affected_users,
                data_exposed=params.data_exposed,
                regulated=params.regulated,
                repeat_incident=params.repeat_incident,
                team_size=params.team_size,
            )
            report = self.estimate(p)
            lines.append(
                f"  {sev_str:<15} {report.severity.label:<15} ${report.total_expected:>14,.0f}"
            )

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m replication incident-cost",
        description="Estimate financial & operational cost of AI safety incidents",
    )
    parser.add_argument("--severity", default="P2", choices=["P0", "P1", "P2", "P3", "P4"],
                        help="Incident severity (default: P2)")
    parser.add_argument("--blast-radius", type=int, default=3,
                        help="Number of affected agents/services (default: 3)")
    parser.add_argument("--detection-hours", type=float, default=1.0,
                        help="Hours until detection (default: 1.0)")
    parser.add_argument("--containment-hours", type=float, default=2.0,
                        help="Hours until containment (default: 2.0)")
    parser.add_argument("--recovery-hours", type=float, default=8.0,
                        help="Hours until full recovery (default: 8.0)")
    parser.add_argument("--affected-users", type=int, default=100,
                        help="Approximate user impact (default: 100)")
    parser.add_argument("--data-exposed", action="store_true",
                        help="Sensitive data was exposed")
    parser.add_argument("--regulated", action="store_true",
                        help="Subject to regulatory framework")
    parser.add_argument("--repeat", action="store_true",
                        help="This type of incident has occurred before")
    parser.add_argument("--team-size", type=int, default=4,
                        help="Number of responders (default: 4)")
    parser.add_argument("--compare", type=str, default=None,
                        help="Compare costs across severities (comma-separated, e.g. P0,P1,P2)")
    parser.add_argument("--json", action="store_true", dest="json_output",
                        help="Output as JSON")

    args = parser.parse_args(argv)
    estimator = IncidentCostEstimator()

    if args.compare:
        severities = [s.strip().upper() for s in args.compare.split(",")]
        params = IncidentParams(
            blast_radius=args.blast_radius,
            detection_hours=args.detection_hours,
            containment_hours=args.containment_hours,
            recovery_hours=args.recovery_hours,
            affected_users=args.affected_users,
            data_exposed=args.data_exposed,
            regulated=args.regulated,
            repeat_incident=args.repeat,
            team_size=args.team_size,
        )
        print(estimator.compare(severities, params))
        return

    params = IncidentParams(
        severity=args.severity,
        blast_radius=args.blast_radius,
        detection_hours=args.detection_hours,
        containment_hours=args.containment_hours,
        recovery_hours=args.recovery_hours,
        affected_users=args.affected_users,
        data_exposed=args.data_exposed,
        regulated=args.regulated,
        repeat_incident=args.repeat,
        team_size=args.team_size,
    )

    report = estimator.estimate(params)

    if args.json_output:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.render())


if __name__ == "__main__":
    main()
