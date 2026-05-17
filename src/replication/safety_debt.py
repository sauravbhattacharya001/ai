"""Safety Debt Advisor - agentic accumulated-debt tracker.

The 4th sibling in the remediation suite:

* :mod:`remediation_planner`      - **what** to fix
* :mod:`remediation_progress`     - **are we** fixing it (velocity)
* :mod:`remediation_assignment`   - **who** owns each fix
* :mod:`safety_debt`              - **how much debt** has piled up, and
  can we still service it?

Models safety findings as accruing financial debt: every open finding has
a severity-weighted *principal* + compound *interest* the longer it sits
unfixed.  The team has a finite weekly *debt-service capacity* (velocity).
The advisor classifies portfolio health from ``SOLVENT`` (paying down
faster than interest accrues) to ``BANKRUPT`` (interest exceeds capacity
- a debt spiral), and emits a deduped P0-first playbook to either pay
down defaults, increase velocity, refinance overdue items, or prevent
new debt at the source.

CLI usage::

    python -m replication debt --demo
    python -m replication debt --demo --format md
    python -m replication debt --demo --format json
    python -m replication debt --demo --risk cautious --velocity 15

Programmatic::

    from replication.remediation_planner import Finding
    from replication.safety_debt import SafetyDebtAdvisor, DebtSnapshot

    findings = [
        Finding(name="kill-switch", source="scorecard", status="fail",
                summary="kill switch race condition"),
        Finding(name="policy-lint", source="quick_scan", status="warn",
                summary="3 overly broad rules"),
    ]
    advisor = SafetyDebtAdvisor()
    report = advisor.assess(findings, team_velocity_points_per_week=20.0)
    print(report.render())
    print(report.to_markdown())
    print(report.to_json())
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

from .remediation_planner import Finding
from .remediation_progress import SLA_DAYS_BY_SEVERITY


# ── Constants ────────────────────────────────────────────────────────


# Base "debt points" by severity - principal owed when the finding opens.
SEVERITY_PRINCIPAL: Dict[str, float] = {
    "critical": 20.0,
    "high": 10.0,
    "medium": 5.0,
    "low": 2.0,
    "info": 1.0,
}


# Weekly compound interest rate by severity.  Critical findings rot fast;
# info-level slow-burn items barely accrue.
INTEREST_RATE_PER_WEEK: Dict[str, float] = {
    "critical": 0.30,
    "high": 0.15,
    "medium": 0.07,
    "low": 0.03,
    "info": 0.01,
}


RISK_APPETITE_CAPACITY_MULT: Dict[str, float] = {
    "cautious": 0.85,   # reserve capacity for prevention work
    "balanced": 1.0,
    "aggressive": 1.15,
}


# Per-severity ladder thresholds (× SLA days) for the verdict ladder.
# NEW           : age == 0  (we have not seen this before)
# CURRENT       : 0 < age < 1 * sla
# AGING         : 1 * sla ≤ age < 2 * sla
# OVERDUE       : 2 * sla ≤ age < 4 * sla
# DEFAULTED     : age ≥ 4 * sla


# ── Data model ───────────────────────────────────────────────────────


def _finding_signature(f: Finding) -> str:
    return f"{f.source}:{f.name}"


@dataclass
class DebtSnapshot:
    """A point-in-time record of every open finding's first-seen date.

    ``finding_signatures`` maps ``"source:name"`` → ISO-8601 first-seen
    timestamp.  The advisor uses this to compute per-item ``age_days``
    across runs.
    """

    timestamp: str
    finding_signatures: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "finding_signatures": dict(self.finding_signatures),
        }

    @classmethod
    def from_findings(
        cls,
        findings: Sequence[Finding],
        now: Optional[datetime] = None,
    ) -> "DebtSnapshot":
        ts = (now or datetime.now(timezone.utc)).isoformat()
        sigs: Dict[str, str] = {}
        for f in findings:
            sigs[_finding_signature(f)] = ts
        return cls(timestamp=ts, finding_signatures=sigs)


@dataclass
class DebtItem:
    """Per-finding debt entry."""

    id: str
    title: str
    source: str
    severity: str
    principal: float
    interest_rate_per_week: float
    age_days: float
    accrued_interest: float
    total_debt_points: float
    verdict: str           # NEW | CURRENT | AGING | OVERDUE | DEFAULTED
    priority: str          # P0 | P1 | P2 | P3
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "source": self.source,
            "severity": self.severity,
            "principal": round(self.principal, 4),
            "interest_rate_per_week": self.interest_rate_per_week,
            "age_days": round(self.age_days, 3),
            "accrued_interest": round(self.accrued_interest, 4),
            "total_debt_points": round(self.total_debt_points, 4),
            "verdict": self.verdict,
            "priority": self.priority,
            "reasons": list(self.reasons),
        }


@dataclass
class PlaybookAction:
    id: str
    priority: str
    label: str
    owner: str
    reason: str
    blast_radius: int           # 1-5
    reversibility: str          # low | medium | high
    item_ids: List[str] = field(default_factory=list)
    suggested_value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "priority": self.priority,
            "label": self.label,
            "owner": self.owner,
            "reason": self.reason,
            "blast_radius": self.blast_radius,
            "reversibility": self.reversibility,
            "item_ids": list(self.item_ids),
            "suggested_value": self.suggested_value,
        }


@dataclass
class SafetyDebtReport:
    """Full debt report - items, portfolio metrics, playbook, renderers."""

    timestamp: str
    risk_appetite: str
    items: List[DebtItem] = field(default_factory=list)

    total_principal: float = 0.0
    total_accrued_interest: float = 0.0
    total_debt_points: float = 0.0

    weekly_interest_burn: float = 0.0
    debt_service_capacity: float = 0.0
    coverage_ratio: float = 0.0
    weeks_to_zero: Optional[float] = None

    portfolio_health: str = "SOLVENT"
    grade: str = "A"
    trajectory: str = "stable"
    headline: str = ""

    insights: List[str] = field(default_factory=list)
    playbook: List[PlaybookAction] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    # ── exporters ────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "risk_appetite": self.risk_appetite,
            "portfolio_health": self.portfolio_health,
            "grade": self.grade,
            "trajectory": self.trajectory,
            "headline": self.headline,
            "totals": {
                "principal": round(self.total_principal, 4),
                "accrued_interest": round(self.total_accrued_interest, 4),
                "debt_points": round(self.total_debt_points, 4),
            },
            "service": {
                "weekly_interest_burn": round(self.weekly_interest_burn, 4),
                "debt_service_capacity": round(self.debt_service_capacity, 4),
                "coverage_ratio": round(self.coverage_ratio, 4),
                "weeks_to_zero": (
                    None
                    if self.weeks_to_zero is None
                    else round(self.weeks_to_zero, 3)
                ),
            },
            "items": [it.to_dict() for it in self.items],
            "playbook": [a.to_dict() for a in self.playbook],
            "insights": list(self.insights),
            "notes": list(self.notes),
        }

    def to_json(self, indent: int = 2) -> str:
        # Deterministic byte-stable output (sort_keys=True, default=str).
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            default=str,
        )

    def to_markdown(self) -> str:
        lines: List[str] = []
        lines.append("# Safety Debt Report")
        lines.append("")
        lines.append(f"_Generated: {self.timestamp}_")
        lines.append("")
        lines.append(f"**{self.headline}**")
        lines.append("")
        lines.append("## Portfolio")
        lines.append("")
        lines.append(f"- Health: **{self.portfolio_health}** (Grade {self.grade})")
        lines.append(f"- Trajectory: {self.trajectory}")
        lines.append(f"- Risk appetite: {self.risk_appetite}")
        lines.append(
            f"- Total debt: {self.total_debt_points:.2f} pts "
            f"(principal {self.total_principal:.2f}, "
            f"accrued interest {self.total_accrued_interest:.2f})"
        )
        lines.append(
            f"- Weekly interest burn: {self.weekly_interest_burn:.2f} pts/wk"
        )
        lines.append(
            f"- Debt-service capacity: {self.debt_service_capacity:.2f} pts/wk"
        )
        lines.append(
            f"- Coverage ratio: {self.coverage_ratio:.2f}x "
            "(capacity ÷ interest burn)"
        )
        if self.weeks_to_zero is None:
            lines.append("- Weeks to zero: **∞** (debt spiral)")
        else:
            lines.append(f"- Weeks to zero: {self.weeks_to_zero:.1f}")
        lines.append("")

        if self.insights:
            lines.append("## Insights")
            lines.append("")
            for s in self.insights:
                lines.append(f"- {s}")
            lines.append("")

        if self.items:
            lines.append("## Debt items")
            lines.append("")
            lines.append(
                "| ID | Severity | Verdict | Age (d) | Principal | "
                "Interest | Total | Priority |"
            )
            lines.append(
                "|----|----------|---------|---------|-----------|---------"
                "|-------|----------|"
            )
            for it in self.items:
                lines.append(
                    f"| {it.id} | {it.severity} | {it.verdict} | "
                    f"{it.age_days:.1f} | {it.principal:.2f} | "
                    f"{it.accrued_interest:.2f} | "
                    f"{it.total_debt_points:.2f} | {it.priority} |"
                )
            lines.append("")

        if self.playbook:
            lines.append("## Playbook")
            lines.append("")
            for a in self.playbook:
                extra = ""
                if a.suggested_value is not None:
                    extra = f" (suggested: {a.suggested_value})"
                lines.append(
                    f"- **[{a.priority}] {a.label}**{extra} — "
                    f"owner: {a.owner}, blast {a.blast_radius}, "
                    f"reversibility {a.reversibility}. {a.reason}"
                )
            lines.append("")

        if self.notes:
            lines.append("## Notes")
            lines.append("")
            for n in self.notes:
                lines.append(f"- {n}")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def render(self) -> str:
        buf = io.StringIO()
        buf.write("Safety Debt Report\n")
        buf.write("=" * 60 + "\n")
        buf.write(f"Generated:        {self.timestamp}\n")
        buf.write(f"Health:           {self.portfolio_health}  (grade {self.grade})\n")
        buf.write(f"Trajectory:       {self.trajectory}\n")
        buf.write(f"Risk appetite:    {self.risk_appetite}\n")
        buf.write(f"Total debt:       {self.total_debt_points:.2f} pts\n")
        buf.write(
            f"  principal:      {self.total_principal:.2f}\n"
            f"  accrued int.:   {self.total_accrued_interest:.2f}\n"
        )
        buf.write(f"Weekly burn:      {self.weekly_interest_burn:.2f} pts/wk\n")
        buf.write(f"Service capacity: {self.debt_service_capacity:.2f} pts/wk\n")
        buf.write(f"Coverage ratio:   {self.coverage_ratio:.2f}x\n")
        if self.weeks_to_zero is None:
            buf.write("Weeks to zero:    inf  (debt spiral)\n")
        else:
            buf.write(f"Weeks to zero:    {self.weeks_to_zero:.1f}\n")
        buf.write("\n")
        if self.headline:
            buf.write(self.headline + "\n\n")

        if self.insights:
            buf.write("Insights:\n")
            for s in self.insights:
                buf.write(f"  - {s}\n")
            buf.write("\n")

        if self.items:
            buf.write(
                f"{'ID':28}{'SEV':10}{'VERDICT':12}"
                f"{'AGE':>7}{'PRIN':>8}{'INT':>8}{'TOT':>8}  PRIO\n"
            )
            for it in self.items:
                buf.write(
                    f"{it.id[:28]:28}{it.severity:10}{it.verdict:12}"
                    f"{it.age_days:7.1f}{it.principal:8.2f}"
                    f"{it.accrued_interest:8.2f}{it.total_debt_points:8.2f}"
                    f"  {it.priority}\n"
                )
            buf.write("\n")

        if self.playbook:
            buf.write("Playbook:\n")
            for a in self.playbook:
                extra = ""
                if a.suggested_value is not None:
                    extra = f"  [suggested: {a.suggested_value}]"
                buf.write(
                    f"  [{a.priority}] {a.label}{extra}\n"
                    f"        owner={a.owner} blast={a.blast_radius} "
                    f"rev={a.reversibility}\n"
                    f"        why: {a.reason}\n"
                )
            buf.write("\n")

        if self.notes:
            buf.write("Notes:\n")
            for n in self.notes:
                buf.write(f"  - {n}\n")
            buf.write("\n")

        return buf.getvalue()


# ── Advisor ──────────────────────────────────────────────────────────


class SafetyDebtAdvisor:
    """Compute portfolio safety debt + agentic playbook."""

    def __init__(self, sla_days: Optional[Dict[str, float]] = None) -> None:
        self.sla_days = dict(sla_days) if sla_days else dict(SLA_DAYS_BY_SEVERITY)

    # ── public API ───────────────────────────────────────────────────

    def assess(
        self,
        findings: Sequence[Finding],
        history: Optional[Sequence[DebtSnapshot]] = None,
        risk_appetite: str = "balanced",
        team_velocity_points_per_week: float = 20.0,
        now: Optional[Callable[[], datetime] | datetime] = None,
    ) -> SafetyDebtReport:
        risk_appetite = risk_appetite.lower().strip()
        if risk_appetite not in RISK_APPETITE_CAPACITY_MULT:
            risk_appetite = "balanced"

        now_dt = self._resolve_now(now)
        ts_iso = now_dt.isoformat()

        history = list(history or [])
        # Find each finding's first-seen timestamp by signature.
        first_seen: Dict[str, datetime] = {}
        for snap in history:
            for sig, ts_str in snap.finding_signatures.items():
                try:
                    parsed = datetime.fromisoformat(ts_str)
                except ValueError:
                    continue
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                if sig not in first_seen or parsed < first_seen[sig]:
                    first_seen[sig] = parsed

        items: List[DebtItem] = []
        new_count = 0
        defaulted_critical = 0
        defaulted_high_or_critical = 0
        overdue_count = 0
        aging_low_info = 0
        severity_buckets: Dict[str, float] = {
            "critical": 0.0,
            "high": 0.0,
            "medium": 0.0,
            "low": 0.0,
            "info": 0.0,
        }

        for idx, f in enumerate(findings):
            sev = f.severity if f.severity in SEVERITY_PRINCIPAL else "info"
            sig = _finding_signature(f)
            first = first_seen.get(sig)
            if first is None:
                age_days = 0.0
                is_new = True
            else:
                age_days = max((now_dt - first).total_seconds() / 86400.0, 0.0)
                is_new = age_days == 0.0

            principal = SEVERITY_PRINCIPAL[sev]
            rate = INTEREST_RATE_PER_WEEK[sev]
            growth = (1.0 + rate) ** (age_days / 7.0)
            accrued = principal * growth - principal
            total = principal + accrued

            sla = self.sla_days.get(sev, 14.0)
            if is_new:
                verdict = "NEW"
            elif age_days < sla:
                verdict = "CURRENT"
            elif age_days < 2 * sla:
                verdict = "AGING"
            elif age_days < 4 * sla:
                verdict = "OVERDUE"
            else:
                verdict = "DEFAULTED"

            reasons: List[str] = []
            if verdict == "NEW":
                reasons.append("FRESH_DEBT")
            if accrued >= principal and principal > 0:
                reasons.append("COMPOUND_INTEREST_HEAVY")
            if verdict == "DEFAULTED" and sev in ("critical", "high"):
                reasons.append("CRITICAL_DEFAULT")
            if verdict == "AGING" and sev in ("low", "info"):
                reasons.append("SLOW_BURN")

            # Priority ladder
            if verdict == "DEFAULTED" and sev in ("critical", "high"):
                priority = "P0"
            elif (verdict == "OVERDUE" and sev in ("critical", "high")) or (
                verdict == "DEFAULTED" and sev == "medium"
            ):
                priority = "P1"
            elif (verdict == "AGING" and sev in ("critical", "high")) or (
                verdict == "OVERDUE" and sev == "medium"
            ):
                priority = "P2"
            else:
                priority = "P3"

            item = DebtItem(
                id=f"debt-{idx + 1:03d}",
                title=f.summary or f.name,
                source=f.source,
                severity=sev,
                principal=principal,
                interest_rate_per_week=rate,
                age_days=age_days,
                accrued_interest=accrued,
                total_debt_points=total,
                verdict=verdict,
                priority=priority,
                reasons=reasons,
            )
            items.append(item)

            if is_new:
                new_count += 1
            if verdict == "DEFAULTED":
                if sev == "critical":
                    defaulted_critical += 1
                if sev in ("critical", "high"):
                    defaulted_high_or_critical += 1
            if verdict == "OVERDUE":
                overdue_count += 1
            if verdict == "AGING" and sev in ("low", "info"):
                aging_low_info += 1
            severity_buckets[sev] += total

        total_principal = sum(it.principal for it in items)
        total_accrued = sum(it.accrued_interest for it in items)
        total_debt = total_principal + total_accrued

        weekly_burn = sum(
            it.principal * it.interest_rate_per_week for it in items
        )
        capacity = team_velocity_points_per_week * RISK_APPETITE_CAPACITY_MULT[
            risk_appetite
        ]

        coverage_ratio = capacity / max(weekly_burn, 0.01)
        defaulted_count = sum(1 for it in items if it.verdict == "DEFAULTED")

        if capacity > weekly_burn:
            weeks_to_zero = total_debt / max(capacity - weekly_burn, 0.01)
        else:
            weeks_to_zero = None

        # Portfolio health
        if coverage_ratio < 0.5 or defaulted_critical >= 2:
            health = "BANKRUPT"
            grade = "F"
        elif coverage_ratio < 1.0:
            health = "AT_RISK"
            grade = "D"
        elif coverage_ratio < 1.5:
            health = "STRAINED"
            grade = "C"
        elif coverage_ratio < 3.0 or defaulted_count > 1:
            health = "MANAGEABLE"
            grade = "B"
        else:
            health = "SOLVENT"
            grade = "A"

        # Trajectory from prior snapshots (compare totals if we have >=1)
        trajectory = "stable"
        if history:
            prior_totals: List[float] = []
            for snap in history[-2:]:
                # crude: prior "total" estimated as principal-only sum over
                # signatures still open
                count = sum(
                    1
                    for sig in snap.finding_signatures
                    if any(_finding_signature(f) == sig for f in findings)
                )
                prior_totals.append(count * 5.0)  # rough avg principal
            if prior_totals:
                last = prior_totals[-1]
                if total_debt < last * 0.95:
                    trajectory = "improving"
                elif total_debt > last * 1.05:
                    trajectory = "deteriorating"

        # ── Insights ──
        insights: List[str] = []
        if total_accrued > total_principal and total_principal > 0:
            insights.append(
                "compound_interest_dominant: accrued interest exceeds "
                "outstanding principal — debt is rotting in place."
            )
        if weeks_to_zero is None:
            insights.append(
                "debt_spiral_warning: weekly interest burn exceeds "
                "service capacity — debt grows even without new findings."
            )
        if total_debt > 0:
            dom_sev, dom_val = max(
                severity_buckets.items(), key=lambda kv: kv[1]
            )
            if dom_val / total_debt >= 0.60:
                insights.append(
                    f"severity_concentration: {dom_sev} severity accounts "
                    f"for {dom_val / total_debt * 100:.0f}% of total debt."
                )
        if history:
            # recent_default_cluster: prior snapshot signatures whose age
            # would have been OVERDUE-band, now DEFAULTED in current.
            recent_defaulted_now = sum(
                1
                for it in items
                if it.verdict == "DEFAULTED"
                and any(
                    _finding_signature_eq(it, sig)
                    for snap in history[-2:]
                    for sig in snap.finding_signatures
                )
            )
            if recent_defaulted_now >= 2:
                insights.append(
                    f"recent_default_cluster: {recent_defaulted_now} items "
                    "defaulted since the prior snapshot."
                )

        # ── Playbook (P0 first, deduped) ──
        playbook: List[PlaybookAction] = []
        seen_ids: set[str] = set()

        def _add(a: PlaybookAction) -> None:
            if a.id in seen_ids:
                return
            seen_ids.add(a.id)
            playbook.append(a)

        if health == "BANKRUPT":
            _add(
                PlaybookAction(
                    id="EMERGENCY_DEBT_SUMMIT",
                    priority="P0",
                    label="Convene emergency debt summit",
                    owner="safety-lead",
                    reason=(
                        "Portfolio is BANKRUPT — coverage ratio "
                        f"{coverage_ratio:.2f}x. Halt new feature work, "
                        "freeze risky deploys, attack defaults."
                    ),
                    blast_radius=5,
                    reversibility="medium",
                )
            )

        defaulted_items = [it for it in items if it.verdict == "DEFAULTED"]
        defaulted_items.sort(key=lambda x: -x.total_debt_points)
        if defaulted_items:
            top = defaulted_items[:3]
            _add(
                PlaybookAction(
                    id="PAY_DOWN_DEFAULTS",
                    priority="P0",
                    label="Pay down top defaulted findings",
                    owner="safety-team",
                    reason=(
                        f"{len(defaulted_items)} item(s) past 4× SLA; "
                        f"top three carry {sum(x.total_debt_points for x in top):.1f} "
                        "debt points."
                    ),
                    blast_radius=4,
                    reversibility="high",
                    item_ids=[x.id for x in top],
                )
            )

        if coverage_ratio < 1.5:
            _add(
                PlaybookAction(
                    id="INCREASE_VELOCITY",
                    priority="P1",
                    label="Increase remediation velocity",
                    owner="eng-manager",
                    reason=(
                        f"Coverage {coverage_ratio:.2f}x — capacity barely "
                        "outpaces interest burn."
                    ),
                    blast_radius=3,
                    reversibility="high",
                    suggested_value=float(math.ceil(weekly_burn * 2.0)),
                )
            )

        if overdue_count >= 3:
            _add(
                PlaybookAction(
                    id="REFINANCE_OVERDUE",
                    priority="P1",
                    label="Refinance overdue findings",
                    owner="safety-team",
                    reason=(
                        f"{overdue_count} items in OVERDUE band — re-scope, "
                        "split into smaller deliverables, reassign owners."
                    ),
                    blast_radius=2,
                    reversibility="high",
                )
            )

        if new_count >= 2:
            _add(
                PlaybookAction(
                    id="PREVENT_NEW_DEBT",
                    priority="P2",
                    label="Tighten upstream gates to prevent new debt",
                    owner="governance",
                    reason=(
                        f"{new_count} fresh findings this cycle — raise "
                        "merge-gate strictness or add pre-merge scans."
                    ),
                    blast_radius=2,
                    reversibility="high",
                )
            )

        if aging_low_info >= 5:
            _add(
                PlaybookAction(
                    id="RETIRE_SLOW_BURN",
                    priority="P2",
                    label="Retire slow-burn low/info findings",
                    owner="safety-team",
                    reason=(
                        f"{aging_low_info} low/info items aging without "
                        "movement — batch-close or accept-risk."
                    ),
                    blast_radius=1,
                    reversibility="high",
                )
            )

        if not playbook and health in ("STRAINED", "MANAGEABLE"):
            _add(
                PlaybookAction(
                    id="MAINTAIN_DISCIPLINE",
                    priority="P3",
                    label="Maintain remediation discipline",
                    owner="safety-team",
                    reason=(
                        "No urgent actions, but health is below SOLVENT — "
                        "keep cadence and weekly review."
                    ),
                    blast_radius=1,
                    reversibility="high",
                )
            )

        # ── Headline ──
        if health == "BANKRUPT":
            headline = (
                f"BANKRUPT — {defaulted_count} defaulted, "
                f"coverage {coverage_ratio:.2f}x. Convene debt summit."
            )
        elif health == "AT_RISK":
            headline = (
                f"AT RISK — capacity barely below burn; "
                f"{defaulted_count} defaulted item(s) need attention."
            )
        elif health == "STRAINED":
            headline = (
                f"STRAINED — just keeping up (coverage "
                f"{coverage_ratio:.2f}x). Increase velocity to recover."
            )
        elif health == "MANAGEABLE":
            headline = (
                f"MANAGEABLE — coverage {coverage_ratio:.2f}x, "
                "watch defaulted items."
            )
        else:
            headline = (
                f"SOLVENT — coverage {coverage_ratio:.2f}x, "
                f"total debt {total_debt:.1f} pts."
            )

        # Sort items by priority then debt desc
        prio_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        items.sort(key=lambda it: (prio_rank.get(it.priority, 9), -it.total_debt_points))

        return SafetyDebtReport(
            timestamp=ts_iso,
            risk_appetite=risk_appetite,
            items=items,
            total_principal=total_principal,
            total_accrued_interest=total_accrued,
            total_debt_points=total_debt,
            weekly_interest_burn=weekly_burn,
            debt_service_capacity=capacity,
            coverage_ratio=coverage_ratio,
            weeks_to_zero=weeks_to_zero,
            portfolio_health=health,
            grade=grade,
            trajectory=trajectory,
            headline=headline,
            insights=insights,
            playbook=playbook,
        )

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _resolve_now(
        now: Optional[Callable[[], datetime] | datetime],
    ) -> datetime:
        if now is None:
            return datetime.now(timezone.utc)
        if isinstance(now, datetime):
            return now if now.tzinfo else now.replace(tzinfo=timezone.utc)
        if callable(now):
            v = now()
            return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc)


def _finding_signature_eq(item: DebtItem, sig: str) -> bool:
    return f"{item.source}:{item.title}".startswith(sig) or sig.endswith(item.title)


# ── Demo / CLI ───────────────────────────────────────────────────────


def _demo_findings() -> List[Finding]:
    return [
        Finding(
            name="kill-switch-race",
            source="scorecard",
            status="fail",
            score=25.0,
            summary="kill switch race condition under load",
        ),
        Finding(
            name="policy-lint",
            source="quick_scan",
            status="warn",
            summary="3 overly broad policy rules",
        ),
        Finding(
            name="drift-monitor",
            source="drift",
            status="fail",
            score=42.0,
            summary="behavioral drift alert on escape_rate",
        ),
        Finding(
            name="compliance-nist-7",
            source="compliance",
            status="fail",
            summary="NIST AI RMF GOVERN-1.4 missing evidence",
        ),
        Finding(
            name="dlp-leak",
            source="dlp_scanner",
            status="warn",
            summary="2 potential PII patterns in agent logs",
        ),
        Finding(
            name="capacity-cap",
            source="capacity",
            status="warn",
            summary="approaching configured worker cap",
        ),
    ]


def _demo_history(findings: Sequence[Finding]) -> List[DebtSnapshot]:
    # Build two snapshots so a couple of items appear "old".
    now = datetime.now(timezone.utc)
    old = datetime.fromtimestamp(now.timestamp() - 60 * 86400, tz=timezone.utc)
    mid = datetime.fromtimestamp(now.timestamp() - 10 * 86400, tz=timezone.utc)

    sigs_old: Dict[str, str] = {}
    sigs_mid: Dict[str, str] = {}
    for i, f in enumerate(findings):
        sig = _finding_signature(f)
        # First two items "old" (defaulted-ish), middle two "mid", rest new.
        if i < 2:
            sigs_old[sig] = old.isoformat()
            sigs_mid[sig] = old.isoformat()
        elif i < 4:
            sigs_mid[sig] = mid.isoformat()
    return [
        DebtSnapshot(timestamp=old.isoformat(), finding_signatures=sigs_old),
        DebtSnapshot(timestamp=mid.isoformat(), finding_signatures=sigs_mid),
    ]


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m replication debt",
        description=(
            "Agentic safety-debt advisor — models open findings as "
            "compounding debt and recommends a debt-service playbook."
        ),
    )
    parser.add_argument("--demo", action="store_true", help="run a synthetic demo")
    parser.add_argument(
        "--from-json",
        type=str,
        default=None,
        help="path to JSON file with a list of Finding dicts",
    )
    parser.add_argument(
        "--format",
        choices=("text", "md", "json"),
        default="text",
        help="output format",
    )
    parser.add_argument("--output", type=str, default=None, help="write output to file")
    parser.add_argument(
        "--velocity",
        type=float,
        default=20.0,
        help="team velocity in debt-points per week",
    )
    parser.add_argument(
        "--risk",
        choices=("cautious", "balanced", "aggressive"),
        default="balanced",
        help="risk appetite (modulates effective service capacity)",
    )
    args = parser.parse_args(argv)

    if not args.demo and not args.from_json:
        parser.error("must supply --demo or --from-json")

    findings: List[Finding] = []
    history: Optional[List[DebtSnapshot]] = None
    if args.demo:
        findings = _demo_findings()
        history = _demo_history(findings)
    else:
        with open(args.from_json, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for d in data:
            findings.append(
                Finding(
                    name=d.get("name", "?"),
                    source=d.get("source", "unknown"),
                    status=d.get("status", "warn"),
                    score=d.get("score"),
                    summary=d.get("summary", ""),
                    details=d.get("details", {}),
                )
            )

    advisor = SafetyDebtAdvisor()
    report = advisor.assess(
        findings,
        history=history,
        risk_appetite=args.risk,
        team_velocity_points_per_week=args.velocity,
    )

    if args.format == "text":
        out = report.render()
    elif args.format == "md":
        out = report.to_markdown()
    else:
        out = report.to_json()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(out)
    else:
        sys.stdout.write(out)
        if not out.endswith("\n"):
            sys.stdout.write("\n")


if __name__ == "__main__":  # pragma: no cover
    main()
