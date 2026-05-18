"""Remediation ROI Advisor - agentic per-action ROI ranker.

5th sibling in the remediation suite:

* :mod:`remediation_planner`     - **what** to fix
* :mod:`remediation_progress`    - **are we** fixing it (velocity)
* :mod:`remediation_assignment`  - **who** owns each fix
* :mod:`safety_debt`             - **how much debt** has piled up
* :mod:`remediation_roi`         - **which fixes are worth it** this sprint

For every action in a :class:`RemediationPlan`, the advisor projects how
much accumulated debt (principal + future compounding interest) it would
*pay down* if shipped now, against its effort cost. Output is a ranked
verdict ladder ``QUICK_WIN`` / ``STRATEGIC`` / ``OPTIONAL`` / ``DEFER`` /
``SKIP`` so a team can fund the right work this sprint.

CLI usage::

    python -m replication roi --demo
    python -m replication roi --demo --format md --risk cautious
    python -m replication roi --demo --velocity 25 --horizon 12

Programmatic::

    from replication.remediation_planner import RemediationPlanner
    from replication.remediation_roi import RemediationROIAdvisor
    from replication.safety_debt import _demo_findings

    findings = _demo_findings()
    plan = RemediationPlanner().plan_from_findings(findings)
    advisor = RemediationROIAdvisor(risk_appetite="balanced")
    report = advisor.assess(plan, findings=findings)
    print(report.render())
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from .remediation_planner import (
    Finding,
    RemediationAction,
    RemediationPlan,
    RemediationPlanner,
)
from .remediation_progress import SLA_DAYS_BY_SEVERITY
from .safety_debt import (
    INTEREST_RATE_PER_WEEK,
    SEVERITY_PRINCIPAL,
    SafetyDebtAdvisor,
    SafetyDebtReport,
)


# ── Constants ────────────────────────────────────────────────────────


APPETITE_COST_MULT: Dict[str, float] = {
    "cautious": 1.10,
    "balanced": 1.00,
    "aggressive": 0.90,
}


VALID_APPETITES = ("cautious", "balanced", "aggressive")


# ── Data model ───────────────────────────────────────────────────────


@dataclass
class ROIAction:
    """Per-action ROI evaluation."""

    action_id: str
    title: str
    severity: str
    effort_days: int
    sla_days: float
    debt_paid_down: float
    interest_avoided: float
    principal: float
    cost_units: float
    roi_ratio: float
    breakeven_weeks: Optional[float]
    verdict: str         # QUICK_WIN | STRATEGIC | OPTIONAL | DEFER | SKIP
    priority: str        # P0 | P1 | P2 | P3
    reasons: List[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "title": self.title,
            "severity": self.severity,
            "effort_days": self.effort_days,
            "sla_days": round(self.sla_days, 3),
            "principal": round(self.principal, 4),
            "interest_avoided": round(self.interest_avoided, 4),
            "debt_paid_down": round(self.debt_paid_down, 4),
            "cost_units": round(self.cost_units, 4),
            "roi_ratio": round(self.roi_ratio, 4),
            "breakeven_weeks": (
                None if self.breakeven_weeks is None
                else round(self.breakeven_weeks, 4)
            ),
            "verdict": self.verdict,
            "priority": self.priority,
            "reasons": list(self.reasons),
            "rationale": self.rationale,
        }


@dataclass
class ROIPlaybookItem:
    """One deduped P0-first playbook action."""

    id: str
    priority: str
    label: str
    reason: str
    owner: str
    blast_radius: int          # 1-5
    reversibility: str         # low | medium | high
    action_ids: List[str] = field(default_factory=list)
    suggested_value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "priority": self.priority,
            "label": self.label,
            "reason": self.reason,
            "owner": self.owner,
            "blast_radius": self.blast_radius,
            "reversibility": self.reversibility,
            "action_ids": list(self.action_ids),
            "suggested_value": self.suggested_value,
        }


@dataclass
class RemediationROIReport:
    """Full ROI report."""

    generated_at: datetime
    horizon_weeks: float
    risk_appetite: str
    team_velocity_points_per_week: float

    actions: List[ROIAction] = field(default_factory=list)
    total_debt_paid_down: float = 0.0
    total_effort_days: float = 0.0
    portfolio_roi: float = 0.0
    portfolio_grade: str = "A"
    portfolio_verdict: str = "HIGH_ROI"
    portfolio_breakeven_weeks: Optional[float] = None

    playbook: List[ROIPlaybookItem] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    headline: str = ""

    # ── exporters ────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "horizon_weeks": self.horizon_weeks,
            "risk_appetite": self.risk_appetite,
            "team_velocity_points_per_week": self.team_velocity_points_per_week,
            "headline": self.headline,
            "portfolio": {
                "grade": self.portfolio_grade,
                "verdict": self.portfolio_verdict,
                "roi": round(self.portfolio_roi, 4),
                "total_debt_paid_down": round(self.total_debt_paid_down, 4),
                "total_effort_days": round(self.total_effort_days, 4),
                "breakeven_weeks": (
                    None if self.portfolio_breakeven_weeks is None
                    else round(self.portfolio_breakeven_weeks, 4)
                ),
            },
            "actions": [a.to_dict() for a in self.actions],
            "playbook": [p.to_dict() for p in self.playbook],
            "insights": list(self.insights),
        }

    def to_json(self) -> str:
        # Deterministic byte-stable output.
        return json.dumps(self.to_dict(), sort_keys=True, indent=2, default=str)

    def render(self) -> str:
        bar = "-" * 60
        lines: List[str] = []
        lines.append(bar)
        lines.append(" REMEDIATION ROI REPORT")
        lines.append(bar)
        lines.append(f" Generated:     {self.generated_at.isoformat()}")
        lines.append(f" Risk appetite: {self.risk_appetite}")
        lines.append(f" Horizon:       {self.horizon_weeks:.1f} weeks")
        lines.append(f" Velocity:      {self.team_velocity_points_per_week:.1f} pts/wk")
        lines.append(bar)
        lines.append(f" {self.headline}")
        lines.append(bar)

        if not self.actions:
            lines.append(" (no actions to evaluate)")
        else:
            prio_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
            ordered = sorted(
                self.actions,
                key=lambda a: (prio_rank.get(a.priority, 9), -a.roi_ratio),
            )
            for a in ordered:
                lines.append(
                    f" [{a.priority}] {a.verdict:<9} | {a.title}  "
                    f"(effort {a.effort_days}d, ROI {a.roi_ratio:.2f}x)"
                )
                if a.rationale:
                    lines.append(f"    -> {a.rationale}")

        lines.append("")
        lines.append(" Playbook:")
        if self.playbook:
            for p in self.playbook:
                lines.append(f"  - [{p.priority}] {p.label}  (owner={p.owner})")
                lines.append(f"      reason: {p.reason}")
        else:
            lines.append("  (none)")

        lines.append("")
        lines.append(" Insights:")
        if self.insights:
            for ins in self.insights:
                lines.append(f"  * {ins}")
        else:
            lines.append("  (none)")
        lines.append(bar)
        return "\n".join(lines)

    def to_markdown(self) -> str:
        lines: List[str] = []
        lines.append("# Remediation ROI Report")
        lines.append("")
        lines.append(f"_Generated: {self.generated_at.isoformat()}_")
        lines.append("")
        lines.append(f"**{self.headline}**")
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- Grade: **{self.portfolio_grade}** ({self.portfolio_verdict})")
        lines.append(f"- Portfolio ROI: **{self.portfolio_roi:.2f}x**")
        lines.append(
            f"- Total debt paid down: {self.total_debt_paid_down:.2f} pts "
            f"for {self.total_effort_days:.1f} engineer-days"
        )
        if self.portfolio_breakeven_weeks is not None:
            lines.append(
                f"- Portfolio breakeven: {self.portfolio_breakeven_weeks:.2f} weeks "
                f"at {self.team_velocity_points_per_week:.1f} pts/wk velocity"
            )
        lines.append(f"- Horizon: {self.horizon_weeks:.1f} weeks")
        lines.append(f"- Risk appetite: {self.risk_appetite}")
        lines.append("")

        lines.append("## Actions")
        lines.append("")
        lines.append(
            "| id | severity | effort | debt paid | ROI | breakeven (wk) | "
            "verdict | priority | reasons |"
        )
        lines.append("|----|----------|--------|-----------|-----|----------------|"
                     "---------|----------|---------|")
        prio_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        ordered = sorted(
            self.actions,
            key=lambda a: (prio_rank.get(a.priority, 9), -a.roi_ratio),
        )
        for a in ordered:
            be = "-" if a.breakeven_weeks is None else f"{a.breakeven_weeks:.2f}"
            reasons = ", ".join(a.reasons) if a.reasons else "-"
            lines.append(
                f"| `{a.action_id}` | {a.severity} | {a.effort_days}d | "
                f"{a.debt_paid_down:.2f} | {a.roi_ratio:.2f}x | {be} | "
                f"{a.verdict} | {a.priority} | {reasons} |"
            )
        if not self.actions:
            lines.append("| (none) |  |  |  |  |  |  |  |  |")
        lines.append("")

        lines.append("## Playbook")
        lines.append("")
        if self.playbook:
            lines.append("| priority | id | label | owner | blast | reversibility | reason |")
            lines.append("|----------|----|-------|-------|-------|---------------|--------|")
            for p in self.playbook:
                lines.append(
                    f"| {p.priority} | `{p.id}` | {p.label} | {p.owner} | "
                    f"{p.blast_radius} | {p.reversibility} | {p.reason} |"
                )
        else:
            lines.append("_(empty)_")
        lines.append("")

        lines.append("## Insights")
        lines.append("")
        if self.insights:
            for ins in self.insights:
                lines.append(f"- {ins}")
        else:
            lines.append("- (none)")
        lines.append("")
        return "\n".join(lines)


# ── Advisor ──────────────────────────────────────────────────────────


class RemediationROIAdvisor:
    """Agentic ROI advisor for remediation plans."""

    def __init__(
        self,
        *,
        horizon_weeks: float = 8.0,
        team_velocity_points_per_week: float = 20.0,
        risk_appetite: str = "balanced",
        now: Optional[Callable[[], datetime]] = None,
    ) -> None:
        if risk_appetite not in VALID_APPETITES:
            raise ValueError(
                f"risk_appetite must be one of {VALID_APPETITES}, "
                f"got {risk_appetite!r}"
            )
        if horizon_weeks <= 0:
            raise ValueError("horizon_weeks must be positive")
        self.horizon_weeks = float(horizon_weeks)
        self.team_velocity_points_per_week = float(team_velocity_points_per_week)
        self.risk_appetite = risk_appetite
        self._now = now or (lambda: datetime.now(timezone.utc))

    # ── core ────────────────────────────────────────────────────────

    def assess(
        self,
        plan_or_actions: Union[RemediationPlan, Sequence[RemediationAction]],
        findings: Optional[Sequence[Finding]] = None,
        debt_report: Optional[SafetyDebtReport] = None,
    ) -> RemediationROIReport:
        # Resolve action list without mutating inputs.
        if isinstance(plan_or_actions, RemediationPlan):
            actions: List[RemediationAction] = list(plan_or_actions.actions)
        else:
            actions = list(plan_or_actions)

        # debt_report is accepted for future cross-correlation; we
        # currently derive avoidance directly from severity to stay
        # robust when callers do not pre-compute it.  We still touch
        # it defensively so type-checkers see the usage.
        _ = debt_report

        roi_actions: List[ROIAction] = [
            self._evaluate_action(a) for a in actions
        ]

        # Portfolio aggregates.
        selected = [a for a in roi_actions if a.verdict != "SKIP"]
        total_debt = sum(a.debt_paid_down for a in selected)
        total_effort = float(sum(a.effort_days for a in selected))
        portfolio_roi = total_debt / max(total_effort, 0.5)

        if portfolio_roi >= 8:
            portfolio_verdict = "HIGH_ROI"
        elif portfolio_roi >= 3:
            portfolio_verdict = "BALANCED"
        elif portfolio_roi >= 1:
            portfolio_verdict = "LOW_ROI"
        else:
            portfolio_verdict = "UPSIDE_DOWN"

        skip_count = sum(1 for a in roi_actions if a.verdict == "SKIP")
        non_skip_share = (
            (len(roi_actions) - skip_count) / len(roi_actions)
            if roi_actions else 1.0
        )

        if portfolio_verdict == "UPSIDE_DOWN":
            grade = "F"
        elif portfolio_verdict == "LOW_ROI":
            grade = "D"
        elif portfolio_verdict == "BALANCED":
            grade = "C" if non_skip_share < 0.7 else "B"
        else:  # HIGH_ROI
            grade = "A" if skip_count == 0 else "B"

        # Empty-portfolio short-circuit: nothing to do means we're fine.
        if not roi_actions:
            grade = "A"
            portfolio_verdict = "BALANCED"

        portfolio_breakeven: Optional[float] = None
        if self.team_velocity_points_per_week > 0 and total_effort > 0:
            portfolio_breakeven = total_effort / self.team_velocity_points_per_week

        playbook = self._build_playbook(
            roi_actions=roi_actions,
            total_effort=total_effort,
            portfolio_breakeven=portfolio_breakeven,
            grade=grade,
        )

        insights = self._build_insights(
            roi_actions=roi_actions,
            grade=grade,
            total_effort=total_effort,
        )

        headline = self._build_headline(
            roi_actions=roi_actions,
            portfolio_roi=portfolio_roi,
            portfolio_verdict=portfolio_verdict,
        )

        return RemediationROIReport(
            generated_at=self._now(),
            horizon_weeks=self.horizon_weeks,
            risk_appetite=self.risk_appetite,
            team_velocity_points_per_week=self.team_velocity_points_per_week,
            actions=roi_actions,
            total_debt_paid_down=total_debt,
            total_effort_days=total_effort,
            portfolio_roi=portfolio_roi,
            portfolio_grade=grade,
            portfolio_verdict=portfolio_verdict,
            portfolio_breakeven_weeks=portfolio_breakeven,
            playbook=playbook,
            insights=insights,
            headline=headline,
        )

    # ── per-action evaluation ───────────────────────────────────────

    def _evaluate_action(self, action: RemediationAction) -> ROIAction:
        sev = action.severity if action.severity in SEVERITY_PRINCIPAL else "info"
        principal = SEVERITY_PRINCIPAL[sev]
        weekly_rate = INTEREST_RATE_PER_WEEK[sev]
        sla_days = float(SLA_DAYS_BY_SEVERITY.get(sev, 30.0))

        # Total avoided debt if shipped now = principal + interest over horizon.
        debt_paid_down = principal * ((1.0 + weekly_rate) ** self.horizon_weeks)
        interest_avoided = debt_paid_down - principal

        effort = max(int(action.effort), 1)
        appetite_mult = APPETITE_COST_MULT[self.risk_appetite]
        cost_units = effort * appetite_mult

        roi_ratio = debt_paid_down / max(cost_units, 0.5)

        breakeven_weeks: Optional[float] = None
        if self.team_velocity_points_per_week > 0:
            breakeven_weeks = cost_units / self.team_velocity_points_per_week

        # Verdict ladder (highest first wins).
        if roi_ratio >= 10 and effort <= 2:
            verdict = "QUICK_WIN"
            priority = "P0"
        elif roi_ratio >= 4 and sev in ("critical", "high"):
            verdict = "STRATEGIC"
            priority = "P0" if sev == "critical" else "P1"
        elif roi_ratio >= 1.5:
            verdict = "OPTIONAL"
            priority = "P2"
        elif roi_ratio >= 0.5:
            verdict = "DEFER"
            priority = "P3"
        else:
            verdict = "SKIP"
            priority = "P3"

        reasons: List[str] = []
        if weekly_rate >= 0.15:
            reasons.append("INTEREST_COMPOUNDS_FAST")
        if effort <= 2:
            reasons.append("LOW_EFFORT")
        if effort >= 4:
            reasons.append("HIGH_EFFORT")
        if sev == "critical":
            reasons.append("CRITICAL_SEVERITY")
        if roi_ratio >= 10:
            reasons.append("OUTSIZED_RETURN")
        if 0.5 <= roi_ratio < 1.5:
            reasons.append("MARGINAL_RETURN")
        if roi_ratio < 0.5:
            reasons.append("LOW_ROI_NOT_WORTH_IT")
        if sla_days <= 3:
            reasons.append("SLA_TIGHT")
        if (
            self.risk_appetite == "aggressive"
            and verdict in ("DEFER", "OPTIONAL")
        ):
            reasons.append("AGGRESSIVE_PRUNED")

        reasons = sorted(set(reasons))

        rationale = (
            f"Pays down {debt_paid_down:.2f} debt-points for "
            f"{effort} effort-day{'s' if effort != 1 else ''} "
            f"(ROI {roi_ratio:.2f}x); SLA {sla_days:.0f}d."
        )

        return ROIAction(
            action_id=action.id,
            title=action.title,
            severity=sev,
            effort_days=effort,
            sla_days=sla_days,
            debt_paid_down=debt_paid_down,
            interest_avoided=interest_avoided,
            principal=principal,
            cost_units=cost_units,
            roi_ratio=roi_ratio,
            breakeven_weeks=breakeven_weeks,
            verdict=verdict,
            priority=priority,
            reasons=reasons,
            rationale=rationale,
        )

    # ── playbook ────────────────────────────────────────────────────

    def _build_playbook(
        self,
        *,
        roi_actions: List[ROIAction],
        total_effort: float,
        portfolio_breakeven: Optional[float],
        grade: str,
    ) -> List[ROIPlaybookItem]:
        items: List[ROIPlaybookItem] = []

        quick_wins = [a for a in roi_actions if a.verdict == "QUICK_WIN"]
        strategic = [a for a in roi_actions if a.verdict == "STRATEGIC"]
        skips = [a for a in roi_actions if a.verdict == "SKIP"]
        optionals = [a for a in roi_actions if a.verdict == "OPTIONAL"]
        high_effort = [a for a in roi_actions if a.effort_days >= 4]
        sla_tight = [a for a in roi_actions if "SLA_TIGHT" in a.reasons]

        if len(quick_wins) >= 2:
            items.append(ROIPlaybookItem(
                id="FUND_QUICK_WINS_FIRST",
                priority="P0",
                label="Fund quick-win batch this sprint",
                reason=(
                    f"{len(quick_wins)} quick wins each return >=10x debt "
                    f"per effort-day; ship them before anything heavier."
                ),
                owner="eng_manager",
                blast_radius=3,
                reversibility="high",
                action_ids=[a.action_id for a in quick_wins],
            ))

        if strategic:
            crit_strategic = [a for a in strategic if a.severity == "critical"]
            items.append(ROIPlaybookItem(
                id="STAFF_STRATEGIC_BETS",
                priority="P0" if crit_strategic else "P1",
                label="Staff strategic high-severity remediation",
                reason=(
                    f"{len(strategic)} strategic actions targeting "
                    f"critical/high severity findings deserve dedicated owners."
                ),
                owner="eng_lead",
                blast_radius=4,
                reversibility="medium",
                action_ids=[a.action_id for a in strategic],
            ))

        if len(skips) >= 2:
            items.append(ROIPlaybookItem(
                id="KILL_LOW_ROI_TICKETS",
                priority="P1",
                label="Close out low-ROI tickets",
                reason=(
                    f"{len(skips)} actions return <0.5x debt per effort-day; "
                    f"document as accepted risk and stop polishing them."
                ),
                owner="product",
                blast_radius=2,
                reversibility="medium",
                action_ids=[a.action_id for a in skips],
                suggested_value=float(len(skips)),
            ))

        if (
            len(high_effort) >= 2
            and self.team_velocity_points_per_week > 0
            and self.team_velocity_points_per_week
                < (total_effort / max(self.horizon_weeks, 1.0))
        ):
            items.append(ROIPlaybookItem(
                id="RE_NEGOTIATE_SCOPE",
                priority="P1",
                label="Re-negotiate scope of high-effort items",
                reason=(
                    f"{len(high_effort)} actions need >=4 engineer-days each; "
                    f"sprint velocity cannot absorb them within the horizon."
                ),
                owner="product",
                blast_radius=3,
                reversibility="medium",
                action_ids=[a.action_id for a in high_effort],
            ))

        if (
            portfolio_breakeven is not None
            and portfolio_breakeven > self.horizon_weeks / 2.0
        ):
            items.append(ROIPlaybookItem(
                id="INCREASE_VELOCITY_OR_DEFER",
                priority="P1",
                label="Increase velocity or defer non-critical work",
                reason=(
                    f"Portfolio breakeven of {portfolio_breakeven:.1f} weeks "
                    f"exceeds half the {self.horizon_weeks:.0f}-week horizon."
                ),
                owner="eng_manager",
                blast_radius=3,
                reversibility="medium",
                suggested_value=float(
                    math.ceil(total_effort / max(self.horizon_weeks, 1.0))
                ),
            ))

        if len(sla_tight) >= 2:
            items.append(ROIPlaybookItem(
                id="REVISIT_SLA_BUDGETS",
                priority="P2",
                label="Revisit SLA budgets for tight-deadline items",
                reason=(
                    f"{len(sla_tight)} actions sit under a 3-day SLA; "
                    f"either widen the SLA or pre-allocate on-call time."
                ),
                owner="governance",
                blast_radius=2,
                reversibility="high",
                action_ids=[a.action_id for a in sla_tight],
            ))

        if len(optionals) >= 3:
            items.append(ROIPlaybookItem(
                id="LOCK_IN_OPTIONAL_BETS",
                priority="P2",
                label="Lock in optional bets with positive ROI",
                reason=(
                    f"{len(optionals)} actions have moderate ROI; "
                    f"schedule them once quick wins land."
                ),
                owner="eng_manager",
                blast_radius=2,
                reversibility="high",
                action_ids=[a.action_id for a in optionals],
            ))

        # Cautious-only addition.
        if (
            self.risk_appetite == "cautious"
            and grade in ("C", "D", "F")
            and not any(p.id == "SCHEDULE_ROI_REVIEW" for p in items)
        ):
            items.append(ROIPlaybookItem(
                id="SCHEDULE_ROI_REVIEW",
                priority="P2",
                label="Schedule cross-team ROI review",
                reason=(
                    "Cautious appetite + degraded grade — bring in governance "
                    "to sanity-check the model assumptions."
                ),
                owner="governance",
                blast_radius=1,
                reversibility="high",
            ))

        # Aggressive trims.
        if self.risk_appetite == "aggressive":
            has_p0_p1 = any(p.priority in ("P0", "P1") for p in items)
            items = [
                p for p in items
                if not (p.priority == "P3" and p.id != "PORTFOLIO_HEALTHY")
            ]
            if has_p0_p1:
                p2_items = [p for p in items if p.priority == "P2"]
                if len(p2_items) == 1:
                    items = [p for p in items if p.priority != "P2"]

        if not items:
            items.append(ROIPlaybookItem(
                id="PORTFOLIO_HEALTHY",
                priority="P3",
                label="Portfolio healthy — maintain current cadence",
                reason="No actionable ROI gaps detected this cycle.",
                owner="eng_manager",
                blast_radius=1,
                reversibility="high",
            ))

        # Dedup by id (preserving first occurrence) then P0-first stable sort.
        seen: set = set()
        deduped: List[ROIPlaybookItem] = []
        for item in items:
            if item.id in seen:
                continue
            seen.add(item.id)
            deduped.append(item)
        prio_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        deduped.sort(key=lambda p: (prio_rank.get(p.priority, 9), p.id))
        return deduped

    # ── insights ────────────────────────────────────────────────────

    def _build_insights(
        self,
        *,
        roi_actions: List[ROIAction],
        grade: str,
        total_effort: float,
    ) -> List[str]:
        insights: List[str] = []
        n = len(roi_actions)
        if n == 0:
            insights.append("THIN_PIPELINE")
            return insights

        quick_share = sum(1 for a in roi_actions if a.verdict == "QUICK_WIN") / n
        skip_share = sum(1 for a in roi_actions if a.verdict == "SKIP") / n

        if quick_share >= 0.4:
            insights.append("QUICK_WIN_HEAVY")
        if skip_share >= 0.25:
            insights.append("SKIP_BURDEN")

        capacity = self.team_velocity_points_per_week * self.horizon_weeks
        if total_effort > capacity and capacity > 0:
            insights.append("EFFORT_BOUND")

        total_interest = sum(a.interest_avoided for a in roi_actions)
        total_principal = sum(a.principal for a in roi_actions)
        if total_interest >= total_principal and total_principal > 0:
            insights.append("INTEREST_DOMINATED")

        if n < 3:
            insights.append("THIN_PIPELINE")

        if not insights and grade in ("A", "B"):
            insights.append("BALANCED_PORTFOLIO")

        return sorted(set(insights))

    # ── headline ────────────────────────────────────────────────────

    def _build_headline(
        self,
        *,
        roi_actions: List[ROIAction],
        portfolio_roi: float,
        portfolio_verdict: str,
    ) -> str:
        qw = sum(1 for a in roi_actions if a.verdict == "QUICK_WIN")
        st = sum(1 for a in roi_actions if a.verdict == "STRATEGIC")
        sk = sum(1 for a in roi_actions if a.verdict == "SKIP")
        return (
            f"VERDICT: {portfolio_roi:.2f}x portfolio ROI ({portfolio_verdict}) "
            f"— {qw} quick win{'s' if qw != 1 else ''}, "
            f"{st} strategic, {sk} to skip"
        )


# ── CLI ──────────────────────────────────────────────────────────────


def _ensure_utf8() -> None:
    """Best-effort UTF-8 stdout on Windows consoles."""
    try:
        if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
            sys.stdout = io.TextIOWrapper(  # type: ignore[assignment]
                sys.stdout.buffer, encoding="utf-8", errors="replace"
            )
    except Exception:
        pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    _ensure_utf8()
    parser = argparse.ArgumentParser(
        description=(
            "Agentic remediation ROI advisor — rank fixes by "
            "debt-paid-down per effort-day."
        )
    )
    parser.add_argument("--demo", action="store_true",
                        help="Use a built-in synthetic plan.")
    parser.add_argument("--format", default="text",
                        choices=("text", "md", "json"))
    parser.add_argument("--risk", default="balanced",
                        choices=VALID_APPETITES)
    parser.add_argument("--velocity", type=float, default=20.0,
                        help="Team velocity in debt points per week.")
    parser.add_argument("--horizon", type=float, default=8.0,
                        help="Forecast horizon in weeks.")
    parser.add_argument("--output", default=None,
                        help="Write report to this path instead of stdout.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.demo:
        parser.error("only --demo is supported in this CLI")

    # Local import to avoid surfacing demo helpers in the public API.
    from .safety_debt import _demo_findings  # type: ignore[attr-defined]

    findings = _demo_findings()
    plan = RemediationPlanner().plan_from_findings(findings)

    # Also synthesize a debt report so the advisor can be passed one,
    # mirroring how real callers will wire the suite together.
    debt_report: Optional[SafetyDebtReport] = None
    try:
        debt_report = SafetyDebtAdvisor().assess(
            findings, team_velocity_points_per_week=args.velocity
        )
    except Exception:
        debt_report = None

    advisor = RemediationROIAdvisor(
        horizon_weeks=args.horizon,
        team_velocity_points_per_week=args.velocity,
        risk_appetite=args.risk,
    )
    report = advisor.assess(plan, findings=findings, debt_report=debt_report)

    if args.format == "json":
        out = report.to_json()
    elif args.format == "md":
        out = report.to_markdown()
    else:
        out = report.render()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(out)
    else:
        print(out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
