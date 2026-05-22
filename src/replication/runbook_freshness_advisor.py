"""Runbook Freshness Advisor — agentic incident-runbook library auditor.

Sibling to:
  - :mod:`replication.runbook` (generates runbooks for new threats)
  - :mod:`replication.remediation_staleness` (audits in-flight actions)
  - :mod:`replication.safety_drill` (runs drills)
  - :mod:`replication.ir_playbook` (rich IR playbooks)

This module answers a different question: **which incident runbooks in our
library have rotted?**  A runbook the team has not exercised in months —
or, worse, not *updated* despite repeated executions — is a liability
during a real incident.  The advisor consumes a list of runbook records
with timestamps and emits per-runbook verdicts plus a P0-first deduplicated
playbook of concrete interventions for the safety team.

Per-runbook verdicts
~~~~~~~~~~~~~~~~~~~~

* ``ARCHIVED``                — runbook is marked archived; excluded from health.
* ``ORPHANED_OWNER``          — no owner assigned for a live runbook.
* ``NEVER_EXECUTED``          — runbook has never been drilled or run.
* ``DRILL_OVERDUE``           — last drill is older than the drill cadence.
* ``STALE_CONTENT``           — not updated in N days while still active.
* ``DRIFT_VS_EXECUTION``      — executed since last update by a large margin
                                (content lags real-world usage).
* ``AT_RISK``                 — close to crossing a freshness threshold.
* ``FRESH``                   — owner present, updated and drilled recently.
* ``INSUFFICIENT_DATA``       — missing critical timestamps.

Cross-portfolio insights
~~~~~~~~~~~~~~~~~~~~~~~~

* ``WIDESPREAD_STALENESS``    — >40 % of active runbooks are STALE_CONTENT.
* ``DRILL_CULTURE_GAP``       — >50 % have NEVER_EXECUTED or DRILL_OVERDUE.
* ``OWNERLESS_BACKLOG``       — >25 % active runbooks are ORPHANED_OWNER.
* ``CRITICAL_RUNBOOKS_ROTTING`` — any critical/high runbook is STALE or worse.
* ``EXECUTION_OUTPACING_DOCS``  — multiple runbooks show DRIFT_VS_EXECUTION.
* ``HEALTHY_LIBRARY``         — majority FRESH and no rotting critical books.
* ``EMPTY_LIBRARY``           — no runbooks supplied.

Risk appetite (``cautious`` | ``balanced`` | ``aggressive``) scales the
freshness thresholds — cautious tightens by ×0.7, aggressive loosens
by ×1.4.

CLI demo::

    python -m replication.runbook_freshness_advisor --demo --format markdown
    python -m replication.runbook_freshness_advisor --from-json library.json --risk cautious

Programmatic::

    from datetime import datetime, timezone
    from replication.runbook_freshness_advisor import (
        RunbookFreshnessAdvisor,
        FreshnessInput,
        RunbookRecord,
    )

    now = datetime(2026, 5, 22, tzinfo=timezone.utc)
    advisor = RunbookFreshnessAdvisor(now=lambda: now)
    report = advisor.audit(FreshnessInput(runbooks=[
        RunbookRecord(
            id="rb-001",
            title="Self-replication containment",
            severity="critical",
            owner="safety-oncall",
            last_updated_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
            last_executed_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
            last_drill_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
        ),
    ]))
    print(report.to_markdown())
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from ._helpers import (
    APPETITE_THRESHOLD_MULT,
    APPETITES,
    SEVERITY_LEVELS,
    SEVERITY_WEIGHT,
)


# ── Constants ─────────────────────────────────────────────────────────

# Verdicts (exported for downstream filtering / tests).
VERDICT_ARCHIVED = "ARCHIVED"
VERDICT_ORPHANED_OWNER = "ORPHANED_OWNER"
VERDICT_NEVER_EXECUTED = "NEVER_EXECUTED"
VERDICT_DRILL_OVERDUE = "DRILL_OVERDUE"
VERDICT_STALE_CONTENT = "STALE_CONTENT"
VERDICT_DRIFT_VS_EXECUTION = "DRIFT_VS_EXECUTION"
VERDICT_AT_RISK = "AT_RISK"
VERDICT_FRESH = "FRESH"
VERDICT_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

ALL_VERDICTS: Tuple[str, ...] = (
    VERDICT_ARCHIVED,
    VERDICT_ORPHANED_OWNER,
    VERDICT_NEVER_EXECUTED,
    VERDICT_DRILL_OVERDUE,
    VERDICT_STALE_CONTENT,
    VERDICT_DRIFT_VS_EXECUTION,
    VERDICT_AT_RISK,
    VERDICT_FRESH,
    VERDICT_INSUFFICIENT_DATA,
)

# Verdict severity ordering — used when one runbook trips multiple
# conditions; the worst wins.
_VERDICT_RANK: Dict[str, int] = {
    VERDICT_FRESH: 0,
    VERDICT_INSUFFICIENT_DATA: 1,
    VERDICT_AT_RISK: 2,
    VERDICT_DRIFT_VS_EXECUTION: 3,
    VERDICT_DRILL_OVERDUE: 4,
    VERDICT_STALE_CONTENT: 5,
    VERDICT_NEVER_EXECUTED: 6,
    VERDICT_ORPHANED_OWNER: 7,
    VERDICT_ARCHIVED: -1,  # excluded from worst-of comparisons
}

STATUS_ACTIVE = {"active", "live", "published", "in_use", ""}
STATUS_ARCHIVED = {"archived", "retired", "deprecated", "draft", "wip"}

# Balanced-appetite day thresholds.
BASE_THRESHOLDS: Dict[str, float] = {
    # No content edit for > N days while active -> STALE_CONTENT.
    "stale_days": 180.0,
    # Approaching stale_days (within this many days) -> AT_RISK.
    "at_risk_days_before_stale": 30.0,
    # No drill in > N days -> DRILL_OVERDUE.
    "drill_cadence_days": 180.0,
    # Executions newer than last_update by > N days -> DRIFT_VS_EXECUTION.
    "drift_days": 30.0,
    # Minimum execution count that makes DRIFT meaningful.
    "drift_min_executions": 1.0,
    # Books updated within this window are unambiguously FRESH.
    "fresh_recent_days": 60.0,
}

# Cautious tightens thresholds (multiplier < 1), aggressive loosens. Mirrors
# the pattern in ``remediation_staleness`` so the whole suite responds to
# the same risk_appetite knob.
# APPETITE_THRESHOLD_MULT is re-exported from ``_helpers``.


# ── Data model ────────────────────────────────────────────────────────


@dataclass
class RunbookRecord:
    """A single runbook entry under review.

    All timestamps are timezone-aware UTC :class:`datetime`. Optional
    fields default to ``None`` so partial library exports still audit
    cleanly (missing data shows up as INSUFFICIENT_DATA, never crashes).
    """

    id: str
    title: str = ""
    severity: str = "medium"
    owner: Optional[str] = None
    status: str = "active"
    last_updated_at: Optional[datetime] = None
    last_executed_at: Optional[datetime] = None
    last_drill_at: Optional[datetime] = None
    execution_count: int = 0
    tags: Tuple[str, ...] = ()

    def normalized_status(self) -> str:
        s = (self.status or "").strip().lower()
        if s in STATUS_ARCHIVED:
            return "archived"
        return "active"

    def normalized_severity(self) -> str:
        s = (self.severity or "medium").strip().lower()
        return s if s in SEVERITY_WEIGHT else "medium"


@dataclass
class FreshnessInput:
    runbooks: List[RunbookRecord] = field(default_factory=list)
    risk_appetite: str = "balanced"


@dataclass
class FreshnessFinding:
    runbook_id: str
    title: str
    verdict: str
    priority: str  # P0..P3
    freshness_score: float  # 0..100 (higher = healthier)
    days_since_update: Optional[float]
    days_since_execution: Optional[float]
    days_since_drill: Optional[float]
    reasons: List[str]
    suggested_action: str
    owner: Optional[str] = None
    severity: str = "medium"


@dataclass
class FreshnessPortfolio:
    total: int
    active: int
    archived: int
    fresh: int
    at_risk: int
    stale: int
    overdue_drills: int
    orphaned: int
    never_executed: int
    drift: int
    insufficient_data: int
    mean_freshness: float
    grade: str  # A..F


@dataclass
class FreshnessPlaybookAction:
    priority: str
    label: str
    reason: str
    runbook_ids: List[str]
    suggested_value: Optional[str] = None


@dataclass
class FreshnessReport:
    generated_at: datetime
    risk_appetite: str
    thresholds: Dict[str, float]
    portfolio: FreshnessPortfolio
    findings: List[FreshnessFinding]
    insights: List[str]
    playbook: List[FreshnessPlaybookAction]

    # ── Renderers ───────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        return _serialize(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2, default=str)

    def to_text(self) -> str:
        p = self.portfolio
        lines: List[str] = []
        lines.append(
            f"Runbook Freshness — grade={p.grade} appetite={self.risk_appetite} "
            f"active={p.active} fresh={p.fresh} stale={p.stale} "
            f"overdue_drills={p.overdue_drills} orphaned={p.orphaned} "
            f"mean_freshness={p.mean_freshness:.1f}"
        )
        lines.append("")
        lines.append("Findings:")
        for f in self.findings:
            since_upd = (
                "-" if f.days_since_update is None
                else f"{f.days_since_update:.0f}d"
            )
            lines.append(
                f"  [{f.priority}] {f.runbook_id} {f.verdict} "
                f"score={f.freshness_score:.0f} updated={since_upd} "
                f"owner={f.owner or '-'} sev={f.severity}"
            )
        if self.insights:
            lines.append("")
            lines.append("Insights:")
            for i in self.insights:
                lines.append(f"  - {i}")
        if self.playbook:
            lines.append("")
            lines.append("Playbook:")
            for a in self.playbook:
                lines.append(
                    f"  [{a.priority}] {a.label} "
                    f"({len(a.runbook_ids)} runbook(s)) — {a.reason}"
                )
        return "\n".join(lines)

    def to_markdown(self) -> str:
        p = self.portfolio
        lines: List[str] = []
        lines.append("# Runbook Freshness Report")
        lines.append("")
        lines.append(f"- generated_at: `{self.generated_at.isoformat()}`")
        lines.append(f"- risk_appetite: **{self.risk_appetite}**")
        lines.append(
            f"- portfolio: grade **{p.grade}** | "
            f"active={p.active}/{p.total} fresh={p.fresh} "
            f"stale={p.stale} overdue_drills={p.overdue_drills} "
            f"orphaned={p.orphaned} drift={p.drift} "
            f"never_executed={p.never_executed} "
            f"mean_freshness={p.mean_freshness:.1f}"
        )
        lines.append("")
        if self.insights:
            lines.append("## Insights")
            lines.append("")
            for i in self.insights:
                lines.append(f"- {i}")
            lines.append("")
        lines.append("## Findings")
        lines.append("")
        lines.append("| Priority | Runbook | Verdict | Score | Updated | Drilled | Owner | Sev |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for f in self.findings:
            upd = "-" if f.days_since_update is None else f"{f.days_since_update:.0f}d ago"
            dr = "-" if f.days_since_drill is None else f"{f.days_since_drill:.0f}d ago"
            lines.append(
                f"| {f.priority} | `{f.runbook_id}` {f.title} | {f.verdict} | "
                f"{f.freshness_score:.0f} | {upd} | {dr} | "
                f"{f.owner or '-'} | {f.severity} |"
            )
        if self.playbook:
            lines.append("")
            lines.append("## Playbook")
            lines.append("")
            for a in self.playbook:
                ids = ", ".join(f"`{x}`" for x in a.runbook_ids[:5])
                more = "" if len(a.runbook_ids) <= 5 else f" (+{len(a.runbook_ids)-5} more)"
                val = "" if not a.suggested_value else f" — _{a.suggested_value}_"
                lines.append(
                    f"- **[{a.priority}] {a.label}** — {a.reason}{val}\n"
                    f"  - runbooks: {ids}{more}"
                )
        return "\n".join(lines)


# ── Advisor ───────────────────────────────────────────────────────────


class RunbookFreshnessAdvisor:
    """Stateless advisor; pass a ``now`` callable for deterministic tests."""

    def __init__(
        self,
        now: Optional[Callable[[], datetime]] = None,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._base_thresholds = dict(BASE_THRESHOLDS)
        if thresholds:
            self._base_thresholds.update(thresholds)

    # ── Public API ──────────────────────────────────────────────────

    def audit(self, payload: FreshnessInput) -> FreshnessReport:
        appetite = payload.risk_appetite if payload.risk_appetite in APPETITES else "balanced"
        thresholds = self._scaled_thresholds(appetite)
        now = self._ensure_utc(self._now())

        findings: List[FreshnessFinding] = []
        for rb in payload.runbooks:
            findings.append(self._evaluate(rb, now=now, thresholds=thresholds))

        portfolio = self._summarize(findings)
        insights = self._derive_insights(findings, portfolio)
        playbook = self._build_playbook(findings)

        return FreshnessReport(
            generated_at=now,
            risk_appetite=appetite,
            thresholds=thresholds,
            portfolio=portfolio,
            findings=findings,
            insights=insights,
            playbook=playbook,
        )

    # ── Per-runbook evaluation ──────────────────────────────────────

    def _evaluate(
        self,
        rb: RunbookRecord,
        *,
        now: datetime,
        thresholds: Dict[str, float],
    ) -> FreshnessFinding:
        status = rb.normalized_status()
        severity = rb.normalized_severity()

        days_upd = self._days_between(rb.last_updated_at, now)
        days_exec = self._days_between(rb.last_executed_at, now)
        days_drill = self._days_between(rb.last_drill_at, now)

        reasons: List[str] = []
        candidate_verdicts: List[str] = []

        if status == "archived":
            return FreshnessFinding(
                runbook_id=rb.id,
                title=rb.title,
                verdict=VERDICT_ARCHIVED,
                priority="P3",
                freshness_score=0.0,
                days_since_update=days_upd,
                days_since_execution=days_exec,
                days_since_drill=days_drill,
                reasons=["Runbook is archived; excluded from health metrics."],
                suggested_action="No action; archived.",
                owner=rb.owner,
                severity=severity,
            )

        if rb.last_updated_at is None:
            reasons.append("Missing last_updated_at — cannot assess content age.")
            candidate_verdicts.append(VERDICT_INSUFFICIENT_DATA)

        if not (rb.owner or "").strip():
            reasons.append("No owner assigned to runbook.")
            candidate_verdicts.append(VERDICT_ORPHANED_OWNER)

        if rb.last_executed_at is None and rb.last_drill_at is None:
            reasons.append("Runbook has never been executed or drilled.")
            candidate_verdicts.append(VERDICT_NEVER_EXECUTED)
        else:
            # Drill cadence (drills count; production executions also count
            # for "we have proven this works recently").
            most_recent_use = self._max_optional(rb.last_drill_at, rb.last_executed_at)
            days_since_use = self._days_between(most_recent_use, now)
            if (
                days_since_use is not None
                and days_since_use > thresholds["drill_cadence_days"]
            ):
                reasons.append(
                    f"Last drill/execution {days_since_use:.0f}d ago "
                    f"(cadence {thresholds['drill_cadence_days']:.0f}d)."
                )
                candidate_verdicts.append(VERDICT_DRILL_OVERDUE)

        if days_upd is not None:
            if days_upd > thresholds["stale_days"]:
                reasons.append(
                    f"Content unchanged for {days_upd:.0f}d "
                    f"(stale_days={thresholds['stale_days']:.0f})."
                )
                candidate_verdicts.append(VERDICT_STALE_CONTENT)
            elif days_upd > thresholds["stale_days"] - thresholds["at_risk_days_before_stale"]:
                reasons.append(
                    f"Approaching stale threshold ({days_upd:.0f}d / "
                    f"{thresholds['stale_days']:.0f}d)."
                )
                candidate_verdicts.append(VERDICT_AT_RISK)

        # DRIFT: executed since last update by a wide margin AND enough
        # executions to make the gap meaningful.
        if (
            rb.last_updated_at is not None
            and rb.last_executed_at is not None
            and rb.execution_count >= thresholds["drift_min_executions"]
        ):
            upd_utc = self._ensure_utc(rb.last_updated_at)
            exec_utc = self._ensure_utc(rb.last_executed_at)
            gap_days = (exec_utc - upd_utc).total_seconds() / 86400.0
            if gap_days > thresholds["drift_days"]:
                reasons.append(
                    f"Executed {gap_days:.0f}d after last update — content may "
                    f"lag real-world usage."
                )
                candidate_verdicts.append(VERDICT_DRIFT_VS_EXECUTION)

        if not candidate_verdicts:
            reasons.append("Owned, recently updated, drilled within cadence.")
            verdict = VERDICT_FRESH
        else:
            verdict = max(candidate_verdicts, key=_VERDICT_RANK.__getitem__)

        score = self._freshness_score(
            verdict=verdict,
            days_upd=days_upd,
            thresholds=thresholds,
            severity=severity,
        )
        priority = self._priority_for(verdict, severity)
        suggested = self._suggest_action(verdict, rb)

        return FreshnessFinding(
            runbook_id=rb.id,
            title=rb.title,
            verdict=verdict,
            priority=priority,
            freshness_score=score,
            days_since_update=days_upd,
            days_since_execution=days_exec,
            days_since_drill=days_drill,
            reasons=reasons,
            suggested_action=suggested,
            owner=rb.owner,
            severity=severity,
        )

    # ── Portfolio + insights ────────────────────────────────────────

    @staticmethod
    def _summarize(findings: List[FreshnessFinding]) -> FreshnessPortfolio:
        counts: Dict[str, int] = {v: 0 for v in ALL_VERDICTS}
        for f in findings:
            counts[f.verdict] = counts.get(f.verdict, 0) + 1
        total = len(findings)
        archived = counts[VERDICT_ARCHIVED]
        active = total - archived
        active_findings = [f for f in findings if f.verdict != VERDICT_ARCHIVED]
        mean_freshness = (
            sum(f.freshness_score for f in active_findings) / len(active_findings)
            if active_findings else 100.0
        )

        # Letter grade based on mean freshness across active books.
        if active == 0:
            grade = "A"
        elif mean_freshness >= 85:
            grade = "A"
        elif mean_freshness >= 70:
            grade = "B"
        elif mean_freshness >= 55:
            grade = "C"
        elif mean_freshness >= 40:
            grade = "D"
        else:
            grade = "F"

        return FreshnessPortfolio(
            total=total,
            active=active,
            archived=archived,
            fresh=counts[VERDICT_FRESH],
            at_risk=counts[VERDICT_AT_RISK],
            stale=counts[VERDICT_STALE_CONTENT],
            overdue_drills=counts[VERDICT_DRILL_OVERDUE],
            orphaned=counts[VERDICT_ORPHANED_OWNER],
            never_executed=counts[VERDICT_NEVER_EXECUTED],
            drift=counts[VERDICT_DRIFT_VS_EXECUTION],
            insufficient_data=counts[VERDICT_INSUFFICIENT_DATA],
            mean_freshness=round(mean_freshness, 2),
            grade=grade,
        )

    @staticmethod
    def _derive_insights(
        findings: List[FreshnessFinding],
        portfolio: FreshnessPortfolio,
    ) -> List[str]:
        if portfolio.total == 0:
            return ["EMPTY_LIBRARY — supply at least one runbook to audit."]
        insights: List[str] = []
        active = max(portfolio.active, 1)

        if portfolio.stale / active > 0.4:
            insights.append(
                f"WIDESPREAD_STALENESS — {portfolio.stale}/{active} active "
                f"runbooks have stale content."
            )
        if (portfolio.overdue_drills + portfolio.never_executed) / active > 0.5:
            insights.append(
                f"DRILL_CULTURE_GAP — {portfolio.overdue_drills + portfolio.never_executed}"
                f"/{active} runbooks lack a recent drill/execution."
            )
        if portfolio.orphaned / active > 0.25:
            insights.append(
                f"OWNERLESS_BACKLOG — {portfolio.orphaned}/{active} active "
                f"runbooks have no assigned owner."
            )
        if any(
            f.severity in ("critical", "high")
            and f.verdict in (
                VERDICT_STALE_CONTENT,
                VERDICT_NEVER_EXECUTED,
                VERDICT_ORPHANED_OWNER,
                VERDICT_DRILL_OVERDUE,
            )
            for f in findings
        ):
            insights.append(
                "CRITICAL_RUNBOOKS_ROTTING — high/critical-severity runbooks "
                "are stale, orphaned, or undrilled; prioritize remediation."
            )
        if portfolio.drift >= 2:
            insights.append(
                f"EXECUTION_OUTPACING_DOCS — {portfolio.drift} runbooks were "
                f"executed well after their last content update."
            )
        if not insights and portfolio.fresh / active >= 0.6:
            insights.append(
                f"HEALTHY_LIBRARY — {portfolio.fresh}/{active} active runbooks "
                f"are fresh, no critical rot detected."
            )
        return insights

    # ── Playbook ────────────────────────────────────────────────────

    @staticmethod
    def _build_playbook(
        findings: List[FreshnessFinding],
    ) -> List[FreshnessPlaybookAction]:
        # Deduplicated, P0-first. Each verdict gets at most one action.
        groups: Dict[str, List[str]] = {v: [] for v in ALL_VERDICTS}
        for f in findings:
            groups[f.verdict].append(f.runbook_id)

        actions: List[FreshnessPlaybookAction] = []

        if groups[VERDICT_ORPHANED_OWNER]:
            actions.append(FreshnessPlaybookAction(
                priority="P0",
                label="Assign owners",
                reason="Orphaned runbooks have no clear responder during an incident.",
                runbook_ids=sorted(groups[VERDICT_ORPHANED_OWNER]),
                suggested_value="Assign each runbook to a named on-call rotation.",
            ))
        if groups[VERDICT_STALE_CONTENT]:
            crit = sorted(groups[VERDICT_STALE_CONTENT])
            actions.append(FreshnessPlaybookAction(
                priority="P0" if any(
                    f.severity in ("critical", "high")
                    and f.verdict == VERDICT_STALE_CONTENT
                    for f in findings
                ) else "P1",
                label="Refresh stale content",
                reason="Stale runbooks may reference defunct systems or contacts.",
                runbook_ids=crit,
                suggested_value="Schedule editorial review within 14 days.",
            ))
        if groups[VERDICT_NEVER_EXECUTED]:
            actions.append(FreshnessPlaybookAction(
                priority="P1",
                label="Run first drill",
                reason="Unexercised runbooks are unproven; cheaper to fail in a drill.",
                runbook_ids=sorted(groups[VERDICT_NEVER_EXECUTED]),
                suggested_value="Schedule tabletop within current sprint.",
            ))
        if groups[VERDICT_DRILL_OVERDUE]:
            actions.append(FreshnessPlaybookAction(
                priority="P1",
                label="Schedule overdue drills",
                reason="Drill cadence keeps muscle memory and exposes content drift.",
                runbook_ids=sorted(groups[VERDICT_DRILL_OVERDUE]),
                suggested_value="Add to next quarterly drill calendar.",
            ))
        if groups[VERDICT_DRIFT_VS_EXECUTION]:
            actions.append(FreshnessPlaybookAction(
                priority="P1",
                label="Reconcile docs vs. real executions",
                reason="Content lags real usage; capture lessons from recent runs.",
                runbook_ids=sorted(groups[VERDICT_DRIFT_VS_EXECUTION]),
                suggested_value="Pair each execution with a same-week doc update.",
            ))
        if groups[VERDICT_AT_RISK]:
            actions.append(FreshnessPlaybookAction(
                priority="P2",
                label="Pre-empt staleness",
                reason="Runbooks approaching the stale threshold; cheap to refresh now.",
                runbook_ids=sorted(groups[VERDICT_AT_RISK]),
                suggested_value="Light-touch review during normal maintenance window.",
            ))
        if groups[VERDICT_INSUFFICIENT_DATA]:
            actions.append(FreshnessPlaybookAction(
                priority="P2",
                label="Backfill metadata",
                reason="Missing timestamps prevent meaningful audit.",
                runbook_ids=sorted(groups[VERDICT_INSUFFICIENT_DATA]),
                suggested_value="Populate last_updated_at and ownership fields.",
            ))
        return actions

    # ── Helpers ─────────────────────────────────────────────────────

    def _scaled_thresholds(self, appetite: str) -> Dict[str, float]:
        mult = APPETITE_THRESHOLD_MULT.get(appetite, 1.0)
        scaled: Dict[str, float] = {}
        for k, v in self._base_thresholds.items():
            # Counts (drift_min_executions) shouldn't be scaled by day-mult.
            if k == "drift_min_executions":
                scaled[k] = v
            else:
                scaled[k] = round(v * mult, 4)
        return scaled

    @staticmethod
    def _ensure_utc(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @classmethod
    def _days_between(
        cls,
        earlier: Optional[datetime],
        later: datetime,
    ) -> Optional[float]:
        if earlier is None:
            return None
        earlier = cls._ensure_utc(earlier)
        delta = later - earlier
        return max(delta.total_seconds() / 86400.0, 0.0)

    @staticmethod
    def _max_optional(a: Optional[datetime], b: Optional[datetime]) -> Optional[datetime]:
        if a is None:
            return b
        if b is None:
            return a
        return a if a >= b else b

    @staticmethod
    def _freshness_score(
        *,
        verdict: str,
        days_upd: Optional[float],
        thresholds: Dict[str, float],
        severity: str,
    ) -> float:
        # Baseline by verdict.
        base = {
            VERDICT_FRESH: 95.0,
            VERDICT_AT_RISK: 70.0,
            VERDICT_INSUFFICIENT_DATA: 50.0,
            VERDICT_DRIFT_VS_EXECUTION: 45.0,
            VERDICT_DRILL_OVERDUE: 40.0,
            VERDICT_STALE_CONTENT: 25.0,
            VERDICT_NEVER_EXECUTED: 20.0,
            VERDICT_ORPHANED_OWNER: 15.0,
            VERDICT_ARCHIVED: 0.0,
        }.get(verdict, 50.0)

        # Penalize further by how far past stale_days we are.
        if days_upd is not None and days_upd > thresholds["stale_days"]:
            overage = days_upd - thresholds["stale_days"]
            base -= min(overage / max(thresholds["stale_days"], 1.0) * 20.0, 20.0)

        # High/critical runbooks penalized harder so they sink in
        # mean-freshness summaries.
        sev_weight = {"low": 0.0, "medium": 0.0, "high": -5.0, "critical": -10.0}
        if verdict not in (VERDICT_FRESH, VERDICT_ARCHIVED):
            base += sev_weight.get(severity, 0.0)

        return round(max(0.0, min(100.0, base)), 2)

    @staticmethod
    def _priority_for(verdict: str, severity: str) -> str:
        if verdict == VERDICT_ARCHIVED:
            return "P3"
        if verdict == VERDICT_FRESH:
            return "P3"
        # Owner gaps are always P0 — there is no responder.
        if verdict == VERDICT_ORPHANED_OWNER:
            return "P0"
        if verdict == VERDICT_STALE_CONTENT and severity in ("critical", "high"):
            return "P0"
        if verdict in (VERDICT_STALE_CONTENT, VERDICT_NEVER_EXECUTED):
            return "P1"
        if verdict in (VERDICT_DRILL_OVERDUE, VERDICT_DRIFT_VS_EXECUTION):
            return "P1"
        if verdict == VERDICT_AT_RISK:
            return "P2"
        return "P2"

    @staticmethod
    def _suggest_action(verdict: str, rb: RunbookRecord) -> str:
        if verdict == VERDICT_ARCHIVED:
            return "No action; archived."
        if verdict == VERDICT_FRESH:
            return "Keep on normal review cadence."
        if verdict == VERDICT_ORPHANED_OWNER:
            return "Assign a named owner from the safety on-call rotation."
        if verdict == VERDICT_NEVER_EXECUTED:
            return "Schedule first tabletop drill within current sprint."
        if verdict == VERDICT_DRILL_OVERDUE:
            return "Add to the next quarterly drill calendar."
        if verdict == VERDICT_STALE_CONTENT:
            return "Refresh content; verify referenced systems and contacts."
        if verdict == VERDICT_DRIFT_VS_EXECUTION:
            return "Capture lessons from recent executions into the runbook."
        if verdict == VERDICT_AT_RISK:
            return "Light-touch content review before the stale threshold trips."
        if verdict == VERDICT_INSUFFICIENT_DATA:
            return "Backfill last_updated_at / owner so the audit is meaningful."
        return "Investigate."


# ── Serialization helpers ────────────────────────────────────────────


def _serialize(obj) -> Dict:
    if isinstance(obj, FreshnessReport):
        return {
            "generated_at": obj.generated_at.isoformat(),
            "risk_appetite": obj.risk_appetite,
            "thresholds": obj.thresholds,
            "portfolio": asdict(obj.portfolio),
            "findings": [_finding_to_dict(f) for f in obj.findings],
            "insights": list(obj.insights),
            "playbook": [asdict(a) for a in obj.playbook],
        }
    if isinstance(obj, FreshnessFinding):
        return _finding_to_dict(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()  # type: ignore[return-value]
    raise TypeError(f"Cannot serialize {type(obj)!r}")


def _finding_to_dict(f: FreshnessFinding) -> Dict:
    d = asdict(f)
    return d


# ── CLI ──────────────────────────────────────────────────────────────


def _demo_payload() -> FreshnessInput:
    now = datetime(2026, 5, 22, tzinfo=timezone.utc)
    return FreshnessInput(
        runbooks=[
            RunbookRecord(
                id="rb-self-replication",
                title="Self-replication containment",
                severity="critical",
                owner="safety-oncall",
                last_updated_at=now - timedelta(days=400),
                last_executed_at=now - timedelta(days=10),
                last_drill_at=now - timedelta(days=300),
                execution_count=3,
            ),
            RunbookRecord(
                id="rb-data-exfil",
                title="Data exfiltration response",
                severity="high",
                owner="",
                last_updated_at=now - timedelta(days=20),
                last_executed_at=None,
                last_drill_at=None,
            ),
            RunbookRecord(
                id="rb-canary-failure",
                title="Canary failure rollback",
                severity="medium",
                owner="release-eng",
                last_updated_at=now - timedelta(days=15),
                last_executed_at=now - timedelta(days=10),
                last_drill_at=now - timedelta(days=45),
                execution_count=1,
            ),
            RunbookRecord(
                id="rb-deprecated",
                title="Legacy worker manual recycle",
                status="archived",
                severity="low",
                owner="legacy-team",
                last_updated_at=now - timedelta(days=900),
            ),
            RunbookRecord(
                id="rb-drift",
                title="Cross-region failover",
                severity="high",
                owner="reliability",
                last_updated_at=now - timedelta(days=120),
                last_executed_at=now - timedelta(days=20),
                last_drill_at=now - timedelta(days=50),
                execution_count=4,
            ),
            RunbookRecord(
                id="rb-at-risk",
                title="Prompt-injection containment",
                severity="medium",
                owner="ml-platform",
                last_updated_at=now - timedelta(days=160),
                last_executed_at=now - timedelta(days=40),
                last_drill_at=now - timedelta(days=40),
                execution_count=2,
            ),
        ],
        risk_appetite="balanced",
    )


def _records_from_json(data) -> List[RunbookRecord]:
    out: List[RunbookRecord] = []
    for raw in data:
        rec = RunbookRecord(
            id=str(raw["id"]),
            title=str(raw.get("title", "")),
            severity=str(raw.get("severity", "medium")),
            owner=raw.get("owner"),
            status=str(raw.get("status", "active")),
            last_updated_at=_parse_dt(raw.get("last_updated_at")),
            last_executed_at=_parse_dt(raw.get("last_executed_at")),
            last_drill_at=_parse_dt(raw.get("last_drill_at")),
            execution_count=int(raw.get("execution_count", 0)),
            tags=tuple(raw.get("tags", [])),
        )
        out.append(rec)
    return out


def _parse_dt(value) -> Optional[datetime]:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    s = str(value).replace("Z", "+00:00")
    return datetime.fromisoformat(s)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="replication.runbook_freshness_advisor",
        description="Audit incident runbook freshness.",
    )
    parser.add_argument("--demo", action="store_true", help="Use bundled demo payload.")
    parser.add_argument("--from-json", dest="from_json", help="Path to JSON list of runbooks.")
    parser.add_argument(
        "--risk",
        choices=APPETITES,
        default="balanced",
        help="Risk appetite (default balanced).",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json", "markdown"),
        default="text",
        help="Output format.",
    )
    parser.add_argument("--output", "-o", help="Write report to file instead of stdout.")
    args = parser.parse_args(argv)

    if args.demo:
        payload = _demo_payload()
    elif args.from_json:
        with open(args.from_json, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if isinstance(raw, dict) and "runbooks" in raw:
            records = _records_from_json(raw["runbooks"])
            appetite = raw.get("risk_appetite", args.risk)
        else:
            records = _records_from_json(raw)
            appetite = args.risk
        payload = FreshnessInput(runbooks=records, risk_appetite=appetite)
    else:
        parser.error("Provide --demo or --from-json")
        return 2

    if args.risk and not args.from_json:
        payload.risk_appetite = args.risk

    advisor = RunbookFreshnessAdvisor()
    report = advisor.audit(payload)

    if args.format == "json":
        out = report.to_json()
    elif args.format == "markdown":
        out = report.to_markdown()
    else:
        out = report.to_text()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(out)
        print(f"wrote {args.output}")
    else:
        print(out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
