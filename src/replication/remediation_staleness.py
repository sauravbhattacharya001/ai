"""Remediation Staleness Advisor - agentic per-action aging/idleness auditor.

Sibling to:
  - ``remediation_planner`` (synthesizes the plan)
  - ``remediation_progress`` (diffs plan vs. plan)
  - ``remediation_assignment`` (owner + load balancer)
  - ``safety_debt`` (compound-interest open-finding debt)
  - ``remediation_roi`` (cost/benefit ranker)
  - ``kill_switch_tuner`` (threshold tuner)
  - ``defense_layer_redundancy_advisor`` (defense-in-depth auditor)

This advisor answers a different question: **which open remediation actions
are rotting on the board?**  It consumes a list of in-flight actions with
their ``opened_at`` / ``last_updated_at`` / ``due_at`` / ``status`` / owner
fields and emits per-action verdicts plus a P0-first deduplicated playbook
of concrete unblocking interventions.

Per-action verdicts
~~~~~~~~~~~~~~~~~~~

ABANDONED, OVERDUE, STALE, IDLE, AT_RISK, ON_TRACK, RECENTLY_COMPLETED,
INSUFFICIENT_DATA, BLOCKED_NO_OWNER.

Cross-portfolio insights
~~~~~~~~~~~~~~~~~~~~~~~~

WIDESPREAD_STALENESS, ABANDONED_CLUSTER, OVERDUE_CRITICAL_WORK,
OWNERLESS_BACKLOG, OWNER_OVERLOAD, HEALTHY_BOARD, EMPTY_BOARD.

CLI demo::

    python -m replication.remediation_staleness --demo --format markdown

Programmatic::

    from replication.remediation_staleness import (
        RemediationStalenessAdvisor,
        StalenessInput,
        StalenessAction,
    )

    advisor = RemediationStalenessAdvisor()
    report = advisor.audit(StalenessInput(actions=[...]))
    print(report.to_markdown())
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional, Tuple


# ── Constants ─────────────────────────────────────────────────────────

SEVERITY_LEVELS: Tuple[str, ...] = ("info", "low", "medium", "high", "critical")
SEVERITY_WEIGHT: Dict[str, int] = {
    "info": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}

STATUS_OPEN = {"open", "in_progress", "blocked", "todo", "wip", "review"}
STATUS_DONE = {"done", "closed", "resolved", "completed", "fixed"}
STATUS_DROPPED = {"wontfix", "cancelled", "rejected", "duplicate"}

APPETITES: Tuple[str, ...] = ("cautious", "balanced", "aggressive")

# Multiplier on staleness_score: cautious is stricter (higher score),
# aggressive is more lenient.
APPETITE_SCORE_MULT: Dict[str, float] = {
    "cautious": 1.15,
    "balanced": 1.0,
    "aggressive": 0.85,
}

# Day thresholds (balanced defaults). Cautious tightens (× 0.7), aggressive
# loosens (× 1.4).
BASE_THRESHOLDS: Dict[str, float] = {
    "idle_days": 7.0,         # no update for > N days while open -> IDLE
    "stale_days": 21.0,       # opened > N days ago and not done -> STALE
    "abandoned_days": 60.0,   # opened > N days ago, no update for > 30d
    "at_risk_days_before_due": 3.0,  # < N days to due -> AT_RISK
    "recently_completed_days": 7.0,
}

APPETITE_THRESHOLD_MULT: Dict[str, float] = {
    "cautious": 0.70,
    "balanced": 1.00,
    "aggressive": 1.40,
}


# ── Data model ────────────────────────────────────────────────────────


@dataclass
class StalenessAction:
    """A single remediation action under review.

    All timestamps are timezone-aware UTC :class:`datetime`.
    """

    id: str
    title: str = ""
    status: str = "open"
    severity: str = "medium"
    owner: Optional[str] = None
    opened_at: Optional[datetime] = None
    last_updated_at: Optional[datetime] = None
    due_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    tags: Tuple[str, ...] = ()

    def normalized_status(self) -> str:
        s = (self.status or "").strip().lower()
        if s in STATUS_DONE:
            return "done"
        if s in STATUS_DROPPED:
            return "dropped"
        if s in STATUS_OPEN:
            return "open"
        # Unknown statuses are treated as open so they cannot hide from
        # the auditor.
        return "open"

    def normalized_severity(self) -> str:
        s = (self.severity or "medium").strip().lower()
        return s if s in SEVERITY_WEIGHT else "medium"


@dataclass
class StalenessInput:
    actions: List[StalenessAction] = field(default_factory=list)
    risk_appetite: str = "balanced"
    # Optional per-owner WIP cap; default falls back to ``default_wip_cap``.
    wip_cap_per_owner: Dict[str, int] = field(default_factory=dict)
    default_wip_cap: int = 5


@dataclass
class StalenessFinding:
    action_id: str
    verdict: str
    priority: str  # P0..P3
    staleness_score: float  # 0..100
    days_open: Optional[float]
    days_since_update: Optional[float]
    days_to_due: Optional[float]
    reasons: List[str]
    suggested_action: str
    owner: Optional[str] = None
    severity: str = "medium"


@dataclass
class StalenessPlaybookAction:
    id: str
    priority: str
    label: str
    reason: str
    owner: str
    blast_radius: int  # 1..5
    reversibility: str  # low | medium | high
    related_action_ids: List[str] = field(default_factory=list)
    suggested_value: Optional[str] = None


@dataclass
class StalenessPortfolio:
    total_actions: int
    open_actions: int
    done_recent: int
    dropped: int
    abandoned: int
    overdue: int
    stale: int
    idle: int
    at_risk: int
    on_track: int
    ownerless: int
    overloaded_owners: List[str]
    mean_staleness: float
    max_staleness: float
    grade: str  # A..F
    concentration_band: str  # HEALTHY | WATCH | DEGRADED | CRITICAL


@dataclass
class StalenessReport:
    generated_at: datetime
    risk_appetite: str
    portfolio: StalenessPortfolio
    findings: List[StalenessFinding]
    playbook: List[StalenessPlaybookAction]
    insights: List[str]

    # ── Renderers ────────────────────────────────────────────────────

    def to_json(self) -> str:
        return json.dumps(_serialize(self), sort_keys=True, indent=2, default=str)

    def to_text(self) -> str:
        lines: List[str] = []
        p = self.portfolio
        lines.append(
            f"Remediation Staleness Report — grade={p.grade} band={p.concentration_band} "
            f"open={p.open_actions} stale={p.stale} overdue={p.overdue} "
            f"abandoned={p.abandoned} appetite={self.risk_appetite}"
        )
        lines.append(
            f"mean_staleness={p.mean_staleness:.1f} max_staleness={p.max_staleness:.1f} "
            f"ownerless={p.ownerless} overloaded_owners={','.join(p.overloaded_owners) or '-'}"
        )
        lines.append("")
        lines.append("Findings:")
        for f in self.findings:
            lines.append(
                f"  [{f.priority}] {f.action_id} {f.verdict} score={f.staleness_score:.1f} "
                f"owner={f.owner or '-'} sev={f.severity} -> {f.suggested_action}"
            )
            if f.reasons:
                lines.append(f"      reasons: {', '.join(f.reasons)}")
        lines.append("")
        lines.append("Playbook:")
        for a in self.playbook:
            lines.append(
                f"  [{a.priority}] {a.id} owner={a.owner} blast={a.blast_radius} "
                f"rev={a.reversibility}: {a.label}"
            )
            lines.append(f"      reason: {a.reason}")
        lines.append("")
        lines.append("Insights: " + (", ".join(self.insights) if self.insights else "-"))
        return "\n".join(lines)

    def to_markdown(self) -> str:
        p = self.portfolio
        lines: List[str] = []
        lines.append("# Remediation Staleness Report")
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| grade | {p.grade} |")
        lines.append(f"| band | {p.concentration_band} |")
        lines.append(f"| risk_appetite | {self.risk_appetite} |")
        lines.append(f"| total_actions | {p.total_actions} |")
        lines.append(f"| open_actions | {p.open_actions} |")
        lines.append(f"| stale | {p.stale} |")
        lines.append(f"| idle | {p.idle} |")
        lines.append(f"| overdue | {p.overdue} |")
        lines.append(f"| at_risk | {p.at_risk} |")
        lines.append(f"| abandoned | {p.abandoned} |")
        lines.append(f"| ownerless | {p.ownerless} |")
        lines.append(f"| mean_staleness | {p.mean_staleness:.1f} |")
        lines.append(f"| max_staleness | {p.max_staleness:.1f} |")
        lines.append(f"| overloaded_owners | {', '.join(p.overloaded_owners) or '-'} |")
        lines.append("")
        lines.append("## Findings")
        lines.append("")
        lines.append("| Priority | ID | Verdict | Score | Owner | Sev | Days open | Days since update | Days to due | Suggested |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for f in self.findings:
            lines.append(
                f"| {f.priority} | {f.action_id} | {f.verdict} | {f.staleness_score:.1f} | "
                f"{f.owner or '-'} | {f.severity} | "
                f"{_fmt_days(f.days_open)} | {_fmt_days(f.days_since_update)} | "
                f"{_fmt_days(f.days_to_due)} | {f.suggested_action} |"
            )
        lines.append("")
        lines.append("## Playbook")
        lines.append("")
        lines.append("| Priority | ID | Owner | Blast | Reversibility | Label | Reason |")
        lines.append("|---|---|---|---|---|---|---|")
        for a in self.playbook:
            lines.append(
                f"| {a.priority} | {a.id} | {a.owner} | {a.blast_radius} | "
                f"{a.reversibility} | {a.label} | {a.reason} |"
            )
        lines.append("")
        lines.append("## Insights")
        lines.append("")
        if self.insights:
            for ins in self.insights:
                lines.append(f"- {ins}")
        else:
            lines.append("- (none)")
        return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────


def _fmt_days(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.1f}"


def _serialize(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (StalenessReport, StalenessPortfolio, StalenessFinding,
                        StalenessPlaybookAction)):
        return _serialize(asdict(obj))
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    return obj


def _to_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _days_between(later: datetime, earlier: datetime) -> float:
    return (later - earlier).total_seconds() / 86400.0


def _grade_from_score(mean_score: float, max_score: float,
                      overdue: int, abandoned: int) -> Tuple[str, str]:
    # Concentration band uses mean_score; grade may be force-dropped by
    # critical conditions.
    if mean_score < 10 and max_score < 25:
        band = "HEALTHY"
    elif mean_score < 25:
        band = "WATCH"
    elif mean_score < 50:
        band = "DEGRADED"
    else:
        band = "CRITICAL"

    if abandoned >= 3 or (overdue >= 2 and mean_score >= 50):
        return "F", band
    if overdue >= 1 or mean_score >= 50 or max_score >= 80:
        return "D", band
    if mean_score >= 30 or max_score >= 60:
        return "C", band
    if mean_score >= 15 or max_score >= 40:
        return "B", band
    return "A", band


# ── Advisor ───────────────────────────────────────────────────────────


class RemediationStalenessAdvisor:
    """Agentic per-action aging auditor for an open remediation board."""

    def __init__(self, now_fn: Optional[Callable[[], datetime]] = None):
        self._now_fn: Callable[[], datetime] = now_fn or (
            lambda: datetime.now(timezone.utc)
        )

    # Public API ──────────────────────────────────────────────────────

    def audit(self, payload: StalenessInput) -> StalenessReport:
        # Defensive deep copy so callers can keep mutating their inputs.
        payload = copy.deepcopy(payload)

        appetite = payload.risk_appetite if payload.risk_appetite in APPETITES else "balanced"
        score_mult = APPETITE_SCORE_MULT[appetite]
        thr_mult = APPETITE_THRESHOLD_MULT[appetite]
        thresholds = {k: v * thr_mult for k, v in BASE_THRESHOLDS.items()}

        now = self._now_fn()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        findings: List[StalenessFinding] = []
        owner_open_counts: Dict[str, int] = {}

        # Determine recent completions / dropped for the portfolio summary
        # but exclude them from per-action findings (handled separately).
        done_recent = 0
        dropped = 0

        for a in payload.actions:
            status = a.normalized_status()
            opened = _to_utc(a.opened_at)
            updated = _to_utc(a.last_updated_at) or opened
            due = _to_utc(a.due_at)
            closed = _to_utc(a.closed_at)

            days_open = _days_between(now, opened) if opened else None
            days_since_update = (
                _days_between(now, updated) if updated else None
            )
            days_to_due = _days_between(due, now) if due else None

            if status == "dropped":
                dropped += 1
                continue

            if status == "done":
                if closed is not None and _days_between(now, closed) <= thresholds["recently_completed_days"]:
                    done_recent += 1
                    findings.append(self._build_finding(
                        a, "RECENTLY_COMPLETED", "P3", 0.0,
                        days_open, days_since_update, days_to_due,
                        reasons=["closed_within_recent_window"],
                        suggested="Verify fix landed and update runbook",
                    ))
                # older completions are simply absent from findings
                continue

            # Track owner load for open actions only.
            if a.owner:
                owner_open_counts[a.owner] = owner_open_counts.get(a.owner, 0) + 1

            verdict, reasons, base_score, suggested = self._classify_open(
                a, days_open, days_since_update, days_to_due, thresholds
            )
            staleness_score = min(100.0, max(0.0, base_score * score_mult))
            priority = self._priority_for(verdict, staleness_score, a.normalized_severity())
            findings.append(self._build_finding(
                a, verdict, priority, staleness_score,
                days_open, days_since_update, days_to_due,
                reasons=reasons, suggested=suggested,
            ))

        # ── Aggregate the portfolio ──────────────────────────────────
        open_findings = [
            f for f in findings if f.verdict not in ("RECENTLY_COMPLETED",)
        ]

        overloaded: List[str] = []
        for owner, count in sorted(owner_open_counts.items()):
            cap = payload.wip_cap_per_owner.get(owner, payload.default_wip_cap)
            if count > cap:
                overloaded.append(owner)

        verdict_counts = {v: 0 for v in (
            "ABANDONED", "OVERDUE", "STALE", "IDLE", "AT_RISK", "ON_TRACK",
            "BLOCKED_NO_OWNER", "INSUFFICIENT_DATA",
        )}
        for f in open_findings:
            verdict_counts[f.verdict] = verdict_counts.get(f.verdict, 0) + 1
        ownerless = sum(1 for f in open_findings if not f.owner)

        scores = [f.staleness_score for f in open_findings] or [0.0]
        mean_score = sum(scores) / len(scores)
        max_score = max(scores)

        grade, band = _grade_from_score(
            mean_score, max_score,
            overdue=verdict_counts["OVERDUE"],
            abandoned=verdict_counts["ABANDONED"],
        )

        portfolio = StalenessPortfolio(
            total_actions=len(payload.actions),
            open_actions=len(open_findings),
            done_recent=done_recent,
            dropped=dropped,
            abandoned=verdict_counts["ABANDONED"],
            overdue=verdict_counts["OVERDUE"],
            stale=verdict_counts["STALE"],
            idle=verdict_counts["IDLE"],
            at_risk=verdict_counts["AT_RISK"],
            on_track=verdict_counts["ON_TRACK"],
            ownerless=ownerless,
            overloaded_owners=overloaded,
            mean_staleness=mean_score,
            max_staleness=max_score,
            grade=grade,
            concentration_band=band,
        )

        # ── Insights ─────────────────────────────────────────────────
        insights = self._build_insights(portfolio, open_findings)

        # ── Playbook ─────────────────────────────────────────────────
        playbook = self._build_playbook(portfolio, open_findings, appetite)

        # Order findings: priority asc then score desc then id asc.
        priority_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        findings.sort(key=lambda f: (
            priority_rank.get(f.priority, 9),
            -f.staleness_score,
            f.action_id,
        ))

        return StalenessReport(
            generated_at=now,
            risk_appetite=appetite,
            portfolio=portfolio,
            findings=findings,
            playbook=playbook,
            insights=insights,
        )

    # Internals ───────────────────────────────────────────────────────

    def _build_finding(
        self,
        a: StalenessAction,
        verdict: str,
        priority: str,
        score: float,
        days_open: Optional[float],
        days_since_update: Optional[float],
        days_to_due: Optional[float],
        reasons: List[str],
        suggested: str,
    ) -> StalenessFinding:
        return StalenessFinding(
            action_id=a.id,
            verdict=verdict,
            priority=priority,
            staleness_score=round(score, 2),
            days_open=None if days_open is None else round(days_open, 2),
            days_since_update=None if days_since_update is None else round(days_since_update, 2),
            days_to_due=None if days_to_due is None else round(days_to_due, 2),
            reasons=list(reasons),
            suggested_action=suggested,
            owner=a.owner,
            severity=a.normalized_severity(),
        )

    def _classify_open(
        self,
        a: StalenessAction,
        days_open: Optional[float],
        days_since_update: Optional[float],
        days_to_due: Optional[float],
        thresholds: Dict[str, float],
    ) -> Tuple[str, List[str], float, str]:
        reasons: List[str] = []
        score = 0.0

        # Missing timestamps -> we can still classify ownerless / insufficient.
        if days_open is None and days_since_update is None and days_to_due is None:
            if not a.owner:
                return (
                    "BLOCKED_NO_OWNER",
                    ["no_timestamps", "no_owner"],
                    35.0,
                    "Assign an owner and set opened_at to start tracking",
                )
            return (
                "INSUFFICIENT_DATA",
                ["missing_timestamps"],
                15.0,
                "Backfill opened_at / last_updated_at so this can be aged",
            )

        sev_w = SEVERITY_WEIGHT.get(a.normalized_severity(), 2)

        # Overdue: hard deadline crossed.
        if days_to_due is not None and days_to_due < 0:
            reasons.append(f"overdue_by_{abs(days_to_due):.1f}d")
            score = 70.0 + min(20.0, abs(days_to_due) * 1.5) + sev_w * 2
            if not a.owner:
                reasons.append("no_owner")
                score += 5
            return ("OVERDUE", reasons, score,
                    "Escalate to owner or reassign immediately")

        # Abandoned: very old and untouched for a long stretch.
        if (days_open is not None and days_open >= thresholds["abandoned_days"]
                and days_since_update is not None
                and days_since_update >= max(30.0 * APPETITE_THRESHOLD_MULT.get("balanced", 1.0),
                                             thresholds["idle_days"] * 3)):
            reasons.append(f"open_{days_open:.0f}d_no_update_{days_since_update:.0f}d")
            score = 75.0 + min(20.0, days_since_update / 5.0) + sev_w
            if not a.owner:
                reasons.append("no_owner")
                score += 5
            return ("ABANDONED", reasons, score,
                    "Decide: revive with owner, escalate, or formally drop")

        # No owner at all on an open item.
        if not a.owner:
            reasons.append("no_owner")
            base = 40.0 + sev_w * 5
            if days_open is not None and days_open >= thresholds["idle_days"]:
                reasons.append(f"open_{days_open:.0f}d")
                base += min(15.0, days_open / 3.0)
            return ("BLOCKED_NO_OWNER", reasons, base,
                    "Assign an owner before this rots further")

        # At risk: deadline approaching.
        if days_to_due is not None and 0 <= days_to_due <= thresholds["at_risk_days_before_due"]:
            reasons.append(f"due_in_{days_to_due:.1f}d")
            score = 45.0 + sev_w * 5
            if days_since_update is not None and days_since_update >= thresholds["idle_days"]:
                reasons.append(f"idle_{days_since_update:.0f}d")
                score += 15
            return ("AT_RISK", reasons, score,
                    "Confirm progress today; surface blockers")

        # Stale: opened long ago, still not done.
        if days_open is not None and days_open >= thresholds["stale_days"]:
            reasons.append(f"open_{days_open:.0f}d")
            score = 35.0 + min(25.0, days_open / 3.0) + sev_w * 2
            if days_since_update is not None and days_since_update >= thresholds["idle_days"]:
                reasons.append(f"idle_{days_since_update:.0f}d")
                score += 10
            return ("STALE", reasons, score,
                    "Re-scope, split, or hand off to clear it")

        # Idle: open but no recent activity.
        if (days_since_update is not None
                and days_since_update >= thresholds["idle_days"]):
            reasons.append(f"idle_{days_since_update:.0f}d")
            score = 25.0 + min(20.0, days_since_update / 2.0) + sev_w
            return ("IDLE", reasons, score,
                    "Owner check-in: bump status or unblock")

        # Healthy.
        score = max(0.0, 5.0 + sev_w)
        return ("ON_TRACK", reasons or ["recent_activity"], score,
                "Keep moving; no intervention needed")

    def _priority_for(self, verdict: str, score: float, severity: str) -> str:
        sev_w = SEVERITY_WEIGHT.get(severity, 2)
        if verdict in ("ABANDONED", "OVERDUE"):
            return "P0"
        if verdict == "BLOCKED_NO_OWNER" and sev_w >= 3:
            return "P0"
        if verdict in ("AT_RISK", "BLOCKED_NO_OWNER"):
            return "P1"
        if verdict == "STALE":
            return "P1" if sev_w >= 3 or score >= 60 else "P2"
        if verdict == "IDLE":
            return "P2" if sev_w >= 3 else "P3"
        if verdict == "INSUFFICIENT_DATA":
            return "P2"
        return "P3"

    def _build_insights(
        self,
        portfolio: StalenessPortfolio,
        open_findings: List[StalenessFinding],
    ) -> List[str]:
        out: List[str] = []
        if portfolio.total_actions == 0:
            return ["EMPTY_BOARD"]
        if portfolio.open_actions == 0:
            out.append("HEALTHY_BOARD")
            return out

        share_stale = (
            (portfolio.stale + portfolio.abandoned + portfolio.overdue)
            / max(1, portfolio.open_actions)
        )
        if share_stale >= 0.40:
            out.append(f"WIDESPREAD_STALENESS:{share_stale * 100:.0f}pct_of_open")
        if portfolio.abandoned >= 2:
            out.append(f"ABANDONED_CLUSTER:{portfolio.abandoned}")
        if any(
            f.verdict == "OVERDUE" and f.severity in ("high", "critical")
            for f in open_findings
        ):
            out.append("OVERDUE_CRITICAL_WORK")
        if portfolio.ownerless >= 2:
            out.append(f"OWNERLESS_BACKLOG:{portfolio.ownerless}")
        if portfolio.overloaded_owners:
            out.append(
                "OWNER_OVERLOAD:" + ",".join(portfolio.overloaded_owners)
            )
        if not out:
            out.append("HEALTHY_BOARD")
        return out

    def _build_playbook(
        self,
        portfolio: StalenessPortfolio,
        open_findings: List[StalenessFinding],
        appetite: str,
    ) -> List[StalenessPlaybookAction]:
        actions: List[StalenessPlaybookAction] = []

        overdue_ids = [f.action_id for f in open_findings if f.verdict == "OVERDUE"]
        abandoned_ids = [f.action_id for f in open_findings if f.verdict == "ABANDONED"]
        ownerless_ids = [f.action_id for f in open_findings if f.verdict == "BLOCKED_NO_OWNER"]
        at_risk_ids = [f.action_id for f in open_findings if f.verdict == "AT_RISK"]
        stale_ids = [f.action_id for f in open_findings if f.verdict == "STALE"]
        idle_ids = [f.action_id for f in open_findings if f.verdict == "IDLE"]
        insufficient_ids = [
            f.action_id for f in open_findings if f.verdict == "INSUFFICIENT_DATA"
        ]

        if overdue_ids:
            actions.append(StalenessPlaybookAction(
                id="ESCALATE_OVERDUE_ACTIONS",
                priority="P0",
                label=f"Escalate {len(overdue_ids)} overdue remediation action(s)",
                reason="Hard deadlines already crossed; risk window is open.",
                owner="program_manager",
                blast_radius=3,
                reversibility="high",
                related_action_ids=sorted(overdue_ids),
            ))
        if abandoned_ids:
            actions.append(StalenessPlaybookAction(
                id="TRIAGE_ABANDONED_ACTIONS",
                priority="P0",
                label=f"Triage {len(abandoned_ids)} abandoned action(s): revive, reassign, or drop",
                reason="Items have been untouched long enough that staleness debt is compounding.",
                owner="safety_lead",
                blast_radius=2,
                reversibility="high",
                related_action_ids=sorted(abandoned_ids),
            ))
        if ownerless_ids:
            sev_high = any(
                f.severity in ("high", "critical")
                for f in open_findings
                if f.action_id in ownerless_ids
            )
            actions.append(StalenessPlaybookAction(
                id="ASSIGN_OWNERS",
                priority="P0" if sev_high else "P1",
                label=f"Assign owners to {len(ownerless_ids)} unassigned action(s)",
                reason="Without an owner these cannot make progress.",
                owner="program_manager",
                blast_radius=2,
                reversibility="high",
                related_action_ids=sorted(ownerless_ids),
            ))
        if at_risk_ids:
            actions.append(StalenessPlaybookAction(
                id="UNBLOCK_AT_RISK",
                priority="P1",
                label=f"Pull {len(at_risk_ids)} at-risk action(s) into today's standup",
                reason="Deadline is within the at-risk window; blockers must surface now.",
                owner="program_manager",
                blast_radius=2,
                reversibility="high",
                related_action_ids=sorted(at_risk_ids),
            ))
        if stale_ids:
            actions.append(StalenessPlaybookAction(
                id="RESCOPE_STALE_ACTIONS",
                priority="P1" if any(
                    f.severity in ("high", "critical")
                    for f in open_findings if f.action_id in stale_ids
                ) else "P2",
                label=f"Re-scope or split {len(stale_ids)} stale action(s)",
                reason="Open too long without closure; size is likely the bottleneck.",
                owner="safety_lead",
                blast_radius=2,
                reversibility="high",
                related_action_ids=sorted(stale_ids),
            ))
        if portfolio.overloaded_owners:
            actions.append(StalenessPlaybookAction(
                id="REBALANCE_OWNER_LOAD",
                priority="P1",
                label=f"Rebalance work for overloaded owners ({', '.join(portfolio.overloaded_owners)})",
                reason="WIP cap exceeded; downstream slippage is likely.",
                owner="program_manager",
                blast_radius=3,
                reversibility="high",
                related_action_ids=[],
                suggested_value=",".join(portfolio.overloaded_owners),
            ))
        if idle_ids:
            actions.append(StalenessPlaybookAction(
                id="CHECKIN_IDLE_ACTIONS",
                priority="P2",
                label=f"Status check-in on {len(idle_ids)} idle action(s)",
                reason="No recent updates; cheap nudge prevents drift to STALE.",
                owner="program_manager",
                blast_radius=1,
                reversibility="high",
                related_action_ids=sorted(idle_ids),
            ))
        if insufficient_ids:
            actions.append(StalenessPlaybookAction(
                id="BACKFILL_TIMESTAMPS",
                priority="P2",
                label=f"Backfill timestamps on {len(insufficient_ids)} action(s)",
                reason="Cannot age work that has no opened_at / last_updated_at.",
                owner="program_manager",
                blast_radius=1,
                reversibility="high",
                related_action_ids=sorted(insufficient_ids),
            ))

        if appetite == "cautious" and portfolio.grade in ("C", "D", "F"):
            actions.append(StalenessPlaybookAction(
                id="SCHEDULE_BOARD_REVIEW",
                priority="P2",
                label="Schedule a board review to attack staleness",
                reason="Cautious appetite + degraded grade warrants a synchronous review.",
                owner="safety_lead",
                blast_radius=2,
                reversibility="high",
                related_action_ids=[],
            ))

        if not actions:
            actions.append(StalenessPlaybookAction(
                id="HEALTHY_BOARD",
                priority="P3",
                label="Board is healthy — no staleness interventions needed",
                reason="No stale, overdue, abandoned, or ownerless work detected.",
                owner="safety_lead",
                blast_radius=1,
                reversibility="high",
                related_action_ids=[],
            ))

        # Aggressive trims pure-P3 padding and lone-P2 noise when P0/P1 exist.
        if appetite == "aggressive":
            has_high = any(a.priority in ("P0", "P1") for a in actions)
            kept: List[StalenessPlaybookAction] = []
            for a in actions:
                if a.priority == "P3" and len(actions) > 1:
                    continue
                if has_high and a.priority == "P2":
                    continue
                kept.append(a)
            actions = kept or [actions[0]]

        # Deterministic order: priority asc then id asc.
        priority_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        actions.sort(key=lambda a: (priority_rank.get(a.priority, 9), a.id))
        return actions


# ── CLI ───────────────────────────────────────────────────────────────


def _demo_input(now: datetime) -> StalenessInput:
    def days_ago(d: float) -> datetime:
        return now - timedelta(days=d)

    return StalenessInput(
        risk_appetite="balanced",
        actions=[
            StalenessAction(
                id="REM-001", title="Patch policy lint warnings", status="open",
                severity="medium", owner="alice",
                opened_at=days_ago(3), last_updated_at=days_ago(1),
                due_at=now + timedelta(days=10),
            ),
            StalenessAction(
                id="REM-002", title="Rotate leaked credential", status="in_progress",
                severity="critical", owner="bob",
                opened_at=days_ago(2), last_updated_at=days_ago(2),
                due_at=now + timedelta(days=2),
            ),
            StalenessAction(
                id="REM-003", title="Tighten kill-switch threshold", status="open",
                severity="high", owner="carol",
                opened_at=days_ago(25), last_updated_at=days_ago(12),
            ),
            StalenessAction(
                id="REM-004", title="Old audit follow-up", status="open",
                severity="medium", owner=None,
                opened_at=days_ago(80), last_updated_at=days_ago(45),
            ),
            StalenessAction(
                id="REM-005", title="Add corrigibility test", status="open",
                severity="medium", owner=None,
                opened_at=days_ago(4), last_updated_at=days_ago(4),
            ),
            StalenessAction(
                id="REM-006", title="File regression report", status="open",
                severity="high", owner="alice",
                opened_at=days_ago(15), last_updated_at=days_ago(2),
                due_at=now - timedelta(days=1),
            ),
            StalenessAction(
                id="REM-007", title="Ship runbook update", status="done",
                severity="low", owner="alice",
                opened_at=days_ago(10), last_updated_at=days_ago(1),
                closed_at=days_ago(1),
            ),
            StalenessAction(
                id="REM-008", title="Closed long ago", status="done",
                severity="low", owner="bob",
                opened_at=days_ago(120), last_updated_at=days_ago(90),
                closed_at=days_ago(90),
            ),
        ],
        wip_cap_per_owner={"alice": 2},
        default_wip_cap=5,
    )


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="remediation_staleness",
        description="Agentic per-action aging/idleness auditor.",
    )
    parser.add_argument("--demo", action="store_true",
                        help="Use a built-in demo board.")
    parser.add_argument("--from-json", type=str, default=None,
                        help="Path to a JSON file with {actions:[...], "
                             "risk_appetite, wip_cap_per_owner, default_wip_cap}.")
    parser.add_argument("--risk", choices=APPETITES, default=None,
                        help="Override risk_appetite.")
    parser.add_argument("--format", choices=("text", "markdown", "md", "json"),
                        default="text")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args(argv)

    now = datetime.now(timezone.utc)
    if args.demo and args.from_json:
        parser.error("Pick --demo or --from-json, not both.")
    if not args.demo and not args.from_json:
        parser.error("One of --demo or --from-json is required.")

    if args.demo:
        payload = _demo_input(now)
    else:
        with open(args.from_json, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        payload = _payload_from_dict(raw)

    if args.risk:
        payload.risk_appetite = args.risk

    report = RemediationStalenessAdvisor(now_fn=lambda: now).audit(payload)

    fmt = args.format
    if fmt == "md":
        fmt = "markdown"
    rendered = {
        "text": report.to_text,
        "markdown": report.to_markdown,
        "json": report.to_json,
    }[fmt]()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(rendered)
    else:
        print(rendered)
    return 0


def _payload_from_dict(raw: Dict) -> StalenessInput:
    actions: List[StalenessAction] = []
    for row in raw.get("actions", []):
        actions.append(StalenessAction(
            id=row["id"],
            title=row.get("title", ""),
            status=row.get("status", "open"),
            severity=row.get("severity", "medium"),
            owner=row.get("owner"),
            opened_at=_parse_dt(row.get("opened_at")),
            last_updated_at=_parse_dt(row.get("last_updated_at")),
            due_at=_parse_dt(row.get("due_at")),
            closed_at=_parse_dt(row.get("closed_at")),
            tags=tuple(row.get("tags", []) or ()),
        ))
    return StalenessInput(
        actions=actions,
        risk_appetite=raw.get("risk_appetite", "balanced"),
        wip_cap_per_owner=dict(raw.get("wip_cap_per_owner", {})),
        default_wip_cap=int(raw.get("default_wip_cap", 5)),
    )


def _parse_dt(value) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return _to_utc(value)
    if isinstance(value, str):
        s = value
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return _to_utc(datetime.fromisoformat(s))
    raise TypeError(f"Cannot parse datetime from {type(value).__name__}: {value!r}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
