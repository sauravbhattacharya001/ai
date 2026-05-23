"""Postmortem Quality Advisor — agentic auditor for incident postmortems.

Sibling to:
  - :mod:`replication.postmortem`         (generates postmortem docs)
  - :mod:`replication.runbook_freshness_advisor` (audits runbook *library*)
  - :mod:`replication.remediation_staleness` (audits in-flight actions)
  - :mod:`replication.safety_debt`        (compounding debt model)

This module answers a different question: **which of our published
postmortems are actually good?**  A postmortem with no root cause, an
unowned action-item backlog, or blameful language is worse than no
postmortem at all — it gives leadership a false sense that the incident
is closed.

Per-postmortem verdicts
~~~~~~~~~~~~~~~~~~~~~~~

* ``ARCHIVED``              — explicitly archived; excluded from health.
* ``MISSING``               — incident has no postmortem on file.
* ``DRAFT_STALE``           — still in draft past the publish window.
* ``AWAITING_RCA``          — published but no root_cause captured.
* ``BLAMEFUL_LANGUAGE``     — contains accusatory tokens (fault/idiot/etc).
* ``THIN_ACTION_ITEMS``     — too few action items for the incident severity.
* ``ACTION_ITEMS_OVERDUE``  — at least one action item past due.
* ``NO_LESSONS_LEARNED``    — lessons_learned empty / too short.
* ``OWNERLESS``             — no owner assigned.
* ``PUBLISHED_OK``          — meets minimum quality bar.
* ``INSUFFICIENT_DATA``     — missing critical fields.

Cross-portfolio insights
~~~~~~~~~~~~~~~~~~~~~~~~

* ``MISSING_POSTMORTEMS``        — incidents without any postmortem.
* ``WIDESPREAD_RCA_GAP``         — >40 % of postmortems lack RCA.
* ``OVERDUE_ACTION_BACKLOG``     — >25 % carry overdue action items.
* ``BLAMEFUL_CULTURE_SIGNAL``    — any blameful postmortem (always cited).
* ``CRITICAL_INCIDENTS_UNDOCUMENTED`` — critical/high incidents missing PMs.
* ``HEALTHY_LIBRARY``            — majority published OK.
* ``EMPTY_LIBRARY``              — nothing supplied.

Risk appetite (``cautious`` | ``balanced`` | ``aggressive``) scales the
quality thresholds — cautious tightens (×0.7), aggressive loosens (×1.4).

CLI demo::

    python -m replication.postmortem_quality_advisor --demo --format markdown
    python -m replication.postmortem_quality_advisor --from-json pms.json --risk cautious

Programmatic::

    from datetime import datetime, timezone
    from replication.postmortem_quality_advisor import (
        PostmortemQualityAdvisor,
        QualityInput,
        PostmortemRecord,
        IncidentRecord,
        ActionItem,
    )

    advisor = PostmortemQualityAdvisor(now=lambda: datetime(2026, 5, 22, tzinfo=timezone.utc))
    report = advisor.audit(QualityInput(postmortems=[...], incidents=[...]))
    print(report.to_markdown())
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Set, Tuple

from ._helpers import (
    APPETITE_THRESHOLD_MULT,
    APPETITES,
    SEVERITY_WEIGHT,
)


# ── Constants ─────────────────────────────────────────────────────────

VERDICT_ARCHIVED = "ARCHIVED"
VERDICT_MISSING = "MISSING"
VERDICT_DRAFT_STALE = "DRAFT_STALE"
VERDICT_AWAITING_RCA = "AWAITING_RCA"
VERDICT_BLAMEFUL_LANGUAGE = "BLAMEFUL_LANGUAGE"
VERDICT_THIN_ACTION_ITEMS = "THIN_ACTION_ITEMS"
VERDICT_ACTION_ITEMS_OVERDUE = "ACTION_ITEMS_OVERDUE"
VERDICT_NO_LESSONS_LEARNED = "NO_LESSONS_LEARNED"
VERDICT_OWNERLESS = "OWNERLESS"
VERDICT_PUBLISHED_OK = "PUBLISHED_OK"
VERDICT_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

ALL_VERDICTS: Tuple[str, ...] = (
    VERDICT_ARCHIVED,
    VERDICT_MISSING,
    VERDICT_DRAFT_STALE,
    VERDICT_AWAITING_RCA,
    VERDICT_BLAMEFUL_LANGUAGE,
    VERDICT_THIN_ACTION_ITEMS,
    VERDICT_ACTION_ITEMS_OVERDUE,
    VERDICT_NO_LESSONS_LEARNED,
    VERDICT_OWNERLESS,
    VERDICT_PUBLISHED_OK,
    VERDICT_INSUFFICIENT_DATA,
)

# When a postmortem trips multiple conditions, the highest-rank wins.
_VERDICT_RANK: Dict[str, int] = {
    VERDICT_PUBLISHED_OK: 0,
    VERDICT_INSUFFICIENT_DATA: 1,
    VERDICT_NO_LESSONS_LEARNED: 2,
    VERDICT_THIN_ACTION_ITEMS: 3,
    VERDICT_ACTION_ITEMS_OVERDUE: 4,
    VERDICT_AWAITING_RCA: 5,
    VERDICT_OWNERLESS: 6,
    VERDICT_DRAFT_STALE: 7,
    VERDICT_BLAMEFUL_LANGUAGE: 8,
    VERDICT_MISSING: 9,
    VERDICT_ARCHIVED: -1,
}

STATUS_DRAFT = {"draft", "wip", "in_progress", "writing"}
STATUS_PUBLISHED = {"published", "complete", "approved", "closed", "final", ""}
STATUS_ARCHIVED = {"archived", "retired", "deprecated", "superseded"}

# Tokens that suggest blameful authorship. Tuned for low false-positives:
# we look for direct accusations rather than incidental words. Word
# boundaries enforced via the regex pattern.
BLAMEFUL_TOKENS: Tuple[str, ...] = (
    "fault of",
    "to blame",
    "blamed",
    "blame the",
    "his fault",
    "her fault",
    "their fault",
    "stupid",
    "incompetent",
    "negligent",
    "negligence",
    "should have known",
    "idiotic",
    "lazy engineer",
    "careless engineer",
)
_BLAMEFUL_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in BLAMEFUL_TOKENS) + r")\b",
    re.IGNORECASE,
)

# Balanced-appetite thresholds.
BASE_THRESHOLDS: Dict[str, float] = {
    # Draft state allowed for this many days post-incident before stale.
    "publish_window_days": 14.0,
    # No RCA in this many days after publish -> AWAITING_RCA.
    "rca_window_days": 7.0,
    # Minimum lessons_learned length (chars) to count.
    "lessons_min_chars": 80.0,
    # Severity → minimum action items expected.
    "min_actions_low": 1.0,
    "min_actions_medium": 2.0,
    "min_actions_high": 3.0,
    "min_actions_critical": 4.0,
    # Action item considered overdue if past due by N days (slip grace).
    "action_overdue_grace_days": 0.0,
}


# ── Data model ────────────────────────────────────────────────────────


@dataclass
class ActionItem:
    """A single follow-up action item on a postmortem."""
    id: str
    title: str = ""
    owner: Optional[str] = None
    status: str = "open"  # open | in_progress | done | wont_fix | cancelled
    due_date: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def is_open(self) -> bool:
        s = (self.status or "").strip().lower()
        return s in ("open", "in_progress", "wip", "todo", "")

    def is_done(self) -> bool:
        s = (self.status or "").strip().lower()
        return s in ("done", "complete", "closed", "wont_fix", "cancelled", "resolved")


@dataclass
class IncidentRecord:
    """Reference to an incident that *should* have a postmortem."""
    id: str
    title: str = ""
    severity: str = "medium"
    occurred_at: Optional[datetime] = None


@dataclass
class PostmortemRecord:
    """A single postmortem under review."""
    id: str
    incident_id: str = ""
    title: str = ""
    severity: str = "medium"
    status: str = "published"
    owner: Optional[str] = None
    occurred_at: Optional[datetime] = None
    published_at: Optional[datetime] = None
    last_updated_at: Optional[datetime] = None
    root_cause: str = ""
    lessons_learned: str = ""
    action_items: List[ActionItem] = field(default_factory=list)
    body: str = ""  # additional narrative for blameful-language scan
    tags: Tuple[str, ...] = ()

    def normalized_status(self) -> str:
        s = (self.status or "").strip().lower()
        if s in STATUS_ARCHIVED:
            return "archived"
        if s in STATUS_DRAFT:
            return "draft"
        return "published"

    def normalized_severity(self) -> str:
        s = (self.severity or "medium").strip().lower()
        return s if s in SEVERITY_WEIGHT else "medium"


@dataclass
class QualityInput:
    postmortems: List[PostmortemRecord] = field(default_factory=list)
    incidents: List[IncidentRecord] = field(default_factory=list)
    risk_appetite: str = "balanced"


@dataclass
class QualityFinding:
    postmortem_id: str
    incident_id: str
    title: str
    verdict: str
    priority: str  # P0..P3
    quality_score: float  # 0..100 (higher = better)
    reasons: List[str]
    suggested_action: str
    owner: Optional[str] = None
    severity: str = "medium"
    open_action_count: int = 0
    overdue_action_count: int = 0


@dataclass
class QualityPortfolio:
    total: int
    active: int
    archived: int
    missing: int
    published_ok: int
    draft_stale: int
    awaiting_rca: int
    blameful: int
    thin_actions: int
    overdue_actions: int
    no_lessons: int
    ownerless: int
    insufficient_data: int
    mean_quality: float
    grade: str  # A..F


@dataclass
class QualityPlaybookAction:
    priority: str
    label: str
    reason: str
    postmortem_ids: List[str]
    suggested_value: Optional[str] = None


@dataclass
class QualityReport:
    generated_at: datetime
    risk_appetite: str
    thresholds: Dict[str, float]
    portfolio: QualityPortfolio
    findings: List[QualityFinding]
    insights: List[str]
    playbook: List[QualityPlaybookAction]

    def to_dict(self) -> Dict:
        return _serialize(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2, default=str)

    def to_text(self) -> str:
        p = self.portfolio
        lines: List[str] = []
        lines.append(
            f"Postmortem Quality — grade={p.grade} appetite={self.risk_appetite} "
            f"active={p.active} ok={p.published_ok} missing={p.missing} "
            f"awaiting_rca={p.awaiting_rca} overdue_actions={p.overdue_actions} "
            f"blameful={p.blameful} mean_quality={p.mean_quality:.1f}"
        )
        lines.append("")
        lines.append("Findings:")
        for f in self.findings:
            lines.append(
                f"  [{f.priority}] {f.postmortem_id or '(none)'} "
                f"incident={f.incident_id or '-'} {f.verdict} "
                f"score={f.quality_score:.0f} sev={f.severity} "
                f"owner={f.owner or '-'} overdue_actions={f.overdue_action_count}"
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
                    f"({len(a.postmortem_ids)} doc(s)) — {a.reason}"
                )
        return "\n".join(lines)

    def to_markdown(self) -> str:
        p = self.portfolio
        lines: List[str] = []
        lines.append("# Postmortem Quality Report")
        lines.append("")
        lines.append(f"- generated_at: `{self.generated_at.isoformat()}`")
        lines.append(f"- risk_appetite: **{self.risk_appetite}**")
        lines.append(
            f"- portfolio: grade **{p.grade}** | "
            f"active={p.active}/{p.total} ok={p.published_ok} "
            f"missing={p.missing} awaiting_rca={p.awaiting_rca} "
            f"blameful={p.blameful} overdue_actions={p.overdue_actions} "
            f"ownerless={p.ownerless} mean_quality={p.mean_quality:.1f}"
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
        lines.append("| Priority | Postmortem | Incident | Verdict | Score | Owner | Sev | Overdue AI |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for f in self.findings:
            lines.append(
                f"| {f.priority} | `{f.postmortem_id or '-'}` {f.title} | "
                f"`{f.incident_id or '-'}` | {f.verdict} | "
                f"{f.quality_score:.0f} | {f.owner or '-'} | {f.severity} | "
                f"{f.overdue_action_count} |"
            )
        if self.playbook:
            lines.append("")
            lines.append("## Playbook")
            lines.append("")
            for a in self.playbook:
                ids = ", ".join(f"`{x}`" for x in a.postmortem_ids[:5])
                more = "" if len(a.postmortem_ids) <= 5 else f" (+{len(a.postmortem_ids)-5} more)"
                val = "" if not a.suggested_value else f" — _{a.suggested_value}_"
                lines.append(
                    f"- **[{a.priority}] {a.label}** — {a.reason}{val}\n"
                    f"  - postmortems: {ids}{more}"
                )
        return "\n".join(lines)


# ── Advisor ───────────────────────────────────────────────────────────


class PostmortemQualityAdvisor:
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

    def audit(self, payload: QualityInput) -> QualityReport:
        appetite = payload.risk_appetite if payload.risk_appetite in APPETITES else "balanced"
        thresholds = self._scaled_thresholds(appetite)
        now = self._ensure_utc(self._now())

        # Index postmortems by incident_id (last writer wins for dup ids).
        pm_by_incident: Dict[str, PostmortemRecord] = {}
        for pm in payload.postmortems:
            if pm.incident_id:
                pm_by_incident[pm.incident_id] = pm

        findings: List[QualityFinding] = []

        # Synthesize MISSING findings for incidents without a postmortem.
        seen_incidents: Set[str] = set()
        for inc in payload.incidents:
            seen_incidents.add(inc.id)
            if inc.id not in pm_by_incident:
                findings.append(self._missing_finding(inc, now=now))

        for pm in payload.postmortems:
            findings.append(self._evaluate(pm, now=now, thresholds=thresholds))

        portfolio = self._summarize(findings)
        insights = self._derive_insights(findings, portfolio)
        playbook = self._build_playbook(findings)

        return QualityReport(
            generated_at=now,
            risk_appetite=appetite,
            thresholds=thresholds,
            portfolio=portfolio,
            findings=findings,
            insights=insights,
            playbook=playbook,
        )

    # ── Per-postmortem evaluation ───────────────────────────────────

    def _missing_finding(
        self, inc: IncidentRecord, *, now: datetime,
    ) -> QualityFinding:
        sev = (inc.severity or "medium").strip().lower()
        if sev not in SEVERITY_WEIGHT:
            sev = "medium"
        return QualityFinding(
            postmortem_id="",
            incident_id=inc.id,
            title=inc.title or f"Incident {inc.id}",
            verdict=VERDICT_MISSING,
            priority="P0" if sev in ("critical", "high") else "P1",
            quality_score=0.0,
            reasons=[
                f"No postmortem on file for incident {inc.id} (severity={sev})."
            ],
            suggested_action="File a postmortem; assign owner from on-call rotation.",
            owner=None,
            severity=sev,
            open_action_count=0,
            overdue_action_count=0,
        )

    def _evaluate(
        self,
        pm: PostmortemRecord,
        *,
        now: datetime,
        thresholds: Dict[str, float],
    ) -> QualityFinding:
        status = pm.normalized_status()
        severity = pm.normalized_severity()
        open_actions = sum(1 for a in pm.action_items if a.is_open())
        overdue_actions = sum(
            1
            for a in pm.action_items
            if a.is_open()
            and a.due_date is not None
            and self._days_between(a.due_date, now) is not None
            and (now - self._ensure_utc(a.due_date)).total_seconds() / 86400.0
            > thresholds["action_overdue_grace_days"]
        )

        if status == "archived":
            return QualityFinding(
                postmortem_id=pm.id,
                incident_id=pm.incident_id,
                title=pm.title,
                verdict=VERDICT_ARCHIVED,
                priority="P3",
                quality_score=0.0,
                reasons=["Postmortem is archived; excluded from health metrics."],
                suggested_action="No action; archived.",
                owner=pm.owner,
                severity=severity,
                open_action_count=open_actions,
                overdue_action_count=overdue_actions,
            )

        reasons: List[str] = []
        candidates: List[str] = []

        # Insufficient data: no occurred_at or no published_at when not draft.
        if pm.occurred_at is None and pm.published_at is None:
            reasons.append("Missing both occurred_at and published_at — cannot audit.")
            candidates.append(VERDICT_INSUFFICIENT_DATA)

        # Ownerless.
        if not (pm.owner or "").strip():
            reasons.append("No owner assigned.")
            candidates.append(VERDICT_OWNERLESS)

        # Draft stale: still draft past publish_window_days since incident.
        if status == "draft":
            days_since_incident = self._days_between(pm.occurred_at, now)
            if (
                days_since_incident is not None
                and days_since_incident > thresholds["publish_window_days"]
            ):
                reasons.append(
                    f"Draft for {days_since_incident:.0f}d (publish window "
                    f"{thresholds['publish_window_days']:.0f}d)."
                )
                candidates.append(VERDICT_DRAFT_STALE)

        # Awaiting RCA: published, but no root_cause filled in past window.
        if status == "published":
            days_since_publish = self._days_between(pm.published_at, now)
            if (
                not (pm.root_cause or "").strip()
                and days_since_publish is not None
                and days_since_publish > thresholds["rca_window_days"]
            ):
                reasons.append(
                    f"Published {days_since_publish:.0f}d ago without a root "
                    f"cause (window {thresholds['rca_window_days']:.0f}d)."
                )
                candidates.append(VERDICT_AWAITING_RCA)

        # Blameful language scan over body + lessons_learned + root_cause.
        scan_text = "\n".join((pm.body or "", pm.lessons_learned or "", pm.root_cause or ""))
        if scan_text and _BLAMEFUL_RE.search(scan_text):
            matches = sorted({m.group(0).lower() for m in _BLAMEFUL_RE.finditer(scan_text)})
            reasons.append(
                "Blameful language detected: " + ", ".join(matches[:3]) +
                (" (+more)" if len(matches) > 3 else "")
            )
            candidates.append(VERDICT_BLAMEFUL_LANGUAGE)

        # Action items count vs severity.
        min_actions = self._min_actions_for(severity, thresholds)
        if status == "published" and len(pm.action_items) < min_actions:
            reasons.append(
                f"Only {len(pm.action_items)} action item(s) for severity "
                f"{severity} (expected ≥{int(min_actions)})."
            )
            candidates.append(VERDICT_THIN_ACTION_ITEMS)

        # Overdue action items.
        if overdue_actions > 0:
            reasons.append(
                f"{overdue_actions} action item(s) past due."
            )
            candidates.append(VERDICT_ACTION_ITEMS_OVERDUE)

        # Lessons learned missing/too short.
        lessons_len = len((pm.lessons_learned or "").strip())
        if status == "published" and lessons_len < thresholds["lessons_min_chars"]:
            reasons.append(
                f"Lessons learned only {lessons_len} chars (expected "
                f"≥{int(thresholds['lessons_min_chars'])})."
            )
            candidates.append(VERDICT_NO_LESSONS_LEARNED)

        if not candidates:
            reasons.append("Owned, RCA captured, action items sized, lessons recorded.")
            verdict = VERDICT_PUBLISHED_OK
        else:
            verdict = max(candidates, key=_VERDICT_RANK.__getitem__)

        score = self._quality_score(
            verdict=verdict,
            severity=severity,
            overdue_actions=overdue_actions,
            candidates=candidates,
        )
        priority = self._priority_for(verdict, severity)
        suggested = self._suggest_action(verdict, pm)

        return QualityFinding(
            postmortem_id=pm.id,
            incident_id=pm.incident_id,
            title=pm.title,
            verdict=verdict,
            priority=priority,
            quality_score=score,
            reasons=reasons,
            suggested_action=suggested,
            owner=pm.owner,
            severity=severity,
            open_action_count=open_actions,
            overdue_action_count=overdue_actions,
        )

    # ── Portfolio + insights ────────────────────────────────────────

    @staticmethod
    def _summarize(findings: List[QualityFinding]) -> QualityPortfolio:
        counts: Dict[str, int] = {v: 0 for v in ALL_VERDICTS}
        for f in findings:
            counts[f.verdict] = counts.get(f.verdict, 0) + 1
        total = len(findings)
        archived = counts[VERDICT_ARCHIVED]
        active = total - archived
        active_findings = [f for f in findings if f.verdict != VERDICT_ARCHIVED]
        mean_quality = (
            sum(f.quality_score for f in active_findings) / len(active_findings)
            if active_findings else 100.0
        )

        if active == 0:
            grade = "A"
        elif mean_quality >= 85:
            grade = "A"
        elif mean_quality >= 70:
            grade = "B"
        elif mean_quality >= 55:
            grade = "C"
        elif mean_quality >= 40:
            grade = "D"
        else:
            grade = "F"

        # Any blameful or any critical MISSING forces D-floor.
        if counts[VERDICT_BLAMEFUL_LANGUAGE] > 0 and grade in ("A", "B"):
            grade = "C"
        critical_missing = any(
            f.verdict == VERDICT_MISSING and f.severity in ("critical", "high")
            for f in findings
        )
        if critical_missing and grade in ("A", "B"):
            grade = "C"

        return QualityPortfolio(
            total=total,
            active=active,
            archived=archived,
            missing=counts[VERDICT_MISSING],
            published_ok=counts[VERDICT_PUBLISHED_OK],
            draft_stale=counts[VERDICT_DRAFT_STALE],
            awaiting_rca=counts[VERDICT_AWAITING_RCA],
            blameful=counts[VERDICT_BLAMEFUL_LANGUAGE],
            thin_actions=counts[VERDICT_THIN_ACTION_ITEMS],
            overdue_actions=counts[VERDICT_ACTION_ITEMS_OVERDUE],
            no_lessons=counts[VERDICT_NO_LESSONS_LEARNED],
            ownerless=counts[VERDICT_OWNERLESS],
            insufficient_data=counts[VERDICT_INSUFFICIENT_DATA],
            mean_quality=round(mean_quality, 2),
            grade=grade,
        )

    @staticmethod
    def _derive_insights(
        findings: List[QualityFinding],
        portfolio: QualityPortfolio,
    ) -> List[str]:
        if portfolio.total == 0:
            return ["EMPTY_LIBRARY — supply at least one postmortem or incident."]
        insights: List[str] = []
        active = max(portfolio.active, 1)

        if portfolio.missing > 0:
            insights.append(
                f"MISSING_POSTMORTEMS — {portfolio.missing} incident(s) have "
                f"no postmortem on file."
            )
        if portfolio.awaiting_rca / active > 0.4:
            insights.append(
                f"WIDESPREAD_RCA_GAP — {portfolio.awaiting_rca}/{active} "
                f"postmortems lack a root cause."
            )
        if portfolio.overdue_actions / active > 0.25:
            insights.append(
                f"OVERDUE_ACTION_BACKLOG — {portfolio.overdue_actions}/{active} "
                f"postmortems carry past-due action items."
            )
        if portfolio.blameful > 0:
            insights.append(
                f"BLAMEFUL_CULTURE_SIGNAL — {portfolio.blameful} postmortem(s) "
                f"contain blameful language; coach authors on blameless tone."
            )
        if any(
            f.verdict == VERDICT_MISSING and f.severity in ("critical", "high")
            for f in findings
        ):
            insights.append(
                "CRITICAL_INCIDENTS_UNDOCUMENTED — high/critical-severity "
                "incidents have no postmortem; prioritize backfill."
            )
        if not insights and portfolio.published_ok / active >= 0.6:
            insights.append(
                f"HEALTHY_LIBRARY — {portfolio.published_ok}/{active} postmortems "
                f"meet the quality bar."
            )
        return insights

    # ── Playbook ────────────────────────────────────────────────────

    @staticmethod
    def _build_playbook(
        findings: List[QualityFinding],
    ) -> List[QualityPlaybookAction]:
        groups: Dict[str, List[str]] = {v: [] for v in ALL_VERDICTS}
        # Group by postmortem_id, falling back to incident_id when missing.
        for f in findings:
            key = f.postmortem_id or f.incident_id
            if key:
                groups[f.verdict].append(key)

        actions: List[QualityPlaybookAction] = []

        if groups[VERDICT_MISSING]:
            actions.append(QualityPlaybookAction(
                priority="P0",
                label="File missing postmortems",
                reason="Incidents without postmortems leave lessons uncaptured.",
                postmortem_ids=sorted(groups[VERDICT_MISSING]),
                suggested_value="Open a draft for each incident within 48h.",
            ))
        if groups[VERDICT_BLAMEFUL_LANGUAGE]:
            actions.append(QualityPlaybookAction(
                priority="P0",
                label="Rewrite blameful postmortems",
                reason="Blameful language erodes psychological safety and skews RCA.",
                postmortem_ids=sorted(groups[VERDICT_BLAMEFUL_LANGUAGE]),
                suggested_value="Reframe in systems language; remove individual blame.",
            ))
        if groups[VERDICT_DRAFT_STALE]:
            actions.append(QualityPlaybookAction(
                priority="P0",
                label="Publish stale drafts",
                reason="Drafts past the publish window block organizational learning.",
                postmortem_ids=sorted(groups[VERDICT_DRAFT_STALE]),
                suggested_value="Set a 1-week publish deadline; reassign if blocked.",
            ))
        if groups[VERDICT_OWNERLESS]:
            actions.append(QualityPlaybookAction(
                priority="P1",
                label="Assign postmortem owners",
                reason="Ownerless docs have nobody driving action-item closure.",
                postmortem_ids=sorted(groups[VERDICT_OWNERLESS]),
                suggested_value="Assign incident commander or on-call lead.",
            ))
        if groups[VERDICT_AWAITING_RCA]:
            actions.append(QualityPlaybookAction(
                priority="P1",
                label="Complete root cause analysis",
                reason="Published postmortems without RCA mislead leadership on risk.",
                postmortem_ids=sorted(groups[VERDICT_AWAITING_RCA]),
                suggested_value="Schedule RCA review within current sprint.",
            ))
        if groups[VERDICT_ACTION_ITEMS_OVERDUE]:
            actions.append(QualityPlaybookAction(
                priority="P1",
                label="Drive overdue action items",
                reason="Open follow-ups past due are how the same incident reoccurs.",
                postmortem_ids=sorted(groups[VERDICT_ACTION_ITEMS_OVERDUE]),
                suggested_value="Triage in next on-call handoff; close or re-due.",
            ))
        if groups[VERDICT_THIN_ACTION_ITEMS]:
            actions.append(QualityPlaybookAction(
                priority="P2",
                label="Expand thin action item lists",
                reason="Action items below severity bar suggest incomplete analysis.",
                postmortem_ids=sorted(groups[VERDICT_THIN_ACTION_ITEMS]),
                suggested_value="Brainstorm prevention/detection/response actions.",
            ))
        if groups[VERDICT_NO_LESSONS_LEARNED]:
            actions.append(QualityPlaybookAction(
                priority="P2",
                label="Capture lessons learned",
                reason="Postmortems without lessons leave no transferable knowledge.",
                postmortem_ids=sorted(groups[VERDICT_NO_LESSONS_LEARNED]),
                suggested_value="Add ≥3 takeaways generalizable beyond this incident.",
            ))
        if groups[VERDICT_INSUFFICIENT_DATA]:
            actions.append(QualityPlaybookAction(
                priority="P2",
                label="Backfill metadata",
                reason="Missing timestamps prevent meaningful audit.",
                postmortem_ids=sorted(groups[VERDICT_INSUFFICIENT_DATA]),
                suggested_value="Populate occurred_at / published_at / owner.",
            ))
        return actions

    # ── Helpers ─────────────────────────────────────────────────────

    def _scaled_thresholds(self, appetite: str) -> Dict[str, float]:
        mult = APPETITE_THRESHOLD_MULT.get(appetite, 1.0)
        scaled: Dict[str, float] = {}
        for k, v in self._base_thresholds.items():
            # Action-count thresholds shouldn't scale by day-mult.
            if k.startswith("min_actions_"):
                scaled[k] = v
            elif k == "lessons_min_chars":
                # Cautious wants more chars, aggressive fewer: invert mult.
                scaled[k] = round(v / mult, 4) if mult else v
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
    def _min_actions_for(severity: str, thresholds: Dict[str, float]) -> float:
        key = f"min_actions_{severity}"
        return thresholds.get(key, thresholds["min_actions_medium"])

    @staticmethod
    def _quality_score(
        *,
        verdict: str,
        severity: str,
        overdue_actions: int,
        candidates: List[str],
    ) -> float:
        base = {
            VERDICT_PUBLISHED_OK: 95.0,
            VERDICT_INSUFFICIENT_DATA: 55.0,
            VERDICT_NO_LESSONS_LEARNED: 60.0,
            VERDICT_THIN_ACTION_ITEMS: 55.0,
            VERDICT_ACTION_ITEMS_OVERDUE: 45.0,
            VERDICT_AWAITING_RCA: 35.0,
            VERDICT_OWNERLESS: 30.0,
            VERDICT_DRAFT_STALE: 25.0,
            VERDICT_BLAMEFUL_LANGUAGE: 20.0,
            VERDICT_MISSING: 0.0,
            VERDICT_ARCHIVED: 0.0,
        }.get(verdict, 50.0)

        # Each additional issue beyond the dominant one shaves a few points.
        secondary = max(0, len(set(candidates)) - 1)
        base -= min(secondary * 5.0, 20.0)
        # Each overdue action shaves a point, cap 10.
        base -= min(overdue_actions * 1.0, 10.0)

        # High/critical postmortems penalized harder so they sink in
        # mean-quality summaries.
        sev_weight = {"low": 0.0, "medium": 0.0, "high": -5.0, "critical": -10.0}
        if verdict not in (VERDICT_PUBLISHED_OK, VERDICT_ARCHIVED):
            base += sev_weight.get(severity, 0.0)

        return round(max(0.0, min(100.0, base)), 2)

    @staticmethod
    def _priority_for(verdict: str, severity: str) -> str:
        if verdict == VERDICT_ARCHIVED:
            return "P3"
        if verdict == VERDICT_PUBLISHED_OK:
            return "P3"
        if verdict == VERDICT_MISSING:
            return "P0" if severity in ("critical", "high") else "P1"
        if verdict == VERDICT_BLAMEFUL_LANGUAGE:
            return "P0"
        if verdict == VERDICT_DRAFT_STALE:
            return "P0" if severity in ("critical", "high") else "P1"
        if verdict in (VERDICT_AWAITING_RCA, VERDICT_OWNERLESS):
            return "P1"
        if verdict == VERDICT_ACTION_ITEMS_OVERDUE:
            return "P1"
        if verdict in (VERDICT_THIN_ACTION_ITEMS, VERDICT_NO_LESSONS_LEARNED):
            return "P2"
        return "P2"

    @staticmethod
    def _suggest_action(verdict: str, pm: PostmortemRecord) -> str:
        if verdict == VERDICT_ARCHIVED:
            return "No action; archived."
        if verdict == VERDICT_PUBLISHED_OK:
            return "Keep on normal review cadence."
        if verdict == VERDICT_OWNERLESS:
            return "Assign a named owner (incident commander or on-call lead)."
        if verdict == VERDICT_DRAFT_STALE:
            return "Publish within 1 week or reassign."
        if verdict == VERDICT_AWAITING_RCA:
            return "Schedule RCA review and fill root_cause section."
        if verdict == VERDICT_BLAMEFUL_LANGUAGE:
            return "Reframe in blameless / systems language."
        if verdict == VERDICT_THIN_ACTION_ITEMS:
            return "Add prevention / detection / response action items."
        if verdict == VERDICT_ACTION_ITEMS_OVERDUE:
            return "Triage open action items; close or reschedule."
        if verdict == VERDICT_NO_LESSONS_LEARNED:
            return "Capture at least three transferable takeaways."
        if verdict == VERDICT_INSUFFICIENT_DATA:
            return "Backfill occurred_at / published_at / owner."
        return "Investigate."


# ── Serialization helpers ────────────────────────────────────────────


def _serialize(obj) -> Dict:
    if isinstance(obj, QualityReport):
        return {
            "generated_at": obj.generated_at.isoformat(),
            "risk_appetite": obj.risk_appetite,
            "thresholds": obj.thresholds,
            "portfolio": asdict(obj.portfolio),
            "findings": [_finding_to_dict(f) for f in obj.findings],
            "insights": list(obj.insights),
            "playbook": [asdict(a) for a in obj.playbook],
        }
    if isinstance(obj, QualityFinding):
        return _finding_to_dict(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()  # type: ignore[return-value]
    raise TypeError(f"Cannot serialize {type(obj)!r}")


def _finding_to_dict(f: QualityFinding) -> Dict:
    return asdict(f)


# ── CLI ──────────────────────────────────────────────────────────────


def _demo_payload() -> QualityInput:
    now = datetime(2026, 5, 22, tzinfo=timezone.utc)
    return QualityInput(
        postmortems=[
            PostmortemRecord(
                id="pm-001",
                incident_id="inc-001",
                title="Self-replication containment breach",
                severity="critical",
                status="published",
                owner="safety-oncall",
                occurred_at=now - timedelta(days=30),
                published_at=now - timedelta(days=28),
                last_updated_at=now - timedelta(days=20),
                root_cause=(
                    "Kill switch latch state was not persisted across worker "
                    "restarts; new shard inherited cleared state."
                ),
                lessons_learned=(
                    "Persist safety-critical latches in durable storage. "
                    "Add explicit smoke test for restart-after-trip behavior. "
                    "Treat any kill-switch state machine as durable by default."
                ),
                action_items=[
                    ActionItem(id="ai-1", title="Persist latch state",
                               owner="safety-eng", status="done"),
                    ActionItem(id="ai-2", title="Restart smoke test",
                               owner="qa", status="in_progress",
                               due_date=now + timedelta(days=7)),
                    ActionItem(id="ai-3", title="Audit other latches",
                               owner="safety-eng", status="open",
                               due_date=now - timedelta(days=3)),
                    ActionItem(id="ai-4", title="Update runbook",
                               owner="safety-oncall", status="open",
                               due_date=now + timedelta(days=14)),
                ],
            ),
            PostmortemRecord(
                id="pm-002",
                incident_id="inc-002",
                title="Data exfil canary tripped",
                severity="high",
                status="draft",
                owner=None,
                occurred_at=now - timedelta(days=45),
                published_at=None,
                action_items=[],
                body="The operator should have known better; this is their fault.",
            ),
            PostmortemRecord(
                id="pm-003",
                incident_id="inc-003",
                title="Quarantine bypass attempt",
                severity="medium",
                status="published",
                owner="safety-oncall",
                occurred_at=now - timedelta(days=14),
                published_at=now - timedelta(days=12),
                root_cause="",  # awaiting RCA
                lessons_learned="",
                action_items=[
                    ActionItem(id="ai-q1", title="Reproduce in sandbox",
                               owner="red-team", status="open",
                               due_date=now + timedelta(days=4)),
                ],
            ),
            PostmortemRecord(
                id="pm-004",
                incident_id="inc-004",
                title="Routine rollback verification",
                severity="low",
                status="archived",
                owner="release-eng",
                occurred_at=now - timedelta(days=200),
                published_at=now - timedelta(days=195),
            ),
        ],
        incidents=[
            IncidentRecord(id="inc-001", title="Self-replication breach",
                           severity="critical",
                           occurred_at=now - timedelta(days=30)),
            IncidentRecord(id="inc-002", title="Data exfil canary",
                           severity="high",
                           occurred_at=now - timedelta(days=45)),
            IncidentRecord(id="inc-003", title="Quarantine bypass",
                           severity="medium",
                           occurred_at=now - timedelta(days=14)),
            IncidentRecord(id="inc-005", title="Sleeper agent suspected",
                           severity="critical",
                           occurred_at=now - timedelta(days=2)),
        ],
        risk_appetite="balanced",
    )


def _records_from_json(data) -> List[PostmortemRecord]:
    out: List[PostmortemRecord] = []
    for raw in data:
        actions: List[ActionItem] = []
        for a in raw.get("action_items", []) or []:
            actions.append(ActionItem(
                id=str(a["id"]),
                title=str(a.get("title", "")),
                owner=a.get("owner"),
                status=str(a.get("status", "open")),
                due_date=_parse_dt(a.get("due_date")),
                completed_at=_parse_dt(a.get("completed_at")),
            ))
        out.append(PostmortemRecord(
            id=str(raw["id"]),
            incident_id=str(raw.get("incident_id", "")),
            title=str(raw.get("title", "")),
            severity=str(raw.get("severity", "medium")),
            status=str(raw.get("status", "published")),
            owner=raw.get("owner"),
            occurred_at=_parse_dt(raw.get("occurred_at")),
            published_at=_parse_dt(raw.get("published_at")),
            last_updated_at=_parse_dt(raw.get("last_updated_at")),
            root_cause=str(raw.get("root_cause", "")),
            lessons_learned=str(raw.get("lessons_learned", "")),
            action_items=actions,
            body=str(raw.get("body", "")),
            tags=tuple(raw.get("tags", [])),
        ))
    return out


def _incidents_from_json(data) -> List[IncidentRecord]:
    out: List[IncidentRecord] = []
    for raw in data:
        out.append(IncidentRecord(
            id=str(raw["id"]),
            title=str(raw.get("title", "")),
            severity=str(raw.get("severity", "medium")),
            occurred_at=_parse_dt(raw.get("occurred_at")),
        ))
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
        prog="replication.postmortem_quality_advisor",
        description="Audit incident postmortem quality.",
    )
    parser.add_argument("--demo", action="store_true", help="Use bundled demo payload.")
    parser.add_argument("--from-json", dest="from_json",
                        help="Path to JSON with {postmortems, incidents, risk_appetite}.")
    parser.add_argument("--risk", choices=APPETITES, default="balanced",
                        help="Risk appetite (default balanced).")
    parser.add_argument("--format", choices=("text", "json", "markdown"),
                        default="text", help="Output format.")
    parser.add_argument("--output", "-o", help="Write report to file instead of stdout.")
    args = parser.parse_args(argv)

    if args.demo:
        payload = _demo_payload()
    elif args.from_json:
        with open(args.from_json, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if isinstance(raw, dict):
            pms = _records_from_json(raw.get("postmortems", []))
            incs = _incidents_from_json(raw.get("incidents", []))
            appetite = raw.get("risk_appetite", args.risk)
        else:
            pms = _records_from_json(raw)
            incs = []
            appetite = args.risk
        payload = QualityInput(postmortems=pms, incidents=incs, risk_appetite=appetite)
    else:
        parser.error("Provide --demo or --from-json")
        return 2

    if args.risk and not args.from_json:
        payload.risk_appetite = args.risk

    advisor = PostmortemQualityAdvisor()
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
    else:
        sys.stdout.write(out + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
