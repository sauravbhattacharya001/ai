"""On-Call Load Balance Advisor — agentic shift-rotation fairness auditor.

Sibling to:

  - :mod:`replication.runbook_freshness_advisor` (audits the runbook library)
  - :mod:`replication.remediation_sprint_planner` (plans remediation sprints)
  - :mod:`replication.fatigue_detector` (per-agent decision fatigue)
  - :mod:`replication.safety_drill` (drills against the on-call team)

This module answers a different question: **is the safety on-call rotation
balanced, rested, and resilient?**  A roster where one person carries 60 %
of pages, where two consecutive 24 h shifts have no rest gap, or where a
coverage tier has a single point of failure is a real liability when the
next replication incident lands.

The advisor consumes a list of completed/scheduled on-call shifts plus the
current roster and emits per-responder verdicts, a portfolio summary, and
a P0-first deduplicated playbook of concrete interventions for the safety
team lead.

Per-responder verdicts
~~~~~~~~~~~~~~~~~~~~~~

* ``BURNOUT_RISK``        — hours or pages in the lookback window exceed
                            the burnout threshold.
* ``INSUFFICIENT_REST``   — at least one rest gap between consecutive
                            shifts is shorter than the minimum rest hours.
* ``SPOF_PRIMARY``        — only responder covering one or more tiers.
* ``OVERLOADED``          — load (hours or pages) ≥ 1.5× fair share.
* ``UNDERUSED``           — load ≤ 0.5× fair share while on the roster.
* ``HEALTHY``             — load within fair-share band, rested, no SPOF.
* ``INSUFFICIENT_DATA``   — responder has no shifts in the window.

Portfolio-level signals
~~~~~~~~~~~~~~~~~~~~~~~

* ``COVERAGE_GAP``               — at least one declared coverage tier has
                                   zero scheduled responders.
* ``WIDESPREAD_BURNOUT``         — >25 % of active responders burned out.
* ``LOAD_INEQUALITY``            — Gini coefficient of per-responder hours
                                   exceeds the appetite-specific cutoff.
* ``THIN_BENCH``                 — total roster ≤ 3 responders.
* ``FRESH_FACES_AVAILABLE``      — ≥ 2 underused responders that could be
                                   promoted to primary.
* ``HEALTHY_ROTATION``           — Gini below cutoff, no burnout, no SPOF.
* ``EMPTY_ROSTER``               — no responders or no shifts provided.

Risk appetite (``cautious`` | ``balanced`` | ``aggressive``) scales the
burnout / fair-share / rest thresholds — cautious tightens by ×0.7 (more
findings, more interventions), aggressive loosens by ×1.4.

CLI demo::

    python -m replication.oncall_load_balance_advisor --demo --format markdown
    python -m replication.oncall_load_balance_advisor --from-json roster.json \\
        --risk cautious --format json

Programmatic::

    from datetime import datetime, timezone
    from replication.oncall_load_balance_advisor import (
        OnCallLoadBalanceAdvisor,
        OnCallInput,
        OnCallShift,
    )

    now = datetime(2026, 5, 22, tzinfo=timezone.utc)
    advisor = OnCallLoadBalanceAdvisor(now=lambda: now)
    report = advisor.audit(OnCallInput(
        roster=["alice", "bob", "carol"],
        shifts=[
            OnCallShift(
                responder_id="alice",
                start=datetime(2026, 5, 15, tzinfo=timezone.utc),
                end=datetime(2026, 5, 22, tzinfo=timezone.utc),
                tier="critical",
                page_count=14,
            ),
        ],
    ))
    print(report.to_markdown())
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Tuple

from ._helpers import APPETITES, APPETITE_THRESHOLD_MULT


# ── Constants ─────────────────────────────────────────────────────────

VERDICT_BURNOUT_RISK = "BURNOUT_RISK"
VERDICT_INSUFFICIENT_REST = "INSUFFICIENT_REST"
VERDICT_SPOF_PRIMARY = "SPOF_PRIMARY"
VERDICT_OVERLOADED = "OVERLOADED"
VERDICT_UNDERUSED = "UNDERUSED"
VERDICT_HEALTHY = "HEALTHY"
VERDICT_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

ALL_VERDICTS: Tuple[str, ...] = (
    VERDICT_BURNOUT_RISK,
    VERDICT_INSUFFICIENT_REST,
    VERDICT_SPOF_PRIMARY,
    VERDICT_OVERLOADED,
    VERDICT_UNDERUSED,
    VERDICT_HEALTHY,
    VERDICT_INSUFFICIENT_DATA,
)

# Worst-of ranking. SPOF and burnout dominate; UNDERUSED is mild.
_VERDICT_RANK: Dict[str, int] = {
    VERDICT_HEALTHY: 0,
    VERDICT_INSUFFICIENT_DATA: 1,
    VERDICT_UNDERUSED: 2,
    VERDICT_OVERLOADED: 3,
    VERDICT_INSUFFICIENT_REST: 4,
    VERDICT_SPOF_PRIMARY: 5,
    VERDICT_BURNOUT_RISK: 6,
}

# Priorities per verdict.
_VERDICT_PRIORITY: Dict[str, str] = {
    VERDICT_BURNOUT_RISK: "P0",
    VERDICT_SPOF_PRIMARY: "P0",
    VERDICT_INSUFFICIENT_REST: "P0",
    VERDICT_OVERLOADED: "P1",
    VERDICT_UNDERUSED: "P2",
    VERDICT_HEALTHY: "P3",
    VERDICT_INSUFFICIENT_DATA: "P3",
}

# Balanced-appetite thresholds.
BASE_THRESHOLDS: Dict[str, float] = {
    # Lookback window for burnout / fair-share calculations.
    "lookback_days": 30.0,
    # Hours on-call in the window that trip BURNOUT_RISK.
    "burnout_hours": 168.0,
    # Pages handled in the window that trip BURNOUT_RISK.
    "burnout_pages": 30.0,
    # Minimum rest hours between consecutive shifts.
    "min_rest_hours": 12.0,
    # Multipliers vs. fair-share load that trip OVERLOADED / UNDERUSED.
    "overload_multiplier": 1.5,
    "underuse_multiplier": 0.5,
    # Gini coefficient over per-responder hours above which the portfolio
    # is judged unequal.
    "gini_cutoff": 0.35,
}


# ── Data model ────────────────────────────────────────────────────────


@dataclass
class OnCallShift:
    """A single on-call assignment.

    ``start`` / ``end`` are timezone-aware UTC :class:`datetime`. ``tier``
    is a free-form coverage tier label (e.g. ``"critical"``, ``"weekend"``,
    ``"secondary"``). ``page_count`` is how many real pages / incidents the
    responder handled during the shift.
    """

    responder_id: str
    start: datetime
    end: datetime
    tier: str = "primary"
    page_count: int = 0

    def duration_hours(self) -> float:
        if self.end <= self.start:
            return 0.0
        return (self.end - self.start).total_seconds() / 3600.0


@dataclass
class OnCallInput:
    roster: List[str] = field(default_factory=list)
    shifts: List[OnCallShift] = field(default_factory=list)
    tiers: List[str] = field(default_factory=list)
    risk_appetite: str = "balanced"


@dataclass
class ResponderFinding:
    responder_id: str
    verdict: str
    priority: str  # P0..P3
    load_score: float  # 0..100 (higher = more loaded)
    hours_in_window: float
    pages_in_window: int
    shifts_in_window: int
    fair_share_ratio: float  # observed_hours / fair_share_hours
    min_rest_hours: Optional[float]
    tiers_covered: List[str]
    spof_tiers: List[str]
    reasons: List[str]
    suggested_action: str


@dataclass
class OnCallPortfolio:
    roster_size: int
    active_responders: int
    total_shifts: int
    total_hours: float
    total_pages: int
    burnout_count: int
    overloaded_count: int
    underused_count: int
    spof_count: int
    rest_violations: int
    coverage_gaps: List[str]
    gini_hours: float
    grade: str  # A..F


@dataclass
class OnCallPlaybookAction:
    priority: str
    label: str
    reason: str
    responder_ids: List[str]
    suggested_value: Optional[str] = None


@dataclass
class OnCallReport:
    generated_at: datetime
    risk_appetite: str
    thresholds: Dict[str, float]
    portfolio: OnCallPortfolio
    findings: List[ResponderFinding]
    insights: List[str]
    playbook: List[OnCallPlaybookAction]

    # ── Renderers ───────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        return _serialize(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2, default=str)

    def to_text(self) -> str:
        p = self.portfolio
        lines: List[str] = []
        lines.append(
            f"On-Call Load Balance — grade={p.grade} appetite={self.risk_appetite} "
            f"roster={p.roster_size} active={p.active_responders} "
            f"shifts={p.total_shifts} hours={p.total_hours:.0f} pages={p.total_pages} "
            f"burnout={p.burnout_count} spof={p.spof_count} "
            f"rest_violations={p.rest_violations} gini={p.gini_hours:.2f}"
        )
        if p.coverage_gaps:
            lines.append(f"Coverage gaps: {', '.join(p.coverage_gaps)}")
        lines.append("")
        lines.append("Findings:")
        for f in self.findings:
            rest = "-" if f.min_rest_hours is None else f"{f.min_rest_hours:.1f}h"
            lines.append(
                f"  [{f.priority}] {f.responder_id} {f.verdict} "
                f"load={f.load_score:.0f} hrs={f.hours_in_window:.0f} "
                f"pages={f.pages_in_window} share={f.fair_share_ratio:.2f}x "
                f"rest={rest}"
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
                    f"({len(a.responder_ids)} responder(s)) — {a.reason}"
                )
        return "\n".join(lines)

    def to_markdown(self) -> str:
        p = self.portfolio
        lines: List[str] = []
        lines.append("# On-Call Load Balance Report")
        lines.append("")
        lines.append(f"- generated_at: `{self.generated_at.isoformat()}`")
        lines.append(f"- risk_appetite: **{self.risk_appetite}**")
        lines.append(f"- grade: **{p.grade}**")
        lines.append("")
        lines.append("## Portfolio")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("| --- | --- |")
        lines.append(f"| roster_size | {p.roster_size} |")
        lines.append(f"| active_responders | {p.active_responders} |")
        lines.append(f"| total_shifts | {p.total_shifts} |")
        lines.append(f"| total_hours | {p.total_hours:.0f} |")
        lines.append(f"| total_pages | {p.total_pages} |")
        lines.append(f"| burnout_count | {p.burnout_count} |")
        lines.append(f"| overloaded_count | {p.overloaded_count} |")
        lines.append(f"| underused_count | {p.underused_count} |")
        lines.append(f"| spof_count | {p.spof_count} |")
        lines.append(f"| rest_violations | {p.rest_violations} |")
        lines.append(f"| gini_hours | {p.gini_hours:.3f} |")
        lines.append(
            f"| coverage_gaps | {', '.join(p.coverage_gaps) if p.coverage_gaps else '—'} |"
        )
        lines.append("")
        lines.append("## Findings")
        lines.append("")
        lines.append(
            "| priority | responder | verdict | hours | pages | share | rest | reasons |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for f in self.findings:
            rest = "—" if f.min_rest_hours is None else f"{f.min_rest_hours:.1f}h"
            lines.append(
                f"| {f.priority} | {f.responder_id} | {f.verdict} | "
                f"{f.hours_in_window:.0f} | {f.pages_in_window} | "
                f"{f.fair_share_ratio:.2f}x | {rest} | "
                f"{', '.join(f.reasons) if f.reasons else '—'} |"
            )
        lines.append("")
        lines.append("## Insights")
        lines.append("")
        if self.insights:
            for i in self.insights:
                lines.append(f"- {i}")
        else:
            lines.append("- (none)")
        lines.append("")
        lines.append("## Playbook")
        lines.append("")
        if self.playbook:
            lines.append("| priority | label | responders | reason |")
            lines.append("| --- | --- | --- | --- |")
            for a in self.playbook:
                lines.append(
                    f"| {a.priority} | {a.label} | "
                    f"{', '.join(a.responder_ids) if a.responder_ids else '—'} | "
                    f"{a.reason} |"
                )
        else:
            lines.append("- (none)")
        return "\n".join(lines)


# ── Serialization ─────────────────────────────────────────────────────


def _serialize(obj):
    if isinstance(obj, OnCallReport):
        return {
            "generated_at": obj.generated_at.isoformat(),
            "risk_appetite": obj.risk_appetite,
            "thresholds": dict(obj.thresholds),
            "portfolio": _serialize(obj.portfolio),
            "findings": [_serialize(f) for f in obj.findings],
            "insights": list(obj.insights),
            "playbook": [_serialize(a) for a in obj.playbook],
        }
    if isinstance(obj, OnCallPortfolio):
        return {
            "roster_size": obj.roster_size,
            "active_responders": obj.active_responders,
            "total_shifts": obj.total_shifts,
            "total_hours": round(obj.total_hours, 3),
            "total_pages": obj.total_pages,
            "burnout_count": obj.burnout_count,
            "overloaded_count": obj.overloaded_count,
            "underused_count": obj.underused_count,
            "spof_count": obj.spof_count,
            "rest_violations": obj.rest_violations,
            "coverage_gaps": list(obj.coverage_gaps),
            "gini_hours": round(obj.gini_hours, 4),
            "grade": obj.grade,
        }
    if isinstance(obj, ResponderFinding):
        return {
            "responder_id": obj.responder_id,
            "verdict": obj.verdict,
            "priority": obj.priority,
            "load_score": round(obj.load_score, 2),
            "hours_in_window": round(obj.hours_in_window, 3),
            "pages_in_window": obj.pages_in_window,
            "shifts_in_window": obj.shifts_in_window,
            "fair_share_ratio": round(obj.fair_share_ratio, 3),
            "min_rest_hours": (
                None if obj.min_rest_hours is None
                else round(obj.min_rest_hours, 3)
            ),
            "tiers_covered": list(obj.tiers_covered),
            "spof_tiers": list(obj.spof_tiers),
            "reasons": list(obj.reasons),
            "suggested_action": obj.suggested_action,
        }
    if isinstance(obj, OnCallPlaybookAction):
        return {
            "priority": obj.priority,
            "label": obj.label,
            "reason": obj.reason,
            "responder_ids": list(obj.responder_ids),
            "suggested_value": obj.suggested_value,
        }
    return obj


# ── Helpers ───────────────────────────────────────────────────────────


def _scaled_thresholds(appetite: str) -> Dict[str, float]:
    mult = APPETITE_THRESHOLD_MULT.get(appetite, 1.0)
    out = dict(BASE_THRESHOLDS)
    # Tighten thresholds for cautious (smaller numbers trip findings).
    for k in (
        "burnout_hours",
        "burnout_pages",
        "overload_multiplier",
        "gini_cutoff",
    ):
        out[k] = BASE_THRESHOLDS[k] * mult
    # Min rest gap moves *up* under cautious (mirror: divide by mult).
    out["min_rest_hours"] = BASE_THRESHOLDS["min_rest_hours"] / max(mult, 1e-6)
    # Underuse stays at the same fraction regardless of appetite.
    out["underuse_multiplier"] = BASE_THRESHOLDS["underuse_multiplier"]
    out["lookback_days"] = BASE_THRESHOLDS["lookback_days"]
    return out


def _gini(values: List[float]) -> float:
    """Standard Gini coefficient. 0 = perfectly equal, 1 = max inequality."""
    xs = [v for v in values if v > 0]
    n = len(xs)
    if n == 0:
        return 0.0
    if n == 1:
        return 0.0
    xs = sorted(xs)
    total = sum(xs)
    if total <= 0:
        return 0.0
    cum = 0.0
    for i, v in enumerate(xs, start=1):
        cum += i * v
    # Mean Absolute Difference form: G = (2*sum(i*xi)/(n*sum(xi))) - (n+1)/n
    return max(0.0, (2.0 * cum) / (n * total) - (n + 1.0) / n)


# ── Advisor ───────────────────────────────────────────────────────────


class OnCallLoadBalanceAdvisor:
    """Audit a safety on-call rotation for fairness, rest, and resilience.

    The advisor is read-only: it never mutates the input shifts or
    roster (a deep copy is taken on entry).  All time math uses the
    injected ``now`` callable so tests can pin a deterministic clock.
    """

    def __init__(self, now: Optional[Callable[[], datetime]] = None):
        self._now = now or (lambda: datetime.now(timezone.utc))

    # ----- public API -------------------------------------------------

    def audit(self, payload: OnCallInput) -> OnCallReport:
        appetite = payload.risk_appetite
        if appetite not in APPETITES:
            appetite = "balanced"
        thresholds = _scaled_thresholds(appetite)
        now = self._now()
        lookback_start = now - timedelta(days=thresholds["lookback_days"])

        roster = list(dict.fromkeys(payload.roster or []))
        shifts = sorted(
            (copy.deepcopy(s) for s in payload.shifts or []),
            key=lambda s: (s.responder_id, s.start),
        )

        # Limit to shifts overlapping the lookback window.
        in_window: List[OnCallShift] = []
        for s in shifts:
            if s.end <= lookback_start:
                continue
            if s.start >= now:
                # Future-scheduled — still counts for coverage gap.
                in_window.append(s)
                continue
            # Clip to the window.
            effective_start = max(s.start, lookback_start)
            effective_end = min(s.end, now)
            if effective_end > effective_start:
                clipped = OnCallShift(
                    responder_id=s.responder_id,
                    start=effective_start,
                    end=effective_end,
                    tier=s.tier,
                    page_count=s.page_count,
                )
                in_window.append(clipped)

        # Derive declared tiers (input.tiers + observed).
        declared_tiers = list(dict.fromkeys(payload.tiers or []))
        observed_tiers = list({s.tier for s in in_window})
        all_tiers = list(dict.fromkeys(declared_tiers + observed_tiers))

        # Per-tier responders.
        tier_responders: Dict[str, List[str]] = {t: [] for t in all_tiers}
        for s in in_window:
            if s.responder_id not in tier_responders[s.tier]:
                tier_responders[s.tier].append(s.responder_id)

        coverage_gaps = sorted(
            t for t in declared_tiers if not tier_responders.get(t)
        )

        # Per-responder load.
        per_resp_hours: Dict[str, float] = {}
        per_resp_pages: Dict[str, int] = {}
        per_resp_shift_count: Dict[str, int] = {}
        per_resp_tiers: Dict[str, List[str]] = {}
        for s in in_window:
            if s.start >= now:
                # Don't double-count future shifts toward used-hours.
                hrs = 0.0
            else:
                hrs = s.duration_hours()
            per_resp_hours[s.responder_id] = (
                per_resp_hours.get(s.responder_id, 0.0) + hrs
            )
            per_resp_pages[s.responder_id] = (
                per_resp_pages.get(s.responder_id, 0) + int(s.page_count)
            )
            per_resp_shift_count[s.responder_id] = (
                per_resp_shift_count.get(s.responder_id, 0) + 1
            )
            tiers = per_resp_tiers.setdefault(s.responder_id, [])
            if s.tier not in tiers:
                tiers.append(s.tier)

        # Make sure roster members appear with zeros if no shifts.
        for r in roster:
            per_resp_hours.setdefault(r, 0.0)
            per_resp_pages.setdefault(r, 0)
            per_resp_shift_count.setdefault(r, 0)
            per_resp_tiers.setdefault(r, [])

        total_hours = sum(per_resp_hours.values())
        total_pages = sum(per_resp_pages.values())
        roster_size = max(len(roster), len(per_resp_hours))
        fair_share_hours = (
            total_hours / max(1, roster_size) if total_hours > 0 else 0.0
        )

        # Per-responder rest analysis (look at all sorted shifts per resp).
        per_resp_min_rest: Dict[str, Optional[float]] = {}
        # Group sorted shifts per responder.
        ordered: Dict[str, List[OnCallShift]] = {}
        for s in in_window:
            ordered.setdefault(s.responder_id, []).append(s)
        for rid, lst in ordered.items():
            lst.sort(key=lambda x: x.start)
            min_gap: Optional[float] = None
            for i in range(1, len(lst)):
                gap_h = (lst[i].start - lst[i - 1].end).total_seconds() / 3600.0
                if min_gap is None or gap_h < min_gap:
                    min_gap = gap_h
            per_resp_min_rest[rid] = min_gap

        # Build findings.
        findings: List[ResponderFinding] = []
        responder_ids = sorted(per_resp_hours.keys())
        for rid in responder_ids:
            hrs = per_resp_hours[rid]
            pages = per_resp_pages[rid]
            shift_count = per_resp_shift_count[rid]
            tiers = sorted(per_resp_tiers[rid])
            min_rest = per_resp_min_rest.get(rid)

            spof_tiers = sorted(
                t for t in tiers
                if len(tier_responders.get(t, [])) == 1
            )

            reasons: List[str] = []
            if fair_share_hours > 0:
                share_ratio = hrs / fair_share_hours
            else:
                share_ratio = 0.0

            burned_out = (
                hrs >= thresholds["burnout_hours"]
                or pages >= thresholds["burnout_pages"]
            )
            if burned_out:
                if hrs >= thresholds["burnout_hours"]:
                    reasons.append(
                        f"hours {hrs:.0f} ≥ burnout threshold "
                        f"{thresholds['burnout_hours']:.0f}"
                    )
                if pages >= thresholds["burnout_pages"]:
                    reasons.append(
                        f"pages {pages} ≥ burnout threshold "
                        f"{thresholds['burnout_pages']:.0f}"
                    )

            insufficient_rest = (
                min_rest is not None
                and min_rest < thresholds["min_rest_hours"]
            )
            if insufficient_rest:
                reasons.append(
                    f"min rest gap {min_rest:.1f}h < required "
                    f"{thresholds['min_rest_hours']:.1f}h"
                )

            is_spof = bool(spof_tiers)
            if is_spof:
                reasons.append(
                    f"sole coverage for tier(s): {', '.join(spof_tiers)}"
                )

            overloaded = share_ratio >= thresholds["overload_multiplier"]
            if overloaded and not burned_out:
                reasons.append(
                    f"hours share {share_ratio:.2f}× ≥ overload "
                    f"{thresholds['overload_multiplier']:.2f}×"
                )

            underused = (
                shift_count > 0
                and share_ratio <= thresholds["underuse_multiplier"]
                and not burned_out
                and not is_spof
                and not insufficient_rest
            ) or (shift_count == 0 and rid in roster)

            if shift_count == 0 and rid in roster:
                reasons.append("no shifts in lookback window")

            # Decide verdict — worst-of.
            candidates: List[str] = []
            if burned_out:
                candidates.append(VERDICT_BURNOUT_RISK)
            if is_spof:
                candidates.append(VERDICT_SPOF_PRIMARY)
            if insufficient_rest:
                candidates.append(VERDICT_INSUFFICIENT_REST)
            if overloaded:
                candidates.append(VERDICT_OVERLOADED)
            if underused:
                candidates.append(VERDICT_UNDERUSED)
            if shift_count == 0 and rid not in roster:
                candidates.append(VERDICT_INSUFFICIENT_DATA)
            if not candidates:
                candidates.append(VERDICT_HEALTHY)

            verdict = max(candidates, key=lambda v: _VERDICT_RANK[v])
            priority = _VERDICT_PRIORITY[verdict]

            load_score = _compute_load_score(
                hrs=hrs,
                pages=pages,
                thresholds=thresholds,
                share_ratio=share_ratio,
            )

            suggested = _suggested_action(verdict, spof_tiers)

            findings.append(
                ResponderFinding(
                    responder_id=rid,
                    verdict=verdict,
                    priority=priority,
                    load_score=load_score,
                    hours_in_window=hrs,
                    pages_in_window=pages,
                    shifts_in_window=shift_count,
                    fair_share_ratio=share_ratio,
                    min_rest_hours=min_rest,
                    tiers_covered=tiers,
                    spof_tiers=spof_tiers,
                    reasons=reasons,
                    suggested_action=suggested,
                )
            )

        # Stable order: priority asc (P0 first), then load desc, then id asc.
        findings.sort(
            key=lambda f: (
                _priority_rank(f.priority),
                -f.load_score,
                f.responder_id,
            )
        )

        # Portfolio.
        burnout_count = sum(
            1 for f in findings if f.verdict == VERDICT_BURNOUT_RISK
        )
        overloaded_count = sum(
            1 for f in findings if f.verdict == VERDICT_OVERLOADED
        )
        underused_count = sum(
            1 for f in findings if f.verdict == VERDICT_UNDERUSED
        )
        spof_count = sum(
            1 for f in findings if f.verdict == VERDICT_SPOF_PRIMARY
        )
        rest_violations = sum(
            1 for f in findings
            if f.verdict == VERDICT_INSUFFICIENT_REST
        )
        active_responders = sum(1 for f in findings if f.shifts_in_window > 0)

        gini = _gini(
            [f.hours_in_window for f in findings if f.hours_in_window > 0]
        )

        grade = _grade(
            findings=findings,
            gini=gini,
            gini_cutoff=thresholds["gini_cutoff"],
            coverage_gaps=coverage_gaps,
        )

        portfolio = OnCallPortfolio(
            roster_size=roster_size,
            active_responders=active_responders,
            total_shifts=len(in_window),
            total_hours=total_hours,
            total_pages=total_pages,
            burnout_count=burnout_count,
            overloaded_count=overloaded_count,
            underused_count=underused_count,
            spof_count=spof_count,
            rest_violations=rest_violations,
            coverage_gaps=coverage_gaps,
            gini_hours=gini,
            grade=grade,
        )

        insights = _build_insights(
            findings=findings,
            portfolio=portfolio,
            thresholds=thresholds,
        )

        playbook = _build_playbook(
            findings=findings,
            portfolio=portfolio,
            thresholds=thresholds,
        )

        return OnCallReport(
            generated_at=now,
            risk_appetite=appetite,
            thresholds=thresholds,
            portfolio=portfolio,
            findings=findings,
            insights=insights,
            playbook=playbook,
        )


# ── Scoring / grading helpers ─────────────────────────────────────────


_PRIORITY_RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


def _priority_rank(p: str) -> int:
    return _PRIORITY_RANK.get(p, 4)


def _compute_load_score(
    *,
    hrs: float,
    pages: float,
    thresholds: Dict[str, float],
    share_ratio: float,
) -> float:
    """Compose a 0..100 load score for ranking findings."""
    hours_component = (
        min(1.0, hrs / max(1.0, thresholds["burnout_hours"])) * 50.0
    )
    pages_component = (
        min(1.0, pages / max(1.0, thresholds["burnout_pages"])) * 30.0
    )
    share_component = min(2.0, share_ratio) * 10.0  # 0..20
    return max(0.0, min(100.0, hours_component + pages_component + share_component))


def _suggested_action(verdict: str, spof_tiers: List[str]) -> str:
    if verdict == VERDICT_BURNOUT_RISK:
        return "rotate out of next two on-call windows; book recovery PTO"
    if verdict == VERDICT_INSUFFICIENT_REST:
        return "enforce minimum rest gap; reshuffle the next shift swap"
    if verdict == VERDICT_SPOF_PRIMARY:
        return (
            "add a secondary responder for tier(s) "
            + (", ".join(spof_tiers) if spof_tiers else "—")
        )
    if verdict == VERDICT_OVERLOADED:
        return "redistribute future shifts onto under-loaded responders"
    if verdict == VERDICT_UNDERUSED:
        return "schedule a shadow shift to ramp up to primary"
    if verdict == VERDICT_INSUFFICIENT_DATA:
        return "include in next rotation to build coverage signal"
    return "no action needed"


def _grade(
    *,
    findings: List[ResponderFinding],
    gini: float,
    gini_cutoff: float,
    coverage_gaps: List[str],
) -> str:
    burnout = any(f.verdict == VERDICT_BURNOUT_RISK for f in findings)
    spof = any(f.verdict == VERDICT_SPOF_PRIMARY for f in findings)
    rest = any(f.verdict == VERDICT_INSUFFICIENT_REST for f in findings)
    if coverage_gaps:
        return "F"
    if burnout and (spof or rest):
        return "F"
    if burnout or spof or rest:
        return "D"
    if any(f.verdict == VERDICT_OVERLOADED for f in findings):
        return "C" if gini > gini_cutoff else "B"
    if gini > gini_cutoff:
        return "C"
    return "A"


# ── Insights ──────────────────────────────────────────────────────────


def _build_insights(
    *,
    findings: List[ResponderFinding],
    portfolio: OnCallPortfolio,
    thresholds: Dict[str, float],
) -> List[str]:
    insights: List[str] = []
    if portfolio.roster_size == 0 or portfolio.total_shifts == 0:
        insights.append("EMPTY_ROSTER")
        return insights
    if portfolio.coverage_gaps:
        insights.append("COVERAGE_GAP")
    if (
        portfolio.active_responders > 0
        and portfolio.burnout_count / portfolio.active_responders > 0.25
    ):
        insights.append("WIDESPREAD_BURNOUT")
    if portfolio.gini_hours > thresholds["gini_cutoff"]:
        insights.append("LOAD_INEQUALITY")
    if portfolio.roster_size <= 3:
        insights.append("THIN_BENCH")
    underused = [f for f in findings if f.verdict == VERDICT_UNDERUSED]
    if len(underused) >= 2:
        insights.append("FRESH_FACES_AVAILABLE")
    if (
        not portfolio.coverage_gaps
        and portfolio.burnout_count == 0
        and portfolio.spof_count == 0
        and portfolio.rest_violations == 0
        and portfolio.gini_hours <= thresholds["gini_cutoff"]
    ):
        insights.append("HEALTHY_ROTATION")
    return insights


# ── Playbook ──────────────────────────────────────────────────────────


def _build_playbook(
    *,
    findings: List[ResponderFinding],
    portfolio: OnCallPortfolio,
    thresholds: Dict[str, float],
) -> List[OnCallPlaybookAction]:
    actions: List[OnCallPlaybookAction] = []

    burnout = [f for f in findings if f.verdict == VERDICT_BURNOUT_RISK]
    spof = [f for f in findings if f.verdict == VERDICT_SPOF_PRIMARY]
    rest = [f for f in findings if f.verdict == VERDICT_INSUFFICIENT_REST]
    overloaded = [f for f in findings if f.verdict == VERDICT_OVERLOADED]
    underused = [f for f in findings if f.verdict == VERDICT_UNDERUSED]

    if portfolio.coverage_gaps:
        actions.append(
            OnCallPlaybookAction(
                priority="P0",
                label="FILL_COVERAGE_GAP",
                reason=(
                    "tier(s) with zero scheduled responders: "
                    + ", ".join(portfolio.coverage_gaps)
                ),
                responder_ids=[],
                suggested_value=", ".join(portfolio.coverage_gaps),
            )
        )

    if burnout:
        actions.append(
            OnCallPlaybookAction(
                priority="P0",
                label="ROTATE_OUT_BURNED_OUT_RESPONDERS",
                reason=(
                    f"{len(burnout)} responder(s) exceeded burnout "
                    f"threshold ({thresholds['burnout_hours']:.0f}h "
                    f"or {thresholds['burnout_pages']:.0f} pages)"
                ),
                responder_ids=sorted(f.responder_id for f in burnout),
            )
        )

    if spof:
        spof_tier_set = sorted({
            t for f in spof for t in f.spof_tiers
        })
        actions.append(
            OnCallPlaybookAction(
                priority="P0",
                label="RECRUIT_BACKUP_FOR_SPOF_TIERS",
                reason=(
                    "single point of failure on tier(s): "
                    + ", ".join(spof_tier_set)
                ),
                responder_ids=sorted(f.responder_id for f in spof),
                suggested_value=", ".join(spof_tier_set),
            )
        )

    if rest:
        actions.append(
            OnCallPlaybookAction(
                priority="P0",
                label="ENFORCE_MIN_REST_GAP",
                reason=(
                    f"{len(rest)} responder(s) had a rest gap below "
                    f"{thresholds['min_rest_hours']:.1f}h"
                ),
                responder_ids=sorted(f.responder_id for f in rest),
                suggested_value=f"{thresholds['min_rest_hours']:.1f}h",
            )
        )

    if overloaded:
        actions.append(
            OnCallPlaybookAction(
                priority="P1",
                label="REDISTRIBUTE_LOAD",
                reason=(
                    f"{len(overloaded)} responder(s) at ≥"
                    f"{thresholds['overload_multiplier']:.2f}× fair share"
                ),
                responder_ids=sorted(f.responder_id for f in overloaded),
            )
        )

    if portfolio.gini_hours > thresholds["gini_cutoff"] and not overloaded:
        actions.append(
            OnCallPlaybookAction(
                priority="P1",
                label="REBALANCE_ROTATION",
                reason=(
                    f"Gini coefficient {portfolio.gini_hours:.2f} exceeds "
                    f"cutoff {thresholds['gini_cutoff']:.2f}"
                ),
                responder_ids=[],
            )
        )

    if len(underused) >= 2:
        actions.append(
            OnCallPlaybookAction(
                priority="P2",
                label="PROMOTE_UNDERUSED_RESPONDERS",
                reason=(
                    f"{len(underused)} responder(s) at ≤"
                    f"{thresholds['underuse_multiplier']:.2f}× fair share — "
                    "promote via shadow shifts"
                ),
                responder_ids=sorted(f.responder_id for f in underused),
            )
        )

    if portfolio.roster_size <= 3 and not portfolio.coverage_gaps:
        actions.append(
            OnCallPlaybookAction(
                priority="P1",
                label="GROW_ON_CALL_BENCH",
                reason=(
                    f"only {portfolio.roster_size} responder(s) on roster — "
                    "single illness/vacation creates a coverage gap"
                ),
                responder_ids=[],
            )
        )

    if not actions:
        actions.append(
            OnCallPlaybookAction(
                priority="P3",
                label="HEALTHY_ROTATION_NO_ACTION_NEEDED",
                reason="no findings exceeded the configured thresholds",
                responder_ids=[],
            )
        )

    # Dedup by (label, priority) preserving first occurrence; stable order
    # already P0-first because we emit them in that order.
    seen: Dict[Tuple[str, str], int] = {}
    deduped: List[OnCallPlaybookAction] = []
    for a in actions:
        key = (a.priority, a.label)
        if key in seen:
            continue
        seen[key] = len(deduped)
        deduped.append(a)
    deduped.sort(key=lambda a: (_priority_rank(a.priority), a.label))
    return deduped


# ── CLI ───────────────────────────────────────────────────────────────


def _demo_input(now: datetime) -> OnCallInput:
    def hours_ago(h):
        return now - timedelta(hours=h)

    def days_ago(d):
        return now - timedelta(days=d)

    return OnCallInput(
        roster=["alice", "bob", "carol", "dave"],
        tiers=["critical", "weekend", "secondary"],
        shifts=[
            # Alice: heavy load on critical, no rest gap between two shifts.
            OnCallShift("alice", days_ago(20), days_ago(13), "critical", 12),
            OnCallShift("alice", days_ago(13), days_ago(6), "critical", 10),
            OnCallShift("alice", days_ago(5), hours_ago(12), "critical", 9),
            # Bob: weekend SPOF, modest pages.
            OnCallShift("bob", days_ago(22), days_ago(19), "weekend", 2),
            OnCallShift("bob", days_ago(15), days_ago(12), "weekend", 3),
            # Carol: very light load.
            OnCallShift("carol", days_ago(18), days_ago(17), "secondary", 1),
            # Dave: zero shifts in window → underused (he's on the roster).
        ],
    )


def _from_json_path(path: str) -> OnCallInput:
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    roster = list(raw.get("roster", []) or [])
    tiers = list(raw.get("tiers", []) or [])
    shifts_raw = raw.get("shifts", []) or []
    shifts: List[OnCallShift] = []
    for s in shifts_raw:
        shifts.append(
            OnCallShift(
                responder_id=str(s["responder_id"]),
                start=_parse_dt(s["start"]),
                end=_parse_dt(s["end"]),
                tier=str(s.get("tier", "primary")),
                page_count=int(s.get("page_count", 0)),
            )
        )
    appetite = str(raw.get("risk_appetite", "balanced"))
    return OnCallInput(
        roster=roster,
        tiers=tiers,
        shifts=shifts,
        risk_appetite=appetite,
    )


def _parse_dt(value) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    s = str(value).strip()
    # Accept trailing 'Z'.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="oncall_load_balance_advisor",
        description="Audit safety on-call rotation for load balance and burnout.",
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--demo", action="store_true", help="run with built-in demo data")
    src.add_argument("--from-json", dest="from_json", default=None,
                     help="path to a JSON file with roster/tiers/shifts")
    parser.add_argument(
        "--risk", choices=APPETITES, default="balanced",
        help="risk appetite (default: balanced)",
    )
    parser.add_argument(
        "--format", choices=("text", "markdown", "json"), default="text",
    )
    args = parser.parse_args(argv)

    if not args.demo and not args.from_json:
        parser.error("either --demo or --from-json is required")

    now = datetime.now(timezone.utc)
    if args.demo:
        payload = _demo_input(now)
    else:
        payload = _from_json_path(args.from_json)
    payload.risk_appetite = args.risk

    advisor = OnCallLoadBalanceAdvisor(now=lambda: now)
    report = advisor.audit(payload)

    if args.format == "json":
        out = report.to_json()
    elif args.format == "markdown":
        out = report.to_markdown()
    else:
        out = report.to_text()
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    print(out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
