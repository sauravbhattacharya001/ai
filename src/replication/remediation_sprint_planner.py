"""Remediation Sprint Planner — agentic sprint-capacity packer.

9th sibling in the remediation suite. Consumes a list of
:class:`replication.remediation_planner.RemediationAction` (or any
duck-typed equivalent) plus team capacity per sprint, topologically sorts
by ``depends_on``, and greedily packs work into a fixed-horizon sprint
plan respecting per-owner capacity.

Per-action verdicts:

* ``READY_NOW``               — P0, critical/high severity scheduled in sprint 0.
* ``SCHEDULED``               — P1/P2, placed inside the horizon.
* ``DEFERRED_PAST_HORIZON``   — P1, did not fit within horizon.
* ``BLOCKED_BY_DEPENDENCY``   — depends on a deferred / blocked action.
* ``OVERSIZED_ACTION``        — P1, ``effort_days`` exceeds per-owner sprint cap.
* ``INSUFFICIENT_CAPACITY``   — P0, a P0 action could not be packed.

Risk appetite (``cautious`` | ``balanced`` | ``aggressive``) scales the
per-owner sprint capacity by ``0.8 | 1.0 | 1.2``.

CLI::

    python -m replication.remediation_sprint_planner --demo --format markdown
    python -m replication.remediation_sprint_planner --from-plan plan.json --risk cautious
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field, asdict, is_dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


# ── Constants ────────────────────────────────────────────────────────

SEVERITY_LEVELS: Tuple[str, ...] = ("info", "low", "medium", "high", "critical")
SEVERITY_WEIGHT: Dict[str, int] = {
    "info": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}

APPETITES: Tuple[str, ...] = ("cautious", "balanced", "aggressive")
APPETITE_CAPACITY_MULT: Dict[str, float] = {
    "cautious": 0.80,
    "balanced": 1.00,
    "aggressive": 1.20,
}

DEFAULT_HORIZON_SPRINTS = 4
DEFAULT_SPRINT_LENGTH_DAYS = 14
DEFAULT_CAPACITY_PER_OWNER = 10.0
UNASSIGNED_OWNER = "unassigned"


# ── Data model ───────────────────────────────────────────────────────


@dataclass
class SprintAssignment:
    action_id: str
    sprint_index: Optional[int]  # None when deferred or oversized/insufficient
    owner: str
    effort_days: float
    severity: str
    urgency: int
    verdict: str
    priority: str  # P0..P3
    reasons: List[str] = field(default_factory=list)


@dataclass
class SprintSlot:
    sprint_index: int
    start_date: datetime
    end_date: datetime
    total_effort: float
    capacity: float
    utilization_pct: float
    action_ids: List[str] = field(default_factory=list)
    owner_load: Dict[str, float] = field(default_factory=dict)


@dataclass
class SprintPlaybookAction:
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
class SprintPortfolio:
    total_actions: int
    scheduled: int
    deferred: int
    blocked: int
    oversized: int
    insufficient: int
    mean_utilization: float
    max_utilization: float
    sprints_to_clear_p0: Optional[int]
    grade: str  # A..F
    concentration_band: str  # HEALTHY | WATCH | STRETCHED | CRITICAL


@dataclass
class SprintPlanReport:
    generated_at: datetime
    risk_appetite: str
    horizon_sprints: int
    sprint_length_days: int
    portfolio: SprintPortfolio
    sprints: List[SprintSlot]
    assignments: List[SprintAssignment]
    playbook: List[SprintPlaybookAction]
    insights: List[str]

    # ── Renderers ───────────────────────────────────────────────────

    def to_json(self) -> str:
        return json.dumps(_serialize(self), sort_keys=True, indent=2, default=str)

    def to_text(self) -> str:
        p = self.portfolio
        lines: List[str] = []
        lines.append(
            f"Remediation Sprint Plan — grade={p.grade} band={p.concentration_band} "
            f"scheduled={p.scheduled} deferred={p.deferred} blocked={p.blocked} "
            f"oversized={p.oversized} appetite={self.risk_appetite}"
        )
        lines.append(
            f"horizon={self.horizon_sprints} sprints x {self.sprint_length_days}d, "
            f"mean_util={p.mean_utilization:.1f}% max_util={p.max_utilization:.1f}% "
            f"p0_clear={p.sprints_to_clear_p0 if p.sprints_to_clear_p0 is not None else '-'}"
        )
        lines.append("")
        lines.append("Sprints:")
        for s in self.sprints:
            lines.append(
                f"  S{s.sprint_index} [{s.start_date.date()}..{s.end_date.date()}] "
                f"effort={s.total_effort:.1f}/{s.capacity:.1f}d "
                f"util={s.utilization_pct:.1f}% actions={len(s.action_ids)}"
            )
        lines.append("")
        lines.append("Assignments:")
        for a in self.assignments:
            si = "-" if a.sprint_index is None else f"S{a.sprint_index}"
            lines.append(
                f"  [{a.priority}] {a.action_id} -> {si} {a.verdict} "
                f"owner={a.owner} effort={a.effort_days:.1f}d sev={a.severity}"
            )
            if a.reasons:
                lines.append(f"      reasons: {', '.join(a.reasons)}")
        lines.append("")
        lines.append("Playbook:")
        for pa in self.playbook:
            lines.append(
                f"  [{pa.priority}] {pa.id} owner={pa.owner} blast={pa.blast_radius} "
                f"rev={pa.reversibility}: {pa.label}"
            )
            lines.append(f"      reason: {pa.reason}")
        lines.append("")
        lines.append("Insights: " + (", ".join(self.insights) if self.insights else "-"))
        return "\n".join(lines)

    def to_markdown(self) -> str:
        p = self.portfolio
        out: List[str] = []
        out.append("# Remediation Sprint Plan")
        out.append("")
        out.append("## Summary")
        out.append("")
        out.append("| Metric | Value |")
        out.append("|---|---|")
        out.append(f"| grade | {p.grade} |")
        out.append(f"| band | {p.concentration_band} |")
        out.append(f"| risk_appetite | {self.risk_appetite} |")
        out.append(f"| horizon_sprints | {self.horizon_sprints} |")
        out.append(f"| sprint_length_days | {self.sprint_length_days} |")
        out.append(f"| total_actions | {p.total_actions} |")
        out.append(f"| scheduled | {p.scheduled} |")
        out.append(f"| deferred | {p.deferred} |")
        out.append(f"| blocked | {p.blocked} |")
        out.append(f"| oversized | {p.oversized} |")
        out.append(f"| insufficient | {p.insufficient} |")
        out.append(f"| mean_utilization_pct | {p.mean_utilization:.1f} |")
        out.append(f"| max_utilization_pct | {p.max_utilization:.1f} |")
        out.append(
            f"| sprints_to_clear_p0 | "
            f"{p.sprints_to_clear_p0 if p.sprints_to_clear_p0 is not None else '-'} |"
        )
        out.append("")
        out.append("## Sprints")
        out.append("")
        out.append("| Sprint | Start | End | Effort | Capacity | Utilization % | Actions |")
        out.append("|---|---|---|---|---|---|---|")
        for s in self.sprints:
            out.append(
                f"| S{s.sprint_index} | {s.start_date.date()} | {s.end_date.date()} | "
                f"{s.total_effort:.1f} | {s.capacity:.1f} | {s.utilization_pct:.1f} | "
                f"{len(s.action_ids)} |"
            )
        out.append("")
        out.append("## Assignments")
        out.append("")
        out.append(
            "| Priority | ID | Sprint | Verdict | Owner | Effort | Severity | Urgency | Reasons |"
        )
        out.append("|---|---|---|---|---|---|---|---|---|")
        for a in self.assignments:
            si = "-" if a.sprint_index is None else f"S{a.sprint_index}"
            out.append(
                f"| {a.priority} | {a.action_id} | {si} | {a.verdict} | {a.owner} | "
                f"{a.effort_days:.1f} | {a.severity} | {a.urgency} | "
                f"{'; '.join(a.reasons) or '-'} |"
            )
        out.append("")
        out.append("## Playbook")
        out.append("")
        out.append("| Priority | ID | Owner | Blast | Reversibility | Label | Reason |")
        out.append("|---|---|---|---|---|---|---|")
        for pa in self.playbook:
            out.append(
                f"| {pa.priority} | {pa.id} | {pa.owner} | {pa.blast_radius} | "
                f"{pa.reversibility} | {pa.label} | {pa.reason} |"
            )
        out.append("")
        out.append("## Insights")
        out.append("")
        if self.insights:
            for ins in self.insights:
                out.append(f"- {ins}")
        else:
            out.append("- (none)")
        return "\n".join(out)


# ── Helpers ──────────────────────────────────────────────────────────


def _serialize(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if is_dataclass(obj):
        return _serialize(asdict(obj))
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    return obj


def _norm_severity(s: Any) -> str:
    s = (str(s) if s is not None else "medium").strip().lower()
    return s if s in SEVERITY_WEIGHT else "medium"


@dataclass
class _NormAction:
    id: str
    title: str
    severity: str
    effort_days: float
    urgency: int
    depends_on: List[str]
    owner: str


def _coerce_action(raw: Any) -> _NormAction:
    """Accept :class:`RemediationAction` or any duck-typed equivalent."""

    def _get(name: str, default: Any = None) -> Any:
        if isinstance(raw, dict):
            return raw.get(name, default)
        return getattr(raw, name, default)

    aid = str(_get("id", ""))
    if not aid:
        raise ValueError("action is missing 'id'")
    title = str(_get("title", aid))
    severity = _norm_severity(_get("severity", "medium"))
    effort_raw = _get("effort", _get("effort_days", 1))
    try:
        effort_days = max(0.5, float(effort_raw))
    except (TypeError, ValueError):
        effort_days = 1.0
    urgency_raw = _get("urgency", 3)
    try:
        urgency = int(urgency_raw)
    except (TypeError, ValueError):
        urgency = 3
    deps_raw = _get("depends_on", []) or []
    depends_on = [str(d) for d in deps_raw]
    owner = str(_get("owner_hint", _get("owner", ""))).strip() or UNASSIGNED_OWNER
    return _NormAction(
        id=aid,
        title=title,
        severity=severity,
        effort_days=effort_days,
        urgency=urgency,
        depends_on=depends_on,
        owner=owner,
    )


def _topo_layers(actions: List[_NormAction]) -> Tuple[List[List[_NormAction]], List[str]]:
    """Return (layers, cycle_ids). Cycle nodes are excluded from layers."""

    by_id = {a.id: a for a in actions}
    indeg: Dict[str, int] = {a.id: 0 for a in actions}
    children: Dict[str, List[str]] = {a.id: [] for a in actions}
    for a in actions:
        for d in a.depends_on:
            if d in by_id:
                indeg[a.id] += 1
                children[d].append(a.id)
    # Kahn's algorithm
    layers: List[List[_NormAction]] = []
    remaining = dict(indeg)
    ready = [aid for aid, deg in remaining.items() if deg == 0]
    while ready:
        # sort ready deterministically: severity desc, urgency desc, effort asc, id asc
        ready_sorted = sorted(
            ready,
            key=lambda aid: (
                -SEVERITY_WEIGHT[by_id[aid].severity],
                -by_id[aid].urgency,
                by_id[aid].effort_days,
                aid,
            ),
        )
        layers.append([by_id[aid] for aid in ready_sorted])
        next_ready: List[str] = []
        for aid in ready_sorted:
            for child in children[aid]:
                remaining[child] -= 1
                if remaining[child] == 0:
                    next_ready.append(child)
            del remaining[aid]
        ready = next_ready
    cycle_ids = sorted(remaining.keys())
    return layers, cycle_ids


def _priority_from_severity(severity: str, urgency: int) -> str:
    w = SEVERITY_WEIGHT[severity]
    if w >= 3:  # high/critical
        return "P0"
    if w == 2 or urgency >= 4:
        return "P1"
    if w == 1:
        return "P2"
    return "P3"


def _grade(p0_deferred: int, p0_insufficient: int, deferred: int,
           oversized: int, mean_util: float) -> Tuple[str, str]:
    if p0_insufficient > 0 or p0_deferred > 0:
        return "F", "CRITICAL"
    if deferred >= 3 or oversized > 0:
        return "D", "STRETCHED"
    if mean_util > 95.0:
        return "C", "STRETCHED"
    if mean_util > 80.0:
        return "B", "WATCH"
    return "A", "HEALTHY"


# ── Planner ──────────────────────────────────────────────────────────


class RemediationSprintPlanner:
    """Pack a remediation plan into a finite sprint horizon."""

    def __init__(self, now_fn: Optional[Callable[[], datetime]] = None) -> None:
        self.now_fn = now_fn or (lambda: datetime.now(timezone.utc))

    def plan(
        self,
        actions: Iterable[Any],
        *,
        horizon_sprints: int = DEFAULT_HORIZON_SPRINTS,
        sprint_length_days: int = DEFAULT_SPRINT_LENGTH_DAYS,
        default_capacity_per_owner: float = DEFAULT_CAPACITY_PER_OWNER,
        capacity_per_owner: Optional[Dict[str, float]] = None,
        risk_appetite: str = "balanced",
        start_date: Optional[datetime] = None,
    ) -> SprintPlanReport:
        if risk_appetite not in APPETITES:
            raise ValueError(
                f"risk_appetite must be one of {APPETITES}, got {risk_appetite!r}"
            )
        if horizon_sprints < 1:
            raise ValueError("horizon_sprints must be >= 1")
        if sprint_length_days < 1:
            raise ValueError("sprint_length_days must be >= 1")

        cap_mult = APPETITE_CAPACITY_MULT[risk_appetite]
        cap_override = dict(capacity_per_owner or {})
        # Deep-copy + normalize. Never mutate caller's list/objects.
        raw_list = list(actions)
        raw_list = copy.deepcopy(raw_list)
        norm = [_coerce_action(r) for r in raw_list]

        now = self.now_fn()
        start = start_date or now

        def _owner_cap(owner: str) -> float:
            base = cap_override.get(owner, default_capacity_per_owner)
            return max(0.5, base * cap_mult)

        # Pre-build sprint slots
        sprints: List[SprintSlot] = []
        for i in range(horizon_sprints):
            sstart = start + timedelta(days=i * sprint_length_days)
            send = sstart + timedelta(days=sprint_length_days)
            sprints.append(
                SprintSlot(
                    sprint_index=i,
                    start_date=sstart,
                    end_date=send,
                    total_effort=0.0,
                    capacity=0.0,  # filled per-sprint at the end
                    utilization_pct=0.0,
                    action_ids=[],
                    owner_load={},
                )
            )

        # Per-(sprint, owner) remaining capacity dict
        rem: Dict[Tuple[int, str], float] = {}

        def _rem_cap(sprint_idx: int, owner: str) -> float:
            key = (sprint_idx, owner)
            if key not in rem:
                rem[key] = _owner_cap(owner)
            return rem[key]

        layers, cycle_ids = _topo_layers(norm)
        scheduled_sprint: Dict[str, int] = {}
        deferred_ids: set = set()
        blocked_ids: set = set()
        oversized_ids: set = set()

        assignments: List[SprintAssignment] = []
        id_to_assignment: Dict[str, SprintAssignment] = {}

        # Process layer by layer; within a layer keep deterministic order
        for layer in layers:
            for a in layer:
                base_priority = _priority_from_severity(a.severity, a.urgency)
                # earliest sprint after all deps
                earliest = 0
                blocked_by_outside = False
                for d in a.depends_on:
                    if d in scheduled_sprint:
                        earliest = max(earliest, scheduled_sprint[d] + 1)
                    elif d in deferred_ids or d in blocked_ids or d in oversized_ids:
                        blocked_by_outside = True
                    else:
                        # dep not in plan or in cycle — treat as outside
                        blocked_by_outside = True

                owner = a.owner
                cap_per_sprint = _owner_cap(owner)

                # OVERSIZED?
                if a.effort_days > cap_per_sprint + 1e-9:
                    oversized_ids.add(a.id)
                    asg = SprintAssignment(
                        action_id=a.id,
                        sprint_index=None,
                        owner=owner,
                        effort_days=a.effort_days,
                        severity=a.severity,
                        urgency=a.urgency,
                        verdict="OVERSIZED_ACTION",
                        priority="P1",
                        reasons=[
                            f"effort {a.effort_days:.1f}d exceeds owner sprint cap "
                            f"{cap_per_sprint:.1f}d"
                        ],
                    )
                    assignments.append(asg)
                    id_to_assignment[a.id] = asg
                    continue

                if blocked_by_outside:
                    blocked_ids.add(a.id)
                    asg = SprintAssignment(
                        action_id=a.id,
                        sprint_index=None,
                        owner=owner,
                        effort_days=a.effort_days,
                        severity=a.severity,
                        urgency=a.urgency,
                        verdict="BLOCKED_BY_DEPENDENCY",
                        priority="P0" if base_priority == "P0" else "P2",
                        reasons=["upstream dependency is deferred, blocked, oversized, or missing"],
                    )
                    assignments.append(asg)
                    id_to_assignment[a.id] = asg
                    continue

                placed = False
                for s_idx in range(earliest, horizon_sprints):
                    if _rem_cap(s_idx, owner) + 1e-9 >= a.effort_days:
                        rem[(s_idx, owner)] = _rem_cap(s_idx, owner) - a.effort_days
                        scheduled_sprint[a.id] = s_idx
                        slot = sprints[s_idx]
                        slot.action_ids.append(a.id)
                        slot.total_effort += a.effort_days
                        slot.owner_load[owner] = slot.owner_load.get(owner, 0.0) + a.effort_days
                        if s_idx == 0 and base_priority == "P0":
                            verdict = "READY_NOW"
                            priority = "P0"
                        else:
                            verdict = "SCHEDULED"
                            priority = base_priority
                        asg = SprintAssignment(
                            action_id=a.id,
                            sprint_index=s_idx,
                            owner=owner,
                            effort_days=a.effort_days,
                            severity=a.severity,
                            urgency=a.urgency,
                            verdict=verdict,
                            priority=priority,
                            reasons=[f"placed by greedy packer in sprint {s_idx}"],
                        )
                        assignments.append(asg)
                        id_to_assignment[a.id] = asg
                        placed = True
                        break
                if not placed:
                    deferred_ids.add(a.id)
                    if base_priority == "P0":
                        verdict = "INSUFFICIENT_CAPACITY"
                        priority = "P0"
                        reasons = [
                            "no sprint within horizon has remaining capacity for this P0 action"
                        ]
                    else:
                        verdict = "DEFERRED_PAST_HORIZON"
                        priority = "P1"
                        reasons = [
                            f"horizon of {horizon_sprints} sprints exhausted for owner {owner}"
                        ]
                    asg = SprintAssignment(
                        action_id=a.id,
                        sprint_index=None,
                        owner=owner,
                        effort_days=a.effort_days,
                        severity=a.severity,
                        urgency=a.urgency,
                        verdict=verdict,
                        priority=priority,
                        reasons=reasons,
                    )
                    assignments.append(asg)
                    id_to_assignment[a.id] = asg

        # Cycle nodes: emit as BLOCKED_BY_DEPENDENCY with cycle reason
        for cid in cycle_ids:
            a = next(x for x in norm if x.id == cid)
            base_priority = _priority_from_severity(a.severity, a.urgency)
            asg = SprintAssignment(
                action_id=a.id,
                sprint_index=None,
                owner=a.owner,
                effort_days=a.effort_days,
                severity=a.severity,
                urgency=a.urgency,
                verdict="BLOCKED_BY_DEPENDENCY",
                priority="P0" if base_priority == "P0" else "P2",
                reasons=["cycle in depends_on graph"],
            )
            assignments.append(asg)
            id_to_assignment[a.id] = asg
            blocked_ids.add(a.id)

        # Fill per-sprint capacity now that all assignments are placed.
        for s in sprints:
            owners_in_sprint = set(s.owner_load.keys())
            if not owners_in_sprint:
                # capacity = sum of unique-owner caps that actually exist in any sprint,
                # but at minimum default capacity for "team" representation
                s.capacity = max(default_capacity_per_owner * cap_mult, 0.5)
            else:
                s.capacity = sum(_owner_cap(o) for o in owners_in_sprint)
            s.utilization_pct = (s.total_effort / s.capacity * 100.0) if s.capacity > 0 else 0.0

        # Sort assignments deterministically: priority asc, sprint asc (None last),
        # severity desc, id asc.
        prio_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        assignments.sort(
            key=lambda x: (
                prio_rank.get(x.priority, 9),
                x.sprint_index if x.sprint_index is not None else 9999,
                -SEVERITY_WEIGHT[x.severity],
                x.action_id,
            )
        )

        # Portfolio
        scheduled_count = sum(1 for a in assignments if a.sprint_index is not None)
        deferred_count = sum(
            1 for a in assignments
            if a.verdict in {"DEFERRED_PAST_HORIZON", "INSUFFICIENT_CAPACITY"}
        )
        blocked_count = sum(1 for a in assignments if a.verdict == "BLOCKED_BY_DEPENDENCY")
        oversized_count = sum(1 for a in assignments if a.verdict == "OVERSIZED_ACTION")
        insufficient_count = sum(1 for a in assignments if a.verdict == "INSUFFICIENT_CAPACITY")
        p0_deferred = sum(
            1 for a in assignments
            if a.priority == "P0" and a.verdict in {"DEFERRED_PAST_HORIZON"}
        )
        p0_deferred += insufficient_count  # P0 insufficient counts as P0 deferred

        utils = [s.utilization_pct for s in sprints]
        mean_util = sum(utils) / len(utils) if utils else 0.0
        max_util = max(utils) if utils else 0.0

        # sprints_to_clear_p0: smallest N such that all scheduled P0 actions
        # land in sprints [0..N-1]. None if any P0 deferred/blocked/oversized.
        p0_all_placed = all(
            a.sprint_index is not None
            for a in assignments
            if _priority_from_severity(a.severity, a.urgency) == "P0"
            and a.verdict not in {"BLOCKED_BY_DEPENDENCY"}
        )
        p0_unblocked_total = sum(
            1 for a in assignments
            if _priority_from_severity(a.severity, a.urgency) == "P0"
        )
        p0_problems = sum(
            1 for a in assignments
            if _priority_from_severity(a.severity, a.urgency) == "P0"
            and a.sprint_index is None
        )
        if p0_problems == 0 and p0_unblocked_total > 0:
            sprints_to_clear_p0 = (
                1 + max(
                    a.sprint_index for a in assignments
                    if _priority_from_severity(a.severity, a.urgency) == "P0"
                    and a.sprint_index is not None
                )
            )
        elif p0_unblocked_total == 0:
            sprints_to_clear_p0 = 0
        else:
            sprints_to_clear_p0 = None

        grade, band = _grade(
            p0_deferred=p0_deferred,
            p0_insufficient=insufficient_count,
            deferred=deferred_count,
            oversized=oversized_count,
            mean_util=mean_util,
        )

        portfolio = SprintPortfolio(
            total_actions=len(assignments),
            scheduled=scheduled_count,
            deferred=deferred_count,
            blocked=blocked_count,
            oversized=oversized_count,
            insufficient=insufficient_count,
            mean_utilization=round(mean_util, 2),
            max_utilization=round(max_util, 2),
            sprints_to_clear_p0=sprints_to_clear_p0,
            grade=grade,
            concentration_band=band,
        )

        playbook = self._build_playbook(
            assignments=assignments,
            sprints=sprints,
            insufficient_count=insufficient_count,
            oversized_count=oversized_count,
            deferred_count=deferred_count,
            blocked_count=blocked_count,
            risk_appetite=risk_appetite,
            grade=grade,
            horizon_sprints=horizon_sprints,
        )

        insights = self._build_insights(
            assignments=assignments,
            sprints=sprints,
            layers=layers,
        )
        if not assignments:
            insights = ["EMPTY_PLAN"]

        return SprintPlanReport(
            generated_at=now,
            risk_appetite=risk_appetite,
            horizon_sprints=horizon_sprints,
            sprint_length_days=sprint_length_days,
            portfolio=portfolio,
            sprints=sprints,
            assignments=assignments,
            playbook=playbook,
            insights=insights,
        )

    # ── Playbook ────────────────────────────────────────────────────

    def _build_playbook(
        self,
        *,
        assignments: List[SprintAssignment],
        sprints: List[SprintSlot],
        insufficient_count: int,
        oversized_count: int,
        deferred_count: int,
        blocked_count: int,
        risk_appetite: str,
        grade: str,
        horizon_sprints: int,
    ) -> List[SprintPlaybookAction]:

        actions: List[SprintPlaybookAction] = []

        p0_blocked_ids = [
            a.action_id for a in assignments
            if a.verdict == "BLOCKED_BY_DEPENDENCY" and a.priority == "P0"
        ]
        deferred_or_insuff_ids = [
            a.action_id for a in assignments
            if a.verdict in {"DEFERRED_PAST_HORIZON", "INSUFFICIENT_CAPACITY"}
        ]
        oversized_ids = [
            a.action_id for a in assignments if a.verdict == "OVERSIZED_ACTION"
        ]
        unassigned_ids = [
            a.action_id for a in assignments if a.owner == UNASSIGNED_OWNER
        ]

        if insufficient_count > 0 or any(
            a.priority == "P0" and a.verdict == "DEFERRED_PAST_HORIZON" for a in assignments
        ):
            actions.append(SprintPlaybookAction(
                id="INCREASE_CAPACITY_OR_DEFER_SCOPE",
                priority="P0",
                label="Increase sprint capacity or defer scope to clear P0 backlog",
                reason=(
                    f"{insufficient_count} P0 action(s) could not be packed inside "
                    f"the {horizon_sprints}-sprint horizon"
                ),
                owner="engineering_lead",
                blast_radius=4,
                reversibility="medium",
                related_action_ids=[
                    a.action_id for a in assignments
                    if a.priority == "P0" and a.sprint_index is None
                ],
            ))

        if oversized_ids:
            actions.append(SprintPlaybookAction(
                id="SPLIT_OVERSIZED_ACTIONS",
                priority="P0",
                label="Split oversized actions into smaller increments",
                reason=(
                    f"{len(oversized_ids)} action(s) exceed a single sprint's owner capacity "
                    "and cannot be packed as-is"
                ),
                owner="tech_lead",
                blast_radius=2,
                reversibility="high",
                related_action_ids=oversized_ids,
            ))

        if p0_blocked_ids:
            actions.append(SprintPlaybookAction(
                id="UNBLOCK_DEPENDENCY_CHAIN",
                priority="P0",
                label="Unblock upstream dependencies for P0 actions",
                reason=(
                    f"{len(p0_blocked_ids)} P0 action(s) are gated by deferred, blocked, "
                    "or missing dependencies"
                ),
                owner="program_manager",
                blast_radius=3,
                reversibility="medium",
                related_action_ids=p0_blocked_ids,
            ))

        # Rebalance owner load: any owner > 90% util in sprint 0 while another < 50%
        if sprints:
            s0 = sprints[0]
            high_owners = [
                o for o, v in s0.owner_load.items()
                if (v / max(s0.capacity, 1.0)) >= 0.45 and len(s0.owner_load) >= 2
            ]
            low_owners = [
                o for o, v in s0.owner_load.items()
                if (v / max(s0.capacity, 1.0)) < 0.10
            ]
            if high_owners and low_owners:
                actions.append(SprintPlaybookAction(
                    id="REBALANCE_OWNER_LOAD",
                    priority="P1",
                    label="Rebalance owner load in sprint 0",
                    reason=(
                        f"owners {sorted(high_owners)} are heavily loaded while "
                        f"{sorted(low_owners)} have slack"
                    ),
                    owner="program_manager",
                    blast_radius=2,
                    reversibility="high",
                ))

        # SPILL_LOW_PRIORITY_TO_LATER: sprint 0 overpacked AND has P2/P3 work
        if sprints and sprints[0].total_effort > sprints[0].capacity * 0.95:
            low_in_s0 = [
                a.action_id for a in assignments
                if a.sprint_index == 0 and a.priority in {"P2", "P3"}
            ]
            if low_in_s0:
                actions.append(SprintPlaybookAction(
                    id="SPILL_LOW_PRIORITY_TO_LATER",
                    priority="P1",
                    label="Spill low-priority work in sprint 0 into later sprints",
                    reason=(
                        "sprint 0 is over-packed; lower-priority work can wait without "
                        "moving the P0 line"
                    ),
                    owner="scrum_master",
                    blast_radius=2,
                    reversibility="high",
                    related_action_ids=low_in_s0,
                ))

        if unassigned_ids:
            actions.append(SprintPlaybookAction(
                id="ASSIGN_OWNER_TO_UNASSIGNED_WORK",
                priority="P1",
                label="Assign explicit owners to unassigned actions",
                reason=(
                    f"{len(unassigned_ids)} action(s) currently land in the "
                    f"'{UNASSIGNED_OWNER}' bucket"
                ),
                owner="program_manager",
                blast_radius=2,
                reversibility="high",
                related_action_ids=unassigned_ids,
            ))

        if any(a.verdict == "DEFERRED_PAST_HORIZON" for a in assignments):
            new_horizon = horizon_sprints + max(
                1, sum(1 for a in assignments if a.verdict == "DEFERRED_PAST_HORIZON") // 3
            )
            actions.append(SprintPlaybookAction(
                id="EXPAND_HORIZON",
                priority="P2",
                label="Expand planning horizon to land deferred work",
                reason=(
                    "non-P0 work was pushed past the configured horizon; widening the "
                    "plan keeps everything visible"
                ),
                owner="program_manager",
                blast_radius=1,
                reversibility="high",
                suggested_value=str(new_horizon),
                related_action_ids=deferred_or_insuff_ids,
            ))

        if risk_appetite == "cautious" and grade in {"C", "D", "F"}:
            actions.append(SprintPlaybookAction(
                id="SCHEDULE_SPRINT_REVIEW",
                priority="P2",
                label="Schedule a sprint-plan review with leadership",
                reason="cautious appetite + degraded portfolio grade",
                owner="engineering_lead",
                blast_radius=1,
                reversibility="high",
            ))

        # Dedup by id, keep first (highest priority wins from earlier ordering)
        dedup: Dict[str, SprintPlaybookAction] = {}
        for a in actions:
            dedup.setdefault(a.id, a)
        ordered = list(dedup.values())

        prio_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        ordered.sort(key=lambda x: (prio_rank.get(x.priority, 9), x.id))

        if not ordered:
            ordered.append(SprintPlaybookAction(
                id="SPRINT_PLAN_OK",
                priority="P3",
                label="Sprint plan packs cleanly — no intervention needed",
                reason="all actions scheduled within horizon and capacity",
                owner="scrum_master",
                blast_radius=1,
                reversibility="high",
            ))

        if risk_appetite == "aggressive" and any(
            a.priority in {"P0", "P1"} for a in ordered
        ):
            # Trim P3 + lone P2 when P0/P1 present
            ordered = [a for a in ordered if a.priority != "P3"]
            p2s = [a for a in ordered if a.priority == "P2"]
            if len(p2s) == 1:
                ordered = [a for a in ordered if a.priority != "P2"]

        return ordered

    # ── Insights ────────────────────────────────────────────────────

    def _build_insights(
        self,
        *,
        assignments: List[SprintAssignment],
        sprints: List[SprintSlot],
        layers: List[List[_NormAction]],
    ) -> List[str]:
        out: List[str] = []
        if sprints and sprints[0].utilization_pct > 100.0:
            out.append("OVERCOMMITTED_SPRINT_0")
        if len(sprints) >= 2:
            tail = sprints[len(sprints) // 2:]
            if tail and all(s.utilization_pct < 25.0 for s in tail) and any(
                a.verdict in {"DEFERRED_PAST_HORIZON", "INSUFFICIENT_CAPACITY"}
                for a in assignments
            ):
                out.append("IDLE_CAPACITY_LATE_SPRINTS")
        if sprints:
            s0 = sprints[0]
            cap = max(s0.capacity, 1.0)
            for owner, load in sorted(s0.owner_load.items()):
                owner_cap = APPETITE_CAPACITY_MULT["balanced"]  # unused, real cap from below
                # Estimate per-owner cap as load/util_target; use a simpler test:
                # owner_load / (cap / number_of_owners) ratio.
                if len(s0.owner_load) >= 1 and load / cap >= 0.90 / len(s0.owner_load):
                    if load / cap >= 0.45 and len(s0.owner_load) >= 2:
                        out.append(f"OWNER_HOTSPOT:{owner}")
                        break
        if layers and len(layers) >= 3:
            out.append("DEEP_DEPENDENCY_CHAIN")
        if not out and assignments:
            out.append("WELL_BALANCED_PLAN")
        if not assignments:
            out.append("EMPTY_PLAN")
        # Dedup while preserving order
        seen = set()
        dedup_out: List[str] = []
        for ins in out:
            if ins not in seen:
                seen.add(ins)
                dedup_out.append(ins)
        return dedup_out


# ── Demo / CLI ───────────────────────────────────────────────────────


def _demo_actions() -> List[Dict[str, Any]]:
    return [
        {"id": "fix-auth-bypass", "title": "Patch auth bypass",
         "severity": "critical", "effort": 4, "urgency": 5,
         "depends_on": [], "owner_hint": "alice"},
        {"id": "rotate-keys", "title": "Rotate signing keys",
         "severity": "high", "effort": 2, "urgency": 4,
         "depends_on": ["fix-auth-bypass"], "owner_hint": "alice"},
        {"id": "tighten-rbac", "title": "Tighten RBAC roles",
         "severity": "high", "effort": 5, "urgency": 4,
         "depends_on": [], "owner_hint": "bob"},
        {"id": "log-redaction", "title": "Add log redaction",
         "severity": "medium", "effort": 3, "urgency": 3,
         "depends_on": [], "owner_hint": "carol"},
        {"id": "incident-drill", "title": "Run incident drill",
         "severity": "medium", "effort": 4, "urgency": 3,
         "depends_on": ["rotate-keys", "tighten-rbac"], "owner_hint": "bob"},
        {"id": "audit-trail", "title": "Backfill audit trail",
         "severity": "low", "effort": 2, "urgency": 2,
         "depends_on": [], "owner_hint": "carol"},
        {"id": "training-refresh", "title": "Quarterly safety training",
         "severity": "low", "effort": 1, "urgency": 2,
         "depends_on": [], "owner_hint": ""},
    ]


def _ensure_utf8() -> None:
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main(argv: Optional[List[str]] = None) -> int:
    _ensure_utf8()
    ap = argparse.ArgumentParser(description="Remediation Sprint Planner")
    ap.add_argument("--demo", action="store_true",
                    help="Use built-in demo action list")
    ap.add_argument("--from-plan", type=str, default=None,
                    help="Path to a RemediationPlan JSON file (with .actions[])")
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON_SPRINTS)
    ap.add_argument("--sprint-days", type=int, default=DEFAULT_SPRINT_LENGTH_DAYS)
    ap.add_argument("--capacity", type=float, default=DEFAULT_CAPACITY_PER_OWNER,
                    help="Default capacity per owner per sprint (person-days)")
    ap.add_argument("--risk", choices=APPETITES, default="balanced")
    ap.add_argument("--format", choices=("text", "markdown", "md", "json"),
                    default="text")
    args = ap.parse_args(argv)

    if args.from_plan:
        with open(args.from_plan, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if isinstance(raw, dict) and "actions" in raw:
            raw_actions = raw["actions"]
        elif isinstance(raw, list):
            raw_actions = raw
        else:
            print("ERROR: --from-plan must point to a list or {actions:[...]}",
                  file=sys.stderr)
            return 2
    elif args.demo:
        raw_actions = _demo_actions()
    else:
        print("ERROR: pass --demo or --from-plan", file=sys.stderr)
        return 2

    planner = RemediationSprintPlanner()
    report = planner.plan(
        raw_actions,
        horizon_sprints=args.horizon,
        sprint_length_days=args.sprint_days,
        default_capacity_per_owner=args.capacity,
        risk_appetite=args.risk,
    )

    fmt = args.format
    if fmt in ("md", "markdown"):
        print(report.to_markdown())
    elif fmt == "json":
        print(report.to_json())
    else:
        print(report.to_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
