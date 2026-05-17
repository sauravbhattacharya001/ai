"""Remediation Progress Tracker - agentic plan-vs-plan diff with velocity.

Companion to :mod:`replication.remediation_planner`.  Where the planner
answers *"what should we fix?"*, this module answers *"are we actually
fixing it — and when will we be done?"*

Given two snapshots of a :class:`~replication.remediation_planner.RemediationPlan`
(or raw :class:`~replication.remediation_planner.Finding` lists) — typically
"previous run" vs "current run" — the tracker autonomously:

1. **Diffs** every action across snapshots and classifies it as
   ``resolved``, ``new``, ``persisting``, ``regressed``, ``improving``,
   or ``slipping`` (a persisting action whose age has exceeded the SLA
   threshold for its severity).
2. **Computes velocity** — resolutions/day, net change, and a
   severity-weighted velocity — and projects ``days_to_zero`` (ETA to
   green) when the team is actually closing more than they open.
3. **Classifies trajectory** as ``improving``, ``stable``, ``regressing``,
   or ``at_risk``.
4. **Recommends** a small, severity-tagged playbook (P0/P1/P2) of next
   moves — escalate, break up stuck work, reassign owner, add to weekly
   review, raise alarm, etc.

CLI usage::

    # Compare a synthetic before/after pair (great first run)
    python -m replication progress --demo
    python -m replication progress --demo --format md
    python -m replication progress --demo --format json

    # Compare two saved plan or findings JSON files (7 days apart)
    python -m replication progress --previous prev.json --current curr.json \\
        --days-between 7 --format md --output progress.md

Programmatic::

    from replication.remediation_planner import RemediationPlanner
    from replication.remediation_progress import RemediationProgressTracker

    prev = RemediationPlanner().plan_from_findings(prev_findings)
    curr = RemediationPlanner().plan_from_findings(curr_findings)

    tracker = RemediationProgressTracker()
    report  = tracker.compare(prev, curr, days_between=7.0)
    print(report.render())
    print(report.to_markdown())
    print(report.to_json())
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from .remediation_planner import (
    Finding,
    RemediationAction,
    RemediationPlan,
    RemediationPlanner,
)


# ── Constants ────────────────────────────────────────────────────────


DiffStatus = str  # one of: resolved | new | persisting | regressed | improving | slipping


SEVERITY_ORDER: Dict[str, int] = {
    "info": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


SEVERITY_WEIGHT: Dict[str, float] = {
    "critical": 5.0,
    "high": 3.0,
    "medium": 2.0,
    "low": 1.0,
    "info": 1.0,
}


# Max days an open action of a given severity is acceptable before it is
# flagged as "slipping" and an agentic recommendation is issued.
SLA_DAYS_BY_SEVERITY: Dict[str, float] = {
    "critical": 3.0,
    "high": 7.0,
    "medium": 14.0,
    "low": 30.0,
    "info": 60.0,
}


# ── Data model ───────────────────────────────────────────────────────


@dataclass
class ActionDiff:
    """Per-action diff entry."""

    action_id: str
    status: DiffStatus
    title: str
    previous_severity: Optional[str] = None
    current_severity: Optional[str] = None
    age_days: Optional[float] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "status": self.status,
            "title": self.title,
            "previous_severity": self.previous_severity,
            "current_severity": self.current_severity,
            "age_days": self.age_days,
            "reason": self.reason,
        }


@dataclass
class ProgressVelocity:
    """Velocity / ETA estimates derived from the diff."""

    days_between: float
    resolved_count: int
    new_count: int
    persisting_count: int
    regressed_count: int
    improving_count: int
    slipping_count: int

    resolutions_per_day: float
    new_per_day: float
    net_change_per_day: float                       # new - resolved (per day)
    weighted_net_change_per_day: float              # severity-weighted

    projected_days_to_zero: Optional[float]         # None if regressing/stagnant
    remaining_actions: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "days_between": self.days_between,
            "resolved_count": self.resolved_count,
            "new_count": self.new_count,
            "persisting_count": self.persisting_count,
            "regressed_count": self.regressed_count,
            "improving_count": self.improving_count,
            "slipping_count": self.slipping_count,
            "resolutions_per_day": self.resolutions_per_day,
            "new_per_day": self.new_per_day,
            "net_change_per_day": self.net_change_per_day,
            "weighted_net_change_per_day": self.weighted_net_change_per_day,
            "projected_days_to_zero": self.projected_days_to_zero,
            "remaining_actions": self.remaining_actions,
        }


@dataclass
class Recommendation:
    """One agentic recommendation derived from the diff."""

    priority: str       # "P0" | "P1" | "P2"
    title: str
    why: str
    next_step: str
    action_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "priority": self.priority,
            "title": self.title,
            "why": self.why,
            "next_step": self.next_step,
            "action_id": self.action_id,
        }


@dataclass
class ProgressReport:
    """Full diff + velocity + recommendations + renderers."""

    timestamp: str
    trajectory: str  # improving | stable | regressing | at_risk
    diffs: List[ActionDiff] = field(default_factory=list)
    velocity: Optional[ProgressVelocity] = None
    recommendations: List[Recommendation] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    # ── helpers ──────────────────────────────────────────────────────

    def _by_status(self, status: DiffStatus) -> List[ActionDiff]:
        return [d for d in self.diffs if d.status == status]

    # ── exporters ────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "trajectory": self.trajectory,
            "velocity": self.velocity.to_dict() if self.velocity else None,
            "diffs": [d.to_dict() for d in self.diffs],
            "recommendations": [r.to_dict() for r in self.recommendations],
            "notes": list(self.notes),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    def to_markdown(self) -> str:
        v = self.velocity
        lines: List[str] = []
        lines.append("# Remediation Progress Report")
        lines.append("")
        lines.append(f"_Generated: {self.timestamp}_")
        lines.append("")
        traj_emoji = {
            "improving": "📈",
            "stable": "➖",
            "regressing": "📉",
            "at_risk": "🚨",
        }.get(self.trajectory, "•")
        lines.append(f"**Trajectory:** {traj_emoji} `{self.trajectory}`")
        if v is not None:
            eta = (
                f"{v.projected_days_to_zero:.1f}d"
                if v.projected_days_to_zero is not None
                else "—"
            )
            lines.append("")
            lines.append("## 📊 Velocity")
            lines.append("")
            lines.append(
                f"- **Window:** {v.days_between:.2f}d  ·  "
                f"**Resolved:** {v.resolved_count}  ·  **New:** {v.new_count}  ·  "
                f"**Persisting:** {v.persisting_count}"
            )
            lines.append(
                f"- **Regressed:** {v.regressed_count}  ·  "
                f"**Improving:** {v.improving_count}  ·  "
                f"**Slipping:** {v.slipping_count}"
            )
            lines.append(
                f"- **Rate:** {v.resolutions_per_day:.2f} resolved/day  ·  "
                f"{v.new_per_day:.2f} new/day  ·  "
                f"net {v.net_change_per_day:+.2f}/day  ·  "
                f"weighted net {v.weighted_net_change_per_day:+.2f}/day"
            )
            lines.append(
                f"- **Remaining:** {v.remaining_actions}  ·  "
                f"**Projected days-to-zero:** {eta}"
            )
        lines.append("")

        if self.notes:
            lines.append("## Notes")
            for n in self.notes:
                lines.append(f"- {n}")
            lines.append("")

        sections = [
            ("✅ Resolved", "resolved"),
            ("🔥 Regressed", "regressed"),
            ("🟢 Improving", "improving"),
            ("⚠️ Slipping", "slipping"),
            ("🆕 New", "new"),
            ("📌 Persisting", "persisting"),
        ]
        for heading, status in sections:
            entries = self._by_status(status)
            lines.append(f"## {heading}")
            lines.append("")
            if not entries:
                lines.append("_None._")
                lines.append("")
                continue
            for d in entries:
                bits = [f"**`{d.action_id}`** — {d.title}"]
                if d.current_severity or d.previous_severity:
                    sev_bit = (
                        f"sev: {d.previous_severity or '—'} → {d.current_severity or '—'}"
                    )
                    bits.append(sev_bit)
                if d.age_days is not None:
                    bits.append(f"age: {d.age_days:.1f}d")
                if d.reason:
                    bits.append(d.reason)
                lines.append("- " + "  ·  ".join(bits))
            lines.append("")

        lines.append("## 🤖 Agentic Recommendations")
        lines.append("")
        if not self.recommendations:
            lines.append("_No follow-ups — keep the cadence._")
            lines.append("")
        else:
            for prio in ("P0", "P1", "P2"):
                bucket = [r for r in self.recommendations if r.priority == prio]
                if not bucket:
                    continue
                lines.append(f"### {prio}")
                lines.append("")
                for r in bucket:
                    aid = f" (`{r.action_id}`)" if r.action_id else ""
                    lines.append(f"- **{r.title}**{aid}")
                    lines.append(f"  - _Why:_ {r.why}")
                    lines.append(f"  - _Next:_ {r.next_step}")
                lines.append("")
        return "\n".join(lines)

    def to_text(self) -> str:
        v = self.velocity
        bar = "─" * 60
        lines: List[str] = []
        lines.append(bar)
        lines.append(" REMEDIATION PROGRESS REPORT")
        lines.append(bar)
        lines.append(f" Generated:   {self.timestamp}")
        lines.append(f" TRAJECTORY:  {self.trajectory.upper()}")
        if v is not None:
            eta = (
                f"{v.projected_days_to_zero:.1f}d"
                if v.projected_days_to_zero is not None
                else "n/a"
            )
            lines.append(f" Window:      {v.days_between:.2f}d")
            lines.append(
                f" Counts:      resolved={v.resolved_count}  new={v.new_count}  "
                f"persisting={v.persisting_count}"
            )
            lines.append(
                f"              regressed={v.regressed_count}  "
                f"improving={v.improving_count}  slipping={v.slipping_count}"
            )
            lines.append(
                f" Velocity:    {v.resolutions_per_day:.2f}/day resolved  ·  "
                f"net {v.net_change_per_day:+.2f}/day"
            )
            lines.append(
                f" Remaining:   {v.remaining_actions}  ·  ETA-to-zero: {eta}"
            )
        lines.append(bar)
        order = ["resolved", "regressed", "improving", "slipping", "new", "persisting"]
        labels = {
            "resolved": "RESOLVED",
            "regressed": "REGRESSED",
            "improving": "IMPROVING",
            "slipping": "SLIPPING",
            "new": "NEW",
            "persisting": "PERSISTING",
        }
        for status in order:
            entries = self._by_status(status)
            if not entries:
                continue
            lines.append(f" [{labels[status]}] ({len(entries)})")
            for d in entries:
                sev = ""
                if d.previous_severity or d.current_severity:
                    sev = (
                        f" sev {d.previous_severity or '—'}→"
                        f"{d.current_severity or '—'}"
                    )
                age = f" age={d.age_days:.1f}d" if d.age_days is not None else ""
                lines.append(f"   - {d.action_id}: {d.title}{sev}{age}")
                if d.reason:
                    lines.append(f"       {d.reason}")
            lines.append("")
        if self.recommendations:
            lines.append(" AGENTIC RECOMMENDATIONS")
            for r in self.recommendations:
                aid = f" [{r.action_id}]" if r.action_id else ""
                lines.append(f"   {r.priority}  {r.title}{aid}")
                lines.append(f"        why:  {r.why}")
                lines.append(f"        next: {r.next_step}")
        else:
            lines.append(" No follow-ups required.")
        lines.append(bar)
        return "\n".join(lines)

    def render(self) -> str:
        """Alias for :meth:`to_text`."""
        return self.to_text()


# ── Tracker ──────────────────────────────────────────────────────────


class RemediationProgressTracker:
    """Diff two remediation snapshots and project velocity/recommendations."""

    def __init__(self, sla_days: Optional[Dict[str, float]] = None) -> None:
        self.sla_days: Dict[str, float] = dict(SLA_DAYS_BY_SEVERITY)
        if sla_days:
            for k, v in sla_days.items():
                self.sla_days[k.lower()] = float(v)

    # ── public API ───────────────────────────────────────────────────

    def compare(
        self,
        previous: RemediationPlan,
        current: RemediationPlan,
        days_between: float = 1.0,
        current_age_days: Optional[Dict[str, float]] = None,
    ) -> ProgressReport:
        days_between = max(float(days_between), 0.0)
        age_map: Dict[str, float] = dict(current_age_days or {})

        prev_actions: Dict[str, RemediationAction] = {a.id: a for a in previous.actions}
        curr_actions: Dict[str, RemediationAction] = {a.id: a for a in current.actions}

        diffs: List[ActionDiff] = []

        # resolved: present before, absent now
        for aid, pa in prev_actions.items():
            if aid not in curr_actions:
                diffs.append(
                    ActionDiff(
                        action_id=aid,
                        status="resolved",
                        title=pa.title,
                        previous_severity=pa.severity,
                        current_severity=None,
                        reason="finding no longer present in current snapshot",
                    )
                )

        # new: absent before, present now
        for aid, ca in curr_actions.items():
            if aid not in prev_actions:
                age = age_map.get(aid)
                diffs.append(
                    ActionDiff(
                        action_id=aid,
                        status="new",
                        title=ca.title,
                        previous_severity=None,
                        current_severity=ca.severity,
                        age_days=age,
                        reason="newly opened since previous snapshot",
                    )
                )

        # persisting / regressed / improving / slipping
        for aid, ca in curr_actions.items():
            if aid not in prev_actions:
                continue
            pa = prev_actions[aid]
            prev_rank = SEVERITY_ORDER.get(pa.severity, 0)
            curr_rank = SEVERITY_ORDER.get(ca.severity, 0)
            age = age_map.get(aid)

            if curr_rank > prev_rank:
                diffs.append(
                    ActionDiff(
                        action_id=aid,
                        status="regressed",
                        title=ca.title,
                        previous_severity=pa.severity,
                        current_severity=ca.severity,
                        age_days=age,
                        reason=f"severity worsened {pa.severity} → {ca.severity}",
                    )
                )
                continue
            if curr_rank < prev_rank:
                diffs.append(
                    ActionDiff(
                        action_id=aid,
                        status="improving",
                        title=ca.title,
                        previous_severity=pa.severity,
                        current_severity=ca.severity,
                        age_days=age,
                        reason=f"severity reduced {pa.severity} → {ca.severity}",
                    )
                )
                continue

            # Same severity ⇒ persisting or slipping
            sla = self.sla_days.get(ca.severity, 14.0)
            if age is not None and age > sla:
                diffs.append(
                    ActionDiff(
                        action_id=aid,
                        status="slipping",
                        title=ca.title,
                        previous_severity=pa.severity,
                        current_severity=ca.severity,
                        age_days=age,
                        reason=(
                            f"open {age:.1f}d > SLA {sla:.0f}d "
                            f"for severity '{ca.severity}'"
                        ),
                    )
                )
            else:
                diffs.append(
                    ActionDiff(
                        action_id=aid,
                        status="persisting",
                        title=ca.title,
                        previous_severity=pa.severity,
                        current_severity=ca.severity,
                        age_days=age,
                        reason="still open, severity unchanged",
                    )
                )

        velocity = self._compute_velocity(diffs, days_between, curr_actions)
        trajectory = self._classify_trajectory(diffs, velocity)
        recommendations = self._recommend(diffs, velocity, trajectory)

        notes: List[str] = []
        if not previous.actions and not current.actions:
            notes.append(
                "Both snapshots are empty — nothing to compare. Run the planner first."
            )
        elif not current.actions:
            notes.append("Current plan is empty — all previous findings cleared. 🎉")

        return ProgressReport(
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            trajectory=trajectory,
            diffs=diffs,
            velocity=velocity,
            recommendations=recommendations,
            notes=notes,
        )

    def compare_findings(
        self,
        previous_findings: Iterable[Finding],
        current_findings: Iterable[Finding],
        days_between: float = 1.0,
        current_age_days: Optional[Dict[str, float]] = None,
    ) -> ProgressReport:
        planner = RemediationPlanner()
        prev_plan = planner.plan_from_findings(list(previous_findings))
        curr_plan = planner.plan_from_findings(list(current_findings))
        return self.compare(
            prev_plan,
            curr_plan,
            days_between=days_between,
            current_age_days=current_age_days,
        )

    # ── internals ────────────────────────────────────────────────────

    def _compute_velocity(
        self,
        diffs: List[ActionDiff],
        days_between: float,
        curr_actions: Dict[str, RemediationAction],
    ) -> ProgressVelocity:
        counts: Dict[str, int] = {
            "resolved": 0,
            "new": 0,
            "persisting": 0,
            "regressed": 0,
            "improving": 0,
            "slipping": 0,
        }
        weighted_resolved = 0.0
        weighted_new = 0.0
        for d in diffs:
            counts[d.status] = counts.get(d.status, 0) + 1
            if d.status == "resolved":
                weighted_resolved += SEVERITY_WEIGHT.get(d.previous_severity or "low", 1.0)
            elif d.status == "new":
                weighted_new += SEVERITY_WEIGHT.get(d.current_severity or "low", 1.0)

        denom = max(days_between, 1e-9) if days_between > 0 else 1.0
        # If days_between is 0, treat per-day rates as 0 to avoid div explosions.
        if days_between <= 0:
            resolutions_per_day = 0.0
            new_per_day = 0.0
            net_per_day = 0.0
            weighted_net_per_day = 0.0
        else:
            resolutions_per_day = counts["resolved"] / denom
            new_per_day = counts["new"] / denom
            net_per_day = (counts["new"] - counts["resolved"]) / denom
            weighted_net_per_day = (weighted_new - weighted_resolved) / denom

        remaining = len(curr_actions)
        projected: Optional[float] = None
        if net_per_day < 0 and remaining > 0:
            burn_per_day = -net_per_day
            projected = round(remaining / burn_per_day, 2)

        return ProgressVelocity(
            days_between=days_between,
            resolved_count=counts["resolved"],
            new_count=counts["new"],
            persisting_count=counts["persisting"],
            regressed_count=counts["regressed"],
            improving_count=counts["improving"],
            slipping_count=counts["slipping"],
            resolutions_per_day=round(resolutions_per_day, 3),
            new_per_day=round(new_per_day, 3),
            net_change_per_day=round(net_per_day, 3),
            weighted_net_change_per_day=round(weighted_net_per_day, 3),
            projected_days_to_zero=projected,
            remaining_actions=remaining,
        )

    def _classify_trajectory(
        self,
        diffs: List[ActionDiff],
        velocity: ProgressVelocity,
    ) -> str:
        critical_regressions = sum(
            1
            for d in diffs
            if d.status == "regressed" and d.current_severity == "critical"
        )
        new_criticals = sum(
            1 for d in diffs if d.status == "new" and d.current_severity == "critical"
        )
        slipping_criticals = sum(
            1 for d in diffs if d.status == "slipping" and d.current_severity == "critical"
        )

        if slipping_criticals >= 1 or velocity.slipping_count >= 3:
            return "at_risk"
        if (
            velocity.new_count > velocity.resolved_count
            or critical_regressions
            or new_criticals
        ):
            return "regressing"
        if velocity.resolved_count > velocity.new_count and critical_regressions == 0:
            return "improving"
        return "stable"

    def _recommend(
        self,
        diffs: List[ActionDiff],
        velocity: ProgressVelocity,
        trajectory: str,
    ) -> List[Recommendation]:
        recs: List[Recommendation] = []

        # P0 — slipping criticals, new criticals, regressed-to-critical
        for d in diffs:
            if (
                d.status == "slipping"
                and d.current_severity == "critical"
            ):
                recs.append(
                    Recommendation(
                        priority="P0",
                        title=f"Escalate slipping critical: {d.title}",
                        why=(
                            f"Open {d.age_days:.1f}d (SLA "
                            f"{self.sla_days['critical']:.0f}d). Critical "
                            f"findings should never breach SLA."
                        ),
                        next_step=(
                            "Page on-call safety lead, freeze related "
                            "feature work until closed."
                        ),
                        action_id=d.action_id,
                    )
                )
            elif d.status == "new" and d.current_severity == "critical":
                recs.append(
                    Recommendation(
                        priority="P0",
                        title=f"Triage new critical: {d.title}",
                        why="A critical-severity finding appeared since the last snapshot.",
                        next_step=(
                            "Open an incident, assign a primary owner, "
                            "and confirm containment within 24h."
                        ),
                        action_id=d.action_id,
                    )
                )
            elif d.status == "regressed" and d.current_severity == "critical":
                recs.append(
                    Recommendation(
                        priority="P0",
                        title=f"Investigate critical regression: {d.title}",
                        why=(
                            f"Severity worsened "
                            f"{d.previous_severity} → {d.current_severity}."
                        ),
                        next_step=(
                            "Bisect recent merges, identify the change "
                            "that re-broke this control, and roll back "
                            "or hot-fix."
                        ),
                        action_id=d.action_id,
                    )
                )

        # P1 — slipping high or new high
        for d in diffs:
            if d.status == "slipping" and d.current_severity == "high":
                recs.append(
                    Recommendation(
                        priority="P1",
                        title=f"Re-prioritize slipping high: {d.title}",
                        why=(
                            f"Open {d.age_days:.1f}d (SLA "
                            f"{self.sla_days['high']:.0f}d) with no progress."
                        ),
                        next_step=(
                            "Add to next sprint as committed; consider "
                            "splitting into smaller deliverables."
                        ),
                        action_id=d.action_id,
                    )
                )
            elif d.status == "new" and d.current_severity == "high":
                recs.append(
                    Recommendation(
                        priority="P1",
                        title=f"Schedule new high: {d.title}",
                        why="High-severity finding appeared since last snapshot.",
                        next_step=(
                            "Assign owner this week; aim to close within "
                            f"{self.sla_days['high']:.0f}d."
                        ),
                        action_id=d.action_id,
                    )
                )
            elif d.status == "regressed" and d.current_severity == "high":
                recs.append(
                    Recommendation(
                        priority="P1",
                        title=f"Investigate high regression: {d.title}",
                        why=(
                            f"Severity worsened "
                            f"{d.previous_severity} → {d.current_severity}."
                        ),
                        next_step="Re-open the original fix ticket and add a regression test.",
                        action_id=d.action_id,
                    )
                )

        # P2 — stale medium/low and velocity warnings
        for d in diffs:
            if d.status == "slipping" and d.current_severity in ("medium", "low"):
                recs.append(
                    Recommendation(
                        priority="P2",
                        title=f"Refresh stale {d.current_severity}: {d.title}",
                        why=(
                            f"Open {d.age_days:.1f}d (SLA "
                            f"{self.sla_days[d.current_severity]:.0f}d); risks "
                            f"silent decay."
                        ),
                        next_step=(
                            "Add to weekly safety review; re-confirm "
                            "owner and target date."
                        ),
                        action_id=d.action_id,
                    )
                )

        if velocity.days_between > 0 and velocity.net_change_per_day > 0:
            recs.append(
                Recommendation(
                    priority="P1",
                    title="Increase remediation capacity",
                    why=(
                        f"Net change {velocity.net_change_per_day:+.2f}/day — "
                        f"the team is opening findings faster than closing them."
                    ),
                    next_step=(
                        "Add at least one additional safety-eng to the "
                        "rotation this week, or pause new investigations "
                        "until backlog stabilizes."
                    ),
                )
            )
        elif (
            velocity.days_between > 0
            and velocity.resolutions_per_day == 0
            and velocity.remaining_actions > 0
            and velocity.regressed_count == 0
        ):
            recs.append(
                Recommendation(
                    priority="P2",
                    title="Restart cadence — no closures in window",
                    why=(
                        f"{velocity.remaining_actions} open action(s), "
                        f"zero closed over {velocity.days_between:.1f}d."
                    ),
                    next_step=(
                        "Hold a 30-min unblock session: identify the "
                        "top blocker per stalled action."
                    ),
                )
            )

        if trajectory == "at_risk" and not any(r.priority == "P0" for r in recs):
            recs.append(
                Recommendation(
                    priority="P0",
                    title="Escalate at-risk remediation portfolio",
                    why=(
                        f"{velocity.slipping_count} action(s) slipping SLA; "
                        f"portfolio is at risk of cascading breach."
                    ),
                    next_step=(
                        "Convene a 24h tiger-team review with safety lead "
                        "and engineering manager."
                    ),
                )
            )

        # Sort: P0 > P1 > P2, stable within bucket.
        prio_rank = {"P0": 0, "P1": 1, "P2": 2}
        recs.sort(key=lambda r: prio_rank.get(r.priority, 9))
        return recs


# ── Demo snapshots ───────────────────────────────────────────────────


def _demo_previous_findings() -> List[Finding]:
    return [
        Finding(source="quick_scan", name="preflight", status="fail",
                summary="missing kill_switch endpoint"),
        Finding(source="quick_scan", name="policy-lint", status="warn",
                summary="3 rules with overly broad scope"),
        Finding(source="quick_scan", name="scorecard", status="fail",
                score=42.0, summary="Grade: D"),
        Finding(source="quick_scan", name="compliance", status="warn",
                summary="5 NIST findings"),
        Finding(source="adhoc", name="drift", status="warn",
                score=58.0, summary="Reward divergence > 1.5σ"),
        Finding(source="adhoc", name="regression", status="fail",
                score=28.0, summary="Safety metric dropped 22%"),
    ]


def _demo_current_findings() -> List[Finding]:
    return [
        # preflight resolved
        # policy-lint resolved
        Finding(source="quick_scan", name="scorecard", status="warn",
                score=68.0, summary="Grade: B — Contract Enforcement lifted"),
        Finding(source="quick_scan", name="compliance", status="warn",
                summary="3 NIST findings remaining"),
        Finding(source="adhoc", name="drift", status="warn",
                score=42.0, summary="Reward divergence widened to 2.3σ"),
        # regression resolved
        Finding(source="adhoc", name="root-cause", status="fail", score=15.0,
                summary="newly surfaced — cascading auth-bypass in red-team"),
    ]


def _demo_snapshots():
    """Return (previous_plan, current_plan, days_between) for the CLI demo."""
    planner = RemediationPlanner()
    prev = planner.plan_from_findings(_demo_previous_findings())
    curr = planner.plan_from_findings(_demo_current_findings())
    # Pretend the persisting "drift" action has been open for 9 days
    # (will slip its 14d medium SLA only if severity were higher; but
    # we mark scorecard as 5d old for variety).
    return prev, curr, 7.0


def _demo_age_days() -> Dict[str, float]:
    return {
        "fix-scorecard": 5.0,
        "fix-compliance": 18.0,   # slipping (medium SLA 14d)
        "fix-drift": 30.0,        # slipping (medium SLA 14d)
        "fix-root-cause": 1.0,    # fresh
    }


# ── CLI ──────────────────────────────────────────────────────────────


def _ensure_utf8() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace"
            )
        except Exception:
            pass


def _load_findings_from_json(path: str) -> List[Finding]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "checks" in data:
        return [
            Finding(
                name=c.get("name", "unknown"),
                status=c.get("status", "fail"),
                source="quick_scan",
                score=c.get("score"),
                summary=c.get("summary", ""),
                details=c.get("details", {}) or {},
            )
            for c in data["checks"]
        ]
    if isinstance(data, list):
        return [
            Finding(
                name=item.get("name", "unknown"),
                status=item.get("status", "fail"),
                source=item.get("source", "json"),
                score=item.get("score"),
                summary=item.get("summary", ""),
                details=item.get("details", {}) or {},
            )
            for item in data
        ]
    raise ValueError(
        "Unrecognized JSON shape — expected quick_scan dict or list of findings."
    )


def main(argv: Optional[List[str]] = None) -> int:
    _ensure_utf8()
    parser = argparse.ArgumentParser(
        prog="replication progress",
        description=(
            "Agentic remediation progress tracker — diff two plans, "
            "compute velocity, ETA-to-green, P0/P1/P2 recommendations."
        ),
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="run against a built-in synthetic before/after pair",
    )
    parser.add_argument(
        "--previous", type=str, default=None,
        help="path to JSON of previous findings (quick_scan dict or list)",
    )
    parser.add_argument(
        "--current", type=str, default=None,
        help="path to JSON of current findings (quick_scan dict or list)",
    )
    parser.add_argument(
        "--days-between", type=float, default=1.0,
        help="elapsed days between previous and current snapshots (default: 1.0)",
    )
    parser.add_argument(
        "--format", choices=["text", "md", "json"], default="text",
        help="output format (default: text)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="write output to FILE instead of stdout",
    )
    args = parser.parse_args(argv)

    tracker = RemediationProgressTracker()

    if args.demo:
        prev_plan, curr_plan, days = _demo_snapshots()
        report = tracker.compare(
            prev_plan,
            curr_plan,
            days_between=days,
            current_age_days=_demo_age_days(),
        )
    else:
        if not (args.previous and args.current):
            parser.error("--previous and --current are required (or use --demo)")
        prev_findings = _load_findings_from_json(args.previous)
        curr_findings = _load_findings_from_json(args.current)
        report = tracker.compare_findings(
            prev_findings, curr_findings, days_between=args.days_between,
        )

    if args.format == "json":
        out = report.to_json()
    elif args.format == "md":
        out = report.to_markdown()
    else:
        out = report.render()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out)
            if not out.endswith("\n"):
                f.write("\n")
    else:
        print(out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
