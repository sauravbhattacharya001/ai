"""Remediation Assignment Advisor - agentic per-action assignment + load balancing.

Third sibling in the remediation trilogy:

* :mod:`replication.remediation_planner` answers *"what should we fix?"*
* This module answers *"who should fix it, and is anyone going to get crushed?"*
* :mod:`replication.remediation_progress` answers *"are we actually fixing it?"*

Given a :class:`~replication.remediation_planner.RemediationPlan` and a roster of
:class:`Team` records (skills, weekly capacity, current load, seniority,
on-call status), the advisor autonomously:

1. **Estimates per-action hours** from each action's ``effort`` field via a
   configurable lookup map.
2. **Scores each (action, team) pairing** along four dimensions
   (skill fit, capacity headroom, seniority fit, on-call avoidance) and picks
   the best fit per action, updating projected team load as it goes so later
   picks see realistic load.
3. **Classifies every assignment** as ``ASSIGNED`` / ``ASSIGNED_STRETCH`` /
   ``NEEDS_HIRE_OR_TRAINING`` / ``UNASSIGNED_OVERLOADED`` with a structured
   reason list.
4. **Recommends pairing** (``requires_pair_with``) when a critical action
   lands on an on-call team to protect against context loss.
5. **Modulates by risk appetite** (cautious / balanced / aggressive) -
   cautious tightens overload + seniority gates, aggressive loosens them.
6. **Emits an org-level playbook** (HIRE_OR_CONTRACT, RE_PRIORITIZE,
   LOAD_BALANCE, CROSS_TRAIN, REVIEW_SENIORITY_MIX) plus autonomous
   insights and an A-F portfolio grade with a paste-ready headline.

CLI::

    python -m replication assign --demo
    python -m replication assign --demo --format md
    python -m replication assign --plan plan.json --teams teams.json --format json

Programmatic::

    from replication.remediation_planner import RemediationPlanner
    from replication.remediation_assignment import RemediationAssignmentAdvisor, Team

    plan = RemediationPlanner().plan_from_findings(findings)
    advisor = RemediationAssignmentAdvisor(teams=[
        Team("alpha", skills=["safety-eng", "policy"], weekly_capacity_hours=40),
        Team("beta",  skills=["platform", "ml-ops"],  weekly_capacity_hours=40),
    ], risk_appetite="balanced")
    report = advisor.assign(plan)
    print(report.render())
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .remediation_planner import (
    Finding,
    RemediationAction,
    RemediationPlan,
    RemediationPlanner,
)


# ── Constants ────────────────────────────────────────────────────────

SENIORITY_RANK: Dict[str, int] = {
    "junior": 1,
    "mid": 2,
    "senior": 3,
    "principal": 4,
}

DEFAULT_HOURS_PER_EFFORT: Dict[Union[int, str], float] = {
    # action.effort is 1-5 engineer-days in the planner.  Map to hours.
    1: 4.0,
    2: 8.0,
    3: 16.0,
    4: 28.0,
    5: 40.0,
    "small": 4.0,
    "medium": 8.0,
    "large": 28.0,
    "xlarge": 40.0,
}

VERDICTS = (
    "ASSIGNED",
    "ASSIGNED_STRETCH",
    "NEEDS_HIRE_OR_TRAINING",
    "UNASSIGNED_OVERLOADED",
)

RISK_APPETITES = ("cautious", "balanced", "aggressive")


# ── Public dataclasses ───────────────────────────────────────────────


@dataclass
class Team:
    """A team that can pick up remediation actions."""

    name: str
    skills: List[str] = field(default_factory=list)
    weekly_capacity_hours: float = 40.0
    current_load_hours: float = 0.0
    seniority: str = "mid"  # junior | mid | senior | principal
    on_call: bool = False

    def __post_init__(self) -> None:
        # Normalise so set-based skill matching is case-insensitive.
        self.skills = [str(s).lower().strip() for s in self.skills if str(s).strip()]
        if self.seniority not in SENIORITY_RANK:
            self.seniority = "mid"
        if self.weekly_capacity_hours <= 0:
            self.weekly_capacity_hours = 1.0
        if self.current_load_hours < 0:
            self.current_load_hours = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "skills": list(self.skills),
            "weekly_capacity_hours": self.weekly_capacity_hours,
            "current_load_hours": self.current_load_hours,
            "seniority": self.seniority,
            "on_call": self.on_call,
        }


@dataclass
class Assignment:
    action_id: str
    action_title: str
    severity: str
    priority_tier: str  # P0 | P1 | P2 | P3
    team: str
    hours: float
    verdict: str
    score: float
    score_breakdown: Dict[str, float]
    reasons: List[str]
    requires_pair_with: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_title": self.action_title,
            "severity": self.severity,
            "priority_tier": self.priority_tier,
            "team": self.team,
            "hours": round(self.hours, 2),
            "verdict": self.verdict,
            "score": round(self.score, 3),
            "score_breakdown": {k: round(v, 3) for k, v in self.score_breakdown.items()},
            "reasons": list(self.reasons),
            "requires_pair_with": self.requires_pair_with,
        }


@dataclass
class TeamLoad:
    team: str
    projected_hours: float
    capacity: float
    utilization_pct: float
    overloaded: bool
    action_count: int
    p0_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "team": self.team,
            "projected_hours": round(self.projected_hours, 2),
            "capacity": round(self.capacity, 2),
            "utilization_pct": round(self.utilization_pct, 1),
            "overloaded": self.overloaded,
            "action_count": self.action_count,
            "p0_count": self.p0_count,
        }


@dataclass
class PlaybookItem:
    priority: str  # P0 | P1 | P2
    code: str
    title: str
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "priority": self.priority,
            "code": self.code,
            "title": self.title,
            "detail": self.detail,
        }


@dataclass
class AssignmentReport:
    assignments: List[Assignment]
    team_loads: Dict[str, TeamLoad]
    playbook: List[PlaybookItem]
    insights: List[str]
    grade: str
    headline: str
    risk_appetite: str
    overload_factor: float
    generated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "headline": self.headline,
            "grade": self.grade,
            "risk_appetite": self.risk_appetite,
            "overload_factor": self.overload_factor,
            "generated_at": self.generated_at,
            "assignments": [a.to_dict() for a in self.assignments],
            "team_loads": {k: v.to_dict() for k, v in self.team_loads.items()},
            "playbook": [p.to_dict() for p in self.playbook],
            "insights": list(self.insights),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def render(self) -> str:
        lines: List[str] = []
        lines.append(self.headline)
        lines.append(
            f"  risk_appetite={self.risk_appetite}  overload_factor={self.overload_factor}"
        )
        lines.append("")
        lines.append("Team loads:")
        for name, load in sorted(self.team_loads.items()):
            flag = "  [OVERLOADED]" if load.overloaded else ""
            lines.append(
                f"  - {name}: {load.projected_hours:.1f}/{load.capacity:.1f}h "
                f"({load.utilization_pct:.0f}%)  actions={load.action_count}  p0={load.p0_count}{flag}"
            )
        lines.append("")
        lines.append("Assignments:")
        for a in self.assignments:
            pair = f"  pair_with={a.requires_pair_with}" if a.requires_pair_with else ""
            lines.append(
                f"  [{a.priority_tier}] {a.action_id} {a.action_title[:60]}"
            )
            lines.append(
                f"      -> {a.team}  verdict={a.verdict}  score={a.score:.2f}  hours={a.hours:.1f}{pair}"
            )
            for r in a.reasons:
                lines.append(f"         . {r}")
        if self.playbook:
            lines.append("")
            lines.append("Org playbook:")
            for p in self.playbook:
                lines.append(f"  [{p.priority}] {p.code}: {p.title}")
                lines.append(f"        {p.detail}")
        if self.insights:
            lines.append("")
            lines.append("Insights:")
            for ins in self.insights:
                lines.append(f"  - {ins}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        lines: List[str] = []
        lines.append(f"# Remediation Assignment Report")
        lines.append("")
        lines.append(f"**{self.headline}**")
        lines.append("")
        lines.append(
            f"- risk_appetite: `{self.risk_appetite}`  overload_factor: `{self.overload_factor}`"
        )
        lines.append(f"- generated_at: `{self.generated_at}`")
        lines.append("")
        lines.append("## Team Loads")
        lines.append("")
        lines.append("| Team | Projected | Capacity | Util % | Actions | P0 | Overloaded |")
        lines.append("|------|-----------|----------|--------|---------|----|------------|")
        for name, load in sorted(self.team_loads.items()):
            lines.append(
                f"| {name} | {load.projected_hours:.1f}h | {load.capacity:.1f}h | "
                f"{load.utilization_pct:.0f}% | {load.action_count} | {load.p0_count} | "
                f"{'YES' if load.overloaded else 'no'} |"
            )
        lines.append("")
        lines.append("## Assignments")
        lines.append("")
        for a in self.assignments:
            pair = f" -- pair_with **{a.requires_pair_with}**" if a.requires_pair_with else ""
            lines.append(
                f"### [{a.priority_tier}] {a.action_id} -- {a.action_title}"
            )
            lines.append(
                f"- **Team:** `{a.team}` -- verdict `{a.verdict}` -- score {a.score:.2f} -- hours {a.hours:.1f}h{pair}"
            )
            if a.reasons:
                lines.append("- Reasons:")
                for r in a.reasons:
                    lines.append(f"  - {r}")
            lines.append("")
        if self.playbook:
            lines.append("## Org Playbook")
            lines.append("")
            for p in self.playbook:
                lines.append(f"- **[{p.priority}] {p.code}** -- {p.title}")
                lines.append(f"  - {p.detail}")
            lines.append("")
        if self.insights:
            lines.append("## Insights")
            lines.append("")
            for ins in self.insights:
                lines.append(f"- {ins}")
            lines.append("")
        return "\n".join(lines)


# ── Helpers ──────────────────────────────────────────────────────────


def _priority_tier(action: RemediationAction) -> str:
    sev = (action.severity or "").lower()
    pr = action.priority  # property: (impact*urgency)/effort
    if sev == "critical" or pr >= 12.0:
        return "P0"
    if sev == "high" or pr >= 6.0:
        return "P1"
    if sev == "medium" or pr >= 3.0:
        return "P2"
    return "P3"


def _effort_to_hours(action: RemediationAction, table: Dict[Union[int, str], float]) -> float:
    e = action.effort
    if e in table:
        return float(table[e])
    # Fallback: clamp 1..5 numeric.
    try:
        ei = int(e)
        if ei in table:
            return float(table[ei])
        # Linear extrapolate beyond 5.
        return float(table.get(5, 40.0)) * max(1.0, ei / 5.0)
    except (TypeError, ValueError):
        pass
    return float(table.get("medium", table.get(2, 8.0)))


def _action_skill_tags(action: RemediationAction) -> List[str]:
    """Derive a tag set for skill matching from the action."""
    tags: List[str] = []
    if action.owner_hint and action.owner_hint.lower() != "safety-team":
        tags.append(action.owner_hint.lower())
    # Pull keywords from title/rationale/finding source.
    text = " ".join(
        [
            action.title or "",
            action.rationale or "",
            (action.finding.source if action.finding else "") or "",
            (action.finding.name if action.finding else "") or "",
        ]
    ).lower()
    keyword_skills = {
        "policy": ["policy", "lint", "guardrail"],
        "compliance": ["compliance", "audit", "regulatory", "regulation"],
        "safety-eng": ["safety", "alignment", "scorecard", "drift"],
        "ml-ops": ["model", "regression", "monitor", "metric"],
        "platform": ["platform", "infra", "preflight", "deploy"],
        "security": ["secret", "credential", "leak", "vuln", "injection"],
    }
    for skill, words in keyword_skills.items():
        if any(w in text for w in words):
            tags.append(skill)
    # De-dup, preserve order.
    seen: set[str] = set()
    out: List[str] = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _seniority_fit(team: Team, severity: str, risk_appetite: str) -> float:
    rank = SENIORITY_RANK.get(team.seniority, 2)
    sev = (severity or "").lower()
    if sev == "critical":
        if risk_appetite == "cautious":
            return 1.0 if rank >= 3 else 0.0
        return 1.0 if rank >= 3 else (0.4 if rank == 2 else 0.0)
    if sev == "high":
        if risk_appetite == "aggressive":
            return 1.0 if rank >= 2 else 0.8
        return 1.0 if rank >= 2 else 0.5
    return 1.0  # medium / low / info — anyone is fine


# ── Advisor ──────────────────────────────────────────────────────────


class RemediationAssignmentAdvisor:
    """Agentic per-action owner picker with capacity-aware re-balancing."""

    def __init__(
        self,
        teams: Iterable[Team],
        hours_per_effort_unit: Optional[Dict[Union[int, str], float]] = None,
        risk_appetite: str = "balanced",
        overload_factor: float = 1.15,
        now: Optional[datetime] = None,
    ) -> None:
        team_list = list(teams)
        if not team_list:
            raise ValueError("at least one team is required")
        # Stable order by name for determinism.
        self._teams: List[Team] = [copy.deepcopy(t) for t in team_list]
        self._teams.sort(key=lambda t: t.name)
        self._hours_table: Dict[Union[int, str], float] = dict(DEFAULT_HOURS_PER_EFFORT)
        if hours_per_effort_unit:
            self._hours_table.update(hours_per_effort_unit)
        if risk_appetite not in RISK_APPETITES:
            risk_appetite = "balanced"
        self._risk = risk_appetite
        # Risk-appetite-modulated overload factor.
        if risk_appetite == "cautious":
            self._overload = min(overload_factor, 0.95)
        elif risk_appetite == "aggressive":
            self._overload = max(overload_factor, 1.30)
        else:
            self._overload = overload_factor
        self._now = now or datetime.now(timezone.utc)

    # ── public ───────────────────────────────────────────────────────

    def assign(
        self,
        plan: Union[RemediationPlan, Iterable[RemediationAction]],
    ) -> AssignmentReport:
        if isinstance(plan, RemediationPlan):
            actions = list(plan.actions)
        else:
            actions = list(plan)

        # Process in priority order: planner orders by priority desc; resort by
        # (P-tier, -priority, id) for stability.
        def _sort_key(a: RemediationAction) -> Tuple[int, float, str]:
            tier_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}[_priority_tier(a)]
            return (tier_rank, -float(a.priority), a.id)

        actions_sorted = sorted(actions, key=_sort_key)

        # Snapshot team loads so the advisor never mutates the caller's data.
        team_loads_h: Dict[str, float] = {t.name: t.current_load_hours for t in self._teams}
        team_action_counts: Dict[str, int] = {t.name: 0 for t in self._teams}
        team_p0_counts: Dict[str, int] = {t.name: 0 for t in self._teams}

        assignments: List[Assignment] = []

        # Risk-appetite weights.
        if self._risk == "cautious":
            w_skill, w_cap, w_sen, w_oncall = 0.55, 0.25, 0.15, 0.05
        elif self._risk == "aggressive":
            w_skill, w_cap, w_sen, w_oncall = 0.35, 0.40, 0.20, 0.05
        else:
            w_skill, w_cap, w_sen, w_oncall = 0.45, 0.30, 0.20, 0.05

        stretch_skill_needs: Dict[str, int] = {}
        junior_mid_critical_count = 0
        p0p1_count = 0

        for action in actions_sorted:
            hours = _effort_to_hours(action, self._hours_table)
            tier = _priority_tier(action)
            tags = _action_skill_tags(action)

            best_team: Optional[Team] = None
            best_score = -1.0
            best_breakdown: Dict[str, float] = {}
            any_skill_overlap = False
            all_overloaded_before = True

            for team in self._teams:
                projected = team_loads_h[team.name] + hours
                cap = team.weekly_capacity_hours
                # Headroom AFTER taking this action.
                headroom = max(0.0, (cap - projected) / cap) if cap > 0 else 0.0
                cap_score = headroom  # in [0,1]
                # Skill match.
                skill_score = _jaccard(tags, team.skills)
                if skill_score > 0:
                    any_skill_overlap = True
                sen_score = _seniority_fit(team, action.severity or "", self._risk)
                oncall_score = 0.0 if team.on_call else 1.0
                score = (
                    w_skill * skill_score
                    + w_cap * cap_score
                    + w_sen * sen_score
                    + w_oncall * oncall_score
                )
                # Track whether at least one team was below overload BEFORE this action.
                if team_loads_h[team.name] < self._overload * cap:
                    all_overloaded_before = False
                if score > best_score:
                    best_score = score
                    best_team = team
                    best_breakdown = {
                        "skill": skill_score,
                        "capacity_headroom": cap_score,
                        "seniority_fit": sen_score,
                        "on_call_avoidance": oncall_score,
                    }

            assert best_team is not None  # roster non-empty

            reasons: List[str] = []
            verdict: str
            requires_pair_with: Optional[str] = None

            # Verdict decision tree.
            if not any_skill_overlap:
                verdict = "NEEDS_HIRE_OR_TRAINING"
                reasons.append(
                    f"No team has overlap with required skills {tags or ['(unspecified)']}"
                )
            elif all_overloaded_before:
                verdict = "UNASSIGNED_OVERLOADED"
                reasons.append(
                    f"Every team is already past overload threshold "
                    f"({self._overload:.2f} x capacity)"
                )
            elif best_score >= 0.55:
                verdict = "ASSIGNED"
            else:
                verdict = "ASSIGNED_STRETCH"
                if best_breakdown.get("skill", 0) < 0.34:
                    reasons.append(
                        f"Weak skill overlap ({best_breakdown.get('skill', 0):.2f}) "
                        f"between action tags {tags} and team skills {best_team.skills}"
                    )
                if best_breakdown.get("capacity_headroom", 0) < 0.25:
                    reasons.append(
                        f"Low capacity headroom after assignment "
                        f"({best_breakdown.get('capacity_headroom', 0):.2f})"
                    )
                if best_breakdown.get("seniority_fit", 1.0) < 1.0:
                    reasons.append(
                        f"Seniority gap: {best_team.seniority} on {action.severity} action"
                    )
                # Track unmet skills for cross-training playbook.
                for tag in tags:
                    if tag not in set(best_team.skills):
                        stretch_skill_needs[tag] = stretch_skill_needs.get(tag, 0) + 1

            # Pairing recommendation: critical action lands on on-call team.
            if (
                (action.severity or "").lower() == "critical"
                and best_team.on_call
                and verdict in ("ASSIGNED", "ASSIGNED_STRETCH")
            ):
                # Find a non-on-call team with at least some skill overlap.
                pair_candidates = [
                    t for t in self._teams
                    if not t.on_call and t.name != best_team.name
                    and _jaccard(tags, t.skills) > 0
                ]
                if not pair_candidates:
                    pair_candidates = [
                        t for t in self._teams
                        if not t.on_call and t.name != best_team.name
                    ]
                if pair_candidates:
                    pair_candidates.sort(
                        key=lambda t: (
                            -_jaccard(tags, t.skills),
                            team_loads_h[t.name],
                        )
                    )
                    requires_pair_with = pair_candidates[0].name
                    reasons.append(
                        f"On-call team {best_team.name} - pair with "
                        f"{requires_pair_with} for context resilience"
                    )

            # Reason for the happy path.
            if verdict == "ASSIGNED" and not reasons:
                reasons.append(
                    f"Best fit on skill ({best_breakdown.get('skill', 0):.2f}) "
                    f"and capacity ({best_breakdown.get('capacity_headroom', 0):.2f})"
                )

            # Update load only when actually assigning.
            if verdict in ("ASSIGNED", "ASSIGNED_STRETCH", "NEEDS_HIRE_OR_TRAINING"):
                team_loads_h[best_team.name] += hours
                team_action_counts[best_team.name] += 1
                if tier == "P0":
                    team_p0_counts[best_team.name] += 1
                if (action.severity or "").lower() == "critical" and SENIORITY_RANK.get(
                    best_team.seniority, 2
                ) <= 2:
                    junior_mid_critical_count += 1

            if tier in ("P0", "P1"):
                p0p1_count += 1

            assignments.append(
                Assignment(
                    action_id=action.id,
                    action_title=action.title,
                    severity=(action.severity or "").lower(),
                    priority_tier=tier,
                    team=best_team.name,
                    hours=hours,
                    verdict=verdict,
                    score=max(0.0, best_score),
                    score_breakdown=best_breakdown,
                    reasons=reasons,
                    requires_pair_with=requires_pair_with,
                )
            )

        # Build team-loads dict.
        team_loads: Dict[str, TeamLoad] = {}
        for team in self._teams:
            proj = team_loads_h[team.name]
            cap = team.weekly_capacity_hours
            util = (proj / cap) * 100.0 if cap > 0 else 0.0
            team_loads[team.name] = TeamLoad(
                team=team.name,
                projected_hours=proj,
                capacity=cap,
                utilization_pct=util,
                overloaded=proj > self._overload * cap,
                action_count=team_action_counts[team.name],
                p0_count=team_p0_counts[team.name],
            )

        # Playbook.
        playbook: List[PlaybookItem] = []
        needs_hire = [a for a in assignments if a.verdict == "NEEDS_HIRE_OR_TRAINING"]
        unassigned = [a for a in assignments if a.verdict == "UNASSIGNED_OVERLOADED"]
        p0_unassigned = [a for a in unassigned if a.priority_tier == "P0"]
        overloaded_teams = [n for n, l in team_loads.items() if l.overloaded]
        underused_teams = [
            n for n, l in team_loads.items() if l.utilization_pct < 60.0
        ]

        if needs_hire:
            tag_freq: Dict[str, int] = {}
            for a in needs_hire:
                # crude: pull the first reason that lists skills
                for tag in _action_skill_tags_from_assignment(a):
                    tag_freq[tag] = tag_freq.get(tag, 0) + 1
            wanted = ", ".join(
                f"{k} ({v})" for k, v in sorted(tag_freq.items(), key=lambda kv: -kv[1])[:3]
            )
            playbook.append(
                PlaybookItem(
                    priority="P0",
                    code="HIRE_OR_CONTRACT",
                    title=f"{len(needs_hire)} action(s) have no skill match in the roster",
                    detail=f"Most-needed skills: {wanted or '(no tags)'}",
                )
            )

        if p0_unassigned:
            playbook.append(
                PlaybookItem(
                    priority="P0",
                    code="RE_PRIORITIZE",
                    title=f"{len(p0_unassigned)} P0 action(s) UNASSIGNED_OVERLOADED",
                    detail=(
                        "All teams past overload threshold before these landed. "
                        "Defer lower-tier work or pull in contractors."
                    ),
                )
            )

        if overloaded_teams and underused_teams:
            playbook.append(
                PlaybookItem(
                    priority="P1",
                    code="LOAD_BALANCE",
                    title=(
                        f"Overloaded: {', '.join(overloaded_teams)} / "
                        f"Underused (<60%): {', '.join(underused_teams)}"
                    ),
                    detail=(
                        "Reassign 1-2 stretch actions from overloaded to underused teams, "
                        "or cross-train underused team in the missing skill."
                    ),
                )
            )

        if stretch_skill_needs:
            top_skill, top_count = max(stretch_skill_needs.items(), key=lambda kv: kv[1])
            playbook.append(
                PlaybookItem(
                    priority="P1",
                    code="CROSS_TRAIN",
                    title=f"Cross-train teams in `{top_skill}` (gap on {top_count} action(s))",
                    detail=(
                        f"`{top_skill}` showed up as a missing skill on stretch assignments "
                        f"{top_count} time(s). One training cycle pays back across the backlog."
                    ),
                )
            )

        if p0p1_count > 0 and junior_mid_critical_count / max(1, p0p1_count) > 0.25:
            playbook.append(
                PlaybookItem(
                    priority="P2",
                    code="REVIEW_SENIORITY_MIX",
                    title=(
                        f"{junior_mid_critical_count}/{p0p1_count} P0/P1 critical actions "
                        f"landed on junior/mid teams"
                    ),
                    detail=(
                        "Consider rotating a senior in for these, or pair-programming the "
                        "first 1-2 hours of each."
                    ),
                )
            )

        # Insights.
        insights: List[str] = []
        if team_loads:
            top_load = max(team_loads.values(), key=lambda l: l.utilization_pct)
            insights.append(
                f"Highest utilization: {top_load.team} at {top_load.utilization_pct:.0f}% "
                f"({top_load.action_count} actions, {top_load.p0_count} P0)"
            )
        clean_count = sum(1 for a in assignments if a.verdict == "ASSIGNED")
        stretch_count = sum(1 for a in assignments if a.verdict == "ASSIGNED_STRETCH")
        if assignments:
            insights.append(
                f"{clean_count}/{len(assignments)} clean, {stretch_count} stretch, "
                f"{len(needs_hire)} needs-hire, {len(unassigned)} overloaded"
            )
        oncall_collisions = sum(1 for a in assignments if a.requires_pair_with)
        if oncall_collisions:
            insights.append(
                f"{oncall_collisions} critical action(s) landed on an on-call team - "
                f"pairing recommended"
            )
        high_util_teams = [n for n, l in team_loads.items() if l.utilization_pct >= 90.0]
        if high_util_teams:
            insights.append(
                f"{len(high_util_teams)} team(s) at >=90% utilization: "
                f"{', '.join(sorted(high_util_teams))}"
            )

        # Grade.
        grade = self._compute_grade(
            assignments=assignments,
            needs_hire_n=len(needs_hire),
            unassigned_n=len(unassigned),
            overloaded_n=len(overloaded_teams),
        )

        headline = (
            f"[{grade}] {clean_count + stretch_count}/{len(assignments)} actions assigned"
            f"{', ' + str(len(needs_hire)) + ' needs hire' if needs_hire else ''}"
            f"{', ' + str(len(unassigned)) + ' overloaded' if unassigned else ''}"
            f", {len(overloaded_teams)} team(s) overloaded"
        )

        return AssignmentReport(
            assignments=assignments,
            team_loads=team_loads,
            playbook=playbook,
            insights=insights,
            grade=grade,
            headline=headline,
            risk_appetite=self._risk,
            overload_factor=self._overload,
            generated_at=self._now.isoformat(),
        )

    # ── helpers ──────────────────────────────────────────────────────

    def _compute_grade(
        self,
        assignments: List[Assignment],
        needs_hire_n: int,
        unassigned_n: int,
        overloaded_n: int,
    ) -> str:
        total = len(assignments)
        if total == 0:
            return "A"
        clean = sum(1 for a in assignments if a.verdict == "ASSIGNED")
        clean_pct = clean / total
        # Hard floor: everyone overloaded = F.
        if unassigned_n == total:
            return "F"
        # Start from base.
        if clean_pct >= 0.80:
            base = "A"
        elif clean_pct >= 0.60:
            base = "B"
        elif clean_pct >= 0.40:
            base = "C"
        elif clean_pct >= 0.20:
            base = "D"
        else:
            base = "F"
        # Penalties.
        penalty = 0
        if needs_hire_n > 0:
            penalty += 1
        if overloaded_n >= 1:
            penalty += 1
        if unassigned_n >= 1:
            penalty += 1
        order = ["A", "B", "C", "D", "F"]
        idx = min(len(order) - 1, order.index(base) + penalty)
        return order[idx]


def _action_skill_tags_from_assignment(a: Assignment) -> List[str]:
    """Best-effort extraction of tag tokens from a NEEDS_HIRE assignment's reason."""
    out: List[str] = []
    for r in a.reasons:
        # Find anything inside [..] or {..}.
        for open_c, close_c in (("[", "]"), ("{", "}")):
            if open_c in r and close_c in r:
                inside = r.split(open_c, 1)[1].split(close_c, 1)[0]
                for tok in inside.replace("'", "").replace('"', "").split(","):
                    tok = tok.strip()
                    if tok and tok != "(unspecified)":
                        out.append(tok)
    return out


# ── Demo + CLI ───────────────────────────────────────────────────────


def _demo_teams() -> List[Team]:
    return [
        Team(
            name="alpha",
            skills=["safety-eng", "policy", "platform"],
            weekly_capacity_hours=40,
            current_load_hours=10,
            seniority="senior",
            on_call=False,
        ),
        Team(
            name="beta",
            skills=["ml-ops", "platform"],
            weekly_capacity_hours=40,
            current_load_hours=24,
            seniority="mid",
            on_call=True,
        ),
        Team(
            name="gamma",
            skills=["compliance"],
            weekly_capacity_hours=24,
            current_load_hours=4,
            seniority="junior",
            on_call=False,
        ),
    ]


def _teams_from_dicts(rows: Iterable[Dict[str, Any]]) -> List[Team]:
    out: List[Team] = []
    for r in rows:
        out.append(
            Team(
                name=str(r["name"]),
                skills=list(r.get("skills", [])),
                weekly_capacity_hours=float(r.get("weekly_capacity_hours", 40.0)),
                current_load_hours=float(r.get("current_load_hours", 0.0)),
                seniority=str(r.get("seniority", "mid")),
                on_call=bool(r.get("on_call", False)),
            )
        )
    return out


def _plan_from_json(path: str) -> RemediationPlan:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    # Accept either a plan-shaped dict (with 'actions') or a list of findings.
    if isinstance(data, dict) and "actions" in data:
        # Rebuild Finding + RemediationAction from the planner's to_dict format.
        actions: List[RemediationAction] = []
        for ad in data["actions"]:
            f = ad.get("finding", {})
            finding = Finding(
                name=f.get("name", ""),
                status=f.get("status", "fail"),
                source=f.get("source", ""),
                score=float(f.get("score", 0.0)),
                summary=f.get("summary", ""),
                details=f.get("details", {}) or {},
            )
            actions.append(
                RemediationAction(
                    id=ad.get("id", ""),
                    title=ad.get("title", ""),
                    finding=finding,
                    severity=ad.get("severity", "medium"),
                    impact=int(ad.get("impact", 5)),
                    effort=int(ad.get("effort", 2)),
                    urgency=int(ad.get("urgency", 3)),
                    depends_on=list(ad.get("depends_on", []) or []),
                    rationale=ad.get("rationale", ""),
                    suggested_steps=list(ad.get("suggested_steps", []) or []),
                    owner_hint=ad.get("owner_hint", "safety-team"),
                )
            )
        return RemediationPlan(
            actions=actions,
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            total_effort_days=int(data.get("total_effort_days", 0)),
            quick_wins=list(data.get("quick_wins", []) or []),
            critical_path=list(data.get("critical_path", []) or []),
            notes=list(data.get("notes", []) or []),
        )
    if isinstance(data, list):
        findings = [
            Finding(
                name=f.get("name", ""),
                status=f.get("status", "fail"),
                source=f.get("source", ""),
                score=float(f.get("score", 0.0)),
                summary=f.get("summary", ""),
                details=f.get("details", {}) or {},
            )
            for f in data
        ]
        return RemediationPlanner().plan_from_findings(findings)
    raise ValueError("plan JSON must be a plan-dict or list of findings")


def _ensure_utf8() -> None:
    try:
        if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace"
            )
    except Exception:
        pass


def main(argv: Optional[List[str]] = None) -> int:
    _ensure_utf8()
    parser = argparse.ArgumentParser(
        prog="replication assign",
        description=(
            "Assign remediation actions to teams with capacity + skill-aware "
            "balancing and emit an autonomous org playbook."
        ),
    )
    parser.add_argument("--demo", action="store_true", help="Run with built-in demo data.")
    parser.add_argument("--plan", help="Path to a plan or findings JSON file.")
    parser.add_argument("--teams", help="Path to a teams JSON file.")
    parser.add_argument(
        "--format",
        choices=("text", "markdown", "md", "json"),
        default="text",
    )
    parser.add_argument(
        "--risk",
        choices=RISK_APPETITES,
        default="balanced",
    )
    parser.add_argument(
        "--overload-factor",
        type=float,
        default=1.15,
        help="Utilization ratio above which a team is flagged as overloaded.",
    )
    parser.add_argument("--output", help="Write output to this path instead of stdout.")
    args = parser.parse_args(argv)

    if not args.demo and not (args.plan and args.teams):
        parser.error("either --demo or both --plan and --teams are required")

    if args.demo:
        plan = RemediationPlanner().plan_from_findings(_demo_findings_for_cli())
        teams = _demo_teams()
    else:
        plan = _plan_from_json(args.plan)
        with open(args.teams, "r", encoding="utf-8") as fh:
            teams = _teams_from_dicts(json.load(fh))

    advisor = RemediationAssignmentAdvisor(
        teams=teams,
        risk_appetite=args.risk,
        overload_factor=args.overload_factor,
    )
    report = advisor.assign(plan)

    fmt = "markdown" if args.format == "md" else args.format
    if fmt == "json":
        out = report.to_json()
    elif fmt == "markdown":
        out = report.to_markdown()
    else:
        out = report.render()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(out)
        print(f"Wrote {fmt} report to {args.output}")
    else:
        print(out)
    return 0


def _demo_findings_for_cli() -> List[Finding]:
    """Standalone demo findings (independent of the planner's internal demo)."""
    return [
        Finding(
            name="policy_lint_failed",
            status="fail",
            source="policy_linter",
            score=0.2,
            summary="Critical policy lint errors detected",
            details={"errors": 4},
        ),
        Finding(
            name="scorecard_dropped",
            status="fail",
            source="scorecard",
            score=0.55,
            summary="Safety scorecard dropped below threshold",
            details={"grade": "D"},
        ),
        Finding(
            name="regression_in_alignment",
            status="fail",
            source="regression",
            score=0.4,
            summary="Alignment regression vs previous run",
            details={},
        ),
        Finding(
            name="drift_detected",
            status="warn",
            source="drift",
            score=0.6,
            summary="Behavioural drift outside window",
            details={},
        ),
        Finding(
            name="compliance_gap",
            status="fail",
            source="compliance",
            score=0.5,
            summary="Compliance audit gap (NIST AI RMF)",
            details={},
        ),
        Finding(
            name="model_card_missing",
            status="warn",
            source="model_card",
            score=0.7,
            summary="Model card incomplete",
            details={},
        ),
    ]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
