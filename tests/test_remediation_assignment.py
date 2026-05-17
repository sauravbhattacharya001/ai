"""Tests for replication.remediation_assignment."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

import pytest

from replication.remediation_assignment import (
    Assignment,
    AssignmentReport,
    PlaybookItem,
    RemediationAssignmentAdvisor,
    Team,
    TeamLoad,
    main,
)
from replication.remediation_planner import (
    Finding,
    RemediationAction,
    RemediationPlan,
    RemediationPlanner,
)


# ── Fixtures ─────────────────────────────────────────────────────────


def _finding(name: str, source: str, score: float = 0.4, status: str = "fail") -> Finding:
    return Finding(name=name, status=status, source=source, score=score, summary=name, details={})


def _action(
    aid: str,
    title: str,
    finding: Finding,
    severity: str = "high",
    impact: int = 7,
    effort: int = 2,
    urgency: int = 4,
    owner_hint: str = "safety-eng",
    rationale: str = "",
) -> RemediationAction:
    return RemediationAction(
        id=aid,
        title=title,
        finding=finding,
        severity=severity,
        impact=impact,
        effort=effort,
        urgency=urgency,
        depends_on=[],
        rationale=rationale,
        suggested_steps=[],
        owner_hint=owner_hint,
    )


def _basic_plan() -> RemediationPlan:
    actions = [
        _action(
            "a1", "Fix policy lint",
            _finding("policy_lint_failed", "policy_linter"),
            severity="high", impact=8, effort=1, urgency=5,
            owner_hint="policy",
        ),
        _action(
            "a2", "Fix scorecard drop",
            _finding("scorecard_dropped", "scorecard"),
            severity="critical", impact=9, effort=2, urgency=5,
            owner_hint="safety-eng",
        ),
        _action(
            "a3", "Address compliance gap",
            _finding("compliance_gap", "compliance"),
            severity="high", impact=7, effort=3, urgency=3,
            owner_hint="compliance",
        ),
    ]
    return RemediationPlan(
        actions=actions, timestamp="2026-01-01T00:00:00+00:00",
        total_effort_days=6, quick_wins=["a1"], critical_path=["a2"], notes=[],
    )


def _three_teams() -> list[Team]:
    return [
        Team("alpha", skills=["safety-eng", "policy", "platform"],
             weekly_capacity_hours=40, current_load_hours=5, seniority="senior"),
        Team("beta", skills=["ml-ops", "platform"],
             weekly_capacity_hours=40, current_load_hours=10, seniority="mid"),
        Team("gamma", skills=["compliance"],
             weekly_capacity_hours=24, current_load_hours=2, seniority="senior"),
    ]


# ── Tests ────────────────────────────────────────────────────────────


def test_happy_path_assigns_all_actions():
    advisor = RemediationAssignmentAdvisor(teams=_three_teams())
    report = advisor.assign(_basic_plan())
    assert len(report.assignments) == 3
    assert all(a.verdict in ("ASSIGNED", "ASSIGNED_STRETCH") for a in report.assignments)
    # Each action goes to the team with best skill overlap.
    by_id = {a.action_id: a for a in report.assignments}
    assert by_id["a1"].team == "alpha"        # policy
    assert by_id["a2"].team == "alpha"        # safety-eng
    assert by_id["a3"].team == "gamma"        # compliance


def test_overload_triggers_unassigned_overloaded():
    teams = [
        Team("alpha", skills=["safety-eng"], weekly_capacity_hours=8,
             current_load_hours=12, seniority="senior"),
    ]
    plan = _basic_plan()
    advisor = RemediationAssignmentAdvisor(teams=teams, overload_factor=1.0)
    report = advisor.assign(plan)
    assert any(a.verdict == "UNASSIGNED_OVERLOADED" for a in report.assignments)


def test_no_skill_match_flags_needs_hire():
    teams = [
        Team("alpha", skills=["random-skill"], weekly_capacity_hours=40,
             current_load_hours=0, seniority="senior"),
    ]
    f = _finding("weird", "obscure_module")
    a = _action("x1", "Do an obscure thing", f, severity="medium",
                owner_hint="weirdo")
    plan = RemediationPlan(actions=[a], timestamp="t", total_effort_days=1,
                           quick_wins=[], critical_path=[], notes=[])
    advisor = RemediationAssignmentAdvisor(teams=teams)
    report = advisor.assign(plan)
    assert report.assignments[0].verdict == "NEEDS_HIRE_OR_TRAINING"
    assert any(p.code == "HIRE_OR_CONTRACT" for p in report.playbook)


def test_cautious_appetite_blocks_junior_on_critical():
    teams = [
        Team("alpha", skills=["safety-eng"], weekly_capacity_hours=40,
             current_load_hours=0, seniority="junior"),
        Team("beta", skills=["safety-eng"], weekly_capacity_hours=40,
             current_load_hours=0, seniority="senior"),
    ]
    a = _action("c1", "Critical fix",
                _finding("scorecard_dropped", "scorecard"),
                severity="critical", impact=9, effort=2, urgency=5,
                owner_hint="safety-eng")
    plan = RemediationPlan(actions=[a], timestamp="t", total_effort_days=1,
                           quick_wins=[], critical_path=[], notes=[])
    advisor = RemediationAssignmentAdvisor(teams=teams, risk_appetite="cautious")
    report = advisor.assign(plan)
    # Senior should win because junior fit=0 under cautious.
    assert report.assignments[0].team == "beta"
    # And if we only had junior, seniority_fit should be 0.
    only_junior = RemediationAssignmentAdvisor(
        teams=[teams[0]], risk_appetite="cautious"
    )
    r2 = only_junior.assign(plan)
    assert r2.assignments[0].score_breakdown["seniority_fit"] == 0.0


def test_oncall_critical_triggers_pair_with():
    teams = [
        Team("alpha", skills=["safety-eng"], weekly_capacity_hours=40,
             seniority="senior", on_call=True),
        Team("beta", skills=["safety-eng"], weekly_capacity_hours=40,
             seniority="senior", on_call=False),
    ]
    a = _action("c1", "Critical fix",
                _finding("scorecard_dropped", "scorecard"),
                severity="critical", impact=9, effort=2, urgency=5,
                owner_hint="safety-eng")
    plan = RemediationPlan(actions=[a], timestamp="t", total_effort_days=1,
                           quick_wins=[], critical_path=[], notes=[])
    # Force alpha to win by giving it more headroom + same skill.
    teams[0].current_load_hours = 0
    teams[1].current_load_hours = 35
    advisor = RemediationAssignmentAdvisor(teams=teams)
    report = advisor.assign(plan)
    asn = report.assignments[0]
    assert asn.team == "alpha"
    assert asn.requires_pair_with == "beta"


def test_aggressive_tolerates_higher_load():
    teams = [
        Team("alpha", skills=["safety-eng"], weekly_capacity_hours=10,
             current_load_hours=10, seniority="senior"),
    ]
    a = _action("c1", "Fix", _finding("scorecard_dropped", "scorecard"),
                severity="high", impact=7, effort=1, urgency=4,
                owner_hint="safety-eng")
    plan = RemediationPlan(actions=[a], timestamp="t", total_effort_days=1,
                           quick_wins=[], critical_path=[], notes=[])
    cautious = RemediationAssignmentAdvisor(teams=teams, risk_appetite="cautious").assign(plan)
    aggressive = RemediationAssignmentAdvisor(teams=teams, risk_appetite="aggressive").assign(plan)
    # Cautious: overload_factor clamped to 0.95, alpha already at 100% -> overloaded.
    assert cautious.assignments[0].verdict == "UNASSIGNED_OVERLOADED"
    # Aggressive: overload_factor 1.30 -> not overloaded yet, action lands.
    assert aggressive.assignments[0].verdict in ("ASSIGNED", "ASSIGNED_STRETCH")


def test_determinism():
    teams = _three_teams()
    plan = _basic_plan()
    a = RemediationAssignmentAdvisor(teams=teams).assign(plan)
    b = RemediationAssignmentAdvisor(teams=teams).assign(plan)
    # generated_at can differ; compare everything except that.
    da = a.to_dict()
    db = b.to_dict()
    da.pop("generated_at"); db.pop("generated_at")
    assert json.dumps(da, sort_keys=True) == json.dumps(db, sort_keys=True)


def test_grade_all_overloaded_is_f():
    teams = [
        Team("alpha", skills=["safety-eng"], weekly_capacity_hours=4,
             current_load_hours=10, seniority="senior"),
    ]
    plan = _basic_plan()
    report = RemediationAssignmentAdvisor(teams=teams, overload_factor=1.0).assign(plan)
    assert report.grade == "F"


def test_grade_all_clean_is_a_or_b():
    # Plenty of capacity, perfect skill match, no penalties.
    teams = [
        Team("alpha", skills=["policy"], weekly_capacity_hours=200, seniority="senior"),
        Team("beta", skills=["safety-eng"], weekly_capacity_hours=200, seniority="senior"),
        Team("gamma", skills=["compliance"], weekly_capacity_hours=200, seniority="senior"),
    ]
    report = RemediationAssignmentAdvisor(teams=teams).assign(_basic_plan())
    assert all(a.verdict == "ASSIGNED" for a in report.assignments)
    assert report.grade in ("A", "B")  # no overloads, no needs-hire


def test_playbook_load_balance_when_uneven():
    # alpha starts already past 100% util from current_load_hours and has
    # no skill match for these actions, so it stays overloaded with 0
    # new actions while beta absorbs the work but stays underused.
    teams = [
        Team("alpha", skills=["bizdev"], weekly_capacity_hours=20,
             current_load_hours=30, seniority="senior"),
        Team("beta", skills=["safety-eng", "policy", "compliance"],
             weekly_capacity_hours=200, current_load_hours=0, seniority="senior"),
    ]
    advisor = RemediationAssignmentAdvisor(teams=teams)
    report = advisor.assign(_basic_plan())
    assert report.team_loads["alpha"].overloaded is True
    assert report.team_loads["beta"].utilization_pct < 60.0
    codes = {p.code for p in report.playbook}
    assert "LOAD_BALANCE" in codes


def test_render_and_format_outputs():
    report = RemediationAssignmentAdvisor(teams=_three_teams()).assign(_basic_plan())
    text = report.render()
    assert isinstance(text, str) and len(text) > 0
    md = report.to_markdown()
    assert "##" in md
    j = report.to_json()
    parsed = json.loads(j)
    assert "assignments" in parsed
    assert "team_loads" in parsed
    assert "playbook" in parsed


def test_cli_demo_json_runs():
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["--demo", "--format", "json"])
    assert rc == 0
    out = buf.getvalue().strip()
    # Output is the JSON report (no trailing wrap).
    parsed = json.loads(out)
    assert "assignments" in parsed
    assert "grade" in parsed


def test_empty_teams_raises():
    with pytest.raises(ValueError):
        RemediationAssignmentAdvisor(teams=[])


def test_team_load_dict_shape():
    report = RemediationAssignmentAdvisor(teams=_three_teams()).assign(_basic_plan())
    for name, load in report.team_loads.items():
        d = load.to_dict()
        for k in ("team", "projected_hours", "capacity", "utilization_pct",
                  "overloaded", "action_count", "p0_count"):
            assert k in d
