"""Tests for ``replication.remediation_sprint_planner``."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from replication.remediation_sprint_planner import (
    APPETITES,
    DEFAULT_CAPACITY_PER_OWNER,
    RemediationSprintPlanner,
    SprintAssignment,
    SprintPlanReport,
    UNASSIGNED_OWNER,
)


FIXED_NOW = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)


def _planner() -> RemediationSprintPlanner:
    return RemediationSprintPlanner(now_fn=lambda: FIXED_NOW)


def _action(aid, sev="medium", effort=2, urgency=3, deps=None, owner="alice"):
    return {
        "id": aid,
        "title": aid,
        "severity": sev,
        "effort": effort,
        "urgency": urgency,
        "depends_on": list(deps or []),
        "owner_hint": owner,
    }


def test_empty_plan_emits_empty_insight_and_grade_A():
    r = _planner().plan([])
    assert r.assignments == []
    assert r.portfolio.grade == "A"
    assert "EMPTY_PLAN" in r.insights


def test_single_small_p0_action_lands_in_sprint_0():
    r = _planner().plan([_action("fix", sev="critical", effort=1)])
    assert len(r.assignments) == 1
    asg = r.assignments[0]
    assert asg.sprint_index == 0
    assert asg.verdict == "READY_NOW"
    assert asg.priority == "P0"
    assert r.portfolio.scheduled == 1


def test_oversized_action_flagged_and_playbook_split():
    # cap default = 10 d. Effort 30 -> oversized for any owner.
    r = _planner().plan([_action("big", sev="critical", effort=30)])
    asg = r.assignments[0]
    assert asg.verdict == "OVERSIZED_ACTION"
    assert asg.priority == "P1"
    ids = {p.id for p in r.playbook}
    assert "SPLIT_OVERSIZED_ACTIONS" in ids


def test_dependency_chain_layered_into_sprints():
    acts = [
        _action("A", sev="high", effort=1),
        _action("B", sev="high", effort=1, deps=["A"]),
        _action("C", sev="high", effort=1, deps=["B"]),
    ]
    r = _planner().plan(acts)
    sprint_by_id = {a.action_id: a.sprint_index for a in r.assignments}
    assert sprint_by_id["A"] == 0
    assert sprint_by_id["B"] == 1
    assert sprint_by_id["C"] == 2


def test_circular_dependency_emits_blocked():
    acts = [
        _action("A", deps=["B"]),
        _action("B", deps=["A"]),
    ]
    r = _planner().plan(acts)
    verdicts = {a.action_id: a.verdict for a in r.assignments}
    assert verdicts["A"] == "BLOCKED_BY_DEPENDENCY"
    assert verdicts["B"] == "BLOCKED_BY_DEPENDENCY"


def test_horizon_2_defers_low_priority_overflow():
    # 6 low-severity small actions, single owner with cap 4/sprint -> only 4 fit
    # in 2 sprints? cap=4 * 2 = 8 d total. 6 * 2d each = 12d -> 2 deferred.
    acts = [_action(f"x{i}", sev="low", effort=2, owner="solo") for i in range(6)]
    r = _planner().plan(
        acts,
        horizon_sprints=2,
        default_capacity_per_owner=4,
    )
    deferred = [a for a in r.assignments if a.verdict == "DEFERRED_PAST_HORIZON"]
    assert len(deferred) == 2
    ids = {p.id for p in r.playbook}
    assert "EXPAND_HORIZON" in ids


def test_per_owner_capacity_respected():
    # Each owner only has 3d capacity.
    acts = [
        _action("a1", effort=2, owner="alice"),
        _action("a2", effort=2, owner="alice"),  # alice's sprint 0 full after a1
        _action("b1", effort=2, owner="bob"),
    ]
    r = _planner().plan(acts, default_capacity_per_owner=3, horizon_sprints=4)
    by_id = {a.action_id: a for a in r.assignments}
    assert by_id["a1"].sprint_index == 0
    # alice has only 1d left in sprint 0, a2 needs 2d -> sprint 1
    assert by_id["a2"].sprint_index == 1
    # bob is independent
    assert by_id["b1"].sprint_index == 0


def test_cautious_shrinks_capacity_vs_balanced():
    # 6 actions each 2d, single owner cap=10 -> balanced fits all in sprint 0 (10d).
    # Cautious shrinks to 8d -> only 4 fit, 1 spills to sprint 1, etc.
    acts = [_action(f"x{i}", sev="low", effort=2, owner="solo") for i in range(6)]
    bal = _planner().plan(acts, horizon_sprints=4, default_capacity_per_owner=10,
                          risk_appetite="balanced")
    cau = _planner().plan(acts, horizon_sprints=4, default_capacity_per_owner=10,
                          risk_appetite="cautious")
    bal_s0 = sum(1 for a in bal.assignments if a.sprint_index == 0)
    cau_s0 = sum(1 for a in cau.assignments if a.sprint_index == 0)
    assert cau_s0 < bal_s0


def test_aggressive_expands_capacity_vs_balanced():
    acts = [_action(f"x{i}", sev="low", effort=2, owner="solo") for i in range(7)]
    bal = _planner().plan(acts, horizon_sprints=2, default_capacity_per_owner=5,
                          risk_appetite="balanced")
    agg = _planner().plan(acts, horizon_sprints=2, default_capacity_per_owner=5,
                          risk_appetite="aggressive")
    bal_def = sum(1 for a in bal.assignments if a.verdict == "DEFERRED_PAST_HORIZON")
    agg_def = sum(1 for a in agg.assignments if a.verdict == "DEFERRED_PAST_HORIZON")
    assert agg_def <= bal_def


def test_unassigned_actions_trigger_assign_owner_playbook():
    acts = [_action("ghost", owner="")]
    r = _planner().plan(acts)
    assert r.assignments[0].owner == UNASSIGNED_OWNER
    ids = {p.id for p in r.playbook}
    assert "ASSIGN_OWNER_TO_UNASSIGNED_WORK" in ids


def test_invalid_risk_appetite_raises():
    with pytest.raises(ValueError):
        _planner().plan([], risk_appetite="paranoid")


def test_input_immutability():
    acts = [_action("a", deps=["x"])]
    snapshot = copy.deepcopy(acts)
    _planner().plan(acts)
    assert acts == snapshot


def test_to_json_byte_stable():
    acts = [
        _action("a", sev="high", effort=3),
        _action("b", sev="medium", effort=2, deps=["a"]),
    ]
    r1 = _planner().plan(acts)
    r2 = _planner().plan(acts)
    assert r1.to_json() == r2.to_json()
    # also: parseable
    json.loads(r1.to_json())


def test_to_markdown_contains_all_sections():
    r = _planner().plan([_action("a", sev="critical", effort=1)])
    md = r.to_markdown()
    for section in ("## Summary", "## Sprints", "## Assignments",
                    "## Playbook", "## Insights"):
        assert section in md


def test_to_text_headline_format():
    r = _planner().plan([_action("a", sev="critical", effort=1)])
    text = r.to_text()
    assert "Remediation Sprint Plan" in text
    assert "grade=" in text
    assert "appetite=balanced" in text


def test_grade_A_for_small_balanced_load():
    r = _planner().plan([_action("a", sev="low", effort=1)])
    assert r.portfolio.grade == "A"


def test_grade_F_when_p0_deferred_past_horizon():
    # Single owner, cap 2d, horizon 1 sprint; 3 P0 actions of 2d each -> 2 deferred.
    acts = [_action(f"crit{i}", sev="critical", effort=2, owner="solo") for i in range(3)]
    r = _planner().plan(
        acts, horizon_sprints=1, default_capacity_per_owner=2,
    )
    assert r.portfolio.grade == "F"
    insufficients = [a for a in r.assignments if a.verdict == "INSUFFICIENT_CAPACITY"]
    assert len(insufficients) >= 1


def test_cli_demo_exits_zero(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    env = {**__import__("os").environ, "PYTHONPATH": str(repo_root / "src")}
    proc = subprocess.run(
        [sys.executable, "-m", "replication.remediation_sprint_planner",
         "--demo", "--format", "markdown"],
        capture_output=True, text=True, env=env, cwd=str(repo_root),
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "# Remediation Sprint Plan" in proc.stdout


def test_appetites_constant():
    assert set(APPETITES) == {"cautious", "balanced", "aggressive"}
