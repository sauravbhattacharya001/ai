"""Tests for replication.remediation_roi.RemediationROIAdvisor."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from typing import List

import pytest

from replication.remediation_planner import (
    Finding,
    RemediationAction,
    RemediationPlanner,
)
from replication.remediation_roi import (
    APPETITE_COST_MULT,
    RemediationROIAdvisor,
    RemediationROIReport,
    ROIAction,
)


FIXED_NOW = datetime(2026, 5, 18, 9, 30, tzinfo=timezone.utc)


def _fixed_now() -> datetime:
    return FIXED_NOW


def _mk_action(
    *,
    aid: str = "fix-thing",
    severity: str = "high",
    effort: int = 2,
    impact: int = 7,
    urgency: int = 4,
    title: str = "fix the thing",
) -> RemediationAction:
    f = Finding(
        name=aid,
        status="fail" if severity in ("critical", "high") else "warn",
        source="test",
        summary=f"synthetic {severity} finding",
    )
    return RemediationAction(
        id=aid,
        title=title,
        finding=f,
        severity=severity,
        impact=impact,
        effort=effort,
        urgency=urgency,
        rationale="",
    )


def _advisor(**kw) -> RemediationROIAdvisor:
    kw.setdefault("now", _fixed_now)
    return RemediationROIAdvisor(**kw)


# ── tests ──────────────────────────────────────────────────────────


def test_empty_actions_portfolio_healthy() -> None:
    rpt = _advisor().assess([])
    assert rpt.actions == []
    assert rpt.portfolio_grade in ("A", "B")
    assert rpt.headline.startswith("VERDICT:")
    ids = [p.id for p in rpt.playbook]
    assert "PORTFOLIO_HEALTHY" in ids


def test_critical_low_effort_is_quick_win_p0() -> None:
    a = _mk_action(aid="q1", severity="critical", effort=1)
    rpt = _advisor().assess([a])
    ra = rpt.actions[0]
    assert ra.verdict == "QUICK_WIN"
    assert ra.priority == "P0"
    assert "OUTSIZED_RETURN" in ra.reasons
    assert "CRITICAL_SEVERITY" in ra.reasons


def test_critical_high_effort_is_strategic_p0() -> None:
    a = _mk_action(aid="s1", severity="critical", effort=5)
    rpt = _advisor().assess([a])
    ra = rpt.actions[0]
    assert ra.verdict == "STRATEGIC"
    assert ra.priority == "P0"
    assert "HIGH_EFFORT" in ra.reasons


def test_medium_effort_three_is_optional_or_defer() -> None:
    a = _mk_action(aid="m1", severity="medium", effort=3)
    rpt = _advisor().assess([a])
    assert rpt.actions[0].verdict in ("OPTIONAL", "DEFER")


def test_info_high_effort_is_skip() -> None:
    a = _mk_action(aid="i1", severity="info", effort=5)
    rpt = _advisor().assess([a])
    ra = rpt.actions[0]
    assert ra.verdict == "SKIP"
    assert "LOW_ROI_NOT_WORTH_IT" in ra.reasons


def test_risk_appetite_cost_monotonicity() -> None:
    a = _mk_action(aid="c1", severity="high", effort=3)
    cautious = _advisor(risk_appetite="cautious").assess([a]).actions[0].cost_units
    balanced = _advisor(risk_appetite="balanced").assess([a]).actions[0].cost_units
    aggressive = _advisor(risk_appetite="aggressive").assess([a]).actions[0].cost_units
    assert cautious >= balanced >= aggressive
    # And appetite multiplier itself is monotonic.
    assert APPETITE_COST_MULT["cautious"] >= APPETITE_COST_MULT["balanced"]
    assert APPETITE_COST_MULT["balanced"] >= APPETITE_COST_MULT["aggressive"]


def test_aggressive_trims_p3_when_p0_present() -> None:
    actions = [
        _mk_action(aid="q1", severity="critical", effort=1),
        _mk_action(aid="q2", severity="critical", effort=1),
    ]
    rpt = _advisor(risk_appetite="aggressive").assess(actions)
    # Should retain P0 quick-wins playbook and not have lone P3 items.
    prios = [p.priority for p in rpt.playbook]
    assert "P0" in prios
    # No bare P3 except possibly PORTFOLIO_HEALTHY (which only fires when empty).
    for p in rpt.playbook:
        if p.priority == "P3":
            assert p.id == "PORTFOLIO_HEALTHY"


def test_cautious_adds_roi_review_when_grade_low() -> None:
    # Mostly SKIPs -> low ROI -> grade D/F.
    actions = [
        _mk_action(aid=f"lo{i}", severity="info", effort=5) for i in range(3)
    ]
    rpt = _advisor(risk_appetite="cautious").assess(actions)
    assert rpt.portfolio_grade in ("C", "D", "F")
    assert any(p.id == "SCHEDULE_ROI_REVIEW" for p in rpt.playbook)


def test_to_json_is_byte_stable() -> None:
    actions = [
        _mk_action(aid="a", severity="high", effort=1),
        _mk_action(aid="b", severity="medium", effort=2),
    ]
    adv = _advisor()
    r1 = adv.assess(copy.deepcopy(actions))
    r2 = adv.assess(copy.deepcopy(actions))
    j1 = r1.to_json()
    j2 = r2.to_json()
    assert j1 == j2
    # And it parses as valid JSON.
    json.loads(j1)


def test_markdown_has_required_sections() -> None:
    rpt = _advisor().assess([_mk_action()])
    md = rpt.to_markdown()
    assert "## Summary" in md
    assert "## Actions" in md
    assert "## Playbook" in md
    assert "## Insights" in md


def test_text_headline_prefix() -> None:
    rpt = _advisor().assess([_mk_action()])
    text = rpt.render()
    assert "VERDICT:" in text
    assert rpt.headline.startswith("VERDICT:")


def test_inputs_not_mutated() -> None:
    actions: List[RemediationAction] = [_mk_action(aid="x"), _mk_action(aid="y")]
    snapshot = [(a.id, a.title, a.severity, a.effort, a.impact) for a in actions]
    _advisor().assess(actions)
    after = [(a.id, a.title, a.severity, a.effort, a.impact) for a in actions]
    assert snapshot == after
    assert len(actions) == 2


def test_playbook_p0_first_ordering() -> None:
    actions = [
        _mk_action(aid="q1", severity="critical", effort=1),
        _mk_action(aid="q2", severity="critical", effort=1),
        _mk_action(aid="i1", severity="info", effort=5),
        _mk_action(aid="i2", severity="info", effort=5),
    ]
    rpt = _advisor().assess(actions)
    prios = [p.priority for p in rpt.playbook]
    ranks = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    rank_vals = [ranks[p] for p in prios]
    assert rank_vals == sorted(rank_vals)


def test_fund_quick_wins_requires_at_least_two() -> None:
    one = [_mk_action(aid="q1", severity="critical", effort=1)]
    two = [
        _mk_action(aid="q1", severity="critical", effort=1),
        _mk_action(aid="q2", severity="critical", effort=1),
    ]
    rpt1 = _advisor().assess(one)
    rpt2 = _advisor().assess(two)
    assert not any(p.id == "FUND_QUICK_WINS_FIRST" for p in rpt1.playbook)
    assert any(p.id == "FUND_QUICK_WINS_FIRST" for p in rpt2.playbook)


def test_effort_bound_insight_fires() -> None:
    # 5 high-severity high-effort actions vs tiny velocity + short horizon.
    actions = [
        _mk_action(aid=f"h{i}", severity="high", effort=5) for i in range(5)
    ]
    rpt = _advisor(
        team_velocity_points_per_week=1.0,
        horizon_weeks=2.0,
    ).assess(actions)
    assert "EFFORT_BOUND" in rpt.insights


def test_portfolio_roi_matches_arithmetic() -> None:
    # Build two crisp actions; compute expected ROI by hand.
    a1 = _mk_action(aid="x", severity="critical", effort=1)
    a2 = _mk_action(aid="y", severity="high", effort=2)
    rpt = _advisor(risk_appetite="balanced").assess([a1, a2])
    # Selected = anything not SKIP.
    selected = [a for a in rpt.actions if a.verdict != "SKIP"]
    expected_total_debt = sum(a.debt_paid_down for a in selected)
    expected_total_effort = float(sum(a.effort_days for a in selected))
    assert rpt.total_debt_paid_down == pytest.approx(expected_total_debt)
    assert rpt.total_effort_days == pytest.approx(expected_total_effort)
    expected_roi = expected_total_debt / max(expected_total_effort, 0.5)
    assert rpt.portfolio_roi == pytest.approx(expected_roi)


def test_planner_integration_smoke() -> None:
    # End-to-end: build findings -> plan -> ROI.
    findings = [
        Finding(name="kill-switch", status="fail", source="scorecard",
                score=20.0, summary="kill switch race condition"),
        Finding(name="policy-lint", status="warn", source="quick_scan",
                summary="3 rules with overly broad scope"),
        Finding(name="drift", status="fail", source="drift",
                summary="behavior drift across replicas"),
    ]
    plan = RemediationPlanner().plan_from_findings(findings)
    rpt = _advisor().assess(plan, findings=findings)
    assert isinstance(rpt, RemediationROIReport)
    assert len(rpt.actions) == len(plan.actions)
    # Headline format.
    assert "portfolio ROI" in rpt.headline


def test_breakeven_weeks_none_when_velocity_zero() -> None:
    a = _mk_action(aid="z", severity="high", effort=2)
    rpt = _advisor(team_velocity_points_per_week=0).assess([a])
    assert rpt.actions[0].breakeven_weeks is None
    assert rpt.portfolio_breakeven_weeks is None
