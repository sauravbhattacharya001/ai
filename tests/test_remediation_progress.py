"""Tests for replication.remediation_progress."""

from __future__ import annotations

import json

import pytest

from replication.remediation_planner import (
    Finding,
    RemediationAction,
    RemediationPlan,
    RemediationPlanner,
)
from replication.remediation_progress import (
    ActionDiff,
    ProgressReport,
    ProgressVelocity,
    Recommendation,
    RemediationProgressTracker,
    SLA_DAYS_BY_SEVERITY,
    _demo_snapshots,
    main,
)


def _plan(findings):
    return RemediationPlanner().plan_from_findings(list(findings))


def test_diff_classifies_resolved_new_persisting():
    prev = _plan([
        Finding(name="preflight",   status="fail", source="quick_scan"),
        Finding(name="policy-lint", status="warn", source="quick_scan"),
        Finding(name="scorecard",   status="fail", source="quick_scan", score=42.0),
    ])
    curr = _plan([
        # preflight resolved
        Finding(name="policy-lint", status="warn", source="quick_scan"),  # persisting
        Finding(name="scorecard",   status="fail", source="quick_scan", score=42.0),  # persisting
        Finding(name="drift",       status="warn", source="adhoc"),  # new
    ])
    report = RemediationProgressTracker().compare(prev, curr, days_between=1.0)
    by_status = {d.action_id: d.status for d in report.diffs}
    assert by_status.get("fix-preflight") == "resolved"
    assert by_status.get("fix-drift") == "new"
    assert by_status.get("fix-policy-lint") == "persisting"
    assert by_status.get("fix-scorecard") == "persisting"


def test_diff_detects_regression_when_severity_worsens():
    # warn (low/medium) → fail with score 10 (critical)
    prev = _plan([Finding(name="scorecard", status="warn", source="qs", score=70.0)])
    curr = _plan([Finding(name="scorecard", status="fail", source="qs", score=10.0)])
    report = RemediationProgressTracker().compare(prev, curr, days_between=1.0)
    d = next(d for d in report.diffs if d.action_id == "fix-scorecard")
    assert d.status == "regressed"
    assert d.previous_severity == "low"
    assert d.current_severity == "critical"


def test_diff_detects_improvement_when_severity_lessens():
    prev = _plan([Finding(name="scorecard", status="fail", source="qs", score=10.0)])  # critical
    curr = _plan([Finding(name="scorecard", status="warn", source="qs", score=70.0)])  # low
    report = RemediationProgressTracker().compare(prev, curr, days_between=1.0)
    d = next(d for d in report.diffs if d.action_id == "fix-scorecard")
    assert d.status == "improving"
    assert d.previous_severity == "critical"
    assert d.current_severity == "low"


def test_slipping_flag_uses_sla_thresholds():
    # high severity SLA = 7d
    prev = _plan([Finding(name="scorecard", status="fail", source="qs", score=50.0)])  # high
    curr = _plan([Finding(name="scorecard", status="fail", source="qs", score=50.0)])  # high
    tracker = RemediationProgressTracker()

    # 1 day old → persisting
    r_young = tracker.compare(prev, curr, days_between=1.0,
                              current_age_days={"fix-scorecard": 1.0})
    assert next(d for d in r_young.diffs).status == "persisting"

    # 10 days old → slipping (SLA high = 7d)
    r_old = tracker.compare(prev, curr, days_between=10.0,
                            current_age_days={"fix-scorecard": 10.0})
    d_old = next(d for d in r_old.diffs)
    assert d_old.status == "slipping"
    assert d_old.age_days == 10.0
    assert SLA_DAYS_BY_SEVERITY["high"] == 7.0


def test_velocity_handles_zero_resolved():
    prev = _plan([Finding(name="scorecard", status="fail", source="qs", score=10.0)])
    curr = _plan([Finding(name="scorecard", status="fail", source="qs", score=10.0)])
    report = RemediationProgressTracker().compare(prev, curr, days_between=3.0,
                                                  current_age_days={"fix-scorecard": 1.0})
    v = report.velocity
    assert v is not None
    assert v.resolved_count == 0
    assert v.resolutions_per_day == 0.0
    assert v.projected_days_to_zero is None


def test_velocity_projects_eta_when_improving():
    # Previous: 4 findings. Current: 0 findings. days_between=2 ⇒ 2.0 days_to_zero
    # But to keep some remaining: previous 5, current 1, days_between 2 ⇒
    #   resolved=4, new=0 ⇒ net=-2/day ⇒ remaining=1 ⇒ 0.5d
    prev = _plan([
        Finding(name="preflight",   status="fail", source="qs"),
        Finding(name="policy-lint", status="warn", source="qs"),
        Finding(name="scorecard",   status="fail", source="qs", score=10.0),
        Finding(name="compliance",  status="warn", source="qs"),
        Finding(name="drift",       status="warn", source="qs"),
    ])
    curr = _plan([
        Finding(name="drift", status="warn", source="qs"),
    ])
    report = RemediationProgressTracker().compare(prev, curr, days_between=2.0)
    v = report.velocity
    assert v is not None
    assert v.resolved_count == 4
    assert v.new_count == 0
    assert v.projected_days_to_zero == pytest.approx(0.5, rel=1e-3)


def test_trajectory_improving():
    prev = _plan([
        Finding(name="preflight",   status="fail", source="qs"),
        Finding(name="policy-lint", status="warn", source="qs"),
        Finding(name="scorecard",   status="warn", source="qs", score=70.0),
    ])
    curr = _plan([
        Finding(name="scorecard", status="warn", source="qs", score=70.0),
    ])
    report = RemediationProgressTracker().compare(prev, curr, days_between=2.0)
    assert report.trajectory == "improving"


def test_trajectory_regressing_when_new_critical_appears():
    prev = _plan([Finding(name="scorecard", status="warn", source="qs", score=80.0)])
    curr = _plan([
        Finding(name="scorecard", status="warn", source="qs", score=80.0),
        Finding(name="root-cause", status="fail", source="qs", score=5.0),  # new critical
    ])
    report = RemediationProgressTracker().compare(prev, curr, days_between=1.0)
    assert report.trajectory == "regressing"


def test_trajectory_at_risk_when_three_slipping():
    prev_findings = [
        Finding(name="preflight",   status="warn", source="qs", score=80.0),
        Finding(name="policy-lint", status="warn", source="qs", score=80.0),
        Finding(name="compliance",  status="warn", source="qs", score=80.0),
    ]
    curr_findings = list(prev_findings)
    prev = _plan(prev_findings)
    curr = _plan(curr_findings)
    ages = {
        "fix-preflight": 100.0,
        "fix-policy-lint": 100.0,
        "fix-compliance": 100.0,
    }
    report = RemediationProgressTracker().compare(prev, curr, days_between=30.0,
                                                  current_age_days=ages)
    assert sum(1 for d in report.diffs if d.status == "slipping") >= 3
    assert report.trajectory == "at_risk"


def test_recommendations_p0_for_slipping_critical():
    prev = _plan([Finding(name="scorecard", status="fail", source="qs", score=10.0)])
    curr = _plan([Finding(name="scorecard", status="fail", source="qs", score=10.0)])
    report = RemediationProgressTracker().compare(
        prev, curr, days_between=10.0,
        current_age_days={"fix-scorecard": 10.0},
    )
    p0s = [r for r in report.recommendations if r.priority == "P0"]
    assert any("Escalate slipping critical" in r.title for r in p0s)


def test_to_markdown_contains_required_sections():
    prev, curr, days = _demo_snapshots()
    report = RemediationProgressTracker().compare(prev, curr, days_between=days)
    md = report.to_markdown()
    for needle in (
        "## ✅ Resolved",
        "## 🔥 Regressed",
        "## 🆕 New",
        "## 🤖 Agentic Recommendations",
        "**Trajectory:**",
    ):
        assert needle in md, f"missing section: {needle!r}"


def test_to_json_round_trip_is_valid_json():
    prev, curr, days = _demo_snapshots()
    report = RemediationProgressTracker().compare(prev, curr, days_between=days)
    data = json.loads(report.to_json())
    assert "trajectory" in data
    assert "velocity" in data
    assert "diffs" in data
    assert "recommendations" in data
    assert isinstance(data["diffs"], list)


def test_compare_findings_runs_end_to_end():
    tracker = RemediationProgressTracker()
    report = tracker.compare_findings(
        previous_findings=[
            Finding(name="preflight", status="fail", source="qs"),
        ],
        current_findings=[
            Finding(name="scorecard", status="fail", source="qs", score=20.0),
        ],
        days_between=2.0,
    )
    statuses = {d.status for d in report.diffs}
    assert "resolved" in statuses  # preflight gone
    assert "new" in statuses  # scorecard appeared


def test_cli_demo_runs_text(capsys):
    rc = main(["--demo"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "TRAJECTORY" in captured
    assert "REMEDIATION PROGRESS REPORT" in captured


def test_cli_demo_json_is_parsable(capsys):
    rc = main(["--demo", "--format", "json"])
    assert rc == 0
    captured = capsys.readouterr().out
    data = json.loads(captured)
    assert "trajectory" in data
    assert "diffs" in data
