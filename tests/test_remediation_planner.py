"""Tests for replication.remediation_planner."""

from __future__ import annotations

import json

import pytest

from replication.remediation_planner import (
    Finding,
    RemediationAction,
    RemediationPlan,
    RemediationPlanner,
    _demo_findings,
    main,
)


def test_finding_severity_buckets() -> None:
    assert Finding(name="x", status="fail", score=10).severity == "critical"
    assert Finding(name="x", status="fail", score=80).severity == "high"
    assert Finding(name="x", status="error").severity == "high"
    assert Finding(name="x", status="warn", score=30).severity == "medium"
    assert Finding(name="x", status="warn", score=90).severity == "low"
    assert Finding(name="x", status="pass").severity == "info"


def test_planner_skips_passing_findings() -> None:
    findings = [
        Finding(name="scorecard", status="pass", score=88.0),
        Finding(name="preflight", status="skip"),
    ]
    plan = RemediationPlanner().plan_from_findings(findings)
    assert plan.actions == []
    assert plan.total_effort_days == 0
    assert any("No actionable findings" in n for n in plan.notes)


def test_planner_orders_blockers_before_dependents() -> None:
    findings = [
        Finding(name="scorecard", status="fail", score=40.0),
        Finding(name="policy-lint", status="warn",
                summary="overly broad scope"),
        Finding(name="preflight", status="fail",
                summary="missing kill_switch"),
    ]
    plan = RemediationPlanner().plan_from_findings(findings)
    ids = [a.id for a in plan.actions]
    # Both preflight and policy-lint must come before scorecard
    assert ids.index("fix-preflight") < ids.index("fix-scorecard")
    assert ids.index("fix-policy-lint") < ids.index("fix-scorecard")
    # Scorecard action should record both as dependencies
    sc = next(a for a in plan.actions if a.id == "fix-scorecard")
    assert "fix-preflight" in sc.depends_on
    assert "fix-policy-lint" in sc.depends_on


def test_priority_is_impact_times_urgency_over_effort() -> None:
    finding = Finding(name="scorecard", status="fail", score=20.0)
    action = RemediationPlanner()._action_for(finding)
    expected = round((action.impact * action.urgency) / max(action.effort, 1), 3)
    assert action.priority == expected
    assert action.severity == "critical"
    assert action.urgency == 5


def test_quick_win_classification() -> None:
    # Low effort, high impact, no deps → quick win
    quick = Finding(name="policy-lint", status="warn",
                    summary="broad scope on critical rule")
    qa = RemediationPlanner()._action_for(quick)
    qa.impact = 7
    qa.effort = 1
    qa.depends_on = []
    assert qa.is_quick_win

    # Same impact/effort but with a blocker → not a quick win
    qa.depends_on = ["fix-preflight"]
    assert not qa.is_quick_win


def test_critical_path_is_longest_chain() -> None:
    plan = RemediationPlanner().plan_from_findings([
        Finding(name="preflight",  status="fail",
                summary="missing config"),
        Finding(name="policy-lint", status="warn",
                summary="rule too broad"),
        Finding(name="scorecard",  status="fail", score=30.0,
                summary="Grade: F"),
        Finding(name="regression", status="fail", score=25.0,
                summary="22% drop"),
    ])
    cp = plan.critical_path
    # regression depends on scorecard, scorecard on preflight + policy-lint
    assert "fix-regression" in cp
    assert "fix-scorecard" in cp
    assert cp.index("fix-scorecard") < cp.index("fix-regression")


def test_to_dict_round_trips_via_json() -> None:
    plan = RemediationPlanner().plan_from_findings(_demo_findings())
    blob = plan.to_json()
    parsed = json.loads(blob)
    assert parsed["action_count"] == len(plan.actions)
    assert parsed["total_effort_days"] == plan.total_effort_days
    assert {a["id"] for a in parsed["actions"]} == {a.id for a in plan.actions}


def test_markdown_render_includes_sections() -> None:
    plan = RemediationPlanner().plan_from_findings(_demo_findings())
    md = plan.to_markdown()
    assert "# Safety Remediation Plan" in md
    assert "## 📋 Ordered Actions" in md
    assert "## 🔍 Action Details" in md


def test_text_render_handles_empty_plan() -> None:
    plan = RemediationPlanner().plan_from_findings([])
    txt = plan.to_text()
    assert "No findings to remediate" in txt


def test_top_filter_truncates_actions() -> None:
    plan = RemediationPlanner(max_actions=2).plan_from_findings(_demo_findings())
    assert len(plan.actions) == 2


def test_quick_wins_only_filter() -> None:
    plan = RemediationPlanner(quick_wins_only=True).plan_from_findings(
        _demo_findings()
    )
    for a in plan.actions:
        assert a.is_quick_win


def test_plan_from_quick_scan_dict_shape() -> None:
    scan = {
        "passed": False,
        "checks": [
            {"name": "preflight",  "status": "fail",
             "summary": "missing key"},
            {"name": "scorecard",  "status": "warn", "score": 65.0,
             "summary": "Grade: C"},
            {"name": "compliance", "status": "pass", "score": 95.0},
        ],
    }
    plan = RemediationPlanner().plan_from_quick_scan_dict(scan)
    ids = [a.id for a in plan.actions]
    assert "fix-preflight" in ids
    assert "fix-scorecard" in ids
    # pass status skipped
    assert "fix-compliance" not in ids


def test_cli_demo_text(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["--demo", "--format", "text"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "SAFETY REMEDIATION PLAN" in captured


def test_cli_demo_json(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["--demo", "--format", "json"])
    assert rc == 0
    captured = capsys.readouterr().out
    parsed = json.loads(captured)
    assert "actions" in parsed
    assert parsed["action_count"] >= 1


def test_cli_writes_to_output(tmp_path) -> None:
    out = tmp_path / "plan.md"
    rc = main(["--demo", "--format", "md", "--output", str(out)])
    assert rc == 0
    assert out.exists()
    body = out.read_text(encoding="utf-8")
    assert "# Safety Remediation Plan" in body


def test_cli_from_json(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    scan = {"checks": [
        {"name": "preflight", "status": "fail", "summary": "x"},
    ]}
    p = tmp_path / "scan.json"
    p.write_text(json.dumps(scan), encoding="utf-8")
    rc = main(["--from-json", str(p), "--format", "json"])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert any(a["id"] == "fix-preflight" for a in parsed["actions"])
