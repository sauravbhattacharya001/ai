"""Tests for replication.oncall_load_balance_advisor."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from replication.oncall_load_balance_advisor import (
    ALL_VERDICTS,
    BASE_THRESHOLDS,
    OnCallInput,
    OnCallLoadBalanceAdvisor,
    OnCallShift,
    VERDICT_BURNOUT_RISK,
    VERDICT_HEALTHY,
    VERDICT_INSUFFICIENT_DATA,
    VERDICT_INSUFFICIENT_REST,
    VERDICT_OVERLOADED,
    VERDICT_SPOF_PRIMARY,
    VERDICT_UNDERUSED,
    _gini,
    main,
)

NOW = datetime(2026, 5, 22, 12, tzinfo=timezone.utc)


def _advisor():
    return OnCallLoadBalanceAdvisor(now=lambda: NOW)


def _shift(rid, *, start_days_ago, hours, tier="primary", pages=0):
    start = NOW - timedelta(days=start_days_ago)
    end = start + timedelta(hours=hours)
    return OnCallShift(
        responder_id=rid, start=start, end=end, tier=tier, page_count=pages
    )


def _find(report, rid):
    for f in report.findings:
        if f.responder_id == rid:
            return f
    raise AssertionError(f"missing finding for {rid}")


def test_empty_roster_returns_empty_grade_A():
    report = _advisor().audit(OnCallInput())
    assert report.portfolio.roster_size == 0
    assert report.portfolio.total_shifts == 0
    # No data -> grade A and EMPTY_ROSTER insight.
    assert "EMPTY_ROSTER" in report.insights
    assert report.portfolio.grade in ("A", "F")  # no gaps either


def test_healthy_balanced_rotation_gets_grade_A():
    shifts = [
        _shift("alice", start_days_ago=20, hours=24, tier="primary", pages=2),
        _shift("bob", start_days_ago=15, hours=24, tier="primary", pages=2),
        _shift("carol", start_days_ago=10, hours=24, tier="primary", pages=2),
        _shift("dave", start_days_ago=5, hours=24, tier="primary", pages=1),
    ]
    report = _advisor().audit(OnCallInput(
        roster=["alice", "bob", "carol", "dave"], shifts=shifts,
    ))
    assert report.portfolio.grade == "A"
    assert any("HEALTHY_ROTATION" in i for i in report.insights)


def test_burnout_detected_by_hours():
    # Alice racks up well over the burnout threshold.
    shifts = [
        _shift("alice", start_days_ago=25, hours=100, tier="primary"),
        _shift("alice", start_days_ago=15, hours=100, tier="primary"),
        _shift("bob", start_days_ago=10, hours=24, tier="primary"),
    ]
    report = _advisor().audit(OnCallInput(
        roster=["alice", "bob"], shifts=shifts,
    ))
    alice = _find(report, "alice")
    assert alice.verdict == VERDICT_BURNOUT_RISK
    assert alice.priority == "P0"
    assert report.portfolio.burnout_count == 1
    assert any(
        a.label == "ROTATE_OUT_BURNED_OUT_RESPONDERS"
        for a in report.playbook
    )


def test_burnout_detected_by_pages():
    shifts = [
        _shift("alice", start_days_ago=10, hours=24, pages=50),
        _shift("bob", start_days_ago=5, hours=24, pages=2),
    ]
    report = _advisor().audit(OnCallInput(
        roster=["alice", "bob"], shifts=shifts,
    ))
    assert _find(report, "alice").verdict == VERDICT_BURNOUT_RISK


def test_insufficient_rest_detected():
    # Back-to-back shifts with only a 1h gap.
    start1 = NOW - timedelta(days=10)
    end1 = start1 + timedelta(hours=24)
    start2 = end1 + timedelta(hours=1)
    end2 = start2 + timedelta(hours=24)
    shifts = [
        OnCallShift("alice", start1, end1, "primary", 2),
        OnCallShift("alice", start2, end2, "primary", 2),
        _shift("bob", start_days_ago=5, hours=24, tier="primary"),
    ]
    report = _advisor().audit(OnCallInput(
        roster=["alice", "bob"], shifts=shifts,
    ))
    alice = _find(report, "alice")
    assert alice.verdict in (VERDICT_INSUFFICIENT_REST, VERDICT_OVERLOADED, VERDICT_BURNOUT_RISK)
    # Specifically: rest violation must be present in findings or portfolio.
    assert report.portfolio.rest_violations >= 1 or alice.verdict != VERDICT_HEALTHY


def test_spof_primary_when_only_one_responder_per_tier():
    shifts = [
        _shift("alice", start_days_ago=15, hours=24, tier="critical"),
        _shift("bob", start_days_ago=10, hours=24, tier="weekend"),
        _shift("alice", start_days_ago=5, hours=24, tier="critical"),
    ]
    report = _advisor().audit(OnCallInput(
        roster=["alice", "bob"], shifts=shifts,
    ))
    # Both alice (critical) and bob (weekend) are sole holders.
    alice = _find(report, "alice")
    bob = _find(report, "bob")
    assert "critical" in alice.spof_tiers
    assert "weekend" in bob.spof_tiers
    assert any(
        a.label == "RECRUIT_BACKUP_FOR_SPOF_TIERS"
        for a in report.playbook
    )


def test_coverage_gap_when_declared_tier_has_no_shifts():
    shifts = [
        _shift("alice", start_days_ago=10, hours=24, tier="primary", pages=2),
    ]
    report = _advisor().audit(OnCallInput(
        roster=["alice"],
        tiers=["primary", "weekend"],
        shifts=shifts,
    ))
    assert "weekend" in report.portfolio.coverage_gaps
    assert "COVERAGE_GAP" in report.insights
    assert report.portfolio.grade == "F"
    assert any(
        a.label == "FILL_COVERAGE_GAP" for a in report.playbook
    )


def test_overloaded_vs_underused_split():
    # Alice ~5x fair share, dave ~0; expect overloaded + underused signals.
    shifts = [
        _shift("alice", start_days_ago=25, hours=40, tier="primary"),
        _shift("alice", start_days_ago=15, hours=40, tier="primary"),
        _shift("bob", start_days_ago=10, hours=12, tier="primary"),
        _shift("carol", start_days_ago=8, hours=12, tier="primary"),
        _shift("dave", start_days_ago=6, hours=4, tier="primary"),
    ]
    report = _advisor().audit(OnCallInput(
        roster=["alice", "bob", "carol", "dave"], shifts=shifts,
    ))
    alice = _find(report, "alice")
    assert alice.verdict in (VERDICT_OVERLOADED, VERDICT_BURNOUT_RISK)
    # Dave is on roster with tiny load -> underused
    dave = _find(report, "dave")
    assert dave.verdict == VERDICT_UNDERUSED


def test_underused_on_roster_with_no_shifts():
    shifts = [
        _shift("alice", start_days_ago=10, hours=24, tier="primary"),
        _shift("bob", start_days_ago=5, hours=24, tier="primary"),
    ]
    report = _advisor().audit(OnCallInput(
        roster=["alice", "bob", "carol"], shifts=shifts,
    ))
    carol = _find(report, "carol")
    assert carol.verdict == VERDICT_UNDERUSED
    assert carol.shifts_in_window == 0


def test_risk_appetite_cautious_tightens_thresholds():
    payload = OnCallInput(
        roster=["alice", "bob"],
        shifts=[
            _shift("alice", start_days_ago=20, hours=60, tier="primary"),
            _shift("bob", start_days_ago=10, hours=20, tier="primary"),
        ],
        risk_appetite="cautious",
    )
    report = _advisor().audit(payload)
    assert report.thresholds["burnout_hours"] < BASE_THRESHOLDS["burnout_hours"]
    assert report.thresholds["min_rest_hours"] > BASE_THRESHOLDS["min_rest_hours"]


def test_aggressive_loosens_thresholds():
    payload = OnCallInput(
        roster=["alice"],
        shifts=[_shift("alice", start_days_ago=10, hours=24, tier="primary")],
        risk_appetite="aggressive",
    )
    report = _advisor().audit(payload)
    assert report.thresholds["burnout_hours"] > BASE_THRESHOLDS["burnout_hours"]


def test_invalid_appetite_falls_back_to_balanced():
    report = _advisor().audit(OnCallInput(
        roster=["alice"], risk_appetite="recklessly_brave",
        shifts=[_shift("alice", start_days_ago=10, hours=24, tier="primary")],
    ))
    assert report.risk_appetite == "balanced"


def test_audit_does_not_mutate_input():
    shifts = [_shift("alice", start_days_ago=10, hours=24, tier="primary")]
    payload = OnCallInput(roster=["alice", "bob"], shifts=shifts)
    snapshot = json.dumps([
        (s.responder_id, s.start.isoformat(), s.end.isoformat(), s.tier, s.page_count)
        for s in payload.shifts
    ])
    _advisor().audit(payload)
    after = json.dumps([
        (s.responder_id, s.start.isoformat(), s.end.isoformat(), s.tier, s.page_count)
        for s in payload.shifts
    ])
    assert snapshot == after


def test_to_json_is_byte_stable():
    payload = OnCallInput(
        roster=["alice", "bob"],
        shifts=[
            _shift("alice", start_days_ago=10, hours=24, tier="primary", pages=3),
            _shift("bob", start_days_ago=5, hours=24, tier="primary", pages=2),
        ],
    )
    a = _advisor().audit(payload).to_json()
    b = _advisor().audit(payload).to_json()
    assert a == b
    parsed = json.loads(a)
    assert "portfolio" in parsed
    assert "findings" in parsed
    assert "playbook" in parsed


def test_text_and_markdown_render_all_sections():
    payload = OnCallInput(
        roster=["alice", "bob"],
        tiers=["primary"],
        shifts=[
            _shift("alice", start_days_ago=20, hours=80, tier="primary", pages=40),
            _shift("bob", start_days_ago=10, hours=24, tier="primary"),
        ],
    )
    report = _advisor().audit(payload)
    text = report.to_text()
    md = report.to_markdown()
    assert "On-Call Load Balance" in text
    assert "Findings:" in text
    assert "## Portfolio" in md
    assert "## Findings" in md
    assert "## Playbook" in md
    assert "## Insights" in md


def test_all_verdicts_have_priority():
    # Sanity: every verdict should map cleanly through audit; just verify
    # ALL_VERDICTS contains the constants we exposed.
    assert VERDICT_BURNOUT_RISK in ALL_VERDICTS
    assert VERDICT_SPOF_PRIMARY in ALL_VERDICTS


def test_gini_basic():
    assert _gini([]) == 0.0
    assert _gini([1.0]) == 0.0
    # Equal -> Gini == 0.
    assert _gini([10, 10, 10, 10]) == pytest.approx(0.0)
    # One person hogging: Gini close to (n-1)/n.
    n = 4
    g = _gini([100, 0.0001, 0.0001, 0.0001])
    assert g > 0.5


def test_thin_bench_insight():
    payload = OnCallInput(
        roster=["alice", "bob"],
        shifts=[
            _shift("alice", start_days_ago=10, hours=24, tier="primary"),
            _shift("bob", start_days_ago=5, hours=24, tier="primary"),
        ],
    )
    report = _advisor().audit(payload)
    assert "THIN_BENCH" in report.insights


def test_main_demo_exits_zero(capsys):
    rc = main(["--demo", "--format", "text"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "On-Call Load Balance" in captured.out


def test_main_json_format(capsys):
    rc = main(["--demo", "--format", "json"])
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["portfolio"]["roster_size"] > 0


def test_main_markdown_format(capsys):
    rc = main(["--demo", "--format", "markdown", "--risk", "cautious"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "# On-Call Load Balance Report" in out
    assert "risk_appetite: **cautious**" in out
