"""Tests for replication.safety_debt."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from replication.remediation_planner import Finding
from replication.safety_debt import (
    DebtSnapshot,
    INTEREST_RATE_PER_WEEK,
    SafetyDebtAdvisor,
    SEVERITY_PRINCIPAL,
    _finding_signature,
)


NOW = datetime(2026, 5, 17, 12, 0, 0, tzinfo=timezone.utc)


def _f(name: str, status: str = "fail", score: float | None = None,
       source: str = "scorecard", summary: str = "") -> Finding:
    return Finding(name=name, source=source, status=status, score=score,
                   summary=summary or name)


def _hist(findings: List[Finding], days_ago: float) -> DebtSnapshot:
    ts = (NOW - timedelta(days=days_ago)).isoformat()
    return DebtSnapshot(
        timestamp=ts,
        finding_signatures={_finding_signature(f): ts for f in findings},
    )


# ── basics ──


def test_principal_by_severity_critical():
    f = _f("k", status="fail", score=10)  # critical (score<30)
    adv = SafetyDebtAdvisor()
    report = adv.assess([f], now=NOW)
    assert report.items[0].severity == "critical"
    assert report.items[0].principal == SEVERITY_PRINCIPAL["critical"]


def test_principal_by_severity_low():
    f = _f("k", status="warn", source="quick_scan")  # low (status=warn, no score)
    report = SafetyDebtAdvisor().assess([f], now=NOW)
    assert report.items[0].severity == "low"
    assert report.items[0].principal == SEVERITY_PRINCIPAL["low"]


# ── compounding ──


def test_compound_interest_grows_with_age():
    f = _f("k", status="fail", score=10)  # critical
    hist_fresh = [_hist([f], days_ago=0)]
    hist_old = [_hist([f], days_ago=21)]  # 3 weeks
    fresh = SafetyDebtAdvisor().assess([f], history=hist_fresh, now=NOW)
    old = SafetyDebtAdvisor().assess([f], history=hist_old, now=NOW)
    assert fresh.items[0].accrued_interest == pytest.approx(0.0, abs=1e-6)
    assert old.items[0].accrued_interest > 0.0
    # 3 weeks at 30%/wk => growth ~ 1.3^3 = 2.197 => accrued ~ 1.197 * principal
    expected = SEVERITY_PRINCIPAL["critical"] * (1.30 ** 3 - 1.0)
    assert old.items[0].accrued_interest == pytest.approx(expected, rel=1e-4)


# ── verdict ladder ──


def test_verdict_ladder():
    f = _f("k", status="fail", score=10)  # critical, SLA=3 days
    cases = [
        (0.0, "NEW"),         # first time seen
        (2.0, "CURRENT"),     # <sla
        (5.0, "AGING"),       # 1*sla .. 2*sla
        (8.0, "OVERDUE"),     # 2*sla .. 4*sla
        (15.0, "DEFAULTED"),  # >=4*sla
    ]
    for age, expected in cases:
        if age == 0.0:
            report = SafetyDebtAdvisor().assess([f], history=None, now=NOW)
        else:
            report = SafetyDebtAdvisor().assess(
                [f], history=[_hist([f], days_ago=age)], now=NOW
            )
        assert report.items[0].verdict == expected, (age, expected, report.items[0].verdict)


# ── coverage / capacity ──


def test_coverage_ratio_math():
    f = _f("k", status="fail", score=10)  # critical: principal=20, rate=0.30/wk
    report = SafetyDebtAdvisor().assess(
        [f], history=None, now=NOW,
        team_velocity_points_per_week=12.0, risk_appetite="balanced",
    )
    # weekly burn = 20*0.30 = 6.0, capacity=12 -> coverage 2.0
    assert report.weekly_interest_burn == pytest.approx(6.0)
    assert report.debt_service_capacity == pytest.approx(12.0)
    assert report.coverage_ratio == pytest.approx(2.0)


def test_weeks_to_zero_none_on_spiral():
    findings = [_f(f"k{i}", status="fail", score=10) for i in range(10)]
    report = SafetyDebtAdvisor().assess(
        findings, history=None, now=NOW,
        team_velocity_points_per_week=1.0,
    )
    # 10 criticals -> burn 60/wk, capacity 1 -> spiral
    assert report.weeks_to_zero is None
    assert report.portfolio_health == "BANKRUPT"


# ── health classification ──


def test_health_solvent():
    f = _f("k", status="warn", source="quick_scan")  # low
    report = SafetyDebtAdvisor().assess(
        [f], history=None, now=NOW,
        team_velocity_points_per_week=50.0,
    )
    assert report.portfolio_health == "SOLVENT"
    assert report.grade == "A"


def test_health_bankrupt_two_defaulted_critical():
    findings = [_f(f"k{i}", status="fail", score=10) for i in range(2)]
    # Each "old" to push into DEFAULTED (4*sla = 12 days for critical)
    history = [_hist(findings, days_ago=30)]
    report = SafetyDebtAdvisor().assess(
        findings, history=history, now=NOW,
        team_velocity_points_per_week=200.0,  # high capacity
    )
    # Both DEFAULTED + critical -> BANKRUPT path via defaulted_critical>=2
    assert all(it.verdict == "DEFAULTED" for it in report.items)
    assert report.portfolio_health == "BANKRUPT"
    assert report.grade == "F"


def test_risk_appetite_modulates_capacity():
    f = _f("k", status="fail", score=10)
    base_velocity = 10.0
    c = SafetyDebtAdvisor().assess([f], now=NOW,
                                   team_velocity_points_per_week=base_velocity,
                                   risk_appetite="cautious")
    b = SafetyDebtAdvisor().assess([f], now=NOW,
                                   team_velocity_points_per_week=base_velocity,
                                   risk_appetite="balanced")
    a = SafetyDebtAdvisor().assess([f], now=NOW,
                                   team_velocity_points_per_week=base_velocity,
                                   risk_appetite="aggressive")
    assert c.debt_service_capacity < b.debt_service_capacity < a.debt_service_capacity


# ── trajectory ──


def test_trajectory_stable_no_history():
    f = _f("k", status="warn", source="quick_scan")
    report = SafetyDebtAdvisor().assess([f], history=None, now=NOW)
    assert report.trajectory == "stable"


# ── playbook ──


def test_playbook_emergency_debt_summit_on_bankrupt():
    findings = [_f(f"k{i}", status="fail", score=10) for i in range(2)]
    history = [_hist(findings, days_ago=40)]
    report = SafetyDebtAdvisor().assess(
        findings, history=history, now=NOW,
        team_velocity_points_per_week=200.0,
    )
    ids = [a.id for a in report.playbook]
    assert "EMERGENCY_DEBT_SUMMIT" in ids


def test_playbook_pay_down_defaults_top_three():
    findings = [_f(f"k{i}", status="fail", score=10) for i in range(5)]
    history = [_hist(findings, days_ago=40)]
    report = SafetyDebtAdvisor().assess(
        findings, history=history, now=NOW,
        team_velocity_points_per_week=500.0,
    )
    pay = [a for a in report.playbook if a.id == "PAY_DOWN_DEFAULTS"][0]
    assert len(pay.item_ids) == 3


# ── insights ──


def test_insights_compound_interest_dominant():
    f = _f("k", status="fail", score=10)
    # Make it very old so accrued > principal
    history = [_hist([f], days_ago=70)]  # 10 weeks @ 30% -> grows ~13.79x
    report = SafetyDebtAdvisor().assess([f], history=history, now=NOW,
                                        team_velocity_points_per_week=200.0)
    assert any("compound_interest_dominant" in s for s in report.insights)


def test_insights_debt_spiral_warning():
    findings = [_f(f"k{i}", status="fail", score=10) for i in range(10)]
    report = SafetyDebtAdvisor().assess(
        findings, history=None, now=NOW,
        team_velocity_points_per_week=1.0,
    )
    assert any("debt_spiral_warning" in s for s in report.insights)


# ── exporters ──


def test_to_json_deterministic():
    findings = [_f("k1", status="fail", score=10),
                _f("k2", status="warn", source="quick_scan")]
    history = [_hist(findings, days_ago=15)]
    a = SafetyDebtAdvisor().assess(findings, history=history, now=NOW).to_json()
    b = SafetyDebtAdvisor().assess(findings, history=history, now=NOW).to_json()
    assert a == b
    # ensure sort_keys produced sorted top-level keys
    parsed = json.loads(a)
    assert list(parsed.keys()) == sorted(parsed.keys())


def test_to_markdown_contains_headers():
    f = _f("k", status="fail", score=10)
    md = SafetyDebtAdvisor().assess([f], now=NOW).to_markdown()
    assert "# Safety Debt Report" in md
    assert "## Portfolio" in md
    assert "## Debt items" in md


def test_debt_snapshot_from_findings_round_trip():
    findings = [_f("k1"), _f("k2", source="drift")]
    snap = DebtSnapshot.from_findings(findings, now=NOW)
    assert "scorecard:k1" in snap.finding_signatures
    assert "drift:k2" in snap.finding_signatures
    d = snap.to_dict()
    assert d["timestamp"] == NOW.isoformat()
    assert set(d["finding_signatures"].keys()) == set(snap.finding_signatures.keys())


def test_interest_rate_constants_present():
    for sev in ("critical", "high", "medium", "low", "info"):
        assert sev in INTEREST_RATE_PER_WEEK
        assert sev in SEVERITY_PRINCIPAL
