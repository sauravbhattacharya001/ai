"""Tests for the agentic Remediation Staleness Advisor."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from replication.remediation_staleness import (
    APPETITES,
    RemediationStalenessAdvisor,
    StalenessAction,
    StalenessInput,
)


FIXED_NOW = datetime(2026, 5, 21, 12, 0, 0, tzinfo=timezone.utc)


def _advisor() -> RemediationStalenessAdvisor:
    return RemediationStalenessAdvisor(now_fn=lambda: FIXED_NOW)


def _ago(days: float) -> datetime:
    return FIXED_NOW - timedelta(days=days)


def _ahead(days: float) -> datetime:
    return FIXED_NOW + timedelta(days=days)


def test_empty_board_is_healthy_with_grade_a():
    report = _advisor().audit(StalenessInput(actions=[]))
    assert report.portfolio.total_actions == 0
    assert report.portfolio.grade == "A"
    assert report.insights == ["EMPTY_BOARD"]
    # Single P3 fallback action.
    assert any(a.id == "HEALTHY_BOARD" for a in report.playbook)


def test_overdue_action_classified_p0():
    payload = StalenessInput(actions=[
        StalenessAction(
            id="A", owner="alice", severity="high", status="open",
            opened_at=_ago(10), last_updated_at=_ago(2),
            due_at=_ago(2),
        ),
    ])
    report = _advisor().audit(payload)
    f = next(f for f in report.findings if f.action_id == "A")
    assert f.verdict == "OVERDUE"
    assert f.priority == "P0"
    assert any(a.id == "ESCALATE_OVERDUE_ACTIONS" for a in report.playbook)


def test_abandoned_when_old_and_no_updates():
    payload = StalenessInput(actions=[
        StalenessAction(
            id="A", owner="alice", status="open", severity="medium",
            opened_at=_ago(120), last_updated_at=_ago(90),
        ),
    ])
    report = _advisor().audit(payload)
    f = report.findings[0]
    assert f.verdict == "ABANDONED"
    assert f.priority == "P0"
    assert "ABANDONED_CLUSTER:2" not in report.insights  # only one
    assert any(a.id == "TRIAGE_ABANDONED_ACTIONS" for a in report.playbook)


def test_blocked_no_owner_for_open_action_without_owner():
    payload = StalenessInput(actions=[
        StalenessAction(
            id="A", owner=None, status="open", severity="medium",
            opened_at=_ago(3), last_updated_at=_ago(1),
        ),
    ])
    report = _advisor().audit(payload)
    f = report.findings[0]
    assert f.verdict == "BLOCKED_NO_OWNER"
    # medium severity -> P1
    assert f.priority == "P1"


def test_blocked_no_owner_critical_is_p0():
    payload = StalenessInput(actions=[
        StalenessAction(
            id="A", owner=None, status="open", severity="critical",
            opened_at=_ago(3), last_updated_at=_ago(1),
        ),
    ])
    report = _advisor().audit(payload)
    f = report.findings[0]
    assert f.verdict == "BLOCKED_NO_OWNER"
    assert f.priority == "P0"


def test_at_risk_when_due_soon():
    payload = StalenessInput(actions=[
        StalenessAction(
            id="A", owner="bob", status="in_progress", severity="high",
            opened_at=_ago(5), last_updated_at=_ago(1),
            due_at=_ahead(2),
        ),
    ])
    report = _advisor().audit(payload)
    assert report.findings[0].verdict == "AT_RISK"
    assert report.findings[0].priority == "P1"


def test_stale_when_open_too_long():
    payload = StalenessInput(actions=[
        StalenessAction(
            id="A", owner="carol", status="open", severity="medium",
            opened_at=_ago(30), last_updated_at=_ago(1),
        ),
    ])
    report = _advisor().audit(payload)
    assert report.findings[0].verdict == "STALE"


def test_idle_when_no_recent_updates():
    payload = StalenessInput(actions=[
        StalenessAction(
            id="A", owner="dave", status="open", severity="medium",
            opened_at=_ago(10), last_updated_at=_ago(9),
        ),
    ])
    report = _advisor().audit(payload)
    # 10d open exceeds default stale_days (21d * balanced=21) -> false; idle
    # threshold 7d -> exceeded.
    assert report.findings[0].verdict == "IDLE"


def test_recently_completed_action_listed_but_not_in_overdue_counts():
    payload = StalenessInput(actions=[
        StalenessAction(
            id="A", owner="alice", status="done", severity="low",
            opened_at=_ago(5), last_updated_at=_ago(1), closed_at=_ago(1),
        ),
    ])
    report = _advisor().audit(payload)
    assert report.portfolio.done_recent == 1
    assert report.portfolio.overdue == 0
    assert report.findings[0].verdict == "RECENTLY_COMPLETED"


def test_old_completed_action_omitted_entirely():
    payload = StalenessInput(actions=[
        StalenessAction(
            id="A", owner="alice", status="done", severity="low",
            opened_at=_ago(120), last_updated_at=_ago(90), closed_at=_ago(90),
        ),
    ])
    report = _advisor().audit(payload)
    assert report.findings == []


def test_dropped_actions_counted_separately():
    payload = StalenessInput(actions=[
        StalenessAction(
            id="A", owner="alice", status="wontfix", severity="medium",
            opened_at=_ago(30), last_updated_at=_ago(10),
        ),
    ])
    report = _advisor().audit(payload)
    assert report.portfolio.dropped == 1
    assert report.findings == []


def test_insufficient_data_when_no_timestamps_but_has_owner():
    payload = StalenessInput(actions=[
        StalenessAction(id="A", owner="alice", status="open", severity="medium"),
    ])
    report = _advisor().audit(payload)
    f = report.findings[0]
    assert f.verdict == "INSUFFICIENT_DATA"
    assert any(a.id == "BACKFILL_TIMESTAMPS" for a in report.playbook)


def test_owner_overload_triggers_playbook_and_insight():
    payload = StalenessInput(
        actions=[
            StalenessAction(
                id=f"A{i}", owner="alice", status="open", severity="low",
                opened_at=_ago(2), last_updated_at=_ago(1),
            )
            for i in range(4)
        ],
        wip_cap_per_owner={"alice": 2},
    )
    report = _advisor().audit(payload)
    assert "alice" in report.portfolio.overloaded_owners
    assert any(a.id == "REBALANCE_OWNER_LOAD" for a in report.playbook)
    assert any(ins.startswith("OWNER_OVERLOAD:") for ins in report.insights)


def test_cautious_appetite_inflates_score_vs_aggressive():
    base = StalenessInput(actions=[
        StalenessAction(
            id="A", owner="alice", status="open", severity="medium",
            opened_at=_ago(10), last_updated_at=_ago(9),
        ),
    ])
    cautious = _advisor().audit(StalenessInput(
        actions=base.actions, risk_appetite="cautious"))
    aggressive = _advisor().audit(StalenessInput(
        actions=base.actions, risk_appetite="aggressive"))
    assert (
        cautious.findings[0].staleness_score
        > aggressive.findings[0].staleness_score
    )


def test_aggressive_trims_p3_padding_when_higher_priority_exists():
    payload = StalenessInput(
        actions=[
            StalenessAction(
                id="A", owner=None, status="open", severity="high",
                opened_at=_ago(3), last_updated_at=_ago(1),
            ),
        ],
        risk_appetite="aggressive",
    )
    report = _advisor().audit(payload)
    priorities = {a.priority for a in report.playbook}
    assert "P3" not in priorities


def test_cautious_appends_board_review_on_degraded_grade():
    payload = StalenessInput(
        actions=[
            StalenessAction(
                id="A", owner="alice", status="open", severity="critical",
                opened_at=_ago(40), last_updated_at=_ago(30),
                due_at=_ago(5),
            ),
        ],
        risk_appetite="cautious",
    )
    report = _advisor().audit(payload)
    assert report.portfolio.grade in ("C", "D", "F")
    assert any(a.id == "SCHEDULE_BOARD_REVIEW" for a in report.playbook)


def test_json_render_is_deterministic_and_byte_stable():
    payload = StalenessInput(actions=[
        StalenessAction(
            id="A", owner="alice", status="open", severity="medium",
            opened_at=_ago(10), last_updated_at=_ago(9),
        ),
        StalenessAction(
            id="B", owner="bob", status="open", severity="high",
            opened_at=_ago(30), last_updated_at=_ago(1),
        ),
    ])
    a = _advisor().audit(payload).to_json()
    b = _advisor().audit(payload).to_json()
    assert a == b
    parsed = json.loads(a)
    # Top-level keys are present and sorted.
    assert list(parsed.keys()) == sorted(parsed.keys())


def test_markdown_render_has_all_sections():
    payload = StalenessInput(actions=[
        StalenessAction(
            id="A", owner="alice", status="open", severity="medium",
            opened_at=_ago(10), last_updated_at=_ago(9),
        ),
    ])
    md = _advisor().audit(payload).to_markdown()
    for section in ("## Summary", "## Findings", "## Playbook", "## Insights"):
        assert section in md


def test_input_is_not_mutated():
    actions = [
        StalenessAction(
            id="A", owner="alice", status="open", severity="medium",
            opened_at=_ago(10), last_updated_at=_ago(9),
        ),
    ]
    snapshot = (actions[0].id, actions[0].owner, actions[0].status,
                actions[0].opened_at, actions[0].last_updated_at)
    _advisor().audit(StalenessInput(actions=actions))
    after = (actions[0].id, actions[0].owner, actions[0].status,
             actions[0].opened_at, actions[0].last_updated_at)
    assert snapshot == after


def test_unknown_risk_appetite_falls_back_to_balanced():
    payload = StalenessInput(
        actions=[
            StalenessAction(
                id="A", owner="alice", status="open", severity="medium",
                opened_at=_ago(10), last_updated_at=_ago(9),
            ),
        ],
        risk_appetite="bogus",
    )
    report = _advisor().audit(payload)
    assert report.risk_appetite == "balanced"


def test_priority_rank_orders_findings_correctly():
    payload = StalenessInput(actions=[
        StalenessAction(
            id="P3-1", owner="alice", status="open", severity="low",
            opened_at=_ago(1), last_updated_at=_ago(1),
        ),
        StalenessAction(
            id="P0-1", owner="alice", status="open", severity="high",
            opened_at=_ago(10), last_updated_at=_ago(2), due_at=_ago(2),
        ),
        StalenessAction(
            id="P1-1", owner="alice", status="open", severity="medium",
            opened_at=_ago(5), last_updated_at=_ago(1), due_at=_ahead(2),
        ),
    ])
    report = _advisor().audit(payload)
    priorities = [f.priority for f in report.findings]
    # P0 must come before P1 must come before P3.
    assert priorities.index("P0") < priorities.index("P1") < priorities.index("P3")


@pytest.mark.parametrize("appetite", APPETITES)
def test_audit_runs_for_all_supported_appetites(appetite):
    payload = StalenessInput(
        actions=[
            StalenessAction(
                id="A", owner="alice", status="open", severity="medium",
                opened_at=_ago(5), last_updated_at=_ago(4),
            ),
        ],
        risk_appetite=appetite,
    )
    report = _advisor().audit(payload)
    assert report.risk_appetite == appetite
    assert report.portfolio.total_actions == 1
