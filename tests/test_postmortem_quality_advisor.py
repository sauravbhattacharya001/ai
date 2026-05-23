"""Tests for replication.postmortem_quality_advisor."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from replication.postmortem_quality_advisor import (
    ALL_VERDICTS,
    BASE_THRESHOLDS,
    ActionItem,
    IncidentRecord,
    PostmortemQualityAdvisor,
    PostmortemRecord,
    QualityInput,
    VERDICT_ARCHIVED,
    VERDICT_ACTION_ITEMS_OVERDUE,
    VERDICT_AWAITING_RCA,
    VERDICT_BLAMEFUL_LANGUAGE,
    VERDICT_DRAFT_STALE,
    VERDICT_INSUFFICIENT_DATA,
    VERDICT_MISSING,
    VERDICT_NO_LESSONS_LEARNED,
    VERDICT_OWNERLESS,
    VERDICT_PUBLISHED_OK,
    VERDICT_THIN_ACTION_ITEMS,
    main,
)


NOW = datetime(2026, 5, 22, tzinfo=timezone.utc)


def _advisor():
    return PostmortemQualityAdvisor(now=lambda: NOW)


def _pm(**kw) -> PostmortemRecord:
    defaults = dict(
        id="pm-x",
        incident_id="inc-x",
        title="x",
        severity="medium",
        status="published",
        owner="team-x",
        occurred_at=NOW - timedelta(days=5),
        published_at=NOW - timedelta(days=3),
        root_cause="The cron worker did not propagate the kill signal to its child processes.",
        lessons_learned=(
            "Always propagate signals to child processes. Add a regression test "
            "for graceful shutdown. Document the cron worker lifecycle for future "
            "maintainers."
        ),
        action_items=[
            ActionItem(id="a1", title="Fix signal", owner="o", status="done"),
            ActionItem(id="a2", title="Add test", owner="o", status="done"),
        ],
    )
    defaults.update(kw)
    return PostmortemRecord(**defaults)


def test_published_ok_baseline():
    adv = _advisor()
    rep = adv.audit(QualityInput(postmortems=[_pm()]))
    assert len(rep.findings) == 1
    f = rep.findings[0]
    assert f.verdict == VERDICT_PUBLISHED_OK
    assert f.priority == "P3"
    assert f.quality_score >= 90
    assert rep.portfolio.grade == "A"


def test_missing_for_uncovered_incident():
    adv = _advisor()
    rep = adv.audit(QualityInput(
        postmortems=[],
        incidents=[IncidentRecord(id="inc-1", severity="critical",
                                  occurred_at=NOW - timedelta(days=3))],
    ))
    assert len(rep.findings) == 1
    f = rep.findings[0]
    assert f.verdict == VERDICT_MISSING
    assert f.priority == "P0"
    assert "MISSING_POSTMORTEMS" in " ".join(rep.insights)


def test_missing_medium_severity_is_p1():
    adv = _advisor()
    rep = adv.audit(QualityInput(
        incidents=[IncidentRecord(id="inc-2", severity="medium")],
    ))
    assert rep.findings[0].priority == "P1"


def test_incident_with_matching_pm_not_missing():
    adv = _advisor()
    rep = adv.audit(QualityInput(
        postmortems=[_pm(id="pm-1", incident_id="inc-1")],
        incidents=[IncidentRecord(id="inc-1", severity="critical")],
    ))
    # One finding (the PM); no MISSING synthesized.
    assert len(rep.findings) == 1
    assert rep.findings[0].verdict == VERDICT_PUBLISHED_OK


def test_archived_excluded_from_health():
    adv = _advisor()
    rep = adv.audit(QualityInput(postmortems=[_pm(status="archived")]))
    f = rep.findings[0]
    assert f.verdict == VERDICT_ARCHIVED
    assert f.priority == "P3"
    assert rep.portfolio.archived == 1
    assert rep.portfolio.active == 0


def test_draft_stale_triggers_after_publish_window():
    adv = _advisor()
    rep = adv.audit(QualityInput(postmortems=[
        _pm(status="draft", occurred_at=NOW - timedelta(days=30),
            published_at=None, owner="team-x"),
    ]))
    assert rep.findings[0].verdict == VERDICT_DRAFT_STALE


def test_awaiting_rca_when_published_without_root_cause():
    adv = _advisor()
    rep = adv.audit(QualityInput(postmortems=[
        _pm(root_cause="", published_at=NOW - timedelta(days=30)),
    ]))
    # Without RCA + with thin lessons it may collapse, but RCA wins by rank.
    assert rep.findings[0].verdict == VERDICT_AWAITING_RCA


def test_blameful_language_detected():
    adv = _advisor()
    rep = adv.audit(QualityInput(postmortems=[
        _pm(body="This was their fault and the engineer was negligent."),
    ]))
    assert rep.findings[0].verdict == VERDICT_BLAMEFUL_LANGUAGE
    assert rep.findings[0].priority == "P0"


def test_blameful_does_not_false_positive_on_innocuous_text():
    adv = _advisor()
    rep = adv.audit(QualityInput(postmortems=[
        _pm(body="The system experienced a fault in the queue subsystem."),
    ]))
    # "fault" alone (not "fault of"/"his fault"/etc) should not trip.
    assert rep.findings[0].verdict == VERDICT_PUBLISHED_OK


def test_thin_action_items_for_critical():
    adv = _advisor()
    rep = adv.audit(QualityInput(postmortems=[
        _pm(severity="critical", action_items=[
            ActionItem(id="a", status="done"),
        ]),
    ]))
    # Critical wants >=4 actions; only 1 -> thin.
    assert rep.findings[0].verdict == VERDICT_THIN_ACTION_ITEMS


def test_overdue_action_items():
    adv = _advisor()
    rep = adv.audit(QualityInput(postmortems=[
        _pm(action_items=[
            ActionItem(id="a", status="open",
                       due_date=NOW - timedelta(days=10)),
            ActionItem(id="b", status="open",
                       due_date=NOW - timedelta(days=1)),
        ]),
    ]))
    f = rep.findings[0]
    assert f.verdict == VERDICT_ACTION_ITEMS_OVERDUE
    assert f.overdue_action_count == 2


def test_done_actions_not_overdue():
    adv = _advisor()
    rep = adv.audit(QualityInput(postmortems=[
        _pm(action_items=[
            ActionItem(id="a", status="done",
                       due_date=NOW - timedelta(days=10)),
            ActionItem(id="b", status="done",
                       due_date=NOW - timedelta(days=1)),
        ]),
    ]))
    assert rep.findings[0].overdue_action_count == 0


def test_no_lessons_learned_when_too_short():
    adv = _advisor()
    rep = adv.audit(QualityInput(postmortems=[
        _pm(lessons_learned="Be careful."),
    ]))
    assert rep.findings[0].verdict == VERDICT_NO_LESSONS_LEARNED


def test_ownerless_published():
    adv = _advisor()
    rep = adv.audit(QualityInput(postmortems=[
        _pm(owner=None),
    ]))
    # Owner takes precedence over no_lessons here by verdict rank.
    assert rep.findings[0].verdict == VERDICT_OWNERLESS


def test_insufficient_data_when_no_dates():
    adv = _advisor()
    rep = adv.audit(QualityInput(postmortems=[
        _pm(occurred_at=None, published_at=None),
    ]))
    assert rep.findings[0].verdict == VERDICT_INSUFFICIENT_DATA


def test_cautious_tightens_thresholds():
    adv = _advisor()
    base = adv.audit(QualityInput(postmortems=[_pm()], risk_appetite="balanced"))
    cautious = adv.audit(QualityInput(postmortems=[_pm()], risk_appetite="cautious"))
    assert cautious.thresholds["publish_window_days"] < base.thresholds["publish_window_days"]
    # Lessons-min-chars goes UP under cautious (inverted multiplier).
    assert cautious.thresholds["lessons_min_chars"] > base.thresholds["lessons_min_chars"]


def test_aggressive_loosens_thresholds():
    adv = _advisor()
    base = adv.audit(QualityInput(postmortems=[_pm()], risk_appetite="balanced"))
    aggressive = adv.audit(QualityInput(postmortems=[_pm()], risk_appetite="aggressive"))
    assert aggressive.thresholds["publish_window_days"] > base.thresholds["publish_window_days"]


def test_invalid_risk_appetite_falls_back():
    adv = _advisor()
    rep = adv.audit(QualityInput(postmortems=[_pm()], risk_appetite="bogus"))
    assert rep.risk_appetite == "balanced"


def test_to_text_to_markdown_to_json_all_render():
    adv = _advisor()
    rep = adv.audit(QualityInput(postmortems=[_pm()]))
    assert "Postmortem Quality" in rep.to_text()
    md = rep.to_markdown()
    assert "# Postmortem Quality Report" in md
    js = rep.to_json()
    parsed = json.loads(js)
    assert "portfolio" in parsed
    assert "findings" in parsed
    assert "playbook" in parsed


def test_to_json_byte_stable():
    adv = _advisor()
    rep = adv.audit(QualityInput(postmortems=[_pm()]))
    assert rep.to_json() == rep.to_json()


def test_critical_missing_floors_grade_to_c():
    adv = _advisor()
    rep = adv.audit(QualityInput(
        postmortems=[_pm()],  # one perfect doc
        incidents=[IncidentRecord(id="inc-crit", severity="critical")],
    ))
    # Otherwise might grade A; critical-missing forces ≤ C.
    assert rep.portfolio.grade in ("C", "D", "F")
    assert any("CRITICAL_INCIDENTS_UNDOCUMENTED" in i for i in rep.insights)


def test_blameful_floors_grade_to_c():
    adv = _advisor()
    rep = adv.audit(QualityInput(postmortems=[
        _pm(id="pm-good"),
        _pm(id="pm-bad", incident_id="inc-bad",
            body="This was the operator's fault. They should have known."),
    ]))
    assert rep.portfolio.grade in ("C", "D", "F")
    assert rep.portfolio.blameful == 1


def test_playbook_p0_first():
    adv = _advisor()
    rep = adv.audit(QualityInput(postmortems=[
        _pm(id="pm-blame", body="their fault"),
        _pm(id="pm-no-lessons", lessons_learned="short"),
    ]))
    assert rep.playbook
    # Priority is ordered alphabetically when we sort by P-letter; but our
    # build order pushes P0 categories first.
    first = rep.playbook[0]
    assert first.priority == "P0"


def test_unknown_severity_normalized_to_medium():
    adv = _advisor()
    rep = adv.audit(QualityInput(postmortems=[
        _pm(severity="bogus"),
    ]))
    assert rep.findings[0].severity == "medium"


def test_naive_datetimes_treated_as_utc():
    adv = _advisor()
    naive = datetime(2026, 5, 1)  # naive
    rep = adv.audit(QualityInput(postmortems=[
        _pm(occurred_at=naive, published_at=naive),
    ]))
    # Should not crash; counts as published.
    assert rep.findings[0].verdict in ALL_VERDICTS


def test_main_demo_text(capsys):
    rc = main(["--demo", "--format", "text"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Postmortem Quality" in out


def test_main_demo_markdown(capsys):
    rc = main(["--demo", "--format", "markdown"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "# Postmortem Quality Report" in out


def test_main_demo_json(capsys):
    rc = main(["--demo", "--format", "json"])
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert "findings" in parsed


def test_main_from_json_roundtrip(tmp_path, capsys):
    path = tmp_path / "pms.json"
    payload = {
        "postmortems": [{
            "id": "pm-r",
            "incident_id": "inc-r",
            "title": "R",
            "severity": "medium",
            "status": "published",
            "owner": "team",
            "occurred_at": (NOW - timedelta(days=5)).isoformat(),
            "published_at": (NOW - timedelta(days=3)).isoformat(),
            "root_cause": "Race condition in the leader-election retry loop.",
            "lessons_learned": (
                "Add jitter to retry loops. Use a proven library for leader "
                "election. Write a chaos test for split-brain scenarios."
            ),
            "action_items": [
                {"id": "ai-1", "status": "done"},
                {"id": "ai-2", "status": "done"},
            ],
        }],
        "incidents": [],
        "risk_appetite": "balanced",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    rc = main(["--from-json", str(path), "--format", "json"])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["findings"][0]["verdict"] == VERDICT_PUBLISHED_OK
