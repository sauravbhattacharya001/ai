"""Tests for replication.runbook_freshness_advisor."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from replication.runbook_freshness_advisor import (
    ALL_VERDICTS,
    BASE_THRESHOLDS,
    FreshnessInput,
    RunbookFreshnessAdvisor,
    RunbookRecord,
    VERDICT_ARCHIVED,
    VERDICT_AT_RISK,
    VERDICT_DRIFT_VS_EXECUTION,
    VERDICT_DRILL_OVERDUE,
    VERDICT_FRESH,
    VERDICT_INSUFFICIENT_DATA,
    VERDICT_NEVER_EXECUTED,
    VERDICT_ORPHANED_OWNER,
    VERDICT_STALE_CONTENT,
    main,
)

NOW = datetime(2026, 5, 22, tzinfo=timezone.utc)


def _advisor():
    return RunbookFreshnessAdvisor(now=lambda: NOW)


def _record(**kwargs) -> RunbookRecord:
    defaults = dict(
        id="rb-x",
        title="x",
        severity="medium",
        owner="team-x",
        status="active",
        last_updated_at=NOW - timedelta(days=10),
        last_executed_at=NOW - timedelta(days=10),
        last_drill_at=NOW - timedelta(days=10),
        execution_count=1,
    )
    defaults.update(kwargs)
    return RunbookRecord(**defaults)


def _find(report, rid):
    for f in report.findings:
        if f.runbook_id == rid:
            return f
    raise AssertionError(f"missing finding for {rid}")


def test_fresh_runbook_passes():
    report = _advisor().audit(FreshnessInput(runbooks=[_record(id="rb-1")]))
    f = _find(report, "rb-1")
    assert f.verdict == VERDICT_FRESH
    assert f.priority == "P3"
    assert f.freshness_score > 80
    assert report.portfolio.fresh == 1
    assert report.portfolio.grade == "A"


def test_orphaned_owner_is_p0():
    rb = _record(id="rb-orph", owner="")
    f = _find(_advisor().audit(FreshnessInput(runbooks=[rb])), "rb-orph")
    assert f.verdict == VERDICT_ORPHANED_OWNER
    assert f.priority == "P0"


def test_never_executed():
    rb = _record(id="rb-new", last_executed_at=None, last_drill_at=None)
    f = _find(_advisor().audit(FreshnessInput(runbooks=[rb])), "rb-new")
    assert f.verdict == VERDICT_NEVER_EXECUTED


def test_drill_overdue():
    rb = _record(
        id="rb-od",
        last_drill_at=NOW - timedelta(days=400),
        last_executed_at=NOW - timedelta(days=400),
    )
    f = _find(_advisor().audit(FreshnessInput(runbooks=[rb])), "rb-od")
    assert f.verdict in (VERDICT_DRILL_OVERDUE, VERDICT_STALE_CONTENT)
    # We chose recently updated content, so drill cadence should dominate.
    rb2 = _record(
        id="rb-od2",
        last_updated_at=NOW - timedelta(days=5),
        last_drill_at=NOW - timedelta(days=400),
        last_executed_at=NOW - timedelta(days=400),
    )
    f2 = _find(_advisor().audit(FreshnessInput(runbooks=[rb2])), "rb-od2")
    assert f2.verdict == VERDICT_DRILL_OVERDUE


def test_stale_content_high_severity_is_p0():
    rb = _record(
        id="rb-stale",
        severity="high",
        last_updated_at=NOW - timedelta(days=400),
        last_drill_at=NOW - timedelta(days=10),
        last_executed_at=NOW - timedelta(days=10),
    )
    f = _find(_advisor().audit(FreshnessInput(runbooks=[rb])), "rb-stale")
    assert f.verdict == VERDICT_STALE_CONTENT
    assert f.priority == "P0"


def test_drift_vs_execution():
    rb = _record(
        id="rb-drift",
        last_updated_at=NOW - timedelta(days=120),
        last_executed_at=NOW - timedelta(days=20),
        last_drill_at=NOW - timedelta(days=20),
        execution_count=4,
    )
    f = _find(_advisor().audit(FreshnessInput(runbooks=[rb])), "rb-drift")
    # The gap is 100d which exceeds drift_days(30); content age (120d)
    # is still below stale_days(180). Worst-of ranking picks
    # DRIFT_VS_EXECUTION over AT_RISK.
    assert f.verdict == VERDICT_DRIFT_VS_EXECUTION


def test_at_risk_window():
    # 170d old; stale_days=180, at_risk band = 30 days before -> in band.
    rb = _record(
        id="rb-near",
        last_updated_at=NOW - timedelta(days=170),
        last_drill_at=NOW - timedelta(days=10),
        last_executed_at=NOW - timedelta(days=10),
        execution_count=0,
    )
    f = _find(_advisor().audit(FreshnessInput(runbooks=[rb])), "rb-near")
    assert f.verdict == VERDICT_AT_RISK
    assert f.priority == "P2"


def test_archived_excluded():
    rb = _record(id="rb-arch", status="archived")
    report = _advisor().audit(FreshnessInput(runbooks=[rb]))
    f = _find(report, "rb-arch")
    assert f.verdict == VERDICT_ARCHIVED
    assert report.portfolio.archived == 1
    assert report.portfolio.active == 0


def test_insufficient_data_when_no_update_ts():
    rb = _record(id="rb-missing", last_updated_at=None)
    f = _find(_advisor().audit(FreshnessInput(runbooks=[rb])), "rb-missing")
    # No update timestamp + recent drill -> INSUFFICIENT_DATA is the only signal.
    assert f.verdict == VERDICT_INSUFFICIENT_DATA


def test_appetite_scaling_changes_thresholds():
    cautious = _advisor()._scaled_thresholds("cautious")
    balanced = _advisor()._scaled_thresholds("balanced")
    aggressive = _advisor()._scaled_thresholds("aggressive")
    assert cautious["stale_days"] < balanced["stale_days"] < aggressive["stale_days"]
    # drift_min_executions is a count, must not be day-scaled.
    assert cautious["drift_min_executions"] == BASE_THRESHOLDS["drift_min_executions"]


def test_appetite_flips_verdict():
    rb = _record(
        id="rb-edge",
        last_updated_at=NOW - timedelta(days=150),
        last_drill_at=NOW - timedelta(days=5),
        last_executed_at=NOW - timedelta(days=5),
        execution_count=0,
    )
    cautious = _find(
        _advisor().audit(FreshnessInput(runbooks=[rb], risk_appetite="cautious")),
        "rb-edge",
    )
    aggressive = _find(
        _advisor().audit(FreshnessInput(runbooks=[rb], risk_appetite="aggressive")),
        "rb-edge",
    )
    # cautious tightens stale_days (×0.7 -> 126) so 150d -> STALE.
    # aggressive loosens (×1.4 -> 252) so 150d is firmly FRESH.
    assert cautious.verdict == VERDICT_STALE_CONTENT
    assert aggressive.verdict == VERDICT_FRESH


def test_portfolio_grade_and_insights():
    rbs = [
        _record(id=f"stale-{i}",
                last_updated_at=NOW - timedelta(days=400),
                last_drill_at=NOW - timedelta(days=5),
                last_executed_at=NOW - timedelta(days=5))
        for i in range(3)
    ]
    rbs.append(_record(id="orphan", owner=""))
    rbs.append(_record(id="fresh"))
    report = _advisor().audit(FreshnessInput(runbooks=rbs))
    assert report.portfolio.stale == 3
    assert report.portfolio.orphaned == 1
    assert report.portfolio.fresh == 1
    assert report.portfolio.grade in {"D", "F"}
    text_insights = "\n".join(report.insights)
    assert "WIDESPREAD_STALENESS" in text_insights


def test_healthy_library_insight():
    rbs = [_record(id=f"ok-{i}") for i in range(5)]
    report = _advisor().audit(FreshnessInput(runbooks=rbs))
    assert any("HEALTHY_LIBRARY" in i for i in report.insights)


def test_empty_library_insight():
    report = _advisor().audit(FreshnessInput(runbooks=[]))
    assert report.insights == ["EMPTY_LIBRARY — supply at least one runbook to audit."]
    assert report.portfolio.total == 0
    assert report.portfolio.grade == "A"


def test_playbook_dedup_and_priority_order():
    rbs = [
        _record(id="o1", owner=""),
        _record(id="o2", owner=""),
        _record(id="s1",
                last_updated_at=NOW - timedelta(days=400),
                last_drill_at=NOW - timedelta(days=5),
                last_executed_at=NOW - timedelta(days=5)),
    ]
    report = _advisor().audit(FreshnessInput(runbooks=rbs))
    priorities = [a.priority for a in report.playbook]
    # P0 must come first.
    assert priorities[0] == "P0"
    # Orphaned actions are deduped into one entry with both ids.
    orph_actions = [a for a in report.playbook if a.label == "Assign owners"]
    assert len(orph_actions) == 1
    assert set(orph_actions[0].runbook_ids) == {"o1", "o2"}


def test_naive_datetime_normalized_to_utc():
    rb = _record(
        id="rb-naive",
        last_updated_at=datetime(2026, 5, 12),  # naive
    )
    f = _find(_advisor().audit(FreshnessInput(runbooks=[rb])), "rb-naive")
    # 10 days old (approx), within fresh -> FRESH.
    assert f.verdict == VERDICT_FRESH
    assert f.days_since_update is not None
    assert 9.0 <= f.days_since_update <= 11.0


def test_future_timestamp_clamped_to_zero():
    rb = _record(
        id="rb-future",
        last_updated_at=NOW + timedelta(days=5),
    )
    f = _find(_advisor().audit(FreshnessInput(runbooks=[rb])), "rb-future")
    assert f.days_since_update == 0.0


def test_unknown_severity_normalizes_to_medium():
    rb = _record(id="rb-sev", severity="bogus")
    f = _find(_advisor().audit(FreshnessInput(runbooks=[rb])), "rb-sev")
    assert f.severity == "medium"


def test_renderers_produce_strings():
    report = _advisor().audit(FreshnessInput(runbooks=[_record(id="rb-r")]))
    assert "Runbook Freshness" in report.to_text()
    assert "# Runbook Freshness Report" in report.to_markdown()
    parsed = json.loads(report.to_json())
    assert parsed["portfolio"]["total"] == 1
    assert {f["runbook_id"] for f in parsed["findings"]} == {"rb-r"}


def test_all_verdicts_have_rank_entries():
    # Sanity: every public verdict must be representable in the playbook code.
    for v in ALL_VERDICTS:
        # Should not raise when evaluating priority logic with arbitrary severity.
        RunbookFreshnessAdvisor._priority_for(v, "medium")


def test_cli_demo_text(capsys):
    rc = main(["--demo", "--format", "text"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Runbook Freshness" in out


def test_cli_demo_json(capsys):
    rc = main(["--demo", "--format", "json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "portfolio" in payload
    assert payload["risk_appetite"] in ("balanced", "cautious", "aggressive")


def test_cli_from_json_roundtrip(tmp_path, capsys):
    library = {
        "risk_appetite": "cautious",
        "runbooks": [
            {
                "id": "rb-json-1",
                "title": "JSON-loaded book",
                "severity": "high",
                "owner": "ops",
                "last_updated_at": "2026-05-12T00:00:00+00:00",
                "last_executed_at": "2026-05-15T00:00:00Z",
                "last_drill_at": "2026-05-15T00:00:00+00:00",
                "execution_count": 2,
            }
        ],
    }
    p = tmp_path / "lib.json"
    p.write_text(json.dumps(library), encoding="utf-8")
    rc = main(["--from-json", str(p), "--format", "json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["risk_appetite"] == "cautious"
    assert any(f["runbook_id"] == "rb-json-1" for f in payload["findings"])


def test_cli_output_file(tmp_path, capsys):
    out_path = tmp_path / "report.md"
    rc = main(["--demo", "--format", "markdown", "-o", str(out_path)])
    assert rc == 0
    assert out_path.read_text(encoding="utf-8").startswith("# Runbook Freshness Report")
    assert "wrote" in capsys.readouterr().out


def test_unrecognized_appetite_falls_back_to_balanced():
    rb = _record(id="rb-app")
    report = _advisor().audit(FreshnessInput(runbooks=[rb], risk_appetite="reckless"))
    assert report.risk_appetite == "balanced"
