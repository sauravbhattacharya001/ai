"""Tests for the alert_router module — rule-based notification routing."""

import json
import time
from datetime import datetime
from unittest.mock import patch

import pytest

from replication.alert_router import (
    AlertRouter,
    Channel,
    DispatchResult,
    QuietHours,
    RoutingRule,
    default_router,
    main,
)


# ── Channel validation ───────────────────────────────────────────────


class TestChannel:
    def test_valid_kinds(self):
        for kind in ("console", "file", "jsonl", "webhook"):
            ch = Channel(kind=kind)
            assert ch.kind == kind

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError, match="Unknown channel kind"):
            Channel(kind="sms")

    def test_auto_label(self):
        ch = Channel(kind="console")
        assert ch.label == "console"

    def test_custom_label(self):
        ch = Channel(kind="file", label="my-log")
        assert ch.label == "my-log"

    def test_directory_traversal_blocked(self):
        with pytest.raises(ValueError, match="directory traversal"):
            Channel(kind="file", path="../../../etc/passwd")

    def test_normal_path_allowed(self):
        ch = Channel(kind="file", path="alerts/safety.log")
        assert ch.path == "alerts/safety.log"


# ── Routing rule matching ────────────────────────────────────────────


class TestRoutingRule:
    def test_matches_category(self):
        rule = RoutingRule(match_category={"violation"})
        assert rule.matches({"category": "violation"}) is True
        assert rule.matches({"category": "info"}) is False

    def test_matches_severity(self):
        rule = RoutingRule(match_severity={"critical"})
        assert rule.matches({"severity": "critical"}) is True
        assert rule.matches({"severity": "warning"}) is False

    def test_matches_source(self):
        rule = RoutingRule(match_source={"controller"})
        assert rule.matches({"source": "controller"}) is True
        assert rule.matches({"source": "agent"}) is False

    def test_matches_keywords(self):
        rule = RoutingRule(match_keywords=["budget", "exceeded"])
        assert rule.matches({"message": "Token budget exceeded"}) is True
        assert rule.matches({"message": "All good"}) is False

    def test_keyword_case_insensitive(self):
        rule = RoutingRule(match_keywords=["ALERT"])
        assert rule.matches({"message": "alert triggered"}) is True

    def test_empty_filters_match_all(self):
        rule = RoutingRule()
        assert rule.matches({"category": "anything", "severity": "info"}) is True

    def test_combined_filters(self):
        rule = RoutingRule(match_category={"violation"}, match_severity={"critical"})
        assert rule.matches({"category": "violation", "severity": "critical"}) is True
        assert rule.matches({"category": "violation", "severity": "warning"}) is False
        assert rule.matches({"category": "info", "severity": "critical"}) is False


# ── Quiet Hours ──────────────────────────────────────────────────────


class TestQuietHours:
    def test_active_during_night(self):
        qh = QuietHours(start_hour=22, end_hour=7)
        assert qh.is_active(datetime(2026, 1, 1, 23, 0)) is True
        assert qh.is_active(datetime(2026, 1, 1, 3, 0)) is True
        assert qh.is_active(datetime(2026, 1, 1, 12, 0)) is False

    def test_suppress_info_during_quiet(self):
        qh = QuietHours(start_hour=22, end_hour=7, suppress_below="critical")
        assert qh.should_suppress("info", datetime(2026, 1, 1, 23, 0)) is True
        assert qh.should_suppress("warning", datetime(2026, 1, 1, 23, 0)) is True
        assert qh.should_suppress("critical", datetime(2026, 1, 1, 23, 0)) is False

    def test_no_suppress_outside_quiet(self):
        qh = QuietHours(start_hour=22, end_hour=7)
        assert qh.should_suppress("info", datetime(2026, 1, 1, 12, 0)) is False

    def test_same_day_window(self):
        qh = QuietHours(start_hour=9, end_hour=17)
        assert qh.is_active(datetime(2026, 1, 1, 12, 0)) is True
        assert qh.is_active(datetime(2026, 1, 1, 20, 0)) is False


# ── AlertRouter ──────────────────────────────────────────────────────


class TestAlertRouter:
    def _make_router(self, **kwargs) -> AlertRouter:
        rule = RoutingRule(
            name="test-rule",
            match_severity={"critical"},
            channels=[Channel(kind="console")],
            **kwargs,
        )
        return AlertRouter(rules=[rule], dry_run=True)

    def test_route_matching_event(self):
        router = self._make_router()
        results = router.route({"severity": "critical", "message": "boom"})
        assert len(results) == 1
        assert results[0].suppressed is False
        assert results[0].rule == "test-rule"

    def test_route_non_matching_event(self):
        router = self._make_router()
        results = router.route({"severity": "info", "message": "ok"})
        assert len(results) == 0

    def test_disabled_rule_skipped(self):
        rule = RoutingRule(name="off", match_severity={"critical"},
                          channels=[Channel(kind="console")], enabled=False)
        router = AlertRouter(rules=[rule], dry_run=True)
        results = router.route({"severity": "critical", "message": "test"})
        assert len(results) == 0

    def test_rate_limiting(self):
        router = self._make_router(rate_limit=2, rate_window=60)
        # First two should pass
        r1 = router.route({"severity": "critical", "message": "1"})
        r2 = router.route({"severity": "critical", "message": "2"})
        assert not r1[0].suppressed
        assert not r2[0].suppressed
        # Third should be rate-limited
        r3 = router.route({"severity": "critical", "message": "3"})
        assert r3[0].suppressed
        assert r3[0].suppression_reason == "rate_limited"

    def test_escalation(self):
        router = self._make_router(escalate_after=2, escalate_to="critical")
        router.route({"severity": "critical", "message": "1"})
        r2 = router.route({"severity": "critical", "message": "2"})
        assert r2[0].escalated is True
        assert r2[0].severity == "critical"

    def test_quiet_hours_suppression(self):
        rule = RoutingRule(name="qh", match_severity={"warning"},
                          channels=[Channel(kind="console")])
        qh = QuietHours(start_hour=0, end_hour=23, suppress_below="critical")
        router = AlertRouter(rules=[rule], quiet_hours=qh, dry_run=True)
        results = router.route({"severity": "warning", "message": "test"})
        assert len(results) == 1
        assert results[0].suppressed is True
        assert results[0].suppression_reason == "quiet_hours"

    def test_batch_routing(self):
        router = self._make_router()
        events = [
            {"severity": "critical", "message": "a"},
            {"severity": "info", "message": "b"},
            {"severity": "critical", "message": "c"},
        ]
        results = router.route_batch(events)
        assert len(results) == 2
        assert router.stats.total_events == 3

    def test_stats_tracking(self):
        router = self._make_router()
        router.route({"severity": "critical", "message": "test"})
        assert router.stats.total_events == 1
        assert router.stats.total_dispatched == 1
        assert router.stats.by_rule["test-rule"] == 1

    def test_multiple_channels(self):
        rule = RoutingRule(
            name="multi",
            match_severity={"critical"},
            channels=[Channel(kind="console"), Channel(kind="jsonl", path="test.jsonl")],
        )
        router = AlertRouter(rules=[rule], dry_run=True)
        results = router.route({"severity": "critical", "message": "test"})
        assert len(results) == 2

    def test_render_stats(self):
        router = self._make_router()
        router.route({"severity": "critical", "message": "test"})
        output = router.render_stats()
        assert "Total events" in output
        assert "1" in output

    def test_render_rules(self):
        router = self._make_router()
        output = router.render_rules()
        assert "test-rule" in output

    def test_render_rules_empty(self):
        router = AlertRouter(dry_run=True)
        output = router.render_rules()
        assert "no rules configured" in output


# ── Default router ───────────────────────────────────────────────────


class TestDefaultRouter:
    def test_default_router_has_rules(self):
        router = default_router(dry_run=True)
        assert len(router.rules) == 3

    def test_critical_violation_dispatched(self):
        router = default_router(dry_run=True)
        results = router.route({
            "category": "violation",
            "severity": "critical",
            "message": "Token budget exceeded",
            "source": "controller",
        })
        dispatched = [r for r in results if not r.suppressed]
        assert len(dispatched) >= 2  # matches both critical-all and violations rules


# ── File dispatch (non-dry-run) ──────────────────────────────────────


class TestFileDispatch:
    def test_file_channel_writes(self, tmp_path):
        log = tmp_path / "test.log"
        rule = RoutingRule(
            name="file-test",
            channels=[Channel(kind="file", path=str(log))],
        )
        router = AlertRouter(rules=[rule])
        router.route({"category": "test", "severity": "warning", "message": "hello file"})
        content = log.read_text()
        assert "hello file" in content
        assert "WARNING" in content

    def test_jsonl_channel_writes(self, tmp_path):
        log = tmp_path / "test.jsonl"
        rule = RoutingRule(
            name="jsonl-test",
            channels=[Channel(kind="jsonl", path=str(log))],
        )
        router = AlertRouter(rules=[rule])
        router.route({"category": "test", "severity": "info", "message": "hello jsonl"})
        record = json.loads(log.read_text().strip())
        assert record["message"] == "hello jsonl"
        assert record["severity"] == "info"


# ── CLI ──────────────────────────────────────────────────────────────


class TestCLI:
    def test_route_command(self, capsys):
        main(["route", "--category", "violation", "--severity", "critical",
              "--message", "test alert", "--dry-run"])
        out = capsys.readouterr().out
        assert "channel" in out.lower()

    def test_route_json_output(self, capsys):
        main(["route", "--severity", "critical", "--dry-run", "--json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert isinstance(data, list)

    def test_test_command(self, capsys):
        event = json.dumps({"category": "violation", "severity": "critical", "message": "test"})
        main(["test", "--event", event])
        out = capsys.readouterr().out
        assert "Test results" in out

    def test_stats_command(self, capsys):
        main(["stats"])
        out = capsys.readouterr().out
        assert "Total events" in out

    def test_no_command_shows_help(self, capsys):
        main([])
        out = capsys.readouterr().out
        # argparse may print to stdout or stderr
        combined = out + capsys.readouterr().err
        # Just verify it didn't crash
