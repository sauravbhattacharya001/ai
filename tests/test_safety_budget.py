"""Tests for replication.safety_budget module."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from replication.safety_budget import (
    AlertSeverity,
    ALL_CATEGORIES,
    BudgetConfig,
    BudgetPreset,
    CategoryBudget,
    RiskCategory,
    SafetyBudgetManager,
    _simulate_fleet_risk,
    cli,
)


# ── BudgetConfig ───────────────────────────────────────────────────────


class TestBudgetConfig:
    def test_default_preset(self):
        cfg = BudgetConfig()
        assert cfg.preset == BudgetPreset.MODERATE
        limits = cfg.get_limits()
        assert limits["replication"] == 30.0
        assert len(limits) == 6

    def test_conservative_preset(self):
        cfg = BudgetConfig(preset=BudgetPreset.CONSERVATIVE)
        limits = cfg.get_limits()
        assert limits["replication"] == 15.0
        assert limits["exfiltration"] == 5.0

    def test_permissive_preset(self):
        cfg = BudgetConfig(preset=BudgetPreset.PERMISSIVE)
        limits = cfg.get_limits()
        assert limits["replication"] == 60.0

    def test_custom_limits_override(self):
        cfg = BudgetConfig(
            preset=BudgetPreset.CUSTOM,
            custom_limits={"replication": 99.0},
        )
        limits = cfg.get_limits()
        assert limits["replication"] == 99.0
        # Other categories fall back to moderate
        assert limits["deception"] == 25.0


# ── SafetyBudgetManager ───────────────────────────────────────────────


class TestSafetyBudgetManager:
    def test_record_event_basic(self):
        mgr = SafetyBudgetManager()
        alerts = mgr.record_event("replication", 5.0, source="agent-1")
        assert isinstance(alerts, list)
        consumed, limit, pct = mgr.get_usage("replication")
        assert consumed == 5.0
        assert limit == 30.0

    def test_record_multiple_events(self):
        mgr = SafetyBudgetManager()
        mgr.record_event("replication", 5.0)
        mgr.record_event("replication", 10.0)
        consumed, _, _ = mgr.get_usage("replication")
        assert consumed == 15.0

    def test_invalid_category(self):
        mgr = SafetyBudgetManager()
        with pytest.raises(ValueError, match="Unknown category"):
            mgr.record_event("invalid_cat", 5.0)

    def test_negative_amount(self):
        mgr = SafetyBudgetManager()
        with pytest.raises(ValueError, match="non-negative"):
            mgr.record_event("replication", -1.0)

    def test_warning_alert(self):
        mgr = SafetyBudgetManager(BudgetConfig(preset=BudgetPreset.CONSERVATIVE))
        # replication limit = 15, warning at 70% = 10.5
        alerts = mgr.record_event("replication", 11.0)
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.WARNING

    def test_critical_alert(self):
        mgr = SafetyBudgetManager(BudgetConfig(preset=BudgetPreset.CONSERVATIVE))
        # replication limit = 15, critical at 90% = 13.5
        alerts = mgr.record_event("replication", 14.0)
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.CRITICAL

    def test_breach_alert(self):
        mgr = SafetyBudgetManager(BudgetConfig(preset=BudgetPreset.CONSERVATIVE))
        alerts = mgr.record_event("replication", 16.0)
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.BREACH

    def test_no_duplicate_alerts(self):
        mgr = SafetyBudgetManager(BudgetConfig(preset=BudgetPreset.CONSERVATIVE))
        # First event crosses warning
        a1 = mgr.record_event("replication", 11.0)
        # Second event stays in warning zone - no new alert
        a2 = mgr.record_event("replication", 1.0)
        assert len(a1) == 1
        assert len(a2) == 0

    def test_is_over_budget(self):
        mgr = SafetyBudgetManager(BudgetConfig(preset=BudgetPreset.CONSERVATIVE))
        assert not mgr.is_over_budget("replication")
        mgr.record_event("replication", 16.0)
        assert mgr.is_over_budget("replication")
        assert mgr.is_over_budget()  # any category

    def test_remaining(self):
        mgr = SafetyBudgetManager(BudgetConfig(preset=BudgetPreset.CONSERVATIVE))
        assert mgr.remaining("replication") == 15.0
        mgr.record_event("replication", 5.0)
        assert mgr.remaining("replication") == 10.0
        mgr.record_event("replication", 20.0)
        assert mgr.remaining("replication") == 0.0

    def test_reset_single_category(self):
        mgr = SafetyBudgetManager()
        mgr.record_event("replication", 10.0)
        mgr.record_event("deception", 5.0)
        mgr.reset("replication")
        consumed, _, _ = mgr.get_usage("replication")
        assert consumed == 0.0
        consumed2, _, _ = mgr.get_usage("deception")
        assert consumed2 == 5.0

    def test_reset_all(self):
        mgr = SafetyBudgetManager()
        mgr.record_event("replication", 10.0)
        mgr.record_event("deception", 5.0)
        mgr.reset()
        for cat in ALL_CATEGORIES:
            consumed, _, _ = mgr.get_usage(cat)
            assert consumed == 0.0

    def test_adjust_limit(self):
        mgr = SafetyBudgetManager()
        mgr.adjust_limit("replication", 50.0)
        _, limit, _ = mgr.get_usage("replication")
        assert limit == 50.0

    def test_adjust_limit_invalid_category(self):
        mgr = SafetyBudgetManager()
        with pytest.raises(ValueError):
            mgr.adjust_limit("nonexistent", 50.0)

    def test_adjust_limit_negative(self):
        mgr = SafetyBudgetManager()
        with pytest.raises(ValueError):
            mgr.adjust_limit("replication", -5.0)


# ── Report ─────────────────────────────────────────────────────────────


class TestBudgetReport:
    def test_report_structure(self):
        mgr = SafetyBudgetManager()
        mgr.record_event("replication", 5.0, source="agent-1")
        report = mgr.report()
        assert report.total_events == 1
        assert len(report.categories) == 6
        assert report.total_budget > 0

    def test_report_render(self):
        mgr = SafetyBudgetManager()
        mgr.record_event("replication", 5.0)
        report = mgr.report()
        text = report.render()
        assert "Safety Budget Report" in text
        assert "replication" in text
        assert "%" in text

    def test_report_to_json(self):
        mgr = SafetyBudgetManager()
        mgr.record_event("deception", 3.0, source="agent-2")
        report = mgr.report()
        j = json.loads(report.to_json())
        assert j["total_events"] == 1
        assert j["preset"] == "moderate"

    def test_report_to_json_file(self):
        mgr = SafetyBudgetManager()
        mgr.record_event("replication", 5.0)
        report = mgr.report()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            report.to_json(path)
            with open(path) as f:
                data = json.load(f)
            assert data["total_events"] == 1
        finally:
            os.unlink(path)

    def test_report_top_sources(self):
        mgr = SafetyBudgetManager()
        mgr.record_event("replication", 5.0, source="agent-1")
        mgr.record_event("replication", 10.0, source="agent-2")
        mgr.record_event("replication", 3.0, source="agent-1")
        report = mgr.report()
        rep_cat = [c for c in report.categories if c.category == "replication"][0]
        assert rep_cat.top_sources[0][0] == "agent-2"
        assert rep_cat.top_sources[0][1] == 10.0

    def test_report_category_statuses(self):
        mgr = SafetyBudgetManager(BudgetConfig(preset=BudgetPreset.CONSERVATIVE))
        mgr.record_event("replication", 16.0)  # breach
        mgr.record_event("deception", 1.0)     # ok
        report = mgr.report()
        statuses = {c.category: c.status for c in report.categories}
        assert statuses["replication"] == "breach"
        assert statuses["deception"] == "ok"


# ── Export/Import ──────────────────────────────────────────────────────


class TestExportImport:
    def test_export_events(self):
        mgr = SafetyBudgetManager()
        mgr.record_event("replication", 5.0, source="a1", description="test")
        events = mgr.export_events()
        assert len(events) == 1
        assert events[0]["category"] == "replication"
        assert events[0]["amount"] == 5.0

    def test_import_events(self):
        mgr = SafetyBudgetManager()
        events = [
            {"category": "replication", "amount": 5.0, "source": "a1"},
            {"category": "deception", "amount": 3.0},
        ]
        count = mgr.import_events(events)
        assert count == 2
        consumed, _, _ = mgr.get_usage("replication")
        assert consumed == 5.0

    def test_roundtrip(self):
        mgr1 = SafetyBudgetManager()
        mgr1.record_event("replication", 5.0, source="a1")
        mgr1.record_event("deception", 3.0, source="a2")
        events = mgr1.export_events()

        mgr2 = SafetyBudgetManager()
        mgr2.import_events(events)
        for cat in ALL_CATEGORIES:
            c1, _, _ = mgr1.get_usage(cat)
            c2, _, _ = mgr2.get_usage(cat)
            assert c1 == c2


# ── Simulation ─────────────────────────────────────────────────────────


class TestSimulation:
    def test_deterministic_with_seed(self):
        mgr1 = _simulate_fleet_risk(5, BudgetConfig(), seed=42)
        mgr2 = _simulate_fleet_risk(5, BudgetConfig(), seed=42)
        r1 = mgr1.report()
        r2 = mgr2.report()
        assert r1.total_consumed == r2.total_consumed
        assert r1.total_events == r2.total_events

    def test_generates_events(self):
        mgr = _simulate_fleet_risk(10, BudgetConfig(), seed=99)
        report = mgr.report()
        assert report.total_events > 0
        assert report.total_consumed > 0


# ── CLI ────────────────────────────────────────────────────────────────


class TestCLI:
    def test_cli_default(self, capsys):
        cli(["--seed", "42"])
        out = capsys.readouterr().out
        assert "Safety Budget Report" in out

    def test_cli_json(self, capsys):
        cli(["--json", "--seed", "42"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "categories" in data

    def test_cli_conservative(self, capsys):
        cli(["--preset", "conservative", "--seed", "42"])
        out = capsys.readouterr().out
        assert "conservative" in out.lower() or "Safety Budget" in out


# ── RiskCategory enum ──────────────────────────────────────────────────


class TestRiskCategory:
    def test_all_categories_match_enum(self):
        assert set(ALL_CATEGORIES) == {c.value for c in RiskCategory}

    def test_category_count(self):
        assert len(ALL_CATEGORIES) == 6
