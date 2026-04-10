"""Tests for replication.metrics_aggregator module."""
from __future__ import annotations

import json
from dataclasses import asdict
from unittest.mock import patch

import pytest

from replication.metrics_aggregator import (
    ModuleMetric,
    aggregate,
    main,
    probe,
    _PROBE_REGISTRY,
    _render_table,
    _status_from_score,
    STATUS_ICON,
    STATUS_RANK,
    ALL_PROBES,
)


# ── _status_from_score ───────────────────────────────────────────────


class TestStatusFromScore:
    def test_high_score_ok(self):
        assert _status_from_score(80, ok_threshold=70, warn_threshold=40) == "ok"

    def test_mid_score_warn(self):
        assert _status_from_score(55, ok_threshold=70, warn_threshold=40) == "warn"

    def test_low_score_error(self):
        assert _status_from_score(30, ok_threshold=70, warn_threshold=40) == "error"

    def test_exact_ok_threshold(self):
        assert _status_from_score(70, ok_threshold=70, warn_threshold=40) == "ok"

    def test_exact_warn_threshold(self):
        assert _status_from_score(40, ok_threshold=70, warn_threshold=40) == "warn"

    def test_inverted_low_ok(self):
        assert _status_from_score(0, ok_threshold=0, warn_threshold=2, invert=True) == "ok"

    def test_inverted_mid_warn(self):
        assert _status_from_score(1, ok_threshold=0, warn_threshold=2, invert=True) == "warn"

    def test_inverted_high_error(self):
        assert _status_from_score(5, ok_threshold=0, warn_threshold=2, invert=True) == "error"


# ── ModuleMetric ─────────────────────────────────────────────────────


class TestModuleMetric:
    def test_defaults(self):
        m = ModuleMetric(module="test", status="ok")
        assert m.score is None
        assert m.detail == ""
        assert m.extra == {}

    def test_asdict(self):
        m = ModuleMetric("x", "warn", score=42.0, detail="d", extra={"k": 1})
        d = asdict(m)
        assert d["module"] == "x"
        assert d["extra"]["k"] == 1


# ── probe decorator ──────────────────────────────────────────────────


class TestProbeDecorator:
    def test_registers_function(self):
        @probe("_test_probe_xyz")
        def _dummy():
            return ModuleMetric("_test_probe_xyz", "ok")

        assert "_test_probe_xyz" in _PROBE_REGISTRY
        result = _PROBE_REGISTRY["_test_probe_xyz"]()
        assert result.status == "ok"
        # cleanup
        del _PROBE_REGISTRY["_test_probe_xyz"]

    def test_all_probes_alias(self):
        assert ALL_PROBES is _PROBE_REGISTRY


# ── aggregate ────────────────────────────────────────────────────────


class TestAggregate:
    def test_unknown_module_skipped(self):
        results = aggregate(["nonexistent_module_xyz"])
        assert len(results) == 1
        assert results[0].status == "skip"
        assert "Unknown" in results[0].detail

    def test_default_runs_all(self):
        results = aggregate()
        # Should have one result per registered probe
        assert len(results) == len(_PROBE_REGISTRY)

    def test_specific_modules(self):
        # All built-in probes handle missing deps gracefully (return skip)
        results = aggregate(["scorecard", "drift"])
        assert len(results) == 2
        names = [r.module for r in results]
        assert "scorecard" in names
        assert "drift" in names

    def test_probe_exception_safe(self):
        """If a probe raises, aggregate should still handle it via the probe's own try/except."""
        # Built-in probes catch their own exceptions, so this tests the pattern
        results = aggregate(["scorecard"])
        assert len(results) == 1
        # Status should be one of the valid values
        assert results[0].status in ("ok", "warn", "error", "skip")


# ── _render_table ────────────────────────────────────────────────────


class TestRenderTable:
    def test_empty_list(self):
        text = _render_table([])
        assert "No modules probed" in text

    def test_single_ok(self):
        metrics = [ModuleMetric("test", "ok", score=95.0, detail="All good")]
        text = _render_table(metrics)
        assert "test" in text
        assert "95" in text
        assert "HEALTHY" in text

    def test_warn_shows_degraded(self):
        metrics = [ModuleMetric("a", "warn", detail="Alert")]
        text = _render_table(metrics)
        assert "DEGRADED" in text

    def test_error_shows_critical(self):
        metrics = [ModuleMetric("a", "error", detail="Fail")]
        text = _render_table(metrics)
        assert "CRITICAL" in text

    def test_skip_only_no_modules(self):
        metrics = [ModuleMetric("a", "skip", detail="N/A")]
        text = _render_table(metrics)
        assert "No modules probed" in text

    def test_none_score_renders_dash(self):
        metrics = [ModuleMetric("a", "ok")]
        text = _render_table(metrics)
        assert "—" in text


# ── STATUS constants ─────────────────────────────────────────────────


class TestConstants:
    def test_status_icons_complete(self):
        for s in ("ok", "warn", "error", "skip"):
            assert s in STATUS_ICON

    def test_status_rank_ordering(self):
        assert STATUS_RANK["error"] < STATUS_RANK["warn"] < STATUS_RANK["ok"]


# ── CLI ──────────────────────────────────────────────────────────────


class TestCLI:
    def test_main_table_output(self, capsys):
        main(["--modules", "scorecard"])
        out = capsys.readouterr().out
        assert "scorecard" in out

    def test_main_json_output(self, capsys):
        main(["--json", "--modules", "scorecard"])
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert isinstance(parsed, list)
        assert parsed[0]["module"] == "scorecard"

    def test_main_all_modules(self, capsys):
        main([])
        out = capsys.readouterr().out
        # Should render without crashing
        assert "Module" in out or "module" in out.lower()
