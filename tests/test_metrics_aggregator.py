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

    # ── Regression tests for issue #89 ──────────────────────────────
    #
    # The previous implementation wrapped every probe body in a bare
    # `except Exception: return ModuleMetric(..., "skip", ...)`, so a
    # real bug like `AttributeError` on a `None` return showed up as
    # a benign "skip" row and the CLI exited 0. The fix moves error
    # classification into `_run_probe`: ImportError → skip, everything
    # else → error.

    def test_real_bug_surfaces_as_error(self):
        """A registered probe that raises a real exception must be reported as error."""
        @probe("_test_real_bug")
        def _failing():
            raise TypeError("'NoneType' object has no attribute 'get'")

        try:
            results = aggregate(["_test_real_bug"])
            assert len(results) == 1
            assert results[0].status == "error", (
                f"real bugs must surface as 'error', got {results[0].status!r}"
            )
            # Detail must include both the exception type and the message
            # so operators can actually debug from the dashboard.
            assert "TypeError" in results[0].detail
            assert "NoneType" in results[0].detail
        finally:
            del _PROBE_REGISTRY["_test_real_bug"]

    def test_import_error_remains_skip(self):
        """ImportError still maps to 'skip' — optional deps shouldn't fail the run."""
        @probe("_test_missing_dep")
        def _missing():
            raise ImportError("No module named 'optional_thing'")

        try:
            results = aggregate(["_test_missing_dep"])
            assert results[0].status == "skip"
            assert "ImportError" in results[0].detail
        finally:
            del _PROBE_REGISTRY["_test_missing_dep"]

    def test_module_not_found_error_remains_skip(self):
        """ModuleNotFoundError (subclass of ImportError) is also a skip."""
        @probe("_test_mnfe")
        def _mnfe():
            raise ModuleNotFoundError("No module named 'optional_thing'")

        try:
            results = aggregate(["_test_mnfe"])
            assert results[0].status == "skip"
        finally:
            del _PROBE_REGISTRY["_test_mnfe"]

    def test_other_exceptions_classified_as_error(self):
        """AttributeError, KeyError, ZeroDivisionError → error (not skip)."""
        cases = [
            ("_test_attr", AttributeError("oops")),
            ("_test_key", KeyError("missing")),
            ("_test_div", ZeroDivisionError("x/0")),
            ("_test_val", ValueError("bad input")),
        ]
        for name, exc in cases:
            @probe(name)
            def _fail(_e=exc):
                raise _e

            try:
                results = aggregate([name])
                assert results[0].status == "error", (
                    f"{type(exc).__name__} should classify as 'error'"
                )
                assert type(exc).__name__ in results[0].detail
            finally:
                del _PROBE_REGISTRY[name]


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
        rc = main(["--modules", "scorecard"])
        out = capsys.readouterr().out
        assert "scorecard" in out
        # The scorecard probe may currently surface an error in some
        # environments (its underlying module API drifted). After the
        # #89 fix that's exactly what we want — the row must be
        # rendered either way. Exit code is asserted in the dedicated
        # error/skip CLI tests below.
        assert rc in (0, 1)

    def test_main_json_output(self, capsys):
        rc = main(["--json", "--modules", "scorecard"])
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert isinstance(parsed, list)
        assert parsed[0]["module"] == "scorecard"
        assert rc in (0, 1)

    def test_main_all_modules(self, capsys):
        rc = main([])
        out = capsys.readouterr().out
        # Should render without crashing
        assert "Module" in out or "module" in out.lower()
        assert rc in (0, 1)  # depends on test env state

    def test_main_nonzero_exit_when_probe_errors(self, capsys):
        """Regression for #89: real bugs must produce a non-zero exit code
        so cron/monitoring loops can alert on them.
        """
        @probe("_test_cli_error")
        def _broken():
            raise RuntimeError("simulated regression")

        try:
            rc = main(["--modules", "_test_cli_error"])
            assert rc == 1, "errors must produce non-zero CLI exit"
            out = capsys.readouterr().out
            assert "_test_cli_error" in out
            assert "RuntimeError" in out
        finally:
            del _PROBE_REGISTRY["_test_cli_error"]

    def test_main_zero_exit_on_only_skips(self, capsys):
        """Skips (optional deps) must NOT fail the run — would create
        false alarms on environments where optional probes aren't
        installed.
        """
        @probe("_test_cli_skip")
        def _skip():
            raise ImportError("optional dep missing")

        try:
            rc = main(["--modules", "_test_cli_skip"])
            assert rc == 0, "skips must not fail the CLI"
        finally:
            del _PROBE_REGISTRY["_test_cli_skip"]

    def test_main_json_includes_error_detail(self, capsys):
        """The JSON output must include the full ExceptionType: message
        detail (the old impl truncated it to ~36 chars in the table
        and the JSON view never even tried to show it as an error).
        """
        @probe("_test_cli_detail")
        def _broken():
            raise AttributeError("'NoneType' object has no attribute 'get'")

        try:
            rc = main(["--json", "--modules", "_test_cli_detail"])
            out = capsys.readouterr().out
            parsed = json.loads(out)
            assert parsed[0]["status"] == "error"
            assert "AttributeError" in parsed[0]["detail"]
            assert "NoneType" in parsed[0]["detail"]
            assert rc == 1
        finally:
            del _PROBE_REGISTRY["_test_cli_detail"]
