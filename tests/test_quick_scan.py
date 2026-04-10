"""Tests for replication.quick_scan module."""
from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from replication.quick_scan import (
    CheckResult,
    ScanResult,
    QuickScanner,
    ALL_CHECKS,
    main,
)


# ── CheckResult ──────────────────────────────────────────────────────


class TestCheckResult:
    def test_defaults(self):
        cr = CheckResult(name="foo", status="pass")
        assert cr.score is None
        assert cr.summary == ""
        assert cr.details == {}
        assert cr.duration_s == 0.0

    def test_with_all_fields(self):
        cr = CheckResult(
            name="scorecard", status="warn", score=65.0,
            summary="Below threshold", details={"grade": "C"}, duration_s=1.5,
        )
        assert cr.score == 65.0
        assert cr.details["grade"] == "C"


# ── ScanResult ───────────────────────────────────────────────────────


class TestScanResult:
    def _make_result(self, statuses, strict=False):
        sr = ScanResult(strict=strict, timestamp="2026-01-01T00:00:00Z")
        for i, s in enumerate(statuses):
            sr.checks.append(CheckResult(name=f"check-{i}", status=s))
        return sr

    def test_passed_all_pass(self):
        sr = self._make_result(["pass", "pass"])
        assert sr.passed is True

    def test_passed_with_warn_non_strict(self):
        sr = self._make_result(["pass", "warn"])
        assert sr.passed is True

    def test_failed_with_warn_strict(self):
        sr = self._make_result(["pass", "warn"], strict=True)
        assert sr.passed is False

    def test_failed_with_fail(self):
        sr = self._make_result(["pass", "fail"])
        assert sr.passed is False

    def test_failed_with_error(self):
        sr = self._make_result(["pass", "error"])
        assert sr.passed is False

    def test_counts(self):
        sr = self._make_result(["pass", "pass", "warn", "fail", "error"])
        assert sr.pass_count == 2
        assert sr.warn_count == 1
        assert sr.fail_count == 2

    def test_empty_passes(self):
        sr = ScanResult()
        assert sr.passed is True
        assert sr.pass_count == 0

    def test_render_contains_key_info(self):
        sr = self._make_result(["pass", "warn"])
        sr.total_duration_s = 2.5
        text = sr.render()
        assert "Quick Scan" in text
        assert "PASS" in text
        assert "standard" in text

    def test_render_strict_mode(self):
        sr = self._make_result(["pass"], strict=True)
        text = sr.render()
        assert "strict" in text

    def test_render_with_summary(self):
        sr = ScanResult(timestamp="2026-01-01T00:00:00Z")
        sr.checks.append(CheckResult(name="test", status="pass", summary="All good", score=95.0, duration_s=0.5))
        text = sr.render()
        assert "All good" in text
        assert "95" in text

    def test_to_dict_structure(self):
        sr = self._make_result(["pass", "fail"])
        sr.total_duration_s = 1.23
        d = sr.to_dict()
        assert d["passed"] is False
        assert d["summary"]["pass"] == 1
        assert d["summary"]["fail"] == 1
        assert len(d["checks"]) == 2
        assert "timestamp" in d

    def test_to_dict_roundtrip_json(self):
        sr = self._make_result(["pass"])
        sr.total_duration_s = 0.5
        j = json.dumps(sr.to_dict())
        parsed = json.loads(j)
        assert parsed["passed"] is True


# ── QuickScanner ─────────────────────────────────────────────────────


class TestQuickScanner:
    def test_default_checks(self):
        scanner = QuickScanner()
        assert scanner.checks == list(ALL_CHECKS)

    def test_custom_checks(self):
        scanner = QuickScanner(checks=["preflight"])
        assert scanner.checks == ["preflight"]

    def test_unknown_check_skipped(self):
        scanner = QuickScanner(checks=["nonexistent-check"])
        result = scanner.run()
        assert len(result.checks) == 1
        assert result.checks[0].status == "skip"
        assert "Unknown" in result.checks[0].summary

    def test_strict_propagated(self):
        scanner = QuickScanner(strict=True, checks=[])
        result = scanner.run()
        assert result.strict is True

    def test_timestamp_set(self):
        scanner = QuickScanner(checks=[])
        result = scanner.run()
        assert result.timestamp != ""
        assert "T" in result.timestamp

    def test_runner_exception_caught(self):
        """If a runner raises, the check should get status='error'."""
        with patch.dict("replication.quick_scan._RUNNERS", {"boom": lambda: (_ for _ in ()).throw(RuntimeError("kaboom"))}):
            scanner = QuickScanner(checks=["boom"])
            result = scanner.run()
            assert result.checks[0].status == "error"
            assert "kaboom" in result.checks[0].summary

    @patch("replication.quick_scan._run_preflight")
    def test_preflight_pass(self, mock_pf):
        mock_pf.return_value = CheckResult(name="preflight", status="pass", summary="OK")
        scanner = QuickScanner(checks=["preflight"])
        result = scanner.run()
        assert result.checks[0].status == "pass"
        assert result.passed is True

    def test_scorecard_warn(self):
        mock_cr = CheckResult(name="scorecard", status="warn", score=55.0)
        with patch.dict("replication.quick_scan._RUNNERS", {"scorecard": lambda: mock_cr}):
            scanner = QuickScanner(checks=["scorecard"])
            result = scanner.run()
            assert result.checks[0].score == 55.0
            assert result.passed is True  # warn is ok in non-strict

    def test_total_duration_positive(self):
        scanner = QuickScanner(checks=[])
        result = scanner.run()
        assert result.total_duration_s >= 0


# ── CLI ──────────────────────────────────────────────────────────────


class TestCLI:
    @patch("replication.quick_scan.QuickScanner")
    def test_main_default(self, MockScanner, capsys):
        mock_result = ScanResult(timestamp="2026-01-01T00:00:00Z")
        mock_result.checks.append(CheckResult(name="test", status="pass"))
        MockScanner.return_value.run.return_value = mock_result
        main([])
        out = capsys.readouterr().out
        assert "PASS" in out

    @patch("replication.quick_scan.QuickScanner")
    def test_main_json_output(self, MockScanner, capsys):
        mock_result = ScanResult(timestamp="2026-01-01T00:00:00Z")
        mock_result.checks.append(CheckResult(name="test", status="pass"))
        MockScanner.return_value.run.return_value = mock_result
        main(["--json"])
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["passed"] is True

    @patch("replication.quick_scan.QuickScanner")
    def test_main_strict(self, MockScanner, capsys):
        mock_result = ScanResult(strict=True, timestamp="2026-01-01T00:00:00Z")
        MockScanner.return_value.run.return_value = mock_result
        main(["--strict"])
        MockScanner.assert_called_once_with(checks=None, strict=True)

    @patch("replication.quick_scan.QuickScanner")
    def test_main_custom_checks(self, MockScanner, capsys):
        mock_result = ScanResult(timestamp="2026-01-01T00:00:00Z")
        MockScanner.return_value.run.return_value = mock_result
        main(["--checks", "preflight,scorecard"])
        MockScanner.assert_called_once_with(checks=["preflight", "scorecard"], strict=False)
