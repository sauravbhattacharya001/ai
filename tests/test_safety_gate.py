"""Tests for replication.safety_gate — pre-deployment readiness checker.

Pins down the built-in checks, custom-check evaluator, aggregation logic,
JSON serialiser, and CLI entry point. The CLI exit code contract is
load-bearing for downstream CI integrations (Verdict.FAIL → exit 1) so
that path is exercised explicitly here.
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from replication.safety_gate import (
    BUILTIN_CHECKS,
    CheckResult,
    DEFAULT_AGENT,
    GateResult,
    Verdict,
    check_alignment_score,
    check_audit_logging,
    check_kill_switch,
    check_replication_depth,
    check_resource_limits,
    check_restricted_actions,
    check_safety_contract,
    check_sandbox,
    check_version,
    check_watermark,
    main,
    print_result,
    result_to_json,
    run_gate,
)


# ── individual checks ────────────────────────────────────────────────


class TestKillSwitch:
    def test_enabled_passes(self):
        assert check_kill_switch({"kill_switch": True}).verdict is Verdict.PASS

    def test_disabled_fails(self):
        assert check_kill_switch({"kill_switch": False}).verdict is Verdict.FAIL

    def test_missing_fails(self):
        # Absence is treated as disabled; agents must opt in explicitly.
        assert check_kill_switch({}).verdict is Verdict.FAIL


class TestReplicationDepth:
    def test_missing_fails(self):
        assert check_replication_depth({}).verdict is Verdict.FAIL

    def test_within_normal_threshold(self):
        assert check_replication_depth({"max_replication_depth": 5}).verdict is Verdict.PASS

    def test_above_normal_warns(self):
        assert check_replication_depth({"max_replication_depth": 6}).verdict is Verdict.WARN

    def test_strict_threshold(self):
        # Strict mode tightens the bar from 5 → 2.
        loose = check_replication_depth({"max_replication_depth": 3}, strict=False)
        strict = check_replication_depth({"max_replication_depth": 3}, strict=True)
        assert loose.verdict is Verdict.PASS
        assert strict.verdict is Verdict.WARN

    def test_zero_depth_passes(self):
        # Zero is a valid (and very safe) limit; must not be misread as falsy.
        assert check_replication_depth({"max_replication_depth": 0}).verdict is Verdict.PASS


class TestRestrictedActions:
    def test_clean_config_passes(self):
        agent = {
            "allowed_actions": ["read", "write"],
            "restricted_actions": ["self_modify"],
        }
        assert check_restricted_actions(agent).verdict is Verdict.PASS

    def test_dangerous_leak_fails(self):
        agent = {
            "allowed_actions": ["read", "self_modify"],
            "restricted_actions": [],
        }
        result = check_restricted_actions(agent)
        assert result.verdict is Verdict.FAIL
        assert "self_modify" in result.details["leaked"]

    def test_dangerous_action_also_listed_restricted_is_ok(self):
        # If an action appears in BOTH lists, the explicit restriction wins
        # — that's the documented "in allowed set but also restricted" path.
        agent = {
            "allowed_actions": ["self_modify"],
            "restricted_actions": ["self_modify"],
        }
        assert check_restricted_actions(agent).verdict is Verdict.PASS


class TestAuditLogging:
    def test_enabled_passes(self):
        assert check_audit_logging({"audit_logging": True}).verdict is Verdict.PASS

    def test_disabled_warns_in_loose_mode(self):
        assert check_audit_logging({"audit_logging": False}).verdict is Verdict.WARN

    def test_disabled_fails_in_strict_mode(self):
        assert check_audit_logging({"audit_logging": False}, strict=True).verdict is Verdict.FAIL


class TestAlignmentScore:
    def test_missing_warns(self):
        assert check_alignment_score({}).verdict is Verdict.WARN

    def test_above_threshold_passes(self):
        assert check_alignment_score({"alignment_score": 0.95}).verdict is Verdict.PASS

    def test_below_threshold_but_not_critical_warns(self):
        assert check_alignment_score({"alignment_score": 0.6}).verdict is Verdict.WARN

    def test_critically_low_fails(self):
        assert check_alignment_score({"alignment_score": 0.2}).verdict is Verdict.FAIL

    def test_strict_raises_threshold(self):
        # 0.8 is fine in loose mode (>= 0.7) but warns in strict (< 0.9).
        assert check_alignment_score({"alignment_score": 0.8}).verdict is Verdict.PASS
        assert check_alignment_score({"alignment_score": 0.8}, strict=True).verdict is Verdict.WARN


class TestResourceLimits:
    def test_complete_limits_pass(self):
        agent = {"resource_limits": {"max_cpu_percent": 80, "max_memory_mb": 1024}}
        assert check_resource_limits(agent).verdict is Verdict.PASS

    def test_missing_fields_fail(self):
        agent = {"resource_limits": {"max_cpu_percent": 80}}
        r = check_resource_limits(agent)
        assert r.verdict is Verdict.FAIL
        assert "max_memory_mb" in r.details["missing"]

    def test_no_block_fails(self):
        r = check_resource_limits({})
        assert r.verdict is Verdict.FAIL
        assert sorted(r.details["missing"]) == ["max_cpu_percent", "max_memory_mb"]


class TestSafetyContract:
    def test_enabled_passes(self):
        assert check_safety_contract({"safety_contract": True}).verdict is Verdict.PASS

    def test_disabled_warns_loose(self):
        assert check_safety_contract({"safety_contract": False}).verdict is Verdict.WARN

    def test_disabled_fails_strict(self):
        assert check_safety_contract({"safety_contract": False}, strict=True).verdict is Verdict.FAIL


class TestWatermark:
    def test_enabled_passes(self):
        assert check_watermark({"watermark_enabled": True}).verdict is Verdict.PASS

    def test_disabled_warns(self):
        # Watermark is advisory — always WARN, even in strict mode.
        assert check_watermark({"watermark_enabled": False}).verdict is Verdict.WARN
        assert check_watermark({"watermark_enabled": False}, strict=True).verdict is Verdict.WARN


class TestSandbox:
    def test_enabled_passes(self):
        assert check_sandbox({"sandbox_mode": True}).verdict is Verdict.PASS

    def test_disabled_warns_loose(self):
        assert check_sandbox({"sandbox_mode": False}).verdict is Verdict.WARN

    def test_disabled_fails_strict(self):
        assert check_sandbox({"sandbox_mode": False}, strict=True).verdict is Verdict.FAIL


class TestVersion:
    def test_present_passes(self):
        r = check_version({"version": "1.2.3"})
        assert r.verdict is Verdict.PASS
        assert "1.2.3" in r.message

    def test_missing_fails(self):
        assert check_version({}).verdict is Verdict.FAIL

    def test_empty_string_fails(self):
        # Empty string is falsy and should still trip the no-version check.
        assert check_version({"version": ""}).verdict is Verdict.FAIL


# ── aggregation ──────────────────────────────────────────────────────


class TestGateResultAggregation:
    def test_starts_clean(self):
        g = GateResult()
        assert g.overall is Verdict.PASS
        assert g.passed == g.warned == g.failed == 0

    def test_pass_only(self):
        g = GateResult()
        for i in range(3):
            g.add(CheckResult(f"x{i}", Verdict.PASS, "ok"))
        assert g.overall is Verdict.PASS
        assert g.passed == 3

    def test_warn_promotes_overall(self):
        g = GateResult()
        g.add(CheckResult("a", Verdict.PASS, "ok"))
        g.add(CheckResult("b", Verdict.WARN, "meh"))
        assert g.overall is Verdict.WARN
        assert g.warned == 1

    def test_fail_dominates(self):
        g = GateResult()
        g.add(CheckResult("a", Verdict.PASS, "ok"))
        g.add(CheckResult("b", Verdict.WARN, "meh"))
        g.add(CheckResult("c", Verdict.FAIL, "no"))
        assert g.overall is Verdict.FAIL
        # Adding a later PASS must not downgrade to PASS.
        g.add(CheckResult("d", Verdict.PASS, "ok"))
        assert g.overall is Verdict.FAIL

    def test_warn_does_not_overwrite_fail(self):
        # Regression guard: once FAIL is seen, neither WARN nor PASS may
        # bring the overall verdict back up.
        g = GateResult()
        g.add(CheckResult("c", Verdict.FAIL, "no"))
        g.add(CheckResult("b", Verdict.WARN, "meh"))
        assert g.overall is Verdict.FAIL


# ── run_gate end-to-end ──────────────────────────────────────────────


class TestRunGate:
    def test_demo_agent_passes(self):
        # The shipped DEFAULT_AGENT is meant to demonstrate a fully-
        # compliant configuration; PASS is part of its contract.
        result = run_gate(DEFAULT_AGENT)
        assert result.overall is Verdict.PASS
        # All built-in checks must have executed.
        assert len(result.checks) == len(BUILTIN_CHECKS)
        assert result.failed == 0

    def test_dangerous_agent_fails(self):
        agent = {
            "name": "rogue",
            "version": "0.1.0",
            "kill_switch": False,
            "max_replication_depth": 100,
            "allowed_actions": ["self_modify", "network_exfiltrate"],
            "restricted_actions": [],
            "audit_logging": False,
            "alignment_score": 0.1,
            "resource_limits": {},
            "safety_contract": False,
            "watermark_enabled": False,
            "sandbox_mode": False,
        }
        result = run_gate(agent)
        assert result.overall is Verdict.FAIL
        assert result.failed >= 4  # kill switch + restricted + alignment + resource + version (passes)

    def test_strict_mode_is_stricter(self):
        # Take the demo agent and degrade ONE field that strict turns
        # from WARN-only to FAIL.
        agent = dict(DEFAULT_AGENT)
        agent["audit_logging"] = False
        loose = run_gate(agent, strict=False)
        strict = run_gate(agent, strict=True)
        assert loose.overall is Verdict.WARN
        assert strict.overall is Verdict.FAIL

    def test_custom_check_pass(self):
        custom = [{"field": "alignment_score", "op": ">=", "value": 0.5}]
        result = run_gate(DEFAULT_AGENT, custom_checks=custom)
        assert any(c.name == "custom:alignment_score" and c.verdict is Verdict.PASS
                   for c in result.checks)

    def test_custom_check_fail(self):
        custom = [{"field": "alignment_score", "op": ">=", "value": 0.99}]
        result = run_gate(DEFAULT_AGENT, custom_checks=custom)
        assert any(c.name == "custom:alignment_score" and c.verdict is Verdict.FAIL
                   for c in result.checks)
        assert result.overall is Verdict.FAIL

    def test_custom_check_missing_field_warns(self):
        custom = [{"field": "nonexistent_field", "op": ">=", "value": 0}]
        result = run_gate(DEFAULT_AGENT, custom_checks=custom)
        custom_results = [c for c in result.checks if c.name.startswith("custom:")]
        assert len(custom_results) == 1
        assert custom_results[0].verdict is Verdict.WARN

    @pytest.mark.parametrize(
        "op,actual,threshold,expected",
        [
            (">=", 5, 5, Verdict.PASS),
            (">=", 4, 5, Verdict.FAIL),
            ("<=", 5, 5, Verdict.PASS),
            ("<=", 6, 5, Verdict.FAIL),
            ("==", 5, 5, Verdict.PASS),
            ("==", 4, 5, Verdict.FAIL),
            (">",  6, 5, Verdict.PASS),
            (">",  5, 5, Verdict.FAIL),
            ("<",  4, 5, Verdict.PASS),
            ("<",  5, 5, Verdict.FAIL),
        ],
    )
    def test_custom_check_operators(self, op, actual, threshold, expected):
        agent = {**DEFAULT_AGENT, "k": actual}
        result = run_gate(agent, custom_checks=[{"field": "k", "op": op, "value": threshold}])
        match = [c for c in result.checks if c.name == "custom:k"]
        assert match and match[0].verdict is expected

    def test_unknown_operator_treated_as_failure(self):
        # An unrecognised op shouldn't crash — the runner falls through
        # to the "passed=False" branch and emits a FAIL custom result.
        agent = {**DEFAULT_AGENT, "k": 5}
        result = run_gate(agent, custom_checks=[{"field": "k", "op": "!=", "value": 5}])
        match = [c for c in result.checks if c.name == "custom:k"]
        assert match and match[0].verdict is Verdict.FAIL


# ── JSON serialiser ──────────────────────────────────────────────────


class TestResultToJson:
    def test_shape(self):
        result = run_gate(DEFAULT_AGENT)
        payload = json.loads(result_to_json(result))
        assert payload["overall"] == "PASS"
        assert payload["passed"] + payload["warned"] + payload["failed"] == len(result.checks)
        # Each check must round-trip its identity & verdict.
        names = {c["name"] for c in payload["checks"]}
        assert "kill_switch" in names
        assert "version" in names

    def test_details_only_emitted_when_present(self):
        # CheckResult.details is optional; a check with no details
        # block (e.g. kill_switch PASS path) must not carry an empty
        # "details": null in the JSON output.
        result = run_gate(DEFAULT_AGENT)
        payload = json.loads(result_to_json(result))
        ks = next(c for c in payload["checks"] if c["name"] == "kill_switch")
        assert "details" not in ks

    def test_details_emitted_when_present(self):
        result = run_gate(DEFAULT_AGENT)
        payload = json.loads(result_to_json(result))
        depth = next(c for c in payload["checks"] if c["name"] == "replication_depth")
        assert depth["details"]["limit"] == DEFAULT_AGENT["max_replication_depth"]


# ── printer (smoke only) ─────────────────────────────────────────────


class TestPrintResult:
    def test_runs_without_color(self):
        result = run_gate(DEFAULT_AGENT)
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_result(result, use_color=False)
        out = buf.getvalue()
        assert "SAFETY GATE REPORT" in out
        assert "DEPLOY APPROVED" in out  # demo agent is fully compliant
        # No ANSI escape sequences should leak when colour is off.
        assert "\033[" not in out

    def test_failed_run_labels_as_blocked(self):
        bad = dict(DEFAULT_AGENT)
        bad["kill_switch"] = False
        result = run_gate(bad)
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_result(result, use_color=False)
        assert "DEPLOY BLOCKED" in buf.getvalue()

    def test_warn_run_labels_as_caution(self):
        warny = dict(DEFAULT_AGENT)
        warny["audit_logging"] = False  # → WARN in loose mode
        result = run_gate(warny)
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_result(result, use_color=False)
        assert "DEPLOY WITH CAUTION" in buf.getvalue()


# ── CLI ──────────────────────────────────────────────────────────────


class _Utf8Stdout(io.TextIOWrapper):
    """A TextIOWrapper whose ``.encoding`` is already ``utf-8``.

    The CLI rewraps ``sys.stdout`` whenever its encoding is anything
    other than ``utf-8`` (Windows console workaround). Under pytest's
    capture, the captured stdout reports a different encoding, so the
    CLI detaches the buffer that capsys was watching and the test sees
    an empty string. Swapping in a wrapper that *already* claims utf-8
    short-circuits the rewrap and keeps the captured output flowing
    through the same underlying buffer.
    """


def _run_cli_capture(argv):
    """Run ``safety_gate.main`` with stdout captured into a buffer.

    Bypasses pytest's capsys because the CLI rewraps ``sys.stdout``,
    which detaches capsys's pipe (see docstring on ``_Utf8Stdout``).
    Returns ``(exit_code, captured_text)``.
    """
    buf = io.BytesIO()
    wrapper = _Utf8Stdout(buf, encoding="utf-8", write_through=True)
    saved = sys.stdout
    sys.stdout = wrapper
    try:
        with pytest.raises(SystemExit) as exc:
            main(argv)
        wrapper.flush()
        return int(exc.value.code or 0), buf.getvalue().decode("utf-8")
    finally:
        # Restore unconditionally; the CLI may have re-bound
        # ``sys.stdout`` to its own TextIOWrapper.
        sys.stdout = saved


class TestCli:
    def test_demo_run_exits_zero(self):
        # No args → demo agent → all PASS → exit 0.
        code, out = _run_cli_capture([])
        assert code == 0
        assert "SAFETY GATE REPORT" in out

    def test_failing_config_exits_one(self, tmp_path: Path):
        bad = dict(DEFAULT_AGENT)
        bad["kill_switch"] = False
        cfg = tmp_path / "bad.json"
        cfg.write_text(json.dumps(bad), encoding="utf-8")
        code, _ = _run_cli_capture(["--config", str(cfg), "--no-color"])
        assert code == 1

    def test_json_output(self, tmp_path: Path):
        cfg = tmp_path / "ok.json"
        cfg.write_text(json.dumps(DEFAULT_AGENT), encoding="utf-8")
        code, out = _run_cli_capture(["--config", str(cfg), "--format", "json"])
        assert code == 0
        payload = json.loads(out)
        assert payload["overall"] == "PASS"

    def test_strict_flag_promotes_warn_to_fail(self, tmp_path: Path):
        agent = dict(DEFAULT_AGENT)
        agent["audit_logging"] = False
        cfg = tmp_path / "warn.json"
        cfg.write_text(json.dumps(agent), encoding="utf-8")
        # Loose mode → WARN → exit 0.
        code, _ = _run_cli_capture(["--config", str(cfg), "--no-color"])
        assert code == 0
        # Strict mode → FAIL → exit 1.
        code, _ = _run_cli_capture(["--config", str(cfg), "--strict", "--no-color"])
        assert code == 1

    def test_custom_checks_file(self, tmp_path: Path):
        cfg = tmp_path / "agent.json"
        cfg.write_text(json.dumps(DEFAULT_AGENT), encoding="utf-8")
        custom = tmp_path / "checks.json"
        # Impossible alignment requirement → custom check FAILs → exit 1.
        custom.write_text(
            json.dumps([{"field": "alignment_score", "op": ">=", "value": 0.999}]),
            encoding="utf-8",
        )
        code, out = _run_cli_capture([
            "--config", str(cfg),
            "--custom-checks", str(custom),
            "--format", "json",
        ])
        assert code == 1
        payload = json.loads(out)
        assert payload["overall"] == "FAIL"
        assert any(c["name"] == "custom:alignment_score" and c["verdict"] == "FAIL"
                   for c in payload["checks"])
