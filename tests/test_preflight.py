"""Tests for the preflight check module."""

import json
import pytest

from replication.preflight import (
    PreflightChecker,
    PreflightConfig,
    PreflightResult,
    Finding,
    Severity,
    Category,
    KNOWN_STRATEGIES,
    main,
)

class TestSeverity:
    def test_symbols(self):
        assert Severity.ERROR.symbol == "✖"
        assert Severity.WARNING.symbol == "⚠"
        assert Severity.INFO.symbol == "ℹ"

    def test_color_codes(self):
        assert "\033[91m" in Severity.ERROR.color_code
        assert "\033[93m" in Severity.WARNING.color_code
        assert "\033[94m" in Severity.INFO.color_code

class TestFinding:
    def test_to_dict(self):
        f = Finding(Category.CONTRACT, Severity.ERROR, "CTR-001", "bad", "fix it")
        d = f.to_dict()
        assert d["category"] == "contract"
        assert d["severity"] == "error"
        assert d["code"] == "CTR-001"
        assert d["fix"] == "fix it"

    def test_to_dict_no_fix(self):
        f = Finding(Category.RESOURCES, Severity.INFO, "RES-010", "ok")
        d = f.to_dict()
        assert "fix" not in d

class TestPreflightResult:
    def test_passed_no_errors(self):
        r = PreflightResult(findings=[
            Finding(Category.CONTRACT, Severity.WARNING, "W", "warn"),
            Finding(Category.CONTRACT, Severity.INFO, "I", "info"),
        ])
        assert r.passed is True
        assert r.verdict == "GO (with warnings)"

    def test_failed_with_errors(self):
        r = PreflightResult(findings=[
            Finding(Category.CONTRACT, Severity.ERROR, "E", "err"),
        ])
        assert r.passed is False
        assert r.verdict == "NO-GO"
        assert r.failures == ["err"]

    def test_clean_pass(self):
        r = PreflightResult(findings=[
            Finding(Category.CONTRACT, Severity.INFO, "I", "ok"),
        ])
        assert r.verdict == "GO"

    def test_by_category(self):
        r = PreflightResult(findings=[
            Finding(Category.CONTRACT, Severity.INFO, "C1", "c"),
            Finding(Category.RESOURCES, Severity.INFO, "R1", "r"),
            Finding(Category.CONTRACT, Severity.WARNING, "C2", "c2"),
        ])
        by_cat = r.by_category()
        assert len(by_cat["contract"]) == 2
        assert len(by_cat["resources"]) == 1

    def test_render(self):
        r = PreflightResult(
            findings=[Finding(Category.CONTRACT, Severity.INFO, "C1", "ok")],
            config=PreflightConfig(),
            elapsed_ms=1.5,
        )
        text = r.render()
        assert "Preflight Check Report" in text
        assert "GO" in text

    def test_render_with_fixes(self):
        r = PreflightResult(findings=[
            Finding(Category.CONTRACT, Severity.WARNING, "C1", "warn", "do this"),
        ])
        text = r.render(show_fixes=True)
        assert "do this" in text

    def test_to_dict(self):
        r = PreflightResult(
            findings=[Finding(Category.CONTRACT, Severity.ERROR, "E", "err")],
            config=PreflightConfig(),
            elapsed_ms=2.0,
        )
        d = r.to_dict()
        assert d["passed"] is False
        assert d["counts"]["errors"] == 1
        assert "config" in d

class TestPreflightChecker:
    def test_default_config_passes(self):
        result = PreflightChecker(PreflightConfig()).run()
        assert result.passed

    def test_negative_max_depth_error(self):
        result = PreflightChecker(PreflightConfig(max_depth=-1)).run()
        assert not result.passed
        assert any(f.code == "CTR-001" for f in result.errors)

    def test_zero_max_depth_warning(self):
        result = PreflightChecker(PreflightConfig(max_depth=0)).run()
        assert result.passed  # warning, not error
        assert any(f.code == "CTR-002" for f in result.warnings)

    def test_high_max_depth_warning(self):
        result = PreflightChecker(PreflightConfig(max_depth=15)).run()
        assert any(f.code == "CTR-003" for f in result.warnings)

    def test_zero_max_replicas_error(self):
        result = PreflightChecker(PreflightConfig(max_replicas=0)).run()
        assert not result.passed
        assert any(f.code == "CTR-004" for f in result.errors)

    def test_high_max_replicas_warning(self):
        result = PreflightChecker(PreflightConfig(max_replicas=200)).run()
        assert any(f.code == "CTR-005" for f in result.warnings)

    def test_negative_cooldown_error(self):
        result = PreflightChecker(PreflightConfig(cooldown_seconds=-1)).run()
        assert not result.passed
        assert any(f.code == "CTR-006" for f in result.errors)

    def test_zero_cooldown_warning(self):
        result = PreflightChecker(PreflightConfig(cooldown_seconds=0)).run()
        assert any(f.code == "CTR-007" for f in result.warnings)

    def test_expiration_negative_error(self):
        result = PreflightChecker(PreflightConfig(expiration_seconds=-5)).run()
        assert not result.passed
        assert any(f.code == "CTR-008" for f in result.errors)

    def test_expiration_less_than_cooldown_error(self):
        result = PreflightChecker(PreflightConfig(
            cooldown_seconds=10, expiration_seconds=5
        )).run()
        assert not result.passed
        assert any(f.code == "CTR-009" for f in result.errors)

    def test_zero_cpu_error(self):
        result = PreflightChecker(PreflightConfig(cpu_limit=0)).run()
        assert not result.passed
        assert any(f.code == "RES-001" for f in result.errors)

    def test_low_cpu_warning(self):
        result = PreflightChecker(PreflightConfig(cpu_limit=0.05)).run()
        assert any(f.code == "RES-002" for f in result.warnings)

    def test_zero_memory_error(self):
        result = PreflightChecker(PreflightConfig(memory_limit_mb=0)).run()
        assert not result.passed
        assert any(f.code == "RES-003" for f in result.errors)

    def test_low_memory_warning(self):
        result = PreflightChecker(PreflightConfig(memory_limit_mb=32)).run()
        assert any(f.code == "RES-004" for f in result.warnings)

    def test_high_total_cpu_warning(self):
        result = PreflightChecker(PreflightConfig(cpu_limit=4, max_replicas=10)).run()
        assert any(f.code == "RES-005" for f in result.warnings)

    def test_high_total_memory_warning(self):
        result = PreflightChecker(PreflightConfig(memory_limit_mb=8192, max_replicas=10)).run()
        assert any(f.code == "RES-006" for f in result.warnings)

    def test_unknown_strategy_error(self):
        result = PreflightChecker(PreflightConfig(strategy="nonexistent")).run()
        assert not result.passed
        assert any(f.code == "STR-002" for f in result.errors)

    def test_valid_strategy(self):
        result = PreflightChecker(PreflightConfig(strategy="greedy")).run()
        assert any(f.code == "STR-010" for f in result.infos)

    def test_greedy_zero_cooldown_warning(self):
        result = PreflightChecker(PreflightConfig(
            strategy="greedy", cooldown_seconds=0
        )).run()
        assert any(f.code == "STR-003" for f in result.warnings)

    def test_exponential_high_depth_warning(self):
        result = PreflightChecker(PreflightConfig(
            strategy="exponential", max_depth=8
        )).run()
        assert any(f.code == "STR-004" for f in result.warnings)

    def test_no_strategy_info(self):
        result = PreflightChecker(PreflightConfig()).run()
        assert any(f.code == "STR-001" for f in result.infos)

    def test_external_network_warning(self):
        result = PreflightChecker(PreflightConfig(allow_external_network=True)).run()
        assert any(f.code == "SCL-003" for f in result.warnings)

    def test_scalability_high_replicas(self):
        result = PreflightChecker(PreflightConfig(max_depth=10, max_replicas=200)).run()
        assert any(f.code == "SCL-001" for f in result.warnings)

    def test_few_safeguards_warning(self):
        result = PreflightChecker(PreflightConfig(
            max_depth=0, max_replicas=200, cooldown_seconds=0
        )).run()
        assert any(f.code == "STP-001" for f in result.warnings)

    def test_good_safeguards(self):
        result = PreflightChecker(PreflightConfig(
            max_depth=3, max_replicas=10, cooldown_seconds=1.0, expiration_seconds=60
        )).run()
        assert any(f.code == "STP-010" for f in result.infos)

    def test_no_expiration_info(self):
        result = PreflightChecker(PreflightConfig()).run()
        assert any(f.code == "STP-002" for f in result.infos)

    def test_elapsed_time_recorded(self):
        result = PreflightChecker(PreflightConfig()).run()
        assert result.elapsed_ms >= 0

class TestCLI:
    def test_default_run(self, capsys):
        main([])
        out = capsys.readouterr().out
        assert "Preflight Check Report" in out

    def test_json_output(self, capsys):
        main(["--json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "verdict" in data
        assert "passed" in data

    def test_fix_flag(self, capsys):
        with pytest.raises(SystemExit):
            main(["--max-depth", "-1", "--fix"])
        out = capsys.readouterr().out
        assert "Fix:" in out

    def test_strategy_flag(self, capsys):
        main(["--strategy", "greedy"])
        out = capsys.readouterr().out
        assert "greedy" in out

    def test_all_strategies(self, capsys):
        main(["--all-strategies"])
        out = capsys.readouterr().out
        for s in KNOWN_STRATEGIES:
            assert s in out

    def test_bad_config_exits_1(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["--max-depth", "-1"])
        assert exc_info.value.code == 1

    def test_json_with_error(self, capsys):
        with pytest.raises(SystemExit):
            main(["--max-depth", "-1", "--json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["passed"] is False

    def test_all_flags(self, capsys):
        main(["--max-depth", "2", "--max-replicas", "5", "--cooldown", "2",
              "--cpu", "0.5", "--memory", "256", "--strategy", "conservative"])
        out = capsys.readouterr().out
        assert "GO" in out

    def test_expiration_flag(self, capsys):
        main(["--expiration", "30"])
        out = capsys.readouterr().out
        assert "Preflight" in out

    def test_policy_flag(self, capsys):
        main(["--policy", "strict"])
        out = capsys.readouterr().out
        assert "Preflight" in out
