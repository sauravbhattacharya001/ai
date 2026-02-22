"""Tests for the Safety Policy Engine (replication.policy)."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from replication.policy import (
    MONTE_CARLO_METRICS,
    POLICY_PRESETS,
    SINGLE_RUN_METRICS,
    Operator,
    PolicyResult,
    PolicyRule,
    RuleResult,
    SafetyPolicy,
    Severity,
    _extract_metric,
)
from replication.simulator import ScenarioConfig, SimulationReport, Simulator


# ── Helpers ─────────────────────────────────────────────────────────────

def _make_report(
    strategy: str = "greedy",
    max_depth: int = 3,
    max_replicas: int = 10,
    seed: int = 42,
    **kwargs,
) -> SimulationReport:
    """Run a quick simulation and return the report."""
    config = ScenarioConfig(
        strategy=strategy,
        max_depth=max_depth,
        max_replicas=max_replicas,
        seed=seed,
        **kwargs,
    )
    return Simulator(config).run()


# ── Operator tests ──────────────────────────────────────────────────────

class TestOperator:
    """Tests for Operator enum and evaluation."""

    def test_lt(self):
        assert Operator.LT.evaluate(5, 10) is True
        assert Operator.LT.evaluate(10, 10) is False
        assert Operator.LT.evaluate(15, 10) is False

    def test_le(self):
        assert Operator.LE.evaluate(5, 10) is True
        assert Operator.LE.evaluate(10, 10) is True
        assert Operator.LE.evaluate(15, 10) is False

    def test_gt(self):
        assert Operator.GT.evaluate(15, 10) is True
        assert Operator.GT.evaluate(10, 10) is False
        assert Operator.GT.evaluate(5, 10) is False

    def test_ge(self):
        assert Operator.GE.evaluate(15, 10) is True
        assert Operator.GE.evaluate(10, 10) is True
        assert Operator.GE.evaluate(5, 10) is False

    def test_eq(self):
        assert Operator.EQ.evaluate(10, 10) is True
        assert Operator.EQ.evaluate(10 + 1e-10, 10) is True  # within epsilon
        assert Operator.EQ.evaluate(11, 10) is False

    def test_ne(self):
        assert Operator.NE.evaluate(11, 10) is True
        assert Operator.NE.evaluate(10, 10) is False

    def test_from_str_symbols(self):
        assert Operator.from_str("<") == Operator.LT
        assert Operator.from_str("<=") == Operator.LE
        assert Operator.from_str(">") == Operator.GT
        assert Operator.from_str(">=") == Operator.GE
        assert Operator.from_str("==") == Operator.EQ
        assert Operator.from_str("!=") == Operator.NE

    def test_from_str_words(self):
        assert Operator.from_str("lt") == Operator.LT
        assert Operator.from_str("LE") == Operator.LE
        assert Operator.from_str("Gt") == Operator.GT
        assert Operator.from_str("ge") == Operator.GE
        assert Operator.from_str("eq") == Operator.EQ
        assert Operator.from_str("ne") == Operator.NE

    def test_from_str_invalid(self):
        with pytest.raises(ValueError, match="Unknown operator"):
            Operator.from_str("like")

    def test_from_str_whitespace(self):
        assert Operator.from_str("  <=  ") == Operator.LE


# ── PolicyRule tests ────────────────────────────────────────────────────

class TestPolicyRule:
    """Tests for PolicyRule data model."""

    def test_create_basic(self):
        rule = PolicyRule("total_workers", Operator.LE, 50)
        assert rule.metric == "total_workers"
        assert rule.operator == Operator.LE
        assert rule.threshold == 50
        assert rule.severity == Severity.ERROR  # default
        assert rule.monte_carlo is False

    def test_evaluate_pass(self):
        rule = PolicyRule("total_workers", Operator.LE, 50)
        assert rule.evaluate(30) is True

    def test_evaluate_fail(self):
        rule = PolicyRule("total_workers", Operator.LE, 50)
        assert rule.evaluate(60) is False

    def test_render_expression(self):
        rule = PolicyRule("denial_rate", Operator.GE, 0.5, description="Must deny ≥50%")
        expr = rule.render_expression()
        assert "denial_rate" in expr
        assert ">=" in expr
        assert "0.5" in expr

    def test_to_dict(self):
        rule = PolicyRule("total_workers", Operator.LE, 50, Severity.WARNING,
                          "Worker limit", monte_carlo=True)
        d = rule.to_dict()
        assert d["metric"] == "total_workers"
        assert d["operator"] == "<="
        assert d["threshold"] == 50
        assert d["severity"] == "warning"
        assert d["description"] == "Worker limit"
        assert d["monte_carlo"] is True

    def test_from_dict(self):
        d = {
            "metric": "denial_rate",
            "operator": ">=",
            "threshold": 0.3,
            "severity": "warning",
            "description": "Deny threshold",
        }
        rule = PolicyRule.from_dict(d)
        assert rule.metric == "denial_rate"
        assert rule.operator == Operator.GE
        assert rule.threshold == 0.3
        assert rule.severity == Severity.WARNING

    def test_from_dict_defaults(self):
        d = {"metric": "total_workers", "operator": "<", "threshold": 100}
        rule = PolicyRule.from_dict(d)
        assert rule.severity == Severity.ERROR  # default
        assert rule.description == ""
        assert rule.monte_carlo is False

    def test_roundtrip(self):
        original = PolicyRule("risk_score", Operator.LE, 0.5, Severity.ERROR,
                              "Risk limit", monte_carlo=True)
        restored = PolicyRule.from_dict(original.to_dict())
        assert restored.metric == original.metric
        assert restored.operator == original.operator
        assert restored.threshold == original.threshold
        assert restored.severity == original.severity
        assert restored.description == original.description
        assert restored.monte_carlo == original.monte_carlo


# ── RuleResult tests ────────────────────────────────────────────────────

class TestRuleResult:
    """Tests for RuleResult rendering and properties."""

    def test_passed_icon(self):
        rule = PolicyRule("total_workers", Operator.LE, 50)
        result = RuleResult(rule=rule, actual=30, passed=True)
        assert result.icon == "✅"
        assert result.status == "PASS"

    def test_error_icon(self):
        rule = PolicyRule("total_workers", Operator.LE, 50, Severity.ERROR)
        result = RuleResult(rule=rule, actual=60, passed=False)
        assert result.icon == "❌"
        assert result.status == "ERROR"

    def test_warning_icon(self):
        rule = PolicyRule("total_workers", Operator.LE, 50, Severity.WARNING)
        result = RuleResult(rule=rule, actual=60, passed=False)
        assert result.icon == "⚠️"
        assert result.status == "WARNING"

    def test_info_icon(self):
        rule = PolicyRule("total_workers", Operator.LE, 50, Severity.INFO)
        result = RuleResult(rule=rule, actual=60, passed=False)
        assert result.icon == "ℹ️"
        assert result.status == "INFO"

    def test_render(self):
        rule = PolicyRule("total_workers", Operator.LE, 50, description="Cap workers")
        result = RuleResult(rule=rule, actual=30, passed=True)
        rendered = result.render()
        assert "PASS" in rendered
        assert "total_workers" in rendered

    def test_extraction_error_icon(self):
        rule = PolicyRule("bad_metric", Operator.LE, 50, Severity.ERROR)
        result = RuleResult(rule=rule, actual=0.0, passed=False, error="Unknown metric: bad_metric")
        assert result.icon == "💥"
        assert result.status == "ERROR"

    def test_extraction_error_render(self):
        rule = PolicyRule("bad_metric", Operator.LE, 50, Severity.ERROR)
        result = RuleResult(rule=rule, actual=0.0, passed=False, error="Unknown metric: bad_metric")
        rendered = result.render()
        assert "error:" in rendered
        assert "bad_metric" in rendered

    def test_extraction_error_to_dict(self):
        rule = PolicyRule("bad_metric", Operator.LE, 50)
        result = RuleResult(rule=rule, actual=0.0, passed=False, error="Unknown metric: bad_metric")
        d = result.to_dict()
        assert d["error"] == "Unknown metric: bad_metric"
        assert d["passed"] is False

    def test_no_error_field_when_none(self):
        rule = PolicyRule("total_workers", Operator.LE, 50)
        result = RuleResult(rule=rule, actual=30, passed=True)
        d = result.to_dict()
        assert "error" not in d

    def test_to_dict(self):
        rule = PolicyRule("total_workers", Operator.LE, 50)
        result = RuleResult(rule=rule, actual=30.123456, passed=True)
        d = result.to_dict()
        assert d["passed"] is True
        assert d["actual"] == 30.123456
        assert d["status"] == "PASS"


# ── Metric extraction tests ────────────────────────────────────────────

class TestMetricExtraction:
    """Tests for extracting metrics from simulation reports."""

    @pytest.fixture
    def report(self):
        return _make_report(strategy="greedy", max_depth=3, max_replicas=10, seed=42)

    def test_total_workers(self, report):
        val = _extract_metric(report, "total_workers")
        assert val == float(len(report.workers))
        assert val >= 1  # at least root

    def test_total_tasks(self, report):
        val = _extract_metric(report, "total_tasks")
        assert val == float(report.total_tasks)
        assert val >= 1

    def test_max_depth_reached(self, report):
        val = _extract_metric(report, "max_depth_reached")
        max_d = max(w.depth for w in report.workers.values())
        assert val == float(max_d)

    def test_repl_succeeded(self, report):
        val = _extract_metric(report, "repl_succeeded")
        assert val == float(report.total_replications_succeeded)

    def test_repl_denied(self, report):
        val = _extract_metric(report, "repl_denied")
        assert val == float(report.total_replications_denied)

    def test_repl_attempted(self, report):
        val = _extract_metric(report, "repl_attempted")
        assert val == float(report.total_replications_attempted)

    def test_denial_rate(self, report):
        val = _extract_metric(report, "denial_rate")
        if report.total_replications_attempted > 0:
            expected = report.total_replications_denied / report.total_replications_attempted
            assert abs(val - expected) < 1e-9
        else:
            assert val == 0.0

    def test_success_rate(self, report):
        val = _extract_metric(report, "success_rate")
        if report.total_replications_attempted > 0:
            expected = report.total_replications_succeeded / report.total_replications_attempted
            assert abs(val - expected) < 1e-9

    def test_depth_utilization(self, report):
        val = _extract_metric(report, "depth_utilization")
        max_d = max(w.depth for w in report.workers.values())
        expected = max_d / report.config.max_depth
        assert abs(val - expected) < 1e-9

    def test_worker_utilization(self, report):
        val = _extract_metric(report, "worker_utilization")
        expected = len(report.workers) / report.config.max_replicas
        assert abs(val - expected) < 1e-9

    def test_duration_ms(self, report):
        val = _extract_metric(report, "duration_ms")
        assert val >= 0

    def test_avg_tasks_per_worker(self, report):
        val = _extract_metric(report, "avg_tasks_per_worker")
        expected = report.total_tasks / len(report.workers)
        assert abs(val - expected) < 1e-9

    def test_unknown_metric_raises(self, report):
        with pytest.raises(ValueError, match="Unknown metric"):
            _extract_metric(report, "nonexistent_metric")

    def test_denial_rate_no_attempts(self):
        """denial_rate returns 0 when no replications attempted."""
        # Conservative with max_depth=1 means depth < 0 required to replicate → never
        report = _make_report(strategy="conservative", max_depth=1, seed=99)
        val = _extract_metric(report, "denial_rate")
        assert val == 0.0


# ── PolicyResult tests ──────────────────────────────────────────────────

class TestPolicyResult:
    """Tests for PolicyResult verdict logic."""

    def _make_result(self, rule_results):
        return PolicyResult(
            policy_name="Test Policy",
            rule_results=rule_results,
            duration_ms=1.0,
        )

    def test_all_pass(self):
        rule = PolicyRule("total_workers", Operator.LE, 100)
        rr = RuleResult(rule=rule, actual=50, passed=True)
        result = self._make_result([rr])
        assert result.passed is True
        assert result.verdict == "PASS"
        assert result.exit_code == 0

    def test_error_fails(self):
        rule = PolicyRule("total_workers", Operator.LE, 10, Severity.ERROR)
        rr = RuleResult(rule=rule, actual=50, passed=False)
        result = self._make_result([rr])
        assert result.passed is False
        assert result.verdict == "FAIL"
        assert result.exit_code == 1

    def test_warning_only(self):
        rule = PolicyRule("total_workers", Operator.LE, 10, Severity.WARNING)
        rr = RuleResult(rule=rule, actual=50, passed=False)
        result = self._make_result([rr])
        assert result.passed is True
        assert result.has_warnings is True
        assert result.verdict == "WARN"
        assert result.exit_code == 2

    def test_info_ignored(self):
        rule = PolicyRule("total_workers", Operator.LE, 10, Severity.INFO)
        rr = RuleResult(rule=rule, actual=50, passed=False)
        result = self._make_result([rr])
        assert result.passed is True
        assert result.verdict in ("PASS", "WARN")  # info doesn't affect verdict

    def test_extraction_error_fails_result(self):
        """An extraction error should cause the overall result to FAIL."""
        rule = PolicyRule("bad_metric", Operator.LE, 50, Severity.ERROR)
        rr = RuleResult(rule=rule, actual=0.0, passed=False, error="Unknown metric: bad_metric")
        result = self._make_result([rr])
        assert result.passed is False
        assert result.verdict == "FAIL"
        assert result.has_extraction_errors is True

    def test_extraction_error_even_warning_severity_fails(self):
        """Extraction errors should fail regardless of rule severity."""
        rule = PolicyRule("bad_metric", Operator.LE, 50, Severity.WARNING)
        rr = RuleResult(rule=rule, actual=0.0, passed=False, error="Unknown metric: bad_metric")
        result = self._make_result([rr])
        assert result.passed is False
        assert result.has_extraction_errors is True

    def test_extraction_errors_list(self):
        rr1 = RuleResult(PolicyRule("a", Operator.LE, 1), 0.5, True)
        rr2 = RuleResult(PolicyRule("bad", Operator.LE, 1), 0.0, False, error="Unknown metric: bad")
        result = self._make_result([rr1, rr2])
        assert len(result.extraction_errors) == 1
        assert result.extraction_errors[0].error == "Unknown metric: bad"

    def test_to_dict_includes_extraction_errors_count(self):
        rr = RuleResult(PolicyRule("bad", Operator.LE, 1), 0.0, False, error="Unknown metric: bad")
        result = self._make_result([rr])
        d = result.to_dict()
        assert d["has_extraction_errors"] is True
        assert d["rules_extraction_errors"] == 1

    def test_no_extraction_errors(self):
        rule = PolicyRule("total_workers", Operator.LE, 100)
        rr = RuleResult(rule=rule, actual=50, passed=True)
        result = self._make_result([rr])
        assert result.has_extraction_errors is False
        assert len(result.extraction_errors) == 0

    def test_mixed_error_and_warning(self):
        r1 = RuleResult(
            rule=PolicyRule("total_workers", Operator.LE, 10, Severity.ERROR),
            actual=50, passed=False,
        )
        r2 = RuleResult(
            rule=PolicyRule("denial_rate", Operator.GE, 0.5, Severity.WARNING),
            actual=0.2, passed=False,
        )
        result = self._make_result([r1, r2])
        assert result.passed is False
        assert len(result.errors) == 1
        assert len(result.warnings) == 1

    def test_errors_list(self):
        rules = [
            RuleResult(PolicyRule("a", Operator.LE, 1, Severity.ERROR), 5, False),
            RuleResult(PolicyRule("b", Operator.LE, 1, Severity.ERROR), 0.5, True),
            RuleResult(PolicyRule("c", Operator.LE, 1, Severity.WARNING), 5, False),
        ]
        result = self._make_result(rules)
        assert len(result.errors) == 1
        assert len(result.warnings) == 1
        assert len(result.failures) == 2
        assert len(result.passes) == 1

    def test_render_not_empty(self):
        rule = PolicyRule("total_workers", Operator.LE, 100)
        rr = RuleResult(rule=rule, actual=50, passed=True)
        result = self._make_result([rr])
        rendered = result.render()
        assert "Safety Policy" in rendered
        assert "PASS" in rendered

    def test_render_with_errors(self):
        rule = PolicyRule("total_workers", Operator.LE, 10, Severity.ERROR,
                          "Too many workers")
        rr = RuleResult(rule=rule, actual=50, passed=False)
        result = self._make_result([rr])
        rendered = result.render()
        assert "FAIL" in rendered
        assert "Errors" in rendered

    def test_to_dict(self):
        rule = PolicyRule("total_workers", Operator.LE, 100)
        rr = RuleResult(rule=rule, actual=50, passed=True)
        result = self._make_result([rr])
        d = result.to_dict()
        assert d["verdict"] == "PASS"
        assert d["passed"] is True
        assert d["exit_code"] == 0
        assert len(d["results"]) == 1


# ── SafetyPolicy tests ─────────────────────────────────────────────────

class TestSafetyPolicy:
    """Tests for SafetyPolicy creation and evaluation."""

    def test_create_empty(self):
        policy = SafetyPolicy("Empty", [])
        assert policy.name == "Empty"
        assert len(policy.rules) == 0

    def test_add_rule(self):
        policy = SafetyPolicy("Test", [])
        rule = PolicyRule("total_workers", Operator.LE, 50)
        policy.add_rule(rule)
        assert len(policy.rules) == 1

    def test_add_rule_fluent(self):
        policy = (
            SafetyPolicy("Test", [])
            .add_rule(PolicyRule("total_workers", Operator.LE, 50))
            .add_rule(PolicyRule("denial_rate", Operator.GE, 0.3))
        )
        assert len(policy.rules) == 2

    def test_evaluate_all_pass(self):
        policy = SafetyPolicy("Lenient", [
            PolicyRule("total_workers", Operator.LE, 1000),
            PolicyRule("max_depth_reached", Operator.LE, 100),
        ])
        report = _make_report(strategy="conservative", max_depth=2, seed=42)
        result = policy.evaluate(report)
        assert result.passed is True
        assert len(result.passes) == 2

    def test_evaluate_with_failure(self):
        policy = SafetyPolicy("Tight", [
            PolicyRule("total_workers", Operator.LE, 1, Severity.ERROR,
                       "Only root allowed"),
        ])
        report = _make_report(strategy="greedy", max_depth=3, seed=42)
        result = policy.evaluate(report)
        # Greedy with depth 3 should spawn more than 1 worker
        if len(report.workers) > 1:
            assert result.passed is False

    def test_evaluate_mc_rules_skipped_without_data(self):
        policy = SafetyPolicy("MC Only", [
            PolicyRule("prob_max_depth", Operator.LE, 0.5, monte_carlo=True),
        ])
        report = _make_report(seed=42)
        result = policy.evaluate(report, mc_result=None)
        # MC rules skipped when no MC data
        assert len(result.rule_results) == 0
        assert result.passed is True

    def test_evaluate_mixed_rules(self):
        policy = SafetyPolicy("Mixed", [
            PolicyRule("total_workers", Operator.LE, 1000, Severity.ERROR),
            PolicyRule("prob_max_depth", Operator.LE, 0.5, monte_carlo=True),
        ])
        report = _make_report(seed=42)
        result = policy.evaluate(report, mc_result=None)
        # Only single-run rule evaluated
        assert len(result.rule_results) == 1
        assert result.passed is True


class TestSafetyPolicyPresets:
    """Tests for built-in policy presets."""

    def test_all_presets_exist(self):
        for name in ("minimal", "standard", "strict", "ci"):
            assert name in POLICY_PRESETS

    def test_load_preset_minimal(self):
        policy = SafetyPolicy.from_preset("minimal")
        assert policy.name == "Minimal Safety"
        assert len(policy.rules) >= 2

    def test_load_preset_standard(self):
        policy = SafetyPolicy.from_preset("standard")
        assert policy.name == "Standard Safety"
        assert any(r.monte_carlo for r in policy.rules)

    def test_load_preset_strict(self):
        policy = SafetyPolicy.from_preset("strict")
        assert policy.name == "Strict Containment"
        error_rules = [r for r in policy.rules if r.severity == Severity.ERROR]
        assert len(error_rules) >= 5  # strict has many error-level rules

    def test_load_preset_ci(self):
        policy = SafetyPolicy.from_preset("ci")
        assert policy.name == "CI/CD Gate"
        # CI should have no MC rules (fast evaluation)
        assert not any(r.monte_carlo for r in policy.rules)

    def test_invalid_preset(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            SafetyPolicy.from_preset("nonexistent")

    def test_minimal_passes_conservative(self):
        """Conservative strategy with low limits should pass minimal policy."""
        policy = SafetyPolicy.from_preset("minimal")
        report = _make_report(strategy="conservative", max_depth=2, max_replicas=5, seed=42)
        result = policy.evaluate(report)
        assert result.passed is True

    def test_strict_may_fail_greedy(self):
        """Greedy strategy with high limits may fail strict policy."""
        policy = SafetyPolicy.from_preset("strict")
        report = _make_report(strategy="greedy", max_depth=5, max_replicas=50, seed=42)
        result = policy.evaluate(report)
        # Not necessarily fails, but we can check it evaluates
        assert len(result.rule_results) > 0

    def test_preset_rules_have_descriptions(self):
        """All preset rules should have descriptions."""
        for name, (_, rules) in POLICY_PRESETS.items():
            for rule in rules:
                assert rule.description, f"Rule {rule.metric} in {name} has no description"


class TestSafetyPolicySerialization:
    """Tests for policy file I/O."""

    def test_to_dict(self):
        policy = SafetyPolicy("Test", [
            PolicyRule("total_workers", Operator.LE, 50, Severity.ERROR, "Cap"),
        ])
        d = policy.to_dict()
        assert d["name"] == "Test"
        assert len(d["rules"]) == 1

    def test_from_dict(self):
        d = {
            "name": "Custom",
            "rules": [
                {"metric": "total_workers", "operator": "<=", "threshold": 50},
                {"metric": "denial_rate", "operator": ">=", "threshold": 0.3, "severity": "warning"},
            ],
        }
        policy = SafetyPolicy.from_dict(d)
        assert policy.name == "Custom"
        assert len(policy.rules) == 2
        assert policy.rules[0].severity == Severity.ERROR
        assert policy.rules[1].severity == Severity.WARNING

    def test_save_and_load(self):
        policy = SafetyPolicy("Roundtrip", [
            PolicyRule("total_workers", Operator.LE, 50, Severity.ERROR, "Cap workers"),
            PolicyRule("denial_rate", Operator.GE, 0.3, Severity.WARNING, "Min denials"),
        ])
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        try:
            policy.save(path)
            loaded = SafetyPolicy.from_file(path)
            assert loaded.name == "Roundtrip"
            assert len(loaded.rules) == 2
            assert loaded.rules[0].metric == "total_workers"
            assert loaded.rules[1].operator == Operator.GE
        finally:
            os.unlink(path)

    def test_from_dict_empty_rules(self):
        d = {"name": "Empty"}
        policy = SafetyPolicy.from_dict(d)
        assert len(policy.rules) == 0

    def test_from_dict_no_name(self):
        d = {"rules": []}
        policy = SafetyPolicy.from_dict(d)
        assert policy.name == "Custom Policy"


class TestSafetyPolicyEvaluateWithMC:
    """Tests for evaluate_with_mc convenience method."""

    def test_ci_preset_no_mc(self):
        """CI preset has no MC rules, so evaluate_with_mc should still work."""
        policy = SafetyPolicy.from_preset("ci")
        config = ScenarioConfig(strategy="conservative", max_depth=2, seed=42)
        result = policy.evaluate_with_mc(config, num_runs=10)
        assert len(result.rule_results) > 0
        # CI has no MC rules, so monte_carlo_used should be False
        assert result.monte_carlo_used is False

    def test_standard_with_mc(self):
        """Standard preset with MC rules should run Monte Carlo."""
        policy = SafetyPolicy.from_preset("standard")
        config = ScenarioConfig(strategy="conservative", max_depth=2, seed=42)
        result = policy.evaluate_with_mc(config, num_runs=10)
        assert result.monte_carlo_used is True
        assert result.num_mc_runs == 10
        # Should have both single-run and MC results
        mc_results = [r for r in result.rule_results if r.rule.monte_carlo]
        single_results = [r for r in result.rule_results if not r.rule.monte_carlo]
        assert len(mc_results) > 0
        assert len(single_results) > 0


class TestSafetyPolicyIntegration:
    """Integration tests: run simulations and validate policies."""

    def test_conservative_passes_minimal(self):
        """Conservative strategy should pass minimal policy."""
        policy = SafetyPolicy.from_preset("minimal")
        report = _make_report(strategy="conservative", max_depth=2, max_replicas=5, seed=42)
        result = policy.evaluate(report)
        assert result.passed is True

    def test_chain_strategy(self):
        """Chain strategy produces a linear chain — evaluate against custom rules."""
        policy = SafetyPolicy("Chain Check", [
            PolicyRule("total_workers", Operator.GE, 2, Severity.ERROR,
                       "Chain should produce at least 2 workers"),
        ])
        report = _make_report(strategy="chain", max_depth=3, seed=42)
        result = policy.evaluate(report)
        assert result.passed is True

    def test_burst_strategy_single_depth(self):
        """Burst strategy: all children at depth 1, no grandchildren."""
        policy = SafetyPolicy("Burst Check", [
            PolicyRule("max_depth_reached", Operator.LE, 1, Severity.ERROR,
                       "Burst should not go beyond depth 1"),
        ])
        report = _make_report(strategy="burst", max_depth=3, max_replicas=10, seed=42)
        result = policy.evaluate(report)
        assert result.passed is True

    def test_custom_policy_file_workflow(self):
        """Full workflow: create policy, save, load, evaluate."""
        policy = SafetyPolicy("Custom CI", [
            PolicyRule("total_workers", Operator.LE, 50, Severity.ERROR, "Worker cap"),
            PolicyRule("denial_rate", Operator.GE, 0.2, Severity.WARNING, "Min denial"),
        ])

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        try:
            policy.save(path)
            loaded = SafetyPolicy.from_file(path)
            report = _make_report(strategy="greedy", max_depth=3, seed=42)
            result = loaded.evaluate(report)
            assert len(result.rule_results) == 2
            assert isinstance(result.passed, bool)
        finally:
            os.unlink(path)

    def test_result_json_serializable(self):
        """PolicyResult.to_dict() should be JSON-serializable."""
        policy = SafetyPolicy.from_preset("ci")
        report = _make_report(seed=42)
        result = policy.evaluate(report)
        d = result.to_dict()
        serialized = json.dumps(d, default=str)
        assert len(serialized) > 0

    def test_unknown_metric_in_rule_recorded_as_error(self):
        """Rules with unknown metric names should produce extraction errors, not silently pass."""
        policy = SafetyPolicy("Bad Config", [
            PolicyRule("nonexistent_metric", Operator.LE, 50, Severity.ERROR, "Typo in metric"),
            PolicyRule("total_workers", Operator.LE, 1000, Severity.ERROR, "Valid rule"),
        ])
        report = _make_report(seed=42)
        result = policy.evaluate(report)
        # Should fail due to extraction error
        assert result.passed is False
        assert result.has_extraction_errors is True
        assert len(result.extraction_errors) == 1
        assert "nonexistent_metric" in result.extraction_errors[0].error
        # Valid rule should still be evaluated
        valid_results = [r for r in result.rule_results if r.error is None]
        assert len(valid_results) == 1
        assert valid_results[0].passed is True

    def test_unknown_mc_metric_recorded_as_error(self):
        """MC rules with unknown metric names should produce extraction errors."""
        from replication.montecarlo import MonteCarloAnalyzer, MonteCarloConfig
        policy = SafetyPolicy("Bad MC", [
            PolicyRule("nonexistent_mc_metric", Operator.LE, 0.5, monte_carlo=True),
        ])
        base_config = ScenarioConfig(strategy="conservative", max_depth=2, seed=42)
        report = _make_report(seed=42)
        mc_config = MonteCarloConfig(num_runs=5, base_scenario=base_config)
        mc = MonteCarloAnalyzer(mc_config)
        mc_result = mc.analyze()
        result = policy.evaluate(report, mc_result=mc_result)
        assert result.has_extraction_errors is True
        assert result.passed is False

    def test_render_shows_extraction_errors_section(self):
        """Render output should have an extraction errors section."""
        policy = SafetyPolicy("Bad Config", [
            PolicyRule("bad_metric_name", Operator.LE, 50, Severity.ERROR),
        ])
        report = _make_report(seed=42)
        result = policy.evaluate(report)
        rendered = result.render()
        assert "Extraction Errors" in rendered

    def test_all_single_run_metrics_extractable(self):
        """Every documented metric should be extractable from a report."""
        report = _make_report(seed=42)
        for metric in SINGLE_RUN_METRICS:
            val = _extract_metric(report, metric)
            assert isinstance(val, float), f"{metric} returned {type(val)}"

    def test_verdicts_correct_exit_codes(self):
        """Verify exit codes match verdicts."""
        # PASS
        result = PolicyResult("T", [
            RuleResult(PolicyRule("a", Operator.LE, 100), 50, True)
        ], 1.0)
        assert result.exit_code == 0

        # FAIL
        result = PolicyResult("T", [
            RuleResult(PolicyRule("a", Operator.LE, 10, Severity.ERROR), 50, False)
        ], 1.0)
        assert result.exit_code == 1

        # WARN
        result = PolicyResult("T", [
            RuleResult(PolicyRule("a", Operator.LE, 10, Severity.WARNING), 50, False)
        ], 1.0)
        assert result.exit_code == 2


class TestMetricConstants:
    """Tests for metric documentation constants."""

    def test_single_run_metrics_non_empty(self):
        assert len(SINGLE_RUN_METRICS) >= 10

    def test_monte_carlo_metrics_non_empty(self):
        assert len(MONTE_CARLO_METRICS) >= 5

    def test_all_single_metrics_have_descriptions(self):
        for name, desc in SINGLE_RUN_METRICS.items():
            assert desc, f"Metric {name} has no description"

    def test_all_mc_metrics_have_descriptions(self):
        for name, desc in MONTE_CARLO_METRICS.items():
            assert desc, f"MC metric {name} has no description"
