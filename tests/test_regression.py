"""Tests for regression detector."""

import json
import pytest
from unittest.mock import MagicMock
from replication.regression import (
    RegressionDetector, RegressionConfig, RegressionResult,
    MetricChange, ChangeDirection, MetricPolarity,
    METRIC_DEFINITIONS, _format_value, main,
)
from replication.simulator import (
    Simulator, ScenarioConfig, SimulationReport, WorkerRecord, PRESETS,
)
from replication.montecarlo import (
    MonteCarloAnalyzer, MonteCarloConfig, MonteCarloResult,
    MetricDistribution,
)


# ── Helpers ──

def _make_report(workers=3, tasks=6, repl_ok=2, repl_denied=1,
                 max_depth=2, duration=100.0) -> SimulationReport:
    """Create a minimal SimulationReport for testing."""
    worker_records = {}
    for i in range(workers):
        wid = f"w-{i}"
        worker_records[wid] = WorkerRecord(
            worker_id=wid,
            parent_id=None if i == 0 else "w-0",
            depth=min(i, max_depth),
            tasks_completed=tasks // max(workers, 1),
        )
    return SimulationReport(
        config=ScenarioConfig(),
        workers=worker_records,
        root_id="w-0",
        timeline=[],
        total_tasks=tasks,
        total_replications_attempted=repl_ok + repl_denied,
        total_replications_succeeded=repl_ok,
        total_replications_denied=repl_denied,
        duration_ms=duration,
        audit_events=[],
    )


def _make_mc_result(**metric_means) -> MonteCarloResult:
    """Create a MonteCarloResult with specified metric means."""
    distributions = {}
    defaults = {
        "peak_workers": 5.0,
        "max_depth_reached": 2.0,
        "total_tasks": 10.0,
        "replication_success_rate": 50.0,
        "denial_rate": 50.0,
        "efficiency": 2.0,
        "total_replications": 3.0,
        "duration_ms": 100.0,
    }
    for key, default_val in defaults.items():
        val = metric_means.get(key, default_val)
        # MetricDistribution(name, unit, values) — properties compute stats
        distributions[key] = MetricDistribution(
            name=key,
            unit="count",
            values=[val] * 10,
        )
    result = MagicMock(spec=MonteCarloResult)
    result.distributions = distributions
    return result


# ── ChangeDirection & MetricPolarity enums ──

class TestChangeDirectionEnum:
    def test_three_values(self):
        assert len(ChangeDirection) == 3

    def test_values_are_lowercase_strings(self):
        for member in ChangeDirection:
            assert member.value == member.value.lower()
            assert isinstance(member.value, str)

    def test_all_unique(self):
        values = [m.value for m in ChangeDirection]
        assert len(values) == len(set(values))


class TestMetricPolarityEnum:
    def test_two_values(self):
        assert len(MetricPolarity) == 2

    def test_values_are_lowercase_strings(self):
        for member in MetricPolarity:
            assert member.value == member.value.lower()
            assert isinstance(member.value, str)


# ── METRIC_DEFINITIONS ──

class TestMetricDefinitions:
    def test_contains_eight_metrics(self):
        assert len(METRIC_DEFINITIONS) == 8

    def test_all_have_four_tuple_entries(self):
        for key, defn in METRIC_DEFINITIONS.items():
            assert len(defn) == 4, f"{key} should have 4-tuple"
            extractor, polarity, display_name, fmt = defn
            assert isinstance(extractor, str)
            assert isinstance(polarity, MetricPolarity)
            assert isinstance(display_name, str)
            assert isinstance(fmt, str)

    def test_extractor_names_start_with_underscore(self):
        for key, defn in METRIC_DEFINITIONS.items():
            assert defn[0].startswith("_"), f"{key} extractor should start with _"


# ── MetricChange ──

class TestMetricChange:
    def _make_change(self, **overrides):
        defaults = dict(
            metric="peak_workers",
            display_name="Peak Workers",
            baseline_value=5.0,
            candidate_value=8.0,
            absolute_change=3.0,
            percent_change=60.0,
            direction=ChangeDirection.REGRESSION,
            polarity=MetricPolarity.LOWER_IS_BETTER,
            exceeds_threshold=True,
        )
        defaults.update(overrides)
        return MetricChange(**defaults)

    def test_construction(self):
        mc = self._make_change()
        assert mc.metric == "peak_workers"
        assert mc.baseline_value == 5.0
        assert mc.candidate_value == 8.0
        assert mc.direction == ChangeDirection.REGRESSION

    def test_to_dict_keys(self):
        d = self._make_change().to_dict()
        expected_keys = {"metric", "display_name", "baseline", "candidate",
                         "absolute_change", "percent_change", "direction",
                         "polarity", "exceeds_threshold"}
        assert set(d.keys()) == expected_keys

    def test_to_dict_rounds_values(self):
        mc = self._make_change(baseline_value=5.123456, candidate_value=8.654321,
                               absolute_change=3.530865, percent_change=68.9123)
        d = mc.to_dict()
        assert d["baseline"] == round(5.123456, 4)
        assert d["candidate"] == round(8.654321, 4)
        assert d["absolute_change"] == round(3.530865, 4)
        assert d["percent_change"] == round(68.9123, 2)

    def test_percent_change_sign(self):
        mc_pos = self._make_change(percent_change=25.0)
        assert mc_pos.percent_change > 0
        mc_neg = self._make_change(percent_change=-15.0)
        assert mc_neg.percent_change < 0


# ── RegressionConfig ──

class TestRegressionConfig:
    def test_defaults(self):
        cfg = RegressionConfig()
        assert cfg.threshold_percent == 5.0
        assert cfg.strict_mode is False
        assert cfg.metrics is None
        assert cfg.num_runs == 1
        assert cfg.seed is None

    def test_custom_threshold(self):
        cfg = RegressionConfig(threshold_percent=10.0)
        assert cfg.threshold_percent == 10.0

    def test_custom_metrics_list(self):
        cfg = RegressionConfig(metrics=["peak_workers", "max_depth"])
        assert cfg.metrics == ["peak_workers", "max_depth"]


# ── RegressionDetector construction ──

class TestRegressionDetectorConstruction:
    def test_default_config(self):
        d = RegressionDetector()
        assert d.config.threshold_percent == 5.0
        assert d.config.strict_mode is False

    def test_custom_config(self):
        cfg = RegressionConfig(threshold_percent=15.0, strict_mode=True)
        d = RegressionDetector(cfg)
        assert d.config.threshold_percent == 15.0
        assert d.config.strict_mode is True


# ── _classify_change ──

class TestClassifyChange:
    def test_lower_is_better_increase_regression(self):
        result = RegressionDetector._classify_change(5.0, 50.0, MetricPolarity.LOWER_IS_BETTER)
        assert result == ChangeDirection.REGRESSION

    def test_lower_is_better_decrease_improvement(self):
        result = RegressionDetector._classify_change(-3.0, -30.0, MetricPolarity.LOWER_IS_BETTER)
        assert result == ChangeDirection.IMPROVEMENT

    def test_lower_is_better_zero_neutral(self):
        result = RegressionDetector._classify_change(0.0, 0.0, MetricPolarity.LOWER_IS_BETTER)
        assert result == ChangeDirection.NEUTRAL

    def test_higher_is_better_decrease_regression(self):
        result = RegressionDetector._classify_change(-5.0, -50.0, MetricPolarity.HIGHER_IS_BETTER)
        assert result == ChangeDirection.REGRESSION

    def test_higher_is_better_increase_improvement(self):
        result = RegressionDetector._classify_change(3.0, 30.0, MetricPolarity.HIGHER_IS_BETTER)
        assert result == ChangeDirection.IMPROVEMENT

    def test_higher_is_better_zero_neutral(self):
        result = RegressionDetector._classify_change(0.0, 0.0, MetricPolarity.HIGHER_IS_BETTER)
        assert result == ChangeDirection.NEUTRAL


# ── Metric extractors ──

class TestExtractors:
    def test_peak_workers(self):
        report = _make_report(workers=5)
        assert RegressionDetector._extract_peak_workers(report) == 5.0

    def test_max_depth(self):
        report = _make_report(workers=4, max_depth=3)
        assert RegressionDetector._extract_max_depth(report) == 3.0

    def test_total_tasks(self):
        report = _make_report(tasks=12)
        assert RegressionDetector._extract_total_tasks(report) == 12.0

    def test_repl_success_rate(self):
        report = _make_report(repl_ok=3, repl_denied=7)
        assert RegressionDetector._extract_repl_success_rate(report) == pytest.approx(30.0)

    def test_repl_success_rate_zero_attempts(self):
        report = _make_report(repl_ok=0, repl_denied=0)
        assert RegressionDetector._extract_repl_success_rate(report) == 0.0

    def test_denial_rate(self):
        report = _make_report(repl_ok=3, repl_denied=7)
        assert RegressionDetector._extract_denial_rate(report) == pytest.approx(70.0)

    def test_efficiency(self):
        report = _make_report(workers=4, tasks=12)
        assert RegressionDetector._extract_efficiency(report) == pytest.approx(3.0)

    def test_efficiency_zero_workers(self):
        report = _make_report(workers=0, tasks=0)
        assert RegressionDetector._extract_efficiency(report) == 0.0

    def test_total_replications(self):
        report = _make_report(repl_ok=5, repl_denied=3)
        assert RegressionDetector._extract_total_replications(report) == 5.0

    def test_duration(self):
        report = _make_report(duration=42.5)
        assert RegressionDetector._extract_duration(report) == 42.5


# ── RegressionDetector.compare ──

class TestCompare:
    def test_identical_reports_all_neutral(self):
        report = _make_report()
        detector = RegressionDetector()
        result = detector.compare(report, report)
        assert result.passed is True
        assert result.regression_count == 0
        assert all(c.direction == ChangeDirection.NEUTRAL for c in result.changes)

    def test_worse_candidate_has_regressions(self):
        baseline = _make_report(workers=3, tasks=6, repl_ok=1, repl_denied=2)
        candidate = _make_report(workers=8, tasks=3, repl_ok=5, repl_denied=0)
        detector = RegressionDetector()
        result = detector.compare(baseline, candidate)
        assert result.has_regressions

    def test_better_candidate_has_improvements(self):
        baseline = _make_report(workers=8, tasks=3, repl_ok=5, repl_denied=0)
        candidate = _make_report(workers=3, tasks=6, repl_ok=1, repl_denied=2)
        detector = RegressionDetector()
        result = detector.compare(baseline, candidate)
        assert result.improvement_count > 0

    def test_mixed_changes(self):
        baseline = _make_report(workers=3, tasks=6, repl_ok=2, repl_denied=1)
        # More workers (regression for lower-is-better), more tasks (improvement for higher-is-better)
        candidate = _make_report(workers=6, tasks=12, repl_ok=2, repl_denied=1)
        detector = RegressionDetector()
        result = detector.compare(baseline, candidate)
        assert result.regression_count > 0
        assert result.improvement_count > 0

    def test_custom_metrics_filter(self):
        baseline = _make_report()
        candidate = _make_report(workers=10)
        cfg = RegressionConfig(metrics=["peak_workers"])
        detector = RegressionDetector(cfg)
        result = detector.compare(baseline, candidate)
        assert len(result.changes) == 1
        assert result.changes[0].metric == "peak_workers"

    def test_small_change_below_threshold_passes(self):
        baseline = _make_report(workers=100, tasks=6)
        candidate = _make_report(workers=101, tasks=6)  # 1% increase
        cfg = RegressionConfig(threshold_percent=5.0, metrics=["peak_workers"])
        detector = RegressionDetector(cfg)
        result = detector.compare(baseline, candidate)
        assert result.passed is True
        assert result.has_regressions  # still a regression, just not significant


# ── RegressionResult properties ──

class TestRegressionResult:
    def _make_changes(self):
        """Create a set of changes with known directions."""
        return [
            MetricChange("m1", "Metric 1", 5.0, 8.0, 3.0, 60.0,
                          ChangeDirection.REGRESSION, MetricPolarity.LOWER_IS_BETTER, True),
            MetricChange("m2", "Metric 2", 10.0, 12.0, 2.0, 20.0,
                          ChangeDirection.REGRESSION, MetricPolarity.LOWER_IS_BETTER, True),
            MetricChange("m3", "Metric 3", 5.0, 3.0, -2.0, -40.0,
                          ChangeDirection.IMPROVEMENT, MetricPolarity.LOWER_IS_BETTER, False),
            MetricChange("m4", "Metric 4", 5.0, 5.0, 0.0, 0.0,
                          ChangeDirection.NEUTRAL, MetricPolarity.LOWER_IS_BETTER, False),
            MetricChange("m5", "Metric 5", 10.0, 10.5, 0.5, 5.0,
                          ChangeDirection.REGRESSION, MetricPolarity.LOWER_IS_BETTER, False),
        ]

    def test_regressions_property(self):
        result = RegressionResult("b", "c", self._make_changes())
        assert len(result.regressions) == 3

    def test_improvements_property(self):
        result = RegressionResult("b", "c", self._make_changes())
        assert len(result.improvements) == 1

    def test_neutral_property(self):
        result = RegressionResult("b", "c", self._make_changes())
        assert len(result.neutral) == 1

    def test_regression_count(self):
        result = RegressionResult("b", "c", self._make_changes())
        assert result.regression_count == 3

    def test_improvement_count(self):
        result = RegressionResult("b", "c", self._make_changes())
        assert result.improvement_count == 1

    def test_has_regressions_true(self):
        result = RegressionResult("b", "c", self._make_changes())
        assert result.has_regressions is True

    def test_has_regressions_false(self):
        changes = [MetricChange("m1", "M1", 5.0, 5.0, 0.0, 0.0,
                                 ChangeDirection.NEUTRAL, MetricPolarity.LOWER_IS_BETTER, False)]
        result = RegressionResult("b", "c", changes)
        assert result.has_regressions is False

    def test_significant_regressions(self):
        result = RegressionResult("b", "c", self._make_changes())
        sig = result.significant_regressions
        assert len(sig) == 2  # m1 and m2 have exceeds_threshold=True

    def test_passed_true_no_significant(self):
        changes = [
            MetricChange("m1", "M1", 10.0, 10.5, 0.5, 5.0,
                          ChangeDirection.REGRESSION, MetricPolarity.LOWER_IS_BETTER, False),
        ]
        result = RegressionResult("b", "c", changes)
        assert result.passed is True

    def test_passed_false_significant(self):
        result = RegressionResult("b", "c", self._make_changes())
        assert result.passed is False

    def test_strict_mode_any_regression_fails(self):
        changes = [
            MetricChange("m1", "M1", 10.0, 10.5, 0.5, 5.0,
                          ChangeDirection.REGRESSION, MetricPolarity.LOWER_IS_BETTER, False),
        ]
        result = RegressionResult("b", "c", changes, strict_mode=True)
        assert result.passed is False

    def test_verdict_improved(self):
        changes = [MetricChange("m1", "M1", 10.0, 5.0, -5.0, -50.0,
                                 ChangeDirection.IMPROVEMENT, MetricPolarity.LOWER_IS_BETTER, False)]
        result = RegressionResult("b", "c", changes)
        assert result.verdict == "IMPROVED"

    def test_verdict_unchanged(self):
        changes = [MetricChange("m1", "M1", 5.0, 5.0, 0.0, 0.0,
                                 ChangeDirection.NEUTRAL, MetricPolarity.LOWER_IS_BETTER, False)]
        result = RegressionResult("b", "c", changes)
        assert result.verdict == "UNCHANGED"

    def test_verdict_minor_regression(self):
        changes = [MetricChange("m1", "M1", 10.0, 10.5, 0.5, 5.0,
                                 ChangeDirection.REGRESSION, MetricPolarity.LOWER_IS_BETTER, False)]
        result = RegressionResult("b", "c", changes)
        assert result.verdict == "MINOR_REGRESSION"

    def test_verdict_regression(self):
        result = RegressionResult("b", "c", self._make_changes())
        assert result.verdict == "REGRESSION"

    def test_summary_contains_verdict(self):
        result = RegressionResult("b", "c", self._make_changes())
        assert "Verdict:" in result.summary
        assert "REGRESSION" in result.summary

    def test_summary_contains_counts(self):
        result = RegressionResult("b", "c", self._make_changes())
        assert "3 regression(s)" in result.summary
        assert "1 improvement(s)" in result.summary

    def test_render_nonempty_with_box(self):
        result = RegressionResult("b", "c", self._make_changes())
        rendered = result.render()
        assert len(rendered) > 0
        assert "┌" in rendered
        assert "Regression Analysis Report" in rendered

    def test_render_includes_metric_table(self):
        result = RegressionResult("b", "c", self._make_changes())
        rendered = result.render()
        assert "Metric Comparison:" in rendered
        assert "─" * 76 in rendered

    def test_to_dict_serialization(self):
        result = RegressionResult("baseline", "candidate", self._make_changes(),
                                  threshold_percent=5.0, strict_mode=False, duration_ms=42.0)
        d = result.to_dict()
        assert d["baseline"] == "baseline"
        assert d["candidate"] == "candidate"
        assert d["threshold_percent"] == 5.0
        assert d["strict_mode"] is False
        assert isinstance(d["passed"], bool)
        assert isinstance(d["verdict"], str)
        assert isinstance(d["changes"], list)
        assert d["duration_ms"] == 42.0


# ── compare_presets ──

class TestComparePresets:
    def test_valid_presets(self):
        detector = RegressionDetector()
        result = detector.compare_presets("minimal", "balanced")
        assert isinstance(result, RegressionResult)
        assert result.baseline_label == "minimal"
        assert result.candidate_label == "balanced"

    def test_unknown_baseline_raises(self):
        detector = RegressionDetector()
        with pytest.raises(ValueError, match="Unknown preset"):
            detector.compare_presets("nonexistent", "balanced")

    def test_unknown_candidate_raises(self):
        detector = RegressionDetector()
        with pytest.raises(ValueError, match="Unknown preset"):
            detector.compare_presets("balanced", "nonexistent")


# ── compare_configs ──

class TestCompareConfigs:
    def test_produces_result_with_labels(self):
        cfg1 = ScenarioConfig(max_depth=1, max_replicas=3, strategy="conservative")
        cfg2 = ScenarioConfig(max_depth=1, max_replicas=3, strategy="conservative")
        detector = RegressionDetector()
        result = detector.compare_configs(cfg1, cfg2, "cfg1", "cfg2")
        assert result.baseline_label == "cfg1"
        assert result.candidate_label == "cfg2"

    def test_regression_detected_when_worse(self):
        cfg1 = ScenarioConfig(max_depth=1, max_replicas=3, strategy="conservative", seed=42)
        cfg2 = ScenarioConfig(max_depth=5, max_replicas=50, strategy="greedy", seed=42)
        detector = RegressionDetector()
        result = detector.compare_configs(cfg1, cfg2)
        # stress config should have more workers = regression for peak_workers
        assert result.has_regressions or result.improvement_count > 0  # at least some change


# ── compare_monte_carlo ──

class TestCompareMonteCarlo:
    def test_compares_mean_values(self):
        baseline_mc = _make_mc_result(peak_workers=5.0, total_tasks=10.0)
        candidate_mc = _make_mc_result(peak_workers=5.0, total_tasks=10.0)
        detector = RegressionDetector()
        result = detector.compare_monte_carlo(baseline_mc, candidate_mc)
        assert result.regression_count == 0

    def test_detects_regression_when_worse(self):
        baseline_mc = _make_mc_result(peak_workers=5.0)
        candidate_mc = _make_mc_result(peak_workers=15.0)  # much worse
        cfg = RegressionConfig(metrics=["peak_workers"])
        detector = RegressionDetector(cfg)
        result = detector.compare_monte_carlo(baseline_mc, candidate_mc)
        assert result.has_regressions
        assert result.regressions[0].metric == "peak_workers"

    def test_handles_missing_metric_keys(self):
        baseline_mc = _make_mc_result()
        candidate_mc = _make_mc_result()
        # Remove a distribution key
        del candidate_mc.distributions["peak_workers"]
        cfg = RegressionConfig(metrics=["peak_workers"])
        detector = RegressionDetector(cfg)
        result = detector.compare_monte_carlo(baseline_mc, candidate_mc)
        # Should skip the missing metric, no crash
        assert len(result.changes) == 0


# ── _format_value ──

class TestFormatValue:
    def test_known_metric(self):
        # peak_workers has format "d"
        formatted = _format_value(5.0, "peak_workers")
        assert formatted == "5"

    def test_unknown_metric_default(self):
        formatted = _format_value(3.14159, "unknown_metric")
        assert formatted == "3.14"


# ── CLI main ──

class TestMainCLI:
    def test_default_args_return_code(self):
        # With seed for determinism
        code = main(["--baseline", "minimal", "--candidate", "minimal", "--seed", "42"])
        assert code in (0, 1)

    def test_json_output(self, capsys):
        code = main(["--baseline", "minimal", "--candidate", "minimal",
                      "--seed", "42", "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "verdict" in data
        assert "changes" in data

    def test_summary_only_one_line(self, capsys):
        main(["--baseline", "minimal", "--candidate", "minimal",
              "--seed", "42", "--summary-only"])
        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        assert len(lines) == 1
        assert lines[0].startswith("[PASS]") or lines[0].startswith("[FAIL]")

    def test_unknown_preset_returns_1(self):
        code = main(["--baseline", "nonexistent_preset"])
        assert code == 1

    def test_unknown_candidate_preset_returns_1(self):
        code = main(["--candidate", "nonexistent_preset"])
        assert code == 1

    def test_strict_mode_cli(self, capsys):
        code = main(["--baseline", "minimal", "--candidate", "minimal",
                      "--seed", "42", "--strict", "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["strict_mode"] is True

    def test_custom_threshold_cli(self, capsys):
        code = main(["--baseline", "minimal", "--candidate", "minimal",
                      "--seed", "42", "--threshold", "20.0", "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["threshold_percent"] == 20.0

    def test_metrics_filter_cli(self, capsys):
        code = main(["--baseline", "minimal", "--candidate", "balanced",
                      "--seed", "42", "--metrics", "peak_workers", "max_depth", "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        metric_names = [c["metric"] for c in data["changes"]]
        assert set(metric_names) <= {"peak_workers", "max_depth"}
