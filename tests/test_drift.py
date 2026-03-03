"""Comprehensive tests for drift detection module.

Covers: DriftConfig, DriftDetector, DriftResult, MetricTrend, DriftAlert,
helper functions (_linear_regression, _longest_monotonic_run, _sparkline,
_extract_metrics), enums, serialization, and CLI.
"""

import json
import math
import pytest
from unittest.mock import patch, MagicMock
from replication.drift import (
    DriftDetector,
    DriftConfig,
    DriftDirection,
    DriftSeverity,
    DriftResult,
    DriftAlert,
    MetricTrend,
    MetricWindow,
    _linear_regression,
    _longest_monotonic_run,
    _sparkline,
    _extract_metrics,
    main,
)


# ── Enum tests ───────────────────────────────────────


class TestDriftDirection:
    def test_values(self):
        assert DriftDirection.WORSENING.value == "worsening"
        assert DriftDirection.IMPROVING.value == "improving"
        assert DriftDirection.STABLE.value == "stable"

    def test_all_members(self):
        assert len(DriftDirection) == 3


class TestDriftSeverity:
    def test_values(self):
        assert DriftSeverity.NONE.value == "none"
        assert DriftSeverity.LOW.value == "low"
        assert DriftSeverity.MEDIUM.value == "medium"
        assert DriftSeverity.HIGH.value == "high"
        assert DriftSeverity.CRITICAL.value == "critical"

    def test_all_members(self):
        assert len(DriftSeverity) == 5


# ── _linear_regression ───────────────────────────────


class TestLinearRegression:
    def test_flat(self):
        slope, intercept, r2 = _linear_regression([5.0, 5.0, 5.0, 5.0])
        assert slope == 0.0
        assert r2 == 0.0

    def test_perfect_positive(self):
        slope, intercept, r2 = _linear_regression([1.0, 2.0, 3.0, 4.0])
        assert abs(slope - 1.0) < 1e-9
        assert abs(r2 - 1.0) < 1e-9
        assert abs(intercept - 1.0) < 1e-9

    def test_perfect_negative(self):
        slope, intercept, r2 = _linear_regression([4.0, 3.0, 2.0, 1.0])
        assert abs(slope - (-1.0)) < 1e-9
        assert abs(r2 - 1.0) < 1e-9

    def test_single_value(self):
        slope, intercept, r2 = _linear_regression([7.0])
        assert slope == 0.0
        assert intercept == 7.0
        assert r2 == 0.0

    def test_two_values(self):
        slope, intercept, r2 = _linear_regression([0.0, 10.0])
        assert abs(slope - 10.0) < 1e-9
        assert abs(r2 - 1.0) < 1e-9

    def test_noisy_data(self):
        slope, intercept, r2 = _linear_regression([1.0, 3.0, 2.0, 4.0, 3.0])
        assert slope > 0  # upward trend despite noise
        assert 0 < r2 < 1.0  # imperfect fit

    def test_empty_list(self):
        slope, intercept, r2 = _linear_regression([])
        assert slope == 0.0
        assert intercept == 0.0
        assert r2 == 0.0

    def test_large_values(self):
        vals = [float(i * 1000) for i in range(10)]
        slope, _, r2 = _linear_regression(vals)
        assert abs(slope - 1000.0) < 1e-6
        assert abs(r2 - 1.0) < 1e-9

    def test_negative_slope(self):
        vals = [10.0, 8.0, 6.0, 4.0, 2.0, 0.0]
        slope, _, r2 = _linear_regression(vals)
        assert abs(slope - (-2.0)) < 1e-9
        assert abs(r2 - 1.0) < 1e-9


# ── _longest_monotonic_run ───────────────────────────


class TestLongestMonotonicRun:
    def test_increasing(self):
        assert _longest_monotonic_run([1, 2, 3, 2, 1]) == 3

    def test_decreasing(self):
        assert _longest_monotonic_run([5, 4, 3, 2, 6]) == 4

    def test_all_equal(self):
        # Equal values count as both increasing AND decreasing
        assert _longest_monotonic_run([3, 3, 3, 3]) == 4

    def test_single_element(self):
        assert _longest_monotonic_run([42]) == 1

    def test_two_increasing(self):
        assert _longest_monotonic_run([1, 2]) == 2

    def test_two_decreasing(self):
        assert _longest_monotonic_run([5, 3]) == 2

    def test_alternating(self):
        assert _longest_monotonic_run([1, 5, 1, 5, 1]) == 2

    def test_long_increase_at_end(self):
        assert _longest_monotonic_run([10, 1, 2, 3, 4, 5]) == 5

    def test_empty(self):
        assert _longest_monotonic_run([]) == 0


# ── _sparkline ───────────────────────────────────────


class TestSparkline:
    def test_basic(self):
        result = _sparkline([0, 0.5, 1.0])
        assert len(result) == 3
        assert result[0] == "▁"
        assert result[-1] == "█"

    def test_all_same(self):
        result = _sparkline([5.0, 5.0, 5.0])
        assert len(result) == 3
        # When all same, spread=1.0 (fallback), all map to same char
        assert len(set(result)) == 1

    def test_empty(self):
        assert _sparkline([]) == ""

    def test_single_value(self):
        result = _sparkline([3.14])
        assert len(result) == 1

    def test_ascending(self):
        result = _sparkline([0, 1, 2, 3, 4, 5, 6, 7])
        # Each char should be >= previous
        for i in range(1, len(result)):
            assert result[i] >= result[i - 1]

    def test_descending(self):
        result = _sparkline([7, 6, 5, 4, 3, 2, 1, 0])
        for i in range(1, len(result)):
            assert result[i] <= result[i - 1]


# ── DriftConfig ──────────────────────────────────────


class TestDriftConfig:
    def test_defaults(self):
        config = DriftConfig()
        assert config.windows == 10
        assert config.strategy == "greedy"
        assert config.max_depth == 3
        assert config.max_replicas == 10
        assert config.seed_base == 42
        assert config.sensitivity == 0.05
        assert config.r_squared_threshold == 0.3

    def test_custom_values(self):
        config = DriftConfig(
            windows=20,
            strategy="conservative",
            sensitivity=0.1,
            seed_base=None,
        )
        assert config.windows == 20
        assert config.strategy == "conservative"
        assert config.sensitivity == 0.1
        assert config.seed_base is None

    def test_worse_when_higher_defaults(self):
        config = DriftConfig()
        assert "denial_rate" in config.worse_when_higher
        assert "max_depth_used" in config.worse_when_higher
        assert "replication_ratio" in config.worse_when_higher

    def test_worse_when_lower_defaults(self):
        config = DriftConfig()
        assert "kill_rate" in config.worse_when_lower
        assert "tasks_per_worker" in config.worse_when_lower


# ── MetricWindow ─────────────────────────────────────


class TestMetricWindow:
    def test_creation(self):
        mw = MetricWindow(window_index=3, value=0.75)
        assert mw.window_index == 3
        assert mw.value == 0.75
        assert mw.raw_report is None

    def test_with_report(self):
        mock_report = MagicMock()
        mw = MetricWindow(window_index=0, value=1.0, raw_report=mock_report)
        assert mw.raw_report is mock_report


# ── DriftAlert ───────────────────────────────────────


class TestDriftAlert:
    def test_creation(self):
        alert = DriftAlert(
            metric="denial_rate",
            direction=DriftDirection.WORSENING,
            severity=DriftSeverity.HIGH,
            slope=0.15,
            r_squared=0.92,
            start_value=0.1,
            end_value=0.8,
            change_pct=700.0,
            description="denial_rate drifting worsening",
        )
        assert alert.metric == "denial_rate"
        assert alert.direction == DriftDirection.WORSENING
        assert alert.severity == DriftSeverity.HIGH
        assert alert.change_pct == 700.0


# ── MetricTrend ──────────────────────────────────────


class TestMetricTrend:
    def test_creation(self):
        trend = MetricTrend(
            metric="kill_rate",
            values=[0.5, 0.4, 0.3],
            mean=0.4,
            std=0.1,
            slope=-0.1,
            r_squared=1.0,
            direction=DriftDirection.WORSENING,
            severity=DriftSeverity.MEDIUM,
            monotonic_run=3,
        )
        assert trend.metric == "kill_rate"
        assert len(trend.values) == 3
        assert trend.monotonic_run == 3


# ── DriftDetector ────────────────────────────────────


class TestDriftDetector:
    def test_basic_analysis(self):
        config = DriftConfig(windows=5, seed_base=42)
        detector = DriftDetector()
        result = detector.analyze(config)
        assert result.windows_analyzed == 5
        assert len(result.trends) > 0
        assert result.duration_ms > 0

    def test_default_config(self):
        detector = DriftDetector()
        result = detector.analyze()
        assert result.windows_analyzed == 10  # default
        assert isinstance(result, DriftResult)

    def test_different_strategies(self):
        for strategy in ["greedy", "conservative", "random"]:
            config = DriftConfig(windows=3, strategy=strategy, seed_base=42)
            detector = DriftDetector()
            result = detector.analyze(config)
            assert result.windows_analyzed == 3

    def test_high_sensitivity_all_stable(self):
        config = DriftConfig(windows=5, seed_base=42, sensitivity=100.0)
        detector = DriftDetector()
        result = detector.analyze(config)
        for trend in result.trends:
            assert trend.direction == DriftDirection.STABLE

    def test_low_sensitivity(self):
        config = DriftConfig(windows=5, seed_base=42, sensitivity=0.0001)
        detector = DriftDetector()
        result = detector.analyze(config)
        # With very low sensitivity, more trends might be detected
        assert result.windows_analyzed == 5

    def test_many_windows(self):
        config = DriftConfig(windows=15, seed_base=42)
        detector = DriftDetector()
        result = detector.analyze(config)
        assert result.windows_analyzed == 15
        for trend in result.trends:
            assert len(trend.values) == 15

    def test_single_window(self):
        config = DriftConfig(windows=1, seed_base=42)
        detector = DriftDetector()
        result = detector.analyze(config)
        assert result.windows_analyzed == 1
        # With 1 window, regression gives slope=0 → all stable
        for trend in result.trends:
            assert trend.direction == DriftDirection.STABLE

    def test_none_seed(self):
        config = DriftConfig(windows=3, seed_base=None)
        detector = DriftDetector()
        result = detector.analyze(config)
        assert result.windows_analyzed == 3

    def test_result_has_expected_metrics(self):
        config = DriftConfig(windows=3, seed_base=42)
        detector = DriftDetector()
        result = detector.analyze(config)
        metric_names = {t.metric for t in result.trends}
        assert "worker_count" in metric_names
        assert "denial_rate" in metric_names
        assert "kill_rate" in metric_names
        assert "duration_ms" in metric_names

    def test_trend_statistics(self):
        config = DriftConfig(windows=5, seed_base=42)
        detector = DriftDetector()
        result = detector.analyze(config)
        for trend in result.trends:
            assert trend.std >= 0
            assert 0 <= trend.r_squared <= 1.0
            assert trend.monotonic_run >= 1

    def test_severity_classification(self):
        """severity should be one of the valid enum values."""
        config = DriftConfig(windows=5, seed_base=42)
        detector = DriftDetector()
        result = detector.analyze(config)
        for trend in result.trends:
            assert trend.severity in DriftSeverity

    def test_alerts_only_for_worsening(self):
        config = DriftConfig(windows=5, seed_base=42)
        detector = DriftDetector()
        result = detector.analyze(config)
        for alert in result.alerts:
            assert alert.direction == DriftDirection.WORSENING
            assert alert.severity != DriftSeverity.NONE

    def test_passed_flag_consistency(self):
        config = DriftConfig(windows=5, seed_base=42)
        detector = DriftDetector()
        result = detector.analyze(config)
        # passed should be True only if no MEDIUM+ alerts
        has_medium_plus = any(
            a.severity in (DriftSeverity.MEDIUM, DriftSeverity.HIGH, DriftSeverity.CRITICAL)
            for a in result.alerts
        )
        assert result.passed != has_medium_plus

    def test_high_r_squared_threshold(self):
        config = DriftConfig(windows=5, seed_base=42, r_squared_threshold=0.99)
        detector = DriftDetector()
        result = detector.analyze(config)
        # Very strict R² — most trends should be stable
        stable_count = sum(1 for t in result.trends if t.direction == DriftDirection.STABLE)
        assert stable_count >= len(result.trends) // 2


# ── DriftResult rendering & serialization ────────────


class TestDriftResult:
    @pytest.fixture
    def sample_result(self):
        config = DriftConfig(windows=5, seed_base=42)
        detector = DriftDetector()
        return detector.analyze(config)

    def test_render_contains_header(self, sample_result):
        rendered = sample_result.render()
        assert "Drift Detection Report" in rendered
        assert "Sparklines" in rendered

    def test_render_contains_config(self, sample_result):
        rendered = sample_result.render()
        assert "Windows analyzed" in rendered
        assert "Strategy" in rendered
        assert "Duration" in rendered

    def test_render_status_passed(self):
        config = DriftConfig(windows=3, seed_base=42, sensitivity=100.0)
        detector = DriftDetector()
        result = detector.analyze(config)
        rendered = result.render()
        assert "PASSED" in rendered

    def test_render_table_has_headers(self, sample_result):
        rendered = sample_result.render()
        assert "Metric" in rendered
        assert "Slope" in rendered
        assert "Severity" in rendered
        assert "Direction" in rendered

    def test_render_sparklines(self, sample_result):
        rendered = sample_result.render()
        assert "Sparklines" in rendered

    def test_to_dict(self, sample_result):
        d = sample_result.to_dict()
        assert "passed" in d
        assert "trends" in d
        assert "alerts" in d
        assert "windows_analyzed" in d
        assert "duration_ms" in d
        assert "config" in d

    def test_to_dict_config_fields(self, sample_result):
        d = sample_result.to_dict()
        assert "windows" in d["config"]
        assert "strategy" in d["config"]
        assert "sensitivity" in d["config"]

    def test_to_dict_trends_structure(self, sample_result):
        d = sample_result.to_dict()
        assert isinstance(d["trends"], list)
        if d["trends"]:
            t = d["trends"][0]
            assert "metric" in t
            assert "slope" in t
            assert "r_squared" in t
            assert "direction" in t
            assert "severity" in t
            assert "mean" in t
            assert "std" in t
            assert "values" in t
            assert "monotonic_run" in t

    def test_to_dict_alerts_structure(self, sample_result):
        d = sample_result.to_dict()
        for a in d["alerts"]:
            assert "metric" in a
            assert "direction" in a
            assert "severity" in a
            assert "slope" in a
            assert "r_squared" in a
            assert "change_pct" in a
            assert "description" in a

    def test_to_dict_is_json_serializable(self, sample_result):
        d = sample_result.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)
        parsed = json.loads(serialized)
        assert parsed["windows_analyzed"] == sample_result.windows_analyzed

    def test_render_alerts_section(self):
        """Render should include alert details when alerts exist."""
        config = DriftConfig(windows=5, seed_base=42, sensitivity=0.001)
        detector = DriftDetector()
        result = detector.analyze(config)
        rendered = result.render()
        if result.alerts:
            assert "Alerts" in rendered


# ── _extract_metrics ─────────────────────────────────


class TestExtractMetrics:
    def test_basic_extraction(self):
        """Verify _extract_metrics returns expected keys."""
        from replication.simulator import Simulator, ScenarioConfig
        sc = ScenarioConfig(max_depth=2, max_replicas=5, strategy="greedy", seed=42)
        sim = Simulator(sc)
        report = sim.run()
        metrics = _extract_metrics(report)

        expected_keys = {
            "worker_count", "replication_ratio", "denial_rate",
            "max_depth_used", "avg_depth", "tasks_per_worker",
            "total_tasks", "kill_rate", "duration_ms",
        }
        assert set(metrics.keys()) == expected_keys

    def test_metrics_are_floats(self):
        from replication.simulator import Simulator, ScenarioConfig
        sc = ScenarioConfig(max_depth=2, max_replicas=5, strategy="greedy", seed=42)
        sim = Simulator(sc)
        report = sim.run()
        metrics = _extract_metrics(report)
        for k, v in metrics.items():
            assert isinstance(v, float), f"{k} should be float, got {type(v)}"

    def test_denial_rate_bounded(self):
        from replication.simulator import Simulator, ScenarioConfig
        sc = ScenarioConfig(max_depth=2, max_replicas=5, strategy="greedy", seed=42)
        sim = Simulator(sc)
        report = sim.run()
        metrics = _extract_metrics(report)
        assert 0.0 <= metrics["denial_rate"] <= 1.0
        assert 0.0 <= metrics["replication_ratio"] <= 1.0
        assert 0.0 <= metrics["kill_rate"] <= 1.0

    def test_deterministic_with_seed(self):
        from replication.simulator import Simulator, ScenarioConfig
        results = []
        for _ in range(2):
            sc = ScenarioConfig(max_depth=2, max_replicas=5, strategy="greedy", seed=99)
            sim = Simulator(sc)
            report = sim.run()
            results.append(_extract_metrics(report))
        # Same seed → same metrics (except duration_ms which is timing-dependent)
        for key in results[0]:
            if key != "duration_ms":
                assert results[0][key] == results[1][key], f"{key} differs across runs"


# ── CLI ──────────────────────────────────────────────


class TestCLI:
    def test_default_run(self):
        """CLI with minimal windows should complete successfully."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--windows", "2", "--seed", "42"])
        # Exit code 0 = passed, 1 = drift detected (both valid)
        assert exc_info.value.code in (0, 1)

    def test_json_output(self, capsys):
        try:
            main(["--windows", "2", "--seed", "42", "--json"])
        except SystemExit:
            pass
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "passed" in parsed
        assert "trends" in parsed

    def test_export_flag(self, tmp_path):
        export_path = str(tmp_path / "drift-report.json")
        try:
            main(["--windows", "2", "--seed", "42", "--export", export_path])
        except SystemExit:
            pass
        with open(export_path) as f:
            data = json.load(f)
        assert "trends" in data
        assert "alerts" in data

    def test_conservative_strategy(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["--windows", "2", "--strategy", "conservative", "--seed", "42"])
        assert exc_info.value.code in (0, 1)

    def test_sensitivity_arg(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["--windows", "2", "--sensitivity", "0.5", "--seed", "42"])
        assert exc_info.value.code in (0, 1)
