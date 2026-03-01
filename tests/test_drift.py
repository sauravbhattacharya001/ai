"""Tests for drift detection module."""

from replication.drift import (
    DriftDetector,
    DriftConfig,
    DriftDirection,
    DriftSeverity,
    _linear_regression,
    _longest_monotonic_run,
    _sparkline,
)


class TestHelpers:
    """Test helper functions."""

    def test_linear_regression_flat(self):
        slope, intercept, r2 = _linear_regression([5.0, 5.0, 5.0, 5.0])
        assert slope == 0.0
        assert r2 == 0.0

    def test_linear_regression_perfect(self):
        slope, intercept, r2 = _linear_regression([1.0, 2.0, 3.0, 4.0])
        assert abs(slope - 1.0) < 1e-9
        assert abs(r2 - 1.0) < 1e-9

    def test_longest_monotonic_run_increasing(self):
        assert _longest_monotonic_run([1, 2, 3, 2, 1]) == 3

    def test_longest_monotonic_run_decreasing(self):
        assert _longest_monotonic_run([5, 4, 3, 2, 6]) == 4

    def test_sparkline(self):
        result = _sparkline([0, 0.5, 1.0])
        assert len(result) == 3
        assert result[0] == "▁"
        assert result[-1] == "█"


class TestDriftDetector:
    """Test drift detection."""

    def test_basic_analysis(self):
        config = DriftConfig(windows=5, seed_base=42)
        detector = DriftDetector()
        result = detector.analyze(config)
        assert result.windows_analyzed == 5
        assert len(result.trends) > 0
        assert result.duration_ms > 0

    def test_render_output(self):
        config = DriftConfig(windows=5, seed_base=42)
        detector = DriftDetector()
        result = detector.analyze(config)
        rendered = result.render()
        assert "Drift Detection Report" in rendered
        assert "Sparklines" in rendered

    def test_to_dict(self):
        config = DriftConfig(windows=5, seed_base=42)
        detector = DriftDetector()
        result = detector.analyze(config)
        d = result.to_dict()
        assert "passed" in d
        assert "trends" in d
        assert "alerts" in d
        assert isinstance(d["trends"], list)

    def test_different_strategies(self):
        for strategy in ["greedy", "conservative", "random"]:
            config = DriftConfig(windows=3, strategy=strategy, seed_base=42)
            detector = DriftDetector()
            result = detector.analyze(config)
            assert result.windows_analyzed == 3

    def test_high_sensitivity_catches_less(self):
        config = DriftConfig(windows=5, seed_base=42, sensitivity=100.0)
        detector = DriftDetector()
        result = detector.analyze(config)
        # With very high sensitivity threshold, everything should be stable
        for trend in result.trends:
            assert trend.direction == DriftDirection.STABLE
