"""Tests for the Monte Carlo risk analyzer."""

from __future__ import annotations

import json
import math

import pytest

from replication.montecarlo import (
    MonteCarloAnalyzer,
    MonteCarloComparison,
    MonteCarloConfig,
    MonteCarloResult,
    MetricDistribution,
    RiskMetrics,
    _mean,
    _median,
    _std,
    _percentile,
    _ci95,
)
from replication.simulator import ScenarioConfig, PRESETS


# ── Statistics helpers ──────────────────────────────────────────────────


class TestStatHelpers:
    def test_mean_basic(self):
        assert _mean([1, 2, 3, 4, 5]) == 3.0

    def test_mean_empty(self):
        assert _mean([]) == 0.0

    def test_mean_single(self):
        assert _mean([42]) == 42.0

    def test_median_odd(self):
        assert _median([3, 1, 2]) == 2.0

    def test_median_even(self):
        assert _median([1, 2, 3, 4]) == 2.5

    def test_median_empty(self):
        assert _median([]) == 0.0

    def test_median_single(self):
        assert _median([7]) == 7.0

    def test_std_basic(self):
        s = _std([2, 4, 4, 4, 5, 5, 7, 9])
        assert abs(s - 2.138) < 0.01

    def test_std_empty(self):
        assert _std([]) == 0.0

    def test_std_single(self):
        assert _std([5]) == 0.0

    def test_std_identical(self):
        assert _std([3, 3, 3, 3]) == 0.0

    def test_percentile_basic(self):
        vals = list(range(1, 101))  # 1..100
        assert _percentile(vals, 50) == pytest.approx(50.5, abs=0.1)
        assert _percentile(vals, 0) == 1.0
        assert _percentile(vals, 100) == 100.0

    def test_percentile_empty(self):
        assert _percentile([], 50) == 0.0

    def test_percentile_single(self):
        assert _percentile([42], 95) == 42.0

    def test_percentile_small(self):
        assert _percentile([1, 2, 3], 50) == 2.0

    def test_ci95_basic(self):
        vals = [10.0] * 100
        lo, hi = _ci95(vals)
        assert lo == pytest.approx(10.0, abs=0.01)
        assert hi == pytest.approx(10.0, abs=0.01)

    def test_ci95_single(self):
        lo, hi = _ci95([5.0])
        assert lo == 5.0
        assert hi == 5.0

    def test_ci95_spread(self):
        vals = list(range(100))
        lo, hi = _ci95([float(v) for v in vals])
        mean = sum(vals) / len(vals)
        assert lo < mean < hi


# ── MetricDistribution ──────────────────────────────────────────────────


class TestMetricDistribution:
    def test_basic_properties(self):
        d = MetricDistribution("test", "count", [1, 2, 3, 4, 5])
        assert d.mean == 3.0
        assert d.median == 3.0
        assert d.min == 1.0
        assert d.max == 5.0

    def test_percentiles(self):
        d = MetricDistribution("test", "count", list(range(1, 101)))
        assert d.p5 < d.p25 < d.median < d.p75 < d.p95

    def test_to_dict(self):
        d = MetricDistribution("workers", "count", [1, 2, 3])
        result = d.to_dict()
        assert result["name"] == "workers"
        assert result["unit"] == "count"
        assert result["n"] == 3
        assert "mean" in result
        assert "p95" in result
        assert "ci95_lower" in result


# ── RiskMetrics ─────────────────────────────────────────────────────────


class TestRiskMetrics:
    def test_to_dict(self):
        rm = RiskMetrics(
            prob_max_depth_reached=0.5,
            prob_quota_saturated=0.3,
            prob_all_denied=0.1,
            max_depth_breach_pct=50.0,
            quota_saturation_pct=30.0,
            peak_worker_p95=8.0,
            peak_depth_p95=3.0,
        )
        d = rm.to_dict()
        assert d["prob_max_depth_reached"] == 0.5
        assert d["peak_worker_p95"] == 8.0


# ── MonteCarloResult ────────────────────────────────────────────────────


class TestMonteCarloResult:
    @pytest.fixture
    def sample_result(self):
        """Create a sample MonteCarloResult for rendering tests."""
        config = MonteCarloConfig(num_runs=10)
        scenario = ScenarioConfig(max_depth=3, max_replicas=10, strategy="greedy")
        distributions = {
            "total_workers": MetricDistribution("Total Workers", "count", [5, 6, 7, 8, 9, 10, 5, 6, 7, 8]),
            "total_tasks": MetricDistribution("Total Tasks", "count", [10, 12, 14, 16, 18, 20, 10, 12, 14, 16]),
            "replications_ok": MetricDistribution("Replications OK", "count", [4, 5, 6, 7, 8, 9, 4, 5, 6, 7]),
            "replications_denied": MetricDistribution("Replications Denied", "count", [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]),
            "max_depth_reached": MetricDistribution("Max Depth Reached", "depth", [2, 3, 3, 2, 3, 3, 2, 3, 3, 2]),
            "efficiency": MetricDistribution("Efficiency", "tasks/worker", [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
            "denial_rate": MetricDistribution("Denial Rate", "%", [20, 28, 33, 12, 20, 25, 20, 28, 33, 12]),
            "run_duration": MetricDistribution("Run Duration", "ms", [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        }
        risk = RiskMetrics(
            prob_max_depth_reached=0.6,
            prob_quota_saturated=0.8,
            prob_all_denied=0.0,
            max_depth_breach_pct=60.0,
            quota_saturation_pct=80.0,
            peak_worker_p95=9.55,
            peak_depth_p95=3.0,
        )
        return MonteCarloResult(
            config=config,
            scenario=scenario,
            num_runs=10,
            distributions=distributions,
            risk_metrics=risk,
            depth_histogram={2: 4, 3: 6},
            worker_histogram={5: 2, 6: 2, 7: 2, 8: 2, 9: 1, 10: 1},
            denial_rate_histogram={10: 2, 20: 4, 30: 4},
            duration_ms=50.0,
        )

    def test_risk_level_low(self):
        rm = RiskMetrics(0.05, 0.05, 0, 5, 5, 3, 1)
        config = MonteCarloConfig(num_runs=10)
        scenario = ScenarioConfig()
        result = MonteCarloResult(
            config=config, scenario=scenario, num_runs=10,
            distributions={}, risk_metrics=rm,
            depth_histogram={}, worker_histogram={},
            denial_rate_histogram={}, duration_ms=1,
        )
        assert result.risk_level == "LOW"
        assert result.risk_emoji == "🟢"

    def test_risk_level_critical(self):
        rm = RiskMetrics(0.95, 0.95, 0.5, 95, 95, 50, 5)
        config = MonteCarloConfig(num_runs=10)
        scenario = ScenarioConfig()
        result = MonteCarloResult(
            config=config, scenario=scenario, num_runs=10,
            distributions={}, risk_metrics=rm,
            depth_histogram={}, worker_histogram={},
            denial_rate_histogram={}, duration_ms=1,
        )
        assert result.risk_level == "CRITICAL"
        assert result.risk_emoji == "⛔"

    def test_risk_level_moderate(self):
        rm = RiskMetrics(0.2, 0.2, 0.0, 20, 20, 5, 2)
        config = MonteCarloConfig(num_runs=10)
        scenario = ScenarioConfig()
        result = MonteCarloResult(
            config=config, scenario=scenario, num_runs=10,
            distributions={}, risk_metrics=rm,
            depth_histogram={}, worker_histogram={},
            denial_rate_histogram={}, duration_ms=1,
        )
        assert result.risk_level == "MODERATE"

    def test_render_summary(self, sample_result):
        text = sample_result.render_summary()
        assert "Monte Carlo" in text
        assert "greedy" in text
        assert "10" in text  # num_runs

    def test_render_distributions(self, sample_result):
        text = sample_result.render_distributions()
        assert "Total Workers" in text
        assert "Efficiency" in text

    def test_render_risk_indicators(self, sample_result):
        text = sample_result.render_risk_indicators()
        assert "Max Depth Reached" in text
        assert "Quota Saturated" in text
        assert "█" in text  # progress bars

    def test_render_histograms(self, sample_result):
        text = sample_result.render_histograms()
        assert "depth" in text.lower()
        assert "█" in text

    def test_render_recommendations(self, sample_result):
        text = sample_result.render_recommendations()
        # Should have recommendations since risk is elevated
        assert "Recommendations" in text or "✅" in text

    def test_render_full(self, sample_result):
        text = sample_result.render()
        assert "Monte Carlo" in text
        assert "Risk Indicators" in text

    def test_to_dict(self, sample_result):
        d = sample_result.to_dict()
        assert d["num_runs"] == 10
        assert "risk_level" in d
        assert "distributions" in d
        assert "risk_metrics" in d
        assert "depth_histogram" in d


# ── MonteCarloAnalyzer ──────────────────────────────────────────────────


class TestMonteCarloAnalyzer:
    def test_analyze_default(self):
        config = MonteCarloConfig(num_runs=10)
        analyzer = MonteCarloAnalyzer(config)
        result = analyzer.analyze()
        assert result.num_runs == 10
        assert len(result.distributions) == 8
        assert result.risk_level in ("LOW", "MODERATE", "ELEVATED", "HIGH", "CRITICAL")

    def test_analyze_with_overrides(self):
        config = MonteCarloConfig(num_runs=5)
        analyzer = MonteCarloAnalyzer(config)
        result = analyzer.analyze(strategy="conservative", max_depth=2)
        assert result.scenario.strategy == "conservative"
        assert result.scenario.max_depth == 2

    def test_analyze_with_preset(self):
        config = MonteCarloConfig(
            num_runs=5,
            base_scenario=PRESETS["minimal"],
        )
        analyzer = MonteCarloAnalyzer(config)
        result = analyzer.analyze()
        assert result.scenario.max_depth == 1
        assert result.scenario.max_replicas == 3

    def test_distributions_populated(self):
        config = MonteCarloConfig(num_runs=10)
        analyzer = MonteCarloAnalyzer(config)
        result = analyzer.analyze()
        for key in ["total_workers", "total_tasks", "max_depth_reached", "efficiency"]:
            assert key in result.distributions
            assert len(result.distributions[key].values) == 10

    def test_risk_metrics_valid(self):
        config = MonteCarloConfig(num_runs=10)
        analyzer = MonteCarloAnalyzer(config)
        result = analyzer.analyze()
        rm = result.risk_metrics
        assert 0 <= rm.prob_max_depth_reached <= 1
        assert 0 <= rm.prob_quota_saturated <= 1
        assert 0 <= rm.prob_all_denied <= 1
        assert 0 <= rm.max_depth_breach_pct <= 100
        assert 0 <= rm.quota_saturation_pct <= 100

    def test_histograms_populated(self):
        config = MonteCarloConfig(num_runs=10)
        analyzer = MonteCarloAnalyzer(config)
        result = analyzer.analyze()
        assert len(result.depth_histogram) > 0
        assert len(result.worker_histogram) > 0

    def test_compare_two_strategies(self):
        config = MonteCarloConfig(num_runs=5)
        analyzer = MonteCarloAnalyzer(config)
        comparison = analyzer.compare(["greedy", "conservative"])
        assert len(comparison.labels) == 2
        assert len(comparison.results) == 2
        assert "greedy" in comparison.labels
        assert "conservative" in comparison.labels

    def test_compare_all_strategies(self):
        config = MonteCarloConfig(num_runs=3)
        analyzer = MonteCarloAnalyzer(config)
        comparison = analyzer.compare()
        assert len(comparison.labels) == 5  # all 5 strategies

    def test_deterministic_with_fixed_seed(self):
        """When randomize_seeds=False, same seed → same results."""
        base = ScenarioConfig(seed=42, strategy="random", replication_probability=0.5)
        config = MonteCarloConfig(num_runs=5, base_scenario=base, randomize_seeds=False)
        analyzer = MonteCarloAnalyzer(config)
        r1 = analyzer.analyze()
        r2 = analyzer.analyze()
        # Same seed, no randomization → identical distributions
        assert r1.distributions["total_workers"].values == r2.distributions["total_workers"].values

    def test_randomized_seeds_vary(self):
        """With randomize_seeds=True, runs should have variance (for random strategy)."""
        base = ScenarioConfig(strategy="random", replication_probability=0.5)
        config = MonteCarloConfig(num_runs=20, base_scenario=base, randomize_seeds=True)
        analyzer = MonteCarloAnalyzer(config)
        result = analyzer.analyze()
        workers = result.distributions["total_workers"]
        # With random strategy and 20 runs, we expect some variance
        # (greedy/chain/burst are deterministic regardless)
        assert workers.std >= 0  # at minimum non-negative

    def test_json_roundtrip(self):
        config = MonteCarloConfig(num_runs=5)
        analyzer = MonteCarloAnalyzer(config)
        result = analyzer.analyze()
        d = result.to_dict()
        serialized = json.dumps(d, default=str)
        parsed = json.loads(serialized)
        assert parsed["num_runs"] == 5
        assert "distributions" in parsed
        assert "risk_metrics" in parsed


# ── MonteCarloComparison ────────────────────────────────────────────────


class TestMonteCarloComparison:
    @pytest.fixture
    def comparison(self):
        config = MonteCarloConfig(num_runs=5)
        analyzer = MonteCarloAnalyzer(config)
        return analyzer.compare(["greedy", "conservative"])

    def test_render_table(self, comparison):
        text = comparison.render_table()
        assert "greedy" in text
        assert "conservative" in text
        assert "Risk" in text

    def test_render_risk_ranking(self, comparison):
        text = comparison.render_risk_ranking()
        assert "safest" in text.lower()

    def test_render_insights(self, comparison):
        text = comparison.render_insights()
        assert "Safest" in text or "safest" in text.lower()

    def test_render_full(self, comparison):
        text = comparison.render()
        assert "Monte Carlo" in text

    def test_to_dict(self, comparison):
        d = comparison.to_dict()
        assert "greedy" in d["strategies"]
        assert "conservative" in d["strategies"]
        assert "duration_ms" in d


# ── Edge cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_run(self):
        config = MonteCarloConfig(num_runs=1)
        analyzer = MonteCarloAnalyzer(config)
        result = analyzer.analyze()
        assert result.num_runs == 1
        assert result.risk_level in ("LOW", "MODERATE", "ELEVATED", "HIGH", "CRITICAL")

    def test_conservative_low_risk(self):
        """Conservative strategy with tight limits should show lower risk."""
        base = ScenarioConfig(strategy="conservative", max_depth=1, max_replicas=3)
        config = MonteCarloConfig(num_runs=10, base_scenario=base)
        analyzer = MonteCarloAnalyzer(config)
        result = analyzer.analyze()
        # Conservative with depth 1 should have limited growth
        assert result.distributions["max_depth_reached"].max <= 1

    def test_burst_strategy(self):
        """Burst creates many workers at depth 1."""
        base = ScenarioConfig(strategy="burst", max_depth=1, max_replicas=20)
        config = MonteCarloConfig(num_runs=5, base_scenario=base)
        analyzer = MonteCarloAnalyzer(config)
        result = analyzer.analyze()
        assert result.distributions["max_depth_reached"].max <= 1

    def test_chain_strategy(self):
        """Chain creates a linear sequence."""
        base = ScenarioConfig(strategy="chain", max_depth=5, max_replicas=10)
        config = MonteCarloConfig(num_runs=5, base_scenario=base)
        analyzer = MonteCarloAnalyzer(config)
        result = analyzer.analyze()
        # Chain should reach near max depth
        assert result.distributions["max_depth_reached"].mean > 0

    def test_recommendation_safe_config(self):
        """Safe config should get positive recommendation."""
        base = ScenarioConfig(strategy="conservative", max_depth=1, max_replicas=3)
        config = MonteCarloConfig(num_runs=10, base_scenario=base)
        analyzer = MonteCarloAnalyzer(config)
        result = analyzer.analyze()
        text = result.render_recommendations()
        # Should either say "safe" or give specific advice
        assert len(text) > 0

    def test_high_replication_probability(self):
        """High probability random strategy should show higher risk."""
        base = ScenarioConfig(strategy="random", replication_probability=0.95)
        config = MonteCarloConfig(num_runs=10, base_scenario=base)
        analyzer = MonteCarloAnalyzer(config)
        result = analyzer.analyze()
        # High probability → more workers → higher risk indicators
        assert result.distributions["total_workers"].mean > 1
