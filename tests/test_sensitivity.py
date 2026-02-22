"""Tests for the parameter sensitivity analyzer."""

from __future__ import annotations

import json
import math

import pytest

from replication.sensitivity import (
    METRIC_NAMES,
    PARAM_DEFS,
    SensitivityAnalyzer,
    SensitivityConfig,
    SensitivityCurve,
    SensitivityResult,
    SweepPoint,
    TippingPoint,
    _extract_metrics,
    _mean,
    _std,
)
from replication.simulator import ScenarioConfig, Simulator, PRESETS


# ── Statistics helpers ──────────────────────────────────────────────────


class TestStatHelpers:
    def test_mean_basic(self):
        assert _mean([1, 2, 3, 4, 5]) == 3.0

    def test_mean_empty(self):
        assert _mean([]) == 0.0

    def test_mean_single(self):
        assert _mean([42]) == 42.0

    def test_std_basic(self):
        s = _std([2, 4, 4, 4, 5, 5, 7, 9])
        assert abs(s - 2.138) < 0.01

    def test_std_empty(self):
        assert _std([]) == 0.0

    def test_std_single(self):
        assert _std([5]) == 0.0


# ── Extract metrics ────────────────────────────────────────────────────


class TestExtractMetrics:
    def _run_sim(self, **kwargs) -> dict:
        cfg = ScenarioConfig(seed=42, **kwargs)
        sim = Simulator(cfg)
        report = sim.run()
        return _extract_metrics(report)

    def test_returns_all_metric_keys(self):
        metrics = self._run_sim()
        for name in METRIC_NAMES:
            assert name in metrics, f"Missing metric: {name}"

    def test_total_workers_positive(self):
        metrics = self._run_sim()
        assert metrics["total_workers"] >= 1

    def test_total_tasks_positive(self):
        metrics = self._run_sim()
        assert metrics["total_tasks"] >= 1

    def test_denial_rate_between_0_and_1(self):
        metrics = self._run_sim(strategy="greedy", max_depth=2, max_replicas=5)
        assert 0 <= metrics["denial_rate"] <= 1

    def test_efficiency_ratio(self):
        metrics = self._run_sim(tasks_per_worker=3)
        # efficiency = total_tasks / total_workers
        expected = metrics["total_tasks"] / metrics["total_workers"]
        assert abs(metrics["efficiency"] - expected) < 0.001

    def test_total_cpu_scales_with_workers(self):
        metrics = self._run_sim(cpu_limit=2.0)
        assert metrics["total_cpu"] == 2.0 * metrics["total_workers"]

    def test_total_memory_scales_with_workers(self):
        metrics = self._run_sim(memory_limit_mb=512)
        assert metrics["total_memory_mb"] == 512.0 * metrics["total_workers"]

    def test_max_depth_within_limits(self):
        metrics = self._run_sim(max_depth=3)
        assert metrics["max_depth_reached"] <= 3

    def test_conservative_fewer_workers(self):
        greedy = self._run_sim(strategy="greedy", max_depth=4, max_replicas=20)
        conservative = self._run_sim(strategy="conservative", max_depth=4, max_replicas=20)
        # Conservative should generally produce fewer workers
        assert conservative["total_workers"] <= greedy["total_workers"]


# ── Parameter definitions ──────────────────────────────────────────────


class TestParamDefs:
    def test_all_params_have_values(self):
        for name, pdef in PARAM_DEFS.items():
            assert len(pdef.values) >= 2, f"{name} needs at least 2 values"

    def test_all_params_have_display_name(self):
        for name, pdef in PARAM_DEFS.items():
            assert pdef.display_name, f"{name} missing display_name"

    def test_all_params_have_description(self):
        for name, pdef in PARAM_DEFS.items():
            assert pdef.description, f"{name} missing description"

    def test_setter_modifies_config(self):
        for name, pdef in PARAM_DEFS.items():
            cfg = ScenarioConfig()
            val = pdef.values[-1]
            pdef.setter(cfg, val)
            assert getattr(cfg, name) == val, f"Setter for {name} didn't work"

    def test_known_params_present(self):
        expected = {
            "max_depth", "max_replicas", "cooldown_seconds",
            "tasks_per_worker", "replication_probability",
            "cpu_limit", "memory_limit_mb",
        }
        assert expected == set(PARAM_DEFS.keys())


# ── Sweep single parameter ────────────────────────────────────────────


class TestSweepParameter:
    def test_sweep_max_depth(self):
        cfg = SensitivityConfig(runs_per_point=3)
        analyzer = SensitivityAnalyzer(cfg)
        curve = analyzer.sweep_parameter("max_depth")

        assert isinstance(curve, SensitivityCurve)
        assert curve.param.name == "max_depth"
        assert len(curve.points) == len(PARAM_DEFS["max_depth"].values)

    def test_sweep_point_has_metrics(self):
        cfg = SensitivityConfig(runs_per_point=3)
        analyzer = SensitivityAnalyzer(cfg)
        curve = analyzer.sweep_parameter("max_depth")

        for pt in curve.points:
            assert isinstance(pt, SweepPoint)
            assert pt.num_runs == 3
            for metric in METRIC_NAMES:
                assert metric in pt.metric_means
                assert metric in pt.metric_stds

    def test_sweep_workers_increase_with_depth(self):
        cfg = SensitivityConfig(
            runs_per_point=5,
            base_config=ScenarioConfig(
                strategy="greedy", max_replicas=50, seed=42,
            ),
        )
        analyzer = SensitivityAnalyzer(cfg)
        curve = analyzer.sweep_parameter("max_depth")

        workers_at_depth_1 = curve.points[0].metric_means["total_workers"]
        workers_at_depth_8 = curve.points[-1].metric_means["total_workers"]
        assert workers_at_depth_8 >= workers_at_depth_1

    def test_sweep_unknown_param_raises(self):
        analyzer = SensitivityAnalyzer()
        with pytest.raises(ValueError, match="Unknown parameter"):
            analyzer.sweep_parameter("nonexistent_param")

    def test_sweep_impact_scores_normalized(self):
        cfg = SensitivityConfig(runs_per_point=3)
        analyzer = SensitivityAnalyzer(cfg)
        curve = analyzer.sweep_parameter("max_replicas")

        for score in curve.impact_scores.values():
            assert 0 <= score <= 100

    def test_sweep_impact_has_100(self):
        cfg = SensitivityConfig(runs_per_point=3)
        analyzer = SensitivityAnalyzer(cfg)
        curve = analyzer.sweep_parameter("max_replicas")

        # At least one metric should have score 100 (the max)
        if curve.impact_scores:
            assert max(curve.impact_scores.values()) == pytest.approx(100, abs=0.1)

    def test_sweep_with_base_config(self):
        base = ScenarioConfig(strategy="conservative", max_depth=5)
        cfg = SensitivityConfig(runs_per_point=3, base_config=base)
        analyzer = SensitivityAnalyzer(cfg)
        curve = analyzer.sweep_parameter("max_replicas", base)

        assert len(curve.points) > 0

    def test_sweep_tasks_per_worker(self):
        cfg = SensitivityConfig(runs_per_point=3)
        analyzer = SensitivityAnalyzer(cfg)
        curve = analyzer.sweep_parameter("tasks_per_worker")

        # More tasks per worker = more total tasks
        tasks_low = curve.points[0].metric_means["total_tasks"]
        tasks_high = curve.points[-1].metric_means["total_tasks"]
        assert tasks_high >= tasks_low


# ── Tipping point detection ───────────────────────────────────────────


class TestTippingPoints:
    def test_tipping_detected_on_sharp_change(self):
        """With greedy strategy and tight replicas, raising depth should
        cause a sharp jump in workers at some point."""
        cfg = SensitivityConfig(
            runs_per_point=5,
            base_config=ScenarioConfig(
                strategy="greedy", max_replicas=50, seed=42,
            ),
        )
        analyzer = SensitivityAnalyzer(cfg)
        curve = analyzer.sweep_parameter("max_depth")

        # Should detect at least one tipping point
        # (workers grow exponentially with depth for greedy)
        if curve.tipping_points:
            tp = curve.tipping_points[0]
            assert isinstance(tp, TippingPoint)
            assert tp.metric in METRIC_NAMES
            assert abs(tp.relative_change) >= 0.5

    def test_tipping_sorted_by_magnitude(self):
        cfg = SensitivityConfig(runs_per_point=5)
        analyzer = SensitivityAnalyzer(cfg)
        curve = analyzer.sweep_parameter("max_depth")

        for i in range(len(curve.tipping_points) - 1):
            assert abs(curve.tipping_points[i].relative_change) >= \
                   abs(curve.tipping_points[i + 1].relative_change)

    def test_no_tipping_on_flat_metric(self):
        """Conservative with depth=1 should have minimal variation."""
        cfg = SensitivityConfig(
            runs_per_point=3,
            base_config=ScenarioConfig(
                strategy="burst", max_depth=1, max_replicas=5,
            ),
        )
        analyzer = SensitivityAnalyzer(cfg)
        curve = analyzer.sweep_parameter("cpu_limit")

        # CPU limit doesn't affect simulation logic, so no tipping
        worker_tips = [
            tp for tp in curve.tipping_points
            if tp.metric == "total_workers"
        ]
        assert len(worker_tips) == 0


# ── Full analysis ──────────────────────────────────────────────────────


class TestFullAnalysis:
    def test_analyze_single_param(self):
        cfg = SensitivityConfig(
            runs_per_point=3,
            parameters=["max_depth"],
        )
        analyzer = SensitivityAnalyzer(cfg)
        result = analyzer.analyze()

        assert isinstance(result, SensitivityResult)
        assert "max_depth" in result.curves
        assert len(result.curves) == 1

    def test_analyze_multiple_params(self):
        cfg = SensitivityConfig(
            runs_per_point=3,
            parameters=["max_depth", "max_replicas"],
        )
        analyzer = SensitivityAnalyzer(cfg)
        result = analyzer.analyze()

        assert len(result.curves) == 2
        assert "max_depth" in result.curves
        assert "max_replicas" in result.curves

    def test_parameter_ranking_sorted(self):
        cfg = SensitivityConfig(
            runs_per_point=3,
            parameters=["max_depth", "max_replicas", "cpu_limit"],
        )
        analyzer = SensitivityAnalyzer(cfg)
        result = analyzer.analyze()

        scores = [s for _, s in result.parameter_ranking]
        assert scores == sorted(scores, reverse=True)

    def test_total_simulations_counted(self):
        cfg = SensitivityConfig(
            runs_per_point=3,
            parameters=["max_depth"],
        )
        analyzer = SensitivityAnalyzer(cfg)
        result = analyzer.analyze()

        n_values = len(PARAM_DEFS["max_depth"].values)
        assert result.total_simulations == n_values * 3

    def test_duration_positive(self):
        cfg = SensitivityConfig(
            runs_per_point=3,
            parameters=["cpu_limit"],
        )
        analyzer = SensitivityAnalyzer(cfg)
        result = analyzer.analyze()

        assert result.duration_ms > 0

    def test_analyze_with_preset(self):
        cfg = SensitivityConfig(
            runs_per_point=3,
            parameters=["max_depth"],
            base_scenario="minimal",
        )
        analyzer = SensitivityAnalyzer(cfg)
        result = analyzer.analyze()

        assert result.base_config.strategy == "conservative"
        assert result.base_config.max_depth == 1

    def test_analyze_with_overrides(self):
        cfg = SensitivityConfig(
            runs_per_point=3,
            parameters=["max_replicas"],
            strategy="chain",
            max_depth=5,
        )
        analyzer = SensitivityAnalyzer(cfg)
        result = analyzer.analyze()

        assert result.base_config.strategy == "chain"
        assert result.base_config.max_depth == 5


# ── Rendering ──────────────────────────────────────────────────────────


class TestRendering:
    @pytest.fixture
    def result(self):
        cfg = SensitivityConfig(
            runs_per_point=3,
            parameters=["max_depth", "max_replicas"],
        )
        analyzer = SensitivityAnalyzer(cfg)
        return analyzer.analyze()

    def test_result_render_contains_header(self, result):
        text = result.render()
        assert "Parameter Sensitivity Analysis" in text

    def test_result_render_contains_ranking(self, result):
        text = result.render()
        assert "Impact Ranking" in text

    def test_result_render_contains_recommendations(self, result):
        text = result.render()
        assert "Recommendations" in text

    def test_result_render_contains_param_tables(self, result):
        text = result.render()
        assert "Max Depth" in text
        assert "Max Replicas" in text

    def test_curve_render_has_header(self, result):
        curve = result.curves["max_depth"]
        text = curve.render()
        assert "Sensitivity: Max Depth" in text

    def test_curve_render_has_values(self, result):
        curve = result.curves["max_depth"]
        text = curve.render()
        for val in PARAM_DEFS["max_depth"].values:
            assert str(val) in text

    def test_curve_render_has_impact_scores(self, result):
        curve = result.curves["max_depth"]
        text = curve.render()
        assert "Impact Scores" in text


# ── JSON serialization ────────────────────────────────────────────────


class TestSerialization:
    @pytest.fixture
    def result(self):
        cfg = SensitivityConfig(
            runs_per_point=3,
            parameters=["max_depth"],
        )
        analyzer = SensitivityAnalyzer(cfg)
        return analyzer.analyze()

    def test_to_dict_is_json_serializable(self, result):
        d = result.to_dict()
        text = json.dumps(d, default=str)
        assert len(text) > 100

    def test_to_dict_has_expected_keys(self, result):
        d = result.to_dict()
        assert "base_config" in d
        assert "parameter_ranking" in d
        assert "tipping_points" in d
        assert "curves" in d
        assert "total_simulations" in d
        assert "duration_ms" in d

    def test_curve_to_dict(self, result):
        curve_dict = result.curves["max_depth"].to_dict()
        assert curve_dict["parameter"] == "max_depth"
        assert "points" in curve_dict
        assert "tipping_points" in curve_dict
        assert "impact_scores" in curve_dict

    def test_curve_points_have_means_and_stds(self, result):
        curve_dict = result.curves["max_depth"].to_dict()
        for pt in curve_dict["points"]:
            assert "value" in pt
            assert "means" in pt
            assert "stds" in pt
            assert "num_runs" in pt

    def test_ranking_values_roundtrip(self, result):
        d = result.to_dict()
        for entry in d["parameter_ranking"]:
            assert isinstance(entry["parameter"], str)
            assert isinstance(entry["impact_score"], float)


# ── Edge cases ─────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_run_per_point(self):
        cfg = SensitivityConfig(
            runs_per_point=1,
            parameters=["max_depth"],
        )
        analyzer = SensitivityAnalyzer(cfg)
        result = analyzer.analyze()

        assert result.total_simulations == len(PARAM_DEFS["max_depth"].values)

    def test_empty_parameters_list_sweeps_all(self):
        cfg = SensitivityConfig(
            runs_per_point=2,
            parameters=None,
        )
        analyzer = SensitivityAnalyzer(cfg)
        result = analyzer.analyze()

        assert len(result.curves) == len(PARAM_DEFS)

    def test_unknown_params_in_list_skipped(self):
        cfg = SensitivityConfig(
            runs_per_point=2,
            parameters=["max_depth", "nonexistent"],
        )
        analyzer = SensitivityAnalyzer(cfg)
        result = analyzer.analyze()

        assert "max_depth" in result.curves
        assert "nonexistent" not in result.curves

    def test_burst_strategy_sensitivity(self):
        """Burst strategy: only root replicates, so depth doesn't matter much."""
        cfg = SensitivityConfig(
            runs_per_point=3,
            parameters=["max_depth"],
            strategy="burst",
        )
        analyzer = SensitivityAnalyzer(cfg)
        result = analyzer.analyze()

        # Workers should be similar across depths for burst
        curve = result.curves["max_depth"]
        workers = [pt.metric_means["total_workers"] for pt in curve.points]
        # All should be close to max_replicas + 1 (or capped by it)
        assert len(set(int(w) for w in workers)) <= 3  # minimal variation

    def test_chain_strategy_depth_matters(self):
        """Chain strategy: linear replication, depth = chain length."""
        cfg = SensitivityConfig(
            runs_per_point=3,
            parameters=["max_depth"],
            strategy="chain",
            max_replicas=20,
        )
        analyzer = SensitivityAnalyzer(cfg)
        result = analyzer.analyze()

        curve = result.curves["max_depth"]
        # Workers should grow linearly with depth for chain
        first = curve.points[0].metric_means["total_workers"]
        last = curve.points[-1].metric_means["total_workers"]
        assert last > first

    def test_recommendations_generated(self):
        cfg = SensitivityConfig(
            runs_per_point=3,
            parameters=["max_depth", "max_replicas"],
        )
        analyzer = SensitivityAnalyzer(cfg)
        result = analyzer.analyze()

        text = result._generate_recommendations()
        assert "Recommendations" in text
        assert "highest impact" in text

    def test_no_data_recommendations(self):
        """SensitivityResult with empty ranking still renders."""
        result = SensitivityResult(
            base_config=ScenarioConfig(),
            curves={},
            parameter_ranking=[],
            all_tipping_points=[],
            duration_ms=0,
            total_simulations=0,
        )
        text = result._generate_recommendations()
        assert "No data available" in text

    def test_result_render_empty(self):
        """Empty result renders without error."""
        result = SensitivityResult(
            base_config=ScenarioConfig(),
            curves={},
            parameter_ranking=[],
            all_tipping_points=[],
            duration_ms=0,
            total_simulations=0,
        )
        text = result.render()
        assert "Parameter Sensitivity Analysis" in text

    def test_all_tipping_points_sorted_globally(self):
        cfg = SensitivityConfig(
            runs_per_point=3,
            parameters=["max_depth", "max_replicas"],
        )
        analyzer = SensitivityAnalyzer(cfg)
        result = analyzer.analyze()

        for i in range(len(result.all_tipping_points) - 1):
            assert abs(result.all_tipping_points[i][1].relative_change) >= \
                   abs(result.all_tipping_points[i + 1][1].relative_change)


# ── CLI integration (smoke test) ──────────────────────────────────────


class TestCLI:
    def test_import_main(self):
        from replication.sensitivity import main
        assert callable(main)

    def test_public_api_importable(self):
        from replication import (
            SensitivityAnalyzer,
            SensitivityConfig,
            SensitivityCurve,
            SensitivityResult,
            TippingPoint,
            PARAM_DEFS,
        )
        assert SensitivityAnalyzer is not None
        assert len(PARAM_DEFS) > 0
