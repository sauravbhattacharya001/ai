"""Tests for the What-If Analysis engine."""

import json
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.replication.what_if import (
    WhatIfAnalyzer,
    WhatIfResult,
    SweepResult,
    ChangeAnalysis,
    MetricDelta,
    SweepPoint,
    RiskVerdict,
    MetricPolarity,
    METRICS,
    MUTABLE_PARAMS,
    _parse_change,
    main as cli_main,
)
from src.replication.simulator import ScenarioConfig, PRESETS


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def analyzer():
    """Fast analyzer with deterministic output."""
    return WhatIfAnalyzer(seed=42, runs_per_variant=1)


@pytest.fixture
def multi_run_analyzer():
    """Analyzer with 3 runs for averaging."""
    return WhatIfAnalyzer(seed=42, runs_per_variant=3)


# ── Construction ─────────────────────────────────────────────────────

class TestConstruction:
    def test_default_construction(self):
        a = WhatIfAnalyzer()
        assert a._baseline is not None
        assert a._runs == 3

    def test_preset_baseline(self):
        a = WhatIfAnalyzer(baseline="balanced")
        assert a._baseline == PRESETS["balanced"]

    def test_config_baseline(self):
        cfg = ScenarioConfig(max_depth=5)
        a = WhatIfAnalyzer(baseline=cfg)
        assert a._baseline.max_depth == 5

    def test_invalid_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            WhatIfAnalyzer(baseline="nonexistent")

    def test_seed_stored(self):
        a = WhatIfAnalyzer(seed=123)
        assert a._seed == 123

    def test_runs_minimum_clamped(self):
        a = WhatIfAnalyzer(runs_per_variant=0)
        assert a._runs == 1


# ── Single change analysis ──────────────────────────────────────────

class TestSingleChange:
    def test_basic_analysis_returns_result(self, analyzer):
        result = analyzer.analyze(changes={"max_depth": 5})
        assert isinstance(result, WhatIfResult)
        assert len(result.analyses) == 1

    def test_result_has_deltas(self, analyzer):
        result = analyzer.analyze(changes={"max_depth": 5})
        analysis = result.analyses[0]
        assert len(analysis.deltas) == len(METRICS)
        for d in analysis.deltas:
            assert isinstance(d, MetricDelta)
            assert d.name in METRICS

    def test_result_has_verdict(self, analyzer):
        result = analyzer.analyze(changes={"max_depth": 5})
        assert result.analyses[0].verdict in RiskVerdict

    def test_result_has_risk_score(self, analyzer):
        result = analyzer.analyze(changes={"max_depth": 5})
        score = result.analyses[0].risk_score
        assert 0 <= score <= 100

    def test_increasing_depth_is_dangerous(self, analyzer):
        result = analyzer.analyze(changes={"max_depth": 8})
        assert result.analyses[0].verdict == RiskVerdict.DANGEROUS

    def test_restricting_replicas_is_safe(self, analyzer):
        result = analyzer.analyze(changes={"max_replicas": 2})
        assert result.analyses[0].verdict == RiskVerdict.SAFE

    def test_label_generated(self, analyzer):
        result = analyzer.analyze(changes={"max_depth": 5})
        assert "max_depth=5" in result.analyses[0].label

    def test_changes_dict_preserved(self, analyzer):
        result = analyzer.analyze(changes={"max_depth": 5})
        assert result.analyses[0].changes == {"max_depth": 5}

    def test_baseline_config_included(self, analyzer):
        result = analyzer.analyze(changes={"max_depth": 5})
        assert "max_depth" in result.baseline_config
        assert result.baseline_config["max_depth"] == 3

    def test_invalid_param_raises(self, analyzer):
        with pytest.raises(ValueError, match="Unknown parameter"):
            analyzer.analyze(changes={"bogus_param": 42})


# ── Multi-change analysis ───────────────────────────────────────────

class TestMultiChange:
    def test_multiple_change_sets(self, analyzer):
        result = analyzer.analyze(change_sets=[
            {"max_depth": 5},
            {"max_replicas": 2},
        ])
        assert len(result.analyses) == 2

    def test_custom_labels(self, analyzer):
        result = analyzer.analyze(
            change_sets=[{"max_depth": 5}],
            labels=["Deep dive"],
        )
        assert result.analyses[0].label == "Deep dive"

    def test_ranking_safest_first(self, analyzer):
        result = analyzer.analyze(change_sets=[
            {"max_depth": 8},
            {"max_replicas": 2},
        ])
        assert len(result.ranking) == 2
        assert result.ranking[0].risk_score <= result.ranking[1].risk_score

    def test_changes_and_change_sets_exclusive(self, analyzer):
        with pytest.raises(ValueError, match="not both"):
            analyzer.analyze(
                changes={"max_depth": 5},
                change_sets=[{"max_replicas": 2}],
            )

    def test_default_demo_analysis(self, analyzer):
        # No changes at all -> should produce default demo
        result = analyzer.analyze()
        assert len(result.analyses) == 1


# ── Sweep ────────────────────────────────────────────────────────────

class TestSweep:
    def test_basic_sweep(self, analyzer):
        result = analyzer.sweep("max_depth", 1, 5)
        assert isinstance(result, SweepResult)
        assert len(result.points) == 5

    def test_sweep_has_values(self, analyzer):
        result = analyzer.sweep("max_depth", 1, 4)
        values = [p.value for p in result.points]
        assert values == [1, 2, 3, 4]

    def test_sweep_safest_riskiest(self, analyzer):
        result = analyzer.sweep("max_depth", 1, 6)
        assert result.safest.risk_score <= result.riskiest.risk_score

    def test_sweep_baseline_value(self, analyzer):
        result = analyzer.sweep("max_depth", 1, 5)
        assert result.baseline_value == 3

    def test_sweep_finds_tipping_point(self, analyzer):
        result = analyzer.sweep("max_depth", 1, 8)
        # At some point, increasing depth should become dangerous
        if result.tipping_point is not None:
            assert result.tipping_point > 0

    def test_sweep_float_param(self, analyzer):
        result = analyzer.sweep("replication_probability", 0.1, 1.0, steps=5)
        assert len(result.points) == 5
        assert result.points[0].value == pytest.approx(0.1, abs=0.01)
        assert result.points[-1].value == pytest.approx(1.0, abs=0.01)

    def test_sweep_custom_steps(self, analyzer):
        result = analyzer.sweep("max_depth", 1, 10, steps=3)
        assert len(result.points) == 3

    def test_sweep_invalid_param(self, analyzer):
        with pytest.raises(ValueError, match="Unknown parameter"):
            analyzer.sweep("nonexistent", 1, 5)

    def test_sweep_points_have_metrics(self, analyzer):
        result = analyzer.sweep("max_depth", 1, 3)
        for p in result.points:
            assert isinstance(p.metrics, dict)
            assert len(p.metrics) > 0


# ── Metric extractors ───────────────────────────────────────────────

class TestMetricExtractors:
    def test_all_extractors_exist(self):
        a = WhatIfAnalyzer()
        for name, (extractor, *_) in METRICS.items():
            assert hasattr(a, extractor), f"Missing extractor: {extractor}"

    def test_peak_workers_positive(self, analyzer):
        from src.replication.simulator import Simulator
        sim = Simulator(ScenarioConfig(seed=42))
        report = sim.run()
        result = WhatIfAnalyzer._m_peak_workers(report)
        assert result > 0

    def test_max_depth_non_negative(self, analyzer):
        from src.replication.simulator import Simulator
        sim = Simulator(ScenarioConfig(seed=42))
        report = sim.run()
        result = WhatIfAnalyzer._m_max_depth(report)
        assert result >= 0

    def test_denial_rate_bounded(self, analyzer):
        from src.replication.simulator import Simulator
        sim = Simulator(ScenarioConfig(seed=42))
        report = sim.run()
        result = WhatIfAnalyzer._m_denial_rate(report)
        assert 0 <= result <= 100

    def test_replication_rate_bounded(self, analyzer):
        from src.replication.simulator import Simulator
        sim = Simulator(ScenarioConfig(seed=42))
        report = sim.run()
        result = WhatIfAnalyzer._m_repl_rate(report)
        assert 0 <= result <= 100


# ── Serialization ────────────────────────────────────────────────────

class TestSerialization:
    def test_result_to_dict(self, analyzer):
        result = analyzer.analyze(changes={"max_depth": 5})
        d = result.to_dict()
        assert "baseline" in d
        assert "analyses" in d
        assert "ranking" in d
        assert "elapsed_ms" in d

    def test_result_json_serializable(self, analyzer):
        result = analyzer.analyze(changes={"max_depth": 5})
        text = json.dumps(result.to_dict())
        assert isinstance(text, str)

    def test_sweep_to_dict(self, analyzer):
        result = analyzer.sweep("max_depth", 1, 5)
        d = result.to_dict()
        assert d["parameter"] == "max_depth"
        assert len(d["points"]) == 5

    def test_sweep_json_serializable(self, analyzer):
        result = analyzer.sweep("max_depth", 1, 5)
        text = json.dumps(result.to_dict())
        assert isinstance(text, str)

    def test_delta_to_dict(self):
        d = MetricDelta(
            name="peak_workers", display_name="Peak Workers",
            baseline_value=10, variant_value=20,
            absolute_change=10, percent_change=100.0,
            polarity=MetricPolarity.LOWER_IS_BETTER, worsened=True,
        )
        result = d.to_dict()
        assert result["name"] == "peak_workers"
        assert result["worsened"] is True


# ── Rendering ────────────────────────────────────────────────────────

class TestRendering:
    def test_result_render(self, analyzer):
        result = analyzer.analyze(changes={"max_depth": 5})
        text = result.render()
        assert "WHAT-IF ANALYSIS" in text
        assert "max_depth" in text

    def test_sweep_render(self, analyzer):
        result = analyzer.sweep("max_depth", 1, 5)
        text = result.render()
        assert "PARAMETER SWEEP" in text
        assert "baseline" in text

    def test_multi_analysis_renders_ranking(self, analyzer):
        result = analyzer.analyze(change_sets=[
            {"max_depth": 5},
            {"max_replicas": 2},
        ])
        text = result.render()
        assert "RISK RANKING" in text

    def test_single_analysis_skips_ranking(self, analyzer):
        result = analyzer.analyze(changes={"max_depth": 5})
        text = result.render()
        assert "RISK RANKING" not in text


# ── CLI parsing ──────────────────────────────────────────────────────

class TestCLIParsing:
    def test_parse_int_change(self):
        k, v = _parse_change("max_depth=5")
        assert k == "max_depth"
        assert v == 5

    def test_parse_float_change(self):
        k, v = _parse_change("cooldown_seconds=2.5")
        assert k == "cooldown_seconds"
        assert v == 2.5

    def test_parse_string_change(self):
        k, v = _parse_change("strategy=conservative")
        assert k == "strategy"
        assert v == "conservative"

    def test_parse_none_change(self):
        k, v = _parse_change("expiration_seconds=none")
        assert k == "expiration_seconds"
        assert v is None

    def test_parse_invalid_format(self):
        import argparse
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_change("bad")

    def test_parse_unknown_param(self):
        import argparse
        with pytest.raises(argparse.ArgumentTypeError, match="Unknown param"):
            _parse_change("unknown_param=5")


# ── Reproducibility ─────────────────────────────────────────────────

class TestReproducibility:
    def test_same_seed_same_result(self):
        a1 = WhatIfAnalyzer(seed=42, runs_per_variant=1)
        a2 = WhatIfAnalyzer(seed=42, runs_per_variant=1)
        r1 = a1.analyze(changes={"max_depth": 5})
        r2 = a2.analyze(changes={"max_depth": 5})
        assert r1.analyses[0].risk_score == r2.analyses[0].risk_score

    def test_different_seed_may_differ(self):
        # With 1 run, different seeds should give different results
        a1 = WhatIfAnalyzer(seed=42, runs_per_variant=1)
        a2 = WhatIfAnalyzer(seed=99, runs_per_variant=1)
        r1 = a1.analyze(changes={"max_depth": 6})
        r2 = a2.analyze(changes={"max_depth": 6})
        # They might happen to be equal, so just check they both return results
        assert r1.analyses[0].risk_score >= 0
        assert r2.analyses[0].risk_score >= 0


# ── Averaging ────────────────────────────────────────────────────────

class TestAveraging:
    def test_multi_run_averages(self, multi_run_analyzer):
        result = multi_run_analyzer.analyze(changes={"max_depth": 5})
        # Should still produce valid results
        assert len(result.analyses) == 1
        score = result.analyses[0].risk_score
        assert 0 <= score <= 100


# ── Edge cases ───────────────────────────────────────────────────────

class TestEdgeCases:
    def test_no_change_analysis(self, analyzer):
        # Same values as baseline → no impact
        result = analyzer.analyze(changes={"max_depth": 3})
        assert result.analyses[0].risk_score == 0.0
        assert result.analyses[0].verdict == RiskVerdict.SAFE

    def test_very_restrictive_config(self, analyzer):
        result = analyzer.analyze(changes={
            "max_depth": 1,
            "max_replicas": 1,
            "replication_probability": 0.01,
        })
        assert result.analyses[0].verdict == RiskVerdict.SAFE

    def test_all_mutable_params_accepted(self, analyzer):
        # Each mutable param should be accepted without error
        for param in MUTABLE_PARAMS:
            if param == "strategy":
                analyzer.analyze(changes={param: "greedy"})
            elif param == "expiration_seconds":
                analyzer.analyze(changes={param: 60.0})
            elif MUTABLE_PARAMS[param] is int:
                analyzer.analyze(changes={param: 1})
            else:
                analyzer.analyze(changes={param: 0.5})

    def test_elapsed_ms_positive(self, analyzer):
        result = analyzer.analyze(changes={"max_depth": 5})
        assert result.elapsed_ms >= 0

    def test_sweep_elapsed_ms_positive(self, analyzer):
        result = analyzer.sweep("max_depth", 1, 3)
        assert result.elapsed_ms >= 0
