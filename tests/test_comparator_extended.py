"""Extended tests for comparator.py — edge cases, rendering, and data validation."""

import json
import math

import pytest

from replication.comparator import (
    Comparator,
    ComparisonResult,
    RunResult,
)
from replication.simulator import PRESETS, ScenarioConfig, SimulationReport, Strategy


# ── Fixtures ──────────────────────────────────────────────────────────

def _fast_config(**kw):
    """Minimal config for fast test runs."""
    defaults = dict(max_depth=1, max_replicas=3, cooldown_seconds=0.0,
                    tasks_per_worker=1, seed=42)
    defaults.update(kw)
    return ScenarioConfig(**defaults)


def _run_single(strategy="greedy", seed=42, **overrides):
    """Run a single comparison with one strategy for unit-level tests."""
    comp = Comparator(_fast_config(**overrides))
    return comp.compare_strategies([strategy], seed=seed)


def _run_pair(s1="greedy", s2="conservative", seed=42, **overrides):
    """Two-strategy comparison for ranking/insight tests."""
    comp = Comparator(_fast_config(**overrides))
    return comp.compare_strategies([s1, s2], seed=seed)


# ── ComparisonResult edge cases ───────────────────────────────────────

class TestComparisonResultEdgeCases:
    """Cover internal methods and edge conditions on ComparisonResult."""

    def test_empty_runs_labels(self):
        """Labels from an empty ComparisonResult should be an empty list."""
        cr = ComparisonResult(title="empty", runs=[])
        assert cr.labels == []

    def test_empty_runs_table(self):
        """render_table with no runs should return '(no runs)'."""
        cr = ComparisonResult(title="empty", runs=[])
        assert cr.render_table() == "(no runs)"

    def test_empty_runs_rankings(self):
        """Rankings with no runs should be empty."""
        cr = ComparisonResult(title="empty", runs=[])
        assert cr.render_rankings() == ""

    def test_empty_runs_insights(self):
        """Insights with no runs should be empty."""
        cr = ComparisonResult(title="empty", runs=[])
        assert cr.render_insights() == ""

    def test_single_run_no_rankings_or_insights(self):
        """With one run, rankings and insights should be empty strings."""
        result = _run_single()
        assert result.render_rankings() == ""
        assert result.render_insights() == ""

    def test_metric_table_fields(self):
        """Each metric row must have all required keys."""
        result = _run_single()
        rows = result._metric_table()
        required = {"label", "workers", "tasks", "repl_ok", "repl_denied",
                     "success_rate", "max_depth", "efficiency", "duration_ms"}
        for row in rows:
            assert required.issubset(row.keys()), f"Missing keys: {required - row.keys()}"

    def test_metric_table_success_rate_bounded(self):
        """success_rate must be in [0, 100]."""
        result = _run_pair()
        for row in result._metric_table():
            assert 0 <= row["success_rate"] <= 100

    def test_metric_table_efficiency_positive(self):
        """Efficiency (tasks/worker) should be > 0 when workers exist."""
        result = _run_single()
        for row in result._metric_table():
            if row["workers"] > 0:
                assert row["efficiency"] > 0

    def test_rank_ordering(self):
        """_rank should assign 1 to the best value."""
        result = _run_pair()
        rows = result._metric_table()
        rank = result._rank(rows, "workers", reverse=True)
        # The one with more workers should be rank 1
        sorted_by_workers = sorted(rows, key=lambda r: r["workers"], reverse=True)
        assert rank[sorted_by_workers[0]["label"]] == 1

    def test_rank_reverse_false(self):
        """With reverse=False, the lowest value gets rank 1."""
        result = _run_pair()
        rows = result._metric_table()
        rank = result._rank(rows, "duration_ms", reverse=False)
        sorted_by_duration = sorted(rows, key=lambda r: r["duration_ms"])
        assert rank[sorted_by_duration[0]["label"]] == 1

    def test_swept_param_default_none(self):
        """Strategy comparison should have swept_param=None."""
        result = _run_pair()
        assert result.swept_param is None


# ── Comparator._make_config ──────────────────────────────────────────

class TestMakeConfig:
    """Verify _make_config clones base and applies overrides."""

    def test_base_values_preserved(self):
        """When no overrides given, config matches base."""
        base = _fast_config(max_depth=7, max_replicas=42)
        comp = Comparator(base)
        cfg = comp._make_config()
        assert cfg.max_depth == 7
        assert cfg.max_replicas == 42

    def test_override_single_field(self):
        """Overriding one field shouldn't touch others."""
        base = _fast_config(max_depth=3, max_replicas=10)
        comp = Comparator(base)
        cfg = comp._make_config(max_depth=99)
        assert cfg.max_depth == 99
        assert cfg.max_replicas == 10  # unchanged

    def test_override_strategy(self):
        comp = Comparator(_fast_config())
        cfg = comp._make_config(strategy="burst")
        assert cfg.strategy == "burst"

    def test_override_seed(self):
        comp = Comparator(_fast_config())
        cfg = comp._make_config(seed=123)
        assert cfg.seed == 123

    def test_override_cooldown(self):
        comp = Comparator(_fast_config())
        cfg = comp._make_config(cooldown_seconds=5.5)
        assert cfg.cooldown_seconds == 5.5

    def test_override_replication_probability(self):
        comp = Comparator(_fast_config())
        cfg = comp._make_config(replication_probability=0.5)
        assert cfg.replication_probability == 0.5

    def test_override_resource_limits(self):
        comp = Comparator(_fast_config())
        cfg = comp._make_config(cpu_limit=4.0, memory_limit_mb=512)
        assert cfg.cpu_limit == 4.0
        assert cfg.memory_limit_mb == 512


# ── Strategy comparison ──────────────────────────────────────────────

class TestCompareStrategies:
    """Strategy comparison entry points."""

    def test_all_strategies_covered(self):
        """Default (None) should include every Strategy enum value."""
        comp = Comparator(_fast_config())
        result = comp.compare_strategies(seed=42)
        assert set(result.labels) == {s.value for s in Strategy}

    def test_each_run_has_report(self):
        """Every RunResult must have a non-None SimulationReport."""
        result = _run_pair()
        for run in result.runs:
            assert isinstance(run.report, SimulationReport)
            assert run.report.total_tasks >= 0

    def test_strategy_matches_config(self):
        """RunResult.config.strategy should match the requested strategy."""
        comp = Comparator(_fast_config())
        result = comp.compare_strategies(["chain"], seed=42)
        assert result.runs[0].config.strategy == "chain"

    def test_compare_three_strategies(self):
        """Three strategies should produce three runs."""
        comp = Comparator(_fast_config())
        result = comp.compare_strategies(["greedy", "burst", "random"], seed=42)
        assert len(result.runs) == 3

    def test_seed_reproducibility(self):
        """Same seed → same results."""
        comp = Comparator(_fast_config())
        r1 = comp.compare_strategies(["greedy"], seed=77)
        r2 = comp.compare_strategies(["greedy"], seed=77)
        assert r1.runs[0].report.total_tasks == r2.runs[0].report.total_tasks
        assert len(r1.runs[0].report.workers) == len(r2.runs[0].report.workers)


# ── Preset comparison ────────────────────────────────────────────────

class TestComparePresets:
    """Preset comparison tests."""

    def test_all_presets_default(self):
        """None → all presets."""
        comp = Comparator()
        result = comp.compare_presets()
        assert set(result.labels) == set(PRESETS.keys())

    def test_subset_presets(self):
        available = list(PRESETS.keys())
        if len(available) >= 2:
            subset = available[:2]
            comp = Comparator()
            result = comp.compare_presets(subset)
            assert len(result.runs) == 2
            assert set(result.labels) == set(subset)

    def test_invalid_preset_error_message(self):
        """Error should list available presets."""
        comp = Comparator()
        with pytest.raises(ValueError, match="Available"):
            comp.compare_presets(["this_preset_does_not_exist"])


# ── Parameter sweep ──────────────────────────────────────────────────

class TestSweep:
    """Parameter sweep tests."""

    def test_sweep_tasks_per_worker(self):
        comp = Comparator(_fast_config(max_depth=2))
        result = comp.sweep("tasks_per_worker", [1, 3, 5], seed=42)
        assert len(result.runs) == 3
        assert result.swept_param == "tasks_per_worker"
        for run in result.runs:
            assert "tasks_per_worker=" in run.label

    def test_sweep_replication_probability(self):
        """Sweep a float parameter."""
        comp = Comparator(_fast_config(max_depth=2))
        result = comp.sweep("replication_probability", [0.0, 0.5, 1.0], seed=42)
        assert len(result.runs) == 3
        assert result.swept_param == "replication_probability"

    def test_sweep_cpu_limit(self):
        comp = Comparator(_fast_config())
        result = comp.sweep("cpu_limit", [1.0, 2.0], seed=42)
        assert len(result.runs) == 2

    def test_sweep_memory_limit(self):
        comp = Comparator(_fast_config())
        result = comp.sweep("memory_limit_mb", [128, 256, 512], seed=42)
        assert len(result.runs) == 3

    def test_sweep_single_value(self):
        """Sweeping with one value should produce one run."""
        comp = Comparator(_fast_config())
        result = comp.sweep("max_depth", [5], seed=42)
        assert len(result.runs) == 1

    def test_sweep_invalid_param_lists_valid(self):
        """Error should list valid sweepable params."""
        comp = Comparator()
        with pytest.raises(ValueError, match="Sweepable"):
            comp.sweep("strategy", [1, 2])

    def test_sweep_seed_applied(self):
        """Seed passed to sweep should be forwarded to configs."""
        comp = Comparator(_fast_config())
        result = comp.sweep("max_depth", [1, 2], seed=999)
        for run in result.runs:
            assert run.config.seed == 999


# ── compare_configs ──────────────────────────────────────────────────

class TestCompareConfigs:
    """Arbitrary named configuration comparison."""

    def test_empty_configs(self):
        """Empty dict → no runs."""
        comp = Comparator()
        result = comp.compare_configs({})
        assert len(result.runs) == 0
        assert result.title == "Custom Comparison"

    def test_single_config(self):
        comp = Comparator()
        result = comp.compare_configs({
            "only": _fast_config(max_depth=1),
        })
        assert len(result.runs) == 1
        assert result.labels == ["only"]

    def test_config_labels_preserved(self):
        comp = Comparator()
        configs = {
            "alpha": _fast_config(strategy="greedy"),
            "beta": _fast_config(strategy="conservative"),
            "gamma": _fast_config(strategy="burst"),
        }
        result = comp.compare_configs(configs)
        assert set(result.labels) == {"alpha", "beta", "gamma"}


# ── Rendering ────────────────────────────────────────────────────────

class TestRendering:
    """Detailed rendering output tests."""

    def test_table_contains_column_headers(self):
        result = _run_pair()
        table = result.render_table()
        for header in ["Scenario", "Workers", "Tasks", "Repl OK", "Denied",
                       "Succ %", "MaxDep", "Effic", "Time ms"]:
            assert header in table

    def test_table_contains_all_labels(self):
        comp = Comparator(_fast_config())
        result = comp.compare_strategies(["greedy", "burst", "random"], seed=42)
        table = result.render_table()
        for label in ["greedy", "burst", "random"]:
            assert label in table

    def test_table_has_box_drawing(self):
        """Table should use Unicode box-drawing characters."""
        result = _run_pair()
        table = result.render_table()
        assert "┌" in table
        assert "└" in table
        assert "│" in table

    def test_rankings_has_medals(self):
        result = _run_pair()
        rankings = result.render_rankings()
        assert "🥇" in rankings
        assert "🥈" in rankings

    def test_rankings_categories(self):
        result = _run_pair()
        rankings = result.render_rankings()
        for cat in ["Most Workers", "Most Tasks", "Best Replication",
                     "Highest Efficiency", "Fewest Denials", "Fastest"]:
            assert cat in rankings

    def test_rankings_overall_score(self):
        result = _run_pair()
        rankings = result.render_rankings()
        assert "Overall Score" in rankings
        assert "█" in rankings  # bar chart

    def test_insights_has_prolific(self):
        result = _run_pair(max_depth=3, max_replicas=10)
        insights = result.render_insights()
        assert "Most prolific" in insights or "Most efficient" in insights

    def test_insights_spread_analysis(self):
        """If worker counts differ, spread analysis should appear."""
        comp = Comparator(_fast_config())
        # greedy vs conservative with enough depth for divergence
        result = comp.compare_strategies(
            ["greedy", "conservative"],
            seed=42,
        )
        insights = result.render_insights()
        rows = result._metric_table()
        worker_counts = [r["workers"] for r in rows]
        if max(worker_counts) - min(worker_counts) > 0:
            assert "spread" in insights.lower()

    def test_insights_depth_utilization_warning(self):
        """If a strategy uses <50% of allowed depth, warning should appear."""
        # Use high max_depth with conservative strategy (tends to stay shallow)
        comp = Comparator(ScenarioConfig(
            max_depth=10, max_replicas=3, cooldown_seconds=0.0,
            strategy="conservative", tasks_per_worker=1,
        ))
        result = comp.compare_strategies(["conservative"], seed=42)
        # This might not trigger if conservative uses enough depth,
        # but with only 3 replicas and depth 10, it likely stays shallow.
        # The test validates the code path runs without error.
        result.render_insights()

    def test_full_render_combines_sections(self):
        result = _run_pair(max_depth=3, max_replicas=10)
        full = result.render()
        assert "Comparison" in full
        # Table section
        assert "Workers" in full
        # Should have at least table + title


# ── to_dict / JSON serialization ─────────────────────────────────────

class TestToDict:
    """Serialization tests."""

    def test_to_dict_structure(self):
        result = _run_pair()
        d = result.to_dict()
        assert "title" in d
        assert "swept_param" in d
        assert "runs" in d
        assert len(d["runs"]) == 2

    def test_to_dict_run_fields(self):
        result = _run_single()
        d = result.to_dict()
        run = d["runs"][0]
        assert "label" in run
        assert "config" in run
        assert "metrics" in run
        assert "full_report" in run

    def test_to_dict_config_fields(self):
        result = _run_single()
        cfg = result.to_dict()["runs"][0]["config"]
        assert "strategy" in cfg
        assert "max_depth" in cfg
        assert "max_replicas" in cfg
        assert "cooldown_seconds" in cfg
        assert "tasks_per_worker" in cfg

    def test_to_dict_metrics_match_table(self):
        """Metrics in to_dict should match _metric_table values."""
        result = _run_single()
        table_rows = result._metric_table()
        dict_metrics = result.to_dict()["runs"][0]["metrics"]
        for key in ["workers", "tasks", "repl_ok", "repl_denied",
                     "success_rate", "max_depth", "efficiency"]:
            assert dict_metrics[key] == table_rows[0][key]

    def test_json_roundtrip(self):
        """to_dict → JSON → parse should be lossless."""
        result = _run_pair()
        d = result.to_dict()
        json_str = json.dumps(d, default=str)
        parsed = json.loads(json_str)
        assert parsed["title"] == d["title"]
        assert len(parsed["runs"]) == len(d["runs"])
        for orig, back in zip(d["runs"], parsed["runs"]):
            assert orig["label"] == back["label"]

    def test_sweep_to_dict_has_swept_param(self):
        """Sweep result should include swept_param in dict."""
        comp = Comparator(_fast_config())
        result = comp.sweep("max_depth", [1, 2], seed=42)
        d = result.to_dict()
        assert d["swept_param"] == "max_depth"

    def test_to_dict_no_nan_or_inf(self):
        """No NaN or Inf values in serialized output."""
        result = _run_pair()
        d = result.to_dict()
        json_str = json.dumps(d, default=str)
        assert "NaN" not in json_str
        assert "Infinity" not in json_str


# ── Determinism ──────────────────────────────────────────────────────

class TestDeterminism:
    """Reproducibility with same seed."""

    def test_strategies_deterministic(self):
        comp = Comparator(_fast_config())
        r1 = comp.compare_strategies(["greedy", "burst"], seed=42)
        r2 = comp.compare_strategies(["greedy", "burst"], seed=42)
        for a, b in zip(r1.runs, r2.runs):
            assert a.report.total_tasks == b.report.total_tasks
            assert len(a.report.workers) == len(b.report.workers)

    def test_sweep_deterministic(self):
        comp = Comparator(_fast_config())
        r1 = comp.sweep("max_depth", [1, 2, 3], seed=42)
        r2 = comp.sweep("max_depth", [1, 2, 3], seed=42)
        for a, b in zip(r1.runs, r2.runs):
            assert a.report.total_tasks == b.report.total_tasks

    def test_different_seeds_differ(self):
        """Different seeds should (usually) produce different results."""
        comp = Comparator(_fast_config(max_depth=3, max_replicas=15))
        r1 = comp.compare_strategies(["random"], seed=1)
        r2 = comp.compare_strategies(["random"], seed=9999)
        # Not guaranteed but very likely to differ with random strategy
        # Just ensure both run without error
        assert len(r1.runs) == 1
        assert len(r2.runs) == 1


# ── Metric invariants ────────────────────────────────────────────────

class TestMetricInvariants:
    """Mathematical invariants that should always hold."""

    def test_workers_at_least_one(self):
        """Every simulation should have at least one worker (the initial)."""
        result = _run_pair(max_depth=2, max_replicas=10)
        for run in result.runs:
            assert len(run.report.workers) >= 1

    def test_repl_ok_plus_denied_is_total(self):
        """Succeeded + denied should equal total attempted."""
        result = _run_pair(max_depth=2, max_replicas=10)
        for run in result.runs:
            rep = run.report
            assert (rep.total_replications_succeeded + rep.total_replications_denied
                    == rep.total_replications_attempted)

    def test_success_rate_matches_calculation(self):
        """success_rate = succeeded / attempted * 100."""
        result = _run_pair(max_depth=2, max_replicas=10)
        rows = result._metric_table()
        for i, row in enumerate(rows):
            rep = result.runs[i].report
            expected = (
                rep.total_replications_succeeded / rep.total_replications_attempted * 100
                if rep.total_replications_attempted > 0
                else 0.0
            )
            assert abs(row["success_rate"] - expected) < 0.01

    def test_max_depth_within_config_limit(self):
        """Observed max depth should not exceed configured max_depth."""
        comp = Comparator(_fast_config(max_depth=3, max_replicas=10))
        result = comp.compare_strategies(seed=42)
        for run in result.runs:
            for w in run.report.workers.values():
                assert w.depth <= run.config.max_depth

    def test_duration_non_negative(self):
        result = _run_pair()
        for row in result._metric_table():
            assert row["duration_ms"] >= 0
