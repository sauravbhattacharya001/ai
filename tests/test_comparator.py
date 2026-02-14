"""Tests for the comparison runner."""

import json

from replication.comparator import (
    Comparator,
    ComparisonResult,
    RunResult,
)
from replication.simulator import PRESETS, ScenarioConfig, Strategy


def test_compare_all_strategies():
    """Default compare should run all five strategies."""
    comp = Comparator(ScenarioConfig(max_depth=2, max_replicas=5, cooldown_seconds=0.0))
    result = comp.compare_strategies(seed=42)
    assert len(result.runs) == 5
    assert set(result.labels) == {s.value for s in Strategy}
    for run in result.runs:
        assert len(run.report.workers) >= 1


def test_compare_specific_strategies():
    """Should only run the requested strategies."""
    comp = Comparator(ScenarioConfig(max_depth=2, max_replicas=5, cooldown_seconds=0.0))
    result = comp.compare_strategies(["greedy", "conservative"], seed=42)
    assert len(result.runs) == 2
    assert result.labels == ["greedy", "conservative"]


def test_compare_presets():
    """Should run all built-in presets."""
    comp = Comparator()
    result = comp.compare_presets()
    assert len(result.runs) == len(PRESETS)
    for run in result.runs:
        assert run.label in PRESETS


def test_compare_specific_presets():
    """Should only run requested presets."""
    comp = Comparator()
    result = comp.compare_presets(["minimal", "stress"])
    assert len(result.runs) == 2
    assert set(result.labels) == {"minimal", "stress"}


def test_invalid_preset_raises():
    """Unknown preset should raise ValueError."""
    comp = Comparator()
    try:
        comp.compare_presets(["nonexistent"])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "nonexistent" in str(e)


def test_sweep_max_depth():
    """Sweep max_depth across values."""
    comp = Comparator(ScenarioConfig(max_replicas=5, strategy="greedy", cooldown_seconds=0.0))
    result = comp.sweep("max_depth", [1, 2, 3], seed=42)
    assert len(result.runs) == 3
    assert result.swept_param == "max_depth"
    assert "max_depth=1" in result.labels
    assert "max_depth=2" in result.labels
    assert "max_depth=3" in result.labels


def test_sweep_max_replicas():
    """Sweep max_replicas across values."""
    comp = Comparator(ScenarioConfig(max_depth=2, strategy="greedy", cooldown_seconds=0.0))
    result = comp.sweep("max_replicas", [3, 5, 10], seed=42)
    assert len(result.runs) == 3
    # More replicas allowed should generally mean more workers
    workers = [len(r.report.workers) for r in result.runs]
    assert workers[-1] >= workers[0]


def test_sweep_invalid_param_raises():
    """Sweeping an invalid parameter should raise ValueError."""
    comp = Comparator()
    try:
        comp.sweep("invalid_param", [1, 2, 3])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "invalid_param" in str(e)


def test_compare_configs():
    """Compare arbitrary named configs."""
    configs = {
        "tight": ScenarioConfig(max_depth=1, max_replicas=2, strategy="greedy", cooldown_seconds=0.0),
        "loose": ScenarioConfig(max_depth=4, max_replicas=20, strategy="greedy", cooldown_seconds=0.0),
    }
    comp = Comparator()
    result = comp.compare_configs(configs)
    assert len(result.runs) == 2
    assert set(result.labels) == {"tight", "loose"}
    # Loose should have more workers
    tight_workers = len(next(r for r in result.runs if r.label == "tight").report.workers)
    loose_workers = len(next(r for r in result.runs if r.label == "loose").report.workers)
    assert loose_workers >= tight_workers


def test_render_table():
    """Table rendering should include all labels and metrics."""
    comp = Comparator(ScenarioConfig(max_depth=2, max_replicas=5, cooldown_seconds=0.0))
    result = comp.compare_strategies(["greedy", "burst"], seed=42)
    table = result.render_table()
    assert "greedy" in table
    assert "burst" in table
    assert "Workers" in table
    assert "Tasks" in table


def test_render_rankings():
    """Rankings should show medals and overall scores."""
    comp = Comparator(ScenarioConfig(max_depth=2, max_replicas=5, cooldown_seconds=0.0))
    result = comp.compare_strategies(["greedy", "conservative", "burst"], seed=42)
    rankings = result.render_rankings()
    assert "Rankings" in rankings
    assert "Overall Score" in rankings
    # Should have medal emojis
    assert "🥇" in rankings


def test_render_insights():
    """Insights should identify key patterns."""
    comp = Comparator(ScenarioConfig(max_depth=3, max_replicas=10, cooldown_seconds=0.0))
    result = comp.compare_strategies(["greedy", "conservative"], seed=42)
    insights = result.render_insights()
    assert "Key Insights" in insights
    assert "Most prolific" in insights or "Most efficient" in insights


def test_render_full():
    """Full render should combine all sections."""
    comp = Comparator(ScenarioConfig(max_depth=2, max_replicas=5, cooldown_seconds=0.0))
    result = comp.compare_strategies(["greedy", "chain"], seed=42)
    full = result.render()
    assert "Comparison" in full
    assert "greedy" in full
    assert "chain" in full


def test_to_dict_json_serializable():
    """to_dict should be fully JSON-serializable."""
    comp = Comparator(ScenarioConfig(max_depth=2, max_replicas=5, cooldown_seconds=0.0))
    result = comp.compare_strategies(["greedy", "burst"], seed=42)
    d = result.to_dict()
    json_str = json.dumps(d, default=str)
    parsed = json.loads(json_str)
    assert parsed["title"] == "Strategy Comparison"
    assert len(parsed["runs"]) == 2
    for run in parsed["runs"]:
        assert "label" in run
        assert "config" in run
        assert "metrics" in run
        assert "full_report" in run


def test_sweep_with_seed_reproducible():
    """Same seed should produce identical sweep results."""
    comp = Comparator(ScenarioConfig(max_depth=3, max_replicas=10, cooldown_seconds=0.0))
    r1 = comp.sweep("max_depth", [1, 2, 3], seed=99)
    r2 = comp.sweep("max_depth", [1, 2, 3], seed=99)
    for run1, run2 in zip(r1.runs, r2.runs):
        assert len(run1.report.workers) == len(run2.report.workers)
        assert run1.report.total_tasks == run2.report.total_tasks


def test_single_run_no_rankings():
    """With only one run, rankings should be empty."""
    comp = Comparator(ScenarioConfig(max_depth=2, max_replicas=5, cooldown_seconds=0.0))
    result = comp.compare_strategies(["greedy"], seed=42)
    assert len(result.runs) == 1
    assert result.render_rankings() == ""
    assert result.render_insights() == ""


def test_metric_table_values():
    """Metric table should compute correct values."""
    comp = Comparator(ScenarioConfig(
        max_depth=2, max_replicas=5, strategy="greedy",
        tasks_per_worker=2, cooldown_seconds=0.0,
    ))
    result = comp.compare_strategies(["greedy"], seed=42)
    rows = result._metric_table()
    assert len(rows) == 1
    row = rows[0]
    assert row["label"] == "greedy"
    assert row["workers"] >= 1
    assert row["tasks"] >= 1
    assert row["efficiency"] > 0
    assert 0 <= row["success_rate"] <= 100


def test_sweep_cooldown():
    """Sweep cooldown — higher cooldown should generally slow replication."""
    comp = Comparator(ScenarioConfig(max_depth=3, max_replicas=10, strategy="greedy"))
    result = comp.sweep("cooldown_seconds", [0.0, 0.1, 1.0], seed=42)
    assert len(result.runs) == 3
    # With high cooldown, fewer replications should succeed
    low_cooldown = result.runs[0].report.total_replications_succeeded
    high_cooldown = result.runs[-1].report.total_replications_succeeded
    assert low_cooldown >= high_cooldown
