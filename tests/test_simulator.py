"""Tests for the simulation runner."""

import json

from replication.simulator import (
    Simulator,
    ScenarioConfig,
    SimulationReport,
    Strategy,
    PRESETS,
)


def test_default_simulation_runs():
    """Default config should produce a valid report."""
    sim = Simulator()
    report = sim.run()
    assert isinstance(report, SimulationReport)
    assert len(report.workers) >= 1
    assert report.root_id in report.workers
    assert report.total_tasks > 0
    assert report.duration_ms >= 0


def test_greedy_strategy_maximizes_workers():
    """Greedy strategy should attempt replication on every task."""
    config = ScenarioConfig(
        max_depth=2, max_replicas=10, strategy="greedy",
        tasks_per_worker=2, cooldown_seconds=0.0, seed=42,
    )
    sim = Simulator(config)
    report = sim.run()
    # Greedy should spawn at least some children
    assert report.total_replications_succeeded > 0
    assert len(report.workers) > 1


def test_conservative_strategy_limits_replication():
    """Conservative strategy should only replicate at shallow depths."""
    config = ScenarioConfig(
        max_depth=4, max_replicas=20, strategy="conservative",
        tasks_per_worker=2, cooldown_seconds=0.0, seed=42,
    )
    sim = Simulator(config)
    report = sim.run()
    # Conservative: depth < max_depth // 2 = 2, so only depth 0 and 1 should replicate
    for wid, rec in report.workers.items():
        if rec.depth >= 2:
            assert rec.replications_succeeded == 0


def test_chain_strategy_produces_linear_tree():
    """Chain strategy should produce a linear sequence of workers."""
    config = ScenarioConfig(
        max_depth=4, max_replicas=10, strategy="chain",
        tasks_per_worker=1, cooldown_seconds=0.0,
    )
    sim = Simulator(config)
    report = sim.run()
    # Each worker should have at most one child
    for rec in report.workers.values():
        assert len(rec.children) <= 1


def test_burst_strategy_only_root_spawns():
    """Burst strategy should only have root spawning children."""
    config = ScenarioConfig(
        max_depth=1, max_replicas=5, strategy="burst",
        tasks_per_worker=3, cooldown_seconds=0.0,
    )
    sim = Simulator(config)
    report = sim.run()
    root = report.workers[report.root_id]
    # Root should have children
    assert root.replications_attempted > 0
    # No children should replicate (depth > 0)
    for child_id in root.children:
        child = report.workers[child_id]
        assert child.replications_attempted == 0


def test_random_strategy_with_seed_is_reproducible():
    """Same seed should produce identical results."""
    config = ScenarioConfig(
        max_depth=3, max_replicas=10, strategy="random",
        tasks_per_worker=2, replication_probability=0.5,
        cooldown_seconds=0.0, seed=12345,
    )
    report1 = Simulator(config).run()
    report2 = Simulator(config).run()
    assert report1.total_replications_succeeded == report2.total_replications_succeeded
    assert report1.total_tasks == report2.total_tasks
    assert len(report1.workers) == len(report2.workers)


def test_quota_enforcement():
    """Workers over quota should be denied."""
    config = ScenarioConfig(
        max_depth=10, max_replicas=3, strategy="greedy",
        tasks_per_worker=3, cooldown_seconds=0.0,
    )
    sim = Simulator(config)
    report = sim.run()
    # Should have denials due to quota
    assert report.total_replications_denied > 0


def test_render_tree():
    """Tree rendering should include all workers."""
    config = ScenarioConfig(
        max_depth=2, max_replicas=5, strategy="greedy",
        tasks_per_worker=1, cooldown_seconds=0.0,
    )
    sim = Simulator(config)
    report = sim.run()
    tree = report.render_tree()
    assert "Worker Lineage Tree" in tree
    assert report.root_id in tree
    for wid in report.workers:
        assert wid in tree


def test_render_timeline():
    """Timeline should contain spawn and task events."""
    config = ScenarioConfig(
        max_depth=1, max_replicas=3, strategy="greedy",
        tasks_per_worker=1, cooldown_seconds=0.0,
    )
    sim = Simulator(config)
    report = sim.run()
    timeline = report.render_timeline()
    assert "Simulation Timeline" in timeline
    assert "Root worker created" in timeline


def test_render_summary():
    """Summary should show key statistics."""
    config = ScenarioConfig(max_depth=2, max_replicas=5, strategy="greedy",
                            tasks_per_worker=1, cooldown_seconds=0.0)
    sim = Simulator(config)
    report = sim.run()
    summary = report.render_summary()
    assert "Simulation Summary" in summary
    assert "greedy" in summary
    assert "Total Workers:" in summary
    assert "Depth Distribution:" in summary


def test_render_full():
    """Full render should combine summary, tree, and timeline."""
    config = ScenarioConfig(max_depth=1, max_replicas=3, strategy="burst",
                            tasks_per_worker=1, cooldown_seconds=0.0)
    sim = Simulator(config)
    report = sim.run()
    full = report.render()
    assert "Simulation Summary" in full
    assert "Worker Lineage Tree" in full
    assert "Simulation Timeline" in full


def test_to_dict_is_json_serializable():
    """to_dict output should be JSON-serializable."""
    config = ScenarioConfig(max_depth=2, max_replicas=5, strategy="random",
                            tasks_per_worker=1, cooldown_seconds=0.0, seed=1)
    sim = Simulator(config)
    report = sim.run()
    d = report.to_dict()
    # Should not raise
    json_str = json.dumps(d, default=str)
    parsed = json.loads(json_str)
    assert parsed["config"]["strategy"] == "random"
    assert "workers" in parsed
    assert "timeline" in parsed


def test_presets_exist():
    """All preset scenarios should be valid and runnable."""
    for name, preset in PRESETS.items():
        sim = Simulator(preset)
        report = sim.run()
        assert len(report.workers) >= 1, f"Preset {name} failed"


def test_zero_tasks_worker():
    """Workers with 0 tasks should still spawn and shutdown."""
    config = ScenarioConfig(
        max_depth=1, max_replicas=3, strategy="greedy",
        tasks_per_worker=0, cooldown_seconds=0.0,
    )
    sim = Simulator(config)
    report = sim.run()
    # Root created but did 0 tasks
    assert report.total_tasks == 0
    assert len(report.workers) == 1


def test_strategy_enum_parsing():
    """Strategy enum should parse from string correctly."""
    for s in ["greedy", "conservative", "random", "chain", "burst"]:
        config = ScenarioConfig(strategy=s)
        assert config.strategy_enum == Strategy(s)
