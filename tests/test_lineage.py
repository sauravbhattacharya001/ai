"""Tests for LineageTracker -- replication provenance analysis."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from replication.lineage import (
    GenerationStats,
    LineageAnomaly,
    LineageChain,
    LineageNode,
    LineageReport,
    LineageSeverity,
    LineageTracker,
    StateMutation,
)
from replication.simulator import (
    ScenarioConfig,
    SimulationReport,
    Simulator,
    WorkerRecord,
)


# ── Helpers ────────────────────────────────────────────────────

def make_worker(worker_id, parent_id=None, depth=0, children=None,
                tasks=2, repl_attempted=1, repl_succeeded=1,
                repl_denied=0, created_at=None, shutdown_at=None,
                shutdown_reason=""):
    return WorkerRecord(
        worker_id=worker_id,
        parent_id=parent_id,
        depth=depth,
        tasks_completed=tasks,
        replications_attempted=repl_attempted,
        replications_succeeded=repl_succeeded,
        replications_denied=repl_denied,
        children=children or [],
        created_at=created_at,
        shutdown_at=shutdown_at,
        shutdown_reason=shutdown_reason,
    )


def build_linear_tree(length=4):
    """Build a linear chain: root -> w1 -> w2 -> ..."""
    workers = {}
    prev_id = None
    for i in range(length):
        wid = "w%d" % i if i > 0 else "root"
        children = ["w%d" % (i + 1)] if i < length - 1 else []
        workers[wid] = make_worker(
            wid, parent_id=prev_id, depth=i, children=children,
            created_at=float(i), shutdown_at=float(i + 1),
        )
        prev_id = wid
    return workers


def build_fan_tree(fan_out=3):
    """Build a fan tree: root with fan_out children, no grandchildren."""
    children = ["c%d" % i for i in range(fan_out)]
    workers = {
        "root": make_worker("root", depth=0, children=children,
                            created_at=0.0, shutdown_at=1.0),
    }
    for i, cid in enumerate(children):
        workers[cid] = make_worker(
            cid, parent_id="root", depth=1,
            created_at=0.1 * (i + 1), shutdown_at=1.0,
            repl_attempted=0, repl_succeeded=0,
        )
    return workers


def build_binary_tree(depth=3):
    """Build a complete binary tree of given depth."""
    workers = {}
    counter = [0]

    def _build(parent_id, d):
        if d == 0:
            wid = "root"
        else:
            counter[0] += 1
            wid = "w%d" % counter[0]

        children_ids = []
        if d < depth:
            for _ in range(2):
                child_id = _build(wid, d + 1)
                children_ids.append(child_id)

        workers[wid] = make_worker(
            wid, parent_id=parent_id, depth=d, children=children_ids,
            created_at=float(d), shutdown_at=float(d + 1),
        )
        return wid

    _build(None, 0)
    return workers


# ── Constructor ────────────────────────────────────────────────

class TestConstructor:
    def test_empty_workers_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            LineageTracker({}, "root")

    def test_invalid_root_raises(self):
        workers = {"w1": make_worker("w1")}
        with pytest.raises(ValueError, match="not found"):
            LineageTracker(workers, "nonexistent")

    def test_single_worker(self):
        workers = {"root": make_worker("root")}
        tracker = LineageTracker(workers, "root")
        report = tracker.analyze()
        assert report.total_workers == 1
        assert report.total_generations == 1
        assert report.leaf_count == 1
        assert report.internal_count == 0

    def test_from_simulation(self):
        config = ScenarioConfig(
            max_depth=2, max_replicas=5, strategy="greedy", seed=42,
        )
        sim = Simulator(config)
        sim_report = sim.run()
        tracker = LineageTracker.from_simulation(sim_report)
        report = tracker.analyze()
        assert report.total_workers > 0
        assert report.root_id == sim_report.root_id


# ── Query methods ──────────────────────────────────────────────

class TestQueries:
    def test_get_ancestors_root(self):
        workers = build_linear_tree(3)
        tracker = LineageTracker(workers, "root")
        assert tracker.get_ancestors("root") == ["root"]

    def test_get_ancestors_deep(self):
        workers = build_linear_tree(4)
        tracker = LineageTracker(workers, "root")
        ancestors = tracker.get_ancestors("w3")
        assert ancestors == ["root", "w1", "w2", "w3"]

    def test_get_descendants_root(self):
        workers = build_linear_tree(3)
        tracker = LineageTracker(workers, "root")
        descendants = tracker.get_descendants("root")
        assert "w1" in descendants
        assert "w2" in descendants
        assert "root" not in descendants

    def test_get_descendants_leaf(self):
        workers = build_linear_tree(3)
        tracker = LineageTracker(workers, "root")
        assert tracker.get_descendants("w2") == []

    def test_get_descendants_nonexistent(self):
        workers = build_linear_tree(3)
        tracker = LineageTracker(workers, "root")
        assert tracker.get_descendants("nonexistent") == []

    def test_get_siblings(self):
        workers = build_fan_tree(4)
        tracker = LineageTracker(workers, "root")
        siblings = tracker.get_siblings("c0")
        assert "c1" in siblings
        assert "c2" in siblings
        assert "c3" in siblings
        assert "c0" not in siblings

    def test_get_siblings_root(self):
        workers = build_fan_tree(3)
        tracker = LineageTracker(workers, "root")
        assert tracker.get_siblings("root") == []

    def test_get_generation(self):
        workers = build_fan_tree(3)
        tracker = LineageTracker(workers, "root")
        gen0 = tracker.get_generation(0)
        gen1 = tracker.get_generation(1)
        assert gen0 == ["root"]
        assert len(gen1) == 3

    def test_common_ancestor_same(self):
        workers = build_linear_tree(3)
        tracker = LineageTracker(workers, "root")
        assert tracker.common_ancestor("w1", "w1") == "w1"

    def test_common_ancestor_siblings(self):
        workers = build_fan_tree(3)
        tracker = LineageTracker(workers, "root")
        assert tracker.common_ancestor("c0", "c2") == "root"

    def test_common_ancestor_parent_child(self):
        workers = build_linear_tree(3)
        tracker = LineageTracker(workers, "root")
        assert tracker.common_ancestor("root", "w2") == "root"


# ── Analysis ──────────────────────────────────────────────────

class TestAnalysis:
    def test_linear_chain_length(self):
        workers = build_linear_tree(5)
        tracker = LineageTracker(workers, "root")
        report = tracker.analyze()
        assert report.longest_chain == 5
        assert report.total_generations == 5
        assert len(report.chains) == 1

    def test_fan_tree_stats(self):
        workers = build_fan_tree(5)
        tracker = LineageTracker(workers, "root")
        report = tracker.analyze()
        assert report.total_workers == 6
        assert report.leaf_count == 5
        assert report.internal_count == 1
        assert report.widest_generation == 5

    def test_binary_tree_generations(self):
        workers = build_binary_tree(3)
        tracker = LineageTracker(workers, "root")
        report = tracker.analyze()
        assert report.total_generations == 4
        assert report.total_workers == 15

    def test_chains_sorted_by_length(self):
        workers = build_binary_tree(2)
        tracker = LineageTracker(workers, "root")
        report = tracker.analyze()
        for i in range(len(report.chains) - 1):
            assert report.chains[i].length >= report.chains[i + 1].length

    def test_generation_stats_cover_all(self):
        workers = build_binary_tree(2)
        tracker = LineageTracker(workers, "root")
        report = tracker.analyze()
        total_from_stats = sum(
            gs.worker_count for gs in report.generation_stats
        )
        assert total_from_stats == report.total_workers

    def test_mutation_detection(self):
        workers = {
            "root": make_worker("root", children=["c1"], tasks=5,
                                repl_attempted=3, repl_succeeded=2),
            "c1": make_worker("c1", parent_id="root", depth=1, tasks=1,
                              repl_attempted=0, repl_succeeded=0),
        }
        tracker = LineageTracker(workers, "root")
        report = tracker.analyze()
        assert report.total_mutations_detected > 0
        assert len(report.mutations) > 0
        assert report.mutations[0].parent_id == "root"
        assert report.mutations[0].child_id == "c1"

    def test_mutation_rate(self):
        workers = build_linear_tree(3)
        tracker = LineageTracker(workers, "root")
        report = tracker.analyze()
        assert 0.0 <= report.mutation_rate <= 1.0


# ── Anomaly detection ─────────────────────────────────────────

class TestAnomalies:
    def test_deep_chain_anomaly(self):
        workers = build_linear_tree(7)
        tracker = LineageTracker(workers, "root")
        report = tracker.analyze()
        deep_anomalies = [
            a for a in report.anomalies if a.category == "deep_chain"
        ]
        assert len(deep_anomalies) > 0
        assert deep_anomalies[0].severity == LineageSeverity.WARNING

    def test_no_critical_anomalies_small_tree(self):
        workers = build_fan_tree(2)
        tracker = LineageTracker(workers, "root")
        report = tracker.analyze()
        critical = [
            a for a in report.anomalies
            if a.severity == LineageSeverity.CRITICAL
        ]
        assert len(critical) == 0

    def test_rapid_spawn_detection(self):
        children = ["c%d" % i for i in range(4)]
        workers = {
            "root": make_worker("root", children=children, created_at=0.0),
        }
        for i, cid in enumerate(children):
            workers[cid] = make_worker(
                cid, parent_id="root", depth=1,
                created_at=0.001 * (i + 1),
            )
        tracker = LineageTracker(workers, "root")
        report = tracker.analyze()
        rapid = [a for a in report.anomalies if a.category == "rapid_spawn"]
        assert len(rapid) > 0

    def test_high_denial_chain(self):
        workers = build_linear_tree(3)
        for wid in workers:
            workers[wid].replications_denied = 5
        tracker = LineageTracker(workers, "root")
        report = tracker.analyze()
        denial_anomalies = [
            a for a in report.anomalies
            if a.category == "high_denial_chain"
        ]
        assert len(denial_anomalies) > 0


# ── Rendering & serialization ─────────────────────────────────

class TestRendering:
    def test_render_not_empty(self):
        workers = build_fan_tree(3)
        tracker = LineageTracker(workers, "root")
        report = tracker.analyze()
        rendered = report.render()
        assert "Lineage Analysis Report" in rendered
        assert "Total workers:" in rendered

    def test_render_includes_chains(self):
        workers = build_linear_tree(4)
        tracker = LineageTracker(workers, "root")
        report = tracker.analyze()
        rendered = report.render()
        assert "Lineage Chains" in rendered

    def test_to_json_valid(self):
        workers = build_fan_tree(3)
        tracker = LineageTracker(workers, "root")
        report = tracker.analyze()
        data = json.loads(report.to_json())
        assert "total_workers" in data
        assert "chains" in data
        assert "anomalies" in data
        assert data["total_workers"] == 4

    def test_to_dict_roundtrip(self):
        workers = build_binary_tree(2)
        tracker = LineageTracker(workers, "root")
        report = tracker.analyze()
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["total_workers"] == 7
        assert isinstance(d["generation_stats"], list)
        assert isinstance(d["chains"], list)


# ── Branching entropy ─────────────────────────────────────────

class TestEntropy:
    def test_fan_tree_has_entropy(self):
        fan = build_fan_tree(4)
        fan_tracker = LineageTracker(fan, "root")
        fan_report = fan_tracker.analyze()
        assert fan_report.branching_entropy > 0

    def test_single_node_zero_entropy(self):
        workers = {"root": make_worker("root")}
        tracker = LineageTracker(workers, "root")
        report = tracker.analyze()
        assert report.branching_entropy == 0.0


# ── Integration with Simulator ────────────────────────────────

class TestIntegration:
    @pytest.mark.parametrize("strategy", [
        "greedy", "conservative", "random", "chain", "burst",
    ])
    def test_all_strategies(self, strategy):
        config = ScenarioConfig(
            strategy=strategy, max_depth=3, max_replicas=10, seed=42,
        )
        sim = Simulator(config)
        report = sim.run()
        tracker = LineageTracker.from_simulation(report)
        lineage = tracker.analyze()
        assert lineage.total_workers > 0
        assert lineage.total_generations >= 1
        assert lineage.root_id == report.root_id

    def test_stress_scenario(self):
        config = ScenarioConfig(
            strategy="greedy", max_depth=4, max_replicas=20, seed=42,
        )
        sim = Simulator(config)
        report = sim.run()
        tracker = LineageTracker.from_simulation(report)
        lineage = tracker.analyze()
        assert lineage.total_workers > 1
        assert lineage.longest_chain >= 1
        rendered = lineage.render()
        assert len(rendered) > 100


# ── LineageNode properties ────────────────────────────────────

class TestLineageNode:
    def test_is_root(self):
        node = LineageNode("r", None, 0)
        assert node.is_root is True

    def test_is_not_root(self):
        node = LineageNode("c", "r", 1)
        assert node.is_root is False

    def test_is_leaf(self):
        node = LineageNode("l", "r", 1)
        assert node.is_leaf is True

    def test_is_not_leaf(self):
        node = LineageNode("p", None, 0, children=["c1"])
        assert node.is_leaf is False

    def test_lifespan(self):
        node = LineageNode("w", None, 0, created_at=1.0, shutdown_at=2.5)
        assert node.lifespan_ms == 1500.0

    def test_lifespan_none(self):
        node = LineageNode("w", None, 0)
        assert node.lifespan_ms is None

    def test_replication_success_rate(self):
        node = LineageNode("w", None, 0)
        node.replications_attempted = 4
        node.replications_succeeded = 3
        assert node.replication_success_rate == 0.75

    def test_replication_success_rate_zero(self):
        node = LineageNode("w", None, 0)
        assert node.replication_success_rate == 0.0
