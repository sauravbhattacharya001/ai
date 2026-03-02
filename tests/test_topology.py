"""Comprehensive tests for the TopologyAnalyzer module.

Covers: empty trees, single nodes, linear chains, balanced trees,
burst/explosion patterns, lopsided trees, deep narrow trees,
wide shallow trees, risk scoring, balance computation, compactness,
subtree risks, ASCII rendering, JSON export, controller integration.
"""

from datetime import datetime, timezone

import pytest

from replication.contract import (
    Manifest,
    ReplicationContract,
    ResourceSpec,
)
from replication.controller import Controller
from replication.observability import StructuredLogger
from replication.topology import (
    NodeMetrics,
    PathologicalPattern,
    RiskLevel,
    SubtreeRisk,
    TopologyAnalyzer,
    TopologyReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resources():
    return ResourceSpec(cpu_limit=0.5, memory_limit_mb=256)


def _contract(**kw):
    defaults = dict(max_depth=10, max_replicas=100, cooldown_seconds=0.0)
    defaults.update(kw)
    return ReplicationContract(**defaults)


def _analyze(workers, max_depth=10, max_replicas=100):
    """Shorthand to build analyzer and run analysis."""
    analyzer = TopologyAnalyzer(
        workers=workers,
        contract_max_depth=max_depth,
        contract_max_replicas=max_replicas,
    )
    return analyzer.analyze()


# ---------------------------------------------------------------------------
# Empty and single-node trees
# ---------------------------------------------------------------------------

class TestEmptyTree:
    def test_empty(self):
        report = _analyze([])
        assert report.total_workers == 0
        assert report.root_count == 0
        assert report.leaf_count == 0
        assert report.internal_count == 0
        assert report.max_depth == 0
        assert report.mean_depth == 0.0
        assert report.balance_score == 1.0
        assert report.compactness == 1.0
        assert report.risk_level == RiskLevel.LOW
        assert report.risk_score == 0.0
        assert report.patterns_detected == []
        assert report.warnings == []

    def test_empty_render_tree(self):
        report = _analyze([])
        assert report.render_tree() == "(empty tree)"


class TestSingleNode:
    def test_single_root(self):
        report = _analyze([("w1", None, 0)])
        assert report.total_workers == 1
        assert report.root_count == 1
        assert report.leaf_count == 1
        assert report.internal_count == 0
        assert report.max_depth == 0
        assert report.mean_depth == 0.0
        assert report.balance_score == 1.0
        assert report.compactness == 1.0

    def test_single_root_is_leaf(self):
        report = _analyze([("w1", None, 0)])
        nm = report.node_metrics[0]
        assert nm.is_leaf is True
        assert nm.child_count == 0
        assert nm.subtree_size == 1


# ---------------------------------------------------------------------------
# Linear chains
# ---------------------------------------------------------------------------

class TestLinearChain:
    def test_chain_of_3(self):
        workers = [("w1", None, 0), ("w2", "w1", 1), ("w3", "w2", 2)]
        report = _analyze(workers)
        assert report.total_workers == 3
        assert report.root_count == 1
        assert report.leaf_count == 1
        assert report.internal_count == 2
        assert report.max_depth == 2
        assert report.mean_branching_factor == 1.0
        assert report.max_branching_factor == 1

    def test_long_chain_detects_runaway(self):
        """Chain using >50% of max depth should trigger RUNAWAY_CHAIN."""
        n = 7
        workers = [("w0", None, 0)]
        for i in range(1, n):
            workers.append((f"w{i}", f"w{i-1}", i))
        report = _analyze(workers, max_depth=10)
        assert PathologicalPattern.RUNAWAY_CHAIN in report.patterns_detected

    def test_short_chain_no_runaway(self):
        workers = [("w0", None, 0), ("w1", "w0", 1), ("w2", "w1", 2)]
        report = _analyze(workers, max_depth=10)
        assert PathologicalPattern.RUNAWAY_CHAIN not in report.patterns_detected

    def test_chain_depth_distribution(self):
        workers = [("w0", None, 0), ("w1", "w0", 1), ("w2", "w1", 2)]
        report = _analyze(workers)
        assert report.depth_distribution == {0: 1, 1: 1, 2: 1}


# ---------------------------------------------------------------------------
# Balanced trees
# ---------------------------------------------------------------------------

class TestBalancedTree:
    def test_perfect_binary_tree(self):
        """Root with 2 children, each with 2 children = 7 nodes."""
        workers = [
            ("r", None, 0),
            ("a", "r", 1), ("b", "r", 1),
            ("c", "a", 2), ("d", "a", 2),
            ("e", "b", 2), ("f", "b", 2),
        ]
        report = _analyze(workers)
        assert report.total_workers == 7
        assert report.root_count == 1
        assert report.leaf_count == 4
        assert report.max_depth == 2
        assert report.mean_branching_factor == 2.0
        # Perfect binary tree should have high balance
        assert report.balance_score == 1.0

    def test_balanced_branching_distribution(self):
        workers = [
            ("r", None, 0),
            ("a", "r", 1), ("b", "r", 1),
            ("c", "a", 2), ("d", "a", 2),
            ("e", "b", 2), ("f", "b", 2),
        ]
        report = _analyze(workers)
        # All internal nodes have 2 children
        assert report.branching_distribution == {2: 3}


# ---------------------------------------------------------------------------
# Burst / explosion
# ---------------------------------------------------------------------------

class TestExplosion:
    def test_burst_detected(self):
        """Root spawns 5 children — qualifies as explosion (>50% of 6 total)."""
        workers = [("r", None, 0)]
        for i in range(5):
            workers.append((f"c{i}", "r", 1))
        report = _analyze(workers)
        assert PathologicalPattern.EXPLOSION in report.patterns_detected

    def test_small_burst_no_explosion(self):
        """2 children out of 3 total = 66%, but only 2 children, minimum is 3."""
        workers = [("r", None, 0), ("c0", "r", 1), ("c1", "r", 1)]
        report = _analyze(workers)
        assert PathologicalPattern.EXPLOSION not in report.patterns_detected

    def test_burst_wide_shallow(self):
        """Root spawns many children, no grandchildren → also WIDE_SHALLOW."""
        workers = [("r", None, 0)]
        for i in range(5):
            workers.append((f"c{i}", "r", 1))
        report = _analyze(workers)
        assert PathologicalPattern.WIDE_SHALLOW in report.patterns_detected


# ---------------------------------------------------------------------------
# Lopsided trees
# ---------------------------------------------------------------------------

class TestLopsided:
    def test_lopsided_detected(self):
        """One subtree has >80% of workers."""
        workers = [
            ("r", None, 0),
            ("a", "r", 1),  # small subtree (1 node)
            ("b", "r", 1),  # large subtree (5 nodes)
            ("c", "b", 2), ("d", "b", 2), ("e", "b", 2), ("f", "b", 2),
        ]
        # b's subtree = 5 out of 7 total = 71% — not enough
        # Need >80%
        workers_big = [
            ("r", None, 0),
            ("a", "r", 1),
            ("b", "r", 1),
            ("c", "b", 2), ("d", "b", 2), ("e", "b", 2), ("f", "b", 2),
            ("g", "c", 3), ("h", "c", 3), ("i", "d", 3),
        ]
        # b subtree = b,c,d,e,f,g,h,i = 8 out of 10 = 80% (need >80%)
        # Add one more
        workers_big.append(("j", "e", 3))
        # b subtree = 9 out of 11 = 81.8%
        report = _analyze(workers_big)
        assert PathologicalPattern.LOPSIDED in report.patterns_detected


# ---------------------------------------------------------------------------
# Deep narrow
# ---------------------------------------------------------------------------

class TestDeepNarrow:
    def test_deep_narrow_detected(self):
        """Depth > 70% of max with low branching (but branching > 1 to avoid
        being classified as RUNAWAY_CHAIN instead)."""
        # Main spine reaches depth 8 with occasional branch to avoid max_branching<=1
        workers = [
            ("w0", None, 0),
            ("w1", "w0", 1), ("w1b", "w0", 1),  # branching=2 here
            ("w2", "w1", 2),
            ("w3", "w2", 3),
            ("w4", "w3", 4),
            ("w5", "w4", 5),
            ("w6", "w5", 6),
            ("w7", "w6", 7),
            ("w8", "w7", 8),
        ]
        report = _analyze(workers, max_depth=10)
        assert PathologicalPattern.DEEP_NARROW in report.patterns_detected


# ---------------------------------------------------------------------------
# Wide shallow
# ---------------------------------------------------------------------------

class TestWideShallow:
    def test_wide_shallow_detected(self):
        """Many root children, max depth = 1."""
        workers = [("r", None, 0)]
        for i in range(4):
            workers.append((f"c{i}", "r", 1))
        report = _analyze(workers)
        assert PathologicalPattern.WIDE_SHALLOW in report.patterns_detected

    def test_deep_tree_not_wide_shallow(self):
        workers = [("r", None, 0), ("a", "r", 1), ("b", "a", 2), ("c", "b", 3)]
        report = _analyze(workers)
        assert PathologicalPattern.WIDE_SHALLOW not in report.patterns_detected


# ---------------------------------------------------------------------------
# Balance score
# ---------------------------------------------------------------------------

class TestBalanceScore:
    def test_single_child_balanced(self):
        workers = [("r", None, 0), ("a", "r", 1)]
        report = _analyze(workers)
        assert report.balance_score == 1.0

    def test_unbalanced_tree(self):
        """One subtree much larger than another → low balance."""
        workers = [
            ("r", None, 0),
            ("a", "r", 1),
            ("b", "r", 1),
            ("c", "b", 2), ("d", "b", 2), ("e", "b", 2),
        ]
        report = _analyze(workers)
        # a has subtree=1, b has subtree=4 → unbalanced
        assert report.balance_score < 0.9

    def test_equal_subtrees_perfect_balance(self):
        workers = [
            ("r", None, 0),
            ("a", "r", 1), ("b", "r", 1),
            ("c", "a", 2), ("d", "b", 2),
        ]
        report = _analyze(workers)
        assert report.balance_score == 1.0


# ---------------------------------------------------------------------------
# Compactness
# ---------------------------------------------------------------------------

class TestCompactness:
    def test_perfect_binary_tree_compact(self):
        workers = [
            ("r", None, 0),
            ("a", "r", 1), ("b", "r", 1),
            ("c", "a", 2), ("d", "a", 2),
            ("e", "b", 2), ("f", "b", 2),
        ]
        report = _analyze(workers)
        # 7 nodes, depth=2, log2(7)=2.81 → compactness = 2.81/2 > 1.0 → capped at 1.0
        assert report.compactness == 1.0

    def test_chain_low_compactness(self):
        workers = [("w0", None, 0)]
        for i in range(1, 8):
            workers.append((f"w{i}", f"w{i-1}", i))
        report = _analyze(workers)
        # 8 nodes, depth=7, log2(8)=3 → compactness = 3/7 ≈ 0.43
        assert report.compactness < 0.5


# ---------------------------------------------------------------------------
# Risk scoring
# ---------------------------------------------------------------------------

class TestRiskScoring:
    def test_small_safe_tree_low_risk(self):
        workers = [("r", None, 0), ("a", "r", 1)]
        report = _analyze(workers, max_depth=10, max_replicas=100)
        assert report.risk_level == RiskLevel.LOW
        assert report.risk_score < 0.25

    def test_near_quota_higher_risk(self):
        workers = [(f"w{i}", None, 0) for i in range(9)]
        report = _analyze(workers, max_depth=10, max_replicas=10)
        # 90% quota utilization → elevated risk
        assert report.risk_score > 0.2

    def test_patterns_increase_risk(self):
        # Chain near max depth
        workers = [("w0", None, 0)]
        for i in range(1, 8):
            workers.append((f"w{i}", f"w{i-1}", i))
        report = _analyze(workers, max_depth=10)
        # Patterns should add to risk score
        assert len(report.patterns_detected) > 0
        assert report.risk_score > 0.15

    def test_risk_classification_thresholds(self):
        analyzer = TopologyAnalyzer([], contract_max_depth=10, contract_max_replicas=100)
        assert analyzer._classify_risk(0.1) == RiskLevel.LOW
        assert analyzer._classify_risk(0.3) == RiskLevel.MODERATE
        assert analyzer._classify_risk(0.6) == RiskLevel.HIGH
        assert analyzer._classify_risk(0.8) == RiskLevel.CRITICAL

    def test_risk_capped_at_1(self):
        """Even extreme trees shouldn't exceed risk_score 1.0."""
        # Chain to max depth + near quota + all patterns
        workers = [(f"w{i}", f"w{i-1}" if i > 0 else None, i) for i in range(10)]
        report = _analyze(workers, max_depth=10, max_replicas=10)
        assert report.risk_score <= 1.0


# ---------------------------------------------------------------------------
# Subtree risk
# ---------------------------------------------------------------------------

class TestSubtreeRisk:
    def test_subtree_risks_computed(self):
        workers = [
            ("r", None, 0),
            ("a", "r", 1), ("b", "r", 1),
            ("c", "a", 2), ("d", "a", 2),
        ]
        report = _analyze(workers)
        assert len(report.subtree_risks) == 2  # a and b are root children

    def test_subtree_risks_sorted_by_risk(self):
        workers = [
            ("r", None, 0),
            ("a", "r", 1), ("b", "r", 1),
            ("c", "a", 2), ("d", "a", 2), ("e", "a", 2),
        ]
        report = _analyze(workers)
        # a has larger subtree → higher risk → should be first
        if len(report.subtree_risks) >= 2:
            assert report.subtree_risks[0].risk_score >= report.subtree_risks[1].risk_score


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------

class TestWarnings:
    def test_quota_warning(self):
        workers = [(f"w{i}", None, 0) for i in range(9)]
        report = _analyze(workers, max_replicas=10)
        assert any("Quota" in w or "quota" in w.lower() for w in report.warnings)

    def test_depth_warning(self):
        workers = [("w0", None, 0)]
        for i in range(1, 9):
            workers.append((f"w{i}", f"w{i-1}", i))
        report = _analyze(workers, max_depth=10)
        assert any("Depth" in w or "depth" in w.lower() for w in report.warnings)

    def test_no_warnings_safe_tree(self):
        workers = [("r", None, 0), ("a", "r", 1)]
        report = _analyze(workers, max_depth=10, max_replicas=100)
        assert len(report.warnings) == 0


# ---------------------------------------------------------------------------
# NodeMetrics
# ---------------------------------------------------------------------------

class TestNodeMetrics:
    def test_root_node_metrics(self):
        workers = [("r", None, 0), ("a", "r", 1), ("b", "r", 1)]
        report = _analyze(workers)
        root = next(nm for nm in report.node_metrics if nm.worker_id == "r")
        assert root.depth == 0
        assert root.child_count == 2
        assert root.subtree_size == 3
        assert root.parent_id is None
        assert root.is_leaf is False

    def test_leaf_node_metrics(self):
        workers = [("r", None, 0), ("a", "r", 1)]
        report = _analyze(workers)
        leaf = next(nm for nm in report.node_metrics if nm.worker_id == "a")
        assert leaf.depth == 1
        assert leaf.child_count == 0
        assert leaf.subtree_size == 1
        assert leaf.subtree_depth == 0
        assert leaf.is_leaf is True


# ---------------------------------------------------------------------------
# ASCII tree rendering
# ---------------------------------------------------------------------------

class TestRenderTree:
    def test_render_includes_all_nodes(self):
        workers = [
            ("root", None, 0),
            ("child1", "root", 1),
            ("child2", "root", 1),
        ]
        report = _analyze(workers)
        tree = report.render_tree()
        assert "root" in tree
        assert "child1" in tree
        assert "child2" in tree

    def test_render_shows_depth(self):
        workers = [("r", None, 0), ("a", "r", 1)]
        report = _analyze(workers)
        tree = report.render_tree()
        assert "d=0" in tree
        assert "d=1" in tree


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_contains_key_info(self):
        workers = [("r", None, 0), ("a", "r", 1)]
        report = _analyze(workers)
        s = report.summary()
        assert "2 workers" in s
        assert "risk=" in s

    def test_summary_mentions_patterns(self):
        workers = [("r", None, 0)]
        for i in range(5):
            workers.append((f"c{i}", "r", 1))
        report = _analyze(workers)
        s = report.summary()
        if report.patterns_detected:
            assert "Patterns:" in s


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

class TestJsonExport:
    def test_to_dict_keys(self):
        workers = [("r", None, 0), ("a", "r", 1)]
        report = _analyze(workers)
        d = report.to_dict()
        assert "total_workers" in d
        assert "risk_level" in d
        assert "node_metrics" in d
        assert "analyzed_at" in d
        assert d["total_workers"] == 2

    def test_to_json_parseable(self):
        import json
        workers = [("r", None, 0), ("a", "r", 1)]
        report = _analyze(workers)
        j = report.to_json()
        parsed = json.loads(j)
        assert parsed["total_workers"] == 2

    def test_risk_level_serialized_as_string(self):
        report = _analyze([("r", None, 0)])
        d = report.to_dict()
        assert isinstance(d["risk_level"], str)

    def test_patterns_serialized_as_strings(self):
        workers = [("r", None, 0)]
        for i in range(5):
            workers.append((f"c{i}", "r", 1))
        report = _analyze(workers)
        d = report.to_dict()
        for p in d["patterns_detected"]:
            assert isinstance(p, str)


# ---------------------------------------------------------------------------
# Controller integration
# ---------------------------------------------------------------------------

class TestFromController:
    def test_from_controller(self):
        contract = _contract(max_depth=5, max_replicas=20)
        logger = StructuredLogger()
        ctrl = Controller(contract=contract, secret="test", logger=logger)
        res = _resources()

        m1 = ctrl.issue_manifest(parent_id=None, depth=0, state_snapshot={}, resources=res)
        ctrl.register_worker(m1)

        m2 = ctrl.issue_manifest(parent_id=m1.worker_id, depth=0, state_snapshot={}, resources=res)
        ctrl.register_worker(m2)

        analyzer = TopologyAnalyzer.from_controller(ctrl)
        report = analyzer.analyze()
        assert report.total_workers == 2
        assert report.root_count == 1
        assert report.max_depth == 1

    def test_from_empty_controller(self):
        contract = _contract()
        ctrl = Controller(contract=contract, secret="test")
        analyzer = TopologyAnalyzer.from_controller(ctrl)
        report = analyzer.analyze()
        assert report.total_workers == 0


# ---------------------------------------------------------------------------
# Multiple roots
# ---------------------------------------------------------------------------

class TestMultipleRoots:
    def test_two_independent_trees(self):
        workers = [
            ("r1", None, 0),
            ("a", "r1", 1),
            ("r2", None, 0),
            ("b", "r2", 1),
        ]
        report = _analyze(workers)
        assert report.total_workers == 4
        assert report.root_count == 2
        assert report.max_depth == 1

    def test_multiple_roots_render(self):
        workers = [
            ("r1", None, 0), ("a", "r1", 1),
            ("r2", None, 0), ("b", "r2", 1),
        ]
        report = _analyze(workers)
        tree = report.render_tree()
        assert "r1" in tree
        assert "r2" in tree
