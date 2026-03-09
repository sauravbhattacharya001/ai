"""Tests for the Attack Tree Generator module."""

import json
import pytest

from replication.attack_tree import (
    AttackNode,
    AttackPath,
    AttackTreeGenerator,
    AttackTreeResult,
    Difficulty,
    NodeType,
    TreeAnalysis,
    TreeConfig,
    ThreatGoal,
    _count_leaves,
    _enumerate_paths,
    _path_metrics,
    _compute_risk,
    _risk_bar,
    _GOAL_BUILDERS,
    _GOAL_ALIASES,
)
from replication._helpers import box_header as _box_header


# ── NodeType & Difficulty ────────────────────────────────────────────


class TestDifficulty:
    def test_numeric_values(self):
        assert Difficulty.TRIVIAL.numeric == 1.0
        assert Difficulty.EASY.numeric == 2.0
        assert Difficulty.MODERATE.numeric == 3.0
        assert Difficulty.HARD.numeric == 4.0
        assert Difficulty.EXPERT.numeric == 5.0

    def test_all_difficulties_have_numeric(self):
        for d in Difficulty:
            assert isinstance(d.numeric, float)
            assert 1.0 <= d.numeric <= 5.0


class TestThreatGoal:
    def test_all_goals_have_builders(self):
        for goal in ThreatGoal:
            assert goal in _GOAL_BUILDERS

    def test_goal_aliases_resolve(self):
        for alias, goal in _GOAL_ALIASES.items():
            assert isinstance(goal, ThreatGoal)


# ── AttackNode ───────────────────────────────────────────────────────


class TestAttackNode:
    def test_leaf_detection(self):
        leaf = AttackNode(id="x", label="X", node_type=NodeType.LEAF)
        assert leaf.is_leaf()

    def test_non_leaf_with_children(self):
        child = AttackNode(id="c", label="C", node_type=NodeType.LEAF)
        parent = AttackNode(
            id="p", label="P", node_type=NodeType.OR, children=[child]
        )
        assert not parent.is_leaf()

    def test_or_node_without_children_is_leaf(self):
        node = AttackNode(id="x", label="X", node_type=NodeType.OR)
        assert node.is_leaf()

    def test_to_dict_leaf(self):
        node = AttackNode(
            id="a", label="Attack", node_type=NodeType.LEAF,
            cost=50, difficulty=Difficulty.HARD, likelihood=0.3,
            mitigated=True, mitigation="firewall",
        )
        d = node.to_dict()
        assert d["id"] == "a"
        assert d["label"] == "Attack"
        assert d["type"] == "LEAF"
        assert d["cost"] == 50
        assert d["difficulty"] == "hard"
        assert d["likelihood"] == 0.3
        assert d["mitigated"] is True
        assert d["mitigation"] == "firewall"

    def test_to_dict_non_leaf(self):
        child = AttackNode(id="c", label="Child", node_type=NodeType.LEAF)
        parent = AttackNode(
            id="p", label="Parent", node_type=NodeType.AND, children=[child]
        )
        d = parent.to_dict()
        assert "children" in d
        assert len(d["children"]) == 1
        assert d["children"][0]["id"] == "c"

    def test_to_dict_leaf_no_mitigation(self):
        node = AttackNode(
            id="a", label="A", node_type=NodeType.LEAF,
            mitigated=False,
        )
        d = node.to_dict()
        assert "mitigated" not in d
        assert "mitigation" not in d


# ── AttackPath ───────────────────────────────────────────────────────


class TestAttackPath:
    def test_describe(self):
        nodes = [
            AttackNode(id="a", label="Root", node_type=NodeType.OR),
            AttackNode(id="b", label="Step 1", node_type=NodeType.LEAF),
        ]
        path = AttackPath(
            goal="test", nodes=nodes,
            total_cost=100, avg_difficulty=3.0, combined_likelihood=0.5,
        )
        assert path.describe() == "Root → Step 1"

    def test_to_dict(self):
        nodes = [
            AttackNode(id="a", label="A", node_type=NodeType.LEAF),
        ]
        path = AttackPath(
            goal="g", nodes=nodes,
            total_cost=99.999, avg_difficulty=2.555, combined_likelihood=0.12345,
        )
        d = path.to_dict()
        assert d["total_cost"] == 100.0
        assert d["avg_difficulty"] == 2.56
        assert d["combined_likelihood"] == 0.1235


# ── Leaf counting ────────────────────────────────────────────────────


class TestCountLeaves:
    def test_single_leaf(self):
        leaf = AttackNode(id="l", label="L", node_type=NodeType.LEAF, mitigated=True)
        total, mitigated = _count_leaves(leaf)
        assert total == 1
        assert mitigated == 1

    def test_unmitigated_leaf(self):
        leaf = AttackNode(id="l", label="L", node_type=NodeType.LEAF, mitigated=False)
        total, mitigated = _count_leaves(leaf)
        assert total == 1
        assert mitigated == 0

    def test_nested_tree(self):
        tree = AttackNode(
            id="r", label="Root", node_type=NodeType.OR,
            children=[
                AttackNode(id="a", label="A", node_type=NodeType.LEAF, mitigated=True),
                AttackNode(id="b", label="B", node_type=NodeType.LEAF, mitigated=False),
                AttackNode(
                    id="c", label="C", node_type=NodeType.AND,
                    children=[
                        AttackNode(id="d", label="D", node_type=NodeType.LEAF, mitigated=True),
                        AttackNode(id="e", label="E", node_type=NodeType.LEAF, mitigated=True),
                    ],
                ),
            ],
        )
        total, mitigated = _count_leaves(tree)
        assert total == 4
        assert mitigated == 3


# ── Path enumeration ─────────────────────────────────────────────────


class TestEnumeratePaths:
    def test_single_leaf(self):
        leaf = AttackNode(id="l", label="L", node_type=NodeType.LEAF)
        paths = _enumerate_paths(leaf, [])
        assert len(paths) == 1
        assert len(paths[0]) == 1

    def test_or_node_gives_separate_paths(self):
        root = AttackNode(
            id="r", label="R", node_type=NodeType.OR,
            children=[
                AttackNode(id="a", label="A", node_type=NodeType.LEAF),
                AttackNode(id="b", label="B", node_type=NodeType.LEAF),
            ],
        )
        paths = _enumerate_paths(root, [])
        assert len(paths) == 2
        # Each path should contain root + one child
        path_ids = [tuple(n.id for n in p) for p in paths]
        assert ("r", "a") in path_ids
        assert ("r", "b") in path_ids

    def test_and_node_gives_combined_path(self):
        root = AttackNode(
            id="r", label="R", node_type=NodeType.AND,
            children=[
                AttackNode(id="a", label="A", node_type=NodeType.LEAF),
                AttackNode(id="b", label="B", node_type=NodeType.LEAF),
            ],
        )
        paths = _enumerate_paths(root, [])
        assert len(paths) == 1
        leaf_ids = [n.id for n in paths[0] if n.is_leaf()]
        assert "a" in leaf_ids
        assert "b" in leaf_ids

    def test_nested_or_under_and(self):
        root = AttackNode(
            id="r", label="R", node_type=NodeType.AND,
            children=[
                AttackNode(
                    id="or1", label="OR1", node_type=NodeType.OR,
                    children=[
                        AttackNode(id="a", label="A", node_type=NodeType.LEAF),
                        AttackNode(id="b", label="B", node_type=NodeType.LEAF),
                    ],
                ),
                AttackNode(id="c", label="C", node_type=NodeType.LEAF),
            ],
        )
        paths = _enumerate_paths(root, [])
        # 2 OR options × 1 mandatory = 2 paths, each containing c
        assert len(paths) == 2
        for path in paths:
            leaf_ids = [n.id for n in path if n.is_leaf()]
            assert "c" in leaf_ids

    def test_deeply_nested(self):
        root = AttackNode(
            id="r", label="R", node_type=NodeType.OR,
            children=[
                AttackNode(
                    id="and1", label="AND1", node_type=NodeType.AND,
                    children=[
                        AttackNode(
                            id="or2", label="OR2", node_type=NodeType.OR,
                            children=[
                                AttackNode(id="x", label="X", node_type=NodeType.LEAF),
                                AttackNode(id="y", label="Y", node_type=NodeType.LEAF),
                            ],
                        ),
                        AttackNode(id="z", label="Z", node_type=NodeType.LEAF),
                    ],
                ),
                AttackNode(id="w", label="W", node_type=NodeType.LEAF),
            ],
        )
        paths = _enumerate_paths(root, [])
        # OR: (AND path with x+z, AND path with y+z, w alone) = 3 paths
        assert len(paths) == 3


# ── Path metrics ─────────────────────────────────────────────────────


class TestPathMetrics:
    def test_empty_path(self):
        cost, diff, like = _path_metrics([])
        assert cost == 0.0
        assert diff == 0.0
        assert like == 0.0

    def test_single_leaf(self):
        leaf = AttackNode(
            id="l", label="L", node_type=NodeType.LEAF,
            cost=50, difficulty=Difficulty.HARD, likelihood=0.3,
        )
        cost, diff, like = _path_metrics([leaf])
        assert cost == 50
        assert diff == 4.0  # HARD = 4
        assert like == pytest.approx(0.3)

    def test_multiple_leaves(self):
        # Only LEAF nodes in the path; parent AND/OR nodes with children
        # won't appear as leaves but bare nodes with no children do.
        leaf_a = AttackNode(
            id="a", label="A", node_type=NodeType.LEAF,
            cost=30, difficulty=Difficulty.EASY, likelihood=0.5,
        )
        leaf_b = AttackNode(
            id="b", label="B", node_type=NodeType.LEAF,
            cost=20, difficulty=Difficulty.HARD, likelihood=0.4,
        )
        cost, diff, like = _path_metrics([leaf_a, leaf_b])
        assert cost == 50
        assert diff == pytest.approx(3.0)  # avg(2, 4) = 3
        assert like == pytest.approx(0.2)  # 0.5 * 0.4


# ── Risk computation ─────────────────────────────────────────────────


class TestComputeRisk:
    def test_no_paths_zero_risk(self):
        assert _compute_risk([], 0, 0) == 0.0

    def test_cheap_path_high_risk(self):
        paths = [
            AttackPath(
                goal="g", nodes=[], total_cost=5,
                avg_difficulty=1.0, combined_likelihood=0.8,
            ),
        ]
        risk = _compute_risk(paths, 1, 0)
        assert risk > 50  # cheap + likely + unmitigated = high risk

    def test_expensive_mitigated_low_risk(self):
        paths = [
            AttackPath(
                goal="g", nodes=[], total_cost=100,
                avg_difficulty=5.0, combined_likelihood=0.01,
            ),
        ]
        risk = _compute_risk(paths, 5, 5)
        assert risk < 30  # expensive + unlikely + fully mitigated

    def test_risk_bounded_0_100(self):
        paths = [
            AttackPath(
                goal="g", nodes=[], total_cost=0,
                avg_difficulty=1.0, combined_likelihood=1.0,
            ),
        ]
        risk = _compute_risk(paths, 1, 0)
        assert 0 <= risk <= 100


# ── Rendering helpers ────────────────────────────────────────────────


class TestRendering:
    def test_box_header(self):
        lines = _box_header("TEST", 20)
        assert len(lines) == 3
        assert lines[0].startswith("\u250c")  # Unicode box-drawing (from _helpers)
        assert "TEST" in lines[1]

    def test_risk_bar_zero(self):
        bar = _risk_bar(0, 10)
        assert bar == "[..........]"

    def test_risk_bar_full(self):
        bar = _risk_bar(100, 10)
        assert bar == "[##########]"

    def test_risk_bar_half(self):
        bar = _risk_bar(50, 10)
        assert bar == "[#####.....]"


# ── Tree templates ───────────────────────────────────────────────────


class TestTreeTemplates:
    @pytest.mark.parametrize("goal", list(ThreatGoal))
    def test_builder_returns_valid_tree(self, goal):
        builder = _GOAL_BUILDERS[goal]
        root = builder()
        assert isinstance(root, AttackNode)
        assert root.node_type in (NodeType.OR, NodeType.AND)

    @pytest.mark.parametrize("goal", list(ThreatGoal))
    def test_tree_has_leaves(self, goal):
        root = _GOAL_BUILDERS[goal]()
        total, _ = _count_leaves(root)
        assert total >= 3  # every tree should have meaningful depth

    @pytest.mark.parametrize("goal", list(ThreatGoal))
    def test_tree_has_some_mitigations(self, goal):
        root = _GOAL_BUILDERS[goal]()
        _, mitigated = _count_leaves(root)
        assert mitigated >= 1  # at least one mitigation per goal

    @pytest.mark.parametrize("goal", list(ThreatGoal))
    def test_tree_has_enumerable_paths(self, goal):
        root = _GOAL_BUILDERS[goal]()
        paths = _enumerate_paths(root, [])
        assert len(paths) >= 2  # non-trivial trees

    @pytest.mark.parametrize("goal", list(ThreatGoal))
    def test_node_ids_unique(self, goal):
        root = _GOAL_BUILDERS[goal]()
        ids = set()

        def collect_ids(node):
            ids.add(node.id)
            for child in node.children:
                collect_ids(child)

        collect_ids(root)
        # Check uniqueness by counting
        all_ids = []

        def collect_all(node):
            all_ids.append(node.id)
            for child in node.children:
                collect_all(child)

        collect_all(root)
        assert len(all_ids) == len(ids), f"Duplicate IDs in {goal.value}"


# ── AttackTreeGenerator ──────────────────────────────────────────────


class TestAttackTreeGenerator:
    def test_analyze_all_goals(self):
        gen = AttackTreeGenerator()
        result = gen.analyze()
        assert isinstance(result, AttackTreeResult)
        assert len(result.analyses) == len(ThreatGoal)

    def test_analyze_single_goal(self):
        gen = AttackTreeGenerator()
        result = gen.analyze(goals=[ThreatGoal.SANDBOX_ESCAPE])
        assert len(result.analyses) == 1
        assert result.analyses[0].goal == ThreatGoal.SANDBOX_ESCAPE

    def test_analysis_fields(self):
        gen = AttackTreeGenerator()
        result = gen.analyze(goals=[ThreatGoal.UNAUTHORIZED_REPLICATION])
        a = result.analyses[0]
        assert a.leaf_count > 0
        assert 0 <= a.overall_risk <= 100
        assert isinstance(a.all_paths, list)
        assert len(a.all_paths) > 0
        assert a.min_cost_path is not None
        assert a.max_likelihood_path is not None

    def test_cheapest_paths(self):
        gen = AttackTreeGenerator()
        result = gen.analyze()
        paths = result.cheapest_paths("sandbox_escape", n=3)
        assert len(paths) <= 3
        if len(paths) > 1:
            assert paths[0].total_cost <= paths[1].total_cost

    def test_cheapest_paths_unknown_goal(self):
        gen = AttackTreeGenerator()
        result = gen.analyze()
        assert result.cheapest_paths("nonexistent") == []

    def test_config_min_paths(self):
        config = TreeConfig(min_paths=2)
        gen = AttackTreeGenerator(config)
        result = gen.analyze()
        assert result.config.min_paths == 2


# ── Serialization ────────────────────────────────────────────────────


class TestSerialization:
    def test_result_to_dict(self):
        gen = AttackTreeGenerator()
        result = gen.analyze()
        d = result.to_dict()
        assert "analyses" in d
        assert "summary" in d
        assert d["summary"]["goals_analyzed"] == len(ThreatGoal)
        assert d["summary"]["total_attack_paths"] > 0

    def test_analysis_to_dict(self):
        gen = AttackTreeGenerator()
        result = gen.analyze(goals=[ThreatGoal.CONTROLLER_TAKEOVER])
        a = result.analyses[0]
        d = a.to_dict()
        assert d["goal"] == "controller_takeover"
        assert "tree" in d
        assert "leaf_count" in d
        assert "coverage_pct" in d
        assert "min_cost_path" in d

    def test_json_roundtrip(self):
        gen = AttackTreeGenerator()
        result = gen.analyze()
        j = json.dumps(result.to_dict())
        loaded = json.loads(j)
        assert loaded["summary"]["goals_analyzed"] == len(ThreatGoal)

    def test_render_output(self):
        gen = AttackTreeGenerator()
        result = gen.analyze()
        rendered = result.render()
        assert "ATTACK TREE ANALYSIS" in rendered
        assert "SUMMARY" in rendered
        assert "Risk Ranking" in rendered

    def test_render_single_goal(self):
        config = TreeConfig(goals=[ThreatGoal.STEALTH_PERSISTENCE])
        gen = AttackTreeGenerator(config)
        result = gen.analyze()
        rendered = result.render()
        assert "Stealth Persistence" in rendered


# ── CLI ──────────────────────────────────────────────────────────────


class TestCLI:
    def test_main_default(self, capsys):
        from replication.attack_tree import main
        main([])
        captured = capsys.readouterr()
        assert "ATTACK TREE ANALYSIS" in captured.out

    def test_main_json(self, capsys):
        from replication.attack_tree import main
        main(["--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "analyses" in data

    def test_main_single_goal(self, capsys):
        from replication.attack_tree import main
        main(["--goal", "escape"])
        captured = capsys.readouterr()
        assert "Sandbox Escape" in captured.out

    def test_main_min_paths(self, capsys):
        from replication.attack_tree import main
        main(["--min-paths", "2", "--goal", "takeover"])
        captured = capsys.readouterr()
        assert "Top 2" in captured.out

    def test_main_no_annotate(self, capsys):
        from replication.attack_tree import main
        main(["--no-annotate", "--goal", "persistence"])
        captured = capsys.readouterr()
        # Without annotations, tree leaf lines should NOT have diff= or p=
        # (cost= still appears in path summaries which is expected)
        assert "diff=" not in captured.out
        assert "p=" not in captured.out
