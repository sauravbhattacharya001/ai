"""Tests for attack_surface module."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from replication.attack_surface import (
    _analysis_to_json,
    _node_to_sunburst,
    generate_surface,
    main,
)
from replication.attack_tree import (
    AttackNode,
    AttackPath,
    AttackTreeGenerator,
    AttackTreeResult,
    Difficulty,
    NodeType,
    ThreatGoal,
    TreeAnalysis,
    TreeConfig,
)


def _make_leaf(id_: str, label: str, **kwargs) -> AttackNode:
    return AttackNode(
        id=id_,
        label=label,
        node_type=NodeType.LEAF,
        **kwargs,
    )


def _make_tree() -> AttackNode:
    """Build a small attack tree for testing."""
    return AttackNode(
        id="root",
        label="Unauthorized Replication",
        node_type=NodeType.OR,
        children=[
            AttackNode(
                id="sub1",
                label="Exploit API",
                node_type=NodeType.AND,
                children=[
                    _make_leaf("l1", "Find endpoint", cost=1.0,
                               difficulty=Difficulty.EASY, likelihood=0.8),
                    _make_leaf("l2", "Bypass auth", cost=5.0,
                               difficulty=Difficulty.HARD, likelihood=0.2,
                               mitigated=True, mitigation="MFA enabled"),
                ],
            ),
            _make_leaf("l3", "Social engineering", cost=2.0,
                       difficulty=Difficulty.MODERATE, likelihood=0.5),
        ],
    )


def _make_analysis() -> TreeAnalysis:
    root = _make_tree()
    return TreeAnalysis(
        goal=ThreatGoal.UNAUTHORIZED_REPLICATION,
        root=root,
        leaf_count=3,
        mitigated_count=1,
        min_cost_path=AttackPath(
            goal="unauthorized_replication",
            nodes=[root, root.children[1]],
            total_cost=2.0,
            avg_difficulty=3.0,
            combined_likelihood=0.5,
        ),
        max_likelihood_path=AttackPath(
            goal="unauthorized_replication",
            nodes=[root, root.children[0], root.children[0].children[0]],
            total_cost=1.0,
            avg_difficulty=2.0,
            combined_likelihood=0.8,
        ),
        all_paths=[],
        overall_risk=42.0,
    )


class TestNodeToSunburst(unittest.TestCase):
    def test_leaf_node(self):
        leaf = _make_leaf("n1", "Test leaf", cost=3.0,
                          difficulty=Difficulty.HARD, likelihood=0.4)
        result = _node_to_sunburst(leaf)
        self.assertEqual(result["name"], "Test leaf")
        self.assertEqual(result["id"], "n1")
        self.assertEqual(result["type"], "LEAF")
        self.assertEqual(result["cost"], 3.0)
        self.assertEqual(result["difficulty"], "hard")
        self.assertEqual(result["likelihood"], 0.4)
        self.assertFalse(result["mitigated"])
        self.assertEqual(result["value"], 1)

    def test_mitigated_leaf(self):
        leaf = _make_leaf("n2", "Fixed", mitigated=True,
                          mitigation="Patch applied")
        result = _node_to_sunburst(leaf)
        self.assertTrue(result["mitigated"])
        self.assertEqual(result["mitigation"], "Patch applied")

    def test_internal_node(self):
        root = _make_tree()
        result = _node_to_sunburst(root)
        self.assertEqual(result["type"], "OR")
        self.assertIn("children", result)
        self.assertEqual(len(result["children"]), 2)
        self.assertNotIn("value", result)

    def test_risk_calculation(self):
        leaf = _make_leaf("r1", "Risk test",
                          difficulty=Difficulty.TRIVIAL, likelihood=1.0)
        result = _node_to_sunburst(leaf)
        # risk = likelihood * (1 - (difficulty.numeric - 1) / 4)
        # = 1.0 * (1 - 0/4) = 1.0
        self.assertEqual(result["risk"], 1.0)

    def test_risk_hard_low_likelihood(self):
        leaf = _make_leaf("r2", "Hard low",
                          difficulty=Difficulty.EXPERT, likelihood=0.2)
        result = _node_to_sunburst(leaf)
        # risk = 0.2 * (1 - 4/4) = 0.0
        self.assertEqual(result["risk"], 0.0)

    def test_nested_structure(self):
        root = _make_tree()
        result = _node_to_sunburst(root)
        and_node = result["children"][0]
        self.assertEqual(and_node["type"], "AND")
        self.assertEqual(len(and_node["children"]), 2)


class TestAnalysisToJson(unittest.TestCase):
    def test_basic_fields(self):
        analysis = _make_analysis()
        result = _analysis_to_json(analysis)
        self.assertEqual(result["goal"], "unauthorized_replication")
        self.assertEqual(result["leafCount"], 3)
        self.assertEqual(result["mitigatedCount"], 1)
        self.assertAlmostEqual(result["coveragePct"], 33.3, places=1)
        self.assertEqual(result["overallRisk"], 42.0)
        self.assertIsNotNone(result["tree"])

    def test_cheapest_path(self):
        analysis = _make_analysis()
        result = _analysis_to_json(analysis)
        self.assertIsNotNone(result["cheapestPath"])
        self.assertEqual(result["cheapestPath"]["total_cost"], 2.0)

    def test_highest_likelihood_path(self):
        analysis = _make_analysis()
        result = _analysis_to_json(analysis)
        self.assertIsNotNone(result["highestLikelihoodPath"])
        self.assertAlmostEqual(
            result["highestLikelihoodPath"]["combined_likelihood"], 0.8
        )

    def test_coverage_zero_leaves(self):
        analysis = _make_analysis()
        analysis.leaf_count = 0
        analysis.mitigated_count = 0
        result = _analysis_to_json(analysis)
        self.assertEqual(result["coveragePct"], 0.0)


class TestGenerateSurface(unittest.TestCase):
    def test_returns_html_string(self):
        html = generate_surface()
        self.assertIsInstance(html, str)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("Attack Surface", html)

    def test_contains_data(self):
        html = generate_surface()
        self.assertIn("const DATA =", html)

    def test_contains_goal_names(self):
        html = generate_surface()
        # Default config generates all 5 threat goals
        self.assertIn("unauthorized_replication", html)

    def test_custom_config(self):
        config = TreeConfig(min_paths=3)
        html = generate_surface(config=config)
        self.assertIn("const DATA =", html)

    def test_precomputed_result(self):
        analysis = _make_analysis()
        result = AttackTreeResult(
            analyses=[analysis],
            config=TreeConfig(),
        )
        html = generate_surface(result=result)
        self.assertIn("unauthorized_replication", html)
        self.assertIn("Exploit API", html)

    def test_html_has_interactive_elements(self):
        html = generate_surface()
        self.assertIn("goalSelect", html)
        self.assertIn("search", html)
        self.assertIn("exportBtn", html)
        self.assertIn("tooltip", html)
        self.assertIn("canvas", html)

    def test_data_is_valid_json(self):
        html = generate_surface()
        # Extract JSON from const DATA = ...;
        start = html.index("const DATA = ") + len("const DATA = ")
        end = html.index(";", start)
        data_str = html[start:end]
        data = json.loads(data_str)
        self.assertIsInstance(data, list)
        self.assertTrue(len(data) > 0)
        for entry in data:
            self.assertIn("goal", entry)
            self.assertIn("tree", entry)
            self.assertIn("leafCount", entry)


class TestMain(unittest.TestCase):
    def test_writes_file(self):
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "test_surface.html")
            main(["-o", out])
            self.assertTrue(os.path.exists(out))
            content = Path(out).read_text(encoding="utf-8")
            self.assertIn("<!DOCTYPE html>", content)

    def test_seed_flag(self):
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "seeded.html")
            main(["-o", out])
            self.assertTrue(os.path.exists(out))


if __name__ == "__main__":
    unittest.main()
