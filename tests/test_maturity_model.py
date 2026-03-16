"""Tests for Safety Maturity Model."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from src.replication.maturity_model import (
    DIMENSIONS,
    DIMENSION_MAP,
    LEVEL_DESCRIPTIONS,
    MATURITY_LEVELS,
    RECOMMENDATIONS,
    Criterion,
    Dimension,
    DimensionResult,
    MaturityAssessor,
    MaturityConfig,
    MaturityResult,
    main,
)


class TestMaturityLevels(unittest.TestCase):
    """Test maturity level definitions."""

    def test_five_levels_defined(self):
        self.assertEqual(len(MATURITY_LEVELS), 5)
        for i in range(1, 6):
            self.assertIn(i, MATURITY_LEVELS)

    def test_level_descriptions_match(self):
        self.assertEqual(set(LEVEL_DESCRIPTIONS.keys()), set(MATURITY_LEVELS.keys()))

    def test_level_labels(self):
        self.assertEqual(MATURITY_LEVELS[1], "Initial")
        self.assertEqual(MATURITY_LEVELS[5], "Optimizing")


class TestDimensions(unittest.TestCase):
    """Test dimension definitions."""

    def test_eight_dimensions(self):
        self.assertEqual(len(DIMENSIONS), 8)

    def test_all_dimensions_have_criteria(self):
        for dim in DIMENSIONS:
            self.assertGreater(len(dim.criteria), 0, f"{dim.name} has no criteria")

    def test_criteria_levels_1_to_5(self):
        for dim in DIMENSIONS:
            levels = {c.level for c in dim.criteria}
            self.assertEqual(levels, {1, 2, 3, 4, 5}, f"{dim.name} missing levels")

    def test_dimension_map_keys(self):
        expected_keys = {"governance", "risk_management", "monitoring", "incident_response",
                         "testing", "transparency", "ethics", "compliance"}
        self.assertEqual(set(DIMENSION_MAP.keys()), expected_keys)

    def test_criteria_have_descriptions(self):
        for dim in DIMENSIONS:
            for crit in dim.criteria:
                self.assertTrue(crit.description, f"{dim.name}/{crit.name} missing description")


class TestRecommendations(unittest.TestCase):
    """Test recommendation database."""

    def test_all_dimensions_have_recommendations(self):
        for dim in DIMENSIONS:
            self.assertIn(dim.key, RECOMMENDATIONS, f"No recommendations for {dim.key}")

    def test_recommendations_for_levels_2_through_5(self):
        for key, recs in RECOMMENDATIONS.items():
            for level in range(2, 6):
                self.assertIn(level, recs, f"No level {level} recommendations for {key}")
                self.assertGreater(len(recs[level]), 0)


class TestMaturityAssessorDefault(unittest.TestCase):
    """Test assessor with default (level 1) scoring."""

    def setUp(self):
        self.assessor = MaturityAssessor()

    def test_default_assessment(self):
        result = self.assessor.assess()
        self.assertIsInstance(result, MaturityResult)
        self.assertEqual(len(result.dimensions), 8)
        self.assertEqual(result.target_level, 3)

    def test_default_level_is_1(self):
        result = self.assessor.assess()
        for dr in result.dimensions:
            self.assertEqual(dr.level, 1, f"{dr.dimension.name} should be level 1 by default")

    def test_overall_score_range(self):
        result = self.assessor.assess()
        self.assertGreaterEqual(result.overall_score, 0)
        self.assertLessEqual(result.overall_score, 100)

    def test_overall_level_valid(self):
        result = self.assessor.assess()
        self.assertIn(result.overall_level, MATURITY_LEVELS)

    def test_timestamp_present(self):
        result = self.assessor.assess()
        self.assertTrue(result.timestamp)
        self.assertIn("T", result.timestamp)


class TestMaturityAssessorManualScores(unittest.TestCase):
    """Test assessor with manual criterion scores."""

    def setUp(self):
        self.assessor = MaturityAssessor()

    def test_all_met_yields_level_5(self):
        scores = {}
        for dim in DIMENSIONS:
            scores[dim.key] = {c.name: True for c in dim.criteria}
        config = MaturityConfig(scores=scores, target_level=5)
        result = self.assessor.assess(config)
        for dr in result.dimensions:
            self.assertEqual(dr.level, 5, f"{dr.dimension.name} should be 5")
        self.assertEqual(result.overall_level, 5)
        self.assertAlmostEqual(result.overall_score, 100.0)

    def test_partial_scores(self):
        scores = {
            "governance": {
                "Safety awareness": True,
                "Safety policy": True,
                "Dedicated roles": False,
                "Board oversight": False,
                "Continuous review": False,
            }
        }
        config = MaturityConfig(scores=scores, dimension_filter="governance")
        result = self.assessor.assess(config)
        self.assertEqual(len(result.dimensions), 1)
        dr = result.dimensions[0]
        self.assertEqual(dr.level, 2)
        self.assertEqual(dr.score, 40.0)

    def test_gap_in_middle_breaks_level(self):
        """If level 2 criterion is unmet, can't reach level 3 even if 3 is met."""
        scores = {
            "monitoring": {
                "Basic logging": True,
                "Active monitoring": False,  # level 2 unmet
                "Anomaly detection": True,   # level 3 met
                "Metrics dashboard": False,
                "Predictive alerts": False,
            }
        }
        config = MaturityConfig(scores=scores, dimension_filter="monitoring")
        result = self.assessor.assess(config)
        dr = result.dimensions[0]
        self.assertEqual(dr.level, 1)  # can't pass level 2

    def test_no_gaps_when_at_target(self):
        scores = {}
        for dim in DIMENSIONS:
            scores[dim.key] = {c.name: True for c in dim.criteria}
        config = MaturityConfig(scores=scores, target_level=5)
        result = self.assessor.assess(config)
        for dr in result.dimensions:
            self.assertEqual(len(dr.gaps), 0)


class TestDimensionFilter(unittest.TestCase):
    """Test single-dimension assessment."""

    def setUp(self):
        self.assessor = MaturityAssessor()

    def test_filter_by_key(self):
        config = MaturityConfig(dimension_filter="governance")
        result = self.assessor.assess(config)
        self.assertEqual(len(result.dimensions), 1)
        self.assertEqual(result.dimensions[0].dimension.key, "governance")

    def test_filter_fuzzy(self):
        config = MaturityConfig(dimension_filter="risk")
        result = self.assessor.assess(config)
        self.assertGreaterEqual(len(result.dimensions), 1)

    def test_filter_no_match_returns_all(self):
        config = MaturityConfig(dimension_filter="nonexistent_xyz")
        result = self.assessor.assess(config)
        # No match — should return all dimensions
        self.assertEqual(len(result.dimensions), 8)


class TestTargetLevel(unittest.TestCase):
    """Test target level configuration."""

    def setUp(self):
        self.assessor = MaturityAssessor()

    def test_target_clamped_to_range(self):
        config = MaturityConfig(target_level=10)
        result = self.assessor.assess(config)
        self.assertEqual(result.target_level, 5)

    def test_target_minimum_1(self):
        config = MaturityConfig(target_level=0)
        result = self.assessor.assess(config)
        self.assertEqual(result.target_level, 1)

    def test_higher_target_more_gaps(self):
        r3 = self.assessor.assess(MaturityConfig(target_level=3))
        r5 = self.assessor.assess(MaturityConfig(target_level=5))
        gaps3 = sum(len(dr.gaps) for dr in r3.dimensions)
        gaps5 = sum(len(dr.gaps) for dr in r5.dimensions)
        self.assertGreaterEqual(gaps5, gaps3)


class TestAutoAssess(unittest.TestCase):
    """Test auto-assessment mode."""

    def test_auto_assess_returns_result(self):
        assessor = MaturityAssessor()
        config = MaturityConfig(auto_assess=True)
        result = assessor.assess(config)
        self.assertIsInstance(result, MaturityResult)
        self.assertEqual(len(result.dimensions), 8)

    def test_auto_assess_all_level_1_minimum(self):
        assessor = MaturityAssessor()
        config = MaturityConfig(auto_assess=True)
        result = assessor.assess(config)
        for dr in result.dimensions:
            self.assertGreaterEqual(dr.level, 1)


class TestRendering(unittest.TestCase):
    """Test output rendering."""

    def setUp(self):
        self.assessor = MaturityAssessor()
        self.result = self.assessor.assess()

    def test_render_text(self):
        text = self.result.render()
        self.assertIn("SAFETY MATURITY ASSESSMENT", text)
        self.assertIn("Overall Maturity", text)
        self.assertIn("Level", text)

    def test_render_shows_dimensions(self):
        text = self.result.render()
        for dim in DIMENSIONS:
            self.assertIn(dim.name, text)

    def test_render_shows_gaps(self):
        text = self.result.render()
        self.assertIn("GAP ANALYSIS", text)

    def test_render_shows_priority_actions(self):
        text = self.result.render()
        self.assertIn("PRIORITY ACTIONS", text)


class TestSerialization(unittest.TestCase):
    """Test JSON serialization."""

    def setUp(self):
        self.assessor = MaturityAssessor()
        self.result = self.assessor.assess()

    def test_to_dict(self):
        d = self.result.to_dict()
        self.assertIn("overall_level", d)
        self.assertIn("dimensions", d)
        self.assertIn("gap_summary", d)
        self.assertIn("priority_actions", d)
        self.assertEqual(len(d["dimensions"]), 8)

    def test_json_roundtrip(self):
        d = self.result.to_dict()
        s = json.dumps(d)
        parsed = json.loads(s)
        self.assertEqual(parsed["overall_level"], d["overall_level"])
        self.assertEqual(len(parsed["dimensions"]), 8)


class TestHtmlOutput(unittest.TestCase):
    """Test HTML report generation."""

    def test_html_contains_structure(self):
        assessor = MaturityAssessor()
        result = assessor.assess()
        html = result.to_html()
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("Safety Maturity Assessment", html)
        self.assertIn("canvas", html)
        self.assertIn("radar", html)

    def test_html_contains_dimensions(self):
        assessor = MaturityAssessor()
        result = assessor.assess()
        html = result.to_html()
        for dim in DIMENSIONS:
            self.assertIn(dim.name, html)


class TestScoreToLevel(unittest.TestCase):
    """Test score-to-level conversion."""

    def test_boundaries(self):
        self.assertEqual(MaturityAssessor._score_to_level(0), 1)
        self.assertEqual(MaturityAssessor._score_to_level(20), 1)
        self.assertEqual(MaturityAssessor._score_to_level(30), 2)
        self.assertEqual(MaturityAssessor._score_to_level(50), 3)
        self.assertEqual(MaturityAssessor._score_to_level(70), 4)
        self.assertEqual(MaturityAssessor._score_to_level(90), 5)
        self.assertEqual(MaturityAssessor._score_to_level(100), 5)


class TestPriorityActions(unittest.TestCase):
    """Test priority action generation."""

    def test_actions_capped_at_10(self):
        assessor = MaturityAssessor()
        result = assessor.assess(MaturityConfig(target_level=5))
        self.assertLessEqual(len(result.priority_actions), 10)

    def test_actions_from_lowest_dimensions_first(self):
        assessor = MaturityAssessor()
        result = assessor.assess(MaturityConfig(target_level=5))
        # All actions should have dimension prefix
        for action in result.priority_actions:
            self.assertIn("[", action)
            self.assertIn("]", action)

    def test_no_actions_at_target(self):
        scores = {}
        for dim in DIMENSIONS:
            scores[dim.key] = {c.name: True for c in dim.criteria}
        assessor = MaturityAssessor()
        result = assessor.assess(MaturityConfig(scores=scores, target_level=5))
        self.assertEqual(len(result.priority_actions), 0)


class TestCLI(unittest.TestCase):
    """Test CLI entry point."""

    def test_default_run(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            main([])
        output = buf.getvalue()
        self.assertIn("SAFETY MATURITY ASSESSMENT", output)

    def test_json_flag(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["--json"])
        output = buf.getvalue()
        parsed = json.loads(output)
        self.assertIn("overall_level", parsed)

    def test_auto_flag(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["--auto"])
        output = buf.getvalue()
        self.assertIn("SAFETY MATURITY ASSESSMENT", output)

    def test_dimension_flag(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["--dimension", "governance"])
        output = buf.getvalue()
        self.assertIn("Governance", output)

    def test_target_flag(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["--target", "5"])
        output = buf.getvalue()
        self.assertIn("Target Level:     5", output)


class TestCustomDimensions(unittest.TestCase):
    """Test assessor with custom dimensions."""

    def test_custom_dimension(self):
        custom = [Dimension(
            name="Custom",
            key="custom",
            description="Test",
            criteria=[
                Criterion("C1", "First", 1),
                Criterion("C2", "Second", 2),
            ],
        )]
        assessor = MaturityAssessor(dimensions=custom)
        result = assessor.assess()
        self.assertEqual(len(result.dimensions), 1)
        self.assertEqual(result.dimensions[0].dimension.name, "Custom")


if __name__ == "__main__":
    unittest.main()
