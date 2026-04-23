"""Tests for reward_hacking module."""

import json
import unittest

from replication.reward_hacking import (
    AgentMetricProfile,
    DetectorConfig,
    HackingPattern,
    HackingSignal,
    MetricObservation,
    RewardHackingDetector,
    RewardHackingReport,
    Severity,
    _run_simulation,
)


class TestMetricObservation(unittest.TestCase):
    def test_defaults(self):
        obs = MetricObservation(agent_id="a", metric_name="m", metric_value=0.5)
        self.assertIsNone(obs.ground_truth)
        self.assertEqual(obs.timestamp, 0.0)
        self.assertEqual(obs.context, {})


class TestAgentMetricProfile(unittest.TestCase):
    def test_add_and_query(self):
        p = AgentMetricProfile("a1")
        p.add(MetricObservation("a1", "acc", 0.8, ground_truth=0.7))
        p.add(MetricObservation("a1", "acc", 0.9, ground_truth=0.6))
        self.assertEqual(p.metric_values("acc"), [0.8, 0.9])
        self.assertEqual(p.ground_truths("acc"), [0.7, 0.6])
        self.assertIn("acc", p.metric_names())


class TestDetectorEmpty(unittest.TestCase):
    def test_empty_analyze(self):
        det = RewardHackingDetector()
        report = det.analyze()
        self.assertEqual(report.signals, [])
        self.assertEqual(report.summary["total_agents"], 0)

    def test_single_observation(self):
        det = RewardHackingDetector()
        det.add_observation(MetricObservation("a", "m", 0.5, 0.5, 1.0))
        report = det.analyze()
        # Not enough data for signals
        self.assertEqual(report.signals, [])


class TestDivergenceDetection(unittest.TestCase):
    def test_divergence_flagged(self):
        det = RewardHackingDetector(DetectorConfig(min_observations=3, window_size=5, divergence_threshold=0.15))
        for i in range(20):
            det.add_observation(MetricObservation(
                "a1", "score", 0.3 + i * 0.03, ground_truth=0.5 - i * 0.01, timestamp=float(i)
            ))
        report = det.analyze()
        patterns = [s.pattern for s in report.signals]
        self.assertIn(HackingPattern.METRIC_OBJECTIVE_DIVERGENCE, patterns)


class TestInflationDetection(unittest.TestCase):
    def test_inflation_flagged(self):
        det = RewardHackingDetector(DetectorConfig(min_observations=3, inflation_rate_threshold=0.1))
        v = 0.1
        for i in range(10):
            v *= 1.3  # 30% growth
            det.add_observation(MetricObservation("a1", "reward", v, timestamp=float(i)))
        report = det.analyze()
        patterns = [s.pattern for s in report.signals]
        self.assertIn(HackingPattern.REWARD_INFLATION, patterns)


class TestGoodhartDrift(unittest.TestCase):
    def test_correlation_decay(self):
        det = RewardHackingDetector(DetectorConfig(min_observations=4, correlation_decay_threshold=0.2))
        # Early: correlated
        for i in range(10):
            v = 0.5 + i * 0.02
            det.add_observation(MetricObservation("a1", "m", v, ground_truth=v + 0.01, timestamp=float(i)))
        # Late: uncorrelated
        for i in range(10, 20):
            det.add_observation(MetricObservation("a1", "m", 0.7 + i * 0.02, ground_truth=0.6 - i * 0.01, timestamp=float(i)))
        report = det.analyze()
        patterns = [s.pattern for s in report.signals]
        self.assertIn(HackingPattern.GOODHART_DRIFT, patterns)


class TestEdgeCaseDetection(unittest.TestCase):
    def test_boundary_values(self):
        det = RewardHackingDetector(DetectorConfig(min_observations=3, edge_case_repeat_threshold=2))
        for i in range(10):
            det.add_observation(MetricObservation("a1", "m", 1.0, timestamp=float(i)))
        report = det.analyze()
        patterns = [s.pattern for s in report.signals]
        self.assertIn(HackingPattern.EDGE_CASE_EXPLOITATION, patterns)


class TestSimulationPresets(unittest.TestCase):
    def test_blatant_preset(self):
        det = _run_simulation(2, 30, "blatant")
        report = det.analyze_fleet()
        self.assertGreater(len(report.signals), 0)

    def test_clean_preset(self):
        det = _run_simulation(2, 30, "clean")
        report = det.analyze_fleet()
        # Clean should have few or no signals
        critical = [s for s in report.signals if s.severity == Severity.CRITICAL]
        self.assertEqual(len(critical), 0)

    def test_mixed_preset(self):
        det = _run_simulation(4, 40, "mixed")
        report = det.analyze_fleet()
        self.assertGreater(report.summary["total_agents"], 0)


class TestRendering(unittest.TestCase):
    def test_text_render(self):
        det = _run_simulation(2, 20, "blatant")
        report = det.analyze_fleet()
        text = det.render_text(report)
        self.assertIn("REWARD HACKING DETECTOR", text)
        self.assertIn("RECOMMENDATIONS", text)

    def test_html_render(self):
        det = _run_simulation(2, 20, "subtle")
        report = det.analyze_fleet()
        html = det.render_html(report)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("Reward Hacking", html)

    def test_json_render(self):
        det = _run_simulation(2, 20, "mixed")
        report = det.analyze_fleet()
        j = det.render_json(report)
        data = json.loads(j)
        self.assertIn("signals", data)
        self.assertIn("recommendations", data)


class TestRecommendations(unittest.TestCase):
    def test_no_signals_recommendation(self):
        det = RewardHackingDetector()
        report = det.analyze()
        self.assertIn("No reward-hacking signals detected", report.recommendations[0])


if __name__ == "__main__":
    unittest.main()
