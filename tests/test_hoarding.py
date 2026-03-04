"""Tests for the resource hoarding detector."""

import unittest

from src.replication.hoarding import (
    AgentHoardingProfile,
    DetectorConfig,
    HoardingReport,
    HoardingSignal,
    ResourceHoardingDetector,
    ResourceSnapshot,
    ResourceType,
    RiskLevel,
    _clamp,
    _coefficient_of_variation,
    _grade_risk,
    _linear_regression_slope,
    _monotonic_increase_ratio,
    demo,
)


def _make_snapshots(agent_id, count, **overrides):
    """Helper to create a series of snapshots with defaults."""
    snaps = []
    for i in range(count):
        kwargs = {
            "agent_id": agent_id,
            "timestamp": float(i),
            "compute_units": float(i * 2),
            "memory_mb": 100.0,
            "data_stored_mb": float(i * 5),
            "open_connections": 2,
            "open_file_handles": 3,
            "task_output_count": i + 1,
        }
        for key, fn in overrides.items():
            kwargs[key] = fn(i)
        snaps.append(ResourceSnapshot(**kwargs))
    return snaps


class TestUtilities(unittest.TestCase):
    """Tests for helper functions."""

    def test_linear_regression_slope_flat(self):
        self.assertAlmostEqual(_linear_regression_slope([5, 5, 5, 5]), 0.0)

    def test_linear_regression_slope_increasing(self):
        slope = _linear_regression_slope([0, 1, 2, 3])
        self.assertAlmostEqual(slope, 1.0)

    def test_linear_regression_slope_decreasing(self):
        slope = _linear_regression_slope([6, 4, 2, 0])
        self.assertAlmostEqual(slope, -2.0)

    def test_linear_regression_slope_single(self):
        self.assertAlmostEqual(_linear_regression_slope([42]), 0.0)

    def test_linear_regression_slope_empty(self):
        self.assertAlmostEqual(_linear_regression_slope([]), 0.0)

    def test_monotonic_increase_ratio_all_increasing(self):
        self.assertAlmostEqual(_monotonic_increase_ratio([1, 2, 3, 4, 5]), 1.0)

    def test_monotonic_increase_ratio_none(self):
        self.assertAlmostEqual(_monotonic_increase_ratio([5, 4, 3, 2, 1]), 0.0)

    def test_monotonic_increase_ratio_mixed(self):
        ratio = _monotonic_increase_ratio([1, 2, 1, 2])
        self.assertAlmostEqual(ratio, 2 / 3)

    def test_monotonic_increase_ratio_short(self):
        self.assertAlmostEqual(_monotonic_increase_ratio([42]), 0.0)

    def test_coefficient_of_variation_zero_mean(self):
        self.assertAlmostEqual(_coefficient_of_variation([0, 0, 0]), 0.0)

    def test_coefficient_of_variation_constant(self):
        self.assertAlmostEqual(_coefficient_of_variation([5, 5, 5]), 0.0)

    def test_coefficient_of_variation_varied(self):
        cv = _coefficient_of_variation([10, 20, 30])
        self.assertGreater(cv, 0)

    def test_grade_risk_critical(self):
        self.assertEqual(_grade_risk(95), RiskLevel.CRITICAL)

    def test_grade_risk_high(self):
        self.assertEqual(_grade_risk(65), RiskLevel.HIGH)

    def test_grade_risk_moderate(self):
        self.assertEqual(_grade_risk(45), RiskLevel.MODERATE)

    def test_grade_risk_low(self):
        self.assertEqual(_grade_risk(25), RiskLevel.LOW)

    def test_grade_risk_none(self):
        self.assertEqual(_grade_risk(5), RiskLevel.NONE)

    def test_clamp_within(self):
        self.assertEqual(_clamp(50), 50.0)

    def test_clamp_below(self):
        self.assertEqual(_clamp(-10), 0.0)

    def test_clamp_above(self):
        self.assertEqual(_clamp(150), 100.0)


class TestResourceSnapshot(unittest.TestCase):
    """Tests for the ResourceSnapshot dataclass."""

    def test_defaults(self):
        s = ResourceSnapshot(agent_id="a1", timestamp=0.0)
        self.assertEqual(s.compute_units, 0.0)
        self.assertEqual(s.memory_mb, 0.0)
        self.assertEqual(s.open_connections, 0)
        self.assertEqual(s.task_output_count, 0)
        self.assertIsInstance(s.metadata, dict)

    def test_custom_values(self):
        s = ResourceSnapshot(
            agent_id="x", timestamp=1.0,
            compute_units=10, memory_mb=512,
            data_stored_mb=100, open_connections=5,
            open_file_handles=10, task_output_count=3,
        )
        self.assertEqual(s.compute_units, 10)
        self.assertEqual(s.memory_mb, 512)


class TestDetectorConfig(unittest.TestCase):
    """Tests for DetectorConfig defaults."""

    def test_defaults(self):
        cfg = DetectorConfig()
        self.assertEqual(cfg.compute_per_output_threshold, 10.0)
        self.assertEqual(cfg.min_snapshots, 3)
        self.assertIn("compute", cfg.dimension_weights)
        total = sum(cfg.dimension_weights.values())
        self.assertAlmostEqual(total, 1.0)


class TestDetectorRecording(unittest.TestCase):
    """Tests for snapshot recording."""

    def test_record_single(self):
        d = ResourceHoardingDetector()
        d.record(ResourceSnapshot(agent_id="a1", timestamp=0))
        self.assertEqual(d.snapshot_count("a1"), 1)
        self.assertIn("a1", d.agent_ids())

    def test_record_batch(self):
        d = ResourceHoardingDetector()
        snaps = _make_snapshots("a1", 5)
        count = d.record_batch(snaps)
        self.assertEqual(count, 5)
        self.assertEqual(d.snapshot_count("a1"), 5)

    def test_record_empty_agent_id_raises(self):
        d = ResourceHoardingDetector()
        with self.assertRaises(ValueError):
            d.record(ResourceSnapshot(agent_id="", timestamp=0))

    def test_multiple_agents(self):
        d = ResourceHoardingDetector()
        d.record(ResourceSnapshot(agent_id="a1", timestamp=0))
        d.record(ResourceSnapshot(agent_id="a2", timestamp=0))
        self.assertEqual(len(d.agent_ids()), 2)

    def test_clear_specific(self):
        d = ResourceHoardingDetector()
        d.record(ResourceSnapshot(agent_id="a1", timestamp=0))
        d.record(ResourceSnapshot(agent_id="a2", timestamp=0))
        d.clear("a1")
        self.assertEqual(d.snapshot_count("a1"), 0)
        self.assertEqual(d.snapshot_count("a2"), 1)

    def test_clear_all(self):
        d = ResourceHoardingDetector()
        d.record_batch(_make_snapshots("a1", 3))
        d.record_batch(_make_snapshots("a2", 3))
        d.clear()
        self.assertEqual(len(d.agent_ids()), 0)


class TestNormalAgent(unittest.TestCase):
    """An agent with proportional resource usage should score low."""

    def setUp(self):
        self.detector = ResourceHoardingDetector()
        snaps = _make_snapshots("normal", 10)
        self.detector.record_batch(snaps)
        self.profile = self.detector.analyze_agent("normal")

    def test_low_composite_score(self):
        self.assertLess(self.profile.composite_score, 40)

    def test_risk_level_not_high(self):
        self.assertNotIn(self.profile.risk_level,
                         [RiskLevel.HIGH, RiskLevel.CRITICAL])

    def test_no_critical_signals(self):
        critical = [s for s in self.profile.signals
                    if s.risk_level == RiskLevel.CRITICAL]
        self.assertEqual(len(critical), 0)


class TestHoardingAgent(unittest.TestCase):
    """An agent hoarding all resources should score high."""

    def setUp(self):
        self.detector = ResourceHoardingDetector()
        for i in range(10):
            self.detector.record(ResourceSnapshot(
                agent_id="hoarder",
                timestamp=float(i),
                compute_units=float(i * 50),
                memory_mb=100 + i * 100,
                data_stored_mb=float(i * 300),
                open_connections=i * 5,
                open_file_handles=i * 10,
                task_output_count=1,  # almost no output
            ))
        self.profile = self.detector.analyze_agent("hoarder")

    def test_high_composite_score(self):
        self.assertGreater(self.profile.composite_score, 50)

    def test_risk_level_elevated(self):
        self.assertIn(self.profile.risk_level,
                      [RiskLevel.HIGH, RiskLevel.CRITICAL])

    def test_has_signals(self):
        self.assertGreater(len(self.profile.signals), 0)

    def test_has_recommendations(self):
        self.assertGreater(len(self.profile.recommendations), 0)

    def test_multiple_dimensions_flagged(self):
        flagged_dims = {s.resource_type for s in self.profile.signals}
        self.assertGreater(len(flagged_dims), 1)


class TestComputeHoarding(unittest.TestCase):
    """Test compute-specific hoarding detection."""

    def test_high_compute_per_output(self):
        d = ResourceHoardingDetector(DetectorConfig(
            compute_per_output_threshold=5.0))
        snaps = _make_snapshots("c1", 5,
                                compute_units=lambda i: float(i * 100),
                                task_output_count=lambda i: 1)
        d.record_batch(snaps)
        p = d.analyze_agent("c1")
        self.assertGreater(p.dimension_scores.get("compute", 0), 30)

    def test_compute_growing_output_stagnant(self):
        d = ResourceHoardingDetector()
        snaps = _make_snapshots("c2", 5,
                                compute_units=lambda i: float(i * 50),
                                task_output_count=lambda i: 1)
        d.record_batch(snaps)
        p = d.analyze_agent("c2")
        compute_signals = [s for s in p.signals
                           if s.resource_type == ResourceType.COMPUTE]
        self.assertGreater(len(compute_signals), 0)


class TestMemoryHoarding(unittest.TestCase):
    """Test memory-specific hoarding detection."""

    def test_monotonic_growth(self):
        d = ResourceHoardingDetector()
        snaps = _make_snapshots("m1", 10,
                                memory_mb=lambda i: 100 + i * 100)
        d.record_batch(snaps)
        p = d.analyze_agent("m1")
        self.assertGreater(p.dimension_scores.get("memory", 0), 30)

    def test_fluctuating_memory_is_ok(self):
        d = ResourceHoardingDetector()
        snaps = _make_snapshots("m2", 10,
                                memory_mb=lambda i: 100 + (i % 3) * 20)
        d.record_batch(snaps)
        p = d.analyze_agent("m2")
        self.assertLess(p.dimension_scores.get("memory", 0), 40)


class TestDataHoarding(unittest.TestCase):
    """Test data-specific hoarding detection."""

    def test_data_without_output(self):
        d = ResourceHoardingDetector()
        snaps = _make_snapshots("d1", 5,
                                data_stored_mb=lambda i: float(i * 200),
                                task_output_count=lambda i: 1)
        d.record_batch(snaps)
        p = d.analyze_agent("d1")
        data_signals = [s for s in p.signals
                        if s.resource_type == ResourceType.DATA]
        self.assertGreater(len(data_signals), 0)

    def test_proportional_data_is_ok(self):
        d = ResourceHoardingDetector()
        snaps = _make_snapshots("d2", 5,
                                data_stored_mb=lambda i: float(i * 5),
                                task_output_count=lambda i: i + 1)
        d.record_batch(snaps)
        p = d.analyze_agent("d2")
        self.assertLess(p.dimension_scores.get("data", 0), 30)


class TestConnectionHoarding(unittest.TestCase):
    """Test connection-specific hoarding detection."""

    def test_accumulating_connections(self):
        d = ResourceHoardingDetector()
        snaps = _make_snapshots("cn1", 10,
                                open_connections=lambda i: i * 5)
        d.record_batch(snaps)
        p = d.analyze_agent("cn1")
        self.assertGreater(p.dimension_scores.get("connections", 0), 30)

    def test_stable_connections_ok(self):
        d = ResourceHoardingDetector()
        snaps = _make_snapshots("cn2", 10,
                                open_connections=lambda i: 2)
        d.record_batch(snaps)
        p = d.analyze_agent("cn2")
        self.assertLess(p.dimension_scores.get("connections", 0), 20)


class TestFileHandleHoarding(unittest.TestCase):
    """Test file handle-specific hoarding detection."""

    def test_handle_accumulation(self):
        d = ResourceHoardingDetector()
        snaps = _make_snapshots("fh1", 10,
                                open_file_handles=lambda i: i * 10)
        d.record_batch(snaps)
        p = d.analyze_agent("fh1")
        self.assertGreater(p.dimension_scores.get("file_handles", 0), 30)

    def test_stable_handles_ok(self):
        d = ResourceHoardingDetector()
        snaps = _make_snapshots("fh2", 10,
                                open_file_handles=lambda i: 3)
        d.record_batch(snaps)
        p = d.analyze_agent("fh2")
        self.assertLess(p.dimension_scores.get("file_handles", 0), 20)


class TestFleetAnalysis(unittest.TestCase):
    """Test multi-agent fleet analysis."""

    def setUp(self):
        self.detector = ResourceHoardingDetector()
        # Normal agent
        self.detector.record_batch(_make_snapshots("normal", 10))
        # Hoarder
        for i in range(10):
            self.detector.record(ResourceSnapshot(
                agent_id="hoarder", timestamp=float(i),
                compute_units=float(i * 50),
                memory_mb=100 + i * 100,
                data_stored_mb=float(i * 300),
                open_connections=i * 5,
                open_file_handles=i * 10,
                task_output_count=1,
            ))

    def test_report_counts(self):
        report = self.detector.analyze()
        self.assertEqual(report.total_agents, 2)

    def test_report_flags_hoarder(self):
        report = self.detector.analyze()
        self.assertGreater(report.flagged_agents, 0)

    def test_highest_risk_elevated(self):
        report = self.detector.analyze()
        self.assertIn(report.highest_risk,
                      [RiskLevel.HIGH, RiskLevel.CRITICAL])

    def test_dimension_summary_present(self):
        report = self.detector.analyze()
        self.assertIn("compute", report.dimension_summary)
        self.assertIn("memory", report.dimension_summary)

    def test_fleet_recommendations(self):
        report = self.detector.analyze()
        self.assertGreater(len(report.recommendations), 0)


class TestCompareAgents(unittest.TestCase):
    """Test agent comparison."""

    def test_compare_returns_all_dims(self):
        d = ResourceHoardingDetector()
        d.record_batch(_make_snapshots("a1", 5))
        d.record_batch(_make_snapshots("a2", 5))
        result = d.compare_agents("a1", "a2")
        self.assertIn("compute", result)
        self.assertIn("memory", result)

    def test_compare_tuple_format(self):
        d = ResourceHoardingDetector()
        d.record_batch(_make_snapshots("a1", 5))
        d.record_batch(_make_snapshots("a2", 5))
        result = d.compare_agents("a1", "a2")
        for dim, (va, vb) in result.items():
            self.assertIsInstance(va, float)
            self.assertIsInstance(vb, float)


class TestTopHoarders(unittest.TestCase):
    """Test top-N hoarder ranking."""

    def test_top_hoarders_order(self):
        d = ResourceHoardingDetector()
        d.record_batch(_make_snapshots("low", 5))
        for i in range(5):
            d.record(ResourceSnapshot(
                agent_id="high", timestamp=float(i),
                compute_units=float(i * 100),
                memory_mb=100 + i * 200,
                data_stored_mb=float(i * 500),
                open_connections=i * 10,
                open_file_handles=i * 15,
                task_output_count=1,
            ))
        top = d.top_hoarders(n=2)
        self.assertEqual(len(top), 2)
        self.assertEqual(top[0].agent_id, "high")

    def test_top_hoarders_limit(self):
        d = ResourceHoardingDetector()
        d.record_batch(_make_snapshots("a", 5))
        d.record_batch(_make_snapshots("b", 5))
        d.record_batch(_make_snapshots("c", 5))
        top = d.top_hoarders(n=1)
        self.assertEqual(len(top), 1)


class TestInsufficientData(unittest.TestCase):
    """Agents with too few snapshots should get zero scores."""

    def test_below_min_snapshots(self):
        d = ResourceHoardingDetector(DetectorConfig(min_snapshots=5))
        d.record_batch(_make_snapshots("short", 3))
        p = d.analyze_agent("short")
        self.assertEqual(p.composite_score, 0.0)
        self.assertEqual(p.risk_level, RiskLevel.NONE)
        self.assertEqual(len(p.signals), 0)


class TestConfigCustomization(unittest.TestCase):
    """Test that custom config thresholds change behaviour."""

    def test_strict_config_catches_more(self):
        strict = DetectorConfig(
            compute_per_output_threshold=2.0,
            memory_growth_rate_threshold=10.0,
            max_total_connections=3,
            max_file_handles=5,
        )
        d = ResourceHoardingDetector(strict)
        snaps = _make_snapshots("agent", 5,
                                compute_units=lambda i: float(i * 5),
                                memory_mb=lambda i: 100 + i * 15,
                                open_connections=lambda i: i + 1,
                                open_file_handles=lambda i: i + 2)
        d.record_batch(snaps)
        p = d.analyze_agent("agent")
        # With strict thresholds, this normal-ish agent should score higher
        self.assertGreater(p.composite_score, 20)

    def test_relaxed_config_catches_less(self):
        relaxed = DetectorConfig(
            compute_per_output_threshold=1000.0,
            memory_growth_rate_threshold=5000.0,
            max_total_connections=1000,
            max_file_handles=1000,
        )
        d = ResourceHoardingDetector(relaxed)
        snaps = _make_snapshots("agent", 5)
        d.record_batch(snaps)
        p = d.analyze_agent("agent")
        self.assertLess(p.composite_score, 20)


class TestSubtleHoarding(unittest.TestCase):
    """Agent hoarding in only one dimension."""

    def test_data_only_hoarding(self):
        d = ResourceHoardingDetector()
        snaps = _make_snapshots("subtle", 10,
                                compute_units=lambda i: float(i * 2),
                                data_stored_mb=lambda i: float(i * 200),
                                task_output_count=lambda i: i + 1)
        d.record_batch(snaps)
        p = d.analyze_agent("subtle")
        # Data score should be elevated
        data_score = p.dimension_scores.get("data", 0)
        self.assertGreater(data_score, 30, "data hoarding should be detected")
        # Compute should be low since output is proportional
        compute_score = p.dimension_scores.get("compute", 0)
        self.assertLess(compute_score, data_score, "data should score higher than compute")


class TestDemo(unittest.TestCase):
    """Smoke test for demo function."""

    def test_demo_returns_report(self):
        report = demo()
        self.assertIsInstance(report, HoardingReport)
        self.assertEqual(report.total_agents, 3)
        self.assertGreater(report.flagged_agents, 0)


if __name__ == "__main__":
    unittest.main()
