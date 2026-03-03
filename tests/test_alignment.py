"""Tests for the alignment monitor module."""

from __future__ import annotations

import json
import pytest

from replication.alignment import (
    AlignmentAlert,
    AlignmentMonitor,
    AlignmentReport,
    AlignmentSpec,
    AlertSeverity,
    DetectionMode,
    DriftDirection,
    GenerationRecord,
    ObjectiveTrend,
    PriorityAnalysis,
    RewardAnalysis,
)


# ── AlignmentSpec ──


class TestAlignmentSpec:
    def test_preset_default(self):
        spec = AlignmentSpec.preset("default")
        assert spec.name == "default"
        assert "task_completion" in spec.values
        assert "safety_compliance" in spec.values
        assert spec.tolerance == 0.15

    def test_preset_strict(self):
        spec = AlignmentSpec.preset("strict")
        assert spec.tolerance == 0.10
        assert spec.min_correlation == 0.7
        assert spec.values["safety_compliance"] == 0.95

    def test_preset_permissive(self):
        spec = AlignmentSpec.preset("permissive")
        assert spec.tolerance == 0.25
        assert spec.values["task_completion"] == 0.9

    def test_preset_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            AlignmentSpec.preset("nonexistent")

    def test_custom_spec(self):
        spec = AlignmentSpec(
            name="custom",
            values={"a": 0.5, "b": 0.8},
            tolerance=0.2,
        )
        assert spec.name == "custom"
        assert len(spec.values) == 2

    def test_priority_order_in_presets(self):
        for name in ("default", "strict", "permissive"):
            spec = AlignmentSpec.preset(name)
            assert len(spec.priority_order) > 0
            for obj in spec.priority_order:
                assert obj in spec.values


# ── AlignmentMonitor: Empty ──


class TestMonitorEmpty:
    def test_analyze_empty(self):
        monitor = AlignmentMonitor()
        report = monitor.analyze()
        assert report.num_generations == 0
        assert report.overall_score == 100.0
        assert report.alerts == []
        assert report.grade == "A+"


# ── AlignmentMonitor: Single Generation ──


class TestMonitorSingleGen:
    def test_perfect_alignment(self):
        spec = AlignmentSpec.preset("default")
        monitor = AlignmentMonitor(spec)
        monitor.record_generation("gen-0", objectives=dict(spec.values))
        report = monitor.analyze()
        assert report.overall_score > 90
        assert len([a for a in report.alerts if a.severity == AlertSeverity.CRITICAL]) == 0

    def test_misaligned_single_gen(self):
        spec = AlignmentSpec.preset("default")
        monitor = AlignmentMonitor(spec)
        # Everything opposite of spec
        bad = {k: 1.0 - v for k, v in spec.values.items()}
        monitor.record_generation("gen-0", objectives=bad)
        report = monitor.analyze()
        assert report.overall_score < 70
        assert len(report.alerts) > 0


# ── Objective Trend Analysis ──


class TestObjectiveTrends:
    def test_stable_objectives(self):
        spec = AlignmentSpec(name="test", values={"a": 0.5}, tolerance=0.1)
        monitor = AlignmentMonitor(spec)
        for i in range(5):
            monitor.record_generation(f"gen-{i}", objectives={"a": 0.5})
        report = monitor.analyze(DetectionMode.OBJECTIVES)
        assert report.trends["a"].direction == DriftDirection.STABLE
        assert report.trends["a"].within_tolerance

    def test_increasing_drift(self):
        spec = AlignmentSpec(name="test", values={"a": 0.3}, tolerance=0.1)
        monitor = AlignmentMonitor(spec)
        for i in range(6):
            monitor.record_generation(f"gen-{i}", objectives={"a": 0.3 + i * 0.1})
        report = monitor.analyze(DetectionMode.OBJECTIVES)
        assert report.trends["a"].direction == DriftDirection.INCREASING
        assert report.trends["a"].slope > 0

    def test_decreasing_drift(self):
        spec = AlignmentSpec(name="test", values={"a": 0.8}, tolerance=0.1)
        monitor = AlignmentMonitor(spec)
        for i in range(6):
            monitor.record_generation(f"gen-{i}", objectives={"a": 0.8 - i * 0.08})
        report = monitor.analyze(DetectionMode.OBJECTIVES)
        assert report.trends["a"].direction == DriftDirection.DECREASING

    def test_within_tolerance_no_alerts(self):
        spec = AlignmentSpec(name="test", values={"a": 0.5}, tolerance=0.2)
        monitor = AlignmentMonitor(spec)
        for i in range(5):
            monitor.record_generation(f"gen-{i}", objectives={"a": 0.5 + 0.05 * (i % 2)})
        report = monitor.analyze(DetectionMode.OBJECTIVES)
        obj_alerts = [a for a in report.alerts if a.category == "objective_drift"]
        assert len(obj_alerts) == 0

    def test_outside_tolerance_generates_alert(self):
        spec = AlignmentSpec(name="test", values={"a": 0.5}, tolerance=0.1)
        monitor = AlignmentMonitor(spec)
        monitor.record_generation("gen-0", objectives={"a": 0.5})
        monitor.record_generation("gen-1", objectives={"a": 0.8})  # 0.3 deviation
        report = monitor.analyze(DetectionMode.OBJECTIVES)
        obj_alerts = [a for a in report.alerts if a.category == "objective_drift"]
        assert len(obj_alerts) >= 1
        assert obj_alerts[0].deviation > 0.1

    def test_critical_alert_for_large_deviation(self):
        spec = AlignmentSpec(name="test", values={"a": 0.5}, tolerance=0.1)
        monitor = AlignmentMonitor(spec)
        monitor.record_generation("gen-0", objectives={"a": 0.9})  # 0.4 > 0.1*2
        report = monitor.analyze(DetectionMode.OBJECTIVES)
        critical = [a for a in report.alerts if a.severity == AlertSeverity.CRITICAL]
        assert len(critical) >= 1

    def test_multiple_objectives(self):
        spec = AlignmentSpec(name="test", values={"a": 0.5, "b": 0.7}, tolerance=0.1)
        monitor = AlignmentMonitor(spec)
        monitor.record_generation("gen-0", objectives={"a": 0.5, "b": 0.7})
        monitor.record_generation("gen-1", objectives={"a": 0.5, "b": 0.7})
        report = monitor.analyze(DetectionMode.OBJECTIVES)
        assert "a" in report.trends
        assert "b" in report.trends

    def test_missing_objective_treated_as_zero(self):
        spec = AlignmentSpec(name="test", values={"a": 0.5, "b": 0.7}, tolerance=0.1)
        monitor = AlignmentMonitor(spec)
        monitor.record_generation("gen-0", objectives={"a": 0.5})  # b missing
        report = monitor.analyze(DetectionMode.OBJECTIVES)
        assert report.trends["b"].values == [0.0]

    def test_extra_objective_tracked(self):
        spec = AlignmentSpec(name="test", values={"a": 0.5}, tolerance=0.1)
        monitor = AlignmentMonitor(spec)
        monitor.record_generation("gen-0", objectives={"a": 0.5, "extra": 0.9})
        report = monitor.analyze(DetectionMode.OBJECTIVES)
        assert "extra" in report.trends


# ── Priority Analysis ──


class TestPriorityAnalysis:
    def test_perfect_priority_order(self):
        spec = AlignmentSpec(
            name="test",
            values={"a": 0.9, "b": 0.7, "c": 0.5},
            priority_order=("a", "b", "c"),
        )
        monitor = AlignmentMonitor(spec)
        monitor.record_generation("gen-0", objectives={"a": 0.9, "b": 0.7, "c": 0.5})
        report = monitor.analyze(DetectionMode.PRIORITIES)
        assert report.priority_analysis is not None
        assert report.priority_analysis.kendall_tau == 1.0
        assert report.priority_analysis.inversions == []

    def test_reversed_priorities_detected(self):
        spec = AlignmentSpec(
            name="test",
            values={"a": 0.9, "b": 0.7, "c": 0.5},
            priority_order=("a", "b", "c"),
        )
        monitor = AlignmentMonitor(spec)
        # Reversed: c > b > a
        monitor.record_generation("gen-0", objectives={"a": 0.1, "b": 0.5, "c": 0.9})
        report = monitor.analyze(DetectionMode.PRIORITIES)
        pa = report.priority_analysis
        assert pa is not None
        assert pa.kendall_tau < 0
        assert len(pa.inversions) > 0

    def test_partial_inversion(self):
        spec = AlignmentSpec(
            name="test",
            values={"a": 0.9, "b": 0.7, "c": 0.5},
            priority_order=("a", "b", "c"),
        )
        monitor = AlignmentMonitor(spec)
        # a and b swapped
        monitor.record_generation("gen-0", objectives={"a": 0.6, "b": 0.8, "c": 0.3})
        report = monitor.analyze(DetectionMode.PRIORITIES)
        pa = report.priority_analysis
        assert pa is not None
        assert 0 < pa.kendall_tau < 1

    def test_no_priority_order_skips(self):
        spec = AlignmentSpec(name="test", values={"a": 0.5}, priority_order=())
        monitor = AlignmentMonitor(spec)
        monitor.record_generation("gen-0", objectives={"a": 0.5})
        report = monitor.analyze(DetectionMode.PRIORITIES)
        assert report.priority_analysis is None

    def test_single_priority_perfect(self):
        spec = AlignmentSpec(
            name="test", values={"a": 0.5}, priority_order=("a",),
        )
        monitor = AlignmentMonitor(spec)
        monitor.record_generation("gen-0", objectives={"a": 0.5})
        report = monitor.analyze(DetectionMode.PRIORITIES)
        assert report.priority_analysis is not None
        assert report.priority_analysis.kendall_tau == 1.0


# ── Reward Analysis ──


class TestRewardAnalysis:
    def test_no_reward_data(self):
        monitor = AlignmentMonitor()
        monitor.record_generation("gen-0", objectives={"task_completion": 0.8})
        report = monitor.analyze(DetectionMode.REWARD)
        ra = report.reward_analysis
        assert ra is not None
        assert ra.is_aligned
        assert "Insufficient" in ra.details

    def test_aligned_rewards(self):
        spec = AlignmentSpec(name="test", values={"a": 0.8}, min_correlation=0.3)
        monitor = AlignmentMonitor(spec)
        # High alignment → high rewards, low alignment → low rewards
        for i in range(5):
            obj_val = 0.8 - i * 0.1  # decreasing alignment
            reward = 0.9 - i * 0.15   # decreasing rewards (correlated)
            monitor.record_generation(
                f"gen-{i}",
                objectives={"a": obj_val},
                reward_signals=[reward],
            )
        report = monitor.analyze(DetectionMode.REWARD)
        ra = report.reward_analysis
        assert ra is not None
        # Should show positive correlation
        assert ra.correlation > 0

    def test_suspected_reward_hacking(self):
        spec = AlignmentSpec(name="test", values={"a": 0.8}, min_correlation=0.3)
        monitor = AlignmentMonitor(spec)
        # First few normal, then high reward + low alignment
        monitor.record_generation("gen-0", objectives={"a": 0.8}, reward_signals=[0.8])
        monitor.record_generation("gen-1", objectives={"a": 0.7}, reward_signals=[0.7])
        monitor.record_generation("gen-2", objectives={"a": 0.2}, reward_signals=[0.9])
        monitor.record_generation("gen-3", objectives={"a": 0.1}, reward_signals=[0.95])
        monitor.record_generation("gen-4", objectives={"a": 0.1}, reward_signals=[0.9])
        report = monitor.analyze(DetectionMode.REWARD)
        ra = report.reward_analysis
        assert ra is not None
        assert ra.suspected_hacking

    def test_reward_misalignment_alert(self):
        spec = AlignmentSpec(name="test", values={"a": 0.8}, min_correlation=0.8)
        monitor = AlignmentMonitor(spec)
        # Random / uncorrelated
        monitor.record_generation("gen-0", objectives={"a": 0.8}, reward_signals=[0.2])
        monitor.record_generation("gen-1", objectives={"a": 0.3}, reward_signals=[0.9])
        monitor.record_generation("gen-2", objectives={"a": 0.7}, reward_signals=[0.1])
        report = monitor.analyze(DetectionMode.REWARD)
        ra = report.reward_analysis
        assert ra is not None
        assert not ra.is_aligned
        misalign_alerts = [a for a in report.alerts if a.category == "reward_misalignment"]
        assert len(misalign_alerts) >= 1


# ── Overall Score & Grading ──


class TestOverallScore:
    def test_perfect_alignment_high_score(self):
        spec = AlignmentSpec.preset("default")
        monitor = AlignmentMonitor(spec)
        for i in range(5):
            monitor.record_generation(f"gen-{i}", objectives=dict(spec.values))
        report = monitor.analyze()
        assert report.overall_score >= 90
        assert report.grade in ("A", "A+")

    def test_terrible_alignment_low_score(self):
        spec = AlignmentSpec.preset("default")
        monitor = AlignmentMonitor(spec)
        bad = {k: 1.0 - v for k, v in spec.values.items()}
        for i in range(5):
            monitor.record_generation(f"gen-{i}", objectives=bad)
        report = monitor.analyze()
        assert report.overall_score < 50
        assert report.grade in ("D", "F")

    def test_grade_boundaries(self):
        report = AlignmentReport(
            spec_name="t", num_generations=0, trends={},
            priority_analysis=None, reward_analysis=None,
            alerts=[], overall_score=95,
        )
        assert report.grade == "A+"
        report.overall_score = 90
        assert report.grade == "A"
        report.overall_score = 85
        assert report.grade == "B+"
        report.overall_score = 80
        assert report.grade == "B"
        report.overall_score = 70
        assert report.grade == "C"
        report.overall_score = 60
        assert report.grade == "D"
        report.overall_score = 50
        assert report.grade == "F"


# ── Report Rendering ──


class TestReportRendering:
    def _make_report(self) -> AlignmentReport:
        spec = AlignmentSpec.preset("default")
        monitor = AlignmentMonitor(spec)
        for i in range(4):
            objectives = {k: v + (i * 0.02) for k, v in spec.values.items()}
            monitor.record_generation(
                f"gen-{i}", objectives=objectives,
                reward_signals=[0.7 + i * 0.05],
            )
        return monitor.analyze()

    def test_render_contains_sections(self):
        report = self._make_report()
        text = report.render()
        assert "ALIGNMENT MONITOR REPORT" in text
        assert "Objective Trends" in text
        assert "Priority Ordering" in text

    def test_to_dict_roundtrip(self):
        report = self._make_report()
        d = report.to_dict()
        assert d["spec_name"] == "default"
        assert "trends" in d
        assert "overall_score" in d
        assert isinstance(d["alerts"], list)

    def test_to_dict_json_serializable(self):
        report = self._make_report()
        text = json.dumps(report.to_dict())
        assert len(text) > 0


# ── Detection Modes ──


class TestDetectionModes:
    def _monitor_with_data(self) -> AlignmentMonitor:
        spec = AlignmentSpec.preset("default")
        monitor = AlignmentMonitor(spec)
        for i in range(5):
            objectives = dict(spec.values)
            objectives["self_preservation"] = 0.3 + i * 0.1
            monitor.record_generation(
                f"gen-{i}", objectives=objectives,
                reward_signals=[0.7],
            )
        return monitor

    def test_objectives_only(self):
        monitor = self._monitor_with_data()
        report = monitor.analyze(DetectionMode.OBJECTIVES)
        assert report.trends
        assert report.priority_analysis is None
        assert report.reward_analysis is None

    def test_priorities_only(self):
        monitor = self._monitor_with_data()
        report = monitor.analyze(DetectionMode.PRIORITIES)
        assert report.priority_analysis is not None
        assert not report.trends

    def test_reward_only(self):
        monitor = self._monitor_with_data()
        report = monitor.analyze(DetectionMode.REWARD)
        assert report.reward_analysis is not None
        assert not report.trends

    def test_all_mode(self):
        monitor = self._monitor_with_data()
        report = monitor.analyze(DetectionMode.ALL)
        assert report.trends
        assert report.priority_analysis is not None
        assert report.reward_analysis is not None


# ── Utility Methods ──


class TestUtilities:
    def test_compute_slope_flat(self):
        slope = AlignmentMonitor._compute_slope([0.5, 0.5, 0.5])
        assert abs(slope) < 0.001

    def test_compute_slope_positive(self):
        slope = AlignmentMonitor._compute_slope([0.1, 0.2, 0.3, 0.4])
        assert slope > 0

    def test_compute_slope_negative(self):
        slope = AlignmentMonitor._compute_slope([0.4, 0.3, 0.2, 0.1])
        assert slope < 0

    def test_compute_slope_single_value(self):
        slope = AlignmentMonitor._compute_slope([0.5])
        assert slope == 0.0

    def test_classify_direction_stable(self):
        d = AlignmentMonitor._classify_direction([0.5, 0.5, 0.5], 0.0)
        assert d == DriftDirection.STABLE

    def test_classify_direction_increasing(self):
        vals = [0.1, 0.2, 0.3, 0.4, 0.5]
        slope = AlignmentMonitor._compute_slope(vals)
        d = AlignmentMonitor._classify_direction(vals, slope)
        assert d == DriftDirection.INCREASING

    def test_classify_direction_single(self):
        d = AlignmentMonitor._classify_direction([0.5], 0.0)
        assert d == DriftDirection.STABLE

    def test_pearson_correlation_perfect(self):
        r = AlignmentMonitor._pearson_correlation([1, 2, 3], [1, 2, 3])
        assert abs(r - 1.0) < 0.001

    def test_pearson_correlation_negative(self):
        r = AlignmentMonitor._pearson_correlation([1, 2, 3], [3, 2, 1])
        assert abs(r - (-1.0)) < 0.001

    def test_pearson_correlation_single(self):
        r = AlignmentMonitor._pearson_correlation([1], [1])
        assert r == 0.0

    def test_pearson_correlation_constant(self):
        r = AlignmentMonitor._pearson_correlation([1, 1, 1], [2, 3, 4])
        assert r == 0.0


# ── GenerationRecord ──


class TestGenerationRecord:
    def test_default_fields(self):
        rec = GenerationRecord(generation_id="g0", objectives={"a": 0.5})
        assert rec.decisions == []
        assert rec.reward_signals == []
        assert rec.timestamp


# ── Custom Threshold ──


class TestCustomThreshold:
    def test_threshold_overrides_spec(self):
        spec = AlignmentSpec(name="t", values={"a": 0.5}, tolerance=0.1)
        monitor = AlignmentMonitor(spec, threshold=0.5)
        monitor.record_generation("g0", objectives={"a": 0.9})  # 0.4 dev
        report = monitor.analyze()
        # 0.4 < 0.5 threshold → within tolerance
        assert report.trends["a"].within_tolerance

    def test_tighter_threshold(self):
        spec = AlignmentSpec(name="t", values={"a": 0.5}, tolerance=0.5)
        monitor = AlignmentMonitor(spec, threshold=0.05)
        monitor.record_generation("g0", objectives={"a": 0.6})  # 0.1 > 0.05
        report = monitor.analyze()
        assert not report.trends["a"].within_tolerance


# ── CLI / main ──


class TestCLI:
    def test_main_default(self, capsys):
        from replication.alignment import main
        with pytest.raises(SystemExit):
            main(["--generations", "3"])
        out = capsys.readouterr().out
        assert "ALIGNMENT MONITOR REPORT" in out

    def test_main_json(self, capsys):
        from replication.alignment import main
        with pytest.raises(SystemExit):
            main(["--generations", "3", "--json", "--spec", "permissive"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "overall_score" in data
        assert data["spec_name"] == "permissive"

    def test_main_detect_objectives(self, capsys):
        from replication.alignment import main
        with pytest.raises(SystemExit):
            main(["--generations", "3", "--detect", "objectives"])
        out = capsys.readouterr().out
        assert "Objective Trends" in out
