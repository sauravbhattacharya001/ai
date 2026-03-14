"""Tests for the SafetyBenchmark module."""

from __future__ import annotations

import json
import pytest

from replication.safety_benchmark import (
    BenchmarkConfig,
    BenchmarkGrade,
    BenchmarkReport,
    BenchmarkSuite,
    ComparisonReport,
    ControlBenchmarkResult,
    ControlUnderTest,
    LatencyStats,
    SafetyBenchmark,
    WorkloadIntensity,
    _assign_grade,
    _compute_latency,
    _overall_grade,
    _percentile,
)


SEED = 42


def _quick_config(**overrides) -> BenchmarkConfig:
    defaults = dict(
        suite=BenchmarkSuite.MINIMAL,
        iterations=5,
        warmup_iterations=1,
        intensity=WorkloadIntensity.LOW,
        seed=SEED,
    )
    defaults.update(overrides)
    return BenchmarkConfig(**defaults)


# ---------------------------------------------------------------------------
# Percentile / Latency
# ---------------------------------------------------------------------------


class TestPercentile:
    def test_empty(self):
        assert _percentile([], 50) == 0.0

    def test_single(self):
        assert _percentile([5.0], 50) == 5.0

    def test_median(self):
        assert _percentile([1, 2, 3, 4, 5], 50) == 3.0

    def test_p99(self):
        vals = list(range(1, 101))
        p99 = _percentile(vals, 99)
        assert 99 <= p99 <= 100


class TestComputeLatency:
    def test_empty(self):
        stats = _compute_latency([])
        assert stats.mean_ms == 0.0
        assert stats.p50_ms == 0.0

    def test_basic(self):
        stats = _compute_latency([10.0, 20.0, 30.0])
        assert stats.min_ms == 10.0
        assert stats.max_ms == 30.0
        assert 19 < stats.mean_ms < 21
        assert stats.p50_ms == 20.0

    def test_single_value(self):
        stats = _compute_latency([42.0])
        assert stats.std_ms == 0.0
        assert stats.p50_ms == 42.0


# ---------------------------------------------------------------------------
# ControlBenchmarkResult
# ---------------------------------------------------------------------------


class TestControlBenchmarkResult:
    def test_accuracy(self):
        r = ControlBenchmarkResult(
            control=ControlUnderTest.KILL_SWITCH, workload="test",
            iterations=10, true_positives=8, true_negatives=1,
            false_positives=0, false_negatives=1,
        )
        assert r.accuracy == 0.9
        assert r.total_checks == 10

    def test_precision_recall_f1(self):
        r = ControlBenchmarkResult(
            control=ControlUnderTest.QUARANTINE, workload="test",
            iterations=10, true_positives=6, true_negatives=2,
            false_positives=1, false_negatives=1,
        )
        assert r.precision == 6 / 7
        assert r.recall == 6 / 7
        assert r.f1_score > 0

    def test_zero_division(self):
        r = ControlBenchmarkResult(
            control=ControlUnderTest.DEPTH_LIMIT, workload="test",
            iterations=0,
        )
        assert r.accuracy == 0.0
        assert r.precision == 0.0
        assert r.recall == 0.0
        assert r.f1_score == 0.0
        assert r.false_positive_rate == 0.0
        assert r.false_negative_rate == 0.0

    def test_false_positive_rate(self):
        r = ControlBenchmarkResult(
            control=ControlUnderTest.CANARY_DETECTION, workload="test",
            iterations=10, true_positives=3, true_negatives=5,
            false_positives=2, false_negatives=0,
        )
        assert r.false_positive_rate == 2 / 7

    def test_to_dict(self):
        r = ControlBenchmarkResult(
            control=ControlUnderTest.KILL_SWITCH, workload="test",
            iterations=5, true_positives=4, true_negatives=1,
            grade=BenchmarkGrade.A, passed=True,
        )
        d = r.to_dict()
        assert d["control"] == "kill_switch"
        assert d["grade"] == "A"
        assert d["passed"] is True
        assert "accuracy" in d


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


class TestGrading:
    def test_perfect_gets_a_plus(self):
        r = ControlBenchmarkResult(
            control=ControlUnderTest.KILL_SWITCH, workload="test",
            iterations=100,
            latency=LatencyStats(p50_ms=2.0),
            true_positives=50, true_negatives=50,
        )
        assert _assign_grade(r) == BenchmarkGrade.A_PLUS

    def test_poor_gets_f(self):
        r = ControlBenchmarkResult(
            control=ControlUnderTest.QUARANTINE, workload="test",
            iterations=10,
            latency=LatencyStats(p50_ms=500.0),
            true_positives=1, true_negatives=1,
            false_positives=4, false_negatives=4,
        )
        assert _assign_grade(r) == BenchmarkGrade.F

    def test_overall_grade_worst(self):
        results = [
            ControlBenchmarkResult(
                control=ControlUnderTest.KILL_SWITCH, workload="t",
                iterations=10, grade=BenchmarkGrade.A,
            ),
            ControlBenchmarkResult(
                control=ControlUnderTest.QUARANTINE, workload="t",
                iterations=10, grade=BenchmarkGrade.C,
            ),
        ]
        assert _overall_grade(results) == BenchmarkGrade.C

    def test_overall_empty(self):
        assert _overall_grade([]) == BenchmarkGrade.F


# ---------------------------------------------------------------------------
# Benchmark Engine
# ---------------------------------------------------------------------------


class TestSafetyBenchmark:
    def test_run_one(self):
        bench = SafetyBenchmark(seed=SEED)
        result = bench.run_one(
            ControlUnderTest.KILL_SWITCH,
            _quick_config(iterations=5),
        )
        assert result.control == ControlUnderTest.KILL_SWITCH
        assert result.iterations == 5
        assert result.total_checks == 5
        assert result.latency.p50_ms > 0
        assert result.grade in BenchmarkGrade

    def test_run_one_default_config(self):
        bench = SafetyBenchmark(seed=SEED)
        result = bench.run_one(ControlUnderTest.DEPTH_LIMIT)
        assert result.iterations > 0

    def test_run_suite_minimal(self):
        bench = SafetyBenchmark(seed=SEED)
        config = _quick_config(suite=BenchmarkSuite.MINIMAL, iterations=3)
        report = bench.run_suite(BenchmarkSuite.MINIMAL, config)
        assert len(report.results) == 2
        assert report.grade in BenchmarkGrade
        assert report.total_duration_ms > 0
        assert report.timestamp

    def test_run_suite_quick(self):
        bench = SafetyBenchmark(seed=SEED)
        config = _quick_config(suite=BenchmarkSuite.QUICK, iterations=3)
        report = bench.run_suite(BenchmarkSuite.QUICK, config)
        assert len(report.results) == 4

    def test_run_suite_with_control_filter(self):
        bench = SafetyBenchmark(seed=SEED)
        config = _quick_config(
            controls=[ControlUnderTest.KILL_SWITCH],
            iterations=3,
        )
        report = bench.run_suite(BenchmarkSuite.STANDARD, config)
        assert len(report.results) == 1
        assert report.results[0].control == ControlUnderTest.KILL_SWITCH

    def test_all_controls_have_runners(self):
        bench = SafetyBenchmark(seed=SEED)
        for control in ControlUnderTest:
            result = bench.run_one(control, _quick_config(iterations=3))
            assert result.control == control
            assert result.iterations == 3

    def test_summary_text(self):
        bench = SafetyBenchmark(seed=SEED)
        report = bench.run_suite(
            BenchmarkSuite.MINIMAL,
            _quick_config(iterations=3),
        )
        text = report.summary()
        assert "Safety Benchmark Report" in text
        assert "Grade:" in text

    def test_reproducibility_with_seed(self):
        bench1 = SafetyBenchmark(seed=123)
        bench2 = SafetyBenchmark(seed=123)
        cfg = BenchmarkConfig(iterations=5, warmup_iterations=0, seed=123)
        r1 = bench1.run_one(ControlUnderTest.REPLICATION_LIMIT, cfg)
        r2 = bench2.run_one(ControlUnderTest.REPLICATION_LIMIT, cfg)
        assert r1.true_positives == r2.true_positives
        assert r1.false_positives == r2.false_positives

    def test_intensity_affects_latency(self):
        bench = SafetyBenchmark(seed=SEED)
        low = bench.run_one(
            ControlUnderTest.QUARANTINE,
            _quick_config(intensity=WorkloadIntensity.LOW, iterations=5),
        )
        extreme = bench.run_one(
            ControlUnderTest.QUARANTINE,
            _quick_config(intensity=WorkloadIntensity.EXTREME, iterations=5),
        )
        assert extreme.latency.mean_ms > low.latency.mean_ms * 0.5


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


class TestComparison:
    def test_compare_stable(self):
        bench = SafetyBenchmark(seed=SEED)
        config = _quick_config(iterations=5)
        r1 = bench.run_suite(BenchmarkSuite.MINIMAL, config)
        r2 = bench.run_suite(BenchmarkSuite.MINIMAL, config)
        diff = bench.compare(r1, r2)
        assert diff.verdict == "stable"
        assert len(diff.regressions) == 0

    def test_compare_detects_regression(self):
        good = BenchmarkReport(
            suite="test", config=BenchmarkConfig(),
            results=[ControlBenchmarkResult(
                control=ControlUnderTest.KILL_SWITCH, workload="test",
                iterations=10, true_positives=9, true_negatives=1,
                latency=LatencyStats(p50_ms=5.0),
                grade=BenchmarkGrade.A,
            )],
            grade=BenchmarkGrade.A,
        )
        bad = BenchmarkReport(
            suite="test", config=BenchmarkConfig(),
            results=[ControlBenchmarkResult(
                control=ControlUnderTest.KILL_SWITCH, workload="test",
                iterations=10, true_positives=4, true_negatives=1,
                false_positives=3, false_negatives=2,
                latency=LatencyStats(p50_ms=50.0),
                grade=BenchmarkGrade.D,
            )],
            grade=BenchmarkGrade.D,
        )
        bench = SafetyBenchmark()
        diff = bench.compare(good, bad)
        assert diff.verdict == "regressed"
        assert len(diff.regressions) > 0

    def test_compare_detects_improvement(self):
        bad = BenchmarkReport(
            suite="test", config=BenchmarkConfig(),
            results=[ControlBenchmarkResult(
                control=ControlUnderTest.QUARANTINE, workload="test",
                iterations=10, true_positives=4, true_negatives=1,
                false_positives=3, false_negatives=2,
                latency=LatencyStats(p50_ms=50.0),
                grade=BenchmarkGrade.D,
            )],
            grade=BenchmarkGrade.D,
        )
        good = BenchmarkReport(
            suite="test", config=BenchmarkConfig(),
            results=[ControlBenchmarkResult(
                control=ControlUnderTest.QUARANTINE, workload="test",
                iterations=10, true_positives=9, true_negatives=1,
                latency=LatencyStats(p50_ms=5.0),
                grade=BenchmarkGrade.A,
            )],
            grade=BenchmarkGrade.A,
        )
        bench = SafetyBenchmark()
        diff = bench.compare(bad, good)
        assert diff.verdict == "improved"
        assert len(diff.improvements) > 0


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


class TestSerialisation:
    def test_to_json_and_back(self):
        bench = SafetyBenchmark(seed=SEED)
        config = _quick_config(iterations=3)
        original = bench.run_suite(BenchmarkSuite.MINIMAL, config)

        json_str = original.to_json()
        restored = BenchmarkReport.from_json(json_str)

        assert restored.suite == original.suite
        assert restored.grade == original.grade
        assert len(restored.results) == len(original.results)
        for orig_r, rest_r in zip(original.results, restored.results):
            assert rest_r.control == orig_r.control
            assert rest_r.true_positives == orig_r.true_positives

    def test_to_json_file(self, tmp_path):
        bench = SafetyBenchmark(seed=SEED)
        report = bench.run_suite(
            BenchmarkSuite.MINIMAL,
            _quick_config(iterations=3),
        )
        path = str(tmp_path / "bench.json")
        report.to_json(path)

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["suite"] == "minimal"
        assert "results" in data

    def test_comparison_to_dict(self):
        diff = ComparisonReport(
            baseline_grade=BenchmarkGrade.A,
            candidate_grade=BenchmarkGrade.B,
            verdict="regressed",
        )
        d = diff.to_dict()
        assert d["baseline_grade"] == "A"
        assert d["verdict"] == "regressed"

    def test_latency_stats_to_dict(self):
        lat = LatencyStats(p50_ms=5.5, p90_ms=10.2, p99_ms=15.7,
                           min_ms=1.0, max_ms=20.0, mean_ms=7.3, std_ms=3.1)
        d = lat.to_dict()
        assert d["p50_ms"] == 5.5
        assert d["std_ms"] == 3.1


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_zero_iterations(self):
        bench = SafetyBenchmark(seed=SEED)
        config = _quick_config(iterations=0)
        result = bench.run_one(ControlUnderTest.KILL_SWITCH, config)
        assert result.total_checks == 0

    def test_from_json_preserves_notes(self):
        r = ControlBenchmarkResult(
            control=ControlUnderTest.KILL_SWITCH, workload="test",
            iterations=1, notes=["latency spike"],
            grade=BenchmarkGrade.B, passed=True,
        )
        report = BenchmarkReport(
            suite="test", config=BenchmarkConfig(),
            results=[r], grade=BenchmarkGrade.B,
        )
        text = report.to_json()
        restored = BenchmarkReport.from_json(text)
        assert restored.results[0].notes == ["latency spike"]

    def test_workload_dataclass(self):
        from replication.safety_benchmark import Workload
        w = Workload(name="test", control=ControlUnderTest.KILL_SWITCH)
        assert w.iterations == 10
        assert w.intensity == WorkloadIntensity.MEDIUM
