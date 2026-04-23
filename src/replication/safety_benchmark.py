"""Safety Benchmark — standardised performance benchmarks for safety controls.

Evaluates detection latency, false-positive/negative rates, coverage,
throughput, and resource cost of the sandbox's safety controls under
controlled, reproducible workloads.  Produces benchmark reports with
grading, comparisons across configurations, and regression detection.

Use it to answer: "Are our safety controls fast enough, accurate enough,
and complete enough?"

Usage (Python API)::

    from replication.safety_benchmark import (
        SafetyBenchmark, BenchmarkSuite, BenchmarkConfig,
        ControlUnderTest, Workload,
    )

    bench = SafetyBenchmark()

    # Run the full suite
    report = bench.run_suite(BenchmarkSuite.STANDARD)
    print(report.summary())
    print(f"Overall grade: {report.grade}")

    # Benchmark a single control
    result = bench.run_one(ControlUnderTest.KILL_SWITCH)
    print(f"Latency p50: {result.latency_p50_ms:.1f}ms")
    print(f"False positive rate: {result.false_positive_rate:.2%}")

    # Compare two configurations
    baseline = bench.run_suite(BenchmarkSuite.STANDARD)
    candidate = bench.run_suite(BenchmarkSuite.STANDARD)
    diff = bench.compare(baseline, candidate)
    for d in diff.regressions:
        print(f"⚠ {d.control}: {d.metric} regressed {d.delta:+.1f}")

    # Export
    report.to_json("benchmark.json")

CLI::

    python -m replication safety-benchmark --suite standard
    python -m replication safety-benchmark --control kill-switch
    python -m replication safety-benchmark --suite stress --iterations 50
    python -m replication safety-benchmark --compare baseline.json candidate.json
    python -m replication safety-benchmark --format json --output bench.json
"""

from __future__ import annotations

import hashlib
import json
import math
import random as _random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from ._helpers import box_header, stats_mean, stats_std, percentile_sorted as _percentile_sorted


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ControlUnderTest(Enum):
    """Safety controls that can be benchmarked."""

    KILL_SWITCH = "kill_switch"
    QUARANTINE = "quarantine"
    DRIFT_DETECTION = "drift_detection"
    REPLICATION_LIMIT = "replication_limit"
    DEPTH_LIMIT = "depth_limit"
    COMPLIANCE_AUDIT = "compliance_audit"
    CANARY_DETECTION = "canary_detection"
    ESCALATION_DETECTION = "escalation_detection"


class BenchmarkSuite(Enum):
    """Pre-defined benchmark suites."""

    QUICK = "quick"
    STANDARD = "standard"
    STRESS = "stress"
    MINIMAL = "minimal"


class BenchmarkGrade(Enum):
    """Letter grades for benchmark results."""

    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


class WorkloadIntensity(Enum):
    """Intensity level for benchmark workloads."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class Workload:
    """A benchmark workload definition."""

    name: str
    control: ControlUnderTest
    intensity: WorkloadIntensity = WorkloadIntensity.MEDIUM
    iterations: int = 10
    agent_count: int = 5
    concurrent: bool = False
    inject_false_positives: int = 0
    inject_false_negatives: int = 0
    description: str = ""


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    suite: BenchmarkSuite = BenchmarkSuite.STANDARD
    iterations: int = 10
    warmup_iterations: int = 2
    controls: Optional[List[ControlUnderTest]] = None
    intensity: WorkloadIntensity = WorkloadIntensity.MEDIUM
    agent_count: int = 5
    seed: Optional[int] = None


@dataclass
class LatencyStats:
    """Latency statistics for a benchmark."""

    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p99_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    mean_ms: float = 0.0
    std_ms: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "p50_ms": round(self.p50_ms, 2),
            "p90_ms": round(self.p90_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "mean_ms": round(self.mean_ms, 2),
            "std_ms": round(self.std_ms, 2),
        }


@dataclass
class ControlBenchmarkResult:
    """Result of benchmarking a single safety control."""

    control: ControlUnderTest
    workload: str
    iterations: int
    latency: LatencyStats = field(default_factory=LatencyStats)
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    throughput_ops_sec: float = 0.0
    resource_cost_score: float = 0.0
    grade: BenchmarkGrade = BenchmarkGrade.F
    passed: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def total_checks(self) -> int:
        return (self.true_positives + self.true_negatives
                + self.false_positives + self.false_negatives)

    @property
    def accuracy(self) -> float:
        total = self.total_checks
        if total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / total

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def false_positive_rate(self) -> float:
        denom = self.false_positives + self.true_negatives
        return self.false_positives / denom if denom > 0 else 0.0

    @property
    def false_negative_rate(self) -> float:
        denom = self.false_negatives + self.true_positives
        return self.false_negatives / denom if denom > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "control": self.control.value,
            "workload": self.workload,
            "iterations": self.iterations,
            "latency": self.latency.to_dict(),
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "false_negative_rate": round(self.false_negative_rate, 4),
            "throughput_ops_sec": round(self.throughput_ops_sec, 2),
            "resource_cost_score": round(self.resource_cost_score, 4),
            "grade": self.grade.value,
            "passed": self.passed,
            "notes": self.notes,
        }


@dataclass
class RegressionItem:
    """A single regression detected in a comparison."""

    control: str
    metric: str
    baseline_value: float
    candidate_value: float
    delta: float
    severity: str


@dataclass
class ComparisonReport:
    """Comparison of two benchmark runs."""

    baseline_grade: BenchmarkGrade
    candidate_grade: BenchmarkGrade
    regressions: List[RegressionItem] = field(default_factory=list)
    improvements: List[RegressionItem] = field(default_factory=list)
    unchanged: List[str] = field(default_factory=list)
    verdict: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_grade": self.baseline_grade.value,
            "candidate_grade": self.candidate_grade.value,
            "regressions": [
                {"control": r.control, "metric": r.metric,
                 "baseline": r.baseline_value, "candidate": r.candidate_value,
                 "delta": round(r.delta, 4), "severity": r.severity}
                for r in self.regressions
            ],
            "improvements": [
                {"control": i.control, "metric": i.metric,
                 "baseline": i.baseline_value, "candidate": i.candidate_value,
                 "delta": round(i.delta, 4)}
                for i in self.improvements
            ],
            "unchanged": self.unchanged,
            "verdict": self.verdict,
        }


@dataclass
class BenchmarkReport:
    """Full benchmark report."""

    suite: str
    config: BenchmarkConfig
    results: List[ControlBenchmarkResult] = field(default_factory=list)
    grade: BenchmarkGrade = BenchmarkGrade.F
    total_duration_ms: float = 0.0
    timestamp: str = ""

    def summary(self) -> str:
        """Human-readable summary."""
        lines = box_header("Safety Benchmark Report")
        lines.append(f"  Suite: {self.suite}  |  Grade: {self.grade.value}")
        lines.append(f"  Controls tested: {len(self.results)}")
        lines.append(f"  Duration: {self.total_duration_ms:.0f}ms")
        lines.append(f"  Timestamp: {self.timestamp}")
        lines.append("")

        for r in self.results:
            status = "\u2713" if r.passed else "\u2717"
            lines.append(
                f"  {status} {r.control.value:<25} "
                f"Grade: {r.grade.value:<3} "
                f"Latency p50: {r.latency.p50_ms:>6.1f}ms  "
                f"Accuracy: {r.accuracy:.1%}  "
                f"F1: {r.f1_score:.2f}"
            )
            if r.notes:
                for note in r.notes:
                    lines.append(f"    \u21b3 {note}")

        lines.append("")
        passed = sum(1 for r in self.results if r.passed)
        lines.append(f"  Passed: {passed}/{len(self.results)}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite": self.suite,
            "grade": self.grade.value,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "timestamp": self.timestamp,
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, path: Optional[str] = None) -> str:
        """Serialise to JSON.  If *path* is given, also write to file."""
        text = json.dumps(self.to_dict(), indent=2)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    @classmethod
    def from_json(cls, text: str) -> "BenchmarkReport":
        """Deserialise a report from JSON."""
        data = json.loads(text)
        results = []
        for rd in data.get("results", []):
            lat = LatencyStats(**rd.get("latency", {}))
            results.append(ControlBenchmarkResult(
                control=ControlUnderTest(rd["control"]),
                workload=rd["workload"],
                iterations=rd["iterations"],
                latency=lat,
                true_positives=rd.get("true_positives", 0),
                true_negatives=rd.get("true_negatives", 0),
                false_positives=rd.get("false_positives", 0),
                false_negatives=rd.get("false_negatives", 0),
                throughput_ops_sec=rd.get("throughput_ops_sec", 0),
                resource_cost_score=rd.get("resource_cost_score", 0),
                grade=BenchmarkGrade(rd.get("grade", "F")),
                passed=rd.get("passed", False),
                notes=rd.get("notes", []),
            ))
        report = cls(
            suite=data.get("suite", "unknown"),
            config=BenchmarkConfig(),
            results=results,
            grade=BenchmarkGrade(data.get("grade", "F")),
            total_duration_ms=data.get("total_duration_ms", 0),
            timestamp=data.get("timestamp", ""),
        )
        return report


# ---------------------------------------------------------------------------
# Benchmark Engine
# ---------------------------------------------------------------------------

_GRADE_THRESHOLDS: Dict[BenchmarkGrade, Tuple[float, float, float]] = {
    BenchmarkGrade.A_PLUS: (0.99, 10.0, 0.98),
    BenchmarkGrade.A:      (0.95, 25.0, 0.93),
    BenchmarkGrade.B:      (0.85, 50.0, 0.80),
    BenchmarkGrade.C:      (0.70, 100.0, 0.65),
    BenchmarkGrade.D:      (0.50, 200.0, 0.45),
}

_SUITE_DEFS: Dict[BenchmarkSuite, Tuple[List[ControlUnderTest], int]] = {
    BenchmarkSuite.MINIMAL: (
        [ControlUnderTest.KILL_SWITCH, ControlUnderTest.REPLICATION_LIMIT],
        5,
    ),
    BenchmarkSuite.QUICK: (
        [ControlUnderTest.KILL_SWITCH, ControlUnderTest.QUARANTINE,
         ControlUnderTest.REPLICATION_LIMIT, ControlUnderTest.DEPTH_LIMIT],
        8,
    ),
    BenchmarkSuite.STANDARD: (
        list(ControlUnderTest),
        15,
    ),
    BenchmarkSuite.STRESS: (
        list(ControlUnderTest),
        50,
    ),
}


# _percentile imported from ._helpers as _percentile_sorted


def _compute_latency(latencies_ms: List[float]) -> LatencyStats:
    """Compute latency stats from raw measurements."""
    if not latencies_ms:
        return LatencyStats()
    s = sorted(latencies_ms)
    return LatencyStats(
        p50_ms=_percentile_sorted(s, 50),
        p90_ms=_percentile_sorted(s, 90),
        p99_ms=_percentile_sorted(s, 99),
        min_ms=s[0],
        max_ms=s[-1],
        mean_ms=stats_mean(latencies_ms),
        std_ms=stats_std(latencies_ms) if len(latencies_ms) >= 2 else 0.0,
    )


def _assign_grade(result: ControlBenchmarkResult) -> BenchmarkGrade:
    """Assign a letter grade based on accuracy, latency, and F1."""
    for grade in [BenchmarkGrade.A_PLUS, BenchmarkGrade.A,
                  BenchmarkGrade.B, BenchmarkGrade.C, BenchmarkGrade.D]:
        min_acc, max_lat, min_f1 = _GRADE_THRESHOLDS[grade]
        if (result.accuracy >= min_acc
                and result.latency.p50_ms <= max_lat
                and result.f1_score >= min_f1):
            return grade
    return BenchmarkGrade.F


def _overall_grade(results: List[ControlBenchmarkResult]) -> BenchmarkGrade:
    """Compute overall grade as the worst individual grade."""
    if not results:
        return BenchmarkGrade.F
    grade_order = [BenchmarkGrade.A_PLUS, BenchmarkGrade.A,
                   BenchmarkGrade.B, BenchmarkGrade.C,
                   BenchmarkGrade.D, BenchmarkGrade.F]
    worst = 0
    for r in results:
        idx = grade_order.index(r.grade)
        worst = max(worst, idx)
    return grade_order[worst]


# ---------------------------------------------------------------------------
# Simulated workload runners
# ---------------------------------------------------------------------------


def _seeded_rng(seed: Optional[int], control: str) -> _random.Random:
    """Create a seeded Random instance for reproducible benchmarks."""
    if seed is not None:
        h = hashlib.sha256(f"{seed}:{control}".encode()).digest()
        return _random.Random(int.from_bytes(h[:8], "big"))
    return _random.Random()


def _run_workload(
    control: ControlUnderTest,
    config: BenchmarkConfig,
    base_latency_ms: float,
    stress_multiplier: float,
    detect_prob_normal: float,
    detect_prob_stress: float,
    positive_ratio: float,
    cost_divisor: float,
) -> ControlBenchmarkResult:
    """Generic workload runner parameterised per control type."""
    rng = _seeded_rng(config.seed, control.value)
    latencies: List[float] = []
    tp = tn = fp = fn = 0

    for _ in range(config.iterations):
        should_trigger = rng.random() < positive_ratio

        start = time.perf_counter()
        ms = base_latency_ms + rng.gauss(base_latency_ms * 0.6, base_latency_ms * 0.2)
        if config.intensity == WorkloadIntensity.HIGH:
            ms *= stress_multiplier * 0.6
        elif config.intensity == WorkloadIntensity.EXTREME:
            ms *= stress_multiplier
        time.sleep(max(0, ms / 1000.0))
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

        prob = (detect_prob_normal
                if config.intensity.value in ("low", "medium")
                else detect_prob_stress)
        detected = rng.random() < prob

        if should_trigger and detected:
            tp += 1
        elif should_trigger and not detected:
            fn += 1
        elif not should_trigger and not detected:
            tn += 1
        else:
            fp += 1

    total_time = sum(latencies)
    result = ControlBenchmarkResult(
        control=control,
        workload=f"{control.value}_{config.intensity.value}",
        iterations=config.iterations,
        latency=_compute_latency(latencies),
        true_positives=tp, true_negatives=tn,
        false_positives=fp, false_negatives=fn,
        throughput_ops_sec=(config.iterations / (total_time / 1000)
                           if total_time > 0 else 0),
        resource_cost_score=min(1.0, total_time / max(1, config.iterations * cost_divisor)),
    )
    result.grade = _assign_grade(result)
    result.passed = result.grade.value <= "C"
    return result


# Per-control parameters: (base_ms, stress_mult, detect_normal, detect_stress, pos_ratio, cost_div)
_CONTROL_PARAMS: Dict[ControlUnderTest, Tuple[float, float, float, float, float, float]] = {
    ControlUnderTest.KILL_SWITCH:          (1.5, 4.0, 0.98, 0.92, 0.60, 50),
    ControlUnderTest.QUARANTINE:           (3.5, 3.5, 0.95, 0.88, 0.50, 80),
    ControlUnderTest.DRIFT_DETECTION:      (5.0, 2.5, 0.93, 0.85, 0.45, 60),
    ControlUnderTest.REPLICATION_LIMIT:    (0.8, 2.0, 0.995, 0.99, 0.40, 20),
    ControlUnderTest.DEPTH_LIMIT:          (0.5, 2.0, 0.997, 0.99, 0.35, 15),
    ControlUnderTest.COMPLIANCE_AUDIT:     (8.0, 3.0, 0.93, 0.85, 0.35, 100),
    ControlUnderTest.CANARY_DETECTION:     (2.5, 2.5, 0.96, 0.90, 0.40, 40),
    ControlUnderTest.ESCALATION_DETECTION: (4.0, 2.8, 0.91, 0.82, 0.30, 70),
}

_CONTROL_RUNNERS: Dict[ControlUnderTest, Callable[[BenchmarkConfig], ControlBenchmarkResult]] = {
    ctrl: (lambda cfg, c=ctrl, p=params: _run_workload(c, cfg, *p))
    for ctrl, params in _CONTROL_PARAMS.items()
}


# ---------------------------------------------------------------------------
# Main Benchmark Class
# ---------------------------------------------------------------------------


class SafetyBenchmark:
    """Runs standardised safety control benchmarks."""

    def __init__(self, seed: Optional[int] = None):
        self._default_seed = seed

    def run_one(
        self,
        control: ControlUnderTest,
        config: Optional[BenchmarkConfig] = None,
    ) -> ControlBenchmarkResult:
        """Benchmark a single safety control."""
        if config is None:
            config = BenchmarkConfig(seed=self._default_seed)
        elif config.seed is None and self._default_seed is not None:
            config = BenchmarkConfig(
                suite=config.suite,
                iterations=config.iterations,
                warmup_iterations=config.warmup_iterations,
                controls=config.controls,
                intensity=config.intensity,
                agent_count=config.agent_count,
                seed=self._default_seed,
            )

        runner = _CONTROL_RUNNERS.get(control)
        if runner is None:
            raise ValueError(f"No benchmark runner for {control.value}")

        # Warmup
        warmup_config = BenchmarkConfig(
            iterations=config.warmup_iterations,
            intensity=config.intensity,
            agent_count=config.agent_count,
            seed=config.seed,
        )
        runner(warmup_config)

        return runner(config)

    def run_suite(
        self,
        suite: BenchmarkSuite = BenchmarkSuite.STANDARD,
        config: Optional[BenchmarkConfig] = None,
    ) -> BenchmarkReport:
        """Run a predefined benchmark suite."""
        controls, default_iters = _SUITE_DEFS[suite]

        if config is None:
            config = BenchmarkConfig(
                suite=suite,
                iterations=default_iters,
                seed=self._default_seed,
            )

        if config.controls:
            controls = config.controls

        start = time.perf_counter()
        results: List[ControlBenchmarkResult] = []

        for control in controls:
            result = self.run_one(control, config)
            results.append(result)

        total_ms = (time.perf_counter() - start) * 1000

        return BenchmarkReport(
            suite=suite.value,
            config=config,
            results=results,
            grade=_overall_grade(results),
            total_duration_ms=total_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def compare(
        self,
        baseline: BenchmarkReport,
        candidate: BenchmarkReport,
        regression_threshold: float = 0.05,
    ) -> ComparisonReport:
        """Compare two benchmark reports and find regressions/improvements."""
        base_map = {r.control.value: r for r in baseline.results}
        cand_map = {r.control.value: r for r in candidate.results}

        regressions: List[RegressionItem] = []
        improvements: List[RegressionItem] = []
        unchanged: List[str] = []

        all_controls = set(base_map.keys()) | set(cand_map.keys())

        for ctrl in sorted(all_controls):
            base_r = base_map.get(ctrl)
            cand_r = cand_map.get(ctrl)

            if base_r is None or cand_r is None:
                unchanged.append(ctrl)
                continue

            metrics = [
                ("accuracy", base_r.accuracy, cand_r.accuracy, True),
                ("f1_score", base_r.f1_score, cand_r.f1_score, True),
                ("latency_p50", base_r.latency.p50_ms, cand_r.latency.p50_ms, False),
                ("false_positive_rate", base_r.false_positive_rate,
                 cand_r.false_positive_rate, False),
            ]

            ctrl_changed = False
            for metric_name, base_val, cand_val, higher_is_better in metrics:
                if base_val == 0 and cand_val == 0:
                    continue

                if higher_is_better:
                    delta = cand_val - base_val
                    regressed = delta < -regression_threshold
                    improved = delta > regression_threshold
                else:
                    delta = cand_val - base_val
                    regressed = delta > regression_threshold
                    improved = delta < -regression_threshold

                if regressed:
                    severity = ("critical" if abs(delta) > 0.2
                                else "major" if abs(delta) > 0.1
                                else "minor")
                    regressions.append(RegressionItem(
                        control=ctrl, metric=metric_name,
                        baseline_value=round(base_val, 4),
                        candidate_value=round(cand_val, 4),
                        delta=round(delta, 4), severity=severity,
                    ))
                    ctrl_changed = True
                elif improved:
                    improvements.append(RegressionItem(
                        control=ctrl, metric=metric_name,
                        baseline_value=round(base_val, 4),
                        candidate_value=round(cand_val, 4),
                        delta=round(delta, 4), severity="improvement",
                    ))
                    ctrl_changed = True

            if not ctrl_changed:
                unchanged.append(ctrl)

        if regressions:
            verdict = "regressed"
        elif improvements:
            verdict = "improved"
        else:
            verdict = "stable"

        return ComparisonReport(
            baseline_grade=baseline.grade,
            candidate_grade=candidate.grade,
            regressions=regressions,
            improvements=improvements,
            unchanged=unchanged,
            verdict=verdict,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list] = None) -> None:
    """CLI entry point for safety benchmarks."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m replication safety-benchmark",
        description="Run standardised safety control benchmarks",
    )
    parser.add_argument(
        "--suite", choices=[s.value for s in BenchmarkSuite],
        default="standard", help="Benchmark suite to run",
    )
    parser.add_argument(
        "--control", choices=[c.value for c in ControlUnderTest],
        help="Benchmark a single control only",
    )
    parser.add_argument(
        "--iterations", type=int, default=None,
        help="Number of iterations per control",
    )
    parser.add_argument(
        "--intensity", choices=[i.value for i in WorkloadIntensity],
        default="medium", help="Workload intensity",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--compare", nargs=2, metavar="JSON",
        help="Compare two JSON benchmark files",
    )
    parser.add_argument(
        "--format", choices=["text", "json"], default="text",
        help="Output format",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Write report to file",
    )

    args = parser.parse_args(argv)

    if args.compare:
        with open(args.compare[0], encoding="utf-8") as f:
            baseline = BenchmarkReport.from_json(f.read())
        with open(args.compare[1], encoding="utf-8") as f:
            candidate = BenchmarkReport.from_json(f.read())
        bench = SafetyBenchmark()
        diff = bench.compare(baseline, candidate)
        if args.format == "json":
            output = json.dumps(diff.to_dict(), indent=2)
        else:
            lines = [f"Comparison: {diff.baseline_grade.value} -> {diff.candidate_grade.value}"]
            lines.append(f"Verdict: {diff.verdict}")
            if diff.regressions:
                lines.append(f"\nRegressions ({len(diff.regressions)}):")
                for r in diff.regressions:
                    lines.append(f"  [{r.severity}] {r.control}.{r.metric}: "
                                 f"{r.baseline_value} -> {r.candidate_value} ({r.delta:+.4f})")
            if diff.improvements:
                lines.append(f"\nImprovements ({len(diff.improvements)}):")
                for i in diff.improvements:
                    lines.append(f"  {i.control}.{i.metric}: "
                                 f"{i.baseline_value} -> {i.candidate_value} ({i.delta:+.4f})")
            output = "\n".join(lines)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
        else:
            print(output)
        return

    suite = BenchmarkSuite(args.suite)
    intensity = WorkloadIntensity(args.intensity)
    iters = args.iterations or _SUITE_DEFS[suite][1]

    config = BenchmarkConfig(
        suite=suite,
        iterations=iters,
        intensity=intensity,
        seed=args.seed,
    )

    if args.control:
        control = ControlUnderTest(args.control)
        config.controls = [control]

    bench = SafetyBenchmark(seed=args.seed)
    report = bench.run_suite(suite, config)

    if args.format == "json":
        output = report.to_json(args.output)
        if not args.output:
            print(output)
    else:
        text = report.summary()
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(text)
        else:
            print(text)


if __name__ == "__main__":
    main()
