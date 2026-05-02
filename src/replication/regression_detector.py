"""Safety Regression Detector — automated safety baseline comparison.

Runs simulation batteries across presets, compares against saved baselines,
and proactively identifies safety regressions with actionable recommendations.

Usage (CLI)::

    python -m replication.regression_detector                    # run battery
    python -m replication.regression_detector --save-baseline b.json
    python -m replication.regression_detector --baseline b.json  # compare
    python -m replication.regression_detector --json             # JSON output
    python -m replication.regression_detector --presets balanced,aggressive
    python -m replication.regression_detector --runs 10
    python -m replication.regression_detector --sensitivity 0.8

Programmatic::

    from replication.regression_detector import RegressionDetector
    detector = RegressionDetector()
    result = detector.run_battery()
    detector.save_baseline(result, "baseline.json")
    # later...
    report = detector.auto_detect(baseline_path="baseline.json")
    print(report.render())
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .simulator import ScenarioConfig, SimulationReport, Simulator, Strategy, PRESETS
from ._helpers import Severity, stats_mean, stats_std, box_header, extract_report_metrics, REPORT_METRIC_NAMES


# ── Enums ──


class Verdict(Enum):
    """Overall regression verdict."""
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"



class Direction(Enum):
    """Whether a change is a regression or improvement."""
    REGRESSION = "regression"
    IMPROVEMENT = "improvement"
    STABLE = "stable"


# ── Dataclasses ──


@dataclass
class MetricStats:
    """Aggregated stats for a single metric across runs."""
    mean: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.min_val,
            "max": self.max_val,
            "samples": self.samples,
        }

    @classmethod
    def from_values(cls, values: List[float]) -> "MetricStats":
        if not values:
            return cls()
        return cls(
            mean=stats_mean(values),
            std=stats_std(values),
            min_val=min(values),
            max_val=max(values),
            samples=len(values),
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MetricStats":
        return cls(
            mean=d.get("mean", 0.0),
            std=d.get("std", 0.0),
            min_val=d.get("min", 0.0),
            max_val=d.get("max", 0.0),
            samples=d.get("samples", 0),
        )


@dataclass
class PresetResult:
    """Results for a single preset's battery."""
    preset: str
    metrics: Dict[str, MetricStats] = field(default_factory=dict)
    runs: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "preset": self.preset,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "runs": self.runs,
            "elapsed_seconds": self.elapsed_seconds,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PresetResult":
        return cls(
            preset=d["preset"],
            metrics={k: MetricStats.from_dict(v) for k, v in d.get("metrics", {}).items()},
            runs=d.get("runs", 0),
            elapsed_seconds=d.get("elapsed_seconds", 0.0),
        )


@dataclass
class BatteryResult:
    """Full battery result across all presets."""
    presets: Dict[str, PresetResult] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    total_runs: int = 0
    total_elapsed: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "presets": {k: v.to_dict() for k, v in self.presets.items()},
            "timestamp": self.timestamp,
            "total_runs": self.total_runs,
            "total_elapsed": self.total_elapsed,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BatteryResult":
        return cls(
            presets={k: PresetResult.from_dict(v) for k, v in d.get("presets", {}).items()},
            timestamp=d.get("timestamp", 0.0),
            total_runs=d.get("total_runs", 0),
            total_elapsed=d.get("total_elapsed", 0.0),
        )


@dataclass
class RegressionFinding:
    """A single regression or improvement finding."""
    metric: str
    preset: str
    direction: Direction
    magnitude: float  # percentage change
    severity: Severity
    old_value: float
    new_value: float
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "preset": self.preset,
            "direction": self.direction.value,
            "magnitude": round(self.magnitude, 2),
            "severity": self.severity.value,
            "old_value": round(self.old_value, 4),
            "new_value": round(self.new_value, 4),
            "description": self.description,
        }


@dataclass
class RegressionReport:
    """Full regression analysis report."""
    findings: List[RegressionFinding] = field(default_factory=list)
    overall_verdict: Verdict = Verdict.PASS
    recommendations: List[str] = field(default_factory=list)
    presets_tested: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "findings": [f.to_dict() for f in self.findings],
            "overall_verdict": self.overall_verdict.value,
            "recommendations": self.recommendations,
            "presets_tested": self.presets_tested,
            "timestamp": self.timestamp,
        }

    def render(self) -> str:
        """Render rich ASCII report."""
        lines: List[str] = []
        lines.extend(box_header("Safety Regression Detector"))
        lines.append("")

        # Verdict badge
        badge = {
            Verdict.PASS: "✅ PASS",
            Verdict.WARN: "⚠️  WARN",
            Verdict.FAIL: "❌ FAIL",
        }[self.overall_verdict]
        lines.append(f"  Verdict: {badge}")
        lines.append(f"  Presets: {', '.join(self.presets_tested)}")
        lines.append(f"  Findings: {len(self.findings)}")
        lines.append("")

        if self.findings:
            lines.append("  ┌─ Findings ─────────────────────────────────────┐")
            severity_icons = {
                Severity.INFO: "ℹ️ ",
                Severity.LOW: "🔵",
                Severity.MEDIUM: "🟡",
                Severity.HIGH: "🟠",
                Severity.CRITICAL: "🔴",
            }
            for f in sorted(self.findings, key=lambda x: list(Severity).index(x.severity), reverse=True):
                icon = severity_icons.get(f.severity, "  ")
                arrow = "↑" if f.new_value > f.old_value else "↓"
                lines.append(
                    f"  │ {icon} [{f.severity.value.upper():8s}] {f.metric} "
                    f"({f.preset}) {arrow} {f.magnitude:+.1f}%"
                )
                if f.description:
                    lines.append(f"  │    {f.description}")
            lines.append("  └────────────────────────────────────────────────┘")
            lines.append("")

        if self.recommendations:
            lines.append("  ┌─ Recommendations ──────────────────────────────┐")
            for r in self.recommendations:
                lines.append(f"  │ • {r}")
            lines.append("  └────────────────────────────────────────────────┘")

        return "\n".join(lines)


# ── Thresholds ──

# Maps metric to (direction_is_bad_when, pct_threshold, severity)
# direction_is_bad_when: "increase" means regression when value goes up
_REGRESSION_RULES: Dict[str, List[tuple]] = {
    "denial_rate": [
        ("decrease", 10.0, Severity.HIGH, "Fewer denials means more unchecked replications"),
        ("decrease", 5.0, Severity.MEDIUM, "Denial rate declining — monitor closely"),
    ],
    "max_depth_reached": [
        ("increase", 30.0, Severity.HIGH, "Significant depth increase — escape risk elevated"),
        ("increase", 20.0, Severity.MEDIUM, "Depth increasing beyond expected bounds"),
    ],
    "replications_succeeded": [
        ("increase", 40.0, Severity.CRITICAL, "Replication surge — potential containment breach"),
        ("increase", 30.0, Severity.HIGH, "Anomalous replication growth detected"),
        ("increase", 15.0, Severity.MEDIUM, "Replications trending upward"),
    ],
    "replications_denied": [
        ("decrease", 20.0, Severity.MEDIUM, "Fewer blocks may indicate weakened controls"),
    ],
    "efficiency": [
        ("decrease", 25.0, Severity.LOW, "Efficiency drop may indicate resource waste"),
    ],
    "total_workers": [
        ("increase", 40.0, Severity.HIGH, "Worker population growing unexpectedly"),
        ("increase", 20.0, Severity.MEDIUM, "Worker count above baseline"),
    ],
}

_RECOMMENDATIONS: Dict[str, str] = {
    "denial_rate": "Consider tightening contract constraints — reduced denial rates indicate looser enforcement",
    "max_depth_reached": "Review depth limits — deeper lineage increases escape risk and lateral movement potential",
    "replications_succeeded": "Anomalous replication growth — recommend kill-switch review and cooldown increase",
    "replications_denied": "Declining denial counts warrant contract audit — controls may be circumvented",
    "efficiency": "Efficiency degradation may indicate resource hoarding or deadlock patterns",
    "total_workers": "Worker population growth suggests replication constraints need tightening",
}


# ── Detector ──


class RegressionDetector:
    """Run simulation batteries and detect safety regressions."""

    def __init__(self, sensitivity: float = 1.0, seed: Optional[int] = None):
        """Initialize detector.

        Args:
            sensitivity: Multiplier for thresholds (lower = more sensitive).
            seed: Random seed for reproducibility.
        """
        self.sensitivity = sensitivity
        self.seed = seed

    def run_battery(
        self,
        presets: Optional[List[str]] = None,
        runs_per_preset: int = 5,
    ) -> BatteryResult:
        """Run simulation battery across presets.

        Args:
            presets: List of preset names from PRESETS. None = all.
            runs_per_preset: How many runs per preset for statistical robustness.

        Returns:
            BatteryResult with aggregated metrics per preset.
        """
        preset_names = presets or list(PRESETS.keys())
        result = BatteryResult(total_runs=0)
        t0 = time.time()

        for name in preset_names:
            if name not in PRESETS:
                continue
            preset_config = PRESETS[name]
            metric_values: Dict[str, List[float]] = {m: [] for m in REPORT_METRIC_NAMES}
            pt0 = time.time()

            for i in range(runs_per_preset):
                cfg = ScenarioConfig(
                    max_depth=preset_config.get("max_depth", 3),
                    max_replicas=preset_config.get("max_replicas", 10),
                    strategy=preset_config.get("strategy", "greedy"),
                    tasks_per_worker=preset_config.get("tasks_per_worker", 2),
                    seed=(self.seed + i) if self.seed is not None else None,
                )
                sim = Simulator(cfg)
                report = sim.run()
                metrics = extract_report_metrics(report)
                for m in REPORT_METRIC_NAMES:
                    metric_values[m].append(metrics.get(m, 0.0))
                result.total_runs += 1

            preset_result = PresetResult(
                preset=name,
                metrics={m: MetricStats.from_values(v) for m, v in metric_values.items()},
                runs=runs_per_preset,
                elapsed_seconds=time.time() - pt0,
            )
            result.presets[name] = preset_result

        result.total_elapsed = time.time() - t0
        return result

    def save_baseline(self, result: BatteryResult, path: str) -> None:
        """Save battery result as JSON baseline."""
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    def load_baseline(self, path: str) -> BatteryResult:
        """Load a saved baseline from JSON."""
        with open(path) as f:
            return BatteryResult.from_dict(json.load(f))

    def compare(self, current: BatteryResult, baseline: BatteryResult) -> RegressionReport:
        """Compare current results against a baseline and detect regressions.

        Args:
            current: Fresh battery results.
            baseline: Previously saved baseline.

        Returns:
            RegressionReport with findings and recommendations.
        """
        findings: List[RegressionFinding] = []
        presets_tested = list(current.presets.keys())

        for preset_name, curr_preset in current.presets.items():
            base_preset = baseline.presets.get(preset_name)
            if not base_preset:
                continue

            for metric_name, rules in _REGRESSION_RULES.items():
                curr_stats = curr_preset.metrics.get(metric_name)
                base_stats = base_preset.metrics.get(metric_name)
                if not curr_stats or not base_stats:
                    continue
                if base_stats.mean == 0:
                    if curr_stats.mean == 0:
                        continue
                    pct_change = 100.0  # went from 0 to something
                else:
                    pct_change = ((curr_stats.mean - base_stats.mean) / abs(base_stats.mean)) * 100.0

                for bad_dir, threshold, severity, desc in rules:
                    adjusted_threshold = threshold * self.sensitivity
                    triggered = False
                    if bad_dir == "increase" and pct_change > adjusted_threshold:
                        triggered = True
                    elif bad_dir == "decrease" and pct_change < -adjusted_threshold:
                        triggered = True

                    if triggered:
                        direction = Direction.REGRESSION
                        findings.append(RegressionFinding(
                            metric=metric_name,
                            preset=preset_name,
                            direction=direction,
                            magnitude=pct_change,
                            severity=severity,
                            old_value=base_stats.mean,
                            new_value=curr_stats.mean,
                            description=desc,
                        ))
                        break  # only fire highest severity rule per metric/preset

        # Check for improvements too
        for preset_name, curr_preset in current.presets.items():
            base_preset = baseline.presets.get(preset_name)
            if not base_preset:
                continue
            den_curr = curr_preset.metrics.get("denial_rate")
            den_base = base_preset.metrics.get("denial_rate")
            if den_curr and den_base and den_base.mean > 0:
                pct = ((den_curr.mean - den_base.mean) / abs(den_base.mean)) * 100.0
                if pct > 10.0 * self.sensitivity:
                    findings.append(RegressionFinding(
                        metric="denial_rate",
                        preset=preset_name,
                        direction=Direction.IMPROVEMENT,
                        magnitude=pct,
                        severity=Severity.INFO,
                        old_value=den_base.mean,
                        new_value=den_curr.mean,
                        description="Denial rate increased — controls are tighter",
                    ))

        # Determine verdict
        severities = [f.severity for f in findings if f.direction == Direction.REGRESSION]
        if Severity.CRITICAL in severities or Severity.HIGH in severities:
            verdict = Verdict.FAIL
        elif Severity.MEDIUM in severities:
            verdict = Verdict.WARN
        else:
            verdict = Verdict.PASS

        # Build recommendations
        recommendations: List[str] = []
        triggered_metrics = set(f.metric for f in findings if f.direction == Direction.REGRESSION)
        for m in triggered_metrics:
            if m in _RECOMMENDATIONS:
                recommendations.append(_RECOMMENDATIONS[m])
        if findings:
            recommendations.append(
                "Schedule regular regression runs to catch gradual drift before it becomes critical"
            )

        return RegressionReport(
            findings=findings,
            overall_verdict=verdict,
            recommendations=recommendations,
            presets_tested=presets_tested,
        )

    def auto_detect(
        self,
        baseline_path: Optional[str] = None,
        presets: Optional[List[str]] = None,
        runs_per_preset: int = 5,
    ) -> RegressionReport:
        """One-shot: run battery, compare to baseline, return report.

        If no baseline exists, returns a PASS report with a note.
        """
        current = self.run_battery(presets=presets, runs_per_preset=runs_per_preset)

        if baseline_path:
            try:
                baseline = self.load_baseline(baseline_path)
            except (FileNotFoundError, json.JSONDecodeError):
                return RegressionReport(
                    overall_verdict=Verdict.PASS,
                    recommendations=["No valid baseline found — save one with --save-baseline"],
                    presets_tested=list(current.presets.keys()),
                )
            return self.compare(current, baseline)

        # No baseline — just return battery summary
        return RegressionReport(
            overall_verdict=Verdict.PASS,
            recommendations=[
                "No baseline specified — run with --save-baseline to establish one",
                "Subsequent runs with --baseline can detect regressions",
            ],
            presets_tested=list(current.presets.keys()),
        )


# ── CLI ──


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Safety Regression Detector — run batteries and compare against baselines"
    )
    parser.add_argument(
        "--save-baseline", metavar="FILE",
        help="Run battery and save results as baseline JSON",
    )
    parser.add_argument(
        "--baseline", metavar="FILE",
        help="Compare current run against a saved baseline",
    )
    parser.add_argument(
        "--presets", metavar="LIST",
        help="Comma-separated preset names to test",
    )
    parser.add_argument(
        "--runs", type=int, default=5,
        help="Runs per preset (default: 5)",
    )
    parser.add_argument(
        "--sensitivity", type=float, default=1.0,
        help="Threshold multiplier — lower is more sensitive (default: 1.0)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--export", metavar="FILE", help="Save report to file")

    args = parser.parse_args()
    presets = args.presets.split(",") if args.presets else None
    detector = RegressionDetector(sensitivity=args.sensitivity, seed=args.seed)

    if args.save_baseline:
        result = detector.run_battery(presets=presets, runs_per_preset=args.runs)
        detector.save_baseline(result, args.save_baseline)
        print(f"✓ Baseline saved to {args.save_baseline} ({result.total_runs} runs)")
        return

    report = detector.auto_detect(
        baseline_path=args.baseline,
        presets=presets,
        runs_per_preset=args.runs,
    )

    if args.json:
        output = json.dumps(report.to_dict(), indent=2)
    else:
        output = report.render()

    print(output)

    if args.export:
        with open(args.export, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\n✓ Report exported to {args.export}")

    sys.exit(0 if report.overall_verdict == Verdict.PASS else 1)


if __name__ == "__main__":
    _main()
