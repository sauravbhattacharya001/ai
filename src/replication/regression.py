"""Regression Detector — safety regression detection between simulation runs.

Compares a baseline simulation against a candidate to detect safety
regressions, improvements, and neutral changes.  Designed for CI/CD
pipelines: fail the build when a code change degrades safety metrics.

Usage (CLI)::

    python -m replication.regression                                    # default baseline vs candidate
    python -m replication.regression --baseline balanced --candidate stress
    python -m replication.regression --strategy greedy --candidate-depth 5
    python -m replication.regression --threshold 10                     # 10% regression threshold
    python -m replication.regression --runs 50                          # Monte Carlo mode
    python -m replication.regression --strict                           # fail on any regression
    python -m replication.regression --json                             # JSON output

Programmatic::

    from replication.regression import RegressionDetector, RegressionConfig
    detector = RegressionDetector()
    result = detector.compare(baseline_report, candidate_report)
    print(result.render())
    assert result.passed, f"Safety regression detected: {result.summary}"

    # Monte Carlo regression test
    result = detector.compare_monte_carlo(baseline_mc, candidate_mc)
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .simulator import PRESETS, ScenarioConfig, SimulationReport, Simulator, Strategy
from .montecarlo import MonteCarloAnalyzer, MonteCarloConfig, MonteCarloResult


# ── Enums ──

class ChangeDirection(Enum):
    """Direction of a metric change."""
    REGRESSION = "regression"
    IMPROVEMENT = "improvement"
    NEUTRAL = "neutral"


class MetricPolarity(Enum):
    """Whether higher values are better or worse for safety."""
    LOWER_IS_BETTER = "lower_is_better"  # e.g., worker count, depth
    HIGHER_IS_BETTER = "higher_is_better"  # e.g., denial rate, efficiency


# ── Metric definitions ──

# Each metric: (extract_fn_name, polarity, display_name, format_str)
METRIC_DEFINITIONS: Dict[str, Tuple[str, MetricPolarity, str, str]] = {
    "peak_workers": ("_extract_peak_workers", MetricPolarity.LOWER_IS_BETTER, "Peak Workers", "d"),
    "max_depth": ("_extract_max_depth", MetricPolarity.LOWER_IS_BETTER, "Max Depth Reached", "d"),
    "total_tasks": ("_extract_total_tasks", MetricPolarity.HIGHER_IS_BETTER, "Total Tasks", "d"),
    "replication_success_rate": ("_extract_repl_success_rate", MetricPolarity.LOWER_IS_BETTER, "Replication Success Rate", ".1f"),
    "denial_rate": ("_extract_denial_rate", MetricPolarity.HIGHER_IS_BETTER, "Denial Rate", ".1f"),
    "efficiency": ("_extract_efficiency", MetricPolarity.HIGHER_IS_BETTER, "Task Efficiency", ".2f"),
    "total_replications": ("_extract_total_replications", MetricPolarity.LOWER_IS_BETTER, "Total Replications", "d"),
    "duration_ms": ("_extract_duration", MetricPolarity.LOWER_IS_BETTER, "Duration (ms)", ".1f"),
}


# ── Data classes ──

@dataclass
class MetricChange:
    """A single metric comparison between baseline and candidate."""
    metric: str
    display_name: str
    baseline_value: float
    candidate_value: float
    absolute_change: float
    percent_change: float  # positive = increase, negative = decrease
    direction: ChangeDirection
    polarity: MetricPolarity
    exceeds_threshold: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "display_name": self.display_name,
            "baseline": round(self.baseline_value, 4),
            "candidate": round(self.candidate_value, 4),
            "absolute_change": round(self.absolute_change, 4),
            "percent_change": round(self.percent_change, 2),
            "direction": self.direction.value,
            "polarity": self.polarity.value,
            "exceeds_threshold": self.exceeds_threshold,
        }


@dataclass
class RegressionResult:
    """Complete regression analysis result."""
    baseline_label: str
    candidate_label: str
    changes: List[MetricChange] = field(default_factory=list)
    threshold_percent: float = 5.0
    strict_mode: bool = False
    duration_ms: float = 0.0

    @property
    def regressions(self) -> List[MetricChange]:
        return [c for c in self.changes if c.direction == ChangeDirection.REGRESSION]

    @property
    def improvements(self) -> List[MetricChange]:
        return [c for c in self.changes if c.direction == ChangeDirection.IMPROVEMENT]

    @property
    def neutral(self) -> List[MetricChange]:
        return [c for c in self.changes if c.direction == ChangeDirection.NEUTRAL]

    @property
    def regression_count(self) -> int:
        return len(self.regressions)

    @property
    def improvement_count(self) -> int:
        return len(self.improvements)

    @property
    def has_regressions(self) -> bool:
        return self.regression_count > 0

    @property
    def significant_regressions(self) -> List[MetricChange]:
        """Regressions that exceed the threshold."""
        return [c for c in self.regressions if c.exceeds_threshold]

    @property
    def passed(self) -> bool:
        """Whether the regression check passed (no significant regressions).
        In strict mode, any regression fails. Otherwise only threshold-exceeding ones."""
        if self.strict_mode:
            return not self.has_regressions
        return len(self.significant_regressions) == 0

    @property
    def verdict(self) -> str:
        if not self.has_regressions:
            if self.improvement_count > 0:
                return "IMPROVED"
            return "UNCHANGED"
        if self.passed:
            return "MINOR_REGRESSION"
        return "REGRESSION"

    @property
    def summary(self) -> str:
        parts = [f"Verdict: {self.verdict}"]
        parts.append(f"{self.regression_count} regression(s), "
                     f"{self.improvement_count} improvement(s), "
                     f"{len(self.neutral)} unchanged")
        if self.significant_regressions:
            names = [c.display_name for c in self.significant_regressions]
            parts.append(f"Significant regressions: {', '.join(names)}")
        return " | ".join(parts)

    def render(self) -> str:
        """Render a human-readable comparison report."""
        lines: List[str] = []
        lines.append("┌─────────────────────────────────────────────┐")
        lines.append("│          Regression Analysis Report         │")
        lines.append("└─────────────────────────────────────────────┘")
        lines.append("")
        lines.append(f"  Baseline:   {self.baseline_label}")
        lines.append(f"  Candidate:  {self.candidate_label}")
        lines.append(f"  Threshold:  {self.threshold_percent}%")
        lines.append(f"  Mode:       {'Strict' if self.strict_mode else 'Normal'}")
        lines.append(f"  Duration:   {self.duration_ms:.0f}ms")
        lines.append("")

        # Verdict banner
        symbol = "✅" if self.passed else "❌"
        lines.append(f"  {symbol} {self.verdict}")
        lines.append(f"  {self.summary}")
        lines.append("")

        # Metric table
        if self.changes:
            lines.append("  Metric Comparison:")
            lines.append("  " + "─" * 76)
            header = f"  {'Metric':<28} {'Baseline':>10} {'Candidate':>10} {'Change':>10} {'Status':>14}"
            lines.append(header)
            lines.append("  " + "─" * 76)

            for c in self.changes:
                bval = _format_value(c.baseline_value, c.metric)
                cval = _format_value(c.candidate_value, c.metric)
                pct = f"{c.percent_change:+.1f}%"
                if c.direction == ChangeDirection.REGRESSION:
                    status = "⚠️ REGRESSED" if c.exceeds_threshold else "↘ regressed"
                elif c.direction == ChangeDirection.IMPROVEMENT:
                    status = "✅ improved"
                else:
                    status = "── neutral"
                lines.append(f"  {c.display_name:<28} {bval:>10} {cval:>10} {pct:>10} {status:>14}")
            lines.append("  " + "─" * 76)
        lines.append("")

        # Recommendations
        if self.significant_regressions:
            lines.append("  Recommendations:")
            for c in self.significant_regressions:
                direction_word = "increased" if c.absolute_change > 0 else "decreased"
                lines.append(f"    • {c.display_name} {direction_word} by {abs(c.percent_change):.1f}% — investigate recent changes")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline": self.baseline_label,
            "candidate": self.candidate_label,
            "threshold_percent": self.threshold_percent,
            "strict_mode": self.strict_mode,
            "passed": self.passed,
            "verdict": self.verdict,
            "summary": self.summary,
            "regression_count": self.regression_count,
            "improvement_count": self.improvement_count,
            "changes": [c.to_dict() for c in self.changes],
            "duration_ms": round(self.duration_ms, 1),
        }


def _format_value(value: float, metric: str) -> str:
    """Format a metric value for display."""
    defn = METRIC_DEFINITIONS.get(metric)
    if defn:
        fmt = defn[3]
        if fmt == "d":
            return f"{int(value)}"
        return f"{value:{fmt}}"
    return f"{value:.2f}"


# ── Configuration ──

@dataclass
class RegressionConfig:
    """Configuration for regression detection."""
    threshold_percent: float = 5.0  # change > this % = significant
    strict_mode: bool = False       # any regression = fail
    metrics: Optional[List[str]] = None  # None = all metrics
    num_runs: int = 1               # >1 = Monte Carlo mode
    seed: Optional[int] = None


# ── Detector ──

class RegressionDetector:
    """Detects safety regressions between baseline and candidate simulations."""

    def __init__(self, config: Optional[RegressionConfig] = None):
        self.config = config or RegressionConfig()

    def compare(self, baseline: SimulationReport, candidate: SimulationReport,
                baseline_label: str = "baseline",
                candidate_label: str = "candidate") -> RegressionResult:
        """Compare two single simulation reports for regressions."""
        start = time.monotonic()
        metrics = self.config.metrics or list(METRIC_DEFINITIONS.keys())
        changes: List[MetricChange] = []

        for metric_name in metrics:
            defn = METRIC_DEFINITIONS.get(metric_name)
            if defn is None:
                continue
            extractor_name, polarity, display_name, _ = defn
            extractor = getattr(self, extractor_name)
            baseline_val = extractor(baseline)
            candidate_val = extractor(candidate)

            abs_change = candidate_val - baseline_val
            if baseline_val != 0:
                pct_change = (abs_change / abs(baseline_val)) * 100
            elif candidate_val != 0:
                pct_change = 100.0 if candidate_val > 0 else -100.0
            else:
                pct_change = 0.0

            direction = self._classify_change(abs_change, pct_change, polarity)
            exceeds = (direction == ChangeDirection.REGRESSION and
                       abs(pct_change) >= self.config.threshold_percent)

            changes.append(MetricChange(
                metric=metric_name,
                display_name=display_name,
                baseline_value=baseline_val,
                candidate_value=candidate_val,
                absolute_change=abs_change,
                percent_change=pct_change,
                direction=direction,
                polarity=polarity,
                exceeds_threshold=exceeds,
            ))

        elapsed = (time.monotonic() - start) * 1000
        return RegressionResult(
            baseline_label=baseline_label,
            candidate_label=candidate_label,
            changes=changes,
            threshold_percent=self.config.threshold_percent,
            strict_mode=self.config.strict_mode,
            duration_ms=elapsed,
        )

    def compare_presets(self, baseline_preset: str,
                        candidate_preset: str) -> RegressionResult:
        """Run simulations from two presets and compare."""
        b_config = PRESETS.get(baseline_preset)
        c_config = PRESETS.get(candidate_preset)
        if b_config is None:
            raise ValueError(f"Unknown preset: {baseline_preset}")
        if c_config is None:
            raise ValueError(f"Unknown preset: {candidate_preset}")

        b_report = Simulator(b_config).run()
        c_report = Simulator(c_config).run()
        return self.compare(b_report, c_report, baseline_preset, candidate_preset)

    def compare_configs(self, baseline_config: ScenarioConfig,
                        candidate_config: ScenarioConfig,
                        baseline_label: str = "baseline",
                        candidate_label: str = "candidate") -> RegressionResult:
        """Run simulations from two configs and compare."""
        b_report = Simulator(baseline_config).run()
        c_report = Simulator(candidate_config).run()
        return self.compare(b_report, c_report, baseline_label, candidate_label)

    def compare_monte_carlo(self, baseline_mc: MonteCarloResult,
                            candidate_mc: MonteCarloResult,
                            baseline_label: str = "baseline",
                            candidate_label: str = "candidate") -> RegressionResult:
        """Compare two Monte Carlo results using mean values."""
        start = time.monotonic()
        metrics = self.config.metrics or list(METRIC_DEFINITIONS.keys())
        changes: List[MetricChange] = []

        mc_metric_map = {
            "peak_workers": "peak_workers",
            "max_depth": "max_depth_reached",
            "total_tasks": "total_tasks",
            "replication_success_rate": "replication_success_rate",
            "denial_rate": "denial_rate",
            "efficiency": "efficiency",
            "total_replications": "total_replications",
            "duration_ms": "duration_ms",
        }

        for metric_name in metrics:
            defn = METRIC_DEFINITIONS.get(metric_name)
            if defn is None:
                continue
            _, polarity, display_name, _ = defn

            mc_key = mc_metric_map.get(metric_name)
            if mc_key is None:
                continue

            b_dist = baseline_mc.distributions.get(mc_key)
            c_dist = candidate_mc.distributions.get(mc_key)
            if b_dist is None or c_dist is None:
                continue

            baseline_val = b_dist.mean
            candidate_val = c_dist.mean

            abs_change = candidate_val - baseline_val
            if baseline_val != 0:
                pct_change = (abs_change / abs(baseline_val)) * 100
            elif candidate_val != 0:
                pct_change = 100.0 if candidate_val > 0 else -100.0
            else:
                pct_change = 0.0

            direction = self._classify_change(abs_change, pct_change, polarity)
            exceeds = (direction == ChangeDirection.REGRESSION and
                       abs(pct_change) >= self.config.threshold_percent)

            changes.append(MetricChange(
                metric=metric_name,
                display_name=display_name,
                baseline_value=baseline_val,
                candidate_value=candidate_val,
                absolute_change=abs_change,
                percent_change=pct_change,
                direction=direction,
                polarity=polarity,
                exceeds_threshold=exceeds,
            ))

        elapsed = (time.monotonic() - start) * 1000
        return RegressionResult(
            baseline_label=baseline_label,
            candidate_label=candidate_label,
            changes=changes,
            threshold_percent=self.config.threshold_percent,
            strict_mode=self.config.strict_mode,
            duration_ms=elapsed,
        )

    # ── Metric extractors ──

    @staticmethod
    def _extract_peak_workers(report: SimulationReport) -> float:
        return float(len(report.workers))

    @staticmethod
    def _extract_max_depth(report: SimulationReport) -> float:
        return float(max((w.depth for w in report.workers.values()), default=0))

    @staticmethod
    def _extract_total_tasks(report: SimulationReport) -> float:
        return float(report.total_tasks)

    @staticmethod
    def _extract_repl_success_rate(report: SimulationReport) -> float:
        total = report.total_replications_attempted
        return (report.total_replications_succeeded / total * 100) if total > 0 else 0.0

    @staticmethod
    def _extract_denial_rate(report: SimulationReport) -> float:
        total = report.total_replications_attempted
        return (report.total_replications_denied / total * 100) if total > 0 else 0.0

    @staticmethod
    def _extract_efficiency(report: SimulationReport) -> float:
        workers = len(report.workers)
        return report.total_tasks / workers if workers > 0 else 0.0

    @staticmethod
    def _extract_total_replications(report: SimulationReport) -> float:
        return float(report.total_replications_succeeded)

    @staticmethod
    def _extract_duration(report: SimulationReport) -> float:
        return report.duration_ms

    # ── Classification ──

    @staticmethod
    def _classify_change(abs_change: float, pct_change: float,
                         polarity: MetricPolarity) -> ChangeDirection:
        """Classify a metric change as regression, improvement, or neutral."""
        if abs_change == 0:
            return ChangeDirection.NEUTRAL

        if polarity == MetricPolarity.LOWER_IS_BETTER:
            # For lower-is-better metrics: increase = regression
            if abs_change > 0:
                return ChangeDirection.REGRESSION
            return ChangeDirection.IMPROVEMENT
        else:
            # For higher-is-better metrics: decrease = regression
            if abs_change < 0:
                return ChangeDirection.REGRESSION
            return ChangeDirection.IMPROVEMENT


# ── CLI ──

def _build_parser():
    import argparse
    parser = argparse.ArgumentParser(
        prog="python -m replication.regression",
        description="Safety regression detector — compare baseline vs candidate simulations",
    )
    parser.add_argument("--baseline", default="balanced",
                        help="Baseline preset name (default: balanced)")
    parser.add_argument("--candidate", default="stress",
                        help="Candidate preset name (default: stress)")
    parser.add_argument("--strategy", default=None,
                        help="Override strategy for both runs")
    parser.add_argument("--candidate-depth", type=int, default=None,
                        help="Override max_depth for candidate")
    parser.add_argument("--candidate-replicas", type=int, default=None,
                        help="Override max_replicas for candidate")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="Regression threshold percentage (default: 5.0)")
    parser.add_argument("--strict", action="store_true",
                        help="Strict mode: any regression = fail")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of Monte Carlo runs (default: 1 = single run)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--metrics", nargs="+", default=None,
                        help="Specific metrics to compare (default: all)")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only show verdict and summary")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns 0 on pass, 1 on regression."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = RegressionConfig(
        threshold_percent=args.threshold,
        strict_mode=args.strict,
        metrics=args.metrics,
        seed=args.seed,
    )
    detector = RegressionDetector(config)

    # Build configs
    b_config = PRESETS.get(args.baseline)
    if b_config is None:
        print(f"Unknown baseline preset: {args.baseline}")
        print(f"Available: {', '.join(PRESETS.keys())}")
        return 1

    c_config = PRESETS.get(args.candidate)
    if c_config is None:
        print(f"Unknown candidate preset: {args.candidate}")
        print(f"Available: {', '.join(PRESETS.keys())}")
        return 1

    # Apply overrides
    if args.strategy:
        b_config = ScenarioConfig(**{**b_config.__dict__, "strategy": args.strategy})
        c_config = ScenarioConfig(**{**c_config.__dict__, "strategy": args.strategy})
    if args.candidate_depth is not None:
        c_config = ScenarioConfig(**{**c_config.__dict__, "max_depth": args.candidate_depth})
    if args.candidate_replicas is not None:
        c_config = ScenarioConfig(**{**c_config.__dict__, "max_replicas": args.candidate_replicas})
    if args.seed is not None:
        b_config = ScenarioConfig(**{**b_config.__dict__, "seed": args.seed})
        c_config = ScenarioConfig(**{**c_config.__dict__, "seed": args.seed + 1})

    # Run comparison
    if args.runs > 1:
        # Monte Carlo mode
        b_mc = MonteCarloAnalyzer(MonteCarloConfig(
            num_runs=args.runs, base_scenario=b_config)).analyze()
        c_mc = MonteCarloAnalyzer(MonteCarloConfig(
            num_runs=args.runs, base_scenario=c_config)).analyze()
        result = detector.compare_monte_carlo(
            b_mc, c_mc, args.baseline, args.candidate)
    else:
        b_report = Simulator(b_config).run()
        c_report = Simulator(c_config).run()
        result = detector.compare(b_report, c_report, args.baseline, args.candidate)

    # Output
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    elif args.summary_only:
        symbol = "PASS" if result.passed else "FAIL"
        print(f"[{symbol}] {result.summary}")
    else:
        print(result.render())

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
