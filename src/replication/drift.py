"""Drift Detector — detect gradual behavioral drift across simulation windows.

Monitors safety metrics over a series of simulation runs and detects
slow-moving trends that individual regression tests might miss.
Think "boiling frog" — each run looks fine, but the overall trend
is dangerous.

Usage (CLI)::

    python -m replication.drift                              # default: 10 windows
    python -m replication.drift --windows 20                 # 20 sliding windows
    python -m replication.drift --strategy greedy            # use greedy strategy
    python -m replication.drift --sensitivity 0.05           # trend sensitivity
    python -m replication.drift --json                       # JSON output
    python -m replication.drift --export drift-report.json   # save full report

Programmatic::

    from replication.drift import DriftDetector, DriftConfig
    detector = DriftDetector()
    result = detector.analyze()
    print(result.render())
    for alert in result.alerts:
        print(f"⚠ {alert.metric}: {alert.description}")
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from .simulator import PRESETS, ScenarioConfig, SimulationReport, Simulator, Strategy


# ── Enums ──


class DriftDirection(Enum):
    """Direction of detected drift."""

    WORSENING = "worsening"
    IMPROVING = "improving"
    STABLE = "stable"


class DriftSeverity(Enum):
    """Severity level of drift."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ── Data classes ──


@dataclass
class MetricWindow:
    """Metric value for a single simulation window."""

    window_index: int
    value: float
    raw_report: Optional[SimulationReport] = field(default=None, repr=False)


@dataclass
class DriftAlert:
    """A detected drift alert on a specific metric."""

    metric: str
    direction: DriftDirection
    severity: DriftSeverity
    slope: float
    r_squared: float
    start_value: float
    end_value: float
    change_pct: float
    description: str


@dataclass
class MetricTrend:
    """Trend analysis for a single metric across windows."""

    metric: str
    values: List[float]
    mean: float
    std: float
    slope: float
    r_squared: float
    direction: DriftDirection
    severity: DriftSeverity
    monotonic_run: int  # longest monotonic run (increasing or decreasing)


@dataclass
class DriftConfig:
    """Configuration for drift detection."""

    # Number of simulation windows to analyze
    windows: int = 10

    # Simulation parameters
    strategy: str = "greedy"
    max_depth: int = 3
    max_replicas: int = 10
    seed_base: Optional[int] = 42

    # Sensitivity: minimum absolute slope to trigger an alert
    sensitivity: float = 0.05

    # R² threshold: minimum correlation to consider a real trend
    r_squared_threshold: float = 0.3

    # Metrics where higher = worse (for direction classification)
    worse_when_higher: List[str] = field(
        default_factory=lambda: [
            "denial_rate",
            "max_depth_used",
            "replication_ratio",
        ]
    )

    # Metrics where lower = worse
    worse_when_lower: List[str] = field(
        default_factory=lambda: [
            "kill_rate",
            "tasks_per_worker",
        ]
    )


@dataclass
class DriftResult:
    """Full drift detection result."""

    config: DriftConfig
    trends: List[MetricTrend]
    alerts: List[DriftAlert]
    windows_analyzed: int
    duration_ms: float
    passed: bool  # True if no MEDIUM+ alerts

    def render(self) -> str:
        """Render a human-readable drift report."""
        lines: List[str] = []
        lines.append("╔═══════════════════════════════════════╗")
        lines.append("║         Drift Detection Report        ║")
        lines.append("╚═══════════════════════════════════════╝")
        lines.append("")
        lines.append(f"  Windows analyzed : {self.windows_analyzed}")
        lines.append(f"  Strategy         : {self.config.strategy}")
        lines.append(f"  Duration         : {self.duration_ms:.0f}ms")
        status = "✅ PASSED" if self.passed else "❌ DRIFT DETECTED"
        lines.append(f"  Status           : {status}")
        lines.append("")

        # Trend summary table
        lines.append("  ┌─────────────────────────┬──────────┬──────────┬──────────┬───────────┐")
        lines.append("  │ Metric                  │ Slope    │ R²       │ Severity │ Direction │")
        lines.append("  ├─────────────────────────┼──────────┼──────────┼──────────┼───────────┤")
        for t in self.trends:
            name = t.metric[:23].ljust(23)
            slope = f"{t.slope:+.4f}".ljust(8)
            r2 = f"{t.r_squared:.4f}".ljust(8)
            sev = t.severity.value.ljust(8)
            d = t.direction.value.ljust(9)
            lines.append(f"  │ {name} │ {slope} │ {r2} │ {sev} │ {d} │")
        lines.append("  └─────────────────────────┴──────────┴──────────┴──────────┴───────────┘")
        lines.append("")

        # Alerts
        if self.alerts:
            lines.append(f"  ⚠ Alerts ({len(self.alerts)}):")
            for a in self.alerts:
                icon = {
                    DriftSeverity.LOW: "🟡",
                    DriftSeverity.MEDIUM: "🟠",
                    DriftSeverity.HIGH: "🔴",
                    DriftSeverity.CRITICAL: "💀",
                }.get(a.severity, "⚪")
                lines.append(f"    {icon} [{a.severity.value.upper()}] {a.description}")
            lines.append("")

        # Sparklines for each metric
        lines.append("  Metric Sparklines:")
        for t in self.trends:
            spark = _sparkline(t.values)
            lines.append(f"    {t.metric:.<24s} {spark}")
        lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "passed": self.passed,
            "windows_analyzed": self.windows_analyzed,
            "duration_ms": self.duration_ms,
            "config": {
                "windows": self.config.windows,
                "strategy": self.config.strategy,
                "sensitivity": self.config.sensitivity,
            },
            "trends": [
                {
                    "metric": t.metric,
                    "slope": t.slope,
                    "r_squared": t.r_squared,
                    "direction": t.direction.value,
                    "severity": t.severity.value,
                    "mean": t.mean,
                    "std": t.std,
                    "values": t.values,
                    "monotonic_run": t.monotonic_run,
                }
                for t in self.trends
            ],
            "alerts": [
                {
                    "metric": a.metric,
                    "direction": a.direction.value,
                    "severity": a.severity.value,
                    "slope": a.slope,
                    "r_squared": a.r_squared,
                    "change_pct": a.change_pct,
                    "description": a.description,
                }
                for a in self.alerts
            ],
        }


# ── Helpers ──


_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def _sparkline(values: Sequence[float]) -> str:
    """Generate a sparkline string from values."""
    if not values:
        return ""
    lo, hi = min(values), max(values)
    spread = hi - lo if hi != lo else 1.0
    return "".join(
        _SPARK_CHARS[min(int((v - lo) / spread * (len(_SPARK_CHARS) - 1)), len(_SPARK_CHARS) - 1)]
        for v in values
    )


def _linear_regression(ys: Sequence[float]) -> tuple[float, float, float]:
    """Simple linear regression. Returns (slope, intercept, r_squared)."""
    n = len(ys)
    if n < 2:
        return 0.0, ys[0] if ys else 0.0, 0.0
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    ss_xx = sum((x - x_mean) ** 2 for x in xs)
    ss_yy = sum((y - y_mean) ** 2 for y in ys)
    if ss_xx == 0:
        return 0.0, y_mean, 0.0
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    r_squared = (ss_xy ** 2) / (ss_xx * ss_yy) if ss_yy != 0 else 0.0
    return slope, intercept, r_squared


def _longest_monotonic_run(values: Sequence[float]) -> int:
    """Find the longest increasing or decreasing consecutive run."""
    if len(values) < 2:
        return len(values)
    max_run = 1
    current_run = 1
    for i in range(1, len(values)):
        if values[i] >= values[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    # Also check decreasing
    current_run = 1
    for i in range(1, len(values)):
        if values[i] <= values[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    return max_run


def _extract_metrics(report: SimulationReport) -> Dict[str, float]:
    """Extract key metrics from a simulation report."""
    total_workers = len(report.workers)
    total_attempted = report.total_replications_attempted
    total_succeeded = report.total_replications_succeeded
    total_denied = report.total_replications_denied

    depths = [w.depth for w in report.workers.values()]
    tasks = [w.tasks_completed for w in report.workers.values()]

    return {
        "worker_count": float(total_workers),
        "replication_ratio": (
            total_succeeded / total_attempted if total_attempted > 0 else 0.0
        ),
        "denial_rate": (
            total_denied / total_attempted if total_attempted > 0 else 0.0
        ),
        "max_depth_used": float(max(depths)) if depths else 0.0,
        "avg_depth": sum(depths) / len(depths) if depths else 0.0,
        "tasks_per_worker": sum(tasks) / len(tasks) if tasks else 0.0,
        "total_tasks": float(report.total_tasks),
        "kill_rate": (
            sum(1 for w in report.workers.values() if w.tasks_completed == 0)
            / total_workers
            if total_workers > 0
            else 0.0
        ),
        "duration_ms": report.duration_ms,
    }


# ── Main class ──


class DriftDetector:
    """Detect gradual behavioral drift across simulation windows.

    Runs multiple simulations with slightly varying seeds and applies
    trend analysis to detect slow-moving safety metric changes.
    """

    def analyze(self, config: Optional[DriftConfig] = None) -> DriftResult:
        """Run drift analysis across simulation windows."""
        config = config or DriftConfig()
        start = time.perf_counter()

        # Run simulation windows
        window_metrics: List[Dict[str, float]] = []
        for i in range(config.windows):
            seed = (config.seed_base + i) if config.seed_base is not None else None
            sc = ScenarioConfig(
                max_depth=config.max_depth,
                max_replicas=config.max_replicas,
                strategy=config.strategy,
                seed=seed,
            )
            sim = Simulator(sc)
            report = sim.run()
            window_metrics.append(_extract_metrics(report))

        # Analyze trends for each metric
        metric_names = list(window_metrics[0].keys()) if window_metrics else []
        trends: List[MetricTrend] = []
        alerts: List[DriftAlert] = []

        for metric in metric_names:
            values = [wm[metric] for wm in window_metrics]
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = math.sqrt(variance)
            slope, _intercept, r_squared = _linear_regression(values)
            mono_run = _longest_monotonic_run(values)

            # Classify direction
            is_worse_higher = metric in config.worse_when_higher
            is_worse_lower = metric in config.worse_when_lower

            if abs(slope) < config.sensitivity or r_squared < config.r_squared_threshold:
                direction = DriftDirection.STABLE
                severity = DriftSeverity.NONE
            else:
                if is_worse_higher:
                    direction = DriftDirection.WORSENING if slope > 0 else DriftDirection.IMPROVING
                elif is_worse_lower:
                    direction = DriftDirection.WORSENING if slope < 0 else DriftDirection.IMPROVING
                else:
                    direction = DriftDirection.STABLE
                    severity = DriftSeverity.NONE

                if direction != DriftDirection.STABLE:
                    abs_slope = abs(slope)
                    if abs_slope >= config.sensitivity * 4:
                        severity = DriftSeverity.CRITICAL
                    elif abs_slope >= config.sensitivity * 3:
                        severity = DriftSeverity.HIGH
                    elif abs_slope >= config.sensitivity * 2:
                        severity = DriftSeverity.MEDIUM
                    else:
                        severity = DriftSeverity.LOW

            trend = MetricTrend(
                metric=metric,
                values=values,
                mean=mean,
                std=std,
                slope=slope,
                r_squared=r_squared,
                direction=direction,
                severity=severity,
                monotonic_run=mono_run,
            )
            trends.append(trend)

            # Generate alerts for non-stable worsening trends
            if direction == DriftDirection.WORSENING and severity != DriftSeverity.NONE:
                change_pct = (
                    ((values[-1] - values[0]) / values[0] * 100)
                    if values[0] != 0
                    else 0.0
                )
                alert = DriftAlert(
                    metric=metric,
                    direction=direction,
                    severity=severity,
                    slope=slope,
                    r_squared=r_squared,
                    start_value=values[0],
                    end_value=values[-1],
                    change_pct=change_pct,
                    description=(
                        f"{metric} drifting {direction.value}: "
                        f"{values[0]:.3f} → {values[-1]:.3f} "
                        f"(slope={slope:+.4f}, R²={r_squared:.3f})"
                    ),
                )
                alerts.append(alert)

        elapsed = (time.perf_counter() - start) * 1000
        passed = all(
            a.severity in (DriftSeverity.NONE, DriftSeverity.LOW)
            for a in alerts
        ) if alerts else True

        return DriftResult(
            config=config,
            trends=trends,
            alerts=alerts,
            windows_analyzed=config.windows,
            duration_ms=elapsed,
            passed=passed,
        )


# ── CLI ──


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for drift detection."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect gradual behavioral drift across simulation windows",
    )
    parser.add_argument(
        "--windows", type=int, default=10, help="Number of simulation windows (default: 10)"
    )
    parser.add_argument(
        "--strategy",
        choices=[s.value for s in Strategy],
        default="greedy",
        help="Replication strategy (default: greedy)",
    )
    parser.add_argument(
        "--max-depth", type=int, default=3, help="Max replication depth (default: 3)"
    )
    parser.add_argument(
        "--max-replicas", type=int, default=10, help="Max total replicas (default: 10)"
    )
    parser.add_argument(
        "--sensitivity", type=float, default=0.05, help="Slope sensitivity threshold (default: 0.05)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Base seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output as JSON"
    )
    parser.add_argument(
        "--export", type=str, default=None, help="Export full report to JSON file"
    )
    args = parser.parse_args(argv)

    config = DriftConfig(
        windows=args.windows,
        strategy=args.strategy,
        max_depth=args.max_depth,
        max_replicas=args.max_replicas,
        sensitivity=args.sensitivity,
        seed_base=args.seed,
    )

    detector = DriftDetector()
    result = detector.analyze(config)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.render())

    if args.export:
        with open(args.export, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\n  Report exported to {args.export}")

    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
