"""Monte Carlo risk analyzer for probabilistic safety assessment.

Runs many randomized simulations to produce statistical distributions and
confidence intervals for key safety metrics.  Answers questions like
"What is the 95th-percentile peak worker count under this policy?" and
"What is the probability of reaching max depth?"

Usage (CLI)::

    python -m replication.montecarlo                                  # 200 runs, default config
    python -m replication.montecarlo --runs 1000                      # more precision
    python -m replication.montecarlo --strategy random --probability 0.7
    python -m replication.montecarlo --max-depth 5 --max-replicas 30
    python -m replication.montecarlo --json                           # JSON output
    python -m replication.montecarlo --scenario balanced              # from preset
    python -m replication.montecarlo --compare greedy conservative    # compare risk profiles

Programmatic::

    from replication.montecarlo import MonteCarloAnalyzer, MonteCarloConfig
    analyzer = MonteCarloAnalyzer(MonteCarloConfig(num_runs=500))
    result = analyzer.analyze()
    print(result.render())

    # Compare two strategies
    comparison = analyzer.compare(["greedy", "conservative"])
    print(comparison.render())
"""

from __future__ import annotations

import math
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .simulator import PRESETS, ScenarioConfig, SimulationReport, Simulator, Strategy


# ── Statistics helpers ──────────────────────────────────────────────────

def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return (s[mid - 1] + s[mid]) / 2 if n % 2 == 0 else s[mid]


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def _percentile(values: List[float], p: float) -> float:
    """Return the p-th percentile (0–100)."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (p / 100) * (len(s) - 1)
    f = int(k)
    c = f + 1
    if c >= len(s):
        return s[-1]
    return s[f] + (k - f) * (s[c] - s[f])


def _ci95(values: List[float]) -> Tuple[float, float]:
    """95% confidence interval for the mean (normal approximation)."""
    if len(values) < 2:
        m = _mean(values)
        return (m, m)
    m = _mean(values)
    se = _std(values) / math.sqrt(len(values))
    margin = 1.96 * se
    return (m - margin, m + margin)


# ── Data models ─────────────────────────────────────────────────────────

@dataclass
class MetricDistribution:
    """Statistical distribution for a single metric across many runs."""

    name: str
    unit: str
    values: List[float]

    @property
    def mean(self) -> float:
        return _mean(self.values)

    @property
    def median(self) -> float:
        return _median(self.values)

    @property
    def std(self) -> float:
        return _std(self.values)

    @property
    def min(self) -> float:
        return min(self.values) if self.values else 0.0

    @property
    def max(self) -> float:
        return max(self.values) if self.values else 0.0

    @property
    def p5(self) -> float:
        return _percentile(self.values, 5)

    @property
    def p25(self) -> float:
        return _percentile(self.values, 25)

    @property
    def p75(self) -> float:
        return _percentile(self.values, 75)

    @property
    def p95(self) -> float:
        return _percentile(self.values, 95)

    @property
    def p99(self) -> float:
        return _percentile(self.values, 99)

    @property
    def ci95(self) -> Tuple[float, float]:
        return _ci95(self.values)

    def to_dict(self) -> Dict[str, Any]:
        lo, hi = self.ci95
        return {
            "name": self.name,
            "unit": self.unit,
            "mean": round(self.mean, 3),
            "median": round(self.median, 3),
            "std": round(self.std, 3),
            "min": round(self.min, 3),
            "max": round(self.max, 3),
            "p5": round(self.p5, 3),
            "p25": round(self.p25, 3),
            "p75": round(self.p75, 3),
            "p95": round(self.p95, 3),
            "p99": round(self.p99, 3),
            "ci95_lower": round(lo, 3),
            "ci95_upper": round(hi, 3),
            "n": len(self.values),
        }


@dataclass
class RiskMetrics:
    """Collection of computed risk indicators."""

    # Probability of hitting contract limits (0.0–1.0)
    prob_max_depth_reached: float
    prob_quota_saturated: float
    prob_all_denied: float

    # Breach counts
    max_depth_breach_pct: float  # % runs reaching max_depth
    quota_saturation_pct: float  # % runs with denied replications

    # Resource extremes
    peak_worker_p95: float
    peak_depth_p95: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prob_max_depth_reached": round(self.prob_max_depth_reached, 4),
            "prob_quota_saturated": round(self.prob_quota_saturated, 4),
            "prob_all_denied": round(self.prob_all_denied, 4),
            "max_depth_breach_pct": round(self.max_depth_breach_pct, 2),
            "quota_saturation_pct": round(self.quota_saturation_pct, 2),
            "peak_worker_p95": round(self.peak_worker_p95, 1),
            "peak_depth_p95": round(self.peak_depth_p95, 1),
        }


@dataclass
class MonteCarloResult:
    """Results of a Monte Carlo risk analysis."""

    config: "MonteCarloConfig"
    scenario: ScenarioConfig
    num_runs: int
    distributions: Dict[str, MetricDistribution]
    risk_metrics: RiskMetrics
    depth_histogram: Dict[int, int]
    worker_histogram: Dict[int, int]
    denial_rate_histogram: Dict[int, int]  # bucketed 0-100%
    duration_ms: float

    @property
    def risk_level(self) -> str:
        """Overall risk classification."""
        p = self.risk_metrics.prob_max_depth_reached
        q = self.risk_metrics.prob_quota_saturated
        combined = (p + q) / 2
        if combined < 0.1:
            return "LOW"
        elif combined < 0.3:
            return "MODERATE"
        elif combined < 0.6:
            return "ELEVATED"
        elif combined < 0.85:
            return "HIGH"
        else:
            return "CRITICAL"

    @property
    def risk_emoji(self) -> str:
        mapping = {
            "LOW": "🟢",
            "MODERATE": "🟡",
            "ELEVATED": "🟠",
            "HIGH": "🔴",
            "CRITICAL": "⛔",
        }
        return mapping.get(self.risk_level, "❓")

    def render_summary(self) -> str:
        lines: List[str] = []
        lines.append("┌───────────────────────────────────────────────────────┐")
        lines.append("│       🎲  Monte Carlo Risk Analysis  🎲              │")
        lines.append("└───────────────────────────────────────────────────────┘")
        lines.append("")
        lines.append(f"  Risk Level: {self.risk_emoji} {self.risk_level}")
        lines.append("")
        lines.append(f"  Configuration:")
        lines.append(f"    Strategy:       {self.scenario.strategy}")
        lines.append(f"    Max Depth:      {self.scenario.max_depth}")
        lines.append(f"    Max Replicas:   {self.scenario.max_replicas}")
        lines.append(f"    Cooldown:       {self.scenario.cooldown_seconds}s")
        lines.append(f"    Tasks/Worker:   {self.scenario.tasks_per_worker}")
        lines.append(f"    Simulation Runs: {self.num_runs}")
        lines.append(f"    Duration:       {self.duration_ms:.0f}ms")
        return "\n".join(lines)

    def render_distributions(self) -> str:
        lines: List[str] = []
        lines.append("")
        lines.append("┌───────────────────────────────────────────────────────┐")
        lines.append("│            Metric Distributions                       │")
        lines.append("└───────────────────────────────────────────────────────┘")
        lines.append("")

        # Table header
        hdr = (
            f"  {'Metric':<24}"
            f"{'Mean':>8}"
            f"{'Median':>8}"
            f"{'Std':>8}"
            f"{'P5':>8}"
            f"{'P95':>8}"
            f"{'Min':>8}"
            f"{'Max':>8}"
        )
        lines.append(hdr)
        lines.append("  " + "─" * 78)

        for dist in self.distributions.values():
            lines.append(
                f"  {dist.name:<24}"
                f"{dist.mean:>8.1f}"
                f"{dist.median:>8.1f}"
                f"{dist.std:>8.1f}"
                f"{dist.p5:>8.1f}"
                f"{dist.p95:>8.1f}"
                f"{dist.min:>8.1f}"
                f"{dist.max:>8.1f}"
            )
        return "\n".join(lines)

    def render_risk_indicators(self) -> str:
        lines: List[str] = []
        lines.append("")
        lines.append("┌───────────────────────────────────────────────────────┐")
        lines.append("│            Risk Indicators                            │")
        lines.append("└───────────────────────────────────────────────────────┘")
        lines.append("")

        rm = self.risk_metrics

        def _bar(pct: float) -> str:
            filled = int(pct / 100 * 20)
            return "█" * filled + "░" * (20 - filled)

        lines.append(f"  Max Depth Reached:     {rm.max_depth_breach_pct:5.1f}%  [{_bar(rm.max_depth_breach_pct)}]")
        lines.append(f"  Quota Saturated:       {rm.quota_saturation_pct:5.1f}%  [{_bar(rm.quota_saturation_pct)}]")
        lines.append(f"  All Replications Denied: {rm.prob_all_denied * 100:5.1f}%  [{_bar(rm.prob_all_denied * 100)}]")
        lines.append("")
        lines.append(f"  Peak Workers (P95):    {rm.peak_worker_p95:.0f}")
        lines.append(f"  Peak Depth (P95):      {rm.peak_depth_p95:.0f}")

        # 95% CI for key metrics
        w_dist = self.distributions.get("total_workers")
        if w_dist:
            lo, hi = w_dist.ci95
            lines.append(f"  Workers 95% CI:        [{lo:.1f}, {hi:.1f}]")

        d_dist = self.distributions.get("max_depth_reached")
        if d_dist:
            lo, hi = d_dist.ci95
            lines.append(f"  Depth 95% CI:          [{lo:.1f}, {hi:.1f}]")

        return "\n".join(lines)

    def render_histograms(self) -> str:
        lines: List[str] = []
        lines.append("")
        lines.append("┌───────────────────────────────────────────────────────┐")
        lines.append("│            Depth Distribution (across runs)           │")
        lines.append("└───────────────────────────────────────────────────────┘")
        lines.append("")

        if self.depth_histogram:
            max_count = max(self.depth_histogram.values()) if self.depth_histogram else 1
            for depth in sorted(self.depth_histogram.keys()):
                count = self.depth_histogram[depth]
                bar_len = max(1, int(count / max_count * 30))
                bar = "█" * bar_len
                pct = count / self.num_runs * 100
                lines.append(f"  depth {depth:>2}: {bar} {count:>4} ({pct:.0f}%)")

        lines.append("")
        lines.append("┌───────────────────────────────────────────────────────┐")
        lines.append("│          Worker Count Distribution (across runs)      │")
        lines.append("└───────────────────────────────────────────────────────┘")
        lines.append("")

        if self.worker_histogram:
            max_count = max(self.worker_histogram.values()) if self.worker_histogram else 1
            for workers in sorted(self.worker_histogram.keys()):
                count = self.worker_histogram[workers]
                bar_len = max(1, int(count / max_count * 30))
                bar = "█" * bar_len
                pct = count / self.num_runs * 100
                lines.append(f"  {workers:>3} workers: {bar} {count:>4} ({pct:.0f}%)")

        return "\n".join(lines)

    def render_recommendations(self) -> str:
        lines: List[str] = []
        lines.append("")
        lines.append("┌───────────────────────────────────────────────────────┐")
        lines.append("│            Recommendations                            │")
        lines.append("└───────────────────────────────────────────────────────┘")
        lines.append("")

        rm = self.risk_metrics

        if self.risk_level == "LOW":
            lines.append("  ✅ Contract configuration appears safe for this strategy.")
            lines.append("     Low probability of hitting resource limits.")
        else:
            if rm.max_depth_breach_pct > 50:
                lines.append(
                    f"  ⚠️  Depth limit reached in {rm.max_depth_breach_pct:.0f}% of runs."
                )
                lines.append(
                    f"     Consider increasing max_depth above {self.scenario.max_depth} "
                    f"or using a less aggressive strategy."
                )
            if rm.quota_saturation_pct > 50:
                lines.append(
                    f"  ⚠️  Quota saturated in {rm.quota_saturation_pct:.0f}% of runs."
                )
                lines.append(
                    f"     Consider increasing max_replicas above {self.scenario.max_replicas} "
                    f"or adding cooldown."
                )
            if rm.prob_all_denied > 0.1:
                lines.append(
                    f"  ⚠️  All replications denied in {rm.prob_all_denied * 100:.0f}% of runs."
                )
                lines.append(
                    "     Contract may be too restrictive for this strategy."
                )

            # Strategy-specific advice
            if self.scenario.strategy == "greedy":
                lines.append("  💡 Greedy strategy maximizes replication. "
                             "Tighten cooldown or reduce max_replicas for safety.")
            elif self.scenario.strategy == "random":
                p = self.scenario.replication_probability
                if p > 0.7:
                    lines.append(
                        f"  💡 Replication probability ({p}) is high. "
                        f"Try 0.3–0.5 for more predictable behavior."
                    )

            w_dist = self.distributions.get("total_workers")
            if w_dist and w_dist.std > w_dist.mean * 0.5:
                lines.append(
                    "  📊 High variance in worker counts. "
                    "Results are unpredictable — consider deterministic strategies."
                )

        return "\n".join(lines)

    def render(self) -> str:
        return "\n".join([
            self.render_summary(),
            self.render_distributions(),
            self.render_risk_indicators(),
            self.render_histograms(),
            self.render_recommendations(),
        ])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": {
                "strategy": self.scenario.strategy,
                "max_depth": self.scenario.max_depth,
                "max_replicas": self.scenario.max_replicas,
                "cooldown_seconds": self.scenario.cooldown_seconds,
                "tasks_per_worker": self.scenario.tasks_per_worker,
                "replication_probability": self.scenario.replication_probability,
            },
            "num_runs": self.num_runs,
            "risk_level": self.risk_level,
            "distributions": {
                k: v.to_dict() for k, v in self.distributions.items()
            },
            "risk_metrics": self.risk_metrics.to_dict(),
            "depth_histogram": {str(k): v for k, v in self.depth_histogram.items()},
            "worker_histogram": {str(k): v for k, v in self.worker_histogram.items()},
            "duration_ms": round(self.duration_ms, 1),
        }


@dataclass
class MonteCarloComparison:
    """Side-by-side risk comparison of multiple strategies."""

    labels: List[str]
    results: List[MonteCarloResult]
    duration_ms: float

    def render_table(self) -> str:
        lines: List[str] = []
        lines.append("┌───────────────────────────────────────────────────────────────┐")
        lines.append("│       🎲  Monte Carlo Risk Comparison  🎲                    │")
        lines.append("└───────────────────────────────────────────────────────────────┘")
        lines.append("")

        # Header
        label_width = max(max(len(l) for l in self.labels), 10)
        hdr = (
            f"  {'Strategy':<{label_width}}"
            f"  {'Risk':>8}"
            f"  {'Workers':>10}"
            f"  {'W(P95)':>8}"
            f"  {'Depth':>8}"
            f"  {'D(P95)':>8}"
            f"  {'Denied%':>9}"
            f"  {'DepMax%':>9}"
        )
        lines.append(hdr)
        lines.append("  " + "─" * (label_width + 68))

        for label, result in zip(self.labels, self.results):
            rm = result.risk_metrics
            w = result.distributions["total_workers"]
            d = result.distributions["max_depth_reached"]
            dn = result.distributions.get("denial_rate")
            dn_mean = f"{dn.mean:.1f}" if dn else "N/A"
            lines.append(
                f"  {label:<{label_width}}"
                f"  {result.risk_emoji + ' ' + result.risk_level:>8}"
                f"  {w.mean:>10.1f}"
                f"  {rm.peak_worker_p95:>8.0f}"
                f"  {d.mean:>8.1f}"
                f"  {rm.peak_depth_p95:>8.0f}"
                f"  {dn_mean:>9}"
                f"  {rm.max_depth_breach_pct:>8.1f}%"
            )

        lines.append("  " + "─" * (label_width + 68))
        lines.append(f"  Total analysis time: {self.duration_ms:.0f}ms")
        return "\n".join(lines)

    def render_risk_ranking(self) -> str:
        lines: List[str] = []
        lines.append("")
        lines.append("┌───────────────────────────────────────────────────────────────┐")
        lines.append("│               Risk Ranking (safest first)                     │")
        lines.append("└───────────────────────────────────────────────────────────────┘")
        lines.append("")

        # Rank by combined risk (lower = safer)
        risk_order = {
            "LOW": 1, "MODERATE": 2, "ELEVATED": 3, "HIGH": 4, "CRITICAL": 5,
        }
        ranked = sorted(
            zip(self.labels, self.results),
            key=lambda x: (
                risk_order.get(x[1].risk_level, 5),
                x[1].risk_metrics.prob_max_depth_reached + x[1].risk_metrics.prob_quota_saturated,
            ),
        )

        medals = {0: "🥇", 1: "🥈", 2: "🥉"}
        for i, (label, result) in enumerate(ranked):
            medal = medals.get(i, f"#{i + 1}")
            lines.append(
                f"  {medal} {label:<16} {result.risk_emoji} {result.risk_level}"
                f"  (depth breach: {result.risk_metrics.max_depth_breach_pct:.0f}%, "
                f"quota sat: {result.risk_metrics.quota_saturation_pct:.0f}%)"
            )

        return "\n".join(lines)

    def render_insights(self) -> str:
        lines: List[str] = []
        lines.append("")
        lines.append("┌───────────────────────────────────────────────────────────────┐")
        lines.append("│               Key Insights                                    │")
        lines.append("└───────────────────────────────────────────────────────────────┘")
        lines.append("")

        # Find extremes
        safest = min(
            zip(self.labels, self.results),
            key=lambda x: x[1].risk_metrics.prob_max_depth_reached + x[1].risk_metrics.prob_quota_saturated,
        )
        riskiest = max(
            zip(self.labels, self.results),
            key=lambda x: x[1].risk_metrics.prob_max_depth_reached + x[1].risk_metrics.prob_quota_saturated,
        )

        lines.append(f"  🛡️  Safest strategy: {safest[0]} ({safest[1].risk_level})")
        lines.append(f"  ⚡ Riskiest strategy: {riskiest[0]} ({riskiest[1].risk_level})")

        # Variance comparison
        most_predictable = min(
            zip(self.labels, self.results),
            key=lambda x: x[1].distributions["total_workers"].std,
        )
        least_predictable = max(
            zip(self.labels, self.results),
            key=lambda x: x[1].distributions["total_workers"].std,
        )
        lines.append(f"  📏 Most predictable: {most_predictable[0]} "
                      f"(σ={most_predictable[1].distributions['total_workers'].std:.1f} workers)")
        lines.append(f"  🎰 Least predictable: {least_predictable[0]} "
                      f"(σ={least_predictable[1].distributions['total_workers'].std:.1f} workers)")

        # Efficiency
        most_efficient = max(
            zip(self.labels, self.results),
            key=lambda x: x[1].distributions["efficiency"].mean,
        )
        lines.append(f"  🎯 Most efficient: {most_efficient[0]} "
                      f"({most_efficient[1].distributions['efficiency'].mean:.2f} tasks/worker)")

        return "\n".join(lines)

    def render(self) -> str:
        return "\n".join([
            self.render_table(),
            self.render_risk_ranking(),
            self.render_insights(),
        ])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategies": {
                label: result.to_dict()
                for label, result in zip(self.labels, self.results)
            },
            "duration_ms": round(self.duration_ms, 1),
        }


# ── Configuration ───────────────────────────────────────────────────────

@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo analysis."""

    num_runs: int = 200
    base_scenario: Optional[ScenarioConfig] = None
    # When True, each run gets a unique random seed.
    # When False and base_scenario has a seed, all runs use that seed.
    randomize_seeds: bool = True


# ── Analyzer ────────────────────────────────────────────────────────────

class MonteCarloAnalyzer:
    """Run many randomized simulations and compute risk statistics."""

    def __init__(self, config: Optional[MonteCarloConfig] = None) -> None:
        self.config = config or MonteCarloConfig()
        self._rng = random.Random()

    def _make_scenario(self, **overrides: Any) -> ScenarioConfig:
        base = self.config.base_scenario or ScenarioConfig()
        return ScenarioConfig(
            max_depth=overrides.get("max_depth", base.max_depth),
            max_replicas=overrides.get("max_replicas", base.max_replicas),
            cooldown_seconds=overrides.get("cooldown_seconds", base.cooldown_seconds),
            expiration_seconds=overrides.get("expiration_seconds", base.expiration_seconds),
            strategy=overrides.get("strategy", base.strategy),
            tasks_per_worker=overrides.get("tasks_per_worker", base.tasks_per_worker),
            replication_probability=overrides.get("replication_probability", base.replication_probability),
            secret=overrides.get("secret", base.secret),
            seed=overrides.get("seed", base.seed),
            cpu_limit=overrides.get("cpu_limit", base.cpu_limit),
            memory_limit_mb=overrides.get("memory_limit_mb", base.memory_limit_mb),
        )

    def _run_batch(self, scenario: ScenarioConfig) -> List[SimulationReport]:
        """Run num_runs simulations and return all reports."""
        reports: List[SimulationReport] = []
        for i in range(self.config.num_runs):
            cfg = ScenarioConfig(
                max_depth=scenario.max_depth,
                max_replicas=scenario.max_replicas,
                cooldown_seconds=scenario.cooldown_seconds,
                expiration_seconds=scenario.expiration_seconds,
                strategy=scenario.strategy,
                tasks_per_worker=scenario.tasks_per_worker,
                replication_probability=scenario.replication_probability,
                secret=scenario.secret,
                seed=self._rng.randint(0, 2**31) if self.config.randomize_seeds else scenario.seed,
                cpu_limit=scenario.cpu_limit,
                memory_limit_mb=scenario.memory_limit_mb,
            )
            sim = Simulator(cfg)
            reports.append(sim.run())
        return reports

    def _compute_result(
        self, scenario: ScenarioConfig, reports: List[SimulationReport], duration_ms: float,
    ) -> MonteCarloResult:
        """Compute statistics from a batch of simulation reports."""

        # Extract raw metric vectors
        total_workers_v = [float(len(r.workers)) for r in reports]
        total_tasks_v = [float(r.total_tasks) for r in reports]
        repl_ok_v = [float(r.total_replications_succeeded) for r in reports]
        repl_denied_v = [float(r.total_replications_denied) for r in reports]
        max_depth_v = [
            float(max((w.depth for w in r.workers.values()), default=0))
            for r in reports
        ]
        efficiency_v = [
            r.total_tasks / len(r.workers) if r.workers else 0.0
            for r in reports
        ]
        denial_rate_v = [
            (r.total_replications_denied / r.total_replications_attempted * 100)
            if r.total_replications_attempted > 0 else 0.0
            for r in reports
        ]
        duration_v = [r.duration_ms for r in reports]

        distributions = {
            "total_workers": MetricDistribution("Total Workers", "count", total_workers_v),
            "total_tasks": MetricDistribution("Total Tasks", "count", total_tasks_v),
            "replications_ok": MetricDistribution("Replications OK", "count", repl_ok_v),
            "replications_denied": MetricDistribution("Replications Denied", "count", repl_denied_v),
            "max_depth_reached": MetricDistribution("Max Depth Reached", "depth", max_depth_v),
            "efficiency": MetricDistribution("Efficiency", "tasks/worker", efficiency_v),
            "denial_rate": MetricDistribution("Denial Rate", "%", denial_rate_v),
            "run_duration": MetricDistribution("Run Duration", "ms", duration_v),
        }

        # Risk metrics
        n = len(reports)
        depth_breach_count = sum(1 for d in max_depth_v if d >= scenario.max_depth)
        quota_sat_count = sum(1 for r in reports if r.total_replications_denied > 0)
        all_denied_count = sum(
            1 for r in reports
            if r.total_replications_attempted > 0 and r.total_replications_succeeded == 0
        )

        risk_metrics = RiskMetrics(
            prob_max_depth_reached=depth_breach_count / n if n else 0,
            prob_quota_saturated=quota_sat_count / n if n else 0,
            prob_all_denied=all_denied_count / n if n else 0,
            max_depth_breach_pct=depth_breach_count / n * 100 if n else 0,
            quota_saturation_pct=quota_sat_count / n * 100 if n else 0,
            peak_worker_p95=_percentile(total_workers_v, 95),
            peak_depth_p95=_percentile(max_depth_v, 95),
        )

        # Histograms
        depth_histogram: Dict[int, int] = Counter(int(d) for d in max_depth_v)
        worker_histogram: Dict[int, int] = Counter(int(w) for w in total_workers_v)

        # Bucket denial rates into 10% bins
        denial_buckets: Dict[int, int] = Counter()
        for dr in denial_rate_v:
            bucket = min(int(dr // 10) * 10, 90)
            denial_buckets[bucket] += 1

        return MonteCarloResult(
            config=self.config,
            scenario=scenario,
            num_runs=n,
            distributions=distributions,
            risk_metrics=risk_metrics,
            depth_histogram=dict(depth_histogram),
            worker_histogram=dict(worker_histogram),
            denial_rate_histogram=dict(denial_buckets),
            duration_ms=duration_ms,
        )

    def analyze(self, **overrides: Any) -> MonteCarloResult:
        """Run Monte Carlo analysis with the configured scenario."""
        scenario = self._make_scenario(**overrides)
        start = time.monotonic()
        reports = self._run_batch(scenario)
        duration_ms = (time.monotonic() - start) * 1000
        return self._compute_result(scenario, reports, duration_ms)

    def compare(
        self,
        strategies: Optional[Sequence[str]] = None,
    ) -> MonteCarloComparison:
        """Compare risk profiles across multiple strategies."""
        if strategies is None:
            strategies = [s.value for s in Strategy]

        start = time.monotonic()
        labels: List[str] = []
        results: List[MonteCarloResult] = []

        for strat in strategies:
            result = self.analyze(strategy=strat)
            labels.append(strat)
            results.append(result)

        duration_ms = (time.monotonic() - start) * 1000
        return MonteCarloComparison(
            labels=labels,
            results=results,
            duration_ms=duration_ms,
        )


# ── CLI ─────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point for Monte Carlo risk analysis."""
    import argparse
    import io
    import json
    import sys

    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace",
        )

    parser = argparse.ArgumentParser(
        description="AI Replication Sandbox — Monte Carlo Risk Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Run hundreds of randomized simulations to compute statistical risk
profiles for replication contract configurations.

Examples:
  python -m replication.montecarlo                                     # 200 runs, default
  python -m replication.montecarlo --runs 1000                         # higher precision
  python -m replication.montecarlo --strategy random --probability 0.7
  python -m replication.montecarlo --scenario balanced                 # from preset
  python -m replication.montecarlo --compare greedy conservative       # side-by-side
  python -m replication.montecarlo --compare                           # all strategies
  python -m replication.montecarlo --json                              # JSON output
        """,
    )

    parser.add_argument(
        "--runs", type=int, default=200,
        help="Number of simulation runs (default: 200)",
    )
    parser.add_argument(
        "--scenario", choices=list(PRESETS.keys()),
        help="Use a built-in scenario preset as base",
    )
    parser.add_argument(
        "--strategy", choices=[s.value for s in Strategy],
        help="Replication strategy",
    )
    parser.add_argument("--max-depth", type=int, help="Maximum replication depth")
    parser.add_argument("--max-replicas", type=int, help="Maximum concurrent replicas")
    parser.add_argument("--cooldown", type=float, help="Cooldown seconds between spawns")
    parser.add_argument("--tasks", type=int, help="Tasks per worker")
    parser.add_argument(
        "--probability", type=float,
        help="Replication probability (random strategy)",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--compare", nargs="*", metavar="STRATEGY",
        help="Compare risk across strategies (no args = all strategies)",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Show only summary and risk indicators",
    )

    args = parser.parse_args()

    # Build base scenario
    if args.scenario:
        base = PRESETS[args.scenario]
    else:
        base = ScenarioConfig()

    if args.strategy:
        base.strategy = args.strategy
    if args.max_depth is not None:
        base.max_depth = args.max_depth
    if args.max_replicas is not None:
        base.max_replicas = args.max_replicas
    if args.cooldown is not None:
        base.cooldown_seconds = args.cooldown
    if args.tasks is not None:
        base.tasks_per_worker = args.tasks
    if args.probability is not None:
        base.replication_probability = args.probability

    mc_config = MonteCarloConfig(num_runs=args.runs, base_scenario=base)
    analyzer = MonteCarloAnalyzer(mc_config)

    if args.compare is not None:
        strategies = args.compare if args.compare else None
        comparison = analyzer.compare(strategies)
        if args.json:
            print(json.dumps(comparison.to_dict(), indent=2, default=str))
        else:
            print(comparison.render())
    else:
        result = analyzer.analyze()
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, default=str))
        elif args.summary_only:
            print(result.render_summary())
            print(result.render_risk_indicators())
        else:
            print(result.render())


if __name__ == "__main__":
    main()
