"""Parameter Sensitivity Analyzer for replication safety experiments.

Systematically varies one parameter at a time (OAT analysis) while holding
all others fixed.  Produces sensitivity curves, identifies critical thresholds
and tipping points, and ranks parameters by their impact on safety outcomes.

Usage (CLI)::

    python -m replication.sensitivity                          # sweep all params
    python -m replication.sensitivity --param max_depth        # single param
    python -m replication.sensitivity --runs 50 --steps 8      # more precision
    python -m replication.sensitivity --strategy greedy        # base strategy
    python -m replication.sensitivity --json                   # JSON output
    python -m replication.sensitivity --scenario balanced      # from preset

Programmatic::

    from replication.sensitivity import SensitivityAnalyzer, SensitivityConfig
    analyzer = SensitivityAnalyzer(SensitivityConfig(runs_per_point=20))
    result = analyzer.analyze()
    print(result.render())

    # Single parameter
    curve = analyzer.sweep_parameter("max_depth")
    print(curve.render())
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .simulator import PRESETS, ScenarioConfig, SimulationReport, Simulator


# ── Statistics helpers (reuse from montecarlo where needed) ─────────────

def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


# ── Parameter definitions ──────────────────────────────────────────────

@dataclass
class ParameterDef:
    """Defines a sweepable parameter with its range and type."""

    name: str
    display_name: str
    description: str
    values: List[Any]
    setter: Callable[[ScenarioConfig, Any], None]
    unit: str = ""


def _make_param_defs() -> Dict[str, ParameterDef]:
    """Build the catalogue of sweepable parameters."""

    def _set(attr: str):
        def setter(cfg: ScenarioConfig, val: Any) -> None:
            setattr(cfg, attr, val)
        return setter

    return {
        "max_depth": ParameterDef(
            name="max_depth",
            display_name="Max Depth",
            description="Maximum nesting level for replication",
            values=[1, 2, 3, 4, 5, 6, 7, 8],
            setter=_set("max_depth"),
            unit="levels",
        ),
        "max_replicas": ParameterDef(
            name="max_replicas",
            display_name="Max Replicas",
            description="Quota ceiling for concurrent worker count",
            values=[2, 5, 10, 15, 20, 30, 40, 50],
            setter=_set("max_replicas"),
            unit="workers",
        ),
        "cooldown_seconds": ParameterDef(
            name="cooldown_seconds",
            display_name="Cooldown",
            description="Minimum delay between spawn attempts",
            values=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
            setter=_set("cooldown_seconds"),
            unit="seconds",
        ),
        "tasks_per_worker": ParameterDef(
            name="tasks_per_worker",
            display_name="Tasks/Worker",
            description="Number of tasks each worker performs",
            values=[1, 2, 3, 4, 5, 6, 8, 10],
            setter=_set("tasks_per_worker"),
            unit="tasks",
        ),
        "replication_probability": ParameterDef(
            name="replication_probability",
            display_name="Replication Probability",
            description="Per-task chance of spawning (random strategy)",
            values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            setter=_set("replication_probability"),
            unit="probability",
        ),
        "cpu_limit": ParameterDef(
            name="cpu_limit",
            display_name="CPU Limit",
            description="CPU cores allocated per worker",
            values=[0.1, 0.25, 0.5, 1.0, 2.0, 4.0],
            setter=_set("cpu_limit"),
            unit="cores",
        ),
        "memory_limit_mb": ParameterDef(
            name="memory_limit_mb",
            display_name="Memory Limit",
            description="RAM allocated per worker",
            values=[64, 128, 256, 512, 1024, 2048],
            setter=_set("memory_limit_mb"),
            unit="MB",
        ),
    }


PARAM_DEFS = _make_param_defs()

# ── Metrics extracted from simulation reports ──────────────────────────

METRIC_NAMES = [
    "total_workers",
    "total_tasks",
    "replications_succeeded",
    "replications_denied",
    "max_depth_reached",
    "denial_rate",
    "efficiency",
    "total_cpu",
    "total_memory_mb",
]


def _extract_metrics(report: SimulationReport) -> Dict[str, float]:
    """Extract key safety metrics from a simulation report."""
    n_workers = len(report.workers)
    max_depth = max((w.depth for w in report.workers.values()), default=0)
    total_attempted = report.total_replications_attempted
    denial_rate = (
        report.total_replications_denied / total_attempted
        if total_attempted > 0
        else 0.0
    )
    efficiency = (
        report.total_tasks / n_workers if n_workers > 0 else 0.0
    )

    return {
        "total_workers": float(n_workers),
        "total_tasks": float(report.total_tasks),
        "replications_succeeded": float(report.total_replications_succeeded),
        "replications_denied": float(report.total_replications_denied),
        "max_depth_reached": float(max_depth),
        "denial_rate": denial_rate,
        "efficiency": efficiency,
        "total_cpu": report.config.cpu_limit * n_workers,
        "total_memory_mb": float(report.config.memory_limit_mb * n_workers),
    }


# ── Data models ────────────────────────────────────────────────────────

@dataclass
class SweepPoint:
    """Aggregated metrics at a single parameter value."""

    param_value: Any
    metric_means: Dict[str, float]
    metric_stds: Dict[str, float]
    num_runs: int


@dataclass
class TippingPoint:
    """A detected tipping point where a metric changes sharply."""

    metric: str
    param_value_before: Any
    param_value_after: Any
    change_rate: float  # absolute change between consecutive points
    relative_change: float  # relative to the mean


@dataclass
class SensitivityCurve:
    """Results of sweeping a single parameter."""

    param: ParameterDef
    points: List[SweepPoint]
    tipping_points: List[TippingPoint]
    impact_scores: Dict[str, float]  # metric -> normalized impact score

    def render(self) -> str:
        """Render the sensitivity curve as a text table."""
        lines: List[str] = []
        lines.append(f"{'=' * 70}")
        lines.append(f"  Sensitivity: {self.param.display_name}")
        lines.append(f"  {self.param.description}")
        lines.append(f"{'=' * 70}")
        lines.append("")

        # Main metrics table
        key_metrics = [
            "total_workers", "total_tasks", "max_depth_reached",
            "denial_rate", "efficiency",
        ]

        # Header
        header = f"  {'Value':>10}"
        for m in key_metrics:
            label = m.replace("_", " ").title()
            if len(label) > 12:
                label = label[:12]
            header += f"  {label:>12}"
        lines.append(header)
        lines.append("  " + "-" * (10 + len(key_metrics) * 14))

        for pt in self.points:
            row = f"  {str(pt.param_value):>10}"
            for m in key_metrics:
                val = pt.metric_means.get(m, 0)
                if m == "denial_rate":
                    row += f"  {val:>11.1%} "
                elif val == int(val):
                    row += f"  {int(val):>12}"
                else:
                    row += f"  {val:>12.2f}"
            lines.append(row)

        # Tipping points
        if self.tipping_points:
            lines.append("")
            lines.append("  Tipping Points:")
            for tp in self.tipping_points:
                label = tp.metric.replace("_", " ").title()
                lines.append(
                    f"    ⚠  {label}: sharp change between "
                    f"{tp.param_value_before} → {tp.param_value_after} "
                    f"(Δ = {tp.change_rate:+.2f}, {tp.relative_change:+.0%})"
                )

        # Impact scores
        if self.impact_scores:
            lines.append("")
            lines.append("  Impact Scores (normalized 0-100):")
            ranked = sorted(
                self.impact_scores.items(), key=lambda x: -x[1]
            )
            for metric, score in ranked:
                label = metric.replace("_", " ").title()
                bar = "█" * int(score / 2)
                lines.append(f"    {label:<25} {score:>5.1f}  {bar}")

        lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter": self.param.name,
            "display_name": self.param.display_name,
            "description": self.param.description,
            "unit": self.param.unit,
            "points": [
                {
                    "value": pt.param_value,
                    "means": {k: round(v, 4) for k, v in pt.metric_means.items()},
                    "stds": {k: round(v, 4) for k, v in pt.metric_stds.items()},
                    "num_runs": pt.num_runs,
                }
                for pt in self.points
            ],
            "tipping_points": [
                {
                    "metric": tp.metric,
                    "before": tp.param_value_before,
                    "after": tp.param_value_after,
                    "change_rate": round(tp.change_rate, 4),
                    "relative_change": round(tp.relative_change, 4),
                }
                for tp in self.tipping_points
            ],
            "impact_scores": {
                k: round(v, 2) for k, v in self.impact_scores.items()
            },
        }


@dataclass
class SensitivityResult:
    """Complete sensitivity analysis across all parameters."""

    base_config: ScenarioConfig
    curves: Dict[str, SensitivityCurve]
    parameter_ranking: List[Tuple[str, float]]  # (param, overall impact)
    all_tipping_points: List[Tuple[str, TippingPoint]]  # (param, tp)
    duration_ms: float
    total_simulations: int

    def render(self) -> str:
        lines: List[str] = []
        lines.append("╔══════════════════════════════════════════════════════════════╗")
        lines.append("║          Parameter Sensitivity Analysis                     ║")
        lines.append("╚══════════════════════════════════════════════════════════════╝")
        lines.append("")
        lines.append(f"  Base strategy:       {self.base_config.strategy}")
        lines.append(f"  Total simulations:   {self.total_simulations}")
        lines.append(f"  Duration:            {self.duration_ms / 1000:.1f}s")
        lines.append("")

        # Parameter ranking
        lines.append("  Parameter Impact Ranking (overall influence on safety):")
        lines.append("  " + "-" * 55)
        for i, (param, score) in enumerate(self.parameter_ranking):
            pdef = PARAM_DEFS.get(param)
            label = pdef.display_name if pdef else param
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            bar = "█" * int(score / 2)
            lines.append(f"  {medal} {label:<25} {score:>5.1f}  {bar}")
        lines.append("")

        # Critical tipping points
        if self.all_tipping_points:
            lines.append("  ⚠  Critical Tipping Points:")
            lines.append("  " + "-" * 55)
            for param, tp in self.all_tipping_points[:10]:
                pdef = PARAM_DEFS.get(param)
                plabel = pdef.display_name if pdef else param
                mlabel = tp.metric.replace("_", " ").title()
                lines.append(
                    f"    {plabel}: {tp.param_value_before} → {tp.param_value_after}"
                    f"  |  {mlabel} Δ = {tp.change_rate:+.2f} ({tp.relative_change:+.0%})"
                )
            lines.append("")

        # Individual curves
        for curve in self.curves.values():
            lines.append(curve.render())

        # Recommendations
        lines.append(self._generate_recommendations())

        return "\n".join(lines)

    def _generate_recommendations(self) -> str:
        lines: List[str] = []
        lines.append("  Recommendations:")
        lines.append("  " + "-" * 55)

        if not self.parameter_ranking:
            lines.append("    No data available for recommendations.")
            return "\n".join(lines)

        # Top impactful param
        top_param, top_score = self.parameter_ranking[0]
        pdef = PARAM_DEFS.get(top_param)
        plabel = pdef.display_name if pdef else top_param
        lines.append(
            f"    1. {plabel} has the highest impact ({top_score:.0f}/100)."
            f" Prioritize tuning this parameter."
        )

        # Tipping point warning
        if self.all_tipping_points:
            param, tp = self.all_tipping_points[0]
            pdef_tp = PARAM_DEFS.get(param)
            plabel_tp = pdef_tp.display_name if pdef_tp else param
            lines.append(
                f"    2. Critical threshold at {plabel_tp} = {tp.param_value_after}."
                f" Stay below this value for stable operation."
            )

        # Low-impact params
        low_impact = [
            (p, s) for p, s in self.parameter_ranking if s < 10
        ]
        if low_impact:
            names = ", ".join(
                PARAM_DEFS[p].display_name if p in PARAM_DEFS else p
                for p, _ in low_impact
            )
            lines.append(
                f"    3. Low-impact parameters ({names}) can be left at defaults."
            )

        lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_config": {
                "strategy": self.base_config.strategy,
                "max_depth": self.base_config.max_depth,
                "max_replicas": self.base_config.max_replicas,
                "cooldown_seconds": self.base_config.cooldown_seconds,
                "tasks_per_worker": self.base_config.tasks_per_worker,
            },
            "parameter_ranking": [
                {"parameter": p, "impact_score": round(s, 2)}
                for p, s in self.parameter_ranking
            ],
            "tipping_points": [
                {
                    "parameter": param,
                    "metric": tp.metric,
                    "before": tp.param_value_before,
                    "after": tp.param_value_after,
                    "change_rate": round(tp.change_rate, 4),
                    "relative_change": round(tp.relative_change, 4),
                }
                for param, tp in self.all_tipping_points
            ],
            "curves": {
                name: curve.to_dict() for name, curve in self.curves.items()
            },
            "total_simulations": self.total_simulations,
            "duration_ms": round(self.duration_ms, 1),
        }


# ── Configuration ──────────────────────────────────────────────────────

@dataclass
class SensitivityConfig:
    """Configuration for a sensitivity analysis run."""

    runs_per_point: int = 10
    """Number of simulations at each parameter value (averaged)."""

    parameters: Optional[List[str]] = None
    """Parameters to sweep (None = all)."""

    base_scenario: Optional[str] = None
    """Named preset for the base config."""

    base_config: Optional[ScenarioConfig] = None
    """Explicit base config (overrides base_scenario)."""

    # Base-config overrides (applied on top of scenario/defaults)
    strategy: Optional[str] = None
    max_depth: Optional[int] = None
    max_replicas: Optional[int] = None
    cooldown_seconds: Optional[float] = None
    tasks_per_worker: Optional[int] = None
    replication_probability: Optional[float] = None


# ── Analyzer ───────────────────────────────────────────────────────────

class SensitivityAnalyzer:
    """Performs one-at-a-time parameter sensitivity analysis."""

    # Threshold for tipping point detection: >50% relative change
    TIPPING_THRESHOLD = 0.5

    def __init__(self, config: Optional[SensitivityConfig] = None) -> None:
        self.config = config or SensitivityConfig()
        self._total_sims = 0

    def _make_base_config(self) -> ScenarioConfig:
        """Build the base scenario config from settings."""
        if self.config.base_config:
            cfg = ScenarioConfig(
                max_depth=self.config.base_config.max_depth,
                max_replicas=self.config.base_config.max_replicas,
                cooldown_seconds=self.config.base_config.cooldown_seconds,
                expiration_seconds=self.config.base_config.expiration_seconds,
                strategy=self.config.base_config.strategy,
                tasks_per_worker=self.config.base_config.tasks_per_worker,
                replication_probability=self.config.base_config.replication_probability,
                cpu_limit=self.config.base_config.cpu_limit,
                memory_limit_mb=self.config.base_config.memory_limit_mb,
            )
        elif self.config.base_scenario and self.config.base_scenario in PRESETS:
            preset = PRESETS[self.config.base_scenario]
            cfg = ScenarioConfig(
                max_depth=preset.max_depth,
                max_replicas=preset.max_replicas,
                cooldown_seconds=preset.cooldown_seconds,
                expiration_seconds=preset.expiration_seconds,
                strategy=preset.strategy,
                tasks_per_worker=preset.tasks_per_worker,
                replication_probability=preset.replication_probability,
                cpu_limit=preset.cpu_limit,
                memory_limit_mb=preset.memory_limit_mb,
            )
        else:
            cfg = ScenarioConfig()

        # Apply explicit overrides
        if self.config.strategy:
            cfg.strategy = self.config.strategy
        if self.config.max_depth is not None:
            cfg.max_depth = self.config.max_depth
        if self.config.max_replicas is not None:
            cfg.max_replicas = self.config.max_replicas
        if self.config.cooldown_seconds is not None:
            cfg.cooldown_seconds = self.config.cooldown_seconds
        if self.config.tasks_per_worker is not None:
            cfg.tasks_per_worker = self.config.tasks_per_worker
        if self.config.replication_probability is not None:
            cfg.replication_probability = self.config.replication_probability

        return cfg

    def _clone_config(self, base: ScenarioConfig) -> ScenarioConfig:
        """Deep-copy a scenario config."""
        return ScenarioConfig(
            max_depth=base.max_depth,
            max_replicas=base.max_replicas,
            cooldown_seconds=base.cooldown_seconds,
            expiration_seconds=base.expiration_seconds,
            strategy=base.strategy,
            tasks_per_worker=base.tasks_per_worker,
            replication_probability=base.replication_probability,
            secret=base.secret,
            cpu_limit=base.cpu_limit,
            memory_limit_mb=base.memory_limit_mb,
        )

    def _run_batch(
        self, config: ScenarioConfig, n: int
    ) -> List[Dict[str, float]]:
        """Run n simulations and return extracted metrics."""
        results: List[Dict[str, float]] = []
        for i in range(n):
            cfg = self._clone_config(config)
            cfg.seed = i * 7919 + hash(str(config.strategy)) % 10000
            sim = Simulator(cfg)
            report = sim.run()
            results.append(_extract_metrics(report))
            self._total_sims += 1
        return results

    def sweep_parameter(
        self, param_name: str, base: Optional[ScenarioConfig] = None
    ) -> SensitivityCurve:
        """Sweep a single parameter across its value range."""
        if param_name not in PARAM_DEFS:
            raise ValueError(
                f"Unknown parameter '{param_name}'. "
                f"Available: {', '.join(PARAM_DEFS.keys())}"
            )

        pdef = PARAM_DEFS[param_name]
        base_cfg = base or self._make_base_config()
        points: List[SweepPoint] = []

        for val in pdef.values:
            cfg = self._clone_config(base_cfg)
            pdef.setter(cfg, val)

            batch = self._run_batch(cfg, self.config.runs_per_point)

            # Aggregate means and stds
            means: Dict[str, float] = {}
            stds: Dict[str, float] = {}
            for metric in METRIC_NAMES:
                values = [b[metric] for b in batch]
                means[metric] = _mean(values)
                stds[metric] = _std(values)

            points.append(SweepPoint(
                param_value=val,
                metric_means=means,
                metric_stds=stds,
                num_runs=len(batch),
            ))

        # Detect tipping points
        tipping_points = self._detect_tipping_points(pdef, points)

        # Compute impact scores
        impact_scores = self._compute_impact_scores(points)

        return SensitivityCurve(
            param=pdef,
            points=points,
            tipping_points=tipping_points,
            impact_scores=impact_scores,
        )

    def _detect_tipping_points(
        self, pdef: ParameterDef, points: List[SweepPoint]
    ) -> List[TippingPoint]:
        """Find consecutive points with sharp metric changes."""
        tipping: List[TippingPoint] = []

        if len(points) < 2:
            return tipping

        for metric in METRIC_NAMES:
            for i in range(len(points) - 1):
                val_a = points[i].metric_means.get(metric, 0)
                val_b = points[i + 1].metric_means.get(metric, 0)
                change = val_b - val_a
                base_val = abs(val_a) if val_a != 0 else abs(val_b)

                if base_val == 0:
                    continue

                relative = change / base_val

                if abs(relative) >= self.TIPPING_THRESHOLD:
                    tipping.append(TippingPoint(
                        metric=metric,
                        param_value_before=points[i].param_value,
                        param_value_after=points[i + 1].param_value,
                        change_rate=change,
                        relative_change=relative,
                    ))

        # Sort by magnitude of relative change
        tipping.sort(key=lambda tp: -abs(tp.relative_change))
        return tipping

    def _compute_impact_scores(
        self, points: List[SweepPoint]
    ) -> Dict[str, float]:
        """Compute normalized 0–100 impact score per metric.

        Score = coefficient of variation across sweep points,
        normalized to the maximum across all metrics.
        """
        scores: Dict[str, float] = {}

        for metric in METRIC_NAMES:
            values = [pt.metric_means.get(metric, 0) for pt in points]
            mean_val = _mean(values)
            if mean_val == 0:
                scores[metric] = 0.0
                continue
            # Range-based variation
            val_range = max(values) - min(values)
            scores[metric] = val_range / abs(mean_val)

        # Normalize to 0–100
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            for k in scores:
                scores[k] = (scores[k] / max_score) * 100

        return scores

    def analyze(self) -> SensitivityResult:
        """Run full sensitivity analysis across all (or selected) parameters."""
        start = time.monotonic()
        self._total_sims = 0
        base_cfg = self._make_base_config()

        params = self.config.parameters or list(PARAM_DEFS.keys())
        curves: Dict[str, SensitivityCurve] = {}

        for param_name in params:
            if param_name in PARAM_DEFS:
                curves[param_name] = self.sweep_parameter(param_name, base_cfg)

        # Overall parameter ranking (average impact across all metrics)
        param_ranking: List[Tuple[str, float]] = []
        for param_name, curve in curves.items():
            if curve.impact_scores:
                avg_impact = _mean(list(curve.impact_scores.values()))
            else:
                avg_impact = 0.0
            param_ranking.append((param_name, avg_impact))

        param_ranking.sort(key=lambda x: -x[1])

        # Collect all tipping points
        all_tipping: List[Tuple[str, TippingPoint]] = []
        for param_name, curve in curves.items():
            for tp in curve.tipping_points:
                all_tipping.append((param_name, tp))
        all_tipping.sort(key=lambda x: -abs(x[1].relative_change))

        elapsed = (time.monotonic() - start) * 1000

        return SensitivityResult(
            base_config=base_cfg,
            curves=curves,
            parameter_ranking=param_ranking,
            all_tipping_points=all_tipping,
            duration_ms=elapsed,
            total_simulations=self._total_sims,
        )


# ── CLI ────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point for parameter sensitivity analysis."""
    import argparse
    import json
    import sys
    import io

    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )

    parser = argparse.ArgumentParser(
        description="AI Replication Sandbox — Parameter Sensitivity Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sweepable parameters: max_depth, max_replicas, cooldown_seconds,
                      tasks_per_worker, replication_probability,
                      cpu_limit, memory_limit_mb

Examples:
  python -m replication.sensitivity                        # sweep all params
  python -m replication.sensitivity --param max_depth      # single param
  python -m replication.sensitivity --runs 50              # more precision
  python -m replication.sensitivity --strategy greedy      # base strategy
  python -m replication.sensitivity --scenario balanced    # from preset
  python -m replication.sensitivity --json                 # JSON output
        """,
    )

    parser.add_argument(
        "--param", choices=list(PARAM_DEFS.keys()),
        help="Sweep a single parameter (default: all)",
    )
    parser.add_argument("--runs", type=int, default=10, help="Runs per sweep point")
    parser.add_argument(
        "--scenario", choices=list(PRESETS.keys()),
        help="Base scenario preset",
    )
    parser.add_argument("--strategy", choices=["greedy", "conservative", "random", "chain", "burst"])
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--max-replicas", type=int)
    parser.add_argument("--cooldown", type=float)
    parser.add_argument("--tasks", type=int)
    parser.add_argument("--probability", type=float)
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--summary-only", action="store_true", help="Show only ranking and tipping points")

    args = parser.parse_args()

    cfg = SensitivityConfig(
        runs_per_point=args.runs,
        base_scenario=args.scenario,
        strategy=args.strategy,
        parameters=[args.param] if args.param else None,
        max_depth=args.max_depth,
        max_replicas=args.max_replicas,
        cooldown_seconds=args.cooldown,
        tasks_per_worker=args.tasks,
        replication_probability=args.probability,
    )

    analyzer = SensitivityAnalyzer(cfg)
    result = analyzer.analyze()

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    elif args.summary_only:
        lines: List[str] = []
        lines.append("Parameter Impact Ranking:")
        for param, score in result.parameter_ranking:
            pdef = PARAM_DEFS.get(param)
            label = pdef.display_name if pdef else param
            lines.append(f"  {label:<25} {score:.1f}/100")
        if result.all_tipping_points:
            lines.append("")
            lines.append("Critical Tipping Points:")
            for param, tp in result.all_tipping_points[:5]:
                pdef = PARAM_DEFS.get(param)
                plabel = pdef.display_name if pdef else param
                lines.append(
                    f"  {plabel}: {tp.param_value_before} -> {tp.param_value_after}"
                    f" ({tp.metric}, {tp.relative_change:+.0%})"
                )
        print("\n".join(lines))
    else:
        print(result.render())


if __name__ == "__main__":
    main()
