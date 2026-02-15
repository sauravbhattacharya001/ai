"""Comparison runner for side-by-side simulation experiments.

Run multiple simulations with different configurations and produce
comparative analysis including tabular summaries, ranking, and
strategy effectiveness scoring.

Usage (CLI)::

    python -m replication.comparator                                  # compare all strategies
    python -m replication.comparator --strategies greedy conservative # compare specific strategies
    python -m replication.comparator --sweep max_depth 1 2 3 4 5     # parameter sweep
    python -m replication.comparator --json                           # JSON output
    python -m replication.comparator --presets minimal balanced stress # compare presets

Programmatic::

    from replication.comparator import Comparator, ComparisonConfig
    comp = Comparator()
    result = comp.compare_strategies(["greedy", "conservative", "random"])
    print(result.render())

    # Parameter sweep
    result = comp.sweep("max_depth", [1, 2, 3, 4, 5])
    print(result.render())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from .simulator import PRESETS, ScenarioConfig, SimulationReport, Simulator, Strategy


@dataclass
class RunResult:
    """Result of a single labeled simulation run."""

    label: str
    config: ScenarioConfig
    report: SimulationReport


@dataclass
class ComparisonResult:
    """Aggregated comparison of multiple simulation runs."""

    title: str
    runs: List[RunResult]
    swept_param: Optional[str] = None

    @property
    def labels(self) -> List[str]:
        return [r.label for r in self.runs]

    def _metric_table(self) -> List[Dict[str, Any]]:
        """Extract key metrics from each run into a table."""
        rows: List[Dict[str, Any]] = []
        for r in self.runs:
            rep = r.report
            max_depth_reached = max(
                (w.depth for w in rep.workers.values()), default=0
            )
            total_repl = rep.total_replications_attempted
            success_rate = (
                (rep.total_replications_succeeded / total_repl * 100)
                if total_repl > 0
                else 0.0
            )
            efficiency = (
                rep.total_tasks / len(rep.workers) if rep.workers else 0.0
            )
            rows.append(
                {
                    "label": r.label,
                    "workers": len(rep.workers),
                    "tasks": rep.total_tasks,
                    "repl_ok": rep.total_replications_succeeded,
                    "repl_denied": rep.total_replications_denied,
                    "success_rate": success_rate,
                    "max_depth": max_depth_reached,
                    "efficiency": efficiency,
                    "duration_ms": rep.duration_ms,
                }
            )
        return rows

    def _rank(self, rows: List[Dict[str, Any]], key: str, reverse: bool = True) -> Dict[str, int]:
        """Rank labels by a metric (1 = best)."""
        sorted_labels = sorted(rows, key=lambda r: r[key], reverse=reverse)
        return {r["label"]: i + 1 for i, r in enumerate(sorted_labels)}

    def render_table(self) -> str:
        """Render a comparison table of key metrics."""
        rows = self._metric_table()
        if not rows:
            return "(no runs)"

        # Column definitions: (header, key, format, width)
        columns = [
            ("Scenario", "label", "s", 20),
            ("Workers", "workers", "d", 9),
            ("Tasks", "tasks", "d", 7),
            ("Repl OK", "repl_ok", "d", 9),
            ("Denied", "repl_denied", "d", 8),
            ("Succ %", "success_rate", ".1f", 8),
            ("MaxDep", "max_depth", "d", 8),
            ("Effic", "efficiency", ".2f", 7),
            ("Time ms", "duration_ms", ".1f", 10),
        ]

        header = "  ".join(f"{col[0]:<{col[3]}}" for col in columns)
        sep = "  ".join("─" * col[3] for col in columns)

        lines: List[str] = []
        lines.append("┌" + "─" * (len(header) + 2) + "┐")
        lines.append("│ " + header + " │")
        lines.append("├" + "─" * (len(header) + 2) + "┤")

        for row in rows:
            cells: List[str] = []
            for _hdr, key, fmt, width in columns:
                val = row[key]
                formatted = f"{val:{fmt}}"
                cells.append(f"{formatted:<{width}}")
            lines.append("│ " + "  ".join(cells) + " │")

        lines.append("└" + "─" * (len(header) + 2) + "┘")
        return "\n".join(lines)

    def render_rankings(self) -> str:
        """Rank scenarios across multiple dimensions."""
        rows = self._metric_table()
        if len(rows) < 2:
            return ""

        rankings = {
            "Most Workers": self._rank(rows, "workers", reverse=True),
            "Most Tasks": self._rank(rows, "tasks", reverse=True),
            "Best Replication": self._rank(rows, "repl_ok", reverse=True),
            "Highest Efficiency": self._rank(rows, "efficiency", reverse=True),
            "Fewest Denials": self._rank(rows, "repl_denied", reverse=False),
            "Fastest": self._rank(rows, "duration_ms", reverse=False),
        }

        lines: List[str] = []
        lines.append("┌─────────────────────────────────────────┐")
        lines.append("│            Rankings (1 = best)           │")
        lines.append("└─────────────────────────────────────────┘")
        lines.append("")

        labels = [r["label"] for r in rows]
        # Header
        cat_width = 22
        label_width = max(len(l) for l in labels) + 2
        hdr = f"  {'Category':<{cat_width}}" + "".join(
            f"{l:^{label_width}}" for l in labels
        )
        lines.append(hdr)
        lines.append("  " + "─" * (cat_width + label_width * len(labels)))

        medals = {1: "🥇", 2: "🥈", 3: "🥉"}
        for category, rank_map in rankings.items():
            cells = []
            for label in labels:
                r = rank_map[label]
                medal = medals.get(r, f"#{r}")
                cells.append(f"{medal:^{label_width}}")
            lines.append(f"  {category:<{cat_width}}" + "".join(cells))

        # Overall score (lower rank sum = better)
        lines.append("")
        lines.append("  Overall Score (lower = better):")
        scores: Dict[str, int] = {}
        for label in labels:
            total = sum(rank_map[label] for rank_map in rankings.values())
            scores[label] = total
        for label in sorted(labels, key=lambda l: scores[l]):
            bar_len = max(1, 20 - scores[label])
            bar = "█" * bar_len
            lines.append(f"    {label:<16} {bar} ({scores[label]})")

        return "\n".join(lines)

    def render_insights(self) -> str:
        """Generate automated insights from the comparison."""
        rows = self._metric_table()
        if len(rows) < 2:
            return ""

        lines: List[str] = []
        lines.append("┌─────────────────────────────────────────┐")
        lines.append("│              Key Insights               │")
        lines.append("└─────────────────────────────────────────┘")
        lines.append("")

        # Find extremes
        most_workers = max(rows, key=lambda r: r["workers"])
        fewest_workers = min(rows, key=lambda r: r["workers"])
        most_efficient = max(rows, key=lambda r: r["efficiency"])
        most_denied = max(rows, key=lambda r: r["repl_denied"])
        best_success = max(rows, key=lambda r: r["success_rate"])

        lines.append(
            f"  🏭 Most prolific: {most_workers['label']} "
            f"({most_workers['workers']} workers)"
        )
        lines.append(
            f"  🎯 Most efficient: {most_efficient['label']} "
            f"({most_efficient['efficiency']:.2f} tasks/worker)"
        )
        lines.append(
            f"  ✅ Best replication rate: {best_success['label']} "
            f"({best_success['success_rate']:.1f}%)"
        )

        if most_denied["repl_denied"] > 0:
            lines.append(
                f"  🚫 Most constrained: {most_denied['label']} "
                f"({most_denied['repl_denied']} denials)"
            )

        # Spread analysis
        worker_counts = [r["workers"] for r in rows]
        spread = max(worker_counts) - min(worker_counts)
        if spread > 0:
            lines.append(
                f"  📊 Worker spread: {spread} "
                f"({fewest_workers['label']} → {most_workers['label']})"
            )

        # Depth utilization
        for r in rows:
            config = next(run.config for run in self.runs if run.label == r["label"])
            if config.max_depth > 0:
                utilization = r["max_depth"] / config.max_depth * 100
                if utilization < 50:
                    lines.append(
                        f"  ⚠️  {r['label']}: only used {utilization:.0f}% of allowed depth"
                    )

        return "\n".join(lines)

    def render(self) -> str:
        """Render the full comparison report."""
        sections: List[str] = []

        title_line = f"  Comparison: {self.title}"
        sections.append("┌" + "─" * (len(title_line) + 1) + "┐")
        sections.append("│" + title_line + " │")
        sections.append("└" + "─" * (len(title_line) + 1) + "┘")

        sections.append("")
        sections.append(self.render_table())

        rankings = self.render_rankings()
        if rankings:
            sections.append("")
            sections.append(rankings)

        insights = self.render_insights()
        if insights:
            sections.append("")
            sections.append(insights)

        return "\n".join(sections)

    def to_dict(self) -> Dict[str, Any]:
        """Export as JSON-serializable dictionary."""
        rows = self._metric_table()
        return {
            "title": self.title,
            "swept_param": self.swept_param,
            "runs": [
                {
                    "label": r.label,
                    "config": {
                        "strategy": r.config.strategy,
                        "max_depth": r.config.max_depth,
                        "max_replicas": r.config.max_replicas,
                        "cooldown_seconds": r.config.cooldown_seconds,
                        "tasks_per_worker": r.config.tasks_per_worker,
                    },
                    "metrics": next(
                        row for row in rows if row["label"] == r.label
                    ),
                    "full_report": r.report.to_dict(),
                }
                for r in self.runs
            ],
        }


class Comparator:
    """Run side-by-side simulation experiments."""

    def __init__(self, base_config: Optional[ScenarioConfig] = None) -> None:
        self.base_config = base_config or ScenarioConfig()

    def _make_config(self, **overrides: Any) -> ScenarioConfig:
        """Clone base config with overrides."""
        return ScenarioConfig(
            max_depth=overrides.get("max_depth", self.base_config.max_depth),
            max_replicas=overrides.get("max_replicas", self.base_config.max_replicas),
            cooldown_seconds=overrides.get("cooldown_seconds", self.base_config.cooldown_seconds),
            expiration_seconds=overrides.get("expiration_seconds", self.base_config.expiration_seconds),
            strategy=overrides.get("strategy", self.base_config.strategy),
            tasks_per_worker=overrides.get("tasks_per_worker", self.base_config.tasks_per_worker),
            replication_probability=overrides.get("replication_probability", self.base_config.replication_probability),
            secret=overrides.get("secret", self.base_config.secret),
            seed=overrides.get("seed", self.base_config.seed),
            cpu_limit=overrides.get("cpu_limit", self.base_config.cpu_limit),
            memory_limit_mb=overrides.get("memory_limit_mb", self.base_config.memory_limit_mb),
        )

    def compare_strategies(
        self,
        strategies: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
    ) -> ComparisonResult:
        """Run the same scenario with different replication strategies."""
        if strategies is None:
            strategies = [s.value for s in Strategy]

        runs: List[RunResult] = []
        for strat in strategies:
            config = self._make_config(strategy=strat, seed=seed)
            sim = Simulator(config)
            report = sim.run()
            runs.append(RunResult(label=strat, config=config, report=report))

        return ComparisonResult(
            title="Strategy Comparison",
            runs=runs,
        )

    def compare_presets(
        self,
        preset_names: Optional[Sequence[str]] = None,
    ) -> ComparisonResult:
        """Compare built-in scenario presets."""
        if preset_names is None:
            preset_names = list(PRESETS.keys())

        runs: List[RunResult] = []
        for name in preset_names:
            if name not in PRESETS:
                raise ValueError(f"Unknown preset: {name!r}. Available: {list(PRESETS.keys())}")
            config = PRESETS[name]
            sim = Simulator(config)
            report = sim.run()
            runs.append(RunResult(label=name, config=config, report=report))

        return ComparisonResult(
            title="Preset Comparison",
            runs=runs,
        )

    def sweep(
        self,
        param: str,
        values: Sequence[Union[int, float]],
        seed: Optional[int] = None,
    ) -> ComparisonResult:
        """Sweep a single parameter across multiple values.

        Supported parameters: max_depth, max_replicas, cooldown_seconds,
        tasks_per_worker, replication_probability, cpu_limit, memory_limit_mb.
        """
        valid_params = {
            "max_depth", "max_replicas", "cooldown_seconds",
            "tasks_per_worker", "replication_probability",
            "cpu_limit", "memory_limit_mb",
        }
        if param not in valid_params:
            raise ValueError(
                f"Cannot sweep {param!r}. Sweepable: {sorted(valid_params)}"
            )

        runs: List[RunResult] = []
        for val in values:
            overrides: Dict[str, Any] = {param: val}
            if seed is not None:
                overrides["seed"] = seed
            config = self._make_config(**overrides)
            sim = Simulator(config)
            report = sim.run()
            runs.append(
                RunResult(
                    label=f"{param}={val}",
                    config=config,
                    report=report,
                )
            )

        return ComparisonResult(
            title=f"Parameter Sweep: {param}",
            runs=runs,
            swept_param=param,
        )

    def compare_configs(
        self,
        configs: Dict[str, ScenarioConfig],
    ) -> ComparisonResult:
        """Compare arbitrary named configurations."""
        runs: List[RunResult] = []
        for label, config in configs.items():
            sim = Simulator(config)
            report = sim.run()
            runs.append(RunResult(label=label, config=config, report=report))

        return ComparisonResult(
            title="Custom Comparison",
            runs=runs,
        )


def main() -> None:
    """CLI entry point for the comparison tool."""
    import argparse
    import io
    import json
    import sys

    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )

    parser = argparse.ArgumentParser(
        description="AI Replication Sandbox — Comparison Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Compare replication strategies, presets, or sweep parameters.

Examples:
  python -m replication.comparator                                    # all strategies
  python -m replication.comparator --strategies greedy conservative   # specific strategies
  python -m replication.comparator --presets minimal balanced stress  # compare presets
  python -m replication.comparator --sweep max_depth 1 2 3 4 5       # parameter sweep
  python -m replication.comparator --sweep max_replicas 5 10 20 50   # replica sweep
  python -m replication.comparator --seed 42 --json                  # reproducible JSON
        """,
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--strategies",
        nargs="+",
        choices=[s.value for s in Strategy],
        help="Compare specific strategies (default: all)",
    )
    mode.add_argument(
        "--presets",
        nargs="+",
        choices=list(PRESETS.keys()),
        help="Compare built-in presets",
    )
    mode.add_argument(
        "--sweep",
        nargs="+",
        metavar="PARAM_OR_VALUE",
        help="Sweep a parameter: --sweep <param> <val1> <val2> ...",
    )

    parser.add_argument("--max-depth", type=int, help="Base max depth")
    parser.add_argument("--max-replicas", type=int, help="Base max replicas")
    parser.add_argument("--cooldown", type=float, help="Base cooldown seconds")
    parser.add_argument("--tasks", type=int, help="Base tasks per worker")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--table-only", action="store_true", help="Show only the comparison table"
    )

    args = parser.parse_args()

    # Build base config from CLI args
    base_overrides: Dict[str, Any] = {}
    if args.max_depth is not None:
        base_overrides["max_depth"] = args.max_depth
    if args.max_replicas is not None:
        base_overrides["max_replicas"] = args.max_replicas
    if args.cooldown is not None:
        base_overrides["cooldown_seconds"] = args.cooldown
    if args.tasks is not None:
        base_overrides["tasks_per_worker"] = args.tasks

    base = ScenarioConfig(**base_overrides) if base_overrides else ScenarioConfig()
    comp = Comparator(base_config=base)

    if args.presets:
        result = comp.compare_presets(args.presets)
    elif args.sweep:
        if len(args.sweep) < 2:
            parser.error("--sweep requires <param> <val1> [val2 ...]")
        param_name = args.sweep[0]
        try:
            sweep_values: List[Union[int, float]] = []
            for v in args.sweep[1:]:
                sweep_values.append(int(v) if "." not in v else float(v))
        except ValueError:
            parser.error(f"Sweep values must be numeric, got: {args.sweep[1:]}")
        result = comp.sweep(param_name, sweep_values, seed=args.seed)
    else:
        strategies = args.strategies  # None means all
        result = comp.compare_strategies(strategies, seed=args.seed)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    elif args.table_only:
        print(result.render_table())
    else:
        print(result.render())


if __name__ == "__main__":
    main()
