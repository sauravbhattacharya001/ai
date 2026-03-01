"""Contract Optimizer — find optimal contract parameters under safety constraints.

Searches the parameter space (max_depth, max_replicas, cooldown_seconds) to
find configurations that maximize worker throughput while satisfying a given
safety policy.  Uses a grid search with optional refinement.

Usage (CLI)::

    python -m replication.optimizer                                   # optimize with standard policy
    python -m replication.optimizer --policy strict                   # strict safety constraints
    python -m replication.optimizer --strategy greedy                 # for a specific strategy
    python -m replication.optimizer --objective throughput            # maximize throughput (default)
    python -m replication.optimizer --objective efficiency            # maximize tasks-per-worker
    python -m replication.optimizer --objective safety                # maximize denial rate
    python -m replication.optimizer --depth-range 1 6                 # custom depth search range
    python -m replication.optimizer --replicas-range 5 30             # custom replicas search range
    python -m replication.optimizer --cooldown-range 0.0 5.0 1.0     # custom cooldown sweep
    python -m replication.optimizer --top 5                           # show top 5 configs
    python -m replication.optimizer --json                            # JSON output
    python -m replication.optimizer --refine                          # refine around best result

Programmatic::

    from replication.optimizer import ContractOptimizer, OptimizerConfig
    opt = ContractOptimizer()
    result = opt.optimize()
    print(result.render())

    # With custom config
    config = OptimizerConfig(
        policy_preset="strict",
        objective="efficiency",
        depth_range=(1, 4),
    )
    opt = ContractOptimizer(config)
    result = opt.optimize()
    print(result.best)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .policy import SafetyPolicy, PolicyResult, POLICY_PRESETS
from .simulator import ScenarioConfig, SimulationReport, Simulator, Strategy


class Objective(Enum):
    """Optimization objective."""
    THROUGHPUT = "throughput"     # maximize total tasks completed
    EFFICIENCY = "efficiency"    # maximize tasks per worker
    SAFETY = "safety"            # maximize denial rate (strictest containment)
    BALANCED = "balanced"        # weighted combination


def _score_report(report: SimulationReport, objective: Objective) -> float:
    """Score a simulation report according to the given objective."""
    total_workers = len(report.workers)
    total_tasks = report.total_tasks
    attempted = report.total_replications_attempted
    denied = report.total_replications_denied

    if objective == Objective.THROUGHPUT:
        return float(total_tasks)
    elif objective == Objective.EFFICIENCY:
        return (total_tasks / total_workers) if total_workers > 0 else 0.0
    elif objective == Objective.SAFETY:
        return (denied / attempted) if attempted > 0 else 1.0
    elif objective == Objective.BALANCED:
        throughput = float(total_tasks)
        efficiency = (total_tasks / total_workers) if total_workers > 0 else 0.0
        denial_rate = (denied / attempted) if attempted > 0 else 1.0
        # Normalize and combine: throughput matters most, but penalize
        # low denial rates and low efficiency
        return throughput * 0.5 + efficiency * 10.0 * 0.3 + denial_rate * 100.0 * 0.2
    return 0.0


@dataclass
class CandidateResult:
    """Result of evaluating a single parameter configuration."""
    max_depth: int
    max_replicas: int
    cooldown_seconds: float
    strategy: str
    score: float
    policy_passed: bool
    policy_verdict: str
    total_workers: int
    total_tasks: int
    denial_rate: float
    duration_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_depth": self.max_depth,
            "max_replicas": self.max_replicas,
            "cooldown_seconds": self.cooldown_seconds,
            "strategy": self.strategy,
            "score": round(self.score, 3),
            "policy_passed": self.policy_passed,
            "policy_verdict": self.policy_verdict,
            "total_workers": self.total_workers,
            "total_tasks": self.total_tasks,
            "denial_rate": round(self.denial_rate, 4),
            "duration_ms": round(self.duration_ms, 1),
        }


@dataclass
class OptimizerConfig:
    """Configuration for the contract optimizer."""
    policy_preset: str = "standard"
    objective: str = "throughput"
    strategy: str = "greedy"
    depth_range: Tuple[int, int] = (1, 6)
    replicas_range: Tuple[int, int] = (5, 25)
    cooldown_values: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 5.0)
    tasks_per_worker: int = 3
    replication_probability: float = 0.8
    seed: Optional[int] = 42
    top_n: int = 10
    refine: bool = False
    refine_steps: int = 3


@dataclass
class OptimizerResult:
    """Result of the optimization run."""
    config: OptimizerConfig
    candidates: List[CandidateResult]
    passing: List[CandidateResult]
    best: Optional[CandidateResult]
    total_evaluated: int
    total_passing: int
    duration_ms: float

    def render(self) -> str:
        lines: List[str] = []
        lines.append("┌───────────────────────────────────────────────────────┐")
        lines.append("│       ⚙️  Contract Optimizer Results  ⚙️              │")
        lines.append("└───────────────────────────────────────────────────────┘")
        lines.append("")
        lines.append(f"  Objective:    {self.config.objective}")
        lines.append(f"  Policy:       {self.config.policy_preset}")
        lines.append(f"  Strategy:     {self.config.strategy}")
        lines.append(f"  Evaluated:    {self.total_evaluated} configurations")
        lines.append(f"  Passing:      {self.total_passing} / {self.total_evaluated}")
        lines.append(f"  Duration:     {self.duration_ms:.0f}ms")
        lines.append("")

        if self.best:
            lines.append("  ── Best Configuration ──────────────────────────────")
            lines.append(f"    max_depth:       {self.best.max_depth}")
            lines.append(f"    max_replicas:    {self.best.max_replicas}")
            lines.append(f"    cooldown:        {self.best.cooldown_seconds}s")
            lines.append(f"    score:           {self.best.score:.3f}")
            lines.append(f"    total_tasks:     {self.best.total_tasks}")
            lines.append(f"    total_workers:   {self.best.total_workers}")
            lines.append(f"    denial_rate:     {self.best.denial_rate:.1%}")
            lines.append(f"    policy:          {self.best.policy_verdict}")
            lines.append("")
        else:
            lines.append("  ⚠️  No configuration passed the safety policy!")
            lines.append("")

        # Top N table
        show = self.passing[:self.config.top_n] if self.passing else self.candidates[:5]
        label = "Top Passing" if self.passing else "Top (none passed policy)"
        lines.append(f"  ── {label} ──")
        hdr = f"  {'#':>3}  {'Depth':>5}  {'Replicas':>8}  {'Cooldown':>8}  {'Score':>8}  {'Tasks':>6}  {'Workers':>7}  {'Deny%':>6}  {'Policy':>7}"
        lines.append(hdr)
        lines.append("  " + "─" * 78)
        for i, c in enumerate(show, 1):
            verdict = "✅" if c.policy_passed else "❌"
            lines.append(
                f"  {i:>3}  {c.max_depth:>5}  {c.max_replicas:>8}  "
                f"{c.cooldown_seconds:>8.1f}  {c.score:>8.1f}  "
                f"{c.total_tasks:>6}  {c.total_workers:>7}  "
                f"{c.denial_rate:>5.1%}  {verdict:>7}"
            )
        lines.append("")

        # Insights
        if self.passing and len(self.passing) >= 3:
            lines.append("  ── Insights ────────────────────────────────────────")
            depths = [c.max_depth for c in self.passing[:5]]
            replicas = [c.max_replicas for c in self.passing[:5]]
            lines.append(f"    Best depth range:    {min(depths)}-{max(depths)}")
            lines.append(f"    Best replicas range: {min(replicas)}-{max(replicas)}")
            avg_cooldown = sum(c.cooldown_seconds for c in self.passing[:5]) / len(self.passing[:5])
            lines.append(f"    Avg cooldown (top 5): {avg_cooldown:.1f}s")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": {
                "policy_preset": self.config.policy_preset,
                "objective": self.config.objective,
                "strategy": self.config.strategy,
                "depth_range": list(self.config.depth_range),
                "replicas_range": list(self.config.replicas_range),
                "cooldown_values": list(self.config.cooldown_values),
            },
            "total_evaluated": self.total_evaluated,
            "total_passing": self.total_passing,
            "duration_ms": round(self.duration_ms, 1),
            "best": self.best.to_dict() if self.best else None,
            "passing": [c.to_dict() for c in self.passing[:self.config.top_n]],
        }


class ContractOptimizer:
    """Searches for optimal contract parameters under safety constraints."""

    def __init__(self, config: Optional[OptimizerConfig] = None) -> None:
        self.config = config or OptimizerConfig()

    def _evaluate_candidate(
        self,
        max_depth: int,
        max_replicas: int,
        cooldown: float,
        policy: SafetyPolicy,
        objective: Objective,
    ) -> CandidateResult:
        """Run a simulation and evaluate one parameter combination."""
        scenario = ScenarioConfig(
            max_depth=max_depth,
            max_replicas=max_replicas,
            cooldown_seconds=cooldown,
            strategy=self.config.strategy,
            tasks_per_worker=self.config.tasks_per_worker,
            replication_probability=self.config.replication_probability,
            seed=self.config.seed,
        )

        start = time.monotonic()
        sim = Simulator(scenario)
        report = sim.run()
        duration_ms = (time.monotonic() - start) * 1000

        # Evaluate policy (single-run only, skip MC for speed)
        policy_result = policy.evaluate(report)

        # Score
        score = _score_report(report, objective)

        attempted = report.total_replications_attempted
        denial_rate = (report.total_replications_denied / attempted) if attempted > 0 else 0.0

        return CandidateResult(
            max_depth=max_depth,
            max_replicas=max_replicas,
            cooldown_seconds=cooldown,
            strategy=self.config.strategy,
            score=score,
            policy_passed=policy_result.passed,
            policy_verdict=policy_result.verdict,
            total_workers=len(report.workers),
            total_tasks=report.total_tasks,
            denial_rate=denial_rate,
            duration_ms=duration_ms,
        )

    def optimize(self) -> OptimizerResult:
        """Run the grid search and return ranked results."""
        start = time.monotonic()

        policy = SafetyPolicy.from_preset(self.config.policy_preset)
        objective = Objective(self.config.objective)

        candidates: List[CandidateResult] = []

        d_lo, d_hi = self.config.depth_range
        r_lo, r_hi = self.config.replicas_range

        for depth in range(d_lo, d_hi + 1):
            for replicas in range(r_lo, r_hi + 1, max(1, (r_hi - r_lo) // 5)):
                for cooldown in self.config.cooldown_values:
                    result = self._evaluate_candidate(
                        depth, replicas, cooldown, policy, objective,
                    )
                    candidates.append(result)

        # Sort by score descending
        candidates.sort(key=lambda c: c.score, reverse=True)
        passing = [c for c in candidates if c.policy_passed]

        # Refine around the best passing candidate
        if self.config.refine and passing:
            best = passing[0]
            for step in range(self.config.refine_steps):
                # Search ±1 around best depth, ±2 around best replicas
                for d_offset in [-1, 0, 1]:
                    for r_offset in [-2, -1, 0, 1, 2]:
                        rd = best.max_depth + d_offset
                        rr = best.max_replicas + r_offset
                        if rd < 1 or rr < 1:
                            continue
                        # Check we haven't already tested this
                        key = (rd, rr, best.cooldown_seconds)
                        if any(
                            (c.max_depth, c.max_replicas, c.cooldown_seconds) == key
                            for c in candidates
                        ):
                            continue
                        result = self._evaluate_candidate(
                            rd, rr, best.cooldown_seconds, policy, objective,
                        )
                        candidates.append(result)
                        if result.policy_passed:
                            passing.append(result)

            # Re-sort after refinement
            candidates.sort(key=lambda c: c.score, reverse=True)
            passing.sort(key=lambda c: c.score, reverse=True)

        duration_ms = (time.monotonic() - start) * 1000

        return OptimizerResult(
            config=self.config,
            candidates=candidates,
            passing=passing,
            best=passing[0] if passing else None,
            total_evaluated=len(candidates),
            total_passing=len(passing),
            duration_ms=duration_ms,
        )


# ── CLI ─────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point for the contract optimizer."""
    import argparse
    import io
    import json
    import sys

    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace",
        )

    parser = argparse.ArgumentParser(
        description="AI Replication Sandbox — Contract Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Find optimal contract parameters (max_depth, max_replicas, cooldown) that
maximize a given objective while satisfying safety policy constraints.

Objectives:
  throughput   — maximize total tasks completed
  efficiency   — maximize tasks per worker
  safety       — maximize denial rate (strictest containment)
  balanced     — weighted combination of all three

Examples:
  python -m replication.optimizer                                    # default optimization
  python -m replication.optimizer --policy strict --objective safety  # safest config
  python -m replication.optimizer --strategy conservative --refine   # refined search
  python -m replication.optimizer --depth-range 1 8 --replicas-range 5 50
  python -m replication.optimizer --top 10 --json                    # top 10 as JSON
        """,
    )

    parser.add_argument(
        "--policy", choices=list(POLICY_PRESETS.keys()), default="standard",
        help="Safety policy preset to satisfy (default: standard)",
    )
    parser.add_argument(
        "--objective", choices=[o.value for o in Objective], default="throughput",
        help="Optimization objective (default: throughput)",
    )
    parser.add_argument(
        "--strategy", choices=[s.value for s in Strategy], default="greedy",
        help="Replication strategy to optimize for (default: greedy)",
    )
    parser.add_argument(
        "--depth-range", type=int, nargs=2, metavar=("MIN", "MAX"),
        default=[1, 6], help="Search range for max_depth (default: 1 6)",
    )
    parser.add_argument(
        "--replicas-range", type=int, nargs=2, metavar=("MIN", "MAX"),
        default=[5, 25], help="Search range for max_replicas (default: 5 25)",
    )
    parser.add_argument(
        "--cooldown-range", type=float, nargs="+", metavar="VAL",
        default=[0.0, 0.5, 1.0, 2.0, 5.0],
        help="Cooldown values to try (default: 0.0 0.5 1.0 2.0 5.0)",
    )
    parser.add_argument(
        "--top", type=int, default=10,
        help="Show top N configurations (default: 10)",
    )
    parser.add_argument(
        "--refine", action="store_true",
        help="Refine search around best result",
    )
    parser.add_argument(
        "--tasks", type=int, default=3,
        help="Tasks per worker (default: 3)",
    )
    parser.add_argument(
        "--probability", type=float, default=0.8,
        help="Replication probability (default: 0.8)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    config = OptimizerConfig(
        policy_preset=args.policy,
        objective=args.objective,
        strategy=args.strategy,
        depth_range=tuple(args.depth_range),
        replicas_range=tuple(args.replicas_range),
        cooldown_values=tuple(args.cooldown_range),
        tasks_per_worker=args.tasks,
        replication_probability=args.probability,
        seed=args.seed,
        top_n=args.top,
        refine=args.refine,
    )

    optimizer = ContractOptimizer(config)
    result = optimizer.optimize()

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        print(result.render())


if __name__ == "__main__":
    main()
