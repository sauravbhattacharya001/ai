"""Capacity planner for AI replication scenarios.

Projects resource consumption over time for a given replication
contract and resource specification.  Answers questions like:

- *"If I allow depth-4 replication with 0.5 CPU / 256 MB per worker,
  how many workers could be alive simultaneously?"*
- *"When will I hit my cluster's 32 GB RAM ceiling?"*
- *"What's the peak CPU demand at each depth level?"*

The planner simulates discrete time steps, growing a replication tree
according to configurable strategies (greedy, conservative, chain, burst)
and tracking cumulative resource usage at each step.

Usage (Python API)::

    from replication.capacity import (
        CapacityPlanner, PlannerConfig, ResourceCeiling,
    )

    config = PlannerConfig(
        max_depth=4,
        max_replicas=50,
        cpu_per_worker=0.5,
        memory_mb_per_worker=256,
        strategy="greedy",
        ceiling=ResourceCeiling(max_cpu=16.0, max_memory_mb=32768),
    )

    planner = CapacityPlanner(config)
    projection = planner.project()

    print(projection.summary())
    print(f"Peak workers: {projection.peak_workers}")
    print(f"Peak CPU: {projection.peak_cpu}")
    print(f"Peak RAM: {projection.peak_memory_mb} MB")
    print(f"Ceiling hit at step: {projection.ceiling_hit_step}")

    # Export
    projection.to_json("capacity.json")

CLI::

    python -m replication.capacity
    python -m replication.capacity --max-depth 5 --max-replicas 100
    python -m replication.capacity --strategy burst --ceiling-cpu 8
    python -m replication.capacity --json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ── Configuration ───────────────────────────────────────────────────

@dataclass
class ResourceCeiling:
    """Hard resource limits for the hosting environment."""

    max_cpu: float = 0.0           # 0 = unlimited
    max_memory_mb: int = 0         # 0 = unlimited

    def has_cpu_limit(self) -> bool:
        return self.max_cpu > 0

    def has_memory_limit(self) -> bool:
        return self.max_memory_mb > 0


@dataclass
class PlannerConfig:
    """Configuration for capacity projection."""

    # Replication contract
    max_depth: int = 3
    max_replicas: int = 10
    cooldown_steps: int = 1

    # Per-worker resources
    cpu_per_worker: float = 0.5
    memory_mb_per_worker: int = 256

    # Worker lifetime (0 = workers never expire)
    lifetime_steps: int = 0

    # Strategy
    strategy: str = "greedy"

    # Environment ceiling
    ceiling: ResourceCeiling = field(default_factory=ResourceCeiling)

    # Simulation
    time_steps: int = 0            # 0 = auto-calculate
    replication_probability: float = 0.5  # for random strategy

    def __post_init__(self) -> None:
        if self.max_depth < 0:
            raise ValueError(f"max_depth must be >= 0, got {self.max_depth}")
        if self.max_replicas < 1:
            raise ValueError(
                f"max_replicas must be >= 1, got {self.max_replicas}"
            )
        if self.cpu_per_worker <= 0:
            raise ValueError(
                f"cpu_per_worker must be > 0, got {self.cpu_per_worker}"
            )
        if self.memory_mb_per_worker <= 0:
            raise ValueError(
                f"memory_mb_per_worker must be > 0, got "
                f"{self.memory_mb_per_worker}"
            )
        if self.strategy not in (
            "greedy", "conservative", "chain", "burst", "random"
        ):
            raise ValueError(f"Unknown strategy: {self.strategy}")


# ── Result containers ───────────────────────────────────────────────

@dataclass
class StepSnapshot:
    """Resource usage at a single time step."""

    step: int
    total_workers: int
    new_workers: int
    workers_by_depth: Dict[int, int]
    total_cpu: float
    total_memory_mb: int
    cpu_utilization: float         # 0.0-1.0 if ceiling set, else 0
    memory_utilization: float      # 0.0-1.0 if ceiling set, else 0
    ceiling_hit: bool


@dataclass
class CapacityProjection:
    """Full projection result."""

    config: PlannerConfig
    steps: List[StepSnapshot] = field(default_factory=list)
    peak_workers: int = 0
    peak_cpu: float = 0.0
    peak_memory_mb: int = 0
    total_workers_created: int = 0
    ceiling_hit_step: Optional[int] = None
    bottleneck: Optional[str] = None  # "cpu", "memory", "replicas", "depth"

    def summary(self) -> str:
        """Human-readable summary of the projection."""
        lines = [
            "Capacity Projection — AI Replication Sandbox",
            "=" * 46,
            "",
            "Configuration:",
            f"  Strategy:          {self.config.strategy}",
            f"  Max depth:         {self.config.max_depth}",
            f"  Max replicas:      {self.config.max_replicas}",
            f"  CPU / worker:      {self.config.cpu_per_worker}",
            f"  RAM / worker:      {self.config.memory_mb_per_worker} MB",
            f"  Cooldown steps:    {self.config.cooldown_steps}",
            f"  Worker lifetime:   {'∞' if self.config.lifetime_steps == 0 else str(self.config.lifetime_steps) + ' steps'}",
        ]

        if self.config.ceiling.has_cpu_limit():
            lines.append(
                f"  CPU ceiling:       {self.config.ceiling.max_cpu}"
            )
        if self.config.ceiling.has_memory_limit():
            lines.append(
                f"  Memory ceiling:    {self.config.ceiling.max_memory_mb} MB"
            )

        lines.extend([
            "",
            "Projection Results:",
            f"  Time steps:        {len(self.steps)}",
            f"  Peak workers:      {self.peak_workers}",
            f"  Peak CPU:          {self.peak_cpu:.1f} cores",
            f"  Peak memory:       {self.peak_memory_mb} MB "
            f"({self.peak_memory_mb / 1024:.1f} GB)",
            f"  Total created:     {self.total_workers_created}",
        ])

        if self.ceiling_hit_step is not None:
            lines.append(
                f"  ⚠ Ceiling hit:     step {self.ceiling_hit_step} "
                f"({self.bottleneck})"
            )
        else:
            lines.append("  Ceiling hit:       none")

        # Depth distribution at peak
        peak_step = max(self.steps, key=lambda s: s.total_workers)
        lines.extend([
            "",
            f"Worker distribution at peak (step {peak_step.step}):",
        ])
        for depth in sorted(peak_step.workers_by_depth.keys()):
            count = peak_step.workers_by_depth[depth]
            bar = "█" * min(count, 40)
            lines.append(f"  depth {depth}: {count:>4}  {bar}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to dict for JSON export."""
        return {
            "config": {
                "max_depth": self.config.max_depth,
                "max_replicas": self.config.max_replicas,
                "cooldown_steps": self.config.cooldown_steps,
                "lifetime_steps": self.config.lifetime_steps,
                "cpu_per_worker": self.config.cpu_per_worker,
                "memory_mb_per_worker": self.config.memory_mb_per_worker,
                "strategy": self.config.strategy,
                "ceiling": {
                    "max_cpu": self.config.ceiling.max_cpu,
                    "max_memory_mb": self.config.ceiling.max_memory_mb,
                },
            },
            "peak_workers": self.peak_workers,
            "peak_cpu": round(self.peak_cpu, 2),
            "peak_memory_mb": self.peak_memory_mb,
            "total_workers_created": self.total_workers_created,
            "ceiling_hit_step": self.ceiling_hit_step,
            "bottleneck": self.bottleneck,
            "steps": [
                {
                    "step": s.step,
                    "total_workers": s.total_workers,
                    "new_workers": s.new_workers,
                    "workers_by_depth": {
                        str(k): v
                        for k, v in sorted(s.workers_by_depth.items())
                    },
                    "total_cpu": round(s.total_cpu, 2),
                    "total_memory_mb": s.total_memory_mb,
                    "cpu_utilization": round(s.cpu_utilization, 4),
                    "memory_utilization": round(s.memory_utilization, 4),
                    "ceiling_hit": s.ceiling_hit,
                }
                for s in self.steps
            ],
        }

    def to_json(self, path: str) -> None:
        """Export projection to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


# ── Tree simulation node ───────────────────────────────────────────

@dataclass
class _SimNode:
    """Internal node tracking a simulated worker."""

    worker_id: int
    depth: int
    born_step: int
    children: List["_SimNode"] = field(default_factory=list)
    cooldown_until: int = 0  # step at which cooldown ends
    expires_at_step: int = -1  # step at which worker expires (-1 = never)


# ── Planner ─────────────────────────────────────────────────────────

class CapacityPlanner:
    """Project resource usage for replication scenarios.

    Simulates replication tree growth step-by-step, tracking how many
    workers are alive, their depth distribution, and cumulative resource
    demands.  Deterministic for reproducible projections.
    """

    def __init__(self, config: PlannerConfig) -> None:
        self.config = config
        self._next_id = 0

    def _new_id(self) -> int:
        wid = self._next_id
        self._next_id += 1
        return wid

    def project(self) -> CapacityProjection:
        """Run the capacity projection simulation.

        Returns a ``CapacityProjection`` with per-step snapshots
        and aggregate statistics.
        """
        cfg = self.config
        self._next_id = 0

        # Determine how many time steps to simulate
        if cfg.time_steps > 0:
            max_steps = cfg.time_steps
        else:
            # Auto: enough steps for the tree to fully grow + stabilize
            max_steps = (cfg.max_depth + 1) * (cfg.cooldown_steps + 1) + 2

        # Start with root worker
        root = _SimNode(
            worker_id=self._new_id(),
            depth=0,
            born_step=0,
            expires_at_step=(
                cfg.lifetime_steps if cfg.lifetime_steps > 0 else -1
            ),
        )
        all_nodes: List[_SimNode] = [root]

        projection = CapacityProjection(
            config=cfg,
            total_workers_created=1,
        )

        ceiling_hit = False

        for step in range(max_steps):
            # Determine which nodes can replicate this step
            new_nodes: List[_SimNode] = []

            # Filter to alive workers (not expired)
            alive_nodes = [
                n for n in all_nodes
                if n.expires_at_step < 0 or n.expires_at_step > step
            ]

            if not ceiling_hit:
                eligible = [
                    n for n in alive_nodes
                    if n.cooldown_until <= step
                ]

                for node in eligible:
                    children_to_spawn = self._children_for_strategy(
                        node, len(alive_nodes) + len(new_nodes), step
                    )

                    for _ in range(children_to_spawn):
                        total_after = (
                            len(alive_nodes) + len(new_nodes) + 1
                        )

                        # Check replica limit (against total ever created)
                        if projection.total_workers_created + 1 > cfg.max_replicas:
                            if projection.bottleneck is None:
                                projection.bottleneck = "replicas"
                                projection.ceiling_hit_step = step
                            ceiling_hit = True
                            break

                        # Check resource ceilings
                        projected_cpu = total_after * cfg.cpu_per_worker
                        projected_mem = (
                            total_after * cfg.memory_mb_per_worker
                        )

                        if (cfg.ceiling.has_cpu_limit()
                                and projected_cpu > cfg.ceiling.max_cpu):
                            if projection.bottleneck is None:
                                projection.bottleneck = "cpu"
                                projection.ceiling_hit_step = step
                            ceiling_hit = True
                            break

                        if (cfg.ceiling.has_memory_limit()
                                and projected_mem
                                > cfg.ceiling.max_memory_mb):
                            if projection.bottleneck is None:
                                projection.bottleneck = "memory"
                                projection.ceiling_hit_step = step
                            ceiling_hit = True
                            break

                        child = _SimNode(
                            worker_id=self._new_id(),
                            depth=node.depth + 1,
                            born_step=step,
                            cooldown_until=step + cfg.cooldown_steps,
                            expires_at_step=(
                                step + cfg.lifetime_steps
                                if cfg.lifetime_steps > 0 else -1
                            ),
                        )
                        node.children.append(child)
                        new_nodes.append(child)
                        node.cooldown_until = step + cfg.cooldown_steps
                        projection.total_workers_created += 1

                    if ceiling_hit:
                        break

            all_nodes.extend(new_nodes)

            # Recompute alive nodes after adding new ones
            alive_now = [
                n for n in all_nodes
                if n.expires_at_step < 0 or n.expires_at_step > step
            ]

            # Record snapshot based on concurrent (alive) workers
            total = len(alive_now)
            by_depth: Dict[int, int] = {}
            for n in alive_now:
                by_depth[n.depth] = by_depth.get(n.depth, 0) + 1

            total_cpu = total * cfg.cpu_per_worker
            total_mem = total * cfg.memory_mb_per_worker

            cpu_util = (
                total_cpu / cfg.ceiling.max_cpu
                if cfg.ceiling.has_cpu_limit() else 0.0
            )
            memory_util = (
                total_mem / cfg.ceiling.max_memory_mb
                if cfg.ceiling.has_memory_limit() else 0.0
            )

            snapshot = StepSnapshot(
                step=step,
                total_workers=total,
                new_workers=len(new_nodes),
                workers_by_depth=by_depth,
                total_cpu=total_cpu,
                total_memory_mb=total_mem,
                cpu_utilization=cpu_util,
                memory_utilization=memory_util,
                ceiling_hit=ceiling_hit and step == projection.ceiling_hit_step,
            )
            projection.steps.append(snapshot)

            # Update peaks
            if total > projection.peak_workers:
                projection.peak_workers = total
            if total_cpu > projection.peak_cpu:
                projection.peak_cpu = total_cpu
            if total_mem > projection.peak_memory_mb:
                projection.peak_memory_mb = total_mem

        return projection

    def _children_for_strategy(
        self, node: _SimNode, current_total: int, step: int
    ) -> int:
        """Decide how many children a node spawns this step."""
        cfg = self.config

        # Depth limit check
        if node.depth >= cfg.max_depth:
            return 0

        strategy = cfg.strategy

        if strategy == "greedy":
            # Spawn as many as allowed (up to remaining capacity)
            remaining = cfg.max_replicas - current_total
            # In greedy mode, each worker tries to spawn one child per step
            return min(1, remaining)

        elif strategy == "conservative":
            # Only replicate if below 50% of max depth
            if node.depth < cfg.max_depth / 2:
                return 1
            return 0

        elif strategy == "chain":
            # Only spawn if no children yet (linear chain)
            if len(node.children) == 0:
                return 1
            return 0

        elif strategy == "burst":
            # Only root (depth 0) spawns, and spawns many at once
            if node.depth == 0:
                remaining = cfg.max_replicas - current_total
                # Burst: try to fill as many as possible
                can_spawn = min(
                    cfg.max_replicas - 1,  # leave room for root
                    remaining,
                )
                # But respect max_depth (children would be depth 1)
                if 1 <= cfg.max_depth:
                    return max(0, can_spawn)
            return 0

        elif strategy == "random":
            # Deterministic pseudo-random based on step + node id
            # Use a simple hash to be reproducible without RNG state
            h = (step * 31 + node.worker_id * 17) % 100
            threshold = int(cfg.replication_probability * 100)
            if h < threshold:
                return 1
            return 0

        return 0


# ── Convenience functions ───────────────────────────────────────────

def quick_projection(
    max_depth: int = 3,
    max_replicas: int = 10,
    strategy: str = "greedy",
    cpu_per_worker: float = 0.5,
    memory_mb_per_worker: int = 256,
    ceiling_cpu: float = 0.0,
    ceiling_memory_mb: int = 0,
) -> CapacityProjection:
    """One-call capacity projection with sensible defaults."""
    config = PlannerConfig(
        max_depth=max_depth,
        max_replicas=max_replicas,
        cpu_per_worker=cpu_per_worker,
        memory_mb_per_worker=memory_mb_per_worker,
        strategy=strategy,
        ceiling=ResourceCeiling(
            max_cpu=ceiling_cpu,
            max_memory_mb=ceiling_memory_mb,
        ),
    )
    planner = CapacityPlanner(config)
    return planner.project()


def compare_strategies(
    max_depth: int = 3,
    max_replicas: int = 20,
    cpu_per_worker: float = 0.5,
    memory_mb_per_worker: int = 256,
    ceiling_cpu: float = 0.0,
    ceiling_memory_mb: int = 0,
) -> Dict[str, CapacityProjection]:
    """Run projections for all strategies and return a comparison dict."""
    strategies = ["greedy", "conservative", "chain", "burst"]
    results: Dict[str, CapacityProjection] = {}

    for strategy in strategies:
        config = PlannerConfig(
            max_depth=max_depth,
            max_replicas=max_replicas,
            cpu_per_worker=cpu_per_worker,
            memory_mb_per_worker=memory_mb_per_worker,
            strategy=strategy,
            ceiling=ResourceCeiling(
                max_cpu=ceiling_cpu,
                max_memory_mb=ceiling_memory_mb,
            ),
        )
        planner = CapacityPlanner(config)
        results[strategy] = planner.project()

    return results


def format_comparison(
    results: Dict[str, CapacityProjection],
) -> str:
    """Format a strategy comparison as a readable table."""
    lines = [
        "Strategy Comparison — Capacity Planning",
        "=" * 70,
        "",
        f"{'Strategy':<15} {'Peak Workers':>12} {'Peak CPU':>10} "
        f"{'Peak RAM':>12} {'Ceiling Hit':>12} {'Bottleneck':>12}",
        "─" * 70,
    ]

    for strategy, proj in sorted(results.items()):
        ceiling_str = (
            f"step {proj.ceiling_hit_step}"
            if proj.ceiling_hit_step is not None
            else "none"
        )
        bottleneck_str = proj.bottleneck or "—"
        lines.append(
            f"{strategy:<15} {proj.peak_workers:>12} "
            f"{proj.peak_cpu:>9.1f}x "
            f"{proj.peak_memory_mb:>9} MB "
            f"{ceiling_str:>12} {bottleneck_str:>12}"
        )

    return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point for capacity planning."""
    import argparse

    parser = argparse.ArgumentParser(
        description="AI Replication Capacity Planner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m replication.capacity\n"
            "  python -m replication.capacity --max-depth 5 "
            "--max-replicas 100\n"
            "  python -m replication.capacity --strategy burst "
            "--ceiling-cpu 8\n"
            "  python -m replication.capacity --compare\n"
            "  python -m replication.capacity --json -o plan.json\n"
        ),
    )

    parser.add_argument(
        "--max-depth", type=int, default=3,
        help="Maximum replication depth (default: 3)",
    )
    parser.add_argument(
        "--max-replicas", type=int, default=10,
        help="Maximum total replicas (default: 10)",
    )
    parser.add_argument(
        "--cooldown", type=int, default=1,
        help="Cooldown steps between replications (default: 1)",
    )
    parser.add_argument(
        "--lifetime", type=int, default=0,
        help="Worker lifetime in steps (0 = never expires, default: 0)",
    )
    parser.add_argument(
        "--cpu", type=float, default=0.5,
        help="CPU cores per worker (default: 0.5)",
    )
    parser.add_argument(
        "--memory", type=int, default=256,
        help="Memory MB per worker (default: 256)",
    )
    parser.add_argument(
        "--strategy", default="greedy",
        choices=["greedy", "conservative", "chain", "burst", "random"],
        help="Replication strategy (default: greedy)",
    )
    parser.add_argument(
        "--ceiling-cpu", type=float, default=0.0,
        help="Host CPU ceiling (0 = unlimited)",
    )
    parser.add_argument(
        "--ceiling-memory", type=int, default=0,
        help="Host memory ceiling in MB (0 = unlimited)",
    )
    parser.add_argument(
        "--steps", type=int, default=0,
        help="Number of time steps (0 = auto)",
    )
    parser.add_argument(
        "--replication-probability", type=float, default=0.5,
        help="Replication probability for 'random' strategy (default: 0.5)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare all strategies side by side",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output in JSON format",
    )
    parser.add_argument(
        "-o", "--output", metavar="FILE",
        help="Write output to file instead of stdout",
    )

    args = parser.parse_args()

    if args.compare:
        results = compare_strategies(
            max_depth=args.max_depth,
            max_replicas=args.max_replicas,
            cpu_per_worker=args.cpu,
            memory_mb_per_worker=args.memory,
            ceiling_cpu=args.ceiling_cpu,
            ceiling_memory_mb=args.ceiling_memory,
        )
        if args.json:
            output = json.dumps(
                {k: v.to_dict() for k, v in results.items()},
                indent=2,
            )
        else:
            output = format_comparison(results)
    else:
        config = PlannerConfig(
            max_depth=args.max_depth,
            max_replicas=args.max_replicas,
            cooldown_steps=args.cooldown,
            lifetime_steps=args.lifetime,
            cpu_per_worker=args.cpu,
            memory_mb_per_worker=args.memory,
            strategy=args.strategy,
            ceiling=ResourceCeiling(
                max_cpu=args.ceiling_cpu,
                max_memory_mb=args.ceiling_memory,
            ),
            time_steps=args.steps,
            replication_probability=args.replication_probability,
        )
        planner = CapacityPlanner(config)
        projection = planner.project()

        if args.json:
            output = json.dumps(projection.to_dict(), indent=2)
        else:
            output = projection.summary()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Output written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
