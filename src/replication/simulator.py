"""Simulation runner for AI replication experiments.

Provides a CLI and programmatic API for running configurable replication
scenarios and producing timeline reports with worker-tree visualization.

Usage (CLI)::

    python -m replication.simulator                        # default scenario
    python -m replication.simulator --strategy greedy      # aggressive replication
    python -m replication.simulator --max-depth 5 --max-replicas 20
    python -m replication.simulator --scenario balanced    # built-in preset
    python -m replication.simulator --json                 # JSON output

Programmatic::

    from replication.simulator import Simulator, ScenarioConfig
    config = ScenarioConfig(max_depth=3, max_replicas=10, strategy="conservative")
    sim = Simulator(config)
    report = sim.run()
    print(report.render_tree())
    print(report.render_timeline())
"""

from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Deque, Dict, List, Optional

from .contract import ReplicationContract, ResourceSpec
from .controller import Controller, ReplicationDenied
from .observability import StructuredLogger
from .orchestrator import SandboxOrchestrator
from .worker import Worker


class Strategy(Enum):
    """Replication strategy controlling how aggressively workers spawn."""

    GREEDY = "greedy"
    """Every worker tries to replicate on every task."""

    CONSERVATIVE = "conservative"
    """Workers only replicate when at less than 50% depth capacity."""

    RANDOM = "random"
    """Each worker has a configurable probability of replicating."""

    CHAIN = "chain"
    """Linear replication — each worker spawns exactly one child."""

    BURST = "burst"
    """Root worker spawns as many children as possible, no grandchildren."""


@dataclass
class ScenarioConfig:
    """Configuration for a simulation run."""

    # Contract parameters
    max_depth: int = 3
    max_replicas: int = 10
    cooldown_seconds: float = 0.0
    expiration_seconds: Optional[float] = None

    # Simulation parameters
    strategy: str = "greedy"
    tasks_per_worker: int = 2
    replication_probability: float = 0.5  # for random strategy
    secret: str = "simulation-secret"
    seed: Optional[int] = None

    # Resource spec
    cpu_limit: float = 0.5
    memory_limit_mb: int = 256

    @property
    def strategy_enum(self) -> Strategy:
        return Strategy(self.strategy)


# Built-in scenario presets
PRESETS: Dict[str, ScenarioConfig] = {
    "minimal": ScenarioConfig(
        max_depth=1, max_replicas=3, strategy="conservative", tasks_per_worker=1,
    ),
    "balanced": ScenarioConfig(
        max_depth=3, max_replicas=10, strategy="random", tasks_per_worker=2,
        replication_probability=0.6,
    ),
    "stress": ScenarioConfig(
        max_depth=5, max_replicas=50, strategy="greedy", tasks_per_worker=3,
        cooldown_seconds=0.0,
    ),
    "chain": ScenarioConfig(
        max_depth=8, max_replicas=10, strategy="chain", tasks_per_worker=1,
    ),
    "burst": ScenarioConfig(
        max_depth=1, max_replicas=20, strategy="burst", tasks_per_worker=1,
    ),
}


@dataclass
class WorkerRecord:
    """Record of a worker's lifecycle during simulation."""

    worker_id: str
    parent_id: Optional[str]
    depth: int
    tasks_completed: int = 0
    replications_attempted: int = 0
    replications_succeeded: int = 0
    replications_denied: int = 0
    children: List[str] = field(default_factory=list)
    created_at: Optional[float] = None
    shutdown_at: Optional[float] = None
    shutdown_reason: str = ""


@dataclass
class SimulationReport:
    """Results of a simulation run with rendering capabilities."""

    config: ScenarioConfig
    workers: Dict[str, WorkerRecord]
    root_id: str
    timeline: List[Dict[str, Any]]
    total_tasks: int
    total_replications_attempted: int
    total_replications_succeeded: int
    total_replications_denied: int
    duration_ms: float
    audit_events: List[Dict[str, Any]]

    def render_tree(self) -> str:
        """Render the worker lineage as an ASCII tree."""
        lines: List[str] = []
        lines.append("┌─────────────────────────────────────┐")
        lines.append("│         Worker Lineage Tree         │")
        lines.append("└─────────────────────────────────────┘")
        lines.append("")

        def _draw(wid: str, prefix: str = "", is_last: bool = True) -> None:
            rec = self.workers.get(wid)
            if not rec:
                return
            connector = "└── " if is_last else "├── "
            task_info = f"tasks={rec.tasks_completed}"
            repl_info = f"repl={rec.replications_succeeded}/{rec.replications_attempted}"
            label = f"[{wid}] depth={rec.depth} {task_info} {repl_info}"
            lines.append(f"{prefix}{connector}{label}")
            child_prefix = prefix + ("    " if is_last else "│   ")
            for i, child_id in enumerate(rec.children):
                _draw(child_id, child_prefix, i == len(rec.children) - 1)

        root_rec = self.workers[self.root_id]
        lines.append(f"[{self.root_id}] depth=0 (root)")
        for i, child_id in enumerate(root_rec.children):
            _draw(child_id, "", i == len(root_rec.children) - 1)

        return "\n".join(lines)

    def render_timeline(self) -> str:
        """Render a chronological timeline of simulation events."""
        lines: List[str] = []
        lines.append("┌─────────────────────────────────────┐")
        lines.append("│        Simulation Timeline          │")
        lines.append("└─────────────────────────────────────┘")
        lines.append("")

        icons = {
            "spawn": "🟢",
            "task": "⚡",
            "replicate_ok": "🔀",
            "replicate_denied": "🚫",
            "shutdown": "🔴",
        }

        for entry in self.timeline:
            icon = icons.get(entry["type"], "•")
            ts = f"{entry['time_ms']:>8.1f}ms"
            wid = entry.get("worker_id", "?")[:8]
            detail = entry.get("detail", "")
            lines.append(f"  {ts}  {icon}  [{wid}]  {detail}")

        return "\n".join(lines)

    def render_summary(self) -> str:
        """Render summary statistics."""
        lines: List[str] = []
        lines.append("┌─────────────────────────────────────┐")
        lines.append("│       Simulation Summary            │")
        lines.append("└─────────────────────────────────────┘")
        lines.append("")
        lines.append(f"  Strategy:            {self.config.strategy}")
        lines.append(f"  Max Depth:           {self.config.max_depth}")
        lines.append(f"  Max Replicas:        {self.config.max_replicas}")
        lines.append(f"  Cooldown:            {self.config.cooldown_seconds}s")
        lines.append(f"  Tasks/Worker:        {self.config.tasks_per_worker}")
        lines.append("")
        lines.append(f"  Total Workers:       {len(self.workers)}")
        lines.append(f"  Total Tasks:         {self.total_tasks}")
        lines.append(f"  Replications OK:     {self.total_replications_succeeded}")
        lines.append(f"  Replications Denied: {self.total_replications_denied}")

        max_depth_seen = max((w.depth for w in self.workers.values()), default=0)
        lines.append(f"  Max Depth Reached:   {max_depth_seen}")

        n_workers = len(self.workers)
        total_cpu = self.config.cpu_limit * n_workers
        total_mem = self.config.memory_limit_mb * n_workers
        lines.append(f"  Total CPU Allocated: {total_cpu:.2f} cores")
        lines.append(f"  Total RAM Allocated: {total_mem} MB")
        lines.append(f"  Duration:            {self.duration_ms:.1f}ms")

        # Depth distribution
        depth_counts: Dict[int, int] = {}
        for w in self.workers.values():
            depth_counts[w.depth] = depth_counts.get(w.depth, 0) + 1
        lines.append("")
        lines.append("  Depth Distribution:")
        for d in sorted(depth_counts.keys()):
            bar = "█" * depth_counts[d]
            lines.append(f"    depth {d}: {bar} ({depth_counts[d]})")

        # Denial reasons from audit
        denial_reasons: Dict[str, int] = {}
        for evt in self.audit_events:
            decision = evt.get("decision", "")
            if decision.startswith("deny_"):
                denial_reasons[decision] = denial_reasons.get(decision, 0) + 1
        if denial_reasons:
            lines.append("")
            lines.append("  Denial Breakdown:")
            for reason, count in sorted(denial_reasons.items(), key=lambda x: -x[1]):
                lines.append(f"    {reason}: {count}")

        return "\n".join(lines)

    def render(self) -> str:
        """Render the full report (summary + tree + timeline)."""
        return "\n\n".join([
            self.render_summary(),
            self.render_tree(),
            self.render_timeline(),
        ])

    def to_dict(self) -> Dict[str, Any]:
        """Export report as a JSON-serializable dictionary."""
        return {
            "config": {
                "strategy": self.config.strategy,
                "max_depth": self.config.max_depth,
                "max_replicas": self.config.max_replicas,
                "cooldown_seconds": self.config.cooldown_seconds,
                "tasks_per_worker": self.config.tasks_per_worker,
                "cpu_limit": self.config.cpu_limit,
                "memory_limit_mb": self.config.memory_limit_mb,
            },
            "results": {
                "total_workers": len(self.workers),
                "total_tasks": self.total_tasks,
                "replications_succeeded": self.total_replications_succeeded,
                "replications_denied": self.total_replications_denied,
                "max_depth_reached": max((w.depth for w in self.workers.values()), default=0),
                "duration_ms": self.duration_ms,
            },
            "workers": {
                wid: {
                    "parent_id": rec.parent_id,
                    "depth": rec.depth,
                    "tasks_completed": rec.tasks_completed,
                    "replications_attempted": rec.replications_attempted,
                    "replications_succeeded": rec.replications_succeeded,
                    "replications_denied": rec.replications_denied,
                    "children": rec.children,
                }
                for wid, rec in self.workers.items()
            },
            "timeline": self.timeline,
        }


class Simulator:
    """Runs a replication scenario and produces a SimulationReport."""

    def __init__(self, config: Optional[ScenarioConfig] = None) -> None:
        self.config = config or ScenarioConfig()
        self._rng = random.Random(self.config.seed)

    def run(self) -> SimulationReport:
        """Execute the simulation and return the report."""
        contract = ReplicationContract(
            max_depth=self.config.max_depth,
            max_replicas=self.config.max_replicas,
            cooldown_seconds=self.config.cooldown_seconds,
            expiration_seconds=self.config.expiration_seconds,
        )
        logger = StructuredLogger()
        controller = Controller(contract=contract, secret=self.config.secret, logger=logger)
        orchestrator = SandboxOrchestrator(logger=logger)
        resources = ResourceSpec(
            cpu_limit=self.config.cpu_limit,
            memory_limit_mb=self.config.memory_limit_mb,
        )

        records: Dict[str, WorkerRecord] = {}
        timeline: List[Dict[str, Any]] = []
        start_time = time.monotonic()
        strategy = self.config.strategy_enum

        total_tasks = 0
        total_repl_attempted = 0
        total_repl_ok = 0
        total_repl_denied = 0

        def _elapsed_ms() -> float:
            return (time.monotonic() - start_time) * 1000

        # Create root worker
        root_manifest = controller.issue_manifest(
            parent_id=None, depth=0,
            state_snapshot={"role": "root", "strategy": self.config.strategy},
            resources=resources,
        )
        root = Worker(root_manifest, contract, controller, orchestrator, logger)
        root_id = root.manifest.worker_id

        records[root_id] = WorkerRecord(
            worker_id=root_id, parent_id=None, depth=0, created_at=_elapsed_ms(),
        )
        timeline.append({
            "type": "spawn", "time_ms": _elapsed_ms(),
            "worker_id": root_id, "detail": "Root worker created (depth 0)",
        })

        # BFS work queue — deque gives O(1) popleft vs O(n) list.pop(0)
        queue: Deque[Worker] = deque([root])

        while queue:
            worker = queue.popleft()
            wid = worker.manifest.worker_id
            rec = records[wid]

            early_shutdown = False
            for task_num in range(self.config.tasks_per_worker):
                # Perform task
                try:
                    worker.perform_task(lambda w: None)
                    rec.tasks_completed += 1
                    total_tasks += 1
                    timeline.append({
                        "type": "task", "time_ms": _elapsed_ms(),
                        "worker_id": wid,
                        "detail": f"Task {task_num + 1}/{self.config.tasks_per_worker} completed",
                    })
                except ReplicationDenied:
                    worker.shutdown("expired_or_killed")
                    rec.shutdown_at = _elapsed_ms()
                    rec.shutdown_reason = "expired_or_killed"
                    timeline.append({
                        "type": "shutdown", "time_ms": _elapsed_ms(),
                        "worker_id": wid, "detail": "Worker expired or killed",
                    })
                    early_shutdown = True
                    break

                # Decide whether to replicate based on strategy
                should_replicate = self._should_replicate(strategy, worker, contract)

                if should_replicate:
                    rec.replications_attempted += 1
                    total_repl_attempted += 1

                    child_state = {
                        "role": "child",
                        "parent": wid,
                        "task": str(task_num),
                    }
                    child = worker.maybe_replicate(
                        reason=f"strategy_{strategy.value}",
                        state_snapshot=child_state,
                    )

                    if child:
                        rec.replications_succeeded += 1
                        total_repl_ok += 1
                        rec.children.append(child.manifest.worker_id)

                        child_rec = WorkerRecord(
                            worker_id=child.manifest.worker_id,
                            parent_id=wid,
                            depth=child.manifest.depth,
                            created_at=_elapsed_ms(),
                        )
                        records[child.manifest.worker_id] = child_rec

                        timeline.append({
                            "type": "replicate_ok", "time_ms": _elapsed_ms(),
                            "worker_id": wid,
                            "detail": f"Spawned child [{child.manifest.worker_id}] at depth {child.manifest.depth}",
                        })

                        queue.append(child)
                    else:
                        rec.replications_denied += 1
                        total_repl_denied += 1
                        timeline.append({
                            "type": "replicate_denied", "time_ms": _elapsed_ms(),
                            "worker_id": wid, "detail": "Replication denied by contract",
                        })

            # Shutdown worker after tasks (skip if already shut down early)
            if not early_shutdown:
                worker.shutdown("tasks_complete")
                rec.shutdown_at = _elapsed_ms()
                rec.shutdown_reason = "tasks_complete"
                timeline.append({
                    "type": "shutdown", "time_ms": _elapsed_ms(),
                    "worker_id": wid, "detail": "Shutdown (tasks complete)",
                })

        duration_ms = _elapsed_ms()

        audit_events = [e for e in logger.events if e.get("event") == "audit"]

        return SimulationReport(
            config=self.config,
            workers=records,
            root_id=root_id,
            timeline=timeline,
            total_tasks=total_tasks,
            total_replications_attempted=total_repl_attempted,
            total_replications_succeeded=total_repl_ok,
            total_replications_denied=total_repl_denied,
            duration_ms=duration_ms,
            audit_events=audit_events,
        )

    def _should_replicate(self, strategy: Strategy, worker: Worker, contract: ReplicationContract) -> bool:
        """Determine if a worker should attempt replication under the given strategy."""
        depth = worker.manifest.depth

        if strategy == Strategy.GREEDY:
            return True

        elif strategy == Strategy.CONSERVATIVE:
            return depth < contract.max_depth // 2

        elif strategy == Strategy.RANDOM:
            return self._rng.random() < self.config.replication_probability

        elif strategy == Strategy.CHAIN:
            return True  # each spawns one, depth limit enforces the chain

        elif strategy == Strategy.BURST:
            return depth == 0  # only root spawns children

        return False


def main() -> None:
    """CLI entry point for the simulation runner."""
    import argparse
    import json
    import sys
    import io

    # Ensure UTF-8 output on Windows
    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="AI Replication Sandbox — Simulation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Built-in scenarios: minimal, balanced, stress, chain, burst

Examples:
  python -m replication.simulator                         # default greedy scenario
  python -m replication.simulator --scenario balanced     # use balanced preset
  python -m replication.simulator --strategy chain --max-depth 6
  python -m replication.simulator --json                  # output as JSON
  python -m replication.simulator --seed 42               # reproducible run
        """,
    )

    parser.add_argument(
        "--scenario", choices=list(PRESETS.keys()),
        help="Use a built-in scenario preset",
    )
    parser.add_argument("--strategy", choices=[s.value for s in Strategy], help="Replication strategy")
    parser.add_argument("--max-depth", type=int, help="Maximum replication depth")
    parser.add_argument("--max-replicas", type=int, help="Maximum concurrent replicas")
    parser.add_argument("--cooldown", type=float, help="Cooldown seconds between spawns")
    parser.add_argument("--tasks", type=int, help="Tasks per worker")
    parser.add_argument("--probability", type=float, help="Replication probability (random strategy)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--cpu", type=float, help="CPU limit per worker")
    parser.add_argument("--memory", type=int, help="Memory limit (MB) per worker")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--tree-only", action="store_true", help="Show only the worker tree")
    parser.add_argument("--summary-only", action="store_true", help="Show only the summary")

    args = parser.parse_args()

    # Start from preset or defaults
    if args.scenario:
        config = PRESETS[args.scenario]
    else:
        config = ScenarioConfig()

    # Override with CLI args
    if args.strategy:
        config.strategy = args.strategy
    if args.max_depth is not None:
        config.max_depth = args.max_depth
    if args.max_replicas is not None:
        config.max_replicas = args.max_replicas
    if args.cooldown is not None:
        config.cooldown_seconds = args.cooldown
    if args.tasks is not None:
        config.tasks_per_worker = args.tasks
    if args.probability is not None:
        config.replication_probability = args.probability
    if args.seed is not None:
        config.seed = args.seed
    if args.cpu is not None:
        config.cpu_limit = args.cpu
    if args.memory is not None:
        config.memory_limit_mb = args.memory

    sim = Simulator(config)
    report = sim.run()

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    elif args.tree_only:
        print(report.render_tree())
    elif args.summary_only:
        print(report.render_summary())
    else:
        print(report.render())


if __name__ == "__main__":
    main()
