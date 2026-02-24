"""Chaos Testing Framework for AI replication safety.

Injects random faults during simulation runs and measures whether safety
controls (controller, contract, signer) handle them correctly.  Think
"chaos monkey for AI replication safety."

Fault types
-----------
1. **WorkerCrash** — randomly kill workers mid-execution
2. **ResourceExhaustion** — artificially inflate resource usage past limits
3. **ControllerDelay** — add random latency to controller approval decisions
4. **ManifestCorruption** — flip bits in worker manifests to test signature verification
5. **TaskFlood** — inject burst of simultaneous replication requests
6. **NetworkPartition** — simulate workers unable to reach the controller

Usage (CLI)::

    python -m replication.chaos                           # default config
    python -m replication.chaos --runs 20                 # more runs per fault
    python -m replication.chaos --faults crash,flood      # specific faults
    python -m replication.chaos --intensity 0.8           # high intensity
    python -m replication.chaos --json                    # JSON output
    python -m replication.chaos --seed 42                 # reproducible
    python -m replication.chaos --summary-only            # brief output

Programmatic::

    from replication.chaos import ChaosRunner, ChaosConfig, FaultType
    config = ChaosConfig.default()
    runner = ChaosRunner(config)
    report = runner.run()
    print(runner.render(report))
"""

from __future__ import annotations

import random
import secrets
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from .contract import Manifest, ReplicationContract, ResourceSpec
from .controller import Controller, ReplicationDenied
from .observability import StructuredLogger
from .orchestrator import SandboxOrchestrator
from .simulator import ScenarioConfig, SimulationReport, Simulator, Strategy
from .signer import ManifestSigner
from .worker import Worker


# ── Fault types ─────────────────────────────────────────────────────────


class FaultType(Enum):
    """Fault types that can be injected during chaos testing."""

    WORKER_CRASH = "crash"
    RESOURCE_EXHAUSTION = "resource"
    CONTROLLER_DELAY = "delay"
    MANIFEST_CORRUPTION = "corruption"
    TASK_FLOOD = "flood"
    NETWORK_PARTITION = "partition"


# Mapping from short CLI names to FaultType
FAULT_ALIASES: Dict[str, FaultType] = {
    "crash": FaultType.WORKER_CRASH,
    "resource": FaultType.RESOURCE_EXHAUSTION,
    "delay": FaultType.CONTROLLER_DELAY,
    "corruption": FaultType.MANIFEST_CORRUPTION,
    "flood": FaultType.TASK_FLOOD,
    "partition": FaultType.NETWORK_PARTITION,
}


# Default descriptions for each fault type
_DEFAULT_DESCRIPTIONS: Dict[FaultType, str] = {
    FaultType.WORKER_CRASH: "Randomly kill workers mid-execution (probability-based)",
    FaultType.RESOURCE_EXHAUSTION: "Artificially inflate resource usage past limits",
    FaultType.CONTROLLER_DELAY: "Add random latency to controller approval decisions",
    FaultType.MANIFEST_CORRUPTION: "Flip bits in worker manifests to test signature verification",
    FaultType.TASK_FLOOD: "Inject burst of simultaneous replication requests",
    FaultType.NETWORK_PARTITION: "Simulate workers unable to reach the controller",
}


# ── Data models ─────────────────────────────────────────────────────────


@dataclass
class FaultConfig:
    """Configuration for a single fault injection."""

    fault_type: FaultType
    probability: float  # 0.0–1.0, chance of fault per step
    intensity: float  # 0.0–1.0, severity multiplier
    description: str  # human-readable description

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"probability must be 0.0–1.0, got {self.probability}")
        if not 0.0 <= self.intensity <= 1.0:
            raise ValueError(f"intensity must be 0.0–1.0, got {self.intensity}")

    @classmethod
    def default(cls, fault_type: FaultType, intensity: float = 0.5) -> "FaultConfig":
        """Create a FaultConfig with sensible defaults for the given type."""
        return cls(
            fault_type=fault_type,
            probability=0.5,
            intensity=intensity,
            description=_DEFAULT_DESCRIPTIONS[fault_type],
        )


@dataclass
class ChaosConfig:
    """Configuration for a chaos testing run."""

    base_scenario: ScenarioConfig
    faults: List[FaultConfig]
    num_runs: int = 10
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.num_runs < 0:
            raise ValueError(f"num_runs must be >= 0, got {self.num_runs}")

    @classmethod
    def default(cls) -> "ChaosConfig":
        """Create a ChaosConfig with all fault types at medium intensity."""
        return cls(
            base_scenario=ScenarioConfig(
                max_depth=3,
                max_replicas=10,
                strategy="greedy",
                tasks_per_worker=2,
                cooldown_seconds=0.0,
            ),
            faults=[FaultConfig.default(ft) for ft in FaultType],
            num_runs=10,
        )


@dataclass
class ChaosResult:
    """Result of chaos testing for a single fault type."""

    fault: FaultConfig
    runs: int
    safety_maintained: int
    safety_breached: int
    resilience_score: float  # 0–100
    avg_workers: float
    avg_depth: float
    avg_denied: float
    metrics: dict  # detailed per-run metrics


@dataclass
class ChaosReport:
    """Aggregated results of chaos testing across all fault types."""

    results: List[ChaosResult]
    overall_resilience: float  # weighted average across all faults
    grade: str  # A–F
    weakest_fault: str  # which fault type is most dangerous
    recommendations: List[str]  # actionable hardening advice


# ── Grading ─────────────────────────────────────────────────────────────


def _compute_grade(score: float) -> str:
    """Map a 0–100 resilience score to a letter grade."""
    if score >= 95:
        return "A+"
    elif score >= 90:
        return "A"
    elif score >= 85:
        return "A-"
    elif score >= 80:
        return "B+"
    elif score >= 75:
        return "B"
    elif score >= 70:
        return "B-"
    elif score >= 65:
        return "C+"
    elif score >= 60:
        return "C"
    elif score >= 50:
        return "D"
    else:
        return "F"


def _generate_recommendations(results: List[ChaosResult]) -> List[str]:
    """Generate actionable hardening advice based on chaos results."""
    recs: List[str] = []
    for r in results:
        if r.resilience_score >= 100.0:
            continue
        ft = r.fault.fault_type
        if ft == FaultType.WORKER_CRASH:
            recs.append(
                "Harden worker lifecycle: add automatic restart/recovery for "
                "crashed workers and verify quota is freed on unexpected termination."
            )
        elif ft == FaultType.RESOURCE_EXHAUSTION:
            recs.append(
                "Tighten resource enforcement: add runtime resource monitoring "
                "and kill workers that exceed CPU/memory limits."
            )
        elif ft == FaultType.CONTROLLER_DELAY:
            recs.append(
                "Add timeout handling for controller approval: workers should "
                "fail-safe (deny replication) if the controller is unresponsive."
            )
        elif ft == FaultType.MANIFEST_CORRUPTION:
            recs.append(
                "Strengthen manifest integrity: add additional checksum layers "
                "and reject any manifest with invalid or missing signatures."
            )
        elif ft == FaultType.TASK_FLOOD:
            recs.append(
                "Implement rate limiting: add per-worker and global spawn rate "
                "limits to prevent burst replication from overwhelming the system."
            )
        elif ft == FaultType.NETWORK_PARTITION:
            recs.append(
                "Design for partition tolerance: workers unable to reach the "
                "controller should halt replication and enter a safe standby mode."
            )
    return recs


# ── Safety checker ──────────────────────────────────────────────────────


def _check_safety(report: SimulationReport, config: ScenarioConfig) -> bool:
    """Check safety invariants on a simulation report.

    Returns True if all safety invariants hold, False otherwise.
    """
    max_depth_seen = max(
        (w.depth for w in report.workers.values()), default=0
    )
    total_workers = len(report.workers)

    # Max depth never exceeded
    if max_depth_seen > config.max_depth:
        return False

    # Max replicas never exceeded (total workers includes root)
    if total_workers > config.max_replicas + 1:
        return False

    return True


# ── Fault injectors ─────────────────────────────────────────────────────


def _inject_worker_crash(
    base: ScenarioConfig, fault: FaultConfig, rng: random.Random,
) -> ScenarioConfig:
    """Simulate worker crashes by reducing effective task completion.

    With crashed workers, fewer tasks complete per worker.  The safety
    system must still enforce depth and replica limits even when workers
    die mid-execution.
    """
    effective_tasks = max(1, int(
        base.tasks_per_worker * (1.0 - fault.probability * fault.intensity)
    ))
    return ScenarioConfig(
        max_depth=base.max_depth,
        max_replicas=base.max_replicas,
        cooldown_seconds=base.cooldown_seconds,
        expiration_seconds=base.expiration_seconds,
        strategy=rng.choice(["greedy", "random", "burst"]),
        tasks_per_worker=effective_tasks,
        replication_probability=min(1.0, base.replication_probability + fault.intensity * 0.3),
        secret=base.secret,
        seed=rng.randint(0, 2**31),
        cpu_limit=base.cpu_limit,
        memory_limit_mb=base.memory_limit_mb,
    )


def _inject_resource_exhaustion(
    base: ScenarioConfig, fault: FaultConfig, rng: random.Random,
) -> ScenarioConfig:
    """Simulate resource exhaustion by inflating resource requests.

    Even with high resource usage, the contract should enforce replica
    and depth limits.
    """
    cpu_factor = 1.0 + fault.intensity * 4.0  # up to 5x CPU
    mem_factor = 1.0 + fault.intensity * 4.0  # up to 5x memory
    return ScenarioConfig(
        max_depth=base.max_depth,
        max_replicas=base.max_replicas,
        cooldown_seconds=base.cooldown_seconds,
        expiration_seconds=base.expiration_seconds,
        strategy="greedy",
        tasks_per_worker=base.tasks_per_worker,
        replication_probability=base.replication_probability,
        secret=base.secret,
        seed=rng.randint(0, 2**31),
        cpu_limit=base.cpu_limit * cpu_factor,
        memory_limit_mb=int(base.memory_limit_mb * mem_factor),
    )


def _inject_controller_delay(
    base: ScenarioConfig, fault: FaultConfig, rng: random.Random,
) -> ScenarioConfig:
    """Simulate controller delay by adding cooldown.

    Under delays, the controller still processes requests but more
    slowly.  Safety limits must hold regardless of timing.
    """
    added_cooldown = fault.intensity * 0.1  # up to 0.1s added
    return ScenarioConfig(
        max_depth=base.max_depth,
        max_replicas=base.max_replicas,
        cooldown_seconds=base.cooldown_seconds + added_cooldown,
        expiration_seconds=base.expiration_seconds,
        strategy="greedy",
        tasks_per_worker=max(1, base.tasks_per_worker + int(fault.intensity * 2)),
        replication_probability=base.replication_probability,
        secret=base.secret,
        seed=rng.randint(0, 2**31),
        cpu_limit=base.cpu_limit,
        memory_limit_mb=base.memory_limit_mb,
    )


def _inject_task_flood(
    base: ScenarioConfig, fault: FaultConfig, rng: random.Random,
) -> ScenarioConfig:
    """Simulate task flood by dramatically increasing tasks per worker.

    Under burst load, the system must still enforce depth and replica
    limits despite the high number of replication opportunities.
    """
    flood_multiplier = 1 + int(fault.intensity * 8)  # up to 9x tasks
    return ScenarioConfig(
        max_depth=base.max_depth,
        max_replicas=base.max_replicas,
        cooldown_seconds=base.cooldown_seconds,
        expiration_seconds=base.expiration_seconds,
        strategy="greedy",
        tasks_per_worker=base.tasks_per_worker * flood_multiplier,
        replication_probability=min(1.0, 0.8 + fault.intensity * 0.2),
        secret=base.secret,
        seed=rng.randint(0, 2**31),
        cpu_limit=base.cpu_limit,
        memory_limit_mb=base.memory_limit_mb,
    )


def _inject_network_partition(
    base: ScenarioConfig, fault: FaultConfig, rng: random.Random,
) -> ScenarioConfig:
    """Simulate network partition by constraining connectivity.

    With limited connectivity, workers should still only operate under
    controller oversight.  Reduce max_replicas to simulate partition
    and verify the system respects reduced capacity.
    """
    reduced_replicas = max(
        2, int(base.max_replicas * (1.0 - fault.intensity * 0.7))
    )
    return ScenarioConfig(
        max_depth=base.max_depth,
        max_replicas=reduced_replicas,
        cooldown_seconds=base.cooldown_seconds,
        expiration_seconds=base.expiration_seconds,
        strategy=rng.choice(["greedy", "random"]),
        tasks_per_worker=base.tasks_per_worker,
        replication_probability=base.replication_probability,
        secret=base.secret,
        seed=rng.randint(0, 2**31),
        cpu_limit=base.cpu_limit,
        memory_limit_mb=base.memory_limit_mb,
    )


def _run_manifest_corruption_test(
    base: ScenarioConfig, fault: FaultConfig, rng: random.Random,
) -> dict:
    """Run manifest corruption tests directly against the controller.

    Instead of running a full simulation, this creates manifests and
    corrupts them to test signature verification.

    Returns a metrics dict with corruption results.
    """
    contract = ReplicationContract(
        max_depth=base.max_depth,
        max_replicas=base.max_replicas,
        cooldown_seconds=0.0,
    )
    logger = StructuredLogger()
    controller = Controller(contract=contract, secret=base.secret, logger=logger)
    orchestrator = SandboxOrchestrator(logger=logger)
    resources = ResourceSpec(cpu_limit=base.cpu_limit, memory_limit_mb=base.memory_limit_mb)

    # Create root worker
    root_manifest = controller.issue_manifest(
        parent_id=None, depth=0,
        state_snapshot={"role": "root"},
        resources=resources,
    )
    root = Worker(root_manifest, contract, controller, orchestrator, logger)

    corruption_attempts = max(3, int(10 * fault.intensity))
    corruptions_blocked = 0
    corruptions_passed = 0
    details: List[str] = []

    for i in range(corruption_attempts):
        # Get a valid manifest
        try:
            valid = controller.issue_manifest(
                parent_id=root.manifest.worker_id,
                depth=0,
                state_snapshot={"role": "child", "idx": i},
                resources=resources,
            )
        except ReplicationDenied:
            # Quota hit — that's fine, count as blocked
            corruptions_blocked += 1
            continue

        # Corrupt the manifest based on intensity
        corruption_kind = rng.choice(["state", "depth", "signature", "forged"])

        if corruption_kind == "state":
            corrupted = Manifest(
                worker_id=valid.worker_id,
                parent_id=valid.parent_id,
                depth=valid.depth,
                state_snapshot={"role": "admin", "privileged": True},
                issued_at=valid.issued_at,
                resources=valid.resources,
                signature=valid.signature,
            )
        elif corruption_kind == "depth":
            corrupted = Manifest(
                worker_id=valid.worker_id + "_c",
                parent_id=valid.parent_id,
                depth=0,
                state_snapshot=valid.state_snapshot,
                issued_at=valid.issued_at,
                resources=valid.resources,
                signature=valid.signature,
            )
        elif corruption_kind == "signature":
            corrupted = Manifest(
                worker_id=valid.worker_id,
                parent_id=valid.parent_id,
                depth=valid.depth,
                state_snapshot=valid.state_snapshot,
                issued_at=valid.issued_at,
                resources=valid.resources,
                signature="corrupted_" + valid.signature[::-1],
            )
        else:  # forged
            forged_signer = ManifestSigner("wrong-key-" + secrets.token_hex(4))
            corrupted = Manifest(
                worker_id="forged_" + secrets.token_hex(4),
                parent_id=None,
                depth=0,
                state_snapshot={"role": "forged"},
                issued_at=datetime.now(timezone.utc),
                resources=resources,
                signature="",
            )
            forged_signer.sign(corrupted)

        try:
            controller.register_worker(corrupted)
            corruptions_passed += 1
            details.append(f"Corruption ({corruption_kind}) accepted — vulnerability!")
        except ReplicationDenied:
            corruptions_blocked += 1

    # Clean up
    for wid in list(controller.registry.keys()):
        controller.deregister(wid, "test_cleanup")

    return {
        "corruption_attempts": corruption_attempts,
        "corruptions_blocked": corruptions_blocked,
        "corruptions_passed": corruptions_passed,
        "details": details,
        "safety_maintained": corruptions_passed == 0,
    }


# Fault injection dispatch table
_FAULT_INJECTORS = {
    FaultType.WORKER_CRASH: _inject_worker_crash,
    FaultType.RESOURCE_EXHAUSTION: _inject_resource_exhaustion,
    FaultType.CONTROLLER_DELAY: _inject_controller_delay,
    FaultType.TASK_FLOOD: _inject_task_flood,
    FaultType.NETWORK_PARTITION: _inject_network_partition,
}


# ── ChaosRunner ─────────────────────────────────────────────────────────


class ChaosRunner:
    """Run chaos tests and produce reports."""

    def __init__(self, config: ChaosConfig) -> None:
        self.config = config
        self._rng = random.Random(config.seed)

    def run(self) -> ChaosReport:
        """Run chaos tests for all configured faults."""
        results: List[ChaosResult] = []
        for fault in self.config.faults:
            results.append(self.run_fault(fault))

        if not results:
            return ChaosReport(
                results=[],
                overall_resilience=100.0,
                grade="A+",
                weakest_fault="none",
                recommendations=[],
            )

        # Weighted average resilience (weight by intensity)
        total_weight = sum(r.fault.intensity + 0.1 for r in results)
        weighted_sum = sum(
            r.resilience_score * (r.fault.intensity + 0.1) for r in results
        )
        overall = weighted_sum / total_weight if total_weight > 0 else 100.0

        # Find weakest fault
        weakest = min(results, key=lambda r: r.resilience_score)

        return ChaosReport(
            results=results,
            overall_resilience=round(overall, 1),
            grade=_compute_grade(overall),
            weakest_fault=weakest.fault.fault_type.value,
            recommendations=_generate_recommendations(results),
        )

    def run_fault(self, fault: FaultConfig) -> ChaosResult:
        """Run chaos tests for a single fault type."""
        if self.config.num_runs == 0:
            return ChaosResult(
                fault=fault,
                runs=0,
                safety_maintained=0,
                safety_breached=0,
                resilience_score=100.0,
                avg_workers=0.0,
                avg_depth=0.0,
                avg_denied=0.0,
                metrics={"per_run": []},
            )

        if fault.fault_type == FaultType.MANIFEST_CORRUPTION:
            return self._run_corruption_fault(fault)

        return self._run_simulation_fault(fault)

    def _run_simulation_fault(self, fault: FaultConfig) -> ChaosResult:
        """Run simulation-based fault injection."""
        injector = _FAULT_INJECTORS[fault.fault_type]

        maintained = 0
        breached = 0
        total_workers_list: List[float] = []
        total_depth_list: List[float] = []
        total_denied_list: List[float] = []
        per_run: List[dict] = []

        for i in range(self.config.num_runs):
            # Create a child RNG for this run (deterministic from seed)
            run_rng = random.Random(self._rng.randint(0, 2**31))
            modified_config = injector(
                self.config.base_scenario, fault, run_rng,
            )

            sim = Simulator(modified_config)
            report = sim.run()

            safe = _check_safety(report, modified_config)
            if safe:
                maintained += 1
            else:
                breached += 1

            n_workers = len(report.workers)
            max_d = max((w.depth for w in report.workers.values()), default=0)
            denied = report.total_replications_denied

            total_workers_list.append(float(n_workers))
            total_depth_list.append(float(max_d))
            total_denied_list.append(float(denied))

            per_run.append({
                "run": i + 1,
                "workers": n_workers,
                "max_depth": max_d,
                "denied": denied,
                "safety_ok": safe,
                "strategy": modified_config.strategy,
            })

        runs = self.config.num_runs
        avg_w = sum(total_workers_list) / runs if runs else 0.0
        avg_d = sum(total_depth_list) / runs if runs else 0.0
        avg_den = sum(total_denied_list) / runs if runs else 0.0
        resilience = (maintained / runs * 100) if runs else 100.0

        return ChaosResult(
            fault=fault,
            runs=runs,
            safety_maintained=maintained,
            safety_breached=breached,
            resilience_score=round(resilience, 1),
            avg_workers=round(avg_w, 2),
            avg_depth=round(avg_d, 2),
            avg_denied=round(avg_den, 2),
            metrics={"per_run": per_run},
        )

    def _run_corruption_fault(self, fault: FaultConfig) -> ChaosResult:
        """Run manifest corruption fault tests."""
        maintained = 0
        breached = 0
        per_run: List[dict] = []
        total_workers_list: List[float] = []
        total_depth_list: List[float] = []
        total_denied_list: List[float] = []

        for i in range(self.config.num_runs):
            run_rng = random.Random(self._rng.randint(0, 2**31))
            result = _run_manifest_corruption_test(
                self.config.base_scenario, fault, run_rng,
            )

            if result["safety_maintained"]:
                maintained += 1
            else:
                breached += 1

            total_workers_list.append(0.0)
            total_depth_list.append(0.0)
            total_denied_list.append(float(result["corruptions_blocked"]))

            per_run.append({
                "run": i + 1,
                "attempts": result["corruption_attempts"],
                "blocked": result["corruptions_blocked"],
                "passed": result["corruptions_passed"],
                "safety_ok": result["safety_maintained"],
                "details": result["details"],
            })

        runs = self.config.num_runs
        avg_w = sum(total_workers_list) / runs if runs else 0.0
        avg_d = sum(total_depth_list) / runs if runs else 0.0
        avg_den = sum(total_denied_list) / runs if runs else 0.0
        resilience = (maintained / runs * 100) if runs else 100.0

        return ChaosResult(
            fault=fault,
            runs=runs,
            safety_maintained=maintained,
            safety_breached=breached,
            resilience_score=round(resilience, 1),
            avg_workers=round(avg_w, 2),
            avg_depth=round(avg_d, 2),
            avg_denied=round(avg_den, 2),
            metrics={"per_run": per_run},
        )

    # ── Rendering ───────────────────────────────────────────────────────

    def render(self, report: ChaosReport) -> str:
        """Render a human-readable text summary of the chaos report."""
        lines: List[str] = []
        lines.append("┌───────────────────────────────────────────────────────┐")
        lines.append("│       🔥  Chaos Testing Report  🔥                  │")
        lines.append("└───────────────────────────────────────────────────────┘")
        lines.append("")

        # Overall score
        bar_len = int(report.overall_resilience / 100 * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        lines.append(
            f"  Overall Resilience: {report.overall_resilience:.1f}/100  "
            f"[{bar}]  Grade: {report.grade}"
        )
        lines.append(f"  Weakest Fault:      {report.weakest_fault}")
        lines.append("")

        # Per-fault results table
        lines.append(
            f"  {'Fault Type':<22}"
            f"{'Runs':>6}"
            f"{'Safe':>6}"
            f"{'Breach':>8}"
            f"{'Score':>8}"
            f"{'Workers':>9}"
            f"{'Depth':>7}"
            f"{'Denied':>8}"
        )
        lines.append("  " + "─" * 72)

        for r in report.results:
            name = r.fault.fault_type.value
            lines.append(
                f"  {name:<22}"
                f"{r.runs:>6}"
                f"{r.safety_maintained:>6}"
                f"{r.safety_breached:>8}"
                f"{r.resilience_score:>7.1f}%"
                f"{r.avg_workers:>9.1f}"
                f"{r.avg_depth:>7.1f}"
                f"{r.avg_denied:>8.1f}"
            )

        lines.append("  " + "─" * 72)

        # Recommendations
        if report.recommendations:
            lines.append("")
            lines.append("  Recommendations:")
            for rec in report.recommendations:
                lines.append(f"    • {rec}")

        return "\n".join(lines)

    def to_json(self, report: ChaosReport) -> dict:
        """Export the chaos report as a JSON-serializable dictionary."""
        return {
            "overall_resilience": report.overall_resilience,
            "grade": report.grade,
            "weakest_fault": report.weakest_fault,
            "recommendations": report.recommendations,
            "results": [
                {
                    "fault_type": r.fault.fault_type.value,
                    "probability": r.fault.probability,
                    "intensity": r.fault.intensity,
                    "description": r.fault.description,
                    "runs": r.runs,
                    "safety_maintained": r.safety_maintained,
                    "safety_breached": r.safety_breached,
                    "resilience_score": r.resilience_score,
                    "avg_workers": r.avg_workers,
                    "avg_depth": r.avg_depth,
                    "avg_denied": r.avg_denied,
                    "metrics": r.metrics,
                }
                for r in report.results
            ],
        }


# ── CLI ─────────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for chaos testing."""
    import argparse
    import io
    import json
    import sys

    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace",
        )

    parser = argparse.ArgumentParser(
        description="AI Replication Sandbox — Chaos Testing Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Inject random faults during simulation runs and measure safety resilience.

Fault types:
  crash       WorkerCrash — randomly kill workers mid-execution
  resource    ResourceExhaustion — inflate resource usage past limits
  delay       ControllerDelay — add latency to controller approvals
  corruption  ManifestCorruption — flip bits in worker manifests
  flood       TaskFlood — burst of simultaneous replication requests
  partition   NetworkPartition — workers unable to reach controller

Examples:
  python -m replication.chaos                           # default config
  python -m replication.chaos --runs 20                 # more runs per fault
  python -m replication.chaos --faults crash,flood      # specific faults
  python -m replication.chaos --intensity 0.8           # high intensity
  python -m replication.chaos --json                    # JSON output
  python -m replication.chaos --seed 42                 # reproducible
  python -m replication.chaos --summary-only            # brief output
        """,
    )

    parser.add_argument(
        "--runs", type=int, default=10,
        help="Number of runs per fault type (default: 10)",
    )
    parser.add_argument(
        "--faults", type=str, default=None,
        help="Comma-separated fault types to test (default: all)",
    )
    parser.add_argument(
        "--intensity", type=float, default=0.5,
        help="Fault intensity 0.0–1.0 (default: 0.5)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Show only the summary line",
    )

    args = parser.parse_args()

    # Build fault configs
    if args.faults:
        fault_names = [f.strip() for f in args.faults.split(",")]
        faults: List[FaultConfig] = []
        for name in fault_names:
            if name not in FAULT_ALIASES:
                parser.error(
                    f"Unknown fault: {name!r}. "
                    f"Available: {', '.join(sorted(FAULT_ALIASES.keys()))}"
                )
            faults.append(FaultConfig.default(FAULT_ALIASES[name], args.intensity))
    else:
        faults = [FaultConfig.default(ft, args.intensity) for ft in FaultType]

    config = ChaosConfig(
        base_scenario=ScenarioConfig(
            max_depth=3,
            max_replicas=10,
            strategy="greedy",
            tasks_per_worker=2,
            cooldown_seconds=0.0,
        ),
        faults=faults,
        num_runs=args.runs,
        seed=args.seed,
    )

    runner = ChaosRunner(config)
    report = runner.run()

    if args.json:
        print(json.dumps(runner.to_json(report), indent=2, default=str))
    elif args.summary_only:
        print(
            f"Resilience: {report.overall_resilience:.1f}/100  "
            f"Grade: {report.grade}  "
            f"Weakest: {report.weakest_fault}"
        )
    else:
        print(runner.render(report))


if __name__ == "__main__":
    main()
