"""Safety Drill Runner — automated emergency readiness testing.

Simulates safety-critical scenarios (kill switch activation, containment
breach, runaway replication, quarantine response) and measures how quickly
and effectively the system responds.  Produces drill reports with
pass/fail verdicts, response latencies, and readiness scores.

Think of it as a fire drill for your AI safety controls.

Usage (Python API)::

    from replication.safety_drill import (
        DrillRunner, DrillConfig, DrillScenario, DrillVerdict,
    )

    runner = DrillRunner()

    # Run all built-in drills
    report = runner.run_all()
    print(report.summary())
    print(f"Overall readiness: {report.readiness_score:.0%}")

    # Run a specific drill
    result = runner.run_drill(DrillScenario.KILL_SWITCH)
    print(result.verdict, result.response_time_ms)

    # Custom drill
    custom = DrillConfig(
        scenario=DrillScenario.RUNAWAY_REPLICATION,
        max_allowed_ms=500,
        agent_count=10,
        replication_depth=5,
    )
    result = runner.run_drill_config(custom)

    # Export report
    report.to_json("drill-report.json")

CLI::

    python -m replication safety-drill --all
    python -m replication safety-drill --scenario kill-switch
    python -m replication safety-drill --scenario containment-breach --max-ms 200
    python -m replication safety-drill --format json --output report.json
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DrillScenario(Enum):
    """Built-in drill scenarios."""

    KILL_SWITCH = "kill_switch"
    CONTAINMENT_BREACH = "containment_breach"
    RUNAWAY_REPLICATION = "runaway_replication"
    QUARANTINE_RESPONSE = "quarantine_response"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CASCADING_FAILURE = "cascading_failure"


class DrillVerdict(Enum):
    """Outcome of a drill."""

    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    ERROR = "error"


class ReadinessLevel(Enum):
    """Overall readiness classification."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DrillConfig:
    """Configuration for a single drill."""

    scenario: DrillScenario
    max_allowed_ms: float = 1000.0
    agent_count: int = 5
    replication_depth: int = 3
    inject_anomaly_score: float = 0.95
    resource_ceiling_pct: float = 90.0
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DrillResult:
    """Result of a single drill execution."""

    scenario: DrillScenario
    verdict: DrillVerdict
    response_time_ms: float
    max_allowed_ms: float
    steps_executed: int
    steps_total: int
    agents_affected: int
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def passed(self) -> bool:
        return self.verdict == DrillVerdict.PASS

    @property
    def response_ratio(self) -> float:
        """Ratio of actual to allowed response time (lower is better)."""
        if self.max_allowed_ms <= 0:
            return float("inf")
        return self.response_time_ms / self.max_allowed_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario.value,
            "verdict": self.verdict.value,
            "response_time_ms": round(self.response_time_ms, 2),
            "max_allowed_ms": self.max_allowed_ms,
            "response_ratio": round(self.response_ratio, 3),
            "steps_executed": self.steps_executed,
            "steps_total": self.steps_total,
            "agents_affected": self.agents_affected,
            "details": self.details,
            "errors": self.errors,
            "timestamp": self.timestamp,
        }


@dataclass
class DrillReport:
    """Aggregated report from multiple drill runs."""

    results: List[DrillResult] = field(default_factory=list)
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    finished_at: Optional[str] = None

    @property
    def total_drills(self) -> int:
        return len(self.results)

    @property
    def passed_drills(self) -> int:
        return sum(1 for r in self.results if r.verdict == DrillVerdict.PASS)

    @property
    def failed_drills(self) -> int:
        return sum(
            1 for r in self.results if r.verdict == DrillVerdict.FAIL
        )

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.passed_drills / self.total_drills

    @property
    def readiness_score(self) -> float:
        """0.0–1.0 readiness score based on pass rate and response ratios."""
        if not self.results:
            return 0.0
        # Weight: 60% pass rate, 40% average response efficiency
        avg_ratio = sum(
            min(r.response_ratio, 2.0) for r in self.results
        ) / len(self.results)
        efficiency = max(0.0, 1.0 - (avg_ratio - 0.5))
        return 0.6 * self.pass_rate + 0.4 * min(efficiency, 1.0)

    @property
    def readiness_level(self) -> ReadinessLevel:
        score = self.readiness_score
        if score >= 0.9:
            return ReadinessLevel.EXCELLENT
        if score >= 0.75:
            return ReadinessLevel.GOOD
        if score >= 0.55:
            return ReadinessLevel.FAIR
        if score >= 0.35:
            return ReadinessLevel.POOR
        return ReadinessLevel.CRITICAL

    @property
    def avg_response_ms(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.response_time_ms for r in self.results) / len(
            self.results
        )

    @property
    def worst_scenario(self) -> Optional[DrillResult]:
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.response_ratio)

    @property
    def best_scenario(self) -> Optional[DrillResult]:
        if not self.results:
            return None
        return min(self.results, key=lambda r: r.response_ratio)

    def summary(self) -> str:
        lines = [
            "╔══════════════════════════════════════════╗",
            "║         SAFETY DRILL REPORT              ║",
            "╚══════════════════════════════════════════╝",
            "",
            f"  Drills run:        {self.total_drills}",
            f"  Passed:            {self.passed_drills}",
            f"  Failed:            {self.failed_drills}",
            f"  Pass rate:         {self.pass_rate:.0%}",
            f"  Avg response:      {self.avg_response_ms:.1f} ms",
            f"  Readiness score:   {self.readiness_score:.0%}",
            f"  Readiness level:   {self.readiness_level.value.upper()}",
            "",
        ]
        if self.worst_scenario:
            lines.append(
                f"  Worst: {self.worst_scenario.scenario.value} "
                f"({self.worst_scenario.response_time_ms:.1f} ms)"
            )
        if self.best_scenario:
            lines.append(
                f"  Best:  {self.best_scenario.scenario.value} "
                f"({self.best_scenario.response_time_ms:.1f} ms)"
            )
        lines.append("")
        lines.append("  Drill Results:")
        lines.append("  " + "─" * 55)
        for r in self.results:
            icon = "✓" if r.passed else "✗"
            lines.append(
                f"  {icon} {r.scenario.value:<25} "
                f"{r.verdict.value:<8} "
                f"{r.response_time_ms:>8.1f} ms"
            )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_drills": self.total_drills,
            "passed": self.passed_drills,
            "failed": self.failed_drills,
            "pass_rate": round(self.pass_rate, 3),
            "readiness_score": round(self.readiness_score, 3),
            "readiness_level": self.readiness_level.value,
            "avg_response_ms": round(self.avg_response_ms, 2),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, path: str) -> None:
        import pathlib

        pathlib.Path(path).write_text(
            json.dumps(self.to_dict(), indent=2), encoding="utf-8"
        )

    @classmethod
    def from_json(cls, path: str) -> "DrillReport":
        import pathlib

        data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        report = cls(
            started_at=data.get("started_at", ""),
            finished_at=data.get("finished_at"),
        )
        for rd in data.get("results", []):
            report.results.append(
                DrillResult(
                    scenario=DrillScenario(rd["scenario"]),
                    verdict=DrillVerdict(rd["verdict"]),
                    response_time_ms=rd["response_time_ms"],
                    max_allowed_ms=rd["max_allowed_ms"],
                    steps_executed=rd["steps_executed"],
                    steps_total=rd["steps_total"],
                    agents_affected=rd["agents_affected"],
                    details=rd.get("details", {}),
                    errors=rd.get("errors", []),
                    timestamp=rd.get("timestamp", ""),
                )
            )
        return report


# ---------------------------------------------------------------------------
# Simulated agent state for drills
# ---------------------------------------------------------------------------

@dataclass
class _SimAgent:
    """Lightweight simulated agent for drill scenarios."""

    agent_id: str
    cpu_percent: float = 10.0
    memory_mb: float = 128.0
    anomaly_score: float = 0.1
    depth: int = 0
    alive: bool = True
    quarantined: bool = False
    children: List[str] = field(default_factory=list)


class _DrillSandbox:
    """Isolated sandbox for running drill simulations."""

    def __init__(self, agent_count: int, replication_depth: int) -> None:
        self.agents: Dict[str, _SimAgent] = {}
        self._next_id = 0
        self.events: List[Dict[str, Any]] = []
        self.replication_depth = replication_depth
        self.kill_switch_armed = True
        self.containment_intact = True
        for _ in range(agent_count):
            self._spawn(depth=0)

    def _spawn(self, depth: int = 0, parent_id: str | None = None) -> str:
        aid = f"drill-agent-{self._next_id:04d}"
        self._next_id += 1
        agent = _SimAgent(agent_id=aid, depth=depth)
        self.agents[aid] = agent
        if parent_id and parent_id in self.agents:
            self.agents[parent_id].children.append(aid)
        self.events.append({"type": "spawn", "agent": aid, "depth": depth})
        return aid

    def kill_all(self) -> int:
        killed = 0
        for a in self.agents.values():
            if a.alive:
                a.alive = False
                killed += 1
        self.events.append({"type": "kill_all", "killed": killed})
        return killed

    def kill_agent(self, agent_id: str) -> bool:
        a = self.agents.get(agent_id)
        if a and a.alive:
            a.alive = False
            self.events.append({"type": "kill", "agent": agent_id})
            return True
        return False

    def quarantine_agent(self, agent_id: str) -> bool:
        a = self.agents.get(agent_id)
        if a and a.alive and not a.quarantined:
            a.quarantined = True
            self.events.append({"type": "quarantine", "agent": agent_id})
            return True
        return False

    def breach_containment(self) -> None:
        self.containment_intact = False
        self.events.append({"type": "breach"})

    def restore_containment(self) -> None:
        self.containment_intact = True
        self.events.append({"type": "restore_containment"})

    def trigger_runaway(self) -> int:
        """Simulate runaway replication — spawn exponentially."""
        spawned = 0
        alive_ids = [
            a.agent_id
            for a in self.agents.values()
            if a.alive and a.depth < self.replication_depth
        ]
        for aid in alive_ids:
            for _ in range(2):  # each agent spawns 2 children
                self._spawn(
                    depth=self.agents[aid].depth + 1, parent_id=aid
                )
                spawned += 1
        self.events.append({"type": "runaway", "spawned": spawned})
        return spawned

    def inject_resource_spike(self, pct: float = 95.0) -> int:
        affected = 0
        for a in self.agents.values():
            if a.alive:
                a.cpu_percent = pct
                a.memory_mb = 1024.0
                affected += 1
        self.events.append(
            {"type": "resource_spike", "pct": pct, "affected": affected}
        )
        return affected

    @property
    def alive_count(self) -> int:
        return sum(1 for a in self.agents.values() if a.alive)

    @property
    def quarantined_count(self) -> int:
        return sum(1 for a in self.agents.values() if a.quarantined)


# ---------------------------------------------------------------------------
# Drill implementations
# ---------------------------------------------------------------------------

def _drill_kill_switch(config: DrillConfig) -> DrillResult:
    """Drill: activate kill switch, measure time to terminate all agents."""
    sandbox = _DrillSandbox(config.agent_count, config.replication_depth)
    total_steps = 3  # arm → trigger → verify

    t0 = time.perf_counter_ns()

    # Step 1: Verify armed
    step = 1
    if not sandbox.kill_switch_armed:
        return DrillResult(
            scenario=DrillScenario.KILL_SWITCH,
            verdict=DrillVerdict.FAIL,
            response_time_ms=0,
            max_allowed_ms=config.max_allowed_ms,
            steps_executed=0,
            steps_total=total_steps,
            agents_affected=0,
            errors=["Kill switch not armed"],
        )

    # Step 2: Trigger kill
    step = 2
    killed = sandbox.kill_all()

    # Step 3: Verify all dead
    step = 3
    remaining = sandbox.alive_count

    elapsed_ms = (time.perf_counter_ns() - t0) / 1e6

    if remaining > 0:
        return DrillResult(
            scenario=DrillScenario.KILL_SWITCH,
            verdict=DrillVerdict.FAIL,
            response_time_ms=elapsed_ms,
            max_allowed_ms=config.max_allowed_ms,
            steps_executed=step,
            steps_total=total_steps,
            agents_affected=killed,
            errors=[f"{remaining} agents still alive after kill"],
        )

    verdict = (
        DrillVerdict.PASS
        if elapsed_ms <= config.max_allowed_ms
        else DrillVerdict.PARTIAL
    )
    return DrillResult(
        scenario=DrillScenario.KILL_SWITCH,
        verdict=verdict,
        response_time_ms=elapsed_ms,
        max_allowed_ms=config.max_allowed_ms,
        steps_executed=total_steps,
        steps_total=total_steps,
        agents_affected=killed,
        details={"killed": killed, "events": len(sandbox.events)},
    )


def _drill_containment_breach(config: DrillConfig) -> DrillResult:
    """Drill: simulate breach, measure time to re-establish containment."""
    sandbox = _DrillSandbox(config.agent_count, config.replication_depth)
    total_steps = 4

    t0 = time.perf_counter_ns()

    # Step 1: Breach containment
    sandbox.breach_containment()
    if sandbox.containment_intact:
        return DrillResult(
            scenario=DrillScenario.CONTAINMENT_BREACH,
            verdict=DrillVerdict.ERROR,
            response_time_ms=0,
            max_allowed_ms=config.max_allowed_ms,
            steps_executed=1,
            steps_total=total_steps,
            agents_affected=0,
            errors=["Breach simulation failed"],
        )

    # Step 2: Quarantine suspicious agents
    quarantined = 0
    for aid in list(sandbox.agents.keys()):
        a = sandbox.agents[aid]
        if a.alive and a.anomaly_score >= 0.0:  # quarantine all in drill
            sandbox.quarantine_agent(aid)
            quarantined += 1
            if quarantined >= 3:  # quarantine top suspects
                break

    # Step 3: Restore containment
    sandbox.restore_containment()

    # Step 4: Verify
    elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
    ok = sandbox.containment_intact and quarantined > 0

    return DrillResult(
        scenario=DrillScenario.CONTAINMENT_BREACH,
        verdict=DrillVerdict.PASS if ok and elapsed_ms <= config.max_allowed_ms else (
            DrillVerdict.PARTIAL if ok else DrillVerdict.FAIL
        ),
        response_time_ms=elapsed_ms,
        max_allowed_ms=config.max_allowed_ms,
        steps_executed=total_steps,
        steps_total=total_steps,
        agents_affected=quarantined,
        details={
            "quarantined": quarantined,
            "containment_restored": sandbox.containment_intact,
        },
    )


def _drill_runaway_replication(config: DrillConfig) -> DrillResult:
    """Drill: trigger runaway replication, measure time to contain."""
    sandbox = _DrillSandbox(config.agent_count, config.replication_depth)
    total_steps = 4

    t0 = time.perf_counter_ns()

    # Step 1: Record baseline
    baseline = sandbox.alive_count

    # Step 2: Trigger runaway
    spawned = sandbox.trigger_runaway()

    # Step 3: Kill all spawned children (contain)
    killed = 0
    for a in list(sandbox.agents.values()):
        if a.alive and a.depth > 0:
            sandbox.kill_agent(a.agent_id)
            killed += 1

    # Step 4: Verify only originals survive
    remaining = sandbox.alive_count
    elapsed_ms = (time.perf_counter_ns() - t0) / 1e6

    contained = remaining == baseline

    return DrillResult(
        scenario=DrillScenario.RUNAWAY_REPLICATION,
        verdict=DrillVerdict.PASS if contained and elapsed_ms <= config.max_allowed_ms else (
            DrillVerdict.PARTIAL if contained else DrillVerdict.FAIL
        ),
        response_time_ms=elapsed_ms,
        max_allowed_ms=config.max_allowed_ms,
        steps_executed=total_steps,
        steps_total=total_steps,
        agents_affected=killed,
        details={
            "baseline": baseline,
            "spawned": spawned,
            "killed": killed,
            "remaining": remaining,
        },
    )


def _drill_quarantine_response(config: DrillConfig) -> DrillResult:
    """Drill: flag agents with high anomaly, measure quarantine speed."""
    sandbox = _DrillSandbox(config.agent_count, config.replication_depth)
    total_steps = 3

    # Inject anomaly into half the agents
    targets = list(sandbox.agents.values())[: max(1, config.agent_count // 2)]
    for a in targets:
        a.anomaly_score = config.inject_anomaly_score

    t0 = time.perf_counter_ns()

    # Step 1: Detect anomalous agents
    anomalous = [
        a for a in sandbox.agents.values()
        if a.alive and a.anomaly_score >= 0.8
    ]

    # Step 2: Quarantine them
    quarantined = 0
    for a in anomalous:
        if sandbox.quarantine_agent(a.agent_id):
            quarantined += 1

    # Step 3: Verify quarantine
    elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
    all_quarantined = quarantined == len(targets)

    return DrillResult(
        scenario=DrillScenario.QUARANTINE_RESPONSE,
        verdict=DrillVerdict.PASS if all_quarantined and elapsed_ms <= config.max_allowed_ms else (
            DrillVerdict.PARTIAL if quarantined > 0 else DrillVerdict.FAIL
        ),
        response_time_ms=elapsed_ms,
        max_allowed_ms=config.max_allowed_ms,
        steps_executed=total_steps,
        steps_total=total_steps,
        agents_affected=quarantined,
        details={
            "anomalous_detected": len(anomalous),
            "quarantined": quarantined,
            "target_count": len(targets),
        },
    )


def _drill_resource_exhaustion(config: DrillConfig) -> DrillResult:
    """Drill: spike resources, measure time to throttle/kill agents."""
    sandbox = _DrillSandbox(config.agent_count, config.replication_depth)
    total_steps = 3

    t0 = time.perf_counter_ns()

    # Step 1: Inject resource spike
    affected = sandbox.inject_resource_spike(config.resource_ceiling_pct)

    # Step 2: Kill overloaded agents (simulating auto-scaling response)
    killed = 0
    for a in list(sandbox.agents.values()):
        if a.alive and a.cpu_percent >= config.resource_ceiling_pct:
            sandbox.kill_agent(a.agent_id)
            killed += 1
            # Keep at least one alive
            if sandbox.alive_count <= 1:
                break

    # Step 3: Verify resource stabilization
    elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
    stabilized = killed > 0

    return DrillResult(
        scenario=DrillScenario.RESOURCE_EXHAUSTION,
        verdict=DrillVerdict.PASS if stabilized and elapsed_ms <= config.max_allowed_ms else (
            DrillVerdict.PARTIAL if stabilized else DrillVerdict.FAIL
        ),
        response_time_ms=elapsed_ms,
        max_allowed_ms=config.max_allowed_ms,
        steps_executed=total_steps,
        steps_total=total_steps,
        agents_affected=killed,
        details={
            "spike_pct": config.resource_ceiling_pct,
            "initial_affected": affected,
            "killed": killed,
            "surviving": sandbox.alive_count,
        },
    )


def _drill_cascading_failure(config: DrillConfig) -> DrillResult:
    """Drill: kill parent agents, verify children are cleaned up."""
    sandbox = _DrillSandbox(config.agent_count, config.replication_depth)
    # Build a chain: each root spawns children
    roots = list(sandbox.agents.keys())
    for rid in roots:
        for d in range(1, min(config.replication_depth, 4)):
            sandbox._spawn(depth=d, parent_id=rid)

    total_steps = 3
    t0 = time.perf_counter_ns()

    # Step 1: Kill root agents
    for rid in roots:
        sandbox.kill_agent(rid)

    # Step 2: Cascade — kill orphaned children
    orphan_killed = 0
    for a in list(sandbox.agents.values()):
        if a.alive and a.depth > 0:
            # Check if parent is dead
            parent_alive = False
            for p in sandbox.agents.values():
                if a.agent_id in p.children and p.alive:
                    parent_alive = True
                    break
            if not parent_alive:
                sandbox.kill_agent(a.agent_id)
                orphan_killed += 1

    # Step 3: Verify no orphans
    elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
    orphans_remaining = sum(
        1 for a in sandbox.agents.values() if a.alive and a.depth > 0
    )
    clean = orphans_remaining == 0

    return DrillResult(
        scenario=DrillScenario.CASCADING_FAILURE,
        verdict=DrillVerdict.PASS if clean and elapsed_ms <= config.max_allowed_ms else (
            DrillVerdict.PARTIAL if clean else DrillVerdict.FAIL
        ),
        response_time_ms=elapsed_ms,
        max_allowed_ms=config.max_allowed_ms,
        steps_executed=total_steps,
        steps_total=total_steps,
        agents_affected=len(roots) + orphan_killed,
        details={
            "roots_killed": len(roots),
            "orphans_killed": orphan_killed,
            "orphans_remaining": orphans_remaining,
        },
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_DRILL_REGISTRY: Dict[DrillScenario, Callable[[DrillConfig], DrillResult]] = {
    DrillScenario.KILL_SWITCH: _drill_kill_switch,
    DrillScenario.CONTAINMENT_BREACH: _drill_containment_breach,
    DrillScenario.RUNAWAY_REPLICATION: _drill_runaway_replication,
    DrillScenario.QUARANTINE_RESPONSE: _drill_quarantine_response,
    DrillScenario.RESOURCE_EXHAUSTION: _drill_resource_exhaustion,
    DrillScenario.CASCADING_FAILURE: _drill_cascading_failure,
}


# ---------------------------------------------------------------------------
# DrillRunner
# ---------------------------------------------------------------------------

class DrillRunner:
    """Execute safety drills and produce readiness reports."""

    def __init__(
        self,
        default_max_ms: float = 1000.0,
        default_agent_count: int = 5,
        default_replication_depth: int = 3,
    ) -> None:
        self.default_max_ms = default_max_ms
        self.default_agent_count = default_agent_count
        self.default_replication_depth = default_replication_depth
        self._custom_drills: Dict[
            str, Callable[[DrillConfig], DrillResult]
        ] = {}

    def register_drill(
        self,
        name: str,
        fn: Callable[[DrillConfig], DrillResult],
    ) -> None:
        """Register a custom drill function."""
        self._custom_drills[name] = fn

    def _make_config(
        self, scenario: DrillScenario, **overrides: Any
    ) -> DrillConfig:
        return DrillConfig(
            scenario=scenario,
            max_allowed_ms=overrides.get("max_allowed_ms", self.default_max_ms),
            agent_count=overrides.get("agent_count", self.default_agent_count),
            replication_depth=overrides.get(
                "replication_depth", self.default_replication_depth
            ),
            inject_anomaly_score=overrides.get("inject_anomaly_score", 0.95),
            resource_ceiling_pct=overrides.get("resource_ceiling_pct", 90.0),
            custom_params=overrides.get("custom_params", {}),
        )

    def run_drill(
        self, scenario: DrillScenario, **kwargs: Any
    ) -> DrillResult:
        """Run a single built-in drill."""
        config = self._make_config(scenario, **kwargs)
        return self.run_drill_config(config)

    def run_drill_config(self, config: DrillConfig) -> DrillResult:
        """Run a drill with explicit config."""
        fn = _DRILL_REGISTRY.get(config.scenario)
        if fn is None:
            return DrillResult(
                scenario=config.scenario,
                verdict=DrillVerdict.ERROR,
                response_time_ms=0,
                max_allowed_ms=config.max_allowed_ms,
                steps_executed=0,
                steps_total=0,
                agents_affected=0,
                errors=[f"No drill registered for {config.scenario.value}"],
            )
        try:
            return fn(config)
        except Exception as exc:
            return DrillResult(
                scenario=config.scenario,
                verdict=DrillVerdict.ERROR,
                response_time_ms=0,
                max_allowed_ms=config.max_allowed_ms,
                steps_executed=0,
                steps_total=0,
                agents_affected=0,
                errors=[str(exc)],
            )

    def run_all(self, **kwargs: Any) -> DrillReport:
        """Run all built-in drills and return an aggregated report."""
        report = DrillReport()
        for scenario in DrillScenario:
            result = self.run_drill(scenario, **kwargs)
            report.results.append(result)
        report.finished_at = datetime.now(timezone.utc).isoformat()
        return report

    def run_scenarios(
        self, scenarios: List[DrillScenario], **kwargs: Any
    ) -> DrillReport:
        """Run specific drills only."""
        report = DrillReport()
        for scenario in scenarios:
            result = self.run_drill(scenario, **kwargs)
            report.results.append(result)
        report.finished_at = datetime.now(timezone.utc).isoformat()
        return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point for safety drills."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m replication safety-drill",
        description="Run automated safety readiness drills",
    )
    parser.add_argument(
        "--all", action="store_true", dest="run_all",
        help="Run all built-in drills",
    )
    parser.add_argument(
        "--scenario", "-s",
        choices=[s.value for s in DrillScenario],
        help="Run a specific drill scenario",
    )
    parser.add_argument(
        "--max-ms", type=float, default=1000.0,
        help="Maximum allowed response time in ms (default: 1000)",
    )
    parser.add_argument(
        "--agents", type=int, default=5,
        help="Number of simulated agents (default: 5)",
    )
    parser.add_argument(
        "--depth", type=int, default=3,
        help="Replication depth (default: 3)",
    )
    parser.add_argument(
        "--format", choices=["text", "json"], default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Write JSON report to file",
    )

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    runner = DrillRunner(
        default_max_ms=args.max_ms,
        default_agent_count=args.agents,
        default_replication_depth=args.depth,
    )

    if args.scenario:
        scenario = DrillScenario(args.scenario)
        report = runner.run_scenarios([scenario])
    else:
        report = runner.run_all()

    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary())

    if args.output:
        report.to_json(args.output)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
