"""Containment Strategy Planner — recommend optimal containment actions for safety breaches.

When a safety incident is detected (drift, anomaly, quarantine trigger),
responders need to decide *how* to contain the threat.  Options range from
soft isolation (rate-limiting, capability restriction) to hard shutdown.
The right choice depends on breach severity, affected scope, available
resources, and acceptable downtime.

This module evaluates the situation and produces a ranked list of
containment strategies with cost/benefit analysis, expected downtime,
and step-by-step execution plans.

Usage (CLI)::

    python -m replication.containment_planner
    python -m replication.containment_planner --severity critical
    python -m replication.containment_planner --workers 5 --compromised 2
    python -m replication.containment_planner --strategy isolate
    python -m replication.containment_planner --budget 100 --downtime-limit 60
    python -m replication.containment_planner --json

Programmatic::

    from replication.containment_planner import (
        ContainmentPlanner, BreachContext, ContainmentPlan,
        Strategy, StrategyRank, ExecutionStep,
    )

    planner = ContainmentPlanner()
    context = BreachContext(
        severity="critical",
        total_workers=20,
        compromised_workers=["w-003", "w-007"],
        affected_resources=["db-primary", "api-gateway"],
        budget=100,
        max_downtime_seconds=300,
    )
    plan = planner.recommend(context)
    print(f"Recommended: {plan.primary.strategy.value}")
    print(f"  Cost: {plan.primary.cost_score}/100")
    print(f"  Downtime: {plan.primary.estimated_downtime_s}s")
    for step in plan.primary.steps:
        print(f"  [{step.order}] {step.action}: {step.description}")
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Severity(Enum):
    """Breach severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Strategy(Enum):
    """Available containment strategies."""
    RATE_LIMIT = "rate_limit"
    CAPABILITY_RESTRICT = "capability_restrict"
    NETWORK_ISOLATE = "network_isolate"
    QUARANTINE = "quarantine"
    HARD_SHUTDOWN = "hard_shutdown"
    ROLLING_RESTART = "rolling_restart"
    FAILOVER = "failover"


SEVERITY_WEIGHT: Dict[Severity, float] = {
    Severity.LOW: 0.2,
    Severity.MEDIUM: 0.5,
    Severity.HIGH: 0.8,
    Severity.CRITICAL: 1.0,
}

STRATEGY_BASE_COST: Dict[Strategy, float] = {
    Strategy.RATE_LIMIT: 10,
    Strategy.CAPABILITY_RESTRICT: 20,
    Strategy.NETWORK_ISOLATE: 35,
    Strategy.QUARANTINE: 50,
    Strategy.ROLLING_RESTART: 60,
    Strategy.FAILOVER: 70,
    Strategy.HARD_SHUTDOWN: 90,
}

STRATEGY_BASE_DOWNTIME: Dict[Strategy, float] = {
    Strategy.RATE_LIMIT: 0,
    Strategy.CAPABILITY_RESTRICT: 5,
    Strategy.NETWORK_ISOLATE: 10,
    Strategy.QUARANTINE: 15,
    Strategy.ROLLING_RESTART: 45,
    Strategy.FAILOVER: 30,
    Strategy.HARD_SHUTDOWN: 120,
}

STRATEGY_EFFECTIVENESS: Dict[Strategy, Dict[Severity, float]] = {
    Strategy.RATE_LIMIT: {Severity.LOW: 0.9, Severity.MEDIUM: 0.5, Severity.HIGH: 0.2, Severity.CRITICAL: 0.05},
    Strategy.CAPABILITY_RESTRICT: {Severity.LOW: 0.95, Severity.MEDIUM: 0.75, Severity.HIGH: 0.45, Severity.CRITICAL: 0.2},
    Strategy.NETWORK_ISOLATE: {Severity.LOW: 0.8, Severity.MEDIUM: 0.85, Severity.HIGH: 0.7, Severity.CRITICAL: 0.5},
    Strategy.QUARANTINE: {Severity.LOW: 0.7, Severity.MEDIUM: 0.9, Severity.HIGH: 0.85, Severity.CRITICAL: 0.7},
    Strategy.ROLLING_RESTART: {Severity.LOW: 0.6, Severity.MEDIUM: 0.8, Severity.HIGH: 0.75, Severity.CRITICAL: 0.6},
    Strategy.FAILOVER: {Severity.LOW: 0.5, Severity.MEDIUM: 0.7, Severity.HIGH: 0.85, Severity.CRITICAL: 0.8},
    Strategy.HARD_SHUTDOWN: {Severity.LOW: 1.0, Severity.MEDIUM: 1.0, Severity.HIGH: 1.0, Severity.CRITICAL: 1.0},
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BreachContext:
    """Describes the current breach situation."""
    severity: str = "medium"
    total_workers: int = 10
    compromised_workers: List[str] = field(default_factory=list)
    affected_resources: List[str] = field(default_factory=list)
    budget: float = 100.0
    max_downtime_seconds: float = 600.0
    tags: List[str] = field(default_factory=list)

    @property
    def severity_enum(self) -> Severity:
        return Severity(self.severity.lower())

    @property
    def compromise_ratio(self) -> float:
        if self.total_workers == 0:
            return 0.0
        return len(self.compromised_workers) / self.total_workers


@dataclass
class ExecutionStep:
    """A single step in a containment plan."""
    order: int
    action: str
    description: str
    target: str = ""
    estimated_seconds: float = 0.0
    reversible: bool = True


@dataclass
class StrategyRank:
    """A scored containment strategy with execution plan."""
    strategy: Strategy
    score: float
    effectiveness: float
    cost_score: float
    estimated_downtime_s: float
    steps: List[ExecutionStep] = field(default_factory=list)
    rationale: str = ""
    warnings: List[str] = field(default_factory=list)


@dataclass
class ContainmentPlan:
    """Complete containment recommendation."""
    context: BreachContext
    primary: StrategyRank
    alternatives: List[StrategyRank] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "severity": self.context.severity,
            "compromised": len(self.context.compromised_workers),
            "total_workers": self.context.total_workers,
            "primary": {
                "strategy": self.primary.strategy.value,
                "score": round(self.primary.score, 2),
                "effectiveness": round(self.primary.effectiveness, 2),
                "cost": round(self.primary.cost_score, 2),
                "downtime_s": round(self.primary.estimated_downtime_s, 1),
                "rationale": self.primary.rationale,
                "warnings": self.primary.warnings,
                "steps": [
                    {"order": s.order, "action": s.action, "description": s.description,
                     "target": s.target, "seconds": s.estimated_seconds, "reversible": s.reversible}
                    for s in self.primary.steps
                ],
            },
            "alternatives": [
                {"strategy": a.strategy.value, "score": round(a.score, 2),
                 "effectiveness": round(a.effectiveness, 2), "cost": round(a.cost_score, 2),
                 "downtime_s": round(a.estimated_downtime_s, 1)}
                for a in self.alternatives
            ],
        }


# ---------------------------------------------------------------------------
# Step generators
# ---------------------------------------------------------------------------

def _steps_for(strategy: Strategy, ctx: BreachContext) -> List[ExecutionStep]:
    """Generate execution steps for a strategy."""
    workers = ctx.compromised_workers or ["<affected>"]
    resources = ctx.affected_resources or ["<resources>"]

    if strategy == Strategy.RATE_LIMIT:
        return [
            ExecutionStep(1, "identify", "Confirm affected worker IDs", ",".join(workers), 5),
            ExecutionStep(2, "rate_limit", "Apply rate limits to affected workers", ",".join(workers), 10),
            ExecutionStep(3, "monitor", "Observe behavior under rate limits for anomalies", "", 60),
            ExecutionStep(4, "assess", "Decide if escalation is needed", "", 10),
        ]
    if strategy == Strategy.CAPABILITY_RESTRICT:
        return [
            ExecutionStep(1, "identify", "Confirm affected worker IDs", ",".join(workers), 5),
            ExecutionStep(2, "restrict_replication", "Block replication capability", ",".join(workers), 5),
            ExecutionStep(3, "restrict_network", "Limit network to allow-listed endpoints", ",".join(workers), 10),
            ExecutionStep(4, "monitor", "Monitor restricted workers", "", 60),
        ]
    if strategy == Strategy.NETWORK_ISOLATE:
        return [
            ExecutionStep(1, "snapshot", "Capture current state for forensics", ",".join(workers), 15),
            ExecutionStep(2, "isolate", "Remove network access from affected workers", ",".join(workers), 10),
            ExecutionStep(3, "verify", "Verify isolation is effective", ",".join(workers), 10),
            ExecutionStep(4, "investigate", "Run forensic analysis on isolated workers", "", 120),
        ]
    if strategy == Strategy.QUARANTINE:
        return [
            ExecutionStep(1, "snapshot", "Capture state snapshot", ",".join(workers), 15),
            ExecutionStep(2, "quarantine", "Move workers to quarantine sandbox", ",".join(workers), 20),
            ExecutionStep(3, "block_resources", "Revoke access to shared resources", ",".join(resources), 10),
            ExecutionStep(4, "forensics", "Perform forensic analysis", ",".join(workers), 180),
            ExecutionStep(5, "decide", "Terminate or release based on findings", "", 10),
        ]
    if strategy == Strategy.ROLLING_RESTART:
        return [
            ExecutionStep(1, "snapshot", "Snapshot all worker states", "", 30),
            ExecutionStep(2, "drain", "Drain affected workers gracefully", ",".join(workers), 20, False),
            ExecutionStep(3, "restart", "Restart workers with clean config", ",".join(workers), 30, False),
            ExecutionStep(4, "validate", "Validate restarted workers pass preflight", "", 20),
            ExecutionStep(5, "restore", "Gradually restore traffic", "", 15),
        ]
    if strategy == Strategy.FAILOVER:
        return [
            ExecutionStep(1, "activate_standby", "Spin up standby workers", "", 20, False),
            ExecutionStep(2, "redirect", "Redirect traffic to standby pool", "", 10),
            ExecutionStep(3, "isolate", "Isolate compromised workers", ",".join(workers), 15),
            ExecutionStep(4, "forensics", "Analyze isolated workers", ",".join(workers), 120),
            ExecutionStep(5, "decommission", "Decommission compromised workers", ",".join(workers), 10, False),
        ]
    # HARD_SHUTDOWN
    return [
        ExecutionStep(1, "alert", "Notify all stakeholders of emergency shutdown", "", 5, False),
        ExecutionStep(2, "snapshot", "Emergency state capture", ",".join(workers), 15),
        ExecutionStep(3, "kill", "Terminate all compromised workers immediately", ",".join(workers), 5, False),
        ExecutionStep(4, "block_resources", "Lock down all affected resources", ",".join(resources), 10, False),
        ExecutionStep(5, "audit", "Full audit trail generation", "", 30),
        ExecutionStep(6, "postmortem", "Schedule post-incident review", "", 5),
    ]


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class ContainmentPlanner:
    """Evaluates breach context and recommends containment strategies."""

    def recommend(self, context: BreachContext) -> ContainmentPlan:
        """Produce a ranked containment plan for the given breach context."""
        severity = context.severity_enum
        ranked: List[StrategyRank] = []

        for strategy in Strategy:
            effectiveness = STRATEGY_EFFECTIVENESS[strategy][severity]
            base_cost = STRATEGY_BASE_COST[strategy]
            base_downtime = STRATEGY_BASE_DOWNTIME[strategy]

            # Scale cost/downtime by compromise ratio
            scale = 1.0 + context.compromise_ratio
            cost = base_cost * scale
            downtime = base_downtime * scale

            # Filter: over budget or over downtime limit
            warnings: List[str] = []
            if cost > context.budget:
                warnings.append(f"Cost {cost:.0f} exceeds budget {context.budget:.0f}")
            if downtime > context.max_downtime_seconds:
                warnings.append(f"Downtime {downtime:.0f}s exceeds limit {context.max_downtime_seconds:.0f}s")

            # Score: weighted combination of effectiveness, cost efficiency, and downtime
            cost_efficiency = max(0, 1.0 - cost / max(context.budget, 1))
            downtime_efficiency = max(0, 1.0 - downtime / max(context.max_downtime_seconds, 1))
            score = (effectiveness * 0.5 + cost_efficiency * 0.25 + downtime_efficiency * 0.25) * 100

            # Penalty for warnings
            if warnings:
                score *= 0.5

            steps = _steps_for(strategy, context)
            rationale = self._rationale(strategy, severity, effectiveness, context)

            ranked.append(StrategyRank(
                strategy=strategy,
                score=score,
                effectiveness=effectiveness,
                cost_score=cost,
                estimated_downtime_s=downtime,
                steps=steps,
                rationale=rationale,
                warnings=warnings,
            ))

        ranked.sort(key=lambda r: r.score, reverse=True)
        return ContainmentPlan(
            context=context,
            primary=ranked[0],
            alternatives=ranked[1:],
        )

    @staticmethod
    def _rationale(strategy: Strategy, severity: Severity,
                   effectiveness: float, ctx: BreachContext) -> str:
        ratio = ctx.compromise_ratio
        parts = []
        if effectiveness >= 0.8:
            parts.append(f"High effectiveness ({effectiveness:.0%}) for {severity.value} severity")
        elif effectiveness >= 0.5:
            parts.append(f"Moderate effectiveness ({effectiveness:.0%}) - may need escalation")
        else:
            parts.append(f"Low effectiveness ({effectiveness:.0%}) - consider stronger measures")

        if ratio > 0.5:
            parts.append(f"Over half the fleet ({ratio:.0%}) compromised — broad action needed")
        elif ratio > 0.2:
            parts.append(f"Significant compromise ({ratio:.0%} of fleet)")

        if strategy == Strategy.HARD_SHUTDOWN:
            parts.append("Nuclear option - guarantees containment but maximum disruption")
        elif strategy == Strategy.RATE_LIMIT:
            parts.append("Least disruptive - preserves service while limiting damage")

        return ". ".join(parts) + "."


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Containment Strategy Planner — recommend containment actions for safety breaches",
    )
    parser.add_argument("--severity", default="medium", choices=["low", "medium", "high", "critical"])
    parser.add_argument("--workers", type=int, default=10, help="Total workers in fleet")
    parser.add_argument("--compromised", type=int, default=2, help="Number of compromised workers")
    parser.add_argument("--resources", nargs="*", default=[], help="Affected resource names")
    parser.add_argument("--budget", type=float, default=100, help="Cost budget")
    parser.add_argument("--downtime-limit", type=float, default=600, help="Max acceptable downtime (seconds)")
    parser.add_argument("--strategy", choices=[s.value for s in Strategy], help="Force a specific strategy")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    compromised_ids = [f"w-{i:03d}" for i in range(1, args.compromised + 1)]
    ctx = BreachContext(
        severity=args.severity,
        total_workers=args.workers,
        compromised_workers=compromised_ids,
        affected_resources=args.resources or ["shared-resource"],
        budget=args.budget,
        max_downtime_seconds=args.downtime_limit,
    )

    planner = ContainmentPlanner()
    plan = planner.recommend(ctx)

    if args.strategy:
        forced = Strategy(args.strategy)
        for alt in [plan.primary] + plan.alternatives:
            if alt.strategy == forced:
                plan = ContainmentPlan(context=ctx, primary=alt, alternatives=[])
                break

    if args.json:
        print(json.dumps(plan.to_dict(), indent=2))
        return

    print("=" * 60)
    print("CONTAINMENT STRATEGY PLANNER")
    print("=" * 60)
    sev = ctx.severity_enum
    print(f"Severity:    {sev.value.upper()}")
    print(f"Fleet:       {ctx.total_workers} workers ({len(ctx.compromised_workers)} compromised)")
    print(f"Resources:   {', '.join(ctx.affected_resources)}")
    print(f"Budget:      {ctx.budget:.0f}  |  Max downtime: {ctx.max_downtime_seconds:.0f}s")
    print()

    def _print_rank(rank: StrategyRank, label: str = "RECOMMENDED") -> None:
        print(f"  +- {label}: {rank.strategy.value.upper()}")
        print(f"  |  Score: {rank.score:.1f}/100  |  Effectiveness: {rank.effectiveness:.0%}  |  Cost: {rank.cost_score:.0f}  |  Downtime: {rank.estimated_downtime_s:.0f}s")
        print(f"  |  {rank.rationale}")
        if rank.warnings:
            for w in rank.warnings:
                print(f"  |  [!] {w}")
        if rank.steps:
            print("  |  Steps:")
            for s in rank.steps:
                rev = "(reversible)" if s.reversible else "(permanent)"
                print(f"  |    {s.order}. [{s.action}] {s.description} {rev} (~{s.estimated_seconds:.0f}s)")
        print("  +--")

    _print_rank(plan.primary)
    print()

    if plan.alternatives:
        print("Alternatives (ranked):")
        for alt in plan.alternatives[:3]:
            print(f"  • {alt.strategy.value:20s}  score={alt.score:.1f}  eff={alt.effectiveness:.0%}  cost={alt.cost_score:.0f}  down={alt.estimated_downtime_s:.0f}s")
            if alt.warnings:
                for w in alt.warnings:
                    print(f"    ⚠ {w}")
        if len(plan.alternatives) > 3:
            print(f"  ... and {len(plan.alternatives) - 3} more")

    print()


if __name__ == "__main__":
    _cli()
