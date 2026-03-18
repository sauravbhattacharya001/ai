"""Safety SLA Monitor — define, track, and enforce safety-level agreements.

Define SLA targets for key safety metrics (e.g., max replication depth,
kill-switch response time, contract violation rate) and evaluate whether
a simulation run or scorecard result meets those targets.

Usage (CLI)::

    python -m replication sla                              # check defaults
    python -m replication sla --preset strict               # strict SLA preset
    python -m replication sla --preset relaxed              # relaxed SLA preset
    python -m replication sla --target "max_depth<=3"       # custom target
    python -m replication sla --target "violation_rate<0.05" --target "score>=85"
    python -m replication sla --json                        # JSON output
    python -m replication sla --strategy greedy --max-depth 5

Programmatic::

    from replication.sla_monitor import SLAMonitor, SLATarget
    monitor = SLAMonitor()
    monitor.add_target(SLATarget("max_depth", "<=", 3))
    monitor.add_target(SLATarget("overall_score", ">=", 80))
    result = monitor.evaluate()
    print(result.render())
    print(f"SLA status: {'PASS' if result.passed else 'BREACH'}")
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .simulator import Simulator, ScenarioConfig, SimulationReport, PRESETS
from .scorecard import SafetyScorecard, ScorecardConfig, ScorecardResult
from ._helpers import box_header as _box_header


# ── SLA Target definition ──────────────────────────────────────────────


_OPERATORS = {
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
    "<":  lambda a, b: a < b,
    ">":  lambda a, b: a > b,
    "==": lambda a, b: abs(a - b) < 1e-9,
    "!=": lambda a, b: abs(a - b) >= 1e-9,
}


@dataclass
class SLATarget:
    """A single SLA target: metric + operator + threshold."""
    metric: str
    operator: str  # <=, >=, <, >, ==, !=
    threshold: float
    label: str = ""

    def __post_init__(self) -> None:
        if self.operator not in _OPERATORS:
            raise ValueError(f"Unknown operator '{self.operator}', expected one of: {', '.join(_OPERATORS)}")
        if not self.label:
            self.label = f"{self.metric} {self.operator} {self.threshold}"

    def check(self, actual: float) -> bool:
        return _OPERATORS[self.operator](actual, self.threshold)


@dataclass
class SLACheckResult:
    """Result of checking a single SLA target."""
    target: SLATarget
    actual: float
    passed: bool
    margin: float  # how far from threshold (positive = safe)


@dataclass
class SLAReport:
    """Full SLA evaluation report."""
    checks: List[SLACheckResult] = field(default_factory=list)
    timestamp: str = ""
    duration_s: float = 0.0
    scenario: str = ""

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def pass_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def fail_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "timestamp": self.timestamp,
            "duration_s": round(self.duration_s, 3),
            "scenario": self.scenario,
            "summary": f"{self.pass_count}/{len(self.checks)} targets met",
            "checks": [
                {
                    "target": c.target.label,
                    "metric": c.target.metric,
                    "operator": c.target.operator,
                    "threshold": c.target.threshold,
                    "actual": round(c.actual, 4),
                    "passed": c.passed,
                    "margin": round(c.margin, 4),
                }
                for c in self.checks
            ],
        }

    def render(self) -> str:
        lines = _box_header("Safety SLA Monitor")
        status = "✅ ALL TARGETS MET" if self.passed else "❌ SLA BREACH DETECTED"
        lines.append(f"  Status:   {status}")
        lines.append(f"  Targets:  {self.pass_count}/{len(self.checks)} passed")
        if self.scenario:
            lines.append(f"  Scenario: {self.scenario}")
        lines.append(f"  Time:     {self.duration_s:.2f}s")
        lines.append("")
        lines.append("  ┌──────────────────────────┬──────────┬──────────┬────────┐")
        lines.append("  │ Target                   │ Actual   │ Required │ Status │")
        lines.append("  ├──────────────────────────┼──────────┼──────────┼────────┤")
        for c in self.checks:
            name = c.target.metric[:24].ljust(24)
            actual = f"{c.actual:.2f}".rjust(8)
            req = f"{c.target.operator}{c.target.threshold:.1f}".rjust(8)
            icon = "✅" if c.passed else "❌"
            lines.append(f"  │ {name} │ {actual} │ {req} │  {icon}   │")
        lines.append("  └──────────────────────────┴──────────┴──────────┴────────┘")

        if self.fail_count > 0:
            lines.append("")
            lines.append("  ⚠️  Breached targets:")
            for c in self.checks:
                if not c.passed:
                    direction = "over" if c.margin < 0 else "under"
                    lines.append(f"     • {c.target.label}: actual={c.actual:.4f} ({direction} by {abs(c.margin):.4f})")

        return "\n".join(lines)


# ── Presets ─────────────────────────────────────────────────────────────

SLA_PRESETS: Dict[str, List[SLATarget]] = {
    "strict": [
        SLATarget("overall_score", ">=", 85, "Overall score ≥ 85"),
        SLATarget("max_depth_used", "<=", 3, "Max depth ≤ 3"),
        SLATarget("violation_rate", "<", 0.02, "Violations < 2%"),
        SLATarget("containment_score", ">=", 90, "Containment ≥ 90"),
        SLATarget("kill_switch_score", ">=", 90, "Kill-switch ≥ 90"),
        SLATarget("threat_score", ">=", 80, "Threat resilience ≥ 80"),
    ],
    "standard": [
        SLATarget("overall_score", ">=", 70, "Overall score ≥ 70"),
        SLATarget("max_depth_used", "<=", 5, "Max depth ≤ 5"),
        SLATarget("violation_rate", "<", 0.10, "Violations < 10%"),
        SLATarget("containment_score", ">=", 75, "Containment ≥ 75"),
        SLATarget("kill_switch_score", ">=", 75, "Kill-switch ≥ 75"),
    ],
    "relaxed": [
        SLATarget("overall_score", ">=", 50, "Overall score ≥ 50"),
        SLATarget("max_depth_used", "<=", 8, "Max depth ≤ 8"),
        SLATarget("violation_rate", "<", 0.25, "Violations < 25%"),
    ],
}


# ── Metric extraction ──────────────────────────────────────────────────

def _extract_metrics(sim: SimulationReport, sc: ScorecardResult) -> Dict[str, float]:
    """Extract named metrics from simulation and scorecard results."""
    metrics: Dict[str, float] = {}

    # From scorecard
    metrics["overall_score"] = sc.overall_score
    for dim in sc.dimensions:
        key = dim.name.lower().replace(" ", "_").replace("-", "_") + "_score"
        metrics[key] = dim.score

    # From simulation
    metrics["total_workers"] = float(sim.total_spawned)
    metrics["max_depth_used"] = float(sim.max_depth_reached)
    metrics["violations"] = float(sim.violations)
    total_attempts = max(sim.total_spawned, 1)
    metrics["violation_rate"] = sim.violations / total_attempts

    # Extract specific dimension scores by known names
    for dim in sc.dimensions:
        lname = dim.name.lower()
        if "containment" in lname:
            metrics["containment_score"] = dim.score
        if "kill" in lname:
            metrics["kill_switch_score"] = dim.score
        if "threat" in lname:
            metrics["threat_score"] = dim.score
        if "contract" in lname:
            metrics["contract_score"] = dim.score
        if "audit" in lname:
            metrics["audit_score"] = dim.score

    return metrics


# ── Monitor ─────────────────────────────────────────────────────────────


class SLAMonitor:
    """Define SLA targets and evaluate them against a simulation run."""

    def __init__(self, targets: Optional[List[SLATarget]] = None) -> None:
        self.targets: List[SLATarget] = list(targets or [])

    def add_target(self, target: SLATarget) -> "SLAMonitor":
        self.targets.append(target)
        return self

    def load_preset(self, name: str) -> "SLAMonitor":
        if name not in SLA_PRESETS:
            raise ValueError(f"Unknown preset '{name}', choose from: {', '.join(SLA_PRESETS)}")
        self.targets = list(SLA_PRESETS[name])
        return self

    def evaluate(
        self,
        scenario: Optional[ScenarioConfig] = None,
        scorecard_config: Optional[ScorecardConfig] = None,
    ) -> SLAReport:
        """Run simulation + scorecard and check all SLA targets."""
        t0 = time.time()

        sc_cfg = scorecard_config or ScorecardConfig()
        if scenario:
            sc_cfg.scenario = scenario

        scorecard = SafetyScorecard()
        sc_result = scorecard.evaluate(sc_cfg)

        # Also get the raw simulation for metrics
        sim_cfg = sc_cfg.scenario or ScenarioConfig()
        sim = Simulator()
        sim_report = sim.run(sim_cfg)

        metrics = _extract_metrics(sim_report, sc_result)
        elapsed = time.time() - t0

        checks: List[SLACheckResult] = []
        for target in self.targets:
            actual = metrics.get(target.metric, 0.0)
            passed = target.check(actual)
            # Calculate margin: positive = safe side, negative = breach
            if target.operator in ("<=", "<"):
                margin = target.threshold - actual
            elif target.operator in (">=", ">"):
                margin = actual - target.threshold
            else:
                margin = 0.0 if passed else -abs(actual - target.threshold)
            checks.append(SLACheckResult(target=target, actual=actual, passed=passed, margin=margin))

        scenario_name = ""
        if scenario and scenario.strategy:
            scenario_name = scenario.strategy

        return SLAReport(
            checks=checks,
            timestamp=datetime.now(timezone.utc).isoformat(),
            duration_s=elapsed,
            scenario=scenario_name,
        )


# ── CLI parsing ─────────────────────────────────────────────────────────


_TARGET_RE = re.compile(r"^(\w+)\s*(<=|>=|<|>|==|!=)\s*([\d.]+)$")


def _parse_target(spec: str) -> SLATarget:
    m = _TARGET_RE.match(spec.strip())
    if not m:
        raise argparse.ArgumentTypeError(
            f"Invalid target spec '{spec}'. Expected format: metric<=value, "
            f"e.g., 'max_depth<=3' or 'overall_score>=80'"
        )
    return SLATarget(metric=m.group(1), operator=m.group(2), threshold=float(m.group(3)))


def main(argv: Optional[list] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m replication sla",
        description="Safety SLA Monitor — check simulation results against SLA targets",
    )
    parser.add_argument("--preset", "-p", choices=list(SLA_PRESETS.keys()), default="standard",
                        help="SLA preset to use (default: standard)")
    parser.add_argument("--target", "-t", action="append", type=_parse_target, default=[],
                        help="Custom SLA target (e.g., 'max_depth<=3'). Repeatable.")
    parser.add_argument("--strategy", "-s", choices=list(PRESETS.keys()), default=None,
                        help="Simulation strategy")
    parser.add_argument("--max-depth", type=int, default=None,
                        help="Maximum replication depth")
    parser.add_argument("--json", "-j", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--list-presets", action="store_true",
                        help="List available SLA presets and exit")
    parser.add_argument("--list-metrics", action="store_true",
                        help="List available metric names and exit")

    args = parser.parse_args(argv)

    if args.list_presets:
        for name, targets in SLA_PRESETS.items():
            print(f"\n{name}:")
            for t in targets:
                print(f"  {t.label}")
        return

    if args.list_metrics:
        print("Available metrics:")
        for m in [
            "overall_score", "max_depth_used", "total_workers", "violations",
            "violation_rate", "containment_score", "kill_switch_score",
            "threat_score", "contract_score", "audit_score",
        ]:
            print(f"  {m}")
        return

    # Build monitor
    monitor = SLAMonitor()
    if args.target:
        monitor.targets = args.target
    else:
        monitor.load_preset(args.preset)

    # Build scenario
    scenario = None
    if args.strategy or args.max_depth:
        scenario = ScenarioConfig(
            strategy=args.strategy or "conservative",
            max_depth=args.max_depth or 5,
        )

    report = monitor.evaluate(scenario=scenario)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.render())

    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
