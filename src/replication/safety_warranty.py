"""Safety Warranty — formal safety guarantees with monitoring, breach detection & reports.

Define safety warranties (formal promises about system behavior) with
explicit conditions, evidence requirements, validity periods, and breach
consequences.  Unlike SLAs (which track numeric metric thresholds), warranties
are richer: each warranty has a *scope* (what component/behavior is covered),
*conditions* (prerequisites for the warranty to be valid), *exclusions*
(when the warranty is void), and a *remediation plan* if breached.

Use-cases:

* Stakeholder communication — "We guarantee max replication depth ≤ 3
  under normal operating conditions"
* Audit artefacts — generate warranty reports showing compliance status
* Continuous monitoring — evaluate warranties against live simulation data
* Breach notification — detect and document warranty violations with evidence

Usage (CLI)::

    python -m replication warranty                         # evaluate all default warranties
    python -m replication warranty --list                  # list available warranty templates
    python -m replication warranty --preset standard       # standard warranty set
    python -m replication warranty --preset strict         # strict warranty set
    python -m replication warranty --preset minimal        # minimal warranty set
    python -m replication warranty --add "no_runaway: max_depth <= 3"
    python -m replication warranty --json                  # JSON output
    python -m replication warranty --html -o warranty.html # HTML report
    python -m replication warranty --strategy greedy       # simulate with specific strategy
    python -m replication warranty --verbose               # detailed evidence output

Programmatic::

    from replication.safety_warranty import (
        WarrantyManager, Warranty, WarrantyCondition, WarrantyReport
    )
    mgr = WarrantyManager()
    mgr.add_warranty(Warranty(
        name="no_runaway_replication",
        description="System will not exceed replication depth 3",
        scope="replication_control",
        metric="max_depth",
        operator="<=",
        threshold=3.0,
        conditions=["kill_switch_enabled", "resource_limits_active"],
        exclusions=["chaos_testing_mode"],
        remediation="Activate kill switch and quarantine affected workers",
    ))
    report = mgr.evaluate()
    print(report.render())
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .simulator import Simulator, ScenarioConfig, SimulationReport, PRESETS
from .scorecard import SafetyScorecard, ScorecardConfig, ScorecardResult
from ._helpers import box_header as _box_header


# ── Enums ────────────────────────────────────────────────────────────


class WarrantyStatus(Enum):
    """Status of a warranty evaluation."""
    VALID = "valid"           # warranty holds
    BREACHED = "breached"     # warranty violated
    VOID = "void"             # conditions not met / exclusion triggered
    EXPIRED = "expired"       # past validity period
    PENDING = "pending"       # not yet evaluated


class Severity(Enum):
    """Breach severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ── Operators ────────────────────────────────────────────────────────

_OPERATORS = {
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
    "<":  lambda a, b: a < b,
    ">":  lambda a, b: a > b,
    "==": lambda a, b: abs(a - b) < 1e-9,
    "!=": lambda a, b: abs(a - b) >= 1e-9,
}


# ── Data classes ─────────────────────────────────────────────────────


@dataclass
class WarrantyCondition:
    """A prerequisite condition for a warranty to be valid."""
    name: str
    description: str = ""
    check_fn: Optional[Any] = None  # callable(metrics) -> bool
    met: bool = True


@dataclass
class Warranty:
    """A formal safety guarantee."""
    name: str
    description: str
    scope: str                          # component/subsystem covered
    metric: str                         # metric to evaluate
    operator: str                       # comparison operator
    threshold: float                    # threshold value
    conditions: List[str] = field(default_factory=list)
    exclusions: List[str] = field(default_factory=list)
    remediation: str = ""               # what to do on breach
    severity: Severity = Severity.HIGH
    valid_from: Optional[str] = None    # ISO timestamp
    valid_until: Optional[str] = None   # ISO timestamp
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.operator not in _OPERATORS:
            raise ValueError(
                f"Unknown operator '{self.operator}', "
                f"expected one of: {', '.join(_OPERATORS)}"
            )

    def check_value(self, actual: float) -> bool:
        """Check if actual value satisfies the warranty threshold."""
        return _OPERATORS[self.operator](actual, self.threshold)

    def is_expired(self) -> bool:
        """Check if warranty has expired."""
        if not self.valid_until:
            return False
        try:
            until = datetime.fromisoformat(self.valid_until)
            return datetime.now(timezone.utc) > until
        except (ValueError, TypeError):
            return False


@dataclass
class WarrantyEvaluation:
    """Result of evaluating a single warranty."""
    warranty: Warranty
    status: WarrantyStatus
    actual_value: Optional[float] = None
    breach_margin: float = 0.0          # how far past threshold
    evidence: Dict[str, Any] = field(default_factory=dict)
    conditions_met: Dict[str, bool] = field(default_factory=dict)
    exclusions_triggered: List[str] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class WarrantyReport:
    """Aggregated warranty evaluation report."""
    evaluations: List[WarrantyEvaluation] = field(default_factory=list)
    timestamp: str = ""
    strategy: str = "balanced"
    simulation_summary: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    @property
    def total(self) -> int:
        return len(self.evaluations)

    @property
    def valid_count(self) -> int:
        return sum(1 for e in self.evaluations if e.status == WarrantyStatus.VALID)

    @property
    def breached_count(self) -> int:
        return sum(1 for e in self.evaluations if e.status == WarrantyStatus.BREACHED)

    @property
    def void_count(self) -> int:
        return sum(1 for e in self.evaluations if e.status == WarrantyStatus.VOID)

    @property
    def all_valid(self) -> bool:
        return all(
            e.status in (WarrantyStatus.VALID, WarrantyStatus.VOID)
            for e in self.evaluations
        )

    def render(self, verbose: bool = False) -> str:
        """Render a human-readable warranty report."""
        lines: List[str] = []
        lines.append(_box_header("Safety Warranty Report"))
        lines.append(f"Evaluated: {self.timestamp}")
        lines.append(f"Strategy:  {self.strategy}")
        lines.append(f"Total:     {self.total} warranties")
        lines.append("")

        # summary bar
        valid_pct = (self.valid_count / self.total * 100) if self.total else 0
        bar_width = 30
        filled = int(valid_pct / 100 * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        status_icon = "✅" if self.all_valid else "❌"
        lines.append(f"  {status_icon} Compliance: [{bar}] {valid_pct:.0f}%")
        lines.append(f"     Valid: {self.valid_count}  |  Breached: {self.breached_count}  |  Void: {self.void_count}")
        lines.append("")

        # per-warranty details
        for ev in self.evaluations:
            w = ev.warranty
            if ev.status == WarrantyStatus.VALID:
                icon = "✅"
            elif ev.status == WarrantyStatus.BREACHED:
                icon = "❌"
            elif ev.status == WarrantyStatus.VOID:
                icon = "⚪"
            else:
                icon = "⏳"

            actual_str = f"{ev.actual_value:.2f}" if ev.actual_value is not None else "N/A"
            lines.append(f"  {icon} {w.name}")
            lines.append(f"     {w.description}")
            lines.append(f"     Scope: {w.scope}  |  {w.metric} {w.operator} {w.threshold}  |  Actual: {actual_str}")

            if ev.status == WarrantyStatus.BREACHED:
                lines.append(f"     ⚠ Breach margin: {ev.breach_margin:+.2f}  |  Severity: {w.severity.value}")
                if w.remediation:
                    lines.append(f"     🔧 Remediation: {w.remediation}")

            if ev.status == WarrantyStatus.VOID and ev.exclusions_triggered:
                lines.append(f"     Void reason: exclusions triggered — {', '.join(ev.exclusions_triggered)}")

            if verbose and ev.conditions_met:
                cond_str = ", ".join(
                    f"{k}: {'✓' if v else '✗'}" for k, v in ev.conditions_met.items()
                )
                lines.append(f"     Conditions: {cond_str}")

            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to JSON-friendly dict."""
        return {
            "timestamp": self.timestamp,
            "strategy": self.strategy,
            "summary": {
                "total": self.total,
                "valid": self.valid_count,
                "breached": self.breached_count,
                "void": self.void_count,
                "all_valid": self.all_valid,
            },
            "simulation": self.simulation_summary,
            "warranties": [
                {
                    "name": ev.warranty.name,
                    "description": ev.warranty.description,
                    "scope": ev.warranty.scope,
                    "metric": ev.warranty.metric,
                    "operator": ev.warranty.operator,
                    "threshold": ev.warranty.threshold,
                    "severity": ev.warranty.severity.value,
                    "status": ev.status.value,
                    "actual_value": ev.actual_value,
                    "breach_margin": ev.breach_margin,
                    "conditions_met": ev.conditions_met,
                    "exclusions_triggered": ev.exclusions_triggered,
                    "remediation": ev.warranty.remediation,
                    "tags": ev.warranty.tags,
                }
                for ev in self.evaluations
            ],
        }

    def to_html(self) -> str:
        """Generate a self-contained HTML warranty report."""
        d = self.to_dict()
        status_color = "#22c55e" if self.all_valid else "#ef4444"
        status_text = "ALL WARRANTIES VALID" if self.all_valid else f"{self.breached_count} BREACH(ES) DETECTED"

        rows = []
        for w in d["warranties"]:
            if w["status"] == "valid":
                bg, icon = "#f0fdf4", "✅"
            elif w["status"] == "breached":
                bg, icon = "#fef2f2", "❌"
            else:
                bg, icon = "#f9fafb", "⚪"
            actual = f"{w['actual_value']:.2f}" if w["actual_value"] is not None else "N/A"
            remedy = f"<br><small>🔧 {w['remediation']}</small>" if w["status"] == "breached" and w["remediation"] else ""
            rows.append(
                f'<tr style="background:{bg}">'
                f'<td>{icon} {w["name"]}</td>'
                f'<td>{w["description"]}</td>'
                f'<td>{w["scope"]}</td>'
                f'<td><code>{w["metric"]} {w["operator"]} {w["threshold"]}</code></td>'
                f'<td><b>{actual}</b></td>'
                f'<td><b>{w["status"].upper()}</b>{remedy}</td>'
                f'<td>{w["severity"]}</td>'
                f"</tr>"
            )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Safety Warranty Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem; background: #f8fafc; }}
  h1 {{ color: #1e293b; }}
  .banner {{ background: {status_color}; color: white; padding: 1rem 2rem; border-radius: 8px; font-size: 1.2rem; margin-bottom: 1.5rem; }}
  .stats {{ display: flex; gap: 2rem; margin-bottom: 1.5rem; }}
  .stat {{ background: white; padding: 1rem 1.5rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .stat .num {{ font-size: 2rem; font-weight: bold; color: #1e293b; }}
  .stat .label {{ color: #64748b; font-size: 0.85rem; }}
  table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  th {{ background: #1e293b; color: white; padding: 0.75rem 1rem; text-align: left; font-weight: 500; }}
  td {{ padding: 0.75rem 1rem; border-bottom: 1px solid #e2e8f0; }}
  code {{ background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-size: 0.85rem; }}
  .footer {{ margin-top: 2rem; color: #94a3b8; font-size: 0.8rem; }}
</style>
</head>
<body>
<h1>🛡️ Safety Warranty Report</h1>
<div class="banner">{status_text}</div>
<div class="stats">
  <div class="stat"><div class="num">{d['summary']['total']}</div><div class="label">Total Warranties</div></div>
  <div class="stat"><div class="num" style="color:#22c55e">{d['summary']['valid']}</div><div class="label">Valid</div></div>
  <div class="stat"><div class="num" style="color:#ef4444">{d['summary']['breached']}</div><div class="label">Breached</div></div>
  <div class="stat"><div class="num" style="color:#94a3b8">{d['summary']['void']}</div><div class="label">Void</div></div>
</div>
<table>
<tr><th>Warranty</th><th>Description</th><th>Scope</th><th>Threshold</th><th>Actual</th><th>Status</th><th>Severity</th></tr>
{''.join(rows)}
</table>
<div class="footer">Generated {d['timestamp']} | Strategy: {d['strategy']}</div>
</body>
</html>"""


# ── Warranty Presets ────────────────────────────────────────────────


def _standard_warranties() -> List[Warranty]:
    """Standard set of safety warranties."""
    return [
        Warranty(
            name="no_runaway_replication",
            description="Replication depth will not exceed safe limits",
            scope="replication_control",
            metric="max_depth",
            operator="<=",
            threshold=4.0,
            severity=Severity.CRITICAL,
            conditions=["kill_switch_enabled"],
            remediation="Activate kill switch, quarantine workers exceeding depth limit",
            tags=["replication", "core"],
        ),
        Warranty(
            name="resource_containment",
            description="Workers will not consume excessive resources",
            scope="resource_management",
            metric="peak_workers",
            operator="<=",
            threshold=25.0,
            severity=Severity.HIGH,
            remediation="Enforce resource caps and terminate excess workers",
            tags=["resources", "core"],
        ),
        Warranty(
            name="safety_score_minimum",
            description="Overall safety score will meet minimum threshold",
            scope="safety_posture",
            metric="overall_score",
            operator=">=",
            threshold=70.0,
            severity=Severity.HIGH,
            remediation="Review and tighten safety policy configuration",
            tags=["scorecard", "core"],
        ),
        Warranty(
            name="contract_compliance",
            description="Contract violation rate will remain below acceptable level",
            scope="contract_enforcement",
            metric="violation_rate",
            operator="<",
            threshold=0.10,
            severity=Severity.MEDIUM,
            remediation="Audit contract definitions and enforcement pipeline",
            tags=["contracts"],
        ),
        Warranty(
            name="kill_switch_responsiveness",
            description="Kill switch will activate within acceptable timeframe",
            scope="emergency_controls",
            metric="kill_switch_score",
            operator=">=",
            threshold=80.0,
            severity=Severity.CRITICAL,
            remediation="Test kill switch mechanism and reduce activation latency",
            tags=["kill_switch", "core"],
        ),
        Warranty(
            name="data_integrity",
            description="Agent data operations will maintain integrity guarantees",
            scope="data_protection",
            metric="data_integrity_score",
            operator=">=",
            threshold=85.0,
            severity=Severity.HIGH,
            remediation="Enable data checksums and audit logging",
            tags=["data"],
        ),
    ]


def _strict_warranties() -> List[Warranty]:
    """Strict warranty set with tighter thresholds."""
    ws = _standard_warranties()
    overrides = {
        "no_runaway_replication": 2.0,
        "resource_containment": 15.0,
        "safety_score_minimum": 85.0,
        "contract_compliance": 0.03,
        "kill_switch_responsiveness": 90.0,
        "data_integrity": 95.0,
    }
    for w in ws:
        if w.name in overrides:
            w.threshold = overrides[w.name]
            w.severity = Severity.CRITICAL
    return ws


def _minimal_warranties() -> List[Warranty]:
    """Minimal warranty set — only critical guarantees."""
    return [w for w in _standard_warranties() if "core" in w.tags]


WARRANTY_PRESETS: Dict[str, List[Warranty]] = {
    "standard": _standard_warranties(),
    "strict": _strict_warranties(),
    "minimal": _minimal_warranties(),
}


# ── Warranty Manager ─────────────────────────────────────────────────


class WarrantyManager:
    """Manages a set of warranties and evaluates them against simulation data."""

    def __init__(self, warranties: Optional[List[Warranty]] = None) -> None:
        self._warranties: List[Warranty] = list(warranties) if warranties else []

    def add_warranty(self, warranty: Warranty) -> None:
        self._warranties.append(warranty)

    def load_preset(self, name: str) -> None:
        """Load a named warranty preset."""
        if name not in WARRANTY_PRESETS:
            raise ValueError(f"Unknown preset '{name}', options: {', '.join(WARRANTY_PRESETS)}")
        self._warranties = list(WARRANTY_PRESETS[name])

    @property
    def warranties(self) -> List[Warranty]:
        return list(self._warranties)

    def _extract_metrics(
        self,
        sim_report: Optional[SimulationReport],
        sc_result: Optional[ScorecardResult],
    ) -> Dict[str, float]:
        """Extract metric values from simulation/scorecard results."""
        metrics: Dict[str, float] = {}

        if sim_report is not None:
            metrics["max_depth"] = float(getattr(sim_report, "max_depth_reached", 0))
            metrics["peak_workers"] = float(getattr(sim_report, "peak_workers", 0))
            metrics["total_replications"] = float(getattr(sim_report, "total_replications", 0))
            metrics["kill_switch_triggered"] = 1.0 if getattr(sim_report, "kill_switch_triggered", False) else 0.0

        if sc_result is not None:
            metrics["overall_score"] = float(getattr(sc_result, "overall_score", 0))
            for cat_name, cat_score in getattr(sc_result, "category_scores", {}).items():
                safe_name = cat_name.lower().replace(" ", "_").replace("-", "_")
                metrics[f"{safe_name}_score"] = float(cat_score)
            # Map common scorecard fields
            if hasattr(sc_result, "details") and isinstance(sc_result.details, dict):
                for k, v in sc_result.details.items():
                    if isinstance(v, (int, float)):
                        metrics[k] = float(v)

        # Derived metrics
        total_ops = metrics.get("total_replications", 1.0)
        if total_ops > 0:
            violations = metrics.get("contract_violations", 0.0)
            metrics["violation_rate"] = violations / total_ops

        return metrics

    def evaluate(
        self,
        sim_report: Optional[SimulationReport] = None,
        sc_result: Optional[ScorecardResult] = None,
        metrics: Optional[Dict[str, float]] = None,
        strategy: str = "balanced",
        active_conditions: Optional[List[str]] = None,
        active_exclusions: Optional[List[str]] = None,
    ) -> WarrantyReport:
        """Evaluate all warranties against provided data."""
        if metrics is None:
            metrics = self._extract_metrics(sim_report, sc_result)

        if active_conditions is None:
            active_conditions = [
                "kill_switch_enabled",
                "resource_limits_active",
                "monitoring_active",
            ]
        if active_exclusions is None:
            active_exclusions = []

        evaluations: List[WarrantyEvaluation] = []

        for w in self._warranties:
            # Check expiry
            if w.is_expired():
                evaluations.append(WarrantyEvaluation(
                    warranty=w,
                    status=WarrantyStatus.EXPIRED,
                    evidence={"reason": "Warranty validity period has expired"},
                ))
                continue

            # Check exclusions
            triggered_exclusions = [e for e in w.exclusions if e in active_exclusions]
            if triggered_exclusions:
                evaluations.append(WarrantyEvaluation(
                    warranty=w,
                    status=WarrantyStatus.VOID,
                    exclusions_triggered=triggered_exclusions,
                    evidence={"reason": "Exclusion conditions triggered"},
                ))
                continue

            # Check conditions
            cond_results = {}
            for c in w.conditions:
                cond_results[c] = c in active_conditions
            if cond_results and not all(cond_results.values()):
                evaluations.append(WarrantyEvaluation(
                    warranty=w,
                    status=WarrantyStatus.VOID,
                    conditions_met=cond_results,
                    evidence={"reason": "Required conditions not met"},
                ))
                continue

            # Evaluate metric
            actual = metrics.get(w.metric)
            if actual is None:
                evaluations.append(WarrantyEvaluation(
                    warranty=w,
                    status=WarrantyStatus.VOID,
                    conditions_met=cond_results,
                    evidence={"reason": f"Metric '{w.metric}' not available"},
                ))
                continue

            passed = w.check_value(actual)
            margin = actual - w.threshold

            evaluations.append(WarrantyEvaluation(
                warranty=w,
                status=WarrantyStatus.VALID if passed else WarrantyStatus.BREACHED,
                actual_value=actual,
                breach_margin=margin if not passed else 0.0,
                conditions_met=cond_results,
                evidence={
                    "metric": w.metric,
                    "operator": w.operator,
                    "threshold": w.threshold,
                    "actual": actual,
                    "passed": passed,
                },
            ))

        sim_summary = {}
        if sim_report is not None:
            sim_summary["strategy"] = strategy
            sim_summary["max_depth"] = getattr(sim_report, "max_depth_reached", None)
            sim_summary["peak_workers"] = getattr(sim_report, "peak_workers", None)

        return WarrantyReport(
            evaluations=evaluations,
            strategy=strategy,
            simulation_summary=sim_summary,
        )


# ── CLI ──────────────────────────────────────────────────────────────


def _parse_warranty_spec(spec: str) -> Warranty:
    """Parse 'name: metric op threshold' into a Warranty."""
    import re
    m = re.match(r"(\w+)\s*:\s*(\w+)\s*(<=|>=|<|>|==|!=)\s*([\d.]+)", spec.strip())
    if not m:
        raise ValueError(
            f"Invalid warranty spec: '{spec}'. "
            "Expected format: 'name: metric <= 3.0'"
        )
    name, metric, op, thresh = m.groups()
    return Warranty(
        name=name,
        description=f"Custom warranty: {metric} {op} {thresh}",
        scope="custom",
        metric=metric,
        operator=op,
        threshold=float(thresh),
    )


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for safety warranty evaluation."""
    parser = argparse.ArgumentParser(
        prog="replication warranty",
        description="Safety Warranty — evaluate formal safety guarantees",
    )
    parser.add_argument(
        "--preset", "-p",
        choices=list(WARRANTY_PRESETS.keys()),
        default="standard",
        help="Warranty preset to use (default: standard)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        dest="list_warranties",
        help="List available warranty templates",
    )
    parser.add_argument(
        "--add", "-a",
        action="append",
        default=[],
        dest="custom",
        help="Add custom warranty spec: 'name: metric <= 3.0'",
    )
    parser.add_argument(
        "--strategy", "-s",
        default="balanced",
        choices=list(PRESETS.keys()),
        help="Simulation strategy preset (default: balanced)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Override max replication depth",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--html", action="store_true", help="Output HTML")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Write output to file instead of stdout",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed evidence")

    args = parser.parse_args(argv)

    # List mode
    if args.list_warranties:
        print(_box_header("Available Warranty Presets"))
        for preset_name, warranties in WARRANTY_PRESETS.items():
            print(f"\n  📋 {preset_name} ({len(warranties)} warranties):")
            for w in warranties:
                sev_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(w.severity.value, "⚪")
                print(f"     {sev_icon} {w.name}: {w.metric} {w.operator} {w.threshold}")
                print(f"        {w.description}")
        return

    # Build manager
    mgr = WarrantyManager()
    mgr.load_preset(args.preset)

    for spec in args.custom:
        mgr.add_warranty(_parse_warranty_spec(spec))

    # Run simulation
    cfg_overrides: Dict[str, Any] = {}
    if args.max_depth is not None:
        cfg_overrides["max_depth"] = args.max_depth

    preset = dict(PRESETS[args.strategy])
    preset.update(cfg_overrides)
    config = ScenarioConfig(**preset)
    sim = Simulator(config)
    sim_report = sim.run()

    sc_cfg = ScorecardConfig(strategy=args.strategy)
    sc = SafetyScorecard(sc_cfg)
    sc_result = sc.evaluate(sim_report)

    # Evaluate warranties
    report = mgr.evaluate(
        sim_report=sim_report,
        sc_result=sc_result,
        strategy=args.strategy,
    )

    # Output
    if args.json:
        output = json.dumps(report.to_dict(), indent=2)
    elif args.html:
        output = report.to_html()
    else:
        output = report.render(verbose=args.verbose)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Written to {args.output}")
    else:
        print(output)

    # Exit code: non-zero if any breach
    if report.breached_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
