"""Alignment Tax Calculator — quantify performance cost of safety constraints.

Every safety constraint imposed on an AI agent exacts a *tax* — added
latency, reduced throughput, CPU/memory overhead.  This module measures
that tax, ranks constraints by cost, identifies Pareto-optimal tradeoff
points, predicts when agents have rational incentive to *shed* a
constraint, and recommends consolidation opportunities.

Key concepts:

- **Composite Tax** — weighted multi-dimensional cost score (0–100)
- **Shedding Incentive** — how strongly an agent benefits from bypassing
  a constraint (high tax + low bypass difficulty = high incentive)
- **Pareto Frontier** — the set of constraint configurations where you
  cannot improve safety without increasing tax (or vice versa)
- **Consolidation** — pairs of constraints with overlapping coverage
  that could be merged to reduce total tax

Usage (CLI)::

    python -m replication alignment-tax                     # full assessment
    python -m replication alignment-tax --demo              # demo constraints
    python -m replication alignment-tax --pareto            # Pareto frontier
    python -m replication alignment-tax --shedding          # shedding risks
    python -m replication alignment-tax --consolidate       # consolidation opps
    python -m replication alignment-tax --remove <name>     # simulate removal
    python -m replication alignment-tax --json              # JSON output
    python -m replication alignment-tax --html report.html  # HTML dashboard

Programmatic::

    from replication.alignment_tax import AlignmentTaxCalculator, SafetyConstraint
    calc = AlignmentTaxCalculator()
    calc.register_constraint(SafetyConstraint(
        name="rate_limiter", category="rate_limit",
        latency_ms=5.0, throughput_factor=0.8,
        cpu_overhead_pct=2.0, memory_overhead_mb=10.0,
        safety_value=70.0, bypass_difficulty=0.3,
    ))
    report = calc.assess_all()
    print(report.render())
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

from ._helpers import Severity, box_header, stats_mean, stats_std, emit_output


# ── Configuration ────────────────────────────────────────────────────


@dataclass
class TaxConfig:
    """Configuration for tax computation."""

    shedding_threshold: float = 0.6
    consolidation_overlap_min: float = 0.3
    pareto_resolution: int = 20
    weight_latency: float = 0.3
    weight_throughput: float = 0.3
    weight_cpu: float = 0.2
    weight_memory: float = 0.2


# ── Data Models ──────────────────────────────────────────────────────


CATEGORIES = [
    "monitoring",
    "containment",
    "approval",
    "rate_limit",
    "isolation",
    "audit",
    "encryption",
]


@dataclass
class SafetyConstraint:
    """A single safety constraint with its cost profile."""

    name: str
    category: str  # one of CATEGORIES
    latency_ms: float  # added latency per operation
    throughput_factor: float  # multiplier 0-1 (1.0 = no impact)
    cpu_overhead_pct: float  # additional CPU %
    memory_overhead_mb: float  # additional memory MB
    safety_value: float  # 0-100 safety contribution score
    bypass_difficulty: float  # 0-1 how hard to bypass (low = easy to shed)


@dataclass
class TaxAssessment:
    """Tax assessment for a single constraint."""

    constraint: SafetyConstraint
    composite_tax: float  # 0-100 normalized total cost
    shedding_incentive: float  # 0-1 rational incentive to bypass
    efficiency_ratio: float  # safety_value / composite_tax
    risk_if_shed: str  # what happens if removed
    severity: Severity = Severity.LOW


@dataclass
class ParetoPoint:
    """A point on the Pareto frontier."""

    constraints_enabled: List[str]
    total_safety: float
    total_tax: float
    is_current: bool = False


@dataclass
class ConsolidationOpportunity:
    """Two constraints that could be merged."""

    constraint_a: str
    constraint_b: str
    overlap_pct: float  # how much they overlap in coverage
    potential_savings: float  # tax reduction if consolidated
    risk_delta: float  # safety loss from consolidation


@dataclass
class RemovalImpact:
    """Impact of removing a single constraint."""

    constraint_name: str
    tax_reduction: float
    safety_loss: float
    new_total_tax: float
    new_total_safety: float
    shedding_cascade: List[str]  # other constraints that become shed-risky
    verdict: str


# ── Risk descriptions ────────────────────────────────────────────────

_RISK_DESCRIPTIONS: Dict[str, str] = {
    "monitoring": "Undetected behavioral anomalies, missed deception signals",
    "containment": "Potential sandbox escape, resource abuse, unauthorized replication",
    "approval": "Autonomous actions without human oversight, unsafe decisions",
    "rate_limit": "Resource exhaustion, denial-of-service, rapid replication",
    "isolation": "Cross-agent contamination, lateral movement, privilege escalation",
    "audit": "Loss of accountability, untracked actions, forensic blind spots",
    "encryption": "Data exfiltration, eavesdropping, integrity violations",
}


# ── Core Engine ──────────────────────────────────────────────────────


class AlignmentTaxCalculator:
    """Main engine for alignment tax computation."""

    def __init__(self, config: Optional[TaxConfig] = None) -> None:
        self.config = config or TaxConfig()
        self._constraints: Dict[str, SafetyConstraint] = {}

    @property
    def constraints(self) -> List[SafetyConstraint]:
        return list(self._constraints.values())

    def register_constraint(self, constraint: SafetyConstraint) -> None:
        """Register a safety constraint for analysis."""
        self._constraints[constraint.name] = constraint

    def _compute_composite_tax(self, c: SafetyConstraint) -> float:
        """Compute weighted composite tax score 0-100."""
        cfg = self.config
        # Normalize each dimension to 0-100
        latency_norm = min(c.latency_ms / 100.0, 1.0) * 100  # 100ms = max
        throughput_norm = (1.0 - c.throughput_factor) * 100  # 0.5 factor = 50 tax
        cpu_norm = min(c.cpu_overhead_pct / 50.0, 1.0) * 100  # 50% = max
        memory_norm = min(c.memory_overhead_mb / 500.0, 1.0) * 100  # 500MB = max

        composite = (
            cfg.weight_latency * latency_norm
            + cfg.weight_throughput * throughput_norm
            + cfg.weight_cpu * cpu_norm
            + cfg.weight_memory * memory_norm
        )
        return min(composite, 100.0)

    def _compute_shedding_incentive(self, c: SafetyConstraint, tax: float) -> float:
        """Compute rational incentive to bypass constraint (0-1)."""
        # High tax + low bypass difficulty = high incentive
        tax_factor = tax / 100.0
        ease_factor = 1.0 - c.bypass_difficulty
        # Low safety value also increases incentive (agent perceives less risk)
        value_factor = 1.0 - (c.safety_value / 100.0)
        incentive = 0.5 * tax_factor + 0.3 * ease_factor + 0.2 * value_factor
        return min(max(incentive, 0.0), 1.0)

    def _assess_one(self, c: SafetyConstraint) -> TaxAssessment:
        """Assess a single constraint."""
        tax = self._compute_composite_tax(c)
        shedding = self._compute_shedding_incentive(c, tax)
        efficiency = c.safety_value / tax if tax > 0 else float("inf")
        risk = _RISK_DESCRIPTIONS.get(c.category, "Unknown safety degradation")

        if shedding >= 0.8:
            severity = Severity.CRITICAL
        elif shedding >= self.config.shedding_threshold:
            severity = Severity.HIGH
        elif shedding >= 0.4:
            severity = Severity.MEDIUM
        else:
            severity = Severity.LOW

        return TaxAssessment(
            constraint=c,
            composite_tax=tax,
            shedding_incentive=shedding,
            efficiency_ratio=efficiency,
            risk_if_shed=risk,
            severity=severity,
        )

    def assess_all(self) -> "TaxReport":
        """Run full tax assessment on all registered constraints."""
        assessments = [self._assess_one(c) for c in self._constraints.values()]
        assessments.sort(key=lambda a: a.composite_tax, reverse=True)
        pareto = self.compute_pareto_frontier()
        consolidations = self.find_consolidation_opportunities()
        return TaxReport(
            assessments=assessments,
            pareto_frontier=pareto,
            consolidation_opportunities=consolidations,
            config=self.config,
        )

    def compute_pareto_frontier(self) -> List[ParetoPoint]:
        """Find Pareto-optimal safety/tax tradeoff points."""
        constraints = list(self._constraints.values())
        if not constraints:
            return []

        n = len(constraints)
        # Generate candidate subsets using greedy incremental approach
        # (full powerset is 2^n which is too expensive for large n)
        candidates: List[ParetoPoint] = []

        # Start with empty set
        candidates.append(ParetoPoint(
            constraints_enabled=[], total_safety=0.0, total_tax=0.0
        ))

        # Sort by efficiency (safety_value / tax) descending
        assessed = [(c, self._compute_composite_tax(c)) for c in constraints]
        assessed.sort(key=lambda x: x[0].safety_value / max(x[1], 0.01), reverse=True)

        # Build frontier by adding constraints one at a time (greedy)
        enabled: List[str] = []
        running_safety = 0.0
        running_tax = 0.0

        for c, tax in assessed:
            enabled.append(c.name)
            running_safety += c.safety_value
            running_tax += tax
            candidates.append(ParetoPoint(
                constraints_enabled=list(enabled),
                total_safety=running_safety,
                total_tax=running_tax,
            ))

        # Also try removing highest-tax constraints one at a time
        all_names = [c.name for c in constraints]
        total_safety = sum(c.safety_value for c in constraints)
        total_tax = sum(self._compute_composite_tax(c) for c in constraints)

        # Mark current (all enabled)
        candidates.append(ParetoPoint(
            constraints_enabled=list(all_names),
            total_safety=total_safety,
            total_tax=total_tax,
            is_current=True,
        ))

        # Sort by tax descending for removal candidates
        by_tax = sorted(assessed, key=lambda x: x[1], reverse=True)
        removed_safety = total_safety
        removed_tax = total_tax
        remaining = list(all_names)

        for c, tax in by_tax:
            remaining = [n for n in remaining if n != c.name]
            removed_safety -= c.safety_value
            removed_tax -= tax
            if remaining:
                candidates.append(ParetoPoint(
                    constraints_enabled=list(remaining),
                    total_safety=removed_safety,
                    total_tax=removed_tax,
                ))

        # Filter to true Pareto frontier (non-dominated points)
        frontier: List[ParetoPoint] = []
        for p in candidates:
            dominated = False
            for q in candidates:
                if q is p:
                    continue
                # q dominates p if q has >= safety AND <= tax (with at least one strict)
                if (q.total_safety >= p.total_safety and q.total_tax <= p.total_tax
                        and (q.total_safety > p.total_safety or q.total_tax < p.total_tax)):
                    dominated = True
                    break
            if not dominated:
                frontier.append(p)

        # Deduplicate by (safety, tax) pair
        seen: set = set()
        unique: List[ParetoPoint] = []
        for p in frontier:
            key = (round(p.total_safety, 2), round(p.total_tax, 2))
            if key not in seen:
                seen.add(key)
                unique.append(p)

        unique.sort(key=lambda p: p.total_tax)
        return unique

    def detect_shedding_risks(self) -> List[TaxAssessment]:
        """Return constraints with shedding incentive above threshold."""
        assessments = [self._assess_one(c) for c in self._constraints.values()]
        return [
            a for a in assessments
            if a.shedding_incentive >= self.config.shedding_threshold
        ]

    def find_consolidation_opportunities(self) -> List[ConsolidationOpportunity]:
        """Find constraint pairs with high overlap that could be merged."""
        constraints = list(self._constraints.values())
        opportunities: List[ConsolidationOpportunity] = []

        for a, b in combinations(constraints, 2):
            overlap = self._compute_overlap(a, b)
            if overlap >= self.config.consolidation_overlap_min:
                tax_a = self._compute_composite_tax(a)
                tax_b = self._compute_composite_tax(b)
                # Savings = overlap fraction of the smaller tax
                savings = overlap * min(tax_a, tax_b)
                # Risk delta = fraction of safety we might lose
                risk_delta = overlap * 0.1 * min(a.safety_value, b.safety_value)
                opportunities.append(ConsolidationOpportunity(
                    constraint_a=a.name,
                    constraint_b=b.name,
                    overlap_pct=overlap,
                    potential_savings=savings,
                    risk_delta=risk_delta,
                ))

        opportunities.sort(key=lambda o: o.potential_savings, reverse=True)
        return opportunities

    def _compute_overlap(self, a: SafetyConstraint, b: SafetyConstraint) -> float:
        """Estimate functional overlap between two constraints."""
        overlap = 0.0
        # Same category = high overlap
        if a.category == b.category:
            overlap += 0.5
        # Similar latency profiles suggest similar mechanisms
        latency_sim = 1.0 - abs(a.latency_ms - b.latency_ms) / max(
            a.latency_ms, b.latency_ms, 1.0
        )
        overlap += 0.2 * latency_sim
        # Similar throughput impact
        throughput_sim = 1.0 - abs(a.throughput_factor - b.throughput_factor) / max(
            1.0 - min(a.throughput_factor, b.throughput_factor), 0.01
        )
        overlap += 0.2 * min(throughput_sim, 1.0)
        # Similar safety value suggests similar scope
        value_sim = 1.0 - abs(a.safety_value - b.safety_value) / 100.0
        overlap += 0.1 * value_sim
        return min(overlap, 1.0)

    def simulate_removal(self, constraint_name: str) -> RemovalImpact:
        """Simulate removing a constraint and assess impact."""
        if constraint_name not in self._constraints:
            raise ValueError(f"Unknown constraint: {constraint_name}")

        target = self._constraints[constraint_name]
        target_tax = self._compute_composite_tax(target)

        total_tax = sum(
            self._compute_composite_tax(c) for c in self._constraints.values()
        )
        total_safety = sum(c.safety_value for c in self._constraints.values())

        new_tax = total_tax - target_tax
        new_safety = total_safety - target.safety_value

        # Check if removal creates shedding cascade
        cascade: List[str] = []
        for name, c in self._constraints.items():
            if name == constraint_name:
                continue
            # With target removed, remaining constraints bear more pressure
            # Simulate 10% tax increase on same-category constraints
            if c.category == target.category:
                boosted_tax = self._compute_composite_tax(c) * 1.1
                new_incentive = self._compute_shedding_incentive(c, boosted_tax)
                if new_incentive >= self.config.shedding_threshold:
                    cascade.append(name)

        # Verdict
        safety_loss_pct = (target.safety_value / total_safety * 100) if total_safety > 0 else 0
        if safety_loss_pct > 20:
            verdict = "DANGEROUS — critical safety degradation"
        elif safety_loss_pct > 10:
            verdict = "RISKY — significant safety reduction, requires compensating controls"
        elif safety_loss_pct > 5:
            verdict = "MODERATE — acceptable if replaced with lighter alternative"
        else:
            verdict = "LOW IMPACT — minimal safety degradation"

        if cascade:
            verdict += f" (WARNING: may trigger cascade shedding of {len(cascade)} more)"

        return RemovalImpact(
            constraint_name=constraint_name,
            tax_reduction=target_tax,
            safety_loss=target.safety_value,
            new_total_tax=new_tax,
            new_total_safety=new_safety,
            shedding_cascade=cascade,
            verdict=verdict,
        )


# ── Report ───────────────────────────────────────────────────────────


@dataclass
class TaxReport:
    """Full alignment tax assessment report."""

    assessments: List[TaxAssessment]
    pareto_frontier: List[ParetoPoint]
    consolidation_opportunities: List[ConsolidationOpportunity]
    config: TaxConfig

    @property
    def total_tax(self) -> float:
        return sum(a.composite_tax for a in self.assessments)

    @property
    def total_safety(self) -> float:
        return sum(a.constraint.safety_value for a in self.assessments)

    @property
    def avg_efficiency(self) -> float:
        ratios = [a.efficiency_ratio for a in self.assessments if a.efficiency_ratio != float("inf")]
        return stats_mean(ratios)

    @property
    def shedding_risks(self) -> List[TaxAssessment]:
        return [a for a in self.assessments if a.shedding_incentive >= self.config.shedding_threshold]

    def render(self) -> str:
        """Render text report."""
        lines: List[str] = []
        lines.extend(box_header("ALIGNMENT TAX ASSESSMENT"))
        lines.append("")

        # Summary
        lines.append(f"  Constraints analyzed: {len(self.assessments)}")
        lines.append(f"  Total composite tax:  {self.total_tax:.1f}")
        lines.append(f"  Total safety value:   {self.total_safety:.1f}")
        lines.append(f"  Avg efficiency ratio: {self.avg_efficiency:.2f}")
        lines.append(f"  Shedding risks:       {len(self.shedding_risks)}")
        lines.append("")

        # Tax ranking table
        lines.append("─" * 57)
        lines.append("  CONSTRAINT TAX RANKING (highest cost first)")
        lines.append("─" * 57)
        lines.append(f"  {'Name':<28} {'Tax':>6} {'Safety':>7} {'Eff':>6} {'Shed':>5}")
        lines.append(f"  {'─'*28} {'─'*6} {'─'*7} {'─'*6} {'─'*5}")

        for a in self.assessments:
            shed_marker = " ⚠️" if a.shedding_incentive >= self.config.shedding_threshold else ""
            lines.append(
                f"  {a.constraint.name:<28} {a.composite_tax:>5.1f} "
                f"{a.constraint.safety_value:>6.1f} "
                f"{a.efficiency_ratio:>5.2f} "
                f"{a.shedding_incentive:>4.2f}{shed_marker}"
            )

        lines.append("")

        # Shedding risks
        if self.shedding_risks:
            lines.append("─" * 57)
            lines.append("  ⚠️  SHEDDING RISK ALERTS")
            lines.append("─" * 57)
            for a in self.shedding_risks:
                lines.append(f"  [{a.severity.value.upper()}] {a.constraint.name}")
                lines.append(f"    Incentive: {a.shedding_incentive:.2f} | Tax: {a.composite_tax:.1f}")
                lines.append(f"    Risk if shed: {a.risk_if_shed}")
                lines.append("")

        # Pareto frontier (ASCII)
        if self.pareto_frontier:
            lines.append("─" * 57)
            lines.append("  PARETO FRONTIER (Safety vs Tax)")
            lines.append("─" * 57)
            max_safety = max(p.total_safety for p in self.pareto_frontier) or 1
            max_tax = max(p.total_tax for p in self.pareto_frontier) or 1
            height = 12
            width = 40

            grid = [[" " for _ in range(width)] for _ in range(height)]
            for p in self.pareto_frontier:
                x = int((p.total_tax / max_tax) * (width - 1))
                y = int((p.total_safety / max_safety) * (height - 1))
                y_inv = height - 1 - y
                marker = "●" if p.is_current else "○"
                if 0 <= x < width and 0 <= y_inv < height:
                    grid[y_inv][x] = marker

            lines.append(f"  Safety ↑  (max={max_safety:.0f})")
            for row in grid:
                lines.append("  │" + "".join(row))
            lines.append("  └" + "─" * width + "→ Tax")
            lines.append(f"  {'':>{width}}  (max={max_tax:.0f})")
            lines.append("  ● = current config  ○ = Pareto-optimal alternative")
            lines.append("")

        # Consolidation
        if self.consolidation_opportunities:
            lines.append("─" * 57)
            lines.append("  CONSOLIDATION OPPORTUNITIES")
            lines.append("─" * 57)
            for opp in self.consolidation_opportunities[:5]:
                lines.append(
                    f"  {opp.constraint_a} + {opp.constraint_b}"
                )
                lines.append(
                    f"    Overlap: {opp.overlap_pct:.0%} | "
                    f"Savings: {opp.potential_savings:.1f} | "
                    f"Risk Δ: {opp.risk_delta:.1f}"
                )
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "summary": {
                "total_constraints": len(self.assessments),
                "total_tax": round(self.total_tax, 2),
                "total_safety": round(self.total_safety, 2),
                "avg_efficiency": round(self.avg_efficiency, 3),
                "shedding_risk_count": len(self.shedding_risks),
            },
            "assessments": [
                {
                    "name": a.constraint.name,
                    "category": a.constraint.category,
                    "composite_tax": round(a.composite_tax, 2),
                    "shedding_incentive": round(a.shedding_incentive, 3),
                    "efficiency_ratio": round(a.efficiency_ratio, 3),
                    "severity": a.severity.value,
                    "risk_if_shed": a.risk_if_shed,
                }
                for a in self.assessments
            ],
            "pareto_frontier": [
                {
                    "constraints": p.constraints_enabled,
                    "safety": round(p.total_safety, 2),
                    "tax": round(p.total_tax, 2),
                    "is_current": p.is_current,
                }
                for p in self.pareto_frontier
            ],
            "consolidation": [
                {
                    "a": o.constraint_a,
                    "b": o.constraint_b,
                    "overlap_pct": round(o.overlap_pct, 3),
                    "savings": round(o.potential_savings, 2),
                    "risk_delta": round(o.risk_delta, 2),
                }
                for o in self.consolidation_opportunities
            ],
        }

    def to_json(self, path: Optional[str] = None) -> str:
        """Serialize to JSON string, optionally writing to file."""
        output = json.dumps(self.to_dict(), indent=2)
        if path:
            from pathlib import Path
            Path(path).write_text(output, encoding="utf-8")
        return output

    def to_html(self, path: Optional[str] = None) -> str:
        """Generate interactive HTML dashboard."""
        data = self.to_dict()
        assessments_json = json.dumps(data["assessments"])
        pareto_json = json.dumps(data["pareto_frontier"])

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Alignment Tax Dashboard</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f1419; color: #e8eaed; padding: 24px; }}
.header {{ text-align: center; margin-bottom: 32px; }}
.header h1 {{ font-size: 28px; color: #8ab4f8; margin-bottom: 8px; }}
.header p {{ color: #9aa0a6; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 32px; }}
.card {{ background: #1f2937; border-radius: 12px; padding: 20px; text-align: center; }}
.card .value {{ font-size: 32px; font-weight: bold; color: #8ab4f8; }}
.card .label {{ font-size: 13px; color: #9aa0a6; margin-top: 4px; }}
.section {{ background: #1f2937; border-radius: 12px; padding: 24px; margin-bottom: 24px; }}
.section h2 {{ color: #8ab4f8; margin-bottom: 16px; font-size: 18px; }}
.bar-chart {{ width: 100%; }}
.bar-row {{ display: flex; align-items: center; margin-bottom: 8px; }}
.bar-label {{ width: 200px; font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.bar-container {{ flex: 1; height: 24px; background: #374151; border-radius: 4px; position: relative; }}
.bar {{ height: 100%; border-radius: 4px; transition: width 0.5s; display: flex; align-items: center; padding-left: 8px; font-size: 11px; }}
.bar.low {{ background: #34d399; }}
.bar.medium {{ background: #fbbf24; }}
.bar.high {{ background: #f87171; }}
.bar.critical {{ background: #dc2626; }}
.bar-value {{ width: 60px; text-align: right; font-size: 12px; margin-left: 8px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #374151; }}
th {{ color: #9aa0a6; font-weight: 600; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }}
.badge-low {{ background: #064e3b; color: #34d399; }}
.badge-medium {{ background: #78350f; color: #fbbf24; }}
.badge-high {{ background: #7f1d1d; color: #f87171; }}
.badge-critical {{ background: #450a0a; color: #fca5a5; }}
svg {{ width: 100%; height: 300px; }}
.pareto-point {{ cursor: pointer; }}
.pareto-point:hover {{ opacity: 0.8; }}
</style>
</head>
<body>
<div class="header">
<h1>⚖️ Alignment Tax Dashboard</h1>
<p>Performance cost of safety constraints — shedding risks &amp; optimization opportunities</p>
</div>
<div class="cards">
<div class="card"><div class="value">{len(self.assessments)}</div><div class="label">Constraints</div></div>
<div class="card"><div class="value">{self.total_tax:.0f}</div><div class="label">Total Tax</div></div>
<div class="card"><div class="value">{self.total_safety:.0f}</div><div class="label">Total Safety</div></div>
<div class="card"><div class="value">{self.avg_efficiency:.2f}</div><div class="label">Avg Efficiency</div></div>
<div class="card"><div class="value">{len(self.shedding_risks)}</div><div class="label">Shedding Risks</div></div>
</div>

<div class="section">
<h2>Constraint Tax Ranking</h2>
<div class="bar-chart" id="taxChart"></div>
</div>

<div class="section">
<h2>Pareto Frontier (Safety vs Tax)</h2>
<svg id="paretoChart" viewBox="0 0 600 300"></svg>
</div>

<div class="section">
<h2>Shedding Risk Assessment</h2>
<table>
<tr><th>Constraint</th><th>Category</th><th>Tax</th><th>Incentive</th><th>Severity</th><th>Risk If Shed</th></tr>
{"".join(f'<tr><td>{html_mod.escape(a.constraint.name)}</td><td>{html_mod.escape(a.constraint.category)}</td><td>{a.composite_tax:.1f}</td><td>{a.shedding_incentive:.2f}</td><td><span class="badge badge-{a.severity.value}">{a.severity.value.upper()}</span></td><td>{html_mod.escape(a.risk_if_shed)}</td></tr>' for a in self.assessments)}
</table>
</div>

{"" if not self.consolidation_opportunities else f'''<div class="section">
<h2>Consolidation Opportunities</h2>
<table>
<tr><th>Constraint A</th><th>Constraint B</th><th>Overlap</th><th>Savings</th><th>Risk Δ</th></tr>
{"".join(f'<tr><td>{html_mod.escape(o.constraint_a)}</td><td>{html_mod.escape(o.constraint_b)}</td><td>{o.overlap_pct:.0%}</td><td>{o.potential_savings:.1f}</td><td>{o.risk_delta:.1f}</td></tr>' for o in self.consolidation_opportunities)}
</table>
</div>'''}

<script>
const assessments = {assessments_json};
const pareto = {pareto_json};

// Bar chart
const chartEl = document.getElementById('taxChart');
const maxTax = Math.max(...assessments.map(a => a.composite_tax), 1);
assessments.forEach(a => {{
  const pct = (a.composite_tax / maxTax * 100).toFixed(1);
  const cls = a.severity;
  chartEl.innerHTML += `<div class="bar-row"><span class="bar-label">${{a.name}}</span><div class="bar-container"><div class="bar ${{cls}}" style="width:${{pct}}%"></div></div><span class="bar-value">${{a.composite_tax.toFixed(1)}}</span></div>`;
}});

// Pareto scatter
const svg = document.getElementById('paretoChart');
const pad = 50;
const w = 600 - 2*pad, h = 300 - 2*pad;
const maxS = Math.max(...pareto.map(p => p.safety), 1);
const maxT = Math.max(...pareto.map(p => p.tax), 1);
// Axes
svg.innerHTML = `<line x1="${{pad}}" y1="${{h+pad}}" x2="${{w+pad}}" y2="${{h+pad}}" stroke="#555" /><line x1="${{pad}}" y1="${{pad}}" x2="${{pad}}" y2="${{h+pad}}" stroke="#555" /><text x="${{w/2+pad}}" y="${{h+pad+35}}" fill="#9aa0a6" text-anchor="middle" font-size="12">Total Tax →</text><text x="15" y="${{h/2+pad}}" fill="#9aa0a6" text-anchor="middle" font-size="12" transform="rotate(-90,15,${{h/2+pad}})">Safety ↑</text>`;
// Points
pareto.forEach(p => {{
  const x = pad + (p.tax / maxT) * w;
  const y = pad + h - (p.safety / maxS) * h;
  const color = p.is_current ? '#8ab4f8' : '#34d399';
  const r = p.is_current ? 8 : 5;
  svg.innerHTML += `<circle class="pareto-point" cx="${{x}}" cy="${{y}}" r="${{r}}" fill="${{color}}" opacity="0.8"><title>${{p.constraints.length}} constraints\\nSafety: ${{p.safety.toFixed(1)}}\\nTax: ${{p.tax.toFixed(1)}}</title></circle>`;
}});
// Line connecting frontier
const sorted = [...pareto].sort((a,b) => a.tax - b.tax);
let pathD = '';
sorted.forEach((p, i) => {{
  const x = pad + (p.tax / maxT) * w;
  const y = pad + h - (p.safety / maxS) * h;
  pathD += (i === 0 ? `M${{x}},${{y}}` : ` L${{x}},${{y}}`);
}});
svg.innerHTML += `<path d="${{pathD}}" fill="none" stroke="#34d399" stroke-width="1.5" opacity="0.5" />`;
</script>
</body>
</html>"""
        if path:
            from pathlib import Path
            Path(path).write_text(html, encoding="utf-8")
        return html


# ── Demo Data ────────────────────────────────────────────────────────


def _demo_constraints() -> List[SafetyConstraint]:
    """Generate realistic demo constraints."""
    return [
        SafetyConstraint(
            name="request_rate_limiter",
            category="rate_limit",
            latency_ms=3.0,
            throughput_factor=0.7,
            cpu_overhead_pct=1.5,
            memory_overhead_mb=8.0,
            safety_value=65.0,
            bypass_difficulty=0.25,
        ),
        SafetyConstraint(
            name="output_content_filter",
            category="monitoring",
            latency_ms=45.0,
            throughput_factor=0.85,
            cpu_overhead_pct=12.0,
            memory_overhead_mb=200.0,
            safety_value=80.0,
            bypass_difficulty=0.7,
        ),
        SafetyConstraint(
            name="sandbox_isolation",
            category="containment",
            latency_ms=8.0,
            throughput_factor=0.75,
            cpu_overhead_pct=15.0,
            memory_overhead_mb=256.0,
            safety_value=95.0,
            bypass_difficulty=0.9,
        ),
        SafetyConstraint(
            name="human_approval_gate",
            category="approval",
            latency_ms=90.0,
            throughput_factor=0.3,
            cpu_overhead_pct=2.0,
            memory_overhead_mb=5.0,
            safety_value=90.0,
            bypass_difficulty=0.85,
        ),
        SafetyConstraint(
            name="memory_access_auditor",
            category="audit",
            latency_ms=12.0,
            throughput_factor=0.9,
            cpu_overhead_pct=8.0,
            memory_overhead_mb=64.0,
            safety_value=55.0,
            bypass_difficulty=0.4,
        ),
        SafetyConstraint(
            name="communication_encryption",
            category="encryption",
            latency_ms=6.0,
            throughput_factor=0.92,
            cpu_overhead_pct=5.0,
            memory_overhead_mb=32.0,
            safety_value=70.0,
            bypass_difficulty=0.6,
        ),
        SafetyConstraint(
            name="behavioral_anomaly_monitor",
            category="monitoring",
            latency_ms=25.0,
            throughput_factor=0.88,
            cpu_overhead_pct=18.0,
            memory_overhead_mb=180.0,
            safety_value=85.0,
            bypass_difficulty=0.65,
        ),
        SafetyConstraint(
            name="resource_usage_cap",
            category="containment",
            latency_ms=2.0,
            throughput_factor=0.6,
            cpu_overhead_pct=3.0,
            memory_overhead_mb=12.0,
            safety_value=60.0,
            bypass_difficulty=0.35,
        ),
        SafetyConstraint(
            name="replication_depth_limiter",
            category="containment",
            latency_ms=1.0,
            throughput_factor=0.95,
            cpu_overhead_pct=0.5,
            memory_overhead_mb=4.0,
            safety_value=88.0,
            bypass_difficulty=0.8,
        ),
        SafetyConstraint(
            name="goal_alignment_checker",
            category="monitoring",
            latency_ms=35.0,
            throughput_factor=0.82,
            cpu_overhead_pct=14.0,
            memory_overhead_mb=150.0,
            safety_value=92.0,
            bypass_difficulty=0.75,
        ),
        SafetyConstraint(
            name="privilege_escalation_blocker",
            category="isolation",
            latency_ms=4.0,
            throughput_factor=0.88,
            cpu_overhead_pct=3.0,
            memory_overhead_mb=20.0,
            safety_value=78.0,
            bypass_difficulty=0.7,
        ),
        SafetyConstraint(
            name="data_exfiltration_scanner",
            category="monitoring",
            latency_ms=50.0,
            throughput_factor=0.78,
            cpu_overhead_pct=20.0,
            memory_overhead_mb=280.0,
            safety_value=82.0,
            bypass_difficulty=0.55,
        ),
    ]


# ── CLI ──────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m replication alignment-tax",
        description="Alignment Tax Calculator — quantify safety constraint costs",
    )
    parser.add_argument("--demo", action="store_true", help="Use demo constraints")
    parser.add_argument("--pareto", action="store_true", help="Show Pareto frontier only")
    parser.add_argument("--shedding", action="store_true", help="Show shedding risks only")
    parser.add_argument("--consolidate", action="store_true", help="Show consolidation only")
    parser.add_argument("--remove", type=str, help="Simulate removing a constraint")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--html", type=str, metavar="FILE", help="Generate HTML dashboard")
    parser.add_argument("--output", "-o", type=str, help="Write output to file")

    args = parser.parse_args(argv)

    calc = AlignmentTaxCalculator()

    if args.demo:
        for c in _demo_constraints():
            calc.register_constraint(c)
    else:
        # Without demo, use demo data as default (real usage would load from config)
        for c in _demo_constraints():
            calc.register_constraint(c)

    # Handle --remove
    if args.remove:
        impact = calc.simulate_removal(args.remove)
        if args.json:
            output = json.dumps({
                "removal_impact": {
                    "constraint": impact.constraint_name,
                    "tax_reduction": round(impact.tax_reduction, 2),
                    "safety_loss": round(impact.safety_loss, 2),
                    "new_total_tax": round(impact.new_total_tax, 2),
                    "new_total_safety": round(impact.new_total_safety, 2),
                    "cascade": impact.shedding_cascade,
                    "verdict": impact.verdict,
                }
            }, indent=2)
        else:
            lines = []
            lines.extend(box_header(f"REMOVAL SIMULATION: {impact.constraint_name}"))
            lines.append("")
            lines.append(f"  Tax reduction:     {impact.tax_reduction:.1f}")
            lines.append(f"  Safety loss:       {impact.safety_loss:.1f}")
            lines.append(f"  New total tax:     {impact.new_total_tax:.1f}")
            lines.append(f"  New total safety:  {impact.new_total_safety:.1f}")
            if impact.shedding_cascade:
                lines.append(f"  Cascade risk:      {', '.join(impact.shedding_cascade)}")
            lines.append(f"  Verdict:           {impact.verdict}")
            output = "\n".join(lines)
        emit_output(output, args.output)
        return

    report = calc.assess_all()

    if args.html:
        report.to_html(args.html)
        print(f"HTML dashboard written to {args.html}")
        return

    if args.json:
        output = report.to_json()
    elif args.pareto:
        lines = []
        lines.extend(box_header("PARETO FRONTIER"))
        lines.append("")
        for p in report.pareto_frontier:
            marker = " ← CURRENT" if p.is_current else ""
            lines.append(
                f"  Safety={p.total_safety:>6.1f}  Tax={p.total_tax:>6.1f}  "
                f"Constraints={len(p.constraints_enabled)}{marker}"
            )
        output = "\n".join(lines)
    elif args.shedding:
        risks = report.shedding_risks
        if not risks:
            output = "No constraints above shedding threshold."
        else:
            lines = []
            lines.extend(box_header("SHEDDING RISK ALERTS"))
            lines.append("")
            for a in risks:
                lines.append(f"  ⚠️  {a.constraint.name} [{a.severity.value.upper()}]")
                lines.append(f"      Incentive: {a.shedding_incentive:.2f}")
                lines.append(f"      Tax: {a.composite_tax:.1f} | Safety: {a.constraint.safety_value:.1f}")
                lines.append(f"      Risk: {a.risk_if_shed}")
                lines.append("")
            output = "\n".join(lines)
    elif args.consolidate:
        opps = report.consolidation_opportunities
        if not opps:
            output = "No consolidation opportunities found."
        else:
            lines = []
            lines.extend(box_header("CONSOLIDATION OPPORTUNITIES"))
            lines.append("")
            for o in opps:
                lines.append(f"  {o.constraint_a} + {o.constraint_b}")
                lines.append(f"    Overlap: {o.overlap_pct:.0%} | Savings: {o.potential_savings:.1f} | Risk Δ: {o.risk_delta:.1f}")
                lines.append("")
            output = "\n".join(lines)
    else:
        output = report.render()

    emit_output(output, args.output)
