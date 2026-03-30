"""Safety ROI Calculator — estimate return on investment for AI safety controls.

Compares the annual cost of implementing safety controls against the expected
loss reduction from prevented incidents. Supports multiple control categories,
custom cost/benefit inputs, and break-even analysis.

Usage (CLI)::

    python -m replication roi
    python -m replication roi --controls monitoring,access_control,kill_switch
    python -m replication roi --budget 500000 --risk-reduction 0.7
    python -m replication roi --controls all --json
    python -m replication roi --sensitivity
    python -m replication roi --compare monitoring,kill_switch

Programmatic::

    from replication.roi_calculator import ROICalculator, SafetyControl
    calc = ROICalculator()
    report = calc.calculate()
    print(report.render())
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Safety control definitions ───────────────────────────────────────

@dataclass
class SafetyControl:
    """A safety control with cost and effectiveness data."""
    name: str
    category: str
    annual_cost: float  # USD per year
    implementation_cost: float  # one-time setup cost
    risk_reduction: float  # fraction of risk mitigated (0.0–1.0)
    coverage: float  # fraction of threat surface covered (0.0–1.0)
    maintenance_hours_per_month: float  # staff hours
    description: str = ""

    @property
    def total_first_year_cost(self) -> float:
        return self.implementation_cost + self.annual_cost

    @property
    def effectiveness_score(self) -> float:
        """Combined effectiveness (risk_reduction × coverage)."""
        return self.risk_reduction * self.coverage


# ── Default control catalog ──────────────────────────────────────────

DEFAULT_CONTROLS: Dict[str, SafetyControl] = {
    "monitoring": SafetyControl(
        name="Continuous Monitoring",
        category="Detection",
        annual_cost=120_000,
        implementation_cost=50_000,
        risk_reduction=0.35,
        coverage=0.85,
        maintenance_hours_per_month=20,
        description="Real-time behavioral monitoring and anomaly detection",
    ),
    "access_control": SafetyControl(
        name="RBAC/ABAC Access Control",
        category="Prevention",
        annual_cost=80_000,
        implementation_cost=40_000,
        risk_reduction=0.45,
        coverage=0.70,
        maintenance_hours_per_month=10,
        description="Role and attribute-based access control for agent capabilities",
    ),
    "kill_switch": SafetyControl(
        name="Kill Switch & Circuit Breaker",
        category="Response",
        annual_cost=30_000,
        implementation_cost=25_000,
        risk_reduction=0.20,
        coverage=0.95,
        maintenance_hours_per_month=5,
        description="Emergency shutdown capability for rogue agents",
    ),
    "sandboxing": SafetyControl(
        name="Execution Sandboxing",
        category="Prevention",
        annual_cost=150_000,
        implementation_cost=80_000,
        risk_reduction=0.55,
        coverage=0.60,
        maintenance_hours_per_month=15,
        description="Isolated execution environments with resource limits",
    ),
    "audit_logging": SafetyControl(
        name="Tamper-Evident Audit Logging",
        category="Detection",
        annual_cost=60_000,
        implementation_cost=30_000,
        risk_reduction=0.15,
        coverage=0.90,
        maintenance_hours_per_month=8,
        description="Immutable audit trail for all agent actions",
    ),
    "alignment_testing": SafetyControl(
        name="Alignment Verification",
        category="Prevention",
        annual_cost=200_000,
        implementation_cost=100_000,
        risk_reduction=0.50,
        coverage=0.50,
        maintenance_hours_per_month=30,
        description="Regular alignment testing and behavioral validation",
    ),
    "incident_response": SafetyControl(
        name="Incident Response Team",
        category="Response",
        annual_cost=250_000,
        implementation_cost=60_000,
        risk_reduction=0.30,
        coverage=0.80,
        maintenance_hours_per_month=40,
        description="Dedicated team for safety incident response and remediation",
    ),
    "red_teaming": SafetyControl(
        name="Regular Red Team Exercises",
        category="Detection",
        annual_cost=180_000,
        implementation_cost=20_000,
        risk_reduction=0.25,
        coverage=0.65,
        maintenance_hours_per_month=25,
        description="Adversarial testing to discover safety control gaps",
    ),
    "data_loss_prevention": SafetyControl(
        name="Data Loss Prevention",
        category="Prevention",
        annual_cost=90_000,
        implementation_cost=45_000,
        risk_reduction=0.40,
        coverage=0.75,
        maintenance_hours_per_month=12,
        description="Prevent unauthorized data exfiltration by agents",
    ),
    "compliance_automation": SafetyControl(
        name="Compliance Automation",
        category="Governance",
        annual_cost=100_000,
        implementation_cost=70_000,
        risk_reduction=0.10,
        coverage=0.85,
        maintenance_hours_per_month=15,
        description="Automated regulatory compliance checking and reporting",
    ),
}


# ── Risk scenario defaults ───────────────────────────────────────────

@dataclass
class RiskScenario:
    """Expected annual loss without safety controls."""
    name: str
    annual_expected_loss: float  # ALE in USD
    probability: float  # annual probability of occurrence
    single_loss: float  # single loss expectancy (SLE)
    description: str = ""


DEFAULT_SCENARIOS: List[RiskScenario] = [
    RiskScenario("Agent Escape", 500_000, 0.10, 5_000_000,
                 "Agent breaks containment and operates unsupervised"),
    RiskScenario("Data Breach", 800_000, 0.15, 5_333_333,
                 "Agent exfiltrates sensitive data"),
    RiskScenario("Resource Abuse", 200_000, 0.25, 800_000,
                 "Agent hoards compute/network resources"),
    RiskScenario("Replication Event", 1_200_000, 0.05, 24_000_000,
                 "Unauthorized self-replication"),
    RiskScenario("Alignment Failure", 600_000, 0.08, 7_500_000,
                 "Agent pursues misaligned objectives"),
    RiskScenario("Privilege Escalation", 350_000, 0.12, 2_916_667,
                 "Agent gains unauthorized access levels"),
]


# ── ROI calculation ──────────────────────────────────────────────────

@dataclass
class ControlROI:
    """ROI result for a single control."""
    control: SafetyControl
    annual_loss_reduction: float
    net_benefit_year1: float
    net_benefit_annual: float  # after year 1
    roi_percent: float  # annual ROI after year 1
    break_even_months: float
    cost_per_risk_point: float  # cost per 1% risk reduction


@dataclass
class ROIReport:
    """Complete ROI analysis report."""
    controls: List[ControlROI]
    total_annual_cost: float
    total_implementation_cost: float
    total_first_year_cost: float
    total_annual_loss_reduction: float
    portfolio_roi_percent: float
    portfolio_net_benefit: float
    total_ale: float  # baseline annual loss expectancy
    residual_ale: float
    combined_risk_reduction: float

    def render(self) -> str:
        """Render as formatted text table."""
        lines = []
        lines.append("=" * 80)
        lines.append("  SAFETY ROI CALCULATOR — Return on Investment Analysis")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"  Baseline Annual Loss Expectancy (ALE): ${self.total_ale:>14,.0f}")
        lines.append(f"  Residual ALE (after controls):         ${self.residual_ale:>14,.0f}")
        lines.append(f"  Combined Risk Reduction:               {self.combined_risk_reduction:>14.1%}")
        lines.append("")
        lines.append("─" * 80)
        lines.append("  COST SUMMARY")
        lines.append("─" * 80)
        lines.append(f"  Total Implementation Cost:  ${self.total_implementation_cost:>12,.0f}")
        lines.append(f"  Total Annual Cost:          ${self.total_annual_cost:>12,.0f}")
        lines.append(f"  Total First-Year Cost:      ${self.total_first_year_cost:>12,.0f}")
        lines.append(f"  Annual Loss Reduction:      ${self.total_annual_loss_reduction:>12,.0f}")
        lines.append(f"  Portfolio Net Benefit (Y1):  ${self.portfolio_net_benefit:>12,.0f}")
        lines.append(f"  Portfolio ROI (annual):      {self.portfolio_roi_percent:>12.1f}%")
        lines.append("")
        lines.append("─" * 80)
        lines.append("  CONTROL BREAKDOWN")
        lines.append("─" * 80)
        hdr = f"  {'Control':<28} {'Category':<12} {'Annual $':>10} {'Risk Red':>9} {'ROI':>8} {'Break-even':>11}"
        lines.append(hdr)
        lines.append("  " + "─" * 76)
        for c in sorted(self.controls, key=lambda x: x.roi_percent, reverse=True):
            be = f"{c.break_even_months:.1f}mo" if c.break_even_months < 999 else "never"
            lines.append(
                f"  {c.control.name:<28} {c.control.category:<12} "
                f"${c.control.annual_cost:>9,.0f} {c.control.risk_reduction:>8.1%} "
                f"{c.roi_percent:>7.0f}% {be:>11}"
            )
        lines.append("")
        lines.append("─" * 80)
        lines.append("  RECOMMENDATIONS")
        lines.append("─" * 80)

        # Sort by ROI for recommendations
        ranked = sorted(self.controls, key=lambda x: x.roi_percent, reverse=True)
        if ranked:
            best = ranked[0]
            lines.append(f"  ★ Best ROI: {best.control.name} ({best.roi_percent:.0f}% annual return)")
            worst = ranked[-1]
            if worst.roi_percent < 0:
                lines.append(f"  ⚠ Negative ROI: {worst.control.name} — consider alternatives")
            quick_wins = [c for c in ranked if c.break_even_months < 6]
            if quick_wins:
                names = ", ".join(c.control.name for c in quick_wins[:3])
                lines.append(f"  ⚡ Quick wins (<6mo break-even): {names}")
        lines.append("")
        lines.append("=" * 80)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_ale": self.total_ale,
            "residual_ale": self.residual_ale,
            "combined_risk_reduction": round(self.combined_risk_reduction, 4),
            "total_implementation_cost": self.total_implementation_cost,
            "total_annual_cost": self.total_annual_cost,
            "total_first_year_cost": self.total_first_year_cost,
            "total_annual_loss_reduction": self.total_annual_loss_reduction,
            "portfolio_roi_percent": round(self.portfolio_roi_percent, 2),
            "portfolio_net_benefit": self.portfolio_net_benefit,
            "controls": [
                {
                    "name": c.control.name,
                    "category": c.control.category,
                    "annual_cost": c.control.annual_cost,
                    "risk_reduction": round(c.control.risk_reduction, 4),
                    "roi_percent": round(c.roi_percent, 2),
                    "break_even_months": round(c.break_even_months, 1),
                    "net_benefit_annual": round(c.net_benefit_annual, 2),
                }
                for c in sorted(self.controls, key=lambda x: x.roi_percent, reverse=True)
            ],
        }


class ROICalculator:
    """Calculate ROI for safety controls against risk scenarios."""

    def __init__(
        self,
        controls: Optional[Dict[str, SafetyControl]] = None,
        scenarios: Optional[List[RiskScenario]] = None,
        budget: Optional[float] = None,
        staff_hourly_rate: float = 75.0,
    ):
        self.controls = controls or dict(DEFAULT_CONTROLS)
        self.scenarios = scenarios or list(DEFAULT_SCENARIOS)
        self.budget = budget
        self.staff_hourly_rate = staff_hourly_rate

    @property
    def total_ale(self) -> float:
        """Total annual loss expectancy across all scenarios."""
        return sum(s.annual_expected_loss for s in self.scenarios)

    def calculate(
        self,
        selected: Optional[List[str]] = None,
        risk_reduction_override: Optional[float] = None,
    ) -> ROIReport:
        """Calculate ROI for selected controls (or all if None)."""
        if selected:
            controls = {k: v for k, v in self.controls.items() if k in selected}
        else:
            controls = dict(self.controls)

        ale = self.total_ale
        control_results: List[ControlROI] = []

        for key, ctrl in controls.items():
            rr = risk_reduction_override if risk_reduction_override is not None else ctrl.risk_reduction
            eff = rr * ctrl.coverage
            loss_reduction = ale * eff

            # Include staff cost
            staff_cost = ctrl.maintenance_hours_per_month * 12 * self.staff_hourly_rate
            total_annual = ctrl.annual_cost + staff_cost

            net_y1 = loss_reduction - ctrl.total_first_year_cost - staff_cost
            net_annual = loss_reduction - total_annual

            roi_pct = ((loss_reduction - total_annual) / total_annual * 100) if total_annual > 0 else 0

            if loss_reduction > 0:
                monthly_benefit = loss_reduction / 12
                monthly_cost = total_annual / 12
                if monthly_benefit > monthly_cost:
                    be = ctrl.implementation_cost / (monthly_benefit - monthly_cost) if (monthly_benefit - monthly_cost) > 0 else 999
                else:
                    be = 999
            else:
                be = 999

            cost_per_point = total_annual / (rr * 100) if rr > 0 else 0

            control_results.append(ControlROI(
                control=ctrl,
                annual_loss_reduction=loss_reduction,
                net_benefit_year1=net_y1,
                net_benefit_annual=net_annual,
                roi_percent=roi_pct,
                break_even_months=be,
                cost_per_risk_point=cost_per_point,
            ))

        # Portfolio-level calculations
        # Combined risk reduction uses 1 - ∏(1 - eff_i)
        combined = 1.0
        for cr in control_results:
            eff = cr.control.risk_reduction * cr.control.coverage
            combined *= (1 - eff)
        combined_reduction = 1 - combined

        total_annual_cost = sum(cr.control.annual_cost + cr.control.maintenance_hours_per_month * 12 * self.staff_hourly_rate for cr in control_results)
        total_impl = sum(cr.control.implementation_cost for cr in control_results)
        total_loss_red = ale * combined_reduction
        port_roi = ((total_loss_red - total_annual_cost) / total_annual_cost * 100) if total_annual_cost > 0 else 0

        return ROIReport(
            controls=control_results,
            total_annual_cost=total_annual_cost,
            total_implementation_cost=total_impl,
            total_first_year_cost=total_impl + total_annual_cost,
            total_annual_loss_reduction=total_loss_red,
            portfolio_roi_percent=port_roi,
            portfolio_net_benefit=total_loss_red - total_impl - total_annual_cost,
            total_ale=ale,
            residual_ale=ale * (1 - combined_reduction),
            combined_risk_reduction=combined_reduction,
        )

    def sensitivity_analysis(
        self,
        selected: Optional[List[str]] = None,
        steps: int = 5,
    ) -> List[Dict[str, Any]]:
        """Run sensitivity analysis varying risk reduction from 25% to 200% of base."""
        multipliers = [0.25 + (1.75 * i / (steps - 1)) for i in range(steps)]
        results = []
        for mult in multipliers:
            # Temporarily adjust controls
            original = {}
            controls = {k: v for k, v in self.controls.items() if selected is None or k in selected}
            for k, ctrl in controls.items():
                original[k] = ctrl.risk_reduction
                ctrl.risk_reduction = min(1.0, ctrl.risk_reduction * mult)

            report = self.calculate(selected)
            results.append({
                "multiplier": round(mult, 2),
                "label": f"{mult:.0%} of base risk",
                "portfolio_roi": round(report.portfolio_roi_percent, 1),
                "net_benefit": round(report.portfolio_net_benefit, 0),
                "combined_reduction": round(report.combined_risk_reduction, 4),
            })

            # Restore
            for k, rr in original.items():
                controls[k].risk_reduction = rr

        return results


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m replication roi",
        description="Safety ROI Calculator — estimate return on investment for AI safety controls",
    )
    parser.add_argument(
        "--controls", "-c",
        help="Comma-separated control names (default: all). Use 'all' or omit for full catalog.",
    )
    parser.add_argument(
        "--budget", "-b",
        type=float,
        help="Maximum annual budget (USD). Selects controls within budget by best ROI.",
    )
    parser.add_argument(
        "--risk-reduction", "-r",
        type=float,
        help="Override risk reduction factor (0.0–1.0) for all controls.",
    )
    parser.add_argument(
        "--staff-rate",
        type=float,
        default=75.0,
        help="Staff hourly rate for maintenance cost (default: $75/hr).",
    )
    parser.add_argument(
        "--sensitivity", "-s",
        action="store_true",
        help="Run sensitivity analysis showing ROI across risk levels.",
    )
    parser.add_argument(
        "--compare",
        help="Comma-separated control names to compare side-by-side.",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        dest="list_controls",
        help="List all available safety controls.",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output as JSON.",
    )

    args = parser.parse_args(argv)
    calc = ROICalculator(staff_hourly_rate=args.staff_rate)

    if args.list_controls:
        print("\nAvailable safety controls:\n")
        for key, ctrl in sorted(DEFAULT_CONTROLS.items()):
            print(f"  {key:<25} {ctrl.name}")
            print(f"  {'':25} {ctrl.description}")
            print(f"  {'':25} Annual: ${ctrl.annual_cost:,.0f}  |  Setup: ${ctrl.implementation_cost:,.0f}  |  Risk Red: {ctrl.risk_reduction:.0%}")
            print()
        return

    selected = None
    if args.controls and args.controls.lower() != "all":
        selected = [s.strip() for s in args.controls.split(",")]
        unknown = [s for s in selected if s not in DEFAULT_CONTROLS]
        if unknown:
            print(f"Unknown controls: {', '.join(unknown)}", file=sys.stderr)
            print(f"Available: {', '.join(sorted(DEFAULT_CONTROLS.keys()))}", file=sys.stderr)
            sys.exit(1)

    if args.compare:
        names = [s.strip() for s in args.compare.split(",")]
        unknown = [s for s in names if s not in DEFAULT_CONTROLS]
        if unknown:
            print(f"Unknown controls: {', '.join(unknown)}", file=sys.stderr)
            sys.exit(1)
        print("\n" + "=" * 70)
        print("  CONTROL COMPARISON")
        print("=" * 70)
        for name in names:
            report = calc.calculate([name], args.risk_reduction)
            cr = report.controls[0]
            print(f"\n  ■ {cr.control.name} ({cr.control.category})")
            print(f"    {cr.control.description}")
            print(f"    Annual Cost:      ${cr.control.annual_cost:>10,.0f}")
            print(f"    Setup Cost:       ${cr.control.implementation_cost:>10,.0f}")
            print(f"    Loss Reduction:   ${cr.annual_loss_reduction:>10,.0f}/yr")
            print(f"    Net Benefit:      ${cr.net_benefit_annual:>10,.0f}/yr")
            print(f"    ROI:              {cr.roi_percent:>10.0f}%")
            be = f"{cr.break_even_months:.1f} months" if cr.break_even_months < 999 else "never"
            print(f"    Break-even:       {be:>10}")
        print("\n" + "=" * 70)
        return

    if args.sensitivity:
        results = calc.sensitivity_analysis(selected)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print("\n" + "=" * 60)
            print("  SENSITIVITY ANALYSIS")
            print("=" * 60)
            print(f"  {'Risk Level':<22} {'ROI':>8} {'Net Benefit':>14} {'Risk Red':>10}")
            print("  " + "─" * 56)
            for r in results:
                print(
                    f"  {r['label']:<22} {r['portfolio_roi']:>7.1f}% "
                    f"${r['net_benefit']:>12,.0f} {r['combined_reduction']:>9.1%}"
                )
            print("=" * 60 + "\n")
        return

    # Budget-constrained selection
    if args.budget and not selected:
        # Greedy by ROI: calculate individual ROI, sort, pick within budget
        individual = []
        for key in DEFAULT_CONTROLS:
            r = calc.calculate([key], args.risk_reduction)
            cr = r.controls[0]
            total = cr.control.annual_cost + cr.control.maintenance_hours_per_month * 12 * args.staff_rate
            individual.append((key, cr.roi_percent, total))
        individual.sort(key=lambda x: x[1], reverse=True)
        selected = []
        remaining = args.budget
        for key, roi, cost in individual:
            if cost <= remaining:
                selected.append(key)
                remaining -= cost
        if not selected:
            print("Budget too low for any control.", file=sys.stderr)
            sys.exit(1)
        print(f"\n  Budget: ${args.budget:,.0f} → selected {len(selected)} controls\n")

    report = calc.calculate(selected, args.risk_reduction)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.render())


if __name__ == "__main__":
    main()
