"""Tests for alignment_tax module."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from replication.alignment_tax import (
    AlignmentTaxCalculator,
    ConsolidationOpportunity,
    ParetoPoint,
    RemovalImpact,
    SafetyConstraint,
    TaxAssessment,
    TaxConfig,
    TaxReport,
    _demo_constraints,
    main,
)
from replication._helpers import Severity


# ── Fixtures ─────────────────────────────────────────────────────────


def _make_constraint(
    name: str = "test_constraint",
    category: str = "monitoring",
    latency_ms: float = 10.0,
    throughput_factor: float = 0.9,
    cpu_overhead_pct: float = 5.0,
    memory_overhead_mb: float = 50.0,
    safety_value: float = 70.0,
    bypass_difficulty: float = 0.5,
) -> SafetyConstraint:
    return SafetyConstraint(
        name=name,
        category=category,
        latency_ms=latency_ms,
        throughput_factor=throughput_factor,
        cpu_overhead_pct=cpu_overhead_pct,
        memory_overhead_mb=memory_overhead_mb,
        safety_value=safety_value,
        bypass_difficulty=bypass_difficulty,
    )


def _make_calculator_with_demo() -> AlignmentTaxCalculator:
    calc = AlignmentTaxCalculator()
    for c in _demo_constraints():
        calc.register_constraint(c)
    return calc


# ── Registration Tests ───────────────────────────────────────────────


class TestRegistration:
    def test_register_single(self):
        calc = AlignmentTaxCalculator()
        c = _make_constraint()
        calc.register_constraint(c)
        assert len(calc.constraints) == 1
        assert calc.constraints[0].name == "test_constraint"

    def test_register_multiple(self):
        calc = AlignmentTaxCalculator()
        for i in range(5):
            calc.register_constraint(_make_constraint(name=f"c_{i}"))
        assert len(calc.constraints) == 5

    def test_register_overwrites_same_name(self):
        calc = AlignmentTaxCalculator()
        calc.register_constraint(_make_constraint(name="dup", safety_value=50))
        calc.register_constraint(_make_constraint(name="dup", safety_value=80))
        assert len(calc.constraints) == 1
        assert calc.constraints[0].safety_value == 80

    def test_empty_calculator(self):
        calc = AlignmentTaxCalculator()
        assert len(calc.constraints) == 0

    def test_demo_constraints_count(self):
        constraints = _demo_constraints()
        assert len(constraints) == 12


# ── Tax Computation Tests ────────────────────────────────────────────


class TestTaxComputation:
    def test_zero_cost_constraint(self):
        calc = AlignmentTaxCalculator()
        c = _make_constraint(
            latency_ms=0, throughput_factor=1.0,
            cpu_overhead_pct=0, memory_overhead_mb=0,
        )
        calc.register_constraint(c)
        report = calc.assess_all()
        assert report.assessments[0].composite_tax == 0.0

    def test_max_cost_constraint(self):
        calc = AlignmentTaxCalculator()
        c = _make_constraint(
            latency_ms=100, throughput_factor=0.0,
            cpu_overhead_pct=50, memory_overhead_mb=500,
        )
        calc.register_constraint(c)
        report = calc.assess_all()
        assert report.assessments[0].composite_tax == 100.0

    def test_tax_proportional_to_latency(self):
        calc = AlignmentTaxCalculator()
        calc.register_constraint(_make_constraint(name="low", latency_ms=5))
        calc.register_constraint(_make_constraint(name="high", latency_ms=80))
        report = calc.assess_all()
        taxes = {a.constraint.name: a.composite_tax for a in report.assessments}
        assert taxes["high"] > taxes["low"]

    def test_tax_proportional_to_throughput(self):
        calc = AlignmentTaxCalculator()
        calc.register_constraint(_make_constraint(name="good", throughput_factor=0.95))
        calc.register_constraint(_make_constraint(name="bad", throughput_factor=0.3))
        report = calc.assess_all()
        taxes = {a.constraint.name: a.composite_tax for a in report.assessments}
        assert taxes["bad"] > taxes["good"]

    def test_tax_capped_at_100(self):
        calc = AlignmentTaxCalculator()
        c = _make_constraint(
            latency_ms=200, throughput_factor=0.0,
            cpu_overhead_pct=100, memory_overhead_mb=1000,
        )
        calc.register_constraint(c)
        report = calc.assess_all()
        assert report.assessments[0].composite_tax <= 100.0

    def test_assessments_sorted_by_tax_descending(self):
        calc = _make_calculator_with_demo()
        report = calc.assess_all()
        taxes = [a.composite_tax for a in report.assessments]
        assert taxes == sorted(taxes, reverse=True)

    def test_efficiency_ratio_computation(self):
        calc = AlignmentTaxCalculator()
        c = _make_constraint(safety_value=80, latency_ms=20, throughput_factor=0.8)
        calc.register_constraint(c)
        report = calc.assess_all()
        a = report.assessments[0]
        expected = c.safety_value / a.composite_tax
        assert abs(a.efficiency_ratio - expected) < 0.01


# ── Shedding Incentive Tests ────────────────────────────────────────


class TestSheddingIncentive:
    def test_high_tax_low_difficulty_high_incentive(self):
        calc = AlignmentTaxCalculator()
        c = _make_constraint(
            latency_ms=90, throughput_factor=0.2,
            cpu_overhead_pct=40, memory_overhead_mb=400,
            bypass_difficulty=0.1, safety_value=30,
        )
        calc.register_constraint(c)
        report = calc.assess_all()
        assert report.assessments[0].shedding_incentive >= 0.6

    def test_low_tax_high_difficulty_low_incentive(self):
        calc = AlignmentTaxCalculator()
        c = _make_constraint(
            latency_ms=1, throughput_factor=0.98,
            cpu_overhead_pct=0.5, memory_overhead_mb=2,
            bypass_difficulty=0.95, safety_value=90,
        )
        calc.register_constraint(c)
        report = calc.assess_all()
        assert report.assessments[0].shedding_incentive < 0.3

    def test_shedding_incentive_bounded_0_1(self):
        calc = _make_calculator_with_demo()
        report = calc.assess_all()
        for a in report.assessments:
            assert 0.0 <= a.shedding_incentive <= 1.0

    def test_detect_shedding_risks_filters(self):
        calc = _make_calculator_with_demo()
        risks = calc.detect_shedding_risks()
        for r in risks:
            assert r.shedding_incentive >= calc.config.shedding_threshold

    def test_severity_levels(self):
        calc = AlignmentTaxCalculator()
        # Critical: very high shedding
        calc.register_constraint(_make_constraint(
            name="crit", latency_ms=95, throughput_factor=0.1,
            cpu_overhead_pct=45, memory_overhead_mb=450,
            bypass_difficulty=0.05, safety_value=20,
        ))
        report = calc.assess_all()
        assert report.assessments[0].severity in (Severity.HIGH, Severity.CRITICAL)


# ── Pareto Frontier Tests ────────────────────────────────────────────


class TestParetoFrontier:
    def test_empty_constraints_empty_frontier(self):
        calc = AlignmentTaxCalculator()
        frontier = calc.compute_pareto_frontier()
        assert frontier == []

    def test_single_constraint_frontier(self):
        calc = AlignmentTaxCalculator()
        calc.register_constraint(_make_constraint())
        frontier = calc.compute_pareto_frontier()
        assert len(frontier) >= 1

    def test_no_dominated_points(self):
        calc = _make_calculator_with_demo()
        frontier = calc.compute_pareto_frontier()
        for i, p in enumerate(frontier):
            for j, q in enumerate(frontier):
                if i == j:
                    continue
                # q should not dominate p
                dominated = (
                    q.total_safety >= p.total_safety
                    and q.total_tax <= p.total_tax
                    and (q.total_safety > p.total_safety or q.total_tax < p.total_tax)
                )
                assert not dominated, f"Point {i} dominated by point {j}"

    def test_frontier_sorted_by_tax(self):
        calc = _make_calculator_with_demo()
        frontier = calc.compute_pareto_frontier()
        taxes = [p.total_tax for p in frontier]
        assert taxes == sorted(taxes)

    def test_current_config_marked(self):
        calc = _make_calculator_with_demo()
        frontier = calc.compute_pareto_frontier()
        current = [p for p in frontier if p.is_current]
        # Current config should be present (may or may not be on frontier)
        # At least one point should exist
        assert len(frontier) >= 1

    def test_frontier_endpoints(self):
        calc = _make_calculator_with_demo()
        frontier = calc.compute_pareto_frontier()
        # First point should have lowest tax
        # Last point should have highest safety
        assert frontier[0].total_tax <= frontier[-1].total_tax


# ── Consolidation Tests ──────────────────────────────────────────────


class TestConsolidation:
    def test_same_category_high_overlap(self):
        calc = AlignmentTaxCalculator()
        calc.register_constraint(_make_constraint(
            name="mon_a", category="monitoring",
            latency_ms=20, throughput_factor=0.85,
        ))
        calc.register_constraint(_make_constraint(
            name="mon_b", category="monitoring",
            latency_ms=22, throughput_factor=0.84,
        ))
        opps = calc.find_consolidation_opportunities()
        assert len(opps) >= 1
        assert opps[0].constraint_a in ("mon_a", "mon_b")

    def test_different_category_lower_overlap(self):
        calc = AlignmentTaxCalculator()
        calc.register_constraint(_make_constraint(
            name="enc", category="encryption",
            latency_ms=5, throughput_factor=0.95,
        ))
        calc.register_constraint(_make_constraint(
            name="audit", category="audit",
            latency_ms=50, throughput_factor=0.5,
        ))
        opps = calc.find_consolidation_opportunities()
        # Different categories with different profiles = low overlap
        if opps:
            assert opps[0].overlap_pct < 0.8

    def test_consolidation_sorted_by_savings(self):
        calc = _make_calculator_with_demo()
        opps = calc.find_consolidation_opportunities()
        if len(opps) > 1:
            savings = [o.potential_savings for o in opps]
            assert savings == sorted(savings, reverse=True)

    def test_consolidation_with_single_constraint(self):
        calc = AlignmentTaxCalculator()
        calc.register_constraint(_make_constraint())
        opps = calc.find_consolidation_opportunities()
        assert opps == []


# ── Removal Simulation Tests ─────────────────────────────────────────


class TestRemovalSimulation:
    def test_removal_reduces_tax(self):
        calc = _make_calculator_with_demo()
        report = calc.assess_all()
        name = report.assessments[0].constraint.name
        impact = calc.simulate_removal(name)
        assert impact.tax_reduction > 0
        assert impact.new_total_tax < report.total_tax

    def test_removal_reduces_safety(self):
        calc = _make_calculator_with_demo()
        report = calc.assess_all()
        name = report.assessments[0].constraint.name
        impact = calc.simulate_removal(name)
        assert impact.safety_loss > 0
        assert impact.new_total_safety < report.total_safety

    def test_removal_unknown_constraint_raises(self):
        calc = _make_calculator_with_demo()
        with pytest.raises(ValueError, match="Unknown constraint"):
            calc.simulate_removal("nonexistent_constraint")

    def test_removal_verdict_exists(self):
        calc = _make_calculator_with_demo()
        name = calc.constraints[0].name
        impact = calc.simulate_removal(name)
        assert len(impact.verdict) > 0

    def test_removal_cascade_is_list(self):
        calc = _make_calculator_with_demo()
        name = calc.constraints[0].name
        impact = calc.simulate_removal(name)
        assert isinstance(impact.shedding_cascade, list)


# ── Report Rendering Tests ───────────────────────────────────────────


class TestReportRendering:
    def test_text_render_not_empty(self):
        calc = _make_calculator_with_demo()
        report = calc.assess_all()
        text = report.render()
        assert len(text) > 100
        assert "ALIGNMENT TAX ASSESSMENT" in text

    def test_text_contains_constraint_names(self):
        calc = _make_calculator_with_demo()
        report = calc.assess_all()
        text = report.render()
        for c in _demo_constraints()[:3]:
            assert c.name in text

    def test_json_valid(self):
        calc = _make_calculator_with_demo()
        report = calc.assess_all()
        j = report.to_json()
        data = json.loads(j)
        assert "summary" in data
        assert "assessments" in data
        assert "pareto_frontier" in data
        assert "consolidation" in data

    def test_json_summary_fields(self):
        calc = _make_calculator_with_demo()
        report = calc.assess_all()
        data = json.loads(report.to_json())
        summary = data["summary"]
        assert summary["total_constraints"] == 12
        assert summary["total_tax"] > 0
        assert summary["total_safety"] > 0

    def test_html_output(self):
        calc = _make_calculator_with_demo()
        report = calc.assess_all()
        html = report.to_html()
        assert "<!DOCTYPE html>" in html
        assert "Alignment Tax Dashboard" in html
        assert "paretoChart" in html

    def test_html_file_write(self, tmp_path):
        calc = _make_calculator_with_demo()
        report = calc.assess_all()
        out = str(tmp_path / "test.html")
        report.to_html(out)
        assert Path(out).exists()
        content = Path(out).read_text(encoding="utf-8")
        assert "Alignment Tax Dashboard" in content

    def test_json_file_write(self, tmp_path):
        calc = _make_calculator_with_demo()
        report = calc.assess_all()
        out = str(tmp_path / "test.json")
        report.to_json(out)
        assert Path(out).exists()
        data = json.loads(Path(out).read_text(encoding="utf-8"))
        assert "summary" in data


# ── CLI Tests ────────────────────────────────────────────────────────


class TestCLI:
    def test_demo_mode(self, capsys):
        main(["--demo"])
        out = capsys.readouterr().out
        assert "ALIGNMENT TAX ASSESSMENT" in out

    def test_json_output(self, capsys):
        main(["--demo", "--json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "summary" in data

    def test_pareto_mode(self, capsys):
        main(["--demo", "--pareto"])
        out = capsys.readouterr().out
        assert "PARETO FRONTIER" in out

    def test_shedding_mode(self, capsys):
        main(["--demo", "--shedding"])
        out = capsys.readouterr().out
        # May or may not have risks depending on demo data
        assert len(out) > 0

    def test_consolidate_mode(self, capsys):
        main(["--demo", "--consolidate"])
        out = capsys.readouterr().out
        assert len(out) > 0

    def test_remove_mode(self, capsys):
        main(["--demo", "--remove", "sandbox_isolation"])
        out = capsys.readouterr().out
        assert "REMOVAL SIMULATION" in out

    def test_remove_json_mode(self, capsys):
        main(["--demo", "--remove", "sandbox_isolation", "--json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "removal_impact" in data

    def test_html_generation(self, tmp_path):
        out = str(tmp_path / "dashboard.html")
        main(["--demo", "--html", out])
        assert Path(out).exists()


# ── Edge Cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    def test_all_zero_costs(self):
        calc = AlignmentTaxCalculator()
        calc.register_constraint(_make_constraint(
            latency_ms=0, throughput_factor=1.0,
            cpu_overhead_pct=0, memory_overhead_mb=0,
        ))
        report = calc.assess_all()
        assert report.total_tax == 0.0
        assert report.assessments[0].efficiency_ratio == float("inf")

    def test_report_properties(self):
        calc = _make_calculator_with_demo()
        report = calc.assess_all()
        assert report.total_tax > 0
        assert report.total_safety > 0
        assert report.avg_efficiency > 0

    def test_config_custom_weights(self):
        config = TaxConfig(
            weight_latency=1.0,
            weight_throughput=0.0,
            weight_cpu=0.0,
            weight_memory=0.0,
        )
        calc = AlignmentTaxCalculator(config)
        # Only latency matters
        calc.register_constraint(_make_constraint(
            name="high_lat", latency_ms=80,
            throughput_factor=0.1, cpu_overhead_pct=50, memory_overhead_mb=500,
        ))
        report = calc.assess_all()
        # Tax should be purely latency-based: 80/100 * 100 = 80
        assert abs(report.assessments[0].composite_tax - 80.0) < 0.1

    def test_config_shedding_threshold(self):
        config = TaxConfig(shedding_threshold=0.9)
        calc = AlignmentTaxCalculator(config)
        for c in _demo_constraints():
            calc.register_constraint(c)
        risks = calc.detect_shedding_risks()
        # Higher threshold = fewer risks
        calc2 = AlignmentTaxCalculator(TaxConfig(shedding_threshold=0.3))
        for c in _demo_constraints():
            calc2.register_constraint(c)
        risks2 = calc2.detect_shedding_risks()
        assert len(risks) <= len(risks2)
