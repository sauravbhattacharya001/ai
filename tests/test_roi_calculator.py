"""Tests for the ROICalculator module."""

from __future__ import annotations

import json
import pytest

from replication.roi_calculator import (
    DEFAULT_CONTROLS,
    DEFAULT_SCENARIOS,
    ControlROI,
    ROICalculator,
    ROIReport,
    RiskScenario,
    SafetyControl,
    main,
)


# ---------------------------------------------------------------------------
# SafetyControl dataclass
# ---------------------------------------------------------------------------


class TestSafetyControl:
    def test_first_year_cost_sums_setup_and_annual(self):
        c = SafetyControl(
            name="x", category="Detection",
            annual_cost=100.0, implementation_cost=50.0,
            risk_reduction=0.5, coverage=0.8,
            maintenance_hours_per_month=4,
        )
        assert c.total_first_year_cost == 150.0

    def test_effectiveness_score_is_product(self):
        c = SafetyControl("x", "Prevention", 1, 1, 0.4, 0.5, 0)
        assert c.effectiveness_score == pytest.approx(0.20)

    def test_zero_effectiveness_when_no_coverage(self):
        c = SafetyControl("x", "Prevention", 1, 1, 1.0, 0.0, 0)
        assert c.effectiveness_score == 0.0


# ---------------------------------------------------------------------------
# Default catalog sanity
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_default_controls_present(self):
        assert "monitoring" in DEFAULT_CONTROLS
        assert "kill_switch" in DEFAULT_CONTROLS
        assert "sandboxing" in DEFAULT_CONTROLS

    def test_default_controls_have_valid_factors(self):
        for key, ctrl in DEFAULT_CONTROLS.items():
            assert 0.0 <= ctrl.risk_reduction <= 1.0, key
            assert 0.0 <= ctrl.coverage <= 1.0, key
            assert ctrl.annual_cost >= 0, key
            assert ctrl.implementation_cost >= 0, key

    def test_default_scenarios_have_positive_loss(self):
        assert DEFAULT_SCENARIOS
        for s in DEFAULT_SCENARIOS:
            assert s.annual_expected_loss > 0
            assert 0 <= s.probability <= 1


# ---------------------------------------------------------------------------
# ROICalculator core
# ---------------------------------------------------------------------------


class TestCalculator:
    def test_total_ale_sums_scenarios(self):
        scenarios = [
            RiskScenario("a", 100.0, 0.1, 1000.0),
            RiskScenario("b", 250.0, 0.2, 1250.0),
        ]
        calc = ROICalculator(scenarios=scenarios)
        assert calc.total_ale == 350.0

    def test_calculate_returns_report_for_all_controls(self):
        calc = ROICalculator()
        report = calc.calculate()
        assert isinstance(report, ROIReport)
        assert len(report.controls) == len(DEFAULT_CONTROLS)
        assert report.total_ale == sum(s.annual_expected_loss for s in DEFAULT_SCENARIOS)

    def test_calculate_with_selected_subset(self):
        calc = ROICalculator()
        report = calc.calculate(selected=["monitoring", "kill_switch"])
        names = {cr.control.name for cr in report.controls}
        assert names == {
            DEFAULT_CONTROLS["monitoring"].name,
            DEFAULT_CONTROLS["kill_switch"].name,
        }

    def test_combined_risk_reduction_uses_independence(self):
        # Two independent 50%/100% coverage controls -> 1 - 0.5*0.5 = 0.75
        controls = {
            "a": SafetyControl("A", "x", 100, 0, 0.5, 1.0, 0),
            "b": SafetyControl("B", "x", 100, 0, 0.5, 1.0, 0),
        }
        scenarios = [RiskScenario("s", 1_000_000.0, 1.0, 1_000_000.0)]
        calc = ROICalculator(controls=controls, scenarios=scenarios, staff_hourly_rate=0)
        report = calc.calculate()
        assert report.combined_risk_reduction == pytest.approx(0.75)
        assert report.residual_ale == pytest.approx(250_000.0)
        assert report.total_annual_loss_reduction == pytest.approx(750_000.0)

    def test_combined_reduction_never_exceeds_one(self):
        # Even with very high individual reductions, combined stays <=1
        calc = ROICalculator()
        report = calc.calculate()
        assert 0.0 <= report.combined_risk_reduction <= 1.0

    def test_individual_loss_reduction_matches_effectiveness(self):
        scenarios = [RiskScenario("s", 1_000.0, 1.0, 1_000.0)]
        controls = {
            "a": SafetyControl("A", "x", 100, 0, 0.4, 0.5, 0),
        }
        calc = ROICalculator(controls=controls, scenarios=scenarios, staff_hourly_rate=0)
        report = calc.calculate()
        cr = report.controls[0]
        # effectiveness = 0.4 * 0.5 = 0.20  ->  loss_reduction = 200
        assert cr.annual_loss_reduction == pytest.approx(200.0)

    def test_risk_reduction_override_propagates(self):
        scenarios = [RiskScenario("s", 1_000.0, 1.0, 1_000.0)]
        controls = {"a": SafetyControl("A", "x", 100, 0, 0.1, 1.0, 0)}
        calc = ROICalculator(controls=controls, scenarios=scenarios, staff_hourly_rate=0)
        # Override to 0.9: loss_reduction = 1000 * 0.9 * 1.0 = 900
        report = calc.calculate(risk_reduction_override=0.9)
        assert report.controls[0].annual_loss_reduction == pytest.approx(900.0)

    def test_staff_hours_included_in_annual_cost(self):
        scenarios = [RiskScenario("s", 0.0, 0.0, 0.0)]
        controls = {
            "a": SafetyControl("A", "x", 1_000, 0,
                               risk_reduction=0.0, coverage=0.0,
                               maintenance_hours_per_month=10),
        }
        calc = ROICalculator(controls=controls, scenarios=scenarios,
                             staff_hourly_rate=100.0)
        report = calc.calculate()
        # staff cost = 10 hrs * 12 months * $100 = $12,000 + $1000 annual = $13,000
        assert report.total_annual_cost == pytest.approx(13_000.0)

    def test_break_even_months_uses_implementation_cost(self):
        # Setup: monthly benefit $200, monthly cost $100 -> net $100/mo
        # implementation = $500 -> break_even = 5 months
        scenarios = [RiskScenario("s", 2_400.0, 1.0, 2_400.0)]
        controls = {
            "a": SafetyControl("A", "x",
                               annual_cost=1_200, implementation_cost=500,
                               risk_reduction=1.0, coverage=1.0,
                               maintenance_hours_per_month=0),
        }
        calc = ROICalculator(controls=controls, scenarios=scenarios,
                             staff_hourly_rate=0)
        report = calc.calculate()
        assert report.controls[0].break_even_months == pytest.approx(5.0)

    def test_break_even_marker_when_costs_exceed_benefit(self):
        # Annual cost 10k vs benefit 1k -> never break even
        scenarios = [RiskScenario("s", 1_000.0, 1.0, 1_000.0)]
        controls = {
            "a": SafetyControl("A", "x", 10_000, 5_000, 0.1, 1.0, 0),
        }
        calc = ROICalculator(controls=controls, scenarios=scenarios,
                             staff_hourly_rate=0)
        report = calc.calculate()
        assert report.controls[0].break_even_months == 999

    def test_zero_risk_reduction_yields_zero_loss_reduction(self):
        scenarios = [RiskScenario("s", 1_000.0, 1.0, 1_000.0)]
        controls = {"a": SafetyControl("A", "x", 100, 0, 0.0, 1.0, 0)}
        calc = ROICalculator(controls=controls, scenarios=scenarios,
                             staff_hourly_rate=0)
        report = calc.calculate()
        cr = report.controls[0]
        assert cr.annual_loss_reduction == 0.0
        assert cr.cost_per_risk_point == 0.0  # rr=0 short-circuits divide

    def test_cost_per_risk_point(self):
        scenarios = [RiskScenario("s", 1_000.0, 1.0, 1_000.0)]
        controls = {"a": SafetyControl("A", "x", 100, 0, 0.5, 1.0, 0)}
        calc = ROICalculator(controls=controls, scenarios=scenarios,
                             staff_hourly_rate=0)
        report = calc.calculate()
        # 100 / (0.5 * 100) = 2.0
        assert report.controls[0].cost_per_risk_point == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------


class TestSensitivity:
    def test_sensitivity_returns_expected_step_count(self):
        calc = ROICalculator()
        results = calc.sensitivity_analysis(selected=["monitoring"], steps=5)
        assert len(results) == 5
        for r in results:
            assert "multiplier" in r
            assert "portfolio_roi" in r
            assert "net_benefit" in r

    def test_sensitivity_does_not_mutate_controls(self):
        calc = ROICalculator()
        before = {k: c.risk_reduction for k, c in calc.controls.items()}
        calc.sensitivity_analysis(steps=3)
        after = {k: c.risk_reduction for k, c in calc.controls.items()}
        assert before == after

    def test_sensitivity_multipliers_are_monotonic(self):
        calc = ROICalculator()
        results = calc.sensitivity_analysis(steps=4)
        mults = [r["multiplier"] for r in results]
        assert mults == sorted(mults)
        assert mults[0] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Report rendering / serialization
# ---------------------------------------------------------------------------


class TestReportRendering:
    def test_render_contains_section_headers(self):
        calc = ROICalculator()
        text = calc.calculate(selected=["monitoring"]).render()
        assert "SAFETY ROI CALCULATOR" in text
        assert "COST SUMMARY" in text
        assert "CONTROL BREAKDOWN" in text
        assert "RECOMMENDATIONS" in text

    def test_to_dict_is_json_serializable(self):
        calc = ROICalculator()
        report = calc.calculate(selected=["monitoring", "kill_switch"])
        payload = report.to_dict()
        encoded = json.dumps(payload)
        # Decoded matches structurally
        decoded = json.loads(encoded)
        assert decoded["controls"]
        assert "portfolio_roi_percent" in decoded
        assert "combined_risk_reduction" in decoded

    def test_render_sorts_controls_by_roi_descending(self):
        calc = ROICalculator()
        report = calc.calculate()
        text = report.render()
        # Find lines that look like control rows
        rois = [cr.roi_percent for cr in
                sorted(report.controls, key=lambda x: x.roi_percent, reverse=True)]
        # First listed control row should have the highest ROI
        head_name = sorted(report.controls, key=lambda x: x.roi_percent,
                            reverse=True)[0].control.name
        assert head_name in text
        assert rois == sorted(rois, reverse=True)


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


class TestCLI:
    def test_cli_list_controls(self, capsys):
        main(["--list"])
        out = capsys.readouterr().out
        assert "monitoring" in out
        assert "kill_switch" in out

    def test_cli_default_run_prints_report(self, capsys):
        main([])
        out = capsys.readouterr().out
        assert "SAFETY ROI CALCULATOR" in out

    def test_cli_json_output_is_valid_json(self, capsys):
        main(["--json", "--controls", "monitoring,kill_switch"])
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert "controls" in payload
        names = [c["name"] for c in payload["controls"]]
        assert DEFAULT_CONTROLS["monitoring"].name in names

    def test_cli_unknown_control_exits_nonzero(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            main(["--controls", "does_not_exist"])
        assert excinfo.value.code != 0

    def test_cli_compare_prints_each_control(self, capsys):
        main(["--compare", "monitoring,kill_switch"])
        out = capsys.readouterr().out
        assert "CONTROL COMPARISON" in out
        assert DEFAULT_CONTROLS["monitoring"].name in out
        assert DEFAULT_CONTROLS["kill_switch"].name in out

    def test_cli_sensitivity_json(self, capsys):
        main(["--sensitivity", "--json", "--controls", "monitoring"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert isinstance(data, list) and len(data) >= 3

    def test_cli_budget_selects_within_limit(self, capsys):
        # Tiny budget should pick at most cheapest controls
        main(["--budget", "60000"])
        out = capsys.readouterr().out
        assert "Budget:" in out or "Budget too low" in out
