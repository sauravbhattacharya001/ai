"""Tests for the roi_calculator module."""
import json
import pytest
from replication.roi_calculator import (
    ROICalculator,
    SafetyControl,
    DEFAULT_CONTROLS,
    DEFAULT_SCENARIOS,
    ROIReport,
)


@pytest.fixture
def calculator():
    return ROICalculator()


class TestSafetyControl:
    def test_total_first_year_cost(self):
        ctrl = SafetyControl(
            name="Test", category="Test",
            annual_cost=100_000, implementation_cost=50_000,
            risk_reduction=0.5, coverage=0.8,
            maintenance_hours_per_month=10,
        )
        assert ctrl.total_first_year_cost == 150_000

    def test_effectiveness_score(self):
        ctrl = SafetyControl(
            name="Test", category="Test",
            annual_cost=0, implementation_cost=0,
            risk_reduction=0.5, coverage=0.8,
            maintenance_hours_per_month=0,
        )
        assert ctrl.effectiveness_score == pytest.approx(0.4)


class TestROICalculator:
    def test_total_ale(self, calculator):
        expected = sum(s.annual_expected_loss for s in DEFAULT_SCENARIOS)
        assert calculator.total_ale == expected

    def test_calculate_all_controls(self, calculator):
        report = calculator.calculate()
        assert report.total_annual_cost > 0
        assert report.combined_risk_reduction > 0
        assert len(report.controls) == len(DEFAULT_CONTROLS)

    def test_calculate_selected(self, calculator):
        report = calculator.calculate(selected=["monitoring", "kill_switch"])
        assert len(report.controls) == 2

    def test_report_render(self, calculator):
        report = calculator.calculate()
        text = report.render()
        assert "SAFETY ROI CALCULATOR" in text
        assert "RECOMMENDATIONS" in text

    def test_report_to_dict(self, calculator):
        report = calculator.calculate()
        d = report.to_dict()
        assert "total_ale" in d
        assert "controls" in d
        assert len(d["controls"]) > 0

    def test_risk_reduction_override(self, calculator):
        r1 = calculator.calculate(selected=["monitoring"])
        r2 = calculator.calculate(selected=["monitoring"], risk_reduction_override=0.9)
        # Override changes individual control loss reduction
        assert r2.controls[0].annual_loss_reduction > r1.controls[0].annual_loss_reduction

    def test_sensitivity_analysis(self, calculator):
        results = calculator.sensitivity_analysis(selected=["monitoring"], steps=3)
        assert len(results) == 3
        assert all("multiplier" in r for r in results)

    def test_residual_ale(self, calculator):
        report = calculator.calculate()
        assert report.residual_ale < report.total_ale

    def test_portfolio_roi(self, calculator):
        report = calculator.calculate()
        assert isinstance(report.portfolio_roi_percent, float)
