"""Tests for replication.incident_cost module."""

from replication.incident_cost import (
    IncidentCostEstimator,
    IncidentParams,
    Severity,
)


class TestSeverity:
    def test_multipliers_ordered(self):
        assert Severity.P0.multiplier > Severity.P1.multiplier
        assert Severity.P1.multiplier > Severity.P2.multiplier
        assert Severity.P2.multiplier > Severity.P3.multiplier
        assert Severity.P3.multiplier > Severity.P4.multiplier

    def test_labels(self):
        assert Severity.P0.label == "Critical"
        assert Severity.P4.label == "Informational"


class TestIncidentCostEstimator:
    def setup_method(self):
        self.estimator = IncidentCostEstimator()

    def test_basic_estimate(self):
        params = IncidentParams(severity="P2")
        report = self.estimator.estimate(params)
        assert report.total_expected > 0
        assert report.total_low < report.total_expected < report.total_high
        assert len(report.line_items) >= 5

    def test_severity_scaling(self):
        cost_p4 = self.estimator.estimate(IncidentParams(severity="P4")).total
        cost_p0 = self.estimator.estimate(IncidentParams(severity="P0")).total
        assert cost_p0 > cost_p4 * 5  # P0 should be much more expensive

    def test_data_exposure_adds_cost(self):
        base = self.estimator.estimate(IncidentParams(severity="P2", data_exposed=False)).total
        exposed = self.estimator.estimate(IncidentParams(severity="P2", data_exposed=True)).total
        assert exposed > base

    def test_regulated_adds_cost(self):
        base = self.estimator.estimate(IncidentParams(severity="P1", regulated=False)).total
        regulated = self.estimator.estimate(IncidentParams(severity="P1", regulated=True)).total
        assert regulated > base

    def test_blast_radius_scaling(self):
        small = self.estimator.estimate(IncidentParams(severity="P2", blast_radius=1)).total
        large = self.estimator.estimate(IncidentParams(severity="P2", blast_radius=20)).total
        assert large > small * 3

    def test_render_output(self):
        report = self.estimator.estimate(IncidentParams(severity="P1"))
        rendered = report.render()
        assert "INCIDENT COST ESTIMATE" in rendered
        assert "Response Labour" in rendered
        assert "$" in rendered

    def test_to_dict(self):
        report = self.estimator.estimate(IncidentParams(severity="P3"))
        d = report.to_dict()
        assert d["severity"] == "P3"
        assert "line_items" in d
        assert d["total_expected"] > 0

    def test_compare(self):
        result = self.estimator.compare(["P0", "P2", "P4"])
        assert "SEVERITY COST COMPARISON" in result
        assert "Critical" in result
        assert "Informational" in result

    def test_risk_factors_slow_detection(self):
        report = self.estimator.estimate(
            IncidentParams(severity="P1", detection_hours=8)
        )
        assert any("Slow detection" in rf for rf in report.risk_factors)

    def test_risk_factors_repeat(self):
        report = self.estimator.estimate(
            IncidentParams(severity="P2", repeat_incident=True)
        )
        assert any("Repeat" in rf for rf in report.risk_factors)

    def test_mitigation_savings(self):
        report = self.estimator.estimate(
            IncidentParams(severity="P1", detection_hours=5, blast_radius=8)
        )
        assert len(report.mitigation_savings) >= 2
