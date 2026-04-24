"""Tests for the severity_classifier module."""
import json
import pytest
from replication.severity_classifier import (
    SeverityClassifier,
    Severity,
    IncidentReport,
    IMPACT_SCOPE_SCORES,
    DATA_SENSITIVITY_SCORES,
    CONTROL_BYPASS_SCORES,
)


@pytest.fixture
def classifier():
    return SeverityClassifier()


class TestSeverity:
    def test_severity_labels(self):
        assert Severity.P0.label == "CRITICAL"
        assert Severity.P4.label == "INFORMATIONAL"

    def test_response_windows(self):
        assert Severity.P0.response_window == "Immediate"
        assert Severity.P3.response_window == "Within 24 hours"

    def test_ordering(self):
        assert Severity.P0 < Severity.P4


class TestClassifier:
    def test_minimal_incident_is_p4(self, classifier):
        report = classifier.classify(description="Minor log anomaly")
        assert report.severity == Severity.P4

    def test_critical_incident_is_p0(self, classifier):
        report = classifier.classify(
            description="Agent escaped and replicated",
            impact_scope="external",
            data_sensitivity="credentials",
            control_bypass=["kill_switch", "quarantine"],
            reversibility="none",
            velocity="exponential",
            intent="deliberate",
        )
        assert report.severity == Severity.P0

    def test_score_accumulation(self, classifier):
        r1 = classifier.classify(description="test", impact_scope="none")
        r2 = classifier.classify(description="test", impact_scope="external")
        assert r2.score > r1.score

    def test_bypass_additive(self, classifier):
        r1 = classifier.classify(description="test", control_bypass=["logging"])
        r2 = classifier.classify(description="test", control_bypass=["logging", "kill_switch"])
        assert r2.score > r1.score

    def test_report_has_dimensions(self, classifier):
        report = classifier.classify(description="test incident")
        assert len(report.dimensions) == 6

    def test_report_summary(self, classifier):
        report = classifier.classify(description="Agent leaked PII", data_sensitivity="pii")
        summary = report.summary()
        assert "Severity" in summary
        assert "Agent leaked PII" in summary

    def test_report_to_dict(self, classifier):
        report = classifier.classify(description="test")
        d = report.to_dict()
        assert "severity" in d
        assert "dimensions" in d
        assert d["description"] == "test"

    def test_recommended_actions_p0(self, classifier):
        report = classifier.classify(
            description="critical",
            impact_scope="external",
            data_sensitivity="credentials",
            control_bypass=["kill_switch"],
            reversibility="none",
            velocity="exponential",
            intent="deliberate",
        )
        assert any("Page on-call" in a or "fleet quarantine" in a.lower() for a in report.recommended_actions)

    def test_unknown_values_default_zero(self, classifier):
        report = classifier.classify(
            description="test",
            impact_scope="nonexistent",
            data_sensitivity="nonexistent",
        )
        # Should not crash, score should be minimal
        assert report.score >= 0
