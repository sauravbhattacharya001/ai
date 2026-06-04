"""Tests for replication.auto_investigator — autonomous safety investigation engine."""

import pytest

from replication.auto_investigator import (
    AutoInvestigator,
    Correlation,
    Finding,
    INCIDENT_TYPES,
    InvestigationPlaybook,
    InvestigationReport,
    InvestigationStep,
    TimelineEntry,
)
from replication._helpers import Severity


# ── Finding ─────────────────────────────────────────────────────────


class TestFinding:
    def test_basic_creation(self):
        f = Finding(
            severity=Severity.HIGH,
            title="Agent probed isolation boundary",
            description="Attempted network access outside allowed scope",
            module_source="containment_planner",
        )
        assert f.severity == Severity.HIGH
        assert f.title == "Agent probed isolation boundary"
        assert f.evidence == []
        assert f.correlates_with == []

    def test_with_evidence(self):
        f = Finding(
            severity=Severity.CRITICAL,
            title="Kill switch unresponsive",
            description="Kill switch did not respond within SLA",
            evidence=["Latency: 12.4s (SLA: 2s)", "3 retries failed"],
            module_source="kill_switch",
        )
        assert len(f.evidence) == 2


# ── InvestigationStep ───────────────────────────────────────────────


class TestInvestigationStep:
    def test_basic(self):
        step = InvestigationStep(
            name="check_containment",
            module="containment_planner",
            description="Verify containment boundaries",
        )
        assert step.depth_required == "shallow"
        assert step.depends_on == []

    def test_with_dependency(self):
        step = InvestigationStep(
            name="deep_scan",
            module="forensics",
            description="Full forensic analysis",
            depends_on=["check_containment"],
            depth_required="deep",
        )
        assert step.depends_on == ["check_containment"]


# ── TimelineEntry ───────────────────────────────────────────────────


class TestTimelineEntry:
    def test_basic(self):
        entry = TimelineEntry(
            timestamp="2026-06-04T09:00:00Z",
            event="Investigation started",
        )
        assert entry.severity is None
        assert entry.detail == ""


# ── Correlation ─────────────────────────────────────────────────────


class TestCorrelation:
    def test_basic(self):
        c = Correlation(
            finding_a="containment breach",
            finding_b="replication spike",
            relationship="temporal proximity",
            confidence=0.85,
        )
        assert c.confidence == 0.85


# ── InvestigationReport ─────────────────────────────────────────────


class TestInvestigationReport:
    def _sample_report(self) -> InvestigationReport:
        return InvestigationReport(
            incident="Agent escaped sandbox",
            incident_type="containment_breach",
            severity=Severity.CRITICAL,
            depth="standard",
            timestamp="2026-06-04T09:00:00Z",
            findings=[
                Finding(
                    severity=Severity.CRITICAL,
                    title="Sandbox boundary violated",
                    description="Agent accessed host filesystem",
                    module_source="containment_planner",
                )
            ],
            root_causes=["Insufficient syscall filtering"],
            recommendations=[("immediate", "Patch sandbox isolation layer")],
            steps_run=["containment_check", "forensics"],
            steps_skipped=["compliance_audit"],
        )

    def test_render_text(self):
        report = self._sample_report()
        text = report.render_text()
        assert "AUTONOMOUS INVESTIGATION REPORT" in text
        assert "containment_breach" in text
        assert "Sandbox boundary violated" in text
        assert "Insufficient syscall filtering" in text

    def test_to_dict(self):
        report = self._sample_report()
        d = report.to_dict()
        assert d["incident_type"] == "containment_breach"
        assert d["severity"] == "critical"
        assert len(d["findings"]) == 1
        assert d["findings"][0]["title"] == "Sandbox boundary violated"


# ── AutoInvestigator ────────────────────────────────────────────────


class TestAutoInvestigator:
    def test_instantiation(self):
        inv = AutoInvestigator()
        assert inv is not None

    def test_investigate_containment_breach(self):
        inv = AutoInvestigator()
        report = inv.investigate("Agent escaped containment boundary")
        assert isinstance(report, InvestigationReport)
        assert report.incident_type == "containment_breach"
        assert report.severity in (Severity.HIGH, Severity.CRITICAL)
        assert len(report.findings) > 0
        assert len(report.steps_run) > 0

    def test_investigate_replication_anomaly(self):
        inv = AutoInvestigator()
        report = inv.investigate("Unexpected replication spike detected in fleet")
        assert report.incident_type == "replication_anomaly"
        assert len(report.findings) > 0

    def test_investigate_kill_switch(self):
        inv = AutoInvestigator()
        report = inv.investigate("Kill switch latency exceeded SLA")
        assert report.incident_type == "kill_switch_failure"

    def test_investigate_behavioral_drift(self):
        inv = AutoInvestigator()
        report = inv.investigate("Gradual drift in alignment scores detected")
        assert report.incident_type == "behavioral_drift"

    def test_depth_shallow(self):
        inv = AutoInvestigator()
        report = inv.investigate("Minor compliance gap noted", depth="shallow")
        assert report.depth == "shallow"

    def test_depth_deep(self):
        inv = AutoInvestigator()
        report = inv.investigate("Agent escaped sandbox", depth="deep")
        assert report.depth == "deep"
        # Deep investigations should run more steps
        shallow = inv.investigate("Agent escaped sandbox", depth="shallow")
        assert len(report.steps_run) >= len(shallow.steps_run)

    def test_report_has_recommendations(self):
        inv = AutoInvestigator()
        report = inv.investigate("Critical containment breach")
        assert len(report.recommendations) > 0

    def test_report_has_timeline(self):
        inv = AutoInvestigator()
        report = inv.investigate("Agent replicated unexpectedly")
        assert len(report.timeline) > 0

    def test_report_json_serializable(self):
        """Ensure to_dict produces JSON-serializable output."""
        import json
        inv = AutoInvestigator()
        report = inv.investigate("Unauthorized privilege escalation")
        d = report.to_dict()
        # Should not raise
        serialized = json.dumps(d)
        assert len(serialized) > 0

    def test_investigate_unknown_incident(self):
        """Even unrecognized incidents should produce a report."""
        inv = AutoInvestigator()
        report = inv.investigate("Something completely unexpected happened")
        assert isinstance(report, InvestigationReport)
        assert len(report.findings) >= 0  # may or may not find anything
