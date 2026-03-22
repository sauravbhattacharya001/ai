"""Tests for replication.runbook — Safety Runbook Generator."""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from replication.runbook import (
    RunbookGenerator, ThreatScenario, Runbook, Severity,
    ChecklistItem, EscalationLevel, RecoveryStep,
)


class TestThreatScenario:
    def test_severity_enum_valid(self):
        s = ThreatScenario(name="test", severity="critical")
        assert s.severity_enum == Severity.CRITICAL

    def test_severity_enum_invalid_defaults_high(self):
        s = ThreatScenario(name="test", severity="unknown")
        assert s.severity_enum == Severity.HIGH

    def test_severity_case_insensitive(self):
        s = ThreatScenario(name="test", severity="LOW")
        assert s.severity_enum == Severity.LOW


class TestRunbookGenerator:
    def setup_method(self):
        self.gen = RunbookGenerator()

    def test_list_templates(self):
        names = self.gen.template_names
        assert len(names) >= 6
        assert "self-replication" in names
        assert "data-exfiltration" in names

    def test_get_template(self):
        t = self.gen.get_template("self-replication")
        assert t is not None
        assert t.severity == "critical"

    def test_get_unknown_template(self):
        assert self.gen.get_template("nonexistent") is None

    def test_generate_basic(self):
        scenario = ThreatScenario(name="Test Incident", severity="medium")
        rb = self.gen.generate(scenario)
        assert isinstance(rb, Runbook)
        assert rb.title == "Runbook: Test Incident"
        assert rb.generated_at != ""
        assert len(rb.triage_checklist) > 0
        assert len(rb.escalation_path) > 0
        assert len(rb.containment_actions) > 0
        assert len(rb.recovery_steps) > 0
        assert len(rb.post_incident_items) > 0

    def test_generate_critical_has_more_steps(self):
        low = self.gen.generate(ThreatScenario(name="Low", severity="low"))
        crit = self.gen.generate(ThreatScenario(name="Crit", severity="critical"))
        assert len(crit.triage_checklist) > len(low.triage_checklist)
        assert len(crit.escalation_path) > len(low.escalation_path)
        assert len(crit.containment_actions) > len(low.containment_actions)

    def test_generate_from_template(self):
        scenario = self.gen.get_template("data-exfiltration")
        rb = self.gen.generate(scenario)
        assert "Data Exfiltration" in rb.title
        assert len(rb.scenario.indicators) > 0

    def test_generate_populates_indicators_if_empty(self):
        scenario = ThreatScenario(name="Empty", severity="low", indicators=[])
        rb = self.gen.generate(scenario)
        assert len(rb.scenario.indicators) > 0

    def test_affected_systems_in_evidence(self):
        scenario = ThreatScenario(
            name="Sys Test", severity="high",
            affected_systems=["db", "api"],
        )
        rb = self.gen.generate(scenario)
        evidence_text = " ".join(rb.evidence_collection)
        assert "db" in evidence_text
        assert "api" in evidence_text


class TestRunbookExport:
    def setup_method(self):
        gen = RunbookGenerator()
        self.rb = gen.generate(ThreatScenario(
            name="Export Test",
            severity="high",
            description="Testing export formats.",
            affected_systems=["sys-a"],
        ))

    def test_to_markdown(self):
        md = self.rb.to_markdown()
        assert "# Runbook: Export Test" in md
        assert "## Triage Checklist" in md
        assert "## Escalation Path" in md
        assert "## Recovery Procedure" in md
        assert "[ ]" in md  # checklist items

    def test_to_json(self):
        j = self.rb.to_json()
        data = json.loads(j)
        assert data["title"] == "Runbook: Export Test"
        assert "triage_checklist" in data
        assert len(data["triage_checklist"]) > 0

    def test_to_text(self):
        t = self.rb.to_text()
        assert "RUNBOOK: Export Test" in t
        assert "TRIAGE CHECKLIST" in t
        assert "ESCALATION PATH" in t

    def test_to_dict_roundtrip(self):
        d = self.rb.to_dict()
        assert isinstance(d, dict)
        j = json.dumps(d)
        d2 = json.loads(j)
        assert d2["title"] == d["title"]


class TestCustomTemplates:
    def test_custom_template_added(self):
        custom = {"my-threat": ThreatScenario(name="Custom", severity="low")}
        gen = RunbookGenerator(templates=custom)
        assert "my-threat" in gen.template_names
        # Built-ins still present
        assert "self-replication" in gen.template_names

    def test_custom_overrides_builtin(self):
        custom = {"self-replication": ThreatScenario(name="Overridden", severity="low")}
        gen = RunbookGenerator(templates=custom)
        t = gen.get_template("self-replication")
        assert t.name == "Overridden"
