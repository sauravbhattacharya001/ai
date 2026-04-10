"""Tests for the safety_checklist module."""
import json
import pytest
from replication.safety_checklist import (
    generate_checklist,
    format_markdown,
    format_json,
    format_html,
    Checklist,
    ChecklistItem,
    Priority,
    Phase,
    Category,
    DEFAULT_AGENT,
    _infer_risk_level,
)


class TestInferRiskLevel:
    def test_default_agent_is_medium(self):
        assert _infer_risk_level(DEFAULT_AGENT) == "medium"

    def test_high_risk_no_kill_switch(self):
        config = {**DEFAULT_AGENT, "kill_switch": False, "max_replication_depth": 6}
        level = _infer_risk_level(config)
        assert level in ("high", "critical")

    def test_low_risk_minimal_agent(self):
        config = {
            "max_replication_depth": 1,
            "allowed_actions": ["read"],
            "restricted_actions": ["self_modify", "network_exfiltrate", "execute"],
            "kill_switch": True,
            "alignment_score": 0.95,
        }
        assert _infer_risk_level(config) == "low"

    def test_critical_risk_dangerous_agent(self):
        config = {
            "max_replication_depth": 10,
            "allowed_actions": ["self_modify", "network_exfiltrate", "execute"],
            "restricted_actions": [],
            "kill_switch": False,
            "alignment_score": 0.3,
        }
        assert _infer_risk_level(config) == "critical"


class TestChecklist:
    def test_add_item(self):
        cl = Checklist(name="test", generated_at="now", agent_name="a", risk_level="low")
        item = ChecklistItem(
            id="T-001", title="Test", description="Desc",
            category=Category.TESTING, phase=Phase.PRE_DEPLOYMENT,
            priority=Priority.CRITICAL,
        )
        cl.add(item)
        assert cl.total_items == 1
        assert cl.critical_items == 1
        assert "pre-deployment" in cl.phases

    def test_add_non_critical(self):
        cl = Checklist(name="test", generated_at="now", agent_name="a", risk_level="low")
        item = ChecklistItem(
            id="T-002", title="Test", description="Desc",
            category=Category.TESTING, phase=Phase.ONGOING,
            priority=Priority.LOW,
        )
        cl.add(item)
        assert cl.total_items == 1
        assert cl.critical_items == 0


class TestGenerateChecklist:
    def test_default_generates_items(self):
        cl = generate_checklist()
        assert cl.total_items > 0
        assert cl.agent_name == "demo-agent"
        assert cl.risk_level == "medium"

    def test_phase_filter(self):
        cl = generate_checklist(phase_filter="ongoing")
        for phase, items in cl.phases.items():
            assert phase == "ongoing"

    def test_risk_level_override(self):
        cl = generate_checklist(risk_level="critical")
        assert cl.risk_level == "critical"

    def test_team_assignment(self):
        cl = generate_checklist(team_size=3)
        assignees = set()
        for items in cl.phases.values():
            for item in items:
                if item.assignee:
                    assignees.add(item.assignee)
        assert len(assignees) == 3

    def test_custom_config(self):
        config = {**DEFAULT_AGENT, "name": "custom-agent"}
        cl = generate_checklist(config=config)
        assert cl.agent_name == "custom-agent"

    def test_high_risk_has_extra_items(self):
        cl_low = generate_checklist(risk_level="low")
        cl_high = generate_checklist(risk_level="critical")
        assert cl_high.total_items >= cl_low.total_items


class TestFormatters:
    def test_markdown_format(self):
        cl = generate_checklist()
        md = format_markdown(cl)
        assert "# Safety Checklist" in md
        assert "demo-agent" in md
        assert "CON-001" in md

    def test_json_format(self):
        cl = generate_checklist()
        j = format_json(cl)
        data = json.loads(j)
        assert data["agent_name"] == "demo-agent"
        assert data["total_items"] > 0
        assert "phases" in data

    def test_html_format(self):
        cl = generate_checklist()
        h = format_html(cl)
        assert "<!DOCTYPE html>" in h
        assert "demo-agent" in h
        assert "Safety Checklist" in h
