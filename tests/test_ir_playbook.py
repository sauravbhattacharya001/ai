"""Tests for the Incident Response Playbook Generator."""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from replication.ir_playbook import (
    ChecklistItem,
    EscalationContact,
    Playbook,
    PlaybookConfig,
    PlaybookGenerator,
    PlaybookPhase,
    Severity,
    ThreatCategory,
    _PLAYBOOK_BUILDERS,
)


def test_all_categories_have_builders():
    """Every threat category should have a playbook builder."""
    for cat in ThreatCategory:
        assert cat in _PLAYBOOK_BUILDERS, f"Missing builder for {cat.value}"


def test_generate_all_playbooks():
    """Generate all playbooks and verify basic structure."""
    gen = PlaybookGenerator()
    playbooks = gen.generate()
    assert len(playbooks) == len(ThreatCategory)
    for pb in playbooks:
        assert isinstance(pb, Playbook)
        assert pb.title
        assert pb.description
        assert len(pb.indicators) >= 3
        assert len(pb.phases) == 4  # detect, contain, eradicate, recover
        assert pb.escalation
        assert pb.post_incident


def test_severity_filtering():
    """Only playbooks at or above min severity should be returned."""
    config = PlaybookConfig(min_severity=Severity.HIGH)
    gen = PlaybookGenerator(config)
    playbooks = gen.generate()
    for pb in playbooks:
        assert pb.severity in (Severity.CRITICAL, Severity.HIGH)


def test_critical_only():
    """Filter to critical severity only."""
    config = PlaybookConfig(min_severity=Severity.CRITICAL)
    gen = PlaybookGenerator(config)
    playbooks = gen.generate()
    for pb in playbooks:
        assert pb.severity == Severity.CRITICAL
    assert len(playbooks) >= 3  # replication, exfiltration, escalation, self-mod


def test_category_filtering():
    """Filter to specific category."""
    config = PlaybookConfig(categories=[ThreatCategory.EVASION])
    gen = PlaybookGenerator(config)
    playbooks = gen.generate()
    assert len(playbooks) == 1
    assert playbooks[0].category == ThreatCategory.EVASION


def test_get_single_playbook():
    """Get a specific playbook by category."""
    gen = PlaybookGenerator()
    pb = gen.get_playbook(ThreatCategory.RUNAWAY_REPLICATION)
    assert pb is not None
    assert pb.category == ThreatCategory.RUNAWAY_REPLICATION
    assert pb.severity == Severity.CRITICAL


def test_render_text():
    """Render a playbook as text and verify key sections appear."""
    gen = PlaybookGenerator()
    pb = gen.get_playbook(ThreatCategory.DATA_EXFILTRATION)
    text = pb.render()
    assert "Data Exfiltration" in text
    assert "Indicators of Compromise" in text
    assert "DETECT & TRIAGE" in text
    assert "CONTAIN" in text
    assert "ERADICATE" in text
    assert "RECOVER" in text
    assert "Escalation Matrix" in text
    assert "Post-Incident Review" in text


def test_to_dict():
    """Serialize playbook to dict and verify structure."""
    gen = PlaybookGenerator()
    pb = gen.get_playbook(ThreatCategory.PROMPT_INJECTION)
    d = pb.to_dict()
    assert d["category"] == "prompt_injection"
    assert d["severity"] == "high"
    assert len(d["phases"]) == 4
    assert len(d["escalation"]) >= 3
    assert len(d["post_incident"]) >= 5
    assert isinstance(d["indicators"], list)
    assert isinstance(d["tags"], list)


def test_json_roundtrip():
    """Serialize to JSON and verify it parses back."""
    gen = PlaybookGenerator()
    playbooks = gen.generate()
    j = json.dumps([pb.to_dict() for pb in playbooks])
    parsed = json.loads(j)
    assert len(parsed) == len(playbooks)
    for item in parsed:
        assert "title" in item
        assert "phases" in item


def test_html_output():
    """Generate HTML output and verify it contains key elements."""
    gen = PlaybookGenerator()
    playbooks = gen.generate()
    html = gen.render_html(playbooks)
    assert "<!DOCTYPE html>" in html
    assert "Incident Response Playbooks" in html
    assert "playbook-card" in html
    assert "Escalation Matrix" in html
    for pb in playbooks:
        assert pb.title in html


def test_phase_time_targets():
    """All phases should have time targets."""
    gen = PlaybookGenerator()
    playbooks = gen.generate()
    for pb in playbooks:
        for phase in pb.phases:
            assert phase.time_target, f"Phase {phase.name} in {pb.title} missing time target"


def test_phase_steps_have_priorities():
    """All steps should have valid priorities."""
    valid = {"required", "recommended", "optional"}
    gen = PlaybookGenerator()
    playbooks = gen.generate()
    for pb in playbooks:
        for phase in pb.phases:
            for step in phase.steps:
                assert step.priority in valid, f"Invalid priority: {step.priority}"


def test_escalation_sla_ordering():
    """Escalation SLA should decrease as severity increases."""
    gen = PlaybookGenerator()
    pb = gen.get_playbook(ThreatCategory.RUNAWAY_REPLICATION)
    slas = [(c.notify_at, c.sla_minutes) for c in pb.escalation]
    severity_order = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}
    for i in range(len(slas) - 1):
        s1_sev = severity_order[slas[i][0]]
        s2_sev = severity_order[slas[i + 1][0]]
        if s2_sev > s1_sev:
            assert slas[i + 1][1] <= slas[i][1], "Higher severity should have lower SLA"


def test_related_modules_not_empty():
    """Every playbook should reference at least one related module."""
    gen = PlaybookGenerator()
    playbooks = gen.generate()
    for pb in playbooks:
        assert len(pb.related_modules) >= 2, f"{pb.title} has too few related modules"


def test_tags_not_empty():
    """Every playbook should have tags."""
    gen = PlaybookGenerator()
    playbooks = gen.generate()
    for pb in playbooks:
        assert len(pb.tags) >= 2, f"{pb.title} has too few tags"


def test_sorted_by_severity():
    """Generated playbooks should be sorted critical-first."""
    gen = PlaybookGenerator()
    playbooks = gen.generate()
    severity_order = list(Severity)
    indices = [severity_order.index(pb.severity) for pb in playbooks]
    assert indices == sorted(indices, reverse=True), "Playbooks not sorted by severity"


def test_cli_list(capsys):
    """Test --list flag."""
    from replication.ir_playbook import main
    main(["--list"])
    captured = capsys.readouterr()
    assert "Available threat categories" in captured.out
    for cat in ThreatCategory:
        assert cat.value in captured.out


def test_cli_json(capsys):
    """Test --json flag."""
    from replication.ir_playbook import main
    main(["--json", "--category", "evasion"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert len(data) == 1
    assert data[0]["category"] == "evasion"


def test_cli_category_filter(capsys):
    """Test --category flag."""
    from replication.ir_playbook import main
    main(["--category", "resource_hoarding"])
    captured = capsys.readouterr()
    assert "Resource Hoarding" in captured.out


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
