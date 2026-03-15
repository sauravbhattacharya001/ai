"""Tests for replication.safety_timeline module."""

import json
import pytest

from replication.safety_timeline import (
    EventCategory,
    EventSeverity,
    SafetyTimeline,
    TimelineEvent,
    generate_timeline,
    main,
)
from replication.simulator import ScenarioConfig, PRESETS


class TestTimelineEvent:
    def test_to_dict_serializes_enums(self):
        ev = TimelineEvent(
            tick=5, category=EventCategory.BREACH,
            severity=EventSeverity.CRITICAL,
            title="test breach", description="desc",
            agent="agent-1", metadata={"key": "val"},
        )
        d = ev.to_dict()
        assert d["category"] == "breach"
        assert d["severity"] == "critical"
        assert d["tick"] == 5
        assert d["agent"] == "agent-1"
        assert d["metadata"] == {"key": "val"}

    def test_to_dict_defaults(self):
        ev = TimelineEvent(
            tick=0, category=EventCategory.SYSTEM,
            severity=EventSeverity.INFO,
            title="t", description="d",
        )
        d = ev.to_dict()
        assert d["agent"] == ""
        assert d["metadata"] == {}


class TestSafetyTimeline:
    def test_collect_returns_events(self):
        tl = SafetyTimeline()
        events = tl.collect()
        assert isinstance(events, list)
        assert len(events) > 0
        assert all(isinstance(e, TimelineEvent) for e in events)

    def test_events_are_sorted_by_tick(self):
        tl = SafetyTimeline()
        events = tl.collect()
        ticks = [e.tick for e in events]
        assert ticks == sorted(ticks)

    def test_has_system_start_event(self):
        tl = SafetyTimeline()
        events = tl.collect()
        starts = [e for e in events if e.category == EventCategory.SYSTEM
                  and "started" in e.title.lower()]
        assert len(starts) >= 1

    def test_has_system_complete_event(self):
        tl = SafetyTimeline()
        events = tl.collect()
        completes = [e for e in events if e.category == EventCategory.SYSTEM
                     and "complete" in e.title.lower()]
        assert len(completes) >= 1

    def test_has_threat_events(self):
        tl = SafetyTimeline()
        events = tl.collect()
        threats = [e for e in events if e.category == EventCategory.THREAT]
        assert len(threats) > 0

    def test_custom_scenario(self):
        sc = ScenarioConfig(strategy="greedy", max_depth=3)
        tl = SafetyTimeline(scenario=sc)
        events = tl.collect()
        starts = [e for e in events if e.category == EventCategory.SYSTEM
                  and "started" in e.title.lower()]
        assert any("greedy" in e.description.lower() for e in starts)

    def test_preset_scenario(self):
        for name, preset in PRESETS.items():
            tl = SafetyTimeline(scenario=preset)
            events = tl.collect()
            assert len(events) > 0, f"Preset '{name}' produced no events"

    def test_policy_events_present(self):
        tl = SafetyTimeline()
        events = tl.collect()
        policy_events = [e for e in events if e.category == EventCategory.POLICY]
        assert len(policy_events) > 0

    def test_all_events_have_required_fields(self):
        tl = SafetyTimeline()
        events = tl.collect()
        for e in events:
            assert isinstance(e.tick, int)
            assert isinstance(e.category, EventCategory)
            assert isinstance(e.severity, EventSeverity)
            assert isinstance(e.title, str) and e.title
            assert isinstance(e.description, str)

    def test_severity_distribution(self):
        tl = SafetyTimeline()
        events = tl.collect()
        severities = {e.severity for e in events}
        # Should have at least INFO (from system events)
        assert EventSeverity.INFO in severities

    def test_category_distribution(self):
        tl = SafetyTimeline()
        events = tl.collect()
        categories = {e.category for e in events}
        assert EventCategory.SYSTEM in categories
        assert EventCategory.THREAT in categories


class TestGenerateTimeline:
    def test_returns_html(self):
        events = [
            TimelineEvent(
                tick=0, category=EventCategory.SYSTEM,
                severity=EventSeverity.INFO,
                title="Test", description="Test desc",
            ),
        ]
        html = generate_timeline(events)
        assert "<!DOCTYPE html>" in html
        assert "Safety Timeline" in html

    def test_embeds_events_as_json(self):
        events = [
            TimelineEvent(
                tick=42, category=EventCategory.BREACH,
                severity=EventSeverity.CRITICAL,
                title="Breach!", description="Bad stuff",
                agent="agent-x",
            ),
        ]
        html = generate_timeline(events)
        assert '"tick": 42' in html
        assert '"category": "breach"' in html
        assert '"agent": "agent-x"' in html

    def test_no_placeholder_in_output(self):
        html = generate_timeline([])
        assert "__EVENTS_DATA__" not in html

    def test_empty_events(self):
        html = generate_timeline([])
        assert "<!DOCTYPE html>" in html
        assert "[]" in html


class TestCLI:
    def test_json_output(self, capsys):
        main(["--json"])
        output = capsys.readouterr().out
        data = json.loads(output)
        assert isinstance(data, list)
        assert len(data) > 0

    def test_html_output(self, tmp_path):
        out = tmp_path / "test-timeline.html"
        main(["-o", str(out)])
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content

    def test_scenario_flag(self, capsys):
        main(["--scenario", "balanced", "--json"])
        output = capsys.readouterr().out
        data = json.loads(output)
        assert len(data) > 0

    def test_strategy_flag(self, capsys):
        main(["--strategy", "greedy", "--json"])
        output = capsys.readouterr().out
        data = json.loads(output)
        assert len(data) > 0

    def test_max_depth_flag(self, capsys):
        main(["--max-depth", "2", "--json"])
        output = capsys.readouterr().out
        data = json.loads(output)
        assert len(data) > 0
