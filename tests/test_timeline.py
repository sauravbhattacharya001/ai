"""Tests for the Timeline Reconstructor module."""

from __future__ import annotations

import json

import pytest

from replication.timeline import (
    EventCategory,
    EventSeverity,
    EventSource,
    Timeline,
    TimelineConfig,
    TimelineEvent,
    TimelineReconstructor,
    TimelineSpan,
    TimelineStats,
)


class TestTimelineEvent:
    """Tests for TimelineEvent data class."""

    def test_to_dict(self):
        evt = TimelineEvent(
            timestamp=1.5,
            source=EventSource.SIMULATION,
            severity=EventSeverity.HIGH,
            category=EventCategory.SAFETY,
            summary="Test event",
            details={"key": "value"},
        )
        d = evt.to_dict()
        assert d["timestamp"] == 1.5
        assert d["source"] == "simulation"
        assert d["severity"] == "high"
        assert d["category"] == "safety"
        assert d["summary"] == "Test event"
        assert d["details"] == {"key": "value"}

    def test_default_fields(self):
        evt = TimelineEvent(
            timestamp=0.0,
            source=EventSource.SYSTEM,
            severity=EventSeverity.INFO,
            category=EventCategory.OPERATIONAL,
            summary="Startup",
        )
        assert evt.details == {}
        assert evt.related_events == []


class TestTimelineReconstructor:
    """Tests for the main TimelineReconstructor."""

    def test_build_default(self):
        r = TimelineReconstructor()
        timeline = r.build()
        assert isinstance(timeline, Timeline)
        assert len(timeline.events) > 0
        assert timeline.stats.total_events == len(timeline.events)
        assert timeline.built_at != ""

    def test_build_with_config(self):
        config = TimelineConfig(strategy="greedy", max_depth=3, drift_windows=3)
        r = TimelineReconstructor()
        timeline = r.build(config)
        assert len(timeline.events) > 0

    def test_severity_filter(self):
        config = TimelineConfig(severity_filter=EventSeverity.HIGH)
        r = TimelineReconstructor()
        timeline = r.build(config)
        for evt in timeline.events:
            assert evt.severity in (EventSeverity.HIGH, EventSeverity.CRITICAL)

    def test_source_filter(self):
        config = TimelineConfig(include_sources=[EventSource.THREAT])
        r = TimelineReconstructor()
        timeline = r.build(config)
        for evt in timeline.events:
            assert evt.source == EventSource.THREAT

    def test_category_filter(self):
        config = TimelineConfig(category_filter=EventCategory.COMPLIANCE)
        r = TimelineReconstructor()
        timeline = r.build(config)
        for evt in timeline.events:
            assert evt.category == EventCategory.COMPLIANCE

    def test_search_filter(self):
        config = TimelineConfig(search_query="simulation")
        r = TimelineReconstructor()
        timeline = r.build(config)
        for evt in timeline.events:
            found = (
                "simulation" in evt.summary.lower()
                or any("simulation" in str(v).lower() for v in evt.details.values())
            )
            assert found

    def test_last_n(self):
        config = TimelineConfig(last_n=5)
        r = TimelineReconstructor()
        timeline = r.build(config)
        assert len(timeline.events) <= 5

    def test_render(self):
        r = TimelineReconstructor()
        timeline = r.build()
        rendered = timeline.render()
        assert "TIMELINE RECONSTRUCTION" in rendered
        assert "Events:" in rendered

    def test_to_dict(self):
        r = TimelineReconstructor()
        timeline = r.build()
        d = timeline.to_dict()
        assert "events" in d
        assert "stats" in d
        assert "built_at" in d
        # Roundtrip through JSON
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["stats"]["total_events"] == len(d["events"])

    def test_chronological_order(self):
        r = TimelineReconstructor()
        timeline = r.build()
        timestamps = [e.timestamp for e in timeline.events]
        assert timestamps == sorted(timestamps)

    def test_stats_consistency(self):
        r = TimelineReconstructor()
        timeline = r.build()
        s = timeline.stats
        assert s.total_events == len(timeline.events)
        assert sum(s.by_severity.values()) == s.total_events
        assert sum(s.by_source.values()) == s.total_events
        assert sum(s.by_category.values()) == s.total_events
        assert s.critical_count == s.by_severity.get("critical", 0)
        assert s.high_count == s.by_severity.get("high", 0)

    def test_multiple_sources_present(self):
        r = TimelineReconstructor()
        timeline = r.build()
        sources = {e.source for e in timeline.events}
        # Should have at least simulation and threat sources
        assert len(sources) >= 2


class TestTimelineSpan:
    """Tests for span detection."""

    def test_spans_have_valid_indices(self):
        r = TimelineReconstructor()
        timeline = r.build()
        for span in timeline.stats.spans:
            assert span.start_idx >= 0
            assert span.end_idx >= span.start_idx
            assert span.end_idx < len(timeline.events)
            assert span.end_idx - span.start_idx >= 2  # 3+ events


class TestCLI:
    """Test the CLI main() function."""

    def test_cli_json(self, capsys):
        from replication.timeline import main as timeline_main
        timeline_main(["--json", "--last", "3"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "events" in data
        assert len(data["events"]) <= 3

    def test_cli_render(self, capsys):
        from replication.timeline import main as timeline_main
        timeline_main(["--last", "5"])
        captured = capsys.readouterr()
        assert "TIMELINE RECONSTRUCTION" in captured.out
