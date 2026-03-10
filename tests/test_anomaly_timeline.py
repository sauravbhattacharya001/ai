"""Tests for anomaly_timeline module."""

from __future__ import annotations

import json

import pytest

from replication.anomaly_timeline import (
    AnomalyEvent,
    AnomalyTimeline,
    EscalationChain,
    EventCluster,
    EventSeverity,
    EventSource,
    PatternType,
    RecurrencePattern,
    TimelineConfig,
    TimelineResult,
    _cli,
)


def _event(
    ts: float = 0.0,
    source: EventSource = EventSource.DRIFT,
    severity: EventSeverity = EventSeverity.MEDIUM,
    metric: str = "escape_rate",
    value: float = 0.3,
    baseline: float = 0.0,
    desc: str = "test event",
) -> AnomalyEvent:
    return AnomalyEvent(
        timestamp=ts, source=source, severity=severity,
        metric=metric, value=value, baseline=baseline, description=desc,
    )


class TestEventSeverity:
    def test_weight_ordering(self):
        assert EventSeverity.INFO.weight < EventSeverity.CRITICAL.weight

    def test_all_weights(self):
        assert EventSeverity.INFO.weight == 1
        assert EventSeverity.LOW.weight == 2
        assert EventSeverity.MEDIUM.weight == 3
        assert EventSeverity.HIGH.weight == 4
        assert EventSeverity.CRITICAL.weight == 5


class TestAnomalyEvent:
    def test_deviation_nonzero_baseline(self):
        e = _event(value=0.6, baseline=0.3)
        assert abs(e.deviation - 1.0) < 0.01

    def test_deviation_zero_baseline(self):
        e = _event(value=0.5, baseline=0.0)
        assert e.deviation == 0.5

    def test_metadata_default(self):
        e = _event()
        assert e.metadata == {}


class TestEventCluster:
    def test_basic_properties(self):
        events = [_event(ts=1.0), _event(ts=2.0), _event(ts=3.0)]
        c = EventCluster(events=events)
        assert c.start == 1.0
        assert c.end == 3.0
        assert c.duration == 2.0

    def test_max_severity(self):
        events = [
            _event(severity=EventSeverity.LOW),
            _event(severity=EventSeverity.CRITICAL),
            _event(severity=EventSeverity.MEDIUM),
        ]
        c = EventCluster(events=events)
        assert c.max_severity == EventSeverity.CRITICAL

    def test_multi_source(self):
        events = [_event(source=EventSource.DRIFT), _event(source=EventSource.CANARY)]
        c = EventCluster(events=events)
        assert c.is_multi_source

    def test_single_source(self):
        events = [_event(source=EventSource.DRIFT), _event(source=EventSource.DRIFT)]
        c = EventCluster(events=events)
        assert not c.is_multi_source

    def test_threat_score_positive(self):
        events = [
            _event(source=EventSource.DRIFT, severity=EventSeverity.HIGH),
            _event(source=EventSource.CANARY, severity=EventSeverity.CRITICAL),
        ]
        c = EventCluster(events=events)
        assert c.threat_score > 0

    def test_sources_sorted(self):
        events = [_event(source=EventSource.HONEYPOT), _event(source=EventSource.CANARY)]
        c = EventCluster(events=events)
        names = [s.value for s in c.sources]
        assert names == sorted(names)


class TestEscalationChain:
    def test_properties(self):
        events = [
            _event(ts=0.0, severity=EventSeverity.LOW),
            _event(ts=1.0, severity=EventSeverity.MEDIUM),
            _event(ts=2.0, severity=EventSeverity.HIGH),
        ]
        chain = EscalationChain(events=events)
        assert chain.start_severity == EventSeverity.LOW
        assert chain.end_severity == EventSeverity.HIGH
        assert chain.duration == 2.0
        assert chain.escalation_rate == 1.0

    def test_zero_duration(self):
        events = [
            _event(ts=0.0, severity=EventSeverity.LOW),
            _event(ts=0.0, severity=EventSeverity.HIGH),
        ]
        chain = EscalationChain(events=events)
        assert chain.escalation_rate == float("inf")


class TestRecurrencePattern:
    def test_periodic(self):
        p = RecurrencePattern(
            metric="escape_rate", source=EventSource.DRIFT,
            occurrences=5, mean_interval=10.0, std_interval=1.0,
            timestamps=[0, 10, 20, 30, 40],
        )
        assert p.is_periodic

    def test_irregular(self):
        p = RecurrencePattern(
            metric="escape_rate", source=EventSource.DRIFT,
            occurrences=5, mean_interval=10.0, std_interval=8.0,
            timestamps=[0, 5, 25, 26, 50],
        )
        assert not p.is_periodic

    def test_zero_interval(self):
        p = RecurrencePattern(
            metric="x", source=EventSource.CANARY,
            occurrences=3, mean_interval=0.0, std_interval=0.0,
            timestamps=[0, 0, 0],
        )
        assert not p.is_periodic


class TestClustering:
    def test_gap_clustering(self):
        config = TimelineConfig(cluster_gap=1.0, min_cluster_size=2)
        tl = AnomalyTimeline(config)
        tl.add_events([_event(ts=0.0), _event(ts=0.5), _event(ts=5.0), _event(ts=5.3)])
        result = tl.analyze(collect_drift=False)
        assert len(result.clusters) == 2

    def test_no_clusters_when_sparse(self):
        config = TimelineConfig(cluster_gap=0.1, min_cluster_size=2)
        tl = AnomalyTimeline(config)
        tl.add_events([_event(ts=0.0), _event(ts=10.0)])
        result = tl.analyze(collect_drift=False)
        assert len(result.clusters) == 0

    def test_single_event_no_cluster(self):
        tl = AnomalyTimeline()
        tl.add_event(_event())
        result = tl.analyze(collect_drift=False)
        assert len(result.clusters) == 0

    def test_cluster_pattern_escalation(self):
        config = TimelineConfig(cluster_gap=2.0, min_cluster_size=3)
        tl = AnomalyTimeline(config)
        tl.add_events([
            _event(ts=0, severity=EventSeverity.LOW),
            _event(ts=0.5, severity=EventSeverity.MEDIUM),
            _event(ts=1.0, severity=EventSeverity.HIGH),
        ])
        result = tl.analyze(collect_drift=False)
        assert len(result.clusters) == 1
        assert result.clusters[0].pattern == PatternType.ESCALATION

    def test_cluster_pattern_cascade(self):
        config = TimelineConfig(cluster_gap=2.0, min_cluster_size=2)
        tl = AnomalyTimeline(config)
        tl.add_events([
            _event(ts=0, source=EventSource.DRIFT),
            _event(ts=0.1, source=EventSource.CANARY),
        ])
        result = tl.analyze(collect_drift=False)
        assert result.clusters[0].pattern == PatternType.CASCADE


class TestEscalations:
    def test_detect_escalation(self):
        config = TimelineConfig(escalation_min_steps=3)
        tl = AnomalyTimeline(config)
        tl.add_events([
            _event(ts=0, severity=EventSeverity.INFO),
            _event(ts=1, severity=EventSeverity.LOW),
            _event(ts=2, severity=EventSeverity.MEDIUM),
            _event(ts=3, severity=EventSeverity.HIGH),
        ])
        result = tl.analyze(collect_drift=False)
        assert len(result.escalations) >= 1

    def test_no_escalation_flat(self):
        config = TimelineConfig(escalation_min_steps=3)
        tl = AnomalyTimeline(config)
        tl.add_events([
            _event(ts=0, severity=EventSeverity.MEDIUM),
            _event(ts=1, severity=EventSeverity.MEDIUM),
            _event(ts=2, severity=EventSeverity.MEDIUM),
        ])
        result = tl.analyze(collect_drift=False)
        assert len(result.escalations) == 0


class TestRecurrences:
    def test_detect_recurrence(self):
        config = TimelineConfig(recurrence_min_count=3)
        tl = AnomalyTimeline(config)
        tl.add_events([
            _event(ts=0, metric="x", source=EventSource.CANARY),
            _event(ts=10, metric="x", source=EventSource.CANARY),
            _event(ts=20, metric="x", source=EventSource.CANARY),
        ])
        result = tl.analyze(collect_drift=False)
        assert len(result.recurrences) == 1
        assert result.recurrences[0].metric == "x"

    def test_no_recurrence_few_events(self):
        config = TimelineConfig(recurrence_min_count=5)
        tl = AnomalyTimeline(config)
        tl.add_events([_event(ts=0, metric="x"), _event(ts=10, metric="x")])
        result = tl.analyze(collect_drift=False)
        assert len(result.recurrences) == 0


class TestTimelineResult:
    def _make_result(self) -> TimelineResult:
        config = TimelineConfig(cluster_gap=2.0, min_cluster_size=2, recurrence_min_count=3)
        tl = AnomalyTimeline(config)
        tl.add_events([
            _event(ts=0, source=EventSource.DRIFT, severity=EventSeverity.LOW, metric="a"),
            _event(ts=0.5, source=EventSource.CANARY, severity=EventSeverity.MEDIUM, metric="a"),
            _event(ts=1.0, source=EventSource.DRIFT, severity=EventSeverity.HIGH, metric="a"),
            _event(ts=10, source=EventSource.HONEYPOT, severity=EventSeverity.CRITICAL, metric="b"),
            _event(ts=10.5, source=EventSource.EVASION, severity=EventSeverity.HIGH, metric="b"),
        ])
        return tl.analyze(collect_drift=False)

    def test_total_events(self):
        assert self._make_result().total_events == 5

    def test_severity_distribution(self):
        assert sum(self._make_result().severity_distribution.values()) == 5

    def test_source_distribution(self):
        assert "drift" in self._make_result().source_distribution

    def test_to_dict_keys(self):
        d = self._make_result().to_dict()
        for key in ["total_events", "cluster_details", "escalation_details", "recurrence_details"]:
            assert key in d

    def test_render_nonempty(self):
        text = self._make_result().render()
        assert "ANOMALY TIMELINE" in text
        assert "Severity Distribution" in text

    def test_events_in_window(self):
        assert len(self._make_result().events_in_window(0, 2)) == 3

    def test_multi_source_clusters(self):
        assert len(self._make_result().multi_source_clusters) >= 1

    def test_max_threat_score(self):
        assert self._make_result().max_threat_score > 0


class TestCLI:
    def test_summary_mode(self, capsys):
        _cli(["--summary", "--windows", "3"])
        out = capsys.readouterr().out
        assert "Events:" in out

    def test_json_mode(self, capsys):
        _cli(["--json", "--windows", "3"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "total_events" in data

    def test_default_render(self, capsys):
        _cli(["--windows", "3"])
        out = capsys.readouterr().out
        assert "ANOMALY TIMELINE" in out


class TestEdgeCases:
    def test_empty_timeline(self):
        tl = AnomalyTimeline()
        result = tl.analyze(collect_drift=False)
        assert result.total_events == 0
        assert len(result.clusters) == 0
        assert result.max_threat_score == 0.0

    def test_add_event_and_add_events(self):
        tl = AnomalyTimeline()
        tl.add_event(_event(ts=0))
        tl.add_events([_event(ts=1), _event(ts=2)])
        result = tl.analyze(collect_drift=False)
        assert result.total_events == 3

    def test_manual_source(self):
        e = _event(source=EventSource.MANUAL)
        assert e.source == EventSource.MANUAL
