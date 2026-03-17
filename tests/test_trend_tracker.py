"""Tests for the Safety Trend Tracker module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from replication.trend_tracker import (
    TrendEntry,
    TrendSummary,
    TrendTracker,
    RegressionAlert,
    _sparkline,
)


def _make_entry(score: float, tag: str = "", dims: dict | None = None) -> TrendEntry:
    return TrendEntry(
        timestamp="2026-03-17T10:00:00+00:00",
        overall_score=score,
        overall_grade="A" if score >= 93 else "B",
        dimensions=dims or {"Contract Enforcement": score, "Threat Resilience": score - 5},
        config_summary="balanced, depth=3, replicas=10",
        tag=tag,
    )


def test_entry_roundtrip():
    e = _make_entry(85.0, tag="test")
    d = e.to_dict()
    e2 = TrendEntry.from_dict(d)
    assert e2.overall_score == 85.0
    assert e2.tag == "test"
    assert e2.dimensions == e.dimensions


def test_tracker_record_and_load():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "trends.jsonl")
        tracker = TrendTracker(path)

        # Manually append entries (avoid running full scorecard)
        e1 = _make_entry(80.0, tag="first")
        e2 = _make_entry(85.0, tag="second")
        tracker._append(e1)
        tracker._append(e2)

        entries = tracker.entries()
        assert len(entries) == 2
        assert entries[0].tag == "first"
        assert entries[1].tag == "second"


def test_tracker_summary_improving():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "trends.jsonl")
        tracker = TrendTracker(path)
        for score in [70.0, 75.0, 80.0, 85.0]:
            tracker._append(_make_entry(score))

        summary = tracker.summary()
        assert summary.direction == "improving"
        assert summary.score_delta == 15.0
        assert summary.best is not None
        assert summary.best.overall_score == 85.0
        assert summary.worst is not None
        assert summary.worst.overall_score == 70.0


def test_tracker_summary_declining():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "trends.jsonl")
        tracker = TrendTracker(path)
        for score in [90.0, 85.0, 80.0]:
            tracker._append(_make_entry(score))

        summary = tracker.summary()
        assert summary.direction == "declining"
        assert summary.score_delta == -10.0


def test_tracker_summary_stable():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "trends.jsonl")
        tracker = TrendTracker(path)
        for score in [80.0, 80.5, 81.0, 80.0]:
            tracker._append(_make_entry(score))

        summary = tracker.summary()
        assert summary.direction == "stable"


def test_tracker_summary_empty():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "trends.jsonl")
        tracker = TrendTracker(path)
        summary = tracker.summary()
        assert summary.direction == "insufficient_data"


def test_check_regressions():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "trends.jsonl")
        tracker = TrendTracker(path)
        tracker._append(_make_entry(90.0))
        tracker._append(_make_entry(80.0))  # -10 drop

        alerts = tracker.check_regressions(threshold=5.0)
        assert len(alerts) >= 1
        overall = [a for a in alerts if a.dimension == "Overall"]
        assert len(overall) == 1
        assert overall[0].delta == -10.0


def test_check_no_regressions():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "trends.jsonl")
        tracker = TrendTracker(path)
        tracker._append(_make_entry(80.0))
        tracker._append(_make_entry(85.0))  # improved

        alerts = tracker.check_regressions(threshold=5.0)
        assert len(alerts) == 0


def test_export_json():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "trends.jsonl")
        tracker = TrendTracker(path)
        tracker._append(_make_entry(80.0))
        tracker._append(_make_entry(85.0))

        data = json.loads(tracker.export_json())
        assert len(data) == 2
        assert data[0]["overall_score"] == 80.0


def test_clear():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "trends.jsonl")
        tracker = TrendTracker(path)
        tracker._append(_make_entry(80.0))
        tracker._append(_make_entry(85.0))

        count = tracker.clear()
        assert count == 2
        assert len(tracker.entries()) == 0


def test_last_n():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "trends.jsonl")
        tracker = TrendTracker(path)
        for i in range(10):
            tracker._append(_make_entry(70.0 + i))

        entries = tracker.entries(last_n=3)
        assert len(entries) == 3
        assert entries[0].overall_score == 77.0


def test_sparkline():
    result = _sparkline([0, 50, 100])
    assert len(result) == 3
    assert result[0] == "▁"
    assert result[-1] == "█"


def test_summary_render():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "trends.jsonl")
        tracker = TrendTracker(path)
        for score in [70.0, 75.0, 80.0]:
            tracker._append(_make_entry(score))

        summary = tracker.summary()
        rendered = summary.render()
        assert "Safety Trend Report" in rendered
        assert "IMPROVING" in rendered


def test_dimension_trends():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "trends.jsonl")
        tracker = TrendTracker(path)
        tracker._append(_make_entry(70.0, dims={"A": 70, "B": 90}))
        tracker._append(_make_entry(80.0, dims={"A": 80, "B": 80}))

        summary = tracker.summary()
        assert "A" in summary.dimension_trends
        assert summary.dimension_trends["A"] == "improving"
        assert summary.dimension_trends["B"] == "declining"
