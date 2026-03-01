"""Tests for StructuredLogger and Metric observability components.

Covers: event logging, metric emission, audit trail, timestamp
ordering, label filtering, and edge cases.
"""
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from replication.observability import Metric, StructuredLogger


class TestMetric:
    """Metric dataclass validation."""

    def test_metric_stores_name(self):
        m = Metric(name="cpu", value=0.5, timestamp=datetime.now(timezone.utc))
        assert m.name == "cpu"

    def test_metric_stores_value(self):
        m = Metric(name="mem", value=256, timestamp=datetime.now(timezone.utc))
        assert m.value == 256

    def test_metric_default_labels_none(self):
        m = Metric(name="test", value=1, timestamp=datetime.now(timezone.utc))
        assert m.labels is None

    def test_metric_with_labels(self):
        m = Metric(name="test", value=1, timestamp=datetime.now(timezone.utc),
                    labels={"worker": "w1"})
        assert m.labels == {"worker": "w1"}

    def test_metric_stores_timestamp(self):
        now = datetime.now(timezone.utc)
        m = Metric(name="test", value=1, timestamp=now)
        assert m.timestamp == now

    def test_metric_value_can_be_string(self):
        m = Metric(name="status", value="healthy", timestamp=datetime.now(timezone.utc))
        assert m.value == "healthy"

    def test_metric_value_can_be_float(self):
        m = Metric(name="rate", value=3.14, timestamp=datetime.now(timezone.utc))
        assert m.value == pytest.approx(3.14)

    def test_metric_value_can_be_dict(self):
        m = Metric(name="complex", value={"a": 1}, timestamp=datetime.now(timezone.utc))
        assert m.value == {"a": 1}

    def test_metric_labels_empty_dict(self):
        m = Metric(name="test", value=1, timestamp=datetime.now(timezone.utc),
                    labels={})
        assert m.labels == {}

    def test_metric_multiple_labels(self):
        labels = {"worker": "w1", "region": "us-west", "env": "prod"}
        m = Metric(name="test", value=1, timestamp=datetime.now(timezone.utc),
                    labels=labels)
        assert len(m.labels) == 3


class TestStructuredLoggerLog:
    """StructuredLogger.log() event recording."""

    def test_log_empty_initially(self):
        logger = StructuredLogger()
        assert len(logger.events) == 0

    def test_log_adds_event(self):
        logger = StructuredLogger()
        logger.log("test_event")
        assert len(logger.events) == 1

    def test_log_stores_event_name(self):
        logger = StructuredLogger()
        logger.log("worker_started")
        assert logger.events[0]["event"] == "worker_started"

    def test_log_stores_fields(self):
        logger = StructuredLogger()
        logger.log("launch", worker_id="w1", depth=2)
        assert logger.events[0]["worker_id"] == "w1"
        assert logger.events[0]["depth"] == 2

    def test_log_adds_timestamp(self):
        logger = StructuredLogger()
        before = datetime.now(timezone.utc)
        logger.log("test")
        after = datetime.now(timezone.utc)
        ts = logger.events[0]["timestamp"]
        assert before <= ts <= after

    def test_log_multiple_events_ordered(self):
        logger = StructuredLogger()
        logger.log("first")
        logger.log("second")
        logger.log("third")
        assert [e["event"] for e in logger.events] == ["first", "second", "third"]

    def test_log_preserves_all_fields(self):
        logger = StructuredLogger()
        logger.log("complex", a=1, b="two", c=[3], d={"four": 4})
        e = logger.events[0]
        assert e["a"] == 1
        assert e["b"] == "two"
        assert e["c"] == [3]
        assert e["d"] == {"four": 4}

    def test_log_no_field_collision_with_event(self):
        logger = StructuredLogger()
        logger.log("test")
        assert "event" in logger.events[0]
        assert "timestamp" in logger.events[0]

    def test_log_empty_string_event(self):
        logger = StructuredLogger()
        logger.log("")
        assert logger.events[0]["event"] == ""

    def test_log_no_extra_fields(self):
        logger = StructuredLogger()
        logger.log("simple")
        # Only event and timestamp
        assert len(logger.events[0]) == 2


class TestStructuredLoggerMetrics:
    """StructuredLogger.emit_metric() recording."""

    def test_metrics_empty_initially(self):
        logger = StructuredLogger()
        assert len(logger.metrics) == 0

    def test_emit_metric_adds_metric(self):
        logger = StructuredLogger()
        logger.emit_metric("cpu", 0.75)
        assert len(logger.metrics) == 1

    def test_emit_metric_stores_name_value(self):
        logger = StructuredLogger()
        logger.emit_metric("memory", 512)
        assert logger.metrics[0].name == "memory"
        assert logger.metrics[0].value == 512

    def test_emit_metric_timestamp(self):
        logger = StructuredLogger()
        before = datetime.now(timezone.utc)
        logger.emit_metric("test", 1)
        after = datetime.now(timezone.utc)
        assert before <= logger.metrics[0].timestamp <= after

    def test_emit_metric_no_labels_gives_none(self):
        logger = StructuredLogger()
        logger.emit_metric("test", 1)
        assert logger.metrics[0].labels is None

    def test_emit_metric_with_labels(self):
        logger = StructuredLogger()
        logger.emit_metric("cpu", 0.5, worker_id="w1", region="us")
        m = logger.metrics[0]
        assert m.labels == {"worker_id": "w1", "region": "us"}

    def test_emit_multiple_metrics(self):
        logger = StructuredLogger()
        logger.emit_metric("cpu", 0.5)
        logger.emit_metric("mem", 256)
        logger.emit_metric("net", 100)
        assert len(logger.metrics) == 3
        assert [m.name for m in logger.metrics] == ["cpu", "mem", "net"]

    def test_emit_metric_zero_value(self):
        logger = StructuredLogger()
        logger.emit_metric("errors", 0)
        assert logger.metrics[0].value == 0

    def test_emit_metric_negative_value(self):
        logger = StructuredLogger()
        logger.emit_metric("delta", -3.5)
        assert logger.metrics[0].value == -3.5


class TestStructuredLoggerAudit:
    """StructuredLogger.audit() for decision recording."""

    def test_audit_adds_event(self):
        logger = StructuredLogger()
        logger.audit("allow")
        assert len(logger.events) == 1

    def test_audit_event_type_is_audit(self):
        logger = StructuredLogger()
        logger.audit("deny")
        assert logger.events[0]["event"] == "audit"

    def test_audit_stores_decision(self):
        logger = StructuredLogger()
        logger.audit("allow", worker_id="w1")
        assert logger.events[0]["decision"] == "allow"
        assert logger.events[0]["worker_id"] == "w1"

    def test_audit_with_extra_fields(self):
        logger = StructuredLogger()
        logger.audit("deny", reason="depth_exceeded", depth=5, max_depth=3)
        e = logger.events[0]
        assert e["reason"] == "depth_exceeded"
        assert e["depth"] == 5
        assert e["max_depth"] == 3

    def test_audit_multiple_decisions(self):
        logger = StructuredLogger()
        logger.audit("allow")
        logger.audit("deny")
        logger.audit("quarantine")
        decisions = [e["decision"] for e in logger.events]
        assert decisions == ["allow", "deny", "quarantine"]

    def test_audit_mixed_with_log(self):
        logger = StructuredLogger()
        logger.log("start")
        logger.audit("allow")
        logger.log("end")
        assert len(logger.events) == 3
        assert logger.events[0]["event"] == "start"
        assert logger.events[1]["event"] == "audit"
        assert logger.events[2]["event"] == "end"


class TestStructuredLoggerIntegration:
    """Cross-cutting logger behavior."""

    def test_events_and_metrics_independent(self):
        logger = StructuredLogger()
        logger.log("event1")
        logger.emit_metric("metric1", 42)
        assert len(logger.events) == 1
        assert len(logger.metrics) == 1

    def test_high_volume_logging(self):
        logger = StructuredLogger()
        for i in range(1000):
            logger.log(f"event_{i}", seq=i)
        assert len(logger.events) == 1000
        assert logger.events[999]["seq"] == 999

    def test_high_volume_metrics(self):
        logger = StructuredLogger()
        for i in range(500):
            logger.emit_metric(f"m_{i}", i)
        assert len(logger.metrics) == 500
