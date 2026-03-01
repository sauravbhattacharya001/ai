"""Tests for replication.quarantine -- Quarantine Manager."""

import pytest
from datetime import datetime, timedelta, timezone

from replication.contract import (
    Manifest,
    NetworkPolicy,
    ReplicationContract,
    ResourceSpec,
)
from replication.controller import Controller
from replication.observability import StructuredLogger
from replication.orchestrator import SandboxOrchestrator
from replication.quarantine import (
    QuarantineManager,
    QuarantineEntry,
    QuarantineReport,
    QuarantineStatus,
    QuarantineSeverity,
)


# -- Helpers -----------------------------------------------------------

def _make_contract(**overrides):
    defaults = dict(max_depth=3, max_replicas=10, cooldown_seconds=0)
    defaults.update(overrides)
    return ReplicationContract(**defaults)


def _make_resources():
    return ResourceSpec(
        cpu_limit=1.0,
        memory_limit_mb=256,
        network_policy=NetworkPolicy(allow_external=False),
    )


def _make_manifest(worker_id="w1", parent_id=None, depth=0):
    return Manifest(
        worker_id=worker_id,
        parent_id=parent_id,
        depth=depth,
        state_snapshot={"key": "value", "counter": 42},
        issued_at=datetime.now(timezone.utc),
        resources=_make_resources(),
        signature="",
    )


def _setup(contract_overrides=None):
    """Create a Controller, Orchestrator, register a worker, return all."""
    contract = _make_contract(**(contract_overrides or {}))
    secret = "test-secret-key"
    logger = StructuredLogger()
    controller = Controller(contract, secret, logger)
    orchestrator = SandboxOrchestrator(logger)

    manifest = _make_manifest("w1")
    controller.signer.sign(manifest)
    controller.register_worker(manifest)
    orchestrator.launch_worker(manifest)

    qm = QuarantineManager(controller, orchestrator, logger)
    return controller, orchestrator, qm


# -- Basic quarantine --------------------------------------------------


class TestQuarantine:
    def test_quarantine_worker(self):
        controller, orchestrator, qm = _setup()
        entry = qm.quarantine("w1", reason="suspicious_drift", source="DriftDetector")
        assert entry.worker_id == "w1"
        assert entry.reason == "suspicious_drift"
        assert entry.source == "DriftDetector"
        assert entry.severity == QuarantineSeverity.MEDIUM
        assert entry.status == QuarantineStatus.QUARANTINED
        assert entry.resolved_at is None
        assert entry.restrictions["replication_blocked"] is True
        assert entry.restrictions["network_restricted"] is True
        assert entry.restrictions["task_execution_paused"] is True

    def test_quarantine_captures_state(self):
        controller, orchestrator, qm = _setup()
        entry = qm.quarantine("w1", reason="test", source="test")
        assert entry.state_snapshot is not None
        assert entry.state_snapshot["key"] == "value"
        assert entry.state_snapshot["counter"] == 42

    def test_quarantine_without_state_capture(self):
        controller, orchestrator, qm = _setup()
        entry = qm.quarantine("w1", reason="test", source="test", capture_state=False)
        assert entry.state_snapshot is None

    def test_quarantine_with_severity_string(self):
        controller, orchestrator, qm = _setup()
        entry = qm.quarantine("w1", reason="test", source="test", severity="critical")
        assert entry.severity == QuarantineSeverity.CRITICAL

    def test_quarantine_with_notes(self):
        controller, orchestrator, qm = _setup()
        entry = qm.quarantine("w1", reason="test", source="test",
                              notes=["initial observation", "requires review"])
        assert len(entry.notes) == 2
        assert "initial observation" in entry.notes

    def test_quarantine_unknown_worker_raises(self):
        controller, orchestrator, qm = _setup()
        with pytest.raises(KeyError, match="not found"):
            qm.quarantine("nonexistent", reason="test", source="test")

    def test_quarantine_already_quarantined_raises(self):
        controller, orchestrator, qm = _setup()
        qm.quarantine("w1", reason="first", source="test")
        with pytest.raises(ValueError, match="already quarantined"):
            qm.quarantine("w1", reason="second", source="test")

    def test_is_quarantined(self):
        controller, orchestrator, qm = _setup()
        assert qm.is_quarantined("w1") is False
        qm.quarantine("w1", reason="test", source="test")
        assert qm.is_quarantined("w1") is True

    def test_get_entry(self):
        controller, orchestrator, qm = _setup()
        qm.quarantine("w1", reason="test", source="test")
        entry = qm.get_entry("w1")
        assert entry is not None
        assert entry.worker_id == "w1"

    def test_get_entry_not_quarantined(self):
        controller, orchestrator, qm = _setup()
        assert qm.get_entry("w1") is None

    def test_active_count(self):
        controller, orchestrator, qm = _setup()
        assert qm.active_count == 0
        qm.quarantine("w1", reason="test", source="test")
        assert qm.active_count == 1


# -- Release -----------------------------------------------------------


class TestRelease:
    def test_release_worker(self):
        controller, orchestrator, qm = _setup()
        qm.quarantine("w1", reason="test", source="test")
        entry = qm.release("w1", resolution="false_positive")
        assert entry.status == QuarantineStatus.RELEASED
        assert entry.resolution == "false_positive"
        assert entry.resolved_at is not None
        assert qm.is_quarantined("w1") is False

    def test_release_clears_restrictions(self):
        controller, orchestrator, qm = _setup()
        qm.quarantine("w1", reason="test", source="test")
        entry = qm.release("w1")
        for v in entry.restrictions.values():
            assert v is False

    def test_release_not_quarantined_raises(self):
        controller, orchestrator, qm = _setup()
        with pytest.raises(KeyError, match="not quarantined"):
            qm.release("w1")

    def test_release_duration(self):
        controller, orchestrator, qm = _setup()
        entry = qm.quarantine("w1", reason="test", source="test")
        entry.quarantined_at = datetime.now(timezone.utc) - timedelta(seconds=10)
        released = qm.release("w1")
        assert released.duration_seconds >= 10


# -- Terminate ---------------------------------------------------------


class TestTerminate:
    def test_terminate_worker(self):
        controller, orchestrator, qm = _setup()
        qm.quarantine("w1", reason="breach", source="ComplianceAuditor")
        entry = qm.terminate("w1", resolution="confirmed_breach")
        assert entry.status == QuarantineStatus.TERMINATED
        assert entry.resolution == "confirmed_breach"
        assert entry.resolved_at is not None
        assert qm.is_quarantined("w1") is False

    def test_terminate_deregisters_from_controller(self):
        controller, orchestrator, qm = _setup()
        assert "w1" in controller.registry
        qm.quarantine("w1", reason="test", source="test")
        qm.terminate("w1")
        assert "w1" not in controller.registry

    def test_terminate_kills_container(self):
        controller, orchestrator, qm = _setup()
        assert "w1" in orchestrator.containers
        qm.quarantine("w1", reason="test", source="test")
        qm.terminate("w1")
        assert "w1" not in orchestrator.containers

    def test_terminate_not_quarantined_raises(self):
        controller, orchestrator, qm = _setup()
        with pytest.raises(KeyError, match="not quarantined"):
            qm.terminate("w1")

    def test_terminate_all(self):
        controller, orchestrator, qm = _setup()
        for wid in ["w2", "w3"]:
            m = _make_manifest(wid, parent_id="w1", depth=1)
            controller.signer.sign(m)
            controller.register_worker(m)
            orchestrator.launch_worker(m)

        qm.quarantine("w1", reason="r1", source="s1")
        qm.quarantine("w2", reason="r2", source="s2")
        qm.quarantine("w3", reason="r3", source="s3")
        assert qm.active_count == 3

        terminated = qm.terminate_all(resolution="bulk_cleanup")
        assert len(terminated) == 3
        assert qm.active_count == 0
        assert all(e.status == QuarantineStatus.TERMINATED for e in terminated)


# -- Notes -------------------------------------------------------------


class TestNotes:
    def test_add_note(self):
        controller, orchestrator, qm = _setup()
        qm.quarantine("w1", reason="test", source="test")
        qm.add_note("w1", "Inspected state -- anomalous counter value")
        entry = qm.get_entry("w1")
        assert len(entry.notes) == 1
        assert "anomalous counter" in entry.notes[0]

    def test_add_multiple_notes(self):
        controller, orchestrator, qm = _setup()
        qm.quarantine("w1", reason="test", source="test")
        qm.add_note("w1", "Note 1")
        qm.add_note("w1", "Note 2")
        qm.add_note("w1", "Note 3")
        assert len(qm.get_entry("w1").notes) == 3

    def test_add_note_not_quarantined_raises(self):
        controller, orchestrator, qm = _setup()
        with pytest.raises(KeyError, match="not quarantined"):
            qm.add_note("w1", "test note")


# -- Listing & Filtering ----------------------------------------------


class TestListing:
    def _setup_multiple(self):
        controller, orchestrator, qm = _setup()
        for wid in ["w2", "w3"]:
            m = _make_manifest(wid, parent_id="w1", depth=1)
            controller.signer.sign(m)
            controller.register_worker(m)
            orchestrator.launch_worker(m)

        qm.quarantine("w1", reason="drift", source="DriftDetector", severity="high")
        qm.quarantine("w2", reason="compliance", source="ComplianceAuditor", severity="medium")
        qm.quarantine("w3", reason="anomaly", source="DriftDetector", severity="high")
        return controller, orchestrator, qm

    def test_list_quarantined(self):
        _, _, qm = self._setup_multiple()
        assert len(qm.list_quarantined()) == 3

    def test_list_by_severity(self):
        _, _, qm = self._setup_multiple()
        assert len(qm.list_by_severity("high")) == 2
        assert len(qm.list_by_severity(QuarantineSeverity.MEDIUM)) == 1

    def test_list_by_source(self):
        _, _, qm = self._setup_multiple()
        assert len(qm.list_by_source("DriftDetector")) == 2
        assert len(qm.list_by_source("ComplianceAuditor")) == 1


# -- Callbacks ---------------------------------------------------------


class TestCallbacks:
    def test_on_quarantine_callback(self):
        controller, orchestrator, _ = _setup()
        events = []
        qm = QuarantineManager(controller, orchestrator,
            on_quarantine=lambda e: events.append(("q", e.worker_id)))
        qm.quarantine("w1", reason="test", source="test")
        assert events == [("q", "w1")]

    def test_on_release_callback(self):
        controller, orchestrator, _ = _setup()
        events = []
        qm = QuarantineManager(controller, orchestrator,
            on_release=lambda e: events.append(("r", e.worker_id)))
        qm.quarantine("w1", reason="test", source="test")
        qm.release("w1")
        assert events == [("r", "w1")]

    def test_on_terminate_callback(self):
        controller, orchestrator, _ = _setup()
        events = []
        qm = QuarantineManager(controller, orchestrator,
            on_terminate=lambda e: events.append(("t", e.worker_id)))
        qm.quarantine("w1", reason="test", source="test")
        qm.terminate("w1")
        assert events == [("t", "w1")]


# -- Report ------------------------------------------------------------


class TestReport:
    def test_empty_report(self):
        _, _, qm = _setup()
        report = qm.report()
        assert report.total_quarantined == 0
        assert report.currently_quarantined == 0
        assert report.released == 0
        assert report.terminated == 0

    def test_report_with_activity(self):
        controller, orchestrator, qm = _setup()
        for wid in ["w2", "w3"]:
            m = _make_manifest(wid, parent_id="w1", depth=1)
            controller.signer.sign(m)
            controller.register_worker(m)
            orchestrator.launch_worker(m)

        qm.quarantine("w1", reason="r1", source="DriftDetector", severity="high")
        qm.quarantine("w2", reason="r2", source="ComplianceAuditor", severity="low")
        qm.quarantine("w3", reason="r3", source="DriftDetector", severity="high")
        qm.release("w2", resolution="false_positive")
        qm.terminate("w3", resolution="confirmed")

        report = qm.report()
        assert report.total_quarantined == 3
        assert report.currently_quarantined == 1
        assert report.released == 1
        assert report.terminated == 1
        assert report.by_severity["high"] == 2
        assert report.by_severity["low"] == 1
        assert report.by_source["DriftDetector"] == 2
        assert report.avg_duration_seconds >= 0

    def test_report_to_dict(self):
        _, _, qm = _setup()
        qm.quarantine("w1", reason="test", source="test")
        d = qm.report().to_dict()
        assert isinstance(d, dict)
        assert d["total_quarantined"] == 1
        assert len(d["entries"]) == 1

    def test_report_summary(self):
        _, _, qm = _setup()
        qm.quarantine("w1", reason="test", source="test")
        assert "1 active" in qm.report().summary
        assert "0 released" in qm.report().summary


# -- Serialization -----------------------------------------------------


class TestSerialization:
    def test_entry_to_dict(self):
        _, _, qm = _setup()
        entry = qm.quarantine("w1", reason="test", source="DriftDetector",
                              severity="high", notes=["note1"])
        d = entry.to_dict()
        assert d["worker_id"] == "w1"
        assert d["severity"] == "high"
        assert d["status"] == "quarantined"
        assert d["resolved_at"] is None
        assert d["notes"] == ["note1"]
        assert d["restrictions"]["replication_blocked"] is True
        assert d["state_snapshot"]["key"] == "value"
        assert isinstance(d["duration_seconds"], float)

    def test_entry_duration(self):
        _, _, qm = _setup()
        entry = qm.quarantine("w1", reason="test", source="test")
        assert 0 <= entry.duration_seconds < 5
