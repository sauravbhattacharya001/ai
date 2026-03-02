"""Tests for the Incident Response Playbook module."""

from replication import (
    Controller,
    IncidentResponder,
    IncidentConfig,
    IncidentCategory,
    IncidentSeverity,
    Playbook,
    ResponseStep,
    StepPriority,
    StepStatus,
)
from replication.drift import DriftAlert, DriftSeverity, DriftDirection
from replication.compliance import Finding, Verdict
import json


def _controller() -> Controller:
    from replication import ReplicationContract, StructuredLogger
    c = ReplicationContract(max_depth=3, max_replicas=10, cooldown_seconds=0.0)
    return Controller(contract=c, secret="test-secret", logger=StructuredLogger())


class TestFromDriftAlert:
    def test_creates_playbook(self) -> None:
        r = IncidentResponder(_controller())
        alert = DriftAlert(
            metric="escape_rate", direction=DriftDirection.WORSENING,
            severity=DriftSeverity.CRITICAL, slope=0.05, r_squared=0.9,
            start_value=0.02, end_value=0.15, change_pct=650.0,
            description="escape_rate drifting dangerously",
        )
        pb = r.from_drift_alert(alert, worker_ids=["w-1"])
        assert isinstance(pb, Playbook)
        assert pb.category == IncidentCategory.DRIFT
        assert pb.severity == IncidentSeverity.CRITICAL
        assert len(pb.steps) >= 5
        assert "w-1" in pb.affected_workers

    def test_low_severity_no_quarantine_step(self) -> None:
        r = IncidentResponder(_controller())
        alert = DriftAlert(
            metric="latency", direction=DriftDirection.WORSENING,
            severity=DriftSeverity.LOW, slope=0.01, r_squared=0.3,
            start_value=1.0, end_value=1.1, change_pct=10.0,
            description="minor latency drift",
        )
        pb = r.from_drift_alert(alert)
        actions = [s.action for s in pb.steps]
        assert "Quarantine affected workers" not in actions


class TestFromComplianceFinding:
    def test_creates_playbook(self) -> None:
        r = IncidentResponder(_controller())
        from replication.compliance import Framework
        finding = Finding(
            check_id="max_workers", framework=Framework.NIST,
            title="Too many workers",
            verdict=Verdict.FAIL, rationale="Worker count exceeds limit",
        )
        pb = r.from_compliance_finding(finding, worker_ids=["w-2"])
        assert pb.category == IncidentCategory.COMPLIANCE
        assert pb.severity == IncidentSeverity.HIGH
        assert any("Re-audit" in s.action for s in pb.steps)


class TestFromEscapeEvent:
    def test_critical_escape(self) -> None:
        r = IncidentResponder(_controller())
        pb = r.from_escape_event("w-bad", details={"vector": "network"})
        assert pb.severity == IncidentSeverity.CRITICAL
        assert pb.category == IncidentCategory.ESCAPE
        assert len(pb.steps) == 10
        assert pb.steps[0].priority == StepPriority.IMMEDIATE


class TestFromResourceAnomaly:
    def test_high_ratio(self) -> None:
        r = IncidentResponder(_controller())
        pb = r.from_resource_anomaly("cpu_percent", 95.0, 50.0, ["w-3"])
        assert pb.severity == IncidentSeverity.HIGH
        assert pb.category == IncidentCategory.RESOURCE

    def test_extreme_ratio_critical(self) -> None:
        r = IncidentResponder(_controller())
        pb = r.from_resource_anomaly("memory_mb", 8000.0, 2000.0)
        assert pb.severity == IncidentSeverity.CRITICAL


class TestPlaybookLifecycle:
    def test_progress_tracking(self) -> None:
        r = IncidentResponder(_controller())
        pb = r.from_escape_event("w-x")
        assert pb.progress == 0.0
        pb.steps[0].complete("done")
        assert pb.progress > 0.0
        assert pb.steps[0].status == StepStatus.COMPLETED

    def test_resolve(self) -> None:
        r = IncidentResponder(_controller())
        pb = r.from_escape_event("w-x")
        pb.resolve("False alarm")
        assert pb.resolved is True
        assert pb.resolved_at is not None

    def test_next_step(self) -> None:
        r = IncidentResponder(_controller())
        pb = r.from_escape_event("w-x")
        first = pb.next_step
        assert first is not None
        assert first.id == 1

    def test_json_export(self) -> None:
        r = IncidentResponder(_controller())
        pb = r.from_escape_event("w-x")
        data = json.loads(pb.to_json())
        assert "steps" in data
        assert data["category"] == "escape"


class TestManagement:
    def test_list_and_summary(self) -> None:
        r = IncidentResponder(_controller())
        r.from_escape_event("w-1")
        r.from_resource_anomaly("cpu", 80, 50)
        assert len(r.list_active()) == 2
        r.list_all()[0].resolve()
        assert len(r.list_active()) == 1
        s = r.summary()
        assert s["total_incidents"] == 2
        assert s["active"] == 1

    def test_export_all(self) -> None:
        r = IncidentResponder(_controller())
        r.from_escape_event("w-1")
        data = json.loads(r.export_all())
        assert "summary" in data
        assert len(data["playbooks"]) == 1
