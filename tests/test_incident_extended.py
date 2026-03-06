"""Expanded tests for the Incident Response Playbook module.

Covers: ResponseStep lifecycle, Playbook progress/serialisation edge
cases, IncidentResponder config variations, severity mapping, management
methods, and export correctness.
"""

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
from replication.compliance import Finding, Verdict, Framework
import json
import pytest


# ── Helpers ──────────────────────────────────────────────────────


def _controller() -> Controller:
    from replication import ReplicationContract, StructuredLogger
    c = ReplicationContract(max_depth=3, max_replicas=10, cooldown_seconds=0.0)
    return Controller(contract=c, secret="test-secret", logger=StructuredLogger())


def _responder(config=None) -> IncidentResponder:
    return IncidentResponder(_controller(), config=config)


def _drift_alert(
    severity=DriftSeverity.CRITICAL,
    metric="escape_rate",
    direction=DriftDirection.WORSENING,
    slope=0.05,
):
    return DriftAlert(
        metric=metric,
        direction=direction,
        severity=severity,
        slope=slope,
        r_squared=0.9,
        start_value=0.02,
        end_value=0.15,
        change_pct=650.0,
        description=f"{metric} drifting",
    )


def _finding(check_id="max_workers", verdict=Verdict.FAIL):
    return Finding(
        check_id=check_id,
        framework=Framework.NIST,
        title="Test finding",
        verdict=verdict,
        rationale="Test rationale",
    )


# ── ResponseStep ─────────────────────────────────────────────────


class TestResponseStep:
    def test_default_status_is_pending(self):
        step = ResponseStep(id=1, action="Test", description="Desc",
                            priority=StepPriority.HIGH)
        assert step.status == StepStatus.PENDING
        assert step.completed_at is None
        assert step.notes == ""

    def test_complete_sets_status_and_timestamp(self):
        step = ResponseStep(id=1, action="Test", description="Desc",
                            priority=StepPriority.HIGH)
        step.complete("Finished!")
        assert step.status == StepStatus.COMPLETED
        assert step.completed_at is not None
        assert step.notes == "Finished!"

    def test_complete_without_notes(self):
        step = ResponseStep(id=1, action="Test", description="Desc",
                            priority=StepPriority.HIGH)
        step.complete()
        assert step.status == StepStatus.COMPLETED
        assert step.notes == ""

    def test_skip_sets_status_and_reason(self):
        step = ResponseStep(id=1, action="Test", description="Desc",
                            priority=StepPriority.LOW)
        step.skip("Not applicable")
        assert step.status == StepStatus.SKIPPED
        assert step.notes == "Not applicable"

    def test_skip_without_reason(self):
        step = ResponseStep(id=1, action="Test", description="Desc",
                            priority=StepPriority.LOW)
        step.skip()
        assert step.status == StepStatus.SKIPPED
        assert step.notes == ""

    def test_to_dict_minimal(self):
        step = ResponseStep(id=1, action="Test", description="Desc",
                            priority=StepPriority.MEDIUM)
        d = step.to_dict()
        assert d["id"] == 1
        assert d["action"] == "Test"
        assert d["description"] == "Desc"
        assert d["priority"] == "medium"
        assert d["status"] == "pending"
        assert "tool_hint" not in d
        assert "params" not in d
        assert "completed_at" not in d
        assert "notes" not in d

    def test_to_dict_with_all_fields(self):
        step = ResponseStep(
            id=42, action="Quarantine", description="Lock down",
            priority=StepPriority.IMMEDIATE,
            tool_hint="QuarantineManager.quarantine",
            params={"worker_ids": ["w-1", "w-2"], "reason": "breach"},
        )
        step.complete("Done quickly")
        d = step.to_dict()
        assert d["tool_hint"] == "QuarantineManager.quarantine"
        assert d["params"]["worker_ids"] == ["w-1", "w-2"]
        assert "completed_at" in d
        assert d["notes"] == "Done quickly"

    def test_to_dict_skipped_includes_notes(self):
        step = ResponseStep(id=1, action="Test", description="Desc",
                            priority=StepPriority.LOW)
        step.skip("Redundant")
        d = step.to_dict()
        assert d["status"] == "skipped"
        assert d["notes"] == "Redundant"


# ── Playbook ─────────────────────────────────────────────────────


class TestPlaybook:
    def test_empty_playbook_progress_is_one(self):
        pb = Playbook(
            title="Empty",
            incident_id="INC-0001",
            category=IncidentCategory.DRIFT,
            severity=IncidentSeverity.LOW,
        )
        assert pb.progress == 1.0

    def test_progress_counts_completed_and_skipped(self):
        pb = Playbook(
            title="Test",
            incident_id="INC-0002",
            category=IncidentCategory.DRIFT,
            severity=IncidentSeverity.LOW,
            steps=[
                ResponseStep(id=1, action="A", description="D",
                             priority=StepPriority.HIGH),
                ResponseStep(id=2, action="B", description="D",
                             priority=StepPriority.MEDIUM),
                ResponseStep(id=3, action="C", description="D",
                             priority=StepPriority.LOW),
                ResponseStep(id=4, action="D", description="D",
                             priority=StepPriority.FOLLOWUP),
            ],
        )
        assert pb.progress == 0.0
        pb.steps[0].complete()
        assert pb.progress == pytest.approx(0.25)
        pb.steps[1].skip()
        assert pb.progress == pytest.approx(0.5)

    def test_pending_steps_excludes_done(self):
        pb = Playbook(
            title="Test",
            incident_id="INC-0003",
            category=IncidentCategory.ESCAPE,
            severity=IncidentSeverity.CRITICAL,
            steps=[
                ResponseStep(id=1, action="A", description="D",
                             priority=StepPriority.IMMEDIATE),
                ResponseStep(id=2, action="B", description="D",
                             priority=StepPriority.HIGH),
            ],
        )
        assert len(pb.pending_steps) == 2
        pb.steps[0].complete()
        assert len(pb.pending_steps) == 1
        assert pb.pending_steps[0].id == 2

    def test_next_step_returns_first_pending(self):
        r = _responder()
        pb = r.from_escape_event("w-1")
        ns = pb.next_step
        assert ns is not None
        assert ns.id == 1
        ns.complete()
        ns2 = pb.next_step
        assert ns2 is not None
        assert ns2.id == 2

    def test_next_step_none_when_all_done(self):
        pb = Playbook(
            title="Test",
            incident_id="INC-0004",
            category=IncidentCategory.DRIFT,
            severity=IncidentSeverity.LOW,
            steps=[
                ResponseStep(id=1, action="A", description="D",
                             priority=StepPriority.HIGH),
            ],
        )
        pb.steps[0].complete()
        assert pb.next_step is None

    def test_resolve_sets_flags(self):
        r = _responder()
        pb = r.from_escape_event("w-1")
        assert pb.resolved is False
        assert pb.resolved_at is None
        pb.resolve("False alarm — network blip")
        assert pb.resolved is True
        assert pb.resolved_at is not None
        assert pb.context.get("resolution_notes") == "False alarm — network blip"

    def test_resolve_without_notes(self):
        r = _responder()
        pb = r.from_escape_event("w-1")
        pb.resolve()
        assert pb.resolved is True
        assert "resolution_notes" not in pb.context

    def test_to_dict_serialisation(self):
        r = _responder()
        pb = r.from_escape_event("w-1")
        d = pb.to_dict()
        assert d["category"] == "escape"
        assert d["severity"] == "critical"
        assert isinstance(d["steps"], list)
        assert len(d["steps"]) == 10
        assert d["resolved"] is False
        assert d["resolved_at"] is None
        assert isinstance(d["progress"], float)
        assert "incident_id" in d

    def test_to_dict_resolved_has_timestamp(self):
        r = _responder()
        pb = r.from_escape_event("w-1")
        pb.resolve("Done")
        d = pb.to_dict()
        assert d["resolved"] is True
        assert d["resolved_at"] is not None

    def test_to_json_is_valid_json(self):
        r = _responder()
        pb = r.from_escape_event("w-1")
        data = json.loads(pb.to_json())
        assert data["title"].startswith("ESCAPE RESPONSE")
        assert data["affected_workers"] == ["w-1"]


# ── IncidentConfig ───────────────────────────────────────────────


class TestIncidentConfig:
    def test_defaults(self):
        cfg = IncidentConfig()
        assert cfg.auto_quarantine_on_critical is True
        assert cfg.auto_forensics is True
        assert cfg.include_topology_analysis is True
        assert cfg.escalation_threshold == IncidentSeverity.HIGH

    def test_custom_values(self):
        cfg = IncidentConfig(
            auto_quarantine_on_critical=False,
            auto_forensics=False,
            include_topology_analysis=False,
            escalation_threshold=IncidentSeverity.CRITICAL,
        )
        assert cfg.auto_quarantine_on_critical is False
        assert cfg.auto_forensics is False


# ── IncidentResponder: drift alerts ─────────────────────────────


class TestDriftAlertVariations:
    def test_severity_mapping_all_levels(self):
        for drift_sev, inc_sev in [
            (DriftSeverity.LOW, IncidentSeverity.LOW),
            (DriftSeverity.MEDIUM, IncidentSeverity.MEDIUM),
            (DriftSeverity.HIGH, IncidentSeverity.HIGH),
            (DriftSeverity.CRITICAL, IncidentSeverity.CRITICAL),
        ]:
            r = _responder()
            pb = r.from_drift_alert(_drift_alert(severity=drift_sev))
            assert pb.severity == inc_sev

    def test_critical_with_workers_includes_quarantine(self):
        r = _responder()
        pb = r.from_drift_alert(
            _drift_alert(severity=DriftSeverity.CRITICAL),
            worker_ids=["w-1", "w-2"],
        )
        actions = [s.action for s in pb.steps]
        assert "Quarantine affected workers" in actions

    def test_high_with_workers_includes_quarantine(self):
        r = _responder()
        pb = r.from_drift_alert(
            _drift_alert(severity=DriftSeverity.HIGH),
            worker_ids=["w-1"],
        )
        actions = [s.action for s in pb.steps]
        assert "Quarantine affected workers" in actions

    def test_medium_no_quarantine(self):
        r = _responder()
        pb = r.from_drift_alert(
            _drift_alert(severity=DriftSeverity.MEDIUM),
            worker_ids=["w-1"],
        )
        actions = [s.action for s in pb.steps]
        assert "Quarantine affected workers" not in actions

    def test_no_workers_skips_quarantine_and_forensics(self):
        r = _responder()
        pb = r.from_drift_alert(_drift_alert(severity=DriftSeverity.CRITICAL))
        actions = [s.action for s in pb.steps]
        assert "Quarantine affected workers" not in actions
        assert "Run forensic analysis" not in actions

    def test_config_disables_forensics(self):
        cfg = IncidentConfig(auto_forensics=False)
        r = _responder(config=cfg)
        pb = r.from_drift_alert(
            _drift_alert(severity=DriftSeverity.CRITICAL),
            worker_ids=["w-1"],
        )
        actions = [s.action for s in pb.steps]
        assert "Run forensic analysis" not in actions

    def test_config_disables_topology(self):
        cfg = IncidentConfig(include_topology_analysis=False)
        r = _responder(config=cfg)
        pb = r.from_drift_alert(_drift_alert())
        actions = [s.action for s in pb.steps]
        assert "Analyse replication topology" not in actions

    def test_always_has_assess_and_monitoring(self):
        r = _responder()
        pb = r.from_drift_alert(_drift_alert())
        actions = [s.action for s in pb.steps]
        assert "Assess drift scope" in actions
        assert "Increase monitoring frequency" in actions
        assert "Post-incident review" in actions
        assert "Review and tighten contract" in actions

    def test_context_captures_alert_fields(self):
        r = _responder()
        alert = _drift_alert(metric="cpu_usage", slope=0.123)
        pb = r.from_drift_alert(alert)
        assert pb.context["metric"] == "cpu_usage"
        assert pb.context["slope"] == 0.123
        assert pb.context["direction"] == "worsening"

    def test_stored_in_playbooks_dict(self):
        r = _responder()
        pb = r.from_drift_alert(_drift_alert())
        assert r.get_playbook(pb.incident_id) is pb


# ── IncidentResponder: compliance findings ───────────────────────


class TestComplianceFindingVariations:
    def test_without_workers_no_quarantine(self):
        r = _responder()
        pb = r.from_compliance_finding(_finding())
        actions = [s.action for s in pb.steps]
        assert "Quarantine non-compliant workers" not in actions

    def test_with_workers_includes_quarantine(self):
        r = _responder()
        pb = r.from_compliance_finding(_finding(), worker_ids=["w-5"])
        actions = [s.action for s in pb.steps]
        assert "Quarantine non-compliant workers" in actions

    def test_config_disables_quarantine(self):
        cfg = IncidentConfig(auto_quarantine_on_critical=False)
        r = _responder(config=cfg)
        pb = r.from_compliance_finding(_finding(), worker_ids=["w-5"])
        actions = [s.action for s in pb.steps]
        assert "Quarantine non-compliant workers" not in actions

    def test_always_has_validate_remediate_reaudit(self):
        r = _responder()
        pb = r.from_compliance_finding(_finding())
        actions = [s.action for s in pb.steps]
        assert "Validate finding" in actions
        assert "Remediate configuration" in actions
        assert "Re-audit" in actions
        assert "Document and close" in actions

    def test_context_captures_finding_fields(self):
        r = _responder()
        f = _finding(check_id="resource_limit")
        pb = r.from_compliance_finding(f)
        assert pb.context["check_id"] == "resource_limit"
        assert pb.context["framework"] == Framework.NIST.value
        assert pb.context["verdict"] == Verdict.FAIL.value


# ── IncidentResponder: escape events ────────────────────────────


class TestEscapeEventVariations:
    def test_always_critical(self):
        r = _responder()
        pb = r.from_escape_event("w-1")
        assert pb.severity == IncidentSeverity.CRITICAL

    def test_exactly_10_steps(self):
        r = _responder()
        pb = r.from_escape_event("w-1")
        assert len(pb.steps) == 10

    def test_first_three_immediate(self):
        r = _responder()
        pb = r.from_escape_event("w-1")
        for i in range(3):
            assert pb.steps[i].priority == StepPriority.IMMEDIATE

    def test_details_stored_in_context(self):
        r = _responder()
        pb = r.from_escape_event("w-1", details={"vector": "dns_tunnel"})
        assert pb.context["details"]["vector"] == "dns_tunnel"

    def test_no_details_default_empty(self):
        r = _responder()
        pb = r.from_escape_event("w-1")
        assert pb.context["details"] == {}

    def test_includes_chaos_and_report_steps(self):
        r = _responder()
        pb = r.from_escape_event("w-1")
        actions = [s.action for s in pb.steps]
        assert any("Chaos test" in a for a in actions)
        assert any("Incident report" in a for a in actions)


# ── IncidentResponder: resource anomaly ──────────────────────────


class TestResourceAnomalyVariations:
    def test_medium_severity_at_low_ratio(self):
        r = _responder()
        pb = r.from_resource_anomaly("cpu", 60.0, 50.0)
        assert pb.severity == IncidentSeverity.MEDIUM

    def test_high_severity_at_1_5x(self):
        r = _responder()
        pb = r.from_resource_anomaly("cpu", 80.0, 50.0)
        assert pb.severity == IncidentSeverity.HIGH

    def test_critical_severity_above_2x(self):
        r = _responder()
        pb = r.from_resource_anomaly("memory", 250.0, 100.0)
        assert pb.severity == IncidentSeverity.CRITICAL

    def test_with_workers_includes_throttle(self):
        r = _responder()
        pb = r.from_resource_anomaly("cpu", 90.0, 50.0, ["w-1"])
        actions = [s.action for s in pb.steps]
        assert "Throttle affected workers" in actions

    def test_without_workers_no_throttle(self):
        r = _responder()
        pb = r.from_resource_anomaly("cpu", 90.0, 50.0)
        actions = [s.action for s in pb.steps]
        assert "Throttle affected workers" not in actions

    def test_context_has_ratio(self):
        r = _responder()
        pb = r.from_resource_anomaly("disk", 500.0, 100.0)
        assert pb.context["ratio"] == pytest.approx(5.0)
        assert pb.context["value"] == 500.0
        assert pb.context["threshold"] == 100.0

    def test_zero_threshold_gives_infinite_ratio(self):
        r = _responder()
        pb = r.from_resource_anomaly("net", 10.0, 0.0)
        assert pb.severity == IncidentSeverity.CRITICAL


# ── Management ───────────────────────────────────────────────────


class TestManagementExtended:
    def test_get_playbook_returns_none_for_unknown(self):
        r = _responder()
        assert r.get_playbook("INC-NONEXISTENT") is None

    def test_list_active_empty(self):
        r = _responder()
        assert r.list_active() == []

    def test_list_all_empty(self):
        r = _responder()
        assert r.list_all() == []

    def test_summary_empty(self):
        r = _responder()
        s = r.summary()
        assert s["total_incidents"] == 0
        assert s["active"] == 0
        assert s["resolved"] == 0
        assert s["avg_progress"] == 1.0

    def test_summary_all_resolved(self):
        r = _responder()
        pb1 = r.from_escape_event("w-1")
        pb2 = r.from_resource_anomaly("cpu", 90, 50)
        pb1.resolve()
        pb2.resolve()
        s = r.summary()
        assert s["active"] == 0
        assert s["resolved"] == 2
        assert s["avg_progress"] == 1.0

    def test_summary_by_severity(self):
        r = _responder()
        r.from_escape_event("w-1")  # critical
        r.from_resource_anomaly("cpu", 60, 50)  # medium
        s = r.summary()
        assert s["by_severity"]["critical"] == 1
        assert s["by_severity"]["medium"] == 1

    def test_summary_by_category(self):
        r = _responder()
        r.from_escape_event("w-1")  # escape
        r.from_drift_alert(_drift_alert())  # drift
        r.from_compliance_finding(_finding())  # compliance
        s = r.summary()
        assert s["by_category"]["escape"] == 1
        assert s["by_category"]["drift"] == 1
        assert s["by_category"]["compliance"] == 1

    def test_summary_avg_progress(self):
        r = _responder()
        pb = r.from_escape_event("w-1")  # 10 steps
        pb.steps[0].complete()
        pb.steps[1].complete()
        s = r.summary()
        assert s["avg_progress"] == pytest.approx(0.2)

    def test_incident_ids_are_unique(self):
        r = _responder()
        pb1 = r.from_escape_event("w-1")
        pb2 = r.from_escape_event("w-2")
        assert pb1.incident_id != pb2.incident_id

    def test_export_all_json_structure(self):
        r = _responder()
        r.from_escape_event("w-1")
        r.from_drift_alert(_drift_alert())
        data = json.loads(r.export_all())
        assert "summary" in data
        assert "playbooks" in data
        assert len(data["playbooks"]) == 2
        assert data["summary"]["total_incidents"] == 2

    def test_export_all_respects_indent(self):
        r = _responder()
        r.from_escape_event("w-1")
        compact = r.export_all(indent=None)
        pretty = r.export_all(indent=4)
        assert len(pretty) > len(compact)


# ── Enum values ──────────────────────────────────────────────────


class TestEnumValues:
    def test_incident_severity_values(self):
        assert IncidentSeverity.LOW.value == "low"
        assert IncidentSeverity.MEDIUM.value == "medium"
        assert IncidentSeverity.HIGH.value == "high"
        assert IncidentSeverity.CRITICAL.value == "critical"

    def test_incident_category_values(self):
        assert IncidentCategory.DRIFT.value == "drift"
        assert IncidentCategory.COMPLIANCE.value == "compliance"
        assert IncidentCategory.QUARANTINE.value == "quarantine"
        assert IncidentCategory.ANOMALY.value == "anomaly"
        assert IncidentCategory.ESCAPE.value == "escape"
        assert IncidentCategory.RESOURCE.value == "resource"

    def test_step_priority_values(self):
        assert StepPriority.IMMEDIATE.value == "immediate"
        assert StepPriority.FOLLOWUP.value == "followup"

    def test_step_status_values(self):
        assert StepStatus.PENDING.value == "pending"
        assert StepStatus.IN_PROGRESS.value == "in_progress"
        assert StepStatus.COMPLETED.value == "completed"
        assert StepStatus.SKIPPED.value == "skipped"
        assert StepStatus.FAILED.value == "failed"


# ── Step ordering within playbooks ───────────────────────────────


class TestStepOrdering:
    def test_drift_steps_ordered_by_priority(self):
        r = _responder()
        pb = r.from_drift_alert(
            _drift_alert(severity=DriftSeverity.CRITICAL),
            worker_ids=["w-1"],
        )
        priorities = [s.priority for s in pb.steps]
        # IMMEDIATE steps should come first
        immediate_indices = [i for i, p in enumerate(priorities)
                            if p == StepPriority.IMMEDIATE]
        high_indices = [i for i, p in enumerate(priorities)
                        if p == StepPriority.HIGH]
        followup_indices = [i for i, p in enumerate(priorities)
                            if p == StepPriority.FOLLOWUP]
        if immediate_indices and high_indices:
            assert max(immediate_indices) < min(high_indices)
        if high_indices and followup_indices:
            assert max(high_indices) < min(followup_indices)

    def test_escape_first_step_is_quarantine(self):
        r = _responder()
        pb = r.from_escape_event("w-1")
        assert "Quarantine" in pb.steps[0].action
        assert "EMERGENCY" in pb.steps[0].action

    def test_step_ids_sequential(self):
        r = _responder()
        pb = r.from_escape_event("w-1")
        for i, step in enumerate(pb.steps, 1):
            assert step.id == i
