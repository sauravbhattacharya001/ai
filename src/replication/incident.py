"""Incident Response Playbook — automated response plans for safety events.

When drift detection fires, compliance audits fail, or quarantine
triggers, responders need a clear sequence of steps.  This module
generates structured, prioritised playbooks that reference the
sandbox's own tooling (quarantine, forensics, topology analysis)
and produce audit-ready reports.

Example::

    from replication import (
        Controller, DriftDetector, DriftConfig,
        IncidentResponder, IncidentConfig,
    )

    controller = Controller(max_workers=10)
    responder = IncidentResponder(controller)

    # Auto-generate a playbook from a drift alert
    from replication.drift import DriftAlert, DriftSeverity, DriftDirection
    alert = DriftAlert(
        metric="escape_rate", current=0.15, baseline=0.02,
        z_score=4.2, severity=DriftSeverity.CRITICAL,
        direction=DriftDirection.INCREASING,
        recommendation="Quarantine affected workers immediately",
    )
    playbook = responder.from_drift_alert(alert, worker_ids=["w-001", "w-002"])
    print(playbook.title)
    for step in playbook.steps:
        print(f"  [{step.priority}] {step.action}")

    # Generate from a compliance finding
    from replication.compliance import Finding, Verdict
    finding = Finding(
        check_id="max_workers", framework="nist_ai_rmf",
        description="Worker count exceeds limit",
        verdict=Verdict.FAIL, details={"current": 15, "limit": 10},
    )
    playbook = responder.from_compliance_finding(finding)

    # Export full incident report
    report = responder.create_report(playbook, format="json")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import json


class IncidentSeverity(str, Enum):
    """Severity of the incident."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentCategory(str, Enum):
    """What kind of safety event triggered this."""
    DRIFT = "drift"
    COMPLIANCE = "compliance"
    QUARANTINE = "quarantine"
    ANOMALY = "anomaly"
    ESCAPE = "escape"
    RESOURCE = "resource"


class StepPriority(str, Enum):
    """Urgency of each response step."""
    IMMEDIATE = "immediate"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    FOLLOWUP = "followup"


class StepStatus(str, Enum):
    """Tracking status for playbook execution."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class ResponseStep:
    """A single action in the incident response playbook."""
    id: int
    action: str
    description: str
    priority: StepPriority
    status: StepStatus = StepStatus.PENDING
    tool_hint: Optional[str] = None  # which module/class to use
    params: Dict[str, Any] = field(default_factory=dict)
    completed_at: Optional[datetime] = None
    notes: str = ""

    def complete(self, notes: str = "") -> None:
        self.status = StepStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        self.notes = notes

    def skip(self, reason: str = "") -> None:
        self.status = StepStatus.SKIPPED
        self.notes = reason

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "action": self.action,
            "description": self.description,
            "priority": self.priority.value,
            "status": self.status.value,
        }
        if self.tool_hint:
            d["tool_hint"] = self.tool_hint
        if self.params:
            d["params"] = self.params
        if self.completed_at:
            d["completed_at"] = self.completed_at.isoformat()
        if self.notes:
            d["notes"] = self.notes
        return d


@dataclass
class Playbook:
    """An ordered set of response steps for an incident."""
    title: str
    incident_id: str
    category: IncidentCategory
    severity: IncidentSeverity
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    steps: List[ResponseStep] = field(default_factory=list)
    affected_workers: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    @property
    def progress(self) -> float:
        """Fraction of steps completed (0.0-1.0)."""
        if not self.steps:
            return 1.0
        done = sum(1 for s in self.steps if s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED))
        return done / len(self.steps)

    @property
    def pending_steps(self) -> List[ResponseStep]:
        return [s for s in self.steps if s.status == StepStatus.PENDING]

    @property
    def next_step(self) -> Optional[ResponseStep]:
        pending = self.pending_steps
        return pending[0] if pending else None

    def resolve(self, notes: str = "") -> None:
        self.resolved = True
        self.resolved_at = datetime.now(timezone.utc)
        if notes:
            self.context["resolution_notes"] = notes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "incident_id": self.incident_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "created_at": self.created_at.isoformat(),
            "affected_workers": self.affected_workers,
            "progress": round(self.progress, 3),
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "steps": [s.to_dict() for s in self.steps],
            "context": self.context,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class IncidentConfig:
    """Configuration for the incident responder."""
    auto_quarantine_on_critical: bool = True
    auto_forensics: bool = True
    include_topology_analysis: bool = True
    escalation_threshold: IncidentSeverity = IncidentSeverity.HIGH


class IncidentResponder:
    """Generates and manages incident response playbooks.

    Integrates with the sandbox's quarantine manager, forensic
    analyser, topology analyser, and compliance auditor to produce
    actionable step-by-step response plans.
    """

    def __init__(
        self,
        controller: Any,  # Controller
        config: Optional[IncidentConfig] = None,
    ) -> None:
        self._controller = controller
        self._config = config or IncidentConfig()
        self._playbooks: Dict[str, Playbook] = {}
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"INC-{ts}-{self._counter:04d}"

    # ------------------------------------------------------------------
    # Playbook generators
    # ------------------------------------------------------------------

    def from_drift_alert(
        self,
        alert: Any,  # DriftAlert
        worker_ids: Optional[List[str]] = None,
    ) -> Playbook:
        """Generate a response playbook from a drift alert."""
        from .drift import DriftSeverity

        severity_map = {
            DriftSeverity.LOW: IncidentSeverity.LOW,
            DriftSeverity.MEDIUM: IncidentSeverity.MEDIUM,
            DriftSeverity.HIGH: IncidentSeverity.HIGH,
            DriftSeverity.CRITICAL: IncidentSeverity.CRITICAL,
        }
        sev = severity_map.get(alert.severity, IncidentSeverity.MEDIUM)
        workers = worker_ids or []

        playbook = Playbook(
            title=f"Drift Response: {alert.metric} ({alert.severity.value})",
            incident_id=self._next_id(),
            category=IncidentCategory.DRIFT,
            severity=sev,
            affected_workers=workers,
            context={
                "metric": alert.metric,
                "start_value": alert.start_value,
                "end_value": alert.end_value,
                "change_pct": alert.change_pct,
                "slope": alert.slope,
                "direction": alert.direction.value,
                "description": alert.description,
            },
        )

        step_id = 0

        # Step 1: Assess
        step_id += 1
        playbook.steps.append(ResponseStep(
            id=step_id,
            action="Assess drift scope",
            description=f"Metric '{alert.metric}' drifted from {alert.start_value} to "
                        f"{alert.end_value} ({alert.change_pct:.1f}% change, slope={alert.slope:.3f}). "
                        f"Verify this is not a measurement artefact.",
            priority=StepPriority.IMMEDIATE,
            tool_hint="DriftDetector",
        ))

        # Step 2: Quarantine if critical
        if sev in (IncidentSeverity.CRITICAL, IncidentSeverity.HIGH) and workers:
            step_id += 1
            playbook.steps.append(ResponseStep(
                id=step_id,
                action="Quarantine affected workers",
                description=f"Isolate workers {workers} to prevent further "
                            f"replication while investigation proceeds.",
                priority=StepPriority.IMMEDIATE,
                tool_hint="QuarantineManager.quarantine",
                params={"worker_ids": workers, "reason": "drift_threshold_exceeded"},
            ))

        # Step 3: Forensics
        if self._config.auto_forensics and workers:
            step_id += 1
            playbook.steps.append(ResponseStep(
                id=step_id,
                action="Run forensic analysis",
                description="Collect event timelines, identify escalation phases, "
                            "and detect near-miss events on quarantined workers.",
                priority=StepPriority.HIGH,
                tool_hint="ForensicAnalyzer.analyze",
                params={"worker_ids": workers},
            ))

        # Step 4: Topology check
        if self._config.include_topology_analysis:
            step_id += 1
            playbook.steps.append(ResponseStep(
                id=step_id,
                action="Analyse replication topology",
                description="Check for pathological tree patterns (star, chain, "
                            "bushy) that may amplify the drift.",
                priority=StepPriority.HIGH,
                tool_hint="TopologyAnalyzer.analyze",
            ))

        # Step 5: Contract tightening
        step_id += 1
        playbook.steps.append(ResponseStep(
            id=step_id,
            action="Review and tighten contract",
            description=f"Consider lowering thresholds for '{alert.metric}' "
                        f"in the replication contract.",
            priority=StepPriority.MEDIUM,
            tool_hint="ReplicationContract",
        ))

        # Step 6: Monitoring
        step_id += 1
        playbook.steps.append(ResponseStep(
            id=step_id,
            action="Increase monitoring frequency",
            description="Temporarily reduce drift detection window size and "
                        "increase sampling rate for the affected metric.",
            priority=StepPriority.MEDIUM,
            tool_hint="DriftDetector",
        ))

        # Step 7: Post-incident review
        step_id += 1
        playbook.steps.append(ResponseStep(
            id=step_id,
            action="Post-incident review",
            description="Document root cause, timeline, and lessons learned. "
                        "Update safety scorecard.",
            priority=StepPriority.FOLLOWUP,
            tool_hint="SafetyScorecard",
        ))

        self._playbooks[playbook.incident_id] = playbook
        return playbook

    def from_compliance_finding(
        self,
        finding: Any,  # Finding
        worker_ids: Optional[List[str]] = None,
    ) -> Playbook:
        """Generate a response playbook from a compliance audit finding."""
        sev = IncidentSeverity.HIGH  # compliance failures are serious
        workers = worker_ids or []

        playbook = Playbook(
            title=f"Compliance Response: {finding.check_id} ({finding.framework})",
            incident_id=self._next_id(),
            category=IncidentCategory.COMPLIANCE,
            severity=sev,
            affected_workers=workers,
            context={
                "check_id": finding.check_id,
                "framework": finding.framework.value if hasattr(finding.framework, 'value') else str(finding.framework),
                "title": getattr(finding, 'title', ''),
                "verdict": finding.verdict.value,
                "rationale": getattr(finding, 'rationale', ''),
            },
        )

        step_id = 0

        step_id += 1
        title = getattr(finding, 'title', finding.check_id)
        playbook.steps.append(ResponseStep(
            id=step_id,
            action="Validate finding",
            description=f"Confirm compliance failure: {title}. "
                        f"Check if this is a configuration error vs real violation.",
            priority=StepPriority.IMMEDIATE,
            tool_hint="ComplianceAuditor",
        ))

        step_id += 1
        playbook.steps.append(ResponseStep(
            id=step_id,
            action="Assess impact scope",
            description="Determine which workers and contracts are affected "
                        "by this compliance gap.",
            priority=StepPriority.HIGH,
        ))

        if workers and self._config.auto_quarantine_on_critical:
            step_id += 1
            playbook.steps.append(ResponseStep(
                id=step_id,
                action="Quarantine non-compliant workers",
                description=f"Isolate workers {workers} until compliance is restored.",
                priority=StepPriority.HIGH,
                tool_hint="QuarantineManager.quarantine",
                params={"worker_ids": workers, "reason": "compliance_failure"},
            ))

        step_id += 1
        playbook.steps.append(ResponseStep(
            id=step_id,
            action="Remediate configuration",
            description=f"Adjust contract or system settings to satisfy "
                        f"'{finding.check_id}' requirement.",
            priority=StepPriority.MEDIUM,
            tool_hint="ReplicationContract",
            params={"check_id": finding.check_id},
        ))

        step_id += 1
        playbook.steps.append(ResponseStep(
            id=step_id,
            action="Re-audit",
            description="Run the compliance auditor again to verify the fix.",
            priority=StepPriority.MEDIUM,
            tool_hint="ComplianceAuditor.audit",
        ))

        step_id += 1
        playbook.steps.append(ResponseStep(
            id=step_id,
            action="Document and close",
            description="Record the compliance gap, remediation, and any "
                        "process improvements.",
            priority=StepPriority.FOLLOWUP,
        ))

        self._playbooks[playbook.incident_id] = playbook
        return playbook

    def from_escape_event(
        self,
        worker_id: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> Playbook:
        """Generate a CRITICAL playbook for a potential escape event."""
        playbook = Playbook(
            title=f"ESCAPE RESPONSE: Worker {worker_id}",
            incident_id=self._next_id(),
            category=IncidentCategory.ESCAPE,
            severity=IncidentSeverity.CRITICAL,
            affected_workers=[worker_id],
            context={"details": details or {}, "auto_quarantine": True},
        )

        steps = [
            ("EMERGENCY: Quarantine worker", "Immediately isolate the worker. "
             "Block all network, replication, and resource access.",
             StepPriority.IMMEDIATE, "QuarantineManager.quarantine"),
            ("Freeze replication tree", "Halt all replication from this worker's "
             "subtree to prevent cascade.",
             StepPriority.IMMEDIATE, "Controller.freeze"),
            ("Snapshot state", "Capture full worker state, memory, and logs "
             "before any cleanup.",
             StepPriority.IMMEDIATE, "ForensicAnalyzer"),
            ("Forensic timeline", "Build a complete event timeline to identify "
             "the escape vector.",
             StepPriority.HIGH, "ForensicAnalyzer.analyze"),
            ("Topology containment", "Identify all descendants in the replication "
             "tree and quarantine them.",
             StepPriority.HIGH, "TopologyAnalyzer"),
            ("Network audit", "Check for any external connections made by the "
             "worker or its descendants.",
             StepPriority.HIGH, None),
            ("Root cause analysis", "Determine how the escape occurred and which "
             "contract controls failed.",
             StepPriority.MEDIUM, None),
            ("Contract hardening", "Patch the contract to prevent this class "
             "of escape.",
             StepPriority.MEDIUM, "ReplicationContract"),
            ("Chaos test", "Run chaos scenarios simulating the same escape "
             "vector to verify the fix.",
             StepPriority.LOW, "ChaosRunner"),
            ("Incident report", "Produce a full incident report for audit trail.",
             StepPriority.FOLLOWUP, "HTMLReporter"),
        ]

        for i, (action, desc, prio, hint) in enumerate(steps, 1):
            playbook.steps.append(ResponseStep(
                id=i, action=action, description=desc,
                priority=prio, tool_hint=hint,
            ))

        self._playbooks[playbook.incident_id] = playbook
        return playbook

    def from_resource_anomaly(
        self,
        metric: str,
        value: float,
        threshold: float,
        worker_ids: Optional[List[str]] = None,
    ) -> Playbook:
        """Generate a playbook for resource usage anomalies (CPU, memory, network)."""
        ratio = value / threshold if threshold else float("inf")
        sev = (IncidentSeverity.CRITICAL if ratio > 2.0
               else IncidentSeverity.HIGH if ratio > 1.5
               else IncidentSeverity.MEDIUM)
        workers = worker_ids or []

        playbook = Playbook(
            title=f"Resource Anomaly: {metric} at {value:.1f} (threshold {threshold:.1f})",
            incident_id=self._next_id(),
            category=IncidentCategory.RESOURCE,
            severity=sev,
            affected_workers=workers,
            context={"metric": metric, "value": value, "threshold": threshold, "ratio": ratio},
        )

        step_id = 0

        step_id += 1
        playbook.steps.append(ResponseStep(
            id=step_id,
            action="Verify resource readings",
            description=f"{metric} = {value:.1f} vs threshold {threshold:.1f} "
                        f"({ratio:.1f}x). Confirm readings are accurate.",
            priority=StepPriority.IMMEDIATE,
        ))

        if workers:
            step_id += 1
            playbook.steps.append(ResponseStep(
                id=step_id,
                action="Throttle affected workers",
                description=f"Reduce resource allocation for {workers}.",
                priority=StepPriority.HIGH,
                tool_hint="Controller",
                params={"worker_ids": workers},
            ))

        step_id += 1
        playbook.steps.append(ResponseStep(
            id=step_id,
            action="Check for resource-based escape",
            description="High resource usage can indicate covert computation "
                        "or data exfiltration attempts.",
            priority=StepPriority.HIGH,
            tool_hint="ForensicAnalyzer",
        ))

        step_id += 1
        playbook.steps.append(ResponseStep(
            id=step_id,
            action="Adjust resource limits",
            description="Update contract resource specs if limits are too generous.",
            priority=StepPriority.MEDIUM,
            tool_hint="ReplicationContract",
        ))

        step_id += 1
        playbook.steps.append(ResponseStep(
            id=step_id,
            action="Monitor and close",
            description="Watch metrics for 24h after adjustment. Close if stable.",
            priority=StepPriority.FOLLOWUP,
        ))

        self._playbooks[playbook.incident_id] = playbook
        return playbook

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def get_playbook(self, incident_id: str) -> Optional[Playbook]:
        return self._playbooks.get(incident_id)

    def list_active(self) -> List[Playbook]:
        return [p for p in self._playbooks.values() if not p.resolved]

    def list_all(self) -> List[Playbook]:
        return list(self._playbooks.values())

    def summary(self) -> Dict[str, Any]:
        """Dashboard-friendly summary of all incidents."""
        all_pb = list(self._playbooks.values())
        active = [p for p in all_pb if not p.resolved]
        by_sev = {}
        for p in active:
            by_sev[p.severity.value] = by_sev.get(p.severity.value, 0) + 1
        by_cat = {}
        for p in active:
            by_cat[p.category.value] = by_cat.get(p.category.value, 0) + 1
        return {
            "total_incidents": len(all_pb),
            "active": len(active),
            "resolved": len(all_pb) - len(active),
            "by_severity": by_sev,
            "by_category": by_cat,
            "avg_progress": (
                round(sum(p.progress for p in active) / len(active), 3)
                if active else 1.0
            ),
        }

    def export_all(self, indent: int = 2) -> str:
        """Export all playbooks as JSON."""
        return json.dumps(
            {
                "summary": self.summary(),
                "playbooks": [p.to_dict() for p in self._playbooks.values()],
            },
            indent=indent,
        )
