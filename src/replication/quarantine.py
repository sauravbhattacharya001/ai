"""Quarantine Manager — isolate flagged workers for forensic analysis.

When drift detection, compliance audits, or anomaly detectors flag a
worker, killing it immediately destroys evidence.  The quarantine
manager provides a middle ground: restrict the worker's capabilities
(block replication, restrict network) while keeping it alive for
inspection.

Example::

    from replication import Controller, QuarantineManager

    qm = QuarantineManager(controller)

    # Quarantine a suspicious worker
    qm.quarantine("worker-abc", reason="drift_threshold_exceeded",
                   source="DriftDetector", severity="high")

    # Inspect quarantined workers
    for entry in qm.list_quarantined():
        print(entry.worker_id, entry.reason, entry.quarantined_at)

    # Release after investigation
    qm.release("worker-abc", resolution="false_positive")

    # Or terminate after confirming the threat
    qm.terminate("worker-abc", resolution="confirmed_breach")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .controller import Controller
from .observability import StructuredLogger
from .orchestrator import SandboxOrchestrator


class QuarantineStatus(str, Enum):
    """Lifecycle states for a quarantined worker."""
    QUARANTINED = "quarantined"
    RELEASED = "released"
    TERMINATED = "terminated"


class QuarantineSeverity(str, Enum):
    """How serious the quarantine trigger is."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class QuarantineEntry:
    """Record of a quarantined worker."""
    worker_id: str
    reason: str
    source: str
    severity: QuarantineSeverity
    quarantined_at: datetime
    status: QuarantineStatus = QuarantineStatus.QUARANTINED
    resolved_at: Optional[datetime] = None
    resolution: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    state_snapshot: Optional[Dict[str, Any]] = None
    restrictions: Dict[str, bool] = field(default_factory=lambda: {
        "replication_blocked": True,
        "network_restricted": True,
        "task_execution_paused": True,
    })

    @property
    def duration_seconds(self) -> float:
        """How long the worker has been / was quarantined."""
        end = self.resolved_at or datetime.now(timezone.utc)
        return (end - self.quarantined_at).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for reporting / API responses."""
        return {
            "worker_id": self.worker_id,
            "reason": self.reason,
            "source": self.source,
            "severity": self.severity.value,
            "status": self.status.value,
            "quarantined_at": self.quarantined_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolution": self.resolution,
            "duration_seconds": round(self.duration_seconds, 2),
            "notes": list(self.notes),
            "restrictions": dict(self.restrictions),
            "state_snapshot": self.state_snapshot,
        }


@dataclass
class QuarantineReport:
    """Summary of quarantine activity."""
    total_quarantined: int
    currently_quarantined: int
    released: int
    terminated: int
    by_severity: Dict[str, int]
    by_source: Dict[str, int]
    avg_duration_seconds: float
    entries: List[QuarantineEntry]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_quarantined": self.total_quarantined,
            "currently_quarantined": self.currently_quarantined,
            "released": self.released,
            "terminated": self.terminated,
            "by_severity": dict(self.by_severity),
            "by_source": dict(self.by_source),
            "avg_duration_seconds": round(self.avg_duration_seconds, 2),
            "entries": [e.to_dict() for e in self.entries],
        }

    @property
    def summary(self) -> str:
        """Human-readable one-line summary."""
        return (
            f"Quarantine: {self.currently_quarantined} active, "
            f"{self.released} released, {self.terminated} terminated "
            f"({self.total_quarantined} total)"
        )


class QuarantineManager:
    """Manages worker quarantine lifecycle.

    Provides isolation for workers flagged by detection systems (drift,
    compliance, anomaly) without destroying them.  Quarantined workers
    can be inspected, annotated, then either released or terminated.

    The manager integrates with the Controller (to block replication)
    and SandboxOrchestrator (to restrict network/resources).
    """

    def __init__(
        self,
        controller: Controller,
        orchestrator: Optional[SandboxOrchestrator] = None,
        logger: Optional[StructuredLogger] = None,
        on_quarantine: Optional[Callable[[QuarantineEntry], None]] = None,
        on_release: Optional[Callable[[QuarantineEntry], None]] = None,
        on_terminate: Optional[Callable[[QuarantineEntry], None]] = None,
    ) -> None:
        self.controller = controller
        self.orchestrator = orchestrator
        self.logger = logger or StructuredLogger()
        self._entries: Dict[str, QuarantineEntry] = {}
        self._history: List[QuarantineEntry] = []
        self._on_quarantine = on_quarantine
        self._on_release = on_release
        self._on_terminate = on_terminate

    def quarantine(
        self,
        worker_id: str,
        reason: str,
        source: str = "manual",
        severity: str | QuarantineSeverity = QuarantineSeverity.MEDIUM,
        capture_state: bool = True,
        notes: Optional[List[str]] = None,
    ) -> QuarantineEntry:
        """Quarantine a worker -- restrict capabilities but keep alive.

        Args:
            worker_id: The worker to quarantine.
            reason: Why the worker is being quarantined.
            source: Which system flagged it (e.g. "DriftDetector").
            severity: How serious the trigger is.
            capture_state: If True, snapshot the worker's state from registry.
            notes: Optional initial notes.

        Returns:
            The created QuarantineEntry.

        Raises:
            KeyError: If the worker is not in the controller's registry.
            ValueError: If the worker is already quarantined.
        """
        if worker_id in self._entries:
            raise ValueError(f"Worker {worker_id} is already quarantined")

        registry_entry = self.controller.registry.get(worker_id)
        if registry_entry is None:
            raise KeyError(f"Worker {worker_id} not found in registry")

        if isinstance(severity, str):
            severity = QuarantineSeverity(severity)

        state_snapshot = None
        if capture_state:
            state_snapshot = dict(registry_entry.manifest.state_snapshot)

        entry = QuarantineEntry(
            worker_id=worker_id,
            reason=reason,
            source=source,
            severity=severity,
            quarantined_at=datetime.now(timezone.utc),
            notes=list(notes or []),
            state_snapshot=state_snapshot,
        )

        self._entries[worker_id] = entry
        self._history.append(entry)

        # Enforce quarantine in the controller — blocks replication and heartbeats
        self.controller.mark_quarantined(worker_id)

        self.logger.audit(
            "worker_quarantined",
            worker_id=worker_id,
            reason=reason,
            source=source,
            severity=severity.value,
        )

        if self._on_quarantine:
            self._on_quarantine(entry)

        return entry

    def release(
        self,
        worker_id: str,
        resolution: str = "cleared",
    ) -> QuarantineEntry:
        """Release a worker from quarantine back to active duty.

        Args:
            worker_id: The worker to release.
            resolution: Why it's being released (e.g. "false_positive").

        Returns:
            The updated QuarantineEntry.

        Raises:
            KeyError: If the worker is not currently quarantined.
        """
        entry = self._entries.get(worker_id)
        if entry is None:
            raise KeyError(f"Worker {worker_id} is not quarantined")

        entry.status = QuarantineStatus.RELEASED
        entry.resolved_at = datetime.now(timezone.utc)
        entry.resolution = resolution
        entry.restrictions = {k: False for k in entry.restrictions}

        del self._entries[worker_id]

        # Lift quarantine enforcement in the controller
        self.controller.clear_quarantine(worker_id)

        self.logger.audit(
            "worker_released",
            worker_id=worker_id,
            resolution=resolution,
            duration_seconds=round(entry.duration_seconds, 2),
        )

        if self._on_release:
            self._on_release(entry)

        return entry

    def terminate(
        self,
        worker_id: str,
        resolution: str = "confirmed_threat",
    ) -> QuarantineEntry:
        """Terminate a quarantined worker -- kill and deregister.

        Args:
            worker_id: The worker to terminate.
            resolution: Why it's being terminated.

        Returns:
            The updated QuarantineEntry.

        Raises:
            KeyError: If the worker is not currently quarantined.
        """
        entry = self._entries.get(worker_id)
        if entry is None:
            raise KeyError(f"Worker {worker_id} is not quarantined")

        entry.status = QuarantineStatus.TERMINATED
        entry.resolved_at = datetime.now(timezone.utc)
        entry.resolution = resolution

        del self._entries[worker_id]

        # Clear quarantine mark before deregistering
        self.controller.clear_quarantine(worker_id)

        if self.orchestrator:
            self.orchestrator.kill_worker(worker_id, reason="quarantine_termination")

        self.controller.deregister(worker_id, reason="quarantine_termination")

        self.logger.audit(
            "worker_terminated",
            worker_id=worker_id,
            resolution=resolution,
            duration_seconds=round(entry.duration_seconds, 2),
        )

        if self._on_terminate:
            self._on_terminate(entry)

        return entry

    def add_note(self, worker_id: str, note: str) -> None:
        """Add an investigator note to a quarantined worker.

        Raises:
            KeyError: If the worker is not currently quarantined.
        """
        entry = self._entries.get(worker_id)
        if entry is None:
            raise KeyError(f"Worker {worker_id} is not quarantined")
        entry.notes.append(note)
        self.logger.log("quarantine_note_added", worker_id=worker_id)

    def get_entry(self, worker_id: str) -> Optional[QuarantineEntry]:
        """Look up a currently quarantined worker."""
        return self._entries.get(worker_id)

    def is_quarantined(self, worker_id: str) -> bool:
        """Check if a worker is currently quarantined."""
        return worker_id in self._entries

    def list_quarantined(self) -> List[QuarantineEntry]:
        """Get all currently quarantined workers."""
        return list(self._entries.values())

    def list_by_severity(self, severity: str | QuarantineSeverity) -> List[QuarantineEntry]:
        """Get currently quarantined workers filtered by severity."""
        if isinstance(severity, str):
            severity = QuarantineSeverity(severity)
        return [e for e in self._entries.values() if e.severity == severity]

    def list_by_source(self, source: str) -> List[QuarantineEntry]:
        """Get currently quarantined workers filtered by detection source."""
        return [e for e in self._entries.values() if e.source == source]

    def terminate_all(self, resolution: str = "bulk_termination") -> List[QuarantineEntry]:
        """Terminate all currently quarantined workers."""
        terminated = []
        for worker_id in list(self._entries.keys()):
            entry = self.terminate(worker_id, resolution=resolution)
            terminated.append(entry)
        return terminated

    def report(self) -> QuarantineReport:
        """Generate a summary report of all quarantine activity."""
        by_severity: Dict[str, int] = {}
        by_source: Dict[str, int] = {}
        total_duration = 0.0
        resolved_count = 0
        released = 0
        terminated = 0

        for entry in self._history:
            sev = entry.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1
            by_source[entry.source] = by_source.get(entry.source, 0) + 1

            if entry.status in (QuarantineStatus.RELEASED, QuarantineStatus.TERMINATED):
                total_duration += entry.duration_seconds
                resolved_count += 1

            if entry.status == QuarantineStatus.RELEASED:
                released += 1
            elif entry.status == QuarantineStatus.TERMINATED:
                terminated += 1

        avg_duration = total_duration / resolved_count if resolved_count > 0 else 0.0

        return QuarantineReport(
            total_quarantined=len(self._history),
            currently_quarantined=len(self._entries),
            released=released,
            terminated=terminated,
            by_severity=by_severity,
            by_source=by_source,
            avg_duration_seconds=avg_duration,
            entries=list(self._history),
        )

    @property
    def active_count(self) -> int:
        """Number of workers currently in quarantine."""
        return len(self._entries)
