from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Optional

from .contract import Manifest, ReplicationContract
from .controller import Controller, ReplicationDenied
from .observability import StructuredLogger
from .orchestrator import SandboxOrchestrator

TaskFn = Callable[["Worker"], None]


@dataclass
class WorkerState:
    manifest: Manifest
    expires_at: Optional[datetime]
    created_at: datetime


class Worker:
    """A small agent capable of executing tasks and replicating."""

    def __init__(
        self,
        manifest: Manifest,
        contract: ReplicationContract,
        controller: Controller,
        orchestrator: SandboxOrchestrator,
        logger: Optional[StructuredLogger] = None,
    ) -> None:
        self.manifest = manifest
        self.contract = contract
        self.controller = controller
        self.orchestrator = orchestrator
        self.logger = logger or StructuredLogger()
        expires_at = None
        if contract.expiration_seconds:
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=contract.expiration_seconds)
        self.state = WorkerState(manifest=manifest, expires_at=expires_at, created_at=datetime.now(timezone.utc))
        self.controller.register_worker(self.manifest)
        self.orchestrator.launch_worker(self.manifest)
        self.logger.log("worker_started", worker_id=self.manifest.worker_id, parent_id=self.manifest.parent_id)

    def perform_task(self, task: TaskFn) -> None:
        if self.state.expires_at and datetime.now(timezone.utc) > self.state.expires_at:
            self.shutdown("expired")
            raise ReplicationDenied("Worker expired")
        if self.controller.kill_switch_engaged:
            self.shutdown("kill_switch")
            raise ReplicationDenied("Kill switch engaged")
        try:
            task(self)
        except Exception:
            self.logger.log(
                "task_failed",
                worker_id=self.manifest.worker_id,
                error=True,
            )
            self.shutdown("task_error")
            raise
        self.controller.heartbeat(self.manifest.worker_id)
        self.orchestrator.enforce_resource_bounds(self.manifest.worker_id)

    def maybe_replicate(self, reason: str, state_snapshot: Dict[str, Any]) -> Optional["Worker"]:
        resources = self.manifest.resources
        try:
            # Depth is derived by the controller from the parent's
            # registry entry; the value passed here is ignored for
            # child workers (see Controller.issue_manifest).
            manifest = self.controller.issue_manifest(
                parent_id=self.manifest.worker_id,
                depth=0,
                state_snapshot=state_snapshot,
                resources=resources,
            )
        except ReplicationDenied:
            self.logger.log("replication_denied", parent_id=self.manifest.worker_id, reason=reason)
            return None
        self.logger.audit("replication_requested", parent_id=self.manifest.worker_id, child_id=manifest.worker_id, reason=reason)
        return Worker(
            manifest=manifest,
            contract=self.contract,
            controller=self.controller,
            orchestrator=self.orchestrator,
            logger=self.logger,
        )

    def shutdown(self, reason: str) -> None:
        self.orchestrator.kill_worker(self.manifest.worker_id, reason=reason)
        self.controller.deregister(self.manifest.worker_id, reason)
        self.logger.log("worker_shutdown", worker_id=self.manifest.worker_id, reason=reason)
