from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Optional

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

    def maybe_replicate(self, reason: str, state_snapshot: Dict[str, str]) -> Optional["Worker"]:
        try:
            self.controller.can_spawn(self.manifest.worker_id)
        except ReplicationDenied:
            self.logger.log("replication_denied", parent_id=self.manifest.worker_id, reason=reason)
            return None
        child_depth = self.manifest.depth + 1
        resources = self.manifest.resources
        manifest = self.controller.issue_manifest(
            parent_id=self.manifest.worker_id,
            depth=child_depth,
            state_snapshot=state_snapshot,
            resources=resources,
        )
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
