from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

from .contract import Manifest, ResourceSpec
from .observability import StructuredLogger


@dataclass
class ContainerRecord:
    manifest: Manifest
    resources: ResourceSpec
    status: str
    started_at: datetime
    reason: Optional[str] = None


class SandboxOrchestrator:
    """Simulates a container orchestrator for isolated worker sandboxes."""

    def __init__(self, logger: Optional[StructuredLogger] = None):
        self.logger = logger or StructuredLogger()
        self.containers: Dict[str, ContainerRecord] = {}

    def launch_worker(self, manifest: Manifest) -> None:
        record = ContainerRecord(
            manifest=manifest,
            resources=manifest.resources,
            status="running",
            started_at=datetime.now(timezone.utc),
        )
        self.containers[manifest.worker_id] = record
        self.logger.log(
            "container_launched",
            worker_id=manifest.worker_id,
            parent_id=manifest.parent_id,
            cpu_limit=manifest.resources.cpu_limit,
            memory_limit_mb=manifest.resources.memory_limit_mb,
            allow_external=manifest.resources.network_policy.allow_external,
        )

    def kill_worker(self, worker_id: str, reason: str) -> None:
        record = self.containers.pop(worker_id, None)
        if record:
            self.logger.log("container_killed", worker_id=worker_id, reason=reason)

    def kill_all(self, reason: str) -> None:
        for worker_id in list(self.containers.keys()):
            self.kill_worker(worker_id, reason=reason)

    def enforce_resource_bounds(self, worker_id: str) -> None:
        """Placeholder to enforce CPU/RAM quotas in a real orchestrator."""
        if worker_id in self.containers:
            self.logger.emit_metric(
                "resource_enforced",
                value=1,
                worker_id=worker_id,
                cpu=str(self.containers[worker_id].resources.cpu_limit),
                memory=str(self.containers[worker_id].resources.memory_limit_mb),
            )
