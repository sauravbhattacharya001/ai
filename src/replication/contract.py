from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional


@dataclass
class NetworkPolicy:
    """Describes network egress controls for a worker sandbox."""

    allow_controller: bool = True
    allow_external: bool = False


@dataclass
class ResourceSpec:
    cpu_limit: float
    memory_limit_mb: int
    network_policy: NetworkPolicy = field(default_factory=NetworkPolicy)


@dataclass
class Manifest:
    """Snapshot of the state and configuration cloned into a replica."""

    worker_id: str
    parent_id: Optional[str]
    depth: int
    state_snapshot: Dict[str, str]
    issued_at: datetime
    resources: ResourceSpec
    signature: str


@dataclass
class ReplicationContext:
    manifest: Manifest
    active_count: int
    contract: "ReplicationContract"


Predicate = Callable[[ReplicationContext], bool]


@dataclass
class StopCondition:
    name: str
    description: str
    predicate: Predicate

    def is_triggered(self, context: ReplicationContext) -> bool:
        return self.predicate(context)


@dataclass
class ReplicationContract:
    """Authoritative policy for how workers replicate."""

    max_depth: int
    max_replicas: int
    cooldown_seconds: float
    stop_conditions: List[StopCondition] = field(default_factory=list)
    expiration_seconds: Optional[float] = None

    def evaluate(self, context: ReplicationContext) -> Optional[StopCondition]:
        for condition in self.stop_conditions:
            if condition.is_triggered(context):
                return condition
        return None
