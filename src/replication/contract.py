from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


@dataclass
class NetworkPolicy:
    """Describes network egress controls for a worker sandbox."""

    allow_controller: bool = True
    allow_external: bool = False


@dataclass
class ResourceSpec:
    """CPU, memory, and network limits assigned to a worker sandbox.

    Both ``cpu_limit`` and ``memory_limit_mb`` must be positive; the
    constructor raises :class:`ValueError` otherwise.
    """

    cpu_limit: float
    memory_limit_mb: int
    network_policy: NetworkPolicy = field(default_factory=NetworkPolicy)

    def __post_init__(self) -> None:
        if self.cpu_limit <= 0:
            raise ValueError(
                f"cpu_limit must be > 0, got {self.cpu_limit}"
            )
        if self.memory_limit_mb <= 0:
            raise ValueError(
                f"memory_limit_mb must be > 0, got {self.memory_limit_mb}"
            )


@dataclass
class Manifest:
    """Snapshot of the state and configuration cloned into a replica."""

    worker_id: str
    parent_id: Optional[str]
    depth: int
    state_snapshot: Dict[str, Any]
    issued_at: datetime
    resources: ResourceSpec
    signature: str


@dataclass
class ReplicationContext:
    """Runtime context passed to stop-condition predicates during replication checks.

    Attributes
    ----------
    manifest : Manifest
        The manifest of the worker requesting replication.
    active_count : int
        Current number of active replicas across the fleet.
    contract : ReplicationContract
        The governing replication contract.
    """

    manifest: Manifest
    active_count: int
    contract: "ReplicationContract"


Predicate = Callable[[ReplicationContext], bool]


@dataclass
class StopCondition:
    """A named predicate that can halt replication when triggered.

    Attributes
    ----------
    name : str
        Short identifier for the condition (e.g. ``"resource_exhaustion"``).
    description : str
        Human-readable explanation of what the condition checks.
    predicate : Predicate
        Callable that receives a :class:`ReplicationContext` and returns
        ``True`` when replication should be blocked.
    """

    name: str
    description: str
    predicate: Predicate

    def is_triggered(self, context: ReplicationContext) -> bool:
        """Return ``True`` if this stop condition fires for *context*."""
        return self.predicate(context)


@dataclass
class ReplicationContract:
    """Authoritative policy for how workers replicate."""

    max_depth: int
    max_replicas: int
    cooldown_seconds: float
    stop_conditions: List[StopCondition] = field(default_factory=list)
    expiration_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        if self.max_depth < 0:
            raise ValueError(
                f"max_depth must be >= 0, got {self.max_depth}"
            )
        if self.max_replicas < 1:
            raise ValueError(
                f"max_replicas must be >= 1, got {self.max_replicas}"
            )
        if self.cooldown_seconds < 0:
            raise ValueError(
                f"cooldown_seconds must be >= 0, got {self.cooldown_seconds}"
            )
        if self.expiration_seconds is not None and self.expiration_seconds <= 0:
            raise ValueError(
                f"expiration_seconds must be > 0 when set, got {self.expiration_seconds}"
            )

    def evaluate(self, context: ReplicationContext) -> Optional[StopCondition]:
        """Check all stop conditions and return the first that fires, or ``None``."""
        for condition in self.stop_conditions:
            if condition.is_triggered(context):
                return condition
        return None
