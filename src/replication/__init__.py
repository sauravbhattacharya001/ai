"""AI Replication Sandbox — public API.

Convenience re-exports so consumers can write::

    from replication import Controller, Worker, ReplicationContract, ...

instead of reaching into submodules.
"""

from .contract import (
    Manifest,
    NetworkPolicy,
    Predicate,
    ReplicationContext,
    ReplicationContract,
    ResourceSpec,
    StopCondition,
)
from .controller import Controller, ReplicationDenied, RegistryEntry
from .observability import Metric, StructuredLogger
from .orchestrator import ContainerRecord, SandboxOrchestrator
from .worker import Worker, WorkerState

__all__ = [
    # contract
    "Manifest",
    "NetworkPolicy",
    "Predicate",
    "ReplicationContext",
    "ReplicationContract",
    "ResourceSpec",
    "StopCondition",
    # controller
    "Controller",
    "ReplicationDenied",
    "RegistryEntry",
    # observability
    "Metric",
    "StructuredLogger",
    # orchestrator
    "ContainerRecord",
    "SandboxOrchestrator",
    # worker
    "Worker",
    "WorkerState",
]
