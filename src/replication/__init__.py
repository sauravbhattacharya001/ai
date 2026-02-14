"""AI Replication Sandbox — public API.

Convenience re-exports so consumers can write::

    from replication import Controller, Worker, ReplicationContract, ...

instead of reaching into submodules.
"""

__version__ = "1.0.0"

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
from .signer import ManifestSigner
from .orchestrator import ContainerRecord, SandboxOrchestrator
from .simulator import Simulator, ScenarioConfig, SimulationReport, Strategy, PRESETS
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
    # signer
    "ManifestSigner",
    # orchestrator
    "ContainerRecord",
    "SandboxOrchestrator",
    # simulator
    "Simulator",
    "ScenarioConfig",
    "SimulationReport",
    "Strategy",
    "PRESETS",
    # worker
    "Worker",
    "WorkerState",
]
