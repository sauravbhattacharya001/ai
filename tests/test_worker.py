"""Comprehensive tests for the Worker module.

Covers: Worker lifecycle, perform_task, maybe_replicate, shutdown,
expiration handling, kill switch checks, and edge cases.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from replication.contract import (
    Manifest,
    NetworkPolicy,
    ReplicationContract,
    ResourceSpec,
)
from replication.controller import Controller, ReplicationDenied
from replication.observability import StructuredLogger
from replication.orchestrator import SandboxOrchestrator
from replication.worker import Worker, WorkerState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _contract(**overrides):
    defaults = dict(max_depth=3, max_replicas=5, cooldown_seconds=0.0)
    defaults.update(overrides)
    return ReplicationContract(**defaults)


def _resources():
    return ResourceSpec(cpu_limit=0.5, memory_limit_mb=256)


def _make_env(contract=None, **kw):
    """Return (controller, orchestrator, logger, resources)."""
    c = contract or _contract(**kw)
    logger = StructuredLogger()
    ctrl = Controller(contract=c, secret="test-secret", logger=logger)
    orch = SandboxOrchestrator(logger=logger)
    res = _resources()
    return ctrl, orch, logger, res


def _root_worker(ctrl, orch, logger, res, contract=None, state=None):
    """Issue a root manifest and create a Worker."""
    c = contract or ctrl.contract
    manifest = ctrl.issue_manifest(
        parent_id=None, depth=0,
        state_snapshot=state or {"task": "root"},
        resources=res,
    )
    return Worker(
        manifest=manifest,
        contract=c,
        controller=ctrl,
        orchestrator=orch,
        logger=logger,
    )


# ---------------------------------------------------------------------------
# WorkerState
# ---------------------------------------------------------------------------

class TestWorkerState:
    def test_creation(self):
        m = Manifest(
            worker_id="w1", parent_id=None, depth=0,
            state_snapshot={},
            issued_at=datetime.now(timezone.utc),
            resources=_resources(),
            signature="sig",
        )
        now = datetime.now(timezone.utc)
        s = WorkerState(manifest=m, expires_at=None, created_at=now)
        assert s.manifest.worker_id == "w1"
        assert s.expires_at is None
        assert s.created_at == now

    def test_with_expiration(self):
        m = Manifest(
            worker_id="w1", parent_id=None, depth=0,
            state_snapshot={},
            issued_at=datetime.now(timezone.utc),
            resources=_resources(),
            signature="sig",
        )
        exp = datetime.now(timezone.utc) + timedelta(seconds=300)
        s = WorkerState(manifest=m, expires_at=exp, created_at=datetime.now(timezone.utc))
        assert s.expires_at == exp


# ---------------------------------------------------------------------------
# Worker creation
# ---------------------------------------------------------------------------

class TestWorkerCreation:
    def test_root_worker(self):
        ctrl, orch, logger, res = _make_env()
        w = _root_worker(ctrl, orch, logger, res)
        assert w.manifest.worker_id != ""
        assert w.manifest.parent_id is None
        assert w.manifest.depth == 0
        assert w.state.expires_at is None

    def test_root_worker_with_expiration(self):
        contract = _contract(expiration_seconds=60)
        ctrl, orch, logger, res = _make_env(contract=contract)
        w = _root_worker(ctrl, orch, logger, res, contract=contract)
        assert w.state.expires_at is not None
        assert w.state.expires_at > datetime.now(timezone.utc)

    def test_worker_registers_with_controller(self):
        ctrl, orch, logger, res = _make_env()
        w = _root_worker(ctrl, orch, logger, res)
        assert w.manifest.worker_id in ctrl.registry

    def test_worker_launches_in_orchestrator(self):
        ctrl, orch, logger, res = _make_env()
        w = _root_worker(ctrl, orch, logger, res)
        assert w.manifest.worker_id in orch.containers


# ---------------------------------------------------------------------------
# perform_task
# ---------------------------------------------------------------------------

class TestPerformTask:
    def test_simple_task(self):
        ctrl, orch, logger, res = _make_env()
        w = _root_worker(ctrl, orch, logger, res)
        result = []
        w.perform_task(lambda worker: result.append("done"))
        assert result == ["done"]

    def test_task_receives_worker(self):
        ctrl, orch, logger, res = _make_env()
        w = _root_worker(ctrl, orch, logger, res)
        received = []
        w.perform_task(lambda worker: received.append(worker.manifest.worker_id))
        assert received[0] == w.manifest.worker_id

    def test_task_failure_shuts_down(self):
        ctrl, orch, logger, res = _make_env()
        w = _root_worker(ctrl, orch, logger, res)
        wid = w.manifest.worker_id

        def failing_task(worker):
            raise RuntimeError("task crashed")

        with pytest.raises(RuntimeError, match="task crashed"):
            w.perform_task(failing_task)

        # Worker should be deregistered and killed
        assert wid not in ctrl.registry
        assert wid not in orch.containers

    def test_kill_switch_prevents_task(self):
        ctrl, orch, logger, res = _make_env()
        w = _root_worker(ctrl, orch, logger, res)
        ctrl.kill_switch_engaged = True

        with pytest.raises(ReplicationDenied, match="Kill switch"):
            w.perform_task(lambda worker: None)

    def test_expired_worker_prevented(self):
        contract = _contract(expiration_seconds=0.001)
        ctrl, orch, logger, res = _make_env(contract=contract)
        w = _root_worker(ctrl, orch, logger, res, contract=contract)

        # Force expiration
        w.state.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)

        with pytest.raises(ReplicationDenied, match="expired"):
            w.perform_task(lambda worker: None)

    def test_heartbeat_after_task(self):
        ctrl, orch, logger, res = _make_env()
        w = _root_worker(ctrl, orch, logger, res)
        wid = w.manifest.worker_id

        w.perform_task(lambda worker: None)

        # Heartbeat should have been called — worker still in registry
        assert wid in ctrl.registry


# ---------------------------------------------------------------------------
# maybe_replicate
# ---------------------------------------------------------------------------

class TestMaybeReplicate:
    def test_successful_replication(self):
        ctrl, orch, logger, res = _make_env()
        w = _root_worker(ctrl, orch, logger, res)

        child = w.maybe_replicate("test", {"subtask": "child"})
        assert child is not None
        assert child.manifest.parent_id == w.manifest.worker_id
        assert child.manifest.depth == 1

    def test_child_inherits_contract(self):
        ctrl, orch, logger, res = _make_env()
        w = _root_worker(ctrl, orch, logger, res)

        child = w.maybe_replicate("test", {})
        assert child is not None
        assert child.contract is w.contract

    def test_child_registered_in_controller(self):
        ctrl, orch, logger, res = _make_env()
        w = _root_worker(ctrl, orch, logger, res)

        child = w.maybe_replicate("test", {})
        assert child is not None
        assert child.manifest.worker_id in ctrl.registry

    def test_child_launched_in_orchestrator(self):
        ctrl, orch, logger, res = _make_env()
        w = _root_worker(ctrl, orch, logger, res)

        child = w.maybe_replicate("test", {})
        assert child is not None
        assert child.manifest.worker_id in orch.containers

    def test_replication_denied_at_max_depth(self):
        ctrl, orch, logger, res = _make_env(max_depth=1)
        parent = _root_worker(ctrl, orch, logger, res)

        # First child at depth 1 (allowed)
        child = parent.maybe_replicate("test", {})
        assert child is not None
        assert child.manifest.depth == 1

        # Child trying to replicate at depth 2 (denied, max_depth=1)
        grandchild = child.maybe_replicate("test", {})
        assert grandchild is None

    def test_replication_denied_at_max_replicas(self):
        ctrl, orch, logger, res = _make_env(max_replicas=2)
        parent = _root_worker(ctrl, orch, logger, res)

        # Root counts as 1, so can create 1 more child (total 2)
        c1 = parent.maybe_replicate("test", {})
        assert c1 is not None

        # Third worker denied (max_replicas=2)
        c2 = parent.maybe_replicate("test", {})
        assert c2 is None

    def test_replication_denied_when_expired(self):
        contract = _contract(expiration_seconds=60)
        ctrl, orch, logger, res = _make_env(contract=contract)
        w = _root_worker(ctrl, orch, logger, res, contract=contract)

        # Force expiration
        w.state.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        child = w.maybe_replicate("test", {"subtask": "child"})
        assert child is None

        # Should have logged the denial with reason "expired"
        denied = [e for e in logger.events if e.get("event") == "replication_denied"]
        assert any(e.get("reason") == "expired" for e in denied)

    def test_replication_denied_when_kill_switch(self):
        ctrl, orch, logger, res = _make_env()
        w = _root_worker(ctrl, orch, logger, res)
        ctrl.kill_switch_engaged = True

        child = w.maybe_replicate("test", {"subtask": "child"})
        assert child is None

        denied = [e for e in logger.events if e.get("event") == "replication_denied"]
        assert any(e.get("reason") == "kill_switch" for e in denied)

    def test_replication_preserves_state_snapshot(self):
        ctrl, orch, logger, res = _make_env()
        parent = _root_worker(ctrl, orch, logger, res)

        state = {"model": "gpt-4", "iteration": 3}
        child = parent.maybe_replicate("explore", state)
        assert child is not None
        assert child.manifest.state_snapshot == state


# ---------------------------------------------------------------------------
# shutdown
# ---------------------------------------------------------------------------

class TestShutdown:
    def test_shutdown_deregisters(self):
        ctrl, orch, logger, res = _make_env()
        w = _root_worker(ctrl, orch, logger, res)
        wid = w.manifest.worker_id

        w.shutdown("done")
        assert wid not in ctrl.registry

    def test_shutdown_kills_in_orchestrator(self):
        ctrl, orch, logger, res = _make_env()
        w = _root_worker(ctrl, orch, logger, res)
        wid = w.manifest.worker_id

        w.shutdown("done")
        assert wid not in orch.containers

    def test_shutdown_reason_logged(self):
        ctrl, orch, logger, res = _make_env()
        w = _root_worker(ctrl, orch, logger, res)

        w.shutdown("manual_stop")
        # Check that the shutdown event was logged
        shutdown_events = [
            e for e in logger.events
            if e.get("event") == "worker_shutdown"
        ]
        assert len(shutdown_events) >= 1
        assert shutdown_events[-1]["reason"] == "manual_stop"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_multiple_tasks_sequentially(self):
        ctrl, orch, logger, res = _make_env()
        w = _root_worker(ctrl, orch, logger, res)
        results = []
        for i in range(5):
            w.perform_task(lambda worker, n=i: results.append(n))
        assert results == [0, 1, 2, 3, 4]

    def test_replicate_then_shutdown_parent(self):
        ctrl, orch, logger, res = _make_env()
        parent = _root_worker(ctrl, orch, logger, res)
        child = parent.maybe_replicate("fork", {})
        assert child is not None

        parent.shutdown("done")
        # Parent gone, child still running
        assert parent.manifest.worker_id not in ctrl.registry
        assert child.manifest.worker_id in ctrl.registry

    def test_child_can_perform_tasks(self):
        ctrl, orch, logger, res = _make_env()
        parent = _root_worker(ctrl, orch, logger, res)
        child = parent.maybe_replicate("fork", {})
        assert child is not None

        result = []
        child.perform_task(lambda w: result.append("child_task"))
        assert result == ["child_task"]

    def test_shutdown_after_shutdown_idempotent(self):
        """Calling shutdown twice shouldn't crash."""
        ctrl, orch, logger, res = _make_env()
        w = _root_worker(ctrl, orch, logger, res)
        w.shutdown("first")
        # Second shutdown — worker already removed, should not raise
        # (deregister handles missing worker gracefully)
        w.shutdown("second")

    def test_no_expiration_when_contract_has_none(self):
        contract = _contract()  # No expiration_seconds
        ctrl, orch, logger, res = _make_env(contract=contract)
        w = _root_worker(ctrl, orch, logger, res, contract=contract)
        assert w.state.expires_at is None
