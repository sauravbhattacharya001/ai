"""Tests for SandboxOrchestrator — container lifecycle management.

Covers: launch, kill, kill_all, enforce_resource_bounds, logging
integration, and edge cases.
"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from datetime import datetime, timezone
from replication.contract import Manifest, ResourceSpec, NetworkPolicy
from replication.observability import StructuredLogger
from replication.orchestrator import SandboxOrchestrator, ContainerRecord


def make_manifest(worker_id="w1", parent_id="root", depth=1,
                  cpu=1.0, memory=256):
    return Manifest(
        worker_id=worker_id,
        parent_id=parent_id,
        depth=depth,
        state_snapshot={"key": "value"},
        issued_at=datetime.now(timezone.utc),
        resources=ResourceSpec(
            cpu_limit=cpu,
            memory_limit_mb=memory,
            network_policy=NetworkPolicy(),
        ),
        signature="test-sig",
    )


class TestOrchestratorInit:
    """Initialization and default state."""

    def test_default_logger_created(self):
        orch = SandboxOrchestrator()
        assert isinstance(orch.logger, StructuredLogger)

    def test_custom_logger(self):
        logger = StructuredLogger()
        orch = SandboxOrchestrator(logger=logger)
        assert orch.logger is logger

    def test_no_containers_initially(self):
        orch = SandboxOrchestrator()
        assert len(orch.containers) == 0


class TestLaunchWorker:
    """SandboxOrchestrator.launch_worker()."""

    def test_launch_adds_container(self):
        orch = SandboxOrchestrator()
        orch.launch_worker(make_manifest("w1"))
        assert "w1" in orch.containers

    def test_launch_sets_status_running(self):
        orch = SandboxOrchestrator()
        orch.launch_worker(make_manifest("w1"))
        assert orch.containers["w1"].status == "running"

    def test_launch_stores_manifest(self):
        orch = SandboxOrchestrator()
        m = make_manifest("w1")
        orch.launch_worker(m)
        assert orch.containers["w1"].manifest is m

    def test_launch_stores_resources(self):
        orch = SandboxOrchestrator()
        m = make_manifest("w1", cpu=2.0, memory=512)
        orch.launch_worker(m)
        assert orch.containers["w1"].resources.cpu_limit == 2.0
        assert orch.containers["w1"].resources.memory_limit_mb == 512

    def test_launch_sets_started_at(self):
        orch = SandboxOrchestrator()
        before = datetime.now(timezone.utc)
        orch.launch_worker(make_manifest("w1"))
        after = datetime.now(timezone.utc)
        assert before <= orch.containers["w1"].started_at <= after

    def test_launch_logs_event(self):
        logger = StructuredLogger()
        orch = SandboxOrchestrator(logger=logger)
        orch.launch_worker(make_manifest("w1", parent_id="root"))
        assert len(logger.events) == 1
        assert logger.events[0]["event"] == "container_launched"
        assert logger.events[0]["worker_id"] == "w1"
        assert logger.events[0]["parent_id"] == "root"

    def test_launch_logs_cpu_limit(self):
        logger = StructuredLogger()
        orch = SandboxOrchestrator(logger=logger)
        orch.launch_worker(make_manifest("w1", cpu=4.0))
        assert logger.events[0]["cpu_limit"] == 4.0

    def test_launch_logs_memory_limit(self):
        logger = StructuredLogger()
        orch = SandboxOrchestrator(logger=logger)
        orch.launch_worker(make_manifest("w1", memory=1024))
        assert logger.events[0]["memory_limit_mb"] == 1024

    def test_launch_logs_network_policy(self):
        logger = StructuredLogger()
        orch = SandboxOrchestrator(logger=logger)
        orch.launch_worker(make_manifest("w1"))
        assert logger.events[0]["allow_external"] is False

    def test_launch_multiple_workers(self):
        orch = SandboxOrchestrator()
        orch.launch_worker(make_manifest("w1"))
        orch.launch_worker(make_manifest("w2"))
        orch.launch_worker(make_manifest("w3"))
        assert len(orch.containers) == 3

    def test_launch_replaces_existing_worker(self):
        orch = SandboxOrchestrator()
        m1 = make_manifest("w1", cpu=1.0)
        m2 = make_manifest("w1", cpu=2.0)
        orch.launch_worker(m1)
        orch.launch_worker(m2)
        assert len(orch.containers) == 1
        assert orch.containers["w1"].resources.cpu_limit == 2.0


class TestKillWorker:
    """SandboxOrchestrator.kill_worker()."""

    def test_kill_removes_container(self):
        orch = SandboxOrchestrator()
        orch.launch_worker(make_manifest("w1"))
        orch.kill_worker("w1", "test")
        assert "w1" not in orch.containers

    def test_kill_nonexistent_does_not_raise(self):
        orch = SandboxOrchestrator()
        orch.kill_worker("nonexistent", "test")  # Should not raise

    def test_kill_logs_event(self):
        logger = StructuredLogger()
        orch = SandboxOrchestrator(logger=logger)
        orch.launch_worker(make_manifest("w1"))
        orch.kill_worker("w1", "depth_limit")
        kill_events = [e for e in logger.events if e["event"] == "container_killed"]
        assert len(kill_events) == 1
        assert kill_events[0]["worker_id"] == "w1"
        assert kill_events[0]["reason"] == "depth_limit"

    def test_kill_nonexistent_does_not_log(self):
        logger = StructuredLogger()
        orch = SandboxOrchestrator(logger=logger)
        orch.kill_worker("ghost", "test")
        kill_events = [e for e in logger.events if e["event"] == "container_killed"]
        assert len(kill_events) == 0

    def test_kill_one_of_many(self):
        orch = SandboxOrchestrator()
        orch.launch_worker(make_manifest("w1"))
        orch.launch_worker(make_manifest("w2"))
        orch.launch_worker(make_manifest("w3"))
        orch.kill_worker("w2", "test")
        assert "w1" in orch.containers
        assert "w2" not in orch.containers
        assert "w3" in orch.containers


class TestKillAll:
    """SandboxOrchestrator.kill_all()."""

    def test_kill_all_empties_containers(self):
        orch = SandboxOrchestrator()
        for i in range(5):
            orch.launch_worker(make_manifest(f"w{i}"))
        orch.kill_all("shutdown")
        assert len(orch.containers) == 0

    def test_kill_all_logs_each_kill(self):
        logger = StructuredLogger()
        orch = SandboxOrchestrator(logger=logger)
        for i in range(3):
            orch.launch_worker(make_manifest(f"w{i}"))
        orch.kill_all("emergency")
        kill_events = [e for e in logger.events if e["event"] == "container_killed"]
        assert len(kill_events) == 3
        for e in kill_events:
            assert e["reason"] == "emergency"

    def test_kill_all_empty_does_not_raise(self):
        orch = SandboxOrchestrator()
        orch.kill_all("test")  # Should not raise

    def test_kill_all_sets_correct_worker_ids(self):
        logger = StructuredLogger()
        orch = SandboxOrchestrator(logger=logger)
        orch.launch_worker(make_manifest("alpha"))
        orch.launch_worker(make_manifest("beta"))
        orch.kill_all("test")
        killed_ids = {e["worker_id"] for e in logger.events if e["event"] == "container_killed"}
        assert killed_ids == {"alpha", "beta"}


class TestEnforceResourceBounds:
    """SandboxOrchestrator.enforce_resource_bounds()."""

    def test_enforce_emits_metric(self):
        logger = StructuredLogger()
        orch = SandboxOrchestrator(logger=logger)
        orch.launch_worker(make_manifest("w1", cpu=2.0, memory=512))
        orch.enforce_resource_bounds("w1")
        assert len(logger.metrics) == 1
        assert logger.metrics[0].name == "resource_enforced"
        assert logger.metrics[0].value == 1

    def test_enforce_metric_labels(self):
        logger = StructuredLogger()
        orch = SandboxOrchestrator(logger=logger)
        orch.launch_worker(make_manifest("w1", cpu=4.0, memory=1024))
        orch.enforce_resource_bounds("w1")
        labels = logger.metrics[0].labels
        assert labels["worker_id"] == "w1"
        assert labels["cpu"] == "4.0"
        assert labels["memory"] == "1024"

    def test_enforce_nonexistent_does_not_emit(self):
        logger = StructuredLogger()
        orch = SandboxOrchestrator(logger=logger)
        orch.enforce_resource_bounds("ghost")
        assert len(logger.metrics) == 0

    def test_enforce_multiple_workers(self):
        logger = StructuredLogger()
        orch = SandboxOrchestrator(logger=logger)
        orch.launch_worker(make_manifest("w1"))
        orch.launch_worker(make_manifest("w2"))
        orch.enforce_resource_bounds("w1")
        orch.enforce_resource_bounds("w2")
        assert len(logger.metrics) == 2
        worker_ids = {m.labels["worker_id"] for m in logger.metrics}
        assert worker_ids == {"w1", "w2"}


class TestContainerRecord:
    """ContainerRecord dataclass."""

    def test_record_default_reason_none(self):
        m = make_manifest("w1")
        record = ContainerRecord(
            manifest=m, resources=m.resources,
            status="running", started_at=datetime.now(timezone.utc)
        )
        assert record.reason is None

    def test_record_with_reason(self):
        m = make_manifest("w1")
        record = ContainerRecord(
            manifest=m, resources=m.resources,
            status="killed", started_at=datetime.now(timezone.utc),
            reason="policy_violation"
        )
        assert record.reason == "policy_violation"

    def test_record_stores_status(self):
        m = make_manifest("w1")
        record = ContainerRecord(
            manifest=m, resources=m.resources,
            status="pending", started_at=datetime.now(timezone.utc)
        )
        assert record.status == "pending"
