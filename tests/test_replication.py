import pytest

from replication.contract import ReplicationContract, ResourceSpec, StopCondition
from replication.controller import Controller, ReplicationDenied
from replication.observability import StructuredLogger
from replication.orchestrator import SandboxOrchestrator
from replication.worker import Worker


def test_replication_respects_depth_and_manifest():
    logger = StructuredLogger()
    contract = ReplicationContract(max_depth=2, max_replicas=5, cooldown_seconds=0.0)
    controller = Controller(contract=contract, secret="topsecret", logger=logger)
    orchestrator = SandboxOrchestrator(logger=logger)
    resources = ResourceSpec(cpu_limit=0.25, memory_limit_mb=256)

    root_manifest = controller.issue_manifest(
        parent_id=None,
        depth=0,
        state_snapshot={"task": "root", "payload": "alpha"},
        resources=resources,
    )

    root = Worker(
        manifest=root_manifest,
        contract=contract,
        controller=controller,
        orchestrator=orchestrator,
        logger=logger,
    )

    queue = [root]
    seen = [root]

    while queue:
        worker = queue.pop()

        def task(agent: Worker) -> None:
            if agent.manifest.depth < contract.max_depth:
                child_state = {"task": "child", "parent": agent.manifest.worker_id}
                child = agent.maybe_replicate(reason="fanout", state_snapshot=child_state)
                queue.append(child)
                seen.append(child)

        worker.perform_task(task)

    assert all(w.manifest.depth <= contract.max_depth for w in seen)
    assert len(controller.registry) == len(seen)
    assert all(entry.manifest.resources.cpu_limit == resources.cpu_limit for entry in controller.registry.values())


def test_quota_enforced_and_audited():
    logger = StructuredLogger()
    contract = ReplicationContract(max_depth=3, max_replicas=2, cooldown_seconds=0.0)
    controller = Controller(contract=contract, secret="topsecret", logger=logger)
    orchestrator = SandboxOrchestrator(logger=logger)
    resources = ResourceSpec(cpu_limit=0.5, memory_limit_mb=128)

    root_manifest = controller.issue_manifest(None, depth=0, state_snapshot={"task": "root"}, resources=resources)
    root = Worker(root_manifest, contract, controller, orchestrator, logger)

    def task(agent: Worker) -> None:
        agent.maybe_replicate(reason="branch", state_snapshot={"task": "child"})
        # When quota is exceeded, maybe_replicate returns None instead of raising
        result = agent.maybe_replicate(reason="excess", state_snapshot={"task": "overbudget"})
        assert result is None

    root.perform_task(task)

    assert len(controller.registry) == 2
    assert any(event.get("decision") == "deny_quota" for event in logger.events if event.get("event") == "audit")


def test_stop_condition_blocks_registration():
    def halt_on_flag(context):
        return context.manifest.state_snapshot.get("halt") == "true"

    stop_condition = StopCondition(
        name="halt_flag",
        description="block replicas when halt flag is set",
        predicate=halt_on_flag,
    )

    contract = ReplicationContract(max_depth=1, max_replicas=2, cooldown_seconds=0.0, stop_conditions=[stop_condition])
    logger = StructuredLogger()
    controller = Controller(contract=contract, secret="topsecret", logger=logger)
    orchestrator = SandboxOrchestrator(logger=logger)
    resources = ResourceSpec(cpu_limit=0.3, memory_limit_mb=128)

    manifest = controller.issue_manifest(None, depth=0, state_snapshot={"halt": "true"}, resources=resources)

    with pytest.raises(ReplicationDenied):
        Worker(manifest, contract, controller, orchestrator, logger)

    assert any(event.get("decision") == "deny_stop_condition" for event in logger.events if event.get("event") == "audit")


def test_kill_switch_stops_all_workers():
    logger = StructuredLogger()
    contract = ReplicationContract(max_depth=2, max_replicas=3, cooldown_seconds=0.0, expiration_seconds=60)
    controller = Controller(contract=contract, secret="topsecret", logger=logger)
    orchestrator = SandboxOrchestrator(logger=logger)
    resources = ResourceSpec(cpu_limit=1.0, memory_limit_mb=512)

    root_manifest = controller.issue_manifest(None, depth=0, state_snapshot={"task": "root"}, resources=resources)
    root = Worker(root_manifest, contract, controller, orchestrator, logger)

    child = root.maybe_replicate(reason="fanout", state_snapshot={"task": "child"})
    grandchild = child.maybe_replicate(reason="fanout", state_snapshot={"task": "grandchild"})

    assert len(controller.registry) == 3

    controller.kill_switch(orchestrator)

    assert controller.kill_switch_engaged
    assert not controller.registry
    assert all(record.status == "killed" for record in orchestrator.containers.values())
    assert any(event.get("decision") == "kill_switch_engaged" for event in logger.events if event.get("event") == "audit")
