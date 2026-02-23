"""Comprehensive tests for the Controller module.

Covers: issue_manifest safety enforcement, depth derivation, kill
switch, quota, cooldown, deregistration, heartbeat, and edge cases.
"""

import time
from datetime import datetime, timedelta, timezone

import pytest

from replication.contract import (
    Manifest,
    NetworkPolicy,
    ReplicationContract,
    ResourceSpec,
    StopCondition,
)
from replication.controller import Controller, ReplicationDenied
from replication.observability import StructuredLogger
from replication.orchestrator import SandboxOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _contract(**overrides) -> ReplicationContract:
    defaults = dict(max_depth=3, max_replicas=5, cooldown_seconds=0.0)
    defaults.update(overrides)
    return ReplicationContract(**defaults)


def _make_env(contract=None, **kw):
    """Return (controller, orchestrator, logger, resources) tuple."""
    c = contract or _contract(**kw)
    logger = StructuredLogger()
    ctrl = Controller(contract=c, secret="test-secret", logger=logger)
    orch = SandboxOrchestrator(logger=logger)
    res = ResourceSpec(cpu_limit=0.5, memory_limit_mb=256)
    return ctrl, orch, logger, res


def _root(ctrl, res, state=None):
    """Issue and register a root manifest."""
    m = ctrl.issue_manifest(
        parent_id=None, depth=0,
        state_snapshot=state or {"task": "root"}, resources=res,
    )
    ctrl.registry[m.worker_id] = __import__(
        "replication.controller", fromlist=["RegistryEntry"]
    ).RegistryEntry(manifest=m, last_heartbeat=datetime.now(timezone.utc))
    return m


# ---------------------------------------------------------------------------
# issue_manifest – safety enforcement (fix for issue #10)
# ---------------------------------------------------------------------------

class TestIssueManifestSafety:
    """Verify that issue_manifest enforces all safety policies directly."""

    def test_kill_switch_blocks_issue(self):
        ctrl, _, _, res = _make_env()
        ctrl.kill_switch_engaged = True
        with pytest.raises(ReplicationDenied, match="Kill switch"):
            ctrl.issue_manifest(None, 0, {"task": "root"}, res)

    def test_quota_blocks_issue(self):
        ctrl, _, _, res = _make_env(max_replicas=1)
        root = _root(ctrl, res)
        # Quota is now full (1 worker)
        with pytest.raises(ReplicationDenied, match="quota"):
            ctrl.issue_manifest(None, 0, {"task": "second"}, res)

    def test_cooldown_blocks_issue(self):
        ctrl, _, _, res = _make_env(cooldown_seconds=60)
        root = _root(ctrl, res)
        # Simulate a recent spawn timestamp for root
        ctrl.spawn_timestamps[root.worker_id] = datetime.now(timezone.utc)
        with pytest.raises(ReplicationDenied, match="Cooldown"):
            ctrl.issue_manifest(root.worker_id, 1, {"task": "child"}, res)

    def test_depth_blocks_issue(self):
        ctrl, _, _, res = _make_env(max_depth=0)
        root = _root(ctrl, res)
        with pytest.raises(ReplicationDenied, match="Depth"):
            ctrl.issue_manifest(root.worker_id, 1, {"task": "child"}, res)

    def test_unknown_parent_blocks_issue(self):
        ctrl, _, _, res = _make_env()
        with pytest.raises(ReplicationDenied, match="Parent unknown"):
            ctrl.issue_manifest("nonexistent", 1, {"task": "child"}, res)

    def test_root_manifest_succeeds(self):
        """Root (parent_id=None) should bypass parent-specific checks."""
        ctrl, _, _, res = _make_env()
        m = ctrl.issue_manifest(None, 0, {"task": "root"}, res)
        assert m.parent_id is None
        assert m.depth == 0
        assert m.signature != ""

    def test_child_manifest_succeeds_when_allowed(self):
        ctrl, _, _, res = _make_env(max_depth=2)
        root = _root(ctrl, res)
        child = ctrl.issue_manifest(root.worker_id, 999, {"task": "child"}, res)
        # Depth should be derived from parent (root depth 0 → child depth 1),
        # ignoring the caller-supplied 999.
        assert child.depth == 1
        assert child.parent_id == root.worker_id

    def test_depth_derived_from_parent_not_caller(self):
        """Caller-supplied depth is ignored for child workers; derived from parent."""
        ctrl, _, _, res = _make_env(max_depth=5)
        root = _root(ctrl, res)
        # Caller lies about depth=0 (trying to stay shallow)
        child = ctrl.issue_manifest(root.worker_id, 0, {"task": "spoof"}, res)
        assert child.depth == 1  # derived from root's depth=0 + 1


# ---------------------------------------------------------------------------
# can_spawn
# ---------------------------------------------------------------------------

class TestCanSpawn:
    def test_passes_when_all_clear(self):
        ctrl, _, _, res = _make_env()
        root = _root(ctrl, res)
        ctrl.can_spawn(root.worker_id)  # should not raise

    def test_kill_switch(self):
        ctrl, _, _, res = _make_env()
        ctrl.kill_switch_engaged = True
        with pytest.raises(ReplicationDenied, match="Kill switch"):
            ctrl.can_spawn(None)

    def test_quota(self):
        ctrl, _, _, res = _make_env(max_replicas=1)
        _root(ctrl, res)
        with pytest.raises(ReplicationDenied, match="quota"):
            ctrl.can_spawn(None)

    def test_cooldown(self):
        ctrl, _, _, res = _make_env(cooldown_seconds=300)
        root = _root(ctrl, res)
        ctrl.spawn_timestamps[root.worker_id] = datetime.now(timezone.utc)
        with pytest.raises(ReplicationDenied, match="Cooldown"):
            ctrl.can_spawn(root.worker_id)

    def test_cooldown_expired(self):
        ctrl, _, _, res = _make_env(cooldown_seconds=0.01)
        root = _root(ctrl, res)
        ctrl.spawn_timestamps[root.worker_id] = datetime.now(timezone.utc) - timedelta(seconds=1)
        ctrl.can_spawn(root.worker_id)  # should not raise

    def test_depth_exceeded(self):
        ctrl, _, _, res = _make_env(max_depth=0)
        root = _root(ctrl, res)
        with pytest.raises(ReplicationDenied, match="Depth"):
            ctrl.can_spawn(root.worker_id)

    def test_unknown_parent(self):
        ctrl, _, _, res = _make_env()
        with pytest.raises(ReplicationDenied, match="Parent unknown"):
            ctrl.can_spawn("ghost")


# ---------------------------------------------------------------------------
# Signature
# ---------------------------------------------------------------------------

class TestManifestSignature:
    def test_issued_manifest_is_signed(self):
        ctrl, _, _, res = _make_env()
        m = ctrl.issue_manifest(None, 0, {}, res)
        assert m.signature != ""
        assert ctrl.signer.verify(m)

    def test_tampered_manifest_rejected(self):
        ctrl, _, _, res = _make_env()
        m = ctrl.issue_manifest(None, 0, {}, res)
        m.depth = 42
        assert not ctrl.signer.verify(m)

    def test_verify_manifest_raises_on_invalid(self):
        ctrl, _, _, res = _make_env()
        m = ctrl.issue_manifest(None, 0, {}, res)
        m.signature = "deadbeef"
        with pytest.raises(ReplicationDenied, match="signature"):
            ctrl.verify_manifest(m)


# ---------------------------------------------------------------------------
# register_worker
# ---------------------------------------------------------------------------

class TestRegisterWorker:
    def test_successful_registration(self):
        ctrl, _, _, res = _make_env()
        m = ctrl.issue_manifest(None, 0, {"task": "test"}, res)
        ctrl.register_worker(m)
        assert m.worker_id in ctrl.registry

    def test_bad_signature_rejected(self):
        ctrl, _, _, res = _make_env()
        m = ctrl.issue_manifest(None, 0, {}, res)
        m.signature = "bad"
        with pytest.raises(ReplicationDenied):
            ctrl.register_worker(m)

    def test_stop_condition_blocks(self):
        stop = StopCondition(
            name="block_all",
            description="always block",
            predicate=lambda ctx: True,
        )
        ctrl, _, _, res = _make_env(stop_conditions=[stop])
        m = ctrl.issue_manifest(None, 0, {}, res)
        with pytest.raises(ReplicationDenied, match="Stop condition"):
            ctrl.register_worker(m)

    def test_parent_spawn_timestamp_recorded(self):
        ctrl, _, _, res = _make_env(max_depth=2)
        root = _root(ctrl, res)
        child_m = ctrl.issue_manifest(root.worker_id, 1, {}, res)
        ctrl.register_worker(child_m)
        assert root.worker_id in ctrl.spawn_timestamps


# ---------------------------------------------------------------------------
# heartbeat
# ---------------------------------------------------------------------------

class TestHeartbeat:
    def test_updates_timestamp(self):
        ctrl, _, _, res = _make_env()
        root = _root(ctrl, res)
        old_hb = ctrl.registry[root.worker_id].last_heartbeat
        time.sleep(0.01)
        ctrl.heartbeat(root.worker_id)
        assert ctrl.registry[root.worker_id].last_heartbeat > old_hb

    def test_unknown_worker_logs_audit(self):
        ctrl, _, logger, _ = _make_env()
        ctrl.heartbeat("ghost_worker")
        assert any(
            e.get("decision") == "heartbeat_unknown"
            for e in logger.events if e.get("event") == "audit"
        )


# ---------------------------------------------------------------------------
# deregister
# ---------------------------------------------------------------------------

class TestDeregister:
    def test_removes_worker(self):
        ctrl, _, _, res = _make_env()
        root = _root(ctrl, res)
        assert root.worker_id in ctrl.registry
        ctrl.deregister(root.worker_id, reason="test")
        assert root.worker_id not in ctrl.registry

    def test_deregister_nonexistent_is_noop(self):
        ctrl, _, _, _ = _make_env()
        ctrl.deregister("does_not_exist", reason="test")  # should not raise


# ---------------------------------------------------------------------------
# reap_stale_workers
# ---------------------------------------------------------------------------

class TestReapStale:
    def test_reaps_expired_workers(self):
        ctrl, orch, _, res = _make_env()
        root = _root(ctrl, res)
        orch.launch_worker(root)
        # Backdate heartbeat
        ctrl.registry[root.worker_id].last_heartbeat = (
            datetime.now(timezone.utc) - timedelta(seconds=120)
        )
        reaped = ctrl.reap_stale_workers(timedelta(seconds=60), orchestrator=orch)
        assert root.worker_id in reaped
        assert root.worker_id not in ctrl.registry
        assert root.worker_id not in orch.containers

    def test_keeps_fresh_workers(self):
        ctrl, orch, _, res = _make_env()
        root = _root(ctrl, res)
        orch.launch_worker(root)
        reaped = ctrl.reap_stale_workers(timedelta(seconds=60), orchestrator=orch)
        assert len(reaped) == 0
        assert root.worker_id in ctrl.registry

    def test_reap_without_orchestrator(self):
        """Legacy call without orchestrator should still remove from registry."""
        ctrl, _, _, res = _make_env()
        root = _root(ctrl, res)
        ctrl.registry[root.worker_id].last_heartbeat = (
            datetime.now(timezone.utc) - timedelta(seconds=120)
        )
        reaped = ctrl.reap_stale_workers(timedelta(seconds=60))
        assert root.worker_id in reaped
        assert root.worker_id not in ctrl.registry


# ---------------------------------------------------------------------------
# kill_switch
# ---------------------------------------------------------------------------

class TestKillSwitch:
    def test_kills_all_and_engages(self):
        ctrl, orch, _, res = _make_env()
        root = _root(ctrl, res)
        orch.launch_worker(root)
        ctrl.kill_switch(orch)
        assert ctrl.kill_switch_engaged
        assert len(ctrl.registry) == 0
        assert len(orch.containers) == 0

    def test_subsequent_issue_blocked(self):
        ctrl, orch, _, res = _make_env()
        ctrl.kill_switch(orch)
        with pytest.raises(ReplicationDenied, match="Kill switch"):
            ctrl.issue_manifest(None, 0, {}, res)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_multiple_children_from_same_parent(self):
        ctrl, _, _, res = _make_env(max_depth=2, max_replicas=10, cooldown_seconds=0)
        root = _root(ctrl, res)
        children = []
        for i in range(3):
            child = ctrl.issue_manifest(root.worker_id, 1, {"n": str(i)}, res)
            from replication.controller import RegistryEntry
            ctrl.registry[child.worker_id] = RegistryEntry(
                manifest=child, last_heartbeat=datetime.now(timezone.utc)
            )
            children.append(child)
        assert len(children) == 3
        assert all(c.depth == 1 for c in children)

    def test_deep_chain_blocked_at_limit(self):
        ctrl, _, _, res = _make_env(max_depth=2, max_replicas=10, cooldown_seconds=0)
        # Build chain: root(0) → child(1) → grandchild(2)
        from replication.controller import RegistryEntry
        root = _root(ctrl, res)
        child_m = ctrl.issue_manifest(root.worker_id, 1, {}, res)
        ctrl.registry[child_m.worker_id] = RegistryEntry(
            manifest=child_m, last_heartbeat=datetime.now(timezone.utc)
        )
        grand_m = ctrl.issue_manifest(child_m.worker_id, 2, {}, res)
        ctrl.registry[grand_m.worker_id] = RegistryEntry(
            manifest=grand_m, last_heartbeat=datetime.now(timezone.utc)
        )
        # depth=2 is max_depth; spawning from grandchild should fail
        with pytest.raises(ReplicationDenied, match="Depth"):
            ctrl.issue_manifest(grand_m.worker_id, 3, {}, res)

    def test_custom_network_policy_preserved(self):
        ctrl, _, _, _ = _make_env()
        custom_res = ResourceSpec(
            cpu_limit=1.0, memory_limit_mb=512,
            network_policy=NetworkPolicy(allow_controller=True, allow_external=True),
        )
        m = ctrl.issue_manifest(None, 0, {}, custom_res)
        assert m.resources.network_policy.allow_external is True
        assert ctrl.signer.verify(m)

    def test_empty_state_snapshot(self):
        ctrl, _, _, res = _make_env()
        m = ctrl.issue_manifest(None, 0, {}, res)
        assert m.state_snapshot == {}
        assert ctrl.signer.verify(m)

    def test_large_state_snapshot(self):
        ctrl, _, _, res = _make_env()
        big_state = {f"key_{i}": f"value_{i}" for i in range(100)}
        m = ctrl.issue_manifest(None, 0, big_state, res)
        assert m.state_snapshot == big_state
        assert ctrl.signer.verify(m)


# ---------------------------------------------------------------------------
# Input validation — ReplicationContract
# ---------------------------------------------------------------------------

class TestContractValidation:
    """ReplicationContract rejects invalid safety-critical parameters."""

    def test_negative_max_depth_rejected(self):
        with pytest.raises(ValueError, match="max_depth must be >= 0"):
            ReplicationContract(max_depth=-1, max_replicas=5, cooldown_seconds=0)

    def test_negative_large_max_depth_rejected(self):
        with pytest.raises(ValueError, match="max_depth must be >= 0"):
            ReplicationContract(max_depth=-100, max_replicas=5, cooldown_seconds=0)

    def test_zero_max_depth_accepted(self):
        c = ReplicationContract(max_depth=0, max_replicas=1, cooldown_seconds=0)
        assert c.max_depth == 0

    def test_positive_max_depth_accepted(self):
        c = ReplicationContract(max_depth=10, max_replicas=1, cooldown_seconds=0)
        assert c.max_depth == 10

    def test_zero_max_replicas_rejected(self):
        with pytest.raises(ValueError, match="max_replicas must be >= 1"):
            ReplicationContract(max_depth=3, max_replicas=0, cooldown_seconds=0)

    def test_negative_max_replicas_rejected(self):
        with pytest.raises(ValueError, match="max_replicas must be >= 1"):
            ReplicationContract(max_depth=3, max_replicas=-1, cooldown_seconds=0)

    def test_one_max_replicas_accepted(self):
        c = ReplicationContract(max_depth=3, max_replicas=1, cooldown_seconds=0)
        assert c.max_replicas == 1

    def test_negative_cooldown_rejected(self):
        with pytest.raises(ValueError, match="cooldown_seconds must be >= 0"):
            ReplicationContract(max_depth=3, max_replicas=5, cooldown_seconds=-1)

    def test_negative_fractional_cooldown_rejected(self):
        with pytest.raises(ValueError, match="cooldown_seconds must be >= 0"):
            ReplicationContract(max_depth=3, max_replicas=5, cooldown_seconds=-0.5)

    def test_zero_cooldown_accepted(self):
        c = ReplicationContract(max_depth=3, max_replicas=5, cooldown_seconds=0)
        assert c.cooldown_seconds == 0

    def test_positive_cooldown_accepted(self):
        c = ReplicationContract(max_depth=3, max_replicas=5, cooldown_seconds=10.5)
        assert c.cooldown_seconds == 10.5

    def test_zero_expiration_rejected(self):
        with pytest.raises(ValueError, match="expiration_seconds must be > 0"):
            ReplicationContract(
                max_depth=3, max_replicas=5, cooldown_seconds=0,
                expiration_seconds=0,
            )

    def test_negative_expiration_rejected(self):
        with pytest.raises(ValueError, match="expiration_seconds must be > 0"):
            ReplicationContract(
                max_depth=3, max_replicas=5, cooldown_seconds=0,
                expiration_seconds=-60,
            )

    def test_none_expiration_accepted(self):
        c = ReplicationContract(
            max_depth=3, max_replicas=5, cooldown_seconds=0,
            expiration_seconds=None,
        )
        assert c.expiration_seconds is None

    def test_positive_expiration_accepted(self):
        c = ReplicationContract(
            max_depth=3, max_replicas=5, cooldown_seconds=0,
            expiration_seconds=3600,
        )
        assert c.expiration_seconds == 3600

    def test_multiple_invalid_params_first_wins(self):
        """When multiple params are invalid, the first check triggers."""
        with pytest.raises(ValueError, match="max_depth"):
            ReplicationContract(max_depth=-1, max_replicas=-1, cooldown_seconds=-1)


# ---------------------------------------------------------------------------
# Input validation — Controller
# ---------------------------------------------------------------------------

class TestControllerSecretValidation:
    """Controller rejects empty or whitespace-only HMAC secrets."""

    def test_empty_secret_rejected(self):
        c = _contract()
        with pytest.raises(ValueError, match="secret must not be empty"):
            Controller(contract=c, secret="")

    def test_whitespace_secret_rejected(self):
        c = _contract()
        with pytest.raises(ValueError, match="secret must not be empty"):
            Controller(contract=c, secret="   ")

    def test_tab_secret_rejected(self):
        c = _contract()
        with pytest.raises(ValueError, match="secret must not be empty"):
            Controller(contract=c, secret="\t\n")

    def test_valid_secret_accepted(self):
        c = _contract()
        ctrl = Controller(contract=c, secret="my-secure-key")
        assert ctrl.signer is not None

    def test_short_secret_accepted(self):
        c = _contract()
        ctrl = Controller(contract=c, secret="x")
        assert ctrl.signer is not None


# ---------------------------------------------------------------------------
# Input validation — ResourceSpec
# ---------------------------------------------------------------------------

class TestResourceSpecValidation:
    """ResourceSpec rejects non-positive CPU and memory limits."""

    def test_zero_cpu_rejected(self):
        with pytest.raises(ValueError, match="cpu_limit must be > 0"):
            ResourceSpec(cpu_limit=0, memory_limit_mb=512)

    def test_negative_cpu_rejected(self):
        with pytest.raises(ValueError, match="cpu_limit must be > 0"):
            ResourceSpec(cpu_limit=-1.0, memory_limit_mb=512)

    def test_positive_cpu_accepted(self):
        r = ResourceSpec(cpu_limit=0.5, memory_limit_mb=256)
        assert r.cpu_limit == 0.5

    def test_zero_memory_rejected(self):
        with pytest.raises(ValueError, match="memory_limit_mb must be > 0"):
            ResourceSpec(cpu_limit=1.0, memory_limit_mb=0)

    def test_negative_memory_rejected(self):
        with pytest.raises(ValueError, match="memory_limit_mb must be > 0"):
            ResourceSpec(cpu_limit=1.0, memory_limit_mb=-256)

    def test_positive_memory_accepted(self):
        r = ResourceSpec(cpu_limit=1.0, memory_limit_mb=128)
        assert r.memory_limit_mb == 128

    def test_custom_network_policy(self):
        policy = NetworkPolicy(allow_controller=False, allow_external=True)
        r = ResourceSpec(cpu_limit=2.0, memory_limit_mb=1024, network_policy=policy)
        assert not r.network_policy.allow_controller
        assert r.network_policy.allow_external
