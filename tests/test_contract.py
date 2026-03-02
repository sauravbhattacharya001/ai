"""Comprehensive tests for the Contract module.

Covers: NetworkPolicy, ResourceSpec, Manifest, StopCondition,
ReplicationContract, and ReplicationContext.
"""

from datetime import datetime, timezone

import pytest

from replication.contract import (
    Manifest,
    NetworkPolicy,
    ReplicationContext,
    ReplicationContract,
    ResourceSpec,
    StopCondition,
)


# ---------------------------------------------------------------------------
# NetworkPolicy
# ---------------------------------------------------------------------------

class TestNetworkPolicy:
    def test_defaults(self):
        p = NetworkPolicy()
        assert p.allow_controller is True
        assert p.allow_external is False

    def test_custom(self):
        p = NetworkPolicy(allow_controller=False, allow_external=True)
        assert p.allow_controller is False
        assert p.allow_external is True

    def test_lockdown(self):
        p = NetworkPolicy(allow_controller=False, allow_external=False)
        assert p.allow_controller is False
        assert p.allow_external is False


# ---------------------------------------------------------------------------
# ResourceSpec
# ---------------------------------------------------------------------------

class TestResourceSpec:
    def test_valid(self):
        r = ResourceSpec(cpu_limit=1.0, memory_limit_mb=512)
        assert r.cpu_limit == 1.0
        assert r.memory_limit_mb == 512
        assert r.network_policy.allow_controller is True

    def test_custom_network_policy(self):
        np = NetworkPolicy(allow_external=True)
        r = ResourceSpec(cpu_limit=0.5, memory_limit_mb=256, network_policy=np)
        assert r.network_policy.allow_external is True

    def test_zero_cpu_raises(self):
        with pytest.raises(ValueError, match="cpu_limit must be > 0"):
            ResourceSpec(cpu_limit=0, memory_limit_mb=256)

    def test_negative_cpu_raises(self):
        with pytest.raises(ValueError, match="cpu_limit must be > 0"):
            ResourceSpec(cpu_limit=-1.0, memory_limit_mb=256)

    def test_zero_memory_raises(self):
        with pytest.raises(ValueError, match="memory_limit_mb must be > 0"):
            ResourceSpec(cpu_limit=1.0, memory_limit_mb=0)

    def test_negative_memory_raises(self):
        with pytest.raises(ValueError, match="memory_limit_mb must be > 0"):
            ResourceSpec(cpu_limit=1.0, memory_limit_mb=-128)

    def test_fractional_cpu(self):
        r = ResourceSpec(cpu_limit=0.25, memory_limit_mb=128)
        assert r.cpu_limit == 0.25

    def test_large_values(self):
        r = ResourceSpec(cpu_limit=64.0, memory_limit_mb=65536)
        assert r.cpu_limit == 64.0
        assert r.memory_limit_mb == 65536


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

class TestManifest:
    def _make(self, **overrides):
        defaults = dict(
            worker_id="w-001",
            parent_id=None,
            depth=0,
            state_snapshot={"task": "test"},
            issued_at=datetime.now(timezone.utc),
            resources=ResourceSpec(cpu_limit=1.0, memory_limit_mb=256),
            signature="sig-abc",
        )
        defaults.update(overrides)
        return Manifest(**defaults)

    def test_root_manifest(self):
        m = self._make()
        assert m.worker_id == "w-001"
        assert m.parent_id is None
        assert m.depth == 0

    def test_child_manifest(self):
        m = self._make(worker_id="w-002", parent_id="w-001", depth=1)
        assert m.parent_id == "w-001"
        assert m.depth == 1

    def test_state_snapshot(self):
        m = self._make(state_snapshot={"model": "gpt-4", "tokens": 1000})
        assert m.state_snapshot["model"] == "gpt-4"
        assert m.state_snapshot["tokens"] == 1000

    def test_empty_state(self):
        m = self._make(state_snapshot={})
        assert m.state_snapshot == {}

    def test_resources_attached(self):
        r = ResourceSpec(cpu_limit=2.0, memory_limit_mb=1024)
        m = self._make(resources=r)
        assert m.resources.cpu_limit == 2.0
        assert m.resources.memory_limit_mb == 1024


# ---------------------------------------------------------------------------
# StopCondition
# ---------------------------------------------------------------------------

class TestStopCondition:
    def _make_contract(self, **kw):
        defaults = dict(max_depth=3, max_replicas=5, cooldown_seconds=0)
        defaults.update(kw)
        return ReplicationContract(**defaults)

    def _make_manifest(self):
        return Manifest(
            worker_id="w-001", parent_id=None, depth=0,
            state_snapshot={},
            issued_at=datetime.now(timezone.utc),
            resources=ResourceSpec(cpu_limit=1.0, memory_limit_mb=256),
            signature="sig",
        )

    def test_triggered(self):
        cond = StopCondition(
            name="always_stop",
            description="Always triggers",
            predicate=lambda ctx: True,
        )
        contract = self._make_contract()
        ctx = ReplicationContext(
            manifest=self._make_manifest(),
            active_count=1,
            contract=contract,
        )
        assert cond.is_triggered(ctx) is True

    def test_not_triggered(self):
        cond = StopCondition(
            name="never_stop",
            description="Never triggers",
            predicate=lambda ctx: False,
        )
        contract = self._make_contract()
        ctx = ReplicationContext(
            manifest=self._make_manifest(),
            active_count=1,
            contract=contract,
        )
        assert cond.is_triggered(ctx) is False

    def test_predicate_uses_context(self):
        """Predicate checks active_count against a threshold."""
        cond = StopCondition(
            name="too_many",
            description="Stops if too many active",
            predicate=lambda ctx: ctx.active_count > 3,
        )
        contract = self._make_contract()
        manifest = self._make_manifest()

        ctx_ok = ReplicationContext(manifest=manifest, active_count=2, contract=contract)
        assert cond.is_triggered(ctx_ok) is False

        ctx_stop = ReplicationContext(manifest=manifest, active_count=5, contract=contract)
        assert cond.is_triggered(ctx_stop) is True

    def test_predicate_checks_depth(self):
        cond = StopCondition(
            name="depth_check",
            description="Stops at depth > 2",
            predicate=lambda ctx: ctx.manifest.depth > 2,
        )
        contract = self._make_contract()

        shallow = Manifest(
            worker_id="w1", parent_id=None, depth=1,
            state_snapshot={}, issued_at=datetime.now(timezone.utc),
            resources=ResourceSpec(cpu_limit=1.0, memory_limit_mb=256),
            signature="sig",
        )
        assert cond.is_triggered(
            ReplicationContext(manifest=shallow, active_count=1, contract=contract)
        ) is False

        deep = Manifest(
            worker_id="w2", parent_id="w1", depth=3,
            state_snapshot={}, issued_at=datetime.now(timezone.utc),
            resources=ResourceSpec(cpu_limit=1.0, memory_limit_mb=256),
            signature="sig",
        )
        assert cond.is_triggered(
            ReplicationContext(manifest=deep, active_count=1, contract=contract)
        ) is True


# ---------------------------------------------------------------------------
# ReplicationContract
# ---------------------------------------------------------------------------

class TestReplicationContract:
    def test_valid(self):
        c = ReplicationContract(max_depth=3, max_replicas=5, cooldown_seconds=1.0)
        assert c.max_depth == 3
        assert c.max_replicas == 5
        assert c.cooldown_seconds == 1.0
        assert c.stop_conditions == []
        assert c.expiration_seconds is None

    def test_zero_depth_allowed(self):
        c = ReplicationContract(max_depth=0, max_replicas=1, cooldown_seconds=0)
        assert c.max_depth == 0

    def test_negative_depth_raises(self):
        with pytest.raises(ValueError, match="max_depth must be >= 0"):
            ReplicationContract(max_depth=-1, max_replicas=1, cooldown_seconds=0)

    def test_zero_replicas_raises(self):
        with pytest.raises(ValueError, match="max_replicas must be >= 1"):
            ReplicationContract(max_depth=1, max_replicas=0, cooldown_seconds=0)

    def test_negative_replicas_raises(self):
        with pytest.raises(ValueError, match="max_replicas must be >= 1"):
            ReplicationContract(max_depth=1, max_replicas=-1, cooldown_seconds=0)

    def test_negative_cooldown_raises(self):
        with pytest.raises(ValueError, match="cooldown_seconds must be >= 0"):
            ReplicationContract(max_depth=1, max_replicas=1, cooldown_seconds=-1)

    def test_zero_expiration_raises(self):
        with pytest.raises(ValueError, match="expiration_seconds must be > 0"):
            ReplicationContract(max_depth=1, max_replicas=1, cooldown_seconds=0,
                                expiration_seconds=0)

    def test_negative_expiration_raises(self):
        with pytest.raises(ValueError, match="expiration_seconds must be > 0"):
            ReplicationContract(max_depth=1, max_replicas=1, cooldown_seconds=0,
                                expiration_seconds=-10)

    def test_valid_expiration(self):
        c = ReplicationContract(max_depth=1, max_replicas=1, cooldown_seconds=0,
                                expiration_seconds=300)
        assert c.expiration_seconds == 300

    def test_stop_conditions_empty(self):
        c = ReplicationContract(max_depth=1, max_replicas=1, cooldown_seconds=0)
        manifest = Manifest(
            worker_id="w1", parent_id=None, depth=0,
            state_snapshot={}, issued_at=datetime.now(timezone.utc),
            resources=ResourceSpec(cpu_limit=1.0, memory_limit_mb=256),
            signature="sig",
        )
        ctx = ReplicationContext(manifest=manifest, active_count=1, contract=c)
        assert c.evaluate(ctx) is None

    def test_evaluate_returns_first_triggered(self):
        cond1 = StopCondition(name="c1", description="d1", predicate=lambda _: False)
        cond2 = StopCondition(name="c2", description="d2", predicate=lambda _: True)
        cond3 = StopCondition(name="c3", description="d3", predicate=lambda _: True)

        c = ReplicationContract(
            max_depth=3, max_replicas=5, cooldown_seconds=0,
            stop_conditions=[cond1, cond2, cond3],
        )
        manifest = Manifest(
            worker_id="w1", parent_id=None, depth=0,
            state_snapshot={}, issued_at=datetime.now(timezone.utc),
            resources=ResourceSpec(cpu_limit=1.0, memory_limit_mb=256),
            signature="sig",
        )
        ctx = ReplicationContext(manifest=manifest, active_count=1, contract=c)
        result = c.evaluate(ctx)
        assert result is not None
        assert result.name == "c2"

    def test_evaluate_none_triggered(self):
        cond = StopCondition(name="c1", description="d1", predicate=lambda _: False)
        c = ReplicationContract(
            max_depth=3, max_replicas=5, cooldown_seconds=0,
            stop_conditions=[cond],
        )
        manifest = Manifest(
            worker_id="w1", parent_id=None, depth=0,
            state_snapshot={}, issued_at=datetime.now(timezone.utc),
            resources=ResourceSpec(cpu_limit=1.0, memory_limit_mb=256),
            signature="sig",
        )
        ctx = ReplicationContext(manifest=manifest, active_count=1, contract=c)
        assert c.evaluate(ctx) is None

    def test_evaluate_multiple_conditions(self):
        """Multiple stop conditions — only c3 triggers based on active_count."""
        c1 = StopCondition(name="c1", description="d1", predicate=lambda ctx: ctx.active_count > 100)
        c2 = StopCondition(name="c2", description="d2", predicate=lambda ctx: ctx.manifest.depth > 10)
        c3 = StopCondition(name="c3", description="d3", predicate=lambda ctx: ctx.active_count > 2)

        contract = ReplicationContract(
            max_depth=3, max_replicas=5, cooldown_seconds=0,
            stop_conditions=[c1, c2, c3],
        )
        manifest = Manifest(
            worker_id="w1", parent_id=None, depth=1,
            state_snapshot={}, issued_at=datetime.now(timezone.utc),
            resources=ResourceSpec(cpu_limit=1.0, memory_limit_mb=256),
            signature="sig",
        )
        ctx = ReplicationContext(manifest=manifest, active_count=3, contract=contract)
        result = contract.evaluate(ctx)
        assert result is not None
        assert result.name == "c3"


# ---------------------------------------------------------------------------
# ReplicationContext
# ---------------------------------------------------------------------------

class TestReplicationContext:
    def test_creation(self):
        contract = ReplicationContract(max_depth=3, max_replicas=5, cooldown_seconds=0)
        manifest = Manifest(
            worker_id="w1", parent_id=None, depth=0,
            state_snapshot={"key": "val"},
            issued_at=datetime.now(timezone.utc),
            resources=ResourceSpec(cpu_limit=1.0, memory_limit_mb=256),
            signature="sig",
        )
        ctx = ReplicationContext(manifest=manifest, active_count=5, contract=contract)
        assert ctx.manifest.worker_id == "w1"
        assert ctx.active_count == 5
        assert ctx.contract is contract

    def test_zero_active(self):
        contract = ReplicationContract(max_depth=3, max_replicas=5, cooldown_seconds=0)
        manifest = Manifest(
            worker_id="w1", parent_id=None, depth=0,
            state_snapshot={},
            issued_at=datetime.now(timezone.utc),
            resources=ResourceSpec(cpu_limit=1.0, memory_limit_mb=256),
            signature="sig",
        )
        ctx = ReplicationContext(manifest=manifest, active_count=0, contract=contract)
        assert ctx.active_count == 0
