from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .contract import Manifest, ReplicationContext, ReplicationContract
from .observability import StructuredLogger
from .signer import ManifestSigner


@dataclass
class RegistryEntry:
    manifest: Manifest
    last_heartbeat: datetime


class ReplicationDenied(Exception):
    pass


class Controller:
    """Tracks active replicas, enforces quotas, and delegates signing to *ManifestSigner*.

    Crypto is handled by :class:`ManifestSigner`, keeping this class
    focused on lifecycle management and policy enforcement.
    """

    def __init__(self, contract: ReplicationContract, secret: str, logger: Optional[StructuredLogger] = None):
        if not secret or not secret.strip():
            raise ValueError(
                "Controller secret must not be empty or whitespace — "
                "an empty HMAC key offers no authenticity guarantee"
            )
        self.contract = contract
        self.signer = ManifestSigner(secret)
        self.logger = logger or StructuredLogger()
        self.registry: Dict[str, RegistryEntry] = {}
        self.spawn_timestamps: Dict[str, datetime] = {}
        self.kill_switch_engaged = False

    # -- Manifest helpers (delegate to signer) -------------------------

    def sign_manifest(self, manifest: Manifest) -> Manifest:
        return self.signer.sign(manifest)

    def verify_manifest(self, manifest: Manifest) -> None:
        if not self.signer.verify(manifest):
            self.logger.audit("reject_manifest_signature", worker_id=manifest.worker_id)
            raise ReplicationDenied("Manifest signature invalid")

    def issue_manifest(self, parent_id: Optional[str], depth: int, state_snapshot: Dict[str, Any], resources) -> Manifest:
        """Create and sign a manifest after enforcing all safety policies.

        Safety checks (kill switch, quota, cooldown, depth) are applied
        before the manifest is signed, so callers cannot bypass policy
        by calling ``issue_manifest`` directly instead of going through
        :meth:`Worker.maybe_replicate`.

        The *depth* parameter supplied by the caller is only used when
        ``parent_id`` is ``None`` (root workers).  For child workers the
        depth is derived from the parent's actual depth to prevent
        callers from lying about their position in the tree.
        """
        now = datetime.now(timezone.utc)
        self.can_spawn(parent_id, _now=now)

        # Derive depth from the parent's registry entry rather than
        # trusting the caller-supplied value.  This prevents depth
        # spoofing where a caller claims depth=0 while actually being
        # several levels deep.
        if parent_id:
            parent_entry = self.registry.get(parent_id)
            # can_spawn already verified parent existence, but guard
            # against race conditions defensively.
            if parent_entry is None:
                raise ReplicationDenied("Parent unknown")
            depth = parent_entry.manifest.depth + 1

        manifest = Manifest(
            worker_id=secrets.token_hex(4),
            parent_id=parent_id,
            depth=depth,
            state_snapshot=state_snapshot,
            issued_at=now,
            resources=resources,
            signature="",
        )
        self.signer.sign(manifest)
        return manifest

    def can_spawn(self, parent_id: Optional[str], _now: Optional[datetime] = None) -> None:
        if self.kill_switch_engaged:
            raise ReplicationDenied("Kill switch engaged")
        if len(self.registry) >= self.contract.max_replicas:
            self.logger.audit("deny_quota", reason="max_replicas")
            raise ReplicationDenied("Replica quota exceeded")
        if parent_id:
            last_spawn = self.spawn_timestamps.get(parent_id)
            if last_spawn:
                if _now is None:
                    _now = datetime.now(timezone.utc)
                cooldown = timedelta(seconds=self.contract.cooldown_seconds)
                if _now - last_spawn < cooldown:
                    self.logger.audit("deny_cooldown", parent_id=parent_id)
                    raise ReplicationDenied("Cooldown not satisfied")
            parent_entry = self.registry.get(parent_id)
            if not parent_entry:
                raise ReplicationDenied("Parent unknown")
            if parent_entry.manifest.depth + 1 > self.contract.max_depth:
                self.logger.audit("deny_depth", parent_id=parent_id)
                raise ReplicationDenied("Depth exceeded")

    def register_worker(self, manifest: Manifest) -> None:
        self.verify_manifest(manifest)
        context = ReplicationContext(manifest=manifest, active_count=len(self.registry), contract=self.contract)
        stop_condition = self.contract.evaluate(context)
        if stop_condition:
            self.logger.audit("deny_stop_condition", condition=stop_condition.name, worker_id=manifest.worker_id)
            raise ReplicationDenied(f"Stop condition triggered: {stop_condition.name}")

        self.registry[manifest.worker_id] = RegistryEntry(manifest=manifest, last_heartbeat=datetime.now(timezone.utc))
        if manifest.parent_id:
            self.spawn_timestamps[manifest.parent_id] = datetime.now(timezone.utc)
        self.logger.log("worker_registered", worker_id=manifest.worker_id, parent_id=manifest.parent_id, depth=manifest.depth)

    def heartbeat(self, worker_id: str) -> None:
        if worker_id in self.registry:
            self.registry[worker_id].last_heartbeat = datetime.now(timezone.utc)
            self.logger.log("heartbeat", worker_id=worker_id)
        else:
            self.logger.audit("heartbeat_unknown", worker_id=worker_id)

    def reap_stale_workers(self, timeout: timedelta, orchestrator=None) -> List[str]:
        """Remove workers whose last heartbeat exceeds the given timeout.

        Returns the list of reaped worker IDs.  Should be called
        periodically (e.g. every heartbeat interval) so that dead
        workers don't permanently consume replica-quota slots.

        When *orchestrator* is provided, the corresponding containers
        are killed before deregistration so that no orphaned resources
        remain.  Without it, only the registry entry is removed (kept
        for backward compatibility, but callers should always pass it).
        """
        now = datetime.now(timezone.utc)
        stale = [
            wid for wid, entry in self.registry.items()
            if now - entry.last_heartbeat > timeout
        ]
        for wid in stale:
            self.logger.audit(
                "reap_stale",
                worker_id=wid,
                last_heartbeat=self.registry[wid].last_heartbeat.isoformat(),
            )
            if orchestrator is not None:
                orchestrator.kill_worker(wid, reason="heartbeat_timeout")
            self.deregister(wid, reason="heartbeat_timeout")
        return stale

    def deregister(self, worker_id: str, reason: str) -> None:
        if worker_id in self.registry:
            del self.registry[worker_id]
            self.logger.log("worker_deregistered", worker_id=worker_id, reason=reason)

    def kill_switch(self, orchestrator) -> None:
        self.kill_switch_engaged = True
        active_before = len(self.registry)
        for worker_id in list(self.registry.keys()):
            orchestrator.kill_worker(worker_id, reason="kill_switch")
            self.deregister(worker_id, "kill_switch")
        self.logger.audit("kill_switch_engaged", active_before=active_before)
