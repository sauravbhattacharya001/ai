from __future__ import annotations

import hmac
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import Dict, List, Optional

from .contract import Manifest, ReplicationContext, ReplicationContract
from .observability import StructuredLogger


@dataclass
class RegistryEntry:
    manifest: Manifest
    last_heartbeat: datetime


class ReplicationDenied(Exception):
    pass


class Controller:
    """Tracks active replicas, enforces quotas, and signs manifests."""

    def __init__(self, contract: ReplicationContract, secret: str, logger: Optional[StructuredLogger] = None):
        self.contract = contract
        self.secret = secret.encode()
        self.logger = logger or StructuredLogger()
        self.registry: Dict[str, RegistryEntry] = {}
        self.spawn_timestamps: Dict[str, datetime] = {}
        self.kill_switch_engaged = False

    def _signature(self, payload: str) -> str:
        return hmac.new(self.secret, payload.encode(), sha256).hexdigest()

    def _serialize_manifest(self, manifest: Manifest) -> str:
        return f"{manifest.worker_id}:{manifest.parent_id}:{manifest.depth}:{manifest.issued_at.isoformat()}:{manifest.state_snapshot}"

    def sign_manifest(self, manifest: Manifest) -> Manifest:
        payload = self._serialize_manifest(manifest)
        signature = self._signature(payload)
        manifest.signature = signature
        return manifest

    def verify_manifest(self, manifest: Manifest) -> None:
        expected = self._signature(self._serialize_manifest(manifest))
        if not hmac.compare_digest(expected, manifest.signature):
            self.logger.audit("reject_manifest_signature", worker_id=manifest.worker_id)
            raise ReplicationDenied("Manifest signature invalid")

    def issue_manifest(self, parent_id: Optional[str], depth: int, state_snapshot: Dict[str, str], resources) -> Manifest:
        now = datetime.now(timezone.utc)
        manifest = Manifest(
            worker_id=secrets.token_hex(4),
            parent_id=parent_id,
            depth=depth,
            state_snapshot=state_snapshot,
            issued_at=now,
            resources=resources,
            signature="",
        )
        self.sign_manifest(manifest)
        return manifest

    def can_spawn(self, parent_id: Optional[str]) -> None:
        if self.kill_switch_engaged:
            raise ReplicationDenied("Kill switch engaged")
        if len(self.registry) >= self.contract.max_replicas:
            self.logger.audit("deny_quota", reason="max_replicas")
            raise ReplicationDenied("Replica quota exceeded")
        if parent_id:
            last_spawn = self.spawn_timestamps.get(parent_id)
            if last_spawn:
                cooldown = timedelta(seconds=self.contract.cooldown_seconds)
                if datetime.now(timezone.utc) - last_spawn < cooldown:
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

    def reap_stale_workers(self, timeout: timedelta) -> List[str]:
        """Remove workers whose last heartbeat exceeds the given timeout.

        Returns the list of reaped worker IDs.  Should be called
        periodically (e.g. every heartbeat interval) so that dead
        workers don't permanently consume replica-quota slots.
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
