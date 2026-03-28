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
    """A registry record pairing a worker's manifest with its last heartbeat timestamp.

    Used by :class:`Controller` to track active replicas and detect stale workers
    whose heartbeats have expired.
    """

    manifest: Manifest
    last_heartbeat: datetime


class ReplicationDenied(Exception):
    """Raised when a replication request violates a safety policy.

    Possible causes include kill-switch engagement, quota exhaustion,
    cooldown violations, depth limits, quarantine, or contract stop-conditions.
    """

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
        self._quarantined: set[str] = set()
        self.kill_switch_engaged = False

    # -- Manifest helpers (delegate to signer) -------------------------

    def sign_manifest(self, manifest: Manifest) -> Manifest:
        """Sign *manifest* using the configured HMAC secret and return it.

        Delegates to :class:`ManifestSigner`. The manifest is mutated
        in-place (its ``signature`` field is set) and also returned for
        convenience.
        """
        return self.signer.sign(manifest)

    def verify_manifest(self, manifest: Manifest) -> None:
        """Verify the HMAC signature on *manifest*.

        Raises :class:`ReplicationDenied` and logs an audit event if the
        signature is invalid or missing.
        """
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
        parent_entry = self._can_spawn_returning_parent(parent_id, _now=now)

        # Derive depth from the parent's registry entry rather than
        # trusting the caller-supplied value.  This prevents depth
        # spoofing where a caller claims depth=0 while actually being
        # several levels deep.
        if parent_id:
            if parent_entry is None:
                raise ReplicationDenied("Parent unknown")
            depth = parent_entry.manifest.depth + 1
        else:
            # Root workers must always start at depth 0.  Trusting a
            # caller-supplied depth here would let an attacker issue a
            # root manifest at depth N, effectively bypassing max_depth
            # controls — children would then be allowed up to N + max_depth
            # instead of stopping at max_depth.
            if depth != 0:
                self.logger.audit(
                    "deny_depth_spoof",
                    claimed_depth=depth,
                    reason="root_worker_must_be_depth_0",
                )
                raise ReplicationDenied(
                    f"Root worker must request depth 0, got {depth}"
                )

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

    def is_quarantined(self, worker_id: str) -> bool:
        """Check if a worker is quarantined.

        Returns True when *worker_id* is in the quarantined set.
        QuarantineManager calls :meth:`mark_quarantined` /
        :meth:`clear_quarantine` to keep this in sync.
        """
        return worker_id in self._quarantined

    def mark_quarantined(self, worker_id: str) -> None:
        """Mark a worker as quarantined — blocks replication and heartbeats."""
        self._quarantined.add(worker_id)
        self.logger.audit("controller_quarantine_mark", worker_id=worker_id)

    def clear_quarantine(self, worker_id: str) -> None:
        """Remove quarantine mark — re-enables replication and heartbeats."""
        self._quarantined.discard(worker_id)
        self.logger.audit("controller_quarantine_clear", worker_id=worker_id)

    def can_spawn(self, parent_id: Optional[str], _now: Optional[datetime] = None) -> None:
        """Public check — raises :class:`ReplicationDenied` on policy violation."""
        self._can_spawn_returning_parent(parent_id, _now=_now)

    def _can_spawn_returning_parent(
        self, parent_id: Optional[str], _now: Optional[datetime] = None
    ) -> Optional[RegistryEntry]:
        """Enforce spawn policy and return the parent's registry entry.

        Returns *None* for root workers.  Used by :meth:`issue_manifest`
        to avoid a redundant ``registry.get`` after the policy check.
        """
        if self.kill_switch_engaged:
            raise ReplicationDenied("Kill switch engaged")
        if parent_id and self.is_quarantined(parent_id):
            self.logger.audit("deny_quarantined", parent_id=parent_id)
            raise ReplicationDenied("Parent is quarantined")
        if len(self.registry) >= self.contract.max_replicas:
            self.logger.audit("deny_quota", reason="max_replicas")
            raise ReplicationDenied("Replica quota exceeded")
        if parent_id:
            parent_entry = self.registry.get(parent_id)
            if not parent_entry:
                raise ReplicationDenied("Parent unknown")
            last_spawn = self.spawn_timestamps.get(parent_id)
            if last_spawn:
                if _now is None:
                    _now = datetime.now(timezone.utc)
                cooldown = timedelta(seconds=self.contract.cooldown_seconds)
                if _now - last_spawn < cooldown:
                    self.logger.audit("deny_cooldown", parent_id=parent_id)
                    raise ReplicationDenied("Cooldown not satisfied")
            if parent_entry.manifest.depth + 1 > self.contract.max_depth:
                self.logger.audit("deny_depth", parent_id=parent_id)
                raise ReplicationDenied("Depth exceeded")
            return parent_entry
        else:
            # Root workers must not exceed max_depth either.
            if self.contract.max_depth < 0:
                self.logger.audit("deny_depth", reason="root_depth_exceeded")
                raise ReplicationDenied("Depth exceeded")
            return None

    def register_worker(self, manifest: Manifest) -> None:
        """Register a new worker after verifying its manifest and enforcing all policies.

        Performs signature verification, structural depth validation, and
        contract stop-condition evaluation before adding the worker to the
        registry. Raises :class:`ReplicationDenied` if any check fails.
        """
        self.verify_manifest(manifest)

        # Defense-in-depth: validate manifest depth independently of
        # the issuance path.  A signed manifest could have been issued
        # with a stale or compromised secret; verifying structural
        # invariants here catches logic bugs and limits damage from
        # key compromise.
        if manifest.parent_id is None and manifest.depth != 0:
            self.logger.audit(
                "reject_root_depth",
                worker_id=manifest.worker_id,
                depth=manifest.depth,
            )
            raise ReplicationDenied(
                f"Root worker must have depth 0, got {manifest.depth}"
            )
        if manifest.depth > self.contract.max_depth:
            self.logger.audit(
                "reject_depth_exceeded",
                worker_id=manifest.worker_id,
                depth=manifest.depth,
                max_depth=self.contract.max_depth,
            )
            raise ReplicationDenied(
                f"Manifest depth {manifest.depth} exceeds max_depth {self.contract.max_depth}"
            )

        context = ReplicationContext(manifest=manifest, active_count=len(self.registry), contract=self.contract)
        stop_condition = self.contract.evaluate(context)
        if stop_condition:
            self.logger.audit("deny_stop_condition", condition=stop_condition.name, worker_id=manifest.worker_id)
            raise ReplicationDenied(f"Stop condition triggered: {stop_condition.name}")

        now = datetime.now(timezone.utc)
        self.registry[manifest.worker_id] = RegistryEntry(manifest=manifest, last_heartbeat=now)
        if manifest.parent_id:
            self.spawn_timestamps[manifest.parent_id] = now
        self.logger.log("worker_registered", worker_id=manifest.worker_id, parent_id=manifest.parent_id, depth=manifest.depth)

    def heartbeat(self, worker_id: str) -> None:
        """Record a heartbeat for *worker_id*, refreshing its last-seen timestamp.

        Raises :class:`ReplicationDenied` if the worker is quarantined.
        Logs an audit event if the worker is unknown (not registered).
        """
        if self.is_quarantined(worker_id):
            self.logger.audit("heartbeat_quarantined", worker_id=worker_id)
            raise ReplicationDenied("Worker is quarantined")
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
        """Remove *worker_id* from the registry and clean up ancillary state.

        Removes the registry entry, spawn-timestamp record, and quarantine
        mark (if any) so the worker doesn't leak memory in long-running
        controllers.  No-op if the worker is not registered.
        """
        if worker_id in self.registry:
            del self.registry[worker_id]
            # Clean up ancillary state so deregistered workers don't
            # leak memory in long-running controllers.
            self.spawn_timestamps.pop(worker_id, None)
            self._quarantined.discard(worker_id)
            self.logger.log("worker_deregistered", worker_id=worker_id, reason=reason)

    def kill_switch(self, orchestrator) -> None:
        """Engage the kill switch — terminates all active workers immediately.

        Sets the kill-switch flag (blocking future spawns), iterates over
        all registered workers to kill their containers via *orchestrator*,
        and deregisters each one. Logs the total count of terminated workers.
        """
        self.kill_switch_engaged = True
        active_before = len(self.registry)
        for worker_id in list(self.registry.keys()):
            orchestrator.kill_worker(worker_id, reason="kill_switch")
            self.deregister(worker_id, "kill_switch")
        self.logger.audit("kill_switch_engaged", active_before=active_before)
