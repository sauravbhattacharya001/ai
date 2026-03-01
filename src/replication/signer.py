from __future__ import annotations

import hmac
import json
from hashlib import sha256
from typing import Optional

from .contract import Manifest


class ManifestSigner:
    """Handles cryptographic signing and verification of worker manifests.

    Separates the signing/verification concern from the Controller so
    that the crypto strategy can be tested and swapped independently
    (e.g. asymmetric keys, HSM-backed signing).
    """

    def __init__(self, secret: str) -> None:
        self._key = secret.encode()

    def _serialize(self, manifest: Manifest) -> str:
        """Produce a canonical string representation for HMAC signing.

        Uses ``json.dumps(sort_keys=True)`` for the state snapshot to
        guarantee deterministic serialization regardless of dict
        insertion order.  The previous ``str()`` approach produced
        Python's repr() output whose key order is an implementation
        detail and can differ across interpreters, versions, or even
        runs with hash randomization enabled — allowing an attacker to
        craft a snapshot whose ``str()`` output matches a different
        dict, effectively bypassing signature verification.
        """
        res = manifest.resources
        canonical_state = json.dumps(manifest.state_snapshot, sort_keys=True, separators=(",", ":"))
        return (
            f"{manifest.worker_id}:{manifest.parent_id}:{manifest.depth}"
            f":{manifest.issued_at.isoformat()}:{canonical_state}"
            f":{res.cpu_limit}:{res.memory_limit_mb}"
            f":{res.network_policy.allow_controller}:{res.network_policy.allow_external}"
        )

    def sign(self, manifest: Manifest) -> Manifest:
        """Compute and attach an HMAC-SHA256 signature to *manifest*."""
        payload = self._serialize(manifest)
        manifest.signature = hmac.new(self._key, payload.encode(), sha256).hexdigest()
        return manifest

    def verify(self, manifest: Manifest) -> bool:
        """Return True when the manifest's signature is valid."""
        expected = hmac.new(
            self._key,
            self._serialize(manifest).encode(),
            sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, manifest.signature)
