from __future__ import annotations

import hmac
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
        return (
            f"{manifest.worker_id}:{manifest.parent_id}:{manifest.depth}"
            f":{manifest.issued_at.isoformat()}:{manifest.state_snapshot}"
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
