"""Tests for the ManifestSigner extracted from Controller."""

from datetime import datetime, timezone

from replication.contract import Manifest, ResourceSpec
from replication.signer import ManifestSigner


def _make_manifest(**overrides) -> Manifest:
    defaults = dict(
        worker_id="abc123",
        parent_id=None,
        depth=0,
        state_snapshot={"task": "test"},
        issued_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        resources=ResourceSpec(cpu_limit=0.5, memory_limit_mb=256),
        signature="",
    )
    defaults.update(overrides)
    return Manifest(**defaults)


def test_sign_and_verify_roundtrip():
    signer = ManifestSigner("secret-key")
    manifest = _make_manifest()
    signer.sign(manifest)

    assert manifest.signature != ""
    assert signer.verify(manifest)


def test_verify_rejects_tampered_manifest():
    signer = ManifestSigner("secret-key")
    manifest = _make_manifest()
    signer.sign(manifest)

    manifest.depth = 99  # tamper
    assert not signer.verify(manifest)


def test_different_keys_produce_different_signatures():
    m1 = _make_manifest()
    m2 = _make_manifest()

    ManifestSigner("key-a").sign(m1)
    ManifestSigner("key-b").sign(m2)

    assert m1.signature != m2.signature


def test_verify_fails_with_wrong_key():
    signer_a = ManifestSigner("key-a")
    signer_b = ManifestSigner("key-b")

    manifest = _make_manifest()
    signer_a.sign(manifest)

    assert not signer_b.verify(manifest)
