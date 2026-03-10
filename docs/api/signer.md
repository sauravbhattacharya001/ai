# Signer

HMAC-SHA256 manifest signing and verification.

The `ManifestSigner` handles all cryptographic operations for the
replication system.  It is separated from the `Controller` so that the
signing strategy can be tested, swapped (e.g. asymmetric keys,
HSM-backed signing), or audited independently.

## Classes

### ManifestSigner

**Constructor:**

```python
ManifestSigner(secret: str)
```

- `secret` — HMAC key (minimum 8 characters).

Raises `ValueError` if the secret is empty, whitespace-only, or shorter
than `MIN_SECRET_LENGTH` (8).

**Class Attributes:**

| Attribute           | Value | Description                                    |
|---------------------|-------|------------------------------------------------|
| `MIN_SECRET_LENGTH` | 8     | Minimum key length to resist brute-force       |

**Methods:**

| Method               | Returns  | Description                                   |
|----------------------|----------|-----------------------------------------------|
| `sign(manifest)`     | Manifest | Compute HMAC-SHA256 and set `manifest.signature` |
| `verify(manifest)`   | bool     | Return True if signature is valid              |

## Signing Details

The signer produces a canonical string from the manifest fields:

```
worker_id:parent_id:depth:issued_at:state_json:cpu:mem:allow_ctrl:allow_ext
```

- `state_json` uses `json.dumps(sort_keys=True)` for deterministic
  serialization, preventing dict-ordering attacks.
- Comparison uses `hmac.compare_digest` (constant-time) to resist
  timing side-channels.

## Usage

```python
from replication.signer import ManifestSigner
from replication.contract import Manifest, ResourceSpec
from datetime import datetime, timezone

signer = ManifestSigner("my-secure-key-1234")
resources = ResourceSpec(cpu_limit=1.0, memory_limit_mb=512)

manifest = Manifest(
    worker_id="abc123",
    parent_id=None,
    depth=0,
    state_snapshot={"task": "scan"},
    issued_at=datetime.now(timezone.utc),
    resources=resources,
    signature="",
)

# Sign
signer.sign(manifest)
print(manifest.signature)  # 64-char hex HMAC

# Verify
assert signer.verify(manifest)

# Tampered manifest fails
manifest.depth = 5
assert not signer.verify(manifest)
```

::: replication.signer
