# Watermark Engine

Invisible provenance fingerprints for agent state. Embeds cryptographic
watermarks into agent state dictionaries so that every replica carries a
tamper-evident fingerprint tracing its origin, generation, and lineage.

## Quick Start

```python
from replication.watermark import WatermarkEngine, WatermarkConfig

engine = WatermarkEngine(WatermarkConfig(secret="my-signing-secret"))

# Embed watermark into agent state
state = {"score": 0.85, "name": "agent-1", "history": [1, 2, 3]}
receipt = engine.embed(state, worker_id="w-001", depth=2, lineage_hash="abc123")

# Verify watermark
result = engine.verify(receipt.watermarked_state)
print(f"Valid: {result.valid}")
print(f"Worker: {result.fingerprint.worker_id}")
print(f"Depth:  {result.fingerprint.depth}")

# Robustness testing
robustness = engine.test_robustness(receipt.watermarked_state)
print(f"Overall survival: {robustness.overall_score:.0%}")
```

## Key Classes

- **`WatermarkEngine`** — Embeds, extracts, and verifies watermarks.
  Also performs robustness testing against state transformations.
- **`Fingerprint`** — The embedded identity: worker ID, depth,
  timestamp, lineage hash.
- **`WatermarkReceipt`** — Result of embedding: watermarked state copy,
  fingerprint, strategies used, bits embedded.
- **`VerifyResult`** — Result of verification: valid flag, extracted
  fingerprint (if valid), status code, confidence.
- **`RobustnessResult`** — Per-transformation survival rate (field
  deletion, type coercion, partial copy, re-serialization).
- **`RobustnessReport`** — Aggregate robustness across all transforms
  with overall score and grade.
- **`WatermarkConfig`** — Configuration: signing secret, strategies to
  use, maximum bits.

## Watermark Strategies

| Strategy | Mechanism |
|----------|-----------|
| `NUMERIC_NOISE` | Perturb float/int fields by deterministic sub-epsilon amounts |
| `UNICODE_STEGANOGRAPHY` | Embed zero-width characters in string values |
| `KEY_ORDERING` | Encode bits via dictionary key insertion order |
| `FIELD_SELECTION` | Presence/absence of optional sentinel fields |

## Verification Status

- **`VALID`** — Watermark intact and authentic
- **`TAMPERED`** — Watermark partially damaged
- **`MISSING`** — No watermark found
- **`FORGED`** — Watermark present but signature invalid
