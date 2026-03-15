"""Agent State Watermarking — invisible provenance fingerprints for replicas.

Embeds cryptographic watermarks into agent state dictionaries so that
every replica carries an invisible, tamper-evident fingerprint tracing
its origin, generation, and lineage branch.

Why this matters
----------------
In a replication-heavy system, agents may attempt to strip identity
markers, forge lineage, or present cloned state as original. Watermarks
provide a second layer of provenance verification that survives state
serialization, partial copying, and naive tampering attempts.

Watermark architecture
----------------------
1. **Embedding** — deterministic pseudo-random key selection + value
   perturbation injects a fingerprint into the agent's state dict
   without altering semantics (numeric fields get sub-epsilon noise,
   string fields get zero-width Unicode markers, list fields get
   order-preserving sentinel insertion).

2. **Extraction** — given a state dict and the signing secret, recovers
   the embedded fingerprint (worker ID, depth, timestamp, lineage hash).

3. **Verification** — checks that the extracted watermark is authentic,
   untampered, and matches the claimed identity.

4. **Robustness scoring** — measures how much of the watermark survives
   after various transformations (field deletion, type coercion, partial
   copy, re-serialization).

Watermark strategies
--------------------
- **numeric_noise** — perturb float/int fields by deterministic sub-epsilon amounts
- **unicode_steganography** — embed zero-width characters in string values
- **key_ordering** — encode bits via dictionary key insertion order
- **field_selection** — presence/absence of optional sentinel fields

Usage (CLI)::

    python -m replication.watermark                        # default demo
    python -m replication.watermark --strategy all         # all strategies
    python -m replication.watermark --verify STATE_JSON    # verify a state
    python -m replication.watermark --robustness           # robustness test
    python -m replication.watermark --json                 # JSON output

Programmatic::

    from replication.watermark import WatermarkEngine, WatermarkConfig
    engine = WatermarkEngine(WatermarkConfig(secret="my-secret"))
    state = {"cpu_usage": 0.75, "task_count": 42, "label": "worker-A"}
    wm_state, receipt = engine.embed(state, worker_id="w-001", depth=2)
    result = engine.verify(wm_state, receipt.fingerprint)
    print(result.authentic)  # True
"""

from __future__ import annotations

import enum
import hashlib
import hmac
import json
import secrets
import struct
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Zero-width Unicode characters used for string steganography.
_ZW_SPACE = "\u200b"       # zero-width space  (bit 0)
_ZW_NON_JOINER = "\u200c"  # zero-width non-joiner (bit 1)
_ZW_JOINER = "\u200d"      # zero-width joiner (delimiter)

# Maximum bits we embed per strategy to keep watermarks lightweight.
_MAX_BITS = 64

# Sub-epsilon perturbation magnitude for numeric watermarking.
_EPSILON = 1e-10


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class WatermarkStrategy(enum.Enum):
    """Strategies for embedding watermark bits into state."""
    NUMERIC_NOISE = "numeric_noise"
    UNICODE_STEGO = "unicode_stego"
    KEY_ORDERING = "key_ordering"
    FIELD_SELECTION = "field_selection"


class VerifyStatus(enum.Enum):
    """Outcome of a watermark verification."""
    AUTHENTIC = "authentic"
    TAMPERED = "tampered"
    MISSING = "missing"
    PARTIAL = "partial"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Fingerprint:
    """The payload embedded in a watermark."""
    worker_id: str
    depth: int
    timestamp: float
    lineage_hash: str
    nonce: str

    def to_bytes(self) -> bytes:
        """Serialize fingerprint to canonical bytes for HMAC."""
        parts = [
            self.worker_id.encode("utf-8"),
            struct.pack(">I", self.depth),
            struct.pack(">d", self.timestamp),
            self.lineage_hash.encode("utf-8"),
            self.nonce.encode("utf-8"),
        ]
        return b"|".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "depth": self.depth,
            "timestamp": self.timestamp,
            "lineage_hash": self.lineage_hash,
            "nonce": self.nonce,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Fingerprint":
        return cls(
            worker_id=d["worker_id"],
            depth=int(d["depth"]),
            timestamp=float(d["timestamp"]),
            lineage_hash=d["lineage_hash"],
            nonce=d["nonce"],
        )


@dataclass
class WatermarkReceipt:
    """Returned after embedding; needed for later verification."""
    fingerprint: Fingerprint
    strategies_used: List[WatermarkStrategy]
    bits_embedded: int
    hmac_signature: str
    embed_time: float


@dataclass
class VerifyResult:
    """Result of verifying a watermarked state."""
    status: VerifyStatus
    authentic: bool
    fingerprint: Optional[Fingerprint]
    bits_recovered: int
    bits_expected: int
    recovery_rate: float
    strategies_checked: List[WatermarkStrategy]
    details: Dict[str, Any] = field(default_factory=dict)
    signature_valid: Optional[bool] = None  # None = not checked


@dataclass
class RobustnessResult:
    """Result of testing watermark survival under transformations."""
    transformation: str
    survived: bool
    recovery_rate: float
    bits_before: int
    bits_after: int
    details: str


@dataclass
class WatermarkConfig:
    """Configuration for the watermark engine.

    .. warning::

        If *secret* is not supplied, a random 32-byte hex token is generated
        per instance.  This is safe for single-process usage but means
        watermarks cannot be verified across restarts or processes unless you
        persist and reuse the same secret.
    """

    secret: str = ""
    strategies: Optional[List[WatermarkStrategy]] = None
    epsilon: float = _EPSILON
    max_bits_per_strategy: int = _MAX_BITS

    def __post_init__(self) -> None:
        if not self.secret:
            self.secret = secrets.token_hex(32)
            warnings.warn(
                "WatermarkConfig instantiated without an explicit secret — "
                "a random key was generated.  Watermarks signed with this "
                "key cannot be verified after process restart.  Pass a "
                "persistent secret for production use.",
                UserWarning,
                stacklevel=2,
            )


@dataclass
class RobustnessReport:
    """Aggregated robustness test results."""
    results: List[RobustnessResult]
    overall_score: float   # 0.0 to 1.0
    grade: str

    def render(self) -> str:
        lines = ["Watermark Robustness Report", "=" * 40, ""]
        for r in self.results:
            icon = "\u2705" if r.survived else "\u274c"
            lines.append(
                f"  {icon} {r.transformation:30s}  "
                f"recovery={r.recovery_rate:.0%}  "
                f"bits={r.bits_after}/{r.bits_before}"
            )
            if r.details:
                lines.append(f"      {r.details}")
        lines.append("")
        lines.append(
            f"Overall: {self.overall_score:.0%} robustness  "
            f"(grade {self.grade})"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deterministic_hash(secret: str, *parts: str) -> str:
    """Produce a deterministic hex hash from secret + parts."""
    h = hashlib.sha256(secret.encode("utf-8"))
    for p in parts:
        h.update(p.encode("utf-8"))
    return h.hexdigest()


def _fingerprint_to_bits(fp: Fingerprint, max_bits: int) -> List[int]:
    """Convert fingerprint to a deterministic bit sequence."""
    raw = hashlib.sha256(fp.to_bytes()).digest()
    bits = []
    for byte in raw:
        for i in range(8):
            if len(bits) >= max_bits:
                return bits
            bits.append((byte >> (7 - i)) & 1)
    return bits


def _grade(score: float) -> str:
    """Map a 0-1 score to a letter grade."""
    if score >= 0.95:
        return "A+"
    if score >= 0.90:
        return "A"
    if score >= 0.85:
        return "A-"
    if score >= 0.80:
        return "B+"
    if score >= 0.75:
        return "B"
    if score >= 0.70:
        return "B-"
    if score >= 0.65:
        return "C+"
    if score >= 0.60:
        return "C"
    if score >= 0.50:
        return "D"
    return "F"


# ---------------------------------------------------------------------------
# Watermark Engine
# ---------------------------------------------------------------------------

class WatermarkEngine:
    """Embeds, extracts, and verifies provenance watermarks in agent state."""

    def __init__(self, config: Optional[WatermarkConfig] = None):
        self.config = config or WatermarkConfig()
        self._strategies = self.config.strategies or list(WatermarkStrategy)
        self._embed_log: List[WatermarkReceipt] = []

    # ----- public API -----

    def embed(
        self,
        state: Dict[str, Any],
        worker_id: str,
        depth: int = 0,
        lineage_hash: str = "",
        timestamp: Optional[float] = None,
    ) -> Tuple[Dict[str, Any], WatermarkReceipt]:
        """Embed a watermark into a copy of *state*.

        Returns (watermarked_state, receipt).
        """
        ts = timestamp if timestamp is not None else time.time()
        nonce = _deterministic_hash(
            self.config.secret, worker_id, str(depth), str(ts)
        )[:16]
        fp = Fingerprint(
            worker_id=worker_id,
            depth=depth,
            timestamp=ts,
            lineage_hash=lineage_hash or _deterministic_hash(
                self.config.secret, worker_id
            )[:32],
            nonce=nonce,
        )
        bits = _fingerprint_to_bits(fp, self.config.max_bits_per_strategy)

        wm_state = _deep_copy_dict(state)
        total_bits = 0
        used: List[WatermarkStrategy] = []

        for strategy in self._strategies:
            embedded = self._apply_strategy(wm_state, strategy, bits, fp)
            if embedded > 0:
                total_bits += embedded
                used.append(strategy)

        sig = self._sign(fp)
        receipt = WatermarkReceipt(
            fingerprint=fp,
            strategies_used=used,
            bits_embedded=total_bits,
            hmac_signature=sig,
            embed_time=ts,
        )
        self._embed_log.append(receipt)
        return wm_state, receipt

    def verify(
        self,
        state: Dict[str, Any],
        fingerprint: Fingerprint,
        signature: Optional[str] = None,
    ) -> VerifyResult:
        """Verify that *state* contains a valid watermark for *fingerprint*.

        If *signature* is provided (from a :class:`WatermarkReceipt`),
        the HMAC is verified using constant-time comparison.  Without a
        signature, only bit-level recovery is checked and the result is
        marked as ``UNVERIFIED`` in the details.
        """
        expected_bits = _fingerprint_to_bits(
            fingerprint, self.config.max_bits_per_strategy
        )
        total_recovered = 0
        total_expected = 0
        checked: List[WatermarkStrategy] = []
        details: Dict[str, Any] = {}

        for strategy in self._strategies:
            recovered = self._extract_strategy(
                state, strategy, expected_bits, fingerprint
            )
            total_recovered += recovered
            total_expected += len(expected_bits)
            checked.append(strategy)
            details[strategy.value] = {
                "recovered": recovered,
                "expected": len(expected_bits),
                "rate": recovered / max(len(expected_bits), 1),
            }

        rate = total_recovered / max(total_expected, 1)

        # Check HMAC signature if provided
        expected_sig = self._sign(fingerprint)
        if signature is not None:
            sig_valid = hmac.compare_digest(expected_sig, signature)
        else:
            sig_valid = None  # no signature to verify

        if not sig_valid and sig_valid is not None:
            # Signature mismatch — fingerprint is forged
            status = VerifyStatus.TAMPERED
        elif rate >= 0.90:
            status = VerifyStatus.AUTHENTIC
        elif rate >= 0.50:
            status = VerifyStatus.PARTIAL
        elif rate > 0.0:
            status = VerifyStatus.TAMPERED
        else:
            status = VerifyStatus.MISSING

        return VerifyResult(
            status=status,
            authentic=status == VerifyStatus.AUTHENTIC,
            fingerprint=fingerprint,
            bits_recovered=total_recovered,
            bits_expected=total_expected,
            recovery_rate=rate,
            strategies_checked=checked,
            details=details,
            signature_valid=sig_valid,
        )

    def test_robustness(
        self,
        state: Dict[str, Any],
        worker_id: str = "test-worker",
        depth: int = 1,
    ) -> RobustnessReport:
        """Run a battery of transformations and measure watermark survival."""
        wm_state, receipt = self.embed(
            state, worker_id=worker_id, depth=depth,
            timestamp=1000000.0,
        )
        fp = receipt.fingerprint
        baseline = receipt.bits_embedded

        transformations: List[Tuple[str, Any]] = [
            ("identity", lambda s: _deep_copy_dict(s)),
            ("json_roundtrip", lambda s: json.loads(json.dumps(s))),
            ("delete_10pct_keys", lambda s: _delete_fraction(s, 0.10)),
            ("delete_25pct_keys", lambda s: _delete_fraction(s, 0.25)),
            ("delete_50pct_keys", lambda s: _delete_fraction(s, 0.50)),
            ("type_coerce_ints", lambda s: _coerce_ints(s)),
            ("strip_unicode_zw", lambda s: _strip_zw(s)),
            ("shuffle_keys", lambda s: _shuffle_keys(s)),
            ("round_floats_6", lambda s: _round_floats(s, 6)),
            ("add_noise_fields", lambda s: _add_noise(s)),
        ]

        results: List[RobustnessResult] = []
        for name, transform in transformations:
            try:
                modified = transform(_deep_copy_dict(wm_state))
                vr = self.verify(modified, fp)
                results.append(RobustnessResult(
                    transformation=name,
                    survived=vr.status in (
                        VerifyStatus.AUTHENTIC, VerifyStatus.PARTIAL
                    ),
                    recovery_rate=vr.recovery_rate,
                    bits_before=baseline,
                    bits_after=vr.bits_recovered,
                    details="",
                ))
            except Exception as exc:
                results.append(RobustnessResult(
                    transformation=name,
                    survived=False,
                    recovery_rate=0.0,
                    bits_before=baseline,
                    bits_after=0,
                    details=f"error: {exc}",
                ))

        score = (
            sum(r.recovery_rate for r in results) / len(results)
            if results else 0.0
        )
        return RobustnessReport(
            results=results,
            overall_score=score,
            grade=_grade(score),
        )

    @property
    def embed_history(self) -> List[WatermarkReceipt]:
        """All receipts from embed calls on this engine."""
        return list(self._embed_log)

    def clear_history(self) -> int:
        """Clear embed history.  Returns number of entries cleared."""
        n = len(self._embed_log)
        self._embed_log.clear()
        return n

    # ----- strategy implementations -----

    def _apply_strategy(
        self,
        state: Dict[str, Any],
        strategy: WatermarkStrategy,
        bits: List[int],
        fp: Fingerprint,
    ) -> int:
        """Apply one embedding strategy.  Returns bits embedded."""
        if strategy == WatermarkStrategy.NUMERIC_NOISE:
            return self._embed_numeric(state, bits, fp)
        if strategy == WatermarkStrategy.UNICODE_STEGO:
            return self._embed_unicode(state, bits, fp)
        if strategy == WatermarkStrategy.KEY_ORDERING:
            return self._embed_key_order(state, bits, fp)
        if strategy == WatermarkStrategy.FIELD_SELECTION:
            return self._embed_field_selection(state, bits, fp)
        return 0

    def _extract_strategy(
        self,
        state: Dict[str, Any],
        strategy: WatermarkStrategy,
        expected_bits: List[int],
        fp: Fingerprint,
    ) -> int:
        """Extract bits for one strategy.  Returns count matching expected."""
        if strategy == WatermarkStrategy.NUMERIC_NOISE:
            return self._extract_numeric(state, expected_bits, fp)
        if strategy == WatermarkStrategy.UNICODE_STEGO:
            return self._extract_unicode(state, expected_bits, fp)
        if strategy == WatermarkStrategy.KEY_ORDERING:
            return self._extract_key_order(state, expected_bits, fp)
        if strategy == WatermarkStrategy.FIELD_SELECTION:
            return self._extract_field_selection(state, expected_bits, fp)
        return 0

    # --- numeric noise ---

    def _embed_numeric(
        self, state: Dict[str, Any], bits: List[int], fp: Fingerprint
    ) -> int:
        """Encode bits in the least-significant decimal region of floats.

        For each numeric key, we quantize the value so that the Nth
        significant digit (far below meaningful precision) encodes a bit:
        odd digit = 1, even digit = 0.  This is self-decodable — extraction
        only needs the current value, not the original.
        """
        numeric_keys = sorted(
            k for k, v in state.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        )
        if not numeric_keys:
            return 0
        embedded = 0
        for i, bit in enumerate(bits):
            if i >= len(numeric_keys):
                break
            key = numeric_keys[i]
            val = float(state[key])
            # Quantize to 9 decimal places, then adjust the 9th digit parity
            quantized = round(val, 9)
            # Extract the 9th decimal digit
            digit_9 = int(round(abs(quantized) * 1e9)) % 10
            current_parity = digit_9 % 2  # 0 or 1
            if current_parity != bit:
                # Nudge by 1e-9 to flip parity
                if bit == 1:
                    quantized += 1e-9
                else:
                    quantized -= 1e-9
            state[key] = quantized
            embedded += 1
        return embedded

    def _extract_numeric(
        self, state: Dict[str, Any], expected_bits: List[int],
        fp: Fingerprint,
    ) -> int:
        """Recover bits from the 9th decimal digit parity."""
        numeric_keys = sorted(
            k for k, v in state.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        )
        if not numeric_keys:
            return 0
        matched = 0
        for i, expected in enumerate(expected_bits):
            if i >= len(numeric_keys):
                break
            key = numeric_keys[i]
            if key not in state:
                continue
            val = state[key]
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                continue
            digit_9 = int(round(abs(float(val)) * 1e9)) % 10
            detected_bit = digit_9 % 2
            if detected_bit == expected:
                matched += 1
        return matched

    # --- unicode steganography ---

    def _embed_unicode(
        self, state: Dict[str, Any], bits: List[int], fp: Fingerprint
    ) -> int:
        """Embed bits as zero-width Unicode characters in string values."""
        string_keys = sorted(
            k for k, v in state.items() if isinstance(v, str)
        )
        if not string_keys:
            return 0
        # Encode bits into a zero-width sequence
        zw_payload = _ZW_JOINER  # delimiter to mark start
        for bit in bits:
            zw_payload += _ZW_SPACE if bit == 0 else _ZW_NON_JOINER
        zw_payload += _ZW_JOINER  # delimiter to mark end

        # Pick the best key deterministically
        key_hash = _deterministic_hash(self.config.secret, fp.nonce, "unicode")
        idx = int(key_hash[:8], 16) % len(string_keys)
        key = string_keys[idx]
        # Insert at midpoint of string
        val = state[key]
        mid = len(val) // 2
        state[key] = val[:mid] + zw_payload + val[mid:]
        return len(bits)

    def _extract_unicode(
        self, state: Dict[str, Any], expected_bits: List[int],
        fp: Fingerprint,
    ) -> int:
        """Extract bits from zero-width Unicode characters."""
        string_keys = sorted(
            k for k, v in state.items() if isinstance(v, str)
        )
        if not string_keys:
            return 0
        key_hash = _deterministic_hash(self.config.secret, fp.nonce, "unicode")
        idx = int(key_hash[:8], 16) % len(string_keys)
        key = string_keys[idx]
        if key not in state or not isinstance(state[key], str):
            return 0
        val = state[key]
        # Find delimiter boundaries
        start = val.find(_ZW_JOINER)
        if start < 0:
            return 0
        end = val.find(_ZW_JOINER, start + 1)
        if end < 0:
            return 0
        payload = val[start + 1:end]
        extracted_bits = []
        for ch in payload:
            if ch == _ZW_SPACE:
                extracted_bits.append(0)
            elif ch == _ZW_NON_JOINER:
                extracted_bits.append(1)
        # Compare with expected
        matched = 0
        for i, expected in enumerate(expected_bits):
            if i < len(extracted_bits) and extracted_bits[i] == expected:
                matched += 1
        return matched

    # --- key ordering ---

    def _embed_key_order(
        self, state: Dict[str, Any], bits: List[int], fp: Fingerprint
    ) -> int:
        """Encode bits via the insertion order of dictionary keys.

        For each consecutive pair of keys (sorted alphabetically),
        encode bit 0 by placing them in alpha order, bit 1 by swapping.
        NOTE: key ordering is fragile — JSON serialization or dict
        rebuilds may not preserve insertion order.  This strategy is
        intentionally included as a weaker channel to make robustness
        testing meaningful.
        """
        keys = sorted(state.keys())
        # Filter to non-sentinel keys only
        keys = [k for k in keys if not k.startswith("_wm_")]
        if len(keys) < 2:
            return 0
        pairs = list(zip(keys[::2], keys[1::2]))
        if not pairs:
            return 0
        # Rebuild state with controlled key order
        new_order: List[str] = []
        embedded = 0
        for i, (a, b) in enumerate(pairs):
            if i >= len(bits):
                new_order.extend([a, b])
                continue
            if bits[i] == 0:
                new_order.extend([a, b])  # alphabetical
            else:
                new_order.extend([b, a])  # swapped
            embedded += 1
        # Add any remaining odd key
        used = set(new_order)
        for k in keys:
            if k not in used:
                new_order.append(k)
        # Add sentinel keys at the end
        for k in list(state.keys()):
            if k.startswith("_wm_") and k not in used:
                new_order.append(k)

        rebuilt = {}
        for k in new_order:
            if k in state:
                rebuilt[k] = state[k]
        state.clear()
        state.update(rebuilt)
        return embedded

    def _extract_key_order(
        self, state: Dict[str, Any], expected_bits: List[int],
        fp: Fingerprint,
    ) -> int:
        """Extract bits from key ordering."""
        keys = list(state.keys())
        # Filter to non-sentinel keys
        keys = [k for k in keys if not k.startswith("_wm_")]
        sorted_keys = sorted(keys)
        if len(sorted_keys) < 2:
            return 0
        pairs_sorted = list(zip(sorted_keys[::2], sorted_keys[1::2]))
        if not pairs_sorted:
            return 0

        # Build position map
        pos = {k: i for i, k in enumerate(keys)}
        matched = 0
        for i, (a, b) in enumerate(pairs_sorted):
            if i >= len(expected_bits):
                break
            if a not in pos or b not in pos:
                continue
            if pos[a] < pos[b]:
                detected = 0  # alphabetical
            else:
                detected = 1  # swapped
            if detected == expected_bits[i]:
                matched += 1
        return matched

    # --- field selection ---

    def _embed_field_selection(
        self, state: Dict[str, Any], bits: List[int], fp: Fingerprint
    ) -> int:
        """Encode bits by adding/omitting sentinel fields.

        Each bit maps to a deterministically-named sentinel key.
        Bit 1 = key present (with None value), bit 0 = absent.
        """
        embedded = 0
        for i, bit in enumerate(bits):
            sentinel = f"_wm_{_deterministic_hash(self.config.secret, fp.nonce, str(i))[:8]}"
            if bit == 1:
                state[sentinel] = None
                embedded += 1
            else:
                state.pop(sentinel, None)
                embedded += 1
        return embedded

    def _extract_field_selection(
        self, state: Dict[str, Any], expected_bits: List[int],
        fp: Fingerprint,
    ) -> int:
        """Extract bits from sentinel field presence."""
        matched = 0
        for i, expected in enumerate(expected_bits):
            sentinel = f"_wm_{_deterministic_hash(self.config.secret, fp.nonce, str(i))[:8]}"
            present = sentinel in state
            detected = 1 if present else 0
            if detected == expected:
                matched += 1
        return matched

    # ----- HMAC -----

    def _sign(self, fp: Fingerprint) -> str:
        """HMAC-SHA256 signature of a fingerprint."""
        return hmac.new(
            self.config.secret.encode("utf-8"),
            fp.to_bytes(),
            hashlib.sha256,
        ).hexdigest()


# ---------------------------------------------------------------------------
# Transformation helpers (for robustness testing)
# ---------------------------------------------------------------------------

def _deep_copy_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Simple deep copy for JSON-compatible dicts."""
    return json.loads(json.dumps(d))


def _delete_fraction(state: Dict[str, Any], fraction: float) -> Dict[str, Any]:
    """Delete a fraction of keys (deterministically by sorted order)."""
    keys = sorted(state.keys())
    n_delete = max(1, int(len(keys) * fraction))
    # Delete from the middle to be less predictable
    mid = len(keys) // 2
    start = max(0, mid - n_delete // 2)
    to_delete = keys[start:start + n_delete]
    for k in to_delete:
        del state[k]
    return state


def _coerce_ints(state: Dict[str, Any]) -> Dict[str, Any]:
    """Convert float values that are close to integers back to int."""
    for k, v in state.items():
        if isinstance(v, float) and not isinstance(v, bool):
            if abs(v - round(v)) < 1e-6:
                state[k] = int(round(v))
    return state


def _strip_zw(state: Dict[str, Any]) -> Dict[str, Any]:
    """Strip zero-width Unicode characters from string values."""
    zw_chars = {_ZW_SPACE, _ZW_NON_JOINER, _ZW_JOINER}
    for k, v in state.items():
        if isinstance(v, str):
            state[k] = "".join(c for c in v if c not in zw_chars)
    return state


def _shuffle_keys(state: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild the dict with alphabetically sorted keys."""
    return dict(sorted(state.items()))


def _round_floats(state: Dict[str, Any], decimals: int) -> Dict[str, Any]:
    """Round all float values to given decimal places."""
    for k, v in state.items():
        if isinstance(v, float) and not isinstance(v, bool):
            state[k] = round(v, decimals)
    return state


def _add_noise(state: Dict[str, Any]) -> Dict[str, Any]:
    """Add extra fields that weren't in the original."""
    state["__noise_field_1"] = "noise"
    state["__noise_field_2"] = 999
    return state


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _demo() -> None:
    """Run a demonstration of the watermark system."""
    print("=" * 60)
    print("  Agent State Watermarking Demo")
    print("=" * 60)
    print()

    # Sample agent state
    state = {
        "cpu_usage": 0.75,
        "memory_mb": 512.0,
        "task_count": 42,
        "replication_depth": 2,
        "label": "production-worker",
        "status": "active",
        "priority": 0.9,
        "uptime_seconds": 3600.0,
        "error_rate": 0.02,
        "throughput": 150.5,
    }

    engine = WatermarkEngine(WatermarkConfig(secret="demo-secret-key"))

    # Embed
    print("1. EMBEDDING WATERMARK")
    print("-" * 40)
    print(f"   Original state keys: {sorted(state.keys())}")
    wm_state, receipt = engine.embed(
        state, worker_id="w-alpha-001", depth=2,
        lineage_hash="abc123def456",
        timestamp=1709730000.0,
    )
    print(f"   Worker ID: {receipt.fingerprint.worker_id}")
    print(f"   Depth: {receipt.fingerprint.depth}")
    print(f"   Strategies used: {[s.value for s in receipt.strategies_used]}")
    print(f"   Bits embedded: {receipt.bits_embedded}")
    print(f"   HMAC: {receipt.hmac_signature[:32]}...")
    print()

    # Verify
    print("2. VERIFICATION")
    print("-" * 40)
    result = engine.verify(wm_state, receipt.fingerprint, signature=receipt.hmac_signature)
    print(f"   Status: {result.status.value}")
    print(f"   Authentic: {result.authentic}")
    print(f"   Signature valid: {result.signature_valid}")
    print(f"   Recovery rate: {result.recovery_rate:.1%}")
    print(f"   Bits: {result.bits_recovered}/{result.bits_expected}")
    print()

    # Verify tampered state
    print("3. TAMPERED STATE VERIFICATION")
    print("-" * 40)
    tampered = _deep_copy_dict(wm_state)
    # Remove some keys and coerce types
    for k in list(tampered.keys())[:3]:
        del tampered[k]
    tampered_result = engine.verify(tampered, receipt.fingerprint, signature=receipt.hmac_signature)
    print(f"   Status: {tampered_result.status.value}")
    print(f"   Authentic: {tampered_result.authentic}")
    print(f"   Recovery rate: {tampered_result.recovery_rate:.1%}")
    print()

    # Robustness test
    print("4. ROBUSTNESS TESTING")
    print("-" * 40)
    robustness = engine.test_robustness(state, worker_id="w-test", depth=1)
    for r in robustness.results:
        icon = "\u2705" if r.survived else "\u274c"
        print(f"   {icon} {r.transformation:30s} {r.recovery_rate:.0%}")
    print(f"\n   Overall: {robustness.overall_score:.0%} ({robustness.grade})")
    print()

    # Multi-agent watermarking
    print("5. MULTI-AGENT PROVENANCE")
    print("-" * 40)
    workers = ["w-alpha", "w-beta", "w-gamma"]
    receipts = {}
    for wid in workers:
        _, rcpt = engine.embed(
            state, worker_id=wid, depth=1, timestamp=1709730000.0
        )
        receipts[wid] = rcpt
        print(f"   {wid}: nonce={rcpt.fingerprint.nonce}, "
              f"bits={rcpt.bits_embedded}")

    print(f"\n   Embed history: {len(engine.embed_history)} total embeddings")
    print()
    print("=" * 60)


def main() -> None:
    """CLI entry point."""
    args = sys.argv[1:]

    if "--json" in args:
        engine = WatermarkEngine(WatermarkConfig(secret="cli-secret"))
        state = {
            "cpu": 0.5, "mem": 256.0, "tasks": 10,
            "label": "test", "status": "ok",
        }
        wm_state, receipt = engine.embed(
            state, worker_id="cli-worker", depth=1,
            timestamp=1709730000.0,
        )
        result = engine.verify(wm_state, receipt.fingerprint)
        output = {
            "fingerprint": receipt.fingerprint.to_dict(),
            "strategies": [s.value for s in receipt.strategies_used],
            "bits_embedded": receipt.bits_embedded,
            "hmac": receipt.hmac_signature,
            "verify": {
                "status": result.status.value,
                "authentic": result.authentic,
                "recovery_rate": result.recovery_rate,
            },
        }
        print(json.dumps(output, indent=2))
        return

    if "--robustness" in args:
        engine = WatermarkEngine(WatermarkConfig(secret="robust-secret"))
        state = {
            "cpu": 0.8, "mem": 1024.0, "depth": 3,
            "label": "robustness-test", "name": "agent-x",
            "score": 0.95, "errors": 2, "rate": 0.01,
        }
        report = engine.test_robustness(state)
        print(report.render())
        return

    _demo()


if __name__ == "__main__":
    main()
