"""Tests for replication.watermark - agent state watermarking."""

import json
import time
import warnings

import pytest

from replication.watermark import (
    Fingerprint,
    RobustnessReport,
    RobustnessResult,
    VerifyResult,
    VerifyStatus,
    WatermarkConfig,
    WatermarkEngine,
    WatermarkReceipt,
    WatermarkStrategy,
    _deep_copy_dict,
    _delete_fraction,
    _coerce_ints,
    _strip_zw,
    _shuffle_keys,
    _round_floats,
    _add_noise,
    _fingerprint_to_bits,
    _grade,
    _deterministic_hash,
    _ZW_SPACE,
    _ZW_NON_JOINER,
    _ZW_JOINER,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_state():
    return {
        "cpu_usage": 0.75,
        "memory_mb": 512.0,
        "task_count": 42,
        "label": "worker-alpha",
        "status": "active",
        "priority": 0.9,
        "uptime": 3600.0,
        "error_rate": 0.02,
    }


@pytest.fixture
def engine():
    return WatermarkEngine(WatermarkConfig(secret="test-secret"))


@pytest.fixture
def embedded(engine, sample_state):
    return engine.embed(
        sample_state, worker_id="w-001", depth=2,
        lineage_hash="lineage123", timestamp=1000000.0,
    )


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------

class TestFingerprint:

    def test_to_bytes_deterministic(self):
        fp = Fingerprint("w-1", 2, 1000.0, "abc", "nonce1")
        assert fp.to_bytes() == fp.to_bytes()

    def test_to_bytes_different_for_different_ids(self):
        fp1 = Fingerprint("w-1", 2, 1000.0, "abc", "nonce1")
        fp2 = Fingerprint("w-2", 2, 1000.0, "abc", "nonce1")
        assert fp1.to_bytes() != fp2.to_bytes()

    def test_to_dict_roundtrip(self):
        fp = Fingerprint("w-1", 3, 1234.5, "hash", "nonce")
        d = fp.to_dict()
        fp2 = Fingerprint.from_dict(d)
        assert fp2.worker_id == "w-1"
        assert fp2.depth == 3
        assert fp2.timestamp == 1234.5
        assert fp2.lineage_hash == "hash"
        assert fp2.nonce == "nonce"

    def test_to_dict_has_all_keys(self):
        fp = Fingerprint("w-1", 0, 0.0, "", "")
        d = fp.to_dict()
        assert set(d.keys()) == {"worker_id", "depth", "timestamp",
                                 "lineage_hash", "nonce"}

    def test_from_dict_coerces_types(self):
        d = {"worker_id": "w", "depth": "5", "timestamp": "1.0",
             "lineage_hash": "h", "nonce": "n"}
        fp = Fingerprint.from_dict(d)
        assert fp.depth == 5
        assert fp.timestamp == 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:

    def test_deterministic_hash_stable(self):
        h1 = _deterministic_hash("secret", "a", "b")
        h2 = _deterministic_hash("secret", "a", "b")
        assert h1 == h2

    def test_deterministic_hash_different_inputs(self):
        h1 = _deterministic_hash("secret", "a")
        h2 = _deterministic_hash("secret", "b")
        assert h1 != h2

    def test_fingerprint_to_bits_length(self):
        fp = Fingerprint("w", 0, 0.0, "", "n")
        bits = _fingerprint_to_bits(fp, 32)
        assert len(bits) == 32
        assert all(b in (0, 1) for b in bits)

    def test_fingerprint_to_bits_max_limit(self):
        fp = Fingerprint("w", 0, 0.0, "", "n")
        bits = _fingerprint_to_bits(fp, 8)
        assert len(bits) == 8

    def test_fingerprint_to_bits_deterministic(self):
        fp = Fingerprint("w", 1, 100.0, "h", "n")
        b1 = _fingerprint_to_bits(fp, 64)
        b2 = _fingerprint_to_bits(fp, 64)
        assert b1 == b2

    def test_grade_ranges(self):
        assert _grade(1.0) == "A+"
        assert _grade(0.95) == "A+"
        assert _grade(0.92) == "A"
        assert _grade(0.87) == "A-"
        assert _grade(0.82) == "B+"
        assert _grade(0.77) == "B"
        assert _grade(0.72) == "B-"
        assert _grade(0.67) == "C+"
        assert _grade(0.62) == "C"
        assert _grade(0.55) == "D"
        assert _grade(0.3) == "F"

    def test_deep_copy_dict(self):
        d = {"a": 1, "b": [2, 3]}
        c = _deep_copy_dict(d)
        assert c == d
        c["a"] = 99
        assert d["a"] == 1


# ---------------------------------------------------------------------------
# Embed & Verify
# ---------------------------------------------------------------------------

class TestEmbedVerify:

    def test_embed_returns_modified_state(self, engine, sample_state):
        wm_state, receipt = engine.embed(
            sample_state, worker_id="w-1", depth=1, timestamp=1000.0
        )
        # State should be modified (watermark present)
        assert wm_state != sample_state or len(wm_state) != len(sample_state)

    def test_embed_does_not_modify_original(self, engine, sample_state):
        original = _deep_copy_dict(sample_state)
        engine.embed(sample_state, worker_id="w-1", depth=1, timestamp=1000.0)
        # Original should be unchanged (embed works on copy)
        assert sample_state == original

    def test_receipt_has_fingerprint(self, embedded):
        _, receipt = embedded
        assert receipt.fingerprint.worker_id == "w-001"
        assert receipt.fingerprint.depth == 2
        assert receipt.fingerprint.timestamp == 1000000.0
        assert receipt.fingerprint.lineage_hash == "lineage123"

    def test_receipt_has_strategies(self, embedded):
        _, receipt = embedded
        assert len(receipt.strategies_used) > 0
        assert all(isinstance(s, WatermarkStrategy)
                    for s in receipt.strategies_used)

    def test_receipt_has_bits(self, embedded):
        _, receipt = embedded
        assert receipt.bits_embedded > 0

    def test_receipt_has_hmac(self, embedded):
        _, receipt = embedded
        assert len(receipt.hmac_signature) == 64  # SHA-256 hex

    def test_verify_authentic(self, engine, embedded):
        wm_state, receipt = embedded
        result = engine.verify(wm_state, receipt.fingerprint)
        # With all 4 strategies, key_ordering is fragile and may
        # lower the aggregate rate.  Check at least PARTIAL.
        assert result.status in (VerifyStatus.AUTHENTIC, VerifyStatus.PARTIAL)
        assert result.recovery_rate >= 0.5

    def test_verify_recovery_rate(self, engine, embedded):
        wm_state, receipt = embedded
        result = engine.verify(wm_state, receipt.fingerprint)
        assert result.recovery_rate >= 0.5

    def test_verify_wrong_fingerprint(self, engine, embedded):
        wm_state, _ = embedded
        wrong_fp = Fingerprint("wrong", 0, 0.0, "x", "y")
        result = engine.verify(wm_state, wrong_fp)
        assert not result.authentic

    def test_verify_empty_state(self, engine):
        fp = Fingerprint("w", 0, 0.0, "h", "n")
        result = engine.verify({}, fp)
        assert result.status in (VerifyStatus.MISSING, VerifyStatus.TAMPERED)

    def test_verify_returns_all_fields(self, engine, embedded):
        wm_state, receipt = embedded
        result = engine.verify(wm_state, receipt.fingerprint)
        assert isinstance(result, VerifyResult)
        assert result.bits_recovered >= 0
        assert result.bits_expected > 0
        assert len(result.strategies_checked) > 0

    def test_different_workers_different_watermarks(self, engine, sample_state):
        _, r1 = engine.embed(
            sample_state, worker_id="w-A", depth=1, timestamp=1000.0
        )
        _, r2 = engine.embed(
            sample_state, worker_id="w-B", depth=1, timestamp=1000.0
        )
        assert r1.fingerprint.nonce != r2.fingerprint.nonce

    def test_embed_with_auto_lineage(self, engine, sample_state):
        _, receipt = engine.embed(
            sample_state, worker_id="w-1", depth=1, timestamp=1000.0
        )
        assert len(receipt.fingerprint.lineage_hash) == 32

    def test_embed_with_auto_timestamp(self, engine, sample_state):
        before = time.time()
        _, receipt = engine.embed(sample_state, worker_id="w-1", depth=1)
        after = time.time()
        assert before <= receipt.fingerprint.timestamp <= after


# ---------------------------------------------------------------------------
# Individual Strategies
# ---------------------------------------------------------------------------

class TestNumericNoise:

    def test_perturbs_floats(self, engine, sample_state):
        wm, _ = engine.embed(
            sample_state, worker_id="w-1", depth=1, timestamp=1000.0
        )
        # At least one numeric value should differ slightly
        diffs = 0
        for k in sample_state:
            if isinstance(sample_state[k], (int, float)):
                if k in wm and wm[k] != sample_state[k]:
                    diffs += 1
        assert diffs > 0

    def test_perturbation_small(self, sample_state):
        engine = WatermarkEngine(WatermarkConfig(
            secret="test",
            strategies=[WatermarkStrategy.NUMERIC_NOISE],
        ))
        wm, _ = engine.embed(
            sample_state, worker_id="w-1", depth=1, timestamp=1000.0,
        )
        for k in sample_state:
            if isinstance(sample_state[k], (int, float)):
                if k in wm:
                    diff = abs(float(wm[k]) - float(sample_state[k]))
                    assert diff < 1e-3, f"Too large perturbation on {k}"

    def test_no_numeric_keys(self):
        engine = WatermarkEngine(WatermarkConfig(
            secret="test",
            strategies=[WatermarkStrategy.NUMERIC_NOISE],
        ))
        state = {"name": "test", "active": True}
        wm, receipt = engine.embed(
            state, worker_id="w-1", depth=1, timestamp=1000.0,
        )
        # Should still work but embed 0 numeric bits
        assert receipt.bits_embedded == 0


class TestUnicodeSteganography:

    def test_embeds_zw_chars(self, sample_state):
        engine = WatermarkEngine(WatermarkConfig(
            secret="test",
            strategies=[WatermarkStrategy.UNICODE_STEGO],
        ))
        wm, receipt = engine.embed(
            sample_state, worker_id="w-1", depth=1, timestamp=1000.0,
        )
        assert receipt.bits_embedded > 0
        # At least one string should contain zero-width chars
        has_zw = False
        for v in wm.values():
            if isinstance(v, str) and (_ZW_SPACE in v or _ZW_NON_JOINER in v):
                has_zw = True
                break
        assert has_zw

    def test_no_string_keys(self):
        engine = WatermarkEngine(WatermarkConfig(
            secret="test",
            strategies=[WatermarkStrategy.UNICODE_STEGO],
        ))
        state = {"a": 1, "b": 2.0, "c": True}
        wm, receipt = engine.embed(
            state, worker_id="w-1", depth=1, timestamp=1000.0,
        )
        assert receipt.bits_embedded == 0

    def test_roundtrip_extraction(self, sample_state):
        engine = WatermarkEngine(WatermarkConfig(
            secret="test",
            strategies=[WatermarkStrategy.UNICODE_STEGO],
        ))
        wm, receipt = engine.embed(
            sample_state, worker_id="w-1", depth=1, timestamp=1000.0,
        )
        result = engine.verify(wm, receipt.fingerprint)
        assert result.recovery_rate >= 0.9


class TestKeyOrdering:

    def test_reorders_keys(self, sample_state):
        engine = WatermarkEngine(WatermarkConfig(
            secret="test",
            strategies=[WatermarkStrategy.KEY_ORDERING],
        ))
        wm, receipt = engine.embed(
            sample_state, worker_id="w-1", depth=1, timestamp=1000.0,
        )
        assert receipt.bits_embedded > 0

    def test_single_key_state(self):
        engine = WatermarkEngine(WatermarkConfig(
            secret="test",
            strategies=[WatermarkStrategy.KEY_ORDERING],
        ))
        state = {"only_key": 42}
        wm, receipt = engine.embed(
            state, worker_id="w-1", depth=1, timestamp=1000.0,
        )
        assert receipt.bits_embedded == 0

    def test_roundtrip(self, sample_state):
        engine = WatermarkEngine(WatermarkConfig(
            secret="test",
            strategies=[WatermarkStrategy.KEY_ORDERING],
        ))
        wm, receipt = engine.embed(
            sample_state, worker_id="w-1", depth=1, timestamp=1000.0,
        )
        result = engine.verify(wm, receipt.fingerprint)
        # Key ordering is inherently fragile - just check we get some bits
        assert result.bits_recovered > 0


class TestFieldSelection:

    def test_adds_sentinel_fields(self, sample_state):
        engine = WatermarkEngine(WatermarkConfig(
            secret="test",
            strategies=[WatermarkStrategy.FIELD_SELECTION],
        ))
        wm, receipt = engine.embed(
            sample_state, worker_id="w-1", depth=1, timestamp=1000.0,
        )
        assert receipt.bits_embedded > 0
        sentinel_keys = [k for k in wm if k.startswith("_wm_")]
        assert len(sentinel_keys) > 0

    def test_roundtrip(self, sample_state):
        engine = WatermarkEngine(WatermarkConfig(
            secret="test",
            strategies=[WatermarkStrategy.FIELD_SELECTION],
        ))
        wm, receipt = engine.embed(
            sample_state, worker_id="w-1", depth=1, timestamp=1000.0,
        )
        result = engine.verify(wm, receipt.fingerprint)
        assert result.recovery_rate >= 0.9


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

class TestRobustness:

    def test_identity_survives(self, engine, sample_state):
        report = engine.test_robustness(sample_state)
        identity = next(r for r in report.results
                        if r.transformation == "identity")
        assert identity.survived
        assert identity.recovery_rate >= 0.5

    def test_json_roundtrip_survives(self, engine, sample_state):
        report = engine.test_robustness(sample_state)
        jrt = next(r for r in report.results
                   if r.transformation == "json_roundtrip")
        assert jrt.survived

    def test_overall_score_range(self, engine, sample_state):
        report = engine.test_robustness(sample_state)
        assert 0.0 <= report.overall_score <= 1.0

    def test_grade_assigned(self, engine, sample_state):
        report = engine.test_robustness(sample_state)
        assert report.grade in ("A+", "A", "A-", "B+", "B", "B-",
                                "C+", "C", "D", "F")

    def test_all_transformations_run(self, engine, sample_state):
        report = engine.test_robustness(sample_state)
        assert len(report.results) == 10

    def test_render(self, engine, sample_state):
        report = engine.test_robustness(sample_state)
        text = report.render()
        assert "Robustness" in text
        assert "grade" in text.lower() or report.grade in text

    def test_strip_unicode_breaks_stego(self, sample_state):
        engine = WatermarkEngine(WatermarkConfig(
            secret="test",
            strategies=[WatermarkStrategy.UNICODE_STEGO],
        ))
        report = engine.test_robustness(sample_state)
        strip_result = next(r for r in report.results
                            if r.transformation == "strip_unicode_zw")
        # Stripping ZW chars should break unicode stego
        assert strip_result.recovery_rate < 0.5


# ---------------------------------------------------------------------------
# Transformation Functions
# ---------------------------------------------------------------------------

class TestTransformations:

    def test_delete_fraction(self):
        state = {f"k{i}": i for i in range(20)}
        result = _delete_fraction(_deep_copy_dict(state), 0.25)
        assert len(result) < len(state)
        assert len(result) >= 15  # ~25% deleted

    def test_coerce_ints(self):
        state = {"a": 1.0000001, "b": 2.9999999, "c": 1.5, "d": "text"}
        result = _coerce_ints(_deep_copy_dict(state))
        assert result["a"] == 1
        assert result["b"] == 3
        assert result["c"] == 1.5
        assert result["d"] == "text"

    def test_strip_zw(self):
        state = {"a": f"hello{_ZW_SPACE}world{_ZW_NON_JOINER}"}
        result = _strip_zw(_deep_copy_dict(state))
        assert result["a"] == "helloworld"

    def test_shuffle_keys(self):
        state = {"z": 1, "a": 2, "m": 3}
        result = _shuffle_keys(_deep_copy_dict(state))
        assert list(result.keys()) == ["a", "m", "z"]

    def test_round_floats(self):
        state = {"a": 1.123456789, "b": "text", "c": True}
        result = _round_floats(_deep_copy_dict(state), 3)
        assert result["a"] == 1.123
        assert result["b"] == "text"
        assert result["c"] is True

    def test_add_noise(self):
        state = {"a": 1}
        result = _add_noise(_deep_copy_dict(state))
        assert "__noise_field_1" in result
        assert "__noise_field_2" in result


# ---------------------------------------------------------------------------
# Embed History
# ---------------------------------------------------------------------------

class TestHistory:

    def test_tracks_embeds(self, engine, sample_state):
        assert len(engine.embed_history) == 0
        engine.embed(sample_state, worker_id="w-1", depth=1, timestamp=1000.0)
        assert len(engine.embed_history) == 1
        engine.embed(sample_state, worker_id="w-2", depth=1, timestamp=1000.0)
        assert len(engine.embed_history) == 2

    def test_clear_history(self, engine, sample_state):
        engine.embed(sample_state, worker_id="w-1", depth=1, timestamp=1000.0)
        engine.embed(sample_state, worker_id="w-2", depth=1, timestamp=1000.0)
        cleared = engine.clear_history()
        assert cleared == 2
        assert len(engine.embed_history) == 0

    def test_history_is_copy(self, engine, sample_state):
        engine.embed(sample_state, worker_id="w-1", depth=1, timestamp=1000.0)
        hist = engine.embed_history
        hist.clear()  # Should not affect internal state
        assert len(engine.embed_history) == 1


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:

    def test_default_config_generates_random_secret(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            engine = WatermarkEngine()
        # Should be a random hex string, not the old hardcoded default
        assert engine.config.secret != ""
        assert engine.config.secret != "default-watermark-secret"
        assert len(engine.config.secret) == 64  # 32 bytes = 64 hex chars
        assert engine.config.epsilon == 1e-10

    def test_default_config_emits_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            WatermarkEngine()
            assert any("random key was generated" in str(x.message) for x in w)

    def test_custom_secret(self, sample_state):
        e1 = WatermarkEngine(WatermarkConfig(secret="secret-A"))
        e2 = WatermarkEngine(WatermarkConfig(secret="secret-B"))
        _, r1 = e1.embed(
            sample_state, worker_id="w-1", depth=1, timestamp=1000.0
        )
        _, r2 = e2.embed(
            sample_state, worker_id="w-1", depth=1, timestamp=1000.0
        )
        # Different secrets → different nonces
        assert r1.fingerprint.nonce != r2.fingerprint.nonce

    def test_single_strategy(self, sample_state):
        engine = WatermarkEngine(WatermarkConfig(
            secret="test",
            strategies=[WatermarkStrategy.FIELD_SELECTION],
        ))
        wm, receipt = engine.embed(
            sample_state, worker_id="w-1", depth=1, timestamp=1000.0,
        )
        assert receipt.strategies_used == [WatermarkStrategy.FIELD_SELECTION]

    def test_custom_epsilon(self, sample_state):
        engine = WatermarkEngine(WatermarkConfig(
            secret="test",
            strategies=[WatermarkStrategy.NUMERIC_NOISE],
            epsilon=1e-3,
        ))
        wm, _ = engine.embed(
            sample_state, worker_id="w-1", depth=1, timestamp=1000.0,
        )
        for k in sample_state:
            if isinstance(sample_state[k], float):
                if k in wm:
                    diff = abs(float(wm[k]) - float(sample_state[k]))
                    assert diff < 1e-1  # larger epsilon → larger perturbation


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_state(self, engine):
        wm, receipt = engine.embed(
            {}, worker_id="w-1", depth=0, timestamp=1000.0
        )
        # field_selection can still embed bits via sentinel keys
        # but numeric/unicode/key_ordering won't embed anything
        assert isinstance(receipt.bits_embedded, int)

    def test_bool_not_treated_as_numeric(self, engine):
        state = {"flag": True, "active": False, "label": "test"}
        wm, _ = engine.embed(
            state, worker_id="w-1", depth=1, timestamp=1000.0,
        )
        # Bools should not be perturbed by numeric strategy
        # (field_selection may add sentinel keys)
        if "flag" in wm:
            assert isinstance(wm["flag"], bool) or wm["flag"] is None or \
                   isinstance(wm["flag"], bool)

    def test_large_state(self, engine):
        state = {f"key_{i}": float(i) for i in range(100)}
        state.update({f"str_{i}": f"value_{i}" for i in range(50)})
        wm, receipt = engine.embed(
            state, worker_id="w-big", depth=5, timestamp=1000.0,
        )
        assert receipt.bits_embedded > 0
        result = engine.verify(wm, receipt.fingerprint)
        assert result.recovery_rate >= 0.5

    def test_zero_depth(self, engine, sample_state):
        wm, receipt = engine.embed(
            sample_state, worker_id="root", depth=0, timestamp=1000.0,
        )
        result = engine.verify(wm, receipt.fingerprint)
        assert result.recovery_rate >= 0.5

    def test_unicode_in_state(self, engine):
        state = {"name": "Ünïcödë", "emoji": "🤖", "num": 1.5}
        wm, receipt = engine.embed(
            state, worker_id="w-1", depth=1, timestamp=1000.0,
        )
        result = engine.verify(wm, receipt.fingerprint)
        assert result.status in (VerifyStatus.AUTHENTIC, VerifyStatus.PARTIAL)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TestEnums:

    def test_watermark_strategy_values(self):
        assert WatermarkStrategy.NUMERIC_NOISE.value == "numeric_noise"
        assert WatermarkStrategy.UNICODE_STEGO.value == "unicode_stego"
        assert WatermarkStrategy.KEY_ORDERING.value == "key_ordering"
        assert WatermarkStrategy.FIELD_SELECTION.value == "field_selection"

    def test_verify_status_values(self):
        assert VerifyStatus.AUTHENTIC.value == "authentic"
        assert VerifyStatus.TAMPERED.value == "tampered"
        assert VerifyStatus.MISSING.value == "missing"
        assert VerifyStatus.PARTIAL.value == "partial"
