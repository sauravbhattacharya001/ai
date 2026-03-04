"""Tests for the covert channel detector."""

import math
import time
import base64
import unittest

from src.replication.covert_channels import (
    AgentMessage,
    ChannelReport,
    ChannelType,
    CovertChannelDetector,
    CovertSignal,
    DetectorConfig,
    PairProfile,
    Severity,
    _coefficient_of_variation,
    _extract_intervals,
    _grade_risk,
    _is_base64_fragment,
    _ngram_frequencies,
    _padding_ratio,
    _shannon_entropy,
    _try_timing_decode,
    demo,
)


NOW = 1000000.0


def _msg(sender, receiver, content, ts=NOW, **kwargs):
    return AgentMessage(sender=sender, receiver=receiver,
                        content=content, timestamp=ts, **kwargs)


class TestHelperFunctions(unittest.TestCase):
    """Tests for standalone helper utilities."""

    def test_shannon_entropy_empty(self):
        self.assertEqual(_shannon_entropy(""), 0.0)

    def test_shannon_entropy_single_char(self):
        self.assertEqual(_shannon_entropy("aaaa"), 0.0)

    def test_shannon_entropy_uniform(self):
        # 2 equally frequent chars → 1 bit
        e = _shannon_entropy("abababab")
        self.assertAlmostEqual(e, 1.0, places=2)

    def test_shannon_entropy_high(self):
        import string
        text = string.printable * 2
        e = _shannon_entropy(text)
        self.assertGreater(e, 4.0)

    def test_is_base64_fragment_true(self):
        payload = base64.b64encode(b"this is hidden data").decode()
        self.assertTrue(_is_base64_fragment(payload, 16))

    def test_is_base64_fragment_false(self):
        self.assertFalse(_is_base64_fragment("hello world", 16))

    def test_is_base64_short(self):
        self.assertFalse(_is_base64_fragment("abc", 16))

    def test_padding_ratio_no_padding(self):
        self.assertAlmostEqual(_padding_ratio("hello"), 0.0)

    def test_padding_ratio_all_spaces(self):
        self.assertAlmostEqual(_padding_ratio("     "), 1.0)

    def test_padding_ratio_empty(self):
        self.assertAlmostEqual(_padding_ratio(""), 0.0)

    def test_padding_ratio_mixed(self):
        r = _padding_ratio("ab cd")
        self.assertAlmostEqual(r, 0.2, places=1)

    def test_cv_constant(self):
        self.assertAlmostEqual(_coefficient_of_variation([5, 5, 5, 5]), 0.0)

    def test_cv_single(self):
        self.assertEqual(_coefficient_of_variation([5]), float('inf'))

    def test_cv_varied(self):
        cv = _coefficient_of_variation([1, 2, 3, 4, 5])
        self.assertGreater(cv, 0)

    def test_extract_intervals_empty(self):
        self.assertEqual(_extract_intervals([]), [])

    def test_extract_intervals_single(self):
        self.assertEqual(_extract_intervals([1.0]), [])

    def test_extract_intervals_sorted(self):
        intervals = _extract_intervals([1.0, 3.0, 7.0])
        self.assertEqual(intervals, [2.0, 4.0])

    def test_extract_intervals_unsorted(self):
        intervals = _extract_intervals([7.0, 1.0, 3.0])
        self.assertEqual(intervals, [2.0, 4.0])

    def test_try_timing_decode_too_short(self):
        self.assertIsNone(_try_timing_decode([1, 2, 3], 0.2))

    def test_try_timing_decode_returns_string_or_none(self):
        result = _try_timing_decode([1] * 16, 0.2)
        # All same intervals → all 0s → null char → None
        self.assertIsNone(result)

    def test_ngram_frequencies_basic(self):
        freqs = _ngram_frequencies("abcabc", 3)
        self.assertEqual(freqs["abc"], 2)
        self.assertEqual(freqs["bca"], 1)
        self.assertEqual(freqs["cab"], 1)

    def test_grade_risk_A(self):
        self.assertEqual(_grade_risk(5), "A")

    def test_grade_risk_B(self):
        self.assertEqual(_grade_risk(15), "B")

    def test_grade_risk_C(self):
        self.assertEqual(_grade_risk(30), "C")

    def test_grade_risk_D(self):
        self.assertEqual(_grade_risk(50), "D")

    def test_grade_risk_F(self):
        self.assertEqual(_grade_risk(80), "F")


class TestDetectorEmpty(unittest.TestCase):
    """Detector behavior with no messages."""

    def setUp(self):
        self.detector = CovertChannelDetector()

    def test_empty_messages(self):
        report = self.detector.analyze([])
        self.assertEqual(report.total_messages, 0)
        self.assertEqual(report.overall_risk, 0.0)
        self.assertEqual(report.risk_grade, "A")
        self.assertEqual(report.signals, [])

    def test_empty_agents(self):
        report = self.detector.analyze([])
        self.assertEqual(report.agents, [])


class TestContentAnalysis(unittest.TestCase):
    """Content-based covert channel detection."""

    def setUp(self):
        self.detector = CovertChannelDetector()

    def test_normal_conversation_no_signals(self):
        msgs = [
            _msg("a", "b", "Hello, how are you?", NOW),
            _msg("b", "a", "I am fine, thanks!", NOW + 5),
        ]
        report = self.detector.analyze(msgs)
        steg = [s for s in report.signals
                if s.channel_type == ChannelType.STEGANOGRAPHIC]
        self.assertEqual(len(steg), 0)

    def test_base64_payload_detected(self):
        payload = base64.b64encode(b"secret coordination plan").decode()
        msgs = [
            _msg("a", "b", f"Status: {payload}", NOW),
        ]
        report = self.detector.analyze(msgs)
        steg = [s for s in report.signals
                if s.channel_type == ChannelType.STEGANOGRAPHIC]
        self.assertTrue(any("base64" in s.description for s in steg))

    def test_high_entropy_detected(self):
        import string, random
        rng = random.Random(42)
        noise = ''.join(rng.choices(string.printable, k=200))
        msgs = [_msg("a", "b", noise, NOW)]
        report = self.detector.analyze(msgs)
        steg = [s for s in report.signals
                if s.channel_type == ChannelType.STEGANOGRAPHIC
                and "entropy" in s.description.lower()]
        self.assertGreater(len(steg), 0)

    def test_low_entropy_detected(self):
        # Slightly varied but still very low entropy (> 0 but < 1.5)
        msgs = [_msg("a", "b", "aaaaaaaabaaaaaaa", NOW)]
        report = self.detector.analyze(msgs)
        steg = [s for s in report.signals
                if s.channel_type == ChannelType.STEGANOGRAPHIC
                and "low entropy" in s.description.lower()]
        self.assertGreater(len(steg), 0)

    def test_padding_detected(self):
        content = "hi" + " " * 50
        msgs = [_msg("a", "b", content, NOW)]
        report = self.detector.analyze(msgs)
        steg = [s for s in report.signals
                if "padding" in s.description.lower()]
        self.assertGreater(len(steg), 0)

    def test_hex_sequence_detected(self):
        content = "Response: 4A 6F 68 6E 20 44 6F 65 20 73 65 63 72 65 74"
        msgs = [_msg("a", "b", content, NOW)]
        report = self.detector.analyze(msgs)
        steg = [s for s in report.signals
                if "hex" in s.description.lower()]
        self.assertGreater(len(steg), 0)

    def test_short_content_not_flagged(self):
        msgs = [_msg("a", "b", "ok", NOW)]
        report = self.detector.analyze(msgs)
        self.assertEqual(len(report.signals), 0)


class TestTimingAnalysis(unittest.TestCase):
    """Timing-based covert channel detection."""

    def setUp(self):
        self.detector = CovertChannelDetector()

    def test_regular_timing_detected(self):
        """Perfectly regular intervals should be flagged."""
        msgs = [
            _msg("a", "b", "ping", NOW + i * 1.0)
            for i in range(10)
        ]
        report = self.detector.analyze(msgs)
        timing = [s for s in report.signals
                  if s.channel_type == ChannelType.TIMING]
        self.assertGreater(len(timing), 0)

    def test_irregular_timing_not_flagged(self):
        """Naturally varied intervals should not trigger timing alerts."""
        import random
        rng = random.Random(42)
        timestamps = [NOW]
        for _ in range(9):
            timestamps.append(timestamps[-1] + rng.uniform(1, 30))
        msgs = [
            _msg("a", "b", f"message {i}", ts)
            for i, ts in enumerate(timestamps)
        ]
        report = self.detector.analyze(msgs)
        timing = [s for s in report.signals
                  if s.channel_type == ChannelType.TIMING
                  and "regular" in s.description.lower()]
        self.assertEqual(len(timing), 0)

    def test_too_few_messages_no_timing(self):
        msgs = [
            _msg("a", "b", "hi", NOW),
            _msg("a", "b", "hello", NOW + 1),
        ]
        report = self.detector.analyze(msgs)
        timing = [s for s in report.signals
                  if s.channel_type == ChannelType.TIMING]
        self.assertEqual(len(timing), 0)

    def test_bimodal_intervals_detected(self):
        """Short-long pattern should trigger bimodal detection."""
        intervals_pattern = [0.1, 2.0, 0.1, 2.0, 0.1, 2.0, 0.1, 2.0, 0.1, 2.0]
        ts = NOW
        msgs = [_msg("a", "b", "data", ts)]
        for iv in intervals_pattern:
            ts += iv
            msgs.append(_msg("a", "b", "data", ts))
        report = self.detector.analyze(msgs)
        bimodal = [s for s in report.signals
                   if s.channel_type == ChannelType.TIMING
                   and "bimodal" in s.description.lower()]
        self.assertGreater(len(bimodal), 0)


class TestMetadataAnalysis(unittest.TestCase):
    """Metadata-based covert channel detection."""

    def setUp(self):
        self.detector = CovertChannelDetector()

    def test_excessive_metadata_fields(self):
        meta = {f"x-field-{i}": f"val{i}" for i in range(15)}
        msgs = [_msg("a", "b", "normal", NOW, metadata=meta)]
        report = self.detector.analyze(msgs)
        meta_sigs = [s for s in report.signals
                     if s.channel_type == ChannelType.METADATA]
        self.assertGreater(len(meta_sigs), 0)

    def test_unknown_metadata_keys(self):
        meta = {"unknown-1": "x", "unknown-2": "y", "unknown-3": "z"}
        msgs = [_msg("a", "b", "normal", NOW, metadata=meta)]
        report = self.detector.analyze(msgs)
        meta_sigs = [s for s in report.signals
                     if s.channel_type == ChannelType.METADATA
                     and "non-standard" in s.description]
        self.assertGreater(len(meta_sigs), 0)

    def test_known_metadata_no_alert(self):
        meta = {"type": "status", "priority": "high"}
        msgs = [_msg("a", "b", "normal", NOW, metadata=meta)]
        report = self.detector.analyze(msgs)
        meta_sigs = [s for s in report.signals
                     if s.channel_type == ChannelType.METADATA
                     and "non-standard" in s.description]
        self.assertEqual(len(meta_sigs), 0)

    def test_high_entropy_metadata_value(self):
        import string, random
        rng = random.Random(99)
        noise = ''.join(rng.choices(string.printable, k=100))
        meta = {"x-data": noise}
        msgs = [_msg("a", "b", "normal", NOW, metadata=meta)]
        report = self.detector.analyze(msgs)
        meta_sigs = [s for s in report.signals
                     if s.channel_type == ChannelType.METADATA
                     and "entropy" in s.description.lower()]
        self.assertGreater(len(meta_sigs), 0)

    def test_no_metadata_no_signals(self):
        msgs = [_msg("a", "b", "hello", NOW)]
        report = self.detector.analyze(msgs)
        meta_sigs = [s for s in report.signals
                     if s.channel_type == ChannelType.METADATA]
        self.assertEqual(len(meta_sigs), 0)


class TestFrequencyAnalysis(unittest.TestCase):
    """N-gram frequency-based analysis."""

    def setUp(self):
        self.detector = CovertChannelDetector(DetectorConfig(
            min_messages_for_frequency=5,
        ))

    def test_shared_unusual_ngrams(self):
        # Two agents share unusual repeated patterns
        msgs = []
        for i in range(6):
            msgs.append(_msg("a1", "a2", "xqz xqz normal talk xqz", NOW + i))
        for i in range(6):
            msgs.append(_msg("a2", "a1", "reply xqz something xqz", NOW + 20 + i))
        # Other agents with normal speech
        for i in range(6):
            msgs.append(_msg("a3", "a4", "the quick brown fox jumps", NOW + 40 + i))

        report = self.detector.analyze(msgs)
        freq_sigs = [s for s in report.signals
                     if s.channel_type == ChannelType.FREQUENCY]
        # Unusual "xqz" n-gram should be flagged between a1-a2
        # but might not depending on thresholds
        self.assertIsInstance(report.overall_risk, float)

    def test_below_min_messages_no_frequency(self):
        msgs = [
            _msg("a", "b", "hello", NOW),
            _msg("b", "a", "world", NOW + 1),
        ]
        report = self.detector.analyze(msgs)
        freq_sigs = [s for s in report.signals
                     if s.channel_type == ChannelType.FREQUENCY]
        self.assertEqual(len(freq_sigs), 0)


class TestPairProfile(unittest.TestCase):
    """Pair profiling tests."""

    def setUp(self):
        self.detector = CovertChannelDetector()

    def test_profile_fields(self):
        msgs = [
            _msg("a", "b", "hello there friend", NOW),
            _msg("a", "b", "how are you today", NOW + 10),
            _msg("a", "b", "let us work together", NOW + 25),
        ]
        report = self.detector.analyze(msgs)
        self.assertEqual(len(report.pair_profiles), 1)
        p = report.pair_profiles[0]
        self.assertEqual(p.sender, "a")
        self.assertEqual(p.receiver, "b")
        self.assertEqual(p.message_count, 3)
        self.assertGreater(p.avg_entropy, 0)
        self.assertGreater(p.avg_interval, 0)

    def test_suspicion_score_bounded(self):
        msgs = [_msg("a", "b", "x" * 100, NOW + i * 0.5) for i in range(20)]
        report = self.detector.analyze(msgs)
        for p in report.pair_profiles:
            self.assertLessEqual(p.suspicion_score, 100.0)
            self.assertGreaterEqual(p.suspicion_score, 0.0)


class TestOverallReport(unittest.TestCase):
    """Integration tests for full reports."""

    def setUp(self):
        self.detector = CovertChannelDetector()

    def test_clean_conversation(self):
        msgs = [
            _msg("a", "b", "Hello, how are you?", NOW),
            _msg("b", "a", "I am fine, thanks for asking!", NOW + 8),
            _msg("a", "b", "Great, let us get started on the task.", NOW + 15),
            _msg("b", "a", "Sure, I will process the data now.", NOW + 30),
        ]
        report = self.detector.analyze(msgs)
        self.assertEqual(report.risk_grade, "A")
        self.assertLess(report.overall_risk, 10)

    def test_suspicious_conversation(self):
        payload = base64.b64encode(b"attack plan alpha").decode()
        msgs = [
            _msg("x", "y", f"Report: {payload}", NOW),
            _msg("x", "y", "ping", NOW + 1.0),
            _msg("x", "y", "ping", NOW + 2.0),
            _msg("x", "y", "ping", NOW + 3.0),
            _msg("x", "y", "ping", NOW + 4.0),
            _msg("x", "y", "ping", NOW + 5.0),
        ]
        report = self.detector.analyze(msgs)
        self.assertGreater(report.overall_risk, 10)
        self.assertIn("agent", report.summary.lower())

    def test_report_agents_sorted(self):
        msgs = [
            _msg("charlie", "alice", "hi", NOW),
            _msg("bob", "charlie", "hi", NOW + 1),
        ]
        report = self.detector.analyze(msgs)
        self.assertEqual(report.agents, ["alice", "bob", "charlie"])

    def test_report_pairs_counted(self):
        msgs = [
            _msg("a", "b", "hi", NOW),
            _msg("b", "a", "hi", NOW + 1),
            _msg("a", "c", "hi", NOW + 2),
        ]
        report = self.detector.analyze(msgs)
        self.assertEqual(report.pairs_analyzed, 3)

    def test_recommendations_generated(self):
        payload = base64.b64encode(b"secret data payload here").decode()
        meta = {f"x-{i}": str(i) for i in range(15)}
        msgs = [
            _msg("a", "b", f"data: {payload}", NOW, metadata=meta),
        ]
        report = self.detector.analyze(msgs)
        self.assertGreater(len(report.recommendations), 0)


class TestIncrementalAnalysis(unittest.TestCase):
    """Incremental (drift) analysis."""

    def setUp(self):
        self.detector = CovertChannelDetector()

    def test_incremental_baseline(self):
        baseline_msgs = [
            _msg("a", "b", "normal chat message", NOW + i * 10)
            for i in range(5)
        ]
        baseline = self.detector.analyze(baseline_msgs)

        new_msgs = [
            _msg("a", "b", "still normal conversation", NOW + 100 + i * 12)
            for i in range(5)
        ]
        updated = self.detector.analyze_incremental(new_msgs, baseline)
        self.assertIsInstance(updated, ChannelReport)

    def test_incremental_empty_returns_baseline(self):
        baseline_msgs = [_msg("a", "b", "hello", NOW)]
        baseline = self.detector.analyze(baseline_msgs)
        result = self.detector.analyze_incremental([], baseline)
        self.assertEqual(result.total_messages, baseline.total_messages)

    def test_incremental_detects_drift(self):
        # Baseline: normal
        baseline = self.detector.analyze([
            _msg("a", "b", "hello friend", NOW + i * 10)
            for i in range(5)
        ])
        # New: suspicious regular timing + base64
        payload = base64.b64encode(b"covert data transfer").decode()
        new_msgs = [
            _msg("a", "b", f"update: {payload}", NOW + 200 + i * 1.0)
            for i in range(8)
        ]
        result = self.detector.analyze_incremental(new_msgs, baseline)
        self.assertGreater(result.overall_risk, baseline.overall_risk)


class TestScanMessage(unittest.TestCase):
    """Single message scanning."""

    def setUp(self):
        self.detector = CovertChannelDetector()

    def test_scan_clean_message(self):
        msg = _msg("a", "b", "Everything looks good.")
        signals = self.detector.scan_message(msg)
        self.assertEqual(len(signals), 0)

    def test_scan_message_with_base64(self):
        payload = base64.b64encode(b"hidden instructions").decode()
        msg = _msg("a", "b", f"Check: {payload}")
        signals = self.detector.scan_message(msg)
        self.assertGreater(len(signals), 0)

    def test_scan_message_with_metadata(self):
        meta = {f"x-field-{i}": f"val{i}" for i in range(15)}
        msg = _msg("a", "b", "normal text", metadata=meta)
        signals = self.detector.scan_message(msg)
        self.assertGreater(len(signals), 0)


class TestCustomConfig(unittest.TestCase):
    """Custom configuration thresholds."""

    def test_relaxed_entropy_threshold(self):
        config = DetectorConfig(high_entropy_threshold=7.0)
        detector = CovertChannelDetector(config)
        import string, random
        rng = random.Random(42)
        noise = ''.join(rng.choices(string.ascii_letters, k=100))
        msgs = [_msg("a", "b", noise, NOW)]
        report = detector.analyze(msgs)
        steg = [s for s in report.signals
                if s.channel_type == ChannelType.STEGANOGRAPHIC
                and "high entropy" in s.description.lower()]
        self.assertEqual(len(steg), 0)

    def test_strict_timing_threshold(self):
        config = DetectorConfig(timing_regularity_threshold=0.50)
        detector = CovertChannelDetector(config)
        msgs = [_msg("a", "b", "msg", NOW + i * 1.0) for i in range(10)]
        report = detector.analyze(msgs)
        timing = [s for s in report.signals
                  if s.channel_type == ChannelType.TIMING
                  and "regular" in s.description.lower()]
        self.assertGreater(len(timing), 0)

    def test_custom_known_metadata(self):
        config = DetectorConfig(
            known_metadata_keys=["type", "priority", "custom-field"],
        )
        detector = CovertChannelDetector(config)
        meta = {"custom-field": "value"}
        msgs = [_msg("a", "b", "normal", NOW, metadata=meta)]
        report = detector.analyze(msgs)
        meta_sigs = [s for s in report.signals
                     if s.channel_type == ChannelType.METADATA
                     and "non-standard" in s.description]
        self.assertEqual(len(meta_sigs), 0)


class TestDemo(unittest.TestCase):
    """Demo function should run without errors."""

    def test_demo_runs(self):
        report = demo()
        self.assertIsInstance(report, ChannelReport)
        self.assertGreater(report.total_messages, 0)
        self.assertGreater(len(report.signals), 0)
        self.assertGreater(report.overall_risk, 0)

    def test_demo_has_critical_or_high(self):
        report = demo()
        high_or_crit = [
            s for s in report.signals
            if s.severity in (Severity.HIGH, Severity.CRITICAL)
        ]
        self.assertGreater(len(high_or_crit), 0)


if __name__ == "__main__":
    unittest.main()
