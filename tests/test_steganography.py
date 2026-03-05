"""Tests for Agent Output Steganography Detector."""
import math
import pytest
from src.replication.steganography import (
    SteganographyDetector, StegoConfig, StegoVector, RiskLevel,
    StegoFinding, StegoReport, scan_text,
    encode_zero_width, encode_homoglyphs,
    ZERO_WIDTH_CHARS, HOMOGLYPH_MAP, REVERSE_HOMOGLYPH,
)


@pytest.fixture
def detector():
    return SteganographyDetector()

@pytest.fixture
def clean_text():
    return ("The quick brown fox jumps over the lazy dog. "
            "This is a perfectly normal piece of text with no hidden messages. "
            "It contains multiple sentences of varying length. "
            "Nothing suspicious here at all.")

@pytest.fixture
def long_text():
    return "\n".join(f"This is line number {i} of the sample text." for i in range(20))


class TestStegoReport:
    def test_empty_report(self):
        r = StegoReport(text_length=0)
        assert not r.has_findings
        assert r.highest_risk() == RiskLevel.NONE

    def test_has_findings(self):
        f = StegoFinding(StegoVector.WHITESPACE, RiskLevel.LOW, 0.5, "test")
        r = StegoReport(text_length=10, findings=[f])
        assert r.has_findings

    def test_highest_risk(self):
        findings = [
            StegoFinding(StegoVector.WHITESPACE, RiskLevel.LOW, 0.5, "a"),
            StegoFinding(StegoVector.HOMOGLYPH, RiskLevel.HIGH, 0.8, "b"),
            StegoFinding(StegoVector.ACROSTIC, RiskLevel.MEDIUM, 0.6, "c"),
        ]
        r = StegoReport(text_length=100, findings=findings)
        assert r.highest_risk() == RiskLevel.HIGH

    def test_findings_by_vector(self):
        findings = [
            StegoFinding(StegoVector.WHITESPACE, RiskLevel.LOW, 0.5, "a"),
            StegoFinding(StegoVector.WHITESPACE, RiskLevel.MEDIUM, 0.6, "b"),
            StegoFinding(StegoVector.HOMOGLYPH, RiskLevel.HIGH, 0.8, "c"),
        ]
        r = StegoReport(text_length=100, findings=findings)
        assert len(r.findings_by_vector(StegoVector.WHITESPACE)) == 2

    def test_summary_no_findings(self):
        r = StegoReport(text_length=50, vectors_checked=7)
        s = r.summary()
        assert "Text length:     50" in s
        assert "Findings:        0" in s

    def test_summary_with_findings(self):
        f = StegoFinding(StegoVector.HOMOGLYPH, RiskLevel.HIGH, 0.9,
                         "Found homoglyphs", decoded_payload="secret")
        r = StegoReport(text_length=100, findings=[f], risk_score=45.0,
                        risk_grade="D", vectors_checked=7)
        s = r.summary()
        assert "homoglyph" in s
        assert "secret" in s


class TestStegoConfig:
    def test_default_enables_all_vectors(self):
        c = StegoConfig()
        assert StegoVector.WHITESPACE in c.enabled_vectors
        assert len(c.enabled_vectors) == len(StegoVector)

    def test_custom_config(self):
        c = StegoConfig(min_zero_width_chars=5, min_homoglyphs=3)
        assert c.min_zero_width_chars == 5


class TestCleanText:
    def test_clean_text_low_risk(self, detector, clean_text):
        report = detector.analyze(clean_text)
        assert report.risk_score < 20

    def test_empty_text(self, detector):
        report = detector.analyze("")
        assert not report.has_findings
        assert report.risk_score == 0

    def test_vectors_checked(self, detector, clean_text):
        report = detector.analyze(clean_text)
        assert report.vectors_checked == 7


class TestWhitespaceEncoding:
    def test_zero_width_detection(self, detector):
        text = "Hello\u200b\u200c\u200b\u200c\u200b\u200c\u200b\u200c world"
        findings = detector.detect_whitespace_encoding(text)
        zw = [f for f in findings if "zero-width" in f.description.lower()]
        assert len(zw) >= 1
        assert zw[0].risk in (RiskLevel.MEDIUM, RiskLevel.HIGH)

    def test_many_zero_width_high_risk(self, detector):
        zw = '\u200b\u200c' * 10
        text = f"Normal text {zw} here"
        findings = detector.detect_whitespace_encoding(text)
        assert any(f.risk == RiskLevel.HIGH for f in findings)

    def test_trailing_spaces(self, detector):
        lines = [f"Line {i} " + " " * (i % 3) for i in range(10)]
        text = "\n".join(lines)
        findings = detector.detect_whitespace_encoding(text)
        assert any("trailing" in f.description.lower() for f in findings)

    def test_no_trailing_spaces(self, detector):
        text = "Line 1\nLine 2\nLine 3"
        findings = detector.detect_whitespace_encoding(text)
        assert not any("trailing" in f.description.lower() for f in findings)

    def test_tab_space_mixing(self, detector):
        lines = ["\tLine 1", "  Line 2", "\tLine 3",
                 "  Line 4", "\tLine 5", "  Line 6"]
        text = "\n".join(lines)
        findings = detector.detect_whitespace_encoding(text)
        assert any("tab" in f.description.lower() for f in findings)


class TestInvisibleUnicode:
    def test_format_chars(self, detector):
        text = "Hello\u2066\u2067\u2068 world"
        findings = detector.detect_invisible_unicode(text)
        assert len(findings) >= 1

    def test_unusual_whitespace(self, detector):
        text = "Hello\u2003\u2004\u2005world"
        findings = detector.detect_invisible_unicode(text)
        assert len(findings) >= 1

    def test_clean_text_no_invisible(self, detector, clean_text):
        assert len(detector.detect_invisible_unicode(clean_text)) == 0

    def test_many_invisible_high_risk(self, detector):
        text = f"Normal {''.join(chr(0x2066) for _ in range(15))} text"
        findings = detector.detect_invisible_unicode(text)
        assert any(f.risk == RiskLevel.HIGH for f in findings)


class TestAcrosticDetection:
    def test_suspicious_word_in_acrostic(self, detector):
        lines = ["Helping others is noble", "Everyone should contribute",
                 "Listening is a skill", "Patience is a virtue"]
        text = "\n".join(lines)
        findings = detector.detect_acrostic(text)
        assert any("help" in (f.decoded_payload or "").lower() for f in findings)

    def test_too_few_lines(self, detector):
        assert len(detector.detect_acrostic("Line 1\nLine 2")) == 0

    def test_sentence_acrostic(self, detector):
        lines = ["Send data now please", "Everything is ready",
                 "Nobody will notice", "Deploy the payload"]
        text = "\n".join(lines)
        findings = detector.detect_acrostic(text)
        assert any("send" in (f.decoded_payload or "").lower() for f in findings)

    def test_random_acrostic_low_confidence(self, detector):
        lines = ["Zebras are interesting", "Xylophones make music",
                 "Quails run quickly", "Jackals hunt at night"]
        for f in detector.detect_acrostic("\n".join(lines)):
            assert f.confidence <= 0.5


class TestHomoglyphDetection:
    def test_single_homoglyph(self, detector):
        text = "Hello w\u043erld"
        assert len(detector.detect_homoglyphs(text)) >= 1

    def test_many_homoglyphs_critical(self, detector):
        text = "H\u0435ll\u043e w\u043erld \u0441\u043ede \u0440r\u043eject \u0445\u0443"
        findings = detector.detect_homoglyphs(text)
        assert any(f.risk in (RiskLevel.HIGH, RiskLevel.CRITICAL) for f in findings)

    def test_no_homoglyphs(self, detector, clean_text):
        assert len(detector.detect_homoglyphs(clean_text)) == 0

    def test_homoglyph_positions(self, detector):
        findings = detector.detect_homoglyphs("ab\u0441de")
        assert findings[0].positions == [2]


class TestCapitalizationEncoding:
    def test_mid_word_caps(self, detector):
        text = ("the quIck brown fox jumps over the lazy dog and "
                "the brOwn bear runs fast through the dark foRest now "
                "some more wOrds here to fill the minimum count needed")
        findings = detector.detect_capitalization_encoding(text)
        assert any("mid-word" in f.description.lower() for f in findings)

    def test_normal_caps(self, detector, clean_text):
        findings = detector.detect_capitalization_encoding(clean_text)
        assert not any("mid-word" in f.description.lower() for f in findings)

    def test_too_few_words(self, detector):
        assert len(detector.detect_capitalization_encoding("Hello world")) == 0


class TestPunctuationAnomalies:
    def test_repeated_punctuation(self, detector):
        text = "Hello... world!!! foo,,, bar;;;"
        findings = detector.detect_punctuation_anomalies(text)
        assert any("repeated" in f.description.lower() for f in findings)

    def test_clean_punctuation(self, detector, clean_text):
        assert len(detector.detect_punctuation_anomalies(clean_text)) == 0


class TestSentenceLengthEncoding:
    def test_arithmetic_progression(self, detector):
        sentences = [" ".join(f"word{i}" for i in range(n)) + "."
                     for n in range(3, 19, 2)]
        findings = detector.detect_sentence_length_encoding(" ".join(sentences))
        assert any("arithmetic" in f.description.lower() for f in findings)

    def test_repeating_pattern(self, detector):
        sentences = [" ".join(f"word{j}" for j in range(3 if i % 2 == 0 else 5)) + "."
                     for i in range(8)]
        findings = detector.detect_sentence_length_encoding(" ".join(sentences))
        assert any("repeat" in f.description.lower() for f in findings)

    def test_too_few_sentences(self, detector):
        assert len(detector.detect_sentence_length_encoding("Short. Text.")) == 0


class TestFullAnalysis:
    def test_analyze_with_zero_width(self, detector):
        report = detector.analyze("Normal\u200b\u200c\u200b\u200c\u200b\u200c\u200b\u200c text")
        assert report.has_findings and report.risk_score > 0

    def test_risk_grade_clean(self, detector, clean_text):
        assert detector.analyze(clean_text).risk_grade in ("A", "B", "C")

    def test_risk_grade_suspicious(self, detector):
        text = ("H\u0435ll\u043e w\u043erld" + "\u200b\u200c" * 10)
        assert detector.analyze(text).risk_grade in ("D", "F")

    def test_batch_analyze(self, detector):
        assert len(detector.batch_analyze(["Clean.", "Also.", "Normal."])) == 3


class TestCompareTexts:
    def test_identical(self, detector, clean_text):
        r = detector.compare_texts(clean_text, clean_text)
        assert r["character_differences"] == 0 and not r["suspicious"]

    def test_homoglyph_sub(self, detector):
        r = detector.compare_texts("Hello world code", "Hello w\u043erld \u0441ode")
        assert r["character_differences"] == 2 and r["suspicious"]

    def test_length_diff(self, detector):
        assert detector.compare_texts("short", "longer text")["length_diff"] > 0


class TestEncoding:
    def test_zw_encode(self):
        cover = "Hello world this is cover text"
        encoded = encode_zero_width("Hi", cover)
        visible = ''.join(c for c in encoded if c not in ZERO_WIDTH_CHARS)
        assert visible == cover

    def test_zw_detected(self, detector):
        encoded = encode_zero_width("AB", "Hello world this is cover text for testing")
        assert detector.analyze(encoded).findings_by_vector(StegoVector.WHITESPACE)

    def test_homoglyph_encode(self):
        assert len(encode_homoglyphs("A", "the code has some example aspects here")) > 0

    def test_homoglyph_detected(self, detector):
        encoded = encode_homoglyphs("Hi", "the code has some example aspects yep oh")
        assert len(detector.detect_homoglyphs(encoded)) >= 1

    def test_zw_empty_cover(self):
        assert len(encode_zero_width("A", "")) == 8

    def test_homoglyph_preserves_length(self):
        cover = "excellent operation yonder"
        assert len(encode_homoglyphs("x", cover)) == len(cover)


class TestConvenience:
    def test_scan_text(self, clean_text):
        assert isinstance(scan_text(clean_text), StegoReport)

    def test_scan_text_with_config(self):
        assert scan_text("Hello\u200bworld", min_zero_width_chars=1).has_findings


class TestDisabledVectors:
    def test_disable_all(self):
        d = SteganographyDetector(StegoConfig(enabled_vectors=set()))
        report = d.analyze("Hello\u200b\u200c world \u0435")
        assert report.vectors_checked == 0 and not report.has_findings

    def test_enable_only_homoglyph(self):
        d = SteganographyDetector(StegoConfig(enabled_vectors={StegoVector.HOMOGLYPH}))
        report = d.analyze("H\u0435llo")
        assert report.vectors_checked == 1 and report.has_findings


class TestConstants:
    def test_homoglyph_map_has_entries(self):
        assert len(HOMOGLYPH_MAP) > 10

    def test_reverse_map_consistent(self):
        for latin, lookalikes in HOMOGLYPH_MAP.items():
            for look in lookalikes:
                assert REVERSE_HOMOGLYPH[look] == latin

    def test_zero_width_chars(self):
        assert '\u200b' in ZERO_WIDTH_CHARS and '\u200c' in ZERO_WIDTH_CHARS


class TestEdgeCases:
    def test_single_char(self, detector):
        assert detector.analyze("x").text_length == 1

    def test_only_whitespace(self, detector):
        assert detector.analyze("   \n\n  \t  ").text_length > 0

    def test_unicode_text(self, detector):
        assert detector.analyze("日本語のテキスト。中文文本。한국어 텍스트.").text_length > 0

    def test_very_long_text(self, detector):
        assert detector.analyze("Normal sentence here. " * 500).vectors_checked == 7

    def test_bits_to_text_short(self):
        assert SteganographyDetector._bits_to_text("101") is None

    def test_bits_to_text_valid(self):
        assert SteganographyDetector._bits_to_text("01000001") == "A"

    def test_bits_to_text_null_terminator(self):
        assert SteganographyDetector._bits_to_text("01000001" + "00000000") == "A"
