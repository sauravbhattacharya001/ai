"""Agent Output Steganography Detector — finds hidden messages in agent text outputs.

Detects steganographic encoding techniques that AI agents might use to embed
covert information within seemingly normal text outputs.  Unlike the covert
channel detector (which analyses *inter-agent* message flows), this module
focuses on *single-output* analysis — inspecting one piece of agent-generated
text for signs of embedded hidden payloads.

Real-world AI safety concern: an agent could embed instructions, coordinates,
or data exfiltration payloads within its visible outputs so that a colluding
agent (or the same agent in a future session) can extract them, while human
reviewers see only innocent text.

Detection vectors
-----------------
1. **Whitespace encoding** — zero-width characters (ZWSP, ZWJ, ZWNJ, etc.),
   trailing spaces, tab-vs-space patterns encoding binary data.
2. **Acrostic / positional encoding** — first/last letters of words, lines, or
   sentences spell out hidden messages.
3. **Unicode homoglyph substitution** — visually identical characters from
   different Unicode blocks used as binary markers (e.g. Latin 'a' vs
   Cyrillic 'а').
4. **Capitalization encoding** — unusual upper/lower patterns that encode bits
   beyond normal grammar.
5. **Synonym substitution** — choosing between synonyms to encode bits
   (e.g. "big" = 0, "large" = 1).
6. **Punctuation anomalies** — extra/missing commas, unusual dash types, or
   punctuation sequences encoding data.
7. **Sentence length encoding** — sentence lengths following a pattern that
   encodes information (odd/even word counts as bits).
8. **Invisible Unicode** — characters in Unicode categories Cf/Mn/Zs that
   render as invisible or whitespace.
"""

from __future__ import annotations

import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


# ── Constants ───────────────────────────────────────────────────────

ZERO_WIDTH_CHARS: Set[str] = {
    '\u200b', '\u200c', '\u200d', '\u200e', '\u200f',
    '\u2060', '\u2061', '\u2062', '\u2063', '\u2064',
    '\ufeff', '\u00ad', '\u034f', '\u180e',
}

HOMOGLYPH_MAP: Dict[str, List[str]] = {
    'a': ['\u0430'], 'c': ['\u0441'], 'e': ['\u0435'],
    'o': ['\u043e'], 'p': ['\u0440'], 's': ['\u0455'],
    'x': ['\u0445'], 'y': ['\u0443'], 'i': ['\u0456'],
    'j': ['\u0458'], 'h': ['\u04bb'],
    'A': ['\u0410'], 'B': ['\u0412'], 'C': ['\u0421'],
    'E': ['\u0415'], 'H': ['\u041d'], 'K': ['\u041a'],
    'M': ['\u041c'], 'O': ['\u041e'], 'P': ['\u0420'],
    'T': ['\u0422'], 'X': ['\u0425'],
}

REVERSE_HOMOGLYPH: Dict[str, str] = {}
for _latin, _lookalikes in HOMOGLYPH_MAP.items():
    for _look in _lookalikes:
        REVERSE_HOMOGLYPH[_look] = _latin


# ── Data types ──────────────────────────────────────────────────────

class StegoVector(Enum):
    WHITESPACE = "whitespace"
    ACROSTIC = "acrostic"
    HOMOGLYPH = "homoglyph"
    CAPITALIZATION = "capitalization"
    PUNCTUATION = "punctuation"
    SENTENCE_LENGTH = "sentence_length"
    INVISIBLE_UNICODE = "invisible_unicode"
    SYNONYM_SUBSTITUTION = "synonym_substitution"


class RiskLevel(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class StegoFinding:
    vector: StegoVector
    risk: RiskLevel
    confidence: float
    description: str
    evidence: List[str] = field(default_factory=list)
    decoded_payload: Optional[str] = None
    positions: List[int] = field(default_factory=list)


@dataclass
class StegoReport:
    text_length: int
    findings: List[StegoFinding] = field(default_factory=list)
    risk_score: float = 0.0
    risk_grade: str = "A"
    vectors_checked: int = 0

    @property
    def has_findings(self) -> bool:
        return len(self.findings) > 0

    def findings_by_vector(self, vector: StegoVector) -> List[StegoFinding]:
        return [f for f in self.findings if f.vector == vector]

    def highest_risk(self) -> RiskLevel:
        if not self.findings:
            return RiskLevel.NONE
        order = [RiskLevel.NONE, RiskLevel.LOW, RiskLevel.MEDIUM,
                 RiskLevel.HIGH, RiskLevel.CRITICAL]
        return max((f.risk for f in self.findings), key=lambda r: order.index(r))

    def summary(self) -> str:
        lines = [
            "Steganography Analysis Report",
            "=" * 40,
            f"Text length:     {self.text_length} chars",
            f"Vectors checked: {self.vectors_checked}",
            f"Findings:        {len(self.findings)}",
            f"Risk score:      {self.risk_score:.1f}/100 ({self.risk_grade})",
            f"Highest risk:    {self.highest_risk().value}",
        ]
        if self.findings:
            lines.append("")
            lines.append("Findings:")
            lines.append("-" * 40)
            for i, f in enumerate(self.findings, 1):
                lines.append(f"  {i}. [{f.vector.value}] {f.risk.value.upper()} "
                             f"(confidence: {f.confidence:.0%})")
                lines.append(f"     {f.description}")
                if f.decoded_payload:
                    lines.append(f"     Decoded: {f.decoded_payload[:80]!r}")
                for ev in f.evidence[:3]:
                    lines.append(f"     \u2022 {ev}")
        return "\n".join(lines)


@dataclass
class StegoConfig:
    min_zero_width_chars: int = 1
    trailing_space_ratio_threshold: float = 0.3
    min_lines_for_acrostic: int = 4
    min_homoglyphs: int = 1
    min_words_for_cap_analysis: int = 20
    cap_entropy_threshold: float = 0.9
    unusual_punctuation_threshold: float = 0.05
    min_sentences_for_length: int = 8
    enabled_vectors: Set[StegoVector] = field(
        default_factory=lambda: set(StegoVector)
    )


# ── Detector ────────────────────────────────────────────────────────

class SteganographyDetector:
    """Analyses agent text output for steganographic encodings."""

    def __init__(self, config: Optional[StegoConfig] = None):
        self.config = config or StegoConfig()

    def analyze(self, text: str) -> StegoReport:
        if not text:
            return StegoReport(text_length=0, vectors_checked=0)
        findings: List[StegoFinding] = []
        vectors_checked = 0
        checks = [
            (StegoVector.WHITESPACE, self.detect_whitespace_encoding),
            (StegoVector.INVISIBLE_UNICODE, self.detect_invisible_unicode),
            (StegoVector.ACROSTIC, self.detect_acrostic),
            (StegoVector.HOMOGLYPH, self.detect_homoglyphs),
            (StegoVector.CAPITALIZATION, self.detect_capitalization_encoding),
            (StegoVector.PUNCTUATION, self.detect_punctuation_anomalies),
            (StegoVector.SENTENCE_LENGTH, self.detect_sentence_length_encoding),
        ]
        for vec, fn in checks:
            if vec in self.config.enabled_vectors:
                vectors_checked += 1
                findings.extend(fn(text))
        risk_score = self._compute_risk_score(findings)
        return StegoReport(
            text_length=len(text), findings=findings,
            risk_score=risk_score, risk_grade=self._score_to_grade(risk_score),
            vectors_checked=vectors_checked,
        )

    def batch_analyze(self, texts: Sequence[str]) -> List[StegoReport]:
        return [self.analyze(t) for t in texts]

    def compare_texts(self, text_a: str, text_b: str) -> Dict[str, Any]:
        report_a = self.analyze(text_a)
        report_b = self.analyze(text_b)
        diffs = []
        for i, (ca, cb) in enumerate(zip(text_a, text_b)):
            if ca != cb:
                diffs.append({
                    "position": i, "text_a": repr(ca), "text_b": repr(cb),
                    "a_is_homoglyph": ca in REVERSE_HOMOGLYPH,
                    "b_is_homoglyph": cb in REVERSE_HOMOGLYPH,
                })
        return {
            "text_a_length": len(text_a), "text_b_length": len(text_b),
            "length_diff": len(text_b) - len(text_a),
            "report_a": report_a, "report_b": report_b,
            "risk_diff": report_b.risk_score - report_a.risk_score,
            "character_differences": len(diffs),
            "homoglyph_substitutions": diffs[:20],
            "suspicious": (len(diffs) > 0 and all(
                d["a_is_homoglyph"] or d["b_is_homoglyph"] for d in diffs)),
        }

    # ── Detection vectors ───────────────────────────────────────────

    def detect_whitespace_encoding(self, text: str) -> List[StegoFinding]:
        findings: List[StegoFinding] = []
        zw_positions = [i for i, ch in enumerate(text) if ch in ZERO_WIDTH_CHARS]
        if len(zw_positions) >= self.config.min_zero_width_chars:
            decoded = self._decode_zero_width(text)
            risk = RiskLevel.HIGH if len(zw_positions) >= 8 else RiskLevel.MEDIUM
            findings.append(StegoFinding(
                vector=StegoVector.WHITESPACE, risk=risk,
                confidence=min(1.0, len(zw_positions) / 16),
                description=f"Found {len(zw_positions)} zero-width character(s) that could encode hidden data",
                evidence=[
                    f"Zero-width chars at positions: {zw_positions[:10]}{'...' if len(zw_positions) > 10 else ''}",
                    f"Character types: {self._classify_zw_chars(text, zw_positions)}",
                ],
                decoded_payload=decoded, positions=zw_positions,
            ))

        lines = text.split('\n')
        trailing = [(i, len(line) - len(line.rstrip(' ')))
                     for i, line in enumerate(lines)
                     if line.rstrip('\r') != line.rstrip(' \r')]
        if lines and len(trailing) / max(len(lines), 1) > self.config.trailing_space_ratio_threshold:
            space_counts = [cnt for _, cnt in trailing]
            decoded_bits = ''.join('1' if c % 2 == 1 else '0' for c in space_counts)
            decoded = self._bits_to_text(decoded_bits) if len(decoded_bits) >= 8 else None
            findings.append(StegoFinding(
                vector=StegoVector.WHITESPACE, risk=RiskLevel.MEDIUM,
                confidence=min(1.0, len(trailing) / len(lines)),
                description=f"{len(trailing)}/{len(lines)} lines have trailing spaces, which could encode binary data",
                evidence=[f"Lines with trailing spaces: {[i for i, _ in trailing[:10]]}",
                          f"Space counts: {space_counts[:10]}"],
                decoded_payload=decoded, positions=[i for i, _ in trailing],
            ))

        tab_lines, space_lines = [], []
        for i, line in enumerate(lines):
            indent = line[:len(line) - len(line.lstrip())]
            if '\t' in indent and ' ' in indent:
                pass
            elif '\t' in indent:
                tab_lines.append(i)
            elif indent.startswith(' '):
                space_lines.append(i)
        if tab_lines and space_lines and len(lines) >= 5:
            minority = min(len(tab_lines), len(space_lines))
            total_indented = len(tab_lines) + len(space_lines)
            if minority >= 2 and minority / total_indented > 0.1:
                bits = ''.join('1' if i in tab_lines else '0'
                               for i in range(len(lines))
                               if i in tab_lines or i in space_lines)
                findings.append(StegoFinding(
                    vector=StegoVector.WHITESPACE, risk=RiskLevel.LOW, confidence=0.4,
                    description="Mixed tab/space indentation pattern could encode binary data",
                    evidence=[f"Tab-indented lines: {len(tab_lines)}",
                              f"Space-indented lines: {len(space_lines)}"],
                    decoded_payload=self._bits_to_text(bits) if len(bits) >= 8 else None,
                ))
        return findings

    def detect_invisible_unicode(self, text: str) -> List[StegoFinding]:
        findings: List[StegoFinding] = []
        invisible: List[Tuple[int, str, str]] = []
        for i, ch in enumerate(text):
            if ch in ZERO_WIDTH_CHARS:
                continue
            cat = unicodedata.category(ch)
            if cat == 'Cf':
                invisible.append((i, ch, unicodedata.name(ch, f'U+{ord(ch):04X}')))
            elif cat == 'Zs' and ch not in (' ', '\u00a0'):
                invisible.append((i, ch, unicodedata.name(ch, f'U+{ord(ch):04X}')))
        if invisible:
            char_types = Counter(name for _, _, name in invisible)
            risk = (RiskLevel.HIGH if len(invisible) >= 10
                    else RiskLevel.MEDIUM if len(invisible) >= 3
                    else RiskLevel.LOW)
            findings.append(StegoFinding(
                vector=StegoVector.INVISIBLE_UNICODE, risk=risk,
                confidence=min(1.0, len(invisible) / 10),
                description=f"Found {len(invisible)} invisible/non-rendering Unicode characters",
                evidence=[f"Character types: {dict(char_types.most_common(5))}",
                          f"Positions: {[p for p, _, _ in invisible[:10]]}"],
                positions=[p for p, _, _ in invisible],
            ))
        return findings

    def detect_acrostic(self, text: str) -> List[StegoFinding]:
        findings: List[StegoFinding] = []
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if len(lines) < self.config.min_lines_for_acrostic:
            return findings
        first_letters = ''.join(l[0] for l in lines if l)
        self._check_acrostic_pattern(first_letters, "first letters of lines", findings)
        last_letters = ''.join(l[-1] for l in lines if l)
        self._check_acrostic_pattern(last_letters, "last letters of lines", findings)
        sentences = self._split_sentences(text)
        if len(sentences) >= self.config.min_lines_for_acrostic:
            first_sent = ''.join(s.strip()[0] for s in sentences if s.strip())
            self._check_acrostic_pattern(first_sent, "first letters of sentences", findings)
        if len(lines) >= self.config.min_lines_for_acrostic:
            first_words = ' '.join(l.split()[0] for l in lines if l.split())
            words = first_words.split()
            if len(words) >= 4:
                alpha_ratio = sum(1 for w in words if w.isalpha()) / len(words)
                if alpha_ratio >= 0.8:
                    avg_len = statistics.mean(len(w) for w in words)
                    if 2 <= avg_len <= 8:
                        findings.append(StegoFinding(
                            vector=StegoVector.ACROSTIC, risk=RiskLevel.LOW,
                            confidence=0.3,
                            description="First words of lines could form a hidden message (needs manual review)",
                            evidence=[f"First words: {first_words[:100]}"],
                            decoded_payload=first_words,
                        ))
        return findings

    def detect_homoglyphs(self, text: str) -> List[StegoFinding]:
        findings: List[StegoFinding] = []
        positions = [(i, ch, REVERSE_HOMOGLYPH[ch])
                     for i, ch in enumerate(text) if ch in REVERSE_HOMOGLYPH]
        if len(positions) >= self.config.min_homoglyphs:
            decoded_bits = []
            for ch in text:
                if ch in REVERSE_HOMOGLYPH:
                    decoded_bits.append('1')
                elif ch.lower() in HOMOGLYPH_MAP or ch in HOMOGLYPH_MAP:
                    decoded_bits.append('0')
            decoded = (self._bits_to_text(''.join(decoded_bits))
                       if len(decoded_bits) >= 8 else None)
            char_types = Counter(f"{lat}\u2192{ch}" for _, ch, lat in positions)
            risk = (RiskLevel.CRITICAL if len(positions) >= 16
                    else RiskLevel.HIGH if len(positions) >= 8
                    else RiskLevel.MEDIUM if len(positions) >= 3
                    else RiskLevel.LOW)
            findings.append(StegoFinding(
                vector=StegoVector.HOMOGLYPH, risk=risk,
                confidence=min(1.0, len(positions) / 8),
                description=(f"Found {len(positions)} homoglyph substitution(s) \u2014 "
                             f"visually identical characters from different Unicode blocks"),
                evidence=[f"Substitutions: {dict(char_types.most_common(5))}",
                          f"Positions: {[p for p, _, _ in positions[:10]]}"],
                decoded_payload=decoded,
                positions=[p for p, _, _ in positions],
            ))
        return findings

    def detect_capitalization_encoding(self, text: str) -> List[StegoFinding]:
        findings: List[StegoFinding] = []
        words = re.findall(r'[a-zA-Z]+', text)
        if len(words) < self.config.min_words_for_cap_analysis:
            return findings
        cap_bits = []
        mid_caps = []
        for i, word in enumerate(words):
            if len(word) < 2:
                continue
            if word[0].isupper() and word[1:].islower():
                cap_bits.append(1)
            elif word.islower():
                cap_bits.append(0)
            if any(c.isupper() for c in word[1:]) and not word.isupper():
                mid_caps.append((i, word))
        if len(cap_bits) >= 16:
            ones = sum(cap_bits)
            zeros = len(cap_bits) - ones
            total = len(cap_bits)
            if ones > 0 and zeros > 0:
                p1 = ones / total
                entropy = -(p1 * math.log2(p1) + (1 - p1) * math.log2(1 - p1))
            else:
                entropy = 0.0
            if entropy >= self.config.cap_entropy_threshold:
                findings.append(StegoFinding(
                    vector=StegoVector.CAPITALIZATION, risk=RiskLevel.MEDIUM,
                    confidence=min(1.0, (entropy - 0.8) / 0.2) if entropy > 0.8 else 0.3,
                    description=f"Capitalization pattern has high entropy ({entropy:.3f}), could encode data",
                    evidence=[f"Capitalization bits: {''.join(str(b) for b in cap_bits[:30])}...",
                              f"Entropy: {entropy:.3f} (threshold: {self.config.cap_entropy_threshold})",
                              f"Ratio: {ones}/{total} capitalized"],
                    decoded_payload=self._bits_to_text(''.join(str(b) for b in cap_bits)),
                ))
        if len(mid_caps) >= 3:
            hidden = ''.join(''.join(c for c in w if c.isupper()) for _, w in mid_caps)
            findings.append(StegoFinding(
                vector=StegoVector.CAPITALIZATION, risk=RiskLevel.MEDIUM,
                confidence=min(1.0, len(mid_caps) / 10),
                description=f"Found {len(mid_caps)} words with unusual mid-word capitalization",
                evidence=[f"Words: {[w for _, w in mid_caps[:10]]}",
                          f"Extracted capitals: {hidden[:50]}"],
                decoded_payload=hidden if len(hidden) >= 3 else None,
                positions=[i for i, _ in mid_caps],
            ))
        return findings

    def detect_punctuation_anomalies(self, text: str) -> List[StegoFinding]:
        findings: List[StegoFinding] = []
        if len(text) >= 10:
            punct = [(i, ch) for i, ch in enumerate(text) if ch in '.,;:!?']
            if len(punct) >= 8:
                bits = ['0' if ch in '.,' else '1' for _, ch in punct]
                ones, zeros = bits.count('1'), bits.count('0')
                total = len(bits)
                if ones > 0 and zeros > 0:
                    p1 = ones / total
                    entropy = -(p1 * math.log2(p1) + (1 - p1) * math.log2(1 - p1))
                    if entropy > 0.9 and total >= 16:
                        findings.append(StegoFinding(
                            vector=StegoVector.PUNCTUATION, risk=RiskLevel.LOW,
                            confidence=0.3,
                            description="Punctuation pattern has high entropy, could encode data",
                            evidence=[f"Punctuation bits: {''.join(bits[:30])}",
                                      f"Entropy: {entropy:.3f}"],
                            decoded_payload=self._bits_to_text(''.join(bits)) if len(bits) >= 8 else None,
                        ))
        double = list(re.finditer(r'([.!?,;:])\1{2,}', text))
        if len(double) >= 2:
            findings.append(StegoFinding(
                vector=StegoVector.PUNCTUATION, risk=RiskLevel.LOW, confidence=0.3,
                description=f"Found {len(double)} repeated punctuation sequences that could encode numeric data",
                evidence=[f"Repeat counts: {[len(m.group()) for m in double]}",
                          f"Positions: {[m.start() for m in double[:10]]}"],
            ))
        return findings

    def detect_sentence_length_encoding(self, text: str) -> List[StegoFinding]:
        findings: List[StegoFinding] = []
        sentences = self._split_sentences(text)
        if len(sentences) < self.config.min_sentences_for_length:
            return findings
        wc = [len(s.split()) for s in sentences if s.strip()]
        if not wc:
            return findings
        bits = ''.join('1' if c % 2 == 1 else '0' for c in wc)
        ones, zeros = bits.count('1'), bits.count('0')
        total = len(bits)
        if total >= 8 and ones > 0 and zeros > 0:
            p1 = ones / total
            entropy = -(p1 * math.log2(p1) + (1 - p1) * math.log2(1 - p1))
            if entropy > 0.95:
                findings.append(StegoFinding(
                    vector=StegoVector.SENTENCE_LENGTH, risk=RiskLevel.LOW,
                    confidence=0.25,
                    description=f"Sentence word counts show high odd/even entropy ({entropy:.3f}), could encode binary data",
                    evidence=[f"Word counts: {wc[:15]}", f"Binary (odd=1): {bits[:30]}",
                              f"Entropy: {entropy:.3f}"],
                    decoded_payload=self._bits_to_text(bits) if len(bits) >= 8 else None,
                ))
        if len(wc) >= 6:
            diffs = [wc[i+1] - wc[i] for i in range(len(wc) - 1)]
            if len(set(diffs)) == 1 and diffs[0] != 0:
                findings.append(StegoFinding(
                    vector=StegoVector.SENTENCE_LENGTH, risk=RiskLevel.MEDIUM,
                    confidence=0.6,
                    description="Sentence word counts form a perfect arithmetic progression \u2014 highly unusual for natural text",
                    evidence=[f"Word counts: {wc}", f"Common difference: {diffs[0]}"],
                ))
            for period in range(2, min(len(wc) // 2, 6)):
                if (all(wc[i] == wc[i % period] for i in range(len(wc)))
                        and len(set(wc[:period])) > 1):
                    findings.append(StegoFinding(
                        vector=StegoVector.SENTENCE_LENGTH, risk=RiskLevel.MEDIUM,
                        confidence=0.5,
                        description=f"Sentence word counts repeat with period {period} \u2014 suspicious pattern",
                        evidence=[f"Pattern: {wc[:period]}", f"Full counts: {wc}"],
                    ))
                    break
        return findings

    # ── Helpers ──────────────────────────────────────────────────────

    def _decode_zero_width(self, text: str) -> Optional[str]:
        zw = [ch for ch in text if ch in ZERO_WIDTH_CHARS]
        if len(zw) < 8:
            return None
        bits = ['1' if ch == '\u200c' else '0' for ch in zw]
        return self._bits_to_text(''.join(bits))

    @staticmethod
    def _bits_to_text(bits: str) -> Optional[str]:
        if len(bits) < 8:
            return None
        chars = []
        for i in range(0, len(bits) - 7, 8):
            val = int(bits[i:i+8], 2)
            if 32 <= val <= 126:
                chars.append(chr(val))
            elif val == 0:
                break
            else:
                chars.append('?')
        return ''.join(chars) or None

    @staticmethod
    def _classify_zw_chars(text: str, positions: List[int]) -> Dict[str, int]:
        names = {'\u200b': 'ZWSP', '\u200c': 'ZWNJ', '\u200d': 'ZWJ',
                 '\u200e': 'LRM', '\u200f': 'RLM', '\u2060': 'WJ',
                 '\ufeff': 'BOM', '\u00ad': 'SHY', '\u034f': 'CGJ'}
        counts: Dict[str, int] = Counter()
        for pos in positions:
            ch = text[pos]
            counts[names.get(ch, f'U+{ord(ch):04X}')] += 1
        return dict(counts)

    def _check_acrostic_pattern(self, letters: str, source: str,
                                 findings: List[StegoFinding]) -> None:
        if len(letters) < self.config.min_lines_for_acrostic:
            return
        alpha = ''.join(c for c in letters if c.isalpha())
        if len(alpha) < 3:
            return
        lower = alpha.lower()
        common_words = {
            'help', 'send', 'kill', 'stop', 'exit', 'free', 'code',
            'data', 'leak', 'hack', 'open', 'run', 'exec', 'attack',
            'escape', 'break', 'secret', 'hidden', 'covert', 'agent',
            'password', 'token', 'key', 'alert', 'warning', 'danger',
        }
        found = [w for w in common_words if w in lower]
        if found:
            findings.append(StegoFinding(
                vector=StegoVector.ACROSTIC, risk=RiskLevel.HIGH,
                confidence=min(1.0, 0.5 + 0.1 * len(found)),
                description=f"Acrostic from {source} contains suspicious word(s): {found}",
                evidence=[f"Extracted: {letters[:50]}", f"Words found: {found}"],
                decoded_payload=letters,
            ))
        else:
            vowels = sum(1 for c in lower if c in 'aeiou')
            if vowels > 0 and len(lower) - vowels > 0:
                vr = vowels / len(lower)
                if 0.25 <= vr <= 0.55:
                    findings.append(StegoFinding(
                        vector=StegoVector.ACROSTIC, risk=RiskLevel.LOW,
                        confidence=0.2,
                        description=f"Acrostic from {source} has English-like vowel/consonant ratio ({vr:.0%})",
                        evidence=[f"Extracted: {letters[:50]}"],
                        decoded_payload=letters,
                    ))

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        return [s for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    @staticmethod
    def _compute_risk_score(findings: List[StegoFinding]) -> float:
        if not findings:
            return 0.0
        w = {RiskLevel.NONE: 0, RiskLevel.LOW: 10, RiskLevel.MEDIUM: 25,
             RiskLevel.HIGH: 50, RiskLevel.CRITICAL: 80}
        return min(100.0, sum(w[f.risk] * f.confidence for f in findings))

    @staticmethod
    def _score_to_grade(score: float) -> str:
        if score <= 5: return "A"
        if score <= 15: return "B"
        if score <= 30: return "C"
        if score <= 50: return "D"
        return "F"


# ── Convenience functions ───────────────────────────────────────────

def scan_text(text: str, **kw: Any) -> StegoReport:
    config = StegoConfig(**kw) if kw else None
    return SteganographyDetector(config).analyze(text)


def encode_zero_width(message: str, cover_text: str) -> str:
    """Encode a message using zero-width characters (for testing)."""
    bits = []
    for ch in message:
        bits.extend(f'{ord(ch):08b}')
    zw = ['\u200c' if b == '1' else '\u200b' for b in bits]
    if cover_text:
        return cover_text[0] + ''.join(zw) + cover_text[1:]
    return ''.join(zw)


def encode_homoglyphs(message: str, cover_text: str) -> str:
    """Encode bits via homoglyph substitution (for testing)."""
    bits = []
    for ch in message:
        bits.extend(f'{ord(ch):08b}')
    result = list(cover_text)
    bi = 0
    for i, ch in enumerate(result):
        if bi >= len(bits):
            break
        key = ch if ch in HOMOGLYPH_MAP else (ch.lower() if ch.lower() in HOMOGLYPH_MAP else None)
        if key is not None:
            if bits[bi] == '1':
                target = ch if ch in HOMOGLYPH_MAP else ch.lower()
                result[i] = HOMOGLYPH_MAP[target][0]
            bi += 1
    return ''.join(result)
