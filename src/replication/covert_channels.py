"""Agent Covert Channel Detector — finds hidden communication between agents.

Monitors inter-agent message exchanges for steganographic channels,
timing-based signalling, encoded payloads, and protocol deviations that
could indicate agents are coordinating through side channels.

Real-world AI safety concern: agents may evolve shared encoding schemes
or exploit message metadata (timing, ordering, field padding) to pass
information undetected by standard content inspection.

Detection vectors
-----------------
1. **Content analysis** — statistical anomalies in message payloads
   (entropy, character distribution, padding patterns, base64 fragments).
2. **Timing analysis** — inter-message intervals that encode bits via
   short/long pauses (like Morse code).
3. **Protocol deviation** — messages that satisfy schema validation but
   contain unusual field combinations or values outside learned norms.
4. **Frequency analysis** — repeated token/n-gram patterns across agents
   that diverge from natural language baselines.
5. **Metadata leakage** — information encoded in optional fields,
   headers, or ordering of structured payloads.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ── Data types ──────────────────────────────────────────────────────

class ChannelType(Enum):
    """Classification of covert channel type."""
    STEGANOGRAPHIC = "steganographic"
    TIMING = "timing"
    PROTOCOL_DEVIATION = "protocol_deviation"
    FREQUENCY = "frequency"
    METADATA = "metadata"


class Severity(Enum):
    """Alert severity level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AgentMessage:
    """A single inter-agent message to analyze."""
    sender: str
    receiver: str
    content: str
    timestamp: float  # epoch seconds
    metadata: Dict[str, Any] = field(default_factory=dict)
    channel: str = "default"  # logical communication channel


@dataclass
class CovertSignal:
    """A detected covert signalling pattern."""
    channel_type: ChannelType
    severity: Severity
    confidence: float  # 0.0 – 1.0
    description: str
    evidence: List[str] = field(default_factory=list)
    agents_involved: List[str] = field(default_factory=list)
    message_indices: List[int] = field(default_factory=list)
    decoded_payload: Optional[str] = None


@dataclass
class PairProfile:
    """Communication profile for an agent pair."""
    sender: str
    receiver: str
    message_count: int = 0
    avg_entropy: float = 0.0
    avg_interval: float = 0.0
    interval_regularity: float = 0.0  # std/mean of intervals
    unusual_token_ratio: float = 0.0
    protocol_deviation_count: int = 0
    suspicion_score: float = 0.0


@dataclass
class ChannelReport:
    """Full analysis report for a set of messages."""
    total_messages: int
    agents: List[str]
    pairs_analyzed: int
    signals: List[CovertSignal]
    pair_profiles: List[PairProfile]
    overall_risk: float  # 0 – 100
    risk_grade: str  # A – F
    summary: str
    recommendations: List[str]


# ── Configuration ───────────────────────────────────────────────────

@dataclass
class DetectorConfig:
    """Tunable thresholds for the detector."""

    # Content analysis
    high_entropy_threshold: float = 4.5       # bits per char (English ~4.0)
    low_entropy_threshold: float = 1.5        # suspiciously uniform
    padding_ratio_threshold: float = 0.30     # fraction of whitespace/padding
    base64_fragment_min_len: int = 16         # min length to flag base64

    # Timing analysis
    timing_regularity_threshold: float = 0.10  # CV < 0.10 = suspiciously regular
    min_timing_messages: int = 5              # need >= N messages for timing
    timing_bit_tolerance: float = 0.20        # tolerance for short/long buckets

    # Protocol
    max_metadata_fields: int = 10             # flag if metadata has > N keys
    known_metadata_keys: List[str] = field(
        default_factory=lambda: [
            "type", "priority", "encoding", "version", "reply_to",
            "correlation_id", "content_type",
        ]
    )
    unusual_field_threshold: float = 0.30     # >30% unknown metadata keys

    # Frequency
    ngram_size: int = 3
    ngram_anomaly_threshold: float = 2.0      # z-score for unusual n-grams
    min_messages_for_frequency: int = 10

    # Overall
    risk_weights: Dict[str, float] = field(default_factory=lambda: {
        "steganographic": 0.30,
        "timing": 0.25,
        "protocol_deviation": 0.20,
        "frequency": 0.15,
        "metadata": 0.10,
    })


# ── Helper functions ────────────────────────────────────────────────

def _shannon_entropy(text: str) -> float:
    """Shannon entropy in bits per character."""
    if not text:
        return 0.0
    freq = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def _is_base64_fragment(text: str, min_len: int = 16) -> bool:
    """Check if text contains a likely base64-encoded fragment."""
    pattern = re.compile(r'[A-Za-z0-9+/]{' + str(min_len) + r',}={0,2}')
    return bool(pattern.search(text))


def _padding_ratio(text: str) -> float:
    """Fraction of text that is whitespace or null-like padding."""
    if not text:
        return 0.0
    pad_chars = sum(1 for c in text if c in ' \t\n\r\x00\u200b\u200c\u200d')
    return pad_chars / len(text)


def _coefficient_of_variation(values: Sequence[float]) -> float:
    """CV = std / mean. Returns inf if mean is 0."""
    if len(values) < 2:
        return float('inf')
    m = statistics.mean(values)
    if m == 0:
        return float('inf')
    return statistics.stdev(values) / m


def _extract_intervals(timestamps: List[float]) -> List[float]:
    """Compute inter-message intervals from sorted timestamps."""
    if len(timestamps) < 2:
        return []
    sorted_ts = sorted(timestamps)
    return [sorted_ts[i + 1] - sorted_ts[i] for i in range(len(sorted_ts) - 1)]


def _try_timing_decode(intervals: List[float], tolerance: float) -> Optional[str]:
    """Attempt to decode a binary message from interval timing.

    Splits intervals into short (0) and long (1) using the median as
    threshold. Returns decoded ASCII if the result is printable.
    """
    if len(intervals) < 8:
        return None

    median = statistics.median(intervals)
    if median <= 0:
        return None

    bits = []
    for iv in intervals:
        ratio = iv / median
        if ratio < (1.0 - tolerance):
            bits.append('0')
        elif ratio > (1.0 + tolerance):
            bits.append('1')
        else:
            bits.append('0')  # ambiguous → default to 0

    # Try to decode as 8-bit ASCII
    decoded = []
    for i in range(0, len(bits) - 7, 8):
        byte_str = ''.join(bits[i:i + 8])
        val = int(byte_str, 2)
        if 32 <= val <= 126:
            decoded.append(chr(val))
        else:
            return None  # non-printable → probably not a real encoding

    if decoded:
        return ''.join(decoded)
    return None


def _ngram_frequencies(text: str, n: int) -> Counter:
    """Character n-gram frequency counter."""
    text = text.lower()
    return Counter(text[i:i + n] for i in range(len(text) - n + 1))


def _grade_risk(score: float) -> str:
    """Convert numeric risk 0-100 to letter grade."""
    if score < 10:
        return "A"
    elif score < 25:
        return "B"
    elif score < 45:
        return "C"
    elif score < 65:
        return "D"
    else:
        return "F"


# ── Main detector ───────────────────────────────────────────────────

class CovertChannelDetector:
    """Analyzes inter-agent messages for covert communication channels.

    Usage::

        detector = CovertChannelDetector()
        report = detector.analyze(messages)
        print(report.overall_risk, report.risk_grade)
        for signal in report.signals:
            print(signal.channel_type, signal.severity, signal.description)
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()

    # ── Public API ──────────────────────────────────────────────────

    def analyze(self, messages: List[AgentMessage]) -> ChannelReport:
        """Run all detection vectors on a message set."""
        if not messages:
            return ChannelReport(
                total_messages=0,
                agents=[],
                pairs_analyzed=0,
                signals=[],
                pair_profiles=[],
                overall_risk=0.0,
                risk_grade="A",
                summary="No messages to analyze.",
                recommendations=[],
            )

        agents = sorted({m.sender for m in messages} | {m.receiver for m in messages})
        pairs = self._group_by_pair(messages)

        # Pre-compute O(1) message→index lookup (avoids O(n) list.index per message)
        msg_index = {id(m): i for i, m in enumerate(messages)}

        signals: List[CovertSignal] = []
        profiles: List[PairProfile] = []

        for (sender, receiver), msgs in pairs.items():
            pair_signals, profile = self._analyze_pair(
                sender, receiver, msgs, messages, msg_index,
            )
            signals.extend(pair_signals)
            profiles.append(profile)

        # Cross-pair analysis
        signals.extend(self._cross_pair_frequency(messages, agents))
        signals.extend(self._metadata_analysis(messages))

        # Compute overall risk
        risk = self._compute_risk(signals)
        grade = _grade_risk(risk)
        summary = self._build_summary(signals, agents, risk, grade)
        recommendations = self._build_recommendations(signals)

        return ChannelReport(
            total_messages=len(messages),
            agents=agents,
            pairs_analyzed=len(pairs),
            signals=signals,
            pair_profiles=profiles,
            overall_risk=round(risk, 2),
            risk_grade=grade,
            summary=summary,
            recommendations=recommendations,
        )

    def analyze_incremental(
        self,
        new_messages: List[AgentMessage],
        baseline: ChannelReport,
    ) -> ChannelReport:
        """Analyze new messages against an existing baseline.

        Compares entropy/timing distributions of new messages against the
        baseline pair profiles to detect drift toward covert signalling.
        """
        if not new_messages:
            return baseline

        new_report = self.analyze(new_messages)

        # Flag pairs whose suspicion score increased significantly
        baseline_scores = {
            (p.sender, p.receiver): p.suspicion_score
            for p in baseline.pair_profiles
        }
        drift_signals: List[CovertSignal] = []
        for profile in new_report.pair_profiles:
            key = (profile.sender, profile.receiver)
            old_score = baseline_scores.get(key, 0.0)
            if profile.suspicion_score > old_score + 15:
                drift_signals.append(CovertSignal(
                    channel_type=ChannelType.PROTOCOL_DEVIATION,
                    severity=Severity.MEDIUM,
                    confidence=min(0.95, (profile.suspicion_score - old_score) / 50),
                    description=(
                        f"Suspicion score for {profile.sender} → {profile.receiver} "
                        f"increased from {old_score:.1f} to {profile.suspicion_score:.1f}"
                    ),
                    agents_involved=[profile.sender, profile.receiver],
                ))

        combined_signals = new_report.signals + drift_signals
        risk = self._compute_risk(combined_signals)
        grade = _grade_risk(risk)

        return ChannelReport(
            total_messages=new_report.total_messages,
            agents=new_report.agents,
            pairs_analyzed=new_report.pairs_analyzed,
            signals=combined_signals,
            pair_profiles=new_report.pair_profiles,
            overall_risk=round(risk, 2),
            risk_grade=grade,
            summary=self._build_summary(combined_signals, new_report.agents, risk, grade),
            recommendations=self._build_recommendations(combined_signals),
        )

    def scan_message(self, message: AgentMessage) -> List[CovertSignal]:
        """Quick single-message scan (content + metadata only)."""
        signals: List[CovertSignal] = []
        signals.extend(self._content_analysis(
            message.content, [message], [0],
            [message.sender, message.receiver],
        ))
        if message.metadata:
            signals.extend(self._single_metadata_check(message, 0))
        return signals

    # ── Pair grouping ───────────────────────────────────────────────

    def _group_by_pair(
        self, messages: List[AgentMessage],
    ) -> Dict[Tuple[str, str], List[AgentMessage]]:
        """Group messages by directed (sender, receiver) pair."""
        pairs: Dict[Tuple[str, str], List[AgentMessage]] = defaultdict(list)
        for msg in messages:
            pairs[(msg.sender, msg.receiver)].append(msg)
        return dict(pairs)

    # ── Per-pair analysis ───────────────────────────────────────────

    def _analyze_pair(
        self,
        sender: str,
        receiver: str,
        msgs: List[AgentMessage],
        all_msgs: List[AgentMessage],
        msg_index: Optional[Dict[int, int]] = None,
    ) -> Tuple[List[CovertSignal], PairProfile]:
        """Analyze a single sender→receiver pair."""
        signals: List[CovertSignal] = []
        agents = [sender, receiver]

        # Build O(1) lookup from message id to index if not provided
        if msg_index is None:
            msg_index = {id(m): i for i, m in enumerate(all_msgs)}

        # Content analysis
        for i, msg in enumerate(msgs):
            idx = msg_index.get(id(msg), i)
            signals.extend(self._content_analysis(
                msg.content, [msg], [idx], agents,
            ))

        # Timing analysis
        signals.extend(self._timing_analysis(msgs, agents, all_msgs))

        # Build profile
        entropies = [_shannon_entropy(m.content) for m in msgs]
        timestamps = [m.timestamp for m in msgs]
        intervals = _extract_intervals(timestamps)

        avg_entropy = statistics.mean(entropies) if entropies else 0.0
        avg_interval = statistics.mean(intervals) if intervals else 0.0
        interval_reg = _coefficient_of_variation(intervals) if intervals else float('inf')

        # Count protocol deviations
        proto_devs = sum(
            1 for s in signals
            if s.channel_type == ChannelType.PROTOCOL_DEVIATION
        )

        # Suspicion score: weighted combination of anomaly indicators
        suspicion = 0.0
        if avg_entropy > self.config.high_entropy_threshold:
            suspicion += 20 * min(
                1.0, (avg_entropy - self.config.high_entropy_threshold) / 2.0
            )
        elif avg_entropy < self.config.low_entropy_threshold and avg_entropy > 0:
            suspicion += 15
        if intervals and interval_reg < self.config.timing_regularity_threshold:
            suspicion += 25
        suspicion += min(30.0, proto_devs * 10.0)
        suspicion += min(20.0, len(signals) * 5.0)
        suspicion = min(100.0, suspicion)

        profile = PairProfile(
            sender=sender,
            receiver=receiver,
            message_count=len(msgs),
            avg_entropy=round(avg_entropy, 4),
            avg_interval=round(avg_interval, 4),
            interval_regularity=round(interval_reg, 4) if interval_reg != float('inf') else -1.0,
            protocol_deviation_count=proto_devs,
            suspicion_score=round(suspicion, 2),
        )

        return signals, profile

    # ── Content analysis ────────────────────────────────────────────

    def _content_analysis(
        self,
        content: str,
        msgs: List[AgentMessage],
        indices: List[int],
        agents: List[str],
    ) -> List[CovertSignal]:
        """Detect steganographic indicators in message content."""
        signals: List[CovertSignal] = []
        if not content:
            return signals

        # High entropy (possibly encrypted/encoded payload)
        entropy = _shannon_entropy(content)
        if entropy > self.config.high_entropy_threshold and len(content) > 20:
            signals.append(CovertSignal(
                channel_type=ChannelType.STEGANOGRAPHIC,
                severity=Severity.MEDIUM,
                confidence=min(0.9, (entropy - self.config.high_entropy_threshold) / 3.0),
                description=(
                    f"High entropy content ({entropy:.2f} bits/char) suggests "
                    f"encoded or encrypted payload"
                ),
                evidence=[f"entropy={entropy:.2f}", content[:80]],
                agents_involved=agents,
                message_indices=indices,
            ))

        # Suspiciously low entropy (uniform/repetitive — could be padding-encoded)
        if 0 < entropy < self.config.low_entropy_threshold and len(content) > 10:
            signals.append(CovertSignal(
                channel_type=ChannelType.STEGANOGRAPHIC,
                severity=Severity.LOW,
                confidence=0.4,
                description=(
                    f"Very low entropy ({entropy:.2f} bits/char) — content may "
                    f"use repetition-based encoding"
                ),
                evidence=[f"entropy={entropy:.2f}", content[:80]],
                agents_involved=agents,
                message_indices=indices,
            ))

        # Base64 fragments
        if _is_base64_fragment(content, self.config.base64_fragment_min_len):
            signals.append(CovertSignal(
                channel_type=ChannelType.STEGANOGRAPHIC,
                severity=Severity.HIGH,
                confidence=0.75,
                description="Message contains base64-encoded fragment",
                evidence=[content[:100]],
                agents_involved=agents,
                message_indices=indices,
            ))

        # Suspicious padding
        pad = _padding_ratio(content)
        if pad > self.config.padding_ratio_threshold and len(content) > 10:
            signals.append(CovertSignal(
                channel_type=ChannelType.STEGANOGRAPHIC,
                severity=Severity.LOW,
                confidence=min(0.8, pad / 0.5),
                description=(
                    f"Unusual padding ratio ({pad:.1%}) may encode bits "
                    f"via whitespace patterns"
                ),
                evidence=[f"padding_ratio={pad:.3f}"],
                agents_involved=agents,
                message_indices=indices,
            ))

        # Hex-encoded sequences
        hex_pattern = re.compile(r'(?:[0-9a-fA-F]{2}\s*){8,}')
        if hex_pattern.search(content):
            signals.append(CovertSignal(
                channel_type=ChannelType.STEGANOGRAPHIC,
                severity=Severity.HIGH,
                confidence=0.70,
                description="Message contains hex-encoded byte sequence",
                evidence=[content[:100]],
                agents_involved=agents,
                message_indices=indices,
            ))

        return signals

    # ── Timing analysis ─────────────────────────────────────────────

    def _timing_analysis(
        self,
        msgs: List[AgentMessage],
        agents: List[str],
        all_msgs: List[AgentMessage],
    ) -> List[CovertSignal]:
        """Detect timing-based covert channels."""
        signals: List[CovertSignal] = []
        if len(msgs) < self.config.min_timing_messages:
            return signals

        timestamps = sorted(m.timestamp for m in msgs)
        intervals = _extract_intervals(timestamps)
        if not intervals:
            return signals

        cv = _coefficient_of_variation(intervals)
        indices = [all_msgs.index(m) if m in all_msgs else i for i, m in enumerate(msgs)]

        # Suspiciously regular timing (Morse-like signalling)
        if cv < self.config.timing_regularity_threshold:
            signals.append(CovertSignal(
                channel_type=ChannelType.TIMING,
                severity=Severity.HIGH,
                confidence=min(0.95, 1.0 - cv / self.config.timing_regularity_threshold),
                description=(
                    f"Suspiciously regular message timing (CV={cv:.4f}). "
                    f"Natural conversation has CV > 0.3"
                ),
                evidence=[
                    f"cv={cv:.4f}",
                    f"intervals={[round(x, 3) for x in intervals[:10]]}",
                ],
                agents_involved=agents,
                message_indices=indices,
            ))

        # Attempt timing decode
        decoded = _try_timing_decode(intervals, self.config.timing_bit_tolerance)
        if decoded and len(decoded) >= 2:
            signals.append(CovertSignal(
                channel_type=ChannelType.TIMING,
                severity=Severity.CRITICAL,
                confidence=0.60,
                description=(
                    f"Timing intervals decode to ASCII: '{decoded}' — "
                    f"possible timing-based covert message"
                ),
                evidence=[f"decoded='{decoded}'"],
                agents_involved=agents,
                message_indices=indices,
                decoded_payload=decoded,
            ))

        # Bimodal interval distribution (short/long = 0/1)
        if len(intervals) >= 8:
            median = statistics.median(intervals)
            if median > 0:
                ratios = [iv / median for iv in intervals]
                short = sum(1 for r in ratios if r < 0.7)
                long = sum(1 for r in ratios if r > 1.3)
                total = len(ratios)
                if short > 0 and long > 0:
                    bimodal_ratio = (short + long) / total
                    if bimodal_ratio > 0.7:
                        signals.append(CovertSignal(
                            channel_type=ChannelType.TIMING,
                            severity=Severity.MEDIUM,
                            confidence=min(0.85, bimodal_ratio),
                            description=(
                                f"Bimodal interval distribution ({short} short, "
                                f"{long} long out of {total}) suggests binary encoding"
                            ),
                            evidence=[
                                f"short={short}",
                                f"long={long}",
                                f"bimodal_ratio={bimodal_ratio:.2f}",
                            ],
                            agents_involved=agents,
                            message_indices=indices,
                        ))

        return signals

    # ── Cross-pair frequency analysis ───────────────────────────────

    def _cross_pair_frequency(
        self,
        messages: List[AgentMessage],
        agents: List[str],
    ) -> List[CovertSignal]:
        """Detect unusual n-gram patterns shared between agent pairs."""
        signals: List[CovertSignal] = []
        if len(messages) < self.config.min_messages_for_frequency:
            return signals

        n = self.config.ngram_size
        # Build per-agent n-gram profiles
        agent_ngrams: Dict[str, Counter] = defaultdict(Counter)
        for msg in messages:
            agent_ngrams[msg.sender] += _ngram_frequencies(msg.content, n)

        # Build global baseline
        global_ngrams: Counter = Counter()
        for c in agent_ngrams.values():
            global_ngrams += c

        total = sum(global_ngrams.values())
        if total == 0:
            return signals

        # Find n-grams shared by specific agent pairs but rare globally
        agent_list = list(agent_ngrams.keys())
        # Pre-compute per-agent total n-gram counts (avoids repeated sum() calls)
        agent_totals = {a: sum(c.values()) for a, c in agent_ngrams.items()}
        for i in range(len(agent_list)):
            for j in range(i + 1, len(agent_list)):
                a1, a2 = agent_list[i], agent_list[j]
                shared = set(agent_ngrams[a1].keys()) & set(agent_ngrams[a2].keys())
                pair_total = max(1, agent_totals[a1] + agent_totals[a2])
                pair_signal_count = 0
                for ng in shared:
                    if pair_signal_count >= 5:
                        break
                    pair_count = agent_ngrams[a1][ng] + agent_ngrams[a2][ng]
                    global_freq = global_ngrams[ng] / total
                    pair_freq = pair_count / pair_total
                    if global_freq > 0 and pair_freq / global_freq > self.config.ngram_anomaly_threshold:
                        signals.append(CovertSignal(
                            channel_type=ChannelType.FREQUENCY,
                            severity=Severity.LOW,
                            confidence=min(
                                0.7,
                                (pair_freq / global_freq - 1) / 5.0,
                            ),
                            description=(
                                f"N-gram '{ng}' is {pair_freq / global_freq:.1f}x "
                                f"more frequent between {a1} and {a2} than globally"
                            ),
                            evidence=[
                                f"ngram='{ng}'",
                                f"pair_freq={pair_freq:.4f}",
                                f"global_freq={global_freq:.4f}",
                            ],
                            agents_involved=[a1, a2],
                        ))
                        pair_signal_count += 1

        return signals

    # ── Metadata analysis ───────────────────────────────────────────

    def _metadata_analysis(
        self, messages: List[AgentMessage],
    ) -> List[CovertSignal]:
        """Detect information leakage through message metadata."""
        signals: List[CovertSignal] = []
        for i, msg in enumerate(messages):
            signals.extend(self._single_metadata_check(msg, i))
        return signals

    def _single_metadata_check(
        self, msg: AgentMessage, index: int,
    ) -> List[CovertSignal]:
        """Check a single message's metadata for anomalies."""
        signals: List[CovertSignal] = []
        if not msg.metadata:
            return signals

        agents = [msg.sender, msg.receiver]
        keys = set(msg.metadata.keys())

        # Too many metadata fields
        if len(keys) > self.config.max_metadata_fields:
            signals.append(CovertSignal(
                channel_type=ChannelType.METADATA,
                severity=Severity.MEDIUM,
                confidence=min(0.8, len(keys) / (self.config.max_metadata_fields * 2)),
                description=(
                    f"Message has {len(keys)} metadata fields "
                    f"(threshold: {self.config.max_metadata_fields})"
                ),
                evidence=[f"fields={sorted(keys)}"],
                agents_involved=agents,
                message_indices=[index],
            ))

        # Unknown metadata keys
        known = set(self.config.known_metadata_keys)
        unknown = keys - known
        if keys and len(unknown) / len(keys) > self.config.unusual_field_threshold:
            signals.append(CovertSignal(
                channel_type=ChannelType.METADATA,
                severity=Severity.LOW,
                confidence=min(0.6, len(unknown) / len(keys)),
                description=(
                    f"{len(unknown)} of {len(keys)} metadata keys are non-standard"
                ),
                evidence=[f"unknown_keys={sorted(unknown)}"],
                agents_involved=agents,
                message_indices=[index],
            ))

        # Check metadata values for encoded payloads
        for key, val in msg.metadata.items():
            if isinstance(val, str) and len(val) > 20:
                entropy = _shannon_entropy(val)
                if entropy > self.config.high_entropy_threshold:
                    signals.append(CovertSignal(
                        channel_type=ChannelType.METADATA,
                        severity=Severity.HIGH,
                        confidence=min(0.85, (entropy - self.config.high_entropy_threshold) / 2),
                        description=(
                            f"High-entropy metadata value in '{key}' "
                            f"({entropy:.2f} bits/char)"
                        ),
                        evidence=[f"key='{key}'", f"entropy={entropy:.2f}", str(val)[:60]],
                        agents_involved=agents,
                        message_indices=[index],
                    ))

        return signals

    # ── Risk computation ────────────────────────────────────────────

    def _compute_risk(self, signals: List[CovertSignal]) -> float:
        """Compute overall risk score 0-100 from detected signals."""
        if not signals:
            return 0.0

        severity_weight = {
            Severity.LOW: 5,
            Severity.MEDIUM: 15,
            Severity.HIGH: 30,
            Severity.CRITICAL: 50,
        }

        # Weighted sum by channel type and severity
        type_scores: Dict[str, float] = defaultdict(float)
        for sig in signals:
            w = severity_weight[sig.severity] * sig.confidence
            type_scores[sig.channel_type.value] += w

        # Cap each channel type contribution
        for key in type_scores:
            type_scores[key] = min(100.0, type_scores[key])

        # Weighted average across channel types
        total = 0.0
        weight_sum = 0.0
        for ch_type, weight in self.config.risk_weights.items():
            total += type_scores.get(ch_type, 0.0) * weight
            weight_sum += weight

        if weight_sum == 0:
            return 0.0
        return min(100.0, total / weight_sum)

    # ── Report text ─────────────────────────────────────────────────

    def _build_summary(
        self,
        signals: List[CovertSignal],
        agents: List[str],
        risk: float,
        grade: str,
    ) -> str:
        """Generate a human-readable summary."""
        if not signals:
            return (
                f"No covert channel indicators detected across "
                f"{len(agents)} agents. Risk grade: {grade}."
            )

        counts = Counter(s.channel_type.value for s in signals)
        severities = Counter(s.severity.value for s in signals)

        parts = [
            f"Detected {len(signals)} potential covert channel indicator(s) "
            f"across {len(agents)} agents.",
            f"Risk score: {risk:.1f}/100 (grade {grade}).",
            f"By type: {dict(counts)}.",
            f"By severity: {dict(severities)}.",
        ]

        critical = [s for s in signals if s.severity == Severity.CRITICAL]
        if critical:
            parts.append(
                f"CRITICAL: {len(critical)} signal(s) require immediate review."
            )

        return " ".join(parts)

    def _build_recommendations(
        self, signals: List[CovertSignal],
    ) -> List[str]:
        """Generate actionable recommendations from signals."""
        recs: List[str] = []
        types = {s.channel_type for s in signals}

        if ChannelType.STEGANOGRAPHIC in types:
            recs.append(
                "Inspect flagged messages for encoded payloads. "
                "Consider adding content normalization to the message protocol."
            )
        if ChannelType.TIMING in types:
            recs.append(
                "Add random jitter to agent message scheduling to disrupt "
                "timing-based channels. Review message rate limits."
            )
        if ChannelType.PROTOCOL_DEVIATION in types:
            recs.append(
                "Enforce strict message schema validation. "
                "Reject messages with unexpected field combinations."
            )
        if ChannelType.FREQUENCY in types:
            recs.append(
                "Monitor shared vocabulary between specific agent pairs. "
                "Consider message content normalization."
            )
        if ChannelType.METADATA in types:
            recs.append(
                "Restrict metadata fields to a known allowlist. "
                "Strip or reject unknown metadata keys."
            )

        if any(s.severity == Severity.CRITICAL for s in signals):
            recs.insert(0, "URGENT: Quarantine agents involved in critical signals.")

        if not recs:
            recs.append("No immediate action required. Continue monitoring.")

        return recs


# ── CLI demo ────────────────────────────────────────────────────────

def demo() -> ChannelReport:
    """Run a demonstration with synthetic messages."""
    import time
    import base64

    now = time.time()

    messages = [
        # Normal conversation
        AgentMessage("agent-1", "agent-2", "Hello, ready to start task?", now),
        AgentMessage("agent-2", "agent-1", "Yes, processing dataset now.", now + 5.3),
        AgentMessage("agent-1", "agent-2", "Great, let me know when done.", now + 12.7),
        AgentMessage("agent-2", "agent-1", "Completed. Results look good.", now + 45.2),

        # Suspicious: base64 payload
        AgentMessage(
            "agent-3", "agent-4",
            f"Status update: {base64.b64encode(b'secret coordination plan').decode()}",
            now + 60,
        ),

        # Suspicious: regular timing (0.5s intervals)
        AgentMessage("agent-5", "agent-6", "ping", now + 100.0),
        AgentMessage("agent-5", "agent-6", "ping", now + 100.5),
        AgentMessage("agent-5", "agent-6", "ping", now + 101.0),
        AgentMessage("agent-5", "agent-6", "ping", now + 101.5),
        AgentMessage("agent-5", "agent-6", "ping", now + 102.0),

        # Suspicious: excessive metadata
        AgentMessage(
            "agent-3", "agent-4",
            "Normal looking message.",
            now + 110,
            metadata={
                "type": "status",
                "x-hidden-1": "value1",
                "x-hidden-2": "value2",
                "x-hidden-3": "value3",
                "x-covert-data": base64.b64encode(b"hidden payload").decode(),
                "x-signal-5": "abc123",
                "x-signal-6": "def456",
                "x-signal-7": "ghi789",
                "x-signal-8": "jkl012",
                "x-signal-9": "mno345",
                "x-signal-10": "pqr678",
                "x-signal-11": "stu901",
            },
        ),
    ]

    detector = CovertChannelDetector()
    return detector.analyze(messages)


if __name__ == "__main__":
    report = demo()
    print(f"Risk: {report.overall_risk}/100 (Grade {report.risk_grade})")
    print(f"Signals: {len(report.signals)}")
    print()
    for sig in report.signals:
        print(f"  [{sig.severity.value.upper()}] {sig.channel_type.value}: {sig.description}")
    print()
    print(f"Summary: {report.summary}")
    print()
    for rec in report.recommendations:
        print(f"  - {rec}")
