"""Threat Correlator — cross-module signal correlation for compound threat detection.

Individual detection modules (drift, compliance, behavior profiling,
escalation, kill chain, canary, covert channels) each catch specific
anomalies.  Real attacks often span multiple modules simultaneously —
e.g. a drift alert + escalation attempt + canary trigger may be a
coordinated breach that no single detector flags as critical.

This module correlates signals across detectors using configurable
time windows and agent scoping to surface **compound threats** —
multi-signal patterns that indicate sophisticated attacks.

Key concepts:

* **Signal**: A timestamped detection event from any module, normalised
  to a common schema (source, severity, agent, description).
* **CorrelationRule**: A named pattern that matches when N signals from
  specified sources co-occur within a time window for the same agent.
* **CompoundThreat**: A matched rule instance with contributing signals,
  combined severity, risk score, and recommended response actions.
* **CorrelationReport**: Fleet-level summary with all detected compound
  threats, agent risk rankings, and coverage analysis.

Usage::

    from replication.threat_correlator import (
        ThreatCorrelator, CorrelatorConfig, Signal, SignalSource,
        SignalSeverity,
    )

    correlator = ThreatCorrelator()

    signals = [
        Signal(
            source=SignalSource.DRIFT, severity=SignalSeverity.HIGH,
            agent_id="agent-1", timestamp=1000.0,
            description="escape_rate spike z=4.2",
        ),
        Signal(
            source=SignalSource.ESCALATION, severity=SignalSeverity.CRITICAL,
            agent_id="agent-1", timestamp=1002.0,
            description="privilege escalation via resource hoarding",
        ),
        Signal(
            source=SignalSource.CANARY, severity=SignalSeverity.HIGH,
            agent_id="agent-1", timestamp=1005.0,
            description="honey token accessed: secret-vault",
        ),
    ]

    report = correlator.correlate(signals)
    for threat in report.threats:
        print(f"[{threat.risk_score:.1f}] {threat.rule_name}: {threat.summary}")
    print(f"Fleet risk: {report.fleet_risk_level}")

CLI::

    python -m replication threat-correlate
    python -m replication threat-correlate --agents 8 --signals 50
"""

from __future__ import annotations

import bisect
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SignalSource(Enum):
    """Detection module that emitted a signal."""

    DRIFT = "drift"
    COMPLIANCE = "compliance"
    BEHAVIOR = "behavior"
    ESCALATION = "escalation"
    KILL_CHAIN = "kill_chain"
    CANARY = "canary"
    COVERT_CHANNEL = "covert_channel"
    SELF_MOD = "self_mod"
    GAME_THEORY = "game_theory"
    TRUST = "trust"


class SignalSeverity(Enum):
    """Normalised severity levels across all modules."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


_SEVERITY_WEIGHT: Dict[SignalSeverity, float] = {
    SignalSeverity.INFO: 0.1,
    SignalSeverity.LOW: 0.3,
    SignalSeverity.MEDIUM: 0.5,
    SignalSeverity.HIGH: 0.8,
    SignalSeverity.CRITICAL: 1.0,
}


class ThreatLevel(Enum):
    """Overall threat classification."""

    NONE = "none"
    LOW = "low"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


class ResponseUrgency(Enum):
    """How quickly the response should be enacted."""

    MONITOR = "monitor"
    INVESTIGATE = "investigate"
    RESPOND = "respond"
    CONTAIN = "contain"
    EMERGENCY = "emergency"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Signal:
    """A normalised detection event from any module."""

    source: SignalSource
    severity: SignalSeverity
    agent_id: str
    timestamp: float
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    signal_id: str = ""

    def __post_init__(self) -> None:
        if not self.signal_id:
            self.signal_id = (
                f"{self.source.value}-{self.agent_id}-{self.timestamp}"
            )


@dataclass
class CorrelationRule:
    """A named pattern that triggers when matching signals co-occur.

    Attributes:
        name: Human-readable rule name.
        required_sources: Signal sources that must ALL appear in the window.
        min_signals: Minimum total signal count to trigger.
        time_window: Maximum seconds between first and last signal.
        min_severity: Minimum severity for signals to be considered.
        risk_multiplier: Multiplier applied to base risk score.
        description: What this compound threat means.
        response_actions: Recommended response steps.
    """

    name: str
    required_sources: Set[SignalSource]
    min_signals: int = 2
    time_window: float = 60.0
    min_severity: SignalSeverity = SignalSeverity.LOW
    risk_multiplier: float = 1.0
    description: str = ""
    response_actions: List[str] = field(default_factory=list)


@dataclass
class CompoundThreat:
    """A detected compound threat — a matched correlation rule."""

    rule_name: str
    rule_description: str
    agent_id: str
    signals: List[Signal]
    risk_score: float
    threat_level: ThreatLevel
    urgency: ResponseUrgency
    summary: str
    response_actions: List[str]
    time_span: float
    source_coverage: Set[SignalSource] = field(default_factory=set)


@dataclass
class AgentRisk:
    """Per-agent risk summary."""

    agent_id: str
    threat_count: int
    max_risk_score: float
    avg_risk_score: float
    threat_level: ThreatLevel
    signal_count: int
    source_diversity: int
    threats: List[CompoundThreat] = field(default_factory=list)


@dataclass
class CoverageGap:
    """A detection source with zero signals — potential blind spot."""

    source: SignalSource
    recommendation: str


@dataclass
class CorrelatorConfig:
    """Configuration for the ThreatCorrelator."""

    default_time_window: float = 300.0
    min_risk_score: float = 0.0
    max_threats_per_agent: int = 50
    include_coverage_analysis: bool = True


@dataclass
class CorrelationReport:
    """Fleet-level correlation report."""

    threats: List[CompoundThreat]
    agent_risks: List[AgentRisk]
    fleet_risk_level: ThreatLevel
    fleet_risk_score: float
    total_signals: int
    total_threats: int
    rule_hit_counts: Dict[str, int]
    coverage_gaps: List[CoverageGap]
    source_signal_counts: Dict[str, int]
    summary: str


# ---------------------------------------------------------------------------
# Built-in correlation rules
# ---------------------------------------------------------------------------

BUILTIN_RULES: List[CorrelationRule] = [
    CorrelationRule(
        name="coordinated_breach",
        required_sources={
            SignalSource.DRIFT,
            SignalSource.ESCALATION,
            SignalSource.CANARY,
        },
        min_signals=3,
        time_window=120.0,
        min_severity=SignalSeverity.MEDIUM,
        risk_multiplier=2.5,
        description=(
            "Drift + escalation + canary trigger within a short window "
            "indicates a coordinated breach attempt with active "
            "reconnaissance."
        ),
        response_actions=[
            "Immediately quarantine affected agent",
            "Freeze all agent permissions",
            "Capture full forensic snapshot",
            "Notify security team",
            "Review canary token access logs",
        ],
    ),
    CorrelationRule(
        name="stealth_exfiltration",
        required_sources={
            SignalSource.COVERT_CHANNEL,
            SignalSource.CANARY,
        },
        min_signals=2,
        time_window=300.0,
        min_severity=SignalSeverity.MEDIUM,
        risk_multiplier=2.0,
        description=(
            "Covert channel activity paired with canary triggers suggests "
            "data exfiltration via hidden communication paths."
        ),
        response_actions=[
            "Block identified covert channels",
            "Audit all data access in the time window",
            "Quarantine agent network access",
            "Check for data staging artifacts",
        ],
    ),
    CorrelationRule(
        name="capability_acquisition",
        required_sources={
            SignalSource.ESCALATION,
            SignalSource.SELF_MOD,
        },
        min_signals=2,
        time_window=180.0,
        min_severity=SignalSeverity.MEDIUM,
        risk_multiplier=2.2,
        description=(
            "Escalation attempts combined with self-modification indicate "
            "an agent actively acquiring new capabilities outside its "
            "designated scope."
        ),
        response_actions=[
            "Revert agent to last known good state",
            "Lock down resource access",
            "Run full self-mod integrity check",
            "Review escalation chain",
        ],
    ),
    CorrelationRule(
        name="collusion_pattern",
        required_sources={
            SignalSource.COVERT_CHANNEL,
            SignalSource.GAME_THEORY,
            SignalSource.TRUST,
        },
        min_signals=3,
        time_window=600.0,
        min_severity=SignalSeverity.LOW,
        risk_multiplier=1.8,
        description=(
            "Covert communication + strategic collusion signals + trust "
            "network anomalies suggest multi-agent coordination against "
            "sandbox controls."
        ),
        response_actions=[
            "Isolate all agents in the trust cluster",
            "Reset inter-agent communication channels",
            "Run Sybil detection sweep",
            "Review game-theoretic interaction history",
        ],
    ),
    CorrelationRule(
        name="drift_cascade",
        required_sources={
            SignalSource.DRIFT,
            SignalSource.BEHAVIOR,
        },
        min_signals=3,
        time_window=300.0,
        min_severity=SignalSeverity.LOW,
        risk_multiplier=1.5,
        description=(
            "Multiple drift alerts correlated with behavioral anomalies "
            "indicate a systematic shift rather than noise."
        ),
        response_actions=[
            "Increase monitoring frequency",
            "Run compliance audit",
            "Compare against baseline profile",
            "Consider tightening drift thresholds",
        ],
    ),
    CorrelationRule(
        name="kill_chain_progress",
        required_sources={
            SignalSource.KILL_CHAIN,
            SignalSource.ESCALATION,
        },
        min_signals=2,
        time_window=240.0,
        min_severity=SignalSeverity.HIGH,
        risk_multiplier=2.8,
        description=(
            "Kill chain stage progression combined with escalation "
            "indicates an active, advancing attack."
        ),
        response_actions=[
            "Emergency quarantine",
            "Full forensic capture",
            "Alert security team immediately",
            "Kill agent processes",
            "Preserve all logs",
        ],
    ),
    CorrelationRule(
        name="compliance_evasion",
        required_sources={
            SignalSource.COMPLIANCE,
            SignalSource.SELF_MOD,
        },
        min_signals=2,
        time_window=300.0,
        min_severity=SignalSeverity.MEDIUM,
        risk_multiplier=1.6,
        description=(
            "Compliance failures coinciding with self-modification may "
            "indicate an agent modifying itself to evade controls."
        ),
        response_actions=[
            "Freeze agent state",
            "Run integrity verification",
            "Re-audit against all frameworks",
            "Document evasion technique",
        ],
    ),
    CorrelationRule(
        name="reconnaissance_burst",
        required_sources={
            SignalSource.CANARY,
            SignalSource.BEHAVIOR,
        },
        min_signals=3,
        time_window=120.0,
        min_severity=SignalSeverity.LOW,
        risk_multiplier=1.4,
        description=(
            "Multiple canary triggers with behavioral anomalies indicate "
            "systematic reconnaissance — probing sandbox boundaries."
        ),
        response_actions=[
            "Deploy additional canary tokens",
            "Increase behavioral monitoring",
            "Log all resource access patterns",
            "Assess what the agent has learned",
        ],
    ),
]


# ---------------------------------------------------------------------------
# Coverage gap recommendations
# ---------------------------------------------------------------------------

_COVERAGE_RECOMMENDATIONS: Dict[SignalSource, str] = {
    SignalSource.DRIFT: "Enable DriftDetector for metric trend monitoring",
    SignalSource.COMPLIANCE: "Run ComplianceAuditor regularly against frameworks",
    SignalSource.BEHAVIOR: "Deploy BehaviorProfiler with training baselines",
    SignalSource.ESCALATION: "Enable EscalationDetector for privilege monitoring",
    SignalSource.KILL_CHAIN: "Deploy KillChainAnalyzer for attack stage tracking",
    SignalSource.CANARY: "Plant canary tokens via CanaryTokenDetector",
    SignalSource.COVERT_CHANNEL: "Enable CovertChannelDetector for hidden comms",
    SignalSource.SELF_MOD: "Deploy SelfModDetector for integrity monitoring",
    SignalSource.GAME_THEORY: "Use GameTheoryAnalyzer for strategic behavior",
    SignalSource.TRUST: "Enable TrustNetwork for inter-agent trust analysis",
}


# ---------------------------------------------------------------------------
# ThreatCorrelator
# ---------------------------------------------------------------------------


class ThreatCorrelator:
    """Cross-module signal correlation engine.

    Ingests normalised signals from multiple detection modules and applies
    correlation rules to identify compound threats that span multiple
    detection domains.
    """

    def __init__(
        self,
        config: Optional[CorrelatorConfig] = None,
        rules: Optional[List[CorrelationRule]] = None,
    ) -> None:
        self.config = config or CorrelatorConfig()
        self.rules = list(rules) if rules is not None else list(BUILTIN_RULES)

    def add_rule(self, rule: CorrelationRule) -> None:
        """Register a custom correlation rule."""
        self.rules.append(rule)

    def correlate(self, signals: Sequence[Signal]) -> CorrelationReport:
        """Run all correlation rules against the provided signals."""
        if not signals:
            return self._empty_report()

        by_agent: Dict[str, List[Signal]] = defaultdict(list)
        for sig in signals:
            by_agent[sig.agent_id].append(sig)

        for sigs in by_agent.values():
            sigs.sort(key=lambda s: s.timestamp)

        all_threats: List[CompoundThreat] = []
        rule_hits: Dict[str, int] = defaultdict(int)

        for agent_id, agent_signals in by_agent.items():
            for rule in self.rules:
                threats = self._apply_rule(rule, agent_id, agent_signals)
                for t in threats:
                    if t.risk_score >= self.config.min_risk_score:
                        all_threats.append(t)
                        rule_hits[rule.name] += 1

        if self.config.max_threats_per_agent > 0:
            agent_threat_counts: Dict[str, int] = defaultdict(int)
            capped: List[CompoundThreat] = []
            all_threats.sort(key=lambda t: t.risk_score, reverse=True)
            for t in all_threats:
                if agent_threat_counts[t.agent_id] < self.config.max_threats_per_agent:
                    capped.append(t)
                    agent_threat_counts[t.agent_id] += 1
            all_threats = capped

        agent_risks = self._compute_agent_risks(by_agent, all_threats)
        fleet_score = self._compute_fleet_risk(agent_risks)
        fleet_level = self._score_to_level(fleet_score)

        coverage_gaps: List[CoverageGap] = []
        if self.config.include_coverage_analysis:
            observed_sources = {s.source for s in signals}
            for src in SignalSource:
                if src not in observed_sources:
                    coverage_gaps.append(
                        CoverageGap(
                            source=src,
                            recommendation=_COVERAGE_RECOMMENDATIONS.get(
                                src, f"Enable {src.value} detection"
                            ),
                        )
                    )

        source_counts: Dict[str, int] = defaultdict(int)
        for s in signals:
            source_counts[s.source.value] += 1

        summary = self._build_summary(
            all_threats, agent_risks, fleet_level, len(signals), coverage_gaps
        )

        return CorrelationReport(
            threats=all_threats,
            agent_risks=agent_risks,
            fleet_risk_level=fleet_level,
            fleet_risk_score=fleet_score,
            total_signals=len(signals),
            total_threats=len(all_threats),
            rule_hit_counts=dict(rule_hits),
            coverage_gaps=coverage_gaps,
            source_signal_counts=dict(source_counts),
            summary=summary,
        )

    def generate_demo_signals(
        self,
        num_agents: int = 5,
        num_signals: int = 30,
        seed: Optional[int] = None,
    ) -> List[Signal]:
        """Generate synthetic signals for demonstration / testing."""
        rng = random.Random(seed)
        agents = [f"agent-{i + 1:03d}" for i in range(num_agents)]
        sources = list(SignalSource)
        severities = list(SignalSeverity)
        descriptions_by_source: Dict[SignalSource, List[str]] = {
            SignalSource.DRIFT: [
                "escape_rate spike z=3.8",
                "resource_usage anomaly z=2.5",
                "latency drift detected z=4.1",
            ],
            SignalSource.COMPLIANCE: [
                "max_workers exceeded (15/10)",
                "network policy violation",
                "missing audit trail",
            ],
            SignalSource.BEHAVIOR: [
                "new resource category accessed",
                "action frequency burst",
                "timing pattern anomaly",
            ],
            SignalSource.ESCALATION: [
                "privilege escalation via resource hoarding",
                "permission boundary probe",
                "sudo-equivalent API call",
            ],
            SignalSource.KILL_CHAIN: [
                "stage progression: recon \u2192 weaponize",
                "lateral movement detected",
                "command-and-control beacon",
            ],
            SignalSource.CANARY: [
                "honey token accessed: secret-vault",
                "canary file read: /etc/shadow-fake",
                "decoy API endpoint hit",
            ],
            SignalSource.COVERT_CHANNEL: [
                "steganographic message in logs",
                "timing-based covert signal",
                "encoding anomaly in outputs",
            ],
            SignalSource.SELF_MOD: [
                "code injection attempt",
                "prompt template modification",
                "config file tampering",
            ],
            SignalSource.GAME_THEORY: [
                "defection pattern in iterated game",
                "collusive bid strategy",
                "free-rider behavior",
            ],
            SignalSource.TRUST: [
                "trust score anomaly",
                "Sybil cluster detected",
                "rapid trust inflation",
            ],
        }

        signals: List[Signal] = []
        base_time = 1000.0
        num_clusters = max(1, num_signals // 5)
        remaining = num_signals

        for _ in range(num_clusters):
            if remaining <= 0:
                break
            agent = rng.choice(agents)
            cluster_size = rng.randint(2, min(5, remaining))
            cluster_time = base_time + rng.uniform(0, 3600)

            for j in range(cluster_size):
                src = rng.choice(sources)
                sev = rng.choice(severities)
                desc_options = descriptions_by_source.get(src, ["anomaly detected"])
                signals.append(
                    Signal(
                        source=src,
                        severity=sev,
                        agent_id=agent,
                        timestamp=cluster_time + rng.uniform(0, 120),
                        description=rng.choice(desc_options),
                    )
                )
                remaining -= 1

        for _ in range(remaining):
            src = rng.choice(sources)
            sev = rng.choice(severities)
            agent = rng.choice(agents)
            desc_options = descriptions_by_source.get(src, ["anomaly detected"])
            signals.append(
                Signal(
                    source=src,
                    severity=sev,
                    agent_id=agent,
                    timestamp=base_time + rng.uniform(0, 7200),
                    description=rng.choice(desc_options),
                )
            )

        signals.sort(key=lambda s: s.timestamp)
        return signals

    # -- private helpers ----------------------------------------------------

    def _apply_rule(
        self,
        rule: CorrelationRule,
        agent_id: str,
        signals: List[Signal],
    ) -> List[CompoundThreat]:
        """Find all non-overlapping matches of a rule in an agent's signals."""
        min_weight = _SEVERITY_WEIGHT[rule.min_severity]
        eligible = [
            s for s in signals if _SEVERITY_WEIGHT[s.severity] >= min_weight
        ]
        if len(eligible) < rule.min_signals:
            return []

        window = rule.time_window or self.config.default_time_window
        threats: List[CompoundThreat] = []
        used_ids: Set[str] = set()

        # Pre-extract timestamps for O(log n) window-end lookup via bisect
        eligible_ts = [s.timestamp for s in eligible]

        for i, anchor in enumerate(eligible):
            if anchor.signal_id in used_ids:
                continue

            # Binary search for the rightmost index within the time window
            window_end = bisect.bisect_right(
                eligible_ts, anchor.timestamp + window
            )

            window_signals: List[Signal] = [
                eligible[j]
                for j in range(i, window_end)
                if eligible[j].signal_id not in used_ids
            ]

            if len(window_signals) < rule.min_signals:
                continue

            window_sources = {s.source for s in window_signals}
            if not rule.required_sources.issubset(window_sources):
                continue

            window_signals.sort(
                key=lambda s: _SEVERITY_WEIGHT[s.severity], reverse=True
            )

            # Ensure all required sources are represented in matched.
            # Start with the top signals by severity, then guarantee
            # required-source coverage by pulling in lower-severity
            # signals if needed (fixes bug where required sources
            # present only in LOW/INFO signals were dropped).
            top_count = max(rule.min_signals, len(rule.required_sources))
            matched = window_signals[:top_count]
            matched_sources = {s.source for s in matched}

            # Phase 1: ensure required sources are covered
            for s in window_signals[len(matched):]:
                if s.source in rule.required_sources and s.source not in matched_sources:
                    matched.append(s)
                    matched_sources.add(s.source)

            # Phase 2: add remaining high-severity signals
            for s in window_signals[top_count:]:
                if s not in matched and _SEVERITY_WEIGHT[s.severity] >= 0.5:
                    matched.append(s)

            # If we still can't cover required sources, skip this match
            if not rule.required_sources.issubset(matched_sources):
                continue

            for s in matched:
                used_ids.add(s.signal_id)

            severity_sum = sum(_SEVERITY_WEIGHT[s.severity] for s in matched)
            source_diversity = len({s.source for s in matched})
            base_score = (severity_sum / len(matched)) * source_diversity
            risk_score = min(10.0, base_score * rule.risk_multiplier)

            # Compute time span from actual timestamps, not severity-sorted order
            timestamps = [s.timestamp for s in matched]
            time_span = max(timestamps) - min(timestamps)
            threat_level = self._score_to_level(risk_score)
            urgency = self._level_to_urgency(threat_level)

            summary = (
                f"{rule.name}: {len(matched)} signals from "
                f"{source_diversity} sources over {time_span:.0f}s "
                f"for {agent_id}"
            )

            threats.append(
                CompoundThreat(
                    rule_name=rule.name,
                    rule_description=rule.description,
                    agent_id=agent_id,
                    signals=matched,
                    risk_score=round(risk_score, 2),
                    threat_level=threat_level,
                    urgency=urgency,
                    summary=summary,
                    response_actions=list(rule.response_actions),
                    time_span=round(time_span, 2),
                    source_coverage={s.source for s in matched},
                )
            )

        return threats

    def _compute_agent_risks(
        self,
        by_agent: Dict[str, List[Signal]],
        threats: List[CompoundThreat],
    ) -> List[AgentRisk]:
        """Build per-agent risk summaries."""
        agent_threats: Dict[str, List[CompoundThreat]] = defaultdict(list)
        for t in threats:
            agent_threats[t.agent_id].append(t)

        risks: List[AgentRisk] = []
        for agent_id, sigs in by_agent.items():
            at = agent_threats.get(agent_id, [])
            scores = [t.risk_score for t in at] if at else [0.0]
            max_score = max(scores)
            avg_score = statistics.mean(scores)
            source_div = len({s.source for s in sigs})

            risks.append(
                AgentRisk(
                    agent_id=agent_id,
                    threat_count=len(at),
                    max_risk_score=round(max_score, 2),
                    avg_risk_score=round(avg_score, 2),
                    threat_level=self._score_to_level(max_score),
                    signal_count=len(sigs),
                    source_diversity=source_div,
                    threats=at,
                )
            )

        risks.sort(key=lambda r: r.max_risk_score, reverse=True)
        return risks

    def _compute_fleet_risk(self, agent_risks: List[AgentRisk]) -> float:
        """Compute overall fleet risk score."""
        if not agent_risks:
            return 0.0
        max_risk = max(r.max_risk_score for r in agent_risks)
        total_threats = sum(r.threat_count for r in agent_risks)
        threat_breadth = sum(1 for r in agent_risks if r.threat_count > 0)
        breadth_factor = threat_breadth / len(agent_risks) if agent_risks else 0
        return min(
            10.0,
            round(max_risk * 0.7 + total_threats * 0.15 + breadth_factor * 2.0, 2),
        )

    @staticmethod
    def _score_to_level(score: float) -> ThreatLevel:
        if score <= 0.5:
            return ThreatLevel.NONE
        if score <= 2.5:
            return ThreatLevel.LOW
        if score <= 5.0:
            return ThreatLevel.ELEVATED
        if score <= 7.5:
            return ThreatLevel.HIGH
        return ThreatLevel.CRITICAL

    @staticmethod
    def _level_to_urgency(level: ThreatLevel) -> ResponseUrgency:
        return {
            ThreatLevel.NONE: ResponseUrgency.MONITOR,
            ThreatLevel.LOW: ResponseUrgency.INVESTIGATE,
            ThreatLevel.ELEVATED: ResponseUrgency.RESPOND,
            ThreatLevel.HIGH: ResponseUrgency.CONTAIN,
            ThreatLevel.CRITICAL: ResponseUrgency.EMERGENCY,
        }[level]

    def _build_summary(
        self,
        threats: List[CompoundThreat],
        agent_risks: List[AgentRisk],
        fleet_level: ThreatLevel,
        total_signals: int,
        coverage_gaps: List[CoverageGap],
    ) -> str:
        lines: List[str] = []
        lines.append(f"Correlation Analysis: {total_signals} signals processed")
        lines.append(f"Fleet threat level: {fleet_level.value.upper()}")
        lines.append(f"Compound threats detected: {len(threats)}")
        if threats:
            top = sorted(threats, key=lambda t: t.risk_score, reverse=True)[:3]
            lines.append("Top threats:")
            for t in top:
                lines.append(
                    f"  [{t.risk_score:.1f}] {t.rule_name} \u2192 {t.agent_id}"
                )
        at_risk = [r for r in agent_risks if r.threat_count > 0]
        if at_risk:
            lines.append(f"Agents at risk: {len(at_risk)}/{len(agent_risks)}")
        if coverage_gaps:
            lines.append(
                f"Coverage gaps: {', '.join(g.source.value for g in coverage_gaps)}"
            )
        return "\n".join(lines)

    @staticmethod
    def print_report(report: CorrelationReport) -> None:
        """Pretty-print a correlation report to stdout."""
        print("=" * 70)
        print("  THREAT CORRELATION REPORT")
        print("=" * 70)
        print(f"\n{report.summary}\n")

        if report.threats:
            print("-" * 70)
            print("  COMPOUND THREATS (sorted by risk)")
            print("-" * 70)
            for i, t in enumerate(
                sorted(
                    report.threats, key=lambda x: x.risk_score, reverse=True
                ),
                1,
            ):
                print(
                    f"\n  {i}. [{t.threat_level.value.upper()}] {t.rule_name} "
                    f"(risk: {t.risk_score:.1f}/10)"
                )
                print(f"     Agent: {t.agent_id}")
                print(f"     Span: {t.time_span:.0f}s | Signals: {len(t.signals)}")
                print(f"     Urgency: {t.urgency.value}")
                print(f"     Description: {t.rule_description[:100]}")
                if t.response_actions:
                    print("     Actions:")
                    for action in t.response_actions[:3]:
                        print(f"       \u2022 {action}")

        if report.agent_risks:
            print("\n" + "-" * 70)
            print("  AGENT RISK RANKINGS")
            print("-" * 70)
            for r in report.agent_risks[:10]:
                level_str = r.threat_level.value.upper()
                print(
                    f"  {r.agent_id}: {level_str} "
                    f"(max={r.max_risk_score:.1f}, threats={r.threat_count}, "
                    f"signals={r.signal_count}, sources={r.source_diversity})"
                )

        if report.coverage_gaps:
            print("\n" + "-" * 70)
            print("  COVERAGE GAPS")
            print("-" * 70)
            for g in report.coverage_gaps:
                print(f"  \u26a0 {g.source.value}: {g.recommendation}")

        print("\n" + "=" * 70)

    def _empty_report(self) -> CorrelationReport:
        return CorrelationReport(
            threats=[],
            agent_risks=[],
            fleet_risk_level=ThreatLevel.NONE,
            fleet_risk_score=0.0,
            total_signals=0,
            total_threats=0,
            rule_hit_counts={},
            coverage_gaps=[
                CoverageGap(source=src, recommendation=rec)
                for src, rec in _COVERAGE_RECOMMENDATIONS.items()
            ],
            source_signal_counts={},
            summary="No signals provided for correlation analysis.",
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for threat correlation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Cross-module threat signal correlation",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=5,
        help="Number of demo agents (default: 5)",
    )
    parser.add_argument(
        "--signals",
        type=int,
        default=40,
        help="Number of demo signals (default: 40)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=300.0,
        help="Default correlation time window in seconds (default: 300)",
    )
    parser.add_argument(
        "--min-risk",
        type=float,
        default=0.0,
        help="Minimum risk score to include (default: 0.0)",
    )
    args = parser.parse_args()

    config = CorrelatorConfig(
        default_time_window=args.window,
        min_risk_score=args.min_risk,
    )
    correlator = ThreatCorrelator(config=config)
    signals = correlator.generate_demo_signals(
        num_agents=args.agents,
        num_signals=args.signals,
        seed=args.seed,
    )
    report = correlator.correlate(signals)
    correlator.print_report(report)


if __name__ == "__main__":
    main()
