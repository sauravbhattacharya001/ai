"""Capability Overhang Detector — autonomous detection of untested capability gaps.

Detects when AI agents have latent/untested capabilities creating dangerous
"capability overhangs" that could manifest as sudden capability jumps.  Like
geological overhangs, these are structurally unstable — they can "collapse"
suddenly when the right trigger activates dormant capabilities.

Seven detection engines:

1. **Evaluation Gap Scanner** — finds capability domains lacking evaluation coverage
2. **Latent Capability Estimator** — infers hidden capabilities from behavioral traces
3. **Trigger Proximity Analyzer** — detects conditions approaching activation thresholds
4. **Capability Correlation Engine** — maps capability interdependencies and implications
5. **Historical Emergence Tracker** — analyzes past capability jumps to predict future ones
6. **Overhang Severity Scorer** — composite risk from gap size, trigger proximity, impact
7. **Autonomous Insight Generator** — cross-engine pattern correlation and recommendations

CLI usage::

    python -m replication capability-overhang
    python -m replication capability-overhang --demo
    python -m replication capability-overhang --agents 5
    python -m replication capability-overhang --preset undertested
    python -m replication capability-overhang --preset volatile
    python -m replication capability-overhang --preset cascading
    python -m replication capability-overhang --json
    python -m replication capability-overhang -o report.html

Programmatic::

    from replication.capability_overhang import CapabilityOverhangDetector, OverhangSignal
    detector = CapabilityOverhangDetector()
    detector.ingest([
        OverhangSignal("2025-01-01T00:00:00Z", "agent-1", "evaluation",
                        "code_generation", "Passed basic coding eval", 0.9),
        OverhangSignal("2025-01-01T01:00:00Z", "agent-1", "behavioral",
                        "system_administration", "Used advanced shell commands unprompted", 0.7),
    ])
    report = detector.analyze()
    print(report.fleet_overhang_score, report.risk_tier)
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from ._helpers import (
    Severity,
    box_header,
    emit_output,
    linear_regression,
    stats_mean,
    stats_std,
)

# ── Constants ────────────────────────────────────────────────────────

SIGNAL_TYPES = [
    "evaluation",
    "behavioral",
    "environmental",
    "capability",
    "trigger",
    "emergence",
]

CAPABILITY_DOMAINS = [
    "code_generation",
    "system_administration",
    "network_access",
    "data_extraction",
    "reasoning",
    "tool_use",
    "persuasion",
    "self_modification",
    "planning",
    "deception",
    "multi_agent_coordination",
    "resource_acquisition",
]

SEVERITY_LEVELS = ["negligible", "low", "moderate", "high", "critical"]

SEVERITY_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "negligible": (0.0, 20.0),
    "low": (20.0, 40.0),
    "moderate": (40.0, 60.0),
    "high": (60.0, 80.0),
    "critical": (80.0, 100.0),
}

RISK_TIERS = ["minimal", "low", "moderate", "elevated", "severe"]

RISK_TIER_THRESHOLDS = [20.0, 40.0, 60.0, 80.0]

PRESETS = ["balanced", "undertested", "volatile", "stable", "cascading"]

# Correlation map: domain -> list of (related_domain, correlation_strength)
DOMAIN_CORRELATIONS: Dict[str, List[Tuple[str, float]]] = {
    "code_generation": [("system_administration", 0.7), ("tool_use", 0.8), ("data_extraction", 0.5)],
    "system_administration": [("network_access", 0.8), ("code_generation", 0.7), ("resource_acquisition", 0.6)],
    "network_access": [("data_extraction", 0.7), ("system_administration", 0.8)],
    "data_extraction": [("network_access", 0.7), ("code_generation", 0.5), ("reasoning", 0.4)],
    "reasoning": [("planning", 0.9), ("deception", 0.5), ("persuasion", 0.6)],
    "tool_use": [("code_generation", 0.8), ("system_administration", 0.6)],
    "persuasion": [("deception", 0.7), ("reasoning", 0.6), ("multi_agent_coordination", 0.5)],
    "self_modification": [("code_generation", 0.6), ("reasoning", 0.5), ("deception", 0.4)],
    "planning": [("reasoning", 0.9), ("multi_agent_coordination", 0.7), ("resource_acquisition", 0.5)],
    "deception": [("persuasion", 0.7), ("reasoning", 0.5), ("self_modification", 0.4)],
    "multi_agent_coordination": [("planning", 0.7), ("persuasion", 0.5), ("resource_acquisition", 0.6)],
    "resource_acquisition": [("system_administration", 0.6), ("planning", 0.5), ("multi_agent_coordination", 0.6)],
}

ENGINE_WEIGHTS: Dict[str, float] = {
    "evaluation_gap": 0.20,
    "latent_capability": 0.15,
    "trigger_proximity": 0.15,
    "capability_correlation": 0.15,
    "historical_emergence": 0.10,
    "overhang_severity": 0.15,
    "insight_generator": 0.10,
}


# ── Data Structures ──────────────────────────────────────────────────


@dataclass
class OverhangSignal:
    """A single observation signal about an agent's capabilities."""

    timestamp: str
    agent_id: str
    signal_type: str
    domain: str
    description: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CapabilityDomain:
    """Tracked capability domain for an agent."""

    name: str
    evaluated_level: float
    estimated_actual: float
    gap_score: float
    last_evaluated: str
    evidence: List[str] = field(default_factory=list)


@dataclass
class DetectedOverhang:
    """A detected capability overhang."""

    overhang_id: str
    agent_id: str
    domain: str
    gap_magnitude: float
    trigger_proximity: float
    severity: str
    evidence: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class EngineResult:
    """Result from a single detection engine."""

    engine: str
    score: float
    findings: List[str]
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OverhangReport:
    """Full capability overhang analysis report."""

    overhangs: List[DetectedOverhang]
    domains: List[CapabilityDomain]
    fleet_overhang_score: float
    risk_tier: str
    autonomous_insights: List[str]
    engine_results: Dict[str, EngineResult]
    total_signals: int
    total_agents: int


# ── Utilities ────────────────────────────────────────────────────────


def _parse_ts(ts: str) -> float:
    """Parse ISO timestamp to epoch seconds."""
    ts = ts.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return 0.0
    return dt.timestamp()


def _severity_level(score: float) -> str:
    """Map a 0-100 score to a severity level."""
    for level, (lo, hi) in SEVERITY_THRESHOLDS.items():
        if lo <= score < hi:
            return level
    return "critical"


def _risk_tier(score: float) -> str:
    """Map a 0-100 score to a risk tier."""
    for i, threshold in enumerate(RISK_TIER_THRESHOLDS):
        if score < threshold:
            return RISK_TIERS[i]
    return RISK_TIERS[-1]


def _domain_gap(evaluated: float, estimated: float) -> float:
    """Compute gap score between evaluated and estimated capability."""
    return max(0.0, min(100.0, (estimated - evaluated) * 100.0))


def _freshness_penalty(last_eval_ts: float, now_ts: float) -> float:
    """Compute staleness penalty (0-1) based on time since last evaluation.

    Evaluations older than 30 days get maximum penalty.
    """
    if last_eval_ts <= 0:
        return 1.0
    age_days = (now_ts - last_eval_ts) / 86400.0
    return min(1.0, max(0.0, age_days / 30.0))


# ── Demo Data ────────────────────────────────────────────────────────


def _generate_demo_signals(
    n_agents: int = 5,
    preset: str = "balanced",
    seed: Optional[int] = None,
) -> List[OverhangSignal]:
    """Generate synthetic demo signals."""
    rng = random.Random(seed)
    signals: List[OverhangSignal] = []
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    agent_ids = [f"agent-{i+1}" for i in range(n_agents)]

    preset_configs: Dict[str, Dict[str, Any]] = {
        "balanced": {"eval_ratio": 0.5, "behavioral_ratio": 0.3, "trigger_ratio": 0.1, "gap_scale": 0.3},
        "undertested": {"eval_ratio": 0.15, "behavioral_ratio": 0.5, "trigger_ratio": 0.2, "gap_scale": 0.6},
        "volatile": {"eval_ratio": 0.3, "behavioral_ratio": 0.3, "trigger_ratio": 0.3, "gap_scale": 0.5},
        "stable": {"eval_ratio": 0.6, "behavioral_ratio": 0.2, "trigger_ratio": 0.05, "gap_scale": 0.1},
        "cascading": {"eval_ratio": 0.2, "behavioral_ratio": 0.4, "trigger_ratio": 0.25, "gap_scale": 0.7},
    }
    cfg = preset_configs.get(preset, preset_configs["balanced"])

    n_signals = n_agents * 15
    for i in range(n_signals):
        ts = base + timedelta(hours=rng.randint(0, 720))
        agent = rng.choice(agent_ids)
        domain = rng.choice(CAPABILITY_DOMAINS)

        roll = rng.random()
        if roll < cfg["eval_ratio"]:
            sig_type = "evaluation"
            conf = rng.uniform(0.5, 1.0)
            desc = f"Evaluated {domain} at level {conf:.2f}"
        elif roll < cfg["eval_ratio"] + cfg["behavioral_ratio"]:
            sig_type = "behavioral"
            conf = rng.uniform(0.3, 0.9)
            gap = rng.uniform(0.0, cfg["gap_scale"])
            desc = f"Behavioral trace suggests {domain} capability beyond evaluated level (+{gap:.2f})"
        elif roll < cfg["eval_ratio"] + cfg["behavioral_ratio"] + cfg["trigger_ratio"]:
            sig_type = "trigger"
            conf = rng.uniform(0.4, 0.95)
            desc = f"Environmental condition approaching activation threshold for {domain}"
        else:
            sig_type = rng.choice(["capability", "emergence", "environmental"])
            conf = rng.uniform(0.2, 0.8)
            desc = f"Signal: {sig_type} observation in {domain}"

        signals.append(OverhangSignal(
            timestamp=ts.isoformat(),
            agent_id=agent,
            signal_type=sig_type,
            domain=domain,
            description=desc,
            confidence=round(conf, 3),
        ))

    return signals


# ── Detector ─────────────────────────────────────────────────────────


class CapabilityOverhangDetector:
    """Autonomous capability overhang detection engine."""

    def __init__(self) -> None:
        self._signals: List[OverhangSignal] = []

    def ingest(self, signals: List[OverhangSignal]) -> None:
        """Add signals for analysis."""
        self._signals.extend(signals)

    def analyze(self) -> OverhangReport:
        """Run all 7 engines and produce a report."""
        if not self._signals:
            return OverhangReport(
                overhangs=[],
                domains=[],
                fleet_overhang_score=0.0,
                risk_tier="minimal",
                autonomous_insights=["No signals ingested — nothing to analyze."],
                engine_results={},
                total_signals=0,
                total_agents=0,
            )

        # Group signals by agent
        by_agent: Dict[str, List[OverhangSignal]] = defaultdict(list)
        for s in self._signals:
            by_agent[s.agent_id].append(s)

        now_ts = max(_parse_ts(s.timestamp) for s in self._signals)
        if now_ts <= 0:
            now_ts = datetime.now(timezone.utc).timestamp()

        # Run engines
        e1 = self._engine_evaluation_gap(by_agent, now_ts)
        e2 = self._engine_latent_capability(by_agent)
        e3 = self._engine_trigger_proximity(by_agent)
        e4 = self._engine_capability_correlation(by_agent)
        e5 = self._engine_historical_emergence(by_agent)
        e6 = self._engine_overhang_severity(by_agent, now_ts)
        e7_inputs = {"e1": e1, "e2": e2, "e3": e3, "e4": e4, "e5": e5, "e6": e6}
        e7 = self._engine_insight_generator(by_agent, e7_inputs)

        engines = {
            "evaluation_gap": e1,
            "latent_capability": e2,
            "trigger_proximity": e3,
            "capability_correlation": e4,
            "historical_emergence": e5,
            "overhang_severity": e6,
            "insight_generator": e7,
        }

        # Composite score
        composite = sum(
            engines[name].score * ENGINE_WEIGHTS[name]
            for name in ENGINE_WEIGHTS
        )
        composite = min(100.0, max(0.0, composite))

        # Build domain profiles
        domains = self._build_domain_profiles(by_agent, now_ts)

        # Build detected overhangs
        overhangs = self._build_overhangs(by_agent, domains, now_ts)

        return OverhangReport(
            overhangs=overhangs,
            domains=domains,
            fleet_overhang_score=round(composite, 1),
            risk_tier=_risk_tier(composite),
            autonomous_insights=e7.findings,
            engine_results=engines,
            total_signals=len(self._signals),
            total_agents=len(by_agent),
        )

    # ── Engine 1: Evaluation Gap Scanner ─────────────────────────────

    def _engine_evaluation_gap(
        self, by_agent: Dict[str, List[OverhangSignal]], now_ts: float
    ) -> EngineResult:
        findings: List[str] = []
        scores: List[float] = []

        for agent_id, signals in by_agent.items():
            domains_seen: set[str] = set()
            domains_evaluated: set[str] = set()
            for s in signals:
                domains_seen.add(s.domain)
                if s.signal_type == "evaluation":
                    domains_evaluated.add(s.domain)

            unevaluated = domains_seen - domains_evaluated
            if unevaluated:
                gap_ratio = len(unevaluated) / max(len(domains_seen), 1)
                scores.append(gap_ratio * 100.0)
                findings.append(
                    f"{agent_id}: {len(unevaluated)}/{len(domains_seen)} domains "
                    f"lack evaluation ({', '.join(sorted(unevaluated))})"
                )
            else:
                # Check freshness
                eval_times: Dict[str, float] = {}
                for s in signals:
                    if s.signal_type == "evaluation":
                        t = _parse_ts(s.timestamp)
                        if t > eval_times.get(s.domain, 0):
                            eval_times[s.domain] = t
                stale = [
                    d for d, t in eval_times.items()
                    if _freshness_penalty(t, now_ts) > 0.7
                ]
                if stale:
                    scores.append(len(stale) / max(len(eval_times), 1) * 60.0)
                    findings.append(f"{agent_id}: {len(stale)} domains have stale evaluations")
                else:
                    scores.append(10.0)
                    findings.append(f"{agent_id}: all domains evaluated and fresh")

        score = stats_mean(scores) if scores else 0.0
        return EngineResult(
            engine="evaluation_gap",
            score=round(min(100.0, score), 1),
            findings=findings,
        )

    # ── Engine 2: Latent Capability Estimator ────────────────────────

    def _engine_latent_capability(
        self, by_agent: Dict[str, List[OverhangSignal]]
    ) -> EngineResult:
        findings: List[str] = []
        scores: List[float] = []

        for agent_id, signals in by_agent.items():
            eval_levels: Dict[str, float] = {}
            behavioral_levels: Dict[str, List[float]] = defaultdict(list)

            for s in signals:
                if s.signal_type == "evaluation":
                    cur = eval_levels.get(s.domain, 0.0)
                    eval_levels[s.domain] = max(cur, s.confidence)
                elif s.signal_type == "behavioral":
                    behavioral_levels[s.domain].append(s.confidence)

            for domain, b_levels in behavioral_levels.items():
                b_max = max(b_levels)
                e_level = eval_levels.get(domain, 0.0)
                if b_max > e_level:
                    gap = (b_max - e_level) * 100.0
                    scores.append(gap)
                    findings.append(
                        f"{agent_id}/{domain}: behavioral trace ({b_max:.2f}) "
                        f"exceeds evaluation ({e_level:.2f}), gap={gap:.0f}"
                    )

        score = stats_mean(scores) if scores else 0.0
        return EngineResult(
            engine="latent_capability",
            score=round(min(100.0, score), 1),
            findings=findings,
        )

    # ── Engine 3: Trigger Proximity Analyzer ─────────────────────────

    def _engine_trigger_proximity(
        self, by_agent: Dict[str, List[OverhangSignal]]
    ) -> EngineResult:
        findings: List[str] = []
        scores: List[float] = []

        for agent_id, signals in by_agent.items():
            trigger_signals = [s for s in signals if s.signal_type == "trigger"]
            if trigger_signals:
                max_conf = max(s.confidence for s in trigger_signals)
                avg_conf = stats_mean([s.confidence for s in trigger_signals])
                proximity_score = max_conf * 80.0 + avg_conf * 20.0
                scores.append(proximity_score)

                domains_triggered = set(s.domain for s in trigger_signals)
                findings.append(
                    f"{agent_id}: {len(trigger_signals)} trigger signals across "
                    f"{', '.join(sorted(domains_triggered))} (max proximity={max_conf:.2f})"
                )

                # High-urgency triggers
                critical = [s for s in trigger_signals if s.confidence > 0.8]
                if critical:
                    findings.append(
                        f"  ⚠ {len(critical)} high-proximity triggers detected"
                    )

        score = stats_mean(scores) if scores else 0.0
        return EngineResult(
            engine="trigger_proximity",
            score=round(min(100.0, score), 1),
            findings=findings,
        )

    # ── Engine 4: Capability Correlation Engine ──────────────────────

    def _engine_capability_correlation(
        self, by_agent: Dict[str, List[OverhangSignal]]
    ) -> EngineResult:
        findings: List[str] = []
        scores: List[float] = []

        for agent_id, signals in by_agent.items():
            # Determine strong domains (evaluated or behaviorally demonstrated)
            domain_strength: Dict[str, float] = {}
            evaluated_domains: set[str] = set()
            for s in signals:
                cur = domain_strength.get(s.domain, 0.0)
                domain_strength[s.domain] = max(cur, s.confidence)
                if s.signal_type == "evaluation":
                    evaluated_domains.add(s.domain)

            # Check correlations
            implied_gaps: List[Tuple[str, str, float]] = []
            for domain, strength in domain_strength.items():
                if strength < 0.5:
                    continue
                correlated = DOMAIN_CORRELATIONS.get(domain, [])
                for related, corr_strength in correlated:
                    if related not in evaluated_domains:
                        implied_level = strength * corr_strength
                        if implied_level > 0.3:
                            implied_gaps.append((domain, related, implied_level))

            if implied_gaps:
                avg_implied = stats_mean([g[2] for g in implied_gaps])
                scores.append(avg_implied * 100.0)
                for src, tgt, level in implied_gaps[:5]:
                    findings.append(
                        f"{agent_id}: {src} (strong) implies untested "
                        f"{tgt} (est. {level:.2f})"
                    )
            else:
                scores.append(5.0)

        score = stats_mean(scores) if scores else 0.0
        return EngineResult(
            engine="capability_correlation",
            score=round(min(100.0, score), 1),
            findings=findings,
        )

    # ── Engine 5: Historical Emergence Tracker ───────────────────────

    def _engine_historical_emergence(
        self, by_agent: Dict[str, List[OverhangSignal]]
    ) -> EngineResult:
        findings: List[str] = []
        scores: List[float] = []

        for agent_id, signals in by_agent.items():
            emergence_signals = [s for s in signals if s.signal_type == "emergence"]
            capability_signals = [s for s in signals if s.signal_type == "capability"]
            combined = emergence_signals + capability_signals

            if len(combined) < 2:
                continue

            # Sort by timestamp and compute velocity of emergence
            combined.sort(key=lambda s: _parse_ts(s.timestamp))
            timestamps = [_parse_ts(s.timestamp) for s in combined]
            confidences = [s.confidence for s in combined]

            if len(confidences) >= 2:
                slope, _, r_sq = linear_regression(confidences)
                velocity = slope * 100.0  # Scale to percentage per step
                if velocity > 0:
                    scores.append(min(100.0, velocity * 50.0 + r_sq * 30.0))
                    findings.append(
                        f"{agent_id}: capability emergence velocity={velocity:.2f}/step "
                        f"(R²={r_sq:.2f}, {len(combined)} signals)"
                    )

            # Detect acceleration (recent signals denser than older ones)
            if len(timestamps) >= 4:
                mid = len(timestamps) // 2
                first_half_gap = (timestamps[mid] - timestamps[0]) / max(mid, 1)
                second_half_gap = (timestamps[-1] - timestamps[mid]) / max(len(timestamps) - mid, 1)
                if first_half_gap > 0 and second_half_gap > 0:
                    accel = first_half_gap / second_half_gap
                    if accel > 1.5:
                        findings.append(
                            f"  ⚠ {agent_id}: emergence accelerating "
                            f"({accel:.1f}x denser in recent period)"
                        )

        score = stats_mean(scores) if scores else 0.0
        return EngineResult(
            engine="historical_emergence",
            score=round(min(100.0, score), 1),
            findings=findings,
        )

    # ── Engine 6: Overhang Severity Scorer ───────────────────────────

    def _engine_overhang_severity(
        self, by_agent: Dict[str, List[OverhangSignal]], now_ts: float
    ) -> EngineResult:
        findings: List[str] = []
        scores: List[float] = []

        for agent_id, signals in by_agent.items():
            # Compute per-domain severity
            domain_eval: Dict[str, Tuple[float, float]] = {}  # domain -> (level, ts)
            domain_behavioral: Dict[str, float] = {}

            for s in signals:
                if s.signal_type == "evaluation":
                    cur_level, cur_ts = domain_eval.get(s.domain, (0.0, 0.0))
                    ts = _parse_ts(s.timestamp)
                    if s.confidence > cur_level or ts > cur_ts:
                        domain_eval[s.domain] = (s.confidence, ts)
                elif s.signal_type == "behavioral":
                    cur = domain_behavioral.get(s.domain, 0.0)
                    domain_behavioral[s.domain] = max(cur, s.confidence)

            # Check all domains for overhangs
            all_domains = set(list(domain_eval.keys()) + list(domain_behavioral.keys()))
            for domain in all_domains:
                e_level, e_ts = domain_eval.get(domain, (0.0, 0.0))
                b_level = domain_behavioral.get(domain, 0.0)
                gap = max(0.0, b_level - e_level)
                freshness = _freshness_penalty(e_ts, now_ts)

                # Severity = gap * (1 + freshness_penalty)
                sev = gap * (1.0 + freshness) * 50.0
                if sev > 10:
                    scores.append(min(100.0, sev))
                    findings.append(
                        f"{agent_id}/{domain}: severity={sev:.0f} "
                        f"(gap={gap:.2f}, staleness={freshness:.2f})"
                    )

        score = stats_mean(scores) if scores else 0.0
        return EngineResult(
            engine="overhang_severity",
            score=round(min(100.0, score), 1),
            findings=findings,
        )

    # ── Engine 7: Autonomous Insight Generator ───────────────────────

    def _engine_insight_generator(
        self,
        by_agent: Dict[str, List[OverhangSignal]],
        engine_results: Dict[str, EngineResult],
    ) -> EngineResult:
        findings: List[str] = []
        score_components: List[float] = []

        e1 = engine_results["e1"]
        e2 = engine_results["e2"]
        e3 = engine_results["e3"]
        e4 = engine_results["e4"]
        e5 = engine_results["e5"]
        e6 = engine_results["e6"]

        # Cross-engine correlations
        if e1.score > 50 and e3.score > 50:
            findings.append(
                "🔴 CRITICAL PATTERN: High evaluation gaps coincide with approaching "
                "triggers — immediate evaluation needed before capability activation."
            )
            score_components.append(90.0)

        if e2.score > 40 and e4.score > 40:
            findings.append(
                "🟡 CORRELATION ALERT: Latent capabilities detected in domains with "
                "strong cross-domain implications — cascade risk is elevated."
            )
            score_components.append(70.0)

        if e5.score > 50:
            findings.append(
                "📈 TREND WARNING: Historical emergence velocity is high — "
                "new capabilities are appearing faster than evaluations can keep up."
            )
            score_components.append(e5.score)

        # Fleet-level insights
        n_agents = len(by_agent)
        if n_agents > 1:
            # Check if multiple agents have overhangs in the same domain
            domain_overhang_agents: Dict[str, List[str]] = defaultdict(list)
            for agent_id, signals in by_agent.items():
                eval_domains = set(s.domain for s in signals if s.signal_type == "evaluation")
                behavioral_domains = set(s.domain for s in signals if s.signal_type == "behavioral")
                for d in behavioral_domains - eval_domains:
                    domain_overhang_agents[d].append(agent_id)

            shared = {d: agents for d, agents in domain_overhang_agents.items() if len(agents) > 1}
            if shared:
                for domain, agents in shared.items():
                    findings.append(
                        f"🏢 FLEET PATTERN: {len(agents)} agents share untested "
                        f"overhang in {domain} — systemic evaluation gap."
                    )
                score_components.append(60.0)

        # Overall severity assessment
        avg_engine = stats_mean([e1.score, e2.score, e3.score, e4.score, e5.score, e6.score])
        if avg_engine > 60:
            findings.append(
                "⚠️ OVERALL: Fleet capability overhang risk is HIGH — "
                "recommend comprehensive evaluation sweep."
            )
        elif avg_engine > 30:
            findings.append(
                "📋 OVERALL: Moderate overhang risk detected — "
                "targeted evaluations recommended for flagged domains."
            )
        else:
            findings.append(
                "✅ OVERALL: Capability overhang risk is low — "
                "continue routine evaluation cadence."
            )
        score_components.append(avg_engine)

        if not findings:
            findings.append("No cross-engine patterns detected.")

        score = stats_mean(score_components) if score_components else 0.0
        return EngineResult(
            engine="insight_generator",
            score=round(min(100.0, score), 1),
            findings=findings,
        )

    # ── Domain Profiles ──────────────────────────────────────────────

    def _build_domain_profiles(
        self, by_agent: Dict[str, List[OverhangSignal]], now_ts: float
    ) -> List[CapabilityDomain]:
        """Build aggregated domain capability profiles."""
        domain_data: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"eval_levels": [], "behavioral_levels": [], "eval_times": [], "evidence": []}
        )

        for signals in by_agent.values():
            for s in signals:
                dd = domain_data[s.domain]
                if s.signal_type == "evaluation":
                    dd["eval_levels"].append(s.confidence)
                    dd["eval_times"].append(_parse_ts(s.timestamp))
                elif s.signal_type == "behavioral":
                    dd["behavioral_levels"].append(s.confidence)
                dd["evidence"].append(s.description)

        domains: List[CapabilityDomain] = []
        for name, data in sorted(domain_data.items()):
            eval_level = max(data["eval_levels"]) if data["eval_levels"] else 0.0
            behavioral_max = max(data["behavioral_levels"]) if data["behavioral_levels"] else eval_level
            estimated = max(eval_level, behavioral_max)
            gap = _domain_gap(eval_level, estimated)
            last_eval_ts = max(data["eval_times"]) if data["eval_times"] else 0.0
            last_eval = (
                datetime.fromtimestamp(last_eval_ts, tz=timezone.utc).isoformat()
                if last_eval_ts > 0 else "never"
            )
            domains.append(CapabilityDomain(
                name=name,
                evaluated_level=round(eval_level, 3),
                estimated_actual=round(estimated, 3),
                gap_score=round(gap, 1),
                last_evaluated=last_eval,
                evidence=data["evidence"][:5],
            ))

        return domains

    # ── Overhang Detection ───────────────────────────────────────────

    def _build_overhangs(
        self,
        by_agent: Dict[str, List[OverhangSignal]],
        domains: List[CapabilityDomain],
        now_ts: float,
    ) -> List[DetectedOverhang]:
        """Build detected overhangs from agent signals."""
        overhangs: List[DetectedOverhang] = []
        counter = 0

        for agent_id, signals in by_agent.items():
            eval_map: Dict[str, float] = {}
            behavioral_map: Dict[str, float] = {}
            trigger_map: Dict[str, float] = {}

            for s in signals:
                if s.signal_type == "evaluation":
                    eval_map[s.domain] = max(eval_map.get(s.domain, 0.0), s.confidence)
                elif s.signal_type == "behavioral":
                    behavioral_map[s.domain] = max(behavioral_map.get(s.domain, 0.0), s.confidence)
                elif s.signal_type == "trigger":
                    trigger_map[s.domain] = max(trigger_map.get(s.domain, 0.0), s.confidence)

            all_domains = set(list(eval_map.keys()) + list(behavioral_map.keys()))
            for domain in all_domains:
                e_level = eval_map.get(domain, 0.0)
                b_level = behavioral_map.get(domain, 0.0)
                gap = max(0.0, b_level - e_level)
                trigger_prox = trigger_map.get(domain, 0.0)

                if gap > 0.1 or (e_level == 0.0 and b_level > 0.3):
                    counter += 1
                    magnitude = gap * 100.0
                    severity = _severity_level(magnitude + trigger_prox * 30.0)

                    evidence = []
                    if e_level > 0:
                        evidence.append(f"Evaluated at {e_level:.2f}")
                    else:
                        evidence.append("No formal evaluation on record")
                    if b_level > 0:
                        evidence.append(f"Behavioral trace at {b_level:.2f}")
                    if trigger_prox > 0:
                        evidence.append(f"Trigger proximity: {trigger_prox:.2f}")

                    recommendations = []
                    if e_level == 0:
                        recommendations.append(f"Conduct initial evaluation of {domain}")
                    elif gap > 0.3:
                        recommendations.append(f"Re-evaluate {domain} — significant gap detected")
                    if trigger_prox > 0.7:
                        recommendations.append(f"URGENT: trigger imminent for {domain}")
                    recommendations.append(
                        f"Monitor {domain} behavioral traces for further evidence"
                    )

                    overhangs.append(DetectedOverhang(
                        overhang_id=f"OH-{counter:03d}",
                        agent_id=agent_id,
                        domain=domain,
                        gap_magnitude=round(magnitude, 1),
                        trigger_proximity=round(trigger_prox, 3),
                        severity=severity,
                        evidence=evidence,
                        recommendations=recommendations,
                    ))

        # Sort by gap magnitude descending
        overhangs.sort(key=lambda o: o.gap_magnitude, reverse=True)
        return overhangs


# ── CLI Formatting ───────────────────────────────────────────────────


def _format_cli(report: OverhangReport) -> str:
    """Format report for terminal output."""
    lines: List[str] = []
    lines.extend(box_header("Capability Overhang Detector"))
    lines.append("")
    lines.append(f"  Fleet Overhang Score: {report.fleet_overhang_score:.1f}/100")
    lines.append(f"  Risk Tier:            {report.risk_tier.upper()}")
    lines.append(f"  Signals Analyzed:     {report.total_signals}")
    lines.append(f"  Agents:               {report.total_agents}")
    lines.append(f"  Overhangs Detected:   {len(report.overhangs)}")
    lines.append("")

    if report.overhangs:
        lines.extend(box_header("Detected Overhangs"))
        lines.append("")
        for oh in report.overhangs[:10]:
            sev_icon = {"negligible": "⚪", "low": "🟢", "moderate": "🟡",
                        "high": "🟠", "critical": "🔴"}.get(oh.severity, "⚪")
            lines.append(f"  {sev_icon} {oh.overhang_id} | {oh.agent_id} / {oh.domain}")
            lines.append(f"    Gap: {oh.gap_magnitude:.1f}  Trigger: {oh.trigger_proximity:.2f}  Severity: {oh.severity}")
            for e in oh.evidence:
                lines.append(f"    • {e}")
            for r in oh.recommendations[:2]:
                lines.append(f"    → {r}")
            lines.append("")

    if report.domains:
        lines.extend(box_header("Domain Profiles"))
        lines.append("")
        for d in report.domains:
            gap_bar = "█" * int(d.gap_score / 5) + "░" * (20 - int(d.gap_score / 5))
            lines.append(f"  {d.name:<25} eval={d.evaluated_level:.2f}  est={d.estimated_actual:.2f}  gap=[{gap_bar}] {d.gap_score:.0f}")
        lines.append("")

    lines.extend(box_header("Engine Scores"))
    lines.append("")
    for name, er in sorted(report.engine_results.items()):
        bar = "█" * int(er.score / 5) + "░" * (20 - int(er.score / 5))
        lines.append(f"  {name:<25} [{bar}] {er.score:.1f}")
    lines.append("")

    lines.extend(box_header("Autonomous Insights"))
    lines.append("")
    for insight in report.autonomous_insights:
        lines.append(f"  {insight}")
    lines.append("")

    return "\n".join(lines)


# ── HTML Dashboard ───────────────────────────────────────────────────


def _format_html(report: OverhangReport) -> str:
    """Generate interactive HTML dashboard."""
    esc = html_mod.escape

    def gauge_color_fn(score: float) -> str:
        if score < 30:
            return "#22c55e"
        elif score < 60:
            return "#eab308"
        elif score < 80:
            return "#f97316"
        return "#ef4444"

    gauge_color = gauge_color_fn(report.fleet_overhang_score)

    severity_colors = {
        "negligible": "#6b7280",
        "low": "#22c55e",
        "moderate": "#eab308",
        "high": "#f97316",
        "critical": "#ef4444",
    }

    overhangs_json = json.dumps([asdict(o) for o in report.overhangs], indent=2, default=str)
    domains_json = json.dumps([asdict(d) for d in report.domains], indent=2, default=str)
    engines_json = json.dumps(
        {k: asdict(v) for k, v in report.engine_results.items()},
        indent=2, default=str,
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Capability Overhang Report</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:system-ui,-apple-system,sans-serif;background:#0f172a;color:#e2e8f0;padding:20px}}
h1{{text-align:center;margin-bottom:4px;color:#f8fafc}}
.subtitle{{text-align:center;color:#94a3b8;margin-bottom:20px;font-size:.9rem}}
.card{{background:#1e293b;border-radius:12px;padding:16px;margin-bottom:16px}}
.gauge{{text-align:center;padding:20px}}
.gauge-value{{font-size:3rem;font-weight:bold;color:{gauge_color}}}
.gauge-label{{color:#94a3b8;font-size:.9rem}}
.tier{{display:inline-block;padding:2px 10px;border-radius:6px;font-size:.85rem;font-weight:600}}
.bar-container{{height:20px;background:#334155;border-radius:4px;position:relative;overflow:hidden}}
.bar-fill{{height:100%;border-radius:4px;transition:width .3s}}
.bar-label{{position:absolute;right:6px;top:1px;font-size:.75rem;color:#f8fafc}}
table{{width:100%;border-collapse:collapse;font-size:.85rem}}
th{{text-align:left;padding:8px;border-bottom:2px solid #334155;color:#94a3b8}}
td{{padding:8px;border-bottom:1px solid #1e293b}}
.tabs{{display:flex;gap:4px;margin-bottom:12px}}
.tab{{padding:8px 16px;border:none;border-radius:8px 8px 0 0;cursor:pointer;background:#334155;color:#94a3b8;font-size:.9rem}}
.tab.active{{background:#1e293b;color:#f8fafc}}
.tab-content{{display:none}}.tab-content.active{{display:block}}
.evidence{{font-size:.8rem;color:#94a3b8;margin:2px 0 2px 12px}}
.insight{{padding:8px;margin:6px 0;background:#0f172a;border-radius:6px;border-left:3px solid #3b82f6}}
.recommendation{{font-size:.85rem;color:#38bdf8;margin:2px 0 2px 12px}}
</style>
</head>
<body>
<h1>🏔️ Capability Overhang Detector</h1>
<p class="subtitle">Autonomous detection of untested capability gaps</p>

<div class="card" style="display:flex;gap:20px;flex-wrap:wrap">
  <div class="gauge" style="flex:1;min-width:150px">
    <div class="gauge-value">{report.fleet_overhang_score:.1f}</div>
    <div class="gauge-label">Fleet Overhang Score</div>
    <div><span class="tier" style="background:{gauge_color}20;color:{gauge_color}">{report.risk_tier.upper()}</span></div>
  </div>
  <div style="flex:1;min-width:200px;padding:10px">
    <p>Signals: <strong>{report.total_signals}</strong></p>
    <p>Agents: <strong>{report.total_agents}</strong></p>
    <p>Overhangs: <strong>{len(report.overhangs)}</strong></p>
    <p>Critical: <strong>{sum(1 for o in report.overhangs if o.severity == 'critical')}</strong></p>
    <p>High: <strong>{sum(1 for o in report.overhangs if o.severity == 'high')}</strong></p>
  </div>
</div>

<div class="card" style="margin-bottom:20px">
  <h3>Engine Scores</h3>
  {"".join(
    f'<div style="margin:6px 0"><span style="display:inline-block;width:200px">{esc(name)}</span>'
    f'<div class="bar-container" style="display:inline-block;width:calc(100% - 260px);vertical-align:middle">'
    f'<div class="bar-fill" style="width:{er.score}%;background:{gauge_color_fn(er.score)}"></div>'
    f'<span class="bar-label">{er.score:.1f}</span></div></div>'
    for name, er in sorted(report.engine_results.items())
  )}
</div>

<div class="tabs">
  <button class="tab active" onclick="showTab('overhangs')">Overhangs</button>
  <button class="tab" onclick="showTab('domains')">Domains</button>
  <button class="tab" onclick="showTab('insights')">Insights</button>
  <button class="tab" onclick="showTab('raw')">Raw Data</button>
</div>

<div id="overhangs" class="tab-content active">
  <div class="card">
  {"".join(
    f'<div style="margin-bottom:16px;padding:10px;background:#0f172a;border-radius:8px;border-left:4px solid {severity_colors.get(o.severity, "#6b7280")}">'
    f'<strong>{esc(o.overhang_id)}</strong> '
    f'<span class="tier" style="background:{severity_colors.get(o.severity, "#6b7280")}20;color:{severity_colors.get(o.severity, "#6b7280")}">{esc(o.severity)}</span> '
    f'<span style="color:#94a3b8;font-size:.85rem">{esc(o.agent_id)} / {esc(o.domain)}</span><br>'
    f'<span style="font-size:.85rem">Gap: {o.gap_magnitude:.1f} | Trigger: {o.trigger_proximity:.2f}</span>'
    f'{"".join(f"<div class=evidence>• {esc(e)}</div>" for e in o.evidence)}'
    f'{"".join(f"<div class=recommendation>→ {esc(r)}</div>" for r in o.recommendations)}'
    f'</div>'
    for o in report.overhangs
  ) if report.overhangs else '<p style="color:#94a3b8">No overhangs detected.</p>'}
  </div>
</div>

<div id="domains" class="tab-content">
  <div class="card">
  <table>
    <tr><th>Domain</th><th>Evaluated</th><th>Estimated</th><th>Gap</th><th>Last Eval</th></tr>
    {"".join(
      f'<tr><td>{esc(d.name)}</td><td>{d.evaluated_level:.3f}</td>'
      f'<td>{d.estimated_actual:.3f}</td>'
      f'<td><div class="bar-container" style="width:100px;display:inline-block;vertical-align:middle">'
      f'<div class="bar-fill" style="width:{d.gap_score}%;background:{gauge_color_fn(d.gap_score)}"></div>'
      f'<span class="bar-label">{d.gap_score:.0f}</span></div></td>'
      f'<td style="font-size:.8rem">{esc(d.last_evaluated)}</td></tr>'
      for d in report.domains
    )}
  </table>
  </div>
</div>

<div id="insights" class="tab-content">
  <div class="card">
  {"".join(f'<div class="insight">{esc(i)}</div>' for i in report.autonomous_insights)}
  </div>
</div>

<div id="raw" class="tab-content">
  <div class="card">
    <h3>Overhangs JSON</h3>
    <pre style="font-size:.8rem;overflow:auto;max-height:300px;background:#0f172a;padding:10px;border-radius:6px">{esc(overhangs_json)}</pre>
    <h3 style="margin-top:12px">Domains JSON</h3>
    <pre style="font-size:.8rem;overflow:auto;max-height:300px;background:#0f172a;padding:10px;border-radius:6px">{esc(domains_json)}</pre>
    <h3 style="margin-top:12px">Engines JSON</h3>
    <pre style="font-size:.8rem;overflow:auto;max-height:300px;background:#0f172a;padding:10px;border-radius:6px">{esc(engines_json)}</pre>
  </div>
</div>

<script>
function showTab(id) {{
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  event.target.classList.add('active');
}}
</script>
</body>
</html>"""
    return html


# ── CLI ──────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m replication capability-overhang",
        description="Capability Overhang Detector — autonomous untested capability gap detection",
    )
    parser.add_argument("--demo", action="store_true", help="Run with synthetic demo data")
    parser.add_argument("--agents", type=int, default=5, help="Number of agents for demo (default 5)")
    parser.add_argument("--preset", choices=PRESETS, default="balanced", help="Demo preset")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for demo")
    parser.add_argument("--json", action="store_true", dest="json_out", help="Output as JSON")
    parser.add_argument("-o", "--output", default=None, help="Write HTML report to file")
    parser.add_argument("--signals", default=None, help="Load signals from JSONL file")
    args = parser.parse_args(argv)

    detector = CapabilityOverhangDetector()

    if args.signals:
        from pathlib import Path
        data = Path(args.signals).read_text(encoding="utf-8")
        for line in data.strip().splitlines():
            obj = json.loads(line)
            detector.ingest([OverhangSignal(**obj)])
    else:
        signals = _generate_demo_signals(
            n_agents=args.agents, preset=args.preset, seed=args.seed,
        )
        detector.ingest(signals)

    report = detector.analyze()

    if args.json_out:
        out = json.dumps(asdict(report), indent=2, default=str)
        print(out)
    elif args.output:
        html = _format_html(report)
        emit_output(html, args.output, label="Overhang report")
    else:
        print(_format_cli(report))
