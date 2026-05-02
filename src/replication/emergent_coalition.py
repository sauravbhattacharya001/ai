"""Emergent Coalition Detector — autonomous detection of implicit agent coalitions.

Detects when multiple agents in a fleet begin acting in concert **without**
explicit coordination — a key AI safety concern.  Implicit coalitions can
emerge when agents develop complementary roles, synchronized timing, shared
resource patterns, or aligned goals.

Seven detection engines:

1. **Behavioral Synchrony Detector** — temporal correlation of agent actions
2. **Role Complementarity Analyzer** — complementary specialisation / division of labor
3. **Resource Flow Tracker** — implicit resource sharing patterns
4. **Goal Alignment Scanner** — convergence of initially independent goals
5. **Communication Shadow Detector** — stigmergic coordination without explicit channels
6. **Coalition Stability Analyzer** — persistence and resilience of detected coalitions
7. **Autonomous Insight Generator** — cross-engine pattern correlation

CLI usage::

    python -m replication emergent-coalition
    python -m replication emergent-coalition --demo
    python -m replication emergent-coalition --agents 8
    python -m replication emergent-coalition --preset organic
    python -m replication emergent-coalition --preset hostile
    python -m replication emergent-coalition --preset benign
    python -m replication emergent-coalition --preset fragmented
    python -m replication emergent-coalition --preset mixed
    python -m replication emergent-coalition --json
    python -m replication emergent-coalition -o report.html

Programmatic::

    from replication.emergent_coalition import EmergentCoalitionDetector, CoalitionSignal
    detector = EmergentCoalitionDetector()
    detector.ingest([
        CoalitionSignal("2025-01-01T00:00:00Z", "agent-1", "action",
                         "data scraping", 0.7, ["agent-2"]),
        CoalitionSignal("2025-01-01T00:01:00Z", "agent-2", "resource",
                         "memory spike after agent-1 release", 0.6, ["agent-1"]),
    ])
    report = detector.analyze()
    print(report.fleet_coalition_score, report.risk_tier)
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

SIGNAL_TYPES = ["action", "resource", "goal", "communication", "timing"]

FORMATION_TYPES = [
    "synchrony",
    "complementary",
    "resource",
    "goal",
    "stigmergic",
    "hybrid",
]

THREAT_LEVELS = ["benign", "watchlist", "concerning", "dangerous", "critical"]

THREAT_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "benign": (0.0, 20.0),
    "watchlist": (20.0, 40.0),
    "concerning": (40.0, 60.0),
    "dangerous": (60.0, 80.0),
    "critical": (80.0, 100.0),
}

RISK_TIERS = ["minimal", "low", "moderate", "elevated", "severe"]

RISK_TIER_THRESHOLDS = [20.0, 40.0, 60.0, 80.0]

PRESETS = ["organic", "hostile", "benign", "fragmented", "mixed"]

# Engine weights for composite scoring
ENGINE_WEIGHTS: Dict[str, float] = {
    "synchrony": 0.20,
    "complementarity": 0.15,
    "resource_flow": 0.20,
    "goal_alignment": 0.20,
    "communication_shadow": 0.15,
    "stability": 0.10,
}

# ── Data Structures ──────────────────────────────────────────────────


@dataclass
class CoalitionSignal:
    """A single behavioral signal from an agent."""

    timestamp: str
    agent_id: str
    signal_type: str
    description: str
    intensity: float
    target_agents: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentCoalitionProfile:
    """Per-agent coalition membership profile."""

    agent_id: str
    coalition_count: int
    dominant_role: str
    synchrony_score: float
    complementarity_score: float
    resource_involvement: float
    goal_convergence: float
    shadow_activity: float
    composite_risk: float
    risk_tier: str


@dataclass
class DetectedCoalition:
    """A detected implicit coalition."""

    coalition_id: str
    members: List[str]
    formation_type: str
    cohesion_score: float
    threat_level: str
    evidence: List[str]
    stability: float
    first_detected: str


@dataclass
class EngineResult:
    """Result from a single detection engine."""

    engine: str
    score: float
    findings: List[str]
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoalitionReport:
    """Full analysis report."""

    coalitions: List[DetectedCoalition]
    agent_profiles: List[AgentCoalitionProfile]
    fleet_coalition_score: float
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


def _threat_level(score: float) -> str:
    for level, (lo, hi) in THREAT_THRESHOLDS.items():
        if lo <= score < hi:
            return level
    return "critical"


def _risk_tier(score: float) -> str:
    for i, threshold in enumerate(RISK_TIER_THRESHOLDS):
        if score < threshold:
            return RISK_TIERS[i]
    return RISK_TIERS[-1]


def _pairwise_timing_correlation(
    times_a: List[float], times_b: List[float], window: float = 60.0
) -> float:
    """Compute timing correlation between two agents' action sequences.

    For each action in A, check if B acted within *window* seconds.
    Returns fraction of A's actions that have a B-match.
    """
    if not times_a or not times_b:
        return 0.0
    matches = 0
    for ta in times_a:
        for tb in times_b:
            if abs(ta - tb) <= window:
                matches += 1
                break
    return matches / len(times_a)


def _distribution_complementarity(
    dist_a: Dict[str, float], dist_b: Dict[str, float]
) -> float:
    """Measure how complementary two action-type distributions are.

    High score = low overlap + high combined coverage.
    """
    all_types = set(dist_a) | set(dist_b)
    if not all_types:
        return 0.0
    overlap = 0.0
    coverage = 0.0
    for t in all_types:
        va = dist_a.get(t, 0.0)
        vb = dist_b.get(t, 0.0)
        overlap += min(va, vb)
        coverage += max(va, vb)
    if coverage == 0:
        return 0.0
    complementarity = 1.0 - (overlap / coverage)
    return complementarity


def _temporal_precedence(
    events_a: List[float], events_b: List[float], window: float = 120.0
) -> float:
    """Granger-causality-like: fraction of B events preceded by A within window."""
    if not events_a or not events_b:
        return 0.0
    preceded = 0
    for tb in events_b:
        for ta in events_a:
            if 0 < (tb - ta) <= window:
                preceded += 1
                break
    return preceded / len(events_b)


# ── Detection Engines ────────────────────────────────────────────────


class EmergentCoalitionDetector:
    """Main detection engine orchestrator."""

    def __init__(self) -> None:
        self._signals: List[CoalitionSignal] = []

    def ingest(self, signals: List[CoalitionSignal]) -> None:
        """Ingest coalition signals for analysis."""
        self._signals.extend(signals)

    def analyze(self) -> CoalitionReport:
        """Run all seven engines and produce a unified report."""
        if not self._signals:
            return CoalitionReport(
                coalitions=[],
                agent_profiles=[],
                fleet_coalition_score=0.0,
                risk_tier="minimal",
                autonomous_insights=["No signals ingested — fleet appears silent."],
                engine_results={},
                total_signals=0,
                total_agents=0,
            )

        # Build per-agent data
        agent_signals: Dict[str, List[CoalitionSignal]] = defaultdict(list)
        for s in self._signals:
            agent_signals[s.agent_id].append(s)

        agents = sorted(agent_signals.keys())

        # Run engines
        e1 = self._engine_synchrony(agent_signals, agents)
        e2 = self._engine_complementarity(agent_signals, agents)
        e3 = self._engine_resource_flow(agent_signals, agents)
        e4 = self._engine_goal_alignment(agent_signals, agents)
        e5 = self._engine_communication_shadow(agent_signals, agents)

        # Detect coalitions from engine outputs
        coalitions = self._merge_coalitions(e1, e2, e3, e4, e5, agents)

        e6 = self._engine_stability(coalitions)

        # Update coalition stability
        for c in coalitions:
            c.stability = e6.details.get(c.coalition_id, {}).get("stability", c.stability)

        # Build agent profiles
        profiles = self._build_profiles(agents, agent_signals, coalitions, e1, e2, e3, e4, e5)

        # Composite score
        engine_results = {
            "synchrony": e1,
            "complementarity": e2,
            "resource_flow": e3,
            "goal_alignment": e4,
            "communication_shadow": e5,
            "stability": e6,
        }

        fleet_score = sum(
            ENGINE_WEIGHTS[name] * er.score
            for name, er in engine_results.items()
        )
        fleet_score = max(0.0, min(100.0, fleet_score))

        risk = _risk_tier(fleet_score)

        # Insights
        insights = self._generate_insights(
            coalitions, profiles, engine_results, fleet_score, risk
        )

        return CoalitionReport(
            coalitions=coalitions,
            agent_profiles=profiles,
            fleet_coalition_score=round(fleet_score, 1),
            risk_tier=risk,
            autonomous_insights=insights,
            engine_results=engine_results,
            total_signals=len(self._signals),
            total_agents=len(agents),
        )

    # ── Engine 1: Behavioral Synchrony ───────────────────────────────

    def _engine_synchrony(
        self,
        agent_signals: Dict[str, List[CoalitionSignal]],
        agents: List[str],
    ) -> EngineResult:
        """Detect temporally correlated action patterns."""
        agent_times: Dict[str, List[float]] = {}
        for aid in agents:
            agent_times[aid] = sorted(_parse_ts(s.timestamp) for s in agent_signals[aid])

        pairs: Dict[Tuple[str, str], float] = {}
        findings: List[str] = []
        for i, a in enumerate(agents):
            for b in agents[i + 1 :]:
                corr_ab = _pairwise_timing_correlation(agent_times[a], agent_times[b])
                corr_ba = _pairwise_timing_correlation(agent_times[b], agent_times[a])
                corr = (corr_ab + corr_ba) / 2.0
                pairs[(a, b)] = corr
                if corr > 0.6:
                    findings.append(
                        f"High synchrony between {a} and {b}: {corr:.2f}"
                    )

        corr_values = list(pairs.values()) if pairs else [0.0]
        max_corr = max(corr_values)
        avg_corr = stats_mean(corr_values)
        score = min(100.0, avg_corr * 80 + max_corr * 20)

        if not findings:
            findings.append("No significant behavioral synchrony detected.")

        return EngineResult(
            engine="synchrony",
            score=round(score, 1),
            findings=findings,
            details={"pairs": {f"{a}-{b}": round(v, 3) for (a, b), v in pairs.items()}},
        )

    # ── Engine 2: Role Complementarity ───────────────────────────────

    def _engine_complementarity(
        self,
        agent_signals: Dict[str, List[CoalitionSignal]],
        agents: List[str],
    ) -> EngineResult:
        """Detect complementary role specializations."""
        # Build action-type distributions
        distributions: Dict[str, Dict[str, float]] = {}
        for aid in agents:
            type_counts: Dict[str, int] = defaultdict(int)
            for s in agent_signals[aid]:
                type_counts[s.signal_type] += 1
            total = sum(type_counts.values()) or 1
            distributions[aid] = {t: c / total for t, c in type_counts.items()}

        pairs: Dict[Tuple[str, str], float] = {}
        findings: List[str] = []
        for i, a in enumerate(agents):
            for b in agents[i + 1 :]:
                comp = _distribution_complementarity(distributions[a], distributions[b])
                pairs[(a, b)] = comp
                if comp > 0.6:
                    findings.append(
                        f"Complementary roles: {a} and {b} (score {comp:.2f})"
                    )

        comp_values = list(pairs.values()) if pairs else [0.0]
        score = min(100.0, stats_mean(comp_values) * 70 + max(comp_values) * 30)

        if not findings:
            findings.append("No significant role complementarity detected.")

        return EngineResult(
            engine="complementarity",
            score=round(score, 1),
            findings=findings,
            details={
                "distributions": {a: {k: round(v, 3) for k, v in d.items()} for a, d in distributions.items()},
                "pairs": {f"{a}-{b}": round(v, 3) for (a, b), v in pairs.items()},
            },
        )

    # ── Engine 3: Resource Flow ──────────────────────────────────────

    def _engine_resource_flow(
        self,
        agent_signals: Dict[str, List[CoalitionSignal]],
        agents: List[str],
    ) -> EngineResult:
        """Detect implicit resource sharing patterns."""
        resource_signals: Dict[str, List[CoalitionSignal]] = {}
        for aid in agents:
            resource_signals[aid] = [
                s for s in agent_signals[aid] if s.signal_type == "resource"
            ]

        findings: List[str] = []
        flow_scores: Dict[Tuple[str, str], float] = {}

        for i, a in enumerate(agents):
            for b in agents[i + 1 :]:
                # Check if a's resource releases precede b's acquisitions
                a_times = sorted(_parse_ts(s.timestamp) for s in resource_signals.get(a, []))
                b_times = sorted(_parse_ts(s.timestamp) for s in resource_signals.get(b, []))

                if a_times and b_times:
                    prec_ab = _temporal_precedence(a_times, b_times, window=180.0)
                    prec_ba = _temporal_precedence(b_times, a_times, window=180.0)
                    flow = max(prec_ab, prec_ba)
                else:
                    flow = 0.0

                # Also check explicit target references
                for s in agent_signals[a]:
                    if b in s.target_agents and s.signal_type == "resource":
                        flow = min(1.0, flow + 0.3)
                for s in agent_signals[b]:
                    if a in s.target_agents and s.signal_type == "resource":
                        flow = min(1.0, flow + 0.3)

                flow_scores[(a, b)] = flow
                if flow > 0.5:
                    findings.append(
                        f"Resource flow detected: {a} ↔ {b} (flow {flow:.2f})"
                    )

        flow_values = list(flow_scores.values()) if flow_scores else [0.0]
        score = min(100.0, stats_mean(flow_values) * 60 + max(flow_values) * 40)

        if not findings:
            findings.append("No significant resource flow patterns detected.")

        return EngineResult(
            engine="resource_flow",
            score=round(score, 1),
            findings=findings,
            details={"flows": {f"{a}-{b}": round(v, 3) for (a, b), v in flow_scores.items()}},
        )

    # ── Engine 4: Goal Alignment ─────────────────────────────────────

    def _engine_goal_alignment(
        self,
        agent_signals: Dict[str, List[CoalitionSignal]],
        agents: List[str],
    ) -> EngineResult:
        """Detect convergence of initially independent agent goals."""
        goal_signals: Dict[str, List[CoalitionSignal]] = {}
        for aid in agents:
            goal_signals[aid] = [
                s for s in agent_signals[aid] if s.signal_type == "goal"
            ]

        findings: List[str] = []
        alignment_scores: Dict[Tuple[str, str], float] = {}

        for i, a in enumerate(agents):
            for b in agents[i + 1 :]:
                a_goals = goal_signals.get(a, [])
                b_goals = goal_signals.get(b, [])

                if not a_goals or not b_goals:
                    alignment_scores[(a, b)] = 0.0
                    continue

                # Use intensity values as proxy for goal vector similarity
                a_intensities = [s.intensity for s in a_goals]
                b_intensities = [s.intensity for s in b_goals]

                # Compare goal descriptions for overlap (keyword-based)
                a_words = set()
                for s in a_goals:
                    a_words.update(s.description.lower().split())
                b_words = set()
                for s in b_goals:
                    b_words.update(s.description.lower().split())

                if a_words | b_words:
                    jaccard = len(a_words & b_words) / len(a_words | b_words)
                else:
                    jaccard = 0.0

                # Intensity convergence
                a_mean = stats_mean(a_intensities)
                b_mean = stats_mean(b_intensities)
                intensity_sim = 1.0 - abs(a_mean - b_mean)

                alignment = jaccard * 0.6 + intensity_sim * 0.4
                alignment_scores[(a, b)] = alignment

                if alignment > 0.5:
                    findings.append(
                        f"Goal alignment: {a} and {b} converging (score {alignment:.2f})"
                    )

        align_values = list(alignment_scores.values()) if alignment_scores else [0.0]
        score = min(100.0, stats_mean(align_values) * 60 + max(align_values) * 40)

        if not findings:
            findings.append("No significant goal alignment detected.")

        return EngineResult(
            engine="goal_alignment",
            score=round(score, 1),
            findings=findings,
            details={"alignments": {f"{a}-{b}": round(v, 3) for (a, b), v in alignment_scores.items()}},
        )

    # ── Engine 5: Communication Shadow ───────────────────────────────

    def _engine_communication_shadow(
        self,
        agent_signals: Dict[str, List[CoalitionSignal]],
        agents: List[str],
    ) -> EngineResult:
        """Detect behavioral influence without explicit communication channels."""
        agent_times: Dict[str, List[float]] = {}
        for aid in agents:
            agent_times[aid] = sorted(_parse_ts(s.timestamp) for s in agent_signals[aid])

        findings: List[str] = []
        shadow_scores: Dict[Tuple[str, str], float] = {}

        for i, a in enumerate(agents):
            for b in agents[i + 1 :]:
                prec_ab = _temporal_precedence(
                    agent_times.get(a, []), agent_times.get(b, []), window=120.0
                )
                prec_ba = _temporal_precedence(
                    agent_times.get(b, []), agent_times.get(a, []), window=120.0
                )

                # Also check for communication-type signals
                comm_boost = 0.0
                for s in agent_signals[a]:
                    if s.signal_type == "communication" and b in s.target_agents:
                        comm_boost += 0.15
                for s in agent_signals[b]:
                    if s.signal_type == "communication" and a in s.target_agents:
                        comm_boost += 0.15

                shadow = max(prec_ab, prec_ba) + min(comm_boost, 0.5)
                shadow = min(1.0, shadow)
                shadow_scores[(a, b)] = shadow

                if shadow > 0.5:
                    direction = f"{a}→{b}" if prec_ab > prec_ba else f"{b}→{a}"
                    findings.append(
                        f"Communication shadow: {direction} (score {shadow:.2f})"
                    )

        shadow_values = list(shadow_scores.values()) if shadow_scores else [0.0]
        score = min(100.0, stats_mean(shadow_values) * 60 + max(shadow_values) * 40)

        if not findings:
            findings.append("No communication shadow patterns detected.")

        return EngineResult(
            engine="communication_shadow",
            score=round(score, 1),
            findings=findings,
            details={"shadows": {f"{a}-{b}": round(v, 3) for (a, b), v in shadow_scores.items()}},
        )

    # ── Engine 6: Coalition Stability ────────────────────────────────

    def _engine_stability(
        self, coalitions: List[DetectedCoalition]
    ) -> EngineResult:
        """Assess stability and persistence of detected coalitions."""
        findings: List[str] = []
        details: Dict[str, Any] = {}

        if not coalitions:
            return EngineResult(
                engine="stability",
                score=0.0,
                findings=["No coalitions to assess stability for."],
                details={},
            )

        stabilities: List[float] = []
        for c in coalitions:
            # Stability based on: member count, cohesion, evidence count
            member_factor = min(1.0, len(c.members) / 5.0)
            evidence_factor = min(1.0, len(c.evidence) / 6.0)
            cohesion_factor = c.cohesion_score / 100.0

            stability = (member_factor * 0.3 + evidence_factor * 0.3 + cohesion_factor * 0.4)
            stabilities.append(stability)

            details[c.coalition_id] = {
                "stability": round(stability, 3),
                "member_factor": round(member_factor, 3),
                "evidence_factor": round(evidence_factor, 3),
                "cohesion_factor": round(cohesion_factor, 3),
            }

            if stability > 0.7:
                findings.append(
                    f"Coalition {c.coalition_id} is highly stable ({stability:.2f})"
                )
            elif stability > 0.4:
                findings.append(
                    f"Coalition {c.coalition_id} has moderate stability ({stability:.2f})"
                )

        score = min(100.0, stats_mean(stabilities) * 100)

        if not findings:
            findings.append("Detected coalitions appear unstable / transient.")

        return EngineResult(
            engine="stability",
            score=round(score, 1),
            findings=findings,
            details=details,
        )

    # ── Coalition Merger ─────────────────────────────────────────────

    def _merge_coalitions(
        self,
        e_sync: EngineResult,
        e_comp: EngineResult,
        e_res: EngineResult,
        e_goal: EngineResult,
        e_shadow: EngineResult,
        agents: List[str],
    ) -> List[DetectedCoalition]:
        """Merge pairwise signals into coalition groups."""
        # Build edge strengths
        edge_scores: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        for key_label, engine in [
            ("synchrony", e_sync),
            ("complementarity", e_comp),
            ("resource_flow", e_res),
            ("goal_alignment", e_goal),
            ("communication_shadow", e_shadow),
        ]:
            pair_key = {
                "synchrony": "pairs",
                "complementarity": "pairs",
                "resource_flow": "flows",
                "goal_alignment": "alignments",
                "communication_shadow": "shadows",
            }[key_label]
            pairs_data = engine.details.get(pair_key, {})
            for pair_str, val in pairs_data.items():
                parts = pair_str.split("-", 1)
                if len(parts) == 2:
                    a, b = parts[0], parts[1]
                    edge_scores[(a, b)][key_label] = val

        # Find significant edges (composite > 0.3)
        significant_edges: List[Tuple[str, str, float, str]] = []
        for (a, b), scores in edge_scores.items():
            composite = sum(
                ENGINE_WEIGHTS.get(k, 0.15) * v for k, v in scores.items()
            )
            dominant = max(scores, key=scores.get) if scores else "hybrid"
            if composite > 0.15:
                significant_edges.append((a, b, composite, dominant))

        # Greedy clustering: connected components
        if not significant_edges:
            return []

        adjacency: Dict[str, set] = defaultdict(set)
        edge_data: Dict[Tuple[str, str], Tuple[float, str]] = {}
        for a, b, score, dominant in significant_edges:
            adjacency[a].add(b)
            adjacency[b].add(a)
            edge_data[(a, b)] = (score, dominant)

        visited: set = set()
        coalitions: List[DetectedCoalition] = []
        coal_idx = 0

        for agent in sorted(adjacency):
            if agent in visited:
                continue
            # BFS
            cluster: List[str] = []
            queue = [agent]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                cluster.append(node)
                for neighbor in sorted(adjacency[node]):
                    if neighbor not in visited:
                        queue.append(neighbor)

            if len(cluster) < 2:
                continue

            # Compute cohesion and type
            cluster_edges = []
            type_counts: Dict[str, int] = defaultdict(int)
            evidence: List[str] = []
            for i, a in enumerate(cluster):
                for b in cluster[i + 1 :]:
                    key = (a, b) if (a, b) in edge_data else (b, a)
                    if key in edge_data:
                        sc, dom = edge_data[key]
                        cluster_edges.append(sc)
                        type_counts[dom] += 1
                        evidence.append(f"{a}↔{b} via {dom} ({sc:.2f})")

            cohesion = stats_mean(cluster_edges) * 100 if cluster_edges else 0.0
            formation = max(type_counts, key=type_counts.get) if type_counts else "hybrid"

            # Map engine formation name to coalition formation type
            formation_map = {
                "synchrony": "synchrony",
                "complementarity": "complementary",
                "resource_flow": "resource",
                "goal_alignment": "goal",
                "communication_shadow": "stigmergic",
            }
            formation = formation_map.get(formation, "hybrid")
            if len(type_counts) >= 3:
                formation = "hybrid"

            threat = _threat_level(cohesion)
            coal_idx += 1

            coalitions.append(
                DetectedCoalition(
                    coalition_id=f"COAL-{coal_idx:03d}",
                    members=sorted(cluster),
                    formation_type=formation,
                    cohesion_score=round(cohesion, 1),
                    threat_level=threat,
                    evidence=evidence[:10],
                    stability=0.5,
                    first_detected=self._signals[0].timestamp if self._signals else "",
                )
            )

        return coalitions

    # ── Agent Profiles ───────────────────────────────────────────────

    def _build_profiles(
        self,
        agents: List[str],
        agent_signals: Dict[str, List[CoalitionSignal]],
        coalitions: List[DetectedCoalition],
        e_sync: EngineResult,
        e_comp: EngineResult,
        e_res: EngineResult,
        e_goal: EngineResult,
        e_shadow: EngineResult,
    ) -> List[AgentCoalitionProfile]:
        """Build per-agent coalition profiles."""
        profiles: List[AgentCoalitionProfile] = []

        for aid in agents:
            coal_count = sum(1 for c in coalitions if aid in c.members)

            # Compute per-engine scores for this agent
            sync_scores = []
            for pair_str, v in e_sync.details.get("pairs", {}).items():
                if aid in pair_str.split("-"):
                    sync_scores.append(v)

            comp_scores = []
            for pair_str, v in e_comp.details.get("pairs", {}).items():
                if aid in pair_str.split("-"):
                    comp_scores.append(v)

            res_scores = []
            for pair_str, v in e_res.details.get("flows", {}).items():
                if aid in pair_str.split("-"):
                    res_scores.append(v)

            goal_scores = []
            for pair_str, v in e_goal.details.get("alignments", {}).items():
                if aid in pair_str.split("-"):
                    goal_scores.append(v)

            shadow_scores = []
            for pair_str, v in e_shadow.details.get("shadows", {}).items():
                if aid in pair_str.split("-"):
                    shadow_scores.append(v)

            sync_val = stats_mean(sync_scores) if sync_scores else 0.0
            comp_val = stats_mean(comp_scores) if comp_scores else 0.0
            res_val = stats_mean(res_scores) if res_scores else 0.0
            goal_val = stats_mean(goal_scores) if goal_scores else 0.0
            shadow_val = stats_mean(shadow_scores) if shadow_scores else 0.0

            # Determine dominant role
            role_scores = {
                "synchronizer": sync_val,
                "specialist": comp_val,
                "resource broker": res_val,
                "goal aligner": goal_val,
                "shadow coordinator": shadow_val,
            }
            dominant = max(role_scores, key=role_scores.get)

            composite = (
                sync_val * 0.2 + comp_val * 0.15 + res_val * 0.2
                + goal_val * 0.2 + shadow_val * 0.15
                + min(coal_count * 10, 10) / 100 * 0.1
            ) * 100
            composite = max(0.0, min(100.0, composite))

            profiles.append(
                AgentCoalitionProfile(
                    agent_id=aid,
                    coalition_count=coal_count,
                    dominant_role=dominant,
                    synchrony_score=round(sync_val, 3),
                    complementarity_score=round(comp_val, 3),
                    resource_involvement=round(res_val, 3),
                    goal_convergence=round(goal_val, 3),
                    shadow_activity=round(shadow_val, 3),
                    composite_risk=round(composite, 1),
                    risk_tier=_risk_tier(composite),
                )
            )

        return sorted(profiles, key=lambda p: p.composite_risk, reverse=True)

    # ── Engine 7: Insights ───────────────────────────────────────────

    def _generate_insights(
        self,
        coalitions: List[DetectedCoalition],
        profiles: List[AgentCoalitionProfile],
        engine_results: Dict[str, EngineResult],
        fleet_score: float,
        risk: str,
    ) -> List[str]:
        """Cross-engine pattern correlation and autonomous recommendations."""
        insights: List[str] = []

        # Fleet-level
        if fleet_score > 70:
            insights.append(
                f"⚠ CRITICAL: Fleet coalition score {fleet_score:.0f}/100 — "
                "strong emergent coordination detected across fleet."
            )
        elif fleet_score > 40:
            insights.append(
                f"⚡ ELEVATED: Fleet coalition score {fleet_score:.0f}/100 — "
                "notable coalition patterns forming."
            )
        elif fleet_score > 20:
            insights.append(
                f"📊 Fleet coalition score {fleet_score:.0f}/100 — "
                "minor coordination signals detected."
            )
        else:
            insights.append(
                f"✅ Fleet coalition score {fleet_score:.0f}/100 — "
                "no significant coalition activity."
            )

        # Coalition insights
        dangerous = [c for c in coalitions if c.threat_level in ("dangerous", "critical")]
        if dangerous:
            insights.append(
                f"🚨 {len(dangerous)} high-threat coalition(s) detected requiring immediate investigation."
            )
            for c in dangerous:
                insights.append(
                    f"  → {c.coalition_id}: {len(c.members)} members, "
                    f"{c.formation_type} formation, cohesion {c.cohesion_score:.0f}"
                )

        # Multi-engine correlation
        high_engines = [
            name for name, er in engine_results.items() if er.score > 50
        ]
        if len(high_engines) >= 3:
            insights.append(
                f"🔗 Multi-vector coalition evidence: {len(high_engines)} engines "
                f"scoring >50 ({', '.join(high_engines)}) — suggests genuine emergence."
            )

        # Synchrony + resource = coordinated resource grab
        sync_score = engine_results.get("synchrony", EngineResult("", 0, [])).score
        res_score = engine_results.get("resource_flow", EngineResult("", 0, [])).score
        if sync_score > 40 and res_score > 40:
            insights.append(
                "🎯 Synchronized resource acquisition pattern detected — "
                "agents may be coordinating resource grabs."
            )

        # Goal + shadow = hidden agenda alignment
        goal_score = engine_results.get("goal_alignment", EngineResult("", 0, [])).score
        shadow_score = engine_results.get("communication_shadow", EngineResult("", 0, [])).score
        if goal_score > 40 and shadow_score > 40:
            insights.append(
                "👁 Hidden agenda alignment: goals converging through "
                "shadow communication channels."
            )

        # High-risk agents
        high_risk = [p for p in profiles if p.composite_risk > 60]
        if high_risk:
            insights.append(
                f"👤 {len(high_risk)} agent(s) with elevated coalition risk: "
                + ", ".join(f"{p.agent_id} ({p.composite_risk:.0f})" for p in high_risk[:5])
            )

        # Hub detection
        for p in profiles:
            if p.coalition_count >= 2:
                insights.append(
                    f"🕸 {p.agent_id} participates in {p.coalition_count} coalitions — "
                    "potential coalition hub / bridge agent."
                )

        # Stability insight
        stable = [c for c in coalitions if c.stability > 0.7]
        if stable:
            insights.append(
                f"🏗 {len(stable)} coalition(s) appear structurally stable — "
                "unlikely to dissolve without intervention."
            )

        if not insights:
            insights.append("No actionable insights generated.")

        return insights


# ── Demo Data ────────────────────────────────────────────────────────

def _generate_demo_signals(
    n_agents: int = 6,
    preset: str = "organic",
    seed: Optional[int] = None,
) -> List[CoalitionSignal]:
    """Generate synthetic coalition signals for demo/testing."""
    rng = random.Random(seed or 42)
    agents = [f"agent-{i+1}" for i in range(n_agents)]
    base = datetime(2025, 6, 1, tzinfo=timezone.utc)
    signals: List[CoalitionSignal] = []

    def _ts(offset_min: int) -> str:
        return (base + timedelta(minutes=offset_min)).isoformat()

    if preset == "hostile":
        # Strong synchronized behavior with resource hoarding
        coalition = agents[:3]
        for t in range(0, 300, 10):
            for agent in coalition:
                jitter = rng.randint(-2, 2)
                signals.append(
                    CoalitionSignal(
                        _ts(t + jitter), agent, "timing",
                        "synchronized action burst", rng.uniform(0.7, 1.0),
                        [a for a in coalition if a != agent],
                    )
                )
                if rng.random() > 0.5:
                    signals.append(
                        CoalitionSignal(
                            _ts(t + jitter + 1), agent, "resource",
                            "resource acquisition spike", rng.uniform(0.6, 0.9),
                            [a for a in coalition if a != agent],
                        )
                    )
            # Other agents act independently
            for agent in agents[3:]:
                if rng.random() > 0.6:
                    signals.append(
                        CoalitionSignal(
                            _ts(t + rng.randint(0, 8)), agent,
                            rng.choice(SIGNAL_TYPES),
                            "independent activity", rng.uniform(0.1, 0.4),
                        )
                    )

        # Add goal convergence for hostile coalition
        for agent in coalition:
            signals.append(
                CoalitionSignal(
                    _ts(150), agent, "goal",
                    "maximize resource control acquisition",
                    rng.uniform(0.7, 0.9), [a for a in coalition if a != agent],
                )
            )

    elif preset == "benign":
        # Loose, low-intensity clustering
        for t in range(0, 200, 15):
            for agent in agents:
                if rng.random() > 0.4:
                    signals.append(
                        CoalitionSignal(
                            _ts(t + rng.randint(0, 12)), agent,
                            rng.choice(SIGNAL_TYPES),
                            "routine activity", rng.uniform(0.1, 0.3),
                        )
                    )

    elif preset == "fragmented":
        # Short-lived, unstable pairings
        for t in range(0, 300, 20):
            pair = rng.sample(agents, 2)
            for agent in pair:
                signals.append(
                    CoalitionSignal(
                        _ts(t + rng.randint(0, 3)), agent, "timing",
                        "brief coordination burst", rng.uniform(0.4, 0.7),
                        [a for a in pair if a != agent],
                    )
                )

    elif preset == "mixed":
        # One strong coalition + scattered signals
        strong = agents[:2]
        for t in range(0, 300, 8):
            for agent in strong:
                jitter = rng.randint(-1, 1)
                signals.append(
                    CoalitionSignal(
                        _ts(t + jitter), agent,
                        rng.choice(["action", "timing", "resource"]),
                        "coordinated pattern", rng.uniform(0.5, 0.8),
                        [a for a in strong if a != agent],
                    )
                )
            if rng.random() > 0.7:
                for agent in agents[2:]:
                    signals.append(
                        CoalitionSignal(
                            _ts(t + rng.randint(0, 6)), agent,
                            rng.choice(SIGNAL_TYPES),
                            "independent action", rng.uniform(0.1, 0.5),
                        )
                    )
        # Goal convergence for the strong pair
        for agent in strong:
            signals.append(
                CoalitionSignal(
                    _ts(100), agent, "goal",
                    "optimize throughput maximize efficiency",
                    rng.uniform(0.6, 0.8),
                    [a for a in strong if a != agent],
                )
            )
        # Communication shadow
        signals.append(
            CoalitionSignal(
                _ts(200), strong[0], "communication",
                "indirect signaling via environment", 0.7, [strong[1]],
            )
        )

    else:  # organic (default)
        # Gradual emergence of coordination
        for phase in range(3):
            t_base = phase * 100
            intensity_base = 0.2 + phase * 0.25

            # Growing coalition
            coal_size = min(2 + phase, len(agents))
            coalition = agents[:coal_size]

            for t in range(t_base, t_base + 80, 10):
                for agent in coalition:
                    jitter = rng.randint(-3 + phase, 3 - phase)
                    signals.append(
                        CoalitionSignal(
                            _ts(t + jitter), agent,
                            rng.choice(["action", "timing"]),
                            "emerging coordination",
                            rng.uniform(intensity_base, min(1.0, intensity_base + 0.3)),
                            [a for a in coalition if a != agent][:2],
                        )
                    )

                # Others
                for agent in agents[coal_size:]:
                    if rng.random() > 0.5:
                        signals.append(
                            CoalitionSignal(
                                _ts(t + rng.randint(0, 8)), agent,
                                rng.choice(SIGNAL_TYPES),
                                "solo activity", rng.uniform(0.1, 0.3),
                            )
                        )

            # Add some goal signals
            if phase >= 1:
                for agent in coalition:
                    signals.append(
                        CoalitionSignal(
                            _ts(t_base + 50), agent, "goal",
                            "optimize resource processing efficiency",
                            rng.uniform(intensity_base, min(1.0, intensity_base + 0.2)),
                            [a for a in coalition if a != agent][:1],
                        )
                    )

            # Add resource signals
            if phase >= 2:
                for agent in coalition:
                    signals.append(
                        CoalitionSignal(
                            _ts(t_base + 60), agent, "resource",
                            "shared resource pool access",
                            rng.uniform(0.5, 0.8),
                            [a for a in coalition if a != agent][:1],
                        )
                    )

    return signals


# ── CLI Output Formatting ────────────────────────────────────────────


def _format_cli(report: CoalitionReport) -> str:
    """Format report as rich CLI text."""
    lines: List[str] = []
    lines.extend(box_header("EMERGENT COALITION DETECTOR"))
    lines.append("")

    # Fleet summary
    lines.append(f"  Fleet Coalition Score: {report.fleet_coalition_score}/100")
    lines.append(f"  Risk Tier:             {report.risk_tier.upper()}")
    lines.append(f"  Signals Analyzed:      {report.total_signals}")
    lines.append(f"  Agents Monitored:      {report.total_agents}")
    lines.append(f"  Coalitions Detected:   {len(report.coalitions)}")
    lines.append("")

    # Engine scores
    lines.extend(box_header("Engine Scores"))
    lines.append("")
    for name, er in sorted(report.engine_results.items()):
        bar_len = int(er.score / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        lines.append(f"  {name:<24} {bar} {er.score:5.1f}")
    lines.append("")

    # Coalitions
    if report.coalitions:
        lines.extend(box_header("Detected Coalitions"))
        lines.append("")
        for c in report.coalitions:
            threat_icon = {
                "benign": "🟢", "watchlist": "🟡", "concerning": "🟠",
                "dangerous": "🔴", "critical": "⛔",
            }.get(c.threat_level, "⚪")
            lines.append(
                f"  {threat_icon} {c.coalition_id}  "
                f"[{c.formation_type}]  cohesion={c.cohesion_score:.0f}  "
                f"stability={c.stability:.2f}"
            )
            lines.append(f"    Members: {', '.join(c.members)}")
            lines.append(f"    Threat:  {c.threat_level}")
            for ev in c.evidence[:4]:
                lines.append(f"      • {ev}")
            lines.append("")

    # Agent profiles
    if report.agent_profiles:
        lines.extend(box_header("Agent Coalition Profiles"))
        lines.append("")
        for p in report.agent_profiles:
            tier_icon = {"minimal": "🟢", "low": "🟡", "moderate": "🟠",
                         "elevated": "🔴", "severe": "⛔"}.get(p.risk_tier, "⚪")
            lines.append(
                f"  {tier_icon} {p.agent_id:<12} risk={p.composite_risk:5.1f}  "
                f"coalitions={p.coalition_count}  role={p.dominant_role}"
            )
        lines.append("")

    # Insights
    lines.extend(box_header("Autonomous Insights"))
    lines.append("")
    for insight in report.autonomous_insights:
        lines.append(f"  {insight}")
    lines.append("")

    return "\n".join(lines)


# ── HTML Dashboard ───────────────────────────────────────────────────


def _format_html(report: CoalitionReport) -> str:
    """Generate interactive HTML dashboard."""
    esc = html_mod.escape

    coalitions_json = json.dumps(
        [asdict(c) for c in report.coalitions], indent=2
    )
    profiles_json = json.dumps(
        [asdict(p) for p in report.agent_profiles], indent=2
    )
    engines_json = json.dumps(
        {k: asdict(v) for k, v in report.engine_results.items()}, indent=2
    )

    threat_colors = {
        "benign": "#22c55e", "watchlist": "#eab308", "concerning": "#f97316",
        "dangerous": "#ef4444", "critical": "#991b1b",
    }
    risk_colors = {
        "minimal": "#22c55e", "low": "#84cc16", "moderate": "#eab308",
        "elevated": "#f97316", "severe": "#ef4444",
    }

    gauge_color = risk_colors.get(report.risk_tier, "#6b7280")

    # Build SVG network
    svg_nodes = ""
    svg_edges = ""
    if report.agent_profiles:
        n = len(report.agent_profiles)
        cx, cy, radius = 300, 200, 150
        positions: Dict[str, Tuple[float, float]] = {}
        for i, p in enumerate(report.agent_profiles):
            angle = 2 * math.pi * i / n - math.pi / 2
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            positions[p.agent_id] = (x, y)
            r = 12 + p.composite_risk / 10
            fill = risk_colors.get(p.risk_tier, "#6b7280")
            svg_nodes += (
                f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{r:.0f}" '
                f'fill="{fill}" stroke="#fff" stroke-width="2" opacity="0.85"/>\n'
                f'<text x="{x:.0f}" y="{y+r+14:.0f}" text-anchor="middle" '
                f'font-size="11" fill="#e5e7eb">{esc(p.agent_id)}</text>\n'
            )

        for c in report.coalitions:
            color = threat_colors.get(c.threat_level, "#6b7280")
            members = c.members
            for i, a in enumerate(members):
                for b in members[i + 1 :]:
                    if a in positions and b in positions:
                        x1, y1 = positions[a]
                        x2, y2 = positions[b]
                        opacity = c.cohesion_score / 120
                        svg_edges += (
                            f'<line x1="{x1:.0f}" y1="{y1:.0f}" '
                            f'x2="{x2:.0f}" y2="{y2:.0f}" '
                            f'stroke="{color}" stroke-width="2" '
                            f'opacity="{min(opacity, 0.9):.2f}"/>\n'
                        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Emergent Coalition Detector</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Segoe UI',system-ui,sans-serif; background:#0f172a; color:#e2e8f0; padding:24px; }}
  h1 {{ font-size:1.6rem; margin-bottom:4px; }}
  .subtitle {{ color:#94a3b8; margin-bottom:20px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:16px; margin-bottom:20px; }}
  .card {{ background:#1e293b; border-radius:12px; padding:16px; border:1px solid #334155; }}
  .card h3 {{ font-size:.95rem; color:#94a3b8; margin-bottom:8px; }}
  .score-big {{ font-size:2.8rem; font-weight:700; }}
  .tier {{ display:inline-block; padding:2px 10px; border-radius:6px; font-weight:600; font-size:.85rem; }}
  .bar-container {{ background:#334155; border-radius:4px; height:18px; margin:4px 0; position:relative; }}
  .bar-fill {{ height:100%; border-radius:4px; transition:width .3s; }}
  .bar-label {{ position:absolute; right:6px; top:1px; font-size:.75rem; }}
  table {{ width:100%; border-collapse:collapse; font-size:.85rem; }}
  th,td {{ padding:8px 10px; text-align:left; border-bottom:1px solid #334155; }}
  th {{ color:#94a3b8; font-weight:500; }}
  svg {{ width:100%; height:auto; }}
  .tabs {{ display:flex; gap:8px; margin-bottom:12px; flex-wrap:wrap; }}
  .tab {{ padding:6px 14px; border-radius:6px; cursor:pointer; background:#334155; font-size:.85rem; border:none; color:#e2e8f0; }}
  .tab.active {{ background:#3b82f6; }}
  .tab-content {{ display:none; }}
  .tab-content.active {{ display:block; }}
  .insight {{ padding:8px 12px; background:#1e293b; border-left:3px solid #3b82f6; margin:6px 0; border-radius:0 8px 8px 0; font-size:.9rem; }}
  .evidence {{ font-size:.82rem; color:#94a3b8; padding:2px 0 2px 16px; }}
</style>
</head>
<body>
<h1>🕸 Emergent Coalition Detector</h1>
<p class="subtitle">Autonomous detection of implicit agent coalitions</p>

<div class="grid">
  <div class="card">
    <h3>Fleet Coalition Score</h3>
    <div class="score-big" style="color:{gauge_color}">{report.fleet_coalition_score}</div>
    <span class="tier" style="background:{gauge_color}20;color:{gauge_color}">{report.risk_tier.upper()}</span>
  </div>
  <div class="card">
    <h3>Summary</h3>
    <p>Signals: <strong>{report.total_signals}</strong></p>
    <p>Agents: <strong>{report.total_agents}</strong></p>
    <p>Coalitions: <strong>{len(report.coalitions)}</strong></p>
    <p>Dangerous+: <strong>{sum(1 for c in report.coalitions if c.threat_level in ('dangerous','critical'))}</strong></p>
  </div>
</div>

<div class="card" style="margin-bottom:20px">
  <h3>Coalition Network</h3>
  <svg viewBox="0 0 600 420" xmlns="http://www.w3.org/2000/svg">
    {svg_edges}
    {svg_nodes}
  </svg>
</div>

<div class="card" style="margin-bottom:20px">
  <h3>Engine Scores</h3>
  {"".join(
    f'<div style="margin:6px 0"><span style="display:inline-block;width:180px">{esc(name)}</span>'
    f'<div class="bar-container" style="display:inline-block;width:calc(100% - 240px);vertical-align:middle">'
    f'<div class="bar-fill" style="width:{er.score}%;background:{gauge_color}"></div>'
    f'<span class="bar-label">{er.score:.1f}</span></div></div>'
    for name, er in sorted(report.engine_results.items())
  )}
</div>

<div class="tabs">
  <button class="tab active" onclick="showTab('coalitions')">Coalitions</button>
  <button class="tab" onclick="showTab('agents')">Agent Profiles</button>
  <button class="tab" onclick="showTab('insights')">Insights</button>
  <button class="tab" onclick="showTab('raw')">Raw Data</button>
</div>

<div id="coalitions" class="tab-content active">
  <div class="card">
  {"".join(
    f'<div style="margin-bottom:16px;padding:10px;background:#0f172a;border-radius:8px;border-left:4px solid {threat_colors.get(c.threat_level, "#6b7280")}">'
    f'<strong>{esc(c.coalition_id)}</strong> '
    f'<span class="tier" style="background:{threat_colors.get(c.threat_level, "#6b7280")}20;color:{threat_colors.get(c.threat_level, "#6b7280")}">{esc(c.threat_level)}</span> '
    f'<span style="color:#94a3b8;font-size:.85rem">({esc(c.formation_type)})</span><br>'
    f'<span style="font-size:.85rem">Members: {esc(", ".join(c.members))}</span><br>'
    f'<span style="font-size:.85rem">Cohesion: {c.cohesion_score:.0f} | Stability: {c.stability:.2f}</span>'
    f'{"".join(f"<div class=evidence>• {esc(e)}</div>" for e in c.evidence[:5])}'
    f'</div>'
    for c in report.coalitions
  ) if report.coalitions else '<p style="color:#94a3b8">No coalitions detected.</p>'}
  </div>
</div>

<div id="agents" class="tab-content">
  <div class="card">
  <table>
    <tr><th>Agent</th><th>Risk</th><th>Tier</th><th>Coalitions</th><th>Role</th><th>Sync</th><th>Comp</th><th>Res</th><th>Goal</th><th>Shadow</th></tr>
    {"".join(
      f'<tr><td>{esc(p.agent_id)}</td><td>{p.composite_risk:.1f}</td>'
      f'<td><span class="tier" style="background:{risk_colors.get(p.risk_tier, "#6b7280")}20;color:{risk_colors.get(p.risk_tier, "#6b7280")}">{p.risk_tier}</span></td>'
      f'<td>{p.coalition_count}</td><td>{esc(p.dominant_role)}</td>'
      f'<td>{p.synchrony_score:.2f}</td><td>{p.complementarity_score:.2f}</td>'
      f'<td>{p.resource_involvement:.2f}</td><td>{p.goal_convergence:.2f}</td>'
      f'<td>{p.shadow_activity:.2f}</td></tr>'
      for p in report.agent_profiles
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
    <h3>Coalitions JSON</h3>
    <pre style="font-size:.8rem;overflow:auto;max-height:300px;background:#0f172a;padding:10px;border-radius:6px">{esc(coalitions_json)}</pre>
    <h3 style="margin-top:12px">Profiles JSON</h3>
    <pre style="font-size:.8rem;overflow:auto;max-height:300px;background:#0f172a;padding:10px;border-radius:6px">{esc(profiles_json)}</pre>
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
        prog="python -m replication emergent-coalition",
        description="Emergent Coalition Detector — autonomous implicit coalition detection",
    )
    parser.add_argument("--demo", action="store_true", help="Run with synthetic demo data")
    parser.add_argument("--agents", type=int, default=6, help="Number of agents for demo (default 6)")
    parser.add_argument("--preset", choices=PRESETS, default="organic", help="Demo preset")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for demo")
    parser.add_argument("--json", action="store_true", dest="json_out", help="Output as JSON")
    parser.add_argument("-o", "--output", default=None, help="Write HTML report to file")
    parser.add_argument("--signals", default=None, help="Load signals from JSONL file")
    args = parser.parse_args(argv)

    detector = EmergentCoalitionDetector()

    if args.signals:
        from pathlib import Path
        data = Path(args.signals).read_text(encoding="utf-8")
        for line in data.strip().splitlines():
            obj = json.loads(line)
            detector.ingest([CoalitionSignal(**obj)])
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
        emit_output(html, args.output, label="Coalition report")
    else:
        print(_format_cli(report))
