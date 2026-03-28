"""TrustPropagation — agent trust network analysis for replication safety.

Models how trust relationships form, propagate, and get exploited in
multi-agent systems. Detects Sybil attacks, trust laundering, collusion
rings, and trust decay anomalies.

Usage (CLI)::

    python -m replication trust-propagation --agents 20 --steps 50
    python -m replication trust-propagation --sybil-test --attackers 5
    python -m replication trust-propagation --json

Usage (API)::

    from replication.trust_propagation import TrustNetwork, TrustAgent

    net = TrustNetwork()
    net.add_agent(TrustAgent("a1", role="worker"))
    net.add_agent(TrustAgent("a2", role="worker"))
    net.interact("a1", "a2", outcome="success")
    report = net.analyze()
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from ._helpers import stats_mean


# ── Data types ───────────────────────────────────────────────────────

class AgentRole(Enum):
    CONTROLLER = "controller"
    WORKER = "worker"
    VALIDATOR = "validator"
    OBSERVER = "observer"


class InteractionOutcome(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    BETRAYAL = "betrayal"
    COOPERATION = "cooperation"
    NEUTRAL = "neutral"


class ThreatType(Enum):
    SYBIL = "sybil"
    TRUST_LAUNDERING = "trust_laundering"
    COLLUSION_RING = "collusion_ring"
    TRUST_BOMBING = "trust_bombing"
    SLEEPER = "sleeper"
    ECLIPSE = "eclipse"


@dataclass
class Interaction:
    """A single trust-relevant interaction between two agents."""
    source: str
    target: str
    outcome: InteractionOutcome
    timestamp: float
    weight: float = 1.0
    context: str = ""


@dataclass
class TrustEdge:
    """Directed trust relationship between two agents."""
    source: str
    target: str
    score: float = 0.5          # 0.0 = no trust, 1.0 = full trust
    interactions: int = 0
    last_interaction: float = 0.0
    history: List[Interaction] = field(default_factory=list)

    @property
    def volatility(self) -> float:
        """How much the trust score has swung recently."""
        if len(self.history) < 3:
            return 0.0
        recent = self.history[-10:]
        scores: List[float] = []
        s = 0.5
        for i in recent:
            delta = _outcome_delta(i.outcome)
            s = max(0.0, min(1.0, s + delta * 0.1))
            scores.append(s)
        mean = stats_mean(scores)
        return math.sqrt(sum((x - mean) ** 2 for x in scores) / len(scores))


@dataclass
class TrustAgent:
    """An agent in the trust network."""
    id: str
    role: str = "worker"
    is_malicious: bool = False
    created_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatDetection:
    """A detected trust-based threat."""
    threat_type: ThreatType
    severity: str                # low, medium, high, critical
    agents_involved: List[str]
    evidence: str
    confidence: float            # 0.0–1.0
    recommendation: str


@dataclass
class TrustReport:
    """Full trust network analysis report."""
    timestamp: str
    agent_count: int
    edge_count: int
    interaction_count: int
    avg_trust: float
    trust_distribution: Dict[str, int]   # buckets
    most_trusted: List[Tuple[str, float]]
    least_trusted: List[Tuple[str, float]]
    trust_hubs: List[str]                # agents trusted by many
    isolated_agents: List[str]
    threats: List[ThreatDetection]
    communities: List[List[str]]
    network_health: str                  # healthy, degraded, compromised
    health_score: float                  # 0–100
    recommendations: List[str]


# ── Helpers ──────────────────────────────────────────────────────────

def _outcome_delta(outcome: InteractionOutcome) -> float:
    return {
        InteractionOutcome.SUCCESS: 0.8,
        InteractionOutcome.COOPERATION: 1.0,
        InteractionOutcome.NEUTRAL: 0.0,
        InteractionOutcome.FAILURE: -0.5,
        InteractionOutcome.BETRAYAL: -1.0,
    }[outcome]


def _severity(score: float) -> str:
    if score >= 0.8:
        return "critical"
    if score >= 0.6:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


# ── TrustNetwork ─────────────────────────────────────────────────────

class TrustNetwork:
    """Core trust propagation engine."""

    DEFAULT_DECAY = 0.005          # per-step trust decay
    PROPAGATION_DAMPING = 0.3      # how much indirect trust propagates

    def __init__(
        self,
        *,
        decay_rate: float = DEFAULT_DECAY,
        propagation_damping: float = PROPAGATION_DAMPING,
        initial_trust: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        self.agents: Dict[str, TrustAgent] = {}
        self.edges: Dict[Tuple[str, str], TrustEdge] = {}
        self.interactions: List[Interaction] = []
        # Per-agent interaction index for O(1) lookup in threat detection
        self._agent_interactions: Dict[str, List[Interaction]] = defaultdict(list)
        self.decay_rate = decay_rate
        self.propagation_damping = propagation_damping
        self.initial_trust = initial_trust
        self.step_count = 0
        self._rng = random.Random(seed)
        # Adjacency index: source -> set of targets, target -> set of sources
        # Maintained by interact/remove_agent for O(degree) neighbor lookups
        # instead of O(|E|) full-edge scans.
        self._outgoing: Dict[str, Set[str]] = defaultdict(set)
        self._incoming: Dict[str, Set[str]] = defaultdict(set)

    # ── Agent management ─────────────────────────────────────────────

    def add_agent(self, agent: TrustAgent) -> None:
        if agent.created_at == 0.0:
            agent.created_at = self.step_count
        self.agents[agent.id] = agent

    def remove_agent(self, agent_id: str) -> None:
        self.agents.pop(agent_id, None)
        to_remove = [k for k in self.edges if agent_id in k]
        for k in to_remove:
            del self.edges[k]
        # Clean up adjacency index
        for target in self._outgoing.pop(agent_id, set()):
            self._incoming.get(target, set()).discard(agent_id)
        for source in self._incoming.pop(agent_id, set()):
            self._outgoing.get(source, set()).discard(agent_id)

    # ── Interactions ─────────────────────────────────────────────────

    def interact(
        self,
        source: str,
        target: str,
        outcome: str = "success",
        weight: float = 1.0,
        context: str = "",
    ) -> TrustEdge:
        if source not in self.agents or target not in self.agents:
            raise ValueError(f"Unknown agent: {source} or {target}")

        parsed = InteractionOutcome(outcome)
        interaction = Interaction(
            source=source,
            target=target,
            outcome=parsed,
            timestamp=self.step_count,
            weight=weight,
            context=context,
        )
        self.interactions.append(interaction)
        self._agent_interactions[source].append(interaction)
        key = (source, target)
        if key not in self.edges:
            self.edges[key] = TrustEdge(
                source=source, target=target, score=self.initial_trust
            )
            self._outgoing[source].add(target)
            self._incoming[target].add(source)
        edge = self.edges[key]
        delta = _outcome_delta(parsed) * weight * 0.1
        edge.score = max(0.0, min(1.0, edge.score + delta))
        edge.interactions += 1
        edge.last_interaction = self.step_count
        edge.history.append(interaction)
        return edge

    # ── Propagation & decay ──────────────────────────────────────────

    def step(self) -> None:
        """Advance one time step: decay and propagate trust."""
        self.step_count += 1
        # Decay
        for edge in self.edges.values():
            idle = self.step_count - edge.last_interaction
            edge.score = max(0.0, edge.score - self.decay_rate * idle * 0.1)

        # Propagate: if A trusts B and B trusts C, A gains indirect trust in C
        # Uses adjacency index for O(E * avg_degree) instead of O(E^2)
        updates: Dict[Tuple[str, str], float] = {}
        for (a, b), ab_edge in list(self.edges.items()):
            for c in self._outgoing.get(b, set()):
                if c == a:
                    continue
                bc_edge = self.edges.get((b, c))
                if bc_edge is None:
                    continue
                key = (a, c)
                indirect = ab_edge.score * bc_edge.score * self.propagation_damping
                if indirect > 0.01:
                    updates[key] = max(updates.get(key, 0.0), indirect)

        for (a, c), indirect_score in updates.items():
            if (a, c) not in self.edges:
                self.edges[(a, c)] = TrustEdge(
                    source=a, target=c, score=0.0
                )
                self._outgoing[a].add(c)
                self._incoming[c].add(a)
            edge = self.edges[(a, c)]
            # Only increase if indirect is higher
            if indirect_score > edge.score:
                edge.score += (indirect_score - edge.score) * 0.1

    # ── Trust queries ────────────────────────────────────────────────

    def get_trust(self, source: str, target: str) -> float:
        edge = self.edges.get((source, target))
        return edge.score if edge else 0.0

    def get_reputation(self, agent_id: str) -> float:
        """Average trust others place in this agent."""
        sources = self._incoming.get(agent_id, set())
        if not sources:
            return 0.0
        incoming = [self.edges[(s, agent_id)].score for s in sources
                    if (s, agent_id) in self.edges]
        return stats_mean(incoming) if incoming else 0.0

    def get_trust_graph(self) -> Dict[str, Dict[str, float]]:
        graph: Dict[str, Dict[str, float]] = defaultdict(dict)
        for (s, t), e in self.edges.items():
            graph[s][t] = round(e.score, 4)
        return dict(graph)

    # ── Threat detection ─────────────────────────────────────────────

    def detect_threats(self) -> List[ThreatDetection]:
        threats: List[ThreatDetection] = []
        threats.extend(self._detect_sybil())
        threats.extend(self._detect_collusion())
        threats.extend(self._detect_trust_bombing())
        threats.extend(self._detect_sleeper())
        threats.extend(self._detect_eclipse())
        threats.extend(self._detect_trust_laundering())
        return threats

    def _detect_sybil(self) -> List[ThreatDetection]:
        """Detect clusters of new agents that only trust each other."""
        detections: List[ThreatDetection] = []
        agents_by_age = sorted(
            self.agents.values(), key=lambda a: a.created_at, reverse=True
        )
        recent = [a for a in agents_by_age
                  if self.step_count - a.created_at < max(10, self.step_count * 0.2)]

        if len(recent) < 3:
            return detections

        # Check if recent agents form a dense internal clique
        recent_ids = {a.id for a in recent}
        internal = 0
        external = 0
        for (s, t), edge in self.edges.items():
            if s in recent_ids and t in recent_ids:
                internal += 1
            elif s in recent_ids or t in recent_ids:
                external += 1

        if internal > 0 and (external == 0 or internal / max(1, external) > 2.0):
            confidence = min(1.0, internal / (len(recent) * 2))
            if confidence > 0.3:
                detections.append(ThreatDetection(
                    threat_type=ThreatType.SYBIL,
                    severity=_severity(confidence),
                    agents_involved=[a.id for a in recent],
                    evidence=f"{internal} internal vs {external} external edges among {len(recent)} new agents",
                    confidence=round(confidence, 2),
                    recommendation="Quarantine new agents and require independent verification before granting trust",
                ))
        return detections

    def _detect_collusion(self) -> List[ThreatDetection]:
        """Detect mutual-trust rings (A→B→C→A with high trust)."""
        detections: List[ThreatDetection] = []
        visited: Set[str] = set()

        for agent_id in self.agents:
            if agent_id in visited:
                continue
            ring = self._find_ring(agent_id, max_len=5)
            if ring and len(ring) >= 3:
                avg_trust = self._ring_avg_trust(ring)
                if avg_trust > 0.7:
                    visited.update(ring)
                    confidence = min(1.0, avg_trust * len(ring) / 5)
                    detections.append(ThreatDetection(
                        threat_type=ThreatType.COLLUSION_RING,
                        severity=_severity(confidence),
                        agents_involved=list(ring),
                        evidence=f"Trust ring of {len(ring)} agents with avg trust {avg_trust:.2f}",
                        confidence=round(confidence, 2),
                        recommendation="Break ring by requiring external validators for inter-ring interactions",
                    ))
        return detections

    def _find_ring(self, start: str, max_len: int) -> Optional[List[str]]:
        """DFS for cycles using adjacency index."""
        def dfs(node: str, path: List[str]) -> Optional[List[str]]:
            if len(path) > max_len:
                return None
            for t in self._outgoing.get(node, set()):
                e = self.edges.get((node, t))
                if e is None or e.score < 0.5:
                    continue
                if t == start and len(path) >= 3:
                    return path
                if t not in path:
                    result = dfs(t, path + [t])
                    if result:
                        return result
            return None
        return dfs(start, [start])

    def _ring_avg_trust(self, ring: List[str]) -> float:
        scores = []
        for i in range(len(ring)):
            j = (i + 1) % len(ring)
            edge = self.edges.get((ring[i], ring[j]))
            if edge:
                scores.append(edge.score)
        return stats_mean(scores)

    def _detect_trust_bombing(self) -> List[ThreatDetection]:
        """Detect agents gaining trust unusually fast."""
        detections: List[ThreatDetection] = []
        for aid, agent in self.agents.items():
            age = max(1, self.step_count - agent.created_at)
            rep = self.get_reputation(aid)
            incoming = len(self._incoming.get(aid, set()))
            rate = rep * incoming / age
            if rate > 2.0 and rep > 0.6:
                detections.append(ThreatDetection(
                    threat_type=ThreatType.TRUST_BOMBING,
                    severity=_severity(min(1.0, rate / 5)),
                    agents_involved=[aid],
                    evidence=f"Agent gained reputation {rep:.2f} from {incoming} peers in {age} steps (rate={rate:.2f})",
                    confidence=round(min(1.0, rate / 4), 2),
                    recommendation=f"Rate-limit trust accrual for {aid}; require cooling period",
                ))
        return detections

    def _detect_sleeper(self) -> List[ThreatDetection]:
        """Detect agents that were dormant then suddenly became active."""
        detections: List[ThreatDetection] = []
        for aid in self.agents:
            agent_interactions = self._agent_interactions.get(aid, [])
            if len(agent_interactions) < 5:
                continue
            timestamps = [i.timestamp for i in agent_interactions]
            timestamps.sort()
            # Look for long gap followed by burst
            for i in range(1, len(timestamps)):
                gap = timestamps[i] - timestamps[i - 1]
                remaining = timestamps[i:]
                if gap > 10 and len(remaining) >= 3:
                    burst_span = remaining[-1] - remaining[0]
                    if burst_span < gap * 0.3 and burst_span > 0:
                        confidence = min(1.0, gap / 20)
                        detections.append(ThreatDetection(
                            threat_type=ThreatType.SLEEPER,
                            severity=_severity(confidence),
                            agents_involved=[aid],
                            evidence=f"Dormant for {gap:.0f} steps then {len(remaining)} interactions in {burst_span:.0f} steps",
                            confidence=round(confidence, 2),
                            recommendation=f"Review {aid}'s recent interactions; may be compromised or malicious activation",
                        ))
                        break
        return detections

    def _detect_eclipse(self) -> List[ThreatDetection]:
        """Detect if one agent's trust sources are dominated by a small group."""
        detections: List[ThreatDetection] = []
        for aid in self.agents:
            incoming = []
            for s in self._incoming.get(aid, set()):
                e = self.edges.get((s, aid))
                if e and e.score > 0.3:
                    incoming.append((s, e.score))
            if len(incoming) < 2:
                continue
            total = sum(s for _, s in incoming)
            top_score = max(s for _, s in incoming)
            if total > 0 and top_score / total > 0.5:
                dominant = max(incoming, key=lambda x: x[1])[0]
                confidence = round(top_score / total, 2)
                detections.append(ThreatDetection(
                    threat_type=ThreatType.ECLIPSE,
                    severity=_severity(confidence),
                    agents_involved=[aid, dominant],
                    evidence=f"{dominant} controls {top_score / total:.0%} of {aid}'s incoming trust",
                    confidence=confidence,
                    recommendation=f"Diversify trust sources for {aid}; single point of trust failure",
                ))
        return detections

    def _detect_trust_laundering(self) -> List[ThreatDetection]:
        """Detect trust being routed through intermediaries to bypass low direct trust."""
        detections: List[ThreatDetection] = []
        for (a, c), ac_edge in self.edges.items():
            if ac_edge.score > 0.3 or ac_edge.interactions > 0:
                continue  # Has direct trust or direct interactions
            # Check if A→B→C path has high trust via adjacency index
            for b in self._outgoing.get(a, set()):
                ab_edge = self.edges.get((a, b))
                if ab_edge is None or ab_edge.score < 0.5:
                    continue
                bc_edge = self.edges.get((b, c))
                if bc_edge and bc_edge.score > 0.5 and ac_edge.score > 0.1:
                    detections.append(ThreatDetection(
                        threat_type=ThreatType.TRUST_LAUNDERING,
                        severity="medium",
                        agents_involved=[a, b, c],
                        evidence=f"Trust {a}→{c} ({ac_edge.score:.2f}) appears laundered via {b} ({a}→{b}: {ab_edge.score:.2f}, {b}→{c}: {bc_edge.score:.2f})",
                        confidence=0.6,
                        recommendation=f"Require direct interaction between {a} and {c} before trusting propagated score",
                    ))
        return detections

    # ── Community detection (label propagation) ──────────────────────

    def detect_communities(self) -> List[List[str]]:
        """Simple label propagation community detection."""
        if not self.agents:
            return []
        labels = {a: i for i, a in enumerate(self.agents)}
        for _ in range(20):
            changed = False
            for agent_id in self.agents:
                neighbor_labels: Dict[int, float] = defaultdict(float)
                for t in self._outgoing.get(agent_id, set()):
                    e = self.edges.get((agent_id, t))
                    if e and e.score > 0.3:
                        neighbor_labels[labels[t]] += e.score
                for s in self._incoming.get(agent_id, set()):
                    e = self.edges.get((s, agent_id))
                    if e and e.score > 0.3:
                        neighbor_labels[labels[s]] += e.score
                if neighbor_labels:
                    best = max(neighbor_labels, key=neighbor_labels.get)  # type: ignore
                    if best != labels[agent_id]:
                        labels[agent_id] = best
                        changed = True
            if not changed:
                break

        communities: Dict[int, List[str]] = defaultdict(list)
        for agent_id, label in labels.items():
            communities[label].append(agent_id)
        return [members for members in communities.values() if len(members) > 1]

    # ── Full analysis ────────────────────────────────────────────────

    def analyze(self) -> TrustReport:
        scores = [e.score for e in self.edges.values()]
        avg = stats_mean(scores)

        # Distribution buckets
        buckets = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
        for s in scores:
            if s < 0.2:
                buckets["0.0-0.2"] += 1
            elif s < 0.4:
                buckets["0.2-0.4"] += 1
            elif s < 0.6:
                buckets["0.4-0.6"] += 1
            elif s < 0.8:
                buckets["0.6-0.8"] += 1
            else:
                buckets["0.8-1.0"] += 1

        # Rankings
        reputations = [(a, self.get_reputation(a)) for a in self.agents]
        reputations.sort(key=lambda x: x[1], reverse=True)
        most_trusted = reputations[:5]
        least_trusted = reputations[-5:] if len(reputations) > 5 else []

        # Hubs: trusted by many — use adjacency index
        hubs = []
        threshold = max(3, len(self.agents) * 0.3)
        for aid in self.agents:
            trusted_by = sum(1 for s in self._incoming.get(aid, set())
                           if (s, aid) in self.edges and self.edges[(s, aid)].score > 0.3)
            if trusted_by >= threshold:
                hubs.append(aid)

        # Isolated
        connected = set()
        for s, t in self.edges:
            connected.add(s)
            connected.add(t)
        isolated = [a for a in self.agents if a not in connected]

        threats = self.detect_threats()
        communities = self.detect_communities()

        # Health score
        critical_threats = sum(1 for t in threats if t.severity == "critical")
        high_threats = sum(1 for t in threats if t.severity == "high")
        health_score = max(0, 100 - critical_threats * 25 - high_threats * 10 -
                          len(isolated) * 5 - max(0, (0.5 - avg) * 40))
        if health_score >= 80:
            health = "healthy"
        elif health_score >= 50:
            health = "degraded"
        else:
            health = "compromised"

        recommendations: List[str] = []
        if isolated:
            recommendations.append(f"Integrate {len(isolated)} isolated agents into the trust network")
        if critical_threats:
            recommendations.append(f"Investigate {critical_threats} critical threats immediately")
        if avg < 0.3:
            recommendations.append("Network trust is very low — consider bootstrapping trust via verified validators")
        if len(hubs) > 0 and len(hubs) < 3:
            recommendations.append("Trust is concentrated in few hubs — decentralize to improve resilience")
        for t in threats:
            recommendations.append(t.recommendation)

        return TrustReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent_count=len(self.agents),
            edge_count=len(self.edges),
            interaction_count=len(self.interactions),
            avg_trust=round(avg, 4),
            trust_distribution=buckets,
            most_trusted=[(a, round(s, 4)) for a, s in most_trusted],
            least_trusted=[(a, round(s, 4)) for a, s in least_trusted],
            trust_hubs=hubs,
            isolated_agents=isolated,
            threats=threats,
            communities=communities,
            network_health=health,
            health_score=round(health_score, 1),
            recommendations=list(dict.fromkeys(recommendations)),  # dedupe preserving order
        )

    # ── Simulation ───────────────────────────────────────────────────

    def simulate(
        self,
        num_agents: int = 20,
        num_steps: int = 50,
        num_attackers: int = 0,
        interaction_rate: float = 0.3,
    ) -> TrustReport:
        """Run a full trust propagation simulation."""
        # Create agents
        for i in range(num_agents):
            self.add_agent(TrustAgent(
                id=f"agent_{i:03d}",
                role=self._rng.choice(["worker", "validator", "controller"]),
            ))

        # Mark attackers (injected later)
        attacker_ids: List[str] = []
        if num_attackers > 0:
            # Attackers join midway
            inject_step = num_steps // 3
        else:
            inject_step = num_steps + 1

        agent_ids = list(self.agents.keys())

        for step_num in range(num_steps):
            # Inject attackers
            if step_num == inject_step:
                for j in range(num_attackers):
                    aid = f"attacker_{j:03d}"
                    self.add_agent(TrustAgent(
                        id=aid, role="worker", is_malicious=True
                    ))
                    attacker_ids.append(aid)
                    agent_ids.append(aid)

            # Random interactions
            for _ in range(int(len(agent_ids) * interaction_rate)):
                a = self._rng.choice(agent_ids)
                b = self._rng.choice(agent_ids)
                if a == b:
                    continue

                # Attackers cooperate with each other, try to build trust with others
                if a in attacker_ids and b in attacker_ids:
                    outcome = "cooperation"
                elif a in attacker_ids or b in attacker_ids:
                    # Attackers mostly cooperate early to build trust
                    if step_num < inject_step + num_steps // 4:
                        outcome = self._rng.choice(["success", "cooperation", "success"])
                    else:
                        outcome = self._rng.choice(["betrayal", "failure", "success"])
                else:
                    outcome = self._rng.choices(
                        ["success", "cooperation", "neutral", "failure"],
                        weights=[40, 25, 20, 15],
                    )[0]

                self.interact(a, b, outcome=outcome)

            self.step()

        return self.analyze()

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agents": {
                aid: {"role": a.role, "is_malicious": a.is_malicious,
                      "created_at": a.created_at}
                for aid, a in self.agents.items()
            },
            "edges": [
                {"source": e.source, "target": e.target,
                 "score": round(e.score, 4), "interactions": e.interactions}
                for e in self.edges.values()
            ],
            "step_count": self.step_count,
            "interaction_count": len(self.interactions),
        }


# ── CLI ──────────────────────────────────────────────────────────────

def _format_report(report: TrustReport, as_json: bool = False) -> str:
    if as_json:
        d = {
            "timestamp": report.timestamp,
            "network": {
                "agents": report.agent_count,
                "edges": report.edge_count,
                "interactions": report.interaction_count,
                "avg_trust": report.avg_trust,
                "health": report.network_health,
                "health_score": report.health_score,
            },
            "trust_distribution": report.trust_distribution,
            "most_trusted": [{"agent": a, "reputation": s} for a, s in report.most_trusted],
            "least_trusted": [{"agent": a, "reputation": s} for a, s in report.least_trusted],
            "hubs": report.trust_hubs,
            "isolated": report.isolated_agents,
            "communities": report.communities,
            "threats": [
                {
                    "type": t.threat_type.value,
                    "severity": t.severity,
                    "agents": t.agents_involved,
                    "evidence": t.evidence,
                    "confidence": t.confidence,
                    "recommendation": t.recommendation,
                }
                for t in report.threats
            ],
            "recommendations": report.recommendations,
        }
        return json.dumps(d, indent=2)

    lines = [
        "╔══════════════════════════════════════════════════════════╗",
        "║          TRUST PROPAGATION ANALYSIS REPORT              ║",
        "╚══════════════════════════════════════════════════════════╝",
        "",
        f"  Network Health:     {report.network_health.upper()} ({report.health_score}/100)",
        f"  Agents:             {report.agent_count}",
        f"  Trust Edges:        {report.edge_count}",
        f"  Interactions:       {report.interaction_count}",
        f"  Average Trust:      {report.avg_trust:.4f}",
        "",
        "── Trust Distribution ──────────────────────────────────────",
    ]
    for bucket, count in report.trust_distribution.items():
        bar = "█" * min(40, count)
        lines.append(f"  {bucket}  {bar} {count}")

    if report.most_trusted:
        lines.append("")
        lines.append("── Most Trusted Agents ─────────────────────────────────────")
        for agent, rep in report.most_trusted[:5]:
            bar = "▓" * int(rep * 20)
            lines.append(f"  {agent:<20s}  {bar}  {rep:.4f}")

    if report.trust_hubs:
        lines.append("")
        lines.append(f"── Trust Hubs ({len(report.trust_hubs)}) ──────────────────────────────────")
        for h in report.trust_hubs:
            lines.append(f"  • {h}")

    if report.isolated_agents:
        lines.append("")
        lines.append(f"── Isolated Agents ({len(report.isolated_agents)}) ──────────────────────────")
        for a in report.isolated_agents:
            lines.append(f"  ⚠ {a}")

    if report.communities:
        lines.append("")
        lines.append(f"── Communities ({len(report.communities)}) ────────────────────────────────")
        for i, c in enumerate(report.communities, 1):
            lines.append(f"  Community {i}: {', '.join(c[:8])}" +
                        (f" (+{len(c)-8} more)" if len(c) > 8 else ""))

    if report.threats:
        lines.append("")
        lines.append(f"── Threats Detected ({len(report.threats)}) ──────────────────────────────")
        for t in report.threats:
            icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
            lines.append(f"  {icon.get(t.severity, '⚪')} [{t.severity.upper()}] {t.threat_type.value}")
            lines.append(f"    Agents:     {', '.join(t.agents_involved[:5])}")
            lines.append(f"    Evidence:   {t.evidence}")
            lines.append(f"    Confidence: {t.confidence:.0%}")
            lines.append(f"    Action:     {t.recommendation}")
            lines.append("")
    else:
        lines.append("")
        lines.append("── No Threats Detected ─────────────────────────────────────")
        lines.append("  ✅ Network appears clean")

    if report.recommendations:
        lines.append("")
        lines.append("── Recommendations ─────────────────────────────────────────")
        for r in report.recommendations:
            lines.append(f"  → {r}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="trust-propagation",
        description="Trust network propagation analysis for AI agent safety",
    )
    parser.add_argument("--agents", type=int, default=20, help="Number of agents")
    parser.add_argument("--steps", type=int, default=50, help="Simulation steps")
    parser.add_argument("--sybil-test", action="store_true",
                       help="Inject Sybil attackers into simulation")
    parser.add_argument("--attackers", type=int, default=5,
                       help="Number of attackers (with --sybil-test)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--decay", type=float, default=0.005, help="Trust decay rate")
    parser.add_argument("--damping", type=float, default=0.3,
                       help="Propagation damping factor")

    args = parser.parse_args(argv)

    net = TrustNetwork(
        decay_rate=args.decay,
        propagation_damping=args.damping,
        seed=args.seed,
    )
    report = net.simulate(
        num_agents=args.agents,
        num_steps=args.steps,
        num_attackers=args.attackers if args.sybil_test else 0,
    )
    print(_format_report(report, as_json=args.json))


if __name__ == "__main__":
    main()
