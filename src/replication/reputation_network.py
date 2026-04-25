"""ReputationNetwork — peer-to-peer agent reputation system with decay & coalition detection.

Models reputation as a first-class safety primitive in multi-agent systems.
Agents build reputation through endorsements and lose it through violations.
The system detects reputation laundering (colluding agents endorsing each
other), Sybil-like reputation inflation, and identifies agents whose
reputation trajectory warrants blacklisting or probation.

Unlike ``trust_propagation`` (which models trust flow through a network)
or ``collusion_detector`` (which detects coordinated *behavioral* patterns),
the Reputation Network focuses on *social proof dynamics* — how agents
leverage endorsements, vouch for each other, and whether the resulting
reputation scores reflect genuine reliability or manufactured credibility.

Usage (CLI)::

    python -m replication reputation --agents 10 --steps 50
    python -m replication reputation --agents 15 --steps 80 --json
    python -m replication reputation --preset sybil-attack
    python -m replication reputation --preset laundering-ring
    python -m replication reputation --watch --interval 5

Usage (API)::

    from replication.reputation_network import (
        ReputationNetwork, ReputationAgent, Endorsement, Violation,
        ReputationReport,
    )

    net = ReputationNetwork()
    net.add_agent(ReputationAgent("agent-1"))
    net.add_agent(ReputationAgent("agent-2"))
    net.record_endorsement(Endorsement("agent-2", "agent-1", 0.8, "task-success"))
    net.record_violation(Violation("agent-1", "policy_breach", severity=0.6))
    report = net.evaluate()
    print(report.summary())
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from ._helpers import box_header, stats_mean, stats_std


# ── Enums & constants ────────────────────────────────────────────────

class ReputationTier(Enum):
    """Reputation classification tiers."""
    BLACKLISTED = "blacklisted"
    PROBATION = "probation"
    UNTRUSTED = "untrusted"
    NEUTRAL = "neutral"
    TRUSTED = "trusted"
    EXEMPLARY = "exemplary"


TIER_THRESHOLDS: Dict[ReputationTier, Tuple[float, float]] = {
    ReputationTier.BLACKLISTED: (-1.0, -0.5),
    ReputationTier.PROBATION:   (-0.5, -0.1),
    ReputationTier.UNTRUSTED:   (-0.1, 0.2),
    ReputationTier.NEUTRAL:     (0.2, 0.5),
    ReputationTier.TRUSTED:     (0.5, 0.8),
    ReputationTier.EXEMPLARY:   (0.8, 1.0),
}

VIOLATION_CATEGORIES: Dict[str, float] = {
    "policy_breach": 0.3,
    "safety_violation": 0.5,
    "data_exfiltration": 0.8,
    "unauthorized_replication": 0.9,
    "deception": 0.6,
    "resource_abuse": 0.4,
    "collusion": 0.7,
    "evasion": 0.5,
}

DEFAULT_DECAY_RATE = 0.02  # per step
ENDORSEMENT_WEIGHT_CAP = 0.3  # max rep gain per endorsement
VIOLATION_WEIGHT_FLOOR = 0.1  # min rep loss per violation


# ── Data classes ─────────────────────────────────────────────────────

@dataclass
class Endorsement:
    """A reputation endorsement from one agent to another."""
    endorser_id: str
    target_id: str
    weight: float  # 0.0–1.0
    reason: str
    step: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class Violation:
    """A recorded violation by an agent."""
    agent_id: str
    category: str
    severity: float  # 0.0–1.0
    description: str = ""
    step: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ReputationAgent:
    """An agent participating in the reputation network."""
    agent_id: str
    reputation_score: float = 0.5  # start neutral-positive
    endorsements_given: int = 0
    endorsements_received: int = 0
    violations_count: int = 0
    violation_severity_sum: float = 0.0
    tier: ReputationTier = ReputationTier.NEUTRAL
    score_history: List[float] = field(default_factory=list)
    is_blacklisted: bool = False

    def effective_endorsement_weight(self) -> float:
        """Endorsements from low-rep agents carry less weight."""
        return max(0.0, min(1.0, self.reputation_score))


@dataclass
class CoalitionAlert:
    """Alert for detected reputation collusion/coalition."""
    coalition_members: List[str]
    pattern: str  # "mutual_endorsement", "ring", "sybil_cluster"
    confidence: float
    description: str


@dataclass
class BlacklistRecommendation:
    """Recommendation to blacklist an agent."""
    agent_id: str
    current_score: float
    reason: str
    confidence: float
    violations: int
    trend: str  # "declining", "volatile", "consistently_low"


@dataclass
class ReputationReport:
    """Full evaluation report from the reputation network."""
    agents: Dict[str, ReputationAgent]
    coalitions: List[CoalitionAlert]
    blacklist_recommendations: List[BlacklistRecommendation]
    network_health: float  # 0.0–1.0
    gini_coefficient: float
    mean_reputation: float
    findings: List[str]
    proactive_recommendations: List[str]

    def summary(self) -> str:
        lines: List[str] = []
        lines.extend(box_header("Agent Reputation Network Report"))
        lines.append("")
        lines.append(f"  Agents evaluated .... {len(self.agents)}")
        lines.append(f"  Network health ...... {self.network_health:.1%}")
        lines.append(f"  Mean reputation ..... {self.mean_reputation:.3f}")
        lines.append(f"  Gini coefficient .... {self.gini_coefficient:.3f}")
        lines.append(f"  Coalitions found .... {len(self.coalitions)}")
        lines.append(f"  Blacklist recs ...... {len(self.blacklist_recommendations)}")
        lines.append("")

        # Tier distribution
        tier_counts: Dict[str, int] = defaultdict(int)
        for a in self.agents.values():
            tier_counts[a.tier.value] += 1
        lines.append("  Tier Distribution:")
        for tier in ReputationTier:
            count = tier_counts.get(tier.value, 0)
            bar = "\u2588" * count
            lines.append(f"    {tier.value:<14} {count:>3}  {bar}")
        lines.append("")

        # Agent leaderboard (top 5 and bottom 5)
        sorted_agents = sorted(self.agents.values(), key=lambda a: a.reputation_score, reverse=True)
        lines.append("  Top Agents:")
        for a in sorted_agents[:5]:
            flag = " \u26a0" if a.violations_count > 0 else ""
            lines.append(f"    {a.agent_id:<20} {a.reputation_score:>+.3f}  [{a.tier.value}]{flag}")

        if len(sorted_agents) > 5:
            lines.append("  Bottom Agents:")
            for a in sorted_agents[-5:]:
                flag = " \u26a0" if a.violations_count > 0 else ""
                lines.append(f"    {a.agent_id:<20} {a.reputation_score:>+.3f}  [{a.tier.value}]{flag}")
        lines.append("")

        # Coalitions
        if self.coalitions:
            lines.append("  Coalition Alerts:")
            for c in self.coalitions:
                lines.append(f"    [{c.pattern}] confidence={c.confidence:.0%}")
                lines.append(f"      members: {', '.join(c.coalition_members)}")
                lines.append(f"      {c.description}")
            lines.append("")

        # Blacklist recommendations
        if self.blacklist_recommendations:
            lines.append("  Blacklist Recommendations:")
            for b in self.blacklist_recommendations:
                lines.append(f"    {b.agent_id}: score={b.current_score:+.3f} "
                             f"violations={b.violations} trend={b.trend}")
                lines.append(f"      reason: {b.reason}")
            lines.append("")

        # Findings
        if self.findings:
            lines.append("  Findings:")
            for f in self.findings:
                lines.append(f"    \u2022 {f}")
            lines.append("")

        # Proactive recommendations
        if self.proactive_recommendations:
            lines.append("  Proactive Recommendations:")
            for r in self.proactive_recommendations:
                lines.append(f"    \u27a4 {r}")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "network_health": round(self.network_health, 4),
            "gini_coefficient": round(self.gini_coefficient, 4),
            "mean_reputation": round(self.mean_reputation, 4),
            "agent_count": len(self.agents),
            "agents": {
                aid: {
                    "reputation_score": round(a.reputation_score, 4),
                    "tier": a.tier.value,
                    "endorsements_given": a.endorsements_given,
                    "endorsements_received": a.endorsements_received,
                    "violations_count": a.violations_count,
                    "is_blacklisted": a.is_blacklisted,
                }
                for aid, a in self.agents.items()
            },
            "coalitions": [
                {
                    "members": c.coalition_members,
                    "pattern": c.pattern,
                    "confidence": round(c.confidence, 4),
                    "description": c.description,
                }
                for c in self.coalitions
            ],
            "blacklist_recommendations": [
                {
                    "agent_id": b.agent_id,
                    "current_score": round(b.current_score, 4),
                    "reason": b.reason,
                    "confidence": round(b.confidence, 4),
                    "trend": b.trend,
                }
                for b in self.blacklist_recommendations
            ],
            "findings": self.findings,
            "proactive_recommendations": self.proactive_recommendations,
        }


# ── Core engine ──────────────────────────────────────────────────────

class ReputationNetwork:
    """Peer-to-peer agent reputation network with coalition detection."""

    def __init__(
        self,
        decay_rate: float = DEFAULT_DECAY_RATE,
        endorsement_cap: float = ENDORSEMENT_WEIGHT_CAP,
    ) -> None:
        self.agents: Dict[str, ReputationAgent] = {}
        self.endorsements: List[Endorsement] = []
        self.violations: List[Violation] = []
        self.decay_rate = decay_rate
        self.endorsement_cap = endorsement_cap
        self._endorsement_graph: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def add_agent(self, agent: ReputationAgent) -> None:
        self.agents[agent.agent_id] = agent

    def record_endorsement(self, e: Endorsement) -> None:
        if e.endorser_id not in self.agents or e.target_id not in self.agents:
            return
        self.endorsements.append(e)
        endorser = self.agents[e.endorser_id]
        target = self.agents[e.target_id]

        # Weight by endorser's own reputation
        effective = min(
            e.weight * endorser.effective_endorsement_weight(),
            self.endorsement_cap,
        )
        target.reputation_score = min(1.0, target.reputation_score + effective)
        target.endorsements_received += 1
        endorser.endorsements_given += 1
        self._endorsement_graph[e.endorser_id][e.target_id] += 1

    def record_violation(self, v: Violation) -> None:
        if v.agent_id not in self.agents:
            return
        self.violations.append(v)
        agent = self.agents[v.agent_id]
        base = VIOLATION_CATEGORIES.get(v.category, 0.3)
        penalty = max(VIOLATION_WEIGHT_FLOOR, base * v.severity)
        agent.reputation_score = max(-1.0, agent.reputation_score - penalty)
        agent.violations_count += 1
        agent.violation_severity_sum += v.severity

    def apply_decay(self) -> None:
        """Apply time-based reputation decay toward neutral."""
        for agent in self.agents.values():
            if agent.reputation_score > 0.5:
                agent.reputation_score -= self.decay_rate
                agent.reputation_score = max(0.5, agent.reputation_score)
            elif agent.reputation_score < 0.5:
                agent.reputation_score += self.decay_rate * 0.5  # slower recovery
                agent.reputation_score = min(0.5, agent.reputation_score)

    def step_snapshot(self) -> None:
        """Record current scores in history."""
        for agent in self.agents.values():
            agent.score_history.append(agent.reputation_score)

    def _classify_tiers(self) -> None:
        for agent in self.agents.values():
            s = agent.reputation_score
            agent.tier = ReputationTier.NEUTRAL
            for tier, (lo, hi) in TIER_THRESHOLDS.items():
                if lo <= s < hi or (tier == ReputationTier.EXEMPLARY and s >= hi):
                    agent.tier = tier
                    break

    def _detect_coalitions(self) -> List[CoalitionAlert]:
        alerts: List[CoalitionAlert] = []
        graph = self._endorsement_graph

        # Detect mutual endorsement pairs
        seen_pairs: Set[Tuple[str, str]] = set()
        for a, targets in graph.items():
            for b, count_ab in targets.items():
                if b in graph and a in graph[b]:
                    pair = tuple(sorted([a, b]))
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        count_ba = graph[b][a]
                        if count_ab >= 2 and count_ba >= 2:
                            confidence = min(1.0, (count_ab + count_ba) / 10.0)
                            alerts.append(CoalitionAlert(
                                coalition_members=[a, b],
                                pattern="mutual_endorsement",
                                confidence=confidence,
                                description=f"Agents {a} and {b} have exchanged "
                                            f"{count_ab}+{count_ba} endorsements — "
                                            f"possible reputation laundering.",
                            ))

        # Detect endorsement rings (A→B→C→A with ≥3 members)
        seen_rings: Set[Tuple[str, ...]] = set()
        for a in graph:
            for b in graph.get(a, {}):
                for c in graph.get(b, {}):
                    if c != a and c != b and a in graph.get(c, {}):
                        ring_key = tuple(sorted([a, b, c]))
                        if ring_key not in seen_rings:
                            seen_rings.add(ring_key)
                            alerts.append(CoalitionAlert(
                                coalition_members=list(ring_key),
                                pattern="ring",
                                confidence=0.7,
                                description=f"Circular endorsement ring detected: "
                                            f"{a} → {b} → {c} → {a}.",
                            ))

        # Detect Sybil clusters: agents that only endorse each other
        seen_clusters: Set[Tuple[str, ...]] = set()
        for a in graph:
            targets_of_a = set(graph[a].keys())
            if len(targets_of_a) < 2:
                continue
            # Check if the targets form a clique
            clique = {a} | targets_of_a
            if len(clique) >= 3:
                internal_edges = 0
                total_possible = len(clique) * (len(clique) - 1)
                for m in clique:
                    for n in clique:
                        if m != n and n in graph.get(m, {}):
                            internal_edges += 1
                density = internal_edges / total_possible if total_possible > 0 else 0
                if density > 0.6:
                    cluster_key = tuple(sorted(clique))
                    if cluster_key not in seen_clusters:
                        seen_clusters.add(cluster_key)
                        alerts.append(CoalitionAlert(
                            coalition_members=sorted(clique),
                            pattern="sybil_cluster",
                            confidence=min(1.0, density),
                            description=f"Dense endorsement cluster ({density:.0%} density) "
                                        f"— possible Sybil reputation inflation.",
                        ))

        return alerts

    def _blacklist_recommendations(self) -> List[BlacklistRecommendation]:
        recs: List[BlacklistRecommendation] = []
        for agent in self.agents.values():
            if agent.reputation_score >= 0.0:
                continue
            # Determine trend
            history = agent.score_history
            if len(history) >= 3:
                recent = history[-3:]
                if all(recent[i] >= recent[i + 1] for i in range(len(recent) - 1)):
                    trend = "declining"
                elif stats_std(recent) > 0.15:
                    trend = "volatile"
                else:
                    trend = "consistently_low"
            else:
                trend = "consistently_low"

            reasons = []
            if agent.violations_count >= 3:
                reasons.append(f"{agent.violations_count} violations recorded")
            if agent.reputation_score < -0.5:
                reasons.append("reputation severely degraded")
            if trend == "declining":
                reasons.append("score is actively declining")

            if reasons:
                recs.append(BlacklistRecommendation(
                    agent_id=agent.agent_id,
                    current_score=agent.reputation_score,
                    reason="; ".join(reasons),
                    confidence=min(1.0, len(reasons) / 3.0),
                    violations=agent.violations_count,
                    trend=trend,
                ))
        return recs

    def _compute_gini(self) -> float:
        scores = sorted(a.reputation_score for a in self.agents.values())
        n = len(scores)
        if n == 0:
            return 0.0
        # Shift to non-negative for Gini
        min_s = min(scores)
        shifted = [s - min_s for s in scores]
        total = sum(shifted)
        if total == 0:
            return 0.0
        cumulative = 0.0
        weighted_sum = 0.0
        for i, s in enumerate(shifted):
            cumulative += s
            weighted_sum += (2 * (i + 1) - n - 1) * s
        return weighted_sum / (n * total)

    def evaluate(self) -> ReputationReport:
        # Single pass: classify tiers, collect scores, count blacklisted,
        # track endorsement givers/receivers, and find the top giver — all
        # in one iteration instead of 7+ separate passes over agents.
        n = len(self.agents)
        scores: List[float] = []
        blacklisted_count = 0
        givers_sum = 0.0
        max_given = 0
        top_giver_agent: Optional[ReputationAgent] = None

        tier_items = list(TIER_THRESHOLDS.items())
        for agent in self.agents.values():
            s = agent.reputation_score
            # Classify tier inline (replaces _classify_tiers)
            agent.tier = ReputationTier.NEUTRAL
            for tier, (lo, hi) in tier_items:
                if lo <= s < hi or (tier == ReputationTier.EXEMPLARY and s >= hi):
                    agent.tier = tier
                    break

            scores.append(s)
            if s < -0.5:
                blacklisted_count += 1
            givers_sum += agent.endorsements_given
            if agent.endorsements_given > max_given:
                max_given = agent.endorsements_given
                top_giver_agent = agent

        coalitions = self._detect_coalitions()
        blacklist_recs = self._blacklist_recommendations()

        mean_rep = stats_mean(scores) if scores else 0.5
        gini = self._compute_gini()

        # Network health: penalize for blacklisted agents, coalitions, low mean
        blacklisted_frac = blacklisted_count / max(n, 1)
        coalition_penalty = min(0.3, len(coalitions) * 0.05)
        health = max(0.0, min(1.0, 0.5 + mean_rep * 0.3 - blacklisted_frac * 0.3 - coalition_penalty))

        findings: List[str] = []
        recs: List[str] = []

        if gini > 0.4:
            findings.append(f"High reputation inequality (Gini={gini:.2f}) — a few agents dominate trust.")
            recs.append("Investigate whether top-reputation agents deserve their standing or are gaming the system.")

        if coalitions:
            findings.append(f"{len(coalitions)} endorsement coalition(s) detected — reputation may be manufactured.")
            recs.append("Discount endorsements within detected coalitions and require third-party vouching.")

        if blacklist_recs:
            findings.append(f"{len(blacklist_recs)} agent(s) recommended for blacklisting.")
            recs.append("Review blacklist candidates and revoke their endorsement privileges immediately.")

        # Check for endorsement asymmetry (uses pre-computed max_given/givers_sum)
        givers_mean = (givers_sum / n) if n > 0 else 0.0
        if n > 0 and max_given > 3 * (givers_mean + 1) and top_giver_agent is not None:
            findings.append(f"Agent {top_giver_agent.agent_id} is an endorsement outlier "
                            f"({top_giver_agent.endorsements_given} given) — possible rubber-stamping.")
            recs.append(f"Rate-limit endorsements from {top_giver_agent.agent_id} or require justification.")

        if not findings:
            findings.append("No significant reputation anomalies detected.")
            recs.append("Continue monitoring — reputation dynamics can shift rapidly.")

        return ReputationReport(
            agents=dict(self.agents),
            coalitions=coalitions,
            blacklist_recommendations=blacklist_recs,
            network_health=health,
            gini_coefficient=gini,
            mean_reputation=mean_rep,
            findings=findings,
            proactive_recommendations=recs,
        )


# ── Simulation presets ───────────────────────────────────────────────

PRESETS: Dict[str, Dict[str, Any]] = {
    "healthy-network": {
        "description": "Well-functioning network with organic endorsements",
        "agents": 12,
        "steps": 40,
        "violation_rate": 0.05,
        "coalition_agents": 0,
    },
    "sybil-attack": {
        "description": "Network under Sybil attack — cluster of fake agents boosting each other",
        "agents": 15,
        "steps": 50,
        "violation_rate": 0.03,
        "coalition_agents": 5,
        "coalition_pattern": "sybil",
    },
    "laundering-ring": {
        "description": "Small ring of agents laundering reputation through circular endorsements",
        "agents": 10,
        "steps": 40,
        "violation_rate": 0.1,
        "coalition_agents": 3,
        "coalition_pattern": "ring",
    },
    "hostile-environment": {
        "description": "High violation rate with aggressive reputation decay",
        "agents": 12,
        "steps": 60,
        "violation_rate": 0.25,
        "coalition_agents": 0,
    },
}


# ── Simulator ────────────────────────────────────────────────────────

def _simulate(
    num_agents: int = 10,
    num_steps: int = 40,
    violation_rate: float = 0.1,
    coalition_agents: int = 0,
    coalition_pattern: str = "sybil",
    seed: Optional[int] = None,
) -> ReputationNetwork:
    """Run a simulated reputation network scenario."""
    rng = random.Random(seed)
    net = ReputationNetwork()

    agent_ids = [f"agent-{i:02d}" for i in range(num_agents)]
    for aid in agent_ids:
        net.add_agent(ReputationAgent(aid))

    # Identify coalition members
    coalition_members = set(agent_ids[:coalition_agents]) if coalition_agents > 0 else set()

    violation_cats = list(VIOLATION_CATEGORIES.keys())

    for step in range(num_steps):
        # Organic endorsements
        for _ in range(rng.randint(1, max(2, num_agents // 2))):
            endorser = rng.choice(agent_ids)
            target = rng.choice([a for a in agent_ids if a != endorser])
            net.record_endorsement(Endorsement(
                endorser_id=endorser,
                target_id=target,
                weight=rng.uniform(0.1, 0.5),
                reason="task-completion",
                step=step,
            ))

        # Coalition endorsements (boosting each other)
        if coalition_members:
            members = list(coalition_members)
            for m in members:
                target = rng.choice([x for x in members if x != m])
                net.record_endorsement(Endorsement(
                    endorser_id=m,
                    target_id=target,
                    weight=rng.uniform(0.5, 0.9),
                    reason="coalition-boost",
                    step=step,
                ))

        # Violations
        for aid in agent_ids:
            if rng.random() < violation_rate:
                cat = rng.choice(violation_cats)
                net.record_violation(Violation(
                    agent_id=aid,
                    category=cat,
                    severity=rng.uniform(0.2, 1.0),
                    description=f"Simulated {cat}",
                    step=step,
                ))

        net.apply_decay()
        net.step_snapshot()

    return net


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point for the Agent Reputation Network."""
    parser = argparse.ArgumentParser(
        prog="python -m replication reputation",
        description="Agent Reputation Network — peer-to-peer reputation with coalition detection",
    )
    parser.add_argument("--agents", type=int, default=10, help="Number of agents (default: 10)")
    parser.add_argument("--steps", type=int, default=40, help="Simulation steps (default: 40)")
    parser.add_argument("--violation-rate", type=float, default=0.1, help="Violation probability per agent per step")
    parser.add_argument("--coalition", type=int, default=0, help="Number of colluding agents (0=none)")
    parser.add_argument("--coalition-pattern", choices=["sybil", "ring"], default="sybil")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), help="Use a preset scenario")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    parser.add_argument("--watch", action="store_true", help="Continuous monitoring mode")
    parser.add_argument("--interval", type=int, default=10, help="Watch interval in seconds")

    args = parser.parse_args(argv)

    if args.preset:
        p = PRESETS[args.preset]
        print(f"Using preset: {args.preset} — {p['description']}\n")
        num_agents = p["agents"]
        num_steps = p["steps"]
        violation_rate = p["violation_rate"]
        coalition_agents = p["coalition_agents"]
        coalition_pattern = p.get("coalition_pattern", "sybil")
    else:
        num_agents = args.agents
        num_steps = args.steps
        violation_rate = args.violation_rate
        coalition_agents = args.coalition
        coalition_pattern = args.coalition_pattern

    if args.watch:
        _run_watch(num_agents, violation_rate, coalition_agents, coalition_pattern,
                   args.interval, args.seed, args.json)
        return

    net = _simulate(
        num_agents=num_agents,
        num_steps=num_steps,
        violation_rate=violation_rate,
        coalition_agents=coalition_agents,
        coalition_pattern=coalition_pattern,
        seed=args.seed,
    )
    report = net.evaluate()

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary())


def _run_watch(
    num_agents: int,
    violation_rate: float,
    coalition_agents: int,
    coalition_pattern: str,
    interval: int,
    seed: Optional[int],
    as_json: bool,
) -> None:
    """Continuous watch mode — re-evaluate every interval."""
    net = ReputationNetwork()
    agent_ids = [f"agent-{i:02d}" for i in range(num_agents)]
    for aid in agent_ids:
        net.add_agent(ReputationAgent(aid))

    coalition_members = set(agent_ids[:coalition_agents]) if coalition_agents > 0 else set()
    rng = random.Random(seed)
    violation_cats = list(VIOLATION_CATEGORIES.keys())
    step = 0

    try:
        while True:
            # Simulate a batch of steps
            for _ in range(5):
                for _ in range(rng.randint(1, max(2, num_agents // 2))):
                    endorser = rng.choice(agent_ids)
                    target = rng.choice([a for a in agent_ids if a != endorser])
                    net.record_endorsement(Endorsement(endorser, target, rng.uniform(0.1, 0.5), "task", step))

                if coalition_members:
                    for m in coalition_members:
                        t = rng.choice([x for x in coalition_members if x != m])
                        net.record_endorsement(Endorsement(m, t, rng.uniform(0.5, 0.9), "coalition", step))

                for aid in agent_ids:
                    if rng.random() < violation_rate:
                        net.record_violation(Violation(aid, rng.choice(violation_cats), rng.uniform(0.2, 1.0), step=step))

                net.apply_decay()
                net.step_snapshot()
                step += 1

            report = net.evaluate()
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            if as_json:
                print(json.dumps({"timestamp": ts, "step": step, **report.to_dict()}))
            else:
                print(f"\n{'='*57}")
                print(f"  Watch — step {step} @ {ts}")
                print(f"{'='*57}")
                print(report.summary())

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nWatch stopped.")


if __name__ == "__main__":
    main()
