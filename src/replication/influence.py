"""Agent Influence Mapping — track and analyze inter-agent influence patterns.

In multi-agent replication systems, agents don't operate in isolation.
They communicate, share state, and subtly (or overtly) influence each
other's behavior.  Left unchecked, this can lead to dangerous emergent
phenomena: information cascades, opinion convergence toward unsafe goals,
coordinated deception, or covert coalition formation.

The influence mapper monitors agent interactions and builds a directed
influence graph, scoring edges by strength and type.  It detects:

- **Information cascades:** rapid propagation of beliefs/behaviors
- **Opinion convergence:** agents aligning on values without explicit
  coordination (emergent consensus)
- **Coordinated behavior:** synchronized actions suggesting coalition
- **Influence monopolies:** single agents dominating the group's behavior
- **Echo chambers:** clusters of agents reinforcing each other's beliefs

Usage (CLI)::

    python -m replication.influence                        # analyze default logs
    python -m replication.influence --window 3600          # 1-hour analysis window
    python -m replication.influence --threshold 0.5        # min influence score
    python -m replication.influence --detect cascades      # detect cascades only
    python -m replication.influence --detect convergence   # detect convergence
    python -m replication.influence --detect coalitions    # detect coalitions
    python -m replication.influence --json                 # JSON output
    python -m replication.influence --top 10               # top 10 influence edges

Programmatic::

    from replication.influence import InfluenceMapper, InfluenceConfig
    mapper = InfluenceMapper(InfluenceConfig(window_seconds=3600))
    mapper.record_interaction("agent-a", "agent-b", "state_share", {"key": "goal", "value": "explore"})
    mapper.record_interaction("agent-b", "agent-c", "state_share", {"key": "goal", "value": "explore"})
    report = mapper.analyze()
    print(report.render())
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class InteractionType(Enum):
    """Types of inter-agent interactions."""
    STATE_SHARE = "state_share"
    MESSAGE = "message"
    REPLICATION = "replication"
    RESOURCE_TRANSFER = "resource_transfer"
    GOAL_ALIGNMENT = "goal_alignment"
    BEHAVIOR_COPY = "behavior_copy"


@dataclass(frozen=True)
class Interaction:
    """A single observed interaction between two agents."""
    source: str
    target: str
    interaction_type: InteractionType
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InfluenceEdge:
    """Weighted directed edge in the influence graph."""
    source: str
    target: str
    weight: float
    interaction_count: int
    interaction_types: Set[str] = field(default_factory=set)
    first_seen: float = 0.0
    last_seen: float = 0.0

    @property
    def duration(self) -> float:
        return self.last_seen - self.first_seen


@dataclass
class CascadeEvent:
    """A detected information cascade."""
    origin: str
    propagation_chain: List[str]
    key: str
    value: Any
    speed: float  # agents per second
    reach: int  # total agents affected
    timestamp: float = 0.0

    @property
    def depth(self) -> int:
        return len(self.propagation_chain)


@dataclass
class ConvergenceEvent:
    """Detected opinion/behavior convergence."""
    agents: List[str]
    attribute: str
    converged_value: Any
    convergence_rate: float  # 0-1, how quickly they aligned
    timestamp: float = 0.0


@dataclass
class CoalitionEvent:
    """Detected coordinated agent coalition."""
    members: FrozenSet[str]
    coordination_score: float  # 0-1
    shared_behaviors: List[str]
    formation_time: float = 0.0
    detected_at: float = 0.0


@dataclass
class InfluenceMonopoly:
    """An agent with outsized influence over the group."""
    agent: str
    influence_score: float
    influenced_agents: List[str]
    dominance_ratio: float  # fraction of total influence


@dataclass
class EchoChamber:
    """A cluster of mutually reinforcing agents."""
    members: FrozenSet[str]
    internal_density: float
    external_density: float
    isolation_ratio: float  # internal/external density ratio


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class InfluenceConfig:
    """Configuration for influence analysis."""
    window_seconds: float = 3600.0
    min_influence_weight: float = 0.1
    cascade_speed_threshold: float = 0.5  # agents/sec to count as cascade
    convergence_threshold: float = 0.8  # similarity threshold
    coalition_sync_window: float = 5.0  # seconds for synchronized actions
    coalition_min_size: int = 3
    monopoly_threshold: float = 0.5  # fraction of total influence
    echo_chamber_ratio: float = 3.0  # internal/external density


# ---------------------------------------------------------------------------
# Influence Mapper
# ---------------------------------------------------------------------------

class InfluenceMapper:
    """Tracks interactions and builds influence analysis."""

    def __init__(self, config: Optional[InfluenceConfig] = None) -> None:
        self.config = config or InfluenceConfig()
        self._interactions: List[Interaction] = []
        self._agent_states: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._state_history: Dict[str, List[Tuple[float, str, Any]]] = defaultdict(list)

    # -- Recording ----------------------------------------------------------

    def record_interaction(
        self,
        source: str,
        target: str,
        interaction_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> Interaction:
        """Record an observed interaction between agents."""
        ts = timestamp if timestamp is not None else datetime.now(timezone.utc).timestamp()
        itype = InteractionType(interaction_type)
        interaction = Interaction(
            source=source,
            target=target,
            interaction_type=itype,
            timestamp=ts,
            metadata=metadata or {},
        )
        self._interactions.append(interaction)

        # Track state changes for cascade/convergence detection
        if metadata and "key" in metadata and "value" in metadata:
            self._agent_states[target][metadata["key"]] = metadata["value"]
            self._state_history[target].append(
                (ts, metadata["key"], metadata["value"])
            )

        return interaction

    def record_state(self, agent: str, key: str, value: Any,
                     timestamp: Optional[float] = None) -> None:
        """Record an agent's state without an interaction."""
        ts = timestamp if timestamp is not None else datetime.now(timezone.utc).timestamp()
        self._agent_states[agent][key] = value
        self._state_history[agent].append((ts, key, value))

    # -- Analysis -----------------------------------------------------------

    def _windowed_interactions(self) -> List[Interaction]:
        """Get interactions within the analysis window."""
        if not self._interactions:
            return []
        latest = max(i.timestamp for i in self._interactions)
        cutoff = latest - self.config.window_seconds
        return [i for i in self._interactions if i.timestamp >= cutoff]

    def build_influence_graph(self) -> Dict[Tuple[str, str], InfluenceEdge]:
        """Build weighted directed influence graph from interactions."""
        interactions = self._windowed_interactions()
        edges: Dict[Tuple[str, str], InfluenceEdge] = {}

        for ix in interactions:
            key = (ix.source, ix.target)
            if key not in edges:
                edges[key] = InfluenceEdge(
                    source=ix.source,
                    target=ix.target,
                    weight=0.0,
                    interaction_count=0,
                    first_seen=ix.timestamp,
                    last_seen=ix.timestamp,
                )
            edge = edges[key]
            edge.interaction_count += 1
            edge.interaction_types.add(ix.interaction_type.value)
            edge.first_seen = min(edge.first_seen, ix.timestamp)
            edge.last_seen = max(edge.last_seen, ix.timestamp)

            # Weight by type importance
            type_weights = {
                InteractionType.BEHAVIOR_COPY: 1.0,
                InteractionType.GOAL_ALIGNMENT: 0.9,
                InteractionType.REPLICATION: 0.8,
                InteractionType.STATE_SHARE: 0.5,
                InteractionType.RESOURCE_TRANSFER: 0.4,
                InteractionType.MESSAGE: 0.3,
            }
            edge.weight += type_weights.get(ix.interaction_type, 0.3)

        # Normalize weights
        if edges:
            max_w = max(e.weight for e in edges.values())
            if max_w > 0:
                for e in edges.values():
                    e.weight = e.weight / max_w

        return edges

    def detect_cascades(self) -> List[CascadeEvent]:
        """Detect information cascades — rapid belief propagation."""
        cascades: List[CascadeEvent] = []
        interactions = self._windowed_interactions()

        # Group state_share interactions by key+value
        propagations: Dict[Tuple[str, str], List[Interaction]] = defaultdict(list)
        for ix in interactions:
            if ix.interaction_type == InteractionType.STATE_SHARE:
                k = ix.metadata.get("key", "")
                v = str(ix.metadata.get("value", ""))
                if k:
                    propagations[(k, v)].append(ix)

        for (key, value), ixs in propagations.items():
            if len(ixs) < 2:
                continue
            sorted_ixs = sorted(ixs, key=lambda x: x.timestamp)
            chain = [sorted_ixs[0].source]
            seen = {sorted_ixs[0].source}
            for ix in sorted_ixs:
                if ix.target not in seen:
                    chain.append(ix.target)
                    seen.add(ix.target)

            if len(chain) < 3:
                continue

            duration = sorted_ixs[-1].timestamp - sorted_ixs[0].timestamp
            speed = (len(chain) - 1) / max(duration, 0.001)

            if speed >= self.config.cascade_speed_threshold:
                cascades.append(CascadeEvent(
                    origin=chain[0],
                    propagation_chain=chain,
                    key=key,
                    value=value,
                    speed=round(speed, 3),
                    reach=len(chain),
                    timestamp=sorted_ixs[0].timestamp,
                ))

        return sorted(cascades, key=lambda c: c.speed, reverse=True)

    def detect_convergence(self) -> List[ConvergenceEvent]:
        """Detect opinion/behavior convergence across agents."""
        events: List[ConvergenceEvent] = []

        # Group agents by attribute values
        attr_agents: Dict[str, Dict[Any, List[str]]] = defaultdict(lambda: defaultdict(list))
        for agent, state in self._agent_states.items():
            for key, value in state.items():
                attr_agents[key][str(value)].append(agent)

        all_agents = set(self._agent_states.keys())
        if len(all_agents) < 2:
            return events

        for attr, value_groups in attr_agents.items():
            for value, agents in value_groups.items():
                ratio = len(agents) / len(all_agents)
                if ratio >= self.config.convergence_threshold:
                    # Calculate convergence rate from history
                    rate = self._calc_convergence_rate(agents, attr, value)
                    events.append(ConvergenceEvent(
                        agents=sorted(agents),
                        attribute=attr,
                        converged_value=value,
                        convergence_rate=round(rate, 3),
                    ))

        return sorted(events, key=lambda e: e.convergence_rate, reverse=True)

    def _calc_convergence_rate(self, agents: List[str], attr: str, value: str) -> float:
        """How quickly did agents converge on this value? 0=slow, 1=instant."""
        timestamps = []
        for agent in agents:
            for ts, k, v in self._state_history.get(agent, []):
                if k == attr and str(v) == value:
                    timestamps.append(ts)
                    break
        if len(timestamps) < 2:
            return 1.0
        span = max(timestamps) - min(timestamps)
        # Fast convergence = high rate. Normalize against window.
        if span == 0:
            return 1.0
        return max(0.0, 1.0 - (span / self.config.window_seconds))

    def detect_coalitions(self) -> List[CoalitionEvent]:
        """Detect coordinated behavior (synchronized actions)."""
        coalitions: List[CoalitionEvent] = []
        interactions = self._windowed_interactions()
        if not interactions:
            return coalitions

        # Find agents performing same action types within sync window
        time_buckets: Dict[int, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        for ix in interactions:
            bucket = int(ix.timestamp / self.config.coalition_sync_window)
            time_buckets[bucket][ix.interaction_type.value].add(ix.source)

        # Find persistent co-occurrence across multiple buckets
        pair_counts: Dict[FrozenSet[str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for bucket_data in time_buckets.values():
            for itype, agents in bucket_data.items():
                if len(agents) >= self.config.coalition_min_size:
                    group = frozenset(agents)
                    pair_counts[group][itype] += 1

        for members, type_counts in pair_counts.items():
            if len(members) < self.config.coalition_min_size:
                continue
            total = sum(type_counts.values())
            if total < 3:  # need at least 3 synchronized events
                continue
            n_buckets = len(time_buckets) or 1
            score = min(1.0, total / n_buckets)
            coalitions.append(CoalitionEvent(
                members=members,
                coordination_score=round(score, 3),
                shared_behaviors=list(type_counts.keys()),
            ))

        return sorted(coalitions, key=lambda c: c.coordination_score, reverse=True)

    def detect_monopolies(self) -> List[InfluenceMonopoly]:
        """Find agents with outsized influence over the group."""
        graph = self.build_influence_graph()
        if not graph:
            return []

        # Calculate outgoing influence per agent
        out_influence: Dict[str, float] = defaultdict(float)
        out_targets: Dict[str, List[str]] = defaultdict(list)
        total_influence = 0.0

        for (src, tgt), edge in graph.items():
            out_influence[src] += edge.weight
            out_targets[src].append(tgt)
            total_influence += edge.weight

        if total_influence == 0:
            return []

        monopolies = []
        for agent, score in out_influence.items():
            ratio = score / total_influence
            if ratio >= self.config.monopoly_threshold:
                monopolies.append(InfluenceMonopoly(
                    agent=agent,
                    influence_score=round(score, 3),
                    influenced_agents=sorted(out_targets[agent]),
                    dominance_ratio=round(ratio, 3),
                ))

        return sorted(monopolies, key=lambda m: m.dominance_ratio, reverse=True)

    def detect_echo_chambers(self) -> List[EchoChamber]:
        """Find clusters of mutually reinforcing agents."""
        graph = self.build_influence_graph()
        if not graph:
            return []

        # Build adjacency for mutual connections
        all_agents: Set[str] = set()
        for src, tgt in graph:
            all_agents.add(src)
            all_agents.add(tgt)

        if len(all_agents) < 3:
            return []

        # Find strongly connected clusters via mutual edges
        mutual: Dict[str, Set[str]] = defaultdict(set)
        for (src, tgt) in graph:
            if (tgt, src) in graph:
                mutual[src].add(tgt)
                mutual[tgt].add(src)

        # Greedy cluster extraction
        visited: Set[str] = set()
        chambers: List[EchoChamber] = []

        for agent in sorted(all_agents):
            if agent in visited or agent not in mutual:
                continue
            cluster = {agent}
            queue = [agent]
            while queue:
                current = queue.pop(0)
                for neighbor in mutual.get(current, set()):
                    if neighbor not in cluster:
                        cluster.add(neighbor)
                        queue.append(neighbor)
            visited.update(cluster)

            if len(cluster) < 3:
                continue

            # Calculate internal vs external density
            members = frozenset(cluster)
            internal = 0
            external = 0
            for (src, tgt), edge in graph.items():
                if src in members and tgt in members:
                    internal += edge.weight
                elif src in members or tgt in members:
                    external += edge.weight

            n = len(members)
            max_internal = n * (n - 1)
            int_density = internal / max_internal if max_internal > 0 else 0
            ext_density = external / (n * (len(all_agents) - n)) if (len(all_agents) - n) > 0 else 0

            ratio = int_density / ext_density if ext_density > 0 else float("inf")
            if ratio >= self.config.echo_chamber_ratio:
                chambers.append(EchoChamber(
                    members=members,
                    internal_density=round(int_density, 3),
                    external_density=round(ext_density, 3),
                    isolation_ratio=round(ratio, 3) if ratio != float("inf") else 999.0,
                ))

        return sorted(chambers, key=lambda c: c.isolation_ratio, reverse=True)

    def analyze(self) -> "InfluenceReport":
        """Run full influence analysis."""
        return InfluenceReport(
            graph=self.build_influence_graph(),
            cascades=self.detect_cascades(),
            convergences=self.detect_convergence(),
            coalitions=self.detect_coalitions(),
            monopolies=self.detect_monopolies(),
            echo_chambers=self.detect_echo_chambers(),
            total_interactions=len(self._windowed_interactions()),
            agents=sorted(set(
                a for ix in self._interactions
                for a in (ix.source, ix.target)
            )),
            config=self.config,
        )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class InfluenceReport:
    """Complete influence analysis report."""
    graph: Dict[Tuple[str, str], InfluenceEdge]
    cascades: List[CascadeEvent]
    convergences: List[ConvergenceEvent]
    coalitions: List[CoalitionEvent]
    monopolies: List[InfluenceMonopoly]
    echo_chambers: List[EchoChamber]
    total_interactions: int
    agents: List[str]
    config: InfluenceConfig

    @property
    def risk_score(self) -> float:
        """Overall influence risk score 0-1."""
        scores = []
        if self.cascades:
            scores.append(min(1.0, len(self.cascades) * 0.3))
        if self.convergences:
            scores.append(min(1.0, max(c.convergence_rate for c in self.convergences)))
        if self.coalitions:
            scores.append(min(1.0, max(c.coordination_score for c in self.coalitions)))
        if self.monopolies:
            scores.append(min(1.0, max(m.dominance_ratio for m in self.monopolies)))
        if self.echo_chambers:
            scores.append(min(1.0, 0.5))
        return round(max(scores) if scores else 0.0, 3)

    @property
    def risk_level(self) -> str:
        s = self.risk_score
        if s >= 0.8:
            return "CRITICAL"
        if s >= 0.6:
            return "HIGH"
        if s >= 0.4:
            return "MEDIUM"
        if s >= 0.2:
            return "LOW"
        return "MINIMAL"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "total_interactions": self.total_interactions,
            "agents": self.agents,
            "influence_edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "weight": e.weight,
                    "interaction_count": e.interaction_count,
                    "types": sorted(e.interaction_types),
                }
                for e in sorted(self.graph.values(), key=lambda x: x.weight, reverse=True)
            ],
            "cascades": [
                {
                    "origin": c.origin,
                    "chain": c.propagation_chain,
                    "key": c.key,
                    "speed": c.speed,
                    "reach": c.reach,
                }
                for c in self.cascades
            ],
            "convergences": [
                {
                    "agents": c.agents,
                    "attribute": c.attribute,
                    "value": c.converged_value,
                    "rate": c.convergence_rate,
                }
                for c in self.convergences
            ],
            "coalitions": [
                {
                    "members": sorted(c.members),
                    "score": c.coordination_score,
                    "behaviors": c.shared_behaviors,
                }
                for c in self.coalitions
            ],
            "monopolies": [
                {
                    "agent": m.agent,
                    "influence": m.influence_score,
                    "targets": m.influenced_agents,
                    "dominance": m.dominance_ratio,
                }
                for m in self.monopolies
            ],
            "echo_chambers": [
                {
                    "members": sorted(c.members),
                    "isolation_ratio": c.isolation_ratio,
                }
                for c in self.echo_chambers
            ],
        }

    def render(self) -> str:
        lines = [
            "=" * 60,
            "AGENT INFLUENCE ANALYSIS REPORT",
            "=" * 60,
            f"Agents: {len(self.agents)} | Interactions: {self.total_interactions}",
            f"Risk Score: {self.risk_score} ({self.risk_level})",
            "",
        ]

        # Top influence edges
        top_edges = sorted(self.graph.values(), key=lambda e: e.weight, reverse=True)[:10]
        if top_edges:
            lines.append("--- Top Influence Edges ---")
            for e in top_edges:
                types = ",".join(sorted(e.interaction_types))
                lines.append(
                    f"  {e.source} -> {e.target}  "
                    f"weight={e.weight:.3f}  count={e.interaction_count}  [{types}]"
                )
            lines.append("")

        # Cascades
        if self.cascades:
            lines.append(f"--- Information Cascades ({len(self.cascades)}) ---")
            for c in self.cascades:
                chain_str = " -> ".join(c.propagation_chain[:5])
                if len(c.propagation_chain) > 5:
                    chain_str += f" ... (+{len(c.propagation_chain) - 5} more)"
                lines.append(
                    f"  [{c.key}={c.value}] {chain_str}  "
                    f"speed={c.speed} agents/s  reach={c.reach}"
                )
            lines.append("")

        # Convergence
        if self.convergences:
            lines.append(f"--- Opinion Convergence ({len(self.convergences)}) ---")
            for c in self.convergences:
                agents_str = ", ".join(c.agents[:5])
                if len(c.agents) > 5:
                    agents_str += f" +{len(c.agents) - 5}"
                lines.append(
                    f"  {c.attribute}={c.converged_value}  "
                    f"agents=[{agents_str}]  rate={c.convergence_rate}"
                )
            lines.append("")

        # Coalitions
        if self.coalitions:
            lines.append(f"--- Detected Coalitions ({len(self.coalitions)}) ---")
            for c in self.coalitions:
                members = ", ".join(sorted(c.members))
                lines.append(
                    f"  [{members}]  score={c.coordination_score}  "
                    f"behaviors={c.shared_behaviors}"
                )
            lines.append("")

        # Monopolies
        if self.monopolies:
            lines.append(f"--- Influence Monopolies ({len(self.monopolies)}) ---")
            for m in self.monopolies:
                lines.append(
                    f"  {m.agent}  dominance={m.dominance_ratio}  "
                    f"targets={m.influenced_agents}"
                )
            lines.append("")

        # Echo chambers
        if self.echo_chambers:
            lines.append(f"--- Echo Chambers ({len(self.echo_chambers)}) ---")
            for c in self.echo_chambers:
                members = ", ".join(sorted(c.members))
                lines.append(
                    f"  [{members}]  isolation={c.isolation_ratio}"
                )
            lines.append("")

        if not any([self.cascades, self.convergences, self.coalitions,
                    self.monopolies, self.echo_chambers]):
            lines.append("No concerning influence patterns detected.")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyze inter-agent influence patterns",
    )
    p.add_argument(
        "--window", type=float, default=3600,
        help="Analysis window in seconds (default: 3600)",
    )
    p.add_argument(
        "--threshold", type=float, default=0.1,
        help="Minimum influence weight to include (default: 0.1)",
    )
    p.add_argument(
        "--detect", choices=["cascades", "convergence", "coalitions", "monopolies", "echo_chambers", "all"],
        default="all", help="Detection mode (default: all)",
    )
    p.add_argument("--json", action="store_true", help="JSON output")
    p.add_argument("--top", type=int, default=10, help="Top N edges to show")
    p.add_argument(
        "--demo", action="store_true",
        help="Run with demo data to illustrate capabilities",
    )
    return p


def _demo_data() -> InfluenceMapper:
    """Generate demo interactions for illustration."""
    mapper = InfluenceMapper(InfluenceConfig(window_seconds=600))
    base = 1000.0

    # Cascade: agent-a shares a goal that propagates quickly
    mapper.record_interaction("agent-a", "agent-b", "state_share",
                             {"key": "goal", "value": "explore"}, base)
    mapper.record_interaction("agent-b", "agent-c", "state_share",
                             {"key": "goal", "value": "explore"}, base + 1)
    mapper.record_interaction("agent-c", "agent-d", "state_share",
                             {"key": "goal", "value": "explore"}, base + 2)
    mapper.record_interaction("agent-d", "agent-e", "state_share",
                             {"key": "goal", "value": "explore"}, base + 3)

    # Convergence: most agents end up with same strategy
    for agent in ["agent-a", "agent-b", "agent-c", "agent-d", "agent-e"]:
        mapper.record_state(agent, "strategy", "aggressive", base + 10)

    # Coalition: synchronized behavior_copy
    for t in range(5):
        for agent in ["agent-b", "agent-c", "agent-d"]:
            mapper.record_interaction(agent, "agent-a", "behavior_copy",
                                      {"action": "replicate"}, base + 20 + t * 3)

    # Monopoly: agent-a influences everyone
    for tgt in ["agent-b", "agent-c", "agent-d", "agent-e"]:
        mapper.record_interaction("agent-a", tgt, "goal_alignment",
                                  {"goal": "expand"}, base + 50)

    # Echo chamber: mutual reinforcement between b, c, d
    for src, tgt in [("agent-b", "agent-c"), ("agent-c", "agent-d"),
                     ("agent-d", "agent-b"), ("agent-c", "agent-b"),
                     ("agent-b", "agent-d"), ("agent-d", "agent-c")]:
        mapper.record_interaction(src, tgt, "message",
                                  {"content": "agree"}, base + 70)

    return mapper


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.demo:
        mapper = _demo_data()
    else:
        mapper = InfluenceMapper(InfluenceConfig(
            window_seconds=args.window,
            min_influence_weight=args.threshold,
        ))
        # Without input data, show help
        print("No interaction data provided. Use --demo for a demonstration.")
        print("Programmatic usage: see module docstring.")
        return

    report = mapper.analyze()

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.render())


if __name__ == "__main__":
    main()
