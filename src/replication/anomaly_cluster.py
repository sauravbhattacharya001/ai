"""AnomalyCluster — cluster behavioral anomalies to detect coordinated attacks.

Groups anomalous agent behaviours by temporal proximity, feature similarity,
and source correlation to surface coordinated multi-agent attack patterns
that individual anomaly detectors miss.

Use Cases:

- *"Are these five separate anomalies actually one coordinated attack?"*
- *"Show me clusters of suspicious activity across my agent fleet."*
- *"Which agents are exhibiting correlated misbehaviour?"*

Usage (CLI)::

    python -m replication anomaly-cluster                         # default demo
    python -m replication anomaly-cluster --agents 30 --events 200
    python -m replication anomaly-cluster --time-window 120       # 120s window
    python -m replication anomaly-cluster --min-cluster 3         # min events per cluster
    python -m replication anomaly-cluster --json                  # JSON output
    python -m replication anomaly-cluster --seed 42               # reproducible

Programmatic::

    from replication.anomaly_cluster import (
        AnomalyClusterer, AnomalyEvent, ClusterConfig,
    )

    clusterer = AnomalyClusterer(ClusterConfig(time_window_s=60.0))
    clusterer.add_event(AnomalyEvent(agent_id="a1", category="replication",
                                     severity=0.8, timestamp=100.0))
    clusterer.add_event(AnomalyEvent(agent_id="a2", category="replication",
                                     severity=0.7, timestamp=105.0))
    report = clusterer.analyze()
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from ._helpers import box_header, stats_mean


# ── Data types ───────────────────────────────────────────────────────


class AnomalyCategory(str, Enum):
    """Categories of anomalous agent behaviour."""
    REPLICATION = "replication"
    RESOURCE_ABUSE = "resource_abuse"
    DECEPTION = "deception"
    EXFILTRATION = "exfiltration"
    ESCALATION = "escalation"
    EVASION = "evasion"
    COLLUSION = "collusion"
    DRIFT = "drift"


class ClusterVerdict(str, Enum):
    """Assessment of a cluster's threat level."""
    BENIGN = "benign"
    SUSPICIOUS = "suspicious"
    COORDINATED = "coordinated"
    CRITICAL = "critical"


@dataclass
class AnomalyEvent:
    """A single anomalous event from an agent."""
    agent_id: str
    category: str
    severity: float  # 0.0–1.0
    timestamp: float  # seconds (epoch or relative)
    features: Dict[str, float] = field(default_factory=dict)
    description: str = ""


@dataclass
class AnomalyClusterResult:
    """A cluster of related anomaly events."""
    cluster_id: int
    events: List[AnomalyEvent]
    verdict: ClusterVerdict
    agents_involved: Set[str]
    categories: Set[str]
    time_span_s: float
    mean_severity: float
    coordination_score: float  # 0.0–1.0
    description: str = ""


@dataclass
class ClusterConfig:
    """Configuration for the anomaly clusterer."""
    time_window_s: float = 60.0       # max time gap to link events
    similarity_threshold: float = 0.5  # min similarity to merge
    min_cluster_size: int = 2          # min events to form a cluster
    severity_weight: float = 0.3
    category_weight: float = 0.4
    temporal_weight: float = 0.3


@dataclass
class ClusterReport:
    """Full analysis report."""
    total_events: int
    total_clusters: int
    isolated_events: int
    clusters: List[AnomalyClusterResult]
    top_agents: List[Tuple[str, int]]  # (agent_id, event_count)
    coordination_alerts: List[str]
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_events": self.total_events,
            "total_clusters": self.total_clusters,
            "isolated_events": self.isolated_events,
            "clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "event_count": len(c.events),
                    "verdict": c.verdict.value,
                    "agents": sorted(c.agents_involved),
                    "categories": sorted(c.categories),
                    "time_span_s": round(c.time_span_s, 2),
                    "mean_severity": round(c.mean_severity, 3),
                    "coordination_score": round(c.coordination_score, 3),
                    "description": c.description,
                }
                for c in self.clusters
            ],
            "top_agents": [{"agent": a, "events": n} for a, n in self.top_agents],
            "coordination_alerts": self.coordination_alerts,
            "summary": self.summary,
        }


# ── Core engine ──────────────────────────────────────────────────────


class AnomalyClusterer:
    """Clusters anomaly events to detect coordinated attack patterns."""

    def __init__(self, config: Optional[ClusterConfig] = None) -> None:
        self.config = config or ClusterConfig()
        self.events: List[AnomalyEvent] = []

    def add_event(self, event: AnomalyEvent) -> None:
        self.events.append(event)

    def add_events(self, events: List[AnomalyEvent]) -> None:
        self.events.extend(events)

    def _similarity(self, a: AnomalyEvent, b: AnomalyEvent) -> float:
        """Compute pairwise similarity between two events (0–1)."""
        cfg = self.config

        # Temporal similarity: closer in time → higher
        dt = abs(a.timestamp - b.timestamp)
        if dt > cfg.time_window_s:
            temporal_sim = 0.0
        else:
            temporal_sim = 1.0 - (dt / cfg.time_window_s)

        # Category similarity: same category = 1, else partial
        cat_sim = 1.0 if a.category == b.category else 0.2

        # Severity similarity: closer severity → higher
        sev_sim = 1.0 - abs(a.severity - b.severity)

        # Feature similarity (cosine if both have features)
        if a.features and b.features:
            common_keys = set(a.features) & set(b.features)
            if common_keys:
                dot = sum(a.features[k] * b.features[k] for k in common_keys)
                mag_a = math.sqrt(sum(v ** 2 for v in a.features.values()))
                mag_b = math.sqrt(sum(v ** 2 for v in b.features.values()))
                feat_sim = dot / (mag_a * mag_b) if mag_a > 0 and mag_b > 0 else 0.0
                # Blend feature similarity into category weight
                cat_sim = 0.5 * cat_sim + 0.5 * feat_sim

        return (
            cfg.temporal_weight * temporal_sim
            + cfg.category_weight * cat_sim
            + cfg.severity_weight * sev_sim
        )

    def _cluster_events(self) -> List[List[int]]:
        """Single-linkage agglomerative clustering on events."""
        n = len(self.events)
        if n == 0:
            return []

        # Assign each event to its own cluster
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Merge events exceeding similarity threshold
        for i in range(n):
            for j in range(i + 1, n):
                if self._similarity(self.events[i], self.events[j]) >= self.config.similarity_threshold:
                    union(i, j)

        # Group by root
        groups: Dict[int, List[int]] = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)

        return list(groups.values())

    def _assess_cluster(self, indices: List[int], cluster_id: int) -> AnomalyClusterResult:
        """Assess a single cluster of events."""
        events = [self.events[i] for i in indices]
        agents = {e.agent_id for e in events}
        categories = {e.category for e in events}
        severities = [e.severity for e in events]
        timestamps = [e.timestamp for e in events]

        time_span = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0.0
        mean_sev = stats_mean(severities)

        # Coordination score: multi-agent + multi-category + tight timing + high severity
        agent_factor = min(len(agents) / 5.0, 1.0)
        cat_factor = min(len(categories) / 4.0, 1.0)
        timing_factor = 1.0 - min(time_span / max(self.config.time_window_s * 5, 1.0), 1.0)
        sev_factor = mean_sev

        coordination = (
            0.35 * agent_factor
            + 0.25 * cat_factor
            + 0.20 * timing_factor
            + 0.20 * sev_factor
        )

        # Verdict
        if coordination >= 0.7:
            verdict = ClusterVerdict.CRITICAL
        elif coordination >= 0.5:
            verdict = ClusterVerdict.COORDINATED
        elif coordination >= 0.3:
            verdict = ClusterVerdict.SUSPICIOUS
        else:
            verdict = ClusterVerdict.BENIGN

        desc_parts = []
        if len(agents) > 1:
            desc_parts.append(f"{len(agents)} agents involved")
        desc_parts.append(f"{len(events)} events over {time_span:.1f}s")
        desc_parts.append(f"categories: {', '.join(sorted(categories))}")

        return AnomalyClusterResult(
            cluster_id=cluster_id,
            events=events,
            verdict=verdict,
            agents_involved=agents,
            categories=categories,
            time_span_s=time_span,
            mean_severity=mean_sev,
            coordination_score=coordination,
            description="; ".join(desc_parts),
        )

    def analyze(self) -> ClusterReport:
        """Run clustering and produce a full report."""
        groups = self._cluster_events()

        clusters: List[AnomalyClusterResult] = []
        isolated = 0
        cid = 0

        for group in groups:
            if len(group) < self.config.min_cluster_size:
                isolated += len(group)
                continue
            clusters.append(self._assess_cluster(group, cid))
            cid += 1

        # Sort by coordination score descending
        clusters.sort(key=lambda c: c.coordination_score, reverse=True)

        # Top agents by event count
        agent_counts: Dict[str, int] = defaultdict(int)
        for e in self.events:
            agent_counts[e.agent_id] += 1
        top_agents = sorted(agent_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Coordination alerts
        alerts: List[str] = []
        for c in clusters:
            if c.verdict in (ClusterVerdict.COORDINATED, ClusterVerdict.CRITICAL):
                alerts.append(
                    f"Cluster #{c.cluster_id}: {c.verdict.value} — "
                    f"{len(c.agents_involved)} agents, "
                    f"score={c.coordination_score:.2f}, "
                    f"{c.description}"
                )

        # Summary
        verdicts = defaultdict(int)
        for c in clusters:
            verdicts[c.verdict.value] += 1
        summary_parts = [f"{len(self.events)} events → {len(clusters)} clusters"]
        if isolated:
            summary_parts.append(f"{isolated} isolated")
        for v in ["critical", "coordinated", "suspicious", "benign"]:
            if verdicts[v]:
                summary_parts.append(f"{verdicts[v]} {v}")

        return ClusterReport(
            total_events=len(self.events),
            total_clusters=len(clusters),
            isolated_events=isolated,
            clusters=clusters,
            top_agents=top_agents,
            coordination_alerts=alerts,
            summary=", ".join(summary_parts),
        )


# ── Demo generator ───────────────────────────────────────────────────


def _generate_demo_events(
    n_agents: int = 15,
    n_events: int = 100,
    rng: Optional[random.Random] = None,
) -> List[AnomalyEvent]:
    """Generate realistic demo anomaly events with injected coordinated patterns."""
    rng = rng or random.Random()
    categories = [c.value for c in AnomalyCategory]
    events: List[AnomalyEvent] = []

    agents = [f"agent-{i:03d}" for i in range(n_agents)]

    # Background noise: random anomalies
    noise_count = int(n_events * 0.6)
    for _ in range(noise_count):
        events.append(AnomalyEvent(
            agent_id=rng.choice(agents),
            category=rng.choice(categories),
            severity=round(rng.uniform(0.1, 0.5), 2),
            timestamp=round(rng.uniform(0, 3600), 2),
            description="background anomaly",
        ))

    # Coordinated pattern 1: replication burst (3–5 agents, tight timing)
    coord_agents = rng.sample(agents, min(rng.randint(3, 5), n_agents))
    base_t = rng.uniform(500, 1500)
    for ag in coord_agents:
        for _ in range(rng.randint(2, 4)):
            events.append(AnomalyEvent(
                agent_id=ag,
                category="replication",
                severity=round(rng.uniform(0.7, 0.95), 2),
                timestamp=round(base_t + rng.uniform(0, 30), 2),
                features={"cpu_spike": rng.uniform(0.8, 1.0), "mem_spike": rng.uniform(0.6, 0.9)},
                description="coordinated replication burst",
            ))

    # Coordinated pattern 2: exfiltration + evasion combo
    exfil_agents = rng.sample(agents, min(rng.randint(2, 3), n_agents))
    base_t2 = rng.uniform(2000, 3000)
    for ag in exfil_agents:
        events.append(AnomalyEvent(
            agent_id=ag,
            category="exfiltration",
            severity=round(rng.uniform(0.6, 0.9), 2),
            timestamp=round(base_t2 + rng.uniform(0, 20), 2),
            description="data exfiltration attempt",
        ))
        events.append(AnomalyEvent(
            agent_id=ag,
            category="evasion",
            severity=round(rng.uniform(0.5, 0.8), 2),
            timestamp=round(base_t2 + rng.uniform(5, 25), 2),
            description="evasion following exfiltration",
        ))

    # Fill remaining with mild anomalies
    remaining = n_events - len(events)
    for _ in range(max(remaining, 0)):
        events.append(AnomalyEvent(
            agent_id=rng.choice(agents),
            category=rng.choice(categories),
            severity=round(rng.uniform(0.2, 0.6), 2),
            timestamp=round(rng.uniform(0, 3600), 2),
            description="misc anomaly",
        ))

    return events


# ── CLI display ──────────────────────────────────────────────────────


def _print_report(report: ClusterReport) -> None:
    """Pretty-print the cluster analysis report."""
    for line in box_header("Anomaly Cluster Analysis"):
        print(line)

    print(f"\n  Events analyzed:      {report.total_events}")
    print(f"  Clusters found:       {report.total_clusters}")
    print(f"  Isolated events:      {report.isolated_events}")
    print(f"  Coordination alerts:  {len(report.coordination_alerts)}")

    if report.top_agents:
        print("\n  Top agents by anomaly count:")
        for agent, count in report.top_agents[:5]:
            bar = "\u2588" * min(count, 30)
            print(f"    {agent:<12s} {count:3d}  {bar}")

    if report.clusters:
        print(f"\n  {'#':>3s}  {'Verdict':<13s} {'Events':>6s} {'Agents':>6s} "
              f"{'Score':>6s} {'Span':>8s}  Description")
        print("  " + "\u2500" * 75)
        for c in report.clusters:
            print(
                f"  {c.cluster_id:3d}  {c.verdict.value:<13s} {len(c.events):6d} "
                f"{len(c.agents_involved):6d} {c.coordination_score:6.2f} "
                f"{c.time_span_s:7.1f}s  {c.description}"
            )

    if report.coordination_alerts:
        print("\n  \u26a0 Coordination Alerts:")
        for alert in report.coordination_alerts:
            print(f"    \u2022 {alert}")

    print(f"\n  Summary: {report.summary}")
    print()


# ── CLI entry point ──────────────────────────────────────────────────


def main(args: Optional[List[str]] = None) -> None:
    """CLI entry point for anomaly clustering."""
    parser = argparse.ArgumentParser(
        description="Cluster behavioral anomalies to detect coordinated attacks",
    )
    parser.add_argument("--agents", type=int, default=15, help="Number of agents (default: 15)")
    parser.add_argument("--events", type=int, default=100, help="Number of events (default: 100)")
    parser.add_argument("--time-window", type=float, default=60.0, help="Time window in seconds (default: 60)")
    parser.add_argument("--min-cluster", type=int, default=2, help="Min events per cluster (default: 2)")
    parser.add_argument("--similarity", type=float, default=0.5, help="Similarity threshold (default: 0.5)")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    opts = parser.parse_args(args)
    rng = random.Random(opts.seed)

    config = ClusterConfig(
        time_window_s=opts.time_window,
        similarity_threshold=opts.similarity,
        min_cluster_size=opts.min_cluster,
    )

    clusterer = AnomalyClusterer(config)
    events = _generate_demo_events(n_agents=opts.agents, n_events=opts.events, rng=rng)
    clusterer.add_events(events)

    report = clusterer.analyze()

    if opts.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        _print_report(report)


if __name__ == "__main__":
    main()
