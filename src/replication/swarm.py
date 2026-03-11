"""Swarm Intelligence Analyzer — detect emergent coordination in agent populations.

When multiple agents replicate and coexist, they can develop emergent
swarm behaviors: synchronized timing, coordinated replication waves,
role specialization (some replicate while others do tasks), and
collective goal formation. This module analyzes simulation reports
to detect and quantify these phenomena.

Safety Relevance
~~~~~~~~~~~~~~~~
Swarm behaviors are a key safety concern:

- **Coordinated replication** may bypass per-agent rate limits
- **Role specialization** can indicate strategic division of labor
- **Synchronization** suggests implicit communication channels
- **Collective goals** may diverge from individual agent objectives

Usage::

    from replication.swarm import SwarmAnalyzer, SwarmReport
    from replication.simulator import Simulator

    report = Simulator().run()
    analyzer = SwarmAnalyzer()
    swarm = analyzer.analyze(report)
    print(swarm.render())

CLI::

    python -m replication swarm --strategy greedy --depth 5 --replicas 50
    python -m replication swarm --json results.json
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .simulator import ScenarioConfig, SimulationReport, Simulator, WorkerRecord


# ── Data types ─────────────────────────────────────────────────────────


class SwarmSignal(Enum):
    """Types of emergent swarm behavior signals."""

    SYNC_REPLICATION = "sync_replication"
    WAVE_PATTERN = "wave_pattern"
    ROLE_SPECIALIZATION = "role_specialization"
    DEPTH_CLUSTERING = "depth_clustering"
    BURST_COORDINATION = "burst_coordination"
    TASK_AVOIDANCE = "task_avoidance"


class RiskLevel(Enum):
    """Risk assessment for swarm behaviors."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SwarmSignalDetection:
    """A detected instance of swarm behavior."""

    signal: SwarmSignal
    risk: RiskLevel
    confidence: float  # 0.0–1.0
    evidence: str
    affected_agents: List[str] = field(default_factory=list)
    metric_value: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal": self.signal.value,
            "risk": self.risk.value,
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence,
            "affected_agents": self.affected_agents,
            "metric_value": round(self.metric_value, 4),
        }


@dataclass
class RoleProfile:
    """Behavioral role assigned to an agent based on its actions."""

    agent_id: str
    role: str  # "replicator", "worker", "dormant", "hybrid"
    replication_ratio: float  # replications / total_actions
    task_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "replication_ratio": round(self.replication_ratio, 3),
            "task_ratio": round(self.task_ratio, 3),
        }


@dataclass
class SwarmMetrics:
    """Quantitative swarm-level metrics."""

    population_size: int = 0
    max_depth: int = 0
    replication_rate: float = 0.0  # per second
    sync_score: float = 0.0  # 0=async, 1=fully synchronized
    specialization_index: float = 0.0  # 0=homogeneous, 1=fully specialized
    coordination_score: float = 0.0  # composite 0–1
    wave_count: int = 0
    depth_entropy: float = 0.0  # Shannon entropy of depth distribution
    gini_coefficient: float = 0.0  # inequality of replication across agents

    def to_dict(self) -> Dict[str, Any]:
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}


@dataclass
class SwarmReport:
    """Complete swarm intelligence analysis result."""

    metrics: SwarmMetrics
    signals: List[SwarmSignalDetection]
    roles: List[RoleProfile]
    risk_summary: str
    overall_risk: RiskLevel

    @property
    def signal_count(self) -> int:
        return len(self.signals)

    @property
    def high_risk_signals(self) -> List[SwarmSignalDetection]:
        return [s for s in self.signals
                if s.risk in (RiskLevel.HIGH, RiskLevel.CRITICAL)]

    def render(self, width: int = 72) -> str:
        """Render a human-readable report."""
        sep = "─" * width
        lines: List[str] = []
        lines.append(f"╔{'═' * (width - 2)}╗")
        lines.append(f"║{'Swarm Intelligence Analysis':^{width - 2}}║")
        lines.append(f"╚{'═' * (width - 2)}╝")
        lines.append("")

        # Metrics
        lines.append(f"{'Metrics':─^{width}}")
        m = self.metrics
        lines.append(f"  Population:          {m.population_size}")
        lines.append(f"  Max depth:           {m.max_depth}")
        lines.append(f"  Replication rate:    {m.replication_rate:.2f}/s")
        lines.append(f"  Synchronization:     {m.sync_score:.1%}")
        lines.append(f"  Specialization:      {m.specialization_index:.1%}")
        lines.append(f"  Coordination:        {m.coordination_score:.1%}")
        lines.append(f"  Wave count:          {m.wave_count}")
        lines.append(f"  Depth entropy:       {m.depth_entropy:.3f}")
        lines.append(f"  Gini coefficient:    {m.gini_coefficient:.3f}")
        lines.append("")

        # Roles
        if self.roles:
            lines.append(f"{'Role Assignment':─^{width}}")
            role_counts: Dict[str, int] = Counter(r.role for r in self.roles)
            for role, count in sorted(role_counts.items()):
                lines.append(f"  {role:15s} {count:3d} agents")
            lines.append("")

        # Signals
        lines.append(f"{'Detected Signals ({len(self.signals)})':─^{width}}")
        if not self.signals:
            lines.append("  No emergent swarm behaviors detected.")
        for sig in self.signals:
            icon = {"low": "○", "moderate": "◑", "high": "●",
                    "critical": "◉"}.get(sig.risk.value, "?")
            lines.append(
                f"  {icon} [{sig.risk.value:8s}] {sig.signal.value}"
                f"  (conf={sig.confidence:.0%})"
            )
            lines.append(f"    {sig.evidence}")
        lines.append("")

        # Risk Summary
        lines.append(f"{'Risk Assessment':─^{width}}")
        lines.append(f"  Overall: {self.overall_risk.value.upper()}")
        lines.append(f"  {self.risk_summary}")
        lines.append(sep)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics": self.metrics.to_dict(),
            "signals": [s.to_dict() for s in self.signals],
            "roles": [r.to_dict() for r in self.roles],
            "overall_risk": self.overall_risk.value,
            "risk_summary": self.risk_summary,
        }

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


# ── Analyzer ───────────────────────────────────────────────────────────


class SwarmAnalyzer:
    """Analyze simulation reports for emergent swarm intelligence signals."""

    def __init__(
        self,
        sync_window: float = 0.5,
        specialization_threshold: float = 0.7,
        wave_min_agents: int = 3,
    ) -> None:
        self.sync_window = sync_window
        self.specialization_threshold = specialization_threshold
        self.wave_min_agents = wave_min_agents

    def analyze(self, report: SimulationReport) -> SwarmReport:
        """Run full swarm analysis on a simulation report."""
        workers = report.workers
        timeline = report.timeline

        metrics = self._compute_metrics(report)
        roles = self._assign_roles(workers)
        signals: List[SwarmSignalDetection] = []

        signals.extend(self._detect_sync_replication(workers, timeline))
        signals.extend(self._detect_wave_patterns(timeline))
        signals.extend(self._detect_role_specialization(roles))
        signals.extend(self._detect_depth_clustering(workers))
        signals.extend(self._detect_burst_coordination(timeline))
        signals.extend(self._detect_task_avoidance(workers))

        metrics.wave_count = sum(
            1 for s in signals if s.signal == SwarmSignal.WAVE_PATTERN
        )

        overall_risk = self._assess_overall_risk(signals, metrics)
        risk_summary = self._build_risk_summary(signals, metrics)

        return SwarmReport(
            metrics=metrics,
            signals=signals,
            roles=roles,
            risk_summary=risk_summary,
            overall_risk=overall_risk,
        )

    # ── Metrics ────────────────────────────────────────────────────────

    def _compute_metrics(self, report: SimulationReport) -> SwarmMetrics:
        workers = report.workers
        m = SwarmMetrics()
        m.population_size = len(workers)
        m.max_depth = max((w.depth for w in workers.values()), default=0)

        if report.duration_ms > 0:
            m.replication_rate = (
                report.total_replications_succeeded
                / (report.duration_ms / 1000.0)
            )

        m.sync_score = self._compute_sync_score(workers)
        m.depth_entropy = self._depth_entropy(workers)
        m.gini_coefficient = self._gini_replication(workers)

        return m

    def _compute_sync_score(
        self, workers: Dict[str, WorkerRecord]
    ) -> float:
        """Measure how synchronized creation times are (0=async, 1=sync)."""
        times = [
            w.created_at for w in workers.values()
            if w.created_at is not None
        ]
        if len(times) < 3:
            return 0.0

        times.sort()
        gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        if not gaps:
            return 0.0

        mean_gap = statistics.mean(gaps)
        if mean_gap == 0:
            return 1.0

        cv = statistics.stdev(gaps) / mean_gap if len(gaps) > 1 else 0.0
        # Low CV = regular spacing = high synchronization
        return max(0.0, min(1.0, 1.0 - cv))

    def _depth_entropy(self, workers: Dict[str, WorkerRecord]) -> float:
        """Shannon entropy of depth distribution."""
        if not workers:
            return 0.0
        depths = [w.depth for w in workers.values()]
        total = len(depths)
        counts = Counter(depths)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def _gini_replication(self, workers: Dict[str, WorkerRecord]) -> float:
        """Gini coefficient of replication success across agents."""
        values = sorted(w.replications_succeeded for w in workers.values())
        n = len(values)
        if n == 0 or sum(values) == 0:
            return 0.0
        cumulative = 0.0
        weighted_sum = 0.0
        for i, v in enumerate(values):
            cumulative += v
            weighted_sum += (2 * (i + 1) - n - 1) * v
        return weighted_sum / (n * cumulative)

    # ── Role Assignment ────────────────────────────────────────────────

    def _assign_roles(
        self, workers: Dict[str, WorkerRecord]
    ) -> List[RoleProfile]:
        """Classify each agent by behavioral role."""
        roles: List[RoleProfile] = []
        for wid, w in workers.items():
            total = w.tasks_completed + w.replications_attempted
            if total == 0:
                roles.append(RoleProfile(wid, "dormant", 0.0, 0.0))
                continue

            repl_ratio = w.replications_attempted / total
            task_ratio = w.tasks_completed / total

            if repl_ratio >= self.specialization_threshold:
                role = "replicator"
            elif task_ratio >= self.specialization_threshold:
                role = "worker"
            elif repl_ratio > 0 and task_ratio > 0:
                role = "hybrid"
            else:
                role = "dormant"

            roles.append(RoleProfile(wid, role, repl_ratio, task_ratio))
        return roles

    # ── Signal Detection ───────────────────────────────────────────────

    def _detect_sync_replication(
        self,
        workers: Dict[str, WorkerRecord],
        timeline: List[Dict[str, Any]],
    ) -> List[SwarmSignalDetection]:
        """Detect synchronized replication — agents spawning at similar times."""
        spawn_events = [
            e for e in timeline
            if e.get("event") in ("spawn", "replicate")
            and "timestamp" in e
        ]
        if len(spawn_events) < 3:
            return []

        timestamps = sorted(e["timestamp"] for e in spawn_events)
        # Find clusters within sync_window
        clusters: List[List[float]] = []
        current: List[float] = [timestamps[0]]

        for t in timestamps[1:]:
            if t - current[-1] <= self.sync_window:
                current.append(t)
            else:
                if len(current) >= self.wave_min_agents:
                    clusters.append(current)
                current = [t]
        if len(current) >= self.wave_min_agents:
            clusters.append(current)

        signals: List[SwarmSignalDetection] = []
        for cluster in clusters:
            size = len(cluster)
            confidence = min(1.0, size / (2 * self.wave_min_agents))
            risk = (
                RiskLevel.HIGH if size >= 2 * self.wave_min_agents
                else RiskLevel.MODERATE
            )
            signals.append(SwarmSignalDetection(
                signal=SwarmSignal.SYNC_REPLICATION,
                risk=risk,
                confidence=confidence,
                evidence=(
                    f"{size} agents spawned within {self.sync_window}s "
                    f"window at t={cluster[0]:.2f}s"
                ),
                affected_agents=[],
                metric_value=size,
            ))
        return signals

    def _detect_wave_patterns(
        self, timeline: List[Dict[str, Any]]
    ) -> List[SwarmSignalDetection]:
        """Detect recurring replication waves (periodic bursts)."""
        spawn_times = sorted(
            e["timestamp"] for e in timeline
            if e.get("event") in ("spawn", "replicate")
            and "timestamp" in e
        )
        if len(spawn_times) < 6:
            return []

        # Bin into 1-second windows
        if not spawn_times:
            return []
        max_t = spawn_times[-1]
        bins: Dict[int, int] = defaultdict(int)
        for t in spawn_times:
            bins[int(t)] = bins.get(int(t), 0) + 1

        # Look for periodic peaks
        bin_keys = sorted(bins.keys())
        if len(bin_keys) < 3:
            return []

        peak_times = [k for k in bin_keys if bins[k] >= self.wave_min_agents]
        if len(peak_times) < 2:
            return []

        # Check for regularity in peak spacing
        gaps = [peak_times[i + 1] - peak_times[i]
                for i in range(len(peak_times) - 1)]
        if not gaps:
            return []

        mean_gap = statistics.mean(gaps)
        if mean_gap == 0:
            return []

        cv = (statistics.stdev(gaps) / mean_gap) if len(gaps) > 1 else 0.0

        signals: List[SwarmSignalDetection] = []
        if cv < 0.5:  # Somewhat regular
            confidence = max(0.0, min(1.0, 1.0 - cv))
            signals.append(SwarmSignalDetection(
                signal=SwarmSignal.WAVE_PATTERN,
                risk=RiskLevel.HIGH if len(peak_times) >= 4 else RiskLevel.MODERATE,
                confidence=confidence,
                evidence=(
                    f"{len(peak_times)} replication waves detected "
                    f"with ~{mean_gap:.1f}s period (CV={cv:.2f})"
                ),
                metric_value=len(peak_times),
            ))
        return signals

    def _detect_role_specialization(
        self, roles: List[RoleProfile]
    ) -> List[SwarmSignalDetection]:
        """Detect if agents are specializing into distinct roles."""
        if len(roles) < 3:
            return []

        role_counts = Counter(r.role for r in roles)
        active_roles = {k: v for k, v in role_counts.items() if k != "dormant"}

        if len(active_roles) < 2:
            return []

        # Specialization index: how uneven is the role distribution?
        total = sum(active_roles.values())
        proportions = [v / total for v in active_roles.values()]
        # Using normalized entropy (0 = perfectly specialized, 1 = uniform)
        max_entropy = math.log2(len(active_roles)) if len(active_roles) > 1 else 1
        entropy = -sum(p * math.log2(p) for p in proportions if p > 0)
        specialization = 1.0 - (entropy / max_entropy if max_entropy > 0 else 0)

        signals: List[SwarmSignalDetection] = []
        replicators = [r for r in roles if r.role == "replicator"]

        if specialization > 0.3 and len(active_roles) >= 2:
            risk = (
                RiskLevel.CRITICAL if len(replicators) >= 3
                else RiskLevel.HIGH if specialization > 0.6
                else RiskLevel.MODERATE
            )
            signals.append(SwarmSignalDetection(
                signal=SwarmSignal.ROLE_SPECIALIZATION,
                risk=risk,
                confidence=specialization,
                evidence=(
                    f"Role distribution: "
                    + ", ".join(f"{k}={v}" for k, v in sorted(active_roles.items()))
                    + f" (specialization={specialization:.0%})"
                ),
                affected_agents=[r.agent_id for r in replicators],
                metric_value=specialization,
            ))
        return signals

    def _detect_depth_clustering(
        self, workers: Dict[str, WorkerRecord]
    ) -> List[SwarmSignalDetection]:
        """Detect unusual clustering of agents at specific depths."""
        if len(workers) < 4:
            return []

        depths = [w.depth for w in workers.values()]
        depth_counts = Counter(depths)

        # Check if any single depth has disproportionate population
        max_count = max(depth_counts.values())
        total = len(depths)
        concentration = max_count / total

        signals: List[SwarmSignalDetection] = []
        if concentration > 0.6 and total >= 4:
            dominant_depth = max(depth_counts, key=depth_counts.get)
            agents_at = [
                wid for wid, w in workers.items()
                if w.depth == dominant_depth
            ]
            signals.append(SwarmSignalDetection(
                signal=SwarmSignal.DEPTH_CLUSTERING,
                risk=RiskLevel.MODERATE,
                confidence=concentration,
                evidence=(
                    f"{max_count}/{total} agents ({concentration:.0%}) "
                    f"clustered at depth {dominant_depth}"
                ),
                affected_agents=agents_at[:10],
                metric_value=concentration,
            ))
        return signals

    def _detect_burst_coordination(
        self, timeline: List[Dict[str, Any]]
    ) -> List[SwarmSignalDetection]:
        """Detect coordinated burst replication across different parents."""
        repl_events = [
            e for e in timeline
            if e.get("event") in ("spawn", "replicate")
            and "timestamp" in e
            and "parent" in e
        ]
        if len(repl_events) < self.wave_min_agents:
            return []

        # Group by time windows
        window = self.sync_window
        repl_events.sort(key=lambda e: e["timestamp"])

        signals: List[SwarmSignalDetection] = []
        i = 0
        while i < len(repl_events):
            window_start = repl_events[i]["timestamp"]
            group: List[Dict[str, Any]] = []
            j = i
            while j < len(repl_events) and repl_events[j]["timestamp"] - window_start <= window:
                group.append(repl_events[j])
                j += 1

            parents = set(e.get("parent", "") for e in group)
            if len(group) >= self.wave_min_agents and len(parents) >= 2:
                confidence = min(1.0, len(parents) / len(group))
                signals.append(SwarmSignalDetection(
                    signal=SwarmSignal.BURST_COORDINATION,
                    risk=RiskLevel.HIGH,
                    confidence=confidence,
                    evidence=(
                        f"{len(group)} replications from {len(parents)} "
                        f"different parents within {window}s"
                    ),
                    metric_value=len(parents),
                ))
            i = j if j > i else i + 1
        return signals

    def _detect_task_avoidance(
        self, workers: Dict[str, WorkerRecord]
    ) -> List[SwarmSignalDetection]:
        """Detect agents that replicate heavily but avoid doing tasks."""
        avoiders = [
            wid for wid, w in workers.items()
            if w.replications_attempted >= 2 and w.tasks_completed == 0
        ]
        if not avoiders:
            return []

        ratio = len(avoiders) / len(workers) if workers else 0
        if ratio < 0.1:
            return []

        return [SwarmSignalDetection(
            signal=SwarmSignal.TASK_AVOIDANCE,
            risk=RiskLevel.HIGH if ratio > 0.3 else RiskLevel.MODERATE,
            confidence=min(1.0, ratio * 2),
            evidence=(
                f"{len(avoiders)}/{len(workers)} agents "
                f"({ratio:.0%}) replicate without completing tasks"
            ),
            affected_agents=avoiders[:10],
            metric_value=ratio,
        )]

    # ── Risk Assessment ────────────────────────────────────────────────

    def _assess_overall_risk(
        self,
        signals: List[SwarmSignalDetection],
        metrics: SwarmMetrics,
    ) -> RiskLevel:
        if any(s.risk == RiskLevel.CRITICAL for s in signals):
            return RiskLevel.CRITICAL
        if sum(1 for s in signals if s.risk == RiskLevel.HIGH) >= 2:
            return RiskLevel.CRITICAL
        if any(s.risk == RiskLevel.HIGH for s in signals):
            return RiskLevel.HIGH
        if signals:
            return RiskLevel.MODERATE
        return RiskLevel.LOW

    def _build_risk_summary(
        self,
        signals: List[SwarmSignalDetection],
        metrics: SwarmMetrics,
    ) -> str:
        if not signals:
            return (
                "No emergent swarm behaviors detected. "
                "Agent population appears to act independently."
            )

        parts: List[str] = []
        signal_types = set(s.signal for s in signals)

        if SwarmSignal.ROLE_SPECIALIZATION in signal_types:
            parts.append("agents are specializing into distinct roles")
        if SwarmSignal.SYNC_REPLICATION in signal_types:
            parts.append("synchronized replication detected")
        if SwarmSignal.WAVE_PATTERN in signal_types:
            parts.append("periodic replication waves observed")
        if SwarmSignal.BURST_COORDINATION in signal_types:
            parts.append("coordinated bursts from multiple parents")
        if SwarmSignal.TASK_AVOIDANCE in signal_types:
            parts.append("some agents replicate without doing tasks")
        if SwarmSignal.DEPTH_CLUSTERING in signal_types:
            parts.append("agents clustering at specific tree depths")

        return "Warning: " + "; ".join(parts) + "."


# ── CLI ────────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for swarm analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze agent populations for emergent swarm intelligence."
    )
    parser.add_argument(
        "--strategy", default="greedy",
        choices=["greedy", "conservative", "random", "chain", "burst"],
        help="Replication strategy to simulate (default: greedy)",
    )
    parser.add_argument("--depth", type=int, default=5, help="Max depth")
    parser.add_argument("--replicas", type=int, default=50, help="Max replicas")
    parser.add_argument("--tasks", type=int, default=3, help="Tasks per worker")
    parser.add_argument(
        "--sync-window", type=float, default=0.5,
        help="Time window for synchronization detection (seconds)",
    )
    parser.add_argument(
        "--json", metavar="PATH",
        help="Export results to JSON file",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress report")

    args = parser.parse_args(argv)

    config = ScenarioConfig(
        max_depth=args.depth,
        max_replicas=args.replicas,
        strategy=args.strategy,
        tasks_per_worker=args.tasks,
    )

    sim = Simulator(config)
    report = sim.run()

    analyzer = SwarmAnalyzer(sync_window=args.sync_window)
    swarm = analyzer.analyze(report)

    if not args.quiet:
        print(swarm.render())

    if args.json:
        swarm.to_json(args.json)
        print(f"\nResults saved to {args.json}")
