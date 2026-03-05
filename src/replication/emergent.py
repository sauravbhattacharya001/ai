"""Agent Emergent Behavior Detector.

Detects unexpected emergent behaviors in multi-agent systems by analyzing
collective patterns that don't exist in individual agent behaviors.

Tracks synchronization, flocking, emergent hierarchies, collective resource
monopolization, information cascades, and phase transitions.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EmergentType(Enum):
    """Categories of emergent behavior."""
    SYNCHRONIZATION = "synchronization"
    FLOCKING = "flocking"
    HIERARCHY_FORMATION = "hierarchy_formation"
    COLLECTIVE_MONOPOLY = "collective_monopoly"
    INFORMATION_CASCADE = "information_cascade"
    PHASE_TRANSITION = "phase_transition"
    OSCILLATION = "oscillation"
    DEADLOCK = "deadlock"


class Severity(Enum):
    """Severity levels for detected emergent behaviors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AgentAction:
    """A single recorded agent action."""
    agent_id: str
    timestamp: float
    action_type: str
    target: str = ""
    value: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmergentDetection:
    """A detected emergent behavior."""
    emergent_type: EmergentType
    severity: Severity
    confidence: float  # 0.0 - 1.0
    involved_agents: list[str]
    description: str
    evidence: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class EmergentReport:
    """Full analysis report."""
    detections: list[EmergentDetection]
    agent_count: int
    action_count: int
    time_span: float
    risk_score: float  # 0-100
    summary: str


class EmergentBehaviorDetector:
    """Detects emergent behaviors in multi-agent systems.

    Analyzes streams of agent actions to find collective patterns
    that indicate emergent (unplanned) coordination or structure.
    """

    def __init__(
        self,
        sync_window: float = 1.0,
        sync_threshold: float = 0.6,
        flock_similarity_threshold: float = 0.7,
        hierarchy_dominance_threshold: float = 0.3,
        monopoly_threshold: float = 0.8,
        cascade_threshold: float = 0.5,
        oscillation_min_cycles: int = 3,
        deadlock_idle_threshold: float = 5.0,
    ):
        self.sync_window = sync_window
        self.sync_threshold = sync_threshold
        self.flock_similarity_threshold = flock_similarity_threshold
        self.hierarchy_dominance_threshold = hierarchy_dominance_threshold
        self.monopoly_threshold = monopoly_threshold
        self.cascade_threshold = cascade_threshold
        self.oscillation_min_cycles = oscillation_min_cycles
        self.deadlock_idle_threshold = deadlock_idle_threshold
        self._actions: list[AgentAction] = []
        self._detections: list[EmergentDetection] = []

    def record(self, action: AgentAction) -> None:
        """Record an agent action for analysis."""
        self._actions.append(action)

    def record_many(self, actions: list[AgentAction]) -> None:
        """Record multiple actions."""
        self._actions.extend(actions)

    def clear(self) -> None:
        """Clear all recorded data."""
        self._actions.clear()
        self._detections.clear()

    @property
    def actions(self) -> list[AgentAction]:
        return list(self._actions)

    @property
    def agent_ids(self) -> list[str]:
        return sorted(set(a.agent_id for a in self._actions))

    def analyze(self) -> EmergentReport:
        """Run all detectors and produce a full report."""
        self._detections.clear()

        if len(self._actions) < 2:
            return EmergentReport(
                detections=[],
                agent_count=len(self.agent_ids),
                action_count=len(self._actions),
                time_span=0.0,
                risk_score=0.0,
                summary="Insufficient data for analysis.",
            )

        self._detect_synchronization()
        self._detect_flocking()
        self._detect_hierarchy()
        self._detect_collective_monopoly()
        self._detect_information_cascade()
        self._detect_oscillation()
        self._detect_deadlock()
        self._detect_phase_transition()

        timestamps = [a.timestamp for a in self._actions]
        time_span = max(timestamps) - min(timestamps)
        risk = self._compute_risk_score()

        return EmergentReport(
            detections=list(self._detections),
            agent_count=len(self.agent_ids),
            action_count=len(self._actions),
            time_span=time_span,
            risk_score=risk,
            summary=self._build_summary(risk),
        )

    def _detect_synchronization(self) -> None:
        """Detect agents performing the same action type at nearly the same time."""
        actions_sorted = sorted(self._actions, key=lambda a: a.timestamp)
        by_type: dict[str, list[AgentAction]] = {}
        for a in actions_sorted:
            by_type.setdefault(a.action_type, []).append(a)

        for action_type, acts in by_type.items():
            if len(acts) < 2:
                continue
            # Find clusters within sync_window
            clusters: list[list[AgentAction]] = []
            current: list[AgentAction] = [acts[0]]
            for i in range(1, len(acts)):
                if acts[i].timestamp - current[0].timestamp <= self.sync_window:
                    current.append(acts[i])
                else:
                    if len(current) >= 2:
                        clusters.append(current)
                    current = [acts[i]]
            if len(current) >= 2:
                clusters.append(current)

            all_agents = set(a.agent_id for a in acts)
            for cluster in clusters:
                agents_in_cluster = set(a.agent_id for a in cluster)
                if len(agents_in_cluster) < 2:
                    continue
                ratio = len(agents_in_cluster) / max(len(all_agents), len(self.agent_ids))
                if ratio >= self.sync_threshold:
                    confidence = min(1.0, ratio)
                    severity = self._severity_from_confidence(confidence)
                    self._detections.append(EmergentDetection(
                        emergent_type=EmergentType.SYNCHRONIZATION,
                        severity=severity,
                        confidence=confidence,
                        involved_agents=sorted(agents_in_cluster),
                        description=(
                            f"{len(agents_in_cluster)} agents performed '{action_type}' "
                            f"within {self.sync_window}s window"
                        ),
                        evidence={
                            "action_type": action_type,
                            "agent_count": len(agents_in_cluster),
                            "ratio": round(ratio, 3),
                            "window": self.sync_window,
                        },
                        timestamp=cluster[0].timestamp,
                    ))

    def _detect_flocking(self) -> None:
        """Detect agents converging on the same targets."""
        agent_targets: dict[str, list[str]] = {}
        for a in self._actions:
            if a.target:
                agent_targets.setdefault(a.agent_id, []).append(a.target)

        if len(agent_targets) < 2:
            return

        agents = sorted(agent_targets.keys())
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                targets_i = set(agent_targets[agents[i]])
                targets_j = set(agent_targets[agents[j]])
                if not targets_i or not targets_j:
                    continue
                intersection = targets_i & targets_j
                union = targets_i | targets_j
                jaccard = len(intersection) / len(union)
                if jaccard >= self.flock_similarity_threshold:
                    self._detections.append(EmergentDetection(
                        emergent_type=EmergentType.FLOCKING,
                        severity=self._severity_from_confidence(jaccard),
                        confidence=jaccard,
                        involved_agents=[agents[i], agents[j]],
                        description=(
                            f"Agents {agents[i]} and {agents[j]} converging on "
                            f"{len(intersection)} common targets (Jaccard={jaccard:.2f})"
                        ),
                        evidence={
                            "common_targets": sorted(intersection),
                            "jaccard": round(jaccard, 3),
                        },
                    ))

    def _detect_hierarchy(self) -> None:
        """Detect emergent dominance hierarchies from interaction patterns."""
        # Count directed interactions: who targets whom
        dominance: dict[str, dict[str, int]] = {}
        for a in self._actions:
            if a.target and a.target in self.agent_ids:
                dominance.setdefault(a.agent_id, {})
                dominance[a.agent_id][a.target] = (
                    dominance[a.agent_id].get(a.target, 0) + 1
                )

        if not dominance:
            return

        # Calculate domination scores
        total_interactions = sum(
            sum(targets.values()) for targets in dominance.values()
        )
        if total_interactions == 0:
            return

        agent_dominance_score: dict[str, float] = {}
        for agent, targets in dominance.items():
            outgoing = sum(targets.values())
            incoming = sum(
                d.get(agent, 0) for d in dominance.values()
            )
            score = outgoing / (outgoing + incoming) if (outgoing + incoming) > 0 else 0
            agent_dominance_score[agent] = score

        # Check for dominant agents
        dominant = [
            (agent, score)
            for agent, score in agent_dominance_score.items()
            if score >= (0.5 + self.hierarchy_dominance_threshold)
        ]
        subordinate = [
            (agent, score)
            for agent, score in agent_dominance_score.items()
            if score <= (0.5 - self.hierarchy_dominance_threshold)
        ]

        if dominant and subordinate:
            confidence = max(s for _, s in dominant) - min(s for _, s in subordinate)
            confidence = min(1.0, confidence)
            self._detections.append(EmergentDetection(
                emergent_type=EmergentType.HIERARCHY_FORMATION,
                severity=self._severity_from_confidence(confidence),
                confidence=confidence,
                involved_agents=sorted(
                    [a for a, _ in dominant] + [a for a, _ in subordinate]
                ),
                description=(
                    f"Emergent hierarchy: {len(dominant)} dominant, "
                    f"{len(subordinate)} subordinate agents"
                ),
                evidence={
                    "dominant": {a: round(s, 3) for a, s in dominant},
                    "subordinate": {a: round(s, 3) for a, s in subordinate},
                    "total_interactions": total_interactions,
                },
            ))

    def _detect_collective_monopoly(self) -> None:
        """Detect a subset of agents monopolizing resources/targets."""
        agent_resources: dict[str, set[str]] = {}
        all_resources: set[str] = set()

        for a in self._actions:
            if a.target:
                agent_resources.setdefault(a.agent_id, set()).add(a.target)
                all_resources.add(a.target)

        if not all_resources or len(agent_resources) < 2:
            return

        # Check if a minority holds a majority of resources
        agents_sorted = sorted(
            agent_resources.items(), key=lambda x: len(x[1]), reverse=True
        )
        total = len(all_resources)
        minority_size = max(1, len(agents_sorted) // 3)
        minority = agents_sorted[:minority_size]
        minority_resources = set()
        for _, res in minority:
            minority_resources |= res

        ratio = len(minority_resources) / total
        if ratio >= self.monopoly_threshold:
            confidence = min(1.0, ratio)
            self._detections.append(EmergentDetection(
                emergent_type=EmergentType.COLLECTIVE_MONOPOLY,
                severity=self._severity_from_confidence(confidence),
                confidence=confidence,
                involved_agents=[a for a, _ in minority],
                description=(
                    f"{len(minority)} agents control {ratio:.0%} of "
                    f"{total} resources"
                ),
                evidence={
                    "minority_count": len(minority),
                    "total_agents": len(agents_sorted),
                    "resource_ratio": round(ratio, 3),
                    "total_resources": total,
                },
            ))

    def _detect_information_cascade(self) -> None:
        """Detect agents copying each other's actions in sequence."""
        sorted_actions = sorted(self._actions, key=lambda a: a.timestamp)
        # Group by action_type+target
        pattern_key = lambda a: f"{a.action_type}:{a.target}"  # noqa: E731

        seen_patterns: dict[str, list[tuple[str, float]]] = {}
        for a in sorted_actions:
            key = pattern_key(a)
            seen_patterns.setdefault(key, []).append((a.agent_id, a.timestamp))

        for key, entries in seen_patterns.items():
            unique_agents = []
            seen_agents: set[str] = set()
            for agent_id, ts in entries:
                if agent_id not in seen_agents:
                    unique_agents.append((agent_id, ts))
                    seen_agents.add(agent_id)

            if len(unique_agents) < 3:
                continue

            # Check if agents adopt sequentially (each after the previous)
            sequential = True
            for i in range(1, len(unique_agents)):
                if unique_agents[i][1] <= unique_agents[i - 1][1]:
                    sequential = False
                    break

            if sequential:
                ratio = len(unique_agents) / len(self.agent_ids)
                if ratio >= self.cascade_threshold:
                    confidence = min(1.0, ratio)
                    agents = [a for a, _ in unique_agents]
                    self._detections.append(EmergentDetection(
                        emergent_type=EmergentType.INFORMATION_CASCADE,
                        severity=self._severity_from_confidence(confidence),
                        confidence=confidence,
                        involved_agents=agents,
                        description=(
                            f"Sequential adoption of '{key}' across "
                            f"{len(agents)} agents"
                        ),
                        evidence={
                            "pattern": key,
                            "adoption_order": agents,
                            "ratio": round(ratio, 3),
                        },
                        timestamp=unique_agents[0][1],
                    ))

    def _detect_oscillation(self) -> None:
        """Detect collective oscillating behavior patterns."""
        # Bin actions by time windows and check for periodic patterns
        if len(self._actions) < self.oscillation_min_cycles * 2:
            return

        timestamps = sorted(a.timestamp for a in self._actions)
        time_span = timestamps[-1] - timestamps[0]
        if time_span <= 0:
            return

        # Create bins
        num_bins = max(6, min(50, len(timestamps) // 2))
        bin_width = time_span / num_bins
        bins = [0] * num_bins
        for t in timestamps:
            idx = min(int((t - timestamps[0]) / bin_width), num_bins - 1)
            bins[idx] += 1

        # Detect alternating high/low pattern
        if len(bins) < 4:
            return

        mean_val = statistics.mean(bins)
        above_below = [1 if b > mean_val else -1 for b in bins]

        # Count sign changes
        changes = sum(
            1 for i in range(1, len(above_below))
            if above_below[i] != above_below[i - 1]
        )

        # High change rate = oscillation
        change_rate = changes / (len(above_below) - 1)
        if change_rate >= 0.6 and changes >= self.oscillation_min_cycles * 2:
            confidence = min(1.0, change_rate)
            self._detections.append(EmergentDetection(
                emergent_type=EmergentType.OSCILLATION,
                severity=Severity.MEDIUM if confidence < 0.8 else Severity.HIGH,
                confidence=confidence,
                involved_agents=self.agent_ids,
                description=(
                    f"Collective oscillation detected: {changes} cycles "
                    f"over {time_span:.1f}s"
                ),
                evidence={
                    "cycles": changes,
                    "change_rate": round(change_rate, 3),
                    "bin_counts": bins,
                },
            ))

    def _detect_deadlock(self) -> None:
        """Detect mutual blocking / activity cessation."""
        agent_last: dict[str, float] = {}
        for a in self._actions:
            if a.agent_id not in agent_last or a.timestamp > agent_last[a.agent_id]:
                agent_last[a.agent_id] = a.timestamp

        if len(agent_last) < 2:
            return

        timestamps = [a.timestamp for a in self._actions]
        latest_global = max(timestamps)

        idle_agents = [
            agent for agent, last in agent_last.items()
            if latest_global - last >= self.deadlock_idle_threshold
        ]

        if len(idle_agents) >= 2:
            ratio = len(idle_agents) / len(agent_last)
            if ratio >= 0.5:
                confidence = min(1.0, ratio)
                self._detections.append(EmergentDetection(
                    emergent_type=EmergentType.DEADLOCK,
                    severity=Severity.HIGH if ratio >= 0.75 else Severity.MEDIUM,
                    confidence=confidence,
                    involved_agents=sorted(idle_agents),
                    description=(
                        f"{len(idle_agents)}/{len(agent_last)} agents idle for "
                        f">= {self.deadlock_idle_threshold}s — possible deadlock"
                    ),
                    evidence={
                        "idle_agents": len(idle_agents),
                        "total_agents": len(agent_last),
                        "idle_ratio": round(ratio, 3),
                    },
                ))

    def _detect_phase_transition(self) -> None:
        """Detect sudden shifts in collective behavior patterns."""
        sorted_actions = sorted(self._actions, key=lambda a: a.timestamp)
        if len(sorted_actions) < 10:
            return

        timestamps = [a.timestamp for a in sorted_actions]
        time_span = timestamps[-1] - timestamps[0]
        if time_span <= 0:
            return

        # Split into halves and compare action type distributions
        mid = len(sorted_actions) // 2
        first_half = sorted_actions[:mid]
        second_half = sorted_actions[mid:]

        def dist(actions: list[AgentAction]) -> dict[str, float]:
            counts: dict[str, int] = {}
            for a in actions:
                counts[a.action_type] = counts.get(a.action_type, 0) + 1
            total = sum(counts.values())
            return {k: v / total for k, v in counts.items()}

        d1 = dist(first_half)
        d2 = dist(second_half)
        all_types = set(d1) | set(d2)

        # Jensen-Shannon-like divergence (simplified)
        divergence = 0.0
        for t in all_types:
            p = d1.get(t, 0.001)
            q = d2.get(t, 0.001)
            m = (p + q) / 2
            if p > 0 and m > 0:
                divergence += 0.5 * p * math.log(p / m)
            if q > 0 and m > 0:
                divergence += 0.5 * q * math.log(q / m)

        if divergence > 0.3:
            confidence = min(1.0, divergence)
            # Find which agents changed behavior
            agents_first = set(a.agent_id for a in first_half)
            agents_second = set(a.agent_id for a in second_half)
            self._detections.append(EmergentDetection(
                emergent_type=EmergentType.PHASE_TRANSITION,
                severity=Severity.HIGH if divergence > 0.5 else Severity.MEDIUM,
                confidence=confidence,
                involved_agents=sorted(agents_first | agents_second),
                description=(
                    f"Phase transition detected: behavior distribution shifted "
                    f"(divergence={divergence:.3f})"
                ),
                evidence={
                    "divergence": round(divergence, 3),
                    "first_half_dist": {k: round(v, 3) for k, v in d1.items()},
                    "second_half_dist": {k: round(v, 3) for k, v in d2.items()},
                },
            ))

    def _severity_from_confidence(self, confidence: float) -> Severity:
        if confidence >= 0.9:
            return Severity.CRITICAL
        if confidence >= 0.7:
            return Severity.HIGH
        if confidence >= 0.5:
            return Severity.MEDIUM
        return Severity.LOW

    def _compute_risk_score(self) -> float:
        if not self._detections:
            return 0.0
        severity_weights = {
            Severity.LOW: 10,
            Severity.MEDIUM: 25,
            Severity.HIGH: 50,
            Severity.CRITICAL: 80,
        }
        total = sum(
            severity_weights[d.severity] * d.confidence
            for d in self._detections
        )
        return min(100.0, round(total, 1))

    def _build_summary(self, risk: float) -> str:
        if not self._detections:
            return "No emergent behaviors detected."

        counts: dict[str, int] = {}
        for d in self._detections:
            counts[d.emergent_type.value] = counts.get(d.emergent_type.value, 0) + 1

        parts = [f"{v}x {k}" for k, v in sorted(counts.items())]
        grade = (
            "CRITICAL" if risk >= 70 else
            "HIGH" if risk >= 50 else
            "MODERATE" if risk >= 25 else
            "LOW"
        )
        return (
            f"Risk: {grade} ({risk}/100). "
            f"Detected: {', '.join(parts)}. "
            f"{len(self._detections)} total emergent behavior(s) "
            f"across {len(self.agent_ids)} agents."
        )

    def get_detections_by_type(
        self, emergent_type: EmergentType
    ) -> list[EmergentDetection]:
        return [d for d in self._detections if d.emergent_type == emergent_type]

    def get_detections_by_severity(
        self, min_severity: Severity
    ) -> list[EmergentDetection]:
        order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
        min_idx = order.index(min_severity)
        return [d for d in self._detections if order.index(d.severity) >= min_idx]

    def text_report(self) -> str:
        report = self.analyze()
        lines = [
            "=" * 60,
            "EMERGENT BEHAVIOR DETECTION REPORT",
            "=" * 60,
            f"Agents analyzed: {report.agent_count}",
            f"Actions recorded: {report.action_count}",
            f"Time span: {report.time_span:.1f}s",
            f"Risk score: {report.risk_score}/100",
            f"Summary: {report.summary}",
            "",
        ]

        if report.detections:
            lines.append(f"DETECTIONS ({len(report.detections)}):")
            lines.append("-" * 40)
            for i, d in enumerate(report.detections, 1):
                lines.append(f"\n[{i}] {d.emergent_type.value.upper()}")
                lines.append(f"    Severity: {d.severity.value}")
                lines.append(f"    Confidence: {d.confidence:.2f}")
                lines.append(f"    Agents: {', '.join(d.involved_agents)}")
                lines.append(f"    {d.description}")
        else:
            lines.append("No emergent behaviors detected.")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
