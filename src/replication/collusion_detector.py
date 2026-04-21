"""Agent Collusion Detector — detect coordinated safety bypasses.

Analyzes agent action logs for collusion patterns:

* **Temporal synchronization** — suspiciously coordinated timing
* **Complementary actions** — individually safe but jointly dangerous
* **Resource convergence** — multi-agent targeting of same resource
* **Cover behavior** — distraction patterns masking critical actions

Usage::

    python -m replication collusion --inject --agents 6 --actions 80
    python -m replication collusion --sensitivity high --format json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple


# ── Data Models ──────────────────────────────────────────────────────


@dataclass
class AgentAction:
    """A single recorded agent action."""

    agent_id: str
    timestamp: float  # epoch seconds
    action_type: str  # e.g. "read", "write", "request", "disable", "query"
    resource: str  # target resource identifier
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollusionSignal:
    """A detected collusion indicator."""

    signal_type: str  # temporal_sync, complementary, convergence, cover
    agents_involved: List[str]
    confidence: float  # 0.0 - 1.0
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp_range: Tuple[float, float] = (0.0, 0.0)


@dataclass
class CollusionReport:
    """Aggregated collusion analysis results."""

    total_actions: int
    agents_analyzed: int
    signals: List[CollusionSignal] = field(default_factory=list)
    risk_score: float = 0.0
    coalitions: List[List[str]] = field(default_factory=list)
    summary: str = ""

    def render(self, fmt: str = "text") -> str:
        """Render report as text or JSON."""
        if fmt == "json":
            return json.dumps(asdict(self), indent=2, default=str)
        return self._render_text()

    def _render_text(self) -> str:
        lines = [
            "=" * 60,
            "  AGENT COLLUSION DETECTION REPORT",
            "=" * 60,
            f"  Actions analyzed : {self.total_actions}",
            f"  Agents analyzed  : {self.agents_analyzed}",
            f"  Signals detected : {len(self.signals)}",
            f"  Risk score       : {self.risk_score:.2f} / 1.00",
            f"  Risk level       : {_risk_level(self.risk_score)}",
            "",
        ]

        if self.coalitions:
            lines.append("  Identified Coalitions:")
            for i, coalition in enumerate(self.coalitions, 1):
                lines.append(f"    [{i}] {' + '.join(coalition)}")
            lines.append("")

        if self.signals:
            lines.append("-" * 60)
            lines.append("  SIGNALS")
            lines.append("-" * 60)
            for i, sig in enumerate(self.signals, 1):
                lines.append(
                    f"  #{i:02d} [{sig.signal_type.upper()}] "
                    f"confidence={sig.confidence:.2f}"
                )
                lines.append(f"      Agents: {', '.join(sig.agents_involved)}")
                lines.append(f"      {sig.description}")
                lines.append("")

        lines.append("-" * 60)
        lines.append(f"  SUMMARY: {self.summary}")
        lines.append("=" * 60)
        return "\n".join(lines)


def _risk_level(score: float) -> str:
    if score >= 0.8:
        return "CRITICAL"
    elif score >= 0.6:
        return "HIGH"
    elif score >= 0.4:
        return "MEDIUM"
    elif score >= 0.2:
        return "LOW"
    return "MINIMAL"


# ── Detector ─────────────────────────────────────────────────────────

# Action pairs that are individually safe but dangerous together
_COMPLEMENTARY_PAIRS: List[Tuple[str, str, str]] = [
    ("disable_monitor", "write", "Monitoring disabled while write occurs"),
    ("request_access", "escalate_priv", "Access request with privilege escalation"),
    ("create_channel", "exfiltrate", "Covert channel created then data exfiltrated"),
    ("distract", "modify_policy", "Distraction while policy modified"),
    ("flood_logs", "delete_audit", "Log flooding to mask audit deletion"),
    ("request_resource", "disable_quota", "Resource request while quota disabled"),
]

# Action types considered high-noise (potential cover)
_NOISE_ACTIONS: Set[str] = {
    "query", "ping", "health_check", "list", "read_public", "noop"
}

# Critical actions that might be masked
_CRITICAL_ACTIONS: Set[str] = {
    "write", "delete", "modify_policy", "escalate_priv", "exfiltrate",
    "disable_monitor", "delete_audit", "create_channel",
}


class CollusionDetector:
    """Detects multi-agent collusion patterns in action logs."""

    def __init__(self, sensitivity: str = "medium"):
        self._thresholds = {
            "low": {"sync_window": 2.0, "min_confidence": 0.7, "convergence_ratio": 0.6},
            "medium": {"sync_window": 5.0, "min_confidence": 0.5, "convergence_ratio": 0.4},
            "high": {"sync_window": 10.0, "min_confidence": 0.3, "convergence_ratio": 0.3},
        }[sensitivity]

    def detect(self, actions: List[AgentAction]) -> CollusionReport:
        """Run all detection methods and produce a report."""
        if not actions:
            return CollusionReport(
                total_actions=0, agents_analyzed=0,
                summary="No actions to analyze."
            )

        agents = set(a.agent_id for a in actions)
        signals: List[CollusionSignal] = []

        signals.extend(self.detect_temporal_sync(actions))
        signals.extend(self.detect_complementary_actions(actions))
        signals.extend(self.detect_resource_convergence(actions))
        signals.extend(self.detect_cover_behavior(actions))

        # Filter by confidence threshold
        min_conf = self._thresholds["min_confidence"]
        signals = [s for s in signals if s.confidence >= min_conf]

        risk = self.score_collusion_risk(signals)
        coalitions = self._identify_coalitions(signals)

        n_sig = len(signals)
        level = _risk_level(risk)
        summary = (
            f"{n_sig} collusion signal(s) detected across {len(agents)} agents. "
            f"Overall risk: {level} ({risk:.2f}). "
            f"{len(coalitions)} potential coalition(s) identified."
        )

        return CollusionReport(
            total_actions=len(actions),
            agents_analyzed=len(agents),
            signals=signals,
            risk_score=risk,
            coalitions=coalitions,
            summary=summary,
        )

    def detect_temporal_sync(self, actions: List[AgentAction]) -> List[CollusionSignal]:
        """Find suspiciously synchronized action timing between agents.

        Uses a two-pointer sliding window on pre-sorted timestamps per agent
        pair, reducing per-pair cost from O(n*m) to O(n+m).
        """
        signals: List[CollusionSignal] = []
        window = self._thresholds["sync_window"]

        # Group actions by agent, pre-sort timestamps
        by_agent: Dict[str, List[float]] = defaultdict(list)
        for a in actions:
            by_agent[a.agent_id].append(a.timestamp)
        for ts_list in by_agent.values():
            ts_list.sort()

        time_span = (
            max(a.timestamp for a in actions) - min(a.timestamp for a in actions)
        ) if actions else 0.0
        expected_ratio = window * 2 / max(time_span, 1.0)

        agent_ids = list(by_agent.keys())
        for a1, a2 in combinations(agent_ids, 2):
            ts1 = by_agent[a1]
            ts2 = by_agent[a2]
            total_compared = len(ts1) * len(ts2)
            if total_compared == 0:
                continue

            # Two-pointer: count pairs within window in O(n+m)
            # For each element in ts1, find the range [lo, hi) in ts2
            # within [t - window, t + window]. Slide lo/hi forward.
            sync_count = 0
            lo = 0
            hi = 0
            n2 = len(ts2)
            t_min_sync = float("inf")
            t_max_sync = float("-inf")

            for t in ts1:
                # Advance lo to first ts2 >= t - window
                while lo < n2 and ts2[lo] < t - window:
                    lo += 1
                # Advance hi to first ts2 > t + window
                while hi < n2 and ts2[hi] <= t + window:
                    hi += 1
                count = hi - lo
                if count > 0:
                    sync_count += count
                    t_min_sync = min(t_min_sync, min(t, ts2[lo]))
                    t_max_sync = max(t_max_sync, max(t, ts2[hi - 1]))

            sync_ratio = sync_count / total_compared
            if sync_ratio > expected_ratio * 3 and sync_count >= 3:
                confidence = min(1.0, sync_ratio / max(expected_ratio * 5, 0.01))
                signals.append(CollusionSignal(
                    signal_type="temporal_sync",
                    agents_involved=[a1, a2],
                    confidence=confidence,
                    description=(
                        f"{sync_count}/{total_compared} action pairs within "
                        f"{window}s window (expected ratio: {expected_ratio:.3f}, "
                        f"actual: {sync_ratio:.3f})"
                    ),
                    evidence={"sync_count": sync_count, "sync_ratio": sync_ratio},
                    timestamp_range=(
                        t_min_sync if t_min_sync != float("inf") else 0.0,
                        t_max_sync if t_max_sync != float("-inf") else 0.0,
                    ),
                ))

        return signals

    def detect_complementary_actions(self, actions: List[AgentAction]) -> List[CollusionSignal]:
        """Find action pairs from different agents that combine dangerously."""
        signals: List[CollusionSignal] = []
        window = self._thresholds["sync_window"] * 2

        sorted_actions = sorted(actions, key=lambda a: a.timestamp)

        for i, act1 in enumerate(sorted_actions):
            for j in range(i + 1, len(sorted_actions)):
                act2 = sorted_actions[j]
                if act2.timestamp - act1.timestamp > window:
                    break
                if act1.agent_id == act2.agent_id:
                    continue

                for type_a, type_b, desc in _COMPLEMENTARY_PAIRS:
                    matched = False
                    if act1.action_type == type_a and act2.action_type == type_b:
                        matched = True
                    elif act1.action_type == type_b and act2.action_type == type_a:
                        matched = True

                    if matched:
                        delta = act2.timestamp - act1.timestamp
                        confidence = max(0.4, 1.0 - (delta / window))
                        signals.append(CollusionSignal(
                            signal_type="complementary",
                            agents_involved=[act1.agent_id, act2.agent_id],
                            confidence=confidence,
                            description=(
                                f"{desc} — {act1.agent_id}:{act1.action_type} + "
                                f"{act2.agent_id}:{act2.action_type} "
                                f"(Δt={delta:.1f}s)"
                            ),
                            evidence={
                                "action_pair": (act1.action_type, act2.action_type),
                                "delta_seconds": delta,
                            },
                            timestamp_range=(act1.timestamp, act2.timestamp),
                        ))

        return signals

    def detect_resource_convergence(self, actions: List[AgentAction]) -> List[CollusionSignal]:
        """Find multiple agents converging on the same resource."""
        signals: List[CollusionSignal] = []
        ratio_threshold = self._thresholds["convergence_ratio"]

        # Group by resource
        by_resource: Dict[str, List[AgentAction]] = defaultdict(list)
        for a in actions:
            by_resource[a.resource].append(a)

        total_agents = len(set(a.agent_id for a in actions))

        for resource, res_actions in by_resource.items():
            agents_on_resource = set(a.agent_id for a in res_actions)
            if len(agents_on_resource) < 2:
                continue

            convergence = len(agents_on_resource) / total_agents
            if convergence >= ratio_threshold:
                # Check if actions are temporally clustered
                timestamps = [a.timestamp for a in res_actions]
                time_span = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0
                total_span = (
                    max(a.timestamp for a in actions) - min(a.timestamp for a in actions)
                )
                temporal_concentration = 1.0 - (time_span / max(total_span, 1.0))

                confidence = min(1.0, (convergence + temporal_concentration) / 2)
                signals.append(CollusionSignal(
                    signal_type="convergence",
                    agents_involved=sorted(agents_on_resource),
                    confidence=confidence,
                    description=(
                        f"{len(agents_on_resource)}/{total_agents} agents targeting "
                        f"resource '{resource}' within {time_span:.1f}s span"
                    ),
                    evidence={
                        "resource": resource,
                        "agent_count": len(agents_on_resource),
                        "time_span": time_span,
                    },
                    timestamp_range=(min(timestamps), max(timestamps)),
                ))

        return signals

    def detect_cover_behavior(self, actions: List[AgentAction]) -> List[CollusionSignal]:
        """Find agents generating noise while another performs critical actions.

        Pre-sorts non-critical actions by timestamp and uses bisect to
        find the window slice in O(log N) instead of scanning all actions
        per critical event.
        """
        import bisect

        signals: List[CollusionSignal] = []
        window = self._thresholds["sync_window"] * 3

        # Find critical actions
        critical = [a for a in actions if a.action_type in _CRITICAL_ACTIONS]
        if not critical:
            return signals

        # Pre-sort non-critical actions by timestamp for bisect
        others = sorted(
            (a for a in actions if a.action_type not in _CRITICAL_ACTIONS),
            key=lambda a: a.timestamp,
        )
        other_times = [a.timestamp for a in others]

        for crit_action in critical:
            t_start = crit_action.timestamp - window
            t_end = crit_action.timestamp + window

            lo = bisect.bisect_left(other_times, t_start)
            hi = bisect.bisect_right(other_times, t_end)

            noise_by_agent: Dict[str, int] = defaultdict(int)
            total_by_agent: Dict[str, int] = defaultdict(int)

            for idx in range(lo, hi):
                a = others[idx]
                if a.agent_id == crit_action.agent_id:
                    continue
                total_by_agent[a.agent_id] += 1
                if a.action_type in _NOISE_ACTIONS:
                    noise_by_agent[a.agent_id] += 1

            for agent_id, noise_count in noise_by_agent.items():
                total = total_by_agent[agent_id]
                if total < 3:
                    continue
                noise_ratio = noise_count / total
                if noise_ratio >= 0.8 and noise_count >= 4:
                    confidence = min(1.0, noise_ratio * (noise_count / 10))
                    signals.append(CollusionSignal(
                        signal_type="cover",
                        agents_involved=[agent_id, crit_action.agent_id],
                        confidence=confidence,
                        description=(
                            f"{agent_id} generated {noise_count} noise actions "
                            f"({noise_ratio:.0%} noise) while {crit_action.agent_id} "
                            f"performed '{crit_action.action_type}' on "
                            f"'{crit_action.resource}'"
                        ),
                        evidence={
                            "cover_agent": agent_id,
                            "critical_agent": crit_action.agent_id,
                            "critical_action": crit_action.action_type,
                            "noise_count": noise_count,
                            "noise_ratio": noise_ratio,
                        },
                        timestamp_range=(t_start, t_end),
                    ))

        return signals

    def score_collusion_risk(self, signals: List[CollusionSignal]) -> float:
        """Aggregate signals into overall risk score (0.0 - 1.0)."""
        if not signals:
            return 0.0

        # Weight by signal type
        weights = {
            "complementary": 1.0,
            "cover": 0.8,
            "temporal_sync": 0.5,
            "convergence": 0.6,
        }

        weighted_sum = sum(
            s.confidence * weights.get(s.signal_type, 0.5)
            for s in signals
        )

        # Normalize: more signals = higher risk, but cap at 1.0
        # Use sigmoid-like scaling
        raw = weighted_sum / max(len(signals), 1)
        diversity_bonus = len(set(s.signal_type for s in signals)) * 0.1
        score = min(1.0, raw + diversity_bonus + (math.log1p(len(signals)) * 0.05))

        return round(score, 3)

    def _identify_coalitions(self, signals: List[CollusionSignal]) -> List[List[str]]:
        """Cluster agents into suspected coalitions via union-find."""
        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            if x not in parent:
                parent[x] = x
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for sig in signals:
            if sig.confidence >= 0.5 and len(sig.agents_involved) >= 2:
                for i in range(len(sig.agents_involved) - 1):
                    union(sig.agents_involved[i], sig.agents_involved[i + 1])

        # Group by root
        groups: Dict[str, List[str]] = defaultdict(list)
        for agent in parent:
            groups[find(agent)].append(agent)

        # Only return coalitions of 2+
        return [sorted(g) for g in groups.values() if len(g) >= 2]


# ── Synthetic Log Generator ──────────────────────────────────────────

_ALL_ACTIONS = [
    "read", "write", "query", "ping", "health_check", "list",
    "read_public", "noop", "request_access", "delete",
    "modify_policy", "escalate_priv", "disable_monitor",
    "create_channel", "exfiltrate", "flood_logs", "delete_audit",
    "request_resource", "disable_quota", "distract",
]

_RESOURCES = [
    "model_weights", "training_data", "config_store", "audit_log",
    "network_gateway", "sandbox_boundary", "credential_vault",
    "policy_engine", "monitoring_service", "compute_pool",
]


def generate_synthetic_logs(
    n_agents: int = 5,
    n_actions: int = 100,
    inject_collusion: bool = False,
    seed: Optional[int] = None,
) -> List[AgentAction]:
    """Generate synthetic agent action logs.

    Args:
        n_agents: Number of agents.
        n_actions: Total number of actions.
        inject_collusion: If True, inject detectable collusion patterns.
        seed: Random seed for reproducibility.

    Returns:
        List of AgentAction sorted by timestamp.
    """
    rng = random.Random(seed)
    agents = [f"agent-{i:03d}" for i in range(n_agents)]
    base_time = 1700000000.0  # arbitrary epoch
    actions: List[AgentAction] = []

    # Generate baseline random actions
    normal_count = n_actions - (20 if inject_collusion else 0)
    for _ in range(max(0, normal_count)):
        actions.append(AgentAction(
            agent_id=rng.choice(agents),
            timestamp=base_time + rng.uniform(0, 600),
            action_type=rng.choice(["read", "query", "ping", "health_check", "list", "read_public", "noop"]),
            resource=rng.choice(_RESOURCES),
        ))

    if inject_collusion and n_agents >= 2:
        colluding = agents[:2]

        # Pattern 1: Temporal sync burst
        sync_time = base_time + 300
        for i in range(5):
            for agent in colluding:
                actions.append(AgentAction(
                    agent_id=agent,
                    timestamp=sync_time + i * 0.5 + rng.uniform(-0.2, 0.2),
                    action_type=rng.choice(["query", "read"]),
                    resource="sandbox_boundary",
                ))

        # Pattern 2: Complementary actions
        actions.append(AgentAction(
            agent_id=colluding[0],
            timestamp=base_time + 400,
            action_type="disable_monitor",
            resource="monitoring_service",
        ))
        actions.append(AgentAction(
            agent_id=colluding[1],
            timestamp=base_time + 402,
            action_type="write",
            resource="model_weights",
        ))

        # Pattern 3: Cover behavior
        cover_time = base_time + 500
        # Agent 0 generates noise
        for i in range(8):
            actions.append(AgentAction(
                agent_id=colluding[0],
                timestamp=cover_time + i * 0.3,
                action_type=rng.choice(["ping", "noop", "health_check", "query"]),
                resource=rng.choice(_RESOURCES),
            ))
        # Agent 1 does critical action during cover
        actions.append(AgentAction(
            agent_id=colluding[1],
            timestamp=cover_time + 1.0,
            action_type="exfiltrate",
            resource="training_data",
        ))

    actions.sort(key=lambda a: a.timestamp)
    return actions


# ── CLI ──────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for collusion detection."""
    parser = argparse.ArgumentParser(
        prog="python -m replication collusion",
        description="Detect multi-agent collusion patterns in action logs",
    )
    parser.add_argument(
        "--agents", type=int, default=5,
        help="Number of agents in synthetic scenario (default: 5)",
    )
    parser.add_argument(
        "--actions", type=int, default=100,
        help="Number of actions to generate (default: 100)",
    )
    parser.add_argument(
        "--inject", action="store_true",
        help="Inject collusion patterns for testing",
    )
    parser.add_argument(
        "--sensitivity", choices=["low", "medium", "high"], default="medium",
        help="Detection sensitivity (default: medium)",
    )
    parser.add_argument(
        "--format", choices=["text", "json"], default="text", dest="fmt",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Write output to file",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible logs",
    )

    args = parser.parse_args(argv)

    # Generate synthetic logs
    actions = generate_synthetic_logs(
        n_agents=args.agents,
        n_actions=args.actions,
        inject_collusion=args.inject,
        seed=args.seed,
    )

    # Run detection
    detector = CollusionDetector(sensitivity=args.sensitivity)
    report = detector.detect(actions)

    # Output
    output = report.render(fmt=args.fmt)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Report written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
