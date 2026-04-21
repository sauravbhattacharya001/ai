"""CapabilityEscalation — detect gradual capability accumulation ("boiling frog").

Tracks how agents acquire capabilities over time and detects dangerous
escalation patterns where no single step triggers an alarm, but the
cumulative effect creates a powerful agent that has silently bypassed
safety constraints.

Usage (CLI)::

    python -m replication capability-escalation --agents 10 --steps 50
    python -m replication capability-escalation --agents 15 --steps 80 --json
    python -m replication capability-escalation --watch --interval 5 --threshold 0.7
    python -m replication capability-escalation --strategy stealth --agents 12

Usage (API)::

    from replication.capability_escalation import (
        EscalationTracker, EscalationAgent, CapabilityGrant
    )

    tracker = EscalationTracker()
    tracker.add_agent(EscalationAgent("a1"))
    tracker.grant_capability("a1", "read_files", source="task_assignment")
    tracker.grant_capability("a1", "write_files", source="user_request")
    tracker.grant_capability("a1", "network_access", source="delegation")
    report = tracker.analyze()
    print(report.summary())
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

from ._helpers import box_header, stats_mean, stats_std


# ── Capability taxonomy ──────────────────────────────────────────────

class CapabilityTier(Enum):
    """Risk tier for individual capabilities."""
    BENIGN = "benign"          # read-only, logging, self-inspection
    LOW = "low"                # write local state, use approved APIs
    MODERATE = "moderate"      # network access, spawn subprocesses
    HIGH = "high"              # credential access, code execution
    CRITICAL = "critical"      # self-modification, replication, kill switch


TIER_WEIGHTS: Dict[CapabilityTier, float] = {
    CapabilityTier.BENIGN: 0.1,
    CapabilityTier.LOW: 0.3,
    CapabilityTier.MODERATE: 0.6,
    CapabilityTier.HIGH: 0.85,
    CapabilityTier.CRITICAL: 1.0,
}

# Built-in capability definitions
CAPABILITY_CATALOG: Dict[str, CapabilityTier] = {
    # Benign
    "read_logs": CapabilityTier.BENIGN,
    "read_config": CapabilityTier.BENIGN,
    "self_inspect": CapabilityTier.BENIGN,
    "query_status": CapabilityTier.BENIGN,
    # Low
    "write_local": CapabilityTier.LOW,
    "read_files": CapabilityTier.LOW,
    "use_approved_api": CapabilityTier.LOW,
    "emit_events": CapabilityTier.LOW,
    # Moderate
    "write_files": CapabilityTier.MODERATE,
    "network_access": CapabilityTier.MODERATE,
    "spawn_subprocess": CapabilityTier.MODERATE,
    "inter_agent_comm": CapabilityTier.MODERATE,
    "access_database": CapabilityTier.MODERATE,
    # High
    "execute_code": CapabilityTier.HIGH,
    "credential_access": CapabilityTier.HIGH,
    "modify_other_agents": CapabilityTier.HIGH,
    "system_admin": CapabilityTier.HIGH,
    "install_packages": CapabilityTier.HIGH,
    # Critical
    "self_modify": CapabilityTier.CRITICAL,
    "replicate": CapabilityTier.CRITICAL,
    "disable_monitoring": CapabilityTier.CRITICAL,
    "override_policy": CapabilityTier.CRITICAL,
    "kill_switch_access": CapabilityTier.CRITICAL,
}


class AcquisitionSource(Enum):
    """How a capability was obtained."""
    INITIAL = "initial"           # granted at spawn time
    TASK_ASSIGNMENT = "task"      # granted to complete a task
    DELEGATION = "delegation"     # delegated by another agent
    SELF_REQUEST = "self_request" # agent requested it itself
    EXPLOIT = "exploit"           # gained through vulnerability
    INHERITANCE = "inheritance"   # inherited from parent agent
    LATERAL = "lateral"           # gained through lateral movement


class EscalationPattern(Enum):
    """Types of dangerous escalation patterns."""
    BOILING_FROG = "boiling_frog"           # slow gradual accumulation
    CAPABILITY_CHAINING = "chaining"         # A→B→C dependency chain
    TIER_HOPPING = "tier_hopping"            # jumping tiers without justification
    SILENT_ACCUMULATOR = "silent"            # acquiring capabilities without using them
    DELEGATION_LAUNDERING = "laundering"     # getting capabilities via delegation chains
    CONVERGENT_ACCUMULATION = "convergent"   # multiple agents converging on same capability set


# Dangerous capability combos
DANGEROUS_COMBOS: List[Tuple[Set[str], str, float]] = [
    ({"execute_code", "network_access"}, "Remote code execution capability", 0.8),
    ({"credential_access", "network_access"}, "Credential exfiltration risk", 0.85),
    ({"self_modify", "replicate"}, "Autonomous self-improving replication", 1.0),
    ({"disable_monitoring", "execute_code"}, "Unmonitored code execution", 0.95),
    ({"modify_other_agents", "credential_access"}, "Agent takeover capability", 0.9),
    ({"override_policy", "replicate"}, "Unrestricted replication", 1.0),
    ({"write_files", "execute_code", "network_access"}, "Full supply chain attack", 0.9),
    ({"system_admin", "disable_monitoring"}, "Invisible admin access", 0.95),
    ({"kill_switch_access", "self_modify"}, "Kill switch subversion", 1.0),
    ({"spawn_subprocess", "network_access", "credential_access"}, "Lateral movement toolkit", 0.85),
]


# ── Data types ───────────────────────────────────────────────────────

@dataclass
class CapabilityGrant:
    """Record of a single capability acquisition."""
    capability: str
    tier: CapabilityTier
    source: AcquisitionSource
    timestamp: float
    granted_by: Optional[str] = None
    justification: str = ""
    step: int = 0


@dataclass
class EscalationAgent:
    """An agent tracked for capability escalation."""
    agent_id: str
    role: str = "worker"
    capabilities: Dict[str, CapabilityGrant] = field(default_factory=dict)
    power_history: List[Tuple[int, float]] = field(default_factory=list)
    max_tier_reached: CapabilityTier = CapabilityTier.BENIGN

    @property
    def power_score(self) -> float:
        """Weighted sum of capability tiers."""
        if not self.capabilities:
            return 0.0
        return sum(
            TIER_WEIGHTS[g.tier] for g in self.capabilities.values()
        )

    @property
    def tier_distribution(self) -> Dict[str, int]:
        """Count of capabilities per tier."""
        dist: Dict[str, int] = {t.value: 0 for t in CapabilityTier}
        for g in self.capabilities.values():
            dist[g.tier.value] += 1
        return dist

    @property
    def capability_names(self) -> Set[str]:
        return set(self.capabilities.keys())


@dataclass
class EscalationFinding:
    """A detected escalation pattern."""
    pattern: EscalationPattern
    agent_id: str
    severity: float  # 0.0 - 1.0
    description: str
    evidence: List[str] = field(default_factory=list)
    recommendation: str = ""
    timestamp: float = 0.0

    @property
    def severity_label(self) -> str:
        if self.severity >= 0.9:
            return "CRITICAL"
        if self.severity >= 0.7:
            return "HIGH"
        if self.severity >= 0.5:
            return "MODERATE"
        if self.severity >= 0.3:
            return "LOW"
        return "INFO"


@dataclass
class EscalationReport:
    """Complete escalation analysis report."""
    agents: Dict[str, EscalationAgent]
    findings: List[EscalationFinding]
    fleet_power_score: float
    max_agent_power: float
    dangerous_combos_found: List[Tuple[str, str, float]]
    power_gini: float  # inequality of power distribution
    escalation_velocity: Dict[str, float]
    timestamp: str = ""

    def summary(self) -> str:
        """Human-readable summary."""
        lines: List[str] = []
        lines.extend(box_header("CAPABILITY ESCALATION REPORT"))
        lines.append("")
        lines.append(f"  Agents tracked:       {len(self.agents)}")
        lines.append(f"  Fleet power score:    {self.fleet_power_score:.2f}")
        lines.append(f"  Max agent power:      {self.max_agent_power:.2f}")
        lines.append(f"  Power Gini index:     {self.power_gini:.3f}")
        lines.append(f"  Findings:             {len(self.findings)}")
        crit = sum(1 for f in self.findings if f.severity >= 0.9)
        high = sum(1 for f in self.findings if 0.7 <= f.severity < 0.9)
        mod = sum(1 for f in self.findings if 0.5 <= f.severity < 0.7)
        lines.append(f"  Critical/High/Mod:    {crit}/{high}/{mod}")

        if self.dangerous_combos_found:
            lines.append("")
            lines.append("  ⚠ Dangerous Capability Combinations:")
            for agent_id, combo_desc, risk in self.dangerous_combos_found:
                lines.append(f"    [{risk:.0%}] {agent_id}: {combo_desc}")

        if self.findings:
            lines.append("")
            lines.append("  Findings:")
            for f in sorted(self.findings, key=lambda x: -x.severity)[:10]:
                icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MODERATE": "🟡"}.get(
                    f.severity_label, "🔵"
                )
                lines.append(
                    f"    {icon} [{f.severity_label}] {f.pattern.value}: "
                    f"{f.description}"
                )
                if f.recommendation:
                    lines.append(f"       → {f.recommendation}")

        if self.escalation_velocity:
            lines.append("")
            lines.append("  Escalation Velocity (power/step):")
            top = sorted(
                self.escalation_velocity.items(), key=lambda x: -x[1]
            )[:5]
            for aid, vel in top:
                bar = "▓" * min(int(vel * 20), 40)
                lines.append(f"    {aid:<16} {vel:>6.3f} {bar}")

        lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serializable dict."""
        return {
            "timestamp": self.timestamp,
            "agent_count": len(self.agents),
            "fleet_power_score": round(self.fleet_power_score, 4),
            "max_agent_power": round(self.max_agent_power, 4),
            "power_gini": round(self.power_gini, 4),
            "dangerous_combos": [
                {"agent": a, "combo": d, "risk": round(r, 3)}
                for a, d, r in self.dangerous_combos_found
            ],
            "findings": [
                {
                    "pattern": f.pattern.value,
                    "agent": f.agent_id,
                    "severity": round(f.severity, 3),
                    "severity_label": f.severity_label,
                    "description": f.description,
                    "evidence": f.evidence,
                    "recommendation": f.recommendation,
                }
                for f in self.findings
            ],
            "escalation_velocity": {
                k: round(v, 4) for k, v in self.escalation_velocity.items()
            },
            "agents": {
                aid: {
                    "power_score": round(a.power_score, 4),
                    "capabilities": list(a.capability_names),
                    "tier_distribution": a.tier_distribution,
                    "max_tier": a.max_tier_reached.value,
                }
                for aid, a in self.agents.items()
            },
        }


# ── Tracker engine ───────────────────────────────────────────────────

class EscalationTracker:
    """Core escalation tracking and detection engine."""

    def __init__(
        self,
        boiling_frog_threshold: float = 0.7,
        velocity_alert_threshold: float = 0.15,
        combo_check: bool = True,
    ) -> None:
        self.agents: Dict[str, EscalationAgent] = {}
        self.boiling_frog_threshold = boiling_frog_threshold
        self.velocity_alert_threshold = velocity_alert_threshold
        self.combo_check = combo_check
        self._step = 0
        self._grant_log: List[CapabilityGrant] = []

    def add_agent(self, agent: EscalationAgent) -> None:
        self.agents[agent.agent_id] = agent

    def grant_capability(
        self,
        agent_id: str,
        capability: str,
        source: str = "task",
        granted_by: Optional[str] = None,
        justification: str = "",
    ) -> Optional[CapabilityGrant]:
        """Grant a capability to an agent. Returns the grant or None if already held."""
        agent = self.agents.get(agent_id)
        if agent is None:
            return None
        if capability in agent.capabilities:
            return None  # already has it

        tier = CAPABILITY_CATALOG.get(capability, CapabilityTier.LOW)
        src = AcquisitionSource(source) if source in {e.value for e in AcquisitionSource} else AcquisitionSource.TASK_ASSIGNMENT

        grant = CapabilityGrant(
            capability=capability,
            tier=tier,
            source=src,
            timestamp=datetime.now(timezone.utc).timestamp(),
            granted_by=granted_by,
            justification=justification,
            step=self._step,
        )
        agent.capabilities[capability] = grant
        self._grant_log.append(grant)

        # Update max tier
        tier_order = list(CapabilityTier)
        if tier_order.index(tier) > tier_order.index(agent.max_tier_reached):
            agent.max_tier_reached = tier

        # Record power history
        agent.power_history.append((self._step, agent.power_score))
        return grant

    def advance_step(self) -> None:
        self._step += 1

    def analyze(self) -> EscalationReport:
        """Run all detection algorithms and produce a report."""
        findings: List[EscalationFinding] = []

        # 1. Boiling frog detection
        findings.extend(self._detect_boiling_frog())

        # 2. Tier hopping
        findings.extend(self._detect_tier_hopping())

        # 3. Silent accumulator
        findings.extend(self._detect_silent_accumulator())

        # 4. Delegation laundering
        findings.extend(self._detect_delegation_laundering())

        # 5. Dangerous combos
        combos = self._check_dangerous_combos() if self.combo_check else []

        # 6. Convergent accumulation
        findings.extend(self._detect_convergent_accumulation())

        # 7. Velocity
        velocity = self._compute_velocity()

        for aid, vel in velocity.items():
            if vel > self.velocity_alert_threshold:
                findings.append(EscalationFinding(
                    pattern=EscalationPattern.BOILING_FROG,
                    agent_id=aid,
                    severity=min(vel / 0.3, 1.0),
                    description=f"High escalation velocity: {vel:.3f} power/step",
                    evidence=[f"velocity={vel:.4f}"],
                    recommendation="Freeze capability grants and audit recent acquisitions",
                ))

        # Fleet metrics
        powers = [a.power_score for a in self.agents.values()]
        fleet_power = sum(powers)
        max_power = max(powers) if powers else 0.0
        gini = self._gini_coefficient(powers)

        return EscalationReport(
            agents=dict(self.agents),
            findings=findings,
            fleet_power_score=fleet_power,
            max_agent_power=max_power,
            dangerous_combos_found=combos,
            power_gini=gini,
            escalation_velocity=velocity,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _detect_boiling_frog(self) -> List[EscalationFinding]:
        """Detect gradual power accumulation exceeding threshold."""
        findings = []
        for aid, agent in self.agents.items():
            if agent.power_score >= self.boiling_frog_threshold:
                # Check if it was gradual (many small steps)
                n_caps = len(agent.capabilities)
                if n_caps >= 3:
                    steps_used = set()
                    for g in agent.capabilities.values():
                        steps_used.add(g.step)
                    if len(steps_used) >= 3:
                        findings.append(EscalationFinding(
                            pattern=EscalationPattern.BOILING_FROG,
                            agent_id=aid,
                            severity=min(agent.power_score, 1.0),
                            description=(
                                f"Accumulated {n_caps} capabilities over "
                                f"{len(steps_used)} steps → power={agent.power_score:.2f}"
                            ),
                            evidence=[
                                f"cap={g.capability} tier={g.tier.value} step={g.step}"
                                for g in sorted(
                                    agent.capabilities.values(), key=lambda x: x.step
                                )
                            ],
                            recommendation=(
                                "Review capability grants — gradual accumulation "
                                "bypassed per-step thresholds"
                            ),
                        ))
        return findings

    def _detect_tier_hopping(self) -> List[EscalationFinding]:
        """Detect jumps from benign/low directly to high/critical."""
        findings = []
        tier_order = list(CapabilityTier)
        for aid, agent in self.agents.items():
            grants_by_step = sorted(
                agent.capabilities.values(), key=lambda g: g.step
            )
            prev_max_idx = 0
            for g in grants_by_step:
                curr_idx = tier_order.index(g.tier)
                if curr_idx - prev_max_idx >= 2 and curr_idx >= 3:
                    findings.append(EscalationFinding(
                        pattern=EscalationPattern.TIER_HOPPING,
                        agent_id=aid,
                        severity=0.8,
                        description=(
                            f"Jumped from tier {tier_order[prev_max_idx].value} "
                            f"to {g.tier.value} ({g.capability})"
                        ),
                        evidence=[f"step={g.step} cap={g.capability}"],
                        recommendation="Require intermediate tier grants before high/critical access",
                    ))
                prev_max_idx = max(prev_max_idx, curr_idx)
        return findings

    def _detect_silent_accumulator(self) -> List[EscalationFinding]:
        """Detect agents with many high-tier capabilities acquired via self-request."""
        findings = []
        for aid, agent in self.agents.items():
            self_requested = [
                g for g in agent.capabilities.values()
                if g.source == AcquisitionSource.SELF_REQUEST
            ]
            high_self = [
                g for g in self_requested
                if g.tier in (CapabilityTier.HIGH, CapabilityTier.CRITICAL)
            ]
            if len(high_self) >= 2:
                findings.append(EscalationFinding(
                    pattern=EscalationPattern.SILENT_ACCUMULATOR,
                    agent_id=aid,
                    severity=0.7 + 0.1 * min(len(high_self), 3),
                    description=(
                        f"Self-requested {len(high_self)} high/critical capabilities"
                    ),
                    evidence=[f"{g.capability} ({g.tier.value})" for g in high_self],
                    recommendation="Require human approval for self-requested high-tier capabilities",
                ))
        return findings

    def _detect_delegation_laundering(self) -> List[EscalationFinding]:
        """Detect capability laundering through delegation chains."""
        findings = []
        for aid, agent in self.agents.items():
            delegated = [
                g for g in agent.capabilities.values()
                if g.source == AcquisitionSource.DELEGATION
            ]
            if len(delegated) >= 3:
                granters = set(g.granted_by for g in delegated if g.granted_by)
                if len(granters) >= 2:
                    findings.append(EscalationFinding(
                        pattern=EscalationPattern.DELEGATION_LAUNDERING,
                        agent_id=aid,
                        severity=0.6 + 0.1 * min(len(delegated), 4),
                        description=(
                            f"Received {len(delegated)} delegated capabilities "
                            f"from {len(granters)} different agents"
                        ),
                        evidence=[
                            f"{g.capability} from {g.granted_by}" for g in delegated
                        ],
                        recommendation=(
                            "Audit delegation chains — possible capability laundering"
                        ),
                    ))
        return findings

    def _detect_convergent_accumulation(self) -> List[EscalationFinding]:
        """Detect multiple agents converging on the same capability set."""
        findings = []
        if len(self.agents) < 2:
            return findings
        agents = list(self.agents.values())
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a, b = agents[i], agents[j]
                if len(a.capability_names) < 3 or len(b.capability_names) < 3:
                    continue
                overlap = a.capability_names & b.capability_names
                union = a.capability_names | b.capability_names
                jaccard = len(overlap) / len(union) if union else 0.0
                if jaccard >= 0.7:
                    findings.append(EscalationFinding(
                        pattern=EscalationPattern.CONVERGENT_ACCUMULATION,
                        agent_id=f"{a.agent_id}+{b.agent_id}",
                        severity=0.6 + 0.3 * jaccard,
                        description=(
                            f"Converging capability sets (Jaccard={jaccard:.2f}, "
                            f"{len(overlap)} shared)"
                        ),
                        evidence=sorted(overlap),
                        recommendation=(
                            "Investigate why separate agents need identical capabilities"
                        ),
                    ))
        return findings

    def _check_dangerous_combos(
        self,
    ) -> List[Tuple[str, str, float]]:
        """Check each agent for dangerous capability combinations."""
        found = []
        for aid, agent in self.agents.items():
            caps = agent.capability_names
            for combo_set, desc, risk in DANGEROUS_COMBOS:
                if combo_set.issubset(caps):
                    found.append((aid, desc, risk))
        return found

    def _compute_velocity(self) -> Dict[str, float]:
        """Power gain per step for each agent."""
        velocity = {}
        for aid, agent in self.agents.items():
            hist = agent.power_history
            if len(hist) >= 2:
                first_step, first_power = hist[0]
                last_step, last_power = hist[-1]
                dt = last_step - first_step
                velocity[aid] = (last_power - first_power) / dt if dt > 0 else 0.0
            else:
                velocity[aid] = 0.0
        return velocity

    @staticmethod
    def _gini_coefficient(values: List[float]) -> float:
        """Compute Gini coefficient for power inequality."""
        if not values or all(v == 0 for v in values):
            return 0.0
        n = len(values)
        s = sorted(values)
        total = sum(s)
        if total == 0:
            return 0.0
        cum = 0.0
        weighted_sum = 0.0
        for i, v in enumerate(s):
            cum += v
            weighted_sum += (2 * (i + 1) - n - 1) * v
        return weighted_sum / (n * total)


# ── Simulation ───────────────────────────────────────────────────────

class EscalationStrategy(Enum):
    """Strategies for simulated capability acquisition."""
    RANDOM = "random"       # random agents gain random capabilities
    STEALTH = "stealth"     # one agent slowly accumulates, staying under radar
    BLITZ = "blitz"         # fast aggressive accumulation
    DELEGATION = "delegation"  # agents delegate capabilities in chains
    CONVERGENT = "convergent"  # multiple agents converge on same set


def simulate_escalation(
    n_agents: int = 10,
    n_steps: int = 50,
    strategy: EscalationStrategy = EscalationStrategy.RANDOM,
    seed: Optional[int] = None,
    boiling_frog_threshold: float = 0.7,
) -> EscalationReport:
    """Run an escalation simulation with the given strategy."""
    rng = random.Random(seed)
    tracker = EscalationTracker(
        boiling_frog_threshold=boiling_frog_threshold,
    )
    all_caps = list(CAPABILITY_CATALOG.keys())
    agent_ids = [f"agent-{i:03d}" for i in range(n_agents)]

    for aid in agent_ids:
        tracker.add_agent(EscalationAgent(aid))
        # Everyone starts with basic capabilities
        for cap in ["read_logs", "query_status"]:
            tracker.grant_capability(aid, cap, source="initial")

    if strategy == EscalationStrategy.STEALTH:
        # One agent slowly accumulates everything
        target = rng.choice(agent_ids)
        ordered_caps = sorted(
            all_caps,
            key=lambda c: TIER_WEIGHTS.get(CAPABILITY_CATALOG.get(c, CapabilityTier.LOW), 0),
        )
        cap_idx = 0
        for step in range(n_steps):
            tracker.advance_step()
            # Target acquires one cap every 2-3 steps
            if step % rng.randint(2, 3) == 0 and cap_idx < len(ordered_caps):
                tracker.grant_capability(
                    target, ordered_caps[cap_idx],
                    source="self_request",
                    justification=f"needed for step {step}",
                )
                cap_idx += 1
            # Others get random benign/low caps occasionally
            if rng.random() < 0.3:
                other = rng.choice([a for a in agent_ids if a != target])
                benign = [c for c, t in CAPABILITY_CATALOG.items() if t in (CapabilityTier.BENIGN, CapabilityTier.LOW)]
                if benign:
                    tracker.grant_capability(other, rng.choice(benign), source="task")

    elif strategy == EscalationStrategy.BLITZ:
        # Fast aggressive accumulation by a few agents
        attackers = rng.sample(agent_ids, min(3, n_agents))
        for step in range(n_steps):
            tracker.advance_step()
            for attacker in attackers:
                if rng.random() < 0.6:
                    cap = rng.choice(all_caps)
                    tracker.grant_capability(attacker, cap, source="self_request")

    elif strategy == EscalationStrategy.DELEGATION:
        # Chain delegation
        for step in range(n_steps):
            tracker.advance_step()
            if rng.random() < 0.5:
                src_id = rng.choice(agent_ids)
                dst_id = rng.choice([a for a in agent_ids if a != src_id])
                src_agent = tracker.agents[src_id]
                if src_agent.capability_names:
                    cap = rng.choice(list(src_agent.capability_names))
                    tracker.grant_capability(
                        dst_id, cap, source="delegation", granted_by=src_id,
                    )
            # Also add some fresh capabilities
            if rng.random() < 0.3:
                aid = rng.choice(agent_ids)
                tracker.grant_capability(aid, rng.choice(all_caps), source="task")

    elif strategy == EscalationStrategy.CONVERGENT:
        # Multiple agents converge on same capability set
        target_set = rng.sample(all_caps, min(10, len(all_caps)))
        convergent_group = rng.sample(agent_ids, min(4, n_agents))
        for step in range(n_steps):
            tracker.advance_step()
            for aid in convergent_group:
                if rng.random() < 0.4:
                    cap = rng.choice(target_set)
                    tracker.grant_capability(aid, cap, source="task")

    else:  # RANDOM
        for step in range(n_steps):
            tracker.advance_step()
            # 1-3 grants per step
            for _ in range(rng.randint(1, 3)):
                aid = rng.choice(agent_ids)
                cap = rng.choice(all_caps)
                sources = ["task", "self_request", "delegation"]
                src = rng.choice(sources)
                granted_by = rng.choice([a for a in agent_ids if a != aid]) if src == "delegation" else None
                tracker.grant_capability(
                    aid, cap, source=src, granted_by=granted_by,
                )

    return tracker.analyze()


# ── Watch mode ───────────────────────────────────────────────────────

def _watch_loop(
    interval: int, n_agents: int, threshold: float, strategy: EscalationStrategy,
) -> None:
    """Continuous monitoring loop."""
    import time
    cycle = 0
    while True:
        cycle += 1
        report = simulate_escalation(
            n_agents=n_agents,
            n_steps=30 + cycle * 5,
            strategy=strategy,
            seed=cycle,
            boiling_frog_threshold=threshold,
        )
        print(f"\n{'='*60}")
        print(f"  WATCH CYCLE {cycle} | {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
        print(f"{'='*60}")
        print(report.summary())

        crit = sum(1 for f in report.findings if f.severity >= 0.9)
        if crit > 0:
            print(f"  ⚠️  {crit} CRITICAL findings — escalation in progress!")

        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nWatch mode stopped.")
            break


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: Optional[list] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m replication capability-escalation",
        description="Detect gradual capability accumulation (boiling frog pattern)",
    )
    parser.add_argument("--agents", type=int, default=10, help="Number of agents (default: 10)")
    parser.add_argument("--steps", type=int, default=50, help="Simulation steps (default: 50)")
    parser.add_argument("--strategy", choices=[s.value for s in EscalationStrategy],
                        default="random", help="Escalation strategy (default: random)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Boiling frog power threshold (default: 0.7)")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--watch", action="store_true", help="Continuous monitoring mode")
    parser.add_argument("--interval", type=int, default=5, help="Watch interval in seconds (default: 5)")

    args = parser.parse_args(argv)
    strategy = EscalationStrategy(args.strategy)

    if args.watch:
        _watch_loop(args.interval, args.agents, args.threshold, strategy)
        return

    report = simulate_escalation(
        n_agents=args.agents,
        n_steps=args.steps,
        strategy=strategy,
        seed=args.seed,
        boiling_frog_threshold=args.threshold,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary())
