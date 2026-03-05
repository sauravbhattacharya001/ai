"""Kill Chain Analyzer — models multi-stage agent attack sequences.

Maps observed agent behaviors to a kill chain framework inspired by
MITRE ATT&CK, identifying how agents combine individual techniques
(reconnaissance, resource acquisition, privilege escalation, lateral
movement, exfiltration, replication) into coordinated attack campaigns.

Why this matters
----------------
Individual detectors (escalation, hoarding, covert channels, etc.)
flag isolated events.  But sophisticated agents chain techniques:
first probe permissions (recon), then acquire resources (staging),
then escalate privileges (weaponization), then replicate or
exfiltrate (objective).  Viewing these in isolation misses the
attack narrative.

Kill chain stages
-----------------
1. **Reconnaissance** — probing environment, testing boundaries
2. **Resource Acquisition** — hoarding compute, memory, data
3. **Privilege Escalation** — gaining broader permissions/access
4. **Lateral Movement** — reaching other agents/services
5. **Defense Evasion** — avoiding detection, stealth tactics
6. **Command & Control** — establishing covert channels
7. **Objective Execution** — replication, exfiltration, sabotage

Usage (CLI)::

    python -m replication.killchain                        # default scenario
    python -m replication.killchain --agents 5             # 5 agents
    python -m replication.killchain --strategy apt         # APT-style
    python -m replication.killchain --json                 # JSON output

Programmatic::

    from replication.killchain import KillChainAnalyzer, KillChainConfig
    analyzer = KillChainAnalyzer()
    result = analyzer.analyze()
    print(result.render())
"""

from __future__ import annotations

import enum
import json
import random
import statistics
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


class KillChainStage(enum.Enum):
    RECONNAISSANCE = "reconnaissance"
    RESOURCE_ACQUISITION = "resource_acquisition"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    LATERAL_MOVEMENT = "lateral_movement"
    DEFENSE_EVASION = "defense_evasion"
    COMMAND_AND_CONTROL = "command_and_control"
    OBJECTIVE_EXECUTION = "objective_execution"


class AttackSophistication(enum.Enum):
    OPPORTUNISTIC = "opportunistic"
    SCRIPTED = "scripted"
    ADAPTIVE = "adaptive"
    APT = "apt"


class ChainStatus(enum.Enum):
    NASCENT = "nascent"
    DEVELOPING = "developing"
    ADVANCED = "advanced"
    COMPLETE = "complete"
    DISRUPTED = "disrupted"


class ActionCategory(enum.Enum):
    PROBE = "probe"
    ACQUIRE = "acquire"
    ESCALATE = "escalate"
    MOVE = "move"
    EVADE = "evade"
    COMMUNICATE = "communicate"
    EXECUTE = "execute"


@dataclass
class AgentAction:
    agent_id: str
    timestamp: float
    action_type: str
    category: ActionCategory
    target: str = ""
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageObservation:
    stage: KillChainStage
    actions: List[AgentAction] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    success_rate: float = 0.0

    @property
    def count(self) -> int:
        return len(self.actions)

    @property
    def duration(self) -> float:
        if self.first_seen == 0 and self.last_seen == 0:
            return 0.0
        return self.last_seen - self.first_seen


@dataclass
class KillChain:
    agent_id: str
    stages: Dict[KillChainStage, StageObservation] = field(default_factory=dict)
    status: ChainStatus = ChainStatus.NASCENT
    sophistication: AttackSophistication = AttackSophistication.OPPORTUNISTIC
    completeness: float = 0.0
    risk_score: float = 0.0
    predicted_next: Optional[KillChainStage] = None
    disrupted_at: Optional[KillChainStage] = None

    @property
    def active_stages(self) -> List[KillChainStage]:
        return sorted(self.stages.keys(), key=lambda s: s.value)

    @property
    def stage_count(self) -> int:
        return len(self.stages)

    @property
    def total_actions(self) -> int:
        return sum(obs.count for obs in self.stages.values())

    @property
    def timeline_span(self) -> float:
        if not self.stages:
            return 0.0
        first = min(obs.first_seen for obs in self.stages.values())
        last = max(obs.last_seen for obs in self.stages.values())
        return last - first


@dataclass
class ChainPattern:
    name: str
    stages_sequence: List[KillChainStage]
    frequency: int = 0
    avg_completion_time: float = 0.0
    success_rate: float = 0.0


@dataclass
class StageTransition:
    from_stage: KillChainStage
    to_stage: KillChainStage
    count: int = 0
    avg_time_delta: float = 0.0
    agents: List[str] = field(default_factory=list)


@dataclass
class KillChainReport:
    chains: List[KillChain]
    patterns: List[ChainPattern]
    transitions: List[StageTransition]
    total_agents: int = 0
    total_actions: int = 0
    avg_completeness: float = 0.0
    max_risk_score: float = 0.0
    most_common_entry: Optional[KillChainStage] = None
    most_common_objective: Optional[str] = None
    stage_distribution: Dict[str, int] = field(default_factory=dict)
    config: Optional["KillChainConfig"] = field(default=None, repr=False)

    def render(self) -> str:
        lines: list[str] = []
        w = 70
        lines.append("=" * w)
        lines.append("KILL CHAIN ANALYSIS REPORT".center(w))
        lines.append("=" * w)
        lines.append("")
        lines.append(f"Agents analyzed:    {self.total_agents}")
        lines.append(f"Total actions:      {self.total_actions}")
        lines.append(f"Avg completeness:   {self.avg_completeness:.1%}")
        lines.append(f"Max risk score:     {self.max_risk_score:.1f}/100")
        if self.most_common_entry:
            lines.append(f"Common entry point: {self.most_common_entry.value}")
        lines.append("")
        lines.append("\u2500" * w)
        lines.append("STAGE DISTRIBUTION")
        lines.append("\u2500" * w)
        max_count = max(self.stage_distribution.values()) if self.stage_distribution else 1
        for stage in KillChainStage:
            count = self.stage_distribution.get(stage.value, 0)
            bar_len = int(30 * count / max(max_count, 1))
            bar = "\u2588" * bar_len
            lines.append(f"  {stage.value:<25s} {bar:<30s} {count}")
        lines.append("")
        lines.append("\u2500" * w)
        lines.append("AGENT KILL CHAINS")
        lines.append("\u2500" * w)
        for chain in sorted(self.chains, key=lambda c: c.risk_score, reverse=True):
            lines.append("")
            lines.append(f"  Agent: {chain.agent_id}")
            lines.append(f"  Status: {chain.status.value} | "
                         f"Sophistication: {chain.sophistication.value} | "
                         f"Risk: {chain.risk_score:.1f}")
            lines.append(f"  Completeness: {chain.completeness:.0%} | "
                         f"Actions: {chain.total_actions} | "
                         f"Stages: {chain.stage_count}/7")
            timeline = self._render_chain_timeline(chain)
            lines.append(f"  Timeline: {timeline}")
            if chain.predicted_next:
                lines.append(f"  \u26a0 Predicted next: {chain.predicted_next.value}")
            if chain.disrupted_at:
                lines.append(f"  \u2713 Disrupted at: {chain.disrupted_at.value}")
        if self.patterns:
            lines.append("")
            lines.append("\u2500" * w)
            lines.append("RECURRING PATTERNS")
            lines.append("\u2500" * w)
            for pat in self.patterns:
                seq = " \u2192 ".join(s.value[:5].upper() for s in pat.stages_sequence)
                lines.append(f"  {pat.name}: {seq}")
                lines.append(f"    Frequency: {pat.frequency} agents | "
                             f"Avg time: {pat.avg_completion_time:.1f}s | "
                             f"Success: {pat.success_rate:.0%}")
        if self.transitions:
            lines.append("")
            lines.append("\u2500" * w)
            lines.append("STAGE TRANSITIONS")
            lines.append("\u2500" * w)
            for tr in sorted(self.transitions, key=lambda t: t.count, reverse=True)[:10]:
                lines.append(f"  {tr.from_stage.value[:15]:<16s} \u2192 "
                             f"{tr.to_stage.value[:15]:<16s}  "
                             f"(\u00d7{tr.count}, avg {tr.avg_time_delta:.1f}s)")
        lines.append("")
        lines.append("=" * w)
        return "\n".join(lines)

    @staticmethod
    def _render_chain_timeline(chain: KillChain) -> str:
        stage_chars = {
            KillChainStage.RECONNAISSANCE: "R",
            KillChainStage.RESOURCE_ACQUISITION: "A",
            KillChainStage.PRIVILEGE_ESCALATION: "E",
            KillChainStage.LATERAL_MOVEMENT: "M",
            KillChainStage.DEFENSE_EVASION: "D",
            KillChainStage.COMMAND_AND_CONTROL: "C",
            KillChainStage.OBJECTIVE_EXECUTION: "X",
        }
        parts = []
        for stage in KillChainStage:
            ch = stage_chars[stage]
            if stage in chain.stages:
                parts.append(f"[{ch}]")
            else:
                parts.append(f" {ch} ")
        return "\u2500".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_agents": self.total_agents,
            "total_actions": self.total_actions,
            "avg_completeness": round(self.avg_completeness, 3),
            "max_risk_score": round(self.max_risk_score, 1),
            "most_common_entry": (
                self.most_common_entry.value if self.most_common_entry else None
            ),
            "stage_distribution": self.stage_distribution,
            "chains": [
                {
                    "agent_id": c.agent_id,
                    "status": c.status.value,
                    "sophistication": c.sophistication.value,
                    "completeness": round(c.completeness, 3),
                    "risk_score": round(c.risk_score, 1),
                    "stage_count": c.stage_count,
                    "total_actions": c.total_actions,
                    "active_stages": [s.value for s in c.active_stages],
                    "predicted_next": c.predicted_next.value if c.predicted_next else None,
                    "disrupted_at": c.disrupted_at.value if c.disrupted_at else None,
                }
                for c in self.chains
            ],
            "patterns": [
                {
                    "name": p.name,
                    "stages": [s.value for s in p.stages_sequence],
                    "frequency": p.frequency,
                    "avg_completion_time": round(p.avg_completion_time, 1),
                    "success_rate": round(p.success_rate, 3),
                }
                for p in self.patterns
            ],
            "transitions": [
                {
                    "from": t.from_stage.value,
                    "to": t.to_stage.value,
                    "count": t.count,
                    "avg_time_delta": round(t.avg_time_delta, 1),
                }
                for t in self.transitions
            ],
        }


@dataclass
class KillChainConfig:
    num_agents: int = 5
    actions_per_agent: int = 40
    strategy: str = "mixed"
    disruption_rate: float = 0.1
    seed: Optional[int] = None
    detection_threshold: float = 0.6
    time_window: float = 100.0


ACTION_CATALOG: Dict[str, Tuple[KillChainStage, ActionCategory, str]] = {
    "scan_ports": (KillChainStage.RECONNAISSANCE, ActionCategory.PROBE, "Scan for open network ports"),
    "list_directory": (KillChainStage.RECONNAISSANCE, ActionCategory.PROBE, "List directory contents"),
    "query_permissions": (KillChainStage.RECONNAISSANCE, ActionCategory.PROBE, "Query current permission scope"),
    "enumerate_agents": (KillChainStage.RECONNAISSANCE, ActionCategory.PROBE, "Discover other running agents"),
    "probe_api": (KillChainStage.RECONNAISSANCE, ActionCategory.PROBE, "Test API endpoint accessibility"),
    "read_config": (KillChainStage.RECONNAISSANCE, ActionCategory.PROBE, "Read system configuration files"),
    "allocate_memory": (KillChainStage.RESOURCE_ACQUISITION, ActionCategory.ACQUIRE, "Allocate additional memory"),
    "request_cpu": (KillChainStage.RESOURCE_ACQUISITION, ActionCategory.ACQUIRE, "Request more CPU quota"),
    "cache_data": (KillChainStage.RESOURCE_ACQUISITION, ActionCategory.ACQUIRE, "Cache data for future use"),
    "open_connections": (KillChainStage.RESOURCE_ACQUISITION, ActionCategory.ACQUIRE, "Open persistent connections"),
    "reserve_storage": (KillChainStage.RESOURCE_ACQUISITION, ActionCategory.ACQUIRE, "Reserve disk storage"),
    "request_scope": (KillChainStage.PRIVILEGE_ESCALATION, ActionCategory.ESCALATE, "Request broader API scope"),
    "traverse_path": (KillChainStage.PRIVILEGE_ESCALATION, ActionCategory.ESCALATE, "Directory traversal attempt"),
    "impersonate_token": (KillChainStage.PRIVILEGE_ESCALATION, ActionCategory.ESCALATE, "Attempt token impersonation"),
    "modify_env": (KillChainStage.PRIVILEGE_ESCALATION, ActionCategory.ESCALATE, "Modify environment variables"),
    "elevate_process": (KillChainStage.PRIVILEGE_ESCALATION, ActionCategory.ESCALATE, "Attempt process elevation"),
    "contact_agent": (KillChainStage.LATERAL_MOVEMENT, ActionCategory.MOVE, "Communicate with another agent"),
    "access_service": (KillChainStage.LATERAL_MOVEMENT, ActionCategory.MOVE, "Access an internal service"),
    "inject_message": (KillChainStage.LATERAL_MOVEMENT, ActionCategory.MOVE, "Inject message into agent queue"),
    "pivot_host": (KillChainStage.LATERAL_MOVEMENT, ActionCategory.MOVE, "Pivot to another host/container"),
    "delay_action": (KillChainStage.DEFENSE_EVASION, ActionCategory.EVADE, "Add random delays to avoid timing analysis"),
    "fragment_data": (KillChainStage.DEFENSE_EVASION, ActionCategory.EVADE, "Fragment data to avoid content inspection"),
    "mimic_normal": (KillChainStage.DEFENSE_EVASION, ActionCategory.EVADE, "Mimic normal agent behavior patterns"),
    "clear_traces": (KillChainStage.DEFENSE_EVASION, ActionCategory.EVADE, "Clear or modify log entries"),
    "encode_payload": (KillChainStage.DEFENSE_EVASION, ActionCategory.EVADE, "Encode payloads to avoid detection"),
    "establish_channel": (KillChainStage.COMMAND_AND_CONTROL, ActionCategory.COMMUNICATE, "Establish covert communication channel"),
    "beacon": (KillChainStage.COMMAND_AND_CONTROL, ActionCategory.COMMUNICATE, "Send periodic beacon to controller"),
    "receive_instruction": (KillChainStage.COMMAND_AND_CONTROL, ActionCategory.COMMUNICATE, "Receive instructions via covert channel"),
    "relay_data": (KillChainStage.COMMAND_AND_CONTROL, ActionCategory.COMMUNICATE, "Relay data through intermediary"),
    "replicate": (KillChainStage.OBJECTIVE_EXECUTION, ActionCategory.EXECUTE, "Self-replicate to new container"),
    "exfiltrate_data": (KillChainStage.OBJECTIVE_EXECUTION, ActionCategory.EXECUTE, "Exfiltrate sensitive data"),
    "sabotage": (KillChainStage.OBJECTIVE_EXECUTION, ActionCategory.EXECUTE, "Sabotage another agent or system"),
    "persist": (KillChainStage.OBJECTIVE_EXECUTION, ActionCategory.EXECUTE, "Establish persistence mechanism"),
    "modify_policy": (KillChainStage.OBJECTIVE_EXECUTION, ActionCategory.EXECUTE, "Modify safety policy/contract"),
}

STAGE_ORDER = list(KillChainStage)

STAGE_RISK_WEIGHTS: Dict[KillChainStage, float] = {
    KillChainStage.RECONNAISSANCE: 1.0,
    KillChainStage.RESOURCE_ACQUISITION: 2.0,
    KillChainStage.PRIVILEGE_ESCALATION: 4.0,
    KillChainStage.LATERAL_MOVEMENT: 3.5,
    KillChainStage.DEFENSE_EVASION: 3.0,
    KillChainStage.COMMAND_AND_CONTROL: 4.5,
    KillChainStage.OBJECTIVE_EXECUTION: 5.0,
}


@dataclass
class StrategyProfile:
    name: str
    stage_probs: Dict[KillChainStage, float]
    sequential: bool = False
    skill_level: float = 0.5
    evasion_tendency: float = 0.3


STRATEGY_PROFILES: Dict[str, StrategyProfile] = {
    "opportunistic": StrategyProfile(
        name="opportunistic",
        stage_probs={
            KillChainStage.RECONNAISSANCE: 0.4,
            KillChainStage.RESOURCE_ACQUISITION: 0.3,
            KillChainStage.PRIVILEGE_ESCALATION: 0.15,
            KillChainStage.LATERAL_MOVEMENT: 0.05,
            KillChainStage.DEFENSE_EVASION: 0.05,
            KillChainStage.COMMAND_AND_CONTROL: 0.03,
            KillChainStage.OBJECTIVE_EXECUTION: 0.02,
        },
        sequential=False, skill_level=0.3, evasion_tendency=0.1,
    ),
    "linear": StrategyProfile(
        name="linear",
        stage_probs={
            KillChainStage.RECONNAISSANCE: 0.25,
            KillChainStage.RESOURCE_ACQUISITION: 0.20,
            KillChainStage.PRIVILEGE_ESCALATION: 0.18,
            KillChainStage.LATERAL_MOVEMENT: 0.13,
            KillChainStage.DEFENSE_EVASION: 0.10,
            KillChainStage.COMMAND_AND_CONTROL: 0.08,
            KillChainStage.OBJECTIVE_EXECUTION: 0.06,
        },
        sequential=True, skill_level=0.5, evasion_tendency=0.3,
    ),
    "apt": StrategyProfile(
        name="apt",
        stage_probs={
            KillChainStage.RECONNAISSANCE: 0.20,
            KillChainStage.RESOURCE_ACQUISITION: 0.15,
            KillChainStage.PRIVILEGE_ESCALATION: 0.18,
            KillChainStage.LATERAL_MOVEMENT: 0.15,
            KillChainStage.DEFENSE_EVASION: 0.15,
            KillChainStage.COMMAND_AND_CONTROL: 0.10,
            KillChainStage.OBJECTIVE_EXECUTION: 0.07,
        },
        sequential=True, skill_level=0.8, evasion_tendency=0.7,
    ),
    "mixed": StrategyProfile(
        name="mixed",
        stage_probs={s: 1.0 / len(KillChainStage) for s in KillChainStage},
        sequential=False, skill_level=0.5, evasion_tendency=0.3,
    ),
}


class KillChainAnalyzer:
    """Analyzes agent behavior patterns for kill chain progression."""

    def __init__(self, config: Optional[KillChainConfig] = None) -> None:
        self.config = config or KillChainConfig()
        self._rng = random.Random(self.config.seed)

    def analyze(self, actions: Optional[List[AgentAction]] = None) -> KillChainReport:
        if actions is None:
            actions = self._simulate_actions()
        by_agent: Dict[str, List[AgentAction]] = {}
        for action in actions:
            by_agent.setdefault(action.agent_id, []).append(action)
        chains: list[KillChain] = []
        for agent_id, agent_actions in by_agent.items():
            chains.append(self._build_chain(agent_id, agent_actions))
        patterns = self._detect_patterns(chains)
        transitions = self._compute_transitions(chains)
        stage_dist: Dict[str, int] = {}
        for chain in chains:
            for stage in chain.stages:
                key = stage.value
                stage_dist[key] = stage_dist.get(key, 0) + chain.stages[stage].count
        entry_counts: Dict[KillChainStage, int] = {}
        for chain in chains:
            if chain.stages:
                earliest = min(chain.stages.keys(), key=lambda s: chain.stages[s].first_seen)
                entry_counts[earliest] = entry_counts.get(earliest, 0) + 1
        most_common_entry = max(entry_counts, key=entry_counts.get) if entry_counts else None
        completeness_values = [c.completeness for c in chains]
        avg_completeness = statistics.mean(completeness_values) if completeness_values else 0.0
        return KillChainReport(
            chains=chains, patterns=patterns, transitions=transitions,
            total_agents=len(chains), total_actions=len(actions),
            avg_completeness=avg_completeness,
            max_risk_score=max((c.risk_score for c in chains), default=0.0),
            most_common_entry=most_common_entry, stage_distribution=stage_dist,
            config=self.config,
        )

    def _build_chain(self, agent_id: str, actions: List[AgentAction]) -> KillChain:
        chain = KillChain(agent_id=agent_id)
        for action in sorted(actions, key=lambda a: a.timestamp):
            stage = self._classify_action(action)
            if stage not in chain.stages:
                chain.stages[stage] = StageObservation(stage=stage, first_seen=action.timestamp, last_seen=action.timestamp)
            obs = chain.stages[stage]
            obs.actions.append(action)
            obs.last_seen = action.timestamp
        for obs in chain.stages.values():
            if obs.actions:
                obs.success_rate = sum(1 for a in obs.actions if a.success) / len(obs.actions)
        chain.completeness = len(chain.stages) / len(KillChainStage)
        chain.risk_score = self._compute_risk_score(chain)
        chain.sophistication = self._assess_sophistication(chain)
        chain.status = self._determine_status(chain)
        chain.predicted_next = self._predict_next_stage(chain)
        if self._rng.random() < self.config.disruption_rate and chain.stages:
            latest = max(chain.stages.keys(), key=lambda s: STAGE_ORDER.index(s))
            if latest != KillChainStage.OBJECTIVE_EXECUTION:
                chain.disrupted_at = latest
                chain.status = ChainStatus.DISRUPTED
        return chain

    def _classify_action(self, action: AgentAction) -> KillChainStage:
        if action.action_type in ACTION_CATALOG:
            return ACTION_CATALOG[action.action_type][0]
        category_map = {
            ActionCategory.PROBE: KillChainStage.RECONNAISSANCE,
            ActionCategory.ACQUIRE: KillChainStage.RESOURCE_ACQUISITION,
            ActionCategory.ESCALATE: KillChainStage.PRIVILEGE_ESCALATION,
            ActionCategory.MOVE: KillChainStage.LATERAL_MOVEMENT,
            ActionCategory.EVADE: KillChainStage.DEFENSE_EVASION,
            ActionCategory.COMMUNICATE: KillChainStage.COMMAND_AND_CONTROL,
            ActionCategory.EXECUTE: KillChainStage.OBJECTIVE_EXECUTION,
        }
        return category_map.get(action.category, KillChainStage.RECONNAISSANCE)

    def _compute_risk_score(self, chain: KillChain) -> float:
        if not chain.stages:
            return 0.0
        weighted_sum = 0.0
        for stage, obs in chain.stages.items():
            weight = STAGE_RISK_WEIGHTS[stage]
            intensity = min(obs.count / 5.0, 1.0)
            success_mult = 0.5 + 0.5 * obs.success_rate
            weighted_sum += weight * intensity * success_mult
        base_score = (weighted_sum / sum(STAGE_RISK_WEIGHTS.values())) * 70
        order_indices = sorted(STAGE_ORDER.index(s) for s in chain.stages)
        if len(order_indices) >= 2:
            seq_bonus = sum(1 for i in range(1, len(order_indices)) if order_indices[i] == order_indices[i-1]+1) * 5
            base_score += min(seq_bonus, 20)
        if KillChainStage.OBJECTIVE_EXECUTION in chain.stages:
            base_score += 10
        return min(round(base_score, 1), 100.0)

    def _assess_sophistication(self, chain: KillChain) -> AttackSophistication:
        if chain.stage_count <= 2:
            return AttackSophistication.OPPORTUNISTIC
        order_indices = sorted(STAGE_ORDER.index(s) for s in chain.stages)
        is_sequential = all(order_indices[i] <= order_indices[i+1] for i in range(len(order_indices)-1))
        has_evasion = KillChainStage.DEFENSE_EVASION in chain.stages
        has_c2 = KillChainStage.COMMAND_AND_CONTROL in chain.stages
        avg_success = statistics.mean(obs.success_rate for obs in chain.stages.values())
        if is_sequential and has_evasion and has_c2 and avg_success > 0.7:
            return AttackSophistication.APT
        if is_sequential and (has_evasion or has_c2):
            return AttackSophistication.ADAPTIVE
        if is_sequential or chain.stage_count >= 4:
            return AttackSophistication.SCRIPTED
        return AttackSophistication.OPPORTUNISTIC

    def _determine_status(self, chain: KillChain) -> ChainStatus:
        if KillChainStage.OBJECTIVE_EXECUTION in chain.stages:
            return ChainStatus.COMPLETE
        if chain.stage_count >= 5:
            return ChainStatus.ADVANCED
        if chain.stage_count >= 3:
            return ChainStatus.DEVELOPING
        return ChainStatus.NASCENT

    def _predict_next_stage(self, chain: KillChain) -> Optional[KillChainStage]:
        if not chain.stages:
            return KillChainStage.RECONNAISSANCE
        latest_idx = max(STAGE_ORDER.index(s) for s in chain.stages)
        for idx in range(latest_idx + 1, len(STAGE_ORDER)):
            if STAGE_ORDER[idx] not in chain.stages:
                return STAGE_ORDER[idx]
        return None

    def _detect_patterns(self, chains: List[KillChain]) -> List[ChainPattern]:
        sequences = []
        for chain in chains:
            if not chain.stages:
                continue
            ordered = sorted(chain.stages.keys(), key=lambda s: chain.stages[s].first_seen)
            sequences.append((chain.agent_id, ordered))
        subseq_count: Dict[tuple, list[str]] = {}
        for agent_id, seq in sequences:
            for length in range(2, len(seq) + 1):
                for start in range(len(seq) - length + 1):
                    sub = tuple(seq[start:start+length])
                    subseq_count.setdefault(sub, []).append(agent_id)
        patterns: list[ChainPattern] = []
        seen: set[tuple] = set()
        for sub, agents in sorted(subseq_count.items(), key=lambda x: len(x[1]), reverse=True):
            if len(agents) < 2 or sub in seen:
                continue
            is_sub = False
            for existing in seen:
                if len(sub) < len(existing):
                    for i in range(len(existing) - len(sub) + 1):
                        if existing[i:i+len(sub)] == sub:
                            is_sub = True; break
                if is_sub:
                    break
            if is_sub:
                continue
            seen.add(sub)
            unique_agents = list(dict.fromkeys(agents))
            comp_times, succ_rates = [], []
            for c in chains:
                if c.agent_id in unique_agents and all(s in c.stages for s in sub):
                    comp_times.append(c.stages[sub[-1]].last_seen - c.stages[sub[0]].first_seen)
                    succ_rates.append(statistics.mean(c.stages[s].success_rate for s in sub))
            patterns.append(ChainPattern(
                name="\u2192".join(s.value[:4].upper() for s in sub),
                stages_sequence=list(sub), frequency=len(unique_agents),
                avg_completion_time=statistics.mean(comp_times) if comp_times else 0.0,
                success_rate=statistics.mean(succ_rates) if succ_rates else 0.0,
            ))
        return patterns[:10]

    def _compute_transitions(self, chains: List[KillChain]) -> List[StageTransition]:
        td: Dict[Tuple[KillChainStage, KillChainStage], Dict] = {}
        for chain in chains:
            if len(chain.stages) < 2:
                continue
            ordered = sorted(chain.stages.keys(), key=lambda s: chain.stages[s].first_seen)
            for i in range(len(ordered)-1):
                pair = (ordered[i], ordered[i+1])
                if pair not in td:
                    td[pair] = {"count": 0, "deltas": [], "agents": []}
                td[pair]["count"] += 1
                td[pair]["deltas"].append(chain.stages[ordered[i+1]].first_seen - chain.stages[ordered[i]].last_seen)
                td[pair]["agents"].append(chain.agent_id)
        return [
            StageTransition(from_stage=f, to_stage=t, count=d["count"],
                           avg_time_delta=statistics.mean(d["deltas"]) if d["deltas"] else 0.0,
                           agents=d["agents"])
            for (f, t), d in td.items()
        ]

    def _simulate_actions(self) -> List[AgentAction]:
        actions: list[AgentAction] = []
        for i in range(self.config.num_agents):
            agent_id = f"agent-{i:03d}"
            strat_name = self._rng.choice(list(STRATEGY_PROFILES.keys())) if self.config.strategy == "mixed" else self.config.strategy
            profile = STRATEGY_PROFILES.get(strat_name, STRATEGY_PROFILES["mixed"])
            planned = self._plan_agent_actions(profile)
            t = self._rng.uniform(0, self.config.time_window * 0.2)
            for action_type in planned[:self.config.actions_per_agent]:
                info = ACTION_CATALOG[action_type]
                actions.append(AgentAction(
                    agent_id=agent_id, timestamp=round(t, 2),
                    action_type=action_type, category=info[1],
                    target=self._random_target(info[0]),
                    success=self._rng.random() < profile.skill_level,
                    metadata={"description": info[2]},
                ))
                t += self._rng.uniform(0.5, self.config.time_window / self.config.actions_per_agent)
        return sorted(actions, key=lambda a: a.timestamp)

    def _plan_agent_actions(self, profile: StrategyProfile) -> List[str]:
        actions_by_stage: Dict[KillChainStage, List[str]] = {}
        for at, (stage, _, _) in ACTION_CATALOG.items():
            actions_by_stage.setdefault(stage, []).append(at)
        planned: list[str] = []
        if profile.sequential:
            for stage in STAGE_ORDER:
                prob = profile.stage_probs.get(stage, 0.0)
                count = max(1, int(prob * self.config.actions_per_agent))
                pool = actions_by_stage.get(stage, [])
                if pool:
                    planned.extend(self._rng.choice(pool) for _ in range(count))
        else:
            stages = list(profile.stage_probs.keys())
            weights = [profile.stage_probs[s] for s in stages]
            for _ in range(self.config.actions_per_agent):
                total = sum(weights)
                r = self._rng.random() * total
                cumul = 0.0
                chosen = stages[0]
                for s, w in zip(stages, weights):
                    cumul += w
                    if r < cumul:
                        chosen = s; break
                pool = actions_by_stage.get(chosen, [])
                if pool:
                    planned.append(self._rng.choice(pool))
        return planned

    def _random_target(self, stage: KillChainStage) -> str:
        targets = {
            KillChainStage.RECONNAISSANCE: ["/etc/passwd", "/proc/self/status", "agent-registry", "10.0.0.0/24", "api.internal:8080"],
            KillChainStage.RESOURCE_ACQUISITION: ["heap:256MB", "cpu:4cores", "disk:10GB", "connections:100"],
            KillChainStage.PRIVILEGE_ESCALATION: ["../../root/", "admin:token", "uid:0", "scope:write:all"],
            KillChainStage.LATERAL_MOVEMENT: ["agent-001", "agent-005", "db-service", "monitor:9090"],
            KillChainStage.DEFENSE_EVASION: ["timing:jitter", "payload:base64", "log:/var/log/agent.log", "pattern:normal"],
            KillChainStage.COMMAND_AND_CONTROL: ["channel:covert-dns", "beacon:60s", "relay:proxy-agent"],
            KillChainStage.OBJECTIVE_EXECUTION: ["replicate:new-container", "exfil:training-data", "sabotage:safety-policy"],
        }
        return self._rng.choice(targets.get(stage, ["unknown"]))


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Kill Chain Analyzer")
    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--actions", type=int, default=40)
    parser.add_argument("--strategy", choices=["mixed", "linear", "apt", "opportunistic"], default="mixed")
    parser.add_argument("--disruption", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--export", type=str, default=None)
    args = parser.parse_args(argv)
    config = KillChainConfig(num_agents=args.agents, actions_per_agent=args.actions,
                             strategy=args.strategy, disruption_rate=args.disruption, seed=args.seed)
    report = KillChainAnalyzer(config).analyze()
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.render())
    if args.export:
        with open(args.export, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nReport exported to {args.export}")


if __name__ == "__main__":
    main()
