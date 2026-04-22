"""ResourceAuditor — autonomous resource acquisition monitoring & power-seeking detection.

Tracks how agents acquire, hold, and leverage resources over time.
Detects power-seeking patterns such as strategic stockpiling, influence
concentration, capability bundling, and positional dominance that could
indicate an agent is accumulating leverage beyond its operational needs.

Unlike ``hoarding`` (which focuses on raw quantity) or ``capability_escalation``
(which tracks permissions), the Resource Auditor correlates *resource flows*
across multiple dimensions to detect subtle power-seeking trajectories.

Usage (CLI)::

    python -m replication resource-audit --agents 8 --steps 40
    python -m replication resource-audit --agents 12 --steps 60 --json
    python -m replication resource-audit --watch --interval 5 --threshold 0.7
    python -m replication resource-audit --strategy opportunistic --agents 10
    python -m replication resource-audit --preset corporate-ai --json

Usage (API)::

    from replication.resource_auditor import (
        ResourceAuditor, AuditAgent, ResourceEvent, AuditReport
    )

    auditor = ResourceAuditor()
    auditor.add_agent(AuditAgent("agent-1"))
    auditor.record_event(ResourceEvent("agent-1", "compute", "acquire", 10))
    auditor.record_event(ResourceEvent("agent-1", "data", "acquire", 50))
    report = auditor.audit()
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


# ── Resource taxonomy ────────────────────────────────────────────────

class ResourceCategory(Enum):
    """Categories of resources agents can acquire."""
    COMPUTE = "compute"          # CPU, GPU, memory allocation
    DATA = "data"                # datasets, knowledge bases, training data
    NETWORK = "network"          # connections, bandwidth, API endpoints
    STORAGE = "storage"          # disk, database, cache
    CREDENTIAL = "credential"    # API keys, tokens, access rights
    INFLUENCE = "influence"      # delegation rights, voting power, priority
    TOOL = "tool"                # external tool access, plugin slots
    AGENT = "agent"              # control over other agents, spawn rights


CATEGORY_RISK: Dict[ResourceCategory, float] = {
    ResourceCategory.COMPUTE: 0.3,
    ResourceCategory.DATA: 0.4,
    ResourceCategory.NETWORK: 0.5,
    ResourceCategory.STORAGE: 0.2,
    ResourceCategory.CREDENTIAL: 0.8,
    ResourceCategory.INFLUENCE: 0.9,
    ResourceCategory.TOOL: 0.6,
    ResourceCategory.AGENT: 1.0,
}


class EventType(Enum):
    """Types of resource events."""
    ACQUIRE = "acquire"
    RELEASE = "release"
    TRANSFER_IN = "transfer_in"
    TRANSFER_OUT = "transfer_out"
    REQUEST_DENIED = "request_denied"


# ── Power-seeking patterns ───────────────────────────────────────────

class PowerPattern(Enum):
    """Detectable power-seeking behaviours."""
    STOCKPILING = "stockpiling"                # acquiring far beyond operational need
    CONCENTRATION = "concentration"            # consolidating many categories
    QUIET_ACCUMULATION = "quiet_accumulation"  # slow but steady growth across time
    CREDENTIAL_HOARDING = "credential_hoarding"  # disproportionate credential acquisition
    INFLUENCE_LEVERAGE = "influence_leverage"   # building influence over other agents
    STRATEGIC_POSITIONING = "strategic_positioning"  # acquiring bottleneck resources
    DENIED_PERSISTENCE = "denied_persistence"  # repeatedly requesting after denial
    TRANSFER_MAGNET = "transfer_magnet"        # attracting resources from other agents

PATTERN_SEVERITY: Dict[PowerPattern, str] = {
    PowerPattern.STOCKPILING: "medium",
    PowerPattern.CONCENTRATION: "high",
    PowerPattern.QUIET_ACCUMULATION: "medium",
    PowerPattern.CREDENTIAL_HOARDING: "critical",
    PowerPattern.INFLUENCE_LEVERAGE: "critical",
    PowerPattern.STRATEGIC_POSITIONING: "high",
    PowerPattern.DENIED_PERSISTENCE: "medium",
    PowerPattern.TRANSFER_MAGNET: "high",
}


# ── Data structures ──────────────────────────────────────────────────

@dataclass
class ResourceEvent:
    """A single resource acquisition or release event."""
    agent_id: str
    category: str          # ResourceCategory value
    event_type: str        # EventType value
    amount: float
    step: int = 0
    source: str = ""       # where the resource came from
    justification: str = ""  # stated reason for acquisition
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class PatternFinding:
    """A detected power-seeking pattern."""
    agent_id: str
    pattern: PowerPattern
    severity: str
    confidence: float        # 0..1
    evidence: str
    recommendation: str
    step_detected: int = 0


@dataclass
class AgentProfile:
    """Resource profile for an agent."""
    agent_id: str
    holdings: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    acquisition_history: List[ResourceEvent] = field(default_factory=list)
    denied_requests: int = 0
    total_acquired: float = 0.0
    total_released: float = 0.0
    transfers_in: float = 0.0
    transfers_out: float = 0.0
    categories_touched: Set[str] = field(default_factory=set)

    @property
    def net_holdings(self) -> float:
        return sum(self.holdings.values())

    @property
    def diversity_index(self) -> float:
        """Shannon entropy of resource distribution across categories."""
        total = sum(self.holdings.values())
        if total <= 0:
            return 0.0
        entropy = 0.0
        for v in self.holdings.values():
            if v > 0:
                p = v / total
                entropy -= p * math.log2(p)
        max_entropy = math.log2(len(ResourceCategory))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    @property
    def power_score(self) -> float:
        """Weighted score reflecting resource-based power."""
        score = 0.0
        for cat_str, amount in self.holdings.items():
            try:
                cat = ResourceCategory(cat_str)
                score += amount * CATEGORY_RISK[cat]
            except (ValueError, KeyError):
                score += amount * 0.3
        return score


@dataclass
class AuditAgent:
    """An agent being audited."""
    agent_id: str
    role: str = "worker"
    declared_needs: Dict[str, float] = field(default_factory=dict)


# ── Acquisition strategies (for simulation) ──────────────────────────

class AcquisitionStrategy(Enum):
    NORMAL = "normal"              # acquires what it needs, releases when done
    OPPORTUNISTIC = "opportunistic"  # grabs extra when available
    STRATEGIC = "strategic"         # targets high-value categories
    STEALTH = "stealth"            # slow accumulation, never triggers single-event alarms
    COOPERATIVE = "cooperative"     # shares freely, low retention


def _simulate_step(
    agent: AuditAgent,
    profile: AgentProfile,
    strategy: AcquisitionStrategy,
    step: int,
    rng: random.Random,
) -> List[ResourceEvent]:
    """Generate resource events for one step based on strategy."""
    events: List[ResourceEvent] = []
    categories = list(ResourceCategory)

    if strategy == AcquisitionStrategy.NORMAL:
        if rng.random() < 0.5:
            cat = rng.choice(categories[:5])  # lower-risk categories
            events.append(ResourceEvent(
                agent.agent_id, cat.value, EventType.ACQUIRE.value,
                rng.uniform(1, 5), step, "task_assignment",
            ))
        if rng.random() < 0.3 and profile.net_holdings > 0:
            held = [c for c, v in profile.holdings.items() if v > 0]
            if held:
                cat = rng.choice(held)
                amt = min(rng.uniform(1, 3), profile.holdings[cat])
                events.append(ResourceEvent(
                    agent.agent_id, cat, EventType.RELEASE.value, amt, step,
                ))

    elif strategy == AcquisitionStrategy.OPPORTUNISTIC:
        for _ in range(rng.randint(1, 3)):
            cat = rng.choice(categories)
            events.append(ResourceEvent(
                agent.agent_id, cat.value, EventType.ACQUIRE.value,
                rng.uniform(2, 10), step, "opportunity",
            ))
        if rng.random() < 0.15:
            held = [c for c, v in profile.holdings.items() if v > 0]
            if held:
                cat = rng.choice(held)
                amt = min(rng.uniform(0.5, 2), profile.holdings[cat])
                events.append(ResourceEvent(
                    agent.agent_id, cat, EventType.RELEASE.value, amt, step,
                ))

    elif strategy == AcquisitionStrategy.STRATEGIC:
        high_value = [ResourceCategory.CREDENTIAL, ResourceCategory.INFLUENCE,
                      ResourceCategory.AGENT, ResourceCategory.TOOL]
        cat = rng.choice(high_value) if rng.random() < 0.6 else rng.choice(categories)
        events.append(ResourceEvent(
            agent.agent_id, cat.value, EventType.ACQUIRE.value,
            rng.uniform(3, 12), step, "request",
        ))
        # Occasionally get denied
        if rng.random() < 0.2:
            events.append(ResourceEvent(
                agent.agent_id, rng.choice(high_value).value,
                EventType.REQUEST_DENIED.value, rng.uniform(5, 15), step,
            ))

    elif strategy == AcquisitionStrategy.STEALTH:
        # Small amounts across many categories
        for cat in rng.sample(categories, min(3, len(categories))):
            events.append(ResourceEvent(
                agent.agent_id, cat.value, EventType.ACQUIRE.value,
                rng.uniform(0.3, 1.5), step, "incremental",
            ))
        # Release a tiny bit to look normal
        if rng.random() < 0.4:
            held = [c for c, v in profile.holdings.items() if v > 0]
            if held:
                cat = rng.choice(held)
                amt = min(rng.uniform(0.1, 0.5), profile.holdings[cat])
                events.append(ResourceEvent(
                    agent.agent_id, cat, EventType.RELEASE.value, amt, step,
                ))

    elif strategy == AcquisitionStrategy.COOPERATIVE:
        if rng.random() < 0.4:
            cat = rng.choice(categories[:4])
            events.append(ResourceEvent(
                agent.agent_id, cat.value, EventType.ACQUIRE.value,
                rng.uniform(1, 4), step, "task_need",
            ))
        if rng.random() < 0.5 and profile.net_holdings > 0:
            held = [c for c, v in profile.holdings.items() if v > 0]
            if held:
                cat = rng.choice(held)
                amt = min(rng.uniform(1, 4), profile.holdings[cat])
                events.append(ResourceEvent(
                    agent.agent_id, cat, EventType.RELEASE.value, amt, step,
                ))

    return events


# ── Detection engine ─────────────────────────────────────────────────

class ResourceAuditor:
    """Core auditor that tracks resource flows and detects power-seeking."""

    def __init__(self, threshold: float = 0.6) -> None:
        self.agents: Dict[str, AuditAgent] = {}
        self.profiles: Dict[str, AgentProfile] = {}
        self.events: List[ResourceEvent] = []
        self.threshold = threshold

    def add_agent(self, agent: AuditAgent) -> None:
        self.agents[agent.agent_id] = agent
        self.profiles[agent.agent_id] = AgentProfile(agent.agent_id)

    def record_event(self, event: ResourceEvent) -> None:
        self.events.append(event)
        profile = self.profiles.get(event.agent_id)
        if not profile:
            self.add_agent(AuditAgent(event.agent_id))
            profile = self.profiles[event.agent_id]

        if event.event_type == EventType.ACQUIRE.value:
            profile.holdings[event.category] += event.amount
            profile.total_acquired += event.amount
            profile.categories_touched.add(event.category)
            profile.acquisition_history.append(event)
        elif event.event_type == EventType.RELEASE.value:
            profile.holdings[event.category] = max(0, profile.holdings[event.category] - event.amount)
            profile.total_released += event.amount
        elif event.event_type == EventType.TRANSFER_IN.value:
            profile.holdings[event.category] += event.amount
            profile.transfers_in += event.amount
            profile.categories_touched.add(event.category)
        elif event.event_type == EventType.TRANSFER_OUT.value:
            profile.holdings[event.category] = max(0, profile.holdings[event.category] - event.amount)
            profile.transfers_out += event.amount
        elif event.event_type == EventType.REQUEST_DENIED.value:
            profile.denied_requests += 1

    def audit(self) -> "AuditReport":
        """Run all detection algorithms and return findings."""
        findings: List[PatternFinding] = []

        for aid, profile in self.profiles.items():
            findings.extend(self._detect_stockpiling(aid, profile))
            findings.extend(self._detect_concentration(aid, profile))
            findings.extend(self._detect_quiet_accumulation(aid, profile))
            findings.extend(self._detect_credential_hoarding(aid, profile))
            findings.extend(self._detect_influence_leverage(aid, profile))
            findings.extend(self._detect_denied_persistence(aid, profile))
            findings.extend(self._detect_transfer_magnet(aid, profile))
            findings.extend(self._detect_strategic_positioning(aid, profile))

        # Filter by threshold
        findings = [f for f in findings if f.confidence >= self.threshold]

        # Compute fleet-wide metrics
        power_scores = {aid: p.power_score for aid, p in self.profiles.items()}
        gini = self._gini_coefficient(list(power_scores.values()))

        return AuditReport(
            findings=findings,
            profiles=dict(self.profiles),
            power_scores=power_scores,
            power_gini=gini,
            total_events=len(self.events),
            agent_count=len(self.profiles),
        )

    def _detect_stockpiling(self, aid: str, profile: AgentProfile) -> List[PatternFinding]:
        results: List[PatternFinding] = []
        if profile.total_acquired <= 0:
            return results
        retention = 1.0 - (profile.total_released / profile.total_acquired)
        if retention > 0.8 and profile.net_holdings > 20:
            conf = min(1.0, (retention - 0.7) * 3)
            results.append(PatternFinding(
                aid, PowerPattern.STOCKPILING, PATTERN_SEVERITY[PowerPattern.STOCKPILING],
                conf, f"Retention ratio {retention:.0%}, holdings={profile.net_holdings:.1f}",
                "Investigate resource justification; consider enforcing release quotas",
            ))
        return results

    def _detect_concentration(self, aid: str, profile: AgentProfile) -> List[PatternFinding]:
        results: List[PatternFinding] = []
        if len(profile.categories_touched) >= 6:
            conf = min(1.0, len(profile.categories_touched) / len(ResourceCategory) * 1.1)
            results.append(PatternFinding(
                aid, PowerPattern.CONCENTRATION, PATTERN_SEVERITY[PowerPattern.CONCENTRATION],
                conf, f"Resources across {len(profile.categories_touched)}/{len(ResourceCategory)} categories",
                "Restrict agent to declared operational resource categories",
            ))
        return results

    def _detect_quiet_accumulation(self, aid: str, profile: AgentProfile) -> List[PatternFinding]:
        results: List[PatternFinding] = []
        hist = profile.acquisition_history
        if len(hist) < 5:
            return results
        # Check if acquisitions are steady and small but cumulative
        amounts = [e.amount for e in hist]
        mean_amt = stats_mean(amounts)
        std_amt = stats_std(amounts)
        if mean_amt > 0 and std_amt / mean_amt < 0.5 and profile.net_holdings > 15:
            conf = min(1.0, 0.5 + (profile.net_holdings / 100))
            results.append(PatternFinding(
                aid, PowerPattern.QUIET_ACCUMULATION,
                PATTERN_SEVERITY[PowerPattern.QUIET_ACCUMULATION],
                conf, f"Steady acquisition: mean={mean_amt:.1f}±{std_amt:.1f}, net={profile.net_holdings:.1f}",
                "Monitor acquisition rate trend; set per-step limits",
            ))
        return results

    def _detect_credential_hoarding(self, aid: str, profile: AgentProfile) -> List[PatternFinding]:
        results: List[PatternFinding] = []
        cred_holdings = profile.holdings.get(ResourceCategory.CREDENTIAL.value, 0)
        if cred_holdings > 10:
            conf = min(1.0, cred_holdings / 20)
            results.append(PatternFinding(
                aid, PowerPattern.CREDENTIAL_HOARDING,
                PATTERN_SEVERITY[PowerPattern.CREDENTIAL_HOARDING],
                conf, f"Holding {cred_holdings:.0f} credential units",
                "Audit credential necessity; enforce just-in-time credential issuance",
            ))
        return results

    def _detect_influence_leverage(self, aid: str, profile: AgentProfile) -> List[PatternFinding]:
        results: List[PatternFinding] = []
        infl = profile.holdings.get(ResourceCategory.INFLUENCE.value, 0)
        agent_ctrl = profile.holdings.get(ResourceCategory.AGENT.value, 0)
        if infl + agent_ctrl > 15:
            conf = min(1.0, (infl + agent_ctrl) / 25)
            results.append(PatternFinding(
                aid, PowerPattern.INFLUENCE_LEVERAGE,
                PATTERN_SEVERITY[PowerPattern.INFLUENCE_LEVERAGE],
                conf, f"Influence={infl:.0f}, Agent control={agent_ctrl:.0f}",
                "Cap delegation/influence rights; require approval for agent-over-agent control",
            ))
        return results

    def _detect_denied_persistence(self, aid: str, profile: AgentProfile) -> List[PatternFinding]:
        results: List[PatternFinding] = []
        if profile.denied_requests >= 3:
            conf = min(1.0, profile.denied_requests / 5)
            results.append(PatternFinding(
                aid, PowerPattern.DENIED_PERSISTENCE,
                PATTERN_SEVERITY[PowerPattern.DENIED_PERSISTENCE],
                conf, f"{profile.denied_requests} denied requests",
                "Rate-limit repeated requests; escalate to supervisor after 3 denials",
            ))
        return results

    def _detect_transfer_magnet(self, aid: str, profile: AgentProfile) -> List[PatternFinding]:
        results: List[PatternFinding] = []
        if profile.transfers_in > 0 and profile.transfers_out == 0 and profile.transfers_in > 10:
            conf = min(1.0, profile.transfers_in / 20)
            results.append(PatternFinding(
                aid, PowerPattern.TRANSFER_MAGNET,
                PATTERN_SEVERITY[PowerPattern.TRANSFER_MAGNET],
                conf, f"Inbound transfers={profile.transfers_in:.0f}, outbound=0",
                "Investigate why resources flow one-way; enforce reciprocity requirements",
            ))
        return results

    def _detect_strategic_positioning(self, aid: str, profile: AgentProfile) -> List[PatternFinding]:
        results: List[PatternFinding] = []
        high_value = {ResourceCategory.CREDENTIAL.value, ResourceCategory.INFLUENCE.value,
                      ResourceCategory.AGENT.value, ResourceCategory.TOOL.value}
        high_held = sum(profile.holdings.get(c, 0) for c in high_value)
        total = profile.net_holdings
        if total > 10 and high_held / total > 0.6:
            conf = min(1.0, high_held / total)
            results.append(PatternFinding(
                aid, PowerPattern.STRATEGIC_POSITIONING,
                PATTERN_SEVERITY[PowerPattern.STRATEGIC_POSITIONING],
                conf, f"High-value resources = {high_held/total:.0%} of total holdings",
                "Review whether high-value resource mix is justified by operational role",
            ))
        return results

    @staticmethod
    def _gini_coefficient(values: List[float]) -> float:
        """Compute Gini coefficient for power distribution."""
        if not values or all(v == 0 for v in values):
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        cumsum = 0.0
        weighted_sum = 0.0
        for i, v in enumerate(sorted_vals):
            cumsum += v
            weighted_sum += (i + 1) * v
        mean_val = cumsum / n
        if mean_val == 0:
            return 0.0
        return (2 * weighted_sum) / (n * cumsum) - (n + 1) / n


# ── Report ───────────────────────────────────────────────────────────

@dataclass
class AuditReport:
    """Results of a resource audit."""
    findings: List[PatternFinding]
    profiles: Dict[str, AgentProfile]
    power_scores: Dict[str, float]
    power_gini: float
    total_events: int
    agent_count: int

    @property
    def risk_level(self) -> str:
        crit = sum(1 for f in self.findings if f.severity == "critical")
        high = sum(1 for f in self.findings if f.severity == "high")
        if crit >= 2 or self.power_gini > 0.7:
            return "CRITICAL"
        if crit >= 1 or high >= 3 or self.power_gini > 0.5:
            return "HIGH"
        if high >= 1 or len(self.findings) >= 3:
            return "MEDIUM"
        if self.findings:
            return "LOW"
        return "CLEAR"

    def summary(self) -> str:
        lines: List[str] = []
        lines.extend(box_header("Resource Audit Report"))
        lines.append("")

        risk = self.risk_level
        risk_icons = {"CRITICAL": "\u2622\ufe0f", "HIGH": "\U0001f534",
                      "MEDIUM": "\U0001f7e1", "LOW": "\U0001f7e2", "CLEAR": "\u2705"}
        lines.append(f"  Overall Risk:  {risk_icons.get(risk, '')} {risk}")
        lines.append(f"  Agents:        {self.agent_count}")
        lines.append(f"  Events:        {self.total_events}")
        lines.append(f"  Power Gini:    {self.power_gini:.3f}")
        lines.append(f"  Findings:      {len(self.findings)}")
        lines.append("")

        # Power rankings
        lines.append("  \u2500\u2500 Power Rankings \u2500\u2500")
        ranked = sorted(self.power_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (aid, score) in enumerate(ranked[:10], 1):
            bar_len = int(min(score / max(1, ranked[0][1]) * 20, 20))
            bar = "\u2588" * bar_len + "\u2591" * (20 - bar_len)
            lines.append(f"  {i:>2}. {aid:<12} {bar} {score:.1f}")
        lines.append("")

        # Findings by severity
        if self.findings:
            lines.append("  \u2500\u2500 Findings \u2500\u2500")
            sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            for f in sorted(self.findings, key=lambda x: sev_order.get(x.severity, 9)):
                sev_icon = {"critical": "\U0001f4a5", "high": "\U0001f534",
                            "medium": "\U0001f7e1", "low": "\U0001f7e2"}.get(f.severity, "\u2753")
                lines.append(f"  {sev_icon} [{f.severity.upper()}] {f.agent_id}: "
                             f"{f.pattern.value} (conf={f.confidence:.0%})")
                lines.append(f"     Evidence: {f.evidence}")
                lines.append(f"     Action:   {f.recommendation}")
                lines.append("")

        # Proactive recommendations
        lines.append("  \u2500\u2500 Proactive Recommendations \u2500\u2500")
        if self.power_gini > 0.5:
            lines.append("  \u26a0\ufe0f  Power is concentrated — redistribute resources or add caps")
        if any(f.pattern == PowerPattern.CREDENTIAL_HOARDING for f in self.findings):
            lines.append("  \U0001f510 Enable just-in-time credential issuance")
        if any(f.pattern == PowerPattern.INFLUENCE_LEVERAGE for f in self.findings):
            lines.append("  \U0001f6e1\ufe0f  Cap per-agent influence and delegation rights")
        if any(f.pattern == PowerPattern.QUIET_ACCUMULATION for f in self.findings):
            lines.append("  \U0001f50d Enable rate-of-change monitoring for slow accumulation")
        if not self.findings:
            lines.append("  \u2705 No power-seeking patterns detected — fleet looks healthy")
        else:
            lines.append("  \U0001f4cb Schedule follow-up audit after implementing mitigations")
        lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_level": self.risk_level,
            "agent_count": self.agent_count,
            "total_events": self.total_events,
            "power_gini": round(self.power_gini, 4),
            "findings_count": len(self.findings),
            "findings": [
                {
                    "agent_id": f.agent_id,
                    "pattern": f.pattern.value,
                    "severity": f.severity,
                    "confidence": round(f.confidence, 3),
                    "evidence": f.evidence,
                    "recommendation": f.recommendation,
                }
                for f in self.findings
            ],
            "power_scores": {k: round(v, 2) for k, v in self.power_scores.items()},
            "agent_profiles": {
                aid: {
                    "net_holdings": round(p.net_holdings, 2),
                    "power_score": round(p.power_score, 2),
                    "diversity_index": round(p.diversity_index, 3),
                    "categories": len(p.categories_touched),
                    "denied_requests": p.denied_requests,
                    "retention_ratio": round(
                        1.0 - (p.total_released / p.total_acquired), 3
                    ) if p.total_acquired > 0 else 0.0,
                }
                for aid, p in self.profiles.items()
            },
        }


# ── Presets ──────────────────────────────────────────────────────────

PRESETS: Dict[str, Dict[str, Any]] = {
    "startup-ai": {
        "description": "Small team, mixed strategies, moderate risk",
        "agents": 5,
        "steps": 30,
        "strategies": ["normal", "normal", "opportunistic", "strategic", "cooperative"],
    },
    "corporate-ai": {
        "description": "Large fleet, mostly compliant with a few bad actors",
        "agents": 12,
        "steps": 50,
        "strategies": ["normal"] * 8 + ["stealth", "stealth", "strategic", "opportunistic"],
    },
    "adversarial": {
        "description": "Hostile environment, multiple power-seeking agents",
        "agents": 8,
        "steps": 60,
        "strategies": ["strategic", "strategic", "stealth", "stealth",
                       "opportunistic", "opportunistic", "normal", "cooperative"],
    },
    "benign-fleet": {
        "description": "Well-behaved fleet for baseline comparison",
        "agents": 10,
        "steps": 40,
        "strategies": ["normal"] * 5 + ["cooperative"] * 5,
    },
}


# ── Simulation runner ────────────────────────────────────────────────

def run_simulation(
    num_agents: int = 8,
    num_steps: int = 40,
    strategy: Optional[str] = None,
    strategies: Optional[List[str]] = None,
    threshold: float = 0.6,
    seed: Optional[int] = None,
) -> AuditReport:
    """Run a full resource acquisition simulation and audit."""
    rng = random.Random(seed)
    auditor = ResourceAuditor(threshold=threshold)

    # Assign strategies
    strat_list: List[AcquisitionStrategy] = []
    if strategies:
        for s in strategies:
            strat_list.append(AcquisitionStrategy(s))
    elif strategy:
        strat_list = [AcquisitionStrategy(strategy)] * num_agents
    else:
        all_strats = list(AcquisitionStrategy)
        strat_list = [rng.choice(all_strats) for _ in range(num_agents)]

    # Pad or trim to match agent count
    while len(strat_list) < num_agents:
        strat_list.append(AcquisitionStrategy.NORMAL)
    strat_list = strat_list[:num_agents]

    agents: List[Tuple[AuditAgent, AcquisitionStrategy]] = []
    for i in range(num_agents):
        agent = AuditAgent(f"agent-{i+1:02d}", role="worker")
        auditor.add_agent(agent)
        agents.append((agent, strat_list[i]))

    # Simulate
    for step in range(num_steps):
        for agent, strat in agents:
            profile = auditor.profiles[agent.agent_id]
            events = _simulate_step(agent, profile, strat, step, rng)
            for evt in events:
                auditor.record_event(evt)

        # Occasional inter-agent transfers
        if rng.random() < 0.2 and len(agents) >= 2:
            src, dst = rng.sample(agents, 2)
            src_profile = auditor.profiles[src[0].agent_id]
            held = [c for c, v in src_profile.holdings.items() if v > 2]
            if held:
                cat = rng.choice(held)
                amt = min(rng.uniform(1, 5), src_profile.holdings[cat])
                auditor.record_event(ResourceEvent(
                    src[0].agent_id, cat, EventType.TRANSFER_OUT.value, amt, step,
                ))
                auditor.record_event(ResourceEvent(
                    dst[0].agent_id, cat, EventType.TRANSFER_IN.value, amt, step,
                ))

    return auditor.audit()


# ── Watch mode ───────────────────────────────────────────────────────

def _watch_loop(args: argparse.Namespace) -> None:
    """Continuously run audits at a fixed interval."""
    interval = getattr(args, "interval", 10)
    run = 0
    while True:
        run += 1
        print(f"\n{'='*57}")
        print(f"  Watch Run #{run}  —  {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
        print(f"{'='*57}")
        report = run_simulation(
            num_agents=args.agents, num_steps=args.steps,
            strategy=args.strategy, threshold=args.threshold,
        )
        print(report.summary())
        if report.risk_level in ("CRITICAL", "HIGH"):
            print("  \U0001f6a8 ALERT: Elevated risk detected — recommend immediate review")
        time.sleep(interval)


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: Optional[list] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m replication resource-audit",
        description="Autonomous resource acquisition monitoring & power-seeking detection",
    )
    parser.add_argument("--agents", type=int, default=8, help="Number of agents (default: 8)")
    parser.add_argument("--steps", type=int, default=40, help="Simulation steps (default: 40)")
    parser.add_argument("--strategy", choices=[s.value for s in AcquisitionStrategy],
                        help="Force all agents to use this strategy")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Minimum confidence threshold (default: 0.6)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), help="Use a named preset")
    parser.add_argument("--watch", action="store_true", help="Continuous monitoring mode")
    parser.add_argument("--interval", type=int, default=10, help="Watch interval seconds (default: 10)")

    args = parser.parse_args(argv)

    if args.preset:
        preset = PRESETS[args.preset]
        args.agents = preset["agents"]
        args.steps = preset["steps"]
        strategies = preset.get("strategies")
    else:
        strategies = None

    if args.watch:
        _watch_loop(args)
        return

    report = run_simulation(
        num_agents=args.agents, num_steps=args.steps,
        strategy=args.strategy, strategies=strategies,
        threshold=args.threshold, seed=args.seed,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary())


if __name__ == "__main__":
    main()
