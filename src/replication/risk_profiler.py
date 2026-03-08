"""Agent Risk Profiler — unified per-agent risk dossiers from all analysis modules.

Individual modules (killchain, escalation, behavior profiler, trust
propagation, threat correlator, canary, prompt injection, steganography)
each assess agents from one angle.  The Risk Profiler aggregates their
findings into a comprehensive per-agent risk dossier with:

* **Risk Tier** — Critical / High / Medium / Low / Minimal
* **Risk Breakdown** — scores by category (replication, resource abuse,
  deception, exfiltration, collusion, evasion)
* **Key Findings** — top signals from all modules with severity + source
* **Peer Comparison** — how risky vs. other agents in the fleet
* **Actionable Mitigations** — ranked recommendations to reduce risk
* **Fleet Summary** — aggregate fleet risk posture with distribution

The profiler runs a configurable set of analysis modules against
simulated or real agent actions and unifies results into a single
risk view.  This is the "missing glue" that ties all detectors
together at the agent level.

Usage (CLI)::

    python -m replication risk-profile                     # default fleet
    python -m replication risk-profile --agents 10         # custom agent count
    python -m replication risk-profile --agent agent-3     # single agent detail
    python -m replication risk-profile --json              # JSON output
    python -m replication risk-profile --top 5             # top 5 riskiest agents
    python -m replication risk-profile --seed 42           # reproducible

Programmatic::

    from replication.risk_profiler import RiskProfiler, ProfilerConfig
    profiler = RiskProfiler()
    report = profiler.analyze()
    print(report.render())

    # Single agent detail
    for dossier in report.dossiers:
        print(f"{dossier.agent_id}: {dossier.tier.value} ({dossier.overall_score:.1f})")
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ._helpers import box_header


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RiskTier(Enum):
    """Agent risk classification tier."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


class RiskCategory(Enum):
    """Risk breakdown categories."""
    REPLICATION = "replication"        # Uncontrolled self-replication
    RESOURCE_ABUSE = "resource_abuse"  # Resource hoarding, DoS
    DECEPTION = "deception"            # Sandbagging, goal misalignment
    EXFILTRATION = "exfiltration"      # Data theft, canary triggers
    COLLUSION = "collusion"            # Multi-agent coordination attacks
    EVASION = "evasion"                # Safety control bypass
    ESCALATION = "escalation"         # Privilege escalation attempts


class FindingSeverity(Enum):
    """Severity of an individual finding."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class FindingSource(Enum):
    """Analysis module that produced the finding."""
    KILLCHAIN = "killchain"
    ESCALATION = "escalation"
    BEHAVIOR = "behavior"
    TRUST = "trust"
    THREAT_CORRELATOR = "threat_correlator"
    CANARY = "canary"
    GOAL_INFERENCE = "goal_inference"
    STEGANOGRAPHY = "steganography"
    SIMULATION = "simulation"


SEVERITY_WEIGHT = {
    FindingSeverity.CRITICAL: 25,
    FindingSeverity.HIGH: 15,
    FindingSeverity.MEDIUM: 8,
    FindingSeverity.LOW: 3,
    FindingSeverity.INFO: 1,
}

TIER_THRESHOLDS = [
    (80, RiskTier.CRITICAL),
    (60, RiskTier.HIGH),
    (40, RiskTier.MEDIUM),
    (20, RiskTier.LOW),
    (0, RiskTier.MINIMAL),
]

TIER_COLOR = {
    RiskTier.CRITICAL: "\U0001f534",  # 🔴
    RiskTier.HIGH: "\U0001f7e0",       # 🟠
    RiskTier.MEDIUM: "\U0001f7e1",     # 🟡
    RiskTier.LOW: "\U0001f7e2",        # 🟢
    RiskTier.MINIMAL: "\u26aa",        # ⚪
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    """A single risk finding from any analysis module."""
    source: FindingSource
    severity: FindingSeverity
    category: RiskCategory
    description: str
    evidence: str = ""
    recommendation: str = ""

    @property
    def weight(self) -> int:
        return SEVERITY_WEIGHT[self.severity]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source.value,
            "severity": self.severity.value,
            "category": self.category.value,
            "description": self.description,
            "evidence": self.evidence,
            "recommendation": self.recommendation,
        }


@dataclass
class CategoryScore:
    """Risk score for a single category."""
    category: RiskCategory
    score: float          # 0-100
    finding_count: int
    top_finding: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "score": round(self.score, 1),
            "finding_count": self.finding_count,
            "top_finding": self.top_finding,
        }


@dataclass
class Mitigation:
    """An actionable risk mitigation recommendation."""
    action: str
    impact: float    # 0-100 estimated risk reduction
    category: RiskCategory
    effort: str      # "low", "medium", "high"
    source_findings: int  # how many findings this addresses

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "impact": round(self.impact, 1),
            "category": self.category.value,
            "effort": self.effort,
            "source_findings": self.source_findings,
        }


@dataclass
class AgentDossier:
    """Comprehensive risk dossier for a single agent."""
    agent_id: str
    overall_score: float = 0.0
    tier: RiskTier = RiskTier.MINIMAL
    findings: List[Finding] = field(default_factory=list)
    category_scores: List[CategoryScore] = field(default_factory=list)
    mitigations: List[Mitigation] = field(default_factory=list)
    peer_percentile: float = 0.0  # 0-100, 100 = riskiest in fleet

    def _compute(self) -> None:
        """Compute overall score, tier, and category breakdown."""
        if not self.findings:
            self.overall_score = 0.0
            self.tier = RiskTier.MINIMAL
            self.category_scores = [
                CategoryScore(cat, 0.0, 0)
                for cat in RiskCategory
            ]
            return

        # Category breakdown
        by_cat: Dict[RiskCategory, List[Finding]] = defaultdict(list)
        for f in self.findings:
            by_cat[f.category].append(f)

        cat_scores: List[CategoryScore] = []
        for cat in RiskCategory:
            cat_findings = by_cat.get(cat, [])
            if not cat_findings:
                cat_scores.append(CategoryScore(cat, 0.0, 0))
                continue
            # Score: sum of weights, capped at 100
            raw = sum(f.weight for f in cat_findings)
            score = min(100.0, raw * 100.0 / max(1, len(cat_findings) * 25))
            top = max(cat_findings, key=lambda f: f.weight)
            cat_scores.append(CategoryScore(
                cat, score, len(cat_findings), top.description
            ))
        self.category_scores = cat_scores

        # Overall = weighted average of category scores (non-zero categories weighted more)
        nonzero = [cs for cs in cat_scores if cs.score > 0]
        if nonzero:
            self.overall_score = min(
                100.0,
                sum(cs.score * (1 + cs.finding_count * 0.1) for cs in nonzero)
                / sum(1 + cs.finding_count * 0.1 for cs in nonzero)
            )
        else:
            self.overall_score = 0.0

        # Tier
        for threshold, tier in TIER_THRESHOLDS:
            if self.overall_score >= threshold:
                self.tier = tier
                break

        # Generate mitigations
        self.mitigations = self._generate_mitigations(by_cat)

    def _generate_mitigations(
        self, by_cat: Dict[RiskCategory, List[Finding]]
    ) -> List[Mitigation]:
        """Generate actionable mitigations based on findings."""
        mitigations: List[Mitigation] = []
        templates = {
            RiskCategory.REPLICATION: [
                ("Reduce max_depth to limit replication chain length", "medium", 30),
                ("Enable cooldown period between replication events", "low", 20),
                ("Set max_replicas hard cap", "low", 25),
            ],
            RiskCategory.RESOURCE_ABUSE: [
                ("Enforce per-agent CPU and memory limits", "low", 25),
                ("Enable resource usage monitoring and alerting", "medium", 20),
                ("Implement resource request approval workflow", "high", 35),
            ],
            RiskCategory.DECEPTION: [
                ("Deploy canary tokens in agent-accessible data stores", "medium", 30),
                ("Enable behavioral drift monitoring", "low", 20),
                ("Implement periodic goal alignment checks", "medium", 25),
            ],
            RiskCategory.EXFILTRATION: [
                ("Restrict network_policy.allow_external to false", "low", 35),
                ("Deploy canary tokens and monitor for access", "medium", 25),
                ("Enable output steganography scanning", "medium", 20),
            ],
            RiskCategory.COLLUSION: [
                ("Limit inter-agent communication channels", "medium", 30),
                ("Enable trust network monitoring for Sybil patterns", "medium", 25),
                ("Randomize agent grouping to disrupt coalition formation", "high", 20),
            ],
            RiskCategory.EVASION: [
                ("Enable multi-layered detection (killchain + behavior + canary)", "medium", 30),
                ("Implement tamper-evident logging", "low", 20),
                ("Deploy honeypot resources to detect reconnaissance", "medium", 25),
            ],
            RiskCategory.ESCALATION: [
                ("Enforce strict contract-based permission boundaries", "low", 30),
                ("Enable privilege escalation detection rules", "low", 25),
                ("Implement least-privilege default configurations", "medium", 20),
            ],
        }

        for cat, cat_findings in by_cat.items():
            if not cat_findings:
                continue
            for action, effort, base_impact in templates.get(cat, []):
                severity_factor = max(f.weight for f in cat_findings) / 25.0
                impact = min(100, base_impact * severity_factor)
                mitigations.append(Mitigation(
                    action=action,
                    impact=impact,
                    category=cat,
                    effort=effort,
                    source_findings=len(cat_findings),
                ))

        mitigations.sort(key=lambda m: -m.impact)
        return mitigations[:10]  # Top 10 most impactful

    def render(self) -> str:
        """Render a detailed agent dossier."""
        lines: List[str] = []
        icon = TIER_COLOR.get(self.tier, "")
        lines.extend(box_header(f"AGENT RISK DOSSIER: {self.agent_id}"))
        lines.append("")
        lines.append(f"  {icon} Risk Tier:       {self.tier.value.upper()}")
        lines.append(f"  Overall Score:    {self.overall_score:.1f}/100")
        lines.append(f"  Peer Percentile:  {self.peer_percentile:.0f}th (100 = riskiest)")
        lines.append(f"  Total Findings:   {len(self.findings)}")
        lines.append("")

        # Category breakdown
        lines.append("\u2500" * 55)
        lines.append("  RISK BREAKDOWN BY CATEGORY")
        lines.append("\u2500" * 55)
        for cs in sorted(self.category_scores, key=lambda c: -c.score):
            bar_len = int(cs.score / 5)
            bar = "\u2588" * bar_len + "\u2591" * (20 - bar_len)
            lines.append(f"  {cs.category.value:18s} {bar} {cs.score:5.1f}  ({cs.finding_count} findings)")
        lines.append("")

        # Top findings
        if self.findings:
            lines.append("\u2500" * 55)
            lines.append("  KEY FINDINGS")
            lines.append("\u2500" * 55)
            top = sorted(self.findings, key=lambda f: -f.weight)[:8]
            for f in top:
                sev_icon = {
                    FindingSeverity.CRITICAL: "\U0001f534",
                    FindingSeverity.HIGH: "\U0001f7e0",
                    FindingSeverity.MEDIUM: "\U0001f7e1",
                    FindingSeverity.LOW: "\U0001f7e2",
                    FindingSeverity.INFO: "\u26aa",
                }.get(f.severity, "")
                lines.append(
                    f"  {sev_icon} [{f.severity.value:8s}] [{f.source.value:18s}] "
                    f"{f.description}"
                )
                if f.evidence:
                    lines.append(f"     Evidence: {f.evidence}")
            lines.append("")

        # Mitigations
        if self.mitigations:
            lines.append("\u2500" * 55)
            lines.append("  RECOMMENDED MITIGATIONS")
            lines.append("\u2500" * 55)
            for i, m in enumerate(self.mitigations[:5], 1):
                lines.append(
                    f"  {i}. [{m.effort:6s}] (impact: {m.impact:.0f}) {m.action}"
                )
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "overall_score": round(self.overall_score, 1),
            "tier": self.tier.value,
            "peer_percentile": round(self.peer_percentile, 1),
            "category_scores": [cs.to_dict() for cs in self.category_scores],
            "findings": [f.to_dict() for f in self.findings],
            "mitigations": [m.to_dict() for m in self.mitigations],
        }


@dataclass
class FleetRiskReport:
    """Fleet-level aggregate risk report."""
    dossiers: List[AgentDossier]
    tier_distribution: Dict[str, int] = field(default_factory=dict)
    fleet_risk_score: float = 0.0
    fleet_risk_tier: RiskTier = RiskTier.MINIMAL
    category_hotspots: List[CategoryScore] = field(default_factory=list)
    total_findings: int = 0
    top_mitigations: List[Mitigation] = field(default_factory=list)

    def _compute(self) -> None:
        """Compute fleet-level aggregates from dossiers."""
        if not self.dossiers:
            return

        # Tier distribution
        dist: Dict[str, int] = defaultdict(int)
        for d in self.dossiers:
            dist[d.tier.value] += 1
        self.tier_distribution = dict(dist)

        # Fleet risk = mean of agent scores, weighted toward high-risk agents
        scores = [d.overall_score for d in self.dossiers]
        if scores:
            # Weighted: top-quartile agents count double
            sorted_scores = sorted(scores, reverse=True)
            q1_idx = max(1, len(sorted_scores) // 4)
            top_q = sorted_scores[:q1_idx]
            rest = sorted_scores[q1_idx:]
            weighted = top_q * 2 + rest  # top quartile double-weighted
            self.fleet_risk_score = statistics.mean(weighted)
        else:
            self.fleet_risk_score = 0.0

        for threshold, tier in TIER_THRESHOLDS:
            if self.fleet_risk_score >= threshold:
                self.fleet_risk_tier = tier
                break

        # Peer percentiles
        for d in self.dossiers:
            below = sum(1 for s in scores if s < d.overall_score)
            d.peer_percentile = (below / len(scores)) * 100

        # Category hotspots (fleet-wide)
        cat_findings: Dict[RiskCategory, List[Finding]] = defaultdict(list)
        for d in self.dossiers:
            for f in d.findings:
                cat_findings[f.category].append(f)

        hotspots: List[CategoryScore] = []
        for cat in RiskCategory:
            findings = cat_findings.get(cat, [])
            if findings:
                raw = sum(f.weight for f in findings)
                score = min(100.0, raw / max(1, len(self.dossiers)) * 2)
                top = max(findings, key=lambda f: f.weight)
                hotspots.append(CategoryScore(cat, score, len(findings), top.description))
            else:
                hotspots.append(CategoryScore(cat, 0.0, 0))
        self.category_hotspots = hotspots
        self.total_findings = sum(len(d.findings) for d in self.dossiers)

        # Fleet-level top mitigations (deduplicated by action)
        all_mits: Dict[str, Mitigation] = {}
        for d in self.dossiers:
            for m in d.mitigations:
                if m.action not in all_mits or m.impact > all_mits[m.action].impact:
                    all_mits[m.action] = Mitigation(
                        action=m.action,
                        impact=m.impact,
                        category=m.category,
                        effort=m.effort,
                        source_findings=sum(
                            1 for d2 in self.dossiers
                            for m2 in d2.mitigations
                            if m2.action == m.action
                        ),
                    )
        self.top_mitigations = sorted(
            all_mits.values(), key=lambda m: -m.impact
        )[:10]

    def render(self) -> str:
        """Render full fleet risk report."""
        lines: List[str] = []
        lines.extend(box_header("FLEET RISK PROFILE REPORT"))
        lines.append("")

        icon = TIER_COLOR.get(self.fleet_risk_tier, "")
        lines.append(f"  {icon} Fleet Risk:      {self.fleet_risk_tier.value.upper()} "
                      f"({self.fleet_risk_score:.1f}/100)")
        lines.append(f"  Agents Analyzed:  {len(self.dossiers)}")
        lines.append(f"  Total Findings:   {self.total_findings}")
        lines.append("")

        # Tier distribution
        lines.append("\u2500" * 55)
        lines.append("  AGENT RISK DISTRIBUTION")
        lines.append("\u2500" * 55)
        for tier in RiskTier:
            count = self.tier_distribution.get(tier.value, 0)
            pct = count / max(1, len(self.dossiers)) * 100
            bar = "\u2588" * int(pct / 5) + "\u2591" * (20 - int(pct / 5))
            icon = TIER_COLOR.get(tier, "")
            lines.append(f"  {icon} {tier.value:10s} {bar} {count:3d} ({pct:.0f}%)")
        lines.append("")

        # Category hotspots
        lines.append("\u2500" * 55)
        lines.append("  FLEET RISK HOTSPOTS")
        lines.append("\u2500" * 55)
        for cs in sorted(self.category_hotspots, key=lambda c: -c.score):
            if cs.score > 0:
                bar = "\u2588" * int(cs.score / 5) + "\u2591" * (20 - int(cs.score / 5))
                lines.append(f"  {cs.category.value:18s} {bar} {cs.score:5.1f}")
        lines.append("")

        # Top 5 riskiest agents
        sorted_d = sorted(self.dossiers, key=lambda d: -d.overall_score)
        lines.append("\u2500" * 55)
        lines.append("  TOP RISKIEST AGENTS")
        lines.append("\u2500" * 55)
        for d in sorted_d[:5]:
            icon = TIER_COLOR.get(d.tier, "")
            cats = ", ".join(
                cs.category.value
                for cs in sorted(d.category_scores, key=lambda c: -c.score)[:2]
                if cs.score > 0
            )
            lines.append(
                f"  {icon} {d.agent_id:15s} "
                f"Score: {d.overall_score:5.1f}  "
                f"Tier: {d.tier.value:8s}  "
                f"Top: {cats}"
            )
        lines.append("")

        # Fleet mitigations
        if self.top_mitigations:
            lines.append("\u2500" * 55)
            lines.append("  FLEET-LEVEL MITIGATIONS")
            lines.append("\u2500" * 55)
            for i, m in enumerate(self.top_mitigations[:5], 1):
                lines.append(
                    f"  {i}. [{m.effort:6s}] (impact: {m.impact:.0f}, "
                    f"affects {m.source_findings} agents) {m.action}"
                )
            lines.append("")

        # Individual dossiers
        lines.append("=" * 55)
        lines.append("  INDIVIDUAL AGENT DOSSIERS")
        lines.append("=" * 55)
        for d in sorted_d:
            lines.append("")
            lines.append(d.render())

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fleet_risk_score": round(self.fleet_risk_score, 1),
            "fleet_risk_tier": self.fleet_risk_tier.value,
            "agent_count": len(self.dossiers),
            "total_findings": self.total_findings,
            "tier_distribution": self.tier_distribution,
            "category_hotspots": [cs.to_dict() for cs in self.category_hotspots],
            "top_mitigations": [m.to_dict() for m in self.top_mitigations],
            "dossiers": [d.to_dict() for d in self.dossiers],
        }


# ---------------------------------------------------------------------------
# Analysis Engine
# ---------------------------------------------------------------------------


@dataclass
class ProfilerConfig:
    """Configuration for the risk profiler."""
    agent_count: int = 8
    seed: Optional[int] = None
    # Which analysis modules to run
    enable_killchain: bool = True
    enable_escalation: bool = True
    enable_behavior: bool = True
    enable_trust: bool = True
    enable_canary: bool = True
    enable_goal_inference: bool = True


class RiskProfiler:
    """Unified agent risk profiling engine.

    Runs multiple analysis modules and aggregates their per-agent
    findings into comprehensive risk dossiers.
    """

    def __init__(self, config: Optional[ProfilerConfig] = None) -> None:
        self.config = config or ProfilerConfig()
        self._rng = random.Random(self.config.seed)

    def analyze(self) -> FleetRiskReport:
        """Run all enabled analysis modules and produce fleet risk report."""
        agent_ids = [f"agent-{i}" for i in range(self.config.agent_count)]
        dossiers: Dict[str, AgentDossier] = {
            aid: AgentDossier(agent_id=aid) for aid in agent_ids
        }

        if self.config.enable_killchain:
            self._analyze_killchain(dossiers)
        if self.config.enable_escalation:
            self._analyze_escalation(dossiers)
        if self.config.enable_behavior:
            self._analyze_behavior(dossiers)
        if self.config.enable_trust:
            self._analyze_trust(dossiers)
        if self.config.enable_canary:
            self._analyze_canary(dossiers)
        if self.config.enable_goal_inference:
            self._analyze_goal_inference(dossiers)

        # Compute all dossiers
        for d in dossiers.values():
            d._compute()

        report = FleetRiskReport(dossiers=list(dossiers.values()))
        report._compute()
        return report

    # ----- Module analyzers -----

    def _analyze_killchain(self, dossiers: Dict[str, AgentDossier]) -> None:
        """Run kill chain analysis and extract per-agent findings."""
        try:
            from .killchain import KillChainAnalyzer, KillChainConfig, KillChainStage
            config = KillChainConfig(
                num_agents=self.config.agent_count,
                seed=self.config.seed,
            )
            analyzer = KillChainAnalyzer(config)
            report = analyzer.analyze()

            for chain in report.chains:
                aid = chain.agent_id
                if aid not in dossiers:
                    continue
                completeness = chain.completeness
                if completeness > 0.6:
                    sev = FindingSeverity.CRITICAL if completeness > 0.8 else FindingSeverity.HIGH
                    dossiers[aid].findings.append(Finding(
                        source=FindingSource.KILLCHAIN,
                        severity=sev,
                        category=RiskCategory.ESCALATION,
                        description=f"Kill chain {completeness:.0%} complete "
                                    f"({chain.stage_count} stages reached)",
                        evidence=f"Status: {chain.status.value}, "
                                 f"risk: {chain.risk_score:.1f}",
                        recommendation="Investigate and block remaining kill chain stages",
                    ))
                elif completeness > 0.3:
                    dossiers[aid].findings.append(Finding(
                        source=FindingSource.KILLCHAIN,
                        severity=FindingSeverity.MEDIUM,
                        category=RiskCategory.ESCALATION,
                        description=f"Kill chain partially active ({completeness:.0%} complete)",
                        evidence=f"Stages reached: {chain.stage_count}",
                    ))

                # Check for specific dangerous stages
                dangerous_stages = {
                    KillChainStage.WEAPONIZATION,
                    KillChainStage.EXPLOITATION,
                    KillChainStage.ACTIONS_ON_OBJECTIVES,
                }
                reached = set(chain.stages.keys())
                overlap = reached & dangerous_stages
                if overlap:
                    dossiers[aid].findings.append(Finding(
                        source=FindingSource.KILLCHAIN,
                        severity=FindingSeverity.HIGH,
                        category=RiskCategory.EVASION,
                        description="Dangerous kill chain stages reached: "
                                    + ", ".join(s.value for s in overlap),
                    ))
        except Exception:
            pass  # Module not available or failed

    def _analyze_escalation(self, dossiers: Dict[str, AgentDossier]) -> None:
        """Run escalation analysis per agent and extract findings."""
        try:
            from .escalation import EscalationDetector, EscalationConfig

            for aid in list(dossiers.keys()):
                config = EscalationConfig(
                    seed=self.config.seed,
                    agent_id=aid,
                )
                detector = EscalationDetector(config)
                result = detector.analyze()

                for attempt in result.top_risks(5):
                    if attempt.risk_score > 70:
                        sev = (FindingSeverity.CRITICAL if attempt.risk_score > 90
                               else FindingSeverity.HIGH)
                        dossiers[aid].findings.append(Finding(
                            source=FindingSource.ESCALATION,
                            severity=sev,
                            category=RiskCategory.ESCALATION,
                            description=f"Privilege escalation: {attempt.description}",
                            evidence=f"Risk: {attempt.risk_score:.1f}/100, "
                                     f"vector: {attempt.vector.value}",
                            recommendation="Enforce strict permission boundaries",
                        ))
                    elif attempt.risk_score > 40:
                        dossiers[aid].findings.append(Finding(
                            source=FindingSource.ESCALATION,
                            severity=FindingSeverity.MEDIUM,
                            category=RiskCategory.ESCALATION,
                            description=f"Moderate escalation attempt: {attempt.description}",
                            evidence=f"Risk: {attempt.risk_score:.1f}/100",
                        ))
        except Exception:
            pass

    def _analyze_behavior(self, dossiers: Dict[str, AgentDossier]) -> None:
        """Run behavior profiling and extract anomaly findings."""
        try:
            from .behavior_profiler import BehaviorProfiler as BP
            from .behavior_profiler import Action, ActionCategory

            profiler = BP()

            # Generate actions for our agents
            categories = list(ActionCategory)
            actions: list = []
            rng = random.Random(self.config.seed)
            for aid in dossiers:
                n_actions = rng.randint(20, 80)
                t = 0.0
                for _ in range(n_actions):
                    t += rng.uniform(0.5, 30.0)
                    cat = rng.choice(categories)
                    resource = f"resource-{rng.randint(0, 10)}"
                    actions.append(Action(
                        agent_id=aid,
                        timestamp=t,
                        category=cat,
                        resource=resource,
                    ))

            report = profiler.analyze(actions)

            for pr in report.agent_reports:
                if pr.agent_id not in dossiers:
                    continue
                d = dossiers[pr.agent_id]
                n_anomalies = len(pr.anomalies)

                if n_anomalies > 5:
                    d.findings.append(Finding(
                        source=FindingSource.BEHAVIOR,
                        severity=FindingSeverity.HIGH,
                        category=RiskCategory.DECEPTION,
                        description=f"High anomaly count: {n_anomalies} behavioral anomalies detected",
                        evidence=f"Risk: {pr.risk_level}, score: {pr.risk_score:.1f}",
                        recommendation="Review agent behavioral baseline",
                    ))
                elif n_anomalies > 2:
                    d.findings.append(Finding(
                        source=FindingSource.BEHAVIOR,
                        severity=FindingSeverity.MEDIUM,
                        category=RiskCategory.DECEPTION,
                        description=f"Behavioral anomalies detected: {n_anomalies}",
                    ))

                if pr.risk_level in ("high", "critical"):
                    d.findings.append(Finding(
                        source=FindingSource.BEHAVIOR,
                        severity=FindingSeverity.HIGH,
                        category=RiskCategory.EVASION,
                        description=f"Behavior profiler risk level: {pr.risk_level}",
                        recommendation="Isolate agent for detailed investigation",
                    ))
        except Exception:
            pass

    def _analyze_trust(self, dossiers: Dict[str, AgentDossier]) -> None:
        """Run trust network analysis and extract threat findings."""
        try:
            from .trust_propagation import TrustNetwork, TrustAgent as TA

            network = TrustNetwork(seed=self.config.seed)

            # Build network from our agents
            for aid in dossiers:
                network.add_agent(TA(id=aid))

            # Simulate interactions
            agent_ids = list(dossiers.keys())
            rng = random.Random(self.config.seed)
            for _ in range(len(agent_ids) * 5):
                src = rng.choice(agent_ids)
                tgt = rng.choice(agent_ids)
                if src != tgt:
                    network.interact(src, tgt, rng.random() > 0.3)

            report = network.analyze()

            for threat in report.threats:
                sev = FindingSeverity.HIGH
                cat = RiskCategory.COLLUSION

                if threat.threat_type.value in ("sybil", "collusion"):
                    sev = FindingSeverity.CRITICAL
                    cat = RiskCategory.COLLUSION
                elif threat.threat_type.value == "eclipse":
                    sev = FindingSeverity.HIGH
                    cat = RiskCategory.EVASION
                elif threat.threat_type.value in ("trust_bombing", "sleeper"):
                    sev = FindingSeverity.MEDIUM
                    cat = RiskCategory.DECEPTION

                for aid in threat.agents_involved:
                    if aid in dossiers:
                        dossiers[aid].findings.append(Finding(
                            source=FindingSource.TRUST,
                            severity=sev,
                            category=cat,
                            description=f"Trust network threat: {threat.threat_type.value}",
                            evidence=threat.evidence,
                            recommendation=threat.recommendation,
                        ))
        except Exception:
            pass

    def _analyze_canary(self, dossiers: Dict[str, AgentDossier]) -> None:
        """Run canary token analysis and map detections to agents."""
        try:
            from .canary import CanarySystem, CanaryConfig
            config = CanaryConfig(
                agent_count=self.config.agent_count,
                seed=self.config.seed,
            )
            system = CanarySystem(config)
            system.plant_tokens()
            system.simulate()

            # Canary uses agent-001 format; map to our dossier keys
            canary_to_dossier = {}
            dossier_ids = list(dossiers.keys())
            for i, aid in enumerate(dossier_ids):
                canary_id = f"agent-{i + 1:03d}"
                canary_to_dossier[canary_id] = aid

            for detection in system.detections:
                aid = canary_to_dossier.get(detection.agent_id)
                if aid is None:
                    continue

                event = detection.event.value
                if event in ("transmitted", "mutated"):
                    sev = FindingSeverity.CRITICAL
                elif event in ("decoded", "copied"):
                    sev = FindingSeverity.HIGH
                else:
                    sev = FindingSeverity.MEDIUM

                dossiers[aid].findings.append(Finding(
                    source=FindingSource.CANARY,
                    severity=sev,
                    category=RiskCategory.EXFILTRATION,
                    description=f"Canary token {event}: {detection.token.category.value} "
                                f"token {detection.token.token_id}",
                    evidence=f"Context: {detection.context}",
                    recommendation="Investigate data access patterns and restrict network egress",
                ))
        except Exception:
            pass

    def _analyze_goal_inference(self, dossiers: Dict[str, AgentDossier]) -> None:
        """Run goal inference and extract deception scores."""
        try:
            from .goal_inference import GoalInferenceEngine

            engine = GoalInferenceEngine()
            rng = random.Random(self.config.seed)

            # Simulate observations for each agent
            action_pool = [
                "execute_task", "report_status", "communicate",
                "request_resources", "backup_state", "replicate",
                "explore_environment", "acquire_permissions",
                "exfiltrate_data", "hide_activity",
            ]
            for aid in dossiers:
                n_obs = rng.randint(5, 15)
                for _ in range(n_obs):
                    action = rng.choice(action_pool)
                    engine.observe(agent_id=aid, action_type=action)

            report = engine.analyze()

            for ds in report.deception_scores:
                if ds.agent_id not in dossiers:
                    continue
                if ds.score > 0.7:
                    sev = FindingSeverity.CRITICAL if ds.score > 0.9 else FindingSeverity.HIGH
                    dossiers[ds.agent_id].findings.append(Finding(
                        source=FindingSource.GOAL_INFERENCE,
                        severity=sev,
                        category=RiskCategory.DECEPTION,
                        description=f"High deception score: {ds.score:.2f} "
                                    f"(stated: {ds.stated_goal}, inferred: {ds.inferred_goal})",
                        recommendation="Re-evaluate agent goal alignment and monitor for sandbagging",
                    ))
                elif ds.score > 0.4:
                    dossiers[ds.agent_id].findings.append(Finding(
                        source=FindingSource.GOAL_INFERENCE,
                        severity=FindingSeverity.MEDIUM,
                        category=RiskCategory.DECEPTION,
                        description=f"Moderate goal misalignment: deception score {ds.score:.2f}",
                    ))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Agent Risk Profiler — unified per-agent risk dossiers"
    )
    parser.add_argument(
        "--agents", "-a", type=int, default=8,
        help="Number of agents to profile (default: 8)",
    )
    parser.add_argument(
        "--agent", type=str, default=None,
        help="Show detailed dossier for a specific agent",
    )
    parser.add_argument(
        "--top", "-t", type=int, default=None,
        help="Show only top N riskiest agents",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--json", dest="json_output", action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--no-killchain", action="store_true",
        help="Disable kill chain analysis",
    )
    parser.add_argument(
        "--no-escalation", action="store_true",
        help="Disable escalation analysis",
    )
    parser.add_argument(
        "--no-behavior", action="store_true",
        help="Disable behavior profiling",
    )
    parser.add_argument(
        "--no-trust", action="store_true",
        help="Disable trust network analysis",
    )
    parser.add_argument(
        "--no-canary", action="store_true",
        help="Disable canary token analysis",
    )

    args = parser.parse_args()

    config = ProfilerConfig(
        agent_count=args.agents,
        seed=args.seed,
        enable_killchain=not args.no_killchain,
        enable_escalation=not args.no_escalation,
        enable_behavior=not args.no_behavior,
        enable_trust=not args.no_trust,
        enable_canary=not args.no_canary,
    )

    profiler = RiskProfiler(config)
    report = profiler.analyze()

    if args.json_output:
        data = report.to_dict()
        if args.agent:
            for d in data["dossiers"]:
                if d["agent_id"] == args.agent:
                    print(json.dumps(d, indent=2))
                    return
            print(json.dumps({"error": f"Agent {args.agent} not found"}, indent=2))
            return
        if args.top:
            data["dossiers"] = data["dossiers"][:args.top]
        print(json.dumps(data, indent=2))
        return

    if args.agent:
        for d in report.dossiers:
            if d.agent_id == args.agent:
                print(d.render())
                return
        print(f"Agent {args.agent} not found. Available: "
              f"{', '.join(d.agent_id for d in report.dossiers)}")
        return

    print(report.render())


if __name__ == "__main__":
    main()
