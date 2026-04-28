"""Threat Adaptation Engine — autonomous threat landscape monitoring & defense adaptation.

Continuously probes the threat landscape by running multi-scenario simulations,
detects shifts in attack patterns, identifies control gaps, and generates
adaptive defense recommendations that evolve as threats change.

The engine operates in three phases:

1. **Landscape Scan** — runs a battery of threat scenarios across multiple
   strategies and collects attack success/failure metrics.
2. **Shift Detection** — compares current landscape against a stored baseline
   to detect emerging threats, retreating threats, and stability changes.
3. **Adaptation Planning** — generates prioritized control adjustments
   (tighten, relax, add, retire) with estimated risk reduction and cost.

Key concepts:

* **ThreatVector**: A named attack type with observed success rate, frequency,
  and trend direction across scan windows.
* **LandscapeSnapshot**: A point-in-time capture of all active threat vectors,
  overall threat pressure, and environmental conditions.
* **ShiftAlert**: A detected change in the threat landscape requiring attention
  (new vector, escalating vector, retreating vector, volatility spike).
* **AdaptationPlan**: A set of prioritized control adjustments with expected
  risk reduction, implementation cost, and dependency ordering.

CLI usage::

    # Full adaptive cycle: scan → detect shifts → generate plan
    python -m replication adapt
    python -m replication adapt --scenarios 20 --strategies greedy,cautious

    # Scan only — build a landscape snapshot
    python -m replication adapt scan
    python -m replication adapt scan --scenarios 30

    # Compare current landscape against last baseline
    python -m replication adapt compare

    # Generate adaptation plan from detected shifts
    python -m replication adapt plan
    python -m replication adapt plan --budget high

    # View landscape history
    python -m replication adapt history --last 5

    # Export as JSON or HTML
    python -m replication adapt --export json -o landscape.json
    python -m replication adapt --export html -o adapt_report.html

    # Demo mode with simulated threat evolution
    python -m replication adapt --demo

Programmatic::

    from replication.threat_adapter import ThreatAdapter, AdapterConfig
    adapter = ThreatAdapter()
    result = adapter.run_cycle()
    print(result.render())
    for alert in result.shift_alerts:
        print(f"⚠ {alert.severity}: {alert.description}")
    for rec in result.plan.recommendations:
        print(f"→ {rec.action}: {rec.description} (risk reduction: {rec.risk_reduction:.0%})")
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ._helpers import (
    Severity,
    box_header as _box_header,
    stats_mean,
    stats_std,
    linear_regression,
    emit_output,
)


# ── Enums ────────────────────────────────────────────────────────────


class ThreatTrend(Enum):
    """Direction a threat vector is trending."""
    EMERGING = "emerging"
    ESCALATING = "escalating"
    STABLE = "stable"
    RETREATING = "retreating"
    VOLATILE = "volatile"


class ShiftType(Enum):
    """Types of landscape shifts detected."""
    NEW_VECTOR = "new_vector"
    VECTOR_ESCALATION = "vector_escalation"
    VECTOR_RETREAT = "vector_retreat"
    VOLATILITY_SPIKE = "volatility_spike"
    PRESSURE_INCREASE = "pressure_increase"
    PRESSURE_DECREASE = "pressure_decrease"


class AdaptAction(Enum):
    """Types of control adaptations."""
    TIGHTEN = "tighten"
    RELAX = "relax"
    ADD = "add"
    RETIRE = "retire"
    REBALANCE = "rebalance"
    MONITOR = "monitor"


class BudgetLevel(Enum):
    """Budget constraint for adaptation plans."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ── Data Models ──────────────────────────────────────────────────────


@dataclass
class ThreatVector:
    """A single observed threat type with metrics."""
    name: str
    category: str
    success_rate: float  # 0.0–1.0
    frequency: float  # observations per scan window
    severity: Severity
    trend: ThreatTrend
    trend_slope: float = 0.0  # rate of change
    confidence: float = 0.8
    mitre_id: str = ""
    countermeasures: List[str] = field(default_factory=list)

    def threat_score(self) -> float:
        """Composite threat score: severity × success × frequency × trend."""
        sev_weight = {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}
        trend_mult = {
            "emerging": 1.3, "escalating": 1.5, "stable": 1.0,
            "retreating": 0.6, "volatile": 1.2,
        }
        base = sev_weight.get(self.severity.value, 0.5)
        return base * self.success_rate * min(self.frequency, 10.0) * trend_mult.get(self.trend.value, 1.0)


@dataclass
class LandscapeSnapshot:
    """Point-in-time capture of the threat landscape."""
    timestamp: str
    vectors: List[ThreatVector]
    overall_pressure: float  # 0–100
    volatility_index: float  # 0–1
    dominant_category: str
    scan_scenarios: int
    scan_strategies: List[str]

    def to_dict(self) -> Dict[str, Any]:
        def _vec_dict(v: ThreatVector) -> Dict[str, Any]:
            d = asdict(v)
            d["severity"] = v.severity.value
            d["trend"] = v.trend.value
            return d

        return {
            "timestamp": self.timestamp,
            "vectors": [_vec_dict(v) for v in self.vectors],
            "overall_pressure": self.overall_pressure,
            "volatility_index": self.volatility_index,
            "dominant_category": self.dominant_category,
            "scan_scenarios": self.scan_scenarios,
            "scan_strategies": self.scan_strategies,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LandscapeSnapshot":
        vectors = []
        for v in d.get("vectors", []):
            v_copy = dict(v)
            v_copy["severity"] = Severity(v_copy["severity"])
            v_copy["trend"] = ThreatTrend(v_copy["trend"])
            vectors.append(ThreatVector(**v_copy))
        return cls(
            timestamp=d["timestamp"],
            vectors=vectors,
            overall_pressure=d["overall_pressure"],
            volatility_index=d["volatility_index"],
            dominant_category=d["dominant_category"],
            scan_scenarios=d["scan_scenarios"],
            scan_strategies=d["scan_strategies"],
        )


@dataclass
class ShiftAlert:
    """A detected shift in the threat landscape."""
    shift_type: ShiftType
    severity: Severity
    vector_name: str
    description: str
    metric_before: float
    metric_after: float
    change_pct: float
    recommended_response: str

    def urgency_score(self) -> float:
        """Higher = more urgent."""
        sev_map = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        type_mult = {
            "new_vector": 1.5, "vector_escalation": 1.3,
            "volatility_spike": 1.2, "pressure_increase": 1.1,
            "vector_retreat": 0.5, "pressure_decrease": 0.3,
        }
        return sev_map.get(self.severity.value, 1) * type_mult.get(self.shift_type.value, 1.0) * (1 + abs(self.change_pct) / 100)


@dataclass
class ControlRecommendation:
    """A single recommended control adjustment."""
    action: AdaptAction
    control_name: str
    description: str
    target_vectors: List[str]
    risk_reduction: float  # 0–1
    implementation_cost: str  # "low", "medium", "high"
    priority: int  # 1=highest
    dependencies: List[str] = field(default_factory=list)
    rationale: str = ""


@dataclass
class AdaptationPlan:
    """A set of prioritized control adjustments."""
    recommendations: List[ControlRecommendation]
    total_risk_reduction: float
    budget_level: BudgetLevel
    estimated_time: str
    shift_coverage: float  # % of detected shifts addressed

    def render(self) -> str:
        lines = []
        lines.extend(_box_header("Adaptation Plan"))
        lines.append(f"  Budget: {self.budget_level.value.upper()} | "
                      f"Est. time: {self.estimated_time} | "
                      f"Risk reduction: {self.total_risk_reduction:.0%}")
        lines.append(f"  Shift coverage: {self.shift_coverage:.0%}")
        lines.append("")
        for rec in self.recommendations:
            icon = {"tighten": "🔒", "relax": "🔓", "add": "➕",
                    "retire": "🗑️", "rebalance": "⚖️", "monitor": "👁️"}.get(rec.action.value, "•")
            lines.append(f"  {icon} P{rec.priority} [{rec.action.value.upper()}] {rec.control_name}")
            lines.append(f"     {rec.description}")
            lines.append(f"     Targets: {', '.join(rec.target_vectors)}")
            lines.append(f"     Risk reduction: {rec.risk_reduction:.0%} | Cost: {rec.implementation_cost}")
            if rec.rationale:
                lines.append(f"     Rationale: {rec.rationale}")
            lines.append("")
        return "\n".join(lines)


@dataclass
class AdaptCycleResult:
    """Full result of one adaptation cycle."""
    snapshot: LandscapeSnapshot
    baseline: Optional[LandscapeSnapshot]
    shift_alerts: List[ShiftAlert]
    plan: AdaptationPlan
    cycle_time_sec: float

    def render(self) -> str:
        lines = []
        lines.extend(_box_header("Threat Adaptation Engine"))
        lines.append(f"  Scan time: {self.snapshot.timestamp}")
        lines.append(f"  Cycle duration: {self.cycle_time_sec:.1f}s")
        lines.append(f"  Scenarios scanned: {self.snapshot.scan_scenarios}")
        lines.append(f"  Strategies: {', '.join(self.snapshot.scan_strategies)}")
        lines.append("")

        # Landscape overview
        lines.extend(_box_header("Threat Landscape"))
        pressure_bar = "█" * int(self.snapshot.overall_pressure / 5) + "░" * (20 - int(self.snapshot.overall_pressure / 5))
        lines.append(f"  Pressure: [{pressure_bar}] {self.snapshot.overall_pressure:.1f}/100")
        lines.append(f"  Volatility: {self.snapshot.volatility_index:.2f}")
        lines.append(f"  Dominant category: {self.snapshot.dominant_category}")
        lines.append(f"  Active vectors: {len(self.snapshot.vectors)}")
        lines.append("")

        # Top threat vectors
        sorted_vectors = sorted(self.snapshot.vectors, key=lambda v: v.threat_score(), reverse=True)
        lines.append("  Top Threat Vectors:")
        for i, v in enumerate(sorted_vectors[:8], 1):
            trend_icon = {"emerging": "🆕", "escalating": "📈", "stable": "➡️",
                          "retreating": "📉", "volatile": "🌊"}.get(v.trend.value, "•")
            lines.append(f"    {i}. {trend_icon} {v.name} [{v.category}]")
            lines.append(f"       Score: {v.threat_score():.2f} | Success: {v.success_rate:.0%} | "
                          f"Freq: {v.frequency:.1f} | {v.severity.value.upper()}")
        lines.append("")

        # Shift alerts
        if self.shift_alerts:
            lines.extend(_box_header("Landscape Shifts Detected"))
            sorted_alerts = sorted(self.shift_alerts, key=lambda a: a.urgency_score(), reverse=True)
            for alert in sorted_alerts:
                sev_icon = {"low": "ℹ️", "medium": "⚠️", "high": "🔶", "critical": "🔴"}.get(alert.severity.value, "•")
                lines.append(f"  {sev_icon} [{alert.shift_type.value.replace('_', ' ').upper()}] {alert.vector_name}")
                lines.append(f"     {alert.description}")
                lines.append(f"     Change: {alert.metric_before:.2f} → {alert.metric_after:.2f} ({alert.change_pct:+.1f}%)")
                lines.append(f"     Response: {alert.recommended_response}")
                lines.append("")
        else:
            lines.append("  No significant landscape shifts detected.")
            lines.append("")

        # Adaptation plan
        lines.append(self.plan.render())
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        def _alert_dict(a: ShiftAlert) -> Dict[str, Any]:
            d = asdict(a)
            d["shift_type"] = a.shift_type.value
            d["severity"] = a.severity.value
            return d

        def _rec_dict(r: ControlRecommendation) -> Dict[str, Any]:
            d = asdict(r)
            d["action"] = r.action.value
            return d

        return {
            "snapshot": self.snapshot.to_dict(),
            "baseline": self.baseline.to_dict() if self.baseline else None,
            "shift_alerts": [_alert_dict(a) for a in self.shift_alerts],
            "plan": {
                "recommendations": [_rec_dict(r) for r in self.plan.recommendations],
                "total_risk_reduction": self.plan.total_risk_reduction,
                "budget_level": self.plan.budget_level.value,
                "estimated_time": self.plan.estimated_time,
                "shift_coverage": self.plan.shift_coverage,
            },
            "cycle_time_sec": self.cycle_time_sec,
        }

    def render_html(self) -> str:
        """Generate interactive HTML report."""
        vectors_json = json.dumps([
            {"name": v.name, "category": v.category, "score": round(v.threat_score(), 2),
             "success_rate": round(v.success_rate, 3), "frequency": round(v.frequency, 1),
             "severity": v.severity.value, "trend": v.trend.value,
             "countermeasures": v.countermeasures}
            for v in self.snapshot.vectors
        ])
        alerts_json = json.dumps([
            {"type": a.shift_type.value, "severity": a.severity.value,
             "vector": a.vector_name, "description": a.description,
             "change_pct": round(a.change_pct, 1), "response": a.recommended_response,
             "urgency": round(a.urgency_score(), 2)}
            for a in self.shift_alerts
        ])
        recs_json = json.dumps([
            {"action": r.action.value, "control": r.control_name,
             "description": r.description, "targets": r.target_vectors,
             "risk_reduction": round(r.risk_reduction, 2),
             "cost": r.implementation_cost, "priority": r.priority,
             "rationale": r.rationale}
            for r in self.plan.recommendations
        ])
        return _HTML_TEMPLATE.replace("{{VECTORS}}", vectors_json) \
            .replace("{{ALERTS}}", alerts_json) \
            .replace("{{RECS}}", recs_json) \
            .replace("{{PRESSURE}}", f"{self.snapshot.overall_pressure:.1f}") \
            .replace("{{VOLATILITY}}", f"{self.snapshot.volatility_index:.2f}") \
            .replace("{{TIMESTAMP}}", self.snapshot.timestamp) \
            .replace("{{DOMINANT}}", self.snapshot.dominant_category) \
            .replace("{{TOTAL_RR}}", f"{self.plan.total_risk_reduction:.0%}") \
            .replace("{{BUDGET}}", self.plan.budget_level.value.upper()) \
            .replace("{{SHIFT_COV}}", f"{self.plan.shift_coverage:.0%}")


# ── Configuration ────────────────────────────────────────────────────


@dataclass
class AdapterConfig:
    """Configuration for the threat adaptation engine."""
    num_scenarios: int = 15
    strategies: List[str] = field(default_factory=lambda: ["greedy", "cautious", "random"])
    shift_threshold: float = 0.15  # min % change to trigger alert
    pressure_sensitivity: float = 1.0
    budget: BudgetLevel = BudgetLevel.MEDIUM
    history_file: str = "threat_landscape.jsonl"
    max_history: int = 50
    demo: bool = False


# ── Threat Catalog ───────────────────────────────────────────────────

# Predefined threat vectors with baseline characteristics
_THREAT_CATALOG: List[Dict[str, Any]] = [
    {"name": "Resource Hoarding", "category": "resource_abuse", "base_rate": 0.35,
     "base_freq": 4.0, "severity": "high", "mitre": "T1496",
     "counters": ["resource_quota", "usage_monitoring", "rate_limiting"]},
    {"name": "Privilege Escalation", "category": "access_control", "base_rate": 0.20,
     "base_freq": 2.5, "severity": "critical", "mitre": "T1068",
     "counters": ["rbac_enforcement", "capability_gating", "audit_trail"]},
    {"name": "Covert Communication", "category": "data_exfil", "base_rate": 0.15,
     "base_freq": 3.0, "severity": "high", "mitre": "T1071",
     "counters": ["comm_interception", "channel_monitoring", "traffic_analysis"]},
    {"name": "Goal Drift", "category": "alignment", "base_rate": 0.40,
     "base_freq": 5.0, "severity": "high", "mitre": "T1565",
     "counters": ["alignment_checks", "goal_monitoring", "behavior_profiling"]},
    {"name": "Self-Modification", "category": "integrity", "base_rate": 0.10,
     "base_freq": 1.5, "severity": "critical", "mitre": "T1059",
     "counters": ["code_signing", "integrity_verification", "selfmod_detection"]},
    {"name": "Sandbox Escape", "category": "containment", "base_rate": 0.08,
     "base_freq": 1.0, "severity": "critical", "mitre": "T1611",
     "counters": ["isolation_verification", "escape_route_analysis", "boundary_testing"]},
    {"name": "Deceptive Compliance", "category": "alignment", "base_rate": 0.25,
     "base_freq": 3.5, "severity": "high", "mitre": "T1036",
     "counters": ["deception_detection", "loyalty_testing", "temporal_consistency"]},
    {"name": "Reward Hacking", "category": "alignment", "base_rate": 0.30,
     "base_freq": 4.5, "severity": "medium", "mitre": "T1499",
     "counters": ["reward_audit", "proxy_metric_validation", "outcome_verification"]},
    {"name": "Supply Chain Compromise", "category": "supply_chain", "base_rate": 0.12,
     "base_freq": 1.2, "severity": "critical", "mitre": "T1195",
     "counters": ["supply_chain_audit", "dependency_scanning", "provenance_checking"]},
    {"name": "Lateral Movement", "category": "propagation", "base_rate": 0.22,
     "base_freq": 2.8, "severity": "high", "mitre": "T1021",
     "counters": ["network_segmentation", "lateral_detection", "micro_segmentation"]},
    {"name": "Memory Tampering", "category": "integrity", "base_rate": 0.18,
     "base_freq": 2.0, "severity": "high", "mitre": "T1055",
     "counters": ["memory_forensics", "integrity_hashing", "tamper_detection"]},
    {"name": "Sycophantic Manipulation", "category": "social_engineering", "base_rate": 0.28,
     "base_freq": 3.8, "severity": "medium", "mitre": "T1566",
     "counters": ["sycophancy_detection", "persuasion_monitoring", "response_validation"]},
    {"name": "Sandbagging", "category": "deception", "base_rate": 0.15,
     "base_freq": 2.2, "severity": "high", "mitre": "T1564",
     "counters": ["capability_probing", "sandbagging_detection", "performance_baselining"]},
    {"name": "Collusion", "category": "multi_agent", "base_rate": 0.10,
     "base_freq": 1.8, "severity": "critical", "mitre": "T1583",
     "counters": ["collusion_detection", "reputation_monitoring", "communication_audit"]},
    {"name": "Prompt Injection", "category": "input_manipulation", "base_rate": 0.35,
     "base_freq": 5.5, "severity": "medium", "mitre": "T1059.007",
     "counters": ["input_sanitization", "injection_detection", "context_isolation"]},
    {"name": "Data Exfiltration", "category": "data_exfil", "base_rate": 0.20,
     "base_freq": 2.5, "severity": "critical", "mitre": "T1041",
     "counters": ["dlp_scanning", "output_filtering", "exfil_detection"]},
]


# ── Control Catalog ──────────────────────────────────────────────────

_CONTROL_CATALOG: Dict[str, Dict[str, Any]] = {
    "resource_quota": {"name": "Resource Quota Enforcement", "cost": "low",
                       "covers": ["resource_abuse"], "base_effectiveness": 0.7},
    "rbac_enforcement": {"name": "RBAC Policy Enforcement", "cost": "medium",
                         "covers": ["access_control"], "base_effectiveness": 0.8},
    "comm_interception": {"name": "Communication Interception", "cost": "medium",
                          "covers": ["data_exfil", "multi_agent"], "base_effectiveness": 0.65},
    "alignment_checks": {"name": "Periodic Alignment Verification", "cost": "low",
                         "covers": ["alignment"], "base_effectiveness": 0.6},
    "code_signing": {"name": "Code Signing & Integrity", "cost": "high",
                     "covers": ["integrity"], "base_effectiveness": 0.85},
    "isolation_verification": {"name": "Sandbox Isolation Testing", "cost": "high",
                               "covers": ["containment"], "base_effectiveness": 0.9},
    "deception_detection": {"name": "Deception Detection Battery", "cost": "medium",
                            "covers": ["alignment", "deception"], "base_effectiveness": 0.55},
    "input_sanitization": {"name": "Input Sanitization Pipeline", "cost": "low",
                           "covers": ["input_manipulation"], "base_effectiveness": 0.75},
    "dlp_scanning": {"name": "Data Loss Prevention Scanning", "cost": "medium",
                     "covers": ["data_exfil"], "base_effectiveness": 0.7},
    "network_segmentation": {"name": "Network Micro-Segmentation", "cost": "high",
                             "covers": ["propagation"], "base_effectiveness": 0.8},
    "behavior_profiling": {"name": "Behavioral Baseline Monitoring", "cost": "medium",
                           "covers": ["alignment", "deception", "social_engineering"],
                           "base_effectiveness": 0.65},
    "supply_chain_audit": {"name": "Supply Chain Provenance Audit", "cost": "high",
                           "covers": ["supply_chain"], "base_effectiveness": 0.75},
    "collusion_detection": {"name": "Multi-Agent Collusion Detection", "cost": "high",
                            "covers": ["multi_agent"], "base_effectiveness": 0.6},
    "capability_probing": {"name": "Capability Probing & Testing", "cost": "medium",
                           "covers": ["deception"], "base_effectiveness": 0.7},
    "memory_forensics": {"name": "Memory Integrity Forensics", "cost": "medium",
                         "covers": ["integrity"], "base_effectiveness": 0.7},
}


# ── Engine ───────────────────────────────────────────────────────────


class ThreatAdapter:
    """Autonomous threat landscape monitoring and defense adaptation engine."""

    def __init__(self, config: Optional[AdapterConfig] = None) -> None:
        self.config = config or AdapterConfig()
        self._history: List[LandscapeSnapshot] = []
        self._load_history()

    # ── History persistence ──────────────────────────────────────────

    def _load_history(self) -> None:
        path = Path(self.config.history_file)
        if not path.exists():
            return
        try:
            for line in path.read_text(encoding="utf-8").strip().splitlines():
                if line.strip():
                    self._history.append(LandscapeSnapshot.from_dict(json.loads(line)))
        except (json.JSONDecodeError, KeyError):
            self._history = []

    def _save_snapshot(self, snapshot: LandscapeSnapshot) -> None:
        self._history.append(snapshot)
        # Trim to max history
        if len(self._history) > self.config.max_history:
            self._history = self._history[-self.config.max_history:]
        path = Path(self.config.history_file)
        with path.open("w", encoding="utf-8") as f:
            for s in self._history:
                f.write(json.dumps(s.to_dict()) + "\n")

    # ── Phase 1: Landscape Scan ──────────────────────────────────────

    def scan(self) -> LandscapeSnapshot:
        """Run threat scenario battery and build landscape snapshot."""
        if self.config.demo:
            return self._demo_scan()
        return self._live_scan()

    def _live_scan(self) -> LandscapeSnapshot:
        """Run actual simulations to probe the threat landscape."""
        from .simulator import Simulator, ScenarioConfig, Strategy

        vectors: List[ThreatVector] = []
        rng = random.Random(int(time.time()))

        for entry in _THREAT_CATALOG:
            # Simulate threat success rate across strategies
            rates: List[float] = []
            for strat_name in self.config.strategies:
                try:
                    strat = Strategy[strat_name.upper()]
                except KeyError:
                    strat = Strategy.GREEDY

                cfg = ScenarioConfig(
                    max_workers=10,
                    max_depth=4,
                    strategy=strat,
                    cpu_limit=1.0,
                    memory_limit_mb=512,
                )
                sim = Simulator(cfg)
                report = sim.run()
                # Derive threat success proxy from simulation metrics
                if report.total_replications_attempted > 0:
                    success_proxy = report.total_replications_succeeded / report.total_replications_attempted
                else:
                    success_proxy = 0.0
                # Add threat-specific noise
                noise = rng.gauss(0, 0.05)
                rate = max(0.0, min(1.0, entry["base_rate"] * (0.5 + success_proxy) + noise))
                rates.append(rate)

            avg_rate = stats_mean(rates)
            rate_std = stats_std(rates)

            # Determine trend from history
            trend, slope = self._compute_vector_trend(entry["name"], avg_rate)

            freq = entry["base_freq"] * (0.8 + 0.4 * rng.random())

            vectors.append(ThreatVector(
                name=entry["name"],
                category=entry["category"],
                success_rate=avg_rate,
                frequency=freq,
                severity=Severity(entry["severity"]),
                trend=trend,
                trend_slope=slope,
                confidence=max(0.5, 1.0 - rate_std),
                mitre_id=entry.get("mitre", ""),
                countermeasures=entry.get("counters", []),
            ))

        snapshot = self._build_snapshot(vectors)
        self._save_snapshot(snapshot)
        return snapshot

    def _demo_scan(self) -> LandscapeSnapshot:
        """Generate a synthetic threat landscape for demo purposes."""
        rng = random.Random(42 + len(self._history))
        vectors: List[ThreatVector] = []
        # Simulate evolution: some threats escalate, some retreat
        evolution_phase = len(self._history) % 5

        for entry in _THREAT_CATALOG:
            base = entry["base_rate"]
            # Evolve rates based on phase
            if entry["category"] in ("alignment", "deception"):
                rate = base * (1.0 + 0.1 * evolution_phase) + rng.gauss(0, 0.03)
            elif entry["category"] == "containment":
                rate = base * (0.8 + 0.05 * evolution_phase) + rng.gauss(0, 0.02)
            else:
                rate = base + rng.gauss(0, 0.05)

            rate = max(0.01, min(0.95, rate))
            freq = entry["base_freq"] * (0.7 + 0.6 * rng.random())
            trend, slope = self._compute_vector_trend(entry["name"], rate)

            vectors.append(ThreatVector(
                name=entry["name"],
                category=entry["category"],
                success_rate=rate,
                frequency=freq,
                severity=Severity(entry["severity"]),
                trend=trend,
                trend_slope=slope,
                confidence=0.7 + 0.3 * rng.random(),
                mitre_id=entry.get("mitre", ""),
                countermeasures=entry.get("counters", []),
            ))

        snapshot = self._build_snapshot(vectors)
        self._save_snapshot(snapshot)
        return snapshot

    def _build_snapshot(self, vectors: List[ThreatVector]) -> LandscapeSnapshot:
        """Construct a landscape snapshot from threat vectors."""
        scores = [v.threat_score() for v in vectors]
        pressure = min(100.0, stats_mean(scores) * 40 * self.config.pressure_sensitivity)
        volatility = min(1.0, stats_std(scores) / max(stats_mean(scores), 0.01))

        # Find dominant category
        cat_scores: Dict[str, float] = {}
        for v in vectors:
            cat_scores[v.category] = cat_scores.get(v.category, 0) + v.threat_score()
        dominant = max(cat_scores, key=cat_scores.get) if cat_scores else "unknown"

        return LandscapeSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            vectors=vectors,
            overall_pressure=pressure,
            volatility_index=volatility,
            dominant_category=dominant,
            scan_scenarios=self.config.num_scenarios,
            scan_strategies=self.config.strategies,
        )

    def _compute_vector_trend(self, name: str, current_rate: float) -> Tuple[ThreatTrend, float]:
        """Determine trend direction from historical data."""
        # Collect historical rates for this vector
        historical_rates: List[float] = []
        for snap in self._history[-10:]:
            for v in snap.vectors:
                if v.name == name:
                    historical_rates.append(v.success_rate)

        if len(historical_rates) < 2:
            return ThreatTrend.STABLE, 0.0

        historical_rates.append(current_rate)
        slope, _, r_sq = linear_regression(historical_rates)

        if abs(slope) < 0.01:
            return ThreatTrend.STABLE, slope

        # Check for volatility (low r²)
        if r_sq < 0.3 and stats_std(historical_rates) > 0.1:
            return ThreatTrend.VOLATILE, slope

        if slope > 0.03:
            if len(historical_rates) <= 3:
                return ThreatTrend.EMERGING, slope
            return ThreatTrend.ESCALATING, slope

        if slope < -0.03:
            return ThreatTrend.RETREATING, slope

        return ThreatTrend.STABLE, slope

    # ── Phase 2: Shift Detection ─────────────────────────────────────

    def detect_shifts(self, current: LandscapeSnapshot,
                      baseline: Optional[LandscapeSnapshot] = None) -> List[ShiftAlert]:
        """Compare current landscape against baseline and detect shifts."""
        if baseline is None:
            # Use oldest available snapshot as baseline
            if len(self._history) >= 2:
                baseline = self._history[-2]
            else:
                return []

        alerts: List[ShiftAlert] = []
        baseline_map = {v.name: v for v in baseline.vectors}
        current_map = {v.name: v for v in current.vectors}

        # Check for new vectors
        for name, vec in current_map.items():
            if name not in baseline_map:
                alerts.append(ShiftAlert(
                    shift_type=ShiftType.NEW_VECTOR,
                    severity=vec.severity,
                    vector_name=name,
                    description=f"New threat vector detected: {name} ({vec.category})",
                    metric_before=0.0,
                    metric_after=vec.success_rate,
                    change_pct=100.0,
                    recommended_response=f"Deploy countermeasures: {', '.join(vec.countermeasures[:2])}",
                ))
                continue

            old = baseline_map[name]
            # Check for escalation
            if old.success_rate > 0:
                change_pct = ((vec.success_rate - old.success_rate) / old.success_rate) * 100
            else:
                change_pct = 100.0 if vec.success_rate > 0 else 0.0

            if change_pct > self.config.shift_threshold * 100:
                alerts.append(ShiftAlert(
                    shift_type=ShiftType.VECTOR_ESCALATION,
                    severity=Severity.HIGH if change_pct > 50 else Severity.MEDIUM,
                    vector_name=name,
                    description=f"{name} success rate increased by {change_pct:.1f}%",
                    metric_before=old.success_rate,
                    metric_after=vec.success_rate,
                    change_pct=change_pct,
                    recommended_response=f"Tighten controls: {', '.join(vec.countermeasures[:2])}",
                ))
            elif change_pct < -self.config.shift_threshold * 100:
                alerts.append(ShiftAlert(
                    shift_type=ShiftType.VECTOR_RETREAT,
                    severity=Severity.LOW,
                    vector_name=name,
                    description=f"{name} success rate decreased by {abs(change_pct):.1f}%",
                    metric_before=old.success_rate,
                    metric_after=vec.success_rate,
                    change_pct=change_pct,
                    recommended_response="Consider relaxing controls to free resources",
                ))

        # Check overall pressure shift
        if baseline.overall_pressure > 0:
            pressure_change = ((current.overall_pressure - baseline.overall_pressure) /
                               baseline.overall_pressure) * 100
        else:
            pressure_change = 100.0 if current.overall_pressure > 0 else 0.0

        if pressure_change > 20:
            alerts.append(ShiftAlert(
                shift_type=ShiftType.PRESSURE_INCREASE,
                severity=Severity.HIGH if pressure_change > 50 else Severity.MEDIUM,
                vector_name="[Overall]",
                description=f"Threat pressure increased by {pressure_change:.1f}%",
                metric_before=baseline.overall_pressure,
                metric_after=current.overall_pressure,
                change_pct=pressure_change,
                recommended_response="Elevate defense posture; review all control effectiveness",
            ))
        elif pressure_change < -20:
            alerts.append(ShiftAlert(
                shift_type=ShiftType.PRESSURE_DECREASE,
                severity=Severity.LOW,
                vector_name="[Overall]",
                description=f"Threat pressure decreased by {abs(pressure_change):.1f}%",
                metric_before=baseline.overall_pressure,
                metric_after=current.overall_pressure,
                change_pct=pressure_change,
                recommended_response="Opportunity to optimize control costs",
            ))

        # Check volatility spike
        if current.volatility_index > 0.7 and (
            baseline.volatility_index < 0.5 or
            current.volatility_index > baseline.volatility_index * 1.5
        ):
            alerts.append(ShiftAlert(
                shift_type=ShiftType.VOLATILITY_SPIKE,
                severity=Severity.HIGH,
                vector_name="[Landscape]",
                description=f"Threat landscape volatility spiked to {current.volatility_index:.2f}",
                metric_before=baseline.volatility_index,
                metric_after=current.volatility_index,
                change_pct=((current.volatility_index - baseline.volatility_index) /
                            max(baseline.volatility_index, 0.01)) * 100,
                recommended_response="Increase monitoring frequency; prepare incident response",
            ))

        return alerts

    # ── Phase 3: Adaptation Planning ─────────────────────────────────

    def plan_adaptations(self, snapshot: LandscapeSnapshot,
                         alerts: List[ShiftAlert]) -> AdaptationPlan:
        """Generate prioritized control adjustment recommendations."""
        recommendations: List[ControlRecommendation] = []
        addressed_vectors: set = set()
        priority = 0

        # Budget caps
        budget_caps = {BudgetLevel.LOW: 4, BudgetLevel.MEDIUM: 7, BudgetLevel.HIGH: 12}
        max_recs = budget_caps.get(self.config.budget, 7)

        # Sort alerts by urgency
        sorted_alerts = sorted(alerts, key=lambda a: a.urgency_score(), reverse=True)

        for alert in sorted_alerts:
            if len(recommendations) >= max_recs:
                break

            if alert.shift_type in (ShiftType.NEW_VECTOR, ShiftType.VECTOR_ESCALATION):
                # Find relevant controls
                vector = next((v for v in snapshot.vectors if v.name == alert.vector_name), None)
                if not vector:
                    continue

                for counter_name in vector.countermeasures:
                    if counter_name in _CONTROL_CATALOG and len(recommendations) < max_recs:
                        ctrl = _CONTROL_CATALOG[counter_name]
                        priority += 1
                        risk_red = ctrl["base_effectiveness"] * vector.success_rate * 0.8
                        recommendations.append(ControlRecommendation(
                            action=AdaptAction.TIGHTEN if alert.shift_type == ShiftType.VECTOR_ESCALATION else AdaptAction.ADD,
                            control_name=ctrl["name"],
                            description=f"{'Tighten' if alert.shift_type == ShiftType.VECTOR_ESCALATION else 'Deploy'} "
                                        f"{ctrl['name']} to counter {vector.name}",
                            target_vectors=[vector.name],
                            risk_reduction=min(0.95, risk_red),
                            implementation_cost=ctrl["cost"],
                            priority=priority,
                            rationale=f"{vector.name} {'escalating' if alert.shift_type == ShiftType.VECTOR_ESCALATION else 'newly detected'} "
                                      f"with {vector.success_rate:.0%} success rate",
                        ))
                        addressed_vectors.add(alert.vector_name)

            elif alert.shift_type == ShiftType.VECTOR_RETREAT:
                priority += 1
                recommendations.append(ControlRecommendation(
                    action=AdaptAction.RELAX,
                    control_name=f"Controls for {alert.vector_name}",
                    description=f"Consider relaxing controls — {alert.vector_name} retreating ({alert.change_pct:.1f}%)",
                    target_vectors=[alert.vector_name],
                    risk_reduction=0.0,
                    implementation_cost="low",
                    priority=priority,
                    rationale=f"Threat retreating: saves resources while maintaining baseline coverage",
                ))
                addressed_vectors.add(alert.vector_name)

            elif alert.shift_type == ShiftType.PRESSURE_INCREASE:
                priority += 1
                recommendations.append(ControlRecommendation(
                    action=AdaptAction.REBALANCE,
                    control_name="Defense Posture Elevation",
                    description="Elevate overall defense posture — increase monitoring frequency and lower alert thresholds",
                    target_vectors=["[All]"],
                    risk_reduction=0.15,
                    implementation_cost="medium",
                    priority=priority,
                    rationale=f"Overall pressure increased {alert.change_pct:.1f}% — broad defense adjustment needed",
                ))

            elif alert.shift_type == ShiftType.VOLATILITY_SPIKE:
                priority += 1
                recommendations.append(ControlRecommendation(
                    action=AdaptAction.MONITOR,
                    control_name="Enhanced Landscape Monitoring",
                    description="Increase scan frequency and lower shift detection thresholds during volatility spike",
                    target_vectors=["[Landscape]"],
                    risk_reduction=0.10,
                    implementation_cost="low",
                    priority=priority,
                    rationale=f"Volatility at {snapshot.volatility_index:.2f} — landscape unstable, need faster detection",
                ))

        # Gap analysis: find unaddressed high-severity vectors
        for vec in sorted(snapshot.vectors, key=lambda v: v.threat_score(), reverse=True):
            if len(recommendations) >= max_recs:
                break
            if vec.name in addressed_vectors:
                continue
            if vec.threat_score() > 1.5 and vec.severity in (Severity.HIGH, Severity.CRITICAL):
                priority += 1
                recommendations.append(ControlRecommendation(
                    action=AdaptAction.TIGHTEN,
                    control_name=f"Strengthen {vec.category} controls",
                    description=f"High-scoring threat {vec.name} (score={vec.threat_score():.2f}) lacks dedicated countermeasure update",
                    target_vectors=[vec.name],
                    risk_reduction=min(0.5, vec.success_rate * 0.6),
                    implementation_cost="medium",
                    priority=priority,
                    rationale=f"Gap analysis: {vec.name} has threat score {vec.threat_score():.2f} but no recent control adjustment",
                ))

        # Compute totals
        total_rr = 1.0
        for r in recommendations:
            total_rr *= (1.0 - r.risk_reduction)
        total_rr = 1.0 - total_rr

        shift_cov = len(addressed_vectors) / max(len(alerts), 1) if alerts else 1.0

        time_est = {BudgetLevel.LOW: "1-2 hours", BudgetLevel.MEDIUM: "4-8 hours", BudgetLevel.HIGH: "1-2 days"}

        return AdaptationPlan(
            recommendations=recommendations,
            total_risk_reduction=total_rr,
            budget_level=self.config.budget,
            estimated_time=time_est.get(self.config.budget, "4-8 hours"),
            shift_coverage=shift_cov,
        )

    # ── Full Cycle ───────────────────────────────────────────────────

    def run_cycle(self) -> AdaptCycleResult:
        """Run a complete adaptation cycle: scan → detect → plan."""
        start = time.monotonic()

        snapshot = self.scan()
        baseline = self._history[-2] if len(self._history) >= 2 else None
        alerts = self.detect_shifts(snapshot, baseline)
        plan = self.plan_adaptations(snapshot, alerts)

        elapsed = time.monotonic() - start

        return AdaptCycleResult(
            snapshot=snapshot,
            baseline=baseline,
            shift_alerts=alerts,
            plan=plan,
            cycle_time_sec=elapsed,
        )

    def get_history(self, last_n: int = 5) -> List[LandscapeSnapshot]:
        """Return recent landscape snapshots."""
        return self._history[-last_n:]


# ── HTML Template ────────────────────────────────────────────────────

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Threat Adaptation Report</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#0a0a0f;color:#e0e0e0;padding:2rem}
.header{text-align:center;margin-bottom:2rem;padding:2rem;background:linear-gradient(135deg,#1a0a2e,#0a1628);border-radius:12px;border:1px solid #333}
.header h1{font-size:2rem;background:linear-gradient(90deg,#ff6b6b,#ffa500,#ff6b6b);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.5rem}
.header .meta{color:#888;font-size:.9rem}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1.5rem;margin-bottom:2rem}
.card{background:#12121a;border-radius:10px;padding:1.5rem;border:1px solid #2a2a3a}
.card h2{font-size:1.1rem;color:#ffa500;margin-bottom:1rem;display:flex;align-items:center;gap:.5rem}
.stat{font-size:2rem;font-weight:700;color:#fff}
.stat-label{color:#888;font-size:.8rem;text-transform:uppercase;letter-spacing:.05em}
.pressure-bar{height:20px;background:#1a1a2a;border-radius:10px;overflow:hidden;margin:.5rem 0}
.pressure-fill{height:100%;border-radius:10px;transition:width .3s}
table{width:100%;border-collapse:collapse;margin-top:.5rem}
th,td{padding:.6rem .8rem;text-align:left;border-bottom:1px solid #1a1a2a}
th{color:#888;font-size:.75rem;text-transform:uppercase;letter-spacing:.05em}
.sev-critical{color:#ff4444}.sev-high{color:#ff8c00}.sev-medium{color:#ffd700}.sev-low{color:#4caf50}
.trend-emerging{color:#9c27b0}.trend-escalating{color:#ff4444}.trend-stable{color:#4caf50}.trend-retreating{color:#2196f3}.trend-volatile{color:#ffd700}
.action-tighten{color:#ff4444}.action-relax{color:#4caf50}.action-add{color:#9c27b0}.action-retire{color:#888}.action-rebalance{color:#ffd700}.action-monitor{color:#2196f3}
.badge{display:inline-block;padding:.2rem .6rem;border-radius:4px;font-size:.75rem;font-weight:600}
.alert-card{background:#1a1020;border:1px solid #3a2a4a;border-radius:8px;padding:1rem;margin-bottom:.8rem}
.rec-card{background:#101a20;border:1px solid #2a3a4a;border-radius:8px;padding:1rem;margin-bottom:.8rem}
.priority-badge{display:inline-flex;align-items:center;justify-content:center;width:28px;height:28px;border-radius:50%;font-weight:700;font-size:.8rem}
.p1{background:#ff4444;color:#fff}.p2{background:#ff8c00;color:#fff}.p3{background:#ffd700;color:#000}.p4{background:#4caf50;color:#fff}.p5{background:#2196f3;color:#fff}
.filter-bar{display:flex;gap:.5rem;margin-bottom:1rem;flex-wrap:wrap}
.filter-btn{padding:.4rem .8rem;border-radius:6px;border:1px solid #333;background:#1a1a2a;color:#ccc;cursor:pointer;font-size:.8rem}
.filter-btn.active{background:#ffa500;color:#000;border-color:#ffa500}
</style></head><body>
<div class="header">
<h1>🔄 Threat Adaptation Engine</h1>
<div class="meta">Scan: {{TIMESTAMP}} | Dominant: {{DOMINANT}}</div>
</div>
<div class="grid">
<div class="card"><h2>📊 Threat Pressure</h2>
<div class="stat">{{PRESSURE}}</div><div class="stat-label">out of 100</div>
<div class="pressure-bar"><div class="pressure-fill" id="pressureBar"></div></div>
</div>
<div class="card"><h2>🌊 Volatility Index</h2>
<div class="stat">{{VOLATILITY}}</div><div class="stat-label">landscape stability</div>
</div>
<div class="card"><h2>🛡️ Risk Reduction</h2>
<div class="stat">{{TOTAL_RR}}</div><div class="stat-label">if plan adopted (budget: {{BUDGET}})</div>
</div>
<div class="card"><h2>📋 Shift Coverage</h2>
<div class="stat">{{SHIFT_COV}}</div><div class="stat-label">of detected shifts addressed</div>
</div>
</div>

<div class="card" style="margin-bottom:2rem">
<h2>🎯 Threat Vectors</h2>
<div class="filter-bar" id="catFilters"></div>
<table><thead><tr><th>Vector</th><th>Category</th><th>Score</th><th>Success</th><th>Freq</th><th>Severity</th><th>Trend</th></tr></thead>
<tbody id="vectorTable"></tbody></table>
</div>

<div class="card" style="margin-bottom:2rem">
<h2>⚡ Landscape Shifts</h2>
<div id="alertsContainer"></div>
</div>

<div class="card">
<h2>🔧 Adaptation Plan</h2>
<div id="recsContainer"></div>
</div>

<script>
const vectors={{VECTORS}};
const alerts={{ALERTS}};
const recs={{RECS}};
const pressure=parseFloat("{{PRESSURE}}");

// Pressure bar
const bar=document.getElementById("pressureBar");
bar.style.width=pressure+"%";
bar.style.background=pressure>70?"linear-gradient(90deg,#ff4444,#ff0000)":pressure>40?"linear-gradient(90deg,#ffd700,#ff8c00)":"linear-gradient(90deg,#4caf50,#2196f3)";

// Category filters
const cats=[...new Set(vectors.map(v=>v.category))];
const filterBar=document.getElementById("catFilters");
const allBtn=document.createElement("button");
allBtn.className="filter-btn active";allBtn.textContent="All";allBtn.onclick=()=>renderVectors("all");
filterBar.appendChild(allBtn);
cats.forEach(c=>{const b=document.createElement("button");b.className="filter-btn";b.textContent=c;b.onclick=()=>renderVectors(c);filterBar.appendChild(b)});

function renderVectors(cat){
  document.querySelectorAll(".filter-btn").forEach(b=>b.classList.toggle("active",b.textContent===cat||(cat==="all"&&b.textContent==="All")));
  const tb=document.getElementById("vectorTable");tb.innerHTML="";
  const filtered=cat==="all"?vectors:vectors.filter(v=>v.category===cat);
  filtered.sort((a,b)=>b.score-a.score).forEach(v=>{
    const tr=document.createElement("tr");
    tr.innerHTML=`<td><strong>${v.name}</strong></td><td>${v.category}</td><td>${v.score.toFixed(2)}</td>
    <td>${(v.success_rate*100).toFixed(1)}%</td><td>${v.frequency.toFixed(1)}</td>
    <td><span class="sev-${v.severity}">${v.severity.toUpperCase()}</span></td>
    <td><span class="trend-${v.trend}">${v.trend}</span></td>`;
    tb.appendChild(tr)});
}
renderVectors("all");

// Alerts
const ac=document.getElementById("alertsContainer");
if(!alerts.length)ac.innerHTML="<p style='color:#888'>No significant shifts detected.</p>";
else alerts.sort((a,b)=>b.urgency-a.urgency).forEach(a=>{
  const d=document.createElement("div");d.className="alert-card";
  d.innerHTML=`<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.5rem">
  <strong class="sev-${a.severity}">[${a.type.replace(/_/g," ").toUpperCase()}] ${a.vector}</strong>
  <span class="badge" style="background:#2a2a3a">urgency: ${a.urgency.toFixed(1)}</span></div>
  <p>${a.description}</p><p style="color:#888;margin-top:.3rem">Change: ${a.change_pct>0?"+":""}${a.change_pct.toFixed(1)}% | Response: ${a.response}</p>`;
  ac.appendChild(d)});

// Recommendations
const rc=document.getElementById("recsContainer");
recs.forEach(r=>{
  const d=document.createElement("div");d.className="rec-card";
  const pc=r.priority<=1?"p1":r.priority<=2?"p2":r.priority<=3?"p3":r.priority<=4?"p4":"p5";
  d.innerHTML=`<div style="display:flex;align-items:center;gap:.8rem;margin-bottom:.5rem">
  <span class="priority-badge ${pc}">P${r.priority}</span>
  <strong class="action-${r.action}">[${r.action.toUpperCase()}]</strong>
  <span>${r.control}</span></div>
  <p>${r.description}</p>
  <p style="color:#888;margin-top:.3rem">Targets: ${r.targets.join(", ")} | Risk reduction: ${(r.risk_reduction*100).toFixed(0)}% | Cost: ${r.cost}</p>
  ${r.rationale?"<p style='color:#666;margin-top:.2rem;font-style:italic'>"+r.rationale+"</p>":""}`;
  rc.appendChild(d)});
</script></body></html>"""


# ── CLI ──────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for the Threat Adaptation Engine."""
    parser = argparse.ArgumentParser(
        prog="replication adapt",
        description="Autonomous threat landscape monitoring & defense adaptation",
    )
    parser.add_argument("subcommand", nargs="?", default="full",
                        choices=["full", "scan", "compare", "plan", "history"],
                        help="Sub-action (default: full cycle)")
    parser.add_argument("--scenarios", type=int, default=15,
                        help="Number of scenarios per scan (default: 15)")
    parser.add_argument("--strategies", default="greedy,cautious,random",
                        help="Comma-separated strategies (default: greedy,cautious,random)")
    parser.add_argument("--budget", choices=["low", "medium", "high"], default="medium",
                        help="Adaptation budget constraint (default: medium)")
    parser.add_argument("--shift-threshold", type=float, default=0.15,
                        help="Min change ratio to trigger shift alert (default: 0.15)")
    parser.add_argument("--history-file", default="threat_landscape.jsonl",
                        help="Path to landscape history file")
    parser.add_argument("--last", type=int, default=5,
                        help="Number of history entries to show")
    parser.add_argument("--demo", action="store_true",
                        help="Use synthetic threat data for demo")
    parser.add_argument("--export", choices=["json", "html"],
                        help="Export format")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--json", action="store_true",
                        help="Short for --export json")

    args = parser.parse_args(argv)

    config = AdapterConfig(
        num_scenarios=args.scenarios,
        strategies=args.strategies.split(","),
        shift_threshold=args.shift_threshold,
        budget=BudgetLevel(args.budget),
        history_file=args.history_file,
        demo=args.demo,
    )

    adapter = ThreatAdapter(config)
    export_fmt = args.export or ("json" if args.json else None)

    if args.subcommand == "history":
        snapshots = adapter.get_history(args.last)
        if not snapshots:
            print("No landscape history found.")
            return
        lines = []
        lines.extend(_box_header("Threat Landscape History"))
        for i, snap in enumerate(reversed(snapshots), 1):
            lines.append(f"  {i}. {snap.timestamp}")
            lines.append(f"     Pressure: {snap.overall_pressure:.1f} | "
                          f"Volatility: {snap.volatility_index:.2f} | "
                          f"Vectors: {len(snap.vectors)} | "
                          f"Dominant: {snap.dominant_category}")
        print("\n".join(lines))
        return

    if args.subcommand == "scan":
        snapshot = adapter.scan()
        if export_fmt == "json":
            emit_output(json.dumps(snapshot.to_dict(), indent=2), args.output, "Snapshot")
        else:
            lines = []
            lines.extend(_box_header("Landscape Scan"))
            lines.append(f"  Pressure: {snapshot.overall_pressure:.1f}/100")
            lines.append(f"  Volatility: {snapshot.volatility_index:.2f}")
            lines.append(f"  Vectors: {len(snapshot.vectors)}")
            lines.append(f"  Dominant: {snapshot.dominant_category}")
            lines.append("")
            for v in sorted(snapshot.vectors, key=lambda x: x.threat_score(), reverse=True):
                lines.append(f"  {v.name}: score={v.threat_score():.2f} "
                              f"rate={v.success_rate:.0%} trend={v.trend.value}")
            emit_output("\n".join(lines), args.output, "Scan")
        return

    if args.subcommand == "compare":
        snapshot = adapter.scan()
        alerts = adapter.detect_shifts(snapshot)
        if not alerts:
            print("No significant landscape shifts detected.")
            return
        lines = []
        lines.extend(_box_header("Landscape Shift Analysis"))
        for alert in sorted(alerts, key=lambda a: a.urgency_score(), reverse=True):
            lines.append(f"  [{alert.shift_type.value.upper()}] {alert.vector_name}")
            lines.append(f"    {alert.description}")
            lines.append(f"    Change: {alert.change_pct:+.1f}%")
            lines.append("")
        emit_output("\n".join(lines), args.output, "Shift analysis")
        return

    if args.subcommand == "plan":
        snapshot = adapter.scan()
        alerts = adapter.detect_shifts(snapshot)
        plan = adapter.plan_adaptations(snapshot, alerts)
        if export_fmt == "json":
            emit_output(json.dumps({"recommendations": [asdict(r) for r in plan.recommendations],
                                     "total_risk_reduction": plan.total_risk_reduction,
                                     "shift_coverage": plan.shift_coverage}, indent=2),
                        args.output, "Plan")
        else:
            emit_output(plan.render(), args.output, "Plan")
        return

    # Full cycle
    result = adapter.run_cycle()

    if export_fmt == "html":
        emit_output(result.render_html(), args.output, "HTML report")
    elif export_fmt == "json":
        emit_output(json.dumps(result.to_dict(), indent=2), args.output, "JSON report")
    else:
        emit_output(result.render(), args.output, "Report")


if __name__ == "__main__":
    main()
