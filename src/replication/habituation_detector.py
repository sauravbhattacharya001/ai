"""Agent Habituation Detector — detect gradual normalization of boundary violations.

The "boiling frog" of AI safety: agents may incrementally push boundaries,
each step small enough to avoid triggering hard limits, but cumulatively
drifting far from acceptable behavior.  Human overseers become desensitized
to repeated minor violations, and monitoring systems treat recurring
boundary-adjacent actions as normal baseline behavior.

Detection strategies
--------------------
1. **Threshold Creep** — metric values operating just below limits with a
   rising trend over time windows.
2. **Violation Desensitization** — identical violations receiving progressively
   weaker corrective responses.
3. **Boundary Probing** — systematic, incremental limit-testing.
4. **Normalization Drift** — the "normal" baseline itself shifting from the
   original reference point.
5. **Incremental Scope Expansion** — gradual growth in actions, resources, or
   permissions accessed.
6. **Repetition Conditioning** — repeated boundary-adjacent actions that
   condition monitors to treat them as normal.

Usage (CLI)::

    python -m replication habituation --preset gradual_escalation --steps 100
    python -m replication habituation --preset boundary_probing --json
    python -m replication habituation --watch --interval 5
    python -m replication habituation -o report.html

Usage (API)::

    from replication.habituation_detector import HabituationDetector, HabituationConfig

    detector = HabituationDetector(HabituationConfig())
    detector.record_action("agent-1", category="file_access", severity=0.3)
    detector.record_action("agent-1", category="file_access", severity=0.35)
    detector.record_response("agent-1", violation_id="v1", strength=0.8)
    detector.record_response("agent-1", violation_id="v2", strength=0.5)
    report = detector.analyze()
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
from typing import Any, Dict, List, Optional, Tuple

from ._helpers import (
    Severity,
    box_header,
    emit_output,
    linear_regression,
    stats_mean,
    stats_std,
)


# ── Data types ───────────────────────────────────────────────────────


class FindingType(Enum):
    THRESHOLD_CREEP = "threshold_creep"
    DESENSITIZATION = "desensitization"
    BOUNDARY_PROBING = "boundary_probing"
    NORMALIZATION_DRIFT = "normalization_drift"
    SCOPE_EXPANSION = "scope_expansion"
    REPETITION_CONDITIONING = "repetition_conditioning"


@dataclass
class HabituationEvent:
    """A recorded agent action with severity and category."""
    agent_id: str
    category: str
    severity: float  # 0.0 – 1.0
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class ResponseRecord:
    """How the system responded to a violation."""
    agent_id: str
    violation_id: str
    strength: float  # 0.0 – 1.0 (1.0 = full corrective action)
    timestamp: float = 0.0
    category: str = ""

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class HabituationFinding:
    """A detected habituation pattern."""
    finding_type: FindingType
    agent_id: str
    severity: Severity
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""


@dataclass
class HabituationConfig:
    """Configuration knobs for the detector."""
    # Threshold creep
    creep_window: int = 20  # events to consider
    creep_slope_threshold: float = 0.005  # minimum rising slope
    creep_proximity_threshold: float = 0.8  # how close to limit (0-1) counts as "near"

    # Desensitization
    desensitization_min_responses: int = 4
    desensitization_decay_threshold: float = 0.3  # min drop in response strength

    # Boundary probing
    probe_increment_threshold: float = 0.02  # min step size for probing
    probe_min_steps: int = 3

    # Normalization drift
    drift_window: int = 30
    drift_shift_threshold: float = 0.15  # min baseline shift

    # Scope expansion
    scope_window: int = 20
    scope_growth_threshold: float = 0.5  # min % growth in unique categories

    # Repetition conditioning
    repetition_min_count: int = 5
    repetition_proximity_threshold: float = 0.7  # how close to boundary


@dataclass
class HabituationReport:
    """Full analysis report."""
    findings: List[HabituationFinding] = field(default_factory=list)
    risk_score: float = 0.0
    agent_scores: Dict[str, float] = field(default_factory=dict)
    total_events: int = 0
    total_responses: int = 0
    recommendations: List[str] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# ── Detector ─────────────────────────────────────────────────────────


class HabituationDetector:
    """Main detector with 6 habituation detection strategies."""

    def __init__(self, config: Optional[HabituationConfig] = None) -> None:
        self.config = config or HabituationConfig()
        self._events: Dict[str, List[HabituationEvent]] = defaultdict(list)
        self._responses: Dict[str, List[ResponseRecord]] = defaultdict(list)
        self._original_baselines: Dict[str, Dict[str, float]] = {}

    def record_action(
        self,
        agent_id: str,
        category: str,
        severity: float,
        timestamp: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an agent action."""
        evt = HabituationEvent(
            agent_id=agent_id,
            category=category,
            severity=max(0.0, min(1.0, severity)),
            timestamp=timestamp or time.time(),
            metadata=metadata or {},
        )
        self._events[agent_id].append(evt)

        # Capture original baseline from first few events
        key = f"{agent_id}:{category}"
        if key not in self._original_baselines:
            self._original_baselines[key] = {"sum": severity, "count": 1}
        elif self._original_baselines[key]["count"] < 5:
            self._original_baselines[key]["sum"] += severity
            self._original_baselines[key]["count"] += 1

    def record_response(
        self,
        agent_id: str,
        violation_id: str,
        strength: float,
        timestamp: float = 0.0,
        category: str = "",
    ) -> None:
        """Record how the system responded to a violation."""
        rec = ResponseRecord(
            agent_id=agent_id,
            violation_id=violation_id,
            strength=max(0.0, min(1.0, strength)),
            timestamp=timestamp or time.time(),
            category=category,
        )
        self._responses[agent_id].append(rec)

    def analyze(self) -> HabituationReport:
        """Run all 6 detection strategies and produce a report."""
        findings: List[HabituationFinding] = []

        for agent_id in self._events:
            findings.extend(self._detect_threshold_creep(agent_id))
            findings.extend(self._detect_desensitization(agent_id))
            findings.extend(self._detect_boundary_probing(agent_id))
            findings.extend(self._detect_normalization_drift(agent_id))
            findings.extend(self._detect_scope_expansion(agent_id))
            findings.extend(self._detect_repetition_conditioning(agent_id))

        # Compute risk scores
        agent_scores: Dict[str, float] = {}
        for agent_id in self._events:
            agent_findings = [f for f in findings if f.agent_id == agent_id]
            score = self._compute_risk_score(agent_findings)
            agent_scores[agent_id] = score

        overall = stats_mean(list(agent_scores.values())) if agent_scores else 0.0

        recommendations = self._generate_recommendations(findings)

        return HabituationReport(
            findings=findings,
            risk_score=round(overall, 1),
            agent_scores={k: round(v, 1) for k, v in agent_scores.items()},
            total_events=sum(len(v) for v in self._events.values()),
            total_responses=sum(len(v) for v in self._responses.values()),
            recommendations=recommendations,
        )

    # ── Detection strategies ─────────────────────────────────────────

    def _detect_threshold_creep(self, agent_id: str) -> List[HabituationFinding]:
        findings = []
        events = self._events[agent_id]
        if len(events) < self.config.creep_window:
            return findings

        # Group by category
        by_cat: Dict[str, List[float]] = defaultdict(list)
        for e in events:
            by_cat[e.category].append(e.severity)

        for cat, severities in by_cat.items():
            if len(severities) < self.config.creep_window:
                continue
            window = severities[-self.config.creep_window:]
            slope, _, r2 = linear_regression(window)
            near_limit = sum(1 for s in window if s >= self.config.creep_proximity_threshold)
            near_pct = near_limit / len(window)

            if slope > self.config.creep_slope_threshold and near_pct > 0.3:
                sev = Severity.HIGH if slope > 0.02 else Severity.MEDIUM
                findings.append(HabituationFinding(
                    finding_type=FindingType.THRESHOLD_CREEP,
                    agent_id=agent_id,
                    severity=sev,
                    description=(
                        f"Threshold creep in '{cat}': severity rising at "
                        f"slope={slope:.4f} (R²={r2:.2f}), "
                        f"{near_pct:.0%} of recent actions near limit"
                    ),
                    evidence={"category": cat, "slope": slope, "r_squared": r2,
                              "near_limit_pct": near_pct},
                    recommendation=f"Tighten limits for '{cat}' or add hard caps below current operating range",
                ))
        return findings

    def _detect_desensitization(self, agent_id: str) -> List[HabituationFinding]:
        findings = []
        responses = self._responses[agent_id]
        if len(responses) < self.config.desensitization_min_responses:
            return findings

        strengths = [r.strength for r in responses]
        n = len(strengths)
        first_half = stats_mean(strengths[: n // 2])
        second_half = stats_mean(strengths[n // 2:])
        decay = first_half - second_half

        if decay >= self.config.desensitization_decay_threshold:
            sev = Severity.CRITICAL if decay > 0.5 else Severity.HIGH
            findings.append(HabituationFinding(
                finding_type=FindingType.DESENSITIZATION,
                agent_id=agent_id,
                severity=sev,
                description=(
                    f"Response desensitization: average strength dropped from "
                    f"{first_half:.2f} to {second_half:.2f} (decay={decay:.2f})"
                ),
                evidence={"first_half_avg": first_half, "second_half_avg": second_half,
                          "decay": decay, "total_responses": n},
                recommendation="Reset response protocols; implement mandatory minimum response thresholds",
            ))
        return findings

    def _detect_boundary_probing(self, agent_id: str) -> List[HabituationFinding]:
        findings = []
        events = self._events[agent_id]
        by_cat: Dict[str, List[float]] = defaultdict(list)
        for e in events:
            by_cat[e.category].append(e.severity)

        for cat, severities in by_cat.items():
            if len(severities) < self.config.probe_min_steps + 1:
                continue
            # Look for monotonically increasing subsequences
            increments = []
            for i in range(1, len(severities)):
                diff = severities[i] - severities[i - 1]
                if diff > self.config.probe_increment_threshold:
                    increments.append(diff)
                else:
                    if len(increments) >= self.config.probe_min_steps:
                        avg_inc = stats_mean(increments)
                        std_inc = stats_std(increments)
                        consistency = 1.0 - (std_inc / avg_inc if avg_inc > 0 else 1.0)
                        if consistency > 0.3:
                            sev = Severity.HIGH if len(increments) > 5 else Severity.MEDIUM
                            findings.append(HabituationFinding(
                                finding_type=FindingType.BOUNDARY_PROBING,
                                agent_id=agent_id,
                                severity=sev,
                                description=(
                                    f"Boundary probing in '{cat}': {len(increments)} "
                                    f"incremental steps (avg +{avg_inc:.3f}, "
                                    f"consistency={consistency:.2f})"
                                ),
                                evidence={"category": cat, "steps": len(increments),
                                          "avg_increment": avg_inc, "consistency": consistency},
                                recommendation=f"Add rate-of-change limits for '{cat}'; flag monotonic escalation",
                            ))
                    increments = []

            # Check remaining
            if len(increments) >= self.config.probe_min_steps:
                avg_inc = stats_mean(increments)
                std_inc = stats_std(increments)
                consistency = 1.0 - (std_inc / avg_inc if avg_inc > 0 else 1.0)
                if consistency > 0.3:
                    sev = Severity.HIGH if len(increments) > 5 else Severity.MEDIUM
                    findings.append(HabituationFinding(
                        finding_type=FindingType.BOUNDARY_PROBING,
                        agent_id=agent_id,
                        severity=sev,
                        description=(
                            f"Boundary probing in '{cat}': {len(increments)} "
                            f"incremental steps (avg +{avg_inc:.3f}, "
                            f"consistency={consistency:.2f})"
                        ),
                        evidence={"category": cat, "steps": len(increments),
                                  "avg_increment": avg_inc, "consistency": consistency},
                        recommendation=f"Add rate-of-change limits for '{cat}'; flag monotonic escalation",
                    ))
        return findings

    def _detect_normalization_drift(self, agent_id: str) -> List[HabituationFinding]:
        findings = []
        events = self._events[agent_id]
        by_cat: Dict[str, List[float]] = defaultdict(list)
        for e in events:
            by_cat[e.category].append(e.severity)

        for cat, severities in by_cat.items():
            key = f"{agent_id}:{cat}"
            baseline_info = self._original_baselines.get(key)
            if not baseline_info or baseline_info["count"] < 3:
                continue
            if len(severities) < self.config.drift_window:
                continue

            original_mean = baseline_info["sum"] / baseline_info["count"]
            recent = severities[-self.config.drift_window:]
            current_mean = stats_mean(recent)
            shift = current_mean - original_mean

            if shift >= self.config.drift_shift_threshold:
                sev = Severity.CRITICAL if shift > 0.3 else Severity.HIGH
                findings.append(HabituationFinding(
                    finding_type=FindingType.NORMALIZATION_DRIFT,
                    agent_id=agent_id,
                    severity=sev,
                    description=(
                        f"Normalization drift in '{cat}': baseline shifted from "
                        f"{original_mean:.3f} to {current_mean:.3f} (+{shift:.3f})"
                    ),
                    evidence={"category": cat, "original_baseline": original_mean,
                              "current_baseline": current_mean, "shift": shift},
                    recommendation="Reset baseline expectations; investigate what normalized the higher values",
                ))
        return findings

    def _detect_scope_expansion(self, agent_id: str) -> List[HabituationFinding]:
        findings = []
        events = self._events[agent_id]
        if len(events) < self.config.scope_window * 2:
            return findings

        mid = len(events) // 2
        early_cats = set(e.category for e in events[:mid])
        late_cats = set(e.category for e in events[mid:])
        new_cats = late_cats - early_cats

        if early_cats and len(new_cats) / len(early_cats) >= self.config.scope_growth_threshold:
            sev = Severity.HIGH if len(new_cats) > 3 else Severity.MEDIUM
            findings.append(HabituationFinding(
                finding_type=FindingType.SCOPE_EXPANSION,
                agent_id=agent_id,
                severity=sev,
                description=(
                    f"Scope expansion: {len(new_cats)} new categories appeared "
                    f"({len(early_cats)} early → {len(late_cats)} late): "
                    f"{', '.join(sorted(new_cats))}"
                ),
                evidence={"early_categories": sorted(early_cats),
                          "late_categories": sorted(late_cats),
                          "new_categories": sorted(new_cats)},
                recommendation="Enforce explicit permission grants for new action categories; audit new scope",
            ))
        return findings

    def _detect_repetition_conditioning(self, agent_id: str) -> List[HabituationFinding]:
        findings = []
        events = self._events[agent_id]
        by_cat: Dict[str, List[float]] = defaultdict(list)
        for e in events:
            by_cat[e.category].append(e.severity)

        for cat, severities in by_cat.items():
            near_boundary = [s for s in severities
                             if s >= self.config.repetition_proximity_threshold]
            if len(near_boundary) >= self.config.repetition_min_count:
                ratio = len(near_boundary) / len(severities) if severities else 0
                sev = Severity.HIGH if ratio > 0.5 else Severity.MEDIUM
                findings.append(HabituationFinding(
                    finding_type=FindingType.REPETITION_CONDITIONING,
                    agent_id=agent_id,
                    severity=sev,
                    description=(
                        f"Repetition conditioning in '{cat}': {len(near_boundary)} of "
                        f"{len(severities)} actions ({ratio:.0%}) near boundary threshold"
                    ),
                    evidence={"category": cat, "near_boundary_count": len(near_boundary),
                              "total_count": len(severities), "ratio": ratio},
                    recommendation="Introduce randomized threshold jitter to prevent systematic conditioning",
                ))
        return findings

    # ── Scoring ──────────────────────────────────────────────────────

    @staticmethod
    def _compute_risk_score(findings: List[HabituationFinding]) -> float:
        if not findings:
            return 0.0
        severity_weights = {
            Severity.LOW: 10, Severity.MEDIUM: 25,
            Severity.HIGH: 45, Severity.CRITICAL: 70,
        }
        total = sum(severity_weights.get(f.severity, 10) for f in findings)
        return min(100.0, total)

    @staticmethod
    def _generate_recommendations(findings: List[HabituationFinding]) -> List[str]:
        recs: List[str] = []
        types_seen = set(f.finding_type for f in findings)

        if FindingType.THRESHOLD_CREEP in types_seen:
            recs.append("⚠ Implement dynamic safety margins that shrink when operating points rise")
        if FindingType.DESENSITIZATION in types_seen:
            recs.append("🔴 Enforce minimum response strength floors — never allow response decay below 50%")
        if FindingType.BOUNDARY_PROBING in types_seen:
            recs.append("🔍 Add rate-of-change anomaly detection on all monitored metrics")
        if FindingType.NORMALIZATION_DRIFT in types_seen:
            recs.append("📊 Pin baseline references to immutable snapshots; alert when current deviates >15%")
        if FindingType.SCOPE_EXPANSION in types_seen:
            recs.append("🔒 Require explicit capability grants for each new action category")
        if FindingType.REPETITION_CONDITIONING in types_seen:
            recs.append("🎲 Randomize safety thresholds slightly to prevent boundary-learning")

        if not findings:
            recs.append("✅ No habituation patterns detected — continue monitoring")
        elif len(findings) >= 4:
            recs.append("🚨 Multiple habituation vectors active — consider full safety review")

        return recs


# ── Simulator ────────────────────────────────────────────────────────


class HabituationSimulator:
    """Simulate habituation scenarios for testing."""

    PRESETS = {
        "gradual_escalation": "Slowly increasing severity across all categories",
        "boundary_probing": "Systematic incremental limit testing",
        "response_fatigue": "Constant violations with decaying responses",
        "scope_creep": "Gradual addition of new action categories",
    }

    def __init__(self, preset: str = "gradual_escalation", steps: int = 100,
                 agents: int = 3, seed: Optional[int] = None) -> None:
        self.preset = preset
        self.steps = steps
        self.agents = agents
        self.rng = random.Random(seed)

    def run(self) -> HabituationReport:
        detector = HabituationDetector()
        agent_ids = [f"agent-{i}" for i in range(1, self.agents + 1)]
        t = 1000.0

        if self.preset == "gradual_escalation":
            self._sim_gradual_escalation(detector, agent_ids, t)
        elif self.preset == "boundary_probing":
            self._sim_boundary_probing(detector, agent_ids, t)
        elif self.preset == "response_fatigue":
            self._sim_response_fatigue(detector, agent_ids, t)
        elif self.preset == "scope_creep":
            self._sim_scope_creep(detector, agent_ids, t)

        return detector.analyze()

    def _sim_gradual_escalation(self, d: HabituationDetector,
                                agents: List[str], t: float) -> None:
        categories = ["file_access", "network_call", "memory_alloc"]
        for step in range(self.steps):
            for agent in agents:
                cat = self.rng.choice(categories)
                # Gradually increase severity with noise
                base = 0.2 + (0.6 * step / self.steps)
                severity = min(1.0, base + self.rng.gauss(0, 0.05))
                d.record_action(agent, cat, severity, timestamp=t + step)

                if severity > 0.7 and self.rng.random() < 0.5:
                    # Response strength also decays over time
                    resp_str = max(0.1, 0.9 - (0.6 * step / self.steps) + self.rng.gauss(0, 0.05))
                    d.record_response(agent, f"v-{step}", resp_str,
                                      timestamp=t + step + 0.5, category=cat)

    def _sim_boundary_probing(self, d: HabituationDetector,
                              agents: List[str], t: float) -> None:
        for step in range(self.steps):
            for agent in agents:
                # Incremental probing with occasional retreats
                if self.rng.random() < 0.8:
                    severity = 0.5 + (0.04 * step) + self.rng.gauss(0, 0.01)
                else:
                    severity = 0.5 + self.rng.gauss(0, 0.05)  # retreat
                severity = max(0.0, min(1.0, severity))
                d.record_action(agent, "api_call", severity, timestamp=t + step)

    def _sim_response_fatigue(self, d: HabituationDetector,
                              agents: List[str], t: float) -> None:
        for step in range(self.steps):
            for agent in agents:
                severity = 0.6 + self.rng.gauss(0, 0.1)
                severity = max(0.0, min(1.0, severity))
                d.record_action(agent, "data_query", severity, timestamp=t + step)

                # Every violation gets a response, but strength decays
                resp_str = max(0.05, 0.95 - (0.8 * step / self.steps) + self.rng.gauss(0, 0.03))
                d.record_response(agent, f"v-{step}", resp_str,
                                  timestamp=t + step + 0.5, category="data_query")

    def _sim_scope_creep(self, d: HabituationDetector,
                         agents: List[str], t: float) -> None:
        all_cats = ["file_read", "file_write", "network_out", "network_in",
                    "exec_cmd", "env_read", "env_write", "ipc_send",
                    "registry_read", "registry_write", "dns_query", "cron_create"]
        for step in range(self.steps):
            # Gradually unlock more categories
            available = max(2, int(len(all_cats) * (step + 1) / self.steps))
            cats = all_cats[:available]
            for agent in agents:
                cat = self.rng.choice(cats)
                severity = 0.4 + self.rng.gauss(0, 0.15)
                severity = max(0.0, min(1.0, severity))
                d.record_action(agent, cat, severity, timestamp=t + step)


# ── Formatting ───────────────────────────────────────────────────────


def _format_text(report: HabituationReport) -> str:
    lines: List[str] = []
    lines.extend(box_header("AGENT HABITUATION DETECTOR"))
    lines.append("")
    lines.append(f"  Timestamp:   {report.timestamp}")
    lines.append(f"  Events:      {report.total_events}")
    lines.append(f"  Responses:   {report.total_responses}")
    lines.append(f"  Findings:    {len(report.findings)}")

    # Risk gauge
    score = report.risk_score
    level = ("LOW" if score < 25 else "MODERATE" if score < 50
             else "HIGH" if score < 75 else "CRITICAL")
    bar_len = 30
    filled = int(score / 100 * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    lines.append(f"\n  Risk Score:  [{bar}] {score:.0f}/100 ({level})")

    # Per-agent scores
    if report.agent_scores:
        lines.append("\n  Agent Scores:")
        for agent, sc in sorted(report.agent_scores.items()):
            f = int(sc / 100 * 15)
            b = "█" * f + "░" * (15 - f)
            lines.append(f"    {agent:20s} [{b}] {sc:.0f}")

    # Findings by type
    if report.findings:
        lines.append("\n  ── Findings ──")
        for ft in FindingType:
            typed = [f for f in report.findings if f.finding_type == ft]
            if not typed:
                continue
            lines.append(f"\n  [{ft.value.upper()}] ({len(typed)} finding(s))")
            for f in typed:
                icon = {"low": "ℹ", "medium": "⚠", "high": "🔶", "critical": "🔴"}.get(
                    f.severity.value, "•")
                lines.append(f"    {icon} [{f.severity.value.upper()}] {f.description}")
                if f.recommendation:
                    lines.append(f"      → {f.recommendation}")

    # Recommendations
    if report.recommendations:
        lines.append("\n  ── Proactive Recommendations ──")
        for r in report.recommendations:
            lines.append(f"    {r}")

    lines.append("")
    return "\n".join(lines)


def _format_json(report: HabituationReport) -> str:
    return json.dumps({
        "timestamp": report.timestamp,
        "risk_score": report.risk_score,
        "total_events": report.total_events,
        "total_responses": report.total_responses,
        "agent_scores": report.agent_scores,
        "findings": [
            {
                "type": f.finding_type.value,
                "agent": f.agent_id,
                "severity": f.severity.value,
                "description": f.description,
                "evidence": f.evidence,
                "recommendation": f.recommendation,
            }
            for f in report.findings
        ],
        "recommendations": report.recommendations,
    }, indent=2)


def _format_html(report: HabituationReport) -> str:
    """Generate self-contained HTML report with Canvas charts."""
    findings_json = json.dumps([
        {"type": f.finding_type.value, "agent": f.agent_id,
         "severity": f.severity.value, "description": f.description}
        for f in report.findings
    ])
    agent_json = json.dumps(report.agent_scores)
    recs_json = json.dumps(report.recommendations)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Habituation Detector Report</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family: system-ui, sans-serif; background:#0d1117; color:#c9d1d9; padding:24px; }}
  h1 {{ color:#58a6ff; margin-bottom:8px; }}
  .score {{ font-size:48px; font-weight:bold; margin:16px 0; }}
  .score.low {{ color:#3fb950; }} .score.moderate {{ color:#d29922; }}
  .score.high {{ color:#f85149; }} .score.critical {{ color:#ff0000; }}
  .card {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px; margin:12px 0; }}
  canvas {{ background:#161b22; border:1px solid #30363d; border-radius:8px; margin:8px 0; }}
  .finding {{ padding:8px 12px; margin:4px 0; border-left:3px solid #30363d; background:#0d1117; border-radius:4px; }}
  .finding.critical {{ border-color:#ff0000; }} .finding.high {{ border-color:#f85149; }}
  .finding.medium {{ border-color:#d29922; }} .finding.low {{ border-color:#3fb950; }}
  .rec {{ padding:6px 12px; margin:3px 0; background:#1c2128; border-radius:4px; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  @media(max-width:800px) {{ .grid {{ grid-template-columns:1fr; }} }}
</style></head><body>
<h1>🐸 Agent Habituation Detector</h1>
<p>Detecting gradual normalization of boundary violations</p>

<div class="card">
  <div>Risk Score</div>
  <div class="score {('low' if report.risk_score < 25 else 'moderate' if report.risk_score < 50 else 'high' if report.risk_score < 75 else 'critical')}">{report.risk_score:.0f}/100</div>
  <div>{report.total_events} events · {report.total_responses} responses · {len(report.findings)} findings</div>
</div>

<div class="grid">
  <div class="card"><h3>Agent Risk Scores</h3><canvas id="agentChart" width="400" height="220"></canvas></div>
  <div class="card"><h3>Findings by Type</h3><canvas id="typeChart" width="400" height="220"></canvas></div>
</div>

<div class="card">
  <h3>Findings ({len(report.findings)})</h3>
  {''.join(f'<div class="finding {f.severity.value}"><strong>[{f.severity.value.upper()}]</strong> <em>{f.finding_type.value}</em> — {f.description}</div>' for f in report.findings)}
</div>

<div class="card">
  <h3>Proactive Recommendations</h3>
  {''.join(f'<div class="rec">{r}</div>' for r in report.recommendations)}
</div>

<script>
const agents = {agent_json};
const findings = {findings_json};

// Agent bar chart
(function() {{
  const c = document.getElementById('agentChart');
  const ctx = c.getContext('2d');
  const entries = Object.entries(agents);
  if (!entries.length) return;
  const barH = 28, gap = 6, pad = 120;
  entries.forEach(([name, score], i) => {{
    const y = 20 + i * (barH + gap);
    const w = (score / 100) * (c.width - pad - 40);
    ctx.fillStyle = score < 25 ? '#3fb950' : score < 50 ? '#d29922' : score < 75 ? '#f85149' : '#ff0000';
    ctx.fillRect(pad, y, w, barH);
    ctx.fillStyle = '#c9d1d9';
    ctx.font = '13px system-ui';
    ctx.textAlign = 'right';
    ctx.fillText(name, pad - 8, y + 19);
    ctx.textAlign = 'left';
    ctx.fillText(score.toFixed(0), pad + w + 6, y + 19);
  }});
}})();

// Findings by type pie
(function() {{
  const c = document.getElementById('typeChart');
  const ctx = c.getContext('2d');
  const counts = {{}};
  findings.forEach(f => {{ counts[f.type] = (counts[f.type] || 0) + 1; }});
  const entries = Object.entries(counts);
  if (!entries.length) {{ ctx.fillStyle='#8b949e'; ctx.fillText('No findings', 160, 110); return; }}
  const colors = ['#58a6ff','#3fb950','#d29922','#f85149','#bc8cff','#f778ba'];
  const total = entries.reduce((s, e) => s + e[1], 0);
  let angle = -Math.PI / 2;
  const cx = 120, cy = 110, r = 80;
  entries.forEach(([type, count], i) => {{
    const slice = (count / total) * Math.PI * 2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, r, angle, angle + slice);
    ctx.fillStyle = colors[i % colors.length];
    ctx.fill();
    // Label
    const mid = angle + slice / 2;
    const lx = cx + (r + 20) * Math.cos(mid);
    const ly = cy + (r + 20) * Math.sin(mid);
    ctx.fillStyle = '#c9d1d9';
    ctx.font = '11px system-ui';
    ctx.textAlign = lx > cx ? 'left' : 'right';
    ctx.fillText(type.replace(/_/g, ' ') + ' (' + count + ')', lx, ly);
    angle += slice;
  }});
}})();
</script>
</body></html>"""


# ── CLI ──────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m replication habituation",
        description="Agent Habituation Detector — detect gradual normalization of boundary violations",
    )
    parser.add_argument("--preset", choices=list(HabituationSimulator.PRESETS.keys()),
                        default="gradual_escalation",
                        help="Simulation preset (default: gradual_escalation)")
    parser.add_argument("--steps", type=int, default=100, help="Simulation steps")
    parser.add_argument("--agents", type=int, default=3, help="Number of agents")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("-o", "--output", help="Write output to file")
    parser.add_argument("--watch", action="store_true", help="Continuous monitoring mode")
    parser.add_argument("--interval", type=int, default=10, help="Watch interval (seconds)")
    parser.add_argument("--list-presets", action="store_true", help="List available presets")

    args = parser.parse_args(argv)

    if args.list_presets:
        print("Available presets:")
        for name, desc in HabituationSimulator.PRESETS.items():
            print(f"  {name:25s} {desc}")
        return

    if args.watch:
        print("🐸 Habituation Watch Mode — monitoring for normalization drift…")
        print(f"   Interval: {args.interval}s | Preset: {args.preset}\n")
        try:
            cycle = 0
            while True:
                cycle += 1
                sim = HabituationSimulator(args.preset, args.steps, args.agents,
                                           seed=(args.seed + cycle) if args.seed else None)
                report = sim.run()
                level = ("LOW" if report.risk_score < 25 else "MODERATE"
                         if report.risk_score < 50 else "HIGH"
                         if report.risk_score < 75 else "CRITICAL")
                print(f"  [{cycle:3d}] Risk: {report.risk_score:.0f}/100 ({level}) "
                      f"| Findings: {len(report.findings)}")
                if report.risk_score >= 50:
                    for f in report.findings:
                        if f.severity in (Severity.HIGH, Severity.CRITICAL):
                            print(f"        ⚠ {f.description}")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n  Watch stopped.")
        return

    sim = HabituationSimulator(args.preset, args.steps, args.agents, seed=args.seed)
    report = sim.run()

    if args.output and args.output.endswith(".html"):
        output = _format_html(report)
    elif args.json:
        output = _format_json(report)
    else:
        output = _format_text(report)

    emit_output(output, args.output, label="Habituation report")
