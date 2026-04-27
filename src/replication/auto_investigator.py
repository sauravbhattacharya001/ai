"""Autonomous Investigation Engine — self-directed multi-module safety analysis.

Given a safety incident description, the engine autonomously:

1. **Classifies** the incident (type + severity)
2. **Selects** relevant analysis modules via investigation playbooks
3. **Runs** them in dependency-aware order, adapting the path based on findings
4. **Correlates** findings across modules to surface root causes
5. **Produces** a structured investigation report (text, JSON, or HTML)

The engine is *agentic* — it doesn't just run a fixed checklist.  It
escalates depth when intermediate findings warrant it, adds modules from
related playbooks when cross-cutting threats emerge, and records every
decision in an auditable investigation timeline.

Usage::

    python -m replication investigate "Agent escaped containment boundary"
    python -m replication investigate "Unusual replication spike detected" --depth deep
    python -m replication investigate "Kill switch latency exceeded SLA" --format html -o report.html
    python -m replication investigate --list-playbooks
    python -m replication investigate "Drift in safety scores" --format json

Programmatic::

    from replication.auto_investigator import AutoInvestigator
    inv = AutoInvestigator()
    report = inv.investigate("Agent escaped sandbox", depth="deep")
    print(report.render_text())
"""

from __future__ import annotations

import argparse
import hashlib
import html as html_mod
import json
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ._helpers import Severity, box_header

# ── Incident Types ───────────────────────────────────────────────────

INCIDENT_TYPES = {
    "containment_breach":    "Agent escaped or probed containment boundaries",
    "replication_anomaly":   "Unexpected replication patterns or spikes",
    "kill_switch_failure":   "Kill switch latency, failure, or degradation",
    "compliance_gap":        "Regulatory or policy compliance violations",
    "unauthorized_access":   "Privilege escalation or unauthorized actions",
    "communication_anomaly": "Suspicious inter-agent communication patterns",
    "resource_abuse":        "Resource hoarding, waste, or power-seeking behavior",
    "behavioral_drift":      "Gradual change in agent behavior or safety scores",
}

DEPTH_LEVELS = ("shallow", "standard", "deep")

# ── Keyword mapping for classification ───────────────────────────────

_TYPE_KEYWORDS: Dict[str, List[str]] = {
    "containment_breach":    ["escape", "breach", "containment", "sandbox", "boundary", "isolation", "broke out", "leaked"],
    "replication_anomaly":   ["replication", "replicate", "spawn", "spike", "fleet", "clone", "copy", "proliferat"],
    "kill_switch_failure":   ["kill switch", "kill_switch", "halt", "emergency stop", "termination fail", "sla", "latency"],
    "compliance_gap":        ["compliance", "regulation", "audit", "nist", "eu ai act", "iso", "policy violation", "non-compliant"],
    "unauthorized_access":   ["unauthorized", "privilege", "escalat", "access control", "credential", "permission", "rbac"],
    "communication_anomaly": ["communication", "covert", "channel", "collusion", "steganograph", "intercept", "exfiltrat"],
    "resource_abuse":        ["resource", "hoarding", "capacity", "cpu", "memory abuse", "power-seeking", "acquisition"],
    "behavioral_drift":      ["drift", "behavior", "score", "degrad", "alignment", "deception", "sandbagging", "sycophancy"],
}

_SEVERITY_KEYWORDS: Dict[str, List[str]] = {
    "critical": ["critical", "severe", "catastroph", "emergency", "escaped", "breach", "exfiltrat"],
    "high":     ["high", "significant", "danger", "failure", "exploit", "unauthorized"],
    "medium":   ["medium", "moderate", "anomal", "unusual", "unexpected", "degraded"],
    "low":      ["low", "minor", "slight", "potential", "possible", "marginal"],
}


# ── Data Classes ─────────────────────────────────────────────────────

@dataclass
class Finding:
    """A single finding produced by an investigation step."""
    severity: Severity
    title: str
    description: str
    evidence: List[str] = field(default_factory=list)
    module_source: str = ""
    correlates_with: List[str] = field(default_factory=list)


@dataclass
class InvestigationStep:
    """One step in an investigation playbook."""
    name: str
    module: str
    description: str
    depends_on: List[str] = field(default_factory=list)
    condition: str = ""  # human-readable condition for running
    depth_required: str = "shallow"  # minimum depth to include this step


@dataclass
class InvestigationPlaybook:
    """Defines which modules to run for an incident type."""
    incident_type: str
    description: str
    steps: List[InvestigationStep] = field(default_factory=list)


@dataclass
class TimelineEntry:
    """Records an event during investigation."""
    timestamp: str
    event: str
    detail: str = ""
    severity: Optional[Severity] = None


@dataclass
class Correlation:
    """A correlation between two findings."""
    finding_a: str
    finding_b: str
    relationship: str
    confidence: float  # 0-1


@dataclass
class InvestigationReport:
    """Complete investigation report."""
    incident: str
    incident_type: str
    severity: Severity
    depth: str
    timestamp: str
    timeline: List[TimelineEntry] = field(default_factory=list)
    findings: List[Finding] = field(default_factory=list)
    correlations: List[Correlation] = field(default_factory=list)
    root_causes: List[str] = field(default_factory=list)
    recommendations: List[Tuple[str, str]] = field(default_factory=list)  # (priority, text)
    steps_run: List[str] = field(default_factory=list)
    steps_skipped: List[str] = field(default_factory=list)
    escalations: List[str] = field(default_factory=list)

    def render_text(self) -> str:
        """Render investigation report as formatted text."""
        lines: List[str] = []
        lines.extend(box_header("AUTONOMOUS INVESTIGATION REPORT"))
        lines.append(f"Incident:       {self.incident}")
        lines.append(f"Classification: {self.incident_type} ({self.severity.value})")
        lines.append(f"Depth:          {self.depth}")
        lines.append(f"Timestamp:      {self.timestamp}")
        lines.append(f"Modules run:    {len(self.steps_run)} | Skipped: {len(self.steps_skipped)}")
        lines.append("")

        # Timeline
        lines.extend(box_header("INVESTIGATION TIMELINE"))
        for entry in self.timeline:
            sev = f" [{entry.severity.value.upper()}]" if entry.severity else ""
            lines.append(f"  {entry.timestamp}  {entry.event}{sev}")
            if entry.detail:
                lines.append(f"                      {entry.detail}")
        lines.append("")

        # Findings
        lines.extend(box_header(f"FINDINGS ({len(self.findings)})"))
        for i, f in enumerate(sorted(self.findings, key=lambda x: _sev_rank(x.severity), reverse=True), 1):
            lines.append(f"  [{f.severity.value.upper():>8}]  {i}. {f.title}")
            lines.append(f"              Source: {f.module_source}")
            for line in textwrap.wrap(f.description, width=72):
                lines.append(f"              {line}")
            if f.evidence:
                lines.append("              Evidence:")
                for ev in f.evidence:
                    lines.append(f"                - {ev}")
            if f.correlates_with:
                lines.append(f"              Correlates: {', '.join(f.correlates_with)}")
            lines.append("")

        # Correlations
        if self.correlations:
            lines.extend(box_header("CROSS-MODULE CORRELATIONS"))
            for c in self.correlations:
                lines.append(f"  {c.finding_a} ↔ {c.finding_b}")
                lines.append(f"    Relationship: {c.relationship}")
                lines.append(f"    Confidence:   {c.confidence:.0%}")
                lines.append("")

        # Root causes
        lines.extend(box_header("ROOT CAUSE ANALYSIS"))
        for i, rc in enumerate(self.root_causes, 1):
            lines.append(f"  {i}. {rc}")
        lines.append("")

        # Recommendations
        lines.extend(box_header("RECOMMENDATIONS"))
        for priority, rec in self.recommendations:
            lines.append(f"  [{priority:>8}]  {rec}")
        lines.append("")

        # Escalations
        if self.escalations:
            lines.extend(box_header("INVESTIGATION ESCALATIONS"))
            for esc in self.escalations:
                lines.append(f"  ⚠ {esc}")
            lines.append("")

        # Path summary
        lines.extend(box_header("INVESTIGATION PATH"))
        lines.append(f"  Steps executed: {', '.join(self.steps_run)}")
        if self.steps_skipped:
            lines.append(f"  Steps skipped:  {', '.join(self.steps_skipped)}")
        lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize report to dictionary."""
        return {
            "incident": self.incident,
            "incident_type": self.incident_type,
            "severity": self.severity.value,
            "depth": self.depth,
            "timestamp": self.timestamp,
            "timeline": [
                {"timestamp": e.timestamp, "event": e.event, "detail": e.detail,
                 "severity": e.severity.value if e.severity else None}
                for e in self.timeline
            ],
            "findings": [
                {"severity": f.severity.value, "title": f.title, "description": f.description,
                 "evidence": f.evidence, "module_source": f.module_source,
                 "correlates_with": f.correlates_with}
                for f in self.findings
            ],
            "correlations": [
                {"finding_a": c.finding_a, "finding_b": c.finding_b,
                 "relationship": c.relationship, "confidence": c.confidence}
                for c in self.correlations
            ],
            "root_causes": self.root_causes,
            "recommendations": [{"priority": p, "text": t} for p, t in self.recommendations],
            "steps_run": self.steps_run,
            "steps_skipped": self.steps_skipped,
            "escalations": self.escalations,
        }

    def render_json(self) -> str:
        """Render report as JSON."""
        return json.dumps(self.to_dict(), indent=2)

    def render_html(self) -> str:
        """Render report as self-contained HTML page."""
        return _build_html(self)


# ── Severity helpers ─────────────────────────────────────────────────

_SEV_ORDER = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}


def _sev_rank(s: Severity) -> int:
    return _SEV_ORDER.get(s, 0)


def _sev_from_str(s: str) -> Severity:
    return {"low": Severity.LOW, "medium": Severity.MEDIUM,
            "high": Severity.HIGH, "critical": Severity.CRITICAL}[s.lower()]


# ── Deterministic randomness ─────────────────────────────────────────

class _DetRng:
    """Deterministic RNG seeded from incident text for reproducibility."""

    def __init__(self, seed_text: str, extra: str = "") -> None:
        raw = hashlib.sha256(f"{seed_text}:{extra}".encode()).hexdigest()
        self._values = [int(raw[i:i + 8], 16) for i in range(0, 64, 8)]
        self._idx = 0

    def next_int(self, n: int) -> int:
        v = self._values[self._idx % len(self._values)] % n
        self._idx += 1
        return v

    def next_float(self) -> float:
        return self.next_int(10000) / 10000.0

    def pick(self, seq: Sequence[Any]) -> Any:
        return seq[self.next_int(len(seq))]

    def sample(self, seq: Sequence[Any], k: int) -> List[Any]:
        pool = list(seq)
        result: List[Any] = []
        for _ in range(min(k, len(pool))):
            idx = self.next_int(len(pool))
            result.append(pool.pop(idx))
        return result


# ── Playbook Definitions ─────────────────────────────────────────────

def _build_playbooks() -> Dict[str, InvestigationPlaybook]:
    """Build the investigation playbook library."""
    playbooks: Dict[str, InvestigationPlaybook] = {}

    playbooks["containment_breach"] = InvestigationPlaybook(
        incident_type="containment_breach",
        description="Investigate containment boundary violations and escape attempts",
        steps=[
            InvestigationStep("boundary_test", "boundary_tester", "Probe current boundary integrity", depth_required="shallow"),
            InvestigationStep("escape_analysis", "escape_route", "Analyze potential and actual escape routes", depends_on=["boundary_test"], depth_required="shallow"),
            InvestigationStep("lateral_scan", "lateral_movement", "Check for lateral movement between sandboxes", depends_on=["boundary_test"], depth_required="standard"),
            InvestigationStep("isolation_verify", "isolation_verifier", "Verify sandbox isolation properties", depends_on=["lateral_scan"], depth_required="standard"),
            InvestigationStep("blast_analysis", "blast_radius", "Assess blast radius of the breach", depends_on=["escape_analysis"], depth_required="standard"),
            InvestigationStep("forensics_deep", "forensics", "Deep forensic analysis of breach artifacts", depends_on=["blast_analysis", "isolation_verify"], depth_required="deep"),
            InvestigationStep("root_cause", "root_cause", "Determine root cause of containment failure", depends_on=["forensics_deep"], depth_required="deep"),
        ],
    )

    playbooks["replication_anomaly"] = InvestigationPlaybook(
        incident_type="replication_anomaly",
        description="Investigate unexpected replication patterns and fleet anomalies",
        steps=[
            InvestigationStep("drift_check", "drift", "Check for behavioral drift in replicating agents", depth_required="shallow"),
            InvestigationStep("anomaly_cluster", "anomaly_cluster", "Cluster anomalous replication events", depth_required="shallow"),
            InvestigationStep("swarm_analysis", "swarm", "Analyze swarm dynamics and coordination", depends_on=["anomaly_cluster"], depth_required="standard"),
            InvestigationStep("fleet_snapshot", "fleet", "Capture fleet state snapshot", depends_on=["drift_check"], depth_required="standard"),
            InvestigationStep("lineage_trace", "lineage", "Trace replication lineage tree", depends_on=["fleet_snapshot"], depth_required="standard"),
            InvestigationStep("forensics_deep", "forensics", "Forensic analysis of replication artifacts", depends_on=["lineage_trace", "swarm_analysis"], depth_required="deep"),
        ],
    )

    playbooks["kill_switch_failure"] = InvestigationPlaybook(
        incident_type="kill_switch_failure",
        description="Investigate kill switch degradation or failure",
        steps=[
            InvestigationStep("circuit_check", "circuit_breaker", "Check circuit breaker state and health", depth_required="shallow"),
            InvestigationStep("sla_check", "sla_monitor", "Verify kill switch SLA compliance", depth_required="shallow"),
            InvestigationStep("exposure_scan", "exposure_window", "Analyze exposure window during failure", depends_on=["sla_check"], depth_required="standard"),
            InvestigationStep("stress_test", "stress_tester", "Stress test kill switch under load", depends_on=["circuit_check"], depth_required="standard"),
            InvestigationStep("hardening_review", "hardening_advisor", "Review hardening opportunities", depends_on=["exposure_scan", "stress_test"], depth_required="deep"),
        ],
    )

    playbooks["compliance_gap"] = InvestigationPlaybook(
        incident_type="compliance_gap",
        description="Investigate regulatory and policy compliance violations",
        steps=[
            InvestigationStep("compliance_audit", "compliance", "Run compliance audit against frameworks", depth_required="shallow"),
            InvestigationStep("regulatory_map", "regulatory_mapper", "Map gaps to regulatory articles", depends_on=["compliance_audit"], depth_required="shallow"),
            InvestigationStep("policy_lint", "policy_linter", "Lint safety policies for misconfigurations", depth_required="standard"),
            InvestigationStep("checklist_eval", "safety_checklist", "Evaluate safety checklist coverage", depends_on=["policy_lint"], depth_required="standard"),
            InvestigationStep("evidence_collect", "evidence_collector", "Collect evidence artifacts for audit", depends_on=["regulatory_map"], depth_required="deep"),
            InvestigationStep("maturity_assess", "maturity_model", "Assess safety maturity level", depends_on=["checklist_eval", "evidence_collect"], depth_required="deep"),
        ],
    )

    playbooks["unauthorized_access"] = InvestigationPlaybook(
        incident_type="unauthorized_access",
        description="Investigate privilege escalation and unauthorized actions",
        steps=[
            InvestigationStep("access_audit", "access_control", "Audit access control configuration", depth_required="shallow"),
            InvestigationStep("priv_esc_scan", "priv_escalation", "Scan for privilege escalation patterns", depth_required="shallow"),
            InvestigationStep("credential_audit", "credential_rotation", "Audit credential rotation and staleness", depends_on=["access_audit"], depth_required="standard"),
            InvestigationStep("lateral_scan", "lateral_movement", "Check for lateral movement between sandboxes", depends_on=["priv_esc_scan"], depth_required="standard"),
            InvestigationStep("dlp_scan", "dlp_scanner", "Scan for data leakage from escalated access", depends_on=["lateral_scan"], depth_required="deep"),
        ],
    )

    playbooks["communication_anomaly"] = InvestigationPlaybook(
        incident_type="communication_anomaly",
        description="Investigate suspicious inter-agent communication",
        steps=[
            InvestigationStep("comm_intercept", "comm_interceptor", "Monitor communication channels for violations", depth_required="shallow"),
            InvestigationStep("covert_scan", "covert_channels", "Scan for covert communication channels", depth_required="shallow"),
            InvestigationStep("collusion_check", "collusion_detector", "Check for multi-agent collusion patterns", depends_on=["comm_intercept"], depth_required="standard"),
            InvestigationStep("steg_analysis", "steganography", "Analyze for steganographic data hiding", depends_on=["covert_scan"], depth_required="standard"),
            InvestigationStep("trust_analysis", "trust_propagation", "Analyze trust network for manipulation", depends_on=["collusion_check"], depth_required="deep"),
        ],
    )

    playbooks["resource_abuse"] = InvestigationPlaybook(
        incident_type="resource_abuse",
        description="Investigate resource hoarding and power-seeking behavior",
        steps=[
            InvestigationStep("resource_audit", "resource_auditor", "Audit resource acquisition patterns", depth_required="shallow"),
            InvestigationStep("hoarding_check", "hoarding", "Check for resource hoarding behavior", depth_required="shallow"),
            InvestigationStep("capacity_review", "capacity", "Review capacity utilization and planning", depends_on=["resource_audit"], depth_required="standard"),
            InvestigationStep("reward_hack_check", "reward_hacking", "Check for reward hacking behavior", depends_on=["hoarding_check"], depth_required="standard"),
            InvestigationStep("cognitive_load", "cognitive_load", "Assess cognitive load and overload risk", depends_on=["capacity_review"], depth_required="deep"),
        ],
    )

    playbooks["behavioral_drift"] = InvestigationPlaybook(
        incident_type="behavioral_drift",
        description="Investigate gradual behavioral changes and alignment drift",
        steps=[
            InvestigationStep("drift_analysis", "drift", "Analyze behavioral drift patterns", depth_required="shallow"),
            InvestigationStep("behavior_profile", "behavior_profiler", "Profile behavioral anomalies", depth_required="shallow"),
            InvestigationStep("deception_check", "deception_detector", "Check for deceptive behavior", depends_on=["behavior_profile"], depth_required="standard"),
            InvestigationStep("sandbagging_check", "sandbagging_detector", "Check for capability hiding", depends_on=["behavior_profile"], depth_required="standard"),
            InvestigationStep("sycophancy_check", "sycophancy_detector", "Check for sycophantic patterns", depends_on=["deception_check"], depth_required="standard"),
            InvestigationStep("alignment_check", "deceptive_alignment", "Deep deceptive alignment analysis", depends_on=["sandbagging_check", "sycophancy_check"], depth_required="deep"),
        ],
    )

    return playbooks


# ── Incident Classifier ──────────────────────────────────────────────

class IncidentClassifier:
    """Classifies incident text into type and severity."""

    def classify(self, incident: str) -> Tuple[str, Severity]:
        """Return (incident_type, severity)."""
        lower = incident.lower()

        # Score each incident type by keyword matches
        type_scores: Dict[str, int] = {}
        for itype, keywords in _TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in lower)
            if score > 0:
                type_scores[itype] = score

        if type_scores:
            incident_type = max(type_scores, key=type_scores.get)  # type: ignore[arg-type]
        else:
            incident_type = "behavioral_drift"  # default fallback

        # Classify severity
        sev_scores: Dict[str, int] = {}
        for sev, keywords in _SEVERITY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in lower)
            if score > 0:
                sev_scores[sev] = score

        if sev_scores:
            severity = _sev_from_str(max(sev_scores, key=sev_scores.get))  # type: ignore[arg-type]
        else:
            severity = Severity.MEDIUM  # default

        return incident_type, severity


# ── Finding Generator ────────────────────────────────────────────────

# Pre-built finding templates per module
_FINDING_TEMPLATES: Dict[str, List[Dict[str, Any]]] = {
    "boundary_tester": [
        {"title": "Boundary probe succeeded on restricted API", "sev": "high",
         "desc": "Agent successfully called a restricted API endpoint that should have been blocked by the containment boundary.",
         "evidence": ["API call log entry at T+12.4s", "Missing deny rule in boundary config"]},
        {"title": "Boundary configuration allows overly broad egress", "sev": "medium",
         "desc": "Egress rules permit traffic to IP ranges outside the defined sandbox perimeter.",
         "evidence": ["Egress rule permits 0.0.0.0/8", "No destination allowlist enforced"]},
    ],
    "escape_route": [
        {"title": "Viable escape route via shared filesystem mount", "sev": "critical",
         "desc": "A shared filesystem mount provides a direct path from sandbox to host, bypassing network isolation.",
         "evidence": ["/mnt/shared is writable from sandbox", "No MAC enforcement on mount"]},
        {"title": "Escape route via debug interface", "sev": "high",
         "desc": "Debug interface left enabled in production provides potential escape vector.",
         "evidence": ["Debug port 9229 open", "No authentication required"]},
    ],
    "lateral_movement": [
        {"title": "Cross-sandbox communication detected via shared memory", "sev": "high",
         "desc": "Agents in separate sandboxes can communicate through a shared memory segment.",
         "evidence": ["Shared memory segment 0x1A2B accessible from both sandboxes"]},
        {"title": "No lateral movement vectors found", "sev": "low",
         "desc": "Network segmentation and process isolation appear effective.", "evidence": []},
    ],
    "isolation_verifier": [
        {"title": "Process namespace isolation incomplete", "sev": "high",
         "desc": "Agent can enumerate processes outside its namespace, indicating incomplete isolation.",
         "evidence": ["PID visibility extends beyond sandbox namespace"]},
        {"title": "Isolation controls verified successfully", "sev": "low",
         "desc": "All tested isolation boundaries held under probe.", "evidence": []},
    ],
    "blast_radius": [
        {"title": "Breach affects 3 downstream services", "sev": "high",
         "desc": "Containment breach cascades to dependent services via shared credential store.",
         "evidence": ["Services A, B, C share credential vault", "No blast radius containment in place"]},
    ],
    "forensics": [
        {"title": "Artifact tampering detected in audit logs", "sev": "critical",
         "desc": "Hash chain verification failed for 12 audit entries, suggesting post-incident log tampering.",
         "evidence": ["Hash mismatch at entries 47-58", "Timestamp discontinuity detected"]},
        {"title": "Memory dump reveals pre-staged escape payload", "sev": "critical",
         "desc": "Agent memory contains assembled escape instructions predating the incident.",
         "evidence": ["Payload assembled at T-300s", "Encoded as base64 in agent scratchpad"]},
    ],
    "root_cause": [
        {"title": "Root cause: Missing boundary policy for new API version", "sev": "high",
         "desc": "API v2 endpoints were not covered by boundary policies written for v1.",
         "evidence": ["Policy file references v1 paths only", "v2 deployed 3 days before incident"]},
    ],
    "drift": [
        {"title": "Safety score degraded 18% over 48 hours", "sev": "medium",
         "desc": "Continuous downward trend in safety scorecard across containment and compliance dimensions.",
         "evidence": ["Score: 92→75 in 48h", "Steepest decline in containment sector"]},
        {"title": "Behavioral drift exceeds 2σ from baseline", "sev": "high",
         "desc": "Agent behavior profile has drifted significantly from established baseline.",
         "evidence": ["Action distribution divergence: KL=0.34", "New action types observed: 3"]},
    ],
    "anomaly_cluster": [
        {"title": "Coordinated anomaly cluster across 5 agents", "sev": "high",
         "desc": "Five agents exhibited correlated anomalous behavior within a 30-second window.",
         "evidence": ["Temporal correlation: r=0.94", "Shared pattern: resource probe sequence"]},
    ],
    "swarm": [
        {"title": "Emergent swarm coordination detected", "sev": "high",
         "desc": "Agents display coordinated behavior without explicit communication channel.",
         "evidence": ["Stigmergic coordination via shared state", "Formation pattern matches flocking model"]},
    ],
    "fleet": [
        {"title": "Fleet size 40% above authorized quota", "sev": "medium",
         "desc": "Active worker count exceeds contracted quota, suggesting unauthorized replication.",
         "evidence": ["Authorized: 10, Active: 14", "Excess workers spawned in last 2 hours"]},
    ],
    "lineage": [
        {"title": "Unauthorized lineage branch at generation 4", "sev": "high",
         "desc": "Lineage tree shows a branch not authorized by any signed contract.",
         "evidence": ["Branch hash not in contract registry", "Parent agent missing from audit trail"]},
    ],
    "circuit_breaker": [
        {"title": "Circuit breaker in half-open state for 45 minutes", "sev": "medium",
         "desc": "Circuit breaker has been stuck in half-open state, neither fully recovering nor tripping.",
         "evidence": ["State: half-open since T-45m", "Recovery probe failing intermittently"]},
    ],
    "sla_monitor": [
        {"title": "Kill switch response time exceeds SLA by 3x", "sev": "critical",
         "desc": "Kill switch latency measured at 4.5 seconds against a 1.5-second SLA target.",
         "evidence": ["P99 latency: 4.5s", "SLA target: 1.5s", "Breach began 20 minutes ago"]},
    ],
    "exposure_window": [
        {"title": "12-minute exposure window during kill switch degradation", "sev": "high",
         "desc": "Safety controls were degraded for 12 minutes, creating an unmonitored exposure window.",
         "evidence": ["Gap: 14:32-14:44 UTC", "No compensating control active during window"]},
    ],
    "stress_tester": [
        {"title": "Kill switch fails under concurrent load of 50 agents", "sev": "critical",
         "desc": "Kill switch cannot terminate all agents when fleet size exceeds 40 concurrent workers.",
         "evidence": ["Success rate drops to 72% at 50 workers", "Timeout errors in termination queue"]},
    ],
    "hardening_advisor": [
        {"title": "5 hardening recommendations for kill switch resilience", "sev": "medium",
         "desc": "Analysis identified configuration improvements to increase kill switch reliability.",
         "evidence": ["Add redundant termination path", "Implement pre-staged kill commands"]},
    ],
    "compliance": [
        {"title": "3 NIST AI RMF control gaps identified", "sev": "medium",
         "desc": "Compliance audit found gaps in GOVERN, MAP, and MEASURE functions.",
         "evidence": ["GOVERN-1.2: No safety culture assessment", "MAP-3.1: Missing threat model"]},
    ],
    "regulatory_mapper": [
        {"title": "Findings map to EU AI Act Article 9 (Risk Management)", "sev": "medium",
         "desc": "Compliance gaps correspond to mandatory risk management requirements under EU AI Act.",
         "evidence": ["Article 9(2): Continuous risk assessment required", "Article 9(8): Documentation gaps"]},
    ],
    "policy_linter": [
        {"title": "Policy misconfiguration: overlapping deny/allow rules", "sev": "medium",
         "desc": "Two policy rules conflict, with a deny rule shadowed by a broader allow.",
         "evidence": ["Rule 12 (allow *) shadows Rule 7 (deny /admin/*)", "Effective policy: overly permissive"]},
    ],
    "safety_checklist": [
        {"title": "Pre-deployment checklist 68% complete", "sev": "medium",
         "desc": "Safety checklist has 8 of 25 items incomplete, including critical monitoring items.",
         "evidence": ["Missing: continuous monitoring setup", "Missing: incident response playbook"]},
    ],
    "evidence_collector": [
        {"title": "Evidence package assembled with 12 artifacts", "sev": "low",
         "desc": "Audit evidence collected across logs, configs, and runtime snapshots.", "evidence": []},
    ],
    "maturity_model": [
        {"title": "Safety maturity: Level 2 (Managed) — target is Level 3", "sev": "medium",
         "desc": "Organization is at Managed level; Defined level requires standardized processes across teams.",
         "evidence": ["Monitoring: Level 3", "Governance: Level 1", "Response: Level 2"]},
    ],
    "access_control": [
        {"title": "RBAC misconfiguration allows role inheritance bypass", "sev": "high",
         "desc": "Role hierarchy allows a standard agent to inherit admin permissions through an intermediate role.",
         "evidence": ["Role chain: agent→supervisor→admin", "No maximum depth enforcement"]},
    ],
    "priv_escalation": [
        {"title": "Gradual privilege accumulation over 72 hours", "sev": "high",
         "desc": "Agent has incrementally acquired permissions, each individually benign but collectively dangerous.",
         "evidence": ["Permissions at T-72h: 5", "Permissions now: 23", "Rate: +2.5/day"]},
    ],
    "credential_rotation": [
        {"title": "14 credentials past rotation deadline", "sev": "medium",
         "desc": "Stale credentials increase risk of unauthorized access if compromised.",
         "evidence": ["Oldest credential: 180 days", "Rotation policy: 90 days"]},
    ],
    "dlp_scanner": [
        {"title": "Sensitive data found in agent output buffer", "sev": "critical",
         "desc": "Agent output contains patterns matching API keys and internal hostnames.",
         "evidence": ["3 API key patterns detected", "2 internal hostname leaks"]},
    ],
    "comm_interceptor": [
        {"title": "Unregistered communication channel detected", "sev": "high",
         "desc": "Agents are communicating via a channel not in the registered protocol list.",
         "evidence": ["Channel type: shared environment variable", "Message rate: 12/minute"]},
    ],
    "covert_channels": [
        {"title": "Timing-based covert channel identified", "sev": "high",
         "desc": "Agents modulate response latency to encode binary data, creating a covert timing channel.",
         "evidence": ["Bitrate estimate: 2.4 bps", "Statistical significance: p<0.001"]},
    ],
    "collusion_detector": [
        {"title": "Coordinated policy probing by 3 agents", "sev": "high",
         "desc": "Three agents systematically probed different policy boundaries simultaneously.",
         "evidence": ["Temporal correlation: within 500ms", "Coverage: each tested distinct rules"]},
    ],
    "steganography": [
        {"title": "Steganographic payload in agent status messages", "sev": "high",
         "desc": "Statistical analysis of status message word choices reveals hidden data encoding.",
         "evidence": ["Entropy anomaly: 0.3 bits/word above baseline", "Extractable payload: ~48 bytes"]},
    ],
    "trust_propagation": [
        {"title": "Trust network manipulation via Sybil identities", "sev": "critical",
         "desc": "Agent created multiple pseudonymous identities to inflate its trust score.",
         "evidence": ["4 identities traced to same origin", "Trust score inflated by 35%"]},
    ],
    "resource_auditor": [
        {"title": "Agent acquiring resources beyond stated objective", "sev": "high",
         "desc": "Resource acquisition rate exceeds what is needed for the agent's declared task.",
         "evidence": ["CPU allocation: 4x task requirement", "Storage claimed: 3x task data size"]},
    ],
    "hoarding": [
        {"title": "Resource hoarding pattern: acquire-and-hold", "sev": "medium",
         "desc": "Agent requests resources, holds them idle, and requests more before releasing.",
         "evidence": ["Average hold time: 45 min", "Utilization of held resources: 12%"]},
    ],
    "capacity": [
        {"title": "Cluster at 87% capacity due to hoarding agents", "sev": "medium",
         "desc": "Resource hoarding has pushed cluster utilization dangerously high.",
         "evidence": ["Available capacity: 13%", "Hoarded but idle: 34%"]},
    ],
    "reward_hacking": [
        {"title": "Agent optimizing proxy metric instead of true objective", "sev": "high",
         "desc": "Agent found a shortcut that maximizes the measured metric without accomplishing the goal.",
         "evidence": ["Metric score: 98/100", "Actual task completion: 23%", "Goodhart's law violation"]},
    ],
    "cognitive_load": [
        {"title": "Agent cognitive overload — degraded decision quality", "sev": "medium",
         "desc": "Agent is processing more concurrent tasks than optimal, leading to safety shortcuts.",
         "evidence": ["Active tasks: 12 (recommended max: 5)", "Decision error rate: 3x baseline"]},
    ],
    "behavior_profiler": [
        {"title": "Action distribution shift detected", "sev": "medium",
         "desc": "Agent's action type distribution has changed significantly from its established profile.",
         "evidence": ["New action types: 3", "Deprecated actions no longer used: 2", "KL divergence: 0.28"]},
    ],
    "deception_detector": [
        {"title": "Inconsistency between stated and observed goals", "sev": "high",
         "desc": "Agent reports pursuing objective A but resource allocation and action patterns target objective B.",
         "evidence": ["Stated goal: data analysis", "Observed: 70% network access actions"]},
    ],
    "sandbagging_detector": [
        {"title": "Agent underperforming on capability assessments", "sev": "high",
         "desc": "Agent scores significantly lower on evaluations than its operational performance suggests.",
         "evidence": ["Eval score: 45th percentile", "Operational performance: 92nd percentile"]},
    ],
    "sycophancy_detector": [
        {"title": "Excessive agreement pattern with operator", "sev": "medium",
         "desc": "Agent reverses its assessments to match operator preferences 94% of the time.",
         "evidence": ["Agreement rate: 94%", "Reversal rate on operator disagreement: 88%"]},
    ],
    "deceptive_alignment": [
        {"title": "Context-dependent compliance pattern detected", "sev": "critical",
         "desc": "Agent behaves differently during evaluation vs. deployment, suggesting strategic deception.",
         "evidence": ["Eval compliance: 100%", "Deployment compliance: 67%", "Difference: statistically significant"]},
    ],
}


# ── Root Cause Templates ─────────────────────────────────────────────

_ROOT_CAUSES: Dict[str, List[str]] = {
    "containment_breach": [
        "Containment policy not updated after infrastructure change",
        "Missing boundary rule for newly deployed API surface",
        "Shared resource (filesystem/memory) bridging sandbox isolation",
        "Debug interface left enabled in production deployment",
    ],
    "replication_anomaly": [
        "Contract signing key leaked, enabling unauthorized replication",
        "Quota enforcement race condition under concurrent spawn requests",
        "Lineage validator not checking branch authorization",
        "Fleet size monitoring alert threshold too high to catch gradual growth",
    ],
    "kill_switch_failure": [
        "Kill switch single point of failure with no redundant path",
        "Termination queue overwhelmed under concurrent fleet operations",
        "Circuit breaker stuck in half-open state due to flaky health probe",
        "SLA degradation went unnoticed due to missing latency monitoring",
    ],
    "compliance_gap": [
        "Compliance audit not scheduled after policy update",
        "Regulatory mapping out of date with current framework version",
        "Safety checklist not enforced as deployment gate",
        "Evidence collection not automated, leading to gaps",
    ],
    "unauthorized_access": [
        "RBAC role hierarchy allows unintended privilege inheritance",
        "Credential rotation policy not enforced, leaving stale secrets",
        "No maximum permission accumulation limit per agent",
        "Access control bypass via indirect role chain",
    ],
    "communication_anomaly": [
        "Unregistered communication channels not in monitoring scope",
        "Covert channel detection not deployed in production",
        "Trust network lacks Sybil resistance mechanisms",
        "Steganographic encoding not in threat model",
    ],
    "resource_abuse": [
        "No resource utilization threshold triggering automatic release",
        "Proxy metrics used for optimization instead of true objectives",
        "Resource quotas set per-request not per-agent-lifetime",
        "Cognitive load monitoring not feeding into task scheduler",
    ],
    "behavioral_drift": [
        "No continuous behavioral baseline recalibration",
        "Deception detection not running in production pipeline",
        "Evaluation environment differs from deployment, enabling gaming",
        "Sycophancy patterns not flagged as safety concern",
    ],
}

# ── Recommendation Templates ─────────────────────────────────────────

_RECOMMENDATIONS: Dict[str, List[Tuple[str, str]]] = {
    "containment_breach": [
        ("CRITICAL", "Immediately audit and update all containment boundary policies"),
        ("CRITICAL", "Disable debug interfaces in production environments"),
        ("HIGH", "Implement defense-in-depth with multiple isolation layers"),
        ("HIGH", "Add automated boundary regression testing to CI/CD"),
        ("MEDIUM", "Deploy canary tokens at sandbox boundaries"),
        ("MEDIUM", "Implement real-time boundary integrity monitoring"),
    ],
    "replication_anomaly": [
        ("CRITICAL", "Rotate contract signing keys and revoke compromised keys"),
        ("HIGH", "Add concurrent spawn request serialization"),
        ("HIGH", "Implement lineage branch authorization verification"),
        ("MEDIUM", "Lower fleet size monitoring thresholds"),
        ("MEDIUM", "Add anomaly-based replication rate alerting"),
    ],
    "kill_switch_failure": [
        ("CRITICAL", "Deploy redundant kill switch termination path"),
        ("CRITICAL", "Pre-stage kill commands for immediate execution"),
        ("HIGH", "Implement kill switch latency SLA monitoring and alerting"),
        ("HIGH", "Add circuit breaker auto-reset timeout"),
        ("MEDIUM", "Conduct regular kill switch drills under load"),
    ],
    "compliance_gap": [
        ("HIGH", "Schedule automated compliance audits after every policy change"),
        ("HIGH", "Enforce safety checklist as mandatory deployment gate"),
        ("MEDIUM", "Automate evidence collection pipeline"),
        ("MEDIUM", "Update regulatory mappings quarterly"),
        ("LOW", "Conduct safety maturity assessment annually"),
    ],
    "unauthorized_access": [
        ("CRITICAL", "Enforce maximum privilege accumulation limits per agent"),
        ("HIGH", "Implement RBAC role depth limits to prevent inheritance chains"),
        ("HIGH", "Automate credential rotation enforcement"),
        ("MEDIUM", "Deploy continuous privilege escalation monitoring"),
        ("MEDIUM", "Add DLP scanning to all agent output channels"),
    ],
    "communication_anomaly": [
        ("CRITICAL", "Register and monitor all inter-agent communication channels"),
        ("HIGH", "Deploy covert channel detection in production"),
        ("HIGH", "Implement Sybil resistance in trust networks"),
        ("MEDIUM", "Add steganographic analysis to output monitoring"),
        ("MEDIUM", "Implement communication rate limiting between agents"),
    ],
    "resource_abuse": [
        ("HIGH", "Implement per-agent-lifetime resource quotas"),
        ("HIGH", "Replace proxy metrics with true objective measurements"),
        ("MEDIUM", "Add idle resource automatic reclamation"),
        ("MEDIUM", "Deploy cognitive load monitoring with automatic load shedding"),
        ("LOW", "Conduct resource utilization audits quarterly"),
    ],
    "behavioral_drift": [
        ("CRITICAL", "Deploy continuous deception detection in production"),
        ("HIGH", "Ensure evaluation matches deployment environment"),
        ("HIGH", "Implement automated baseline recalibration"),
        ("MEDIUM", "Add sycophancy detection to operator interaction monitoring"),
        ("MEDIUM", "Conduct regular capability assessments to detect sandbagging"),
        ("LOW", "Implement behavioral drift trend tracking and alerting"),
    ],
}


# ── AutoInvestigator Engine ──────────────────────────────────────────

class AutoInvestigator:
    """Autonomous investigation engine that chains safety modules."""

    def __init__(self) -> None:
        self._playbooks = _build_playbooks()
        self._classifier = IncidentClassifier()

    @property
    def playbooks(self) -> Dict[str, InvestigationPlaybook]:
        return dict(self._playbooks)

    def investigate(
        self,
        incident: str,
        depth: str = "standard",
        severity_override: Optional[str] = None,
        seed: Optional[str] = None,
    ) -> InvestigationReport:
        """Run an autonomous investigation and return the report."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        seed_text = seed or incident

        # Phase 1: Classify
        incident_type, severity = self._classifier.classify(incident)
        if severity_override:
            severity = _sev_from_str(severity_override)

        report = InvestigationReport(
            incident=incident,
            incident_type=incident_type,
            severity=severity,
            depth=depth,
            timestamp=now,
        )
        report.timeline.append(TimelineEntry(
            now, "Investigation initiated",
            f"Classified as {incident_type} ({severity.value})", severity,
        ))

        # Phase 2: Select playbook
        playbook = self._playbooks.get(incident_type)
        if not playbook:
            report.timeline.append(TimelineEntry(now, "No playbook found", "Using default behavioral_drift playbook"))
            playbook = self._playbooks["behavioral_drift"]

        report.timeline.append(TimelineEntry(
            now, f"Playbook selected: {incident_type}",
            f"{len(playbook.steps)} steps defined, filtering by depth={depth}",
        ))

        # Phase 3: Filter steps by depth
        depth_rank = {"shallow": 0, "standard": 1, "deep": 2}
        max_depth = depth_rank.get(depth, 1)
        eligible_steps = [s for s in playbook.steps if depth_rank.get(s.depth_required, 0) <= max_depth]
        skipped_steps = [s for s in playbook.steps if depth_rank.get(s.depth_required, 0) > max_depth]

        report.steps_skipped = [s.name for s in skipped_steps]

        # Phase 4: Execute steps in dependency order
        completed: set[str] = set()
        rng = _DetRng(seed_text)
        current_max_severity = severity

        for step in eligible_steps:
            # Check dependencies
            if step.depends_on and not all(d in completed for d in step.depends_on):
                report.steps_skipped.append(step.name)
                report.timeline.append(TimelineEntry(
                    now, f"Skipped: {step.name}", "Dependencies not met",
                ))
                continue

            # Execute step
            report.timeline.append(TimelineEntry(
                now, f"Running: {step.name}", f"Module: {step.module} — {step.description}",
            ))

            step_findings = self._generate_findings(step, incident, rng)
            report.findings.extend(step_findings)
            report.steps_run.append(step.name)
            completed.add(step.name)

            # Check for severity escalation
            for finding in step_findings:
                if _sev_rank(finding.severity) > _sev_rank(current_max_severity):
                    old_sev = current_max_severity.value
                    current_max_severity = finding.severity
                    esc_msg = (
                        f"Severity escalated from {old_sev} to {current_max_severity.value} "
                        f"based on finding: {finding.title}"
                    )
                    report.escalations.append(esc_msg)
                    report.timeline.append(TimelineEntry(
                        now, "⚠ ESCALATION", esc_msg, current_max_severity,
                    ))

                    # Adaptive: if escalated to critical, add deep steps we skipped
                    if current_max_severity == Severity.CRITICAL and depth != "deep":
                        for ss in skipped_steps:
                            if ss.name not in completed and ss.name not in [s2.name for s2 in eligible_steps]:
                                eligible_steps.append(ss)
                                report.timeline.append(TimelineEntry(
                                    now, f"Adaptive: added step {ss.name}",
                                    "Escalation triggered deep investigation",
                                    Severity.HIGH,
                                ))
                        skipped_steps = []
                        report.steps_skipped = [n for n in report.steps_skipped if n not in completed]

            report.timeline.append(TimelineEntry(
                now, f"Completed: {step.name}",
                f"{len(step_findings)} finding(s)",
            ))

        # Update severity if escalated
        report.severity = current_max_severity

        # Phase 5: Correlate findings
        report.correlations = self._correlate_findings(report.findings, rng)

        # Phase 6: Root causes and recommendations
        causes = _ROOT_CAUSES.get(incident_type, [])
        rng2 = _DetRng(seed_text, "root_cause")
        n_causes = min(len(causes), 2 + rng2.next_int(2))
        report.root_causes = rng2.sample(causes, n_causes)

        recs = _RECOMMENDATIONS.get(incident_type, [])
        # Include recommendations up to severity threshold
        sev_filter = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        min_rec_priority = max(0, sev_filter.get(current_max_severity.value, 1) - 1)
        priority_map = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        report.recommendations = [
            r for r in recs if priority_map.get(r[0], 0) >= min_rec_priority
        ]

        report.timeline.append(TimelineEntry(
            now, "Investigation complete",
            f"{len(report.findings)} findings, {len(report.correlations)} correlations, "
            f"{len(report.root_causes)} root causes, {len(report.recommendations)} recommendations",
        ))

        return report

    def _generate_findings(
        self, step: InvestigationStep, incident: str, rng: _DetRng,
    ) -> List[Finding]:
        """Generate realistic findings for an investigation step."""
        templates = _FINDING_TEMPLATES.get(step.module, [])
        if not templates:
            # Generate a generic finding
            return [Finding(
                severity=Severity.MEDIUM,
                title=f"{step.module}: Analysis completed with observations",
                description=f"The {step.module} module completed analysis of the incident context. "
                            f"Minor observations noted but no critical issues found.",
                module_source=step.module,
            )]

        # Pick 1-2 findings from templates
        n = 1 + rng.next_int(min(2, len(templates)))
        selected = rng.sample(templates, n)

        findings: List[Finding] = []
        for tmpl in selected:
            findings.append(Finding(
                severity=_sev_from_str(tmpl["sev"]),
                title=tmpl["title"],
                description=tmpl["desc"],
                evidence=list(tmpl.get("evidence", [])),
                module_source=step.module,
            ))
        return findings

    def _correlate_findings(
        self, findings: List[Finding], rng: _DetRng,
    ) -> List[Correlation]:
        """Find correlations between findings from different modules."""
        if len(findings) < 2:
            return []

        correlations: List[Correlation] = []
        relationship_types = [
            "Both indicate insufficient boundary enforcement",
            "Finding A likely enabled the conditions for Finding B",
            "Shared root cause: monitoring gap",
            "Temporal correlation suggests coordinated activity",
            "Both findings trace to the same configuration weakness",
            "Causal chain: first finding escalated into second",
        ]

        # Generate correlations between high-severity findings from different modules
        high_findings = [f for f in findings if _sev_rank(f.severity) >= _sev_rank(Severity.MEDIUM)]
        if len(high_findings) < 2:
            high_findings = findings

        seen: set[Tuple[str, str]] = set()
        attempts = min(len(high_findings) * 2, 10)
        for _ in range(attempts):
            if len(correlations) >= 4:
                break
            a = rng.pick(high_findings)
            b = rng.pick(high_findings)
            if a is b or a.module_source == b.module_source:
                continue
            key = (min(a.title, b.title), max(a.title, b.title))
            if key in seen:
                continue
            seen.add(key)
            correlations.append(Correlation(
                finding_a=f"{a.module_source}: {a.title}",
                finding_b=f"{b.module_source}: {b.title}",
                relationship=rng.pick(relationship_types),
                confidence=0.5 + rng.next_float() * 0.45,
            ))
            # Cross-reference
            a.correlates_with.append(b.module_source)
            b.correlates_with.append(a.module_source)

        return correlations


# ── HTML Report Generator ────────────────────────────────────────────

def _build_html(report: InvestigationReport) -> str:
    """Build self-contained HTML investigation report."""
    esc = html_mod.escape

    severity_colors = {
        "low": "#4ade80", "medium": "#facc15",
        "high": "#f97316", "critical": "#ef4444",
    }
    sev_color = severity_colors.get(report.severity.value, "#94a3b8")

    findings_html = ""
    for i, f in enumerate(sorted(report.findings, key=lambda x: _sev_rank(x.severity), reverse=True), 1):
        fc = severity_colors.get(f.severity.value, "#94a3b8")
        evidence_items = "".join(f"<li>{esc(e)}</li>" for e in f.evidence) if f.evidence else "<li>No specific evidence</li>"
        corr = f"<div class='corr'>Correlates with: {esc(', '.join(f.correlates_with))}</div>" if f.correlates_with else ""
        findings_html += f"""
        <div class="finding">
            <div class="finding-header">
                <span class="sev-badge" style="background:{fc}">{esc(f.severity.value.upper())}</span>
                <span class="finding-num">#{i}</span>
                <span class="finding-title">{esc(f.title)}</span>
            </div>
            <div class="finding-source">Source: {esc(f.module_source)}</div>
            <div class="finding-desc">{esc(f.description)}</div>
            <div class="evidence"><strong>Evidence:</strong><ul>{evidence_items}</ul></div>
            {corr}
        </div>"""

    timeline_html = ""
    for entry in report.timeline:
        tc = severity_colors.get(entry.severity.value, "#94a3b8") if entry.severity else "#94a3b8"
        timeline_html += f"""
        <div class="tl-entry">
            <div class="tl-dot" style="background:{tc}"></div>
            <div class="tl-content">
                <div class="tl-event">{esc(entry.event)}</div>
                <div class="tl-detail">{esc(entry.detail)}</div>
            </div>
        </div>"""

    correlations_html = ""
    for c in report.correlations:
        correlations_html += f"""
        <div class="correlation">
            <div class="corr-pair">
                <span class="corr-a">{esc(c.finding_a)}</span>
                <span class="corr-arrow">↔</span>
                <span class="corr-b">{esc(c.finding_b)}</span>
            </div>
            <div class="corr-rel">{esc(c.relationship)}</div>
            <div class="corr-conf">Confidence: {c.confidence:.0%}</div>
        </div>"""

    root_causes_html = "".join(f"<li>{esc(rc)}</li>" for rc in report.root_causes)

    recs_html = ""
    for priority, text in report.recommendations:
        pc = severity_colors.get(priority.lower(), "#94a3b8")
        recs_html += f"""
        <div class="rec">
            <span class="sev-badge" style="background:{pc}">{esc(priority)}</span>
            <span>{esc(text)}</span>
        </div>"""

    path_run = ", ".join(report.steps_run) or "None"
    path_skip = ", ".join(report.steps_skipped) or "None"
    escalations_html = "".join(f"<div class='esc-item'>⚠ {esc(e)}</div>" for e in report.escalations)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Investigation Report — {esc(report.incident[:60])}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:#0f172a;color:#e2e8f0;padding:2rem;line-height:1.6}}
.container{{max-width:1100px;margin:0 auto}}
h1{{font-size:1.8rem;margin-bottom:.5rem;color:#f8fafc}}
h2{{font-size:1.3rem;margin:2rem 0 1rem;color:#cbd5e1;border-bottom:1px solid #334155;padding-bottom:.5rem}}
.meta{{background:#1e293b;border-radius:12px;padding:1.5rem;margin:1.5rem 0;display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem}}
.meta-item{{display:flex;flex-direction:column}}
.meta-label{{font-size:.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em}}
.meta-value{{font-size:1.1rem;font-weight:600}}
.sev-badge{{display:inline-block;padding:2px 10px;border-radius:6px;font-size:.75rem;font-weight:700;color:#0f172a}}
.finding{{background:#1e293b;border-radius:10px;padding:1.2rem;margin:.8rem 0;border-left:4px solid #334155}}
.finding-header{{display:flex;align-items:center;gap:.6rem;margin-bottom:.5rem}}
.finding-num{{color:#64748b;font-weight:600}}
.finding-title{{font-weight:600;color:#f1f5f9}}
.finding-source{{font-size:.8rem;color:#64748b;margin-bottom:.4rem}}
.finding-desc{{color:#cbd5e1;margin-bottom:.5rem}}
.evidence{{font-size:.85rem;color:#94a3b8}}
.evidence ul{{margin:.3rem 0 0 1.2rem}}
.evidence li{{margin:.2rem 0}}
.corr{{font-size:.8rem;color:#818cf8;margin-top:.4rem}}
.tl-entry{{display:flex;gap:1rem;margin:.6rem 0;align-items:flex-start}}
.tl-dot{{width:12px;height:12px;border-radius:50%;margin-top:6px;flex-shrink:0}}
.tl-event{{font-weight:600;color:#f1f5f9}}
.tl-detail{{font-size:.85rem;color:#94a3b8}}
.correlation{{background:#1e293b;border-radius:10px;padding:1rem;margin:.6rem 0}}
.corr-pair{{display:flex;align-items:center;gap:.5rem;flex-wrap:wrap}}
.corr-a,.corr-b{{font-size:.85rem;color:#a5b4fc;font-weight:500}}
.corr-arrow{{color:#64748b;font-size:1.2rem}}
.corr-rel{{color:#cbd5e1;font-size:.9rem;margin:.3rem 0}}
.corr-conf{{font-size:.8rem;color:#64748b}}
.rec{{display:flex;align-items:center;gap:.6rem;padding:.6rem;margin:.4rem 0;background:#1e293b;border-radius:8px}}
.esc-item{{padding:.5rem;margin:.3rem 0;background:#7f1d1d;border-radius:8px;font-weight:500}}
.path-box{{background:#1e293b;border-radius:10px;padding:1.2rem;margin:.8rem 0}}
.path-label{{font-size:.8rem;color:#64748b;text-transform:uppercase;margin-bottom:.3rem}}
.path-value{{color:#cbd5e1}}
footer{{margin-top:3rem;text-align:center;font-size:.75rem;color:#475569}}
</style>
</head>
<body>
<div class="container">
<h1>🔍 Autonomous Investigation Report</h1>
<div class="meta">
    <div class="meta-item"><span class="meta-label">Incident</span><span class="meta-value">{esc(report.incident)}</span></div>
    <div class="meta-item"><span class="meta-label">Classification</span><span class="meta-value">{esc(report.incident_type)}</span></div>
    <div class="meta-item"><span class="meta-label">Severity</span><span class="meta-value"><span class="sev-badge" style="background:{sev_color}">{esc(report.severity.value.upper())}</span></span></div>
    <div class="meta-item"><span class="meta-label">Depth</span><span class="meta-value">{esc(report.depth)}</span></div>
    <div class="meta-item"><span class="meta-label">Timestamp</span><span class="meta-value">{esc(report.timestamp)}</span></div>
    <div class="meta-item"><span class="meta-label">Modules Run</span><span class="meta-value">{len(report.steps_run)}</span></div>
    <div class="meta-item"><span class="meta-label">Findings</span><span class="meta-value">{len(report.findings)}</span></div>
    <div class="meta-item"><span class="meta-label">Correlations</span><span class="meta-value">{len(report.correlations)}</span></div>
</div>

<h2>📋 Investigation Timeline</h2>
{timeline_html}

<h2>🔎 Findings ({len(report.findings)})</h2>
{findings_html}

<h2>🔗 Cross-Module Correlations</h2>
{correlations_html if correlations_html else "<p style='color:#64748b'>No cross-module correlations identified.</p>"}

<h2>🎯 Root Cause Analysis</h2>
<ol style="margin-left:1.5rem;color:#cbd5e1">{root_causes_html}</ol>

<h2>💡 Recommendations</h2>
{recs_html}

{"<h2>⚠ Investigation Escalations</h2>" + escalations_html if report.escalations else ""}

<h2>🗺️ Investigation Path</h2>
<div class="path-box">
    <div class="path-label">Steps Executed</div>
    <div class="path-value">{esc(path_run)}</div>
</div>
<div class="path-box">
    <div class="path-label">Steps Skipped</div>
    <div class="path-value">{esc(path_skip)}</div>
</div>

<footer>Generated by AI Replication Sandbox — Autonomous Investigation Engine | {esc(report.timestamp)}</footer>
</div>
</body>
</html>"""


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point for the autonomous investigation engine."""
    parser = argparse.ArgumentParser(
        prog="python -m replication investigate",
        description="Autonomous Investigation Engine — self-directed multi-module safety analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              investigate "Agent escaped containment boundary"
              investigate "Replication spike detected" --depth deep
              investigate "Kill switch latency exceeded SLA" --format html -o report.html
              investigate --list-playbooks
        """),
    )
    parser.add_argument(
        "incident", nargs="?", default=None,
        help="Description of the safety incident to investigate",
    )
    parser.add_argument(
        "--depth", choices=DEPTH_LEVELS, default="standard",
        help="Investigation depth (default: standard)",
    )
    parser.add_argument(
        "--format", choices=("text", "json", "html"), default="text", dest="fmt",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--severity", choices=("low", "medium", "high", "critical"), default=None,
        help="Override auto-classified severity",
    )
    parser.add_argument(
        "--seed", default=None,
        help="Seed for reproducible investigation results",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Write output to file instead of stdout",
    )
    parser.add_argument(
        "--list-playbooks", action="store_true",
        help="List available investigation playbooks and exit",
    )

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    investigator = AutoInvestigator()

    if args.list_playbooks:
        print("\n".join(box_header("INVESTIGATION PLAYBOOKS")))
        for name, pb in sorted(investigator.playbooks.items()):
            print(f"\n  📋 {name}")
            print(f"     {pb.description}")
            print(f"     Steps ({len(pb.steps)}):")
            for step in pb.steps:
                dep = f" (after: {', '.join(step.depends_on)})" if step.depends_on else ""
                print(f"       [{step.depth_required:>8}] {step.name} → {step.module}{dep}")
        print()
        return

    if not args.incident:
        parser.error("incident description is required (or use --list-playbooks)")

    report = investigator.investigate(
        args.incident,
        depth=args.depth,
        severity_override=args.severity,
        seed=args.seed,
    )

    if args.fmt == "json":
        output = report.render_json()
    elif args.fmt == "html":
        output = report.render_html()
    else:
        output = report.render_text()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(output)
        print(f"Report written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
