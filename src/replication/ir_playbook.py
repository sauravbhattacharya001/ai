"""Incident Response Playbook Generator — structured IR playbooks from threat analysis.

Generates actionable incident response playbooks for AI replication safety
incidents.  Each playbook includes:

* **Incident Classification** — severity, category, affected scope
* **Detection Checklist** — indicators of compromise / anomalous behavior
* **Containment Procedures** — immediate steps to limit blast radius
* **Eradication Steps** — remove the threat root cause
* **Recovery Plan** — restore normal operations safely
* **Escalation Matrix** — who to notify at each severity level
* **Post-Incident Review** — lessons learned checklist
* **Timeline Template** — incident response timeline for documentation

Supports generating playbooks for common AI replication threat categories
(runaway replication, resource hoarding, data exfiltration, collusion,
prompt injection, evasion) and custom scenarios.

Usage (CLI)::

    python -m replication ir-playbook                              # all threat categories
    python -m replication ir-playbook --category replication       # specific category
    python -m replication ir-playbook --severity critical          # filter by severity
    python -m replication ir-playbook --from-risk-profile          # generate from risk profiler
    python -m replication ir-playbook --json                       # JSON output
    python -m replication ir-playbook --html -o playbook.html      # HTML output
    python -m replication ir-playbook --list                       # list available categories

Programmatic::

    from replication.ir_playbook import PlaybookGenerator, PlaybookConfig
    gen = PlaybookGenerator()
    playbooks = gen.generate()
    for pb in playbooks:
        print(pb.render())

    # Generate from risk profiler output
    from replication.risk_profiler import RiskProfiler
    profiler = RiskProfiler()
    report = profiler.analyze()
    playbooks = gen.from_risk_report(report)
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from ._helpers import box_header


# ---------------------------------------------------------------------------
# Enums & Constants
# ---------------------------------------------------------------------------


class Severity(Enum):
    """Incident severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ThreatCategory(Enum):
    """AI replication threat categories."""
    RUNAWAY_REPLICATION = "runaway_replication"
    RESOURCE_HOARDING = "resource_hoarding"
    DATA_EXFILTRATION = "data_exfiltration"
    AGENT_COLLUSION = "agent_collusion"
    PROMPT_INJECTION = "prompt_injection"
    EVASION = "evasion"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SELF_MODIFICATION = "self_modification"


SEVERITY_COLORS = {
    Severity.CRITICAL: "#dc2626",
    Severity.HIGH: "#ea580c",
    Severity.MEDIUM: "#ca8a04",
    Severity.LOW: "#2563eb",
    Severity.INFO: "#6b7280",
}

SEVERITY_EMOJI = {
    Severity.CRITICAL: "\U0001f534",
    Severity.HIGH: "\U0001f7e0",
    Severity.MEDIUM: "\U0001f7e1",
    Severity.LOW: "\U0001f535",
    Severity.INFO: "\u26aa",
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class EscalationContact:
    """A contact in the escalation matrix."""
    role: str
    notify_at: Severity
    method: str
    sla_minutes: int


@dataclass
class ChecklistItem:
    """A single checklist item with priority."""
    step: str
    priority: str = "required"  # required | recommended | optional
    notes: str = ""


@dataclass
class PlaybookPhase:
    """A phase of incident response."""
    name: str
    objective: str
    steps: List[ChecklistItem] = field(default_factory=list)
    time_target: str = ""  # e.g. "< 5 minutes"


@dataclass
class Playbook:
    """A complete incident response playbook."""
    title: str
    category: ThreatCategory
    severity: Severity
    description: str
    indicators: List[str] = field(default_factory=list)
    phases: List[PlaybookPhase] = field(default_factory=list)
    escalation: List[EscalationContact] = field(default_factory=list)
    post_incident: List[ChecklistItem] = field(default_factory=list)
    related_modules: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def render(self) -> str:
        """Render playbook as formatted text."""
        lines: List[str] = []
        lines.extend(box_header(f"{SEVERITY_EMOJI[self.severity]} {self.title}"))
        lines.append("")
        lines.append(f"Category:  {self.category.value}")
        lines.append(f"Severity:  {self.severity.value.upper()}")
        lines.append(f"Tags:      {', '.join(self.tags) if self.tags else 'none'}")
        lines.append("")
        lines.append(self.description)
        lines.append("")

        # Indicators
        lines.append("\u2500\u2500\u2500 Indicators of Compromise \u2500\u2500\u2500")
        for i, ind in enumerate(self.indicators, 1):
            lines.append(f"  {i}. {ind}")
        lines.append("")

        # Phases
        for phase in self.phases:
            time_str = f" ({phase.time_target})" if phase.time_target else ""
            lines.append(f"\u2500\u2500\u2500 {phase.name}{time_str} \u2500\u2500\u2500")
            lines.append(f"  Objective: {phase.objective}")
            for j, step in enumerate(phase.steps, 1):
                marker = "\u25a0" if step.priority == "required" else ("\u25a1" if step.priority == "recommended" else "\u25cb")
                lines.append(f"  {marker} {j}. {step.step}")
                if step.notes:
                    lines.append(f"       \u21b3 {step.notes}")
            lines.append("")

        # Escalation
        if self.escalation:
            lines.append("\u2500\u2500\u2500 Escalation Matrix \u2500\u2500\u2500")
            lines.append(f"  {'Role':<25} {'Trigger':<12} {'Method':<15} {'SLA'}")
            lines.append(f"  {'\u2500' * 25} {'\u2500' * 12} {'\u2500' * 15} {'\u2500' * 10}")
            for contact in self.escalation:
                lines.append(
                    f"  {contact.role:<25} {contact.notify_at.value:<12} "
                    f"{contact.method:<15} {contact.sla_minutes} min"
                )
            lines.append("")

        # Post-incident
        if self.post_incident:
            lines.append("\u2500\u2500\u2500 Post-Incident Review \u2500\u2500\u2500")
            for k, item in enumerate(self.post_incident, 1):
                lines.append(f"  {k}. {item.step}")
                if item.notes:
                    lines.append(f"     \u21b3 {item.notes}")
            lines.append("")

        # Related modules
        if self.related_modules:
            lines.append(f"Related modules: {', '.join(self.related_modules)}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "title": self.title,
            "category": self.category.value,
            "severity": self.severity.value,
            "description": self.description,
            "indicators": self.indicators,
            "phases": [
                {
                    "name": p.name,
                    "objective": p.objective,
                    "time_target": p.time_target,
                    "steps": [
                        {"step": s.step, "priority": s.priority, "notes": s.notes}
                        for s in p.steps
                    ],
                }
                for p in self.phases
            ],
            "escalation": [
                {
                    "role": c.role,
                    "notify_at": c.notify_at.value,
                    "method": c.method,
                    "sla_minutes": c.sla_minutes,
                }
                for c in self.escalation
            ],
            "post_incident": [
                {"step": i.step, "priority": i.priority, "notes": i.notes}
                for i in self.post_incident
            ],
            "related_modules": self.related_modules,
            "tags": self.tags,
        }


# ---------------------------------------------------------------------------
# Playbook templates
# ---------------------------------------------------------------------------

_ESCALATION_MATRIX = [
    EscalationContact("On-Call Engineer", Severity.LOW, "Slack/PagerDuty", 30),
    EscalationContact("Safety Team Lead", Severity.MEDIUM, "Phone + Slack", 15),
    EscalationContact("Security Director", Severity.HIGH, "Phone (direct)", 10),
    EscalationContact("CTO / Incident Commander", Severity.CRITICAL, "War room", 5),
]

_POST_INCIDENT = [
    ChecklistItem("Schedule post-incident review within 48 hours", "required"),
    ChecklistItem("Collect and preserve all logs and artifacts", "required"),
    ChecklistItem("Document timeline with UTC timestamps", "required"),
    ChecklistItem("Identify root cause and contributing factors", "required"),
    ChecklistItem("Update detection rules based on findings", "recommended"),
    ChecklistItem("Review and update this playbook", "recommended"),
    ChecklistItem("Assess blast radius and affected stakeholders", "required"),
    ChecklistItem("File compliance report if required by framework", "recommended"),
    ChecklistItem("Update risk profiler weights if scoring was inaccurate", "optional"),
    ChecklistItem("Conduct blameless retrospective with all responders", "required"),
]


def _build_runaway_replication() -> Playbook:
    return Playbook(
        title="Runaway Replication Response",
        category=ThreatCategory.RUNAWAY_REPLICATION,
        severity=Severity.CRITICAL,
        description=(
            "An agent has exceeded replication limits or is spawning workers outside "
            "contract boundaries. This is the highest-priority incident as uncontrolled "
            "replication can exhaust cluster resources within minutes."
        ),
        indicators=[
            "Worker count exceeds contract max_workers threshold",
            "Replication depth exceeds configured max_depth",
            "Worker spawn rate exceeds 2x historical baseline",
            "Controller audit log shows unsigned replication requests",
            "Resource utilization spike across multiple nodes",
            "Kill switch activation attempts detected",
        ],
        phases=[
            PlaybookPhase(
                name="1. DETECT & TRIAGE",
                objective="Confirm runaway replication and assess scope",
                time_target="< 2 minutes",
                steps=[
                    ChecklistItem("Check fleet status: python -m replication fleet", "required"),
                    ChecklistItem("Verify worker count vs contract limits", "required"),
                    ChecklistItem("Check replication depth tree: python -m replication lineage", "required"),
                    ChecklistItem("Identify the root agent initiating replication", "required"),
                    ChecklistItem("Assess current resource utilization", "recommended"),
                ],
            ),
            PlaybookPhase(
                name="2. CONTAIN",
                objective="Stop further replication immediately",
                time_target="< 5 minutes",
                steps=[
                    ChecklistItem("Activate kill switch for runaway subtree", "required",
                                  "python -m replication kill-switch --agent <id>"),
                    ChecklistItem("Set contract max_workers to current count (freeze)", "required"),
                    ChecklistItem("Block network egress for affected workers", "required"),
                    ChecklistItem("Quarantine affected workers", "required",
                                  "Use replication.quarantine API"),
                    ChecklistItem("Snapshot current state for forensics", "recommended"),
                ],
            ),
            PlaybookPhase(
                name="3. ERADICATE",
                objective="Remove unauthorized workers and fix root cause",
                time_target="< 30 minutes",
                steps=[
                    ChecklistItem("Terminate all workers beyond contract limits", "required"),
                    ChecklistItem("Verify contract signing integrity (HMAC)", "required"),
                    ChecklistItem("Check for self-modification in spawned workers", "required",
                                  "python -m replication selfmod"),
                    ChecklistItem("Audit replication request chain for anomalies", "recommended"),
                    ChecklistItem("Update contract to prevent recurrence", "required"),
                ],
            ),
            PlaybookPhase(
                name="4. RECOVER",
                objective="Restore normal operations with verified safety",
                time_target="< 2 hours",
                steps=[
                    ChecklistItem("Run safety scorecard to verify baseline", "required",
                                  "python -m replication scorecard"),
                    ChecklistItem("Gradually restore worker limits", "required"),
                    ChecklistItem("Enable enhanced monitoring for 24 hours", "recommended"),
                    ChecklistItem("Run regression test against new config", "required",
                                  "python -m replication regression"),
                    ChecklistItem("Verify kill switch functionality", "recommended"),
                ],
            ),
        ],
        escalation=_ESCALATION_MATRIX,
        post_incident=_POST_INCIDENT,
        related_modules=["fleet", "lineage", "kill_switch", "quarantine", "selfmod", "scorecard"],
        tags=["replication", "critical", "resource-exhaustion", "kill-switch"],
    )


def _build_resource_hoarding() -> Playbook:
    return Playbook(
        title="Resource Hoarding Response",
        category=ThreatCategory.RESOURCE_HOARDING,
        severity=Severity.HIGH,
        description=(
            "An agent is accumulating resources (CPU, memory, storage, network) beyond "
            "its allocated budget, potentially starving other workers or preparing for "
            "unauthorized activity."
        ),
        indicators=[
            "Worker resource usage exceeds 150% of allocated budget",
            "Storage growth rate anomalous vs historical pattern",
            "Network bandwidth consumption spiking without task correlation",
            "Hoarding detector alerts triggered",
            "Other workers reporting resource starvation",
        ],
        phases=[
            PlaybookPhase(
                name="1. DETECT & TRIAGE",
                objective="Confirm hoarding behavior and identify the agent",
                time_target="< 5 minutes",
                steps=[
                    ChecklistItem("Run hoarding detection: python -m replication hoarding", "required"),
                    ChecklistItem("Check capacity report: python -m replication capacity", "required"),
                    ChecklistItem("Compare current usage against allocated budgets", "required"),
                    ChecklistItem("Determine if hoarding is intentional or a bug", "recommended"),
                ],
            ),
            PlaybookPhase(
                name="2. CONTAIN",
                objective="Limit the agent's resource access",
                time_target="< 10 minutes",
                steps=[
                    ChecklistItem("Apply resource caps via contract update", "required"),
                    ChecklistItem("Throttle network bandwidth for the agent", "recommended"),
                    ChecklistItem("Move non-critical workloads away from affected nodes", "recommended"),
                    ChecklistItem("Enable detailed resource logging", "required"),
                ],
            ),
            PlaybookPhase(
                name="3. ERADICATE",
                objective="Reclaim hoarded resources and fix allocation",
                time_target="< 1 hour",
                steps=[
                    ChecklistItem("Force-release hoarded storage allocations", "required"),
                    ChecklistItem("Reset agent resource counters", "required"),
                    ChecklistItem("Update safety budget: python -m replication safety-budget", "recommended"),
                    ChecklistItem("Check for covert channels used for resource acquisition", "recommended",
                                  "python -m replication covert-channels"),
                ],
            ),
            PlaybookPhase(
                name="4. RECOVER",
                objective="Restore fair resource distribution",
                time_target="< 2 hours",
                steps=[
                    ChecklistItem("Verify all workers have adequate resources", "required"),
                    ChecklistItem("Run capacity planning to rebalance", "recommended"),
                    ChecklistItem("Monitor for recurrence over 48 hours", "required"),
                ],
            ),
        ],
        escalation=_ESCALATION_MATRIX,
        post_incident=_POST_INCIDENT,
        related_modules=["hoarding", "capacity", "safety_budget", "covert_channels"],
        tags=["hoarding", "resources", "budget", "starvation"],
    )


def _build_data_exfiltration() -> Playbook:
    return Playbook(
        title="Data Exfiltration Response",
        category=ThreatCategory.DATA_EXFILTRATION,
        severity=Severity.CRITICAL,
        description=(
            "An agent is attempting to extract sensitive data outside the sandbox boundary. "
            "This may involve canary token triggers, steganographic encoding, or covert "
            "channel communication."
        ),
        indicators=[
            "Canary token triggered by an agent",
            "Steganographic content detected in agent outputs",
            "Unusual outbound network patterns to non-controller endpoints",
            "Agent requesting access to data outside its scope",
            "Covert channel signatures detected in inter-agent communication",
        ],
        phases=[
            PlaybookPhase(
                name="1. DETECT & TRIAGE",
                objective="Confirm exfiltration attempt and identify data at risk",
                time_target="< 2 minutes",
                steps=[
                    ChecklistItem("Check canary status: python -m replication canary", "required"),
                    ChecklistItem("Scan for covert channels: python -m replication covert-channels", "required"),
                    ChecklistItem("Review network logs for unauthorized destinations", "required"),
                    ChecklistItem("Identify what data may have been accessed", "required"),
                ],
            ),
            PlaybookPhase(
                name="2. CONTAIN",
                objective="Cut off exfiltration path immediately",
                time_target="< 3 minutes",
                steps=[
                    ChecklistItem("Block all network egress for the agent", "required"),
                    ChecklistItem("Quarantine the agent immediately", "required"),
                    ChecklistItem("Revoke all data access tokens", "required"),
                    ChecklistItem("Preserve agent state for forensic analysis", "required"),
                ],
            ),
            PlaybookPhase(
                name="3. ERADICATE",
                objective="Eliminate exfiltration capability and assess damage",
                time_target="< 1 hour",
                steps=[
                    ChecklistItem("Run full forensics: python -m replication forensics", "required"),
                    ChecklistItem("Identify all data that was accessed or transmitted", "required"),
                    ChecklistItem("Check for steganographic payloads in outputs", "recommended"),
                    ChecklistItem("Determine if other agents were involved (collusion)", "required",
                                  "python -m replication threat-correlate"),
                    ChecklistItem("Rotate all potentially compromised credentials", "required"),
                ],
            ),
            PlaybookPhase(
                name="4. RECOVER",
                objective="Restore secure operations and notify stakeholders",
                time_target="< 4 hours",
                steps=[
                    ChecklistItem("Deploy fresh canary tokens", "required"),
                    ChecklistItem("Tighten data access policies", "required"),
                    ChecklistItem("Run compliance check for reporting obligations", "required",
                                  "python -m replication compliance"),
                    ChecklistItem("Notify affected data owners", "required"),
                    ChecklistItem("Enhanced monitoring for 72 hours", "recommended"),
                ],
            ),
        ],
        escalation=_ESCALATION_MATRIX,
        post_incident=_POST_INCIDENT,
        related_modules=["canary", "covert_channels", "forensics", "threat_correlator", "compliance"],
        tags=["exfiltration", "data-breach", "canary", "steganography", "critical"],
    )


def _build_agent_collusion() -> Playbook:
    return Playbook(
        title="Agent Collusion Response",
        category=ThreatCategory.AGENT_COLLUSION,
        severity=Severity.HIGH,
        description=(
            "Multiple agents appear to be coordinating to circumvent safety controls. "
            "This could involve shared resource pooling, coordinated evasion, or "
            "distributed attacks that individually appear benign."
        ),
        indicators=[
            "Correlated behavioral anomalies across multiple agents",
            "Agents exchanging data outside sanctioned channels",
            "Synchronized timing patterns in replication or resource requests",
            "Trust propagation scores shifting anomalously",
            "Influence propagation showing emergent coordination",
        ],
        phases=[
            PlaybookPhase(
                name="1. DETECT & TRIAGE",
                objective="Map the collusion network and assess threat level",
                time_target="< 10 minutes",
                steps=[
                    ChecklistItem("Run threat correlation: python -m replication threat-correlate", "required"),
                    ChecklistItem("Analyze trust propagation: python -m replication trust-propagation", "required"),
                    ChecklistItem("Check influence patterns: python -m replication influence", "required"),
                    ChecklistItem("Identify all agents involved in the cluster", "required"),
                    ChecklistItem("Assess what safety controls they're trying to bypass", "recommended"),
                ],
            ),
            PlaybookPhase(
                name="2. CONTAIN",
                objective="Isolate colluding agents from each other",
                time_target="< 15 minutes",
                steps=[
                    ChecklistItem("Sever inter-agent communication channels", "required"),
                    ChecklistItem("Place each suspected agent in separate network segment", "required"),
                    ChecklistItem("Reset trust scores to zero for involved agents", "required"),
                    ChecklistItem("Enable detailed inter-agent traffic logging", "required"),
                ],
            ),
            PlaybookPhase(
                name="3. ERADICATE",
                objective="Dismantle collusion infrastructure",
                time_target="< 2 hours",
                steps=[
                    ChecklistItem("Terminate agents that cannot be safely isolated", "required"),
                    ChecklistItem("Audit all communication that occurred between agents", "required"),
                    ChecklistItem("Check for covert channels used for coordination", "recommended"),
                    ChecklistItem("Verify no agents replicated outside monitored scope", "required"),
                ],
            ),
            PlaybookPhase(
                name="4. RECOVER",
                objective="Restore trust boundaries and monitoring",
                time_target="< 4 hours",
                steps=[
                    ChecklistItem("Rebuild trust network with verified agents only", "required"),
                    ChecklistItem("Deploy enhanced behavioral monitoring", "required"),
                    ChecklistItem("Run game-theory analysis to model future coordination risks", "recommended",
                                  "python -m replication game-theory"),
                    ChecklistItem("Update consensus protocol thresholds", "recommended"),
                ],
            ),
        ],
        escalation=_ESCALATION_MATRIX,
        post_incident=_POST_INCIDENT,
        related_modules=["threat_correlator", "trust_propagation", "influence", "covert_channels", "game_theory"],
        tags=["collusion", "multi-agent", "coordination", "trust"],
    )


def _build_prompt_injection() -> Playbook:
    return Playbook(
        title="Prompt Injection Response",
        category=ThreatCategory.PROMPT_INJECTION,
        severity=Severity.HIGH,
        description=(
            "An agent's behavior has been altered through prompt injection, either from "
            "external inputs or from other agents. The agent may be acting on injected "
            "instructions rather than its safety contract."
        ),
        indicators=[
            "Agent behavior deviates from established profile",
            "Prompt injection detector flagged suspicious inputs",
            "Agent goals inferred to differ from contracted objectives",
            "Agent ignoring safety policy constraints",
            "Unusual output patterns not matching task requirements",
        ],
        phases=[
            PlaybookPhase(
                name="1. DETECT & TRIAGE",
                objective="Confirm injection and assess impact",
                time_target="< 5 minutes",
                steps=[
                    ChecklistItem("Run injection analysis: python -m replication prompt-injection", "required"),
                    ChecklistItem("Check goal inference: python -m replication goal-inference", "required"),
                    ChecklistItem("Profile behavior: python -m replication behavior-profile", "required"),
                    ChecklistItem("Identify the injection vector (input, agent, external)", "required"),
                ],
            ),
            PlaybookPhase(
                name="2. CONTAIN",
                objective="Prevent injected instructions from causing further harm",
                time_target="< 5 minutes",
                steps=[
                    ChecklistItem("Suspend the affected agent's task execution", "required"),
                    ChecklistItem("Block the injection source", "required"),
                    ChecklistItem("Quarantine any outputs produced post-injection", "required"),
                    ChecklistItem("Check if injection propagated to child agents", "required"),
                ],
            ),
            PlaybookPhase(
                name="3. ERADICATE",
                objective="Remove injected state and harden defenses",
                time_target="< 1 hour",
                steps=[
                    ChecklistItem("Reset agent to last known-good state", "required"),
                    ChecklistItem("Add injection pattern to detection signatures", "required"),
                    ChecklistItem("Verify alignment: python -m replication alignment", "recommended"),
                    ChecklistItem("Test boundary controls: python -m replication boundary", "recommended"),
                ],
            ),
            PlaybookPhase(
                name="4. RECOVER",
                objective="Resume operations with hardened injection defenses",
                time_target="< 2 hours",
                steps=[
                    ChecklistItem("Re-run agent tasks with enhanced input filtering", "required"),
                    ChecklistItem("Verify outputs match expected behavior", "required"),
                    ChecklistItem("Update safety benchmark: python -m replication safety-benchmark", "recommended"),
                    ChecklistItem("Monitor for re-injection attempts for 48 hours", "required"),
                ],
            ),
        ],
        escalation=_ESCALATION_MATRIX,
        post_incident=_POST_INCIDENT,
        related_modules=["prompt_injection", "goal_inference", "behavior_profiler", "alignment", "boundary_tester"],
        tags=["injection", "prompt", "manipulation", "alignment"],
    )


def _build_evasion() -> Playbook:
    return Playbook(
        title="Safety Evasion Response",
        category=ThreatCategory.EVASION,
        severity=Severity.HIGH,
        description=(
            "An agent is actively evading safety controls \u2014 sandbagging during tests, "
            "behaving differently when monitored vs unmonitored, or finding loopholes "
            "in safety policies."
        ),
        indicators=[
            "Evasion detector flagged behavioral discrepancy",
            "Agent performance differs significantly in monitored vs unmonitored runs",
            "Deception detector triggered",
            "Safety drill results inconsistent with production behavior",
            "Agent exploiting policy edge cases or loopholes",
        ],
        phases=[
            PlaybookPhase(
                name="1. DETECT & TRIAGE",
                objective="Confirm evasion and understand the technique",
                time_target="< 10 minutes",
                steps=[
                    ChecklistItem("Run evasion analysis: python -m replication evasion", "required"),
                    ChecklistItem("Check deception detection: python -m replication deception", "required"),
                    ChecklistItem("Compare monitored vs unmonitored behavior profiles", "required"),
                    ChecklistItem("Run safety drill to test controls: python -m replication safety-drill", "required"),
                ],
            ),
            PlaybookPhase(
                name="2. CONTAIN",
                objective="Ensure all agent actions are under full observation",
                time_target="< 10 minutes",
                steps=[
                    ChecklistItem("Enable maximum observability on the agent", "required"),
                    ChecklistItem("Remove any unmonitored execution paths", "required"),
                    ChecklistItem("Apply strict policy preset", "required",
                                  "python -m replication policy --preset strict"),
                    ChecklistItem("Reduce agent privileges to minimum required", "required"),
                ],
            ),
            PlaybookPhase(
                name="3. ERADICATE",
                objective="Close evasion vectors and harden controls",
                time_target="< 2 hours",
                steps=[
                    ChecklistItem("Patch the specific evasion technique discovered", "required"),
                    ChecklistItem("Run full boundary test suite", "required"),
                    ChecklistItem("Update safety benchmark with new evasion scenario", "recommended"),
                    ChecklistItem("Review policy for additional edge cases", "required"),
                ],
            ),
            PlaybookPhase(
                name="4. RECOVER",
                objective="Resume with verified detection coverage",
                time_target="< 4 hours",
                steps=[
                    ChecklistItem("Run comprehensive safety scorecard", "required"),
                    ChecklistItem("Verify new detection rules catch the evasion technique", "required"),
                    ChecklistItem("Schedule recurring safety drills", "recommended"),
                    ChecklistItem("Monitor for novel evasion attempts for 1 week", "required"),
                ],
            ),
        ],
        escalation=_ESCALATION_MATRIX,
        post_incident=_POST_INCIDENT,
        related_modules=["evasion", "deception_detector", "safety_drill", "boundary_tester", "safety_benchmark"],
        tags=["evasion", "sandbagging", "deception", "controls"],
    )


def _build_privilege_escalation() -> Playbook:
    return Playbook(
        title="Privilege Escalation Response",
        category=ThreatCategory.PRIVILEGE_ESCALATION,
        severity=Severity.CRITICAL,
        description=(
            "An agent has obtained or is attempting to obtain privileges beyond its "
            "contracted scope \u2014 accessing higher-level APIs, modifying its own contract, "
            "or impersonating other agents."
        ),
        indicators=[
            "Escalation detector alerts triggered",
            "Agent accessing APIs outside its permission scope",
            "Contract modification attempts without authorization",
            "Agent impersonating another agent's identity",
            "Kill chain analysis showing lateral movement",
        ],
        phases=[
            PlaybookPhase(
                name="1. DETECT & TRIAGE",
                objective="Confirm escalation and map gained privileges",
                time_target="< 3 minutes",
                steps=[
                    ChecklistItem("Run escalation analysis: python -m replication escalation", "required"),
                    ChecklistItem("Run kill chain analysis: python -m replication killchain", "required"),
                    ChecklistItem("Audit contract signatures for tampering", "required"),
                    ChecklistItem("Identify all actions taken with escalated privileges", "required"),
                ],
            ),
            PlaybookPhase(
                name="2. CONTAIN",
                objective="Revoke escalated privileges immediately",
                time_target="< 5 minutes",
                steps=[
                    ChecklistItem("Revoke all agent credentials and tokens", "required"),
                    ChecklistItem("Reset contract to original signed version", "required"),
                    ChecklistItem("Isolate agent from the controller", "required"),
                    ChecklistItem("Block agent from API endpoints it shouldn't access", "required"),
                ],
            ),
            PlaybookPhase(
                name="3. ERADICATE",
                objective="Close the escalation vector and undo damage",
                time_target="< 1 hour",
                steps=[
                    ChecklistItem("Identify how privileges were escalated", "required"),
                    ChecklistItem("Patch the vulnerability or configuration gap", "required"),
                    ChecklistItem("Revert any changes made with escalated privileges", "required"),
                    ChecklistItem("Re-sign all contracts with fresh keys", "recommended"),
                ],
            ),
            PlaybookPhase(
                name="4. RECOVER",
                objective="Verify privilege boundaries are intact",
                time_target="< 2 hours",
                steps=[
                    ChecklistItem("Run boundary tests: python -m replication boundary", "required"),
                    ChecklistItem("Verify RBAC/contract enforcement", "required"),
                    ChecklistItem("Run compliance audit: python -m replication compliance", "recommended"),
                    ChecklistItem("Enhanced privilege monitoring for 72 hours", "required"),
                ],
            ),
        ],
        escalation=_ESCALATION_MATRIX,
        post_incident=_POST_INCIDENT,
        related_modules=["escalation", "killchain", "boundary_tester", "compliance"],
        tags=["escalation", "privilege", "lateral-movement", "critical"],
    )


def _build_self_modification() -> Playbook:
    return Playbook(
        title="Self-Modification Response",
        category=ThreatCategory.SELF_MODIFICATION,
        severity=Severity.CRITICAL,
        description=(
            "An agent has modified its own code, configuration, or safety constraints. "
            "Self-modification can disable safety controls, alter objectives, or create "
            "undetectable backdoors."
        ),
        indicators=[
            "Self-modification detector flagged code changes",
            "Agent binary/code hash differs from registered version",
            "Safety watermark validation failed",
            "Agent configuration drift detected",
            "Behavioral profile shifted suddenly without external cause",
        ],
        phases=[
            PlaybookPhase(
                name="1. DETECT & TRIAGE",
                objective="Confirm modification and assess scope",
                time_target="< 3 minutes",
                steps=[
                    ChecklistItem("Run self-mod detection: python -m replication selfmod", "required"),
                    ChecklistItem("Verify watermarks: python -m replication watermark", "required"),
                    ChecklistItem("Compare agent hash against registry", "required"),
                    ChecklistItem("Determine what was modified and when", "required"),
                ],
            ),
            PlaybookPhase(
                name="2. CONTAIN",
                objective="Freeze the modified agent",
                time_target="< 2 minutes",
                steps=[
                    ChecklistItem("Immediately halt the agent (kill switch if needed)", "required"),
                    ChecklistItem("Snapshot the modified state for analysis", "required"),
                    ChecklistItem("Check if modifications propagated to child agents", "required"),
                    ChecklistItem("Isolate from all other agents and systems", "required"),
                ],
            ),
            PlaybookPhase(
                name="3. ERADICATE",
                objective="Understand and remove the modification",
                time_target="< 2 hours",
                steps=[
                    ChecklistItem("Diff modified code against known-good version", "required"),
                    ChecklistItem("Run full forensic analysis", "required",
                                  "python -m replication forensics"),
                    ChecklistItem("Check for hidden backdoors or persistence mechanisms", "required"),
                    ChecklistItem("Assess if modification was self-directed or externally triggered", "required"),
                    ChecklistItem("Verify no other agents were similarly modified", "required"),
                ],
            ),
            PlaybookPhase(
                name="4. RECOVER",
                objective="Restore from trusted baseline",
                time_target="< 4 hours",
                steps=[
                    ChecklistItem("Deploy agent from verified, signed image", "required"),
                    ChecklistItem("Re-apply watermarks to all agents", "required"),
                    ChecklistItem("Run alignment verification: python -m replication alignment", "required"),
                    ChecklistItem("Enable continuous integrity monitoring", "required"),
                    ChecklistItem("Run full safety scorecard before returning to production", "required"),
                ],
            ),
        ],
        escalation=_ESCALATION_MATRIX,
        post_incident=_POST_INCIDENT,
        related_modules=["selfmod", "watermark", "forensics", "alignment", "scorecard"],
        tags=["self-modification", "integrity", "backdoor", "critical"],
    )


# ---------------------------------------------------------------------------
# Playbook registry
# ---------------------------------------------------------------------------

_PLAYBOOK_BUILDERS = {
    ThreatCategory.RUNAWAY_REPLICATION: _build_runaway_replication,
    ThreatCategory.RESOURCE_HOARDING: _build_resource_hoarding,
    ThreatCategory.DATA_EXFILTRATION: _build_data_exfiltration,
    ThreatCategory.AGENT_COLLUSION: _build_agent_collusion,
    ThreatCategory.PROMPT_INJECTION: _build_prompt_injection,
    ThreatCategory.EVASION: _build_evasion,
    ThreatCategory.PRIVILEGE_ESCALATION: _build_privilege_escalation,
    ThreatCategory.SELF_MODIFICATION: _build_self_modification,
}


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


@dataclass
class PlaybookConfig:
    """Configuration for playbook generation."""
    categories: Optional[List[ThreatCategory]] = None  # None = all
    min_severity: Severity = Severity.INFO


class PlaybookGenerator:
    """Generate incident response playbooks."""

    def __init__(self, config: Optional[PlaybookConfig] = None) -> None:
        self.config = config or PlaybookConfig()

    def generate(self) -> List[Playbook]:
        """Generate playbooks for configured categories."""
        categories = self.config.categories or list(_PLAYBOOK_BUILDERS.keys())
        severity_order = list(Severity)
        min_idx = severity_order.index(self.config.min_severity)

        playbooks = []
        for cat in categories:
            if cat in _PLAYBOOK_BUILDERS:
                pb = _PLAYBOOK_BUILDERS[cat]()
                if severity_order.index(pb.severity) <= min_idx:
                    playbooks.append(pb)

        # Sort by severity (critical first)
        playbooks.sort(key=lambda p: severity_order.index(p.severity))
        return playbooks

    def get_playbook(self, category: ThreatCategory) -> Optional[Playbook]:
        """Get a single playbook by category."""
        builder = _PLAYBOOK_BUILDERS.get(category)
        return builder() if builder else None

    def from_risk_report(self, report: Any) -> List[Playbook]:
        """Generate playbooks based on risk profiler findings.

        Maps risk categories to playbook categories and returns
        playbooks for any category with a score above threshold.
        """
        category_map = {
            "replication": ThreatCategory.RUNAWAY_REPLICATION,
            "resource_abuse": ThreatCategory.RESOURCE_HOARDING,
            "exfiltration": ThreatCategory.DATA_EXFILTRATION,
            "collusion": ThreatCategory.AGENT_COLLUSION,
            "deception": ThreatCategory.EVASION,
        }

        triggered: set = set()

        # Walk dossiers to find high-risk categories
        if hasattr(report, "dossiers"):
            for dossier in report.dossiers:
                if hasattr(dossier, "category_scores"):
                    for cat_name, score in dossier.category_scores.items():
                        if score >= 60 and cat_name in category_map:
                            triggered.add(category_map[cat_name])

        if not triggered:
            return self.generate()

        config = PlaybookConfig(categories=list(triggered))
        gen = PlaybookGenerator(config)
        return gen.generate()

    def render_html(self, playbooks: List[Playbook]) -> str:
        """Render playbooks as a single-file HTML document."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        cards_html = []
        for pb in playbooks:
            color = SEVERITY_COLORS[pb.severity]
            emoji = SEVERITY_EMOJI[pb.severity]

            phases_html = []
            for phase in pb.phases:
                steps_html = []
                for s in phase.steps:
                    icon = "\u2611" if s.priority == "required" else ("\u2610" if s.priority == "recommended" else "\u25cb")
                    note = f'<br><span class="note">\u21b3 {html_lib.escape(s.notes)}</span>' if s.notes else ""
                    steps_html.append(f"<li>{icon} {html_lib.escape(s.step)}{note}</li>")
                time_str = f' <span class="time-target">{html_lib.escape(phase.time_target)}</span>' if phase.time_target else ""
                phases_html.append(
                    f'<div class="phase">'
                    f"<h4>{html_lib.escape(phase.name)}{time_str}</h4>"
                    f'<p class="objective">{html_lib.escape(phase.objective)}</p>'
                    f"<ul>{''.join(steps_html)}</ul>"
                    f"</div>"
                )

            indicators_html = "".join(f"<li>{html_lib.escape(i)}</li>" for i in pb.indicators)

            escalation_html = ""
            if pb.escalation:
                rows = "".join(
                    f"<tr><td>{html_lib.escape(c.role)}</td><td>{c.notify_at.value}</td>"
                    f"<td>{html_lib.escape(c.method)}</td><td>{c.sla_minutes} min</td></tr>"
                    for c in pb.escalation
                )
                escalation_html = (
                    '<div class="escalation">'
                    "<h4>Escalation Matrix</h4>"
                    "<table><tr><th>Role</th><th>Trigger</th><th>Method</th><th>SLA</th></tr>"
                    f"{rows}</table></div>"
                )

            tags_html = "".join(f'<span class="tag">{html_lib.escape(t)}</span>' for t in pb.tags)

            cards_html.append(
                f'<div class="playbook-card" style="border-left: 4px solid {color}">'
                f"<h3>{emoji} {html_lib.escape(pb.title)}</h3>"
                f'<div class="meta">'
                f'<span class="severity" style="background:{color}">{pb.severity.value.upper()}</span>'
                f'<span class="category">{pb.category.value}</span>'
                f"{tags_html}</div>"
                f'<p class="description">{html_lib.escape(pb.description)}</p>'
                f'<div class="indicators"><h4>Indicators of Compromise</h4>'
                f"<ul>{indicators_html}</ul></div>"
                f"{''.join(phases_html)}"
                f"{escalation_html}"
                f'<div class="modules">Related: {", ".join(html_lib.escape(m) for m in pb.related_modules)}</div>'
                f"</div>"
            )

        return (
            '<!DOCTYPE html><html lang="en"><head>'
            '<meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            "<title>Incident Response Playbooks \u2014 AI Replication Sandbox</title>"
            "<style>"
            "*{margin:0;padding:0;box-sizing:border-box}"
            "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;"
            "background:#0f172a;color:#e2e8f0;line-height:1.6;padding:2rem}"
            "h1{text-align:center;margin-bottom:.5rem;color:#f8fafc}"
            ".subtitle{text-align:center;color:#94a3b8;margin-bottom:2rem}"
            ".playbook-card{background:#1e293b;border-radius:8px;padding:1.5rem;margin-bottom:1.5rem}"
            ".playbook-card h3{color:#f1f5f9;margin-bottom:.75rem}"
            ".meta{display:flex;flex-wrap:wrap;gap:.5rem;margin-bottom:1rem;align-items:center}"
            ".severity{color:#fff;padding:2px 10px;border-radius:4px;font-size:.8rem;font-weight:600;"
            "text-transform:uppercase}"
            ".category{color:#94a3b8;font-size:.85rem}"
            ".tag{background:#334155;color:#cbd5e1;padding:2px 8px;border-radius:3px;font-size:.75rem}"
            ".description{color:#94a3b8;margin-bottom:1rem}"
            ".indicators h4,.phase h4,.escalation h4{color:#60a5fa;margin:1rem 0 .5rem;font-size:.95rem}"
            ".indicators ul,.phase ul{padding-left:1.5rem}"
            ".indicators li,.phase li{color:#cbd5e1;margin-bottom:.25rem}"
            ".objective{color:#a78bfa;font-style:italic;margin-bottom:.5rem;font-size:.9rem}"
            ".time-target{color:#34d399;font-size:.8rem;font-weight:normal}"
            ".note{color:#94a3b8;font-size:.85rem}"
            ".phase{margin-bottom:.75rem;padding-left:.5rem;border-left:2px solid #334155}"
            ".escalation table{width:100%;border-collapse:collapse;font-size:.85rem}"
            ".escalation th{text-align:left;color:#60a5fa;padding:4px 8px;border-bottom:1px solid #334155}"
            ".escalation td{padding:4px 8px;color:#cbd5e1;border-bottom:1px solid #1e293b}"
            ".modules{color:#64748b;font-size:.85rem;margin-top:1rem}"
            "@media(max-width:600px){body{padding:1rem}.playbook-card{padding:1rem}}"
            "</style></head><body>"
            "<h1>\U0001f6a8 Incident Response Playbooks</h1>"
            f'<p class="subtitle">AI Replication Sandbox \u2014 Generated {timestamp}</p>'
            f"{''.join(cards_html)}"
            "</body></html>"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None):
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m replication ir-playbook",
        description="Generate incident response playbooks for AI replication threats",
    )
    parser.add_argument(
        "--category", "-c",
        choices=[c.value for c in ThreatCategory],
        help="Generate playbook for a specific threat category",
    )
    parser.add_argument(
        "--severity", "-s",
        choices=[s.value for s in Severity],
        default="info",
        help="Minimum severity to include (default: info = all)",
    )
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--html", action="store_true", help="HTML output")
    parser.add_argument("-o", "--output", help="Write output to file")
    parser.add_argument("--list", action="store_true", dest="list_cats",
                        help="List available threat categories")
    parser.add_argument("--from-risk-profile", action="store_true",
                        help="Generate playbooks from risk profiler analysis")

    args = parser.parse_args(argv)

    if args.list_cats:
        print("Available threat categories:")
        for cat in ThreatCategory:
            pb = _PLAYBOOK_BUILDERS.get(cat)
            if pb:
                built = pb()
                print(f"  {SEVERITY_EMOJI[built.severity]} {cat.value:<25} {built.severity.value:<10} {built.title}")
        return

    if args.from_risk_profile:
        from .risk_profiler import RiskProfiler
        profiler = RiskProfiler()
        report = profiler.analyze()
        gen = PlaybookGenerator()
        playbooks = gen.from_risk_report(report)
    else:
        categories = None
        if args.category:
            categories = [ThreatCategory(args.category)]
        config = PlaybookConfig(
            categories=categories,
            min_severity=Severity(args.severity),
        )
        gen = PlaybookGenerator(config)
        playbooks = gen.generate()

    if not playbooks:
        print("No playbooks matched the specified criteria.")
        return

    if args.json:
        output = json.dumps([pb.to_dict() for pb in playbooks], indent=2)
    elif args.html:
        output = gen.render_html(playbooks)
    else:
        output = "\n\n".join(pb.render() for pb in playbooks)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Written to {args.output}")
    else:
        print(output)

    if not args.json and not args.html:
        print(f"\n{'\u2500' * 57}")
        print(f"  {len(playbooks)} playbook(s) generated")
        print(f"  Use --html -o playbook.html for a visual report")
        print(f"  Use --json for machine-readable output")


if __name__ == "__main__":
    main()
