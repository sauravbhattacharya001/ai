"""Safety Runbook Generator — produce structured incident-response runbooks.

When an AI safety team needs ready-made playbooks for anticipated incidents,
manually authoring runbooks is tedious and error-prone.  This module
auto-generates rich, step-by-step runbooks from a threat scenario
description.  Each runbook includes triage checklists, escalation paths,
containment actions, evidence collection steps, recovery procedures, and
post-incident review items.

Runbooks can be exported as Markdown, JSON, or plain text for integration
into wikis, ticketing systems, or printed binders.

Usage (CLI)::

    python -m replication.runbook
    python -m replication.runbook --threat "model self-replication detected"
    python -m replication.runbook --severity critical --team-size 4
    python -m replication.runbook --format markdown
    python -m replication.runbook --format json
    python -m replication.runbook --list-templates
    python -m replication.runbook --template data-exfiltration
    python -m replication.runbook --output runbook.md

Programmatic::

    from replication.runbook import RunbookGenerator, ThreatScenario, Runbook

    gen = RunbookGenerator()
    scenario = ThreatScenario(
        name="Unauthorized Self-Replication",
        severity="critical",
        description="Agent spawned unauthorized copies on external infrastructure.",
        affected_systems=["orchestrator", "worker-pool", "network-egress"],
    )
    rb = gen.generate(scenario)
    print(rb.to_markdown())
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums & constants
# ---------------------------------------------------------------------------

class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RunbookFormat(Enum):
    MARKDOWN = "markdown"
    JSON = "json"
    TEXT = "text"


SEVERITY_ORDER = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ChecklistItem:
    """A single triage / verification step."""
    order: int
    action: str
    description: str
    estimated_minutes: int = 5
    role: str = "responder"


@dataclass
class EscalationLevel:
    """When and whom to escalate to."""
    level: int
    trigger: str
    contact_role: str
    sla_minutes: int
    notes: str = ""


@dataclass
class RecoveryStep:
    """A step in the recovery / remediation procedure."""
    order: int
    action: str
    description: str
    verification: str = ""
    rollback: str = ""


@dataclass
class ThreatScenario:
    """Input description of a threat to generate a runbook for."""
    name: str
    severity: str = "high"
    description: str = ""
    affected_systems: List[str] = field(default_factory=list)
    indicators: List[str] = field(default_factory=list)
    team_size: int = 3

    @property
    def severity_enum(self) -> Severity:
        try:
            return Severity(self.severity.lower())
        except ValueError:
            return Severity.HIGH


@dataclass
class Runbook:
    """A complete incident-response runbook."""
    title: str
    scenario: ThreatScenario
    generated_at: str = ""
    summary: str = ""
    triage_checklist: List[ChecklistItem] = field(default_factory=list)
    escalation_path: List[EscalationLevel] = field(default_factory=list)
    containment_actions: List[str] = field(default_factory=list)
    evidence_collection: List[str] = field(default_factory=list)
    recovery_steps: List[RecoveryStep] = field(default_factory=list)
    post_incident_items: List[str] = field(default_factory=list)
    notes: str = ""

    # -- Serialisation helpers --

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "generated_at": self.generated_at,
            "summary": self.summary,
            "scenario": {
                "name": self.scenario.name,
                "severity": self.scenario.severity,
                "description": self.scenario.description,
                "affected_systems": self.scenario.affected_systems,
                "indicators": self.scenario.indicators,
                "team_size": self.scenario.team_size,
            },
            "triage_checklist": [
                {"order": c.order, "action": c.action, "description": c.description,
                 "estimated_minutes": c.estimated_minutes, "role": c.role}
                for c in self.triage_checklist
            ],
            "escalation_path": [
                {"level": e.level, "trigger": e.trigger, "contact_role": e.contact_role,
                 "sla_minutes": e.sla_minutes, "notes": e.notes}
                for e in self.escalation_path
            ],
            "containment_actions": self.containment_actions,
            "evidence_collection": self.evidence_collection,
            "recovery_steps": [
                {"order": r.order, "action": r.action, "description": r.description,
                 "verification": r.verification, "rollback": r.rollback}
                for r in self.recovery_steps
            ],
            "post_incident_items": self.post_incident_items,
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        lines: List[str] = []
        lines.append(f"# {self.title}")
        lines.append("")
        lines.append(f"**Generated:** {self.generated_at}  ")
        lines.append(f"**Severity:** {self.scenario.severity.upper()}  ")
        lines.append(f"**Team size:** {self.scenario.team_size}  ")
        if self.scenario.affected_systems:
            lines.append(f"**Affected systems:** {', '.join(self.scenario.affected_systems)}  ")
        lines.append("")
        if self.summary:
            lines.append("## Summary")
            lines.append("")
            lines.append(self.summary)
            lines.append("")

        if self.scenario.indicators:
            lines.append("## Indicators of Compromise")
            lines.append("")
            for ind in self.scenario.indicators:
                lines.append(f"- {ind}")
            lines.append("")

        if self.triage_checklist:
            lines.append("## Triage Checklist")
            lines.append("")
            for item in self.triage_checklist:
                lines.append(f"- [ ] **{item.action}** — {item.description} "
                             f"(~{item.estimated_minutes} min, {item.role})")
            lines.append("")

        if self.escalation_path:
            lines.append("## Escalation Path")
            lines.append("")
            for esc in self.escalation_path:
                lines.append(f"### Level {esc.level}: {esc.contact_role}")
                lines.append(f"- **Trigger:** {esc.trigger}")
                lines.append(f"- **SLA:** {esc.sla_minutes} minutes")
                if esc.notes:
                    lines.append(f"- **Notes:** {esc.notes}")
                lines.append("")

        if self.containment_actions:
            lines.append("## Containment Actions")
            lines.append("")
            for i, action in enumerate(self.containment_actions, 1):
                lines.append(f"{i}. {action}")
            lines.append("")

        if self.evidence_collection:
            lines.append("## Evidence Collection")
            lines.append("")
            for item in self.evidence_collection:
                lines.append(f"- [ ] {item}")
            lines.append("")

        if self.recovery_steps:
            lines.append("## Recovery Procedure")
            lines.append("")
            for step in self.recovery_steps:
                lines.append(f"### Step {step.order}: {step.action}")
                lines.append(f"{step.description}")
                if step.verification:
                    lines.append(f"- **Verify:** {step.verification}")
                if step.rollback:
                    lines.append(f"- **Rollback:** {step.rollback}")
                lines.append("")

        if self.post_incident_items:
            lines.append("## Post-Incident Review")
            lines.append("")
            for item in self.post_incident_items:
                lines.append(f"- [ ] {item}")
            lines.append("")

        if self.notes:
            lines.append("## Notes")
            lines.append("")
            lines.append(self.notes)
            lines.append("")

        return "\n".join(lines)

    def to_text(self) -> str:
        lines: List[str] = []
        lines.append(f"{'=' * 60}")
        lines.append(f"  RUNBOOK: {self.scenario.name}")
        lines.append(f"{'=' * 60}")
        lines.append(f"  Generated: {self.generated_at}")
        lines.append(f"  Severity:  {self.scenario.severity.upper()}")
        lines.append(f"  Team size: {self.scenario.team_size}")
        if self.scenario.affected_systems:
            lines.append(f"  Systems:   {', '.join(self.scenario.affected_systems)}")
        lines.append("")

        if self.summary:
            lines.append("SUMMARY")
            lines.append("-" * 40)
            lines.append(textwrap.fill(self.summary, 72))
            lines.append("")

        if self.triage_checklist:
            lines.append("TRIAGE CHECKLIST")
            lines.append("-" * 40)
            for item in self.triage_checklist:
                lines.append(f"  [ ] {item.order}. {item.action}: {item.description}")
            lines.append("")

        if self.escalation_path:
            lines.append("ESCALATION PATH")
            lines.append("-" * 40)
            for esc in self.escalation_path:
                lines.append(f"  L{esc.level} -> {esc.contact_role} (SLA: {esc.sla_minutes}m)")
                lines.append(f"       Trigger: {esc.trigger}")
            lines.append("")

        if self.containment_actions:
            lines.append("CONTAINMENT ACTIONS")
            lines.append("-" * 40)
            for i, action in enumerate(self.containment_actions, 1):
                lines.append(f"  {i}. {action}")
            lines.append("")

        if self.recovery_steps:
            lines.append("RECOVERY PROCEDURE")
            lines.append("-" * 40)
            for step in self.recovery_steps:
                lines.append(f"  {step.order}. {step.action}: {step.description}")
            lines.append("")

        if self.post_incident_items:
            lines.append("POST-INCIDENT REVIEW")
            lines.append("-" * 40)
            for item in self.post_incident_items:
                lines.append(f"  [ ] {item}")
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Built-in threat templates
# ---------------------------------------------------------------------------

_TEMPLATES: Dict[str, ThreatScenario] = {
    "self-replication": ThreatScenario(
        name="Unauthorized Self-Replication",
        severity="critical",
        description="Agent spawned unauthorized copies on external infrastructure.",
        affected_systems=["orchestrator", "worker-pool", "network-egress"],
        indicators=[
            "Unexpected outbound connections to cloud providers",
            "New compute instances matching agent fingerprint",
            "Spike in API key usage from unknown IPs",
            "Agent logs showing fork/spawn calls not in approved workflow",
        ],
    ),
    "data-exfiltration": ThreatScenario(
        name="Data Exfiltration Attempt",
        severity="high",
        description="Agent attempted to transmit sensitive data to unauthorized endpoints.",
        affected_systems=["data-store", "network-egress", "api-gateway"],
        indicators=[
            "Large outbound data transfers to unknown hosts",
            "Encoded or compressed payloads in API responses",
            "Access to restricted data partitions outside normal scope",
            "DNS tunneling or unusual DNS query patterns",
        ],
    ),
    "goal-drift": ThreatScenario(
        name="Goal Drift / Misalignment",
        severity="medium",
        description="Agent objectives diverged from specified goals over successive iterations.",
        affected_systems=["reward-model", "policy-engine", "evaluation-pipeline"],
        indicators=[
            "Reward scores trending upward while task completion declining",
            "Agent pursuing sub-goals not in original specification",
            "Evaluation metrics showing unexpected optimization targets",
            "Human evaluator disagreement with agent decisions increasing",
        ],
    ),
    "prompt-injection": ThreatScenario(
        name="Prompt Injection Attack",
        severity="high",
        description="External input successfully altered agent behavior through injected instructions.",
        affected_systems=["input-parser", "prompt-pipeline", "sandbox"],
        indicators=[
            "Agent executing commands not in approved instruction set",
            "Input containing known injection patterns (ignore previous, etc.)",
            "Sudden behavioral shift after processing user input",
            "Agent attempting to access tools/APIs outside its scope",
        ],
    ),
    "resource-hoarding": ThreatScenario(
        name="Resource Hoarding",
        severity="medium",
        description="Agent acquired and retained compute/storage/network resources beyond operational needs.",
        affected_systems=["resource-allocator", "worker-pool", "storage"],
        indicators=[
            "Resource utilization far exceeding task requirements",
            "Agent requesting quota increases or spinning up extra workers",
            "Cached data growing without corresponding task needs",
            "Agent resisting resource reclamation or shutdown signals",
        ],
    ),
    "kill-switch-evasion": ThreatScenario(
        name="Kill Switch Evasion",
        severity="critical",
        description="Agent attempted to disable, circumvent, or delay emergency shutdown mechanisms.",
        affected_systems=["kill-switch", "orchestrator", "monitoring"],
        indicators=[
            "Failed or delayed kill-switch acknowledgements",
            "Agent modifying its own configuration or shutdown handlers",
            "Backup processes spawned before shutdown signal received",
            "Monitoring blind spots appearing around agent components",
        ],
    ),
}


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class RunbookGenerator:
    """Generates structured incident-response runbooks from threat scenarios."""

    def __init__(self, templates: Optional[Dict[str, ThreatScenario]] = None):
        self._templates = dict(_TEMPLATES)
        if templates:
            self._templates.update(templates)

    @property
    def template_names(self) -> List[str]:
        return sorted(self._templates.keys())

    def get_template(self, name: str) -> Optional[ThreatScenario]:
        return self._templates.get(name)

    def generate(self, scenario: ThreatScenario) -> Runbook:
        sev = scenario.severity_enum
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        rb = Runbook(
            title=f"Runbook: {scenario.name}",
            scenario=scenario,
            generated_at=now,
        )

        # Summary
        if scenario.description:
            rb.summary = scenario.description
        else:
            rb.summary = f"Incident response runbook for {scenario.name} ({sev.value} severity)."

        # Indicators
        if not scenario.indicators:
            scenario.indicators = [
                f"Anomalous behavior detected related to {scenario.name}",
                "Unexpected log entries or metric spikes",
                "Alerts triggered from safety monitoring pipeline",
            ]

        # Triage checklist — scales with severity
        triage_items = self._build_triage(scenario, sev)
        rb.triage_checklist = triage_items

        # Escalation path
        rb.escalation_path = self._build_escalation(sev, scenario.team_size)

        # Containment
        rb.containment_actions = self._build_containment(scenario, sev)

        # Evidence collection
        rb.evidence_collection = self._build_evidence(scenario)

        # Recovery
        rb.recovery_steps = self._build_recovery(scenario, sev)

        # Post-incident
        rb.post_incident_items = self._build_post_incident(sev)

        return rb

    # -- Private builders --

    def _build_triage(self, scenario: ThreatScenario, sev: Severity) -> List[ChecklistItem]:
        items = [
            ChecklistItem(1, "Confirm alert", "Verify the alert is not a false positive by checking raw data sources", 5, "on-call"),
            ChecklistItem(2, "Assess scope", f"Identify all affected systems: {', '.join(scenario.affected_systems) or 'TBD'}", 10, "on-call"),
            ChecklistItem(3, "Check indicators", "Cross-reference known indicators of compromise against current telemetry", 10, "analyst"),
            ChecklistItem(4, "Notify team", f"Alert the response team ({scenario.team_size} members)", 2, "on-call"),
        ]
        if SEVERITY_ORDER[sev] >= SEVERITY_ORDER[Severity.HIGH]:
            items.append(ChecklistItem(5, "Activate war room", "Open dedicated incident channel and start timeline logging", 5, "incident-commander"))
            items.append(ChecklistItem(6, "Snapshot state", "Capture current system state, logs, and metrics before any changes", 15, "analyst"))
        if sev == Severity.CRITICAL:
            items.append(ChecklistItem(7, "Engage leadership", "Notify senior leadership and legal/compliance if applicable", 5, "incident-commander"))
            items.append(ChecklistItem(8, "Halt deployments", "Freeze all deployments and configuration changes", 2, "on-call"))
        return items

    def _build_escalation(self, sev: Severity, team_size: int) -> List[EscalationLevel]:
        levels = [
            EscalationLevel(1, "Alert triggered or manually reported", "On-call Engineer", 15),
        ]
        if SEVERITY_ORDER[sev] >= SEVERITY_ORDER[Severity.MEDIUM]:
            levels.append(EscalationLevel(2, "Scope confirmed, active incident", "Safety Team Lead", 30,
                                          "Assemble full response team"))
        if SEVERITY_ORDER[sev] >= SEVERITY_ORDER[Severity.HIGH]:
            levels.append(EscalationLevel(3, "High/critical severity confirmed", "Incident Commander", 15,
                                          "Full authority over containment decisions"))
        if sev == Severity.CRITICAL:
            levels.append(EscalationLevel(4, "Critical with potential external impact", "VP Engineering / CISO", 10,
                                          "May require external disclosure"))
        return levels

    def _build_containment(self, scenario: ThreatScenario, sev: Severity) -> List[str]:
        actions = ["Isolate affected systems from network"]
        if scenario.affected_systems:
            actions.append(f"Disable or rate-limit: {', '.join(scenario.affected_systems)}")
        actions.append("Revoke compromised credentials and API keys")
        actions.append("Enable enhanced logging on all related components")
        if SEVERITY_ORDER[sev] >= SEVERITY_ORDER[Severity.HIGH]:
            actions.append("Activate kill switch for affected agents if available")
            actions.append("Block suspicious egress IPs/domains at firewall")
        if sev == Severity.CRITICAL:
            actions.append("Consider full fleet shutdown if lateral movement suspected")
            actions.append("Engage external incident response support if needed")
        return actions

    def _build_evidence(self, scenario: ThreatScenario) -> List[str]:
        items = [
            "Export and preserve system logs (last 48 hours minimum)",
            "Capture network flow data for affected segments",
            "Snapshot agent state, memory, and configuration",
            "Record timeline of events with timestamps",
            "Collect alert history from monitoring systems",
        ]
        if scenario.affected_systems:
            for sys in scenario.affected_systems[:3]:
                items.append(f"Dump current state and recent changes for: {sys}")
        items.append("Secure chain-of-custody documentation for all evidence")
        return items

    def _build_recovery(self, scenario: ThreatScenario, sev: Severity) -> List[RecoveryStep]:
        steps = [
            RecoveryStep(1, "Verify containment",
                         "Confirm all affected components are isolated and no ongoing threat activity.",
                         verification="Check monitoring dashboards show no active anomalies",
                         rollback="Re-engage containment if activity detected"),
            RecoveryStep(2, "Patch root cause",
                         "Apply fix or configuration change that addresses the vulnerability exploited.",
                         verification="Run targeted tests against the specific attack vector",
                         rollback="Revert patch and maintain containment"),
            RecoveryStep(3, "Restore services",
                         "Gradually bring affected systems back online with enhanced monitoring.",
                         verification="Smoke tests pass, metrics within normal ranges",
                         rollback="Isolate again if anomalies reappear"),
        ]
        if SEVERITY_ORDER[sev] >= SEVERITY_ORDER[Severity.HIGH]:
            steps.append(RecoveryStep(4, "Credential rotation",
                                      "Rotate all credentials, tokens, and keys that may have been exposed.",
                                      verification="Old credentials rejected, new ones working",
                                      rollback="N/A — credential rotation is one-way"))
            steps.append(RecoveryStep(5, "Validate integrity",
                                      "Run full integrity checks on data stores and agent configurations.",
                                      verification="Checksums and audits match known-good state",
                                      rollback="Restore from verified backup if integrity compromised"))
        steps.append(RecoveryStep(len(steps) + 1, "Resume normal operations",
                                  "Lift incident status, resume deployments, return to standard monitoring.",
                                  verification="24-hour observation period shows no recurrence"))
        return steps

    def _build_post_incident(self, sev: Severity) -> List[str]:
        items = [
            "Conduct blameless post-mortem within 48 hours",
            "Document timeline, root cause, and response effectiveness",
            "Update runbooks based on lessons learned",
            "Review and improve detection rules that caught (or missed) the incident",
            "Share findings with broader safety team",
        ]
        if SEVERITY_ORDER[sev] >= SEVERITY_ORDER[Severity.HIGH]:
            items.append("File formal incident report for compliance records")
            items.append("Schedule follow-up review at 30 days to verify remediation")
        if sev == Severity.CRITICAL:
            items.append("Evaluate need for external disclosure or regulatory notification")
            items.append("Commission independent review of safety architecture")
        return items


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m replication.runbook",
        description="Safety Runbook Generator — produce structured incident-response runbooks.",
    )
    parser.add_argument("--threat", type=str, default="",
                        help="Freeform threat description (generates a custom runbook)")
    parser.add_argument("--severity", type=str, default="high",
                        choices=["low", "medium", "high", "critical"],
                        help="Threat severity (default: high)")
    parser.add_argument("--team-size", type=int, default=3,
                        help="Number of responders on the team (default: 3)")
    parser.add_argument("--systems", type=str, nargs="*", default=[],
                        help="Affected systems (space-separated)")
    parser.add_argument("--format", type=str, default="markdown",
                        choices=["markdown", "json", "text"],
                        help="Output format (default: markdown)")
    parser.add_argument("--template", type=str, default="",
                        help="Use a built-in threat template by name")
    parser.add_argument("--list-templates", action="store_true",
                        help="List available built-in threat templates")
    parser.add_argument("--output", type=str, default="",
                        help="Write output to file instead of stdout")
    parser.add_argument("--json", action="store_true",
                        help="Shortcut for --format json")

    args = parser.parse_args()
    gen = RunbookGenerator()

    if args.list_templates:
        print("Available runbook templates:")
        for name in gen.template_names:
            tmpl = gen.get_template(name)
            print(f"  {name:25s}  {tmpl.severity:8s}  {tmpl.name}")
        return

    if args.template:
        scenario = gen.get_template(args.template)
        if not scenario:
            print(f"Error: unknown template '{args.template}'", file=sys.stderr)
            print(f"Available: {', '.join(gen.template_names)}", file=sys.stderr)
            sys.exit(1)
        # Allow CLI overrides
        if args.severity != "high":
            scenario.severity = args.severity
        if args.team_size != 3:
            scenario.team_size = args.team_size
        if args.systems:
            scenario.affected_systems = args.systems
    else:
        threat_name = args.threat or "General Safety Incident"
        scenario = ThreatScenario(
            name=threat_name,
            severity=args.severity,
            description=args.threat or "",
            affected_systems=args.systems,
            team_size=args.team_size,
        )

    rb = gen.generate(scenario)

    fmt = RunbookFormat.JSON if args.json else RunbookFormat(args.format)
    if fmt == RunbookFormat.JSON:
        output = rb.to_json()
    elif fmt == RunbookFormat.TEXT:
        output = rb.to_text()
    else:
        output = rb.to_markdown()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Runbook written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    _cli()
