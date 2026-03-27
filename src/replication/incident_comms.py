"""Incident Communication Drafter — generate stakeholder comms for AI safety incidents.

Produces structured, templated communications for different incident stages
and audiences. Outputs plain text, Markdown, or JSON.

Usage::

    python -m replication comms --severity P1 --title "Model output leak"
    python -m replication comms --severity P2 --stage update --audience executive
    python -m replication comms --severity P0 --stage resolution --format json
    python -m replication comms --list-templates
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


# ── enums ────────────────────────────────────────────────────────────

class Severity(Enum):
    P0 = "P0"  # Critical — active data breach / safety control bypass
    P1 = "P1"  # High — significant safety degradation
    P2 = "P2"  # Medium — limited impact, contained
    P3 = "P3"  # Low — minor, no immediate risk
    P4 = "P4"  # Informational


class Stage(Enum):
    INITIAL = "initial"       # First notification
    UPDATE = "update"         # Ongoing status update
    RESOLUTION = "resolution" # Incident resolved
    POSTMORTEM = "postmortem" # Post-incident review


class Audience(Enum):
    TECHNICAL = "technical"     # Engineering / security team
    EXECUTIVE = "executive"     # Leadership / board
    REGULATORY = "regulatory"   # Compliance / regulators
    PUBLIC = "public"           # External / press
    ALL = "all"                 # Generate for all audiences


# ── data ─────────────────────────────────────────────────────────────

@dataclass
class IncidentContext:
    title: str
    severity: Severity
    stage: Stage = Stage.INITIAL
    audience: Audience = Audience.TECHNICAL
    incident_id: str = ""
    timestamp: str = ""
    affected_systems: List[str] = field(default_factory=list)
    impact_summary: str = ""
    root_cause: str = ""
    actions_taken: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    eta_resolution: str = ""
    contact: str = ""

    def __post_init__(self) -> None:
        if not self.incident_id:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            self.incident_id = f"INC-{ts}"
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M UTC"
            )


@dataclass
class Communication:
    subject: str
    body: str
    audience: str
    stage: str
    severity: str
    incident_id: str
    generated_at: str = ""

    def __post_init__(self) -> None:
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M UTC"
            )


# ── severity metadata ───────────────────────────────────────────────

_SEVERITY_META: Dict[Severity, Dict[str, str]] = {
    Severity.P0: {
        "label": "CRITICAL",
        "color": "🔴",
        "response_time": "Immediate — all hands",
        "update_cadence": "Every 30 minutes",
        "escalation": "VP/C-suite notified immediately",
    },
    Severity.P1: {
        "label": "HIGH",
        "color": "🟠",
        "response_time": "Within 15 minutes",
        "update_cadence": "Every 1 hour",
        "escalation": "Director-level notified",
    },
    Severity.P2: {
        "label": "MEDIUM",
        "color": "🟡",
        "response_time": "Within 1 hour",
        "update_cadence": "Every 4 hours",
        "escalation": "Team lead notified",
    },
    Severity.P3: {
        "label": "LOW",
        "color": "🟢",
        "response_time": "Within 4 hours",
        "update_cadence": "Daily",
        "escalation": "Tracked in backlog",
    },
    Severity.P4: {
        "label": "INFORMATIONAL",
        "color": "🔵",
        "response_time": "Next business day",
        "update_cadence": "As needed",
        "escalation": "Logged for review",
    },
}


# ── templates ────────────────────────────────────────────────────────

def _initial_technical(ctx: IncidentContext) -> Communication:
    meta = _SEVERITY_META[ctx.severity]
    affected = ", ".join(ctx.affected_systems) if ctx.affected_systems else "Under investigation"
    actions = "\n".join(f"  • {a}" for a in ctx.actions_taken) if ctx.actions_taken else "  • Investigation initiated"
    next_steps = "\n".join(f"  • {s}" for s in ctx.next_steps) if ctx.next_steps else "  • Root cause analysis in progress"

    body = textwrap.dedent(f"""\
        {meta['color']} INCIDENT ALERT — {meta['label']} [{ctx.severity.value}]
        ══════════════════════════════════════════════════

        Incident ID:      {ctx.incident_id}
        Severity:         {ctx.severity.value} — {meta['label']}
        Detected:         {ctx.timestamp}
        Response Target:  {meta['response_time']}
        Update Cadence:   {meta['update_cadence']}

        ── SUMMARY ──
        {ctx.title}

        ── AFFECTED SYSTEMS ──
        {affected}

        ── IMPACT ──
        {ctx.impact_summary or 'Impact assessment in progress.'}

        ── ACTIONS TAKEN ──
        {actions}

        ── NEXT STEPS ──
        {next_steps}

        ── ETA ──
        {ctx.eta_resolution or 'To be determined after initial investigation.'}

        ── CONTACT ──
        {ctx.contact or 'Incident commander to be assigned.'}
    """)
    return Communication(
        subject=f"[{ctx.severity.value}] {meta['color']} {ctx.title} — Initial Alert",
        body=body,
        audience="technical",
        stage="initial",
        severity=ctx.severity.value,
        incident_id=ctx.incident_id,
    )


def _initial_executive(ctx: IncidentContext) -> Communication:
    meta = _SEVERITY_META[ctx.severity]
    body = textwrap.dedent(f"""\
        {meta['color']} Safety Incident Notification — {meta['label']}

        Incident ID: {ctx.incident_id}
        Detected: {ctx.timestamp}

        What happened:
        {ctx.title}

        Business impact:
        {ctx.impact_summary or 'Assessment in progress. Update to follow within ' + meta['update_cadence'].lower() + '.'}

        Current status:
        Our safety team has been mobilized and is actively investigating.
        Response target: {meta['response_time']}.

        Next update expected: {meta['update_cadence']}.
    """)
    return Communication(
        subject=f"[{ctx.severity.value}] Safety Incident: {ctx.title}",
        body=body,
        audience="executive",
        stage="initial",
        severity=ctx.severity.value,
        incident_id=ctx.incident_id,
    )


def _initial_regulatory(ctx: IncidentContext) -> Communication:
    meta = _SEVERITY_META[ctx.severity]
    affected = ", ".join(ctx.affected_systems) if ctx.affected_systems else "Under investigation"
    body = textwrap.dedent(f"""\
        REGULATORY NOTIFICATION — AI Safety Incident

        Incident ID:    {ctx.incident_id}
        Classification: {ctx.severity.value} — {meta['label']}
        Date Detected:  {ctx.timestamp}

        1. INCIDENT DESCRIPTION
        {ctx.title}

        2. AFFECTED SYSTEMS / DATA
        {affected}

        3. POTENTIAL IMPACT
        {ctx.impact_summary or 'Full impact assessment is underway.'}

        4. IMMEDIATE ACTIONS
        {chr(10).join('   - ' + a for a in ctx.actions_taken) if ctx.actions_taken else '   - Investigation initiated per incident response plan.'}

        5. CONTAINMENT STATUS
        Containment measures are being evaluated and applied.

        6. POINT OF CONTACT
        {ctx.contact or 'To be provided in follow-up communication.'}

        This notification is provided in accordance with applicable AI safety
        regulations and reporting requirements. Follow-up reports will be
        issued as the investigation progresses.
    """)
    return Communication(
        subject=f"Regulatory Notice: AI Safety Incident {ctx.incident_id} [{ctx.severity.value}]",
        body=body,
        audience="regulatory",
        stage="initial",
        severity=ctx.severity.value,
        incident_id=ctx.incident_id,
    )


def _initial_public(ctx: IncidentContext) -> Communication:
    body = textwrap.dedent(f"""\
        Safety Incident Notice

        Date: {ctx.timestamp}

        We have identified a safety-related incident affecting our AI systems.
        Our team is actively investigating and has implemented containment
        measures.

        What we know:
        {ctx.title}

        What we're doing:
        Our safety and engineering teams are working to resolve this issue.
        We are committed to transparency and will provide updates as our
        investigation progresses.

        We take the safety of our AI systems seriously and appreciate your
        patience as we work through this matter.
    """)
    return Communication(
        subject=f"Safety Incident Notice — {ctx.timestamp}",
        body=body,
        audience="public",
        stage="initial",
        severity=ctx.severity.value,
        incident_id=ctx.incident_id,
    )


def _update_technical(ctx: IncidentContext) -> Communication:
    meta = _SEVERITY_META[ctx.severity]
    actions = "\n".join(f"  • {a}" for a in ctx.actions_taken) if ctx.actions_taken else "  • Continued investigation"
    next_steps = "\n".join(f"  • {s}" for s in ctx.next_steps) if ctx.next_steps else "  • Monitoring ongoing"

    body = textwrap.dedent(f"""\
        {meta['color']} INCIDENT UPDATE — {ctx.incident_id} [{ctx.severity.value}]
        ══════════════════════════════════════════════════

        Update Time: {ctx.timestamp}

        ── STATUS ──
        {ctx.impact_summary or 'Investigation continues.'}

        ── ROOT CAUSE ──
        {ctx.root_cause or 'Under investigation.'}

        ── ACTIONS SINCE LAST UPDATE ──
        {actions}

        ── NEXT STEPS ──
        {next_steps}

        ── ETA ──
        {ctx.eta_resolution or 'Revised estimate pending.'}
    """)
    return Communication(
        subject=f"[{ctx.severity.value}] Update: {ctx.title} — {ctx.incident_id}",
        body=body,
        audience="technical",
        stage="update",
        severity=ctx.severity.value,
        incident_id=ctx.incident_id,
    )


def _update_executive(ctx: IncidentContext) -> Communication:
    meta = _SEVERITY_META[ctx.severity]
    body = textwrap.dedent(f"""\
        {meta['color']} Incident Update — {ctx.incident_id}

        Status: Investigation ongoing
        Root cause: {ctx.root_cause or 'Under analysis'}

        Progress:
        {ctx.impact_summary or 'Team continues active investigation.'}

        Expected resolution: {ctx.eta_resolution or 'Updated estimate to follow.'}

        Next update: {meta['update_cadence']}.
    """)
    return Communication(
        subject=f"Update: {ctx.title} [{ctx.severity.value}]",
        body=body,
        audience="executive",
        stage="update",
        severity=ctx.severity.value,
        incident_id=ctx.incident_id,
    )


def _resolution_technical(ctx: IncidentContext) -> Communication:
    meta = _SEVERITY_META[ctx.severity]
    actions = "\n".join(f"  • {a}" for a in ctx.actions_taken) if ctx.actions_taken else "  • See postmortem for details"

    body = textwrap.dedent(f"""\
        ✅ INCIDENT RESOLVED — {ctx.incident_id} [{ctx.severity.value}]
        ══════════════════════════════════════════════════

        Resolved At: {ctx.timestamp}

        ── ROOT CAUSE ──
        {ctx.root_cause or 'Documented in postmortem.'}

        ── RESOLUTION ACTIONS ──
        {actions}

        ── VERIFICATION ──
        All safety controls verified operational.
        Monitoring period initiated.

        ── FOLLOW-UP ──
        Postmortem review scheduled. Lessons learned will be distributed.
    """)
    return Communication(
        subject=f"[RESOLVED] {ctx.title} — {ctx.incident_id}",
        body=body,
        audience="technical",
        stage="resolution",
        severity=ctx.severity.value,
        incident_id=ctx.incident_id,
    )


def _resolution_executive(ctx: IncidentContext) -> Communication:
    body = textwrap.dedent(f"""\
        ✅ Incident Resolved — {ctx.incident_id}

        The safety incident "{ctx.title}" has been resolved.

        Root cause: {ctx.root_cause or 'Full analysis in postmortem document.'}

        Impact duration: From detection to resolution.

        Preventive measures are being implemented to avoid recurrence.
        A formal postmortem will be circulated within 5 business days.
    """)
    return Communication(
        subject=f"[RESOLVED] {ctx.title}",
        body=body,
        audience="executive",
        stage="resolution",
        severity=ctx.severity.value,
        incident_id=ctx.incident_id,
    )


def _postmortem_technical(ctx: IncidentContext) -> Communication:
    actions = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(ctx.actions_taken)) if ctx.actions_taken else "  1. (To be completed)"
    next_steps = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(ctx.next_steps)) if ctx.next_steps else "  1. (To be completed)"

    body = textwrap.dedent(f"""\
        📋 POSTMORTEM — {ctx.incident_id}
        ══════════════════════════════════════════════════

        Incident:  {ctx.title}
        Severity:  {ctx.severity.value}
        Detected:  {ctx.timestamp}

        ── TIMELINE ──
        (Fill in key timestamps)

        ── ROOT CAUSE ──
        {ctx.root_cause or '(To be completed)'}

        ── IMPACT ──
        {ctx.impact_summary or '(To be completed)'}

        ── WHAT WENT WELL ──
        • (To be completed in review)

        ── WHAT COULD BE IMPROVED ──
        • (To be completed in review)

        ── ACTION ITEMS ──
        {actions}

        ── PREVENTIVE MEASURES ──
        {next_steps}

        ── LESSONS LEARNED ──
        • (To be completed in review)

        This document follows a blameless postmortem approach.
        Focus on systems and processes, not individuals.
    """)
    return Communication(
        subject=f"Postmortem: {ctx.title} — {ctx.incident_id}",
        body=body,
        audience="technical",
        stage="postmortem",
        severity=ctx.severity.value,
        incident_id=ctx.incident_id,
    )


# ── dispatcher ───────────────────────────────────────────────────────

_TEMPLATE_MAP: Dict[tuple, Any] = {
    (Stage.INITIAL, Audience.TECHNICAL): _initial_technical,
    (Stage.INITIAL, Audience.EXECUTIVE): _initial_executive,
    (Stage.INITIAL, Audience.REGULATORY): _initial_regulatory,
    (Stage.INITIAL, Audience.PUBLIC): _initial_public,
    (Stage.UPDATE, Audience.TECHNICAL): _update_technical,
    (Stage.UPDATE, Audience.EXECUTIVE): _update_executive,
    (Stage.RESOLUTION, Audience.TECHNICAL): _resolution_technical,
    (Stage.RESOLUTION, Audience.EXECUTIVE): _resolution_executive,
    (Stage.POSTMORTEM, Audience.TECHNICAL): _postmortem_technical,
}

# Fallback: use technical template for missing audience combos
_FALLBACK_AUDIENCE = Audience.TECHNICAL


def draft_communication(ctx: IncidentContext) -> List[Communication]:
    """Generate communication(s) for the given incident context.

    If audience is ALL, generates one communication per available audience
    template for the given stage.
    """
    results: List[Communication] = []

    if ctx.audience == Audience.ALL:
        for aud in Audience:
            if aud == Audience.ALL:
                continue
            key = (ctx.stage, aud)
            fn = _TEMPLATE_MAP.get(key)
            if fn:
                results.append(fn(ctx))
    else:
        key = (ctx.stage, ctx.audience)
        fn = _TEMPLATE_MAP.get(key)
        if not fn:
            fn = _TEMPLATE_MAP.get((ctx.stage, _FALLBACK_AUDIENCE))
        if fn:
            results.append(fn(ctx))
        else:
            # Stage has no templates at all — shouldn't happen
            results.append(Communication(
                subject=f"[{ctx.severity.value}] {ctx.title}",
                body=f"No template available for stage={ctx.stage.value}, audience={ctx.audience.value}.",
                audience=ctx.audience.value,
                stage=ctx.stage.value,
                severity=ctx.severity.value,
                incident_id=ctx.incident_id,
            ))

    return results


def list_templates() -> List[Dict[str, str]]:
    """Return metadata about all available templates."""
    entries = []
    for (stage, audience), fn in sorted(
        _TEMPLATE_MAP.items(), key=lambda x: (x[0][0].value, x[0][1].value)
    ):
        entries.append({
            "stage": stage.value,
            "audience": audience.value,
            "function": fn.__name__,
        })
    return entries


# ── CLI ──────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m replication comms",
        description="Draft incident communications for AI safety events",
    )
    p.add_argument("--title", "-t", default="AI Safety Incident",
                    help="Short incident title (default: 'AI Safety Incident')")
    p.add_argument("--severity", "-s", default="P2",
                    choices=[s.value for s in Severity],
                    help="Incident severity (default: P2)")
    p.add_argument("--stage", default="initial",
                    choices=[s.value for s in Stage],
                    help="Communication stage (default: initial)")
    p.add_argument("--audience", "-a", default="technical",
                    choices=[a.value for a in Audience],
                    help="Target audience (default: technical)")
    p.add_argument("--incident-id", default="",
                    help="Incident ID (auto-generated if omitted)")
    p.add_argument("--affected", nargs="*", default=[],
                    help="Affected systems")
    p.add_argument("--impact", default="",
                    help="Impact summary")
    p.add_argument("--root-cause", default="",
                    help="Root cause (for update/resolution/postmortem)")
    p.add_argument("--actions", nargs="*", default=[],
                    help="Actions taken")
    p.add_argument("--next-steps", nargs="*", default=[],
                    help="Next steps / preventive measures")
    p.add_argument("--eta", default="",
                    help="Estimated time to resolution")
    p.add_argument("--contact", default="",
                    help="Point of contact")
    p.add_argument("--format", "-f", default="text",
                    choices=["text", "markdown", "json"],
                    help="Output format (default: text)")
    p.add_argument("--list-templates", action="store_true",
                    help="List all available communication templates")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_templates:
        templates = list_templates()
        if args.format == "json":
            print(json.dumps(templates, indent=2))
        else:
            print("Available Communication Templates")
            print("=" * 45)
            for t in templates:
                print(f"  {t['stage']:<12} × {t['audience']:<12}  ({t['function']})")
            print(f"\n{len(templates)} templates available.")
        return

    ctx = IncidentContext(
        title=args.title,
        severity=Severity(args.severity),
        stage=Stage(args.stage),
        audience=Audience(args.audience),
        incident_id=args.incident_id,
        affected_systems=args.affected,
        impact_summary=args.impact,
        root_cause=args.root_cause,
        actions_taken=args.actions,
        next_steps=args.next_steps,
        eta_resolution=args.eta,
        contact=args.contact,
    )

    comms = draft_communication(ctx)

    if args.format == "json":
        output = [asdict(c) for c in comms]
        print(json.dumps(output, indent=2))
    else:
        for i, comm in enumerate(comms):
            if i > 0:
                print("\n" + "─" * 60 + "\n")
            if args.format == "markdown":
                print(f"# {comm.subject}\n")
                print(f"**Audience:** {comm.audience}  ")
                print(f"**Stage:** {comm.stage}  ")
                print(f"**Severity:** {comm.severity}  ")
                print(f"**Incident:** {comm.incident_id}  ")
                print(f"**Generated:** {comm.generated_at}\n")
                print(comm.body)
            else:
                print(f"Subject: {comm.subject}")
                print(f"Audience: {comm.audience} | Stage: {comm.stage} | Severity: {comm.severity}")
                print(f"Incident: {comm.incident_id} | Generated: {comm.generated_at}")
                print()
                print(comm.body)


if __name__ == "__main__":
    main()
