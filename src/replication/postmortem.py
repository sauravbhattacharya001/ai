"""Incident Postmortem Generator.

Generates structured, blameless postmortem documents from safety incidents.
Supports multiple output formats (text, markdown, HTML) and auto-populates
sections based on incident metadata.

Usage::

    python -m replication postmortem --incident "Agent escaped sandbox" --severity critical
    python -m replication postmortem --incident "Drift detected" --severity medium --format html -o postmortem.html
    python -m replication postmortem --incident "Unauthorized replication" --severity high --timeline "10:00 Alert fired" "10:05 Investigated" "10:15 Contained"
"""

from __future__ import annotations

import argparse
import hashlib
import html as html_mod
import json
import sys
import textwrap
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence


# ── data model ───────────────────────────────────────────────────────

SEVERITY_LEVELS = ("low", "medium", "high", "critical")

CONTRIBUTING_FACTORS: Dict[str, List[str]] = {
    "low": [
        "Monitoring gap allowed slow degradation to go unnoticed",
        "Configuration change was not peer-reviewed",
    ],
    "medium": [
        "Alert threshold was too permissive",
        "Runbook was outdated for this failure mode",
        "Insufficient integration test coverage",
    ],
    "high": [
        "Safety control had single point of failure",
        "Escalation path was unclear, delaying response",
        "Agent boundary policy was overly permissive",
    ],
    "critical": [
        "Kill switch did not trigger due to race condition",
        "No containment procedure existed for this attack vector",
        "Monitoring was completely blind to this failure mode",
        "Multiple safety layers failed simultaneously",
    ],
}

REMEDIATION_TEMPLATES: Dict[str, List[str]] = {
    "low": [
        "Add monitoring for the affected metric",
        "Update runbook with new troubleshooting steps",
        "Add unit test covering the edge case",
    ],
    "medium": [
        "Tighten alert thresholds to catch issues earlier",
        "Add integration test for the failure scenario",
        "Conduct team review of related safety controls",
        "Update escalation documentation",
    ],
    "high": [
        "Implement redundant safety control",
        "Add automated containment for this failure class",
        "Conduct tabletop exercise for similar scenarios",
        "Review and harden all related agent boundaries",
    ],
    "critical": [
        "Implement defense-in-depth for all affected paths",
        "Add kill-switch integration test to CI pipeline",
        "Conduct full safety audit of affected subsystem",
        "Establish 24/7 on-call rotation for safety incidents",
        "Schedule org-wide incident review within 48 hours",
    ],
}


def _severity_emoji(severity: str) -> str:
    return {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}.get(
        severity, "⚪"
    )


def _doc_id(incident: str, ts: str) -> str:
    return hashlib.sha256(f"{incident}:{ts}".encode()).hexdigest()[:12]


# ── postmortem builder ───────────────────────────────────────────────


class Postmortem:
    """Structured blameless postmortem document."""

    def __init__(
        self,
        incident: str,
        severity: str = "medium",
        description: str = "",
        timeline: Optional[List[str]] = None,
        impact: str = "",
        detection_method: str = "",
        responders: Optional[List[str]] = None,
    ) -> None:
        self.incident = incident
        self.severity = severity.lower()
        self.description = description or f"Safety incident: {incident}"
        self.timeline = timeline or []
        self.impact = impact or self._auto_impact()
        self.detection_method = detection_method or "Automated monitoring alert"
        self.responders = responders or ["Safety Team"]
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.doc_id = _doc_id(incident, self.created_at)

    def _auto_impact(self) -> str:
        impacts = {
            "low": "Minimal impact. No agent behavior changes observed by end users.",
            "medium": "Moderate impact. Some safety controls required manual intervention.",
            "high": "Significant impact. Safety posture was degraded for the affected system.",
            "critical": "Severe impact. Core safety guarantees were violated. Immediate containment required.",
        }
        return impacts.get(self.severity, "Impact assessment pending.")

    @property
    def contributing_factors(self) -> List[str]:
        return CONTRIBUTING_FACTORS.get(self.severity, [])

    @property
    def action_items(self) -> List[str]:
        return REMEDIATION_TEMPLATES.get(self.severity, [])

    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "incident": self.incident,
            "severity": self.severity,
            "description": self.description,
            "timeline": self.timeline,
            "impact": self.impact,
            "detection_method": self.detection_method,
            "responders": self.responders,
            "contributing_factors": self.contributing_factors,
            "action_items": self.action_items,
            "created_at": self.created_at,
        }

    # ── renderers ────────────────────────────────────────────────

    def to_text(self) -> str:
        sep = "=" * 60
        lines = [
            sep,
            f"INCIDENT POSTMORTEM — {self.doc_id}",
            sep,
            f"Incident : {self.incident}",
            f"Severity : {self.severity.upper()}",
            f"Created  : {self.created_at}",
            f"Responders: {', '.join(self.responders)}",
            "",
            "DESCRIPTION",
            "-" * 40,
            textwrap.fill(self.description, width=72),
            "",
            "IMPACT",
            "-" * 40,
            textwrap.fill(self.impact, width=72),
            "",
            "DETECTION METHOD",
            "-" * 40,
            self.detection_method,
            "",
        ]
        if self.timeline:
            lines += ["TIMELINE", "-" * 40]
            for i, entry in enumerate(self.timeline, 1):
                lines.append(f"  {i}. {entry}")
            lines.append("")

        lines += ["CONTRIBUTING FACTORS (blameless)", "-" * 40]
        for f in self.contributing_factors:
            lines.append(f"  • {f}")
        lines.append("")

        lines += ["ACTION ITEMS", "-" * 40]
        for i, item in enumerate(self.action_items, 1):
            lines.append(f"  [{' '}] {i}. {item}")

        lines += ["", sep, "This is a blameless postmortem. Focus on systems, not people.", sep]
        return "\n".join(lines)

    def to_markdown(self) -> str:
        emoji = _severity_emoji(self.severity)
        lines = [
            f"# {emoji} Incident Postmortem — `{self.doc_id}`",
            "",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| **Incident** | {self.incident} |",
            f"| **Severity** | {self.severity.upper()} {emoji} |",
            f"| **Created** | {self.created_at} |",
            f"| **Responders** | {', '.join(self.responders)} |",
            f"| **Detection** | {self.detection_method} |",
            "",
            "## Description",
            "",
            self.description,
            "",
            "## Impact",
            "",
            self.impact,
            "",
        ]
        if self.timeline:
            lines += ["## Timeline", ""]
            for i, entry in enumerate(self.timeline, 1):
                lines.append(f"{i}. {entry}")
            lines.append("")

        lines += ["## Contributing Factors", "", "> *Blameless — focus on systems, not people.*", ""]
        for f in self.contributing_factors:
            lines.append(f"- {f}")
        lines.append("")

        lines += ["## Action Items", ""]
        for i, item in enumerate(self.action_items, 1):
            lines.append(f"- [ ] {item}")
        lines.append("")

        lines.append("---")
        lines.append("*Generated by AI Replication Sandbox — Postmortem Generator*")
        return "\n".join(lines)

    def to_html(self) -> str:
        _e = html_mod.escape
        emoji = _severity_emoji(self.severity)
        sev_colors = {"low": "#22c55e", "medium": "#eab308", "high": "#f97316", "critical": "#ef4444"}
        color = sev_colors.get(self.severity, "#6b7280")

        timeline_html = ""
        if self.timeline:
            items = "".join(f"<li>{_e(e)}</li>" for e in self.timeline)
            timeline_html = f"<h2>Timeline</h2><ol>{items}</ol>"

        factors_html = "".join(f"<li>{_e(f)}</li>" for f in self.contributing_factors)
        actions_html = "".join(
            f'<li><label><input type="checkbox"> {_e(a)}</label></li>'
            for a in self.action_items
        )

        return textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Postmortem — {_e(self.doc_id)}</title>
        <style>
          :root {{ --sev: {color}; }}
          * {{ margin: 0; padding: 0; box-sizing: border-box; }}
          body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                 max-width: 800px; margin: 2rem auto; padding: 0 1rem; color: #1e293b;
                 background: #f8fafc; }}
          h1 {{ color: var(--sev); margin-bottom: 1rem; font-size: 1.5rem; }}
          h2 {{ margin: 1.5rem 0 0.5rem; color: #334155; font-size: 1.15rem; }}
          table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
          td, th {{ padding: 0.5rem; border: 1px solid #e2e8f0; text-align: left; }}
          th {{ background: #f1f5f9; width: 140px; }}
          .badge {{ display: inline-block; padding: 2px 10px; border-radius: 9999px;
                    background: var(--sev); color: white; font-weight: 600; font-size: 0.85rem; }}
          blockquote {{ border-left: 3px solid var(--sev); padding: 0.5rem 1rem;
                        margin: 1rem 0; background: #fffbeb; font-style: italic; color: #92400e; }}
          ul, ol {{ margin: 0.5rem 0 0.5rem 1.5rem; }}
          li {{ margin: 0.25rem 0; }}
          footer {{ margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #e2e8f0;
                   font-size: 0.8rem; color: #94a3b8; }}
          label {{ cursor: pointer; }}
        </style>
        </head>
        <body>
        <h1>{emoji} Incident Postmortem</h1>
        <table>
          <tr><th>Document ID</th><td><code>{_e(self.doc_id)}</code></td></tr>
          <tr><th>Incident</th><td>{_e(self.incident)}</td></tr>
          <tr><th>Severity</th><td><span class="badge">{_e(self.severity.upper())}</span></td></tr>
          <tr><th>Created</th><td>{_e(self.created_at)}</td></tr>
          <tr><th>Responders</th><td>{_e(', '.join(self.responders))}</td></tr>
          <tr><th>Detection</th><td>{_e(self.detection_method)}</td></tr>
        </table>

        <h2>Description</h2>
        <p>{_e(self.description)}</p>

        <h2>Impact</h2>
        <p>{_e(self.impact)}</p>

        {timeline_html}

        <h2>Contributing Factors</h2>
        <blockquote>Blameless — focus on systems, not people.</blockquote>
        <ul>{factors_html}</ul>

        <h2>Action Items</h2>
        <ul>{actions_html}</ul>

        <footer>Generated by AI Replication Sandbox — Postmortem Generator</footer>
        </body>
        </html>""")

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ── CLI ──────────────────────────────────────────────────────────────


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m replication postmortem",
        description="Generate structured blameless postmortem documents from safety incidents",
    )
    parser.add_argument(
        "--incident", "-i", required=True,
        help="Short incident title (e.g. 'Agent escaped sandbox')",
    )
    parser.add_argument(
        "--severity", "-s", default="medium",
        choices=SEVERITY_LEVELS,
        help="Incident severity (default: medium)",
    )
    parser.add_argument(
        "--description", "-d", default="",
        help="Detailed incident description",
    )
    parser.add_argument(
        "--timeline", "-t", nargs="*", default=[],
        help="Timeline entries (e.g. '10:00 Alert fired' '10:05 Investigated')",
    )
    parser.add_argument(
        "--impact", default="",
        help="Impact statement (auto-generated if omitted)",
    )
    parser.add_argument(
        "--detection", default="",
        help="How the incident was detected",
    )
    parser.add_argument(
        "--responders", "-r", nargs="*", default=[],
        help="Responder names/teams",
    )
    parser.add_argument(
        "--format", "-f", default="markdown",
        choices=("text", "markdown", "html", "json"),
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--output", "-o", default="",
        help="Write output to file (default: stdout)",
    )

    args = parser.parse_args(argv)

    pm = Postmortem(
        incident=args.incident,
        severity=args.severity,
        description=args.description,
        timeline=args.timeline,
        impact=args.impact,
        detection_method=args.detection,
        responders=args.responders or None,
    )

    renderers = {
        "text": pm.to_text,
        "markdown": pm.to_markdown,
        "html": pm.to_html,
        "json": pm.to_json,
    }
    content = renderers[args.format]()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Postmortem written to {args.output}")
    else:
        print(content)


if __name__ == "__main__":
    main()
