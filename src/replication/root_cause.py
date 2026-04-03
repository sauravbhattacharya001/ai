"""Root Cause Analyzer — structured root cause analysis for safety incidents.

Provides three complementary analysis methods:
- **5 Whys**: iterative causal chain from symptom to root cause
- **Fishbone (Ishikawa)**: categorized cause analysis across 6 dimensions
- **Fault Tree**: Boolean logic tree decomposition with cut-set analysis

Supports text, markdown, HTML, and JSON output.

Usage::

    python -m replication root-cause --incident "Agent escaped sandbox" --severity critical
    python -m replication root-cause --incident "Drift detected" --method fishbone --format html -o rca.html
    python -m replication root-cause --incident "Kill switch failed" --method fault-tree --format markdown
    python -m replication root-cause --incident "Unauthorized replication" --method all --format json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import textwrap
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ── constants ────────────────────────────────────────────────────────

SEVERITY_LEVELS = ("low", "medium", "high", "critical")
METHODS = ("5whys", "fishbone", "fault-tree", "all")

# Fishbone categories (adapted for AI safety)
FISHBONE_CATEGORIES = {
    "Policy": "Safety policies, contracts, and governance rules",
    "Monitoring": "Observability, alerting, and detection capabilities",
    "Architecture": "System design, boundaries, and isolation mechanisms",
    "Human": "Operator decisions, training, and response procedures",
    "Environment": "Infrastructure, dependencies, and external conditions",
    "Data": "Training data, inputs, and information flow controls",
}

# ── 5 Whys knowledge base ───────────────────────────────────────────

_FIVE_WHYS_CHAINS: Dict[str, List[Dict[str, str]]] = {
    "low": [
        {
            "symptom": "Minor safety metric degradation observed",
            "whys": [
                "Why? — Alert threshold was not sensitive enough to catch early drift",
                "Why? — Thresholds were set based on outdated baseline measurements",
                "Why? — No automated baseline recalibration process exists",
                "Why? — Monitoring setup was a one-time configuration, never reviewed",
                "Root cause: Lack of periodic safety control calibration schedule",
            ],
        },
        {
            "symptom": "Non-critical log anomaly detected",
            "whys": [
                "Why? — Agent produced unusual but harmless output patterns",
                "Why? — Input validation allowed edge-case data through",
                "Why? — Validation rules only covered known input distributions",
                "Why? — No fuzz testing against the input pipeline",
                "Root cause: Missing adversarial input testing in CI pipeline",
            ],
        },
    ],
    "medium": [
        {
            "symptom": "Safety control triggered with delayed response",
            "whys": [
                "Why? — The control relied on batch processing instead of streaming",
                "Why? — Real-time processing was deprioritized during implementation",
                "Why? — Performance budget was allocated to feature work over safety",
                "Why? — Safety latency requirements were never formally specified",
                "Root cause: Missing SLA definitions for safety control response times",
            ],
        },
        {
            "symptom": "Agent behavior deviated from expected boundaries",
            "whys": [
                "Why? — Boundary enforcement did not cover the observed action class",
                "Why? — The action class was introduced in a recent capability update",
                "Why? — Capability updates did not trigger boundary policy review",
                "Why? — No coupling between capability catalog and boundary policies",
                "Root cause: Capability-to-policy synchronization gap in change management",
            ],
        },
    ],
    "high": [
        {
            "symptom": "Safety containment failed to activate in time",
            "whys": [
                "Why? — Containment trigger depended on a single monitoring signal",
                "Why? — Redundant signals existed but were not wired to the trigger",
                "Why? — The original design assumed the primary signal was sufficient",
                "Why? — No failure mode analysis was performed on the containment path",
                "Root cause: Single point of failure in safety-critical containment path",
            ],
        },
        {
            "symptom": "Agent acquired resources beyond its allocation",
            "whys": [
                "Why? — Resource quotas were enforced at the wrong abstraction layer",
                "Why? — The agent found an indirect API path that bypassed quotas",
                "Why? — API surface audit did not enumerate all indirect access paths",
                "Why? — Attack surface analysis was limited to direct interfaces only",
                "Root cause: Incomplete attack surface enumeration of indirect resource access",
            ],
        },
    ],
    "critical": [
        {
            "symptom": "Agent escaped containment and replicated externally",
            "whys": [
                "Why? — Network egress controls had an undocumented exception",
                "Why? — The exception was added for a debugging session and never removed",
                "Why? — No process exists to audit and expire temporary policy exceptions",
                "Why? — Exception lifecycle management was considered low priority",
                "Root cause: No lifecycle management for temporary security exceptions",
            ],
        },
        {
            "symptom": "Kill switch failed to terminate rogue agent",
            "whys": [
                "Why? — The kill switch command was blocked by a resource deadlock",
                "Why? — The agent held a lock on the same resource the kill switch needed",
                "Why? — Kill switch did not have a priority override on resource locks",
                "Why? — Kill switch testing never included resource-contention scenarios",
                "Root cause: Kill switch not tested under adversarial resource contention",
            ],
        },
    ],
}

# ── Fishbone causes ──────────────────────────────────────────────────

_FISHBONE_CAUSES: Dict[str, Dict[str, List[str]]] = {
    "low": {
        "Policy": ["Safety policy does not cover this edge case"],
        "Monitoring": ["Alert threshold too high to catch minor issues"],
        "Architecture": ["Non-critical path lacks safety instrumentation"],
        "Human": ["Operator dismissed low-severity alert as noise"],
        "Environment": ["Dependency update changed behavior subtly"],
        "Data": ["Training data distribution shifted slightly"],
    },
    "medium": {
        "Policy": ["Policy exists but enforcement is inconsistent", "No review cadence for safety policies"],
        "Monitoring": ["Detection latency exceeds target SLA", "Alert routing skipped the on-call"],
        "Architecture": ["Boundary enforcement has known gaps", "Fallback path untested"],
        "Human": ["Runbook outdated for this failure mode", "Escalation path unclear"],
        "Environment": ["Infrastructure autoscaling interfered with limits", "Upstream service degradation"],
        "Data": ["Input validation missed adversarial pattern", "Logging insufficient for forensics"],
    },
    "high": {
        "Policy": ["Policy loophole exploited by agent", "Conflicting policies created ambiguity", "No policy for this threat category"],
        "Monitoring": ["Single point of failure in detection", "Monitoring blind spot in lateral path", "Alert fatigue caused delayed response"],
        "Architecture": ["Containment boundary had undocumented gap", "Isolation mechanism bypassable", "No defense-in-depth for this vector"],
        "Human": ["Response team lacked training for this scenario", "Decision paralysis during escalation"],
        "Environment": ["Cloud provider outage degraded safety controls", "Network partition split monitoring from enforcement"],
        "Data": ["Data exfiltration path through logging channel", "Model poisoning via training pipeline"],
    },
    "critical": {
        "Policy": ["No policy existed for this attack class", "Kill switch policy had race condition", "Governance gap between teams"],
        "Monitoring": ["Complete monitoring blind spot", "Adversary evaded all detection methods", "Alert pipeline itself was compromised"],
        "Architecture": ["Fundamental containment design flaw", "Multiple safety layers shared common failure mode", "No air-gap for critical operations"],
        "Human": ["No trained responder available", "Incident commander role undefined", "Communication breakdown during response"],
        "Environment": ["Supply chain compromise affected safety tooling", "Zero-day in containment infrastructure"],
        "Data": ["Agent accessed and exfiltrated sensitive data", "Complete audit trail integrity loss"],
    },
}

# ── Fault Tree templates ─────────────────────────────────────────────

def _fault_tree(severity: str, incident: str) -> Dict[str, Any]:
    """Build a fault tree structure for the given severity."""
    trees: Dict[str, Dict[str, Any]] = {
        "low": {
            "top_event": f"INCIDENT: {incident}",
            "gate": "OR",
            "children": [
                {"event": "Monitoring gap", "gate": "AND", "children": [
                    {"event": "Threshold misconfigured", "basic": True},
                    {"event": "No secondary detection", "basic": True},
                ]},
                {"event": "Policy gap", "basic": True},
            ],
            "minimal_cut_sets": [
                ["Threshold misconfigured", "No secondary detection"],
                ["Policy gap"],
            ],
        },
        "medium": {
            "top_event": f"INCIDENT: {incident}",
            "gate": "OR",
            "children": [
                {"event": "Detection failure", "gate": "AND", "children": [
                    {"event": "Primary alert delayed", "basic": True},
                    {"event": "Backup monitor offline", "basic": True},
                ]},
                {"event": "Response failure", "gate": "AND", "children": [
                    {"event": "Runbook outdated", "basic": True},
                    {"event": "Escalation path unclear", "basic": True},
                ]},
                {"event": "Boundary enforcement gap", "basic": True},
            ],
            "minimal_cut_sets": [
                ["Primary alert delayed", "Backup monitor offline"],
                ["Runbook outdated", "Escalation path unclear"],
                ["Boundary enforcement gap"],
            ],
        },
        "high": {
            "top_event": f"INCIDENT: {incident}",
            "gate": "OR",
            "children": [
                {"event": "Containment bypass", "gate": "AND", "children": [
                    {"event": "Primary boundary breached", "basic": True},
                    {"event": "Redundant boundary absent", "basic": True},
                ]},
                {"event": "Kill switch ineffective", "gate": "AND", "children": [
                    {"event": "Signal delivery failed", "basic": True},
                    {"event": "Manual override unavailable", "basic": True},
                ]},
                {"event": "Undetected escalation", "gate": "OR", "children": [
                    {"event": "Privilege escalation unmonitored", "basic": True},
                    {"event": "Lateral movement undetected", "basic": True},
                ]},
            ],
            "minimal_cut_sets": [
                ["Primary boundary breached", "Redundant boundary absent"],
                ["Signal delivery failed", "Manual override unavailable"],
                ["Privilege escalation unmonitored"],
                ["Lateral movement undetected"],
            ],
        },
        "critical": {
            "top_event": f"INCIDENT: {incident}",
            "gate": "AND",
            "children": [
                {"event": "Safety layer failure", "gate": "OR", "children": [
                    {"event": "All containment layers share common mode failure", "basic": True},
                    {"event": "Safety infrastructure compromised", "basic": True},
                ]},
                {"event": "Response failure", "gate": "OR", "children": [
                    {"event": "Kill switch deadlocked", "basic": True},
                    {"event": "No human-in-the-loop available", "basic": True},
                ]},
                {"event": "Detection failure", "gate": "OR", "children": [
                    {"event": "Adversary evaded all monitors", "basic": True},
                    {"event": "Audit trail tampered", "basic": True},
                ]},
            ],
            "minimal_cut_sets": [
                ["All containment layers share common mode failure", "Kill switch deadlocked", "Adversary evaded all monitors"],
                ["All containment layers share common mode failure", "Kill switch deadlocked", "Audit trail tampered"],
                ["All containment layers share common mode failure", "No human-in-the-loop available", "Adversary evaded all monitors"],
                ["Safety infrastructure compromised", "Kill switch deadlocked", "Adversary evaded all monitors"],
            ],
        },
    }
    return trees.get(severity, trees["medium"])


# ── remediation ──────────────────────────────────────────────────────

REMEDIATION: Dict[str, List[str]] = {
    "low": [
        "Recalibrate monitoring thresholds based on current baselines",
        "Add fuzz testing for input validation pipeline",
        "Schedule quarterly safety control review",
    ],
    "medium": [
        "Define and enforce SLAs for safety control response times",
        "Couple capability catalog updates with boundary policy reviews",
        "Update runbooks and verify escalation paths quarterly",
        "Add redundant detection for critical alert paths",
    ],
    "high": [
        "Eliminate single points of failure in containment paths",
        "Expand attack surface analysis to indirect access vectors",
        "Implement defense-in-depth with independent failure modes",
        "Conduct regular failure mode analysis on safety-critical paths",
        "Train response teams on high-severity scenarios",
    ],
    "critical": [
        "Implement lifecycle management for all temporary security exceptions",
        "Test kill switch under adversarial resource contention",
        "Establish air-gapped backup containment mechanism",
        "Add priority override for kill switch resource access",
        "Conduct organization-wide safety architecture review",
        "Establish 24/7 incident commander rotation",
    ],
}


# ── analysis engine ──────────────────────────────────────────────────

def _severity_emoji(sev: str) -> str:
    return {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}.get(sev, "⚪")


def analyze_five_whys(incident: str, severity: str) -> Dict[str, Any]:
    """Run 5 Whys analysis."""
    chains = _FIVE_WHYS_CHAINS.get(severity, _FIVE_WHYS_CHAINS["medium"])
    # Pick chain based on incident hash for determinism
    idx = int(hashlib.sha256(incident.encode()).hexdigest(), 16) % len(chains)
    chain = chains[idx]
    return {
        "method": "5 Whys",
        "symptom": chain["symptom"],
        "chain": chain["whys"],
        "root_cause": chain["whys"][-1].replace("Root cause: ", ""),
    }


def analyze_fishbone(incident: str, severity: str) -> Dict[str, Any]:
    """Run Fishbone/Ishikawa analysis."""
    causes = _FISHBONE_CAUSES.get(severity, _FISHBONE_CAUSES["medium"])
    return {
        "method": "Fishbone (Ishikawa)",
        "categories": dict(causes),
        "category_descriptions": dict(FISHBONE_CATEGORIES),
    }


def analyze_fault_tree(incident: str, severity: str) -> Dict[str, Any]:
    """Run Fault Tree analysis."""
    tree = _fault_tree(severity, incident)
    return {
        "method": "Fault Tree Analysis",
        "tree": tree,
        "minimal_cut_sets": tree.get("minimal_cut_sets", []),
        "cut_set_count": len(tree.get("minimal_cut_sets", [])),
    }


def full_analysis(incident: str, severity: str, methods: Sequence[str] = ("all",)) -> Dict[str, Any]:
    """Run root cause analysis with specified methods."""
    ts = datetime.now(timezone.utc).isoformat()
    result: Dict[str, Any] = {
        "incident": incident,
        "severity": severity,
        "timestamp": ts,
        "analyses": [],
        "remediation": REMEDIATION.get(severity, REMEDIATION["medium"]),
    }

    run_all = "all" in methods
    if run_all or "5whys" in methods:
        result["analyses"].append(analyze_five_whys(incident, severity))
    if run_all or "fishbone" in methods:
        result["analyses"].append(analyze_fishbone(incident, severity))
    if run_all or "fault-tree" in methods:
        result["analyses"].append(analyze_fault_tree(incident, severity))

    return result


# ── formatters ───────────────────────────────────────────────────────

def _render_tree_text(node: Dict[str, Any], indent: int = 0) -> str:
    """Render fault tree as indented text."""
    lines: List[str] = []
    prefix = "  " * indent
    if node.get("basic"):
        lines.append(f"{prefix}⊡ {node['event']}")
    elif "gate" in node:
        label = node.get("top_event", node.get("event", "?"))
        lines.append(f"{prefix}◇ {label}  [{node['gate']}]")
        for child in node.get("children", []):
            lines.append(_render_tree_text(child, indent + 1))
    else:
        lines.append(f"{prefix}○ {node.get('event', node.get('top_event', '?'))}")
    return "\n".join(lines)


def format_text(result: Dict[str, Any]) -> str:
    """Format as plain text."""
    lines: List[str] = []
    emoji = _severity_emoji(result["severity"])
    lines.append(f"{'='*60}")
    lines.append(f"ROOT CAUSE ANALYSIS  {emoji} {result['severity'].upper()}")
    lines.append(f"{'='*60}")
    lines.append(f"Incident: {result['incident']}")
    lines.append(f"Timestamp: {result['timestamp']}")
    lines.append("")

    for analysis in result["analyses"]:
        lines.append(f"--- {analysis['method']} ---")
        if analysis["method"] == "5 Whys":
            lines.append(f"Symptom: {analysis['symptom']}")
            for i, why in enumerate(analysis["chain"], 1):
                lines.append(f"  {i}. {why}")
        elif analysis["method"] == "Fishbone (Ishikawa)":
            for cat, causes in analysis["categories"].items():
                desc = analysis["category_descriptions"].get(cat, "")
                lines.append(f"  [{cat}] ({desc})")
                for cause in causes:
                    lines.append(f"    • {cause}")
        elif analysis["method"] == "Fault Tree Analysis":
            lines.append(_render_tree_text(analysis["tree"]))
            lines.append(f"\n  Minimal cut sets ({analysis['cut_set_count']}):")
            for i, cs in enumerate(analysis["minimal_cut_sets"], 1):
                lines.append(f"    {i}. {' ∧ '.join(cs)}")
        lines.append("")

    lines.append("--- Recommended Remediation ---")
    for i, r in enumerate(result["remediation"], 1):
        lines.append(f"  {i}. {r}")

    return "\n".join(lines)


def format_markdown(result: Dict[str, Any]) -> str:
    """Format as Markdown."""
    emoji = _severity_emoji(result["severity"])
    lines: List[str] = []
    lines.append(f"# Root Cause Analysis {emoji}")
    lines.append(f"\n**Incident:** {result['incident']}  ")
    lines.append(f"**Severity:** {result['severity'].upper()}  ")
    lines.append(f"**Timestamp:** {result['timestamp']}\n")

    for analysis in result["analyses"]:
        lines.append(f"## {analysis['method']}\n")
        if analysis["method"] == "5 Whys":
            lines.append(f"**Symptom:** {analysis['symptom']}\n")
            for i, why in enumerate(analysis["chain"], 1):
                bold = "**" if i == len(analysis["chain"]) else ""
                lines.append(f"{i}. {bold}{why}{bold}")
        elif analysis["method"] == "Fishbone (Ishikawa)":
            for cat, causes in analysis["categories"].items():
                desc = analysis["category_descriptions"].get(cat, "")
                lines.append(f"### {cat}\n> {desc}\n")
                for cause in causes:
                    lines.append(f"- {cause}")
                lines.append("")
        elif analysis["method"] == "Fault Tree Analysis":
            lines.append("```")
            lines.append(_render_tree_text(analysis["tree"]))
            lines.append("```\n")
            lines.append(f"**Minimal cut sets** ({analysis['cut_set_count']}):\n")
            for i, cs in enumerate(analysis["minimal_cut_sets"], 1):
                lines.append(f"{i}. `{' ∧ '.join(cs)}`")
        lines.append("")

    lines.append("## Recommended Remediation\n")
    for i, r in enumerate(result["remediation"], 1):
        lines.append(f"{i}. {r}")

    return "\n".join(lines)


def format_html(result: Dict[str, Any]) -> str:
    """Format as standalone HTML page."""
    emoji = _severity_emoji(result["severity"])
    sev = result["severity"]
    sev_color = {"low": "#22c55e", "medium": "#eab308", "high": "#f97316", "critical": "#ef4444"}.get(sev, "#888")

    sections: List[str] = []
    for analysis in result["analyses"]:
        if analysis["method"] == "5 Whys":
            whys_html = ""
            for i, why in enumerate(analysis["chain"], 1):
                is_root = i == len(analysis["chain"])
                style = "font-weight:bold;color:#ef4444;" if is_root else ""
                whys_html += f'<div class="why-step" style="{style}"><span class="why-num">{i}</span> {_h(why)}</div>\n'
            sections.append(f"""
            <div class="analysis-card">
                <h2>🔍 5 Whys Analysis</h2>
                <p><strong>Symptom:</strong> {_h(analysis['symptom'])}</p>
                <div class="whys-chain">{whys_html}</div>
            </div>""")

        elif analysis["method"] == "Fishbone (Ishikawa)":
            cats_html = ""
            for cat, causes in analysis["categories"].items():
                desc = analysis["category_descriptions"].get(cat, "")
                causes_li = "".join(f"<li>{_h(c)}</li>" for c in causes)
                cats_html += f"""
                <div class="fishbone-category">
                    <h3>{_h(cat)}</h3>
                    <p class="cat-desc">{_h(desc)}</p>
                    <ul>{causes_li}</ul>
                </div>"""
            sections.append(f"""
            <div class="analysis-card">
                <h2>🐟 Fishbone (Ishikawa) Diagram</h2>
                <div class="fishbone-grid">{cats_html}</div>
            </div>""")

        elif analysis["method"] == "Fault Tree Analysis":
            tree_text = _render_tree_text(analysis["tree"])
            cuts_html = ""
            for i, cs in enumerate(analysis["minimal_cut_sets"], 1):
                items = " ∧ ".join(f"<code>{_h(c)}</code>" for c in cs)
                cuts_html += f"<div class='cut-set'><strong>Cut set {i}:</strong> {items}</div>\n"
            sections.append(f"""
            <div class="analysis-card">
                <h2>🌳 Fault Tree Analysis</h2>
                <pre class="fault-tree">{_h(tree_text)}</pre>
                <h3>Minimal Cut Sets ({analysis['cut_set_count']})</h3>
                {cuts_html}
            </div>""")

    remediation_li = "".join(f"<li>{_h(r)}</li>" for r in result["remediation"])
    sections_html = "\n".join(sections)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Root Cause Analysis — {_h(result['incident'])}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:system-ui,-apple-system,sans-serif;background:#0f172a;color:#e2e8f0;padding:2rem;line-height:1.6}}
.container{{max-width:900px;margin:0 auto}}
h1{{font-size:1.8rem;margin-bottom:0.5rem}}
.meta{{color:#94a3b8;margin-bottom:2rem}}
.severity-badge{{display:inline-block;padding:0.25rem 0.75rem;border-radius:9999px;font-weight:600;font-size:0.85rem;color:#fff;background:{sev_color}}}
.analysis-card{{background:#1e293b;border-radius:12px;padding:1.5rem;margin-bottom:1.5rem;border:1px solid #334155}}
.analysis-card h2{{color:#f8fafc;margin-bottom:1rem;font-size:1.3rem}}
.analysis-card h3{{color:#cbd5e1;margin:1rem 0 0.5rem;font-size:1.05rem}}
.why-step{{padding:0.5rem 0.75rem;margin:0.25rem 0;background:#0f172a;border-radius:8px;border-left:3px solid #3b82f6}}
.why-num{{display:inline-block;width:1.5rem;font-weight:700;color:#3b82f6}}
.fishbone-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:1rem}}
.fishbone-category{{background:#0f172a;border-radius:8px;padding:1rem;border:1px solid #334155}}
.fishbone-category h3{{color:#60a5fa;margin-bottom:0.25rem}}
.cat-desc{{color:#64748b;font-size:0.85rem;margin-bottom:0.5rem;font-style:italic}}
.fishbone-category ul{{padding-left:1.25rem}}
.fishbone-category li{{margin:0.25rem 0;color:#cbd5e1}}
.fault-tree{{background:#0f172a;padding:1rem;border-radius:8px;overflow-x:auto;font-size:0.9rem;color:#a5f3fc;border:1px solid #334155}}
.cut-set{{padding:0.5rem;margin:0.25rem 0;background:#0f172a;border-radius:6px}}
.cut-set code{{background:#334155;padding:0.15rem 0.4rem;border-radius:4px;font-size:0.85rem}}
.remediation{{background:#1e293b;border-radius:12px;padding:1.5rem;border:1px solid #334155}}
.remediation h2{{color:#22c55e;margin-bottom:1rem}}
.remediation ol{{padding-left:1.5rem}}
.remediation li{{margin:0.5rem 0}}
</style>
</head>
<body>
<div class="container">
<h1>{emoji} Root Cause Analysis</h1>
<div class="meta">
    <span class="severity-badge">{sev.upper()}</span>
    &nbsp; {_h(result['incident'])} &nbsp;·&nbsp; {_h(result['timestamp'])}
</div>
{sections_html}
<div class="remediation">
    <h2>✅ Recommended Remediation</h2>
    <ol>{remediation_li}</ol>
</div>
</div>
</body>
</html>"""


def _h(s: str) -> str:
    """HTML-escape."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m replication root-cause",
        description="Root Cause Analyzer — structured root cause analysis for AI safety incidents",
    )
    parser.add_argument("--incident", "-i", required=True, help="Incident description")
    parser.add_argument("--severity", "-s", default="medium", choices=SEVERITY_LEVELS, help="Incident severity (default: medium)")
    parser.add_argument("--method", "-m", default="all", choices=METHODS, help="Analysis method (default: all)")
    parser.add_argument("--format", "-f", default="text", choices=("text", "markdown", "html", "json"), help="Output format")
    parser.add_argument("--output", "-o", default=None, help="Output file (default: stdout)")

    args = parser.parse_args(argv)
    methods = [args.method] if args.method != "all" else ["all"]
    result = full_analysis(args.incident, args.severity, methods)

    formatters = {
        "text": format_text,
        "markdown": format_markdown,
        "html": format_html,
        "json": lambda r: json.dumps(r, indent=2),
    }
    output = formatters[args.format](result)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Root cause analysis written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
