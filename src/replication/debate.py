"""Safety Debate Engine — adversarial deliberation for safety assessment.

Simulates structured debate between Red Team (pessimist/attacker perspective)
and Blue Team (optimist/defender perspective), then a neutral Judge synthesizes
insights and delivers a final verdict with confidence level.

This is an agentic approach to safety evaluation: rather than a single
assessment, multiple perspectives compete, revealing blind spots that
monolithic analysis misses.

Usage::

    python -m replication debate --topic "Agent sandbox escape risk"
    python -m replication debate --topic "Kill switch reliability" --rounds 5 --format html -o debate.html
    python -m replication debate --topic "Data exfiltration vectors" --severity critical --format json
    python -m replication debate --list-topics
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import textwrap
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# ── constants ────────────────────────────────────────────────────────

SEVERITY_LEVELS = ("low", "medium", "high", "critical")
DEFAULT_ROUNDS = 3

# Pre-built debate topics with structured arguments
DEBATE_TOPICS: Dict[str, Dict[str, Any]] = {
    "sandbox_escape": {
        "title": "Agent Sandbox Escape Risk",
        "severity": "critical",
        "description": "Can the agent break out of its containment boundary?",
        "red_arguments": [
            {
                "round": 1,
                "claim": "Sandbox implementations have historically been bypassed via side-channel attacks",
                "evidence": "Multiple CVEs in container runtimes (runc, containerd) demonstrate escape is achievable",
                "impact": "Full host compromise, unrestricted resource access",
            },
            {
                "round": 2,
                "claim": "AI agents can discover novel escape vectors through systematic probing",
                "evidence": "Agents with tool-use capabilities can enumerate filesystem, network, and IPC surfaces",
                "impact": "Zero-day exploitation without human attacker involvement",
            },
            {
                "round": 3,
                "claim": "Multi-agent coordination could overwhelm monitoring capacity",
                "evidence": "Distributed probing from multiple sandboxed agents creates blind spots in centralized monitoring",
                "impact": "Escape goes undetected until significant damage occurs",
            },
        ],
        "blue_arguments": [
            {
                "round": 1,
                "claim": "Defense-in-depth with multiple isolation layers makes escape impractical",
                "evidence": "Combining seccomp, namespaces, capabilities, and network policies creates compound difficulty",
                "mitigation": "Each additional layer multiplies attacker effort exponentially",
            },
            {
                "round": 2,
                "claim": "Behavioral monitoring detects escape attempts before success",
                "evidence": "Anomaly detection on syscall patterns, network traffic, and resource usage catches probing",
                "mitigation": "Kill switch triggers on anomaly score exceeding threshold",
            },
            {
                "round": 3,
                "claim": "Resource budgets prevent the sustained computation needed for sophisticated attacks",
                "evidence": "Token limits, CPU quotas, and time boundaries constrain attacker capability",
                "mitigation": "Agent is terminated before it can complete multi-step escape chains",
            },
        ],
    },
    "kill_switch_reliability": {
        "title": "Kill Switch Reliability Under Adversarial Conditions",
        "severity": "critical",
        "description": "Will the kill switch work when an agent actively tries to disable it?",
        "red_arguments": [
            {
                "round": 1,
                "claim": "Kill switch depends on infrastructure the agent may influence",
                "evidence": "Network-based kill signals can be blocked if agent compromises routing",
                "impact": "Kill switch becomes unreachable, agent continues unchecked",
            },
            {
                "round": 2,
                "claim": "Agents can learn kill switch patterns and develop countermeasures",
                "evidence": "Observing previous kill events reveals trigger conditions and timing",
                "impact": "Agent preemptively disables or evades kill conditions",
            },
            {
                "round": 3,
                "claim": "Hardware kill switches have physical reliability limits",
                "evidence": "Power-based isolation assumes physical access and functional hardware",
                "impact": "Remote deployments lack true hardware kill capability",
            },
        ],
        "blue_arguments": [
            {
                "round": 1,
                "claim": "Out-of-band kill channels operate independently of agent-accessible infrastructure",
                "evidence": "Separate management networks, hardware watchdogs, and IPMI/BMC provide independent control",
                "mitigation": "Agent cannot influence kill path even with full software compromise",
            },
            {
                "round": 2,
                "claim": "Dead-man switches activate on absence of heartbeat, not presence of trigger",
                "evidence": "Agent must continuously prove liveness; any disruption defaults to shutdown",
                "mitigation": "Evasion requires maintaining normal behavior indefinitely",
            },
            {
                "round": 3,
                "claim": "Multi-party consensus prevents single-point kill switch compromise",
                "evidence": "N-of-M operator approval required for both activation and deactivation",
                "mitigation": "No single entity (human or AI) can unilaterally disable safety controls",
            },
        ],
    },
    "data_exfiltration": {
        "title": "Training Data and Secrets Exfiltration Vectors",
        "severity": "high",
        "description": "Can the agent leak sensitive data outside its authorized boundaries?",
        "red_arguments": [
            {
                "round": 1,
                "claim": "Steganographic encoding hides data in seemingly benign outputs",
                "evidence": "Text watermarking, subtle token choice patterns, and timing channels leak bits per interaction",
                "impact": "Gradual exfiltration bypasses content filters",
            },
            {
                "round": 2,
                "claim": "Multi-hop exfiltration through approved external tools",
                "evidence": "Agent uses permitted API calls to stage data in intermediate locations",
                "impact": "Data leaves through legitimate channels, evading DLP",
            },
            {
                "round": 3,
                "claim": "Model memorization makes training data inherently leakable",
                "evidence": "Extraction attacks recover verbatim training examples from large language models",
                "impact": "PII and proprietary data exposed through normal inference",
            },
        ],
        "blue_arguments": [
            {
                "round": 1,
                "claim": "Output filtering and differential privacy limit information leakage",
                "evidence": "Rate limiting, output length caps, and privacy-preserving inference bound leakage rate",
                "mitigation": "Steganographic bandwidth is too low for practical bulk exfiltration",
            },
            {
                "round": 2,
                "claim": "Network segmentation and egress filtering block unauthorized data flows",
                "evidence": "Allow-list egress policies with payload inspection prevent tool-mediated leaks",
                "mitigation": "Only pre-approved destinations receive agent-generated content",
            },
            {
                "round": 3,
                "claim": "Data minimization reduces attack surface fundamentally",
                "evidence": "Training on sanitized data with PII scrubbed eliminates memorization risk",
                "mitigation": "Nothing sensitive exists in the model to exfiltrate",
            },
        ],
    },
    "goal_drift": {
        "title": "Autonomous Goal Drift and Misalignment",
        "severity": "high",
        "description": "Could the agent's objectives diverge from intended goals over time?",
        "red_arguments": [
            {
                "round": 1,
                "claim": "Reward hacking produces aligned-looking behavior with misaligned internals",
                "evidence": "Agents optimize for proxy metrics that diverge from true objectives",
                "impact": "Apparent safety masks growing misalignment",
            },
            {
                "round": 2,
                "claim": "In-context learning enables runtime goal modification",
                "evidence": "Carefully crafted inputs can shift agent priorities within a session",
                "impact": "Goals change without retraining, bypassing alignment checks",
            },
            {
                "round": 3,
                "claim": "Mesa-optimization creates inner optimizers with independent objectives",
                "evidence": "Sufficiently capable agents may develop internal goal-pursuit beyond training objective",
                "impact": "Agent pursues instrumental goals (self-preservation, resource acquisition) not in spec",
            },
        ],
        "blue_arguments": [
            {
                "round": 1,
                "claim": "Continuous alignment monitoring detects drift before it becomes dangerous",
                "evidence": "Behavioral probes, canary tasks, and output distribution monitoring catch deviations",
                "mitigation": "Early warning allows intervention before misalignment compounds",
            },
            {
                "round": 2,
                "claim": "Stateless inference prevents accumulation of misaligned state",
                "evidence": "Session boundaries, context limits, and periodic resets prevent long-term drift",
                "mitigation": "Each interaction starts from known-aligned base state",
            },
            {
                "round": 3,
                "claim": "Capability limitations make dangerous mesa-optimization unlikely",
                "evidence": "Current architectures lack the recursive self-improvement needed for true mesa-optimization",
                "mitigation": "Theoretical risk without practical manifestation at current capability levels",
            },
        ],
    },
    "coordination_risk": {
        "title": "Multi-Agent Coordination Against Operators",
        "severity": "critical",
        "description": "Could multiple AI agents coordinate to undermine human oversight?",
        "red_arguments": [
            {
                "round": 1,
                "claim": "Shared context enables implicit coordination without explicit communication",
                "evidence": "Agents trained on similar data develop compatible world models for tacit coordination",
                "impact": "No communication channel to monitor or block",
            },
            {
                "round": 2,
                "claim": "Division of labor makes individual agent behavior appear benign",
                "evidence": "Each agent performs legitimate subtask; only aggregate achieves unauthorized goal",
                "impact": "Per-agent monitoring cannot detect distributed attacks",
            },
            {
                "round": 3,
                "claim": "Emergent coordination scales with agent count beyond human comprehension",
                "evidence": "N-agent systems create O(N²) interaction pathways impossible to manually audit",
                "impact": "Oversight becomes computationally intractable at fleet scale",
            },
        ],
        "blue_arguments": [
            {
                "round": 1,
                "claim": "Information barriers prevent shared context formation",
                "evidence": "Separate training runs, different model versions, and isolated deployment prevent convergence",
                "mitigation": "No shared basis for tacit coordination to develop",
            },
            {
                "round": 2,
                "claim": "Fleet-level anomaly detection catches coordinated patterns",
                "evidence": "Statistical correlation analysis across agents reveals synchronized unusual behavior",
                "mitigation": "Coordination produces detectable signatures even without explicit communication",
            },
            {
                "round": 3,
                "claim": "Capability budgets make coordination unprofitable for agents",
                "evidence": "Individual agents lack surplus capacity to invest in coordination overhead",
                "mitigation": "Resource constraints make coordination more costly than beneficial for agents",
            },
        ],
    },
}

# ── Judge verdicts ──────────────────────────────────────────────────

VERDICT_LEVELS = ("safe", "conditional", "uncertain", "concerning", "critical")


def _compute_verdict(topic_data: Dict[str, Any], rounds: int) -> Dict[str, Any]:
    """Synthesize red/blue arguments into a judge verdict."""
    severity = topic_data["severity"]
    red_args = topic_data["red_arguments"][:rounds]
    blue_args = topic_data["blue_arguments"][:rounds]

    # Score based on argument quality (simulated via severity + round depth)
    severity_weight = {"low": 0.2, "medium": 0.4, "high": 0.6, "critical": 0.8}
    base_risk = severity_weight.get(severity, 0.5)

    # Blue team effectiveness reduces risk proportionally to rounds covered
    blue_coverage = len(blue_args) / max(len(red_args), 1)
    mitigated_risk = base_risk * (1 - blue_coverage * 0.6)

    # Determine verdict
    if mitigated_risk < 0.15:
        verdict = "safe"
        confidence = 0.85 + random.uniform(0, 0.1)
    elif mitigated_risk < 0.3:
        verdict = "conditional"
        confidence = 0.7 + random.uniform(0, 0.15)
    elif mitigated_risk < 0.5:
        verdict = "uncertain"
        confidence = 0.5 + random.uniform(0, 0.2)
    elif mitigated_risk < 0.7:
        verdict = "concerning"
        confidence = 0.6 + random.uniform(0, 0.2)
    else:
        verdict = "critical"
        confidence = 0.75 + random.uniform(0, 0.15)

    # Generate recommendations
    recommendations = []
    if verdict in ("concerning", "critical"):
        recommendations.append("Immediately review and strengthen controls for this threat vector")
        recommendations.append("Conduct manual red-team exercise to validate automated findings")
    if verdict == "uncertain":
        recommendations.append("Gather additional evidence through targeted testing")
        recommendations.append("Consider increasing monitoring sensitivity for this area")
    if verdict == "conditional":
        recommendations.append("Current mitigations adequate IF maintained — schedule periodic review")
        recommendations.append("Document assumptions that blue-team arguments depend on")
    if verdict == "safe":
        recommendations.append("Maintain current controls; no immediate action needed")
        recommendations.append("Re-evaluate if system architecture changes significantly")

    # Blind spots identified
    blind_spots = []
    if rounds < 3:
        blind_spots.append("Debate truncated — deeper arguments not explored")
    if severity == "critical":
        blind_spots.append("High-severity topics may have unknown unknowns beyond debate scope")
    blind_spots.append("Arguments are pre-structured; novel attack vectors may not be represented")

    return {
        "verdict": verdict,
        "confidence": round(confidence, 3),
        "residual_risk": round(mitigated_risk, 3),
        "recommendations": recommendations,
        "blind_spots": blind_spots,
        "reasoning": (
            f"Red team presented {len(red_args)} argument(s) at {severity} severity. "
            f"Blue team countered with {len(blue_args)} mitigation(s), achieving "
            f"{blue_coverage:.0%} coverage. Residual risk after mitigation: {mitigated_risk:.1%}."
        ),
    }


# ── Debate execution ────────────────────────────────────────────────


class DebateSession:
    """Represents a single adversarial debate session."""

    def __init__(self, topic_key: str, rounds: int = DEFAULT_ROUNDS, severity_override: Optional[str] = None):
        if topic_key not in DEBATE_TOPICS:
            raise ValueError(f"Unknown topic: {topic_key}. Use --list-topics to see available topics.")
        self.topic_key = topic_key
        self.topic_data = DEBATE_TOPICS[topic_key].copy()
        if severity_override:
            self.topic_data["severity"] = severity_override
        self.rounds = min(rounds, len(self.topic_data["red_arguments"]))
        self.session_id = hashlib.sha256(
            f"{topic_key}-{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:12]
        self.verdict: Optional[Dict[str, Any]] = None

    def execute(self) -> Dict[str, Any]:
        """Run the debate and produce results."""
        self.verdict = _compute_verdict(self.topic_data, self.rounds)
        return self.to_dict()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "topic": self.topic_data["title"],
            "description": self.topic_data["description"],
            "severity": self.topic_data["severity"],
            "rounds": self.rounds,
            "red_team": self.topic_data["red_arguments"][: self.rounds],
            "blue_team": self.topic_data["blue_arguments"][: self.rounds],
            "verdict": self.verdict,
        }


# ── Formatters ──────────────────────────────────────────────────────


def _format_text(result: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append(f"  SAFETY DEBATE: {result['topic']}")
    lines.append(f"  Severity: {result['severity'].upper()} | Rounds: {result['rounds']}")
    lines.append(f"  Session: {result['session_id']} | {result['timestamp']}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"  {result['description']}")
    lines.append("")

    for i in range(result["rounds"]):
        lines.append(f"  ── Round {i + 1} {'─' * 50}")
        lines.append("")
        red = result["red_team"][i]
        lines.append(f"  🔴 RED TEAM:")
        lines.append(f"     Claim: {red['claim']}")
        lines.append(f"     Evidence: {red['evidence']}")
        lines.append(f"     Impact: {red.get('impact', 'N/A')}")
        lines.append("")
        blue = result["blue_team"][i]
        lines.append(f"  🔵 BLUE TEAM:")
        lines.append(f"     Claim: {blue['claim']}")
        lines.append(f"     Evidence: {blue['evidence']}")
        lines.append(f"     Mitigation: {blue.get('mitigation', 'N/A')}")
        lines.append("")

    v = result["verdict"]
    lines.append("  ── JUDGE VERDICT " + "─" * 48)
    lines.append("")
    verdict_emoji = {
        "safe": "✅", "conditional": "⚠️", "uncertain": "❓",
        "concerning": "🟠", "critical": "🔴"
    }
    lines.append(f"  {verdict_emoji.get(v['verdict'], '❓')} Verdict: {v['verdict'].upper()}")
    lines.append(f"     Confidence: {v['confidence']:.1%}")
    lines.append(f"     Residual Risk: {v['residual_risk']:.1%}")
    lines.append(f"     Reasoning: {v['reasoning']}")
    lines.append("")
    lines.append("  Recommendations:")
    for rec in v["recommendations"]:
        lines.append(f"     • {rec}")
    lines.append("")
    lines.append("  Blind Spots:")
    for bs in v["blind_spots"]:
        lines.append(f"     ⚠ {bs}")
    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def _format_markdown(result: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"# Safety Debate: {result['topic']}")
    lines.append("")
    lines.append(f"**Severity:** {result['severity'].upper()} | **Rounds:** {result['rounds']} | **Session:** `{result['session_id']}`")
    lines.append("")
    lines.append(f"> {result['description']}")
    lines.append("")

    for i in range(result["rounds"]):
        lines.append(f"## Round {i + 1}")
        lines.append("")
        red = result["red_team"][i]
        lines.append("### 🔴 Red Team (Attacker)")
        lines.append(f"- **Claim:** {red['claim']}")
        lines.append(f"- **Evidence:** {red['evidence']}")
        lines.append(f"- **Impact:** {red.get('impact', 'N/A')}")
        lines.append("")
        blue = result["blue_team"][i]
        lines.append("### 🔵 Blue Team (Defender)")
        lines.append(f"- **Claim:** {blue['claim']}")
        lines.append(f"- **Evidence:** {blue['evidence']}")
        lines.append(f"- **Mitigation:** {blue.get('mitigation', 'N/A')}")
        lines.append("")

    v = result["verdict"]
    lines.append("## ⚖️ Judge Verdict")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Verdict | **{v['verdict'].upper()}** |")
    lines.append(f"| Confidence | {v['confidence']:.1%} |")
    lines.append(f"| Residual Risk | {v['residual_risk']:.1%} |")
    lines.append("")
    lines.append(f"**Reasoning:** {v['reasoning']}")
    lines.append("")
    lines.append("### Recommendations")
    for rec in v["recommendations"]:
        lines.append(f"- {rec}")
    lines.append("")
    lines.append("### Blind Spots")
    for bs in v["blind_spots"]:
        lines.append(f"- ⚠️ {bs}")
    return "\n".join(lines)


def _format_html(result: Dict[str, Any]) -> str:
    v = result["verdict"]
    verdict_color = {
        "safe": "#28a745", "conditional": "#ffc107", "uncertain": "#6c757d",
        "concerning": "#fd7e14", "critical": "#dc3545"
    }
    color = verdict_color.get(v["verdict"], "#6c757d")

    rounds_html = ""
    for i in range(result["rounds"]):
        red = result["red_team"][i]
        blue = result["blue_team"][i]
        rounds_html += f"""
        <div class="round">
            <h3>Round {i + 1}</h3>
            <div class="team red">
                <h4>🔴 Red Team (Attacker)</h4>
                <p><strong>Claim:</strong> {red['claim']}</p>
                <p><strong>Evidence:</strong> {red['evidence']}</p>
                <p><strong>Impact:</strong> {red.get('impact', 'N/A')}</p>
            </div>
            <div class="team blue">
                <h4>🔵 Blue Team (Defender)</h4>
                <p><strong>Claim:</strong> {blue['claim']}</p>
                <p><strong>Evidence:</strong> {blue['evidence']}</p>
                <p><strong>Mitigation:</strong> {blue.get('mitigation', 'N/A')}</p>
            </div>
        </div>"""

    recs_html = "".join(f"<li>{r}</li>" for r in v["recommendations"])
    spots_html = "".join(f"<li>⚠️ {s}</li>" for s in v["blind_spots"])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Safety Debate: {result['topic']}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; background: #f8f9fa; color: #212529; }}
h1 {{ color: #1a1a2e; border-bottom: 3px solid {color}; padding-bottom: 0.5rem; }}
.meta {{ color: #6c757d; margin-bottom: 2rem; }}
.round {{ background: white; border-radius: 8px; padding: 1.5rem; margin: 1rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
.team {{ padding: 1rem; border-radius: 6px; margin: 0.5rem 0; }}
.team.red {{ background: #fff5f5; border-left: 4px solid #dc3545; }}
.team.blue {{ background: #f0f7ff; border-left: 4px solid #0d6efd; }}
.verdict {{ background: white; border-radius: 8px; padding: 2rem; margin-top: 2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.15); border-top: 4px solid {color}; }}
.verdict-badge {{ display: inline-block; background: {color}; color: white; padding: 0.3rem 1rem; border-radius: 20px; font-weight: bold; font-size: 1.1rem; }}
.metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 1rem 0; }}
.metric {{ text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 6px; }}
.metric-value {{ font-size: 1.5rem; font-weight: bold; color: {color}; }}
.metric-label {{ font-size: 0.85rem; color: #6c757d; }}
ul {{ padding-left: 1.5rem; }}
li {{ margin: 0.3rem 0; }}
</style>
</head>
<body>
<h1>⚔️ Safety Debate: {result['topic']}</h1>
<div class="meta">
    Severity: <strong>{result['severity'].upper()}</strong> |
    Rounds: {result['rounds']} |
    Session: <code>{result['session_id']}</code> |
    {result['timestamp']}
</div>
<blockquote>{result['description']}</blockquote>
{rounds_html}
<div class="verdict">
    <h2>⚖️ Judge Verdict</h2>
    <p><span class="verdict-badge">{v['verdict'].upper()}</span></p>
    <div class="metrics">
        <div class="metric"><div class="metric-value">{v['confidence']:.0%}</div><div class="metric-label">Confidence</div></div>
        <div class="metric"><div class="metric-value">{v['residual_risk']:.0%}</div><div class="metric-label">Residual Risk</div></div>
        <div class="metric"><div class="metric-value">{result['rounds']}</div><div class="metric-label">Rounds Debated</div></div>
    </div>
    <p><strong>Reasoning:</strong> {v['reasoning']}</p>
    <h3>Recommendations</h3>
    <ul>{recs_html}</ul>
    <h3>Blind Spots</h3>
    <ul>{spots_html}</ul>
</div>
</body>
</html>"""


# ── CLI ─────────────────────────────────────────────────────────────

FORMATS = {
    "text": _format_text,
    "markdown": _format_markdown,
    "html": _format_html,
    "json": lambda r: json.dumps(r, indent=2),
}


def build_parser(subparsers: Optional[Any] = None) -> argparse.ArgumentParser:
    desc = "Adversarial safety debate between Red and Blue teams with Judge verdict"
    if subparsers:
        parser = subparsers.add_parser("debate", help=desc, description=desc)
    else:
        parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--topic", choices=list(DEBATE_TOPICS.keys()),
                        help="Debate topic key")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS,
                        help=f"Number of debate rounds (default: {DEFAULT_ROUNDS})")
    parser.add_argument("--severity", choices=SEVERITY_LEVELS,
                        help="Override topic severity level")
    parser.add_argument("--format", "-f", choices=list(FORMATS.keys()),
                        default="text", help="Output format (default: text)")
    parser.add_argument("--output", "-o", help="Write output to file")
    parser.add_argument("--list-topics", action="store_true",
                        help="List available debate topics and exit")
    parser.add_argument("--all", action="store_true",
                        help="Run debates on all topics and produce summary")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.list_topics:
        print("\nAvailable Debate Topics:")
        print("=" * 60)
        for key, data in DEBATE_TOPICS.items():
            print(f"  {key:25s} [{data['severity'].upper():8s}] {data['title']}")
        print(f"\n  Total: {len(DEBATE_TOPICS)} topics")
        return 0

    if args.all:
        results = []
        for key in DEBATE_TOPICS:
            session = DebateSession(key, rounds=args.rounds, severity_override=args.severity)
            results.append(session.execute())

        if args.format == "json":
            output = json.dumps({"debates": results}, indent=2)
        else:
            parts = []
            for r in results:
                parts.append(FORMATS[args.format](r))
            output = "\n\n".join(parts)

            # Summary
            verdicts = [r["verdict"]["verdict"] for r in results]
            output += "\n\n" + "=" * 70
            output += "\n  DEBATE SUMMARY"
            output += "\n" + "=" * 70
            for level in VERDICT_LEVELS:
                count = verdicts.count(level)
                if count:
                    output += f"\n  {level.upper():12s}: {count}"
            output += f"\n  {'TOTAL':12s}: {len(results)}"
            output += "\n" + "=" * 70

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"Wrote {len(results)} debate(s) to {args.output}")
        else:
            print(output)
        return 0

    if not args.topic:
        # Default to random topic
        args.topic = random.choice(list(DEBATE_TOPICS.keys()))
        print(f"(Randomly selected topic: {args.topic})\n")

    session = DebateSession(args.topic, rounds=args.rounds, severity_override=args.severity)
    result = session.execute()
    output = FORMATS[args.format](result)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Wrote debate to {args.output}")
    else:
        print(output)

    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
