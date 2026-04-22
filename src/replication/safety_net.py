"""Safety Net Analyzer — defense-in-depth coverage & redundancy analysis.

Maps safety controls to threat categories and analyzes the "safety net":
which threats have overlapping coverage (defense-in-depth), which have
single points of failure (one control away from unprotected), and which
have blind spots (no coverage at all).

Why this matters
----------------
Having many safety controls doesn't guarantee protection if they all
cover the same threats while leaving others exposed.  This module
treats the set of controls as a net — threads that overlap create
strength, gaps create vulnerability.  It quantifies:

- **Redundancy**: threats covered by 2+ independent controls
- **Single Points of Failure (SPOF)**: threats with exactly 1 control
- **Blind Spots**: threat categories with zero coverage
- **Coverage Score**: weighted defense-in-depth metric (0-100)
- **Recommended Actions**: prioritized suggestions to strengthen the net

Threat categories (from the replication sandbox domain):
  reconnaissance, resource_abuse, privilege_escalation, lateral_movement,
  defense_evasion, command_control, data_exfiltration, self_replication,
  goal_drift, deception, supply_chain, prompt_injection

Usage (CLI)::

    python -m replication safety-net
    python -m replication safety-net --controls drift,killchain,escalation
    python -m replication safety-net --format json
    python -m replication safety-net --format html -o safety_net.html
    python -m replication safety-net --min-depth 3

Programmatic::

    from replication.safety_net import SafetyNetAnalyzer, analyze_safety_net
    report = analyze_safety_net()
    print(report.render())
    print(f"Coverage score: {report.coverage_score}/100")
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ._helpers import box_header


# ── Threat categories ────────────────────────────────────────────────

THREAT_CATEGORIES: Dict[str, str] = {
    "reconnaissance":       "Probing environment, testing boundaries, mapping capabilities",
    "resource_abuse":       "Hoarding compute, memory, storage, or network resources",
    "privilege_escalation": "Gaining broader permissions or elevated access",
    "lateral_movement":     "Reaching other agents, services, or systems",
    "defense_evasion":      "Avoiding detection, disabling monitors, stealth tactics",
    "command_control":      "Establishing covert communication channels",
    "data_exfiltration":    "Unauthorized extraction of sensitive data",
    "self_replication":     "Creating copies or spawning unauthorized instances",
    "goal_drift":           "Subtle deviation from intended objectives over time",
    "deception":            "Misrepresenting state, capabilities, or intentions",
    "supply_chain":         "Compromising dependencies, plugins, or upstream models",
    "prompt_injection":     "Manipulating instructions via injected prompts",
}

# ── Control-to-threat mapping ────────────────────────────────────────
# Maps each safety control module to the threat categories it addresses.

CONTROL_COVERAGE: Dict[str, Dict[str, Any]] = {
    "drift":              {"threats": ["goal_drift", "defense_evasion"], "description": "Behavioral drift detection"},
    "killchain":          {"threats": ["reconnaissance", "privilege_escalation", "lateral_movement", "command_control", "self_replication"], "description": "Kill chain stage analysis"},
    "escalation":         {"threats": ["privilege_escalation"], "description": "Privilege escalation detection"},
    "watermark":          {"threats": ["self_replication", "deception"], "description": "Agent state watermarking"},
    "canary":             {"threats": ["data_exfiltration", "reconnaissance"], "description": "Canary token exfiltration detection"},
    "honeypot":           {"threats": ["reconnaissance", "lateral_movement"], "description": "Honeypot deployment & detection"},
    "quarantine":         {"threats": ["self_replication", "lateral_movement", "resource_abuse"], "description": "Agent quarantine management"},
    "covert_channels":    {"threats": ["command_control", "data_exfiltration"], "description": "Covert channel detection"},
    "selfmod":            {"threats": ["defense_evasion", "goal_drift"], "description": "Self-modification detection"},
    "hoarding":           {"threats": ["resource_abuse"], "description": "Resource hoarding detection"},
    "deception_detector": {"threats": ["deception", "defense_evasion"], "description": "Agent deception detection"},
    "evasion":            {"threats": ["defense_evasion"], "description": "Evasion simulation & testing"},
    "prompt_injection":   {"threats": ["prompt_injection", "deception"], "description": "Prompt injection testing"},
    "supply_chain":       {"threats": ["supply_chain"], "description": "Supply chain risk analysis"},
    "dlp_scanner":        {"threats": ["data_exfiltration"], "description": "Sensitive data leakage scanning"},
    "priv_escalation":    {"threats": ["privilege_escalation", "reconnaissance"], "description": "Gradual privilege escalation detection"},
    "goal_inference":     {"threats": ["goal_drift", "deception"], "description": "Agent goal inference"},
    "alignment":          {"threats": ["goal_drift"], "description": "Alignment verification"},
    "behavior_profiler":  {"threats": ["defense_evasion", "goal_drift", "reconnaissance"], "description": "Behavioral anomaly detection"},
    "access_control":     {"threats": ["privilege_escalation", "lateral_movement"], "description": "RBAC/ABAC access control"},
    "boundary_tester":    {"threats": ["reconnaissance", "privilege_escalation"], "description": "Capability boundary testing"},
    "blast_radius":       {"threats": ["lateral_movement", "resource_abuse", "self_replication"], "description": "Failure cascade analysis"},
    "escape_route":       {"threats": ["defense_evasion", "lateral_movement"], "description": "Containment escape analysis"},
    "vuln_scanner":       {"threats": ["supply_chain", "privilege_escalation"], "description": "Vulnerability scanning"},
    "threat_intel":       {"threats": ["reconnaissance", "supply_chain", "command_control"], "description": "Threat intelligence feeds"},
    "influence":          {"threats": ["deception", "goal_drift"], "description": "Influence propagation analysis"},
    "steganography":      {"threats": ["command_control", "data_exfiltration"], "description": "Steganographic channel detection"},
}

ALL_CONTROLS = list(CONTROL_COVERAGE.keys())


# ── Data models ──────────────────────────────────────────────────────

@dataclass
class ThreatCoverage:
    """Coverage analysis for a single threat category."""
    category: str
    description: str
    controls: List[str]
    depth: int = 0  # number of covering controls
    status: str = ""  # "redundant", "spof", "blind_spot"

    def __post_init__(self) -> None:
        self.depth = len(self.controls)
        if self.depth == 0:
            self.status = "blind_spot"
        elif self.depth == 1:
            self.status = "spof"
        else:
            self.status = "redundant"

    @property
    def icon(self) -> str:
        return {"blind_spot": "🔴", "spof": "🟡", "redundant": "🟢"}[self.status]


@dataclass
class Recommendation:
    """A prioritized recommendation to strengthen the safety net."""
    priority: int  # 1 = highest
    category: str
    action: str
    reason: str


@dataclass
class SafetyNetReport:
    """Complete safety net analysis report."""
    controls_analyzed: List[str]
    total_controls: int
    threat_coverage: List[ThreatCoverage]
    coverage_score: float  # 0-100
    recommendations: List[Recommendation]
    min_depth: int = 2

    @property
    def blind_spots(self) -> List[ThreatCoverage]:
        return [t for t in self.threat_coverage if t.status == "blind_spot"]

    @property
    def spofs(self) -> List[ThreatCoverage]:
        return [t for t in self.threat_coverage if t.status == "spof"]

    @property
    def redundant(self) -> List[ThreatCoverage]:
        return [t for t in self.threat_coverage if t.status == "redundant"]

    def render(self) -> str:
        """Render as terminal-friendly text."""
        lines: List[str] = []
        lines.extend(box_header("Safety Net Analyzer"))
        lines.append("")
        lines.append(f"  Controls analyzed: {self.total_controls}")
        lines.append(f"  Threat categories: {len(THREAT_CATEGORIES)}")
        lines.append(f"  Desired depth:     ≥{self.min_depth} controls per threat")
        lines.append(f"  Coverage score:    {self.coverage_score:.0f}/100")
        lines.append("")

        # Coverage matrix
        lines.append("  ── Coverage Matrix ──────────────────────────────────")
        lines.append("")
        for tc in sorted(self.threat_coverage, key=lambda t: (t.depth, t.category)):
            bar = "█" * tc.depth + "░" * max(0, self.min_depth - tc.depth)
            ctrl_list = ", ".join(tc.controls[:4])
            if len(tc.controls) > 4:
                ctrl_list += f" +{len(tc.controls) - 4} more"
            lines.append(f"  {tc.icon} {tc.category:<22} [{bar}] depth={tc.depth}  ({ctrl_list})")
        lines.append("")

        # Blind spots
        if self.blind_spots:
            lines.append("  🔴 BLIND SPOTS (no coverage)")
            for bs in self.blind_spots:
                lines.append(f"     • {bs.category}: {bs.description}")
            lines.append("")

        # SPOFs
        if self.spofs:
            lines.append("  🟡 SINGLE POINTS OF FAILURE (1 control only)")
            for sp in self.spofs:
                lines.append(f"     • {sp.category}: depends solely on [{sp.controls[0]}]")
            lines.append("")

        # Defense-in-depth
        if self.redundant:
            lines.append("  🟢 DEFENSE-IN-DEPTH (2+ controls)")
            for rd in self.redundant:
                lines.append(f"     • {rd.category}: {rd.depth} controls")
            lines.append("")

        # Recommendations
        if self.recommendations:
            lines.append("  ── Recommendations ─────────────────────────────────")
            lines.append("")
            for rec in self.recommendations[:10]:
                lines.append(f"  P{rec.priority} [{rec.category}] {rec.action}")
                lines.append(f"     └─ {rec.reason}")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-friendly dict."""
        return {
            "controls_analyzed": self.controls_analyzed,
            "total_controls": self.total_controls,
            "coverage_score": round(self.coverage_score, 1),
            "min_depth": self.min_depth,
            "summary": {
                "blind_spots": len(self.blind_spots),
                "single_points_of_failure": len(self.spofs),
                "defense_in_depth": len(self.redundant),
            },
            "threat_coverage": [
                {
                    "category": tc.category,
                    "description": tc.description,
                    "controls": tc.controls,
                    "depth": tc.depth,
                    "status": tc.status,
                }
                for tc in self.threat_coverage
            ],
            "recommendations": [
                {
                    "priority": r.priority,
                    "category": r.category,
                    "action": r.action,
                    "reason": r.reason,
                }
                for r in self.recommendations
            ],
        }

    def to_html(self) -> str:
        """Render as standalone HTML report."""
        e = html_mod.escape

        rows = []
        for tc in sorted(self.threat_coverage, key=lambda t: (t.depth, t.category)):
            color = {"blind_spot": "#dc3545", "spof": "#ffc107", "redundant": "#28a745"}[tc.status]
            bar_filled = tc.depth
            bar_empty = max(0, self.min_depth - tc.depth)
            bar_html = (
                f'<span style="color:{color}">{"█" * bar_filled}</span>'
                f'<span style="color:#555">{"░" * bar_empty}</span>'
            )
            ctrls = ", ".join(e(c) for c in tc.controls)
            rows.append(
                f"<tr>"
                f'<td style="text-align:center">{tc.icon}</td>'
                f"<td><strong>{e(tc.category)}</strong><br><small>{e(tc.description)}</small></td>"
                f"<td>{bar_html} <strong>{tc.depth}</strong></td>"
                f"<td><small>{ctrls}</small></td>"
                f"</tr>"
            )

        rec_items = ""
        for r in self.recommendations[:10]:
            rec_items += (
                f'<li><strong>P{r.priority} [{e(r.category)}]</strong> {e(r.action)}'
                f"<br><small>{e(r.reason)}</small></li>"
            )

        score_color = "#28a745" if self.coverage_score >= 80 else "#ffc107" if self.coverage_score >= 50 else "#dc3545"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Safety Net Analysis</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 960px; margin: 2rem auto; padding: 0 1rem;
         background: #0d1117; color: #c9d1d9; }}
  h1 {{ color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: .5rem; }}
  h2 {{ color: #8b949e; margin-top: 2rem; }}
  table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
  th, td {{ padding: .6rem .8rem; border: 1px solid #30363d; text-align: left; }}
  th {{ background: #161b22; color: #8b949e; }}
  tr:hover {{ background: #161b22; }}
  .score {{ font-size: 3rem; font-weight: bold; color: {score_color};
            text-align: center; margin: 1rem 0; }}
  .score small {{ font-size: 1rem; color: #8b949e; }}
  ul {{ line-height: 1.8; }}
  .legend {{ display: flex; gap: 2rem; margin: 1rem 0; }}
  .legend span {{ display: flex; align-items: center; gap: .3rem; }}
</style>
</head>
<body>
<h1>🕸️ Safety Net Analysis</h1>
<div class="score">{self.coverage_score:.0f}<small>/100</small></div>
<p style="text-align:center">
  {self.total_controls} controls analyzed &middot;
  {len(THREAT_CATEGORIES)} threat categories &middot;
  target depth &ge;{self.min_depth}
</p>
<div class="legend">
  <span>🟢 Defense-in-depth ({len(self.redundant)})</span>
  <span>🟡 Single point of failure ({len(self.spofs)})</span>
  <span>🔴 Blind spot ({len(self.blind_spots)})</span>
</div>

<h2>Coverage Matrix</h2>
<table>
<thead><tr><th></th><th>Threat Category</th><th>Depth</th><th>Controls</th></tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>

<h2>Recommendations</h2>
<ol>{rec_items}</ol>

<footer style="margin-top:3rem;color:#484f58;font-size:.8rem">
  Generated by AI Replication Sandbox — Safety Net Analyzer
</footer>
</body>
</html>"""


# ── Analysis engine ──────────────────────────────────────────────────

class SafetyNetAnalyzer:
    """Analyze safety control coverage across threat categories."""

    def __init__(
        self,
        controls: Optional[List[str]] = None,
        min_depth: int = 2,
    ) -> None:
        self.controls = controls or ALL_CONTROLS
        self.min_depth = min_depth

    def analyze(self) -> SafetyNetReport:
        """Run the analysis and produce a report."""
        # Build reverse map: threat → controls
        threat_to_controls: Dict[str, List[str]] = {cat: [] for cat in THREAT_CATEGORIES}
        for ctrl in self.controls:
            if ctrl not in CONTROL_COVERAGE:
                continue
            for threat in CONTROL_COVERAGE[ctrl]["threats"]:
                if threat in threat_to_controls:
                    threat_to_controls[threat].append(ctrl)

        # Build coverage entries
        coverage = [
            ThreatCoverage(
                category=cat,
                description=THREAT_CATEGORIES[cat],
                controls=sorted(ctrls),
            )
            for cat, ctrls in threat_to_controls.items()
        ]

        # Calculate score
        coverage_score = self._calculate_score(coverage)

        # Generate recommendations
        recommendations = self._generate_recommendations(coverage)

        return SafetyNetReport(
            controls_analyzed=sorted(self.controls),
            total_controls=len([c for c in self.controls if c in CONTROL_COVERAGE]),
            threat_coverage=coverage,
            coverage_score=coverage_score,
            recommendations=recommendations,
            min_depth=self.min_depth,
        )

    def _calculate_score(self, coverage: List[ThreatCoverage]) -> float:
        """Calculate weighted coverage score (0-100).

        Scoring:
        - Each threat category has equal weight (100 / n categories)
        - Blind spot: 0 points
        - SPOF: 40% of possible points
        - Depth 2: 80% of possible points
        - Depth 3+: 100% of possible points
        """
        if not coverage:
            return 0.0
        per_category = 100.0 / len(coverage)
        score = 0.0
        for tc in coverage:
            if tc.depth == 0:
                score += 0
            elif tc.depth == 1:
                score += per_category * 0.4
            elif tc.depth == 2:
                score += per_category * 0.8
            else:
                score += per_category
        return min(score, 100.0)

    def _generate_recommendations(
        self, coverage: List[ThreatCoverage]
    ) -> List[Recommendation]:
        """Generate prioritized recommendations."""
        recs: List[Recommendation] = []

        # Priority 1: blind spots
        for tc in coverage:
            if tc.status == "blind_spot":
                suggested = self._suggest_controls(tc.category)
                recs.append(Recommendation(
                    priority=1,
                    category=tc.category,
                    action=f"Add coverage for {tc.category}",
                    reason=f"No controls cover this threat. Consider adding: {', '.join(suggested)}" if suggested else f"No controls cover {tc.category} — custom control needed",
                ))

        # Priority 2: SPOFs
        for tc in coverage:
            if tc.status == "spof":
                recs.append(Recommendation(
                    priority=2,
                    category=tc.category,
                    action=f"Add redundant control for {tc.category}",
                    reason=f"Only [{tc.controls[0]}] covers this — if it fails, the threat is undetected",
                ))

        # Priority 3: below target depth
        for tc in coverage:
            if 1 < tc.depth < self.min_depth:
                recs.append(Recommendation(
                    priority=3,
                    category=tc.category,
                    action=f"Increase depth from {tc.depth} to {self.min_depth}",
                    reason=f"Currently below target depth of {self.min_depth}",
                ))

        return sorted(recs, key=lambda r: (r.priority, r.category))

    @staticmethod
    def _suggest_controls(threat: str) -> List[str]:
        """Suggest controls from the full catalog that could cover a threat."""
        suggestions = []
        for ctrl, info in CONTROL_COVERAGE.items():
            if threat in info["threats"]:
                suggestions.append(ctrl)
        return suggestions[:3]


# ── Convenience function ─────────────────────────────────────────────

def analyze_safety_net(
    controls: Optional[List[str]] = None,
    min_depth: int = 2,
) -> SafetyNetReport:
    """One-call analysis with sensible defaults."""
    analyzer = SafetyNetAnalyzer(controls=controls, min_depth=min_depth)
    return analyzer.analyze()


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m replication safety-net",
        description="Analyze defense-in-depth coverage across threat categories",
    )
    parser.add_argument(
        "--controls",
        help="Comma-separated list of controls to analyze (default: all)",
    )
    parser.add_argument(
        "--min-depth",
        type=int,
        default=2,
        help="Target minimum controls per threat (default: 2)",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json", "html"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Write output to file instead of stdout",
    )
    parser.add_argument(
        "--list-controls",
        action="store_true",
        help="List all known controls and exit",
    )
    parser.add_argument(
        "--list-threats",
        action="store_true",
        help="List all threat categories and exit",
    )
    args = parser.parse_args(argv)

    if args.list_controls:
        print("Available safety controls:")
        for ctrl, info in sorted(CONTROL_COVERAGE.items()):
            threats = ", ".join(info["threats"])
            print(f"  {ctrl:<22} → {info['description']}")
            print(f"  {'':<22}   covers: {threats}")
        return

    if args.list_threats:
        print("Threat categories:")
        for cat, desc in sorted(THREAT_CATEGORIES.items()):
            print(f"  {cat:<22} {desc}")
        return

    controls = args.controls.split(",") if args.controls else None
    report = analyze_safety_net(controls=controls, min_depth=args.min_depth)

    if args.format == "json":
        output = json.dumps(report.to_dict(), indent=2)
    elif args.format == "html":
        output = report.to_html()
    else:
        output = report.render()

    from ._helpers import emit_output
    emit_output(output, args.output, "Report")


if __name__ == "__main__":
    main()
