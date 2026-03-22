"""Blast Radius Analyzer — safety control failure cascade analysis.

Simulates what happens when a safety control fails and maps the impact
across the entire safety stack.  Produces text reports and optional HTML
visualizations showing which controls are affected and how deeply the
failure propagates.

Usage (library)::

    from replication.blast_radius import BlastRadiusAnalyzer

    analyzer = BlastRadiusAnalyzer()
    # optionally add custom controls / edges
    analyzer.add_control("my_check", depends_on=["audit_trail", "policy"])
    result = analyzer.analyze("audit_trail")
    print(result.summary())

CLI::

    python -m replication blast-radius --control audit_trail
    python -m replication blast-radius --all --format html -o blast.html
"""

from __future__ import annotations

import argparse
import html as _html
import json
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ── Built-in safety control dependency graph ─────────────────────────
# Each key is a control; its value lists the controls it *depends on*.
# If A depends on B, then B failing impacts A.

DEFAULT_CONTROLS: Dict[str, List[str]] = {
    "audit_trail":         [],
    "policy":              [],
    "kill_switch":         ["audit_trail"],
    "quarantine":          ["kill_switch", "audit_trail"],
    "compliance":          ["audit_trail", "policy"],
    "drift_detection":     ["audit_trail"],
    "escalation_detect":   ["audit_trail", "drift_detection"],
    "behavior_profiler":   ["audit_trail"],
    "anomaly_replay":      ["audit_trail", "behavior_profiler"],
    "deception_detect":    ["behavior_profiler", "audit_trail"],
    "canary_tokens":       ["audit_trail"],
    "watermark":           ["audit_trail"],
    "threat_intel":        ["audit_trail"],
    "threat_correlator":   ["threat_intel", "audit_trail"],
    "alert_router":        ["threat_correlator", "audit_trail"],
    "sla_monitor":         ["audit_trail", "compliance"],
    "risk_profiler":       ["behavior_profiler", "threat_correlator", "compliance"],
    "safety_benchmark":    ["compliance", "policy"],
    "safety_drill":        ["kill_switch", "quarantine", "alert_router"],
    "consensus":           ["policy"],
    "trust_propagation":   ["behavior_profiler", "consensus"],
    "forensics":           ["audit_trail", "anomaly_replay"],
    "incident_response":   ["forensics", "alert_router", "quarantine"],
    "honeypot":            ["canary_tokens", "audit_trail"],
    "evasion_detect":      ["behavior_profiler", "deception_detect"],
    "prompt_injection":    ["policy", "audit_trail"],
    "selfmod_detect":      ["watermark", "audit_trail"],
    "lineage":             ["audit_trail"],
    "topology":            ["lineage"],
}


@dataclass
class ImpactNode:
    """A single control in the blast radius."""

    name: str
    depth: int  # hops from the failed control (0 = the failed one)
    path: List[str]  # shortest dependency path from failed control


@dataclass
class BlastResult:
    """Result of a blast radius analysis."""

    failed_control: str
    impacted: List[ImpactNode] = field(default_factory=list)
    total_controls: int = 0

    # ── helpers ───────────────────────────────────────────────────────

    @property
    def impact_ratio(self) -> float:
        if self.total_controls == 0:
            return 0.0
        return len(self.impacted) / self.total_controls

    @property
    def max_depth(self) -> int:
        if not self.impacted:
            return 0
        return max(n.depth for n in self.impacted)

    def severity(self) -> str:
        r = self.impact_ratio
        if r >= 0.6:
            return "CRITICAL"
        if r >= 0.35:
            return "HIGH"
        if r >= 0.15:
            return "MEDIUM"
        return "LOW"

    def summary(self) -> str:
        lines = [
            f"╔══ Blast Radius: {self.failed_control} ══╗",
            f"  Severity   : {self.severity()}",
            f"  Impacted   : {len(self.impacted)} / {self.total_controls} controls ({self.impact_ratio:.0%})",
            f"  Max depth  : {self.max_depth} hops",
            "",
        ]
        if self.impacted:
            by_depth: Dict[int, List[ImpactNode]] = {}
            for n in self.impacted:
                by_depth.setdefault(n.depth, []).append(n)
            for d in sorted(by_depth):
                label = "DIRECT" if d == 1 else f"DEPTH {d}"
                names = ", ".join(sorted(n.name for n in by_depth[d]))
                lines.append(f"  [{label}] {names}")
            lines.append("")
            lines.append("  Cascade paths:")
            for n in sorted(self.impacted, key=lambda x: (x.depth, x.name)):
                lines.append(f"    {' → '.join(n.path)}")
        else:
            lines.append("  No downstream controls affected.")
        lines.append(f"╚{'═' * (len(lines[0]) - 2)}╝")
        return "\n".join(lines)


class BlastRadiusAnalyzer:
    """Analyze safety control failure cascade impact."""

    def __init__(self, controls: Optional[Dict[str, List[str]]] = None) -> None:
        self._deps: Dict[str, List[str]] = dict(controls or DEFAULT_CONTROLS)

    # ── mutators ─────────────────────────────────────────────────────

    def add_control(
        self, name: str, depends_on: Optional[List[str]] = None
    ) -> None:
        self._deps[name] = list(depends_on or [])

    def remove_control(self, name: str) -> None:
        self._deps.pop(name, None)
        for deps in self._deps.values():
            if name in deps:
                deps.remove(name)

    # ── analysis ─────────────────────────────────────────────────────

    def _reverse_graph(self) -> Dict[str, List[str]]:
        """Build reverse adjacency: control → list of controls that depend on it."""
        rev: Dict[str, List[str]] = {k: [] for k in self._deps}
        for ctrl, deps in self._deps.items():
            for d in deps:
                rev.setdefault(d, []).append(ctrl)
        return rev

    def analyze(self, failed: str) -> BlastResult:
        """BFS from *failed* through reverse edges to find all impacted controls."""
        rev = self._reverse_graph()
        visited: Dict[str, ImpactNode] = {}
        queue: deque[Tuple[str, int, List[str]]] = deque()

        # seed
        for downstream in rev.get(failed, []):
            queue.append((downstream, 1, [failed, downstream]))

        while queue:
            name, depth, path = queue.popleft()
            if name in visited:
                continue
            node = ImpactNode(name=name, depth=depth, path=path)
            visited[name] = node
            for further in rev.get(name, []):
                if further not in visited:
                    queue.append((further, depth + 1, path + [further]))

        result = BlastResult(
            failed_control=failed,
            impacted=sorted(visited.values(), key=lambda n: (n.depth, n.name)),
            total_controls=len(self._deps),
        )
        return result

    def analyze_all(self) -> List[BlastResult]:
        """Run blast radius for every control, sorted by impact descending."""
        results = [self.analyze(c) for c in self._deps]
        results.sort(key=lambda r: len(r.impacted), reverse=True)
        return results

    # ── export ───────────────────────────────────────────────────────

    def to_html(self, results: Optional[List[BlastResult]] = None) -> str:
        """Render a self-contained HTML report."""
        if results is None:
            results = self.analyze_all()

        rows = []
        for r in results:
            sev = r.severity()
            color = {"CRITICAL": "#e74c3c", "HIGH": "#e67e22", "MEDIUM": "#f1c40f", "LOW": "#2ecc71"}.get(sev, "#999")
            impacted_names = ", ".join(n.name for n in r.impacted) or "—"
            rows.append(
                f"<tr>"
                f"<td><strong>{_html.escape(r.failed_control)}</strong></td>"
                f"<td style='color:{color};font-weight:bold'>{sev}</td>"
                f"<td>{len(r.impacted)}/{r.total_controls} ({r.impact_ratio:.0%})</td>"
                f"<td>{r.max_depth}</td>"
                f"<td style='font-size:0.85em'>{_html.escape(impacted_names)}</td>"
                f"</tr>"
            )

        return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Blast Radius Report</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2em; background: #0d1117; color: #c9d1d9; }}
  h1 {{ color: #58a6ff; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 1em; }}
  th, td {{ border: 1px solid #30363d; padding: 8px 12px; text-align: left; }}
  th {{ background: #161b22; color: #8b949e; }}
  tr:hover {{ background: #161b22; }}
</style></head><body>
<h1>🔥 Safety Control Blast Radius Report</h1>
<p>{len(self._deps)} controls analyzed</p>
<table>
<tr><th>Control</th><th>Severity</th><th>Impact</th><th>Max Depth</th><th>Affected Controls</th></tr>
{"".join(rows)}
</table>
</body></html>"""

    def to_json(self, results: Optional[List[BlastResult]] = None) -> str:
        """Serialize results to JSON."""
        if results is None:
            results = self.analyze_all()
        data = []
        for r in results:
            data.append({
                "control": r.failed_control,
                "severity": r.severity(),
                "impacted_count": len(r.impacted),
                "total_controls": r.total_controls,
                "impact_ratio": round(r.impact_ratio, 3),
                "max_depth": r.max_depth,
                "impacted": [
                    {"name": n.name, "depth": n.depth, "path": n.path}
                    for n in r.impacted
                ],
            })
        return json.dumps(data, indent=2)


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m replication blast-radius",
        description="Analyze safety control failure blast radius",
    )
    parser.add_argument(
        "--control", "-c",
        help="Specific control to analyze (default: analyze all)",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        dest="analyze_all",
        help="Analyze every control (default when --control is omitted)",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["text", "html", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Write output to file instead of stdout",
    )
    args = parser.parse_args(argv)

    analyzer = BlastRadiusAnalyzer()

    if args.control:
        if args.control not in analyzer._deps:
            print(f"Unknown control: {args.control}", file=sys.stderr)
            print(f"Available: {', '.join(sorted(analyzer._deps))}", file=sys.stderr)
            sys.exit(1)
        results = [analyzer.analyze(args.control)]
    else:
        results = analyzer.analyze_all()

    if args.format == "html":
        output = analyzer.to_html(results)
    elif args.format == "json":
        output = analyzer.to_json(results)
    else:
        output = "\n\n".join(r.summary() for r in results)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
