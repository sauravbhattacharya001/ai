"""Supply Chain Risk Analyzer — AI agent dependency & component risk analysis.

Models the software supply chain of an AI agent system and identifies
risk concentrations, single points of failure, and transitive trust
issues.  Useful for pre-deployment audits and ongoing supply-chain
hygiene monitoring.

Usage (library)::

    from replication.supply_chain import SupplyChainAnalyzer, Component

    analyzer = SupplyChainAnalyzer()
    analyzer.add_component(Component("llm-provider", vendor="acme", tier="critical"))
    analyzer.add_component(Component("vector-db", vendor="beta", tier="high",
                                      depends_on=["cloud-storage"]))
    report = analyzer.analyze()
    print(report.summary())

CLI::

    python -m replication supply-chain
    python -m replication supply-chain --add llm:acme:critical --add vectordb:beta:high
    python -m replication supply-chain --format json
    python -m replication supply-chain --format html -o supply-chain.html
"""

from __future__ import annotations

import argparse
import html as _html
import json
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


# ── Risk tiers & weights ─────────────────────────────────────────────

TIER_WEIGHTS = {
    "critical": 10,
    "high": 7,
    "medium": 4,
    "low": 1,
}

# ── Built-in AI agent supply-chain components ────────────────────────
# A realistic starting set; callers can add/override.

@dataclass
class Component:
    """A supply-chain component (dependency, service, or module)."""
    name: str
    vendor: str = "unknown"
    tier: str = "medium"          # critical / high / medium / low
    depends_on: List[str] = field(default_factory=list)
    license: str = ""
    pinned_version: bool = False  # is the version pinned?
    verified: bool = False        # has the component been audited?
    notes: str = ""

    @property
    def weight(self) -> int:
        return TIER_WEIGHTS.get(self.tier, 4)


DEFAULT_COMPONENTS: Dict[str, Component] = {
    "llm-provider": Component("llm-provider", vendor="openai", tier="critical"),
    "embedding-model": Component("embedding-model", vendor="openai", tier="high",
                                  depends_on=["llm-provider"]),
    "vector-database": Component("vector-database", vendor="pinecone", tier="high"),
    "orchestrator": Component("orchestrator", vendor="langchain", tier="critical",
                               depends_on=["llm-provider", "vector-database"]),
    "auth-service": Component("auth-service", vendor="auth0", tier="critical"),
    "cloud-storage": Component("cloud-storage", vendor="aws-s3", tier="high"),
    "logging": Component("logging", vendor="datadog", tier="medium",
                          depends_on=["cloud-storage"]),
    "guardrails": Component("guardrails", vendor="internal", tier="critical",
                             depends_on=["llm-provider"]),
    "plugin-runtime": Component("plugin-runtime", vendor="internal", tier="high",
                                 depends_on=["orchestrator", "auth-service"]),
    "data-pipeline": Component("data-pipeline", vendor="airflow", tier="medium",
                                depends_on=["cloud-storage"]),
    "cache-layer": Component("cache-layer", vendor="redis", tier="medium"),
    "monitoring": Component("monitoring", vendor="prometheus", tier="medium",
                             depends_on=["logging"]),
}


@dataclass
class RiskFinding:
    """A single supply-chain risk finding."""
    severity: str        # critical / high / medium / low
    category: str        # e.g. "single-vendor", "unverified", "transitive-depth"
    component: str
    description: str


@dataclass
class SupplyChainReport:
    """Result of a supply-chain risk analysis."""
    components: Dict[str, Component]
    findings: List[RiskFinding] = field(default_factory=list)
    vendor_concentration: Dict[str, List[str]] = field(default_factory=dict)
    max_transitive_depth: Dict[str, int] = field(default_factory=dict)
    single_points_of_failure: List[str] = field(default_factory=list)
    overall_score: float = 0.0  # 0-100, higher = riskier

    def summary(self) -> str:
        lines = [
            "═══ Supply Chain Risk Report ═══",
            f"Components analyzed: {len(self.components)}",
            f"Risk score: {self.overall_score:.1f}/100",
            f"Findings: {len(self.findings)}",
        ]
        if self.single_points_of_failure:
            lines.append(f"Single points of failure: {', '.join(self.single_points_of_failure)}")

        by_sev = {}
        for f in self.findings:
            by_sev.setdefault(f.severity, []).append(f)

        for sev in ("critical", "high", "medium", "low"):
            items = by_sev.get(sev, [])
            if items:
                lines.append(f"\n── {sev.upper()} ({len(items)}) ──")
                for item in items:
                    lines.append(f"  [{item.category}] {item.component}: {item.description}")

        if self.vendor_concentration:
            lines.append("\n── Vendor Concentration ──")
            for vendor, comps in sorted(self.vendor_concentration.items(),
                                         key=lambda x: -len(x[1])):
                if len(comps) > 1:
                    lines.append(f"  {vendor}: {', '.join(comps)} ({len(comps)} components)")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "overall_score": round(self.overall_score, 1),
            "component_count": len(self.components),
            "finding_count": len(self.findings),
            "single_points_of_failure": self.single_points_of_failure,
            "vendor_concentration": self.vendor_concentration,
            "max_transitive_depth": self.max_transitive_depth,
            "findings": [
                {"severity": f.severity, "category": f.category,
                 "component": f.component, "description": f.description}
                for f in self.findings
            ],
        }


class SupplyChainAnalyzer:
    """Analyze an AI agent's software supply chain for risk."""

    def __init__(self, *, include_defaults: bool = True):
        self._components: Dict[str, Component] = {}
        if include_defaults:
            for name, comp in DEFAULT_COMPONENTS.items():
                self._components[name] = Component(
                    name=comp.name, vendor=comp.vendor, tier=comp.tier,
                    depends_on=list(comp.depends_on), license=comp.license,
                    pinned_version=comp.pinned_version, verified=comp.verified,
                    notes=comp.notes,
                )

    # ── mutation ──────────────────────────────────────────────────────

    def add_component(self, comp: Component) -> None:
        self._components[comp.name] = comp

    def remove_component(self, name: str) -> None:
        self._components.pop(name, None)

    # ── analysis ─────────────────────────────────────────────────────

    def analyze(self) -> SupplyChainReport:
        report = SupplyChainReport(components=dict(self._components))

        self._check_vendor_concentration(report)
        self._check_transitive_depth(report)
        self._check_single_points_of_failure(report)
        self._check_unverified(report)
        self._check_unpinned(report)
        self._check_missing_deps(report)

        # Score: weighted sum of findings / max possible
        score = 0.0
        for f in report.findings:
            score += TIER_WEIGHTS.get(f.severity, 4)
        max_score = len(self._components) * 10 * 3  # rough ceiling
        report.overall_score = min(100.0, (score / max(max_score, 1)) * 100)

        return report

    def _check_vendor_concentration(self, report: SupplyChainReport) -> None:
        vendor_map: Dict[str, List[str]] = {}
        for name, comp in self._components.items():
            vendor_map.setdefault(comp.vendor, []).append(name)
        report.vendor_concentration = vendor_map

        for vendor, comps in vendor_map.items():
            if len(comps) >= 3:
                crit = any(self._components[c].tier == "critical" for c in comps)
                sev = "critical" if crit else "high"
                report.findings.append(RiskFinding(
                    severity=sev,
                    category="vendor-concentration",
                    component=vendor,
                    description=f"Vendor '{vendor}' supplies {len(comps)} components: "
                                f"{', '.join(comps)}. Outage = widespread impact.",
                ))

    def _check_transitive_depth(self, report: SupplyChainReport) -> None:
        for name in self._components:
            depth = self._bfs_depth(name)
            report.max_transitive_depth[name] = depth
            if depth >= 3:
                report.findings.append(RiskFinding(
                    severity="medium",
                    category="transitive-depth",
                    component=name,
                    description=f"Transitive dependency depth of {depth} — "
                                f"deep chains amplify cascading failures.",
                ))

    def _bfs_depth(self, start: str) -> int:
        visited: Set[str] = set()
        queue: deque = deque([(start, 0)])
        max_d = 0
        while queue:
            node, d = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            max_d = max(max_d, d)
            comp = self._components.get(node)
            if comp:
                for dep in comp.depends_on:
                    if dep not in visited:
                        queue.append((dep, d + 1))
        return max_d

    def _check_single_points_of_failure(self, report: SupplyChainReport) -> None:
        # A component is a SPOF if ≥2 other components depend on it
        dependents: Dict[str, List[str]] = {}
        for name, comp in self._components.items():
            for dep in comp.depends_on:
                dependents.setdefault(dep, []).append(name)

        for dep_name, consumers in dependents.items():
            if len(consumers) >= 2:
                comp = self._components.get(dep_name)
                tier = comp.tier if comp else "medium"
                sev = "critical" if tier in ("critical", "high") else "high"
                report.single_points_of_failure.append(dep_name)
                report.findings.append(RiskFinding(
                    severity=sev,
                    category="single-point-of-failure",
                    component=dep_name,
                    description=f"{len(consumers)} components depend on '{dep_name}': "
                                f"{', '.join(consumers)}. If it fails, all fail.",
                ))

    def _check_unverified(self, report: SupplyChainReport) -> None:
        for name, comp in self._components.items():
            if not comp.verified and comp.tier in ("critical", "high"):
                report.findings.append(RiskFinding(
                    severity="high" if comp.tier == "critical" else "medium",
                    category="unverified",
                    component=name,
                    description=f"{comp.tier}-tier component has not been audited/verified.",
                ))

    def _check_unpinned(self, report: SupplyChainReport) -> None:
        for name, comp in self._components.items():
            if not comp.pinned_version and comp.tier in ("critical", "high"):
                report.findings.append(RiskFinding(
                    severity="medium",
                    category="unpinned-version",
                    component=name,
                    description=f"{comp.tier}-tier component version is not pinned. "
                                f"Auto-updates may introduce regressions.",
                ))

    def _check_missing_deps(self, report: SupplyChainReport) -> None:
        known = set(self._components.keys())
        for name, comp in self._components.items():
            for dep in comp.depends_on:
                if dep not in known:
                    report.findings.append(RiskFinding(
                        severity="high",
                        category="missing-dependency",
                        component=name,
                        description=f"Depends on '{dep}' which is not in the component registry.",
                    ))

    # ── HTML output ──────────────────────────────────────────────────

    def render_html(self, report: SupplyChainReport) -> str:
        sev_color = {"critical": "#dc3545", "high": "#fd7e14",
                     "medium": "#ffc107", "low": "#28a745"}
        rows = []
        for f in sorted(report.findings, key=lambda x: -TIER_WEIGHTS.get(x.severity, 0)):
            color = sev_color.get(f.severity, "#6c757d")
            rows.append(
                f"<tr><td style='color:{color};font-weight:bold'>{_html.escape(f.severity.upper())}</td>"
                f"<td>{_html.escape(f.category)}</td>"
                f"<td>{_html.escape(f.component)}</td>"
                f"<td>{_html.escape(f.description)}</td></tr>"
            )
        table = "\n".join(rows)

        spof_list = ", ".join(report.single_points_of_failure) or "None"
        return f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>Supply Chain Risk Report</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #1a1a2e; color: #e0e0e0; }}
h1 {{ color: #00d4ff; }} h2 {{ color: #7b8cff; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th, td {{ border: 1px solid #333; padding: .5rem .75rem; text-align: left; }}
th {{ background: #16213e; }}
.score {{ font-size: 2rem; font-weight: bold; color: {sev_color.get('critical' if report.overall_score > 60 else 'high' if report.overall_score > 30 else 'low', '#28a745')}; }}
</style></head><body>
<h1>🔗 Supply Chain Risk Report</h1>
<p>Components: {len(report.components)} &nbsp;|&nbsp;
   Findings: {len(report.findings)} &nbsp;|&nbsp;
   SPOFs: {_html.escape(spof_list)}</p>
<p class='score'>Risk Score: {report.overall_score:.1f}/100</p>
<h2>Findings</h2>
<table><tr><th>Severity</th><th>Category</th><th>Component</th><th>Description</th></tr>
{table}
</table></body></html>"""


# ── CLI ──────────────────────────────────────────────────────────────

def build_parser(sub: Optional[argparse._SubParsersAction] = None) -> argparse.ArgumentParser:
    desc = "Supply Chain Risk Analyzer"
    if sub:
        parser = sub.add_parser("supply-chain", help="Analyze AI agent supply-chain risk",
                                description=desc)
    else:
        parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--add", action="append", default=[],
                        metavar="NAME:VENDOR:TIER",
                        help="Add component (name:vendor:tier)")
    parser.add_argument("--remove", action="append", default=[],
                        help="Remove a default component by name")
    parser.add_argument("--no-defaults", action="store_true",
                        help="Start with an empty component registry")
    parser.add_argument("--format", choices=["text", "json", "html"], default="text")
    parser.add_argument("-o", "--output", help="Write output to file")
    return parser


def run(args: argparse.Namespace) -> None:
    analyzer = SupplyChainAnalyzer(include_defaults=not args.no_defaults)

    for spec in args.add:
        parts = spec.split(":")
        name = parts[0]
        vendor = parts[1] if len(parts) > 1 else "unknown"
        tier = parts[2] if len(parts) > 2 else "medium"
        analyzer.add_component(Component(name=name, vendor=vendor, tier=tier))

    for name in args.remove:
        analyzer.remove_component(name)

    report = analyzer.analyze()

    if args.format == "json":
        out = json.dumps(report.to_dict(), indent=2)
    elif args.format == "html":
        out = analyzer.render_html(report)
    else:
        out = report.summary()

    from ._helpers import emit_output
    emit_output(out, args.output)


def main(argv: Optional[list] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
