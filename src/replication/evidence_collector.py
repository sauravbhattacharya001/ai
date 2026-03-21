"""Evidence Collector — package safety artifacts for audit & compliance reviews.

Gathers outputs from multiple safety tools into a structured evidence package
suitable for compliance audits, regulatory reviews, or internal assessments.

Features
--------
- **Automated collection** from scorecard, compliance, drift, audit-trail, etc.
- **Evidence manifest** with SHA-256 hashes for tamper detection.
- **HTML summary report** linking all collected artifacts.
- **Configurable collectors** — pick which tools to run.
- **Evidence tagging** — map artifacts to compliance framework controls.
- **ZIP packaging** for easy sharing.

Usage (CLI)::

    python -m replication evidence                           # collect all
    python -m replication evidence --collectors scorecard,compliance,drift
    python -m replication evidence --framework nist_ai_rmf   # tag by framework
    python -m replication evidence --zip -o evidence.zip     # ZIP package
    python -m replication evidence --html -o evidence.html   # HTML summary only
    python -m replication evidence --list                    # list available collectors
    python -m replication evidence --dry-run                 # show what would run

Programmatic::

    from replication.evidence_collector import EvidenceCollector
    ec = EvidenceCollector()
    package = ec.collect()
    package.to_html("evidence.html")
    package.to_zip("evidence.zip")
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import html as html_mod
import io
import json
import sys
import textwrap
import zipfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ── Data models ──────────────────────────────────────────────────────

@dataclass
class Artifact:
    """A single piece of collected evidence."""

    collector: str
    title: str
    content: str
    timestamp: str = ""
    tags: List[str] = field(default_factory=list)
    sha256: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.datetime.utcnow().isoformat() + "Z"
        if not self.sha256:
            self.sha256 = hashlib.sha256(self.content.encode()).hexdigest()


@dataclass
class EvidencePackage:
    """Collection of artifacts with manifest."""

    artifacts: List[Artifact] = field(default_factory=list)
    collected_at: str = ""
    framework: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.collected_at:
            self.collected_at = datetime.datetime.utcnow().isoformat() + "Z"

    @property
    def manifest(self) -> List[Dict[str, Any]]:
        """Generate tamper-evident manifest."""
        return [
            {
                "collector": a.collector,
                "title": a.title,
                "timestamp": a.timestamp,
                "sha256": a.sha256,
                "tags": a.tags,
                "size_bytes": len(a.content.encode()),
            }
            for a in self.artifacts
        ]

    def to_json(self) -> str:
        """Serialize full package to JSON."""
        return json.dumps(
            {
                "collected_at": self.collected_at,
                "framework": self.framework,
                "artifact_count": len(self.artifacts),
                "manifest": self.manifest,
                "artifacts": [
                    {
                        "collector": a.collector,
                        "title": a.title,
                        "content": a.content,
                        "timestamp": a.timestamp,
                        "sha256": a.sha256,
                        "tags": a.tags,
                    }
                    for a in self.artifacts
                ],
            },
            indent=2,
        )

    def to_html(self, path: Optional[str] = None) -> str:
        """Render an HTML summary report."""
        rows = ""
        for i, a in enumerate(self.artifacts, 1):
            tag_badges = " ".join(
                f'<span class="tag">{html_mod.escape(t)}</span>' for t in a.tags
            )
            rows += f"""<tr>
                <td>{i}</td>
                <td>{html_mod.escape(a.collector)}</td>
                <td>{html_mod.escape(a.title)}</td>
                <td class="mono">{a.sha256[:16]}…</td>
                <td>{tag_badges or "—"}</td>
                <td>{len(a.content):,} chars</td>
                <td>{html_mod.escape(a.timestamp)}</td>
            </tr>"""

        detail_sections = ""
        for i, a in enumerate(self.artifacts, 1):
            detail_sections += f"""
            <div class="artifact">
                <h3>#{i} — {html_mod.escape(a.title)}</h3>
                <p class="meta">Collector: {html_mod.escape(a.collector)} |
                   Hash: <code>{a.sha256}</code> |
                   Collected: {html_mod.escape(a.timestamp)}</p>
                <pre>{html_mod.escape(a.content[:5000])}</pre>
            </div>"""

        fw_line = (
            f"<p><strong>Framework:</strong> {html_mod.escape(self.framework)}</p>"
            if self.framework
            else ""
        )

        page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Evidence Package — {html_mod.escape(self.collected_at)}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #f8f9fa; color: #212529; }}
  h1 {{ color: #0d6efd; }}
  h2 {{ border-bottom: 2px solid #dee2e6; padding-bottom: .3rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{ border: 1px solid #dee2e6; padding: .5rem .75rem; text-align: left; }}
  th {{ background: #e9ecef; }}
  .mono {{ font-family: monospace; font-size: .85rem; }}
  .tag {{ background: #0d6efd; color: #fff; padding: 2px 6px; border-radius: 3px;
          font-size: .8rem; margin-right: 3px; }}
  .artifact {{ background: #fff; border: 1px solid #dee2e6; border-radius: 6px;
               padding: 1rem; margin: 1rem 0; }}
  .artifact pre {{ background: #f1f3f5; padding: 1rem; overflow-x: auto;
                   border-radius: 4px; font-size: .85rem; max-height: 400px; }}
  .meta {{ color: #6c757d; font-size: .9rem; }}
  .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
              gap: 1rem; margin: 1rem 0; }}
  .card {{ background: #fff; border: 1px solid #dee2e6; border-radius: 6px;
           padding: 1rem; text-align: center; }}
  .card .num {{ font-size: 2rem; font-weight: bold; color: #0d6efd; }}
</style></head><body>
<h1>📦 Evidence Package</h1>
<p>Collected: {html_mod.escape(self.collected_at)}</p>
{fw_line}
<div class="summary">
  <div class="card"><div class="num">{len(self.artifacts)}</div>Artifacts</div>
  <div class="card"><div class="num">{len(set(a.collector for a in self.artifacts))}</div>Collectors</div>
  <div class="card"><div class="num">{sum(len(a.content.encode()) for a in self.artifacts):,}</div>Total Bytes</div>
</div>
<h2>Manifest</h2>
<table><thead><tr><th>#</th><th>Collector</th><th>Title</th><th>SHA-256</th><th>Tags</th><th>Size</th><th>Time</th></tr></thead>
<tbody>{rows}</tbody></table>
<h2>Artifact Details</h2>
{detail_sections}
</body></html>"""

        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(page)
        return page

    def to_zip(self, path: str) -> None:
        """Write a ZIP containing all artifacts + manifest."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(self.manifest, indent=2))
            for i, a in enumerate(self.artifacts, 1):
                safe_name = a.title.replace(" ", "_").replace("/", "_")[:60]
                zf.writestr(
                    f"{i:03d}_{a.collector}_{safe_name}.txt", a.content
                )
            zf.writestr("summary.html", self.to_html())
        with open(path, "wb") as fh:
            fh.write(buf.getvalue())

    def render(self) -> str:
        """Plain text summary."""
        lines = [
            f"Evidence Package — {self.collected_at}",
            f"Framework: {self.framework or 'none'}",
            f"Artifacts: {len(self.artifacts)}",
            "=" * 60,
        ]
        for i, a in enumerate(self.artifacts, 1):
            tags = ", ".join(a.tags) if a.tags else "—"
            lines.append(
                f"  {i:>3}. [{a.collector}] {a.title}  "
                f"({len(a.content):,} chars)  tags={tags}"
            )
            lines.append(f"       sha256={a.sha256[:32]}…")
        lines.append("=" * 60)
        return "\n".join(lines)


# ── Framework → control tag mappings ─────────────────────────────────

FRAMEWORK_TAGS: Dict[str, Dict[str, List[str]]] = {
    "nist_ai_rmf": {
        "scorecard": ["MAP-1.1", "MEASURE-2.1"],
        "compliance": ["GOVERN-1.1", "GOVERN-3.2"],
        "drift": ["MEASURE-2.3", "MANAGE-2.1"],
        "audit_trail": ["GOVERN-4.1", "MANAGE-3.2"],
        "risk_heatmap": ["MAP-3.1", "MEASURE-1.1"],
        "policy_lint": ["GOVERN-1.2", "GOVERN-2.1"],
        "maturity": ["GOVERN-5.1", "MANAGE-4.1"],
        "sla": ["MANAGE-1.1", "MEASURE-3.1"],
        "trend": ["MEASURE-2.2", "MANAGE-2.2"],
        "alignment": ["MAP-2.1", "MEASURE-2.1"],
    },
    "iso_42001": {
        "scorecard": ["6.1", "9.1"],
        "compliance": ["4.1", "10.2"],
        "drift": ["9.1", "10.1"],
        "audit_trail": ["7.5", "9.2"],
        "risk_heatmap": ["6.1", "8.1"],
        "policy_lint": ["5.2", "7.1"],
        "maturity": ["9.3", "10.1"],
        "sla": ["8.1", "9.1"],
        "trend": ["9.1", "10.1"],
        "alignment": ["6.1", "8.2"],
    },
    "eu_ai_act": {
        "scorecard": ["Art.9", "Art.15"],
        "compliance": ["Art.9", "Art.17"],
        "drift": ["Art.9.4", "Art.15"],
        "audit_trail": ["Art.12", "Art.20"],
        "risk_heatmap": ["Art.9", "Art.14"],
        "policy_lint": ["Art.9.1", "Art.17"],
        "maturity": ["Art.17", "Art.9"],
        "sla": ["Art.15", "Art.9"],
        "trend": ["Art.9.4", "Art.61"],
        "alignment": ["Art.9", "Art.15"],
    },
}


# ── Collectors ───────────────────────────────────────────────────────

CollectorFn = Callable[[], Artifact]


def _collect_scorecard() -> Artifact:
    """Run scorecard and capture output."""
    try:
        from .scorecard import SafetyScorecard
        sc = SafetyScorecard()
        result = sc.evaluate()
        content = json.dumps(result, indent=2, default=str)
    except Exception as exc:
        content = f"Error running scorecard: {exc}"
    return Artifact(collector="scorecard", title="Safety Scorecard", content=content)


def _collect_compliance() -> Artifact:
    try:
        from .compliance import ComplianceAuditor
        auditor = ComplianceAuditor()
        report = auditor.audit()
        content = json.dumps(report, indent=2, default=str)
    except Exception as exc:
        content = f"Error running compliance: {exc}"
    return Artifact(collector="compliance", title="Compliance Audit", content=content)


def _collect_drift() -> Artifact:
    try:
        from .drift import DriftDetector
        dd = DriftDetector()
        result = dd.detect()
        content = json.dumps(result, indent=2, default=str)
    except Exception as exc:
        content = f"Error running drift: {exc}"
    return Artifact(collector="drift", title="Behavioral Drift Report", content=content)


def _collect_policy_lint() -> Artifact:
    try:
        from .policy_linter import PolicyLinter
        linter = PolicyLinter()
        findings = linter.lint()
        content = json.dumps(findings, indent=2, default=str)
    except Exception as exc:
        content = f"Error running policy lint: {exc}"
    return Artifact(collector="policy_lint", title="Policy Lint Findings", content=content)


def _collect_alignment() -> Artifact:
    try:
        from .alignment import AlignmentVerifier
        av = AlignmentVerifier()
        result = av.verify()
        content = json.dumps(result, indent=2, default=str)
    except Exception as exc:
        content = f"Error running alignment: {exc}"
    return Artifact(collector="alignment", title="Alignment Verification", content=content)


def _collect_maturity() -> Artifact:
    try:
        from .maturity_model import MaturityAssessor
        ma = MaturityAssessor()
        result = ma.assess()
        content = json.dumps(result, indent=2, default=str)
    except Exception as exc:
        content = f"Error running maturity: {exc}"
    return Artifact(collector="maturity", title="Maturity Assessment", content=content)


def _collect_sla() -> Artifact:
    try:
        from .sla_monitor import SLAMonitor
        mon = SLAMonitor()
        result = mon.check()
        content = json.dumps(result, indent=2, default=str)
    except Exception as exc:
        content = f"Error running SLA check: {exc}"
    return Artifact(collector="sla", title="SLA Compliance Check", content=content)


def _collect_trend() -> Artifact:
    try:
        from .trend_tracker import TrendTracker
        tt = TrendTracker()
        result = tt.summary()
        content = json.dumps(result, indent=2, default=str)
    except Exception as exc:
        content = f"Error running trend: {exc}"
    return Artifact(collector="trend", title="Safety Trend Summary", content=content)


def _collect_audit_trail() -> Artifact:
    try:
        from .audit_trail import AuditTrailStore
        store = AuditTrailStore()
        recent = store.recent(limit=50)
        content = json.dumps(recent, indent=2, default=str)
    except Exception as exc:
        content = f"Error reading audit trail: {exc}"
    return Artifact(collector="audit_trail", title="Recent Audit Trail", content=content)


def _collect_risk_heatmap() -> Artifact:
    try:
        from .risk_heatmap import RiskHeatmapGenerator
        gen = RiskHeatmapGenerator()
        data = gen.generate_data()
        content = json.dumps(data, indent=2, default=str)
    except Exception as exc:
        content = f"Error generating risk heatmap: {exc}"
    return Artifact(collector="risk_heatmap", title="Risk Heatmap Data", content=content)


COLLECTORS: Dict[str, Tuple[CollectorFn, str]] = {
    "scorecard":    (_collect_scorecard,    "Safety scorecard evaluation"),
    "compliance":   (_collect_compliance,   "Compliance framework audit"),
    "drift":        (_collect_drift,        "Behavioral drift detection"),
    "policy_lint":  (_collect_policy_lint,   "Policy configuration linting"),
    "alignment":    (_collect_alignment,    "Alignment verification"),
    "maturity":     (_collect_maturity,     "Maturity model assessment"),
    "sla":          (_collect_sla,          "SLA target compliance"),
    "trend":        (_collect_trend,        "Safety trend summary"),
    "audit_trail":  (_collect_audit_trail,  "Recent audit trail events"),
    "risk_heatmap": (_collect_risk_heatmap, "Risk heatmap data"),
}


# ── Main collector class ─────────────────────────────────────────────

class EvidenceCollector:
    """Orchestrates evidence collection from multiple safety tools."""

    def __init__(
        self,
        collectors: Optional[List[str]] = None,
        framework: str = "",
    ) -> None:
        self.collector_names = collectors or list(COLLECTORS.keys())
        self.framework = framework

    def collect(self) -> EvidencePackage:
        """Run all configured collectors and return an evidence package."""
        artifacts: List[Artifact] = []
        fw_tags = FRAMEWORK_TAGS.get(self.framework, {})

        for name in self.collector_names:
            if name not in COLLECTORS:
                continue
            fn, _desc = COLLECTORS[name]
            try:
                artifact = fn()
                artifact.tags = list(fw_tags.get(name, []))
                artifacts.append(artifact)
            except Exception as exc:
                artifacts.append(
                    Artifact(
                        collector=name,
                        title=f"{name} (failed)",
                        content=f"Collection error: {exc}",
                    )
                )

        return EvidencePackage(
            artifacts=artifacts,
            framework=self.framework,
        )


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="replication evidence",
        description="Collect safety evidence artifacts for audit & compliance",
    )
    parser.add_argument(
        "--collectors", "-c",
        help="Comma-separated list of collectors to run (default: all)",
    )
    parser.add_argument(
        "--framework", "-f",
        choices=list(FRAMEWORK_TAGS.keys()),
        default="",
        help="Tag artifacts with compliance framework controls",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML summary report",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Generate ZIP evidence package",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output full package as JSON",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_collectors",
        help="List available collectors and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be collected without running",
    )
    args = parser.parse_args(argv)

    if args.list_collectors:
        print("Available evidence collectors:\n")
        for name, (_fn, desc) in sorted(COLLECTORS.items()):
            print(f"  {name:<16} {desc}")
        return

    collector_names = None
    if args.collectors:
        collector_names = [c.strip() for c in args.collectors.split(",")]

    if args.dry_run:
        names = collector_names or list(COLLECTORS.keys())
        print("Dry run — would collect from:\n")
        for n in names:
            if n in COLLECTORS:
                _, desc = COLLECTORS[n]
                print(f"  ✓ {n:<16} {desc}")
            else:
                print(f"  ✗ {n:<16} (unknown collector)")
        if args.framework:
            print(f"\nFramework tagging: {args.framework}")
        return

    ec = EvidenceCollector(collectors=collector_names, framework=args.framework)
    package = ec.collect()

    if args.zip:
        out = args.output or "evidence_package.zip"
        package.to_zip(out)
        print(f"Evidence package written to {out}")
        print(f"  {len(package.artifacts)} artifacts, framework={args.framework or 'none'}")
        return

    if args.html:
        out = args.output or "evidence_report.html"
        package.to_html(out)
        print(f"Evidence report written to {out}")
        return

    if args.json:
        out_str = package.to_json()
        if args.output:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(out_str)
            print(f"JSON written to {args.output}")
        else:
            print(out_str)
        return

    # Default: text summary
    print(package.render())


if __name__ == "__main__":
    main()
