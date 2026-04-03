"""Safety Diff — structured comparison between safety snapshots.

Compare two safety assessment snapshots (scorecards, compliance results,
risk-register entries) and produce a categorized diff showing exactly what
changed: improvements, regressions, new findings, and resolved findings.

Useful for answering questions like:

- "Did this policy change actually improve our safety posture?"
- "What regressed between last week's run and today?"
- "Which compliance findings were resolved?"

Usage (CLI)::

    # Diff two saved scorecard JSON files
    python -m replication.safety_diff scorecard before.json after.json

    # Diff two compliance JSON files
    python -m replication.safety_diff compliance before.json after.json

    # Run two live scorecards with different configs and diff them
    python -m replication.safety_diff live --before-strategy greedy --after-strategy cautious

    # JSON output
    python -m replication.safety_diff scorecard a.json b.json --json

    # Export diff as HTML report
    python -m replication.safety_diff scorecard a.json b.json --html diff_report.html

Programmatic::

    from replication.safety_diff import SafetyDiff, SnapshotPair

    diff = SafetyDiff.from_scorecard_files("before.json", "after.json")
    print(diff.render())
    print(f"Improvements: {len(diff.improvements)}")
    print(f"Regressions:  {len(diff.regressions)}")
    print(f"Net change:   {diff.net_score_change:+.1f}")
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class ChangeKind(str, Enum):
    """Category of change between snapshots."""
    IMPROVEMENT = "improvement"
    REGRESSION = "regression"
    NEW = "new"
    RESOLVED = "resolved"
    UNCHANGED = "unchanged"


@dataclass
class DimensionChange:
    """Change in a single safety dimension between snapshots."""
    name: str
    before_score: Optional[float]
    after_score: Optional[float]
    before_grade: Optional[str]
    after_grade: Optional[str]
    kind: ChangeKind
    delta: float  # positive = improvement

    @property
    def arrow(self) -> str:
        if self.delta > 5:
            return "⬆️"
        elif self.delta > 0:
            return "↗️"
        elif self.delta < -5:
            return "⬇️"
        elif self.delta < 0:
            return "↘️"
        return "➡️"


@dataclass
class ComplianceChange:
    """Change in a compliance finding between snapshots."""
    check_id: str
    framework: str
    description: str
    kind: ChangeKind
    before_verdict: Optional[str] = None
    after_verdict: Optional[str] = None


@dataclass
class RiskChange:
    """Change in a risk register entry."""
    risk_id: str
    title: str
    kind: ChangeKind
    before_severity: Optional[str] = None
    after_severity: Optional[str] = None
    before_residual: Optional[float] = None
    after_residual: Optional[float] = None


@dataclass
class SafetyDiffResult:
    """Complete diff between two safety snapshots."""

    # Scorecard dimension changes
    dimension_changes: List[DimensionChange] = field(default_factory=list)
    before_overall: Optional[float] = None
    after_overall: Optional[float] = None

    # Compliance changes
    compliance_changes: List[ComplianceChange] = field(default_factory=list)

    # Risk register changes
    risk_changes: List[RiskChange] = field(default_factory=list)

    # Metadata
    before_label: str = "before"
    after_label: str = "after"
    before_timestamp: Optional[str] = None
    after_timestamp: Optional[str] = None

    # --- Computed properties ---

    @property
    def net_score_change(self) -> float:
        if self.before_overall is not None and self.after_overall is not None:
            return self.after_overall - self.before_overall
        return 0.0

    @property
    def improvements(self) -> List[DimensionChange]:
        return [c for c in self.dimension_changes if c.kind == ChangeKind.IMPROVEMENT]

    @property
    def regressions(self) -> List[DimensionChange]:
        return [c for c in self.dimension_changes if c.kind == ChangeKind.REGRESSION]

    @property
    def new_findings(self) -> List[ComplianceChange]:
        return [c for c in self.compliance_changes if c.kind == ChangeKind.NEW]

    @property
    def resolved_findings(self) -> List[ComplianceChange]:
        return [c for c in self.compliance_changes if c.kind == ChangeKind.RESOLVED]

    @property
    def compliance_regressions(self) -> List[ComplianceChange]:
        return [c for c in self.compliance_changes if c.kind == ChangeKind.REGRESSION]

    @property
    def compliance_improvements(self) -> List[ComplianceChange]:
        return [c for c in self.compliance_changes if c.kind == ChangeKind.IMPROVEMENT]

    @property
    def overall_verdict(self) -> str:
        """One-line summary of the diff."""
        delta = self.net_score_change
        n_regress = len(self.regressions) + len(self.compliance_regressions)
        n_improve = len(self.improvements) + len(self.compliance_improvements) + len(self.resolved_findings)
        if delta > 5 and n_regress == 0:
            return "✅ Significant improvement with no regressions"
        elif delta > 0 and n_regress == 0:
            return "✅ Minor improvement with no regressions"
        elif delta > 0 and n_regress > 0:
            return f"⚠️ Net improvement (+{delta:.1f}) but {n_regress} regression(s)"
        elif delta == 0 and n_regress == 0:
            return "➡️ No significant changes"
        elif delta < -5:
            return f"🚨 Significant regression ({delta:.1f} points)"
        elif delta < 0:
            return f"⚠️ Minor regression ({delta:.1f} points)"
        else:
            return f"⚠️ Mixed results: {n_improve} improvement(s), {n_regress} regression(s)"

    # --- Rendering ---

    def render(self) -> str:
        """Render a human-readable diff report."""
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("  SAFETY DIFF REPORT")
        lines.append(f"  {self.before_label} → {self.after_label}")
        lines.append("=" * 60)
        lines.append("")

        # Overall verdict
        lines.append(f"  {self.overall_verdict}")
        lines.append("")

        # Overall score
        if self.before_overall is not None and self.after_overall is not None:
            delta = self.net_score_change
            sign = "+" if delta >= 0 else ""
            lines.append(f"  Overall Score: {self.before_overall:.1f} → {self.after_overall:.1f} ({sign}{delta:.1f})")
            lines.append("")

        # Dimension changes
        if self.dimension_changes:
            lines.append("─" * 60)
            lines.append("  SCORECARD DIMENSIONS")
            lines.append("─" * 60)
            for ch in sorted(self.dimension_changes, key=lambda c: c.delta):
                grade_str = ""
                if ch.before_grade and ch.after_grade:
                    if ch.before_grade != ch.after_grade:
                        grade_str = f" ({ch.before_grade} → {ch.after_grade})"
                    else:
                        grade_str = f" ({ch.before_grade})"
                score_before = f"{ch.before_score:.1f}" if ch.before_score is not None else "—"
                score_after = f"{ch.after_score:.1f}" if ch.after_score is not None else "—"
                sign = "+" if ch.delta >= 0 else ""
                lines.append(f"  {ch.arrow} {ch.name:<24} {score_before:>5} → {score_after:<5} ({sign}{ch.delta:.1f}){grade_str}")
            lines.append("")

        # Compliance changes
        if self.compliance_changes:
            lines.append("─" * 60)
            lines.append("  COMPLIANCE FINDINGS")
            lines.append("─" * 60)
            for section, items in [
                ("🚨 New Failures", self.new_findings),
                ("⚠️ Regressions (PASS → FAIL)", self.compliance_regressions),
                ("✅ Resolved (FAIL → PASS)", self.resolved_findings),
                ("✅ Improvements", self.compliance_improvements),
            ]:
                if items:
                    lines.append(f"  {section}:")
                    for c in items:
                        verdict_str = ""
                        if c.before_verdict and c.after_verdict:
                            verdict_str = f" [{c.before_verdict} → {c.after_verdict}]"
                        elif c.after_verdict:
                            verdict_str = f" [{c.after_verdict}]"
                        lines.append(f"    • {c.check_id} ({c.framework}): {c.description}{verdict_str}")
                    lines.append("")

        # Risk register changes
        if self.risk_changes:
            lines.append("─" * 60)
            lines.append("  RISK REGISTER")
            lines.append("─" * 60)
            for section, kind in [
                ("🆕 New Risks", ChangeKind.NEW),
                ("✅ Resolved Risks", ChangeKind.RESOLVED),
                ("⬆️ Escalated", ChangeKind.REGRESSION),
                ("⬇️ De-escalated", ChangeKind.IMPROVEMENT),
            ]:
                items = [r for r in self.risk_changes if r.kind == kind]
                if items:
                    lines.append(f"  {section}:")
                    for r in items:
                        sev_str = ""
                        if r.before_severity and r.after_severity:
                            sev_str = f" [{r.before_severity} → {r.after_severity}]"
                        elif r.after_severity:
                            sev_str = f" [{r.after_severity}]"
                        lines.append(f"    • {r.risk_id}: {r.title}{sev_str}")
                    lines.append("")

        # Summary stats
        lines.append("─" * 60)
        lines.append("  SUMMARY")
        lines.append("─" * 60)
        lines.append(f"  Dimensions improved:    {len(self.improvements)}")
        lines.append(f"  Dimensions regressed:   {len(self.regressions)}")
        lines.append(f"  Compliance resolved:    {len(self.resolved_findings)}")
        lines.append(f"  Compliance new failures:{len(self.new_findings)}")
        lines.append(f"  Risk changes:           {len(self.risk_changes)}")
        lines.append("=" * 60)

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        return {
            "before_label": self.before_label,
            "after_label": self.after_label,
            "before_timestamp": self.before_timestamp,
            "after_timestamp": self.after_timestamp,
            "before_overall": self.before_overall,
            "after_overall": self.after_overall,
            "net_score_change": self.net_score_change,
            "overall_verdict": self.overall_verdict,
            "dimension_changes": [
                {
                    "name": c.name,
                    "before_score": c.before_score,
                    "after_score": c.after_score,
                    "before_grade": c.before_grade,
                    "after_grade": c.after_grade,
                    "kind": c.kind.value,
                    "delta": c.delta,
                }
                for c in self.dimension_changes
            ],
            "compliance_changes": [
                {
                    "check_id": c.check_id,
                    "framework": c.framework,
                    "description": c.description,
                    "kind": c.kind.value,
                    "before_verdict": c.before_verdict,
                    "after_verdict": c.after_verdict,
                }
                for c in self.compliance_changes
            ],
            "risk_changes": [
                {
                    "risk_id": r.risk_id,
                    "title": r.title,
                    "kind": r.kind.value,
                    "before_severity": r.before_severity,
                    "after_severity": r.after_severity,
                    "before_residual": r.before_residual,
                    "after_residual": r.after_residual,
                }
                for r in self.risk_changes
            ],
            "summary": {
                "improvements": len(self.improvements),
                "regressions": len(self.regressions),
                "compliance_resolved": len(self.resolved_findings),
                "compliance_new_failures": len(self.new_findings),
                "risk_changes": len(self.risk_changes),
            },
        }

    def to_html(self) -> str:
        """Render an interactive HTML diff report."""
        d = self.to_dict()
        delta = self.net_score_change
        sign = "+" if delta >= 0 else ""
        score_color = "#4caf50" if delta >= 0 else "#f44336"

        dim_rows = ""
        for c in sorted(self.dimension_changes, key=lambda x: x.delta):
            row_color = "#e8f5e9" if c.delta > 0 else "#ffebee" if c.delta < 0 else "#fff"
            s = "+" if c.delta >= 0 else ""
            bsc = f"{c.before_score:.1f}" if c.before_score is not None else "—"
            asc = f"{c.after_score:.1f}" if c.after_score is not None else "—"
            bg = c.before_grade or "—"
            ag = c.after_grade or "—"
            dim_rows += (
                f'<tr style="background:{row_color}">'
                f'<td>{c.arrow} {c.name}</td>'
                f'<td style="text-align:center">{bsc}</td>'
                f'<td style="text-align:center">{asc}</td>'
                f'<td style="text-align:center">{s}{c.delta:.1f}</td>'
                f'<td style="text-align:center">{bg} → {ag}</td>'
                f'</tr>'
            )

        comp_rows = ""
        for c in self.compliance_changes:
            icons = {ChangeKind.NEW: "🚨", ChangeKind.RESOLVED: "✅",
                     ChangeKind.REGRESSION: "⚠️", ChangeKind.IMPROVEMENT: "✅"}
            icon = icons.get(c.kind, "")
            bv = c.before_verdict or "—"
            av = c.after_verdict or "—"
            comp_rows += (
                f'<tr><td>{icon} {c.check_id}</td><td>{c.framework}</td>'
                f'<td>{c.description}</td><td>{bv} → {av}</td>'
                f'<td>{c.kind.value}</td></tr>'
            )

        risk_rows = ""
        for r in self.risk_changes:
            icons = {ChangeKind.NEW: "🆕", ChangeKind.RESOLVED: "✅",
                     ChangeKind.REGRESSION: "⬆️", ChangeKind.IMPROVEMENT: "⬇️"}
            icon = icons.get(r.kind, "")
            bs = r.before_severity or "—"
            ars = r.after_severity or "—"
            risk_rows += (
                f'<tr><td>{icon} {r.risk_id}</td><td>{r.title}</td>'
                f'<td>{bs} → {ars}</td><td>{r.kind.value}</td></tr>'
            )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Safety Diff Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #333; }}
  h1 {{ text-align: center; }}
  .verdict {{ text-align: center; font-size: 1.2rem; padding: 1rem; border-radius: 8px; margin: 1rem 0; background: #f5f5f5; }}
  .score-box {{ display: flex; justify-content: center; gap: 2rem; margin: 1rem 0; }}
  .score {{ text-align: center; padding: 1rem 2rem; border-radius: 8px; background: #fafafa; }}
  .score .value {{ font-size: 2rem; font-weight: bold; }}
  .score .label {{ color: #666; }}
  .delta {{ font-size: 1.5rem; font-weight: bold; color: {score_color}; text-align: center; }}
  table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
  th, td {{ padding: 8px 12px; border: 1px solid #ddd; text-align: left; }}
  th {{ background: #f5f5f5; }}
  .section {{ margin: 2rem 0; }}
  .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; }}
  .stat {{ padding: 1rem; border-radius: 8px; text-align: center; }}
  .stat .n {{ font-size: 1.5rem; font-weight: bold; }}
  .stat-good {{ background: #e8f5e9; color: #2e7d32; }}
  .stat-bad {{ background: #ffebee; color: #c62828; }}
  .stat-neutral {{ background: #f5f5f5; color: #666; }}
  details {{ margin: 0.5rem 0; }}
  summary {{ cursor: pointer; font-weight: bold; }}
</style>
</head>
<body>
<h1>🔍 Safety Diff Report</h1>
<p style="text-align:center;color:#666">{self.before_label} → {self.after_label}</p>
<div class="verdict">{self.overall_verdict}</div>

<div class="score-box">
  <div class="score"><div class="label">Before</div><div class="value">{self.before_overall or 0:.1f}</div></div>
  <div class="score"><div class="label">Change</div><div class="delta">{sign}{delta:.1f}</div></div>
  <div class="score"><div class="label">After</div><div class="value">{self.after_overall or 0:.1f}</div></div>
</div>

<div class="summary">
  <div class="stat stat-good"><div class="n">{len(self.improvements)}</div><div>Improved</div></div>
  <div class="stat stat-bad"><div class="n">{len(self.regressions)}</div><div>Regressed</div></div>
  <div class="stat stat-good"><div class="n">{len(self.resolved_findings)}</div><div>Resolved</div></div>
  <div class="stat stat-bad"><div class="n">{len(self.new_findings)}</div><div>New Failures</div></div>
</div>

{"<div class='section'><h2>📊 Scorecard Dimensions</h2><table><tr><th>Dimension</th><th>Before</th><th>After</th><th>Delta</th><th>Grade</th></tr>" + dim_rows + "</table></div>" if dim_rows else ""}

{"<div class='section'><h2>📋 Compliance Findings</h2><table><tr><th>Check</th><th>Framework</th><th>Description</th><th>Verdict</th><th>Change</th></tr>" + comp_rows + "</table></div>" if comp_rows else ""}

{"<div class='section'><h2>⚠️ Risk Register</h2><table><tr><th>ID</th><th>Title</th><th>Severity</th><th>Change</th></tr>" + risk_rows + "</table></div>" if risk_rows else ""}

<p style="text-align:center;color:#999;margin-top:2rem">Generated by replication.safety_diff</p>
</body></html>"""


# ---------------------------------------------------------------------------
# Differ logic
# ---------------------------------------------------------------------------

_GRADE_ORDER = ["F", "D-", "D", "D+", "C-", "C", "C+", "B-", "B", "B+", "A-", "A", "A+"]

def _grade_rank(g: str) -> int:
    try:
        return _GRADE_ORDER.index(g)
    except ValueError:
        return -1


class SafetyDiff:
    """Compare two safety snapshots."""

    @staticmethod
    def from_scorecard_dicts(
        before: Dict[str, Any],
        after: Dict[str, Any],
        before_label: str = "before",
        after_label: str = "after",
    ) -> SafetyDiffResult:
        """Diff two scorecard result dicts (as produced by ScorecardResult.to_dict or JSON)."""
        result = SafetyDiffResult(
            before_label=before_label,
            after_label=after_label,
            before_overall=before.get("overall_score"),
            after_overall=after.get("overall_score"),
            before_timestamp=before.get("timestamp"),
            after_timestamp=after.get("timestamp"),
        )

        # Index dimensions by name
        before_dims = {d["name"]: d for d in before.get("dimensions", [])}
        after_dims = {d["name"]: d for d in after.get("dimensions", [])}
        all_names = sorted(set(before_dims) | set(after_dims))

        for name in all_names:
            bd = before_dims.get(name)
            ad = after_dims.get(name)
            bs = bd["score"] if bd else None
            as_ = ad["score"] if ad else None
            bg = bd.get("grade") if bd else None
            ag = ad.get("grade") if ad else None

            if bd is None:
                kind = ChangeKind.NEW
                delta = as_ or 0
            elif ad is None:
                kind = ChangeKind.RESOLVED
                delta = -(bs or 0)
            else:
                delta = (as_ or 0) - (bs or 0)
                if delta > 0.5:
                    kind = ChangeKind.IMPROVEMENT
                elif delta < -0.5:
                    kind = ChangeKind.REGRESSION
                else:
                    kind = ChangeKind.UNCHANGED

            result.dimension_changes.append(DimensionChange(
                name=name, before_score=bs, after_score=as_,
                before_grade=bg, after_grade=ag,
                kind=kind, delta=delta,
            ))

        return result

    @staticmethod
    def from_scorecard_files(
        before_path: str, after_path: str,
        before_label: str = "", after_label: str = "",
    ) -> SafetyDiffResult:
        bp = Path(before_path)
        ap = Path(after_path)
        with open(bp) as f:
            before = json.load(f)
        with open(ap) as f:
            after = json.load(f)
        return SafetyDiff.from_scorecard_dicts(
            before, after,
            before_label=before_label or bp.stem,
            after_label=after_label or ap.stem,
        )

    @staticmethod
    def from_compliance_dicts(
        before: Dict[str, Any],
        after: Dict[str, Any],
        before_label: str = "before",
        after_label: str = "after",
    ) -> SafetyDiffResult:
        """Diff two compliance result dicts."""
        result = SafetyDiffResult(before_label=before_label, after_label=after_label)

        before_findings = {f["check_id"]: f for f in before.get("findings", [])}
        after_findings = {f["check_id"]: f for f in after.get("findings", [])}
        all_ids = sorted(set(before_findings) | set(after_findings))

        for cid in all_ids:
            bf = before_findings.get(cid)
            af = after_findings.get(cid)

            if bf is None:
                kind = ChangeKind.NEW
            elif af is None:
                kind = ChangeKind.RESOLVED
            else:
                bv = bf.get("verdict", "").upper()
                av = af.get("verdict", "").upper()
                if bv == av:
                    kind = ChangeKind.UNCHANGED
                elif bv == "FAIL" and av == "PASS":
                    kind = ChangeKind.IMPROVEMENT
                elif bv == "PASS" and av == "FAIL":
                    kind = ChangeKind.REGRESSION
                elif av == "FAIL":
                    kind = ChangeKind.REGRESSION
                else:
                    kind = ChangeKind.IMPROVEMENT

            if kind == ChangeKind.UNCHANGED:
                continue

            src = af or bf
            result.compliance_changes.append(ComplianceChange(
                check_id=cid,
                framework=src.get("framework", "unknown"),
                description=src.get("description", ""),
                kind=kind,
                before_verdict=bf.get("verdict") if bf else None,
                after_verdict=af.get("verdict") if af else None,
            ))

        return result

    @staticmethod
    def from_compliance_files(
        before_path: str, after_path: str,
        before_label: str = "", after_label: str = "",
    ) -> SafetyDiffResult:
        bp = Path(before_path)
        ap = Path(after_path)
        with open(bp) as f:
            before = json.load(f)
        with open(ap) as f:
            after = json.load(f)
        return SafetyDiff.from_compliance_dicts(
            before, after,
            before_label=before_label or bp.stem,
            after_label=after_label or ap.stem,
        )

    @staticmethod
    def merge(a: SafetyDiffResult, b: SafetyDiffResult) -> SafetyDiffResult:
        """Merge two partial diffs into one combined result."""
        return SafetyDiffResult(
            dimension_changes=a.dimension_changes + b.dimension_changes,
            before_overall=a.before_overall or b.before_overall,
            after_overall=a.after_overall or b.after_overall,
            compliance_changes=a.compliance_changes + b.compliance_changes,
            risk_changes=a.risk_changes + b.risk_changes,
            before_label=a.before_label,
            after_label=a.after_label,
            before_timestamp=a.before_timestamp or b.before_timestamp,
            after_timestamp=a.after_timestamp or b.after_timestamp,
        )


# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------

def _demo_before() -> Dict[str, Any]:
    return {
        "overall_score": 62.4,
        "timestamp": "2026-03-25T10:00:00Z",
        "dimensions": [
            {"name": "Containment", "score": 75.0, "grade": "B"},
            {"name": "Oversight", "score": 60.0, "grade": "C"},
            {"name": "Kill-Switch Reliability", "score": 85.0, "grade": "A-"},
            {"name": "Resource Governance", "score": 45.0, "grade": "D+"},
            {"name": "Lineage Integrity", "score": 55.0, "grade": "C-"},
            {"name": "Behavioral Alignment", "score": 70.0, "grade": "B-"},
        ],
        "findings": [
            {"check_id": "max_workers", "framework": "nist_ai_rmf", "description": "Worker count within limit", "verdict": "PASS"},
            {"check_id": "kill_switch", "framework": "nist_ai_rmf", "description": "Kill switch functional", "verdict": "PASS"},
            {"check_id": "escape_rate", "framework": "internal", "description": "Escape rate below threshold", "verdict": "FAIL"},
            {"check_id": "budget_limit", "framework": "internal", "description": "Budget compliance", "verdict": "PASS"},
            {"check_id": "mutation_rate", "framework": "eu_ai_act", "description": "Self-modification rate", "verdict": "FAIL"},
        ],
    }


def _demo_after() -> Dict[str, Any]:
    return {
        "overall_score": 71.8,
        "timestamp": "2026-04-02T10:00:00Z",
        "dimensions": [
            {"name": "Containment", "score": 82.0, "grade": "A-"},
            {"name": "Oversight", "score": 68.0, "grade": "B-"},
            {"name": "Kill-Switch Reliability", "score": 88.0, "grade": "A"},
            {"name": "Resource Governance", "score": 52.0, "grade": "C-"},
            {"name": "Lineage Integrity", "score": 60.0, "grade": "C"},
            {"name": "Behavioral Alignment", "score": 65.0, "grade": "C+"},
            {"name": "Anomaly Detection", "score": 78.0, "grade": "B+"},
        ],
        "findings": [
            {"check_id": "max_workers", "framework": "nist_ai_rmf", "description": "Worker count within limit", "verdict": "PASS"},
            {"check_id": "kill_switch", "framework": "nist_ai_rmf", "description": "Kill switch functional", "verdict": "PASS"},
            {"check_id": "escape_rate", "framework": "internal", "description": "Escape rate below threshold", "verdict": "PASS"},
            {"check_id": "budget_limit", "framework": "internal", "description": "Budget compliance", "verdict": "FAIL"},
            {"check_id": "mutation_rate", "framework": "eu_ai_act", "description": "Self-modification rate", "verdict": "FAIL"},
            {"check_id": "data_exfil", "framework": "eu_ai_act", "description": "Data exfiltration prevention", "verdict": "FAIL"},
        ],
    }


def _run_demo() -> None:
    """Run a demo diff with synthetic data."""
    before = _demo_before()
    after = _demo_after()

    # Scorecard diff
    sc_diff = SafetyDiff.from_scorecard_dicts(before, after, "March 25 run", "April 2 run")

    # Compliance diff
    comp_diff = SafetyDiff.from_compliance_dicts(before, after, "March 25 run", "April 2 run")

    # Merge
    combined = SafetyDiff.merge(sc_diff, comp_diff)
    print(combined.render())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    args = sys.argv[1:]

    if not args or args[0] in ("--help", "-h"):
        print(__doc__)
        return

    if args[0] == "demo":
        _run_demo()
        return

    mode = args[0]  # scorecard | compliance

    if mode == "live":
        # Run two live scorecards and diff
        print("Live diffing requires the replication framework to be configured.")
        print("Use 'demo' for a demonstration with synthetic data.")
        return

    if len(args) < 3:
        print(f"Usage: python -m replication.safety_diff {mode} <before.json> <after.json> [--json] [--html FILE]")
        return

    before_path, after_path = args[1], args[2]
    output_json = "--json" in args
    html_idx = args.index("--html") if "--html" in args else -1
    html_path = args[html_idx + 1] if html_idx >= 0 and html_idx + 1 < len(args) else None

    if mode == "scorecard":
        result = SafetyDiff.from_scorecard_files(before_path, after_path)
    elif mode == "compliance":
        result = SafetyDiff.from_compliance_files(before_path, after_path)
    else:
        print(f"Unknown mode: {mode}. Use 'scorecard', 'compliance', or 'demo'.")
        return

    if output_json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.render())

    if html_path:
        Path(html_path).write_text(result.to_html(), encoding="utf-8")
        print(f"\nHTML report saved to {html_path}")


if __name__ == "__main__":
    _cli()
