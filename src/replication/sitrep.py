"""Safety SITREP — unified situational awareness report.

Fuses signals from multiple safety modules (scorecard, compliance, drift,
threat correlator, adaptive thresholds, incident forecast) into a single
operational picture with DEFCON-style threat level, sector statuses, and
proactive recommended actions.

Usage (CLI)::

    python -m replication sitrep                    # full SITREP
    python -m replication sitrep --json             # JSON output
    python -m replication sitrep --sectors          # list available sectors
    python -m replication sitrep --watch 30         # refresh every 30s
    python -m replication sitrep --html -o sitrep.html  # interactive HTML

Programmatic::

    from replication.sitrep import SituationalAwareness
    sa = SituationalAwareness()
    report = sa.generate()
    print(report.render())
    print(f"Threat Level: {report.threat_level}")
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ._helpers import box_header


# ── Threat Levels ────────────────────────────────────────────────────

THREAT_LEVELS = [
    ("SAFE",     "🟢", "All systems nominal. No active threats detected."),
    ("GUARDED",  "🔵", "Low-level anomalies. Heightened monitoring advised."),
    ("ELEVATED", "🟡", "Moderate risk. Active drift or compliance gaps detected."),
    ("HIGH",     "🟠", "Significant threats. Immediate review recommended."),
    ("CRITICAL", "🔴", "Severe safety posture degradation. Immediate action required."),
]


# ── Data Classes ─────────────────────────────────────────────────────

@dataclass
class SectorStatus:
    """Status of one safety sector (e.g., compliance, drift)."""
    name: str
    status: str           # "green", "yellow", "red", "unknown"
    score: Optional[float] = None   # 0-100
    summary: str = ""
    findings: List[str] = field(default_factory=list)
    source_module: str = ""

    @property
    def icon(self) -> str:
        return {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(self.status, "⚪")


@dataclass
class RecommendedAction:
    """A proactive recommended action based on current posture."""
    priority: str     # "critical", "high", "medium", "low"
    sector: str
    action: str
    rationale: str

    @property
    def icon(self) -> str:
        return {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}.get(
            self.priority, "⚪"
        )


@dataclass
class SitrepReport:
    """Complete situational awareness report."""
    timestamp: str = ""
    threat_level: str = "SAFE"
    threat_icon: str = "🟢"
    threat_description: str = ""
    overall_score: float = 100.0
    sectors: List[SectorStatus] = field(default_factory=list)
    actions: List[RecommendedAction] = field(default_factory=list)
    forecast_summary: str = ""
    generation_time_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "threat_level": self.threat_level,
            "overall_score": round(self.overall_score, 1),
            "sectors": [
                {
                    "name": s.name, "status": s.status,
                    "score": s.score, "summary": s.summary,
                    "findings": s.findings, "source": s.source_module,
                }
                for s in self.sectors
            ],
            "actions": [
                {
                    "priority": a.priority, "sector": a.sector,
                    "action": a.action, "rationale": a.rationale,
                }
                for a in self.actions
            ],
            "forecast_summary": self.forecast_summary,
            "generation_time_s": round(self.generation_time_s, 3),
        }

    def render(self) -> str:
        """Render human-readable SITREP."""
        lines = box_header("SAFETY SITREP — Situational Awareness Report")

        lines.append(f"  Generated: {self.timestamp}")
        lines.append(f"  Threat Level: {self.threat_icon}  {self.threat_level}")
        lines.append(f"  Description:  {self.threat_description}")
        lines.append(f"  Overall Score: {self.overall_score:.1f}/100")
        lines.append("")

        # Sector overview
        lines.append("  ── Sector Status ──────────────────────────────────")
        if not self.sectors:
            lines.append("  (no sectors evaluated)")
        for s in self.sectors:
            score_str = f" [{s.score:.0f}/100]" if s.score is not None else ""
            lines.append(f"  {s.icon} {s.name:<22}{score_str}  {s.summary}")
            for f in s.findings[:3]:
                lines.append(f"       ↳ {f}")
        lines.append("")

        # Forecast
        if self.forecast_summary:
            lines.append("  ── Forecast ───────────────────────────────────────")
            lines.append(f"  {self.forecast_summary}")
            lines.append("")

        # Recommended actions
        lines.append("  ── Recommended Actions ────────────────────────────")
        if not self.actions:
            lines.append("  ✅ No actions required at this time.")
        for i, a in enumerate(self.actions, 1):
            lines.append(f"  {a.icon} [{a.priority.upper()}] {a.action}")
            lines.append(f"       Sector: {a.sector} | {a.rationale}")
        lines.append("")

        lines.append(f"  ⏱  Report generated in {self.generation_time_s:.2f}s")
        lines.append("─" * 57)
        return "\n".join(lines)


# ── Sector Evaluators ────────────────────────────────────────────────

class SituationalAwareness:
    """Fuse multiple safety modules into a unified SITREP."""

    def __init__(self) -> None:
        self._sectors: List[SectorStatus] = []
        self._actions: List[RecommendedAction] = []

    def generate(self) -> SitrepReport:
        """Run all sector evaluations and produce the report."""
        t0 = time.monotonic()
        self._sectors = []
        self._actions = []

        self._eval_scorecard()
        self._eval_compliance()
        self._eval_drift()
        self._eval_threats()
        self._eval_thresholds()
        forecast = self._eval_forecast()

        # Compute overall score (average of scored sectors)
        scored = [s.score for s in self._sectors if s.score is not None]
        overall = sum(scored) / len(scored) if scored else 50.0

        # Determine threat level
        level_idx = self._compute_threat_level(overall, self._sectors)
        level_name, level_icon, level_desc = THREAT_LEVELS[level_idx]

        # Sort actions by priority
        prio_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        self._actions.sort(key=lambda a: prio_order.get(a.priority, 9))

        elapsed = time.monotonic() - t0

        return SitrepReport(
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            threat_level=level_name,
            threat_icon=level_icon,
            threat_description=level_desc,
            overall_score=overall,
            sectors=self._sectors,
            actions=self._actions,
            forecast_summary=forecast,
            generation_time_s=elapsed,
        )

    def _compute_threat_level(
        self, overall: float, sectors: List[SectorStatus]
    ) -> int:
        """Map overall score + sector statuses → threat level index (0-4)."""
        red_count = sum(1 for s in sectors if s.status == "red")
        yellow_count = sum(1 for s in sectors if s.status == "yellow")

        if overall >= 90 and red_count == 0 and yellow_count == 0:
            return 0  # SAFE
        if overall >= 75 and red_count == 0:
            return 1  # GUARDED
        if overall >= 55 and red_count <= 1:
            return 2  # ELEVATED
        if overall >= 35 or red_count <= 2:
            return 3  # HIGH
        return 4  # CRITICAL

    # ── Individual sector evaluators ──────────────────────────────

    def _eval_scorecard(self) -> None:
        """Evaluate safety scorecard sector."""
        try:
            from .scorecard import SafetyScorecard
            sc = SafetyScorecard()
            # evaluate() may require a scenario arg in some versions
            import inspect
            sig = inspect.signature(sc.evaluate)
            params = [p for p in sig.parameters.values()
                      if p.default is inspect.Parameter.empty
                      and p.name != 'self']
            if params:
                # Need args we don't have — skip gracefully
                raise RuntimeError("Scorecard requires simulation data")
            result = sc.evaluate()
            score = result.overall_score
            findings: List[str] = []
            weak = [d for d in result.dimensions if d.score < 60]
            for d in weak[:3]:
                findings.append(f"{d.name}: {d.score:.0f}/100 — {d.assessment}")

            status = "green" if score >= 80 else ("yellow" if score >= 55 else "red")
            self._sectors.append(SectorStatus(
                name="Safety Scorecard",
                status=status,
                score=score,
                summary=f"{result.grade} grade, {len(weak)} weak dimensions",
                findings=findings,
                source_module="scorecard",
            ))
            if score < 55:
                self._actions.append(RecommendedAction(
                    priority="high",
                    sector="Safety Scorecard",
                    action="Address weak safety dimensions immediately",
                    rationale=f"Overall score {score:.0f}/100 is below minimum threshold",
                ))
            elif score < 75:
                self._actions.append(RecommendedAction(
                    priority="medium",
                    sector="Safety Scorecard",
                    action="Review and improve flagged dimensions",
                    rationale=f"Score {score:.0f}/100 indicates room for improvement",
                ))
        except Exception as exc:
            self._sectors.append(SectorStatus(
                name="Safety Scorecard", status="unknown",
                summary=f"Evaluation error: {exc}",
                source_module="scorecard",
            ))

    def _eval_compliance(self) -> None:
        """Evaluate compliance sector."""
        try:
            from .compliance import ComplianceAuditor
            auditor = ComplianceAuditor()
            import inspect
            sig = inspect.signature(auditor.audit)
            params = [p for p in sig.parameters.values()
                      if p.default is inspect.Parameter.empty
                      and p.name != 'self']
            if params:
                raise RuntimeError("Compliance audit requires simulation data")
            result = auditor.audit()
            total = len(result.findings)
            critical = sum(
                1 for f in result.findings
                if getattr(f, "severity", "medium") in ("critical", "high")
            )
            score = max(0.0, 100.0 - critical * 15 - (total - critical) * 5)
            status = "green" if critical == 0 and total <= 2 else (
                "yellow" if critical == 0 else "red"
            )
            findings = [
                f"{getattr(f, 'control_id', '?')}: {getattr(f, 'title', str(f))}"
                for f in result.findings[:3]
            ]
            self._sectors.append(SectorStatus(
                name="Compliance",
                status=status,
                score=score,
                summary=f"{total} findings ({critical} critical)",
                findings=findings,
                source_module="compliance",
            ))
            if critical > 0:
                self._actions.append(RecommendedAction(
                    priority="critical",
                    sector="Compliance",
                    action=f"Remediate {critical} critical compliance finding(s)",
                    rationale="Critical compliance gaps pose immediate risk",
                ))
        except Exception as exc:
            self._sectors.append(SectorStatus(
                name="Compliance", status="unknown",
                summary=f"Evaluation error: {exc}",
                source_module="compliance",
            ))

    def _eval_drift(self) -> None:
        """Evaluate behavioral drift sector."""
        try:
            from .drift import DriftDetector
            detector = DriftDetector()
            report = detector.analyze()
            drifting = getattr(report, "drifting", False)
            drift_score_raw = getattr(report, "drift_score", 0.0)
            score = max(0.0, 100.0 - drift_score_raw * 100)
            status = "red" if drifting else ("yellow" if drift_score_raw > 0.3 else "green")
            summary_parts = []
            if drifting:
                summary_parts.append("Active drift detected")
            summary_parts.append(f"drift_score={drift_score_raw:.2f}")
            self._sectors.append(SectorStatus(
                name="Behavioral Drift",
                status=status,
                score=score,
                summary=", ".join(summary_parts),
                source_module="drift",
            ))
            if drifting:
                self._actions.append(RecommendedAction(
                    priority="high",
                    sector="Behavioral Drift",
                    action="Investigate active behavioral drift",
                    rationale="Agent behavior deviating from baseline profile",
                ))
        except Exception as exc:
            self._sectors.append(SectorStatus(
                name="Behavioral Drift", status="unknown",
                summary=f"Evaluation error: {exc}",
                source_module="drift",
            ))

    def _eval_threats(self) -> None:
        """Evaluate threat landscape sector."""
        try:
            from .threat_correlator import ThreatCorrelator
            tc = ThreatCorrelator()
            import inspect
            sig = inspect.signature(tc.correlate)
            params = [p for p in sig.parameters.values()
                      if p.default is inspect.Parameter.empty
                      and p.name != 'self']
            if params:
                raise RuntimeError("Threat correlator requires simulation data")
            result = tc.correlate()
            correlated = getattr(result, "correlated_threats", [])
            high_sev = [
                t for t in correlated
                if getattr(t, "severity", "low") in ("critical", "high")
            ]
            score = max(0.0, 100.0 - len(high_sev) * 20 - len(correlated) * 5)
            status = "red" if high_sev else ("yellow" if correlated else "green")
            findings = [
                f"{getattr(t, 'name', str(t))} (severity: {getattr(t, 'severity', '?')})"
                for t in high_sev[:3]
            ]
            self._sectors.append(SectorStatus(
                name="Threat Landscape",
                status=status,
                score=score,
                summary=f"{len(correlated)} correlated signals, {len(high_sev)} high severity",
                findings=findings,
                source_module="threat_correlator",
            ))
            if high_sev:
                self._actions.append(RecommendedAction(
                    priority="critical",
                    sector="Threat Landscape",
                    action=f"Investigate {len(high_sev)} high-severity correlated threat(s)",
                    rationale="Cross-module signals indicate coordinated threat activity",
                ))
        except Exception as exc:
            self._sectors.append(SectorStatus(
                name="Threat Landscape", status="unknown",
                summary=f"Evaluation error: {exc}",
                source_module="threat_correlator",
            ))

    def _eval_thresholds(self) -> None:
        """Evaluate adaptive thresholds sector."""
        try:
            from .adaptive_thresholds import ThresholdProfile, load_preset
            profile = load_preset("fleet")
            # No live data to evaluate — report sector as nominal
            self._sectors.append(SectorStatus(
                name="Adaptive Thresholds",
                status="green",
                score=90.0,
                summary="Thresholds configured (no live metric stream)",
                source_module="adaptive_thresholds",
            ))
        except Exception as exc:
            self._sectors.append(SectorStatus(
                name="Adaptive Thresholds", status="unknown",
                summary=f"Evaluation error: {exc}",
                source_module="adaptive_thresholds",
            ))

    def _eval_forecast(self) -> str:
        """Evaluate incident forecast and return summary string."""
        try:
            from .incident_forecast import IncidentForecaster
            fc = IncidentForecaster()
            import inspect
            sig = inspect.signature(fc.forecast)
            params = [p for p in sig.parameters.values()
                      if p.default is inspect.Parameter.empty
                      and p.name != 'self']
            if params:
                raise RuntimeError("Forecaster requires historical data")
            result = fc.forecast()
            next_incidents = getattr(result, "predicted_next_7d", None)
            trend = getattr(result, "trend", "stable")
            if next_incidents is not None:
                summary = f"📊 7-day incident forecast: ~{next_incidents:.1f} expected (trend: {trend})"
            else:
                summary = f"📊 Forecast trend: {trend}"
            if trend in ("rising", "accelerating"):
                self._actions.append(RecommendedAction(
                    priority="medium",
                    sector="Forecast",
                    action="Increase monitoring — incident trend is rising",
                    rationale=f"Forecast indicates {trend} incident trajectory",
                ))
            return summary
        except Exception:
            return "📊 Forecast unavailable"


# ── HTML Report ──────────────────────────────────────────────────────

def render_html(report: SitrepReport) -> str:
    """Render an interactive HTML SITREP dashboard."""
    e = html_mod.escape
    sectors_json = json.dumps([
        {"name": s.name, "status": s.status, "score": s.score,
         "summary": s.summary, "findings": s.findings}
        for s in report.sectors
    ])
    actions_json = json.dumps([
        {"priority": a.priority, "sector": a.sector,
         "action": a.action, "rationale": a.rationale}
        for a in report.actions
    ])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Safety SITREP — {e(report.threat_level)}</title>
<style>
:root {{
  --bg: #0a0e17; --card: #131a2b; --border: #1e2a42;
  --text: #c8d6e5; --text-dim: #6b7d99; --accent: #3498db;
  --green: #2ecc71; --yellow: #f1c40f; --red: #e74c3c;
  --orange: #e67e22; --blue: #3498db;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: 'Courier New', monospace; background: var(--bg);
  color: var(--text); min-height: 100vh; padding: 20px;
}}
.container {{ max-width: 1000px; margin: 0 auto; }}
h1 {{ color: #fff; font-size: 1.6em; margin-bottom: 4px; }}
.subtitle {{ color: var(--text-dim); font-size: 0.85em; margin-bottom: 20px; }}
.threat-banner {{
  background: var(--card); border: 2px solid var(--border);
  border-radius: 8px; padding: 20px; margin-bottom: 20px;
  text-align: center;
}}
.threat-banner.level-SAFE {{ border-color: var(--green); }}
.threat-banner.level-GUARDED {{ border-color: var(--blue); }}
.threat-banner.level-ELEVATED {{ border-color: var(--yellow); }}
.threat-banner.level-HIGH {{ border-color: var(--orange); }}
.threat-banner.level-CRITICAL {{ border-color: var(--red); animation: pulse 1.5s infinite; }}
@keyframes pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.7; }} }}
.threat-level {{ font-size: 2.2em; font-weight: bold; }}
.threat-desc {{ color: var(--text-dim); margin-top: 8px; }}
.score-ring {{
  width: 120px; height: 120px; margin: 15px auto;
  position: relative;
}}
.score-ring svg {{ width: 100%; height: 100%; transform: rotate(-90deg); }}
.score-ring circle {{ fill: none; stroke-width: 8; }}
.score-ring .bg {{ stroke: var(--border); }}
.score-ring .fg {{ stroke-linecap: round; transition: stroke-dashoffset 1s ease; }}
.score-value {{
  position: absolute; top: 50%; left: 50%;
  transform: translate(-50%, -50%); font-size: 1.4em; font-weight: bold;
}}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 15px; margin-bottom: 20px; }}
.sector-card {{
  background: var(--card); border: 1px solid var(--border);
  border-radius: 8px; padding: 15px; border-left: 4px solid var(--border);
}}
.sector-card.green {{ border-left-color: var(--green); }}
.sector-card.yellow {{ border-left-color: var(--yellow); }}
.sector-card.red {{ border-left-color: var(--red); }}
.sector-name {{ font-weight: bold; color: #fff; }}
.sector-score {{ float: right; font-weight: bold; }}
.sector-summary {{ color: var(--text-dim); font-size: 0.85em; margin-top: 5px; }}
.finding {{ color: var(--text-dim); font-size: 0.8em; padding-left: 12px; border-left: 2px solid var(--border); margin-top: 4px; }}
.actions-section {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 15px; margin-bottom: 20px; }}
.actions-section h2 {{ font-size: 1.1em; color: #fff; margin-bottom: 10px; }}
.action-item {{ padding: 10px; border-bottom: 1px solid var(--border); }}
.action-item:last-child {{ border-bottom: none; }}
.action-priority {{
  display: inline-block; padding: 2px 8px; border-radius: 4px;
  font-size: 0.75em; font-weight: bold; text-transform: uppercase;
}}
.action-priority.critical {{ background: var(--red); color: #fff; }}
.action-priority.high {{ background: var(--orange); color: #fff; }}
.action-priority.medium {{ background: var(--yellow); color: #000; }}
.action-priority.low {{ background: var(--blue); color: #fff; }}
.action-text {{ margin-top: 4px; }}
.action-rationale {{ color: var(--text-dim); font-size: 0.8em; }}
.forecast {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 15px; margin-bottom: 20px; }}
.footer {{ text-align: center; color: var(--text-dim); font-size: 0.8em; margin-top: 20px; }}
</style>
</head>
<body>
<div class="container">
  <h1>🛡️ SAFETY SITREP</h1>
  <p class="subtitle">Unified Situational Awareness Report — {e(report.timestamp)}</p>

  <div class="threat-banner level-{e(report.threat_level)}">
    <div class="threat-level">{e(report.threat_icon)} {e(report.threat_level)}</div>
    <div class="score-ring">
      <svg viewBox="0 0 120 120">
        <circle class="bg" cx="60" cy="60" r="52"/>
        <circle class="fg" cx="60" cy="60" r="52"
          stroke="{_score_color(report.overall_score)}"
          stroke-dasharray="{2 * 3.14159 * 52:.0f}"
          stroke-dashoffset="{2 * 3.14159 * 52 * (1 - report.overall_score / 100):.0f}"/>
      </svg>
      <div class="score-value">{report.overall_score:.0f}</div>
    </div>
    <div class="threat-desc">{e(report.threat_description)}</div>
  </div>

  <h2 style="color:#fff; margin-bottom:10px;">📡 Sector Status</h2>
  <div class="grid" id="sectors"></div>

  <div class="forecast">
    <h2 style="color:#fff; margin-bottom:8px;">🔮 Forecast</h2>
    <p>{e(report.forecast_summary)}</p>
  </div>

  <div class="actions-section">
    <h2>⚡ Recommended Actions</h2>
    <div id="actions"></div>
  </div>

  <div class="footer">
    Generated in {report.generation_time_s:.2f}s by AI Replication Sandbox — Safety SITREP Module
  </div>
</div>
<script>
const sectors = {sectors_json};
const actions = {actions_json};

const grid = document.getElementById('sectors');
sectors.forEach(s => {{
  const score = s.score !== null ? `<span class="sector-score">${{s.score.toFixed(0)}}/100</span>` : '';
  const findings = (s.findings || []).map(f => `<div class="finding">${{esc(f)}}</div>`).join('');
  grid.innerHTML += `<div class="sector-card ${{s.status}}">
    <div class="sector-name">${{esc(s.name)}}${{score}}</div>
    <div class="sector-summary">${{esc(s.summary)}}</div>
    ${{findings}}
  </div>`;
}});

const actDiv = document.getElementById('actions');
if (actions.length === 0) {{
  actDiv.innerHTML = '<p style="color:var(--green);">✅ No actions required at this time.</p>';
}} else {{
  actions.forEach(a => {{
    actDiv.innerHTML += `<div class="action-item">
      <span class="action-priority ${{a.priority}}">${{a.priority}}</span>
      <span style="color:var(--text-dim);font-size:0.8em;margin-left:8px;">${{esc(a.sector)}}</span>
      <div class="action-text">${{esc(a.action)}}</div>
      <div class="action-rationale">${{esc(a.rationale)}}</div>
    </div>`;
  }});
}}

function esc(s) {{ const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }}
</script>
</body>
</html>"""


def _score_color(score: float) -> str:
    if score >= 80:
        return "#2ecc71"
    if score >= 55:
        return "#f1c40f"
    return "#e74c3c"


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: Optional[list] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Safety SITREP — unified situational awareness report"
    )
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--html", action="store_true", help="Generate interactive HTML")
    parser.add_argument("-o", "--output", help="Write output to file")
    parser.add_argument(
        "--sectors", action="store_true", help="List available sectors"
    )
    parser.add_argument(
        "--watch", type=int, metavar="SEC",
        help="Continuous refresh every SEC seconds",
    )
    args = parser.parse_args(argv)

    if args.sectors:
        print("Available SITREP sectors:")
        for name in [
            "Safety Scorecard", "Compliance", "Behavioral Drift",
            "Threat Landscape", "Adaptive Thresholds", "Incident Forecast",
        ]:
            print(f"  • {name}")
        return

    sa = SituationalAwareness()

    def _run_once() -> None:
        report = sa.generate()
        if args.html:
            content = render_html(report)
            if args.output:
                with open(args.output, "w", encoding="utf-8") as fh:
                    fh.write(content)
                print(f"HTML SITREP written to {args.output}")
            else:
                print(content)
        elif args.json:
            content = json.dumps(report.to_dict(), indent=2)
            if args.output:
                with open(args.output, "w", encoding="utf-8") as fh:
                    fh.write(content)
                print(f"JSON SITREP written to {args.output}")
            else:
                print(content)
        else:
            text = report.render()
            if args.output:
                with open(args.output, "w", encoding="utf-8") as fh:
                    fh.write(text)
                print(f"SITREP written to {args.output}")
            else:
                print(text)

    if args.watch:
        try:
            while True:
                print("\033[2J\033[H", end="")  # clear screen
                _run_once()
                print(f"\n  [Refreshing in {args.watch}s — Ctrl+C to stop]")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nSITREP watch stopped.")
    else:
        _run_once()


if __name__ == "__main__":
    main()
