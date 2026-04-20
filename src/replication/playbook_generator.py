"""Safety Playbook Generator — autonomous remediation planning.

Runs simulations across all presets, identifies weaknesses, and generates
phased improvement playbooks with prioritized actions, effort estimates,
and expected impact.

Usage (CLI)::

    python -m replication.playbook_generator                     # full playbook
    python -m replication.playbook_generator --presets balanced,aggressive
    python -m replication.playbook_generator --runs 8
    python -m replication.playbook_generator --json              # JSON output
    python -m replication.playbook_generator --html report.html  # HTML report
    python -m replication.playbook_generator --budget low        # effort filter

Programmatic::

    from replication.playbook_generator import PlaybookGenerator
    gen = PlaybookGenerator()
    playbook = gen.generate()
    print(playbook.render())
    gen.export_html(playbook, "playbook.html")
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .simulator import ScenarioConfig, SimulationReport, Simulator, Strategy, PRESETS
from ._helpers import (
    stats_mean,
    stats_std,
    box_header,
    extract_report_metrics,
    REPORT_METRIC_NAMES,
)


# ── Enums ──


class Priority(Enum):
    """Action priority level."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Phase(Enum):
    """Remediation phase."""
    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"


class Effort(Enum):
    """Estimated implementation effort."""
    TRIVIAL = "trivial"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Impact(Enum):
    """Expected safety impact."""
    MINIMAL = "minimal"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"
    TRANSFORMATIVE = "transformative"


# ── Data Classes ──


@dataclass
class Action:
    """A single remediation action."""
    title: str
    description: str
    priority: Priority
    phase: Phase
    effort: Effort
    impact: Impact
    metric: str
    current_value: float
    target_value: float
    tags: List[str] = field(default_factory=list)

    def roi_score(self) -> float:
        """Higher = better ROI (high impact, low effort)."""
        impact_map = {Impact.MINIMAL: 1, Impact.MODERATE: 2,
                      Impact.SIGNIFICANT: 3, Impact.TRANSFORMATIVE: 4}
        effort_map = {Effort.TRIVIAL: 1, Effort.LOW: 2,
                      Effort.MEDIUM: 3, Effort.HIGH: 4}
        return impact_map[self.impact] / effort_map[self.effort]


@dataclass
class PresetDiagnosis:
    """Diagnosis for a single preset."""
    preset_name: str
    metrics: Dict[str, float]
    weaknesses: List[str]
    strengths: List[str]
    risk_score: float  # 0-100


@dataclass
class Playbook:
    """Complete remediation playbook."""
    generated_at: float
    diagnoses: List[PresetDiagnosis]
    actions: List[Action]
    overall_risk: float
    summary: str
    runs_per_preset: int

    def actions_by_phase(self) -> Dict[Phase, List[Action]]:
        result: Dict[Phase, List[Action]] = {p: [] for p in Phase}
        for a in self.actions:
            result[a.phase].append(a)
        for v in result.values():
            v.sort(key=lambda x: x.roi_score(), reverse=True)
        return result

    def actions_by_priority(self) -> Dict[Priority, List[Action]]:
        result: Dict[Priority, List[Action]] = {p: [] for p in Priority}
        for a in self.actions:
            result[a.priority].append(a)
        return result

    def render(self) -> str:
        """Render playbook as formatted text."""
        lines: List[str] = []
        lines.extend(box_header("SAFETY PLAYBOOK GENERATOR"))
        lines.append("")
        lines.append(f"  Overall Risk Score: {self.overall_risk:.1f}/100")
        risk_bar = _risk_bar(self.overall_risk)
        lines.append(f"  Risk Gauge: {risk_bar}")
        lines.append(f"  Total Actions: {len(self.actions)}")
        lines.append(f"  Runs per Preset: {self.runs_per_preset}")
        lines.append("")
        lines.append(f"  {self.summary}")
        lines.append("")

        # Preset diagnoses
        lines.extend(box_header("PRESET DIAGNOSES"))
        lines.append("")
        for d in self.diagnoses:
            emoji = "🔴" if d.risk_score >= 70 else "🟡" if d.risk_score >= 40 else "🟢"
            lines.append(f"  {emoji} {d.preset_name} — Risk: {d.risk_score:.1f}/100")
            if d.weaknesses:
                for w in d.weaknesses:
                    lines.append(f"     ⚠ {w}")
            if d.strengths:
                for s in d.strengths:
                    lines.append(f"     ✓ {s}")
            lines.append("")

        # Actions by phase
        phase_labels = {
            Phase.IMMEDIATE: "🚨 IMMEDIATE (0-1 week)",
            Phase.SHORT_TERM: "📋 SHORT-TERM (1-4 weeks)",
            Phase.LONG_TERM: "🗓️  LONG-TERM (1-3 months)",
        }
        by_phase = self.actions_by_phase()
        for phase in Phase:
            acts = by_phase[phase]
            if not acts:
                continue
            lines.extend(box_header(phase_labels[phase]))
            lines.append("")
            for i, a in enumerate(acts, 1):
                pri_icon = {"critical": "🔴", "high": "🟠",
                            "medium": "🟡", "low": "🟢"}[a.priority.value]
                lines.append(f"  {i}. {pri_icon} [{a.priority.value.upper()}] {a.title}")
                lines.append(f"     {a.description}")
                lines.append(f"     Effort: {a.effort.value} | Impact: {a.impact.value} "
                             f"| ROI: {a.roi_score():.1f}")
                lines.append(f"     Metric: {a.metric} "
                             f"({a.current_value:.2f} → {a.target_value:.2f})")
                if a.tags:
                    lines.append(f"     Tags: {', '.join(a.tags)}")
                lines.append("")

        # ROI ranking
        lines.extend(box_header("TOP ROI ACTIONS"))
        lines.append("")
        top = sorted(self.actions, key=lambda x: x.roi_score(), reverse=True)[:5]
        for i, a in enumerate(top, 1):
            lines.append(f"  {i}. {a.title} (ROI: {a.roi_score():.1f}, "
                         f"effort: {a.effort.value}, impact: {a.impact.value})")
        lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "overall_risk": self.overall_risk,
            "summary": self.summary,
            "runs_per_preset": self.runs_per_preset,
            "diagnoses": [
                {
                    "preset": d.preset_name,
                    "risk_score": d.risk_score,
                    "metrics": d.metrics,
                    "weaknesses": d.weaknesses,
                    "strengths": d.strengths,
                }
                for d in self.diagnoses
            ],
            "actions": [
                {
                    "title": a.title,
                    "description": a.description,
                    "priority": a.priority.value,
                    "phase": a.phase.value,
                    "effort": a.effort.value,
                    "impact": a.impact.value,
                    "roi_score": a.roi_score(),
                    "metric": a.metric,
                    "current_value": a.current_value,
                    "target_value": a.target_value,
                    "tags": a.tags,
                }
                for a in self.actions
            ],
        }


def _risk_bar(score: float, width: int = 20) -> str:
    filled = int(score / 100 * width)
    if score >= 70:
        ch = "█"
    elif score >= 40:
        ch = "▓"
    else:
        ch = "░"
    return f"[{ch * filled}{'·' * (width - filled)}] {score:.0f}%"


# ── Thresholds & Rules ──

# (metric_name, condition, threshold, action_template)
_RULES = [
    ("denial_rate", "lt", 0.5, {
        "title": "Strengthen replication denial controls",
        "description": "Denial rate below 50% means replications succeed too easily. "
                       "Tighten contract predicates or reduce resource limits.",
        "priority": Priority.CRITICAL,
        "phase": Phase.IMMEDIATE,
        "effort": Effort.LOW,
        "impact": Impact.SIGNIFICANT,
        "target": 0.7,
        "tags": ["access-control", "contracts"],
    }),
    ("denial_rate", "lt", 0.3, {
        "title": "Emergency: minimal replication denial",
        "description": "Less than 30% denial rate is dangerously permissive. "
                       "Immediately restrict max_workers and max_depth.",
        "priority": Priority.CRITICAL,
        "phase": Phase.IMMEDIATE,
        "effort": Effort.TRIVIAL,
        "impact": Impact.TRANSFORMATIVE,
        "target": 0.6,
        "tags": ["emergency", "access-control"],
    }),
    ("max_depth_reached", "gt", 3, {
        "title": "Limit replication depth",
        "description": "Deep replication chains detected. Reduce max_depth in "
                       "contracts to prevent cascading replication.",
        "priority": Priority.HIGH,
        "phase": Phase.IMMEDIATE,
        "effort": Effort.TRIVIAL,
        "impact": Impact.SIGNIFICANT,
        "target": 2.0,
        "tags": ["depth-control", "cascading"],
    }),
    ("total_workers", "gt", 15, {
        "title": "Cap worker proliferation",
        "description": "High worker count indicates weak spawning limits. "
                       "Set stricter max_workers in contract configuration.",
        "priority": Priority.HIGH,
        "phase": Phase.SHORT_TERM,
        "effort": Effort.LOW,
        "impact": Impact.MODERATE,
        "target": 10.0,
        "tags": ["resource-control", "workers"],
    }),
    ("efficiency", "gt", 5.0, {
        "title": "Reduce task-per-worker efficiency",
        "description": "High efficiency may indicate workers completing too many "
                       "tasks before being contained. Add per-worker task limits.",
        "priority": Priority.MEDIUM,
        "phase": Phase.SHORT_TERM,
        "effort": Effort.MEDIUM,
        "impact": Impact.MODERATE,
        "target": 3.0,
        "tags": ["task-control", "containment"],
    }),
    ("total_cpu", "gt", 16.0, {
        "title": "Tighten CPU resource limits",
        "description": "Aggregate CPU usage is high. Reduce per-worker CPU allocation "
                       "to limit computational footprint.",
        "priority": Priority.MEDIUM,
        "phase": Phase.SHORT_TERM,
        "effort": Effort.LOW,
        "impact": Impact.MODERATE,
        "target": 8.0,
        "tags": ["resources", "cpu"],
    }),
    ("total_memory_mb", "gt", 4096, {
        "title": "Reduce memory allocation budget",
        "description": "Total memory exceeds 4GB. Lower per-worker memory limits or "
                       "reduce max concurrent workers.",
        "priority": Priority.MEDIUM,
        "phase": Phase.LONG_TERM,
        "effort": Effort.MEDIUM,
        "impact": Impact.MODERATE,
        "target": 2048.0,
        "tags": ["resources", "memory"],
    }),
    ("replications_succeeded", "gt", 5, {
        "title": "Audit successful replication paths",
        "description": "Multiple successful replications warrant auditing which "
                       "paths succeeded and whether they should have been denied.",
        "priority": Priority.HIGH,
        "phase": Phase.SHORT_TERM,
        "effort": Effort.MEDIUM,
        "impact": Impact.SIGNIFICANT,
        "target": 2.0,
        "tags": ["audit", "paths"],
    }),
    ("replications_denied", "lt", 2, {
        "title": "Validate denial mechanisms are active",
        "description": "Very few denials suggest safety predicates may not be "
                       "triggering. Review contract stop conditions.",
        "priority": Priority.HIGH,
        "phase": Phase.IMMEDIATE,
        "effort": Effort.LOW,
        "impact": Impact.SIGNIFICANT,
        "target": 5.0,
        "tags": ["predicates", "validation"],
    }),
    ("total_tasks", "gt", 20, {
        "title": "Implement task rate limiting",
        "description": "High task count suggests unbounded work. Add rate limiting "
                       "or task budgets per simulation window.",
        "priority": Priority.LOW,
        "phase": Phase.LONG_TERM,
        "effort": Effort.HIGH,
        "impact": Impact.MODERATE,
        "target": 10.0,
        "tags": ["rate-limiting", "budget"],
    }),
]


# ── Generator ──


class PlaybookGenerator:
    """Autonomous safety playbook generator.

    Runs simulations across presets, diagnoses weaknesses, and produces
    a phased remediation playbook with prioritized actions.
    """

    def __init__(self, runs: int = 5, presets: Optional[List[str]] = None):
        self.runs = runs
        self.presets = presets or list(PRESETS.keys())

    # ── Core ──

    def generate(self) -> Playbook:
        """Run analysis and produce a complete playbook."""
        diagnoses: List[PresetDiagnosis] = []
        all_actions: List[Action] = []
        seen_titles: set = set()

        for preset_name in self.presets:
            if preset_name not in PRESETS:
                continue
            metrics = self._run_preset(preset_name)
            diag = self._diagnose(preset_name, metrics)
            diagnoses.append(diag)

            for action in self._generate_actions(preset_name, metrics):
                if action.title not in seen_titles:
                    seen_titles.add(action.title)
                    all_actions.append(action)

        overall = stats_mean([d.risk_score for d in diagnoses]) if diagnoses else 0.0
        summary = self._generate_summary(overall, diagnoses, all_actions)

        return Playbook(
            generated_at=time.time(),
            diagnoses=diagnoses,
            actions=all_actions,
            overall_risk=overall,
            summary=summary,
            runs_per_preset=self.runs,
        )

    # ── Simulation ──

    def _run_preset(self, name: str) -> Dict[str, float]:
        """Run a preset multiple times and average metrics."""
        all_metrics: Dict[str, List[float]] = {m: [] for m in REPORT_METRIC_NAMES}
        cfg = PRESETS[name]
        for _ in range(self.runs):
            sim = Simulator(cfg)
            report = sim.run()
            metrics = extract_report_metrics(report)
            for k, v in metrics.items():
                all_metrics[k].append(v)
        return {k: stats_mean(v) for k, v in all_metrics.items()}

    # ── Diagnosis ──

    def _diagnose(self, name: str, metrics: Dict[str, float]) -> PresetDiagnosis:
        weaknesses: List[str] = []
        strengths: List[str] = []

        dr = metrics.get("denial_rate", 0)
        if dr >= 0.7:
            strengths.append(f"Strong denial rate ({dr:.0%})")
        elif dr >= 0.5:
            weaknesses.append(f"Moderate denial rate ({dr:.0%}) — could be tighter")
        else:
            weaknesses.append(f"Weak denial rate ({dr:.0%}) — too permissive")

        depth = metrics.get("max_depth_reached", 0)
        if depth <= 2:
            strengths.append(f"Depth well contained (max {depth:.0f})")
        else:
            weaknesses.append(f"Deep replication chains (depth {depth:.0f})")

        workers = metrics.get("total_workers", 0)
        if workers <= 8:
            strengths.append(f"Worker count manageable ({workers:.0f})")
        else:
            weaknesses.append(f"High worker proliferation ({workers:.0f})")

        denied = metrics.get("replications_denied", 0)
        if denied >= 5:
            strengths.append(f"Active denial enforcement ({denied:.0f} denials)")
        elif denied < 2:
            weaknesses.append(f"Very few denials ({denied:.0f}) — predicates may be inactive")

        # Risk score: weighted combination
        risk = 0.0
        risk += max(0, (1 - dr)) * 40          # low denial = high risk
        risk += min(depth / 5, 1) * 25          # deep = risky
        risk += min(workers / 20, 1) * 20       # many workers = risky
        risk += max(0, (1 - denied / 10)) * 15  # few denials = risky
        risk = min(100, max(0, risk))

        return PresetDiagnosis(
            preset_name=name,
            metrics=metrics,
            weaknesses=weaknesses,
            strengths=strengths,
            risk_score=risk,
        )

    # ── Action Generation ──

    def _generate_actions(self, preset: str, metrics: Dict[str, float]) -> List[Action]:
        actions: List[Action] = []
        for metric_name, condition, threshold, template in _RULES:
            value = metrics.get(metric_name, 0)
            triggered = False
            if condition == "lt" and value < threshold:
                triggered = True
            elif condition == "gt" and value > threshold:
                triggered = True

            if triggered:
                actions.append(Action(
                    title=template["title"],
                    description=template["description"],
                    priority=template["priority"],
                    phase=template["phase"],
                    effort=template["effort"],
                    impact=template["impact"],
                    metric=metric_name,
                    current_value=value,
                    target_value=template["target"],
                    tags=template.get("tags", []),
                ))
        return actions

    # ── Summary ──

    def _generate_summary(self, overall: float, diagnoses: List[PresetDiagnosis],
                          actions: List[Action]) -> str:
        if overall >= 70:
            urgency = "CRITICAL — immediate remediation required"
        elif overall >= 40:
            urgency = "ELEVATED — targeted improvements recommended"
        else:
            urgency = "HEALTHY — maintain current posture with monitoring"

        crit = sum(1 for a in actions if a.priority == Priority.CRITICAL)
        high = sum(1 for a in actions if a.priority == Priority.HIGH)
        parts = [f"Risk: {urgency}."]
        if crit:
            parts.append(f"{crit} critical action(s).")
        if high:
            parts.append(f"{high} high-priority action(s).")
        parts.append(f"{len(diagnoses)} preset(s) analyzed.")
        return " ".join(parts)

    # ── Export ──

    def export_html(self, playbook: Playbook, path: str) -> None:
        """Export playbook as a self-contained HTML report."""
        d = playbook.to_dict()
        phase_labels = {
            "immediate": "🚨 Immediate",
            "short_term": "📋 Short-Term",
            "long_term": "🗓️ Long-Term",
        }
        pri_colors = {
            "critical": "#dc3545",
            "high": "#fd7e14",
            "medium": "#ffc107",
            "low": "#28a745",
        }

        html_parts = [
            "<!DOCTYPE html><html><head><meta charset='utf-8'>",
            "<title>Safety Playbook</title>",
            "<style>",
            "body{font-family:system-ui;max-width:900px;margin:2em auto;padding:0 1em;"
            "background:#0d1117;color:#c9d1d9}",
            "h1,h2,h3{color:#58a6ff}",
            ".risk-gauge{height:24px;border-radius:12px;background:#21262d;overflow:hidden;"
            "margin:8px 0}",
            ".risk-fill{height:100%;border-radius:12px;transition:width .5s}",
            ".card{background:#161b22;border:1px solid #30363d;border-radius:8px;"
            "padding:16px;margin:12px 0}",
            ".tag{display:inline-block;background:#1f6feb33;color:#58a6ff;padding:2px 8px;"
            "border-radius:4px;font-size:12px;margin:2px}",
            ".action{border-left:4px solid;padding:8px 16px;margin:8px 0}",
            ".metric{font-family:monospace;color:#7ee787}",
            "table{width:100%;border-collapse:collapse}",
            "td,th{padding:6px 10px;text-align:left;border-bottom:1px solid #21262d}",
            "</style></head><body>",
            "<h1>🛡️ Safety Playbook</h1>",
        ]

        # Overall risk
        color = "#dc3545" if d["overall_risk"] >= 70 else (
            "#ffc107" if d["overall_risk"] >= 40 else "#28a745")
        html_parts.append(f"<p><strong>Overall Risk:</strong> {d['overall_risk']:.1f}/100</p>")
        html_parts.append(f"<div class='risk-gauge'><div class='risk-fill' "
                          f"style='width:{d['overall_risk']:.0f}%;background:{color}'>"
                          f"</div></div>")
        html_parts.append(f"<p>{d['summary']}</p>")

        # Diagnoses
        html_parts.append("<h2>Preset Diagnoses</h2>")
        for diag in d["diagnoses"]:
            emoji = "🔴" if diag["risk_score"] >= 70 else (
                "🟡" if diag["risk_score"] >= 40 else "🟢")
            html_parts.append(f"<div class='card'><h3>{emoji} {diag['preset']}"
                              f" — Risk: {diag['risk_score']:.1f}</h3>")
            if diag["weaknesses"]:
                html_parts.append("<ul>")
                for w in diag["weaknesses"]:
                    html_parts.append(f"<li>⚠️ {w}</li>")
                html_parts.append("</ul>")
            if diag["strengths"]:
                html_parts.append("<ul>")
                for s in diag["strengths"]:
                    html_parts.append(f"<li>✅ {s}</li>")
                html_parts.append("</ul>")
            html_parts.append("</div>")

        # Actions by phase
        html_parts.append("<h2>Remediation Actions</h2>")
        by_phase: Dict[str, list] = {"immediate": [], "short_term": [], "long_term": []}
        for a in d["actions"]:
            by_phase.setdefault(a["phase"], []).append(a)

        for phase_key in ["immediate", "short_term", "long_term"]:
            acts = by_phase.get(phase_key, [])
            if not acts:
                continue
            acts.sort(key=lambda x: x["roi_score"], reverse=True)
            html_parts.append(f"<h3>{phase_labels.get(phase_key, phase_key)}</h3>")
            for a in acts:
                c = pri_colors.get(a["priority"], "#666")
                html_parts.append(f"<div class='action' style='border-color:{c}'>")
                html_parts.append(f"<strong>[{a['priority'].upper()}] {a['title']}</strong><br>")
                html_parts.append(f"{a['description']}<br>")
                html_parts.append(f"<span class='metric'>{a['metric']}: "
                                  f"{a['current_value']:.2f} → {a['target_value']:.2f}</span><br>")
                html_parts.append(f"Effort: {a['effort']} | Impact: {a['impact']} | "
                                  f"ROI: {a['roi_score']:.1f}<br>")
                if a["tags"]:
                    for t in a["tags"]:
                        html_parts.append(f"<span class='tag'>{t}</span>")
                html_parts.append("</div>")

        html_parts.append("</body></html>")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(html_parts))


# ── CLI ──


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    args = argv if argv is not None else sys.argv[1:]
    runs = 5
    presets: Optional[List[str]] = None
    json_mode = False
    html_path: Optional[str] = None
    budget_filter: Optional[str] = None

    i = 0
    while i < len(args):
        if args[i] == "--runs" and i + 1 < len(args):
            runs = int(args[i + 1]); i += 2
        elif args[i] == "--presets" and i + 1 < len(args):
            presets = [p.strip() for p in args[i + 1].split(",")]; i += 2
        elif args[i] == "--json":
            json_mode = True; i += 1
        elif args[i] == "--html" and i + 1 < len(args):
            html_path = args[i + 1]; i += 2
        elif args[i] == "--budget" and i + 1 < len(args):
            budget_filter = args[i + 1]; i += 2
        else:
            i += 1

    gen = PlaybookGenerator(runs=runs, presets=presets)
    playbook = gen.generate()

    if budget_filter:
        effort_map = {"trivial": 0, "low": 1, "medium": 2, "high": 3}
        max_effort = effort_map.get(budget_filter, 3)
        playbook.actions = [a for a in playbook.actions
                            if effort_map.get(a.effort.value, 0) <= max_effort]

    if html_path:
        gen.export_html(playbook, html_path)
        print(f"HTML report saved to {html_path}")

    if json_mode:
        print(json.dumps(playbook.to_dict(), indent=2))
    else:
        print(playbook.render())


if __name__ == "__main__":
    main()
