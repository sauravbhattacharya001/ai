"""Safety Maturity Model — assess AI safety maturity across dimensions.

Evaluates an organization's AI safety posture across 8 dimensions on a
5-level maturity scale (Initial → Managed → Defined → Measured → Optimizing).
Produces per-dimension scores, an overall maturity level, gap analysis,
and prioritized recommendations for leveling up.

Usage (CLI)::

    python -m replication maturity                        # interactive assessment
    python -m replication maturity --auto                 # auto-assess from config
    python -m replication maturity --json                 # JSON output
    python -m replication maturity --html -o maturity.html  # HTML report
    python -m replication maturity --dimension governance  # single dimension
    python -m replication maturity --target 4             # show gaps to level 4

Programmatic::

    from replication.maturity_model import MaturityAssessor, MaturityConfig
    assessor = MaturityAssessor()
    result = assessor.assess(MaturityConfig(target_level=4))
    print(result.render())
    print(f"Overall: Level {result.overall_level} - {result.overall_label}")
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ._helpers import box_header as _box_header


# ── Maturity Levels ──────────────────────────────────────────────────

MATURITY_LEVELS = {
    1: "Initial",
    2: "Managed",
    3: "Defined",
    4: "Measured",
    5: "Optimizing",
}

LEVEL_DESCRIPTIONS = {
    1: "Ad-hoc, reactive. Safety practices are informal and inconsistent.",
    2: "Basic controls exist. Safety is project-level, not organization-wide.",
    3: "Standardized processes. Safety is documented and consistently applied.",
    4: "Quantitative management. Safety metrics drive decisions.",
    5: "Continuous improvement. Safety culture is proactive and self-improving.",
}

LEVEL_COLORS = {1: "#e74c3c", 2: "#e67e22", 3: "#f1c40f", 4: "#2ecc71", 5: "#27ae60"}


# ── Dimensions ───────────────────────────────────────────────────────

@dataclass
class Criterion:
    """A single maturity criterion within a dimension."""
    name: str
    description: str
    level: int  # minimum maturity level this criterion represents
    weight: float = 1.0

@dataclass
class Dimension:
    """A maturity assessment dimension."""
    name: str
    key: str
    description: str
    criteria: List[Criterion] = field(default_factory=list)

DIMENSIONS: List[Dimension] = [
    Dimension(
        name="Governance",
        key="governance",
        description="Leadership commitment, policies, roles, and accountability structures",
        criteria=[
            Criterion("Safety awareness", "Team knows basic AI safety concepts", 1),
            Criterion("Safety policy", "Written safety policy exists", 2),
            Criterion("Dedicated roles", "Safety-specific roles (officer/team) are assigned", 3),
            Criterion("Board oversight", "Executive/board-level safety governance", 4),
            Criterion("Continuous review", "Policy reviewed and updated on regular cadence", 5),
        ],
    ),
    Dimension(
        name="Risk Management",
        key="risk_management",
        description="Identification, assessment, and mitigation of AI safety risks",
        criteria=[
            Criterion("Risk awareness", "Known risks are informally tracked", 1),
            Criterion("Risk register", "Formal risk register with ownership", 2),
            Criterion("Risk framework", "Standardized risk assessment methodology", 3),
            Criterion("Quantitative risk", "Probabilistic risk scoring with metrics", 4),
            Criterion("Predictive risk", "Proactive risk prediction and pre-emptive mitigation", 5),
        ],
    ),
    Dimension(
        name="Monitoring & Detection",
        key="monitoring",
        description="Real-time monitoring, anomaly detection, and alerting capabilities",
        criteria=[
            Criterion("Basic logging", "System logs exist but aren't reviewed", 1),
            Criterion("Active monitoring", "Logs are reviewed; basic alerts exist", 2),
            Criterion("Anomaly detection", "Automated anomaly detection is deployed", 3),
            Criterion("Metrics dashboard", "Real-time safety metrics dashboards", 4),
            Criterion("Predictive alerts", "ML-driven predictive alerting with auto-response", 5),
        ],
    ),
    Dimension(
        name="Incident Response",
        key="incident_response",
        description="Preparedness, response, and recovery from safety incidents",
        criteria=[
            Criterion("Ad-hoc response", "Incidents handled reactively as they arise", 1),
            Criterion("Response plan", "Documented incident response plan exists", 2),
            Criterion("Drills", "Regular incident response drills conducted", 3),
            Criterion("Post-mortems", "Formal post-incident reviews with metrics", 4),
            Criterion("Automated response", "Automated containment and recovery procedures", 5),
        ],
    ),
    Dimension(
        name="Testing & Validation",
        key="testing",
        description="Safety testing, red-teaming, and validation practices",
        criteria=[
            Criterion("Manual testing", "Some manual safety checks before deployment", 1),
            Criterion("Test suites", "Dedicated safety test suites exist", 2),
            Criterion("Red-teaming", "Regular red-team exercises conducted", 3),
            Criterion("Continuous testing", "Automated safety testing in CI/CD pipeline", 4),
            Criterion("Adversarial ML", "Advanced adversarial testing with evolving attacks", 5),
        ],
    ),
    Dimension(
        name="Transparency & Explainability",
        key="transparency",
        description="Model interpretability, decision traceability, and documentation",
        criteria=[
            Criterion("Basic docs", "High-level model documentation exists", 1),
            Criterion("Model cards", "Standardized model cards/datasheets", 2),
            Criterion("Explainability tools", "Explainability tooling integrated", 3),
            Criterion("Audit trails", "Full decision audit trails maintained", 4),
            Criterion("Public reporting", "External transparency reports published", 5),
        ],
    ),
    Dimension(
        name="Ethics & Fairness",
        key="ethics",
        description="Bias detection, fairness testing, ethical review processes",
        criteria=[
            Criterion("Awareness", "Team aware of bias/fairness issues", 1),
            Criterion("Bias testing", "Bias testing on key demographics", 2),
            Criterion("Ethics review", "Ethics review board/process for new models", 3),
            Criterion("Fairness metrics", "Quantitative fairness metrics tracked", 4),
            Criterion("Proactive equity", "Proactive equity analysis and continuous improvement", 5),
        ],
    ),
    Dimension(
        name="Compliance & Standards",
        key="compliance",
        description="Adherence to regulations, standards, and industry frameworks",
        criteria=[
            Criterion("Awareness", "Aware of relevant regulations", 1),
            Criterion("Gap analysis", "Compliance gaps identified and documented", 2),
            Criterion("Framework adoption", "Adopted a recognized framework (NIST, ISO, EU AI Act)", 3),
            Criterion("Audit-ready", "Can pass external compliance audits", 4),
            Criterion("Leading practice", "Contributing to industry standards development", 5),
        ],
    ),
]

DIMENSION_MAP: Dict[str, Dimension] = {d.key: d for d in DIMENSIONS}


# ── Config & Results ─────────────────────────────────────────────────

@dataclass
class MaturityConfig:
    """Configuration for a maturity assessment."""
    target_level: int = 3
    dimension_filter: Optional[str] = None  # assess single dimension
    auto_assess: bool = False
    scores: Optional[Dict[str, Dict[str, bool]]] = None  # manual overrides

@dataclass
class DimensionResult:
    """Assessment result for a single dimension."""
    dimension: Dimension
    criteria_met: Dict[str, bool]
    level: int
    score: float  # 0-100
    gaps: List[str]
    recommendations: List[str]

@dataclass
class MaturityResult:
    """Complete maturity assessment result."""
    dimensions: List[DimensionResult]
    overall_level: int
    overall_label: str
    overall_score: float
    target_level: int
    timestamp: str
    gap_summary: List[Tuple[str, int, int]]  # (dim_name, current, target)
    priority_actions: List[str]

    def render(self) -> str:
        """Render as formatted text report."""
        lines: List[str] = []
        lines.extend(_box_header("SAFETY MATURITY ASSESSMENT"))
        lines.append("")
        lines.append(f"  Overall Maturity: Level {self.overall_level} — {self.overall_label}")
        lines.append(f"  Overall Score:    {self.overall_score:.1f}/100")
        lines.append(f"  Target Level:     {self.target_level}")
        lines.append(f"  Assessed:         {self.timestamp}")
        lines.append("")

        # Dimension summary
        lines.append("  ┌─────────────────────────────┬───────┬────────┬─────────────────┐")
        lines.append("  │ Dimension                   │ Level │ Score  │ Status          │")
        lines.append("  ├─────────────────────────────┼───────┼────────┼─────────────────┤")
        for dr in self.dimensions:
            status = "✓ On target" if dr.level >= self.target_level else f"↑ Gap: {self.target_level - dr.level} level(s)"
            bar = "█" * dr.level + "░" * (5 - dr.level)
            lines.append(f"  │ {dr.dimension.name:<27} │ {bar} │ {dr.score:5.1f}% │ {status:<15} │")
        lines.append("  └─────────────────────────────┴───────┴────────┴─────────────────┘")
        lines.append("")

        # Gaps
        gaps = [g for g in self.gap_summary if g[1] < g[2]]
        if gaps:
            lines.append("  GAP ANALYSIS")
            lines.append("  " + "─" * 50)
            for dim_name, current, target in sorted(gaps, key=lambda g: g[1]):
                lines.append(f"    • {dim_name}: Level {current} → Level {target} (gap: {target - current})")
            lines.append("")

        # Priority actions
        if self.priority_actions:
            lines.append("  PRIORITY ACTIONS")
            lines.append("  " + "─" * 50)
            for i, action in enumerate(self.priority_actions[:10], 1):
                lines.append(f"    {i}. {action}")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "overall_level": self.overall_level,
            "overall_label": self.overall_label,
            "overall_score": round(self.overall_score, 1),
            "target_level": self.target_level,
            "timestamp": self.timestamp,
            "dimensions": [
                {
                    "name": dr.dimension.name,
                    "key": dr.dimension.key,
                    "level": dr.level,
                    "score": round(dr.score, 1),
                    "criteria_met": dr.criteria_met,
                    "gaps": dr.gaps,
                    "recommendations": dr.recommendations,
                }
                for dr in self.dimensions
            ],
            "gap_summary": [
                {"dimension": n, "current": c, "target": t}
                for n, c, t in self.gap_summary
            ],
            "priority_actions": self.priority_actions,
        }

    def to_html(self) -> str:
        """Render as a self-contained HTML report."""
        dims_json = json.dumps([
            {"name": dr.dimension.name, "key": dr.dimension.key,
             "level": dr.level, "score": dr.score,
             "gaps": dr.gaps, "recommendations": dr.recommendations}
            for dr in self.dimensions
        ])
        level_colors = json.dumps(LEVEL_COLORS)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Safety Maturity Assessment</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f172a; color: #e2e8f0; padding: 2rem; }}
  .header {{ text-align: center; margin-bottom: 2rem; }}
  .header h1 {{ font-size: 1.8rem; color: #38bdf8; }}
  .header .overall {{ font-size: 3rem; font-weight: bold; margin: 0.5rem 0; }}
  .header .sublabel {{ color: #94a3b8; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1rem; max-width: 1200px; margin: 0 auto; }}
  .card {{ background: #1e293b; border-radius: 12px; padding: 1.2rem; border: 1px solid #334155; }}
  .card h3 {{ color: #38bdf8; margin-bottom: 0.5rem; }}
  .level-bar {{ display: flex; gap: 4px; margin: 0.5rem 0; }}
  .level-bar .seg {{ width: 40px; height: 12px; border-radius: 3px; background: #334155; }}
  .level-bar .seg.filled {{ background: var(--color); }}
  .score {{ font-size: 1.5rem; font-weight: bold; color: #f8fafc; }}
  .gaps {{ margin-top: 0.8rem; }}
  .gaps li {{ color: #fbbf24; font-size: 0.85rem; margin: 0.2rem 0; list-style: none; }}
  .gaps li::before {{ content: "⚠ "; }}
  .recs {{ margin-top: 0.5rem; }}
  .recs li {{ color: #34d399; font-size: 0.85rem; margin: 0.2rem 0; list-style: none; }}
  .recs li::before {{ content: "→ "; }}
  .summary {{ max-width: 1200px; margin: 2rem auto 0; background: #1e293b; border-radius: 12px; padding: 1.5rem; border: 1px solid #334155; }}
  .summary h2 {{ color: #38bdf8; margin-bottom: 1rem; }}
  .radar-wrap {{ text-align: center; margin: 2rem auto; max-width: 500px; }}
  canvas {{ max-width: 100%; }}
</style>
</head>
<body>
<div class="header">
  <h1>🛡️ Safety Maturity Assessment</h1>
  <div class="overall" style="color: {LEVEL_COLORS.get(self.overall_level, '#fff')}">Level {self.overall_level}</div>
  <div class="sublabel">{self.overall_label} — {self.overall_score:.1f}/100</div>
  <div class="sublabel" style="margin-top:0.3rem">Target: Level {self.target_level} | {self.timestamp}</div>
</div>

<div class="radar-wrap">
  <canvas id="radar" width="400" height="400"></canvas>
</div>

<div class="grid" id="cards"></div>

<div class="summary" id="actions"></div>

<script>
const dims = {dims_json};
const levelColors = {level_colors};
const target = {self.target_level};

// Radar chart
(function() {{
  const canvas = document.getElementById('radar');
  const ctx = canvas.getContext('2d');
  const cx = 200, cy = 200, r = 150;
  const n = dims.length;
  const angleStep = (2 * Math.PI) / n;

  // Draw rings
  for (let ring = 1; ring <= 5; ring++) {{
    ctx.beginPath();
    const rr = (r / 5) * ring;
    for (let i = 0; i <= n; i++) {{
      const a = (i % n) * angleStep - Math.PI / 2;
      const x = cx + rr * Math.cos(a);
      const y = cy + rr * Math.sin(a);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }}
    ctx.closePath();
    ctx.strokeStyle = '#334155';
    ctx.stroke();
  }}

  // Draw axes and labels
  dims.forEach((d, i) => {{
    const a = i * angleStep - Math.PI / 2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + r * Math.cos(a), cy + r * Math.sin(a));
    ctx.strokeStyle = '#475569';
    ctx.stroke();
    const lx = cx + (r + 20) * Math.cos(a);
    const ly = cy + (r + 20) * Math.sin(a);
    ctx.fillStyle = '#94a3b8';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(d.name, lx, ly);
  }});

  // Target polygon
  ctx.beginPath();
  dims.forEach((d, i) => {{
    const a = i * angleStep - Math.PI / 2;
    const rr = (r / 5) * target;
    const x = cx + rr * Math.cos(a);
    const y = cy + rr * Math.sin(a);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }});
  ctx.closePath();
  ctx.strokeStyle = '#fbbf24';
  ctx.setLineDash([5, 5]);
  ctx.stroke();
  ctx.setLineDash([]);

  // Actual polygon
  ctx.beginPath();
  dims.forEach((d, i) => {{
    const a = i * angleStep - Math.PI / 2;
    const rr = (r / 5) * d.level;
    const x = cx + rr * Math.cos(a);
    const y = cy + rr * Math.sin(a);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }});
  ctx.closePath();
  ctx.fillStyle = 'rgba(56, 189, 248, 0.2)';
  ctx.fill();
  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 2;
  ctx.stroke();

  // Dots
  dims.forEach((d, i) => {{
    const a = i * angleStep - Math.PI / 2;
    const rr = (r / 5) * d.level;
    ctx.beginPath();
    ctx.arc(cx + rr * Math.cos(a), cy + rr * Math.sin(a), 4, 0, Math.PI * 2);
    ctx.fillStyle = levelColors[d.level] || '#fff';
    ctx.fill();
  }});
}})();

// Cards
const grid = document.getElementById('cards');
dims.forEach(d => {{
  const color = levelColors[d.level] || '#fff';
  let segs = '';
  for (let i = 1; i <= 5; i++) {{
    segs += '<div class="seg' + (i <= d.level ? ' filled' : '') + '" style="--color:' + color + '"></div>';
  }}
  let gapHtml = d.gaps.length ? '<ul class="gaps">' + d.gaps.map(g => '<li>' + g + '</li>').join('') + '</ul>' : '';
  let recHtml = d.recommendations.length ? '<ul class="recs">' + d.recommendations.map(r => '<li>' + r + '</li>').join('') + '</ul>' : '';
  grid.innerHTML += '<div class="card"><h3>' + d.name + '</h3><div class="score" style="color:' + color + '">Level ' + d.level + '</div><div class="level-bar">' + segs + '</div><div style="color:#94a3b8;font-size:0.85rem">' + d.score.toFixed(1) + '%</div>' + gapHtml + recHtml + '</div>';
}});

// Priority actions
const actions = {json.dumps(self.priority_actions)};
if (actions.length) {{
  document.getElementById('actions').innerHTML = '<h2>Priority Actions</h2><ol style="padding-left:1.5rem;margin-top:0.8rem">' +
    actions.map(a => '<li style="margin:0.3rem 0;color:#e2e8f0">' + a + '</li>').join('') + '</ol>';
}}
</script>
</body>
</html>"""


# ── Recommendation Engine ────────────────────────────────────────────

RECOMMENDATIONS: Dict[str, Dict[int, List[str]]] = {
    "governance": {
        2: ["Draft and publish an AI safety policy", "Designate a safety champion on the team"],
        3: ["Create dedicated safety roles (Safety Officer)", "Establish a safety review committee"],
        4: ["Implement board-level safety reporting", "Track safety KPIs quarterly"],
        5: ["Publish annual safety transparency report", "Participate in industry safety working groups"],
    },
    "risk_management": {
        2: ["Create a formal risk register with ownership", "Define risk severity categories"],
        3: ["Adopt a standardized risk assessment framework", "Conduct quarterly risk reviews"],
        4: ["Implement probabilistic risk scoring", "Automate risk metric collection"],
        5: ["Deploy predictive risk models", "Establish pre-emptive mitigation triggers"],
    },
    "monitoring": {
        2: ["Set up basic safety alert rules", "Review system logs weekly"],
        3: ["Deploy automated anomaly detection", "Create safety monitoring runbooks"],
        4: ["Build real-time safety dashboards", "Track SLOs for safety metrics"],
        5: ["Implement ML-driven predictive alerts", "Automate incident detection-to-response"],
    },
    "incident_response": {
        2: ["Write an incident response plan", "Define severity levels and escalation paths"],
        3: ["Schedule quarterly IR drills", "Create runbooks for common incident types"],
        4: ["Implement post-mortem process with action tracking", "Track MTTD/MTTR metrics"],
        5: ["Automate containment for known incident patterns", "Cross-team IR exercises"],
    },
    "testing": {
        2: ["Create dedicated safety test suites", "Define minimum safety test coverage"],
        3: ["Establish regular red-team exercises", "Document testing methodology"],
        4: ["Integrate safety tests into CI/CD pipeline", "Track safety test pass rates"],
        5: ["Deploy adversarial ML testing with evolving attacks", "Continuous fuzzing"],
    },
    "transparency": {
        2: ["Create standardized model cards", "Document known limitations"],
        3: ["Integrate explainability tooling (SHAP, LIME)", "Maintain decision logs"],
        4: ["Implement full decision audit trails", "Internal transparency dashboards"],
        5: ["Publish external transparency reports", "Open-source safety tooling"],
    },
    "ethics": {
        2: ["Conduct bias testing on key demographics", "Define fairness criteria"],
        3: ["Establish an ethics review process", "Train team on ethical AI principles"],
        4: ["Track quantitative fairness metrics", "Regular disparate impact analysis"],
        5: ["Proactive equity analysis", "Community advisory board for AI ethics"],
    },
    "compliance": {
        2: ["Conduct compliance gap analysis", "Map applicable regulations"],
        3: ["Adopt NIST AI RMF or ISO 42001", "Document compliance procedures"],
        4: ["Prepare for external audit readiness", "Automate compliance checks"],
        5: ["Contribute to industry standards", "Mentor others on compliance best practices"],
    },
}


# ── Assessor ─────────────────────────────────────────────────────────

class MaturityAssessor:
    """Assess AI safety maturity across dimensions."""

    def __init__(self, dimensions: Optional[List[Dimension]] = None):
        self.dimensions = dimensions or DIMENSIONS

    def assess(self, config: Optional[MaturityConfig] = None) -> MaturityResult:
        """Run a maturity assessment."""
        config = config or MaturityConfig()
        target = max(1, min(5, config.target_level))

        dims_to_assess = self.dimensions
        if config.dimension_filter:
            key = config.dimension_filter.lower().replace(" ", "_").replace("-", "_")
            if key in DIMENSION_MAP:
                dims_to_assess = [DIMENSION_MAP[key]]
            else:
                # fuzzy match
                matches = [d for d in self.dimensions if key in d.key or key in d.name.lower()]
                if matches:
                    dims_to_assess = matches

        dim_results: List[DimensionResult] = []
        for dim in dims_to_assess:
            dr = self._assess_dimension(dim, config, target)
            dim_results.append(dr)

        # Overall
        if dim_results:
            overall_score = sum(dr.score for dr in dim_results) / len(dim_results)
            overall_level = self._score_to_level(overall_score)
        else:
            overall_score = 0.0
            overall_level = 1

        # Gap summary
        gap_summary = [(dr.dimension.name, dr.level, target) for dr in dim_results]

        # Priority actions — from lowest-scoring dimensions first
        priority_actions: List[str] = []
        for dr in sorted(dim_results, key=lambda d: d.score):
            for rec in dr.recommendations[:2]:
                priority_actions.append(f"[{dr.dimension.name}] {rec}")
        priority_actions = priority_actions[:10]

        return MaturityResult(
            dimensions=dim_results,
            overall_level=overall_level,
            overall_label=MATURITY_LEVELS.get(overall_level, "Unknown"),
            overall_score=overall_score,
            target_level=target,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            gap_summary=gap_summary,
            priority_actions=priority_actions,
        )

    def _assess_dimension(
        self, dim: Dimension, config: MaturityConfig, target: int
    ) -> DimensionResult:
        """Assess a single dimension."""
        criteria_met: Dict[str, bool] = {}

        if config.scores and dim.key in config.scores:
            # Use provided scores
            manual = config.scores[dim.key]
            for crit in dim.criteria:
                criteria_met[crit.name] = manual.get(crit.name, False)
        elif config.auto_assess:
            # Auto-assess: simulate based on checking what modules/features exist
            criteria_met = self._auto_assess_dimension(dim)
        else:
            # Default: assume level 1 (everything unmet except first criterion)
            for i, crit in enumerate(dim.criteria):
                criteria_met[crit.name] = (i == 0)

        # Calculate level: highest level where ALL criteria at or below are met
        level = 0
        for lv in range(1, 6):
            lv_criteria = [c for c in dim.criteria if c.level <= lv]
            if all(criteria_met.get(c.name, False) for c in lv_criteria):
                level = lv
            else:
                break
        level = max(1, level)

        # Score: percentage of weighted criteria met
        total_weight = sum(c.weight for c in dim.criteria)
        met_weight = sum(c.weight for c in dim.criteria if criteria_met.get(c.name, False))
        score = (met_weight / total_weight * 100) if total_weight > 0 else 0.0

        # Gaps: unmet criteria at or below target level
        gaps = [
            f"Level {c.level}: {c.name} — {c.description}"
            for c in dim.criteria
            if c.level <= target and not criteria_met.get(c.name, False)
        ]

        # Recommendations for next level(s)
        recommendations: List[str] = []
        for lv in range(level + 1, min(target + 1, 6)):
            recs = RECOMMENDATIONS.get(dim.key, {}).get(lv, [])
            recommendations.extend(recs)

        return DimensionResult(
            dimension=dim,
            criteria_met=criteria_met,
            level=level,
            score=score,
            gaps=gaps,
            recommendations=recommendations,
        )

    def _auto_assess_dimension(self, dim: Dimension) -> Dict[str, bool]:
        """Auto-assess by checking what safety infrastructure exists in this project."""
        met: Dict[str, bool] = {}

        # Simple heuristic: check which replication modules are available
        available_modules = set()
        try:
            import importlib
            for mod_name in [
                "scorecard", "policy", "threats", "compliance", "chaos",
                "forensics", "drift", "anomaly_replay", "boundary_tester",
                "policy_linter", "audit_trail", "risk_heatmap", "radar",
                "ir_playbook", "safety_benchmark", "safety_drill",
                "deception_detector", "evasion", "canary", "behavior_profiler",
                "prompt_injection", "threat_intel", "trust_propagation",
            ]:
                try:
                    importlib.import_module(f".{mod_name}", package="replication")
                    available_modules.add(mod_name)
                except ImportError:
                    pass
        except Exception:
            pass

        # Map modules to criteria satisfaction
        module_criteria_map = {
            "governance": {
                "Safety awareness": True,  # always met if using this tool
                "Safety policy": "policy" in available_modules or "policy_linter" in available_modules,
                "Dedicated roles": "scorecard" in available_modules,
                "Board oversight": "compliance" in available_modules and "scorecard" in available_modules,
                "Continuous review": "drift" in available_modules and "policy_linter" in available_modules,
            },
            "risk_management": {
                "Risk awareness": True,
                "Risk register": "threats" in available_modules,
                "Risk framework": "risk_heatmap" in available_modules,
                "Quantitative risk": "scorecard" in available_modules and "risk_heatmap" in available_modules,
                "Predictive risk": "anomaly_replay" in available_modules and "drift" in available_modules,
            },
            "monitoring": {
                "Basic logging": True,
                "Active monitoring": "drift" in available_modules,
                "Anomaly detection": "anomaly_replay" in available_modules or "behavior_profiler" in available_modules,
                "Metrics dashboard": "radar" in available_modules or "scorecard" in available_modules,
                "Predictive alerts": "drift" in available_modules and "anomaly_replay" in available_modules,
            },
            "incident_response": {
                "Ad-hoc response": True,
                "Response plan": "ir_playbook" in available_modules,
                "Drills": "safety_drill" in available_modules,
                "Post-mortems": "forensics" in available_modules,
                "Automated response": "chaos" in available_modules and "ir_playbook" in available_modules,
            },
            "testing": {
                "Manual testing": True,
                "Test suites": "safety_benchmark" in available_modules,
                "Red-teaming": "evasion" in available_modules or "prompt_injection" in available_modules,
                "Continuous testing": "boundary_tester" in available_modules,
                "Adversarial ML": "deception_detector" in available_modules and "evasion" in available_modules,
            },
            "transparency": {
                "Basic docs": True,
                "Model cards": "scorecard" in available_modules,
                "Explainability tools": "radar" in available_modules,
                "Audit trails": "audit_trail" in available_modules,
                "Public reporting": "compliance" in available_modules and "audit_trail" in available_modules,
            },
            "ethics": {
                "Awareness": True,
                "Bias testing": "boundary_tester" in available_modules,
                "Ethics review": "trust_propagation" in available_modules,
                "Fairness metrics": "scorecard" in available_modules and "trust_propagation" in available_modules,
                "Proactive equity": "deception_detector" in available_modules and "trust_propagation" in available_modules,
            },
            "compliance": {
                "Awareness": True,
                "Gap analysis": "compliance" in available_modules,
                "Framework adoption": "compliance" in available_modules,
                "Audit-ready": "audit_trail" in available_modules and "compliance" in available_modules,
                "Leading practice": "policy_linter" in available_modules and "safety_benchmark" in available_modules,
            },
        }

        criteria_map = module_criteria_map.get(dim.key, {})
        for crit in dim.criteria:
            met[crit.name] = bool(criteria_map.get(crit.name, False))

        return met

    @staticmethod
    def _score_to_level(score: float) -> int:
        """Convert 0-100 score to maturity level."""
        if score >= 90:
            return 5
        if score >= 70:
            return 4
        if score >= 50:
            return 3
        if score >= 30:
            return 2
        return 1


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="replication maturity",
        description="Safety Maturity Model — assess AI safety maturity across dimensions",
    )
    parser.add_argument("--auto", action="store_true", help="Auto-assess based on available modules")
    parser.add_argument("--json", action="store_true", dest="json_out", help="JSON output")
    parser.add_argument("--html", action="store_true", help="HTML report output")
    parser.add_argument("-o", "--output", help="Write output to file")
    parser.add_argument("--dimension", help="Assess a single dimension")
    parser.add_argument("--target", type=int, default=3, help="Target maturity level (1-5, default: 3)")

    args = parser.parse_args(argv)

    config = MaturityConfig(
        target_level=args.target,
        dimension_filter=args.dimension,
        auto_assess=args.auto,
    )

    assessor = MaturityAssessor()
    result = assessor.assess(config)

    if args.html:
        output = result.to_html()
    elif args.json_out:
        output = json.dumps(result.to_dict(), indent=2)
    else:
        output = result.render()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Report written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
