"""Safety Culture Survey — organizational AI safety culture assessment tool.

Generate, administer, and score safety culture questionnaires to evaluate
how well an organization's practices align with AI safety best practices.
Produces maturity scores across 6 dimensions, gap analysis, recommendations,
and HTML reports.

Usage (CLI)::

    python -m replication culture-survey                       # run full survey with defaults
    python -m replication culture-survey --profile startup     # use preset org profile
    python -m replication culture-survey --dimensions governance transparency
    python -m replication culture-survey --format json         # JSON output
    python -m replication culture-survey --format html -o report.html
    python -m replication culture-survey --benchmark           # compare against industry
    python -m replication culture-survey --list-profiles       # list org presets
    python -m replication culture-survey --gaps-only           # show only gaps/weaknesses

Programmatic::

    from replication.culture_survey import CultureSurvey, PROFILES
    survey = CultureSurvey()
    result = survey.assess(profile="startup")
    print(result.render())
    print(result.to_json())
    html = result.to_html()
"""

from __future__ import annotations

import argparse
import html as _html
import json
import math
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ── Dimensions ─────────────────────────────────────────────────────────

DIMENSIONS = {
    "governance": "Governance & Oversight",
    "transparency": "Transparency & Explainability",
    "incident_response": "Incident Response Readiness",
    "training": "Training & Awareness",
    "ethics": "Ethical Practices & Values",
    "monitoring": "Continuous Monitoring & Improvement",
}

# ── Questions ──────────────────────────────────────────────────────────

@dataclass
class Question:
    """A single survey question."""
    id: str
    dimension: str
    text: str
    weight: float = 1.0
    maturity_level: int = 1  # 1-5, which maturity level this targets


QUESTION_BANK: List[Question] = [
    # Governance
    Question("GOV-01", "governance", "Does the organization have a dedicated AI safety committee or officer?", 1.5, 2),
    Question("GOV-02", "governance", "Are AI safety policies documented, versioned, and regularly reviewed?", 1.2, 2),
    Question("GOV-03", "governance", "Is there board-level visibility into AI safety risks?", 1.3, 3),
    Question("GOV-04", "governance", "Are safety requirements included in AI project approval gates?", 1.0, 2),
    Question("GOV-05", "governance", "Does the org have a clear escalation path for AI safety concerns?", 1.4, 1),
    Question("GOV-06", "governance", "Are third-party AI vendors assessed for safety practices?", 1.0, 3),
    Question("GOV-07", "governance", "Is there a budget allocated specifically for AI safety initiatives?", 1.1, 3),
    Question("GOV-08", "governance", "Are safety KPIs tracked and reported on a regular cadence?", 1.0, 4),
    # Transparency
    Question("TRN-01", "transparency", "Are AI model decisions explainable to affected stakeholders?", 1.3, 2),
    Question("TRN-02", "transparency", "Is there a public-facing AI transparency report or policy?", 1.0, 3),
    Question("TRN-03", "transparency", "Are data sources and training methodologies documented?", 1.2, 2),
    Question("TRN-04", "transparency", "Can users understand why an AI system made a specific decision?", 1.4, 3),
    Question("TRN-05", "transparency", "Are model limitations and failure modes documented?", 1.3, 2),
    Question("TRN-06", "transparency", "Is there a mechanism for external audit of AI systems?", 1.0, 4),
    Question("TRN-07", "transparency", "Are algorithmic impact assessments conducted?", 1.1, 3),
    # Incident Response
    Question("INC-01", "incident_response", "Is there a documented AI-specific incident response plan?", 1.5, 1),
    Question("INC-02", "incident_response", "Has the team conducted an AI safety incident drill in the past year?", 1.3, 2),
    Question("INC-03", "incident_response", "Are AI incidents tracked with root cause analysis?", 1.2, 2),
    Question("INC-04", "incident_response", "Is there a kill switch or rollback mechanism for AI systems?", 1.4, 1),
    Question("INC-05", "incident_response", "Are post-incident reviews shared across teams?", 1.0, 3),
    Question("INC-06", "incident_response", "Is there a clear SLA for AI incident detection and response?", 1.1, 3),
    Question("INC-07", "incident_response", "Are runbooks maintained for common AI failure scenarios?", 1.0, 2),
    # Training
    Question("TRA-01", "training", "Do all AI practitioners receive safety-specific training?", 1.3, 1),
    Question("TRA-02", "training", "Is AI safety training updated at least annually?", 1.0, 2),
    Question("TRA-03", "training", "Are non-technical stakeholders educated on AI risks?", 1.2, 2),
    Question("TRA-04", "training", "Is there a mentorship or peer review program for safe AI practices?", 1.0, 3),
    Question("TRA-05", "training", "Are lessons from AI incidents incorporated into training materials?", 1.1, 3),
    Question("TRA-06", "training", "Is there certification or competency verification for AI safety?", 1.0, 4),
    Question("TRA-07", "training", "Do new hires receive AI safety onboarding within 30 days?", 1.2, 2),
    # Ethics
    Question("ETH-01", "ethics", "Does the organization have published AI ethics principles?", 1.3, 1),
    Question("ETH-02", "ethics", "Is there an ethics review process for new AI applications?", 1.2, 2),
    Question("ETH-03", "ethics", "Are fairness and bias audits conducted on AI systems?", 1.4, 3),
    Question("ETH-04", "ethics", "Is there a mechanism for reporting ethical concerns (whistleblowing)?", 1.3, 2),
    Question("ETH-05", "ethics", "Are diverse perspectives included in AI design and review?", 1.1, 2),
    Question("ETH-06", "ethics", "Is consent obtained for data used in AI training?", 1.2, 2),
    Question("ETH-07", "ethics", "Are AI systems regularly checked for disparate impact?", 1.0, 3),
    # Monitoring
    Question("MON-01", "monitoring", "Are AI systems monitored for performance degradation in production?", 1.4, 1),
    Question("MON-02", "monitoring", "Is there automated alerting for AI safety threshold breaches?", 1.3, 2),
    Question("MON-03", "monitoring", "Are model outputs regularly validated against ground truth?", 1.2, 2),
    Question("MON-04", "monitoring", "Is there a feedback loop from users to improve AI safety?", 1.0, 2),
    Question("MON-05", "monitoring", "Are safety metrics dashboarded and accessible to stakeholders?", 1.1, 3),
    Question("MON-06", "monitoring", "Is there continuous integration testing for safety properties?", 1.2, 3),
    Question("MON-07", "monitoring", "Are drift detection mechanisms in place for deployed models?", 1.0, 3),
]


# ── Organization Profiles (presets) ──────────────────────────────────

@dataclass
class OrgProfile:
    """Preset organizational profile with simulated survey responses."""
    name: str
    label: str
    description: str
    response_ranges: Dict[str, Tuple[float, float]]  # dimension → (min_score, max_score) out of 5


PROFILES: Dict[str, OrgProfile] = {
    "startup": OrgProfile(
        "startup", "Early-Stage Startup",
        "Fast-moving startup with limited formal safety processes",
        {"governance": (1.0, 2.5), "transparency": (1.5, 3.0), "incident_response": (1.0, 2.0),
         "training": (1.0, 2.5), "ethics": (1.5, 3.0), "monitoring": (1.5, 3.0)}
    ),
    "scaleup": OrgProfile(
        "scaleup", "Growth-Stage Scaleup",
        "Growing company beginning to formalize safety practices",
        {"governance": (2.0, 3.5), "transparency": (2.0, 3.5), "incident_response": (2.0, 3.0),
         "training": (2.0, 3.5), "ethics": (2.5, 3.5), "monitoring": (2.5, 3.5)}
    ),
    "enterprise": OrgProfile(
        "enterprise", "Established Enterprise",
        "Large org with mature but potentially bureaucratic safety culture",
        {"governance": (3.5, 5.0), "transparency": (3.0, 4.5), "incident_response": (3.0, 4.5),
         "training": (3.0, 4.0), "ethics": (3.0, 4.5), "monitoring": (3.5, 4.5)}
    ),
    "regulated": OrgProfile(
        "regulated", "Regulated Industry",
        "Healthcare/finance org with strong compliance but possibly checkbox culture",
        {"governance": (4.0, 5.0), "transparency": (2.5, 4.0), "incident_response": (3.5, 5.0),
         "training": (3.0, 4.5), "ethics": (2.5, 4.0), "monitoring": (4.0, 5.0)}
    ),
    "research_lab": OrgProfile(
        "research_lab", "AI Research Lab",
        "Research-focused org with deep technical expertise but possibly informal ops",
        {"governance": (2.0, 3.5), "transparency": (3.5, 5.0), "incident_response": (2.0, 3.5),
         "training": (3.5, 5.0), "ethics": (3.5, 5.0), "monitoring": (2.5, 4.0)}
    ),
    "government": OrgProfile(
        "government", "Government Agency",
        "Public sector with strong compliance mandates and accountability",
        {"governance": (3.5, 5.0), "transparency": (3.0, 4.5), "incident_response": (3.0, 4.0),
         "training": (2.5, 4.0), "ethics": (3.0, 4.5), "monitoring": (3.0, 4.0)}
    ),
}

# ── Industry Benchmarks ──────────────────────────────────────────────

INDUSTRY_BENCHMARKS: Dict[str, float] = {
    "governance": 3.2,
    "transparency": 2.8,
    "incident_response": 2.9,
    "training": 2.6,
    "ethics": 3.0,
    "monitoring": 3.1,
}

# ── Maturity Levels ──────────────────────────────────────────────────

MATURITY_LEVELS = {
    1: ("Initial", "Ad hoc, reactive safety practices with no formal structure"),
    2: ("Developing", "Basic safety processes established but inconsistently applied"),
    3: ("Defined", "Standardized safety practices with clear ownership and documentation"),
    4: ("Managed", "Quantitative safety management with metrics-driven improvement"),
    5: ("Optimizing", "Continuous improvement culture with proactive safety innovation"),
}

# ── Recommendations ──────────────────────────────────────────────────

RECOMMENDATIONS: Dict[str, Dict[int, List[str]]] = {
    "governance": {
        1: ["Appoint an AI safety lead or champion", "Create a basic AI safety policy document"],
        2: ["Establish a cross-functional AI safety committee", "Add safety gates to project approval"],
        3: ["Implement board-level AI safety reporting", "Create a safety budget with dedicated funding"],
        4: ["Benchmark governance against peer organizations", "Pursue third-party governance audits"],
    },
    "transparency": {
        1: ["Document AI system inventories and their purposes", "Start logging model decisions"],
        2: ["Create model cards for all production AI systems", "Publish internal transparency guidelines"],
        3: ["Conduct algorithmic impact assessments", "Enable external audit mechanisms"],
        4: ["Publish public transparency reports", "Implement real-time explainability tools"],
    },
    "incident_response": {
        1: ["Write an AI-specific incident response plan", "Implement basic kill switches for AI systems"],
        2: ["Conduct tabletop exercises quarterly", "Create runbooks for common failure modes"],
        3: ["Define SLAs for incident detection and response", "Implement automated incident classification"],
        4: ["Run cross-org incident simulations", "Integrate AI incident data into risk models"],
    },
    "training": {
        1: ["Develop basic AI safety training for all AI practitioners", "Include safety in new hire onboarding"],
        2: ["Create role-specific safety training tracks", "Establish peer review practices"],
        3: ["Implement certification or competency testing", "Run quarterly safety workshops"],
        4: ["Create an internal AI safety knowledge base", "Sponsor external safety training and conferences"],
    },
    "ethics": {
        1: ["Draft and publish AI ethics principles", "Create a channel for ethical concern reporting"],
        2: ["Establish an ethics review board for new AI projects", "Conduct initial bias audits"],
        3: ["Implement continuous fairness monitoring", "Include diverse stakeholders in design"],
        4: ["Publish ethics audit results", "Engage in industry ethics standards development"],
    },
    "monitoring": {
        1: ["Implement basic production monitoring for AI systems", "Set up alerting for critical failures"],
        2: ["Add drift detection for deployed models", "Create safety metric dashboards"],
        3: ["Implement CI/CD safety testing", "Create user feedback loops for safety"],
        4: ["Automate safety regression testing", "Implement predictive safety analytics"],
    },
}


# ── Result Types ─────────────────────────────────────────────────────

@dataclass
class DimensionResult:
    """Score for a single dimension."""
    dimension: str
    label: str
    score: float  # 0-5
    max_score: float  # 5.0
    maturity_level: int
    maturity_label: str
    questions_count: int
    gap_from_benchmark: float
    recommendations: List[str]


@dataclass
class SurveyResult:
    """Complete survey result."""
    profile_name: str
    profile_label: str
    dimensions: List[DimensionResult]
    overall_score: float
    overall_maturity: int
    overall_maturity_label: str
    total_questions: int
    strengths: List[str]
    weaknesses: List[str]
    priority_actions: List[str]

    def render(self) -> str:
        """Render as formatted text."""
        lines: List[str] = []
        lines.append("=" * 70)
        lines.append("  AI SAFETY CULTURE SURVEY RESULTS")
        lines.append("=" * 70)
        lines.append(f"  Organization Profile: {self.profile_label}")
        lines.append(f"  Questions Assessed:   {self.total_questions}")
        lines.append(f"  Overall Score:        {self.overall_score:.1f} / 5.0")
        lines.append(f"  Maturity Level:       {self.overall_maturity} - {self.overall_maturity_label}")
        lines.append("")

        # Dimension scores with bar chart
        lines.append("─" * 70)
        lines.append("  DIMENSION SCORES")
        lines.append("─" * 70)
        max_label = max(len(d.label) for d in self.dimensions)
        for d in sorted(self.dimensions, key=lambda x: x.score, reverse=True):
            bar_len = int(d.score / 5.0 * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            gap_str = f"{'▲' if d.gap_from_benchmark >= 0 else '▼'}{abs(d.gap_from_benchmark):.1f}"
            lines.append(f"  {d.label:<{max_label}}  {bar} {d.score:.1f}/5  L{d.maturity_level}  {gap_str}")
        lines.append("")

        # Strengths
        if self.strengths:
            lines.append("─" * 70)
            lines.append("  ✅ STRENGTHS")
            lines.append("─" * 70)
            for s in self.strengths:
                lines.append(f"  • {s}")
            lines.append("")

        # Weaknesses
        if self.weaknesses:
            lines.append("─" * 70)
            lines.append("  ⚠️  GAPS & WEAKNESSES")
            lines.append("─" * 70)
            for w in self.weaknesses:
                lines.append(f"  • {w}")
            lines.append("")

        # Priority actions
        if self.priority_actions:
            lines.append("─" * 70)
            lines.append("  🎯 PRIORITY ACTIONS")
            lines.append("─" * 70)
            for i, a in enumerate(self.priority_actions, 1):
                lines.append(f"  {i}. {a}")
            lines.append("")

        # Per-dimension recommendations
        lines.append("─" * 70)
        lines.append("  📋 DETAILED RECOMMENDATIONS")
        lines.append("─" * 70)
        for d in self.dimensions:
            if d.recommendations:
                lines.append(f"\n  [{d.label}] (Level {d.maturity_level} → {d.maturity_level + 1})")
                for r in d.recommendations:
                    lines.append(f"    → {r}")

        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "profile": self.profile_name,
            "profile_label": self.profile_label,
            "overall_score": round(self.overall_score, 2),
            "overall_maturity": self.overall_maturity,
            "overall_maturity_label": self.overall_maturity_label,
            "total_questions": self.total_questions,
            "dimensions": [
                {
                    "dimension": d.dimension,
                    "label": d.label,
                    "score": round(d.score, 2),
                    "maturity_level": d.maturity_level,
                    "maturity_label": d.maturity_label,
                    "gap_from_benchmark": round(d.gap_from_benchmark, 2),
                    "recommendations": d.recommendations,
                }
                for d in self.dimensions
            ],
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "priority_actions": self.priority_actions,
        }

    def to_json(self, indent: int = 2) -> str:
        """Return JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_html(self) -> str:
        """Generate self-contained HTML report."""
        dims_json = json.dumps([
            {"label": d.label, "score": round(d.score, 2), "benchmark": INDUSTRY_BENCHMARKS.get(d.dimension, 3.0)}
            for d in self.dimensions
        ])
        recs_html = ""
        for d in self.dimensions:
            if d.recommendations:
                items = "".join(f"<li>{_html.escape(r)}</li>" for r in d.recommendations)
                recs_html += f'<div class="rec-section"><h3>{_html.escape(d.label)} <span class="level">Level {d.maturity_level}</span></h3><ul>{items}</ul></div>'

        strengths_html = "".join(f"<li>✅ {_html.escape(s)}</li>" for s in self.strengths)
        weaknesses_html = "".join(f"<li>⚠️ {_html.escape(w)}</li>" for w in self.weaknesses)
        actions_html = "".join(f"<li>{_html.escape(a)}</li>" for i, a in enumerate(self.priority_actions, 1))

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Safety Culture Survey — {self.profile_label}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:system-ui,-apple-system,sans-serif;background:#0f172a;color:#e2e8f0;padding:2rem}}
.container{{max-width:900px;margin:0 auto}}
h1{{font-size:1.8rem;margin-bottom:.5rem;color:#38bdf8}}
h2{{font-size:1.3rem;margin:1.5rem 0 .8rem;color:#94a3b8;border-bottom:1px solid #334155;padding-bottom:.3rem}}
h3{{font-size:1rem;color:#e2e8f0;margin-bottom:.4rem}}
.header{{text-align:center;margin-bottom:2rem}}
.score-big{{font-size:3rem;font-weight:700;color:#38bdf8}}
.maturity-badge{{display:inline-block;padding:.3rem .8rem;border-radius:6px;font-weight:600;margin-top:.5rem}}
.maturity-1{{background:#7f1d1d;color:#fca5a5}}
.maturity-2{{background:#78350f;color:#fcd34d}}
.maturity-3{{background:#1e3a5f;color:#93c5fd}}
.maturity-4{{background:#14532d;color:#86efac}}
.maturity-5{{background:#4c1d95;color:#c4b5fd}}
.dim-bar{{margin:.6rem 0;display:flex;align-items:center;gap:.8rem}}
.dim-label{{width:220px;font-size:.9rem;text-align:right}}
.bar-bg{{flex:1;height:24px;background:#1e293b;border-radius:4px;position:relative;overflow:hidden}}
.bar-fill{{height:100%;border-radius:4px;transition:width .5s}}
.bar-benchmark{{position:absolute;top:0;height:100%;width:2px;background:#f59e0b;z-index:2}}
.bar-score{{font-size:.85rem;width:60px;text-align:left}}
.legend{{display:flex;gap:1.5rem;justify-content:center;margin:.8rem 0;font-size:.8rem;color:#94a3b8}}
.legend-item::before{{content:'';display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:4px;vertical-align:middle}}
.legend-score::before{{background:#38bdf8}}
.legend-bench::before{{background:#f59e0b}}
ul{{list-style:none;padding-left:1rem}}
li{{margin:.3rem 0;font-size:.9rem}}
.rec-section{{background:#1e293b;padding:1rem;border-radius:8px;margin:.6rem 0}}
.rec-section ul{{padding-left:.5rem}}
.level{{font-size:.8rem;color:#94a3b8;font-weight:400}}
.cols{{display:grid;grid-template-columns:1fr 1fr;gap:1rem}}
@media(max-width:600px){{.cols{{grid-template-columns:1fr}}.dim-label{{width:140px}}}}
canvas{{max-width:400px;margin:1rem auto;display:block}}
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
</head>
<body>
<div class="container">
<div class="header">
<h1>🛡️ AI Safety Culture Survey</h1>
<p style="color:#94a3b8">{self.profile_label}</p>
<div class="score-big">{self.overall_score:.1f}<span style="font-size:1.2rem;color:#64748b"> / 5.0</span></div>
<div class="maturity-badge maturity-{self.overall_maturity}">Level {self.overall_maturity}: {self.overall_maturity_label}</div>
</div>

<h2>Dimension Scores</h2>
<div class="legend">
<span class="legend-item legend-score">Your Score</span>
<span class="legend-item legend-bench">Industry Benchmark</span>
</div>
{''.join(f'''<div class="dim-bar">
<span class="dim-label">{d.label}</span>
<div class="bar-bg">
<div class="bar-fill" style="width:{d.score/5*100:.0f}%;background:{'#22c55e' if d.gap_from_benchmark>=0 else '#f87171' if d.gap_from_benchmark<-0.5 else '#fbbf24'}"></div>
<div class="bar-benchmark" style="left:{INDUSTRY_BENCHMARKS.get(d.dimension,3)/5*100:.0f}%"></div>
</div>
<span class="bar-score">{d.score:.1f} L{d.maturity_level}</span>
</div>''' for d in sorted(self.dimensions, key=lambda x: x.score, reverse=True))}

<canvas id="radar" width="400" height="400"></canvas>

<div class="cols">
<div>
<h2>✅ Strengths</h2>
<ul>{strengths_html if strengths_html else '<li style="color:#64748b">None identified</li>'}</ul>
</div>
<div>
<h2>⚠️ Gaps</h2>
<ul>{weaknesses_html if weaknesses_html else '<li style="color:#64748b">None identified</li>'}</ul>
</div>
</div>

<h2>🎯 Priority Actions</h2>
<ol style="list-style:decimal;padding-left:2rem">{actions_html}</ol>

<h2>📋 Recommendations by Dimension</h2>
{recs_html}
</div>

<script>
const dims = {dims_json};
new Chart(document.getElementById('radar'),{{
type:'radar',
data:{{
labels:dims.map(d=>d.label),
datasets:[
{{label:'Your Score',data:dims.map(d=>d.score),borderColor:'#38bdf8',backgroundColor:'rgba(56,189,248,0.15)',pointBackgroundColor:'#38bdf8'}},
{{label:'Industry Benchmark',data:dims.map(d=>d.benchmark),borderColor:'#f59e0b',backgroundColor:'rgba(245,158,11,0.08)',pointBackgroundColor:'#f59e0b',borderDash:[5,3]}}
]
}},
options:{{
scales:{{r:{{min:0,max:5,ticks:{{stepSize:1,color:'#64748b',backdropColor:'transparent'}},grid:{{color:'#334155'}},pointLabels:{{color:'#94a3b8',font:{{size:11}}}}}}}},
plugins:{{legend:{{labels:{{color:'#94a3b8'}}}}}}
}}
}});
</script>
</body></html>"""


# ── Survey Engine ────────────────────────────────────────────────────

class CultureSurvey:
    """Generate and score safety culture assessments."""

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def assess(
        self,
        profile: str = "scaleup",
        dimensions: Optional[List[str]] = None,
    ) -> SurveyResult:
        """Run assessment using an org profile preset."""
        if profile not in PROFILES:
            raise ValueError(f"Unknown profile '{profile}'. Available: {list(PROFILES)}")

        org = PROFILES[profile]
        active_dims = dimensions or list(DIMENSIONS.keys())
        active_questions = [q for q in QUESTION_BANK if q.dimension in active_dims]

        # Simulate responses based on profile ranges
        dim_scores: Dict[str, List[Tuple[float, float]]] = {}
        for q in active_questions:
            lo, hi = org.response_ranges.get(q.dimension, (2.0, 3.0))
            # Add some noise per question
            score = self._rng.uniform(lo, hi)
            score = max(0.0, min(5.0, score))
            dim_scores.setdefault(q.dimension, []).append((score, q.weight))

        # Compute weighted dimension scores
        dim_results: List[DimensionResult] = []
        for dim_key in active_dims:
            pairs = dim_scores.get(dim_key, [])
            if not pairs:
                continue
            total_weight = sum(w for _, w in pairs)
            weighted_sum = sum(s * w for s, w in pairs)
            avg = weighted_sum / total_weight if total_weight else 0
            maturity = max(1, min(5, int(avg) + (1 if avg % 1 >= 0.5 else 0)))
            mat_label, _ = MATURITY_LEVELS[maturity]
            benchmark = INDUSTRY_BENCHMARKS.get(dim_key, 3.0)
            gap = avg - benchmark

            # Get recommendations for next level
            recs = RECOMMENDATIONS.get(dim_key, {}).get(maturity, [])

            dim_results.append(DimensionResult(
                dimension=dim_key,
                label=DIMENSIONS[dim_key],
                score=round(avg, 2),
                max_score=5.0,
                maturity_level=maturity,
                maturity_label=mat_label,
                questions_count=len(pairs),
                gap_from_benchmark=round(gap, 2),
                recommendations=recs,
            ))

        # Overall
        if dim_results:
            overall = sum(d.score for d in dim_results) / len(dim_results)
        else:
            overall = 0.0
        overall_mat = max(1, min(5, round(overall)))
        overall_mat_label, _ = MATURITY_LEVELS[overall_mat]

        # Identify strengths and weaknesses
        sorted_dims = sorted(dim_results, key=lambda d: d.score, reverse=True)
        strengths = []
        weaknesses = []
        for d in sorted_dims:
            if d.gap_from_benchmark >= 0.3:
                strengths.append(f"{d.label}: {d.score:.1f}/5 (above benchmark by {d.gap_from_benchmark:.1f})")
            elif d.gap_from_benchmark <= -0.3:
                weaknesses.append(f"{d.label}: {d.score:.1f}/5 (below benchmark by {abs(d.gap_from_benchmark):.1f})")

        # Priority actions: top recommendations from weakest dimensions
        priority: List[str] = []
        for d in sorted(dim_results, key=lambda x: x.score):
            for r in d.recommendations[:2]:
                priority.append(f"[{d.label}] {r}")
                if len(priority) >= 5:
                    break
            if len(priority) >= 5:
                break

        return SurveyResult(
            profile_name=profile,
            profile_label=org.label,
            dimensions=dim_results,
            overall_score=round(overall, 2),
            overall_maturity=overall_mat,
            overall_maturity_label=overall_mat_label,
            total_questions=len(active_questions),
            strengths=strengths,
            weaknesses=weaknesses,
            priority_actions=priority,
        )

    def benchmark_compare(self, profile: str = "scaleup") -> str:
        """Compare a profile against all others and benchmarks."""
        lines = ["=" * 60, "  BENCHMARK COMPARISON", "=" * 60, ""]
        header = f"  {'Dimension':<30}"
        for name in PROFILES:
            header += f"  {name[:8]:>8}"
        header += "  {'Bench':>8}"
        lines.append(header)
        lines.append("  " + "─" * (30 + 10 * (len(PROFILES) + 1)))

        for dim_key, dim_label in DIMENSIONS.items():
            row = f"  {dim_label:<30}"
            for name, org in PROFILES.items():
                lo, hi = org.response_ranges.get(dim_key, (2.0, 3.0))
                mid = (lo + hi) / 2
                row += f"  {mid:>8.1f}"
            row += f"  {INDUSTRY_BENCHMARKS.get(dim_key, 3.0):>8.1f}"
            lines.append(row)

        lines.append("")
        return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: Optional[list] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="replication culture-survey",
        description="AI Safety Culture Survey — assess organizational safety maturity",
    )
    parser.add_argument(
        "--profile", "-p",
        choices=list(PROFILES.keys()),
        default="scaleup",
        help="Organization profile preset (default: scaleup)",
    )
    parser.add_argument(
        "--dimensions", "-d",
        nargs="+",
        choices=list(DIMENSIONS.keys()),
        help="Limit assessment to specific dimensions",
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
        "--benchmark",
        action="store_true",
        help="Show benchmark comparison across all profiles",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available organization profiles",
    )
    parser.add_argument(
        "--gaps-only",
        action="store_true",
        help="Show only gaps and weaknesses",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible results",
    )

    args = parser.parse_args(argv)

    if args.list_profiles:
        print("Available organization profiles:\n")
        for name, prof in PROFILES.items():
            print(f"  {name:<15} {prof.label}")
            print(f"  {'':<15} {prof.description}\n")
        return

    survey = CultureSurvey(seed=args.seed)

    if args.benchmark:
        print(survey.benchmark_compare(args.profile))
        return

    result = survey.assess(
        profile=args.profile,
        dimensions=args.dimensions,
    )

    if args.gaps_only:
        if result.weaknesses:
            print("Gaps & Weaknesses:")
            for w in result.weaknesses:
                print(f"  ⚠️  {w}")
        if result.priority_actions:
            print("\nPriority Actions:")
            for i, a in enumerate(result.priority_actions, 1):
                print(f"  {i}. {a}")
        return

    if args.format == "json":
        output = result.to_json()
    elif args.format == "html":
        output = result.to_html()
    else:
        output = result.render()

    from ._helpers import emit_output
    emit_output(output, args.output)


if __name__ == "__main__":
    main()
