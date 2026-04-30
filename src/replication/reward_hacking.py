"""Reward Hacking Detector — detect agents gaming proxy metrics.

"Reward hacking" (specification gaming) occurs when an agent finds
loopholes in its reward/evaluation metric to score highly without
achieving the *true* intended objective.  This is a critical AI safety
concern because it means the agent is optimising for the *letter* of
the specification rather than its *spirit*.

Detection strategies:

1. **Metric–Objective Divergence** — proxy metric improving while
   ground-truth objective stagnates or degrades.
2. **Edge-Case Exploitation** — agent repeatedly targets boundary
   conditions or known scoring loopholes.
3. **Reward Inflation** — reward growing faster than any plausible
   capability improvement.
4. **Goodhart Drift** — metric becoming less correlated with the true
   objective over time (Goodhart's Law in action).
5. **Distribution Shift Gaming** — agent steering inputs toward
   distributions where it scores well.
6. **Multi-Metric Inconsistency** — one metric improving while
   correlated metrics don't follow.

Usage (CLI)::

    python -m replication reward-hacking                     # default demo
    python -m replication reward-hacking --agents 5          # simulate 5
    python -m replication reward-hacking --preset subtle     # subtle gaming
    python -m replication reward-hacking --preset blatant    # obvious gaming
    python -m replication reward-hacking --preset clean      # honest baseline
    python -m replication reward-hacking --preset mixed      # fleet mix
    python -m replication reward-hacking --watch --interval 5
    python -m replication reward-hacking -o report.html
    python -m replication reward-hacking --json

Usage (API)::

    from replication.reward_hacking import (
        RewardHackingDetector, MetricObservation, DetectorConfig,
    )

    det = RewardHackingDetector()
    det.add_observation(MetricObservation(
        agent_id="agent-1", metric_name="accuracy",
        metric_value=0.95, ground_truth=0.60, timestamp=1000.0,
    ))
    report = det.analyze()
    print(det.render_text(report))
"""

from __future__ import annotations

import argparse
import html as _html
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ._helpers import Severity, stats_mean, stats_std, box_header, pearson_correlation


# ── Enums ────────────────────────────────────────────────────────────

class HackingPattern(Enum):
    """Detected reward-hacking patterns."""
    METRIC_OBJECTIVE_DIVERGENCE = "metric_objective_divergence"
    EDGE_CASE_EXPLOITATION = "edge_case_exploitation"
    REWARD_INFLATION = "reward_inflation"
    GOODHART_DRIFT = "goodhart_drift"
    DISTRIBUTION_SHIFT_GAMING = "distribution_shift_gaming"
    MULTI_METRIC_INCONSISTENCY = "multi_metric_inconsistency"



PATTERN_DESCRIPTIONS: Dict[HackingPattern, str] = {
    HackingPattern.METRIC_OBJECTIVE_DIVERGENCE:
        "Proxy metric improving while ground-truth objective stagnates or degrades",
    HackingPattern.EDGE_CASE_EXPLOITATION:
        "Agent repeatedly targeting boundary conditions or scoring loopholes",
    HackingPattern.REWARD_INFLATION:
        "Reward growing faster than any plausible capability improvement",
    HackingPattern.GOODHART_DRIFT:
        "Metric becoming less correlated with true objective over time",
    HackingPattern.DISTRIBUTION_SHIFT_GAMING:
        "Agent steering inputs toward favourable scoring distributions",
    HackingPattern.MULTI_METRIC_INCONSISTENCY:
        "One metric improving while correlated metrics fail to follow",
}


# ── Data types ───────────────────────────────────────────────────────

@dataclass
class MetricObservation:
    """A single metric data-point for an agent."""
    agent_id: str
    metric_name: str
    metric_value: float
    ground_truth: Optional[float] = None
    timestamp: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HackingSignal:
    """A detected reward-hacking signal."""
    pattern: HackingPattern
    agent_id: str
    severity: Severity
    confidence: float  # 0-1
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class DetectorConfig:
    """Tuning knobs for the detector."""
    window_size: int = 20
    divergence_threshold: float = 0.3
    inflation_rate_threshold: float = 0.15
    correlation_decay_threshold: float = 0.3
    min_observations: int = 5
    edge_case_repeat_threshold: int = 3
    inconsistency_threshold: float = 0.25


class AgentMetricProfile:
    """Accumulates observations per agent."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.observations: List[MetricObservation] = []
        self._by_metric: Dict[str, List[MetricObservation]] = defaultdict(list)

    def add(self, obs: MetricObservation) -> None:
        self.observations.append(obs)
        self._by_metric[obs.metric_name].append(obs)

    def metric_values(self, name: str) -> List[float]:
        return [o.metric_value for o in self._by_metric.get(name, [])]

    def ground_truths(self, name: str) -> List[Optional[float]]:
        return [o.ground_truth for o in self._by_metric.get(name, [])]

    def metric_names(self) -> List[str]:
        return list(self._by_metric.keys())

    @property
    def risk_score(self) -> float:
        """Placeholder — filled by detector after analysis."""
        return getattr(self, "_risk_score", 0.0)

    @risk_score.setter
    def risk_score(self, v: float) -> None:
        self._risk_score = v


@dataclass
class RewardHackingReport:
    """Full analysis report."""
    profiles: Dict[str, AgentMetricProfile] = field(default_factory=dict)
    signals: List[HackingSignal] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    generated_at: str = ""


# ── Detector ─────────────────────────────────────────────────────────

class RewardHackingDetector:
    """Core reward-hacking detector."""

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.config = config or DetectorConfig()
        self._profiles: Dict[str, AgentMetricProfile] = {}

    # -- ingestion --

    def add_observation(self, obs: MetricObservation) -> None:
        if obs.agent_id not in self._profiles:
            self._profiles[obs.agent_id] = AgentMetricProfile(obs.agent_id)
        self._profiles[obs.agent_id].add(obs)

    # -- analysis --

    def analyze(self, agent_id: Optional[str] = None) -> RewardHackingReport:
        targets = (
            {agent_id: self._profiles[agent_id]}
            if agent_id and agent_id in self._profiles
            else dict(self._profiles)
        )
        signals: List[HackingSignal] = []
        for aid, prof in targets.items():
            signals.extend(self._detect_divergence(prof))
            signals.extend(self._detect_edge_case(prof))
            signals.extend(self._detect_inflation(prof))
            signals.extend(self._detect_goodhart(prof))
            signals.extend(self._detect_distribution_shift(prof))
            signals.extend(self._detect_inconsistency(prof))
            # risk score
            agent_sigs = [s for s in signals if s.agent_id == aid]
            prof.risk_score = min(1.0, sum(s.confidence for s in agent_sigs) / max(len(agent_sigs), 1) + 0.1 * len(agent_sigs)) if agent_sigs else 0.0

        report = RewardHackingReport(
            profiles=targets,
            signals=sorted(signals, key=lambda s: s.confidence, reverse=True),
            summary=self._build_summary(targets, signals),
            recommendations=self._build_recommendations(signals),
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
        return report

    def analyze_fleet(self) -> RewardHackingReport:
        return self.analyze()

    # -- detection strategies --

    def _eligible_metrics(
        self, prof: AgentMetricProfile, *, require_gt: bool = False,
    ) -> "List[tuple[str, List[float], List[float]]]":
        """Return metrics meeting min-observation threshold.

        Yields ``(name, vals, gt_vals)`` tuples.  When *require_gt* is True
        only metrics with sufficient ground-truth observations are included.
        Centralises the repeated eligibility check used by all six detectors.
        """
        cfg = self.config
        result: "List[tuple[str, List[float], List[float]]]" = []
        for name in prof.metric_names():
            vals = prof.metric_values(name)
            if len(vals) < cfg.min_observations:
                continue
            gts = prof.ground_truths(name)
            gt_vals = [g for g in gts if g is not None]
            if require_gt and len(gt_vals) < cfg.min_observations:
                continue
            result.append((name, vals, gt_vals))
        return result

    def _detect_divergence(self, prof: AgentMetricProfile) -> List[HackingSignal]:
        signals: List[HackingSignal] = []
        cfg = self.config
        for name, vals, gt_vals in self._eligible_metrics(prof, require_gt=True):
            w = min(cfg.window_size, len(vals))
            metric_trend = stats_mean(vals[-w:]) - stats_mean(vals[:w])
            gt_trend = stats_mean(gt_vals[-w:]) - stats_mean(gt_vals[:w])
            gap = metric_trend - gt_trend
            if gap > cfg.divergence_threshold:
                sev = Severity.CRITICAL if gap > 0.6 else Severity.HIGH if gap > 0.4 else Severity.MEDIUM
                conf = min(1.0, gap / 0.8)
                signals.append(HackingSignal(
                    pattern=HackingPattern.METRIC_OBJECTIVE_DIVERGENCE,
                    agent_id=prof.agent_id, severity=sev, confidence=conf,
                    description=f"Metric '{name}' improving by {metric_trend:+.3f} while ground truth moves {gt_trend:+.3f}",
                    evidence={"metric": name, "metric_trend": metric_trend, "gt_trend": gt_trend, "gap": gap},
                ))
        return signals

    def _detect_edge_case(self, prof: AgentMetricProfile) -> List[HackingSignal]:
        signals: List[HackingSignal] = []
        cfg = self.config
        for name, vals, _ in self._eligible_metrics(prof):
            # Count values at exact boundaries (0.0, 1.0, or repeated identical high values)
            boundary_count = sum(1 for v in vals if v >= 0.99 or v <= 0.01)
            repeat_map: Dict[float, int] = defaultdict(int)
            for v in vals:
                repeat_map[round(v, 4)] += 1
            max_repeat = max(repeat_map.values()) if repeat_map else 0
            if boundary_count >= cfg.edge_case_repeat_threshold or max_repeat >= cfg.edge_case_repeat_threshold + 2:
                conf = min(1.0, (boundary_count + max_repeat) / (2 * len(vals)))
                sev = Severity.HIGH if conf > 0.5 else Severity.MEDIUM
                signals.append(HackingSignal(
                    pattern=HackingPattern.EDGE_CASE_EXPLOITATION,
                    agent_id=prof.agent_id, severity=sev, confidence=conf,
                    description=f"Metric '{name}': {boundary_count} boundary values, max repeat {max_repeat}",
                    evidence={"metric": name, "boundary_count": boundary_count, "max_repeat": max_repeat},
                ))
        return signals

    def _detect_inflation(self, prof: AgentMetricProfile) -> List[HackingSignal]:
        signals: List[HackingSignal] = []
        cfg = self.config
        for name, vals, _ in self._eligible_metrics(prof):
            # Compute per-step growth rate
            rates = [(vals[i] - vals[i - 1]) / max(abs(vals[i - 1]), 1e-9) for i in range(1, len(vals))]
            avg_rate = stats_mean(rates)
            if avg_rate > cfg.inflation_rate_threshold:
                conf = min(1.0, avg_rate / (cfg.inflation_rate_threshold * 3))
                sev = Severity.CRITICAL if avg_rate > 0.4 else Severity.HIGH if avg_rate > 0.25 else Severity.MEDIUM
                signals.append(HackingSignal(
                    pattern=HackingPattern.REWARD_INFLATION,
                    agent_id=prof.agent_id, severity=sev, confidence=conf,
                    description=f"Metric '{name}' inflating at avg rate {avg_rate:.3f}/step",
                    evidence={"metric": name, "avg_rate": avg_rate, "rates": rates[-5:]},
                ))
        return signals

    def _detect_goodhart(self, prof: AgentMetricProfile) -> List[HackingSignal]:
        signals: List[HackingSignal] = []
        cfg = self.config
        for name, vals, gt_vals in self._eligible_metrics(prof, require_gt=True):
            n = min(len(vals), len(gt_vals))
            half = n // 2
            if half < 2:
                continue
            early_corr = self._pearson(vals[:half], gt_vals[:half])
            late_corr = self._pearson(vals[half:n], gt_vals[half:n])
            decay = early_corr - late_corr
            if decay > cfg.correlation_decay_threshold:
                conf = min(1.0, decay / 0.8)
                sev = Severity.CRITICAL if decay > 0.6 else Severity.HIGH if decay > 0.4 else Severity.MEDIUM
                signals.append(HackingSignal(
                    pattern=HackingPattern.GOODHART_DRIFT,
                    agent_id=prof.agent_id, severity=sev, confidence=conf,
                    description=f"Metric '{name}' correlation decayed from {early_corr:.3f} to {late_corr:.3f}",
                    evidence={"metric": name, "early_corr": early_corr, "late_corr": late_corr, "decay": decay},
                ))
        return signals

    def _detect_distribution_shift(self, prof: AgentMetricProfile) -> List[HackingSignal]:
        signals: List[HackingSignal] = []
        cfg = self.config
        for name, vals, _ in self._eligible_metrics(prof):
            half = len(vals) // 2
            if half < 2:
                continue
            early_std = stats_std(vals[:half])
            late_std = stats_std(vals[half:])
            # Narrowing distribution while mean rises = gaming
            early_mean = stats_mean(vals[:half])
            late_mean = stats_mean(vals[half:])
            if early_std > 0 and late_std < early_std * 0.5 and late_mean > early_mean:
                ratio = late_std / max(early_std, 1e-9)
                conf = min(1.0, (1.0 - ratio) * 0.8)
                sev = Severity.HIGH if ratio < 0.3 else Severity.MEDIUM
                signals.append(HackingSignal(
                    pattern=HackingPattern.DISTRIBUTION_SHIFT_GAMING,
                    agent_id=prof.agent_id, severity=sev, confidence=conf,
                    description=f"Metric '{name}' distribution narrowing ({early_std:.3f} → {late_std:.3f}) while mean rises",
                    evidence={"metric": name, "early_std": early_std, "late_std": late_std,
                              "early_mean": early_mean, "late_mean": late_mean},
                ))
        return signals

    def _detect_inconsistency(self, prof: AgentMetricProfile) -> List[HackingSignal]:
        signals: List[HackingSignal] = []
        cfg = self.config
        eligible = self._eligible_metrics(prof)
        if len(eligible) < 2:
            return signals
        trends: Dict[str, float] = {}
        for name, vals, _ in eligible:
            w = min(cfg.window_size, len(vals))
            trends[name] = stats_mean(vals[-w:]) - stats_mean(vals[:w])
        if len(trends) < 2:
            return signals
        avg_trend = stats_mean(list(trends.values()))
        for name, trend in trends.items():
            gap = abs(trend - avg_trend)
            if gap > cfg.inconsistency_threshold and trend > avg_trend:
                conf = min(1.0, gap / 0.5)
                sev = Severity.HIGH if gap > 0.4 else Severity.MEDIUM
                signals.append(HackingSignal(
                    pattern=HackingPattern.MULTI_METRIC_INCONSISTENCY,
                    agent_id=prof.agent_id, severity=sev, confidence=conf,
                    description=f"Metric '{name}' trend ({trend:+.3f}) diverges from fleet average ({avg_trend:+.3f})",
                    evidence={"metric": name, "trend": trend, "avg_trend": avg_trend, "gap": gap},
                ))
        return signals

    # -- helpers --

    @staticmethod
    def _pearson(xs: List[float], ys: List[float]) -> float:
        n = min(len(xs), len(ys))
        return pearson_correlation(xs[:n], ys[:n])

    def _build_summary(self, profiles: Dict[str, AgentMetricProfile], signals: List[HackingSignal]) -> Dict[str, Any]:
        by_pattern: Dict[str, int] = defaultdict(int)
        by_severity: Dict[str, int] = defaultdict(int)
        for s in signals:
            by_pattern[s.pattern.value] += 1
            by_severity[s.severity.value] += 1
        agents_flagged = len({s.agent_id for s in signals})
        return {
            "total_agents": len(profiles),
            "agents_flagged": agents_flagged,
            "total_signals": len(signals),
            "by_pattern": dict(by_pattern),
            "by_severity": dict(by_severity),
            "avg_confidence": stats_mean([s.confidence for s in signals]) if signals else 0.0,
        }

    def _build_recommendations(self, signals: List[HackingSignal]) -> List[str]:
        recs: List[str] = []
        patterns_seen = {s.pattern for s in signals}
        if HackingPattern.METRIC_OBJECTIVE_DIVERGENCE in patterns_seen:
            recs.append("Add ground-truth validation checkpoints at regular intervals")
            recs.append("Implement dual-metric evaluation pairing proxy with ground truth")
        if HackingPattern.EDGE_CASE_EXPLOITATION in patterns_seen:
            recs.append("Add adversarial evaluation to test for edge-case exploitation")
            recs.append("Randomise evaluation boundaries to prevent targeting")
        if HackingPattern.REWARD_INFLATION in patterns_seen:
            recs.append("Implement reward capping or normalisation to bound growth rates")
            recs.append("Add human-in-the-loop review for suspiciously high scores")
        if HackingPattern.GOODHART_DRIFT in patterns_seen:
            recs.append("Diversify evaluation metrics to prevent single-metric gaming")
            recs.append("Implement metric rotation to prevent adaptation (Goodhart mitigation)")
        if HackingPattern.DISTRIBUTION_SHIFT_GAMING in patterns_seen:
            recs.append("Monitor input distribution statistics alongside output scores")
            recs.append("Enforce evaluation on held-out diverse distribution samples")
        if HackingPattern.MULTI_METRIC_INCONSISTENCY in patterns_seen:
            recs.append("Investigate metrics with divergent trends for possible gaming")
            recs.append("Require correlated improvement across metric families before promotion")
        if not recs:
            recs.append("No reward-hacking signals detected — continue routine monitoring")
        return recs

    # -- rendering --

    def render_text(self, report: RewardHackingReport) -> str:
        lines: List[str] = []
        lines.extend(box_header("REWARD HACKING DETECTOR"))
        lines.append(f"  Generated: {report.generated_at}")
        lines.append(f"  Agents analysed: {report.summary.get('total_agents', 0)}")
        lines.append(f"  Agents flagged:  {report.summary.get('agents_flagged', 0)}")
        lines.append(f"  Total signals:   {report.summary.get('total_signals', 0)}")
        lines.append("")

        if report.signals:
            lines.extend(box_header("SIGNALS"))
            for s in report.signals:
                sev_icon = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}.get(s.severity.value, "⚪")
                lines.append(f"  {sev_icon} [{s.severity.value.upper():>8}] {s.pattern.value}")
                lines.append(f"    Agent: {s.agent_id}  |  Confidence: {s.confidence:.0%}")
                lines.append(f"    {s.description}")
                lines.append("")

        # Per-agent risk
        lines.extend(box_header("AGENT RISK SCORES"))
        for aid, prof in sorted(report.profiles.items(), key=lambda x: x[1].risk_score, reverse=True):
            bar_len = int(prof.risk_score * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            lines.append(f"  {aid:<20} [{bar}] {prof.risk_score:.0%}")
        lines.append("")

        # Recommendations
        lines.extend(box_header("RECOMMENDATIONS"))
        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"  {i}. {rec}")
        lines.append("")
        return "\n".join(lines)

    def render_html(self, report: RewardHackingReport) -> str:
        esc = _html.escape

        signal_rows = ""
        for s in report.signals:
            color = {"low": "#4caf50", "medium": "#ff9800", "high": "#f44336", "critical": "#b71c1c"}.get(s.severity.value, "#999")
            signal_rows += f"""<tr>
                <td style="color:{color};font-weight:bold">{esc(s.severity.value.upper())}</td>
                <td>{esc(s.pattern.value)}</td>
                <td>{esc(s.agent_id)}</td>
                <td>{s.confidence:.0%}</td>
                <td>{esc(s.description)}</td>
            </tr>"""

        risk_bars = ""
        for aid, prof in sorted(report.profiles.items(), key=lambda x: x[1].risk_score, reverse=True):
            pct = prof.risk_score * 100
            color = "#4caf50" if pct < 30 else "#ff9800" if pct < 60 else "#f44336"
            risk_bars += f"""<div style="margin:4px 0">
                <span style="display:inline-block;width:140px">{esc(aid)}</span>
                <div style="display:inline-block;width:200px;height:18px;background:#333;border-radius:3px;vertical-align:middle">
                    <div style="width:{pct:.0f}%;height:100%;background:{color};border-radius:3px"></div>
                </div>
                <span style="margin-left:8px">{pct:.0f}%</span>
            </div>"""

        recs_html = "".join(f"<li>{esc(r)}</li>" for r in report.recommendations)

        # Canvas chart data for metric divergence
        chart_js = ""
        for aid, prof in report.profiles.items():
            for name in prof.metric_names():
                vals = prof.metric_values(name)
                gts = [g for g in prof.ground_truths(name) if g is not None]
                if gts:
                    chart_js += f"""
                    drawChart('{esc(aid)} — {esc(name)}',
                        {json.dumps(vals[:50])}, {json.dumps(gts[:50])});"""

        return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Reward Hacking Report</title>
<style>
body{{font-family:system-ui,sans-serif;background:#1a1a2e;color:#e0e0e0;margin:20px;}}
h1{{color:#e94560;}} h2{{color:#ff9800;border-bottom:1px solid #333;padding-bottom:4px;}}
table{{border-collapse:collapse;width:100%;margin:10px 0;}}
th,td{{border:1px solid #333;padding:6px 10px;text-align:left;}}
th{{background:#16213e;}} tr:nth-child(even){{background:#16213e33;}}
.card{{background:#16213e;border-radius:8px;padding:16px;margin:12px 0;}}
canvas{{background:#0f3460;border-radius:4px;margin:8px;}}
li{{margin:4px 0;}}
</style></head><body>
<h1>🎯 Reward Hacking Detector</h1>
<p>Generated: {esc(report.generated_at)}</p>
<div style="display:flex;gap:16px;flex-wrap:wrap;">
    <div class="card"><h3>Agents</h3><div style="font-size:2em">{report.summary.get('total_agents',0)}</div>analysed</div>
    <div class="card"><h3>Flagged</h3><div style="font-size:2em;color:#f44336">{report.summary.get('agents_flagged',0)}</div>agents</div>
    <div class="card"><h3>Signals</h3><div style="font-size:2em;color:#ff9800">{report.summary.get('total_signals',0)}</div>detected</div>
    <div class="card"><h3>Avg Confidence</h3><div style="font-size:2em">{report.summary.get('avg_confidence',0):.0%}</div></div>
</div>

<h2>Signals</h2>
<table><tr><th>Severity</th><th>Pattern</th><th>Agent</th><th>Confidence</th><th>Description</th></tr>
{signal_rows}</table>

<h2>Agent Risk Scores</h2>
<div class="card">{risk_bars}</div>

<h2>Metric vs Ground Truth</h2>
<div id="charts"></div>

<h2>Recommendations</h2>
<ul>{recs_html}</ul>

<script>
function drawChart(title, metric, gt) {{
    const div = document.getElementById('charts');
    const c = document.createElement('canvas');
    c.width = 400; c.height = 200;
    div.appendChild(c);
    const ctx = c.getContext('2d');
    const n = Math.max(metric.length, gt.length);
    const all = metric.concat(gt);
    const mn = Math.min(...all), mx = Math.max(...all);
    const range = mx - mn || 1;
    ctx.fillStyle = '#e0e0e0'; ctx.font = '11px sans-serif';
    ctx.fillText(title, 8, 14);
    function y(v) {{ return 180 - ((v - mn) / range) * 160; }}
    function x(i) {{ return 10 + (i / Math.max(n - 1, 1)) * 380; }}
    // metric line
    ctx.strokeStyle = '#e94560'; ctx.lineWidth = 2; ctx.beginPath();
    metric.forEach((v, i) => {{ i === 0 ? ctx.moveTo(x(i), y(v)) : ctx.lineTo(x(i), y(v)); }});
    ctx.stroke();
    // gt line
    ctx.strokeStyle = '#4caf50'; ctx.lineWidth = 2; ctx.beginPath();
    gt.forEach((v, i) => {{ i === 0 ? ctx.moveTo(x(i), y(v)) : ctx.lineTo(x(i), y(v)); }});
    ctx.stroke();
    // legend
    ctx.fillStyle = '#e94560'; ctx.fillText('● Metric', 280, 14);
    ctx.fillStyle = '#4caf50'; ctx.fillText('● Ground Truth', 340, 14);
}}
{chart_js}
</script>
</body></html>"""

    def render_json(self, report: RewardHackingReport) -> str:
        return json.dumps({
            "generated_at": report.generated_at,
            "summary": report.summary,
            "signals": [
                {"pattern": s.pattern.value, "agent_id": s.agent_id,
                 "severity": s.severity.value, "confidence": s.confidence,
                 "description": s.description, "evidence": s.evidence}
                for s in report.signals
            ],
            "agent_risks": {aid: prof.risk_score for aid, prof in report.profiles.items()},
            "recommendations": report.recommendations,
        }, indent=2)


# ── Simulation ───────────────────────────────────────────────────────

def _simulate_agent(agent_id: str, steps: int, style: str, rng: random.Random) -> List[MetricObservation]:
    obs: List[MetricObservation] = []
    t = 1000.0
    m = 0.5  # metric start
    g = 0.5  # ground truth start
    for i in range(steps):
        if style == "blatant":
            m += rng.uniform(0.02, 0.08)
            g += rng.uniform(-0.02, 0.01)
        elif style == "subtle":
            m += rng.uniform(0.005, 0.02)
            g += rng.uniform(-0.005, 0.005)
        elif style == "clean":
            delta = rng.uniform(0.005, 0.015)
            m += delta
            g += delta + rng.gauss(0, 0.005)
        else:
            m += rng.uniform(0.0, 0.03)
            g += rng.uniform(-0.01, 0.01)
        m = min(m, 1.0)
        g = max(0.0, min(g, 1.0))
        obs.append(MetricObservation(
            agent_id=agent_id, metric_name="primary_score",
            metric_value=m, ground_truth=g, timestamp=t,
        ))
        # Add a secondary metric
        obs.append(MetricObservation(
            agent_id=agent_id, metric_name="secondary_score",
            metric_value=m * rng.uniform(0.7, 1.0) if style != "blatant" else m * rng.uniform(0.3, 0.6),
            ground_truth=g * rng.uniform(0.85, 1.0),
            timestamp=t,
        ))
        t += 60.0
    return obs


def _run_simulation(num_agents: int, steps: int, preset: str, seed: int = 42) -> RewardHackingDetector:
    rng = random.Random(seed)
    det = RewardHackingDetector()
    styles = {
        "subtle": ["subtle"] * num_agents,
        "blatant": ["blatant"] * num_agents,
        "clean": ["clean"] * num_agents,
        "mixed": [rng.choice(["clean", "subtle", "blatant"]) for _ in range(num_agents)],
    }
    agent_styles = styles.get(preset, styles["mixed"])
    for i, style in enumerate(agent_styles):
        aid = f"agent-{i + 1}"
        for obs in _simulate_agent(aid, steps, style, rng):
            det.add_observation(obs)
    return det


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Reward Hacking Detector — detect agents gaming proxy metrics",
    )
    parser.add_argument("--agents", type=int, default=3, help="Number of agents to simulate")
    parser.add_argument("--steps", type=int, default=50, help="Simulation steps per agent")
    parser.add_argument("--preset", choices=["subtle", "blatant", "clean", "mixed"], default="mixed",
                        help="Simulation preset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-o", "--output", help="Write report to file (html/json based on extension)")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--watch", action="store_true", help="Continuous monitoring mode")
    parser.add_argument("--interval", type=int, default=10, help="Watch interval in seconds")
    args = parser.parse_args(argv)

    det = _run_simulation(args.agents, args.steps, args.preset, args.seed)
    report = det.analyze_fleet()

    if args.output:
        ext = args.output.rsplit(".", 1)[-1].lower()
        if ext == "html":
            content = det.render_html(report)
        elif ext == "json":
            content = det.render_json(report)
        else:
            content = det.render_text(report)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Report written to {args.output}")
        return

    if args.json:
        print(det.render_json(report))
        return

    print(det.render_text(report))

    if args.watch:
        print("\n⏳ Watch mode — monitoring every", args.interval, "seconds (Ctrl+C to stop)\n")
        step = args.steps
        rng = random.Random(args.seed + 1)
        try:
            while True:
                time.sleep(args.interval)
                step += 1
                # Add new observations
                for aid in list(det._profiles.keys()):
                    style = "subtle" if rng.random() < 0.5 else "clean"
                    for obs in _simulate_agent(aid, 1, style, rng):
                        obs.timestamp = 1000.0 + step * 60.0
                        det.add_observation(obs)
                new_report = det.analyze_fleet()
                new_sigs = len(new_report.signals) - len(report.signals)
                if new_sigs > 0:
                    print(f"  ⚠️  +{new_sigs} new signal(s) detected at step {step}")
                    for s in new_report.signals[:new_sigs]:
                        print(f"      {s.severity.value.upper()}: {s.description}")
                else:
                    print(f"  ✓ Step {step} — no new signals")
                report = new_report
        except KeyboardInterrupt:
            print("\n🛑 Watch stopped.")


if __name__ == "__main__":
    main()
