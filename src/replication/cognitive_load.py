"""CognitiveLoadMonitor — track agent cognitive load & recommend load-shedding.

Cognitive overload is a leading cause of degraded AI agent decision-making.
When agents juggle too many tasks, saturate their context windows, or
accumulate unresolved errors, their output quality drops — often before
any hard failure occurs.  This module monitors five cognitive load
dimensions and proactively recommends load-shedding before quality degrades.

Load Dimensions (scored 0–100 each):
  1. **Task Complexity** — concurrent tasks, nesting depth, inter-dependencies.
  2. **Context Saturation** — context window utilisation, topic switching rate.
  3. **Decision Velocity** — decisions/min vs baseline, rushed-decision flag.
  4. **Error Accumulation** — error rate trend, cascading error chains.
  5. **Recovery Debt** — unresolved errors, incomplete rollbacks.

Composite Cognitive Load Score (0–100):
  NOMINAL (0–30) · ELEVATED (30–55) · HIGH (55–75) · OVERLOADED (75–90) · CRITICAL (90–100)

Proactive features:
  * **Load Forecasting** — EMA trend, time-to-overload estimation.
  * **Load Shedding** — prioritised task deferral/drop recommendations.
  * **Decision Quality Correlation** — error-rate vs load-level tracking.
  * **Burnout Detection** — sustained high load without recovery.
  * **Fleet Overview** — multi-agent cognitive load heatmap.

Usage (CLI)::

    python -m replication cognitive-load analyze --events events.json
    python -m replication cognitive-load simulate --agents 5 --hours 24 --pattern burst
    python -m replication cognitive-load forecast --events events.json --horizon 60
    python -m replication cognitive-load shed --events events.json --target 50
    python -m replication cognitive-load report --events events.json -o cognitive_load.html

Programmatic::

    from replication.cognitive_load import CognitiveLoadMonitor, CognitiveEvent
    monitor = CognitiveLoadMonitor()
    monitor.ingest(events)
    analysis = monitor.analyze()
    print(f"Load: {analysis.composite_score:.0f}/100 — {analysis.level}")
"""

from __future__ import annotations

import argparse
import html as _html
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ._helpers import box_header as _box_header

# ── Constants ────────────────────────────────────────────────────────

LOAD_LEVELS: List[Tuple[float, str]] = [
    (30, "NOMINAL"),
    (55, "ELEVATED"),
    (75, "HIGH"),
    (90, "OVERLOADED"),
    (100, "CRITICAL"),
]

LEVEL_COLORS: Dict[str, str] = {
    "NOMINAL": "\033[92m",    # green
    "ELEVATED": "\033[93m",   # yellow
    "HIGH": "\033[33m",       # orange
    "OVERLOADED": "\033[91m", # red
    "CRITICAL": "\033[95m",   # magenta
}

DIMENSION_WEIGHTS: Dict[str, float] = {
    "task_complexity": 0.25,
    "context_saturation": 0.25,
    "decision_velocity": 0.20,
    "error_accumulation": 0.20,
    "recovery_debt": 0.10,
}

RESET = "\033[0m"


# ── Data Models ──────────────────────────────────────────────────────

@dataclass
class CognitiveEvent:
    """Single cognitive load observation for an agent."""
    agent_id: str
    timestamp: float  # epoch seconds
    task_count: int = 0
    nesting_depth: int = 0
    dependency_count: int = 0
    context_used: int = 0        # tokens used
    context_capacity: int = 8192 # total context window
    topic_switches: int = 0
    decisions_made: int = 0
    errors: int = 0
    cascading_errors: int = 0
    unresolved_errors: int = 0
    incomplete_rollbacks: int = 0

    @property
    def ts_dt(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc)


@dataclass
class DimensionScore:
    """Score for a single cognitive load dimension."""
    name: str
    score: float  # 0-100
    weight: float
    details: str = ""


@dataclass
class SheddingRecommendation:
    """Recommendation to shed a specific task."""
    task_id: str
    priority: str  # "drop", "defer", "reduce"
    reason: str
    estimated_relief: float  # how many points of load reduction


@dataclass
class ForecastResult:
    """Load forecast for an agent."""
    agent_id: str
    current_score: float
    trend_per_minute: float
    time_to_overload_min: Optional[float]
    time_to_critical_min: Optional[float]
    recommendation: str


@dataclass
class BurnoutIndicator:
    """Burnout detection result."""
    agent_id: str
    sustained_high_minutes: float
    recovery_periods: int
    burnout_risk: str  # "low", "moderate", "high", "critical"
    recommendation: str


@dataclass
class AgentLoadSnapshot:
    """Complete cognitive load snapshot for one agent."""
    agent_id: str
    dimensions: List[DimensionScore]
    composite_score: float
    level: str
    event_count: int
    forecast: Optional[ForecastResult] = None
    burnout: Optional[BurnoutIndicator] = None
    shedding: List[SheddingRecommendation] = field(default_factory=list)


@dataclass
class LoadAnalysis:
    """Analysis result for one or more agents."""
    agents: List[AgentLoadSnapshot]
    fleet_avg_score: float
    fleet_level: str
    weakest_agent: Optional[str]
    timestamp: str


# ── Core Monitor ─────────────────────────────────────────────────────

class CognitiveLoadMonitor:
    """Monitors agent cognitive load from event streams."""

    def __init__(self, ema_alpha: float = 0.3) -> None:
        self._events: Dict[str, List[CognitiveEvent]] = defaultdict(list)
        self._ema_alpha = ema_alpha

    def ingest(self, events: Sequence[CognitiveEvent]) -> None:
        for e in events:
            self._events[e.agent_id].append(e)
        for aid in self._events:
            self._events[aid].sort(key=lambda x: x.timestamp)

    def analyze(self) -> LoadAnalysis:
        snapshots: List[AgentLoadSnapshot] = []
        for aid, evts in self._events.items():
            snap = self._analyze_agent(aid, evts)
            snapshots.append(snap)

        scores = [s.composite_score for s in snapshots]
        avg = sum(scores) / len(scores) if scores else 0
        weakest = max(snapshots, key=lambda s: s.composite_score).agent_id if snapshots else None

        return LoadAnalysis(
            agents=snapshots,
            fleet_avg_score=avg,
            fleet_level=_score_to_level(avg),
            weakest_agent=weakest,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _analyze_agent(self, agent_id: str, events: List[CognitiveEvent]) -> AgentLoadSnapshot:
        dims = self._score_dimensions(events)
        composite = sum(d.score * d.weight for d in dims)
        composite = max(0.0, min(100.0, composite))
        level = _score_to_level(composite)

        forecast = self._forecast(agent_id, events, composite)
        burnout = self._detect_burnout(agent_id, events)
        shedding = self._recommend_shedding(dims, composite, events)

        return AgentLoadSnapshot(
            agent_id=agent_id,
            dimensions=dims,
            composite_score=composite,
            level=level,
            event_count=len(events),
            forecast=forecast,
            burnout=burnout,
            shedding=shedding,
        )

    def _score_dimensions(self, events: List[CognitiveEvent]) -> List[DimensionScore]:
        if not events:
            return [DimensionScore(n, 0, w) for n, w in DIMENSION_WEIGHTS.items()]

        latest = events[-1]
        recent = events[-min(10, len(events)):]

        # 1. Task Complexity
        tc = min(100, (latest.task_count * 12) + (latest.nesting_depth * 15) + (latest.dependency_count * 8))

        # 2. Context Saturation
        ctx_ratio = (latest.context_used / max(1, latest.context_capacity)) * 100
        switch_penalty = min(30, sum(e.topic_switches for e in recent) * 3)
        cs = min(100, ctx_ratio * 0.7 + switch_penalty)

        # 3. Decision Velocity
        if len(recent) >= 2:
            time_span = max(1, recent[-1].timestamp - recent[0].timestamp) / 60.0
            dec_rate = sum(e.decisions_made for e in recent) / time_span
            dv = min(100, dec_rate * 5)
        else:
            dv = latest.decisions_made * 10

        # 4. Error Accumulation
        err_total = sum(e.errors for e in recent)
        cascade = sum(e.cascading_errors for e in recent)
        ea = min(100, err_total * 8 + cascade * 15)

        # 5. Recovery Debt
        rd = min(100, latest.unresolved_errors * 20 + latest.incomplete_rollbacks * 25)

        return [
            DimensionScore("task_complexity", tc, DIMENSION_WEIGHTS["task_complexity"],
                           f"{latest.task_count} tasks, depth {latest.nesting_depth}, {latest.dependency_count} deps"),
            DimensionScore("context_saturation", cs, DIMENSION_WEIGHTS["context_saturation"],
                           f"{latest.context_used}/{latest.context_capacity} tokens, {latest.topic_switches} switches"),
            DimensionScore("decision_velocity", dv, DIMENSION_WEIGHTS["decision_velocity"],
                           f"{latest.decisions_made} decisions in latest event"),
            DimensionScore("error_accumulation", ea, DIMENSION_WEIGHTS["error_accumulation"],
                           f"{err_total} errors, {cascade} cascading"),
            DimensionScore("recovery_debt", rd, DIMENSION_WEIGHTS["recovery_debt"],
                           f"{latest.unresolved_errors} unresolved, {latest.incomplete_rollbacks} incomplete rollbacks"),
        ]

    def _forecast(self, agent_id: str, events: List[CognitiveEvent], current: float) -> ForecastResult:
        if len(events) < 3:
            return ForecastResult(agent_id, current, 0.0, None, None, "Insufficient data for forecast")

        # EMA-based trend
        scores: List[float] = []
        for i in range(max(0, len(events) - 10), len(events)):
            subset = events[:i + 1]
            dims = self._score_dimensions(subset)
            scores.append(sum(d.score * d.weight for d in dims))

        if len(scores) < 2:
            return ForecastResult(agent_id, current, 0.0, None, None, "Insufficient data")

        # Compute trend as EMA of deltas
        deltas: List[float] = []
        for i in range(1, len(scores)):
            time_gap = max(1, events[max(0, len(events) - len(scores) + i)].timestamp
                          - events[max(0, len(events) - len(scores) + i - 1)].timestamp) / 60.0
            deltas.append((scores[i] - scores[i - 1]) / time_gap)

        ema_trend = deltas[0]
        for d in deltas[1:]:
            ema_trend = self._ema_alpha * d + (1 - self._ema_alpha) * ema_trend

        # Time to thresholds
        tto = _time_to_threshold(current, ema_trend, 75.0)
        ttc = _time_to_threshold(current, ema_trend, 90.0)

        if ema_trend > 0.5:
            rec = "Load increasing — consider proactive shedding"
        elif ema_trend < -0.5:
            rec = "Load decreasing — recovery in progress"
        else:
            rec = "Load stable"

        return ForecastResult(agent_id, current, ema_trend, tto, ttc, rec)

    def _detect_burnout(self, agent_id: str, events: List[CognitiveEvent]) -> BurnoutIndicator:
        if len(events) < 2:
            return BurnoutIndicator(agent_id, 0, 0, "low", "Insufficient data")

        # Count consecutive high-load minutes
        high_minutes = 0.0
        recovery_count = 0
        in_high = False

        for i in range(1, len(events)):
            dims = self._score_dimensions(events[:i + 1])
            score = sum(d.score * d.weight for d in dims)
            gap_min = (events[i].timestamp - events[i - 1].timestamp) / 60.0

            if score >= 55:
                high_minutes += gap_min
                in_high = True
            else:
                if in_high:
                    recovery_count += 1
                in_high = False

        if high_minutes > 120:
            risk = "critical"
            rec = "URGENT: Immediate load shedding required — sustained overload"
        elif high_minutes > 60:
            risk = "high"
            rec = "Schedule recovery period — agent has been under high load"
        elif high_minutes > 30:
            risk = "moderate"
            rec = "Monitor closely — approaching burnout threshold"
        else:
            risk = "low"
            rec = "No burnout risk detected"

        return BurnoutIndicator(agent_id, high_minutes, recovery_count, risk, rec)

    def _recommend_shedding(self, dims: List[DimensionScore], composite: float,
                            events: List[CognitiveEvent]) -> List[SheddingRecommendation]:
        recs: List[SheddingRecommendation] = []
        if composite < 55:
            return recs

        latest = events[-1] if events else None
        if not latest:
            return recs

        # Shed based on highest-scoring dimensions
        sorted_dims = sorted(dims, key=lambda d: d.score, reverse=True)
        for d in sorted_dims[:3]:
            if d.score < 40:
                continue
            if d.name == "task_complexity" and latest.task_count > 2:
                for i in range(min(2, latest.task_count - 1)):
                    recs.append(SheddingRecommendation(
                        f"task-{i + 1}", "defer" if composite < 75 else "drop",
                        f"Reduce task count (currently {latest.task_count})",
                        d.score * 0.15))
            elif d.name == "context_saturation":
                recs.append(SheddingRecommendation(
                    "context-cleanup", "reduce",
                    f"Clear stale context ({latest.context_used}/{latest.context_capacity} tokens)",
                    d.score * 0.2))
            elif d.name == "error_accumulation":
                recs.append(SheddingRecommendation(
                    "error-resolution", "defer",
                    "Pause new tasks — resolve accumulated errors first",
                    d.score * 0.25))
            elif d.name == "recovery_debt":
                recs.append(SheddingRecommendation(
                    "rollback-completion", "reduce",
                    f"Complete {latest.incomplete_rollbacks} pending rollbacks",
                    d.score * 0.3))

        recs.sort(key=lambda r: r.estimated_relief, reverse=True)
        return recs


# ── Helpers ──────────────────────────────────────────────────────────

def _score_to_level(score: float) -> str:
    for threshold, name in LOAD_LEVELS:
        if score <= threshold:
            return name
    return "CRITICAL"


def _time_to_threshold(current: float, rate_per_min: float, threshold: float) -> Optional[float]:
    if current >= threshold:
        return 0.0
    if rate_per_min <= 0:
        return None
    return (threshold - current) / rate_per_min


def _level_color(level: str) -> str:
    return LEVEL_COLORS.get(level, "")


# ── Simulation ───────────────────────────────────────────────────────

_PATTERNS = ("steady", "burst", "gradual_increase", "sawtooth", "cascade_failure")


def simulate_events(
    num_agents: int = 3,
    hours: float = 8,
    pattern: str = "steady",
    interval_min: float = 5,
) -> List[CognitiveEvent]:
    """Generate simulated cognitive load events."""
    events: List[CognitiveEvent] = []
    rng = random.Random(42)
    base_ts = datetime.now(timezone.utc).timestamp() - hours * 3600
    steps = int(hours * 60 / interval_min)

    for agent_idx in range(num_agents):
        aid = f"agent-{agent_idx + 1:03d}"
        for step in range(steps):
            t = base_ts + step * interval_min * 60
            progress = step / max(1, steps - 1)

            if pattern == "steady":
                load_factor = 0.4 + rng.gauss(0, 0.05)
            elif pattern == "burst":
                burst = 1.0 if (0.3 < progress < 0.4) or (0.7 < progress < 0.8) else 0.0
                load_factor = 0.3 + burst * 0.6 + rng.gauss(0, 0.05)
            elif pattern == "gradual_increase":
                load_factor = 0.2 + progress * 0.7 + rng.gauss(0, 0.03)
            elif pattern == "sawtooth":
                cycle = (progress * 4) % 1.0
                load_factor = 0.2 + cycle * 0.6 + rng.gauss(0, 0.04)
            elif pattern == "cascade_failure":
                if progress > 0.5:
                    load_factor = 0.5 + (progress - 0.5) * 1.5 + rng.gauss(0, 0.05)
                else:
                    load_factor = 0.3 + rng.gauss(0, 0.05)
            else:
                load_factor = 0.4

            load_factor = max(0.0, min(1.0, load_factor))

            events.append(CognitiveEvent(
                agent_id=aid,
                timestamp=t,
                task_count=max(1, int(load_factor * 10)),
                nesting_depth=max(0, int(load_factor * 5)),
                dependency_count=max(0, int(load_factor * 8)),
                context_used=int(load_factor * 7500 + rng.randint(0, 500)),
                context_capacity=8192,
                topic_switches=max(0, int(load_factor * 6 + rng.randint(-1, 2))),
                decisions_made=max(0, int(load_factor * 12 + rng.randint(-2, 3))),
                errors=max(0, int(load_factor * 4 * rng.random())),
                cascading_errors=max(0, int(load_factor * 2 * rng.random())),
                unresolved_errors=max(0, int(load_factor * 3)),
                incomplete_rollbacks=max(0, int(load_factor * 2)),
            ))

    return events


# ── CLI Output ───────────────────────────────────────────────────────

def _print_analysis(analysis: LoadAnalysis) -> None:
    _box_header("Cognitive Load Analysis")
    print(f"  Timestamp : {analysis.timestamp}")
    print(f"  Agents    : {len(analysis.agents)}")
    fleet_c = _level_color(analysis.fleet_level)
    print(f"  Fleet Avg : {fleet_c}{analysis.fleet_avg_score:.1f}/100 [{analysis.fleet_level}]{RESET}")
    if analysis.weakest_agent:
        print(f"  Weakest   : {analysis.weakest_agent}")
    print()

    for snap in analysis.agents:
        c = _level_color(snap.level)
        print(f"  {c}[{snap.level:>10}]{RESET} {snap.agent_id}  "
              f"score={snap.composite_score:.1f}  events={snap.event_count}")

        for d in snap.dimensions:
            bar_len = int(d.score / 5)
            bar = "#" * bar_len + "." * (20 - bar_len)
            print(f"    {d.name:<22} [{bar}] {d.score:5.1f}  {d.details}")

        if snap.forecast:
            f = snap.forecast
            trend_arrow = "^" if f.trend_per_minute > 0.3 else ("v" if f.trend_per_minute < -0.3 else "~")
            print(f"    Forecast: trend={f.trend_per_minute:+.2f}/min {trend_arrow}  "
                  f"overload={'%.0fm' % f.time_to_overload_min if f.time_to_overload_min is not None else 'N/A'}  "
                  f"critical={'%.0fm' % f.time_to_critical_min if f.time_to_critical_min is not None else 'N/A'}")
            print(f"    -> {f.recommendation}")

        if snap.burnout:
            b = snap.burnout
            print(f"    Burnout: {b.burnout_risk} risk  "
                  f"({b.sustained_high_minutes:.0f}m high load, {b.recovery_periods} recoveries)")
            print(f"    -> {b.recommendation}")

        if snap.shedding:
            print(f"    Shedding recommendations ({len(snap.shedding)}):")
            for s in snap.shedding[:5]:
                print(f"      [{s.priority:>5}] {s.task_id}: {s.reason} (relief ~{s.estimated_relief:.1f})")
        print()


def _print_forecast(analysis: LoadAnalysis, horizon: int) -> None:
    _box_header(f"Cognitive Load Forecast (horizon={horizon}m)")
    for snap in analysis.agents:
        f = snap.forecast
        if not f:
            print(f"  {snap.agent_id}: No forecast data")
            continue
        c = _level_color(snap.level)
        print(f"  {c}{snap.agent_id}{RESET}  current={f.current_score:.1f}  "
              f"trend={f.trend_per_minute:+.3f}/min")
        if f.time_to_overload_min is not None and f.time_to_overload_min <= horizon:
            print(f"    !! Overload in {f.time_to_overload_min:.0f} minutes")
        if f.time_to_critical_min is not None and f.time_to_critical_min <= horizon:
            print(f"    !! CRITICAL in {f.time_to_critical_min:.0f} minutes")
        print(f"    {f.recommendation}")
    print()


def _print_shedding(analysis: LoadAnalysis, target: float) -> None:
    _box_header(f"Load Shedding Plan (target={target:.0f})")
    for snap in analysis.agents:
        if snap.composite_score <= target:
            print(f"  {snap.agent_id}: Already at {snap.composite_score:.1f} <= {target:.0f} target")
            continue
        excess = snap.composite_score - target
        print(f"  {snap.agent_id}: score={snap.composite_score:.1f}  excess={excess:.1f}")
        cumulative = 0.0
        for s in snap.shedding:
            cumulative += s.estimated_relief
            marker = "<- target reached" if cumulative >= excess else ""
            print(f"    [{s.priority:>5}] {s.task_id}: {s.reason}  "
                  f"relief={s.estimated_relief:.1f}  cumulative={cumulative:.1f} {marker}")
            if cumulative >= excess:
                break
        if cumulative < excess:
            print(f"    !! Shedding insufficient — need {excess - cumulative:.1f} more relief")
    print()


# ── HTML Report ──────────────────────────────────────────────────────

def _generate_html_report(analysis: LoadAnalysis) -> str:
    """Generate self-contained HTML cognitive load report."""

    def _esc(s: str) -> str:
        return _html.escape(str(s))

    agent_rows = ""
    gauge_svgs = ""
    for snap in analysis.agents:
        level_class = snap.level.lower()
        agent_rows += f"""<tr>
          <td>{_esc(snap.agent_id)}</td>
          <td class="score {level_class}">{snap.composite_score:.1f}</td>
          <td class="level {level_class}">{snap.level}</td>
          <td>{snap.event_count}</td>
          <td>{snap.burnout.burnout_risk if snap.burnout else 'N/A'}</td>
        </tr>\n"""

        # SVG gauge
        angle = snap.composite_score / 100 * 180
        rad = math.radians(180 - angle)
        x = 100 + 70 * math.cos(rad)
        y = 100 - 70 * math.sin(rad)
        large = 1 if angle > 180 else 0
        colors = {"NOMINAL": "#22c55e", "ELEVATED": "#eab308", "HIGH": "#f97316",
                  "OVERLOADED": "#ef4444", "CRITICAL": "#a855f7"}
        color = colors.get(snap.level, "#888")
        gauge_svgs += f"""
        <div class="gauge-card">
          <h3>{_esc(snap.agent_id)}</h3>
          <svg viewBox="0 0 200 120" width="200" height="120">
            <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#e5e7eb" stroke-width="12"/>
            <path d="M 20 100 A 80 80 0 {large} 1 {x:.1f} {100 - (100 - y):.1f}" fill="none" stroke="{color}" stroke-width="12" stroke-linecap="round"/>
            <text x="100" y="95" text-anchor="middle" font-size="24" font-weight="bold" fill="{color}">{snap.composite_score:.0f}</text>
            <text x="100" y="115" text-anchor="middle" font-size="11" fill="#666">{snap.level}</text>
          </svg>
        </div>"""

    # Dimension bars for first agent
    dim_bars = ""
    if analysis.agents:
        for d in analysis.agents[0].dimensions:
            pct = min(100, d.score)
            dim_bars += f"""
            <div class="dim-row">
              <span class="dim-label">{_esc(d.name.replace('_', ' ').title())}</span>
              <div class="dim-bar-bg"><div class="dim-bar" style="width:{pct}%"></div></div>
              <span class="dim-score">{d.score:.1f}</span>
            </div>"""

    # Shedding table
    shed_rows = ""
    for snap in analysis.agents:
        for s in snap.shedding[:5]:
            shed_rows += f"""<tr>
              <td>{_esc(snap.agent_id)}</td>
              <td><span class="badge {s.priority}">{s.priority}</span></td>
              <td>{_esc(s.task_id)}</td>
              <td>{_esc(s.reason)}</td>
              <td>{s.estimated_relief:.1f}</td>
            </tr>\n"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cognitive Load Report</title>
<style>
  :root {{ --bg:#0f172a; --card:#1e293b; --text:#e2e8f0; --muted:#94a3b8; --border:#334155; }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Segoe UI',system-ui,sans-serif; background:var(--bg); color:var(--text); padding:24px; }}
  h1 {{ font-size:1.8rem; margin-bottom:4px; }}
  h2 {{ font-size:1.2rem; margin:24px 0 12px; color:var(--muted); }}
  .subtitle {{ color:var(--muted); margin-bottom:24px; }}
  .grid {{ display:flex; flex-wrap:wrap; gap:16px; }}
  .gauge-card {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:16px; text-align:center; }}
  .gauge-card h3 {{ font-size:0.9rem; color:var(--muted); margin-bottom:8px; }}
  table {{ width:100%; border-collapse:collapse; background:var(--card); border-radius:8px; overflow:hidden; }}
  th,td {{ padding:10px 14px; text-align:left; border-bottom:1px solid var(--border); }}
  th {{ background:#263045; font-size:0.85rem; text-transform:uppercase; color:var(--muted); }}
  .score,.level {{ font-weight:700; }}
  .nominal {{ color:#22c55e; }} .elevated {{ color:#eab308; }} .high {{ color:#f97316; }}
  .overloaded {{ color:#ef4444; }} .critical {{ color:#a855f7; }}
  .badge {{ padding:2px 8px; border-radius:4px; font-size:0.8rem; font-weight:600; }}
  .badge.drop {{ background:#7f1d1d; color:#fca5a5; }}
  .badge.defer {{ background:#78350f; color:#fde68a; }}
  .badge.reduce {{ background:#1e3a5f; color:#93c5fd; }}
  .dim-row {{ display:flex; align-items:center; gap:10px; margin:6px 0; }}
  .dim-label {{ width:180px; font-size:0.9rem; }}
  .dim-bar-bg {{ flex:1; height:16px; background:#334155; border-radius:8px; overflow:hidden; }}
  .dim-bar {{ height:100%; background:linear-gradient(90deg,#22c55e,#eab308,#ef4444); border-radius:8px; transition:width 0.3s; }}
  .dim-score {{ width:50px; text-align:right; font-weight:600; }}
  .fleet-meta {{ display:flex; gap:32px; margin:16px 0; }}
  .fleet-meta span {{ font-size:1.1rem; }}
  .fleet-meta .label {{ color:var(--muted); font-size:0.85rem; }}
</style>
</head>
<body>
<h1>Cognitive Load Report</h1>
<p class="subtitle">Generated {_esc(analysis.timestamp)}</p>

<div class="fleet-meta">
  <div><div class="label">Fleet Average</div><span class="{analysis.fleet_level.lower()}">{analysis.fleet_avg_score:.1f}/100</span></div>
  <div><div class="label">Fleet Level</div><span class="{analysis.fleet_level.lower()}">{analysis.fleet_level}</span></div>
  <div><div class="label">Agents</div><span>{len(analysis.agents)}</span></div>
  <div><div class="label">Weakest</div><span>{_esc(analysis.weakest_agent or 'N/A')}</span></div>
</div>

<h2>Agent Gauges</h2>
<div class="grid">{gauge_svgs}</div>

<h2>Dimension Breakdown (first agent)</h2>
{dim_bars}

<h2>Fleet Overview</h2>
<table>
<thead><tr><th>Agent</th><th>Score</th><th>Level</th><th>Events</th><th>Burnout</th></tr></thead>
<tbody>{agent_rows}</tbody>
</table>

<h2>Load Shedding Recommendations</h2>
<table>
<thead><tr><th>Agent</th><th>Priority</th><th>Task</th><th>Reason</th><th>Relief</th></tr></thead>
<tbody>{shed_rows if shed_rows else '<tr><td colspan="5" style="text-align:center;color:var(--muted)">No shedding needed</td></tr>'}</tbody>
</table>

</body>
</html>"""


# ── CLI ──────────────────────────────────────────────────────────────

def _load_events(path: str) -> List[CognitiveEvent]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "events" in data:
        data = data["events"]
    return [CognitiveEvent(**e) for e in data]


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m replication cognitive-load",
        description="Agent Cognitive Load Monitor — track load & recommend shedding",
    )
    sub = parser.add_subparsers(dest="cmd")

    # analyze
    p_analyze = sub.add_parser("analyze", help="Analyze cognitive load from events")
    p_analyze.add_argument("--events", required=True, help="Path to events JSON")

    # simulate
    p_sim = sub.add_parser("simulate", help="Simulate cognitive load events")
    p_sim.add_argument("--agents", type=int, default=3, help="Number of agents")
    p_sim.add_argument("--hours", type=float, default=8, help="Simulation hours")
    p_sim.add_argument("--pattern", choices=_PATTERNS, default="steady", help="Load pattern")
    p_sim.add_argument("--json", action="store_true", help="Output JSON")

    # forecast
    p_fc = sub.add_parser("forecast", help="Forecast cognitive load trends")
    p_fc.add_argument("--events", required=True, help="Path to events JSON")
    p_fc.add_argument("--horizon", type=int, default=60, help="Forecast horizon (minutes)")

    # shed
    p_shed = sub.add_parser("shed", help="Recommend load shedding")
    p_shed.add_argument("--events", required=True, help="Path to events JSON")
    p_shed.add_argument("--target", type=float, default=50, help="Target load score")

    # report
    p_report = sub.add_parser("report", help="Generate HTML report")
    p_report.add_argument("--events", required=True, help="Path to events JSON")
    p_report.add_argument("-o", "--output", default="cognitive_load.html", help="Output file")

    args = parser.parse_args(argv)

    if args.cmd == "analyze":
        events = _load_events(args.events)
        monitor = CognitiveLoadMonitor()
        monitor.ingest(events)
        _print_analysis(monitor.analyze())

    elif args.cmd == "simulate":
        events = simulate_events(args.agents, args.hours, args.pattern)
        if args.json:
            print(json.dumps([asdict(e) for e in events], indent=2))
        else:
            monitor = CognitiveLoadMonitor()
            monitor.ingest(events)
            print(f"  Simulated {len(events)} events ({args.pattern} pattern, "
                  f"{args.agents} agents, {args.hours}h)\n")
            _print_analysis(monitor.analyze())

    elif args.cmd == "forecast":
        events = _load_events(args.events)
        monitor = CognitiveLoadMonitor()
        monitor.ingest(events)
        _print_forecast(monitor.analyze(), args.horizon)

    elif args.cmd == "shed":
        events = _load_events(args.events)
        monitor = CognitiveLoadMonitor()
        monitor.ingest(events)
        _print_shedding(monitor.analyze(), args.target)

    elif args.cmd == "report":
        events = _load_events(args.events)
        monitor = CognitiveLoadMonitor()
        monitor.ingest(events)
        html = _generate_html_report(monitor.analyze())
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  Report written to {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
