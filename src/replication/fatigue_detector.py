"""Safety Alert Fatigue Detector — identify when operators are overwhelmed.

Alert fatigue occurs when operators receive too many alerts, leading to
desensitization, slower response times, and missed critical events.  This
module analyzes alert streams to detect fatigue indicators and recommend
mitigations.

Fatigue Indicators
------------------
* **Volume overload**: alerts/hour exceeds sustainable threshold
* **Repetition**: same alert type fires repeatedly (noise)
* **Severity inflation**: too many critical/high alerts diluting urgency
* **Ack lag growth**: acknowledgment time trending upward
* **Suppression rate**: percentage of alerts auto-suppressed or ignored
* **Off-hours load**: alerts during nights/weekends (burnout risk)

Usage (CLI)::

    python -m replication fatigue-detect analyze --alerts alerts.json
    python -m replication fatigue-detect simulate --hours 72 --rate 40
    python -m replication fatigue-detect recommend --alerts alerts.json
    python -m replication fatigue-detect report --alerts alerts.json --output fatigue.html

Programmatic::

    from replication.fatigue_detector import FatigueDetector, AlertEvent
    detector = FatigueDetector(thresholds=FatigueThresholds(max_alerts_per_hour=30))
    detector.ingest(alert_events)
    result = detector.analyze()
    print(f"Fatigue score: {result.score}/100 — {result.level}")
"""

from __future__ import annotations

import argparse
import html as _html
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ._helpers import box_header as _box_header


# ── Data models ──────────────────────────────────────────────────────


@dataclass
class AlertEvent:
    """A single alert event."""

    timestamp: float  # Unix epoch
    alert_type: str = "unknown"
    severity: str = "medium"  # low | medium | high | critical
    source: str = ""
    message: str = ""
    acknowledged: bool = False
    ack_delay_s: Optional[float] = None  # seconds until acknowledged
    suppressed: bool = False


@dataclass
class FatigueThresholds:
    """Configurable thresholds for fatigue detection."""

    max_alerts_per_hour: float = 25.0
    repetition_ratio: float = 0.4  # >40% same type = noisy
    critical_ratio: float = 0.3  # >30% critical = inflation
    max_ack_delay_s: float = 300.0  # 5 min acceptable ack time
    ack_lag_growth_pct: float = 50.0  # >50% growth = fatigue
    suppression_ratio: float = 0.5  # >50% suppressed = too noisy
    off_hours_ratio: float = 0.3  # >30% off-hours = burnout risk
    off_hours_start: int = 22  # 10 PM
    off_hours_end: int = 7  # 7 AM


@dataclass
class FatigueIndicator:
    """A single fatigue signal."""

    name: str
    score: float  # 0-100
    detail: str
    severity: str = "info"  # info | warning | critical


@dataclass
class FatigueResult:
    """Overall fatigue analysis result."""

    score: float  # 0-100 composite
    level: str  # healthy | mild | moderate | severe | critical
    indicators: List[FatigueIndicator] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 1),
            "level": self.level,
            "indicators": [asdict(i) for i in self.indicators],
            "recommendations": self.recommendations,
            "stats": self.stats,
        }


# ── Detector ─────────────────────────────────────────────────────────


class FatigueDetector:
    """Analyzes alert streams for fatigue indicators."""

    LEVELS = [
        (20, "healthy"),
        (40, "mild"),
        (60, "moderate"),
        (80, "severe"),
        (101, "critical"),
    ]

    def __init__(self, thresholds: Optional[FatigueThresholds] = None) -> None:
        self.thresholds = thresholds or FatigueThresholds()
        self._events: List[AlertEvent] = []

    def ingest(self, events: Sequence[AlertEvent]) -> None:
        """Add alert events for analysis."""
        self._events.extend(events)
        self._events.sort(key=lambda e: e.timestamp)

    def clear(self) -> None:
        self._events.clear()

    def analyze(self) -> FatigueResult:
        """Run full fatigue analysis on ingested events."""
        if not self._events:
            return FatigueResult(
                score=0, level="healthy",
                indicators=[FatigueIndicator("no_data", 0, "No alerts to analyze")],
                stats={"total_alerts": 0},
            )

        indicators: List[FatigueIndicator] = []
        t = self.thresholds

        total = len(self._events)
        span_h = max((self._events[-1].timestamp - self._events[0].timestamp) / 3600, 0.01)

        # Single-pass aggregation (was 5 separate iterations over _events)
        sev_counts: Counter = Counter()
        type_counts: Counter = Counter()
        acked: List[AlertEvent] = []
        suppressed = 0
        off_count = 0
        _off_start = t.off_hours_start
        _off_end = t.off_hours_end
        for e in self._events:
            sev_counts[e.severity] += 1
            type_counts[e.alert_type] += 1
            if e.ack_delay_s is not None:
                acked.append(e)
            if e.suppressed:
                suppressed += 1
            if _is_off_hours(e.timestamp, _off_start, _off_end):
                off_count += 1

        rate = total / span_h

        # 1. Volume overload
        vol_score = min(100, (rate / t.max_alerts_per_hour) * 50) if t.max_alerts_per_hour > 0 else 0
        vol_sev = "critical" if rate > t.max_alerts_per_hour * 2 else "warning" if rate > t.max_alerts_per_hour else "info"
        indicators.append(FatigueIndicator(
            "volume_overload", round(vol_score, 1),
            f"{rate:.1f} alerts/hour (threshold: {t.max_alerts_per_hour})",
            vol_sev,
        ))

        # 2. Repetition noise
        if total > 0:
            most_common_type, most_common_count = type_counts.most_common(1)[0]
            rep_ratio = most_common_count / total
            rep_score = min(100, (rep_ratio / t.repetition_ratio) * 50) if t.repetition_ratio > 0 else 0
            rep_sev = "warning" if rep_ratio > t.repetition_ratio else "info"
            indicators.append(FatigueIndicator(
                "repetition_noise", round(rep_score, 1),
                f"Top type '{most_common_type}' is {rep_ratio:.0%} of all alerts",
                rep_sev,
            ))

        # 3. Severity inflation
        crit_high = sev_counts.get("critical", 0) + sev_counts.get("high", 0)
        crit_ratio = crit_high / total if total > 0 else 0
        sev_score = min(100, (crit_ratio / t.critical_ratio) * 50) if t.critical_ratio > 0 else 0
        sev_sev = "warning" if crit_ratio > t.critical_ratio else "info"
        indicators.append(FatigueIndicator(
            "severity_inflation", round(sev_score, 1),
            f"{crit_ratio:.0%} of alerts are critical/high (threshold: {t.critical_ratio:.0%})",
            sev_sev,
        ))

        # 4. Ack lag growth
        ack_score = 0.0
        if len(acked) >= 4:
            half = len(acked) // 2
            first_half_avg = sum(e.ack_delay_s for e in acked[:half]) / half  # type: ignore
            second_half_avg = sum(e.ack_delay_s for e in acked[half:]) / (len(acked) - half)  # type: ignore
            if first_half_avg > 0:
                growth_pct = ((second_half_avg - first_half_avg) / first_half_avg) * 100
            else:
                growth_pct = 0
            ack_score = min(100, max(0, (growth_pct / t.ack_lag_growth_pct) * 50))
            ack_sev = "critical" if growth_pct > t.ack_lag_growth_pct * 2 else "warning" if growth_pct > t.ack_lag_growth_pct else "info"
            indicators.append(FatigueIndicator(
                "ack_lag_growth", round(ack_score, 1),
                f"Ack delay grew {growth_pct:+.0f}% (first half avg: {first_half_avg:.0f}s → second half: {second_half_avg:.0f}s)",
                ack_sev,
            ))

        # 5. Suppression rate (suppressed already counted in single pass)
        sup_ratio = suppressed / total if total > 0 else 0
        sup_score = min(100, (sup_ratio / t.suppression_ratio) * 50) if t.suppression_ratio > 0 else 0
        sup_sev = "warning" if sup_ratio > t.suppression_ratio else "info"
        indicators.append(FatigueIndicator(
            "suppression_rate", round(sup_score, 1),
            f"{sup_ratio:.0%} of alerts suppressed (threshold: {t.suppression_ratio:.0%})",
            sup_sev,
        ))

        # 6. Off-hours load (off_count already counted in single pass)
        off_ratio = off_count / total if total > 0 else 0
        off_score = min(100, (off_ratio / t.off_hours_ratio) * 50) if t.off_hours_ratio > 0 else 0
        off_sev = "warning" if off_ratio > t.off_hours_ratio else "info"
        indicators.append(FatigueIndicator(
            "off_hours_load", round(off_score, 1),
            f"{off_ratio:.0%} of alerts fire during off-hours ({t.off_hours_start}:00-{t.off_hours_end}:00)",
            off_sev,
        ))

        # Composite score (weighted average)
        weights = {
            "volume_overload": 2.5,
            "repetition_noise": 1.5,
            "severity_inflation": 1.5,
            "ack_lag_growth": 2.0,
            "suppression_rate": 1.0,
            "off_hours_load": 1.5,
        }
        total_weight = sum(weights.get(i.name, 1.0) for i in indicators)
        composite = sum(i.score * weights.get(i.name, 1.0) for i in indicators) / total_weight if total_weight else 0

        level = "critical"
        for threshold, lbl in self.LEVELS:
            if composite < threshold:
                level = lbl
                break

        # Recommendations
        recs = _generate_recommendations(indicators, self.thresholds)

        stats = {
            "total_alerts": total,
            "span_hours": round(span_h, 1),
            "alerts_per_hour": round(rate, 1),
            "severity_breakdown": dict(sev_counts),
            "unique_alert_types": len(type_counts),
            "top_types": dict(type_counts.most_common(5)),
            "acknowledged": len(acked),
            "suppressed": suppressed,
            "off_hours_count": off_count,
        }
        if acked:
            delays = [e.ack_delay_s for e in acked if e.ack_delay_s is not None]
            stats["avg_ack_delay_s"] = round(sum(delays) / len(delays), 1)
            stats["max_ack_delay_s"] = round(max(delays), 1)

        return FatigueResult(
            score=round(composite, 1),
            level=level,
            indicators=indicators,
            recommendations=recs,
            stats=stats,
        )


def _is_off_hours(ts: float, start: int, end: int) -> bool:
    """Check if timestamp falls in off-hours window (UTC)."""
    hour = datetime.fromtimestamp(ts, tz=timezone.utc).hour
    if start > end:  # wraps midnight (e.g., 22-7)
        return hour >= start or hour < end
    return start <= hour < end


def _generate_recommendations(
    indicators: List[FatigueIndicator],
    thresholds: FatigueThresholds,
) -> List[str]:
    """Generate actionable recommendations from fatigue indicators."""
    recs: List[str] = []
    by_name = {i.name: i for i in indicators}

    vol = by_name.get("volume_overload")
    if vol and vol.severity != "info":
        recs.append("Reduce alert volume: aggregate related alerts, increase deduplication windows, or raise trigger thresholds for low-value alerts.")

    rep = by_name.get("repetition_noise")
    if rep and rep.severity != "info":
        recs.append("Suppress or consolidate the most repeated alert type. Consider auto-grouping repeated alerts into digest summaries.")

    sev = by_name.get("severity_inflation")
    if sev and sev.severity != "info":
        recs.append("Recalibrate severity levels: reserve 'critical' for genuinely urgent events. Too many high-severity alerts dilute urgency.")

    ack = by_name.get("ack_lag_growth")
    if ack and ack.severity != "info":
        recs.append("Ack times are growing — consider reducing alert volume, adding auto-remediation, or rotating on-call to prevent burnout.")

    sup = by_name.get("suppression_rate")
    if sup and sup.severity != "info":
        recs.append("High suppression rate suggests many alerts are not actionable. Remove or reconfigure suppressed alert rules.")

    off = by_name.get("off_hours_load")
    if off and off.severity != "info":
        recs.append("Significant off-hours alert load detected. Route non-urgent off-hours alerts to queues instead of pagers.")

    if not recs:
        recs.append("Alert hygiene looks good. Continue monitoring for changes.")

    return recs


# ── Simulator ────────────────────────────────────────────────────────


def simulate_alerts(
    hours: int = 72,
    base_rate: float = 15.0,
    fatigue_onset_hour: int = 24,
    spike_hours: Optional[List[int]] = None,
    seed: Optional[int] = None,
) -> List[AlertEvent]:
    """Generate a synthetic alert stream with realistic fatigue patterns.

    After ``fatigue_onset_hour``, ack delays start growing and suppression
    increases — simulating operator fatigue.
    """
    rng = random.Random(seed)
    alert_types = [
        "cpu_threshold", "memory_threshold", "disk_full", "auth_failure",
        "policy_violation", "anomaly_detected", "rate_limit", "connection_drop",
        "certificate_expiry", "replication_lag",
    ]
    severities = ["low", "medium", "high", "critical"]
    sev_weights = [3, 4, 2, 1]

    events: List[AlertEvent] = []
    now = time.time()
    start = now - (hours * 3600)

    t = start
    while t < now:
        hour_offset = (t - start) / 3600
        # Variable rate with optional spikes
        rate = base_rate
        if spike_hours and int(hour_offset) in spike_hours:
            rate *= rng.uniform(3, 6)
        if hour_offset > fatigue_onset_hour:
            rate *= 1 + 0.3 * ((hour_offset - fatigue_onset_hour) / hours)

        interval = 3600 / max(rate, 0.1)
        t += rng.expovariate(1 / interval) if interval > 0 else 60

        if t >= now:
            break

        atype = rng.choices(alert_types, weights=[5, 3, 2, 4, 3, 2, 4, 2, 1, 1])[0]
        sev = rng.choices(severities, weights=sev_weights)[0]

        # Fatigue effects: slower acks, more suppression
        fatigue_factor = max(0, (hour_offset - fatigue_onset_hour) / hours) if hour_offset > fatigue_onset_hour else 0
        base_ack = rng.uniform(10, 120)
        ack_delay = base_ack * (1 + fatigue_factor * 4)
        suppressed = rng.random() < (0.05 + fatigue_factor * 0.5)
        acked = not suppressed and rng.random() > (0.1 + fatigue_factor * 0.3)

        events.append(AlertEvent(
            timestamp=t,
            alert_type=atype,
            severity=sev,
            source=rng.choice(["controller", "monitor", "agent-01", "agent-02", "gateway"]),
            message=f"{atype.replace('_', ' ').title()} alert",
            acknowledged=acked,
            ack_delay_s=round(ack_delay, 1) if acked else None,
            suppressed=suppressed,
        ))

    return events


# ── HTML Report ──────────────────────────────────────────────────────


def generate_html_report(result: FatigueResult) -> str:
    """Generate a self-contained HTML fatigue report."""
    level_colors = {
        "healthy": "#22c55e", "mild": "#84cc16",
        "moderate": "#eab308", "severe": "#f97316", "critical": "#ef4444",
    }
    color = level_colors.get(result.level, "#6b7280")

    indicator_rows = ""
    for ind in result.indicators:
        bar_color = "#22c55e" if ind.score < 30 else "#eab308" if ind.score < 60 else "#ef4444"
        indicator_rows += f"""
        <tr>
          <td><strong>{_html.escape(ind.name.replace('_', ' ').title())}</strong></td>
          <td>
            <div style="background:#e5e7eb;border-radius:4px;overflow:hidden;height:20px;width:200px;display:inline-block;vertical-align:middle">
              <div style="background:{bar_color};height:100%;width:{min(ind.score, 100):.0f}%"></div>
            </div>
            <span style="margin-left:8px">{ind.score:.0f}/100</span>
          </td>
          <td>{_html.escape(ind.detail)}</td>
          <td><span style="color:{bar_color};font-weight:bold">{_html.escape(ind.severity)}</span></td>
        </tr>"""

    rec_items = "".join(f"<li>{_html.escape(r)}</li>" for r in result.recommendations)

    stats_json = json.dumps(result.stats, indent=2)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Alert Fatigue Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; background: #f9fafb; color: #1f2937; }}
  .score-card {{ background: white; border-radius: 12px; padding: 2rem; box-shadow: 0 1px 3px rgba(0,0,0,.1); text-align: center; margin-bottom: 2rem; }}
  .score {{ font-size: 4rem; font-weight: 800; color: {color}; }}
  .level {{ font-size: 1.5rem; text-transform: uppercase; letter-spacing: 2px; color: {color}; }}
  table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.1); margin-bottom: 2rem; }}
  th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #e5e7eb; }}
  th {{ background: #f3f4f6; font-weight: 600; }}
  .section {{ background: white; border-radius: 8px; padding: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,.1); margin-bottom: 2rem; }}
  h2 {{ color: #374151; margin-top: 0; }}
  ul {{ padding-left: 1.5rem; }}
  li {{ margin-bottom: 0.5rem; }}
  pre {{ background: #1f2937; color: #d1d5db; padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem; }}
</style>
</head>
<body>
<h1 style="text-align:center">🔔 Alert Fatigue Report</h1>
<div class="score-card">
  <div class="score">{result.score:.0f}</div>
  <div class="level">{result.level}</div>
  <p style="color:#6b7280;margin-top:0.5rem">Fatigue Score (0 = no fatigue, 100 = critical)</p>
</div>

<h2>📊 Indicators</h2>
<table>
  <tr><th>Indicator</th><th>Score</th><th>Detail</th><th>Severity</th></tr>
  {indicator_rows}
</table>

<div class="section">
  <h2>💡 Recommendations</h2>
  <ul>{rec_items}</ul>
</div>

<div class="section">
  <h2>📈 Statistics</h2>
  <pre>{stats_json}</pre>
</div>

<p style="text-align:center;color:#9ca3af;font-size:0.85rem">
  Generated by AI Replication Sandbox — Alert Fatigue Detector
</p>
</body>
</html>"""


# ── CLI ──────────────────────────────────────────────────────────────


def _print_analysis(result: FatigueResult, fmt: str = "text") -> None:
    """Print analysis to stdout."""
    if fmt == "json":
        print(json.dumps(result.to_dict(), indent=2))
        return

    level_icons = {
        "healthy": "✅", "mild": "🟡", "moderate": "⚠️",
        "severe": "🟠", "critical": "🔴",
    }
    icon = level_icons.get(result.level, "❓")

    for line in _box_header("Alert Fatigue Detector"):
        print(line)
    print()
    print(f"  Fatigue Score : {result.score:.0f}/100  {icon} {result.level.upper()}")
    print(f"  Total Alerts  : {result.stats.get('total_alerts', 0)}")
    print(f"  Span          : {result.stats.get('span_hours', 0)} hours")
    print(f"  Rate          : {result.stats.get('alerts_per_hour', 0)} alerts/hour")
    print()

    print("  Indicators:")
    for ind in result.indicators:
        bar_len = int(ind.score / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        sev_mark = "!" if ind.severity == "critical" else "~" if ind.severity == "warning" else " "
        print(f"    {sev_mark} {ind.name:<22} {bar} {ind.score:5.1f}  {ind.detail}")
    print()

    print("  Recommendations:")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"    {i}. {rec}")
    print()

    if "severity_breakdown" in result.stats:
        print("  Severity breakdown:")
        for sev, count in sorted(result.stats["severity_breakdown"].items()):
            print(f"    {sev:<10} {count}")
    if "top_types" in result.stats:
        print("  Top alert types:")
        for atype, count in result.stats["top_types"].items():
            print(f"    {atype:<25} {count}")


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="replication fatigue-detect",
        description="Detect alert fatigue from safety alert streams",
    )
    sub = parser.add_subparsers(dest="cmd")

    # analyze
    p_analyze = sub.add_parser("analyze", help="Analyze alert stream for fatigue")
    p_analyze.add_argument("--alerts", help="JSON file with alert events")
    p_analyze.add_argument("--max-per-hour", type=float, default=25, help="Max alerts/hour threshold")
    p_analyze.add_argument("--format", choices=["text", "json"], default="text")

    # simulate
    p_sim = sub.add_parser("simulate", help="Generate and analyze synthetic alerts")
    p_sim.add_argument("--hours", type=int, default=72, help="Simulation span in hours")
    p_sim.add_argument("--rate", type=float, default=15, help="Base alert rate per hour")
    p_sim.add_argument("--seed", type=int, help="Random seed for reproducibility")
    p_sim.add_argument("--format", choices=["text", "json"], default="text")
    p_sim.add_argument("--export", help="Export simulated alerts to JSON file")

    # recommend
    p_rec = sub.add_parser("recommend", help="Get recommendations for reducing fatigue")
    p_rec.add_argument("--alerts", help="JSON file with alert events")

    # report
    p_rep = sub.add_parser("report", help="Generate HTML fatigue report")
    p_rep.add_argument("--alerts", help="JSON file with alert events")
    p_rep.add_argument("--output", "-o", default="fatigue_report.html", help="Output HTML path")
    p_rep.add_argument("--hours", type=int, default=72, help="Simulation hours (if no --alerts)")
    p_rep.add_argument("--rate", type=float, default=15, help="Simulation rate (if no --alerts)")

    args = parser.parse_args(argv)

    if not args.cmd:
        parser.print_help()
        return

    def _load_alerts(path: Optional[str]) -> List[AlertEvent]:
        if not path:
            return simulate_alerts(hours=72, seed=42)
        with open(path) as f:
            data = json.load(f)
        return [AlertEvent(**e) for e in data]

    if args.cmd == "analyze":
        events = _load_alerts(args.alerts)
        detector = FatigueDetector(FatigueThresholds(max_alerts_per_hour=args.max_per_hour))
        detector.ingest(events)
        result = detector.analyze()
        _print_analysis(result, args.format)

    elif args.cmd == "simulate":
        events = simulate_alerts(hours=args.hours, base_rate=args.rate, seed=args.seed)
        detector = FatigueDetector()
        detector.ingest(events)
        result = detector.analyze()
        _print_analysis(result, args.format)
        if args.export:
            with open(args.export, "w") as f:
                json.dump([asdict(e) for e in events], f, indent=2)
            print(f"\n  Exported {len(events)} alerts to {args.export}")

    elif args.cmd == "recommend":
        events = _load_alerts(args.alerts)
        detector = FatigueDetector()
        detector.ingest(events)
        result = detector.analyze()
        for line in _box_header("Fatigue Recommendations"):
            print(line)
        print()
        for i, rec in enumerate(result.recommendations, 1):
            print(f"  {i}. {rec}")
        print()

    elif args.cmd == "report":
        if args.alerts:
            events = _load_alerts(args.alerts)
        else:
            events = simulate_alerts(hours=args.hours, base_rate=args.rate, seed=42)
        detector = FatigueDetector()
        detector.ingest(events)
        result = detector.analyze()
        html = generate_html_report(result)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
