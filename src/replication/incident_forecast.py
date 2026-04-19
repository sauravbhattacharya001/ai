"""Incident Forecaster — predict future safety incidents from historical patterns.

Analyze historical safety events (scorecard drops, anomalies, breaches,
escalations) to forecast probability and timing of future incidents.
Uses exponential smoothing, Poisson arrival modeling, and seasonal
decomposition to produce actionable predictions with confidence
intervals and preemptive action recommendations.

CLI usage::

    # Forecast from trend data (uses trend_tracker JSONL)
    python -m replication forecast
    python -m replication forecast --horizon 7 --granularity daily

    # Forecast with custom history file
    python -m replication forecast --history events.jsonl

    # Show top risks only
    python -m replication forecast --top 5

    # Export forecast as JSON
    python -m replication forecast --format json

    # Interactive what-if: "what if we fix policy X?"
    python -m replication forecast --mitigate containment

Programmatic::

    from replication.incident_forecast import IncidentForecaster, ForecastConfig
    forecaster = IncidentForecaster()
    forecaster.ingest_events(events)
    report = forecaster.forecast(horizon_days=7)
    print(report.render())
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ._helpers import box_header as _box_header, stats_mean, stats_std


# ── Data models ──────────────────────────────────────────────────────


@dataclass
class IncidentEvent:
    """A single historical safety event."""
    timestamp: str  # ISO-8601
    category: str  # e.g. "score_drop", "anomaly", "breach", "escalation"
    severity: float  # 0-10
    description: str = ""
    dimension: str = ""  # which safety dimension
    resolved: bool = True
    tags: List[str] = field(default_factory=list)

    @property
    def dt(self) -> datetime:
        return datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "category": self.category,
            "severity": self.severity,
            "description": self.description,
            "dimension": self.dimension,
            "resolved": self.resolved,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IncidentEvent":
        return cls(
            timestamp=d["timestamp"],
            category=d["category"],
            severity=float(d.get("severity", 5.0)),
            description=d.get("description", ""),
            dimension=d.get("dimension", ""),
            resolved=d.get("resolved", True),
            tags=d.get("tags", []),
        )


@dataclass
class ForecastConfig:
    """Configuration for incident forecasting."""
    horizon_days: int = 7
    granularity: str = "daily"  # "daily" or "weekly"
    confidence_level: float = 0.90
    smoothing_alpha: float = 0.3  # EMA decay
    seasonal_period: int = 7  # days
    min_events: int = 3  # minimum events to forecast
    mitigate: Optional[str] = None  # category to simulate mitigation


@dataclass
class CategoryForecast:
    """Forecast for a single incident category."""
    category: str
    expected_count: float  # predicted incidents in horizon
    poisson_lambda: float  # daily arrival rate
    probability: float  # P(at least 1 incident in horizon)
    severity_mean: float
    severity_trend: str  # "rising", "stable", "falling"
    confidence_low: float
    confidence_high: float
    seasonal_peak_day: Optional[str]  # day of week with most incidents
    last_seen_days_ago: float
    recommended_actions: List[str]
    risk_score: float  # composite 0-100


@dataclass
class ForecastReport:
    """Complete incident forecast report."""
    generated_at: str
    horizon_days: int
    total_events_analyzed: int
    date_range: Tuple[str, str]
    overall_risk: float  # 0-100
    risk_trend: str  # "increasing", "stable", "decreasing"
    category_forecasts: List[CategoryForecast]
    seasonal_pattern: Dict[str, float]  # day-of-week -> relative frequency
    top_preemptive_actions: List[str]
    mitigation_applied: Optional[str] = None
    mitigation_reduction_pct: float = 0.0

    def render(self) -> str:
        lines: List[str] = []
        lines.extend(_box_header("INCIDENT FORECAST REPORT"))
        lines.append(f"  Generated : {self.generated_at}")
        lines.append(f"  Horizon   : {self.horizon_days} days")
        lines.append(f"  Events    : {self.total_events_analyzed} analyzed")
        if self.date_range[0]:
            lines.append(f"  Range     : {self.date_range[0][:10]} → {self.date_range[1][:10]}")
        risk_bar = _risk_bar(self.overall_risk)
        lines.append(f"  Risk Level: {risk_bar} {self.overall_risk:.0f}/100 ({self.risk_trend})")
        if self.mitigation_applied:
            lines.append(f"  Mitigation: -{self.mitigation_reduction_pct:.0f}% with '{self.mitigation_applied}' fix")
        lines.append("")

        # Category forecasts
        if self.category_forecasts:
            lines.append("  ┌─ CATEGORY FORECASTS ─────────────────────────────┐")
            for cf in sorted(self.category_forecasts, key=lambda c: -c.risk_score):
                icon = _severity_icon(cf.risk_score)
                lines.append(f"  │ {icon} {cf.category:<18} Risk: {cf.risk_score:5.1f}/100     │")
                lines.append(f"  │   Expected: {cf.expected_count:.1f} events  "
                             f"P(≥1): {cf.probability:.0%}         │")
                lines.append(f"  │   Severity: {cf.severity_mean:.1f}/10 ({cf.severity_trend})"
                             f"  Last: {cf.last_seen_days_ago:.0f}d ago    │")
                if cf.seasonal_peak_day:
                    lines.append(f"  │   Peak day: {cf.seasonal_peak_day:<38}│")
                if cf.recommended_actions:
                    lines.append(f"  │   Actions: {cf.recommended_actions[0]:<37}│")
                    for a in cf.recommended_actions[1:3]:
                        lines.append(f"  │            {a:<37}│")
                lines.append("  │                                                   │")
            lines.append("  └─────────────────────────────────────────────────────┘")
            lines.append("")

        # Seasonal pattern
        if any(v > 0 for v in self.seasonal_pattern.values()):
            lines.append("  ┌─ WEEKLY PATTERN ─────────────────────────────────┐")
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            max_v = max(self.seasonal_pattern.values()) or 1
            for d in days:
                v = self.seasonal_pattern.get(d, 0)
                bar_len = int((v / max_v) * 25) if max_v > 0 else 0
                bar = "█" * bar_len
                lines.append(f"  │  {d} {bar:<25} {v:.2f}     │")
            lines.append("  └─────────────────────────────────────────────────────┘")
            lines.append("")

        # Top actions
        if self.top_preemptive_actions:
            lines.append("  ⚡ TOP PREEMPTIVE ACTIONS:")
            for i, action in enumerate(self.top_preemptive_actions[:7], 1):
                lines.append(f"     {i}. {action}")
            lines.append("")

        lines.append("  ─" * 28)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "horizon_days": self.horizon_days,
            "total_events_analyzed": self.total_events_analyzed,
            "date_range": list(self.date_range),
            "overall_risk": self.overall_risk,
            "risk_trend": self.risk_trend,
            "category_forecasts": [
                {
                    "category": cf.category,
                    "expected_count": cf.expected_count,
                    "poisson_lambda": cf.poisson_lambda,
                    "probability": cf.probability,
                    "severity_mean": cf.severity_mean,
                    "severity_trend": cf.severity_trend,
                    "confidence_interval": [cf.confidence_low, cf.confidence_high],
                    "seasonal_peak_day": cf.seasonal_peak_day,
                    "last_seen_days_ago": cf.last_seen_days_ago,
                    "recommended_actions": cf.recommended_actions,
                    "risk_score": cf.risk_score,
                }
                for cf in self.category_forecasts
            ],
            "seasonal_pattern": self.seasonal_pattern,
            "top_preemptive_actions": self.top_preemptive_actions,
            "mitigation_applied": self.mitigation_applied,
            "mitigation_reduction_pct": self.mitigation_reduction_pct,
        }


# ── Helpers ──────────────────────────────────────────────────────────


def _risk_bar(score: float, width: int = 20) -> str:
    filled = int(score / 100 * width)
    if score >= 70:
        char = "▓"
    elif score >= 40:
        char = "▒"
    else:
        char = "░"
    return f"[{char * filled}{'·' * (width - filled)}]"


def _severity_icon(score: float) -> str:
    if score >= 70:
        return "🔴"
    elif score >= 40:
        return "🟡"
    return "🟢"


_DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Action recommendations per category
_CATEGORY_ACTIONS: Dict[str, List[str]] = {
    "score_drop": [
        "Review recent contract/policy changes",
        "Run quick-scan to identify root cause",
        "Increase monitoring frequency",
        "Check for correlated infrastructure changes",
    ],
    "anomaly": [
        "Investigate anomaly cluster patterns",
        "Tighten adaptive thresholds",
        "Deploy canary checks in affected area",
        "Review agent behavior profiles",
    ],
    "breach": [
        "Activate containment planner",
        "Run forensic evidence collection",
        "Rotate credentials immediately",
        "Engage incident response playbook",
    ],
    "escalation": [
        "Review escalation chain configuration",
        "Update runbook for faster resolution",
        "Add circuit breakers to affected paths",
        "Conduct tabletop exercise",
    ],
    "drift": [
        "Re-baseline safety thresholds",
        "Run safety diff against last known-good",
        "Schedule alignment review",
        "Enable continuous drift monitoring",
    ],
    "policy_violation": [
        "Audit policy linter rules",
        "Retrain teams on updated policies",
        "Add pre-commit policy checks",
        "Review access control settings",
    ],
}

# Mitigation effectiveness estimates (% reduction in arrival rate)
_MITIGATION_EFFECT: Dict[str, Dict[str, float]] = {
    "containment": {"breach": 0.6, "escalation": 0.3, "anomaly": 0.1},
    "monitoring": {"anomaly": 0.4, "drift": 0.3, "score_drop": 0.2},
    "policy": {"policy_violation": 0.5, "drift": 0.3, "breach": 0.2},
    "training": {"policy_violation": 0.3, "escalation": 0.2, "breach": 0.1},
    "automation": {"score_drop": 0.3, "anomaly": 0.3, "drift": 0.4},
}


def _poisson_prob_at_least_one(lam: float) -> float:
    """P(X >= 1) for Poisson(lam)."""
    if lam <= 0:
        return 0.0
    return 1.0 - math.exp(-lam)


def _poisson_ci(lam: float, confidence: float = 0.90) -> Tuple[float, float]:
    """Approximate confidence interval for Poisson count."""
    if lam <= 0:
        return (0.0, 0.0)
    # Normal approximation
    z = 1.645 if confidence >= 0.90 else 1.28  # 90% or 80%
    sd = math.sqrt(lam)
    return (max(0.0, lam - z * sd), lam + z * sd)


def _ema(values: List[float], alpha: float = 0.3) -> List[float]:
    """Exponential moving average."""
    if not values:
        return []
    result = [values[0]]
    for v in values[1:]:
        result.append(alpha * v + (1 - alpha) * result[-1])
    return result


def _severity_trend(severities: List[float]) -> str:
    """Determine severity trend from recent values."""
    if len(severities) < 3:
        return "stable"
    recent = stats_mean(severities[-3:])
    older = stats_mean(severities[:-3]) if len(severities) > 3 else severities[0]
    diff = recent - older
    if diff > 0.5:
        return "rising"
    elif diff < -0.5:
        return "falling"
    return "stable"


# ── Forecaster ───────────────────────────────────────────────────────


class IncidentForecaster:
    """Predict future safety incidents from historical event data."""

    def __init__(self, config: Optional[ForecastConfig] = None):
        self.config = config or ForecastConfig()
        self._events: List[IncidentEvent] = []

    def ingest_events(self, events: List[IncidentEvent]) -> None:
        """Add historical events for analysis."""
        self._events.extend(events)
        self._events.sort(key=lambda e: e.timestamp)

    def ingest_from_jsonl(self, path: str) -> int:
        """Load events from a JSONL file. Returns count loaded."""
        p = Path(path)
        if not p.exists():
            return 0
        count = 0
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                self._events.append(IncidentEvent.from_dict(d))
                count += 1
            except (json.JSONDecodeError, KeyError):
                continue
        self._events.sort(key=lambda e: e.timestamp)
        return count

    def ingest_from_trend_tracker(self, path: str) -> int:
        """Derive incident events from trend tracker JSONL (score drops)."""
        p = Path(path)
        if not p.exists():
            return 0
        entries: List[Dict[str, Any]] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        count = 0
        prev_score: Optional[float] = None
        for entry in entries:
            score = entry.get("overall_score") or entry.get("score", 0)
            ts = entry.get("timestamp", entry.get("recorded_at", ""))
            if prev_score is not None and score < prev_score:
                drop = prev_score - score
                severity = min(10.0, drop / 5.0)  # 50pt drop = severity 10
                self._events.append(IncidentEvent(
                    timestamp=ts,
                    category="score_drop",
                    severity=severity,
                    description=f"Score dropped {drop:.1f} pts ({prev_score:.1f} → {score:.1f})",
                    dimension="overall",
                ))
                count += 1
            prev_score = score
        self._events.sort(key=lambda e: e.timestamp)
        return count

    def forecast(self, horizon_days: Optional[int] = None) -> ForecastReport:
        """Generate incident forecast report."""
        cfg = self.config
        horizon = horizon_days or cfg.horizon_days
        now = datetime.now(timezone.utc)

        if not self._events:
            return self._empty_report(now, horizon)

        # Date range
        first_ts = self._events[0].timestamp
        last_ts = self._events[-1].timestamp
        first_dt = self._events[0].dt
        last_dt = self._events[-1].dt
        span_days = max(1.0, (last_dt - first_dt).total_seconds() / 86400)

        # Group by category
        by_cat: Dict[str, List[IncidentEvent]] = defaultdict(list)
        for ev in self._events:
            by_cat[ev.category].append(ev)

        # Seasonal pattern (day of week)
        dow_counts: Dict[str, int] = Counter()
        for ev in self._events:
            dow_counts[_DAY_NAMES[ev.dt.weekday()]] += 1
        total_dow = sum(dow_counts.values()) or 1
        seasonal = {d: dow_counts.get(d, 0) / total_dow * 7 for d in _DAY_NAMES}

        # Per-category forecasts
        cat_forecasts: List[CategoryForecast] = []
        for cat, events in by_cat.items():
            cf = self._forecast_category(cat, events, span_days, horizon, now, seasonal)
            cat_forecasts.append(cf)

        # Apply mitigation if requested
        mitigation_reduction = 0.0
        if cfg.mitigate and cfg.mitigate in _MITIGATION_EFFECT:
            effects = _MITIGATION_EFFECT[cfg.mitigate]
            orig_risk = sum(cf.risk_score for cf in cat_forecasts)
            for cf in cat_forecasts:
                reduction = effects.get(cf.category, 0.05)
                cf.risk_score *= (1 - reduction)
                cf.expected_count *= (1 - reduction)
                cf.probability = _poisson_prob_at_least_one(
                    cf.poisson_lambda * (1 - reduction) * horizon)
            new_risk = sum(cf.risk_score for cf in cat_forecasts)
            if orig_risk > 0:
                mitigation_reduction = (1 - new_risk / orig_risk) * 100

        # Overall risk
        if cat_forecasts:
            overall_risk = min(100.0, sum(cf.risk_score for cf in cat_forecasts)
                               / len(cat_forecasts) * 1.5)
        else:
            overall_risk = 0.0

        # Risk trend: compare recent half vs older half
        mid = len(self._events) // 2
        if mid > 0:
            older_rate = mid / max(1, (self._events[mid].dt - first_dt).total_seconds() / 86400)
            recent_rate = (len(self._events) - mid) / max(1, (last_dt - self._events[mid].dt).total_seconds() / 86400)
            if recent_rate > older_rate * 1.2:
                risk_trend = "increasing"
            elif recent_rate < older_rate * 0.8:
                risk_trend = "decreasing"
            else:
                risk_trend = "stable"
        else:
            risk_trend = "stable"

        # Top preemptive actions (deduplicated, ordered by risk)
        actions: List[str] = []
        seen: set = set()
        for cf in sorted(cat_forecasts, key=lambda c: -c.risk_score):
            for a in cf.recommended_actions:
                if a not in seen:
                    actions.append(a)
                    seen.add(a)

        return ForecastReport(
            generated_at=now.isoformat(),
            horizon_days=horizon,
            total_events_analyzed=len(self._events),
            date_range=(first_ts, last_ts),
            overall_risk=overall_risk,
            risk_trend=risk_trend,
            category_forecasts=cat_forecasts,
            seasonal_pattern=seasonal,
            top_preemptive_actions=actions[:10],
            mitigation_applied=cfg.mitigate,
            mitigation_reduction_pct=mitigation_reduction,
        )

    def _forecast_category(
        self,
        category: str,
        events: List[IncidentEvent],
        span_days: float,
        horizon: int,
        now: datetime,
        seasonal: Dict[str, float],
    ) -> CategoryForecast:
        cfg = self.config

        # Daily arrival rate
        raw_lambda = len(events) / span_days

        # EMA-smoothed rate from daily counts
        first_dt = events[0].dt
        last_dt = events[-1].dt
        day_span = int((last_dt - first_dt).total_seconds() / 86400) + 1
        daily_counts: List[float] = [0.0] * max(1, day_span)
        for ev in events:
            idx = min(int((ev.dt - first_dt).total_seconds() / 86400), len(daily_counts) - 1)
            daily_counts[idx] += 1

        smoothed = _ema(daily_counts, cfg.smoothing_alpha)
        ema_lambda = smoothed[-1] if smoothed else raw_lambda

        # Blend raw and EMA
        lam = 0.5 * raw_lambda + 0.5 * ema_lambda

        # Expected count in horizon
        expected = lam * horizon
        prob = _poisson_prob_at_least_one(expected)
        ci_low, ci_high = _poisson_ci(expected, cfg.confidence_level)

        # Severity analysis
        severities = [ev.severity for ev in events]
        sev_mean = stats_mean(severities)
        sev_trend = _severity_trend(severities)

        # Seasonal peak day
        cat_dow: Dict[str, int] = Counter()
        for ev in events:
            cat_dow[_DAY_NAMES[ev.dt.weekday()]] += 1
        peak_day = max(cat_dow, key=cat_dow.get) if cat_dow else None  # type: ignore[arg-type]

        # Last seen
        last_seen = (now - events[-1].dt).total_seconds() / 86400

        # Risk score: composite of rate, severity, trend, recency
        rate_score = min(40, lam * 40)  # 0-40
        sev_score = sev_mean / 10 * 30  # 0-30
        trend_bonus = 15 if sev_trend == "rising" else (0 if sev_trend == "stable" else -5)
        recency_score = max(0, 15 - last_seen)  # 0-15, recent = higher
        risk_score = min(100, max(0, rate_score + sev_score + trend_bonus + recency_score))

        # Actions
        actions = list(_CATEGORY_ACTIONS.get(category, [
            f"Investigate {category} patterns",
            "Increase monitoring",
            "Review recent changes",
        ]))

        return CategoryForecast(
            category=category,
            expected_count=expected,
            poisson_lambda=lam,
            probability=prob,
            severity_mean=sev_mean,
            severity_trend=sev_trend,
            confidence_low=ci_low,
            confidence_high=ci_high,
            seasonal_peak_day=peak_day,
            last_seen_days_ago=last_seen,
            recommended_actions=actions,
            risk_score=risk_score,
        )

    def _empty_report(self, now: datetime, horizon: int) -> ForecastReport:
        return ForecastReport(
            generated_at=now.isoformat(),
            horizon_days=horizon,
            total_events_analyzed=0,
            date_range=("", ""),
            overall_risk=0.0,
            risk_trend="stable",
            category_forecasts=[],
            seasonal_pattern={d: 0.0 for d in _DAY_NAMES},
            top_preemptive_actions=["No historical data — start recording events to enable forecasting"],
        )


# ── CLI ──────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="replication forecast",
        description="Predict future safety incidents from historical data",
    )
    p.add_argument("--history", help="Path to events JSONL file")
    p.add_argument("--trends", help="Path to trend tracker JSONL file (derives score_drop events)")
    p.add_argument("--horizon", type=int, default=7, help="Forecast horizon in days (default: 7)")
    p.add_argument("--granularity", choices=["daily", "weekly"], default="daily")
    p.add_argument("--top", type=int, help="Show only top N risk categories")
    p.add_argument("--format", choices=["text", "json"], default="text")
    p.add_argument("--mitigate", help="Simulate mitigation strategy: containment|monitoring|policy|training|automation")
    p.add_argument("--alpha", type=float, default=0.3, help="EMA smoothing factor (default: 0.3)")
    p.add_argument("--demo", action="store_true", help="Run with synthetic demo data")
    return p


def _generate_demo_events() -> List[IncidentEvent]:
    """Create synthetic events for demonstration."""
    import random
    random.seed(42)
    now = datetime.now(timezone.utc)
    events: List[IncidentEvent] = []
    categories = ["score_drop", "anomaly", "breach", "escalation", "drift", "policy_violation"]
    weights = [0.3, 0.25, 0.1, 0.1, 0.15, 0.1]

    for i in range(60):
        days_ago = random.uniform(0, 30)
        ts = (now - timedelta(days=days_ago)).isoformat()
        # Weighted random category
        r = random.random()
        cumulative = 0.0
        cat = categories[0]
        for c, w in zip(categories, weights):
            cumulative += w
            if r <= cumulative:
                cat = c
                break
        sev = min(10, max(1, random.gauss(5, 2)))
        events.append(IncidentEvent(
            timestamp=ts,
            category=cat,
            severity=round(sev, 1),
            description=f"Synthetic {cat} event #{i+1}",
            dimension="overall",
        ))
    return events


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg = ForecastConfig(
        horizon_days=args.horizon,
        granularity=args.granularity,
        smoothing_alpha=args.alpha,
        mitigate=args.mitigate,
    )
    forecaster = IncidentForecaster(cfg)

    # Load data
    loaded = 0
    if args.demo:
        demo_events = _generate_demo_events()
        forecaster.ingest_events(demo_events)
        loaded = len(demo_events)
    else:
        if args.history:
            loaded += forecaster.ingest_from_jsonl(args.history)
        if args.trends:
            loaded += forecaster.ingest_from_trend_tracker(args.trends)
        if loaded == 0:
            # Try default paths
            for default in ["safety_events.jsonl", "safety_trends.jsonl"]:
                if Path(default).exists():
                    if "event" in default:
                        loaded += forecaster.ingest_from_jsonl(default)
                    else:
                        loaded += forecaster.ingest_from_trend_tracker(default)

    if loaded == 0 and not args.demo:
        print("No events found. Use --history <file>, --trends <file>, or --demo")
        print("Event JSONL format: {\"timestamp\": \"...\", \"category\": \"...\", \"severity\": N}")
        sys.exit(1)

    report = forecaster.forecast()

    # Filter top N
    if args.top and report.category_forecasts:
        report.category_forecasts = sorted(
            report.category_forecasts, key=lambda c: -c.risk_score
        )[:args.top]

    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.render())


if __name__ == "__main__":
    main()
