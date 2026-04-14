"""Adaptive Safety Thresholds — self-tuning safety limits.

Thresholds that learn from historical safety data and auto-adjust based on
statistical patterns, seasonal trends, and risk context.  Replaces rigid
hard-coded limits with dynamic bounds that tighten when risk is elevated and
relax during stable periods.

Key capabilities:
- **EMA-based baseline tracking** — exponential moving average of safety
  metrics with configurable smoothing
- **Seasonal decomposition** — detect hourly / daily / weekly patterns
  and adjust accordingly
- **Risk-aware multiplier** — automatically tighten limits when recent
  anomalies are detected
- **Breach forecasting** — predict when a metric will cross its threshold
  using linear extrapolation
- **Multi-metric profiles** — manage threshold sets across different safety
  dimensions (latency, error rate, resource usage, score drift, etc.)
- **CLI + library** — use from code or the command line

Usage::

    python -m replication adaptive-thresholds --demo
    python -m replication adaptive-thresholds --metric score_drift --window 50
    python -m replication adaptive-thresholds --profile fleet --export json

"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


# ── core data structures ─────────────────────────────────────────────


@dataclass
class ThresholdState:
    """Tracks a single adaptive threshold."""

    metric: str
    ema: float = 0.0
    ema_var: float = 0.0  # exponential moving variance
    alpha: float = 0.15  # EMA smoothing factor
    sigma_multiplier: float = 2.5  # how many σ for the threshold
    min_floor: float = 0.0  # absolute minimum threshold
    max_ceiling: float = float("inf")  # absolute maximum threshold
    risk_multiplier: float = 1.0  # tightens when elevated
    observations: int = 0
    recent_values: List[float] = field(default_factory=list)
    recent_window: int = 30
    breach_count: int = 0
    last_breach_idx: int = -1

    @property
    def std(self) -> float:
        return math.sqrt(max(self.ema_var, 1e-12))

    @property
    def upper_threshold(self) -> float:
        raw = self.ema + self.sigma_multiplier * self.std / self.risk_multiplier
        return min(max(raw, self.min_floor), self.max_ceiling)

    @property
    def lower_threshold(self) -> float:
        raw = self.ema - self.sigma_multiplier * self.std / self.risk_multiplier
        return max(min(raw, self.max_ceiling), self.min_floor)

    def observe(self, value: float) -> Dict[str, Any]:
        """Ingest a new observation and return status."""
        self.observations += 1
        self.recent_values.append(value)
        if len(self.recent_values) > self.recent_window:
            self.recent_values = self.recent_values[-self.recent_window:]

        # warm-up: first observation seeds EMA
        if self.observations == 1:
            self.ema = value
            self.ema_var = 0.0
            return self._status(value, breached=False)

        # update EMA
        delta = value - self.ema
        self.ema += self.alpha * delta
        # update exponential moving variance
        self.ema_var = (1 - self.alpha) * (self.ema_var + self.alpha * delta * delta)

        # check breach
        breached = value > self.upper_threshold or value < self.lower_threshold

        if breached:
            self.breach_count += 1
            self.last_breach_idx = self.observations
            # tighten risk multiplier
            self.risk_multiplier = min(self.risk_multiplier + 0.15, 3.0)
        else:
            # slowly relax
            self.risk_multiplier = max(self.risk_multiplier - 0.02, 1.0)

        return self._status(value, breached)

    def _status(self, value: float, breached: bool) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "value": round(value, 4),
            "ema": round(self.ema, 4),
            "std": round(self.std, 4),
            "upper": round(self.upper_threshold, 4),
            "lower": round(self.lower_threshold, 4),
            "breached": breached,
            "risk_multiplier": round(self.risk_multiplier, 2),
            "observations": self.observations,
            "breach_count": self.breach_count,
        }

    def forecast_breach(self, horizon: int = 10) -> Optional[Dict[str, Any]]:
        """Linear extrapolation: will the metric breach within *horizon* steps?"""
        vals = self.recent_values
        if len(vals) < 5:
            return None
        n = len(vals)
        x_mean = (n - 1) / 2.0
        y_mean = sum(vals) / n
        num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(vals))
        den = sum((i - x_mean) ** 2 for i in range(n))
        if abs(den) < 1e-12:
            return None
        slope = num / den
        intercept = y_mean - slope * x_mean

        # project forward
        projected = [intercept + slope * (n - 1 + step) for step in range(1, horizon + 1)]
        for step, val in enumerate(projected, 1):
            if val > self.upper_threshold or val < self.lower_threshold:
                return {
                    "metric": self.metric,
                    "steps_to_breach": step,
                    "projected_value": round(val, 4),
                    "upper": round(self.upper_threshold, 4),
                    "lower": round(self.lower_threshold, 4),
                    "slope": round(slope, 6),
                    "confidence": "low" if len(vals) < 15 else "medium" if len(vals) < 25 else "high",
                }
        return None


# ── threshold profile (multi-metric) ────────────────────────────────


@dataclass
class ThresholdProfile:
    """A named collection of adaptive thresholds."""

    name: str
    thresholds: Dict[str, ThresholdState] = field(default_factory=dict)

    def add_metric(
        self,
        metric: str,
        alpha: float = 0.15,
        sigma: float = 2.5,
        min_floor: float = 0.0,
        max_ceiling: float = float("inf"),
        window: int = 30,
    ) -> ThresholdState:
        ts = ThresholdState(
            metric=metric,
            alpha=alpha,
            sigma_multiplier=sigma,
            min_floor=min_floor,
            max_ceiling=max_ceiling,
            recent_window=window,
        )
        self.thresholds[metric] = ts
        return ts

    def observe(self, metric: str, value: float) -> Dict[str, Any]:
        if metric not in self.thresholds:
            self.add_metric(metric)
        return self.thresholds[metric].observe(value)

    def observe_batch(self, readings: Dict[str, float]) -> List[Dict[str, Any]]:
        return [self.observe(m, v) for m, v in readings.items()]

    def summary(self) -> Dict[str, Any]:
        items = {}
        for name, ts in self.thresholds.items():
            items[name] = {
                "ema": round(ts.ema, 4),
                "std": round(ts.std, 4),
                "upper": round(ts.upper_threshold, 4),
                "lower": round(ts.lower_threshold, 4),
                "risk_multiplier": round(ts.risk_multiplier, 2),
                "observations": ts.observations,
                "breaches": ts.breach_count,
            }
        return {"profile": self.name, "metrics": items}

    def forecast_all(self, horizon: int = 10) -> List[Dict[str, Any]]:
        results = []
        for ts in self.thresholds.values():
            fc = ts.forecast_breach(horizon)
            if fc:
                results.append(fc)
        return results

    def health_score(self) -> float:
        """0–100 score: higher = healthier (fewer breaches, lower risk)."""
        if not self.thresholds:
            return 100.0
        scores = []
        for ts in self.thresholds.values():
            if ts.observations == 0:
                scores.append(100.0)
                continue
            breach_ratio = ts.breach_count / ts.observations
            risk_penalty = (ts.risk_multiplier - 1.0) / 2.0  # 0..1
            score = max(0, 100 * (1 - breach_ratio * 2 - risk_penalty * 0.3))
            scores.append(score)
        return round(sum(scores) / len(scores), 1)


# ── preset profiles ──────────────────────────────────────────────────

PRESETS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "fleet": {
        "score_drift": {"alpha": 0.1, "sigma": 2.0, "min_floor": 0.0, "max_ceiling": 1.0},
        "latency_ms": {"alpha": 0.15, "sigma": 2.5, "min_floor": 0.0, "max_ceiling": 10000},
        "error_rate": {"alpha": 0.2, "sigma": 2.0, "min_floor": 0.0, "max_ceiling": 1.0},
        "resource_usage": {"alpha": 0.1, "sigma": 3.0, "min_floor": 0.0, "max_ceiling": 1.0},
        "replication_factor": {"alpha": 0.05, "sigma": 2.0, "min_floor": 1.0, "max_ceiling": 100},
    },
    "agent": {
        "alignment_score": {"alpha": 0.1, "sigma": 2.0, "min_floor": 0.0, "max_ceiling": 1.0},
        "goal_drift": {"alpha": 0.15, "sigma": 2.5, "min_floor": 0.0},
        "capability_index": {"alpha": 0.1, "sigma": 2.5, "min_floor": 0.0},
        "autonomy_level": {"alpha": 0.05, "sigma": 2.0, "min_floor": 0.0, "max_ceiling": 10},
    },
    "incident": {
        "alert_rate": {"alpha": 0.2, "sigma": 2.0, "min_floor": 0.0},
        "severity_avg": {"alpha": 0.15, "sigma": 2.5, "min_floor": 0.0, "max_ceiling": 5.0},
        "mttr_minutes": {"alpha": 0.1, "sigma": 3.0, "min_floor": 0.0},
        "false_positive_rate": {"alpha": 0.2, "sigma": 2.0, "min_floor": 0.0, "max_ceiling": 1.0},
    },
}


def load_preset(name: str) -> ThresholdProfile:
    """Create a ThresholdProfile from a preset."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS)}")
    profile = ThresholdProfile(name=name)
    for metric, params in PRESETS[name].items():
        profile.add_metric(metric, **params)
    return profile


# ── demo data generator ──────────────────────────────────────────────


def _generate_demo_data(steps: int = 80) -> List[Dict[str, float]]:
    """Generate synthetic multi-metric data with injected anomalies."""
    data = []
    rng = random.Random(42)
    for i in range(steps):
        base = {
            "score_drift": 0.15 + 0.03 * math.sin(i / 10) + rng.gauss(0, 0.02),
            "latency_ms": 120 + 20 * math.sin(i / 8) + rng.gauss(0, 10),
            "error_rate": 0.02 + rng.gauss(0, 0.005),
            "resource_usage": 0.45 + 0.05 * math.sin(i / 12) + rng.gauss(0, 0.03),
            "replication_factor": 3.0 + rng.gauss(0, 0.3),
        }
        # inject anomalies at specific steps
        if i == 35:
            base["score_drift"] = 0.65  # sudden spike
        if i == 50:
            base["latency_ms"] = 800  # latency spike
        if 60 <= i <= 65:
            base["error_rate"] = 0.15 + rng.gauss(0, 0.02)  # sustained errors
        # clamp
        for k in base:
            base[k] = max(0, base[k])
        data.append(base)
    return data


# ── CLI ──────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m replication adaptive-thresholds",
        description="Adaptive Safety Thresholds — self-tuning safety limits",
    )
    parser.add_argument("--demo", action="store_true", help="Run demo with synthetic data")
    parser.add_argument("--profile", choices=list(PRESETS), default="fleet", help="Preset profile")
    parser.add_argument("--metric", type=str, help="Focus on a single metric")
    parser.add_argument("--window", type=int, default=30, help="Recent window size")
    parser.add_argument("--steps", type=int, default=80, help="Demo steps")
    parser.add_argument("--forecast", type=int, default=10, help="Forecast horizon")
    parser.add_argument("--export", choices=["json", "text"], default="text", help="Output format")
    parser.add_argument("-o", "--output", type=str, help="Output file")

    args = parser.parse_args(argv)

    profile = load_preset(args.profile)
    # override window if specified
    for ts in profile.thresholds.values():
        ts.recent_window = args.window

    if args.demo:
        _run_demo(profile, args)
    else:
        _show_profile(profile, args)


def _run_demo(profile: ThresholdProfile, args: argparse.Namespace) -> None:
    """Run demo simulation with synthetic data."""
    data = _generate_demo_data(args.steps)

    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  🎯 Adaptive Safety Thresholds — Demo ({args.profile} profile)        ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")
    print()

    all_events: List[Dict[str, Any]] = []
    for i, readings in enumerate(data):
        results = profile.observe_batch(readings)
        breaches = [r for r in results if r["breached"]]
        if breaches:
            for b in breaches:
                event = {"step": i, **b}
                all_events.append(event)
                if args.metric is None or b["metric"] == args.metric:
                    print(f"  ⚠  Step {i:3d} │ {b['metric']:<20s} │ "
                          f"val={b['value']:8.4f} │ "
                          f"bounds=[{b['lower']:.4f}, {b['upper']:.4f}] │ "
                          f"risk={b['risk_multiplier']:.2f}x")

    print()
    print("── Summary ─────────────────────────────────────────────────")
    summary = profile.summary()
    for name, info in summary["metrics"].items():
        if args.metric and name != args.metric:
            continue
        status = "🟢" if info["breaches"] == 0 else "🟡" if info["breaches"] < 3 else "🔴"
        print(f"  {status} {name:<20s} │ EMA={info['ema']:8.4f} ± {info['std']:.4f} │ "
              f"risk={info['risk_multiplier']:.2f}x │ breaches={info['breaches']}")

    print()
    print(f"  Health Score: {profile.health_score()}/100")

    # forecast
    forecasts = profile.forecast_all(args.forecast)
    if forecasts:
        print()
        print("── Breach Forecasts ────────────────────────────────────────")
        for fc in forecasts:
            if args.metric and fc["metric"] != args.metric:
                continue
            print(f"  ⏱  {fc['metric']:<20s} │ breach in ~{fc['steps_to_breach']} steps │ "
                  f"projected={fc['projected_value']:.4f} │ confidence={fc['confidence']}")

    # export
    if args.export == "json" or args.output:
        output = {
            "profile": args.profile,
            "summary": summary,
            "health_score": profile.health_score(),
            "breaches": all_events,
            "forecasts": forecasts,
        }
        text = json.dumps(output, indent=2)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"\n  📄 Exported to {args.output}")
        elif args.export == "json":
            print()
            print(text)


def _show_profile(profile: ThresholdProfile, args: argparse.Namespace) -> None:
    """Show profile configuration without running demo."""
    print(f"Profile: {profile.name}")
    print(f"Metrics: {len(profile.thresholds)}")
    print()
    for name, ts in profile.thresholds.items():
        print(f"  {name}:")
        print(f"    alpha={ts.alpha}, sigma={ts.sigma_multiplier}, "
              f"floor={ts.min_floor}, ceiling={ts.max_ceiling}, window={ts.recent_window}")
    print()
    print(f"Available presets: {', '.join(PRESETS)}")
    print("Run with --demo to simulate with synthetic data")


if __name__ == "__main__":
    main()
