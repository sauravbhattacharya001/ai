"""Exposure Window Analyzer — quantify time periods with degraded safety coverage.

When safety controls fail, degrade, or get disabled, there's an *exposure
window* — the period where the system operates without adequate protection.
Understanding these windows is critical for risk assessment, incident
response, and compliance reporting.

This module analyzes control status logs to identify, measure, and report
exposure windows across safety controls.

Key Metrics
-----------
* **Total exposure time**: cumulative duration of all windows
* **Longest window**: single worst exposure period
* **MTTR** (Mean Time to Restore): average time to restore controls
* **Coverage ratio**: percentage of time with full protection
* **Concurrent exposures**: overlapping windows across controls
* **Risk-weighted exposure**: exposure × severity of unprotected area

Usage (CLI)::

    python -m replication exposure simulate --controls 5 --hours 168
    python -m replication exposure analyze --events events.json
    python -m replication exposure report --events events.json -o exposure.html
    python -m replication exposure summary --events events.json

Programmatic::

    from replication.exposure_window import ExposureAnalyzer, ControlEvent
    analyzer = ExposureAnalyzer()
    analyzer.ingest(events)
    result = analyzer.analyze()
    print(f"Coverage: {result.coverage_pct:.1f}% | Longest gap: {result.longest_window_min:.0f}min")
"""

from __future__ import annotations

import argparse
import html as _html
import json
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ._helpers import box_header as _box_header


# ── Data models ──────────────────────────────────────────────────────


@dataclass
class ControlEvent:
    """A control status change event."""

    timestamp: float  # Unix epoch
    control_name: str  # e.g. "rate_limiter", "auth_check"
    status: str  # "up" | "down" | "degraded"
    severity: str = "medium"  # low | medium | high | critical — severity if this control is absent
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExposureWindow:
    """A single exposure window for one control."""

    control_name: str
    start: float  # Unix epoch
    end: float  # Unix epoch — 0 means still open
    severity: str = "medium"
    reason: str = ""

    @property
    def duration_sec(self) -> float:
        if self.end <= 0:
            return 0.0
        return self.end - self.start

    @property
    def duration_min(self) -> float:
        return self.duration_sec / 60.0


SEVERITY_WEIGHTS = {"low": 1, "medium": 2, "high": 4, "critical": 8}


@dataclass
class AnalysisResult:
    """Aggregated exposure analysis result."""

    total_exposure_sec: float = 0.0
    longest_window_sec: float = 0.0
    longest_window_min: float = 0.0
    window_count: int = 0
    mttr_sec: float = 0.0
    coverage_pct: float = 100.0
    risk_weighted_exposure: float = 0.0
    max_concurrent: int = 0
    per_control: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    windows: List[ExposureWindow] = field(default_factory=list)
    observation_sec: float = 0.0
    grade: str = "A"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["windows"] = [asdict(w) for w in self.windows]
        return d


# ── Analyzer ─────────────────────────────────────────────────────────


class ExposureAnalyzer:
    """Analyze control status events to find exposure windows."""

    def __init__(self) -> None:
        self._events: List[ControlEvent] = []

    def ingest(self, events: Sequence[ControlEvent]) -> None:
        self._events.extend(events)

    def analyze(self, observation_end: Optional[float] = None) -> AnalysisResult:
        if not self._events:
            return AnalysisResult(grade="A", coverage_pct=100.0)

        sorted_events = sorted(self._events, key=lambda e: e.timestamp)
        obs_start = sorted_events[0].timestamp
        obs_end = observation_end or max(e.timestamp for e in sorted_events)
        obs_duration = max(obs_end - obs_start, 1.0)

        # Track open windows per control
        open_windows: Dict[str, Tuple[float, str, str]] = {}  # control -> (start, severity, reason)
        closed_windows: List[ExposureWindow] = []

        for ev in sorted_events:
            if ev.status in ("down", "degraded"):
                if ev.control_name not in open_windows:
                    open_windows[ev.control_name] = (ev.timestamp, ev.severity, ev.reason)
            elif ev.status == "up":
                if ev.control_name in open_windows:
                    start, sev, reason = open_windows.pop(ev.control_name)
                    closed_windows.append(ExposureWindow(
                        control_name=ev.control_name,
                        start=start,
                        end=ev.timestamp,
                        severity=sev,
                        reason=reason,
                    ))

        # Close still-open windows at observation end
        for ctrl, (start, sev, reason) in open_windows.items():
            closed_windows.append(ExposureWindow(
                control_name=ctrl, start=start, end=obs_end,
                severity=sev, reason=reason,
            ))

        # Metrics
        total_exposure = sum(w.duration_sec for w in closed_windows)
        longest = max((w.duration_sec for w in closed_windows), default=0.0)
        mttr = (total_exposure / len(closed_windows)) if closed_windows else 0.0
        coverage = max(0.0, 100.0 * (1.0 - total_exposure / obs_duration / max(1, len(self._control_names(sorted_events)))))
        risk_weighted = sum(
            w.duration_sec * SEVERITY_WEIGHTS.get(w.severity, 2)
            for w in closed_windows
        )

        # Max concurrent exposures
        max_conc = self._max_concurrent(closed_windows)

        # Per-control breakdown
        per_control: Dict[str, Dict[str, Any]] = {}
        for ctrl in self._control_names(sorted_events):
            ctrl_windows = [w for w in closed_windows if w.control_name == ctrl]
            ctrl_total = sum(w.duration_sec for w in ctrl_windows)
            per_control[ctrl] = {
                "window_count": len(ctrl_windows),
                "total_exposure_sec": round(ctrl_total, 1),
                "total_exposure_min": round(ctrl_total / 60, 1),
                "coverage_pct": round(100.0 * (1.0 - ctrl_total / obs_duration), 1),
                "longest_sec": round(max((w.duration_sec for w in ctrl_windows), default=0), 1),
            }

        # Grade
        grade = self._grade(coverage, longest, max_conc)

        return AnalysisResult(
            total_exposure_sec=round(total_exposure, 1),
            longest_window_sec=round(longest, 1),
            longest_window_min=round(longest / 60, 1),
            window_count=len(closed_windows),
            mttr_sec=round(mttr, 1),
            coverage_pct=round(coverage, 1),
            risk_weighted_exposure=round(risk_weighted, 1),
            max_concurrent=max_conc,
            per_control=per_control,
            windows=closed_windows,
            observation_sec=round(obs_duration, 1),
            grade=grade,
        )

    @staticmethod
    def _control_names(events: List[ControlEvent]) -> List[str]:
        seen: Dict[str, None] = {}
        for e in events:
            seen.setdefault(e.control_name, None)
        return list(seen)

    @staticmethod
    def _max_concurrent(windows: List[ExposureWindow]) -> int:
        if not windows:
            return 0
        events: List[Tuple[float, int]] = []
        for w in windows:
            events.append((w.start, 1))
            events.append((w.end, -1))
        events.sort(key=lambda x: (x[0], x[1]))
        current = 0
        maximum = 0
        for _, delta in events:
            current += delta
            maximum = max(maximum, current)
        return maximum

    @staticmethod
    def _grade(coverage: float, longest_sec: float, max_concurrent: int) -> str:
        if coverage >= 99.0 and longest_sec < 300:
            return "A"
        if coverage >= 95.0 and longest_sec < 1800:
            return "B"
        if coverage >= 90.0 and max_concurrent <= 2:
            return "C"
        if coverage >= 80.0:
            return "D"
        return "F"


# ── Simulator ────────────────────────────────────────────────────────


def simulate_events(
    num_controls: int = 5,
    hours: int = 168,
    failure_rate: float = 0.03,
    seed: Optional[int] = None,
) -> List[ControlEvent]:
    """Generate synthetic control status events."""
    rng = random.Random(seed)
    control_names = [
        "rate_limiter", "auth_check", "input_validator",
        "output_filter", "audit_logger", "resource_monitor",
        "anomaly_detector", "kill_switch", "encryption",
        "access_control",
    ][:num_controls]
    severities = ["low", "medium", "high", "critical"]
    reasons = [
        "config change", "deployment", "dependency failure",
        "resource exhaustion", "network issue", "bug",
        "maintenance window", "overload",
    ]

    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours)
    events: List[ControlEvent] = []

    # Initial up events
    for ctrl in control_names:
        events.append(ControlEvent(
            timestamp=start.timestamp(),
            control_name=ctrl,
            status="up",
        ))

    # Simulate failures and recoveries
    t = start.timestamp()
    end_t = now.timestamp()
    while t < end_t:
        t += rng.expovariate(failure_rate) * 3600  # hours between failures
        if t >= end_t:
            break
        ctrl = rng.choice(control_names)
        sev = rng.choices(severities, weights=[4, 3, 2, 1])[0]
        events.append(ControlEvent(
            timestamp=t,
            control_name=ctrl,
            status=rng.choice(["down", "degraded"]),
            severity=sev,
            reason=rng.choice(reasons),
        ))
        # Recovery after some time
        recovery_delay = rng.expovariate(1.0 / (SEVERITY_WEIGHTS.get(sev, 2) * 15)) * 60
        recovery_t = t + recovery_delay
        if recovery_t < end_t:
            events.append(ControlEvent(
                timestamp=recovery_t,
                control_name=ctrl,
                status="up",
            ))

    events.sort(key=lambda e: e.timestamp)
    return events


# ── Report ───────────────────────────────────────────────────────────


def generate_html_report(result: AnalysisResult) -> str:
    """Generate a self-contained HTML report."""
    rows = ""
    for ctrl, info in sorted(result.per_control.items()):
        rows += f"""<tr>
            <td>{_html.escape(ctrl)}</td>
            <td>{info['window_count']}</td>
            <td>{info['total_exposure_min']:.1f} min</td>
            <td>{info['coverage_pct']:.1f}%</td>
            <td>{info['longest_sec']:.0f}s</td>
        </tr>"""

    window_rows = ""
    for w in sorted(result.windows, key=lambda x: -x.duration_sec)[:20]:
        start_str = datetime.fromtimestamp(w.start, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        window_rows += f"""<tr>
            <td>{_html.escape(w.control_name)}</td>
            <td>{start_str}</td>
            <td>{w.duration_min:.1f} min</td>
            <td><span class="sev-{_html.escape(w.severity)}">{_html.escape(w.severity)}</span></td>
            <td>{_html.escape(w.reason)}</td>
        </tr>"""

    grade_color = {"A": "#27ae60", "B": "#2ecc71", "C": "#f39c12", "D": "#e67e22", "F": "#e74c3c"}.get(result.grade, "#95a5a6")

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Exposure Window Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem; background: #f7f8fa; color: #2d3436; }}
h1 {{ color: #2d3436; }}
.grade {{ display: inline-block; font-size: 3rem; font-weight: bold; color: white;
    background: {grade_color}; border-radius: 12px; padding: 0.3em 0.6em; }}
.metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin: 1.5rem 0; }}
.metric {{ background: white; border-radius: 8px; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.metric .value {{ font-size: 1.5rem; font-weight: bold; color: #2c3e50; }}
.metric .label {{ font-size: 0.85rem; color: #7f8c8d; }}
table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin: 1rem 0; }}
th {{ background: #2d3436; color: white; padding: 0.75rem; text-align: left; }}
td {{ padding: 0.6rem 0.75rem; border-bottom: 1px solid #eee; }}
tr:hover {{ background: #f0f0f0; }}
.sev-critical {{ color: #e74c3c; font-weight: bold; }}
.sev-high {{ color: #e67e22; font-weight: bold; }}
.sev-medium {{ color: #f39c12; }}
.sev-low {{ color: #27ae60; }}
</style></head><body>
<h1>🛡️ Exposure Window Analysis</h1>
<div class="grade">{result.grade}</div>
<div class="metrics">
    <div class="metric"><div class="value">{result.coverage_pct:.1f}%</div><div class="label">Coverage</div></div>
    <div class="metric"><div class="value">{result.window_count}</div><div class="label">Exposure Windows</div></div>
    <div class="metric"><div class="value">{result.longest_window_min:.0f} min</div><div class="label">Longest Window</div></div>
    <div class="metric"><div class="value">{result.mttr_sec/60:.0f} min</div><div class="label">Mean Time to Restore</div></div>
    <div class="metric"><div class="value">{result.max_concurrent}</div><div class="label">Max Concurrent Gaps</div></div>
    <div class="metric"><div class="value">{result.risk_weighted_exposure/3600:.1f}h</div><div class="label">Risk-Weighted Exposure</div></div>
</div>
<h2>Per-Control Breakdown</h2>
<table><thead><tr><th>Control</th><th>Windows</th><th>Total Exposure</th><th>Coverage</th><th>Longest</th></tr></thead>
<tbody>{rows}</tbody></table>
<h2>Top Exposure Windows</h2>
<table><thead><tr><th>Control</th><th>Start (UTC)</th><th>Duration</th><th>Severity</th><th>Reason</th></tr></thead>
<tbody>{window_rows}</tbody></table>
<p style="color:#aaa;font-size:0.8rem;">Generated by AI Replication Sandbox — Exposure Window Analyzer</p>
</body></html>"""


# ── CLI ──────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="replication exposure",
        description="Analyze safety control exposure windows",
    )
    sub = parser.add_subparsers(dest="cmd")

    # simulate
    sim = sub.add_parser("simulate", help="Generate and analyze synthetic control events")
    sim.add_argument("--controls", type=int, default=5, help="Number of controls (default: 5)")
    sim.add_argument("--hours", type=int, default=168, help="Observation window in hours (default: 168)")
    sim.add_argument("--rate", type=float, default=0.03, help="Failure rate per control per hour")
    sim.add_argument("--seed", type=int, default=None, help="Random seed")
    sim.add_argument("--json", action="store_true", help="Output raw JSON")

    # analyze
    ana = sub.add_parser("analyze", help="Analyze events from a JSON file")
    ana.add_argument("--events", required=True, help="Path to events JSON file")
    ana.add_argument("--json", action="store_true", help="Output raw JSON")

    # report
    rep = sub.add_parser("report", help="Generate HTML report")
    rep.add_argument("--events", default=None, help="Path to events JSON (simulates if omitted)")
    rep.add_argument("-o", "--output", default="exposure-report.html", help="Output path")
    rep.add_argument("--controls", type=int, default=5)
    rep.add_argument("--hours", type=int, default=168)

    # summary
    summ = sub.add_parser("summary", help="One-line summary from events file")
    summ.add_argument("--events", required=True, help="Path to events JSON file")

    args = parser.parse_args(argv)
    if not args.cmd:
        parser.print_help()
        return

    if args.cmd == "simulate":
        events = simulate_events(args.controls, args.hours, args.rate, args.seed)
        analyzer = ExposureAnalyzer()
        analyzer.ingest(events)
        result = analyzer.analyze()
        if getattr(args, "json", False):
            print(json.dumps(result.to_dict(), indent=2))
        else:
            _print_result(result)

    elif args.cmd == "analyze":
        events = _load_events(args.events)
        analyzer = ExposureAnalyzer()
        analyzer.ingest(events)
        result = analyzer.analyze()
        if getattr(args, "json", False):
            print(json.dumps(result.to_dict(), indent=2))
        else:
            _print_result(result)

    elif args.cmd == "report":
        if args.events:
            events = _load_events(args.events)
        else:
            events = simulate_events(args.controls, args.hours)
        analyzer = ExposureAnalyzer()
        analyzer.ingest(events)
        result = analyzer.analyze()
        html = generate_html_report(result)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Report written to {args.output}")

    elif args.cmd == "summary":
        events = _load_events(args.events)
        analyzer = ExposureAnalyzer()
        analyzer.ingest(events)
        result = analyzer.analyze()
        print(f"Grade {result.grade} | Coverage {result.coverage_pct:.1f}% | "
              f"{result.window_count} windows | MTTR {result.mttr_sec/60:.0f}min | "
              f"Longest {result.longest_window_min:.0f}min | "
              f"Max concurrent {result.max_concurrent}")


def _load_events(path: str) -> List[ControlEvent]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [ControlEvent(**e) for e in data]


def _print_result(result: AnalysisResult) -> None:
    print(_box_header("Exposure Window Analysis"))
    print(f"  Grade:               {result.grade}")
    print(f"  Coverage:            {result.coverage_pct:.1f}%")
    print(f"  Observation period:  {result.observation_sec/3600:.1f}h")
    print(f"  Exposure windows:    {result.window_count}")
    print(f"  Total exposure:      {result.total_exposure_sec/60:.1f} min")
    print(f"  Longest window:      {result.longest_window_min:.1f} min")
    print(f"  MTTR:                {result.mttr_sec/60:.1f} min")
    print(f"  Max concurrent:      {result.max_concurrent}")
    print(f"  Risk-weighted:       {result.risk_weighted_exposure/3600:.2f}h")
    print()
    print("  Per-control breakdown:")
    for ctrl, info in sorted(result.per_control.items()):
        print(f"    {ctrl:20s}  windows={info['window_count']:2d}  "
              f"exposure={info['total_exposure_min']:6.1f}min  "
              f"coverage={info['coverage_pct']:5.1f}%")


if __name__ == "__main__":
    main()
