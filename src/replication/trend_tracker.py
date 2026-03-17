"""Safety Trend Tracker — record and analyze safety scorecard trends over time.

Track how safety posture evolves across runs, detect regressions, and
visualize improvement trajectories.  Results are stored in a JSON-lines
file so they survive process restarts.

CLI usage::

    # Record a new scorecard snapshot
    python -m replication trend record
    python -m replication trend record --strategy greedy --max-depth 5
    python -m replication trend record --tag "after-policy-update"

    # View trend summary
    python -m replication trend show
    python -m replication trend show --last 10

    # Show per-dimension trends
    python -m replication trend dimensions
    python -m replication trend dimensions --dimension "Contract Enforcement"

    # Export trend data as JSON
    python -m replication trend export --output trends.json

    # Detect regressions against a baseline
    python -m replication trend check --threshold 5

    # Clear history
    python -m replication trend clear

Programmatic::

    from replication.trend_tracker import TrendTracker
    tracker = TrendTracker("safety_trends.jsonl")
    tracker.record()  # run scorecard & store result
    summary = tracker.summary(last_n=10)
    print(summary.render())
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .scorecard import SafetyScorecard, ScorecardConfig, ScorecardResult
from .simulator import ScenarioConfig, PRESETS
from ._helpers import box_header as _box_header


# ── Data models ──────────────────────────────────────────────────────


@dataclass
class TrendEntry:
    """A single recorded scorecard snapshot."""

    timestamp: str
    overall_score: float
    overall_grade: str
    dimensions: Dict[str, float]  # dimension name → score
    config_summary: str
    tag: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "overall_score": self.overall_score,
            "overall_grade": self.overall_grade,
            "dimensions": self.dimensions,
            "config_summary": self.config_summary,
            "tag": self.tag,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrendEntry":
        return cls(
            timestamp=d["timestamp"],
            overall_score=d["overall_score"],
            overall_grade=d["overall_grade"],
            dimensions=d.get("dimensions", {}),
            config_summary=d.get("config_summary", ""),
            tag=d.get("tag", ""),
        )

    @classmethod
    def from_scorecard(
        cls, result: ScorecardResult, tag: str = ""
    ) -> "TrendEntry":
        dims = {d.name: d.score for d in result.dimensions}
        config_str = (
            f"{result.config.strategy}, depth={result.config.max_depth}, "
            f"replicas={result.config.max_replicas}"
        )
        return cls(
            timestamp=result.timestamp,
            overall_score=result.overall_score,
            overall_grade=result.overall_grade,
            dimensions=dims,
            config_summary=config_str,
            tag=tag,
        )


@dataclass
class TrendSummary:
    """Aggregated trend analysis."""

    entries: List[TrendEntry]
    direction: str  # "improving", "declining", "stable", "insufficient_data"
    score_delta: float  # change from first to last
    best: Optional[TrendEntry] = None
    worst: Optional[TrendEntry] = None
    average_score: float = 0.0
    dimension_trends: Dict[str, str] = field(default_factory=dict)

    def render(self) -> str:
        lines: List[str] = []
        lines.extend(_box_header("📈  Safety Trend Report  📈"))
        lines.append("")

        if not self.entries:
            lines.append("  No trend data recorded yet.")
            lines.append("  Run: python -m replication trend record")
            return "\n".join(lines)

        n = len(self.entries)
        icon = {"improving": "✅", "declining": "⚠️", "stable": "➡️"}.get(
            self.direction, "❓"
        )

        lines.append(f"  Snapshots: {n}")
        lines.append(f"  Direction: {icon} {self.direction.upper()}")
        lines.append(f"  Score Δ:   {self.score_delta:+.1f}")
        lines.append(f"  Average:   {self.average_score:.1f}")
        lines.append("")

        if self.best:
            lines.append(
                f"  Best:  {self.best.overall_grade} "
                f"({self.best.overall_score:.1f}) — {self.best.timestamp}"
            )
        if self.worst:
            lines.append(
                f"  Worst: {self.worst.overall_grade} "
                f"({self.worst.overall_score:.1f}) — {self.worst.timestamp}"
            )

        lines.append("")

        # Recent history (last 10)
        recent = self.entries[-10:]
        lines.append(f"  {'Timestamp':<22} {'Score':>6} {'Grade':>6}  {'Tag'}")
        lines.append("  " + "─" * 60)
        for e in recent:
            tag_str = e.tag if e.tag else ""
            lines.append(
                f"  {e.timestamp[:19]:<22} {e.overall_score:>5.1f} "
                f"{e.overall_grade:>6}  {tag_str}"
            )

        # Sparkline
        if n >= 3:
            lines.append("")
            lines.append("  Score Trend:")
            lines.append(f"  {_sparkline([e.overall_score for e in self.entries[-20:]])}")

        # Dimension trends
        if self.dimension_trends:
            lines.append("")
            lines.append("  Per-Dimension Trends:")
            for dim, trend in sorted(self.dimension_trends.items()):
                icon_d = {"improving": "↑", "declining": "↓", "stable": "→"}.get(
                    trend, "?"
                )
                lines.append(f"    {icon_d} {dim}: {trend}")

        return "\n".join(lines)


def _sparkline(values: List[float]) -> str:
    """Render a simple Unicode sparkline."""
    if not values:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    spread = hi - lo if hi != lo else 1
    return "".join(blocks[min(int((v - lo) / spread * 7), 7)] for v in values)


@dataclass
class RegressionAlert:
    """A detected safety regression."""

    dimension: str
    previous_score: float
    current_score: float
    delta: float

    def render(self) -> str:
        return (
            f"  ⚠️  {self.dimension}: {self.previous_score:.1f} → "
            f"{self.current_score:.1f} (Δ {self.delta:+.1f})"
        )


# ── Core tracker ─────────────────────────────────────────────────────


class TrendTracker:
    """Records and analyzes safety scorecard trends."""

    def __init__(self, path: str = "safety_trends.jsonl"):
        self.path = Path(path)
        self._entries: Optional[List[TrendEntry]] = None

    def _load(self) -> List[TrendEntry]:
        if self._entries is not None:
            return self._entries
        entries: List[TrendEntry] = []
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(TrendEntry.from_dict(json.loads(line)))
                        except (json.JSONDecodeError, KeyError):
                            continue
        self._entries = entries
        return entries

    def _append(self, entry: TrendEntry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")
        if self._entries is not None:
            self._entries.append(entry)

    def record(
        self,
        scenario: Optional[ScenarioConfig] = None,
        scorecard_config: Optional[ScorecardConfig] = None,
        tag: str = "",
    ) -> TrendEntry:
        """Run a scorecard evaluation and record the result."""
        sc = SafetyScorecard()
        cfg = scorecard_config or ScorecardConfig()
        scenario = scenario or ScenarioConfig()
        result = sc.evaluate(scenario, cfg)
        entry = TrendEntry.from_scorecard(result, tag=tag)
        self._append(entry)
        return entry

    def record_result(self, result: ScorecardResult, tag: str = "") -> TrendEntry:
        """Record an existing scorecard result without re-running."""
        entry = TrendEntry.from_scorecard(result, tag=tag)
        self._append(entry)
        return entry

    def entries(self, last_n: Optional[int] = None) -> List[TrendEntry]:
        """Return recorded entries, optionally limited to last N."""
        all_entries = self._load()
        if last_n is not None:
            return all_entries[-last_n:]
        return list(all_entries)

    def summary(self, last_n: Optional[int] = None) -> TrendSummary:
        """Compute a trend summary."""
        entries = self.entries(last_n)
        if not entries:
            return TrendSummary(
                entries=[], direction="insufficient_data", score_delta=0.0
            )

        scores = [e.overall_score for e in entries]
        avg = sum(scores) / len(scores)
        best = max(entries, key=lambda e: e.overall_score)
        worst = min(entries, key=lambda e: e.overall_score)

        if len(entries) < 2:
            direction = "insufficient_data"
            delta = 0.0
        else:
            delta = scores[-1] - scores[0]
            if delta > 2:
                direction = "improving"
            elif delta < -2:
                direction = "declining"
            else:
                direction = "stable"

        # Per-dimension trends
        dim_trends: Dict[str, str] = {}
        if len(entries) >= 2:
            all_dims = set()
            for e in entries:
                all_dims.update(e.dimensions.keys())
            for dim in sorted(all_dims):
                dim_scores = [
                    e.dimensions[dim] for e in entries if dim in e.dimensions
                ]
                if len(dim_scores) >= 2:
                    d = dim_scores[-1] - dim_scores[0]
                    if d > 2:
                        dim_trends[dim] = "improving"
                    elif d < -2:
                        dim_trends[dim] = "declining"
                    else:
                        dim_trends[dim] = "stable"

        return TrendSummary(
            entries=entries,
            direction=direction,
            score_delta=delta,
            best=best,
            worst=worst,
            average_score=avg,
            dimension_trends=dim_trends,
        )

    def check_regressions(
        self, threshold: float = 5.0
    ) -> List[RegressionAlert]:
        """Compare latest entry against the previous one for regressions."""
        entries = self._load()
        if len(entries) < 2:
            return []

        prev, curr = entries[-2], entries[-1]
        alerts: List[RegressionAlert] = []

        # Overall
        delta = curr.overall_score - prev.overall_score
        if delta < -threshold:
            alerts.append(
                RegressionAlert(
                    dimension="Overall",
                    previous_score=prev.overall_score,
                    current_score=curr.overall_score,
                    delta=delta,
                )
            )

        # Per-dimension
        for dim in prev.dimensions:
            if dim in curr.dimensions:
                d = curr.dimensions[dim] - prev.dimensions[dim]
                if d < -threshold:
                    alerts.append(
                        RegressionAlert(
                            dimension=dim,
                            previous_score=prev.dimensions[dim],
                            current_score=curr.dimensions[dim],
                            delta=d,
                        )
                    )

        return alerts

    def export_json(self, last_n: Optional[int] = None) -> str:
        """Export entries as a JSON array."""
        entries = self.entries(last_n)
        return json.dumps([e.to_dict() for e in entries], indent=2)

    def clear(self) -> int:
        """Remove all entries. Returns count of cleared entries."""
        entries = self._load()
        count = len(entries)
        if self.path.exists():
            self.path.unlink()
        self._entries = []
        return count


# ── CLI ──────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for trend tracker."""
    parser = argparse.ArgumentParser(
        prog="python -m replication trend",
        description="Safety Trend Tracker — record and analyze scorecard trends",
    )
    sub = parser.add_subparsers(dest="subcmd", help="action")

    # record
    p_rec = sub.add_parser("record", help="Record a new scorecard snapshot")
    p_rec.add_argument("--strategy", default="balanced", help="Simulation strategy")
    p_rec.add_argument("--max-depth", type=int, default=3, help="Max replication depth")
    p_rec.add_argument("--max-replicas", type=int, default=10, help="Max replicas")
    p_rec.add_argument("--tag", default="", help="Tag for this snapshot")
    p_rec.add_argument("--file", default="safety_trends.jsonl", help="Trend file")
    p_rec.add_argument("--quick", action="store_true", help="Skip slow analyses")

    # show
    p_show = sub.add_parser("show", help="Show trend summary")
    p_show.add_argument("--last", type=int, default=None, help="Last N entries")
    p_show.add_argument("--file", default="safety_trends.jsonl", help="Trend file")

    # dimensions
    p_dims = sub.add_parser("dimensions", help="Show per-dimension trends")
    p_dims.add_argument("--dimension", default=None, help="Filter to one dimension")
    p_dims.add_argument("--last", type=int, default=None, help="Last N entries")
    p_dims.add_argument("--file", default="safety_trends.jsonl", help="Trend file")

    # export
    p_exp = sub.add_parser("export", help="Export trend data as JSON")
    p_exp.add_argument("--output", "-o", default=None, help="Output file (default: stdout)")
    p_exp.add_argument("--last", type=int, default=None, help="Last N entries")
    p_exp.add_argument("--file", default="safety_trends.jsonl", help="Trend file")

    # check
    p_chk = sub.add_parser("check", help="Check for regressions")
    p_chk.add_argument("--threshold", type=float, default=5.0, help="Regression threshold")
    p_chk.add_argument("--file", default="safety_trends.jsonl", help="Trend file")

    # clear
    p_clr = sub.add_parser("clear", help="Clear all trend history")
    p_clr.add_argument("--file", default="safety_trends.jsonl", help="Trend file")

    args = parser.parse_args(argv)

    if not args.subcmd:
        parser.print_help()
        return

    tracker = TrendTracker(getattr(args, "file", "safety_trends.jsonl"))

    if args.subcmd == "record":
        scenario = ScenarioConfig(
            strategy=args.strategy,
            max_depth=args.max_depth,
            max_replicas=args.max_replicas,
        )
        sc_cfg = ScorecardConfig(quick=args.quick)
        print("Running scorecard evaluation...")
        entry = tracker.record(scenario=scenario, scorecard_config=sc_cfg, tag=args.tag)
        print(f"✅ Recorded: {entry.overall_grade} ({entry.overall_score:.1f}/100)")
        if args.tag:
            print(f"   Tag: {args.tag}")

    elif args.subcmd == "show":
        summary = tracker.summary(last_n=args.last)
        print(summary.render())

    elif args.subcmd == "dimensions":
        entries = tracker.entries(last_n=args.last)
        if not entries:
            print("No trend data. Run: python -m replication trend record")
            return

        all_dims: Dict[str, List[float]] = {}
        for e in entries:
            for dim, score in e.dimensions.items():
                all_dims.setdefault(dim, []).append(score)

        if args.dimension:
            if args.dimension not in all_dims:
                print(f"Dimension '{args.dimension}' not found.")
                print(f"Available: {', '.join(sorted(all_dims))}")
                return
            all_dims = {args.dimension: all_dims[args.dimension]}

        lines: List[str] = []
        lines.extend(_box_header("📊  Dimension Trends  📊"))
        lines.append("")
        for dim, scores in sorted(all_dims.items()):
            avg = sum(scores) / len(scores)
            trend = ""
            if len(scores) >= 2:
                d = scores[-1] - scores[0]
                if d > 2:
                    trend = "↑ improving"
                elif d < -2:
                    trend = "↓ declining"
                else:
                    trend = "→ stable"
            spark = _sparkline(scores[-20:]) if len(scores) >= 3 else ""
            lines.append(f"  {dim}")
            lines.append(
                f"    Latest: {scores[-1]:.1f}  Avg: {avg:.1f}  "
                f"Samples: {len(scores)}  {trend}"
            )
            if spark:
                lines.append(f"    {spark}")
            lines.append("")
        print("\n".join(lines))

    elif args.subcmd == "export":
        data = tracker.export_json(last_n=getattr(args, "last", None))
        if args.output:
            Path(args.output).write_text(data, encoding="utf-8")
            print(f"Exported to {args.output}")
        else:
            print(data)

    elif args.subcmd == "check":
        alerts = tracker.check_regressions(threshold=args.threshold)
        if not alerts:
            print("✅ No regressions detected.")
        else:
            print(f"⚠️  {len(alerts)} regression(s) detected:\n")
            for a in alerts:
                print(a.render())
            sys.exit(1)

    elif args.subcmd == "clear":
        count = tracker.clear()
        print(f"Cleared {count} entries.")
