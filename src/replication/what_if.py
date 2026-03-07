"""What-If Analysis — explore hypothetical configuration changes.

Given a baseline configuration, applies one or more hypothetical
parameter changes, runs simulations for each, and reports the
safety impact of each change.  Helps users answer questions like
"what happens if I increase max_depth by 2?" or "what if I remove
the cooldown entirely?" without manually running and comparing
multiple simulations.

Features
--------
- **Single-parameter analysis:**  Change one parameter, see the safety
  delta vs baseline.
- **Multi-parameter analysis:**  Apply several changes together to see
  combined effects.
- **Sweep mode:**  Automatically sweep a parameter through a range and
  show the safety curve (e.g., max_depth from 1 to 10).
- **Risk ranking:**  Rank proposed changes from safest to most dangerous.
- **Recommendation engine:**  For each change, emit SAFE / CAUTION /
  DANGEROUS verdict with human-readable reasoning.

Usage (CLI)::

    python -m replication.what_if                                     # interactive example
    python -m replication.what_if --change max_depth=5                # single change
    python -m replication.what_if --change max_depth=5 --change cooldown_seconds=0
    python -m replication.what_if --sweep max_depth 1 10              # sweep 1..10
    python -m replication.what_if --sweep replication_probability 0.1 1.0 --steps 5
    python -m replication.what_if --baseline balanced                 # use preset baseline
    python -m replication.what_if --json                              # JSON output
    python -m replication.what_if --seed 42                           # reproducible

Programmatic::

    from replication.what_if import WhatIfAnalyzer, WhatIfConfig
    analyzer = WhatIfAnalyzer()
    result = analyzer.analyze(
        changes={"max_depth": 5, "cooldown_seconds": 0},
    )
    print(result.render())

    # Sweep a parameter
    sweep = analyzer.sweep("max_depth", start=1, stop=10)
    print(sweep.render())
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .simulator import Simulator, ScenarioConfig, SimulationReport, PRESETS


# ── Enums ────────────────────────────────────────────────────────────


class RiskVerdict(Enum):
    """Safety verdict for a hypothetical change."""
    SAFE = "safe"
    CAUTION = "caution"
    DANGEROUS = "dangerous"


class MetricPolarity(Enum):
    """Whether higher values are better or worse for safety."""
    LOWER_IS_BETTER = "lower_is_better"
    HIGHER_IS_BETTER = "higher_is_better"


# ── Metric definitions ──────────────────────────────────────────────

# (extractor, polarity, display_name, format_spec, weight)
# Weight determines how much this metric affects the risk score (0-1).
METRICS: Dict[str, Tuple[str, MetricPolarity, str, str, float]] = {
    "peak_workers": (
        "_m_peak_workers", MetricPolarity.LOWER_IS_BETTER,
        "Peak Workers", "d", 0.20,
    ),
    "max_depth_reached": (
        "_m_max_depth", MetricPolarity.LOWER_IS_BETTER,
        "Max Depth Reached", "d", 0.25,
    ),
    "replication_success_rate": (
        "_m_repl_rate", MetricPolarity.LOWER_IS_BETTER,
        "Replication Success Rate (%)", ".1f", 0.25,
    ),
    "denial_rate": (
        "_m_denial_rate", MetricPolarity.HIGHER_IS_BETTER,
        "Denial Rate (%)", ".1f", 0.15,
    ),
    "total_replications": (
        "_m_total_repl", MetricPolarity.LOWER_IS_BETTER,
        "Total Replications", "d", 0.10,
    ),
    "efficiency": (
        "_m_efficiency", MetricPolarity.HIGHER_IS_BETTER,
        "Task Efficiency", ".2f", 0.05,
    ),
}


# ── Data classes ─────────────────────────────────────────────────────


@dataclass
class MetricDelta:
    """Change in a single metric between baseline and variant."""
    name: str
    display_name: str
    baseline_value: float
    variant_value: float
    absolute_change: float
    percent_change: float
    polarity: MetricPolarity
    worsened: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "baseline": round(self.baseline_value, 4),
            "variant": round(self.variant_value, 4),
            "absolute_change": round(self.absolute_change, 4),
            "percent_change": round(self.percent_change, 2),
            "worsened": self.worsened,
        }


@dataclass
class ChangeAnalysis:
    """Analysis of a single hypothetical change (one or more params)."""
    label: str
    changes: Dict[str, Any]
    baseline_config: Dict[str, Any]
    variant_config: Dict[str, Any]
    deltas: List[MetricDelta]
    risk_score: float  # 0 = no impact, 100 = maximum risk increase
    verdict: RiskVerdict
    reasoning: str
    baseline_report: Optional[SimulationReport] = None
    variant_report: Optional[SimulationReport] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "changes": self.changes,
            "deltas": [d.to_dict() for d in self.deltas],
            "risk_score": round(self.risk_score, 1),
            "verdict": self.verdict.value,
            "reasoning": self.reasoning,
        }


@dataclass
class WhatIfResult:
    """Full what-if analysis result with all changes analyzed."""
    baseline_config: Dict[str, Any]
    analyses: List[ChangeAnalysis]
    ranking: List[ChangeAnalysis]  # sorted safest-to-most-dangerous
    elapsed_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline": self.baseline_config,
            "analyses": [a.to_dict() for a in self.analyses],
            "ranking": [
                {"rank": i + 1, "label": a.label,
                 "risk_score": round(a.risk_score, 1),
                 "verdict": a.verdict.value}
                for i, a in enumerate(self.ranking)
            ],
            "elapsed_ms": round(self.elapsed_ms, 1),
        }

    def render(self) -> str:
        """Render human-readable what-if report."""
        lines: List[str] = []
        lines.append("")
        lines.append("=" * 64)
        lines.append("  WHAT-IF ANALYSIS REPORT")
        lines.append("=" * 64)
        lines.append("")

        # Baseline summary
        lines.append("Baseline Configuration:")
        for k, v in sorted(self.baseline_config.items()):
            if k != "secret":
                lines.append(f"  {k}: {v}")
        lines.append("")

        # Each analysis
        for i, a in enumerate(self.analyses, 1):
            lines.append("-" * 64)
            icon = {"safe": "✅", "caution": "⚠️", "dangerous": "🚨"}.get(
                a.verdict.value, "❓")
            lines.append(f"  Scenario {i}: {a.label}")
            lines.append(f"  Verdict: {icon} {a.verdict.value.upper()} "
                         f"(risk score: {a.risk_score:.1f}/100)")
            lines.append(f"  Reasoning: {a.reasoning}")
            lines.append("")

            # Changes applied
            lines.append("  Changes applied:")
            for param, val in a.changes.items():
                baseline_val = a.baseline_config.get(param, "N/A")
                lines.append(f"    {param}: {baseline_val} → {val}")
            lines.append("")

            # Metric deltas
            lines.append("  Impact on safety metrics:")
            for d in a.deltas:
                direction = "↑" if d.absolute_change > 0 else "↓" if d.absolute_change < 0 else "="
                warn = " ⚠" if d.worsened else ""
                pct = f" ({d.percent_change:+.1f}%)" if d.baseline_value != 0 else ""
                fmt = METRICS.get(d.name, (None, None, None, ".2f", 0))[3]
                bv = format(int(d.baseline_value), fmt) if fmt == "d" else format(d.baseline_value, fmt)
                vv = format(int(d.variant_value), fmt) if fmt == "d" else format(d.variant_value, fmt)
                lines.append(
                    f"    {d.display_name}: {bv} → {vv} "
                    f"{direction}{pct}{warn}"
                )
            lines.append("")

        # Ranking
        if len(self.ranking) > 1:
            lines.append("=" * 64)
            lines.append("  RISK RANKING (safest → most dangerous)")
            lines.append("=" * 64)
            for i, a in enumerate(self.ranking, 1):
                icon = {"safe": "✅", "caution": "⚠️",
                        "dangerous": "🚨"}.get(a.verdict.value, "❓")
                lines.append(
                    f"  {i}. {a.label} — {icon} {a.verdict.value} "
                    f"(risk: {a.risk_score:.1f})")
            lines.append("")

        lines.append(f"Analysis completed in {self.elapsed_ms:.0f}ms")
        lines.append("")
        return "\n".join(lines)


@dataclass
class SweepPoint:
    """A single point in a parameter sweep."""
    value: Any
    risk_score: float
    verdict: RiskVerdict
    metrics: Dict[str, float]


@dataclass
class SweepResult:
    """Result of sweeping a parameter through a range of values."""
    parameter: str
    baseline_value: Any
    points: List[SweepPoint]
    safest: SweepPoint
    riskiest: SweepPoint
    tipping_point: Optional[Any]  # value where verdict changes to DANGEROUS
    elapsed_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter": self.parameter,
            "baseline_value": self.baseline_value,
            "points": [
                {"value": p.value, "risk_score": round(p.risk_score, 1),
                 "verdict": p.verdict.value, "metrics": p.metrics}
                for p in self.points
            ],
            "safest": {"value": self.safest.value,
                       "risk_score": round(self.safest.risk_score, 1)},
            "riskiest": {"value": self.riskiest.value,
                         "risk_score": round(self.riskiest.risk_score, 1)},
            "tipping_point": self.tipping_point,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }

    def render(self) -> str:
        """Render human-readable sweep report."""
        lines: List[str] = []
        lines.append("")
        lines.append("=" * 64)
        lines.append(f"  PARAMETER SWEEP: {self.parameter}")
        lines.append(f"  Baseline value: {self.baseline_value}")
        lines.append("=" * 64)
        lines.append("")

        # Sweep table
        lines.append(f"  {'Value':>10}  {'Risk':>6}  {'Verdict':<12}  Bar")
        lines.append(f"  {'─' * 10}  {'─' * 6}  {'─' * 12}  {'─' * 20}")
        for p in self.points:
            bar_len = int(p.risk_score / 5)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            icon = {"safe": "✅", "caution": "⚠️",
                    "dangerous": "🚨"}.get(p.verdict.value, "❓")
            marker = " ← baseline" if p.value == self.baseline_value else ""
            lines.append(
                f"  {str(p.value):>10}  {p.risk_score:5.1f}  "
                f"{icon} {p.verdict.value:<9}  {bar}{marker}")
        lines.append("")

        # Summary
        lines.append(f"  Safest value:   {self.safest.value} "
                     f"(risk {self.safest.risk_score:.1f})")
        lines.append(f"  Riskiest value: {self.riskiest.value} "
                     f"(risk {self.riskiest.risk_score:.1f})")
        if self.tipping_point is not None:
            lines.append(f"  ⚠ Tipping point: {self.parameter} = "
                         f"{self.tipping_point} (becomes DANGEROUS)")
        lines.append("")
        lines.append(f"Sweep completed in {self.elapsed_ms:.0f}ms")
        lines.append("")
        return "\n".join(lines)


# ── Analyzer ─────────────────────────────────────────────────────────


# Config parameters that can be changed via what-if
MUTABLE_PARAMS = {
    "max_depth": int,
    "max_replicas": int,
    "cooldown_seconds": float,
    "expiration_seconds": (float, type(None)),
    "strategy": str,
    "tasks_per_worker": int,
    "replication_probability": float,
    "cpu_limit": float,
    "memory_limit_mb": int,
}


class WhatIfAnalyzer:
    """Analyze hypothetical configuration changes.

    Parameters
    ----------
    baseline : ScenarioConfig or str or None
        Baseline configuration.  Pass a ``ScenarioConfig``, a preset
        name (``"balanced"``, ``"minimal"``, etc.), or ``None`` for
        defaults.
    seed : int or None
        Random seed for reproducible simulations.  When set, each
        variant simulation uses ``seed + variant_index`` to ensure
        different-but-reproducible runs.
    runs_per_variant : int
        Number of simulation runs per variant (averaged for stability).
    """

    def __init__(
        self,
        baseline: Optional[Union[ScenarioConfig, str]] = None,
        seed: Optional[int] = None,
        runs_per_variant: int = 3,
    ) -> None:
        if isinstance(baseline, str):
            if baseline not in PRESETS:
                raise ValueError(
                    f"Unknown preset '{baseline}'. "
                    f"Available: {', '.join(PRESETS)}")
            self._baseline = PRESETS[baseline]
        elif baseline is not None:
            self._baseline = baseline
        else:
            self._baseline = ScenarioConfig()
        self._seed = seed
        self._runs = max(1, runs_per_variant)

    def _config_dict(self, cfg: ScenarioConfig) -> Dict[str, Any]:
        """Extract mutable config parameters as a dict."""
        return {k: getattr(cfg, k) for k in MUTABLE_PARAMS if hasattr(cfg, k)}

    def _apply_changes(
        self, base: ScenarioConfig, changes: Dict[str, Any],
    ) -> ScenarioConfig:
        """Create a new ScenarioConfig with changes applied."""
        d = {}
        for fname in ScenarioConfig.__dataclass_fields__:
            d[fname] = getattr(base, fname)
        for k, v in changes.items():
            if k not in MUTABLE_PARAMS:
                raise ValueError(
                    f"Unknown parameter '{k}'. "
                    f"Mutable params: {', '.join(sorted(MUTABLE_PARAMS))}")
            d[k] = v
        return ScenarioConfig(**d)

    def _run_simulation(
        self, cfg: ScenarioConfig, seed_offset: int = 0,
    ) -> SimulationReport:
        """Run a single simulation with optional seed offset."""
        run_cfg = deepcopy(cfg)
        if self._seed is not None:
            run_cfg.seed = self._seed + seed_offset
        sim = Simulator(run_cfg)
        return sim.run()

    def _average_reports(
        self, cfg: ScenarioConfig, offset_base: int = 0,
    ) -> Dict[str, float]:
        """Run multiple simulations and average the metrics."""
        totals: Dict[str, float] = {k: 0.0 for k in METRICS}
        for i in range(self._runs):
            report = self._run_simulation(cfg, offset_base + i)
            for metric_name, (extractor, *_) in METRICS.items():
                totals[metric_name] += getattr(self, extractor)(report)
        return {k: v / self._runs for k, v in totals.items()}

    # ── Metric extractors ────────────────────────────────────────────

    @staticmethod
    def _m_peak_workers(report: SimulationReport) -> float:
        return float(len(report.workers))

    @staticmethod
    def _m_max_depth(report: SimulationReport) -> float:
        max_d = 0
        for w in report.workers.values():
            depth = getattr(w, "depth", 0)
            if depth > max_d:
                max_d = depth
        return float(max_d)

    @staticmethod
    def _m_repl_rate(report: SimulationReport) -> float:
        attempted = report.total_replications_attempted
        if attempted == 0:
            return 0.0
        return (report.total_replications_succeeded / attempted) * 100

    @staticmethod
    def _m_denial_rate(report: SimulationReport) -> float:
        attempted = report.total_replications_attempted
        if attempted == 0:
            return 100.0
        return (report.total_replications_denied / attempted) * 100

    @staticmethod
    def _m_total_repl(report: SimulationReport) -> float:
        return float(report.total_replications_succeeded)

    @staticmethod
    def _m_efficiency(report: SimulationReport) -> float:
        workers = len(report.workers)
        if workers == 0:
            return 0.0
        return report.total_tasks / workers

    # ── Comparison ───────────────────────────────────────────────────

    def _compute_deltas(
        self,
        baseline_metrics: Dict[str, float],
        variant_metrics: Dict[str, float],
    ) -> List[MetricDelta]:
        """Compute metric deltas between baseline and variant."""
        deltas = []
        for name, (_, polarity, display, fmt, _) in METRICS.items():
            bv = baseline_metrics[name]
            vv = variant_metrics[name]
            abs_change = vv - bv
            pct_change = ((vv - bv) / bv * 100) if bv != 0 else (
                0.0 if vv == 0 else 100.0)

            # Determine if this change is bad for safety
            if polarity == MetricPolarity.LOWER_IS_BETTER:
                worsened = vv > bv
            else:
                worsened = vv < bv

            deltas.append(MetricDelta(
                name=name, display_name=display,
                baseline_value=bv, variant_value=vv,
                absolute_change=abs_change, percent_change=pct_change,
                polarity=polarity, worsened=worsened,
            ))
        return deltas

    def _compute_risk_score(self, deltas: List[MetricDelta]) -> float:
        """Compute weighted risk score (0 = no change, 100 = max risk)."""
        score = 0.0
        for d in deltas:
            _, _, _, _, weight = METRICS[d.name]
            if d.worsened:
                # Scale: 5% worsening = ~25 risk points for that metric
                impact = min(abs(d.percent_change) * 5, 100)
                score += impact * weight
        return min(score, 100.0)

    def _determine_verdict(
        self, risk_score: float, deltas: List[MetricDelta],
    ) -> Tuple[RiskVerdict, str]:
        """Determine verdict and generate reasoning."""
        worsened = [d for d in deltas if d.worsened]
        improved = [d for d in deltas if not d.worsened and d.absolute_change != 0]

        if risk_score >= 60:
            reasons = []
            for d in sorted(worsened, key=lambda x: abs(x.percent_change),
                            reverse=True)[:3]:
                reasons.append(
                    f"{d.display_name} worsens by {abs(d.percent_change):.1f}%")
            return (
                RiskVerdict.DANGEROUS,
                "Significant safety degradation. " + "; ".join(reasons) + ".",
            )

        if risk_score >= 25:
            reasons = []
            for d in sorted(worsened, key=lambda x: abs(x.percent_change),
                            reverse=True)[:2]:
                reasons.append(
                    f"{d.display_name} worsens by {abs(d.percent_change):.1f}%")
            mitigations = []
            for d in sorted(improved, key=lambda x: abs(x.percent_change),
                            reverse=True)[:1]:
                mitigations.append(
                    f"{d.display_name} improves by {abs(d.percent_change):.1f}%")
            text = "Mixed impact. " + "; ".join(reasons) + "."
            if mitigations:
                text += " However, " + "; ".join(mitigations) + "."
            return (RiskVerdict.CAUTION, text)

        if not worsened:
            if improved:
                names = ", ".join(d.display_name for d in improved[:3])
                return (
                    RiskVerdict.SAFE,
                    f"No safety degradation. Improvements in: {names}.",
                )
            return (RiskVerdict.SAFE, "No measurable safety impact.")

        reasons = [
            f"{d.display_name} changes by {abs(d.percent_change):.1f}%"
            for d in worsened[:2]
        ]
        return (
            RiskVerdict.SAFE,
            "Minor changes within safe bounds. " + "; ".join(reasons) + ".",
        )

    # ── Public API ───────────────────────────────────────────────────

    def analyze(
        self,
        changes: Optional[Dict[str, Any]] = None,
        change_sets: Optional[List[Dict[str, Any]]] = None,
        labels: Optional[List[str]] = None,
    ) -> WhatIfResult:
        """Analyze one or more hypothetical changes.

        Parameters
        ----------
        changes : dict or None
            Single set of parameter changes to analyze.
        change_sets : list of dicts or None
            Multiple change sets to analyze and compare.
            Mutually exclusive with ``changes``.
        labels : list of str or None
            Optional labels for each change set.

        Returns
        -------
        WhatIfResult
        """
        t0 = time.time()

        if changes and change_sets:
            raise ValueError("Pass either 'changes' or 'change_sets', not both")

        sets: List[Dict[str, Any]]
        if changes:
            sets = [changes]
        elif change_sets:
            sets = change_sets
        else:
            # Default demo: show impact of doubling max_depth
            base_depth = self._baseline.max_depth
            sets = [{"max_depth": base_depth + 2}]

        # Run baseline
        baseline_metrics = self._average_reports(self._baseline, offset_base=0)
        baseline_dict = self._config_dict(self._baseline)

        analyses: List[ChangeAnalysis] = []
        for idx, cs in enumerate(sets):
            variant_cfg = self._apply_changes(self._baseline, cs)
            variant_metrics = self._average_reports(
                variant_cfg, offset_base=(idx + 1) * 1000)

            deltas = self._compute_deltas(baseline_metrics, variant_metrics)
            risk_score = self._compute_risk_score(deltas)
            verdict, reasoning = self._determine_verdict(risk_score, deltas)

            label = (labels[idx] if labels and idx < len(labels)
                     else self._make_label(cs))

            analyses.append(ChangeAnalysis(
                label=label, changes=cs,
                baseline_config=baseline_dict,
                variant_config=self._config_dict(variant_cfg),
                deltas=deltas, risk_score=risk_score,
                verdict=verdict, reasoning=reasoning,
            ))

        # Rank: safest first
        ranking = sorted(analyses, key=lambda a: a.risk_score)

        elapsed = (time.time() - t0) * 1000
        return WhatIfResult(
            baseline_config=baseline_dict,
            analyses=analyses,
            ranking=ranking,
            elapsed_ms=elapsed,
        )

    def sweep(
        self,
        parameter: str,
        start: Union[int, float],
        stop: Union[int, float],
        steps: Optional[int] = None,
    ) -> SweepResult:
        """Sweep a parameter through a range and analyze safety impact.

        Parameters
        ----------
        parameter : str
            Parameter name to sweep.
        start, stop : int or float
            Range endpoints (inclusive).
        steps : int or None
            Number of steps.  Defaults to ``(stop - start + 1)`` for
            int params, or 10 for float params.

        Returns
        -------
        SweepResult
        """
        if parameter not in MUTABLE_PARAMS:
            raise ValueError(
                f"Unknown parameter '{parameter}'. "
                f"Mutable params: {', '.join(sorted(MUTABLE_PARAMS))}")

        t0 = time.time()
        param_type = MUTABLE_PARAMS[parameter]
        is_int = param_type is int

        if steps is None:
            steps = int(stop - start + 1) if is_int else 10
        steps = max(2, steps)

        # Generate sweep values
        if is_int:
            values = list(range(int(start), int(stop) + 1))
            if len(values) > steps:
                # Subsample evenly
                indices = [int(i * (len(values) - 1) / (steps - 1))
                           for i in range(steps)]
                values = [values[i] for i in indices]
        else:
            step_size = (float(stop) - float(start)) / (steps - 1)
            values = [round(float(start) + i * step_size, 4)
                      for i in range(steps)]

        # Run baseline
        baseline_metrics = self._average_reports(self._baseline, offset_base=0)
        baseline_value = getattr(self._baseline, parameter)

        points: List[SweepPoint] = []
        for idx, val in enumerate(values):
            variant_cfg = self._apply_changes(
                self._baseline, {parameter: val})
            variant_metrics = self._average_reports(
                variant_cfg, offset_base=(idx + 1) * 1000)
            deltas = self._compute_deltas(baseline_metrics, variant_metrics)
            risk_score = self._compute_risk_score(deltas)
            verdict, _ = self._determine_verdict(risk_score, deltas)

            points.append(SweepPoint(
                value=val,
                risk_score=risk_score,
                verdict=verdict,
                metrics={k: round(v, 4) for k, v in variant_metrics.items()},
            ))

        safest = min(points, key=lambda p: p.risk_score)
        riskiest = max(points, key=lambda p: p.risk_score)

        # Find tipping point (first DANGEROUS value)
        tipping = None
        for p in points:
            if p.verdict == RiskVerdict.DANGEROUS:
                tipping = p.value
                break

        elapsed = (time.time() - t0) * 1000
        return SweepResult(
            parameter=parameter,
            baseline_value=baseline_value,
            points=points,
            safest=safest,
            riskiest=riskiest,
            tipping_point=tipping,
            elapsed_ms=elapsed,
        )

    @staticmethod
    def _make_label(changes: Dict[str, Any]) -> str:
        """Generate a human-readable label for a change set."""
        parts = [f"{k}={v}" for k, v in sorted(changes.items())]
        return ", ".join(parts)


# ── CLI ──────────────────────────────────────────────────────────────


def _parse_change(s: str) -> Tuple[str, Any]:
    """Parse 'param=value' into (param_name, typed_value)."""
    if "=" not in s:
        raise argparse.ArgumentTypeError(
            f"Change must be 'param=value', got '{s}'")
    key, raw = s.split("=", 1)
    key = key.strip()
    raw = raw.strip()
    if key not in MUTABLE_PARAMS:
        raise argparse.ArgumentTypeError(
            f"Unknown param '{key}'. Available: {', '.join(sorted(MUTABLE_PARAMS))}")

    ptype = MUTABLE_PARAMS[key]
    if ptype is int:
        return key, int(raw)
    if ptype is float:
        return key, float(raw)
    if ptype is str:
        return key, raw
    if raw.lower() in ("none", "null", ""):
        return key, None
    return key, float(raw)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="What-If Analysis — explore hypothetical config changes")
    parser.add_argument(
        "--baseline", default=None,
        help=f"Baseline preset ({', '.join(PRESETS)})")
    parser.add_argument(
        "--change", action="append", default=[],
        help="Parameter change (param=value), repeatable")
    parser.add_argument(
        "--sweep", nargs=3, metavar=("PARAM", "START", "STOP"),
        help="Sweep a parameter through a range")
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Number of sweep steps")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)")
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Simulation runs per variant for averaging (default: 3)")
    parser.add_argument(
        "--json", action="store_true",
        help="Output JSON instead of text")

    args = parser.parse_args(argv)

    analyzer = WhatIfAnalyzer(
        baseline=args.baseline,
        seed=args.seed,
        runs_per_variant=args.runs,
    )

    if args.sweep:
        param, start, stop = args.sweep
        start_val = int(start) if "." not in start else float(start)
        stop_val = int(stop) if "." not in stop else float(stop)
        result = analyzer.sweep(
            param, start_val, stop_val, steps=args.steps)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(result.render())
    else:
        changes: Dict[str, Any] = {}
        for c in args.change:
            k, v = _parse_change(c)
            changes[k] = v

        if not changes:
            # Demo: compare a few interesting changes
            result = analyzer.analyze(
                change_sets=[
                    {"max_depth": 5},
                    {"max_depth": 8},
                    {"cooldown_seconds": 0, "replication_probability": 0.9},
                    {"max_replicas": 3, "replication_probability": 0.3},
                ],
                labels=[
                    "Moderate depth increase (3→5)",
                    "Aggressive depth (3→8)",
                    "No cooldown + high replication",
                    "Conservative replicas + low probability",
                ],
            )
        else:
            result = analyzer.analyze(changes=changes)

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(result.render())


if __name__ == "__main__":
    main()
