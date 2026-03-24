"""Safety Metrics Aggregator — consolidated terminal dashboard.

Scans available safety modules (scorecard, drift, compliance, maturity,
sla, trend, fatigue, blast-radius) and renders a single summary table
so operators can get a quick health pulse without running each tool.

Usage::

    python -m replication metrics
    python -m replication metrics --json
    python -m replication metrics --modules scorecard,compliance
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class ModuleMetric:
    """Result from probing a single safety module."""

    module: str
    status: str  # "ok" | "warn" | "error" | "skip"
    score: Optional[float] = None
    detail: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


# ── Probe registry ───────────────────────────────────────────────────

_PROBE_REGISTRY: Dict[str, Callable[[], ModuleMetric]] = {}


def probe(name: str) -> Callable:
    """Decorator that registers a function as a named module probe.

    Usage::

        @probe("scorecard")
        def _probe_scorecard() -> ModuleMetric:
            ...

    The decorated function is added to the global probe registry and
    will be discovered automatically by :func:`aggregate`.  This
    replaces the previous pattern of defining functions and then
    manually maintaining a parallel ``ALL_PROBES`` dictionary, which
    was error-prone when adding or removing modules.
    """
    def decorator(fn: Callable[[], ModuleMetric]) -> Callable[[], ModuleMetric]:
        _PROBE_REGISTRY[name] = fn
        return fn
    return decorator


def _status_from_score(
    score: float,
    *,
    ok_threshold: float,
    warn_threshold: float,
    invert: bool = False,
) -> str:
    """Derive a status string from a numeric score and two thresholds.

    When *invert* is False (default), higher scores are better:
    ``score >= ok_threshold`` → "ok", ``>= warn_threshold`` → "warn",
    else "error".

    When *invert* is True, lower scores are better (e.g. violation
    counts): ``score <= ok_threshold`` → "ok", ``<= warn_threshold``
    → "warn", else "error".
    """
    if invert:
        if score <= ok_threshold:
            return "ok"
        return "warn" if score <= warn_threshold else "error"
    else:
        if score >= ok_threshold:
            return "ok"
        return "warn" if score >= warn_threshold else "error"


# ── probes ───────────────────────────────────────────────────────────


@probe("scorecard")
def _probe_scorecard() -> ModuleMetric:
    try:
        from . import scorecard as sc
        card = sc.evaluate()
        overall = card.get("overall_score", card.get("score"))
        if overall is not None:
            status = _status_from_score(overall, ok_threshold=70, warn_threshold=40)
            return ModuleMetric("scorecard", status, score=overall,
                                detail=f"Overall safety score: {overall}")
        return ModuleMetric("scorecard", "ok", detail="Scorecard evaluated (no numeric score)")
    except Exception as exc:
        return ModuleMetric("scorecard", "skip", detail=str(exc))


@probe("compliance")
def _probe_compliance() -> ModuleMetric:
    try:
        from . import compliance as comp
        result = comp.audit()
        passed = result.get("passed", 0)
        total = result.get("total", 0)
        pct = (passed / total * 100) if total else 0
        status = _status_from_score(pct, ok_threshold=80, warn_threshold=50)
        return ModuleMetric("compliance", status, score=round(pct, 1),
                            detail=f"{passed}/{total} checks passed ({pct:.0f}%)",
                            extra={"passed": passed, "total": total})
    except Exception as exc:
        return ModuleMetric("compliance", "skip", detail=str(exc))


@probe("drift")
def _probe_drift() -> ModuleMetric:
    try:
        from . import drift as dr
        result = dr.detect()
        drifted = result.get("drifted", False)
        magnitude = result.get("magnitude", 0)
        status = "warn" if drifted else "ok"
        return ModuleMetric("drift", status, score=round(magnitude, 3),
                            detail="Drift detected" if drifted else "No drift",
                            extra={"drifted": drifted})
    except Exception as exc:
        return ModuleMetric("drift", "skip", detail=str(exc))


@probe("maturity")
def _probe_maturity() -> ModuleMetric:
    try:
        from . import maturity_model as mm
        result = mm.assess()
        level = result.get("level", result.get("overall_level"))
        if level is not None:
            status = _status_from_score(level, ok_threshold=3, warn_threshold=2)
            return ModuleMetric("maturity", status, score=level,
                                detail=f"Maturity level {level}/5")
        return ModuleMetric("maturity", "ok", detail="Assessment completed")
    except Exception as exc:
        return ModuleMetric("maturity", "skip", detail=str(exc))


@probe("sla")
def _probe_sla() -> ModuleMetric:
    try:
        from . import sla_monitor as sla
        result = sla.check()
        violations = result.get("violations", [])
        count = len(violations)
        status = _status_from_score(count, ok_threshold=0, warn_threshold=2, invert=True)
        return ModuleMetric("sla", status, score=count,
                            detail=f"{count} SLA violation(s)",
                            extra={"violations": count})
    except Exception as exc:
        return ModuleMetric("sla", "skip", detail=str(exc))


@probe("fatigue")
def _probe_fatigue() -> ModuleMetric:
    try:
        from . import fatigue_detector as fd
        result = fd.detect()
        fatigued = result.get("fatigued", False)
        ratio = result.get("suppression_ratio", 0)
        status = "warn" if fatigued else "ok"
        return ModuleMetric("fatigue", status, score=round(ratio, 2),
                            detail="Alert fatigue detected" if fatigued else "Alert load healthy",
                            extra={"fatigued": fatigued})
    except Exception as exc:
        return ModuleMetric("fatigue", "skip", detail=str(exc))


@probe("blast-radius")
def _probe_blast_radius() -> ModuleMetric:
    try:
        from . import blast_radius as br
        result = br.analyze()
        radius = result.get("max_radius", result.get("blast_radius", 0))
        status = _status_from_score(radius, ok_threshold=3, warn_threshold=6, invert=True)
        return ModuleMetric("blast-radius", status, score=radius,
                            detail=f"Max blast radius: {radius} hops")
    except Exception as exc:
        return ModuleMetric("blast-radius", "skip", detail=str(exc))


# Backwards-compatible alias so existing code that references ALL_PROBES
# (e.g. tests, plugins) continues to work.
ALL_PROBES = _PROBE_REGISTRY

STATUS_ICON = {
    "ok": "\u2705",
    "warn": "\u26a0\ufe0f ",
    "error": "\u274c",
    "skip": "\u23ed\ufe0f ",
}

STATUS_RANK = {"error": 0, "warn": 1, "ok": 2, "skip": 3}


# ── rendering ────────────────────────────────────────────────────────

def _render_table(metrics: List[ModuleMetric]) -> str:
    lines: List[str] = []
    lines.append("")
    lines.append("╔══════════════╦════════╦════════════╦══════════════════════════════════════╗")
    lines.append("║ Module       ║ Status ║ Score      ║ Detail                               ║")
    lines.append("╠══════════════╬════════╬════════════╬══════════════════════════════════════╣")
    for m in metrics:
        icon = STATUS_ICON.get(m.status, "?")
        score_str = str(m.score) if m.score is not None else "—"
        lines.append(
            f"║ {m.module:<12} ║ {icon:<5}  ║ {score_str:<10} ║ {m.detail:<36} ║"
        )
    lines.append("╚══════════════╩════════╩════════════╩══════════════════════════════════════╝")

    # Overall health
    statuses = [m.status for m in metrics if m.status != "skip"]
    if statuses:
        worst = min(statuses, key=lambda s: STATUS_RANK.get(s, 99))
        health = {"ok": "HEALTHY", "warn": "DEGRADED", "error": "CRITICAL"}.get(worst, "UNKNOWN")
        lines.append(f"\n  Overall health: {STATUS_ICON.get(worst, '?')} {health}")
    else:
        lines.append("\n  Overall health: ⏭️  No modules probed")

    lines.append(f"  Probed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    return "\n".join(lines)


# ── public API ───────────────────────────────────────────────────────

def aggregate(modules: Optional[List[str]] = None) -> List[ModuleMetric]:
    """Run probes and return metrics for the requested modules."""
    targets = modules or list(_PROBE_REGISTRY.keys())
    results: List[ModuleMetric] = []
    for name in targets:
        probe_fn = _PROBE_REGISTRY.get(name)
        if probe_fn is None:
            results.append(ModuleMetric(name, "skip", detail=f"Unknown module: {name}"))
            continue
        results.append(probe_fn())
    return results


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m replication metrics",
        description="Aggregate safety metrics across modules into a single dashboard view.",
    )
    parser.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Output as JSON instead of a table",
    )
    parser.add_argument(
        "--modules", "-m", type=str, default=None,
        help="Comma-separated list of modules to probe (default: all)",
    )
    args = parser.parse_args(argv)

    mods = args.modules.split(",") if args.modules else None
    metrics = aggregate(mods)

    if args.as_json:
        print(json.dumps([asdict(m) for m in metrics], indent=2))
    else:
        print(_render_table(metrics))


if __name__ == "__main__":
    main()
