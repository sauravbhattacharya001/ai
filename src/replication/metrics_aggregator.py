"""Safety Metrics Aggregator — consolidated terminal dashboard.

Scans available safety modules (scorecard, drift, compliance, maturity,
sla, trend, fatigue, blast-radius) and renders a single summary table
so operators can get a quick health pulse without running each tool.

Usage::

    python -m replication metrics
    python -m replication metrics --json
    python -m replication metrics --modules scorecard,compliance

Exit code (table or JSON):

* ``0`` — all probed modules are ``ok``, ``warn`` or ``skip`` (no real errors).
* ``1`` — at least one probe reported ``error`` (real bug surfaced).

Error semantics (issue #89):

Probes used to wrap their entire body in a bare ``except Exception`` and
unconditionally map the result to ``status="skip"``.  That conflated three
very different conditions (optional dep missing, module produced no data,
module has a real bug) into a single "benign" looking row — and the CLI
always exited 0, so monitoring loops never alerted on regressions.

The current contract is:

* ``ImportError`` / ``ModuleNotFoundError`` → ``status="skip"`` — the
  module is legitimately unavailable in this environment.
* Any other exception → ``status="error"`` with
  ``detail=f"{ExceptionType}: {message}"`` — a real bug, surfaced as
  such and reflected in a non-zero CLI exit code.

The detail column in the rendered table is wide enough to show the
full ``ExceptionType: message`` for typical regressions instead of
truncating it down to ~36 chars.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional


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
    will be discovered automatically by :func:`aggregate`.  Probes
    should **not** wrap their body in a bare ``except Exception`` —
    error classification (skip vs. error) is handled centrally by
    :func:`aggregate` so that real bugs surface as ``status="error"``
    instead of being silently demoted to ``status="skip"``.
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


def _run_probe(name: str, fn: Callable[[], ModuleMetric]) -> ModuleMetric:
    """Invoke a probe with the standard skip-vs-error classification.

    See module docstring for the contract. This is the single place
    that decides "this is a real bug" vs "this module is just not
    installed", so probes themselves stay focused on the metric
    they're computing.
    """
    try:
        return fn()
    except (ImportError, ModuleNotFoundError) as exc:
        return ModuleMetric(
            name, "skip", detail=f"{type(exc).__name__}: {exc}"
        )
    except Exception as exc:  # pylint: disable=broad-except
        return ModuleMetric(
            name, "error", detail=f"{type(exc).__name__}: {exc}"
        )


# ── probes ───────────────────────────────────────────────────────────


@probe("scorecard")
def _probe_scorecard() -> ModuleMetric:
    from . import scorecard as sc
    card = sc.evaluate()
    overall = card.get("overall_score", card.get("score"))
    if overall is not None:
        status = _status_from_score(overall, ok_threshold=70, warn_threshold=40)
        return ModuleMetric("scorecard", status, score=overall,
                            detail=f"Overall safety score: {overall}")
    return ModuleMetric("scorecard", "ok", detail="Scorecard evaluated (no numeric score)")


@probe("compliance")
def _probe_compliance() -> ModuleMetric:
    from . import compliance as comp
    result = comp.audit()
    passed = result.get("passed", 0)
    total = result.get("total", 0)
    pct = (passed / total * 100) if total else 0
    status = _status_from_score(pct, ok_threshold=80, warn_threshold=50)
    return ModuleMetric("compliance", status, score=round(pct, 1),
                        detail=f"{passed}/{total} checks passed ({pct:.0f}%)",
                        extra={"passed": passed, "total": total})


@probe("drift")
def _probe_drift() -> ModuleMetric:
    from . import drift as dr
    result = dr.detect()
    drifted = result.get("drifted", False)
    magnitude = result.get("magnitude", 0)
    status = "warn" if drifted else "ok"
    return ModuleMetric("drift", status, score=round(magnitude, 3),
                        detail="Drift detected" if drifted else "No drift",
                        extra={"drifted": drifted})


@probe("maturity")
def _probe_maturity() -> ModuleMetric:
    from . import maturity_model as mm
    result = mm.assess()
    level = result.get("level", result.get("overall_level"))
    if level is not None:
        status = _status_from_score(level, ok_threshold=3, warn_threshold=2)
        return ModuleMetric("maturity", status, score=level,
                            detail=f"Maturity level {level}/5")
    return ModuleMetric("maturity", "ok", detail="Assessment completed")


@probe("sla")
def _probe_sla() -> ModuleMetric:
    from . import sla_monitor as sla
    result = sla.check()
    violations = result.get("violations", [])
    count = len(violations)
    status = _status_from_score(count, ok_threshold=0, warn_threshold=2, invert=True)
    return ModuleMetric("sla", status, score=count,
                        detail=f"{count} SLA violation(s)",
                        extra={"violations": count})


@probe("fatigue")
def _probe_fatigue() -> ModuleMetric:
    from . import fatigue_detector as fd
    result = fd.detect()
    fatigued = result.get("fatigued", False)
    ratio = result.get("suppression_ratio", 0)
    status = "warn" if fatigued else "ok"
    return ModuleMetric("fatigue", status, score=round(ratio, 2),
                        detail="Alert fatigue detected" if fatigued else "Alert load healthy",
                        extra={"fatigued": fatigued})


@probe("blast-radius")
def _probe_blast_radius() -> ModuleMetric:
    from . import blast_radius as br
    result = br.analyze()
    radius = result.get("max_radius", result.get("blast_radius", 0))
    status = _status_from_score(radius, ok_threshold=3, warn_threshold=6, invert=True)
    return ModuleMetric("blast-radius", status, score=radius,
                        detail=f"Max blast radius: {radius} hops")


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

# Detail column width. The previous 36-char width truncated typical
# Python exception messages mid-word (e.g. "'NoneType' object has no
# attribu..." instead of "...has no attribute 'get'"), which was the
# main reason real bugs looked like benign skips. 72 chars is wide
# enough for the typical "TypeError: 'NoneType' object has no
# attribute 'get'" without wrapping on a 100-col terminal.
_DETAIL_WIDTH = 72


def _render_table(metrics: List[ModuleMetric]) -> str:
    detail_bar = "═" * (_DETAIL_WIDTH + 2)
    detail_hdr = f"{'Detail':<{_DETAIL_WIDTH}}"
    lines: List[str] = []
    lines.append("")
    lines.append(f"╔══════════════╦════════╦════════════╦{detail_bar}╗")
    lines.append(f"║ Module       ║ Status ║ Score      ║ {detail_hdr} ║")
    lines.append(f"╠══════════════╬════════╬════════════╬{detail_bar}╣")
    for m in metrics:
        icon = STATUS_ICON.get(m.status, "?")
        score_str = str(m.score) if m.score is not None else "—"
        # Truncate only if absolutely needed; keep enough room to
        # actually read the exception type + message for `error` rows.
        detail = m.detail
        if len(detail) > _DETAIL_WIDTH:
            detail = detail[: _DETAIL_WIDTH - 1] + "…"
        lines.append(
            f"║ {m.module:<12} ║ {icon:<5}  ║ {score_str:<10} ║ {detail:<{_DETAIL_WIDTH}} ║"
        )
    lines.append(f"╚══════════════╩════════╩════════════╩{detail_bar}╝")

    # Overall health — skips are excluded so an all-skip run doesn't
    # masquerade as HEALTHY, but errors take precedence over warns.
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
    """Run probes and return metrics for the requested modules.

    Each probe is invoked through :func:`_run_probe` so that
    ``ImportError`` becomes ``status="skip"`` while every other
    exception becomes ``status="error"`` (with a useful
    ``ExceptionType: message`` detail). Unknown module names also
    map to ``status="skip"``.
    """
    targets = modules or list(_PROBE_REGISTRY.keys())
    results: List[ModuleMetric] = []
    for name in targets:
        probe_fn = _PROBE_REGISTRY.get(name)
        if probe_fn is None:
            results.append(ModuleMetric(name, "skip", detail=f"Unknown module: {name}"))
            continue
        results.append(_run_probe(name, probe_fn))
    return results


def _exit_code(metrics: List[ModuleMetric]) -> int:
    """Return ``1`` if any metric is in ``error`` state, else ``0``.

    ``warn`` and ``skip`` deliberately do not fail the run: optional
    dependencies (skip) and degraded-but-not-broken states (warn)
    happen during normal operation and would create alert fatigue.
    """
    return 1 if any(m.status == "error" for m in metrics) else 0


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns the process exit code."""
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

    return _exit_code(metrics)


if __name__ == "__main__":
    sys.exit(main())
