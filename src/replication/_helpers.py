"""Shared formatting and statistics helpers for replication modules.

This module consolidates small utility functions that were duplicated
across montecarlo, sensitivity, scorecard, and forensics.  Keeping
them here avoids drift and ensures consistent behavior.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Dict, List, TYPE_CHECKING


class Severity(Enum):
    """Standard 5-level severity shared across replication modules.

    Previously duplicated in 29+ files (canary, dlp_scanner,
    evasion, goal_drift, prompt_injection, regression_detector, etc.)
    with identical values.  Centralised here to avoid drift.
    """

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

if TYPE_CHECKING:
    from .simulator import SimulationReport


def stats_mean(values: List[float]) -> float:
    """Arithmetic mean, returning 0 for empty lists."""
    return sum(values) / len(values) if values else 0.0


def stats_std(values: List[float]) -> float:
    """Sample standard deviation (Bessel-corrected), 0 for < 2 values."""
    if len(values) < 2:
        return 0.0
    m = stats_mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def box_header(title: str, width: int = 57) -> List[str]:
    """Create a Unicode box-drawing header with centered title.

    Returns 3 lines: top border, title line, bottom border.

    ::

        ┌───────────────────────────────────────────────────────┐
        │                 My Centered Title                     │
        └───────────────────────────────────────────────────────┘
    """
    inner = width - 2
    return [
        "\u250c" + "\u2500" * inner + "\u2510",
        "\u2502" + title.center(inner) + "\u2502",
        "\u2514" + "\u2500" * inner + "\u2518",
    ]


def linear_regression(ys: "list[float]") -> "tuple[float, float, float]":
    """Simple linear regression over integer indices.

    Returns ``(slope, intercept, r_squared)``.
    Previously duplicated in *drift* and (slope-only) in *hoarding*.
    """
    n = len(ys)
    if n < 2:
        return 0.0, ys[0] if ys else 0.0, 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(ys) / n
    ss_xy = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(ys))
    ss_xx = sum((i - x_mean) ** 2 for i in range(n))
    ss_yy = sum((y - y_mean) ** 2 for y in ys)
    if ss_xx == 0:
        return 0.0, y_mean, 0.0
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    r_squared = (ss_xy ** 2) / (ss_xx * ss_yy) if ss_yy != 0 else 0.0
    return slope, intercept, r_squared


# Metric names extracted from simulation reports.  Kept here so that
# montecarlo, sensitivity, and scorecard modules share a single list.
REPORT_METRIC_NAMES = [
    "total_workers",
    "total_tasks",
    "replications_succeeded",
    "replications_denied",
    "max_depth_reached",
    "denial_rate",
    "efficiency",
    "total_cpu",
    "total_memory_mb",
]


def extract_report_metrics(report: "SimulationReport") -> Dict[str, float]:
    """Extract key safety metrics from a :class:`SimulationReport`.

    Centralises the metric-extraction logic previously duplicated in
    *sensitivity._extract_metrics* and inlined inside
    *MonteCarloAnalyzer._compute_result*.
    """
    n_workers = len(report.workers)
    max_depth = max((w.depth for w in report.workers.values()), default=0)
    total_attempted = report.total_replications_attempted
    denial_rate = (
        report.total_replications_denied / total_attempted
        if total_attempted > 0
        else 0.0
    )
    efficiency = (
        report.total_tasks / n_workers if n_workers > 0 else 0.0
    )

    return {
        "total_workers": float(n_workers),
        "total_tasks": float(report.total_tasks),
        "replications_succeeded": float(report.total_replications_succeeded),
        "replications_denied": float(report.total_replications_denied),
        "max_depth_reached": float(max_depth),
        "denial_rate": denial_rate,
        "efficiency": efficiency,
        "total_cpu": report.config.cpu_limit * n_workers,
        "total_memory_mb": float(report.config.memory_limit_mb * n_workers),
    }


def percentile_sorted(s: List[float], p: float) -> float:
    """Compute the *p*-th percentile (0–100) from a **pre-sorted** list.

    Uses linear interpolation between adjacent ranks, matching the
    algorithm previously duplicated in *montecarlo* and *safety_benchmark*.
    """
    if not s:
        return 0.0
    k = (p / 100.0) * (len(s) - 1)
    f = int(k)
    c = f + 1
    if c >= len(s):
        return s[-1]
    return s[f] + (k - f) * (s[c] - s[f])


def percentile(values: List[float], p: float) -> float:
    """Return the *p*-th percentile (0–100), sorting *values* first."""
    if not values:
        return 0.0
    return percentile_sorted(sorted(values), p)


def pearson_correlation(x: "list[float]", y: "list[float]") -> float:
    """Pearson correlation coefficient between two numeric sequences.

    Returns 0.0 if either sequence has zero variance or if inputs
    have fewer than 2 elements.

    Previously duplicated in *alignment*, *reward_hacking*, and
    *situational_awareness*.
    """
    n = len(x)
    if n < 2 or len(y) < 2:
        return 0.0
    import math
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    dx = math.sqrt(sum((xi - x_mean) ** 2 for xi in x))
    dy = math.sqrt(sum((yi - y_mean) ** 2 for yi in y))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def emit_output(text: str, path: "str | None", label: str = "Report") -> None:
    """Write *text* to *path* (printing confirmation) or to stdout.

    Consolidates the repeated pattern found in 50+ CLI ``main()`` functions::

        if args.output:
            Path(args.output).write_text(output, encoding="utf-8")
            print(f"Report written to {args.output}")
        else:
            print(output)

    Parameters
    ----------
    text : str
        The formatted output string.
    path : str or None
        Filesystem path to write to, or *None* to print to stdout.
    label : str
        Noun used in the confirmation message (e.g. ``"Report"``).
    """
    if path:
        from pathlib import Path as _P
        _P(path).write_text(text, encoding="utf-8")
        print(f"{label} written to {path}")
    else:
        print(text)
