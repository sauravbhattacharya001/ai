"""Shared formatting and statistics helpers for replication modules.

This module consolidates small utility functions that were duplicated
across montecarlo, sensitivity, scorecard, and forensics.  Keeping
them here avoids drift and ensures consistent behavior.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, TYPE_CHECKING

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
