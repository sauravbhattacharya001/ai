"""Shared formatting and statistics helpers for replication modules.

This module consolidates small utility functions that were duplicated
across montecarlo, sensitivity, scorecard, and forensics.  Keeping
them here avoids drift and ensures consistent behavior.
"""

from __future__ import annotations

import math
from typing import List


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
