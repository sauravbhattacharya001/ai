from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional


# Default maximum entries before oldest events/metrics are evicted.
# Prevents unbounded memory growth in long-running simulations.
DEFAULT_MAX_ENTRIES = 100_000


@dataclass(slots=True)
class Metric:
    name: str
    value: Any
    timestamp: datetime
    labels: Optional[Dict[str, str]] = None


class StructuredLogger:
    """In-memory structured logging with audit trail support.

    Parameters
    ----------
    max_events : int or None
        Maximum event records to retain.  When exceeded, the oldest
        events are silently dropped.  Pass ``None`` for unbounded
        (not recommended for long-running simulations).
    max_metrics : int or None
        Maximum metric records to retain.  Same eviction policy.
    """

    def __init__(
        self,
        max_events: Optional[int] = DEFAULT_MAX_ENTRIES,
        max_metrics: Optional[int] = DEFAULT_MAX_ENTRIES,
    ) -> None:
        self.events: deque[Dict[str, Any]] = deque(maxlen=max_events)
        self.metrics: deque[Metric] = deque(maxlen=max_metrics)
        self.dropped_events: int = 0
        self.dropped_metrics: int = 0

    def log(self, event: str, **fields: Any) -> None:
        was_full = len(self.events) == self.events.maxlen if self.events.maxlen else False
        record = {"event": event, **fields, "timestamp": datetime.now(timezone.utc)}
        self.events.append(record)
        if was_full:
            self.dropped_events += 1

    def emit_metric(self, name: str, value: Any, **labels: str) -> None:
        was_full = len(self.metrics) == self.metrics.maxlen if self.metrics.maxlen else False
        self.metrics.append(Metric(name=name, value=value, timestamp=datetime.now(timezone.utc), labels=labels or None))
        if was_full:
            self.dropped_metrics += 1

    def audit(self, decision: str, **fields: Any) -> None:
        self.log("audit", decision=decision, **fields)

