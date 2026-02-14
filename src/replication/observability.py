from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class Metric:
    name: str
    value: Any
    timestamp: datetime
    labels: Optional[Dict[str, str]] = None


class StructuredLogger:
    """In-memory structured logging with audit trail support."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.metrics: List[Metric] = []

    def log(self, event: str, **fields: Any) -> None:
        record = {"event": event, **fields, "timestamp": datetime.now(timezone.utc)}
        self.events.append(record)

    def emit_metric(self, name: str, value: Any, **labels: str) -> None:
        self.metrics.append(Metric(name=name, value=value, timestamp=datetime.now(timezone.utc), labels=labels or None))

    def audit(self, decision: str, **fields: Any) -> None:
        self.log("audit", decision=decision, **fields)
