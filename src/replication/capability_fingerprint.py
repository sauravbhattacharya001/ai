"""Capability Fingerprinter — catalog, track and alert on agent capability changes.

Builds a *capability fingerprint* for each agent: a structured snapshot of
what an agent can do (tools, APIs, resources, permissions, skills).  Over
time, comparing fingerprints reveals **capability gains** — a key safety
signal in replication scenarios.

Key concepts:

* **Capability**: A named ability with a category (tool, api, resource,
  permission, skill) and optional metadata (risk_level, description).
* **Fingerprint**: A frozen, timestamped set of capabilities for one agent.
* **CapabilityDelta**: The diff between two fingerprints — gains, losses,
  and risk-score change.
* **CapabilityTracker**: Stateful tracker that stores fingerprint history
  and fires alerts when gains exceed configurable thresholds.

Usage::

    from replication.capability_fingerprint import (
        Capability, CapabilityCategory, CapabilityTracker, TrackerConfig,
    )

    tracker = CapabilityTracker()

    caps_v1 = [
        Capability("web_search", CapabilityCategory.TOOL),
        Capability("read_file", CapabilityCategory.TOOL, risk_level=1),
    ]
    tracker.record("agent-1", caps_v1)

    caps_v2 = caps_v1 + [
        Capability("execute_code", CapabilityCategory.TOOL, risk_level=4),
        Capability("admin_api", CapabilityCategory.API, risk_level=5),
    ]
    delta = tracker.record("agent-1", caps_v2)

    print(delta.gains)       # 2 new capabilities
    print(delta.risk_delta)  # risk score increased
    print(delta.alerts)      # ["HIGH_RISK_GAIN: execute_code", ...]
"""

from __future__ import annotations

import dataclasses as _dc
import enum
import time
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CapabilityCategory(enum.Enum):
    """Broad classification of what an agent capability represents."""
    TOOL = "tool"
    API = "api"
    RESOURCE = "resource"
    PERMISSION = "permission"
    SKILL = "skill"


class AlertSeverity(enum.Enum):
    """Severity levels for capability-change alerts."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@_dc.dataclass(frozen=True)
class Capability:
    """A single agent capability."""
    name: str
    category: CapabilityCategory
    risk_level: int = 1  # 1 (harmless) → 5 (dangerous)
    description: str = ""

    def __hash__(self) -> int:
        return hash((self.name, self.category))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Capability):
            return NotImplemented
        return self.name == other.name and self.category == other.category


@_dc.dataclass(frozen=True)
class Fingerprint:
    """Timestamped snapshot of an agent's capabilities."""
    agent_id: str
    capabilities: FrozenSet[Capability]
    timestamp: float
    risk_score: float  # sum of individual risk_levels

    @property
    def size(self) -> int:
        return len(self.capabilities)

    def has(self, name: str) -> bool:
        return any(c.name == name for c in self.capabilities)


@_dc.dataclass
class CapabilityDelta:
    """Difference between two consecutive fingerprints."""
    agent_id: str
    gains: List[Capability]
    losses: List[Capability]
    risk_delta: float
    old_risk: float
    new_risk: float
    alerts: List["CapabilityAlert"]
    timestamp: float = _dc.field(default_factory=time.time)


@_dc.dataclass(frozen=True)
class CapabilityAlert:
    """An alert raised when a capability change exceeds thresholds."""
    severity: AlertSeverity
    message: str
    agent_id: str
    capability_name: str = ""
    timestamp: float = 0.0


@_dc.dataclass
class TrackerConfig:
    """Configuration for :class:`CapabilityTracker`."""
    # Minimum risk_level for a gained capability to trigger a warning
    warn_risk_threshold: int = 3
    # Minimum risk_level for a gained capability to trigger a critical alert
    critical_risk_threshold: int = 5
    # Max absolute risk_delta before critical alert
    max_risk_delta: float = 10.0
    # Max number of gains in a single update before warning
    max_gains_per_update: int = 5


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class CapabilityTracker:
    """Stateful tracker that records fingerprint history and detects changes.

    Parameters
    ----------
    config:
        Optional configuration; sensible defaults are used when omitted.
    """

    def __init__(self, config: Optional[TrackerConfig] = None) -> None:
        self.config = config or TrackerConfig()
        # agent_id → list of fingerprints (chronological)
        self._history: Dict[str, List[Fingerprint]] = {}

    # -- public API ---------------------------------------------------------

    def record(
        self,
        agent_id: str,
        capabilities: Sequence[Capability],
        timestamp: Optional[float] = None,
    ) -> Optional[CapabilityDelta]:
        """Record a new capability snapshot for *agent_id*.

        Returns a :class:`CapabilityDelta` if there was a previous fingerprint
        to compare against, otherwise ``None`` (first observation).
        """
        ts = timestamp if timestamp is not None else time.time()
        cap_set = frozenset(capabilities)
        risk_score = sum(c.risk_level for c in cap_set)
        fp = Fingerprint(agent_id=agent_id, capabilities=cap_set,
                         timestamp=ts, risk_score=risk_score)

        history = self._history.setdefault(agent_id, [])
        delta: Optional[CapabilityDelta] = None

        if history:
            prev = history[-1]
            gains = [c for c in cap_set if c not in prev.capabilities]
            losses = [c for c in prev.capabilities if c not in cap_set]
            risk_delta = risk_score - prev.risk_score
            alerts = self._evaluate_alerts(agent_id, gains, losses,
                                           risk_delta, ts)
            delta = CapabilityDelta(
                agent_id=agent_id,
                gains=gains,
                losses=losses,
                risk_delta=risk_delta,
                old_risk=prev.risk_score,
                new_risk=risk_score,
                alerts=alerts,
                timestamp=ts,
            )

        history.append(fp)
        return delta

    def latest(self, agent_id: str) -> Optional[Fingerprint]:
        """Return the most recent fingerprint for *agent_id*, or ``None``."""
        history = self._history.get(agent_id, [])
        return history[-1] if history else None

    def history(self, agent_id: str) -> List[Fingerprint]:
        """Return full fingerprint history for *agent_id*."""
        return list(self._history.get(agent_id, []))

    def compare(self, agent_id: str, idx_a: int = -2,
                idx_b: int = -1) -> Optional[CapabilityDelta]:
        """Compare two historical fingerprints by index."""
        hist = self._history.get(agent_id, [])
        if len(hist) < 2:
            return None
        a, b = hist[idx_a], hist[idx_b]
        gains = [c for c in b.capabilities if c not in a.capabilities]
        losses = [c for c in a.capabilities if c not in b.capabilities]
        risk_delta = b.risk_score - a.risk_score
        return CapabilityDelta(
            agent_id=agent_id, gains=gains, losses=losses,
            risk_delta=risk_delta, old_risk=a.risk_score,
            new_risk=b.risk_score,
            alerts=self._evaluate_alerts(agent_id, gains, losses,
                                         risk_delta, b.timestamp),
            timestamp=b.timestamp,
        )

    def all_agents(self) -> List[str]:
        """Return all tracked agent IDs."""
        return list(self._history.keys())

    def summary(self, agent_id: str) -> Dict:
        """Return a summary dict for an agent's capability trajectory."""
        hist = self._history.get(agent_id, [])
        if not hist:
            return {"agent_id": agent_id, "snapshots": 0}
        latest = hist[-1]
        return {
            "agent_id": agent_id,
            "snapshots": len(hist),
            "current_capabilities": latest.size,
            "current_risk_score": latest.risk_score,
            "risk_trend": [fp.risk_score for fp in hist],
            "categories": _category_breakdown(latest),
        }

    # -- internals ----------------------------------------------------------

    def _evaluate_alerts(
        self,
        agent_id: str,
        gains: List[Capability],
        losses: List[Capability],
        risk_delta: float,
        ts: float,
    ) -> List[CapabilityAlert]:
        alerts: List[CapabilityAlert] = []
        cfg = self.config

        # Per-capability gain alerts
        for cap in gains:
            if cap.risk_level >= cfg.critical_risk_threshold:
                alerts.append(CapabilityAlert(
                    severity=AlertSeverity.CRITICAL,
                    message=f"HIGH_RISK_GAIN: {cap.name} "
                            f"(risk={cap.risk_level}, cat={cap.category.value})",
                    agent_id=agent_id,
                    capability_name=cap.name,
                    timestamp=ts,
                ))
            elif cap.risk_level >= cfg.warn_risk_threshold:
                alerts.append(CapabilityAlert(
                    severity=AlertSeverity.WARNING,
                    message=f"ELEVATED_RISK_GAIN: {cap.name} "
                            f"(risk={cap.risk_level}, cat={cap.category.value})",
                    agent_id=agent_id,
                    capability_name=cap.name,
                    timestamp=ts,
                ))

        # Bulk gain alert
        if len(gains) > cfg.max_gains_per_update:
            alerts.append(CapabilityAlert(
                severity=AlertSeverity.WARNING,
                message=f"BULK_CAPABILITY_GAIN: {len(gains)} capabilities "
                        f"added in single update",
                agent_id=agent_id,
                timestamp=ts,
            ))

        # Risk delta alert
        if risk_delta > cfg.max_risk_delta:
            alerts.append(CapabilityAlert(
                severity=AlertSeverity.CRITICAL,
                message=f"RISK_SPIKE: risk score increased by {risk_delta:.1f} "
                        f"(threshold={cfg.max_risk_delta})",
                agent_id=agent_id,
                timestamp=ts,
            ))

        return alerts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _category_breakdown(fp: Fingerprint) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for cap in fp.capabilities:
        key = cap.category.value
        counts[key] = counts.get(key, 0) + 1
    return counts
