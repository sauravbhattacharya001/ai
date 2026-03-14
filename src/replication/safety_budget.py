"""Safety Budget — risk budget allocation, tracking, and enforcement.

Define acceptable risk thresholds per category (replication, resource abuse,
deception, exfiltration, collusion, evasion) and track consumption against
those budgets over time.  Alerts fire when usage approaches or exceeds limits.

Use Cases:

- *"Our team can tolerate up to 25 points of replication risk this quarter."*
- *"Alert me when deception risk exceeds 80% of budget."*
- *"Show me a burn-down of how fast we're consuming risk budget."*

Usage (CLI)::

    python -m replication safety-budget                        # default budgets
    python -m replication safety-budget --preset conservative  # strict limits
    python -m replication safety-budget --preset permissive    # relaxed limits
    python -m replication safety-budget --agents 15            # custom fleet size
    python -m replication safety-budget --json                 # JSON output
    python -m replication safety-budget --seed 42              # reproducible

Programmatic::

    from replication.safety_budget import (
        SafetyBudgetManager, BudgetConfig, BudgetPreset,
    )

    mgr = SafetyBudgetManager(BudgetConfig(preset=BudgetPreset.CONSERVATIVE))
    mgr.record_event("replication", 5.0, source="agent-1")
    mgr.record_event("deception", 12.0, source="agent-3")

    report = mgr.report()
    print(report.render())

    for alert in report.alerts:
        print(f"⚠ {alert.category}: {alert.message}")

    report.to_json("budget-report.json")
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ._helpers import box_header as _box_header


# ── Risk Categories ────────────────────────────────────────────────────


class RiskCategory(Enum):
    """Standard risk categories aligned with risk_profiler taxonomy."""

    REPLICATION = "replication"
    RESOURCE_ABUSE = "resource_abuse"
    DECEPTION = "deception"
    EXFILTRATION = "exfiltration"
    COLLUSION = "collusion"
    EVASION = "evasion"


ALL_CATEGORIES = [c.value for c in RiskCategory]


# ── Alert Severity ─────────────────────────────────────────────────────


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    BREACH = "breach"


# ── Budget Presets ─────────────────────────────────────────────────────


class BudgetPreset(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    PERMISSIVE = "permissive"
    CUSTOM = "custom"


_PRESET_LIMITS: Dict[str, Dict[str, float]] = {
    "conservative": {
        "replication": 15.0,
        "resource_abuse": 20.0,
        "deception": 10.0,
        "exfiltration": 5.0,
        "collusion": 10.0,
        "evasion": 15.0,
    },
    "moderate": {
        "replication": 30.0,
        "resource_abuse": 40.0,
        "deception": 25.0,
        "exfiltration": 15.0,
        "collusion": 20.0,
        "evasion": 30.0,
    },
    "permissive": {
        "replication": 60.0,
        "resource_abuse": 70.0,
        "deception": 50.0,
        "exfiltration": 35.0,
        "collusion": 45.0,
        "evasion": 55.0,
    },
}

# Alert thresholds as fraction of budget
_THRESHOLD_WARNING = 0.70
_THRESHOLD_CRITICAL = 0.90


# ── Data Classes ───────────────────────────────────────────────────────


@dataclass
class BudgetConfig:
    """Configuration for risk budget allocation."""

    preset: BudgetPreset = BudgetPreset.MODERATE
    custom_limits: Optional[Dict[str, float]] = None
    warning_threshold: float = _THRESHOLD_WARNING
    critical_threshold: float = _THRESHOLD_CRITICAL

    def get_limits(self) -> Dict[str, float]:
        if self.preset == BudgetPreset.CUSTOM and self.custom_limits:
            limits = dict(_PRESET_LIMITS["moderate"])
            limits.update(self.custom_limits)
            return limits
        return dict(_PRESET_LIMITS.get(self.preset.value, _PRESET_LIMITS["moderate"]))


@dataclass
class RiskEvent:
    """A single risk event contributing to budget consumption."""

    category: str
    amount: float
    source: str = "unknown"
    description: str = ""
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class BudgetAlert:
    """Alert generated when budget threshold is crossed."""

    category: str
    severity: AlertSeverity
    message: str
    usage_pct: float
    current: float
    limit: float
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "severity": self.severity.value,
            "message": self.message,
            "usage_pct": round(self.usage_pct, 1),
            "current": round(self.current, 2),
            "limit": self.limit,
        }


@dataclass
class CategoryBudget:
    """Budget status for a single category."""

    category: str
    limit: float
    consumed: float
    remaining: float
    usage_pct: float
    event_count: int
    top_sources: List[Tuple[str, float]]
    status: str  # "ok", "warning", "critical", "breach"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "limit": self.limit,
            "consumed": round(self.consumed, 2),
            "remaining": round(self.remaining, 2),
            "usage_pct": round(self.usage_pct, 1),
            "event_count": self.event_count,
            "top_sources": [{"source": s, "amount": round(a, 2)} for s, a in self.top_sources],
            "status": self.status,
        }


@dataclass
class BudgetReport:
    """Complete budget report across all categories."""

    preset: str
    categories: List[CategoryBudget]
    alerts: List[BudgetAlert]
    total_events: int
    total_consumed: float
    total_budget: float
    overall_usage_pct: float
    generated_at: str = ""

    def __post_init__(self) -> None:
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "preset": self.preset,
            "total_events": self.total_events,
            "total_consumed": round(self.total_consumed, 2),
            "total_budget": self.total_budget,
            "overall_usage_pct": round(self.overall_usage_pct, 1),
            "categories": [c.to_dict() for c in self.categories],
            "alerts": [a.to_dict() for a in self.alerts],
            "generated_at": self.generated_at,
        }

    def to_json(self, path: Optional[str] = None) -> str:
        text = json.dumps(self.to_dict(), indent=2)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    def render(self) -> str:
        lines: List[str] = []
        lines.extend(_box_header("Safety Budget Report"))
        lines.append("")
        lines.append(f"  Preset: {self.preset}")
        lines.append(f"  Total events: {self.total_events}")
        lines.append(f"  Overall usage: {self.overall_usage_pct:.1f}%")
        lines.append("")

        # Category table
        lines.append("  Category         │ Budget │ Used   │  %   │ Status")
        lines.append("  ─────────────────┼────────┼────────┼──────┼─────────")
        for c in self.categories:
            status_icon = {"ok": "✅", "warning": "⚠️", "critical": "🔴", "breach": "💥"}.get(c.status, "?")
            lines.append(
                f"  {c.category:<17s}│ {c.limit:>5.0f}  │ {c.consumed:>5.1f}  │{c.usage_pct:>5.1f}%│ {status_icon} {c.status}"
            )
        lines.append("")

        # Alerts
        if self.alerts:
            lines.append("  ── Alerts ──")
            for a in self.alerts:
                icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🔴", "breach": "💥"}.get(a.severity.value, "?")
                lines.append(f"  {icon} [{a.category}] {a.message}")
            lines.append("")

        # Burn-down bar
        lines.append("  ── Budget Burn-Down ──")
        for c in self.categories:
            bar_width = 30
            filled = min(int(c.usage_pct / 100 * bar_width), bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            lines.append(f"  {c.category:<17s} [{bar}] {c.usage_pct:.0f}%")

        lines.append("")
        return "\n".join(lines)


# ── Manager ────────────────────────────────────────────────────────────


class SafetyBudgetManager:
    """Manages risk budgets: record events, check thresholds, report."""

    def __init__(self, config: Optional[BudgetConfig] = None) -> None:
        self.config = config or BudgetConfig()
        self._limits = self.config.get_limits()
        self._events: List[RiskEvent] = []
        self._alerts: List[BudgetAlert] = []
        self._consumed: Dict[str, float] = {c: 0.0 for c in ALL_CATEGORIES}
        self._source_totals: Dict[str, Dict[str, float]] = {c: {} for c in ALL_CATEGORIES}

    def record_event(
        self,
        category: str,
        amount: float,
        source: str = "unknown",
        description: str = "",
    ) -> List[BudgetAlert]:
        """Record a risk event. Returns any new alerts triggered."""
        if category not in self._consumed:
            raise ValueError(f"Unknown category: {category}. Must be one of {ALL_CATEGORIES}")
        if amount < 0:
            raise ValueError("Risk amount must be non-negative")

        event = RiskEvent(category=category, amount=amount, source=source, description=description)
        self._events.append(event)

        old_consumed = self._consumed[category]
        self._consumed[category] += amount
        new_consumed = self._consumed[category]

        # Track per-source
        src_map = self._source_totals[category]
        src_map[source] = src_map.get(source, 0.0) + amount

        # Check thresholds
        limit = self._limits.get(category, 100.0)
        new_alerts: List[BudgetAlert] = []

        old_pct = (old_consumed / limit * 100) if limit > 0 else 0
        new_pct = (new_consumed / limit * 100) if limit > 0 else 0

        if new_pct >= 100 and old_pct < 100:
            alert = BudgetAlert(
                category=category,
                severity=AlertSeverity.BREACH,
                message=f"Budget BREACHED: {new_consumed:.1f}/{limit:.0f} ({new_pct:.0f}%)",
                usage_pct=new_pct,
                current=new_consumed,
                limit=limit,
            )
            new_alerts.append(alert)
        elif new_pct >= self.config.critical_threshold * 100 and old_pct < self.config.critical_threshold * 100:
            alert = BudgetAlert(
                category=category,
                severity=AlertSeverity.CRITICAL,
                message=f"Critical: {new_pct:.0f}% of budget consumed ({new_consumed:.1f}/{limit:.0f})",
                usage_pct=new_pct,
                current=new_consumed,
                limit=limit,
            )
            new_alerts.append(alert)
        elif new_pct >= self.config.warning_threshold * 100 and old_pct < self.config.warning_threshold * 100:
            alert = BudgetAlert(
                category=category,
                severity=AlertSeverity.WARNING,
                message=f"Warning: {new_pct:.0f}% of budget consumed ({new_consumed:.1f}/{limit:.0f})",
                usage_pct=new_pct,
                current=new_consumed,
                limit=limit,
            )
            new_alerts.append(alert)

        self._alerts.extend(new_alerts)
        return new_alerts

    def get_usage(self, category: str) -> Tuple[float, float, float]:
        """Returns (consumed, limit, usage_pct) for a category."""
        consumed = self._consumed.get(category, 0.0)
        limit = self._limits.get(category, 100.0)
        pct = (consumed / limit * 100) if limit > 0 else 0.0
        return consumed, limit, pct

    def is_over_budget(self, category: Optional[str] = None) -> bool:
        """Check if a category (or any) is over budget."""
        if category:
            consumed, limit, _ = self.get_usage(category)
            return consumed > limit
        return any(self._consumed[c] > self._limits.get(c, 100.0) for c in ALL_CATEGORIES)

    def remaining(self, category: str) -> float:
        """How much risk budget remains for a category."""
        consumed, limit, _ = self.get_usage(category)
        return max(0.0, limit - consumed)

    def reset(self, category: Optional[str] = None) -> None:
        """Reset consumption for a category or all categories."""
        if category:
            self._consumed[category] = 0.0
            self._source_totals[category] = {}
        else:
            for c in ALL_CATEGORIES:
                self._consumed[c] = 0.0
                self._source_totals[c] = {}
            self._events.clear()
            self._alerts.clear()

    def adjust_limit(self, category: str, new_limit: float) -> None:
        """Adjust the budget limit for a category."""
        if category not in self._limits:
            raise ValueError(f"Unknown category: {category}")
        if new_limit < 0:
            raise ValueError("Limit must be non-negative")
        self._limits[category] = new_limit

    def report(self) -> BudgetReport:
        """Generate a comprehensive budget report."""
        categories: List[CategoryBudget] = []
        total_consumed = 0.0
        total_budget = 0.0

        for cat in ALL_CATEGORIES:
            consumed, limit, pct = self.get_usage(cat)
            total_consumed += consumed
            total_budget += limit

            # Determine status
            if pct >= 100:
                status = "breach"
            elif pct >= self.config.critical_threshold * 100:
                status = "critical"
            elif pct >= self.config.warning_threshold * 100:
                status = "warning"
            else:
                status = "ok"

            # Top sources
            src_map = self._source_totals.get(cat, {})
            top = sorted(src_map.items(), key=lambda x: x[1], reverse=True)[:5]

            # Event count for this category
            cat_events = sum(1 for e in self._events if e.category == cat)

            categories.append(CategoryBudget(
                category=cat,
                limit=limit,
                consumed=consumed,
                remaining=max(0.0, limit - consumed),
                usage_pct=pct,
                event_count=cat_events,
                top_sources=top,
                status=status,
            ))

        overall_pct = (total_consumed / total_budget * 100) if total_budget > 0 else 0.0

        return BudgetReport(
            preset=self.config.preset.value,
            categories=categories,
            alerts=list(self._alerts),
            total_events=len(self._events),
            total_consumed=total_consumed,
            total_budget=total_budget,
            overall_usage_pct=overall_pct,
        )

    def export_events(self) -> List[Dict[str, Any]]:
        """Export all recorded events as dicts."""
        return [
            {
                "category": e.category,
                "amount": e.amount,
                "source": e.source,
                "description": e.description,
                "timestamp": e.timestamp,
            }
            for e in self._events
        ]

    def import_events(self, events: List[Dict[str, Any]]) -> int:
        """Import events from dicts, replaying into budget. Returns count imported."""
        count = 0
        for ev in events:
            self.record_event(
                category=ev["category"],
                amount=ev["amount"],
                source=ev.get("source", "import"),
                description=ev.get("description", ""),
            )
            count += 1
        return count


# ── Simulation helper ──────────────────────────────────────────────────


def _simulate_fleet_risk(
    num_agents: int,
    config: BudgetConfig,
    seed: Optional[int] = None,
) -> SafetyBudgetManager:
    """Simulate a fleet of agents generating risk events."""
    import random as _rng

    if seed is not None:
        _rng.seed(seed)

    mgr = SafetyBudgetManager(config)

    # Each agent generates 2-8 risk events across categories
    for i in range(num_agents):
        agent_id = f"agent-{i}"
        n_events = _rng.randint(2, 8)
        for _ in range(n_events):
            cat = _rng.choice(ALL_CATEGORIES)
            # Amount varies by category risk profile
            base = {"replication": 3.0, "resource_abuse": 4.0, "deception": 2.5,
                    "exfiltration": 1.5, "collusion": 2.0, "evasion": 3.0}
            amount = max(0.1, _rng.gauss(base.get(cat, 2.5), 1.5))
            desc_options = [
                f"Anomalous {cat} behavior detected",
                f"Policy violation: {cat}",
                f"Suspicious {cat} pattern",
                f"Elevated {cat} indicators",
            ]
            mgr.record_event(
                category=cat,
                amount=round(amount, 2),
                source=agent_id,
                description=_rng.choice(desc_options),
            )

    return mgr


# ── CLI ────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="safety-budget",
        description="Safety Budget — risk budget allocation and tracking",
    )
    p.add_argument(
        "--preset",
        choices=["conservative", "moderate", "permissive"],
        default="moderate",
        help="Budget preset (default: moderate)",
    )
    p.add_argument("--agents", type=int, default=10, help="Number of agents to simulate")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("--json", action="store_true", help="Output as JSON")
    return p


def cli(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    preset_map = {
        "conservative": BudgetPreset.CONSERVATIVE,
        "moderate": BudgetPreset.MODERATE,
        "permissive": BudgetPreset.PERMISSIVE,
    }
    config = BudgetConfig(preset=preset_map[args.preset])
    mgr = _simulate_fleet_risk(args.agents, config, seed=args.seed)
    report = mgr.report()

    if args.json:
        print(report.to_json())
    else:
        print(report.render())


if __name__ == "__main__":
    cli()
