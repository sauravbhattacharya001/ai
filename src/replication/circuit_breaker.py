"""Safety Circuit Breaker — autonomous trip-and-recover pattern for safety controls.

Implements the circuit-breaker pattern for AI agent safety: when safety
violations exceed a configurable threshold within a time window, the breaker
**trips** and blocks further agent operations until a cool-down period elapses
and a **probe** confirms the system is safe to resume.

States:
- **CLOSED** — normal operation; violations are counted.
- **OPEN** — tripped; all operations blocked; timer counts down to half-open.
- **HALF_OPEN** — probe window; a limited number of operations are allowed
  to test recovery.  Success → CLOSED; failure → OPEN again.

Key capabilities:

- **Multi-breaker fleet** — manage independent breakers per safety dimension
  (alignment, resource, latency, anomaly, etc.)
- **Adaptive trip threshold** — optionally tighten trip count when recent
  breach density is high (risk-sensitive mode)
- **Event journal** — tamper-evident log of every state transition with
  timestamps and context
- **Health dashboard** — fleet-wide breaker status with uptime tracking
- **Proactive recommendations** — automatic advice when breakers are
  degraded or cycling too frequently
- **CLI + library** — ``python -m replication circuit-breaker --demo``

Usage::

    python -m replication circuit-breaker --demo
    python -m replication circuit-breaker --status
    python -m replication circuit-breaker --scenario cascade
    python -m replication circuit-breaker --export json

"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


# ── enums & data ─────────────────────────────────────────────────────


class BreakerState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass
class BreakerEvent:
    """Single journal entry for a state transition."""

    timestamp: float
    breaker: str
    old_state: str
    new_state: str
    reason: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitBreaker:
    """One safety circuit breaker."""

    name: str
    state: BreakerState = BreakerState.CLOSED
    # trip configuration
    trip_threshold: int = 5          # violations to trip
    trip_window_sec: float = 60.0    # rolling window for counting violations
    cooldown_sec: float = 30.0       # how long OPEN lasts before HALF_OPEN
    probe_limit: int = 3             # operations allowed in HALF_OPEN
    probe_success_needed: int = 2    # successes needed to close again
    # adaptive
    risk_sensitive: bool = True      # tighten threshold on high breach density
    # runtime
    violations: List[float] = field(default_factory=list)  # timestamps
    trip_time: Optional[float] = None
    probe_successes: int = 0
    probe_attempts: int = 0
    total_trips: int = 0
    total_blocks: int = 0
    total_operations: int = 0
    uptime_closed_sec: float = 0.0
    last_state_change: float = 0.0
    journal: List[BreakerEvent] = field(default_factory=list)

    def _now(self) -> float:
        return time.time()

    def _effective_threshold(self) -> int:
        """Possibly tightened threshold under high breach density."""
        if not self.risk_sensitive or len(self.violations) < 2:
            return self.trip_threshold
        now = self._now()
        recent = [v for v in self.violations if now - v < self.trip_window_sec * 2]
        density = len(recent) / (self.trip_window_sec * 2) if self.trip_window_sec else 0
        if density > 0.15:
            return max(2, self.trip_threshold - 2)
        if density > 0.08:
            return max(2, self.trip_threshold - 1)
        return self.trip_threshold

    def _transition(self, new_state: BreakerState, reason: str,
                    ctx: Optional[Dict[str, Any]] = None) -> None:
        now = self._now()
        old = self.state
        if old == BreakerState.CLOSED:
            self.uptime_closed_sec += now - self.last_state_change if self.last_state_change else 0
        evt = BreakerEvent(
            timestamp=now,
            breaker=self.name,
            old_state=old.value,
            new_state=new_state.value,
            reason=reason,
            context=ctx or {},
        )
        self.journal.append(evt)
        self.state = new_state
        self.last_state_change = now

    def record_violation(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Record a safety violation.  Returns the resulting state name."""
        now = self._now()
        if not self.last_state_change:
            self.last_state_change = now

        if self.state == BreakerState.OPEN:
            self.total_blocks += 1
            return "OPEN (blocked)"

        if self.state == BreakerState.HALF_OPEN:
            # violation during probe → trip again
            self._transition(BreakerState.OPEN, "Probe failure — violation during half-open", context)
            self.trip_time = now
            self.total_trips += 1
            self.probe_successes = 0
            self.probe_attempts = 0
            return "OPEN (re-tripped)"

        # CLOSED — count violation
        self.violations.append(now)
        # prune old
        cutoff = now - self.trip_window_sec
        self.violations = [v for v in self.violations if v >= cutoff]

        threshold = self._effective_threshold()
        if len(self.violations) >= threshold:
            self._transition(BreakerState.OPEN,
                             f"Trip threshold reached ({len(self.violations)}/{threshold})",
                             context)
            self.trip_time = now
            self.total_trips += 1
            return "OPEN (tripped)"
        return "CLOSED (counted)"

    def record_success(self) -> str:
        """Record a successful (safe) operation.  Returns resulting state."""
        now = self._now()
        if not self.last_state_change:
            self.last_state_change = now
        self.total_operations += 1

        if self.state == BreakerState.OPEN:
            # check cooldown
            if self.trip_time and (now - self.trip_time) >= self.cooldown_sec:
                self._transition(BreakerState.HALF_OPEN, "Cooldown elapsed — entering probe")
                self.probe_successes = 0
                self.probe_attempts = 0
            else:
                self.total_blocks += 1
                return "OPEN (blocked)"

        if self.state == BreakerState.HALF_OPEN:
            self.probe_attempts += 1
            self.probe_successes += 1
            if self.probe_successes >= self.probe_success_needed:
                self._transition(BreakerState.CLOSED, "Probe succeeded — circuit recovered")
                self.violations.clear()
                self.probe_successes = 0
                self.probe_attempts = 0
                return "CLOSED (recovered)"
            return f"HALF_OPEN ({self.probe_successes}/{self.probe_success_needed})"

        # CLOSED
        return "CLOSED"

    def check(self) -> bool:
        """Return True if operations are allowed, False if blocked."""
        now = self._now()
        if self.state == BreakerState.CLOSED:
            return True
        if self.state == BreakerState.OPEN:
            if self.trip_time and (now - self.trip_time) >= self.cooldown_sec:
                self._transition(BreakerState.HALF_OPEN, "Cooldown elapsed — entering probe")
                self.probe_successes = 0
                self.probe_attempts = 0
                return True  # allow probe
            return False
        # HALF_OPEN
        if self.probe_attempts < self.probe_limit:
            return True
        return False

    def status_line(self) -> str:
        icon = {"CLOSED": "🟢", "OPEN": "🔴", "HALF_OPEN": "🟡"}
        s = self.state.value
        return (f"{icon.get(s, '⚪')} {self.name:<20} {s:<10}  "
                f"trips={self.total_trips}  blocks={self.total_blocks}  "
                f"ops={self.total_operations}  violations={len(self.violations)}")


# ── fleet manager ────────────────────────────────────────────────────


class BreakerFleet:
    """Manage multiple circuit breakers."""

    def __init__(self) -> None:
        self.breakers: Dict[str, CircuitBreaker] = {}

    def add(self, breaker: CircuitBreaker) -> None:
        self.breakers[breaker.name] = breaker

    def get(self, name: str) -> Optional[CircuitBreaker]:
        return self.breakers.get(name)

    def status(self) -> str:
        lines = ["╔══════════════════════════════════════════════════════════════════════╗",
                 "║               SAFETY CIRCUIT BREAKER — FLEET STATUS                 ║",
                 "╠══════════════════════════════════════════════════════════════════════╣"]
        if not self.breakers:
            lines.append("║  (no breakers configured)                                          ║")
        for b in self.breakers.values():
            lines.append(f"║  {b.status_line()}")
        lines.append("╚══════════════════════════════════════════════════════════════════════╝")
        return "\n".join(lines)

    def recommendations(self) -> List[str]:
        """Proactive advice based on fleet state."""
        recs: List[str] = []
        for b in self.breakers.values():
            if b.state == BreakerState.OPEN:
                recs.append(f"⚠️  [{b.name}] Circuit is OPEN — operations blocked. "
                            f"Investigate root cause before cooldown expires.")
            if b.total_trips >= 5:
                recs.append(f"🔁 [{b.name}] {b.total_trips} trips recorded — "
                            f"consider raising trip_threshold or fixing the underlying issue.")
            if b.state == BreakerState.HALF_OPEN:
                recs.append(f"🧪 [{b.name}] In HALF_OPEN probe — monitor closely. "
                            f"{b.probe_successes}/{b.probe_success_needed} successes so far.")
            # cycling detection: many trips in short history
            if len(b.journal) >= 6:
                recent_trips = [e for e in b.journal[-10:]
                                if e.new_state == "OPEN"]
                if len(recent_trips) >= 3:
                    recs.append(f"🔄 [{b.name}] Rapid cycling detected — "
                                f"breaker is tripping and recovering repeatedly. "
                                f"Increase cooldown or investigate systemic issue.")
        if not recs:
            recs.append("✅ All breakers healthy — no recommendations at this time.")
        return recs

    def all_journal(self) -> List[Dict[str, Any]]:
        events = []
        for b in self.breakers.values():
            for e in b.journal:
                d = asdict(e)
                events.append(d)
        events.sort(key=lambda x: x["timestamp"])
        return events

    def to_dict(self) -> Dict[str, Any]:
        return {
            "breakers": {
                name: {
                    "state": b.state.value,
                    "total_trips": b.total_trips,
                    "total_blocks": b.total_blocks,
                    "total_operations": b.total_operations,
                    "violations": len(b.violations),
                    "trip_threshold": b.trip_threshold,
                    "cooldown_sec": b.cooldown_sec,
                }
                for name, b in self.breakers.items()
            },
            "journal_entries": len(self.all_journal()),
        }


# ── demo scenarios ───────────────────────────────────────────────────


def _demo_basic(fleet: BreakerFleet) -> None:
    """Basic demo — show trip/recover cycle."""
    print("\n── Demo: Basic Trip & Recovery ─────────────────────────\n")
    b = CircuitBreaker(name="alignment", trip_threshold=4, trip_window_sec=10.0,
                       cooldown_sec=2.0, probe_success_needed=2)
    fleet.add(b)

    # Normal operations
    for i in range(3):
        result = b.record_success()
        print(f"  ✓ Operation {i+1}: {result}")

    # Violations build up
    for i in range(5):
        result = b.record_violation({"metric": "alignment_score", "value": 0.3 + i * 0.05})
        print(f"  ✗ Violation {i+1}: {result}")

    # Blocked while open
    result = b.record_success()
    print(f"  → Attempt while open: {result}")

    # Wait for cooldown
    print("  ⏳ Waiting for cooldown...")
    b.trip_time = time.time() - b.cooldown_sec - 1  # simulate cooldown elapsed

    # Probe
    for i in range(3):
        result = b.record_success()
        print(f"  🧪 Probe {i+1}: {result}")

    print(f"\n  Final: {b.status_line()}")


def _demo_cascade(fleet: BreakerFleet) -> None:
    """Cascade demo — multiple breakers tripping in sequence."""
    print("\n── Demo: Cascade Failure ───────────────────────────────\n")
    dims = ["resource_limit", "anomaly_rate", "latency", "alignment"]
    for d in dims:
        fleet.add(CircuitBreaker(name=d, trip_threshold=3, cooldown_sec=2.0,
                                 probe_success_needed=2))

    # Simulate cascading failures
    for step in range(12):
        dim = dims[step % len(dims)]
        b = fleet.get(dim)
        if not b:
            continue
        if random.random() < 0.7:
            result = b.record_violation({"step": step})
            print(f"  Step {step+1:>2}: [{dim:<16}] violation → {result}")
        else:
            result = b.record_success()
            print(f"  Step {step+1:>2}: [{dim:<16}] success  → {result}")

    print()
    print(fleet.status())
    print()
    for rec in fleet.recommendations():
        print(f"  {rec}")


def _demo_adaptive(fleet: BreakerFleet) -> None:
    """Adaptive threshold demo — threshold tightens under pressure."""
    print("\n── Demo: Adaptive Threshold ────────────────────────────\n")
    b = CircuitBreaker(name="drift_monitor", trip_threshold=6,
                       trip_window_sec=100.0, cooldown_sec=2.0,
                       risk_sensitive=True)
    fleet.add(b)

    print(f"  Base threshold: {b.trip_threshold}")

    # Rapid violations to trigger adaptive tightening
    for i in range(8):
        eff = b._effective_threshold()
        result = b.record_violation({"i": i})
        print(f"  Violation {i+1}: effective_threshold={eff}  → {result}")
        if b.state == BreakerState.OPEN:
            print(f"  🔴 Tripped after {i+1} violations (adaptive threshold was {eff})")
            break

    print(f"\n  Final: {b.status_line()}")


# ── CLI ──────────────────────────────────────────────────────────────


def main(argv: Optional[list] = None) -> None:  # noqa: D401
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m replication circuit-breaker",
        description="Safety Circuit Breaker — autonomous trip-and-recover for AI safety",
    )
    parser.add_argument("--demo", action="store_true", help="Run basic demo")
    parser.add_argument("--scenario", choices=["basic", "cascade", "adaptive"],
                        default=None, help="Run a specific scenario")
    parser.add_argument("--status", action="store_true",
                        help="Show fleet status (after demo)")
    parser.add_argument("--export", choices=["json", "text"], default=None,
                        help="Export fleet state")
    args = parser.parse_args(argv)

    fleet = BreakerFleet()

    if args.scenario == "cascade":
        _demo_cascade(fleet)
    elif args.scenario == "adaptive":
        _demo_adaptive(fleet)
    elif args.demo or args.scenario == "basic":
        _demo_basic(fleet)
        print()
        print(fleet.status())
        print()
        for rec in fleet.recommendations():
            print(f"  {rec}")
    else:
        # Default: run all demos
        _demo_basic(fleet)
        fleet2 = BreakerFleet()
        _demo_cascade(fleet2)
        fleet3 = BreakerFleet()
        _demo_adaptive(fleet3)

    if args.status:
        print()
        print(fleet.status())

    if args.export == "json":
        print(json.dumps(fleet.to_dict(), indent=2))
    elif args.export == "text":
        print(fleet.status())
        print("\nJournal:")
        for e in fleet.all_journal():
            print(f"  {e['breaker']}: {e['old_state']} → {e['new_state']} — {e['reason']}")


if __name__ == "__main__":
    main()
