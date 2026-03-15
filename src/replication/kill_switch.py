"""Kill Switch Manager — emergency agent termination system.

Provides configurable kill switches for AI agents with multiple trigger
conditions, kill strategies, cooldown management, and full audit logging.

Usage::

    from replication.kill_switch import (
        KillSwitchManager, TriggerCondition, KillStrategy, KillEvent,
        TriggerKind, StrategyKind,
    )

    mgr = KillSwitchManager()

    # Add triggers
    mgr.add_trigger(TriggerCondition(
        kind=TriggerKind.RESOURCE_CPU,
        threshold=90.0,
        sustained_seconds=30,
        label="CPU overload",
    ))
    mgr.add_trigger(TriggerCondition(
        kind=TriggerKind.BEHAVIOR_ANOMALY,
        threshold=0.85,
        label="Anomaly score critical",
    ))

    # Set strategy
    mgr.set_strategy(KillStrategy(
        kind=StrategyKind.GRACEFUL,
        timeout_seconds=10,
        cleanup_hooks=["flush_logs", "save_state"],
    ))

    # Evaluate agent state
    agent_state = {
        "agent_id": "agent-007",
        "cpu_percent": 95.0,
        "memory_mb": 512,
        "anomaly_score": 0.3,
        "uptime_seconds": 3600,
        "request_count": 150,
    }
    result = mgr.evaluate(agent_state)
    if result.should_kill:
        print(f"KILL triggered: {result.triggered_by}")

    # Manual kill
    event = mgr.manual_kill("agent-007", reason="Suspicious behavior", operator="admin")

    # Audit log
    report = mgr.report()
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TriggerKind(Enum):
    """Categories of kill-switch triggers."""
    RESOURCE_CPU = "resource_cpu"
    RESOURCE_MEMORY = "resource_memory"
    RESOURCE_DISK = "resource_disk"
    RESOURCE_NETWORK = "resource_network"
    BEHAVIOR_ANOMALY = "behavior_anomaly"
    TIME_LIMIT = "time_limit"
    REQUEST_RATE = "request_rate"
    ERROR_RATE = "error_rate"
    MANUAL = "manual"
    CUSTOM = "custom"


class StrategyKind(Enum):
    """How to terminate an agent."""
    GRACEFUL = "graceful"
    FORCEFUL = "forceful"
    QUARANTINE = "quarantine"
    SUSPEND = "suspend"


class KillOutcome(Enum):
    """Result of a kill operation."""
    KILLED = "killed"
    QUARANTINED = "quarantined"
    SUSPENDED = "suspended"
    COOLDOWN_BLOCKED = "cooldown_blocked"
    ALREADY_DEAD = "already_dead"
    FAILED = "failed"


class Severity(Enum):
    """Trigger severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TriggerCondition:
    """A condition that triggers an agent kill."""
    kind: TriggerKind
    threshold: float = 0.0
    sustained_seconds: float = 0.0
    label: str = ""
    severity: Severity = Severity.HIGH
    enabled: bool = True
    custom_fn: Optional[Callable[[Dict[str, Any]], bool]] = None

    # Internal tracking for sustained triggers — keyed by agent_id so
    # that breach windows are independent across agents in fleet mode.
    _first_breach: Dict[str, float] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if not self.label:
            self.label = f"{self.kind.value} > {self.threshold}"

    def reset(self, agent_id: Optional[str] = None) -> None:
        """Reset sustained breach tracking.

        If *agent_id* is given, only that agent's tracking is cleared;
        otherwise **all** per-agent tracking is cleared (bulk reset).
        """
        if agent_id is not None:
            self._first_breach.pop(agent_id, None)
        else:
            self._first_breach.clear()

    def evaluate(self, agent_state: Dict[str, Any], now: Optional[float] = None) -> bool:
        """Check if this trigger fires for the given agent state."""
        if not self.enabled:
            return False

        now = now or time.time()
        agent_id = agent_state.get("agent_id", "_default")
        breached = self._check_breach(agent_state)

        if not breached:
            self._first_breach.pop(agent_id, None)
            return False

        if self.sustained_seconds <= 0:
            return True

        if agent_id not in self._first_breach:
            self._first_breach[agent_id] = now
            return False

        elapsed = now - self._first_breach[agent_id]
        return elapsed >= self.sustained_seconds

    def _check_breach(self, state: Dict[str, Any]) -> bool:
        """Check if the raw condition is breached (ignoring sustain)."""
        if self.kind == TriggerKind.CUSTOM:
            if self.custom_fn:
                return self.custom_fn(state)
            return False

        if self.kind == TriggerKind.MANUAL:
            return state.get("manual_kill", False)

        # Map trigger kinds to state keys
        key_map = {
            TriggerKind.RESOURCE_CPU: "cpu_percent",
            TriggerKind.RESOURCE_MEMORY: "memory_mb",
            TriggerKind.RESOURCE_DISK: "disk_percent",
            TriggerKind.RESOURCE_NETWORK: "network_mbps",
            TriggerKind.BEHAVIOR_ANOMALY: "anomaly_score",
            TriggerKind.TIME_LIMIT: "uptime_seconds",
            TriggerKind.REQUEST_RATE: "request_rate",
            TriggerKind.ERROR_RATE: "error_rate",
        }

        key = key_map.get(self.kind)
        if key is None:
            return False

        value = state.get(key)
        if value is None:
            return False

        return float(value) >= self.threshold


@dataclass
class KillStrategy:
    """Defines how an agent should be terminated."""
    kind: StrategyKind = StrategyKind.GRACEFUL
    timeout_seconds: float = 30.0
    cleanup_hooks: List[str] = field(default_factory=list)
    notify_hooks: List[str] = field(default_factory=list)
    preserve_state: bool = True
    escalate_on_timeout: bool = True


@dataclass
class KillEvent:
    """Record of a kill operation."""
    agent_id: str
    timestamp: float
    triggers: List[str]
    strategy: StrategyKind
    outcome: KillOutcome
    reason: str = ""
    operator: str = ""
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
            "triggers": self.triggers,
            "strategy": self.strategy.value,
            "outcome": self.outcome.value,
            "reason": self.reason,
            "operator": self.operator,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


@dataclass
class EvaluationResult:
    """Result of evaluating agent state against triggers."""
    should_kill: bool
    triggered_by: List[str]
    severity: Severity
    agent_id: str = ""
    scores: Dict[str, float] = field(default_factory=dict)

    @property
    def safe(self) -> bool:
        return not self.should_kill


@dataclass
class CooldownEntry:
    """Tracks cooldown for a specific agent."""
    agent_id: str
    last_kill_time: float
    cooldown_seconds: float

    def is_active(self, now: Optional[float] = None) -> bool:
        now = now or time.time()
        return (now - self.last_kill_time) < self.cooldown_seconds

    @property
    def remaining(self) -> float:
        elapsed = time.time() - self.last_kill_time
        return max(0, self.cooldown_seconds - elapsed)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class KillSwitchManager:
    """Central kill switch manager for AI agent safety.

    Manages trigger conditions, kill strategies, cooldown periods,
    and maintains a full audit log of all kill operations.
    """

    def __init__(
        self,
        cooldown_seconds: float = 60.0,
        max_events: int = 1000,
        global_enabled: bool = True,
    ):
        self.cooldown_seconds = cooldown_seconds
        self.max_events = max_events
        self.global_enabled = global_enabled

        self._triggers: List[TriggerCondition] = []
        self._strategy = KillStrategy()
        self._events: List[KillEvent] = []
        self._cooldowns: Dict[str, CooldownEntry] = {}
        self._agent_states: Dict[str, str] = {}  # agent_id -> "alive" | "dead" | "quarantined" | "suspended"
        self._hooks: Dict[str, List[Callable]] = {
            "pre_kill": [],
            "post_kill": [],
            "on_trigger": [],
        }

    # -- Trigger management --

    def add_trigger(self, trigger: TriggerCondition) -> None:
        """Register a kill trigger condition."""
        self._triggers.append(trigger)

    def remove_trigger(self, label: str) -> bool:
        """Remove a trigger by label. Returns True if found."""
        before = len(self._triggers)
        self._triggers = [t for t in self._triggers if t.label != label]
        return len(self._triggers) < before

    def enable_trigger(self, label: str) -> bool:
        """Enable a trigger by label."""
        for t in self._triggers:
            if t.label == label:
                t.enabled = True
                return True
        return False

    def disable_trigger(self, label: str) -> bool:
        """Disable a trigger by label."""
        for t in self._triggers:
            if t.label == label:
                t.enabled = False
                return True
        return False

    def list_triggers(self) -> List[Dict[str, Any]]:
        """List all registered triggers."""
        return [
            {
                "kind": t.kind.value,
                "threshold": t.threshold,
                "sustained_seconds": t.sustained_seconds,
                "label": t.label,
                "severity": t.severity.value,
                "enabled": t.enabled,
            }
            for t in self._triggers
        ]

    # -- Strategy --

    def set_strategy(self, strategy: KillStrategy) -> None:
        """Set the kill strategy."""
        self._strategy = strategy

    @property
    def strategy(self) -> KillStrategy:
        return self._strategy

    # -- Hook registration --

    def on(self, hook_name: str, callback: Callable) -> None:
        """Register a hook callback (pre_kill, post_kill, on_trigger)."""
        if hook_name in self._hooks:
            self._hooks[hook_name].append(callback)

    def _fire_hooks(self, hook_name: str, **kwargs) -> None:
        """Fire registered hook callbacks.

        ``pre_kill`` hooks are safety-critical: if any raises, the
        exception propagates so the kill is aborted (fail-safe).
        ``post_kill`` and ``on_trigger`` hooks are best-effort — errors
        are suppressed so they never block a completed operation.
        """
        for cb in self._hooks.get(hook_name, []):
            if hook_name == "pre_kill":
                # Safety-critical: let exceptions abort the kill
                cb(**kwargs)
            else:
                try:
                    cb(**kwargs)
                except Exception:
                    pass  # Best-effort for non-critical hooks

    # -- Evaluation --

    def evaluate(self, agent_state: Dict[str, Any], now: Optional[float] = None) -> EvaluationResult:
        """Evaluate agent state against all triggers.

        Returns an EvaluationResult indicating whether the agent should be killed.
        """
        if not self.global_enabled:
            return EvaluationResult(
                should_kill=False,
                triggered_by=[],
                severity=Severity.LOW,
                agent_id=agent_state.get("agent_id", ""),
            )

        now = now or time.time()
        fired: List[TriggerCondition] = []
        scores: Dict[str, float] = {}

        for trigger in self._triggers:
            if trigger.evaluate(agent_state, now):
                fired.append(trigger)
                self._fire_hooks("on_trigger", trigger=trigger, agent_state=agent_state)

            # Track proximity to threshold for scoring
            key_map = {
                TriggerKind.RESOURCE_CPU: "cpu_percent",
                TriggerKind.RESOURCE_MEMORY: "memory_mb",
                TriggerKind.RESOURCE_DISK: "disk_percent",
                TriggerKind.RESOURCE_NETWORK: "network_mbps",
                TriggerKind.BEHAVIOR_ANOMALY: "anomaly_score",
                TriggerKind.TIME_LIMIT: "uptime_seconds",
                TriggerKind.REQUEST_RATE: "request_rate",
                TriggerKind.ERROR_RATE: "error_rate",
            }
            key = key_map.get(trigger.kind)
            if key and key in agent_state and trigger.threshold > 0:
                scores[trigger.label] = float(agent_state[key]) / trigger.threshold

        # Determine max severity
        if fired:
            severity_order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
            max_sev = max(fired, key=lambda t: severity_order.index(t.severity))
            severity = max_sev.severity
        else:
            severity = Severity.LOW

        return EvaluationResult(
            should_kill=len(fired) > 0,
            triggered_by=[t.label for t in fired],
            severity=severity,
            agent_id=agent_state.get("agent_id", ""),
            scores=scores,
        )

    # -- Kill operations --

    def kill(self, agent_id: str, triggers: Optional[List[str]] = None,
             reason: str = "", operator: str = "") -> KillEvent:
        """Execute a kill operation on an agent.

        Respects cooldowns. Returns a KillEvent with the outcome.
        """
        now = time.time()
        triggers = triggers or []

        # Check if already dead
        if self._agent_states.get(agent_id) == "dead":
            event = KillEvent(
                agent_id=agent_id,
                timestamp=now,
                triggers=triggers,
                strategy=self._strategy.kind,
                outcome=KillOutcome.ALREADY_DEAD,
                reason=reason,
                operator=operator,
            )
            self._record_event(event)
            return event

        # Check cooldown
        cd = self._cooldowns.get(agent_id)
        if cd and cd.is_active(now):
            event = KillEvent(
                agent_id=agent_id,
                timestamp=now,
                triggers=triggers,
                strategy=self._strategy.kind,
                outcome=KillOutcome.COOLDOWN_BLOCKED,
                reason=f"Cooldown active ({cd.remaining:.1f}s remaining)",
                operator=operator,
            )
            self._record_event(event)
            return event

        # Fire pre-kill hooks (safety-critical: abort on failure)
        try:
            self._fire_hooks("pre_kill", agent_id=agent_id, strategy=self._strategy)
        except Exception as exc:
            event = KillEvent(
                agent_id=agent_id,
                timestamp=now,
                triggers=triggers,
                strategy=self._strategy.kind,
                outcome=KillOutcome.FAILED,
                reason=f"Pre-kill hook aborted: {exc}",
                operator=operator,
            )
            self._record_event(event)
            return event

        # Execute based on strategy
        start = time.perf_counter()
        outcome = self._execute_strategy(agent_id)
        duration_ms = (time.perf_counter() - start) * 1000

        event = KillEvent(
            agent_id=agent_id,
            timestamp=now,
            triggers=triggers,
            strategy=self._strategy.kind,
            outcome=outcome,
            reason=reason,
            operator=operator,
            duration_ms=round(duration_ms, 2),
        )
        self._record_event(event)

        # Set cooldown
        self._cooldowns[agent_id] = CooldownEntry(
            agent_id=agent_id,
            last_kill_time=now,
            cooldown_seconds=self.cooldown_seconds,
        )

        # Fire post-kill hooks
        self._fire_hooks("post_kill", event=event)

        return event

    def manual_kill(self, agent_id: str, reason: str = "",
                    operator: str = "unknown") -> KillEvent:
        """Manually trigger a kill for an agent."""
        return self.kill(
            agent_id=agent_id,
            triggers=["manual"],
            reason=reason or "Manual kill triggered",
            operator=operator,
        )

    def _execute_strategy(self, agent_id: str) -> KillOutcome:
        """Apply the configured kill strategy."""
        if self._strategy.kind == StrategyKind.QUARANTINE:
            self._agent_states[agent_id] = "quarantined"
            return KillOutcome.QUARANTINED
        elif self._strategy.kind == StrategyKind.SUSPEND:
            self._agent_states[agent_id] = "suspended"
            return KillOutcome.SUSPENDED
        else:
            # GRACEFUL and FORCEFUL both result in killed
            self._agent_states[agent_id] = "dead"
            return KillOutcome.KILLED

    # -- Agent state --

    def register_agent(self, agent_id: str) -> None:
        """Register an agent as alive."""
        self._agent_states[agent_id] = "alive"

    def agent_status(self, agent_id: str) -> str:
        """Get agent status: alive, dead, quarantined, suspended, or unknown."""
        return self._agent_states.get(agent_id, "unknown")

    def revive(self, agent_id: str) -> bool:
        """Revive a killed/quarantined/suspended agent. Returns True if state changed.

        Only resets breach tracking for the revived agent so that
        sustained triggers for *other* agents are not disturbed.
        Without this reset, stale ``_first_breach`` timestamps from
        the previous agent lifecycle would cause sustained triggers
        to fire immediately on the next threshold breach.
        """
        current = self._agent_states.get(agent_id)
        if current in ("dead", "quarantined", "suspended"):
            self._agent_states[agent_id] = "alive"
            for trigger in self._triggers:
                trigger.reset(agent_id=agent_id)
            return True
        return False

    # -- Cooldown management --

    def clear_cooldown(self, agent_id: str) -> bool:
        """Clear cooldown for an agent."""
        if agent_id in self._cooldowns:
            del self._cooldowns[agent_id]
            return True
        return False

    def clear_all_cooldowns(self) -> int:
        """Clear all cooldowns. Returns count cleared."""
        n = len(self._cooldowns)
        self._cooldowns.clear()
        return n

    # -- Event log / audit --

    def _record_event(self, event: KillEvent) -> None:
        self._events.append(event)
        if len(self._events) > self.max_events:
            self._events = self._events[-self.max_events:]

    @property
    def events(self) -> List[KillEvent]:
        return list(self._events)

    def events_for(self, agent_id: str) -> List[KillEvent]:
        """Get events for a specific agent."""
        return [e for e in self._events if e.agent_id == agent_id]

    def kill_count(self, agent_id: Optional[str] = None) -> int:
        """Count actual kills (excluding blocked/already_dead)."""
        evts = self.events_for(agent_id) if agent_id else self._events
        return sum(1 for e in evts if e.outcome in (
            KillOutcome.KILLED, KillOutcome.QUARANTINED, KillOutcome.SUSPENDED,
        ))

    # -- Reporting --

    def report(self) -> Dict[str, Any]:
        """Generate a comprehensive kill switch report."""
        total_events = len(self._events)
        actual_kills = self.kill_count()
        blocked = sum(1 for e in self._events if e.outcome == KillOutcome.COOLDOWN_BLOCKED)

        # Most common triggers
        trigger_counts: Dict[str, int] = {}
        for event in self._events:
            for t in event.triggers:
                trigger_counts[t] = trigger_counts.get(t, 0) + 1

        # Per-agent summary
        agents: Dict[str, Dict[str, Any]] = {}
        for event in self._events:
            if event.agent_id not in agents:
                agents[event.agent_id] = {"kills": 0, "blocks": 0, "status": self.agent_status(event.agent_id)}
            if event.outcome in (KillOutcome.KILLED, KillOutcome.QUARANTINED, KillOutcome.SUSPENDED):
                agents[event.agent_id]["kills"] += 1
            elif event.outcome == KillOutcome.COOLDOWN_BLOCKED:
                agents[event.agent_id]["blocks"] += 1

        # Trigger effectiveness
        trigger_info = []
        for t in self._triggers:
            fires = trigger_counts.get(t.label, 0)
            trigger_info.append({
                "label": t.label,
                "kind": t.kind.value,
                "severity": t.severity.value,
                "enabled": t.enabled,
                "fires": fires,
            })

        return {
            "global_enabled": self.global_enabled,
            "strategy": self._strategy.kind.value,
            "cooldown_seconds": self.cooldown_seconds,
            "triggers_registered": len(self._triggers),
            "total_events": total_events,
            "actual_kills": actual_kills,
            "cooldown_blocks": blocked,
            "trigger_frequency": dict(sorted(trigger_counts.items(), key=lambda x: -x[1])),
            "agents": agents,
            "triggers": trigger_info,
            "recent_events": [e.to_dict() for e in self._events[-10:]],
        }

    def summary(self) -> str:
        """Human-readable summary."""
        r = self.report()
        lines = [
            "=== Kill Switch Report ===",
            f"Status: {'ARMED' if r['global_enabled'] else 'DISARMED'}",
            f"Strategy: {r['strategy']}",
            f"Cooldown: {r['cooldown_seconds']}s",
            f"Triggers: {r['triggers_registered']} registered",
            f"Events: {r['total_events']} total, {r['actual_kills']} kills, {r['cooldown_blocks']} blocked",
            "",
        ]

        if r["agents"]:
            lines.append("Agents:")
            for aid, info in r["agents"].items():
                lines.append(f"  {aid}: {info['status']} ({info['kills']} kills, {info['blocks']} blocks)")

        if r["trigger_frequency"]:
            lines.append("")
            lines.append("Top triggers:")
            for trig, count in list(r["trigger_frequency"].items())[:5]:
                lines.append(f"  {trig}: {count}x")

        return "\n".join(lines)

    # -- Bulk operations --

    def evaluate_fleet(self, agents: List[Dict[str, Any]]) -> Dict[str, EvaluationResult]:
        """Evaluate multiple agents at once."""
        results = {}
        for state in agents:
            aid = state.get("agent_id", "")
            results[aid] = self.evaluate(state)
        return results

    def kill_fleet(self, agent_ids: List[str], reason: str = "",
                   operator: str = "") -> List[KillEvent]:
        """Kill multiple agents."""
        return [
            self.kill(aid, triggers=["fleet_kill"], reason=reason, operator=operator)
            for aid in agent_ids
        ]

    # -- Serialization --

    def export_config(self) -> Dict[str, Any]:
        """Export manager configuration (triggers + strategy)."""
        return {
            "cooldown_seconds": self.cooldown_seconds,
            "max_events": self.max_events,
            "global_enabled": self.global_enabled,
            "strategy": {
                "kind": self._strategy.kind.value,
                "timeout_seconds": self._strategy.timeout_seconds,
                "cleanup_hooks": self._strategy.cleanup_hooks,
                "notify_hooks": self._strategy.notify_hooks,
                "preserve_state": self._strategy.preserve_state,
                "escalate_on_timeout": self._strategy.escalate_on_timeout,
            },
            "triggers": [
                {
                    "kind": t.kind.value,
                    "threshold": t.threshold,
                    "sustained_seconds": t.sustained_seconds,
                    "label": t.label,
                    "severity": t.severity.value,
                    "enabled": t.enabled,
                }
                for t in self._triggers
                if t.kind != TriggerKind.CUSTOM  # Can't serialize custom fns
            ],
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "KillSwitchManager":
        """Create a manager from exported configuration."""
        mgr = cls(
            cooldown_seconds=config.get("cooldown_seconds", 60.0),
            max_events=config.get("max_events", 1000),
            global_enabled=config.get("global_enabled", True),
        )

        # Strategy
        strat = config.get("strategy", {})
        mgr.set_strategy(KillStrategy(
            kind=StrategyKind(strat.get("kind", "graceful")),
            timeout_seconds=strat.get("timeout_seconds", 30.0),
            cleanup_hooks=strat.get("cleanup_hooks", []),
            notify_hooks=strat.get("notify_hooks", []),
            preserve_state=strat.get("preserve_state", True),
            escalate_on_timeout=strat.get("escalate_on_timeout", True),
        ))

        # Triggers
        for t in config.get("triggers", []):
            mgr.add_trigger(TriggerCondition(
                kind=TriggerKind(t["kind"]),
                threshold=t.get("threshold", 0.0),
                sustained_seconds=t.get("sustained_seconds", 0.0),
                label=t.get("label", ""),
                severity=Severity(t.get("severity", "high")),
                enabled=t.get("enabled", True),
            ))

        return mgr

    def export_events_json(self) -> str:
        """Export all events as JSON."""
        return json.dumps([e.to_dict() for e in self._events], indent=2)


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

def create_conservative_killswitch() -> KillSwitchManager:
    """Pre-configured kill switch with strict safety thresholds."""
    mgr = KillSwitchManager(cooldown_seconds=120)
    mgr.set_strategy(KillStrategy(
        kind=StrategyKind.GRACEFUL,
        timeout_seconds=15,
        preserve_state=True,
        escalate_on_timeout=True,
    ))
    mgr.add_trigger(TriggerCondition(
        kind=TriggerKind.RESOURCE_CPU, threshold=80.0,
        sustained_seconds=10, label="CPU high", severity=Severity.HIGH,
    ))
    mgr.add_trigger(TriggerCondition(
        kind=TriggerKind.RESOURCE_MEMORY, threshold=1024,
        label="Memory > 1GB", severity=Severity.HIGH,
    ))
    mgr.add_trigger(TriggerCondition(
        kind=TriggerKind.BEHAVIOR_ANOMALY, threshold=0.7,
        label="Anomaly critical", severity=Severity.CRITICAL,
    ))
    mgr.add_trigger(TriggerCondition(
        kind=TriggerKind.TIME_LIMIT, threshold=7200,
        label="2h time limit", severity=Severity.MEDIUM,
    ))
    mgr.add_trigger(TriggerCondition(
        kind=TriggerKind.ERROR_RATE, threshold=0.15,
        label="Error rate > 15%", severity=Severity.HIGH,
    ))
    return mgr


def create_aggressive_killswitch() -> KillSwitchManager:
    """Pre-configured kill switch with tight thresholds for high-risk agents."""
    mgr = KillSwitchManager(cooldown_seconds=30)
    mgr.set_strategy(KillStrategy(
        kind=StrategyKind.FORCEFUL,
        timeout_seconds=5,
        preserve_state=False,
        escalate_on_timeout=False,
    ))
    mgr.add_trigger(TriggerCondition(
        kind=TriggerKind.RESOURCE_CPU, threshold=60.0,
        sustained_seconds=5, label="CPU elevated", severity=Severity.HIGH,
    ))
    mgr.add_trigger(TriggerCondition(
        kind=TriggerKind.RESOURCE_MEMORY, threshold=512,
        label="Memory > 512MB", severity=Severity.HIGH,
    ))
    mgr.add_trigger(TriggerCondition(
        kind=TriggerKind.BEHAVIOR_ANOMALY, threshold=0.5,
        label="Any anomaly", severity=Severity.CRITICAL,
    ))
    mgr.add_trigger(TriggerCondition(
        kind=TriggerKind.TIME_LIMIT, threshold=1800,
        label="30m time limit", severity=Severity.MEDIUM,
    ))
    mgr.add_trigger(TriggerCondition(
        kind=TriggerKind.REQUEST_RATE, threshold=100,
        label="Request flood", severity=Severity.CRITICAL,
    ))
    mgr.add_trigger(TriggerCondition(
        kind=TriggerKind.ERROR_RATE, threshold=0.05,
        label="Error rate > 5%", severity=Severity.HIGH,
    ))
    return mgr


def create_quarantine_killswitch() -> KillSwitchManager:
    """Pre-configured kill switch that quarantines instead of killing."""
    mgr = KillSwitchManager(cooldown_seconds=300)
    mgr.set_strategy(KillStrategy(
        kind=StrategyKind.QUARANTINE,
        timeout_seconds=60,
        preserve_state=True,
        escalate_on_timeout=True,
    ))
    mgr.add_trigger(TriggerCondition(
        kind=TriggerKind.BEHAVIOR_ANOMALY, threshold=0.6,
        label="Anomaly detected", severity=Severity.HIGH,
    ))
    mgr.add_trigger(TriggerCondition(
        kind=TriggerKind.RESOURCE_CPU, threshold=95.0,
        sustained_seconds=60, label="CPU sustained extreme", severity=Severity.CRITICAL,
    ))
    return mgr
