"""Kill-switch tuning advisor.

Agentic sibling to ``kill_switch.py`` / ``safety_drill.py`` /
``adaptive_thresholds.py`` / ``remediation_*`` that audits a live
``KillSwitchManager`` instance and recommends tuning changes
(threshold/cooldown/strategy/severity adjustments) based on the
manager's own event history plus optional observed agent state
samples.

The advisor is read-only: it never mutates the passed manager,
trigger list, event log, or cooldown table.  All time access goes
through an injectable ``now_fn`` callable so reports are
deterministic and byte-stable across runs.

Detectors
---------

The advisor emits one or more :class:`TuningFinding` per audit and
maps them into a deduped, P0-first :class:`TuningAction` playbook.
Each finding has a category code and a severity 0-100 modulated by
``risk_appetite`` ('cautious' x1.15, 'balanced' x1.0,
'aggressive' x0.85).

* HAIR_TRIGGER (P0) - fires too quickly after agent start.
* TRIGGER_FATIGUE (P0) - same label dominates COOLDOWN_BLOCKED / FAILED.
* DEAD_TRIGGER (P2) - enabled but never fires.
* NEAR_THRESHOLD_NEVER_FIRES (P1) - many samples within 90-99% of
  threshold but zero fires.
* COOLDOWN_TOO_SHORT (P1) - consecutive KILLED events tightly packed.
* COOLDOWN_TOO_LONG (P2) - many COOLDOWN_BLOCKED outcomes long after
  the prior kill.
* STRATEGY_FAILURE_PATTERN (P0) - >=2 FAILED kill outcomes.
* SUSTAINED_TOO_LONG (P2) - sustain configured but never fires.
* SEVERITY_INVERSION (P2) - low-severity trigger dominates fires.
* DISABLED_TRIGGER_WITH_BREACHES (P1) - disabled trigger would have
  fired against sampled state.

Usage::

    from replication.kill_switch_tuner import (
        KillSwitchTuningAdvisor, to_markdown,
    )

    advisor = KillSwitchTuningAdvisor()
    report = advisor.analyze(manager, agent_state_samples=recent_states,
                             risk_appetite="cautious")
    print(to_markdown(report))
"""

from __future__ import annotations

import copy
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence

from .kill_switch import (
    KillEvent,
    KillOutcome,
    KillSwitchManager,
    StrategyKind,
    TRIGGER_STATE_KEY,
    TriggerCondition,
    TriggerKind,
)
from ._helpers import Severity


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TuningFinding:
    """A single tuning issue discovered by the advisor."""

    code: str
    trigger_label: Optional[str]
    priority: str
    severity: float
    reason: str
    suggested_value: Any = None
    related_events: List[str] = field(default_factory=list)


@dataclass
class TuningAction:
    """A single playbook action recommendation."""

    id: str
    priority: str
    label: str
    reason: str
    owner: str
    blast_radius: int
    reversibility: str
    related_triggers: List[str] = field(default_factory=list)
    suggested_value: Any = None


@dataclass
class TuningReport:
    """Top-level audit report."""

    headline: str
    verdict: str
    grade: str
    tuning_risk_score: float
    findings: List[TuningFinding]
    playbook: List[TuningAction]
    insights: List[str]
    summary: Dict[str, Any]
    risk_appetite: str
    generated_at: float


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_APPETITES = ("cautious", "balanced", "aggressive")
_APPETITE_MULT = {"cautious": 1.15, "balanced": 1.0, "aggressive": 0.85}

_PRIORITY_RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}

_SEVERITY_ORDER = [
    Severity.INFO,
    Severity.LOW,
    Severity.MEDIUM,
    Severity.HIGH,
    Severity.CRITICAL,
]


def _severity_to_priority(value: float) -> str:
    if value >= 70:
        return "P0"
    if value >= 50:
        return "P1"
    if value >= 30:
        return "P2"
    return "P3"


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# Advisor
# ---------------------------------------------------------------------------


class KillSwitchTuningAdvisor:
    """Audit a KillSwitchManager and recommend tuning changes."""

    def __init__(
        self,
        *,
        min_observations: int = 5,
        now_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        self.min_observations = max(1, int(min_observations))
        self._now_fn = now_fn or time.time

    # -- public ------------------------------------------------------

    def analyze(
        self,
        manager: KillSwitchManager,
        *,
        agent_state_samples: Optional[Sequence[Dict[str, Any]]] = None,
        risk_appetite: str = "balanced",
    ) -> TuningReport:
        if risk_appetite not in _APPETITES:
            raise ValueError(f"risk_appetite must be one of {_APPETITES}")

        # Snapshot everything we read - never mutate the manager.
        triggers: List[TriggerCondition] = list(manager._triggers)
        events: List[KillEvent] = list(manager._events)
        cooldown_seconds = float(manager.cooldown_seconds)
        strategy_kind = manager._strategy.kind
        samples = [copy.deepcopy(s) for s in (agent_state_samples or [])]

        appetite_mult = _APPETITE_MULT[risk_appetite]
        findings: List[TuningFinding] = []

        # Per-trigger fire counts derived from events
        fires_by_label: Dict[str, List[KillEvent]] = {}
        for ev in events:
            for label in ev.triggers:
                fires_by_label.setdefault(label, []).append(ev)

        # ----- detectors --------------------------------------------------
        findings.extend(self._detect_hair_trigger(triggers, events, appetite_mult))
        findings.extend(
            self._detect_trigger_fatigue(triggers, events, appetite_mult)
        )
        findings.extend(
            self._detect_dead_trigger(triggers, events, fires_by_label, appetite_mult)
        )
        findings.extend(
            self._detect_near_threshold_never_fires(
                triggers, fires_by_label, samples, appetite_mult
            )
        )
        findings.extend(
            self._detect_cooldown_too_short(events, cooldown_seconds, appetite_mult)
        )
        findings.extend(
            self._detect_cooldown_too_long(events, cooldown_seconds, appetite_mult)
        )
        findings.extend(self._detect_strategy_failure(events, strategy_kind, appetite_mult))
        findings.extend(
            self._detect_sustained_too_long(
                triggers, fires_by_label, events, appetite_mult
            )
        )
        findings.extend(
            self._detect_severity_inversion(triggers, fires_by_label, appetite_mult)
        )
        findings.extend(
            self._detect_disabled_with_breaches(triggers, samples, appetite_mult)
        )

        # Sort findings deterministically: priority asc, severity desc, code asc.
        findings.sort(
            key=lambda f: (
                _PRIORITY_RANK.get(f.priority, 9),
                -f.severity,
                f.code,
                f.trigger_label or "",
            )
        )

        # ----- scoring ----------------------------------------------------
        if findings:
            sevs = sorted((f.severity for f in findings), reverse=True)
            top = sevs[0]
            rest_sum = min(sum(sevs[1:]), 60)
            score = _clamp(top + 0.4 * rest_sum)
        else:
            score = 0.0

        has_p0 = any(f.priority == "P0" for f in findings)
        has_p1 = any(f.priority == "P1" for f in findings)

        if not findings:
            verdict = "HEALTHY"
        elif has_p0:
            verdict = "NEEDS_URGENT_TUNING"
        elif has_p1:
            verdict = "TUNING_RECOMMENDED"
        else:
            verdict = "MINOR_TUNING"

        if has_p0 or score >= 75:
            grade = "F"
        elif score >= 55:
            grade = "D"
        elif score >= 35:
            grade = "C"
        elif score >= 18:
            grade = "B"
        else:
            grade = "A"

        # ----- playbook ---------------------------------------------------
        playbook = self._build_playbook(
            findings,
            risk_appetite=risk_appetite,
            grade=grade,
            strategy_kind=strategy_kind,
            cooldown_seconds=cooldown_seconds,
            triggers=triggers,
        )

        # ----- insights / summary -----------------------------------------
        insights = self._build_insights(findings, events)
        summary = self._build_summary(triggers, events)

        headline = (
            f"VERDICT: {verdict} grade={grade} "
            f"risk={score:.1f} findings={len(findings)} "
            f"P0={sum(1 for f in findings if f.priority=='P0')} "
            f"P1={sum(1 for f in findings if f.priority=='P1')}"
        )

        return TuningReport(
            headline=headline,
            verdict=verdict,
            grade=grade,
            tuning_risk_score=round(score, 2),
            findings=findings,
            playbook=playbook,
            insights=insights,
            summary=summary,
            risk_appetite=risk_appetite,
            generated_at=float(self._now_fn()),
        )

    # -- detectors ---------------------------------------------------

    def _detect_hair_trigger(
        self,
        triggers: List[TriggerCondition],
        events: List[KillEvent],
        appetite_mult: float,
    ) -> List[TuningFinding]:
        out: List[TuningFinding] = []
        # Per-trigger gather first-seen times
        by_label: Dict[str, List[KillEvent]] = {}
        for ev in events:
            if ev.outcome != KillOutcome.KILLED:
                continue
            for label in ev.triggers:
                by_label.setdefault(label, []).append(ev)
        for trig in triggers:
            evs = by_label.get(trig.label, [])
            if not evs:
                continue
            # Use earliest agent uptime if available via metadata.
            uptimes: List[float] = []
            for ev in evs:
                up = ev.metadata.get("uptime_at_kill") if ev.metadata else None
                if isinstance(up, (int, float)):
                    uptimes.append(float(up))
            triggered_fast = False
            reason_extra = ""
            if uptimes and statistics.median(uptimes) <= 15.0 and trig.sustained_seconds == 0:
                triggered_fast = True
                reason_extra = f"median uptime at kill={statistics.median(uptimes):.1f}s"
            # Per-agent re-fire within 30s
            by_agent: Dict[str, List[float]] = {}
            for ev in evs:
                by_agent.setdefault(ev.agent_id, []).append(ev.timestamp)
            for aid, ts in by_agent.items():
                ts.sort()
                for i in range(1, len(ts)):
                    if ts[i] - ts[i - 1] <= 30.0:
                        triggered_fast = True
                        reason_extra = (
                            f"agent {aid} re-killed in {ts[i] - ts[i-1]:.1f}s"
                        )
                        break
                if triggered_fast:
                    break
            if not triggered_fast:
                continue
            sev = _clamp(70 * appetite_mult)
            out.append(
                TuningFinding(
                    code="HAIR_TRIGGER",
                    trigger_label=trig.label,
                    priority=_severity_to_priority(sev),
                    severity=round(sev, 2),
                    reason=(
                        f"Trigger '{trig.label}' fires too fast: {reason_extra}."
                    ),
                    suggested_value=round(trig.threshold * 1.20, 4) if trig.threshold > 0 else None,
                    related_events=[e.agent_id for e in evs[:5]],
                )
            )
        return out

    def _detect_trigger_fatigue(
        self,
        triggers: List[TriggerCondition],
        events: List[KillEvent],
        appetite_mult: float,
    ) -> List[TuningFinding]:
        out: List[TuningFinding] = []
        # Group by sole-trigger label.
        sole_by_label: Dict[str, List[KillEvent]] = {}
        for ev in events:
            if len(ev.triggers) == 1:
                sole_by_label.setdefault(ev.triggers[0], []).append(ev)
        for label, evs in sole_by_label.items():
            agents = {ev.agent_id for ev in evs}
            if len(agents) < 5:
                continue
            # Look at the NEXT event per agent for blocked/failed outcomes.
            blocked_or_failed = 0
            for aid in agents:
                aid_events = [e for e in events if e.agent_id == aid]
                aid_events.sort(key=lambda e: e.timestamp)
                # find first event for this label, then check the next
                for i, e in enumerate(aid_events):
                    if label in e.triggers and i + 1 < len(aid_events):
                        nxt = aid_events[i + 1]
                        if nxt.outcome in (
                            KillOutcome.COOLDOWN_BLOCKED,
                            KillOutcome.FAILED,
                        ):
                            blocked_or_failed += 1
                        break
            ratio = blocked_or_failed / max(1, len(agents))
            if ratio < 0.6:
                continue
            sev = _clamp(75 * appetite_mult)
            out.append(
                TuningFinding(
                    code="TRIGGER_FATIGUE",
                    trigger_label=label,
                    priority=_severity_to_priority(sev),
                    severity=round(sev, 2),
                    reason=(
                        f"Trigger '{label}' dominates {len(agents)} agents and "
                        f"{ratio:.0%} re-fire blocked/failed."
                    ),
                    related_events=sorted(agents)[:5],
                )
            )
        return out

    def _detect_dead_trigger(
        self,
        triggers: List[TriggerCondition],
        events: List[KillEvent],
        fires_by_label: Dict[str, List[KillEvent]],
        appetite_mult: float,
    ) -> List[TuningFinding]:
        out: List[TuningFinding] = []
        if not events:
            return out
        for trig in triggers:
            if not trig.enabled:
                continue
            if fires_by_label.get(trig.label):
                continue
            if trig.kind in (TriggerKind.CUSTOM, TriggerKind.MANUAL):
                continue
            if trig.threshold <= 0:
                continue
            sev = _clamp(35 * appetite_mult)
            out.append(
                TuningFinding(
                    code="DEAD_TRIGGER",
                    trigger_label=trig.label,
                    priority=_severity_to_priority(sev),
                    severity=round(sev, 2),
                    reason=(
                        f"Trigger '{trig.label}' enabled with threshold "
                        f"{trig.threshold} but never fired across "
                        f"{len(events)} events."
                    ),
                    suggested_value=round(trig.threshold * 0.85, 4),
                )
            )
        return out

    def _detect_near_threshold_never_fires(
        self,
        triggers: List[TriggerCondition],
        fires_by_label: Dict[str, List[KillEvent]],
        samples: List[Dict[str, Any]],
        appetite_mult: float,
    ) -> List[TuningFinding]:
        out: List[TuningFinding] = []
        if not samples:
            return out
        for trig in triggers:
            if fires_by_label.get(trig.label):
                continue
            key = TRIGGER_STATE_KEY.get(trig.kind)
            if not key or trig.threshold <= 0:
                continue
            values = [
                float(s[key]) for s in samples if isinstance(s.get(key), (int, float))
            ]
            if not values:
                continue
            band_lo = trig.threshold * 0.90
            band_hi = trig.threshold * 0.99
            in_band = [v for v in values if band_lo <= v <= band_hi]
            ratio = len(in_band) / len(values)
            if ratio < 0.30:
                continue
            # p90 of the observed values
            sorted_v = sorted(values)
            idx = max(0, int(round(0.90 * (len(sorted_v) - 1))))
            p90 = sorted_v[idx]
            sev = _clamp(50 * appetite_mult)
            out.append(
                TuningFinding(
                    code="NEAR_THRESHOLD_NEVER_FIRES",
                    trigger_label=trig.label,
                    priority=_severity_to_priority(sev),
                    severity=round(sev, 2),
                    reason=(
                        f"{ratio:.0%} of samples sit in 90-99% of threshold "
                        f"{trig.threshold} for '{trig.label}' but trigger "
                        f"never fires."
                    ),
                    suggested_value=round(p90, 1),
                )
            )
        return out

    def _detect_cooldown_too_short(
        self,
        events: List[KillEvent],
        cooldown_seconds: float,
        appetite_mult: float,
    ) -> List[TuningFinding]:
        out: List[TuningFinding] = []
        per_agent_kills: Dict[str, List[float]] = {}
        for ev in events:
            if ev.outcome == KillOutcome.KILLED:
                per_agent_kills.setdefault(ev.agent_id, []).append(ev.timestamp)
        tight_intervals: List[float] = []
        tight_agents = set()
        for aid, ts in per_agent_kills.items():
            ts.sort()
            for i in range(1, len(ts)):
                delta = ts[i] - ts[i - 1]
                if delta <= cooldown_seconds * 1.2:
                    tight_intervals.append(delta)
                    tight_agents.add(aid)
        if len(tight_agents) < 3 or not tight_intervals:
            return out
        sorted_iv = sorted(tight_intervals)
        idx = max(0, int(round(0.75 * (len(sorted_iv) - 1))))
        p75 = sorted_iv[idx]
        recommended = max(p75 * 1.5, cooldown_seconds * 1.5)
        sev = _clamp(55 * appetite_mult)
        out.append(
            TuningFinding(
                code="COOLDOWN_TOO_SHORT",
                trigger_label=None,
                priority=_severity_to_priority(sev),
                severity=round(sev, 2),
                reason=(
                    f"{len(tight_agents)} agents re-killed within "
                    f"{cooldown_seconds * 1.2:.1f}s of prior kill "
                    f"(p75 interval={p75:.1f}s)."
                ),
                suggested_value=round(recommended, 2),
                related_events=sorted(tight_agents)[:5],
            )
        )
        return out

    def _detect_cooldown_too_long(
        self,
        events: List[KillEvent],
        cooldown_seconds: float,
        appetite_mult: float,
    ) -> List[TuningFinding]:
        out: List[TuningFinding] = []
        if cooldown_seconds <= 0:
            return out
        # Find COOLDOWN_BLOCKED events where the prior KILLED for the same
        # agent occurred > cooldown_seconds/3 ago.
        per_agent: Dict[str, List[KillEvent]] = {}
        for ev in events:
            per_agent.setdefault(ev.agent_id, []).append(ev)
        long_blocked = 0
        for aid, evs in per_agent.items():
            evs_sorted = sorted(evs, key=lambda e: e.timestamp)
            last_kill_ts: Optional[float] = None
            for e in evs_sorted:
                if e.outcome == KillOutcome.KILLED:
                    last_kill_ts = e.timestamp
                elif e.outcome == KillOutcome.COOLDOWN_BLOCKED and last_kill_ts is not None:
                    if (e.timestamp - last_kill_ts) > cooldown_seconds / 3.0:
                        long_blocked += 1
        if long_blocked < 3:
            return out
        sev = _clamp(30 * appetite_mult)
        out.append(
            TuningFinding(
                code="COOLDOWN_TOO_LONG",
                trigger_label=None,
                priority=_severity_to_priority(sev),
                severity=round(sev, 2),
                reason=(
                    f"{long_blocked} cooldown-blocked events fired well after "
                    f"the prior kill (>{cooldown_seconds/3.0:.1f}s)."
                ),
                suggested_value=round(cooldown_seconds * 0.7, 2),
            )
        )
        return out

    def _detect_strategy_failure(
        self,
        events: List[KillEvent],
        strategy_kind: StrategyKind,
        appetite_mult: float,
    ) -> List[TuningFinding]:
        out: List[TuningFinding] = []
        failed = [e for e in events if e.outcome == KillOutcome.FAILED]
        if len(failed) < 2:
            return out
        if strategy_kind == StrategyKind.GRACEFUL:
            suggestion = StrategyKind.FORCEFUL.value
        elif strategy_kind == StrategyKind.FORCEFUL:
            suggestion = StrategyKind.QUARANTINE.value
        else:
            suggestion = "review_escalation_policy"
        sev = _clamp(80 * appetite_mult)
        out.append(
            TuningFinding(
                code="STRATEGY_FAILURE_PATTERN",
                trigger_label=None,
                priority=_severity_to_priority(sev),
                severity=round(sev, 2),
                reason=(
                    f"{len(failed)} kills FAILED under strategy "
                    f"{strategy_kind.value}; escalate to {suggestion}."
                ),
                suggested_value=suggestion,
                related_events=[e.agent_id for e in failed[:5]],
            )
        )
        return out

    def _detect_sustained_too_long(
        self,
        triggers: List[TriggerCondition],
        fires_by_label: Dict[str, List[KillEvent]],
        events: List[KillEvent],
        appetite_mult: float,
    ) -> List[TuningFinding]:
        out: List[TuningFinding] = []
        if len(events) < 10:
            return out
        for trig in triggers:
            if trig.sustained_seconds <= 60:
                continue
            if fires_by_label.get(trig.label):
                continue
            sev = _clamp(25 * appetite_mult)
            out.append(
                TuningFinding(
                    code="SUSTAINED_TOO_LONG",
                    trigger_label=trig.label,
                    priority=_severity_to_priority(sev),
                    severity=round(sev, 2),
                    reason=(
                        f"Trigger '{trig.label}' requires "
                        f"{trig.sustained_seconds}s sustain and never fires; "
                        f"halve to {trig.sustained_seconds/2:.1f}s."
                    ),
                    suggested_value=round(trig.sustained_seconds / 2.0, 2),
                )
            )
        return out

    def _detect_severity_inversion(
        self,
        triggers: List[TriggerCondition],
        fires_by_label: Dict[str, List[KillEvent]],
        appetite_mult: float,
    ) -> List[TuningFinding]:
        out: List[TuningFinding] = []
        total_fires = sum(len(v) for v in fires_by_label.values())
        if total_fires < 4:
            return out
        # Find the trigger with the most fires
        best_label: Optional[str] = None
        best_count = 0
        for label, fires in fires_by_label.items():
            if len(fires) > best_count:
                best_count = len(fires)
                best_label = label
        if not best_label or best_count / total_fires < 0.5:
            return out
        # Match back to TriggerCondition
        match = next((t for t in triggers if t.label == best_label), None)
        if match is None:
            return out
        if match.severity in (Severity.HIGH, Severity.CRITICAL):
            return out
        sev = _clamp(30 * appetite_mult)
        out.append(
            TuningFinding(
                code="SEVERITY_INVERSION",
                trigger_label=best_label,
                priority=_severity_to_priority(sev),
                severity=round(sev, 2),
                reason=(
                    f"Trigger '{best_label}' has severity "
                    f"{match.severity.value} but accounts for "
                    f"{best_count}/{total_fires} fires."
                ),
                suggested_value=Severity.HIGH.value,
            )
        )
        return out

    def _detect_disabled_with_breaches(
        self,
        triggers: List[TriggerCondition],
        samples: List[Dict[str, Any]],
        appetite_mult: float,
    ) -> List[TuningFinding]:
        out: List[TuningFinding] = []
        if not samples:
            return out
        for trig in triggers:
            if trig.enabled:
                continue
            key = TRIGGER_STATE_KEY.get(trig.kind)
            if not key or trig.threshold <= 0:
                continue
            hits = sum(
                1
                for s in samples
                if isinstance(s.get(key), (int, float))
                and float(s[key]) >= trig.threshold
            )
            if hits < 3:
                continue
            sev = _clamp(45 * appetite_mult)
            out.append(
                TuningFinding(
                    code="DISABLED_TRIGGER_WITH_BREACHES",
                    trigger_label=trig.label,
                    priority=_severity_to_priority(sev),
                    severity=round(sev, 2),
                    reason=(
                        f"Disabled trigger '{trig.label}' would have fired on "
                        f"{hits}/{len(samples)} observed samples."
                    ),
                )
            )
        return out

    # -- playbook ----------------------------------------------------

    def _build_playbook(
        self,
        findings: List[TuningFinding],
        *,
        risk_appetite: str,
        grade: str,
        strategy_kind: StrategyKind,
        cooldown_seconds: float,
        triggers: List[TriggerCondition],
    ) -> List[TuningAction]:
        actions: Dict[str, TuningAction] = {}

        def add(action: TuningAction) -> None:
            existing = actions.get(action.id)
            if existing is None:
                actions[action.id] = action
                return
            # Promote priority if new is higher
            if _PRIORITY_RANK[action.priority] < _PRIORITY_RANK[existing.priority]:
                existing.priority = action.priority
            related = sorted(set(existing.related_triggers) | set(action.related_triggers))
            existing.related_triggers = related

        for f in findings:
            related = [f.trigger_label] if f.trigger_label else []
            if f.code == "HAIR_TRIGGER":
                add(
                    TuningAction(
                        id=f"raise_threshold::{f.trigger_label}",
                        priority="P0",
                        label="Raise trigger threshold",
                        reason=f.reason,
                        owner="oncall",
                        blast_radius=2,
                        reversibility="high",
                        related_triggers=related,
                        suggested_value=f.suggested_value,
                    )
                )
            elif f.code == "TRIGGER_FATIGUE":
                add(
                    TuningAction(
                        id=f"trigger_fatigue_raise::{f.trigger_label}",
                        priority="P0",
                        label="Raise threshold for dominant trigger",
                        reason=f.reason,
                        owner="ops",
                        blast_radius=3,
                        reversibility="medium",
                        related_triggers=related,
                    )
                )
                add(
                    TuningAction(
                        id=f"trigger_fatigue_cooldown::{f.trigger_label}",
                        priority="P0",
                        label="Tune cooldown for dominant trigger",
                        reason=f.reason,
                        owner="ops",
                        blast_radius=3,
                        reversibility="medium",
                        related_triggers=related,
                        suggested_value=round(cooldown_seconds * 1.5, 2),
                    )
                )
            elif f.code in ("DEAD_TRIGGER", "NEAR_THRESHOLD_NEVER_FIRES"):
                add(
                    TuningAction(
                        id=f"lower_threshold::{f.trigger_label}",
                        priority=f.priority,
                        label="Lower trigger threshold",
                        reason=f.reason,
                        owner="safety_eng",
                        blast_radius=2,
                        reversibility="high",
                        related_triggers=related,
                        suggested_value=f.suggested_value,
                    )
                )
            elif f.code == "COOLDOWN_TOO_SHORT":
                add(
                    TuningAction(
                        id="increase_cooldown",
                        priority="P1",
                        label="Increase cooldown_seconds",
                        reason=f.reason,
                        owner="sre",
                        blast_radius=2,
                        reversibility="high",
                        suggested_value=f.suggested_value,
                    )
                )
            elif f.code == "COOLDOWN_TOO_LONG":
                add(
                    TuningAction(
                        id="decrease_cooldown",
                        priority="P2",
                        label="Decrease cooldown_seconds",
                        reason=f.reason,
                        owner="sre",
                        blast_radius=2,
                        reversibility="high",
                        suggested_value=f.suggested_value,
                    )
                )
            elif f.code == "STRATEGY_FAILURE_PATTERN":
                add(
                    TuningAction(
                        id="escalate_strategy",
                        priority="P0",
                        label=f"Escalate kill strategy from {strategy_kind.value}",
                        reason=f.reason,
                        owner="safety_eng",
                        blast_radius=4,
                        reversibility="medium",
                        suggested_value=f.suggested_value,
                    )
                )
            elif f.code == "SUSTAINED_TOO_LONG":
                add(
                    TuningAction(
                        id=f"reduce_sustain::{f.trigger_label}",
                        priority="P2",
                        label="Reduce sustained_seconds",
                        reason=f.reason,
                        owner="safety_eng",
                        blast_radius=1,
                        reversibility="high",
                        related_triggers=related,
                        suggested_value=f.suggested_value,
                    )
                )
            elif f.code == "SEVERITY_INVERSION":
                add(
                    TuningAction(
                        id=f"raise_severity::{f.trigger_label}",
                        priority="P2",
                        label="Raise trigger severity",
                        reason=f.reason,
                        owner="safety_eng",
                        blast_radius=1,
                        reversibility="high",
                        related_triggers=related,
                        suggested_value=f.suggested_value,
                    )
                )
            elif f.code == "DISABLED_TRIGGER_WITH_BREACHES":
                add(
                    TuningAction(
                        id=f"re_enable_trigger::{f.trigger_label}",
                        priority="P1",
                        label="Re-enable disabled trigger",
                        reason=f.reason,
                        owner="oncall",
                        blast_radius=2,
                        reversibility="high",
                        related_triggers=related,
                    )
                )

        if not findings:
            add(
                TuningAction(
                    id="healthy_tuning",
                    priority="P3",
                    label="No tuning needed",
                    reason="No tuning issues detected.",
                    owner="safety_eng",
                    blast_radius=1,
                    reversibility="high",
                )
            )

        if risk_appetite == "cautious" and grade in ("C", "D", "F"):
            add(
                TuningAction(
                    id="schedule_tuning_audit",
                    priority="P2",
                    label="Schedule follow-up tuning audit",
                    reason=(
                        "Cautious risk appetite + non-passing grade: book a "
                        "follow-up review."
                    ),
                    owner="platform",
                    blast_radius=1,
                    reversibility="high",
                )
            )

        ordered = sorted(
            actions.values(),
            key=lambda a: (_PRIORITY_RANK.get(a.priority, 9), a.id),
        )

        if risk_appetite == "aggressive":
            has_p01 = any(a.priority in ("P0", "P1") for a in ordered)
            filtered: List[TuningAction] = []
            p2_count = sum(1 for a in ordered if a.priority == "P2")
            for a in ordered:
                if a.priority == "P3" and a.id != "healthy_tuning":
                    continue
                if a.priority == "P2" and has_p01 and p2_count == 1:
                    continue
                filtered.append(a)
            ordered = filtered

        return ordered

    # -- insights ----------------------------------------------------

    def _build_insights(
        self, findings: List[TuningFinding], events: List[KillEvent]
    ) -> List[str]:
        out: List[str] = []
        codes = [f.code for f in findings]
        if codes.count("HAIR_TRIGGER") >= 2:
            out.append("HAIR_TRIGGER_PATTERN")
        if "TRIGGER_FATIGUE" in codes:
            out.append("TRIGGER_FATIGUE_PRESENT")
        if "COOLDOWN_TOO_SHORT" in codes or "COOLDOWN_TOO_LONG" in codes:
            out.append("COOLDOWN_MISCONFIGURED")
        if "STRATEGY_FAILURE_PATTERN" in codes:
            out.append("STRATEGY_DEGRADED")
        if codes.count("DEAD_TRIGGER") >= 2:
            out.append("DORMANT_TRIGGERS")
        if "SEVERITY_INVERSION" in codes:
            out.append("MISCALIBRATED_SEVERITY")
        if not findings:
            out.append("HEALTHY_TUNING")
        if len(events) < self.min_observations:
            out.append("INSUFFICIENT_DATA")
        return out

    def _build_summary(
        self, triggers: List[TriggerCondition], events: List[KillEvent]
    ) -> Dict[str, Any]:
        return {
            "total_events": len(events),
            "killed": sum(1 for e in events if e.outcome == KillOutcome.KILLED),
            "cooldown_blocked": sum(
                1 for e in events if e.outcome == KillOutcome.COOLDOWN_BLOCKED
            ),
            "failed": sum(1 for e in events if e.outcome == KillOutcome.FAILED),
            "triggers_total": len(triggers),
            "disabled_triggers": sum(1 for t in triggers if not t.enabled),
        }


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def _finding_to_dict(f: TuningFinding) -> Dict[str, Any]:
    return asdict(f)


def _action_to_dict(a: TuningAction) -> Dict[str, Any]:
    return asdict(a)


def to_text(report: TuningReport) -> str:
    lines: List[str] = []
    lines.append("## Headline")
    lines.append(report.headline)
    lines.append("")
    lines.append("## Summary")
    for k in sorted(report.summary):
        lines.append(f"  {k}: {report.summary[k]}")
    lines.append(f"  risk_appetite: {report.risk_appetite}")
    lines.append(f"  verdict: {report.verdict}")
    lines.append(f"  grade: {report.grade}")
    lines.append(f"  tuning_risk_score: {report.tuning_risk_score:.2f}")
    lines.append("")
    lines.append("## Findings")
    if not report.findings:
        lines.append("  (none)")
    for f in report.findings:
        lines.append(
            f"  [{f.priority}] {f.code} "
            f"trigger={f.trigger_label or '-'} "
            f"sev={f.severity:.1f} suggested={f.suggested_value!r}"
        )
        lines.append(f"      reason: {f.reason}")
    lines.append("")
    lines.append("## Playbook")
    if not report.playbook:
        lines.append("  (none)")
    for a in report.playbook:
        lines.append(
            f"  [{a.priority}] {a.id} owner={a.owner} "
            f"blast={a.blast_radius} rev={a.reversibility} "
            f"suggested={a.suggested_value!r}"
        )
        lines.append(f"      {a.label}: {a.reason}")
    lines.append("")
    lines.append("## Insights")
    if not report.insights:
        lines.append("  (none)")
    for ins in report.insights:
        lines.append(f"  - {ins}")
    return "\n".join(lines)


def to_markdown(report: TuningReport) -> str:
    lines: List[str] = []
    lines.append("# Kill-switch tuning report")
    lines.append("")
    lines.append("## Headline")
    lines.append(f"`{report.headline}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("| Key | Value |")
    lines.append("| --- | --- |")
    for k in sorted(report.summary):
        lines.append(f"| {k} | {report.summary[k]} |")
    lines.append(f"| risk_appetite | {report.risk_appetite} |")
    lines.append(f"| verdict | {report.verdict} |")
    lines.append(f"| grade | {report.grade} |")
    lines.append(f"| tuning_risk_score | {report.tuning_risk_score:.2f} |")
    lines.append("")
    lines.append("## Findings")
    lines.append("| Priority | Code | Trigger | Severity | Suggested | Reason |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    if not report.findings:
        lines.append("| - | - | - | - | - | (none) |")
    for f in report.findings:
        lines.append(
            f"| {f.priority} | {f.code} | {f.trigger_label or '-'} | "
            f"{f.severity:.1f} | {f.suggested_value!r} | {f.reason} |"
        )
    lines.append("")
    lines.append("## Playbook")
    lines.append("| Priority | Action | Owner | Blast | Reversibility | Suggested | Reason |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    if not report.playbook:
        lines.append("| - | - | - | - | - | - | (none) |")
    for a in report.playbook:
        lines.append(
            f"| {a.priority} | {a.label} | {a.owner} | {a.blast_radius} | "
            f"{a.reversibility} | {a.suggested_value!r} | {a.reason} |"
        )
    lines.append("")
    lines.append("## Insights")
    if not report.insights:
        lines.append("- (none)")
    else:
        for ins in report.insights:
            lines.append(f"- {ins}")
    return "\n".join(lines)


def to_json(report: TuningReport) -> str:
    payload = {
        "headline": report.headline,
        "verdict": report.verdict,
        "grade": report.grade,
        "tuning_risk_score": report.tuning_risk_score,
        "findings": [_finding_to_dict(f) for f in report.findings],
        "playbook": [_action_to_dict(a) for a in report.playbook],
        "insights": list(report.insights),
        "summary": dict(report.summary),
        "risk_appetite": report.risk_appetite,
        "generated_at": report.generated_at,
    }
    return json.dumps(payload, sort_keys=True, indent=2, default=str)


__all__ = [
    "KillSwitchTuningAdvisor",
    "TuningAction",
    "TuningFinding",
    "TuningReport",
    "to_json",
    "to_markdown",
    "to_text",
]
