"""Tests for KillSwitchTuningAdvisor."""

from __future__ import annotations

import json
from typing import Iterator, List

import pytest

from replication._helpers import Severity
from replication.kill_switch import (
    KillEvent,
    KillOutcome,
    KillStrategy,
    KillSwitchManager,
    StrategyKind,
    TriggerCondition,
    TriggerKind,
)
from replication.kill_switch_tuner import (
    KillSwitchTuningAdvisor,
    TuningReport,
    to_json,
    to_markdown,
    to_text,
)


def make_now(values: List[float]):
    it: Iterator[float] = iter(values)
    last = [values[-1]]

    def _now() -> float:
        try:
            v = next(it)
            last[0] = v
            return v
        except StopIteration:
            return last[0]

    return _now


def _ev(
    agent_id: str,
    ts: float,
    *,
    triggers=("cpu",),
    outcome=KillOutcome.KILLED,
    strategy=StrategyKind.GRACEFUL,
    metadata=None,
) -> KillEvent:
    return KillEvent(
        agent_id=agent_id,
        timestamp=ts,
        triggers=list(triggers),
        strategy=strategy,
        outcome=outcome,
        metadata=metadata or {},
    )


def _basic_manager(cooldown: float = 60.0) -> KillSwitchManager:
    mgr = KillSwitchManager(cooldown_seconds=cooldown)
    mgr.add_trigger(
        TriggerCondition(
            kind=TriggerKind.RESOURCE_CPU,
            threshold=90.0,
            label="cpu",
            severity=Severity.HIGH,
        )
    )
    return mgr


def test_empty_manager_is_healthy_with_insufficient_data() -> None:
    mgr = KillSwitchManager()
    advisor = KillSwitchTuningAdvisor(now_fn=make_now([1000.0]))
    report = advisor.analyze(mgr)
    assert isinstance(report, TuningReport)
    assert report.verdict == "HEALTHY"
    assert report.grade == "A"
    assert report.findings == []
    assert "HEALTHY_TUNING" in report.insights
    assert "INSUFFICIENT_DATA" in report.insights
    assert report.playbook and report.playbook[0].id == "healthy_tuning"


def test_hair_trigger_promotes_to_p0() -> None:
    mgr = _basic_manager()
    # Same agent killed twice within 5s
    mgr._events.extend([
        _ev("a1", 1000.0),
        _ev("a1", 1004.0),
    ])
    advisor = KillSwitchTuningAdvisor(now_fn=make_now([1100.0]))
    report = advisor.analyze(mgr)
    codes = [f.code for f in report.findings]
    assert "HAIR_TRIGGER" in codes
    assert any(f.priority == "P0" for f in report.findings if f.code == "HAIR_TRIGGER")
    assert report.verdict == "NEEDS_URGENT_TUNING"
    assert any(a.id.startswith("raise_threshold::") for a in report.playbook)


def test_dead_trigger_is_p2_with_lower_threshold_suggestion() -> None:
    mgr = _basic_manager()
    mgr.add_trigger(
        TriggerCondition(
            kind=TriggerKind.RESOURCE_MEMORY,
            threshold=4096.0,
            label="mem",
            severity=Severity.MEDIUM,
        )
    )
    # cpu fires (so events list is non-empty), mem never fires
    mgr._events.append(_ev("a1", 1000.0, triggers=("cpu",)))
    advisor = KillSwitchTuningAdvisor(now_fn=make_now([1100.0]))
    report = advisor.analyze(mgr)
    mem_findings = [f for f in report.findings if f.trigger_label == "mem"]
    assert any(f.code == "DEAD_TRIGGER" for f in mem_findings)
    dead = next(f for f in mem_findings if f.code == "DEAD_TRIGGER")
    assert dead.priority == "P2"
    assert dead.suggested_value == pytest.approx(4096.0 * 0.85)


def test_near_threshold_never_fires_with_samples() -> None:
    mgr = _basic_manager()
    samples = [{"cpu_percent": 85.0} for _ in range(10)]
    advisor = KillSwitchTuningAdvisor(now_fn=make_now([1100.0]))
    report = advisor.analyze(mgr, agent_state_samples=samples)
    near = [f for f in report.findings if f.code == "NEAR_THRESHOLD_NEVER_FIRES"]
    assert near, "expected near-threshold finding"
    assert near[0].suggested_value == pytest.approx(85.0)


def test_cooldown_too_short_detection() -> None:
    mgr = _basic_manager(cooldown=60.0)
    # Three different agents, two kills each, tightly packed
    for agent, base in (("a1", 1000.0), ("a2", 2000.0), ("a3", 3000.0)):
        mgr._events.append(_ev(agent, base, triggers=("cpu",)))
        mgr._events.append(_ev(agent, base + 20.0, triggers=("cpu",)))
    advisor = KillSwitchTuningAdvisor(now_fn=make_now([4000.0]))
    report = advisor.analyze(mgr)
    short = [f for f in report.findings if f.code == "COOLDOWN_TOO_SHORT"]
    assert short
    assert short[0].suggested_value is not None
    assert "COOLDOWN_MISCONFIGURED" in report.insights


def test_cooldown_too_long_detection() -> None:
    mgr = _basic_manager(cooldown=120.0)
    for agent, base in (("a1", 1000.0), ("a2", 2000.0), ("a3", 3000.0)):
        mgr._events.append(_ev(agent, base, triggers=("cpu",)))
        mgr._events.append(
            _ev(agent, base + 80.0, triggers=("cpu",), outcome=KillOutcome.COOLDOWN_BLOCKED)
        )
    advisor = KillSwitchTuningAdvisor(now_fn=make_now([4000.0]))
    report = advisor.analyze(mgr)
    long_block = [f for f in report.findings if f.code == "COOLDOWN_TOO_LONG"]
    assert long_block
    assert any(a.id == "decrease_cooldown" for a in report.playbook)


def test_strategy_failure_pattern_escalates() -> None:
    mgr = _basic_manager()
    mgr.set_strategy(KillStrategy(kind=StrategyKind.GRACEFUL))
    mgr._events.extend([
        _ev("a1", 1000.0, outcome=KillOutcome.FAILED),
        _ev("a2", 1100.0, outcome=KillOutcome.FAILED),
    ])
    advisor = KillSwitchTuningAdvisor(now_fn=make_now([1200.0]))
    report = advisor.analyze(mgr)
    fail = [f for f in report.findings if f.code == "STRATEGY_FAILURE_PATTERN"]
    assert fail and fail[0].priority == "P0"
    assert fail[0].suggested_value == "forceful"
    assert any(a.id == "escalate_strategy" for a in report.playbook)
    assert "STRATEGY_DEGRADED" in report.insights
    assert report.grade == "F"


def test_sustained_too_long_never_fires() -> None:
    mgr = KillSwitchManager()
    mgr.add_trigger(
        TriggerCondition(
            kind=TriggerKind.RESOURCE_CPU,
            threshold=90.0,
            sustained_seconds=120.0,
            label="cpu-sustained",
            severity=Severity.HIGH,
        )
    )
    # Need >=10 events present (other triggers can fill the bucket)
    for i in range(12):
        mgr._events.append(
            _ev(f"a{i}", 1000.0 + i, triggers=("other",))
        )
    advisor = KillSwitchTuningAdvisor(now_fn=make_now([2000.0]))
    report = advisor.analyze(mgr)
    sus = [f for f in report.findings if f.code == "SUSTAINED_TOO_LONG"]
    assert sus
    assert sus[0].suggested_value == pytest.approx(60.0)


def test_severity_inversion_when_low_trigger_dominates() -> None:
    mgr = KillSwitchManager()
    mgr.add_trigger(
        TriggerCondition(
            kind=TriggerKind.RESOURCE_CPU,
            threshold=10.0,
            label="noise",
            severity=Severity.LOW,
        )
    )
    mgr.add_trigger(
        TriggerCondition(
            kind=TriggerKind.RESOURCE_MEMORY,
            threshold=4096.0,
            label="mem",
            severity=Severity.HIGH,
        )
    )
    # noise fires 5 times, mem fires once
    for i in range(5):
        mgr._events.append(_ev(f"a{i}", 1000.0 + i, triggers=("noise",)))
    mgr._events.append(_ev("ax", 2000.0, triggers=("mem",)))
    advisor = KillSwitchTuningAdvisor(now_fn=make_now([3000.0]))
    report = advisor.analyze(mgr)
    inv = [f for f in report.findings if f.code == "SEVERITY_INVERSION"]
    assert inv
    assert inv[0].trigger_label == "noise"
    assert inv[0].suggested_value == "high"


def test_disabled_trigger_with_breaches() -> None:
    mgr = KillSwitchManager()
    trig = TriggerCondition(
        kind=TriggerKind.RESOURCE_CPU,
        threshold=80.0,
        label="cpu-off",
        severity=Severity.HIGH,
        enabled=False,
    )
    mgr.add_trigger(trig)
    samples = [{"cpu_percent": v} for v in (95.0, 92.0, 88.0, 99.0)]
    advisor = KillSwitchTuningAdvisor(now_fn=make_now([1000.0]))
    report = advisor.analyze(mgr, agent_state_samples=samples)
    dis = [f for f in report.findings if f.code == "DISABLED_TRIGGER_WITH_BREACHES"]
    assert dis
    assert any(a.id.startswith("re_enable_trigger::") for a in report.playbook)


def test_risk_appetite_cautious_vs_aggressive() -> None:
    mgr = _basic_manager()
    # Single mild finding: SEVERITY_INVERSION via dominant low-sev trigger
    mgr.add_trigger(
        TriggerCondition(
            kind=TriggerKind.RESOURCE_MEMORY,
            threshold=10.0,
            label="noise",
            severity=Severity.LOW,
        )
    )
    for i in range(6):
        mgr._events.append(_ev(f"a{i}", 1000.0 + i, triggers=("noise",)))
    advisor = KillSwitchTuningAdvisor(now_fn=make_now([2000.0]))
    cautious = advisor.analyze(mgr, risk_appetite="cautious")
    balanced = advisor.analyze(mgr, risk_appetite="balanced")
    aggressive = advisor.analyze(mgr, risk_appetite="aggressive")
    assert cautious.tuning_risk_score >= balanced.tuning_risk_score >= aggressive.tuning_risk_score
    # Aggressive should trim P3 healthy_tuning when findings exist
    assert not any(a.id == "healthy_tuning" for a in aggressive.playbook)


def test_json_byte_stability_deterministic() -> None:
    mgr = _basic_manager()
    mgr._events.extend([
        _ev("a1", 1000.0),
        _ev("a1", 1004.0),
    ])
    advisor = KillSwitchTuningAdvisor(now_fn=make_now([1100.0]))
    r1 = advisor.analyze(mgr)
    r2 = advisor.analyze(mgr)
    j1 = to_json(r1)
    j2 = to_json(r2)
    assert j1 == j2
    parsed = json.loads(j1)
    # sort_keys=True at top level
    assert list(parsed.keys()) == sorted(parsed.keys())


def test_markdown_contains_all_sections() -> None:
    mgr = _basic_manager()
    mgr._events.append(_ev("a1", 1000.0))
    advisor = KillSwitchTuningAdvisor(now_fn=make_now([1100.0]))
    report = advisor.analyze(mgr)
    md = to_markdown(report)
    for header in ("## Headline", "## Summary", "## Findings", "## Playbook", "## Insights"):
        assert header in md, f"missing {header}"
    txt = to_text(report)
    for header in ("## Headline", "## Summary", "## Findings", "## Playbook", "## Insights"):
        assert header in txt


def test_advisor_does_not_mutate_manager() -> None:
    mgr = _basic_manager()
    mgr._events.append(_ev("a1", 1000.0))
    snapshot_events = list(mgr._events)
    snapshot_triggers = list(mgr._triggers)
    snapshot_strategy = mgr._strategy.kind
    snapshot_cd = mgr.cooldown_seconds
    advisor = KillSwitchTuningAdvisor(now_fn=make_now([1100.0]))
    advisor.analyze(mgr, agent_state_samples=[{"cpu_percent": 99.0}])
    assert mgr._events == snapshot_events
    assert mgr._triggers == snapshot_triggers
    assert mgr._strategy.kind == snapshot_strategy
    assert mgr.cooldown_seconds == snapshot_cd


def test_invalid_risk_appetite_raises() -> None:
    mgr = _basic_manager()
    advisor = KillSwitchTuningAdvisor()
    with pytest.raises(ValueError):
        advisor.analyze(mgr, risk_appetite="reckless")
