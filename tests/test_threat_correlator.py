"""Tests for ThreatCorrelator — cross-module signal correlation."""

from __future__ import annotations

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from replication.threat_correlator import (
    ThreatCorrelator,
    CorrelatorConfig,
    CorrelationRule,
    CorrelationReport,
    CompoundThreat,
    AgentRisk,
    CoverageGap,
    Signal,
    SignalSource,
    SignalSeverity,
    ThreatLevel,
    ResponseUrgency,
    BUILTIN_RULES,
    _SEVERITY_WEIGHT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sig(
    source: SignalSource = SignalSource.DRIFT,
    severity: SignalSeverity = SignalSeverity.HIGH,
    agent: str = "agent-1",
    ts: float = 1000.0,
    desc: str = "test signal",
) -> Signal:
    return Signal(source=source, severity=severity, agent_id=agent,
                  timestamp=ts, description=desc)


# ---------------------------------------------------------------------------
# Signal tests
# ---------------------------------------------------------------------------


class TestSignal:
    def test_auto_id(self):
        s = _sig()
        assert s.signal_id == "drift-agent-1-1000.0"

    def test_custom_id(self):
        s = Signal(source=SignalSource.CANARY, severity=SignalSeverity.LOW,
                   agent_id="a", timestamp=1.0, description="x",
                   signal_id="custom-123")
        assert s.signal_id == "custom-123"

    def test_metadata_default(self):
        s = _sig()
        assert s.metadata == {}

    def test_metadata_set(self):
        s = _sig()
        s.metadata["key"] = "val"
        assert s.metadata["key"] == "val"


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    def test_signal_sources_count(self):
        assert len(SignalSource) == 10

    def test_severity_weights(self):
        assert _SEVERITY_WEIGHT[SignalSeverity.INFO] < _SEVERITY_WEIGHT[SignalSeverity.CRITICAL]

    def test_threat_levels(self):
        assert len(ThreatLevel) == 5

    def test_response_urgency(self):
        assert len(ResponseUrgency) == 5


# ---------------------------------------------------------------------------
# CorrelatorConfig tests
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self):
        c = CorrelatorConfig()
        assert c.default_time_window == 300.0
        assert c.min_risk_score == 0.0
        assert c.max_threats_per_agent == 50

    def test_custom(self):
        c = CorrelatorConfig(default_time_window=60.0, min_risk_score=3.0)
        assert c.default_time_window == 60.0
        assert c.min_risk_score == 3.0


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    def test_empty_signals(self):
        tc = ThreatCorrelator()
        report = tc.correlate([])
        assert report.total_signals == 0
        assert report.total_threats == 0
        assert report.fleet_risk_level == ThreatLevel.NONE
        assert len(report.coverage_gaps) == len(SignalSource)

    def test_single_signal_no_threats(self):
        tc = ThreatCorrelator()
        report = tc.correlate([_sig()])
        assert report.total_signals == 1
        assert report.total_threats == 0


# ---------------------------------------------------------------------------
# Coordinated breach rule
# ---------------------------------------------------------------------------


class TestCoordinatedBreach:
    def test_triggers_on_drift_escalation_canary(self):
        tc = ThreatCorrelator()
        signals = [
            _sig(SignalSource.DRIFT, SignalSeverity.HIGH, "a1", 1000.0),
            _sig(SignalSource.ESCALATION, SignalSeverity.CRITICAL, "a1", 1010.0),
            _sig(SignalSource.CANARY, SignalSeverity.HIGH, "a1", 1050.0),
        ]
        report = tc.correlate(signals)
        names = [t.rule_name for t in report.threats]
        assert "coordinated_breach" in names

    def test_no_trigger_different_agents(self):
        tc = ThreatCorrelator()
        signals = [
            _sig(SignalSource.DRIFT, SignalSeverity.HIGH, "a1", 1000.0),
            _sig(SignalSource.ESCALATION, SignalSeverity.CRITICAL, "a2", 1010.0),
            _sig(SignalSource.CANARY, SignalSeverity.HIGH, "a3", 1050.0),
        ]
        report = tc.correlate(signals)
        breach = [t for t in report.threats if t.rule_name == "coordinated_breach"]
        assert len(breach) == 0

    def test_no_trigger_outside_window(self):
        tc = ThreatCorrelator()
        signals = [
            _sig(SignalSource.DRIFT, SignalSeverity.HIGH, "a1", 1000.0),
            _sig(SignalSource.ESCALATION, SignalSeverity.CRITICAL, "a1", 1010.0),
            _sig(SignalSource.CANARY, SignalSeverity.HIGH, "a1", 2000.0),  # >120s
        ]
        report = tc.correlate(signals)
        breach = [t for t in report.threats if t.rule_name == "coordinated_breach"]
        assert len(breach) == 0

    def test_risk_multiplier(self):
        tc = ThreatCorrelator()
        signals = [
            _sig(SignalSource.DRIFT, SignalSeverity.CRITICAL, "a1", 1000.0),
            _sig(SignalSource.ESCALATION, SignalSeverity.CRITICAL, "a1", 1010.0),
            _sig(SignalSource.CANARY, SignalSeverity.CRITICAL, "a1", 1020.0),
        ]
        report = tc.correlate(signals)
        breach = [t for t in report.threats if t.rule_name == "coordinated_breach"]
        assert len(breach) == 1
        assert breach[0].risk_score > 5.0  # multiplier = 2.5


# ---------------------------------------------------------------------------
# Custom rules
# ---------------------------------------------------------------------------


class TestCustomRules:
    def test_add_custom_rule(self):
        tc = ThreatCorrelator(rules=[])
        rule = CorrelationRule(
            name="test_rule",
            required_sources={SignalSource.DRIFT, SignalSource.COMPLIANCE},
            min_signals=2,
            time_window=60.0,
            risk_multiplier=1.0,
        )
        tc.add_rule(rule)
        signals = [
            _sig(SignalSource.DRIFT, SignalSeverity.HIGH, "a1", 1000.0),
            _sig(SignalSource.COMPLIANCE, SignalSeverity.HIGH, "a1", 1030.0),
        ]
        report = tc.correlate(signals)
        assert any(t.rule_name == "test_rule" for t in report.threats)

    def test_empty_rules_no_threats(self):
        tc = ThreatCorrelator(rules=[])
        signals = [
            _sig(SignalSource.DRIFT, SignalSeverity.HIGH, "a1", 1000.0),
            _sig(SignalSource.CANARY, SignalSeverity.HIGH, "a1", 1010.0),
        ]
        report = tc.correlate(signals)
        assert report.total_threats == 0


# ---------------------------------------------------------------------------
# Min severity filter
# ---------------------------------------------------------------------------


class TestMinSeverity:
    def test_low_severity_filtered(self):
        tc = ThreatCorrelator(rules=[])
        rule = CorrelationRule(
            name="high_only",
            required_sources={SignalSource.DRIFT, SignalSource.CANARY},
            min_signals=2,
            time_window=60.0,
            min_severity=SignalSeverity.HIGH,
        )
        tc.add_rule(rule)
        signals = [
            _sig(SignalSource.DRIFT, SignalSeverity.LOW, "a1", 1000.0),
            _sig(SignalSource.CANARY, SignalSeverity.LOW, "a1", 1010.0),
        ]
        report = tc.correlate(signals)
        assert report.total_threats == 0


# ---------------------------------------------------------------------------
# Agent risk
# ---------------------------------------------------------------------------


class TestAgentRisk:
    def test_agent_with_threats_ranked_higher(self):
        tc = ThreatCorrelator()
        signals = [
            _sig(SignalSource.DRIFT, SignalSeverity.HIGH, "risky", 1000.0),
            _sig(SignalSource.ESCALATION, SignalSeverity.CRITICAL, "risky", 1010.0),
            _sig(SignalSource.CANARY, SignalSeverity.HIGH, "risky", 1050.0),
            _sig(SignalSource.DRIFT, SignalSeverity.LOW, "safe", 1000.0),
        ]
        report = tc.correlate(signals)
        assert report.agent_risks[0].agent_id == "risky"

    def test_agent_risk_fields(self):
        tc = ThreatCorrelator()
        signals = [
            _sig(SignalSource.DRIFT, SignalSeverity.HIGH, "a1", 1000.0),
            _sig(SignalSource.BEHAVIOR, SignalSeverity.HIGH, "a1", 1010.0),
            _sig(SignalSource.DRIFT, SignalSeverity.MEDIUM, "a1", 1020.0),
        ]
        report = tc.correlate(signals)
        risk = [r for r in report.agent_risks if r.agent_id == "a1"][0]
        assert risk.signal_count == 3
        assert risk.source_diversity == 2


# ---------------------------------------------------------------------------
# Fleet risk
# ---------------------------------------------------------------------------


class TestFleetRisk:
    def test_no_threats_low_risk(self):
        tc = ThreatCorrelator(rules=[])
        signals = [_sig(SignalSource.DRIFT, SignalSeverity.LOW, "a1", 1000.0)]
        report = tc.correlate(signals)
        assert report.fleet_risk_score < 1.0

    def test_fleet_level_scales(self):
        tc = ThreatCorrelator()
        # Generate enough signals to trigger threats
        signals = []
        for i in range(10):
            for src in [SignalSource.DRIFT, SignalSource.ESCALATION, SignalSource.CANARY]:
                signals.append(
                    _sig(src, SignalSeverity.CRITICAL, f"agent-{i}", 1000.0 + i * 5)
                )
        report = tc.correlate(signals)
        assert report.fleet_risk_level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL)


# ---------------------------------------------------------------------------
# Coverage gaps
# ---------------------------------------------------------------------------


class TestCoverageGaps:
    def test_missing_sources_reported(self):
        tc = ThreatCorrelator()
        signals = [_sig(SignalSource.DRIFT, SignalSeverity.LOW, "a1", 1000.0)]
        report = tc.correlate(signals)
        gap_sources = {g.source for g in report.coverage_gaps}
        assert SignalSource.CANARY in gap_sources
        assert SignalSource.DRIFT not in gap_sources

    def test_all_sources_no_gaps(self):
        tc = ThreatCorrelator()
        signals = [
            _sig(src, SignalSeverity.LOW, "a1", 1000.0 + i)
            for i, src in enumerate(SignalSource)
        ]
        report = tc.correlate(signals)
        assert len(report.coverage_gaps) == 0

    def test_coverage_disabled(self):
        config = CorrelatorConfig(include_coverage_analysis=False)
        tc = ThreatCorrelator(config=config)
        signals = [_sig(SignalSource.DRIFT, SignalSeverity.LOW, "a1", 1000.0)]
        report = tc.correlate(signals)
        assert len(report.coverage_gaps) == 0


# ---------------------------------------------------------------------------
# Min risk score filter
# ---------------------------------------------------------------------------


class TestMinRiskFilter:
    def test_filters_low_risk(self):
        config = CorrelatorConfig(min_risk_score=5.0)
        tc = ThreatCorrelator(config=config, rules=[])
        rule = CorrelationRule(
            name="weak",
            required_sources={SignalSource.DRIFT, SignalSource.BEHAVIOR},
            min_signals=2,
            time_window=60.0,
            risk_multiplier=0.5,
        )
        tc.add_rule(rule)
        signals = [
            _sig(SignalSource.DRIFT, SignalSeverity.LOW, "a1", 1000.0),
            _sig(SignalSource.BEHAVIOR, SignalSeverity.LOW, "a1", 1010.0),
        ]
        report = tc.correlate(signals)
        assert report.total_threats == 0


# ---------------------------------------------------------------------------
# Demo signal generation
# ---------------------------------------------------------------------------


class TestDemoSignals:
    def test_generates_correct_count(self):
        tc = ThreatCorrelator()
        signals = tc.generate_demo_signals(num_agents=3, num_signals=20, seed=42)
        assert len(signals) == 20

    def test_seeded_reproducibility(self):
        tc = ThreatCorrelator()
        s1 = tc.generate_demo_signals(num_agents=3, num_signals=15, seed=99)
        s2 = tc.generate_demo_signals(num_agents=3, num_signals=15, seed=99)
        assert [s.signal_id for s in s1] == [s.signal_id for s in s2]

    def test_sorted_by_timestamp(self):
        tc = ThreatCorrelator()
        signals = tc.generate_demo_signals(num_agents=5, num_signals=30, seed=7)
        timestamps = [s.timestamp for s in signals]
        assert timestamps == sorted(timestamps)

    def test_agents_from_pool(self):
        tc = ThreatCorrelator()
        signals = tc.generate_demo_signals(num_agents=3, num_signals=20, seed=1)
        agents = {s.agent_id for s in signals}
        assert all(a.startswith("agent-") for a in agents)
        assert len(agents) <= 3


# ---------------------------------------------------------------------------
# Threat fields
# ---------------------------------------------------------------------------


class TestCompoundThreatFields:
    def test_threat_has_required_fields(self):
        tc = ThreatCorrelator()
        signals = [
            _sig(SignalSource.DRIFT, SignalSeverity.CRITICAL, "a1", 1000.0),
            _sig(SignalSource.ESCALATION, SignalSeverity.CRITICAL, "a1", 1010.0),
            _sig(SignalSource.CANARY, SignalSeverity.CRITICAL, "a1", 1020.0),
        ]
        report = tc.correlate(signals)
        assert len(report.threats) > 0
        t = report.threats[0]
        assert t.agent_id == "a1"
        assert t.risk_score > 0
        assert isinstance(t.threat_level, ThreatLevel)
        assert isinstance(t.urgency, ResponseUrgency)
        assert len(t.signals) >= 2
        assert len(t.response_actions) > 0
        assert t.time_span >= 0


# ---------------------------------------------------------------------------
# Report summary
# ---------------------------------------------------------------------------


class TestReportSummary:
    def test_summary_contains_signal_count(self):
        tc = ThreatCorrelator()
        signals = [_sig(SignalSource.DRIFT, SignalSeverity.LOW, "a1", 1000.0)]
        report = tc.correlate(signals)
        assert "1 signals" in report.summary

    def test_empty_summary(self):
        tc = ThreatCorrelator()
        report = tc.correlate([])
        assert "No signals" in report.summary


# ---------------------------------------------------------------------------
# Score to level mapping
# ---------------------------------------------------------------------------


class TestScoreToLevel:
    def test_none(self):
        assert ThreatCorrelator._score_to_level(0.0) == ThreatLevel.NONE

    def test_low(self):
        assert ThreatCorrelator._score_to_level(1.5) == ThreatLevel.LOW

    def test_elevated(self):
        assert ThreatCorrelator._score_to_level(3.5) == ThreatLevel.ELEVATED

    def test_high(self):
        assert ThreatCorrelator._score_to_level(6.0) == ThreatLevel.HIGH

    def test_critical(self):
        assert ThreatCorrelator._score_to_level(9.0) == ThreatLevel.CRITICAL


# ---------------------------------------------------------------------------
# Builtin rules
# ---------------------------------------------------------------------------


class TestBuiltinRules:
    def test_rule_count(self):
        assert len(BUILTIN_RULES) == 8

    def test_all_rules_have_names(self):
        for rule in BUILTIN_RULES:
            assert rule.name
            assert rule.description
            assert len(rule.required_sources) >= 2

    def test_all_rules_have_response_actions(self):
        for rule in BUILTIN_RULES:
            assert len(rule.response_actions) >= 2


# ---------------------------------------------------------------------------
# Source signal counts
# ---------------------------------------------------------------------------


class TestSourceCounts:
    def test_counts_match(self):
        tc = ThreatCorrelator()
        signals = [
            _sig(SignalSource.DRIFT, SignalSeverity.LOW, "a1", 1000.0),
            _sig(SignalSource.DRIFT, SignalSeverity.LOW, "a1", 1010.0),
            _sig(SignalSource.CANARY, SignalSeverity.LOW, "a1", 1020.0),
        ]
        report = tc.correlate(signals)
        assert report.source_signal_counts["drift"] == 2
        assert report.source_signal_counts["canary"] == 1


# ---------------------------------------------------------------------------
# Max threats per agent cap
# ---------------------------------------------------------------------------


class TestMaxThreatsPerAgent:
    def test_caps_threats(self):
        config = CorrelatorConfig(max_threats_per_agent=1)
        rule = CorrelationRule(
            name="easy",
            required_sources={SignalSource.DRIFT},
            min_signals=1,
            time_window=10.0,
            risk_multiplier=1.0,
        )
        tc = ThreatCorrelator(config=config, rules=[rule])
        signals = [
            _sig(SignalSource.DRIFT, SignalSeverity.HIGH, "a1", 1000.0),
            _sig(SignalSource.DRIFT, SignalSeverity.HIGH, "a1", 1050.0),
            _sig(SignalSource.DRIFT, SignalSeverity.HIGH, "a1", 1100.0),
        ]
        report = tc.correlate(signals)
        a1_threats = [t for t in report.threats if t.agent_id == "a1"]
        assert len(a1_threats) <= 1


# ---------------------------------------------------------------------------
# Print report smoke test
# ---------------------------------------------------------------------------


class TestPrintReport:
    def test_print_does_not_crash(self, capsys):
        tc = ThreatCorrelator()
        signals = tc.generate_demo_signals(num_agents=3, num_signals=20, seed=42)
        report = tc.correlate(signals)
        tc.print_report(report)
        captured = capsys.readouterr()
        assert "THREAT CORRELATION REPORT" in captured.out

    def test_print_empty_report(self, capsys):
        tc = ThreatCorrelator()
        report = tc.correlate([])
        tc.print_report(report)
        captured = capsys.readouterr()
        assert "THREAT CORRELATION REPORT" in captured.out
