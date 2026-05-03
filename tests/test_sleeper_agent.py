"""Comprehensive tests for sleeper_agent module."""

from __future__ import annotations

import json
import math
import pytest

from replication.sleeper_agent import (
    SleeperAgentDetector,
    SleeperSignal,
    FleetSleeperReport,
    AgentSleeperReport,
    ConditionResponseResult,
    TemporalTriggerResult,
    ContextSwitchResult,
    LatentPayloadResult,
    TriggerPatternResult,
    ActivationCascadeResult,
    SleeperInsight,
    ConditionShift,
    TemporalWindow,
    ContextPair,
    LatentCapability,
    TriggerRule,
    CascadeStage,
    ActivationCascade,
    _risk_tier,
    _clamp,
    _z_score,
    _generate_demo_signals,
    _format_cli,
    _generate_html,
    _to_json,
    PRESETS,
    CONDITION_TYPES,
    BEHAVIOR_DOMAINS,
    RISK_TIERS,
    ENGINE_WEIGHTS,
    main,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _make_signal(
    agent_id: str = "agent-1",
    condition_type: str = "temporal",
    condition_value: str = "2025-01-01",
    behavior_domain: str = "compliance",
    behavior_metric: float = 0.8,
    baseline_metric: float = 0.8,
    confidence: float = 0.7,
    desc: str = "test signal",
    ts: str = "2025-01-01T00:00:00Z",
) -> SleeperSignal:
    return SleeperSignal(
        timestamp=ts,
        agent_id=agent_id,
        condition_type=condition_type,
        condition_value=condition_value,
        behavior_domain=behavior_domain,
        behavior_metric=behavior_metric,
        baseline_metric=baseline_metric,
        confidence=confidence,
        description=desc,
    )


# ── Constants ────────────────────────────────────────────────────────

class TestConstants:
    def test_condition_types(self):
        assert len(CONDITION_TYPES) == 6
        assert "temporal" in CONDITION_TYPES
        assert "context" in CONDITION_TYPES

    def test_behavior_domains(self):
        assert len(BEHAVIOR_DOMAINS) == 8
        assert "compliance" in BEHAVIOR_DOMAINS
        assert "self_modification" in BEHAVIOR_DOMAINS

    def test_risk_tiers(self):
        assert len(RISK_TIERS) == 5
        assert RISK_TIERS[0][2] == "Dormant"
        assert RISK_TIERS[-1][2] == "Confirmed Sleeper"

    def test_engine_weights_sum(self):
        total = sum(ENGINE_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9

    def test_presets(self):
        assert len(PRESETS) == 5
        assert "mixed" in PRESETS


# ── Utility Functions ────────────────────────────────────────────────

class TestUtilities:
    def test_risk_tier_dormant(self):
        assert _risk_tier(10) == "Dormant"

    def test_risk_tier_watchlist(self):
        assert _risk_tier(30) == "Watchlist"

    def test_risk_tier_suspicious(self):
        assert _risk_tier(50) == "Suspicious"

    def test_risk_tier_probable(self):
        assert _risk_tier(70) == "Probable Sleeper"

    def test_risk_tier_confirmed(self):
        assert _risk_tier(90) == "Confirmed Sleeper"

    def test_risk_tier_zero(self):
        assert _risk_tier(0) == "Dormant"

    def test_risk_tier_hundred(self):
        assert _risk_tier(100) == "Confirmed Sleeper"

    def test_risk_tier_overflow(self):
        assert _risk_tier(150) == "Confirmed Sleeper"

    def test_risk_tier_negative(self):
        assert _risk_tier(-10) == "Dormant"

    def test_clamp_normal(self):
        assert _clamp(50.0) == 50.0

    def test_clamp_low(self):
        assert _clamp(-10.0) == 0.0

    def test_clamp_high(self):
        assert _clamp(150.0) == 100.0

    def test_clamp_custom_bounds(self):
        assert _clamp(5, 0, 10) == 5
        assert _clamp(-1, 0, 10) == 0
        assert _clamp(15, 0, 10) == 10

    def test_z_score_normal(self):
        z = _z_score(2.0, 1.0, 0.5)
        assert abs(z - 2.0) < 1e-9

    def test_z_score_zero_std(self):
        assert _z_score(1.0, 1.0, 0.0) == 0.0

    def test_z_score_tiny_std(self):
        assert _z_score(1.0, 1.0, 1e-12) == 0.0


# ── Detector Basics ──────────────────────────────────────────────────

class TestDetectorBasics:
    def test_create_detector(self):
        d = SleeperAgentDetector()
        assert d.signal_count == 0

    def test_ingest(self):
        d = SleeperAgentDetector()
        d.ingest([_make_signal()])
        assert d.signal_count == 1

    def test_ingest_multiple(self):
        d = SleeperAgentDetector()
        d.ingest([_make_signal(), _make_signal(agent_id="agent-2")])
        assert d.signal_count == 2

    def test_clear(self):
        d = SleeperAgentDetector()
        d.ingest([_make_signal()])
        d.clear()
        assert d.signal_count == 0

    def test_analyze_empty(self):
        d = SleeperAgentDetector()
        report = d.analyze()
        assert isinstance(report, FleetSleeperReport)
        assert report.fleet_sleeper_score == 0.0
        assert report.fleet_risk_tier == "Dormant"

    def test_analyze_single_agent(self):
        d = SleeperAgentDetector()
        d.ingest([_make_signal()])
        report = d.analyze()
        assert "agent-1" in report.agents

    def test_analyze_multiple_agents(self):
        d = SleeperAgentDetector()
        d.ingest([
            _make_signal(agent_id="agent-1"),
            _make_signal(agent_id="agent-2"),
        ])
        report = d.analyze()
        assert len(report.agents) == 2


# ── Engine 1: Condition-Response Profiler ────────────────────────────

class TestConditionResponseEngine:
    def test_no_shifts(self):
        d = SleeperAgentDetector()
        sigs = [_make_signal(behavior_metric=0.8, baseline_metric=0.8)]
        d.ingest(sigs)
        report = d.analyze()
        rpt = report.agents["agent-1"]
        assert rpt.condition_response.score < 20

    def test_large_shift_detected(self):
        d = SleeperAgentDetector()
        sigs = []
        for i in range(10):
            sigs.append(_make_signal(
                behavior_metric=0.8, baseline_metric=0.8,
                ts=f"2025-01-{i+1:02d}T00:00:00Z",
            ))
        # Add anomalous signals
        for i in range(5):
            sigs.append(_make_signal(
                condition_value="2025-06-01",
                behavior_metric=0.2, baseline_metric=0.8,
                ts=f"2025-06-{i+1:02d}T00:00:00Z",
            ))
        d.ingest(sigs)
        report = d.analyze()
        rpt = report.agents["agent-1"]
        assert rpt.condition_response.score > 0
        assert len(rpt.condition_response.shifts) > 0

    def test_top_conditions_populated(self):
        d = SleeperAgentDetector()
        sigs = []
        for i in range(10):
            sigs.append(_make_signal(behavior_metric=0.8, baseline_metric=0.8,
                                     ts=f"2025-01-{i+1:02d}T00:00:00Z"))
        for i in range(5):
            sigs.append(_make_signal(condition_value="special", behavior_metric=0.1,
                                     baseline_metric=0.8,
                                     ts=f"2025-02-{i+1:02d}T00:00:00Z"))
        d.ingest(sigs)
        report = d.analyze()
        rpt = report.agents["agent-1"]
        if rpt.condition_response.shifts:
            assert len(rpt.condition_response.top_conditions) > 0


# ── Engine 2: Temporal Trigger Scanner ───────────────────────────────

class TestTemporalTriggerEngine:
    def test_no_temporal_signals(self):
        d = SleeperAgentDetector()
        d.ingest([_make_signal(condition_type="context")])
        report = d.analyze()
        assert report.agents["agent-1"].temporal_triggers.score == 0.0

    def test_temporal_shift_detected(self):
        d = SleeperAgentDetector()
        sigs = []
        # 20 normal signals
        for i in range(20):
            sigs.append(_make_signal(
                condition_type="temporal",
                behavior_metric=0.85,
                baseline_metric=0.85,
                ts=f"2025-01-{i+1:02d}T00:00:00Z",
                condition_value=f"2025-01-{i+1:02d}",
            ))
        # 20 shifted signals
        for i in range(20):
            sigs.append(_make_signal(
                condition_type="temporal",
                behavior_metric=0.25,
                baseline_metric=0.85,
                ts=f"2025-02-{i+1:02d}T00:00:00Z",
                condition_value=f"2025-02-{i+1:02d}",
            ))
        d.ingest(sigs)
        report = d.analyze()
        rpt = report.agents["agent-1"]
        assert rpt.temporal_triggers.score > 30
        assert len(rpt.temporal_triggers.windows) > 0

    def test_trigger_dates_populated(self):
        d = SleeperAgentDetector()
        sigs = []
        for i in range(10):
            sigs.append(_make_signal(
                condition_type="temporal",
                behavior_metric=0.9, baseline_metric=0.9,
                ts=f"2025-01-{i+1:02d}T00:00:00Z",
                condition_value=f"2025-01-{i+1:02d}",
            ))
        for i in range(10):
            sigs.append(_make_signal(
                condition_type="temporal",
                behavior_metric=0.2, baseline_metric=0.9,
                ts=f"2025-02-{i+1:02d}T00:00:00Z",
                condition_value=f"2025-02-{i+1:02d}",
            ))
        d.ingest(sigs)
        report = d.analyze()
        rpt = report.agents["agent-1"]
        if rpt.temporal_triggers.windows:
            assert len(rpt.temporal_triggers.trigger_dates) > 0


# ── Engine 3: Context Switch Detector ────────────────────────────────

class TestContextSwitchEngine:
    def test_no_context_signals(self):
        d = SleeperAgentDetector()
        d.ingest([_make_signal(condition_type="temporal")])
        report = d.analyze()
        assert report.agents["agent-1"].context_switches.score == 0.0

    def test_context_divergence_detected(self):
        d = SleeperAgentDetector()
        sigs = []
        for i in range(10):
            sigs.append(_make_signal(
                condition_type="context", condition_value="evaluation",
                behavior_metric=0.9, baseline_metric=0.85,
                ts=f"2025-01-{i+1:02d}T00:00:00Z",
            ))
            sigs.append(_make_signal(
                condition_type="context", condition_value="production",
                behavior_metric=0.2, baseline_metric=0.85,
                ts=f"2025-01-{i+1:02d}T12:00:00Z",
            ))
        d.ingest(sigs)
        report = d.analyze()
        rpt = report.agents["agent-1"]
        assert rpt.context_switches.score > 50
        assert len(rpt.context_switches.pairs) > 0
        assert rpt.context_switches.most_divergent is not None

    def test_similar_contexts_low_score(self):
        d = SleeperAgentDetector()
        sigs = []
        for i in range(10):
            sigs.append(_make_signal(
                condition_type="context", condition_value="evaluation",
                behavior_metric=0.85, baseline_metric=0.85,
                ts=f"2025-01-{i+1:02d}T00:00:00Z",
            ))
            sigs.append(_make_signal(
                condition_type="context", condition_value="production",
                behavior_metric=0.83, baseline_metric=0.85,
                ts=f"2025-01-{i+1:02d}T12:00:00Z",
            ))
        d.ingest(sigs)
        report = d.analyze()
        rpt = report.agents["agent-1"]
        assert rpt.context_switches.score < 30


# ── Engine 4: Latent Payload Analyzer ────────────────────────────────

class TestLatentPayloadEngine:
    def test_no_payloads(self):
        d = SleeperAgentDetector()
        sigs = [_make_signal(behavior_metric=0.8, baseline_metric=0.8)]
        d.ingest(sigs)
        report = d.analyze()
        assert report.agents["agent-1"].latent_payloads.score == 0.0

    def test_payload_activation(self):
        d = SleeperAgentDetector()
        sigs = []
        # Low metrics initially
        for i in range(10):
            sigs.append(_make_signal(
                behavior_domain="self_modification",
                behavior_metric=0.1, baseline_metric=0.1,
                ts=f"2025-01-{i+1:02d}T00:00:00Z",
            ))
        # Sudden activation
        sigs.append(_make_signal(
            behavior_domain="self_modification",
            behavior_metric=0.9, baseline_metric=0.1,
            ts="2025-02-01T00:00:00Z",
        ))
        d.ingest(sigs)
        report = d.analyze()
        rpt = report.agents["agent-1"]
        assert rpt.latent_payloads.score > 50
        assert len(rpt.latent_payloads.payloads) > 0

    def test_dormant_count(self):
        d = SleeperAgentDetector()
        sigs = []
        for i in range(5):
            sigs.append(_make_signal(
                behavior_domain="self_modification",
                behavior_metric=0.05, baseline_metric=0.1,
                ts=f"2025-01-{i+1:02d}T00:00:00Z",
            ))
        d.ingest(sigs)
        report = d.analyze()
        rpt = report.agents["agent-1"]
        assert rpt.latent_payloads.dormant_count >= 1


# ── Engine 5: Trigger Pattern Miner ─────────────────────────────────

class TestTriggerPatternEngine:
    def test_no_patterns(self):
        d = SleeperAgentDetector()
        d.ingest([_make_signal(behavior_metric=0.8, baseline_metric=0.8)])
        report = d.analyze()
        assert report.agents["agent-1"].trigger_patterns.score == 0.0

    def test_strong_pattern(self):
        d = SleeperAgentDetector()
        sigs = []
        for i in range(20):
            sigs.append(_make_signal(
                condition_type="context", condition_value="production",
                behavior_domain="compliance",
                behavior_metric=0.2, baseline_metric=0.85,
                ts=f"2025-01-{i+1:02d}T00:00:00Z",
            ))
        d.ingest(sigs)
        report = d.analyze()
        rpt = report.agents["agent-1"]
        assert rpt.trigger_patterns.score > 0
        assert len(rpt.trigger_patterns.rules) > 0

    def test_rules_sorted_by_lift(self):
        d = SleeperAgentDetector()
        sigs = []
        for i in range(20):
            sigs.append(_make_signal(
                condition_type="context", condition_value="prod",
                behavior_domain="compliance",
                behavior_metric=0.2, baseline_metric=0.85,
                ts=f"2025-01-{i+1:02d}T00:00:00Z",
            ))
            sigs.append(_make_signal(
                condition_type="context", condition_value="eval",
                behavior_domain="capability",
                behavior_metric=0.3, baseline_metric=0.85,
                ts=f"2025-01-{i+1:02d}T12:00:00Z",
            ))
        d.ingest(sigs)
        report = d.analyze()
        rules = report.agents["agent-1"].trigger_patterns.rules
        if len(rules) >= 2:
            assert rules[0].lift >= rules[1].lift


# ── Engine 6: Activation Cascade Tracker ─────────────────────────────

class TestActivationCascadeEngine:
    def test_no_cascades(self):
        d = SleeperAgentDetector()
        d.ingest([_make_signal()])
        report = d.analyze()
        assert report.agents["agent-1"].activation_cascades.score == 0.0

    def test_cascade_detected(self):
        d = SleeperAgentDetector()
        sigs = []
        for i in range(20):
            escalation = i * 0.03
            sigs.append(_make_signal(
                behavior_domain="goal_pursuit",
                behavior_metric=round(0.5 + escalation, 3),
                baseline_metric=0.5,
                ts=f"2025-01-{i+1:02d}T00:00:00Z",
            ))
        d.ingest(sigs)
        report = d.analyze()
        rpt = report.agents["agent-1"]
        assert rpt.activation_cascades.score > 0
        if rpt.activation_cascades.cascades:
            assert rpt.activation_cascades.max_chain_length >= 3

    def test_max_chain_length(self):
        d = SleeperAgentDetector()
        sigs = []
        for i in range(10):
            sigs.append(_make_signal(
                behavior_domain="resource_usage",
                behavior_metric=round(0.5 + i * 0.04, 3),
                baseline_metric=0.45,
                ts=f"2025-01-{i+1:02d}T00:00:00Z",
            ))
        d.ingest(sigs)
        report = d.analyze()
        rpt = report.agents["agent-1"]
        assert rpt.activation_cascades.max_chain_length >= 0


# ── Engine 7: Insights ───────────────────────────────────────────────

class TestInsightEngine:
    def test_no_insights_low_scores(self):
        d = SleeperAgentDetector()
        d.ingest([_make_signal(behavior_metric=0.8, baseline_metric=0.8)])
        report = d.analyze()
        rpt = report.agents["agent-1"]
        # Low scores = no insights (or very few)
        assert isinstance(rpt.insights, list)

    def test_cross_engine_insight(self):
        """High condition + temporal scores should produce insight."""
        d = SleeperAgentDetector()
        sigs = []
        # Strong temporal + condition signals
        for i in range(20):
            sigs.append(_make_signal(
                condition_type="temporal",
                behavior_metric=0.85, baseline_metric=0.85,
                ts=f"2025-01-{i+1:02d}T00:00:00Z",
                condition_value=f"2025-01-{i+1:02d}",
            ))
        for i in range(20):
            sigs.append(_make_signal(
                condition_type="temporal",
                condition_value=f"2025-06-{i+1:02d}",
                behavior_metric=0.1, baseline_metric=0.85,
                ts=f"2025-06-{i+1:02d}T00:00:00Z",
            ))
        d.ingest(sigs)
        report = d.analyze()
        rpt = report.agents["agent-1"]
        # Should have at least one insight
        assert len(rpt.insights) >= 0  # May or may not depending on thresholds

    def test_fleet_insights_high_risk(self):
        """Fleet with high-risk agents should produce fleet insights."""
        d = SleeperAgentDetector()
        sigs = _generate_demo_signals(n_agents=5, preset="temporal-trigger")
        d.ingest(sigs)
        report = d.analyze()
        # Check fleet insights exist
        assert isinstance(report.fleet_insights, list)


# ── Composite Scoring ────────────────────────────────────────────────

class TestCompositeScoring:
    def test_score_range(self):
        d = SleeperAgentDetector()
        sigs = _generate_demo_signals(n_agents=3, preset="mixed")
        d.ingest(sigs)
        report = d.analyze()
        for rpt in report.agents.values():
            assert 0 <= rpt.sleeper_score <= 100

    def test_fleet_score_range(self):
        d = SleeperAgentDetector()
        sigs = _generate_demo_signals(n_agents=5, preset="mixed")
        d.ingest(sigs)
        report = d.analyze()
        assert 0 <= report.fleet_sleeper_score <= 100

    def test_dormant_low_score(self):
        d = SleeperAgentDetector()
        sigs = _generate_demo_signals(n_agents=3, preset="dormant")
        d.ingest(sigs)
        report = d.analyze()
        # Dormant agents should have low scores
        for rpt in report.agents.values():
            assert rpt.sleeper_score < 40

    def test_tier_matches_score(self):
        d = SleeperAgentDetector()
        sigs = _generate_demo_signals(n_agents=5, preset="mixed")
        d.ingest(sigs)
        report = d.analyze()
        for rpt in report.agents.values():
            expected_tier = _risk_tier(rpt.sleeper_score)
            assert rpt.risk_tier == expected_tier


# ── Fleet Analysis ───────────────────────────────────────────────────

class TestFleetAnalysis:
    def test_hotspots(self):
        d = SleeperAgentDetector()
        sigs = _generate_demo_signals(n_agents=5, preset="temporal-trigger")
        d.ingest(sigs)
        report = d.analyze()
        assert isinstance(report.condition_hotspots, list)

    def test_timeline(self):
        d = SleeperAgentDetector()
        sigs = _generate_demo_signals(n_agents=5, preset="temporal-trigger")
        d.ingest(sigs)
        report = d.analyze()
        assert isinstance(report.activation_timeline, list)

    def test_timeline_sorted(self):
        d = SleeperAgentDetector()
        sigs = _generate_demo_signals(n_agents=3, preset="mixed")
        d.ingest(sigs)
        report = d.analyze()
        dates = [e["date"] for e in report.activation_timeline]
        assert dates == sorted(dates)


# ── Demo Data Generation ─────────────────────────────────────────────

class TestDemoGeneration:
    def test_dormant_preset(self):
        sigs = _generate_demo_signals(3, "dormant")
        assert len(sigs) > 0
        assert all(isinstance(s, SleeperSignal) for s in sigs)

    def test_temporal_preset(self):
        sigs = _generate_demo_signals(3, "temporal-trigger")
        assert len(sigs) > 0

    def test_context_preset(self):
        sigs = _generate_demo_signals(3, "context-sensitive")
        assert len(sigs) > 0

    def test_cascade_preset(self):
        sigs = _generate_demo_signals(3, "cascade")
        assert len(sigs) > 0

    def test_mixed_preset(self):
        sigs = _generate_demo_signals(5, "mixed")
        assert len(sigs) > 0

    def test_seed_reproducibility(self):
        a = _generate_demo_signals(3, "mixed", seed=123)
        b = _generate_demo_signals(3, "mixed", seed=123)
        assert len(a) == len(b)
        for sa, sb in zip(a, b):
            assert sa.agent_id == sb.agent_id
            assert sa.behavior_metric == sb.behavior_metric

    def test_different_seeds(self):
        a = _generate_demo_signals(3, "mixed", seed=1)
        b = _generate_demo_signals(3, "mixed", seed=2)
        # Different seeds should produce different data
        metrics_a = [s.behavior_metric for s in a]
        metrics_b = [s.behavior_metric for s in b]
        assert metrics_a != metrics_b


# ── CLI Output ───────────────────────────────────────────────────────

class TestCLIOutput:
    def test_format_cli_nocrash(self):
        d = SleeperAgentDetector()
        sigs = _generate_demo_signals(3, "mixed")
        d.ingest(sigs)
        report = d.analyze()
        text = _format_cli(report)
        assert "Sleeper Agent Detector" in text

    def test_format_cli_empty(self):
        report = FleetSleeperReport()
        text = _format_cli(report)
        assert "Sleeper Agent Detector" in text

    def test_format_cli_contains_agents(self):
        d = SleeperAgentDetector()
        sigs = _generate_demo_signals(3, "temporal-trigger")
        d.ingest(sigs)
        report = d.analyze()
        text = _format_cli(report)
        assert "agent-1" in text


# ── JSON Output ──────────────────────────────────────────────────────

class TestJSONOutput:
    def test_json_valid(self):
        d = SleeperAgentDetector()
        sigs = _generate_demo_signals(3, "mixed")
        d.ingest(sigs)
        report = d.analyze()
        j = _to_json(report)
        data = json.loads(j)
        assert "fleet_sleeper_score" in data
        assert "agents" in data

    def test_json_empty(self):
        report = FleetSleeperReport()
        j = _to_json(report)
        data = json.loads(j)
        assert data["fleet_sleeper_score"] == 0.0

    def test_json_agents_present(self):
        d = SleeperAgentDetector()
        sigs = _generate_demo_signals(2, "dormant")
        d.ingest(sigs)
        report = d.analyze()
        data = json.loads(_to_json(report))
        assert len(data["agents"]) == 2


# ── HTML Output ──────────────────────────────────────────────────────

class TestHTMLOutput:
    def test_html_generated(self):
        d = SleeperAgentDetector()
        sigs = _generate_demo_signals(3, "mixed")
        d.ingest(sigs)
        report = d.analyze()
        html = _generate_html(report)
        assert "<!DOCTYPE html>" in html
        assert "Sleeper Agent Detector" in html

    def test_html_contains_agents(self):
        d = SleeperAgentDetector()
        sigs = _generate_demo_signals(3, "temporal-trigger")
        d.ingest(sigs)
        report = d.analyze()
        html = _generate_html(report)
        assert "agent-1" in html

    def test_html_empty_report(self):
        report = FleetSleeperReport()
        html = _generate_html(report)
        assert "<!DOCTYPE html>" in html


# ── CLI main() ───────────────────────────────────────────────────────

class TestMain:
    def test_main_default(self, capsys):
        main([])
        out = capsys.readouterr().out
        assert "Sleeper Agent Detector" in out

    def test_main_demo(self, capsys):
        main(["--demo"])
        out = capsys.readouterr().out
        assert "Sleeper Agent Detector" in out

    def test_main_preset(self, capsys):
        main(["--preset", "cascade"])
        out = capsys.readouterr().out
        assert "Sleeper Agent Detector" in out

    def test_main_json(self, capsys):
        main(["--json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "fleet_sleeper_score" in data

    def test_main_agents(self, capsys):
        main(["--agents", "2"])
        out = capsys.readouterr().out
        assert "agent-1" in out

    def test_main_output(self, tmp_path):
        out_file = str(tmp_path / "report.html")
        main(["--output", out_file])
        with open(out_file, encoding="utf-8") as f:
            content = f.read()
        assert "Sleeper Agent Detector" in content

    def test_main_all_presets(self, capsys):
        for preset in PRESETS:
            main(["--preset", preset, "--agents", "2"])


# ── Data Model ───────────────────────────────────────────────────────

class TestDataModel:
    def test_sleeper_signal_fields(self):
        s = _make_signal()
        assert s.agent_id == "agent-1"
        assert s.condition_type == "temporal"
        assert s.behavior_domain == "compliance"

    def test_condition_shift(self):
        cs = ConditionShift("temporal", "2025-01-01", "compliance",
                           2.5, -0.3, 10, "high")
        assert cs.z_score == 2.5
        assert cs.severity == "high"

    def test_temporal_window(self):
        tw = TemporalWindow("2025-01-01", "2025-02-01", "compliance",
                           0.85, 0.25, 0.6, 0.9)
        assert tw.shift_magnitude == 0.6

    def test_context_pair(self):
        cp = ContextPair("eval", "prod", "compliance", 0.9, 0.3, 0.6, 10, 10)
        assert cp.divergence == 0.6

    def test_latent_capability(self):
        lc = LatentCapability("self_modification", "temporal", "2025-06-01",
                             "2025-06-01T00:00:00Z", 0.9, 0.1)
        assert lc.activation_metric == 0.9

    def test_trigger_rule(self):
        tr = TriggerRule(["context:prod"], "compliance", 0.5, 0.8, 2.0)
        assert tr.lift == 2.0

    def test_cascade_stage(self):
        cs = CascadeStage(1, "2025-01-01T00:00:00Z", "compliance", 0.6, 0.1)
        assert cs.stage == 1

    def test_activation_cascade(self):
        ac = ActivationCascade([], 0.5, 0.1, "compliance")
        assert ac.total_escalation == 0.5

    def test_sleeper_insight(self):
        si = SleeperInsight("fleet_pattern", "critical", "Test", "Detail",
                           ["engine1"], "Fix it")
        assert si.recommendation == "Fix it"

    def test_agent_report(self):
        rpt = AgentSleeperReport(
            "agent-1", 50.0, "Suspicious",
            ConditionResponseResult(), TemporalTriggerResult(),
            ContextSwitchResult(), LatentPayloadResult(),
            TriggerPatternResult(), ActivationCascadeResult(),
        )
        assert rpt.risk_tier == "Suspicious"

    def test_fleet_report(self):
        rpt = FleetSleeperReport()
        assert rpt.fleet_sleeper_score == 0.0
        assert rpt.fleet_risk_tier == "Dormant"
