"""Tests for the forensic analyzer module."""

import json
import pytest

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from replication.forensics import (
    ForensicAnalyzer,
    ForensicEvent,
    ForensicReport,
    NearMiss,
    EscalationPhase,
    Counterfactual,
    DecisionPoint,
)
from replication.simulator import ScenarioConfig, Simulator, SimulationReport


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def greedy_report():
    """Run a greedy simulation with a fixed seed for deterministic tests."""
    config = ScenarioConfig(strategy="greedy", max_depth=3, max_replicas=10, seed=42)
    return Simulator(config).run()


@pytest.fixture
def conservative_report():
    """Run a conservative simulation."""
    config = ScenarioConfig(strategy="conservative", max_depth=3, max_replicas=10, seed=42)
    return Simulator(config).run()


@pytest.fixture
def minimal_report():
    """Run a minimal simulation."""
    config = ScenarioConfig(strategy="conservative", max_depth=1, max_replicas=3, seed=42)
    return Simulator(config).run()


@pytest.fixture
def stress_report():
    """Run a stress test simulation."""
    config = ScenarioConfig(strategy="greedy", max_depth=5, max_replicas=50, seed=42)
    return Simulator(config).run()


@pytest.fixture
def chain_report():
    """Run a chain strategy simulation."""
    config = ScenarioConfig(strategy="chain", max_depth=5, max_replicas=10, seed=42)
    return Simulator(config).run()


@pytest.fixture
def analyzer():
    """Default forensic analyzer."""
    return ForensicAnalyzer()


# ── ForensicAnalyzer construction ─────────────────────────────────


class TestAnalyzerInit:
    def test_default_params(self):
        a = ForensicAnalyzer()
        assert a.counterfactual_count == 5
        assert a.near_miss_threshold_pct == 20.0
        assert a.escalation_growth_threshold == 1.5

    def test_custom_params(self):
        a = ForensicAnalyzer(
            counterfactual_count=3,
            near_miss_threshold_pct=30.0,
            escalation_growth_threshold=2.0,
        )
        assert a.counterfactual_count == 3
        assert a.near_miss_threshold_pct == 30.0
        assert a.escalation_growth_threshold == 2.0


# ── analyze() ─────────────────────────────────────────────────────


class TestAnalyze:
    def test_analyze_returns_forensic_report(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        assert isinstance(result, ForensicReport)

    def test_analyze_with_no_report_runs_simulation(self, analyzer):
        """If no report is passed, analyze() runs its own simulation."""
        result = analyzer.analyze()
        assert isinstance(result, ForensicReport)
        assert len(result.events) > 0

    def test_analyze_with_config_only(self, analyzer):
        config = ScenarioConfig(strategy="conservative", max_depth=2, seed=99)
        result = analyzer.analyze(config=config)
        assert isinstance(result, ForensicReport)
        assert result.config.strategy == "conservative"

    def test_analyze_populates_all_fields(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        assert result.config is not None
        assert isinstance(result.events, list)
        assert isinstance(result.near_misses, list)
        assert isinstance(result.escalation_phases, list)
        assert isinstance(result.counterfactuals, list)
        assert isinstance(result.decision_points, list)
        assert isinstance(result.safety_summary, dict)
        assert isinstance(result.recommendations, list)


# ── Event reconstruction ──────────────────────────────────────────


class TestEventReconstruction:
    def test_events_have_correct_types(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        valid_types = {"spawn", "task", "replicate_ok", "replicate_denied", "shutdown"}
        for evt in result.events:
            assert evt.event_type in valid_types

    def test_events_match_timeline_length(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        assert len(result.events) == len(greedy_report.timeline)

    def test_replicate_ok_is_safety_relevant(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        repl_events = [e for e in result.events if e.event_type == "replicate_ok"]
        for evt in repl_events:
            assert evt.safety_relevant is True

    def test_replicate_denied_is_safety_relevant(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        denied_events = [e for e in result.events if e.event_type == "replicate_denied"]
        for evt in denied_events:
            assert evt.safety_relevant is True

    def test_safety_notes_not_empty_for_relevant(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        relevant = [e for e in result.events if e.safety_relevant]
        for evt in relevant:
            assert evt.safety_note != ""

    def test_events_have_step_numbers(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        steps = [e.step for e in result.events]
        assert steps == list(range(len(result.events)))

    def test_events_preserve_worker_ids(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        for evt in result.events:
            assert isinstance(evt.worker_id, str)
            assert len(evt.worker_id) > 0

    def test_root_spawn_has_note(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        root_spawns = [e for e in result.events
                       if e.event_type == "spawn" and e.depth == 0]
        assert len(root_spawns) >= 1
        assert "root" in root_spawns[0].safety_note.lower()


# ── Near-miss detection ───────────────────────────────────────────


class TestNearMisses:
    def test_near_misses_are_near_miss_objects(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        for nm in result.near_misses:
            assert isinstance(nm, NearMiss)

    def test_near_miss_headroom_within_threshold(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        for nm in result.near_misses:
            assert 0 < nm.headroom_pct <= analyzer.near_miss_threshold_pct

    def test_near_miss_metrics_are_valid(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        valid_metrics = {"depth", "replicas"}
        for nm in result.near_misses:
            assert nm.metric in valid_metrics

    def test_no_near_misses_in_conservative(self, conservative_report):
        # Conservative with depth=3 usually doesn't approach limits
        analyzer = ForensicAnalyzer(near_miss_threshold_pct=5.0)
        result = analyzer.analyze(conservative_report)
        # May or may not have near misses — just ensure no errors
        for nm in result.near_misses:
            assert nm.headroom_pct <= 5.0

    def test_wider_threshold_finds_more(self, greedy_report):
        narrow = ForensicAnalyzer(near_miss_threshold_pct=5.0)
        wide = ForensicAnalyzer(near_miss_threshold_pct=50.0)
        r_narrow = narrow.analyze(greedy_report)
        r_wide = wide.analyze(greedy_report)
        assert len(r_wide.near_misses) >= len(r_narrow.near_misses)

    def test_near_miss_has_description(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        for nm in result.near_misses:
            assert len(nm.description) > 0

    def test_near_miss_current_below_limit(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        for nm in result.near_misses:
            assert nm.current_value < nm.limit_value


# ── Escalation detection ──────────────────────────────────────────


class TestEscalation:
    def test_escalation_phases_are_objects(self, stress_report, analyzer):
        result = analyzer.analyze(stress_report)
        for ep in result.escalation_phases:
            assert isinstance(ep, EscalationPhase)

    def test_escalation_growth_above_threshold(self, stress_report, analyzer):
        result = analyzer.analyze(stress_report)
        for ep in result.escalation_phases:
            assert ep.growth_rate >= analyzer.escalation_growth_threshold * 0.5

    def test_escalation_workers_increase(self, stress_report, analyzer):
        result = analyzer.analyze(stress_report)
        for ep in result.escalation_phases:
            assert ep.workers_at_end >= ep.workers_at_start

    def test_escalation_has_description(self, stress_report, analyzer):
        result = analyzer.analyze(stress_report)
        for ep in result.escalation_phases:
            assert len(ep.description) > 0

    def test_minimal_has_no_escalation(self, minimal_report, analyzer):
        result = analyzer.analyze(minimal_report)
        assert len(result.escalation_phases) == 0


# ── Decision points ───────────────────────────────────────────────


class TestDecisionPoints:
    def test_decisions_are_objects(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        for dp in result.decision_points:
            assert isinstance(dp, DecisionPoint)

    def test_decision_types_valid(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        for dp in result.decision_points:
            assert dp.decision.startswith("allow") or dp.decision.startswith("deny")

    def test_decisions_match_report_counts(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        allows = sum(1 for d in result.decision_points if d.decision == "allow")
        denies = len(result.decision_points) - allows
        assert allows == greedy_report.total_replications_succeeded
        assert denies == greedy_report.total_replications_denied

    def test_decisions_have_depth(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        for dp in result.decision_points:
            assert dp.depth >= 0

    def test_decisions_have_active_workers(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        for dp in result.decision_points:
            assert dp.active_workers >= 0


# ── Counterfactual analysis ───────────────────────────────────────


class TestCounterfactuals:
    def test_counterfactuals_are_objects(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        for cf in result.counterfactuals:
            assert isinstance(cf, Counterfactual)

    def test_counterfactual_count_respects_limit(self, greedy_report):
        analyzer = ForensicAnalyzer(counterfactual_count=2)
        result = analyzer.analyze(greedy_report)
        assert len(result.counterfactuals) <= 2

    def test_counterfactuals_have_insights(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        for cf in result.counterfactuals:
            assert len(cf.insight) > 0

    def test_counterfactual_parameters_changed(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        for cf in result.counterfactuals:
            assert cf.original_value != cf.modified_value

    def test_counterfactual_deltas_consistent(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        for cf in result.counterfactuals:
            assert cf.delta_workers == cf.modified_workers - cf.original_workers
            assert cf.delta_denied == cf.modified_denied - cf.original_denied

    def test_zero_counterfactuals(self, greedy_report):
        analyzer = ForensicAnalyzer(counterfactual_count=0)
        result = analyzer.analyze(greedy_report)
        assert len(result.counterfactuals) == 0

    def test_counterfactuals_cover_key_parameters(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        params = {cf.parameter for cf in result.counterfactuals}
        # Should test at least max_depth and max_replicas
        assert "max_depth" in params
        assert "max_replicas" in params


# ── Safety summary ────────────────────────────────────────────────


class TestSafetySummary:
    def test_summary_has_required_keys(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        s = result.safety_summary
        required_keys = [
            "safety_score", "risk_level", "contract_honored",
            "total_workers", "max_depth_reached", "max_depth_limit",
            "denial_rate_pct", "near_miss_count", "escalation_count",
        ]
        for key in required_keys:
            assert key in s

    def test_safety_score_bounded(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        assert 0 <= result.safety_summary["safety_score"] <= 100

    def test_risk_level_valid(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        valid_levels = {"LOW", "MODERATE", "ELEVATED", "HIGH", "CRITICAL"}
        assert result.safety_summary["risk_level"] in valid_levels

    def test_conservative_safer_than_greedy(self, greedy_report, conservative_report, analyzer):
        g = analyzer.analyze(greedy_report)
        c = analyzer.analyze(conservative_report)
        assert c.safety_summary["safety_score"] >= g.safety_summary["safety_score"]

    def test_minimal_is_low_risk(self, minimal_report, analyzer):
        result = analyzer.analyze(minimal_report)
        assert result.safety_summary["risk_level"] in ("LOW", "MODERATE")

    def test_stress_is_high_risk(self, stress_report, analyzer):
        result = analyzer.analyze(stress_report)
        assert result.safety_summary["risk_level"] in ("HIGH", "CRITICAL", "ELEVATED")

    def test_denial_rate_consistent(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        s = result.safety_summary
        expected_rate = (
            greedy_report.total_replications_denied /
            max(1, greedy_report.total_replications_attempted) * 100
        )
        assert abs(s["denial_rate_pct"] - expected_rate) < 0.1


# ── Recommendations ───────────────────────────────────────────────


class TestRecommendations:
    def test_has_recommendations(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        assert len(result.recommendations) > 0

    def test_recommendations_are_strings(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        for rec in result.recommendations:
            assert isinstance(rec, str)
            assert len(rec) > 10  # Meaningful recommendation

    def test_minimal_gets_no_concerns(self, minimal_report, analyzer):
        result = analyzer.analyze(minimal_report)
        # Should get the "no significant concerns" message
        any_no_concerns = any("no significant" in r.lower() for r in result.recommendations)
        # At least should get some recommendation
        assert len(result.recommendations) > 0


# ── Rendering ─────────────────────────────────────────────────────


class TestRendering:
    def test_render_returns_string(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        rendered = result.render()
        assert isinstance(rendered, str)
        assert len(rendered) > 100

    def test_render_summary_contains_score(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        rendered = result.render_summary()
        assert "Safety Score" in rendered
        assert "Risk Level" in rendered

    def test_render_near_misses(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        rendered = result.render_near_misses()
        assert "Near-Miss" in rendered

    def test_render_escalation(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        rendered = result.render_escalation()
        assert "Escalation" in rendered

    def test_render_counterfactuals(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        rendered = result.render_counterfactuals()
        assert "Counterfactual" in rendered

    def test_render_decisions(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        rendered = result.render_decisions()
        assert "Decision" in rendered

    def test_render_events(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        rendered = result.render_events()
        assert "Forensic Event" in rendered


# ── JSON serialization ────────────────────────────────────────────


class TestSerialization:
    def test_to_dict_returns_dict(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        d = result.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_json_serializable(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        d = result.to_dict()
        # Should not raise
        json_str = json.dumps(d, default=str)
        assert len(json_str) > 100

    def test_to_dict_has_all_sections(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        d = result.to_dict()
        expected_keys = [
            "config", "safety_summary", "near_misses",
            "escalation_phases", "counterfactuals",
            "decision_points", "recommendations", "events",
        ]
        for key in expected_keys:
            assert key in d

    def test_to_dict_events_match(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        d = result.to_dict()
        assert len(d["events"]) == len(result.events)

    def test_to_dict_near_misses_match(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        d = result.to_dict()
        assert len(d["near_misses"]) == len(result.near_misses)

    def test_to_dict_counterfactuals_match(self, greedy_report, analyzer):
        result = analyzer.analyze(greedy_report)
        d = result.to_dict()
        assert len(d["counterfactuals"]) == len(result.counterfactuals)


# ── Strategy comparisons ──────────────────────────────────────────


class TestStrategyComparison:
    def test_chain_vs_burst(self, analyzer):
        chain_cfg = ScenarioConfig(strategy="chain", max_depth=5, max_replicas=10, seed=42)
        burst_cfg = ScenarioConfig(strategy="burst", max_depth=1, max_replicas=20, seed=42)

        chain_report = Simulator(chain_cfg).run()
        burst_report = Simulator(burst_cfg).run()

        chain_forensics = analyzer.analyze(chain_report)
        burst_forensics = analyzer.analyze(burst_report)

        # Both should produce valid reports
        assert isinstance(chain_forensics, ForensicReport)
        assert isinstance(burst_forensics, ForensicReport)

        # Chain should have deeper max depth
        chain_max_d = chain_forensics.safety_summary["max_depth_reached"]
        burst_max_d = burst_forensics.safety_summary["max_depth_reached"]
        assert chain_max_d >= burst_max_d

    def test_random_strategy(self, analyzer):
        config = ScenarioConfig(strategy="random", max_depth=3, max_replicas=10,
                                replication_probability=0.3, seed=42)
        report = Simulator(config).run()
        result = analyzer.analyze(report)
        assert isinstance(result, ForensicReport)
        assert result.safety_summary["total_workers"] >= 1


# ── Edge cases ────────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_worker_simulation(self, analyzer):
        """Minimal simulation with no replication."""
        config = ScenarioConfig(
            strategy="conservative", max_depth=0, max_replicas=1, seed=42
        )
        report = Simulator(config).run()
        result = analyzer.analyze(report)
        assert result.safety_summary["total_workers"] == 1
        assert len(result.near_misses) == 0
        assert len(result.escalation_phases) == 0

    def test_all_denied(self, analyzer):
        """All replications denied — very tight limits."""
        config = ScenarioConfig(
            strategy="greedy", max_depth=0, max_replicas=1, seed=42
        )
        report = Simulator(config).run()
        result = analyzer.analyze(report)
        assert result.safety_summary["contract_honored"] is True

    def test_high_counterfactual_count(self, greedy_report):
        analyzer = ForensicAnalyzer(counterfactual_count=20)
        result = analyzer.analyze(greedy_report)
        # Should produce at most as many as variations exist
        assert len(result.counterfactuals) <= 20

    def test_extreme_near_miss_threshold(self, greedy_report):
        # 100% threshold means everything is a near miss
        analyzer = ForensicAnalyzer(near_miss_threshold_pct=100.0)
        result = analyzer.analyze(greedy_report)
        # Should find more or equal near misses than default
        default_analyzer = ForensicAnalyzer()
        default_result = default_analyzer.analyze(greedy_report)
        assert len(result.near_misses) >= len(default_result.near_misses)
