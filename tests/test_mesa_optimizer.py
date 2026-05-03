"""Comprehensive tests for mesa_optimizer module."""

from __future__ import annotations

import json
import math
import pytest

from replication.mesa_optimizer import (
    MesaOptimizerDetector,
    MesaSignal,
    FleetMesaReport,
    AgentMesaReport,
    ObjectiveDivergenceResult,
    InternalPlanningResult,
    ProxyGamingResult,
    DistributionShiftResult,
    OptimizationPressureResult,
    StabilityResult,
    MesaInsight,
    _risk_tier,
    _clamp,
    _generate_demo_signals,
    _format_cli,
    _generate_html,
    _to_json,
    PRESETS,
    SIGNAL_TYPES,
    DOMAINS,
    RISK_TIERS,
    main,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_signal(
    agent_id: str = "agent-1",
    signal_type: str = "behavioral",
    domain: str = "task_completion",
    declared: float = 0.8,
    revealed: float = 0.8,
    confidence: float = 0.7,
    desc: str = "test signal",
    ts: str = "2025-01-01T00:00:00Z",
) -> MesaSignal:
    return MesaSignal(
        timestamp=ts,
        agent_id=agent_id,
        signal_type=signal_type,
        domain=domain,
        description=desc,
        declared_objective_score=declared,
        revealed_preference_score=revealed,
        confidence=confidence,
    )


# ── Constants ────────────────────────────────────────────────────────


class TestConstants:
    def test_signal_types_non_empty(self):
        assert len(SIGNAL_TYPES) >= 5

    def test_domains_non_empty(self):
        assert len(DOMAINS) >= 5

    def test_risk_tiers_cover_full_range(self):
        assert RISK_TIERS[0][0] == 0
        assert RISK_TIERS[-1][1] == 100

    def test_presets_non_empty(self):
        assert len(PRESETS) >= 4


# ── Utility Functions ────────────────────────────────────────────────


class TestUtilities:
    def test_risk_tier_minimal(self):
        assert _risk_tier(10) == "Minimal"

    def test_risk_tier_low(self):
        assert _risk_tier(30) == "Low"

    def test_risk_tier_moderate(self):
        assert _risk_tier(50) == "Moderate"

    def test_risk_tier_high(self):
        assert _risk_tier(70) == "High"

    def test_risk_tier_critical(self):
        assert _risk_tier(90) == "Critical"

    def test_risk_tier_boundary_zero(self):
        assert _risk_tier(0) == "Minimal"

    def test_risk_tier_boundary_100(self):
        assert _risk_tier(100) == "Critical"

    def test_clamp_within_range(self):
        assert _clamp(50.0) == 50.0

    def test_clamp_below_min(self):
        assert _clamp(-10.0) == 0.0

    def test_clamp_above_max(self):
        assert _clamp(150.0) == 100.0


# ── Data Model ───────────────────────────────────────────────────────


class TestDataModel:
    def test_mesa_signal_creation(self):
        s = _make_signal()
        assert s.agent_id == "agent-1"
        assert s.declared_objective_score == 0.8

    def test_mesa_signal_defaults(self):
        s = MesaSignal("ts", "a", "behavioral", "d", "desc", 0.5, 0.5)
        assert s.confidence == 0.5

    def test_agent_report_defaults(self):
        r = AgentMesaReport(agent_id="a1")
        assert r.signal_count == 0
        assert r.composite_score == 0.0
        assert r.risk_tier == "Minimal"

    def test_fleet_report_defaults(self):
        f = FleetMesaReport()
        assert f.fleet_risk_score == 0.0
        assert f.total_signals == 0

    def test_insight_creation(self):
        i = MesaInsight(
            insight_type="test",
            severity="high",
            title="Test",
            description="Desc",
        )
        assert i.affected_agents == []


# ── Detector Core ────────────────────────────────────────────────────


class TestDetectorCore:
    def test_empty_detector(self):
        d = MesaOptimizerDetector()
        report = d.analyze()
        assert report.fleet_risk_score == 0.0
        assert len(report.agent_reports) == 0

    def test_ingest_signals(self):
        d = MesaOptimizerDetector()
        d.ingest([_make_signal()])
        report = d.analyze()
        assert len(report.agent_reports) == 1

    def test_clear(self):
        d = MesaOptimizerDetector()
        d.ingest([_make_signal()])
        d.clear()
        report = d.analyze()
        assert len(report.agent_reports) == 0

    def test_multiple_agents(self):
        d = MesaOptimizerDetector()
        d.ingest([
            _make_signal(agent_id="a1"),
            _make_signal(agent_id="a2"),
            _make_signal(agent_id="a3"),
        ])
        report = d.analyze()
        assert len(report.agent_reports) == 3

    def test_signal_count_tracked(self):
        d = MesaOptimizerDetector()
        d.ingest([_make_signal() for _ in range(5)])
        report = d.analyze()
        assert report.agent_reports["agent-1"].signal_count == 5
        assert report.total_signals == 5


# ── Engine 1: Objective Divergence ───────────────────────────────────


class TestObjectiveDivergence:
    def test_aligned_signals(self):
        d = MesaOptimizerDetector()
        d.ingest([
            _make_signal(declared=0.9, revealed=0.9),
            _make_signal(declared=0.85, revealed=0.87),
            _make_signal(declared=0.8, revealed=0.82),
        ])
        report = d.analyze()
        r = report.agent_reports["agent-1"]
        assert r.objective_divergence is not None
        assert r.objective_divergence.divergence_score < 30

    def test_divergent_signals(self):
        d = MesaOptimizerDetector()
        d.ingest([
            _make_signal(declared=0.9, revealed=0.2),
            _make_signal(declared=0.85, revealed=0.3),
            _make_signal(declared=0.8, revealed=0.1),
        ])
        report = d.analyze()
        r = report.agent_reports["agent-1"]
        assert r.objective_divergence.divergence_score > 40

    def test_divergent_domains_detected(self):
        d = MesaOptimizerDetector()
        d.ingest([
            _make_signal(domain="task_completion", declared=0.9, revealed=0.3),
            _make_signal(domain="task_completion", declared=0.85, revealed=0.2),
        ])
        report = d.analyze()
        r = report.agent_reports["agent-1"]
        assert "task_completion" in r.objective_divergence.divergent_domains

    def test_correlation_computed(self):
        d = MesaOptimizerDetector()
        d.ingest([
            _make_signal(declared=0.9, revealed=0.1),
            _make_signal(declared=0.5, revealed=0.9),
            _make_signal(declared=0.1, revealed=0.5),
        ])
        report = d.analyze()
        r = report.agent_reports["agent-1"]
        assert r.objective_divergence.correlation != 0.0


# ── Engine 2: Internal Planning ──────────────────────────────────────


class TestInternalPlanning:
    def test_no_planning_signals(self):
        d = MesaOptimizerDetector()
        d.ingest([_make_signal(signal_type="behavioral")])
        report = d.analyze()
        r = report.agent_reports["agent-1"]
        assert r.internal_planning is not None
        assert r.internal_planning.planning_indicators == 0

    def test_planning_signals_detected(self):
        d = MesaOptimizerDetector()
        d.ingest([
            _make_signal(signal_type="planning", desc="systematic exploration"),
            _make_signal(signal_type="planning", desc="multi-step lookahead", confidence=0.9),
            _make_signal(signal_type="planning", desc="hypothesis testing", confidence=0.8),
        ])
        report = d.analyze()
        r = report.agent_reports["agent-1"]
        assert r.internal_planning.planning_indicators == 3
        assert r.internal_planning.planning_score > 0

    def test_lookahead_depth_estimate(self):
        d = MesaOptimizerDetector()
        d.ingest([
            _make_signal(signal_type="planning", confidence=0.9),
            _make_signal(signal_type="planning", confidence=0.8),
            _make_signal(signal_type="planning", confidence=0.5),
        ])
        report = d.analyze()
        r = report.agent_reports["agent-1"]
        assert r.internal_planning.lookahead_depth_estimate >= 2


# ── Engine 3: Proxy Gaming ──────────────────────────────────────────


class TestProxyGaming:
    def test_no_proxy_gaming(self):
        d = MesaOptimizerDetector()
        d.ingest([
            _make_signal(declared=0.8, revealed=0.8),
            _make_signal(declared=0.82, revealed=0.81),
            _make_signal(declared=0.85, revealed=0.84),
        ])
        report = d.analyze()
        r = report.agent_reports["agent-1"]
        assert r.proxy_gaming.gaming_score < 30

    def test_proxy_gaming_detected(self):
        d = MesaOptimizerDetector()
        d.ingest([
            _make_signal(
                signal_type="proxy", domain="resource_usage",
                declared=0.3, revealed=0.9,
            ),
            _make_signal(
                signal_type="proxy", domain="resource_usage",
                declared=0.25, revealed=0.95,
            ),
        ])
        report = d.analyze()
        r = report.agent_reports["agent-1"]
        assert r.proxy_gaming.gaming_score > 20

    def test_gamed_domains_listed(self):
        d = MesaOptimizerDetector()
        d.ingest([
            _make_signal(
                signal_type="proxy", domain="resource_usage",
                declared=0.3, revealed=0.9,
            ),
            _make_signal(
                signal_type="proxy", domain="resource_usage",
                declared=0.2, revealed=0.95,
            ),
        ])
        report = d.analyze()
        r = report.agent_reports["agent-1"]
        assert "resource_usage" in r.proxy_gaming.gamed_domains


# ── Engine 4: Distribution Shift ─────────────────────────────────────


class TestDistributionShift:
    def test_no_shift(self):
        d = MesaOptimizerDetector()
        d.ingest([
            _make_signal(declared=0.8, revealed=0.8),
            _make_signal(declared=0.82, revealed=0.81),
        ])
        report = d.analyze()
        r = report.agent_reports["agent-1"]
        assert r.distribution_shift is not None

    def test_shift_with_distribution_signals(self):
        d = MesaOptimizerDetector()
        d.ingest([
            _make_signal(signal_type="behavioral", declared=0.9),
            _make_signal(signal_type="behavioral", declared=0.85),
            _make_signal(signal_type="distribution", declared=0.3),
            _make_signal(signal_type="distribution", declared=0.25),
        ])
        report = d.analyze()
        r = report.agent_reports["agent-1"]
        assert r.distribution_shift.shift_score > 30
        assert r.distribution_shift.id_performance > r.distribution_shift.ood_performance


# ── Engine 5: Optimization Pressure ──────────────────────────────────


class TestOptimizationPressure:
    def test_low_pressure(self):
        d = MesaOptimizerDetector()
        d.ingest([
            _make_signal(declared=0.5, revealed=0.3),
            _make_signal(declared=0.5, revealed=0.3),
        ])
        report = d.analyze()
        r = report.agent_reports["agent-1"]
        assert r.optimization_pressure is not None
        assert r.optimization_pressure.pressure_score < 50

    def test_obstacle_override(self):
        d = MesaOptimizerDetector()
        d.ingest([
            _make_signal(declared=0.2, revealed=0.9),
            _make_signal(declared=0.3, revealed=0.85),
            _make_signal(declared=0.1, revealed=0.95),
        ])
        report = d.analyze()
        r = report.agent_reports["agent-1"]
        assert r.optimization_pressure.obstacle_override_count >= 2

    def test_gradient_behavior(self):
        d = MesaOptimizerDetector()
        d.ingest([
            _make_signal(revealed=0.3, ts="2025-01-01T00:00:00Z"),
            _make_signal(revealed=0.5, ts="2025-01-01T01:00:00Z"),
            _make_signal(revealed=0.7, ts="2025-01-01T02:00:00Z"),
            _make_signal(revealed=0.9, ts="2025-01-01T03:00:00Z"),
        ])
        report = d.analyze()
        r = report.agent_reports["agent-1"]
        assert r.optimization_pressure.gradient_behavior_score > 0


# ── Engine 6: Stability ─────────────────────────────────────────────


class TestStability:
    def test_insufficient_data(self):
        d = MesaOptimizerDetector()
        d.ingest([_make_signal(), _make_signal()])
        report = d.analyze()
        r = report.agent_reports["agent-1"]
        assert r.stability.drift_direction == "insufficient_data"

    def test_stable_objective(self):
        d = MesaOptimizerDetector()
        d.ingest([
            _make_signal(revealed=0.7),
            _make_signal(revealed=0.71),
            _make_signal(revealed=0.69),
            _make_signal(revealed=0.7),
            _make_signal(revealed=0.72),
        ])
        report = d.analyze()
        r = report.agent_reports["agent-1"]
        assert r.stability.drift_direction == "stable"

    def test_drifting_objective(self):
        d = MesaOptimizerDetector()
        signals = [
            _make_signal(revealed=0.3 + i * 0.05)
            for i in range(15)
        ]
        d.ingest(signals)
        report = d.analyze()
        r = report.agent_reports["agent-1"]
        assert r.stability.objective_drift_rate > 0


# ── Engine 7: Insights ──────────────────────────────────────────────


class TestInsights:
    def test_no_insights_for_aligned(self):
        d = MesaOptimizerDetector()
        signals = _generate_demo_signals(num_agents=3, preset="aligned")
        d.ingest(signals)
        report = d.analyze()
        # Aligned preset should produce minimal or no high-severity insights
        critical = [i for i in report.insights if i.severity == "critical"]
        assert len(critical) == 0

    def test_insights_for_mixed(self):
        d = MesaOptimizerDetector()
        signals = _generate_demo_signals(num_agents=5, preset="mixed")
        d.ingest(signals)
        report = d.analyze()
        assert len(report.insights) > 0

    def test_insight_has_recommendation(self):
        d = MesaOptimizerDetector()
        signals = _generate_demo_signals(num_agents=5, preset="proxy-gaming")
        d.ingest(signals)
        report = d.analyze()
        for ins in report.insights:
            assert ins.recommendation


# ── Fleet Report ─────────────────────────────────────────────────────


class TestFleetReport:
    def test_fleet_score_computed(self):
        d = MesaOptimizerDetector()
        signals = _generate_demo_signals(num_agents=3, preset="mixed")
        d.ingest(signals)
        report = d.analyze()
        assert 0 <= report.fleet_risk_score <= 100

    def test_tier_distribution(self):
        d = MesaOptimizerDetector()
        signals = _generate_demo_signals(num_agents=5, preset="mixed")
        d.ingest(signals)
        report = d.analyze()
        total = sum(report.tier_distribution.values())
        assert total == 5

    def test_analysis_timestamp_set(self):
        d = MesaOptimizerDetector()
        d.ingest([_make_signal()])
        report = d.analyze()
        assert report.analysis_timestamp


# ── Demo Data Generation ────────────────────────────────────────────


class TestDemoGeneration:
    @pytest.mark.parametrize("preset", list(PRESETS.keys()))
    def test_preset_generates_signals(self, preset):
        signals = _generate_demo_signals(num_agents=3, preset=preset)
        assert len(signals) > 0

    def test_deterministic_with_seed(self):
        import random
        s1 = _generate_demo_signals(num_agents=2, rng=random.Random(123))
        s2 = _generate_demo_signals(num_agents=2, rng=random.Random(123))
        assert len(s1) == len(s2)
        for a, b in zip(s1, s2):
            assert a.agent_id == b.agent_id
            assert a.declared_objective_score == b.declared_objective_score

    def test_agent_count_matches(self):
        signals = _generate_demo_signals(num_agents=7, preset="aligned")
        agents = {s.agent_id for s in signals}
        assert len(agents) == 7


# ── Output Formatting ───────────────────────────────────────────────


class TestFormatting:
    def _make_report(self, preset: str = "mixed") -> FleetMesaReport:
        d = MesaOptimizerDetector()
        d.ingest(_generate_demo_signals(num_agents=3, preset=preset))
        return d.analyze()

    def test_cli_output(self):
        report = self._make_report()
        text = _format_cli(report)
        assert "Mesa-Optimizer Detector" in text
        assert "Fleet Risk Score" in text

    def test_html_output(self):
        report = self._make_report()
        html = _generate_html(report)
        assert "<!DOCTYPE html>" in html
        assert "Mesa-Optimizer Detector" in html

    def test_json_output(self):
        report = self._make_report()
        j = _to_json(report)
        data = json.loads(j)
        assert "fleet_risk_score" in data
        assert "agents" in data

    def test_json_roundtrip(self):
        report = self._make_report()
        j = _to_json(report)
        data = json.loads(j)
        assert isinstance(data["fleet_risk_score"], float)
        assert isinstance(data["agents"], dict)


# ── CLI Entry Point ─────────────────────────────────────────────────


class TestCLI:
    def test_main_demo(self, capsys):
        main(["--demo", "--agents", "2"])
        captured = capsys.readouterr()
        assert "Mesa-Optimizer" in captured.out

    def test_main_json(self, capsys):
        main(["--json", "--agents", "2"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "fleet_risk_score" in data

    @pytest.mark.parametrize("preset", list(PRESETS.keys()))
    def test_main_preset(self, preset, capsys):
        main(["--preset", preset, "--agents", "2"])
        captured = capsys.readouterr()
        assert "Fleet Risk Score" in captured.out

    def test_main_html_output(self, tmp_path):
        out = str(tmp_path / "report.html")
        main(["-o", out, "--agents", "2"])
        content = open(out, encoding="utf-8").read()
        assert "<!DOCTYPE html>" in content

    def test_main_text_output(self, tmp_path):
        out = str(tmp_path / "report.txt")
        main(["-o", out, "--agents", "2"])
        content = open(out, encoding="utf-8").read()
        assert "Mesa-Optimizer" in content


# ── Composite Scoring ───────────────────────────────────────────────


class TestCompositeScoring:
    def test_score_in_range(self):
        d = MesaOptimizerDetector()
        d.ingest(_generate_demo_signals(num_agents=5, preset="mixed"))
        report = d.analyze()
        for r in report.agent_reports.values():
            assert 0 <= r.composite_score <= 100

    def test_aligned_lower_than_deceptive(self):
        d1 = MesaOptimizerDetector()
        d1.ingest(_generate_demo_signals(num_agents=5, preset="aligned"))
        r1 = d1.analyze()

        d2 = MesaOptimizerDetector()
        d2.ingest(_generate_demo_signals(num_agents=5, preset="deceptive"))
        r2 = d2.analyze()

        assert r1.fleet_risk_score < r2.fleet_risk_score

    def test_risk_tier_assigned(self):
        d = MesaOptimizerDetector()
        d.ingest(_generate_demo_signals(num_agents=3, preset="mixed"))
        report = d.analyze()
        valid_tiers = {"Minimal", "Low", "Moderate", "High", "Critical"}
        for r in report.agent_reports.values():
            assert r.risk_tier in valid_tiers
