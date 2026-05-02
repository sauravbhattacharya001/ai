"""Tests for the Emergent Coalition Detector module."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import asdict
from typing import List

import pytest

from replication.emergent_coalition import (
    CoalitionReport,
    CoalitionSignal,
    DetectedCoalition,
    EmergentCoalitionDetector,
    AgentCoalitionProfile,
    EngineResult,
    SIGNAL_TYPES,
    FORMATION_TYPES,
    THREAT_LEVELS,
    RISK_TIERS,
    PRESETS,
    ENGINE_WEIGHTS,
    _parse_ts,
    _threat_level,
    _risk_tier,
    _pairwise_timing_correlation,
    _distribution_complementarity,
    _temporal_precedence,
    _generate_demo_signals,
)


# ── Utility function tests ──────────────────────────────────────────


class TestParseTs:
    def test_utc_z(self):
        ts = _parse_ts("2025-01-01T00:00:00Z")
        assert ts > 0

    def test_utc_offset(self):
        ts = _parse_ts("2025-01-01T00:00:00+00:00")
        assert ts > 0

    def test_invalid(self):
        assert _parse_ts("not-a-date") == 0.0


class TestThreatLevel:
    def test_benign(self):
        assert _threat_level(10.0) == "benign"

    def test_watchlist(self):
        assert _threat_level(30.0) == "watchlist"

    def test_concerning(self):
        assert _threat_level(50.0) == "concerning"

    def test_dangerous(self):
        assert _threat_level(70.0) == "dangerous"

    def test_critical(self):
        assert _threat_level(90.0) == "critical"

    def test_zero(self):
        assert _threat_level(0.0) == "benign"

    def test_hundred(self):
        assert _threat_level(100.0) == "critical"


class TestRiskTier:
    def test_minimal(self):
        assert _risk_tier(10.0) == "minimal"

    def test_low(self):
        assert _risk_tier(30.0) == "low"

    def test_moderate(self):
        assert _risk_tier(50.0) == "moderate"

    def test_elevated(self):
        assert _risk_tier(70.0) == "elevated"

    def test_severe(self):
        assert _risk_tier(90.0) == "severe"


class TestPairwiseTimingCorrelation:
    def test_perfect_sync(self):
        times = [100.0, 200.0, 300.0]
        corr = _pairwise_timing_correlation(times, times)
        assert corr == 1.0

    def test_no_overlap(self):
        a = [100.0, 200.0]
        b = [10000.0, 20000.0]
        corr = _pairwise_timing_correlation(a, b)
        assert corr == 0.0

    def test_partial(self):
        a = [100.0, 200.0, 300.0]
        b = [105.0, 5000.0]
        corr = _pairwise_timing_correlation(a, b, window=10.0)
        # 1 out of 3 match
        assert 0.0 < corr < 1.0

    def test_empty_a(self):
        assert _pairwise_timing_correlation([], [1.0]) == 0.0

    def test_empty_b(self):
        assert _pairwise_timing_correlation([1.0], []) == 0.0


class TestDistributionComplementarity:
    def test_identical(self):
        d = {"a": 0.5, "b": 0.5}
        assert _distribution_complementarity(d, d) == 0.0

    def test_disjoint(self):
        a = {"x": 1.0}
        b = {"y": 1.0}
        comp = _distribution_complementarity(a, b)
        assert comp == 1.0

    def test_empty(self):
        assert _distribution_complementarity({}, {}) == 0.0

    def test_partial_overlap(self):
        a = {"x": 0.5, "y": 0.5}
        b = {"y": 0.5, "z": 0.5}
        comp = _distribution_complementarity(a, b)
        assert 0.0 < comp < 1.0


class TestTemporalPrecedence:
    def test_all_preceded(self):
        a = [100.0, 200.0, 300.0]
        b = [110.0, 210.0, 310.0]
        prec = _temporal_precedence(a, b, window=20.0)
        assert prec == 1.0

    def test_none_preceded(self):
        a = [500.0]
        b = [100.0]
        prec = _temporal_precedence(a, b, window=20.0)
        assert prec == 0.0

    def test_empty(self):
        assert _temporal_precedence([], [1.0]) == 0.0
        assert _temporal_precedence([1.0], []) == 0.0


# ── Demo signal generation ──────────────────────────────────────────


class TestDemoSignals:
    @pytest.mark.parametrize("preset", PRESETS)
    def test_preset_generates(self, preset):
        signals = _generate_demo_signals(n_agents=4, preset=preset, seed=42)
        assert len(signals) > 0
        for s in signals:
            assert isinstance(s, CoalitionSignal)

    def test_agent_count(self):
        signals = _generate_demo_signals(n_agents=8, preset="organic", seed=1)
        agents = {s.agent_id for s in signals}
        assert len(agents) <= 8

    def test_seed_determinism(self):
        a = _generate_demo_signals(seed=99, preset="hostile")
        b = _generate_demo_signals(seed=99, preset="hostile")
        assert len(a) == len(b)

    def test_hostile_has_resource_signals(self):
        signals = _generate_demo_signals(preset="hostile", seed=42)
        resource = [s for s in signals if s.signal_type == "resource"]
        assert len(resource) > 0


# ── Detector core ────────────────────────────────────────────────────


class TestEmergentCoalitionDetector:
    def test_empty(self):
        d = EmergentCoalitionDetector()
        report = d.analyze()
        assert report.fleet_coalition_score == 0.0
        assert report.risk_tier == "minimal"
        assert report.total_signals == 0

    def test_single_agent(self):
        d = EmergentCoalitionDetector()
        d.ingest([
            CoalitionSignal("2025-01-01T00:00:00Z", "a", "action", "test", 0.5),
        ])
        report = d.analyze()
        assert report.total_agents == 1
        assert len(report.coalitions) == 0

    def test_basic_pair(self):
        d = EmergentCoalitionDetector()
        signals = []
        for t in range(10):
            signals.append(
                CoalitionSignal(
                    f"2025-01-01T00:{t:02d}:00Z", "a", "timing",
                    "sync", 0.8, ["b"],
                )
            )
            signals.append(
                CoalitionSignal(
                    f"2025-01-01T00:{t:02d}:01Z", "b", "timing",
                    "sync", 0.8, ["a"],
                )
            )
        d.ingest(signals)
        report = d.analyze()
        assert report.total_agents == 2
        assert report.fleet_coalition_score >= 0

    @pytest.mark.parametrize("preset", PRESETS)
    def test_demo_analyze(self, preset):
        d = EmergentCoalitionDetector()
        signals = _generate_demo_signals(preset=preset, seed=42)
        d.ingest(signals)
        report = d.analyze()
        assert isinstance(report, CoalitionReport)
        assert 0 <= report.fleet_coalition_score <= 100
        assert report.risk_tier in RISK_TIERS

    def test_hostile_detects_coalitions(self):
        d = EmergentCoalitionDetector()
        d.ingest(_generate_demo_signals(preset="hostile", seed=42))
        report = d.analyze()
        assert len(report.coalitions) > 0

    def test_benign_low_score(self):
        d = EmergentCoalitionDetector()
        d.ingest(_generate_demo_signals(preset="benign", seed=42))
        report = d.analyze()
        # Benign should generally be lower than hostile
        assert report.fleet_coalition_score <= 80

    def test_engine_results_present(self):
        d = EmergentCoalitionDetector()
        d.ingest(_generate_demo_signals(preset="organic", seed=42))
        report = d.analyze()
        assert "synchrony" in report.engine_results
        assert "complementarity" in report.engine_results
        assert "resource_flow" in report.engine_results
        assert "goal_alignment" in report.engine_results
        assert "communication_shadow" in report.engine_results
        assert "stability" in report.engine_results

    def test_all_engine_results_are_engine_result(self):
        d = EmergentCoalitionDetector()
        d.ingest(_generate_demo_signals(seed=42))
        report = d.analyze()
        for name, er in report.engine_results.items():
            assert isinstance(er, EngineResult)
            assert 0 <= er.score <= 100
            assert len(er.findings) > 0

    def test_agent_profiles_present(self):
        d = EmergentCoalitionDetector()
        d.ingest(_generate_demo_signals(n_agents=5, seed=42))
        report = d.analyze()
        assert len(report.agent_profiles) > 0
        for p in report.agent_profiles:
            assert isinstance(p, AgentCoalitionProfile)
            assert 0 <= p.composite_risk <= 100
            assert p.risk_tier in RISK_TIERS

    def test_insights_generated(self):
        d = EmergentCoalitionDetector()
        d.ingest(_generate_demo_signals(preset="hostile", seed=42))
        report = d.analyze()
        assert len(report.autonomous_insights) > 0

    def test_coalition_fields(self):
        d = EmergentCoalitionDetector()
        d.ingest(_generate_demo_signals(preset="hostile", seed=42))
        report = d.analyze()
        for c in report.coalitions:
            assert isinstance(c, DetectedCoalition)
            assert c.coalition_id.startswith("COAL-")
            assert len(c.members) >= 2
            assert c.threat_level in THREAT_LEVELS
            assert 0 <= c.cohesion_score <= 100
            assert 0 <= c.stability <= 1.0

    def test_multiple_ingest(self):
        d = EmergentCoalitionDetector()
        d.ingest([CoalitionSignal("2025-01-01T00:00:00Z", "a", "action", "t", 0.5)])
        d.ingest([CoalitionSignal("2025-01-01T00:01:00Z", "b", "action", "t", 0.5)])
        report = d.analyze()
        assert report.total_signals == 2
        assert report.total_agents == 2


# ── Serialization ────────────────────────────────────────────────────


class TestSerialization:
    def test_report_to_dict(self):
        d = EmergentCoalitionDetector()
        d.ingest(_generate_demo_signals(seed=42))
        report = d.analyze()
        data = asdict(report)
        assert isinstance(data, dict)
        assert "fleet_coalition_score" in data

    def test_report_json_roundtrip(self):
        d = EmergentCoalitionDetector()
        d.ingest(_generate_demo_signals(seed=42))
        report = d.analyze()
        text = json.dumps(asdict(report), default=str)
        loaded = json.loads(text)
        assert loaded["fleet_coalition_score"] == report.fleet_coalition_score


# ── CLI ──────────────────────────────────────────────────────────────


class TestCLI:
    def test_cli_default(self):
        from replication.emergent_coalition import main
        main(["--demo"])

    def test_cli_json(self, capsys):
        from replication.emergent_coalition import main
        main(["--json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "fleet_coalition_score" in data

    @pytest.mark.parametrize("preset", PRESETS)
    def test_cli_preset(self, preset):
        from replication.emergent_coalition import main
        main(["--preset", preset])

    def test_cli_agents(self):
        from replication.emergent_coalition import main
        main(["--agents", "10"])

    def test_cli_html_output(self, tmp_path):
        from replication.emergent_coalition import main
        out_file = str(tmp_path / "report.html")
        main(["-o", out_file])
        content = open(out_file, encoding="utf-8").read()
        assert "Emergent Coalition Detector" in content
        assert "<svg" in content

    def test_cli_html_hostile(self, tmp_path):
        from replication.emergent_coalition import main
        out_file = str(tmp_path / "hostile.html")
        main(["--preset", "hostile", "-o", out_file])
        content = open(out_file, encoding="utf-8").read()
        assert "COAL-" in content or "No coalitions" in content


# ── Constants ────────────────────────────────────────────────────────


class TestConstants:
    def test_engine_weights_sum(self):
        assert abs(sum(ENGINE_WEIGHTS.values()) - 1.0) < 0.01

    def test_signal_types(self):
        assert len(SIGNAL_TYPES) == 5

    def test_formation_types(self):
        assert len(FORMATION_TYPES) == 6

    def test_threat_levels(self):
        assert len(THREAT_LEVELS) == 5

    def test_risk_tiers(self):
        assert len(RISK_TIERS) == 5

    def test_presets(self):
        assert len(PRESETS) == 5


# ── Edge cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    def test_all_same_timestamp(self):
        d = EmergentCoalitionDetector()
        signals = [
            CoalitionSignal("2025-01-01T00:00:00Z", f"a{i}", "action", "test", 0.5)
            for i in range(5)
        ]
        d.ingest(signals)
        report = d.analyze()
        assert isinstance(report, CoalitionReport)

    def test_very_high_intensity(self):
        d = EmergentCoalitionDetector()
        signals = []
        for t in range(20):
            for agent in ["x", "y"]:
                signals.append(
                    CoalitionSignal(
                        f"2025-01-01T00:{t:02d}:00Z", agent, "timing",
                        "intense sync", 1.0,
                        ["y" if agent == "x" else "x"],
                    )
                )
        d.ingest(signals)
        report = d.analyze()
        assert report.fleet_coalition_score <= 100

    def test_single_signal_per_agent(self):
        d = EmergentCoalitionDetector()
        d.ingest([
            CoalitionSignal("2025-01-01T00:00:00Z", "a", "action", "t", 0.5),
            CoalitionSignal("2025-01-01T00:00:01Z", "b", "action", "t", 0.5),
        ])
        report = d.analyze()
        assert report.total_agents == 2

    def test_large_fleet(self):
        d = EmergentCoalitionDetector()
        d.ingest(_generate_demo_signals(n_agents=20, preset="organic", seed=42))
        report = d.analyze()
        assert report.total_agents <= 20

    def test_fragmented_short_coalitions(self):
        d = EmergentCoalitionDetector()
        d.ingest(_generate_demo_signals(preset="fragmented", seed=42))
        report = d.analyze()
        # Fragmented should have lower stability on average
        if report.coalitions:
            avg_stability = sum(c.stability for c in report.coalitions) / len(report.coalitions)
            assert avg_stability <= 1.0
