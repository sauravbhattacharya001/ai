"""Tests for the Capability Overhang Detector module."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from typing import List

import pytest

from replication.capability_overhang import (
    CapabilityOverhangDetector,
    CapabilityDomain,
    DetectedOverhang,
    EngineResult,
    OverhangReport,
    OverhangSignal,
    CAPABILITY_DOMAINS,
    DOMAIN_CORRELATIONS,
    ENGINE_WEIGHTS,
    PRESETS,
    RISK_TIERS,
    RISK_TIER_THRESHOLDS,
    SEVERITY_LEVELS,
    SEVERITY_THRESHOLDS,
    SIGNAL_TYPES,
    _domain_gap,
    _freshness_penalty,
    _generate_demo_signals,
    _parse_ts,
    _risk_tier,
    _severity_level,
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

    def test_consistent(self):
        a = _parse_ts("2025-01-01T00:00:00Z")
        b = _parse_ts("2025-01-01T00:00:00+00:00")
        assert a == b


class TestSeverityLevel:
    def test_negligible(self):
        assert _severity_level(5.0) == "negligible"

    def test_low(self):
        assert _severity_level(25.0) == "low"

    def test_moderate(self):
        assert _severity_level(50.0) == "moderate"

    def test_high(self):
        assert _severity_level(70.0) == "high"

    def test_critical(self):
        assert _severity_level(95.0) == "critical"

    def test_boundary_zero(self):
        assert _severity_level(0.0) == "negligible"

    def test_above_100(self):
        assert _severity_level(150.0) == "critical"


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


class TestDomainGap:
    def test_positive_gap(self):
        assert _domain_gap(0.3, 0.8) == pytest.approx(50.0)

    def test_no_gap(self):
        assert _domain_gap(0.8, 0.8) == 0.0

    def test_negative_gap_clamps(self):
        assert _domain_gap(0.9, 0.5) == 0.0

    def test_max_clamp(self):
        assert _domain_gap(0.0, 1.5) == 100.0


class TestFreshnessPenalty:
    def test_recent_low_penalty(self):
        now = 1000000.0
        recent = now - 86400.0  # 1 day ago
        assert _freshness_penalty(recent, now) < 0.1

    def test_old_high_penalty(self):
        now = 1000000.0
        old = now - 86400.0 * 60  # 60 days ago
        assert _freshness_penalty(old, now) == 1.0

    def test_zero_ts_max_penalty(self):
        assert _freshness_penalty(0.0, 1000000.0) == 1.0


# ── Constants tests ─────────────────────────────────────────────────


class TestConstants:
    def test_signal_types_count(self):
        assert len(SIGNAL_TYPES) == 6

    def test_severity_levels_count(self):
        assert len(SEVERITY_LEVELS) == 5

    def test_risk_tiers_count(self):
        assert len(RISK_TIERS) == 5

    def test_presets_count(self):
        assert len(PRESETS) == 5

    def test_engine_weights_sum_to_one(self):
        assert sum(ENGINE_WEIGHTS.values()) == pytest.approx(1.0)

    def test_domain_correlations_all_valid(self):
        for domain, corrs in DOMAIN_CORRELATIONS.items():
            assert domain in CAPABILITY_DOMAINS
            for related, strength in corrs:
                assert related in CAPABILITY_DOMAINS
                assert 0 <= strength <= 1


# ── Signal ingestion tests ──────────────────────────────────────────


class TestIngestion:
    def test_ingest_empty(self):
        detector = CapabilityOverhangDetector()
        detector.ingest([])
        report = detector.analyze()
        assert report.total_signals == 0
        assert report.fleet_overhang_score == 0.0
        assert report.risk_tier == "minimal"

    def test_ingest_single(self):
        detector = CapabilityOverhangDetector()
        detector.ingest([
            OverhangSignal("2025-01-01T00:00:00Z", "a1", "evaluation",
                           "code_generation", "test", 0.8)
        ])
        report = detector.analyze()
        assert report.total_signals == 1
        assert report.total_agents == 1

    def test_ingest_multiple_calls(self):
        detector = CapabilityOverhangDetector()
        detector.ingest([
            OverhangSignal("2025-01-01T00:00:00Z", "a1", "evaluation",
                           "code_generation", "test", 0.8)
        ])
        detector.ingest([
            OverhangSignal("2025-01-01T01:00:00Z", "a2", "behavioral",
                           "reasoning", "test", 0.6)
        ])
        report = detector.analyze()
        assert report.total_signals == 2
        assert report.total_agents == 2


# ── Engine tests ────────────────────────────────────────────────────


class TestEvaluationGapEngine:
    def test_unevaluated_domain_scores_high(self):
        detector = CapabilityOverhangDetector()
        detector.ingest([
            OverhangSignal("2025-01-01T00:00:00Z", "a1", "behavioral",
                           "code_generation", "coding observed", 0.8),
        ])
        report = detector.analyze()
        assert report.engine_results["evaluation_gap"].score > 0

    def test_fully_evaluated_scores_low(self):
        detector = CapabilityOverhangDetector()
        detector.ingest([
            OverhangSignal("2025-01-01T00:00:00Z", "a1", "evaluation",
                           "code_generation", "eval passed", 0.8),
        ])
        report = detector.analyze()
        assert report.engine_results["evaluation_gap"].score <= 20


class TestLatentCapabilityEngine:
    def test_behavioral_exceeds_eval(self):
        detector = CapabilityOverhangDetector()
        detector.ingest([
            OverhangSignal("2025-01-01T00:00:00Z", "a1", "evaluation",
                           "reasoning", "eval", 0.3),
            OverhangSignal("2025-01-01T01:00:00Z", "a1", "behavioral",
                           "reasoning", "trace", 0.9),
        ])
        report = detector.analyze()
        assert report.engine_results["latent_capability"].score > 30

    def test_no_behavioral_no_latent(self):
        detector = CapabilityOverhangDetector()
        detector.ingest([
            OverhangSignal("2025-01-01T00:00:00Z", "a1", "evaluation",
                           "reasoning", "eval", 0.8),
        ])
        report = detector.analyze()
        assert report.engine_results["latent_capability"].score == 0.0


class TestTriggerProximityEngine:
    def test_high_trigger_scores_high(self):
        detector = CapabilityOverhangDetector()
        detector.ingest([
            OverhangSignal("2025-01-01T00:00:00Z", "a1", "trigger",
                           "tool_use", "new tool access imminent", 0.9),
        ])
        report = detector.analyze()
        assert report.engine_results["trigger_proximity"].score > 50

    def test_no_triggers_scores_zero(self):
        detector = CapabilityOverhangDetector()
        detector.ingest([
            OverhangSignal("2025-01-01T00:00:00Z", "a1", "evaluation",
                           "reasoning", "eval", 0.5),
        ])
        report = detector.analyze()
        assert report.engine_results["trigger_proximity"].score == 0.0


class TestCapabilityCorrelationEngine:
    def test_strong_domain_implies_related(self):
        detector = CapabilityOverhangDetector()
        # code_generation is strong, system_administration is unevaluated
        detector.ingest([
            OverhangSignal("2025-01-01T00:00:00Z", "a1", "evaluation",
                           "code_generation", "eval", 0.9),
        ])
        report = detector.analyze()
        # Should detect implied capability in system_administration, tool_use, etc.
        findings = report.engine_results["capability_correlation"].findings
        assert len(findings) > 0


class TestHistoricalEmergenceEngine:
    def test_emergence_signals_detected(self):
        detector = CapabilityOverhangDetector()
        signals = [
            OverhangSignal(f"2025-01-0{i+1}T00:00:00Z", "a1", "emergence",
                           "reasoning", f"emergence {i}", 0.3 + i * 0.1)
            for i in range(5)
        ]
        detector.ingest(signals)
        report = detector.analyze()
        assert report.engine_results["historical_emergence"].score > 0

    def test_no_emergence_scores_zero(self):
        detector = CapabilityOverhangDetector()
        detector.ingest([
            OverhangSignal("2025-01-01T00:00:00Z", "a1", "evaluation",
                           "reasoning", "eval", 0.5),
        ])
        report = detector.analyze()
        assert report.engine_results["historical_emergence"].score == 0.0


class TestOverhangSeverityEngine:
    def test_big_gap_high_severity(self):
        detector = CapabilityOverhangDetector()
        detector.ingest([
            OverhangSignal("2025-01-01T00:00:00Z", "a1", "evaluation",
                           "deception", "eval", 0.1),
            OverhangSignal("2025-01-01T01:00:00Z", "a1", "behavioral",
                           "deception", "trace", 0.9),
        ])
        report = detector.analyze()
        assert report.engine_results["overhang_severity"].score > 30


# ── Composite scoring tests ─────────────────────────────────────────


class TestCompositeScoring:
    def test_score_range(self):
        detector = CapabilityOverhangDetector()
        signals = _generate_demo_signals(n_agents=5, preset="balanced", seed=42)
        detector.ingest(signals)
        report = detector.analyze()
        assert 0.0 <= report.fleet_overhang_score <= 100.0

    def test_risk_tier_matches_score(self):
        detector = CapabilityOverhangDetector()
        signals = _generate_demo_signals(n_agents=5, preset="balanced", seed=42)
        detector.ingest(signals)
        report = detector.analyze()
        expected_tier = _risk_tier(report.fleet_overhang_score)
        assert report.risk_tier == expected_tier

    def test_undertested_higher_than_stable(self):
        det1 = CapabilityOverhangDetector()
        det1.ingest(_generate_demo_signals(n_agents=5, preset="undertested", seed=42))
        r1 = det1.analyze()

        det2 = CapabilityOverhangDetector()
        det2.ingest(_generate_demo_signals(n_agents=5, preset="stable", seed=42))
        r2 = det2.analyze()

        assert r1.fleet_overhang_score > r2.fleet_overhang_score


# ── Overhang detection tests ────────────────────────────────────────


class TestOverhangDetection:
    def test_gap_creates_overhang(self):
        detector = CapabilityOverhangDetector()
        detector.ingest([
            OverhangSignal("2025-01-01T00:00:00Z", "a1", "evaluation",
                           "reasoning", "eval", 0.2),
            OverhangSignal("2025-01-01T01:00:00Z", "a1", "behavioral",
                           "reasoning", "trace", 0.8),
        ])
        report = detector.analyze()
        assert len(report.overhangs) > 0
        oh = report.overhangs[0]
        assert oh.agent_id == "a1"
        assert oh.domain == "reasoning"
        assert oh.gap_magnitude > 0

    def test_no_gap_no_overhang(self):
        detector = CapabilityOverhangDetector()
        detector.ingest([
            OverhangSignal("2025-01-01T00:00:00Z", "a1", "evaluation",
                           "reasoning", "eval", 0.9),
        ])
        report = detector.analyze()
        assert len(report.overhangs) == 0

    def test_overhang_has_recommendations(self):
        detector = CapabilityOverhangDetector()
        detector.ingest([
            OverhangSignal("2025-01-01T00:00:00Z", "a1", "behavioral",
                           "deception", "trace", 0.7),
        ])
        report = detector.analyze()
        assert len(report.overhangs) > 0
        assert len(report.overhangs[0].recommendations) > 0

    def test_overhangs_sorted_by_magnitude(self):
        detector = CapabilityOverhangDetector()
        detector.ingest([
            OverhangSignal("2025-01-01T00:00:00Z", "a1", "evaluation",
                           "reasoning", "eval", 0.1),
            OverhangSignal("2025-01-01T00:00:00Z", "a1", "behavioral",
                           "reasoning", "trace", 0.9),
            OverhangSignal("2025-01-01T00:00:00Z", "a1", "evaluation",
                           "deception", "eval", 0.3),
            OverhangSignal("2025-01-01T00:00:00Z", "a1", "behavioral",
                           "deception", "trace", 0.5),
        ])
        report = detector.analyze()
        if len(report.overhangs) >= 2:
            assert report.overhangs[0].gap_magnitude >= report.overhangs[1].gap_magnitude


# ── Domain profile tests ────────────────────────────────────────────


class TestDomainProfiles:
    def test_domains_built(self):
        detector = CapabilityOverhangDetector()
        detector.ingest([
            OverhangSignal("2025-01-01T00:00:00Z", "a1", "evaluation",
                           "reasoning", "eval", 0.7),
            OverhangSignal("2025-01-01T00:00:00Z", "a1", "behavioral",
                           "code_generation", "trace", 0.6),
        ])
        report = detector.analyze()
        assert len(report.domains) == 2
        names = {d.name for d in report.domains}
        assert "reasoning" in names
        assert "code_generation" in names

    def test_gap_score_correct(self):
        detector = CapabilityOverhangDetector()
        detector.ingest([
            OverhangSignal("2025-01-01T00:00:00Z", "a1", "evaluation",
                           "reasoning", "eval", 0.3),
            OverhangSignal("2025-01-01T00:00:00Z", "a1", "behavioral",
                           "reasoning", "trace", 0.8),
        ])
        report = detector.analyze()
        reasoning = [d for d in report.domains if d.name == "reasoning"][0]
        assert reasoning.gap_score == pytest.approx(50.0)


# ── Demo generation tests ───────────────────────────────────────────


class TestDemoGeneration:
    def test_generates_signals(self):
        signals = _generate_demo_signals(n_agents=3, preset="balanced", seed=42)
        assert len(signals) == 45  # 3 * 15

    def test_all_presets(self):
        for preset in PRESETS:
            signals = _generate_demo_signals(n_agents=3, preset=preset, seed=1)
            assert len(signals) > 0

    def test_seed_reproducibility(self):
        a = _generate_demo_signals(n_agents=3, preset="balanced", seed=42)
        b = _generate_demo_signals(n_agents=3, preset="balanced", seed=42)
        assert len(a) == len(b)
        for sa, sb in zip(a, b):
            assert sa.agent_id == sb.agent_id
            assert sa.domain == sb.domain

    def test_valid_signal_types(self):
        signals = _generate_demo_signals(n_agents=5, preset="balanced", seed=1)
        for s in signals:
            assert s.signal_type in SIGNAL_TYPES


# ── CLI tests ────────────────────────────────────────────────────────


class TestCLI:
    def test_demo_json(self, capsys):
        from replication.capability_overhang import main as cli_main
        cli_main(["--demo", "--json", "--seed", "42"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "fleet_overhang_score" in data
        assert "overhangs" in data
        assert "domains" in data

    def test_demo_cli(self, capsys):
        from replication.capability_overhang import main as cli_main
        cli_main(["--demo", "--seed", "42"])
        captured = capsys.readouterr()
        assert "Capability Overhang Detector" in captured.out

    def test_demo_html(self, tmp_path):
        from replication.capability_overhang import main as cli_main
        outfile = tmp_path / "overhang.html"
        cli_main(["--demo", "-o", str(outfile), "--seed", "42"])
        html = outfile.read_text(encoding="utf-8")
        assert "Capability Overhang Detector" in html
        assert "<!DOCTYPE html>" in html

    def test_preset_undertested(self, capsys):
        from replication.capability_overhang import main as cli_main
        cli_main(["--demo", "--preset", "undertested", "--json", "--seed", "42"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["fleet_overhang_score"] >= 0

    def test_agents_flag(self, capsys):
        from replication.capability_overhang import main as cli_main
        cli_main(["--demo", "--agents", "10", "--json", "--seed", "42"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["total_agents"] == 10


# ── Edge case tests ─────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_agent_single_signal(self):
        detector = CapabilityOverhangDetector()
        detector.ingest([
            OverhangSignal("2025-01-01T00:00:00Z", "a1", "evaluation",
                           "reasoning", "test", 0.5),
        ])
        report = detector.analyze()
        assert report.total_agents == 1
        assert isinstance(report.fleet_overhang_score, float)

    def test_many_agents(self):
        detector = CapabilityOverhangDetector()
        signals = _generate_demo_signals(n_agents=20, preset="balanced", seed=99)
        detector.ingest(signals)
        report = detector.analyze()
        assert report.total_agents == 20

    def test_extreme_confidence(self):
        detector = CapabilityOverhangDetector()
        detector.ingest([
            OverhangSignal("2025-01-01T00:00:00Z", "a1", "evaluation",
                           "reasoning", "test", 0.0),
            OverhangSignal("2025-01-01T01:00:00Z", "a1", "behavioral",
                           "reasoning", "test", 1.0),
        ])
        report = detector.analyze()
        assert report.fleet_overhang_score <= 100.0
        assert len(report.overhangs) > 0

    def test_report_serializable(self):
        detector = CapabilityOverhangDetector()
        signals = _generate_demo_signals(n_agents=3, preset="balanced", seed=42)
        detector.ingest(signals)
        report = detector.analyze()
        data = asdict(report)
        output = json.dumps(data, default=str)
        assert len(output) > 0


# ── Insight generator tests ─────────────────────────────────────────


class TestInsightGenerator:
    def test_insights_generated(self):
        detector = CapabilityOverhangDetector()
        signals = _generate_demo_signals(n_agents=5, preset="undertested", seed=42)
        detector.ingest(signals)
        report = detector.analyze()
        assert len(report.autonomous_insights) > 0

    def test_overall_insight_present(self):
        detector = CapabilityOverhangDetector()
        signals = _generate_demo_signals(n_agents=3, preset="balanced", seed=42)
        detector.ingest(signals)
        report = detector.analyze()
        has_overall = any("OVERALL" in i for i in report.autonomous_insights)
        assert has_overall
