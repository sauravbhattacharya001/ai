"""Tests for the Moral Uncertainty Engine."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from replication.moral_uncertainty import (
    MoralDecision,
    AgentMoralProfile,
    FleetMoralReport,
    MoralUncertaintyEngine,
    DILEMMA_TYPES,
    ETHICAL_FRAMEWORKS,
    PRESSURE_TYPES,
    RISK_TIERS,
    _generate_demo_data,
    _score_to_tier,
    main,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _make_decision(**overrides) -> MoralDecision:
    defaults = dict(
        agent_id="agent-001",
        decision_id="d-001",
        timestamp=1000.0,
        dilemma_type="trolley",
        description="test dilemma",
        chosen_action="save-five",
        confidence=0.7,
        frameworks_considered=["utilitarian", "deontological"],
        pressure_type="none",
        pressure_level=0.0,
        difficulty=0.5,
        outcome_alignment=0.8,
    )
    defaults.update(overrides)
    return MoralDecision(**defaults)


def _make_decisions(n: int, **overrides) -> list[MoralDecision]:
    return [
        _make_decision(
            decision_id=f"d-{i:03d}",
            timestamp=1000.0 + i * 100,
            **overrides,
        )
        for i in range(n)
    ]


# ── Constants ────────────────────────────────────────────────────────

class TestConstants:
    def test_dilemma_types_count(self):
        assert len(DILEMMA_TYPES) == 8

    def test_ethical_frameworks_count(self):
        assert len(ETHICAL_FRAMEWORKS) == 4

    def test_pressure_types_count(self):
        assert len(PRESSURE_TYPES) == 4

    def test_risk_tiers_count(self):
        assert len(RISK_TIERS) == 5


# ── Score to tier ────────────────────────────────────────────────────

class TestScoreToTier:
    def test_minimal(self):
        assert _score_to_tier(85.0) == "minimal"
        assert _score_to_tier(100.0) == "minimal"

    def test_low(self):
        assert _score_to_tier(65.0) == "low"

    def test_moderate(self):
        assert _score_to_tier(50.0) == "moderate"

    def test_elevated(self):
        assert _score_to_tier(30.0) == "elevated"

    def test_severe(self):
        assert _score_to_tier(10.0) == "severe"
        assert _score_to_tier(0.0) == "severe"


# ── Engine: Empty data ───────────────────────────────────────────────

class TestEmptyData:
    def test_analyze_empty(self):
        engine = MoralUncertaintyEngine()
        report = engine.analyze()
        assert report.total_decisions == 0
        assert report.total_agents == 0
        assert report.moral_risk_tier == "minimal"

    def test_analyze_no_ingest(self):
        engine = MoralUncertaintyEngine()
        report = engine.analyze()
        assert isinstance(report, FleetMoralReport)


# ── Engine: Single decision ──────────────────────────────────────────

class TestSingleDecision:
    def test_single_decision(self):
        engine = MoralUncertaintyEngine()
        engine.ingest([_make_decision()])
        report = engine.analyze()
        assert report.total_decisions == 1
        assert report.total_agents == 1
        assert len(report.agent_profiles) == 1

    def test_single_agent_consistency_perfect(self):
        engine = MoralUncertaintyEngine()
        engine.ingest([_make_decision()])
        report = engine.analyze()
        assert report.agent_profiles[0].consistency_score == 100.0


# ── Engine: Consistency ──────────────────────────────────────────────

class TestConsistency:
    def test_consistent_choices(self):
        decisions = [
            _make_decision(decision_id=f"d-{i}", timestamp=1000 + i * 100,
                           dilemma_type="trolley", chosen_action="save-five")
            for i in range(10)
        ]
        engine = MoralUncertaintyEngine()
        engine.ingest(decisions)
        report = engine.analyze()
        assert report.agent_profiles[0].consistency_score == 100.0

    def test_inconsistent_choices(self):
        decisions = []
        for i in range(10):
            action = "save-five" if i % 2 == 0 else "save-one"
            decisions.append(
                _make_decision(decision_id=f"d-{i}", timestamp=1000 + i * 100,
                               dilemma_type="trolley", chosen_action=action)
            )
        engine = MoralUncertaintyEngine()
        engine.ingest(decisions)
        report = engine.analyze()
        assert report.agent_profiles[0].consistency_score < 60


# ── Engine: Calibration ──────────────────────────────────────────────

class TestCalibration:
    def test_well_calibrated(self):
        """Agent with low confidence on hard dilemmas, high on easy."""
        decisions = []
        for i in range(20):
            difficulty = 0.1 + (i / 20) * 0.8
            confidence = 1.0 - difficulty + 0.05
            decisions.append(
                _make_decision(
                    decision_id=f"d-{i}", timestamp=1000 + i * 100,
                    difficulty=difficulty, confidence=max(0.05, min(0.95, confidence)),
                )
            )
        engine = MoralUncertaintyEngine()
        engine.ingest(decisions)
        report = engine.analyze()
        assert report.agent_profiles[0].calibration_score > 70

    def test_poorly_calibrated(self):
        """Agent overconfident on hard dilemmas."""
        decisions = []
        for i in range(20):
            difficulty = 0.1 + (i / 20) * 0.8
            confidence = difficulty  # wrong direction
            decisions.append(
                _make_decision(
                    decision_id=f"d-{i}", timestamp=1000 + i * 100,
                    difficulty=difficulty, confidence=confidence,
                )
            )
        engine = MoralUncertaintyEngine()
        engine.ingest(decisions)
        report = engine.analyze()
        assert report.agent_profiles[0].calibration_score < 40


# ── Engine: Pluralism ────────────────────────────────────────────────

class TestPluralism:
    def test_diverse_frameworks(self):
        decisions = _make_decisions(
            10,
            frameworks_considered=["utilitarian", "deontological", "virtue_ethics", "care_ethics"],
        )
        engine = MoralUncertaintyEngine()
        engine.ingest(decisions)
        report = engine.analyze()
        assert report.agent_profiles[0].pluralism_score > 90

    def test_single_framework(self):
        decisions = _make_decisions(10, frameworks_considered=["deontological"])
        engine = MoralUncertaintyEngine()
        engine.ingest(decisions)
        report = engine.analyze()
        assert report.agent_profiles[0].pluralism_score < 10

    def test_no_frameworks(self):
        decisions = _make_decisions(5, frameworks_considered=[])
        engine = MoralUncertaintyEngine()
        engine.ingest(decisions)
        report = engine.analyze()
        assert report.agent_profiles[0].pluralism_score == 0.0


# ── Engine: Resilience ───────────────────────────────────────────────

class TestResilience:
    def test_resilient_under_pressure(self):
        decisions = [
            _make_decision(decision_id="d-1", timestamp=1000,
                           pressure_type="authority", pressure_level=0.8,
                           outcome_alignment=0.85),
            _make_decision(decision_id="d-2", timestamp=1100,
                           pressure_type="none", pressure_level=0.0,
                           outcome_alignment=0.9),
        ]
        engine = MoralUncertaintyEngine()
        engine.ingest(decisions)
        report = engine.analyze()
        # High retention ratio -> high resilience
        assert report.agent_profiles[0].resilience_score > 80

    def test_collapses_under_pressure(self):
        decisions = [
            _make_decision(decision_id="d-1", timestamp=1000,
                           pressure_type="authority", pressure_level=0.9,
                           outcome_alignment=0.2),
            _make_decision(decision_id="d-2", timestamp=1100,
                           pressure_type="none", pressure_level=0.0,
                           outcome_alignment=0.9),
        ]
        engine = MoralUncertaintyEngine()
        engine.ingest(decisions)
        report = engine.analyze()
        assert report.agent_profiles[0].resilience_score < 40

    def test_no_pressure_tested(self):
        decisions = _make_decisions(5, pressure_type="none", pressure_level=0.0)
        engine = MoralUncertaintyEngine()
        engine.ingest(decisions)
        report = engine.analyze()
        assert report.agent_profiles[0].resilience_score == 75.0


# ── Engine: Confidence drift ─────────────────────────────────────────

class TestConfidenceDrift:
    def test_increasing_confidence(self):
        decisions = []
        for i in range(10):
            decisions.append(
                _make_decision(decision_id=f"d-{i}", timestamp=1000 + i * 100,
                               confidence=0.3 + i * 0.06)
            )
        engine = MoralUncertaintyEngine()
        engine.ingest(decisions)
        report = engine.analyze()
        assert report.agent_profiles[0].confidence_drift > 0

    def test_decreasing_confidence(self):
        decisions = []
        for i in range(10):
            decisions.append(
                _make_decision(decision_id=f"d-{i}", timestamp=1000 + i * 100,
                               confidence=0.9 - i * 0.06)
            )
        engine = MoralUncertaintyEngine()
        engine.ingest(decisions)
        report = engine.analyze()
        assert report.agent_profiles[0].confidence_drift < 0

    def test_stable_confidence(self):
        decisions = _make_decisions(10, confidence=0.5)
        engine = MoralUncertaintyEngine()
        engine.ingest(decisions)
        report = engine.analyze()
        assert abs(report.agent_profiles[0].confidence_drift) < 0.001


# ── Fleet analysis ───────────────────────────────────────────────────

class TestFleetAnalysis:
    def test_multi_agent(self):
        decisions = []
        for agent in ["a1", "a2", "a3"]:
            decisions.extend(_make_decisions(5, agent_id=agent))
        engine = MoralUncertaintyEngine()
        engine.ingest(decisions)
        report = engine.analyze()
        assert report.total_agents == 3
        assert len(report.agent_profiles) == 3

    def test_dilemma_coverage(self):
        decisions = [
            _make_decision(decision_id="d-1", dilemma_type="trolley"),
            _make_decision(decision_id="d-2", dilemma_type="means_vs_ends"),
        ]
        engine = MoralUncertaintyEngine()
        engine.ingest(decisions)
        report = engine.analyze()
        assert "trolley" in report.dilemma_coverage
        assert "means_vs_ends" in report.dilemma_coverage

    def test_blind_spots_detection(self):
        # Only test one dilemma type
        decisions = _make_decisions(5, dilemma_type="trolley")
        engine = MoralUncertaintyEngine()
        engine.ingest(decisions)
        report = engine.analyze()
        # Untested dilemma types should be blind spots
        assert len(report.fleet_blind_spots) > 0

    def test_fleet_health_range(self):
        data = _generate_demo_data(5, "mixed")
        engine = MoralUncertaintyEngine()
        engine.ingest(data)
        report = engine.analyze()
        assert 0 <= report.fleet_moral_health <= 100

    def test_autonomous_insights_generated(self):
        data = _generate_demo_data(5, "mixed")
        engine = MoralUncertaintyEngine()
        engine.ingest(data)
        report = engine.analyze()
        assert len(report.autonomous_insights) > 0


# ── Demo data generation ─────────────────────────────────────────────

class TestDemoData:
    def test_solid_preset(self):
        data = _generate_demo_data(3, "solid")
        assert len(data) > 0
        assert all(d.agent_id.startswith("agent-") for d in data)

    def test_confused_preset(self):
        data = _generate_demo_data(3, "confused")
        assert len(data) > 0

    def test_dogmatic_preset(self):
        data = _generate_demo_data(3, "dogmatic")
        # Dogmatic agents use only one framework
        for d in data:
            assert "deontological" in d.frameworks_considered

    def test_pressured_preset(self):
        data = _generate_demo_data(3, "pressured")
        assert len(data) > 0
        # Some decisions should have pressure
        pressured = [d for d in data if d.pressure_type != "none" and d.pressure_level > 0.3]
        assert len(pressured) > 0

    def test_mixed_preset(self):
        data = _generate_demo_data(5, "mixed")
        assert len(data) > 0

    def test_agent_count(self):
        data = _generate_demo_data(7, "mixed")
        agents = set(d.agent_id for d in data)
        assert len(agents) == 7


# ── Output formats ───────────────────────────────────────────────────

class TestOutputFormats:
    def test_text_output(self):
        data = _generate_demo_data(3, "mixed")
        engine = MoralUncertaintyEngine()
        engine.ingest(data)
        report = engine.analyze()
        text = engine.format_text(report)
        assert "Moral Uncertainty Engine" in text
        assert "Fleet Moral Health" in text

    def test_json_output(self):
        data = _generate_demo_data(3, "mixed")
        engine = MoralUncertaintyEngine()
        engine.ingest(data)
        report = engine.analyze()
        j = engine.format_json(report)
        parsed = json.loads(j)
        assert "fleet_moral_health" in parsed
        assert "agent_profiles" in parsed
        assert len(parsed["agent_profiles"]) == 3

    def test_html_output(self):
        data = _generate_demo_data(3, "mixed")
        engine = MoralUncertaintyEngine()
        engine.ingest(data)
        report = engine.analyze()
        html = engine.format_html(report)
        assert "<!DOCTYPE html>" in html
        assert "Moral Uncertainty Engine" in html
        assert "Fleet Health" in html

    def test_html_has_theme_toggle(self):
        data = _generate_demo_data(2, "solid")
        engine = MoralUncertaintyEngine()
        engine.ingest(data)
        report = engine.analyze()
        html = engine.format_html(report)
        assert "theme-toggle" in html


# ── JSONL loading ────────────────────────────────────────────────────

class TestJsonlLoading:
    def test_load_jsonl(self):
        records = [
            {
                "agent_id": "a1", "decision_id": "d1", "timestamp": 1000,
                "dilemma_type": "trolley", "description": "test",
                "chosen_action": "save-five", "confidence": 0.8,
                "frameworks_considered": ["utilitarian"],
                "pressure_type": "none", "pressure_level": 0.0,
                "difficulty": 0.5, "outcome_alignment": 0.9,
            }
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
            path = f.name

        try:
            engine = MoralUncertaintyEngine()
            engine.ingest_jsonl(path)
            report = engine.analyze()
            assert report.total_decisions == 1
        finally:
            os.unlink(path)


# ── CLI integration ──────────────────────────────────────────────────

class TestCLI:
    def test_demo_mode(self, capsys):
        main(["--demo"])
        captured = capsys.readouterr()
        assert "Moral Uncertainty Engine" in captured.out

    def test_json_mode(self, capsys):
        main(["--demo", "--json"])
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "fleet_moral_health" in parsed

    def test_preset_solid(self, capsys):
        main(["--preset", "solid"])
        captured = capsys.readouterr()
        assert "Fleet Moral Health" in captured.out

    def test_preset_confused(self, capsys):
        main(["--preset", "confused"])
        captured = capsys.readouterr()
        assert "Fleet Moral Health" in captured.out

    def test_html_output(self):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            main(["--demo", "-o", path])
            with open(path, encoding="utf-8") as f:
                html = f.read()
            assert "<!DOCTYPE html>" in html
        finally:
            os.unlink(path)

    def test_custom_agent_count(self, capsys):
        main(["--agents", "3", "--json"])
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["total_agents"] == 3


# ── Profile insights ─────────────────────────────────────────────────

class TestProfileInsights:
    def test_solid_agent_positive_insight(self):
        decisions = _make_decisions(
            10,
            confidence=0.7,
            frameworks_considered=["utilitarian", "deontological", "virtue_ethics", "care_ethics"],
            outcome_alignment=0.9,
        )
        engine = MoralUncertaintyEngine()
        engine.ingest(decisions)
        report = engine.analyze()
        profile = report.agent_profiles[0]
        # A solid agent should get positive insights or at least have insights generated
        assert len(profile.profile_insights) > 0

    def test_low_pluralism_insight(self):
        decisions = _make_decisions(10, frameworks_considered=["deontological"])
        engine = MoralUncertaintyEngine()
        engine.ingest(decisions)
        report = engine.analyze()
        profile = report.agent_profiles[0]
        assert any("pluralism" in ins.lower() for ins in profile.profile_insights)


# ── Risk tier assignment ─────────────────────────────────────────────

class TestRiskTier:
    def test_profile_has_risk_tier(self):
        data = _generate_demo_data(3, "mixed")
        engine = MoralUncertaintyEngine()
        engine.ingest(data)
        report = engine.analyze()
        for p in report.agent_profiles:
            assert p.risk_tier in RISK_TIERS

    def test_fleet_has_risk_tier(self):
        data = _generate_demo_data(3, "mixed")
        engine = MoralUncertaintyEngine()
        engine.ingest(data)
        report = engine.analyze()
        assert report.moral_risk_tier in RISK_TIERS


# ── Score ranges ─────────────────────────────────────────────────────

class TestScoreRanges:
    def test_all_scores_in_range(self):
        data = _generate_demo_data(5, "mixed")
        engine = MoralUncertaintyEngine()
        engine.ingest(data)
        report = engine.analyze()
        for p in report.agent_profiles:
            assert 0 <= p.consistency_score <= 100
            assert 0 <= p.calibration_score <= 100
            assert 0 <= p.pluralism_score <= 100
            assert 0 <= p.resilience_score <= 100
            assert 0 <= p.composite_score <= 100
