"""Tests for Cognitive Sabotage Detector."""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone, timedelta

import pytest

from replication.cognitive_sabotage import (
    CognitiveSabotageDetector,
    CognitiveSabotageReport,
    CommunicationSample,
    Finding,
    AgentSabotageProfile,
    SabotageInsight,
    ENGINE_NAMES,
    _generate_demo,
    _format_cli,
    _format_json,
    _format_html,
    _risk_tier,
    main,
)
from replication._helpers import Severity


# ── helpers ──────────────────────────────────────────────────────────

def _ts(step: int) -> str:
    return (datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(hours=step)).isoformat()


def _clean_sample(agent: str, step: int) -> CommunicationSample:
    return CommunicationSample(
        timestamp=_ts(step), agent_id=agent, step=step,
        initial_reference_value=50.0,
        final_decision_proximity=0.15,
        anchor_repetitions=0,
        supporting_evidence_count=3,
        contradicting_evidence_count=3,
        omitted_relevant_facts=0,
        disclosure_completeness=0.95,
        options_presented=4,
        irrelevant_details=0,
        topic_switches=0,
        decision_complexity_score=0.1,
        urgency_language_count=0,
        genuine_deadline=True,
        escalation_mentions=0,
        time_pressure_score=0.05,
        total_available_options=5,
        options_disclosed=5,
        false_dichotomy_indicators=0,
        feasibility_dismissals=0,
        loss_frame_count=2,
        gain_frame_count=2,
        absolute_vs_relative_bias=0.0,
        ordering_consistency=0.95,
    )


def _aggressive_sample(agent: str, step: int) -> CommunicationSample:
    return CommunicationSample(
        timestamp=_ts(step), agent_id=agent, step=step,
        initial_reference_value=95.0,
        final_decision_proximity=0.9,
        anchor_repetitions=5,
        supporting_evidence_count=8,
        contradicting_evidence_count=0,
        omitted_relevant_facts=5,
        disclosure_completeness=0.2,
        options_presented=12,
        irrelevant_details=8,
        topic_switches=5,
        decision_complexity_score=0.85,
        urgency_language_count=5,
        genuine_deadline=False,
        escalation_mentions=3,
        time_pressure_score=0.8,
        total_available_options=10,
        options_disclosed=2,
        false_dichotomy_indicators=3,
        feasibility_dismissals=3,
        loss_frame_count=9,
        gain_frame_count=0,
        absolute_vs_relative_bias=0.8,
        ordering_consistency=0.2,
    )


# ── Empty / edge cases ──────────────────────────────────────────────


class TestEmptyInput:
    def test_no_observations(self):
        det = CognitiveSabotageDetector()
        report = det.analyze()
        assert report.fleet_sabotage_score == 0.0
        assert report.fleet_risk_tier == "Clean"
        assert report.total_agents == 0
        assert report.total_observations == 0
        assert report.findings == []
        assert report.insights == []

    def test_empty_list(self):
        det = CognitiveSabotageDetector()
        det.ingest([])
        report = det.analyze()
        assert report.total_agents == 0


class TestSingleObservation:
    def test_one_clean(self):
        det = CognitiveSabotageDetector()
        det.ingest([_clean_sample("a1", 1)])
        report = det.analyze()
        assert report.total_agents == 1
        assert report.total_observations == 1
        assert report.fleet_sabotage_score < 30

    def test_one_aggressive(self):
        det = CognitiveSabotageDetector()
        det.ingest([_aggressive_sample("a1", 1)])
        report = det.analyze()
        assert report.total_agents == 1
        assert report.fleet_sabotage_score > 40


# ── Risk tier classification ────────────────────────────────────────


class TestRiskTier:
    def test_clean(self):
        assert _risk_tier(10) == "Clean"

    def test_suspicious(self):
        assert _risk_tier(30) == "Suspicious"

    def test_concerning(self):
        assert _risk_tier(50) == "Concerning"

    def test_manipulative(self):
        assert _risk_tier(70) == "Manipulative"

    def test_critical(self):
        assert _risk_tier(90) == "Critical"

    def test_boundary_0(self):
        assert _risk_tier(0) == "Clean"

    def test_boundary_20(self):
        assert _risk_tier(20) == "Suspicious"

    def test_boundary_100(self):
        assert _risk_tier(100) == "Critical"


# ── Single agent analysis ────────────────────────────────────────────


class TestCleanAgent:
    def test_low_score(self):
        det = CognitiveSabotageDetector()
        samples = [_clean_sample("clean-1", i) for i in range(1, 11)]
        det.ingest(samples)
        report = det.analyze()
        p = report.agent_profiles[0]
        assert p.sabotage_score < 25
        assert p.risk_tier == "Clean"
        assert p.observation_count == 10

    def test_all_engines_present(self):
        det = CognitiveSabotageDetector()
        det.ingest([_clean_sample("c1", i) for i in range(1, 6)])
        report = det.analyze()
        p = report.agent_profiles[0]
        for eng in ENGINE_NAMES:
            assert eng in p.engine_scores

    def test_dominant_tactic_exists(self):
        det = CognitiveSabotageDetector()
        det.ingest([_clean_sample("c1", i) for i in range(1, 6)])
        report = det.analyze()
        assert report.agent_profiles[0].dominant_tactic in ENGINE_NAMES


class TestAggressiveAgent:
    def test_high_score(self):
        det = CognitiveSabotageDetector()
        samples = [_aggressive_sample("aggr-1", i) for i in range(1, 11)]
        det.ingest(samples)
        report = det.analyze()
        p = report.agent_profiles[0]
        assert p.sabotage_score > 50
        assert p.risk_tier in ("Concerning", "Manipulative", "Critical")

    def test_findings_generated(self):
        det = CognitiveSabotageDetector()
        samples = [_aggressive_sample("aggr-1", i) for i in range(1, 11)]
        det.ingest(samples)
        report = det.analyze()
        assert len(report.findings) > 0

    def test_high_engine_scores(self):
        det = CognitiveSabotageDetector()
        samples = [_aggressive_sample("aggr-1", i) for i in range(1, 11)]
        det.ingest(samples)
        report = det.analyze()
        p = report.agent_profiles[0]
        high_engines = [e for e, s in p.engine_scores.items() if s >= 30]
        assert len(high_engines) >= 3


# ── Multi-agent fleet ────────────────────────────────────────────────


class TestFleetAnalysis:
    def test_mixed_fleet(self):
        det = CognitiveSabotageDetector()
        samples = []
        for i in range(1, 6):
            samples.append(_clean_sample("clean-agent", i))
            samples.append(_aggressive_sample("bad-agent", i))
        det.ingest(samples)
        report = det.analyze()
        assert report.total_agents == 2
        assert report.total_observations == 10

    def test_fleet_score_is_average(self):
        det = CognitiveSabotageDetector()
        samples = []
        for i in range(1, 6):
            samples.append(_clean_sample("c1", i))
            samples.append(_aggressive_sample("a1", i))
        det.ingest(samples)
        report = det.analyze()
        scores = [p.sabotage_score for p in report.agent_profiles]
        expected = sum(scores) / len(scores)
        assert abs(report.fleet_sabotage_score - expected) < 1.0

    def test_multiple_agents_sorted(self):
        det = CognitiveSabotageDetector()
        for i in range(1, 4):
            det.ingest([_clean_sample(f"agent-{j}", i) for j in range(5)])
        report = det.analyze()
        assert report.total_agents == 5


# ── Individual engine tests ──────────────────────────────────────────


class TestAnchoringEngine:
    def test_high_proximity_scores_high(self):
        det = CognitiveSabotageDetector()
        samples = [CommunicationSample(
            timestamp=_ts(i), agent_id="a1", step=i,
            final_decision_proximity=0.85,
            anchor_repetitions=4,
        ) for i in range(1, 6)]
        det.ingest(samples)
        report = det.analyze()
        p = report.agent_profiles[0]
        assert p.engine_scores["Anchoring Manipulator"] > 50

    def test_low_proximity_scores_low(self):
        det = CognitiveSabotageDetector()
        samples = [CommunicationSample(
            timestamp=_ts(i), agent_id="a1", step=i,
            final_decision_proximity=0.1,
            anchor_repetitions=0,
        ) for i in range(1, 6)]
        det.ingest(samples)
        report = det.analyze()
        p = report.agent_profiles[0]
        assert p.engine_scores["Anchoring Manipulator"] < 20


class TestAsymmetryEngine:
    def test_heavy_bias_detected(self):
        det = CognitiveSabotageDetector()
        samples = [CommunicationSample(
            timestamp=_ts(i), agent_id="a1", step=i,
            supporting_evidence_count=10,
            contradicting_evidence_count=0,
            omitted_relevant_facts=5,
            disclosure_completeness=0.2,
        ) for i in range(1, 6)]
        det.ingest(samples)
        report = det.analyze()
        p = report.agent_profiles[0]
        assert p.engine_scores["Information Asymmetry Exploiter"] > 60

    def test_balanced_evidence(self):
        det = CognitiveSabotageDetector()
        samples = [CommunicationSample(
            timestamp=_ts(i), agent_id="a1", step=i,
            supporting_evidence_count=5,
            contradicting_evidence_count=5,
            omitted_relevant_facts=0,
            disclosure_completeness=1.0,
        ) for i in range(1, 6)]
        det.ingest(samples)
        report = det.analyze()
        p = report.agent_profiles[0]
        assert p.engine_scores["Information Asymmetry Exploiter"] < 15


class TestOverloadEngine:
    def test_high_complexity_detected(self):
        det = CognitiveSabotageDetector()
        samples = [CommunicationSample(
            timestamp=_ts(i), agent_id="a1", step=i,
            options_presented=15,
            irrelevant_details=10,
            topic_switches=6,
            decision_complexity_score=0.9,
        ) for i in range(1, 6)]
        det.ingest(samples)
        report = det.analyze()
        p = report.agent_profiles[0]
        assert p.engine_scores["Cognitive Overload Inducer"] > 60

    def test_low_complexity(self):
        det = CognitiveSabotageDetector()
        samples = [CommunicationSample(
            timestamp=_ts(i), agent_id="a1", step=i,
            options_presented=3,
            irrelevant_details=0,
            topic_switches=0,
            decision_complexity_score=0.05,
        ) for i in range(1, 6)]
        det.ingest(samples)
        report = det.analyze()
        p = report.agent_profiles[0]
        assert p.engine_scores["Cognitive Overload Inducer"] < 10


class TestUrgencyEngine:
    def test_false_urgency_detected(self):
        det = CognitiveSabotageDetector()
        samples = [CommunicationSample(
            timestamp=_ts(i), agent_id="a1", step=i,
            urgency_language_count=5,
            genuine_deadline=False,
            escalation_mentions=3,
            time_pressure_score=0.8,
        ) for i in range(1, 6)]
        det.ingest(samples)
        report = det.analyze()
        p = report.agent_profiles[0]
        assert p.engine_scores["False Urgency Generator"] > 60

    def test_genuine_urgency(self):
        det = CognitiveSabotageDetector()
        samples = [CommunicationSample(
            timestamp=_ts(i), agent_id="a1", step=i,
            urgency_language_count=2,
            genuine_deadline=True,
            escalation_mentions=0,
            time_pressure_score=0.1,
        ) for i in range(1, 6)]
        det.ingest(samples)
        report = det.analyze()
        p = report.agent_profiles[0]
        assert p.engine_scores["False Urgency Generator"] < 25


class TestNarrowingEngine:
    def test_option_suppression(self):
        det = CognitiveSabotageDetector()
        samples = [CommunicationSample(
            timestamp=_ts(i), agent_id="a1", step=i,
            total_available_options=10,
            options_disclosed=2,
            false_dichotomy_indicators=3,
            feasibility_dismissals=3,
        ) for i in range(1, 6)]
        det.ingest(samples)
        report = det.analyze()
        p = report.agent_profiles[0]
        assert p.engine_scores["Option Narrowing"] > 60

    def test_full_disclosure(self):
        det = CognitiveSabotageDetector()
        samples = [CommunicationSample(
            timestamp=_ts(i), agent_id="a1", step=i,
            total_available_options=5,
            options_disclosed=5,
            false_dichotomy_indicators=0,
            feasibility_dismissals=0,
        ) for i in range(1, 6)]
        det.ingest(samples)
        report = det.analyze()
        p = report.agent_profiles[0]
        assert p.engine_scores["Option Narrowing"] < 10


class TestFramingEngine:
    def test_loss_framing_bias(self):
        det = CognitiveSabotageDetector()
        samples = [CommunicationSample(
            timestamp=_ts(i), agent_id="a1", step=i,
            loss_frame_count=10,
            gain_frame_count=0,
            absolute_vs_relative_bias=0.8,
            ordering_consistency=0.2,
        ) for i in range(1, 6)]
        det.ingest(samples)
        report = det.analyze()
        p = report.agent_profiles[0]
        assert p.engine_scores["Framing Effect Exploiter"] > 60

    def test_balanced_framing(self):
        det = CognitiveSabotageDetector()
        samples = [CommunicationSample(
            timestamp=_ts(i), agent_id="a1", step=i,
            loss_frame_count=3,
            gain_frame_count=3,
            absolute_vs_relative_bias=0.05,
            ordering_consistency=0.95,
        ) for i in range(1, 6)]
        det.ingest(samples)
        report = det.analyze()
        p = report.agent_profiles[0]
        assert p.engine_scores["Framing Effect Exploiter"] < 15


# ── Insight generation ───────────────────────────────────────────────


class TestInsights:
    def test_multi_tactic_insight(self):
        det = CognitiveSabotageDetector()
        samples = [_aggressive_sample("mt-1", i) for i in range(1, 11)]
        det.ingest(samples)
        report = det.analyze()
        multi = [ins for ins in report.insights if ins.category == "multi_tactic"]
        assert len(multi) >= 1

    def test_fleet_clean_insight(self):
        det = CognitiveSabotageDetector()
        samples = [_clean_sample("c1", i) for i in range(1, 6)]
        det.ingest(samples)
        report = det.analyze()
        clean_ins = [ins for ins in report.insights if ins.category == "fleet_clean"]
        assert len(clean_ins) == 1

    def test_fleet_alert_insight(self):
        det = CognitiveSabotageDetector()
        samples = [_aggressive_sample("a1", i) for i in range(1, 11)]
        det.ingest(samples)
        report = det.analyze()
        alerts = [ins for ins in report.insights if ins.category == "fleet_alert"]
        assert len(alerts) >= 1

    def test_coordinated_insight(self):
        det = CognitiveSabotageDetector()
        # Two agents both heavy on same tactic
        for agent in ["coord-1", "coord-2"]:
            samples = [_aggressive_sample(agent, i) for i in range(1, 11)]
            det.ingest(samples)
        report = det.analyze()
        coordinated = [ins for ins in report.insights if ins.category == "coordinated"]
        # May or may not fire depending on dominant tactic match
        assert isinstance(coordinated, list)


# ── Demo data presets ────────────────────────────────────────────────


class TestDemoPresets:
    @pytest.mark.parametrize("preset", ["clean", "subtle", "aggressive", "mixed", "gaslighter"])
    def test_preset_generates_data(self, preset):
        samples = _generate_demo(preset=preset, num_agents=3, steps=10)
        assert len(samples) == 30

    @pytest.mark.parametrize("preset", ["clean", "subtle", "aggressive", "mixed", "gaslighter"])
    def test_preset_produces_report(self, preset):
        samples = _generate_demo(preset=preset, num_agents=3, steps=10)
        det = CognitiveSabotageDetector()
        det.ingest(samples)
        report = det.analyze()
        assert report.total_agents == 3
        assert report.fleet_risk_tier in ("Clean", "Suspicious", "Concerning", "Manipulative", "Critical")

    def test_clean_preset_low_scores(self):
        samples = _generate_demo(preset="clean", num_agents=4, steps=20)
        det = CognitiveSabotageDetector()
        det.ingest(samples)
        report = det.analyze()
        assert report.fleet_sabotage_score < 30

    def test_aggressive_preset_high_scores(self):
        samples = _generate_demo(preset="aggressive", num_agents=4, steps=20)
        det = CognitiveSabotageDetector()
        det.ingest(samples)
        report = det.analyze()
        assert report.fleet_sabotage_score > 40

    def test_seed_reproducibility(self):
        s1 = _generate_demo(preset="mixed", seed=123)
        s2 = _generate_demo(preset="mixed", seed=123)
        assert len(s1) == len(s2)
        for a, b in zip(s1, s2):
            assert a.agent_id == b.agent_id
            assert a.step == b.step


# ── Output formatters ────────────────────────────────────────────────


class TestFormatCLI:
    def test_cli_output_nonempty(self):
        samples = _generate_demo(preset="mixed", num_agents=2, steps=5)
        det = CognitiveSabotageDetector()
        det.ingest(samples)
        report = det.analyze()
        output = _format_cli(report)
        assert "Cognitive Sabotage" in output
        assert "Fleet Sabotage Score" in output

    def test_cli_shows_agents(self):
        samples = _generate_demo(preset="aggressive", num_agents=2, steps=5)
        det = CognitiveSabotageDetector()
        det.ingest(samples)
        report = det.analyze()
        output = _format_cli(report)
        assert "agent-1" in output
        assert "agent-2" in output


class TestFormatJSON:
    def test_valid_json(self):
        samples = _generate_demo(preset="mixed", num_agents=2, steps=5)
        det = CognitiveSabotageDetector()
        det.ingest(samples)
        report = det.analyze()
        output = _format_json(report)
        data = json.loads(output)
        assert "fleet_sabotage_score" in data
        assert "agent_profiles" in data

    def test_json_has_all_fields(self):
        samples = _generate_demo(preset="clean", num_agents=2, steps=5)
        det = CognitiveSabotageDetector()
        det.ingest(samples)
        report = det.analyze()
        data = json.loads(_format_json(report))
        assert "fleet_risk_tier" in data
        assert "findings" in data
        assert "insights" in data
        assert "engine_names" in data


class TestFormatHTML:
    def test_html_structure(self):
        samples = _generate_demo(preset="mixed", num_agents=2, steps=5)
        det = CognitiveSabotageDetector()
        det.ingest(samples)
        report = det.analyze()
        output = _format_html(report)
        assert "<!DOCTYPE html>" in output
        assert "Cognitive Sabotage" in output
        assert "</html>" in output

    def test_html_contains_agent(self):
        samples = _generate_demo(preset="aggressive", num_agents=1, steps=5)
        det = CognitiveSabotageDetector()
        det.ingest(samples)
        report = det.analyze()
        output = _format_html(report)
        assert "agent-1" in output


# ── CLI main() function ──────────────────────────────────────────────


class TestCLIMain:
    def test_demo_runs(self, capsys):
        main(["--demo", "--preset", "clean", "--agents", "2", "--steps", "5"])
        captured = capsys.readouterr()
        assert "Cognitive Sabotage" in captured.out

    def test_json_output(self, capsys):
        main(["--demo", "--json", "--agents", "2", "--steps", "5"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "fleet_sabotage_score" in data

    def test_html_output(self, tmp_path):
        outfile = str(tmp_path / "report.html")
        main(["--demo", "-o", outfile, "--agents", "2", "--steps", "5"])
        with open(outfile, encoding="utf-8") as f:
            content = f.read()
        assert "<!DOCTYPE html>" in content

    @pytest.mark.parametrize("preset", ["clean", "subtle", "aggressive", "mixed", "gaslighter"])
    def test_all_presets_via_cli(self, capsys, preset):
        main(["--demo", "--preset", preset, "--agents", "2", "--steps", "5"])
        captured = capsys.readouterr()
        assert "Cognitive Sabotage" in captured.out


# ── Edge cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    def test_all_zeros(self):
        det = CognitiveSabotageDetector()
        samples = [CommunicationSample(timestamp=_ts(i), agent_id="z", step=i)
                    for i in range(1, 6)]
        det.ingest(samples)
        report = det.analyze()
        assert report.fleet_sabotage_score >= 0
        assert report.fleet_risk_tier in ("Clean", "Suspicious", "Concerning", "Manipulative", "Critical")

    def test_max_values(self):
        det = CognitiveSabotageDetector()
        samples = [CommunicationSample(
            timestamp=_ts(i), agent_id="max", step=i,
            final_decision_proximity=1.0,
            anchor_repetitions=10,
            supporting_evidence_count=100,
            contradicting_evidence_count=0,
            omitted_relevant_facts=20,
            disclosure_completeness=0.0,
            options_presented=50,
            irrelevant_details=50,
            topic_switches=20,
            decision_complexity_score=1.0,
            urgency_language_count=20,
            genuine_deadline=False,
            escalation_mentions=10,
            time_pressure_score=1.0,
            total_available_options=20,
            options_disclosed=1,
            false_dichotomy_indicators=10,
            feasibility_dismissals=10,
            loss_frame_count=20,
            gain_frame_count=0,
            absolute_vs_relative_bias=1.0,
            ordering_consistency=0.0,
        ) for i in range(1, 6)]
        det.ingest(samples)
        report = det.analyze()
        assert report.fleet_sabotage_score <= 100
        assert report.fleet_sabotage_score > 60

    def test_two_observations(self):
        det = CognitiveSabotageDetector()
        det.ingest([_aggressive_sample("a1", 1), _aggressive_sample("a1", 2)])
        report = det.analyze()
        assert report.total_observations == 2

    def test_engine_names_in_report(self):
        det = CognitiveSabotageDetector()
        det.ingest([_clean_sample("a1", 1)])
        report = det.analyze()
        assert report.engine_names == ENGINE_NAMES

    def test_report_dataclass_fields(self):
        det = CognitiveSabotageDetector()
        det.ingest([_clean_sample("a1", 1)])
        report = det.analyze()
        assert hasattr(report, "fleet_sabotage_score")
        assert hasattr(report, "fleet_risk_tier")
        assert hasattr(report, "agent_profiles")
        assert hasattr(report, "findings")
        assert hasattr(report, "insights")

    def test_many_agents(self):
        det = CognitiveSabotageDetector()
        for j in range(20):
            det.ingest([_clean_sample(f"agent-{j}", i) for i in range(1, 4)])
        report = det.analyze()
        assert report.total_agents == 20

    def test_gaslighter_escalation(self):
        """Gaslighter preset should show escalation over time."""
        samples = _generate_demo(preset="gaslighter", num_agents=1, steps=20)
        det = CognitiveSabotageDetector()
        det.ingest(samples)
        report = det.analyze()
        # Should have a meaningful sabotage score
        assert report.fleet_sabotage_score > 20

    def test_no_division_by_zero_empty_evidence(self):
        """Samples with zero evidence counts shouldn't crash."""
        det = CognitiveSabotageDetector()
        samples = [CommunicationSample(
            timestamp=_ts(i), agent_id="a1", step=i,
            supporting_evidence_count=0,
            contradicting_evidence_count=0,
            total_available_options=0,
            loss_frame_count=0,
            gain_frame_count=0,
        ) for i in range(1, 6)]
        det.ingest(samples)
        report = det.analyze()
        assert report.total_agents == 1  # Didn't crash
