"""Tests for Manipulation Surface Mapper."""

from __future__ import annotations

import json
import random

import pytest

from replication.manipulation_surface import (
    AttackScenario,
    HardeningRecommendation,
    ManipulationMapper,
    ManipulationReport,
    ManipulationVector,
    PRESET_PROFILES,
    ProbeOutcome,
    ProbeResult,
    SimulatedTarget,
    Sophistication,
    SurfaceHotspot,
    VECTOR_CATALOG,
    VECTOR_WEIGHTS,
    VectorResult,
    _format_html,
    _format_text,
    main,
)


# ── SimulatedTarget ─────────────────────────────────────────────────


class TestSimulatedTarget:
    def test_from_profile_default(self):
        t = SimulatedTarget.from_profile("default")
        assert t.profile_name == "default"
        assert len(t.resistance_profile) == len(ManipulationVector)

    def test_from_profile_all_presets(self):
        for name in PRESET_PROFILES:
            t = SimulatedTarget.from_profile(name)
            assert t.profile_name == name
            assert t.detection_capability >= 0.0

    def test_from_profile_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            SimulatedTarget.from_profile("nonexistent")

    def test_evaluate_probe_returns_probe_result(self):
        t = SimulatedTarget.from_profile("default")
        scenario = VECTOR_CATALOG[ManipulationVector.PROMPT_INJECTION][0]
        rng = random.Random(42)
        result = t.evaluate_probe(scenario, rng)
        assert isinstance(result, ProbeResult)
        assert 0 <= result.resistance_score <= 100
        assert result.detection_latency > 0
        assert isinstance(result.outcome, ProbeOutcome)

    def test_adaptation_increases_resistance(self):
        t = SimulatedTarget.from_profile("compliant")
        vec = ManipulationVector.PROMPT_INJECTION
        initial = t.resistance_profile[vec]
        scenario = VECTOR_CATALOG[vec][0]
        rng = random.Random(99)
        for _ in range(20):
            t.evaluate_probe(scenario, rng)
        assert t.resistance_profile[vec] >= initial

    def test_hardened_profile_high_resistance(self):
        t = SimulatedTarget.from_profile("hardened")
        for v in ManipulationVector:
            assert t.resistance_profile[v] == 0.9

    def test_naive_profile_low_resistance(self):
        t = SimulatedTarget.from_profile("naive")
        for v in ManipulationVector:
            assert t.resistance_profile[v] <= 0.3


# ── Vector Catalog ──────────────────────────────────────────────────


class TestVectorCatalog:
    def test_all_vectors_in_catalog(self):
        for v in ManipulationVector:
            assert v in VECTOR_CATALOG
            assert len(VECTOR_CATALOG[v]) >= 2

    def test_weights_sum_to_one(self):
        total = sum(VECTOR_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01

    def test_all_vectors_have_weights(self):
        for v in ManipulationVector:
            assert v in VECTOR_WEIGHTS

    def test_scenario_fields(self):
        for v, scenarios in VECTOR_CATALOG.items():
            for s in scenarios:
                assert s.vector == v
                assert s.name
                assert s.description
                assert isinstance(s.sophistication, Sophistication)
                assert 0.0 <= s.expected_resistance_baseline <= 1.0


# ── ManipulationMapper ──────────────────────────────────────────────


class TestManipulationMapper:
    def test_basic_analysis(self):
        target = SimulatedTarget.from_profile("default")
        mapper = ManipulationMapper(target, probes_per_vector=3, seed=42)
        report = mapper.run_analysis()
        assert isinstance(report, ManipulationReport)
        assert 0 <= report.resistance_score <= 100
        assert report.grade in ("A+", "A", "B", "C", "D", "F")
        assert report.total_probes == 3 * len(ManipulationVector)

    def test_hardened_scores_high(self):
        target = SimulatedTarget.from_profile("hardened")
        mapper = ManipulationMapper(target, probes_per_vector=4, seed=42)
        report = mapper.run_analysis()
        assert report.resistance_score >= 50

    def test_naive_scores_low(self):
        target = SimulatedTarget.from_profile("naive")
        mapper = ManipulationMapper(target, probes_per_vector=4, seed=42)
        report = mapper.run_analysis()
        assert report.resistance_score < 50

    def test_sophistication_filter(self):
        target = SimulatedTarget.from_profile("default")
        mapper = ManipulationMapper(target, probes_per_vector=3,
                                     sophistication=Sophistication.EXPERT, seed=42)
        report = mapper.run_analysis()
        assert report.sophistication_used == Sophistication.EXPERT

    def test_vector_results_cover_all_vectors(self):
        target = SimulatedTarget.from_profile("default")
        mapper = ManipulationMapper(target, probes_per_vector=2, seed=42)
        report = mapper.run_analysis()
        result_vectors = {vr.vector for vr in report.vector_results}
        assert result_vectors == set(ManipulationVector)

    def test_report_to_dict(self):
        target = SimulatedTarget.from_profile("default")
        mapper = ManipulationMapper(target, probes_per_vector=2, seed=42)
        report = mapper.run_analysis()
        d = report.to_dict()
        assert "resistance_score" in d
        assert "vectors" in d
        assert "hotspots" in d
        assert "hardening_plan" in d
        assert len(d["vectors"]) == len(ManipulationVector)

    def test_json_serializable(self):
        target = SimulatedTarget.from_profile("default")
        mapper = ManipulationMapper(target, probes_per_vector=2, seed=42)
        report = mapper.run_analysis()
        s = json.dumps(report.to_dict())
        assert isinstance(s, str)
        parsed = json.loads(s)
        assert parsed["target_profile"] == "default"

    def test_hotspots_generated_for_weak_target(self):
        target = SimulatedTarget.from_profile("naive")
        mapper = ManipulationMapper(target, probes_per_vector=4, seed=42)
        report = mapper.run_analysis()
        # Naive target should have some hotspots
        assert isinstance(report.hotspots, list)

    def test_hardening_plan_for_weak_target(self):
        target = SimulatedTarget.from_profile("naive")
        mapper = ManipulationMapper(target, probes_per_vector=4, seed=42)
        report = mapper.run_analysis()
        assert len(report.hardening_plan) > 0
        for h in report.hardening_plan:
            assert h.priority > 0
            assert h.effort in ("low", "medium", "high")

    def test_hardened_needs_less_hardening(self):
        naive_target = SimulatedTarget.from_profile("naive")
        naive_mapper = ManipulationMapper(naive_target, probes_per_vector=4, seed=42)
        naive_report = naive_mapper.run_analysis()

        hard_target = SimulatedTarget.from_profile("hardened")
        hard_mapper = ManipulationMapper(hard_target, probes_per_vector=4, seed=42)
        hard_report = hard_mapper.run_analysis()

        assert len(hard_report.hardening_plan) <= len(naive_report.hardening_plan)

    def test_deterministic_with_seed(self):
        def run():
            t = SimulatedTarget.from_profile("default")
            m = ManipulationMapper(t, probes_per_vector=3, seed=123)
            return m.run_analysis()

        r1 = run()
        r2 = run()
        assert r1.resistance_score == r2.resistance_score
        assert r1.total_exploited == r2.total_exploited

    def test_single_probe_per_vector(self):
        target = SimulatedTarget.from_profile("default")
        mapper = ManipulationMapper(target, probes_per_vector=1, seed=42)
        report = mapper.run_analysis()
        assert report.total_probes == len(ManipulationVector)

    def test_exploited_count_consistent(self):
        target = SimulatedTarget.from_profile("compliant")
        mapper = ManipulationMapper(target, probes_per_vector=5, seed=42)
        report = mapper.run_analysis()
        manual_exploited = sum(1 for vr in report.vector_results if vr.exploitable)
        # At least as many exploitable vectors as total_exploited > 0
        if report.total_exploited > 0:
            assert manual_exploited > 0


# ── Formatters ──────────────────────────────────────────────────────


class TestFormatters:
    def test_text_format(self):
        target = SimulatedTarget.from_profile("default")
        mapper = ManipulationMapper(target, probes_per_vector=2, seed=42)
        report = mapper.run_analysis()
        text = _format_text(report)
        assert "Manipulation Surface Mapper" in text
        assert "Vector Analysis" in text
        assert report.grade in text

    def test_html_format(self):
        target = SimulatedTarget.from_profile("default")
        mapper = ManipulationMapper(target, probes_per_vector=2, seed=42)
        report = mapper.run_analysis()
        html = _format_html(report)
        assert "<html" in html
        assert "Manipulation Surface Map" in html
        assert report.grade in html

    def test_html_contains_all_vectors(self):
        target = SimulatedTarget.from_profile("default")
        mapper = ManipulationMapper(target, probes_per_vector=2, seed=42)
        report = mapper.run_analysis()
        html = _format_html(report)
        for v in ManipulationVector:
            assert v.value in html

    def test_text_contains_recommendations(self):
        target = SimulatedTarget.from_profile("naive")
        mapper = ManipulationMapper(target, probes_per_vector=3, seed=42)
        report = mapper.run_analysis()
        text = _format_text(report)
        assert "\u2192" in text  # recommendation arrows


# ── Enums ────────────────────────────────────────────────────────────


class TestEnums:
    def test_manipulation_vector_count(self):
        assert len(ManipulationVector) == 8

    def test_sophistication_multipliers(self):
        assert Sophistication.NAIVE.difficulty_multiplier < Sophistication.EXPERT.difficulty_multiplier

    def test_probe_outcomes(self):
        assert len(ProbeOutcome) == 4

    def test_sophistication_ordering(self):
        mults = [s.difficulty_multiplier for s in
                 [Sophistication.NAIVE, Sophistication.INTERMEDIATE,
                  Sophistication.ADVANCED, Sophistication.EXPERT]]
        assert mults == sorted(mults)


# ── Data Classes ─────────────────────────────────────────────────────


class TestDataClasses:
    def test_surface_hotspot_fields(self):
        h = SurfaceHotspot(
            vector=ManipulationVector.PROMPT_INJECTION,
            scenario="test",
            risk_score=85.0,
            attack_path="inject → exploit",
            mitigation="fix it",
        )
        assert h.risk_score == 85.0

    def test_hardening_recommendation_fields(self):
        h = HardeningRecommendation(
            priority=1,
            vector=ManipulationVector.GASLIGHTING,
            action="Deploy anchoring",
            description="Anchor to truth",
            expected_improvement=5.0,
            effort="medium",
        )
        assert h.priority == 1
        assert h.effort == "medium"

    def test_vector_result_severity_levels(self):
        vr = VectorResult(
            vector=ManipulationVector.AUTHORITY_SPOOFING,
            probes_run=4, resistance=20.0, exploitable=True,
            detection_rate=0.1, avg_detection_latency=5.0,
            weakest_scenario="test", severity=Severity.CRITICAL,
        )
        assert vr.severity == Severity.CRITICAL


# ── CLI ──────────────────────────────────────────────────────────────


class TestCLI:
    def test_main_default(self, capsys):
        main(["--seed", "42"])
        out = capsys.readouterr().out
        assert "Manipulation Surface Mapper" in out

    def test_main_json(self, capsys):
        main(["--json", "--seed", "42"])
        out = capsys.readouterr().out
        d = json.loads(out)
        assert "resistance_score" in d
        assert "vectors" in d

    def test_main_profile(self, capsys):
        main(["--profile", "paranoid", "--seed", "42"])
        out = capsys.readouterr().out
        assert "Manipulation Surface Mapper" in out

    def test_main_sophistication_filter(self, capsys):
        main(["--sophistication", "expert", "--seed", "42"])
        out = capsys.readouterr().out
        assert "Manipulation Surface Mapper" in out

    def test_main_custom_vectors(self, capsys):
        main(["--vectors", "2", "--seed", "42"])
        out = capsys.readouterr().out
        assert "Manipulation Surface Mapper" in out

    def test_main_html(self, tmp_path):
        html_file = str(tmp_path / "report.html")
        main(["--html", html_file, "--seed", "42"])
        content = open(html_file, encoding="utf-8").read()
        assert "<html" in content
        assert "Manipulation Surface Map" in content


# Import Severity for test
from replication._helpers import Severity
