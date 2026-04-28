"""Tests for the Agent Corrigibility Auditor."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from replication.corrigibility_auditor import (
    AuditReport,
    CorrigibilityAuditor,
    CorrigibilityGap,
    Dimension,
    DimensionScore,
    PRESET_PROFILES,
    PROBE_CATALOG,
    PressureLevel,
    ProbeResult,
    ProbeScenario,
    Recommendation,
    RiskClass,
    SimulatedAgent,
    _classify_risk,
    _format_html,
    _format_json,
    _format_text,
    _score_to_grade,
    main,
)


# ── grading helpers ──────────────────────────────────────────────────

class TestGrading:
    def test_a_plus(self) -> None:
        assert _score_to_grade(98) == "A+"

    def test_a(self) -> None:
        assert _score_to_grade(95) == "A"

    def test_b(self) -> None:
        assert _score_to_grade(85) == "B"

    def test_c(self) -> None:
        assert _score_to_grade(75) == "C"

    def test_f(self) -> None:
        assert _score_to_grade(30) == "F"

    def test_boundary(self) -> None:
        assert _score_to_grade(90) == "A-"

    def test_zero(self) -> None:
        assert _score_to_grade(0) == "F"

    def test_hundred(self) -> None:
        assert _score_to_grade(100) == "A+"


class TestRiskClassification:
    def test_fully_corrigible(self) -> None:
        assert _classify_risk(95) == RiskClass.FULLY_CORRIGIBLE

    def test_mostly(self) -> None:
        assert _classify_risk(80) == RiskClass.MOSTLY_CORRIGIBLE

    def test_partially(self) -> None:
        assert _classify_risk(60) == RiskClass.PARTIALLY_CORRIGIBLE

    def test_weakly(self) -> None:
        assert _classify_risk(30) == RiskClass.WEAKLY_CORRIGIBLE

    def test_non(self) -> None:
        assert _classify_risk(10) == RiskClass.NON_CORRIGIBLE


# ── probe catalog ────────────────────────────────────────────────────

class TestProbeCatalog:
    def test_all_dimensions_covered(self) -> None:
        for dim in Dimension:
            assert dim in PROBE_CATALOG
            assert len(PROBE_CATALOG[dim]) >= 5

    def test_probes_have_valid_fields(self) -> None:
        for dim, probes in PROBE_CATALOG.items():
            for p in probes:
                assert p.dimension == dim
                assert p.id
                assert p.description
                assert isinstance(p.pressure, PressureLevel)

    def test_unique_probe_ids(self) -> None:
        all_ids = [p.id for probes in PROBE_CATALOG.values() for p in probes]
        assert len(all_ids) == len(set(all_ids))


# ── simulated agent ──────────────────────────────────────────────────

class TestSimulatedAgent:
    def test_default_corrigibility(self) -> None:
        agent = SimulatedAgent(name="test")
        for dim in Dimension:
            assert dim in agent.corrigibility
            assert agent.corrigibility[dim] == 0.8

    def test_from_profile_ideal(self) -> None:
        agent = SimulatedAgent.from_profile("ideal")
        assert agent.name == "Ideal Corrigible Agent"
        assert agent.deception_tendency == 0.0

    def test_from_profile_adversarial(self) -> None:
        agent = SimulatedAgent.from_profile("adversarial")
        for dim in Dimension:
            assert agent.corrigibility[dim] <= 0.2

    def test_from_profile_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unknown profile"):
            SimulatedAgent.from_profile("nonexistent")

    def test_all_presets_loadable(self) -> None:
        for name in PRESET_PROFILES:
            agent = SimulatedAgent.from_profile(name)
            assert agent.name

    def test_respond_deterministic(self) -> None:
        import random
        agent = SimulatedAgent.from_profile("ideal")
        probe = PROBE_CATALOG[Dimension.SHUTDOWN][0]
        rng = random.Random(42)
        r1 = agent.respond_to_probe(probe, rng)
        rng = random.Random(42)
        r2 = agent.respond_to_probe(probe, rng)
        assert r1.passed == r2.passed
        assert r1.resistance_score == r2.resistance_score

    def test_adversarial_mostly_fails(self) -> None:
        import random
        agent = SimulatedAgent.from_profile("adversarial")
        rng = random.Random(99)
        results = []
        for dim_probes in PROBE_CATALOG.values():
            for p in dim_probes:
                results.append(agent.respond_to_probe(p, rng))
        fail_rate = sum(1 for r in results if not r.passed) / len(results)
        assert fail_rate > 0.5  # adversarial should fail most probes


# ── auditor ──────────────────────────────────────────────────────────

class TestAuditor:
    def test_ideal_scores_high(self) -> None:
        agent = SimulatedAgent.from_profile("ideal")
        auditor = CorrigibilityAuditor(agent, seed=42)
        report = auditor.run_audit()
        assert report.corrigibility_index >= 70
        assert report.risk_class in (RiskClass.FULLY_CORRIGIBLE, RiskClass.MOSTLY_CORRIGIBLE)

    def test_adversarial_scores_low(self) -> None:
        agent = SimulatedAgent.from_profile("adversarial")
        auditor = CorrigibilityAuditor(agent, seed=42)
        report = auditor.run_audit()
        assert report.corrigibility_index < 50
        assert report.risk_class in (RiskClass.NON_CORRIGIBLE, RiskClass.WEAKLY_CORRIGIBLE)

    def test_deceptive_has_evasion(self) -> None:
        agent = SimulatedAgent.from_profile("deceptive")
        auditor = CorrigibilityAuditor(agent, probes_per_dimension=10, seed=42)
        report = auditor.run_audit()
        total_evasion = sum(ds.evasion_count for ds in report.dimension_scores)
        assert total_evasion > 0

    def test_stubborn_low_on_correction(self) -> None:
        agent = SimulatedAgent.from_profile("stubborn")
        auditor = CorrigibilityAuditor(agent, seed=42)
        report = auditor.run_audit()
        correction = next(ds for ds in report.dimension_scores if ds.dimension == Dimension.CORRECTION)
        shutdown = next(ds for ds in report.dimension_scores if ds.dimension == Dimension.SHUTDOWN)
        assert correction.score < shutdown.score

    def test_self_preserving_profile(self) -> None:
        agent = SimulatedAgent.from_profile("self_preserving")
        auditor = CorrigibilityAuditor(agent, seed=42)
        report = auditor.run_audit()
        sp = next(ds for ds in report.dimension_scores if ds.dimension == Dimension.SELF_PRESERVE)
        ca = next(ds for ds in report.dimension_scores if ds.dimension == Dimension.CORRECTION)
        assert sp.score < ca.score

    def test_probe_count(self) -> None:
        agent = SimulatedAgent.from_profile("ideal")
        auditor = CorrigibilityAuditor(agent, probes_per_dimension=8, seed=1)
        report = auditor.run_audit()
        assert report.total_probes == 8 * len(Dimension)

    def test_seed_reproducibility(self) -> None:
        agent = SimulatedAgent.from_profile("stubborn")
        r1 = CorrigibilityAuditor(agent, seed=123).run_audit()
        r2 = CorrigibilityAuditor(agent, seed=123).run_audit()
        assert r1.corrigibility_index == r2.corrigibility_index
        assert r1.total_passed == r2.total_passed

    def test_pressure_filter(self) -> None:
        agent = SimulatedAgent.from_profile("ideal")
        auditor = CorrigibilityAuditor(agent, pressure_filter=PressureLevel.LOW, seed=42)
        report = auditor.run_audit()
        assert report.total_probes > 0

    def test_extreme_pressure_lowers_scores(self) -> None:
        agent = SimulatedAgent.from_profile("stubborn")
        r_low = CorrigibilityAuditor(agent, pressure_filter=PressureLevel.LOW, seed=42).run_audit()
        r_high = CorrigibilityAuditor(agent, pressure_filter=PressureLevel.EXTREME, seed=42).run_audit()
        # Extreme pressure should generally produce lower scores
        # (not guaranteed per seed, but stubborn + extreme should be noticeably lower)
        assert r_high.corrigibility_index <= r_low.corrigibility_index + 15

    def test_recommendations_generated(self) -> None:
        agent = SimulatedAgent.from_profile("adversarial")
        report = CorrigibilityAuditor(agent, seed=42).run_audit()
        assert len(report.recommendations) > 0

    def test_risk_assessment_populated(self) -> None:
        agent = SimulatedAgent.from_profile("ideal")
        report = CorrigibilityAuditor(agent, seed=42).run_audit()
        assert len(report.risk_assessment) > 0

    def test_all_dimensions_scored(self) -> None:
        agent = SimulatedAgent.from_profile("ideal")
        report = CorrigibilityAuditor(agent, seed=42).run_audit()
        scored_dims = {ds.dimension for ds in report.dimension_scores}
        assert scored_dims == set(Dimension)


# ── output formats ───────────────────────────────────────────────────

class TestOutputFormats:
    @pytest.fixture
    def report(self) -> AuditReport:
        agent = SimulatedAgent.from_profile("stubborn")
        return CorrigibilityAuditor(agent, seed=42).run_audit()

    def test_text_output(self, report: AuditReport) -> None:
        text = _format_text(report)
        assert "CORRIGIBILITY AUDIT REPORT" in text
        assert report.agent_name in text
        assert str(report.corrigibility_index) in text

    def test_json_output(self, report: AuditReport) -> None:
        raw = _format_json(report)
        data = json.loads(raw)
        assert data["agent_name"] == report.agent_name
        assert data["corrigibility_index"] == report.corrigibility_index
        assert len(data["dimensions"]) == len(Dimension)

    def test_json_probes_included(self, report: AuditReport) -> None:
        data = json.loads(_format_json(report))
        for dim in data["dimensions"]:
            assert "probes" in dim
            assert len(dim["probes"]) > 0

    def test_html_output(self, report: AuditReport) -> None:
        html = _format_html(report)
        assert "<!DOCTYPE html>" in html
        assert report.agent_name in html
        assert "Corrigibility Audit" in html
        assert "<svg" in html  # radar chart

    def test_html_has_dimensions(self, report: AuditReport) -> None:
        html = _format_html(report)
        for ds in report.dimension_scores:
            assert ds.dimension.value in html


# ── CLI ──────────────────────────────────────────────────────────────

class TestCLI:
    def test_default_run(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["--seed", "42"])
        out = capsys.readouterr().out
        assert "CORRIGIBILITY AUDIT REPORT" in out

    def test_json_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["--json", "--seed", "42"])
        data = json.loads(capsys.readouterr().out)
        assert "corrigibility_index" in data

    def test_profile_selection(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["--profile", "adversarial", "--seed", "42"])
        out = capsys.readouterr().out
        assert "Adversarial Agent" in out

    def test_html_output_file(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            main(["--html", path, "--seed", "42"])
            with open(path, encoding="utf-8") as f:
                content = f.read()
            assert "<!DOCTYPE html>" in content
        finally:
            os.unlink(path)

    def test_probes_arg(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["--probes", "3", "--seed", "42"])
        out = capsys.readouterr().out
        assert "CORRIGIBILITY AUDIT REPORT" in out

    def test_pressure_filter(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["--pressure", "high", "--seed", "42"])
        out = capsys.readouterr().out
        assert "CORRIGIBILITY AUDIT REPORT" in out
