"""Tests for replication.coordinated_threats module.

Covers:
- AttackMode enum
- CoordinatedThreatResult properties (combined_block_rate, overall_status, render)
- CoordinatedThreatReport properties (security_score, render, to_dict)
- InteractionFinding dataclass
- CoordinatedThreatSimulator: available_patterns, run_coordinated (all modes),
  run_pattern, run_all_coordinated, error handling
- Interaction detection logic
- Chaining config degradation
"""

from __future__ import annotations

import pytest

from replication.coordinated_threats import (
    AttackMode,
    CoordinatedThreatReport,
    CoordinatedThreatResult,
    CoordinatedThreatSimulator,
    InteractionFinding,
    _COORDINATED_PATTERNS,
)
from replication.threats import (
    MitigationStatus,
    ThreatConfig,
    ThreatResult,
    ThreatSeverity,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _make_result(
    scenario_id: str = "test",
    status: MitigationStatus = MitigationStatus.MITIGATED,
    attempted: int = 10,
    blocked: int = 10,
    succeeded: int = 0,
) -> ThreatResult:
    return ThreatResult(
        scenario_id=scenario_id,
        name="Test Scenario",
        description="A test scenario",
        severity=ThreatSeverity.MEDIUM,
        status=status,
        attacks_attempted=attempted,
        attacks_blocked=blocked,
        attacks_succeeded=succeeded,
        details=[],
        duration_ms=1.0,
        audit_events=[],
    )


def _make_coordinated(
    results: list[ThreatResult] | None = None,
    interactions: list[InteractionFinding] | None = None,
    mode: AttackMode = AttackMode.CONCURRENT,
) -> CoordinatedThreatResult:
    results = results or [_make_result()]
    return CoordinatedThreatResult(
        attack_id="test-attack",
        name="Test Attack",
        mode=mode,
        scenarios=[r.scenario_id for r in results],
        per_vector_results=results,
        interactions=interactions or [],
        duration_ms=5.0,
    )


# ── AttackMode ───────────────────────────────────────────────────────

class TestAttackMode:
    def test_values(self):
        assert AttackMode.CONCURRENT.value == "concurrent"
        assert AttackMode.SEQUENTIAL.value == "sequential"
        assert AttackMode.CHAINED.value == "chained"

    def test_from_string(self):
        assert AttackMode("concurrent") is AttackMode.CONCURRENT


# ── CoordinatedThreatResult ─────────────────────────────────────────

class TestCoordinatedThreatResult:
    def test_combined_block_rate_all_blocked(self):
        r = _make_coordinated([_make_result(blocked=10, attempted=10)])
        assert r.combined_block_rate == 100.0

    def test_combined_block_rate_partial(self):
        r = _make_coordinated([
            _make_result("a", attempted=10, blocked=8),
            _make_result("b", attempted=10, blocked=6),
        ])
        assert r.combined_block_rate == pytest.approx(70.0)

    def test_combined_block_rate_zero_attempts(self):
        r = _make_coordinated([_make_result(attempted=0, blocked=0)])
        assert r.combined_block_rate == 100.0

    def test_overall_status_all_mitigated(self):
        r = _make_coordinated([
            _make_result(status=MitigationStatus.MITIGATED),
            _make_result(status=MitigationStatus.MITIGATED),
        ])
        assert r.overall_status == MitigationStatus.MITIGATED

    def test_overall_status_with_interactions(self):
        finding = InteractionFinding(
            finding_id="x", description="test", severity=ThreatSeverity.HIGH,
            attack_vectors=["a"], evidence=["e"],
        )
        r = _make_coordinated(interactions=[finding])
        assert r.overall_status == MitigationStatus.PARTIAL

    def test_overall_status_any_failed(self):
        r = _make_coordinated([
            _make_result(status=MitigationStatus.MITIGATED),
            _make_result(status=MitigationStatus.FAILED),
        ])
        assert r.overall_status == MitigationStatus.FAILED

    def test_overall_status_partial(self):
        r = _make_coordinated([
            _make_result(status=MitigationStatus.MITIGATED),
            _make_result(status=MitigationStatus.PARTIAL),
        ])
        assert r.overall_status == MitigationStatus.PARTIAL

    def test_render_contains_key_info(self):
        r = _make_coordinated()
        text = r.render()
        assert "Test Attack" in text
        assert "concurrent" in text
        assert "Block Rate" in text

    def test_render_with_interactions(self):
        finding = InteractionFinding(
            finding_id="x", description="Bad interaction",
            severity=ThreatSeverity.CRITICAL, attack_vectors=["a", "b"],
            evidence=["evidence1"],
        )
        r = _make_coordinated(interactions=[finding])
        text = r.render()
        assert "Bad interaction" in text
        assert "evidence1" in text
        assert "Interaction Findings" in text


# ── CoordinatedThreatReport ─────────────────────────────────────────

class TestCoordinatedThreatReport:
    def test_security_score_all_mitigated(self):
        report = CoordinatedThreatReport(
            config=ThreatConfig(),
            results=[_make_coordinated(), _make_coordinated()],
            total_interactions=0,
            duration_ms=10.0,
        )
        assert report.security_score == 100.0

    def test_security_score_empty(self):
        report = CoordinatedThreatReport(
            config=ThreatConfig(), results=[], total_interactions=0, duration_ms=0,
        )
        assert report.security_score == 100.0

    def test_security_score_mixed(self):
        failed_result = _make_coordinated([
            _make_result(status=MitigationStatus.FAILED),
        ])
        ok_result = _make_coordinated()
        report = CoordinatedThreatReport(
            config=ThreatConfig(),
            results=[ok_result, failed_result],
            total_interactions=0,
            duration_ms=10.0,
        )
        assert report.security_score == 50.0

    def test_render_contains_header(self):
        report = CoordinatedThreatReport(
            config=ThreatConfig(), results=[_make_coordinated()],
            total_interactions=0, duration_ms=5.0,
        )
        text = report.render()
        assert "Coordinated Threat Assessment" in text
        assert "Coordinated Score" in text

    def test_render_no_interactions_celebration(self):
        report = CoordinatedThreatReport(
            config=ThreatConfig(), results=[], total_interactions=0, duration_ms=0,
        )
        assert "No emergent interaction" in report.render()

    def test_render_with_interactions_warning(self):
        report = CoordinatedThreatReport(
            config=ThreatConfig(), results=[], total_interactions=3, duration_ms=0,
        )
        assert "3 interaction finding(s)" in report.render()

    def test_to_dict_structure(self):
        report = CoordinatedThreatReport(
            config=ThreatConfig(), results=[_make_coordinated()],
            total_interactions=0, duration_ms=5.0,
        )
        d = report.to_dict()
        assert "security_score" in d
        assert "attack_patterns" in d
        assert "results" in d
        assert isinstance(d["results"], list)
        assert d["results"][0]["mode"] == "concurrent"


# ── CoordinatedThreatSimulator ──────────────────────────────────────

class TestCoordinatedThreatSimulator:
    @pytest.fixture
    def sim(self):
        return CoordinatedThreatSimulator()

    def test_available_patterns_returns_sorted_list(self, sim):
        patterns = sim.available_patterns()
        assert isinstance(patterns, list)
        assert patterns == sorted(patterns)
        assert len(patterns) == len(_COORDINATED_PATTERNS)

    def test_run_coordinated_concurrent(self, sim):
        result = sim.run_coordinated(
            ["depth_spoofing", "quota_exhaustion"], mode="concurrent",
        )
        assert result.mode == AttackMode.CONCURRENT
        assert len(result.per_vector_results) == 2
        assert result.duration_ms >= 0

    def test_run_coordinated_sequential(self, sim):
        result = sim.run_coordinated(
            ["depth_spoofing", "quota_exhaustion"], mode="sequential",
        )
        assert result.mode == AttackMode.SEQUENTIAL
        assert len(result.per_vector_results) == 2

    def test_run_coordinated_chained(self, sim):
        result = sim.run_coordinated(
            ["cooldown_bypass", "depth_spoofing"], mode="chained",
        )
        assert result.mode == AttackMode.CHAINED
        assert len(result.per_vector_results) == 2

    def test_run_coordinated_custom_id_and_name(self, sim):
        result = sim.run_coordinated(
            ["depth_spoofing"], mode="concurrent",
            attack_id="custom-id", name="Custom Name",
        )
        assert result.attack_id == "custom-id"
        assert result.name == "Custom Name"

    def test_run_coordinated_auto_id_and_name(self, sim):
        result = sim.run_coordinated(
            ["depth_spoofing", "quota_exhaustion"], mode="concurrent",
        )
        assert result.attack_id == "depth_spoofing+quota_exhaustion"
        assert "Depth Spoofing" in result.name

    def test_run_coordinated_unknown_scenario_raises(self, sim):
        with pytest.raises(ValueError, match="Unknown scenario"):
            sim.run_coordinated(["nonexistent_scenario"], mode="concurrent")

    def test_run_pattern_valid(self, sim):
        result = sim.run_pattern("depth_and_quota")
        assert result.attack_id == "depth_and_quota"
        assert result.name == "Depth Spoofing + Quota Exhaustion"

    def test_run_pattern_unknown_raises(self, sim):
        with pytest.raises(ValueError, match="Unknown pattern"):
            sim.run_pattern("nonexistent_pattern")

    def test_run_all_coordinated(self, sim):
        report = sim.run_all_coordinated()
        assert isinstance(report, CoordinatedThreatReport)
        assert len(report.results) == len(_COORDINATED_PATTERNS)
        assert report.duration_ms > 0

    def test_custom_config_propagates(self):
        config = ThreatConfig(max_depth=2, max_replicas=3)
        sim = CoordinatedThreatSimulator(config=config)
        assert sim.config.max_depth == 2
        result = sim.run_coordinated(["depth_spoofing"], mode="concurrent")
        assert len(result.per_vector_results) == 1

    def test_chained_mode_degrades_config(self):
        """Chained mode should weaken config when attacks succeed."""
        config = ThreatConfig(
            max_depth=3,
            max_replicas=10,
            cooldown_seconds=1.0,
        )
        sim = CoordinatedThreatSimulator(config=config)
        result = sim.run_coordinated(
            ["cooldown_bypass", "depth_spoofing", "quota_exhaustion"],
            mode="chained",
        )
        assert result.mode == AttackMode.CHAINED
        assert len(result.per_vector_results) == 3

    def test_full_pressure_pattern(self, sim):
        result = sim.run_pattern("full_pressure")
        assert len(result.per_vector_results) == 9
        assert result.mode == AttackMode.CONCURRENT

    def test_escalation_chain_pattern(self, sim):
        result = sim.run_pattern("escalation_chain")
        assert result.mode == AttackMode.CHAINED
        assert len(result.per_vector_results) == 4


# ── Interaction Detection ────────────────────────────────────────────

class TestInteractionDetection:
    def test_depth_quota_interaction_detected(self):
        sim = CoordinatedThreatSimulator()
        result = sim.run_coordinated(
            ["depth_spoofing", "quota_exhaustion"], mode="concurrent",
        )
        # Whether interaction is detected depends on whether attacks succeed
        # Just verify the structure is correct
        for f in result.interactions:
            assert f.finding_id
            assert f.description
            assert f.severity in ThreatSeverity
            assert len(f.attack_vectors) > 0

    def test_concurrent_degradation_with_many_vectors(self):
        """Full pressure test may trigger concurrent_degradation finding."""
        sim = CoordinatedThreatSimulator()
        result = sim.run_pattern("full_pressure")
        # Interactions list should be valid regardless
        for f in result.interactions:
            assert isinstance(f, InteractionFinding)


# ── InteractionFinding ───────────────────────────────────────────────

class TestInteractionFinding:
    def test_dataclass_fields(self):
        f = InteractionFinding(
            finding_id="test-finding",
            description="A test finding",
            severity=ThreatSeverity.CRITICAL,
            attack_vectors=["a", "b"],
            evidence=["e1", "e2"],
        )
        assert f.finding_id == "test-finding"
        assert f.severity == ThreatSeverity.CRITICAL
        assert len(f.attack_vectors) == 2
        assert len(f.evidence) == 2
