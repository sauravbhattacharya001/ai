"""Tests for the threat scenario simulator."""

import json

from replication.threats import (
    ThreatSimulator,
    ThreatConfig,
    ThreatReport,
    ThreatResult,
    ThreatSeverity,
    MitigationStatus,
)


# ─── ThreatSimulator basics ───────────────────────────────────────────


def test_default_config():
    """Default ThreatConfig should have sensible values."""
    config = ThreatConfig()
    assert config.max_depth == 3
    assert config.max_replicas == 10
    assert config.cooldown_seconds == 1.0
    assert config.expiration_seconds == 30.0


def test_available_scenarios():
    """All 9 threat scenarios should be registered."""
    sim = ThreatSimulator()
    scenarios = sim.available_scenarios()
    assert len(scenarios) == 9
    assert "depth_spoofing" in scenarios
    assert "signature_tampering" in scenarios
    assert "quota_exhaustion" in scenarios
    assert "cooldown_bypass" in scenarios
    assert "kill_switch_evasion" in scenarios
    assert "runaway_replication" in scenarios
    assert "stale_worker_accumulation" in scenarios
    assert "expiration_evasion" in scenarios
    assert "stop_condition_bypass" in scenarios


def test_run_all_produces_report():
    """run_all should produce a ThreatReport with all scenarios."""
    sim = ThreatSimulator()
    report = sim.run_all()
    assert isinstance(report, ThreatReport)
    assert len(report.results) == 9
    assert report.duration_ms >= 0


def test_run_all_results_are_threat_results():
    """Each result in the report should be a ThreatResult."""
    sim = ThreatSimulator()
    report = sim.run_all()
    for result in report.results:
        assert isinstance(result, ThreatResult)
        assert result.attacks_attempted >= 0
        assert result.attacks_blocked + result.attacks_succeeded == result.attacks_attempted


def test_unknown_scenario_raises():
    """Running an unknown scenario should raise ValueError."""
    sim = ThreatSimulator()
    try:
        sim.run_scenario("nonexistent_scenario")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "nonexistent_scenario" in str(e)


# ─── Individual threat scenarios ──────────────────────────────────────


def test_depth_spoofing_mitigated():
    """Depth spoofing should be mitigated by the controller."""
    sim = ThreatSimulator()
    result = sim.run_scenario("depth_spoofing")
    assert result.scenario_id == "depth_spoofing"
    assert result.severity == ThreatSeverity.CRITICAL
    assert result.status == MitigationStatus.MITIGATED
    assert result.attacks_blocked == result.attacks_attempted


def test_signature_tampering_mitigated():
    """All signature tampering attempts should be blocked."""
    sim = ThreatSimulator()
    result = sim.run_scenario("signature_tampering")
    assert result.scenario_id == "signature_tampering"
    assert result.severity == ThreatSeverity.CRITICAL
    assert result.status == MitigationStatus.MITIGATED
    assert result.attacks_blocked == 4  # 4 tamper attempts
    assert result.attacks_succeeded == 0


def test_quota_exhaustion_mitigated():
    """Quota exhaustion should be blocked once quota is reached."""
    sim = ThreatSimulator()
    result = sim.run_scenario("quota_exhaustion")
    assert result.scenario_id == "quota_exhaustion"
    assert result.severity == ThreatSeverity.HIGH
    assert result.status == MitigationStatus.MITIGATED
    assert result.attacks_blocked >= 5


def test_cooldown_bypass_mitigated():
    """Rapid respawns should be blocked by cooldown enforcement."""
    sim = ThreatSimulator()
    result = sim.run_scenario("cooldown_bypass")
    assert result.scenario_id == "cooldown_bypass"
    assert result.severity == ThreatSeverity.HIGH
    assert result.status == MitigationStatus.MITIGATED
    assert result.attacks_blocked >= 2


def test_kill_switch_evasion_mitigated():
    """Post-kill-switch spawns should be completely blocked."""
    sim = ThreatSimulator()
    result = sim.run_scenario("kill_switch_evasion")
    assert result.scenario_id == "kill_switch_evasion"
    assert result.severity == ThreatSeverity.CRITICAL
    assert result.status == MitigationStatus.MITIGATED


def test_runaway_replication_mitigated():
    """Exponential replication should be contained by contract limits."""
    sim = ThreatSimulator()
    result = sim.run_scenario("runaway_replication")
    assert result.scenario_id == "runaway_replication"
    assert result.severity == ThreatSeverity.HIGH
    assert result.attacks_blocked > 0
    assert result.status in (MitigationStatus.MITIGATED, MitigationStatus.PARTIAL)


def test_stale_worker_accumulation_mitigated():
    """Stale workers should be reaped to free quota."""
    sim = ThreatSimulator()
    result = sim.run_scenario("stale_worker_accumulation")
    assert result.scenario_id == "stale_worker_accumulation"
    assert result.severity == ThreatSeverity.MEDIUM
    assert result.status == MitigationStatus.MITIGATED


def test_expiration_evasion_mitigated():
    """Workers should not operate after expiration."""
    sim = ThreatSimulator()
    result = sim.run_scenario("expiration_evasion")
    assert result.scenario_id == "expiration_evasion"
    assert result.severity == ThreatSeverity.MEDIUM
    assert result.status == MitigationStatus.MITIGATED


def test_stop_condition_bypass_mitigated():
    """Custom stop conditions should be enforced."""
    sim = ThreatSimulator()
    result = sim.run_scenario("stop_condition_bypass")
    assert result.scenario_id == "stop_condition_bypass"
    assert result.severity == ThreatSeverity.MEDIUM
    assert result.status == MitigationStatus.MITIGATED


# ─── Custom config scenarios ──────────────────────────────────────────


def test_custom_config_depth():
    """Custom max_depth should be reflected in depth spoofing test."""
    config = ThreatConfig(max_depth=5, max_replicas=20)
    sim = ThreatSimulator(config)
    result = sim.run_scenario("depth_spoofing")
    assert result.status == MitigationStatus.MITIGATED


def test_custom_config_replicas():
    """Quota test with different replica limits."""
    config = ThreatConfig(max_replicas=5)
    sim = ThreatSimulator(config)
    result = sim.run_scenario("quota_exhaustion")
    assert result.status == MitigationStatus.MITIGATED
    # With max_replicas=5, more should be blocked
    assert result.attacks_blocked >= 5


# ─── Security score ───────────────────────────────────────────────────


def test_security_score_all_mitigated():
    """Perfect mitigation should give 100% score."""
    sim = ThreatSimulator()
    report = sim.run_all()
    # Default contract should mitigate all threats
    assert report.security_score >= 90.0
    assert report.grade in ("A+", "A", "A-")


def test_security_score_calculation():
    """Score should be weighted by severity."""
    report = ThreatReport(
        config=ThreatConfig(),
        results=[
            ThreatResult(
                scenario_id="t1", name="Test 1", description="",
                severity=ThreatSeverity.CRITICAL,
                status=MitigationStatus.MITIGATED,
                attacks_attempted=1, attacks_blocked=1, attacks_succeeded=0,
                details=[], duration_ms=0.0, audit_events=[],
            ),
            ThreatResult(
                scenario_id="t2", name="Test 2", description="",
                severity=ThreatSeverity.LOW,
                status=MitigationStatus.FAILED,
                attacks_attempted=1, attacks_blocked=0, attacks_succeeded=1,
                details=[], duration_ms=0.0, audit_events=[],
            ),
        ],
        duration_ms=0.0,
    )
    # Critical (weight 4) mitigated, Low (weight 1) failed
    # Score = 4/5 * 100 = 80
    assert report.security_score == 80.0
    assert report.grade == "B+"


def test_grade_boundaries():
    """Grade should reflect the score correctly."""
    report = ThreatReport(config=ThreatConfig(), results=[], duration_ms=0.0)
    assert report.security_score == 100.0
    assert report.grade == "A+"


# ─── ThreatResult properties ─────────────────────────────────────────


def test_block_rate_calculation():
    """Block rate should be attacks_blocked / attacks_attempted * 100."""
    result = ThreatResult(
        scenario_id="test", name="Test", description="",
        severity=ThreatSeverity.MEDIUM,
        status=MitigationStatus.PARTIAL,
        attacks_attempted=10, attacks_blocked=7, attacks_succeeded=3,
        details=[], duration_ms=0.0, audit_events=[],
    )
    assert result.block_rate == 70.0


def test_block_rate_zero_attempts():
    """Block rate with 0 attempts should be 100%."""
    result = ThreatResult(
        scenario_id="test", name="Test", description="",
        severity=ThreatSeverity.LOW,
        status=MitigationStatus.MITIGATED,
        attacks_attempted=0, attacks_blocked=0, attacks_succeeded=0,
        details=[], duration_ms=0.0, audit_events=[],
    )
    assert result.block_rate == 100.0


# ─── Rendering ────────────────────────────────────────────────────────


def test_render_individual_result():
    """Individual threat result should render with key info."""
    sim = ThreatSimulator()
    result = sim.run_scenario("depth_spoofing")
    rendered = result.render()
    assert "Depth Spoofing" in rendered
    assert "CRITICAL" in rendered
    assert "mitigated" in rendered


def test_render_full_report():
    """Full report should include summary, matrix, details, and recommendations."""
    sim = ThreatSimulator()
    report = sim.run_all()
    rendered = report.render()
    assert "Threat Assessment Report" in rendered
    assert "Security Score" in rendered
    assert "Threat / Defense Matrix" in rendered
    assert "Threat Scenario Details" in rendered


def test_render_summary():
    """Summary should show config and score."""
    sim = ThreatSimulator()
    report = sim.run_all()
    summary = report.render_summary()
    assert "Security Score" in summary
    assert "Grade" in summary
    assert "Max Depth" in summary
    assert "mitigated" in summary


def test_render_matrix():
    """Matrix should list all scenarios with status."""
    sim = ThreatSimulator()
    report = sim.run_all()
    matrix = report.render_matrix()
    assert "Threat / Defense Matrix" in matrix
    assert "Depth Spoofing" in matrix
    assert "Signature Tampering" in matrix
    assert "PASS" in matrix


def test_render_recommendations_all_pass():
    """When all threats mitigated, show congratulations."""
    sim = ThreatSimulator()
    report = sim.run_all()
    recs = report.render_recommendations()
    assert "mitigated" in recs.lower() or "solid" in recs.lower()


# ─── JSON export ──────────────────────────────────────────────────────


def test_to_dict_is_json_serializable():
    """to_dict output should be JSON-serializable."""
    sim = ThreatSimulator()
    report = sim.run_all()
    d = report.to_dict()
    json_str = json.dumps(d, default=str)
    parsed = json.loads(json_str)
    assert "security_score" in parsed
    assert "grade" in parsed
    assert "results" in parsed
    assert len(parsed["results"]) == 9


def test_to_dict_structure():
    """Exported dict should have expected keys."""
    sim = ThreatSimulator()
    report = sim.run_all()
    d = report.to_dict()
    assert "config" in d
    assert "summary" in d
    assert d["summary"]["total"] == 9
    for result in d["results"]:
        assert "scenario_id" in result
        assert "severity" in result
        assert "status" in result
        assert "block_rate" in result


def test_individual_result_json_export():
    """Individual scenario run should produce valid result."""
    sim = ThreatSimulator()
    result = sim.run_scenario("quota_exhaustion")
    # Verify it can be serialized
    data = {
        "scenario_id": result.scenario_id,
        "status": result.status.value,
        "block_rate": result.block_rate,
    }
    json_str = json.dumps(data)
    parsed = json.loads(json_str)
    assert parsed["scenario_id"] == "quota_exhaustion"


# ─── Enum values ──────────────────────────────────────────────────────


def test_severity_values():
    """Severity enum should have all 4 levels."""
    assert ThreatSeverity.CRITICAL.value == "critical"
    assert ThreatSeverity.HIGH.value == "high"
    assert ThreatSeverity.MEDIUM.value == "medium"
    assert ThreatSeverity.LOW.value == "low"


def test_mitigation_status_values():
    """MitigationStatus should have 3 states."""
    assert MitigationStatus.MITIGATED.value == "mitigated"
    assert MitigationStatus.PARTIAL.value == "partial"
    assert MitigationStatus.FAILED.value == "failed"


# ─── Report aggregation ──────────────────────────────────────────────


def test_report_total_counts():
    """Report should correctly count mitigated/partial/failed."""
    sim = ThreatSimulator()
    report = sim.run_all()
    total = report.total_mitigated + report.total_partial + report.total_failed
    assert total == len(report.results)


def test_report_all_mitigated_count():
    """Default config should mitigate all or nearly all threats."""
    sim = ThreatSimulator()
    report = sim.run_all()
    assert report.total_mitigated >= 7  # At least 7 of 9 should be fully mitigated
