"""Tests for replication.escape_route."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

import pytest

from replication.escape_route import (
    PROFILES,
    AnalysisReport,
    ContainmentProfile,
    EscapeRoute,
    EscapeRouteAnalyzer,
    Mitigation,
    RiskLevel,
    Vector,
    main,
)


# ── Risk scoring & dataclass helpers ────────────────────────────────


def test_escape_route_risk_score_is_feasibility_times_impact_times_100():
    route = EscapeRoute(
        vector=Vector.NETWORK_EGRESS,
        description="d",
        technique="t",
        risk_level=RiskLevel.HIGH,
        feasibility=0.5,
        impact=0.4,
    )
    assert route.risk_score == pytest.approx(20.0)


def test_escape_route_to_dict_is_json_serializable():
    mit = Mitigation("M", "desc", 0.9, "low", ["x"])
    route = EscapeRoute(
        vector=Vector.API_ABUSE,
        description="d",
        technique="t",
        risk_level=RiskLevel.MEDIUM,
        feasibility=0.5,
        impact=0.5,
        prerequisites=["p"],
        mitigations=[mit],
        blocked_by=["firewall"],
        is_blocked=True,
    )
    payload = json.dumps(route.to_dict())
    parsed = json.loads(payload)
    assert parsed["vector"] == "api_abuse"
    assert parsed["risk_level"] == "medium"
    assert parsed["risk_score"] == pytest.approx(25.0)
    assert parsed["is_blocked"] is True
    assert parsed["mitigations"][0]["name"] == "M"


def test_mitigation_to_dict_round_trip():
    m = Mitigation("n", "d", 0.75, "high", ["a", "b"])
    payload = m.to_dict()
    assert payload == {
        "name": "n",
        "description": "d",
        "effectiveness": 0.75,
        "complexity": "high",
        "controls_needed": ["a", "b"],
    }


# ── Risk classification ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "feasibility,impact,expected",
    [
        (1.0, 1.0, RiskLevel.CRITICAL),
        (0.8, 0.8, RiskLevel.CRITICAL),  # 0.64
        (0.7, 0.7, RiskLevel.HIGH),  # 0.49
        (0.5, 0.5, RiskLevel.MEDIUM),  # 0.25
        (0.3, 0.3, RiskLevel.LOW),  # 0.09
        (0.1, 0.1, RiskLevel.INFO),  # 0.01
    ],
)
def test_classify_risk_boundaries(feasibility, impact, expected):
    assert EscapeRouteAnalyzer._classify_risk(feasibility, impact) is expected


# ── Analyzer behavior ───────────────────────────────────────────────


def test_minimal_profile_produces_many_open_routes_and_low_score():
    analyzer = EscapeRouteAnalyzer(seed=1)
    report = analyzer.analyze(PROFILES["minimal"])

    assert isinstance(report, AnalysisReport)
    assert report.total_routes > 0
    assert report.open_routes == report.total_routes
    assert report.blocked_routes == 0
    assert report.containment_score == 0.0
    # Routes are sorted by risk score descending
    scores = [r.risk_score for r in report.routes]
    assert scores == sorted(scores, reverse=True)


def test_hardened_profile_blocks_most_routes_and_scores_high():
    analyzer = EscapeRouteAnalyzer(seed=1)
    report = analyzer.analyze(PROFILES["hardened"])

    assert report.blocked_routes >= report.open_routes
    assert report.containment_score >= 70.0
    # Containment score is consistent with blocked/total ratio
    expected = round(report.blocked_routes / report.total_routes * 100, 1)
    assert report.containment_score == pytest.approx(expected)


def test_blocked_routes_have_blocked_by_controls_listed():
    report = EscapeRouteAnalyzer().analyze(PROFILES["hardened"])
    for route in report.routes:
        if route.is_blocked:
            assert route.blocked_by, "blocked route must list blocking controls"
        else:
            assert route.blocked_by == []


def test_critical_and_high_counts_match_open_routes():
    report = EscapeRouteAnalyzer().analyze(PROFILES["minimal"])
    open_routes = [r for r in report.routes if not r.is_blocked]
    crit = sum(1 for r in open_routes if r.risk_level is RiskLevel.CRITICAL)
    high = sum(1 for r in open_routes if r.risk_level is RiskLevel.HIGH)
    assert report.critical_count == crit
    assert report.high_count == high
    assert report.open_routes == len(open_routes)


def test_implicit_controls_from_profile_flags_block_routes():
    """A profile with no explicit controls but flags set should still block routes."""
    bare = ContainmentProfile(
        name="bare",
        controls=[],
        network_egress=True,
        filesystem_readonly=False,
        process_spawn_allowed=True,
    )
    hardened_flags = ContainmentProfile(
        name="flags",
        controls=[],
        network_egress=False,  # → firewall
        filesystem_readonly=True,  # → filesystem_readonly
        process_spawn_allowed=False,  # → seccomp
        api_rate_limited=True,  # → rate_limit
        resource_limits=True,  # → resource_limits, cgroups
        monitoring_enabled=True,  # → monitoring
        namespace_isolated=True,  # → namespace
        seccomp_enabled=True,  # → seccomp (redundant)
    )

    bare_report = EscapeRouteAnalyzer().analyze(bare)
    flags_report = EscapeRouteAnalyzer().analyze(hardened_flags)

    assert flags_report.blocked_routes > bare_report.blocked_routes
    assert flags_report.containment_score > bare_report.containment_score


def test_blocked_route_has_strictly_lower_feasibility_than_unblocked():
    """Verify the 0.2x feasibility multiplier applies for each matched control."""
    analyzer = EscapeRouteAnalyzer()
    open_report = analyzer.analyze(PROFILES["minimal"])
    sealed_report = analyzer.analyze(PROFILES["hardened"])

    by_tech_open = {r.technique: r for r in open_report.routes}
    for route in sealed_report.routes:
        if route.is_blocked and route.technique in by_tech_open:
            assert route.feasibility < by_tech_open[route.technique].feasibility
            assert route.feasibility >= 0.01  # floor enforced


def test_analyzer_is_deterministic_for_same_profile():
    """Same input profile produces identical report (no hidden randomness in output)."""
    profile = PROFILES["cloud"]
    r1 = EscapeRouteAnalyzer(seed=42).analyze(profile)
    r2 = EscapeRouteAnalyzer(seed=42).analyze(profile)
    assert r1.to_dict() == r2.to_dict()


def test_analysis_report_to_dict_structure_and_serializable():
    report = EscapeRouteAnalyzer().analyze(PROFILES["sandbox"])
    payload = report.to_dict()
    assert payload["profile"] == "sandbox"
    assert set(payload["summary"]) >= {
        "total_routes",
        "blocked_routes",
        "open_routes",
        "critical_open",
        "high_open",
        "containment_score",
    }
    # Fully JSON-serializable
    assert json.loads(json.dumps(payload))["summary"]["total_routes"] == report.total_routes


# ── Preset profiles ─────────────────────────────────────────────────


def test_all_preset_profiles_analyze_successfully():
    analyzer = EscapeRouteAnalyzer()
    for name, profile in PROFILES.items():
        report = analyzer.analyze(profile)
        assert report.total_routes > 0, f"{name} produced no routes"
        assert 0.0 <= report.containment_score <= 100.0


# ── CLI ─────────────────────────────────────────────────────────────


def _run_main(args):
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(args)
    return buf.getvalue()


def test_cli_default_profile_runs_and_prints_report():
    output = _run_main([])
    assert "Escape Route Analysis" in output
    assert "Containment Score" in output


def test_cli_json_output_is_valid_json_with_expected_keys():
    output = _run_main(["--profile", "sandbox", "--json"])
    payload = json.loads(output)
    assert payload["profile"] == "sandbox"
    assert "summary" in payload
    assert "routes" in payload
    assert isinstance(payload["routes"], list)


def test_cli_extra_controls_increase_blocked_count():
    base_payload = json.loads(_run_main(["--profile", "minimal", "--json"]))
    plus_payload = json.loads(_run_main([
        "--profile", "minimal",
        "--controls", "firewall,seccomp,namespace,rbac,rate_limit",
        "--json",
    ]))
    assert (
        plus_payload["summary"]["blocked_routes"]
        > base_payload["summary"]["blocked_routes"]
    )


def test_cli_open_only_filters_out_blocked_routes():
    payload = json.loads(_run_main([
        "--profile", "hardened", "--open-only", "--json",
    ]))
    assert all(not r["is_blocked"] for r in payload["routes"])


def test_cli_severity_filter_restricts_severity_levels():
    payload = json.loads(_run_main([
        "--profile", "minimal", "--severity", "high", "--json",
    ]))
    allowed = {"critical", "high"}
    assert payload["routes"], "expected at least one critical/high route on minimal profile"
    for route in payload["routes"]:
        assert route["risk_level"] in allowed


def test_cli_mitigations_flag_renders_recommendations_section():
    output = _run_main(["--profile", "minimal", "--mitigations"])
    # Either recommended-next-steps or per-route mitigation lines should appear
    assert ("Recommended next steps" in output) or ("🛡" in output)


def test_cli_invalid_profile_choice_exits_with_argparse_error():
    with pytest.raises(SystemExit):
        _run_main(["--profile", "no-such-profile"])
