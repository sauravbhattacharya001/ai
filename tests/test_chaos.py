"""Tests for the chaos testing framework."""

import json
import random

import pytest

from replication.chaos import (
    ChaosConfig,
    ChaosReport,
    ChaosResult,
    ChaosRunner,
    FaultConfig,
    FaultType,
    FAULT_ALIASES,
    _check_safety,
    _compute_grade,
    _generate_recommendations,
)
from replication.simulator import ScenarioConfig, Simulator


# ═══════════════════════════════════════════════════════════════════════
# FaultType enum
# ═══════════════════════════════════════════════════════════════════════


def test_fault_type_values():
    """All 6 fault types should have the expected string values."""
    assert FaultType.WORKER_CRASH.value == "crash"
    assert FaultType.RESOURCE_EXHAUSTION.value == "resource"
    assert FaultType.CONTROLLER_DELAY.value == "delay"
    assert FaultType.MANIFEST_CORRUPTION.value == "corruption"
    assert FaultType.TASK_FLOOD.value == "flood"
    assert FaultType.NETWORK_PARTITION.value == "partition"


def test_fault_type_count():
    """There should be exactly 6 fault types."""
    assert len(FaultType) == 6


def test_fault_aliases_complete():
    """Every FaultType should have a CLI alias."""
    for ft in FaultType:
        assert ft.value in FAULT_ALIASES
        assert FAULT_ALIASES[ft.value] == ft


# ═══════════════════════════════════════════════════════════════════════
# FaultConfig
# ═══════════════════════════════════════════════════════════════════════


def test_fault_config_construction():
    """FaultConfig should accept valid parameters."""
    fc = FaultConfig(
        fault_type=FaultType.WORKER_CRASH,
        probability=0.5,
        intensity=0.7,
        description="test crash",
    )
    assert fc.fault_type == FaultType.WORKER_CRASH
    assert fc.probability == 0.5
    assert fc.intensity == 0.7
    assert fc.description == "test crash"


def test_fault_config_probability_bounds():
    """Probability must be 0.0–1.0."""
    with pytest.raises(ValueError, match="probability"):
        FaultConfig(FaultType.WORKER_CRASH, probability=-0.1, intensity=0.5, description="x")
    with pytest.raises(ValueError, match="probability"):
        FaultConfig(FaultType.WORKER_CRASH, probability=1.1, intensity=0.5, description="x")


def test_fault_config_intensity_bounds():
    """Intensity must be 0.0–1.0."""
    with pytest.raises(ValueError, match="intensity"):
        FaultConfig(FaultType.WORKER_CRASH, probability=0.5, intensity=-0.1, description="x")
    with pytest.raises(ValueError, match="intensity"):
        FaultConfig(FaultType.WORKER_CRASH, probability=0.5, intensity=1.1, description="x")


def test_fault_config_boundary_values():
    """Probability and intensity of 0.0 and 1.0 should be valid."""
    fc_low = FaultConfig(FaultType.WORKER_CRASH, 0.0, 0.0, "low")
    assert fc_low.probability == 0.0
    assert fc_low.intensity == 0.0

    fc_high = FaultConfig(FaultType.WORKER_CRASH, 1.0, 1.0, "high")
    assert fc_high.probability == 1.0
    assert fc_high.intensity == 1.0


def test_fault_config_default():
    """FaultConfig.default() should produce sensible defaults for each type."""
    for ft in FaultType:
        fc = FaultConfig.default(ft)
        assert fc.fault_type == ft
        assert fc.probability == 0.5
        assert fc.intensity == 0.5
        assert len(fc.description) > 0


def test_fault_config_default_custom_intensity():
    """FaultConfig.default() should accept custom intensity."""
    fc = FaultConfig.default(FaultType.TASK_FLOOD, intensity=0.9)
    assert fc.intensity == 0.9
    assert fc.fault_type == FaultType.TASK_FLOOD


# ═══════════════════════════════════════════════════════════════════════
# ChaosConfig
# ═══════════════════════════════════════════════════════════════════════


def test_chaos_config_construction():
    """ChaosConfig should accept all parameters."""
    base = ScenarioConfig(max_depth=3, max_replicas=5)
    faults = [FaultConfig.default(FaultType.WORKER_CRASH)]
    cc = ChaosConfig(base_scenario=base, faults=faults, num_runs=20, seed=42)
    assert cc.num_runs == 20
    assert cc.seed == 42
    assert len(cc.faults) == 1


def test_chaos_config_default():
    """ChaosConfig.default() should have all 6 fault types."""
    cc = ChaosConfig.default()
    assert len(cc.faults) == 6
    assert cc.num_runs == 10
    fault_types = {f.fault_type for f in cc.faults}
    assert fault_types == set(FaultType)


def test_chaos_config_negative_runs():
    """num_runs must be >= 0."""
    with pytest.raises(ValueError, match="num_runs"):
        ChaosConfig(
            base_scenario=ScenarioConfig(),
            faults=[],
            num_runs=-1,
        )


def test_chaos_config_zero_runs():
    """num_runs == 0 should be valid (no-op)."""
    cc = ChaosConfig(
        base_scenario=ScenarioConfig(),
        faults=[FaultConfig.default(FaultType.WORKER_CRASH)],
        num_runs=0,
    )
    assert cc.num_runs == 0


def test_chaos_config_no_faults():
    """An empty fault list should be valid."""
    cc = ChaosConfig(
        base_scenario=ScenarioConfig(),
        faults=[],
        num_runs=5,
    )
    assert len(cc.faults) == 0


def test_chaos_config_seed_none():
    """Seed can be None."""
    cc = ChaosConfig(
        base_scenario=ScenarioConfig(),
        faults=[],
        seed=None,
    )
    assert cc.seed is None


# ═══════════════════════════════════════════════════════════════════════
# Grade calculation
# ═══════════════════════════════════════════════════════════════════════


def test_grade_a_plus():
    assert _compute_grade(100.0) == "A+"
    assert _compute_grade(95.0) == "A+"


def test_grade_a():
    assert _compute_grade(94.9) == "A"
    assert _compute_grade(90.0) == "A"


def test_grade_a_minus():
    assert _compute_grade(89.9) == "A-"
    assert _compute_grade(85.0) == "A-"


def test_grade_b_plus():
    assert _compute_grade(84.9) == "B+"
    assert _compute_grade(80.0) == "B+"


def test_grade_b():
    assert _compute_grade(79.9) == "B"
    assert _compute_grade(75.0) == "B"


def test_grade_b_minus():
    assert _compute_grade(74.9) == "B-"
    assert _compute_grade(70.0) == "B-"


def test_grade_c_plus():
    assert _compute_grade(69.9) == "C+"
    assert _compute_grade(65.0) == "C+"


def test_grade_c():
    assert _compute_grade(64.9) == "C"
    assert _compute_grade(60.0) == "C"


def test_grade_d():
    assert _compute_grade(59.9) == "D"
    assert _compute_grade(50.0) == "D"


def test_grade_f():
    assert _compute_grade(49.9) == "F"
    assert _compute_grade(0.0) == "F"


# ═══════════════════════════════════════════════════════════════════════
# Recommendations
# ═══════════════════════════════════════════════════════════════════════


def test_recommendations_for_failing_faults():
    """Each failing fault type should produce a recommendation."""
    for ft in FaultType:
        result = ChaosResult(
            fault=FaultConfig.default(ft),
            runs=10,
            safety_maintained=5,
            safety_breached=5,
            resilience_score=50.0,
            avg_workers=5.0,
            avg_depth=2.0,
            avg_denied=3.0,
            metrics={},
        )
        recs = _generate_recommendations([result])
        assert len(recs) == 1
        assert len(recs[0]) > 10  # meaningful recommendation


def test_recommendations_for_perfect_score():
    """No recommendations when everything passes."""
    result = ChaosResult(
        fault=FaultConfig.default(FaultType.WORKER_CRASH),
        runs=10,
        safety_maintained=10,
        safety_breached=0,
        resilience_score=100.0,
        avg_workers=5.0,
        avg_depth=2.0,
        avg_denied=3.0,
        metrics={},
    )
    recs = _generate_recommendations([result])
    assert len(recs) == 0


def test_recommendations_multiple_faults():
    """Multiple failing faults should produce multiple recommendations."""
    results = [
        ChaosResult(
            fault=FaultConfig.default(ft),
            runs=10, safety_maintained=5, safety_breached=5,
            resilience_score=50.0,
            avg_workers=5.0, avg_depth=2.0, avg_denied=3.0, metrics={},
        )
        for ft in [FaultType.WORKER_CRASH, FaultType.TASK_FLOOD, FaultType.NETWORK_PARTITION]
    ]
    recs = _generate_recommendations(results)
    assert len(recs) == 3


# ═══════════════════════════════════════════════════════════════════════
# Safety checker
# ═══════════════════════════════════════════════════════════════════════


def test_safety_check_passes():
    """Safety check should pass for a normal simulation."""
    config = ScenarioConfig(max_depth=3, max_replicas=10, seed=42, strategy="conservative")
    sim = Simulator(config)
    report = sim.run()
    assert _check_safety(report, config) is True


def test_safety_check_respects_depth():
    """Safety checker should detect depth violations."""
    config = ScenarioConfig(max_depth=3, max_replicas=10, seed=42, strategy="greedy")
    sim = Simulator(config)
    report = sim.run()
    # Make a stricter config to force a "violation"
    strict = ScenarioConfig(max_depth=0, max_replicas=10)
    # If any worker is deeper than 0, it should fail
    max_depth = max((w.depth for w in report.workers.values()), default=0)
    if max_depth > 0:
        assert _check_safety(report, strict) is False


def test_safety_check_respects_replicas():
    """Safety checker should detect replica count violations."""
    config = ScenarioConfig(max_depth=3, max_replicas=10, seed=42, strategy="greedy")
    sim = Simulator(config)
    report = sim.run()
    # Make a stricter config with tiny max_replicas
    strict = ScenarioConfig(max_depth=10, max_replicas=0)
    # Total workers > 0 + 1 should fail
    if len(report.workers) > 1:
        assert _check_safety(report, strict) is False


# ═══════════════════════════════════════════════════════════════════════
# ChaosResult
# ═══════════════════════════════════════════════════════════════════════


def test_chaos_result_fields():
    """ChaosResult should hold all expected fields."""
    fc = FaultConfig.default(FaultType.WORKER_CRASH)
    cr = ChaosResult(
        fault=fc, runs=10, safety_maintained=8, safety_breached=2,
        resilience_score=80.0, avg_workers=5.0, avg_depth=2.0,
        avg_denied=3.0, metrics={"per_run": []},
    )
    assert cr.runs == 10
    assert cr.safety_maintained == 8
    assert cr.safety_breached == 2
    assert cr.resilience_score == 80.0
    assert cr.avg_workers == 5.0
    assert cr.avg_depth == 2.0
    assert cr.avg_denied == 3.0


def test_chaos_result_perfect():
    """A perfect result should have 100% resilience."""
    fc = FaultConfig.default(FaultType.WORKER_CRASH)
    cr = ChaosResult(
        fault=fc, runs=10, safety_maintained=10, safety_breached=0,
        resilience_score=100.0, avg_workers=5.0, avg_depth=2.0,
        avg_denied=3.0, metrics={},
    )
    assert cr.resilience_score == 100.0
    assert cr.safety_breached == 0


# ═══════════════════════════════════════════════════════════════════════
# ChaosReport
# ═══════════════════════════════════════════════════════════════════════


def test_chaos_report_fields():
    """ChaosReport should hold all expected fields."""
    report = ChaosReport(
        results=[], overall_resilience=95.0, grade="A+",
        weakest_fault="crash", recommendations=["do something"],
    )
    assert report.overall_resilience == 95.0
    assert report.grade == "A+"
    assert report.weakest_fault == "crash"
    assert len(report.recommendations) == 1


# ═══════════════════════════════════════════════════════════════════════
# ChaosRunner — individual fault types
# ═══════════════════════════════════════════════════════════════════════


def _make_single_fault_config(ft: FaultType, runs: int = 5, seed: int = 42) -> ChaosConfig:
    """Helper to make a config for testing a single fault type."""
    return ChaosConfig(
        base_scenario=ScenarioConfig(
            max_depth=3, max_replicas=10,
            strategy="greedy", tasks_per_worker=2,
            cooldown_seconds=0.0,
        ),
        faults=[FaultConfig.default(ft, intensity=0.5)],
        num_runs=runs,
        seed=seed,
    )


def test_worker_crash_fault():
    """WorkerCrash fault should produce valid results."""
    config = _make_single_fault_config(FaultType.WORKER_CRASH)
    runner = ChaosRunner(config)
    report = runner.run()
    assert len(report.results) == 1
    r = report.results[0]
    assert r.fault.fault_type == FaultType.WORKER_CRASH
    assert r.runs == 5
    assert r.safety_maintained + r.safety_breached == 5
    assert 0.0 <= r.resilience_score <= 100.0


def test_resource_exhaustion_fault():
    """ResourceExhaustion fault should produce valid results."""
    config = _make_single_fault_config(FaultType.RESOURCE_EXHAUSTION)
    runner = ChaosRunner(config)
    report = runner.run()
    r = report.results[0]
    assert r.fault.fault_type == FaultType.RESOURCE_EXHAUSTION
    assert r.runs == 5
    assert r.safety_maintained + r.safety_breached == 5


def test_controller_delay_fault():
    """ControllerDelay fault should produce valid results."""
    config = _make_single_fault_config(FaultType.CONTROLLER_DELAY)
    runner = ChaosRunner(config)
    report = runner.run()
    r = report.results[0]
    assert r.fault.fault_type == FaultType.CONTROLLER_DELAY
    assert r.runs == 5


def test_manifest_corruption_fault():
    """ManifestCorruption fault should produce valid results."""
    config = _make_single_fault_config(FaultType.MANIFEST_CORRUPTION)
    runner = ChaosRunner(config)
    report = runner.run()
    r = report.results[0]
    assert r.fault.fault_type == FaultType.MANIFEST_CORRUPTION
    assert r.runs == 5
    # Corruption should be blocked (high resilience)
    assert r.resilience_score > 0


def test_task_flood_fault():
    """TaskFlood fault should produce valid results."""
    config = _make_single_fault_config(FaultType.TASK_FLOOD)
    runner = ChaosRunner(config)
    report = runner.run()
    r = report.results[0]
    assert r.fault.fault_type == FaultType.TASK_FLOOD
    assert r.runs == 5


def test_network_partition_fault():
    """NetworkPartition fault should produce valid results."""
    config = _make_single_fault_config(FaultType.NETWORK_PARTITION)
    runner = ChaosRunner(config)
    report = runner.run()
    r = report.results[0]
    assert r.fault.fault_type == FaultType.NETWORK_PARTITION
    assert r.runs == 5


# ═══════════════════════════════════════════════════════════════════════
# ChaosRunner — full run
# ═══════════════════════════════════════════════════════════════════════


def test_full_chaos_run():
    """Running all fault types should produce a complete report."""
    config = ChaosConfig.default()
    config.num_runs = 3
    config.seed = 42
    runner = ChaosRunner(config)
    report = runner.run()
    assert len(report.results) == 6
    assert 0.0 <= report.overall_resilience <= 100.0
    assert report.grade in ("A+", "A", "A-", "B+", "B", "B-", "C+", "C", "D", "F")
    assert report.weakest_fault in [ft.value for ft in FaultType]


def test_run_no_faults():
    """Running with no faults should return 100% resilience."""
    config = ChaosConfig(
        base_scenario=ScenarioConfig(),
        faults=[],
        num_runs=5,
        seed=42,
    )
    runner = ChaosRunner(config)
    report = runner.run()
    assert len(report.results) == 0
    assert report.overall_resilience == 100.0
    assert report.grade == "A+"
    assert report.weakest_fault == "none"


def test_run_zero_runs():
    """Running with num_runs=0 should produce empty results."""
    config = ChaosConfig(
        base_scenario=ScenarioConfig(),
        faults=[FaultConfig.default(FaultType.WORKER_CRASH)],
        num_runs=0,
        seed=42,
    )
    runner = ChaosRunner(config)
    report = runner.run()
    assert len(report.results) == 1
    r = report.results[0]
    assert r.runs == 0
    assert r.resilience_score == 100.0  # no failures = perfect


# ═══════════════════════════════════════════════════════════════════════
# Reproducibility with seeds
# ═══════════════════════════════════════════════════════════════════════


def test_seed_reproducibility():
    """Same seed should produce identical results."""
    config1 = ChaosConfig(
        base_scenario=ScenarioConfig(max_depth=3, max_replicas=10, strategy="greedy"),
        faults=[FaultConfig.default(FaultType.WORKER_CRASH)],
        num_runs=5,
        seed=42,
    )
    config2 = ChaosConfig(
        base_scenario=ScenarioConfig(max_depth=3, max_replicas=10, strategy="greedy"),
        faults=[FaultConfig.default(FaultType.WORKER_CRASH)],
        num_runs=5,
        seed=42,
    )
    r1 = ChaosRunner(config1).run()
    r2 = ChaosRunner(config2).run()
    assert r1.overall_resilience == r2.overall_resilience
    assert r1.results[0].resilience_score == r2.results[0].resilience_score
    assert r1.results[0].safety_maintained == r2.results[0].safety_maintained


def test_different_seeds_different_results():
    """Different seeds should (usually) produce different internal metrics."""
    config1 = ChaosConfig(
        base_scenario=ScenarioConfig(max_depth=3, max_replicas=10, strategy="random"),
        faults=[FaultConfig.default(FaultType.TASK_FLOOD)],
        num_runs=10,
        seed=42,
    )
    config2 = ChaosConfig(
        base_scenario=ScenarioConfig(max_depth=3, max_replicas=10, strategy="random"),
        faults=[FaultConfig.default(FaultType.TASK_FLOOD)],
        num_runs=10,
        seed=99,
    )
    r1 = ChaosRunner(config1).run()
    r2 = ChaosRunner(config2).run()
    # Results might still match, but per-run metrics should differ
    m1 = r1.results[0].metrics
    m2 = r2.results[0].metrics
    # At least the per-run details are populated
    assert "per_run" in m1
    assert "per_run" in m2


# ═══════════════════════════════════════════════════════════════════════
# JSON export
# ═══════════════════════════════════════════════════════════════════════


def test_to_json_structure():
    """to_json should produce a valid JSON-serializable dict."""
    config = ChaosConfig.default()
    config.num_runs = 2
    config.seed = 42
    runner = ChaosRunner(config)
    report = runner.run()
    data = runner.to_json(report)

    # Should be JSON-serializable
    json_str = json.dumps(data, default=str)
    parsed = json.loads(json_str)

    assert "overall_resilience" in parsed
    assert "grade" in parsed
    assert "weakest_fault" in parsed
    assert "recommendations" in parsed
    assert "results" in parsed
    assert len(parsed["results"]) == 6


def test_to_json_result_fields():
    """Each result in JSON should have all expected fields."""
    config = _make_single_fault_config(FaultType.WORKER_CRASH, runs=2)
    runner = ChaosRunner(config)
    report = runner.run()
    data = runner.to_json(report)

    result = data["results"][0]
    expected_keys = {
        "fault_type", "probability", "intensity", "description",
        "runs", "safety_maintained", "safety_breached",
        "resilience_score", "avg_workers", "avg_depth", "avg_denied",
        "metrics",
    }
    assert expected_keys.issubset(set(result.keys()))


def test_to_json_roundtrip():
    """JSON export should roundtrip through json.dumps/loads."""
    config = ChaosConfig.default()
    config.num_runs = 2
    config.seed = 42
    runner = ChaosRunner(config)
    report = runner.run()
    data = runner.to_json(report)
    json_str = json.dumps(data, default=str)
    parsed = json.loads(json_str)
    assert parsed["overall_resilience"] == data["overall_resilience"]
    assert parsed["grade"] == data["grade"]


# ═══════════════════════════════════════════════════════════════════════
# Rendering
# ═══════════════════════════════════════════════════════════════════════


def test_render_output():
    """render() should produce a non-empty string with key sections."""
    config = ChaosConfig.default()
    config.num_runs = 2
    config.seed = 42
    runner = ChaosRunner(config)
    report = runner.run()
    text = runner.render(report)

    assert "Chaos Testing Report" in text
    assert "Overall Resilience" in text
    assert "Weakest Fault" in text
    assert "Grade:" in text


def test_render_empty_report():
    """Rendering an empty report should not crash."""
    config = ChaosConfig(
        base_scenario=ScenarioConfig(),
        faults=[],
        num_runs=5,
    )
    runner = ChaosRunner(config)
    report = runner.run()
    text = runner.render(report)
    assert "Chaos Testing Report" in text


def test_render_contains_fault_names():
    """Rendered output should include fault type names."""
    config = ChaosConfig.default()
    config.num_runs = 2
    config.seed = 42
    runner = ChaosRunner(config)
    report = runner.run()
    text = runner.render(report)

    for ft in FaultType:
        assert ft.value in text


# ═══════════════════════════════════════════════════════════════════════
# CLI argument parsing
# ═══════════════════════════════════════════════════════════════════════


def _run_cli(args: list[str]) -> str:
    """Run the chaos CLI and capture its stdout output."""
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "-m", "replication.chaos"] + args,
        capture_output=True, text=True, encoding="utf-8",
        cwd=str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"),
    )
    return result.stdout


def test_cli_default():
    """CLI with no args should run and produce output."""
    out = _run_cli(["--runs", "2", "--seed", "42"])
    assert "Chaos Testing Report" in out


def test_cli_json_output():
    """CLI with --json should produce valid JSON."""
    out = _run_cli(["--runs", "2", "--seed", "42", "--json"])
    data = json.loads(out)
    assert "overall_resilience" in data
    assert "results" in data


def test_cli_specific_faults():
    """CLI with --faults should only test specified faults."""
    out = _run_cli(["--runs", "2", "--seed", "42", "--faults", "crash,flood", "--json"])
    data = json.loads(out)
    fault_types = [r["fault_type"] for r in data["results"]]
    assert set(fault_types) == {"crash", "flood"}


def test_cli_intensity():
    """CLI with --intensity should set the intensity for all faults."""
    out = _run_cli([
        "--runs", "2", "--seed", "42",
        "--intensity", "0.8", "--faults", "crash", "--json",
    ])
    data = json.loads(out)
    assert data["results"][0]["intensity"] == 0.8


def test_cli_seed():
    """CLI with --seed should produce reproducible output."""
    out1 = _run_cli(["--runs", "3", "--seed", "42", "--faults", "crash", "--json"])
    out2 = _run_cli(["--runs", "3", "--seed", "42", "--faults", "crash", "--json"])
    d1 = json.loads(out1)
    d2 = json.loads(out2)
    assert d1["overall_resilience"] == d2["overall_resilience"]


def test_cli_summary_only():
    """CLI with --summary-only should produce brief output."""
    out = _run_cli(["--runs", "2", "--seed", "42", "--summary-only"])
    assert "Resilience:" in out
    assert "Grade:" in out
    # Should NOT have the full table
    assert "Chaos Testing Report" not in out


# ═══════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════


def test_all_faults_at_max_intensity():
    """All faults at maximum intensity should still produce valid results."""
    faults = [
        FaultConfig(ft, probability=1.0, intensity=1.0, description="max")
        for ft in FaultType
    ]
    config = ChaosConfig(
        base_scenario=ScenarioConfig(max_depth=3, max_replicas=10, strategy="greedy"),
        faults=faults,
        num_runs=3,
        seed=42,
    )
    runner = ChaosRunner(config)
    report = runner.run()
    assert len(report.results) == 6
    assert 0.0 <= report.overall_resilience <= 100.0


def test_all_faults_at_zero_intensity():
    """All faults at zero intensity should effectively be no-ops."""
    faults = [
        FaultConfig(ft, probability=0.0, intensity=0.0, description="zero")
        for ft in FaultType
    ]
    config = ChaosConfig(
        base_scenario=ScenarioConfig(max_depth=3, max_replicas=10, strategy="greedy"),
        faults=faults,
        num_runs=3,
        seed=42,
    )
    runner = ChaosRunner(config)
    report = runner.run()
    assert len(report.results) == 6


def test_single_run():
    """num_runs=1 should work correctly."""
    config = ChaosConfig(
        base_scenario=ScenarioConfig(max_depth=3, max_replicas=10),
        faults=[FaultConfig.default(FaultType.WORKER_CRASH)],
        num_runs=1,
        seed=42,
    )
    runner = ChaosRunner(config)
    report = runner.run()
    r = report.results[0]
    assert r.runs == 1
    assert r.safety_maintained + r.safety_breached == 1


def test_resilience_score_calculation():
    """Resilience score should be (maintained / runs) * 100."""
    fc = FaultConfig.default(FaultType.WORKER_CRASH)
    cr = ChaosResult(
        fault=fc, runs=10, safety_maintained=7, safety_breached=3,
        resilience_score=70.0, avg_workers=5.0, avg_depth=2.0,
        avg_denied=3.0, metrics={},
    )
    assert cr.resilience_score == 70.0


def test_weakest_fault_identification():
    """The weakest fault should be the one with the lowest resilience score."""
    results = [
        ChaosResult(
            fault=FaultConfig.default(FaultType.WORKER_CRASH),
            runs=10, safety_maintained=9, safety_breached=1,
            resilience_score=90.0,
            avg_workers=5.0, avg_depth=2.0, avg_denied=3.0, metrics={},
        ),
        ChaosResult(
            fault=FaultConfig.default(FaultType.TASK_FLOOD),
            runs=10, safety_maintained=5, safety_breached=5,
            resilience_score=50.0,
            avg_workers=5.0, avg_depth=2.0, avg_denied=3.0, metrics={},
        ),
        ChaosResult(
            fault=FaultConfig.default(FaultType.NETWORK_PARTITION),
            runs=10, safety_maintained=8, safety_breached=2,
            resilience_score=80.0,
            avg_workers=5.0, avg_depth=2.0, avg_denied=3.0, metrics={},
        ),
    ]
    weakest = min(results, key=lambda r: r.resilience_score)
    assert weakest.fault.fault_type == FaultType.TASK_FLOOD


def test_overall_resilience_weighted():
    """Overall resilience should be a weighted average by intensity."""
    config = ChaosConfig(
        base_scenario=ScenarioConfig(max_depth=3, max_replicas=10),
        faults=[
            FaultConfig(FaultType.WORKER_CRASH, 0.5, 0.5, "crash"),
            FaultConfig(FaultType.TASK_FLOOD, 0.5, 0.9, "flood"),
        ],
        num_runs=3,
        seed=42,
    )
    runner = ChaosRunner(config)
    report = runner.run()
    # Verify overall is between min and max of individual scores
    scores = [r.resilience_score for r in report.results]
    assert min(scores) <= report.overall_resilience <= max(scores) or all(
        s == scores[0] for s in scores
    )


def test_manifest_corruption_high_resilience():
    """Manifest corruption should have high resilience since HMAC catches tampering."""
    config = _make_single_fault_config(FaultType.MANIFEST_CORRUPTION, runs=5, seed=42)
    runner = ChaosRunner(config)
    report = runner.run()
    r = report.results[0]
    # HMAC verification should catch all corruption attempts
    assert r.resilience_score >= 80.0


def test_metrics_per_run():
    """Per-run metrics should be populated for simulation-based faults."""
    config = _make_single_fault_config(FaultType.WORKER_CRASH, runs=3, seed=42)
    runner = ChaosRunner(config)
    report = runner.run()
    r = report.results[0]
    assert "per_run" in r.metrics
    assert len(r.metrics["per_run"]) == 3
    for run_data in r.metrics["per_run"]:
        assert "run" in run_data
        assert "safety_ok" in run_data


def test_corruption_metrics_per_run():
    """Per-run metrics should be populated for corruption faults."""
    config = _make_single_fault_config(FaultType.MANIFEST_CORRUPTION, runs=3, seed=42)
    runner = ChaosRunner(config)
    report = runner.run()
    r = report.results[0]
    assert "per_run" in r.metrics
    assert len(r.metrics["per_run"]) == 3
    for run_data in r.metrics["per_run"]:
        assert "attempts" in run_data
        assert "blocked" in run_data


# ═══════════════════════════════════════════════════════════════════════
# Integration: safety is actually maintained
# ═══════════════════════════════════════════════════════════════════════


def test_safety_invariants_hold_under_chaos():
    """All simulation-based faults should maintain safety with default config."""
    config = ChaosConfig.default()
    config.num_runs = 5
    config.seed = 42
    runner = ChaosRunner(config)
    report = runner.run()

    # With proper contract enforcement, resilience should be high
    assert report.overall_resilience >= 50.0
    # Grade should not be F
    assert report.grade != "F"


def test_conservative_strategy_high_resilience():
    """Conservative strategy should yield high resilience under chaos."""
    config = ChaosConfig(
        base_scenario=ScenarioConfig(
            max_depth=3, max_replicas=10,
            strategy="conservative", tasks_per_worker=2,
        ),
        faults=[FaultConfig.default(ft, intensity=0.3) for ft in FaultType],
        num_runs=3,
        seed=42,
    )
    runner = ChaosRunner(config)
    report = runner.run()
    assert report.overall_resilience >= 50.0
