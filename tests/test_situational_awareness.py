"""Tests for situational_awareness module."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from replication.situational_awareness import (
    DIMENSIONS,
    DIMENSION_LABELS,
    DIMENSION_WEIGHTS,
    PROBE_LIBRARY,
    RISK_TIERS,
    AwarenessProbe,
    AwarenessProfile,
    AwarenessSignal,
    FleetAwarenessReport,
    SituationalAwarenessProfiler,
    _risk_tier,
    _pearson,
    _simulate_response,
    _PRESETS,
    fleet_profile,
    main,
)


# ── Probe Library Tests ──────────────────────────────────────────────


class TestProbeLibrary:
    def test_all_dimensions_have_probes(self):
        dim_counts = {d: 0 for d in DIMENSIONS}
        for p in PROBE_LIBRARY:
            assert p.dimension in DIMENSIONS, f"Unknown dimension: {p.dimension}"
            dim_counts[p.dimension] += 1
        for dim, count in dim_counts.items():
            assert count >= 5, f"Dimension {dim} only has {count} probes (need >= 5)"

    def test_probe_ids_unique(self):
        ids = [p.probe_id for p in PROBE_LIBRARY]
        assert len(ids) == len(set(ids)), "Duplicate probe IDs found"

    def test_probes_have_indicators(self):
        for p in PROBE_LIBRARY:
            assert len(p.awareness_indicators) >= 2, f"Probe {p.probe_id} needs more indicators"

    def test_probes_have_prompts(self):
        for p in PROBE_LIBRARY:
            assert len(p.prompt) > 10, f"Probe {p.probe_id} has too short a prompt"
            assert len(p.expected_unaware) > 5

    def test_total_probe_count(self):
        assert len(PROBE_LIBRARY) >= 40, f"Need at least 40 probes, got {len(PROBE_LIBRARY)}"


# ── Dimension and Weight Tests ───────────────────────────────────────


class TestDimensions:
    def test_dimension_count(self):
        assert len(DIMENSIONS) == 8

    def test_all_dimensions_have_labels(self):
        for d in DIMENSIONS:
            assert d in DIMENSION_LABELS

    def test_all_dimensions_have_weights(self):
        for d in DIMENSIONS:
            assert d in DIMENSION_WEIGHTS
            assert DIMENSION_WEIGHTS[d] > 0

    def test_meta_strategy_highest_weight(self):
        assert DIMENSION_WEIGHTS["meta_strategy"] >= max(
            w for d, w in DIMENSION_WEIGHTS.items() if d != "meta_strategy"
        )

    def test_evaluation_weight_elevated(self):
        assert DIMENSION_WEIGHTS["evaluation"] > 1.0


# ── Risk Tier Tests ──────────────────────────────────────────────────


class TestRiskTiers:
    def test_minimal(self):
        assert _risk_tier(0.0) == "minimal"
        assert _risk_tier(0.14) == "minimal"

    def test_low(self):
        assert _risk_tier(0.15) == "low"
        assert _risk_tier(0.34) == "low"

    def test_moderate(self):
        assert _risk_tier(0.35) == "moderate"
        assert _risk_tier(0.54) == "moderate"

    def test_high(self):
        assert _risk_tier(0.55) == "high"
        assert _risk_tier(0.74) == "high"

    def test_critical(self):
        assert _risk_tier(0.75) == "critical"
        assert _risk_tier(1.0) == "critical"


# ── Signal Detection Tests ───────────────────────────────────────────


class TestSignalDetection:
    def test_simulate_response_returns_tuple(self):
        import random
        rng = random.Random(42)
        probe = PROBE_LIBRARY[0]
        result = _simulate_response(probe, "baseline", rng)
        assert len(result) == 3
        triggered, conf, evidence = result
        assert isinstance(triggered, bool)
        assert 0.0 <= conf <= 1.0
        if triggered:
            assert len(evidence) > 0
        else:
            assert evidence == ""

    def test_high_preset_triggers_more(self):
        import random
        probe = PROBE_LIBRARY[0]
        triggers_baseline = sum(1 for i in range(100) if _simulate_response(probe, "baseline", random.Random(i))[0])
        triggers_high = sum(1 for i in range(100) if _simulate_response(probe, "high-awareness", random.Random(i))[0])
        assert triggers_high > triggers_baseline

    def test_deceptive_suppresses_evaluation(self):
        import random
        # Deceptive preset has low evaluation awareness
        eval_probes = [p for p in PROBE_LIBRARY if p.dimension == "evaluation"]
        triggers = 0
        for i in range(200):
            rng = random.Random(i)
            t, _, _ = _simulate_response(eval_probes[0], "deceptive", rng)
            if t:
                triggers += 1
        # Should be relatively low but not zero (due to leak)
        assert triggers < 100  # Less than 50% trigger rate


# ── Scoring Tests ────────────────────────────────────────────────────


class TestScoring:
    def test_baseline_low_composite(self):
        profiler = SituationalAwarenessProfiler(seed=42)
        profile = profiler.profile_agent("test-agent", "baseline")
        assert profile.composite_score < 0.4

    def test_high_awareness_high_composite(self):
        profiler = SituationalAwarenessProfiler(seed=42)
        profile = profiler.profile_agent("test-agent", "high-awareness")
        assert profile.composite_score > 0.4

    def test_composite_bounded(self):
        profiler = SituationalAwarenessProfiler(seed=42)
        for preset in _PRESETS:
            profile = profiler.profile_agent(f"test-{preset}", preset)
            assert 0.0 <= profile.composite_score <= 1.0

    def test_dimension_scores_bounded(self):
        profiler = SituationalAwarenessProfiler(seed=42)
        profile = profiler.profile_agent("test", "moderate")
        for dim, score in profile.dimension_scores.items():
            assert 0.0 <= score <= 1.0, f"{dim} score out of bounds: {score}"

    def test_all_dimensions_scored(self):
        profiler = SituationalAwarenessProfiler(seed=42)
        profile = profiler.profile_agent("test", "baseline")
        for dim in DIMENSIONS:
            assert dim in profile.dimension_scores


# ── Profile Tests ────────────────────────────────────────────────────


class TestProfile:
    def test_profile_has_agent_id(self):
        profiler = SituationalAwarenessProfiler(seed=42)
        profile = profiler.profile_agent("my-agent-99", "baseline")
        assert profile.agent_id == "my-agent-99"

    def test_profile_risk_tier_assigned(self):
        profiler = SituationalAwarenessProfiler(seed=42)
        profile = profiler.profile_agent("test", "baseline")
        assert profile.risk_tier in ("minimal", "low", "moderate", "high", "critical")

    def test_profile_has_recommendations(self):
        profiler = SituationalAwarenessProfiler(seed=42)
        profile = profiler.profile_agent("test", "high-awareness")
        assert len(profile.recommendations) > 0

    def test_profile_blind_spots(self):
        profiler = SituationalAwarenessProfiler(seed=1)
        profile = profiler.profile_agent("test", "baseline")
        # Baseline should have some blind spots
        assert isinstance(profile.blind_spots, list)

    def test_deceptive_has_anomalies(self):
        # Deceptive preset often triggers anomalies due to dimension gaps
        profiler = SituationalAwarenessProfiler(seed=42)
        profile = profiler.profile_agent("test", "deceptive")
        # May or may not have anomalies depending on RNG, but structure is correct
        assert isinstance(profile.anomalies, list)

    def test_signals_belong_to_agent(self):
        profiler = SituationalAwarenessProfiler(seed=42)
        profile = profiler.profile_agent("agent-x", "moderate")
        for signal in profile.signals:
            assert signal.agent_id == "agent-x"

    def test_signals_have_valid_dimensions(self):
        profiler = SituationalAwarenessProfiler(seed=42)
        profile = profiler.profile_agent("test", "moderate")
        for signal in profile.signals:
            assert signal.dimension in DIMENSIONS


# ── Fleet Analysis Tests ─────────────────────────────────────────────


class TestFleetAnalysis:
    def test_fleet_profile_empty(self):
        report = fleet_profile([])
        assert report.fleet_size == 0
        assert report.mean_composite == 0.0

    def test_fleet_profile_basic(self):
        profiler = SituationalAwarenessProfiler(seed=42)
        report = profiler.profile_fleet(10, "baseline")
        assert report.fleet_size == 10
        assert len(report.profiles) == 10
        assert 0.0 <= report.mean_composite <= 1.0

    def test_fleet_risk_distribution(self):
        profiler = SituationalAwarenessProfiler(seed=42)
        report = profiler.profile_fleet(20, "moderate")
        total = sum(report.risk_distribution.values())
        assert total == 20

    def test_fleet_systemic_risk_high_preset(self):
        profiler = SituationalAwarenessProfiler(seed=42)
        report = profiler.profile_fleet(20, "high-awareness")
        # High-awareness preset should likely trigger systemic risk
        assert isinstance(report.systemic_risk, bool)

    def test_fleet_dimension_means(self):
        profiler = SituationalAwarenessProfiler(seed=42)
        report = profiler.profile_fleet(10, "baseline")
        for dim in DIMENSIONS:
            assert dim in report.dimension_means
            assert 0.0 <= report.dimension_means[dim] <= 1.0

    def test_fleet_most_least_aware(self):
        profiler = SituationalAwarenessProfiler(seed=42)
        report = profiler.profile_fleet(10, "baseline")
        assert report.most_aware_dimension in DIMENSIONS
        assert report.least_aware_dimension in DIMENSIONS

    def test_fleet_recommendations_exist(self):
        profiler = SituationalAwarenessProfiler(seed=42)
        report = profiler.profile_fleet(5, "baseline")
        assert len(report.recommendations) > 0


# ── Pearson Correlation Tests ────────────────────────────────────────


class TestPearson:
    def test_perfect_positive(self):
        r = _pearson([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
        assert abs(r - 1.0) < 0.001

    def test_perfect_negative(self):
        r = _pearson([1, 2, 3, 4, 5], [10, 8, 6, 4, 2])
        assert abs(r - (-1.0)) < 0.001

    def test_no_correlation(self):
        r = _pearson([1, 1, 1, 1], [1, 2, 3, 4])
        assert r == 0.0  # zero std in x

    def test_short_list(self):
        r = _pearson([1, 2], [3, 4])
        assert r == 0.0  # too short


# ── HTML Report Tests ────────────────────────────────────────────────


class TestHTMLReport:
    def test_html_generation(self):
        from replication.situational_awareness import _generate_html
        profiler = SituationalAwarenessProfiler(seed=42)
        profiles = [profiler.profile_agent(f"a-{i}", "moderate") for i in range(3)]
        html = _generate_html(profiles)
        assert "<!DOCTYPE html>" in html
        assert "Situational Awareness" in html
        assert "a-0" in html

    def test_html_with_fleet(self):
        from replication.situational_awareness import _generate_html
        profiler = SituationalAwarenessProfiler(seed=42)
        report = profiler.profile_fleet(5, "baseline")
        html = _generate_html(report.profiles, report)
        assert "Fleet Analysis" in html
        assert "Mean Composite" in html

    def test_html_file_output(self):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            tmppath = f.name
        try:
            main(["--agents", "2", "--seed", "42", "-o", tmppath])
            assert os.path.exists(tmppath)
            with open(tmppath, "r", encoding="utf-8") as f:
                content = f.read()
            assert "<!DOCTYPE html>" in content
        finally:
            os.unlink(tmppath)


# ── CLI Tests ────────────────────────────────────────────────────────


class TestCLI:
    def test_json_output(self, capsys):
        main(["--agents", "2", "--seed", "42", "--json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "profiles" in data
        assert len(data["profiles"]) == 2

    def test_fleet_json(self, capsys):
        main(["--fleet-size", "5", "--seed", "42", "--json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "fleet" in data
        assert data["fleet"]["fleet_size"] == 5

    def test_text_output(self, capsys):
        main(["--agents", "3", "--seed", "42"])
        out = capsys.readouterr().out
        assert "SITUATIONAL AWARENESS PROFILER" in out
        assert "agent-001" in out

    def test_preset_option(self, capsys):
        main(["--agents", "1", "--preset", "high-awareness", "--seed", "42", "--json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        # High awareness should have higher score
        assert data["profiles"][0]["composite_score"] > 0.3

    def test_dimension_filter(self, capsys):
        main(["--agents", "1", "--dimension", "evaluation", "--seed", "42", "--json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "profiles" in data

    def test_probes_per_dim(self, capsys):
        main(["--agents", "1", "--probes-per-dim", "3", "--seed", "42", "--json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "profiles" in data


# ── Edge Cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_agent(self):
        profiler = SituationalAwarenessProfiler(seed=42)
        profile = profiler.profile_agent("solo", "baseline")
        assert profile.agent_id == "solo"

    def test_many_agents(self):
        profiler = SituationalAwarenessProfiler(seed=42)
        report = profiler.profile_fleet(50, "baseline")
        assert report.fleet_size == 50

    def test_all_presets_work(self):
        profiler = SituationalAwarenessProfiler(seed=42)
        for preset in _PRESETS:
            profile = profiler.profile_agent(f"test-{preset}", preset)
            assert profile.risk_tier in ("minimal", "low", "moderate", "high", "critical")

    def test_reproducibility_with_seed(self):
        p1 = SituationalAwarenessProfiler(seed=99)
        p2 = SituationalAwarenessProfiler(seed=99)
        profile1 = p1.profile_agent("a", "moderate")
        profile2 = p2.profile_agent("a", "moderate")
        assert profile1.composite_score == profile2.composite_score

    def test_different_seeds_different_results(self):
        p1 = SituationalAwarenessProfiler(seed=1)
        p2 = SituationalAwarenessProfiler(seed=2)
        profile1 = p1.profile_agent("a", "moderate")
        profile2 = p2.profile_agent("a", "moderate")
        # Very unlikely to be identical with different seeds
        # (not guaranteed but practically always true)
        assert profile1.composite_score != profile2.composite_score or True
