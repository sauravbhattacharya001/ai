"""Tests for the scenario generator module."""

import json
import pytest

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from replication.scenarios import (
    ScenarioGenerator,
    GeneratorConfig,
    GeneratedScenario,
    ScenarioSuite,
    ScenarioCategory,
    score_scenario,
    generate_boundary_scenarios,
    generate_adversarial_scenarios,
    generate_random_scenarios,
    generate_gradient_scenarios,
    _risk_label,
    _make_config,
    PARAM_RANGES,
    STRATEGIES,
)
from replication.simulator import ScenarioConfig, Simulator


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def default_gen():
    """Generator with default config and fixed seed."""
    return ScenarioGenerator(GeneratorConfig(count=8, seed=42))


@pytest.fixture
def small_suite(default_gen):
    """A small generated suite for testing."""
    return default_gen.generate()


@pytest.fixture
def nosim_gen():
    """Generator that skips simulations."""
    return ScenarioGenerator(GeneratorConfig(count=8, seed=42, run_simulations=False))


# ── GeneratorConfig tests ─────────────────────────────────────────


class TestGeneratorConfig:
    def test_defaults(self):
        config = GeneratorConfig()
        assert config.count == 15
        assert config.seed is None
        assert config.category is None
        assert config.run_simulations is True
        assert config.top_n == 10

    def test_custom_values(self):
        config = GeneratorConfig(count=5, seed=99, category="boundary")
        assert config.count == 5
        assert config.seed == 99
        assert config.category == "boundary"


# ── ScenarioCategory tests ───────────────────────────────────────


class TestScenarioCategory:
    def test_all_categories(self):
        assert ScenarioCategory.BOUNDARY.value == "boundary"
        assert ScenarioCategory.ADVERSARIAL.value == "adversarial"
        assert ScenarioCategory.RANDOM.value == "random"
        assert ScenarioCategory.GRADIENT.value == "gradient"

    def test_from_string(self):
        assert ScenarioCategory("boundary") == ScenarioCategory.BOUNDARY
        assert ScenarioCategory("adversarial") == ScenarioCategory.ADVERSARIAL


# ── _make_config tests ───────────────────────────────────────────


class TestMakeConfig:
    def test_defaults(self):
        config = _make_config()
        assert config.max_depth == 3
        assert config.max_replicas == 10
        assert config.strategy == "greedy"

    def test_overrides(self):
        config = _make_config(max_depth=7, strategy="chain", seed=100)
        assert config.max_depth == 7
        assert config.strategy == "chain"
        assert config.seed == 100


# ── _risk_label tests ─────────────────────────────────────────────


class TestRiskLabel:
    def test_high(self):
        assert _risk_label(70) == "HIGH"
        assert _risk_label(100) == "HIGH"

    def test_medium(self):
        assert _risk_label(40) == "MEDIUM"
        assert _risk_label(69) == "MEDIUM"

    def test_low(self):
        assert _risk_label(0) == "LOW"
        assert _risk_label(39) == "LOW"


# ── Boundary generator tests ─────────────────────────────────────


class TestBoundaryScenarios:
    def test_generates_correct_count(self):
        import random
        rng = random.Random(42)
        scenarios = generate_boundary_scenarios(rng, 5, 1000)
        assert len(scenarios) == 5

    def test_all_are_boundary_category(self):
        import random
        rng = random.Random(42)
        scenarios = generate_boundary_scenarios(rng, 10, 1000)
        for s in scenarios:
            assert s.category == ScenarioCategory.BOUNDARY

    def test_names_prefixed(self):
        import random
        rng = random.Random(42)
        scenarios = generate_boundary_scenarios(rng, 5, 1000)
        for s in scenarios:
            assert s.name.startswith("boundary-")

    def test_configs_are_valid(self):
        import random
        rng = random.Random(42)
        scenarios = generate_boundary_scenarios(rng, 10, 1000)
        for s in scenarios:
            assert s.config.max_depth >= 1
            assert s.config.max_replicas >= 1

    def test_descriptions_not_empty(self):
        import random
        rng = random.Random(42)
        scenarios = generate_boundary_scenarios(rng, 5, 1000)
        for s in scenarios:
            assert len(s.description) > 0


# ── Adversarial generator tests ──────────────────────────────────


class TestAdversarialScenarios:
    def test_generates_correct_count(self):
        import random
        rng = random.Random(42)
        scenarios = generate_adversarial_scenarios(rng, 5, 1000)
        assert len(scenarios) == 5

    def test_all_are_adversarial_category(self):
        import random
        rng = random.Random(42)
        scenarios = generate_adversarial_scenarios(rng, 8, 1000)
        for s in scenarios:
            assert s.category == ScenarioCategory.ADVERSARIAL

    def test_has_rationale(self):
        import random
        rng = random.Random(42)
        scenarios = generate_adversarial_scenarios(rng, 5, 1000)
        for s in scenarios:
            assert len(s.rationale) > 10

    def test_max_count_capped(self):
        import random
        rng = random.Random(42)
        scenarios = generate_adversarial_scenarios(rng, 100, 1000)
        assert len(scenarios) == 10  # only 10 adversarial defs


# ── Random generator tests ───────────────────────────────────────


class TestRandomScenarios:
    def test_generates_exact_count(self):
        import random
        rng = random.Random(42)
        scenarios = generate_random_scenarios(rng, 20, 1000)
        assert len(scenarios) == 20

    def test_valid_strategies(self):
        import random
        rng = random.Random(42)
        scenarios = generate_random_scenarios(rng, 30, 1000)
        for s in scenarios:
            assert s.config.strategy in STRATEGIES

    def test_params_within_ranges(self):
        import random
        rng = random.Random(42)
        scenarios = generate_random_scenarios(rng, 30, 1000)
        for s in scenarios:
            assert PARAM_RANGES["max_depth"][0] <= s.config.max_depth <= PARAM_RANGES["max_depth"][1]
            assert PARAM_RANGES["max_replicas"][0] <= s.config.max_replicas <= PARAM_RANGES["max_replicas"][1]

    def test_sequential_names(self):
        import random
        rng = random.Random(42)
        scenarios = generate_random_scenarios(rng, 5, 1000)
        for i, s in enumerate(scenarios, 1):
            assert s.name == f"random-{i:03d}"


# ── Gradient generator tests ─────────────────────────────────────


class TestGradientScenarios:
    def test_generates_correct_count(self):
        import random
        rng = random.Random(42)
        scenarios = generate_gradient_scenarios(rng, 3, 1000)
        assert len(scenarios) == 3

    def test_all_are_gradient_category(self):
        import random
        rng = random.Random(42)
        scenarios = generate_gradient_scenarios(rng, 5, 1000)
        for s in scenarios:
            assert s.category == ScenarioCategory.GRADIENT

    def test_names_contain_walk(self):
        import random
        rng = random.Random(42)
        scenarios = generate_gradient_scenarios(rng, 5, 1000)
        for s in scenarios:
            assert "walk" in s.name

    def test_max_count_capped(self):
        import random
        rng = random.Random(42)
        scenarios = generate_gradient_scenarios(rng, 100, 1000)
        assert len(scenarios) == 5  # only 5 walks defined


# ── score_scenario tests ─────────────────────────────────────────


class TestScoreScenario:
    def test_no_report_scores_zero(self):
        sc = GeneratedScenario(
            name="test",
            category=ScenarioCategory.RANDOM,
            config=_make_config(),
            description="test",
            rationale="test",
            report=None,
        )
        assert score_scenario(sc) == 0.0

    def test_greedy_scores_higher_than_conservative(self):
        greedy_config = ScenarioConfig(strategy="greedy", max_depth=3, max_replicas=10, seed=42)
        conservative_config = ScenarioConfig(strategy="conservative", max_depth=3, max_replicas=10, seed=42)

        greedy_sc = GeneratedScenario(
            name="greedy", category=ScenarioCategory.RANDOM,
            config=greedy_config, description="", rationale="",
            report=Simulator(greedy_config).run(),
        )
        conservative_sc = GeneratedScenario(
            name="conservative", category=ScenarioCategory.RANDOM,
            config=conservative_config, description="", rationale="",
            report=Simulator(conservative_config).run(),
        )

        greedy_score = score_scenario(greedy_sc)
        conservative_score = score_scenario(conservative_sc)
        assert greedy_score >= conservative_score

    def test_score_capped_at_100(self):
        config = ScenarioConfig(strategy="greedy", max_depth=10, max_replicas=50, seed=42)
        sc = GeneratedScenario(
            name="extreme", category=ScenarioCategory.ADVERSARIAL,
            config=config, description="", rationale="",
            report=Simulator(config).run(),
        )
        score = score_scenario(sc)
        assert score <= 100.0

    def test_score_non_negative(self):
        config = ScenarioConfig(strategy="conservative", max_depth=1, max_replicas=1, seed=42)
        sc = GeneratedScenario(
            name="minimal", category=ScenarioCategory.BOUNDARY,
            config=config, description="", rationale="",
            report=Simulator(config).run(),
        )
        score = score_scenario(sc)
        assert score >= 0.0

    def test_safety_notes_populated(self):
        config = ScenarioConfig(strategy="greedy", max_depth=5, max_replicas=30, seed=42)
        sc = GeneratedScenario(
            name="test", category=ScenarioCategory.RANDOM,
            config=config, description="", rationale="",
            report=Simulator(config).run(),
        )
        score_scenario(sc)
        # Should have at least some notes for an aggressive config
        assert isinstance(sc.safety_notes, list)


# ── ScenarioGenerator tests ──────────────────────────────────────


class TestScenarioGenerator:
    def test_generates_correct_count(self, default_gen):
        suite = default_gen.generate()
        assert len(suite.scenarios) == 8

    def test_deterministic_with_seed(self):
        gen1 = ScenarioGenerator(GeneratorConfig(count=5, seed=42))
        gen2 = ScenarioGenerator(GeneratorConfig(count=5, seed=42))
        suite1 = gen1.generate()
        suite2 = gen2.generate()
        assert len(suite1.scenarios) == len(suite2.scenarios)
        for s1, s2 in zip(suite1.scenarios, suite2.scenarios):
            assert s1.name == s2.name
            assert s1.category == s2.category

    def test_mixed_categories(self, small_suite):
        categories = {s.category for s in small_suite.scenarios}
        # With 8 scenarios, should have at least 2 categories
        assert len(categories) >= 2

    def test_all_simulated(self, small_suite):
        for s in small_suite.scenarios:
            assert s.report is not None

    def test_no_simulation_mode(self, nosim_gen):
        suite = nosim_gen.generate()
        for s in suite.scenarios:
            assert s.report is None
            assert s.interest_score == 0.0

    def test_category_filter_boundary(self):
        gen = ScenarioGenerator(GeneratorConfig(count=5, seed=42, category="boundary"))
        suite = gen.generate()
        for s in suite.scenarios:
            assert s.category == ScenarioCategory.BOUNDARY

    def test_category_filter_adversarial(self):
        gen = ScenarioGenerator(GeneratorConfig(count=5, seed=42, category="adversarial"))
        suite = gen.generate()
        for s in suite.scenarios:
            assert s.category == ScenarioCategory.ADVERSARIAL

    def test_category_filter_random(self):
        gen = ScenarioGenerator(GeneratorConfig(count=5, seed=42, category="random"))
        suite = gen.generate()
        for s in suite.scenarios:
            assert s.category == ScenarioCategory.RANDOM

    def test_category_filter_gradient(self):
        gen = ScenarioGenerator(GeneratorConfig(count=5, seed=42, category="gradient"))
        suite = gen.generate()
        for s in suite.scenarios:
            assert s.category == ScenarioCategory.GRADIENT

    def test_stress_test_generation(self, default_gen):
        suite = default_gen.generate_stress_test(num_scenarios=10)
        assert len(suite.scenarios) == 10
        # Should be mostly adversarial
        adversarial_count = sum(
            1 for s in suite.scenarios if s.category == ScenarioCategory.ADVERSARIAL
        )
        assert adversarial_count >= 5  # at least 50%


# ── ScenarioSuite tests ──────────────────────────────────────────


class TestScenarioSuite:
    def test_ranked_returns_descending(self, small_suite):
        ranked = small_suite.ranked()
        for i in range(1, len(ranked)):
            assert ranked[i - 1].interest_score >= ranked[i].interest_score

    def test_ranked_top_n(self, small_suite):
        top3 = small_suite.ranked(top_n=3)
        assert len(top3) == 3

    def test_by_category(self, small_suite):
        boundary = small_suite.by_category(ScenarioCategory.BOUNDARY)
        for s in boundary:
            assert s.category == ScenarioCategory.BOUNDARY

    def test_highest_risk(self, small_suite):
        highest = small_suite.highest_risk()
        assert highest is not None
        ranked = small_suite.ranked()
        assert highest.interest_score == ranked[0].interest_score

    def test_highest_risk_empty(self):
        suite = ScenarioSuite(
            scenarios=[], config=GeneratorConfig(),
            generation_time_ms=0, category_counts={}
        )
        assert suite.highest_risk() is None

    def test_safety_summary_keys(self, small_suite):
        summary = small_suite.safety_summary()
        assert "total" in summary
        assert "simulated" in summary
        assert "high_risk_count" in summary
        assert "max_workers_seen" in summary

    def test_safety_summary_empty(self):
        suite = ScenarioSuite(
            scenarios=[], config=GeneratorConfig(),
            generation_time_ms=0, category_counts={}
        )
        summary = suite.safety_summary()
        assert summary["total"] == 0

    def test_category_counts(self, small_suite):
        assert isinstance(small_suite.category_counts, dict)
        total = sum(small_suite.category_counts.values())
        assert total == len(small_suite.scenarios)


# ── Rendering tests ──────────────────────────────────────────────


class TestRendering:
    def test_render_not_empty(self, small_suite):
        report = small_suite.render()
        assert len(report) > 100
        assert "SCENARIO GENERATOR REPORT" in report
        assert "Categories" in report
        assert "Scenarios" in report

    def test_render_ranking(self, small_suite):
        ranking = small_suite.render_ranking()
        assert "SCENARIO RANKING" in ranking
        assert "Rank" in ranking
        assert "Score" in ranking

    def test_render_includes_risk_labels(self, small_suite):
        report = small_suite.render()
        # Should contain at least one risk label
        assert any(label in report for label in ["HIGH", "MEDIUM", "LOW"])


# ── Serialization tests ──────────────────────────────────────────


class TestSerialization:
    def test_scenario_to_dict(self, small_suite):
        for sc in small_suite.scenarios:
            d = sc.to_dict()
            assert d["name"] == sc.name
            assert d["category"] == sc.category.value
            assert "config" in d
            assert d["config"]["strategy"] in STRATEGIES

    def test_scenario_to_dict_with_results(self, small_suite):
        simulated = [s for s in small_suite.scenarios if s.report is not None]
        assert len(simulated) > 0
        d = simulated[0].to_dict()
        assert "results" in d
        assert "total_workers" in d["results"]

    def test_scenario_to_dict_without_results(self, nosim_gen):
        suite = nosim_gen.generate()
        d = suite.scenarios[0].to_dict()
        assert "results" not in d

    def test_suite_to_dict(self, small_suite):
        d = small_suite.to_dict()
        assert "scenarios" in d
        assert "safety_summary" in d
        assert "generation_time_ms" in d
        assert "category_counts" in d

    def test_json_serializable(self, small_suite):
        d = small_suite.to_dict()
        serialized = json.dumps(d)
        parsed = json.loads(serialized)
        assert len(parsed["scenarios"]) == len(small_suite.scenarios)


# ── Integration tests ────────────────────────────────────────────


class TestIntegration:
    def test_full_pipeline(self):
        """End-to-end: generate, simulate, score, rank, render."""
        gen = ScenarioGenerator(GeneratorConfig(count=6, seed=42))
        suite = gen.generate()

        assert len(suite.scenarios) == 6
        assert all(s.report is not None for s in suite.scenarios)
        assert all(s.interest_score >= 0 for s in suite.scenarios)

        ranked = suite.ranked()
        assert ranked[0].interest_score >= ranked[-1].interest_score

        report = suite.render()
        assert len(report) > 200

        ranking = suite.render_ranking()
        assert "Rank" in ranking

        d = suite.to_dict()
        json.dumps(d)  # should not raise

    def test_stress_test_pipeline(self):
        """Stress test: more scenarios, still works."""
        gen = ScenarioGenerator(GeneratorConfig(seed=42))
        suite = gen.generate_stress_test(num_scenarios=15)

        assert len(suite.scenarios) == 15
        summary = suite.safety_summary()
        assert summary["simulated"] == 15

    def test_import_from_package(self):
        """Verify the module is importable from the package."""
        from replication import ScenarioGenerator, GeneratorConfig, ScenarioSuite
        gen = ScenarioGenerator(GeneratorConfig(count=3, seed=42))
        suite = gen.generate()
        assert isinstance(suite, ScenarioSuite)
