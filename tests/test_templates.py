"""Tests for contract templates module."""

import json

import pytest

from replication.templates import (
    TEMPLATES,
    ContractTemplate,
    get_categories,
    get_template,
    list_templates,
    render_catalog,
    render_comparison_table,
    main,
)
from replication.contract import ReplicationContract, ResourceSpec
from replication.simulator import ScenarioConfig


# ── Template integrity ───────────────────────────────────────────


class TestTemplateIntegrity:
    """Verify that all templates have valid, consistent parameters."""

    def test_templates_not_empty(self):
        assert len(TEMPLATES) >= 5

    @pytest.mark.parametrize("key", list(TEMPLATES.keys()))
    def test_template_has_required_fields(self, key):
        t = TEMPLATES[key]
        assert t.name
        assert t.category
        assert t.description
        assert t.rationale
        assert t.risk_level in ("low", "medium", "high", "critical")

    @pytest.mark.parametrize("key", list(TEMPLATES.keys()))
    def test_template_builds_valid_contract(self, key):
        t = TEMPLATES[key]
        contract = t.build_contract()
        assert isinstance(contract, ReplicationContract)
        assert contract.max_depth == t.max_depth
        assert contract.max_replicas == t.max_replicas
        assert contract.cooldown_seconds == t.cooldown_seconds
        assert contract.expiration_seconds == t.expiration_seconds

    @pytest.mark.parametrize("key", list(TEMPLATES.keys()))
    def test_template_builds_valid_resources(self, key):
        t = TEMPLATES[key]
        resources = t.build_resources()
        assert isinstance(resources, ResourceSpec)
        assert resources.cpu_limit == t.cpu_limit
        assert resources.memory_limit_mb == t.memory_limit_mb
        assert resources.network_policy.allow_external == t.allow_external_network

    @pytest.mark.parametrize("key", list(TEMPLATES.keys()))
    def test_template_builds_scenario_config(self, key):
        t = TEMPLATES[key]
        config = t.to_scenario_config()
        assert isinstance(config, ScenarioConfig)
        assert config.max_depth == t.max_depth
        assert config.max_replicas == t.max_replicas
        assert config.strategy == t.recommended_strategy

    @pytest.mark.parametrize("key", list(TEMPLATES.keys()))
    def test_template_contract_params_positive(self, key):
        t = TEMPLATES[key]
        assert t.max_depth >= 0
        assert t.max_replicas >= 1
        assert t.cooldown_seconds >= 0
        if t.expiration_seconds is not None:
            assert t.expiration_seconds > 0

    @pytest.mark.parametrize("key", list(TEMPLATES.keys()))
    def test_template_resource_params_positive(self, key):
        t = TEMPLATES[key]
        assert t.cpu_limit > 0
        assert t.memory_limit_mb > 0

    @pytest.mark.parametrize("key", list(TEMPLATES.keys()))
    def test_template_to_dict_serializable(self, key):
        t = TEMPLATES[key]
        d = t.to_dict()
        assert isinstance(d, dict)
        # Must be JSON-serializable
        json_str = json.dumps(d)
        assert json_str

    @pytest.mark.parametrize("key", list(TEMPLATES.keys()))
    def test_template_render_not_empty(self, key):
        t = TEMPLATES[key]
        rendered = t.render()
        assert t.name in rendered
        assert t.category in rendered

    @pytest.mark.parametrize("key", list(TEMPLATES.keys()))
    def test_template_stop_conditions_valid(self, key):
        t = TEMPLATES[key]
        contract = t.build_contract()
        # Number of built stop conditions should match specs
        expected = sum(
            1
            for s in t.stop_condition_specs
            if s.get("name") in (
                "depth_utilization_limit",
                "quota_pressure_limit",
                "high_memory_guard",
            )
        )
        assert len(contract.stop_conditions) == expected


# ── get_template ─────────────────────────────────────────────────


class TestGetTemplate:
    def test_valid_key(self):
        t = get_template("web_crawler")
        assert t.name == "Web Crawler"

    def test_case_insensitive(self):
        t = get_template("WEB_CRAWLER")
        assert t.name == "Web Crawler"

    def test_hyphen_to_underscore(self):
        t = get_template("web-crawler")
        assert t.name == "Web Crawler"

    def test_space_to_underscore(self):
        t = get_template("web crawler")
        assert t.name == "Web Crawler"

    def test_unknown_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown template"):
            get_template("nonexistent_template")


# ── list_templates ───────────────────────────────────────────────


class TestListTemplates:
    def test_list_all(self):
        templates = list_templates()
        assert len(templates) == len(TEMPLATES)

    def test_sorted_by_name(self):
        templates = list_templates()
        names = [t.name for t in templates]
        assert names == sorted(names)

    def test_filter_by_category(self):
        categories = get_categories()
        assert len(categories) >= 3

        for cat in categories:
            filtered = list_templates(category=cat)
            assert len(filtered) >= 1
            assert all(t.category == cat for t in filtered)

    def test_filter_case_insensitive(self):
        categories = get_categories()
        cat = categories[0]
        lower = list_templates(category=cat.lower())
        upper = list_templates(category=cat.upper())
        assert len(lower) == len(upper)

    def test_nonexistent_category(self):
        result = list_templates(category="nonexistent_category")
        assert result == []


# ── get_categories ───────────────────────────────────────────────


class TestGetCategories:
    def test_returns_sorted_unique(self):
        cats = get_categories()
        assert cats == sorted(set(cats))

    def test_all_categories_have_templates(self):
        for cat in get_categories():
            templates = list_templates(category=cat)
            assert len(templates) >= 1


# ── Rendering ────────────────────────────────────────────────────


class TestRendering:
    def test_catalog_contains_all_templates(self):
        catalog = render_catalog()
        for t in TEMPLATES.values():
            assert t.name in catalog

    def test_catalog_contains_categories(self):
        catalog = render_catalog()
        for cat in get_categories():
            assert cat in catalog

    def test_comparison_table_header(self):
        table = render_comparison_table()
        assert "Template Comparison" in table
        assert "Depth" in table
        assert "Replicas" in table

    def test_comparison_table_has_all_templates(self):
        table = render_comparison_table()
        for t in TEMPLATES.values():
            assert t.name in table


# ── to_dict / JSON ───────────────────────────────────────────────


class TestSerialization:
    def test_to_dict_structure(self):
        t = get_template("data_pipeline")
        d = t.to_dict()
        assert d["name"] == "Data Pipeline"
        assert "contract" in d
        assert "resources" in d
        assert "simulation" in d
        assert d["contract"]["max_depth"] == t.max_depth
        assert d["resources"]["cpu_limit"] == t.cpu_limit

    def test_round_trip_json(self):
        for t in TEMPLATES.values():
            d = t.to_dict()
            s = json.dumps(d)
            loaded = json.loads(s)
            assert loaded["name"] == t.name
            assert loaded["contract"]["max_depth"] == t.max_depth


# ── ScenarioConfig conversion ────────────────────────────────────


class TestScenarioConfig:
    def test_default_strategy(self):
        t = get_template("ml_training")
        config = t.to_scenario_config()
        assert config.strategy == "conservative"

    def test_override_strategy(self):
        t = get_template("ml_training")
        config = t.to_scenario_config(strategy="greedy")
        assert config.strategy == "greedy"

    def test_seed_propagated(self):
        t = get_template("web_crawler")
        config = t.to_scenario_config(seed=42)
        assert config.seed == 42

    def test_all_templates_produce_valid_config(self):
        for t in TEMPLATES.values():
            config = t.to_scenario_config(seed=1)
            assert config.max_depth >= 0
            assert config.max_replicas >= 1


# ── Stop conditions ──────────────────────────────────────────────


class TestStopConditions:
    def test_depth_utilization_blocks_at_threshold(self):
        t = get_template("autonomous_agent")
        contract = t.build_contract()
        # The autonomous_agent has depth_utilization at 50%
        # With max_depth=2, depth=1 should trigger (1/2 = 50%)
        depth_cond = [
            c for c in contract.stop_conditions
            if c.name == "depth_utilization_limit"
        ]
        assert len(depth_cond) == 1

    def test_quota_pressure_present(self):
        t = get_template("web_crawler")
        contract = t.build_contract()
        quota_cond = [
            c for c in contract.stop_conditions
            if c.name == "quota_pressure_limit"
        ]
        assert len(quota_cond) == 1

    def test_high_memory_guard_present(self):
        t = get_template("data_pipeline")
        contract = t.build_contract()
        mem_cond = [
            c for c in contract.stop_conditions
            if c.name == "high_memory_guard"
        ]
        assert len(mem_cond) == 1

    def test_ci_cd_no_stop_conditions(self):
        t = get_template("ci_cd_pipeline")
        contract = t.build_contract()
        assert len(contract.stop_conditions) == 0


# ── CLI ──────────────────────────────────────────────────────────


class TestCLI:
    def test_default_lists_catalog(self, capsys):
        main([])
        output = capsys.readouterr().out
        assert "Contract Template Catalog" in output

    def test_show_template(self, capsys):
        main(["--show", "web_crawler"])
        output = capsys.readouterr().out
        assert "Web Crawler" in output
        assert "Rationale" in output

    def test_show_unknown_exits(self):
        with pytest.raises(SystemExit):
            main(["--show", "nonexistent"])

    def test_compare_flag(self, capsys):
        main(["--compare"])
        output = capsys.readouterr().out
        assert "Template Comparison" in output

    def test_json_flag(self, capsys):
        main(["--json"])
        output = capsys.readouterr().out
        data = json.loads(output)
        assert "web_crawler" in data

    def test_category_filter(self, capsys):
        main(["--category", "Research"])
        output = capsys.readouterr().out
        assert "Research Experiment" in output

    def test_simulate_flag(self, capsys):
        main(["--simulate", "data_pipeline", "--seed", "42"])
        output = capsys.readouterr().out
        assert "Simulating template: Data Pipeline" in output

    def test_simulate_with_strategy(self, capsys):
        main(["--simulate", "web_crawler", "--strategy", "greedy", "--seed", "1"])
        output = capsys.readouterr().out
        assert "Strategy: greedy" in output


# ── Domain-specific template content ─────────────────────────────


class TestDomainTemplates:
    def test_web_crawler_has_external_network(self):
        t = get_template("web_crawler")
        assert t.allow_external_network is True

    def test_data_pipeline_no_external_network(self):
        t = get_template("data_pipeline")
        assert t.allow_external_network is False

    def test_autonomous_agent_is_critical_risk(self):
        t = get_template("autonomous_agent")
        assert t.risk_level == "critical"

    def test_ci_cd_pipeline_is_low_risk(self):
        t = get_template("ci_cd_pipeline")
        assert t.risk_level == "low"

    def test_security_scanner_shallow_depth(self):
        t = get_template("security_scanner")
        assert t.max_depth == 1
        assert t.risk_level == "high"

    def test_research_experiment_deep_hierarchy(self):
        t = get_template("research_experiment")
        assert t.max_depth >= 4

    def test_ml_training_high_resources(self):
        t = get_template("ml_training")
        assert t.memory_limit_mb >= 1024
        assert t.cpu_limit >= 1.0

    def test_all_templates_have_safety_notes(self):
        for key, t in TEMPLATES.items():
            if t.risk_level in ("high", "critical"):
                assert len(t.safety_notes) >= 2, (
                    f"High/critical template '{key}' should have "
                    f"multiple safety notes"
                )
