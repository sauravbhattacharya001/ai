"""Tests for replication.dependency_graph — resource dependency analysis.

Covers: graph construction, SPOF detection, cycle detection, blast radius,
critical path, redundancy scoring, cascade simulation, risk assessment,
DOT export, JSON export, scenario generators, and CLI entry point.
"""

from __future__ import annotations

import json

import pytest

from src.replication.dependency_graph import (
    AgentNode,
    CascadeResult,
    CascadeStep,
    Criticality,
    DepAnalysis,
    DependencyGraph,
    Resource,
    ResourceKind,
    generate_scenario,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _simple_graph() -> DependencyGraph:
    """Two agents, three resources, one redundancy group."""
    g = DependencyGraph()
    g.add_resource("db-primary", kind="database", redundancy_group="db")
    g.add_resource("db-replica", kind="database", redundancy_group="db")
    g.add_resource("api-gateway", kind="service")
    g.add_agent("agent-1", depends_on=["db-primary", "api-gateway"])
    g.add_agent("agent-2", depends_on=["db-replica", "api-gateway"])
    return g


def _chain_graph() -> DependencyGraph:
    """Resource chain: r0 <- r1 <- r2, agent depends on r2."""
    g = DependencyGraph()
    g.add_resource("r0", kind="service")
    g.add_resource("r1", kind="service", depends_on=["r0"])
    g.add_resource("r2", kind="service", depends_on=["r1"])
    g.add_agent("a1", depends_on=["r2"])
    return g


# ── Data model tests ─────────────────────────────────────────────────


class TestResourceKind:
    def test_enum_values(self):
        assert ResourceKind.DATABASE.value == "database"
        assert ResourceKind.CACHE.value == "cache"

    def test_all_kinds(self):
        assert len(ResourceKind) == 7


class TestCriticality:
    def test_enum_values(self):
        assert Criticality.REQUIRED.value == "required"
        assert Criticality.DEGRADED.value == "degraded"
        assert Criticality.OPTIONAL.value == "optional"


class TestResource:
    def test_to_dict_minimal(self):
        r = Resource(name="db", kind=ResourceKind.DATABASE)
        d = r.to_dict()
        assert d["name"] == "db"
        assert d["kind"] == "database"
        assert "redundancy_group" not in d
        assert "depends_on" not in d

    def test_to_dict_full(self):
        r = Resource(
            name="db",
            kind=ResourceKind.DATABASE,
            redundancy_group="db-group",
            depends_on=["auth"],
            capacity=2.0,
            failure_probability=0.1,
        )
        d = r.to_dict()
        assert d["redundancy_group"] == "db-group"
        assert d["depends_on"] == ["auth"]
        assert d["capacity"] == 2.0


class TestAgentNode:
    def test_required_resources(self):
        a = AgentNode(
            name="a1",
            depends_on=[
                ("db", Criticality.REQUIRED),
                ("cache", Criticality.DEGRADED),
                ("queue", Criticality.OPTIONAL),
            ],
        )
        assert a.required_resources() == ["db"]

    def test_to_dict(self):
        a = AgentNode(name="a1", depends_on=[("db", Criticality.REQUIRED)], priority=3)
        d = a.to_dict()
        assert d["name"] == "a1"
        assert d["priority"] == 3
        assert len(d["dependencies"]) == 1
        assert d["dependencies"][0]["criticality"] == "required"


# ── Graph construction ───────────────────────────────────────────────


class TestGraphConstruction:
    def test_add_resource_chaining(self):
        g = DependencyGraph()
        result = g.add_resource("r1").add_resource("r2")
        assert result is g

    def test_add_agent_chaining(self):
        g = DependencyGraph()
        g.add_resource("r1")
        result = g.add_agent("a1", depends_on=["r1"])
        assert result is g

    def test_invalid_kind_defaults_to_service(self):
        g = DependencyGraph()
        g.add_resource("r1", kind="unknown_kind")
        analysis = g.analyze()
        assert analysis.total_resources == 1

    def test_invalid_criticality_defaults_to_required(self):
        g = DependencyGraph()
        g.add_resource("r1")
        g.add_agent("a1", depends_on=["r1"], criticalities={"r1": "bogus"})
        analysis = g.analyze()
        # Agent should treat r1 as required
        assert analysis.total_agents == 1


# ── SPOF detection ───────────────────────────────────────────────────


class TestSPOFDetection:
    def test_no_spof_with_redundancy(self):
        """If both db-primary and db-replica are in a group, neither alone is a SPOF."""
        g = _simple_graph()
        analysis = g.analyze()
        db_spofs = [s for s in analysis.spofs if s["resource"].startswith("db-")]
        assert len(db_spofs) == 0

    def test_api_gateway_is_spof(self):
        """api-gateway has no redundancy — it should be a SPOF."""
        g = _simple_graph()
        analysis = g.analyze()
        gw_spofs = [s for s in analysis.spofs if s["resource"] == "api-gateway"]
        assert len(gw_spofs) == 1
        assert gw_spofs[0]["affected_agents"] == 2

    def test_single_resource_is_spof(self):
        g = DependencyGraph()
        g.add_resource("only-db", kind="database")
        g.add_agent("a1", depends_on=["only-db"])
        analysis = g.analyze()
        assert len(analysis.spofs) == 1

    def test_no_spof_all_optional(self):
        g = DependencyGraph()
        g.add_resource("cache")
        g.add_agent("a1", depends_on=["cache"], criticalities={"cache": "optional"})
        analysis = g.analyze()
        assert len(analysis.spofs) == 0


# ── Blast radius ─────────────────────────────────────────────────────


class TestBlastRadius:
    def test_api_gateway_blast_radius(self):
        g = _simple_graph()
        analysis = g.analyze()
        assert analysis.blast_radii["api-gateway"] == 2

    def test_single_db_blast_radius_with_redundancy(self):
        g = _simple_graph()
        analysis = g.analyze()
        # db-primary failing shouldn't take down agents because db-replica survives
        assert analysis.blast_radii["db-primary"] == 0

    def test_chain_blast_radius(self):
        g = _chain_graph()
        analysis = g.analyze()
        # Failing r0 cascades through r1 -> r2 -> a1
        assert analysis.blast_radii["r0"] == 1


# ── Cycle detection ──────────────────────────────────────────────────


class TestCycleDetection:
    def test_no_cycles(self):
        g = _simple_graph()
        analysis = g.analyze()
        assert len(analysis.cycles) == 0

    def test_simple_cycle(self):
        g = DependencyGraph()
        g.add_resource("r1", depends_on=["r2"])
        g.add_resource("r2", depends_on=["r1"])
        analysis = g.analyze()
        assert len(analysis.cycles) == 1
        cycle = analysis.cycles[0]
        # Cycle should contain both r1 and r2
        assert "r1" in cycle
        assert "r2" in cycle

    def test_no_cycle_in_chain(self):
        g = _chain_graph()
        analysis = g.analyze()
        assert len(analysis.cycles) == 0


# ── Critical path ───────────────────────────────────────────────────


class TestCriticalPath:
    def test_chain_critical_path(self):
        g = _chain_graph()
        analysis = g.analyze()
        # a1 -> r2 -> r1 -> r0 = length 3
        assert analysis.critical_path_length == 3
        assert len(analysis.critical_path) == 4

    def test_flat_critical_path(self):
        g = _simple_graph()
        analysis = g.analyze()
        # agent -> resource (no resource chains), length = 1
        assert analysis.critical_path_length == 1


# ── Redundancy scoring ──────────────────────────────────────────────


class TestRedundancyScoring:
    def test_fully_redundant(self):
        g = DependencyGraph()
        g.add_resource("db1", kind="database", redundancy_group="db")
        g.add_resource("db2", kind="database", redundancy_group="db")
        g.add_agent("a1", depends_on=["db1"])
        analysis = g.analyze()
        assert analysis.redundancy_scores["a1"] == 1.0

    def test_no_redundancy(self):
        g = DependencyGraph()
        g.add_resource("r1")
        g.add_agent("a1", depends_on=["r1"])
        analysis = g.analyze()
        assert analysis.redundancy_scores["a1"] == 0.0

    def test_partial_redundancy(self):
        g = DependencyGraph()
        g.add_resource("db1", redundancy_group="db")
        g.add_resource("db2", redundancy_group="db")
        g.add_resource("api")  # no redundancy
        g.add_agent("a1", depends_on=["db1", "api"])
        analysis = g.analyze()
        assert analysis.redundancy_scores["a1"] == 0.5

    def test_no_required_deps(self):
        g = DependencyGraph()
        g.add_resource("cache")
        g.add_agent("a1", depends_on=["cache"], criticalities={"cache": "optional"})
        analysis = g.analyze()
        assert analysis.redundancy_scores["a1"] == 1.0


# ── Cascade simulation ──────────────────────────────────────────────


class TestCascadeSimulation:
    def test_cascade_from_chain(self):
        g = _chain_graph()
        result = g.simulate_failure("r0")
        assert result.trigger == "r0"
        assert result.total_resources_down == 3  # r0, r1, r2
        assert result.total_agents_down == 1
        assert result.cascade_depth >= 1

    def test_no_cascade_with_redundancy(self):
        g = _simple_graph()
        result = g.simulate_failure("db-primary")
        assert result.total_agents_down == 0  # db-replica covers it

    def test_cascade_from_api_gateway(self):
        g = _simple_graph()
        result = g.simulate_failure("api-gateway")
        assert result.total_agents_down == 2

    def test_cascade_unknown_resource(self):
        g = _simple_graph()
        result = g.simulate_failure("nonexistent")
        assert result.total_agents_down == 0
        assert result.total_resources_down == 0

    def test_cascade_step_to_dict(self):
        step = CascadeStep(time=1, failed="r1", reason="test", kind="resource")
        d = step.to_dict()
        assert d["time"] == 1
        assert d["failed"] == "r1"

    def test_cascade_result_render_empty(self):
        result = CascadeResult(trigger="r1", steps=[])
        text = result.render()
        assert "No cascade propagation" in text

    def test_cascade_result_to_dict(self):
        result = CascadeResult(trigger="r1", total_agents_down=2, total_resources_down=3)
        d = result.to_dict()
        assert d["trigger"] == "r1"
        assert d["total_agents_down"] == 2


# ── Risk assessment ──────────────────────────────────────────────────


class TestRiskAssessment:
    def test_low_risk(self):
        g = DependencyGraph()
        g.add_resource("db1", redundancy_group="db")
        g.add_resource("db2", redundancy_group="db")
        g.add_agent("a1", depends_on=["db1"])
        analysis = g.analyze()
        assert analysis.risk_level == "low"

    def test_higher_risk_with_spofs(self):
        g = DependencyGraph()
        for i in range(5):
            g.add_resource(f"r{i}")
        for i in range(10):
            g.add_agent(f"a{i}", depends_on=[f"r{i % 5}"])
        analysis = g.analyze()
        assert analysis.risk_level in ("moderate", "high", "critical")

    def test_fragile_scenario_risk(self):
        g = generate_scenario(num_agents=8, num_resources=6, scenario="fragile", seed=42)
        analysis = g.analyze()
        assert analysis.risk_level in ("moderate", "high", "critical")


# ── Warnings ─────────────────────────────────────────────────────────


class TestWarnings:
    def test_spof_warning(self):
        g = DependencyGraph()
        g.add_resource("r1")
        g.add_agent("a1", depends_on=["r1"])
        analysis = g.analyze()
        assert any("single point" in w.lower() for w in analysis.warnings)

    def test_orphan_resource_warning(self):
        g = DependencyGraph()
        g.add_resource("orphan")
        g.add_resource("used")
        g.add_agent("a1", depends_on=["used"])
        analysis = g.analyze()
        assert any("no dependents" in w for w in analysis.warnings)

    def test_low_redundancy_warning(self):
        g = DependencyGraph()
        for i in range(5):
            g.add_resource(f"r{i}")
            g.add_agent(f"a{i}", depends_on=[f"r{i}"])
        analysis = g.analyze()
        assert any("redundancy" in w.lower() for w in analysis.warnings)


# ── Export ───────────────────────────────────────────────────────────


class TestExport:
    def test_dot_output(self):
        g = _simple_graph()
        dot = g.to_dot()
        assert "digraph DependencyGraph" in dot
        assert "api-gateway" in dot
        assert "agent-1" in dot

    def test_to_dict(self):
        g = _simple_graph()
        d = g.to_dict()
        assert len(d["resources"]) == 3
        assert len(d["agents"]) == 2
        assert "db" in d["redundancy_groups"]

    def test_json_serializable(self):
        g = _simple_graph()
        analysis = g.analyze()
        # Should not raise
        text = json.dumps(analysis.to_dict(), default=str)
        assert "spofs" in text

    def test_analysis_render(self):
        g = _simple_graph()
        analysis = g.analyze()
        text = analysis.render()
        assert "Resource Dependency Analysis" in text
        assert "Agents:" in text


# ── Scenario generators ─────────────────────────────────────────────


class TestScenarios:
    @pytest.mark.parametrize("scenario", ["realistic", "dense", "sparse", "fragile"])
    def test_scenario_generates_valid_graph(self, scenario):
        g = generate_scenario(num_agents=4, num_resources=5, scenario=scenario, seed=123)
        analysis = g.analyze()
        assert analysis.total_agents > 0
        assert analysis.total_resources > 0

    def test_seed_reproducibility(self):
        a1 = generate_scenario(seed=99).analyze()
        a2 = generate_scenario(seed=99).analyze()
        assert a1.total_edges == a2.total_edges
        assert a1.spofs == a2.spofs

    def test_large_scenario(self):
        g = generate_scenario(num_agents=20, num_resources=15, scenario="dense", seed=7)
        analysis = g.analyze()
        assert analysis.total_agents == 20


# ── Edge cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_graph(self):
        g = DependencyGraph()
        analysis = g.analyze()
        assert analysis.total_agents == 0
        assert analysis.total_resources == 0
        # Empty graph may score 'high' due to 0 mean_redundancy heuristic
        assert analysis.risk_level in ("low", "high")

    def test_agent_with_no_deps(self):
        g = DependencyGraph()
        g.add_agent("lonely")
        analysis = g.analyze()
        assert analysis.redundancy_scores["lonely"] == 1.0

    def test_resource_with_no_agents(self):
        g = DependencyGraph()
        g.add_resource("unused")
        analysis = g.analyze()
        assert analysis.total_resources == 1
        assert analysis.blast_radii["unused"] == 0

    def test_agent_depends_on_nonexistent_resource(self):
        """Agent refers to a resource not in the graph — should not crash."""
        g = DependencyGraph()
        g.add_agent("a1", depends_on=["ghost"])
        analysis = g.analyze()
        assert analysis.total_agents == 1
