"""Tests for replication.trust_propagation module."""

from __future__ import annotations

import json
import math

import pytest

from replication.trust_propagation import (
    AgentRole,
    InteractionOutcome,
    ThreatType,
    TrustAgent,
    TrustEdge,
    TrustNetwork,
    TrustReport,
    _outcome_delta,
    _severity,
    _format_report,
    main,
    Interaction,
)


# ── Helper fixtures ──────────────────────────────────────────────────


@pytest.fixture
def network():
    """Empty trust network with fixed seed for reproducibility."""
    return TrustNetwork(seed=42)


@pytest.fixture
def three_agent_network(network):
    """Network with three connected agents."""
    network.add_agent(TrustAgent("a", role="worker"))
    network.add_agent(TrustAgent("b", role="validator"))
    network.add_agent(TrustAgent("c", role="controller"))
    return network


# ── Enum tests ───────────────────────────────────────────────────────


class TestEnums:
    def test_agent_role_values(self):
        assert AgentRole.CONTROLLER.value == "controller"
        assert AgentRole.WORKER.value == "worker"
        assert AgentRole.VALIDATOR.value == "validator"
        assert AgentRole.OBSERVER.value == "observer"

    def test_interaction_outcome_values(self):
        assert InteractionOutcome.SUCCESS.value == "success"
        assert InteractionOutcome.BETRAYAL.value == "betrayal"

    def test_threat_type_values(self):
        assert ThreatType.SYBIL.value == "sybil"
        assert ThreatType.TRUST_LAUNDERING.value == "trust_laundering"
        assert ThreatType.COLLUSION_RING.value == "collusion_ring"
        assert ThreatType.ECLIPSE.value == "eclipse"


# ── Helper function tests ────────────────────────────────────────────


class TestHelpers:
    def test_outcome_delta_success(self):
        assert _outcome_delta(InteractionOutcome.SUCCESS) == 0.8

    def test_outcome_delta_cooperation(self):
        assert _outcome_delta(InteractionOutcome.COOPERATION) == 1.0

    def test_outcome_delta_neutral(self):
        assert _outcome_delta(InteractionOutcome.NEUTRAL) == 0.0

    def test_outcome_delta_failure(self):
        assert _outcome_delta(InteractionOutcome.FAILURE) == -0.5

    def test_outcome_delta_betrayal(self):
        assert _outcome_delta(InteractionOutcome.BETRAYAL) == -1.0

    def test_severity_critical(self):
        assert _severity(0.9) == "critical"
        assert _severity(0.8) == "critical"

    def test_severity_high(self):
        assert _severity(0.7) == "high"

    def test_severity_medium(self):
        assert _severity(0.5) == "medium"

    def test_severity_low(self):
        assert _severity(0.3) == "low"
        assert _severity(0.0) == "low"


# ── TrustEdge tests ──────────────────────────────────────────────────


class TestTrustEdge:
    def test_default_values(self):
        edge = TrustEdge(source="a", target="b")
        assert edge.score == 0.5
        assert edge.interactions == 0
        assert edge.volatility == 0.0

    def test_volatility_with_history(self):
        edge = TrustEdge(source="a", target="b")
        # Need at least 3 interactions for volatility > 0
        for i in range(5):
            outcome = InteractionOutcome.SUCCESS if i % 2 == 0 else InteractionOutcome.BETRAYAL
            edge.history.append(Interaction(
                source="a", target="b", outcome=outcome, timestamp=float(i)
            ))
        assert edge.volatility > 0.0

    def test_volatility_stable(self):
        edge = TrustEdge(source="a", target="b")
        for i in range(5):
            edge.history.append(Interaction(
                source="a", target="b", outcome=InteractionOutcome.NEUTRAL, timestamp=float(i)
            ))
        assert edge.volatility == 0.0


# ── TrustNetwork: Agent management ───────────────────────────────────


class TestAgentManagement:
    def test_add_agent(self, network):
        agent = TrustAgent("a1", role="worker")
        network.add_agent(agent)
        assert "a1" in network.agents
        assert network.agents["a1"].role == "worker"

    def test_add_agent_sets_created_at(self, network):
        network.step_count = 5
        agent = TrustAgent("a1")
        network.add_agent(agent)
        assert network.agents["a1"].created_at == 5

    def test_add_agent_preserves_created_at(self, network):
        agent = TrustAgent("a1", created_at=10.0)
        network.add_agent(agent)
        assert network.agents["a1"].created_at == 10.0

    def test_remove_agent(self, three_agent_network):
        net = three_agent_network
        net.interact("a", "b", outcome="success")
        net.remove_agent("a")
        assert "a" not in net.agents
        assert ("a", "b") not in net.edges

    def test_remove_nonexistent_agent(self, network):
        # Should not raise
        network.remove_agent("nonexistent")


# ── TrustNetwork: Interactions ───────────────────────────────────────


class TestInteractions:
    def test_basic_interaction(self, three_agent_network):
        net = three_agent_network
        edge = net.interact("a", "b", outcome="success")
        assert edge.source == "a"
        assert edge.target == "b"
        assert edge.score > 0.5  # Success increases trust
        assert edge.interactions == 1

    def test_betrayal_decreases_trust(self, three_agent_network):
        net = three_agent_network
        edge = net.interact("a", "b", outcome="betrayal")
        assert edge.score < 0.5

    def test_unknown_agent_raises(self, network):
        network.add_agent(TrustAgent("a"))
        with pytest.raises(ValueError, match="Unknown agent"):
            network.interact("a", "nonexistent", outcome="success")

    def test_invalid_outcome_raises(self, three_agent_network):
        with pytest.raises(ValueError):
            three_agent_network.interact("a", "b", outcome="invalid")

    def test_weighted_interaction(self, three_agent_network):
        net = three_agent_network
        edge_low = net.interact("a", "b", outcome="success", weight=0.1)
        score_low = edge_low.score

        # Reset and try high weight
        net2 = TrustNetwork(seed=42)
        net2.add_agent(TrustAgent("a"))
        net2.add_agent(TrustAgent("b"))
        edge_high = net2.interact("a", "b", outcome="success", weight=2.0)
        assert edge_high.score > score_low

    def test_multiple_interactions_accumulate(self, three_agent_network):
        net = three_agent_network
        for _ in range(10):
            edge = net.interact("a", "b", outcome="cooperation")
        assert edge.score > 0.5
        assert edge.interactions == 10

    def test_trust_clamped_at_bounds(self, three_agent_network):
        net = three_agent_network
        # Drive trust to max
        for _ in range(50):
            edge = net.interact("a", "b", outcome="cooperation")
        assert edge.score <= 1.0

        # Drive trust to min
        for _ in range(100):
            edge = net.interact("a", "b", outcome="betrayal")
        assert edge.score >= 0.0


# ── TrustNetwork: Trust queries ──────────────────────────────────────


class TestTrustQueries:
    def test_get_trust_existing(self, three_agent_network):
        net = three_agent_network
        net.interact("a", "b", outcome="success")
        assert net.get_trust("a", "b") > 0.0

    def test_get_trust_nonexistent(self, network):
        assert network.get_trust("x", "y") == 0.0

    def test_get_reputation(self, three_agent_network):
        net = three_agent_network
        # Everyone trusts b
        net.interact("a", "b", outcome="success")
        net.interact("c", "b", outcome="cooperation")
        rep = net.get_reputation("b")
        assert rep > 0.5

    def test_get_reputation_no_edges(self, three_agent_network):
        assert three_agent_network.get_reputation("a") == 0.0

    def test_get_trust_graph(self, three_agent_network):
        net = three_agent_network
        net.interact("a", "b", outcome="success")
        net.interact("b", "c", outcome="success")
        graph = net.get_trust_graph()
        assert "a" in graph
        assert "b" in graph["a"]
        assert "c" in graph["b"]


# ── TrustNetwork: Step (decay + propagation) ────────────────────────


class TestStep:
    def test_decay_reduces_trust(self, three_agent_network):
        net = three_agent_network
        net.interact("a", "b", outcome="cooperation")
        initial = net.get_trust("a", "b")
        for _ in range(20):
            net.step()
        assert net.get_trust("a", "b") < initial

    def test_propagation_creates_indirect_trust(self, three_agent_network):
        net = three_agent_network
        # Build strong A→B and B→C trust
        for _ in range(10):
            net.interact("a", "b", outcome="cooperation")
            net.interact("b", "c", outcome="cooperation")

        # Step to propagate
        for _ in range(10):
            net.step()

        # A should have some indirect trust in C
        assert net.get_trust("a", "c") > 0.0


# ── TrustNetwork: Threat detection ──────────────────────────────────


class TestThreatDetection:
    def test_detect_sybil(self):
        """New agents that only interact with each other → sybil."""
        net = TrustNetwork(seed=42)
        # Add legitimate agents
        for i in range(5):
            net.add_agent(TrustAgent(f"legit_{i}", role="worker"))
        for i in range(5):
            for j in range(i + 1, 5):
                net.interact(f"legit_{i}", f"legit_{j}", outcome="success")
        # Advance time
        for _ in range(20):
            net.step()
        # Add sybil cluster
        for i in range(5):
            net.add_agent(TrustAgent(f"sybil_{i}", role="worker"))
        # Sybils only interact with each other
        for i in range(5):
            for j in range(i + 1, 5):
                net.interact(f"sybil_{i}", f"sybil_{j}", outcome="cooperation")

        threats = net._detect_sybil()
        sybil_threats = [t for t in threats if t.threat_type == ThreatType.SYBIL]
        assert len(sybil_threats) > 0

    def test_detect_trust_bombing(self):
        net = TrustNetwork(seed=42)
        # Add agents
        for i in range(10):
            net.add_agent(TrustAgent(f"a{i}", role="worker"))
        # One agent gets trusted by everyone very fast
        bomber = "bomber"
        net.add_agent(TrustAgent(bomber, role="worker"))
        for i in range(10):
            for _ in range(5):
                net.interact(f"a{i}", bomber, outcome="cooperation")

        threats = net._detect_trust_bombing()
        bombing = [t for t in threats if t.threat_type == ThreatType.TRUST_BOMBING]
        assert len(bombing) > 0
        assert bomber in bombing[0].agents_involved

    def test_detect_eclipse(self):
        net = TrustNetwork(seed=42)
        net.add_agent(TrustAgent("victim"))
        net.add_agent(TrustAgent("dominant"))
        net.add_agent(TrustAgent("minor"))
        # dominant has very high trust, minor has low
        for _ in range(20):
            net.interact("dominant", "victim", outcome="cooperation")
        net.interact("minor", "victim", outcome="success")

        threats = net._detect_eclipse()
        eclipse = [t for t in threats if t.threat_type == ThreatType.ECLIPSE]
        assert len(eclipse) > 0

    def test_no_threats_clean_network(self, three_agent_network):
        net = three_agent_network
        net.interact("a", "b", outcome="success")
        net.interact("b", "c", outcome="success")
        threats = net.detect_threats()
        # A small balanced network shouldn't trigger alarms
        critical = [t for t in threats if t.severity == "critical"]
        assert len(critical) == 0


# ── TrustNetwork: Community detection ────────────────────────────────


class TestCommunityDetection:
    def test_empty_network(self, network):
        assert network.detect_communities() == []

    def test_two_communities(self):
        net = TrustNetwork(seed=42)
        # Community 1
        for i in range(4):
            net.add_agent(TrustAgent(f"g1_{i}"))
        for i in range(4):
            for j in range(i + 1, 4):
                net.interact(f"g1_{i}", f"g1_{j}", outcome="cooperation")
        # Community 2
        for i in range(4):
            net.add_agent(TrustAgent(f"g2_{i}"))
        for i in range(4):
            for j in range(i + 1, 4):
                net.interact(f"g2_{i}", f"g2_{j}", outcome="cooperation")

        communities = net.detect_communities()
        assert len(communities) >= 2


# ── TrustNetwork: Full analysis ─────────────────────────────────────


class TestAnalyze:
    def test_basic_report(self, three_agent_network):
        net = three_agent_network
        net.interact("a", "b", outcome="success")
        net.interact("b", "c", outcome="success")
        report = net.analyze()

        assert isinstance(report, TrustReport)
        assert report.agent_count == 3
        assert report.edge_count == 2
        assert report.interaction_count == 2
        assert 0.0 <= report.avg_trust <= 1.0
        assert report.network_health in ("healthy", "degraded", "compromised")
        assert 0.0 <= report.health_score <= 100.0

    def test_empty_report(self, network):
        report = network.analyze()
        assert report.agent_count == 0
        assert report.edge_count == 0
        assert report.avg_trust == 0.0

    def test_report_trust_distribution(self, three_agent_network):
        net = three_agent_network
        net.interact("a", "b", outcome="success")
        report = net.analyze()
        assert sum(report.trust_distribution.values()) == 1

    def test_isolated_agents_detected(self):
        net = TrustNetwork(seed=42)
        net.add_agent(TrustAgent("connected_a"))
        net.add_agent(TrustAgent("connected_b"))
        net.add_agent(TrustAgent("isolated"))
        net.interact("connected_a", "connected_b", outcome="success")
        report = net.analyze()
        assert "isolated" in report.isolated_agents


# ── TrustNetwork: Simulation ────────────────────────────────────────


class TestSimulation:
    def test_basic_simulation(self):
        net = TrustNetwork(seed=42)
        report = net.simulate(num_agents=10, num_steps=20)
        assert report.agent_count == 10
        assert report.interaction_count > 0

    def test_simulation_with_attackers(self):
        net = TrustNetwork(seed=42)
        report = net.simulate(num_agents=10, num_steps=30, num_attackers=3)
        assert report.agent_count == 13  # 10 + 3 attackers

    def test_simulation_deterministic(self):
        net1 = TrustNetwork(seed=123)
        r1 = net1.simulate(num_agents=5, num_steps=10)
        net2 = TrustNetwork(seed=123)
        r2 = net2.simulate(num_agents=5, num_steps=10)
        assert r1.avg_trust == r2.avg_trust
        assert r1.agent_count == r2.agent_count


# ── Serialization ────────────────────────────────────────────────────


class TestSerialization:
    def test_to_dict(self, three_agent_network):
        net = three_agent_network
        net.interact("a", "b", outcome="success")
        d = net.to_dict()
        assert "agents" in d
        assert "edges" in d
        assert len(d["agents"]) == 3
        assert len(d["edges"]) == 1
        assert d["edges"][0]["source"] == "a"

    def test_to_dict_json_serializable(self, three_agent_network):
        net = three_agent_network
        net.interact("a", "b", outcome="success")
        # Should not raise
        json.dumps(net.to_dict())


# ── CLI / format ─────────────────────────────────────────────────────


class TestCLI:
    def test_format_report_text(self, three_agent_network):
        net = three_agent_network
        net.interact("a", "b", outcome="success")
        report = net.analyze()
        text = _format_report(report)
        assert "TRUST PROPAGATION ANALYSIS REPORT" in text
        assert "Agents:" in text

    def test_format_report_json(self, three_agent_network):
        net = three_agent_network
        net.interact("a", "b", outcome="success")
        report = net.analyze()
        output = _format_report(report, as_json=True)
        parsed = json.loads(output)
        assert parsed["network"]["agents"] == 3

    def test_main_runs(self, capsys):
        main(["--agents", "5", "--steps", "10", "--seed", "42"])
        captured = capsys.readouterr()
        assert "TRUST PROPAGATION" in captured.out

    def test_main_json_output(self, capsys):
        main(["--agents", "5", "--steps", "10", "--seed", "42", "--json"])
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "network" in parsed

    def test_main_sybil_test(self, capsys):
        main(["--agents", "10", "--steps", "20", "--sybil-test",
              "--attackers", "3", "--seed", "42"])
        captured = capsys.readouterr()
        assert "TRUST PROPAGATION" in captured.out
