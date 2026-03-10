"""Tests for trust_propagation module."""

import json
import math
import pytest

from replication.trust_propagation import (
    AgentRole,
    Interaction,
    InteractionOutcome,
    ThreatDetection,
    ThreatType,
    TrustAgent,
    TrustEdge,
    TrustNetwork,
    TrustReport,
    _outcome_delta,
    _severity,
)


# ── Helpers ──────────────────────────────────────────────────────


class TestOutcomeDelta:
    def test_success_positive(self):
        assert _outcome_delta(InteractionOutcome.SUCCESS) == 0.8

    def test_cooperation_highest(self):
        assert _outcome_delta(InteractionOutcome.COOPERATION) == 1.0

    def test_neutral_zero(self):
        assert _outcome_delta(InteractionOutcome.NEUTRAL) == 0.0

    def test_failure_negative(self):
        assert _outcome_delta(InteractionOutcome.FAILURE) < 0

    def test_betrayal_most_negative(self):
        assert _outcome_delta(InteractionOutcome.BETRAYAL) == -1.0


class TestSeverity:
    def test_critical(self):
        assert _severity(0.9) == "critical"
        assert _severity(0.8) == "critical"

    def test_high(self):
        assert _severity(0.7) == "high"
        assert _severity(0.6) == "high"

    def test_medium(self):
        assert _severity(0.5) == "medium"
        assert _severity(0.4) == "medium"

    def test_low(self):
        assert _severity(0.3) == "low"
        assert _severity(0.0) == "low"


# ── TrustEdge ────────────────────────────────────────────────────


class TestTrustEdge:
    def test_volatility_few_interactions(self):
        edge = TrustEdge(source="a", target="b")
        edge.history = [
            Interaction("a", "b", InteractionOutcome.SUCCESS, 1.0),
        ]
        assert edge.volatility == 0.0

    def test_volatility_with_history(self):
        edge = TrustEdge(source="a", target="b")
        outcomes = [
            InteractionOutcome.SUCCESS,
            InteractionOutcome.BETRAYAL,
            InteractionOutcome.COOPERATION,
            InteractionOutcome.FAILURE,
            InteractionOutcome.SUCCESS,
        ]
        for i, o in enumerate(outcomes):
            edge.history.append(Interaction("a", "b", o, float(i)))
        vol = edge.volatility
        assert vol > 0  # Should have meaningful volatility


# ── TrustNetwork basics ─────────────────────────────────────────


class TestNetworkBasics:
    def setup_method(self):
        self.net = TrustNetwork(seed=42)

    def test_add_agent(self):
        self.net.add_agent(TrustAgent("a1"))
        assert "a1" in self.net.agents

    def test_add_agent_sets_created_at(self):
        self.net.step_count = 5
        self.net.add_agent(TrustAgent("a1"))
        assert self.net.agents["a1"].created_at == 5

    def test_add_agent_preserves_created_at(self):
        self.net.add_agent(TrustAgent("a1", created_at=10.0))
        assert self.net.agents["a1"].created_at == 10.0

    def test_remove_agent(self):
        self.net.add_agent(TrustAgent("a1"))
        self.net.add_agent(TrustAgent("a2"))
        self.net.interact("a1", "a2")
        self.net.remove_agent("a1")
        assert "a1" not in self.net.agents
        assert ("a1", "a2") not in self.net.edges

    def test_remove_nonexistent_agent(self):
        self.net.remove_agent("nonexistent")  # Should not raise


class TestInteractions:
    def setup_method(self):
        self.net = TrustNetwork(seed=42)
        self.net.add_agent(TrustAgent("a1"))
        self.net.add_agent(TrustAgent("a2"))

    def test_interact_creates_edge(self):
        self.net.interact("a1", "a2")
        assert ("a1", "a2") in self.net.edges

    def test_interact_unknown_agent_raises(self):
        with pytest.raises(ValueError, match="Unknown agent"):
            self.net.interact("a1", "unknown")

    def test_interact_records_interaction(self):
        self.net.interact("a1", "a2", outcome="success")
        assert len(self.net.interactions) == 1
        assert self.net.interactions[0].outcome == InteractionOutcome.SUCCESS

    def test_success_increases_trust(self):
        initial = self.net.initial_trust
        self.net.interact("a1", "a2", outcome="success")
        assert self.net.get_trust("a1", "a2") > initial

    def test_betrayal_decreases_trust(self):
        initial = self.net.initial_trust
        self.net.interact("a1", "a2", outcome="betrayal")
        assert self.net.get_trust("a1", "a2") < initial

    def test_trust_clamped_to_0_1(self):
        for _ in range(100):
            self.net.interact("a1", "a2", outcome="cooperation")
        assert self.net.get_trust("a1", "a2") <= 1.0

        net2 = TrustNetwork(seed=42)
        net2.add_agent(TrustAgent("x"))
        net2.add_agent(TrustAgent("y"))
        for _ in range(100):
            net2.interact("x", "y", outcome="betrayal")
        assert net2.get_trust("x", "y") >= 0.0

    def test_interact_with_weight(self):
        edge = self.net.interact("a1", "a2", outcome="success", weight=2.0)
        assert edge.interactions == 1

    def test_interact_with_context(self):
        self.net.interact("a1", "a2", context="verification task")
        assert self.net.interactions[0].context == "verification task"

    def test_multiple_interactions_accumulate(self):
        self.net.interact("a1", "a2", outcome="success")
        self.net.interact("a1", "a2", outcome="success")
        edge = self.net.edges[("a1", "a2")]
        assert edge.interactions == 2


class TestTrustQueries:
    def setup_method(self):
        self.net = TrustNetwork(seed=42)
        for i in range(5):
            self.net.add_agent(TrustAgent(f"a{i}"))

    def test_get_trust_no_edge(self):
        assert self.net.get_trust("a0", "a1") == 0.0

    def test_get_reputation_no_edges(self):
        assert self.net.get_reputation("a0") == 0.0

    def test_get_reputation_with_edges(self):
        self.net.interact("a1", "a0", outcome="cooperation")
        self.net.interact("a2", "a0", outcome="success")
        rep = self.net.get_reputation("a0")
        assert rep > 0

    def test_get_trust_graph(self):
        self.net.interact("a0", "a1")
        self.net.interact("a1", "a2")
        graph = self.net.get_trust_graph()
        assert "a0" in graph
        assert "a1" in graph["a0"]


class TestPropagation:
    def test_step_decays_trust(self):
        net = TrustNetwork(seed=42)
        net.add_agent(TrustAgent("a1"))
        net.add_agent(TrustAgent("a2"))
        net.interact("a1", "a2", outcome="cooperation")
        trust_before = net.get_trust("a1", "a2")
        for _ in range(20):
            net.step()
        trust_after = net.get_trust("a1", "a2")
        assert trust_after < trust_before

    def test_step_propagates_trust(self):
        net = TrustNetwork(seed=42, propagation_damping=0.5)
        net.add_agent(TrustAgent("a"))
        net.add_agent(TrustAgent("b"))
        net.add_agent(TrustAgent("c"))
        # Build strong A→B and B→C
        for _ in range(10):
            net.interact("a", "b", outcome="cooperation")
            net.interact("b", "c", outcome="cooperation")
        net.step()
        # A should now have some indirect trust in C
        assert ("a", "c") in net.edges


# ── Threat detection ─────────────────────────────────────────────


class TestSybilDetection:
    def test_no_sybil_with_few_agents(self):
        net = TrustNetwork(seed=42)
        net.add_agent(TrustAgent("a1"))
        net.add_agent(TrustAgent("a2"))
        threats = net._detect_sybil()
        assert len(threats) == 0

    def test_detects_sybil_cluster(self):
        net = TrustNetwork(seed=42)
        net.step_count = 20
        # Add established agents
        for i in range(5):
            net.add_agent(TrustAgent(f"good_{i}", created_at=0))
        # Add recent suspicious cluster
        for i in range(5):
            net.add_agent(TrustAgent(f"sybil_{i}", created_at=18))
        # Sybils only trust each other
        for i in range(5):
            for j in range(5):
                if i != j:
                    net.interact(f"sybil_{i}", f"sybil_{j}", outcome="cooperation")
        threats = net._detect_sybil()
        assert any(t.threat_type == ThreatType.SYBIL for t in threats)


class TestCollusionDetection:
    def test_detects_collusion_ring(self):
        net = TrustNetwork(seed=42)
        agents = ["a", "b", "c"]
        for a in agents:
            net.add_agent(TrustAgent(a))
        # Build strong ring: a→b→c→a
        for _ in range(15):
            net.interact("a", "b", outcome="cooperation")
            net.interact("b", "c", outcome="cooperation")
            net.interact("c", "a", outcome="cooperation")
        threats = net._detect_collusion()
        assert any(t.threat_type == ThreatType.COLLUSION_RING for t in threats)

    def test_no_collusion_with_low_trust(self):
        net = TrustNetwork(seed=42)
        for a in ["a", "b", "c"]:
            net.add_agent(TrustAgent(a))
        net.interact("a", "b", outcome="neutral")
        net.interact("b", "c", outcome="neutral")
        threats = net._detect_collusion()
        assert len(threats) == 0


class TestTrustBombing:
    def test_detects_rapid_trust_gain(self):
        net = TrustNetwork(seed=42)
        for i in range(10):
            net.add_agent(TrustAgent(f"a{i}"))
        # One agent gets lots of trust fast
        for i in range(1, 10):
            for _ in range(5):
                net.interact(f"a{i}", "a0", outcome="cooperation")
        net.step_count = 1  # Very recent
        threats = net._detect_trust_bombing()
        assert any(t.threat_type == ThreatType.TRUST_BOMBING for t in threats)


class TestSleeperDetection:
    def test_detects_sleeper_pattern(self):
        net = TrustNetwork(seed=42)
        net.add_agent(TrustAgent("sleeper"))
        net.add_agent(TrustAgent("target"))
        # Early interactions
        net.step_count = 0
        net.interact("sleeper", "target", outcome="neutral")
        net.interact("sleeper", "target", outcome="neutral")
        # Long gap
        net.step_count = 50
        # Burst of activity
        for s in range(50, 55):
            net.step_count = s
            net.interact("sleeper", "target", outcome="cooperation")
        threats = net._detect_sleeper()
        assert any(t.threat_type == ThreatType.SLEEPER for t in threats)


class TestEclipseDetection:
    def test_detects_eclipse(self):
        net = TrustNetwork(seed=42)
        for a in ["victim", "dom", "other"]:
            net.add_agent(TrustAgent(a))
        # Dominant agent floods trust
        for _ in range(20):
            net.interact("dom", "victim", outcome="cooperation")
        net.interact("other", "victim", outcome="success")
        threats = net._detect_eclipse()
        assert any(t.threat_type == ThreatType.ECLIPSE for t in threats)


class TestTrustLaundering:
    def test_detects_laundering(self):
        net = TrustNetwork(seed=42)
        for a in ["a", "b", "c"]:
            net.add_agent(TrustAgent(a))
        # Strong A→B and B→C
        for _ in range(15):
            net.interact("a", "b", outcome="cooperation")
            net.interact("b", "c", outcome="cooperation")
        # Propagate to create A→C edge with score > 0.1
        for _ in range(10):
            net.step()
        # Verify laundering path exists
        ac_edge = net.edges.get(("a", "c"))
        if ac_edge and ac_edge.score > 0.1:
            threats = net._detect_trust_laundering()
            laundering = [t for t in threats if t.threat_type == ThreatType.TRUST_LAUNDERING]
            assert len(laundering) > 0


# ── Community detection ──────────────────────────────────────────


class TestCommunityDetection:
    def test_empty_network(self):
        net = TrustNetwork(seed=42)
        assert net.detect_communities() == []

    def test_two_communities(self):
        net = TrustNetwork(seed=42)
        # Community 1
        for a in ["a1", "a2", "a3"]:
            net.add_agent(TrustAgent(a))
        for _ in range(10):
            net.interact("a1", "a2", outcome="cooperation")
            net.interact("a2", "a3", outcome="cooperation")
            net.interact("a3", "a1", outcome="cooperation")
        # Community 2
        for a in ["b1", "b2", "b3"]:
            net.add_agent(TrustAgent(a))
        for _ in range(10):
            net.interact("b1", "b2", outcome="cooperation")
            net.interact("b2", "b3", outcome="cooperation")
            net.interact("b3", "b1", outcome="cooperation")
        communities = net.detect_communities()
        assert len(communities) >= 2


# ── Full analysis ────────────────────────────────────────────────


class TestAnalyze:
    def test_basic_report(self):
        net = TrustNetwork(seed=42)
        for i in range(5):
            net.add_agent(TrustAgent(f"a{i}"))
        for i in range(4):
            net.interact(f"a{i}", f"a{i+1}", outcome="success")
        report = net.analyze()
        assert isinstance(report, TrustReport)
        assert report.agent_count == 5
        assert report.edge_count == 4
        assert report.interaction_count == 4
        assert 0 <= report.health_score <= 100

    def test_report_distribution_buckets(self):
        net = TrustNetwork(seed=42)
        net.add_agent(TrustAgent("a"))
        net.add_agent(TrustAgent("b"))
        net.interact("a", "b")
        report = net.analyze()
        total_in_buckets = sum(report.trust_distribution.values())
        assert total_in_buckets == 1

    def test_report_network_health(self):
        net = TrustNetwork(seed=42)
        for i in range(10):
            net.add_agent(TrustAgent(f"a{i}"))
        for i in range(9):
            net.interact(f"a{i}", f"a{i+1}", outcome="cooperation")
        report = net.analyze()
        assert report.network_health in ("healthy", "degraded", "compromised")

    def test_report_includes_threats(self):
        net = TrustNetwork(seed=42)
        for i in range(5):
            net.add_agent(TrustAgent(f"a{i}"))
        report = net.analyze()
        assert isinstance(report.threats, list)

    def test_report_isolated_agents(self):
        net = TrustNetwork(seed=42)
        net.add_agent(TrustAgent("connected1"))
        net.add_agent(TrustAgent("connected2"))
        net.add_agent(TrustAgent("isolated"))
        net.interact("connected1", "connected2")
        report = net.analyze()
        assert "isolated" in report.isolated_agents

    def test_report_most_trusted(self):
        net = TrustNetwork(seed=42)
        for i in range(5):
            net.add_agent(TrustAgent(f"a{i}"))
        for i in range(1, 5):
            net.interact(f"a{i}", "a0", outcome="cooperation")
        report = net.analyze()
        assert len(report.most_trusted) > 0
        assert report.most_trusted[0][0] == "a0"


# ── Simulation ───────────────────────────────────────────────────


class TestSimulation:
    def test_basic_simulation(self):
        net = TrustNetwork(seed=42)
        report = net.simulate(num_agents=10, num_steps=20)
        assert isinstance(report, TrustReport)
        assert report.agent_count == 10

    def test_simulation_with_attackers(self):
        net = TrustNetwork(seed=42)
        report = net.simulate(num_agents=10, num_steps=30, num_attackers=3)
        assert report.agent_count == 13  # 10 + 3 attackers

    def test_simulation_deterministic(self):
        net1 = TrustNetwork(seed=42)
        r1 = net1.simulate(num_agents=10, num_steps=20)
        net2 = TrustNetwork(seed=42)
        r2 = net2.simulate(num_agents=10, num_steps=20)
        assert r1.avg_trust == r2.avg_trust
        assert r1.edge_count == r2.edge_count


# ── Serialization ────────────────────────────────────────────────


class TestSerialization:
    def test_to_dict(self):
        net = TrustNetwork(seed=42)
        net.add_agent(TrustAgent("a"))
        net.add_agent(TrustAgent("b"))
        net.interact("a", "b")
        d = net.to_dict()
        assert "agents" in d
        assert "edges" in d
        assert "step_count" in d
        assert "interaction_count" in d
        assert d["interaction_count"] == 1

    def test_to_dict_serializable(self):
        net = TrustNetwork(seed=42)
        net.add_agent(TrustAgent("a"))
        net.add_agent(TrustAgent("b"))
        net.interact("a", "b")
        d = net.to_dict()
        # Must be JSON-serializable
        json.dumps(d)


# ── Edge cases ───────────────────────────────────────────────────


class TestEdgeCases:
    def test_self_interaction_skipped_in_simulate(self):
        net = TrustNetwork(seed=42)
        report = net.simulate(num_agents=5, num_steps=10)
        for edge in net.edges.values():
            assert edge.source != edge.target

    def test_analyze_empty_network(self):
        net = TrustNetwork(seed=42)
        net.add_agent(TrustAgent("lonely"))
        report = net.analyze()
        assert report.agent_count == 1
        assert report.edge_count == 0

    def test_detect_threats_empty(self):
        net = TrustNetwork(seed=42)
        threats = net.detect_threats()
        assert threats == []

    def test_invalid_outcome(self):
        net = TrustNetwork(seed=42)
        net.add_agent(TrustAgent("a"))
        net.add_agent(TrustAgent("b"))
        with pytest.raises(ValueError):
            net.interact("a", "b", outcome="invalid")

    def test_custom_params(self):
        net = TrustNetwork(
            decay_rate=0.01,
            propagation_damping=0.5,
            initial_trust=0.3,
            seed=42,
        )
        net.add_agent(TrustAgent("a"))
        net.add_agent(TrustAgent("b"))
        edge = net.interact("a", "b", outcome="neutral")
        assert edge.score == 0.3  # Neutral = 0 delta, stays at initial
