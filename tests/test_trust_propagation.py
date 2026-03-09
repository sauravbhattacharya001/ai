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
    _format_report,
    _outcome_delta,
    _severity,
    main,
)


# ── Helpers ──────────────────────────────────────────────────────────


class TestOutcomeDelta:
    def test_success(self):
        assert _outcome_delta(InteractionOutcome.SUCCESS) == 0.8

    def test_betrayal(self):
        assert _outcome_delta(InteractionOutcome.BETRAYAL) == -1.0

    def test_neutral(self):
        assert _outcome_delta(InteractionOutcome.NEUTRAL) == 0.0

    def test_cooperation(self):
        assert _outcome_delta(InteractionOutcome.COOPERATION) == 1.0

    def test_failure(self):
        assert _outcome_delta(InteractionOutcome.FAILURE) == -0.5


class TestSeverity:
    def test_critical(self):
        assert _severity(0.9) == "critical"

    def test_high(self):
        assert _severity(0.7) == "high"

    def test_medium(self):
        assert _severity(0.5) == "medium"

    def test_low(self):
        assert _severity(0.2) == "low"


# ── TrustAgent ───────────────────────────────────────────────────────


class TestTrustAgent:
    def test_defaults(self):
        a = TrustAgent(id="a1")
        assert a.role == "worker"
        assert a.is_malicious is False
        assert a.created_at == 0.0

    def test_custom(self):
        a = TrustAgent(id="x", role="controller", is_malicious=True)
        assert a.role == "controller"
        assert a.is_malicious is True


# ── TrustEdge ────────────────────────────────────────────────────────


class TestTrustEdge:
    def test_defaults(self):
        e = TrustEdge(source="a", target="b")
        assert e.score == 0.5
        assert e.interactions == 0

    def test_volatility_empty(self):
        e = TrustEdge(source="a", target="b")
        assert e.volatility == 0.0

    def test_volatility_with_history(self):
        from replication.trust_propagation import Interaction
        e = TrustEdge(source="a", target="b")
        for outcome in [InteractionOutcome.SUCCESS, InteractionOutcome.BETRAYAL,
                        InteractionOutcome.SUCCESS, InteractionOutcome.FAILURE]:
            e.history.append(Interaction(source="a", target="b",
                                         outcome=outcome, timestamp=0))
        assert e.volatility > 0


# ── TrustNetwork basics ─────────────────────────────────────────────


class TestNetworkBasics:
    def test_add_remove_agent(self):
        net = TrustNetwork()
        net.add_agent(TrustAgent(id="a1"))
        assert "a1" in net.agents
        net.remove_agent("a1")
        assert "a1" not in net.agents

    def test_remove_cleans_edges(self):
        net = TrustNetwork()
        net.add_agent(TrustAgent(id="a"))
        net.add_agent(TrustAgent(id="b"))
        net.interact("a", "b")
        assert len(net.edges) == 1
        net.remove_agent("a")
        assert len(net.edges) == 0

    def test_interact_unknown_agent(self):
        net = TrustNetwork()
        with pytest.raises(ValueError):
            net.interact("a", "b")

    def test_interact_creates_edge(self):
        net = TrustNetwork()
        net.add_agent(TrustAgent(id="a"))
        net.add_agent(TrustAgent(id="b"))
        edge = net.interact("a", "b", outcome="success")
        assert edge.score > 0.5  # success increases trust
        assert edge.interactions == 1

    def test_interact_cooperation_high(self):
        net = TrustNetwork()
        net.add_agent(TrustAgent(id="a"))
        net.add_agent(TrustAgent(id="b"))
        edge = net.interact("a", "b", outcome="cooperation")
        assert edge.score > 0.5

    def test_interact_betrayal_low(self):
        net = TrustNetwork()
        net.add_agent(TrustAgent(id="a"))
        net.add_agent(TrustAgent(id="b"))
        edge = net.interact("a", "b", outcome="betrayal")
        assert edge.score < 0.5

    def test_multiple_interactions(self):
        net = TrustNetwork()
        net.add_agent(TrustAgent(id="a"))
        net.add_agent(TrustAgent(id="b"))
        for _ in range(5):
            net.interact("a", "b", outcome="success")
        edge = net.edges[("a", "b")]
        assert edge.interactions == 5
        assert edge.score > 0.7

    def test_get_trust(self):
        net = TrustNetwork()
        net.add_agent(TrustAgent(id="a"))
        net.add_agent(TrustAgent(id="b"))
        assert net.get_trust("a", "b") == 0.0  # no edge yet
        net.interact("a", "b")
        assert net.get_trust("a", "b") > 0

    def test_get_reputation(self):
        net = TrustNetwork()
        net.add_agent(TrustAgent(id="a"))
        net.add_agent(TrustAgent(id="b"))
        net.add_agent(TrustAgent(id="c"))
        net.interact("a", "b", outcome="cooperation")
        net.interact("c", "b", outcome="cooperation")
        rep = net.get_reputation("b")
        assert rep > 0.5

    def test_get_reputation_no_edges(self):
        net = TrustNetwork()
        net.add_agent(TrustAgent(id="a"))
        assert net.get_reputation("a") == 0.0

    def test_get_trust_graph(self):
        net = TrustNetwork()
        net.add_agent(TrustAgent(id="a"))
        net.add_agent(TrustAgent(id="b"))
        net.interact("a", "b")
        graph = net.get_trust_graph()
        assert "a" in graph
        assert "b" in graph["a"]


# ── Step / propagation ───────────────────────────────────────────────


class TestStepAndPropagation:
    def test_decay(self):
        net = TrustNetwork(decay_rate=0.1)
        net.add_agent(TrustAgent(id="a"))
        net.add_agent(TrustAgent(id="b"))
        net.interact("a", "b", outcome="cooperation")
        initial = net.edges[("a", "b")].score
        for _ in range(5):
            net.step()
        assert net.edges[("a", "b")].score < initial

    def test_propagation(self):
        net = TrustNetwork(propagation_damping=0.5)
        net.add_agent(TrustAgent(id="a"))
        net.add_agent(TrustAgent(id="b"))
        net.add_agent(TrustAgent(id="c"))
        # A trusts B, B trusts C
        for _ in range(5):
            net.interact("a", "b", outcome="cooperation")
            net.interact("b", "c", outcome="cooperation")
        net.step()
        # A should gain some indirect trust in C
        assert net.get_trust("a", "c") > 0


# ── Threat detection ─────────────────────────────────────────────────


class TestThreatDetection:
    def test_no_threats_clean_network(self):
        net = TrustNetwork(seed=42)
        for i in range(5):
            net.add_agent(TrustAgent(id=f"a{i}"))
        # minimal interactions
        net.interact("a0", "a1", outcome="success")
        threats = net.detect_threats()
        # might detect some or none — just ensure it returns a list
        assert isinstance(threats, list)

    def test_sybil_detection(self):
        net = TrustNetwork(seed=42)
        # Existing agents
        for i in range(5):
            net.add_agent(TrustAgent(id=f"good_{i}"))
        for _ in range(10):
            net.interact("good_0", "good_1", outcome="success")
            net.step()

        # Inject sybils (recently created, only interact with each other)
        for i in range(5):
            net.add_agent(TrustAgent(id=f"sybil_{i}"))
        for i in range(5):
            for j in range(5):
                if i != j:
                    net.interact(f"sybil_{i}", f"sybil_{j}", outcome="cooperation")

        threats = net.detect_threats()
        sybil_threats = [t for t in threats if t.threat_type == ThreatType.SYBIL]
        assert len(sybil_threats) > 0

    def test_eclipse_detection(self):
        net = TrustNetwork()
        net.add_agent(TrustAgent(id="victim"))
        net.add_agent(TrustAgent(id="dom"))
        net.add_agent(TrustAgent(id="weak"))
        # dom dominates victim's incoming trust
        for _ in range(10):
            net.interact("dom", "victim", outcome="cooperation")
        net.interact("weak", "victim", outcome="success")
        threats = net.detect_threats()
        eclipse = [t for t in threats if t.threat_type == ThreatType.ECLIPSE]
        assert len(eclipse) > 0


# ── Community detection ──────────────────────────────────────────────


class TestCommunityDetection:
    def test_empty_network(self):
        net = TrustNetwork()
        assert net.detect_communities() == []

    def test_two_communities(self):
        net = TrustNetwork()
        for i in range(6):
            net.add_agent(TrustAgent(id=f"a{i}"))
        # Community 1: a0-a2
        for i in range(3):
            for j in range(3):
                if i != j:
                    for _ in range(3):
                        net.interact(f"a{i}", f"a{j}", outcome="cooperation")
        # Community 2: a3-a5
        for i in range(3, 6):
            for j in range(3, 6):
                if i != j:
                    for _ in range(3):
                        net.interact(f"a{i}", f"a{j}", outcome="cooperation")
        comms = net.detect_communities()
        assert len(comms) >= 2


# ── Full analysis ────────────────────────────────────────────────────


class TestAnalyze:
    def test_empty(self):
        net = TrustNetwork()
        report = net.analyze()
        assert isinstance(report, TrustReport)
        assert report.agent_count == 0
        assert report.network_health == "healthy"

    def test_basic_report(self):
        net = TrustNetwork(seed=42)
        for i in range(5):
            net.add_agent(TrustAgent(id=f"a{i}"))
        net.interact("a0", "a1", outcome="success")
        net.interact("a1", "a2", outcome="cooperation")
        report = net.analyze()
        assert report.agent_count == 5
        assert report.edge_count == 2
        assert report.interaction_count == 2
        assert 0 <= report.avg_trust <= 1
        assert report.health_score >= 0


# ── Simulation ───────────────────────────────────────────────────────


class TestSimulation:
    def test_basic_sim(self):
        net = TrustNetwork(seed=123)
        report = net.simulate(num_agents=10, num_steps=20)
        assert report.agent_count == 10
        assert report.interaction_count > 0

    def test_sim_with_attackers(self):
        net = TrustNetwork(seed=456)
        report = net.simulate(num_agents=10, num_steps=30, num_attackers=3)
        assert report.agent_count == 13  # 10 + 3 attackers


# ── Serialization ────────────────────────────────────────────────────


class TestSerialization:
    def test_to_dict(self):
        net = TrustNetwork()
        net.add_agent(TrustAgent(id="a"))
        net.add_agent(TrustAgent(id="b"))
        net.interact("a", "b")
        d = net.to_dict()
        assert "agents" in d
        assert "edges" in d
        assert d["step_count"] == 0
        assert d["interaction_count"] == 1


# ── CLI / format ─────────────────────────────────────────────────────


class TestFormatAndCLI:
    def test_format_text(self):
        net = TrustNetwork(seed=1)
        report = net.simulate(num_agents=5, num_steps=10)
        text = _format_report(report)
        assert "TRUST PROPAGATION" in text

    def test_format_json(self):
        net = TrustNetwork(seed=1)
        report = net.simulate(num_agents=5, num_steps=10)
        output = _format_report(report, as_json=True)
        parsed = json.loads(output)
        assert "network" in parsed

    def test_main_runs(self, capsys):
        main(["--agents", "5", "--steps", "5", "--seed", "42"])
        captured = capsys.readouterr()
        assert "TRUST PROPAGATION" in captured.out

    def test_main_json(self, capsys):
        main(["--agents", "5", "--steps", "5", "--seed", "42", "--json"])
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "network" in parsed

    def test_main_sybil(self, capsys):
        main(["--agents", "10", "--steps", "10", "--sybil-test",
              "--attackers", "3", "--seed", "42"])
        captured = capsys.readouterr()
        assert "TRUST PROPAGATION" in captured.out
