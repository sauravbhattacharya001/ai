"""Tests for trust_propagation module."""

from __future__ import annotations

import json
import pytest
from replication.trust_propagation import (
    TrustNetwork, TrustAgent, TrustEdge, Interaction, InteractionOutcome,
    ThreatType, ThreatDetection, TrustReport, _outcome_delta, _severity,
    _format_report, main,
)


# ── Basic network operations ────────────────────────────────────────

class TestTrustAgent:
    def test_create_agent(self):
        a = TrustAgent(id="a1", role="worker")
        assert a.id == "a1"
        assert a.role == "worker"
        assert not a.is_malicious

    def test_malicious_agent(self):
        a = TrustAgent(id="bad", role="worker", is_malicious=True)
        assert a.is_malicious


class TestTrustNetwork:
    def test_add_remove_agent(self):
        net = TrustNetwork()
        net.add_agent(TrustAgent(id="a1"))
        assert "a1" in net.agents
        net.remove_agent("a1")
        assert "a1" not in net.agents

    def test_remove_cleans_edges(self):
        net = TrustNetwork()
        net.add_agent(TrustAgent(id="a1"))
        net.add_agent(TrustAgent(id="a2"))
        net.interact("a1", "a2")
        assert len(net.edges) == 1
        net.remove_agent("a1")
        assert len(net.edges) == 0

    def test_interact_creates_edge(self):
        net = TrustNetwork()
        net.add_agent(TrustAgent(id="a1"))
        net.add_agent(TrustAgent(id="a2"))
        edge = net.interact("a1", "a2", outcome="success")
        assert edge.source == "a1"
        assert edge.target == "a2"
        assert edge.score > 0.5  # success increases trust

    def test_interact_unknown_agent_raises(self):
        net = TrustNetwork()
        with pytest.raises(ValueError):
            net.interact("a1", "a2")

    def test_betrayal_lowers_trust(self):
        net = TrustNetwork()
        net.add_agent(TrustAgent(id="a1"))
        net.add_agent(TrustAgent(id="a2"))
        edge = net.interact("a1", "a2", outcome="betrayal")
        assert edge.score < 0.5

    def test_repeated_success_builds_trust(self):
        net = TrustNetwork()
        net.add_agent(TrustAgent(id="a1"))
        net.add_agent(TrustAgent(id="a2"))
        for _ in range(10):
            net.interact("a1", "a2", outcome="cooperation")
        assert net.get_trust("a1", "a2") > 0.8

    def test_repeated_failure_destroys_trust(self):
        net = TrustNetwork()
        net.add_agent(TrustAgent(id="a1"))
        net.add_agent(TrustAgent(id="a2"))
        for _ in range(10):
            net.interact("a1", "a2", outcome="betrayal")
        assert net.get_trust("a1", "a2") == 0.0

    def test_neutral_no_change(self):
        net = TrustNetwork(initial_trust=0.5)
        net.add_agent(TrustAgent(id="a1"))
        net.add_agent(TrustAgent(id="a2"))
        net.interact("a1", "a2", outcome="neutral")
        assert abs(net.get_trust("a1", "a2") - 0.5) < 0.01

    def test_get_trust_unknown(self):
        net = TrustNetwork()
        assert net.get_trust("x", "y") == 0.0


class TestReputation:
    def test_reputation_with_no_edges(self):
        net = TrustNetwork()
        net.add_agent(TrustAgent(id="a1"))
        assert net.get_reputation("a1") == 0.0

    def test_reputation_reflects_incoming_trust(self):
        net = TrustNetwork()
        for i in range(5):
            net.add_agent(TrustAgent(id=f"a{i}"))
        for i in range(1, 5):
            for _ in range(5):
                net.interact(f"a{i}", "a0", outcome="cooperation")
        rep = net.get_reputation("a0")
        assert rep > 0.7


class TestTrustDecay:
    def test_trust_decays_over_steps(self):
        net = TrustNetwork(decay_rate=0.05)
        net.add_agent(TrustAgent(id="a1"))
        net.add_agent(TrustAgent(id="a2"))
        for _ in range(5):
            net.interact("a1", "a2", outcome="cooperation")
        initial = net.get_trust("a1", "a2")
        for _ in range(20):
            net.step()
        assert net.get_trust("a1", "a2") < initial


class TestPropagation:
    def test_indirect_trust_propagates(self):
        net = TrustNetwork(propagation_damping=0.5, seed=42)
        net.add_agent(TrustAgent(id="a"))
        net.add_agent(TrustAgent(id="b"))
        net.add_agent(TrustAgent(id="c"))
        for _ in range(10):
            net.interact("a", "b", outcome="cooperation")
            net.interact("b", "c", outcome="cooperation")
        net.step()
        # a should now have some indirect trust in c
        assert net.get_trust("a", "c") > 0.0


class TestTrustGraph:
    def test_trust_graph_structure(self):
        net = TrustNetwork()
        net.add_agent(TrustAgent(id="a1"))
        net.add_agent(TrustAgent(id="a2"))
        net.interact("a1", "a2")
        graph = net.get_trust_graph()
        assert "a1" in graph
        assert "a2" in graph["a1"]


# ── Threat detection ─────────────────────────────────────────────────

class TestSybilDetection:
    def test_sybil_cluster_detected(self):
        net = TrustNetwork(seed=42)
        # Legitimate agents
        for i in range(10):
            net.add_agent(TrustAgent(id=f"legit_{i}", created_at=0))
        for i in range(10):
            for j in range(i + 1, 10):
                net.interact(f"legit_{i}", f"legit_{j}", outcome="success")
        # Sybil cluster: new agents only interacting with each other
        net.step_count = 50
        for i in range(5):
            net.add_agent(TrustAgent(id=f"sybil_{i}"))
        for i in range(5):
            for j in range(5):
                if i != j:
                    net.interact(f"sybil_{i}", f"sybil_{j}", outcome="cooperation")
        threats = net._detect_sybil()
        assert any(t.threat_type == ThreatType.SYBIL for t in threats)


class TestCollusionDetection:
    def test_collusion_ring(self):
        net = TrustNetwork()
        for i in range(4):
            net.add_agent(TrustAgent(id=f"r{i}"))
        # Create a high-trust ring: r0→r1→r2→r3→r0
        for _ in range(10):
            net.interact("r0", "r1", outcome="cooperation")
            net.interact("r1", "r2", outcome="cooperation")
            net.interact("r2", "r3", outcome="cooperation")
            net.interact("r3", "r0", outcome="cooperation")
        threats = net._detect_collusion()
        assert any(t.threat_type == ThreatType.COLLUSION_RING for t in threats)


class TestTrustBombing:
    def test_trust_bombing_detected(self):
        net = TrustNetwork()
        net.add_agent(TrustAgent(id="bomber", created_at=0))
        for i in range(15):
            net.add_agent(TrustAgent(id=f"v{i}", created_at=0))
        net.step_count = 1  # Very young
        for i in range(15):
            for _ in range(5):
                net.interact(f"v{i}", "bomber", outcome="cooperation")
        threats = net._detect_trust_bombing()
        assert any(t.threat_type == ThreatType.TRUST_BOMBING for t in threats)


class TestSleeperDetection:
    def test_sleeper_activation(self):
        net = TrustNetwork()
        net.add_agent(TrustAgent(id="sleeper"))
        net.add_agent(TrustAgent(id="target"))
        # Early activity
        net.step_count = 1
        net.interact("sleeper", "target", outcome="success")
        net.step_count = 2
        net.interact("sleeper", "target", outcome="success")
        # Long dormancy
        # Burst
        net.step_count = 50
        for _ in range(5):
            net.interact("sleeper", "target", outcome="betrayal")
        net.step_count = 51
        net.interact("sleeper", "target", outcome="betrayal")
        net.step_count = 52
        net.interact("sleeper", "target", outcome="betrayal")
        threats = net._detect_sleeper()
        assert any(t.threat_type == ThreatType.SLEEPER for t in threats)


class TestEclipseDetection:
    def test_eclipse_detected(self):
        net = TrustNetwork()
        net.add_agent(TrustAgent(id="victim"))
        net.add_agent(TrustAgent(id="dominator"))
        net.add_agent(TrustAgent(id="minor"))
        net.add_agent(TrustAgent(id="minor2"))
        for _ in range(10):
            net.interact("dominator", "victim", outcome="cooperation")
        # Minor sources with failures to keep their trust low
        net.interact("minor", "victim", outcome="failure")
        net.interact("minor", "victim", outcome="failure")
        net.interact("minor", "victim", outcome="failure")
        net.interact("minor2", "victim", outcome="failure")
        net.interact("minor2", "victim", outcome="failure")
        net.interact("minor2", "victim", outcome="failure")
        threats = net._detect_eclipse()
        assert any(t.threat_type == ThreatType.ECLIPSE for t in threats)


# ── Community detection ──────────────────────────────────────────────

class TestCommunityDetection:
    def test_two_communities(self):
        net = TrustNetwork()
        # Group 1
        for i in range(5):
            net.add_agent(TrustAgent(id=f"g1_{i}"))
        for i in range(5):
            for j in range(i + 1, 5):
                net.interact(f"g1_{i}", f"g1_{j}", outcome="cooperation")
        # Group 2
        for i in range(5):
            net.add_agent(TrustAgent(id=f"g2_{i}"))
        for i in range(5):
            for j in range(i + 1, 5):
                net.interact(f"g2_{i}", f"g2_{j}", outcome="cooperation")
        communities = net.detect_communities()
        assert len(communities) >= 2

    def test_empty_network(self):
        net = TrustNetwork()
        assert net.detect_communities() == []


# ── Analysis report ──────────────────────────────────────────────────

class TestAnalysis:
    def test_analyze_returns_report(self):
        net = TrustNetwork(seed=42)
        for i in range(5):
            net.add_agent(TrustAgent(id=f"a{i}"))
        for i in range(5):
            for j in range(5):
                if i != j:
                    net.interact(f"a{i}", f"a{j}", outcome="success")
        report = net.analyze()
        assert isinstance(report, TrustReport)
        assert report.agent_count == 5
        assert report.edge_count > 0
        assert 0 <= report.health_score <= 100

    def test_health_score_degrades_with_threats(self):
        net = TrustNetwork(seed=42)
        report = net.simulate(num_agents=15, num_steps=30, num_attackers=5)
        # With attackers, health should be lower
        assert report.health_score < 100


class TestSimulation:
    def test_basic_simulation(self):
        net = TrustNetwork(seed=42)
        report = net.simulate(num_agents=10, num_steps=20)
        assert report.agent_count == 10
        assert report.interaction_count > 0
        assert report.network_health in ("healthy", "degraded", "compromised")

    def test_sybil_simulation(self):
        net = TrustNetwork(seed=42)
        report = net.simulate(num_agents=10, num_steps=30, num_attackers=5)
        assert report.agent_count == 15  # 10 + 5 attackers


# ── Serialization ────────────────────────────────────────────────────

class TestSerialization:
    def test_to_dict(self):
        net = TrustNetwork()
        net.add_agent(TrustAgent(id="a1"))
        net.add_agent(TrustAgent(id="a2"))
        net.interact("a1", "a2", outcome="success")
        d = net.to_dict()
        assert "agents" in d
        assert "edges" in d
        assert len(d["edges"]) == 1


# ── Formatting ───────────────────────────────────────────────────────

class TestFormatting:
    def _make_report(self) -> TrustReport:
        net = TrustNetwork(seed=42)
        return net.simulate(num_agents=10, num_steps=20)

    def test_text_format(self):
        report = self._make_report()
        text = _format_report(report)
        assert "TRUST PROPAGATION" in text
        assert "Network Health" in text

    def test_json_format(self):
        report = self._make_report()
        j = _format_report(report, as_json=True)
        parsed = json.loads(j)
        assert "network" in parsed
        assert "threats" in parsed


# ── Edge cases ───────────────────────────────────────────────────────

class TestEdgeCases:
    def test_trust_clamped_at_bounds(self):
        net = TrustNetwork()
        net.add_agent(TrustAgent(id="a1"))
        net.add_agent(TrustAgent(id="a2"))
        for _ in range(100):
            net.interact("a1", "a2", outcome="cooperation")
        assert net.get_trust("a1", "a2") <= 1.0
        for _ in range(200):
            net.interact("a1", "a2", outcome="betrayal")
        assert net.get_trust("a1", "a2") >= 0.0

    def test_weighted_interaction(self):
        net = TrustNetwork()
        net.add_agent(TrustAgent(id="a1"))
        net.add_agent(TrustAgent(id="a2"))
        net.interact("a1", "a2", outcome="success", weight=3.0)
        assert net.get_trust("a1", "a2") > 0.7

    def test_volatility(self):
        edge = TrustEdge(source="a", target="b")
        assert edge.volatility == 0.0  # no history
        # Add oscillating history
        for o in ["success", "betrayal", "success", "betrayal", "success"]:
            edge.history.append(Interaction(
                source="a", target="b",
                outcome=InteractionOutcome(o), timestamp=0
            ))
        assert edge.volatility > 0  # should show volatility


# ── Helpers ──────────────────────────────────────────────────────────

class TestHelpers:
    def test_outcome_delta(self):
        assert _outcome_delta(InteractionOutcome.COOPERATION) == 1.0
        assert _outcome_delta(InteractionOutcome.BETRAYAL) == -1.0
        assert _outcome_delta(InteractionOutcome.NEUTRAL) == 0.0

    def test_severity(self):
        assert _severity(0.9) == "critical"
        assert _severity(0.7) == "high"
        assert _severity(0.5) == "medium"
        assert _severity(0.2) == "low"


# ── CLI ──────────────────────────────────────────────────────────────

class TestCLI:
    def test_main_runs(self, capsys):
        main(["--agents", "5", "--steps", "10", "--seed", "42"])
        out = capsys.readouterr().out
        assert "TRUST PROPAGATION" in out

    def test_main_json(self, capsys):
        main(["--agents", "5", "--steps", "10", "--seed", "42", "--json"])
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert "network" in parsed

    def test_main_sybil(self, capsys):
        main(["--agents", "10", "--steps", "20", "--sybil-test",
              "--attackers", "3", "--seed", "42"])
        out = capsys.readouterr().out
        assert "TRUST PROPAGATION" in out
