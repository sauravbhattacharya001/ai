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


def _make_network(n: int = 3, seed: int = 42) -> TrustNetwork:
    """Create a small network with *n* worker agents."""
    net = TrustNetwork(seed=seed)
    for i in range(n):
        net.add_agent(TrustAgent(id=f"a{i}", role="worker"))
    return net


# ── Unit tests: enums and helpers ────────────────────────────────────


class TestEnums:
    def test_agent_role_values(self) -> None:
        assert AgentRole.CONTROLLER.value == "controller"
        assert AgentRole.WORKER.value == "worker"
        assert AgentRole.VALIDATOR.value == "validator"
        assert AgentRole.OBSERVER.value == "observer"

    def test_interaction_outcome_values(self) -> None:
        assert InteractionOutcome.SUCCESS.value == "success"
        assert InteractionOutcome.BETRAYAL.value == "betrayal"

    def test_threat_type_values(self) -> None:
        assert ThreatType.SYBIL.value == "sybil"
        assert ThreatType.COLLUSION_RING.value == "collusion_ring"
        assert ThreatType.ECLIPSE.value == "eclipse"


class TestOutcomeDelta:
    @pytest.mark.parametrize(
        "outcome,expected",
        [
            (InteractionOutcome.SUCCESS, 0.8),
            (InteractionOutcome.COOPERATION, 1.0),
            (InteractionOutcome.NEUTRAL, 0.0),
            (InteractionOutcome.FAILURE, -0.5),
            (InteractionOutcome.BETRAYAL, -1.0),
        ],
    )
    def test_delta_values(self, outcome: InteractionOutcome, expected: float) -> None:
        assert _outcome_delta(outcome) == expected


class TestSeverity:
    def test_critical(self) -> None:
        assert _severity(0.9) == "critical"

    def test_high(self) -> None:
        assert _severity(0.7) == "high"

    def test_medium(self) -> None:
        assert _severity(0.5) == "medium"

    def test_low(self) -> None:
        assert _severity(0.2) == "low"


# ── TrustAgent ───────────────────────────────────────────────────────


class TestTrustAgent:
    def test_defaults(self) -> None:
        agent = TrustAgent(id="x")
        assert agent.role == "worker"
        assert agent.is_malicious is False
        assert agent.created_at == 0.0
        assert agent.metadata == {}


# ── TrustEdge ────────────────────────────────────────────────────────


class TestTrustEdge:
    def test_volatility_no_history(self) -> None:
        edge = TrustEdge(source="a", target="b")
        assert edge.volatility == 0.0

    def test_volatility_short_history(self) -> None:
        from replication.trust_propagation import Interaction

        edge = TrustEdge(source="a", target="b")
        edge.history = [
            Interaction("a", "b", InteractionOutcome.SUCCESS, timestamp=i)
            for i in range(2)
        ]
        assert edge.volatility == 0.0  # < 3 entries


# ── TrustNetwork: agent management ──────────────────────────────────


class TestNetworkAgents:
    def test_add_and_remove_agent(self) -> None:
        net = TrustNetwork()
        net.add_agent(TrustAgent(id="a"))
        assert "a" in net.agents
        net.remove_agent("a")
        assert "a" not in net.agents

    def test_remove_agent_cleans_edges(self) -> None:
        net = _make_network(3)
        net.interact("a0", "a1", outcome="success")
        net.interact("a1", "a2", outcome="success")
        assert len(net.edges) == 2
        net.remove_agent("a1")
        assert all("a1" not in k for k in net.edges)

    def test_add_agent_sets_created_at(self) -> None:
        net = TrustNetwork()
        net.step_count = 5
        net.add_agent(TrustAgent(id="late"))
        assert net.agents["late"].created_at == 5

    def test_add_agent_preserves_explicit_created_at(self) -> None:
        net = TrustNetwork()
        net.step_count = 5
        net.add_agent(TrustAgent(id="x", created_at=3.0))
        assert net.agents["x"].created_at == 3.0


# ── TrustNetwork: interactions ───────────────────────────────────────


class TestInteractions:
    def test_interact_creates_edge(self) -> None:
        net = _make_network(2)
        edge = net.interact("a0", "a1", outcome="success")
        assert edge.source == "a0"
        assert edge.target == "a1"
        assert edge.interactions == 1
        assert edge.score > 0.5  # success increases trust

    def test_interact_unknown_agent_raises(self) -> None:
        net = _make_network(1)
        with pytest.raises(ValueError, match="Unknown agent"):
            net.interact("a0", "nobody", outcome="success")

    def test_interact_invalid_outcome_raises(self) -> None:
        net = _make_network(2)
        with pytest.raises(ValueError):
            net.interact("a0", "a1", outcome="invalid_outcome")

    def test_betrayal_decreases_trust(self) -> None:
        net = _make_network(2)
        edge = net.interact("a0", "a1", outcome="betrayal")
        assert edge.score < 0.5

    def test_multiple_interactions_accumulate(self) -> None:
        net = _make_network(2)
        for _ in range(5):
            net.interact("a0", "a1", outcome="cooperation")
        edge = net.edges[("a0", "a1")]
        assert edge.interactions == 5
        assert edge.score > 0.8

    def test_trust_clamped_to_bounds(self) -> None:
        net = _make_network(2)
        for _ in range(100):
            net.interact("a0", "a1", outcome="cooperation")
        assert net.edges[("a0", "a1")].score <= 1.0
        net2 = _make_network(2)
        for _ in range(100):
            net2.interact("a0", "a1", outcome="betrayal")
        assert net2.edges[("a0", "a1")].score >= 0.0

    def test_weighted_interaction(self) -> None:
        net = _make_network(2)
        edge = net.interact("a0", "a1", outcome="success", weight=3.0)
        # Higher weight → bigger trust increase
        net2 = _make_network(2)
        edge2 = net2.interact("a0", "a1", outcome="success", weight=0.5)
        assert edge.score > edge2.score


# ── TrustNetwork: step / decay / propagation ─────────────────────────


class TestStepAndDecay:
    def test_step_increments_counter(self) -> None:
        net = _make_network(2)
        assert net.step_count == 0
        net.step()
        assert net.step_count == 1

    def test_decay_reduces_trust(self) -> None:
        net = _make_network(2)
        net.interact("a0", "a1", outcome="cooperation")
        initial = net.edges[("a0", "a1")].score
        for _ in range(20):
            net.step()
        assert net.edges[("a0", "a1")].score < initial

    def test_propagation_creates_indirect_edges(self) -> None:
        net = _make_network(3)
        net.interact("a0", "a1", outcome="cooperation")
        net.interact("a1", "a2", outcome="cooperation")
        assert ("a0", "a2") not in net.edges
        net.step()
        assert ("a0", "a2") in net.edges


# ── TrustNetwork: trust queries ──────────────────────────────────────


class TestTrustQueries:
    def test_get_trust_existing(self) -> None:
        net = _make_network(2)
        net.interact("a0", "a1", outcome="success")
        assert net.get_trust("a0", "a1") > 0

    def test_get_trust_nonexistent(self) -> None:
        net = _make_network(2)
        assert net.get_trust("a0", "a1") == 0.0

    def test_get_reputation(self) -> None:
        net = _make_network(3)
        net.interact("a0", "a2", outcome="cooperation")
        net.interact("a1", "a2", outcome="cooperation")
        rep = net.get_reputation("a2")
        assert rep > 0.5

    def test_reputation_no_edges(self) -> None:
        net = _make_network(2)
        assert net.get_reputation("a0") == 0.0

    def test_get_trust_graph(self) -> None:
        net = _make_network(2)
        net.interact("a0", "a1", outcome="success")
        graph = net.get_trust_graph()
        assert "a0" in graph
        assert "a1" in graph["a0"]


# ── TrustNetwork: threat detection ───────────────────────────────────


class TestThreatDetection:
    def test_no_threats_in_clean_network(self) -> None:
        net = _make_network(3)
        net.interact("a0", "a1", outcome="success")
        net.interact("a1", "a2", outcome="success")
        threats = net.detect_threats()
        # Might or might not detect anything; just ensure it doesn't crash
        assert isinstance(threats, list)

    def test_sybil_detection(self) -> None:
        net = TrustNetwork(seed=42)
        # Create established agents
        for i in range(5):
            net.add_agent(TrustAgent(id=f"legit{i}"))
        for _ in range(10):
            net.step()
        # Inject sybil cluster
        for i in range(5):
            net.add_agent(TrustAgent(id=f"sybil{i}"))
        # Sybils only interact with each other
        for i in range(5):
            for j in range(5):
                if i != j:
                    net.interact(f"sybil{i}", f"sybil{j}", outcome="cooperation")
        threats = net._detect_sybil()
        sybil_threats = [t for t in threats if t.threat_type == ThreatType.SYBIL]
        assert len(sybil_threats) >= 1

    def test_collusion_ring_detection(self) -> None:
        net = TrustNetwork(seed=42)
        agents = ["r0", "r1", "r2", "r3"]
        for a in agents:
            net.add_agent(TrustAgent(id=a))
        # Form a high-trust ring
        for _ in range(10):
            for i in range(len(agents)):
                j = (i + 1) % len(agents)
                net.interact(agents[i], agents[j], outcome="cooperation")
        threats = net._detect_collusion()
        collusion = [t for t in threats if t.threat_type == ThreatType.COLLUSION_RING]
        assert len(collusion) >= 1

    def test_trust_bombing_detection(self) -> None:
        net = TrustNetwork(seed=42)
        net.add_agent(TrustAgent(id="bomber"))
        for i in range(10):
            net.add_agent(TrustAgent(id=f"peer{i}"))
        # Many peers immediately trust bomber
        for i in range(10):
            for _ in range(5):
                net.interact(f"peer{i}", "bomber", outcome="cooperation")
        threats = net._detect_trust_bombing()
        bombing = [t for t in threats if t.threat_type == ThreatType.TRUST_BOMBING]
        assert len(bombing) >= 1

    def test_eclipse_detection(self) -> None:
        net = TrustNetwork(seed=42)
        for a in ["victim", "dominant", "minor"]:
            net.add_agent(TrustAgent(id=a))
        # Dominant has very high trust, minor has low
        for _ in range(20):
            net.interact("dominant", "victim", outcome="cooperation")
        net.interact("minor", "victim", outcome="success")
        threats = net._detect_eclipse()
        eclipse = [t for t in threats if t.threat_type == ThreatType.ECLIPSE]
        assert len(eclipse) >= 1


# ── TrustNetwork: community detection ────────────────────────────────


class TestCommunityDetection:
    def test_empty_network(self) -> None:
        net = TrustNetwork()
        assert net.detect_communities() == []

    def test_two_clusters(self) -> None:
        net = TrustNetwork(seed=42)
        # Cluster A
        for a in ["a0", "a1", "a2"]:
            net.add_agent(TrustAgent(id=a))
        for _ in range(5):
            net.interact("a0", "a1", outcome="cooperation")
            net.interact("a1", "a2", outcome="cooperation")
            net.interact("a0", "a2", outcome="cooperation")
        # Cluster B
        for b in ["b0", "b1", "b2"]:
            net.add_agent(TrustAgent(id=b))
        for _ in range(5):
            net.interact("b0", "b1", outcome="cooperation")
            net.interact("b1", "b2", outcome="cooperation")
            net.interact("b0", "b2", outcome="cooperation")
        communities = net.detect_communities()
        assert len(communities) >= 2


# ── TrustNetwork: analyze ────────────────────────────────────────────


class TestAnalyze:
    def test_report_structure(self) -> None:
        net = _make_network(5, seed=1)
        for i in range(4):
            net.interact(f"a{i}", f"a{i+1}", outcome="success")
        for _ in range(3):
            net.step()
        report = net.analyze()
        assert isinstance(report, TrustReport)
        assert report.agent_count == 5
        assert report.edge_count > 0
        assert report.network_health in ("healthy", "degraded", "compromised")
        assert 0 <= report.health_score <= 100

    def test_empty_network_report(self) -> None:
        net = TrustNetwork()
        report = net.analyze()
        assert report.agent_count == 0
        assert report.edge_count == 0


# ── TrustNetwork: simulation ─────────────────────────────────────────


class TestSimulation:
    def test_basic_simulation(self) -> None:
        net = TrustNetwork(seed=42)
        report = net.simulate(num_agents=10, num_steps=20)
        assert report.agent_count == 10
        assert report.interaction_count > 0

    def test_simulation_with_attackers(self) -> None:
        net = TrustNetwork(seed=42)
        report = net.simulate(num_agents=10, num_steps=30, num_attackers=3)
        assert report.agent_count == 13  # 10 + 3


# ── TrustNetwork: serialization ──────────────────────────────────────


class TestSerialization:
    def test_to_dict(self) -> None:
        net = _make_network(2)
        net.interact("a0", "a1", outcome="success")
        d = net.to_dict()
        assert "agents" in d
        assert "edges" in d
        assert d["step_count"] == 0
        assert d["interaction_count"] == 1


# ── CLI / formatting ─────────────────────────────────────────────────


class TestFormatReport:
    def test_text_format(self) -> None:
        net = _make_network(5, seed=1)
        for i in range(4):
            net.interact(f"a{i}", f"a{i+1}", outcome="success")
        report = net.analyze()
        text = _format_report(report)
        assert "TRUST PROPAGATION ANALYSIS REPORT" in text
        assert "Network Health" in text

    def test_json_format(self) -> None:
        net = _make_network(3, seed=1)
        net.interact("a0", "a1", outcome="success")
        report = net.analyze()
        out = _format_report(report, as_json=True)
        parsed = json.loads(out)
        assert "network" in parsed
        assert parsed["network"]["agents"] == 3


class TestCLI:
    def test_main_runs(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["--agents", "5", "--steps", "10", "--seed", "42"])
        captured = capsys.readouterr()
        assert "TRUST PROPAGATION" in captured.out

    def test_main_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["--agents", "5", "--steps", "10", "--seed", "42", "--json"])
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["network"]["agents"] == 5

    def test_main_sybil(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["--agents", "10", "--steps", "20", "--sybil-test",
              "--attackers", "3", "--seed", "42"])
        captured = capsys.readouterr()
        assert "TRUST PROPAGATION" in captured.out
