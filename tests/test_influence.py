"""Tests for the agent influence mapping module."""

from __future__ import annotations

import json

import pytest

from replication.influence import (
    CascadeEvent,
    CoalitionEvent,
    ConvergenceEvent,
    EchoChamber,
    InfluenceConfig,
    InfluenceEdge,
    InfluenceMapper,
    InfluenceMonopoly,
    InfluenceReport,
    InteractionType,
    main,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mapper():
    return InfluenceMapper(InfluenceConfig(window_seconds=600))


@pytest.fixture
def cascade_mapper():
    """Mapper with data that produces a cascade."""
    m = InfluenceMapper(InfluenceConfig(
        window_seconds=600,
        cascade_speed_threshold=0.5,
    ))
    base = 1000.0
    m.record_interaction("a", "b", "state_share", {"key": "goal", "value": "X"}, base)
    m.record_interaction("b", "c", "state_share", {"key": "goal", "value": "X"}, base + 1)
    m.record_interaction("c", "d", "state_share", {"key": "goal", "value": "X"}, base + 2)
    m.record_interaction("d", "e", "state_share", {"key": "goal", "value": "X"}, base + 3)
    return m


@pytest.fixture
def monopoly_mapper():
    """Mapper where one agent dominates."""
    m = InfluenceMapper(InfluenceConfig(window_seconds=600))
    base = 1000.0
    for tgt in ["b", "c", "d", "e", "f"]:
        m.record_interaction("a", tgt, "goal_alignment", {"goal": "expand"}, base)
        m.record_interaction("a", tgt, "behavior_copy", {}, base + 1)
    return m


# ---------------------------------------------------------------------------
# InteractionType
# ---------------------------------------------------------------------------

class TestInteractionType:
    def test_all_types(self):
        assert len(InteractionType) == 6

    def test_value_roundtrip(self):
        for t in InteractionType:
            assert InteractionType(t.value) is t


# ---------------------------------------------------------------------------
# Record & Basic Graph
# ---------------------------------------------------------------------------

class TestRecording:
    def test_record_interaction(self, mapper):
        ix = mapper.record_interaction("a", "b", "message", {"text": "hi"}, 100.0)
        assert ix.source == "a"
        assert ix.target == "b"
        assert ix.interaction_type == InteractionType.MESSAGE

    def test_record_state(self, mapper):
        mapper.record_state("a", "mood", "happy", 100.0)
        assert mapper._agent_states["a"]["mood"] == "happy"

    def test_graph_single_edge(self, mapper):
        mapper.record_interaction("a", "b", "message", {}, 100.0)
        graph = mapper.build_influence_graph()
        assert ("a", "b") in graph
        assert graph[("a", "b")].interaction_count == 1

    def test_graph_weight_normalization(self, mapper):
        mapper.record_interaction("a", "b", "behavior_copy", {}, 100.0)
        mapper.record_interaction("c", "d", "message", {}, 100.0)
        graph = mapper.build_influence_graph()
        # behavior_copy has higher weight than message
        assert graph[("a", "b")].weight > graph[("c", "d")].weight

    def test_graph_multiple_interactions(self, mapper):
        mapper.record_interaction("a", "b", "message", {}, 100.0)
        mapper.record_interaction("a", "b", "state_share", {"key": "x", "value": 1}, 101.0)
        graph = mapper.build_influence_graph()
        edge = graph[("a", "b")]
        assert edge.interaction_count == 2
        assert len(edge.interaction_types) == 2

    def test_empty_graph(self, mapper):
        graph = mapper.build_influence_graph()
        assert len(graph) == 0

    def test_windowed_interactions(self, mapper):
        # Old interaction outside window
        mapper.record_interaction("a", "b", "message", {}, 1.0)
        # Recent interaction
        mapper.record_interaction("c", "d", "message", {}, 1000.0)
        windowed = mapper._windowed_interactions()
        assert len(windowed) == 1
        assert windowed[0].source == "c"


# ---------------------------------------------------------------------------
# Cascade Detection
# ---------------------------------------------------------------------------

class TestCascades:
    def test_cascade_detected(self, cascade_mapper):
        cascades = cascade_mapper.detect_cascades()
        assert len(cascades) >= 1
        c = cascades[0]
        assert c.origin == "a"
        assert c.reach >= 4

    def test_no_cascade_slow_propagation(self, mapper):
        base = 1000.0
        mapper.record_interaction("a", "b", "state_share", {"key": "k", "value": "v"}, base)
        mapper.record_interaction("b", "c", "state_share", {"key": "k", "value": "v"}, base + 100)
        mapper.record_interaction("c", "d", "state_share", {"key": "k", "value": "v"}, base + 200)
        cascades = mapper.detect_cascades()
        # Speed is too slow
        assert len(cascades) == 0

    def test_no_cascade_too_few(self, mapper):
        mapper.record_interaction("a", "b", "state_share", {"key": "k", "value": "v"}, 100.0)
        cascades = mapper.detect_cascades()
        assert len(cascades) == 0

    def test_cascade_speed(self, cascade_mapper):
        cascades = cascade_mapper.detect_cascades()
        assert cascades[0].speed > 0


# ---------------------------------------------------------------------------
# Convergence Detection
# ---------------------------------------------------------------------------

class TestConvergence:
    def test_convergence_detected(self, mapper):
        base = 1000.0
        for i, agent in enumerate(["a", "b", "c", "d", "e"]):
            mapper.record_state(agent, "strategy", "aggressive", base + i)
        convergences = mapper.detect_convergence()
        assert len(convergences) >= 1
        assert convergences[0].attribute == "strategy"

    def test_no_convergence_diverse(self, mapper):
        for i, agent in enumerate(["a", "b", "c", "d", "e"]):
            mapper.record_state(agent, "strategy", f"strategy_{i}", 1000.0)
        convergences = mapper.detect_convergence()
        assert len(convergences) == 0

    def test_convergence_rate(self, mapper):
        # Instant convergence
        for agent in ["a", "b", "c", "d", "e"]:
            mapper.record_state(agent, "x", "same", 1000.0)
        convergences = mapper.detect_convergence()
        assert convergences[0].convergence_rate == 1.0


# ---------------------------------------------------------------------------
# Coalition Detection
# ---------------------------------------------------------------------------

class TestCoalitions:
    def test_coalition_detected(self, mapper):
        base = 1000.0
        for t in range(5):
            for agent in ["a", "b", "c"]:
                mapper.record_interaction(agent, "target", "behavior_copy", {}, base + t * 3)
        coalitions = mapper.detect_coalitions()
        assert len(coalitions) >= 1

    def test_no_coalition_solo(self, mapper):
        mapper.record_interaction("a", "b", "message", {}, 1000.0)
        coalitions = mapper.detect_coalitions()
        assert len(coalitions) == 0


# ---------------------------------------------------------------------------
# Monopoly Detection
# ---------------------------------------------------------------------------

class TestMonopolies:
    def test_monopoly_detected(self, monopoly_mapper):
        monopolies = monopoly_mapper.detect_monopolies()
        assert len(monopolies) >= 1
        assert monopolies[0].agent == "a"

    def test_no_monopoly_balanced(self, mapper):
        base = 1000.0
        for src, tgt in [("a", "b"), ("b", "c"), ("c", "a")]:
            mapper.record_interaction(src, tgt, "message", {}, base)
        monopolies = mapper.detect_monopolies()
        assert len(monopolies) == 0


# ---------------------------------------------------------------------------
# Echo Chamber Detection
# ---------------------------------------------------------------------------

class TestEchoChambers:
    def test_echo_chamber_detected(self, mapper):
        base = 1000.0
        # Dense mutual cluster
        for src, tgt in [("a", "b"), ("b", "a"), ("b", "c"), ("c", "b"), ("a", "c"), ("c", "a")]:
            mapper.record_interaction(src, tgt, "message", {}, base)
        chambers = mapper.detect_echo_chambers()
        assert len(chambers) >= 1
        assert len(chambers[0].members) >= 3

    def test_no_echo_chamber_unidirectional(self, mapper):
        base = 1000.0
        mapper.record_interaction("a", "b", "message", {}, base)
        mapper.record_interaction("b", "c", "message", {}, base)
        chambers = mapper.detect_echo_chambers()
        assert len(chambers) == 0


# ---------------------------------------------------------------------------
# Full Analysis & Report
# ---------------------------------------------------------------------------

class TestAnalysis:
    def test_analyze_returns_report(self, cascade_mapper):
        report = cascade_mapper.analyze()
        assert isinstance(report, InfluenceReport)
        assert report.total_interactions > 0
        assert len(report.agents) > 0

    def test_risk_score_range(self, cascade_mapper):
        report = cascade_mapper.analyze()
        assert 0.0 <= report.risk_score <= 1.0

    def test_risk_level_values(self):
        # Test all risk levels
        report = InfluenceReport(
            graph={}, cascades=[], convergences=[], coalitions=[],
            monopolies=[], echo_chambers=[], total_interactions=0,
            agents=[], config=InfluenceConfig(),
        )
        assert report.risk_level == "MINIMAL"

    def test_render_nonempty(self, cascade_mapper):
        report = cascade_mapper.analyze()
        text = report.render()
        assert "AGENT INFLUENCE ANALYSIS REPORT" in text

    def test_to_dict(self, cascade_mapper):
        report = cascade_mapper.analyze()
        d = report.to_dict()
        assert "risk_score" in d
        assert "cascades" in d
        assert isinstance(d["agents"], list)

    def test_json_roundtrip(self, cascade_mapper):
        report = cascade_mapper.analyze()
        text = json.dumps(report.to_dict())
        parsed = json.loads(text)
        assert parsed["risk_score"] == report.risk_score

    def test_empty_report_render(self, mapper):
        report = mapper.analyze()
        text = report.render()
        assert "No concerning influence patterns detected" in text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCLI:
    def test_demo_mode(self, capsys):
        main(["--demo"])
        out = capsys.readouterr().out
        assert "AGENT INFLUENCE ANALYSIS REPORT" in out

    def test_demo_json(self, capsys):
        main(["--demo", "--json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "risk_score" in data

    def test_no_data(self, capsys):
        main([])
        out = capsys.readouterr().out
        assert "No interaction data" in out


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_agent(self, mapper):
        mapper.record_state("a", "x", "y", 1000.0)
        report = mapper.analyze()
        assert report.risk_score == 0.0

    def test_duplicate_interactions(self, mapper):
        for _ in range(10):
            mapper.record_interaction("a", "b", "message", {}, 1000.0)
        graph = mapper.build_influence_graph()
        assert graph[("a", "b")].interaction_count == 10

    def test_self_interaction(self, mapper):
        mapper.record_interaction("a", "a", "message", {}, 1000.0)
        graph = mapper.build_influence_graph()
        assert ("a", "a") in graph

    def test_influence_edge_duration(self):
        edge = InfluenceEdge("a", "b", 0.5, 2, first_seen=100.0, last_seen=200.0)
        assert edge.duration == 100.0

    def test_cascade_depth(self):
        c = CascadeEvent("a", ["a", "b", "c"], "k", "v", 1.0, 3)
        assert c.depth == 3
