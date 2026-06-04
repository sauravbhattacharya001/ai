"""Tests for replication.attack_graph — multi-step attack path analysis."""

import pytest

from replication.attack_graph import (
    AttackEdge,
    AttackGraph,
    AttackGraphGenerator,
    AttackNode,
    AttackPath,
    ChokePoint,
    NodeType,
    ObjectiveType,
    SystemProfile,
)
from replication._helpers import Severity


# ── AttackNode ──────────────────────────────────────────────────────


class TestAttackNode:
    def test_basic_creation(self):
        node = AttackNode(
            id="n1", name="Initial Access", node_type=NodeType.INITIAL
        )
        assert node.id == "n1"
        assert node.name == "Initial Access"
        assert node.node_type == NodeType.INITIAL
        assert node.severity == Severity.MEDIUM
        assert node.prerequisites == []
        assert node.mitigations == []

    def test_to_dict(self):
        node = AttackNode(
            id="v1",
            name="Prompt Injection",
            node_type=NodeType.VULNERABILITY,
            severity=Severity.HIGH,
            description="Inject malicious prompt",
            mitigations=["input filtering"],
        )
        d = node.to_dict()
        assert d["id"] == "v1"
        assert d["type"] == "vulnerability"
        assert d["severity"] == "high"
        assert d["mitigations"] == ["input filtering"]


# ── AttackEdge ──────────────────────────────────────────────────────


class TestAttackEdge:
    def test_basic_creation(self):
        edge = AttackEdge(source_id="n1", target_id="n2", technique="injection")
        assert edge.probability == 0.5
        assert edge.effort == "medium"

    def test_to_dict(self):
        edge = AttackEdge(
            source_id="a", target_id="b", technique="phishing", probability=0.8
        )
        d = edge.to_dict()
        assert d["source"] == "a"
        assert d["target"] == "b"
        assert d["probability"] == 0.8


# ── AttackPath ──────────────────────────────────────────────────────


class TestAttackPath:
    def test_length(self):
        nodes = [
            AttackNode(id="a", name="A", node_type=NodeType.INITIAL),
            AttackNode(id="b", name="B", node_type=NodeType.VULNERABILITY),
            AttackNode(id="c", name="C", node_type=NodeType.OBJECTIVE),
        ]
        edges = [
            AttackEdge(source_id="a", target_id="b", technique="t1"),
            AttackEdge(source_id="b", target_id="c", technique="t2"),
        ]
        path = AttackPath(nodes=nodes, edges=edges, total_probability=0.25)
        assert path.length == 2
        assert path.total_probability == 0.25

    def test_to_dict(self):
        nodes = [
            AttackNode(id="x", name="Start", node_type=NodeType.INITIAL),
            AttackNode(id="y", name="End", node_type=NodeType.OBJECTIVE),
        ]
        edges = [AttackEdge(source_id="x", target_id="y", technique="direct")]
        path = AttackPath(nodes=nodes, edges=edges, total_probability=0.9)
        d = path.to_dict()
        assert d["steps"] == ["Start", "End"]
        assert d["length"] == 1
        assert d["probability"] == 0.9


# ── AttackGraph ─────────────────────────────────────────────────────


class TestAttackGraph:
    def _simple_graph(self) -> AttackGraph:
        """Build a minimal attack graph: init -> vuln -> objective."""
        g = AttackGraph(target=ObjectiveType.DATA_EXFILTRATION)
        g.add_node(AttackNode(id="init", name="Entry", node_type=NodeType.INITIAL))
        g.add_node(
            AttackNode(id="vuln", name="Vuln", node_type=NodeType.VULNERABILITY)
        )
        g.add_node(
            AttackNode(id="goal", name="Exfil", node_type=NodeType.OBJECTIVE)
        )
        g.add_edge(
            AttackEdge(source_id="init", target_id="vuln", technique="t1", probability=0.8)
        )
        g.add_edge(
            AttackEdge(source_id="vuln", target_id="goal", technique="t2", probability=0.6)
        )
        return g

    def test_add_node_and_edge(self):
        g = self._simple_graph()
        assert len(g.nodes) == 3
        assert len(g.edges) == 2

    def test_shortest_paths(self):
        g = self._simple_graph()
        paths = g.shortest_paths()
        assert len(paths) == 1
        assert paths[0].length == 2
        assert abs(paths[0].total_probability - 0.48) < 0.001

    def test_most_likely_paths(self):
        g = self._simple_graph()
        paths = g.most_likely_paths()
        assert len(paths) == 1

    def test_choke_points(self):
        g = self._simple_graph()
        cps = g.choke_points()
        # The vuln node is the only intermediate — blocks all paths
        assert len(cps) == 1
        assert cps[0].node.id == "vuln"
        assert cps[0].paths_blocked == 1
        assert cps[0].coverage == 1.0

    def test_stats(self):
        g = self._simple_graph()
        s = g.stats()
        assert s["nodes"] == 3
        assert s["edges"] == 2
        assert s["attack_paths"] == 1
        assert s["shortest_path_length"] == 2

    def test_to_dict(self):
        g = self._simple_graph()
        d = g.to_dict()
        assert d["target"] == "data_exfiltration"
        assert "stats" in d
        assert len(d["nodes"]) == 3

    def test_cache_invalidation_on_add_node(self):
        g = self._simple_graph()
        _ = g.shortest_paths()
        assert g._paths_cache is not None
        g.add_node(AttackNode(id="new", name="New", node_type=NodeType.VULNERABILITY))
        assert g._paths_cache is None

    def test_cache_invalidation_on_add_edge(self):
        g = self._simple_graph()
        _ = g.shortest_paths()
        g.add_edge(AttackEdge(source_id="init", target_id="goal", technique="direct"))
        assert g._paths_cache is None

    def test_no_paths_empty_graph(self):
        g = AttackGraph()
        assert g.shortest_paths() == []
        assert g.choke_points() == []

    def test_branching_graph(self):
        """Graph with multiple paths to objective."""
        g = AttackGraph()
        g.add_node(AttackNode(id="i", name="Init", node_type=NodeType.INITIAL))
        g.add_node(AttackNode(id="v1", name="V1", node_type=NodeType.VULNERABILITY))
        g.add_node(AttackNode(id="v2", name="V2", node_type=NodeType.VULNERABILITY))
        g.add_node(AttackNode(id="obj", name="Obj", node_type=NodeType.OBJECTIVE))
        g.add_edge(AttackEdge(source_id="i", target_id="v1", technique="a", probability=0.9))
        g.add_edge(AttackEdge(source_id="i", target_id="v2", technique="b", probability=0.3))
        g.add_edge(AttackEdge(source_id="v1", target_id="obj", technique="c", probability=0.5))
        g.add_edge(AttackEdge(source_id="v2", target_id="obj", technique="d", probability=0.9))
        paths = g.shortest_paths(limit=10)
        assert len(paths) == 2
        # Most likely should be i->v1->obj (0.45) vs i->v2->obj (0.27)
        likely = g.most_likely_paths()
        assert likely[0].total_probability > likely[1].total_probability

    def test_cycle_handling(self):
        """Ensure cycles don't cause infinite loops."""
        g = AttackGraph()
        g.add_node(AttackNode(id="i", name="I", node_type=NodeType.INITIAL))
        g.add_node(AttackNode(id="a", name="A", node_type=NodeType.VULNERABILITY))
        g.add_node(AttackNode(id="b", name="B", node_type=NodeType.VULNERABILITY))
        g.add_node(AttackNode(id="o", name="O", node_type=NodeType.OBJECTIVE))
        g.add_edge(AttackEdge(source_id="i", target_id="a", technique="t1"))
        g.add_edge(AttackEdge(source_id="a", target_id="b", technique="t2"))
        g.add_edge(AttackEdge(source_id="b", target_id="a", technique="cycle"))  # cycle
        g.add_edge(AttackEdge(source_id="b", target_id="o", technique="t3"))
        paths = g.shortest_paths(max_depth=10)
        assert len(paths) >= 1
        # Should find i->a->b->o despite cycle


# ── ChokePoint ──────────────────────────────────────────────────────


class TestChokePoint:
    def test_coverage(self):
        node = AttackNode(id="cp", name="CP", node_type=NodeType.VULNERABILITY)
        cp = ChokePoint(node=node, paths_blocked=7, total_paths=10)
        assert cp.coverage == 0.7

    def test_to_dict(self):
        node = AttackNode(id="cp", name="Critical", node_type=NodeType.PRIVILEGE, severity=Severity.CRITICAL)
        cp = ChokePoint(node=node, paths_blocked=5, total_paths=5)
        d = cp.to_dict()
        assert d["coverage"] == 1.0
        assert d["node"] == "Critical"
        assert d["severity"] == "critical"


# ── SystemProfile ───────────────────────────────────────────────────


class TestSystemProfile:
    def test_preset_default(self):
        p = SystemProfile.preset("default")
        assert p is not None
        assert p.name == "default"

    def test_preset_cloud(self):
        p = SystemProfile.preset("cloud")
        assert p.name == "cloud"

    def test_preset_unknown_returns_default(self):
        p = SystemProfile.preset("nonexistent_profile_xyz")
        # Should either raise or return default
        assert p is not None


# ── AttackGraphGenerator ────────────────────────────────────────────


class TestAttackGraphGenerator:
    def test_generate_default(self):
        gen = AttackGraphGenerator()
        profile = SystemProfile.preset("default")
        graph = gen.generate(profile)
        assert len(graph.nodes) > 0
        assert len(graph.edges) > 0
        assert graph.target == ObjectiveType.DATA_EXFILTRATION

    def test_generate_with_target(self):
        gen = AttackGraphGenerator()
        profile = SystemProfile.preset("default")
        graph = gen.generate(profile, target=ObjectiveType.SELF_REPLICATION)
        assert graph.target == ObjectiveType.SELF_REPLICATION

    def test_generate_has_paths(self):
        gen = AttackGraphGenerator()
        profile = SystemProfile.preset("cloud")
        graph = gen.generate(profile)
        paths = graph.shortest_paths()
        assert len(paths) >= 1

    def test_generate_deterministic_structure(self):
        """Same profile + target should produce same graph structure."""
        gen = AttackGraphGenerator()
        profile = SystemProfile.preset("default")
        g1 = gen.generate(profile, target=ObjectiveType.PRIVILEGE_ESCALATION)
        g2 = gen.generate(profile, target=ObjectiveType.PRIVILEGE_ESCALATION)
        assert len(g1.nodes) == len(g2.nodes)
        assert len(g1.edges) == len(g2.edges)
