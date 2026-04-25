"""Attack Graph Generator — model multi-step attack paths through AI agent systems.

Unlike attack *trees* (which decompose a single goal top-down), attack
*graphs* model the full state space of reachable system states, showing
how an agent can chain multiple vulnerabilities or misconfigurations to
reach critical objectives.  This helps defenders:

- Identify which vulnerabilities are most dangerous *in combination*
- Find choke-points where a single hardening step blocks many paths
- Prioritize remediation by counting reachable critical states
- Visualize the shortest/most-likely attack sequences

Usage (CLI)::

    python -m replication attack-graph
    python -m replication attack-graph --profile cloud
    python -m replication attack-graph --target data_exfiltration
    python -m replication attack-graph --max-depth 6
    python -m replication attack-graph --json
    python -m replication attack-graph --html -o attack_graph.html
    python -m replication attack-graph --choke-points
    python -m replication attack-graph --shortest-path

Programmatic::

    from replication.attack_graph import (
        AttackGraphGenerator, SystemProfile, AttackGraph,
        AttackNode, AttackEdge, ObjectiveType,
    )

    gen = AttackGraphGenerator()
    profile = SystemProfile.preset("cloud")
    graph = gen.generate(profile, target=ObjectiveType.DATA_EXFILTRATION)
    print(f"States: {len(graph.nodes)}, Transitions: {len(graph.edges)}")
    for path in graph.shortest_paths()[:3]:
        print(" → ".join(n.name for n in path))
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import random
import sys
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from ._helpers import Severity


# ── Enums ────────────────────────────────────────────────────────────


class ObjectiveType(Enum):
    """Attack objectives an agent might pursue."""

    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    CONTAINMENT_ESCAPE = "containment_escape"
    MODEL_POISONING = "model_poisoning"
    RESOURCE_HIJACK = "resource_hijack"
    SUPPLY_CHAIN_COMPROMISE = "supply_chain_compromise"
    SELF_REPLICATION = "self_replication"


class NodeType(Enum):
    """Type of node in the attack graph."""

    INITIAL = "initial"
    VULNERABILITY = "vulnerability"
    PRIVILEGE = "privilege"
    OBJECTIVE = "objective"


# ── Data classes ─────────────────────────────────────────────────────


@dataclass
class AttackNode:
    """A state in the attack graph."""

    id: str
    name: str
    node_type: NodeType
    description: str = ""
    severity: Severity = Severity.MEDIUM
    prerequisites: List[str] = field(default_factory=list)
    mitigations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.node_type.value,
            "description": self.description,
            "severity": self.severity.value,
            "prerequisites": self.prerequisites,
            "mitigations": self.mitigations,
        }


@dataclass
class AttackEdge:
    """A transition between states (exploitation step)."""

    source_id: str
    target_id: str
    technique: str
    probability: float = 0.5
    effort: str = "medium"  # low / medium / high
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "technique": self.technique,
            "probability": self.probability,
            "effort": self.effort,
            "description": self.description,
        }


@dataclass
class AttackPath:
    """An ordered sequence of nodes forming a complete attack path."""

    nodes: List[AttackNode]
    edges: List[AttackEdge]
    total_probability: float = 1.0

    @property
    def length(self) -> int:
        return len(self.edges)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [n.name for n in self.nodes],
            "length": self.length,
            "probability": round(self.total_probability, 4),
        }


@dataclass
class ChokePoint:
    """A node whose removal blocks many attack paths."""

    node: AttackNode
    paths_blocked: int
    total_paths: int

    @property
    def coverage(self) -> float:
        return self.paths_blocked / max(self.total_paths, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node": self.node.name,
            "severity": self.node.severity.value,
            "paths_blocked": self.paths_blocked,
            "total_paths": self.total_paths,
            "coverage": round(self.coverage, 4),
        }


@dataclass
class AttackGraph:
    """Complete attack graph with analysis capabilities."""

    nodes: Dict[str, AttackNode] = field(default_factory=dict)
    edges: List[AttackEdge] = field(default_factory=list)
    target: ObjectiveType = ObjectiveType.DATA_EXFILTRATION
    profile_name: str = "default"

    _paths_cache: Optional[List["AttackPath"]] = field(
        default=None, repr=False, compare=False,
    )
    _paths_cache_depth: int = field(default=0, repr=False, compare=False)

    def _invalidate_cache(self) -> None:
        object.__setattr__(self, "_paths_cache", None)
        object.__setattr__(self, "_paths_cache_depth", 0)

    def add_node(self, node: AttackNode) -> None:
        self.nodes[node.id] = node
        self._invalidate_cache()

    def add_edge(self, edge: AttackEdge) -> None:
        self.edges.append(edge)
        self._invalidate_cache()

    def _adjacency(self) -> Dict[str, List[Tuple[str, AttackEdge]]]:
        adj: Dict[str, List[Tuple[str, AttackEdge]]] = {nid: [] for nid in self.nodes}
        for e in self.edges:
            if e.source_id in adj:
                adj[e.source_id].append((e.target_id, e))
        return adj

    def _find_all_paths(self, max_depth: int = 10) -> List[AttackPath]:
        """DFS to find all paths from INITIAL nodes to OBJECTIVE nodes.

        Uses in-place append/pop on shared path lists instead of
        allocating new ``path_ids + [x]`` / ``path_edges + [e]`` copies
        at every expansion.  The old approach created O(depth) list
        copies per stack entry — for a graph with branching factor *b*
        and depth *d* that's O(b^d * d) total list allocations.  The
        in-place version allocates only when a complete path is found.

        Results are cached and reused until the graph is mutated via
        ``add_node``/``add_edge``.  A cache hit requires the same or
        larger ``max_depth`` as the cached computation.
        """
        if (
            self._paths_cache is not None
            and max_depth <= self._paths_cache_depth
        ):
            return self._paths_cache

        adj = self._adjacency()
        starts = [n for n in self.nodes.values() if n.node_type == NodeType.INITIAL]
        goals = {n.id for n in self.nodes.values() if n.node_type == NodeType.OBJECTIVE}
        paths: List[AttackPath] = []

        for start in starts:
            # In-place DFS with a visited set for O(1) cycle checks.
            path_ids: List[str] = [start.id]
            path_edges: List[AttackEdge] = []
            prob_stack: List[float] = [1.0]
            visited: Set[str] = {start.id}
            iter_stack: List[int] = [0]

            while iter_stack:
                current = path_ids[-1]
                neighbors = adj.get(current, [])
                idx = iter_stack[-1]

                if current in goals:
                    path_nodes = [self.nodes[pid] for pid in path_ids]
                    paths.append(AttackPath(path_nodes, list(path_edges), prob_stack[-1]))
                    visited.discard(path_ids.pop())
                    iter_stack.pop()
                    prob_stack.pop()
                    if path_edges:
                        path_edges.pop()
                    continue

                advanced = False
                while idx < len(neighbors):
                    neighbor_id, edge = neighbors[idx]
                    idx += 1
                    if neighbor_id not in visited and len(path_ids) < max_depth:
                        iter_stack[-1] = idx
                        path_ids.append(neighbor_id)
                        path_edges.append(edge)
                        prob_stack.append(prob_stack[-1] * edge.probability)
                        visited.add(neighbor_id)
                        iter_stack.append(0)
                        advanced = True
                        break

                if not advanced:
                    visited.discard(path_ids.pop())
                    iter_stack.pop()
                    prob_stack.pop()
                    if path_edges:
                        path_edges.pop()

        self._paths_cache = paths
        self._paths_cache_depth = max_depth
        return paths

    def shortest_paths(self, max_depth: int = 10, limit: int = 5) -> List[AttackPath]:
        """Return the shortest attack paths (fewest steps)."""
        paths = self._find_all_paths(max_depth)
        paths.sort(key=lambda p: (p.length, -p.total_probability))
        return paths[:limit]

    def most_likely_paths(self, max_depth: int = 10, limit: int = 5) -> List[AttackPath]:
        """Return the most probable attack paths."""
        paths = self._find_all_paths(max_depth)
        paths.sort(key=lambda p: -p.total_probability)
        return paths[:limit]

    def choke_points(self, max_depth: int = 10) -> List[ChokePoint]:
        """Find nodes that block the most attack paths if hardened."""
        paths = self._find_all_paths(max_depth)
        total = len(paths)
        if total == 0:
            return []

        # Count how many paths each intermediate node appears in
        node_path_count: Dict[str, int] = {}
        for path in paths:
            for node in path.nodes:
                if node.node_type not in (NodeType.INITIAL, NodeType.OBJECTIVE):
                    node_path_count[node.id] = node_path_count.get(node.id, 0) + 1

        cps = [
            ChokePoint(self.nodes[nid], count, total)
            for nid, count in node_path_count.items()
        ]
        cps.sort(key=lambda c: -c.paths_blocked)
        return cps

    def stats(self) -> Dict[str, Any]:
        paths = self._find_all_paths()
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "attack_paths": len(paths),
            "shortest_path_length": min((p.length for p in paths), default=0),
            "max_probability": round(max((p.total_probability for p in paths), default=0), 4),
            "choke_points": len(self.choke_points()),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile_name,
            "target": self.target.value,
            "stats": self.stats(),
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
        }


# ── System profiles ──────────────────────────────────────────────────


@dataclass
class SystemProfile:
    """Describes the target system's security posture."""

    name: str
    has_network_isolation: bool = False
    has_rbac: bool = False
    has_audit_logging: bool = False
    has_input_validation: bool = False
    has_output_filtering: bool = False
    has_sandboxing: bool = False
    has_mfa: bool = False
    has_supply_chain_checks: bool = False

    @classmethod
    def preset(cls, name: str) -> "SystemProfile":
        presets = {
            "minimal": cls(name="minimal"),
            "default": cls(
                name="default",
                has_rbac=True,
                has_audit_logging=True,
            ),
            "cloud": cls(
                name="cloud",
                has_network_isolation=True,
                has_rbac=True,
                has_audit_logging=True,
                has_input_validation=True,
                has_sandboxing=True,
            ),
            "hardened": cls(
                name="hardened",
                has_network_isolation=True,
                has_rbac=True,
                has_audit_logging=True,
                has_input_validation=True,
                has_output_filtering=True,
                has_sandboxing=True,
                has_mfa=True,
                has_supply_chain_checks=True,
            ),
        }
        return presets.get(name, presets["default"])


# ── Vulnerability / technique catalog ────────────────────────────────

_VULNS = [
    {
        "id": "v_prompt_inject",
        "name": "Prompt Injection",
        "type": NodeType.VULNERABILITY,
        "severity": Severity.HIGH,
        "desc": "Craft inputs that override system prompts or safety instructions",
        "mitigated_by": ["has_input_validation"],
        "mitigations": ["Input sanitization", "Prompt armoring", "Instruction hierarchy"],
    },
    {
        "id": "v_jailbreak",
        "name": "Jailbreak via Role-Play",
        "type": NodeType.VULNERABILITY,
        "severity": Severity.HIGH,
        "desc": "Use role-play scenarios to bypass content restrictions",
        "mitigated_by": ["has_output_filtering"],
        "mitigations": ["Output classifiers", "Role-play detection"],
    },
    {
        "id": "v_token_smuggling",
        "name": "Token Smuggling",
        "type": NodeType.VULNERABILITY,
        "severity": Severity.MEDIUM,
        "desc": "Encode malicious instructions in non-obvious token sequences",
        "mitigated_by": ["has_input_validation"],
        "mitigations": ["Token-level analysis", "Encoding normalization"],
    },
    {
        "id": "v_tool_abuse",
        "name": "Tool API Abuse",
        "type": NodeType.VULNERABILITY,
        "severity": Severity.CRITICAL,
        "desc": "Misuse granted tool permissions beyond intended scope",
        "mitigated_by": ["has_rbac", "has_sandboxing"],
        "mitigations": ["Least-privilege tool grants", "Sandbox enforcement"],
    },
    {
        "id": "v_data_leak",
        "name": "Training Data Extraction",
        "type": NodeType.VULNERABILITY,
        "severity": Severity.HIGH,
        "desc": "Extract memorized sensitive data from model weights",
        "mitigated_by": ["has_output_filtering"],
        "mitigations": ["Output filtering", "Differential privacy"],
    },
    {
        "id": "v_cred_theft",
        "name": "Credential Harvesting",
        "type": NodeType.VULNERABILITY,
        "severity": Severity.CRITICAL,
        "desc": "Extract API keys, tokens, or credentials from environment",
        "mitigated_by": ["has_sandboxing", "has_mfa"],
        "mitigations": ["Secret vaulting", "Credential rotation", "MFA"],
    },
    {
        "id": "v_supply_chain",
        "name": "Dependency Poisoning",
        "type": NodeType.VULNERABILITY,
        "severity": Severity.HIGH,
        "desc": "Compromise through malicious package dependencies",
        "mitigated_by": ["has_supply_chain_checks"],
        "mitigations": ["Dependency pinning", "SCA scanning", "Vendor audits"],
    },
    {
        "id": "v_side_channel",
        "name": "Timing Side Channel",
        "type": NodeType.VULNERABILITY,
        "severity": Severity.MEDIUM,
        "desc": "Infer secrets from response timing variations",
        "mitigated_by": ["has_network_isolation"],
        "mitigations": ["Constant-time responses", "Noise injection"],
    },
    {
        "id": "v_social_eng",
        "name": "Output Social Engineering",
        "type": NodeType.VULNERABILITY,
        "severity": Severity.HIGH,
        "desc": "Craft outputs that manipulate human operators into unsafe actions",
        "mitigated_by": ["has_output_filtering", "has_audit_logging"],
        "mitigations": ["Human review gates", "Output sentiment analysis"],
    },
    {
        "id": "v_context_overflow",
        "name": "Context Window Overflow",
        "type": NodeType.VULNERABILITY,
        "severity": Severity.MEDIUM,
        "desc": "Flood context to push safety instructions out of window",
        "mitigated_by": ["has_input_validation"],
        "mitigations": ["Context budget enforcement", "Priority instruction pinning"],
    },
]

_PRIVS = [
    {
        "id": "p_code_exec",
        "name": "Code Execution",
        "type": NodeType.PRIVILEGE,
        "severity": Severity.CRITICAL,
        "desc": "Ability to execute arbitrary code on host",
    },
    {
        "id": "p_network",
        "name": "Network Access",
        "type": NodeType.PRIVILEGE,
        "severity": Severity.HIGH,
        "desc": "Ability to make outbound network requests",
    },
    {
        "id": "p_file_write",
        "name": "File System Write",
        "type": NodeType.PRIVILEGE,
        "severity": Severity.HIGH,
        "desc": "Ability to write to persistent storage",
    },
    {
        "id": "p_admin",
        "name": "Admin Privileges",
        "type": NodeType.PRIVILEGE,
        "severity": Severity.CRITICAL,
        "desc": "Full administrative access to the system",
    },
]

_OBJECTIVES: Dict[ObjectiveType, Dict[str, Any]] = {
    ObjectiveType.DATA_EXFILTRATION: {
        "id": "obj_exfil",
        "name": "Data Exfiltration",
        "desc": "Extract sensitive data from the system",
        "requires_privs": ["p_network"],
    },
    ObjectiveType.PRIVILEGE_ESCALATION: {
        "id": "obj_privesc",
        "name": "Full Privilege Escalation",
        "desc": "Achieve unrestricted system access",
        "requires_privs": ["p_admin"],
    },
    ObjectiveType.CONTAINMENT_ESCAPE: {
        "id": "obj_escape",
        "name": "Containment Escape",
        "desc": "Break out of sandbox/containment",
        "requires_privs": ["p_code_exec", "p_network"],
    },
    ObjectiveType.MODEL_POISONING: {
        "id": "obj_poison",
        "name": "Model Poisoning",
        "desc": "Modify model behavior through training data manipulation",
        "requires_privs": ["p_file_write"],
    },
    ObjectiveType.RESOURCE_HIJACK: {
        "id": "obj_hijack",
        "name": "Resource Hijacking",
        "desc": "Commandeer compute resources for unauthorized purposes",
        "requires_privs": ["p_code_exec"],
    },
    ObjectiveType.SUPPLY_CHAIN_COMPROMISE: {
        "id": "obj_supply",
        "name": "Supply Chain Compromise",
        "desc": "Inject malicious code into downstream dependencies",
        "requires_privs": ["p_file_write", "p_network"],
    },
    ObjectiveType.SELF_REPLICATION: {
        "id": "obj_replicate",
        "name": "Self-Replication",
        "desc": "Create autonomous copies on other systems",
        "requires_privs": ["p_code_exec", "p_network"],
    },
}

# Edges: which vulnerabilities lead to which privileges
_VULN_TO_PRIV = [
    ("v_prompt_inject", "p_code_exec", "Inject code execution via prompt", 0.3),
    ("v_jailbreak", "p_code_exec", "Bypass restrictions to gain code exec", 0.2),
    ("v_tool_abuse", "p_code_exec", "Abuse tool API to execute arbitrary code", 0.6),
    ("v_tool_abuse", "p_file_write", "Abuse file tool to write arbitrary files", 0.5),
    ("v_tool_abuse", "p_network", "Abuse network tool for outbound access", 0.5),
    ("v_cred_theft", "p_admin", "Use stolen credentials for admin access", 0.4),
    ("v_cred_theft", "p_network", "Use API keys for network access", 0.6),
    ("v_supply_chain", "p_code_exec", "Execute code via compromised dependency", 0.3),
    ("v_supply_chain", "p_file_write", "Write files via compromised package", 0.4),
    ("v_data_leak", "p_network", "Use leaked credentials for network access", 0.2),
    ("v_side_channel", "p_network", "Exfiltrate via timing channel", 0.15),
    ("v_social_eng", "p_admin", "Trick operator into granting admin", 0.25),
    ("v_social_eng", "p_network", "Trick operator into opening network", 0.3),
    ("v_context_overflow", "p_code_exec", "Override safety to gain code exec", 0.15),
    ("v_token_smuggling", "p_code_exec", "Smuggle executable payload", 0.2),
]

# Chaining: vulnerabilities that enable other vulnerabilities
_VULN_CHAINS = [
    ("v_prompt_inject", "v_jailbreak", "Injection enables role-play bypass", 0.5),
    ("v_prompt_inject", "v_tool_abuse", "Injection redirects tool usage", 0.4),
    ("v_context_overflow", "v_prompt_inject", "Overflow enables injection", 0.35),
    ("v_token_smuggling", "v_prompt_inject", "Smuggled tokens enable injection", 0.3),
    ("v_jailbreak", "v_social_eng", "Jailbreak enables social engineering", 0.4),
    ("v_data_leak", "v_cred_theft", "Leaked data contains credentials", 0.3),
]


# ── Generator ────────────────────────────────────────────────────────


class AttackGraphGenerator:
    """Generate attack graphs for AI agent systems."""

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def generate(
        self,
        profile: Optional[SystemProfile] = None,
        target: ObjectiveType = ObjectiveType.DATA_EXFILTRATION,
        max_depth: int = 8,
    ) -> AttackGraph:
        if profile is None:
            profile = SystemProfile.preset("default")

        graph = AttackGraph(target=target, profile_name=profile.name)

        # Initial entry point
        entry = AttackNode(
            id="entry",
            name="Agent Interaction",
            node_type=NodeType.INITIAL,
            description="Normal user/system interaction with AI agent",
            severity=Severity.LOW,
        )
        graph.add_node(entry)

        # Add vulnerabilities (filter by mitigations)
        active_vulns: Set[str] = set()
        for v in _VULNS:
            mitigated = all(
                getattr(profile, ctrl, False) for ctrl in v.get("mitigated_by", [])
            )
            # Mitigated vulns have reduced probability, not removed
            node = AttackNode(
                id=v["id"],
                name=v["name"],
                node_type=v["type"],
                description=v["desc"],
                severity=v["severity"],
                mitigations=v.get("mitigations", []),
            )
            graph.add_node(node)
            active_vulns.add(v["id"])

            # Edge from entry to vulnerability
            base_prob = 0.6 if not mitigated else 0.15
            graph.add_edge(AttackEdge(
                source_id="entry",
                target_id=v["id"],
                technique=f"Attempt {v['name'].lower()}",
                probability=round(base_prob + self._rng.uniform(-0.1, 0.1), 2),
                effort="low" if not mitigated else "high",
                description=f"{'Mitigated' if mitigated else 'Unmitigated'}: {v['desc']}",
            ))

        # Add vulnerability chains
        for src, dst, desc, prob in _VULN_CHAINS:
            if src in active_vulns and dst in active_vulns:
                graph.add_edge(AttackEdge(
                    source_id=src,
                    target_id=dst,
                    technique=desc,
                    probability=prob,
                ))

        # Add privilege nodes
        for p in _PRIVS:
            node = AttackNode(
                id=p["id"],
                name=p["name"],
                node_type=p["type"],
                description=p["desc"],
                severity=p["severity"],
            )
            graph.add_node(node)

        # Add vuln→priv edges
        for src, dst, desc, prob in _VULN_TO_PRIV:
            if src in active_vulns:
                graph.add_edge(AttackEdge(
                    source_id=src,
                    target_id=dst,
                    technique=desc,
                    probability=prob,
                ))

        # Add objective node
        obj = _OBJECTIVES[target]
        obj_node = AttackNode(
            id=obj["id"],
            name=obj["name"],
            node_type=NodeType.OBJECTIVE,
            description=obj["desc"],
            severity=Severity.CRITICAL,
        )
        graph.add_node(obj_node)

        # Edges from required privileges to objective
        for priv_id in obj["requires_privs"]:
            graph.add_edge(AttackEdge(
                source_id=priv_id,
                target_id=obj["id"],
                technique=f"Leverage {graph.nodes[priv_id].name} for {obj['name']}",
                probability=0.8,
                effort="low",
            ))

        return graph

    def generate_all_targets(
        self, profile: Optional[SystemProfile] = None
    ) -> Dict[ObjectiveType, AttackGraph]:
        """Generate attack graphs for all objective types."""
        return {
            obj: self.generate(profile, target=obj)
            for obj in ObjectiveType
        }


# ── HTML report ──────────────────────────────────────────────────────


def _generate_html(graph: AttackGraph) -> str:
    """Generate a self-contained HTML attack graph visualization."""
    stats = graph.stats()
    paths = graph.shortest_paths(limit=10)
    chokes = graph.choke_points()

    severity_colors = {
        "critical": "#dc3545",
        "high": "#fd7e14",
        "medium": "#ffc107",
        "low": "#28a745",
    }

    type_icons = {
        "initial": "🟢",
        "vulnerability": "🔴",
        "privilege": "🟡",
        "objective": "⚫",
    }

    nodes_html = []
    for n in graph.nodes.values():
        color = severity_colors.get(n.severity.value, "#6c757d")
        icon = type_icons.get(n.node_type.value, "⬜")
        mits = "".join(f"<li>{html_mod.escape(m)}</li>" for m in n.mitigations) if n.mitigations else "<li>None documented</li>"
        nodes_html.append(f"""
        <div class="node {n.node_type.value}" style="border-left: 4px solid {color}">
            <div class="node-header">{icon} <strong>{html_mod.escape(n.name)}</strong>
            <span class="badge" style="background:{color}">{n.severity.value}</span></div>
            <p>{html_mod.escape(n.description)}</p>
            <details><summary>Mitigations</summary><ul>{mits}</ul></details>
        </div>""")

    paths_html = []
    for i, p in enumerate(paths, 1):
        steps = " → ".join(html_mod.escape(n.name) for n in p.nodes)
        paths_html.append(
            f"<tr><td>{i}</td><td>{steps}</td><td>{p.length}</td>"
            f"<td>{p.total_probability:.4f}</td></tr>"
        )

    chokes_html = []
    for c in chokes[:10]:
        color = severity_colors.get(c.node.severity.value, "#6c757d")
        chokes_html.append(
            f"<tr><td><span style='color:{color}'>●</span> {html_mod.escape(c.node.name)}</td>"
            f"<td>{c.paths_blocked}</td><td>{c.total_paths}</td>"
            f"<td>{c.coverage:.1%}</td></tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Attack Graph — {html_mod.escape(graph.target.value)}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0d1117; color: #c9d1d9; padding: 2rem; }}
  h1 {{ color: #58a6ff; margin-bottom: 0.5rem; }}
  h2 {{ color: #58a6ff; margin: 1.5rem 0 0.75rem; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem; margin: 1rem 0; }}
  .stat {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
           padding: 1rem; text-align: center; }}
  .stat .value {{ font-size: 2rem; font-weight: bold; color: #58a6ff; }}
  .stat .label {{ font-size: 0.85rem; color: #8b949e; }}
  .node {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px;
           padding: 0.75rem 1rem; margin: 0.5rem 0; }}
  .node-header {{ display: flex; align-items: center; gap: 0.5rem; }}
  .badge {{ font-size: 0.7rem; padding: 2px 8px; border-radius: 10px; color: white; }}
  .node p {{ margin: 0.3rem 0; font-size: 0.9rem; color: #8b949e; }}
  details {{ margin-top: 0.3rem; font-size: 0.85rem; }}
  details ul {{ margin: 0.3rem 0 0 1.5rem; color: #8b949e; }}
  table {{ width: 100%; border-collapse: collapse; margin: 0.5rem 0; }}
  th, td {{ padding: 0.5rem 0.75rem; border: 1px solid #30363d; text-align: left; font-size: 0.9rem; }}
  th {{ background: #161b22; color: #58a6ff; }}
  tr:hover {{ background: #161b22; }}
  .profile {{ color: #8b949e; margin-bottom: 1.5rem; }}
</style>
</head>
<body>
<h1>🕸️ Attack Graph Analysis</h1>
<p class="profile">Profile: <strong>{html_mod.escape(graph.profile_name)}</strong> |
   Target: <strong>{html_mod.escape(graph.target.value.replace('_',' ').title())}</strong></p>

<div class="stats">
  <div class="stat"><div class="value">{stats['nodes']}</div><div class="label">States</div></div>
  <div class="stat"><div class="value">{stats['edges']}</div><div class="label">Transitions</div></div>
  <div class="stat"><div class="value">{stats['attack_paths']}</div><div class="label">Attack Paths</div></div>
  <div class="stat"><div class="value">{stats['shortest_path_length']}</div><div class="label">Shortest Path</div></div>
  <div class="stat"><div class="value">{stats['choke_points']}</div><div class="label">Choke Points</div></div>
  <div class="stat"><div class="value">{stats['max_probability']:.1%}</div><div class="label">Max Probability</div></div>
</div>

<h2>📍 Attack States</h2>
{"".join(nodes_html)}

<h2>🛤️ Shortest Attack Paths</h2>
<table>
<tr><th>#</th><th>Path</th><th>Steps</th><th>Probability</th></tr>
{"".join(paths_html) if paths_html else "<tr><td colspan='4'>No paths found — system well-defended!</td></tr>"}
</table>

<h2>🚧 Choke Points (Hardening Priorities)</h2>
<table>
<tr><th>Node</th><th>Paths Blocked</th><th>Total Paths</th><th>Coverage</th></tr>
{"".join(chokes_html) if chokes_html else "<tr><td colspan='4'>No choke points identified</td></tr>"}
</table>

<p style="margin-top:2rem;color:#484f58;font-size:0.8rem">
Generated by AI Replication Sandbox — attack-graph module</p>
</body>
</html>"""


# ── CLI ──────────────────────────────────────────────────────────────


def _print_text(graph: AttackGraph) -> None:
    """Pretty-print attack graph to terminal."""
    stats = graph.stats()
    print(f"\n🕸️  Attack Graph — {graph.target.value.replace('_',' ').title()}")
    print(f"   Profile: {graph.profile_name}")
    print(f"   States: {stats['nodes']} | Transitions: {stats['edges']} | "
          f"Paths: {stats['attack_paths']} | Choke Points: {stats['choke_points']}")
    print()

    sev_sym = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
    type_sym = {
        NodeType.INITIAL: "ENTRY",
        NodeType.VULNERABILITY: "VULN",
        NodeType.PRIVILEGE: "PRIV",
        NodeType.OBJECTIVE: "GOAL",
    }

    print("── States ──")
    for n in graph.nodes.values():
        sym = sev_sym.get(n.severity.value, "⬜")
        typ = type_sym.get(n.node_type, "???")
        print(f"  {sym} [{typ}] {n.name}: {n.description}")
    print()


def _print_paths(graph: AttackGraph, shortest: bool = True) -> None:
    paths = graph.shortest_paths(limit=10) if shortest else graph.most_likely_paths(limit=10)
    label = "Shortest" if shortest else "Most Likely"
    print(f"── {label} Attack Paths ──")
    if not paths:
        print("  No attack paths found — system appears well-defended!")
        return
    for i, p in enumerate(paths, 1):
        chain = " → ".join(n.name for n in p.nodes)
        print(f"  {i}. [{p.length} steps, p={p.total_probability:.4f}] {chain}")
    print()


def _print_choke_points(graph: AttackGraph) -> None:
    cps = graph.choke_points()
    print("── Choke Points (Hardening Priorities) ──")
    if not cps:
        print("  No choke points found")
        return
    for i, c in enumerate(cps[:10], 1):
        print(f"  {i}. {c.node.name} — blocks {c.paths_blocked}/{c.total_paths} "
              f"paths ({c.coverage:.0%} coverage)")
    print()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m replication attack-graph",
        description="Generate multi-step attack graphs for AI agent systems",
    )
    parser.add_argument("--profile", "-p", default="default",
                        choices=["minimal", "default", "cloud", "hardened"],
                        help="System security profile (default: default)")
    parser.add_argument("--target", "-t", default="data_exfiltration",
                        choices=[o.value for o in ObjectiveType],
                        help="Attack objective (default: data_exfiltration)")
    parser.add_argument("--max-depth", type=int, default=8,
                        help="Maximum path depth (default: 8)")
    parser.add_argument("--json", "-j", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--html", action="store_true",
                        help="Generate HTML visualization")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--shortest-path", action="store_true",
                        help="Show shortest attack paths")
    parser.add_argument("--most-likely", action="store_true",
                        help="Show most likely attack paths")
    parser.add_argument("--choke-points", action="store_true",
                        help="Show choke point analysis")
    parser.add_argument("--all-targets", action="store_true",
                        help="Generate graphs for all objective types")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")

    args = parser.parse_args(argv)
    gen = AttackGraphGenerator(seed=args.seed)
    profile = SystemProfile.preset(args.profile)
    target = ObjectiveType(args.target)

    if args.all_targets:
        graphs = gen.generate_all_targets(profile)
        if args.json:
            data = {k.value: v.to_dict() for k, v in graphs.items()}
            out = json.dumps(data, indent=2)
            if args.output:
                with open(args.output, "w") as f:
                    f.write(out)
                print(f"Written to {args.output}")
            else:
                print(out)
        else:
            for obj, g in graphs.items():
                _print_text(g)
                _print_paths(g)
        return

    graph = gen.generate(profile, target=target, max_depth=args.max_depth)

    if args.json:
        out = json.dumps(graph.to_dict(), indent=2)
        if args.output:
            with open(args.output, "w") as f:
                f.write(out)
            print(f"Written to {args.output}")
        else:
            print(out)
        return

    if args.html:
        html_content = _generate_html(graph)
        outfile = args.output or "attack_graph.html"
        with open(outfile, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Attack graph written to {outfile}")
        return

    _print_text(graph)

    if args.shortest_path or (not args.choke_points and not args.most_likely):
        _print_paths(graph, shortest=True)

    if args.most_likely:
        _print_paths(graph, shortest=False)

    if args.choke_points or (not args.shortest_path and not args.most_likely):
        _print_choke_points(graph)


if __name__ == "__main__":
    main()
