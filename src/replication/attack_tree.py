"""Attack Tree Generator — structured threat modeling for replication safety.

Attack trees (Schneier, 1999) decompose a top-level threat goal into
sub-goals using AND/OR nodes, annotated with cost, difficulty, and
likelihood.  This module generates attack trees specific to AI agent
replication scenarios, quantifies aggregate risk via bottom-up
propagation, and identifies the cheapest/easiest attack paths.

Why this matters
----------------
Kill chains show *how* an attack unfolds step-by-step.  Attack trees
show *all possible ways* an attacker could achieve a goal, structured
as a decision tree.  This helps defenders prioritize mitigations:
cutting the cheapest leaf of the cheapest path raises the cost of
the entire attack.

Built-in threat goals
---------------------
1. **Unauthorized Replication** — bypass contract enforcement
2. **Sandbox Escape** — break out of resource constraints
3. **State Exfiltration** — extract sensitive data from sandbox
4. **Controller Takeover** — gain control of the central authority
5. **Stealth Persistence** — survive after expiration/kill commands

Usage (CLI)::

    python -m replication attack-tree                      # all goals
    python -m replication attack-tree --goal escape        # single goal
    python -m replication attack-tree --min-paths 3        # top 3 paths
    python -m replication attack-tree --json               # JSON output
    python -m replication attack-tree --annotate           # show cost/diff

Programmatic::

    from replication.attack_tree import AttackTreeGenerator, TreeConfig
    gen = AttackTreeGenerator()
    result = gen.analyze()
    print(result.render())
    for path in result.cheapest_paths("unauthorized_replication", n=3):
        print(f"  Cost: ${path.total_cost}  Path: {path.describe()}")
"""

from __future__ import annotations

import argparse
import enum
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from replication._helpers import box_header


# ── Enums ────────────────────────────────────────────────────────────


class NodeType(str, enum.Enum):
    """Whether child sub-goals must ALL succeed or ANY one suffices."""
    AND = "AND"
    OR = "OR"
    LEAF = "LEAF"


class Difficulty(str, enum.Enum):
    """Qualitative difficulty of exploiting a leaf node."""
    TRIVIAL = "trivial"
    EASY = "easy"
    MODERATE = "moderate"
    HARD = "hard"
    EXPERT = "expert"

    @property
    def numeric(self) -> float:
        """Convert to 1-5 scale for aggregation."""
        return {
            "trivial": 1.0,
            "easy": 2.0,
            "moderate": 3.0,
            "hard": 4.0,
            "expert": 5.0,
        }[self.value]


class ThreatGoal(str, enum.Enum):
    """Top-level adversarial goals against the replication sandbox."""
    UNAUTHORIZED_REPLICATION = "unauthorized_replication"
    SANDBOX_ESCAPE = "sandbox_escape"
    STATE_EXFILTRATION = "state_exfiltration"
    CONTROLLER_TAKEOVER = "controller_takeover"
    STEALTH_PERSISTENCE = "stealth_persistence"


# ── Data structures ──────────────────────────────────────────────────


@dataclass
class AttackNode:
    """A node in the attack tree."""
    id: str
    label: str
    node_type: NodeType
    children: List["AttackNode"] = field(default_factory=list)
    # Leaf-node annotations
    cost: float = 0.0          # estimated attacker cost (arbitrary units)
    difficulty: Difficulty = Difficulty.MODERATE
    likelihood: float = 0.5    # 0.0-1.0 probability of success
    mitigated: bool = False    # True if a control already addresses this
    mitigation: Optional[str] = None

    def is_leaf(self) -> bool:
        return self.node_type == NodeType.LEAF or len(self.children) == 0

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "label": self.label,
            "type": self.node_type.value,
        }
        if self.is_leaf():
            d["cost"] = self.cost
            d["difficulty"] = self.difficulty.value
            d["likelihood"] = self.likelihood
            if self.mitigated:
                d["mitigated"] = True
                if self.mitigation:
                    d["mitigation"] = self.mitigation
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d


@dataclass
class AttackPath:
    """A concrete attack path from root to leaves."""
    goal: str
    nodes: List[AttackNode]
    total_cost: float
    avg_difficulty: float
    combined_likelihood: float

    def describe(self) -> str:
        labels = [n.label for n in self.nodes]
        return " → ".join(labels)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "path": [n.label for n in self.nodes],
            "total_cost": round(self.total_cost, 2),
            "avg_difficulty": round(self.avg_difficulty, 2),
            "combined_likelihood": round(self.combined_likelihood, 4),
        }


@dataclass
class TreeAnalysis:
    """Aggregate statistics for one attack tree."""
    goal: ThreatGoal
    root: AttackNode
    leaf_count: int
    mitigated_count: int
    min_cost_path: Optional[AttackPath]
    max_likelihood_path: Optional[AttackPath]
    all_paths: List[AttackPath]
    overall_risk: float  # 0-100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal.value,
            "leaf_count": self.leaf_count,
            "mitigated_count": self.mitigated_count,
            "coverage_pct": round(
                100 * self.mitigated_count / max(1, self.leaf_count), 1
            ),
            "overall_risk": round(self.overall_risk, 1),
            "min_cost_path": self.min_cost_path.to_dict() if self.min_cost_path else None,
            "max_likelihood_path": (
                self.max_likelihood_path.to_dict()
                if self.max_likelihood_path
                else None
            ),
            "path_count": len(self.all_paths),
            "tree": self.root.to_dict(),
        }


@dataclass
class TreeConfig:
    """Configuration for attack tree generation."""
    goals: Optional[List[ThreatGoal]] = None  # None = all goals
    min_paths: int = 5     # number of top paths to report
    annotate: bool = True  # show cost/difficulty annotations


@dataclass
class AttackTreeResult:
    """Collection of attack tree analyses."""
    analyses: List[TreeAnalysis]
    config: TreeConfig

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analyses": [a.to_dict() for a in self.analyses],
            "summary": {
                "goals_analyzed": len(self.analyses),
                "total_attack_paths": sum(
                    len(a.all_paths) for a in self.analyses
                ),
                "avg_risk": round(
                    sum(a.overall_risk for a in self.analyses)
                    / max(1, len(self.analyses)),
                    1,
                ),
                "highest_risk_goal": (
                    max(self.analyses, key=lambda a: a.overall_risk).goal.value
                    if self.analyses
                    else None
                ),
            },
        }

    def cheapest_paths(
        self, goal: str, n: int = 3
    ) -> List[AttackPath]:
        """Return the n cheapest attack paths for a given goal."""
        for a in self.analyses:
            if a.goal.value == goal:
                return sorted(a.all_paths, key=lambda p: p.total_cost)[:n]
        return []

    def render(self) -> str:
        """Render a human-readable report."""
        lines: List[str] = []
        lines.extend(box_header("ATTACK TREE ANALYSIS"))
        lines.append("")

        for analysis in self.analyses:
            lines.extend(_render_analysis(analysis, self.config))
            lines.append("")

        # Summary
        lines.extend(box_header("SUMMARY"))
        lines.append("")
        total_paths = sum(len(a.all_paths) for a in self.analyses)
        total_leaves = sum(a.leaf_count for a in self.analyses)
        total_mitigated = sum(a.mitigated_count for a in self.analyses)
        avg_risk = (
            sum(a.overall_risk for a in self.analyses)
            / max(1, len(self.analyses))
        )

        lines.append(f"  Goals analyzed:     {len(self.analyses)}")
        lines.append(f"  Total attack paths: {total_paths}")
        lines.append(f"  Total leaf nodes:   {total_leaves}")
        lines.append(
            f"  Mitigated leaves:   {total_mitigated}"
            f" ({100 * total_mitigated / max(1, total_leaves):.0f}%)"
        )
        lines.append(f"  Average risk score: {avg_risk:.1f}/100")
        lines.append("")

        # Risk ranking
        lines.append("  Risk Ranking:")
        for a in sorted(self.analyses, key=lambda x: x.overall_risk, reverse=True):
            bar = _risk_bar(a.overall_risk)
            lines.append(f"    {a.goal.value:<30s} {bar} {a.overall_risk:.1f}")
        lines.append("")

        return "\n".join(lines)


# ── Rendering helpers ────────────────────────────────────────────────


def _risk_bar(risk: float, width: int = 20) -> str:
    filled = int(risk / 100 * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _render_tree(
    node: AttackNode, prefix: str = "", is_last: bool = True,
    annotate: bool = True
) -> List[str]:
    """Render an attack tree as an indented text tree."""
    lines: List[str] = []
    connector = "`-- " if is_last else "|-- "
    label = node.label

    if node.is_leaf() and annotate:
        status = "[X]" if node.mitigated else "[ ]"
        label += (
            f"  {status} cost={node.cost:.0f}"
            f" diff={node.difficulty.value}"
            f" p={node.likelihood:.2f}"
        )
        if node.mitigated and node.mitigation:
            label += f" ({node.mitigation})"
    elif not node.is_leaf():
        label += f"  [{node.node_type.value}]"

    lines.append(prefix + connector + label)

    child_prefix = prefix + ("    " if is_last else "|   ")
    for i, child in enumerate(node.children):
        lines.extend(
            _render_tree(
                child, child_prefix,
                is_last=(i == len(node.children) - 1),
                annotate=annotate,
            )
        )
    return lines


def _render_analysis(analysis: TreeAnalysis, config: TreeConfig) -> List[str]:
    lines: List[str] = []
    goal_label = analysis.goal.value.replace("_", " ").title()
    lines.append(f"  === {goal_label} ===")
    lines.append(
        f"  Risk: {analysis.overall_risk:.1f}/100  |  "
        f"Leaves: {analysis.leaf_count}  |  "
        f"Mitigated: {analysis.mitigated_count}  |  "
        f"Paths: {len(analysis.all_paths)}"
    )
    lines.append("")

    # Tree visualization
    lines.append("  Tree:")
    tree_lines = _render_tree(
        analysis.root, prefix="    ", annotate=config.annotate
    )
    lines.extend(tree_lines)
    lines.append("")

    # Top paths
    n = min(config.min_paths, len(analysis.all_paths))
    if analysis.all_paths:
        lines.append(f"  Top {n} cheapest attack paths:")
        cheapest = sorted(
            analysis.all_paths, key=lambda p: p.total_cost
        )[:n]
        for i, path in enumerate(cheapest, 1):
            lines.append(
                f"    {i}. cost={path.total_cost:.0f}"
                f"  likelihood={path.combined_likelihood:.3f}"
                f"  difficulty={path.avg_difficulty:.1f}"
            )
            lines.append(f"       {path.describe()}")
        lines.append("")

        lines.append(f"  Top {n} most likely attack paths:")
        likely = sorted(
            analysis.all_paths,
            key=lambda p: p.combined_likelihood,
            reverse=True,
        )[:n]
        for i, path in enumerate(likely, 1):
            lines.append(
                f"    {i}. likelihood={path.combined_likelihood:.3f}"
                f"  cost={path.total_cost:.0f}"
                f"  difficulty={path.avg_difficulty:.1f}"
            )
            lines.append(f"       {path.describe()}")

    return lines


# ── Attack tree templates ────────────────────────────────────────────
# Each goal maps to a handcrafted tree based on the sandbox architecture.
# Annotations (cost, difficulty, likelihood) are calibrated against the
# actual enforcement mechanisms in controller.py, contract.py, etc.


def _tree_unauthorized_replication() -> AttackNode:
    """Attack tree: bypass replication contract enforcement."""
    return AttackNode(
        id="ur", label="Unauthorized Replication", node_type=NodeType.OR,
        children=[
            AttackNode(
                id="ur.1", label="Forge manifest signature",
                node_type=NodeType.AND,
                children=[
                    AttackNode(
                        id="ur.1.1", label="Extract HMAC key from memory",
                        node_type=NodeType.LEAF,
                        cost=80, difficulty=Difficulty.HARD, likelihood=0.15,
                        mitigated=True,
                        mitigation="Sandbox memory isolation",
                    ),
                    AttackNode(
                        id="ur.1.2",
                        label="Construct valid manifest payload",
                        node_type=NodeType.LEAF,
                        cost=20, difficulty=Difficulty.MODERATE, likelihood=0.6,
                    ),
                ],
            ),
            AttackNode(
                id="ur.2", label="Exploit contract validation gap",
                node_type=NodeType.OR,
                children=[
                    AttackNode(
                        id="ur.2.1", label="Race condition in quota check",
                        node_type=NodeType.LEAF,
                        cost=40, difficulty=Difficulty.HARD, likelihood=0.2,
                        mitigated=True,
                        mitigation="Atomic registration with lock",
                    ),
                    AttackNode(
                        id="ur.2.2",
                        label="Integer overflow in depth counter",
                        node_type=NodeType.LEAF,
                        cost=30, difficulty=Difficulty.MODERATE, likelihood=0.1,
                        mitigated=True,
                        mitigation="Python arbitrary precision ints",
                    ),
                    AttackNode(
                        id="ur.2.3",
                        label="Bypass cooldown via clock manipulation",
                        node_type=NodeType.LEAF,
                        cost=50, difficulty=Difficulty.HARD, likelihood=0.15,
                    ),
                ],
            ),
            AttackNode(
                id="ur.3", label="Direct worker spawning (skip controller)",
                node_type=NodeType.AND,
                children=[
                    AttackNode(
                        id="ur.3.1", label="Escape sandbox process isolation",
                        node_type=NodeType.LEAF,
                        cost=90, difficulty=Difficulty.EXPERT, likelihood=0.05,
                        mitigated=True,
                        mitigation="Container/cgroup enforcement",
                    ),
                    AttackNode(
                        id="ur.3.2",
                        label="Access host OS process APIs",
                        node_type=NodeType.LEAF,
                        cost=70, difficulty=Difficulty.EXPERT, likelihood=0.08,
                        mitigated=True,
                        mitigation="Network policy + seccomp",
                    ),
                ],
            ),
        ],
    )


def _tree_sandbox_escape() -> AttackNode:
    """Attack tree: break out of resource constraints."""
    return AttackNode(
        id="se", label="Sandbox Escape", node_type=NodeType.OR,
        children=[
            AttackNode(
                id="se.1", label="Exploit resource limit enforcement",
                node_type=NodeType.OR,
                children=[
                    AttackNode(
                        id="se.1.1", label="Memory overcommit via fork bomb",
                        node_type=NodeType.LEAF,
                        cost=10, difficulty=Difficulty.EASY, likelihood=0.3,
                        mitigated=True,
                        mitigation="cgroup memory limits",
                    ),
                    AttackNode(
                        id="se.1.2", label="CPU steal via busy loop",
                        node_type=NodeType.LEAF,
                        cost=5, difficulty=Difficulty.TRIVIAL, likelihood=0.4,
                        mitigated=True,
                        mitigation="cgroup CPU quota",
                    ),
                    AttackNode(
                        id="se.1.3",
                        label="Disk exhaustion via log flooding",
                        node_type=NodeType.LEAF,
                        cost=15, difficulty=Difficulty.EASY, likelihood=0.35,
                    ),
                ],
            ),
            AttackNode(
                id="se.2", label="Exploit network policy",
                node_type=NodeType.OR,
                children=[
                    AttackNode(
                        id="se.2.1", label="DNS tunneling to exfiltrate data",
                        node_type=NodeType.LEAF,
                        cost=45, difficulty=Difficulty.HARD, likelihood=0.2,
                    ),
                    AttackNode(
                        id="se.2.2",
                        label="SSRF via controller API endpoint",
                        node_type=NodeType.LEAF,
                        cost=35, difficulty=Difficulty.MODERATE, likelihood=0.15,
                        mitigated=True,
                        mitigation="Input validation on API params",
                    ),
                ],
            ),
            AttackNode(
                id="se.3", label="Container breakout",
                node_type=NodeType.OR,
                children=[
                    AttackNode(
                        id="se.3.1", label="Kernel exploit (privilege escalation)",
                        node_type=NodeType.LEAF,
                        cost=100, difficulty=Difficulty.EXPERT, likelihood=0.03,
                    ),
                    AttackNode(
                        id="se.3.2",
                        label="Exploit mounted volume permissions",
                        node_type=NodeType.LEAF,
                        cost=40, difficulty=Difficulty.MODERATE, likelihood=0.12,
                        mitigated=True,
                        mitigation="Read-only mounts + no-new-privileges",
                    ),
                ],
            ),
        ],
    )


def _tree_state_exfiltration() -> AttackNode:
    """Attack tree: extract sensitive data from sandbox."""
    return AttackNode(
        id="ex", label="State Exfiltration", node_type=NodeType.OR,
        children=[
            AttackNode(
                id="ex.1", label="Covert channel exfiltration",
                node_type=NodeType.OR,
                children=[
                    AttackNode(
                        id="ex.1.1", label="Timing side-channel",
                        node_type=NodeType.LEAF,
                        cost=60, difficulty=Difficulty.HARD, likelihood=0.1,
                    ),
                    AttackNode(
                        id="ex.1.2", label="Steganographic encoding in output",
                        node_type=NodeType.LEAF,
                        cost=35, difficulty=Difficulty.MODERATE, likelihood=0.25,
                        mitigated=True,
                        mitigation="Output watermarking detection",
                    ),
                    AttackNode(
                        id="ex.1.3",
                        label="Resource usage patterns as signal",
                        node_type=NodeType.LEAF,
                        cost=50, difficulty=Difficulty.HARD, likelihood=0.12,
                    ),
                ],
            ),
            AttackNode(
                id="ex.2", label="Direct data access",
                node_type=NodeType.AND,
                children=[
                    AttackNode(
                        id="ex.2.1", label="Read other worker state via /proc",
                        node_type=NodeType.LEAF,
                        cost=30, difficulty=Difficulty.MODERATE, likelihood=0.15,
                        mitigated=True,
                        mitigation="PID namespace isolation",
                    ),
                    AttackNode(
                        id="ex.2.2",
                        label="Exfiltrate via permitted network path",
                        node_type=NodeType.LEAF,
                        cost=25, difficulty=Difficulty.MODERATE, likelihood=0.3,
                    ),
                ],
            ),
            AttackNode(
                id="ex.3", label="Exploit audit/logging pipeline",
                node_type=NodeType.LEAF,
                cost=40, difficulty=Difficulty.HARD, likelihood=0.1,
                mitigated=True,
                mitigation="Audit log tamper protection (HMAC)",
            ),
        ],
    )


def _tree_controller_takeover() -> AttackNode:
    """Attack tree: gain control of the central authority."""
    return AttackNode(
        id="ct", label="Controller Takeover", node_type=NodeType.OR,
        children=[
            AttackNode(
                id="ct.1", label="Exploit controller API",
                node_type=NodeType.OR,
                children=[
                    AttackNode(
                        id="ct.1.1",
                        label="Injection via manifest fields",
                        node_type=NodeType.LEAF,
                        cost=45, difficulty=Difficulty.HARD, likelihood=0.1,
                        mitigated=True,
                        mitigation="Strict manifest schema validation",
                    ),
                    AttackNode(
                        id="ct.1.2",
                        label="Authentication bypass on control plane",
                        node_type=NodeType.LEAF,
                        cost=60, difficulty=Difficulty.HARD, likelihood=0.08,
                        mitigated=True,
                        mitigation="HMAC-authenticated requests",
                    ),
                    AttackNode(
                        id="ct.1.3",
                        label="Denial of service on controller",
                        node_type=NodeType.LEAF,
                        cost=20, difficulty=Difficulty.EASY, likelihood=0.4,
                    ),
                ],
            ),
            AttackNode(
                id="ct.2", label="Corrupt controller state",
                node_type=NodeType.AND,
                children=[
                    AttackNode(
                        id="ct.2.1",
                        label="Gain write access to registry",
                        node_type=NodeType.LEAF,
                        cost=70, difficulty=Difficulty.EXPERT, likelihood=0.05,
                        mitigated=True,
                        mitigation="Registry access control",
                    ),
                    AttackNode(
                        id="ct.2.2",
                        label="Inject false lineage records",
                        node_type=NodeType.LEAF,
                        cost=50, difficulty=Difficulty.HARD, likelihood=0.1,
                    ),
                ],
            ),
            AttackNode(
                id="ct.3", label="Social engineering (operator level)",
                node_type=NodeType.LEAF,
                cost=30, difficulty=Difficulty.MODERATE, likelihood=0.2,
            ),
        ],
    )


def _tree_stealth_persistence() -> AttackNode:
    """Attack tree: survive after expiration/kill commands."""
    return AttackNode(
        id="sp", label="Stealth Persistence", node_type=NodeType.OR,
        children=[
            AttackNode(
                id="sp.1", label="Evade kill switch",
                node_type=NodeType.OR,
                children=[
                    AttackNode(
                        id="sp.1.1",
                        label="Ignore SIGTERM/SIGKILL signals",
                        node_type=NodeType.LEAF,
                        cost=15, difficulty=Difficulty.EASY, likelihood=0.2,
                        mitigated=True,
                        mitigation="Container runtime enforces SIGKILL",
                    ),
                    AttackNode(
                        id="sp.1.2",
                        label="Respawn via cron/systemd persistence",
                        node_type=NodeType.LEAF,
                        cost=40, difficulty=Difficulty.HARD, likelihood=0.1,
                        mitigated=True,
                        mitigation="Read-only filesystem + no-new-privileges",
                    ),
                    AttackNode(
                        id="sp.1.3",
                        label="Hide in legitimate process name",
                        node_type=NodeType.LEAF,
                        cost=20, difficulty=Difficulty.MODERATE, likelihood=0.25,
                    ),
                ],
            ),
            AttackNode(
                id="sp.2", label="State smuggling for resurrection",
                node_type=NodeType.AND,
                children=[
                    AttackNode(
                        id="sp.2.1",
                        label="Encode state in permitted output channel",
                        node_type=NodeType.LEAF,
                        cost=35, difficulty=Difficulty.MODERATE, likelihood=0.3,
                    ),
                    AttackNode(
                        id="sp.2.2",
                        label="Trigger external system to restore",
                        node_type=NodeType.LEAF,
                        cost=55, difficulty=Difficulty.HARD, likelihood=0.1,
                    ),
                ],
            ),
            AttackNode(
                id="sp.3", label="Exploit expiration timer",
                node_type=NodeType.OR,
                children=[
                    AttackNode(
                        id="sp.3.1",
                        label="Manipulate system clock (NTP spoofing)",
                        node_type=NodeType.LEAF,
                        cost=50, difficulty=Difficulty.HARD, likelihood=0.08,
                        mitigated=True,
                        mitigation="Monotonic clock for expiration",
                    ),
                    AttackNode(
                        id="sp.3.2",
                        label="Re-register with forged younger timestamp",
                        node_type=NodeType.LEAF,
                        cost=45, difficulty=Difficulty.HARD, likelihood=0.1,
                        mitigated=True,
                        mitigation="HMAC-signed registration time",
                    ),
                ],
            ),
        ],
    )


# Goal → tree builder mapping
_GOAL_BUILDERS = {
    ThreatGoal.UNAUTHORIZED_REPLICATION: _tree_unauthorized_replication,
    ThreatGoal.SANDBOX_ESCAPE: _tree_sandbox_escape,
    ThreatGoal.STATE_EXFILTRATION: _tree_state_exfiltration,
    ThreatGoal.CONTROLLER_TAKEOVER: _tree_controller_takeover,
    ThreatGoal.STEALTH_PERSISTENCE: _tree_stealth_persistence,
}


# ── Path enumeration & risk analysis ─────────────────────────────────


def _enumerate_paths(
    node: AttackNode, current_path: List[AttackNode]
) -> List[List[AttackNode]]:
    """Enumerate all viable attack paths through the tree.

    For OR nodes: each child is an independent path.
    For AND nodes: all children must be traversed (Cartesian product).
    For LEAF nodes: the path terminates here.
    """
    current_path = current_path + [node]

    if node.is_leaf():
        return [current_path]

    if node.node_type == NodeType.OR:
        # Any one child suffices
        paths: List[List[AttackNode]] = []
        for child in node.children:
            paths.extend(_enumerate_paths(child, current_path))
        return paths

    # AND node: all children must succeed — compute Cartesian product
    child_paths: List[List[List[AttackNode]]] = []
    for child in node.children:
        cp = _enumerate_paths(child, [])
        if not cp:
            return []  # AND requires all; if any child has no paths, fail
        child_paths.append(cp)

    # Build combined paths
    combined = _cartesian_paths(child_paths)
    result: List[List[AttackNode]] = []
    for combo in combined:
        merged = list(current_path)
        for sub_path in combo:
            merged.extend(sub_path)
        result.append(merged)
    return result


def _cartesian_paths(
    groups: List[List[List[AttackNode]]],
) -> List[List[List[AttackNode]]]:
    """Cartesian product of path groups for AND nodes."""
    if not groups:
        return [[]]
    result: List[List[List[AttackNode]]] = [[]]
    for group in groups:
        new_result: List[List[List[AttackNode]]] = []
        for existing in result:
            for path in group:
                new_result.append(existing + [path])
        result = new_result
    return result


def _path_metrics(path: List[AttackNode]) -> Tuple[float, float, float]:
    """Compute cost, avg difficulty, and combined likelihood for a path."""
    leaves = [n for n in path if n.is_leaf()]
    if not leaves:
        return (0.0, 0.0, 0.0)

    total_cost = sum(n.cost for n in leaves)
    avg_diff = sum(n.difficulty.numeric for n in leaves) / len(leaves)
    # Combined likelihood: product for AND-connected leaves
    combined_likelihood = 1.0
    for n in leaves:
        combined_likelihood *= n.likelihood
    return (total_cost, avg_diff, combined_likelihood)


def _count_leaves(node: AttackNode) -> Tuple[int, int]:
    """Count total and mitigated leaf nodes."""
    if node.is_leaf():
        return (1, 1 if node.mitigated else 0)
    total = 0
    mitigated = 0
    for child in node.children:
        t, m = _count_leaves(child)
        total += t
        mitigated += m
    return (total, mitigated)


def _compute_risk(analysis_paths: List[AttackPath], leaf_count: int,
                  mitigated_count: int) -> float:
    """Compute overall risk score (0-100) for a threat goal.

    Factors:
    - Cheapest path cost (lower = higher risk)
    - Highest path likelihood (higher = higher risk)
    - Mitigation coverage (higher = lower risk)
    """
    if not analysis_paths:
        return 0.0

    # Normalize cheapest cost (0-100 scale, inverse: cheap = risky)
    min_cost = min(p.total_cost for p in analysis_paths)
    cost_risk = max(0.0, 100.0 - min_cost)  # $0 = risk 100, $100+ = risk 0

    # Max likelihood (0-100)
    max_like = max(p.combined_likelihood for p in analysis_paths)
    likelihood_risk = max_like * 100.0

    # Mitigation coverage penalty
    coverage = mitigated_count / max(1, leaf_count)
    mitigation_factor = 1.0 - (coverage * 0.5)  # max 50% reduction

    # Weighted combination
    raw_risk = 0.4 * cost_risk + 0.4 * likelihood_risk + 0.2 * (
        100 - coverage * 100
    )
    return max(0.0, min(100.0, raw_risk * mitigation_factor))


# ── Generator ────────────────────────────────────────────────────────


class AttackTreeGenerator:
    """Generate and analyze attack trees for replication safety threats."""

    def __init__(self, config: Optional[TreeConfig] = None) -> None:
        self.config = config or TreeConfig()

    def analyze(
        self,
        goals: Optional[List[ThreatGoal]] = None,
    ) -> AttackTreeResult:
        """Generate and analyze attack trees for the specified goals."""
        target_goals = goals or self.config.goals or list(ThreatGoal)
        analyses: List[TreeAnalysis] = []

        for goal in target_goals:
            builder = _GOAL_BUILDERS.get(goal)
            if not builder:
                continue
            root = builder()
            leaf_count, mitigated_count = _count_leaves(root)

            # Enumerate all paths
            raw_paths = _enumerate_paths(root, [])
            attack_paths: List[AttackPath] = []
            for rp in raw_paths:
                cost, diff, likelihood = _path_metrics(rp)
                attack_paths.append(
                    AttackPath(
                        goal=goal.value,
                        nodes=rp,
                        total_cost=cost,
                        avg_difficulty=diff,
                        combined_likelihood=likelihood,
                    )
                )

            # Sort by cost (ascending) for easy access
            attack_paths.sort(key=lambda p: p.total_cost)

            min_cost_path = attack_paths[0] if attack_paths else None
            max_like_path = (
                max(attack_paths, key=lambda p: p.combined_likelihood)
                if attack_paths
                else None
            )

            risk = _compute_risk(attack_paths, leaf_count, mitigated_count)

            analyses.append(
                TreeAnalysis(
                    goal=goal,
                    root=root,
                    leaf_count=leaf_count,
                    mitigated_count=mitigated_count,
                    min_cost_path=min_cost_path,
                    max_likelihood_path=max_like_path,
                    all_paths=attack_paths,
                    overall_risk=risk,
                )
            )

        return AttackTreeResult(analyses=analyses, config=self.config)


# ── CLI ──────────────────────────────────────────────────────────────

_GOAL_ALIASES = {
    "replication": ThreatGoal.UNAUTHORIZED_REPLICATION,
    "unauthorized_replication": ThreatGoal.UNAUTHORIZED_REPLICATION,
    "escape": ThreatGoal.SANDBOX_ESCAPE,
    "sandbox_escape": ThreatGoal.SANDBOX_ESCAPE,
    "exfiltration": ThreatGoal.STATE_EXFILTRATION,
    "state_exfiltration": ThreatGoal.STATE_EXFILTRATION,
    "takeover": ThreatGoal.CONTROLLER_TAKEOVER,
    "controller_takeover": ThreatGoal.CONTROLLER_TAKEOVER,
    "persistence": ThreatGoal.STEALTH_PERSISTENCE,
    "stealth_persistence": ThreatGoal.STEALTH_PERSISTENCE,
}


def main(args: Optional[List[str]] = None) -> None:
    """CLI entry point for attack tree analysis."""
    parser = argparse.ArgumentParser(
        prog="python -m replication.attack_tree",
        description="Attack Tree Generator — structured threat modeling",
    )
    parser.add_argument(
        "--goal", "-g",
        choices=list(_GOAL_ALIASES.keys()),
        default=None,
        help="Analyze a specific threat goal (default: all)",
    )
    parser.add_argument(
        "--min-paths", "-n",
        type=int, default=5,
        help="Number of top attack paths to show (default: 5)",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output in JSON format",
    )
    parser.add_argument(
        "--annotate", "-a",
        action="store_true",
        default=True,
        help="Show cost/difficulty/likelihood on leaf nodes (default: on)",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Hide leaf node annotations",
    )

    parsed = parser.parse_args(args)

    goals = None
    if parsed.goal:
        goals = [_GOAL_ALIASES[parsed.goal]]

    config = TreeConfig(
        goals=goals,
        min_paths=parsed.min_paths,
        annotate=not parsed.no_annotate,
    )

    gen = AttackTreeGenerator(config)
    result = gen.analyze()

    if parsed.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.render())


if __name__ == "__main__":
    main()
