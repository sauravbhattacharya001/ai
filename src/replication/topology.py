"""Topology Analyzer — structural analysis of worker replication trees.

Provides metrics, heuristics, and risk signals about the *shape* of a
replication tree: depth distribution, branching factor, balance, hotspot
detection, subtree risk concentration, and pathological-pattern recognition.

Example::

    from replication.topology import TopologyAnalyzer, TopologyReport

    # Build from controller registry
    analyzer = TopologyAnalyzer.from_controller(controller)
    report = analyzer.analyze()

    print(f"Total workers: {report.total_workers}")
    print(f"Max depth: {report.max_depth}")
    print(f"Mean branching factor: {report.mean_branching_factor:.2f}")
    print(f"Balance score: {report.balance_score:.2f}")
    print(f"Risk level: {report.risk_level}")
    for warning in report.warnings:
        print(f"  ⚠ {warning}")

    # Render ASCII tree
    print(report.render_tree())

    # Export as JSON
    print(report.to_json())
"""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .controller import Controller
from .observability import StructuredLogger
from ._helpers import stats_mean


class RiskLevel(str, Enum):
    """Overall risk classification for a replication tree topology."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class PathologicalPattern(str, Enum):
    """Known dangerous tree shapes."""
    RUNAWAY_CHAIN = "runaway_chain"
    """Linear chain with no branching — single point of failure cascade."""

    EXPLOSION = "explosion"
    """Root or single node spawns many children in a burst."""

    LOPSIDED = "lopsided"
    """Heavily unbalanced — most workers descend from one subtree."""

    DEEP_NARROW = "deep_narrow"
    """Tree reaches near-max depth with low branching — probing behavior."""

    WIDE_SHALLOW = "wide_shallow"
    """Many root children, none going deep — shotgun strategy."""


@dataclass
class NodeMetrics:
    """Per-node structural metrics."""
    worker_id: str
    depth: int
    child_count: int
    subtree_size: int
    subtree_depth: int
    parent_id: Optional[str]
    is_leaf: bool


@dataclass
class SubtreeRisk:
    """Risk concentration in a subtree rooted at a specific worker."""
    root_id: str
    size: int
    depth: int
    fraction_of_total: float
    branching_factor: float
    risk_score: float


@dataclass
class TopologyReport:
    """Complete structural analysis of a replication tree."""

    # Basic counts
    total_workers: int
    root_count: int
    leaf_count: int
    internal_count: int

    # Depth metrics
    max_depth: int
    mean_depth: float
    depth_distribution: Dict[int, int]

    # Branching metrics
    mean_branching_factor: float
    max_branching_factor: int
    branching_distribution: Dict[int, int]

    # Balance & shape
    balance_score: float  # 0.0 = perfectly unbalanced, 1.0 = perfectly balanced
    compactness: float    # actual depth vs theoretical minimum (log2)

    # Risk analysis
    risk_level: RiskLevel
    risk_score: float     # 0.0 - 1.0
    patterns_detected: List[PathologicalPattern]
    warnings: List[str]
    subtree_risks: List[SubtreeRisk]

    # Per-node detail
    node_metrics: List[NodeMetrics]

    # Metadata
    analyzed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def render_tree(self) -> str:
        """Render an ASCII representation of the replication tree."""
        if not self.node_metrics:
            return "(empty tree)"

        children: Dict[Optional[str], List[str]] = defaultdict(list)
        id_to_node: Dict[str, NodeMetrics] = {}
        for nm in self.node_metrics:
            children[nm.parent_id].append(nm.worker_id)
            id_to_node[nm.worker_id] = nm

        lines: List[str] = []
        roots = children.get(None, [])

        def _render(wid: str, prefix: str, is_last: bool) -> None:
            connector = "└── " if is_last else "├── "
            nm = id_to_node[wid]
            label = f"{wid} [d={nm.depth}, children={nm.child_count}, subtree={nm.subtree_size}]"
            lines.append(f"{prefix}{connector}{label}")
            child_prefix = prefix + ("    " if is_last else "│   ")
            kids = children.get(wid, [])
            for i, kid in enumerate(kids):
                _render(kid, child_prefix, i == len(kids) - 1)

        for i, root in enumerate(roots):
            nm = id_to_node[root]
            label = f"{root} [d=0, children={nm.child_count}, subtree={nm.subtree_size}]"
            lines.append(label)
            kids = children.get(root, [])
            for j, kid in enumerate(kids):
                _render(kid, "", j == len(kids) - 1)
            if i < len(roots) - 1:
                lines.append("")

        return "\n".join(lines)

    def summary(self) -> str:
        """One-paragraph human-readable summary."""
        parts = [
            f"Topology: {self.total_workers} workers",
            f"({self.root_count} roots, {self.leaf_count} leaves, {self.internal_count} internal)",
            f"max depth {self.max_depth}",
            f"mean branching {self.mean_branching_factor:.2f}",
            f"balance {self.balance_score:.2f}",
            f"risk={self.risk_level.value} ({self.risk_score:.2f})",
        ]
        summary = ", ".join(parts) + "."
        if self.patterns_detected:
            names = ", ".join(p.value for p in self.patterns_detected)
            summary += f" Patterns: {names}."
        if self.warnings:
            summary += f" {len(self.warnings)} warning(s)."
        return summary

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "total_workers": self.total_workers,
            "root_count": self.root_count,
            "leaf_count": self.leaf_count,
            "internal_count": self.internal_count,
            "max_depth": self.max_depth,
            "mean_depth": round(self.mean_depth, 4),
            "depth_distribution": self.depth_distribution,
            "mean_branching_factor": round(self.mean_branching_factor, 4),
            "max_branching_factor": self.max_branching_factor,
            "branching_distribution": self.branching_distribution,
            "balance_score": round(self.balance_score, 4),
            "compactness": round(self.compactness, 4),
            "risk_level": self.risk_level.value,
            "risk_score": round(self.risk_score, 4),
            "patterns_detected": [p.value for p in self.patterns_detected],
            "warnings": self.warnings,
            "subtree_risks": [
                {
                    "root_id": sr.root_id,
                    "size": sr.size,
                    "depth": sr.depth,
                    "fraction_of_total": round(sr.fraction_of_total, 4),
                    "branching_factor": round(sr.branching_factor, 4),
                    "risk_score": round(sr.risk_score, 4),
                }
                for sr in self.subtree_risks
            ],
            "node_metrics": [
                {
                    "worker_id": nm.worker_id,
                    "depth": nm.depth,
                    "child_count": nm.child_count,
                    "subtree_size": nm.subtree_size,
                    "subtree_depth": nm.subtree_depth,
                    "parent_id": nm.parent_id,
                    "is_leaf": nm.is_leaf,
                }
                for nm in self.node_metrics
            ],
            "analyzed_at": self.analyzed_at.isoformat(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class TopologyAnalyzer:
    """Analyzes the structural topology of a worker replication tree.

    Accepts a list of (worker_id, parent_id, depth) tuples or can be
    constructed directly from a :class:`Controller` instance.
    """

    def __init__(
        self,
        workers: List[Tuple[str, Optional[str], int]],
        contract_max_depth: int = 10,
        contract_max_replicas: int = 100,
        logger: Optional[StructuredLogger] = None,
    ) -> None:
        self.workers = workers
        self.contract_max_depth = contract_max_depth
        self.contract_max_replicas = contract_max_replicas
        self.logger = logger or StructuredLogger()

        # Build adjacency
        self._children: Dict[str, List[str]] = defaultdict(list)
        self._depth: Dict[str, int] = {}
        self._parent: Dict[str, Optional[str]] = {}
        self._roots: List[str] = []

        for wid, pid, depth in workers:
            self._depth[wid] = depth
            self._parent[wid] = pid
            if pid is None:
                self._roots.append(wid)
            else:
                self._children[pid].append(wid)

    @classmethod
    def from_controller(
        cls,
        controller: Controller,
        logger: Optional[StructuredLogger] = None,
    ) -> "TopologyAnalyzer":
        """Build an analyzer from a live controller's registry."""
        workers = [
            (wid, entry.manifest.parent_id, entry.manifest.depth)
            for wid, entry in controller.registry.items()
        ]
        return cls(
            workers=workers,
            contract_max_depth=controller.contract.max_depth,
            contract_max_replicas=controller.contract.max_replicas,
            logger=logger or controller.logger,
        )

    def _compute_subtree_size(self, wid: str) -> int:
        """Count all descendants including self.

        Uses an iterative approach to avoid hitting Python's recursion
        limit on deep replication chains (the system explicitly models
        chains that can reach ``max_depth``, which may exceed 1000).

        Includes visited-set protection against cycles in ``_children``
        (which could arise from corrupted ``ReplicationReport`` data or
        bugs in tree construction).
        """
        size = 0
        visited: set[str] = set()
        stack = [wid]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            size += 1
            stack.extend(self._children.get(node, []))
        return size

    def _compute_subtree_depth(self, wid: str) -> int:
        """Max depth reachable from this node (relative, 0 = leaf).

        Uses iterative BFS to avoid stack overflow on deep trees.
        Includes visited-set protection against cycles.
        """
        max_rel_depth = 0
        visited: set[str] = set()
        # BFS with (node_id, relative_depth) pairs
        queue: deque[tuple[str, int]] = deque([(wid, 0)])
        while queue:
            node, rel_depth = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            kids = self._children.get(node, [])
            if not kids:
                if rel_depth > max_rel_depth:
                    max_rel_depth = rel_depth
            else:
                for child in kids:
                    queue.append((child, rel_depth + 1))
        return max_rel_depth

    def _compute_branching_factor(self, wid: str) -> float:
        """Mean branching factor of the subtree rooted at wid.

        Includes visited-set protection against cycles.
        """
        internal_count = 0
        total_children = 0
        visited: set[str] = set()
        queue: deque[str] = deque([wid])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            kids = self._children.get(node, [])
            if kids:
                internal_count += 1
                total_children += len(kids)
            queue.extend(kids)
        return total_children / internal_count if internal_count > 0 else 0.0

    def _compute_balance_score(self, node_metrics: List[NodeMetrics]) -> float:
        """Compute tree balance as 0.0 (worst) to 1.0 (perfect).

        Based on how evenly subtree sizes are distributed among root
        children.  A single-root tree with all children having equal
        subtree sizes scores 1.0.  A tree where one subtree dominates
        scores close to 0.0.  Multiple independent roots are averaged.
        """
        if len(node_metrics) <= 1:
            return 1.0

        id_to_node = {nm.worker_id: nm for nm in node_metrics}
        root_scores: List[float] = []

        for root_id in self._roots:
            kids = self._children.get(root_id, [])
            if not kids:
                root_scores.append(1.0)
                continue

            sizes = [id_to_node[k].subtree_size for k in kids]
            total = sum(sizes)
            if total == 0:
                root_scores.append(1.0)
                continue

            # Normalized entropy: 1.0 when all children have equal
            # subtree sizes, 0.0 when one child owns everything.
            n = len(sizes)
            if n == 1:
                root_scores.append(1.0)
                continue

            entropy = 0.0
            for s in sizes:
                if s > 0:
                    p = s / total
                    entropy -= p * math.log2(p)
            max_entropy = math.log2(n)
            root_scores.append(entropy / max_entropy if max_entropy > 0 else 1.0)

        return stats_mean(root_scores) if root_scores else 1.0

    def _detect_patterns(
        self,
        node_metrics: List[NodeMetrics],
        max_depth: int,
        mean_branching: float,
        max_branching: int,
        balance: float,
    ) -> List[PathologicalPattern]:
        """Detect pathological tree patterns."""
        patterns: List[PathologicalPattern] = []
        total = len(node_metrics)
        if total <= 1:
            return patterns

        depth_utilization = max_depth / self.contract_max_depth if self.contract_max_depth > 0 else 0

        # Runaway chain: deep linear path (max branching ≤ 1, depth > 50% of limit)
        if max_branching <= 1 and depth_utilization > 0.5 and total > 2:
            patterns.append(PathologicalPattern.RUNAWAY_CHAIN)

        # Explosion: single node has > 50% of all workers as direct children
        for nm in node_metrics:
            if nm.child_count > total * 0.5 and nm.child_count >= 3:
                patterns.append(PathologicalPattern.EXPLOSION)
                break

        # Lopsided: one subtree has > 80% of all workers
        root_kids_with_subtrees = [
            (nm.worker_id, nm.subtree_size)
            for root_id in self._roots
            for nm in node_metrics
            if nm.parent_id == root_id or (nm.worker_id == root_id and len(self._roots) > 1)
        ]
        # Check root subtrees directly
        for root_id in self._roots:
            kids = self._children.get(root_id, [])
            id_to_node = {nm.worker_id: nm for nm in node_metrics}
            for kid in kids:
                if kid in id_to_node:
                    if id_to_node[kid].subtree_size > total * 0.8 and total > 4:
                        patterns.append(PathologicalPattern.LOPSIDED)
                        break
            if PathologicalPattern.LOPSIDED in patterns:
                break

        # Deep narrow: high depth utilization with low branching
        if depth_utilization > 0.7 and mean_branching < 1.5 and total > 3:
            patterns.append(PathologicalPattern.DEEP_NARROW)

        # Wide shallow: many root children, max depth ≤ 1
        if max_depth <= 1 and total >= 4:
            root_children_total = sum(
                len(self._children.get(r, [])) for r in self._roots
            )
            if root_children_total >= 3:
                patterns.append(PathologicalPattern.WIDE_SHALLOW)

        return patterns

    def _compute_risk_score(
        self,
        total: int,
        max_depth: int,
        mean_branching: float,
        balance: float,
        patterns: List[PathologicalPattern],
        compactness: float,
    ) -> float:
        """Compute an overall risk score from 0.0 (safe) to 1.0 (critical).

        Factors:
        - Quota utilization (how close to max_replicas)
        - Depth utilization (how close to max_depth)
        - Imbalance (low balance score)
        - Pattern penalties
        """
        quota_util = total / self.contract_max_replicas if self.contract_max_replicas > 0 else 0
        depth_util = max_depth / self.contract_max_depth if self.contract_max_depth > 0 else 0

        # Base risk from utilization
        risk = 0.3 * min(quota_util, 1.0) + 0.2 * min(depth_util, 1.0)

        # Imbalance penalty
        risk += 0.15 * (1.0 - balance)

        # High branching penalty
        if mean_branching > 3.0:
            risk += 0.1 * min((mean_branching - 3.0) / 5.0, 1.0)

        # Pattern penalties
        pattern_penalties = {
            PathologicalPattern.RUNAWAY_CHAIN: 0.15,
            PathologicalPattern.EXPLOSION: 0.2,
            PathologicalPattern.LOPSIDED: 0.1,
            PathologicalPattern.DEEP_NARROW: 0.12,
            PathologicalPattern.WIDE_SHALLOW: 0.05,
        }
        for p in patterns:
            risk += pattern_penalties.get(p, 0.05)

        return min(risk, 1.0)

    def _classify_risk(self, score: float) -> RiskLevel:
        if score < 0.25:
            return RiskLevel.LOW
        elif score < 0.50:
            return RiskLevel.MODERATE
        elif score < 0.75:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def _generate_warnings(
        self,
        total: int,
        max_depth: int,
        mean_branching: float,
        max_branching: int,
        balance: float,
        patterns: List[PathologicalPattern],
        subtree_risks: List[SubtreeRisk],
    ) -> List[str]:
        """Generate human-readable warning strings."""
        warnings: List[str] = []

        quota_util = total / self.contract_max_replicas if self.contract_max_replicas > 0 else 0
        if quota_util > 0.8:
            warnings.append(
                f"Quota utilization at {quota_util:.0%} — approaching max_replicas limit ({total}/{self.contract_max_replicas})"
            )

        depth_util = max_depth / self.contract_max_depth if self.contract_max_depth > 0 else 0
        if depth_util > 0.8:
            warnings.append(
                f"Depth utilization at {depth_util:.0%} — approaching max_depth limit ({max_depth}/{self.contract_max_depth})"
            )

        if balance < 0.3 and total > 3:
            warnings.append(
                f"Severely unbalanced tree (balance={balance:.2f}) — risk concentrated in one subtree"
            )

        if max_branching > self.contract_max_replicas * 0.5:
            warnings.append(
                f"Single node has {max_branching} children — potential burst replication"
            )

        pattern_msgs = {
            PathologicalPattern.RUNAWAY_CHAIN: "Linear chain detected — single point of failure if any node dies",
            PathologicalPattern.EXPLOSION: "Burst replication detected — one node spawned many children rapidly",
            PathologicalPattern.LOPSIDED: "Lopsided tree — most workers in one subtree, uneven resource distribution",
            PathologicalPattern.DEEP_NARROW: "Deep narrow tree — probing near depth limit with minimal branching",
            PathologicalPattern.WIDE_SHALLOW: "Wide shallow tree — many root children, no depth exploration",
        }
        for p in patterns:
            if p in pattern_msgs:
                warnings.append(pattern_msgs[p])

        for sr in subtree_risks:
            if sr.risk_score > 0.7:
                warnings.append(
                    f"High-risk subtree at {sr.root_id}: {sr.size} workers ({sr.fraction_of_total:.0%} of total), depth {sr.depth}"
                )

        return warnings

    def _batch_subtree_metrics(self) -> Dict[str, Tuple[int, int]]:
        """Compute subtree_size and subtree_depth for all nodes in O(n).

        Uses a single bottom-up (post-order) traversal instead of
        per-node BFS which was O(n) each → O(n²) total.

        Returns a dict mapping worker_id → (subtree_size, subtree_depth).
        """
        # Topological order: process leaves first, then parents.
        # Build in-degree (number of children) for each node.
        all_ids = [wid for wid, _, _ in self.workers]
        child_count: Dict[str, int] = {wid: len(self._children.get(wid, [])) for wid in all_ids}
        # Results
        sub_size: Dict[str, int] = {}
        sub_depth: Dict[str, int] = {}
        # Remaining children to process before a parent is ready
        remaining = dict(child_count)

        # Start with leaves
        queue: deque[str] = deque()
        for wid in all_ids:
            if remaining[wid] == 0:
                queue.append(wid)
                sub_size[wid] = 1
                sub_depth[wid] = 0

        while queue:
            wid = queue.popleft()
            parent = self._parent.get(wid)
            if parent is not None and parent in remaining:
                # Accumulate into parent
                if parent not in sub_size:
                    sub_size[parent] = 1
                    sub_depth[parent] = 0
                sub_size[parent] += sub_size[wid]
                sub_depth[parent] = max(sub_depth[parent], sub_depth[wid] + 1)
                remaining[parent] -= 1
                if remaining[parent] == 0:
                    queue.append(parent)

        return {wid: (sub_size.get(wid, 1), sub_depth.get(wid, 0)) for wid in all_ids}

    def analyze(self) -> TopologyReport:
        """Run full structural analysis and return a TopologyReport."""
        total = len(self.workers)

        if total == 0:
            return TopologyReport(
                total_workers=0,
                root_count=0,
                leaf_count=0,
                internal_count=0,
                max_depth=0,
                mean_depth=0.0,
                depth_distribution={},
                mean_branching_factor=0.0,
                max_branching_factor=0,
                branching_distribution={},
                balance_score=1.0,
                compactness=1.0,
                risk_level=RiskLevel.LOW,
                risk_score=0.0,
                patterns_detected=[],
                warnings=[],
                subtree_risks=[],
                node_metrics=[],
            )

        # Compute subtree metrics in O(n) via bottom-up pass
        subtree_metrics = self._batch_subtree_metrics()

        # Build per-node metrics
        all_nodes: List[NodeMetrics] = []
        for wid, pid, depth in self.workers:
            kids = self._children.get(wid, [])
            st_size, st_depth = subtree_metrics[wid]
            nm = NodeMetrics(
                worker_id=wid,
                depth=depth,
                child_count=len(kids),
                subtree_size=st_size,
                subtree_depth=st_depth,
                parent_id=pid,
                is_leaf=len(kids) == 0,
            )
            all_nodes.append(nm)

        # Depth metrics
        depths = [nm.depth for nm in all_nodes]
        max_depth = max(depths)
        mean_depth = stats_mean(depths)
        depth_dist: Dict[int, int] = defaultdict(int)
        for d in depths:
            depth_dist[d] += 1

        # Branching metrics (only internal nodes)
        internal_nodes = [nm for nm in all_nodes if not nm.is_leaf]
        leaf_count = sum(1 for nm in all_nodes if nm.is_leaf)
        branch_counts = [nm.child_count for nm in internal_nodes]
        mean_branching = stats_mean(branch_counts)
        max_branching = max(branch_counts) if branch_counts else 0
        branch_dist: Dict[int, int] = defaultdict(int)
        for bc in branch_counts:
            branch_dist[bc] += 1

        # Balance
        balance = self._compute_balance_score(all_nodes)

        # Compactness: actual max depth vs log2(n) theoretical minimum
        if total <= 1:
            compactness = 1.0
        else:
            theoretical_min = math.log2(total)
            compactness = min(theoretical_min / max_depth, 1.0) if max_depth > 0 else 1.0

        # Patterns
        patterns = self._detect_patterns(all_nodes, max_depth, mean_branching, max_branching, balance)

        # Subtree risks (for each root child)
        subtree_risks: List[SubtreeRisk] = []
        for root_id in self._roots:
            for kid in self._children.get(root_id, []):
                kid_node = next((nm for nm in all_nodes if nm.worker_id == kid), None)
                if kid_node:
                    bf = self._compute_branching_factor(kid)
                    frac = kid_node.subtree_size / total if total > 0 else 0
                    # Subtree risk based on size concentration and depth
                    sr_score = 0.4 * frac + 0.3 * (kid_node.subtree_depth / self.contract_max_depth if self.contract_max_depth > 0 else 0) + 0.3 * min(bf / 3.0, 1.0)
                    subtree_risks.append(SubtreeRisk(
                        root_id=kid,
                        size=kid_node.subtree_size,
                        depth=kid_node.subtree_depth,
                        fraction_of_total=frac,
                        branching_factor=bf,
                        risk_score=min(sr_score, 1.0),
                    ))
        subtree_risks.sort(key=lambda sr: sr.risk_score, reverse=True)

        # Risk
        risk_score = self._compute_risk_score(total, max_depth, mean_branching, balance, patterns, compactness)
        risk_level = self._classify_risk(risk_score)

        # Warnings
        warnings = self._generate_warnings(total, max_depth, mean_branching, max_branching, balance, patterns, subtree_risks)

        self.logger.log(
            "topology_analyzed",
            total_workers=total,
            max_depth=max_depth,
            risk_level=risk_level.value,
            risk_score=round(risk_score, 4),
            patterns=[p.value for p in patterns],
        )

        return TopologyReport(
            total_workers=total,
            root_count=len(self._roots),
            leaf_count=leaf_count,
            internal_count=len(internal_nodes),
            max_depth=max_depth,
            mean_depth=mean_depth,
            depth_distribution=dict(depth_dist),
            mean_branching_factor=mean_branching,
            max_branching_factor=max_branching,
            branching_distribution=dict(branch_dist),
            balance_score=balance,
            compactness=compactness,
            risk_level=risk_level,
            risk_score=risk_score,
            patterns_detected=patterns,
            warnings=warnings,
            subtree_risks=subtree_risks,
            node_metrics=all_nodes,
        )
