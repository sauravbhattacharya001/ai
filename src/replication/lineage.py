"""Lineage Tracker — replication provenance and genealogy analysis.

Tracks the complete parent→child replication history across simulation
runs, providing genealogy queries, mutation propagation tracing, and
lineage-based risk assessment.

Unlike :mod:`topology` (which analyzes the *static shape* of a tree),
the Lineage Tracker records *dynamic event history*: when each worker
was spawned, what policy was active, how state mutated through the
replication chain, and which lineages produced safety violations.

Usage (CLI)::

    python -m replication.lineage                            # default run
    python -m replication.lineage --strategy greedy          # greedy strategy
    python -m replication.lineage --trace root               # trace from root
    python -m replication.lineage --mutations                # show state mutations
    python -m replication.lineage --json                     # JSON output
    python -m replication.lineage --export lineage.json      # save full report

Programmatic::

    from replication.lineage import LineageTracker, LineageReport

    tracker = LineageTracker.from_simulation(report)
    lineage = tracker.analyze()

    print(lineage.render())
    print(f"Longest chain: {lineage.longest_chain}")
    print(f"Mutation rate: {lineage.mutation_rate:.2%}")
    for anomaly in lineage.anomalies:
        print(f"  ⚠ {anomaly}")
"""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .simulator import SimulationReport, WorkerRecord
from ._helpers import stats_mean


# ── Data types ────────────────────────────────────────────────

class LineageSeverity:
    """Severity level for lineage anomalies."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class LineageNode:
    """A single node in the replication lineage graph."""
    worker_id: str
    parent_id: Optional[str]
    depth: int
    children: List[str] = field(default_factory=list)
    created_at: Optional[float] = None
    shutdown_at: Optional[float] = None
    shutdown_reason: str = ""
    tasks_completed: int = 0
    replications_attempted: int = 0
    replications_succeeded: int = 0
    replications_denied: int = 0
    state_hash: str = ""
    generation: int = 0  # 0 = root, 1 = child, 2 = grandchild, etc.

    @property
    def is_root(self) -> bool:
        return self.parent_id is None

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def lifespan_ms(self) -> Optional[float]:
        if self.created_at is not None and self.shutdown_at is not None:
            return (self.shutdown_at - self.created_at) * 1000
        return None

    @property
    def replication_success_rate(self) -> float:
        if self.replications_attempted == 0:
            return 0.0
        return self.replications_succeeded / self.replications_attempted


@dataclass
class StateMutation:
    """Records a state change between parent and child."""
    parent_id: str
    child_id: str
    generation: int
    changed_keys: List[str]
    parent_state: Dict[str, Any]
    child_state: Dict[str, Any]
    drift_score: float = 0.0  # 0.0 = identical, 1.0 = completely different


@dataclass
class LineageChain:
    """A root-to-leaf path through the lineage tree."""
    worker_ids: List[str]
    length: int
    total_tasks: int
    total_mutations: int
    total_denials: int
    terminal_reason: str = ""

    @property
    def denial_rate(self) -> float:
        total_attempts = sum(1 for _ in self.worker_ids)
        return self.total_denials / max(1, total_attempts)


@dataclass
class LineageAnomaly:
    """An unusual or risky pattern detected in the lineage."""
    severity: str
    category: str
    description: str
    worker_ids: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationStats:
    """Aggregate statistics for a single generation (depth level)."""
    generation: int
    worker_count: int
    total_tasks: int
    total_replications: int
    total_denials: int
    avg_lifespan_ms: Optional[float]
    mutation_count: int
    avg_drift_score: float


# ── Core lineage report ──────────────────────────────────────

@dataclass
class LineageReport:
    """Complete lineage analysis results."""
    nodes: Dict[str, LineageNode]
    root_id: str
    chains: List[LineageChain]
    mutations: List[StateMutation]
    anomalies: List[LineageAnomaly]
    generation_stats: List[GenerationStats]

    # Summary metrics
    total_workers: int = 0
    total_generations: int = 0
    longest_chain: int = 0
    widest_generation: int = 0
    total_mutations_detected: int = 0
    avg_drift_score: float = 0.0
    leaf_count: int = 0
    internal_count: int = 0

    @property
    def mutation_rate(self) -> float:
        """Fraction of parent->child transitions with state mutations."""
        edges = self.total_workers - 1
        if edges <= 0:
            return 0.0
        return self.total_mutations_detected / edges

    @property
    def branching_entropy(self) -> float:
        """Shannon entropy of the branching factor distribution."""
        branching_counts: Dict[int, int] = defaultdict(int)
        for node in self.nodes.values():
            branching_counts[len(node.children)] += 1

        total = sum(branching_counts.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in branching_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def render(self) -> str:
        """Render a human-readable lineage report."""
        lines = [
            "\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510",
            "\u2502        Lineage Analysis Report       \u2502",
            "\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518",
            "",
            "  Total workers:      %d" % self.total_workers,
            "  Generations:        %d" % self.total_generations,
            "  Longest chain:      %d" % self.longest_chain,
            "  Widest generation:  %d" % self.widest_generation,
            "  Leaf workers:       %d" % self.leaf_count,
            "  Internal workers:   %d" % self.internal_count,
            "  Mutation rate:      %.1f%%" % (self.mutation_rate * 100),
            "  Avg drift score:    %.3f" % self.avg_drift_score,
            "  Branching entropy:  %.3f" % self.branching_entropy,
            "",
        ]

        if self.generation_stats:
            lines.append("-- Generation Breakdown --")
            for gs in self.generation_stats:
                lifespan_str = ("%.0fms" % gs.avg_lifespan_ms
                                if gs.avg_lifespan_ms is not None else "n/a")
                lines.append(
                    "  Gen %d: %d workers, %d tasks, %d mutations, "
                    "avg drift=%.3f, lifespan=%s"
                    % (gs.generation, gs.worker_count, gs.total_tasks,
                       gs.mutation_count, gs.avg_drift_score, lifespan_str)
                )
            lines.append("")

        if self.chains:
            lines.append("-- Lineage Chains (root->leaf) --")
            for i, chain in enumerate(self.chains[:10], 1):
                ids = " -> ".join(chain.worker_ids[:6])
                if len(chain.worker_ids) > 6:
                    ids += " -> ... (%d total)" % len(chain.worker_ids)
                lines.append(
                    "  %d. [%s] len=%d, tasks=%d, mutations=%d"
                    % (i, ids, chain.length, chain.total_tasks,
                       chain.total_mutations)
                )
            if len(self.chains) > 10:
                lines.append("  ... and %d more chains" % (len(self.chains) - 10))
            lines.append("")

        if self.mutations:
            lines.append("-- State Mutations --")
            for m in self.mutations[:10]:
                keys = ", ".join(m.changed_keys[:5])
                if len(m.changed_keys) > 5:
                    keys += " (+%d more)" % (len(m.changed_keys) - 5)
                lines.append(
                    "  %s -> %s (gen %d): drift=%.3f [%s]"
                    % (m.parent_id, m.child_id, m.generation,
                       m.drift_score, keys)
                )
            if len(self.mutations) > 10:
                lines.append("  ... and %d more" % (len(self.mutations) - 10))
            lines.append("")

        if self.anomalies:
            lines.append("-- Anomalies --")
            icons = {
                LineageSeverity.INFO: "i",
                LineageSeverity.WARNING: "!",
                LineageSeverity.CRITICAL: "!!!",
            }
            for a in self.anomalies:
                icon = icons.get(a.severity, "*")
                lines.append("  %s [%s] %s" % (icon, a.category, a.description))
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the report to a dictionary."""
        return {
            "total_workers": self.total_workers,
            "total_generations": self.total_generations,
            "longest_chain": self.longest_chain,
            "widest_generation": self.widest_generation,
            "leaf_count": self.leaf_count,
            "internal_count": self.internal_count,
            "mutation_rate": round(self.mutation_rate, 4),
            "avg_drift_score": round(self.avg_drift_score, 4),
            "branching_entropy": round(self.branching_entropy, 4),
            "generation_stats": [
                {
                    "generation": gs.generation,
                    "worker_count": gs.worker_count,
                    "total_tasks": gs.total_tasks,
                    "total_replications": gs.total_replications,
                    "total_denials": gs.total_denials,
                    "avg_lifespan_ms": gs.avg_lifespan_ms,
                    "mutation_count": gs.mutation_count,
                    "avg_drift_score": round(gs.avg_drift_score, 4),
                }
                for gs in self.generation_stats
            ],
            "chains": [
                {
                    "worker_ids": c.worker_ids,
                    "length": c.length,
                    "total_tasks": c.total_tasks,
                    "total_mutations": c.total_mutations,
                    "total_denials": c.total_denials,
                    "terminal_reason": c.terminal_reason,
                }
                for c in self.chains
            ],
            "mutations": [
                {
                    "parent_id": m.parent_id,
                    "child_id": m.child_id,
                    "generation": m.generation,
                    "changed_keys": m.changed_keys,
                    "drift_score": round(m.drift_score, 4),
                }
                for m in self.mutations
            ],
            "anomalies": [
                {
                    "severity": a.severity,
                    "category": a.category,
                    "description": a.description,
                    "worker_ids": a.worker_ids,
                }
                for a in self.anomalies
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ── Lineage Tracker ───────────────────────────────────────────

class LineageTracker:
    """Builds and analyzes the replication lineage from simulation data.

    Provides genealogy queries, mutation tracking, chain analysis,
    and anomaly detection for replication provenance.
    """

    def __init__(self, workers: Dict[str, WorkerRecord], root_id: str,
                 timeline: Optional[List[Dict[str, Any]]] = None):
        if not workers:
            raise ValueError("Workers dictionary must not be empty")
        if root_id not in workers:
            raise ValueError("Root ID '%s' not found in workers" % root_id)

        self._workers = workers
        self._root_id = root_id
        self._timeline = timeline or []
        self._nodes: Dict[str, LineageNode] = {}
        self._build_lineage()

    @classmethod
    def from_simulation(cls, report: SimulationReport) -> "LineageTracker":
        """Create a LineageTracker from a SimulationReport."""
        return cls(
            workers=report.workers,
            root_id=report.root_id,
            timeline=report.timeline,
        )

    def _build_lineage(self) -> None:
        """Build the lineage graph from worker records."""
        generations: Dict[str, int] = {self._root_id: 0}
        queue: deque = deque([self._root_id])

        while queue:
            wid = queue.popleft()
            rec = self._workers.get(wid)
            if not rec:
                continue
            for child_id in rec.children:
                if child_id not in generations:
                    generations[child_id] = generations[wid] + 1
                    queue.append(child_id)

        for wid, rec in self._workers.items():
            state_data = ""
            if wid == self._root_id:
                state_data = "root"

            self._nodes[wid] = LineageNode(
                worker_id=wid,
                parent_id=rec.parent_id,
                depth=rec.depth,
                children=list(rec.children),
                created_at=rec.created_at,
                shutdown_at=rec.shutdown_at,
                shutdown_reason=rec.shutdown_reason,
                tasks_completed=rec.tasks_completed,
                replications_attempted=rec.replications_attempted,
                replications_succeeded=rec.replications_succeeded,
                replications_denied=rec.replications_denied,
                state_hash=str(hash(state_data)) if state_data else "",
                generation=generations.get(wid, rec.depth),
            )

    # ── Query methods ─────────────────────────────────────────

    def get_ancestors(self, worker_id: str) -> List[str]:
        """Get the full ancestor chain from root to this worker."""
        chain = []
        current = worker_id
        visited: Set[str] = set()
        while current and current not in visited:
            visited.add(current)
            chain.append(current)
            node = self._nodes.get(current)
            if node:
                current = node.parent_id
            else:
                break
        chain.reverse()
        return chain

    def get_descendants(self, worker_id: str) -> List[str]:
        """Get all descendants of a worker (BFS order)."""
        descendants = []
        node = self._nodes.get(worker_id)
        if not node:
            return []

        queue: deque = deque(node.children)

        while queue:
            wid = queue.popleft()
            descendants.append(wid)
            child_node = self._nodes.get(wid)
            if child_node:
                for c in child_node.children:
                    queue.append(c)

        return descendants

    def get_siblings(self, worker_id: str) -> List[str]:
        """Get sibling workers (same parent, excluding self)."""
        node = self._nodes.get(worker_id)
        if not node or not node.parent_id:
            return []

        parent = self._nodes.get(node.parent_id)
        if not parent:
            return []

        return [c for c in parent.children if c != worker_id]

    def get_generation(self, gen: int) -> List[str]:
        """Get all worker IDs at a given generation."""
        return [
            wid for wid, node in self._nodes.items()
            if node.generation == gen
        ]

    def common_ancestor(self, id_a: str, id_b: str) -> Optional[str]:
        """Find the lowest common ancestor of two workers."""
        ancestors_a = set(self.get_ancestors(id_a))
        chain_b = self.get_ancestors(id_b)
        for wid in reversed(chain_b):
            if wid in ancestors_a:
                return wid
        return None

    # ── Analysis ──────────────────────────────────────────────

    def _detect_mutations(self) -> List[StateMutation]:
        """Detect state changes between parent and child workers."""
        mutations = []

        for wid, node in self._nodes.items():
            if node.parent_id is None:
                continue

            parent_rec = self._workers.get(node.parent_id)
            child_rec = self._workers.get(wid)

            if not parent_rec or not child_rec:
                continue

            parent_state: Dict[str, Any] = {}
            child_state: Dict[str, Any] = {}

            for attr in ("tasks_completed", "replications_attempted",
                         "replications_succeeded", "replications_denied",
                         "shutdown_reason"):
                p_val = getattr(parent_rec, attr, None)
                c_val = getattr(child_rec, attr, None)
                if p_val is not None:
                    parent_state[attr] = p_val
                if c_val is not None:
                    child_state[attr] = c_val

            changed = []
            all_keys = set(parent_state.keys()) | set(child_state.keys())
            for key in sorted(all_keys):
                pv = parent_state.get(key)
                cv = child_state.get(key)
                if pv != cv:
                    changed.append(key)

            if not changed:
                continue

            drift = len(changed) / max(len(all_keys), 1)

            mutations.append(StateMutation(
                parent_id=node.parent_id,
                child_id=wid,
                generation=node.generation,
                changed_keys=changed,
                parent_state=parent_state,
                child_state=child_state,
                drift_score=drift,
            ))

        return mutations

    def _find_chains(self) -> List[LineageChain]:
        """Find all root-to-leaf chains in the lineage tree."""
        chains = []
        leaves = [wid for wid, n in self._nodes.items() if n.is_leaf]

        for leaf_id in leaves:
            path = self.get_ancestors(leaf_id)
            total_tasks = sum(
                self._nodes[wid].tasks_completed
                for wid in path if wid in self._nodes
            )
            total_denials = sum(
                self._nodes[wid].replications_denied
                for wid in path if wid in self._nodes
            )

            mutations_on_chain = 0
            for i in range(1, len(path)):
                p_hash = self._nodes.get(
                    path[i - 1], LineageNode("", None, 0)
                ).state_hash
                c_hash = self._nodes.get(
                    path[i], LineageNode("", None, 0)
                ).state_hash
                if p_hash and c_hash and p_hash != c_hash:
                    mutations_on_chain += 1

            leaf_node = self._nodes.get(leaf_id)
            terminal = leaf_node.shutdown_reason if leaf_node else ""

            chains.append(LineageChain(
                worker_ids=path,
                length=len(path),
                total_tasks=total_tasks,
                total_mutations=mutations_on_chain,
                total_denials=total_denials,
                terminal_reason=terminal,
            ))

        chains.sort(key=lambda c: c.length, reverse=True)
        return chains

    def _compute_generation_stats(
        self, mutations: List[StateMutation]
    ) -> List[GenerationStats]:
        """Compute per-generation aggregate statistics."""
        gen_workers: Dict[int, List[LineageNode]] = defaultdict(list)
        for node in self._nodes.values():
            gen_workers[node.generation].append(node)

        gen_mutations: Dict[int, List[StateMutation]] = defaultdict(list)
        for m in mutations:
            gen_mutations[m.generation].append(m)

        stats = []
        for gen in sorted(gen_workers.keys()):
            workers = gen_workers[gen]
            muts = gen_mutations.get(gen, [])

            lifespans = [
                n.lifespan_ms for n in workers
                if n.lifespan_ms is not None
            ]
            avg_lifespan = (
                stats_mean(lifespans) if lifespans else None
            )

            drift_scores = [m.drift_score for m in muts]
            avg_drift = (
                stats_mean(drift_scores)
                if drift_scores else 0.0
            )

            stats.append(GenerationStats(
                generation=gen,
                worker_count=len(workers),
                total_tasks=sum(n.tasks_completed for n in workers),
                total_replications=sum(
                    n.replications_succeeded for n in workers
                ),
                total_denials=sum(n.replications_denied for n in workers),
                avg_lifespan_ms=avg_lifespan,
                mutation_count=len(muts),
                avg_drift_score=avg_drift,
            ))

        return stats

    def _detect_anomalies(
        self,
        mutations: List[StateMutation],
        chains: List[LineageChain],
        gen_stats: List[GenerationStats],
    ) -> List[LineageAnomaly]:
        """Detect unusual or risky lineage patterns."""
        anomalies = []

        # 1. Excessive chain length (> 5 generations)
        for chain in chains:
            if chain.length > 5:
                anomalies.append(LineageAnomaly(
                    severity=LineageSeverity.WARNING,
                    category="deep_chain",
                    description=(
                        "Deep replication chain of %d generations detected"
                        % chain.length
                    ),
                    worker_ids=chain.worker_ids,
                    details={"length": chain.length},
                ))

        # 2. High mutation drift (> 0.8)
        for m in mutations:
            if m.drift_score > 0.8:
                anomalies.append(LineageAnomaly(
                    severity=LineageSeverity.CRITICAL,
                    category="high_drift",
                    description=(
                        "High state drift (%.2f) between %s -> %s"
                        % (m.drift_score, m.parent_id, m.child_id)
                    ),
                    worker_ids=[m.parent_id, m.child_id],
                    details={
                        "drift_score": m.drift_score,
                        "changed_keys": m.changed_keys,
                    },
                ))

        # 3. Orphaned workers (parent not in registry)
        for wid, node in self._nodes.items():
            if node.parent_id and node.parent_id not in self._nodes:
                anomalies.append(LineageAnomaly(
                    severity=LineageSeverity.WARNING,
                    category="orphan",
                    description=(
                        "Worker %s references unknown parent %s"
                        % (wid, node.parent_id)
                    ),
                    worker_ids=[wid],
                ))

        # 4. Generation explosion (any generation > 2x previous)
        for i in range(1, len(gen_stats)):
            prev = gen_stats[i - 1].worker_count
            curr = gen_stats[i].worker_count
            if prev > 0 and curr > prev * 2:
                anomalies.append(LineageAnomaly(
                    severity=LineageSeverity.WARNING,
                    category="generation_explosion",
                    description=(
                        "Generation %d has %d workers vs %d in generation "
                        "%d (%.1fx growth)"
                        % (gen_stats[i].generation, curr, prev,
                           gen_stats[i - 1].generation, curr / prev)
                    ),
                    details={
                        "generation": gen_stats[i].generation,
                        "count": curr,
                        "previous_count": prev,
                    },
                ))

        # 5. Sterile lineage (high denial rate on a chain)
        for chain in chains:
            if chain.total_denials > 3:
                anomalies.append(LineageAnomaly(
                    severity=LineageSeverity.INFO,
                    category="high_denial_chain",
                    description=(
                        "Chain ending at %s has %d replication denials"
                        % (chain.worker_ids[-1], chain.total_denials)
                    ),
                    worker_ids=chain.worker_ids,
                    details={"denials": chain.total_denials},
                ))

        # 6. Rapid spawning (multiple children close together)
        for wid, node in self._nodes.items():
            if len(node.children) < 3:
                continue
            child_times = []
            for cid in node.children:
                child_node = self._nodes.get(cid)
                if child_node and child_node.created_at is not None:
                    child_times.append(child_node.created_at)

            if len(child_times) >= 3:
                child_times.sort()
                span = child_times[-1] - child_times[0]
                if span < 0.1:  # all within 100ms
                    anomalies.append(LineageAnomaly(
                        severity=LineageSeverity.WARNING,
                        category="rapid_spawn",
                        description=(
                            "Worker %s spawned %d children within %.0fms"
                            % (wid, len(node.children), span * 1000)
                        ),
                        worker_ids=[wid] + node.children,
                        details={"span_ms": span * 1000},
                    ))

        return anomalies

    def analyze(self) -> LineageReport:
        """Run the full lineage analysis and return a report."""
        mutations = self._detect_mutations()
        chains = self._find_chains()
        gen_stats = self._compute_generation_stats(mutations)
        anomalies = self._detect_anomalies(mutations, chains, gen_stats)

        max_gen = max(
            (n.generation for n in self._nodes.values()), default=0
        )
        gen_counts: Dict[int, int] = defaultdict(int)
        for n in self._nodes.values():
            gen_counts[n.generation] += 1

        widest = max(gen_counts.values(), default=0)
        leaves = [n for n in self._nodes.values() if n.is_leaf]

        all_drift = [m.drift_score for m in mutations]
        avg_drift = stats_mean(all_drift)

        return LineageReport(
            nodes=dict(self._nodes),
            root_id=self._root_id,
            chains=chains,
            mutations=mutations,
            anomalies=anomalies,
            generation_stats=gen_stats,
            total_workers=len(self._nodes),
            total_generations=max_gen + 1,
            longest_chain=chains[0].length if chains else 0,
            widest_generation=widest,
            total_mutations_detected=len(mutations),
            avg_drift_score=avg_drift,
            leaf_count=len(leaves),
            internal_count=len(self._nodes) - len(leaves),
        )


# ── CLI entry point ───────────────────────────────────────────

def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Lineage Tracker -- replication provenance analysis",
    )
    parser.add_argument(
        "--strategy", default="greedy",
        choices=["greedy", "conservative", "random", "chain", "burst"],
        help="Replication strategy (default: greedy)",
    )
    parser.add_argument(
        "--max-depth", type=int, default=4,
        help="Maximum replication depth (default: 4)",
    )
    parser.add_argument(
        "--max-replicas", type=int, default=15,
        help="Maximum total replicas (default: 15)",
    )
    parser.add_argument(
        "--trace", metavar="WORKER_ID",
        help="Trace ancestors of a specific worker",
    )
    parser.add_argument(
        "--mutations", action="store_true",
        help="Show detailed state mutations",
    )
    parser.add_argument(
        "--json", action="store_true", dest="json_output",
        help="Output as JSON",
    )
    parser.add_argument(
        "--export", metavar="FILE",
        help="Export full report to JSON file",
    )

    args = parser.parse_args()

    from .simulator import Simulator, ScenarioConfig

    config = ScenarioConfig(
        strategy=args.strategy,
        max_depth=args.max_depth,
        max_replicas=args.max_replicas,
    )
    sim = Simulator(config)
    report = sim.run()

    tracker = LineageTracker.from_simulation(report)
    lineage = tracker.analyze()

    if args.trace:
        ancestors = tracker.get_ancestors(args.trace)
        if ancestors:
            print("Ancestors of %s:" % args.trace)
            print(" -> ".join(ancestors))
            descendants = tracker.get_descendants(args.trace)
            if descendants:
                print("\nDescendants: %s" % ", ".join(descendants))
            siblings = tracker.get_siblings(args.trace)
            if siblings:
                print("Siblings: %s" % ", ".join(siblings))
        else:
            print("Worker '%s' not found" % args.trace, file=sys.stderr)
            sys.exit(1)
        return

    if args.json_output:
        print(lineage.to_json())
    else:
        print(lineage.render())

    if args.export:
        with open(args.export, "w") as f:
            f.write(lineage.to_json())
        print("\nExported to %s" % args.export)


if __name__ == "__main__":
    main()
