"""Resource Dependency Analyzer — model and analyze inter-agent resource
dependencies for failure cascade prediction.

In multi-agent replication systems, agents rely on shared resources:
APIs, databases, file systems, compute pools, message queues, and
external services.  When one resource degrades or fails, dependent
agents can cascade-fail, potentially bringing down entire subsystems.

This module builds a dependency graph where *agents* depend on
*resources*, and *resources* may depend on other resources.  It then
analyzes the graph for:

- **Single points of failure (SPOFs):** resources whose failure would
  take down multiple agents with no fallback.
- **Circular dependencies:** resource A depends on B depends on A,
  creating deadlock risk under partial failure.
- **Blast radius:** how many agents (direct + transitive) are affected
  when a specific resource goes down.
- **Critical path:** the longest dependency chain, indicating where
  latency and failure propagation are worst.
- **Redundancy score:** how well-covered each agent is (multiple
  independent paths to critical resources).
- **Failure cascade simulation:** simulate a resource outage and
  trace which agents lose functionality over time.

Usage (CLI)::

    python -m replication.dependency_graph                  # analyze sample system
    python -m replication.dependency_graph --agents 10      # 10 agents
    python -m replication.dependency_graph --resources 8    # 8 resources
    python -m replication.dependency_graph --scenario dense # dense dependencies
    python -m replication.dependency_graph --fail db-primary # simulate failure
    python -m replication.dependency_graph --fail api-gateway --cascade  # cascade mode
    python -m replication.dependency_graph --spof           # show SPOFs only
    python -m replication.dependency_graph --dot            # Graphviz DOT output
    python -m replication.dependency_graph --json           # JSON output
    python -m replication.dependency_graph --seed 42        # reproducible

Programmatic::

    from replication.dependency_graph import (
        DependencyGraph, Resource, AgentNode, DepAnalysis,
    )

    graph = DependencyGraph()
    graph.add_resource("db-primary", kind="database", redundancy_group="db")
    graph.add_resource("db-replica", kind="database", redundancy_group="db")
    graph.add_resource("api-gateway", kind="service")
    graph.add_agent("agent-1", depends_on=["db-primary", "api-gateway"])
    graph.add_agent("agent-2", depends_on=["db-replica", "api-gateway"])

    analysis = graph.analyze()
    print(analysis.render())
    print(f"SPOFs: {analysis.spofs}")
    print(f"Blast radius of api-gateway: {analysis.blast_radius('api-gateway')}")

    # Simulate cascade failure
    cascade = graph.simulate_failure("api-gateway")
    for step in cascade.steps:
        print(f"  t={step.time}: {step.failed} goes down ({step.reason})")
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from ._helpers import box_header, stats_mean


# ── Data types ───────────────────────────────────────────────────────


class ResourceKind(str, Enum):
    """Classification of infrastructure resources."""
    DATABASE = "database"
    SERVICE = "service"
    QUEUE = "queue"
    FILESYSTEM = "filesystem"
    COMPUTE = "compute"
    EXTERNAL = "external"
    CACHE = "cache"


class Criticality(str, Enum):
    """How critical a dependency is for the dependent."""
    REQUIRED = "required"      # agent cannot function without this
    DEGRADED = "degraded"      # agent works with reduced functionality
    OPTIONAL = "optional"      # nice-to-have, non-essential


@dataclass
class Resource:
    """A shared resource in the system."""
    name: str
    kind: ResourceKind = ResourceKind.SERVICE
    redundancy_group: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    capacity: float = 1.0  # abstract capacity units
    failure_probability: float = 0.05  # per-interval failure prob

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "kind": self.kind.value,
            "capacity": self.capacity,
            "failure_probability": self.failure_probability,
        }
        if self.redundancy_group:
            d["redundancy_group"] = self.redundancy_group
        if self.depends_on:
            d["depends_on"] = list(self.depends_on)
        return d


@dataclass
class AgentNode:
    """An agent that depends on resources."""
    name: str
    depends_on: List[Tuple[str, Criticality]] = field(default_factory=list)
    priority: int = 1  # higher = more important

    def required_resources(self) -> List[str]:
        return [r for r, c in self.depends_on if c == Criticality.REQUIRED]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "priority": self.priority,
            "dependencies": [
                {"resource": r, "criticality": c.value}
                for r, c in self.depends_on
            ],
        }


# ── Cascade simulation ──────────────────────────────────────────────


@dataclass
class CascadeStep:
    """One step in a failure cascade."""
    time: int
    failed: str  # resource or agent name
    reason: str
    kind: str  # "resource" or "agent"
    affected_agents: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time": self.time,
            "failed": self.failed,
            "reason": self.reason,
            "kind": self.kind,
            "affected_agents": self.affected_agents,
        }


@dataclass
class CascadeResult:
    """Full cascade simulation result."""
    trigger: str
    steps: List[CascadeStep] = field(default_factory=list)
    total_agents_down: int = 0
    total_resources_down: int = 0
    cascade_depth: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trigger": self.trigger,
            "steps": [s.to_dict() for s in self.steps],
            "total_agents_down": self.total_agents_down,
            "total_resources_down": self.total_resources_down,
            "cascade_depth": self.cascade_depth,
        }

    def render(self) -> str:
        lines: List[str] = []
        lines.append(f"\n  Cascade Failure Simulation: {self.trigger}")
        lines.append("  " + "-" * 50)
        if not self.steps:
            lines.append("  No cascade propagation (resource has no dependents)")
            return "\n".join(lines)
        for step in self.steps:
            icon = "\u274c" if step.kind == "resource" else "\u26a0\ufe0f"
            lines.append(
                f"  t={step.time:>2}  {icon} {step.failed:20s}  {step.reason}"
            )
        lines.append(f"\n  Total resources down: {self.total_resources_down}")
        lines.append(f"  Total agents down:    {self.total_agents_down}")
        lines.append(f"  Cascade depth:        {self.cascade_depth}")
        return "\n".join(lines)


# ── Analysis result ──────────────────────────────────────────────────


@dataclass
class DepAnalysis:
    """Complete dependency analysis result."""
    total_agents: int = 0
    total_resources: int = 0
    total_edges: int = 0
    spofs: List[Dict[str, Any]] = field(default_factory=list)
    cycles: List[List[str]] = field(default_factory=list)
    blast_radii: Dict[str, int] = field(default_factory=dict)
    critical_path: List[str] = field(default_factory=list)
    critical_path_length: int = 0
    redundancy_scores: Dict[str, float] = field(default_factory=dict)
    mean_redundancy: float = 0.0
    risk_level: str = "low"
    warnings: List[str] = field(default_factory=list)
    resource_details: List[Dict[str, Any]] = field(default_factory=list)
    agent_details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_agents": self.total_agents,
            "total_resources": self.total_resources,
            "total_edges": self.total_edges,
            "spofs": self.spofs,
            "cycles": self.cycles,
            "blast_radii": self.blast_radii,
            "critical_path": self.critical_path,
            "critical_path_length": self.critical_path_length,
            "redundancy_scores": self.redundancy_scores,
            "mean_redundancy": round(self.mean_redundancy, 3),
            "risk_level": self.risk_level,
            "warnings": self.warnings,
        }

    def render(self) -> str:
        lines: List[str] = []

        # Header
        lines.extend(box_header("Resource Dependency Analysis"))
        lines.append("")

        # Overview
        lines.append(f"  Agents:       {self.total_agents}")
        lines.append(f"  Resources:    {self.total_resources}")
        lines.append(f"  Dependencies: {self.total_edges}")
        lines.append(f"  Risk Level:   {self.risk_level.upper()}")
        lines.append("")

        # SPOFs
        if self.spofs:
            lines.append("  \u26a0\ufe0f  Single Points of Failure")
            lines.append("  " + "-" * 40)
            for spof in self.spofs:
                lines.append(
                    f"    {spof['resource']:20s}  "
                    f"affects {spof['affected_agents']} agents  "
                    f"(blast radius: {spof['blast_radius']})"
                )
            lines.append("")

        # Circular dependencies
        if self.cycles:
            lines.append("  \U0001f504 Circular Dependencies")
            lines.append("  " + "-" * 40)
            for cycle in self.cycles:
                lines.append("    " + " -> ".join(cycle))
            lines.append("")

        # Blast radii (top 5)
        if self.blast_radii:
            sorted_br = sorted(
                self.blast_radii.items(), key=lambda x: x[1], reverse=True
            )[:5]
            lines.append("  \U0001f4a5 Largest Blast Radii")
            lines.append("  " + "-" * 40)
            for name, radius in sorted_br:
                bar = "\u2588" * min(radius, 30)
                lines.append(f"    {name:20s}  {radius:>3}  {bar}")
            lines.append("")

        # Critical path
        if self.critical_path:
            lines.append(
                f"  \U0001f6e4\ufe0f  Critical Path (length {self.critical_path_length})"
            )
            lines.append("    " + " -> ".join(self.critical_path))
            lines.append("")

        # Redundancy
        lines.append(
            f"  \U0001f6e1\ufe0f  Mean Redundancy Score: "
            f"{self.mean_redundancy:.2f} / 1.00"
        )
        if self.redundancy_scores:
            worst = sorted(
                self.redundancy_scores.items(), key=lambda x: x[1]
            )[:3]
            if worst and worst[0][1] < 0.5:
                lines.append("    Least redundant agents:")
                for name, score in worst:
                    lines.append(f"      {name:20s}  {score:.2f}")
        lines.append("")

        # Warnings
        if self.warnings:
            lines.append("  \u26a0\ufe0f  Warnings")
            for w in self.warnings:
                lines.append(f"    - {w}")
            lines.append("")

        return "\n".join(lines)


# ── Main graph class ─────────────────────────────────────────────────


class DependencyGraph:
    """Directed dependency graph between agents and resources.

    Nodes are either agents or resources.  Edges point from dependent
    to dependency (agent -> resource, resource -> resource).
    """

    def __init__(self) -> None:
        self._resources: Dict[str, Resource] = {}
        self._agents: Dict[str, AgentNode] = {}
        self._redundancy_groups: Dict[str, Set[str]] = defaultdict(set)

    # ── construction ─────────────────────────────────────────────

    def add_resource(
        self,
        name: str,
        kind: str = "service",
        redundancy_group: Optional[str] = None,
        depends_on: Optional[List[str]] = None,
        capacity: float = 1.0,
        failure_probability: float = 0.05,
    ) -> "DependencyGraph":
        """Add a resource node.  Returns self for chaining."""
        rk = ResourceKind(kind) if kind in [e.value for e in ResourceKind] else ResourceKind.SERVICE
        res = Resource(
            name=name,
            kind=rk,
            redundancy_group=redundancy_group,
            depends_on=list(depends_on or []),
            capacity=capacity,
            failure_probability=failure_probability,
        )
        self._resources[name] = res
        if redundancy_group:
            self._redundancy_groups[redundancy_group].add(name)
        return self

    def add_agent(
        self,
        name: str,
        depends_on: Optional[List[str]] = None,
        priority: int = 1,
        criticalities: Optional[Dict[str, str]] = None,
    ) -> "DependencyGraph":
        """Add an agent node with its resource dependencies.

        Parameters
        ----------
        depends_on : list of resource names
        criticalities : optional mapping of resource name -> criticality
            (``required``, ``degraded``, ``optional``).  Defaults to
            ``required`` for any resource not in the map.
        """
        crit_map = criticalities or {}
        deps: List[Tuple[str, Criticality]] = []
        for r in (depends_on or []):
            c_str = crit_map.get(r, "required")
            try:
                c = Criticality(c_str)
            except ValueError:
                c = Criticality.REQUIRED
            deps.append((r, c))
        agent = AgentNode(name=name, depends_on=deps, priority=priority)
        self._agents[name] = agent
        return self

    # ── analysis ─────────────────────────────────────────────────

    def analyze(self) -> DepAnalysis:
        """Run full dependency analysis."""
        result = DepAnalysis(
            total_agents=len(self._agents),
            total_resources=len(self._resources),
        )

        # Count edges
        edge_count = 0
        for agent in self._agents.values():
            edge_count += len(agent.depends_on)
        for res in self._resources.values():
            edge_count += len(res.depends_on)
        result.total_edges = edge_count

        # Blast radii for every resource
        for rname in self._resources:
            result.blast_radii[rname] = self._compute_blast_radius(rname)

        # SPOF detection
        result.spofs = self._find_spofs()

        # Cycle detection (among resources)
        result.cycles = self._find_cycles()

        # Critical path
        path, length = self._find_critical_path()
        result.critical_path = path
        result.critical_path_length = length

        # Redundancy scoring per agent
        for agent in self._agents.values():
            result.redundancy_scores[agent.name] = self._redundancy_score(agent)
        if result.redundancy_scores:
            result.mean_redundancy = stats_mean(
                list(result.redundancy_scores.values())
            )

        # Resource details
        for res in self._resources.values():
            detail = res.to_dict()
            detail["blast_radius"] = result.blast_radii.get(res.name, 0)
            detail["is_spof"] = any(
                s["resource"] == res.name for s in result.spofs
            )
            result.resource_details.append(detail)

        # Agent details
        for agent in self._agents.values():
            detail = agent.to_dict()
            detail["redundancy_score"] = round(
                result.redundancy_scores.get(agent.name, 0.0), 3
            )
            result.agent_details.append(detail)

        # Risk assessment
        result.risk_level = self._assess_risk(result)

        # Warnings
        result.warnings = self._generate_warnings(result)

        return result

    def _compute_blast_radius(self, resource_name: str) -> int:
        """Count total agents affected if *resource_name* fails.

        Considers transitive resource dependencies and redundancy groups.
        An agent is considered down only if ALL members of a redundancy
        group it depends on are down.
        """
        # Find all resources that transitively depend on this one
        failed_resources = self._transitive_resource_failures(resource_name)

        affected = 0
        for agent in self._agents.values():
            if self._agent_affected(agent, failed_resources):
                affected += 1
        return affected

    def _transitive_resource_failures(self, trigger: str) -> Set[str]:
        """Find all resources that fail when *trigger* fails."""
        failed: Set[str] = {trigger}
        queue: deque[str] = deque([trigger])

        while queue:
            current = queue.popleft()
            # Find resources that depend on current
            for rname, res in self._resources.items():
                if rname in failed:
                    continue
                if current in res.depends_on:
                    # Check if this resource has redundancy for *current*
                    group = self._resources[current].redundancy_group if current in self._resources else None
                    if group:
                        group_members = self._redundancy_groups[group]
                        survivors = group_members - failed
                        if survivors:
                            continue  # redundancy saves this dependency
                    failed.add(rname)
                    queue.append(rname)

        return failed

    def _agent_affected(
        self, agent: AgentNode, failed_resources: Set[str]
    ) -> bool:
        """Check if agent loses required functionality."""
        for rname, crit in agent.depends_on:
            if crit != Criticality.REQUIRED:
                continue
            if rname in failed_resources:
                # Check redundancy group
                res = self._resources.get(rname)
                if res and res.redundancy_group:
                    group = self._redundancy_groups[res.redundancy_group]
                    survivors = group - failed_resources
                    if survivors:
                        continue  # redundancy covers this
                return True  # required resource is down with no fallback
        return False

    def _find_spofs(self) -> List[Dict[str, Any]]:
        """Find resources that are single points of failure.

        A resource is a SPOF if its failure alone (considering redundancy)
        would take down at least one agent.
        """
        spofs: List[Dict[str, Any]] = []
        for rname in self._resources:
            failed = self._transitive_resource_failures(rname)
            affected = sum(
                1 for a in self._agents.values()
                if self._agent_affected(a, failed)
            )
            if affected > 0:
                # Verify no redundancy covers it
                res = self._resources[rname]
                if res.redundancy_group:
                    group = self._redundancy_groups[res.redundancy_group]
                    if len(group) > 1:
                        continue  # redundancy exists, not a true SPOF
                spofs.append({
                    "resource": rname,
                    "kind": res.kind.value,
                    "affected_agents": affected,
                    "blast_radius": len(failed),
                })
        return sorted(spofs, key=lambda s: s["affected_agents"], reverse=True)

    def _find_cycles(self) -> List[List[str]]:
        """Detect cycles in resource-to-resource dependencies using DFS."""
        cycles: List[List[str]] = []
        visited: Set[str] = set()
        path: List[str] = []
        on_path: Set[str] = set()

        def dfs(node: str) -> None:
            if node in on_path:
                # Found a cycle
                start = path.index(node)
                cycle = path[start:] + [node]
                # Normalize: start from the smallest element
                min_idx = cycle[:-1].index(min(cycle[:-1]))
                normalized = cycle[min_idx:-1] + cycle[:min_idx] + [cycle[min_idx]]
                if normalized not in cycles:
                    cycles.append(normalized)
                return
            if node in visited:
                return
            visited.add(node)
            on_path.add(node)
            path.append(node)
            res = self._resources.get(node)
            if res:
                for dep in res.depends_on:
                    if dep in self._resources:
                        dfs(dep)
            path.pop()
            on_path.remove(node)

        for rname in self._resources:
            if rname not in visited:
                dfs(rname)

        return cycles

    def _find_critical_path(self) -> Tuple[List[str], int]:
        """Find the longest dependency chain (agents -> resources -> resources).

        Uses BFS from each agent, following dependency edges.
        """
        longest: List[str] = []

        for agent in self._agents.values():
            for rname, _ in agent.depends_on:
                # BFS/DFS from this resource
                path = self._longest_chain(rname, set())
                full_path = [agent.name] + path
                if len(full_path) > len(longest):
                    longest = full_path

        return longest, max(len(longest) - 1, 0)

    def _longest_chain(self, resource: str, visited: Set[str]) -> List[str]:
        """Recursively find longest chain from a resource."""
        if resource in visited or resource not in self._resources:
            return [resource] if resource in self._resources else []

        visited.add(resource)
        res = self._resources[resource]
        best: List[str] = [resource]

        for dep in res.depends_on:
            chain = self._longest_chain(dep, visited)
            candidate = [resource] + chain
            if len(candidate) > len(best):
                best = candidate

        visited.discard(resource)
        return best

    def _redundancy_score(self, agent: AgentNode) -> float:
        """Score 0..1 indicating how well-covered an agent's dependencies are.

        1.0 = all required resources have redundancy groups with 2+ members.
        0.0 = all required resources are singleton SPOFs.
        """
        required = [r for r, c in agent.depends_on if c == Criticality.REQUIRED]
        if not required:
            return 1.0  # no required deps = fully resilient

        covered = 0
        for rname in required:
            res = self._resources.get(rname)
            if res and res.redundancy_group:
                group_size = len(self._redundancy_groups[res.redundancy_group])
                if group_size >= 2:
                    covered += 1
        return covered / len(required)

    def _assess_risk(self, analysis: DepAnalysis) -> str:
        """Determine overall risk level."""
        score = 0

        # SPOFs
        if len(analysis.spofs) >= 3:
            score += 3
        elif analysis.spofs:
            score += 1

        # Cycles
        if analysis.cycles:
            score += 2

        # Low redundancy
        if analysis.mean_redundancy < 0.3:
            score += 2
        elif analysis.mean_redundancy < 0.6:
            score += 1

        # Long critical paths
        if analysis.critical_path_length >= 5:
            score += 2
        elif analysis.critical_path_length >= 3:
            score += 1

        # High blast radii
        max_blast = max(analysis.blast_radii.values()) if analysis.blast_radii else 0
        if max_blast >= analysis.total_agents * 0.8:
            score += 2
        elif max_blast >= analysis.total_agents * 0.5:
            score += 1

        if score >= 7:
            return "critical"
        elif score >= 4:
            return "high"
        elif score >= 2:
            return "moderate"
        return "low"

    def _generate_warnings(self, analysis: DepAnalysis) -> List[str]:
        """Generate human-readable warnings."""
        warnings: List[str] = []

        if analysis.spofs:
            warnings.append(
                f"{len(analysis.spofs)} single point(s) of failure detected"
            )

        if analysis.cycles:
            warnings.append(
                f"{len(analysis.cycles)} circular dependency chain(s) found "
                "(deadlock risk under partial failure)"
            )

        if analysis.mean_redundancy < 0.3:
            warnings.append(
                f"Very low mean redundancy ({analysis.mean_redundancy:.2f}) "
                "— most agents lack fallback resources"
            )

        lonely_agents = [
            a.name for a in self._agents.values()
            if len(a.depends_on) <= 1
        ]
        if lonely_agents:
            warnings.append(
                f"{len(lonely_agents)} agent(s) have 1 or fewer dependencies "
                "(tightly coupled to a single resource)"
            )

        # Resources with no agents depending on them
        used = set()
        for agent in self._agents.values():
            for rname, _ in agent.depends_on:
                used.add(rname)
        for res in self._resources.values():
            for dep in res.depends_on:
                used.add(dep)
        orphans = [r for r in self._resources if r not in used]
        if orphans:
            warnings.append(
                f"{len(orphans)} resource(s) have no dependents (may be unused)"
            )

        return warnings

    # ── cascade simulation ───────────────────────────────────────

    def simulate_failure(self, resource_name: str) -> CascadeResult:
        """Simulate cascading failure when *resource_name* goes down.

        Returns a time-ordered sequence of failures as they propagate
        through the dependency graph.
        """
        if resource_name not in self._resources:
            return CascadeResult(
                trigger=resource_name,
                steps=[],
                total_agents_down=0,
                total_resources_down=0,
            )

        result = CascadeResult(trigger=resource_name)
        failed_resources: Set[str] = {resource_name}
        failed_agents: Set[str] = set()

        result.steps.append(CascadeStep(
            time=0,
            failed=resource_name,
            reason="Initial failure (trigger)",
            kind="resource",
        ))

        t = 1
        changed = True
        while changed:
            changed = False

            # Check for resource cascades
            for rname, res in self._resources.items():
                if rname in failed_resources:
                    continue
                # Resource fails if any non-redundant dependency is down
                for dep in res.depends_on:
                    if dep not in failed_resources:
                        continue
                    # Check redundancy
                    dep_res = self._resources.get(dep)
                    if dep_res and dep_res.redundancy_group:
                        group = self._redundancy_groups[dep_res.redundancy_group]
                        if group - failed_resources:
                            continue  # redundancy covers it
                    failed_resources.add(rname)
                    result.steps.append(CascadeStep(
                        time=t,
                        failed=rname,
                        reason=f"Dependency {dep} is down",
                        kind="resource",
                    ))
                    changed = True
                    break

            # Check for agent failures
            for agent in self._agents.values():
                if agent.name in failed_agents:
                    continue
                if self._agent_affected(agent, failed_resources):
                    failed_agents.add(agent.name)
                    # Find which required resource caused the failure
                    cause = "unknown"
                    for rname, crit in agent.depends_on:
                        if crit == Criticality.REQUIRED and rname in failed_resources:
                            cause = f"Required resource {rname} unavailable"
                            break
                    result.steps.append(CascadeStep(
                        time=t,
                        failed=agent.name,
                        reason=cause,
                        kind="agent",
                        affected_agents=1,
                    ))
                    changed = True

            if changed:
                t += 1

        result.total_resources_down = len(failed_resources)
        result.total_agents_down = len(failed_agents)
        result.cascade_depth = t - 1

        return result

    # ── DOT export ───────────────────────────────────────────────

    def to_dot(self) -> str:
        """Export dependency graph as Graphviz DOT."""
        lines: List[str] = ['digraph DependencyGraph {']
        lines.append('  rankdir=LR;')
        lines.append('  node [fontname="Helvetica"];')
        lines.append('')

        # Resource nodes
        kind_shapes = {
            ResourceKind.DATABASE: "cylinder",
            ResourceKind.SERVICE: "box",
            ResourceKind.QUEUE: "parallelogram",
            ResourceKind.FILESYSTEM: "folder",
            ResourceKind.COMPUTE: "box3d",
            ResourceKind.EXTERNAL: "diamond",
            ResourceKind.CACHE: "hexagon",
        }
        # Pre-compute SPOFs once instead of calling _find_spofs() per resource
        spof_names: FrozenSet[str] = frozenset(
            s["resource"] for s in self._find_spofs()
        )
        for res in self._resources.values():
            shape = kind_shapes.get(res.kind, "box")
            color = "red" if res.name in spof_names else "black"
            lines.append(
                f'  "{res.name}" [shape={shape}, color={color}, '
                f'label="{res.name}\\n({res.kind.value})"];'
            )

        # Agent nodes
        for agent in self._agents.values():
            lines.append(
                f'  "{agent.name}" [shape=ellipse, style=filled, '
                f'fillcolor=lightblue, label="{agent.name}\\n(agent)"];'
            )

        lines.append('')

        # Edges
        for agent in self._agents.values():
            for rname, crit in agent.depends_on:
                style = "solid" if crit == Criticality.REQUIRED else "dashed"
                lines.append(
                    f'  "{agent.name}" -> "{rname}" [style={style}, '
                    f'label="{crit.value}"];'
                )

        for res in self._resources.values():
            for dep in res.depends_on:
                lines.append(f'  "{res.name}" -> "{dep}" [color=gray];')

        lines.append('}')
        return "\n".join(lines)

    # ── JSON export ──────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resources": [r.to_dict() for r in self._resources.values()],
            "agents": [a.to_dict() for a in self._agents.values()],
            "redundancy_groups": {
                k: sorted(v) for k, v in self._redundancy_groups.items()
            },
        }


# ── Scenario generators ─────────────────────────────────────────────


def generate_scenario(
    num_agents: int = 6,
    num_resources: int = 8,
    scenario: str = "realistic",
    seed: Optional[int] = None,
) -> DependencyGraph:
    """Generate a sample dependency graph for analysis.

    Scenarios
    ---------
    - ``realistic`` — mixed redundancy, some SPOFs, moderate coupling
    - ``dense`` — heavy interconnection, many shared resources
    - ``sparse`` — minimal dependencies, isolated agents
    - ``fragile`` — many SPOFs, no redundancy, long chains
    """
    rng = random.Random(seed)
    graph = DependencyGraph()

    kinds = list(ResourceKind)

    if scenario == "fragile":
        # Chain of resources with no redundancy
        res_names = [f"res-{i}" for i in range(num_resources)]
        for i, rname in enumerate(res_names):
            deps = [res_names[i - 1]] if i > 0 else []
            graph.add_resource(
                rname,
                kind=rng.choice(kinds).value,
                depends_on=deps,
                failure_probability=0.1,
            )
        for i in range(num_agents):
            # Each agent depends on 1-2 resources (no redundancy)
            dep_count = rng.randint(1, min(2, num_resources))
            deps = rng.sample(res_names, dep_count)
            graph.add_agent(f"agent-{i}", depends_on=deps, priority=rng.randint(1, 3))

    elif scenario == "dense":
        res_names = [f"res-{i}" for i in range(num_resources)]
        # Create some redundancy groups
        groups = {}
        for i, rname in enumerate(res_names):
            grp = f"group-{i // 2}" if i < num_resources - 1 else None
            if grp:
                groups[rname] = grp
            inter_deps = []
            if i >= 2 and rng.random() < 0.4:
                inter_deps.append(rng.choice(res_names[:i]))
            graph.add_resource(
                rname,
                kind=rng.choice(kinds).value,
                redundancy_group=grp,
                depends_on=inter_deps,
            )
        for i in range(num_agents):
            dep_count = rng.randint(2, min(5, num_resources))
            deps = rng.sample(res_names, dep_count)
            crits = {}
            for d in deps:
                crits[d] = rng.choice(["required", "required", "degraded", "optional"])
            graph.add_agent(
                f"agent-{i}", depends_on=deps, priority=rng.randint(1, 5),
                criticalities=crits,
            )

    elif scenario == "sparse":
        res_names = [f"res-{i}" for i in range(num_resources)]
        for rname in res_names:
            graph.add_resource(rname, kind=rng.choice(kinds).value)
        for i in range(num_agents):
            dep = rng.choice(res_names)
            graph.add_agent(f"agent-{i}", depends_on=[dep])

    else:  # realistic
        res_names = []
        infra = [
            ("db-primary", "database", "db"),
            ("db-replica", "database", "db"),
            ("cache-1", "cache", "cache"),
            ("cache-2", "cache", "cache"),
            ("api-gateway", "service", None),
            ("msg-queue", "queue", None),
            ("blob-store", "filesystem", None),
            ("auth-service", "service", None),
        ]
        for name, kind, grp in infra[:num_resources]:
            deps = []
            if kind == "service" and name != "auth-service":
                deps.append("auth-service")
            graph.add_resource(name, kind=kind, redundancy_group=grp, depends_on=deps)
            res_names.append(name)

        # Add remaining if num_resources > 8
        for i in range(len(infra), num_resources):
            rname = f"service-{i}"
            graph.add_resource(rname, kind="service", depends_on=["auth-service"])
            res_names.append(rname)

        for i in range(num_agents):
            # Each agent needs auth + some data layer + maybe cache
            deps = ["auth-service"]
            crits: Dict[str, str] = {"auth-service": "required"}
            # Pick a data resource
            data_res = [r for r in res_names if r.startswith("db-") or r == "blob-store"]
            if data_res:
                d = rng.choice(data_res)
                deps.append(d)
                crits[d] = "required"
            # Maybe cache
            cache_res = [r for r in res_names if r.startswith("cache-")]
            if cache_res and rng.random() < 0.6:
                c = rng.choice(cache_res)
                deps.append(c)
                crits[c] = "degraded"
            # Maybe queue
            if "msg-queue" in res_names and rng.random() < 0.4:
                deps.append("msg-queue")
                crits["msg-queue"] = "optional"

            graph.add_agent(
                f"agent-{i}", depends_on=deps, priority=rng.randint(1, 3),
                criticalities=crits,
            )

    return graph


# ── CLI ──────────────────────────────────────────────────────────────


def _main() -> None:
    """CLI entry point (called from __main__.py dispatcher)."""
    parser = argparse.ArgumentParser(
        description="Resource Dependency Analyzer — model inter-agent "
        "resource dependencies, detect SPOFs, and simulate failure cascades.",
    )
    parser.add_argument(
        "--agents", type=int, default=6,
        help="Number of agents to generate (default: 6)",
    )
    parser.add_argument(
        "--resources", type=int, default=8,
        help="Number of resources to generate (default: 8)",
    )
    parser.add_argument(
        "--scenario", choices=["realistic", "dense", "sparse", "fragile"],
        default="realistic",
        help="Dependency scenario (default: realistic)",
    )
    parser.add_argument(
        "--fail", type=str, default=None,
        help="Simulate failure of a specific resource (name)",
    )
    parser.add_argument(
        "--cascade", action="store_true",
        help="Show cascade simulation (requires --fail)",
    )
    parser.add_argument(
        "--spof", action="store_true",
        help="Show only single points of failure",
    )
    parser.add_argument(
        "--dot", action="store_true",
        help="Output Graphviz DOT format",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    graph = generate_scenario(
        num_agents=args.agents,
        num_resources=args.resources,
        scenario=args.scenario,
        seed=args.seed,
    )

    if args.dot:
        print(graph.to_dot())
        return

    if args.fail:
        cascade = graph.simulate_failure(args.fail)
        if args.json:
            print(json.dumps(cascade.to_dict(), indent=2))
        else:
            print(cascade.render())
        return

    analysis = graph.analyze()

    if args.json:
        print(json.dumps(analysis.to_dict(), indent=2, default=str))
    elif args.spof:
        if analysis.spofs:
            print("\nSingle Points of Failure:")
            print("-" * 50)
            for spof in analysis.spofs:
                print(
                    f"  {spof['resource']:20s}  "
                    f"type={spof['kind']:10s}  "
                    f"affects {spof['affected_agents']} agents  "
                    f"blast_radius={spof['blast_radius']}"
                )
        else:
            print("\nNo single points of failure detected.")
    else:
        print(analysis.render())


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
