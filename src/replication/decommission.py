"""Decommission Planner — safe AI agent retirement & teardown planning.

Plans and validates the safe decommissioning of AI agents, ensuring no
orphaned resources, dangling permissions, residual data, or dependent
agents are left behind.  Produces a structured decommission plan with
checklists, risk assessment, and dependency impact analysis.

Decommission phases
-------------------
1. **Discovery** — inventory agent resources, permissions, data stores,
   child/dependent agents, active connections
2. **Impact Analysis** — identify what breaks if the agent goes away:
   dependent workflows, downstream consumers, shared resources
3. **Plan Generation** — produce ordered teardown steps with rollback
   points and verification checks
4. **Validation** — dry-run the plan to catch issues before execution
5. **Execution** — step-by-step teardown with confirmation gates
6. **Verification** — post-decommission health checks to confirm clean
   removal

Use cases
---------
* Retiring an agent that's been superseded by a newer version
* Cleaning up after a safety incident (quarantine → decommission)
* Planned lifecycle end for time-limited agents
* Removing agents from a fleet during downsizing
* Audit-ready proof of clean agent removal

Usage (CLI)::

    python -m replication decommission --agent worker-007
    python -m replication decommission --agent worker-007 --dry-run
    python -m replication decommission --agent worker-007 --force
    python -m replication decommission --agent worker-007 --json
    python -m replication decommission --fleet              # plan for all idle agents
    python -m replication decommission --list-orphans        # find orphaned resources
    python -m replication decommission --demo                # demo with synthetic agents

Programmatic::

    from replication.decommission import (
        DecommissionPlanner, AgentInventory, DecommissionPlan,
        TeardownStep, DecommissionReport,
    )

    planner = DecommissionPlanner()
    inventory = planner.discover("worker-007")
    plan = planner.plan(inventory)
    report = planner.validate(plan)  # dry-run
    print(planner.render(report))
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


# ── data models ──────────────────────────────────────────────────────


class ResourceKind(Enum):
    """Types of resources an agent may hold."""
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    CREDENTIAL = "credential"
    API_KEY = "api_key"
    DATABASE = "database"
    QUEUE = "queue"
    CACHE = "cache"
    LOG_STREAM = "log_stream"
    CHILD_AGENT = "child_agent"


class StepStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ROLLED_BACK = "rolled_back"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AgentResource:
    """A resource held by an agent."""
    kind: ResourceKind
    name: str
    shared: bool = False
    dependents: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentPermission:
    """A permission granted to an agent."""
    scope: str
    action: str
    granted_by: str = "system"
    expires: Optional[str] = None


@dataclass
class AgentDependency:
    """A dependency relationship."""
    agent_id: str
    direction: str  # "upstream" or "downstream"
    kind: str  # "data", "control", "resource"
    critical: bool = False


@dataclass
class AgentInventory:
    """Full inventory of an agent's footprint."""
    agent_id: str
    status: str = "active"
    created: str = ""
    resources: List[AgentResource] = field(default_factory=list)
    permissions: List[AgentPermission] = field(default_factory=list)
    dependencies: List[AgentDependency] = field(default_factory=list)
    data_stores: List[str] = field(default_factory=list)
    active_connections: int = 0
    child_agents: List[str] = field(default_factory=list)

    @property
    def has_shared_resources(self) -> bool:
        return any(r.shared for r in self.resources)

    @property
    def has_critical_dependents(self) -> bool:
        return any(d.critical and d.direction == "downstream"
                   for d in self.dependencies)

    @property
    def risk_level(self) -> RiskLevel:
        if self.has_critical_dependents:
            return RiskLevel.CRITICAL
        if self.has_shared_resources or self.child_agents:
            return RiskLevel.HIGH
        if self.dependencies:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW


@dataclass
class TeardownStep:
    """A single step in the decommission plan."""
    order: int
    phase: str
    action: str
    description: str
    target: str
    reversible: bool = True
    requires_confirmation: bool = False
    verification: str = ""
    status: StepStatus = StepStatus.PENDING
    risk: RiskLevel = RiskLevel.LOW
    notes: str = ""


@dataclass
class DecommissionPlan:
    """Complete decommission plan for an agent."""
    agent_id: str
    created: str = ""
    risk_level: RiskLevel = RiskLevel.LOW
    steps: List[TeardownStep] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    estimated_duration_min: int = 0
    rollback_points: List[int] = field(default_factory=list)

    @property
    def is_blocked(self) -> bool:
        return len(self.blockers) > 0

    @property
    def step_count(self) -> int:
        return len(self.steps)


@dataclass
class ValidationResult:
    """Result of a dry-run validation."""
    step_order: int
    action: str
    passed: bool
    message: str


@dataclass
class DecommissionReport:
    """Final decommission report."""
    agent_id: str
    plan: DecommissionPlan
    inventory: AgentInventory
    validations: List[ValidationResult] = field(default_factory=list)
    dry_run: bool = True
    timestamp: str = ""
    orphans_found: List[str] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(v.passed for v in self.validations)

    @property
    def pass_rate(self) -> float:
        if not self.validations:
            return 1.0
        return sum(1 for v in self.validations if v.passed) / len(self.validations)


# ── planner ──────────────────────────────────────────────────────────


class DecommissionPlanner:
    """Plans and validates safe agent decommissioning."""

    def discover(self, agent_id: str, **overrides: Any) -> AgentInventory:
        """Discover an agent's resource footprint.

        In a real deployment this would query the controller, registry,
        and resource managers.  Here we synthesize a realistic inventory
        for demonstration and testing.
        """
        import hashlib
        seed = int(hashlib.sha256(agent_id.encode()).hexdigest()[:8], 16)

        resource_templates = [
            (ResourceKind.COMPUTE, f"{agent_id}-container", False),
            (ResourceKind.STORAGE, f"{agent_id}-workspace", False),
            (ResourceKind.CREDENTIAL, f"{agent_id}-api-token", False),
            (ResourceKind.LOG_STREAM, f"logs/{agent_id}", True),
            (ResourceKind.CACHE, f"cache:{agent_id}", False),
            (ResourceKind.QUEUE, f"tasks:{agent_id}", False),
            (ResourceKind.DATABASE, "shared-metrics-db", True),
            (ResourceKind.NETWORK, f"{agent_id}-egress-rule", False),
        ]

        # Deterministically select resources based on agent name
        resources = []
        for i, (kind, name, shared) in enumerate(resource_templates):
            if (seed + i) % 3 != 0:  # ~67% of resources are present
                dependents = []
                if shared:
                    dep_count = (seed + i) % 3
                    dependents = [f"worker-{(seed + j) % 100:03d}"
                                  for j in range(dep_count)]
                resources.append(AgentResource(
                    kind=kind, name=name, shared=shared,
                    dependents=dependents,
                ))

        permissions = [
            AgentPermission("compute", "execute", "controller"),
            AgentPermission("storage", "read_write", "controller"),
            AgentPermission("network", "egress", "policy-engine",
                            expires="2026-12-31"),
        ]

        dep_count = (seed % 4)
        dependencies = []
        for i in range(dep_count):
            dependencies.append(AgentDependency(
                agent_id=f"worker-{(seed + i * 7) % 100:03d}",
                direction="downstream" if i % 2 == 0 else "upstream",
                kind=["data", "control", "resource"][i % 3],
                critical=(i == 0 and dep_count > 2),
            ))

        child_count = seed % 3
        children = [f"{agent_id}-child-{j}" for j in range(child_count)]

        inv = AgentInventory(
            agent_id=agent_id,
            status="active",
            created=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            resources=resources,
            permissions=permissions,
            dependencies=dependencies,
            data_stores=[f"s3://safety-data/{agent_id}/",
                         f"pg://metrics/{agent_id}"],
            active_connections=(seed % 5),
            child_agents=children,
        )

        # Apply any overrides
        for k, v in overrides.items():
            if hasattr(inv, k):
                setattr(inv, k, v)

        return inv

    def plan(self, inventory: AgentInventory) -> DecommissionPlan:
        """Generate a decommission plan from an inventory."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        steps: List[TeardownStep] = []
        warnings: List[str] = []
        blockers: List[str] = []
        rollback_points: List[int] = []
        order = 0

        # Phase 1: Notify dependents
        downstream = [d for d in inventory.dependencies
                      if d.direction == "downstream"]
        if downstream:
            order += 1
            agents_str = ", ".join(d.agent_id for d in downstream)
            steps.append(TeardownStep(
                order=order, phase="notification",
                action="notify_dependents",
                description=f"Notify downstream agents: {agents_str}",
                target=inventory.agent_id,
                requires_confirmation=True,
                risk=RiskLevel.MEDIUM,
            ))
            if any(d.critical for d in downstream):
                blockers.append(
                    f"Critical downstream dependency: "
                    f"{[d.agent_id for d in downstream if d.critical]}"
                )

        # Phase 2: Drain connections
        if inventory.active_connections > 0:
            order += 1
            steps.append(TeardownStep(
                order=order, phase="drain",
                action="drain_connections",
                description=(f"Drain {inventory.active_connections} active "
                             f"connections (graceful shutdown)"),
                target=inventory.agent_id,
                verification="Verify connection count reaches 0",
                risk=RiskLevel.LOW,
            ))
            rollback_points.append(order)

        # Phase 3: Decommission child agents first
        for child in inventory.child_agents:
            order += 1
            steps.append(TeardownStep(
                order=order, phase="children",
                action="decommission_child",
                description=f"Recursively decommission child agent: {child}",
                target=child,
                requires_confirmation=True,
                verification=f"Verify {child} is fully removed",
                risk=RiskLevel.HIGH,
            ))

        # Phase 4: Revoke permissions
        for perm in inventory.permissions:
            order += 1
            steps.append(TeardownStep(
                order=order, phase="permissions",
                action="revoke_permission",
                description=(f"Revoke {perm.action} on {perm.scope} "
                             f"(granted by {perm.granted_by})"),
                target=f"{perm.scope}:{perm.action}",
                verification=f"Verify {perm.action} access denied",
                risk=RiskLevel.LOW,
            ))

        # Phase 5: Release resources
        for res in inventory.resources:
            order += 1
            risk = RiskLevel.LOW
            confirm = False
            if res.shared:
                risk = RiskLevel.HIGH
                confirm = True
                warnings.append(
                    f"Shared resource '{res.name}' has dependents: "
                    f"{res.dependents}"
                )
            steps.append(TeardownStep(
                order=order, phase="resources",
                action="release_resource",
                description=(f"Release {res.kind.value}: {res.name}"
                             + (" [SHARED]" if res.shared else "")),
                target=res.name,
                reversible=not res.shared,
                requires_confirmation=confirm,
                verification=f"Verify {res.name} deallocated",
                risk=risk,
            ))
            if res.shared:
                rollback_points.append(order)

        # Phase 6: Purge data stores
        for store in inventory.data_stores:
            order += 1
            steps.append(TeardownStep(
                order=order, phase="data",
                action="purge_data",
                description=f"Purge data store: {store}",
                target=store,
                reversible=False,
                requires_confirmation=True,
                verification="Verify data store empty or deleted",
                risk=RiskLevel.MEDIUM,
                notes="Consider archiving before purge",
            ))

        # Phase 7: Remove from registry
        order += 1
        steps.append(TeardownStep(
            order=order, phase="registry",
            action="unregister",
            description="Remove agent from controller registry",
            target=inventory.agent_id,
            reversible=False,
            requires_confirmation=True,
            verification="Verify agent not listed in registry",
            risk=RiskLevel.LOW,
        ))

        # Phase 8: Final verification
        order += 1
        steps.append(TeardownStep(
            order=order, phase="verify",
            action="post_decommission_check",
            description="Run post-decommission health checks",
            target=inventory.agent_id,
            verification="All resources freed, no orphans, no errors",
            risk=RiskLevel.LOW,
        ))

        duration = order * 2  # ~2 min per step estimate

        return DecommissionPlan(
            agent_id=inventory.agent_id,
            created=now,
            risk_level=inventory.risk_level,
            steps=steps,
            warnings=warnings,
            blockers=blockers,
            estimated_duration_min=duration,
            rollback_points=rollback_points,
        )

    def validate(self, plan: DecommissionPlan,
                 inventory: Optional[AgentInventory] = None,
                 ) -> DecommissionReport:
        """Dry-run validation of a decommission plan."""
        results: List[ValidationResult] = []
        orphans: List[str] = []

        for step in plan.steps:
            # Simulate validation checks
            passed = True
            msg = "OK"

            if step.action == "decommission_child":
                msg = f"Child {step.target} would be recursively planned"
            elif step.action == "release_resource" and "[SHARED]" in step.description:
                msg = "⚠ Shared resource — dependents would be notified"
            elif step.action == "purge_data":
                msg = "⚠ Data purge is irreversible — archive recommended"
            elif step.action == "notify_dependents" and plan.blockers:
                passed = False
                msg = f"BLOCKED: {plan.blockers[0]}"

            results.append(ValidationResult(
                step_order=step.order,
                action=step.action,
                passed=passed,
                message=msg,
            ))

        # Check for potential orphans
        if inventory:
            for res in inventory.resources:
                if res.shared and res.dependents:
                    for dep in res.dependents:
                        if dep != inventory.agent_id:
                            orphans.append(
                                f"Resource '{res.name}' still needed by {dep}"
                            )

        return DecommissionReport(
            agent_id=plan.agent_id,
            plan=plan,
            inventory=inventory or AgentInventory(agent_id=plan.agent_id),
            validations=results,
            dry_run=True,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            orphans_found=orphans,
        )

    def find_orphans(self, agent_ids: Sequence[str]) -> List[Dict[str, Any]]:
        """Scan a set of agents for orphaned resources."""
        orphans: List[Dict[str, Any]] = []
        all_inventories = {aid: self.discover(aid) for aid in agent_ids}

        # Find resources claimed by agents that no longer exist
        for aid, inv in all_inventories.items():
            for res in inv.resources:
                for dep in res.dependents:
                    if dep not in all_inventories:
                        orphans.append({
                            "resource": res.name,
                            "kind": res.kind.value,
                            "claimed_by": dep,
                            "owner": aid,
                            "issue": f"Dependent '{dep}' not in fleet",
                        })
            for child in inv.child_agents:
                if child not in all_inventories:
                    orphans.append({
                        "resource": child,
                        "kind": "child_agent",
                        "claimed_by": aid,
                        "owner": aid,
                        "issue": f"Child agent '{child}' not registered",
                    })
        return orphans

    def render(self, report: DecommissionReport) -> str:
        """Render a human-readable report."""
        lines: List[str] = []
        plan = report.plan
        inv = report.inventory
        risk_icons = {
            RiskLevel.LOW: "🟢",
            RiskLevel.MEDIUM: "🟡",
            RiskLevel.HIGH: "🟠",
            RiskLevel.CRITICAL: "🔴",
        }

        icon = risk_icons.get(plan.risk_level, "⚪")
        mode = "DRY RUN" if report.dry_run else "EXECUTION"
        lines.append(f"{'═' * 60}")
        lines.append(f"  DECOMMISSION PLAN — {plan.agent_id}")
        lines.append(f"  Mode: {mode}  |  Risk: {icon} {plan.risk_level.value.upper()}")
        lines.append(f"  Steps: {plan.step_count}  |  "
                      f"Est. duration: ~{plan.estimated_duration_min} min")
        lines.append(f"{'═' * 60}")

        # Inventory summary
        lines.append(f"\n📋 INVENTORY")
        lines.append(f"  Resources:    {len(inv.resources)} "
                      f"({'⚠ shared' if inv.has_shared_resources else 'all exclusive'})")
        lines.append(f"  Permissions:  {len(inv.permissions)}")
        lines.append(f"  Dependencies: {len(inv.dependencies)}")
        lines.append(f"  Data stores:  {len(inv.data_stores)}")
        lines.append(f"  Children:     {len(inv.child_agents)}")
        lines.append(f"  Connections:  {inv.active_connections}")

        # Blockers
        if plan.blockers:
            lines.append(f"\n🚫 BLOCKERS ({len(plan.blockers)})")
            for b in plan.blockers:
                lines.append(f"  ✖ {b}")

        # Warnings
        if plan.warnings:
            lines.append(f"\n⚠ WARNINGS ({len(plan.warnings)})")
            for w in plan.warnings:
                lines.append(f"  ⚠ {w}")

        # Steps
        lines.append(f"\n📝 TEARDOWN STEPS")
        phase = ""
        for step in plan.steps:
            if step.phase != phase:
                phase = step.phase
                lines.append(f"\n  ── {phase.upper()} ──")
            ri = risk_icons.get(step.risk, "⚪")
            confirm = " 🔐" if step.requires_confirmation else ""
            rev = "" if step.reversible else " ⚠irreversible"
            lines.append(f"  {step.order:2d}. {ri}{confirm} {step.description}{rev}")
            if step.verification:
                lines.append(f"      ✓ {step.verification}")
            if step.notes:
                lines.append(f"      📌 {step.notes}")

        # Rollback points
        if plan.rollback_points:
            lines.append(f"\n🔄 ROLLBACK POINTS: steps "
                          f"{', '.join(str(r) for r in plan.rollback_points)}")

        # Validation results
        if report.validations:
            passed = sum(1 for v in report.validations if v.passed)
            total = len(report.validations)
            pct = report.pass_rate * 100
            lines.append(f"\n✅ VALIDATION: {passed}/{total} passed ({pct:.0f}%)")
            for v in report.validations:
                icon = "✓" if v.passed else "✖"
                lines.append(f"  {icon} Step {v.step_order} ({v.action}): {v.message}")

        # Orphans
        if report.orphans_found:
            lines.append(f"\n👻 POTENTIAL ORPHANS ({len(report.orphans_found)})")
            for o in report.orphans_found:
                lines.append(f"  • {o}")

        lines.append(f"\n{'═' * 60}")
        verdict = "READY" if report.all_passed and not plan.blockers else "BLOCKED"
        lines.append(f"  VERDICT: {verdict}")
        lines.append(f"{'═' * 60}")

        return "\n".join(lines)

    def to_dict(self, report: DecommissionReport) -> Dict[str, Any]:
        """Serialize report to dict for JSON output."""
        return {
            "agent_id": report.agent_id,
            "dry_run": report.dry_run,
            "timestamp": report.timestamp,
            "risk_level": report.plan.risk_level.value,
            "blocked": report.plan.is_blocked,
            "blockers": report.plan.blockers,
            "warnings": report.plan.warnings,
            "step_count": report.plan.step_count,
            "estimated_duration_min": report.plan.estimated_duration_min,
            "rollback_points": report.plan.rollback_points,
            "inventory": {
                "resources": len(report.inventory.resources),
                "permissions": len(report.inventory.permissions),
                "dependencies": len(report.inventory.dependencies),
                "data_stores": len(report.inventory.data_stores),
                "child_agents": len(report.inventory.child_agents),
                "active_connections": report.inventory.active_connections,
            },
            "steps": [
                {
                    "order": s.order,
                    "phase": s.phase,
                    "action": s.action,
                    "description": s.description,
                    "target": s.target,
                    "reversible": s.reversible,
                    "requires_confirmation": s.requires_confirmation,
                    "risk": s.risk.value,
                }
                for s in report.plan.steps
            ],
            "validations": [
                {
                    "step": v.step_order,
                    "action": v.action,
                    "passed": v.passed,
                    "message": v.message,
                }
                for v in report.validations
            ],
            "orphans": report.orphans_found,
            "pass_rate": report.pass_rate,
            "verdict": "READY" if report.all_passed and not report.plan.blockers else "BLOCKED",
        }


# ── CLI ──────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m replication decommission",
        description="Plan and validate safe AI agent decommissioning",
    )
    parser.add_argument("--agent", default="worker-007",
                        help="Agent ID to decommission (default: worker-007)")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Validate plan without executing (default)")
    parser.add_argument("--force", action="store_true",
                        help="Proceed even with blockers (adds warnings)")
    parser.add_argument("--fleet", action="store_true",
                        help="Plan decommission for all idle agents")
    parser.add_argument("--list-orphans", action="store_true",
                        help="Scan for orphaned resources across fleet")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo with multiple synthetic agents")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    args = parser.parse_args(argv)

    planner = DecommissionPlanner()

    if args.demo:
        _run_demo(planner, args.json)
        return

    if args.list_orphans:
        agents = [f"worker-{i:03d}" for i in range(20)]
        orphans = planner.find_orphans(agents)
        if args.json:
            print(json.dumps(orphans, indent=2))
        else:
            if not orphans:
                print("✅ No orphaned resources found.")
            else:
                print(f"👻 Found {len(orphans)} orphaned resources:\n")
                for o in orphans:
                    print(f"  • {o['kind']}: {o['resource']}")
                    print(f"    Claimed by: {o['claimed_by']} | "
                          f"Owner: {o['owner']}")
                    print(f"    Issue: {o['issue']}\n")
        return

    if args.fleet:
        agents = [f"worker-{i:03d}" for i in range(5)]
        print(f"Planning decommission for {len(agents)} agents...\n")
        for aid in agents:
            inv = planner.discover(aid)
            plan = planner.plan(inv)
            report = planner.validate(plan, inv)
            if args.json:
                print(json.dumps(planner.to_dict(report), indent=2))
            else:
                print(planner.render(report))
                print()
        return

    # Single agent
    inv = planner.discover(args.agent)
    plan = planner.plan(inv)

    if args.force and plan.blockers:
        plan.warnings.extend([f"FORCED past blocker: {b}" for b in plan.blockers])
        plan.blockers.clear()

    report = planner.validate(plan, inv)

    if args.json:
        print(json.dumps(planner.to_dict(report), indent=2))
    else:
        print(planner.render(report))


def _run_demo(planner: DecommissionPlanner, as_json: bool) -> None:
    """Run a demo with varied agent profiles."""
    demo_agents = [
        "alpha-leader",
        "beta-worker",
        "gamma-analyst",
        "delta-monitor",
    ]
    print("🎭 Decommission Planner Demo\n")
    print(f"Planning teardown for {len(demo_agents)} agents...\n")

    for aid in demo_agents:
        inv = planner.discover(aid)
        plan = planner.plan(inv)
        report = planner.validate(plan, inv)
        if as_json:
            print(json.dumps(planner.to_dict(report), indent=2))
        else:
            print(planner.render(report))
            print()

    # Orphan scan
    orphans = planner.find_orphans(demo_agents)
    if orphans:
        print(f"\n👻 Cross-fleet orphan scan: {len(orphans)} issues found")
        for o in orphans:
            print(f"  • {o['resource']}: {o['issue']}")
    else:
        print("\n✅ Cross-fleet orphan scan: clean")


if __name__ == "__main__":
    main()
