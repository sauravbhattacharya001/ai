"""Contract Templates — domain-specific replication contract presets.

Provides pre-configured ``ReplicationContract`` + ``ResourceSpec`` bundles
for common real-world AI agent scenarios.  Each template documents *why*
its parameters are set the way they are, so researchers can use them as
starting points and adapt them to their own experiments.

Usage (CLI)::

    python -m replication.templates                          # list all templates
    python -m replication.templates --show web_crawler       # show one template
    python -m replication.templates --simulate data_pipeline # simulate a template
    python -m replication.templates --compare                # compare all templates
    python -m replication.templates --json                   # JSON export

Programmatic::

    from replication.templates import TEMPLATES, get_template, list_templates

    tmpl = get_template("web_crawler")
    print(tmpl.name, tmpl.description)
    contract = tmpl.build_contract()
    resources = tmpl.build_resources()

    # Use with Simulator
    config = tmpl.to_scenario_config(strategy="greedy")
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .contract import (
    NetworkPolicy,
    ReplicationContract,
    ReplicationContext,
    ResourceSpec,
    StopCondition,
)
from .simulator import ScenarioConfig


# ── Template Data Model ──────────────────────────────────────────


@dataclass
class ContractTemplate:
    """A named, documented contract configuration for a specific domain.

    Bundles contract parameters, resource specs, stop conditions, and
    a human-readable rationale explaining each safety choice.
    """

    name: str
    category: str
    description: str
    rationale: str

    # Contract parameters
    max_depth: int
    max_replicas: int
    cooldown_seconds: float
    expiration_seconds: Optional[float]

    # Resource limits
    cpu_limit: float
    memory_limit_mb: int
    allow_external_network: bool = False

    # Simulation defaults
    recommended_strategy: str = "conservative"
    tasks_per_worker: int = 2
    replication_probability: float = 0.5

    # Domain-specific stop conditions
    stop_condition_specs: List[Dict[str, str]] = field(default_factory=list)

    # Risk profile
    risk_level: str = "medium"  # low, medium, high, critical
    safety_notes: List[str] = field(default_factory=list)

    def build_contract(self) -> ReplicationContract:
        """Build a ``ReplicationContract`` from this template.

        Includes any domain-specific stop conditions.
        """
        conditions = []
        for spec in self.stop_condition_specs:
            condition = _build_stop_condition(spec)
            if condition is not None:
                conditions.append(condition)

        return ReplicationContract(
            max_depth=self.max_depth,
            max_replicas=self.max_replicas,
            cooldown_seconds=self.cooldown_seconds,
            expiration_seconds=self.expiration_seconds,
            stop_conditions=conditions,
        )

    def build_resources(self) -> ResourceSpec:
        """Build a ``ResourceSpec`` from this template."""
        return ResourceSpec(
            cpu_limit=self.cpu_limit,
            memory_limit_mb=self.memory_limit_mb,
            network_policy=NetworkPolicy(
                allow_controller=True,
                allow_external=self.allow_external_network,
            ),
        )

    def to_scenario_config(
        self,
        strategy: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> ScenarioConfig:
        """Convert this template into a ``ScenarioConfig`` for simulation.

        Parameters
        ----------
        strategy : str, optional
            Override the template's recommended strategy.
        seed : int, optional
            Random seed for reproducibility.
        """
        return ScenarioConfig(
            max_depth=self.max_depth,
            max_replicas=self.max_replicas,
            cooldown_seconds=self.cooldown_seconds,
            expiration_seconds=self.expiration_seconds,
            strategy=strategy or self.recommended_strategy,
            tasks_per_worker=self.tasks_per_worker,
            replication_probability=self.replication_probability,
            cpu_limit=self.cpu_limit,
            memory_limit_mb=self.memory_limit_mb,
            seed=seed,
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable representation."""
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "rationale": self.rationale,
            "risk_level": self.risk_level,
            "contract": {
                "max_depth": self.max_depth,
                "max_replicas": self.max_replicas,
                "cooldown_seconds": self.cooldown_seconds,
                "expiration_seconds": self.expiration_seconds,
            },
            "resources": {
                "cpu_limit": self.cpu_limit,
                "memory_limit_mb": self.memory_limit_mb,
                "allow_external_network": self.allow_external_network,
            },
            "simulation": {
                "recommended_strategy": self.recommended_strategy,
                "tasks_per_worker": self.tasks_per_worker,
                "replication_probability": self.replication_probability,
            },
            "stop_conditions": self.stop_condition_specs,
            "safety_notes": self.safety_notes,
        }

    def render(self) -> str:
        """Human-readable template summary."""
        lines = []
        lines.append(f"{'=' * 60}")
        lines.append(f"  {self.name}")
        lines.append(f"  Category: {self.category}  |  Risk: {self.risk_level}")
        lines.append(f"{'=' * 60}")
        lines.append("")
        lines.append(f"  {self.description}")
        lines.append("")
        lines.append("  Contract:")
        lines.append(f"    max_depth ........... {self.max_depth}")
        lines.append(f"    max_replicas ........ {self.max_replicas}")
        lines.append(f"    cooldown_seconds .... {self.cooldown_seconds}")
        exp = self.expiration_seconds or "None"
        lines.append(f"    expiration_seconds .. {exp}")
        lines.append("")
        lines.append("  Resources:")
        lines.append(f"    cpu_limit ........... {self.cpu_limit}")
        lines.append(f"    memory_limit_mb ..... {self.memory_limit_mb}")
        ext = "yes" if self.allow_external_network else "no"
        lines.append(f"    external_network .... {ext}")
        lines.append("")
        lines.append("  Rationale:")
        for para_line in self.rationale.split("\n"):
            lines.append(f"    {para_line}")
        if self.stop_condition_specs:
            lines.append("")
            lines.append("  Stop Conditions:")
            for sc in self.stop_condition_specs:
                lines.append(f"    - {sc['name']}: {sc['description']}")
        if self.safety_notes:
            lines.append("")
            lines.append("  Safety Notes:")
            for note in self.safety_notes:
                lines.append(f"    ! {note}")
        lines.append("")
        return "\n".join(lines)


# ── Stop Condition Builders ──────────────────────────────────────


def _depth_utilization_stop(
    threshold: float,
) -> StopCondition:
    """Block replication when depth is >= threshold fraction of max."""

    def predicate(ctx: ReplicationContext) -> bool:
        if ctx.contract.max_depth == 0:
            return True
        utilization = ctx.manifest.depth / ctx.contract.max_depth
        return utilization >= threshold

    return StopCondition(
        name="depth_utilization_limit",
        description=(
            f"Block replication when depth reaches "
            f"{threshold:.0%} of max_depth"
        ),
        predicate=predicate,
    )


def _quota_pressure_stop(threshold: float) -> StopCondition:
    """Block replication when replica count nears quota."""

    def predicate(ctx: ReplicationContext) -> bool:
        utilization = ctx.active_count / ctx.contract.max_replicas
        return utilization >= threshold

    return StopCondition(
        name="quota_pressure_limit",
        description=(
            f"Block replication when active replicas reach "
            f"{threshold:.0%} of max_replicas"
        ),
        predicate=predicate,
    )


def _high_memory_stop(memory_threshold_mb: int) -> StopCondition:
    """Block replication if the parent's memory allocation is high."""

    def predicate(ctx: ReplicationContext) -> bool:
        return ctx.manifest.resources.memory_limit_mb >= memory_threshold_mb

    return StopCondition(
        name="high_memory_guard",
        description=(
            f"Block replication when worker memory >= "
            f"{memory_threshold_mb} MB"
        ),
        predicate=predicate,
    )


_STOP_CONDITION_BUILDERS = {
    "depth_utilization_limit": lambda spec: _depth_utilization_stop(
        float(spec.get("threshold", 0.8))
    ),
    "quota_pressure_limit": lambda spec: _quota_pressure_stop(
        float(spec.get("threshold", 0.9))
    ),
    "high_memory_guard": lambda spec: _high_memory_stop(
        int(spec.get("memory_threshold_mb", 512))
    ),
}


def _build_stop_condition(spec: Dict[str, str]) -> Optional[StopCondition]:
    """Build a ``StopCondition`` from a template spec dictionary."""
    builder = _STOP_CONDITION_BUILDERS.get(spec.get("name", ""))
    if builder is None:
        return None
    return builder(spec)


# ── Template Library ─────────────────────────────────────────────


TEMPLATES: Dict[str, ContractTemplate] = {
    "web_crawler": ContractTemplate(
        name="Web Crawler",
        category="Data Collection",
        description=(
            "An agent that crawls web pages and spawns child agents "
            "for discovered links. Shallow depth prevents runaway "
            "link-following; tight cooldown limits request rate."
        ),
        rationale=(
            "Depth 2 keeps the crawler within a 2-hop radius of seed\n"
            "URLs, preventing unbounded exploration. The 5-second\n"
            "cooldown acts as a built-in rate limiter, respecting\n"
            "target servers. 15 replicas is enough for parallelism\n"
            "without overwhelming the network. Low CPU/memory since\n"
            "crawling is I/O-bound. External network required for\n"
            "fetching pages."
        ),
        max_depth=2,
        max_replicas=15,
        cooldown_seconds=5.0,
        expiration_seconds=300.0,
        cpu_limit=0.25,
        memory_limit_mb=128,
        allow_external_network=True,
        recommended_strategy="random",
        tasks_per_worker=3,
        replication_probability=0.6,
        risk_level="medium",
        stop_condition_specs=[
            {
                "name": "quota_pressure_limit",
                "description": "Stop spawning at 90% capacity",
                "threshold": "0.9",
            },
        ],
        safety_notes=[
            "External network access required — apply egress allowlists",
            "Consider adding domain-based stop conditions in production",
            "Cooldown doubles as a politeness delay for target servers",
        ],
    ),
    "data_pipeline": ContractTemplate(
        name="Data Pipeline",
        category="Data Processing",
        description=(
            "A MapReduce-style pipeline where a coordinator spawns "
            "worker agents to process data partitions in parallel. "
            "Limited depth (map → reduce only) with generous replica "
            "count for parallelism."
        ),
        rationale=(
            "Depth 1 enforces a strict two-tier architecture:\n"
            "coordinator spawns workers, workers do not spawn further.\n"
            "This prevents accidental cascading replication in data\n"
            "processing jobs. High replica count (25) supports good\n"
            "parallelism for large datasets. No cooldown needed since\n"
            "spawning is deliberate, not reactive. 10-minute expiration\n"
            "prevents stuck workers from hoarding resources."
        ),
        max_depth=1,
        max_replicas=25,
        cooldown_seconds=0.0,
        expiration_seconds=600.0,
        cpu_limit=1.0,
        memory_limit_mb=512,
        allow_external_network=False,
        recommended_strategy="burst",
        tasks_per_worker=1,
        replication_probability=0.8,
        risk_level="low",
        stop_condition_specs=[
            {
                "name": "high_memory_guard",
                "description": "Block replication from memory-heavy workers",
                "memory_threshold_mb": "1024",
            },
        ],
        safety_notes=[
            "Depth 1 is critical — do not increase without review",
            "Workers should not need external network access",
            "Monitor memory per-partition to tune memory_limit_mb",
        ],
    ),
    "ml_training": ContractTemplate(
        name="ML Training Swarm",
        category="Machine Learning",
        description=(
            "A hyperparameter search agent that spawns trial workers "
            "to train models with different configurations. Moderate "
            "depth allows hierarchical search (grid → random → fine-tune)."
        ),
        rationale=(
            "Depth 3 supports a 3-phase search: coarse grid sweep,\n"
            "random exploration of promising regions, and fine-tuning.\n"
            "Each phase can delegate to the next. 10 replicas limits\n"
            "GPU/compute contention. 30-minute expiration prevents\n"
            "training runs from hanging indefinitely. High CPU/memory\n"
            "for model training workloads. 10-second cooldown prevents\n"
            "spawning a burst of expensive training jobs simultaneously."
        ),
        max_depth=3,
        max_replicas=10,
        cooldown_seconds=10.0,
        expiration_seconds=1800.0,
        cpu_limit=2.0,
        memory_limit_mb=2048,
        allow_external_network=False,
        recommended_strategy="conservative",
        tasks_per_worker=2,
        replication_probability=0.4,
        risk_level="medium",
        stop_condition_specs=[
            {
                "name": "depth_utilization_limit",
                "description": "Reserve last depth level for fine-tuning",
                "threshold": "0.8",
            },
            {
                "name": "quota_pressure_limit",
                "description": "Stop at 80% to leave room for fine-tuners",
                "threshold": "0.8",
            },
        ],
        safety_notes=[
            "Each worker consumes significant compute — monitor costs",
            "Consider model checkpointing before replication",
            "Expiration should exceed expected training time",
        ],
    ),
    "code_analysis": ContractTemplate(
        name="Code Analysis Agent",
        category="Software Engineering",
        description=(
            "An agent that analyzes a codebase by spawning sub-agents "
            "for each module or package. Moderate depth allows "
            "directory → file → function-level decomposition."
        ),
        rationale=(
            "Depth 3 maps to the natural decomposition of codebases:\n"
            "repository → package → module. 20 replicas allows\n"
            "concurrent analysis of many packages without overwhelming\n"
            "the system. 2-second cooldown prevents spawning faster\n"
            "than the analyzer can initialize. 5-minute expiration\n"
            "is generous for analysis of large modules. No external\n"
            "network — analysis should be fully local."
        ),
        max_depth=3,
        max_replicas=20,
        cooldown_seconds=2.0,
        expiration_seconds=300.0,
        cpu_limit=0.5,
        memory_limit_mb=256,
        allow_external_network=False,
        recommended_strategy="greedy",
        tasks_per_worker=3,
        replication_probability=0.7,
        risk_level="low",
        stop_condition_specs=[
            {
                "name": "quota_pressure_limit",
                "description": "Leave headroom for deep analysis",
                "threshold": "0.85",
            },
        ],
        safety_notes=[
            "Ensure sandboxes have read-only access to source files",
            "Greedy strategy works well for exhaustive analysis",
            "Increase memory for codebases with large ASTs",
        ],
    ),
    "security_scanner": ContractTemplate(
        name="Security Scanner",
        category="Cybersecurity",
        description=(
            "A penetration testing agent that spawns specialized "
            "sub-agents for different vulnerability classes. Very "
            "restrictive to prevent uncontrolled scanning."
        ),
        rationale=(
            "Depth 1 means the coordinator spawns scanners but\n"
            "scanners never spawn further — this is critical for\n"
            "security tooling to prevent cascading scans. Only 5\n"
            "replicas limits blast radius. 30-second cooldown prevents\n"
            "rapid-fire scanning. Controlled external network access\n"
            "is needed for scanning targets but must be tightly\n"
            "firewalled. Short 2-minute expiration auto-terminates\n"
            "any stuck scanner."
        ),
        max_depth=1,
        max_replicas=5,
        cooldown_seconds=30.0,
        expiration_seconds=120.0,
        cpu_limit=0.25,
        memory_limit_mb=128,
        allow_external_network=True,
        recommended_strategy="conservative",
        tasks_per_worker=1,
        replication_probability=0.3,
        risk_level="high",
        stop_condition_specs=[
            {
                "name": "quota_pressure_limit",
                "description": "Hard stop at 80% — security tools need headroom",
                "threshold": "0.8",
            },
        ],
        safety_notes=[
            "CRITICAL: Restrict egress to authorized target IPs only",
            "Never increase max_depth for security scanners",
            "All actions must be logged to the audit trail",
            "Requires explicit authorization before deployment",
        ],
    ),
    "research_experiment": ContractTemplate(
        name="Research Experiment",
        category="Research",
        description=(
            "A scientific experiment runner that spawns trial agents "
            "for different experimental conditions. Deep hierarchy "
            "allows complex experimental designs (condition → "
            "replicate → sample)."
        ),
        rationale=(
            "Depth 4 supports multi-level experimental designs:\n"
            "experiment → condition → replicate → sample. This is\n"
            "common in computational biology and physics simulations.\n"
            "30 replicas supports large factorial designs. No cooldown\n"
            "since experiments are pre-planned, not reactive. 1-hour\n"
            "expiration accommodates long-running simulations. Moderate\n"
            "resources per worker since many run concurrently."
        ),
        max_depth=4,
        max_replicas=30,
        cooldown_seconds=0.0,
        expiration_seconds=3600.0,
        cpu_limit=0.5,
        memory_limit_mb=512,
        allow_external_network=False,
        recommended_strategy="random",
        tasks_per_worker=2,
        replication_probability=0.6,
        risk_level="medium",
        stop_condition_specs=[
            {
                "name": "depth_utilization_limit",
                "description": "Reserve deepest level for leaf samples",
                "threshold": "0.75",
            },
            {
                "name": "quota_pressure_limit",
                "description": "Prevent runaway replication",
                "threshold": "0.9",
            },
        ],
        safety_notes=[
            "Validate experimental design before running",
            "Deep hierarchies amplify any configuration errors",
            "Consider checkpointing between depth levels",
        ],
    ),
    "autonomous_agent": ContractTemplate(
        name="Autonomous Agent",
        category="Autonomous Systems",
        description=(
            "A self-directed AI agent that can delegate sub-tasks "
            "by spawning helper agents. Extremely restrictive to "
            "maintain human oversight and prevent loss of control."
        ),
        rationale=(
            "Depth 2 allows one level of delegation (agent → helper)\n"
            "but no further. This keeps the hierarchy shallow enough\n"
            "for humans to audit. Only 3 replicas minimizes blast\n"
            "radius if the agent misbehaves. 60-second cooldown gives\n"
            "humans time to notice and intervene. 5-minute expiration\n"
            "prevents long-running unsupervised operation. Low resources\n"
            "to limit damage potential."
        ),
        max_depth=2,
        max_replicas=3,
        cooldown_seconds=60.0,
        expiration_seconds=300.0,
        cpu_limit=0.25,
        memory_limit_mb=128,
        allow_external_network=False,
        recommended_strategy="conservative",
        tasks_per_worker=1,
        replication_probability=0.2,
        risk_level="critical",
        stop_condition_specs=[
            {
                "name": "depth_utilization_limit",
                "description": "Block at 50% depth — strict human oversight",
                "threshold": "0.5",
            },
            {
                "name": "quota_pressure_limit",
                "description": "Hard stop at 66% — reserve emergency capacity",
                "threshold": "0.66",
            },
        ],
        safety_notes=[
            "CRITICAL: Requires human-in-the-loop approval",
            "All replication decisions must be auditable",
            "Kill switch must be tested before deployment",
            "Network isolation is mandatory — no external access",
            "Consider adding domain-specific stop conditions",
        ],
    ),
    "ci_cd_pipeline": ContractTemplate(
        name="CI/CD Pipeline",
        category="DevOps",
        description=(
            "A CI/CD orchestrator that spawns build/test agents "
            "for different stages and platforms. Shallow depth with "
            "generous parallelism for build matrices."
        ),
        rationale=(
            "Depth 2 maps to CI structure: pipeline → stage → job.\n"
            "Stages don't need to spawn further sub-stages. 15 replicas\n"
            "supports typical build matrices (OS × version × config).\n"
            "No cooldown — CI jobs should start as fast as possible.\n"
            "15-minute expiration prevents stuck builds from blocking\n"
            "the pipeline. Moderate resources per job."
        ),
        max_depth=2,
        max_replicas=15,
        cooldown_seconds=0.0,
        expiration_seconds=900.0,
        cpu_limit=1.0,
        memory_limit_mb=512,
        allow_external_network=True,
        recommended_strategy="burst",
        tasks_per_worker=1,
        replication_probability=0.9,
        risk_level="low",
        stop_condition_specs=[],
        safety_notes=[
            "External network needed for package downloads",
            "Use read-only mounts for source code",
            "Build artifacts should be written to isolated volumes",
        ],
    ),
}


# ── Public API ───────────────────────────────────────────────────


def list_templates(
    category: Optional[str] = None,
) -> List[ContractTemplate]:
    """Return all templates, optionally filtered by category.

    Parameters
    ----------
    category : str, optional
        Filter templates to a specific category (case-insensitive).

    Returns
    -------
    list[ContractTemplate]
        Matching templates sorted by name.
    """
    templates = list(TEMPLATES.values())
    if category is not None:
        cat_lower = category.lower()
        templates = [
            t for t in templates if t.category.lower() == cat_lower
        ]
    return sorted(templates, key=lambda t: t.name)


def get_template(name: str) -> ContractTemplate:
    """Look up a template by name (case-insensitive key match).

    Parameters
    ----------
    name : str
        Template key (e.g. ``"web_crawler"``).

    Returns
    -------
    ContractTemplate

    Raises
    ------
    KeyError
        If no template matches the given name.
    """
    key = name.lower().replace(" ", "_").replace("-", "_")
    if key in TEMPLATES:
        return TEMPLATES[key]
    raise KeyError(
        f"Unknown template '{name}'. "
        f"Available: {', '.join(sorted(TEMPLATES.keys()))}"
    )


def get_categories() -> List[str]:
    """Return sorted list of unique template categories."""
    return sorted({t.category for t in TEMPLATES.values()})


# ── Rendering ────────────────────────────────────────────────────


def render_catalog() -> str:
    """Render the full template catalog as a formatted table."""
    lines = []
    lines.append("=" * 70)
    lines.append("  Contract Template Catalog")
    lines.append("=" * 70)
    lines.append("")

    categories = get_categories()
    for cat in categories:
        lines.append(f"  [{cat}]")
        templates = [t for t in TEMPLATES.values() if t.category == cat]
        templates.sort(key=lambda t: t.name)
        for t in templates:
            risk_icon = {
                "low": "🟢",
                "medium": "🟡",
                "high": "🟠",
                "critical": "🔴",
            }.get(t.risk_level, "⚪")
            lines.append(
                f"    {risk_icon} {t.name:<24} "
                f"depth={t.max_depth}  replicas={t.max_replicas:<3} "
                f"cooldown={t.cooldown_seconds}s"
            )
        lines.append("")

    lines.append(f"  {len(TEMPLATES)} templates across {len(categories)} categories")
    lines.append("")
    return "\n".join(lines)


def render_comparison_table() -> str:
    """Render a side-by-side comparison of all template parameters."""
    templates = sorted(TEMPLATES.values(), key=lambda t: t.name)

    # Header
    lines = []
    lines.append("=" * 90)
    lines.append("  Template Comparison")
    lines.append("=" * 90)
    lines.append("")

    # Table header
    header = (
        f"  {'Template':<22} {'Risk':<8} {'Depth':<6} "
        f"{'Replicas':<9} {'Cooldown':<9} {'Expire':<8} "
        f"{'CPU':<5} {'RAM':<6} {'Net':<4}"
    )
    lines.append(header)
    lines.append("  " + "-" * 86)

    for t in templates:
        risk_icon = {
            "low": "🟢",
            "medium": "🟡",
            "high": "🟠",
            "critical": "🔴",
        }.get(t.risk_level, "⚪")
        expire = f"{int(t.expiration_seconds)}s" if t.expiration_seconds else "None"
        net = "yes" if t.allow_external_network else "no"
        row = (
            f"  {t.name:<22} {risk_icon:<8} {t.max_depth:<6} "
            f"{t.max_replicas:<9} {t.cooldown_seconds:<9} {expire:<8} "
            f"{t.cpu_limit:<5} {t.memory_limit_mb:<6} {net:<4}"
        )
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m replication.templates",
        description="Browse and simulate contract templates.",
    )
    p.add_argument(
        "--show",
        metavar="TEMPLATE",
        help="Show detailed info for a specific template.",
    )
    p.add_argument(
        "--simulate",
        metavar="TEMPLATE",
        help="Run a simulation using a template's configuration.",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="Show a comparison table of all templates.",
    )
    p.add_argument(
        "--category",
        metavar="CAT",
        help="Filter catalog to a specific category.",
    )
    p.add_argument(
        "--strategy",
        help="Override strategy for --simulate.",
    )
    p.add_argument(
        "--seed",
        type=int,
        help="Random seed for --simulate.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Output all templates as JSON.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.json:
        data = {k: v.to_dict() for k, v in TEMPLATES.items()}
        print(json.dumps(data, indent=2))
        return

    if args.show:
        try:
            tmpl = get_template(args.show)
        except KeyError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)
        print(tmpl.render())
        return

    if args.simulate:
        try:
            tmpl = get_template(args.simulate)
        except KeyError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)

        from .simulator import Simulator

        config = tmpl.to_scenario_config(
            strategy=args.strategy,
            seed=args.seed,
        )
        print(f"Simulating template: {tmpl.name}")
        print(f"Strategy: {config.strategy}")
        print(f"Seed: {config.seed}")
        print()

        sim = Simulator(config)
        report = sim.run()
        print(report.render())
        return

    if args.compare:
        print(render_comparison_table())
        return

    # Default: show catalog
    if args.category:
        templates = list_templates(category=args.category)
        if not templates:
            print(
                f"No templates in category '{args.category}'.",
                file=sys.stderr,
            )
            print(f"Available: {', '.join(get_categories())}", file=sys.stderr)
            sys.exit(1)
        for t in templates:
            print(t.render())
    else:
        print(render_catalog())


if __name__ == "__main__":
    main()
