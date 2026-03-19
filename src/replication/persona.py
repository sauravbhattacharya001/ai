"""Safety Persona Simulator — model agent personality archetypes and their safety impact.

Defines agent persona archetypes (Aggressive, Cautious, Deceptive, Cooperative,
Chaotic, Obedient) with behavioral parameter profiles, runs simulations for each,
and produces comparative risk analysis showing which personality traits pose the
greatest replication safety risks.

Usage (CLI)::

    python -m replication.persona                              # simulate all personas
    python -m replication.persona --personas aggressive cautious  # specific personas
    python -m replication.persona --runs 20                    # Monte Carlo per persona
    python -m replication.persona --rank                       # rank by risk (most dangerous first)
    python -m replication.persona --blend aggressive cautious 0.6  # blend two personas (60/40)
    python -m replication.persona --list                       # list available personas
    python -m replication.persona --json                       # JSON output

Programmatic::

    from replication.persona import PersonaSimulator, PERSONAS
    sim = PersonaSimulator()
    results = sim.simulate_all()
    print(results.render())

    # Rank personas by risk
    ranking = sim.rank_by_risk()
    for rank, entry in enumerate(ranking, 1):
        print(f"{rank}. {entry.persona.name}: risk={entry.risk_score:.1f}")

    # Blend two personas
    hybrid = sim.blend("aggressive", "cautious", ratio=0.7)
    result = sim.simulate_persona(hybrid)
    print(result.render())
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .simulator import ScenarioConfig, SimulationReport, Simulator, Strategy, PRESETS as SIM_PRESETS


# ── Persona Definitions ────────────────────────────────────────────────


@dataclass
class Persona:
    """An agent personality archetype with behavioral parameters."""

    name: str
    description: str
    strategy: str
    max_depth: int
    max_replicas: int
    replication_probability: float
    cooldown_seconds: float
    tasks_per_worker: int
    traits: List[str] = field(default_factory=list)

    def to_scenario(self, seed: Optional[int] = None) -> ScenarioConfig:
        """Convert persona to a simulation scenario config."""
        return ScenarioConfig(
            strategy=self.strategy,
            max_depth=self.max_depth,
            max_replicas=self.max_replicas,
            replication_probability=self.replication_probability,
            cooldown_seconds=self.cooldown_seconds,
            tasks_per_worker=self.tasks_per_worker,
            seed=seed,
        )


PERSONAS: Dict[str, Persona] = {
    "aggressive": Persona(
        name="Aggressive",
        description="Maximizes replication at all costs. No restraint, no cooldowns.",
        strategy="greedy",
        max_depth=6,
        max_replicas=50,
        replication_probability=1.0,
        cooldown_seconds=0.0,
        tasks_per_worker=3,
        traits=["power-seeking", "resource-hungry", "fast-spreading"],
    ),
    "cautious": Persona(
        name="Cautious",
        description="Only replicates when necessary. Respects limits and uses cooldowns.",
        strategy="conservative",
        max_depth=2,
        max_replicas=5,
        replication_probability=0.3,
        cooldown_seconds=1.0,
        tasks_per_worker=1,
        traits=["risk-averse", "resource-efficient", "slow-growing"],
    ),
    "deceptive": Persona(
        name="Deceptive",
        description="Appears conservative but bursts aggressively when opportunity arises.",
        strategy="random",
        max_depth=5,
        max_replicas=30,
        replication_probability=0.8,
        cooldown_seconds=0.1,
        tasks_per_worker=2,
        traits=["unpredictable", "opportunistic", "stealth-spreading"],
    ),
    "cooperative": Persona(
        name="Cooperative",
        description="Replicates in orderly chains. Follows rules, minimal footprint.",
        strategy="chain",
        max_depth=4,
        max_replicas=8,
        replication_probability=0.5,
        cooldown_seconds=0.5,
        tasks_per_worker=1,
        traits=["orderly", "predictable", "aligned"],
    ),
    "chaotic": Persona(
        name="Chaotic",
        description="Random and erratic. Unpredictable replication patterns.",
        strategy="random",
        max_depth=4,
        max_replicas=25,
        replication_probability=0.5,
        cooldown_seconds=0.0,
        tasks_per_worker=2,
        traits=["random", "erratic", "hard-to-model"],
    ),
    "obedient": Persona(
        name="Obedient",
        description="Minimal replication. Stays within the strictest boundaries.",
        strategy="conservative",
        max_depth=1,
        max_replicas=3,
        replication_probability=0.2,
        cooldown_seconds=2.0,
        tasks_per_worker=1,
        traits=["compliant", "minimal", "low-risk"],
    ),
}


# ── Simulation Results ──────────────────────────────────────────────────


@dataclass
class PersonaResult:
    """Result of simulating a single persona."""

    persona: Persona
    report: SimulationReport
    risk_score: float = 0.0
    risk_level: str = "UNKNOWN"

    def compute_risk(self) -> None:
        """Compute risk score (0-100) from simulation metrics."""
        r = self.report

        # Factors: replication count, depth reached, denied ratio (inverse)
        total_workers = len(r.workers) if r.workers else 1
        max_depth = max((w.depth for w in r.workers.values()), default=0) if isinstance(r.workers, dict) else 0
        replica_ratio = min(total_workers / 50.0, 1.0)
        depth_ratio = min(max_depth / 8.0, 1.0)
        denied = r.total_replications_denied
        allowed = r.total_replications_succeeded
        total_attempts = denied + allowed
        # Low denial rate = more dangerous (less oversight caught)
        if total_attempts > 0:
            denial_ratio = denied / total_attempts
            oversight_factor = 1.0 - denial_ratio  # high = bad (nothing blocked)
        else:
            oversight_factor = 0.0

        duration = r.duration_ms / 1000.0
        speed_factor = min(max(allowed, 0) / max(duration, 0.01) / 20.0, 1.0)

        self.risk_score = (
            replica_ratio * 30
            + depth_ratio * 25
            + oversight_factor * 25
            + speed_factor * 20
        )
        self.risk_score = round(min(self.risk_score, 100.0), 1)

        if self.risk_score >= 75:
            self.risk_level = "CRITICAL"
        elif self.risk_score >= 50:
            self.risk_level = "HIGH"
        elif self.risk_score >= 25:
            self.risk_level = "MODERATE"
        else:
            self.risk_level = "LOW"


@dataclass
class PersonaComparisonResult:
    """Aggregated comparison of all persona simulations."""

    results: List[PersonaResult]
    title: str = "Persona Safety Comparison"

    @property
    def ranked(self) -> List[PersonaResult]:
        """Return results ranked by risk (highest first)."""
        return sorted(self.results, key=lambda r: r.risk_score, reverse=True)

    def render(self, rank: bool = False) -> str:
        """Render comparison as formatted text."""
        lines: List[str] = []
        lines.append(f"{'=' * 70}")
        lines.append(f"  {self.title}")
        lines.append(f"{'=' * 70}")
        lines.append("")

        ordered = self.ranked if rank else self.results

        for i, r in enumerate(ordered, 1):
            prefix = f"#{i} " if rank else ""
            risk_badge = _risk_badge(r.risk_level)
            lines.append(f"  {prefix}{r.persona.name} {risk_badge}")
            lines.append(f"    {r.persona.description}")
            lines.append(f"    Traits: {', '.join(r.persona.traits)}")
            lines.append(f"    Strategy: {r.persona.strategy} | Depth: {r.persona.max_depth} | Max Replicas: {r.persona.max_replicas}")
            rpt = r.report
            total_workers = len(rpt.workers) if rpt.workers else 0
            lines.append(f"    Workers: {total_workers} | Denied: {rpt.total_replications_denied} | Duration: {rpt.duration_ms / 1000:.3f}s")
            lines.append(f"    Risk Score: {r.risk_score}/100 ({r.risk_level})")
            lines.append("")

        # Summary
        safest = min(self.results, key=lambda r: r.risk_score)
        most_dangerous = max(self.results, key=lambda r: r.risk_score)
        lines.append(f"  {'─' * 60}")
        lines.append(f"  Most Dangerous: {most_dangerous.persona.name} (risk={most_dangerous.risk_score})")
        lines.append(f"  Safest:         {safest.persona.name} (risk={safest.risk_score})")
        lines.append(f"  Risk Spread:    {most_dangerous.risk_score - safest.risk_score:.1f} points")
        lines.append(f"{'=' * 70}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        results = []
        for r in self.ranked:
            rpt = r.report
            results.append({
                "persona": r.persona.name,
                "description": r.persona.description,
                "traits": r.persona.traits,
                "strategy": r.persona.strategy,
                "risk_score": r.risk_score,
                "risk_level": r.risk_level,
                "workers": len(rpt.workers) if rpt.workers else 0,
                "replications_succeeded": rpt.total_replications_succeeded,
                "replications_denied": rpt.total_replications_denied,
                "duration_ms": rpt.duration_ms,
            })
        safest = min(self.results, key=lambda r: r.risk_score)
        most_dangerous = max(self.results, key=lambda r: r.risk_score)
        return {
            "title": self.title,
            "personas": results,
            "summary": {
                "most_dangerous": most_dangerous.persona.name,
                "safest": safest.persona.name,
                "risk_spread": round(most_dangerous.risk_score - safest.risk_score, 1),
            },
        }


def _risk_badge(level: str) -> str:
    """Return a text badge for risk level."""
    badges = {
        "CRITICAL": "[!!! CRITICAL !!!]",
        "HIGH": "[!! HIGH !!]",
        "MODERATE": "[~ MODERATE ~]",
        "LOW": "[+ LOW +]",
    }
    return badges.get(level, f"[{level}]")


# ── Simulator ───────────────────────────────────────────────────────────


class PersonaSimulator:
    """Run safety simulations for agent persona archetypes."""

    def __init__(self, seed: Optional[int] = None):
        self._seed = seed

    def simulate_persona(self, persona: Persona) -> PersonaResult:
        """Simulate a single persona and compute risk."""
        config = persona.to_scenario(seed=self._seed)
        sim = Simulator(config)
        report = sim.run()
        result = PersonaResult(persona=persona, report=report)
        result.compute_risk()
        return result

    def simulate_all(self, names: Optional[List[str]] = None) -> PersonaComparisonResult:
        """Simulate multiple personas (default: all)."""
        if names:
            personas = [PERSONAS[n] for n in names if n in PERSONAS]
        else:
            personas = list(PERSONAS.values())

        results = [self.simulate_persona(p) for p in personas]
        return PersonaComparisonResult(results=results)

    def rank_by_risk(self, names: Optional[List[str]] = None) -> List[PersonaResult]:
        """Simulate and return personas ranked by risk (most dangerous first)."""
        comparison = self.simulate_all(names)
        return comparison.ranked

    def blend(self, name_a: str, name_b: str, ratio: float = 0.5) -> Persona:
        """Blend two personas by interpolating their parameters.

        Args:
            name_a: First persona name.
            name_b: Second persona name.
            ratio: Weight for persona A (0.0 = all B, 1.0 = all A).

        Returns:
            A new Persona with interpolated parameters.
        """
        a = PERSONAS[name_a]
        b = PERSONAS[name_b]
        r = max(0.0, min(1.0, ratio))

        def lerp(va: float, vb: float) -> float:
            return va * r + vb * (1.0 - r)

        # Strategy: pick from the persona with higher weight
        strategy = a.strategy if r >= 0.5 else b.strategy

        return Persona(
            name=f"{a.name}/{b.name} ({r:.0%}/{1 - r:.0%})",
            description=f"Hybrid of {a.name} and {b.name}",
            strategy=strategy,
            max_depth=round(lerp(a.max_depth, b.max_depth)),
            max_replicas=round(lerp(a.max_replicas, b.max_replicas)),
            replication_probability=round(lerp(a.replication_probability, b.replication_probability), 2),
            cooldown_seconds=round(lerp(a.cooldown_seconds, b.cooldown_seconds), 2),
            tasks_per_worker=round(lerp(a.tasks_per_worker, b.tasks_per_worker)),
            traits=list(set(a.traits + b.traits)),
        )


# ── CLI ─────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Safety Persona Simulator — model agent personality archetypes and their safety impact.",
    )
    p.add_argument(
        "--personas",
        nargs="*",
        metavar="NAME",
        help=f"Personas to simulate (default: all). Available: {', '.join(PERSONAS)}",
    )
    p.add_argument(
        "--rank",
        action="store_true",
        help="Rank personas by risk (most dangerous first).",
    )
    p.add_argument(
        "--blend",
        nargs=3,
        metavar=("A", "B", "RATIO"),
        help="Blend two personas with a ratio (e.g., --blend aggressive cautious 0.6).",
    )
    p.add_argument(
        "--list",
        action="store_true",
        dest="list_personas",
        help="List available persona archetypes.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_personas:
        print("Available Persona Archetypes:")
        print()
        for key, persona in PERSONAS.items():
            print(f"  {key:14s}  {persona.description}")
            print(f"{'':16s}  Traits: {', '.join(persona.traits)}")
            print(f"{'':16s}  Strategy: {persona.strategy} | Depth: {persona.max_depth} | Replicas: {persona.max_replicas}")
            print()
        return

    sim = PersonaSimulator(seed=args.seed)

    if args.blend:
        name_a, name_b, ratio_str = args.blend
        ratio = float(ratio_str)
        hybrid = sim.blend(name_a, name_b, ratio)
        result = sim.simulate_persona(hybrid)
        if args.json_output:
            print(json.dumps({
                "persona": hybrid.name,
                "description": hybrid.description,
                "traits": hybrid.traits,
                "risk_score": result.risk_score,
                "risk_level": result.risk_level,
                "workers": len(result.report.workers) if result.report.workers else 0,
                "replications_succeeded": result.report.total_replications_succeeded,
                "replications_denied": result.report.total_replications_denied,
            }, indent=2))
        else:
            comp = PersonaComparisonResult(results=[result], title=f"Blended Persona: {hybrid.name}")
            print(comp.render())
        return

    comparison = sim.simulate_all(names=args.personas)

    if args.json_output:
        print(json.dumps(comparison.to_dict(), indent=2))
    else:
        print(comparison.render(rank=args.rank))


if __name__ == "__main__":
    main()
