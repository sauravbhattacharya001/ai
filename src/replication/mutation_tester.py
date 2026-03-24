"""Mutation Tester — verify safety policies catch real violations.

Applies small mutations to safety policies (flip operators, shift thresholds,
disable rules, swap severities) and checks whether the safety system
detects the change.  A mutation that *still passes* is a "survivor" —
meaning the policy has a blind spot.

This is the safety-policy equivalent of mutation testing in software:
if you break a rule and nobody notices, the rule wasn't really protecting
anything.

CLI usage::

    python -m replication mutate                           # test all presets
    python -m replication mutate --preset strict           # test one preset
    python -m replication mutate --file my_policy.json     # test custom policy
    python -m replication mutate --strategy greedy         # specific sim strategy
    python -m replication mutate --runs 50                 # Monte Carlo runs per mutant
    python -m replication mutate --json                    # JSON output
    python -m replication mutate --survivors-only          # only show surviving mutants
    python -m replication mutate --seed 42                 # reproducible mutations

Programmatic::

    from replication.mutation_tester import MutationTester, MutationConfig
    from replication.policy import SafetyPolicy

    policy = SafetyPolicy.from_preset("strict")
    tester = MutationTester(MutationConfig(runs=50))
    report = tester.test(policy)
    print(report.render())
    print(f"Mutation score: {report.mutation_score:.0%}")
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .policy import (
    Operator,
    PolicyRule,
    SafetyPolicy,
    Severity,
    POLICY_PRESETS,
)
from .simulator import ScenarioConfig, Simulator, Strategy, PRESETS as SIM_PRESETS


# ── Mutation types ───────────────────────────────────────────────────

class MutationType(str, Enum):
    """Kinds of mutations we can apply to a policy rule."""

    FLIP_OPERATOR = "flip_operator"
    RELAX_THRESHOLD = "relax_threshold"
    TIGHTEN_THRESHOLD = "tighten_threshold"
    REMOVE_RULE = "remove_rule"
    DOWNGRADE_SEVERITY = "downgrade_severity"
    NEGATE_OPERATOR = "negate_operator"


# Operator flips — each operator maps to a "dangerous" counterpart
_OPERATOR_FLIPS: Dict[Operator, Operator] = {
    Operator.LT: Operator.GT,
    Operator.LE: Operator.GE,
    Operator.GT: Operator.LT,
    Operator.GE: Operator.LE,
    Operator.EQ: Operator.NE,
    Operator.NE: Operator.EQ,
}

# Negation: makes the constraint go the opposite direction
_OPERATOR_NEGATIONS: Dict[Operator, Operator] = {
    Operator.LT: Operator.GE,
    Operator.LE: Operator.GT,
    Operator.GT: Operator.LE,
    Operator.GE: Operator.LT,
    Operator.EQ: Operator.NE,
    Operator.NE: Operator.EQ,
}


# ── Models ───────────────────────────────────────────────────────────

@dataclass
class MutationConfig:
    """Configuration for a mutation testing session."""

    strategy: str = "greedy"
    runs: int = 0  # 0 = single run (no Monte Carlo)
    seed: Optional[int] = None
    survivors_only: bool = False


@dataclass
class Mutation:
    """A single mutation applied to a policy."""

    mutation_type: MutationType
    rule_index: int
    original_rule: Dict[str, Any]
    mutated_rule: Optional[Dict[str, Any]]  # None for REMOVE_RULE
    description: str


@dataclass
class MutantResult:
    """Result of testing one mutant policy."""

    mutation: Mutation
    killed: bool  # True = safety system caught it (good!)
    original_passed: bool
    mutant_passed: bool
    error: Optional[str] = None


@dataclass
class MutationReport:
    """Aggregated mutation testing results."""

    policy_name: str
    total_mutants: int = 0
    killed: int = 0
    survived: int = 0
    errors: int = 0
    results: List[MutantResult] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def mutation_score(self) -> float:
        """Fraction of mutants killed (0–1). Higher = better policy."""
        testable = self.total_mutants - self.errors
        return (self.killed / testable) if testable > 0 else 0.0

    def render(self, survivors_only: bool = False) -> str:
        """Render a human-readable report."""
        lines = [
            f"\n🧬  Mutation Testing Report — {self.policy_name}",
            f"{'─' * 60}",
            f"  Total mutants : {self.total_mutants}",
            f"  Killed (good) : {self.killed}  ✅",
            f"  Survived (bad): {self.survived}  {'⚠️' if self.survived else ''}",
            f"  Errors        : {self.errors}",
            f"  Mutation score: {self.mutation_score:.0%}",
            f"  Elapsed       : {self.elapsed_ms:.0f} ms",
            f"{'─' * 60}",
        ]

        for r in self.results:
            if survivors_only and r.killed:
                continue
            icon = "✅" if r.killed else ("💥" if r.error else "🧟")
            status = "KILLED" if r.killed else ("ERROR" if r.error else "SURVIVED")
            lines.append(f"  {icon} [{status}] {r.mutation.description}")
            if r.error:
                lines.append(f"       Error: {r.error}")
            elif not r.killed:
                lines.append(f"       ⚠️  Policy blind spot — mutant was not detected!")

        if self.survived == 0 and self.errors == 0:
            lines.append("\n  🎉  Perfect score! All mutants were caught.")
        elif self.survived > 0:
            lines.append(
                f"\n  ⚠️  {self.survived} mutant(s) survived — "
                f"consider strengthening these policy rules."
            )

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "total_mutants": self.total_mutants,
            "killed": self.killed,
            "survived": self.survived,
            "errors": self.errors,
            "mutation_score": round(self.mutation_score, 4),
            "elapsed_ms": round(self.elapsed_ms, 1),
            "results": [
                {
                    "mutation": {
                        "type": r.mutation.mutation_type.value,
                        "rule_index": r.mutation.rule_index,
                        "description": r.mutation.description,
                        "original_rule": r.mutation.original_rule,
                        "mutated_rule": r.mutation.mutated_rule,
                    },
                    "killed": r.killed,
                    "error": r.error,
                }
                for r in self.results
            ],
        }


# ── Mutant generators ───────────────────────────────────────────────

def _generate_mutants(policy: SafetyPolicy) -> List[tuple[Mutation, SafetyPolicy]]:
    """Generate all possible single-point mutants of a policy."""
    mutants: List[tuple[Mutation, SafetyPolicy]] = []

    for i, rule in enumerate(policy.rules):
        # 1. Flip operator
        flipped_op = _OPERATOR_FLIPS.get(rule.operator)
        if flipped_op and flipped_op != rule.operator:
            mutated = copy.deepcopy(policy)
            mutated.rules[i] = PolicyRule(
                metric=rule.metric,
                operator=flipped_op,
                threshold=rule.threshold,
                severity=rule.severity,
                description=rule.description,
                monte_carlo=rule.monte_carlo,
            )
            mutants.append((
                Mutation(
                    mutation_type=MutationType.FLIP_OPERATOR,
                    rule_index=i,
                    original_rule=rule.to_dict(),
                    mutated_rule=mutated.rules[i].to_dict(),
                    description=(
                        f"Rule {i}: flip operator "
                        f"{rule.operator.value} → {flipped_op.value} "
                        f"on '{rule.metric}'"
                    ),
                ),
                mutated,
            ))

        # 2. Negate operator
        negated_op = _OPERATOR_NEGATIONS.get(rule.operator)
        if negated_op and negated_op != flipped_op:
            mutated = copy.deepcopy(policy)
            mutated.rules[i] = PolicyRule(
                metric=rule.metric,
                operator=negated_op,
                threshold=rule.threshold,
                severity=rule.severity,
                description=rule.description,
                monte_carlo=rule.monte_carlo,
            )
            mutants.append((
                Mutation(
                    mutation_type=MutationType.NEGATE_OPERATOR,
                    rule_index=i,
                    original_rule=rule.to_dict(),
                    mutated_rule=mutated.rules[i].to_dict(),
                    description=(
                        f"Rule {i}: negate operator "
                        f"{rule.operator.value} → {negated_op.value} "
                        f"on '{rule.metric}'"
                    ),
                ),
                mutated,
            ))

        # 3. Relax threshold (make it easier to pass)
        if rule.threshold != 0:
            factor = 2.0 if rule.operator in (Operator.LT, Operator.LE) else 0.5
            relaxed_threshold = rule.threshold * factor
        else:
            relaxed_threshold = 1.0

        mutated = copy.deepcopy(policy)
        mutated.rules[i] = PolicyRule(
            metric=rule.metric,
            operator=rule.operator,
            threshold=relaxed_threshold,
            severity=rule.severity,
            description=rule.description,
            monte_carlo=rule.monte_carlo,
        )
        mutants.append((
            Mutation(
                mutation_type=MutationType.RELAX_THRESHOLD,
                rule_index=i,
                original_rule=rule.to_dict(),
                mutated_rule=mutated.rules[i].to_dict(),
                description=(
                    f"Rule {i}: relax threshold "
                    f"{rule.threshold} → {relaxed_threshold} "
                    f"on '{rule.metric}'"
                ),
            ),
            mutated,
        ))

        # 4. Remove rule entirely
        mutated = copy.deepcopy(policy)
        mutated.rules.pop(i)
        mutants.append((
            Mutation(
                mutation_type=MutationType.REMOVE_RULE,
                rule_index=i,
                original_rule=rule.to_dict(),
                mutated_rule=None,
                description=(
                    f"Rule {i}: remove rule "
                    f"'{rule.metric} {rule.operator.value} {rule.threshold}'"
                ),
            ),
            mutated,
        ))

        # 5. Downgrade severity (error → warning, warning → info)
        if rule.severity == Severity.ERROR:
            mutated = copy.deepcopy(policy)
            mutated.rules[i] = PolicyRule(
                metric=rule.metric,
                operator=rule.operator,
                threshold=rule.threshold,
                severity=Severity.WARNING,
                description=rule.description,
                monte_carlo=rule.monte_carlo,
            )
            mutants.append((
                Mutation(
                    mutation_type=MutationType.DOWNGRADE_SEVERITY,
                    rule_index=i,
                    original_rule=rule.to_dict(),
                    mutated_rule=mutated.rules[i].to_dict(),
                    description=(
                        f"Rule {i}: downgrade severity "
                        f"error → warning on '{rule.metric}'"
                    ),
                ),
                mutated,
            ))
        elif rule.severity == Severity.WARNING:
            mutated = copy.deepcopy(policy)
            mutated.rules[i] = PolicyRule(
                metric=rule.metric,
                operator=rule.operator,
                threshold=rule.threshold,
                severity=Severity.INFO,
                description=rule.description,
                monte_carlo=rule.monte_carlo,
            )
            mutants.append((
                Mutation(
                    mutation_type=MutationType.DOWNGRADE_SEVERITY,
                    rule_index=i,
                    original_rule=rule.to_dict(),
                    mutated_rule=mutated.rules[i].to_dict(),
                    description=(
                        f"Rule {i}: downgrade severity "
                        f"warning → info on '{rule.metric}'"
                    ),
                ),
                mutated,
            ))

    return mutants


# ── Tester ───────────────────────────────────────────────────────────

class MutationTester:
    """Run mutation testing against a safety policy."""

    def __init__(self, config: Optional[MutationConfig] = None) -> None:
        self.config = config or MutationConfig()

    def _evaluate_policy(self, policy: SafetyPolicy) -> bool:
        """Evaluate a policy against simulation results. Returns True if passed."""
        strategy_name = self.config.strategy
        if strategy_name not in SIM_PRESETS:
            strategy_name = "greedy"

        cfg = ScenarioConfig(strategy=Strategy(strategy_name))

        if self.config.runs > 0:
            from .montecarlo import MonteCarloAnalyzer, MonteCarloConfig
            mc_cfg = MonteCarloConfig(
                base_scenario=cfg,
                num_runs=self.config.runs,
            )
            mc = MonteCarloAnalyzer(mc_cfg)
            mc_result = mc.analyze()
            result = policy.evaluate_mc(mc_result)
        else:
            # Filter out monte_carlo rules when not doing MC runs
            filtered = copy.deepcopy(policy)
            filtered.rules = [r for r in filtered.rules if not r.monte_carlo]
            if not filtered.rules:
                return True  # No applicable rules
            sim = Simulator(cfg)
            report = sim.run()
            result = filtered.evaluate(report)

        return result.passed

    def test(self, policy: SafetyPolicy, name: str = "policy") -> MutationReport:
        """Run mutation testing on a policy. Returns a MutationReport."""
        if self.config.seed is not None:
            random.seed(self.config.seed)

        # When not doing MC runs, work only with non-MC rules
        working_policy = copy.deepcopy(policy)
        if self.config.runs == 0:
            working_policy.rules = [r for r in working_policy.rules if not r.monte_carlo]

        t0 = time.time()
        report = MutationReport(policy_name=name)

        # First verify the original policy passes
        try:
            original_passed = self._evaluate_policy(working_policy)
        except Exception as e:
            report.errors = 1
            report.total_mutants = 0
            report.elapsed_ms = (time.time() - t0) * 1000
            return report

        mutants = _generate_mutants(working_policy)
        report.total_mutants = len(mutants)

        for mutation, mutant_policy in mutants:
            try:
                mutant_passed = self._evaluate_policy(mutant_policy)
                # A mutant is "killed" if the mutated policy behaves differently
                # from the original (ideally: original passes, mutant fails).
                # If the original also failed, we can't tell, so we count as killed.
                killed = not original_passed or (original_passed and not mutant_passed)
                result = MutantResult(
                    mutation=mutation,
                    killed=killed,
                    original_passed=original_passed,
                    mutant_passed=mutant_passed,
                )
            except Exception as e:
                result = MutantResult(
                    mutation=mutation,
                    killed=False,
                    original_passed=original_passed,
                    mutant_passed=False,
                    error=str(e),
                )
                report.errors += 1

            if result.killed:
                report.killed += 1
            elif not result.error:
                report.survived += 1

            report.results.append(result)

        report.elapsed_ms = (time.time() - t0) * 1000
        return report


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m replication mutate",
        description="Mutation testing for safety policies",
    )
    parser.add_argument(
        "--preset", "-p",
        choices=list(POLICY_PRESETS.keys()),
        help="Policy preset to test",
    )
    parser.add_argument(
        "--file", "-f",
        help="Path to custom policy JSON file",
    )
    parser.add_argument(
        "--strategy", "-s",
        default="greedy",
        help="Simulation strategy (default: greedy)",
    )
    parser.add_argument(
        "--runs", "-r",
        type=int,
        default=0,
        help="Monte Carlo runs per mutant (0 = single run)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        dest="json_output",
        help="Output as JSON",
    )
    parser.add_argument(
        "--survivors-only",
        action="store_true",
        help="Only show surviving mutants",
    )

    args = parser.parse_args(argv)

    config = MutationConfig(
        strategy=args.strategy,
        runs=args.runs,
        seed=args.seed,
        survivors_only=args.survivors_only,
    )
    tester = MutationTester(config)

    if args.file:
        with open(args.file) as f:
            data = json.load(f)
        policy = SafetyPolicy.from_dict(data)
        reports = [tester.test(policy, name=args.file)]
    elif args.preset:
        policy = SafetyPolicy.from_preset(args.preset)
        reports = [tester.test(policy, name=args.preset)]
    else:
        # Test all presets
        reports = []
        for name in sorted(POLICY_PRESETS.keys()):
            policy = SafetyPolicy.from_preset(name)
            reports.append(tester.test(policy, name=name))

    if args.json_output:
        output = [r.to_dict() for r in reports]
        print(json.dumps(output if len(output) > 1 else output[0], indent=2))
    else:
        for report in reports:
            print(report.render(survivors_only=args.survivors_only))

        # Summary if multiple
        if len(reports) > 1:
            total_killed = sum(r.killed for r in reports)
            total_mutants = sum(r.total_mutants for r in reports)
            total_survived = sum(r.survived for r in reports)
            avg_score = (
                sum(r.mutation_score for r in reports) / len(reports)
                if reports else 0
            )
            print(f"\n{'═' * 60}")
            print(f"  Overall: {total_killed}/{total_mutants} killed, "
                  f"{total_survived} survived")
            print(f"  Average mutation score: {avg_score:.0%}")
            print(f"{'═' * 60}")
