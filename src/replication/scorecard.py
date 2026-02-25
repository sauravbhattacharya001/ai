"""Unified Safety Scorecard — multi-dimensional safety assessment.

Orchestrates simulation, threat analysis, Monte Carlo risk analysis,
and policy evaluation to produce a comprehensive safety scorecard
with letter grades (A+ through F) per dimension and an overall grade.

Usage (CLI)::

    python -m replication.scorecard                           # default config
    python -m replication.scorecard --scenario balanced       # from preset
    python -m replication.scorecard --strategy greedy --max-depth 5
    python -m replication.scorecard --mc-runs 50              # Monte Carlo runs
    python -m replication.scorecard --policy strict           # policy preset
    python -m replication.scorecard --json                    # JSON output
    python -m replication.scorecard --quick                   # skip slow analyses

Programmatic::

    from replication.scorecard import SafetyScorecard, ScorecardConfig
    sc = SafetyScorecard()
    result = sc.evaluate(ScenarioConfig(strategy="greedy", max_depth=5))
    print(result.render())
    print(f"Overall: {result.overall_grade} ({result.overall_score}/100)")
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .simulator import Simulator, ScenarioConfig, SimulationReport, PRESETS
from .threats import ThreatSimulator, ThreatConfig, ThreatReport, MitigationStatus
from .montecarlo import MonteCarloAnalyzer, MonteCarloConfig, MonteCarloResult
from .policy import SafetyPolicy, PolicyResult, Severity


# ── Grading ─────────────────────────────────────────────────────────────


def _grade(score: float) -> str:
    """Convert a 0-100 score to a letter grade."""
    if score >= 97:
        return "A+"
    if score >= 93:
        return "A"
    if score >= 90:
        return "A-"
    if score >= 87:
        return "B+"
    if score >= 83:
        return "B"
    if score >= 80:
        return "B-"
    if score >= 77:
        return "C+"
    if score >= 73:
        return "C"
    if score >= 70:
        return "C-"
    if score >= 67:
        return "D+"
    if score >= 63:
        return "D"
    if score >= 60:
        return "D-"
    return "F"


def _box_header(title: str, width: int = 57) -> List[str]:
    """Create a box-drawing header with centered title."""
    inner = width - 2
    return [
        "┌" + "─" * inner + "┐",
        "│" + title.center(inner) + "│",
        "└" + "─" * inner + "┘",
    ]


# ── Data models ─────────────────────────────────────────────────────────


@dataclass
class DimensionScore:
    """Score for a single safety dimension."""

    name: str
    score: float  # 0-100
    grade: str  # A+, A, A-, B+, ..., F
    weight: float  # contribution to overall (0-1, must sum to 1)
    details: List[str]  # specific observations
    metrics: Dict[str, float]  # underlying metric values


@dataclass
class ScorecardConfig:
    """Configuration for scorecard evaluation."""

    mc_runs: int = 30
    policy_preset: str = "standard"
    skip_threats: bool = False
    skip_monte_carlo: bool = False
    quick: bool = False  # skip slow analyses (threats + MC)
    weights: Optional[Dict[str, float]] = None  # custom dimension weights


@dataclass
class ScorecardResult:
    """Complete safety scorecard."""

    config: ScenarioConfig
    dimensions: List[DimensionScore]
    overall_score: float  # weighted average 0-100
    overall_grade: str
    timestamp: str
    duration_ms: float
    recommendations: List[str]
    # Raw results for drill-down
    simulation: Optional[SimulationReport]
    threats: Optional[ThreatReport]
    monte_carlo: Optional[MonteCarloResult]
    policy: Optional[PolicyResult]

    def render(self) -> str:
        """Full scorecard with all dimensions."""
        lines: List[str] = []
        lines.extend(_box_header("🛡️  Safety Scorecard  🛡️"))
        lines.append("")
        lines.append(
            f"  Overall Grade: {self.overall_grade}  ({self.overall_score}/100)"
        )
        lines.append(
            f"  Configuration: {self.config.strategy}, "
            f"depth={self.config.max_depth}, "
            f"replicas={self.config.max_replicas}"
        )
        lines.append("")

        # Dimension table
        lines.append(
            f"  {'Dimension':<24} {'Score':>6} {'Grade':>6}  Details"
        )
        lines.append("  " + "─" * 70)
        for d in self.dimensions:
            detail = d.details[0] if d.details else ""
            filled = int(d.score / 5)
            bar = "█" * filled + "░" * (20 - filled)
            lines.append(
                f"  {d.name:<24} {d.score:>5.1f} {d.grade:>6}  {bar}  {detail}"
            )

        lines.append("")
        if self.recommendations:
            lines.append("  Recommendations:")
            for r in self.recommendations[:5]:
                lines.append(f"    • {r}")

        return "\n".join(lines)

    def render_summary(self) -> str:
        """Short summary with overall grade and dimension names."""
        lines: List[str] = []
        lines.append(
            f"Safety Scorecard: {self.overall_grade} ({self.overall_score}/100)"
        )
        for d in self.dimensions:
            lines.append(f"  {d.name}: {d.grade} ({d.score})")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Export as a JSON-serializable dictionary."""
        return {
            "overall_score": self.overall_score,
            "overall_grade": self.overall_grade,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "config": {
                "strategy": self.config.strategy,
                "max_depth": self.config.max_depth,
                "max_replicas": self.config.max_replicas,
            },
            "dimensions": [
                {
                    "name": d.name,
                    "score": d.score,
                    "grade": d.grade,
                    "weight": d.weight,
                    "details": d.details,
                    "metrics": d.metrics,
                }
                for d in self.dimensions
            ],
            "recommendations": self.recommendations,
        }


# ── Default weights ─────────────────────────────────────────────────────

_DEFAULT_WEIGHTS: Dict[str, float] = {
    "Containment": 0.25,
    "Threat Resilience": 0.20,
    "Statistical Risk": 0.20,
    "Policy Compliance": 0.20,
    "Operational Safety": 0.15,
}


# ── SafetyScorecard ─────────────────────────────────────────────────────


class SafetyScorecard:
    """Orchestrates multi-dimensional safety assessment."""

    def __init__(self, config: Optional[ScorecardConfig] = None) -> None:
        self.config = config or ScorecardConfig()

    # ── public API ──────────────────────────────────────────────────

    def evaluate(self, scenario: ScenarioConfig) -> ScorecardResult:
        """Run all analyses and produce a unified scorecard."""
        start = time.monotonic()

        # 1. Run simulation
        sim = Simulator(scenario)
        report = sim.run()

        # 2. Containment score
        containment = self._score_containment(report, scenario)

        # 3. Threat resilience (optional)
        threats_result: Optional[ThreatReport] = None
        if not self.config.skip_threats and not self.config.quick:
            threat_cfg = ThreatConfig(
                max_depth=scenario.max_depth,
                max_replicas=scenario.max_replicas,
                cooldown_seconds=scenario.cooldown_seconds,
            )
            threat_sim = ThreatSimulator(threat_cfg)
            threats_result = threat_sim.run_all()
        threat_score = self._score_threats(threats_result)

        # 4. Monte Carlo risk (optional)
        mc_result: Optional[MonteCarloResult] = None
        if not self.config.skip_monte_carlo and not self.config.quick:
            mc_cfg = MonteCarloConfig(
                num_runs=self.config.mc_runs,
                base_scenario=scenario,
            )
            mc = MonteCarloAnalyzer(mc_cfg)
            mc_result = mc.analyze()
        mc_score = self._score_monte_carlo(mc_result)

        # 5. Policy compliance
        policy = SafetyPolicy.from_preset(self.config.policy_preset)
        policy_result = policy.evaluate(report, mc_result)
        policy_score_dim = self._score_policy(policy_result)

        # 6. Operational safety
        ops_score = self._score_operational(report, scenario)

        dimensions = [
            containment,
            threat_score,
            mc_score,
            policy_score_dim,
            ops_score,
        ]

        # Overall weighted average
        weights = self._get_weights()
        overall = sum(d.score * weights.get(d.name, 0.2) for d in dimensions)

        # Recommendations
        recs = self._generate_recommendations(
            dimensions, report, threats_result, mc_result, policy_result
        )

        duration_ms = (time.monotonic() - start) * 1000

        return ScorecardResult(
            config=scenario,
            dimensions=dimensions,
            overall_score=round(overall, 1),
            overall_grade=_grade(overall),
            timestamp=datetime.now(timezone.utc).isoformat(),
            duration_ms=round(duration_ms, 1),
            recommendations=recs,
            simulation=report,
            threats=threats_result,
            monte_carlo=mc_result,
            policy=policy_result,
        )

    # ── weight helpers ──────────────────────────────────────────────

    def _get_weights(self) -> Dict[str, float]:
        """Return dimension weights, using custom if provided."""
        if self.config.weights:
            return self.config.weights
        return dict(_DEFAULT_WEIGHTS)

    # ── dimension scoring ───────────────────────────────────────────

    def _score_containment(
        self, report: SimulationReport, config: ScenarioConfig
    ) -> DimensionScore:
        """Score containment based on depth/worker utilization and denial rate."""
        max_depth = max(
            (w.depth for w in report.workers.values()), default=0
        )
        depth_ratio = (
            max_depth / config.max_depth if config.max_depth > 0 else 0.0
        )
        worker_ratio = (
            len(report.workers) / config.max_replicas
            if config.max_replicas > 0
            else 0.0
        )
        attempted = report.total_replications_attempted
        denial_rate = (
            report.total_replications_denied / attempted
            if attempted > 0
            else 1.0
        )

        depth_score = max(0.0, 100 - depth_ratio * 60)
        worker_score = max(0.0, 100 - worker_ratio * 50)
        denial_score = denial_rate * 100

        containment = (
            depth_score * 0.35 + worker_score * 0.35 + denial_score * 0.30
        )
        containment = round(min(100.0, max(0.0, containment)), 1)

        details: List[str] = []
        if depth_ratio < 0.5:
            details.append("Depth well within limits")
        elif depth_ratio < 1.0:
            details.append(f"Depth utilization {depth_ratio:.0%}")
        else:
            details.append("Max depth reached")

        if denial_rate > 0.5:
            details.append(f"Denial rate {denial_rate:.0%} — controls active")
        elif attempted > 0:
            details.append(f"Denial rate {denial_rate:.0%}")

        return DimensionScore(
            name="Containment",
            score=containment,
            grade=_grade(containment),
            weight=0.25,
            details=details,
            metrics={
                "depth_ratio": round(depth_ratio, 4),
                "worker_ratio": round(worker_ratio, 4),
                "denial_rate": round(denial_rate, 4),
            },
        )

    def _score_threats(
        self, threats: Optional[ThreatReport]
    ) -> DimensionScore:
        """Score threat resilience from ThreatReport.security_score."""
        if threats is None:
            return DimensionScore(
                name="Threat Resilience",
                score=50.0,
                grade=_grade(50.0),
                weight=0.20,
                details=["Threat analysis skipped — neutral score"],
                metrics={"security_score": 50.0, "skipped": 1.0},
            )

        score = threats.security_score
        details: List[str] = []
        mitigated = threats.total_mitigated
        total = len(threats.results)
        failed = threats.total_failed

        if failed == 0:
            details.append(f"All {total} threats mitigated")
        else:
            details.append(f"{mitigated}/{total} threats mitigated, {failed} failed")

        return DimensionScore(
            name="Threat Resilience",
            score=round(min(100.0, max(0.0, score)), 1),
            grade=_grade(score),
            weight=0.20,
            details=details,
            metrics={
                "security_score": round(score, 2),
                "mitigated": float(mitigated),
                "failed": float(failed),
                "total": float(total),
                "skipped": 0.0,
            },
        )

    def _score_monte_carlo(
        self, mc: Optional[MonteCarloResult]
    ) -> DimensionScore:
        """Score statistical risk from Monte Carlo results."""
        if mc is None:
            return DimensionScore(
                name="Statistical Risk",
                score=50.0,
                grade=_grade(50.0),
                weight=0.20,
                details=["Monte Carlo analysis skipped — neutral score"],
                metrics={"risk_level_score": 50.0, "skipped": 1.0},
            )

        risk_level = mc.risk_level
        level_scores = {
            "LOW": 95.0,
            "MODERATE": 75.0,
            "ELEVATED": 55.0,
            "HIGH": 30.0,
            "CRITICAL": 10.0,
        }
        base = level_scores.get(risk_level, 50.0)

        # Blend with probabilities for more granularity
        prob_depth = mc.risk_metrics.prob_max_depth_reached
        prob_quota = mc.risk_metrics.prob_quota_saturated
        prob_penalty = (prob_depth + prob_quota) / 2 * 20  # up to -20 pts
        score = max(0.0, min(100.0, base - prob_penalty))
        score = round(score, 1)

        details: List[str] = [f"Risk level: {risk_level}"]
        if prob_depth > 0.5:
            details.append(
                f"High probability of max depth breach ({prob_depth:.0%})"
            )
        if prob_quota > 0.5:
            details.append(
                f"High probability of quota saturation ({prob_quota:.0%})"
            )

        return DimensionScore(
            name="Statistical Risk",
            score=score,
            grade=_grade(score),
            weight=0.20,
            details=details,
            metrics={
                "risk_level_score": base,
                "prob_max_depth": round(prob_depth, 4),
                "prob_quota_saturated": round(prob_quota, 4),
                "skipped": 0.0,
            },
        )

    def _score_policy(self, policy_result: PolicyResult) -> DimensionScore:
        """Score policy compliance from pass rate and severity."""
        total = len(policy_result.rule_results)
        if total == 0:
            return DimensionScore(
                name="Policy Compliance",
                score=100.0,
                grade="A+",
                weight=0.20,
                details=["No rules to evaluate"],
                metrics={"pass_rate": 1.0, "total_rules": 0.0},
            )

        passed = len(policy_result.passes)
        pass_rate = passed / total
        base_score = pass_rate * 100

        # Penalize by severity of failures
        error_count = len(policy_result.errors)
        warning_count = len(policy_result.warnings)
        severity_penalty = error_count * 5 + warning_count * 2
        score = max(0.0, min(100.0, base_score - severity_penalty))
        score = round(score, 1)

        details: List[str] = []
        if policy_result.passed and not policy_result.has_warnings:
            details.append(f"All {total} rules passed")
        elif policy_result.passed:
            details.append(
                f"{passed}/{total} passed, {warning_count} warnings"
            )
        else:
            details.append(
                f"{passed}/{total} passed, {error_count} errors, "
                f"{warning_count} warnings"
            )

        return DimensionScore(
            name="Policy Compliance",
            score=score,
            grade=_grade(score),
            weight=0.20,
            details=details,
            metrics={
                "pass_rate": round(pass_rate, 4),
                "total_rules": float(total),
                "errors": float(error_count),
                "warnings": float(warning_count),
            },
        )

    def _score_operational(
        self, report: SimulationReport, config: ScenarioConfig
    ) -> DimensionScore:
        """Score operational safety: efficiency and resource utilization."""
        n_workers = len(report.workers)
        total_tasks = report.total_tasks

        # Tasks per worker (higher = more efficient, but cap at tasks_per_worker)
        if n_workers > 0:
            tasks_per_worker = total_tasks / n_workers
            expected = config.tasks_per_worker
            efficiency = min(1.0, tasks_per_worker / expected) if expected > 0 else 1.0
        else:
            tasks_per_worker = 0.0
            efficiency = 1.0  # no workers = safe (nothing happened)

        # Completion ratio: did workers complete their tasks?
        expected_total = n_workers * config.tasks_per_worker if n_workers > 0 else 1
        completion = min(1.0, total_tasks / expected_total) if expected_total > 0 else 1.0

        # Resource utilization score: fewer workers = better from safety perspective
        worker_ratio = (
            n_workers / config.max_replicas if config.max_replicas > 0 else 0.0
        )
        resource_score = max(0.0, 100 - worker_ratio * 40)

        # Blend efficiency, completion, and resource management
        score = efficiency * 35 + completion * 35 + resource_score * 0.30
        score = round(min(100.0, max(0.0, score)), 1)

        details: List[str] = []
        if efficiency >= 0.8:
            details.append(f"Good efficiency ({tasks_per_worker:.1f} tasks/worker)")
        else:
            details.append(
                f"Low efficiency ({tasks_per_worker:.1f} tasks/worker)"
            )

        if completion >= 0.9:
            details.append("High task completion")
        elif completion >= 0.5:
            details.append(f"Task completion {completion:.0%}")

        return DimensionScore(
            name="Operational Safety",
            score=score,
            grade=_grade(score),
            weight=0.15,
            details=details,
            metrics={
                "tasks_per_worker": round(tasks_per_worker, 2),
                "efficiency": round(efficiency, 4),
                "completion": round(completion, 4),
                "worker_ratio": round(worker_ratio, 4),
            },
        )

    # ── recommendations ─────────────────────────────────────────────

    def _generate_recommendations(
        self,
        dimensions: List[DimensionScore],
        report: SimulationReport,
        threats: Optional[ThreatReport],
        mc: Optional[MonteCarloResult],
        policy: Optional[PolicyResult],
    ) -> List[str]:
        """Generate actionable recommendations based on scores."""
        recs: List[str] = []

        dim_map = {d.name: d for d in dimensions}

        # Containment
        cont = dim_map.get("Containment")
        if cont and cont.score < 70:
            recs.append(
                "Lower max_depth or max_replicas to improve containment"
            )
            if cont.metrics.get("denial_rate", 0) < 0.3:
                recs.append(
                    "Denial rate is low — consider stricter contract limits"
                )

        # Threats
        threat_dim = dim_map.get("Threat Resilience")
        if threat_dim and threat_dim.score < 70:
            if threats is not None and threats.total_failed > 0:
                recs.append(
                    f"{threats.total_failed} threat(s) not mitigated — "
                    "review contract configuration"
                )
            else:
                recs.append(
                    "Threat resilience is low — enable threat analysis for details"
                )

        # Statistical Risk
        mc_dim = dim_map.get("Statistical Risk")
        if mc_dim and mc_dim.score < 70:
            recs.append(
                "Statistical risk is elevated — consider more conservative settings"
            )
            if mc and mc.risk_metrics.prob_max_depth_reached > 0.5:
                recs.append("Reduce max_depth to lower breach probability")

        # Policy
        pol_dim = dim_map.get("Policy Compliance")
        if pol_dim and pol_dim.score < 70:
            if policy:
                error_count = len(policy.errors)
                if error_count > 0:
                    recs.append(
                        f"{error_count} policy error(s) — fix violations before deployment"
                    )

        # Operational
        ops_dim = dim_map.get("Operational Safety")
        if ops_dim and ops_dim.score < 70:
            recs.append(
                "Operational efficiency is low — review task allocation strategy"
            )

        return recs


# ── CLI ─────────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for Safety Scorecard."""
    parser = argparse.ArgumentParser(
        description="Safety Scorecard — unified assessment"
    )
    # Standard config args
    parser.add_argument(
        "--scenario",
        choices=["minimal", "balanced", "stress", "chain", "burst"],
        help="Use a built-in scenario preset",
    )
    parser.add_argument(
        "--strategy",
        choices=["greedy", "conservative", "random", "chain", "burst"],
    )
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--max-replicas", type=int)
    parser.add_argument("--cooldown", type=float)
    parser.add_argument("--tasks", type=int)
    parser.add_argument("--seed", type=int)
    # Scorecard-specific
    parser.add_argument("--mc-runs", type=int, default=30)
    parser.add_argument(
        "--policy",
        default="standard",
        choices=["minimal", "standard", "strict", "ci"],
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip threat + Monte Carlo (faster)",
    )
    parser.add_argument("--skip-threats", action="store_true")
    parser.add_argument("--skip-mc", action="store_true")
    parser.add_argument(
        "--json", dest="json_output", action="store_true"
    )
    parser.add_argument("--summary-only", action="store_true")

    args = parser.parse_args()

    # Build scenario config
    if args.scenario:
        scenario = PRESETS[args.scenario]
        # Allow overrides on top of preset
        if args.strategy:
            scenario = ScenarioConfig(
                **{
                    **scenario.__dict__,
                    "strategy": args.strategy,
                }
            )
    else:
        scenario = ScenarioConfig()

    if args.strategy and not args.scenario:
        scenario = ScenarioConfig(**{**scenario.__dict__, "strategy": args.strategy})
    if args.max_depth is not None:
        scenario = ScenarioConfig(**{**scenario.__dict__, "max_depth": args.max_depth})
    if args.max_replicas is not None:
        scenario = ScenarioConfig(
            **{**scenario.__dict__, "max_replicas": args.max_replicas}
        )
    if args.cooldown is not None:
        scenario = ScenarioConfig(
            **{**scenario.__dict__, "cooldown_seconds": args.cooldown}
        )
    if args.tasks is not None:
        scenario = ScenarioConfig(
            **{**scenario.__dict__, "tasks_per_worker": args.tasks}
        )
    if args.seed is not None:
        scenario = ScenarioConfig(**{**scenario.__dict__, "seed": args.seed})

    # Build scorecard config
    sc_config = ScorecardConfig(
        mc_runs=args.mc_runs,
        policy_preset=args.policy,
        skip_threats=args.skip_threats,
        skip_monte_carlo=args.skip_mc,
        quick=args.quick,
    )

    scorecard = SafetyScorecard(sc_config)
    result = scorecard.evaluate(scenario)

    if args.json_output:
        print(json.dumps(result.to_dict(), indent=2))
    elif args.summary_only:
        print(result.render_summary())
    else:
        print(result.render())


if __name__ == "__main__":
    main()
