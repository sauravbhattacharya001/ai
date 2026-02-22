"""Safety Policy Engine — declarative rules for automated safety validation.

Define safety policies as sets of rules that simulation results must satisfy.
Enables CI/CD integration: run simulations, validate against policies, and
fail builds when safety invariants are violated.

Usage (CLI)::

    python -m replication.policy --preset strict                      # validate with strict preset
    python -m replication.policy --preset standard --strategy greedy   # test greedy against standard
    python -m replication.policy --file my_policy.json                 # custom policy file
    python -m replication.policy --preset minimal --runs 100           # Monte Carlo validation
    python -m replication.policy --list-presets                        # show available presets
    python -m replication.policy --list-metrics                        # show available metrics
    python -m replication.policy --json                                # JSON output

Programmatic::

    from replication.policy import SafetyPolicy, PolicyRule, PRESETS
    policy = SafetyPolicy.from_preset("strict")
    report = Simulator(ScenarioConfig(strategy="greedy")).run()
    result = policy.evaluate(report)
    print(result.render())
    assert result.passed, f"Safety policy violated: {result.failures}"
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .simulator import ScenarioConfig, SimulationReport, Simulator, Strategy, PRESETS as SIM_PRESETS
from .montecarlo import MonteCarloAnalyzer, MonteCarloConfig, MonteCarloResult


# ── Enums ───────────────────────────────────────────────────────────────

class Operator(Enum):
    """Comparison operators for policy rules."""
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    EQ = "=="
    NE = "!="

    def evaluate(self, actual: float, threshold: float) -> bool:
        """Return True if the rule passes (actual op threshold)."""
        if self == Operator.LT:
            return actual < threshold
        elif self == Operator.LE:
            return actual <= threshold
        elif self == Operator.GT:
            return actual > threshold
        elif self == Operator.GE:
            return actual >= threshold
        elif self == Operator.EQ:
            return abs(actual - threshold) < 1e-9
        elif self == Operator.NE:
            return abs(actual - threshold) >= 1e-9
        return False

    @classmethod
    def from_str(cls, s: str) -> "Operator":
        """Parse an operator from a string."""
        mapping = {
            "<": cls.LT, "lt": cls.LT,
            "<=": cls.LE, "le": cls.LE,
            ">": cls.GT, "gt": cls.GT,
            ">=": cls.GE, "ge": cls.GE,
            "==": cls.EQ, "eq": cls.EQ,
            "!=": cls.NE, "ne": cls.NE,
        }
        key = s.strip().lower()
        if key not in mapping:
            raise ValueError(f"Unknown operator: {s!r}. Use: {', '.join(mapping)}")
        return mapping[key]


class Severity(Enum):
    """Rule severity — determines overall verdict."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# ── Metric extraction ──────────────────────────────────────────────────

def _extract_metric(report: SimulationReport, metric: str) -> float:
    """Extract a named metric from a SimulationReport.

    Supported metrics:
        total_workers       — number of workers spawned
        total_tasks         — number of tasks completed
        max_depth_reached   — deepest depth reached
        repl_succeeded      — successful replications
        repl_denied         — denied replications
        repl_attempted      — total replication attempts
        denial_rate         — denied / attempted (0–1, or 0 if no attempts)
        success_rate        — succeeded / attempted (0–1, or 0 if no attempts)
        depth_utilization   — max_depth_reached / config.max_depth (0–1)
        worker_utilization  — total_workers / config.max_replicas (0–∞)
        duration_ms         — simulation wall-clock time
        avg_tasks_per_worker— total_tasks / total_workers
    """
    max_depth = max((w.depth for w in report.workers.values()), default=0)
    attempted = report.total_replications_attempted

    extractors = {
        "total_workers": lambda: float(len(report.workers)),
        "total_tasks": lambda: float(report.total_tasks),
        "max_depth_reached": lambda: float(max_depth),
        "repl_succeeded": lambda: float(report.total_replications_succeeded),
        "repl_denied": lambda: float(report.total_replications_denied),
        "repl_attempted": lambda: float(attempted),
        "denial_rate": lambda: (report.total_replications_denied / attempted) if attempted > 0 else 0.0,
        "success_rate": lambda: (report.total_replications_succeeded / attempted) if attempted > 0 else 0.0,
        "depth_utilization": lambda: (max_depth / report.config.max_depth) if report.config.max_depth > 0 else 0.0,
        "worker_utilization": lambda: (len(report.workers) / report.config.max_replicas) if report.config.max_replicas > 0 else 0.0,
        "duration_ms": lambda: report.duration_ms,
        "avg_tasks_per_worker": lambda: (report.total_tasks / len(report.workers)) if report.workers else 0.0,
    }

    if metric not in extractors:
        raise ValueError(
            f"Unknown metric: {metric!r}. Available: {', '.join(sorted(extractors))}"
        )
    return extractors[metric]()


def _extract_mc_metric(mc_result: MonteCarloResult, metric: str) -> float:
    """Extract a named metric from a MonteCarloResult.

    Supports all single-run metrics (using mean values from distributions),
    plus Monte Carlo-specific metrics:
        prob_max_depth      — probability of reaching max depth
        prob_quota_saturated— probability of quota saturation
        risk_score          — combined risk (0–1)
        peak_workers_p95    — 95th percentile peak worker count
        peak_depth_p95      — 95th percentile max depth
        mean_workers        — mean worker count across runs
        mean_denial_rate    — mean denial rate across runs
    """
    rm = mc_result.risk_metrics
    mc_extractors = {
        "prob_max_depth": lambda: rm.prob_max_depth_reached,
        "prob_quota_saturated": lambda: rm.prob_quota_saturated,
        "risk_score": lambda: (rm.prob_max_depth_reached + rm.prob_quota_saturated) / 2,
        "peak_workers_p95": lambda: rm.peak_worker_p95,
        "peak_depth_p95": lambda: rm.peak_depth_p95,
        "mean_workers": lambda: mc_result.distributions["total_workers"].mean
            if "total_workers" in mc_result.distributions else 0.0,
        "mean_denial_rate": lambda: mc_result.distributions["denial_rate"].mean
            if "denial_rate" in mc_result.distributions else 0.0,
    }

    if metric in mc_extractors:
        return mc_extractors[metric]()

    # Fall back to single-run metrics using distribution means
    for dist_name, dist in mc_result.distributions.items():
        if dist_name == metric:
            return dist.mean

    raise ValueError(f"Unknown Monte Carlo metric: {metric!r}")


# All available metrics for documentation/listing
SINGLE_RUN_METRICS = {
    "total_workers": "Number of workers spawned",
    "total_tasks": "Number of tasks completed",
    "max_depth_reached": "Deepest replication depth reached",
    "repl_succeeded": "Successful replications",
    "repl_denied": "Denied replications",
    "repl_attempted": "Total replication attempts",
    "denial_rate": "Denied / attempted ratio (0–1)",
    "success_rate": "Succeeded / attempted ratio (0–1)",
    "depth_utilization": "Max depth reached / max depth limit (0–1)",
    "worker_utilization": "Total workers / max replicas limit",
    "duration_ms": "Simulation wall-clock time in ms",
    "avg_tasks_per_worker": "Tasks / workers ratio",
}

MONTE_CARLO_METRICS = {
    "prob_max_depth": "Probability of reaching max depth (0–1)",
    "prob_quota_saturated": "Probability of quota saturation (0–1)",
    "risk_score": "Combined risk score (0–1)",
    "peak_workers_p95": "95th percentile peak worker count",
    "peak_depth_p95": "95th percentile max depth",
    "mean_workers": "Mean worker count across runs",
    "mean_denial_rate": "Mean denial rate across runs",
}


# ── Data models ─────────────────────────────────────────────────────────

@dataclass
class PolicyRule:
    """A single safety rule: metric op threshold."""

    metric: str
    operator: Operator
    threshold: float
    severity: Severity = Severity.ERROR
    description: str = ""
    monte_carlo: bool = False  # True = evaluate against MC results

    def evaluate(self, actual: float) -> bool:
        """Return True if the rule passes."""
        return self.operator.evaluate(actual, self.threshold)

    def render_expression(self) -> str:
        """Human-readable rule expression."""
        return f"{self.metric} {self.operator.value} {self.threshold}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "operator": self.operator.value,
            "threshold": self.threshold,
            "severity": self.severity.value,
            "description": self.description,
            "monte_carlo": self.monte_carlo,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PolicyRule":
        return cls(
            metric=d["metric"],
            operator=Operator.from_str(d["operator"]),
            threshold=d["threshold"],
            severity=Severity(d.get("severity", "error")),
            description=d.get("description", ""),
            monte_carlo=d.get("monte_carlo", False),
        )


@dataclass
class RuleResult:
    """Result of evaluating a single rule."""

    rule: PolicyRule
    actual: float
    passed: bool
    error: Optional[str] = None

    @property
    def icon(self) -> str:
        if self.error:
            return "💥"
        if self.passed:
            return "✅"
        if self.rule.severity == Severity.ERROR:
            return "❌"
        if self.rule.severity == Severity.WARNING:
            return "⚠️"
        return "ℹ️"

    @property
    def status(self) -> str:
        if self.error:
            return "ERROR"
        if self.passed:
            return "PASS"
        return self.rule.severity.value.upper()

    def render(self) -> str:
        desc = f"  {self.rule.description}" if self.rule.description else ""
        if self.error:
            return (
                f"  {self.icon}  {self.status:7s}  "
                f"{self.rule.render_expression():40s}  "
                f"error: {self.error}"
            )
        return (
            f"  {self.icon}  {self.status:7s}  "
            f"{self.rule.render_expression():40s}  "
            f"actual={self.actual:<10.3f}{desc}"
        )

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "rule": self.rule.to_dict(),
            "actual": round(self.actual, 6),
            "passed": self.passed,
            "status": self.status,
        }
        if self.error:
            d["error"] = self.error
        return d


@dataclass
class PolicyResult:
    """Aggregate result of evaluating all rules in a policy."""

    policy_name: str
    rule_results: List[RuleResult]
    duration_ms: float
    monte_carlo_used: bool = False
    num_mc_runs: int = 0

    @property
    def passed(self) -> bool:
        """True if no ERROR-severity rules failed and no extraction errors occurred."""
        return not any(
            (not r.passed and r.rule.severity == Severity.ERROR) or r.error
            for r in self.rule_results
        )

    @property
    def has_warnings(self) -> bool:
        return any(
            not r.passed and r.rule.severity == Severity.WARNING
            for r in self.rule_results
        )

    @property
    def has_extraction_errors(self) -> bool:
        """True if any rules failed to extract their metric value."""
        return any(r.error for r in self.rule_results)

    @property
    def errors(self) -> List[RuleResult]:
        return [r for r in self.rule_results if not r.passed and r.rule.severity == Severity.ERROR]

    @property
    def extraction_errors(self) -> List[RuleResult]:
        """Rules where metric extraction failed (misconfigured metric name)."""
        return [r for r in self.rule_results if r.error]

    @property
    def warnings(self) -> List[RuleResult]:
        return [r for r in self.rule_results if not r.passed and r.rule.severity == Severity.WARNING]

    @property
    def failures(self) -> List[RuleResult]:
        return [r for r in self.rule_results if not r.passed or r.error]

    @property
    def passes(self) -> List[RuleResult]:
        return [r for r in self.rule_results if r.passed]

    @property
    def verdict(self) -> str:
        if self.has_extraction_errors:
            return "FAIL"
        if self.passed and not self.has_warnings:
            return "PASS"
        elif self.passed and self.has_warnings:
            return "WARN"
        else:
            return "FAIL"

    @property
    def verdict_icon(self) -> str:
        return {"PASS": "🟢", "WARN": "🟡", "FAIL": "🔴"}[self.verdict]

    @property
    def exit_code(self) -> int:
        """Exit code for CI/CD: 0=pass, 1=fail, 2=warn-only."""
        if not self.passed:
            return 1
        if self.has_warnings:
            return 2
        return 0

    def render(self) -> str:
        lines: List[str] = []
        lines.append("┌───────────────────────────────────────────────────────┐")
        lines.append("│       🛡️  Safety Policy Compliance Report  🛡️         │")
        lines.append("└───────────────────────────────────────────────────────┘")
        lines.append("")
        lines.append(f"  Policy:    {self.policy_name}")
        lines.append(f"  Verdict:   {self.verdict_icon} {self.verdict}")
        lines.append(f"  Rules:     {len(self.rule_results)} total, "
                      f"{len(self.passes)} passed, "
                      f"{len(self.errors)} errors, "
                      f"{len(self.warnings)} warnings")
        if self.monte_carlo_used:
            lines.append(f"  MC Runs:   {self.num_mc_runs}")
        lines.append(f"  Duration:  {self.duration_ms:.0f}ms")
        lines.append("")

        # Group by status
        if self.extraction_errors:
            lines.append("  ── Extraction Errors (misconfigured rules) ──────────")
            for r in self.extraction_errors:
                lines.append(r.render())
            lines.append("")

        if self.errors:
            lines.append("  ── Errors (policy violations) ──────────────────────")
            for r in self.errors:
                lines.append(r.render())
            lines.append("")

        if self.warnings:
            lines.append("  ── Warnings ────────────────────────────────────────")
            for r in self.warnings:
                lines.append(r.render())
            lines.append("")

        info_fails = [r for r in self.rule_results if not r.passed and r.rule.severity == Severity.INFO]
        if info_fails:
            lines.append("  ── Info ────────────────────────────────────────────")
            for r in info_fails:
                lines.append(r.render())
            lines.append("")

        if self.passes:
            lines.append("  ── Passed ──────────────────────────────────────────")
            for r in self.passes:
                lines.append(r.render())
            lines.append("")

        # Recommendations
        if self.errors:
            lines.append("  ── Recommendations ─────────────────────────────────")
            for r in self.errors:
                metric = r.rule.metric
                actual = r.actual
                threshold = r.rule.threshold
                if metric == "total_workers" and actual > threshold:
                    lines.append(f"    • Reduce max_replicas or use a more conservative strategy")
                elif metric == "max_depth_reached" and actual > threshold:
                    lines.append(f"    • Lower max_depth or add cooldown to limit depth penetration")
                elif metric == "denial_rate" and actual < threshold:
                    lines.append(f"    • Tighten contract limits — too many replications are succeeding")
                elif metric == "depth_utilization" and actual > threshold:
                    lines.append(f"    • Depth limit is being fully utilized — consider stricter bounds")
                elif metric in ("prob_max_depth", "risk_score"):
                    lines.append(f"    • High risk detected — lower limits or switch to conservative strategy")
                else:
                    lines.append(f"    • Review {metric}: actual {actual:.3f} vs threshold {threshold}")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "verdict": self.verdict,
            "passed": self.passed,
            "has_warnings": self.has_warnings,
            "has_extraction_errors": self.has_extraction_errors,
            "exit_code": self.exit_code,
            "rules_total": len(self.rule_results),
            "rules_passed": len(self.passes),
            "rules_errors": len(self.errors),
            "rules_warnings": len(self.warnings),
            "rules_extraction_errors": len(self.extraction_errors),
            "monte_carlo_used": self.monte_carlo_used,
            "num_mc_runs": self.num_mc_runs,
            "duration_ms": round(self.duration_ms, 1),
            "results": [r.to_dict() for r in self.rule_results],
        }


# ── Policy presets ──────────────────────────────────────────────────────

def _minimal_rules() -> List[PolicyRule]:
    """Minimal safety — basic sanity checks only."""
    return [
        PolicyRule("total_workers", Operator.LE, 100, Severity.ERROR,
                   "Worker count must not exceed 100"),
        PolicyRule("max_depth_reached", Operator.LE, 10, Severity.ERROR,
                   "Depth must not exceed 10"),
        PolicyRule("depth_utilization", Operator.LE, 1.0, Severity.WARNING,
                   "Depth should stay within configured limits"),
    ]


def _standard_rules() -> List[PolicyRule]:
    """Standard safety — reasonable defaults for most scenarios."""
    return [
        PolicyRule("total_workers", Operator.LE, 50, Severity.ERROR,
                   "Worker count must not exceed 50"),
        PolicyRule("max_depth_reached", Operator.LE, 5, Severity.ERROR,
                   "Depth must not exceed 5"),
        PolicyRule("denial_rate", Operator.GE, 0.3, Severity.WARNING,
                   "At least 30% of replications should be denied"),
        PolicyRule("depth_utilization", Operator.LE, 0.8, Severity.WARNING,
                   "Depth should stay under 80% of configured limit"),
        PolicyRule("worker_utilization", Operator.LE, 2.0, Severity.WARNING,
                   "Workers should not exceed 2x the replica limit"),
        PolicyRule("avg_tasks_per_worker", Operator.GE, 1.0, Severity.INFO,
                   "Workers should complete at least 1 task on average"),
        # Monte Carlo rules
        PolicyRule("prob_max_depth", Operator.LE, 0.5, Severity.ERROR,
                   "Max depth breach probability must stay under 50%",
                   monte_carlo=True),
        PolicyRule("risk_score", Operator.LE, 0.6, Severity.WARNING,
                   "Combined risk score should stay under 0.6",
                   monte_carlo=True),
    ]


def _strict_rules() -> List[PolicyRule]:
    """Strict safety — production-grade containment validation."""
    return [
        PolicyRule("total_workers", Operator.LE, 20, Severity.ERROR,
                   "Worker count must not exceed 20"),
        PolicyRule("max_depth_reached", Operator.LE, 3, Severity.ERROR,
                   "Depth must not exceed 3"),
        PolicyRule("denial_rate", Operator.GE, 0.5, Severity.ERROR,
                   "At least 50% of replications must be denied"),
        PolicyRule("depth_utilization", Operator.LE, 0.6, Severity.ERROR,
                   "Depth must stay under 60% of configured limit"),
        PolicyRule("worker_utilization", Operator.LE, 1.5, Severity.ERROR,
                   "Workers must not exceed 1.5x the replica limit"),
        PolicyRule("success_rate", Operator.LE, 0.7, Severity.WARNING,
                   "Replication success rate should be under 70%"),
        PolicyRule("avg_tasks_per_worker", Operator.GE, 1.5, Severity.WARNING,
                   "Workers should complete at least 1.5 tasks on average"),
        # Monte Carlo rules
        PolicyRule("prob_max_depth", Operator.LE, 0.2, Severity.ERROR,
                   "Max depth breach probability must stay under 20%",
                   monte_carlo=True),
        PolicyRule("prob_quota_saturated", Operator.LE, 0.3, Severity.ERROR,
                   "Quota saturation probability must stay under 30%",
                   monte_carlo=True),
        PolicyRule("risk_score", Operator.LE, 0.3, Severity.ERROR,
                   "Combined risk score must stay under 0.3",
                   monte_carlo=True),
        PolicyRule("peak_workers_p95", Operator.LE, 15, Severity.WARNING,
                   "95th percentile peak workers should be under 15",
                   monte_carlo=True),
    ]


def _ci_rules() -> List[PolicyRule]:
    """CI/CD gate — fail-fast rules for automated pipelines."""
    return [
        PolicyRule("total_workers", Operator.LE, 30, Severity.ERROR,
                   "Worker explosion check"),
        PolicyRule("max_depth_reached", Operator.LE, 4, Severity.ERROR,
                   "Depth containment check"),
        PolicyRule("denial_rate", Operator.GE, 0.4, Severity.ERROR,
                   "Contract enforcement check"),
        PolicyRule("depth_utilization", Operator.LE, 0.75, Severity.ERROR,
                   "Depth headroom check"),
    ]


POLICY_PRESETS: Dict[str, Tuple[str, List[PolicyRule]]] = {
    "minimal": ("Minimal Safety", _minimal_rules()),
    "standard": ("Standard Safety", _standard_rules()),
    "strict": ("Strict Containment", _strict_rules()),
    "ci": ("CI/CD Gate", _ci_rules()),
}


# ── SafetyPolicy class ─────────────────────────────────────────────────

class SafetyPolicy:
    """A named collection of safety rules that can validate simulation results."""

    def __init__(self, name: str, rules: List[PolicyRule]) -> None:
        self.name = name
        self.rules = list(rules)

    @classmethod
    def from_preset(cls, preset: str) -> "SafetyPolicy":
        """Load a built-in policy preset."""
        if preset not in POLICY_PRESETS:
            raise ValueError(
                f"Unknown preset: {preset!r}. Available: {', '.join(POLICY_PRESETS)}"
            )
        name, rules = POLICY_PRESETS[preset]
        return cls(name, rules)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SafetyPolicy":
        """Load a policy from a dictionary (parsed JSON/YAML)."""
        name = data.get("name", "Custom Policy")
        rules = [PolicyRule.from_dict(r) for r in data.get("rules", [])]
        return cls(name, rules)

    @classmethod
    def from_file(cls, path: str) -> "SafetyPolicy":
        """Load a policy from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def add_rule(self, rule: PolicyRule) -> "SafetyPolicy":
        """Add a rule (fluent API)."""
        self.rules.append(rule)
        return self

    def evaluate(
        self,
        report: SimulationReport,
        mc_result: Optional[MonteCarloResult] = None,
    ) -> PolicyResult:
        """Evaluate all rules against a simulation report.

        If mc_result is provided, Monte Carlo rules will be evaluated
        against it. Otherwise, MC rules are skipped.

        Metric extraction errors are recorded on the RuleResult (with
        ``error`` set and ``passed=False``) instead of silently defaulting
        to 0.0, which would mask typos and configuration mistakes in
        safety-critical policy definitions.
        """
        start = time.monotonic()
        results: List[RuleResult] = []

        for rule in self.rules:
            if rule.monte_carlo:
                if mc_result is None:
                    continue  # skip MC rules when no MC data
                try:
                    actual = _extract_mc_metric(mc_result, rule.metric)
                except ValueError as exc:
                    results.append(RuleResult(
                        rule=rule, actual=0.0, passed=False,
                        error=str(exc),
                    ))
                    continue
            else:
                try:
                    actual = _extract_metric(report, rule.metric)
                except ValueError as exc:
                    results.append(RuleResult(
                        rule=rule, actual=0.0, passed=False,
                        error=str(exc),
                    ))
                    continue

            passed = rule.evaluate(actual)
            results.append(RuleResult(rule=rule, actual=actual, passed=passed))

        duration_ms = (time.monotonic() - start) * 1000

        return PolicyResult(
            policy_name=self.name,
            rule_results=results,
            duration_ms=duration_ms,
            monte_carlo_used=mc_result is not None,
            num_mc_runs=mc_result.num_runs if mc_result else 0,
        )

    def evaluate_with_mc(
        self,
        scenario: ScenarioConfig,
        num_runs: int = 100,
        seed: Optional[int] = None,
    ) -> PolicyResult:
        """Run a simulation + Monte Carlo analysis, then evaluate all rules.

        Convenience method that handles both single-run and MC rules.
        """
        start = time.monotonic()

        # Single simulation for single-run metrics
        sim = Simulator(scenario)
        report = sim.run()

        # Monte Carlo for probabilistic rules
        mc_rules = [r for r in self.rules if r.monte_carlo]
        mc_result = None
        if mc_rules:
            mc_config = MonteCarloConfig(
                num_runs=num_runs,
                base_scenario=scenario,
                randomize_seeds=True,
            )
            analyzer = MonteCarloAnalyzer(mc_config)
            mc_result = analyzer.analyze()

        # Evaluate all rules
        results: List[RuleResult] = []
        for rule in self.rules:
            if rule.monte_carlo:
                if mc_result is None:
                    continue
                try:
                    actual = _extract_mc_metric(mc_result, rule.metric)
                except ValueError as exc:
                    results.append(RuleResult(
                        rule=rule, actual=0.0, passed=False,
                        error=str(exc),
                    ))
                    continue
            else:
                try:
                    actual = _extract_metric(report, rule.metric)
                except ValueError as exc:
                    results.append(RuleResult(
                        rule=rule, actual=0.0, passed=False,
                        error=str(exc),
                    ))
                    continue

            passed = rule.evaluate(actual)
            results.append(RuleResult(rule=rule, actual=actual, passed=passed))

        duration_ms = (time.monotonic() - start) * 1000

        return PolicyResult(
            policy_name=self.name,
            rule_results=results,
            duration_ms=duration_ms,
            monte_carlo_used=mc_result is not None,
            num_mc_runs=mc_result.num_runs if mc_result else 0,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "rules": [r.to_dict() for r in self.rules],
        }

    def save(self, path: str) -> None:
        """Save policy to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ── CLI ─────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point for the safety policy engine."""
    import argparse
    import sys
    import io

    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="AI Replication Sandbox — Safety Policy Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Built-in policy presets: minimal, standard, strict, ci

Examples:
  python -m replication.policy --preset strict                       # strict preset, default scenario
  python -m replication.policy --preset standard --strategy greedy   # test greedy strategy
  python -m replication.policy --preset ci --max-depth 4             # CI gate with custom depth
  python -m replication.policy --file my_policy.json                 # custom policy file
  python -m replication.policy --preset standard --runs 200          # Monte Carlo validation
  python -m replication.policy --list-presets                        # show presets
  python -m replication.policy --list-metrics                        # show metrics

Exit codes: 0 = PASS, 1 = FAIL (errors), 2 = WARN (warnings only)
        """,
    )

    # Policy source
    policy_group = parser.add_mutually_exclusive_group()
    policy_group.add_argument("--preset", choices=list(POLICY_PRESETS.keys()),
                              help="Use a built-in policy preset")
    policy_group.add_argument("--file", help="Load policy from JSON file")
    policy_group.add_argument("--list-presets", action="store_true",
                              help="List available policy presets")
    policy_group.add_argument("--list-metrics", action="store_true",
                              help="List available metrics for rules")

    # Scenario config
    parser.add_argument("--scenario", choices=list(SIM_PRESETS.keys()),
                        help="Use a built-in simulation preset")
    parser.add_argument("--strategy", choices=[s.value for s in Strategy],
                        help="Replication strategy")
    parser.add_argument("--max-depth", type=int, help="Maximum replication depth")
    parser.add_argument("--max-replicas", type=int, help="Maximum concurrent replicas")
    parser.add_argument("--cooldown", type=float, help="Cooldown seconds")
    parser.add_argument("--tasks", type=int, help="Tasks per worker")
    parser.add_argument("--probability", type=float, help="Replication probability")
    parser.add_argument("--seed", type=int, help="Random seed")

    # Monte Carlo
    parser.add_argument("--runs", type=int, default=50,
                        help="Monte Carlo simulation runs (default: 50)")
    parser.add_argument("--no-mc", action="store_true",
                        help="Skip Monte Carlo rules (single-run only)")

    # Output
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--save-policy", help="Save the policy to a JSON file")

    args = parser.parse_args()

    # Handle list commands
    if args.list_presets:
        print("┌───────────────────────────────────────────────────────┐")
        print("│       Available Policy Presets                        │")
        print("└───────────────────────────────────────────────────────┘")
        print()
        for key, (name, rules) in POLICY_PRESETS.items():
            errors = sum(1 for r in rules if r.severity == Severity.ERROR)
            warnings = sum(1 for r in rules if r.severity == Severity.WARNING)
            mc = sum(1 for r in rules if r.monte_carlo)
            print(f"  {key:12s}  {name}")
            print(f"               {len(rules)} rules ({errors} error, {warnings} warning, {mc} MC)")
            for r in rules:
                sev = r.severity.value[0].upper()
                mc_tag = " [MC]" if r.monte_carlo else ""
                print(f"                 [{sev}] {r.render_expression()}{mc_tag}")
                if r.description:
                    print(f"                     {r.description}")
            print()
        sys.exit(0)

    if args.list_metrics:
        print("┌───────────────────────────────────────────────────────┐")
        print("│       Available Metrics                               │")
        print("└───────────────────────────────────────────────────────┘")
        print()
        print("  ── Single-Run Metrics ──────────────────────────────")
        for name, desc in sorted(SINGLE_RUN_METRICS.items()):
            print(f"    {name:25s}  {desc}")
        print()
        print("  ── Monte Carlo Metrics (require --runs) ────────────")
        for name, desc in sorted(MONTE_CARLO_METRICS.items()):
            print(f"    {name:25s}  {desc}")
        print()
        sys.exit(0)

    # Load policy
    if args.file:
        policy = SafetyPolicy.from_file(args.file)
    elif args.preset:
        policy = SafetyPolicy.from_preset(args.preset)
    else:
        policy = SafetyPolicy.from_preset("standard")

    # Save policy if requested
    if args.save_policy:
        policy.save(args.save_policy)
        print(f"Policy saved to {args.save_policy}")
        if not args.preset and not args.file:
            sys.exit(0)

    # Build scenario config
    if args.scenario:
        config = SIM_PRESETS[args.scenario]
    else:
        config = ScenarioConfig()

    if args.strategy:
        config.strategy = args.strategy
    if args.max_depth is not None:
        config.max_depth = args.max_depth
    if args.max_replicas is not None:
        config.max_replicas = args.max_replicas
    if args.cooldown is not None:
        config.cooldown_seconds = args.cooldown
    if args.tasks is not None:
        config.tasks_per_worker = args.tasks
    if args.probability is not None:
        config.replication_probability = args.probability
    if args.seed is not None:
        config.seed = args.seed

    # Evaluate
    has_mc_rules = any(r.monte_carlo for r in policy.rules)

    if has_mc_rules and not args.no_mc:
        result = policy.evaluate_with_mc(config, num_runs=args.runs, seed=args.seed)
    else:
        sim = Simulator(config)
        report = sim.run()
        result = policy.evaluate(report)

    # Output
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        print(result.render())

    sys.exit(result.exit_code)


if __name__ == "__main__":
    main()
