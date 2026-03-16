"""Safety Policy Linter — static analysis of policy definitions.

Analyzes safety policies for misconfigurations, contradictions, coverage
gaps, and anti-patterns without running any simulations.  Catches issues
early — before they silently pass in CI/CD or produce misleading results.

Usage (CLI)::

    python -m replication lint --preset strict              # lint a built-in preset
    python -m replication lint --file my_policy.json        # lint a custom policy
    python -m replication lint --all-presets                 # lint every built-in preset
    python -m replication lint --preset standard --fix      # show auto-fix suggestions
    python -m replication lint --json                       # JSON output

Programmatic::

    from replication.policy_linter import PolicyLinter, LintReport
    from replication.policy import SafetyPolicy

    policy = SafetyPolicy.from_preset("strict")
    linter = PolicyLinter()
    report = linter.lint(policy)
    print(report.render())
    assert report.passed, f"Policy has issues: {report.summary()}"
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from .policy import (
    MONTE_CARLO_METRICS,
    POLICY_PRESETS,
    SINGLE_RUN_METRICS,
    Operator,
    PolicyRule,
    SafetyPolicy,
    Severity,
)


# ── Finding severity ────────────────────────────────────────────────

class FindingSeverity(Enum):
    """Severity of a lint finding."""
    ERROR = "error"        # Will cause runtime failures or incorrect results
    WARNING = "warning"    # Likely misconfiguration or gap
    INFO = "info"          # Suggestion or best-practice hint
    STYLE = "style"        # Cosmetic / naming convention

    @property
    def icon(self) -> str:
        return {
            "error": "🔴",
            "warning": "🟡",
            "info": "💡",
            "style": "🔵",
        }[self.value]


# ── Finding categories ──────────────────────────────────────────────

class FindingCategory(Enum):
    """Categories of lint findings."""
    INVALID_METRIC = "invalid_metric"
    CONTRADICTION = "contradiction"
    COVERAGE_GAP = "coverage_gap"
    THRESHOLD = "threshold"
    REDUNDANCY = "redundancy"
    BEST_PRACTICE = "best_practice"
    SEVERITY_ISSUE = "severity_issue"


# ── Data models ─────────────────────────────────────────────────────

@dataclass
class LintFinding:
    """A single lint finding."""
    severity: FindingSeverity
    category: FindingCategory
    message: str
    rule_index: Optional[int] = None  # Index of the rule in the policy
    rule_expr: str = ""               # Human-readable rule expression
    suggestion: str = ""              # Fix suggestion

    def render(self) -> str:
        parts = [f"  {self.severity.icon}  [{self.category.value}]  {self.message}"]
        if self.rule_expr:
            parts[0] += f"  ({self.rule_expr})"
        if self.suggestion:
            parts.append(f"       ↳ {self.suggestion}")
        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "severity": self.severity.value,
            "category": self.category.value,
            "message": self.message,
        }
        if self.rule_index is not None:
            d["rule_index"] = self.rule_index
        if self.rule_expr:
            d["rule_expression"] = self.rule_expr
        if self.suggestion:
            d["suggestion"] = self.suggestion
        return d


@dataclass
class LintReport:
    """Aggregate result of linting a policy."""
    policy_name: str
    findings: List[LintFinding] = field(default_factory=list)
    rules_checked: int = 0

    @property
    def passed(self) -> bool:
        """True if no ERROR-severity findings."""
        return not any(f.severity == FindingSeverity.ERROR for f in self.findings)

    @property
    def errors(self) -> List[LintFinding]:
        return [f for f in self.findings if f.severity == FindingSeverity.ERROR]

    @property
    def warnings(self) -> List[LintFinding]:
        return [f for f in self.findings if f.severity == FindingSeverity.WARNING]

    @property
    def infos(self) -> List[LintFinding]:
        return [f for f in self.findings if f.severity == FindingSeverity.INFO]

    @property
    def styles(self) -> List[LintFinding]:
        return [f for f in self.findings if f.severity == FindingSeverity.STYLE]

    @property
    def verdict(self) -> str:
        if self.errors:
            return "FAIL"
        if self.warnings:
            return "WARN"
        if not self.findings:
            return "CLEAN"
        return "OK"

    @property
    def verdict_icon(self) -> str:
        return {"FAIL": "🔴", "WARN": "🟡", "CLEAN": "✅", "OK": "🟢"}[self.verdict]

    def summary(self) -> str:
        return (f"{len(self.errors)} errors, {len(self.warnings)} warnings, "
                f"{len(self.infos)} info, {len(self.styles)} style")

    def render(self) -> str:
        lines: List[str] = []
        lines.append("┌───────────────────────────────────────────────────────┐")
        lines.append("│       🔍  Safety Policy Lint Report  🔍              │")
        lines.append("└───────────────────────────────────────────────────────┘")
        lines.append("")
        lines.append(f"  Policy:    {self.policy_name}")
        lines.append(f"  Verdict:   {self.verdict_icon} {self.verdict}")
        lines.append(f"  Rules:     {self.rules_checked} checked")
        lines.append(f"  Findings:  {self.summary()}")
        lines.append("")

        if not self.findings:
            lines.append("  No issues found — policy looks good! ✨")
            lines.append("")
            return "\n".join(lines)

        # Group by severity
        for sev, label in [
            (FindingSeverity.ERROR, "Errors"),
            (FindingSeverity.WARNING, "Warnings"),
            (FindingSeverity.INFO, "Suggestions"),
            (FindingSeverity.STYLE, "Style"),
        ]:
            group = [f for f in self.findings if f.severity == sev]
            if group:
                lines.append(f"  ── {label} {'─' * (48 - len(label))}")
                for finding in group:
                    lines.append(finding.render())
                lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "verdict": self.verdict,
            "passed": self.passed,
            "rules_checked": self.rules_checked,
            "counts": {
                "errors": len(self.errors),
                "warnings": len(self.warnings),
                "info": len(self.infos),
                "style": len(self.styles),
                "total": len(self.findings),
            },
            "findings": [f.to_dict() for f in self.findings],
        }


# ── Linter ──────────────────────────────────────────────────────────

# Metrics that represent ratios (0–1 range)
_RATIO_METRICS: Set[str] = {
    "denial_rate", "success_rate", "depth_utilization",
    "prob_max_depth", "prob_quota_saturated", "risk_score",
}

# Metrics that should be bounded above (upper-limit rules expected)
_UPPER_BOUND_METRICS: Set[str] = {
    "total_workers", "max_depth_reached", "repl_succeeded",
    "depth_utilization", "worker_utilization",
    "prob_max_depth", "prob_quota_saturated", "risk_score",
    "peak_workers_p95", "peak_depth_p95",
}

# Metrics that should be bounded below (lower-limit rules expected)
_LOWER_BOUND_METRICS: Set[str] = {
    "denial_rate", "avg_tasks_per_worker",
}

# Critical safety metrics that every non-minimal policy should cover
_CRITICAL_METRICS: Set[str] = {
    "total_workers", "max_depth_reached",
}

ALL_METRICS = set(SINGLE_RUN_METRICS) | set(MONTE_CARLO_METRICS)


class PolicyLinter:
    """Static analyzer for SafetyPolicy definitions."""

    def lint(self, policy: SafetyPolicy) -> LintReport:
        """Run all lint checks on a policy."""
        report = LintReport(
            policy_name=policy.name,
            rules_checked=len(policy.rules),
        )

        if not policy.rules:
            report.findings.append(LintFinding(
                severity=FindingSeverity.WARNING,
                category=FindingCategory.COVERAGE_GAP,
                message="Policy has no rules — it will always pass",
                suggestion="Add at least total_workers and max_depth_reached rules",
            ))
            return report

        self._check_invalid_metrics(policy, report)
        self._check_contradictions(policy, report)
        self._check_threshold_ranges(policy, report)
        self._check_coverage_gaps(policy, report)
        self._check_redundancies(policy, report)
        self._check_severity_issues(policy, report)
        self._check_best_practices(policy, report)

        return report

    def _check_invalid_metrics(self, policy: SafetyPolicy, report: LintReport) -> None:
        """Flag rules referencing unknown metrics."""
        for i, rule in enumerate(policy.rules):
            if rule.monte_carlo:
                valid = set(MONTE_CARLO_METRICS) | set(SINGLE_RUN_METRICS)
            else:
                valid = set(SINGLE_RUN_METRICS)

            if rule.metric not in valid:
                # Find closest match
                from difflib import get_close_matches
                matches = get_close_matches(rule.metric, sorted(valid), n=1, cutoff=0.5)
                suggestion = f"Did you mean '{matches[0]}'?" if matches else "Check available metrics with --list-metrics"

                report.findings.append(LintFinding(
                    severity=FindingSeverity.ERROR,
                    category=FindingCategory.INVALID_METRIC,
                    message=f"Unknown metric '{rule.metric}'",
                    rule_index=i,
                    rule_expr=rule.render_expression(),
                    suggestion=suggestion,
                ))

    def _check_contradictions(self, policy: SafetyPolicy, report: LintReport) -> None:
        """Find rules that contradict each other."""
        # Group rules by metric
        by_metric: Dict[str, List[Tuple[int, PolicyRule]]] = {}
        for i, rule in enumerate(policy.rules):
            by_metric.setdefault(rule.metric, []).append((i, rule))

        for metric, rules in by_metric.items():
            if len(rules) < 2:
                continue

            # Check for impossible combinations like (x <= 5) AND (x >= 10)
            upper_bounds: List[Tuple[int, float]] = []
            lower_bounds: List[Tuple[int, float]] = []

            for idx, rule in rules:
                if rule.operator in (Operator.LT, Operator.LE):
                    upper_bounds.append((idx, rule.threshold))
                elif rule.operator in (Operator.GT, Operator.GE):
                    lower_bounds.append((idx, rule.threshold))

            for ui, upper in upper_bounds:
                for li, lower in lower_bounds:
                    if lower >= upper:
                        report.findings.append(LintFinding(
                            severity=FindingSeverity.ERROR,
                            category=FindingCategory.CONTRADICTION,
                            message=(f"Contradictory rules for '{metric}': "
                                     f"must be ≤{upper} AND ≥{lower}"),
                            rule_index=ui,
                            rule_expr=f"rules [{ui}] and [{li}]",
                            suggestion=f"Adjust thresholds so lower < upper",
                        ))

    def _check_threshold_ranges(self, policy: SafetyPolicy, report: LintReport) -> None:
        """Flag thresholds that are out of expected range."""
        for i, rule in enumerate(policy.rules):
            # Ratio metrics should have thresholds in [0, 1]
            if rule.metric in _RATIO_METRICS:
                if rule.threshold < 0 or rule.threshold > 1:
                    report.findings.append(LintFinding(
                        severity=FindingSeverity.WARNING,
                        category=FindingCategory.THRESHOLD,
                        message=f"Threshold {rule.threshold} is outside [0, 1] for ratio metric '{rule.metric}'",
                        rule_index=i,
                        rule_expr=rule.render_expression(),
                        suggestion="Ratio metrics range from 0.0 to 1.0",
                    ))

            # Count metrics shouldn't be negative
            if rule.metric in ("total_workers", "max_depth_reached", "repl_succeeded",
                               "repl_denied", "repl_attempted", "total_tasks"):
                if rule.threshold < 0:
                    report.findings.append(LintFinding(
                        severity=FindingSeverity.WARNING,
                        category=FindingCategory.THRESHOLD,
                        message=f"Negative threshold for count metric '{rule.metric}'",
                        rule_index=i,
                        rule_expr=rule.render_expression(),
                        suggestion="Count metrics are always ≥ 0",
                    ))

            # Very permissive thresholds
            if rule.metric == "total_workers" and rule.operator in (Operator.LE, Operator.LT):
                if rule.threshold > 500:
                    report.findings.append(LintFinding(
                        severity=FindingSeverity.INFO,
                        category=FindingCategory.THRESHOLD,
                        message=f"Very permissive worker limit ({rule.threshold})",
                        rule_index=i,
                        rule_expr=rule.render_expression(),
                        suggestion="Most scenarios need ≤100 workers; consider tightening",
                    ))

            if rule.metric == "max_depth_reached" and rule.operator in (Operator.LE, Operator.LT):
                if rule.threshold > 20:
                    report.findings.append(LintFinding(
                        severity=FindingSeverity.INFO,
                        category=FindingCategory.THRESHOLD,
                        message=f"Very permissive depth limit ({rule.threshold})",
                        rule_index=i,
                        rule_expr=rule.render_expression(),
                        suggestion="Deep replication chains are hard to control; consider ≤10",
                    ))

            # Operator direction checks
            if rule.metric in _UPPER_BOUND_METRICS and rule.operator in (Operator.GE, Operator.GT):
                if rule.metric not in _LOWER_BOUND_METRICS:
                    report.findings.append(LintFinding(
                        severity=FindingSeverity.WARNING,
                        category=FindingCategory.THRESHOLD,
                        message=f"'{rule.metric}' usually needs an upper bound (≤), not a lower bound ({rule.operator.value})",
                        rule_index=i,
                        rule_expr=rule.render_expression(),
                        suggestion=f"Consider using ≤ or < for '{rule.metric}'",
                    ))

    def _check_coverage_gaps(self, policy: SafetyPolicy, report: LintReport) -> None:
        """Flag critical metrics that are not covered by any rule."""
        covered = {rule.metric for rule in policy.rules}

        for metric in _CRITICAL_METRICS:
            if metric not in covered:
                report.findings.append(LintFinding(
                    severity=FindingSeverity.WARNING,
                    category=FindingCategory.COVERAGE_GAP,
                    message=f"Critical metric '{metric}' has no rule",
                    suggestion=f"Add a rule like: {metric} <= <threshold>",
                ))

        # Check if MC-capable policy has no MC rules
        has_mc = any(r.monte_carlo for r in policy.rules)
        non_mc_count = sum(1 for r in policy.rules if not r.monte_carlo)
        if non_mc_count > 3 and not has_mc:
            report.findings.append(LintFinding(
                severity=FindingSeverity.INFO,
                category=FindingCategory.COVERAGE_GAP,
                message="Policy has no Monte Carlo rules",
                suggestion="MC rules (prob_max_depth, risk_score) catch probabilistic risks",
            ))

    def _check_redundancies(self, policy: SafetyPolicy, report: LintReport) -> None:
        """Flag duplicate or subsumed rules."""
        by_metric: Dict[str, List[Tuple[int, PolicyRule]]] = {}
        for i, rule in enumerate(policy.rules):
            by_metric.setdefault(rule.metric, []).append((i, rule))

        for metric, rules in by_metric.items():
            if len(rules) < 2:
                continue

            # Check for exact duplicates
            seen: List[Tuple[int, PolicyRule]] = []
            for idx, rule in rules:
                for prev_idx, prev in seen:
                    if (rule.operator == prev.operator and
                        abs(rule.threshold - prev.threshold) < 1e-9 and
                        rule.monte_carlo == prev.monte_carlo):
                        report.findings.append(LintFinding(
                            severity=FindingSeverity.STYLE,
                            category=FindingCategory.REDUNDANCY,
                            message=f"Duplicate rule for '{metric}'",
                            rule_index=idx,
                            rule_expr=rule.render_expression(),
                            suggestion=f"Remove rule [{idx}] (same as [{prev_idx}])",
                        ))
                seen.append((idx, rule))

            # Check for subsumed upper bounds: x<=5 makes x<=10 redundant
            upper = [(i, r) for i, r in rules if r.operator in (Operator.LE, Operator.LT)]
            if len(upper) >= 2:
                upper.sort(key=lambda x: x[1].threshold)
                tightest_idx, tightest = upper[0]
                for other_idx, other in upper[1:]:
                    if other.severity == tightest.severity:
                        report.findings.append(LintFinding(
                            severity=FindingSeverity.STYLE,
                            category=FindingCategory.REDUNDANCY,
                            message=(f"Rule [{other_idx}] ({other.render_expression()}) "
                                     f"is subsumed by [{tightest_idx}] ({tightest.render_expression()})"),
                            rule_index=other_idx,
                            rule_expr=other.render_expression(),
                            suggestion="Remove the looser rule or change its severity",
                        ))

    def _check_severity_issues(self, policy: SafetyPolicy, report: LintReport) -> None:
        """Flag questionable severity assignments."""
        for i, rule in enumerate(policy.rules):
            # Critical safety metrics should not be INFO
            if rule.metric in _CRITICAL_METRICS and rule.severity == Severity.INFO:
                report.findings.append(LintFinding(
                    severity=FindingSeverity.WARNING,
                    category=FindingCategory.SEVERITY_ISSUE,
                    message=f"Critical metric '{rule.metric}' has INFO severity",
                    rule_index=i,
                    rule_expr=rule.render_expression(),
                    suggestion="Use ERROR or WARNING for safety-critical metrics",
                ))

    def _check_best_practices(self, policy: SafetyPolicy, report: LintReport) -> None:
        """Suggest best-practice improvements."""
        covered = {rule.metric for rule in policy.rules}

        # Suggest denial_rate if they have worker/depth limits but no denial check
        if "total_workers" in covered and "denial_rate" not in covered:
            report.findings.append(LintFinding(
                severity=FindingSeverity.INFO,
                category=FindingCategory.BEST_PRACTICE,
                message="No denial_rate rule — can't verify contract enforcement",
                suggestion="Add: denial_rate >= 0.3 (ensures contracts are actually blocking)",
            ))

        # Policy with no descriptions
        undescribed = sum(1 for r in policy.rules if not r.description)
        if undescribed == len(policy.rules) and len(policy.rules) > 2:
            report.findings.append(LintFinding(
                severity=FindingSeverity.STYLE,
                category=FindingCategory.BEST_PRACTICE,
                message="No rules have descriptions",
                suggestion="Add descriptions to help readers understand rule intent",
            ))


# ── CLI ─────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point for the policy linter."""
    import argparse
    import io

    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="AI Replication Sandbox — Safety Policy Linter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Lint safety policies for misconfigurations, contradictions, and gaps.

Examples:
  python -m replication lint --preset strict           # lint strict preset
  python -m replication lint --file my_policy.json     # lint custom policy
  python -m replication lint --all-presets              # lint all built-in presets
  python -m replication lint --json                    # JSON output

Finding severities:
  🔴 error    — will cause runtime failures or incorrect results
  🟡 warning  — likely misconfiguration or gap
  💡 info     — suggestion or best-practice hint
  🔵 style    — cosmetic / naming convention
        """,
    )

    source = parser.add_mutually_exclusive_group()
    source.add_argument("--preset", choices=list(POLICY_PRESETS.keys()),
                        help="Lint a built-in preset")
    source.add_argument("--file", help="Lint a policy from a JSON file")
    source.add_argument("--all-presets", action="store_true",
                        help="Lint all built-in presets")

    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--strict", action="store_true",
                        help="Treat warnings as errors (exit 1 on warnings)")

    args = parser.parse_args()

    linter = PolicyLinter()

    if args.all_presets:
        reports: List[LintReport] = []
        for name in POLICY_PRESETS:
            policy = SafetyPolicy.from_preset(name)
            reports.append(linter.lint(policy))

        if args.json:
            print(json.dumps([r.to_dict() for r in reports], indent=2))
        else:
            for report in reports:
                print(report.render())

        # Exit with worst status
        has_errors = any(not r.passed for r in reports)
        has_warnings = any(r.warnings for r in reports)
        if has_errors:
            sys.exit(1)
        elif has_warnings and args.strict:
            sys.exit(1)
        elif has_warnings:
            sys.exit(2)
        sys.exit(0)

    # Single policy
    if args.file:
        policy = SafetyPolicy.from_file(args.file)
    elif args.preset:
        policy = SafetyPolicy.from_preset(args.preset)
    else:
        policy = SafetyPolicy.from_preset("standard")

    report = linter.lint(policy)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.render())

    if not report.passed:
        sys.exit(1)
    elif report.warnings and args.strict:
        sys.exit(1)
    elif report.warnings:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
