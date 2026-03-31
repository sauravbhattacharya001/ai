"""Quick Scan — consolidated safety assessment in one command.

Runs multiple safety checks (scorecard, compliance, policy lint, preflight)
and presents a single pass/fail summary with color-coded results.

Usage (CLI)::

    python -m replication quick-scan                  # run all checks
    python -m replication quick-scan --json           # JSON output
    python -m replication quick-scan --checks scorecard,compliance
    python -m replication quick-scan --strict         # fail on any warning

Programmatic::

    from replication.quick_scan import QuickScanner
    scanner = QuickScanner()
    result = scanner.run()
    print(result.render())
    print(f"Overall: {'PASS' if result.passed else 'FAIL'}")
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ._helpers import box_header


# ── Data classes ─────────────────────────────────────────────────────


@dataclass
class CheckResult:
    """Result of a single safety check."""
    name: str
    status: str  # "pass", "warn", "fail", "error", "skip"
    score: Optional[float] = None  # 0-100 if applicable
    summary: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0


@dataclass
class ScanResult:
    """Aggregated result of all checks."""
    checks: List[CheckResult] = field(default_factory=list)
    total_duration_s: float = 0.0
    strict: bool = False
    timestamp: str = ""

    @property
    def passed(self) -> bool:
        """Overall pass: no failures (and no warnings if strict)."""
        for c in self.checks:
            if c.status == "fail" or c.status == "error":
                return False
            if self.strict and c.status == "warn":
                return False
        return True

    @property
    def pass_count(self) -> int:
        return sum(1 for c in self.checks if c.status == "pass")

    @property
    def warn_count(self) -> int:
        return sum(1 for c in self.checks if c.status == "warn")

    @property
    def fail_count(self) -> int:
        return sum(1 for c in self.checks if c.status in ("fail", "error"))

    def render(self) -> str:
        """Render a human-readable summary."""
        lines = box_header("Quick Scan Results")
        lines.append("")
        lines.append(f"  Timestamp: {self.timestamp}")
        lines.append(f"  Duration:  {self.total_duration_s:.1f}s")
        lines.append(f"  Mode:      {'strict' if self.strict else 'standard'}")
        lines.append("")

        _icons = {"pass": "\u2705", "warn": "\u26a0\ufe0f", "fail": "\u274c", "error": "\u274c", "skip": "\u23ed\ufe0f"}
        max_name = max((len(c.name) for c in self.checks), default=10)

        for c in self.checks:
            icon = _icons.get(c.status, "?")
            score_str = f" ({c.score:.0f}/100)" if c.score is not None else ""
            dur = f" [{c.duration_s:.1f}s]"
            lines.append(f"  {icon} {c.name:<{max_name}}  {c.status.upper()}{score_str}{dur}")
            if c.summary:
                lines.append(f"     {c.summary}")

        lines.append("")
        overall = "PASS \u2705" if self.passed else "FAIL \u274c"
        lines.append(f"  Overall: {overall}  ({self.pass_count} passed, {self.warn_count} warnings, {self.fail_count} failed)")
        lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "strict": self.strict,
            "timestamp": self.timestamp,
            "total_duration_s": round(self.total_duration_s, 2),
            "summary": {
                "pass": self.pass_count,
                "warn": self.warn_count,
                "fail": self.fail_count,
            },
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    "score": c.score,
                    "summary": c.summary,
                    "duration_s": round(c.duration_s, 2),
                    "details": c.details,
                }
                for c in self.checks
            ],
        }


# ── Available checks ─────────────────────────────────────────────────

ALL_CHECKS = ["preflight", "scorecard", "compliance", "policy-lint"]


class QuickScanner:
    """Run multiple safety checks and aggregate results."""

    def __init__(self, checks: Optional[List[str]] = None, strict: bool = False):
        self.checks = checks or list(ALL_CHECKS)
        self.strict = strict

    def run(self) -> ScanResult:
        result = ScanResult(strict=self.strict)
        result.timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        t0 = time.time()

        for check_name in self.checks:
            runner = _RUNNERS.get(check_name)
            if runner is None:
                result.checks.append(CheckResult(
                    name=check_name, status="skip", summary=f"Unknown check: {check_name}"
                ))
                continue
            try:
                cr = runner()
            except Exception as exc:
                cr = CheckResult(name=check_name, status="error", summary=str(exc)[:200])
            result.checks.append(cr)

        result.total_duration_s = time.time() - t0
        return result


# ── Individual check runners ─────────────────────────────────────────


def _run_preflight() -> CheckResult:
    t0 = time.time()
    try:
        from .preflight import PreflightChecker, PreflightConfig
        checker = PreflightChecker(PreflightConfig())
        report = checker.run()
        n_errors = len(report.errors) if hasattr(report, "errors") else 0
        n_warnings = len(report.warnings) if hasattr(report, "warnings") else 0
        n_findings = len(report.findings) if hasattr(report, "findings") else 0
        return CheckResult(
            name="preflight",
            status="pass" if report.passed else ("warn" if n_errors == 0 else "fail"),
            summary=f"{n_findings} finding(s) ({n_errors} errors, {n_warnings} warnings)" if not report.passed else "All preflight checks passed",
            details={"errors": n_errors, "warnings": n_warnings, "findings": n_findings},
            duration_s=time.time() - t0,
        )
    except Exception as exc:
        return CheckResult(name="preflight", status="error", summary=str(exc)[:200], duration_s=time.time() - t0)


def _run_scorecard() -> CheckResult:
    t0 = time.time()
    try:
        from .scorecard import SafetyScorecard, ScorecardConfig
        from .simulator import ScenarioConfig
        sc = SafetyScorecard(ScorecardConfig(quick=True))
        scenario = ScenarioConfig()
        result = sc.evaluate(scenario)
        score = result.overall_score
        grade = result.overall_grade
        status = "pass" if score >= 70 else ("warn" if score >= 50 else "fail")
        return CheckResult(
            name="scorecard",
            status=status,
            score=score,
            summary=f"Grade: {grade}",
            details={"grade": grade, "dimensions": {d.name: d.score for d in result.dimensions} if hasattr(result, "dimensions") else {}},
            duration_s=time.time() - t0,
        )
    except Exception as exc:
        return CheckResult(name="scorecard", status="error", summary=str(exc)[:200], duration_s=time.time() - t0)


def _run_compliance() -> CheckResult:
    t0 = time.time()
    try:
        from .compliance import ComplianceAuditor
        from .contract import ReplicationContract
        auditor = ComplianceAuditor()
        contract = ReplicationContract(max_depth=3, max_replicas=10, cooldown_seconds=5.0)
        report = auditor.audit(contract)
        score = report.score if hasattr(report, "score") else None
        n_findings = report.total_findings if hasattr(report, "total_findings") else 0
        framework = "multi" if hasattr(report, "framework_results") else "default"
        verdict = report.overall_verdict if hasattr(report, "overall_verdict") else "unknown"
        status = "pass" if verdict == "PASS" else ("warn" if verdict == "PARTIAL" else "fail")
        return CheckResult(
            name="compliance",
            status=status,
            score=score,
            summary=f"{n_findings} finding(s), framework: {framework}",
            details={"findings_count": n_findings, "framework": framework},
            duration_s=time.time() - t0,
        )
    except Exception as exc:
        return CheckResult(name="compliance", status="error", summary=str(exc)[:200], duration_s=time.time() - t0)


def _run_policy_lint() -> CheckResult:
    t0 = time.time()
    try:
        from .policy_linter import PolicyLinter
        from .policy import SafetyPolicy
        linter = PolicyLinter()
        policy = SafetyPolicy(name="standard", rules=[])
        report = linter.lint(policy)
        n_errors = report.error_count if hasattr(report, "error_count") else 0
        n_warnings = report.warning_count if hasattr(report, "warning_count") else 0
        status = "pass" if n_errors == 0 and n_warnings == 0 else ("warn" if n_errors == 0 else "fail")
        return CheckResult(
            name="policy-lint",
            status=status,
            summary=f"{n_errors} error(s), {n_warnings} warning(s)",
            details={"errors": n_errors, "warnings": n_warnings},
            duration_s=time.time() - t0,
        )
    except Exception as exc:
        return CheckResult(name="policy-lint", status="error", summary=str(exc)[:200], duration_s=time.time() - t0)


_RUNNERS = {
    "preflight": _run_preflight,
    "scorecard": _run_scorecard,
    "compliance": _run_compliance,
    "policy-lint": _run_policy_lint,
}


# ── CLI ──────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m replication quick-scan",
        description="Run consolidated safety checks and get a single pass/fail verdict",
    )
    parser.add_argument(
        "--checks",
        type=lambda s: s.split(","),
        default=None,
        help=f"Comma-separated checks to run (default: all). Available: {', '.join(ALL_CHECKS)}",
    )
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Output JSON instead of table")

    args = parser.parse_args(argv)
    scanner = QuickScanner(checks=args.checks, strict=args.strict)
    result = scanner.run()

    if args.json_output:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.render())


if __name__ == "__main__":
    main()
