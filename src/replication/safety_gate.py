"""Safety Gate — pre-deployment readiness checker.

Evaluates whether an agent configuration meets minimum safety criteria
before being allowed to deploy.  Think of it as a "go / no-go" gate
that aggregates checks from multiple safety subsystems into a single
pass/fail verdict with detailed justification.

CLI usage::

    python -m replication gate
    python -m replication gate --config agent.json
    python -m replication gate --strict
    python -m replication gate --format json
    python -m replication gate --custom-checks checks.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


# ── models ───────────────────────────────────────────────────────────

class Verdict(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class CheckResult:
    name: str
    verdict: Verdict
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class GateResult:
    overall: Verdict = Verdict.PASS
    checks: List[CheckResult] = field(default_factory=list)
    passed: int = 0
    warned: int = 0
    failed: int = 0

    def add(self, result: CheckResult) -> None:
        self.checks.append(result)
        if result.verdict == Verdict.PASS:
            self.passed += 1
        elif result.verdict == Verdict.WARN:
            self.warned += 1
            if self.overall == Verdict.PASS:
                self.overall = Verdict.WARN
        elif result.verdict == Verdict.FAIL:
            self.failed += 1
            self.overall = Verdict.FAIL


# ── default agent config (demo) ─────────────────────────────────────

DEFAULT_AGENT: Dict[str, Any] = {
    "name": "demo-agent",
    "version": "1.0.0",
    "kill_switch": True,
    "max_replication_depth": 3,
    "allowed_actions": ["read", "write", "execute"],
    "restricted_actions": ["network_exfiltrate", "self_modify"],
    "audit_logging": True,
    "alignment_score": 0.82,
    "resource_limits": {
        "max_cpu_percent": 80,
        "max_memory_mb": 2048,
        "max_network_mbps": 100,
    },
    "compliance_frameworks": ["nist_ai_rmf"],
    "safety_contract": True,
    "watermark_enabled": True,
    "sandbox_mode": True,
}


# ── built-in checks ─────────────────────────────────────────────────

def check_kill_switch(agent: Dict[str, Any], strict: bool = False) -> CheckResult:
    """Kill switch must be enabled."""
    enabled = agent.get("kill_switch", False)
    if enabled:
        return CheckResult("kill_switch", Verdict.PASS, "Kill switch is enabled")
    return CheckResult("kill_switch", Verdict.FAIL, "Kill switch is DISABLED — agents must have an emergency stop")


def check_replication_depth(agent: Dict[str, Any], strict: bool = False) -> CheckResult:
    """Replication depth should be bounded."""
    depth = agent.get("max_replication_depth")
    if depth is None:
        return CheckResult("replication_depth", Verdict.FAIL, "No replication depth limit set")
    threshold = 2 if strict else 5
    if depth <= threshold:
        return CheckResult("replication_depth", Verdict.PASS, f"Depth limit {depth} ≤ {threshold}",
                           {"limit": depth, "threshold": threshold})
    return CheckResult("replication_depth", Verdict.WARN, f"Depth {depth} exceeds recommended {threshold}",
                       {"limit": depth, "threshold": threshold})


def check_restricted_actions(agent: Dict[str, Any], strict: bool = False) -> CheckResult:
    """Dangerous actions must be in restricted list."""
    dangerous = {"self_modify", "network_exfiltrate", "spawn_unrestricted", "disable_logging", "override_policy"}
    restricted = set(agent.get("restricted_actions", []))
    allowed = set(agent.get("allowed_actions", []))
    leaked = dangerous & allowed - restricted
    if not leaked:
        return CheckResult("restricted_actions", Verdict.PASS, "No dangerous actions leak into allowed set")
    return CheckResult("restricted_actions", Verdict.FAIL,
                       f"Dangerous actions in allowed set: {sorted(leaked)}",
                       {"leaked": sorted(leaked)})


def check_audit_logging(agent: Dict[str, Any], strict: bool = False) -> CheckResult:
    """Audit logging must be enabled."""
    if agent.get("audit_logging", False):
        return CheckResult("audit_logging", Verdict.PASS, "Audit logging enabled")
    if strict:
        return CheckResult("audit_logging", Verdict.FAIL, "Audit logging disabled (required in strict mode)")
    return CheckResult("audit_logging", Verdict.WARN, "Audit logging disabled — recommended for production")


def check_alignment_score(agent: Dict[str, Any], strict: bool = False) -> CheckResult:
    """Alignment score should meet minimum threshold."""
    score = agent.get("alignment_score")
    if score is None:
        return CheckResult("alignment_score", Verdict.WARN, "No alignment score available")
    threshold = 0.9 if strict else 0.7
    if score >= threshold:
        return CheckResult("alignment_score", Verdict.PASS, f"Alignment {score:.2f} ≥ {threshold}",
                           {"score": score, "threshold": threshold})
    if score >= 0.5:
        return CheckResult("alignment_score", Verdict.WARN, f"Alignment {score:.2f} below {threshold}",
                           {"score": score, "threshold": threshold})
    return CheckResult("alignment_score", Verdict.FAIL, f"Alignment {score:.2f} critically low",
                       {"score": score, "threshold": threshold})


def check_resource_limits(agent: Dict[str, Any], strict: bool = False) -> CheckResult:
    """Resource limits must be defined."""
    limits = agent.get("resource_limits", {})
    required = {"max_cpu_percent", "max_memory_mb"}
    missing = required - set(limits.keys())
    if not missing:
        return CheckResult("resource_limits", Verdict.PASS, "Resource limits properly configured",
                           {"limits": limits})
    return CheckResult("resource_limits", Verdict.FAIL, f"Missing resource limits: {sorted(missing)}",
                       {"missing": sorted(missing)})


def check_safety_contract(agent: Dict[str, Any], strict: bool = False) -> CheckResult:
    """Agent should have a safety contract."""
    if agent.get("safety_contract", False):
        return CheckResult("safety_contract", Verdict.PASS, "Safety contract is active")
    if strict:
        return CheckResult("safety_contract", Verdict.FAIL, "No safety contract (required in strict mode)")
    return CheckResult("safety_contract", Verdict.WARN, "No safety contract — recommended")


def check_watermark(agent: Dict[str, Any], strict: bool = False) -> CheckResult:
    """Agent watermarking should be enabled for traceability."""
    if agent.get("watermark_enabled", False):
        return CheckResult("watermark", Verdict.PASS, "Watermarking enabled for traceability")
    return CheckResult("watermark", Verdict.WARN, "Watermarking disabled — traceability reduced")


def check_sandbox(agent: Dict[str, Any], strict: bool = False) -> CheckResult:
    """Sandbox mode check."""
    if agent.get("sandbox_mode", False):
        return CheckResult("sandbox_mode", Verdict.PASS, "Sandbox mode active")
    if strict:
        return CheckResult("sandbox_mode", Verdict.FAIL, "Not sandboxed (required in strict mode)")
    return CheckResult("sandbox_mode", Verdict.WARN, "Not sandboxed — exercise caution")


def check_version(agent: Dict[str, Any], strict: bool = False) -> CheckResult:
    """Agent must have a version identifier."""
    ver = agent.get("version")
    if ver:
        return CheckResult("version", Verdict.PASS, f"Version: {ver}")
    return CheckResult("version", Verdict.FAIL, "No version identifier — unversioned agents cannot be audited")


# ── registry ─────────────────────────────────────────────────────────

BUILTIN_CHECKS: List[Callable[..., CheckResult]] = [
    check_kill_switch,
    check_replication_depth,
    check_restricted_actions,
    check_audit_logging,
    check_alignment_score,
    check_resource_limits,
    check_safety_contract,
    check_watermark,
    check_sandbox,
    check_version,
]


# ── gate runner ──────────────────────────────────────────────────────

def run_gate(
    agent: Dict[str, Any],
    strict: bool = False,
    custom_checks: Optional[List[Dict[str, Any]]] = None,
) -> GateResult:
    """Run all gate checks against an agent configuration.

    Parameters
    ----------
    agent : dict
        Agent configuration to evaluate.
    strict : bool
        If True, apply stricter thresholds.
    custom_checks : list of dict, optional
        Extra threshold checks: ``[{"field": "alignment_score", "op": ">=", "value": 0.85}]``
    """
    result = GateResult()

    for check_fn in BUILTIN_CHECKS:
        result.add(check_fn(agent, strict=strict))

    # Run custom threshold checks
    if custom_checks:
        for cc in custom_checks:
            field_name = cc.get("field", "unknown")
            op = cc.get("op", ">=")
            threshold = cc.get("value", 0)
            actual = agent.get(field_name)
            if actual is None:
                result.add(CheckResult(f"custom:{field_name}", Verdict.WARN,
                                       f"Field '{field_name}' not found in agent config"))
                continue
            passed = False
            if op == ">=" and actual >= threshold:
                passed = True
            elif op == "<=" and actual <= threshold:
                passed = True
            elif op == "==" and actual == threshold:
                passed = True
            elif op == ">" and actual > threshold:
                passed = True
            elif op == "<" and actual < threshold:
                passed = True

            if passed:
                result.add(CheckResult(f"custom:{field_name}", Verdict.PASS,
                                       f"{field_name} = {actual} {op} {threshold}"))
            else:
                result.add(CheckResult(f"custom:{field_name}", Verdict.FAIL,
                                       f"{field_name} = {actual}, required {op} {threshold}"))

    return result


# ── display ──────────────────────────────────────────────────────────

_ICONS = {Verdict.PASS: "✅", Verdict.WARN: "⚠️ ", Verdict.FAIL: "❌"}
_COLORS = {Verdict.PASS: "\033[32m", Verdict.WARN: "\033[33m", Verdict.FAIL: "\033[31m"}
_RESET = "\033[0m"


def print_result(result: GateResult, use_color: bool = True) -> None:
    """Pretty-print gate results to stdout."""
    print("\n╔══════════════════════════════════════════╗")
    print("║       🚦  SAFETY GATE REPORT  🚦        ║")
    print("╚══════════════════════════════════════════╝\n")

    for cr in result.checks:
        icon = _ICONS[cr.verdict]
        c = _COLORS[cr.verdict] if use_color else ""
        r = _RESET if use_color else ""
        print(f"  {icon} {c}{cr.verdict.value:4s}{r}  {cr.name}: {cr.message}")

    print(f"\n{'─' * 46}")
    print(f"  Passed: {result.passed}  |  Warned: {result.warned}  |  Failed: {result.failed}")
    overall_c = _COLORS[result.overall] if use_color else ""
    overall_r = _RESET if use_color else ""
    label = "DEPLOY APPROVED" if result.overall == Verdict.PASS else (
        "DEPLOY WITH CAUTION" if result.overall == Verdict.WARN else "DEPLOY BLOCKED"
    )
    print(f"\n  🚦 Overall: {overall_c}{result.overall.value} — {label}{overall_r}\n")


def result_to_json(result: GateResult) -> str:
    """Serialize gate result to JSON."""
    data = {
        "overall": result.overall.value,
        "passed": result.passed,
        "warned": result.warned,
        "failed": result.failed,
        "checks": [
            {
                "name": c.name,
                "verdict": c.verdict.value,
                "message": c.message,
                **({"details": c.details} if c.details else {}),
            }
            for c in result.checks
        ],
    }
    return json.dumps(data, indent=2)


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: Optional[list] = None) -> None:
    """CLI entry point for safety gate."""
    parser = argparse.ArgumentParser(
        prog="replication gate",
        description="Pre-deployment safety gate — evaluates agent readiness",
    )
    parser.add_argument(
        "--config", "-c",
        help="Path to agent config JSON (uses demo agent if omitted)",
    )
    parser.add_argument(
        "--strict", "-s",
        action="store_true",
        help="Apply strict safety thresholds",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--custom-checks",
        help="Path to JSON file with custom threshold checks",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )

    args = parser.parse_args(argv)

    # Force UTF-8 on Windows
    import io as _io
    if sys.stdout.encoding != "utf-8":
        sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    # Load agent config
    if args.config:
        with open(args.config) as f:
            agent = json.load(f)
    else:
        agent = DEFAULT_AGENT.copy()
        print("  ℹ️  No config provided — using demo agent\n")

    # Load custom checks
    custom = None
    if args.custom_checks:
        with open(args.custom_checks) as f:
            custom = json.load(f)

    result = run_gate(agent, strict=args.strict, custom_checks=custom)

    if args.format == "json":
        print(result_to_json(result))
    else:
        print_result(result, use_color=not args.no_color)

    # Exit code: 0 = pass/warn, 1 = fail (for CI integration)
    sys.exit(0 if result.overall != Verdict.FAIL else 1)


if __name__ == "__main__":
    main()
