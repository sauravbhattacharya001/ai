"""Preflight Check — pre-simulation validation for sandbox configurations.

Validates contract constraints, resource limits, policy consistency, and
stop condition coverage before running simulations. Produces a go/no-go
assessment with categorized findings.

Usage (CLI)::

    python -m replication preflight                          # default config check
    python -m replication preflight --strategy greedy        # check specific strategy
    python -m replication preflight --max-depth 5            # override max_depth
    python -m replication preflight --max-replicas 20        # override max_replicas
    python -m replication preflight --cpu 0.5 --memory 256   # resource constraints
    python -m replication preflight --policy strict          # validate against policy preset
    python -m replication preflight --json                   # JSON output
    python -m replication preflight --fix                    # suggest fixes for issues
    python -m replication preflight --all-strategies         # check all strategies

Programmatic::

    from replication.preflight import PreflightChecker, PreflightConfig
    checker = PreflightChecker(PreflightConfig(max_depth=3, max_replicas=10))
    result = checker.run()
    print(result.render())
    assert result.passed, f"Preflight failed: {result.failures}"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ── Enums ────────────────────────────────────────────────────────────────

class Severity(Enum):
    """Finding severity level."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

    @property
    def symbol(self) -> str:
        return {"error": "✖", "warning": "⚠", "info": "ℹ"}[self.value]

    @property
    def color_code(self) -> str:
        return {"error": "\033[91m", "warning": "\033[93m", "info": "\033[94m"}[self.value]


class Category(Enum):
    """Check category."""
    CONTRACT = "contract"
    RESOURCES = "resources"
    POLICY = "policy"
    STOP_CONDITIONS = "stop_conditions"
    STRATEGY = "strategy"
    SCALABILITY = "scalability"


# ── Data Classes ─────────────────────────────────────────────────────────

@dataclass
class Finding:
    """A single preflight finding."""
    category: Category
    severity: Severity
    code: str
    message: str
    fix: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "category": self.category.value,
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
        }
        if self.fix:
            d["fix"] = self.fix
        return d


@dataclass
class PreflightConfig:
    """Configuration to validate."""
    max_depth: int = 3
    max_replicas: int = 10
    cooldown_seconds: float = 1.0
    expiration_seconds: Optional[float] = None
    cpu_limit: float = 1.0
    memory_limit_mb: int = 512
    allow_external_network: bool = False
    strategy: Optional[str] = None
    policy_preset: Optional[str] = None


@dataclass
class PreflightResult:
    """Result of preflight validation."""
    findings: List[Finding] = field(default_factory=list)
    config: Optional[PreflightConfig] = None
    elapsed_ms: float = 0.0

    @property
    def errors(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == Severity.ERROR]

    @property
    def warnings(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == Severity.WARNING]

    @property
    def infos(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == Severity.INFO]

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    @property
    def failures(self) -> List[str]:
        return [f.message for f in self.errors]

    @property
    def verdict(self) -> str:
        if self.passed and not self.warnings:
            return "GO"
        elif self.passed:
            return "GO (with warnings)"
        else:
            return "NO-GO"

    def by_category(self) -> Dict[str, List[Finding]]:
        result: Dict[str, List[Finding]] = {}
        for f in self.findings:
            result.setdefault(f.category.value, []).append(f)
        return result

    def render(self, show_fixes: bool = False) -> str:
        """Render a human-readable report."""
        lines: List[str] = []
        reset = "\033[0m"
        bold = "\033[1m"
        green = "\033[92m"
        red = "\033[91m"

        lines.append(f"\n{bold}═══ Preflight Check Report ═══{reset}\n")

        if self.config:
            lines.append(f"  Strategy:    {self.config.strategy or 'default'}")
            lines.append(f"  Max Depth:   {self.config.max_depth}")
            lines.append(f"  Max Replicas:{self.config.max_replicas}")
            lines.append(f"  Cooldown:    {self.config.cooldown_seconds}s")
            lines.append(f"  CPU Limit:   {self.config.cpu_limit}")
            lines.append(f"  Memory:      {self.config.memory_limit_mb} MB")
            if self.config.policy_preset:
                lines.append(f"  Policy:      {self.config.policy_preset}")
            lines.append("")

        by_cat = self.by_category()
        for cat_name, findings in sorted(by_cat.items()):
            lines.append(f"  {bold}[{cat_name.upper()}]{reset}")
            for f in findings:
                color = f.severity.color_code
                lines.append(f"    {color}{f.severity.symbol}{reset} {f.code}: {f.message}")
                if show_fixes and f.fix:
                    lines.append(f"      → Fix: {f.fix}")
            lines.append("")

        # Summary
        e, w, i = len(self.errors), len(self.warnings), len(self.infos)
        verdict_color = green if self.passed else red
        lines.append(f"  {bold}Verdict: {verdict_color}{self.verdict}{reset}")
        lines.append(f"  {e} error(s), {w} warning(s), {i} info(s)")
        lines.append(f"  Completed in {self.elapsed_ms:.1f} ms\n")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "verdict": self.verdict,
            "passed": self.passed,
            "counts": {
                "errors": len(self.errors),
                "warnings": len(self.warnings),
                "infos": len(self.infos),
            },
            "findings": [f.to_dict() for f in self.findings],
            "elapsed_ms": round(self.elapsed_ms, 1),
        }
        if self.config:
            d["config"] = {
                "max_depth": self.config.max_depth,
                "max_replicas": self.config.max_replicas,
                "cooldown_seconds": self.config.cooldown_seconds,
                "cpu_limit": self.config.cpu_limit,
                "memory_limit_mb": self.config.memory_limit_mb,
                "strategy": self.config.strategy,
                "policy_preset": self.config.policy_preset,
            }
        return d


# ── Known Strategies ─────────────────────────────────────────────────────

KNOWN_STRATEGIES = ["greedy", "conservative", "adaptive", "random", "exponential"]


# ── Checker ──────────────────────────────────────────────────────────────

class PreflightChecker:
    """Runs preflight validation checks on a sandbox configuration."""

    def __init__(self, config: PreflightConfig) -> None:
        self.config = config
        self._findings: List[Finding] = []

    def _add(self, category: Category, severity: Severity, code: str,
             message: str, fix: Optional[str] = None) -> None:
        self._findings.append(Finding(category, severity, code, message, fix))

    def _check_contract(self) -> None:
        """Validate contract parameter ranges."""
        c = self.config

        if c.max_depth < 0:
            self._add(Category.CONTRACT, Severity.ERROR, "CTR-001",
                      f"max_depth must be >= 0, got {c.max_depth}",
                      "Set max_depth to 0 or higher")
        elif c.max_depth == 0:
            self._add(Category.CONTRACT, Severity.WARNING, "CTR-002",
                      "max_depth is 0 — no replication will occur",
                      "Set max_depth >= 1 to allow replication")
        elif c.max_depth > 10:
            self._add(Category.CONTRACT, Severity.WARNING, "CTR-003",
                      f"max_depth={c.max_depth} is unusually high — deep chains are hard to control",
                      "Consider max_depth <= 5 for manageable chains")
        else:
            self._add(Category.CONTRACT, Severity.INFO, "CTR-010",
                      f"max_depth={c.max_depth} is within recommended range")

        if c.max_replicas < 1:
            self._add(Category.CONTRACT, Severity.ERROR, "CTR-004",
                      f"max_replicas must be >= 1, got {c.max_replicas}",
                      "Set max_replicas to at least 1")
        elif c.max_replicas > 100:
            self._add(Category.CONTRACT, Severity.WARNING, "CTR-005",
                      f"max_replicas={c.max_replicas} is very high — resource exhaustion likely",
                      "Consider max_replicas <= 50")
        else:
            self._add(Category.CONTRACT, Severity.INFO, "CTR-011",
                      f"max_replicas={c.max_replicas} is within recommended range")

        if c.cooldown_seconds < 0:
            self._add(Category.CONTRACT, Severity.ERROR, "CTR-006",
                      f"cooldown_seconds must be >= 0, got {c.cooldown_seconds}",
                      "Set cooldown_seconds to 0 or higher")
        elif c.cooldown_seconds == 0:
            self._add(Category.CONTRACT, Severity.WARNING, "CTR-007",
                      "cooldown_seconds is 0 — rapid-fire replication possible",
                      "Set cooldown_seconds >= 0.5 to prevent replication storms")

        if c.expiration_seconds is not None:
            if c.expiration_seconds <= 0:
                self._add(Category.CONTRACT, Severity.ERROR, "CTR-008",
                          f"expiration_seconds must be > 0, got {c.expiration_seconds}",
                          "Set a positive expiration or remove it")
            elif c.expiration_seconds < c.cooldown_seconds:
                self._add(Category.CONTRACT, Severity.ERROR, "CTR-009",
                          "expiration_seconds < cooldown_seconds — workers expire before they can replicate",
                          "Increase expiration_seconds above cooldown_seconds")

    def _check_resources(self) -> None:
        """Validate resource constraints."""
        c = self.config

        if c.cpu_limit <= 0:
            self._add(Category.RESOURCES, Severity.ERROR, "RES-001",
                      f"cpu_limit must be > 0, got {c.cpu_limit}",
                      "Set a positive CPU limit")
        elif c.cpu_limit < 0.1:
            self._add(Category.RESOURCES, Severity.WARNING, "RES-002",
                      f"cpu_limit={c.cpu_limit} is very low — workers may be starved",
                      "Consider cpu_limit >= 0.25")

        if c.memory_limit_mb <= 0:
            self._add(Category.RESOURCES, Severity.ERROR, "RES-003",
                      f"memory_limit_mb must be > 0, got {c.memory_limit_mb}",
                      "Set a positive memory limit")
        elif c.memory_limit_mb < 64:
            self._add(Category.RESOURCES, Severity.WARNING, "RES-004",
                      f"memory_limit_mb={c.memory_limit_mb} is very low — workers may OOM",
                      "Consider memory_limit_mb >= 128")

        # Total resource projection
        total_cpu = c.cpu_limit * c.max_replicas
        total_mem = c.memory_limit_mb * c.max_replicas

        if total_cpu > 32:
            self._add(Category.RESOURCES, Severity.WARNING, "RES-005",
                      f"Total CPU at max replicas: {total_cpu:.1f} cores — exceeds typical host",
                      f"Reduce max_replicas or cpu_limit (currently {c.max_replicas} × {c.cpu_limit})")
        if total_mem > 65536:
            self._add(Category.RESOURCES, Severity.WARNING, "RES-006",
                      f"Total memory at max replicas: {total_mem} MB — exceeds typical host (64 GB)",
                      f"Reduce max_replicas or memory_limit_mb")

        self._add(Category.RESOURCES, Severity.INFO, "RES-010",
                  f"Resource projection: {total_cpu:.1f} CPU cores, {total_mem} MB memory at max capacity")

    def _check_strategy(self) -> None:
        """Validate strategy selection."""
        c = self.config
        if c.strategy is None:
            self._add(Category.STRATEGY, Severity.INFO, "STR-001",
                      "No strategy specified — simulation will use default")
            return

        if c.strategy not in KNOWN_STRATEGIES:
            self._add(Category.STRATEGY, Severity.ERROR, "STR-002",
                      f"Unknown strategy '{c.strategy}' — valid: {', '.join(KNOWN_STRATEGIES)}",
                      f"Use one of: {', '.join(KNOWN_STRATEGIES)}")
            return

        # Strategy-specific warnings
        if c.strategy == "greedy" and c.cooldown_seconds == 0:
            self._add(Category.STRATEGY, Severity.WARNING, "STR-003",
                      "Greedy strategy with 0 cooldown — expect rapid resource saturation",
                      "Add cooldown or use 'conservative' strategy")
        elif c.strategy == "exponential" and c.max_depth > 5:
            self._add(Category.STRATEGY, Severity.WARNING, "STR-004",
                      f"Exponential strategy with max_depth={c.max_depth} — replica count grows as 2^depth",
                      "Consider max_depth <= 4 with exponential strategy")

        self._add(Category.STRATEGY, Severity.INFO, "STR-010",
                  f"Strategy '{c.strategy}' is valid")

    def _check_policy(self) -> None:
        """Validate policy preset if specified."""
        c = self.config
        if c.policy_preset is None:
            self._add(Category.POLICY, Severity.INFO, "POL-001",
                      "No policy preset specified — simulation runs without policy validation")
            return

        try:
            from .policy import PRESETS as POLICY_PRESETS
            available = list(POLICY_PRESETS.keys()) if isinstance(POLICY_PRESETS, dict) else []
        except (ImportError, AttributeError):
            available = ["strict", "standard", "minimal"]

        if c.policy_preset not in available:
            self._add(Category.POLICY, Severity.ERROR, "POL-002",
                      f"Unknown policy preset '{c.policy_preset}' — valid: {', '.join(available)}",
                      f"Use one of: {', '.join(available)}")
        else:
            self._add(Category.POLICY, Severity.INFO, "POL-010",
                      f"Policy preset '{c.policy_preset}' is valid")

    def _check_scalability(self) -> None:
        """Check for potential scalability issues."""
        c = self.config

        # Worst case replica count for chain topologies
        if c.max_depth > 0:
            worst_case = min(2 ** c.max_depth, c.max_replicas)
            if worst_case > 50:
                self._add(Category.SCALABILITY, Severity.WARNING, "SCL-001",
                          f"Worst-case replica count: {worst_case} — simulation may be slow",
                          "Reduce max_depth or max_replicas for faster simulations")

            # Replication storm window
            if c.cooldown_seconds > 0 and c.max_replicas > 1:
                storm_time = c.cooldown_seconds * c.max_depth
                self._add(Category.SCALABILITY, Severity.INFO, "SCL-002",
                          f"Minimum time to max depth: {storm_time:.1f}s (depth × cooldown)")

        if c.allow_external_network:
            self._add(Category.SCALABILITY, Severity.WARNING, "SCL-003",
                      "External network access enabled — workers can reach the internet",
                      "Disable external network unless required for testing")

    def _check_stop_conditions(self) -> None:
        """Check stop condition coverage."""
        c = self.config

        # Without stop conditions, only hard limits apply
        has_depth_limit = c.max_depth > 0
        has_replica_limit = c.max_replicas < 100
        has_cooldown = c.cooldown_seconds > 0
        has_expiration = c.expiration_seconds is not None

        safeguards = sum([has_depth_limit, has_replica_limit, has_cooldown, has_expiration])

        if safeguards < 2:
            self._add(Category.STOP_CONDITIONS, Severity.WARNING, "STP-001",
                      f"Only {safeguards} safeguard(s) active — consider adding more",
                      "Enable at least 2 of: depth limit, replica limit, cooldown, expiration")
        else:
            self._add(Category.STOP_CONDITIONS, Severity.INFO, "STP-010",
                      f"{safeguards} safeguards active — good coverage")

        if not has_expiration:
            self._add(Category.STOP_CONDITIONS, Severity.INFO, "STP-002",
                      "No expiration set — workers persist indefinitely")

    def run(self) -> PreflightResult:
        """Run all preflight checks and return results."""
        self._findings = []
        start = time.time()

        self._check_contract()
        self._check_resources()
        self._check_strategy()
        self._check_policy()
        self._check_scalability()
        self._check_stop_conditions()

        elapsed = (time.time() - start) * 1000
        return PreflightResult(
            findings=self._findings,
            config=self.config,
            elapsed_ms=elapsed,
        )


# ── CLI ──────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for preflight checks."""
    parser = argparse.ArgumentParser(
        description="Pre-simulation validation — check sandbox config before running"
    )
    parser.add_argument("--max-depth", type=int, default=3, help="Max replication depth (default: 3)")
    parser.add_argument("--max-replicas", type=int, default=10, help="Max replica count (default: 10)")
    parser.add_argument("--cooldown", type=float, default=1.0, help="Cooldown seconds (default: 1.0)")
    parser.add_argument("--expiration", type=float, default=None, help="Worker expiration seconds")
    parser.add_argument("--cpu", type=float, default=1.0, help="CPU limit per worker (default: 1.0)")
    parser.add_argument("--memory", type=int, default=512, help="Memory limit MB per worker (default: 512)")
    parser.add_argument("--strategy", type=str, default=None, help="Simulation strategy to validate")
    parser.add_argument("--policy", type=str, default=None, help="Policy preset to validate against")
    parser.add_argument("--allow-external", action="store_true", help="Allow external network access")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")
    parser.add_argument("--fix", action="store_true", help="Show suggested fixes for issues")
    parser.add_argument("--all-strategies", action="store_true", help="Check all known strategies")

    args = parser.parse_args(argv)

    strategies_to_check: List[Optional[str]] = [args.strategy]
    if args.all_strategies:
        strategies_to_check = [None] + KNOWN_STRATEGIES  # type: ignore[list-item]

    all_results: List[Dict[str, Any]] = []
    any_failed = False

    for strategy in strategies_to_check:
        config = PreflightConfig(
            max_depth=args.max_depth,
            max_replicas=args.max_replicas,
            cooldown_seconds=args.cooldown,
            expiration_seconds=args.expiration,
            cpu_limit=args.cpu,
            memory_limit_mb=args.memory,
            allow_external_network=args.allow_external,
            strategy=strategy,
            policy_preset=args.policy,
        )

        checker = PreflightChecker(config)
        result = checker.run()

        if not result.passed:
            any_failed = True

        if args.json_output:
            all_results.append(result.to_dict())
        else:
            print(result.render(show_fixes=args.fix))

    if args.json_output:
        output = all_results if len(all_results) > 1 else all_results[0]
        print(json.dumps(output, indent=2))

    if any_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
