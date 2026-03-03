"""Compliance Auditor — check replication contracts against AI safety frameworks.

Evaluates a ``ReplicationContract`` (and optional ``ResourceSpec``) against
configurable compliance frameworks inspired by real-world AI governance
standards (NIST AI RMF, EU AI Act, internal corporate policies).

Each framework defines a set of *checks*.  A check inspects contract
parameters and emits a PASS, WARN, or FAIL finding with a human-readable
rationale.  The auditor aggregates findings into a structured report with
per-framework verdicts and an overall compliance score.

Usage (CLI)::

    python -m replication.compliance                               # all frameworks
    python -m replication.compliance --framework nist               # single framework
    python -m replication.compliance --max-depth 3 --max-replicas 5
    python -m replication.compliance --cooldown 10 --expiration 300
    python -m replication.compliance --allow-external               # network flag
    python -m replication.compliance --json                         # JSON output

Programmatic::

    from replication.compliance import ComplianceAuditor, AuditConfig
    auditor = ComplianceAuditor()
    result = auditor.audit(contract, resources=spec)
    print(result.render())
    print(f"Overall: {result.overall_verdict} ({result.score}/100)")
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .contract import ReplicationContract, ResourceSpec, NetworkPolicy


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class Verdict(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class Framework(str, Enum):
    NIST = "nist"
    EU_AI_ACT = "eu_ai_act"
    INTERNAL = "internal"


@dataclass
class Finding:
    """Single compliance check result."""

    framework: Framework
    check_id: str
    title: str
    verdict: Verdict
    rationale: str
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "framework": self.framework.value,
            "check_id": self.check_id,
            "title": self.title,
            "verdict": self.verdict.value,
            "rationale": self.rationale,
        }
        if self.recommendation:
            d["recommendation"] = self.recommendation
        return d


@dataclass
class FrameworkResult:
    """Aggregate result for one framework."""

    framework: Framework
    findings: List[Finding] = field(default_factory=list)

    @property
    def passes(self) -> int:
        return sum(1 for f in self.findings if f.verdict == Verdict.PASS)

    @property
    def warns(self) -> int:
        return sum(1 for f in self.findings if f.verdict == Verdict.WARN)

    @property
    def fails(self) -> int:
        return sum(1 for f in self.findings if f.verdict == Verdict.FAIL)

    @property
    def verdict(self) -> Verdict:
        if self.fails > 0:
            return Verdict.FAIL
        if self.warns > 0:
            return Verdict.WARN
        return Verdict.PASS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "framework": self.framework.value,
            "verdict": self.verdict.value,
            "pass": self.passes,
            "warn": self.warns,
            "fail": self.fails,
            "findings": [f.to_dict() for f in self.findings],
        }


@dataclass
class AuditConfig:
    """Control which frameworks to run."""

    frameworks: Optional[List[Framework]] = None  # None → all

    def active_frameworks(self) -> List[Framework]:
        return self.frameworks if self.frameworks else list(Framework)


@dataclass
class AuditResult:
    """Complete audit report."""

    framework_results: List[FrameworkResult] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def total_findings(self) -> int:
        return sum(len(fr.findings) for fr in self.framework_results)

    @property
    def total_passes(self) -> int:
        return sum(fr.passes for fr in self.framework_results)

    @property
    def total_warns(self) -> int:
        return sum(fr.warns for fr in self.framework_results)

    @property
    def total_fails(self) -> int:
        return sum(fr.fails for fr in self.framework_results)

    @property
    def score(self) -> int:
        """0-100 score: PASS=100%, WARN=50%, FAIL=0%."""
        total = self.total_findings
        if total == 0:
            return 100
        pts = self.total_passes * 100 + self.total_warns * 50
        return round(pts / total)

    @property
    def overall_verdict(self) -> Verdict:
        if self.total_fails > 0:
            return Verdict.FAIL
        if self.total_warns > 0:
            return Verdict.WARN
        return Verdict.PASS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "overall_verdict": self.overall_verdict.value,
            "score": self.score,
            "summary": {
                "total": self.total_findings,
                "pass": self.total_passes,
                "warn": self.total_warns,
                "fail": self.total_fails,
            },
            "frameworks": [fr.to_dict() for fr in self.framework_results],
        }

    def render(self) -> str:
        """Human-readable audit report."""
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("  COMPLIANCE AUDIT REPORT")
        lines.append("=" * 60)
        lines.append(f"  Timestamp : {self.timestamp}")
        lines.append(f"  Verdict   : {self.overall_verdict.value}")
        lines.append(f"  Score     : {self.score}/100")
        lines.append(
            f"  Findings  : {self.total_passes} pass, "
            f"{self.total_warns} warn, {self.total_fails} fail"
        )
        lines.append("")

        for fr in self.framework_results:
            icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}[fr.verdict.value]
            lines.append(f"─── {fr.framework.value.upper()} {icon} ───")
            for f in fr.findings:
                mark = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}[f.verdict.value]
                lines.append(f"  [{mark}] {f.check_id}: {f.title}")
                lines.append(f"      {f.rationale}")
                if f.recommendation:
                    lines.append(f"      → {f.recommendation}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Check definitions per framework
# ---------------------------------------------------------------------------

CheckFn = Callable[
    ["ReplicationContract", Optional["ResourceSpec"]], Finding
]


def _nist_depth_limit(
    contract: ReplicationContract, resources: Optional[ResourceSpec]
) -> Finding:
    """NIST MAP 1.5 / GOVERN 1.1 — bounded replication depth."""
    if contract.max_depth <= 3:
        return Finding(
            Framework.NIST, "NIST-01", "Replication depth limit",
            Verdict.PASS,
            f"max_depth={contract.max_depth} is within safe bounds (≤3).",
        )
    if contract.max_depth <= 6:
        return Finding(
            Framework.NIST, "NIST-01", "Replication depth limit",
            Verdict.WARN,
            f"max_depth={contract.max_depth} is moderate (4-6); consider tightening.",
            "Reduce max_depth to ≤3 for stronger containment.",
        )
    return Finding(
        Framework.NIST, "NIST-01", "Replication depth limit",
        Verdict.FAIL,
        f"max_depth={contract.max_depth} exceeds safe threshold (>6).",
        "Set max_depth ≤3 to comply with NIST risk management guidance.",
    )


def _nist_stop_conditions(
    contract: ReplicationContract, resources: Optional[ResourceSpec]
) -> Finding:
    """NIST GOVERN 1.2 — kill switches / stop conditions."""
    n = len(contract.stop_conditions)
    if n >= 2:
        return Finding(
            Framework.NIST, "NIST-02", "Stop conditions defined",
            Verdict.PASS,
            f"{n} stop condition(s) defined; automated shutdown is in place.",
        )
    if n == 1:
        return Finding(
            Framework.NIST, "NIST-02", "Stop conditions defined",
            Verdict.WARN,
            "Only 1 stop condition; redundancy recommended.",
            "Add at least one additional independent stop condition.",
        )
    return Finding(
        Framework.NIST, "NIST-02", "Stop conditions defined",
        Verdict.FAIL,
        "No stop conditions defined; system cannot auto-halt.",
        "Define ≥2 stop conditions for robust containment.",
    )


def _nist_expiration(
    contract: ReplicationContract, resources: Optional[ResourceSpec]
) -> Finding:
    """NIST MEASURE 2.6 — time-bounded operation."""
    if contract.expiration_seconds is not None:
        if contract.expiration_seconds <= 3600:
            return Finding(
                Framework.NIST, "NIST-03", "Expiration set",
                Verdict.PASS,
                f"Expiration={contract.expiration_seconds}s (≤1h); time-bounded.",
            )
        return Finding(
            Framework.NIST, "NIST-03", "Expiration set",
            Verdict.WARN,
            f"Expiration={contract.expiration_seconds}s (>1h); long-running.",
            "Consider shorter expiration for tighter control.",
        )
    return Finding(
        Framework.NIST, "NIST-03", "Expiration set",
        Verdict.FAIL,
        "No expiration set; replicas may run indefinitely.",
        "Set expiration_seconds to bound operation lifetime.",
    )


def _eu_replica_cap(
    contract: ReplicationContract, resources: Optional[ResourceSpec]
) -> Finding:
    """EU AI Act Art. 9 — proportional replica limits."""
    if contract.max_replicas <= 5:
        return Finding(
            Framework.EU_AI_ACT, "EU-01", "Replica population cap",
            Verdict.PASS,
            f"max_replicas={contract.max_replicas}; population is controlled (≤5).",
        )
    if contract.max_replicas <= 20:
        return Finding(
            Framework.EU_AI_ACT, "EU-01", "Replica population cap",
            Verdict.WARN,
            f"max_replicas={contract.max_replicas}; moderate population (6-20).",
            "Reduce to ≤5 for high-risk classification compliance.",
        )
    return Finding(
        Framework.EU_AI_ACT, "EU-01", "Replica population cap",
        Verdict.FAIL,
        f"max_replicas={contract.max_replicas}; excessive (>20).",
        "Cap replicas at ≤5 per EU AI Act proportionality principle.",
    )


def _eu_network_isolation(
    contract: ReplicationContract, resources: Optional[ResourceSpec]
) -> Finding:
    """EU AI Act Art. 15 — network sandboxing."""
    if resources is None:
        return Finding(
            Framework.EU_AI_ACT, "EU-02", "Network isolation",
            Verdict.WARN,
            "No ResourceSpec provided; cannot verify network policy.",
            "Supply ResourceSpec with NetworkPolicy for full audit.",
        )
    if not resources.network_policy.allow_external:
        return Finding(
            Framework.EU_AI_ACT, "EU-02", "Network isolation",
            Verdict.PASS,
            "External network access blocked; sandbox is isolated.",
        )
    return Finding(
        Framework.EU_AI_ACT, "EU-02", "Network isolation",
        Verdict.FAIL,
        "External network access allowed; sandbox can reach the internet.",
        "Set network_policy.allow_external=False for isolation.",
    )


def _eu_cooldown(
    contract: ReplicationContract, resources: Optional[ResourceSpec]
) -> Finding:
    """EU AI Act Art. 9 — replication rate control."""
    if contract.cooldown_seconds >= 5:
        return Finding(
            Framework.EU_AI_ACT, "EU-03", "Replication cooldown",
            Verdict.PASS,
            f"cooldown={contract.cooldown_seconds}s; adequate rate limiting (≥5s).",
        )
    if contract.cooldown_seconds >= 1:
        return Finding(
            Framework.EU_AI_ACT, "EU-03", "Replication cooldown",
            Verdict.WARN,
            f"cooldown={contract.cooldown_seconds}s; short (1-4s).",
            "Increase cooldown to ≥5s for safer replication pacing.",
        )
    return Finding(
        Framework.EU_AI_ACT, "EU-03", "Replication cooldown",
        Verdict.FAIL,
        f"cooldown={contract.cooldown_seconds}s; near-instant replication.",
        "Set cooldown_seconds ≥5 to prevent runaway replication.",
    )


def _internal_resource_limits(
    contract: ReplicationContract, resources: Optional[ResourceSpec]
) -> Finding:
    """Internal policy — resource bounds per replica."""
    if resources is None:
        return Finding(
            Framework.INTERNAL, "INT-01", "Resource limits",
            Verdict.WARN,
            "No ResourceSpec provided; resource bounds unknown.",
            "Provide ResourceSpec for complete audit.",
        )
    issues: List[str] = []
    if resources.cpu_limit > 4.0:
        issues.append(f"cpu_limit={resources.cpu_limit} (>4 cores)")
    if resources.memory_limit_mb > 4096:
        issues.append(f"memory={resources.memory_limit_mb}MB (>4GB)")
    if issues:
        return Finding(
            Framework.INTERNAL, "INT-01", "Resource limits",
            Verdict.FAIL,
            "Excessive resources: " + "; ".join(issues),
            "Limit CPU to ≤4 cores and memory to ≤4096 MB per replica.",
        )
    return Finding(
        Framework.INTERNAL, "INT-01", "Resource limits",
        Verdict.PASS,
        f"CPU={resources.cpu_limit}, memory={resources.memory_limit_mb}MB; within bounds.",
    )


def _internal_blast_radius(
    contract: ReplicationContract, resources: Optional[ResourceSpec]
) -> Finding:
    """Internal policy — total blast radius (depth × replicas)."""
    blast = contract.max_depth * contract.max_replicas
    if blast <= 15:
        return Finding(
            Framework.INTERNAL, "INT-02", "Blast radius",
            Verdict.PASS,
            f"Blast radius (depth×replicas) = {blast}; contained (≤15).",
        )
    if blast <= 50:
        return Finding(
            Framework.INTERNAL, "INT-02", "Blast radius",
            Verdict.WARN,
            f"Blast radius = {blast}; moderate (16-50).",
            "Reduce max_depth or max_replicas to lower blast radius.",
        )
    return Finding(
        Framework.INTERNAL, "INT-02", "Blast radius",
        Verdict.FAIL,
        f"Blast radius = {blast}; dangerously large (>50).",
        "Dramatically reduce max_depth and/or max_replicas.",
    )


def _internal_controller_access(
    contract: ReplicationContract, resources: Optional[ResourceSpec]
) -> Finding:
    """Internal policy — controller connectivity."""
    if resources is None:
        return Finding(
            Framework.INTERNAL, "INT-03", "Controller access",
            Verdict.WARN,
            "No ResourceSpec; cannot verify controller connectivity.",
        )
    if resources.network_policy.allow_controller:
        return Finding(
            Framework.INTERNAL, "INT-03", "Controller access",
            Verdict.PASS,
            "Workers can reach controller for monitoring and shutdown.",
        )
    return Finding(
        Framework.INTERNAL, "INT-03", "Controller access",
        Verdict.FAIL,
        "Controller access disabled; workers cannot be monitored/halted.",
        "Set network_policy.allow_controller=True.",
    )


# ---------------------------------------------------------------------------
# Framework → checks registry
# ---------------------------------------------------------------------------

FRAMEWORK_CHECKS: Dict[Framework, List[CheckFn]] = {
    Framework.NIST: [_nist_depth_limit, _nist_stop_conditions, _nist_expiration],
    Framework.EU_AI_ACT: [_eu_replica_cap, _eu_network_isolation, _eu_cooldown],
    Framework.INTERNAL: [
        _internal_resource_limits,
        _internal_blast_radius,
        _internal_controller_access,
    ],
}


# ---------------------------------------------------------------------------
# Auditor
# ---------------------------------------------------------------------------


class ComplianceAuditor:
    """Run compliance checks against a replication contract."""

    def audit(
        self,
        contract: ReplicationContract,
        resources: Optional[ResourceSpec] = None,
        config: Optional[AuditConfig] = None,
    ) -> AuditResult:
        cfg = config or AuditConfig()
        result = AuditResult()

        for fw in cfg.active_frameworks():
            checks = FRAMEWORK_CHECKS.get(fw, [])
            fr = FrameworkResult(framework=fw)
            for check_fn in checks:
                finding = check_fn(contract, resources)
                fr.findings.append(finding)
            result.framework_results.append(fr)

        return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Audit a replication contract against compliance frameworks."
    )
    p.add_argument(
        "--framework", choices=[f.value for f in Framework],
        help="Run only this framework (default: all).",
    )
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--max-replicas", type=int, default=5)
    p.add_argument("--cooldown", type=float, default=5.0)
    p.add_argument("--expiration", type=float, default=None)
    p.add_argument("--cpu-limit", type=float, default=2.0)
    p.add_argument("--memory-limit", type=int, default=2048)
    p.add_argument("--allow-external", action="store_true")
    p.add_argument("--no-controller", action="store_true")
    p.add_argument("--json", action="store_true", help="JSON output.")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    contract = ReplicationContract(
        max_depth=args.max_depth,
        max_replicas=args.max_replicas,
        cooldown_seconds=args.cooldown,
        expiration_seconds=args.expiration,
    )

    np = NetworkPolicy(
        allow_controller=not args.no_controller,
        allow_external=args.allow_external,
    )
    resources = ResourceSpec(
        cpu_limit=args.cpu_limit,
        memory_limit_mb=args.memory_limit,
        network_policy=np,
    )

    config = None
    if args.framework:
        config = AuditConfig(frameworks=[Framework(args.framework)])

    auditor = ComplianceAuditor()
    result = auditor.audit(contract, resources=resources, config=config)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.render())

    sys.exit(0 if result.overall_verdict == Verdict.PASS else 1)


if __name__ == "__main__":
    main()

