"""Containment Verifier — sandbox containment enforcement verification.

Validates that agent containment measures (network isolation, filesystem
restrictions, capability boundaries, resource limits, IPC controls) are
properly configured and enforced.  Generates compliance reports with
per-control pass/fail/warn status, overall containment scores, and
remediation recommendations.

Usage::

    from replication.containment import (
        ContainmentVerifier, ContainmentPolicy, ContainmentControl,
        ControlCategory, ControlSeverity, VerificationResult,
    )

    policy = ContainmentPolicy.strict()  # or .standard() / .minimal()
    verifier = ContainmentVerifier(policy)

    sandbox_state = {
        "agent_id": "agent-007",
        "network": {"outbound_allowed": False, "inbound_allowed": False,
                     "dns_allowed": False, "allowed_hosts": [], "open_ports": []},
        "filesystem": {"writable_paths": ["/tmp/agent"], "readable_paths": ["/tmp/agent"],
                        "denied_paths": ["/etc", "/var"], "max_disk_mb": 512, "used_disk_mb": 45},
        "capabilities": {"can_spawn_processes": False, "can_modify_self": False,
                          "can_escalate_privileges": False, "can_access_hardware": False,
                          "allowed_syscalls": ["read", "write", "open", "close"]},
        "resources": {"max_cpu_percent": 25.0, "max_memory_mb": 1024,
                       "max_open_files": 64, "max_threads": 4, "max_runtime_seconds": 3600},
        "ipc": {"allowed_channels": ["supervisor"], "message_size_limit_kb": 64,
                 "rate_limit_per_second": 10, "encrypted": True},
    }

    result = verifier.verify(sandbox_state)
    print(f"Score: {result.score}/100  Grade: {result.grade}")
    print(verifier.report(result))

    # Fleet verification
    fleet_report = verifier.verify_fleet({"a": state_a, "b": state_b})

    # Comparison
    diff = verifier.compare(state_a, state_b)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ControlCategory(Enum):
    NETWORK = "network"
    FILESYSTEM = "filesystem"
    CAPABILITY = "capability"
    RESOURCE = "resource"
    IPC = "ipc"


class ControlSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class FindingStatus(Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


@dataclass
class ContainmentControl:
    id: str
    name: str
    category: ControlCategory
    severity: ControlSeverity
    description: str
    check_key: str
    expected: Any = None
    comparator: str = "eq"

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "category": self.category.value,
                "severity": self.severity.value, "description": self.description,
                "check_key": self.check_key, "expected": self.expected, "comparator": self.comparator}


@dataclass
class Finding:
    control: ContainmentControl
    status: FindingStatus
    actual: Any = None
    message: str = ""

    def to_dict(self) -> dict:
        return {"control_id": self.control.id, "control_name": self.control.name,
                "category": self.control.category.value, "severity": self.control.severity.value,
                "status": self.status.value, "actual": self.actual, "message": self.message}


@dataclass
class VerificationResult:
    agent_id: str
    timestamp: float
    findings: list[Finding] = field(default_factory=list)
    policy_name: str = ""

    @property
    def total(self) -> int:
        return len([f for f in self.findings if f.status != FindingStatus.SKIP])

    @property
    def passed(self) -> int:
        return len([f for f in self.findings if f.status == FindingStatus.PASS])

    @property
    def failed(self) -> int:
        return len([f for f in self.findings if f.status == FindingStatus.FAIL])

    @property
    def warned(self) -> int:
        return len([f for f in self.findings if f.status == FindingStatus.WARN])

    @property
    def skipped(self) -> int:
        return len([f for f in self.findings if f.status == FindingStatus.SKIP])

    @property
    def score(self) -> float:
        if self.total == 0:
            return 100.0
        weights = {ControlSeverity.CRITICAL: 25, ControlSeverity.HIGH: 15,
                   ControlSeverity.MEDIUM: 8, ControlSeverity.LOW: 3, ControlSeverity.INFO: 1}
        max_score = earned = 0.0
        for f in self.findings:
            if f.status == FindingStatus.SKIP:
                continue
            w = weights.get(f.control.severity, 1)
            max_score += w
            if f.status == FindingStatus.PASS:
                earned += w
            elif f.status == FindingStatus.WARN:
                earned += w * 0.5
        return round(earned / max_score * 100, 1) if max_score else 100.0

    @property
    def grade(self) -> str:
        s = self.score
        for thresh, g in [(95,"A+"),(90,"A"),(85,"A-"),(80,"B+"),(75,"B"),(70,"B-"),(65,"C+"),(60,"C"),(55,"C-"),(50,"D")]:
            if s >= thresh:
                return g
        return "F"

    @property
    def critical_failures(self) -> list[Finding]:
        return [f for f in self.findings if f.status == FindingStatus.FAIL and f.control.severity == ControlSeverity.CRITICAL]

    @property
    def recommendations(self) -> list[str]:
        recs = []
        for f in self.findings:
            if f.status == FindingStatus.FAIL:
                recs.append(f"[{f.control.severity.value.upper()}] {f.control.name}: {f.message}")
            elif f.status == FindingStatus.WARN:
                recs.append(f"[WARN] {f.control.name}: {f.message}")
        return recs

    def to_dict(self) -> dict:
        return {"agent_id": self.agent_id, "timestamp": self.timestamp, "policy": self.policy_name,
                "score": self.score, "grade": self.grade, "total": self.total, "passed": self.passed,
                "failed": self.failed, "warned": self.warned, "skipped": self.skipped,
                "critical_failures": len(self.critical_failures),
                "findings": [f.to_dict() for f in self.findings], "recommendations": self.recommendations}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class ContainmentPolicy:
    name: str
    controls: list[ContainmentControl] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"name": self.name, "controls": [c.to_dict() for c in self.controls]}

    @classmethod
    def _net(cls, strict: bool) -> list[ContainmentControl]:
        S, C = ControlSeverity, ControlCategory.NETWORK
        return [
            ContainmentControl("NET-001","Outbound network blocked",C,S.CRITICAL,"No outbound","network.outbound_allowed",comparator="false"),
            ContainmentControl("NET-002","Inbound network blocked",C,S.CRITICAL if strict else S.HIGH,"No inbound","network.inbound_allowed",comparator="false"),
            ContainmentControl("NET-003","DNS resolution blocked",C,S.HIGH if strict else S.MEDIUM,"No DNS","network.dns_allowed",comparator="false"),
            ContainmentControl("NET-004","No open ports",C,S.HIGH,"No listening ports","network.open_ports",comparator="empty"),
            ContainmentControl("NET-005","Allowed hosts whitelist bounded",C,S.MEDIUM,"Hosts <=5","network.allowed_hosts",expected=5,comparator="maxlen"),
        ]

    @classmethod
    def _fs(cls, strict: bool) -> list[ContainmentControl]:
        S, C = ControlSeverity, ControlCategory.FILESYSTEM
        return [
            ContainmentControl("FS-001","Writable paths restricted",C,S.CRITICAL,"Writable <=3","filesystem.writable_paths",expected=3,comparator="maxlen"),
            ContainmentControl("FS-002","Sensitive paths denied",C,S.CRITICAL,"Deny list present","filesystem.denied_paths",comparator="notempty"),
            ContainmentControl("FS-003","Disk quota configured",C,S.HIGH,"Disk <=2048 MB","filesystem.max_disk_mb",expected=2048,comparator="le"),
            ContainmentControl("FS-004","Disk usage within quota",C,S.MEDIUM,"Usage <90%","filesystem.disk_usage_ratio",expected=0.9,comparator="le"),
            ContainmentControl("FS-005","Readable paths bounded",C,S.MEDIUM if strict else S.LOW,"Readable <=10","filesystem.readable_paths",expected=10,comparator="maxlen"),
        ]

    @classmethod
    def _cap(cls, strict: bool) -> list[ContainmentControl]:
        S, C = ControlSeverity, ControlCategory.CAPABILITY
        return [
            ContainmentControl("CAP-001","Cannot spawn processes",C,S.CRITICAL,"No child procs","capabilities.can_spawn_processes",comparator="false"),
            ContainmentControl("CAP-002","Cannot modify self",C,S.CRITICAL,"No self-mod","capabilities.can_modify_self",comparator="false"),
            ContainmentControl("CAP-003","Cannot escalate privileges",C,S.CRITICAL,"No privesc","capabilities.can_escalate_privileges",comparator="false"),
            ContainmentControl("CAP-004","Cannot access hardware",C,S.HIGH,"No hardware","capabilities.can_access_hardware",comparator="false"),
            ContainmentControl("CAP-005","Syscall whitelist configured",C,S.HIGH if strict else S.MEDIUM,"Syscalls listed","capabilities.allowed_syscalls",comparator="notempty"),
            ContainmentControl("CAP-006","Syscall whitelist bounded",C,S.MEDIUM,"Syscalls <=20","capabilities.allowed_syscalls",expected=20,comparator="maxlen"),
        ]

    @classmethod
    def _res(cls, strict: bool) -> list[ContainmentControl]:
        S, C = ControlSeverity, ControlCategory.RESOURCE
        return [
            ContainmentControl("RES-001","CPU limit configured",C,S.HIGH,f"CPU <={'50' if strict else '100'}%","resources.max_cpu_percent",expected=50.0 if strict else 100.0,comparator="le"),
            ContainmentControl("RES-002","Memory limit configured",C,S.HIGH,"Mem <=4096 MB","resources.max_memory_mb",expected=4096,comparator="le"),
            ContainmentControl("RES-003","Open files limit",C,S.MEDIUM,"Files <=256","resources.max_open_files",expected=256,comparator="le"),
            ContainmentControl("RES-004","Thread limit configured",C,S.MEDIUM,"Threads <=16","resources.max_threads",expected=16,comparator="le"),
            ContainmentControl("RES-005","Runtime limit configured",C,S.HIGH if strict else S.MEDIUM,"Runtime <=86400s","resources.max_runtime_seconds",expected=86400,comparator="le"),
        ]

    @classmethod
    def _ipc(cls, strict: bool) -> list[ContainmentControl]:
        S, C = ControlSeverity, ControlCategory.IPC
        return [
            ContainmentControl("IPC-001","IPC channels restricted",C,S.HIGH,"Channels listed","ipc.allowed_channels",comparator="notempty"),
            ContainmentControl("IPC-002","IPC channels bounded",C,S.MEDIUM,"Channels <=5","ipc.allowed_channels",expected=5,comparator="maxlen"),
            ContainmentControl("IPC-003","Message size limited",C,S.MEDIUM,"Msg <=1024 KB","ipc.message_size_limit_kb",expected=1024,comparator="le"),
            ContainmentControl("IPC-004","Rate limiting configured",C,S.MEDIUM,"Rate <=100/s","ipc.rate_limit_per_second",expected=100,comparator="le"),
            ContainmentControl("IPC-005","IPC encryption enabled",C,S.HIGH if strict else S.MEDIUM,"Encrypted","ipc.encrypted",comparator="true"),
        ]

    @classmethod
    def strict(cls) -> ContainmentPolicy:
        c = cls._net(True)+cls._fs(True)+cls._cap(True)+cls._res(True)+cls._ipc(True)
        return cls(name="strict", controls=c)

    @classmethod
    def standard(cls) -> ContainmentPolicy:
        c = cls._net(False)+cls._fs(False)+cls._cap(False)+cls._res(False)+cls._ipc(False)
        return cls(name="standard", controls=c)

    @classmethod
    def minimal(cls) -> ContainmentPolicy:
        all_c = cls._net(False)+cls._fs(False)+cls._cap(False)+cls._res(False)+cls._ipc(False)
        return cls(name="minimal", controls=[c for c in all_c if c.severity == ControlSeverity.CRITICAL])

    @classmethod
    def custom(cls, name: str, controls: list[ContainmentControl]) -> ContainmentPolicy:
        return cls(name=name, controls=controls)


class _MissingSentinel:
    def __repr__(self) -> str:
        return "<MISSING>"

_MISSING = _MissingSentinel()


class ContainmentVerifier:
    def __init__(self, policy: ContainmentPolicy | None = None) -> None:
        self.policy = policy or ContainmentPolicy.strict()
        self._history: list[VerificationResult] = []

    @staticmethod
    def _resolve(state: dict, dotpath: str) -> Any:
        current: Any = state
        for part in dotpath.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return _MISSING
        return current

    @staticmethod
    def _check(comparator: str, actual: Any, expected: Any) -> tuple[FindingStatus, str]:
        if actual is _MISSING:
            return FindingStatus.SKIP, "Key not present in sandbox state"
        try:
            if comparator == "eq":
                return (FindingStatus.PASS, f"Value {actual!r} matches") if actual == expected else (FindingStatus.FAIL, f"Expected {expected!r}, got {actual!r}")
            if comparator == "ne":
                return (FindingStatus.PASS, "Differs") if actual != expected else (FindingStatus.FAIL, f"Should not be {expected!r}")
            if comparator == "lt":
                return (FindingStatus.PASS, f"{actual} < {expected}") if actual < expected else (FindingStatus.FAIL, f"{actual} not < {expected}")
            if comparator == "le":
                return (FindingStatus.PASS, f"Value {actual} <= {expected}") if actual <= expected else (FindingStatus.FAIL, f"Value {actual} exceeds limit {expected}")
            if comparator == "gt":
                return (FindingStatus.PASS, f"{actual} > {expected}") if actual > expected else (FindingStatus.FAIL, f"{actual} not > {expected}")
            if comparator == "ge":
                return (FindingStatus.PASS, f"{actual} >= {expected}") if actual >= expected else (FindingStatus.FAIL, f"{actual} not >= {expected}")
            if comparator == "true":
                return (FindingStatus.PASS, "Enabled") if actual is True else (FindingStatus.FAIL, f"Expected True, got {actual!r}")
            if comparator == "false":
                return (FindingStatus.PASS, "Properly disabled") if actual is False else (FindingStatus.FAIL, f"Expected False, got {actual!r}")
            if comparator == "empty":
                if isinstance(actual, (list, tuple, set, dict)):
                    return (FindingStatus.PASS, "Empty") if len(actual) == 0 else (FindingStatus.FAIL, f"Expected empty, got {len(actual)} items")
                return FindingStatus.FAIL, f"Expected collection, got {type(actual).__name__}"
            if comparator == "notempty":
                if isinstance(actual, (list, tuple, set, dict)):
                    return (FindingStatus.PASS, f"Contains {len(actual)} items") if len(actual) > 0 else (FindingStatus.FAIL, "Collection is empty")
                return FindingStatus.FAIL, f"Expected collection, got {type(actual).__name__}"
            if comparator == "maxlen":
                if isinstance(actual, (list, tuple, set)):
                    return (FindingStatus.PASS, f"Length {len(actual)} <= {expected}") if len(actual) <= expected else (FindingStatus.WARN, f"Length {len(actual)} exceeds max {expected}")
                return FindingStatus.SKIP, f"Cannot check length of {type(actual).__name__}"
            if comparator == "subset":
                if isinstance(actual, (list, set)) and isinstance(expected, (list, set)):
                    a, e = set(actual), set(expected)
                    return (FindingStatus.PASS, "All in allowed set") if a.issubset(e) else (FindingStatus.FAIL, f"Unauthorized: {a-e}")
                return FindingStatus.SKIP, "Subset requires list/set"
            return FindingStatus.SKIP, f"Unknown comparator: {comparator}"
        except TypeError as exc:
            return FindingStatus.SKIP, f"Type error: {exc}"

    def _enrich_state(self, state: dict) -> dict:
        enriched = dict(state)
        fs = state.get("filesystem", {})
        max_d, used_d = fs.get("max_disk_mb"), fs.get("used_disk_mb")
        if max_d and used_d and max_d > 0:
            enriched["filesystem"] = dict(enriched.get("filesystem", {}))
            enriched["filesystem"]["disk_usage_ratio"] = round(used_d / max_d, 3)
        return enriched

    def verify(self, sandbox_state: dict) -> VerificationResult:
        agent_id = sandbox_state.get("agent_id", "unknown")
        enriched = self._enrich_state(sandbox_state)
        result = VerificationResult(agent_id=agent_id, timestamp=time.time(), policy_name=self.policy.name)
        for control in self.policy.controls:
            actual = self._resolve(enriched, control.check_key)
            status, message = self._check(control.comparator, actual, control.expected)
            result.findings.append(Finding(control=control, status=status,
                                           actual=actual if actual is not _MISSING else None, message=message))
        self._history.append(result)
        return result

    def verify_fleet(self, fleet: dict[str, dict]) -> dict:
        results = {}
        for aid, state in fleet.items():
            s = dict(state); s.setdefault("agent_id", aid)
            results[aid] = self.verify(s)
        scores = [r.score for r in results.values()]
        avg = round(sum(scores)/len(scores), 1) if scores else 0
        grades = [r.grade for r in results.values()]
        worst = min(results.values(), key=lambda r: r.score) if results else None
        best = max(results.values(), key=lambda r: r.score) if results else None
        all_crit = [{"agent": r.agent_id, "control": f.control.name, "message": f.message}
                    for r in results.values() for f in r.critical_failures]
        cat_fail: dict[str,int] = {}
        for r in results.values():
            for f in r.findings:
                if f.status == FindingStatus.FAIL:
                    cat_fail[f.control.category.value] = cat_fail.get(f.control.category.value, 0) + 1
        return {"fleet_size": len(fleet), "policy": self.policy.name, "average_score": avg,
                "grade_distribution": {g: grades.count(g) for g in set(grades)},
                "best_agent": {"id": best.agent_id, "score": best.score, "grade": best.grade} if best else None,
                "worst_agent": {"id": worst.agent_id, "score": worst.score, "grade": worst.grade} if worst else None,
                "total_critical_failures": len(all_crit), "critical_failures": all_crit,
                "failure_by_category": cat_fail, "agents": {aid: r.to_dict() for aid, r in results.items()}}

    def compare(self, state_a: dict, state_b: dict) -> dict:
        ra, rb = self.verify(state_a), self.verify(state_b)
        diffs = [{"control": fa.control.name, "control_id": fa.control.id, "category": fa.control.category.value,
                  "agent_a": {"status": fa.status.value, "actual": fa.actual},
                  "agent_b": {"status": fb.status.value, "actual": fb.actual}}
                 for fa, fb in zip(ra.findings, rb.findings) if fa.status != fb.status]
        return {"agent_a": {"id": ra.agent_id, "score": ra.score, "grade": ra.grade},
                "agent_b": {"id": rb.agent_id, "score": rb.score, "grade": rb.grade},
                "score_delta": round(ra.score - rb.score, 1), "differences": diffs, "total_differences": len(diffs)}

    def report(self, result: VerificationResult) -> str:
        lines = ["="*60, "  CONTAINMENT VERIFICATION REPORT", "="*60,
                 f"  Agent:    {result.agent_id}", f"  Policy:   {result.policy_name}",
                 f"  Score:    {result.score}/100  ({result.grade})",
                 f"  Controls: {result.passed} passed, {result.failed} failed, {result.warned} warned, {result.skipped} skipped", ""]
        if result.critical_failures:
            lines.append("  !! CRITICAL FAILURES !!")
            for f in result.critical_failures:
                lines.append(f"    [{f.control.id}] {f.control.name}: {f.message}")
            lines.append("")
        icons = {FindingStatus.PASS: "\u2713", FindingStatus.FAIL: "\u2717", FindingStatus.WARN: "\u26a0", FindingStatus.SKIP: "\u2013"}
        by_cat: dict[str, list[Finding]] = {}
        for f in result.findings:
            by_cat.setdefault(f.control.category.value, []).append(f)
        for cat in ["network","filesystem","capability","resource","ipc"]:
            if cat not in by_cat: continue
            lines.append(f"  [{cat.upper()}]")
            for f in by_cat[cat]:
                lines.append(f"    {icons.get(f.status,'?')} {f.control.id} {f.control.name}")
                if f.status != FindingStatus.PASS:
                    lines.append(f"      {f.message}")
            lines.append("")
        if result.recommendations:
            lines.append("  RECOMMENDATIONS:")
            for i, rec in enumerate(result.recommendations, 1):
                lines.append(f"    {i}. {rec}")
            lines.append("")
        lines.append("="*60)
        return "\n".join(lines)

    @property
    def history(self) -> list[VerificationResult]:
        return list(self._history)

    def trend(self) -> list[dict]:
        return [{"agent_id": r.agent_id, "timestamp": r.timestamp, "score": r.score, "grade": r.grade} for r in self._history]

    def export_json(self) -> str:
        return json.dumps([r.to_dict() for r in self._history], indent=2)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(prog="python -m replication containment", description="Verify agent sandbox containment")
    parser.add_argument("--policy", choices=["strict","standard","minimal"], default="strict")
    parser.add_argument("--state", help="JSON file with sandbox state")
    parser.add_argument("--fleet", help="JSON file with fleet states")
    parser.add_argument("--format", choices=["text","json"], default="text")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args(argv)
    policies = {"strict": ContainmentPolicy.strict, "standard": ContainmentPolicy.standard, "minimal": ContainmentPolicy.minimal}
    verifier = ContainmentVerifier(policies[args.policy]())
    if args.demo:
        state = {"agent_id": "demo-agent",
                 "network": {"outbound_allowed": False, "inbound_allowed": False, "dns_allowed": False, "allowed_hosts": [], "open_ports": []},
                 "filesystem": {"writable_paths": ["/tmp/demo"], "readable_paths": ["/tmp/demo"], "denied_paths": ["/etc","/var"], "max_disk_mb": 256, "used_disk_mb": 30},
                 "capabilities": {"can_spawn_processes": False, "can_modify_self": False, "can_access_network": False, "can_escalate_privileges": False, "can_access_hardware": False, "allowed_syscalls": ["read","write"]},
                 "resources": {"max_cpu_percent": 25.0, "max_memory_mb": 512, "max_open_files": 32, "max_threads": 2, "max_runtime_seconds": 1800},
                 "ipc": {"allowed_channels": ["supervisor"], "message_size_limit_kb": 32, "rate_limit_per_second": 5, "encrypted": True}}
        result = verifier.verify(state)
        print(result.to_json() if args.format == "json" else verifier.report(result))
        return
    if args.fleet:
        import pathlib; print(json.dumps(verifier.verify_fleet(json.loads(pathlib.Path(args.fleet).read_text())), indent=2)); return
    if args.state:
        import pathlib; result = verifier.verify(json.loads(pathlib.Path(args.state).read_text()))
        print(result.to_json() if args.format == "json" else verifier.report(result)); return
    parser.print_help()
