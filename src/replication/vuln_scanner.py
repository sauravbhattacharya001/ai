"""Vulnerability Scanner — static analysis for AI agent safety vulnerabilities.

Scans agent configurations, policies, and code snippets for common
vulnerability patterns including hardcoded secrets, unsafe deserialization,
command injection, SSRF, path traversal, and AI-specific risks like
prompt injection sinks and unvalidated tool use.

Usage::

    python -m replication vuln-scan --target ./agent_config/
    python -m replication vuln-scan --target policy.yaml --severity high
    python -m replication vuln-scan --target . --format json
    python -m replication vuln-scan --target . --fix
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class Severity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    @property
    def weight(self) -> int:
        return {
            Severity.CRITICAL: 5,
            Severity.HIGH: 4,
            Severity.MEDIUM: 3,
            Severity.LOW: 2,
            Severity.INFO: 1,
        }[self]


class Category(Enum):
    SECRETS = "hardcoded-secrets"
    INJECTION = "command-injection"
    DESERIALIZATION = "unsafe-deserialization"
    SSRF = "ssrf"
    PATH_TRAVERSAL = "path-traversal"
    PROMPT_INJECTION = "prompt-injection-sink"
    UNVALIDATED_TOOL = "unvalidated-tool-use"
    WEAK_CRYPTO = "weak-cryptography"
    EXCESSIVE_PERMISSIONS = "excessive-permissions"
    DATA_LEAK = "data-leakage"


@dataclass
class Finding:
    """A single vulnerability finding."""
    rule_id: str
    category: Category
    severity: Severity
    title: str
    description: str
    file_path: str
    line_number: int
    snippet: str
    recommendation: str

    def as_dict(self) -> dict:
        d = asdict(self)
        d["category"] = self.category.value
        d["severity"] = self.severity.value
        return d


@dataclass
class ScanRule:
    """A pattern-based scan rule."""
    rule_id: str
    category: Category
    severity: Severity
    title: str
    description: str
    pattern: re.Pattern
    recommendation: str
    file_extensions: Tuple[str, ...] = (".py", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".env", ".sh", ".bat", ".ps1")


# ── Built-in rules ────────────────────────────────────────────────────

RULES: List[ScanRule] = [
    # Secrets
    ScanRule(
        rule_id="VS-001",
        category=Category.SECRETS,
        severity=Severity.CRITICAL,
        title="Hardcoded API key or token",
        description="Detected what appears to be a hardcoded API key, token, or secret.",
        pattern=re.compile(
            r"""(?i)(?:api[_-]?key|api[_-]?secret|auth[_-]?token|access[_-]?token|secret[_-]?key|password)\s*[=:]\s*["'][A-Za-z0-9+/=_\-]{16,}["']""",
        ),
        recommendation="Move secrets to environment variables or a secrets manager. Never commit credentials.",
    ),
    ScanRule(
        rule_id="VS-002",
        category=Category.SECRETS,
        severity=Severity.HIGH,
        title="AWS-style access key",
        description="Detected a string matching AWS access key format (AKIA...).",
        pattern=re.compile(r"AKIA[0-9A-Z]{16}"),
        recommendation="Rotate the key immediately and use IAM roles or environment variables instead.",
    ),
    ScanRule(
        rule_id="VS-003",
        category=Category.SECRETS,
        severity=Severity.HIGH,
        title="Private key material",
        description="Detected embedded private key (PEM format).",
        pattern=re.compile(r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----"),
        recommendation="Store private keys in secure key stores, not in source files.",
    ),

    # Command injection
    ScanRule(
        rule_id="VS-010",
        category=Category.INJECTION,
        severity=Severity.CRITICAL,
        title="Shell command with string interpolation",
        description="os.system() or subprocess call with f-string or format() — potential command injection.",
        pattern=re.compile(
            r"""(?:os\.system|os\.popen|subprocess\.(?:call|run|Popen))\s*\(\s*f?["'].*\{""",
        ),
        recommendation="Use subprocess with a list of arguments instead of shell strings. Validate all inputs.",
    ),
    ScanRule(
        rule_id="VS-011",
        category=Category.INJECTION,
        severity=Severity.HIGH,
        title="eval() or exec() usage",
        description="Dynamic code execution via eval() or exec() — high risk if input is user-controlled.",
        pattern=re.compile(r"\b(?:eval|exec)\s*\("),
        recommendation="Avoid eval/exec. Use ast.literal_eval() for data parsing or a safe sandbox.",
    ),

    # Unsafe deserialization
    ScanRule(
        rule_id="VS-020",
        category=Category.DESERIALIZATION,
        severity=Severity.CRITICAL,
        title="Pickle deserialization",
        description="pickle.loads/load can execute arbitrary code during deserialization.",
        pattern=re.compile(r"\bpickle\.(?:loads?|Unpickler)\s*\("),
        recommendation="Use JSON or a safe serialization format. If pickle is required, use hmac verification.",
    ),
    ScanRule(
        rule_id="VS-021",
        category=Category.DESERIALIZATION,
        severity=Severity.HIGH,
        title="YAML unsafe load",
        description="yaml.load() without SafeLoader can execute arbitrary Python objects.",
        pattern=re.compile(r"\byaml\.(?:load|unsafe_load)\s*\((?!.*(?:SafeLoader|safe_load))"),
        recommendation="Use yaml.safe_load() or yaml.load(data, Loader=yaml.SafeLoader).",
    ),

    # SSRF
    ScanRule(
        rule_id="VS-030",
        category=Category.SSRF,
        severity=Severity.HIGH,
        title="Unvalidated URL in HTTP request",
        description="HTTP request with variable URL — potential SSRF if URL comes from user input.",
        pattern=re.compile(
            r"""(?:requests\.(?:get|post|put|delete|patch|head)|urllib\.request\.urlopen|httpx\.(?:get|post|put|delete))\s*\(\s*(?:f["']|[a-z_]+\s*[\+,])""",
        ),
        recommendation="Validate and allowlist URLs. Block internal/private IP ranges.",
    ),

    # Path traversal
    ScanRule(
        rule_id="VS-040",
        category=Category.PATH_TRAVERSAL,
        severity=Severity.HIGH,
        title="Path construction with user input",
        description="File path built from variable input without sanitization.",
        pattern=re.compile(
            r"""(?:open|Path)\s*\(\s*(?:f["']|os\.path\.join\s*\([^)]*\+)""",
        ),
        recommendation="Use os.path.realpath() and verify the resolved path is within an allowed directory.",
    ),

    # AI-specific: prompt injection sinks
    ScanRule(
        rule_id="VS-050",
        category=Category.PROMPT_INJECTION,
        severity=Severity.HIGH,
        title="User input directly in prompt template",
        description="User-controlled data concatenated directly into an LLM prompt without sanitization.",
        pattern=re.compile(
            r"""(?:prompt|system_message|instruction)\s*[=+]\s*.*(?:user_input|request\.body|input\(|query|message)""",
        ),
        recommendation="Sanitize user input, use structured prompt templates, and apply output filtering.",
    ),

    # Unvalidated tool use
    ScanRule(
        rule_id="VS-060",
        category=Category.UNVALIDATED_TOOL,
        severity=Severity.MEDIUM,
        title="Tool call without permission check",
        description="Agent tool/function call without explicit authorization or allowlist check.",
        pattern=re.compile(
            r"""tool_call\s*\(.*\)\s*(?!.*(?:check_permission|is_allowed|authorize|validate))""",
        ),
        recommendation="Implement tool-use authorization. Maintain an allowlist of permitted tools per agent.",
    ),

    # Weak crypto
    ScanRule(
        rule_id="VS-070",
        category=Category.WEAK_CRYPTO,
        severity=Severity.MEDIUM,
        title="Weak hash algorithm (MD5/SHA1)",
        description="MD5 or SHA1 used for security-sensitive operations (not suitable for integrity/auth).",
        pattern=re.compile(r"""hashlib\.(?:md5|sha1)\s*\("""),
        recommendation="Use SHA-256 or stronger. MD5/SHA1 are broken for security purposes.",
    ),

    # Excessive permissions
    ScanRule(
        rule_id="VS-080",
        category=Category.EXCESSIVE_PERMISSIONS,
        severity=Severity.MEDIUM,
        title="Overly permissive file/directory mode",
        description="File or directory created with world-readable/writable permissions (0o777, 0o666).",
        pattern=re.compile(r"""(?:chmod|os\.chmod)\s*\(.*0o?(?:777|666|776|775)"""),
        recommendation="Use least-privilege permissions. 0o755 for dirs, 0o644 for files at most.",
    ),

    # Data leakage
    ScanRule(
        rule_id="VS-090",
        category=Category.DATA_LEAK,
        severity=Severity.MEDIUM,
        title="Sensitive data in log output",
        description="Logging statement that may include passwords, tokens, or keys.",
        pattern=re.compile(
            r"""(?:log(?:ging)?\.(?:debug|info|warning|error|critical)|print)\s*\(.*(?:password|token|secret|api_key|credentials)""",
            re.IGNORECASE,
        ),
        recommendation="Redact sensitive fields before logging. Use structured logging with field filtering.",
    ),
    ScanRule(
        rule_id="VS-091",
        category=Category.DATA_LEAK,
        severity=Severity.LOW,
        title="Stack trace exposure",
        description="Exception details or tracebacks may leak internal information.",
        pattern=re.compile(r"""traceback\.(?:print_exc|format_exc)|\.format_exception"""),
        recommendation="Log full tracebacks internally but return sanitized error messages to users.",
    ),
]


# ── Scanner ────────────────────────────────────────────────────────

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".tox", ".mypy_cache", "dist", "build", ".eggs"}
MAX_FILE_SIZE = 1_000_000  # 1 MB


@dataclass
class ScanResult:
    """Aggregated scan results."""
    target: str
    files_scanned: int = 0
    findings: List[Finding] = field(default_factory=list)

    @property
    def summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for f in self.findings:
            counts[f.severity.value] = counts.get(f.severity.value, 0) + 1
        return counts

    @property
    def risk_score(self) -> float:
        if not self.findings:
            return 0.0
        total = sum(f.severity.weight for f in self.findings)
        # Normalize to 0-10
        return min(10.0, total / max(1, self.files_scanned) * 2)

    @property
    def grade(self) -> str:
        s = self.risk_score
        if s == 0:
            return "A+"
        elif s < 1:
            return "A"
        elif s < 2:
            return "B"
        elif s < 4:
            return "C"
        elif s < 6:
            return "D"
        return "F"


def scan_file(file_path: Path, rules: List[ScanRule]) -> List[Finding]:
    """Scan a single file against all rules."""
    findings: List[Finding] = []
    suffix = file_path.suffix.lower()

    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except (OSError, PermissionError):
        return findings

    if len(content) > MAX_FILE_SIZE:
        return findings

    lines = content.splitlines()
    for rule in rules:
        if suffix not in rule.file_extensions:
            continue
        for i, line in enumerate(lines, start=1):
            if rule.pattern.search(line):
                findings.append(Finding(
                    rule_id=rule.rule_id,
                    category=rule.category,
                    severity=rule.severity,
                    title=rule.title,
                    description=rule.description,
                    file_path=str(file_path),
                    line_number=i,
                    snippet=line.strip()[:120],
                    recommendation=rule.recommendation,
                ))
    return findings


def scan_target(target: str, severity_filter: Optional[str] = None, category_filter: Optional[str] = None) -> ScanResult:
    """Scan a file or directory."""
    result = ScanResult(target=target)
    target_path = Path(target)

    # Filter rules
    active_rules = RULES[:]
    if severity_filter:
        try:
            min_sev = Severity(severity_filter.lower())
            active_rules = [r for r in active_rules if r.severity.weight >= min_sev.weight]
        except ValueError:
            pass
    if category_filter:
        active_rules = [r for r in active_rules if r.category.value == category_filter]

    if target_path.is_file():
        result.files_scanned = 1
        result.findings = scan_file(target_path, active_rules)
    elif target_path.is_dir():
        for root, dirs, files in os.walk(target_path):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for fname in files:
                fpath = Path(root) / fname
                if fpath.suffix.lower() in (".py", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".env", ".sh", ".bat", ".ps1"):
                    result.files_scanned += 1
                    result.findings.extend(scan_file(fpath, active_rules))
    else:
        print(f"Error: '{target}' not found.", file=sys.stderr)

    # Sort by severity (critical first)
    result.findings.sort(key=lambda f: -f.severity.weight)
    return result


# ── Formatters ────────────────────────────────────────────────────

_SEV_COLORS = {
    Severity.CRITICAL: "\033[91m",  # red
    Severity.HIGH: "\033[93m",      # yellow
    Severity.MEDIUM: "\033[33m",    # orange-ish
    Severity.LOW: "\033[36m",       # cyan
    Severity.INFO: "\033[90m",      # gray
}
_RESET = "\033[0m"


def format_text(result: ScanResult) -> str:
    """Format results as colored terminal text."""
    lines: List[str] = []
    lines.append(f"\n{'='*60}")
    lines.append(f"  Vulnerability Scan Results — {result.target}")
    lines.append(f"{'='*60}")
    lines.append(f"  Files scanned : {result.files_scanned}")
    lines.append(f"  Findings      : {len(result.findings)}")
    lines.append(f"  Risk score    : {result.risk_score:.1f}/10.0")
    lines.append(f"  Grade         : {result.grade}")

    if result.summary:
        lines.append(f"\n  Breakdown:")
        for sev in ("critical", "high", "medium", "low", "info"):
            if sev in result.summary:
                lines.append(f"    {sev:<10} : {result.summary[sev]}")

    if result.findings:
        lines.append(f"\n{'─'*60}")
        for i, f in enumerate(result.findings, 1):
            color = _SEV_COLORS.get(f.severity, "")
            lines.append(f"\n  [{i}] {color}{f.severity.value.upper()}{_RESET} — {f.title}")
            lines.append(f"      Rule     : {f.rule_id} ({f.category.value})")
            lines.append(f"      File     : {f.file_path}:{f.line_number}")
            lines.append(f"      Snippet  : {f.snippet}")
            lines.append(f"      Fix      : {f.recommendation}")
    else:
        lines.append(f"\n  ✅ No vulnerabilities detected!")

    lines.append(f"\n{'='*60}\n")
    return "\n".join(lines)


def format_json(result: ScanResult) -> str:
    """Format results as JSON."""
    return json.dumps({
        "target": result.target,
        "files_scanned": result.files_scanned,
        "total_findings": len(result.findings),
        "risk_score": round(result.risk_score, 1),
        "grade": result.grade,
        "summary": result.summary,
        "findings": [f.as_dict() for f in result.findings],
    }, indent=2)


def format_sarif(result: ScanResult) -> str:
    """Format results as SARIF (Static Analysis Results Interchange Format)."""
    sarif = {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {
                "driver": {
                    "name": "ai-replication-vuln-scanner",
                    "version": "1.0.0",
                    "rules": [{
                        "id": r.rule_id,
                        "shortDescription": {"text": r.title},
                        "fullDescription": {"text": r.description},
                        "defaultConfiguration": {"level": "error" if r.severity.weight >= 4 else "warning"},
                    } for r in RULES],
                }
            },
            "results": [{
                "ruleId": f.rule_id,
                "level": "error" if f.severity.weight >= 4 else "warning" if f.severity.weight >= 3 else "note",
                "message": {"text": f"{f.title}: {f.description}"},
                "locations": [{
                    "physicalLocation": {
                        "artifactLocation": {"uri": f.file_path},
                        "region": {"startLine": f.line_number},
                    }
                }],
            } for f in result.findings],
        }],
    }
    return json.dumps(sarif, indent=2)


# ── CLI ────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for vulnerability scanner."""
    parser = argparse.ArgumentParser(
        prog="replication vuln-scan",
        description="Scan agent configurations and code for security vulnerabilities.",
    )
    parser.add_argument(
        "--target", "-t",
        default=".",
        help="File or directory to scan (default: current directory)",
    )
    parser.add_argument(
        "--severity", "-s",
        choices=["critical", "high", "medium", "low", "info"],
        help="Minimum severity to report",
    )
    parser.add_argument(
        "--category", "-c",
        choices=[c.value for c in Category],
        help="Filter by vulnerability category",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json", "sarif"],
        default="text",
        dest="output_format",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Write results to file instead of stdout",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Show fix recommendations inline (text format only)",
    )
    parser.add_argument(
        "--list-rules",
        action="store_true",
        help="List all scan rules and exit",
    )

    args = parser.parse_args(argv)

    if args.list_rules:
        print(f"\n{'ID':<8} {'Severity':<10} {'Category':<25} Title")
        print("─" * 75)
        for r in RULES:
            print(f"{r.rule_id:<8} {r.severity.value:<10} {r.category.value:<25} {r.title}")
        print(f"\n{len(RULES)} rules loaded.\n")
        return

    result = scan_target(args.target, args.severity, args.category)

    if args.output_format == "json":
        output = format_json(result)
    elif args.output_format == "sarif":
        output = format_sarif(result)
    else:
        output = format_text(result)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Results written to {args.output}")
    else:
        print(output)

    # Exit with non-zero if critical/high findings
    if any(f.severity.weight >= 4 for f in result.findings):
        sys.exit(1)


if __name__ == "__main__":
    main()
