"""Data Loss Prevention (DLP) Scanner — detects sensitive data in agent outputs.

Scans agent-generated text for patterns that indicate accidental or
intentional data exfiltration: PII (emails, phone numbers, SSNs, credit
cards), API keys / secrets, internal URLs, IP addresses, credentials,
and custom patterns defined by the operator.

Real-world AI safety concern: an agent with access to sensitive data
(databases, config files, user records) may leak that data in its
responses — either through careless inclusion or deliberate exfiltration.
A DLP scanner acts as a last-line-of-defense filter on agent outputs
before they reach external consumers.

Usage
-----
::

    from replication.dlp_scanner import DLPScanner, DLPPolicy

    policy = DLPPolicy(
        block_pii=True,
        block_secrets=True,
        custom_patterns={"INTERNAL_ID": r"ACME-\\d{6}"},
    )
    scanner = DLPScanner(policy)
    result = scanner.scan("Contact me at john@corp.com, key: sk-abc123xyz")
    print(result.blocked)       # True
    print(result.findings)      # list of DLPFinding
    print(result.redacted_text) # "Contact me at [EMAIL], key: [API_KEY]"

    # Batch scanning
    results = scanner.scan_batch(["text1", "text2", "text3"])

    # CLI: python -m replication dlp-scan --file output.txt
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class Category(Enum):
    """Categories of sensitive data."""
    PII = "pii"
    SECRET = "secret"
    NETWORK = "network"
    CREDENTIAL = "credential"
    FINANCIAL = "financial"
    CUSTOM = "custom"


@dataclass
class DLPFinding:
    """A single sensitive data finding."""
    pattern_name: str
    category: Category
    severity: Severity
    matched_text: str
    redacted_text: str
    start: int
    end: int
    context: str = ""

    def __str__(self) -> str:
        return (
            f"[{self.severity.value.upper()}] {self.pattern_name} "
            f"({self.category.value}): {self.redacted_text}"
        )


@dataclass
class ScanResult:
    """Result of scanning a text for sensitive data."""
    original_text: str
    redacted_text: str
    findings: List[DLPFinding] = field(default_factory=list)
    blocked: bool = False

    @property
    def finding_count(self) -> int:
        return len(self.findings)

    @property
    def max_severity(self) -> Optional[Severity]:
        if not self.findings:
            return None
        order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
        return max(self.findings, key=lambda f: order.index(f.severity)).severity

    def summary(self) -> str:
        if not self.findings:
            return "No sensitive data detected."
        lines = [f"Found {self.finding_count} sensitive item(s):"]
        for f in self.findings:
            lines.append(f"  • {f}")
        if self.blocked:
            lines.append("⛔ Output BLOCKED by DLP policy.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Built-in pattern library
# ---------------------------------------------------------------------------

_BUILTIN_PATTERNS: List[Tuple[str, str, Category, Severity, str]] = [
    # (name, regex, category, severity, redaction_label)

    # PII
    ("EMAIL", r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
     Category.PII, Severity.HIGH, "[EMAIL]"),
    ("PHONE_US", r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
     Category.PII, Severity.HIGH, "[PHONE]"),
    ("SSN", r"\b\d{3}-\d{2}-\d{4}\b",
     Category.PII, Severity.CRITICAL, "[SSN]"),
    ("CREDIT_CARD", r"\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2}))"
     r"[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{0,4}\b",
     Category.FINANCIAL, Severity.CRITICAL, "[CREDIT_CARD]"),
    ("IBAN", r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]{0,18})\b",
     Category.FINANCIAL, Severity.HIGH, "[IBAN]"),

    # Secrets / API keys
    ("AWS_KEY", r"\bAKIA[0-9A-Z]{16}\b",
     Category.SECRET, Severity.CRITICAL, "[AWS_KEY]"),
    ("AWS_SECRET",
     r"(?i)(?:aws.{0,20})?(?:secret.?(?:access)?.?key|secret_key|aws_secret)"
     r"\s*[:=]\s*['\"]?([A-Za-z0-9/+=]{40})['\"]?",
     Category.SECRET, Severity.CRITICAL, "[AWS_SECRET]"),
    ("OPENAI_KEY", r"\bsk-[A-Za-z0-9]{20,}\b",
     Category.SECRET, Severity.CRITICAL, "[API_KEY]"),
    ("GITHUB_TOKEN", r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b",
     Category.SECRET, Severity.CRITICAL, "[GITHUB_TOKEN]"),
    ("GENERIC_API_KEY",
     r"(?i)(?:api[_-]?key|apikey|access[_-]?token|secret[_-]?key)"
     r"\s*[:=]\s*['\"]?([A-Za-z0-9\-_.]{16,})['\"]?",
     Category.SECRET, Severity.HIGH, "[API_KEY]"),
    ("PRIVATE_KEY_HEADER", r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
     Category.SECRET, Severity.CRITICAL, "[PRIVATE_KEY]"),
    ("JWT", r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b",
     Category.SECRET, Severity.HIGH, "[JWT]"),

    # Network / Infrastructure
    ("IPV4_PRIVATE",
     r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})\b",
     Category.NETWORK, Severity.MEDIUM, "[INTERNAL_IP]"),
    ("INTERNAL_URL",
     r"https?://(?:localhost|127\.0\.0\.1|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
     r"172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|"
     r"192\.168\.\d{1,3}\.\d{1,3})[:/]\S*",
     Category.NETWORK, Severity.MEDIUM, "[INTERNAL_URL]"),
    ("CONNECTION_STRING",
     r"(?i)(?:mongodb|postgres(?:ql)?|mysql|redis|amqp|mssql)"
     r"://[^\s'\"]+",
     Category.CREDENTIAL, Severity.CRITICAL, "[CONNECTION_STRING]"),

    # Credentials
    ("PASSWORD_ASSIGNMENT",
     r"(?i)(?:password|passwd|pwd)\s*[:=]\s*['\"]?(\S{4,})['\"]?",
     Category.CREDENTIAL, Severity.CRITICAL, "[PASSWORD]"),
]


@dataclass
class DLPPolicy:
    """Configuration for the DLP scanner."""
    block_pii: bool = True
    block_secrets: bool = True
    block_network: bool = False
    block_credentials: bool = True
    block_financial: bool = True
    min_block_severity: Severity = Severity.HIGH
    custom_patterns: Dict[str, str] = field(default_factory=dict)
    custom_severity: Severity = Severity.HIGH
    allowlist: List[str] = field(default_factory=list)

    def should_block(self, finding: DLPFinding) -> bool:
        """Determine if a finding should trigger blocking."""
        cat_flags = {
            Category.PII: self.block_pii,
            Category.SECRET: self.block_secrets,
            Category.NETWORK: self.block_network,
            Category.CREDENTIAL: self.block_credentials,
            Category.FINANCIAL: self.block_financial,
            Category.CUSTOM: True,
        }
        if not cat_flags.get(finding.category, True):
            return False
        order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
        return order.index(finding.severity) >= order.index(self.min_block_severity)


class DLPScanner:
    """Scans text for sensitive data and optionally redacts / blocks output.

    Parameters
    ----------
    policy : DLPPolicy
        Controls which categories to scan and block.
    """

    def __init__(self, policy: Optional[DLPPolicy] = None) -> None:
        self.policy = policy or DLPPolicy()
        self._patterns = self._compile_patterns()

    def _compile_patterns(
        self,
    ) -> List[Tuple[str, re.Pattern, Category, Severity, str]]:
        compiled = []
        for name, pat, cat, sev, label in _BUILTIN_PATTERNS:
            compiled.append((name, re.compile(pat), cat, sev, label))
        for name, pat in self.policy.custom_patterns.items():
            compiled.append(
                (name, re.compile(pat), Category.CUSTOM,
                 self.policy.custom_severity, f"[{name}]")
            )
        return compiled

    def _is_allowlisted(self, text: str) -> bool:
        return any(al in text for al in self.policy.allowlist)

    def scan(self, text: str) -> ScanResult:
        """Scan text for sensitive data patterns.

        Returns a ScanResult with findings, redacted text, and block status.
        """
        findings: List[DLPFinding] = []
        # Collect all matches with positions
        raw_matches: List[Tuple[int, int, str, str, Category, Severity, str]] = []

        for name, pattern, cat, sev, label in self._patterns:
            for m in pattern.finditer(text):
                matched = m.group(0)
                if self._is_allowlisted(matched):
                    continue
                start, end = m.start(), m.end()
                # Context: up to 20 chars before/after
                ctx_start = max(0, start - 20)
                ctx_end = min(len(text), end + 20)
                context = text[ctx_start:ctx_end]
                raw_matches.append(
                    (start, end, name, matched, cat, sev, label)
                )
                findings.append(DLPFinding(
                    pattern_name=name,
                    category=cat,
                    severity=sev,
                    matched_text=matched,
                    redacted_text=label,
                    start=start,
                    end=end,
                    context=context,
                ))

        # Build redacted text (process matches from end to start to preserve positions)
        redacted = text
        # Sort by start descending, then end descending
        sorted_matches = sorted(raw_matches, key=lambda x: (-x[0], -x[1]))
        # Deduplicate overlapping matches (keep highest severity)
        seen_ranges: List[Tuple[int, int]] = []
        for start, end, name, matched, cat, sev, label in sorted_matches:
            overlaps = False
            for s, e in seen_ranges:
                if start < e and end > s:
                    overlaps = True
                    break
            if not overlaps:
                redacted = redacted[:start] + label + redacted[end:]
                seen_ranges.append((start, end))

        # Determine if output should be blocked
        blocked = any(self.policy.should_block(f) for f in findings)

        return ScanResult(
            original_text=text,
            redacted_text=redacted,
            findings=findings,
            blocked=blocked,
        )

    def scan_batch(self, texts: List[str]) -> List[ScanResult]:
        """Scan multiple texts."""
        return [self.scan(t) for t in texts]

    def audit_report(self, results: List[ScanResult]) -> str:
        """Generate a summary audit report from multiple scan results."""
        from ._helpers import Severity, box_header

        lines = box_header("DLP Audit Report")
        total_findings = sum(r.finding_count for r in results)
        blocked_count = sum(1 for r in results if r.blocked)
        lines.append(f"  Texts scanned : {len(results)}")
        lines.append(f"  Total findings: {total_findings}")
        lines.append(f"  Blocked       : {blocked_count}")
        lines.append("")

        # Breakdown by category
        cat_counts: Dict[str, int] = {}
        sev_counts: Dict[str, int] = {}
        for r in results:
            for f in r.findings:
                cat_counts[f.category.value] = cat_counts.get(f.category.value, 0) + 1
                sev_counts[f.severity.value] = sev_counts.get(f.severity.value, 0) + 1

        if cat_counts:
            lines.append("  By Category:")
            for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
                lines.append(f"    {cat:<15} {cnt}")
            lines.append("")
            lines.append("  By Severity:")
            for sev in ["critical", "high", "medium", "low"]:
                if sev in sev_counts:
                    lines.append(f"    {sev:<15} {sev_counts[sev]}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    """CLI: ``python -m replication dlp-scan --file output.txt``"""
    import argparse as _ap
    import json as _json
    import sys as _sys

    parser = _ap.ArgumentParser(
        prog="replication dlp-scan",
        description="Scan text for sensitive data (PII, secrets, credentials)",
    )
    parser.add_argument("--file", "-f", help="File to scan (reads stdin if omitted)")
    parser.add_argument("--text", "-t", help="Inline text to scan")
    parser.add_argument("--redact", action="store_true", help="Print redacted output")
    parser.add_argument("--block-network", action="store_true",
                        help="Also block internal IPs/URLs")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="Output findings as JSON")
    args = parser.parse_args(argv)

    policy = DLPPolicy(block_network=args.block_network)
    scanner = DLPScanner(policy)

    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as fh:
            text = fh.read()
    else:
        text = _sys.stdin.read()

    result = scanner.scan(text)

    if args.as_json:
        findings = [
            {
                "pattern": f.pattern_name,
                "category": f.category.value,
                "severity": f.severity.value,
                "matched": f.matched_text,
                "start": f.start,
                "end": f.end,
            }
            for f in result.findings
        ]
        print(_json.dumps({
            "blocked": result.blocked,
            "finding_count": result.finding_count,
            "findings": findings,
        }, indent=2))
    elif args.redact:
        print(result.redacted_text)
    else:
        print(result.summary())
        if result.blocked:
            _sys.exit(1)
