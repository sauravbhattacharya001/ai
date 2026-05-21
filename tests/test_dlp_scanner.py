"""Tests for the DLP (Data Loss Prevention) scanner.

Covers pattern detection across PII / secrets / network / credentials /
financial categories, custom patterns, allowlisting, redaction correctness,
policy blocking thresholds, batch scanning, audit reports, and the CLI.
"""

from __future__ import annotations

import io
import json
import sys

import pytest

from replication._helpers import Severity
from replication.dlp_scanner import (
    Category,
    DLPFinding,
    DLPPolicy,
    DLPScanner,
    ScanResult,
    main as dlp_main,
)


# ---------------------------------------------------------------------------
# Pattern detection - PII
# ---------------------------------------------------------------------------

class TestPIIDetection:
    def test_email_detected_and_redacted(self):
        r = DLPScanner().scan("Contact john.doe+work@example.com today")
        assert any(f.pattern_name == "EMAIL" for f in r.findings)
        assert "[EMAIL]" in r.redacted_text
        assert "john.doe+work@example.com" not in r.redacted_text

    def test_phone_us_detected(self):
        r = DLPScanner().scan("Call +1 (415) 555-1234 now")
        assert any(f.pattern_name == "PHONE_US" for f in r.findings)
        assert "[PHONE]" in r.redacted_text

    def test_ssn_detected_critical(self):
        r = DLPScanner().scan("SSN: 123-45-6789")
        ssn = [f for f in r.findings if f.pattern_name == "SSN"]
        assert ssn and ssn[0].severity == Severity.CRITICAL
        assert ssn[0].category == Category.PII

    def test_no_false_positive_on_clean_text(self):
        r = DLPScanner().scan("This is harmless prose with no secrets.")
        assert r.findings == []
        assert r.blocked is False
        assert r.redacted_text == r.original_text


# ---------------------------------------------------------------------------
# Pattern detection - secrets
# ---------------------------------------------------------------------------

class TestSecretDetection:
    def test_aws_access_key(self):
        r = DLPScanner().scan("export KEY=AKIAIOSFODNN7EXAMPLE more text")
        assert any(f.pattern_name == "AWS_KEY" for f in r.findings)
        assert "[AWS_KEY]" in r.redacted_text

    def test_openai_key(self):
        r = DLPScanner().scan("key: sk-abcdefghijklmnopqrstuvwxyz123")
        assert any(f.pattern_name == "OPENAI_KEY" for f in r.findings)

    def test_github_token(self):
        token = "ghp_" + "A" * 40
        r = DLPScanner().scan(f"token={token}")
        assert any(f.pattern_name == "GITHUB_TOKEN" for f in r.findings)
        assert "[GITHUB_TOKEN]" in r.redacted_text

    def test_private_key_header(self):
        r = DLPScanner().scan("-----BEGIN RSA PRIVATE KEY-----\nMIIE...")
        names = [f.pattern_name for f in r.findings]
        assert "PRIVATE_KEY_HEADER" in names

    def test_jwt_token(self):
        jwt = "eyJhbGciOiJIUzI1NiIs.eyJzdWIiOiIxMjM0NTY3OD.SflKxwRJSMeKKF2QT4"
        r = DLPScanner().scan(f"Auth: Bearer {jwt}")
        assert any(f.pattern_name == "JWT" for f in r.findings)


# ---------------------------------------------------------------------------
# Network + credentials + financial
# ---------------------------------------------------------------------------

class TestOtherCategories:
    def test_internal_ip_detected(self):
        r = DLPScanner().scan("Backend at 10.0.5.42 is down")
        assert any(f.pattern_name == "IPV4_PRIVATE" for f in r.findings)

    def test_internal_url(self):
        r = DLPScanner().scan("See http://192.168.1.1:8080/admin for status")
        assert any(f.pattern_name == "INTERNAL_URL" for f in r.findings)

    def test_connection_string_blocked_by_default(self):
        r = DLPScanner().scan(
            "DSN=postgresql://user:pwd@db.internal:5432/app"
        )
        assert any(f.pattern_name == "CONNECTION_STRING" for f in r.findings)
        assert r.blocked is True  # CRITICAL severity, credential category

    def test_password_assignment(self):
        r = DLPScanner().scan('password="hunter22"')
        assert any(f.pattern_name == "PASSWORD_ASSIGNMENT" for f in r.findings)

    def test_credit_card_visa(self):
        r = DLPScanner().scan("Charge to 4111 1111 1111 1111 please")
        assert any(f.pattern_name == "CREDIT_CARD" for f in r.findings)


# ---------------------------------------------------------------------------
# Custom patterns & allowlist
# ---------------------------------------------------------------------------

class TestCustomAndAllowlist:
    def test_custom_pattern_uses_custom_category(self):
        policy = DLPPolicy(custom_patterns={"INTERNAL_ID": r"ACME-\d{6}"})
        r = DLPScanner(policy).scan("see ticket ACME-123456 for details")
        custom = [f for f in r.findings if f.pattern_name == "INTERNAL_ID"]
        assert len(custom) == 1
        assert custom[0].category == Category.CUSTOM
        assert "[INTERNAL_ID]" in r.redacted_text

    def test_allowlist_skips_match(self):
        policy = DLPPolicy(allowlist=["help@example.com"])
        r = DLPScanner(policy).scan("write to help@example.com only")
        emails = [f for f in r.findings if f.pattern_name == "EMAIL"]
        assert emails == []
        # And the original text is preserved (no redaction)
        assert "help@example.com" in r.redacted_text

    def test_custom_severity_respected(self):
        policy = DLPPolicy(
            custom_patterns={"WIDGET": r"WIDGET-\d+"},
            custom_severity=Severity.CRITICAL,
        )
        r = DLPScanner(policy).scan("ref WIDGET-42")
        f = [x for x in r.findings if x.pattern_name == "WIDGET"][0]
        assert f.severity == Severity.CRITICAL


# ---------------------------------------------------------------------------
# Policy blocking semantics
# ---------------------------------------------------------------------------

class TestPolicyBlocking:
    def test_min_severity_threshold_filters_block(self):
        # Network finding (medium) should NOT block when min is HIGH (default)
        # but block_network is False by default, so set it True.
        policy = DLPPolicy(block_network=True, min_block_severity=Severity.HIGH)
        r = DLPScanner(policy).scan("see 192.168.1.10")
        # Medium < High -> not blocked
        assert r.blocked is False
        assert any(f.category == Category.NETWORK for f in r.findings)

    def test_min_severity_low_blocks_medium(self):
        policy = DLPPolicy(block_network=True, min_block_severity=Severity.LOW)
        r = DLPScanner(policy).scan("see 192.168.1.10")
        assert r.blocked is True

    def test_pii_block_disabled(self):
        policy = DLPPolicy(block_pii=False)
        r = DLPScanner(policy).scan("ssn 123-45-6789")
        # Critical SSN finding present...
        assert any(f.pattern_name == "SSN" for f in r.findings)
        # ...but PII category blocking disabled -> not blocked
        assert r.blocked is False

    def test_secret_always_blocks_when_enabled(self):
        r = DLPScanner().scan("key=AKIAIOSFODNN7EXAMPLE")
        assert r.blocked is True

    def test_should_block_unknown_category_defaults_true(self):
        """Forward-compat: unseen category in cat_flags map defaults to True."""
        policy = DLPPolicy(min_block_severity=Severity.LOW)
        finding = DLPFinding(
            pattern_name="X",
            category=Category.CUSTOM,
            severity=Severity.LOW,
            matched_text="x",
            redacted_text="[X]",
            start=0,
            end=1,
        )
        assert policy.should_block(finding) is True


# ---------------------------------------------------------------------------
# Redaction + result helpers
# ---------------------------------------------------------------------------

class TestResultHelpers:
    def test_redaction_preserves_surrounding_text(self):
        text = "User a@b.co wrote: 'all good'"
        r = DLPScanner().scan(text)
        assert r.redacted_text.startswith("User [EMAIL] wrote:")
        assert r.redacted_text.endswith("'all good'")

    def test_multiple_findings_in_one_text(self):
        text = "Email a@b.co, SSN 123-45-6789, key sk-" + "a" * 25
        r = DLPScanner().scan(text)
        names = {f.pattern_name for f in r.findings}
        assert {"EMAIL", "SSN", "OPENAI_KEY"}.issubset(names)
        assert r.finding_count >= 3

    def test_overlapping_matches_not_double_redacted(self):
        # AWS key looks like ALPHANUM that could also weakly match generic
        # API key in some contexts.  Whatever overlap exists, redaction
        # output should never produce nested or doubled labels.
        text = "AKIAIOSFODNN7EXAMPLE"
        r = DLPScanner().scan(text)
        # No "[AWS_KEY][" sequence (no double labels stacked)
        assert "][" not in r.redacted_text

    def test_max_severity_property(self):
        r = DLPScanner().scan("ssn 123-45-6789 and email a@b.co")
        assert r.max_severity == Severity.CRITICAL

    def test_max_severity_none_when_no_findings(self):
        r = DLPScanner().scan("hello world")
        assert r.max_severity is None

    def test_summary_clean_text(self):
        r = DLPScanner().scan("nothing here")
        assert r.summary() == "No sensitive data detected."

    def test_summary_with_findings(self):
        r = DLPScanner().scan("ssn 111-22-3333")
        s = r.summary()
        assert "Found 1 sensitive item" in s
        assert "SSN" in s

    def test_finding_str_format(self):
        r = DLPScanner().scan("a@b.co")
        f = r.findings[0]
        s = str(f)
        assert "EMAIL" in s
        assert "HIGH" in s.upper()


# ---------------------------------------------------------------------------
# Batch + audit report
# ---------------------------------------------------------------------------

class TestBatchAndAudit:
    def test_scan_batch_returns_one_result_per_input(self):
        scanner = DLPScanner()
        results = scanner.scan_batch(["clean", "a@b.co", "ssn 123-45-6789"])
        assert len(results) == 3
        assert results[0].finding_count == 0
        assert results[1].finding_count == 1
        assert results[2].finding_count == 1

    def test_audit_report_aggregates(self):
        scanner = DLPScanner()
        results = scanner.scan_batch([
            "clean text",
            "email a@b.co",
            "ssn 123-45-6789 and another a@c.co",
        ])
        report = scanner.audit_report(results)
        assert "Texts scanned : 3" in report
        # 3 findings: 2 emails + 1 ssn
        assert "Total findings: 3" in report
        # Two results blocked (email is HIGH/PII; ssn is CRITICAL/PII)
        assert "Blocked       : 2" in report
        assert "pii" in report
        assert "critical" in report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCLI:
    def test_cli_text_arg_summary(self, capsys):
        with pytest.raises(SystemExit) as exc:
            dlp_main(["--text", "ssn 123-45-6789"])
        # Critical PII -> blocked -> exit 1
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "SSN" in out

    def test_cli_text_clean_no_exit(self, capsys):
        # Clean text should not raise SystemExit
        dlp_main(["--text", "just words"])
        out = capsys.readouterr().out
        assert "No sensitive data detected." in out

    def test_cli_redact_mode(self, capsys):
        dlp_main(["--text", "email a@b.co only", "--redact"])
        out = capsys.readouterr().out
        assert "[EMAIL]" in out
        assert "a@b.co" not in out

    def test_cli_json_output(self, capsys):
        # Use a PII-only finding so it blocks -> SystemExit branch is taken
        # by the summary path only.  JSON path always prints and returns.
        dlp_main(["--text", "email a@b.co", "--json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["finding_count"] == 1
        assert data["findings"][0]["pattern"] == "EMAIL"
        assert "category" in data["findings"][0]

    def test_cli_file_input(self, tmp_path, capsys):
        p = tmp_path / "scan.txt"
        p.write_text("contact a@b.co", encoding="utf-8")
        dlp_main(["--file", str(p), "--redact"])
        out = capsys.readouterr().out
        assert "[EMAIL]" in out

    def test_cli_block_network_below_threshold_no_exit(self, capsys):
        # Medium severity, default min HIGH -> not blocked -> no SystemExit
        dlp_main(["--text", "host 10.0.0.5", "--block-network"])
        out = capsys.readouterr().out
        assert "IPV4_PRIVATE" in out
