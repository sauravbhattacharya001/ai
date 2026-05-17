"""Tests for replication.credential_rotation."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from replication.credential_rotation import (
    AuditFinding,
    AuditResult,
    Credential,
    CredentialRotationAuditor,
    main,
)


UTC = dt.timezone.utc


def _cred(name: str, days_ago: int, created_days_ago: int | None = None,
          kind: str = "api_key", owner: str = "alpha") -> Credential:
    now = dt.datetime.now(tz=UTC)
    created_days_ago = created_days_ago or max(days_ago, 30)
    return Credential(
        name=name,
        kind=kind,
        created=now - dt.timedelta(days=created_days_ago),
        last_rotated=now - dt.timedelta(days=days_ago),
        owner=owner,
    )


# ── Credential ────────────────────────────────────────────────────────


class TestCredential:
    def test_fingerprint_autogen(self):
        c = _cred("X", 10)
        assert len(c.fingerprint) == 16
        # deterministic for same inputs
        c2 = Credential(
            name=c.name,
            kind=c.kind,
            created=c.created,
            last_rotated=c.last_rotated,
        )
        assert c2.fingerprint == c.fingerprint

    def test_explicit_fingerprint_preserved(self):
        c = Credential(
            name="X", kind="token",
            created=dt.datetime.now(tz=UTC),
            last_rotated=dt.datetime.now(tz=UTC),
            fingerprint="manual-fp",
        )
        assert c.fingerprint == "manual-fp"

    def test_age_days_uses_now_when_omitted(self):
        c = _cred("X", 5)
        assert c.age_days() >= 4  # tolerate sub-day rounding

    def test_age_days_with_explicit_now(self):
        now = dt.datetime(2026, 1, 1, tzinfo=UTC)
        c = Credential(
            name="X", kind="token",
            created=now - dt.timedelta(days=200),
            last_rotated=now - dt.timedelta(days=42),
        )
        assert c.age_days(now) == 42

    def test_to_dict_round_trip(self):
        c = _cred("X", 1)
        d = c.to_dict()
        assert d["name"] == "X"
        assert d["fingerprint"] == c.fingerprint
        # iso-format strings re-parsable
        assert dt.datetime.fromisoformat(d["created"]) == c.created


# ── auditor logic ─────────────────────────────────────────────────────


class TestAuditor:
    def test_compliant_credential_is_ok(self):
        auditor = CredentialRotationAuditor(rotation_days=90)
        result = auditor.audit([_cred("X", days_ago=10)])
        assert result.ok_count == 1
        assert result.warning_count == 0
        assert result.critical_count == 0
        assert result.score == 100.0

    def test_due_soon_is_warning(self):
        auditor = CredentialRotationAuditor(rotation_days=90)
        # 80%+ of 90 = 72d
        result = auditor.audit([_cred("X", days_ago=80)])
        assert result.warning_count == 1
        assert "due soon" in result.findings[0].message.lower()

    def test_overdue_is_warning(self):
        auditor = CredentialRotationAuditor(rotation_days=90)
        result = auditor.audit([_cred("X", days_ago=120, created_days_ago=200)])
        assert result.warning_count == 1
        assert "overdue" in result.findings[0].message.lower()

    def test_severely_overdue_is_critical(self):
        auditor = CredentialRotationAuditor(rotation_days=90)
        result = auditor.audit([_cred("X", days_ago=200, created_days_ago=300)])
        assert result.critical_count == 1
        assert "severely" in result.findings[0].message.lower()

    def test_empty_audit_perfect_score(self):
        auditor = CredentialRotationAuditor()
        result = auditor.audit([])
        assert result.score == 100.0
        assert result.findings == []

    def test_score_weighted_mix(self):
        auditor = CredentialRotationAuditor(rotation_days=90)
        creds = [
            _cred("ok1", 10),
            _cred("ok2", 20),
            _cred("warn", 95, created_days_ago=200),
            _cred("crit", 250, created_days_ago=400),
        ]
        result = auditor.audit(creds)
        # ok=2 (100), warn=1 (40), crit=1 (0) -> (200 + 40 + 0)/4 = 60.0
        assert result.score == pytest.approx(60.0)
        assert result.ok_count == 2
        assert result.warning_count == 1
        assert result.critical_count == 1

    def test_rotation_due_in_finding(self):
        auditor = CredentialRotationAuditor(rotation_days=30)
        c = _cred("X", days_ago=5)
        result = auditor.audit([c])
        f = result.findings[0]
        assert f.rotation_due == c.last_rotated + dt.timedelta(days=30)

    def test_sample_credentials_generator(self):
        creds = CredentialRotationAuditor.generate_sample_credentials(count=5)
        assert len(creds) == 5
        assert all(isinstance(c, Credential) for c in creds)
        # fingerprints unique
        assert len({c.fingerprint for c in creds}) == 5

    def test_sample_credentials_capped_by_names(self):
        creds = CredentialRotationAuditor.generate_sample_credentials(count=999)
        # there are 15 names in the source list
        assert len(creds) <= 15


# ── AuditResult outputs ───────────────────────────────────────────────


class TestAuditResultOutputs:
    def _result(self):
        auditor = CredentialRotationAuditor(rotation_days=90)
        return auditor.audit([
            _cred("a", 5),
            _cred("b", 80),
            _cred("c", 200, created_days_ago=300),
        ])

    def test_to_dict_shape(self):
        r = self._result()
        d = r.to_dict()
        assert d["total"] == 3
        assert d["policy_days"] == 90
        assert "findings" in d
        assert len(d["findings"]) == 3

    def test_to_json_is_valid(self):
        r = self._result()
        parsed = json.loads(r.to_json())
        assert parsed["total"] == 3

    def test_schedule_sorted_by_due(self):
        r = self._result()
        sched = r.schedule()
        dues = [item["due"] for item in sched]
        assert dues == sorted(dues)
        assert {item["status"] for item in sched} >= {"ok"}

    def test_to_html_writes_file(self, tmp_path):
        r = self._result()
        target = tmp_path / "audit.html"
        r.to_html(str(target))
        text = target.read_text(encoding="utf-8")
        assert "<html" in text.lower()
        assert "Credential Rotation Audit" in text
        # all credential names should appear
        for name in ("a", "b", "c"):
            assert f">{name}<" in text or f">{name}</" in text or name in text

    def test_to_html_score_color_classes(self, tmp_path):
        # Force a low score by using only critical creds
        auditor = CredentialRotationAuditor(rotation_days=30)
        creds = [_cred(f"c{i}", 200, created_days_ago=300) for i in range(3)]
        r = auditor.audit(creds)
        target = tmp_path / "low.html"
        r.to_html(str(target))
        assert "#e74c3c" in target.read_text(encoding="utf-8")

    def test_timestamp_autogenerated(self):
        r = AuditResult()
        assert r.timestamp != ""
        # parseable as ISO
        dt.datetime.fromisoformat(r.timestamp)


# ── AuditFinding ──────────────────────────────────────────────────────


class TestAuditFinding:
    def test_to_dict_contains_credential_metadata(self):
        c = _cred("X", 10)
        f = AuditFinding(
            credential=c,
            status="ok",
            age_days=10,
            rotation_due=c.last_rotated + dt.timedelta(days=90),
            message="fine",
        )
        d = f.to_dict()
        assert d["credential"] == "X"
        assert d["fingerprint"] == c.fingerprint
        assert d["status"] == "ok"
        assert d["age_days"] == 10


# ── CLI ───────────────────────────────────────────────────────────────


class TestCLI:
    def test_default_summary_prints(self, capsys):
        main(["--count", "5"])
        out = capsys.readouterr().out
        assert "Credential Rotation Audit" in out
        assert "/100" in out

    def test_json_mode(self, capsys):
        main(["--json", "--count", "3"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "findings" in data
        assert data["total"] == 3

    def test_schedule_mode(self, capsys):
        main(["--schedule", "--count", "3"])
        out = capsys.readouterr().out
        assert "Rotation Schedule" in out

    def test_html_mode_writes_file(self, tmp_path, capsys):
        target = tmp_path / "report.html"
        main(["--html", str(target), "--count", "4"])
        assert target.exists()
        assert "html" in target.read_text(encoding="utf-8").lower()
        assert "written" in capsys.readouterr().out.lower()

    def test_custom_policy_changes_findings(self, capsys):
        # 1 day policy => everything is critical/warning
        main(["--policy", "1", "--count", "3", "--json"])
        data = json.loads(capsys.readouterr().out)
        assert data["policy_days"] == 1
        # nothing should be OK under a 1-day policy
        assert data["ok"] == 0
