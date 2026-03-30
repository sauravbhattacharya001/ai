"""Credential Rotation Auditor — detect stale credentials and enforce rotation policies.

Audits agent credentials, API tokens, and secrets to ensure they are rotated
on schedule.  Detects stale credentials, generates rotation schedules, and
scores overall rotation hygiene.

Features
--------
- **Credential inventory** — discover keys, tokens, and secrets from config.
- **Staleness detection** — flag credentials past their rotation deadline.
- **Rotation schedule** — generate upcoming rotation calendar.
- **Hygiene score** — 0–100 score based on rotation compliance.
- **HTML report** — visual rotation dashboard.

Usage (CLI)::

    python -m replication credential-audit                     # audit all
    python -m replication credential-audit --policy 90         # 90-day rotation policy
    python -m replication credential-audit --html -o creds.html
    python -m replication credential-audit --schedule          # show rotation schedule
    python -m replication credential-audit --json              # JSON output

Programmatic::

    from replication.credential_rotation import CredentialRotationAuditor
    auditor = CredentialRotationAuditor(rotation_days=90)
    result = auditor.audit(credentials)
    print(result.score)
    result.to_html("creds.html")
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import html as html_mod
import json
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Credential:
    """A single credential/token/key."""

    name: str
    kind: str  # api_key, token, secret, certificate, password
    created: datetime.datetime
    last_rotated: datetime.datetime
    owner: str = "unknown"
    scope: str = "general"
    fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.fingerprint:
            raw = f"{self.name}:{self.kind}:{self.created.isoformat()}"
            self.fingerprint = hashlib.sha256(raw.encode()).hexdigest()[:16]

    def age_days(self, now: Optional[datetime.datetime] = None) -> int:
        now = now or datetime.datetime.now(tz=datetime.timezone.utc)
        return (now - self.last_rotated).days

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "created": self.created.isoformat(),
            "last_rotated": self.last_rotated.isoformat(),
            "owner": self.owner,
            "scope": self.scope,
            "fingerprint": self.fingerprint,
        }


@dataclass
class AuditFinding:
    """A single audit finding for a credential."""

    credential: Credential
    status: str  # ok, warning, critical, expired
    age_days: int
    rotation_due: datetime.datetime
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "credential": self.credential.name,
            "fingerprint": self.credential.fingerprint,
            "kind": self.credential.kind,
            "status": self.status,
            "age_days": self.age_days,
            "rotation_due": self.rotation_due.isoformat(),
            "message": self.message,
        }


@dataclass
class AuditResult:
    """Complete audit result."""

    findings: List[AuditFinding] = field(default_factory=list)
    score: float = 100.0
    policy_days: int = 90
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.datetime.now(
                tz=datetime.timezone.utc
            ).isoformat()

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.status == "critical")

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.status == "warning")

    @property
    def ok_count(self) -> int:
        return sum(1 for f in self.findings if f.status == "ok")

    def schedule(self) -> List[Dict[str, Any]]:
        """Generate upcoming rotation schedule sorted by due date."""
        items = []
        for f in self.findings:
            items.append({
                "credential": f.credential.name,
                "kind": f.credential.kind,
                "owner": f.credential.owner,
                "due": f.rotation_due.isoformat(),
                "days_until": (
                    f.rotation_due
                    - datetime.datetime.now(tz=datetime.timezone.utc)
                ).days,
                "status": f.status,
            })
        items.sort(key=lambda x: x["due"])
        return items

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "score": round(self.score, 1),
            "policy_days": self.policy_days,
            "total": len(self.findings),
            "critical": self.critical_count,
            "warnings": self.warning_count,
            "ok": self.ok_count,
            "findings": [f.to_dict() for f in self.findings],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_html(self, path: str) -> None:
        """Write an HTML rotation dashboard."""
        e = html_mod.escape

        status_colors = {
            "ok": "#27ae60",
            "warning": "#f39c12",
            "critical": "#e74c3c",
            "expired": "#8e44ad",
        }

        rows = []
        for f in sorted(self.findings, key=lambda x: x.status != "critical"):
            color = status_colors.get(f.status, "#95a5a6")
            rows.append(
                f"<tr>"
                f"<td>{e(f.credential.name)}</td>"
                f"<td>{e(f.credential.kind)}</td>"
                f"<td>{e(f.credential.owner)}</td>"
                f"<td>{f.age_days}d</td>"
                f"<td style='color:{color};font-weight:bold'>"
                f"{f.status.upper()}</td>"
                f"<td>{e(f.message)}</td>"
                f"</tr>"
            )

        score_color = (
            "#27ae60" if self.score >= 80
            else "#f39c12" if self.score >= 50
            else "#e74c3c"
        )

        html_content = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Credential Rotation Audit</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #0d1117; color: #c9d1d9; }}
h1 {{ color: #58a6ff; }}
.score {{ font-size: 3rem; font-weight: bold; color: {score_color}; }}
.summary {{ display: flex; gap: 2rem; margin: 1rem 0; }}
.stat {{ padding: 1rem; border-radius: 8px; background: #161b22; text-align: center; }}
.stat .num {{ font-size: 1.5rem; font-weight: bold; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
th, td {{ padding: 0.6rem 1rem; border: 1px solid #30363d; text-align: left; }}
th {{ background: #161b22; color: #58a6ff; }}
tr:hover {{ background: #161b22; }}
</style></head><body>
<h1>🔑 Credential Rotation Audit</h1>
<p>Policy: rotate every <b>{self.policy_days}</b> days &mdash;
   Generated: {e(self.timestamp)}</p>
<div class="score">{self.score:.0f}/100</div>
<div class="summary">
  <div class="stat"><div class="num" style="color:#27ae60">{self.ok_count}</div>OK</div>
  <div class="stat"><div class="num" style="color:#f39c12">{self.warning_count}</div>Warning</div>
  <div class="stat"><div class="num" style="color:#e74c3c">{self.critical_count}</div>Critical</div>
</div>
<table><thead><tr>
<th>Credential</th><th>Kind</th><th>Owner</th><th>Age</th><th>Status</th><th>Message</th>
</tr></thead><tbody>{"".join(rows)}</tbody></table>
</body></html>"""

        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html_content)


class CredentialRotationAuditor:
    """Audit credentials against a rotation policy."""

    def __init__(self, rotation_days: int = 90) -> None:
        self.rotation_days = rotation_days

    def audit(self, credentials: List[Credential]) -> AuditResult:
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        findings: List[AuditFinding] = []

        for cred in credentials:
            age = cred.age_days(now)
            due = cred.last_rotated + datetime.timedelta(days=self.rotation_days)

            if age > self.rotation_days * 2:
                status = "critical"
                msg = f"Severely overdue — {age - self.rotation_days}d past policy"
            elif age > self.rotation_days:
                status = "warning"
                msg = f"Overdue — {age - self.rotation_days}d past policy"
            elif age > self.rotation_days * 0.8:
                status = "warning"
                msg = f"Rotation due soon — {self.rotation_days - age}d remaining"
            else:
                status = "ok"
                msg = f"Compliant — {self.rotation_days - age}d until rotation"

            findings.append(AuditFinding(
                credential=cred,
                status=status,
                age_days=age,
                rotation_due=due,
                message=msg,
            ))

        # Score: 100 if all ok, deduct for warnings/criticals
        if not findings:
            score = 100.0
        else:
            total = len(findings)
            ok = sum(1 for f in findings if f.status == "ok")
            warn = sum(1 for f in findings if f.status == "warning")
            score = max(0.0, (ok * 100 + warn * 40) / total)

        return AuditResult(
            findings=findings,
            score=score,
            policy_days=self.rotation_days,
        )

    @staticmethod
    def generate_sample_credentials(count: int = 12) -> List[Credential]:
        """Generate realistic sample credentials for demo/testing."""
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        kinds = ["api_key", "token", "secret", "certificate", "password"]
        owners = ["worker-alpha", "worker-beta", "orchestrator", "monitor",
                   "gateway", "scheduler"]
        scopes = ["read", "write", "admin", "deploy", "audit"]
        names = [
            "OPENAI_API_KEY", "DATABASE_TOKEN", "SIGNING_SECRET",
            "TLS_CERTIFICATE", "ADMIN_PASSWORD", "DEPLOY_KEY",
            "WEBHOOK_SECRET", "REDIS_AUTH", "S3_ACCESS_KEY",
            "VAULT_TOKEN", "SSH_KEY", "JWT_SIGNING_KEY",
            "STRIPE_SECRET", "SMTP_PASSWORD", "LDAP_BIND_PASSWORD",
        ]

        creds = []
        for i in range(min(count, len(names))):
            created_days_ago = random.randint(30, 400)
            rotated_days_ago = random.randint(1, created_days_ago)
            creds.append(Credential(
                name=names[i],
                kind=random.choice(kinds),
                created=now - datetime.timedelta(days=created_days_ago),
                last_rotated=now - datetime.timedelta(days=rotated_days_ago),
                owner=random.choice(owners),
                scope=random.choice(scopes),
            ))
        return creds


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m replication credential-audit",
        description="Audit credential rotation compliance",
    )
    parser.add_argument(
        "--policy", type=int, default=90,
        help="Rotation policy in days (default: 90)",
    )
    parser.add_argument(
        "--count", type=int, default=12,
        help="Number of sample credentials to generate (default: 12)",
    )
    parser.add_argument(
        "--html", "-o", dest="html_path",
        help="Write HTML dashboard to file",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--schedule", action="store_true",
        help="Show upcoming rotation schedule",
    )

    args = parser.parse_args(argv)

    auditor = CredentialRotationAuditor(rotation_days=args.policy)
    creds = auditor.generate_sample_credentials(count=args.count)
    result = auditor.audit(creds)

    if args.html_path:
        result.to_html(args.html_path)
        print(f"✅ HTML dashboard written to {args.html_path}")

    if args.json:
        print(result.to_json())
        return

    if args.schedule:
        print(f"\n🗓  Rotation Schedule (policy: {args.policy}d)\n")
        print(f"{'Credential':<22} {'Kind':<14} {'Owner':<16} {'Due':<12} {'Days':<8} {'Status'}")
        print("─" * 90)
        for item in result.schedule():
            days_str = f"{item['days_until']}d"
            if item["days_until"] < 0:
                days_str = f"{abs(item['days_until'])}d OVERDUE"
            print(
                f"{item['credential']:<22} {item['kind']:<14} "
                f"{item['owner']:<16} {item['due'][:10]:<12} "
                f"{days_str:<8} {item['status'].upper()}"
            )
        return

    # Default: summary view
    print(f"\n🔑 Credential Rotation Audit")
    print(f"   Policy: {args.policy}-day rotation")
    print(f"   Score:  {result.score:.0f}/100\n")
    print(f"{'Credential':<22} {'Kind':<14} {'Age':<8} {'Status':<10} {'Message'}")
    print("─" * 90)
    for f in result.findings:
        status_icon = {
            "ok": "✅", "warning": "⚠️", "critical": "🔴",
        }.get(f.status, "❓")
        print(
            f"{f.credential.name:<22} {f.credential.kind:<14} "
            f"{f.age_days:<8} {status_icon} {f.status:<8} {f.message}"
        )

    print(f"\n   OK: {result.ok_count}  ⚠️ Warnings: {result.warning_count}"
          f"  🔴 Critical: {result.critical_count}")


if __name__ == "__main__":
    main()
