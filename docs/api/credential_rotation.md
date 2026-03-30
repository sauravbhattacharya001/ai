# Credential Rotation Auditor

Audit agent credentials, API tokens, and secrets to ensure they are rotated on schedule.

## Features

- **Credential inventory** — discover and catalog keys, tokens, and secrets
- **Staleness detection** — flag credentials past their rotation deadline
- **Rotation schedule** — generate an upcoming rotation calendar
- **Hygiene score** — 0–100 compliance score
- **HTML dashboard** — visual rotation status report

## CLI Usage

```bash
# Default audit with 90-day policy
python -m replication credential-audit

# Custom rotation policy (60 days)
python -m replication credential-audit --policy 60

# HTML dashboard
python -m replication credential-audit --html -o creds.html

# Rotation schedule view
python -m replication credential-audit --schedule

# JSON output
python -m replication credential-audit --json
```

## Programmatic Usage

```python
from replication.credential_rotation import (
    CredentialRotationAuditor,
    Credential,
)
import datetime

auditor = CredentialRotationAuditor(rotation_days=90)

# Use sample credentials for testing
creds = auditor.generate_sample_credentials(count=10)

# Or supply your own
creds = [
    Credential(
        name="API_KEY",
        kind="api_key",
        created=datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc),
        last_rotated=datetime.datetime(2025, 6, 1, tzinfo=datetime.timezone.utc),
        owner="gateway",
        scope="admin",
    ),
]

result = auditor.audit(creds)
print(f"Score: {result.score}/100")
print(f"Critical: {result.critical_count}")
result.to_html("dashboard.html")
```

## Scoring

| Status   | Meaning                                  |
|----------|------------------------------------------|
| OK       | Within rotation policy window            |
| Warning  | >80% of policy period elapsed or overdue |
| Critical | >2× the policy period since last rotation |

Score formula: `(ok × 100 + warning × 40) / total`
