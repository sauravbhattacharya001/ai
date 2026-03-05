# Compliance Auditing

Evaluates agent behavior against compliance frameworks (SOC2, NIST, ISO 27001,
GDPR, HIPAA, custom). Produces audit reports with pass/fail/warning verdicts per control.

## Key Classes

| Class | Description |
|-------|-------------|
| `Verdict` | Control outcome: PASS, FAIL, WARNING, NOT_APPLICABLE |
| `Framework` | Compliance framework identifier |
| `Finding` | Individual control evaluation result |
| `FrameworkResult` | Aggregated results for one framework |
| `AuditConfig` | Configuration for audit scope and thresholds |
| `ComplianceAuditor` | Main auditor — evaluates agent traces against framework controls |

## Usage

```python
from replication.compliance import ComplianceAuditor, Framework

auditor = ComplianceAuditor()

report = auditor.audit(
    agent_traces=traces,
    frameworks=[Framework.SOC2, Framework.NIST_800_53],
)

for fw_result in report.framework_results:
    print(f"{fw_result.framework}: {fw_result.pass_rate:.0%} pass rate")
    for finding in fw_result.findings:
        if finding.verdict == "FAIL":
            print(f"  FAIL: {finding.control_id} — {finding.description}")
```

::: replication.compliance
