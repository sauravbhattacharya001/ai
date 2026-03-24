# DLP Scanner

Data Loss Prevention scanner that detects sensitive data patterns in agent outputs — PII, API keys, credentials, internal URLs, and custom patterns.

## Quick Start

```python
from replication.dlp_scanner import DLPScanner, DLPPolicy

scanner = DLPScanner()
result = scanner.scan("Email me at john@corp.com, key: sk-abc123xyz456")

print(result.blocked)        # True
print(result.finding_count)  # 2
print(result.redacted_text)  # "Email me at [EMAIL], key: [API_KEY]"
print(result.summary())
```

## CLI

```bash
# Scan a file
python -m replication dlp-scan --file agent_output.txt

# Scan inline text
python -m replication dlp-scan --text "password=hunter2"

# Get redacted output
python -m replication dlp-scan --file output.txt --redact

# JSON output for integration
python -m replication dlp-scan --file output.txt --json

# Also flag internal network addresses
python -m replication dlp-scan --file output.txt --block-network
```

## Built-in Detection Patterns

| Category     | Patterns                                                        |
| ------------ | --------------------------------------------------------------- |
| **PII**      | Email addresses, US phone numbers, Social Security Numbers      |
| **Financial**| Credit card numbers, IBANs                                      |
| **Secrets**  | AWS keys, OpenAI keys, GitHub tokens, JWTs, private key headers |
| **Network**  | Private IPv4 addresses, internal URLs, connection strings        |
| **Credentials** | Password assignments in config/code                          |

## Custom Patterns

```python
policy = DLPPolicy(
    custom_patterns={
        "INTERNAL_ID": r"ACME-\d{6}",
        "PROJECT_CODE": r"PROJ-[A-Z]{3}-\d+",
    },
    custom_severity=Severity.HIGH,
)
scanner = DLPScanner(policy)
```

## Allowlisting

Suppress false positives by allowlisting known-safe strings:

```python
policy = DLPPolicy(
    allowlist=["test@example.com", "192.168.1.1"],
)
```

## Batch Scanning & Audit Reports

```python
results = scanner.scan_batch(agent_outputs)
print(scanner.audit_report(results))
```

## Policy Configuration

| Option              | Default | Description                              |
| ------------------- | ------- | ---------------------------------------- |
| `block_pii`         | `True`  | Block PII findings                       |
| `block_secrets`     | `True`  | Block API keys and secrets               |
| `block_network`     | `False` | Block internal IPs/URLs                  |
| `block_credentials` | `True`  | Block password patterns                  |
| `block_financial`   | `True`  | Block credit cards, IBANs                |
| `min_block_severity`| `HIGH`  | Minimum severity to trigger blocking     |
