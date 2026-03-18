# Preflight Check

Pre-simulation validation for sandbox configurations. Produces a go/no-go assessment before running experiments.

## Overview

The preflight checker validates contract constraints, resource limits, policy consistency, and stop condition coverage. It catches misconfigurations before they cause runtime failures.

## CLI Usage

```bash
# Default configuration check
python -m replication preflight

# Check specific strategy
python -m replication preflight --strategy greedy

# Override contract parameters
python -m replication preflight --max-depth 5 --max-replicas 20

# Resource constraints
python -m replication preflight --cpu 0.5 --memory 256

# Validate against policy preset
python -m replication preflight --policy strict

# JSON output for CI integration
python -m replication preflight --json

# Suggest fixes for detected issues
python -m replication preflight --fix

# Check all strategies at once
python -m replication preflight --all-strategies
```

## API Usage

```python
from replication.preflight import PreflightChecker, PreflightConfig

config = PreflightConfig(
    max_depth=3,
    max_replicas=10,
    cpu_limit=0.5,
    memory_limit_mb=256,
)

checker = PreflightChecker(config)
result = checker.run()

# Human-readable report
print(result.render())

# Programmatic checks
assert result.passed, f"Preflight failed: {result.failures}"

# Inspect individual findings
for finding in result.findings:
    print(f"[{finding.severity}] {finding.category}: {finding.message}")
```

## Checks Performed

| Check | Description |
|-------|-------------|
| Contract bounds | Validates `max_depth`, `max_replicas`, `cooldown_seconds` are sensible |
| Resource limits | Ensures CPU and memory limits won't cause immediate OOM |
| Policy consistency | Detects conflicting stop conditions or unreachable states |
| Stop condition coverage | Warns if no stop conditions are defined |
| Strategy compatibility | Validates chosen strategy works with the contract |

## CI Integration

Use `--json` output with a non-zero exit code on failure for CI pipelines:

```yaml
- name: Preflight check
  run: python -m replication preflight --json --policy strict
```
