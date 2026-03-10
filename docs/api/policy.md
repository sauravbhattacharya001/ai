# Safety Policy Engine  --  declarative rules for automated safety validation

Safety Policy Engine — declarative rules for automated safety validation.


**Module:** `replication.policy`


## Quick Start

```python
from replication.policy import Operator

instance = Operator()
```


## Enums

### `Operator`

- `LT`
- `LE`
- `GT`
- `GE`
- `EQ`
- `NE`

### `Severity`

- `ERROR`
- `WARNING`
- `INFO`


## Classes

### `Operator`

Comparison operators for policy rules.

| Method | Description |
|--------|-------------|
| `evaluate()` | Return True if the rule passes (actual op threshold). |
| `from_str()` | Parse an operator from a string. |

### `Severity`

Rule severity — determines overall verdict.

### `PolicyRule`

A single safety rule: metric op threshold.

| Method | Description |
|--------|-------------|
| `evaluate()` | Return True if the rule passes. |
| `render_expression()` | Human-readable rule expression. |
| `to_dict()` |  |
| `from_dict()` |  |

### `RuleResult`

Result of evaluating a single rule.

| Method | Description |
|--------|-------------|
| `icon()` |  |
| `status()` |  |
| `render()` |  |
| `to_dict()` |  |

### `PolicyResult`

Aggregate result of evaluating all rules in a policy.

| Method | Description |
|--------|-------------|
| `passed()` | True if no ERROR-severity rules failed and no extraction errors occurred. |
| `has_warnings()` |  |
| `has_extraction_errors()` | True if any rules failed to extract their metric value. |
| `errors()` |  |
| `extraction_errors()` | Rules where metric extraction failed (misconfigured metric name). |
| `warnings()` |  |
| `failures()` |  |
| `passes()` |  |
| `verdict()` |  |
| `verdict_icon()` |  |
| `exit_code()` | Exit code for CI/CD: 0=pass, 1=fail, 2=warn-only. |
| `render()` |  |
| `to_dict()` |  |

### `SafetyPolicy`

A named collection of safety rules that can validate simulation results.

| Method | Description |
|--------|-------------|
| `__init__()` |  |
| `from_preset()` | Load a built-in policy preset. |
| `from_dict()` | Load a policy from a dictionary (parsed JSON/YAML). |
| `from_file()` | Load a policy from a JSON file. |
| `add_rule()` | Add a rule (fluent API). |
| `evaluate()` | Evaluate all rules against a simulation report. |
| `evaluate_with_mc()` | Run a simulation + Monte Carlo analysis, then evaluate all rules. |
| `to_dict()` |  |
| `save()` | Save policy to a JSON file. |


## Functions

| Function | Description |
|----------|-------------|
| `main()` | CLI entry point for the safety policy engine. |


## CLI

```bash
python -m replication policy --help
```
