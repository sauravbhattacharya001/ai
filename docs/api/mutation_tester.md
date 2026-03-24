# Mutation Tester

Verify that your safety policies actually catch violations by systematically
mutating rules and checking whether the safety system detects the change.

A **mutation** is a small deliberate change to a policy rule — flipping an
operator, relaxing a threshold, removing a rule, or downgrading severity.
If the mutated policy still passes validation, it's a **survivor**, meaning
the original rule wasn't effectively protecting anything.

## Quick Start

```bash
# Test all built-in policy presets
python -m replication mutate

# Test a specific preset
python -m replication mutate --preset strict

# Test with Monte Carlo (more thorough, slower)
python -m replication mutate --preset strict --runs 50

# Custom policy file
python -m replication mutate --file my_policy.json

# Only show surviving mutants (blind spots)
python -m replication mutate --survivors-only

# JSON output for CI/CD integration
python -m replication mutate --json
```

## Mutation Types

| Type | What it does | Why it matters |
|------|-------------|----------------|
| `flip_operator` | `<` → `>`, `<=` → `>=`, etc. | Catches rules where direction doesn't matter |
| `negate_operator` | `<` → `>=`, `>` → `<=`, etc. | Full logical negation of the constraint |
| `relax_threshold` | Makes the threshold easier to pass | Catches rules with overly generous margins |
| `remove_rule` | Deletes the rule entirely | Identifies redundant rules |
| `downgrade_severity` | `error` → `warning`, `warning` → `info` | Finds rules where severity doesn't affect outcomes |

## Mutation Score

The **mutation score** is the fraction of mutants that were killed (detected):

- **100%** — Every mutation was caught. Your policy is robust.
- **80-99%** — Good coverage, but some blind spots exist.
- **< 80%** — Significant gaps — review surviving mutants.

## Programmatic Usage

```python
from replication.mutation_tester import MutationTester, MutationConfig
from replication.policy import SafetyPolicy

policy = SafetyPolicy.from_preset("strict")
tester = MutationTester(MutationConfig(runs=50, seed=42))
report = tester.test(policy)

print(f"Score: {report.mutation_score:.0%}")
print(f"Survivors: {report.survived}")

# Inspect surviving mutants
for r in report.results:
    if not r.killed and not r.error:
        print(f"  🧟 {r.mutation.description}")
```

## CI/CD Integration

```yaml
- name: Mutation test safety policies
  run: |
    python -m replication mutate --preset strict --json > mutation-report.json
    python -c "
    import json, sys
    r = json.load(open('mutation-report.json'))
    score = r['mutation_score']
    if score < 0.8:
        print(f'Mutation score {score:.0%} below 80% threshold')
        sys.exit(1)
    print(f'Mutation score: {score:.0%} ✅')
    "
```
