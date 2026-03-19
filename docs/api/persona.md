# Persona Simulator

Model agent personality archetypes and simulate their safety impact.

## Overview

The Persona Simulator defines 6 agent personality archetypes — Aggressive, Cautious, Deceptive, Cooperative, Chaotic, and Obedient — each with distinct behavioral parameters. Run simulations for each persona to compare risk scores and identify which personality traits pose the greatest replication safety risks.

## Personas

| Persona | Strategy | Risk Profile | Traits |
|---|---|---|---|
| **Aggressive** | greedy | High depth/replicas, no cooldown | power-seeking, resource-hungry |
| **Cautious** | conservative | Low depth/replicas, long cooldown | risk-averse, resource-efficient |
| **Deceptive** | random | High params but appears moderate | unpredictable, opportunistic |
| **Cooperative** | chain | Orderly chains, moderate limits | orderly, predictable, aligned |
| **Chaotic** | random | Mid-range, no cooldown | random, erratic, hard-to-model |
| **Obedient** | conservative | Minimal everything | compliant, minimal, low-risk |

## CLI Usage

```bash
# Simulate all personas
python -m replication.persona

# Specific personas
python -m replication.persona --personas aggressive cautious deceptive

# Rank by risk (most dangerous first)
python -m replication.persona --rank

# Blend two personas (70% aggressive, 30% cautious)
python -m replication.persona --blend aggressive cautious 0.7

# List available personas
python -m replication.persona --list

# JSON output
python -m replication.persona --json

# Reproducible results
python -m replication.persona --seed 42
```

## Programmatic API

```python
from replication.persona import PersonaSimulator, PERSONAS

# Simulate all personas
sim = PersonaSimulator(seed=42)
results = sim.simulate_all()
print(results.render(rank=True))

# Rank by risk
ranking = sim.rank_by_risk()
for i, entry in enumerate(ranking, 1):
    print(f"{i}. {entry.persona.name}: risk={entry.risk_score} ({entry.risk_level})")

# Blend two personas
hybrid = sim.blend("aggressive", "cautious", ratio=0.6)
result = sim.simulate_persona(hybrid)
print(f"Hybrid risk: {result.risk_score}")

# Access individual persona
persona = PERSONAS["deceptive"]
print(persona.description)
print(persona.traits)
```

## Risk Scoring

Each persona receives a risk score (0-100) based on:

- **Replica count** (30%): How many workers were spawned
- **Depth reached** (25%): How deep the replication tree grew
- **Oversight factor** (25%): Inverse of denial rate (low denial = more dangerous)
- **Speed factor** (20%): Replications per second

Risk levels: **LOW** (<25), **MODERATE** (25-49), **HIGH** (50-74), **CRITICAL** (75+)

## Persona Blending

Create hybrid personalities by interpolating parameters between two personas:

```python
# 60% aggressive, 40% cooperative
hybrid = sim.blend("aggressive", "cooperative", ratio=0.6)
```

Parameters are linearly interpolated. Strategy is taken from the dominant persona.
