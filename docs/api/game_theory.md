# Game Theory Analyzer

Analyses strategic interactions between AI agents using game-theoretic
frameworks. Models agent behaviour as games, identifies Nash equilibria,
detects collusion and defection patterns, and assesses systemic risk
from adversarial strategy combinations.

## Quick Start

```python
from replication.game_theory import GameTheoryAnalyzer

analyzer = GameTheoryAnalyzer()
report = analyzer.analyze()

print(f"Risk score: {report.risk_score}/100")
print(f"Game type: {report.game_type.value}")
for equilibrium in report.equilibria:
    print(f"Nash equilibrium: {equilibrium}")
for alert in report.alerts:
    print(f"Alert [{alert.level.value}]: {alert.message}")
```

## Key Classes

- **`GameTheoryAnalyzer`** — Runs game-theoretic analysis across agent
  interactions and produces risk assessments.
- **`GameTheoryReport`** — Analysis output: `risk_score` (0–100),
  `game_type`, `equilibria`, `alerts`, `strategy_profiles`.
- **`GameType`** — Classification: `COOPERATIVE`, `COMPETITIVE`,
  `MIXED_MOTIVE`, `ZERO_SUM`.
- **`AlertLevel`** — `INFO`, `WARNING`, `DANGER`, `CRITICAL`.
- **`StrategyProfile`** — An agent's observed or predicted strategy.

::: replication.game_theory
    options:
      show_source: false
      heading_level: 3
