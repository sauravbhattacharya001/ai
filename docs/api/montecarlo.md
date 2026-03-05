# Monte Carlo Simulation

Runs probabilistic simulations of agent replication scenarios to estimate
risk distributions and identify worst-case outcomes.

## Key Classes

| Class | Description |
|-------|-------------|
| `MetricDistribution` | Statistical distribution of a simulation metric (mean, p50, p95, p99) |
| `RiskMetrics` | Aggregated risk metrics across simulation runs |
| `MonteCarloResult` | Full result of a simulation campaign |
| `MonteCarloComparison` | Side-by-side comparison of two configurations |
| `MonteCarloConfig` | Simulation parameters (iterations, seed, thresholds) |
| `MonteCarloSimulator` | Main simulator — runs N iterations of replication scenarios |

## Usage

```python
from replication.montecarlo import MonteCarloSimulator, MonteCarloConfig

config = MonteCarloConfig(iterations=10000, seed=42)
sim = MonteCarloSimulator(config)

result = sim.simulate(
    contract=my_contract,
    scenario="exponential_growth",
)

print(f"P95 population: {result.population.p95}")
print(f"P99 resource usage: {result.resource_usage.p99}")
print(f"Breach probability: {result.breach_probability:.2%}")

# Compare two configurations
comparison = sim.compare(contract_a, contract_b, scenario="burst")
print(f"Config A is {'safer' if comparison.a_safer else 'riskier'}")
```

::: replication.montecarlo
