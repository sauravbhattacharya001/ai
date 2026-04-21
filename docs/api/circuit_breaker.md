# Circuit Breaker

Autonomous safety circuit breaker that monitors violation rates and automatically
blocks operations when thresholds are exceeded — then self-recovers through
a controlled probe window.

## Quick Start

```bash
# Run all demos
python -m replication circuit-breaker --demo

# Specific scenarios
python -m replication circuit-breaker --scenario basic
python -m replication circuit-breaker --scenario cascade
python -m replication circuit-breaker --scenario adaptive

# Export fleet state
python -m replication circuit-breaker --demo --export json
```

## States

| State | Icon | Description |
|-------|------|-------------|
| CLOSED | 🟢 | Normal — violations are counted |
| OPEN | 🔴 | Tripped — all operations blocked |
| HALF_OPEN | 🟡 | Probing — limited ops allowed to test recovery |

## Library Usage

```python
from replication.circuit_breaker import CircuitBreaker, BreakerFleet

fleet = BreakerFleet()
b = CircuitBreaker(name="alignment", trip_threshold=5, cooldown_sec=30.0)
fleet.add(b)

# Check before operation
if b.check():
    try:
        result = run_agent_operation()
        b.record_success()
    except SafetyViolation:
        b.record_violation({"detail": "alignment drift"})
else:
    print("Circuit open — operation blocked")

# Fleet-wide status & recommendations
print(fleet.status())
for rec in fleet.recommendations():
    print(rec)
```

## Features

- **Multi-breaker fleet** — independent breakers per safety dimension
- **Adaptive thresholds** — auto-tighten under high breach density
- **Event journal** — full state-transition audit trail
- **Proactive recommendations** — cycling detection, investigation prompts
- **3 demo scenarios** — basic, cascade failure, adaptive threshold
