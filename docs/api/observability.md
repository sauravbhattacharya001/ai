# Observability

Structured logging, metrics, and audit trail.

The `observability` module provides in-memory structured logging for the
replication system.  It captures events, metrics, and security audit
decisions as structured records rather than text, making them easy to
filter, query, and export.

## Constants

| Name                 | Value    | Description                              |
|----------------------|----------|------------------------------------------|
| `DEFAULT_MAX_ENTRIES`| 100,000  | Default capacity for event/metric deques |

## Classes

### Metric

A single metric data point.

| Field       | Type                     | Description                      |
|-------------|--------------------------|----------------------------------|
| `name`      | str                      | Metric identifier                |
| `value`     | Any                      | Numeric or categorical value     |
| `timestamp` | datetime                 | UTC collection time              |
| `labels`    | Optional[Dict[str, str]] | Key-value tags for filtering     |

### StructuredLogger

In-memory structured logging with bounded retention and audit support.

**Constructor:**

```python
StructuredLogger(max_events: int = 100_000, max_metrics: int = 100_000)
```

Both parameters accept `None` for unbounded storage (not recommended
for long-running simulations).

**Attributes:**

| Attribute         | Type             | Description                          |
|-------------------|------------------|--------------------------------------|
| `events`          | deque[dict]      | Structured event records             |
| `metrics`         | deque[Metric]    | Collected metric data points         |
| `dropped_events`  | int              | Count of evicted events (overflow)   |
| `dropped_metrics` | int              | Count of evicted metrics (overflow)  |

**Methods:**

| Method                        | Description                                       |
|-------------------------------|---------------------------------------------------|
| `log(event, **fields)`        | Append a structured event with UTC timestamp       |
| `emit_metric(name, value, **labels)` | Record a metric data point                 |
| `audit(decision, **fields)`   | Log a security-relevant decision (prefixed audit)  |

When the deque reaches capacity, the oldest entry is silently dropped
and the `dropped_*` counter is incremented.

## Usage

```python
from replication.observability import StructuredLogger

logger = StructuredLogger(max_events=1000)

# Log an event
logger.log("worker_started", worker_id="abc", depth=0)

# Record a metric
logger.emit_metric("cpu_usage", 0.75, worker_id="abc")

# Audit trail
logger.audit("deny_quota", reason="max_replicas", count=10)

# Query events
recent = [e for e in logger.events if e["event"] == "audit"]
```

::: replication.observability
