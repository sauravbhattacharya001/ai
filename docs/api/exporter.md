# Audit Trail Exporter  --  structured data export for simulation analysis

Audit Trail Exporter — structured data export for simulation analysis.


**Module:** `replication.exporter`


## Quick Start

```python
from replication.exporter import ExportConfig

instance = ExportConfig()
```


## Classes

### `ExportConfig`

Configuration for audit trail exports.

### `ExportResult`

Result of an export operation.

| Method | Description |
|--------|-------------|
| `render()` | Human-readable summary of the export. |

### `AuditExporter`

Exports simulation audit trails to structured formats.

| Method | Description |
|--------|-------------|
| `__init__()` |  |
| `workers_csv()` | Export worker lifecycle data as CSV. |
| `workers_jsonl()` | Export worker lifecycle data as JSON Lines. |
| `timeline_csv()` | Export timeline events as CSV. |
| `timeline_jsonl()` | Export timeline events as JSON Lines. |
| `audit_csv()` | Export raw audit events as CSV. |
| `audit_jsonl()` | Export audit events as JSON Lines. |
| `summary_stats()` | Compute comprehensive summary statistics from the report. |
| `summary_csv()` | Export summary statistics as CSV (key-value pairs). |
| `summary_json()` | Export summary statistics as formatted JSON. |
| `export_all()` | Export all requested sections in all requested formats. |
| `comparative_csv()` | Export summary stats from multiple reports as a single CSV table. |


## Functions

| Function | Description |
|----------|-------------|
| `main()` | CLI entry point for audit trail export. |


## CLI

```bash
python -m replication exporter --help
```
