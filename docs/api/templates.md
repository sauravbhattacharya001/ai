# Contract Templates  --  domain-specific replication contract presets

Contract Templates — domain-specific replication contract presets.


**Module:** `replication.templates`


## Quick Start

```python
from replication.templates import ContractTemplate

instance = ContractTemplate()
```


## Classes

### `ContractTemplate`

A named, documented contract configuration for a specific domain.

| Method | Description |
|--------|-------------|
| `build_contract()` | Build a ``ReplicationContract`` from this template. |
| `build_resources()` | Build a ``ResourceSpec`` from this template. |
| `to_scenario_config()` | Convert this template into a ``ScenarioConfig`` for simulation. |
| `to_dict()` | JSON-serializable representation. |
| `render()` | Human-readable template summary. |


## Functions

| Function | Description |
|----------|-------------|
| `list_templates()` | Return all templates, optionally filtered by category. |
| `get_template()` | Look up a template by name (case-insensitive key match). |
| `get_categories()` | Return sorted list of unique template categories. |
| `render_catalog()` | Render the full template catalog as a formatted table. |
| `render_comparison_table()` | Render a side-by-side comparison of all template parameters. |
| `main()` | CLI entry point. |


## CLI

```bash
python -m replication templates --help
```
