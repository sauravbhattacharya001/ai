# Installation

## Requirements

- Python 3.10 or higher
- pip

## From PyPI

```bash
pip install ai-replication-sandbox
```

## From Source

```bash
git clone https://github.com/sauravbhattacharya001/ai.git
cd ai
pip install -e ".[dev]"
```

## Docker

```bash
# Build locally
docker build -t ai-replication-sandbox .

# Run tests inside container
docker run --rm ai-replication-sandbox

# Pull pre-built image
docker pull ghcr.io/sauravbhattacharya001/ai:latest
```

## Development Setup

Install with dev dependencies for testing and linting:

```bash
pip install -e ".[dev]"
```

This includes:

| Tool | Purpose |
|------|---------|
| `pytest` | Test runner |
| `flake8` | Linting |
| `mypy` | Static type checking |

## Verify Installation

```python
import replication
print(replication.__version__)  # 1.0.0
```
