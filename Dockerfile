# ── Build / test stage ──────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install dev deps for testing
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY src/ src/
COPY tests/ tests/

# Run tests during build — fail fast if anything is broken
RUN python -m pytest tests/ -v --tb=short

# ── Runtime stage ──────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="AI Replication Sandbox" \
      org.opencontainers.image.description="Replication-aware worker system with contract enforcement and observability" \
      org.opencontainers.image.source="https://github.com/sauravbhattacharya001/ai" \
      org.opencontainers.image.licenses="MIT"

# Run as non-root for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/sh --create-home appuser

WORKDIR /app

COPY --from=builder /app/src/ src/

# No runtime deps beyond stdlib — the project is pure Python
ENV PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER appuser

# Default: run the test suite (override with your entrypoint)
CMD ["python", "-m", "pytest", "--co", "-q"]
