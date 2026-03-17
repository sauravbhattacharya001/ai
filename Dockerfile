# ── Build / test stage ──────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build deps
COPY requirements-dev.txt pyproject.toml LICENSE README.md ./
COPY src/ src/
COPY tests/ tests/

RUN pip install --no-cache-dir -r requirements-dev.txt \
    && pip install --no-cache-dir .

# Run tests during build — fail fast if anything is broken
RUN python -m pytest tests/ -v --tb=short

# Build wheel for runtime stage
RUN pip wheel --no-deps --wheel-dir /wheels .

# ── Runtime stage ──────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="AI Replication Sandbox" \
      org.opencontainers.image.description="Contract-enforced sandbox for studying AI agent self-replication safety" \
      org.opencontainers.image.source="https://github.com/sauravbhattacharya001/ai" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.vendor="sauravbhattacharya001"

# Run as non-root for security
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/sh --create-home appuser

WORKDIR /app

# Install the built wheel — gets proper entrypoint + metadata
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl \
    && rm -rf /wheels

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import replication; print('ok')"]

USER appuser

ENTRYPOINT ["replication"]
