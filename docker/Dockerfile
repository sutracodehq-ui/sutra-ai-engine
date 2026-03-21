# ─── Stage 1: Builder (installs deps into venv) ────────
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps for compiling Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Create isolated venv — only production deps
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml ./
RUN pip install --no-cache-dir . && \
    pip uninstall -y pip setuptools wheel 2>/dev/null; true

# ─── Stage 2: Runtime (minimal) ───────────────────────
FROM python:3.12-slim

WORKDIR /app

# Only the runtime C library for PostgreSQL — no gcc, no dev headers
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 curl && \
    rm -rf /var/lib/apt/lists/* && \
    # Remove unnecessary Python cache and test dirs
    find /usr/local/lib/python3.12 -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true

# Copy venv from builder (nothing else — no gcc, no headers, no pip)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy only application code (respects .dockerignore)
COPY app/ ./app/
COPY agent_config/ ./agent_config/
COPY alembic/ ./alembic/
COPY alembic.ini pyproject.toml ./
COPY docs/ ./docs/

# Non-root user for security
RUN useradd --no-create-home --shell /bin/false sutra && \
    chown -R sutra:sutra /app
USER sutra

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production: no --reload, 2 workers, graceful shutdown
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--timeout-graceful-shutdown", "30"]
