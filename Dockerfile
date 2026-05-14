# Argus Ultimate — Multi-stage Docker build
# Push 80 / v8.16.0

# ---------------------------------------------------------------------------
# Stage 1: builder — install Python dependencies
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: runtime — slim image
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

LABEL maintainer="Argus Ultimate"
LABEL version="adaptive-runtime"
LABEL description="Argus Ultimate algorithmic trading system"

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd -r argus && useradd -r -g argus -d /app -s /sbin/nologin argus

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY --chown=argus:argus . .

# Create writable directories
RUN mkdir -p /app/reports /app/logs && chown -R argus:argus /app/reports /app/logs

USER argus

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["python", "scripts/start.py"]
