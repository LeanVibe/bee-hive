# LeanVibe Agent Hive 2.0 - Optimized Multi-stage Dockerfile

# Build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION
ARG PYTHON_VERSION=3.12

# Base stage with common dependencies
FROM python:${PYTHON_VERSION}-slim as base

# Metadata labels
LABEL org.opencontainers.image.title="LeanVibe Agent Hive 2.0" \
      org.opencontainers.image.description="Multi-Agent Orchestration System for Autonomous Software Development" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.vendor="LeanVibe" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/leanvibe/agent-hive"

# Security and performance environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONPATH=/app \
    PATH="/app/.local/bin:$PATH" \
    # Security hardening
    PYTHONHASHSEED=random \
    # Performance optimization
    PYTHONOPTIMIZE=1

# Install system dependencies with security updates
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Essential tools
    curl \
    wget \
    # Build dependencies
    build-essential \
    pkg-config \
    # Database clients
    postgresql-client \
    libpq-dev \
    # Git for version control
    git \
    # tmux for session management
    tmux \
    # Security tools
    ca-certificates \
    gnupg \
    # Process monitoring
    procps \
    # Network utilities
    netcat-openbsd \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* /var/tmp/*

# Create non-root user with security hardening
RUN groupadd -g 1000 -r leanvibe && \
    useradd -u 1000 -r -g leanvibe -s /bin/bash -m leanvibe \
    -c "LeanVibe Application User" && \
    # Create application directories
    mkdir -p /app /app/logs /app/temp /app/workspaces && \
    chown -R leanvibe:leanvibe /app

# Set work directory
WORKDIR /app

# Dependency installation stage
FROM base as dependencies

# Copy requirements files
COPY requirements.txt requirements-agent.txt ./

# Install Python dependencies with security and performance optimizations
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --no-deps \
    -r requirements.txt \
    -r requirements-agent.txt && \
    # Security: Remove pip cache and temporary files
    rm -rf ~/.cache/pip /tmp/* /var/tmp/* && \
    # Performance: Compile Python bytecode
    python -m compileall /usr/local/lib/python*/site-packages/

# Development stage
FROM dependencies as development

# Install development and monitoring dependencies
RUN pip install --no-cache-dir \
    pytest-cov \
    pytest-xdist \
    pytest-benchmark \
    debugpy \
    ipython \
    black \
    ruff \
    mypy \
    bandit && \
    rm -rf ~/.cache/pip /tmp/* /var/tmp/*

# Copy source code with proper ownership
COPY --chown=leanvibe:leanvibe . .

# Compile Python bytecode for faster startup
RUN python -m compileall app/ && \
    # Ensure permissions are correct
    chown -R leanvibe:leanvibe /app

# Switch to non-root user
USER leanvibe

# Expose ports
EXPOSE 8000 5678

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Development command with hot reload and debugging support
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "debug"]

# Production stage
FROM dependencies as production

# Install gunicorn for production
RUN pip install --no-cache-dir gunicorn==21.2.0 && \
    rm -rf ~/.cache/pip /tmp/* /var/tmp/*

# Copy only necessary application files
COPY --chown=leanvibe:leanvibe app/ ./app/
COPY --chown=leanvibe:leanvibe alembic.ini ./
COPY --chown=leanvibe:leanvibe migrations/ ./migrations/

# Create startup script for production initialization
COPY --chown=leanvibe:leanvibe <<'EOF' ./startup.sh
#!/bin/bash
set -e

echo "Starting LeanVibe Agent Hive Production Server..."

# Wait for database to be ready
echo "Waiting for database connection..."
python -c "
import asyncio
import asyncpg
import os
import sys
import time
from urllib.parse import urlparse

async def wait_for_db():
    db_url = os.environ.get('DATABASE_URL', '')
    if not db_url:
        print('No DATABASE_URL provided, skipping database check')
        return
    
    parsed = urlparse(db_url)
    max_attempts = 30
    attempt = 0
    
    while attempt < max_attempts:
        try:
            conn = await asyncpg.connect(
                host=parsed.hostname,
                port=parsed.port or 5432,
                user=parsed.username,
                password=parsed.password,
                database=parsed.path.lstrip('/'),
                timeout=10
            )
            await conn.close()
            print('Database connection successful!')
            return
        except Exception as e:
            attempt += 1
            print(f'Database connection attempt {attempt}/{max_attempts} failed: {e}')
            if attempt < max_attempts:
                time.sleep(2)
    
    print('Failed to connect to database after maximum attempts')
    sys.exit(1)

asyncio.run(wait_for_db())
"

# Run database migrations if enabled
if [ "${AUTO_MIGRATE:-false}" = "true" ]; then
    echo "Running database migrations..."
    python -m alembic upgrade head || {
        echo "Migration failed, but continuing..."
    }
fi

# Start the application
exec "$@"
EOF

RUN chmod +x ./startup.sh

# Compile Python bytecode for optimal performance
RUN python -m compileall app/ && \
    # Set proper permissions
    chown -R leanvibe:leanvibe /app

# Switch to non-root user
USER leanvibe

# Expose port
EXPOSE 8000

# Health check with improved reliability and proper headers
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=5 \
    CMD curl -f -H "User-Agent: Docker-Healthcheck" -H "Accept: application/json" http://localhost:8000/health || exit 1

# Production command with optimizations and graceful shutdown
CMD ["./startup.sh", "gunicorn", \
     "app.main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "4", \
     "--worker-connections", "1000", \
     "--max-requests", "10000", \
     "--max-requests-jitter", "1000", \
     "--preload", \
     "--bind", "0.0.0.0:8000", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info", \
     "--timeout", "120", \
     "--keepalive", "5", \
     "--graceful-timeout", "30"]

# Monitoring stage for observability
FROM production as monitoring

USER root

# Install monitoring tools
RUN pip install --no-cache-dir \
    prometheus-client \
    opentelemetry-instrumentation-fastapi \
    opentelemetry-instrumentation-sqlalchemy \
    opentelemetry-instrumentation-redis \
    opentelemetry-exporter-prometheus \
    structlog && \
    rm -rf ~/.cache/pip /tmp/* /var/tmp/*

USER leanvibe

# Expose metrics port
EXPOSE 8000 9090

# Command with monitoring enabled
CMD ["gunicorn", \
     "app.main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "2", \
     "--bind", "0.0.0.0:8000", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info", \
     "--timeout", "120"]
