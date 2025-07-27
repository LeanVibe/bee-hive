# LeanVibe Agent Hive 2.0 - Multi-stage Dockerfile

FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    git \
    tmux \
    postgresql-client \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN groupadd -g 1000 leanvibe && \
    useradd -u 1000 -g leanvibe -s /bin/bash -m leanvibe

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest-cov \
    pytest-xdist \
    debugpy \
    ipython

# Copy source code
COPY --chown=leanvibe:leanvibe . .

# Create necessary directories
RUN mkdir -p /app/workspaces /app/logs /app/temp && \
    chown -R leanvibe:leanvibe /app

# Switch to non-root user
USER leanvibe

# Expose port
EXPOSE 8000

# Development command with hot reload
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production

# Copy source code
COPY --chown=leanvibe:leanvibe . .

# Create necessary directories
RUN mkdir -p /app/workspaces /app/logs /app/temp && \
    chown -R leanvibe:leanvibe /app

# Switch to non-root user
USER leanvibe

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command
CMD ["gunicorn", "src.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
