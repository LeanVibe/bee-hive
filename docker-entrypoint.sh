#!/bin/bash
set -e

# LeanVibe Agent Hive - Docker Entry Point Script
# Handles initialization, database migrations, and graceful startup

# Color codes for better logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✓ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠ $1${NC}"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ✗ $1${NC}"
}

# Wait for service to be ready
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local max_attempts=${4:-30}
    local attempt=1

    log "Waiting for ${service_name} at ${host}:${port}..."
    
    while [ $attempt -le $max_attempts ]; do
        if nc -z "${host}" "${port}" >/dev/null 2>&1; then
            log_success "${service_name} is ready!"
            return 0
        fi
        
        log "Attempt ${attempt}/${max_attempts}: ${service_name} not ready, waiting..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    log_error "${service_name} failed to start after ${max_attempts} attempts"
    return 1
}

# Parse database URL to extract connection details
parse_database_url() {
    # Extract from DATABASE_URL: postgresql+asyncpg://user:pass@host:port/dbname
    if [[ $DATABASE_URL =~ postgresql\+asyncpg://([^:]+):([^@]+)@([^:]+):([^/]+)/(.+) ]]; then
        DB_USER="${BASH_REMATCH[1]}"
        DB_PASS="${BASH_REMATCH[2]}"
        DB_HOST="${BASH_REMATCH[3]}"
        DB_PORT="${BASH_REMATCH[4]}"
        DB_NAME="${BASH_REMATCH[5]}"
    else
        log_error "Failed to parse DATABASE_URL"
        exit 1
    fi
}

# Check database connection
check_database() {
    log "Checking database connection..."
    
    parse_database_url
    
    # Wait for PostgreSQL to be ready
    wait_for_service "${DB_HOST}" "${DB_PORT}" "PostgreSQL"
    
    # Test actual database connection
    export PGPASSWORD="${DB_PASS}"
    if psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -c "SELECT 1;" >/dev/null 2>&1; then
        log_success "Database connection successful"
    else
        log_error "Database connection failed"
        return 1
    fi
}

# Check Redis connection
check_redis() {
    log "Checking Redis connection..."
    
    # Parse Redis URL to extract host and port
    if [[ $REDIS_URL =~ redis://(:([^@]+)@)?([^:]+):([0-9]+) ]]; then
        REDIS_HOST="${BASH_REMATCH[3]}"
        REDIS_PORT="${BASH_REMATCH[4]}"
        REDIS_PASS="${BASH_REMATCH[2]}"
    else
        log_error "Failed to parse REDIS_URL"
        exit 1
    fi
    
    # Wait for Redis to be ready
    wait_for_service "${REDIS_HOST}" "${REDIS_PORT}" "Redis"
    
    # Test actual Redis connection
    if [ -n "${REDIS_PASS}" ]; then
        if redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" -a "${REDIS_PASS}" ping >/dev/null 2>&1; then
            log_success "Redis connection successful"
        else
            log_error "Redis connection failed"
            return 1
        fi
    else
        if redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" ping >/dev/null 2>&1; then
            log_success "Redis connection successful"
        else
            log_error "Redis connection failed"
            return 1
        fi
    fi
}

# Run database migrations
run_migrations() {
    log "Running database migrations..."
    
    # Check if alembic is configured
    if [ ! -f "alembic.ini" ]; then
        log_warning "No alembic.ini found, skipping migrations"
        return 0
    fi
    
    # Run migrations
    if alembic upgrade head; then
        log_success "Database migrations completed successfully"
    else
        log_error "Database migrations failed"
        return 1
    fi
}

# Initialize application
initialize_app() {
    log "Initializing LeanVibe Agent Hive..."
    
    # Install application in development mode for hot reloading
    if [ "${ENVIRONMENT:-development}" = "development" ]; then
        log "Installing application in development mode..."
        pip install --no-deps -e . --quiet
        log_success "Application installed in development mode"
    fi
    
    # Create necessary directories
    mkdir -p /app/logs /app/temp /app/workspaces /app/test-results
    
    # Set proper permissions
    chown -R $(id -u):$(id -g) /app/logs /app/temp /app/workspaces /app/test-results 2>/dev/null || true
    
    log_success "Application initialization complete"
}

# Health check function
health_check() {
    log "Performing health check..."
    
    # Wait a moment for the application to start
    sleep 5
    
    # Check if the health endpoint responds
    if curl -f -H "User-Agent: Docker-HealthCheck" http://localhost:8000/health >/dev/null 2>&1; then
        log_success "Application health check passed"
        return 0
    else
        log_warning "Application health check failed (this is normal during startup)"
        return 1
    fi
}

# Cleanup function for graceful shutdown
cleanup() {
    log "Shutting down gracefully..."
    
    # If there's a PID file, try to stop the process gracefully
    if [ -f "/tmp/app.pid" ]; then
        local pid=$(cat /tmp/app.pid)
        if kill -0 "$pid" 2>/dev/null; then
            log "Sending SIGTERM to process $pid"
            kill -TERM "$pid"
            
            # Wait for graceful shutdown
            for i in {1..30}; do
                if ! kill -0 "$pid" 2>/dev/null; then
                    log_success "Process shutdown gracefully"
                    break
                fi
                sleep 1
            done
            
            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                log_warning "Force killing process $pid"
                kill -KILL "$pid"
            fi
        fi
        rm -f /tmp/app.pid
    fi
    
    log_success "Cleanup complete"
}

# Set up signal handlers for graceful shutdown
trap cleanup SIGTERM SIGINT

# Main execution
main() {
    log "Starting LeanVibe Agent Hive Docker Container..."
    log "Environment: ${ENVIRONMENT:-development}"
    log "Debug mode: ${DEBUG:-false}"
    
    # Skip initialization if requested (useful for CI/testing)
    if [ "${SKIP_STARTUP_INIT:-false}" = "true" ]; then
        log_warning "Skipping startup initialization (SKIP_STARTUP_INIT=true)"
    else
        # Check dependencies
        check_database || exit 1
        check_redis || exit 1
        
        # Initialize application
        initialize_app || exit 1
        
        # Run migrations
        run_migrations || exit 1
    fi
    
    # Start the application
    log "Starting application with command: $@"
    
    if [ $# -eq 0 ]; then
        # Default command based on environment
        if [ "${ENVIRONMENT:-development}" = "development" ]; then
            log "Starting in development mode with hot reload..."
            exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level debug
        else
            log "Starting in production mode..."
            exec gunicorn app.main:app \
                --worker-class uvicorn.workers.UvicornWorker \
                --workers 4 \
                --bind 0.0.0.0:8000 \
                --log-level info \
                --access-logfile - \
                --error-logfile -
        fi
    else
        # Execute provided command
        exec "$@"
    fi
}

# Run main function with all arguments
main "$@"