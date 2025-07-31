#!/bin/bash

# LeanVibe Agent Hive 2.0 - Service Management Script
# Professional service startup with health monitoring and graceful handling
#
# Usage: ./scripts/start.sh [MODE] [OPTIONS]
# Modes: fast (default), full, minimal, sandbox
#
# Environment Variables:
#   START_MODE     - Override mode selection (fast|full|minimal|sandbox)
#   SKIP_HEALTH    - Skip health checks (true/false)
#   BACKGROUND     - Run services in background (true/false)
#   WAIT_TIMEOUT   - Service startup timeout in seconds (default: 60)

set -euo pipefail

# Color codes for professional output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# Script metadata
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_VERSION="2.0.0"

# Configuration
readonly DEFAULT_MODE="fast"
readonly DEFAULT_TIMEOUT=60
readonly API_PORT=8000
readonly PROMETHEUS_PORT=9090
readonly GRAFANA_PORT=3001

# Mode configurations (using functions for compatibility)
get_mode_description() {
    case "$1" in
        "fast") echo "Start core services with optimized configuration" ;;
        "full") echo "Start all services including monitoring and tools" ;;
        "minimal") echo "Start only essential services for CI/CD" ;;
        "sandbox") echo "Start in sandbox mode for demonstrations" ;;
        *) echo "Unknown mode" ;;
    esac
}

get_mode_services() {
    case "$1" in
        "fast") echo "postgres redis api" ;;
        "full") echo "postgres redis api prometheus grafana" ;;
        "minimal") echo "postgres redis api" ;;
        "sandbox") echo "postgres redis api sandbox" ;;
        *) echo "" ;;
    esac
}

# Global variables
START_MODE="${START_MODE:-${1:-$DEFAULT_MODE}}"
SKIP_HEALTH="${SKIP_HEALTH:-false}"
BACKGROUND="${BACKGROUND:-false}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-$DEFAULT_TIMEOUT}"
START_TIME=""
SERVICE_LOG=""

#======================================
# Utility Functions
#======================================

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case "$level" in
        "INFO")  echo -e "${BLUE}[INFO]${NC}  $message" ;;
        "WARN")  echo -e "${YELLOW}[WARN]${NC}  $message" ;;
        "ERROR") echo -e "${RED}[ERROR]${NC} $message" ;;
        "SUCCESS") echo -e "${GREEN}[SUCCESS]${NC} $message" ;;
        "STEP") echo -e "${PURPLE}[STEP]${NC} $message" ;;
    esac
    
    # Log to file if available
    if [[ -n "$SERVICE_LOG" ]]; then
        echo "[$timestamp] [$level] $message" >> "$SERVICE_LOG"
    fi
}

show_header() {
    clear
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║                          LeanVibe Agent Hive 2.0                            ║
║                          Service Management                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo
    log "INFO" "Starting services in mode: ${START_MODE}"
    log "INFO" "Mode: $(get_mode_description "$START_MODE")"
    echo
    START_TIME=$(date +%s)
}

show_help() {
    cat << EOF
${CYAN}LeanVibe Agent Hive 2.0 - Service Management Script${NC}

${YELLOW}USAGE:${NC}
    $SCRIPT_NAME [MODE] [OPTIONS]

${YELLOW}MODES:${NC}
EOF
    for mode in fast full minimal sandbox; do
        echo "    ${GREEN}$mode${NC} - $(get_mode_description "$mode")"
    done
    cat << EOF

${YELLOW}ENVIRONMENT VARIABLES:${NC}
    START_MODE      Override mode selection
    SKIP_HEALTH     Skip health checks (true/false)
    BACKGROUND      Run services in background (true/false)
    WAIT_TIMEOUT    Service startup timeout in seconds

${YELLOW}EXAMPLES:${NC}
    $SCRIPT_NAME                 # Start in fast mode (default)
    $SCRIPT_NAME full            # Start all services including monitoring
    $SCRIPT_NAME sandbox         # Start in sandbox mode
    BACKGROUND=true $SCRIPT_NAME # Start services in background

${YELLOW}CONTROL:${NC}
    Ctrl+C          Graceful shutdown (when running in foreground)
    
${YELLOW}MORE INFO:${NC}
    Health check: make health
    Stop services: make stop
    Service logs: make logs
EOF
}

setup_logging() {
    local log_dir="$PROJECT_ROOT/logs"
    mkdir -p "$log_dir"
    SERVICE_LOG="$log_dir/start-$(date '+%Y%m%d-%H%M%S').log"
    log "INFO" "Service logging to: $SERVICE_LOG"
}

check_prerequisites() {
    log "STEP" "Checking prerequisites..."
    
    local errors=0
    
    # Check if setup was completed
    if [[ ! -f "$PROJECT_ROOT/.env.local" ]]; then
        log "ERROR" "Environment not configured. Run 'make setup' first."
        errors=$((errors + 1))
    fi
    
    # Check virtual environment
    if [[ ! -d "$PROJECT_ROOT/venv" ]]; then
        log "ERROR" "Python virtual environment not found. Run 'make setup' first."
        errors=$((errors + 1))
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log "ERROR" "Docker is required but not installed"
        errors=$((errors + 1))
    elif ! docker info &> /dev/null; then
        log "ERROR" "Docker daemon is not running"
        errors=$((errors + 1))
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log "ERROR" "Docker Compose is required but not available"
        errors=$((errors + 1))
    fi
    
    if [[ $errors -gt 0 ]]; then
        log "ERROR" "Prerequisites check failed"
        exit 1
    fi
    
    log "SUCCESS" "Prerequisites check passed"
}

get_compose_command() {
    if command -v docker-compose &> /dev/null; then
        echo "docker-compose"
    else
        echo "docker compose"
    fi
}

start_infrastructure_services() {
    log "STEP" "Starting infrastructure services..."
    
    cd "$PROJECT_ROOT"
    local compose_cmd=$(get_compose_command)
    
    # Choose appropriate compose file
    local compose_file="docker-compose.yml"
    case "$START_MODE" in
        "fast")
            if [[ -f "docker-compose.fast.yml" ]]; then
                compose_file="docker-compose.fast.yml"
            fi
            ;;
        "sandbox")
            if [[ -f "docker-compose.sandbox.yml" ]]; then
                compose_file="docker-compose.sandbox.yml"
            fi
            ;;
    esac
    
    log "INFO" "Using compose file: $compose_file"
    
    # Start core infrastructure
    log "INFO" "Starting PostgreSQL and Redis..."
    $compose_cmd -f "$compose_file" up -d postgres redis
    
    # Wait for services to be ready
    log "INFO" "Waiting for infrastructure services to be ready..."
    local retries=$((WAIT_TIMEOUT / 2))
    
    # Wait for PostgreSQL
    while [[ $retries -gt 0 ]]; do
        if $compose_cmd -f "$compose_file" exec -T postgres pg_isready -U leanvibe_user &> /dev/null; then
            log "SUCCESS" "PostgreSQL is ready"
            break
        fi
        retries=$((retries - 1))
        sleep 2
        echo -n "."
    done
    
    if [[ $retries -eq 0 ]]; then
        log "ERROR" "PostgreSQL failed to start within timeout"
        return 1
    fi
    
    # Wait for Redis
    retries=$((WAIT_TIMEOUT / 2))
    while [[ $retries -gt 0 ]]; do
        if $compose_cmd -f "$compose_file" exec -T redis redis-cli ping &> /dev/null; then
            log "SUCCESS" "Redis is ready"
            break
        fi
        retries=$((retries - 1))
        sleep 2
        echo -n "."
    done
    
    if [[ $retries -eq 0 ]]; then
        log "ERROR" "Redis failed to start within timeout"
        return 1
    fi
    
    log "SUCCESS" "Infrastructure services started successfully"
}

start_monitoring_services() {
    if [[ "$START_MODE" != "full" ]]; then
        return 0
    fi
    
    log "STEP" "Starting monitoring services..."
    
    cd "$PROJECT_ROOT"
    local compose_cmd=$(get_compose_command)
    local compose_file="docker-compose.yml"
    
    # Start monitoring services
    if $compose_cmd -f "$compose_file" --profile monitoring up -d prometheus grafana; then
        log "SUCCESS" "Monitoring services started"
        log "INFO" "Prometheus: http://localhost:$PROMETHEUS_PORT"
        log "INFO" "Grafana: http://localhost:$GRAFANA_PORT"
    else
        log "WARN" "Failed to start monitoring services"
    fi
}

start_api_service() {
    log "STEP" "Starting API service..."
    
    cd "$PROJECT_ROOT"
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Check if API key is configured
    if ! grep -q "^ANTHROPIC_API_KEY=" .env.local 2>/dev/null || grep -q "^ANTHROPIC_API_KEY=$" .env.local 2>/dev/null; then
        log "WARN" "ANTHROPIC_API_KEY not configured - some features will be limited"
        log "INFO" "Add your API key: echo 'ANTHROPIC_API_KEY=your_key_here' >> .env.local"
    fi
    
    # Start API service
    if [[ "$BACKGROUND" == "true" ]]; then
        log "INFO" "Starting API service in background..."
        nohup uvicorn app.main:app --host 0.0.0.0 --port $API_PORT > "$SERVICE_LOG.api" 2>&1 &
        local api_pid=$!
        echo $api_pid > "$PROJECT_ROOT/.api.pid"
        log "SUCCESS" "API service started in background (PID: $api_pid)"
    else
        log "INFO" "Starting API service in foreground..."
        log "INFO" "API will be available at: http://localhost:$API_PORT"
        log "INFO" "Press Ctrl+C to stop gracefully"
        echo
        
        # Set up signal handling for graceful shutdown
        trap 'log "INFO" "Shutting down gracefully..."; cleanup_services; exit 0' INT TERM
        
        # Start API service in foreground
        uvicorn app.main:app --host 0.0.0.0 --port $API_PORT
    fi
}

start_sandbox_mode() {
    if [[ "$START_MODE" != "sandbox" ]]; then
        return 0
    fi
    
    log "STEP" "Starting sandbox mode..."
    
    cd "$PROJECT_ROOT"
    source venv/bin/activate
    
    # Start sandbox demo service
    if [[ -f "start-sandbox-demo.sh" ]]; then
        if [[ "$BACKGROUND" == "true" ]]; then
            nohup ./start-sandbox-demo.sh > "$SERVICE_LOG.sandbox" 2>&1 &
            local sandbox_pid=$!
            echo $sandbox_pid > "$PROJECT_ROOT/.sandbox.pid"
            log "SUCCESS" "Sandbox mode started in background (PID: $sandbox_pid)"
        else
            log "INFO" "Starting sandbox mode in foreground..."
            ./start-sandbox-demo.sh
        fi
    else
        log "WARN" "Sandbox demo script not found"
    fi
}

run_health_check() {
    if [[ "$SKIP_HEALTH" == "true" ]]; then
        log "INFO" "Skipping health check (SKIP_HEALTH=true)"
        return 0
    fi
    
    log "STEP" "Running health check..."
    
    # Wait a moment for services to fully initialize
    sleep 5
    
    # Check API health
    local retries=10
    while [[ $retries -gt 0 ]]; do
        if curl -s http://localhost:$API_PORT/health > /dev/null 2>&1; then
            log "SUCCESS" "API health check passed"
            break
        fi
        retries=$((retries - 1))
        sleep 3
        echo -n "."
    done
    
    if [[ $retries -eq 0 ]]; then
        log "WARN" "API health check failed - service may still be starting"
    fi
    
    # Run comprehensive health check if available
    if [[ -f "$PROJECT_ROOT/health-check.sh" ]]; then
        if "$PROJECT_ROOT/health-check.sh" --quiet; then
            log "SUCCESS" "Comprehensive health check passed"
        else
            log "WARN" "Health check reported issues"
        fi
    fi
}

cleanup_services() {
    log "INFO" "Cleaning up services..."
    
    cd "$PROJECT_ROOT"
    local compose_cmd=$(get_compose_command)
    
    # Stop API service if running in background
    if [[ -f ".api.pid" ]]; then
        local api_pid=$(cat .api.pid)
        if kill -0 "$api_pid" 2>/dev/null; then
            log "INFO" "Stopping API service (PID: $api_pid)..."
            kill -TERM "$api_pid" || true
        fi
        rm -f .api.pid
    fi
    
    # Stop sandbox service if running in background
    if [[ -f ".sandbox.pid" ]]; then
        local sandbox_pid=$(cat .sandbox.pid)
        if kill -0 "$sandbox_pid" 2>/dev/null; then
            log "INFO" "Stopping sandbox service (PID: $sandbox_pid)..."
            kill -TERM "$sandbox_pid" || true
        fi
        rm -f .sandbox.pid
    fi
    
    # Optional: Stop Docker services (commented out to preserve data)
    # $compose_cmd down
    
    log "SUCCESS" "Cleanup completed"
}

show_service_status() {
    local end_time=$(date +%s)
    local duration=$((end_time - START_TIME))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))
    
    echo
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║                              SERVICES STARTED                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo
    log "SUCCESS" "Services started in ${minutes}m ${seconds}s"
    echo
    
    echo -e "${YELLOW}SERVICE ENDPOINTS:${NC}"
    echo "   ${CYAN}API:${NC}            http://localhost:$API_PORT"
    echo "   ${CYAN}Health Check:${NC}   http://localhost:$API_PORT/health"
    echo "   ${CYAN}API Docs:${NC}       http://localhost:$API_PORT/docs"
    
    if [[ "$START_MODE" == "full" ]]; then
        echo "   ${CYAN}Prometheus:${NC}     http://localhost:$PROMETHEUS_PORT"
        echo "   ${CYAN}Grafana:${NC}        http://localhost:$GRAFANA_PORT"
    fi
    
    if [[ "$START_MODE" == "sandbox" ]]; then
        echo "   ${CYAN}Sandbox Demo:${NC}   http://localhost:8001"
    fi
    
    echo
    echo -e "${YELLOW}USEFUL COMMANDS:${NC}"
    echo "   make health       # Run health check"
    echo "   make logs         # View service logs"
    echo "   make stop         # Stop all services"
    echo "   make ps           # Show service status"
    echo
    
    if [[ "$BACKGROUND" == "true" ]]; then
        echo -e "${CYAN}Services are running in background.${NC}"
    else
        echo -e "${CYAN}API is running in foreground. Press Ctrl+C to stop.${NC}"
    fi
    
    if [[ -n "$SERVICE_LOG" ]]; then
        echo -e "${YELLOW}Service log:${NC} $SERVICE_LOG"
    fi
    echo
}

handle_error() {
    local exit_code=$?
    log "ERROR" "Service startup failed with exit code $exit_code"
    cleanup_services
    exit $exit_code
}

#======================================
# Main Service Flow
#======================================

main() {
    # Handle help request
    if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
        show_help
        exit 0
    fi
    
    # Validate mode
    if [[ -n "${1:-}" ]] && [[ "$1" != "fast" && "$1" != "full" && "$1" != "minimal" && "$1" != "sandbox" ]]; then
        log "ERROR" "Invalid mode: $1"
        echo "Valid modes: fast full minimal sandbox"
        exit 1
    fi
    
    # Set up error handling
    trap handle_error ERR
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Initialize
    show_header
    setup_logging
    check_prerequisites
    
    # Start services
    start_infrastructure_services
    start_monitoring_services
    run_health_check
    start_sandbox_mode
    
    # Show status
    show_service_status
    
    # Start API service (this may run in foreground)
    start_api_service
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi